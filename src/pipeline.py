"""
End-to-end pipeline orchestration: load -> filter -> embed -> split -> train ->
evaluate -> save artefacts -> demo inference.
"""

from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from src.config import (
    BASE_DIR,
    BATCH_SIZE,
    DATASET_PATH,
    DIAGNOSTICS_PATH,
    DROPOUT,
    EMBED_DIM,
    GHG_MAX,
    GHG_MIN,
    HIDDEN_DIMS,
    MIN_CATEGORY_COUNT,
    MODEL_PATH,
    N_CIRC_FEATURES,
    PRED_SCATTER_PATH,
    RANDOM_SEED,
    RESIDUALS_PATH,
    TRAINING_PLOT_PATH,
)
from src.data.features import build_features
from src.data.loader import (
    build_category_index,
    filter_reference_unit_kg,
    load_dataset,
)
from src.data.preprocessing import filter_valid_products
from src.embeddings.vocab import get_vocab
from src.inference.predict import predict_ghg
from src.model.dataset import GHGDataset
from src.model.network import GHGNet
from src.reporting.plots import save_diagnostics, save_plots
from src.train.evaluator import (
    evaluate_model,
    print_category_metrics,
    print_worst_predictions,
)
from src.train.trainer import train_model


def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Script directory: {BASE_DIR}")
    print(f"Current working directory: {Path.cwd()}")

    print(f"\nLoading '{DATASET_PATH.name}' ...")
    products = load_dataset(DATASET_PATH)
    print(f"Total products: {len(products)}")

    products  = filter_reference_unit_kg(products)
    cat_index = build_category_index(products, min_count=MIN_CATEGORY_COUNT)
    print(f"Active categories ({len(cat_index)}): {sorted(cat_index.keys())}")

    valid_products = filter_valid_products(products, cat_index)

    if len(valid_products) < 10:
        raise ValueError("Need at least 10 valid labelled samples after filtering.")

    ghg_values = np.array([p["ghg"] for p in valid_products], dtype=np.float32)
    print(
        f"\nGHG summary after filtering -> "
        f"min: {ghg_values.min():.4f}, "
        f"median: {np.median(ghg_values):.4f}, "
        f"max: {ghg_values.max():.4f}, "
        f"mean: {ghg_values.mean():.4f}"
    )

    cat_counts_valid = Counter(p["category"] for p in valid_products)
    print("\nCategory distribution in valid set:")
    for cat, cnt in sorted(cat_counts_valid.items(), key=lambda x: -x[1]):
        print(f"  {cat:<32}  {cnt:>5} products")

    vocab = get_vocab(valid_products)
    print(f"\nVocabulary loaded: {len(vocab):,}")

    n_categories = len(cat_index)
    input_dim    = EMBED_DIM + n_categories + N_CIRC_FEATURES
    print(
        f"Input dimension: {EMBED_DIM} (material embedding) "
        f"+ {n_categories} (category one-hot) "
        f"+ {N_CIRC_FEATURES} (circularity features) = {input_dim}"
    )

    X, y, categories_all = build_features(valid_products, vocab, cat_index)
    print(f"Feature matrix shape: {X.shape}")
    print(f"Products used for training: {len(y)}")

    y_log    = np.log1p(y)
    scaler_y = StandardScaler()
    sy       = scaler_y.fit_transform(y_log.reshape(-1, 1)).ravel().astype(np.float32)

    if not np.isfinite(sy).all():
        raise ValueError("Target scaling produced non-finite values. Check the dataset.")

    idx = np.arange(len(y))
    idx_tr, idx_tmp = train_test_split(
        idx, test_size=0.30, random_state=RANDOM_SEED,
        stratify=[categories_all[i] for i in idx],
    )
    idx_val, idx_te = train_test_split(
        idx_tmp, test_size=0.50, random_state=RANDOM_SEED,
        stratify=[categories_all[i] for i in idx_tmp],
    )

    print(f"\nStratified split  ->  train: {len(idx_tr)}  val: {len(idx_val)}  test: {len(idx_te)}")

    for split_name, split_idx in [("train", idx_tr), ("val", idx_val), ("test", idx_te)]:
        cats_in_split = set(categories_all[i] for i in split_idx)
        missing       = set(cat_index.keys()) - cats_in_split
        status        = "all categories present" if not missing else f"MISSING: {missing}"
        print(f"  {split_name:<6}: {status}")

    def make_loader(indices, shuffle=False):
        ds = GHGDataset(X[indices], sy[indices])
        return DataLoader(
            ds,
            batch_size=BATCH_SIZE,
            shuffle=shuffle,
            drop_last=(shuffle and len(indices) > BATCH_SIZE),
        )

    train_loader = make_loader(idx_tr, shuffle=True)
    val_loader   = make_loader(idx_val)
    test_loader  = make_loader(idx_te)

    model = GHGNet(input_dim=input_dim).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    scaler_y_mean  = float(scaler_y.mean_[0])
    scaler_y_scale = float(scaler_y.scale_[0])

    model, history = train_model(
        model, train_loader, val_loader, device, scaler_y_mean, scaler_y_scale
    )

    test_results = evaluate_model(model, test_loader, device, scaler_y_mean, scaler_y_scale)

    print("\n" + "=" * 50)
    print("  TEST SET RESULTS")
    print("=" * 50)
    print(f"  MAE           {test_results['mae']:.4f} kg CO2-eq")
    print(f"  RMSE          {test_results['rmse']:.4f} kg CO2-eq")
    print(f"  NRMSE         {test_results['nrmse']:.4f}  (fraction of [{GHG_MIN}, {GHG_MAX}] range)")
    print(f"  R2 (range)    {test_results['r2_range']:.6f}  (vs fixed scale)")
    print(f"  R2 (sample)   {test_results['r2_sample']:.4f}  (vs sample variance, for reference)")
    print(f"  Within +-0.5  {test_results['within_0.5kg']:.1f}%")
    print(f"  Within +-1.0  {test_results['within_1.0kg']:.1f}%")
    print(f"  Within +-2.0  {test_results['within_2.0kg']:.1f}%")
    print(f"  Within +-5.0  {test_results['within_5.0kg']:.1f}%")
    print("=" * 50)

    print("\n-- Sample predictions (first 5 of test set) --")
    print(f"  {'Actual':>10}  {'Predicted':>10}  {'Abs Err':>10}  Category")
    test_categories = [categories_all[i] for i in idx_te]
    for a, p, c in zip(
        test_results["actuals"][:5], test_results["preds"][:5], test_categories[:5]
    ):
        print(f"  {a:>10.4f}  {p:>10.4f}  {abs(a - p):>10.4f}  {c}")

    per_cat_metrics = print_category_metrics(
        test_results["actuals"], test_results["preds"], test_categories
    )

    print_worst_predictions(
        test_results["actuals"], test_results["preds"], test_categories, n=10
    )

    save_plots(history, test_results["actuals"], test_results["preds"])
    print(f"\nSaved plot -> {TRAINING_PLOT_PATH}")
    print(f"Saved plot -> {PRED_SCATTER_PATH}")
    print(f"Saved plot -> {RESIDUALS_PATH}")

    torch.save(
        {
            "model_state": model.state_dict(),
            "y_mean":      scaler_y_mean,
            "y_scale":     scaler_y_scale,
            "hidden_dims": HIDDEN_DIMS,
            "dropout":     DROPOUT,
            "cat_index":   cat_index,
            "input_dim":   input_dim,
        },
        str(MODEL_PATH),
    )
    print(f"Model saved -> {MODEL_PATH}")

    diagnostics = {
        "total_products":         len(products),
        "valid_products":         len(valid_products),
        "train_size":             len(idx_tr),
        "val_size":               len(idx_val),
        "test_size":              len(idx_te),
        "vocab_size":             len(vocab),
        "n_categories":           n_categories,
        "n_circularity_features": N_CIRC_FEATURES,
        "input_dim":              input_dim,
        "categories":             sorted(cat_index.keys()),
        "ghg_min":                GHG_MIN,
        "ghg_max":                GHG_MAX,
        "min_category_count":     MIN_CATEGORY_COUNT,
        "test_metrics": {
            "mae":          test_results["mae"],
            "rmse":         test_results["rmse"],
            "nrmse":        test_results["nrmse"],
            "r2_range":     test_results["r2_range"],
            "r2_sample":    test_results["r2_sample"],
            **{k: v for k, v in test_results.items() if k.startswith("within_")},
            "per_category": per_cat_metrics,
        },
        "history": history,
    }
    save_diagnostics(diagnostics)
    print(f"Diagnostics saved -> {DIAGNOSTICS_PATH}")

    example_raw  = valid_products[idx_te[0]]["raw"]
    example_pred = predict_ghg(example_raw, vocab, checkpoint=MODEL_PATH)
    example_true = valid_products[idx_te[0]]["ghg"]
    print("\n-- Inference demo --")
    print(f"  True GHG : {example_true:.4f} kg CO2-eq")
    print(f"  Pred GHG : {example_pred:.4f} kg CO2-eq")
