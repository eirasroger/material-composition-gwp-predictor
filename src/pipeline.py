"""
End-to-end pipeline orchestration: load -> filter -> embed -> split -> train ->
evaluate -> save artefacts -> demo inference.

The pipeline is domain-agnostic.  All target-specific knowledge (field path,
value range, transform, thresholds, unit) is supplied via TARGET_CONFIGS in
config.py.  To train a new impact category, add an entry there — no code
changes here.
"""

from collections import Counter, defaultdict
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
    DROPOUT,
    EMBED_DIM,
    HIDDEN_DIMS,
    MIN_CATEGORY_COUNT,
    MODELS_DIR,
    N_CIRC_FEATURES,
    SEEDS,
    TARGET_CONFIGS,
)
from src.data.features import build_features
from src.data.loader import (
    build_category_index,
    filter_reference_unit_kg,
    load_dataset,
)
from src.data.preprocessing import filter_valid_products
from src.embeddings.vocab import get_vocab
from src.model.dataset import GHGDataset
from src.model.network import GHGNet
from src.reporting.plots import save_diagnostics, save_plots
from src.train.evaluator import (
    evaluate_model,
    print_category_metrics,
    print_worst_predictions,
)
from src.train.trainer import train_model
from src.utils import make_transforms


def run(target_key: str = "ghg_total"):
    if target_key not in TARGET_CONFIGS:
        raise ValueError(
            f"Unknown target '{target_key}'. "
            f"Available: {sorted(TARGET_CONFIGS.keys())}"
        )
    cfg = TARGET_CONFIGS[target_key]

    field_path   = cfg["field_path"]
    value_min    = cfg["value_min"]
    value_max    = cfg["value_max"]
    transform    = cfg["transform"]
    scale        = cfg["scale"]
    thresholds   = cfg["thresholds"]
    unit         = cfg["unit"]
    display_name = cfg["display_name"]

    forward_fn, inverse_fn = make_transforms(transform, scale)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path       = MODELS_DIR / f"{target_key}.pt"
    figures_dir      = BASE_DIR / "figures" / target_key
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_paths       = {
        "training":  figures_dir / "training_curves.png",
        "scatter":   figures_dir / "pred_vs_actual.png",
        "residuals": figures_dir / "residuals.png",
    }
    diagnostics_dir  = BASE_DIR / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_path = diagnostics_dir / f"{target_key}.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Target: {display_name}  ({target_key})")
    print(
        f"Field:  {field_path}  |  range [{value_min}, {value_max}]  "
        f"|  transform: {transform}(x/{scale:g})"
    )
    print(f"Script directory: {BASE_DIR}")

    print(f"\nLoading '{DATASET_PATH.name}' ...")
    products = load_dataset(DATASET_PATH)
    print(f"Total products: {len(products)}")

    products  = filter_reference_unit_kg(products)
    cat_index = build_category_index(products, min_count=MIN_CATEGORY_COUNT)
    print(f"Active categories ({len(cat_index)}): {sorted(cat_index.keys())}")

    std_cat_index = build_category_index(
        products, min_count=MIN_CATEGORY_COUNT, field="category_standardized"
    )
    print(f"Active standardized categories ({len(std_cat_index)}): {sorted(std_cat_index.keys())}")

    # Coalesced category vocabulary: real PCR labels (N/A excluded -- it is
    # not a category, it is the absence of one) unioned with standardized
    # labels. Each product's feature fires exactly one slot: its PCR if
    # reported, else its standardized category (see normalize_product).
    unified_cat_index = {
        label: i for i, label in enumerate(
            sorted((set(cat_index) - {"N/A", ""}) | set(std_cat_index))
        )
    }
    print(f"Unified (coalesced) categories ({len(unified_cat_index)}): {sorted(unified_cat_index.keys())}")

    valid_products = filter_valid_products(
        products, cat_index,
        value_min=value_min,
        value_max=value_max,
        target_field_path=field_path,
    )

    if len(valid_products) < 10:
        raise ValueError("Need at least 10 valid labelled samples after filtering.")

    # build_category_index applies MIN_CATEGORY_COUNT to the *whole* dataset, but
    # each target drops a different set of products (nulls + value bounds), so a
    # category can fall below it here.  Both splits below stratify on category:
    # a category left with 4 members puts only 1 into the val/test pool, and
    # sklearn refuses to stratify a class of 1.  Re-apply the threshold to the
    # filtered population so the split is always well defined.
    cat_counts_filtered = Counter(p["category"] for p in valid_products)
    thin_categories = {
        cat for cat, cnt in cat_counts_filtered.items() if cnt < MIN_CATEGORY_COUNT
    }
    if thin_categories:
        before = len(valid_products)
        valid_products = [
            p for p in valid_products if p["category"] not in thin_categories
        ]
        print(
            f"\nDropped {before - len(valid_products)} products in "
            f"{len(thin_categories)} category/ies with < {MIN_CATEGORY_COUNT} "
            f"labelled samples for this target (stratified split needs them):"
        )
        for cat in sorted(thin_categories):
            print(f"  {cat:<48} {cat_counts_filtered[cat]:>4}")

    target_values = np.array([p["target"] for p in valid_products], dtype=np.float32)
    print(
        f"\nTarget summary after filtering -> "
        f"min: {target_values.min():.4g}, "
        f"median: {np.median(target_values):.4g}, "
        f"max: {target_values.max():.4g}, "
        f"mean: {target_values.mean():.4g}"
    )

    # The transform is only a log transform when `scale` matches the target's
    # own magnitude; skew of the forward-transformed target is the check.
    pos_median = float(np.median(target_values[target_values > 0])) \
        if (target_values > 0).any() else float("nan")
    print(
        f"Transform scale: {scale:g}  (median positive value: {pos_median:.4g}) "
        f"-- these should be within ~an order of magnitude"
    )

    cat_counts_valid = Counter(p["category"] for p in valid_products)
    print("\nCategory distribution in valid set:")
    for cat, cnt in sorted(cat_counts_valid.items(), key=lambda x: -x[1]):
        print(f"  {cat:<32}  {cnt:>5} products")

    vocab = get_vocab(valid_products)
    print(f"\nVocabulary loaded: {len(vocab):,}")

    n_categories     = len(cat_index)
    n_std_categories = len(std_cat_index)
    n_unified        = len(unified_cat_index)
    input_dim        = EMBED_DIM + n_unified + 1 + N_CIRC_FEATURES
    print(
        f"Input dimension: {EMBED_DIM} (material embedding) "
        f"+ {n_unified} (coalesced category one-hot) "
        f"+ 1 (has_pcr flag) "
        f"+ {N_CIRC_FEATURES} (circularity features) = {input_dim}"
    )

    X, y, categories_all = build_features(valid_products, vocab, unified_cat_index)
    categories_resolved_all = [p["category_resolved"] for p in valid_products]
    print(f"Feature matrix shape: {X.shape}")
    print(f"Products used for training: {len(y)}")

    y_transformed = forward_fn(y.astype(np.float64)).astype(np.float64)
    if not np.isfinite(y_transformed).all():
        raise ValueError(
            "Forward transform produced non-finite values. "
            "Check value_min/value_max or the transform type."
        )

    scaler_y = StandardScaler()
    sy       = scaler_y.fit_transform(y_transformed.reshape(-1, 1)).ravel().astype(np.float32)

    if not np.isfinite(sy).all():
        raise ValueError("Target scaling produced non-finite values. Check the dataset.")

    scaler_y_mean  = float(scaler_y.mean_[0])
    scaler_y_scale = float(scaler_y.scale_[0])

    dummy_model = GHGNet(input_dim=input_dim).to(device)
    print(f"Parameters: {sum(p.numel() for p in dummy_model.parameters() if p.requires_grad):,}")
    del dummy_model

    def make_loader(indices, shuffle=False):
        ds = GHGDataset(X[indices], sy[indices])
        return DataLoader(
            ds,
            batch_size=BATCH_SIZE,
            shuffle=shuffle,
            drop_last=(shuffle and len(indices) > BATCH_SIZE),
        )

    # ── Multi-seed training ───────────────────────────────────────────────────
    seed_results          = []
    best_val_mae          = float("inf")
    best_checkpoint       = None
    best_history          = None
    best_seed             = None
    best_test_res         = None
    best_test_cats        = None
    best_test_cats_resolved = None
    best_example_idx      = None
    all_cat_signed_errors: dict = defaultdict(list)
    all_resolved_cat_signed_errors: dict = defaultdict(list)

    for seed in SEEDS:
        print(f"\n{'=' * 58}")
        print(f"  SEED {seed}  ({SEEDS.index(seed) + 1}/{len(SEEDS)})")
        print(f"{'=' * 58}")

        torch.manual_seed(seed)
        np.random.seed(seed)

        idx = np.arange(len(y))
        idx_tr, idx_tmp = train_test_split(
            idx, test_size=0.30, random_state=seed,
            stratify=[categories_all[i] for i in idx],
        )
        idx_val, idx_te = train_test_split(
            idx_tmp, test_size=0.50, random_state=seed,
            stratify=[categories_all[i] for i in idx_tmp],
        )
        print(f"Split -> train: {len(idx_tr)}  val: {len(idx_val)}  test: {len(idx_te)}")

        train_loader = make_loader(idx_tr, shuffle=True)
        val_loader   = make_loader(idx_val)
        test_loader  = make_loader(idx_te)

        model = GHGNet(input_dim=input_dim).to(device)
        model, history = train_model(
            model, train_loader, val_loader, device,
            scaler_y_mean, scaler_y_scale, inverse_fn=inverse_fn,
        )

        val_res  = evaluate_model(
            model, val_loader,  device, scaler_y_mean, scaler_y_scale,
            inverse_fn=inverse_fn, forward_fn=forward_fn, thresholds=thresholds,
        )
        test_res = evaluate_model(
            model, test_loader, device, scaler_y_mean, scaler_y_scale,
            inverse_fn=inverse_fn, forward_fn=forward_fn, thresholds=thresholds,
        )

        test_cats = [categories_all[i] for i in idx_te]
        for pred, actual, cat in zip(test_res["preds"], test_res["actuals"], test_cats):
            all_cat_signed_errors[cat].append(float(pred) - float(actual))

        test_cats_resolved = [categories_resolved_all[i] for i in idx_te]
        for pred, actual, cat in zip(test_res["preds"], test_res["actuals"], test_cats_resolved):
            all_resolved_cat_signed_errors[cat].append(float(pred) - float(actual))

        print(
            f"\nSeed {seed}: val MAE={val_res['mae']:.4g}  "
            f"test MAE={test_res['mae']:.4g}  test MedAE={test_res['medae']:.4g}  "
            f"test Bias={test_res['bias']:+.4g}  "
            f"test Fold={test_res['fold_median']:.2f}x"
        )

        seed_results.append({
            "seed":        seed,
            "val_mae":     val_res["mae"],
            "test_mae":    test_res["mae"],
            "test_medae":  test_res["medae"],
            "test_bias":   test_res["bias"],
            "test_rmse":   test_res["rmse"],
            "test_fold_median": test_res["fold_median"],
            "test_fold_p90":    test_res["fold_p90"],
            **{k: v for k, v in test_res.items() if k.startswith("within_")},
        })

        if val_res["mae"] < best_val_mae:
            best_val_mae     = val_res["mae"]
            best_checkpoint  = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_history     = history
            best_seed        = seed
            best_test_res    = test_res
            best_test_cats   = test_cats
            best_test_cats_resolved = test_cats_resolved
            best_example_idx = idx_te[0]

    # ── Per-category error bounds (aggregated across all seeds) ──────────────
    category_error_bounds = {}
    for cat, errors in all_cat_signed_errors.items():
        arr = np.array(errors, dtype=np.float32)
        category_error_bounds[cat] = {
            "mean": float(np.mean(arr)),
            "p25":  float(np.percentile(arr, 25)),
            "p75":  float(np.percentile(arr, 75)),
            "n":    len(arr),
        }

    resolved_category_error_bounds = {}
    for cat, errors in all_resolved_cat_signed_errors.items():
        arr = np.array(errors, dtype=np.float32)
        resolved_category_error_bounds[cat] = {
            "mean": float(np.mean(arr)),
            "p25":  float(np.percentile(arr, 25)),
            "p75":  float(np.percentile(arr, 75)),
            "n":    len(arr),
        }

    # ── Multi-seed summary ────────────────────────────────────────────────────
    test_maes   = [r["test_mae"]   for r in seed_results]
    test_medaes = [r["test_medae"] for r in seed_results]
    test_biases = [r["test_bias"]  for r in seed_results]
    test_folds  = [r["test_fold_median"] for r in seed_results]

    print(f"\n{'=' * 64}")
    print(f"  MULTI-SEED SUMMARY  ({len(SEEDS)} seeds)")
    print(f"{'=' * 64}")
    for r in seed_results:
        marker = "  <-- saved" if r["seed"] == best_seed else ""
        print(
            f"  seed {r['seed']}: val MAE={r['val_mae']:.4g}  "
            f"test MAE={r['test_mae']:.4g}  MedAE={r['test_medae']:.4g}  "
            f"Bias={r['test_bias']:+.4g}  Fold={r['test_fold_median']:.2f}x{marker}"
        )
    print(f"  {'─' * 60}")
    print(f"  Test MAE   : {np.mean(test_maes):.4g} ± {np.std(test_maes):.4g}")
    print(f"  Test MedAE : {np.mean(test_medaes):.4g} ± {np.std(test_medaes):.4g}")
    print(f"  Test Bias  : {np.mean(test_biases):+.4g} ± {np.std(test_biases):.4g}")
    print(f"  Test Fold  : {np.mean(test_folds):.3f}x ± {np.std(test_folds):.3f}")
    print(f"{'=' * 64}")

    # ── Full reporting for best seed ──────────────────────────────────────────
    model = GHGNet(input_dim=input_dim).to(device)
    model.load_state_dict(best_checkpoint)

    print(f"\n{'=' * 50}")
    print(f"  BEST MODEL (seed {best_seed}) — TEST SET RESULTS")
    print("=" * 50)
    print(f"  Target        {display_name}  [{unit}]")
    print(f"  MAE           {best_test_res['mae']:.4g}")
    print(f"  MedAE         {best_test_res['medae']:.4g}")
    print(f"  Bias          {best_test_res['bias']:+.4g}  (mean predicted - actual; +over / -under)")
    print(f"  RMSE          {best_test_res['rmse']:.4g}")
    print(f"  Fold (median) {best_test_res['fold_median']:.2f}x  (scale-free: predicted within this factor)")
    print(f"  Fold (p90)    {best_test_res['fold_p90']:.2f}x  (90% of predictions within this factor)")
    for t in thresholds:
        key = f"within_{t}"
        print(f"  Within ±{t:<6} {best_test_res[key]:.1f}%")
    print("=" * 50)

    print("\n-- Sample predictions (first 5 of best-seed test set) --")
    print(f"  {'Actual':>12}  {'Predicted':>12}  {'Abs Err':>12}  Category")
    for a, p, c in zip(
        best_test_res["actuals"][:5], best_test_res["preds"][:5], best_test_cats[:5]
    ):
        print(f"  {a:>12.4g}  {p:>12.4g}  {abs(a - p):>12.4g}  {c}")

    per_cat_metrics = print_category_metrics(
        best_test_res["actuals"], best_test_res["preds"], best_test_cats,
        forward_fn=forward_fn, thresholds=thresholds,
    )
    print("\n(above grouped by raw c_pcr -- 'N/A' row is products with no reported PCR)")

    per_cat_resolved_metrics = print_category_metrics(
        best_test_res["actuals"], best_test_res["preds"], best_test_cats_resolved,
        forward_fn=forward_fn, thresholds=thresholds,
    )
    print("\n(above grouped by coalesced category -- PCR if reported, else standardized category)")

    print_worst_predictions(
        best_test_res["actuals"], best_test_res["preds"], best_test_cats, n=10
    )

    save_plots(
        best_history, best_test_res["actuals"], best_test_res["preds"],
        paths=plot_paths, display_name=display_name, unit=unit,
    )
    for key, path in plot_paths.items():
        print(f"Saved plot ({key}) -> {path}")

    torch.save(
        {
            "model_state":           model.state_dict(),
            "y_mean":                scaler_y_mean,
            "y_scale":               scaler_y_scale,
            "hidden_dims":           HIDDEN_DIMS,
            "dropout":               DROPOUT,
            "cat_index":             cat_index,
            "std_cat_index":         std_cat_index,
            "unified_cat_index":     unified_cat_index,
            "input_dim":             input_dim,
            "transform_type":        transform,
            "transform_scale":       scale,
            "target_key":            target_key,
            "field_path":            field_path,
            "value_min":             value_min,
            "value_max":             value_max,
            "unit":                  unit,
            "display_name":          display_name,
            "category_error_bounds": category_error_bounds,
            "resolved_category_error_bounds": resolved_category_error_bounds,
        },
        str(model_path),
    )
    print(f"Model saved -> {model_path}  (best seed: {best_seed})")

    diagnostics = {
        "target_key":             target_key,
        "display_name":           display_name,
        "unit":                   unit,
        "field_path":             field_path,
        "total_products":         len(products),
        "valid_products":         len(valid_products),
        "vocab_size":             len(vocab),
        "n_categories":           n_categories,
        "n_std_categories":       n_std_categories,
        "n_unified_categories":   n_unified,
        "n_circularity_features": N_CIRC_FEATURES,
        "input_dim":              input_dim,
        "categories":             sorted(cat_index.keys()),
        "std_categories":         sorted(std_cat_index.keys()),
        "unified_categories":     sorted(unified_cat_index.keys()),
        "value_min":              value_min,
        "value_max":              value_max,
        "transform":              transform,
        "transform_scale":        scale,
        "thresholds":             thresholds,
        "min_category_count":     MIN_CATEGORY_COUNT,
        "seeds":                  SEEDS,
        "best_seed":              best_seed,
        "seed_results":           seed_results,
        "test_metrics": {
            "mae":          best_test_res["mae"],
            "medae":        best_test_res["medae"],
            "bias":         best_test_res["bias"],
            "rmse":         best_test_res["rmse"],
            "fold_median":  best_test_res["fold_median"],
            "fold_p90":     best_test_res["fold_p90"],
            **{k: v for k, v in best_test_res.items() if k.startswith("within_")},
            "per_category": per_cat_metrics,
            "per_category_resolved": per_cat_resolved_metrics,
        },
        "multi_seed_summary": {
            "test_mae_mean":   float(np.mean(test_maes)),
            "test_mae_std":    float(np.std(test_maes)),
            "test_medae_mean": float(np.mean(test_medaes)),
            "test_medae_std":  float(np.std(test_medaes)),
            "test_bias_mean":  float(np.mean(test_biases)),
            "test_bias_std":   float(np.std(test_biases)),
            "test_fold_median_mean": float(np.mean(test_folds)),
            "test_fold_median_std":  float(np.std(test_folds)),
        },
        "history": best_history,
    }
    save_diagnostics(diagnostics, diagnostics_path)
    print(f"Diagnostics saved -> {diagnostics_path}")
