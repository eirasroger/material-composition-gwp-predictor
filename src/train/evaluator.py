"""
Held-out evaluation: aggregate test metrics, per-category breakdown,
and the worst-N error printer used for diagnostics.
"""

import math
from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error

from src.config import GHG_MAX, GHG_MIN
from src.utils import r2_safe


def evaluate_model(model, loader, device, scaler_y_mean, scaler_y_scale):
    model.eval()
    preds_s, actuals_s = [], []
    with torch.no_grad():
        for Xb, yb in loader:
            Xb = Xb.to(device)
            preds_s.extend(model(Xb).cpu().numpy())
            actuals_s.extend(yb.numpy())

    preds   = np.expm1(np.asarray(preds_s,  dtype=np.float32) * scaler_y_scale + scaler_y_mean)
    actuals = np.expm1(np.asarray(actuals_s, dtype=np.float32) * scaler_y_scale + scaler_y_mean)

    ghg_range = GHG_MAX - GHG_MIN
    mae        = float(mean_absolute_error(actuals, preds))
    rmse       = float(math.sqrt(np.mean((actuals - preds) ** 2)))
    nrmse      = rmse / ghg_range
    ss_res     = float(np.sum((actuals - preds) ** 2))
    ss_range   = float(len(actuals) * ghg_range ** 2)
    r2_range   = float(1.0 - ss_res / ss_range)
    r2_sample  = r2_safe(actuals, preds)

    abs_err    = np.abs(actuals - preds)
    thresholds = [0.5, 1.0, 2.0, 5.0]
    within     = {f"within_{t}kg": float(np.mean(abs_err <= t) * 100) for t in thresholds}

    return {
        "mae": mae, "rmse": rmse, "nrmse": nrmse,
        "r2_range": r2_range, "r2_sample": r2_sample,
        **within,
        "preds": preds, "actuals": actuals,
    }


def print_category_metrics(
    actuals: np.ndarray,
    preds: np.ndarray,
    categories: List[str],
) -> Dict[str, dict]:
    ghg_range = GHG_MAX - GHG_MIN
    cat_data: Dict[str, list] = defaultdict(list)
    for a, p, c in zip(actuals, preds, categories):
        cat_data[c].append((a, p))

    print(f"\n{chr(9472) * 80}")
    print("  Per-category test metrics")
    print(f"{chr(9472) * 80}")
    print(
        f"  {'Category':<28}  {'N':>5}  {'MAE':>8}  {'RMSE':>8}  "
        f"{'NRMSE':>7}  {'+-0.5':>6}  {'+-1.0':>6}  {'+-2.0':>6}"
    )
    print(
        f"  {chr(9472) * 28}  {chr(9472) * 5}  {chr(9472) * 8}  {chr(9472) * 8}  "
        f"{chr(9472) * 7}  {chr(9472) * 6}  {chr(9472) * 6}  {chr(9472) * 6}"
    )

    per_cat = {}
    for cat in sorted(cat_data.keys()):
        pairs = cat_data[cat]
        n     = len(pairs)
        a_arr = np.array([x[0] for x in pairs])
        p_arr = np.array([x[1] for x in pairs])
        mae   = float(mean_absolute_error(a_arr, p_arr))
        rmse  = float(math.sqrt(np.mean((a_arr - p_arr) ** 2)))
        nrmse = rmse / ghg_range
        ae    = np.abs(a_arr - p_arr)
        w05   = float(np.mean(ae <= 0.5) * 100)
        w10   = float(np.mean(ae <= 1.0) * 100)
        w20   = float(np.mean(ae <= 2.0) * 100)

        print(
            f"  {cat:<28}  {n:>5}  {mae:>8.4f}  {rmse:>8.4f}  "
            f"{nrmse:>7.4f}  {w05:>5.1f}%  {w10:>5.1f}%  {w20:>5.1f}%"
        )
        per_cat[cat] = {
            "n": n, "mae": mae, "rmse": rmse, "nrmse": nrmse,
            "within_0.5kg": w05, "within_1.0kg": w10, "within_2.0kg": w20,
        }

    print(f"{chr(9472) * 80}")
    return per_cat


def print_worst_predictions(
    actuals: np.ndarray,
    preds: np.ndarray,
    categories: List[str],
    n: int = 10,
) -> None:
    abs_err = np.abs(actuals - preds)
    worst   = np.argsort(abs_err)[::-1][:n]

    print(f"\n-- Worst {n} predictions (test set) --")
    print(f"  {'Actual':>10}  {'Predicted':>10}  {'Abs Err':>10}  Category")
    for i in worst:
        a, p, c = actuals[i], preds[i], categories[i]
        print(f"  {a:>10.4f}  {p:>10.4f}  {abs(a - p):>10.4f}  {c}")
