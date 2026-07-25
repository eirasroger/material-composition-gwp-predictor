"""
Held-out evaluation: aggregate test metrics, per-category breakdown,
and the worst-N error printer used for diagnostics.

All domain-specific knowledge (unit, thresholds, value range) is passed
in from the caller rather than read from config.
"""

import math
from collections import defaultdict
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error

from src.utils import r2_safe


def evaluate_model(
    model,
    loader,
    device,
    scaler_y_mean: float,
    scaler_y_scale: float,
    inverse_fn: Optional[Callable] = None,
    value_min: float = 0.0,
    value_max: float = 10.0,
    thresholds: Optional[List[float]] = None,
):
    """
    inverse_fn  : maps scaled model output → original units (default: np.expm1)
    value_min/max: used to compute NRMSE relative to the target range
    thresholds  : list of absolute error thresholds for within-X diagnostics
    """
    if inverse_fn is None:
        inverse_fn = np.expm1
    if thresholds is None:
        thresholds = [0.5, 1.0, 2.0, 5.0]

    model.eval()
    preds_s, actuals_s = [], []
    with torch.no_grad():
        for Xb, yb in loader:
            Xb = Xb.to(device)
            preds_s.extend(model(Xb).cpu().numpy())
            actuals_s.extend(yb.numpy())

    preds   = inverse_fn(np.asarray(preds_s,   dtype=np.float32) * scaler_y_scale + scaler_y_mean)
    actuals = inverse_fn(np.asarray(actuals_s,  dtype=np.float32) * scaler_y_scale + scaler_y_mean)

    target_range = value_max - value_min
    mae          = float(mean_absolute_error(actuals, preds))
    rmse         = float(math.sqrt(np.mean((actuals - preds) ** 2)))
    nrmse        = rmse / target_range if target_range > 0 else float("nan")
    ss_res       = float(np.sum((actuals - preds) ** 2))
    ss_range     = float(len(actuals) * target_range ** 2)
    r2_range     = float(1.0 - ss_res / ss_range) if ss_range > 0 else float("nan")
    r2_sample    = r2_safe(actuals, preds)

    abs_err = np.abs(actuals - preds)
    within  = {f"within_{t}": float(np.mean(abs_err <= t) * 100) for t in thresholds}

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
    value_min: float = 0.0,
    value_max: float = 10.0,
    thresholds: Optional[List[float]] = None,
) -> Dict[str, dict]:
    if thresholds is None:
        thresholds = [0.5, 1.0, 2.0]

    target_range = value_max - value_min
    cat_data: Dict[str, list] = defaultdict(list)
    for a, p, c in zip(actuals, preds, categories):
        cat_data[c].append((a, p))

    th_headers = "  ".join(f"{'±'+str(t):>6}" for t in thresholds)
    col_w = 8 + 9 * len(thresholds)
    print(f"\n{chr(9472) * (72 + col_w)}")
    print("  Per-category test metrics")
    print(f"{chr(9472) * (72 + col_w)}")
    print(
        f"  {'Category':<28}  {'N':>5}  {'MAE':>8}  {'RMSE':>8}  "
        f"{'NRMSE':>7}  {th_headers}"
    )
    sep = f"  {chr(9472)*28}  {chr(9472)*5}  {chr(9472)*8}  {chr(9472)*8}  {chr(9472)*7}  " + \
          "  ".join(chr(9472)*6 for _ in thresholds)
    print(sep)

    per_cat = {}
    for cat in sorted(cat_data.keys()):
        pairs = cat_data[cat]
        n     = len(pairs)
        a_arr = np.array([x[0] for x in pairs])
        p_arr = np.array([x[1] for x in pairs])
        mae   = float(mean_absolute_error(a_arr, p_arr))
        rmse  = float(math.sqrt(np.mean((a_arr - p_arr) ** 2)))
        nrmse = rmse / target_range if target_range > 0 else float("nan")
        ae    = np.abs(a_arr - p_arr)
        within_vals = [float(np.mean(ae <= t) * 100) for t in thresholds]

        within_str = "  ".join(f"{w:>5.1f}%" for w in within_vals)
        print(
            f"  {cat:<28}  {n:>5}  {mae:>8.4f}  {rmse:>8.4f}  "
            f"{nrmse:>7.4f}  {within_str}"
        )
        per_cat[cat] = {
            "n": n, "mae": mae, "rmse": rmse, "nrmse": nrmse,
            **{f"within_{t}": w for t, w in zip(thresholds, within_vals)},
        }

    print(f"{chr(9472) * (72 + col_w)}")
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
