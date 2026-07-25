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


def fold_errors(
    actuals: np.ndarray,
    preds: np.ndarray,
    forward_fn: Callable,
) -> np.ndarray:
    """
    Scale-free error: the multiplicative factor between prediction and actual.

    Computed in the target's own transformed space, so it is well defined at
    zero (the transform's `scale` acts as the offset) and identical in meaning
    across indicators whose native magnitudes differ by six orders of
    magnitude.  A value of 1.5 means "predicted within a factor of 1.5".
    """
    return np.exp(np.abs(
        np.asarray(forward_fn(preds), dtype=np.float64)
        - np.asarray(forward_fn(actuals), dtype=np.float64)
    ))


def evaluate_model(
    model,
    loader,
    device,
    scaler_y_mean: float,
    scaler_y_scale: float,
    inverse_fn: Optional[Callable] = None,
    forward_fn: Optional[Callable] = None,
    thresholds: Optional[List[float]] = None,
):
    """
    inverse_fn  : maps scaled model output → original units (default: np.expm1)
    forward_fn  : inverse of inverse_fn; used for the scale-free fold-error
                  metrics.  Omitted → fold metrics are not reported.
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

    signed_err   = preds - actuals
    abs_err      = np.abs(signed_err)

    mae          = float(np.mean(abs_err))
    medae        = float(np.median(abs_err))
    bias         = float(np.mean(signed_err))
    rmse         = float(math.sqrt(np.mean(signed_err ** 2)))

    within = {f"within_{t}": float(np.mean(abs_err <= t) * 100) for t in thresholds}

    fold = {}
    if forward_fn is not None:
        fe = fold_errors(actuals, preds, forward_fn)
        fold = {
            "fold_median": float(np.median(fe)),
            "fold_p90":    float(np.percentile(fe, 90)),
        }

    return {
        "mae": mae, "medae": medae, "bias": bias, "rmse": rmse,
        **fold,
        **within,
        "preds": preds, "actuals": actuals,
    }


def print_category_metrics(
    actuals: np.ndarray,
    preds: np.ndarray,
    categories: List[str],
    forward_fn: Optional[Callable] = None,
    thresholds: Optional[List[float]] = None,
) -> Dict[str, dict]:
    """
    Per-category breakdown.  All value columns use %g so they stay readable for
    indicators whose native values are ~1e-6 (%f would print them all as 0).
    """
    if thresholds is None:
        thresholds = [0.5, 1.0, 2.0]

    cat_data: Dict[str, list] = defaultdict(list)
    for a, p, c in zip(actuals, preds, categories):
        cat_data[c].append((a, p))

    th_headers = "  ".join(f"{'±'+str(t):>6}" for t in thresholds)
    col_w = 8 + 9 * len(thresholds)
    print(f"\n{chr(9472) * (84 + col_w)}")
    print("  Per-category test metrics")
    print(f"{chr(9472) * (84 + col_w)}")
    print(
        f"  {'Category':<28}  {'N':>5}  {'MAE':>10}  {'MedAE':>10}  "
        f"{'Bias':>11}  {'Fold':>6}  {th_headers}"
    )
    sep = f"  {chr(9472)*28}  {chr(9472)*5}  {chr(9472)*10}  {chr(9472)*10}  " \
          f"{chr(9472)*11}  {chr(9472)*6}  " + \
          "  ".join(chr(9472)*6 for _ in thresholds)
    print(sep)

    per_cat = {}
    for cat in sorted(cat_data.keys()):
        pairs = cat_data[cat]
        n     = len(pairs)
        a_arr = np.array([x[0] for x in pairs])
        p_arr = np.array([x[1] for x in pairs])
        ae    = np.abs(a_arr - p_arr)
        mae   = float(np.mean(ae))
        medae = float(np.median(ae))
        bias  = float(np.mean(p_arr - a_arr))
        within_vals = [float(np.mean(ae <= t) * 100) for t in thresholds]

        fold_med = (
            float(np.median(fold_errors(a_arr, p_arr, forward_fn)))
            if forward_fn is not None else float("nan")
        )

        within_str = "  ".join(f"{w:>5.1f}%" for w in within_vals)
        print(
            f"  {cat:<28}  {n:>5}  {mae:>10.4g}  {medae:>10.4g}  "
            f"{bias:>+11.4g}  {fold_med:>6.2f}  {within_str}"
        )
        per_cat[cat] = {
            "n": n, "mae": mae, "medae": medae, "bias": bias,
            "fold_median": fold_med,
            **{f"within_{t}": w for t, w in zip(thresholds, within_vals)},
        }

    print(f"{chr(9472) * (84 + col_w)}")
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
    print(f"  {'Actual':>12}  {'Predicted':>12}  {'Abs Err':>12}  Category")
    for i in worst:
        a, p, c = actuals[i], preds[i], categories[i]
        print(f"  {a:>12.4g}  {p:>12.4g}  {abs(a - p):>12.4g}  {c}")
