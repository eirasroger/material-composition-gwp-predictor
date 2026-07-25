"""
Diagnostic outputs: training curves, predicted-vs-actual scatter, residual histogram,
and the JSON diagnostics dump.

Paths and display labels are passed in from the caller so this module has no
knowledge of specific targets or directory layout.
"""

import json
from pathlib import Path

import numpy as np


def save_plots(
    history: dict,
    actuals: np.ndarray,
    preds: np.ndarray,
    paths: dict,
    display_name: str = "Target",
    unit: str = "",
):
    """
    paths must contain keys: 'training', 'scatter', 'residuals' (Path objects).
    """
    import matplotlib.pyplot as plt

    for p in paths.values():
        Path(p).parent.mkdir(parents=True, exist_ok=True)

    epochs = np.array(history["epoch"])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, history["train_loss"], label="Train loss")
    ax.plot(epochs, history["val_loss"],   label="Validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Huber loss (transformed target)")
    ax.set_title(f"Training and validation loss — {display_name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(paths["training"], dpi=160)
    plt.close(fig)

    unit_label = f" ({unit})" if unit else ""
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(actuals, preds, alpha=0.55)
    min_v = float(min(actuals.min(), preds.min()))
    max_v = float(max(actuals.max(), preds.max()))
    ax.plot([min_v, max_v], [min_v, max_v], linestyle="--")
    ax.set_xlabel(f"Actual{unit_label}")
    ax.set_ylabel(f"Predicted{unit_label}")
    ax.set_title(f"Predicted vs actual — {display_name}")
    fig.tight_layout()
    fig.savefig(paths["scatter"], dpi=160)
    plt.close(fig)

    residuals = preds - actuals
    fig, ax   = plt.subplots(figsize=(10, 6))
    ax.hist(residuals, bins=40)
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel(f"Residual (predicted − actual){unit_label}")
    ax.set_ylabel("Count")
    ax.set_title(f"Residual distribution — {display_name}")
    fig.tight_layout()
    fig.savefig(paths["residuals"], dpi=160)
    plt.close(fig)


def save_diagnostics(summary: dict, path: Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
