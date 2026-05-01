"""
Diagnostic outputs: training curves, predicted-vs-actual scatter, residual histogram,
and the JSON diagnostics dump.
"""

import json

import numpy as np

from src.config import (
    DIAGNOSTICS_PATH,
    FIGURES_DIR,
    PRED_SCATTER_PATH,
    RESIDUALS_PATH,
    TRAINING_PLOT_PATH,
)


def save_plots(history: dict, actuals: np.ndarray, preds: np.ndarray):
    import matplotlib.pyplot as plt

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    epochs = np.array(history["epoch"])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, history["train_loss"], label="Train loss")
    ax.plot(epochs, history["val_loss"],   label="Validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Huber loss (log-scaled target)")
    ax.set_title("Training and validation loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(TRAINING_PLOT_PATH, dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(actuals, preds, alpha=0.55)
    min_v = float(min(actuals.min(), preds.min()))
    max_v = float(max(actuals.max(), preds.max()))
    ax.plot([min_v, max_v], [min_v, max_v], linestyle="--")
    ax.set_xlabel("Actual GHG (kg CO2-eq)")
    ax.set_ylabel("Predicted GHG (kg CO2-eq)")
    ax.set_title("Predicted vs actual GHG on test set")
    fig.tight_layout()
    fig.savefig(PRED_SCATTER_PATH, dpi=160)
    plt.close(fig)

    residuals = preds - actuals
    fig, ax   = plt.subplots(figsize=(10, 6))
    ax.hist(residuals, bins=40)
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Residual (predicted - actual)")
    ax.set_ylabel("Count")
    ax.set_title("Residual distribution on test set")
    fig.tight_layout()
    fig.savefig(RESIDUALS_PATH, dpi=160)
    plt.close(fig)


def save_diagnostics(summary: dict):
    with open(DIAGNOSTICS_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
