"""
Project configuration: paths, hyperparameters, constants, and seed initialisation.
"""

from pathlib import Path
from typing import Dict, Set

import numpy as np
import torch


# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = BASE_DIR / "dataset.json"

# ── Embedding backend ─────────────────────────────────────────────────────────
# Options:
#   "google_news" : local Word2Vec binary (.bin)
#   "fasttext"    : local/downloaded .vec
#   "custom_vec"  : user-provided .vec
EMBEDDING_BACKEND = "fasttext"

GOOGLE_NEWS_PATH  = BASE_DIR / "GoogleNews-vectors-negative300" / "GoogleNews-vectors-negative300.bin"

FASTTEXT_URL      = "https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip"
FASTTEXT_ZIP_PATH = BASE_DIR / "wiki-news-300d-1M.vec.zip"
FASTTEXT_DIR      = BASE_DIR / "wiki-news-300d-1M"
FASTTEXT_VEC_PATH = FASTTEXT_DIR / "wiki-news-300d-1M.vec"

CUSTOM_VEC_PATH   = BASE_DIR / "custom_vectors.vec"

# ── Output paths ──────────────────────────────────────────────────────────────
FIGURES_DIR        = BASE_DIR / "figures"

MODEL_PATH         = BASE_DIR / "ghg_model.pt"
DIAGNOSTICS_PATH   = BASE_DIR / "diagnostics_ghg.json"
TRAINING_PLOT_PATH = FIGURES_DIR / "training_curves_ghg.png"
PRED_SCATTER_PATH  = FIGURES_DIR / "pred_vs_actual_ghg.png"
RESIDUALS_PATH     = FIGURES_DIR / "residuals_ghg.png"

# ── Model hyperparameters ─────────────────────────────────────────────────────
EMBED_DIM          = 300
HIDDEN_DIMS        = [256, 128]
DROPOUT            = 0.2
LR                 = 1e-3
EPOCHS             = 200
BATCH_SIZE         = 64
PATIENCE           = 30
RANDOM_SEED        = 42

TARGET_FIELD       = "total_ghg"

GHG_MIN            = 0.0
GHG_MAX            = 10.0

MIN_CATEGORY_COUNT = 20

# 1 circularity-origin + 4 end-of-life percentages
N_CIRC_FEATURES    = 5

# ── Material token normalisation ──────────────────────────────────────────────
MATERIAL_VARIATIONS: Dict[str, str] = {
    "aluminium": "aluminum",
    "braces":    "suspenders",
    "colour":    "color",
    "fibre":     "fiber",
    "flavour":   "flavor",
    "gaol":      "jail",
    "kerb":      "curb",
    "litre":     "liter",
    "mould":     "mold",
    "petrol":    "gasoline",
    "sulphur":   "sulfur",
    "tyre":      "tire",
    "woollen":   "woolen",
}

STOP_WORDS: Set[str] = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by",
    "for", "from", "has", "have", "in", "is", "it", "its", "of",
    "on", "or", "that", "the", "their", "there", "they", "this",
    "to", "was", "were", "which", "with",
}

# Reproducibility: applied on import so any module that touches torch / numpy
# downstream sees the same seeds as the original single-file script did.
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
