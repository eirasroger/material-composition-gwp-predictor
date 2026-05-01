"""
Generic helpers: numeric coercion, tokenisation, scoring guards.
"""

import math
import re
from typing import List, Optional

import numpy as np
from sklearn.metrics import r2_score

from src.config import MATERIAL_VARIATIONS, STOP_WORDS


def safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, str):
            s = x.strip().lower()
            if s in {"", "undefined", "null", "none", "nan"}:
                return None
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def tokenise(text: str) -> List[str]:
    """Plain alphabetic tokeniser — no variation mapping, no stopword filtering."""
    return re.findall(r"[a-zA-Z]+", str(text).lower())


def tokenise_material(text: str) -> List[str]:
    """
    Tokenise a material name with British->American normalisation and stopword removal.
    Uses \b word-boundary anchors (single backslash) so re.findall splits
    hyphenated tokens like "glass-fibre" correctly.
    """
    tokens = re.findall(r"\b[a-zA-Z]+\b", str(text).lower())   # ← single \b
    tokens = [MATERIAL_VARIATIONS.get(t, t) for t in tokens]
    return [t for t in tokens if t not in STOP_WORDS]


def r2_safe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return float("nan")
    return float(r2_score(y_true, y_pred))


def normalise_shares_to_100(values: List[float]) -> List[float]:
    total = sum(values)
    if total <= 0:
        return [0.0 for _ in values]
    return [100.0 * v / total for v in values]
