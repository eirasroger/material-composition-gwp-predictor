"""
Generic helpers: numeric coercion, tokenisation, scoring guards.
"""

import math
import re
from typing import List, Optional

import numpy as np

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


def normalise_shares_to_100(values: List[float]) -> List[float]:
    total = sum(values)
    if total <= 0:
        return [0.0 for _ in values]
    return [100.0 * v / total for v in values]


def signed_log1p(x) -> np.ndarray:
    """sign(x) * log1p(|x|) — monotonic log-like transform that handles negatives."""
    a = np.asarray(x, dtype=np.float64)
    return np.sign(a) * np.log1p(np.abs(a))


def signed_expm1(x) -> np.ndarray:
    """Inverse of signed_log1p: sign(x) * expm1(|x|)."""
    a = np.asarray(x, dtype=np.float64)
    return np.sign(a) * np.expm1(np.abs(a))


def make_transforms(transform_type: str, scale: float = 1.0):
    """
    Return (forward_fn, inverse_fn) for a target, at the given crossover `scale`.

    `scale` is where the transform switches from ~linear to ~logarithmic:
    log1p(x/s) behaves like x/s for x << s and like log(x/s) for x >> s.
    Plain log1p (s=1) is therefore only a log transform when the target is
    O(1).  Indicators whose values sit at 1e-6..1e-3 (eutrophication,
    acidification, water depletion, and the GHG C3/C4/D stages) pass through
    log1p almost unchanged — skewness stays at 20-80 instead of dropping to
    ~2 — so each target carries its own `scale`, set to its median positive
    value.  s=1.0 reproduces the untransformed-by-scale behaviour exactly.
    """
    if scale <= 0:
        raise ValueError(f"transform scale must be > 0, got {scale}")

    s = float(scale)

    if transform_type == "log1p":
        return (
            lambda x: np.log1p(np.asarray(x, dtype=np.float64) / s),
            lambda y: s * np.expm1(np.asarray(y, dtype=np.float64)),
        )
    if transform_type == "signed_log1p":
        return (
            lambda x: signed_log1p(np.asarray(x, dtype=np.float64) / s),
            lambda y: s * signed_expm1(np.asarray(y, dtype=np.float64)),
        )
    raise ValueError(
        f"Unknown transform '{transform_type}'. Use 'log1p' or 'signed_log1p'."
    )
