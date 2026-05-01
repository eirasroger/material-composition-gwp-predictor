"""
Material → vector and category → one-hot encoders, plus the L2-normalised,
mass-fraction-weighted product embedding.
"""

from typing import Dict

import numpy as np

from src.config import EMBED_DIM
from src.utils import tokenise_material


def embed_material(name: str, vocab: Dict[str, np.ndarray]) -> np.ndarray:
    tokens = tokenise_material(name)
    if not tokens:
        return np.zeros(EMBED_DIM, dtype=np.float32)

    vecs = [
        np.asarray(vocab[t], dtype=np.float32) if t in vocab
        else np.zeros(EMBED_DIM, dtype=np.float32)
        for t in tokens
    ]
    return np.mean(vecs, axis=0).astype(np.float32)


def product_embedding(materials: list, vocab: Dict[str, np.ndarray]) -> np.ndarray:
    """L2-normalised, mass-fraction-weighted average embedding for a product."""
    total = sum(m["percentage"] for m in materials)
    if total <= 0:
        return np.zeros(EMBED_DIM, dtype=np.float32)

    vec = np.zeros(EMBED_DIM, dtype=np.float32)
    for m in materials:
        vec += (m["percentage"] / total) * embed_material(m["name"], vocab)

    norm = float(np.linalg.norm(vec))
    return (vec / norm).astype(np.float32) if norm > 0.0 else vec


def category_onehot(category: str, cat_index: Dict[str, int]) -> np.ndarray:
    vec = np.zeros(len(cat_index), dtype=np.float32)
    idx = cat_index.get(category)
    if idx is not None:
        vec[idx] = 1.0
    return vec
