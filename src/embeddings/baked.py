"""
Save / load a pre-baked embedding subset as a small ``.npz`` file.

Used by the desktop app to ship only the vectors actually referenced by the
training dataset's materials, instead of the full ~1 GB fastText / Word2Vec
file. Format:

    npz["tokens"]  : np.ndarray[str], shape (N,)
    npz["vectors"] : np.ndarray[float32], shape (N, EMBED_DIM)
"""

from pathlib import Path
from typing import Dict, Union

import numpy as np

from src.config import EMBED_DIM


def save_vocab_npz(vocab: Dict[str, np.ndarray], path: Union[str, Path]) -> None:
    if not vocab:
        raise ValueError("Refusing to save an empty vocab.")

    tokens = np.array(sorted(vocab.keys()), dtype=object)
    vectors = np.stack([np.asarray(vocab[t], dtype=np.float32) for t in tokens], axis=0)

    if vectors.shape[1] != EMBED_DIM:
        raise ValueError(
            f"Vector dim mismatch: got {vectors.shape[1]}, expected {EMBED_DIM}."
        )

    np.savez_compressed(str(path), tokens=tokens, vectors=vectors)


def load_vocab_npz(path: Union[str, Path]) -> Dict[str, np.ndarray]:
    with np.load(str(path), allow_pickle=True) as npz:
        tokens = npz["tokens"]
        vectors = npz["vectors"].astype(np.float32, copy=False)

    if vectors.shape[1] != EMBED_DIM:
        raise ValueError(
            f"Vector dim mismatch in {path}: got {vectors.shape[1]}, expected {EMBED_DIM}."
        )

    return {str(t): vectors[i] for i, t in enumerate(tokens)}
