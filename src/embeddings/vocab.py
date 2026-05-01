"""
Pretrained-vector loaders for the three supported backends:
Google News Word2Vec binary, fastText .vec, and a user-supplied .vec.
"""

import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Set

import numpy as np

from src.config import (
    CUSTOM_VEC_PATH,
    EMBED_DIM,
    EMBEDDING_BACKEND,
    FASTTEXT_DIR,
    FASTTEXT_URL,
    FASTTEXT_VEC_PATH,
    FASTTEXT_ZIP_PATH,
    GOOGLE_NEWS_PATH,
)
from src.utils import tokenise_material


def _load_word2vec_binary(path: Path, needed_tokens: Set[str]) -> Dict[str, np.ndarray]:
    print(f"Loading Word2Vec binary from {path} ...")
    vecs: Dict[str, np.ndarray] = {}
    bytes_per_vec = EMBED_DIM * 4

    with open(path, "rb") as f:
        header = f.readline().decode("utf-8").strip()
        vocab_size, vector_size = map(int, header.split())

        if vector_size != EMBED_DIM:
            raise ValueError(
                f"Binary file has vector_size={vector_size} but EMBED_DIM={EMBED_DIM}. "
                "Update EMBED_DIM to match."
            )

        for _ in range(vocab_size):
            if len(vecs) == len(needed_tokens):
                break

            word_bytes = bytearray()
            while True:
                ch = f.read(1)
                if ch in (b" ", b""):
                    break
                if ch == b"\n":     # ← single backslash = actual newline byte
                    word_bytes = bytearray()
                    break
                word_bytes.extend(ch)

            word = word_bytes.decode("utf-8", errors="ignore").strip()

            raw = f.read(bytes_per_vec)
            if len(raw) < bytes_per_vec:
                break

            if word in needed_tokens:
                vecs[word] = np.frombuffer(raw, dtype=np.float32).copy()

    return vecs


def _ensure_fasttext_downloaded() -> Path:
    FASTTEXT_DIR.mkdir(exist_ok=True)
    if FASTTEXT_VEC_PATH.exists():
        return FASTTEXT_VEC_PATH
    if not FASTTEXT_ZIP_PATH.exists():
        print(f"Downloading fastText vectors ...\n  -> {FASTTEXT_URL}")
        urllib.request.urlretrieve(FASTTEXT_URL, str(FASTTEXT_ZIP_PATH))
    print("Extracting fastText vectors ...")
    with zipfile.ZipFile(str(FASTTEXT_ZIP_PATH), "r") as zf:
        zf.extractall(str(FASTTEXT_DIR))
    if not FASTTEXT_VEC_PATH.exists():
        raise FileNotFoundError(f"Expected extracted file not found: {FASTTEXT_VEC_PATH}")
    return FASTTEXT_VEC_PATH


def _load_vec_subset(vec_path: Path, needed_tokens: Set[str]) -> Dict[str, np.ndarray]:
    print(f"Loading vec subset from {vec_path} ...")
    vecs: Dict[str, np.ndarray] = {}

    with open(vec_path, "r", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline()
        if not first_line:
            return vecs

        first_parts = first_line.rstrip().split()
        has_header = (
            len(first_parts) == 2
            and first_parts[0].isdigit()
            and first_parts[1].isdigit()
        )

        def maybe_store(parts: List[str]) -> None:
            if len(parts) != EMBED_DIM + 1:
                return
            word = parts[0]
            if word not in needed_tokens:
                return
            try:
                vecs[word] = np.asarray(parts[1:], dtype=np.float32)
            except Exception:
                return

        if not has_header:
            maybe_store(first_parts)

        for line in f:
            if len(vecs) == len(needed_tokens):
                break
            maybe_store(line.rstrip().split())

    return vecs


def collect_needed_tokens(products: list) -> Set[str]:
    tokens: Set[str] = set()
    for item in products:
        for m in item["materials"]:
            tokens.update(tokenise_material(m["name"]))
    return tokens


def _report_missing_tokens(needed_tokens: Set[str], vocab: Dict[str, np.ndarray]) -> None:
    missing = needed_tokens - set(vocab.keys())
    found   = len(needed_tokens) - len(missing)
    print(f"  Tokens found : {found} / {len(needed_tokens)}")
    if missing:
        print(f"  Missing tokens ({len(missing)}) -- zero-vector fallback: {sorted(missing)}")


def get_vocab(products: list) -> Dict[str, np.ndarray]:
    needed_tokens = collect_needed_tokens(products)
    print(f"Unique tokens needed from embeddings: {len(needed_tokens):,}")
    print(f"Embedding backend: {EMBEDDING_BACKEND}")

    if EMBEDDING_BACKEND == "google_news":
        if not GOOGLE_NEWS_PATH.exists():
            raise FileNotFoundError(
                f"Google News binary not found at {GOOGLE_NEWS_PATH}.\n"
                "Switch EMBEDDING_BACKEND to 'fasttext' or fix the path."
            )
        vocab = _load_word2vec_binary(GOOGLE_NEWS_PATH, needed_tokens)
        _report_missing_tokens(needed_tokens, vocab)

    elif EMBEDDING_BACKEND == "fasttext":
        vec_path = _ensure_fasttext_downloaded()
        vocab    = _load_vec_subset(vec_path, needed_tokens)
        _report_missing_tokens(needed_tokens, vocab)

    elif EMBEDDING_BACKEND == "custom_vec":
        if not CUSTOM_VEC_PATH.exists():
            raise FileNotFoundError(
                f"Custom vec file not found at {CUSTOM_VEC_PATH}.\n"
                "Set CUSTOM_VEC_PATH to your .vec file and ensure EMBED_DIM matches."
            )
        vocab = _load_vec_subset(CUSTOM_VEC_PATH, needed_tokens)
        _report_missing_tokens(needed_tokens, vocab)

    else:
        raise ValueError(
            f"Unknown EMBEDDING_BACKEND '{EMBEDDING_BACKEND}'. "
            "Valid options: 'google_news', 'fasttext', 'custom_vec'."
        )

    return vocab
