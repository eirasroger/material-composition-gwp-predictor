"""
Feature-matrix construction: material embedding + category one-hot + circularity feats.
"""

from typing import Dict, List, Tuple

import numpy as np

from src.embeddings.encode import category_onehot, product_embedding


def build_features(
    valid_products: list,
    vocab: Dict[str, np.ndarray],
    unified_cat_index: Dict[str, int],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Category encoding is coalesced: one one-hot over the union of PCR labels
    (excluding "N/A") and standardized-category labels. Each product fires
    exactly one slot -- its real PCR if reported, else its standardized
    category. An explicit has_pcr flag preserves the "was a PCR reported"
    signal that coalescing would otherwise discard.
    """
    X, y, categories = [], [], []
    for item in valid_products:
        mat_emb    = product_embedding(item["materials"], vocab)
        cat_emb    = category_onehot(item["category_resolved"], unified_cat_index)
        has_pcr    = np.array([1.0 if item["has_pcr"] else 0.0], dtype=np.float32)
        circ_feats = np.array([
            item["circularity_origin_pct"],
            item["recycling_pct"],
            item["hazardous_pct"],
            item["inert_pct"],
            item["incineration_pct"],
        ], dtype=np.float32) / 100.0

        X.append(np.concatenate([mat_emb, cat_emb, has_pcr, circ_feats]))
        y.append(item["target"])
        categories.append(item["category"])
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32), categories
