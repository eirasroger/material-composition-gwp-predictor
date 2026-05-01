"""
Feature-matrix construction: material embedding + category one-hot + circularity feats.
"""

from typing import Dict, List, Tuple

import numpy as np

from src.embeddings.encode import category_onehot, product_embedding


def build_features(
    valid_products: list,
    vocab: Dict[str, np.ndarray],
    cat_index: Dict[str, int],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    X, y, categories = [], [], []
    for item in valid_products:
        mat_emb    = product_embedding(item["materials"], vocab)
        cat_emb    = category_onehot(item["category"], cat_index)
        circ_feats = np.array([
            item["circularity_origin_pct"],
            item["recycling_pct"],
            item["hazardous_pct"],
            item["inert_pct"],
            item["incineration_pct"],
        ], dtype=np.float32)

        X.append(np.concatenate([mat_emb, cat_emb, circ_feats]))
        y.append(item["ghg"])
        categories.append(item["category"])
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32), categories
