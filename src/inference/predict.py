"""
Inference helper: load a trained checkpoint and produce a single GHG prediction.
"""

from pathlib import Path
from typing import Dict, Union

import numpy as np
import torch

from src.config import GHG_MAX, GHG_MIN, MODEL_PATH
from src.data.preprocessing import normalize_product
from src.embeddings.encode import category_onehot, product_embedding
from src.model.network import GHGNet


def predict_ghg(
    product: dict,
    vocab: Dict[str, np.ndarray],
    checkpoint: Union[str, Path] = MODEL_PATH,
) -> float:
    ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=False)

    y_mean:    float          = float(ckpt["y_mean"])
    y_scale:   float          = float(ckpt["y_scale"])
    cat_index: Dict[str, int] = ckpt["cat_index"]
    input_dim: int            = ckpt["input_dim"]

    model = GHGNet(input_dim=input_dim, hidden=ckpt["hidden_dims"], drop=ckpt["dropout"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    normalized = normalize_product(
        product, cat_index, require_target=False, ghg_min=GHG_MIN, ghg_max=GHG_MAX
    )
    if normalized is None:
        raise ValueError(
            "Invalid product for inference: missing kg unit, unknown/dropped category, "
            "invalid materials, or invalid circularity/material values."
        )

    mat_emb    = product_embedding(normalized["materials"], vocab)
    cat_emb    = category_onehot(normalized["category"], cat_index)
    circ_feats = np.array([
        normalized["circularity_origin_pct"],
        normalized["recycling_pct"],
        normalized["hazardous_pct"],
        normalized["inert_pct"],
        normalized["incineration_pct"],
    ], dtype=np.float32)

    x = torch.tensor(
        np.concatenate([mat_emb, cat_emb, circ_feats]), dtype=torch.float32
    ).unsqueeze(0)

    with torch.no_grad():
        pred_scaled = model(x).item()

    return float(np.expm1(pred_scaled * y_scale + y_mean))
