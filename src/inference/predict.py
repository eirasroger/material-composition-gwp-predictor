"""
Inference helpers: load a trained checkpoint and produce GHG predictions.

Two entry points:
- ``load_model(checkpoint)`` -> ``LoadedModel`` (cache once, reuse for many predictions).
- ``predict_ghg(product, vocab, checkpoint=...)`` (backward-compatible: loads + predicts in one call).

Use ``load_model`` + ``predict_ghg_with_loaded`` for interactive applications where
a fresh ``torch.load`` per prediction would be unacceptable.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Union

import numpy as np
import torch

from src.config import GHG_MAX, GHG_MIN, MODEL_PATH
from src.data.preprocessing import normalize_product
from src.embeddings.encode import category_onehot, product_embedding
from src.model.network import GHGNet


@dataclass
class LoadedModel:
    model: GHGNet
    y_mean: float
    y_scale: float
    cat_index: Dict[str, int]
    input_dim: int


def load_model(checkpoint: Union[str, Path] = MODEL_PATH) -> LoadedModel:
    ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=False)

    model = GHGNet(
        input_dim=ckpt["input_dim"],
        hidden=ckpt["hidden_dims"],
        drop=ckpt["dropout"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    return LoadedModel(
        model=model,
        y_mean=float(ckpt["y_mean"]),
        y_scale=float(ckpt["y_scale"]),
        cat_index=ckpt["cat_index"],
        input_dim=int(ckpt["input_dim"]),
    )


def predict_ghg_with_loaded(
    product: dict,
    vocab: Dict[str, np.ndarray],
    loaded: LoadedModel,
) -> float:
    normalized = normalize_product(
        product, loaded.cat_index, require_target=False, ghg_min=GHG_MIN, ghg_max=GHG_MAX
    )
    if normalized is None:
        raise ValueError(
            "Invalid product for inference: missing kg unit, unknown/dropped category, "
            "invalid materials, or invalid circularity/material values."
        )

    mat_emb    = product_embedding(normalized["materials"], vocab)
    cat_emb    = category_onehot(normalized["category"], loaded.cat_index)
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
        pred_scaled = loaded.model(x).item()

    return float(np.expm1(pred_scaled * loaded.y_scale + loaded.y_mean))


def predict_ghg(
    product: dict,
    vocab: Dict[str, np.ndarray],
    checkpoint: Union[str, Path] = MODEL_PATH,
) -> float:
    return predict_ghg_with_loaded(product, vocab, load_model(checkpoint))
