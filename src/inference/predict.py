"""
Inference helpers: load a trained checkpoint and produce predictions.

Two entry points:
- ``load_model(checkpoint)`` -> ``LoadedModel`` (cache once, reuse for many predictions).
- ``predict_ghg(product, vocab, checkpoint=...)`` (backward-compatible: loads + predicts).

Use ``load_model`` + ``predict_ghg_with_loaded`` for interactive applications.

The inverse transform (log1p or signed_log1p) is stored in the checkpoint under
'transform_type', together with its crossover in 'transform_scale', and both are
applied automatically.  Old checkpoints without those keys default to 'log1p'
and scale 1.0, which is what they were trained with.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import torch

from src.config import GHG_MAX, GHG_MIN, MODEL_PATH
from src.data.preprocessing import normalize_product
from src.embeddings.encode import category_onehot, product_embedding
from src.model.network import GHGNet
from src.utils import make_transforms


@dataclass
class LoadedModel:
    model: GHGNet
    y_mean: float
    y_scale: float
    cat_index: Dict[str, int]
    input_dim: int
    transform_type: str = "log1p"
    transform_scale: float = 1.0
    category_error_bounds: Optional[Dict[str, Dict]] = field(default=None)
    std_cat_index: Optional[Dict[str, int]] = field(default=None)
    unified_cat_index: Optional[Dict[str, int]] = field(default=None)
    resolved_category_error_bounds: Optional[Dict[str, Dict]] = field(default=None)
    value_min: float = GHG_MIN
    value_max: float = GHG_MAX
    target_key: str = "ghg_total"
    display_name: str = "GHG Total"
    unit: str = "kg CO2-eq/kg"


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
        transform_type=ckpt.get("transform_type", "log1p"),
        transform_scale=float(ckpt.get("transform_scale", 1.0)),
        category_error_bounds=ckpt.get("category_error_bounds"),
        std_cat_index=ckpt.get("std_cat_index"),
        unified_cat_index=ckpt.get("unified_cat_index"),
        resolved_category_error_bounds=ckpt.get("resolved_category_error_bounds"),
        value_min=float(ckpt.get("value_min", GHG_MIN)),
        value_max=float(ckpt.get("value_max", GHG_MAX)),
        target_key=ckpt.get("target_key", "ghg_total"),
        display_name=ckpt.get("display_name", "GHG Total"),
        unit=ckpt.get("unit", "kg CO2-eq/kg"),
    )


def predict_ghg_with_loaded(
    product: dict,
    vocab: Dict[str, np.ndarray],
    loaded: LoadedModel,
) -> float:
    normalized = normalize_product(
        product, loaded.cat_index, require_target=False,
        value_min=GHG_MIN, value_max=GHG_MAX,
    )
    if normalized is None:
        raise ValueError(
            "Invalid product for inference: missing kg unit, unknown/dropped category, "
            "invalid materials, or invalid circularity/material values."
        )

    mat_emb    = product_embedding(normalized["materials"], vocab)
    circ_feats = np.array([
        normalized["circularity_origin_pct"],
        normalized["recycling_pct"],
        normalized["hazardous_pct"],
        normalized["inert_pct"],
        normalized["incineration_pct"],
    ], dtype=np.float32) / 100.0

    if loaded.unified_cat_index is not None:
        # Coalesced encoding: PCR if reported, else standardized category,
        # plus an explicit has_pcr flag. Must mirror normalize_product's
        # category_resolved/has_pcr derivation exactly.
        cat_emb = category_onehot(normalized["category_resolved"], loaded.unified_cat_index)
        has_pcr = np.array([1.0 if normalized["has_pcr"] else 0.0], dtype=np.float32)
        blocks  = [mat_emb, cat_emb, has_pcr, circ_feats]
    else:
        cat_emb = category_onehot(normalized["category"], loaded.cat_index)
        blocks  = [mat_emb, cat_emb, circ_feats]

    x = torch.tensor(np.concatenate(blocks), dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        pred_scaled = loaded.model(x).item()

    _, inverse_fn = make_transforms(loaded.transform_type, loaded.transform_scale)
    return float(inverse_fn(pred_scaled * loaded.y_scale + loaded.y_mean))


def predict_ghg(
    product: dict,
    vocab: Dict[str, np.ndarray],
    checkpoint: Union[str, Path] = MODEL_PATH,
) -> float:
    return predict_ghg_with_loaded(product, vocab, load_model(checkpoint))
