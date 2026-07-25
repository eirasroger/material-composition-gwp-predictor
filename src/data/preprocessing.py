"""
Per-product validation, normalisation, and circularity feature extraction.

The target value is extracted via a configurable field_path (list of JSON keys),
so this module is domain-agnostic — it works for GHG, eutrophication, etc.
"""

from typing import Dict, List, Optional

from src.config import GHG_MAX, GHG_MIN, TARGET_FIELD
from src.utils import normalise_shares_to_100, safe_float

_DEFAULT_FIELD_PATH = ["ghg_footprint", TARGET_FIELD]


def _get_materials(product: dict) -> list:
    return (product.get("product_integrity") or {}).get("materials") or []


def _get_field(product: dict, field_path: List[str]):
    """Navigate a list of JSON keys from the product root."""
    val = product
    for key in field_path:
        val = (val or {}).get(key)
    return val


def extract_circularity_features(product: dict) -> Optional[Dict[str, float]]:
    cyclability = product.get("cyclability") or {}

    circularity_origin_pct = safe_float(cyclability.get("circularity_origin_percentage")) or 0.0

    fu_recycling  = safe_float(cyclability.get("future_use_recycling"))                          or 0.0
    fu_composting = safe_float(cyclability.get("future_use_composting"))                         or 0.0
    fu_val_fill   = safe_float(cyclability.get("future_use_valorisation / filling"))             or 0.0
    fu_recond     = safe_float(cyclability.get("future_use_reconditioning"))                     or 0.0
    fu_reuse      = safe_float(cyclability.get("future_use_reuse"))                              or 0.0
    fu_hazardous  = safe_float(cyclability.get("future_use_hazardous waste"))                    or 0.0
    fu_inert      = safe_float(cyclability.get("future_use_inert and non-hazardous landfills"))  or 0.0
    fu_inciner    = safe_float(cyclability.get("future_use_incineration"))                       or 0.0

    recycling_pct = fu_recycling + fu_composting + fu_val_fill + fu_recond + fu_reuse
    eol = normalise_shares_to_100([recycling_pct, fu_hazardous, fu_inert, fu_inciner])

    return {
        "circularity_origin_pct": circularity_origin_pct,
        "recycling_pct":          eol[0],
        "hazardous_pct":          eol[1],
        "inert_pct":              eol[2],
        "incineration_pct":       eol[3],
    }


def normalize_product(
    product: dict,
    cat_index: Dict[str, int],
    require_target: bool = True,
    value_min: float = GHG_MIN,
    value_max: float = GHG_MAX,
    target_field_path: Optional[List[str]] = None,
) -> Optional[dict]:
    """
    Validate and normalise one product.  Returns None if the product should be
    dropped; otherwise returns a dict with keys: target, category, materials,
    circularity_origin_pct, recycling_pct, hazardous_pct, inert_pct,
    incineration_pct, raw.

    target_field_path: JSON key path to the target value
        (e.g. ["ghg_footprint", "total_ghg"]).  Defaults to the legacy GHG path.
    value_min / value_max: hard filter bounds on the raw target value.
    """
    if target_field_path is None:
        target_field_path = _DEFAULT_FIELD_PATH

    if str(product.get("reference_unit", "")).strip().lower() != "kg":
        return None

    category = str(product.get("c_pcr", "")).strip()
    if category not in cat_index:
        return None

    category_std = str(product.get("category_standardized", "")).strip()

    # Coalesced category: use the official PCR when reported, otherwise fall
    # back to the (always-populated) LLM-standardized category. "has_pcr"
    # is kept as an explicit feature so that signal isn't lost by coalescing.
    has_pcr          = category not in ("", "N/A")
    category_resolved = category if has_pcr else category_std

    target_val = None
    if require_target:
        raw_val = _get_field(product, target_field_path)
        target_val = safe_float(raw_val)
        if target_val is None:
            return None
        if target_val < value_min or target_val > value_max:
            return None

    materials = _get_materials(product)
    if not materials:
        return None

    all_pcts_missing = all(
        safe_float(m.get("percentage")) in (None, 0.0)
        for m in materials
    )

    cleaned_materials = []
    for m in materials:
        name = str(m.get("name", "")).strip()
        if not name:
            return None
        pct = safe_float(m.get("percentage")) or 0.0
        cleaned_materials.append({"name": name, "percentage": pct})

    if all_pcts_missing:
        equal_weight = 100.0 / len(cleaned_materials)
        for m in cleaned_materials:
            m["percentage"] = equal_weight
    elif sum(m["percentage"] for m in cleaned_materials) <= 0:
        return None

    circ_feats = extract_circularity_features(product)
    if circ_feats is None:
        return None

    return {
        "target":            target_val,
        "category":          category,
        "category_std":      category_std,
        "category_resolved": category_resolved,
        "has_pcr":           has_pcr,
        "materials":         cleaned_materials,
        "raw":               product,
        **circ_feats,
    }


def filter_valid_products(
    products: list,
    cat_index: Dict[str, int],
    value_min: float = GHG_MIN,
    value_max: float = GHG_MAX,
    target_field_path: Optional[List[str]] = None,
) -> list:
    if target_field_path is None:
        target_field_path = _DEFAULT_FIELD_PATH

    out = []
    skipped_category = skipped_target = skipped_materials = skipped_other = 0

    for p in products:
        category = str(p.get("c_pcr", "")).strip()
        if category not in cat_index:
            skipped_category += 1
            continue

        c = normalize_product(
            p, cat_index,
            require_target=True,
            value_min=value_min,
            value_max=value_max,
            target_field_path=target_field_path,
        )
        if c is None:
            raw_val    = _get_field(p, target_field_path)
            target_val = safe_float(raw_val)
            materials  = _get_materials(p)
            if target_val is None or not (value_min <= target_val <= value_max):
                skipped_target += 1
            elif not materials:
                skipped_materials += 1
            else:
                skipped_other += 1
            continue
        out.append(c)

    total_skipped = skipped_category + skipped_target + skipped_materials + skipped_other
    print("Product validation summary:")
    print(f"  Valid products                  : {len(out)}")
    print(f"  Skipped (low-count category)    : {skipped_category}")
    print(f"  Skipped (target value)          : {skipped_target}"
          f"  (missing, invalid, or outside [{value_min}, {value_max}])")
    print(f"  Skipped (materials)             : {skipped_materials}")
    print(f"  Skipped (other)                 : {skipped_other}")
    print(f"  Total skipped                   : {total_skipped}")
    return out
