"""
Per-product validation, normalisation, and circularity feature extraction.
"""

from typing import Dict, Optional

from src.config import GHG_MAX, GHG_MIN, TARGET_FIELD
from src.utils import normalise_shares_to_100, safe_float


def _get_materials(product: dict) -> list:
    """
    Return the materials list from:
      product["product_integrity"]["materials"]
    """
    return (product.get("product_integrity") or {}).get("materials") or []


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
    ghg_min: float = GHG_MIN,
    ghg_max: float = GHG_MAX,
) -> Optional[dict]:
    if str(product.get("reference_unit", "")).strip().lower() != "kg":
        return None

    category = str(product.get("c_pcr", "")).strip()
    if category not in cat_index:
        return None

    ghg = None
    if require_target:
        ghg_raw = (product.get("ghg_footprint") or {}).get(TARGET_FIELD)
        ghg = safe_float(ghg_raw)
        if ghg is None:
            return None
        if ghg < ghg_min or ghg > ghg_max:
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
        "ghg":       ghg,
        "category":  category,
        "materials": cleaned_materials,
        "raw":       product,
        **circ_feats,
    }


def filter_valid_products(products: list, cat_index: Dict[str, int]) -> list:
    out = []
    skipped_category = skipped_target = skipped_materials = skipped_other = 0

    for p in products:
        category = str(p.get("c_pcr", "")).strip()
        if category not in cat_index:
            skipped_category += 1
            continue

        c = normalize_product(p, cat_index, require_target=True, ghg_min=GHG_MIN, ghg_max=GHG_MAX)
        if c is None:
            ghg_raw   = (p.get("ghg_footprint") or {}).get(TARGET_FIELD)
            ghg       = safe_float(ghg_raw)
            materials = _get_materials(p)
            if ghg is None or not (GHG_MIN <= ghg <= GHG_MAX):
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
    print(f"  Skipped (target GHG)            : {skipped_target}"
          f"  (missing, invalid, or outside [{GHG_MIN}, {GHG_MAX}])")
    print(f"  Skipped (materials)             : {skipped_materials}")
    print(f"  Skipped (other)                 : {skipped_other}")
    print(f"  Total skipped                   : {total_skipped}")
    return out
