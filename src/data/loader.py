"""
Raw dataset loading + first-pass filtering (reference unit, category index).
"""

import json
from collections import Counter
from pathlib import Path
from typing import Dict

from src.config import MIN_CATEGORY_COUNT


def load_dataset(path: Path) -> list:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else [data]


def filter_reference_unit_kg(products: list) -> list:
    filtered, skipped = [], 0
    for p in products:
        unit = p.get("reference_unit")
        if unit is None or str(unit).strip().lower() != "kg":
            skipped += 1
            continue
        filtered.append(p)
    print(f"Reference unit filter (kg only): {len(filtered)} kept | {skipped} removed")
    return filtered


def build_category_index(products: list, min_count: int = MIN_CATEGORY_COUNT) -> Dict[str, int]:
    counts = Counter(
        str(p.get("c_pcr", "")).strip()
        for p in products
        if str(p.get("c_pcr", "")).strip()
    )

    print(f"\n{chr(9472) * 54}")
    print(f"  Category distribution ({len(counts)} unique categories found)")
    print(f"{chr(9472) * 54}")
    print(f"  {'Category':<32}  {'Count':>6}  Status")
    print(f"  {chr(9472) * 32}  {chr(9472) * 6}  {chr(9472) * 20}")

    kept: Dict[str, int] = {}
    dropped: Dict[str, int] = {}
    for cat, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        if cnt >= min_count:
            kept[cat] = cnt
            status = "KEPT"
        else:
            dropped[cat] = cnt
            status = f"DROPPED (< {min_count})"
        print(f"  {cat:<32}  {cnt:>6}  {status}")

    print(f"{chr(9472) * 54}")
    print(f"  Categories kept    : {len(kept)}")
    print(f"  Categories dropped : {len(dropped)}  (threshold: < {min_count} products)")
    print(f"  Products in kept categories   : {sum(kept.values())}")
    print(f"  Products in dropped categories: {sum(dropped.values())}")
    print(f"{chr(9472) * 54}\n")

    if not kept:
        raise ValueError(
            f"No category has >= {min_count} products. "
            "Lower MIN_CATEGORY_COUNT or check your dataset."
        )

    return {cat: i for i, cat in enumerate(sorted(kept.keys()))}
