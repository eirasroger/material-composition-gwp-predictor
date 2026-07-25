"""
Bake the assets shipped in the desktop installer:

- ``vocab.npz``               embedding subset (union of tokens needed across every baked target)
- ``materials.json``          material display strings sorted by global frequency
- ``category_materials.json`` per-category material lists sorted by frequency within each category
- ``models/{target_key}.pt``  copy of every trained checkpoint being baked
- ``targets_manifest.json``   {target_key: {indicator_key, stage_key, display_name, unit}}, lets the
                               app introspect available targets without loading every checkpoint
- ``distributions.json``      per-target reference distributions (p25/median/p75/n) of the REAL
                               training values, both per dropdown category and globally. The UI
                               reports a prediction as a ratio to its own category's median, so
                               "worse than a typical concrete" is answerable; a percentile against
                               all products would compare concrete to electronics and mean nothing.

By default bakes every target listed in TARGET_CONFIGS that has a trained
checkpoint under models/.

Run from anywhere::

    python desktop_app/tools/bake_assets.py

Optional flags::

    --out-dir     destination for assets (default: desktop_app/assets)
    --models-dir  directory of trained checkpoints (default: src.config.MODELS_DIR)
    --targets     space-separated target keys to bake (default: all with a checkpoint present)
    --dataset     training dataset (default: src.config.DATASET_PATH)
"""

import argparse
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from desktop_app.inference_adapter import STD_CATEGORY_PREFIX
from src.config import DATASET_PATH, MODELS_DIR, TARGET_CONFIGS
from src.data.loader import filter_reference_unit_kg, load_dataset
from src.data.preprocessing import filter_valid_products
from src.embeddings.baked import save_vocab_npz
from src.embeddings.vocab import get_vocab

# A category needs at least this many labelled products before its own p25/median/p75
# are trustworthy enough to compare a prediction against. Below it the UI falls back
# to the global distribution and says so, rather than quoting a median drawn from 11
# products as if it meant something.
MIN_REFERENCE_N = 30


def _dropdown_category(product: dict) -> str:
    """
    The exact string InferenceAdapter.categories() puts in the dropdown: the real
    PCR when reported, else "Other: <standardized>". Reference stats must be keyed
    the same way or the lookup silently misses.
    """
    return (
        product["category"] if product["has_pcr"]
        else f"{STD_CATEGORY_PREFIX}{product['category_resolved']}"
    )


def _stats(values: List[float]) -> Optional[dict]:
    """p25/median/p75 over *values*, or None if the median is not a usable divisor."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return None
    p25, p50, p75 = (float(v) for v in np.percentile(arr, [25, 50, 75]))
    # The UI divides by the median. Stage targets are heavily zero-inflated
    # (C3 is ~38% exact zeros), so a category median can legitimately be 0 or
    # negative (stage D) -- in which case there is no meaningful ratio to show.
    if p50 <= 0:
        return None
    return {"n": int(arr.size), "p25": p25, "p50": p50, "p75": p75}


def bake(
    models_dir: Path,
    dataset_path: Path,
    out_dir: Path,
    target_keys: Optional[List[str]] = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    models_out_dir = out_dir / "models"
    models_out_dir.mkdir(parents=True, exist_ok=True)

    available = {p.stem: p for p in models_dir.glob("*.pt")}
    if target_keys is None:
        target_keys = sorted(available)
    missing = [k for k in target_keys if k not in available]
    if missing:
        raise FileNotFoundError(
            f"No trained checkpoint for target(s) {missing} in {models_dir}. "
            f"Train with: python main.py --target <key>"
        )
    if not target_keys:
        raise FileNotFoundError(f"No trained checkpoints found in {models_dir}.")

    print(f"Baking {len(target_keys)} target(s): {target_keys}")

    products = load_dataset(dataset_path)
    products = filter_reference_unit_kg(products)

    manifest: dict = {}
    distributions: dict = {}
    cat_counts: Counter = Counter()
    cat_mat_counts: defaultdict = defaultdict(Counter)
    global_mat_counts: Counter = Counter()
    all_valid_for_vocab: list = []

    for key in target_keys:
        ckpt = torch.load(str(available[key]), map_location="cpu", weights_only=False)
        cat_index = ckpt["cat_index"]
        cfg = TARGET_CONFIGS.get(key, {})
        valid = filter_valid_products(
            products, cat_index,
            value_min=cfg.get("value_min", ckpt.get("value_min", 0.0)),
            value_max=cfg.get("value_max", ckpt.get("value_max", 10.0)),
            target_field_path=cfg.get("field_path", ckpt.get("field_path")),
        )
        print(f"  {key}: {len(valid)} valid products, {len(cat_index)} categories")
        all_valid_for_vocab.extend(valid)

        by_category: defaultdict = defaultdict(list)
        for p in valid:
            # Key by the same string InferenceAdapter.categories() exposes in the
            # dropdown -- the real PCR if reported, else "Other: <standardized>" --
            # so material ordering is tailored for "Other: X" selections too,
            # not just real PCR categories.
            cat = _dropdown_category(p)
            cat_counts[cat] += 1
            by_category[cat].append(p["target"])
            for m in p["materials"]:
                name = m["name"].strip()
                if name:
                    cat_mat_counts[cat][name] += 1
                    global_mat_counts[name] += 1

        per_category = {}
        for cat, values in by_category.items():
            if len(values) < MIN_REFERENCE_N:
                continue
            s = _stats(values)
            if s is not None:
                per_category[cat] = s
        distributions[key] = {
            "global":     _stats([p["target"] for p in valid]),
            "categories": per_category,
        }
        print(
            f"    reference distributions: {len(per_category)} categories "
            f"(n>={MIN_REFERENCE_N}), global "
            f"{'ok' if distributions[key]['global'] else 'unusable (median<=0)'}"
        )

        shutil.copyfile(available[key], models_out_dir / f"{key}.pt")

        manifest[key] = {
            "indicator_key": cfg.get("indicator_key", ckpt.get("target_key", key)),
            "stage_key":     cfg.get("stage_key", "total"),
            "display_name":  cfg.get("display_name", ckpt.get("display_name", key)),
            "unit":          cfg.get("unit", ckpt.get("unit", "")),
        }

    sorted_cats = sorted(cat_counts, key=lambda c: -cat_counts[c])
    category_materials = {
        cat: sorted(cat_mat_counts[cat], key=lambda m: -cat_mat_counts[cat][m])
        for cat in sorted_cats
    }
    materials_list = sorted(global_mat_counts, key=lambda m: -global_mat_counts[m])
    print(f"Unique material strings (union across targets): {len(materials_list)}")

    vocab = get_vocab(all_valid_for_vocab)
    print(f"Vocab tokens loaded: {len(vocab)}")

    vocab_path              = out_dir / "vocab.npz"
    materials_path          = out_dir / "materials.json"
    category_materials_path = out_dir / "category_materials.json"
    manifest_path           = out_dir / "targets_manifest.json"
    distributions_path      = out_dir / "distributions.json"

    save_vocab_npz(vocab, vocab_path)
    print(f"Saved {vocab_path} ({vocab_path.stat().st_size / 1024:.1f} KB)")

    with open(materials_path, "w", encoding="utf-8") as f:
        json.dump(materials_list, f, ensure_ascii=False, indent=2)
    print(f"Saved {materials_path}")

    with open(category_materials_path, "w", encoding="utf-8") as f:
        json.dump(category_materials, f, ensure_ascii=False, indent=2)
    print(f"Saved {category_materials_path} ({len(category_materials)} categories)")

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"Saved {manifest_path} ({len(manifest)} targets)")

    with open(distributions_path, "w", encoding="utf-8") as f:
        json.dump(
            {"min_reference_n": MIN_REFERENCE_N, "targets": distributions},
            f, ensure_ascii=False, indent=2,
        )
    print(
        f"Saved {distributions_path} "
        f"({distributions_path.stat().st_size / 1024:.1f} KB)"
    )
    print(f"Copied checkpoints -> {models_out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir",    type=Path, default=REPO_ROOT / "desktop_app" / "assets")
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--targets",    nargs="*", default=None)
    parser.add_argument("--dataset",    type=Path, default=DATASET_PATH)
    args = parser.parse_args()

    bake(args.models_dir, args.dataset, args.out_dir, target_keys=args.targets)


if __name__ == "__main__":
    main()
