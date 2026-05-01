"""
Bake the assets shipped in the desktop installer:

- ``vocab.npz``       embedding subset (only tokens needed by training-set materials)
- ``materials.json``  sorted list of material display strings for the UI dropdown
- ``ghg_model.pt``    copy of the trained checkpoint

Run from anywhere::

    python desktop_app/tools/bake_assets.py

Optional flags::

    --out-dir   destination for assets (default: desktop_app/assets)
    --model     trained checkpoint (default: src.config.MODEL_PATH)
    --dataset   training dataset (default: src.config.DATASET_PATH)
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.config import DATASET_PATH, MODEL_PATH
from src.data.loader import filter_reference_unit_kg, load_dataset
from src.data.preprocessing import filter_valid_products
from src.embeddings.baked import save_vocab_npz
from src.embeddings.vocab import get_vocab


def bake(model_path: Path, dataset_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(str(model_path), map_location="cpu", weights_only=False)
    cat_index = ckpt["cat_index"]
    print(f"Loaded cat_index from {model_path.name}: {len(cat_index)} categories")

    products = load_dataset(dataset_path)
    products = filter_reference_unit_kg(products)
    valid = filter_valid_products(products, cat_index)
    print(f"Valid products for asset baking: {len(valid)}")

    mat_strings = set()
    for p in valid:
        for m in p["materials"]:
            name = m["name"].strip()
            if name:
                mat_strings.add(name)
    materials_list = sorted(mat_strings, key=lambda s: s.lower())
    print(f"Unique material strings: {len(materials_list)}")

    vocab = get_vocab(valid)
    print(f"Vocab tokens loaded: {len(vocab)}")

    vocab_path     = out_dir / "vocab.npz"
    materials_path = out_dir / "materials.json"
    model_dest     = out_dir / "ghg_model.pt"

    save_vocab_npz(vocab, vocab_path)
    print(f"Saved {vocab_path} ({vocab_path.stat().st_size / 1024:.1f} KB)")

    with open(materials_path, "w", encoding="utf-8") as f:
        json.dump(materials_list, f, ensure_ascii=False, indent=2)
    print(f"Saved {materials_path}")

    if model_path.resolve() != model_dest.resolve():
        shutil.copyfile(model_path, model_dest)
    print(f"Copied {model_path.name} -> {model_dest}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "desktop_app" / "assets")
    parser.add_argument("--model",   type=Path, default=MODEL_PATH)
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    args = parser.parse_args()

    bake(args.model, args.dataset, args.out_dir)


if __name__ == "__main__":
    main()
