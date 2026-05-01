"""
Inference adapter for the desktop GUI.

Loads the trained model, baked vocab, and material dropdown list once at
construction. ``predict(...)`` accepts UI-friendly inputs (category string, list
of material dicts, EoL share dict, origin percentage) and returns the predicted
GHG in kg CO2-eq / kg.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.embeddings.baked import load_vocab_npz
from src.inference.predict import LoadedModel, load_model, predict_ghg_with_loaded


def _default_assets_dir() -> Path:
    """``sys._MEIPASS`` when frozen by PyInstaller, otherwise ``desktop_app/assets``."""
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", "")) / "assets"
    return Path(__file__).resolve().parent / "assets"


@dataclass
class EolShares:
    recycling: float = 0.0
    hazardous: float = 0.0
    inert: float = 0.0
    incineration: float = 0.0


def build_product(
    category: str,
    materials: List[Dict[str, float]],
    eol: EolShares,
    origin_pct: float,
) -> dict:
    """
    Wrap UI inputs in the dict shape ``normalize_product`` expects.

    The four GUI EoL sliders map to a single ``future_use_*`` key each — the
    other "recycling-family" sub-pathways stay at zero. This is mathematically
    equivalent to spreading the recycling slider across them, because
    ``extract_circularity_features`` sums them before normalisation.
    """
    return {
        "reference_unit": "kg",
        "c_pcr": category,
        "product_integrity": {"materials": materials},
        "ghg_footprint": {"total_ghg": 0.0},
        "cyclability": {
            "circularity_origin_percentage":                  float(origin_pct),
            "future_use_recycling":                           float(eol.recycling),
            "future_use_composting":                          0.0,
            "future_use_valorisation / filling":              0.0,
            "future_use_reconditioning":                      0.0,
            "future_use_reuse":                               0.0,
            "future_use_hazardous waste":                     float(eol.hazardous),
            "future_use_inert and non-hazardous landfills":   float(eol.inert),
            "future_use_incineration":                        float(eol.incineration),
        },
    }


class InferenceAdapter:
    def __init__(self, assets_dir: Optional[Path] = None):
        self.assets_dir = Path(assets_dir) if assets_dir else _default_assets_dir()

        model_path     = self.assets_dir / "ghg_model.pt"
        vocab_path     = self.assets_dir / "vocab.npz"
        materials_path = self.assets_dir / "materials.json"

        for p in (model_path, vocab_path, materials_path):
            if not p.exists():
                raise FileNotFoundError(
                    f"Missing asset: {p}. Run desktop_app/tools/bake_assets.py first."
                )

        self.loaded: LoadedModel = load_model(model_path)
        self.vocab               = load_vocab_npz(vocab_path)
        with open(materials_path, "r", encoding="utf-8") as f:
            self.materials: List[str] = json.load(f)

    @property
    def categories(self) -> List[str]:
        return sorted(self.loaded.cat_index.keys())

    def predict(
        self,
        category: str,
        materials: List[Dict[str, float]],
        eol: EolShares,
        origin_pct: float,
    ) -> float:
        product = build_product(category, materials, eol, origin_pct)
        return predict_ghg_with_loaded(product, self.vocab, self.loaded)


# ──────────────────────────────────────────────────────────────────────────────
# Smoke test: pick a random labelled product from the dataset, predict via the
# original (path-based) predict_ghg, then via this adapter, and compare.
# ──────────────────────────────────────────────────────────────────────────────
def _smoke_test() -> None:
    from src.config import DATASET_PATH, MODEL_PATH
    from src.data.loader import filter_reference_unit_kg, load_dataset
    from src.data.preprocessing import filter_valid_products
    from src.inference.predict import predict_ghg
    from src.utils import safe_float

    adapter = InferenceAdapter()
    print(f"Adapter loaded: {len(adapter.categories)} categories, "
          f"{len(adapter.materials)} materials, vocab {len(adapter.vocab)} tokens")

    products  = load_dataset(DATASET_PATH)
    products  = filter_reference_unit_kg(products)
    valid     = filter_valid_products(products, adapter.loaded.cat_index)

    print(f"\nValid products: {len(valid)}")

    # Test 5 evenly-spaced samples.
    n = len(valid)
    indices = [int(i * n / 5) for i in range(5)]
    max_diff = 0.0

    for idx in indices:
        sample = valid[idx]
        raw    = sample["raw"]

        # Path A: original predict (full path-based pipeline).
        pred_a = predict_ghg(raw, adapter.vocab, checkpoint=MODEL_PATH)

        # Path B: adapter (UI-style inputs reconstructed from the raw product).
        cyc = raw.get("cyclability") or {}
        recycling = sum(
            safe_float(cyc.get(k)) or 0.0
            for k in (
                "future_use_recycling",
                "future_use_composting",
                "future_use_valorisation / filling",
                "future_use_reconditioning",
                "future_use_reuse",
            )
        )
        eol = EolShares(
            recycling=recycling,
            hazardous=safe_float(cyc.get("future_use_hazardous waste"))                   or 0.0,
            inert=    safe_float(cyc.get("future_use_inert and non-hazardous landfills")) or 0.0,
            incineration=safe_float(cyc.get("future_use_incineration"))                   or 0.0,
        )
        origin_pct = safe_float(cyc.get("circularity_origin_percentage")) or 0.0

        pred_b = adapter.predict(
            category=sample["category"],
            materials=sample["materials"],
            eol=eol,
            origin_pct=origin_pct,
        )

        diff = abs(pred_a - pred_b)
        max_diff = max(max_diff, diff)
        print(f"  [{idx:>5}] {sample['category']:<32} A={pred_a:.6f}  B={pred_b:.6f}  |dif|={diff:.2e}")

    print(f"\nMax |A - B| across samples: {max_diff:.2e}")
    if max_diff > 1e-5:
        raise SystemExit(f"FAIL: predictions diverge by {max_diff} (> 1e-5).")
    print("PASS: adapter matches original predict_ghg.")


if __name__ == "__main__":
    _smoke_test()
