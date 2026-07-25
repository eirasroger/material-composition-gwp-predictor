"""
Inference adapter for the desktop GUI.

Loads every baked target checkpoint (all sharing one vocab) once at
construction. ``predict(...)`` accepts UI-friendly inputs (category string, list
of material dicts, EoL share dict, origin percentage) and returns the predicted
value for a target (default: ghg_total) in that target's unit.

The app currently only shows GHG Total -- no indicator/stage selector UI yet --
but the adapter loads and exposes every baked target so that UI is additive,
not a rework, once more indicators/stages are trained. See
docs/LEARNINGS.md 2026-07-25 and TARGET_CONFIGS' indicator_key/stage_key.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.embeddings.baked import load_vocab_npz
from src.inference.predict import LoadedModel, load_model, predict_ghg_with_loaded

DEFAULT_TARGET_KEY = "ghg_total"

# Marks a dropdown entry as a standardized (non-PCR) category rather than a
# real c_pcr value -- see build_product() and InferenceAdapter.categories.
STD_CATEGORY_PREFIX = "Other: "


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

    ``category`` is either a real PCR value, or a standardized-category entry
    prefixed with STD_CATEGORY_PREFIX (produced by InferenceAdapter.categories
    for products with no formal PCR) -- see the coalesced category encoding
    in docs/LEARNINGS.md 2026-07-25.

    The four GUI EoL sliders map to a single ``future_use_*`` key each — the
    other "recycling-family" sub-pathways stay at zero. This is mathematically
    equivalent to spreading the recycling slider across them, because
    ``extract_circularity_features`` sums them before normalisation.
    """
    if category.startswith(STD_CATEGORY_PREFIX):
        c_pcr               = "N/A"
        category_standardized = category[len(STD_CATEGORY_PREFIX):]
    else:
        c_pcr               = category
        category_standardized = ""

    return {
        "reference_unit": "kg",
        "c_pcr": c_pcr,
        "category_standardized": category_standardized,
        "product_integrity": {"materials": materials},
        "ghg_footprint": {"total_ghg": 0.0},
        "cyclability": {
            "circularity_origin_percentage":                  float(origin_pct),
            "future_use_recycling":                           float(eol.recycling),
            "future_use_composting":                          0.0,
            "future_use_valorisation / filling":              0.0,
            "future_use_reconditioning":                      0.0,
            "future_use_reuse":                                0.0,
            "future_use_hazardous waste":                     float(eol.hazardous),
            "future_use_inert and non-hazardous landfills":   float(eol.inert),
            "future_use_incineration":                        float(eol.incineration),
        },
    }


class InferenceAdapter:
    def __init__(self, assets_dir: Optional[Path] = None):
        self.assets_dir = Path(assets_dir) if assets_dir else _default_assets_dir()

        vocab_path     = self.assets_dir / "vocab.npz"
        materials_path = self.assets_dir / "materials.json"
        manifest_path  = self.assets_dir / "targets_manifest.json"
        models_dir     = self.assets_dir / "models"

        for p in (vocab_path, materials_path):
            if not p.exists():
                raise FileNotFoundError(
                    f"Missing asset: {p}. Run desktop_app/tools/bake_assets.py first."
                )

        self.loaded: Dict[str, LoadedModel] = {}

        if manifest_path.exists():
            with open(manifest_path, "r", encoding="utf-8") as f:
                self.manifest: Dict[str, dict] = json.load(f)
            for key in self.manifest:
                p = models_dir / f"{key}.pt"
                if not p.exists():
                    raise FileNotFoundError(
                        f"Missing checkpoint {p} referenced by targets_manifest.json. "
                        "Re-run desktop_app/tools/bake_assets.py."
                    )
                self.loaded[key] = load_model(p)
        else:
            # Backward compat: pre-multi-target asset layout (single ghg_model.pt
            # directly under assets_dir, no manifest). Re-bake to upgrade.
            legacy_model_path = self.assets_dir / "ghg_model.pt"
            if not legacy_model_path.exists():
                raise FileNotFoundError(
                    f"No targets_manifest.json and no legacy {legacy_model_path}. "
                    "Run desktop_app/tools/bake_assets.py first."
                )
            loaded = load_model(legacy_model_path)
            self.loaded[DEFAULT_TARGET_KEY] = loaded
            self.manifest = {
                DEFAULT_TARGET_KEY: {
                    "indicator_key": "ghg", "stage_key": "total",
                    "display_name": loaded.display_name, "unit": loaded.unit,
                }
            }

        self.default_target_key = (
            DEFAULT_TARGET_KEY if DEFAULT_TARGET_KEY in self.loaded else next(iter(self.loaded))
        )

        self.vocab = load_vocab_npz(vocab_path)
        with open(materials_path, "r", encoding="utf-8") as f:
            self.materials: List[str] = json.load(f)

        cat_materials_path = self.assets_dir / "category_materials.json"
        if cat_materials_path.exists():
            with open(cat_materials_path, "r", encoding="utf-8") as f:
                raw: Dict[str, List[str]] = json.load(f)
            std_index = self._default_loaded.std_cat_index or {}
            valid_cats = set(self._default_loaded.cat_index.keys()) | {
                f"{STD_CATEGORY_PREFIX}{c}" for c in std_index
            }
            self.category_materials: Dict[str, List[str]] = {
                k: v for k, v in raw.items() if k in valid_cats
            }
        else:
            self.category_materials: Dict[str, List[str]] = {}

    @property
    def _default_loaded(self) -> LoadedModel:
        return self.loaded[self.default_target_key]

    @property
    def target_keys(self) -> List[str]:
        return list(self.loaded.keys())

    @property
    def categories(self) -> List[str]:
        """
        Real PCR categories first (frequency-ordered, "N/A" excluded -- it's
        not a category, it's the absence of one), then standardized categories
        as "Other: X" entries so a user with no formal PCR still gets a
        meaningful, model-usable choice instead of a bare "N/A".
        """
        if self.category_materials:
            pcr_cats = [
                c for c in self.category_materials.keys()
                if c not in ("", "N/A") and not c.startswith(STD_CATEGORY_PREFIX)
            ]
        else:
            pcr_cats = sorted(c for c in self._default_loaded.cat_index.keys() if c not in ("", "N/A"))

        std_index = self._default_loaded.std_cat_index or {}
        other_cats = [f"{STD_CATEGORY_PREFIX}{c}" for c in sorted(std_index.keys())]
        return pcr_cats + other_cats

    @property
    def default_indicator_key(self) -> str:
        return self.manifest[self.default_target_key]["indicator_key"]

    def stage_target_keys(self, indicator_key: Optional[str] = None) -> Dict[str, str]:
        """{stage_key: target_key} for every non-'total' stage of the given (default) indicator."""
        indicator_key = indicator_key or self.default_indicator_key
        return {
            v["stage_key"]: k
            for k, v in self.manifest.items()
            if v["indicator_key"] == indicator_key and v["stage_key"] != "total"
        }

    def predict(
        self,
        category: str,
        materials: List[Dict[str, float]],
        eol: EolShares,
        origin_pct: float,
        target_key: Optional[str] = None,
    ) -> float:
        target_key = target_key or self.default_target_key
        product = build_product(category, materials, eol, origin_pct)
        return predict_ghg_with_loaded(product, self.vocab, self.loaded[target_key])

    def predict_all(
        self,
        category: str,
        materials: List[Dict[str, float]],
        eol: EolShares,
        origin_pct: float,
    ) -> Dict[str, float]:
        """Predict every loaded target off one shared product description."""
        product = build_product(category, materials, eol, origin_pct)
        return {
            key: predict_ghg_with_loaded(product, self.vocab, loaded)
            for key, loaded in self.loaded.items()
        }

    def prediction_range(
        self, prediction: float, category: str, target_key: Optional[str] = None
    ) -> Optional[Tuple[float, float]]:
        target_key = target_key or self.default_target_key
        loaded = self.loaded[target_key]
        bounds = loaded.category_error_bounds
        if not bounds or category not in bounds:
            return None
        b = bounds[category]
        low  = max(loaded.value_min, prediction - b["p75"])
        high = min(loaded.value_max, prediction - b["p25"])
        if low >= high:
            return None
        return (low, high)


# ──────────────────────────────────────────────────────────────────────────────
# Smoke test: pick a random labelled product from the dataset, predict via the
# original (path-based) predict_ghg, then via this adapter, and compare.
# ──────────────────────────────────────────────────────────────────────────────
def _smoke_test() -> None:
    from src.config import DATASET_PATH, MODELS_DIR
    from src.data.loader import filter_reference_unit_kg, load_dataset
    from src.data.preprocessing import filter_valid_products
    from src.inference.predict import predict_ghg
    from src.utils import safe_float

    adapter = InferenceAdapter()
    print(f"Adapter loaded: {len(adapter.target_keys)} target(s) {adapter.target_keys}, "
          f"{len(adapter.categories)} categories, "
          f"{len(adapter.materials)} materials, vocab {len(adapter.vocab)} tokens")

    checkpoint_path = MODELS_DIR / f"{adapter.default_target_key}.pt"

    products  = load_dataset(DATASET_PATH)
    products  = filter_reference_unit_kg(products)
    valid     = filter_valid_products(products, adapter._default_loaded.cat_index)

    print(f"\nValid products: {len(valid)}")

    # Test 5 evenly-spaced samples.
    n = len(valid)
    indices = [int(i * n / 10) for i in range(10)]
    max_diff_known_pcr = 0.0
    max_diff_na = 0.0

    for idx in indices:
        sample = valid[idx]
        raw    = sample["raw"]

        # Path A: original predict (full path-based pipeline).
        pred_a = predict_ghg(raw, adapter.vocab, checkpoint=checkpoint_path)

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

        has_pcr = sample["category"] not in ("", "N/A")
        # Mirror what a live user would actually select: the real PCR if
        # reported, else the "Other: <standardized>" entry the category
        # dropdown now exposes for exactly this case.
        ui_category = sample["category"] if has_pcr else f"{STD_CATEGORY_PREFIX}{sample['category_std']}"

        pred_b = adapter.predict(
            category=ui_category,
            materials=sample["materials"],
            eol=eol,
            origin_pct=origin_pct,
        )

        diff = abs(pred_a - pred_b)
        if has_pcr:
            max_diff_known_pcr = max(max_diff_known_pcr, diff)
        else:
            max_diff_na = max(max_diff_na, diff)
        print(f"  [{idx:>5}] {ui_category:<45} A={pred_a:.6f}  B={pred_b:.6f}  |dif|={diff:.2e}")

    print(f"\nMax |A - B| on known-PCR samples: {max_diff_known_pcr:.2e}")
    print(f"Max |A - B| on 'Other: X' samples: {max_diff_na:.2e}")
    print(
        "\nBoth must match to <=1e-5: the category dropdown's \"Other: X\" entries (for products with\n"
        "no formal PCR) now carry full input parity with what predict_ghg replays from the raw dataset."
    )
    max_diff = max(max_diff_known_pcr, max_diff_na)
    if max_diff > 1e-5:
        raise SystemExit(f"FAIL: predictions diverge by {max_diff} (> 1e-5).")
    print("\nPASS: adapter matches original predict_ghg.")


if __name__ == "__main__":
    _smoke_test()
