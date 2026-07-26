# Environmental impact prediction from material composition

Deep learning models that predict the environmental impact of construction products from their material composition, product category and circularity data. Material names are vectorised with word embeddings and weighted by mass fraction to form a product-level representation.

Two components:

- **ML pipeline** — trains and evaluates one model per target on a labelled product dataset.
- **Desktop application** — a standalone Windows GUI: live prediction across every indicator, multi-product comparison, and a saved-scenario library. See [`desktop_app/README.md`](desktop_app/README.md).

---

## Targets

25 models — 5 indicators × 5 lifecycle stages, one checkpoint each in `models/`:

| indicator | field block | unit | stages |
|---|---|---|---|
| `ghg` | `ghg_footprint` | kg CO₂-eq/kg | `total`, `a1a3`, `c3`, `c4`, `d` |
| `fw` | `water_footprint` | m³/kg | ” |
| `ep` | `biodiversity_impact.eutrophication` | kg PO₄-eq/kg | ” |
| `ap` | `biodiversity_impact.acidification` | kg SO₂-eq/kg | ” |
| `adpf` | `biodiversity_impact.abiotic_depletion_fossil` | MJ/kg | ” |

Stages A4/A5/C1/C2 are transport- and process-driven, so they are outside the scope of composition-based prediction. Stage D (avoided burden) is ~74 % negative and uses a signed transform.

Each model is self-contained; indicators do not share weights. Adding a target takes one `TARGET_CONFIGS` entry in `src/config.py`.

---

## Model inputs

| Feature | Description | Dimensionality |
|---|---|---|
| Material composition | Material names + mass fractions → fastText embeddings, mass-fraction-weighted average, L2-normalised | 300 |
| Product category | One coalesced one-hot: the formal c-PCR when reported, otherwise `category_standardized` | n_categories |
| Has PCR | 1.0 when a formal c-PCR was reported | 1 |
| Circularity | Circular origin %, recycling %, hazardous %, inert %, incineration % | 5 |

73 % of products carry no formal c-PCR, so the category encoding falls back to the standardized category and flags whether a formal one was present.

**Output**: the target's value per kg of product, in that indicator's unit.

---

## Dataset

~8 500 labelled construction products. The dataset is confidential and not included here; the format is documented so compatible datasets can be prepared.

`dataset.json` is a JSON array of product objects:

```jsonc
[
  {
    // ── Required ───────────────────────────────────────────────────────────
    "reference_unit": "kg",          // must be exactly "kg"

    "c_pcr": "Textiles",             // formal PCR category, or "N/A"
    "category_standardized": "Coating and cladding",   // always present

    "product_integrity": {
      "materials": [
        { "name": "cotton",    "percentage": 80 },   // material name + mass fraction
        { "name": "polyester", "percentage": 20 }    // if all are 0 or missing,
      ]                                              // equal weights are assigned
    },

    // At least one target block. Each indicator carries a total plus per-stage
    // values; any individual value may be null.
    "ghg_footprint":   { "total_ghg": 3.14, "ghg_A1_A3": 2.29, "ghg_C3": 0.0,
                         "ghg_C4": 0.03, "ghg_D": 0.0 },
    "water_footprint": { "total_water": 0.038, "fw_A1_A3": 0.034, "fw_C3": 0.0,
                         "fw_C4": 1.6e-4, "fw_D": 0.0 },
    "biodiversity_impact": {
      "eutrophication":           { "total_ep": 1.6e-3, "ep_A1_A3": 1.3e-3 },
      "acidification":            { "total_ap": 0.020,  "ap_A1_A3": 0.018 },
      "abiotic_depletion_fossil": { "total_adpf": 52.6, "adpf_A1_A3": 41.4 }
    },

    // ── Optional (default 0.0 if missing) ──────────────────────────────────
    "cyclability": {
      "circularity_origin_percentage":          30,

      // Five "future_use_*" keys aggregate into one "recycling" share, then
      // normalise with hazardous, inert and incineration to sum to 100 %:
      "future_use_recycling":                   60,
      "future_use_composting":                   0,
      "future_use_valorisation / filling":       0,
      "future_use_reconditioning":               0,
      "future_use_reuse":                        0,
      "future_use_hazardous waste":              0,
      "future_use_inert and non-hazardous landfills": 20,
      "future_use_incineration":                20
    }
  }
]
```

**Filtering, applied per target:**

| Rule | Effect |
|---|---|
| `reference_unit != "kg"` | Product dropped |
| Category has fewer than `MIN_CATEGORY_COUNT` (10) products | Category dropped — applied to the dataset, then again after each target's own filtering |
| Target value missing, non-numeric, or outside that target's `[value_min, value_max]` | Product dropped for that target only |
| `product_integrity.materials` empty or missing | Product dropped |
| All material percentages 0 or missing | Equal mass fractions assigned |
| Any `cyclability` field missing | Defaults to 0.0, product kept |

The category threshold is re-applied after filtering because each target drops a different subset, and a category left with too few products cannot be stratified across the train/val/test split.

---

## Project layout

```
material-composition-attribute-predictor/
├── main.py                          training entrypoint  →  python main.py [--target TARGET]
├── dataset.json                     labelled product dataset (not in git)
├── models/                          one checkpoint per target (25)
├── diagnostics/{target}.json        per-target run diagnostics
├── figures/{target}/                per-target plots
├── requirements.txt                 ML pipeline dependencies
├── src/                             ML pipeline
│   ├── config.py                    all hyperparams, paths, seeds, TARGET_CONFIGS
│   ├── utils.py                     safe_float, tokenise_material, make_transforms
│   ├── pipeline.py                  end-to-end orchestration
│   ├── data/
│   │   ├── loader.py                load_dataset, reference-unit filter, category index
│   │   ├── preprocessing.py         per-product validation, circularity features
│   │   └── features.py              build_features → X matrix
│   ├── embeddings/
│   │   ├── vocab.py                 fastText / W2V / custom .vec backends, OOV reporting
│   │   ├── encode.py                embed_material, product_embedding, category_onehot
│   │   └── baked.py                 save/load vocab.npz for the desktop app
│   ├── model/
│   │   ├── network.py               GHGNet — MLP with BatchNorm + ReLU + Dropout
│   │   └── dataset.py               PyTorch Dataset wrapper
│   ├── train/
│   │   ├── trainer.py               train_model — AdamW + HuberLoss + early stopping
│   │   └── evaluator.py             evaluate_model, per-category metrics
│   ├── inference/
│   │   └── predict.py               load_model → LoadedModel; predict_ghg_with_loaded
│   └── reporting/
│       └── plots.py                 training curves, scatter, residuals, diagnostics
├── desktop_app/                     Windows GUI  (see desktop_app/README.md)
├── wiki-news-300d-1M/               fastText vectors (~1 GB, not in git, auto-downloaded)
├── GoogleNews-vectors-negative300/  W2V binary (optional backend, not in git)
└── docs/
    ├── OVERVIEW.md                  technical reference
    └── LEARNINGS.md                 development diary
```

---

## Model architecture

**GHGNet** — a 3-layer MLP trained on standardised, log-transformed targets:

```
Linear(input_dim, 256) → BatchNorm1d → ReLU → Dropout(0.25)
Linear(256, 128)        → BatchNorm1d → ReLU → Dropout(0.25)
Linear(128, 1)          → scalar output
```

Training: AdamW (lr = 1e-3, weight_decay = 3e-4), HuberLoss (delta = 1.0), ReduceLROnPlateau, gradient clipping (max_norm = 1.0), early stopping (patience = 20, max 200 epochs, batch size 64). Full table in `src/config.py`.

**Transform scale.** Each target's forward transform is `log1p(x / s)` with inverse `s · expm1(y)`, using signed variants for stage D. The scale `s` is each target's median positive value, which sets where the transform crosses from linear to logarithmic — necessary because indicators span six orders of magnitude, from ~1e-6 kg PO₄-eq/kg to ~650 MJ/kg.

**Metrics**: MAE, MedAE, Bias, RMSE, within ±X, and **fold error** (`exp|forward(pred) − forward(actual)|`, i.e. "predicted within a factor of X"), which is comparable across indicators of any magnitude. Typical: median 1.07–1.39×, p90 2–6×.

Fitting in log space recovers the geometric mean, so the tail is systematically under-predicted. If this becomes material, the fix is a Duan smearing factor.

---

## Setup

Requires Python 3.10+.

```powershell
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Running the ML pipeline

Place the dataset as `dataset.json` at the repository root (configurable in `src/config.py`).

```bash
python main.py --list-targets       # print all 25 target keys
python main.py --target ghg_total   # train one target
python main.py                      # train the default target
```

The first run downloads the fastText vectors (~1 GB extracted) into `wiki-news-300d-1M/`. Set `EMBEDDING_BACKEND` in `src/config.py` to `"google_news"` or `"custom_vec"` for an alternative backend. CPU works; CUDA is used when available.

### Outputs

| Path | Contents |
|---|---|
| `models/{target}.pt` | Checkpoint — weights, scaler params, category indices, transform type and scale, per-category error bounds |
| `diagnostics/{target}.json` | Metrics dump: training history and per-category test breakdown |
| `figures/{target}/` | Training curves, predicted-vs-actual scatter, residuals |

All paths and hyperparameters live in `src/config.py`.

---

## Desktop application

Packages the trained models into a standalone Windows installer — no Python, no configuration, no admin rights.

**Install**: download `GHGPredictorSetup-<version>.exe` from the [Releases page](https://github.com/eirasroger/material-composition-attribute-predictor/releases), run it, and launch from the Start Menu. Updates install automatically.

**Features**

- All 25 targets predicted on every edit (~5 ms), so switching indicator is an instant redraw.
- Summary radar showing every indicator at once against the category median.
- Per-indicator view with a large readout, gauge and lifecycle-stage breakdown.
- Comparison of 2–4 products: bar chart plus aligned summary table, with a per-product category override.
- Scenario library: save a whole comparison by name and reopen it later. The last session is restored automatically on launch.
- Results are expressed as a ratio to the median of real products in the same category, with the p25–p75 band shown. Verdicts soften when the prediction's error range cannot settle them.

See [`desktop_app/README.md`](desktop_app/README.md) for the user guide, developer setup, and build and release workflow.

---

## Contact

Roger Vergés — Lead developer — [roger.verges.eiras@upc.edu](mailto:roger.verges.eiras@upc.edu)<a href="https://orcid.org/0009-0001-5887-4785" aria-label="ORCID"><img src="https://orcid.org/sites/default/files/images/orcid_16x16.png" alt="ORCID iD" width="16" height="16" style="vertical-align: text-bottom; margin-left: 4px;"></a>
