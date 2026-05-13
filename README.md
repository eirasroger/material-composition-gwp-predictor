# material-composition-gwp-predictor

Deep learning model to predict the Global Warming Potential (GWP) of construction products from material composition. Materials are vectorised using word embeddings and weighted by their mass fractions to form product-level representations, then combined with product category and circularity/end-of-life data for accurate environmental impact prediction.

This repository contains two main contributions:

- **ML pipeline** — trains and evaluates GHGNet, the prediction model, on a labelled product dataset.
- **Desktop application** — a standalone Windows GUI that wraps the trained model, letting users interactively explore predicted GHG footprints by adjusting material composition, product category, end-of-life shares, and circularity parameters in real time. Supports single-product prediction and side-by-side multi-product comparison.

---

## Model inputs and output

| Feature | Description | Dimensionality |
|---|---|---|
| Material composition | Material names + mass fractions → fastText embeddings, mass-fraction-weighted average, L2-normalised | 300 |
| Product category | c-PCR label → one-hot encoding | n_categories |
| Circularity | Circular origin %, recycling %, inert landfill %, incineration %, hazardous waste % | 5 |

**Output**: predicted total GHG footprint in **kg CO₂-eq per kg of product**.

---

## Dataset

~8 800 labelled construction products. Each record contains material composition (name + percentage), a c-PCR product category, end-of-life breakdown, and a measured GHG footprint. Only products with `reference_unit == "kg"` are used in training. The dataset file is not included in this repository.

---

## Project layout

```
material-composition-gwp-predictor/
├── main.py                          training entrypoint  →  python main.py
├── dataset.json                     labelled product dataset (not in git)
├── ghg_model.pt                     trained checkpoint (produced by training)
├── diagnostics_ghg.json             full run diagnostics
├── figures/                         plots produced by training
├── requirements.txt                 ML pipeline dependencies
├── src/                             ML pipeline
│   ├── config.py                    all hyperparams, paths, constants, seeds
│   ├── utils.py                     helpers: safe_float, tokenise_material, normalise_shares_to_100
│   ├── pipeline.py                  end-to-end orchestration (called by main.py)
│   ├── data/
│   │   ├── loader.py                load_dataset, reference-unit filter, category index
│   │   ├── preprocessing.py         per-product validation and circularity feature extraction
│   │   └── features.py              build_features → X matrix (embedding + one-hot + circularity)
│   ├── embeddings/
│   │   ├── vocab.py                 loads fastText / W2V / custom .vec backends, OOV reporting
│   │   ├── encode.py                embed_material, product_embedding (L2-norm), category_onehot
│   │   └── baked.py                 save/load vocab.npz (compressed token subset for desktop app)
│   ├── model/
│   │   ├── network.py               GHGNet — MLP with BatchNorm + ReLU + Dropout
│   │   └── dataset.py               GHGDataset (PyTorch Dataset wrapper)
│   ├── train/
│   │   ├── trainer.py               train_model — AdamW + HuberLoss + early stopping
│   │   └── evaluator.py             evaluate_model, per-category metrics, worst-prediction report
│   ├── inference/
│   │   └── predict.py               load_model → LoadedModel; predict_ghg_with_loaded; predict_ghg
│   └── reporting/
│       └── plots.py                 training curves, scatter plot, residuals, diagnostics JSON
├── desktop_app/                     Windows desktop GUI  (see desktop_app/README.md)
│   ├── app.py                       entrypoint  →  python -m desktop_app.app
│   ├── inference_adapter.py         wraps model + vocab; single-call predict API for the UI
│   ├── updater.py                   auto-update: checks GitHub Releases, downloads installer
│   ├── splash.py                    loading screen shown while the model initialises
│   ├── _version.py                  app version string
│   ├── ui/
│   │   ├── main_window.py           2-column layout: product inputs left, results right
│   │   ├── category_panel.py        searchable c-PCR dropdown (frequency-ordered)
│   │   ├── product_card.py          collapsible per-product input card (comparison mode)
│   │   ├── materials_panel.py       dynamic material rows + autoscale to 100 %
│   │   ├── eol_panel.py             4 end-of-life sliders + autoscale to 100 %
│   │   ├── origin_panel.py          circular-origin % slider
│   │   ├── prediction_panel.py      large GHG readout + matplotlib gauge (single-product)
│   │   ├── comparison_panel.py      bar chart + summary table (2–4 products)
│   │   ├── widgets.py               SearchableDropdown, PercentSlider, SumIndicator
│   │   └── theme.py                 colour constants and theme helpers
│   ├── tools/
│   │   └── bake_assets.py           produces runtime assets from repo root model + dataset
│   ├── assets/                      runtime assets (not in git; produced by bake_assets.py)
│   └── build/
│       ├── ghg_predictor.spec       PyInstaller one-folder spec
│       ├── installer.iss            Inno Setup script (per-user, %LOCALAPPDATA%, no admin)
│       └── build.ps1                full build pipeline: bake → PyInstaller → Inno Setup
├── wiki-news-300d-1M/               fastText vectors (~1 GB, not in git, auto-downloaded)
├── GoogleNews-vectors-negative300/  W2V binary (optional alternative backend, not in git)
└── docs/
    ├── OVERVIEW.md                  complete technical reference
    └── LEARNINGS.md                 development diary
```

---

## Model architecture

**GHGNet** — a 3-layer MLP trained on standardised log1p-transformed targets:

```
Linear(input_dim, 256) → BatchNorm1d → ReLU → Dropout(0.2)
Linear(256, 128)        → BatchNorm1d → ReLU → Dropout(0.2)
Linear(128, 1)          → scalar output (standardised log1p GHG)
```

Inverse-transformed at inference: `expm1(pred * y_scale + y_mean)`.

Training details: AdamW (lr = 1e-3, weight_decay = 1e-4), HuberLoss (delta = 1.0), ReduceLROnPlateau, gradient clipping (max_norm = 1.0), early stopping (patience = 30, max 200 epochs, batch size 64). Full hyperparameter table in `src/config.py`.

---

## Setup (ML pipeline)

Requires Python 3.10+.

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Windows (cmd)

```cmd
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Running the ML pipeline

Place the dataset as `dataset.json` at the repository root (path is configurable in `src/config.py`).

With the virtual environment active:

```bash
python main.py
```

The first run downloads the fastText vectors (~1 GB extracted) into `wiki-news-300d-1M/`. To use a different embedding backend, set `EMBEDDING_BACKEND` in `src/config.py` to `"google_news"` or `"custom_vec"`.

### Outputs

After a run, the following files are written to the repo root:

| File | Contents |
|---|---|
| `ghg_model.pt` | Trained checkpoint (model weights + scaler params + category index) |
| `diagnostics_ghg.json` | Full metrics dump (training history + per-category test breakdown) |
| `figures/` | Training curves, predicted-vs-actual scatter, residual plot |

### Configuration

All paths and hyperparameters live in `src/config.py` (`EMBED_DIM`, `HIDDEN_DIMS`, `LR`, `EPOCHS`, `BATCH_SIZE`, `GHG_MIN`/`GHG_MAX`, `MIN_CATEGORY_COUNT`, embedding backend, etc.). Never hardcode these values elsewhere.

---

## Desktop application

The desktop application is a primary contribution of this repository. It packages the trained model into a standalone Windows installer — no Python, no configuration, no admin rights required — and provides a graphical interface for real-time GHG prediction and product comparison.

### For end users

Download `GHGPredictorSetup-<version>.exe` from the [Releases page](https://github.com/eirasroger/material-composition-gwp-predictor/releases), run it, and launch **GHG Predictor** from the Start Menu. The app checks for and installs updates automatically on each startup.

### Key features

- **Single-product prediction** — select a product category, define material components by name and mass fraction, set end-of-life shares (recycling, landfill, incineration, hazardous) and circular origin %, and read the predicted GHG footprint live. All percentage inputs autoscale to 100 % on demand.
- **Multi-product comparison** — configure 2–4 products side by side. The right panel switches to a bar chart and summary table comparing all products' predicted footprints.
- **Per-product category override** — in comparison mode, each product card can use its own category rather than the shared one.
- **Frequency-ordered material and category dropdowns** — the most common materials and categories from the training dataset appear first, reducing scrolling for typical use cases.
- **Live updates** — predictions recalculate within ~250 ms of any input change.

For developer setup, build instructions, and the full release workflow, see [`desktop_app/README.md`](desktop_app/README.md).

---

## Contact

Roger Vergés — Lead developer — [roger.verges.eiras@upc.edu](mailto:roger.verges.eiras@upc.edu)<a href="https://orcid.org/0009-0001-5887-4785" aria-label="ORCID"><img src="https://orcid.org/sites/default/files/images/orcid_16x16.png" alt="ORCID iD" width="16" height="16" style="vertical-align: text-bottom; margin-left: 4px;"></a>
