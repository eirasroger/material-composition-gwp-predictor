# material-composition-gwp-predictor
Deep learning model to predict Global Warming Potential (GWP) of construction products from material composition. Materials are vectorised using embeddings and weighted by their proportions to form product-level representations, combined with product category features for accurate environmental impact prediction.

## Input features
- Material composition: name, percentage -> embedding (300 dimensions)
- Product category (one-hot encoded). Probably c-PCR based
- Circularity (5 dimensions: circular origin, future use recycling, future use inert landfilling, future use incineration, future use hazardous waste)

## Output feature
- total ghg (1 dimension)

## Dataset
- aprox 8800 construction products. Not all of them have specific c-PCR

## Core process
The model learns to map product composition to GWP using material embeddings weighted by their proportions, with product category added as a contextual signal to refine the prediction. When a specific PCR is available, that context is used to narrow the category representation; when it is not, the model falls back to a generic `N/A` context so inference can still proceed, albeit with less specificity.

## Project layout
```
material-composition-gwp-predictor/
├── main.py                      # entrypoint -> src.pipeline.run()
├── requirements.txt
└── src/
    ├── config.py                # paths, hyperparameters, constants, seeds
    ├── utils.py                 # safe_float, r2_safe, tokenisers, ...
    ├── data/
    │   ├── loader.py            # load + reference-unit filter + category index
    │   ├── preprocessing.py     # per-product validation + circularity feats
    │   └── features.py          # build feature matrix
    ├── embeddings/
    │   ├── vocab.py             # w2v / fastText / custom .vec loaders
    │   └── encode.py            # material/product/category encoders
    ├── model/
    │   ├── dataset.py           # torch Dataset wrapper
    │   └── network.py           # GHGNet MLP
    ├── train/
    │   ├── trainer.py           # training loop + early stopping
    │   └── evaluator.py         # held-out + per-category eval
    ├── reporting/
    │   └── plots.py             # diagnostic plots + JSON dump
    ├── inference/
    │   └── predict.py           # single-product GHG prediction
    └── pipeline.py              # end-to-end orchestration
```

## Setup

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
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Running

Place the dataset as `dataset.json` at the repository root (the path is configurable in `src/config.py`).

With the venv activated:
```bash
python main.py
```

Or without activating the venv (Windows):
```powershell
.\.venv\Scripts\python.exe main.py
```

The first run will download the fastText vectors (~1 GB extracted) into `wiki-news-300d-1M/` unless `EMBEDDING_BACKEND` in `src/config.py` is changed to `"google_news"` or `"custom_vec"`.

### Outputs
After a run, the following files are written to the repo root:
- `ghg_model.pt` — trained checkpoint (state dict + scaler params + `cat_index`)
- `diagnostics_ghg.json` — full metrics dump (train history + per-category test metrics)
- `training_curves_ghg.png`, `pred_vs_actual_ghg.png`, `residuals_ghg.png` — plots

### Configuration
All paths and hyperparameters live in `src/config.py` (`EMBED_DIM`, `HIDDEN_DIMS`, `LR`, `EPOCHS`, `BATCH_SIZE`, `GHG_MIN`/`GHG_MAX`, `MIN_CATEGORY_COUNT`, embedding backend, etc.).

## Contact

Roger Vergés - Corresponding author and lead developer - [roger.verges.eiras@upc.edu](mailto:roger.verges.eiras@upc.edu)<a href="https://orcid.org/0009-0001-5887-4785" aria-label="ORCID"><img src="https://orcid.org/sites/default/files/images/orcid_16x16.png" alt="ORCID iD" width="16" height="16" style="vertical-align: text-bottom; margin-left: 4px;"></a>
