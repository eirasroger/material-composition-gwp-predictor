# GHG Predictor — Desktop Application

A standalone Windows application for predicting and comparing the GHG footprint of construction products. End users install a single `GHGPredictorSetup-X.Y.Z.exe` — no Python, no configuration, no admin rights required.

---

## What it does

The app wraps the trained GHGNet model in an interactive graphical interface. You describe a product — what it is made of, how those materials are proportioned, what happens to it at end of life, and how much of it comes from recycled or circular sources — and the app displays a predicted GHG footprint in **kg CO₂-eq per kg of product**, updating live as you make changes. You can also configure multiple products at once and compare their footprints side by side.

---

## User guide

### Single-product mode

The default view is split into two columns.

**Left column — product inputs:**

- **Category** — select the product's c-PCR category from a searchable dropdown. Categories are sorted by frequency of occurrence in the training dataset, so the most common ones appear first.
- **Materials** — add one or more material components. Each row takes a material name (searchable and auto-completing from the training vocabulary) and a mass fraction percentage. Rows can be freely added or removed. The "Scale to 100 %" button proportionally normalises all fractions to sum to 100 %.
- **End-of-life shares** — four sliders for recycling, inert landfill, incineration, and hazardous waste. They must sum to 100 %; the "Scale to 100 %" button handles this automatically.
- **Circular origin** — a single slider for the percentage of material that comes from circular or recycled sources.

**Right column — prediction:**

The predicted GHG footprint is shown as a large number alongside a gauge chart. The prediction recalculates within ~250 ms of any input change. Inputs that do not sum to 100 % are flagged with a visible warning; the prediction still runs using the values as entered.

### Comparison mode

Click **+ Add product** to enter comparison mode (up to four products). The left column switches to a scrollable list of collapsible **product cards**, each with its own full set of inputs. A shared category applies to all products by default; each card has a toggle ("Different category") to override it with its own category.

The right column switches to a **comparison panel**: a grouped bar chart showing all products' predicted footprints side by side, plus a summary table listing the key inputs and prediction for each product.

Comparison mode exits automatically when all but one product card is removed.

### Auto-update

On each startup the app silently checks GitHub Releases for a newer version. If one is available, a dialog prompts you to install it. Accepting downloads the new installer and runs it automatically after the app closes — no manual download required.

---

## Installing (end users)

1. Go to the [Releases page](https://github.com/eirasroger/material-composition-gwp-predictor/releases).
2. Download `GHGPredictorSetup-<version>.exe`.
3. Run it — no admin rights or UAC prompt needed.
4. Launch **GHG Predictor** from the Start Menu.

To uninstall, use **Add or remove programs** in Windows Settings.

---

## Layout

```
desktop_app/
├── app.py                    entrypoint  →  python -m desktop_app.app
├── inference_adapter.py      loads model + vocab once; single-call predict API for the UI
├── updater.py                checks GitHub Releases, downloads and launches new installer
├── splash.py                 loading screen shown while the model initialises on startup
├── _version.py               app version string
├── ui/
│   ├── main_window.py        2-column layout + debounced prediction callback
│   ├── category_panel.py     searchable c-PCR dropdown (frequency-ordered)
│   ├── product_card.py       collapsible per-product input card used in comparison mode
│   ├── materials_panel.py    dynamic material rows + autoscale to 100 %
│   ├── eol_panel.py          4 end-of-life sliders + autoscale to 100 %
│   ├── origin_panel.py       circular-origin % slider
│   ├── prediction_panel.py   large GHG readout + matplotlib gauge (single-product mode)
│   ├── comparison_panel.py   bar chart + summary table (comparison mode, 2–4 products)
│   ├── widgets.py            SearchableDropdown, PercentSlider, SumIndicator
│   └── theme.py              colour constants and theme helpers
├── tools/
│   └── bake_assets.py        produces runtime assets from the repo root model + dataset
├── assets/                   runtime assets — NOT in git; produced by bake_assets.py
│   ├── ghg_model.pt          model checkpoint copy
│   ├── vocab.npz             compressed token embeddings (~1.3 MB)
│   ├── materials.json        frequency-sorted list of valid material names
│   └── category_materials.json  per-category material ordering for dropdown sorting
├── build/
│   ├── ghg_predictor.spec    PyInstaller one-folder spec
│   ├── installer.iss         Inno Setup script (per-user, %LOCALAPPDATA%, no admin)
│   └── build.ps1             full build pipeline: bake → PyInstaller → Inno Setup
└── requirements.txt          Python dependencies (CPU-only PyTorch for release builds)
```

---

## Running from source

Requires Python 3.10+ and an active virtual environment with the ML pipeline's `requirements.txt` already installed (from the repo root).

From the repo root:

```powershell
python -m pip install -r desktop_app\requirements.txt
python desktop_app\tools\bake_assets.py     # one-time; re-run after every model retrain
python -m desktop_app.app
```

`bake_assets.py` reads `dataset.json` and `ghg_model.pt` at the repo root and writes four files into `desktop_app/assets/`:

| File | Description |
|---|---|
| `ghg_model.pt` | Model checkpoint copy |
| `vocab.npz` | Compressed token embeddings covering only the materials in the dropdown (~1.3 MB vs ~1 GB for the full fastText file) |
| `materials.json` | Frequency-sorted list of valid material names for the dropdowns |
| `category_materials.json` | Per-category material ordering so dropdowns show the most relevant materials first for each category |

**Always re-run `bake_assets.py` after retraining the model.** Also add any new file placed in `assets/` to the `asset_files` list in `build/ghg_predictor.spec`, otherwise it will be missing from the frozen executable.

---

## Building the installer

Prerequisites:

- A virtual environment with `desktop_app/requirements.txt` installed.
- **CPU-only PyTorch** for release builds — keeps the final bundle under ~200 MB instead of ~700+ MB. The provided `requirements.txt` already pins the CPU index.
- [Inno Setup 6](https://jrsoftware.org/isinfo.php) installed on the build machine.

From the repo root:

```powershell
.\desktop_app\build\build.ps1 -Version 0.1.0
```

`build.ps1` runs three steps in sequence:

1. `python desktop_app\tools\bake_assets.py` — prepares runtime assets.
2. `pyinstaller desktop_app\build\ghg_predictor.spec` — bundles the app into `desktop_app\build\out\GHGPredictor\`.
3. `iscc desktop_app\build\installer.iss` — packages the bundle into `desktop_app\build\out\GHGPredictorSetup-<version>.exe`.

**Flags:**

| Flag | Effect |
|---|---|
| `-SkipBake` | Skip step 1 — use when assets are already up to date (CI uses this flag) |
| `-SkipInstaller` | Stop after step 2 — useful for quick debug runs of the frozen executable |

The installer is per-user (`%LOCALAPPDATA%\GHGPredictor`) and requires no admin rights or UAC prompt.

---

## Cutting a release

CI builds and publishes the installer automatically via [`.github/workflows/release.yml`](../.github/workflows/release.yml).

```powershell
# 1. Bake current assets and commit them:
python desktop_app\tools\bake_assets.py
git add desktop_app/assets ghg_model.pt
git commit -m "release: bake assets for vX.Y.Z"

# 2. Tag and push:
git tag vX.Y.Z
git push origin main --tags
```

Pushing a `v*.*.*` tag triggers the workflow on a `windows-latest` runner (Inno Setup is pre-installed). The workflow runs `build.ps1 -SkipBake`, uploads `GHGPredictorSetup-<version>.exe` to a new GitHub Release, and auto-generates release notes from commit messages.

Tags containing a hyphen (e.g. `v0.1.0-rc1`) are published as pre-releases.

To test the CI build without cutting a real release, use the **Run workflow** button on the Actions tab — this produces a downloadable artifact instead of publishing to Releases.

---

## Auto-update internals

`updater.py` runs a background daemon thread on startup:

1. Queries `api.github.com/repos/eirasroger/material-composition-gwp-predictor/releases/latest`.
2. If a newer `GHGPredictorSetup*.exe` asset is found, shows an update dialog.
3. On user acceptance: downloads the installer to a temporary directory, then spawns a detached PowerShell process that (1) waits for the app to close, (2) runs the Inno Setup installer silently, and (3) relaunches the new executable.

After one silent installation failure, the updater falls back to the interactive Inno Setup wizard. Update state is persisted in `update_state.json` (`pending_target`, `silent_failures`).

---

## Verification checklist

Before tagging a release:

- [ ] `python desktop_app\inference_adapter.py` — smoke test: adapter predictions match `predict_ghg()` to ≤ 1e-5 across 5 sample products.
- [ ] `python -m desktop_app.app` — moving any slider or changing any material updates the GHG display within ~250 ms; sums ≠ 100 % are shown as warnings, not crashes.
- [ ] **Comparison mode**: add 2–4 products, verify the bar chart and summary table render correctly; test the per-product category override toggle.
- [ ] `desktop_app\build\out\GHGPredictor\GHGPredictor.exe` — launches standalone from a clean shell with no Python on PATH.
- [ ] Install via `GHGPredictorSetup-*.exe` on a clean Windows 11 machine with no Python: launches from Start Menu, produces a prediction, uninstalls cleanly.
- [ ] Final `GHGPredictorSetup-*.exe` < 300 MB.
