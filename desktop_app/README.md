# Desktop application

A standalone Windows application for predicting and comparing the environmental impact of construction products. End users install a single `GHGPredictorSetup-X.Y.Z.exe` — no Python, no configuration, no admin rights.

---

## What it does

You describe a product — what it is made of, how those materials are proportioned, what happens to it at end of life, and how much comes from recycled or circular sources — and the app predicts 25 targets per kg of product: 5 indicators across 5 lifecycle stages, updating live as you type.

| Indicator | Unit |
|---|---|
| Greenhouse gas emissions | kg CO₂-eq/kg |
| Water depletion potential | m³/kg |
| Eutrophication potential | kg PO₄-eq/kg |
| Acidification potential | kg SO₂-eq/kg |
| Abiotic depletion potential: fossil | MJ/kg |

All 25 are computed on every edit in ~5 ms, so changing which indicator you look at is an instant redraw.

---

## User guide

### Inputs

- **Category** — searchable dropdown, ordered by frequency in the training data. Formal PCR categories are listed first; `Other: X` entries are standardized categories for products with no formal PCR.
- **Materials** — one row per component: a searchable material name plus a mass fraction. Rows can be added and removed freely, and **Autoscale to 100 %** normalises them proportionally. The material list reorders itself to the selected category's most common materials.
- **End-of-life pathway** — four sliders: recycling, hazardous waste, inert landfilling, incineration, with their own **Autoscale to 100 %**. Recycling covers composting, valorisation, reconditioning and reuse.
- **Circular origin** — share of inputs from recycled or reused sources, independent of the end-of-life shares.

Inputs that do not sum to 100 % are flagged as warnings; the prediction still runs, normalised.

### Results

A **Showing** dropdown selects the view:

| Selection | 1 product | 2–4 products |
|---|---|---|
| Summary (default) | One large radar of all five indicators | Overlaid radars, or small multiples when categories differ |
| A single indicator | Large readout, gauge and lifecycle breakdown | Bar chart with aligned summary table |

**How impact is contextualised.** Results are expressed as a ratio to the median of real products in the same category. The radar's radial axis and the gauge both use `log₁₀(value / category median)` over one decade each side, with the p25–p75 band shaded to show how much composition moves that indicator within that category. Categories with fewer than 30 labelled products fall back to the global distribution, labelled as such.

**Verdicts** ("better / typical / worse than typical") are cut at the same p25/p75 the chart shades, so the wording always agrees with the picture. When the prediction's own error range straddles a threshold, the wording softens to "likely", and full-strength colour indicates the model can settle the claim.

### Comparison

**+ Add product** adds up to four collapsible product cards, each with a full set of inputs. A shared category applies to all cards; each card has a **Different category** switch to override it.

### Scenarios

The scenario bar saves the whole left column — shared category plus every product — under a name:

- **Save as…** stores a new scenario; **Save** overwrites the current one in place.
- **My scenarios** browses saved scenarios with search, rename, duplicate and delete.
- Opening a scenario repopulates every card and recalculates immediately.
- A `*` beside the name marks unsaved changes. Closing the app or opening another scenario with unsaved work prompts Save / Discard / Cancel.
- The last session is autosaved and restored on the next launch, with or without a named scenario.

Scenarios store inputs only; results are recomputed on load, so a retrained model is always reflected. Files live in `%APPDATA%\GHGPredictor\scenarios\`, one JSON each, outside the install directory so updates and uninstalls leave them untouched.

### Auto-update

On startup the app checks GitHub Releases in the background. If a newer version exists, a dialog offers to install it; accepting downloads the installer and runs it after the app closes.

---

## Installing (end users)

1. Go to the [Releases page](https://github.com/eirasroger/material-composition-gwp-predictor/releases).
2. Download `GHGPredictorSetup-<version>.exe`.
3. Run it — no admin rights or UAC prompt needed.
4. Launch from the Start Menu.

To uninstall, use **Add or remove programs**. Saved scenarios in `%APPDATA%\GHGPredictor` are left in place.

---

## Layout

```
desktop_app/
├── app.py                    entrypoint  →  python -m desktop_app.app
├── inference_adapter.py      loads all models + vocab once; single-call predict API
├── library.py                scenario storage (%APPDATA%/GHGPredictor)
├── updater.py                checks GitHub Releases, downloads and launches new installer
├── splash.py                 loading screen shown while the models initialise
├── _version.py               app version string
├── tests/
│   └── test_scenario_roundtrip.py   scenario save/load round trip
├── ui/
│   ├── main_window.py        scenario bar + 2-column layout + debounced prediction
│   ├── category_panel.py     searchable category dropdown
│   ├── product_card.py       collapsible per-product input card
│   ├── materials_panel.py    dynamic material rows + autoscale to 100 %
│   ├── eol_panel.py          4 end-of-life sliders + autoscale to 100 %
│   ├── origin_panel.py       circular-origin % slider
│   ├── prediction_panel.py   readout + gauge + lifecycle breakdown, per target
│   ├── summary_panel.py      radar of every indicator against the category median
│   ├── comparison_panel.py   bar chart + aligned summary table (2–4 products)
│   ├── scenario_dialog.py    library browser, name prompt, save/discard prompts
│   ├── widgets.py            SearchableDropdown, PercentSlider, SumIndicator
│   └── theme.py              colour constants, formatting and status helpers
├── tools/
│   └── bake_assets.py        produces runtime assets from the repo root models + dataset
├── assets/                   runtime assets — NOT in git; produced by bake_assets.py
│   ├── models/*.pt           one checkpoint per target (25)
│   ├── targets_manifest.json target key → indicator, stage, display name, unit
│   ├── distributions.json    per-target p25/median/p75/n, per category and global
│   ├── vocab.npz             compressed token embeddings (~1.3 MB)
│   ├── materials.json        frequency-sorted list of valid material names
│   └── category_materials.json  per-category material ordering
├── build/
│   ├── ghg_predictor.spec    PyInstaller one-folder spec
│   ├── installer.iss         Inno Setup script (per-user, %LOCALAPPDATA%, no admin)
│   └── build.ps1             full build pipeline: bake → PyInstaller → Inno Setup
└── requirements.txt          Python dependencies (CPU-only PyTorch for release builds)
```

---

## Running from source

Requires Python 3.10+ and the ML pipeline's `requirements.txt` already installed. From the repo root:

```powershell
python -m pip install -r desktop_app\requirements.txt
python desktop_app\tools\bake_assets.py     # re-run after every model retrain
python -m desktop_app.app
```

`bake_assets.py` reads `dataset.json` and `models/*.pt` at the repo root and writes the six asset entries listed above.

**Two standing rules:**

- Re-run `bake_assets.py` after retraining any target; the app loads whatever is in `assets/models/`.
- Any new file read at runtime from `assets/` must also be added to `asset_files` in `build/ghg_predictor.spec`, or it will work from source and silently fall back in the frozen executable. Model checkpoints are globbed, so new targets are picked up automatically.

---

## Building the installer

Prerequisites:

- A virtual environment with `desktop_app/requirements.txt` installed.
- **CPU-only PyTorch** for release builds, which keeps the bundle under 300 MB. The provided `requirements.txt` already pins the CPU index.
- [Inno Setup 6](https://jrsoftware.org/isinfo.php) on the build machine.

```powershell
.\desktop_app\build\build.ps1 -Version 0.1.0
```

Three steps in sequence:

1. `bake_assets.py` — prepares runtime assets.
2. `pyinstaller ghg_predictor.spec` — bundles into `desktop_app\build\out\GHGPredictor\`.
3. `iscc installer.iss` — packages into `desktop_app\build\out\GHGPredictorSetup-<version>.exe`.

| Flag | Effect |
|---|---|
| `-SkipBake` | Skip step 1 when assets are current (CI uses this) |
| `-SkipInstaller` | Stop after step 2, for quick debug runs of the frozen executable |

The installer is per-user (`%LOCALAPPDATA%\GHGPredictor`), no admin rights or UAC prompt.

---

## Cutting a release

CI builds and publishes via [`.github/workflows/release.yml`](../.github/workflows/release.yml).

```powershell
# 1. Bake current assets and commit them:
python desktop_app\tools\bake_assets.py
git add desktop_app/assets models
git commit -m "release: bake assets for vX.Y.Z"

# 2. Tag and push:
git tag vX.Y.Z
git push origin main --tags
```

Pushing a `v*.*.*` tag triggers the workflow on `windows-latest` (Inno Setup pre-installed), which runs `build.ps1 -SkipBake`, uploads the installer to a new GitHub Release, and generates release notes from commit messages. Tags containing a hyphen (e.g. `v0.1.0-rc1`) publish as pre-releases.

To test CI without cutting a release, use **Run workflow** on the Actions tab, which produces a downloadable artifact.

---

## Auto-update internals

`updater.py` runs a background daemon thread on startup:

1. Queries `api.github.com/repos/eirasroger/material-composition-gwp-predictor/releases/latest`.
2. If a newer `GHGPredictorSetup*.exe` asset is found, shows an update dialog.
3. On acceptance: downloads the installer to a temp directory, then spawns a PowerShell process that waits for the app to close, runs the installer silently, and relaunches the new executable.

After one silent failure it falls back to the interactive wizard. State lives in `update_state.json` (`pending_target`, `silent_failures`).

Four identifiers are load-bearing for upgrades and must stay in sync if any of them changes: the `GHGPredictorSetup` release-asset prefix (`updater.py`), `AppId` and `DefaultDirName` (`installer.iss`), and the `GHGPredictor.exe` name (`installer.iss` and `ghg_predictor.spec`).

---

## Verification checklist

Before tagging a release:

- [ ] `python desktop_app\inference_adapter.py` — adapter matches `predict_ghg()` to ≤ 1e-5 across 5 samples.
- [ ] `python -m desktop_app.tests.test_scenario_roundtrip` — scenario save/load, against a temp `APPDATA`.
- [ ] `python -m desktop_app.app` — any input change updates the display within ~250 ms; sums ≠ 100 % show warnings.
- [ ] **Indicators**: step through every entry in the Showing dropdown; check the radar, gauge and breakdown render, and that no axis is pinned flat for the small-valued indicators.
- [ ] **Comparison**: add 2–4 products, verify the bar chart, summary table and per-product category override.
- [ ] **Scenarios**: save, reopen, rename, delete; confirm a per-product category override survives a round trip and that the unsaved-changes prompt appears on close.
- [ ] `desktop_app\build\out\GHGPredictor\GHGPredictor.exe` — launches standalone with no Python on PATH.
- [ ] Install `GHGPredictorSetup-*.exe` on a clean Windows 11 machine: launches from the Start Menu, produces a prediction, uninstalls cleanly.
- [ ] Final installer < 300 MB.
