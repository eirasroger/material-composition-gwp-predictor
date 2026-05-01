# GHG Predictor — Desktop App

A self-contained Windows GUI for the trained GWP model. End users get a single
`GHGPredictorSetup-X.Y.Z.exe` installer (no Python, no extras). Developers
build the installer from this folder.

## Layout

```
desktop_app/
├── app.py                  entrypoint (python -m desktop_app.app)
├── inference_adapter.py    loads model + vocab once, wraps predict()
├── ui/
│   ├── main_window.py      2-column layout, debounced predict callback
│   ├── category_panel.py   searchable c-PCR dropdown
│   ├── materials_panel.py  dynamic material rows + autoscale
│   ├── eol_panel.py        4 end-of-life sliders + autoscale
│   ├── origin_panel.py     circular-origin slider
│   ├── prediction_panel.py large GHG number + matplotlib gauge
│   └── widgets.py          SearchableDropdown, PercentSlider, SumIndicator
├── tools/
│   └── bake_assets.py      produces assets/{ghg_model.pt,vocab.npz,materials.json}
├── assets/                 NOT in git — produced by bake_assets.py
├── build/
│   ├── ghg_predictor.spec  PyInstaller one-folder spec
│   ├── installer.iss       Inno Setup script
│   └── build.ps1           bake -> pyinstaller -> iscc
└── requirements.txt
```

## Running from source

From the repo root, with the project's venv active:

```powershell
python -m pip install -r desktop_app\requirements.txt
python desktop_app\tools\bake_assets.py     # one-time per model retrain
python -m desktop_app.app
```

`bake_assets.py` reads `dataset.json` + `ghg_model.pt` at the repo root and
writes `desktop_app/assets/{ghg_model.pt,vocab.npz,materials.json}`. The vocab
.npz is a subset of fastText covering only the tokens the dropdown materials
actually use (~1.3 MB instead of the full ~1 GB `.vec`).

## Building the installer

Prerequisites:

- A venv with `desktop_app/requirements.txt` installed.
- **CPU-only torch** for release builds — keeps the bundle ~200 MB instead of
  ~700+ MB. The provided requirements.txt already pins the CPU index.
- [Inno Setup 6](https://jrsoftware.org/isinfo.php) for the final installer.

Then, from the repo root:

```powershell
.\desktop_app\build\build.ps1 -Version 0.1.0
```

That script does three things in order:

1. `python desktop_app\tools\bake_assets.py`
2. `pyinstaller desktop_app\build\ghg_predictor.spec` →
   `desktop_app\build\out\GHGPredictor\` (one-folder bundle)
3. `iscc desktop_app\build\installer.iss` →
   `desktop_app\build\out\GHGPredictorSetup-<version>.exe`

Skip step 3 with `-SkipInstaller` if you only want the unpackaged dist tree
(useful for fast iteration / debug runs of the frozen exe).

The installer is per-user (`%LOCALAPPDATA%\GHGPredictor`) and requires no
admin rights or UAC.

## Cutting a release

CI builds and publishes the installer automatically — see
[`.github/workflows/release.yml`](../.github/workflows/release.yml).

To cut a release:

```powershell
# 1. Make sure desktop_app/assets/ is up to date with the current model:
python desktop_app\tools\bake_assets.py
git add desktop_app/assets ghg_model.pt
git commit -m "release: bake assets for v0.1.0"

# 2. Tag and push:
git tag v0.1.0
git push origin main --tags
```

Pushing a `v*.*.*` tag triggers the workflow. It runs `build.ps1 -SkipBake`
on a `windows-latest` runner (Inno Setup is preinstalled), uploads the
resulting `GHGPredictorSetup-<version>.exe` to a new GitHub Release named
after the tag, and auto-generates release notes from commit messages.

Tags with a hyphen (e.g. `v0.1.0-rc1`) are marked as pre-releases.

For smoke-testing the workflow without cutting a real release, use the
**Run workflow** button on the Actions tab — that produces a downloadable
artifact instead of publishing to Releases.

## Verification checklist

Before tagging a release:

- [ ] `python desktop_app\inference_adapter.py` — adapter smoke-test matches
      `predict_ghg(...)` to ≤ 1e-5 across 5 sample products.
- [ ] `python -m desktop_app.app` — moving any slider updates the GHG within
      ~250 ms; sums-≠-100 show as warnings, not failures.
- [ ] `desktop_app\build\out\GHGPredictor\GHGPredictor.exe` launches stand-alone
      (close it from a clean shell to confirm no extra deps).
- [ ] Installed via `GHGPredictorSetup-*.exe` on a clean Windows 11 machine
      with no Python: launches from Start Menu, produces a prediction,
      uninstalls cleanly.
- [ ] Final `GHGPredictorSetup-*.exe` < 300 MB.
