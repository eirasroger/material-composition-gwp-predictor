# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for the GHG Predictor desktop app.

One-folder mode (faster startup, easier to debug than one-file). Bundles the
baked assets (model + vocab + materials) and pulls in customtkinter's theme
files via collect_all.

Build from the repo root with:
    .venv\\Scripts\\pyinstaller.exe desktop_app\\build\\ghg_predictor.spec --clean
"""

from pathlib import Path

from PyInstaller.utils.hooks import collect_all


SPEC_DIR  = Path(SPECPATH).resolve()       # desktop_app/build
APP_DIR   = SPEC_DIR.parent                 # desktop_app/
REPO_ROOT = APP_DIR.parent                  # repo root

# customtkinter ships theme JSON / icons that PyInstaller doesn't auto-detect.
ctk_datas, ctk_binaries, ctk_hidden = collect_all("customtkinter")

# PyMuPDF (fitz) ships compiled extensions that need explicit collection.
fitz_datas, fitz_binaries, fitz_hidden = collect_all("fitz")

# Bundle the baked assets under "assets/" inside the frozen tree, so
# inference_adapter._default_assets_dir() and splash._assets_dir() find them
# via sys._MEIPASS / assets.
#
# Assets are DISCOVERED, not listed by name. A hand-maintained list has now
# shipped a broken .exe twice, both times silently: category_materials.json
# (docs/LEARNINGS.md 2026-05-09 -> alphabetical dropdowns) and
# distributions.json (2026-07-27 -> every summary radar drawn identical,
# because every reference lookup returned None). Runtime loaders treat these
# files as optional, so a missing one degrades instead of crashing, and the
# frozen build is the only place it can go missing. Discovery cannot forget.
_ASSETS_DIR = APP_DIR / "assets"

# Present in the source tree but deliberately not shipped.
_EXCLUDED_ASSETS = {
    "ghg_model.pt",  # legacy single-target checkpoint, superseded by models/
    "icon.png",      # source art for icon.ico; nothing reads it at runtime
}

# Floor, not the full set: everything else discovered ships too. Listed so a
# build against a stale/partial assets dir fails here rather than at the user's
# desk. Keep in sync with what bake_assets.py writes.
_REQUIRED_ASSETS = {
    "targets_manifest.json", "distributions.json", "vocab.npz",
    "materials.json", "category_materials.json",
    "icon.ico", "icon_vector.svg", "theme_dark.json",
}

# models/*.pt is globbed too: bake_assets.py bakes one checkpoint per trained
# target and the exact set changes as new targets are trained.
_model_checkpoints = sorted((_ASSETS_DIR / "models").glob("*.pt"))
if not _model_checkpoints:
    raise FileNotFoundError(
        f"No checkpoints in {_ASSETS_DIR / 'models'}. "
        "Run desktop_app/tools/bake_assets.py before building."
    )

_top_level = sorted(
    p for p in _ASSETS_DIR.iterdir()
    if p.is_file() and p.name not in _EXCLUDED_ASSETS
)
_missing = _REQUIRED_ASSETS - {p.name for p in _top_level}
if _missing:
    raise FileNotFoundError(
        f"Missing required asset(s) in {_ASSETS_DIR}: {sorted(_missing)}. "
        "Run desktop_app/tools/bake_assets.py before building."
    )

asset_files = [
    (str(p), "assets/models") for p in _model_checkpoints
] + [
    (str(p), "assets") for p in _top_level
]

a = Analysis(
    [str(APP_DIR / "app.py")],
    pathex=[str(REPO_ROOT)],
    binaries=ctk_binaries + fitz_binaries,
    datas=ctk_datas + fitz_datas + asset_files,
    hiddenimports=ctk_hidden + fitz_hidden + [
        "darkdetect",
        "matplotlib.backends.backend_tkagg",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Trim heavy ML packages we only need for training, not inference.
        "scikit-learn", "sklearn", "scipy", "pandas",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="GHGPredictor",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    icon=str(APP_DIR / "assets" / "icon.ico"),
    console=False,           # GUI app — no console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="GHGPredictor",
)
