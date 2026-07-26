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
# models/*.pt is globbed rather than listed by filename: bake_assets.py bakes
# one checkpoint per trained target (currently 5 GHG stages, more indicators
# later per docs/LEARNINGS.md 2026-07-25) and the exact set changes as new
# targets are trained. A fixed file list here would silently drop new targets
# from the frozen build -- see docs/LEARNINGS.md 2026-05-09 for the bug this
# caused before (a forgotten asset_files entry -> silent fallback at runtime).
_model_checkpoints = sorted((APP_DIR / "assets" / "models").glob("*.pt"))
if not _model_checkpoints:
    raise FileNotFoundError(
        f"No checkpoints in {APP_DIR / 'assets' / 'models'}. "
        "Run desktop_app/tools/bake_assets.py before building."
    )

asset_files = [
    (str(p), "assets/models") for p in _model_checkpoints
] + [
    (str(APP_DIR / "assets" / "targets_manifest.json"),    "assets"),
    (str(APP_DIR / "assets" / "vocab.npz"),                "assets"),
    (str(APP_DIR / "assets" / "materials.json"),           "assets"),
    (str(APP_DIR / "assets" / "category_materials.json"),  "assets"),
    (str(APP_DIR / "assets" / "icon.ico"),                  "assets"),
    (str(APP_DIR / "assets" / "icon_vector.svg"),           "assets"),
    (str(APP_DIR / "assets" / "theme_dark.json"),           "assets"),
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
