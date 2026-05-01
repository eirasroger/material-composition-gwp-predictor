"""
Desktop GUI entrypoint.

Loads the trained model + baked vocab once via ``InferenceAdapter`` and shows
the main window. Run with::

    python -m desktop_app.app
"""

from __future__ import annotations

import sys
import traceback

import customtkinter as ctk

from desktop_app.inference_adapter import InferenceAdapter
from desktop_app.ui.main_window import MainWindow
from desktop_app.updater import check_for_updates

try:
    from desktop_app._version import __version__
except ImportError:
    __version__ = "0.0.0"


def main() -> int:
    ctk.set_appearance_mode("system")
    ctk.set_default_color_theme("blue")

    try:
        adapter = InferenceAdapter()
    except FileNotFoundError as exc:
        print(f"[ghg-predictor] {exc}", file=sys.stderr)
        print(
            "[ghg-predictor] Run `python desktop_app/tools/bake_assets.py` "
            "to produce the missing assets, then retry.",
            file=sys.stderr,
        )
        return 1
    except Exception:
        traceback.print_exc()
        return 2

    app = MainWindow(adapter)
    check_for_updates(app, __version__)
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
