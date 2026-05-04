"""
Circular-origin panel: a single 0-100 slider, independent of any sum-to-100
constraint (origin % is unrelated to EoL shares — see
``src/data/preprocessing.py:22``).
"""

from __future__ import annotations

from typing import Callable, Optional

import customtkinter as ctk

from desktop_app.ui.theme import SURFACE, TEXT_SEC, font
from desktop_app.ui.widgets import PercentSlider


class OriginPanel(ctk.CTkFrame):
    def __init__(
        self,
        master,
        on_change: Optional[Callable[[float], None]] = None,
    ) -> None:
        super().__init__(master, fg_color=SURFACE, corner_radius=8)
        self._on_change = on_change

        ctk.CTkLabel(
            self, text="Circular Origin",
            font=font(13, "bold"),
        ).pack(anchor="w", padx=14, pady=(14, 2))

        ctk.CTkLabel(
            self,
            text="Share of inputs sourced from recycled or reused content.",
            font=font(11),
            text_color=TEXT_SEC,
            anchor="w",
        ).pack(fill="x", padx=14, pady=(0, 8))

        self._slider = PercentSlider(
            self, label="Origin %", initial=0.0,
            command=self._handle_change, label_width=90,
        )
        self._slider.pack(fill="x", padx=14, pady=(0, 14))

    def value(self) -> float:
        return self._slider.get()

    def _handle_change(self, v: float) -> None:
        if self._on_change is not None:
            self._on_change(float(v))


def _smoke() -> None:
    from pathlib import Path
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme(
        str(Path(__file__).resolve().parents[1] / "assets" / "theme_dark.json")
    )
    root = ctk.CTk()
    root.title("origin_panel.py smoke test")
    root.geometry("520x200")

    panel = OriginPanel(root, on_change=lambda v: print(f"origin = {v:.1f}"))
    panel.pack(fill="x", padx=12, pady=12)

    root.mainloop()


if __name__ == "__main__":
    _smoke()
