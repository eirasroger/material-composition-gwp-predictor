"""
Circular-origin panel: a single 0-100 slider, independent of any sum-to-100
constraint (origin % is unrelated to EoL shares — see
``src/data/preprocessing.py:22``).
"""

from __future__ import annotations

from typing import Callable, Optional

import customtkinter as ctk

from desktop_app.ui.widgets import PercentSlider


class OriginPanel(ctk.CTkFrame):
    def __init__(
        self,
        master,
        on_change: Optional[Callable[[float], None]] = None,
    ) -> None:
        super().__init__(master)
        self._on_change = on_change

        ctk.CTkLabel(
            self, text="Circular origin (%)", font=ctk.CTkFont(weight="bold"),
        ).pack(anchor="w", padx=10, pady=(10, 0))

        ctk.CTkLabel(
            self,
            text="Share of inputs sourced from recycled / reused content.",
            text_color="gray", anchor="w",
        ).pack(fill="x", padx=10, pady=(0, 4))

        self._slider = PercentSlider(
            self, label="Origin", initial=0.0,
            command=self._handle_change, label_width=80,
        )
        self._slider.pack(fill="x", padx=10, pady=(0, 10))

    def value(self) -> float:
        return self._slider.get()

    def _handle_change(self, v: float) -> None:
        if self._on_change is not None:
            self._on_change(float(v))


def _smoke() -> None:
    ctk.set_appearance_mode("system")
    root = ctk.CTk()
    root.title("origin_panel.py smoke test")
    root.geometry("520x180")

    panel = OriginPanel(root, on_change=lambda v: print(f"origin = {v:.1f}"))
    panel.pack(fill="x", padx=12, pady=12)

    root.mainloop()


if __name__ == "__main__":
    _smoke()
