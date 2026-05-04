"""
End-of-life panel: 4 fixed PercentSliders (Recycling, Hazardous, Inert,
Incineration), a live SumIndicator, and an Autoscale-to-100 button. Emits an
``EolShares`` snapshot via ``on_change``.
"""

from __future__ import annotations

from typing import Callable, Optional

import customtkinter as ctk

from desktop_app.inference_adapter import EolShares
from desktop_app.ui.theme import BORDER, SURFACE, TEXT_DIM, TEXT_SEC, font
from desktop_app.ui.widgets import PercentSlider, SumIndicator
from src.utils import normalise_shares_to_100


class EolPanel(ctk.CTkFrame):
    LABELS = (
        ("recycling",    "Recycling *"),
        ("hazardous",    "Hazardous waste"),
        ("inert",        "Inert landfilling"),
        ("incineration", "Incineration"),
    )

    def __init__(
        self,
        master,
        on_change: Optional[Callable[[EolShares], None]] = None,
    ) -> None:
        super().__init__(master, fg_color=SURFACE, corner_radius=8)
        self._on_change = on_change

        ctk.CTkLabel(
            self, text="End-of-life pathway",
            font=font(13, "bold"),
        ).pack(anchor="w", padx=14, pady=(14, 2))

        ctk.CTkLabel(
            self,
            text="Percentage of product mass per end-of-life scenario.",
            font=font(11),
            text_color=TEXT_SEC,
            anchor="w",
        ).pack(fill="x", padx=14, pady=(0, 8))

        self._sliders = {}
        for key, label in self.LABELS:
            ps = PercentSlider(
                self, label=label, initial=0.0,
                command=lambda _v: self._handle_change(),
                label_width=190,
            )
            ps.pack(fill="x", padx=14, pady=2)
            self._sliders[key] = ps

        ctk.CTkLabel(
            self,
            text="* Recycling includes composting, valorisation, reconditioning and reuse.",
            font=font(10),
            text_color=TEXT_DIM,
            anchor="w",
            justify="left",
            wraplength=480,
        ).pack(fill="x", padx=14, pady=(4, 0))

        controls = ctk.CTkFrame(self, fg_color="transparent")
        controls.pack(fill="x", padx=14, pady=(8, 6))
        ctk.CTkButton(
            controls, text="Autoscale to 100%", width=150, height=30,
            font=font(12),
            fg_color="transparent",
            border_width=1,
            border_color=BORDER,
            text_color=TEXT_SEC,
            hover_color=BORDER,
            command=self._autoscale,
        ).pack(side="left")

        self._sum_indicator = SumIndicator(self)
        self._sum_indicator.pack(fill="x", padx=14, pady=(2, 14))
        self._sum_indicator.update_total(self.total())

    def shares(self) -> EolShares:
        return EolShares(
            recycling   =self._sliders["recycling"].get(),
            hazardous   =self._sliders["hazardous"].get(),
            inert       =self._sliders["inert"].get(),
            incineration=self._sliders["incineration"].get(),
        )

    def total(self) -> float:
        return sum(s.get() for s in self._sliders.values())

    def _autoscale(self) -> None:
        keys = list(self._sliders.keys())
        values = [self._sliders[k].get() for k in keys]
        scaled = normalise_shares_to_100(values)
        for k, v in zip(keys, scaled):
            self._sliders[k].set(v)
        self._handle_change()

    def _handle_change(self) -> None:
        self._sum_indicator.update_total(self.total())
        if self._on_change is not None:
            self._on_change(self.shares())


def _smoke() -> None:
    from pathlib import Path
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme(
        str(Path(__file__).resolve().parents[1] / "assets" / "theme_dark.json")
    )
    root = ctk.CTk()
    root.title("eol_panel.py smoke test")
    root.geometry("620x360")

    panel = EolPanel(root, on_change=lambda s: print("eol =", s))
    panel.pack(fill="x", padx=12, pady=12)

    root.mainloop()


if __name__ == "__main__":
    _smoke()
