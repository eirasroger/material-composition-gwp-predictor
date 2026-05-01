"""
End-of-life panel: 4 fixed PercentSliders (Recycling, Hazardous, Inert,
Incineration), a live SumIndicator, and an Autoscale-to-100 button. Emits an
``EolShares`` snapshot via ``on_change``.
"""

from __future__ import annotations

from typing import Callable, Optional

import customtkinter as ctk

from desktop_app.inference_adapter import EolShares
from desktop_app.ui.widgets import PercentSlider, SumIndicator
from src.utils import normalise_shares_to_100


class EolPanel(ctk.CTkFrame):
    LABELS = (
        ("recycling",    "Recycling family*"),
        ("hazardous",    "Hazardous waste"),
        ("inert",        "Inert / non-haz landfill"),
        ("incineration", "Incineration"),
    )

    def __init__(
        self,
        master,
        on_change: Optional[Callable[[EolShares], None]] = None,
    ) -> None:
        super().__init__(master)
        self._on_change = on_change

        ctk.CTkLabel(
            self, text="End-of-life shares (%)", font=ctk.CTkFont(weight="bold"),
        ).pack(anchor="w", padx=10, pady=(10, 0))

        self._sliders = {}
        for key, label in self.LABELS:
            ps = PercentSlider(
                self, label=label, initial=0.0,
                command=lambda _v: self._handle_change(),
                label_width=180,
            )
            ps.pack(fill="x", padx=10, pady=2)
            self._sliders[key] = ps

        ctk.CTkLabel(
            self,
            text="*recycling family includes composting, valorisation, "
                 "reconditioning and reuse.",
            text_color="gray", anchor="w", justify="left", wraplength=520,
        ).pack(fill="x", padx=10, pady=(2, 0))

        controls = ctk.CTkFrame(self, fg_color="transparent")
        controls.pack(fill="x", padx=10, pady=(4, 4))
        ctk.CTkButton(controls, text="Autoscale to 100%", width=140,
                      command=self._autoscale).pack(side="left")

        self._sum_indicator = SumIndicator(self)
        self._sum_indicator.pack(fill="x", padx=10, pady=(0, 10))
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
    ctk.set_appearance_mode("system")
    root = ctk.CTk()
    root.title("eol_panel.py smoke test")
    root.geometry("620x340")

    panel = EolPanel(root, on_change=lambda s: print("eol =", s))
    panel.pack(fill="x", padx=12, pady=12)

    root.mainloop()


if __name__ == "__main__":
    _smoke()
