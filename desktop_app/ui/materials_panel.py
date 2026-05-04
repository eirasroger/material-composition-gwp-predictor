"""
Materials panel: a dynamic list of (material name, percentage) rows.

Each row owns a SearchableDropdown + a PercentSlider + a remove button. The
panel exposes ``materials()`` returning ``[{"name": str, "percentage": float}]``
for rows whose name is in the curated dropdown list, fires ``on_change`` on any
edit, and provides "Add material" / "Autoscale to 100%" buttons plus a live
SumIndicator.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import customtkinter as ctk

from desktop_app.ui.theme import BORDER, SURFACE, TEXT_SEC, font
from desktop_app.ui.widgets import PercentSlider, SearchableDropdown, SumIndicator
from src.utils import normalise_shares_to_100


class _MaterialRow(ctk.CTkFrame):
    def __init__(
        self,
        master,
        material_choices: List[str],
        on_change: Callable[[], None],
        on_remove: Callable[["_MaterialRow"], None],
    ) -> None:
        super().__init__(master, fg_color="transparent")
        self._on_change = on_change
        self._on_remove = on_remove
        self._material_choices = material_choices

        self.dropdown = SearchableDropdown(
            self, values=material_choices, command=lambda _v: self._on_change(),
            placeholder="Type to search material ...", width=260,
        )
        self.dropdown.pack(side="left", fill="x", expand=True, padx=(0, 8))

        self.slider = PercentSlider(
            self, label="", initial=0.0,
            command=lambda _v: self._on_change(),
            label_width=0,
        )
        self.slider.pack(side="left", fill="x", expand=True, padx=(0, 8))

        ctk.CTkButton(
            self, text="✕", width=28, height=28,
            fg_color="transparent",
            hover_color=BORDER,
            text_color=TEXT_SEC,
            font=font(12),
            command=lambda: self._on_remove(self),
        ).pack(side="left")

    def name(self) -> str:
        return self.dropdown.get()

    def is_valid_name(self) -> bool:
        return self.name() in self._material_choices

    def percentage(self) -> float:
        return self.slider.get()

    def set_percentage(self, value: float) -> None:
        self.slider.set(value)


class MaterialsPanel(ctk.CTkFrame):
    def __init__(
        self,
        master,
        material_choices: List[str],
        on_change: Optional[Callable[[List[Dict[str, float]]], None]] = None,
    ) -> None:
        super().__init__(master, fg_color=SURFACE, corner_radius=8)
        self._material_choices = sorted(material_choices, key=str.lower)
        self._on_change = on_change
        self._rows: List[_MaterialRow] = []

        ctk.CTkLabel(
            self, text="Material Composition",
            font=font(13, "bold"),
        ).pack(anchor="w", padx=14, pady=(14, 2))

        ctk.CTkLabel(
            self,
            text="Add materials and set their percentage share.",
            font=font(11),
            text_color=TEXT_SEC,
            anchor="w",
        ).pack(fill="x", padx=14, pady=(0, 8))

        self._rows_holder = ctk.CTkFrame(self, fg_color="transparent")
        self._rows_holder.pack(fill="x", padx=14, pady=(0, 6))

        controls = ctk.CTkFrame(self, fg_color="transparent")
        controls.pack(fill="x", padx=14, pady=(0, 6))

        ctk.CTkButton(
            controls, text="+ Add material", width=130, height=30,
            font=font(12, "bold"),
            command=self._add_row,
        ).pack(side="left")
        ctk.CTkButton(
            controls, text="Autoscale to 100%", width=150, height=30,
            font=font(12),
            fg_color="transparent",
            border_width=1,
            border_color=BORDER,
            text_color=TEXT_SEC,
            hover_color=BORDER,
            command=self._autoscale,
        ).pack(side="left", padx=(8, 0))

        self._sum_indicator = SumIndicator(self)
        self._sum_indicator.pack(fill="x", padx=14, pady=(2, 14))

        self._add_row()

    def materials(self) -> List[Dict[str, float]]:
        out: List[Dict[str, float]] = []
        for row in self._rows:
            if row.is_valid_name():
                out.append({"name": row.name(), "percentage": row.percentage()})
        return out

    def total(self) -> float:
        return sum(r.percentage() for r in self._rows)

    def _add_row(self) -> None:
        row = _MaterialRow(
            self._rows_holder,
            material_choices=self._material_choices,
            on_change=self._handle_change,
            on_remove=self._remove_row,
        )
        row.pack(fill="x", pady=2)
        self._rows.append(row)
        self._handle_change()

    def _remove_row(self, row: _MaterialRow) -> None:
        if row in self._rows:
            self._rows.remove(row)
        try:
            row.destroy()
        except Exception:
            pass
        if not self._rows:
            self._add_row()
            return
        self._handle_change()

    def _autoscale(self) -> None:
        values = [r.percentage() for r in self._rows]
        scaled = normalise_shares_to_100(values)
        for row, v in zip(self._rows, scaled):
            row.set_percentage(v)
        self._handle_change()

    def _handle_change(self) -> None:
        self._sum_indicator.update_total(self.total())
        if self._on_change is not None:
            self._on_change(self.materials())


def _smoke() -> None:
    from pathlib import Path
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme(
        str(Path(__file__).resolve().parents[1] / "assets" / "theme_dark.json")
    )
    root = ctk.CTk()
    root.title("materials_panel.py smoke test")
    root.geometry("720x540")

    def report(materials):
        print("materials =", materials)

    panel = MaterialsPanel(
        root,
        material_choices=[
            "Cement (CEM I)", "Cement (CEM II)", "Limestone filler",
            "Crushed stone", "Sand", "Water", "Steel rebar",
            "Polypropylene fibres", "Recycled aggregate", "Fly ash",
            "Slag (GGBFS)", "Superplasticiser",
        ],
        on_change=report,
    )
    panel.pack(fill="x", padx=12, pady=12)

    root.mainloop()


if __name__ == "__main__":
    _smoke()
