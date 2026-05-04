"""Product category selector — a labelled SearchableDropdown."""

from __future__ import annotations

from typing import Callable, List, Optional

import customtkinter as ctk

from desktop_app.ui.theme import SURFACE, TEXT_SEC, font
from desktop_app.ui.widgets import SearchableDropdown


class CategoryPanel(ctk.CTkFrame):
    def __init__(
        self,
        master,
        categories: List[str],
        on_change: Optional[Callable[[Optional[str]], None]] = None,
    ) -> None:
        super().__init__(master, fg_color=SURFACE, corner_radius=8)
        self._categories = sorted(categories, key=str.lower)
        self._on_change = on_change

        ctk.CTkLabel(
            self, text="Product Category",
            font=font(13, "bold"),
        ).pack(anchor="w", padx=14, pady=(14, 2))

        ctk.CTkLabel(
            self,
            text="Select the EPD product category.",
            font=font(11),
            text_color=TEXT_SEC,
            anchor="w",
        ).pack(fill="x", padx=14, pady=(0, 6))

        self._dropdown = SearchableDropdown(
            self, values=self._categories, command=self._handle_select,
            placeholder="Type to search ...",
        )
        self._dropdown.pack(fill="x", padx=14, pady=(0, 14))

    def selected(self) -> Optional[str]:
        v = self._dropdown.get()
        return v if v in self._categories else None

    def _handle_select(self, value: str) -> None:
        if self._on_change is not None:
            self._on_change(value if value in self._categories else None)


def _smoke() -> None:
    from pathlib import Path
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme(
        str(Path(__file__).resolve().parents[1] / "assets" / "theme_dark.json")
    )
    root = ctk.CTk()
    root.title("category_panel.py smoke test")
    root.geometry("520x200")

    panel = CategoryPanel(
        root,
        categories=[
            "N/A",
            "2019:14-c-PCR-003 Concrete and concrete elements (EN 16757) 1.0.0",
            "2019:14-c-PCR-014 Insulation materials 1.0.0",
            "Cement",
            "Coating",
            "Flooring",
        ],
        on_change=lambda v: print("category =", v),
    )
    panel.pack(fill="x", padx=12, pady=12)

    root.mainloop()


if __name__ == "__main__":
    _smoke()
