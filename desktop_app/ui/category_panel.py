"""Product category selector — a labelled SearchableDropdown."""

from __future__ import annotations

from typing import Callable, List, Optional

import customtkinter as ctk

from desktop_app.ui.widgets import SearchableDropdown


class CategoryPanel(ctk.CTkFrame):
    def __init__(
        self,
        master,
        categories: List[str],
        on_change: Optional[Callable[[Optional[str]], None]] = None,
    ) -> None:
        super().__init__(master)
        self._categories = sorted(categories, key=str.lower)
        self._on_change = on_change

        ctk.CTkLabel(
            self, text="Product category", font=ctk.CTkFont(weight="bold"),
        ).pack(anchor="w", padx=10, pady=(10, 0))

        self._dropdown = SearchableDropdown(
            self, values=self._categories, command=self._handle_select,
            placeholder="Type to search ...",
        )
        self._dropdown.pack(fill="x", padx=10, pady=(4, 10))

    def selected(self) -> Optional[str]:
        v = self._dropdown.get()
        return v if v in self._categories else None

    def _handle_select(self, value: str) -> None:
        if self._on_change is not None:
            self._on_change(value if value in self._categories else None)


def _smoke() -> None:
    ctk.set_appearance_mode("system")
    root = ctk.CTk()
    root.title("category_panel.py smoke test")
    root.geometry("520x180")

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
