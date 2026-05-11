"""
Collapsible product card: one full set of input panels (category, materials,
EoL, origin) for a single product in the comparison workflow.
"""

from __future__ import annotations

import tkinter as tk
from typing import Callable, Dict, List, Optional

import customtkinter as ctk

from desktop_app.inference_adapter import EolShares, InferenceAdapter
from desktop_app.ui.category_panel import CategoryPanel
from desktop_app.ui.eol_panel import EolPanel
from desktop_app.ui.materials_panel import MaterialsPanel
from desktop_app.ui.origin_panel import OriginPanel
from desktop_app.ui.theme import BORDER, SURFACE, SURFACE_HI, TEXT_DIM, TEXT_SEC, font


class ProductCard(ctk.CTkFrame):
    def __init__(
        self,
        master,
        adapter: InferenceAdapter,
        color: str,
        on_change: Callable[["ProductCard"], None],
        on_remove: Optional[Callable[["ProductCard"], None]] = None,
        start_expanded: bool = True,
        default_name: str = "Product",
    ) -> None:
        super().__init__(master, fg_color=SURFACE, corner_radius=10)
        self._color = color
        self._on_change = on_change
        self._expanded = start_expanded

        # ── header ────────────────────────────────────────────────────────────
        header = ctk.CTkFrame(self, fg_color=SURFACE_HI, corner_radius=8)
        header.pack(fill="x", padx=6, pady=(6, 0))

        dot = ctk.CTkFrame(header, width=12, height=12, corner_radius=6, fg_color=color)
        dot.pack(side="left", padx=(10, 8), pady=12)
        dot.pack_propagate(False)

        # pack right-side buttons before name so name fills remaining space
        if on_remove is not None:
            ctk.CTkButton(
                header,
                text="×",
                width=28, height=28,
                fg_color="transparent",
                hover_color="#5c1f1f",
                text_color=TEXT_DIM,
                font=font(16),
                command=lambda: on_remove(self),
            ).pack(side="right", padx=(0, 8), pady=6)

        self._toggle_btn = ctk.CTkButton(
            header,
            text="▼" if start_expanded else "▶",
            width=28, height=28,
            fg_color="transparent",
            hover_color=BORDER,
            text_color=TEXT_SEC,
            font=font(11),
            command=self._toggle,
        )
        self._toggle_btn.pack(side="right", padx=(4, 4), pady=6)

        self._name_var = tk.StringVar(value=default_name)
        ctk.CTkEntry(
            header,
            textvariable=self._name_var,
            fg_color=SURFACE_HI,
            border_width=0,
            font=font(13, "bold"),
            text_color=TEXT_SEC,
        ).pack(side="left", fill="x", expand=True, padx=(0, 4))
        self._name_var.trace_add("write", lambda *_: on_change(self))

        # ── body ──────────────────────────────────────────────────────────────
        self._body = ctk.CTkFrame(self, fg_color="transparent")

        self._category_panel = CategoryPanel(
            self._body,
            categories=adapter.categories,
            on_change=self._on_category_change,
        )
        self._category_panel.pack(fill="x", pady=(0, 8))

        self._materials_panel = MaterialsPanel(
            self._body,
            material_choices=adapter.materials,
            on_change=lambda _m: on_change(self),
            category_materials=adapter.category_materials,
        )
        self._materials_panel.pack(fill="x", pady=(0, 8))

        self._eol_panel = EolPanel(
            self._body,
            on_change=lambda _s: on_change(self),
        )
        self._eol_panel.pack(fill="x", pady=(0, 8))

        self._origin_panel = OriginPanel(
            self._body,
            on_change=lambda _v: on_change(self),
        )
        self._origin_panel.pack(fill="x")

        if start_expanded:
            self._body.pack(fill="x", padx=6, pady=(4, 6))

    # ── public read interface ─────────────────────────────────────────────────

    def name(self) -> str:
        v = self._name_var.get().strip()
        return v if v else "Product"

    def category(self) -> Optional[str]:
        return self._category_panel.selected()

    def materials(self) -> List[Dict]:
        return self._materials_panel.materials()

    def eol_shares(self) -> EolShares:
        return self._eol_panel.shares()

    def eol_total(self) -> float:
        return self._eol_panel.total()

    def origin_pct(self) -> float:
        return self._origin_panel.value()

    @property
    def color(self) -> str:
        return self._color

    # ── collapse / expand ─────────────────────────────────────────────────────

    def expand(self) -> None:
        if not self._expanded:
            self._toggle()

    def collapse(self) -> None:
        if self._expanded:
            self._toggle()

    def _toggle(self) -> None:
        self._expanded = not self._expanded
        if self._expanded:
            self._body.pack(fill="x", padx=6, pady=(4, 6))
            self._toggle_btn.configure(text="▼")
        else:
            self._body.pack_forget()
            self._toggle_btn.configure(text="▶")

    # ── internal wiring ───────────────────────────────────────────────────────

    def _on_category_change(self, category: Optional[str]) -> None:
        self._materials_panel.set_category(category)
        self._on_change(self)
