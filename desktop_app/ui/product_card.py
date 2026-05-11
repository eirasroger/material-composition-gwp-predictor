"""
Collapsible product card: one full set of input panels (materials, EoL, origin)
for a single product. Category is shared across cards by default; the
"Different category" switch reveals a local override.
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
        default_name: str = "Product",
    ) -> None:
        super().__init__(master, fg_color=SURFACE, corner_radius=10)
        self._color = color
        self._on_change = on_change
        self._expanded = False

        # ── header ────────────────────────────────────────────────────────────
        header = ctk.CTkFrame(self, fg_color=SURFACE_HI, corner_radius=8)
        header.pack(fill="x", padx=6, pady=(6, 0))

        dot = ctk.CTkFrame(header, width=12, height=12, corner_radius=6, fg_color=color)
        dot.pack(side="left", padx=(10, 8), pady=12)
        dot.pack_propagate(False)

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
            text="▶",
            width=28, height=28,
            fg_color="transparent",
            hover_color=BORDER,
            text_color=TEXT_SEC,
            font=font(11),
            command=self._toggle,
        )
        self._toggle_btn.pack(side="right", padx=(4, 4), pady=6)

        self._name_var = tk.StringVar(value=default_name)
        self._name_idle_after: str | None = None
        self._name_entry = ctk.CTkEntry(
            header,
            textvariable=self._name_var,
            fg_color=SURFACE_HI,
            border_width=0,
            font=font(13, "bold"),
            text_color=TEXT_SEC,
        )
        self._name_entry.pack(side="left", fill="x", expand=True, padx=(0, 4))
        self._name_var.trace_add("write", self._on_name_write)
        self._name_entry.bind("<FocusOut>", lambda e: self._fire_name_change())
        self._name_entry.bind("<Return>", lambda e: self._fire_name_change())

        # ── body ──────────────────────────────────────────────────────────────
        self._body = ctk.CTkFrame(self, fg_color="transparent")

        # Override section: switch + optional local CategoryPanel
        override_frame = ctk.CTkFrame(self._body, fg_color="transparent")
        override_frame.pack(fill="x")

        self._override_var = tk.BooleanVar(value=False)
        self._override_switch = ctk.CTkSwitch(
            override_frame,
            text="Different category",
            variable=self._override_var,
            command=self._on_override_toggle,
            font=font(11),
            text_color=TEXT_SEC,
        )
        self._override_switch.pack(anchor="w", padx=14, pady=(10, 4))

        self._local_category_panel = CategoryPanel(
            override_frame,
            categories=adapter.categories,
            on_change=self._on_local_category_change,
        )
        # Starts hidden; shown when override is toggled on

        self._materials_panel = MaterialsPanel(
            self._body,
            material_choices=adapter.materials,
            on_change=lambda _m: on_change(self),
            category_materials=adapter.category_materials,
        )
        self._materials_panel.pack(fill="x", pady=(4, 8))

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

        # Body starts hidden (all cards start collapsed)

    # ── public read interface ─────────────────────────────────────────────────

    def name(self) -> str:
        v = self._name_var.get().strip()
        return v if v else "Product"

    def has_category_override(self) -> bool:
        return self._override_var.get()

    def local_category(self) -> Optional[str]:
        return self._local_category_panel.selected()

    def apply_shared_category(self, category: Optional[str]) -> None:
        """Update materials ordering to match the shared category (no-op if overriding)."""
        if not self._override_var.get():
            self._materials_panel.set_category(category)

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

    def _on_name_write(self, *_) -> None:
        if self._name_idle_after is not None:
            try:
                self._name_entry.after_cancel(self._name_idle_after)
            except Exception:
                pass
        self._name_idle_after = self._name_entry.after(2000, self._fire_name_change)

    def _fire_name_change(self) -> None:
        if self._name_idle_after is not None:
            try:
                self._name_entry.after_cancel(self._name_idle_after)
            except Exception:
                pass
            self._name_idle_after = None
        self._on_change(self)

    def _on_override_toggle(self) -> None:
        if self._override_var.get():
            self._local_category_panel.pack(fill="x", pady=(0, 4))
        else:
            self._local_category_panel.pack_forget()
        self._on_change(self)

    def _on_local_category_change(self, category: Optional[str]) -> None:
        self._materials_panel.set_category(category)
        self._on_change(self)
