"""
Main window: multi-product comparison layout.

Left column
  ├── Shared CategoryPanel (applies to all cards unless overridden)
  ├── ProductCard 1 … 4  (collapsible; each can override category)
  └── "+ Add product" button

Right column
  Single product → PredictionPanel (unchanged single-product view)
  2+ products    → ComparisonPanel (bar chart + aligned summary table)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

import customtkinter as ctk

from desktop_app.inference_adapter import EolShares, InferenceAdapter
from desktop_app.ui.category_panel import CategoryPanel
from desktop_app.ui.comparison_panel import ComparisonPanel, ProductResult
from desktop_app.ui.prediction_panel import PredictionPanel
from desktop_app.ui.product_card import ProductCard
from desktop_app.ui.theme import (
    ACCENT, BG, BORDER, MAX_PRODUCTS, PRODUCT_COLORS, SURFACE, TEXT_SEC, font,
)
from desktop_app._version import __version__
from src.utils import normalise_shares_to_100


DEBOUNCE_MS = 150


def _icon_path() -> Path:
    base = (
        Path(getattr(sys, "_MEIPASS", "")) / "assets"
        if getattr(sys, "frozen", False)
        else Path(__file__).resolve().parents[1] / "assets"
    )
    return base / "icon.ico"


class MainWindow(ctk.CTk):
    def __init__(self, adapter: InferenceAdapter) -> None:
        super().__init__()
        self.adapter = adapter

        self.title(f"GHG Predictor (version {__version__})")
        self.geometry("1300x820")
        self.minsize(980, 700)
        self.configure(fg_color=BG)

        icon = _icon_path()
        if icon.exists():
            try:
                self.iconbitmap(str(icon))
            except Exception:
                pass

        # shared category state
        self._shared_category: Optional[str] = None

        # per-card state
        self._cards: list[ProductCard] = []
        self._predictions: dict[int, dict | None] = {}     # id(card) → snapshot | None
        self._statuses: dict[int, str] = {}                # id(card) → warning text
        self._pending_after: dict[int, str | None] = {}    # id(card) → after-id | None
        self._used_color_indices: set[int] = set()
        self._card_color_idx: dict[int, int] = {}          # id(card) → PRODUCT_COLORS index
        self._active_right_panel = None

        self.grid_columnconfigure(0, weight=1, uniform="cols")
        self.grid_columnconfigure(1, weight=1, uniform="cols")
        self.grid_rowconfigure(0, weight=1)

        # ── left column ───────────────────────────────────────────────────────
        self._left = ctk.CTkScrollableFrame(
            self,
            label_text="",
            label_fg_color=BG,
            fg_color=BG,
            scrollbar_button_color=BORDER,
            scrollbar_button_hover_color=ACCENT,
        )
        self._left.grid(row=0, column=0, sticky="nsew", padx=(12, 6), pady=12)

        # Shared category panel — above all product cards
        self._shared_cat_panel = CategoryPanel(
            self._left,
            categories=self.adapter.categories,
            on_change=self._on_shared_category_change,
        )
        self._shared_cat_panel.pack(fill="x", pady=(0, 10))

        # Add-product button (packed last; cards are inserted before it)
        self._add_btn = ctk.CTkButton(
            self._left,
            text="+ Add product",
            height=36,
            font=font(12, "bold"),
            fg_color="transparent",
            border_width=1,
            border_color=BORDER,
            text_color=TEXT_SEC,
            hover_color=BORDER,
            command=self._add_product,
        )

        # ── right column ──────────────────────────────────────────────────────
        self._right_container = ctk.CTkFrame(self, fg_color=SURFACE, corner_radius=10)
        self._right_container.grid(row=0, column=1, sticky="nsew", padx=(6, 12), pady=12)
        self._right_container.grid_rowconfigure(0, weight=1)
        self._right_container.grid_columnconfigure(0, weight=1)

        # ── initial product (permanent, cannot be removed) ────────────────────
        card1 = self._create_card(removable=False)
        card1.pack(fill="x", pady=(0, 8))
        self._add_btn.pack(fill="x", pady=(0, 4))

        self._rebuild_right_panel()
        self._schedule_predict(card1)

    # ── shared category ───────────────────────────────────────────────────────

    def _on_shared_category_change(self, category: Optional[str]) -> None:
        self._shared_category = category
        for card in self._cards:
            card.apply_shared_category(category)
            if not card.has_category_override():
                self._schedule_predict(card)

    def _effective_category(self, card: ProductCard) -> Optional[str]:
        if card.has_category_override():
            return card.local_category()
        return self._shared_category

    # ── card creation / removal ───────────────────────────────────────────────

    def _next_color_index(self) -> int:
        for i in range(MAX_PRODUCTS):
            if i not in self._used_color_indices:
                return i
        raise RuntimeError("Max products reached")

    def _create_card(self, removable: bool = True) -> ProductCard:
        color_idx = self._next_color_index()
        self._used_color_indices.add(color_idx)
        n = len(self._cards) + 1
        card = ProductCard(
            self._left,
            adapter=self.adapter,
            color=PRODUCT_COLORS[color_idx],
            on_change=self._on_card_change,
            on_remove=self._on_card_remove if removable else None,
            default_name=f"Product {n}",
        )
        self._cards.append(card)
        self._predictions[id(card)] = None
        self._statuses[id(card)] = ""
        self._pending_after[id(card)] = None
        self._card_color_idx[id(card)] = color_idx
        return card

    def _add_product(self) -> None:
        if len(self._cards) >= MAX_PRODUCTS:
            return
        self._add_btn.pack_forget()
        card = self._create_card(removable=True)
        card.apply_shared_category(self._shared_category)
        card.pack(fill="x", pady=(0, 8))
        self._add_btn.pack(fill="x", pady=(0, 4))
        self._add_btn.configure(
            state="normal" if len(self._cards) < MAX_PRODUCTS else "disabled"
        )
        self._rebuild_right_panel()
        self._schedule_predict(card)

    def _on_card_remove(self, card: ProductCard) -> None:
        if card not in self._cards:
            return
        pending = self._pending_after.pop(id(card), None)
        if pending:
            try:
                self.after_cancel(pending)
            except Exception:
                pass
        color_idx = self._card_color_idx.pop(id(card), None)
        if color_idx is not None:
            self._used_color_indices.discard(color_idx)
        self._predictions.pop(id(card), None)
        self._statuses.pop(id(card), None)
        self._cards.remove(card)
        card.destroy()
        self._add_btn.configure(state="normal")
        self._rebuild_right_panel()
        self._push_all_predictions()

    # ── right panel management ────────────────────────────────────────────────

    def _rebuild_right_panel(self) -> None:
        if self._active_right_panel is not None:
            self._active_right_panel.destroy()
            self._active_right_panel = None

        if len(self._cards) <= 1:
            panel = PredictionPanel(self._right_container, color=PRODUCT_COLORS[0])
        else:
            panel = ComparisonPanel(self._right_container)

        panel.grid(row=0, column=0, sticky="nsew")
        self._active_right_panel = panel

    def _push_all_predictions(self) -> None:
        if self._active_right_panel is None or not self._cards:
            return

        if len(self._cards) == 1:
            card = self._cards[0]
            snap = self._predictions.get(id(card))
            status = self._statuses.get(id(card), "")
            if snap is None:
                self._active_right_panel.clear_prediction()
                self._active_right_panel.set_status(status)
            else:
                self._active_right_panel.set_prediction(snap["value"], snap["bounds"])
                self._active_right_panel.set_status(status)
        else:
            results: List[ProductResult] = []
            for card in self._cards:
                snap = self._predictions.get(id(card))
                if snap is not None:
                    results.append(ProductResult(
                        name=card.name(),
                        value=snap["value"],
                        bounds=snap["bounds"],
                        color=card.color,
                        category=snap["category"],
                        materials=snap["materials"],
                        eol=snap["eol"],
                        origin_pct=snap["origin_pct"],
                    ))
            self._active_right_panel.update(results)

    # ── prediction wiring ─────────────────────────────────────────────────────

    def _on_card_change(self, card: ProductCard) -> None:
        self._schedule_predict(card)

    def _schedule_predict(self, card: ProductCard) -> None:
        pending = self._pending_after.get(id(card))
        if pending is not None:
            try:
                self.after_cancel(pending)
            except Exception:
                pass
        self._pending_after[id(card)] = self.after(
            DEBOUNCE_MS, lambda c=card: self._predict_now(c)
        )

    def _predict_now(self, card: ProductCard) -> None:
        self._pending_after[id(card)] = None

        category   = self._effective_category(card)
        materials  = card.materials()
        eol_shares = card.eol_shares()
        origin_pct = card.origin_pct()

        if category is None:
            self._predictions[id(card)] = None
            self._statuses[id(card)] = "Pick a product category to begin."
            self._push_all_predictions()
            return
        if not materials:
            self._predictions[id(card)] = None
            self._statuses[id(card)] = "Add at least one material."
            self._push_all_predictions()
            return

        status_parts: list[str] = []
        mat_total = sum(m["percentage"] for m in materials)
        if abs(mat_total - 100.0) > 0.05:
            status_parts.append(
                f"Materials sum to {mat_total:.1f}% — predicted as if normalised."
            )
        eol_total = (
            eol_shares.recycling + eol_shares.hazardous
            + eol_shares.inert + eol_shares.incineration
        )
        if abs(eol_total - 100.0) > 0.05:
            status_parts.append(
                f"End-of-life pathways sum to {eol_total:.1f}% — predicted as if normalised."
            )

        eol_for_pred = self._normalised_eol(eol_shares)

        try:
            value = self.adapter.predict(
                category=category,
                materials=materials,
                eol=eol_for_pred,
                origin_pct=origin_pct,
            )
        except Exception as exc:
            self._predictions[id(card)] = None
            self._statuses[id(card)] = f"Prediction failed: {exc}"
            self._push_all_predictions()
            return

        bounds = self.adapter.prediction_range(value, category)

        # Normalised materials for display (what the model actually used)
        if mat_total > 0:
            norm_materials = [
                {"name": m["name"], "percentage": m["percentage"] / mat_total * 100.0}
                for m in materials
            ]
        else:
            norm_materials = list(materials)

        self._predictions[id(card)] = {
            "value":      value,
            "bounds":     bounds,
            "category":   category,
            "materials":  norm_materials,
            "eol":        eol_for_pred,
            "origin_pct": origin_pct,
        }
        self._statuses[id(card)] = "\n".join(status_parts)
        self._push_all_predictions()

    @staticmethod
    def _normalised_eol(eol: EolShares) -> EolShares:
        keys = ("recycling", "hazardous", "inert", "incineration")
        values = [getattr(eol, k) for k in keys]
        scaled = normalise_shares_to_100(values)
        return EolShares(**dict(zip(keys, scaled)))
