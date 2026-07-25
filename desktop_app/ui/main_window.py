"""
Main window: multi-product comparison layout.

Left column
  ├── Shared CategoryPanel (applies to all cards unless overridden)
  ├── ProductCard 1 … 4  (collapsible; each can override category)
  └── "+ Add product" button

Right column — an indicator selector above one of three panels:
  Summary            → SummaryPanel  (radar of every indicator's total)
  indicator, 1 card  → PredictionPanel
  indicator, 2+ cards→ ComparisonPanel

Every card predicts every target on each edit (25 forward passes ~5 ms, so the
selector only chooses what to *display* — it never triggers re-prediction).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

import customtkinter as ctk

from desktop_app.inference_adapter import EolShares, InferenceAdapter
from desktop_app.ui.category_panel import CategoryPanel
from desktop_app.ui.comparison_panel import ComparisonPanel, ProductResult
from desktop_app.ui.prediction_panel import PredictionPanel
from desktop_app.ui.product_card import ProductCard
from desktop_app.ui.summary_panel import IndicatorReading, SummaryPanel, SummaryProduct
from desktop_app.ui.theme import (
    ACCENT, BG, BORDER, MAX_PRODUCTS, PRODUCT_COLORS, SURFACE, SURFACE_HI,
    TEXT_PRI, TEXT_SEC, font, indicator_label, indicator_sort_key,
)
from desktop_app._version import __version__
from src.utils import normalise_shares_to_100


DEBOUNCE_MS = 150

# Selector entry for the all-indicator radar. Not an indicator_key, so it can
# never collide with one from the manifest.
SUMMARY_VIEW = "Summary — all indicators"


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

        # ── indicator selector state ──────────────────────────────────────────
        # One entry per indicator that actually has a "total" target baked, in
        # the canonical display order; unknown indicators sort last rather than
        # being dropped.
        self._indicator_keys = sorted(
            (k for k in self.adapter.indicator_keys()
             if self.adapter.target_for(k, "total") is not None),
            key=indicator_sort_key,
        )
        self._view_options = [SUMMARY_VIEW] + [
            indicator_label(k) for k in self._indicator_keys
        ]
        self._label_to_indicator = {
            indicator_label(k): k for k in self._indicator_keys
        }
        self._current_view = SUMMARY_VIEW

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
            sort=False,
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
        self._right_container.grid_rowconfigure(1, weight=1)
        self._right_container.grid_columnconfigure(0, weight=1)

        selector_bar = ctk.CTkFrame(self._right_container, fg_color="transparent")
        selector_bar.grid(row=0, column=0, sticky="ew", padx=20, pady=(16, 0))
        ctk.CTkLabel(
            selector_bar, text="Showing", font=font(12), text_color=TEXT_SEC,
        ).pack(side="left", padx=(0, 10))
        self._view_menu = ctk.CTkOptionMenu(
            selector_bar,
            values=self._view_options,
            command=self._on_view_change,
            font=font(12),
            dropdown_font=font(12),
            fg_color=SURFACE_HI,
            button_color=SURFACE_HI,
            button_hover_color=BORDER,
            text_color=TEXT_PRI,
            dropdown_fg_color=SURFACE_HI,
            dropdown_text_color=TEXT_PRI,
            dropdown_hover_color=BORDER,
            width=210,
        )
        self._view_menu.set(self._current_view)
        self._view_menu.pack(side="left")

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

    # ── indicator selector ────────────────────────────────────────────────────

    def _on_view_change(self, choice: str) -> None:
        if choice == self._current_view:
            return
        self._current_view = choice
        self._rebuild_right_panel()
        self._push_all_predictions()

    @property
    def _summary_view(self) -> bool:
        return self._current_view == SUMMARY_VIEW

    @property
    def _current_indicator(self) -> str:
        """Indicator key backing the current view (the default when on Summary)."""
        return self._label_to_indicator.get(
            self._current_view, self.adapter.default_indicator_key
        )

    def _current_target_key(self) -> str:
        return (
            self.adapter.target_for(self._current_indicator, "total")
            or self.adapter.default_target_key
        )

    # ── right panel management ────────────────────────────────────────────────

    def _rebuild_right_panel(self) -> None:
        if self._active_right_panel is not None:
            self._active_right_panel.destroy()
            self._active_right_panel = None

        if self._summary_view:
            panel = SummaryPanel(self._right_container)
        elif len(self._cards) <= 1:
            entry = self.adapter.manifest[self._current_target_key()]
            loaded = self.adapter.loaded[self._current_target_key()]
            panel = PredictionPanel(
                self._right_container,
                color=PRODUCT_COLORS[0],
                # Full indicator name, not the manifest's terse "GHG Total" —
                # the manifest name stays as-is for diagnostics and plots.
                display_name=indicator_label(self._current_indicator).lower(),
                unit=entry["unit"],
                value_min=loaded.value_min,
                value_max=loaded.value_max,
            )
        else:
            panel = ComparisonPanel(
                self._right_container,
                unit=self.adapter.manifest[self._current_target_key()]["unit"],
            )

        panel.grid(row=1, column=0, sticky="nsew")
        self._active_right_panel = panel

    def _push_all_predictions(self) -> None:
        if self._active_right_panel is None or not self._cards:
            return

        if self._summary_view:
            self._push_summary()
        elif len(self._cards) == 1:
            self._push_single()
        else:
            self._push_comparison()

    def _push_summary(self) -> None:
        products: List[SummaryProduct] = []
        for card in self._cards:
            snap = self._predictions.get(id(card))
            if snap is None:
                continue
            readings: List[IndicatorReading] = []
            for indicator in self._indicator_keys:
                target_key = self.adapter.target_for(indicator, "total")
                if target_key is None or target_key not in snap["all_preds"]:
                    continue
                entry = self.adapter.manifest[target_key]
                value = snap["all_preds"][target_key]
                readings.append(IndicatorReading(
                    indicator_key=indicator,
                    value=value,
                    unit=entry["unit"],
                    display_name=entry["display_name"],
                    ref=self.adapter.reference(snap["category"], target_key=target_key),
                    bounds=self.adapter.prediction_range(
                        value, snap["category"], target_key=target_key
                    ),
                ))
            if readings:
                products.append(SummaryProduct(
                    name=card.name(),
                    color=card.color,
                    category=snap["category"],
                    readings=readings,
                ))
        self._active_right_panel.update(products)
        self._active_right_panel.set_status(self._combined_status())

    def _push_single(self) -> None:
        card   = self._cards[0]
        snap   = self._predictions.get(id(card))
        status = self._statuses.get(id(card), "")
        panel  = self._active_right_panel

        if snap is None:
            panel.clear_prediction()
            panel.set_status(status)
            return

        target_key = self._current_target_key()
        value      = snap["all_preds"].get(target_key)
        if value is None:
            panel.clear_prediction()
            panel.set_status(status)
            return

        panel.set_prediction(
            value,
            self.adapter.prediction_range(value, snap["category"], target_key=target_key),
            breakdown=self._breakdown_for(snap, self._current_indicator),
            reference=self.adapter.reference(snap["category"], target_key=target_key),
        )
        panel.set_status(status)

    def _push_comparison(self) -> None:
        target_key = self._current_target_key()
        results: List[ProductResult] = []
        for card in self._cards:
            snap = self._predictions.get(id(card))
            if snap is None:
                continue
            value = snap["all_preds"].get(target_key)
            if value is None:
                continue
            results.append(ProductResult(
                name=card.name(),
                value=value,
                bounds=self.adapter.prediction_range(
                    value, snap["category"], target_key=target_key
                ),
                color=card.color,
                category=snap["category"],
                materials=snap["materials"],
                eol=snap["eol"],
                origin_pct=snap["origin_pct"],
                breakdown=self._breakdown_for(snap, self._current_indicator),
            ))
        self._active_right_panel.update(results)

    def _breakdown_for(self, snap: dict, indicator_key: str) -> List[dict]:
        """Per-stage values of one indicator, for the collapsible breakdown chart."""
        out: List[dict] = []
        for stage_key, target_key in self.adapter.stage_target_keys(indicator_key).items():
            value = snap["all_preds"].get(target_key)
            if value is None:
                continue
            entry = self.adapter.manifest[target_key]
            out.append({
                "stage_key":    stage_key,
                "display_name": entry["display_name"],
                "unit":         entry["unit"],
                "value":        value,
                "bounds":       self.adapter.prediction_range(
                    value, snap["category"], target_key=target_key
                ),
            })
        return out

    def _combined_status(self) -> str:
        """One status line for panels that show several cards at once."""
        parts = []
        for card in self._cards:
            text = self._statuses.get(id(card), "")
            if text:
                parts.append(f"{card.name()}: {text}" if len(self._cards) > 1 else text)
        return "\n".join(parts)

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
            all_preds = self.adapter.predict_all(
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

        # Normalised materials for display (what the model actually used)
        if mat_total > 0:
            norm_materials = [
                {"name": m["name"], "percentage": m["percentage"] / mat_total * 100.0}
                for m in materials
            ]
        else:
            norm_materials = list(materials)

        # Every target is stored, not just the displayed one: switching the
        # indicator selector is then a pure redraw with no re-prediction.
        self._predictions[id(card)] = {
            "all_preds":  all_preds,
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
