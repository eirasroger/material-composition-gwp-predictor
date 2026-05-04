"""
Main window: a 2-column layout wiring the five input panels (category,
materials, EoL, origin) to the prediction panel via a debounced predict
callback that drives ``InferenceAdapter.predict``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import customtkinter as ctk

from desktop_app.inference_adapter import EolShares, InferenceAdapter
from desktop_app.ui.category_panel import CategoryPanel
from desktop_app.ui.eol_panel import EolPanel
from desktop_app.ui.materials_panel import MaterialsPanel
from desktop_app.ui.origin_panel import OriginPanel
from desktop_app.ui.prediction_panel import PredictionPanel
from desktop_app.ui.theme import ACCENT, BG, BORDER, SURFACE, font
from src.utils import normalise_shares_to_100


DEBOUNCE_MS = 150


def _icon_path() -> Path:
    base = Path(getattr(sys, "_MEIPASS", "")) / "assets" if getattr(sys, "frozen", False) \
        else Path(__file__).resolve().parents[1] / "assets"
    return base / "icon.ico"


class MainWindow(ctk.CTk):
    def __init__(self, adapter: InferenceAdapter) -> None:
        super().__init__()
        self.adapter = adapter

        self.title("GHG Predictor")
        self.geometry("1200x800")
        self.minsize(980, 700)
        self.configure(fg_color=BG)

        icon = _icon_path()
        if icon.exists():
            try:
                self.iconbitmap(str(icon))
            except Exception:
                pass

        self._pending_after: str | None = None

        self.grid_columnconfigure(0, weight=1, uniform="cols")
        self.grid_columnconfigure(1, weight=1, uniform="cols")
        self.grid_rowconfigure(0, weight=1)

        # ── left column: scrollable input stack ───────────────────────────────
        left = ctk.CTkScrollableFrame(
            self,
            label_text="  Configuration",
            label_font=font(12, "bold"),
            label_fg_color=BORDER,
            fg_color=BG,
            scrollbar_button_color=BORDER,
            scrollbar_button_hover_color=ACCENT,
        )
        left.grid(row=0, column=0, sticky="nsew", padx=(12, 6), pady=12)

        self.category_panel = CategoryPanel(
            left, categories=self.adapter.categories,
            on_change=lambda _v: self._schedule_predict(),
        )
        self.category_panel.pack(fill="x", pady=(6, 8))

        self.materials_panel = MaterialsPanel(
            left, material_choices=self.adapter.materials,
            on_change=lambda _m: self._schedule_predict(),
        )
        self.materials_panel.pack(fill="x", pady=(0, 8))

        self.eol_panel = EolPanel(
            left, on_change=lambda _s: self._schedule_predict(),
        )
        self.eol_panel.pack(fill="x", pady=(0, 8))

        self.origin_panel = OriginPanel(
            left, on_change=lambda _v: self._schedule_predict(),
        )
        self.origin_panel.pack(fill="x", pady=(0, 8))

        # ── right column: prediction ──────────────────────────────────────────
        right = ctk.CTkFrame(self, fg_color=SURFACE, corner_radius=10)
        right.grid(row=0, column=1, sticky="nsew", padx=(6, 12), pady=12)
        right.grid_rowconfigure(0, weight=1)
        right.grid_columnconfigure(0, weight=1)

        self.prediction_panel = PredictionPanel(right)
        self.prediction_panel.grid(row=0, column=0, sticky="nsew")

        self._schedule_predict()

    # ── prediction wiring ─────────────────────────────────────────────────────
    def _schedule_predict(self) -> None:
        if self._pending_after is not None:
            try:
                self.after_cancel(self._pending_after)
            except Exception:
                pass
        self._pending_after = self.after(DEBOUNCE_MS, self._predict_now)

    def _predict_now(self) -> None:
        self._pending_after = None

        category   = self.category_panel.selected()
        materials  = self.materials_panel.materials()
        eol_shares = self.eol_panel.shares()
        origin_pct = self.origin_panel.value()

        if category is None:
            self.prediction_panel.clear_prediction()
            self.prediction_panel.set_status("Pick a product category to begin.")
            return
        if not materials:
            self.prediction_panel.clear_prediction()
            self.prediction_panel.set_status(
                "Add at least one material from the dropdown."
            )
            return

        status_lines: List[str] = []
        mat_total = sum(m["percentage"] for m in materials)
        if abs(mat_total - 100.0) > 0.05:
            status_lines.append(
                f"Materials sum to {mat_total:.1f}% — predicted as if normalised."
            )
        eol_total = self.eol_panel.total()
        if abs(eol_total - 100.0) > 0.05:
            status_lines.append(
                f"End-of-life shares sum to {eol_total:.1f}% — "
                "predicted as if normalised."
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
            self.prediction_panel.clear_prediction()
            self.prediction_panel.set_status(f"Prediction failed: {exc}")
            return

        self.prediction_panel.set_prediction(value)
        self.prediction_panel.set_status("\n".join(status_lines))

    @staticmethod
    def _normalised_eol(eol: EolShares) -> EolShares:
        keys = ("recycling", "hazardous", "inert", "incineration")
        values = [getattr(eol, k) for k in keys]
        scaled = normalise_shares_to_100(values)
        return EolShares(**dict(zip(keys, scaled)))
