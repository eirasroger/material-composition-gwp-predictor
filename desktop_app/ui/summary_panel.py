"""
Summary panel: all indicator totals condensed into one radar.

Each axis is one indicator's ``total``. The radial coordinate is **log(value /
median of real products in this product's category)** — so the ring at 1.0 is
"a typical product of this kind" and the reader's question ("is this high or
low?") is answered by whether the polygon sits inside or outside it.

Why log: the ratios are log-normal (p25 ~0.36x, p75 ~2.5x, p95 up to 85x). On a
linear radius everything below the median collapses into the centre dot. One
decade each side (0.1x .. 10x) covers roughly p5-p95; beyond that the vertex is
pinned to the rim/centre and flagged, rather than silently rescaling the chart.

Why a per-category median and not a global percentile: globally, a concrete is
low-impact per kg no matter how badly it is formulated, so a global rank says
nothing about the composition. "Worse than a typical concrete" does.

The shaded band is that category's p25-p75 — the range normal products occupy.
It carries the uncertainty visually: a narrow band means composition barely
moves this indicator here, a wide one means it dominates.

1 product  -> one large radar.
2-4        -> small multiples, each against its own category's band. Never
              overlaid: cards can carry different categories, so a shared axis
              would not mean the same thing for each polygon.
"""

from __future__ import annotations

import math
import tkinter as tk
from dataclasses import dataclass
from typing import List, Optional

import customtkinter as ctk
import matplotlib
matplotlib.use("TkAgg")  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from desktop_app.ui.theme import (
    ACCENT, BORDER, SURFACE, TEXT_DIM, TEXT_PRI, TEXT_SEC, font,
    fmt_value as _fmt_value, indicator_label, indicator_sort_key,
)

# Radial extent, in decades either side of the category median.
_DECADES     = 1.0          # 0.1x .. 10x
_MEDIAN_R    = 0.5          # normalised radius of the median ring
_DPI         = 100


@dataclass
class IndicatorReading:
    indicator_key: str
    value: float
    unit: str
    display_name: str
    # Reference distribution (desktop_app.inference_adapter.Reference) or None
    # when this target has no usable median to divide by.
    ref: Optional[object] = None


@dataclass
class SummaryProduct:
    name: str
    color: str
    category: str
    readings: List[IndicatorReading]


def _norm_radius(ratio: float) -> float:
    """value/median -> normalised radius in [0, 1], median at 0.5. Clamped."""
    if ratio <= 0:
        return 0.0
    r = (math.log10(ratio) + _DECADES) / (2 * _DECADES)
    return min(1.0, max(0.0, r))


class SummaryPanel(ctk.CTkFrame):
    """Right-hand panel for the "Summary" selector entry."""

    def __init__(self, master) -> None:
        super().__init__(master, fg_color="transparent")

        self._scroll = ctk.CTkScrollableFrame(
            self,
            fg_color=SURFACE,
            corner_radius=0,
            label_text="",
            label_fg_color=SURFACE,
            scrollbar_button_color=BORDER,
            scrollbar_button_hover_color=ACCENT,
        )
        self._scroll.pack(fill="both", expand=True)

        ctk.CTkLabel(
            self._scroll,
            text="All indicators vs. typical products",
            font=font(12),
            text_color=TEXT_SEC,
        ).pack(anchor="w", padx=20, pady=(16, 0))

        self._subtitle = ctk.CTkLabel(
            self._scroll,
            text="",
            font=font(11),
            text_color=TEXT_DIM,
            anchor="w",
            justify="left",
            wraplength=430,
        )
        self._subtitle.pack(anchor="w", padx=20, pady=(2, 6))

        self._canvas_host = tk.Frame(self._scroll, bg=SURFACE, highlightthickness=0, bd=0)
        self._canvas_host.pack(fill="x")

        self._table = ctk.CTkFrame(self._scroll, fg_color="transparent")
        self._table.pack(fill="x", padx=20, pady=(4, 8))

        self._status_label = ctk.CTkLabel(
            self._scroll,
            text="",
            font=font(11),
            text_color=TEXT_SEC,
            anchor="w",
            justify="left",
            wraplength=430,
        )
        self._status_label.pack(fill="x", padx=20, pady=(0, 14))

        self._fig: Optional[Figure] = None
        self._mpl_canvas: Optional[FigureCanvasTkAgg] = None
        self._show_empty()

    # ── public API (mirrors ComparisonPanel.update) ───────────────────────────

    def update(self, products: List[SummaryProduct]) -> None:
        if not products:
            self._show_empty()
            self._clear_table()
            self._subtitle.configure(text="")
            return

        n = len(products)
        if n == 1:
            fig = Figure(figsize=(5.0, 4.8), dpi=_DPI)
            fig.patch.set_facecolor(SURFACE)
            self._draw_radar(fig.add_subplot(111, polar=True), products[0], big=True)
            fig.subplots_adjust(left=0.13, right=0.87, top=0.92, bottom=0.06)
            cats = products[0].category
        else:
            cols = 2
            rows = int(math.ceil(n / cols))
            fig = Figure(figsize=(4.9, 2.75 * rows), dpi=_DPI)
            fig.patch.set_facecolor(SURFACE)
            for i, p in enumerate(products):
                ax = fig.add_subplot(rows, cols, i + 1, polar=True)
                self._draw_radar(ax, p, big=False)
            # Generous top: each subplot carries its product name as a title,
            # which sits above the axes box and is clipped by a tighter margin.
            fig.subplots_adjust(
                left=0.08, right=0.92, top=0.86, bottom=0.04, wspace=0.45, hspace=0.50,
            )
            uniq = list(dict.fromkeys(p.category for p in products))
            cats = uniq[0] if len(uniq) == 1 else f"{len(uniq)} different categories"

        self._subtitle.configure(
            text=(
                f"Ring = median product in {cats}. Shaded band = the middle 50% "
                f"of those products. Outside the ring is worse than typical."
            )
        )
        self._set_canvas(fig)
        self._build_table(products)

    def clear(self) -> None:
        self._show_empty()
        self._clear_table()
        self._subtitle.configure(text="")

    def set_status(self, text: str) -> None:
        self._status_label.configure(text=text)

    # ── radar ─────────────────────────────────────────────────────────────────

    def _draw_radar(self, ax, product: SummaryProduct, big: bool) -> None:
        readings = sorted(
            product.readings, key=lambda r: indicator_sort_key(r.indicator_key)
        )
        n = len(readings)
        if n < 3:
            ax.set_axis_off()
            return

        theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
        closed = np.concatenate([theta, theta[:1]])

        ax.set_facecolor(SURFACE)
        ax.set_theta_offset(np.pi / 2)      # first axis at the top
        ax.set_theta_direction(-1)          # clockwise
        ax.set_ylim(0, 1.0)
        ax.set_yticks([])
        ax.set_xticks(theta)
        ax.set_xticklabels(
            [indicator_label(r.indicator_key) for r in readings],
            color=TEXT_SEC, fontsize=9 if big else 7.5,
        )
        ax.tick_params(pad=6 if big else 2)
        ax.spines["polar"].set_color(BORDER)
        ax.grid(False)

        # Typical range (p25-p75) — neutral, recessive: it is context, not a series.
        lo, hi, has_band = [], [], False
        for r in readings:
            if r.ref is None:
                lo.append(_MEDIAN_R)
                hi.append(_MEDIAN_R)
                continue
            has_band = True
            lo.append(_norm_radius(r.ref.p25 / r.ref.p50))
            hi.append(_norm_radius(r.ref.p75 / r.ref.p50))
        if has_band:
            lo_c = np.concatenate([lo, lo[:1]])
            hi_c = np.concatenate([hi, hi[:1]])
            ax.fill_between(
                closed, lo_c, hi_c,
                color=TEXT_DIM, alpha=0.22, linewidth=0, zorder=1,
            )
            # Hairline edges: in wide-spread categories the fill alone covers most
            # of the disc and loses its boundary.
            for edge in (lo_c, hi_c):
                ax.plot(closed, edge, color=TEXT_DIM, linewidth=0.8, alpha=0.85, zorder=1)

        # Median ring — the reference the whole chart is read against. Drawn
        # through the same 5 vertices as the data (not a true circle) so a
        # product exactly at the median lies exactly on it.
        ring = np.full_like(closed, _MEDIAN_R)
        ax.plot(closed, ring, color=TEXT_SEC, linewidth=1.0, alpha=0.85, zorder=2)

        # Radial scale, on the gap between the first two axes so it never
        # collides with an axis label or a vertex.
        if big:
            gap_theta = float(theta[0] + (theta[1] - theta[0]) / 2)
            for radius, text in ((_MEDIAN_R, "median"), (1.0, f"{10 ** _DECADES:.0f}×")):
                ax.text(
                    gap_theta, radius, text,
                    color=TEXT_SEC, fontsize=7.5, ha="center", va="center", zorder=7,
                    # Opaque plate: these sit on top of the ring and the band
                    # edges, which otherwise strike straight through the text.
                    bbox=dict(
                        boxstyle="round,pad=0.22", facecolor=SURFACE,
                        edgecolor="none", alpha=0.92,
                    ),
                )

        # The product itself.
        radii, off_scale = [], []
        for r in readings:
            if r.ref is None:
                radii.append(_MEDIAN_R)
                off_scale.append(False)
                continue
            ratio = r.ref.ratio(r.value)
            radii.append(_norm_radius(ratio))
            off_scale.append(ratio > 10 ** _DECADES or 0 < ratio < 10 ** -_DECADES)

        closed_r = np.concatenate([radii, radii[:1]])
        ax.plot(closed, closed_r, color=product.color, linewidth=2.0, zorder=4)
        ax.fill(closed, closed_r, color=product.color, alpha=0.22, zorder=3)
        ax.scatter(
            theta, radii, s=34 if big else 18, color=product.color,
            edgecolors=SURFACE, linewidths=1.5, zorder=5,
        )
        # Values past one decade are pinned to the rim/centre — mark them so a
        # clamped vertex is never read as a real value.
        for t, rad, off in zip(theta, radii, off_scale):
            if off:
                ax.scatter(
                    [t], [rad], s=90 if big else 55, marker="^",
                    color=product.color, edgecolors=SURFACE, linewidths=1.0, zorder=6,
                )

        if not big:
            ax.set_title(
                product.name[:16], color=product.color,
                fontsize=9, fontweight="bold", pad=10,
            )

    # ── native-value table ────────────────────────────────────────────────────

    def _clear_table(self) -> None:
        for w in self._table.winfo_children():
            w.destroy()

    def _build_table(self, products: List[SummaryProduct]) -> None:
        """
        The radar shows position only; the actual numbers live here. Also carries
        the "x median" multiple, which is the radar's own quantity in words.
        """
        self._clear_table()

        order = sorted(
            {r.indicator_key for p in products for r in p.readings},
            key=indicator_sort_key,
        )
        by_product = [
            {r.indicator_key: r for r in p.readings} for p in products
        ]

        self._table.grid_columnconfigure(0, weight=0)
        for i in range(len(products)):
            self._table.grid_columnconfigure(i + 1, weight=1)

        if len(products) > 1:
            for i, p in enumerate(products):
                ctk.CTkLabel(
                    self._table, text=p.name[:14], font=font(11, "bold"),
                    text_color=p.color, anchor="e",
                ).grid(row=0, column=i + 1, sticky="e", padx=(8, 0), pady=(0, 2))

        row0 = 1 if len(products) > 1 else 0
        for ri, ind in enumerate(order):
            # The unit is a property of the indicator, not of any one product,
            # so it belongs in the row label. Repeating it in every cell is what
            # overflowed the columns and clipped the values.
            unit = next(
                (lk[ind].unit for lk in by_product if ind in lk), ""
            )
            ctk.CTkLabel(
                self._table,
                text=f"{indicator_label(ind)}\n{unit}",
                font=font(11), text_color=TEXT_SEC,
                anchor="w", justify="left",
            ).grid(row=row0 + ri, column=0, sticky="w", pady=2)

            for ci, lookup in enumerate(by_product):
                reading = lookup.get(ind)
                if reading is None:
                    text = "—"
                else:
                    text = _fmt_value(reading.value)
                    if reading.ref is not None:
                        text += f"\n{reading.ref.ratio(reading.value):.2g}× median"
                ctk.CTkLabel(
                    self._table, text=text, font=font(11),
                    text_color=TEXT_PRI, anchor="e", justify="right",
                ).grid(row=row0 + ri, column=ci + 1, sticky="e", padx=(10, 0), pady=2)

        scopes = {
            r.ref.scope for p in products for r in p.readings if r.ref is not None
        }
        if "global" in scopes:
            ctk.CTkLabel(
                self._table,
                text=(
                    "Some indicators compare against all products — this category "
                    "has too few for its own reference."
                ),
                font=font(10), text_color=TEXT_DIM, anchor="w",
                justify="left", wraplength=420,
            ).grid(
                row=row0 + len(order), column=0, columnspan=len(products) + 1,
                sticky="w", pady=(6, 0),
            )

    # ── canvas ────────────────────────────────────────────────────────────────

    def _set_canvas(self, fig: Figure) -> None:
        if self._mpl_canvas is not None:
            self._mpl_canvas.get_tk_widget().destroy()
            self._mpl_canvas = None
        if self._fig is not None:
            self._fig.clf()
            self._fig = None

        self._fig = fig
        self._mpl_canvas = FigureCanvasTkAgg(fig, master=self._canvas_host)
        widget = self._mpl_canvas.get_tk_widget()
        widget.pack(fill="x")
        widget.configure(bg=SURFACE, highlightthickness=0)
        self._mpl_canvas.draw()

    def _show_empty(self) -> None:
        fig = Figure(figsize=(4.6, 2.4), dpi=_DPI)
        fig.patch.set_facecolor(SURFACE)
        ax = fig.add_subplot(111)
        ax.set_facecolor(SURFACE)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.text(
            0.5, 0.5, "Configure a product to see all indicators",
            ha="center", va="center", color=TEXT_DIM, fontsize=11,
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout(pad=0.5)
        self._set_canvas(fig)


def _smoke() -> None:
    """Renders the radar against the real baked distributions."""
    from pathlib import Path

    from desktop_app.inference_adapter import InferenceAdapter
    from desktop_app.ui.theme import PRODUCT_COLORS

    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme(
        str(Path(__file__).resolve().parents[1] / "assets" / "theme_dark.json")
    )

    adapter = InferenceAdapter()
    category = "Concrete and concrete elements (c-PCR-003)"
    if category not in adapter.categories:
        category = adapter.categories[0]

    def make(name: str, color: str, scale: float) -> SummaryProduct:
        readings = []
        for ind in ("ghg", "fw", "ep", "ap", "adpf"):
            tk_ = adapter.target_for(ind, "total")
            if tk_ is None:
                continue
            ref = adapter.reference(category, target_key=tk_)
            value = (ref.p50 * scale) if ref else 1.0
            readings.append(IndicatorReading(
                indicator_key=ind, value=value,
                unit=adapter.manifest[tk_]["unit"],
                display_name=adapter.manifest[tk_]["display_name"],
                ref=ref,
            ))
        return SummaryProduct(name=name, color=color, category=category, readings=readings)

    root = ctk.CTk()
    root.title("summary_panel.py smoke test")
    root.geometry("560x760")
    root.configure(fg_color=SURFACE)

    panel = SummaryPanel(root)
    panel.pack(fill="both", expand=True)

    products = [
        make("Typical", PRODUCT_COLORS[0], 1.0),
        make("Half",    PRODUCT_COLORS[1], 0.5),
        make("Triple",  PRODUCT_COLORS[2], 3.0),
        make("Extreme", PRODUCT_COLORS[3], 40.0),
    ]

    btns = ctk.CTkFrame(root, fg_color="transparent")
    btns.pack(fill="x", padx=16, pady=(0, 10))
    for k in (1, 2, 3, 4):
        ctk.CTkButton(
            btns, text=f"{k} product(s)", width=90,
            command=lambda k=k: panel.update(products[:k]),
        ).pack(side="left", padx=4)
    ctk.CTkButton(btns, text="clear", width=70, command=panel.clear).pack(side="left", padx=4)

    panel.update(products[:1])
    root.mainloop()


if __name__ == "__main__":
    _smoke()
