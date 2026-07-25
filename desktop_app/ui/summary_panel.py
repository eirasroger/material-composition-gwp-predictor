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

1 product           -> one radar.
2-4, same category  -> one radar, overlaid, with a legend. The axes mean the
                       same thing for every product, so one band and one ring
                       serve all of them.
2-4, mixed category -> small multiples, each against its own band. Overlaying
                       here would put polygons on an axis that does not mean
                       the same thing for each.

Deliberately NOT drawn: the outer rim circle and radial ("median" / "10x")
labels. The rim is where the log scale is clamped, not a value in the data, and
labelling it invites "10x what?". The ring and band are explained once in the
subtitle instead, in words.
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
from matplotlib.lines import Line2D  # noqa: E402

from desktop_app.ui.theme import (
    ACCENT, BORDER, SURFACE, TEXT_DIM, TEXT_PRI, TEXT_SEC, font,
    fmt_value as _fmt_value, indicator_axis_label, indicator_label,
    indicator_sort_key, status_vs_typical,
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
    # (low, high) plausible range for this prediction. Used to hedge the verdict
    # when the model cannot settle which side of typical the product is on.
    bounds: Optional[tuple] = None


@dataclass
class SummaryProduct:
    name: str
    color: str
    category: str
    readings: List[IndicatorReading]


def _sorted(product: "SummaryProduct") -> List["IndicatorReading"]:
    """Readings in canonical indicator order — the axes must never move."""
    return sorted(product.readings, key=lambda r: indicator_sort_key(r.indicator_key))


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
        categories = list(dict.fromkeys(p.category for p in products))
        shared_category = len(categories) == 1

        if n == 1 or shared_category:
            fig = Figure(figsize=(5.0, 5.4 if n == 1 else 5.8), dpi=_DPI)
            fig.patch.set_facecolor(SURFACE)
            ax = fig.add_subplot(111, polar=True)
            theta, closed = self._draw_frame(ax, _sorted(products[0]), big=True)
            for p in products:
                self._draw_polygon(ax, p, theta, closed, big=True)
            if n > 1:
                self._add_legend(fig, products)
            # Axis labels are two lines for the long names and sit outside the
            # axes box, so the top margin has to clear them.
            fig.subplots_adjust(
                left=0.15, right=0.85, top=0.86, bottom=0.15 if n > 1 else 0.08,
            )
            cats = categories[0]
        else:
            cols = 2
            rows = int(math.ceil(n / cols))
            fig = Figure(figsize=(4.9, 2.75 * rows), dpi=_DPI)
            fig.patch.set_facecolor(SURFACE)
            for i, p in enumerate(products):
                ax = fig.add_subplot(rows, cols, i + 1, polar=True)
                theta, closed = self._draw_frame(ax, _sorted(p), big=False)
                self._draw_polygon(ax, p, theta, closed, big=False)
                ax.set_title(
                    p.name[:16], color=p.color, fontsize=9, fontweight="bold", pad=10,
                )
            # Generous top: each subplot carries its product name as a title,
            # which sits above the axes box and is clipped by a tighter margin.
            fig.subplots_adjust(
                left=0.08, right=0.92, top=0.86, bottom=0.04, wspace=0.45, hspace=0.50,
            )
            cats = f"each product's own category ({len(categories)} different)"

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

    def _draw_frame(self, ax, readings: List[IndicatorReading], big: bool):
        """
        Axes, typical-range band and median ring — everything that is context
        rather than data. Shared by every product drawn on this ax.
        """
        n = len(readings)
        if n < 3:
            ax.set_axis_off()
            return np.array([]), np.array([])

        theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
        closed = np.concatenate([theta, theta[:1]])

        ax.set_facecolor(SURFACE)
        ax.set_theta_offset(np.pi / 2)      # first axis at the top
        ax.set_theta_direction(-1)          # clockwise
        ax.set_ylim(0, 1.0)
        ax.set_yticks([])
        ax.set_xticks(theta)
        ax.set_xticklabels(
            [indicator_axis_label(r.indicator_key) for r in readings],
            color=TEXT_SEC, fontsize=8.5 if big else 7,
        )
        ax.tick_params(pad=10 if big else 4)
        # The rim is kept as a faint frame: it gives the sense of how far a
        # polygon is from the extreme. It is labelled in words ("10x worse")
        # rather than as a bare number, which read as data and invited
        # "10x what?".
        ax.spines["polar"].set_color(BORDER)
        ax.spines["polar"].set_linewidth(1.0)
        ax.grid(False)

        # Typical range (p25-p75): one soft fill, no outline. Neutral ink —
        # it is context, not a series.
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
            ax.fill_between(
                closed,
                np.concatenate([lo, lo[:1]]),
                np.concatenate([hi, hi[:1]]),
                color=TEXT_DIM, alpha=0.25, linewidth=0, zorder=1,
            )

        # Median ring — the one reference line. Drawn through the same vertices
        # as the data (not a true circle) so a product exactly at the median
        # lies exactly on it.
        ax.plot(
            closed, np.full_like(closed, _MEDIAN_R),
            color=TEXT_SEC, linewidth=1.0, alpha=0.85, zorder=2,
        )

        # Scale ends, in words. The axis is symmetric in log — the centre is the
        # mirror of the rim (10x better vs 10x worse) — and neither end is
        # readable without saying so. Placed on the gap between the first two
        # axes so they never sit on a spoke or a vertex.
        if big:
            gap = float(theta[0] + (theta[1] - theta[0]) / 2)
            plate = dict(
                boxstyle="round,pad=0.25", facecolor=SURFACE,
                edgecolor="none", alpha=0.9,
            )
            factor = f"{10 ** _DECADES:.0f}"
            ax.text(
                gap, 1.0, f"{factor}× worse", color=TEXT_DIM, fontsize=7.5,
                ha="center", va="center", zorder=7, bbox=plate,
            )
            ax.text(
                gap, 0.045, f"{factor}× better", color=TEXT_DIM, fontsize=7.5,
                ha="center", va="center", zorder=7, bbox=plate,
            )
        return theta, closed

    def _draw_polygon(
        self, ax, product: SummaryProduct, theta, closed, big: bool,
    ) -> None:
        """
        Outline only, never filled — identically for one product or four. A fill
        was previously used in the single-product case, which made the one- and
        many-product views look like different charts, and filled polygons turn
        to mud as soon as they overlap.
        """
        if theta.size == 0:
            return
        readings = _sorted(product)

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

    def _add_legend(self, fig: Figure, products: List[SummaryProduct]) -> None:
        """Overlaid polygons are only tellable apart by colour — so a legend is required."""
        handles = [
            Line2D([], [], color=p.color, linewidth=2.0, label=p.name[:18])
            for p in products
        ]
        fig.legend(
            handles=handles, loc="lower center", ncol=min(len(products), 2),
            frameon=False, fontsize=9, labelcolor=TEXT_SEC,
            bbox_to_anchor=(0.5, 0.0),
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
            cell = ctk.CTkFrame(self._table, fg_color="transparent")
            cell.grid(row=row0 + ri, column=0, sticky="w", pady=3)
            ctk.CTkLabel(
                cell, text=indicator_label(ind), font=font(11),
                text_color=TEXT_SEC, anchor="w", justify="left", wraplength=170,
            ).pack(anchor="w")
            ctk.CTkLabel(
                cell, text=unit, font=font(10),
                text_color=TEXT_DIM, anchor="w", justify="left",
            ).pack(anchor="w")

            for ci, lookup in enumerate(by_product):
                reading = lookup.get(ind)
                box = ctk.CTkFrame(self._table, fg_color="transparent")
                box.grid(row=row0 + ri, column=ci + 1, sticky="e", padx=(10, 0), pady=3)
                if reading is None:
                    ctk.CTkLabel(
                        box, text="—", font=font(11), text_color=TEXT_DIM, anchor="e",
                    ).pack(anchor="e")
                    continue
                ctk.CTkLabel(
                    box, text=_fmt_value(reading.value), font=font(11),
                    text_color=TEXT_PRI, anchor="e",
                ).pack(anchor="e")
                if reading.ref is not None:
                    # Colour-code against the same p25/p75 the radar shades, so
                    # a red row here always corresponds to a vertex outside the
                    # band up there.
                    lo, hi = reading.bounds if reading.bounds else (None, None)
                    _, short, colour = status_vs_typical(
                        reading.value, reading.ref.p25, reading.ref.p75, lo, hi,
                    )
                    ctk.CTkLabel(
                        box,
                        text=f"{reading.ref.ratio(reading.value):.2g}× median · {short}",
                        font=font(11, "bold"), text_color=colour, anchor="e",
                    ).pack(anchor="e")

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
