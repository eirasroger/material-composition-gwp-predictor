"""
Comparison panel: a single matplotlib figure with two vertically stacked
subplots sharing the same x-axis.

  Top  — bar chart: one bar per product, error bars for plausible range.
  Bottom — summary table: normalised material / EoL / origin values aligned
           under each bar. Category row appears only when categories differ.

Products with no prediction are omitted entirely.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import textwrap
import tkinter as tk

import customtkinter as ctk
import matplotlib
matplotlib.use("TkAgg")  # noqa: E402
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from desktop_app.ui.theme import ACCENT, BORDER, BG, SURFACE, TEXT_DIM, TEXT_SEC, font


@dataclass
class ProductResult:
    name: str
    value: float
    bounds: Optional[tuple]         # (lower, upper) plausible range, or None
    color: str
    category: str
    materials: List[Dict]           # normalised: [{"name": str, "percentage": float}]
    eol: Any                        # EolShares (normalised)
    origin_pct: float


# ── layout constants ──────────────────────────────────────────────────────────
_FIG_W        = 5.6    # figure width in inches
_DPI          = 100
_BAR_H        = 8.0    # bar chart subplot height in inches
_ROW_H        = 0.26   # height per summary row in inches
_LABEL_W      = 1.3    # x-units reserved for the label column (negative x side)
_LEFT_MARGIN  = 0.10   # figure left margin fraction (space for y-axis label)
_RIGHT_MARGIN = 0.97   # figure right margin fraction
_CHAR_WIDTH_IN = 0.00425


# ── cell helpers ──────────────────────────────────────────────────────────────

def _truncate_cell(text: str, n_prod: int, fontsize: int = 11) -> str:
    """Truncate *text* so it fits within one column of the summary table.

    The column width is derived from the figure dimensions and the number of
    products.  A trailing ellipsis replaces any characters that would overflow.
    """
    col_w_inches = (_FIG_W - _LABEL_W) / max(n_prod, 1)
    limit = max(6, int(col_w_inches / (fontsize * _CHAR_WIDTH_IN)))
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


class ComparisonPanel(ctk.CTkFrame):
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

        self._canvas_host = tk.Frame(
            self._scroll, bg=SURFACE, highlightthickness=0, bd=0,
        )
        self._canvas_host.pack(fill="x")

        self._fig: Optional[Figure] = None
        self._mpl_canvas: Optional[FigureCanvasTkAgg] = None

        self._show_empty()

    # ── public ───────────────────────────────────────────────────────────────

    def update(self, products: List[ProductResult]) -> None:
        if not products:
            self._show_empty()
            return

        rows = self._build_rows(products)
        n_prod = len(products)

        summary_h = max(len(rows) * _ROW_H + 0.3, 1.0)
        total_h   = _BAR_H + summary_h

        fig = Figure(figsize=(_FIG_W, total_h), dpi=_DPI)
        fig.patch.set_facecolor(SURFACE)

        gs = fig.add_gridspec(
            2, 1,
            height_ratios=[_BAR_H, summary_h],
            hspace=0.0,
        )
        ax_bars    = fig.add_subplot(gs[0])
        ax_summary = fig.add_subplot(gs[1])

        x_left  = -_LABEL_W - 0.2
        x_right = n_prod - 0.4

        self._draw_bars(ax_bars, products, x_left, x_right)
        self._draw_summary(ax_summary, rows, n_prod, x_left, x_right)

        fig.subplots_adjust(
            left=_LEFT_MARGIN, right=_RIGHT_MARGIN,
            top=0.97, bottom=0.02,
            hspace=0.02,
        )

        self._set_canvas(fig)

    # ── bar chart subplot ─────────────────────────────────────────────────────

    def _draw_bars(self, ax, products, x_left, x_right) -> None:
        uppers  = [p.bounds[1] for p in products if p.bounds is not None]
        raw_max = max(uppers) if uppers else max(p.value for p in products)
        y_max   = max(raw_max * 1.20, 1.0)

        label_zone = y_max * 0.07

        for i, p in enumerate(products):
            ax.bar(
                i, p.value,
                color=p.color + "bb", edgecolor=p.color,
                linewidth=0.8, width=0.5, zorder=2,
            )
            if p.bounds is not None:
                lo, hi = p.bounds
                ax.errorbar(
                    i, p.value,
                    yerr=[[max(0.0, p.value - lo)], [max(0.0, hi - p.value)]],
                    fmt="none", color=p.color,
                    capsize=6, elinewidth=1.5, capthick=1.5, zorder=3,
                )
            ann_y = (p.bounds[1] if p.bounds else p.value) + y_max * 0.025
            ax.text(
                i, ann_y, f"{p.value:.3f}",
                ha="center", va="bottom",
                color=p.color, fontsize=10, fontweight="bold",
            )
            # Product name drawn inside the negative zone, below the x-axis line
            ax.text(
                i, -label_zone * 0.52,
                p.name[:14],
                ha="center", va="center",
                color=p.color, fontsize=10.5, fontweight="bold",
            )

        ax.axhline(0, color=BORDER, linewidth=1.0, zorder=1)

        ax.set_xlim(x_left, x_right)
        ax.set_ylim(-label_zone, y_max)
        ax.set_xticks([])
        ax.set_ylabel("kg CO₂eq / kg", color=TEXT_SEC, fontsize=10, labelpad=4)
        ax.tick_params(axis="y", colors=TEXT_SEC, labelsize=10)
        ax.set_facecolor(SURFACE)
        for spine in ("top", "right", "bottom"):
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_color(BORDER)

    # ── summary table subplot ─────────────────────────────────────────────────

    def _draw_summary(self, ax, rows, n_prod, x_left, x_right) -> None:
        ax.set_facecolor(SURFACE)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(x_left, x_right)

        n_rows = len(rows)
        ax.set_ylim(-n_rows - 0.2, 0.8)

        label_x = x_left + 0.08

        for idx, row in enumerate(rows):
            y = -idx
            rtype = row["type"]

            if rtype == "spacer":
                continue

            if rtype == "section":
                ax.text(
                    label_x, y, row["label"],
                    ha="left", va="center",
                    color=TEXT_SEC, fontsize=12, fontweight="bold",
                )
                ax.axhline(
                    y - 0.45,
                    xmin=0.0, xmax=1.0,
                    color=BORDER, linewidth=0.5, zorder=1,
                )
            elif rtype in ("data", "category"):
                ax.text(
                    label_x + 0.12, y, row["label"],
                    ha="left", va="center",
                    color=TEXT_DIM, fontsize=11,
                )
                for i, val in enumerate(row["values"]):
                    if val is None:
                        ax.text(
                            i, y, "—",
                            ha="center", va="center",
                            color=TEXT_DIM, fontsize=11,
                        )
                    else:
                        ax.text(
                            i, y, val,
                            ha="center", va="center",
                            color=TEXT_SEC, fontsize=11,
                        )

    # ── row builder ───────────────────────────────────────────────────────────

    def _build_rows(self, products: List[ProductResult]) -> List[dict]:
        rows: List[dict] = []
        n_prod = len(products)

        # Materials
        rows.append({"type": "section", "label": "Materials"})

        mat_count: Dict[str, int] = {}
        for p in products:
            for m in p.materials:
                mat_count[m["name"]] = mat_count.get(m["name"], 0) + 1

        # Most-shared first, then alphabetical within each count group
        all_mats = sorted(mat_count, key=lambda n: (-mat_count[n], n.lower()))

        mat_lookup = [
            {m["name"]: m["percentage"] for m in p.materials}
            for p in products
        ]
        for mat in all_mats:
            vals = []
            for lk in mat_lookup:
                pct = lk.get(mat)
                vals.append(f"{pct:.1f}%" if pct is not None else None)
            rows.append({
                "type": "data",
                "label": mat[:18],
                "values": vals,
            })

        rows.append({"type": "spacer"})

        # End-of-life
        rows.append({"type": "section", "label": "End-of-life"})
        for key, label in (
            ("recycling",    "Recycling"),
            ("hazardous",    "Hazardous"),
            ("inert",        "Inert landfill"),
            ("incineration", "Incineration"),
        ):
            rows.append({
                "type": "data",
                "label": label,
                "values": [f"{getattr(p.eol, key):.1f}%" for p in products],
            })

        rows.append({"type": "spacer"})

        # Circular origin
        rows.append({"type": "section", "label": "Circular Origin"})
        rows.append({
            "type": "data",
            "label": "Origin %",
            "values": [f"{p.origin_pct:.1f}%" for p in products],
        })

        # Category — only if products differ; placed last
        cats = [p.category for p in products]
        if len(set(cats)) > 1:
            rows.append({"type": "spacer"})
            rows.append({"type": "section", "label": "Category"})
            rows.append({
                "type": "data",
                "label": "c-PCR",
                "values": [_truncate_cell(p.category, n_prod) for p in products],
            })

        return rows

    # ── canvas management ─────────────────────────────────────────────────────

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
        fig = Figure(figsize=(_FIG_W, 2.0), dpi=_DPI)
        fig.patch.set_facecolor(SURFACE)
        ax = fig.add_subplot(111)
        ax.set_facecolor(SURFACE)
        for spine in ax.spines.values():
            spine.set_color(BORDER)
        ax.text(
            0.5, 0.5, "Configure products to see comparison",
            ha="center", va="center",
            color=TEXT_DIM, fontsize=11,
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout(pad=0.5)
        self._set_canvas(fig)