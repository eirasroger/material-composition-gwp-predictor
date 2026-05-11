"""
Comparison panel: bar chart comparing up to 4 products side-by-side.
Each bar shows the point-estimate GHG with an error bar spanning the
plausible range (lower, upper). Shown in the right column when 2+ products
are configured.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import tkinter as tk

import customtkinter as ctk
import matplotlib
matplotlib.use("TkAgg")  # noqa: E402
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from desktop_app.ui.theme import BORDER, SURFACE, TEXT_DIM, TEXT_SEC, font


@dataclass
class ProductResult:
    name: str
    value: float
    bounds: Optional[tuple]   # (lower, upper) plausible range, or None
    color: str


class ComparisonPanel(ctk.CTkFrame):
    def __init__(self, master) -> None:
        super().__init__(master, fg_color="transparent")

        ctk.CTkLabel(
            self,
            text="Predicted greenhouse gas emissions",
            font=font(12),
            text_color=TEXT_SEC,
        ).pack(anchor="w", padx=24, pady=(28, 8))

        self._fig = Figure(figsize=(4.4, 3.8), dpi=100)
        self._fig.patch.set_facecolor(SURFACE)
        self._ax = self._fig.add_subplot(111)

        canvas_host = tk.Frame(self, highlightthickness=0, bd=0, bg=SURFACE)
        canvas_host.pack(fill="x", padx=24, pady=(0, 24))
        self._mpl_canvas = FigureCanvasTkAgg(self._fig, master=canvas_host)
        widget = self._mpl_canvas.get_tk_widget()
        widget.pack(fill="x")
        widget.configure(bg=SURFACE, highlightthickness=0)

        self._draw_empty()

    def update(self, products: List[ProductResult]) -> None:
        self._ax.clear()
        if not products:
            self._draw_empty()
            return

        uppers = [p.bounds[1] for p in products if p.bounds is not None]
        raw_max = max(uppers) if uppers else max(p.value for p in products)
        y_max = max(raw_max * 1.20, 1.0)

        x_pos = list(range(len(products)))
        names = [p.name[:16] for p in products]

        for i, p in enumerate(products):
            self._ax.bar(
                i, p.value,
                color=p.color + "bb",
                edgecolor=p.color,
                linewidth=0.8,
                width=0.5,
                zorder=2,
            )
            if p.bounds is not None:
                lo, hi = p.bounds
                self._ax.errorbar(
                    i, p.value,
                    yerr=[[max(0.0, p.value - lo)], [max(0.0, hi - p.value)]],
                    fmt="none",
                    color=p.color,
                    capsize=6,
                    elinewidth=1.5,
                    capthick=1.5,
                    zorder=3,
                )
            ann_y = (p.bounds[1] if p.bounds else p.value) + y_max * 0.025
            self._ax.text(
                i, ann_y,
                f"{p.value:.3f}",
                ha="center", va="bottom",
                color=p.color,
                fontsize=9,
                fontweight="bold",
            )

        self._ax.set_xlim(-0.6, len(products) - 0.4)
        self._ax.set_ylim(0, y_max)
        self._ax.set_xticks(x_pos)
        self._ax.set_xticklabels(names, fontsize=9, color=TEXT_SEC)
        self._ax.set_ylabel("kg CO₂eq / kg", color=TEXT_SEC, fontsize=9, labelpad=6)
        self._ax.tick_params(axis="y", colors=TEXT_SEC, labelsize=8)
        self._ax.tick_params(axis="x", length=0)
        self._ax.set_facecolor(SURFACE)
        for spine in ("top", "right"):
            self._ax.spines[spine].set_visible(False)
        self._ax.spines["bottom"].set_color(BORDER)
        self._ax.spines["left"].set_color(BORDER)

        self._fig.tight_layout(pad=1.5)
        self._mpl_canvas.draw_idle()

    def _draw_empty(self) -> None:
        self._ax.clear()
        self._ax.set_facecolor(SURFACE)
        for spine in self._ax.spines.values():
            spine.set_color(BORDER)
        self._ax.text(
            0.5, 0.5,
            "Configure products to see comparison",
            ha="center", va="center",
            color=TEXT_DIM,
            fontsize=11,
            transform=self._ax.transAxes,
        )
        self._ax.set_xticks([])
        self._ax.set_yticks([])
        self._fig.tight_layout(pad=1.5)
        self._mpl_canvas.draw_idle()
