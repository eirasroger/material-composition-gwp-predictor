"""
Prediction panel: large numeric display + a matplotlib horizontal gauge over
the trained range [GHG_MIN, GHG_MAX] (see ``src/config.py``). A single
``set_prediction(value)`` updates both. ``set_status(text)`` feeds the warning
line beneath (e.g. "Materials don't sum to 100 — predicted as if normalised.").
"""

from __future__ import annotations

import tkinter as tk

import customtkinter as ctk
import matplotlib
matplotlib.use("TkAgg")  # noqa: E402
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from desktop_app.ui.theme import (
    ACCENT, BORDER, SURFACE, TEXT_SEC, TEXT_DIM, font,
)
from src.config import GHG_MAX, GHG_MIN


class PredictionPanel(ctk.CTkFrame):
    def __init__(self, master, color: str = ACCENT) -> None:
        super().__init__(master, fg_color="transparent")
        self._color = color

        # ── header label ──────────────────────────────────────────────────────
        ctk.CTkLabel(
            self,
            text="Predicted greenhouse gas emissions",
            font=font(12),
            text_color=TEXT_SEC,
        ).pack(anchor="w", padx=24, pady=(28, 0))

        # ── big value ─────────────────────────────────────────────────────────
        self._value_label = ctk.CTkLabel(
            self,
            text="— kg CO₂eq / kg",
            font=font(34, "bold"),
            text_color=color,
        )
        self._value_label.pack(anchor="w", padx=24, pady=(4, 16))

        # ── divider ───────────────────────────────────────────────────────────
        divider = tk.Frame(self, height=1, bg=BORDER)
        divider.pack(fill="x", padx=24, pady=(0, 16))

        # ── matplotlib gauge ──────────────────────────────────────────────────
        ctk.CTkLabel(
            self,
            text="Position within training range",
            font=font(11),
            text_color=TEXT_DIM,
        ).pack(anchor="w", padx=24)

        self._fig = Figure(figsize=(4.4, 1.1), dpi=100)
        self._fig.patch.set_facecolor(SURFACE)
        self._ax = self._fig.add_subplot(111)
        self._configure_axes()

        canvas_host = tk.Frame(self, highlightthickness=0, bd=0, bg=SURFACE)
        canvas_host.pack(fill="x", padx=24, pady=(4, 0))
        self._mpl_canvas = FigureCanvasTkAgg(self._fig, master=canvas_host)
        widget = self._mpl_canvas.get_tk_widget()
        widget.pack(fill="x")
        widget.configure(bg=SURFACE, highlightthickness=0)

        # ── range label ───────────────────────────────────────────────────────
        self._range_label = ctk.CTkLabel(
            self,
            text="",
            text_color=TEXT_SEC,
            font=font(11),
            anchor="w",
        )
        self._range_label.pack(fill="x", padx=24, pady=(8, 0))

        # ── status ────────────────────────────────────────────────────────────
        self._status_label = ctk.CTkLabel(
            self,
            text="",
            text_color=TEXT_SEC,
            font=font(11),
            anchor="w",
            justify="left",
            wraplength=420,
        )
        self._status_label.pack(fill="x", padx=24, pady=(4, 24))

        self._draw_gauge(None)

    def set_prediction(
        self,
        value: float,
        bounds: "tuple[float, float] | None" = None,
    ) -> None:
        self._value_label.configure(text=f"{value:.3f} kg CO₂eq / kg")
        if bounds is not None:
            self._range_label.configure(
                text=f"Plausible range: {bounds[0]:.2f} – {bounds[1]:.2f} kg CO₂eq / kg"
            )
        else:
            self._range_label.configure(text="")
        self._draw_gauge(float(value), bounds)

    def clear_prediction(self) -> None:
        self._value_label.configure(text="— kg CO₂eq / kg")
        self._range_label.configure(text="")
        self._draw_gauge(None)

    def set_status(self, text: str) -> None:
        self._status_label.configure(text=text)

    def _configure_axes(self) -> None:
        self._ax.set_xlim(GHG_MIN, GHG_MAX)
        self._ax.set_ylim(0, 1)
        self._ax.set_yticks([])
        self._ax.set_xlabel(
            "kg CO₂eq / kg  (training range)",
            color=TEXT_SEC, fontsize=8, labelpad=4,
        )
        self._ax.tick_params(colors=TEXT_SEC, labelsize=8, length=3)
        self._ax.set_facecolor(SURFACE)
        for spine in ("top", "right", "left"):
            self._ax.spines[spine].set_visible(False)
        self._ax.spines["bottom"].set_color(BORDER)
        self._fig.subplots_adjust(left=0.04, right=0.98, top=0.82, bottom=0.42)

    def _draw_gauge(self, value, bounds=None) -> None:
        self._ax.clear()
        self._configure_axes()
        # Background track
        self._ax.barh(
            [0.5], [GHG_MAX - GHG_MIN], left=GHG_MIN, height=0.28,
            color=BORDER, edgecolor="none",
        )
        if value is not None:
            clamped = max(GHG_MIN, min(GHG_MAX, value))
            # Plausible range band (wider, behind the fill)
            if bounds is not None:
                r_low  = max(GHG_MIN, bounds[0])
                r_high = min(GHG_MAX, bounds[1])
                self._ax.barh(
                    [0.5], [r_high - r_low], left=r_low, height=0.52,
                    color=self._color + "28", edgecolor=self._color + "70", linewidth=0.8,
                    zorder=1,
                )
            # Filled portion of track
            self._ax.barh(
                [0.5], [clamped - GHG_MIN], left=GHG_MIN, height=0.28,
                color=self._color + "55", edgecolor="none", zorder=2,
            )
            # Needle line + dot
            self._ax.axvline(clamped, color=self._color, linewidth=2, alpha=0.9, zorder=3)
            self._ax.plot(
                [clamped], [0.5], marker="o", color=self._color,
                markersize=9, zorder=4, markeredgecolor=SURFACE, markeredgewidth=1.5,
            )
            if value > GHG_MAX:
                self._ax.text(
                    GHG_MAX, 0.92, "  > range",
                    color="#d44a4a", fontsize=8, va="top", ha="right",
                )
            elif value < GHG_MIN:
                self._ax.text(
                    GHG_MIN, 0.92, "< range  ",
                    color="#d44a4a", fontsize=8, va="top", ha="left",
                )
        self._mpl_canvas.draw_idle()


def _smoke() -> None:
    from pathlib import Path
    import customtkinter as ctk
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme(
        str(Path(__file__).resolve().parents[1] / "assets" / "theme_dark.json")
    )
    root = ctk.CTk()
    root.title("prediction_panel.py smoke test")
    root.geometry("520x380")
    root.configure(fg_color=SURFACE)

    panel = PredictionPanel(root)
    panel.pack(fill="both", expand=True)

    btns = ctk.CTkFrame(root, fg_color="transparent")
    btns.pack(fill="x", padx=24, pady=(0, 12))
    for v in (0.5, 2.3, 4.7, 8.1, 11.0):
        ctk.CTkButton(
            btns, text=f"{v}", width=60,
            command=lambda v=v: panel.set_prediction(v),
        ).pack(side="left", padx=4)
    ctk.CTkButton(
        btns, text="clear", width=60, command=panel.clear_prediction,
    ).pack(side="left", padx=4)

    panel.set_status("Materials sum to 87% — predicted as if normalised.")
    root.mainloop()


if __name__ == "__main__":
    _smoke()
