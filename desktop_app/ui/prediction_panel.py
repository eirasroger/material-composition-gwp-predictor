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
    ACCENT, BORDER, SURFACE, TEXT_SEC, TEXT_DIM, TEXT_PRI, font, stage_color, stage_sort_key,
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
        self._status_label.pack(fill="x", padx=24, pady=(4, 12))

        # ── lifecycle breakdown (collapsed by default) ───────────────────────────
        self._breakdown_data: "list[dict] | None" = None
        self._breakdown_expanded = False
        self._breakdown_fig = None
        self._breakdown_canvas = None

        self._breakdown_toggle = ctk.CTkButton(
            self,
            text="Show lifecycle breakdown  ▸",
            font=font(11),
            height=26,
            fg_color="transparent",
            border_width=1,
            border_color=BORDER,
            text_color=TEXT_SEC,
            hover_color=BORDER,
            command=self._toggle_breakdown,
            state="disabled",
        )
        self._breakdown_toggle.pack(anchor="w", padx=24, pady=(0, 4))

        self._breakdown_frame = ctk.CTkFrame(self, fg_color="transparent")
        # not packed until expanded

        ctk.CTkLabel(
            self._breakdown_frame,
            text=(
                "Estimated from material composition only. Excludes transport, "
                "construction, and use-phase impacts, which depend on process\n"
                "choices the model doesn't see."
            ),
            font=font(10),
            text_color=TEXT_DIM,
            justify="left",
            anchor="w",
        ).pack(anchor="w", padx=0, pady=(0, 6))

        self._breakdown_canvas_host = tk.Frame(
            self._breakdown_frame, highlightthickness=0, bd=0, bg=SURFACE,
        )
        self._breakdown_canvas_host.pack(fill="x")

        self._draw_gauge(None)

    def set_prediction(
        self,
        value: float,
        bounds: "tuple[float, float] | None" = None,
        breakdown: "list[dict] | None" = None,
    ) -> None:
        self._value_label.configure(text=f"{value:.3f} kg CO₂eq / kg")
        if bounds is not None:
            self._range_label.configure(
                text=f"Plausible range: {bounds[0]:.2f} – {bounds[1]:.2f} kg CO₂eq / kg"
            )
        else:
            self._range_label.configure(text="")
        self._draw_gauge(float(value), bounds)

        self._breakdown_data = breakdown or None
        self._breakdown_toggle.configure(
            state="normal" if self._breakdown_data else "disabled"
        )
        if self._breakdown_data is None and self._breakdown_expanded:
            self._set_breakdown_expanded(False)
        elif self._breakdown_expanded:
            self._draw_breakdown()

    def clear_prediction(self) -> None:
        self._value_label.configure(text="— kg CO₂eq / kg")
        self._range_label.configure(text="")
        self._draw_gauge(None)
        self._breakdown_data = None
        self._breakdown_toggle.configure(state="disabled")
        if self._breakdown_expanded:
            self._set_breakdown_expanded(False)

    def set_status(self, text: str) -> None:
        self._status_label.configure(text=text)

    # ── lifecycle breakdown ───────────────────────────────────────────────────

    def _toggle_breakdown(self) -> None:
        self._set_breakdown_expanded(not self._breakdown_expanded)

    def _set_breakdown_expanded(self, expanded: bool) -> None:
        self._breakdown_expanded = expanded
        if expanded:
            self._breakdown_toggle.configure(text="Hide lifecycle breakdown  ▾")
            self._breakdown_frame.pack(fill="x", padx=24, pady=(0, 20))
            self._draw_breakdown()
        else:
            self._breakdown_toggle.configure(text="Show lifecycle breakdown  ▸")
            self._breakdown_frame.pack_forget()

    def _draw_breakdown(self) -> None:
        if not self._breakdown_data:
            return

        stages = sorted(self._breakdown_data, key=lambda s: stage_sort_key(s["stage_key"]))
        n = len(stages)

        if self._breakdown_fig is None:
            self._breakdown_fig = Figure(figsize=(4.4, 2.4), dpi=100)
            self._breakdown_fig.patch.set_facecolor(SURFACE)
            self._breakdown_ax = self._breakdown_fig.add_subplot(111)
            self._breakdown_canvas = FigureCanvasTkAgg(
                self._breakdown_fig, master=self._breakdown_canvas_host
            )
            widget = self._breakdown_canvas.get_tk_widget()
            widget.pack(fill="x")
            widget.configure(bg=SURFACE, highlightthickness=0)

        ax = self._breakdown_ax
        ax.clear()

        x = list(range(n))
        values = [s["value"] for s in stages]
        colors = [stage_color(s["stage_key"]) for s in stages]
        labels = [s["display_name"].replace("GHG ", "") for s in stages]

        # Asymmetric error bars from each stage's own plausible range.
        yerr_low, yerr_high = [], []
        for s in stages:
            b = s.get("bounds")
            if b is None:
                yerr_low.append(0.0)
                yerr_high.append(0.0)
            else:
                yerr_low.append(max(0.0, s["value"] - b[0]))
                yerr_high.append(max(0.0, b[1] - s["value"]))

        ax.bar(x, values, color=colors, width=0.6, zorder=2)
        ax.errorbar(
            x, values, yerr=[yerr_low, yerr_high],
            fmt="none", ecolor=TEXT_SEC, elinewidth=1, capsize=3, zorder=3,
        )
        ax.axhline(0, color=BORDER, linewidth=1, zorder=1)

        pad = 0.02 * max(1.0, max(abs(v) for v in values))
        for xi, v, err_hi, err_lo in zip(x, values, yerr_high, yerr_low):
            label_y = v + err_hi + pad if v >= 0 else v - err_lo - pad
            ax.text(
                xi, label_y, f"{v:.2f}", ha="center",
                va="bottom" if v >= 0 else "top",
                color=TEXT_PRI, fontsize=8,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(labels, color=TEXT_SEC, fontsize=9)
        ax.set_ylabel(
            stages[0].get("unit", ""), color=TEXT_SEC, fontsize=8, labelpad=4,
        )
        ax.tick_params(colors=TEXT_SEC, labelsize=8, length=3)
        ax.set_facecolor(SURFACE)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_color(BORDER)
        ax.spines["bottom"].set_color(BORDER)
        self._breakdown_fig.subplots_adjust(left=0.16, right=0.96, top=0.92, bottom=0.16)

        self._breakdown_canvas.draw_idle()

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
