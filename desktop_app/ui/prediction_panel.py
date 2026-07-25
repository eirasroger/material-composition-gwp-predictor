"""
Prediction panel: large numeric display + a matplotlib gauge, for ONE target.

The gauge shows the prediction against the distribution of real products in the
same category (p25 / median / p75), not against the trained value range. A
linear track over [value_min, value_max] is meaningless for most indicators --
``adpf_a1a3`` spans 0-650 with a median of 8.5, so every ordinary product would
pin to the far left. Where no reference distribution exists (the zero-inflated
C3/C4 stages, whose median is 0) it falls back to the trained range.

``set_target(...)`` re-labels the panel for the selected indicator; everything
that was hardcoded to GHG comes from the target manifest instead.
"""

from __future__ import annotations

import tkinter as tk
from typing import Optional

import customtkinter as ctk
import matplotlib
matplotlib.use("TkAgg")  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from desktop_app.ui.theme import (
    ACCENT, BORDER, SURFACE, TEXT_SEC, TEXT_DIM, TEXT_PRI, font,
    fmt_value as _fmt, stage_color, stage_label, stage_sort_key,
    status_vs_typical,
)


class PredictionPanel(ctk.CTkFrame):
    def __init__(
        self,
        master,
        color: str = ACCENT,
        display_name: str = "GHG Total",
        unit: str = "kg CO2-eq/kg",
        value_min: float = 0.0,
        value_max: float = 10.0,
    ) -> None:
        super().__init__(master, fg_color="transparent")
        self._color        = color
        self._display_name = display_name
        self._unit         = unit
        self._value_min    = value_min
        self._value_max    = value_max
        self._reference    = None

        # ── header label ──────────────────────────────────────────────────────
        self._header_label = ctk.CTkLabel(
            self,
            text=f"Predicted {display_name}",
            font=font(12),
            text_color=TEXT_SEC,
        )
        self._header_label.pack(anchor="w", padx=24, pady=(28, 0))

        # ── big value ─────────────────────────────────────────────────────────
        self._value_label = ctk.CTkLabel(
            self,
            text=f"— {unit}",
            font=font(34, "bold"),
            text_color=color,
        )
        self._value_label.pack(anchor="w", padx=24, pady=(4, 16))

        # ── divider ───────────────────────────────────────────────────────────
        divider = tk.Frame(self, height=1, bg=BORDER)
        divider.pack(fill="x", padx=24, pady=(0, 16))

        # ── matplotlib gauge ──────────────────────────────────────────────────
        caption_row = ctk.CTkFrame(self, fg_color="transparent")
        caption_row.pack(anchor="w", fill="x", padx=24)
        # Verdict is its own bold, colour-coded label; the rest stays dim so the
        # verdict is what the eye lands on.
        self._verdict_label = ctk.CTkLabel(
            caption_row, text="", font=font(12, "bold"),
            text_color=TEXT_PRI, anchor="w",
        )
        self._verdict_label.pack(side="left")
        self._gauge_caption = ctk.CTkLabel(
            caption_row,
            text="Compared with real products",
            font=font(11),
            text_color=TEXT_DIM,
            anchor="w",
            justify="left",
        )
        self._gauge_caption.pack(side="left", padx=(6, 0))

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
        # Opens itself as soon as there is something to show. Sticky once the
        # user hides it, so it does not keep re-opening on every keystroke.
        self._breakdown_user_collapsed = False
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

    def set_target(
        self,
        display_name: str,
        unit: str,
        value_min: float,
        value_max: float,
        color: str | None = None,
    ) -> None:
        """Re-point the panel at another indicator/stage."""
        self._display_name = display_name
        self._unit         = unit
        self._value_min    = value_min
        self._value_max    = value_max
        if color is not None:
            self._color = color
            self._value_label.configure(text_color=color)
        self._header_label.configure(text=f"Predicted {display_name}")
        self.clear_prediction()

    def set_prediction(
        self,
        value: float,
        bounds: "tuple[float, float] | None" = None,
        breakdown: "list[dict] | None" = None,
        reference=None,
    ) -> None:
        self._reference = reference
        self._value_label.configure(text=f"{_fmt(value)} {self._unit}")
        if bounds is not None:
            # Not a full confidence interval: it is the p25-p75 of this
            # category's residuals, so name it for what it is.
            self._range_label.configure(
                text=(
                    f"Middle 50% of likely values: "
                    f"{_fmt(bounds[0])} – {_fmt(bounds[1])} {self._unit}"
                )
            )
        else:
            self._range_label.configure(text="")

        if reference is not None:
            # Thresholds are the category's own middle 50%, the same band the
            # gauge and the summary radar shade — so the wording can never
            # contradict the picture. The prediction's own range decides whether
            # the claim is stated outright or hedged.
            low, high = bounds if bounds else (None, None)
            wording, _, colour = status_vs_typical(
                value, reference.p25, reference.p75, low, high,
            )
            ratio_text = f"{reference.ratio(value):.2g}× median"
            if bounds is not None:
                ratio_text += (
                    f" ({reference.ratio(bounds[0]):.2g}–{reference.ratio(bounds[1]):.2g}×)"
                )
            self._verdict_label.configure(
                text=f"{ratio_text} — {wording}", text_color=colour,
            )
            self._gauge_caption.configure(text=f"vs. {reference.label}")
        else:
            self._verdict_label.configure(text="")
            self._gauge_caption.configure(
                text=f"Position within trained range ({_fmt(self._value_min)} – {_fmt(self._value_max)})"
            )
        self._draw_gauge(float(value), bounds)

        self._breakdown_data = breakdown or None
        self._breakdown_toggle.configure(
            state="normal" if self._breakdown_data else "disabled"
        )
        if self._breakdown_data is None:
            if self._breakdown_expanded:
                self._set_breakdown_expanded(False)
        elif not self._breakdown_expanded and not self._breakdown_user_collapsed:
            self._set_breakdown_expanded(True)
        elif self._breakdown_expanded:
            self._draw_breakdown()

    def clear_prediction(self) -> None:
        self._value_label.configure(text=f"— {self._unit}")
        self._range_label.configure(text="")
        self._reference = None
        self._verdict_label.configure(text="")
        self._gauge_caption.configure(text="Compared with real products")
        self._draw_gauge(None)
        self._breakdown_data = None
        self._breakdown_toggle.configure(state="disabled")
        if self._breakdown_expanded:
            self._set_breakdown_expanded(False)

    def set_status(self, text: str) -> None:
        self._status_label.configure(text=text)

    # ── lifecycle breakdown ───────────────────────────────────────────────────

    def _toggle_breakdown(self) -> None:
        expanded = not self._breakdown_expanded
        self._breakdown_user_collapsed = not expanded
        self._set_breakdown_expanded(expanded)

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
        labels = [stage_label(s["stage_key"]) for s in stages]

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

        # Pad relative to the data's own magnitude -- a fixed 0.02 is invisible
        # on MJ/kg and enormous on kg PO4-eq/kg.
        span = max(abs(v) for v in values) or 1.0
        pad = 0.04 * span
        for xi, v, err_hi, err_lo in zip(x, values, yerr_high, yerr_low):
            label_y = v + err_hi + pad if v >= 0 else v - err_lo - pad
            ax.text(
                xi, label_y, _fmt(v), ha="center",
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

    # The gauge works in log10(value / reference median) when a reference exists,
    # so the axis is "×0.1 … median … ×10" rather than a linear span the data
    # never fills. _DECADES matches the summary radar so the two agree.
    _DECADES = 1.0

    def _scale(self):
        """(lo, hi, to_x, tick_positions, tick_labels, axis_label) for the gauge."""
        ref = self._reference
        if ref is not None:
            d = self._DECADES
            def to_x(v: float) -> float:
                if v <= 0:
                    return -d
                return max(-d, min(d, np.log10(v / ref.p50)))
            ticks  = [-d, np.log10(0.5), 0.0, np.log10(2.0), d]
            labels = ["×0.1", "×0.5", "median", "×2", "×10"]
            return -d, d, to_x, ticks, labels, f"vs. {ref.label}"

        lo, hi = self._value_min, self._value_max
        def to_x(v: float) -> float:
            return max(lo, min(hi, v))
        return lo, hi, to_x, None, None, f"{self._unit}  (trained range)"

    def _configure_axes(self) -> None:
        lo, hi, _, ticks, labels, axis_label = self._scale()
        self._ax.set_xlim(lo, hi)
        self._ax.set_ylim(0, 1)
        self._ax.set_yticks([])
        if ticks is not None:
            self._ax.set_xticks(ticks)
            self._ax.set_xticklabels(labels)
        self._ax.set_xlabel(axis_label, color=TEXT_SEC, fontsize=8, labelpad=4)
        self._ax.tick_params(colors=TEXT_SEC, labelsize=8, length=3)
        self._ax.set_facecolor(SURFACE)
        for spine in ("top", "right", "left"):
            self._ax.spines[spine].set_visible(False)
        self._ax.spines["bottom"].set_color(BORDER)
        self._fig.subplots_adjust(left=0.04, right=0.98, top=0.82, bottom=0.42)

    def _draw_gauge(self, value, bounds=None) -> None:
        self._ax.clear()
        self._configure_axes()
        lo, hi, to_x, _, _, _ = self._scale()
        ref = self._reference

        # Background track
        self._ax.barh(
            [0.5], [hi - lo], left=lo, height=0.28, color=BORDER, edgecolor="none",
        )
        # Typical range (p25-p75) of real products, and the median marker.
        if ref is not None:
            x25, x75 = to_x(ref.p25), to_x(ref.p75)
            self._ax.barh(
                [0.5], [x75 - x25], left=x25, height=0.28,
                color=TEXT_DIM, alpha=0.55, edgecolor="none", zorder=1,
            )
            self._ax.axvline(0.0, color=TEXT_SEC, linewidth=1.0, alpha=0.9, zorder=2)

        if value is not None:
            x = to_x(value)
            if bounds is not None:
                x_low, x_high = to_x(bounds[0]), to_x(bounds[1])
                if x_high > x_low:
                    self._ax.barh(
                        [0.5], [x_high - x_low], left=x_low, height=0.52,
                        color=self._color + "28", edgecolor=self._color + "70",
                        linewidth=0.8, zorder=3,
                    )
            self._ax.axvline(x, color=self._color, linewidth=2, alpha=0.9, zorder=4)
            self._ax.plot(
                [x], [0.5], marker="o", color=self._color,
                markersize=9, zorder=5, markeredgecolor=SURFACE, markeredgewidth=1.5,
            )
            # Flag a clamped needle so the rail is never read as the real value.
            off_hi = (ref is not None and value / ref.p50 > 10 ** self._DECADES) \
                or (ref is None and value > self._value_max)
            off_lo = (ref is not None and 0 < value / ref.p50 < 10 ** -self._DECADES) \
                or (ref is None and value < self._value_min)
            if off_hi:
                self._ax.text(
                    hi, 0.92, "  off scale", color="#d44a4a",
                    fontsize=8, va="top", ha="right",
                )
            elif off_lo:
                self._ax.text(
                    lo, 0.92, "off scale  ", color="#d44a4a",
                    fontsize=8, va="top", ha="left",
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
