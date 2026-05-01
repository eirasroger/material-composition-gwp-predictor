"""
Prediction panel: large numeric display + a matplotlib horizontal gauge over
the trained range [GHG_MIN, GHG_MAX] (see ``src/config.py:56-57``). A single
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

from src.config import GHG_MAX, GHG_MIN


class PredictionPanel(ctk.CTkFrame):
    def __init__(self, master) -> None:
        super().__init__(master)

        ctk.CTkLabel(
            self, text="Predicted GHG", font=ctk.CTkFont(size=14, weight="bold"),
        ).pack(anchor="w", padx=12, pady=(12, 0))

        self._value_label = ctk.CTkLabel(
            self, text="— kg CO₂eq / kg",
            font=ctk.CTkFont(size=32, weight="bold"),
        )
        self._value_label.pack(anchor="w", padx=12, pady=(2, 6))

        self._fig = Figure(figsize=(4.4, 1.2), dpi=100)
        self._ax  = self._fig.add_subplot(111)
        self._configure_axes()

        canvas_host = tk.Frame(self, highlightthickness=0, bd=0)
        canvas_host.pack(fill="x", padx=12, pady=(0, 6))
        self._mpl_canvas = FigureCanvasTkAgg(self._fig, master=canvas_host)
        self._mpl_canvas.get_tk_widget().pack(fill="x")

        self._status_label = ctk.CTkLabel(
            self, text="", text_color="gray", anchor="w", justify="left",
            wraplength=420,
        )
        self._status_label.pack(fill="x", padx=12, pady=(0, 12))

        self._draw_gauge(None)

    def set_prediction(self, value: float) -> None:
        self._value_label.configure(text=f"{value:.3f} kg CO₂eq / kg")
        self._draw_gauge(float(value))

    def clear_prediction(self) -> None:
        self._value_label.configure(text="— kg CO₂eq / kg")
        self._draw_gauge(None)

    def set_status(self, text: str) -> None:
        self._status_label.configure(text=text)

    def _configure_axes(self) -> None:
        self._ax.set_xlim(GHG_MIN, GHG_MAX)
        self._ax.set_ylim(0, 1)
        self._ax.set_yticks([])
        self._ax.set_xlabel("kg CO₂eq / kg (training range)")
        for spine in ("top", "right", "left"):
            self._ax.spines[spine].set_visible(False)
        self._fig.subplots_adjust(left=0.05, right=0.97, top=0.85, bottom=0.40)

    def _draw_gauge(self, value) -> None:
        self._ax.clear()
        self._configure_axes()
        self._ax.barh(
            [0.5], [GHG_MAX - GHG_MIN], left=GHG_MIN, height=0.35,
            color="#d9d9d9", edgecolor="none",
        )
        if value is not None:
            clamped = max(GHG_MIN, min(GHG_MAX, value))
            self._ax.axvline(clamped, color="#2a6fdb", linewidth=3)
            self._ax.plot([clamped], [0.5], marker="o", color="#2a6fdb",
                          markersize=10, zorder=3)
            if value > GHG_MAX:
                self._ax.text(GHG_MAX, 0.92, "  > range",
                              color="#a32a2a", fontsize=8, va="top", ha="right")
            elif value < GHG_MIN:
                self._ax.text(GHG_MIN, 0.92, "< range  ",
                              color="#a32a2a", fontsize=8, va="top", ha="left")
        self._mpl_canvas.draw_idle()


def _smoke() -> None:
    ctk.set_appearance_mode("system")
    root = ctk.CTk()
    root.title("prediction_panel.py smoke test")
    root.geometry("520x340")

    panel = PredictionPanel(root)
    panel.pack(fill="both", expand=True, padx=12, pady=12)

    btns = ctk.CTkFrame(root, fg_color="transparent")
    btns.pack(fill="x", padx=12, pady=(0, 12))
    for v in (0.5, 2.3, 4.7, 8.1, 11.0):
        ctk.CTkButton(
            btns, text=f"{v}", width=60,
            command=lambda v=v: panel.set_prediction(v),
        ).pack(side="left", padx=4)
    ctk.CTkButton(
        btns, text="clear", width=60,
        command=panel.clear_prediction,
    ).pack(side="left", padx=4)

    panel.set_status("Materials sum to 87% — predicted as if normalised.")
    root.mainloop()


if __name__ == "__main__":
    _smoke()
