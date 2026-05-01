"""
Reusable UI building blocks: SearchableDropdown, PercentSlider, SumIndicator.
"""

from __future__ import annotations

import tkinter as tk
from typing import Callable, Iterable, List, Optional

import customtkinter as ctk


class SearchableDropdown(ctk.CTkFrame):
    """
    A text entry whose dropdown shows substring-matches from ``values``.

    The popup is a borderless ``Toplevel`` anchored beneath the entry. Selection
    fires ``command(value)``. Press Escape to dismiss; the popup also dismisses
    on a delayed FocusOut so a click on a result still registers.
    """

    def __init__(
        self,
        master,
        values: Iterable[str],
        command: Optional[Callable[[str], None]] = None,
        width: int = 220,
        height: int = 28,
        max_results: int = 40,
        placeholder: str = "",
    ) -> None:
        super().__init__(master, fg_color="transparent")
        self._all_values: List[str] = sorted(set(values), key=lambda s: s.lower())
        self._command = command
        self._max_results = max_results

        self.entry = ctk.CTkEntry(self, width=width, height=height, placeholder_text=placeholder)
        self.entry.pack(fill="x", expand=True)
        self.entry.bind("<KeyRelease>", self._on_key)
        self.entry.bind("<FocusIn>",   lambda _e: self._show_popup())
        self.entry.bind("<FocusOut>",  lambda _e: self.after(180, self._maybe_hide))
        self.entry.bind("<Escape>",    lambda _e: self._hide_popup())

        self._popup: Optional[tk.Toplevel] = None
        self._popup_frame: Optional[ctk.CTkScrollableFrame] = None

    # ── public API ────────────────────────────────────────────────────────────
    def get(self) -> str:
        return self.entry.get().strip()

    def set(self, value: str) -> None:
        self.entry.delete(0, "end")
        self.entry.insert(0, value)

    def set_values(self, values: Iterable[str]) -> None:
        self._all_values = sorted(set(values), key=lambda s: s.lower())
        if self._popup is not None:
            self._refresh_items()

    # ── popup management ──────────────────────────────────────────────────────
    def _show_popup(self) -> None:
        if self._popup is not None and self._popup.winfo_exists():
            self._reposition()
            self._refresh_items()
            return

        self._popup = tk.Toplevel(self)
        self._popup.wm_overrideredirect(True)
        self._popup.attributes("-topmost", True)
        self._popup_frame = ctk.CTkScrollableFrame(self._popup, height=200)
        self._popup_frame.pack(fill="both", expand=True)
        self._reposition()
        self._refresh_items()

    def _reposition(self) -> None:
        if self._popup is None or not self._popup.winfo_exists():
            return
        self.update_idletasks()
        x = self.entry.winfo_rootx()
        y = self.entry.winfo_rooty() + self.entry.winfo_height()
        w = max(220, self.entry.winfo_width())
        self._popup.geometry(f"{w}x220+{x}+{y}")

    def _refresh_items(self) -> None:
        if self._popup_frame is None:
            return
        for child in self._popup_frame.winfo_children():
            child.destroy()

        query = self.entry.get().strip().lower()
        if not query:
            matches = self._all_values[: self._max_results]
        else:
            matches = [v for v in self._all_values if query in v.lower()][: self._max_results]

        if not matches:
            ctk.CTkLabel(self._popup_frame, text="(no matches)", text_color="gray").pack(padx=6, pady=6)
            return

        for value in matches:
            ctk.CTkButton(
                self._popup_frame,
                text=value,
                anchor="w",
                height=24,
                fg_color="transparent",
                text_color=("black", "white"),
                hover_color=("gray85", "gray30"),
                command=lambda v=value: self._select(v),
            ).pack(fill="x", padx=2, pady=1)

    def _select(self, value: str) -> None:
        self.set(value)
        self._hide_popup()
        if self._command is not None:
            self._command(value)

    def _maybe_hide(self) -> None:
        # If the focus moved into the popup (button hover/click), keep it open.
        focused = self.focus_get()
        if (
            focused is not None
            and self._popup is not None
            and self._popup.winfo_exists()
            and focused.winfo_toplevel() == self._popup
        ):
            return
        self._hide_popup()

    def _hide_popup(self) -> None:
        if self._popup is not None:
            try:
                self._popup.destroy()
            except Exception:
                pass
        self._popup = None
        self._popup_frame = None

    def _on_key(self, event) -> None:
        if event.keysym == "Escape":
            self._hide_popup()
            return
        self._show_popup()
        self._refresh_items()

    def destroy(self) -> None:
        self._hide_popup()
        super().destroy()


class PercentSlider(ctk.CTkFrame):
    """Label + 0-100 horizontal slider + numeric readout. Fires ``command(value)``."""

    def __init__(
        self,
        master,
        label: str,
        initial: float = 0.0,
        command: Optional[Callable[[float], None]] = None,
        label_width: int = 120,
    ) -> None:
        super().__init__(master, fg_color="transparent")
        self._command = command

        self.label = ctk.CTkLabel(self, text=label, width=label_width, anchor="w")
        self.label.pack(side="left", padx=(0, 8))

        self._var = tk.DoubleVar(value=float(initial))
        self.slider = ctk.CTkSlider(
            self, from_=0.0, to=100.0, number_of_steps=200,
            variable=self._var, command=self._on_slider,
        )
        self.slider.pack(side="left", fill="x", expand=True, padx=(0, 8))

        self.readout = ctk.CTkLabel(self, text=self._fmt(initial), width=52, anchor="e")
        self.readout.pack(side="left")

    def get(self) -> float:
        return float(self._var.get())

    def set(self, value: float) -> None:
        v = max(0.0, min(100.0, float(value)))
        self._var.set(v)
        self.readout.configure(text=self._fmt(v))

    def _on_slider(self, value: float) -> None:
        self.readout.configure(text=self._fmt(value))
        if self._command is not None:
            self._command(float(value))

    @staticmethod
    def _fmt(v: float) -> str:
        return f"{float(v):5.1f} %"


class SumIndicator(ctk.CTkLabel):
    """Coloured ``Total: NN.N %`` label. Green at 100, amber within 5, red otherwise."""

    GREEN = ("#2a8a3a", "#3fb04f")
    AMBER = ("#a36b00", "#d49b3a")
    RED   = ("#a32a2a", "#d44a4a")

    def __init__(self, master) -> None:
        super().__init__(master, text="Total: 0.0 %", anchor="w")
        self.update_total(0.0)

    def update_total(self, total: float) -> None:
        diff = abs(total - 100.0)
        if diff < 0.05:
            colour = self.GREEN
        elif diff <= 5.0:
            colour = self.AMBER
        else:
            colour = self.RED
        self.configure(text=f"Total: {total:5.1f} %", text_color=colour)


# ──────────────────────────────────────────────────────────────────────────────
# Visual smoke test
# ──────────────────────────────────────────────────────────────────────────────
def _smoke() -> None:
    ctk.set_appearance_mode("system")
    root = ctk.CTk()
    root.title("widgets.py smoke test")
    root.geometry("520x420")

    ctk.CTkLabel(root, text="Searchable dropdown:").pack(anchor="w", padx=12, pady=(12, 0))
    sd = SearchableDropdown(
        root,
        values=["Apple", "Banana", "Cherry", "Date", "Elderberry",
                "Fig", "Grape", "Honeydew", "Kiwi", "Lemon", "Mango",
                "Nectarine", "Orange", "Papaya", "Quince", "Raspberry"],
        command=lambda v: print("picked:", v),
    )
    sd.pack(fill="x", padx=12, pady=4)

    ctk.CTkLabel(root, text="Percent slider:").pack(anchor="w", padx=12, pady=(12, 0))
    ps = PercentSlider(root, label="Recycling", initial=42.0,
                       command=lambda v: print(f"slider: {v:.1f}"))
    ps.pack(fill="x", padx=12, pady=4)

    ctk.CTkLabel(root, text="Sum indicator (click buttons):").pack(anchor="w", padx=12, pady=(12, 0))
    si = SumIndicator(root)
    si.pack(fill="x", padx=12, pady=4)
    btns = ctk.CTkFrame(root, fg_color="transparent")
    btns.pack(fill="x", padx=12)
    for v in (0, 50, 95, 100, 110):
        ctk.CTkButton(btns, text=str(v), width=60,
                      command=lambda v=v: si.update_total(v)).pack(side="left", padx=4)

    root.mainloop()


if __name__ == "__main__":
    _smoke()
