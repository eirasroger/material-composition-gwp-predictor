"""
Scenario library dialogs: a name prompt (Save as / Rename) and the library
browser (search, load, rename, duplicate, delete).

Deliberately not a file manager. The list shows what is needed to pick the right
scenario — name, when it was last saved, how many products — and nothing else.
"""

from __future__ import annotations

from datetime import datetime
from typing import Callable, Dict, List, Optional

import customtkinter as ctk

from desktop_app import library
from desktop_app.ui.theme import (
    ACCENT, BG, BORDER, SURFACE, SURFACE_HI, TEXT_DIM, TEXT_PRI, TEXT_SEC, font,
)


def _fmt_when(iso: str) -> str:
    """`2026-07-26T09:12:00+00:00` → `26 Jul 2026, 11:12` in local time."""
    if not iso:
        return "—"
    try:
        stamp = datetime.fromisoformat(iso)
    except ValueError:
        return iso
    if stamp.tzinfo is not None:
        stamp = stamp.astimezone()
    return stamp.strftime("%d %b %Y, %H:%M")


class _Modal(ctk.CTkToplevel):
    """Shared plumbing: centred on the parent, modal, Escape closes."""

    def __init__(self, parent, title: str, width: int, height: int) -> None:
        super().__init__(parent)
        self.title(title)
        self.configure(fg_color=BG)
        self.resizable(False, False)

        parent.update_idletasks()
        x = parent.winfo_rootx() + (parent.winfo_width() - width) // 2
        y = parent.winfo_rooty() + (parent.winfo_height() - height) // 3
        self.geometry(f"{width}x{height}+{max(x, 0)}+{max(y, 0)}")

        self.transient(parent)
        self.bind("<Escape>", lambda _e: self._cancel())
        self.protocol("WM_DELETE_WINDOW", self._cancel)
        # grab_set() on a window that is not yet viewable raises TclError, and
        # CTkToplevel does its own deferred setup, so claim the grab on the next
        # idle cycle rather than immediately.
        self.after(120, self._claim_grab)

    def _claim_grab(self) -> None:
        try:
            self.grab_set()
        except Exception:
            pass

    def _cancel(self) -> None:
        self.destroy()


class NamePrompt(_Modal):
    """Single-line name entry. ``result`` is None when cancelled."""

    def __init__(
        self,
        parent,
        title: str = "Save scenario as",
        prompt: str = "Scenario name",
        initial: str = "",
        confirm_text: str = "Save",
    ) -> None:
        super().__init__(parent, title, width=420, height=180)
        self.result: Optional[str] = None

        ctk.CTkLabel(
            self, text=prompt, font=font(12), text_color=TEXT_SEC, anchor="w",
        ).pack(fill="x", padx=20, pady=(22, 6))

        self._entry = ctk.CTkEntry(self, font=font(13), height=34)
        self._entry.pack(fill="x", padx=20)
        self._entry.insert(0, initial)
        self._entry.bind("<Return>", lambda _e: self._confirm())
        self.after(160, self._focus_entry)

        buttons = ctk.CTkFrame(self, fg_color="transparent")
        buttons.pack(fill="x", padx=20, pady=(18, 0))
        ctk.CTkButton(
            buttons, text="Cancel", width=100, height=32,
            font=font(12), fg_color="transparent", border_width=1,
            border_color=BORDER, text_color=TEXT_SEC, hover_color=BORDER,
            command=self._cancel,
        ).pack(side="right")
        ctk.CTkButton(
            buttons, text=confirm_text, width=100, height=32,
            font=font(12, "bold"), command=self._confirm,
        ).pack(side="right", padx=(0, 8))

    def _focus_entry(self) -> None:
        try:
            self._entry.focus_set()
            self._entry.select_range(0, "end")
        except Exception:
            pass

    def _confirm(self) -> None:
        name = self._entry.get().strip()
        if not name:
            return
        self.result = name
        self.destroy()


def ask_name(
    parent,
    title: str = "Save scenario as",
    prompt: str = "Scenario name",
    initial: str = "",
    confirm_text: str = "Save",
) -> Optional[str]:
    dialog = NamePrompt(parent, title, prompt, initial, confirm_text)
    parent.wait_window(dialog)
    return dialog.result


class ScenarioLibraryDialog(_Modal):
    """
    Browser over the saved scenarios.

    ``on_load(scenario_id)`` is called after the dialog closes, so the caller's
    unsaved-changes prompt is not fighting this window for the grab.
    """

    def __init__(self, parent, on_load: Callable[[str], None]) -> None:
        super().__init__(parent, "My scenarios", width=620, height=520)
        self._on_load = on_load
        self._rows: List[Dict] = []
        self._selected_id: Optional[str] = None
        self._row_frames: Dict[str, ctk.CTkFrame] = {}

        ctk.CTkLabel(
            self, text="My scenarios", font=font(15, "bold"), anchor="w",
        ).pack(fill="x", padx=20, pady=(18, 2))
        ctk.CTkLabel(
            self,
            text="Saved compositions and end-of-life settings. "
                 "Opening one recalculates every indicator from scratch.",
            font=font(11), text_color=TEXT_SEC, anchor="w",
            justify="left", wraplength=560,
        ).pack(fill="x", padx=20, pady=(0, 10))

        self._search = ctk.CTkEntry(
            self, placeholder_text="Search scenarios ...", font=font(12), height=32,
        )
        self._search.pack(fill="x", padx=20)
        self._search.bind("<KeyRelease>", lambda _e: self._render())

        self._list = ctk.CTkScrollableFrame(
            self, fg_color=SURFACE, corner_radius=8,
            scrollbar_button_color=BORDER, scrollbar_button_hover_color=ACCENT,
            height=280,
        )
        self._list.pack(fill="both", expand=True, padx=20, pady=(10, 10))

        actions = ctk.CTkFrame(self, fg_color="transparent")
        actions.pack(fill="x", padx=20, pady=(0, 18))

        self._open_btn = ctk.CTkButton(
            actions, text="Open", width=90, height=34,
            font=font(12, "bold"), command=self._open_selected,
        )
        self._open_btn.pack(side="left")

        self._rename_btn = self._secondary(actions, "Rename", self._rename_selected)
        self._dup_btn    = self._secondary(actions, "Duplicate", self._duplicate_selected)
        self._delete_btn = self._secondary(actions, "Delete", self._delete_selected)

        ctk.CTkButton(
            actions, text="Close", width=90, height=34,
            font=font(12), fg_color="transparent", border_width=1,
            border_color=BORDER, text_color=TEXT_SEC, hover_color=BORDER,
            command=self._cancel,
        ).pack(side="right")

        self._refresh()

    @staticmethod
    def _secondary(master, text: str, command) -> ctk.CTkButton:
        button = ctk.CTkButton(
            master, text=text, width=90, height=34,
            font=font(12), fg_color="transparent", border_width=1,
            border_color=BORDER, text_color=TEXT_SEC, hover_color=BORDER,
            command=command,
        )
        button.pack(side="left", padx=(8, 0))
        return button

    # ── data ──────────────────────────────────────────────────────────────────

    def _refresh(self) -> None:
        self._rows = library.list_scenarios()
        if self._selected_id not in {r["id"] for r in self._rows}:
            self._selected_id = None
        self._render()

    def _render(self) -> None:
        for child in self._list.winfo_children():
            child.destroy()
        self._row_frames.clear()

        query = self._search.get().strip().lower()
        visible = [r for r in self._rows if query in r["name"].lower()] if query else self._rows

        if not visible:
            message = (
                "No scenarios match that search."
                if query else
                "No saved scenarios yet — build a comparison and choose Save as."
            )
            ctk.CTkLabel(
                self._list, text=message, font=font(12), text_color=TEXT_DIM,
                wraplength=500, justify="left",
            ).pack(padx=14, pady=20)
            self._sync_buttons()
            return

        for row in visible:
            self._render_row(row)
        self._sync_buttons()

    def _render_row(self, row: Dict) -> None:
        selected = row["id"] == self._selected_id
        frame = ctk.CTkFrame(
            self._list,
            fg_color=SURFACE_HI if selected else "transparent",
            corner_radius=6,
            border_width=1,
            border_color=ACCENT if selected else SURFACE,
        )
        frame.pack(fill="x", padx=6, pady=3)
        self._row_frames[row["id"]] = frame

        name = ctk.CTkLabel(
            frame, text=row["name"], font=font(13, "bold"),
            text_color=TEXT_PRI, anchor="w",
        )
        name.pack(fill="x", padx=12, pady=(9, 0))

        count = row["product_count"]
        meta = ctk.CTkLabel(
            frame,
            text=f"{count} product{'s' if count != 1 else ''}  ·  "
                 f"saved {_fmt_when(row['updated_at'])}",
            font=font(11), text_color=TEXT_SEC, anchor="w",
        )
        meta.pack(fill="x", padx=12, pady=(0, 9))

        for widget in (frame, name, meta):
            widget.bind("<Button-1>", lambda _e, i=row["id"]: self._select(i))
            widget.bind("<Double-Button-1>", lambda _e, i=row["id"]: self._open(i))

    def _select(self, scenario_id: str) -> None:
        self._selected_id = scenario_id
        self._render()

    def _sync_buttons(self) -> None:
        state = "normal" if self._selected_id else "disabled"
        for button in (self._open_btn, self._rename_btn, self._dup_btn, self._delete_btn):
            button.configure(state=state)

    # ── actions ───────────────────────────────────────────────────────────────

    def _open_selected(self) -> None:
        if self._selected_id:
            self._open(self._selected_id)

    def _open(self, scenario_id: str) -> None:
        # Close first: the caller may raise its own "discard unsaved work?"
        # prompt, which must not compete with this window's grab.
        self.destroy()
        self._on_load(scenario_id)

    def _rename_selected(self) -> None:
        row = self._row(self._selected_id)
        if row is None:
            return
        self._release_grab()
        name = ask_name(
            self, title="Rename scenario", prompt="New name",
            initial=row["name"], confirm_text="Rename",
        )
        self._claim_grab()
        if name:
            library.rename_scenario(row["id"], name)
            self._refresh()

    def _duplicate_selected(self) -> None:
        row = self._row(self._selected_id)
        if row is None:
            return
        doc = library.duplicate_scenario(row["id"])
        self._selected_id = doc["id"]
        self._refresh()

    def _delete_selected(self) -> None:
        row = self._row(self._selected_id)
        if row is None:
            return
        self._release_grab()
        confirmed = ConfirmDialog(
            self,
            title="Delete scenario",
            message=f'Delete "{row["name"]}"? This cannot be undone.',
            confirm_text="Delete",
        ).ask()
        self._claim_grab()
        if confirmed:
            library.delete_scenario(row["id"])
            self._selected_id = None
            self._refresh()

    def _row(self, scenario_id: Optional[str]) -> Optional[Dict]:
        return next((r for r in self._rows if r["id"] == scenario_id), None)

    def _release_grab(self) -> None:
        try:
            self.grab_release()
        except Exception:
            pass


class ConfirmDialog(_Modal):
    """Yes/No confirmation. ``ask()`` blocks and returns a bool."""

    def __init__(
        self,
        parent,
        title: str,
        message: str,
        confirm_text: str = "OK",
        cancel_text: str = "Cancel",
    ) -> None:
        super().__init__(parent, title, width=440, height=190)
        self._parent = parent
        self.result = False

        ctk.CTkLabel(
            self, text=message, font=font(12), text_color=TEXT_PRI,
            wraplength=390, justify="left", anchor="w",
        ).pack(fill="both", expand=True, padx=20, pady=(24, 8))

        buttons = ctk.CTkFrame(self, fg_color="transparent")
        buttons.pack(fill="x", padx=20, pady=(0, 20))
        ctk.CTkButton(
            buttons, text=cancel_text, width=100, height=32,
            font=font(12), fg_color="transparent", border_width=1,
            border_color=BORDER, text_color=TEXT_SEC, hover_color=BORDER,
            command=self._cancel,
        ).pack(side="right")
        ctk.CTkButton(
            buttons, text=confirm_text, width=100, height=32,
            font=font(12, "bold"), command=self._confirm,
        ).pack(side="right", padx=(0, 8))

    def _confirm(self) -> None:
        self.result = True
        self.destroy()

    def ask(self) -> bool:
        self._parent.wait_window(self)
        return self.result


class SaveDiscardCancel(_Modal):
    """
    Three-way prompt for unsaved work: Save / Discard / Cancel.

    ``ask()`` returns "save", "discard" or "cancel". Cancel is the default on
    Escape and on the window close button — the safe answer when the user
    dismisses a prompt about losing work.
    """

    def __init__(self, parent, message: str) -> None:
        super().__init__(parent, "Unsaved changes", width=460, height=200)
        self._parent = parent
        self.result = "cancel"

        ctk.CTkLabel(
            self, text=message, font=font(12), text_color=TEXT_PRI,
            wraplength=410, justify="left", anchor="w",
        ).pack(fill="both", expand=True, padx=20, pady=(24, 8))

        buttons = ctk.CTkFrame(self, fg_color="transparent")
        buttons.pack(fill="x", padx=20, pady=(0, 20))
        ctk.CTkButton(
            buttons, text="Cancel", width=100, height=32,
            font=font(12), fg_color="transparent", border_width=1,
            border_color=BORDER, text_color=TEXT_SEC, hover_color=BORDER,
            command=self._cancel,
        ).pack(side="right")
        ctk.CTkButton(
            buttons, text="Discard", width=100, height=32,
            font=font(12), fg_color="transparent", border_width=1,
            border_color=BORDER, text_color=TEXT_SEC, hover_color=BORDER,
            command=self._discard,
        ).pack(side="right", padx=(0, 8))
        ctk.CTkButton(
            buttons, text="Save", width=100, height=32,
            font=font(12, "bold"), command=self._save,
        ).pack(side="right", padx=(0, 8))

    def _save(self) -> None:
        self.result = "save"
        self.destroy()

    def _discard(self) -> None:
        self.result = "discard"
        self.destroy()

    def ask(self) -> str:
        self._parent.wait_window(self)
        return self.result
