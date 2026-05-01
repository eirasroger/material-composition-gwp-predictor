"""
Background update checker against GitHub Releases.

Call check_for_updates(parent, current_version) once after the main window
is created. It spawns a daemon thread; if a newer release is found it
schedules a dialog on the Tkinter main thread — the app never blocks on startup.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import threading
import urllib.request
from typing import Optional, Tuple

import customtkinter as ctk

GITHUB_REPO = "eirasroger/material-composition-gwp-predictor"
_API_URL = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
_REQUEST_TIMEOUT = 8


def check_for_updates(parent: ctk.CTk, current_version: str) -> None:
    threading.Thread(
        target=_check_worker,
        args=(parent, current_version),
        daemon=True,
    ).start()


# ── internals ─────────────────────────────────────────────────────────────────

def _parse_version(v: str) -> tuple:
    try:
        return tuple(int(x) for x in v.lstrip("v").split("."))
    except ValueError:
        return (0, 0, 0)


def _fetch_latest() -> Optional[Tuple[str, str]]:
    """Return (version, installer_url) for the latest GitHub release, or None."""
    try:
        req = urllib.request.Request(
            _API_URL,
            headers={"User-Agent": "GHGPredictor-updater"},
        )
        with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT) as resp:
            data = json.loads(resp.read())
        tag = data.get("tag_name", "")
        for asset in data.get("assets", []):
            name = asset.get("name", "")
            if name.startswith("GHGPredictorSetup") and name.endswith(".exe"):
                return tag.lstrip("v"), asset["browser_download_url"]
        return None
    except Exception:
        return None


def _check_worker(parent: ctk.CTk, current_version: str) -> None:
    result = _fetch_latest()
    if result is None:
        return
    latest, url = result
    if _parse_version(latest) <= _parse_version(current_version):
        return
    parent.after(0, lambda: _show_dialog(parent, latest, url))


def _show_dialog(parent: ctk.CTk, version: str, url: str) -> None:
    dialog = ctk.CTkToplevel(parent)
    dialog.title("Update Available")
    dialog.resizable(False, False)
    dialog.transient(parent)
    dialog.grab_set()

    # Center on parent
    w, h = 420, 170
    px = parent.winfo_x() + (parent.winfo_width() - w) // 2
    py = parent.winfo_y() + (parent.winfo_height() - h) // 2
    dialog.geometry(f"{w}x{h}+{px}+{py}")

    msg = ctk.CTkLabel(
        dialog,
        text=f"Version {version} is available.\nWould you like to update now?",
        font=ctk.CTkFont(size=14),
        justify="center",
    )
    msg.pack(pady=(28, 18))

    btn_row = ctk.CTkFrame(dialog, fg_color="transparent")
    btn_row.pack()

    update_btn = ctk.CTkButton(btn_row, text="Update", width=130)
    update_btn.grid(row=0, column=0, padx=10)

    skip_btn = ctk.CTkButton(
        btn_row, text="Not now", width=130,
        fg_color="gray50", hover_color="gray40",
        command=dialog.destroy,
    )
    skip_btn.grid(row=0, column=1, padx=10)

    def _on_update() -> None:
        update_btn.configure(text="Downloading...", state="disabled")
        skip_btn.configure(state="disabled")
        threading.Thread(
            target=_download_and_launch,
            args=(url, parent, msg, skip_btn),
            daemon=True,
        ).start()

    update_btn.configure(command=_on_update)


def _download_and_launch(
    url: str,
    parent: ctk.CTk,
    msg: ctk.CTkLabel,
    skip_btn: ctk.CTkButton,
) -> None:
    try:
        tmpdir = tempfile.mkdtemp()
        dest = os.path.join(tmpdir, "GHGPredictorSetup.exe")
        urllib.request.urlretrieve(url, dest)
        subprocess.Popen([dest])
        parent.after(0, parent.destroy)
    except Exception as exc:
        def _on_fail() -> None:
            msg.configure(text=f"Download failed:\n{exc}")
            skip_btn.configure(state="normal", text="Close")
        parent.after(0, _on_fail)
