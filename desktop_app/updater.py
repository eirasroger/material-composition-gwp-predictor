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
import tkinter as tk
from PIL import ImageTk

from desktop_app.splash import _assets_dir, _build_frames

GITHUB_REPO = "eirasroger/material-composition-gwp-predictor"
_API_URL = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
_REQUEST_TIMEOUT = 8

# Inno Setup installs per-user under {localappdata}\GHGPredictor (see installer.iss).
_INSTALL_DIR = os.path.join(os.environ.get("LOCALAPPDATA", ""), "GHGPredictor")
_INSTALLED_EXE = os.path.join(_INSTALL_DIR, "GHGPredictor.exe")


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

    w, h = 420, 170
    px = parent.winfo_x() + (parent.winfo_width() - w) // 2
    py = parent.winfo_y() + (parent.winfo_height() - h) // 2
    dialog.geometry(f"{w}x{h}+{px}+{py}")

    ctk.CTkLabel(
        dialog,
        text=f"Version {version} is available.\nWould you like to update now?",
        font=ctk.CTkFont(size=14),
        justify="center",
    ).pack(pady=(28, 18))

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
        dialog.destroy()
        _show_update_splash(parent, url)

    update_btn.configure(command=_on_update)


def _show_update_splash(parent: ctk.CTk, url: str) -> None:
    """Frameless 'Updating...' splash while the installer downloads and runs silently."""
    splash = ctk.CTkToplevel(parent)
    splash.overrideredirect(True)
    splash.attributes("-topmost", True)

    w, h = 340, 420
    sw = splash.winfo_screenwidth()
    sh = splash.winfo_screenheight()
    splash.geometry(f"{w}x{h}+{(sw - w) // 2}+{(sh - h) // 2}")

    _theme_fg = ctk.ThemeManager.theme["CTk"]["fg_color"]
    if isinstance(_theme_fg, (list, tuple)):
        _canvas_bg = _theme_fg[1] if ctk.get_appearance_mode().lower() == "dark" else _theme_fg[0]
    else:
        _canvas_bg = _theme_fg

    icon_size = 180
    canvas = tk.Canvas(
        splash, width=icon_size, height=icon_size,
        bg=_canvas_bg, highlightthickness=0,
    )
    canvas.pack(pady=(60, 16))
    canvas_item = canvas.create_image(icon_size // 2, icon_size // 2, anchor="center")

    ctk.CTkLabel(
        splash, text="GHG Predictor",
        font=ctk.CTkFont(family="Segoe UI", size=22, weight="bold"),
        text_color="#e4f0ec",
    ).pack(pady=(0, 6))
    status_label = ctk.CTkLabel(
        splash, text="Downloading update...",
        font=ctk.CTkFont(family="Segoe UI", size=12),
        text_color="#7aada0",
    )
    status_label.pack(pady=(0, 24))

    progress = ctk.CTkProgressBar(
        splash, width=240, height=3, corner_radius=2,
        progress_color="#3ecf8e", fg_color="#2b3c37",
        mode="indeterminate",
    )
    progress.pack()
    progress.start()

    pil_frames: list = []
    tk_frames: list = []
    frames_ready = [False]
    frame_idx = [0]
    alive = [True]

    def _render() -> None:
        try:
            frames = _build_frames(str(_assets_dir() / "icon_vector.svg"), 48, icon_size)
            pil_frames.extend(frames)
        except Exception:
            pass
        frames_ready[0] = True

    def _animate() -> None:
        if frames_ready[0] and not tk_frames and pil_frames:
            tk_frames.extend([ImageTk.PhotoImage(img) for img in pil_frames])
        if tk_frames:
            canvas.itemconfig(canvas_item, image=tk_frames[frame_idx[0] % len(tk_frames)])
            frame_idx[0] += 1
        if alive[0]:
            splash.after(1000 // 30, _animate)

    threading.Thread(target=_render, daemon=True).start()
    _animate()

    def _download_worker() -> None:
        try:
            tmpdir = tempfile.mkdtemp()
            dest = os.path.join(tmpdir, "GHGPredictorSetup.exe")
            urllib.request.urlretrieve(url, dest)

            def _set_installing() -> None:
                try:
                    status_label.configure(text="Installing update...")
                except Exception:
                    pass

            parent.after(0, _set_installing)

            _spawn_installer_and_restart(dest)
            alive[0] = False
            parent.after(0, parent.destroy)
        except Exception as exc:
            alive[0] = False

            def _on_fail() -> None:
                splash.destroy()
                _show_error(parent, str(exc))

            parent.after(0, _on_fail)

    threading.Thread(target=_download_worker, daemon=True).start()


def _ps_quote(s: str) -> str:
    """Quote a string for inclusion inside a PowerShell single-quoted literal."""
    return "'" + s.replace("'", "''") + "'"


_UPDATE_PS1_TEMPLATE = r"""
$ErrorActionPreference = 'Continue'
$parentPid   = {pid}
$installer   = {installer}
$installedExe= {installed_exe}
$installDir  = {install_dir}
$logFile     = {log_file}
$installerLog= {installer_log}

# Ensure log dir exists.
$logDir = Split-Path $logFile -Parent
if (-not (Test-Path $logDir)) {{ New-Item -ItemType Directory -Path $logDir -Force | Out-Null }}

function Log($msg) {{
    "[{{0}}] {{1}}" -f (Get-Date -Format 'o'), $msg | Out-File -FilePath $logFile -Encoding utf8 -Append
}}

Log "updater starting (parent pid=$parentPid)"

# 1. Wait for the running app to exit so its files are no longer in use.
try {{
    $proc = Get-Process -Id $parentPid -ErrorAction SilentlyContinue
    if ($proc) {{
        Log "waiting for parent to exit (timeout 30s)"
        Wait-Process -Id $parentPid -Timeout 30 -ErrorAction SilentlyContinue
    }} else {{
        Log "parent already exited"
    }}
}} catch {{ Log "Wait-Process error: $_" }}

# 2. Force-kill any lingering GHGPredictor.exe instances. The PyInstaller bundle
#    holds DLLs in _internal\ that the silent installer cannot overwrite while
#    open; with /SUPPRESSMSGBOXES, the 'files in use' prompt defaults to Cancel
#    which silently aborts the install. Killing first prevents that.
try {{
    Get-Process -Name 'GHGPredictor' -ErrorAction SilentlyContinue |
        ForEach-Object {{ Log "killing leftover pid=$($_.Id)"; $_ | Stop-Process -Force -ErrorAction SilentlyContinue }}
}} catch {{ Log "Stop-Process error: $_" }}

# 3. Brief pause so Windows finishes releasing file handles.
Start-Sleep -Seconds 2

# 4. Run the installer fully silently. /TASKS=desktopicon force-checks the
#    desktop-shortcut task which is otherwise unchecked-by-default and would be
#    skipped by /VERYSILENT.
Log "launching installer: $installer"
try {{
    $p = Start-Process -FilePath $installer `
        -ArgumentList '/VERYSILENT','/SUPPRESSMSGBOXES','/NORESTART','/TASKS=desktopicon',"/LOG=$installerLog" `
        -Wait -PassThru
    Log "installer exit code: $($p.ExitCode)"
}} catch {{
    Log "installer launch failed: $_"
}}

# 5. Restart the freshly installed app.
if (Test-Path $installedExe) {{
    Log "starting $installedExe"
    try {{ Start-Process -FilePath $installedExe }} catch {{ Log "restart failed: $_" }}
}} else {{
    Log "installed exe not found at $installedExe"
}}

# 6. Self-delete this script.
try {{ Remove-Item -LiteralPath $PSCommandPath -Force -ErrorAction SilentlyContinue }} catch {{ }}
""".lstrip()


def _spawn_installer_and_restart(installer_path: str) -> None:
    """
    Launch a detached helper that:
      1. Waits for the current process (this app) to exit so files are unlocked.
      2. Force-kills any lingering GHGPredictor.exe before the installer runs so
         /SUPPRESSMSGBOXES never hits a 'files in use' prompt (which defaults to
         Cancel and silently aborts the install).
      3. Runs the Inno Setup installer silently with /TASKS=desktopicon so the
         desktop shortcut is created without user input.
      4. Launches the freshly installed exe.

    Writes the helper to a temp .ps1 file and runs it as a fully detached
    process so it survives our shutdown. Logs to %LOCALAPPDATA%\\GHGPredictor\\
    update.log to make future failures debuggable.
    """
    my_pid = os.getpid()
    log_file = os.path.join(_INSTALL_DIR, "update.log")
    installer_log = os.path.join(_INSTALL_DIR, "installer.log")

    script = _UPDATE_PS1_TEMPLATE.format(
        pid=my_pid,
        installer=_ps_quote(installer_path),
        installed_exe=_ps_quote(_INSTALLED_EXE),
        install_dir=_ps_quote(_INSTALL_DIR),
        log_file=_ps_quote(log_file),
        installer_log=_ps_quote(installer_log),
    )

    script_dir = tempfile.mkdtemp(prefix="ghg_update_")
    script_path = os.path.join(script_dir, "update.ps1")
    with open(script_path, "w", encoding="utf-8") as fh:
        fh.write(script)

    DETACHED_PROCESS = 0x00000008
    CREATE_NEW_PROCESS_GROUP = 0x00000200
    CREATE_NO_WINDOW = 0x08000000

    subprocess.Popen(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy", "Bypass",
            "-WindowStyle", "Hidden",
            "-File", script_path,
        ],
        creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW,
        close_fds=True,
    )


def _show_error(parent: ctk.CTk, message: str) -> None:
    dialog = ctk.CTkToplevel(parent)
    dialog.title("Update Failed")
    dialog.resizable(False, False)
    dialog.transient(parent)
    dialog.grab_set()

    w, h = 360, 150
    px = parent.winfo_x() + (parent.winfo_width() - w) // 2
    py = parent.winfo_y() + (parent.winfo_height() - h) // 2
    dialog.geometry(f"{w}x{h}+{px}+{py}")

    ctk.CTkLabel(
        dialog,
        text=f"Download failed:\n{message}",
        font=ctk.CTkFont(size=13),
        justify="center",
        wraplength=320,
    ).pack(pady=(24, 16))

    ctk.CTkButton(dialog, text="Close", width=100, command=dialog.destroy).pack()
