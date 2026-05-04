from __future__ import annotations

import math
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

import customtkinter as ctk
import tkinter as tk
from PIL import Image, ImageTk


def _assets_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", "")) / "assets"
    return Path(__file__).resolve().parent / "assets"


def _make_frame_svg(svg_content: str, t: float) -> str:
    CX, CY = 338.5, 334.0
    PHASES = [0.0, 0.08, 0.18]
    ARC_INDICES = [2, 3, 4]
    parts = re.split(r'(<path\s+d="[^"]+"\s+fill="[^"]+"\s*/>)', svg_content)
    path_count = 0
    result = []
    for part in parts:
        if not part.startswith('<path'):
            result.append(part)
            continue
        if path_count in ARC_INDICES:
            idx = ARC_INDICES.index(path_count)
            angle = (t + PHASES[idx]) * 2 * math.pi
            sx = math.cos(angle)
            sx = max(sx, 1e-6) if sx >= 0 else min(sx, -1e-6)
            transform = f'translate({CX},{CY}) scale({sx:.6f},1) translate({-CX},{-CY})'
            result.append(f'<g transform="{transform}">{part}</g>')
        else:
            result.append(part)
        path_count += 1
    return ''.join(result)


def _svg_to_pil(svg_str: str, size: int) -> Image.Image:
    import fitz
    doc = fitz.open(stream=svg_str.encode("utf-8"), filetype="svg")
    page = doc[0]
    scale = size / max(page.rect.width, page.rect.height)
    mat = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=mat, alpha=True)
    img = Image.frombytes("RGBA", (pix.width, pix.height), pix.samples)
    doc.close()
    return img


def _build_frames(svg_path: str, n_frames: int, size: int) -> list:
    with open(svg_path, "r", encoding="utf-8") as f:
        svg_content = f.read()
    return [_svg_to_pil(_make_frame_svg(svg_content, i / n_frames), size) for i in range(n_frames)]


class SplashScreen:
    def __init__(
        self,
        app_name: str = "GHG Predictor",
        loading_func: Optional[Callable] = None,
        n_frames: int = 48,
        icon_size: int = 180,
        fps: int = 30,
        min_display_s: float = 2.0,
    ) -> None:
        self.svg_path = str(_assets_dir() / "icon_vector.svg")
        self.app_name = app_name
        self.loading_func = loading_func
        self.n_frames = n_frames
        self.icon_size = icon_size
        self.fps = fps
        self.min_display_s = min_display_s

        self._frame_idx = 0
        self._pil_frames: list = []
        self._tk_frames: list = []
        self._frames_ready = False
        self._loading_done = False
        self._min_time_reached = False
        self._result: Any = None
        self._error: Optional[Exception] = None

        self.root = ctk.CTk()
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)

        w, h = 340, 420
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        self.root.geometry(f"{w}x{h}+{(sw - w) // 2}+{(sh - h) // 2}")

        # Match canvas bg to the CTk window background so there's no seam
        _theme_fg = ctk.ThemeManager.theme["CTk"]["fg_color"]
        if isinstance(_theme_fg, (list, tuple)):
            _canvas_bg = _theme_fg[1] if ctk.get_appearance_mode().lower() == "dark" else _theme_fg[0]
        else:
            _canvas_bg = _theme_fg

        self.canvas = tk.Canvas(
            self.root, width=icon_size, height=icon_size,
            bg=_canvas_bg, highlightthickness=0,
        )
        self.canvas.pack(pady=(60, 16))
        self._canvas_item = self.canvas.create_image(
            icon_size // 2, icon_size // 2, anchor="center",
        )

        ctk.CTkLabel(
            self.root, text=app_name,
            font=ctk.CTkFont(family="Segoe UI", size=22, weight="bold"),
            text_color="#e4f0ec",
        ).pack(pady=(0, 6))
        ctk.CTkLabel(
            self.root, text="Loading model...",
            font=ctk.CTkFont(family="Segoe UI", size=12),
            text_color="#7aada0",
        ).pack(pady=(0, 24))

        self.progress = ctk.CTkProgressBar(
            self.root, width=240, height=3, corner_radius=2,
            progress_color="#3ecf8e", fg_color="#2b3c37",
            mode="indeterminate",
        )
        self.progress.pack()
        self.progress.start()

    def _animate(self) -> None:
        if self._frames_ready and not self._tk_frames and self._pil_frames:
            self._tk_frames = [ImageTk.PhotoImage(img) for img in self._pil_frames]
        if self._tk_frames:
            self.canvas.itemconfig(
                self._canvas_item,
                image=self._tk_frames[self._frame_idx % len(self._tk_frames)],
            )
            self._frame_idx += 1
        if self._loading_done and self._min_time_reached and self._frames_ready:
            self.progress.stop()
            self.root.destroy()
            return
        self.root.after(1000 // self.fps, self._animate)

    def run(self) -> Any:
        def _render() -> None:
            try:
                self._pil_frames = _build_frames(self.svg_path, self.n_frames, self.icon_size)
            except Exception:
                pass  # no animation frames; progress bar still shows
            self._frames_ready = True

        def _load() -> None:
            try:
                if self.loading_func:
                    self._result = self.loading_func()
            except Exception as exc:
                self._error = exc
            finally:
                self._loading_done = True

        def _timer() -> None:
            time.sleep(self.min_display_s)
            self._min_time_reached = True

        threading.Thread(target=_render, daemon=True).start()
        threading.Thread(target=_load, daemon=True).start()
        threading.Thread(target=_timer, daemon=True).start()
        self._animate()
        self.root.mainloop()

        if self._error is not None:
            raise self._error
        return self._result


if __name__ == "__main__":
    from pathlib import Path as _Path
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme(
        str(_Path(__file__).resolve().parent / "assets" / "theme_dark.json")
    )

    def _mock_load():
        time.sleep(3.0)
        return True

    result = SplashScreen(loading_func=_mock_load).run()
    print(f"Splash done, result={result}")
