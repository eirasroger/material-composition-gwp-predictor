"""Central design tokens — imported by all UI modules."""
from __future__ import annotations

import customtkinter as ctk

BG         = "#161c1a"
SURFACE    = "#1e2724"
SURFACE_HI = "#253029"
ACCENT     = "#3ecf8e"
ACCENT_DIM = "#2a9d6a"
BORDER     = "#2b3c37"
TEXT_PRI   = "#e4f0ec"
TEXT_SEC   = "#7aada0"
TEXT_DIM   = "#456058"

MAX_PRODUCTS = 4
PRODUCT_COLORS = [
    "#3ecf8e",  # 0 — green (matches ACCENT)
    "#e07b39",  # 1 — amber-orange
    "#7b8fe0",  # 2 — soft indigo
    "#d96b8e",  # 3 — rose
]

SUM_GREEN  = (ACCENT, ACCENT)
SUM_AMBER  = ("#d49b3a", "#d49b3a")
SUM_RED    = ("#d44a4a", "#d44a4a")

_FAMILY = "Segoe UI"


def font(size: int = 13, weight: str = "normal") -> ctk.CTkFont:
    return ctk.CTkFont(family=_FAMILY, size=size, weight=weight)
