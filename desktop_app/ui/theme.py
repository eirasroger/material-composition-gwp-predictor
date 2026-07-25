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

# Lifecycle-stage display order (EN 15804-ish) and one fixed colour per stage,
# reused everywhere a stage breakdown is drawn so a stage always reads the
# same regardless of indicator or view. Unknown stage_keys fall back to
# STAGE_FALLBACK_COLOR and sort after every known stage.
STAGE_ORDER = ["total", "a1a3", "a4", "a5", "c1", "c2", "c3", "c4", "d"]
STAGE_COLORS = {
    "total": ACCENT,
    "a1a3":  "#3ecf8e",  # production — primary driver, same family as ACCENT
    "a4":    "#5fb0c9",
    "a5":    "#5fb0c9",
    "c1":    "#e0a83e",
    "c2":    "#e0a83e",
    "c3":    "#e0a83e",
    "c4":    "#e07b39",
    "d":     "#7b8fe0",  # avoided-burden credit — visually distinct, usually negative
}
STAGE_FALLBACK_COLOR = TEXT_SEC


def stage_sort_key(stage_key: str) -> tuple:
    try:
        return (0, STAGE_ORDER.index(stage_key))
    except ValueError:
        return (1, stage_key)


def stage_color(stage_key: str) -> str:
    return STAGE_COLORS.get(stage_key, STAGE_FALLBACK_COLOR)

_FAMILY = "Segoe UI"


def font(size: int = 13, weight: str = "normal") -> ctk.CTkFont:
    return ctk.CTkFont(family=_FAMILY, size=size, weight=weight)
