"""Central design tokens — imported by all UI modules."""
from __future__ import annotations

import math

import customtkinter as ctk


def fmt_value(v: float) -> str:
    """
    Format an impact value for display, across the six orders of magnitude the
    indicators span (adpf ~10 MJ/kg down to ep ~1e-6 kg PO4-eq/kg).

    Fixed decimals are useless here ("0.000" for every eutrophication value) and
    %g flips to exponent below 1e-4, which reads badly as a headline number
    ("9.954e-04 m3/kg"). So: plain decimals at 4 significant digits while the
    magnitude is human-readable, exponent only past that.
    """
    if v == 0:
        return "0"
    mag = math.floor(math.log10(abs(v)))
    if -7 < mag < 5:
        decimals = max(0, 3 - mag)
        text = f"{v:,.{decimals}f}"
        return text.rstrip("0").rstrip(".") if "." in text else text
    return f"{v:.3e}"

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

# Indicator display order — fixed, because the summary radar's shape is only
# comparable between products if the axes never move. Short labels keep the
# pentagon legible; the manifest's display_name is too long for an axis.
INDICATOR_ORDER = ["ghg", "fw", "ep", "ap", "adpf"]
INDICATOR_LABELS = {
    "ghg":  "GHG",
    "fw":   "Water",
    "ep":   "Eutroph.",
    "ap":   "Acidif.",
    "adpf": "Fossil",
}


def indicator_sort_key(indicator_key: str) -> tuple:
    try:
        return (0, INDICATOR_ORDER.index(indicator_key))
    except ValueError:
        return (1, indicator_key)


def indicator_label(indicator_key: str) -> str:
    return INDICATOR_LABELS.get(indicator_key, indicator_key.upper())


# Short axis labels for lifecycle stages. Derived from stage_key, not by
# stripping a prefix off display_name -- "GHG A1-A3" strips fine but
# "Water Depletion A1-A3" and "Eutrophication C3" do not.
STAGE_LABELS = {
    "total": "Total", "a1a3": "A1-A3", "a4": "A4", "a5": "A5",
    "c1": "C1", "c2": "C2", "c3": "C3", "c4": "C4", "d": "D",
}


def stage_label(stage_key: str) -> str:
    return STAGE_LABELS.get(stage_key, stage_key.upper())


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
