"""
Local scenario library — persistent storage for the *inputs* of a comparison.

A scenario is exactly what the user typed: the shared category, and per product
the name, category override, materials, end-of-life shares and circular origin.
Nothing derived is persisted. Predicting 25 targets for 4 products costs ~21 ms
(LEARNINGS 2026-07-25c), so there is no case for caching results — and a cached
number would silently outlive the model that produced it, which is worse than
useless in a tool whose models get retrained and shipped by auto-update.

Layout — per-user, Roaming, so it survives reinstall and follows the user:

    %APPDATA%/GHGPredictor/
        scenarios/{uuid}.json      one file per scenario
        last_session.json          autosaved working state, restored on launch

Deliberately NOT under %LOCALAPPDATA%/GHGPredictor: that is the install
directory (installer.iss, updater.py:30), which the silent auto-updater
overwrites and the uninstaller deletes.

One file per scenario rather than one library file: a failed write can only
damage a single scenario, and each file already has the shape of a record in a
future shared/team store (stable uuid + UTC timestamps + an author slot). Those
three fields are the only part of this format that cannot be retrofitted later,
which is why they exist now.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

SCHEMA_VERSION = 1

_APP_DIR_NAME = "GHGPredictor"
_SCENARIO_DIR = "scenarios"
_SESSION_FILE = "last_session.json"


# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
def root_dir() -> Path:
    """`%APPDATA%/GHGPredictor`, or `~/.ghgpredictor` where APPDATA is unset."""
    appdata = os.environ.get("APPDATA")
    base = Path(appdata) / _APP_DIR_NAME if appdata else Path.home() / ".ghgpredictor"
    return base


def scenarios_dir() -> Path:
    return root_dir() / _SCENARIO_DIR


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Atomic IO
# ──────────────────────────────────────────────────────────────────────────────
def _write_json(path: Path, payload: Dict) -> None:
    """
    Write via a temp file + os.replace so an interrupted write cannot leave a
    half-written scenario behind. os.replace is atomic on Windows for same-volume
    moves, which is why the temp file sits next to the target.
    """
    _ensure_dir(path.parent)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _read_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


# ──────────────────────────────────────────────────────────────────────────────
# Provenance (informational only — nothing branches on it)
# ──────────────────────────────────────────────────────────────────────────────
_fingerprint_cache: Optional[str] = None


def assets_fingerprint(assets_dir: Optional[Path] = None) -> Optional[str]:
    """
    Short hash of targets_manifest.json — i.e. of *which* targets the app was
    shipping when the scenario was saved. Recorded for debugging user reports;
    no code path reads it back.
    """
    global _fingerprint_cache
    if _fingerprint_cache is not None:
        return _fingerprint_cache
    if assets_dir is None:
        assets_dir = Path(__file__).resolve().parent / "assets"
    manifest = Path(assets_dir) / "targets_manifest.json"
    try:
        digest = hashlib.sha256(manifest.read_bytes()).hexdigest()[:12]
    except Exception:
        return None
    _fingerprint_cache = digest
    return digest


def _provenance() -> Dict:
    try:
        from desktop_app._version import __version__
    except Exception:
        __version__ = None
    return {"app_version": __version__, "assets_fingerprint": assets_fingerprint()}


# ──────────────────────────────────────────────────────────────────────────────
# Schema
# ──────────────────────────────────────────────────────────────────────────────
def migrate(doc: Dict) -> Dict:
    """
    Bring a stored document up to SCHEMA_VERSION.

    Only ever add keys here; renaming one breaks every file already on disk.
    An unknown *newer* version is returned untouched — a scenario written by a
    later build should degrade to "some fields ignored", not fail to open.
    """
    version = int(doc.get("schema_version", 0) or 0)

    if version < 1:
        doc.setdefault("id", str(uuid.uuid4()))
        doc.setdefault("name", "Untitled scenario")
        doc.setdefault("notes", "")
        doc.setdefault("created_at", _utc_now())
        doc.setdefault("updated_at", doc["created_at"])
        doc.setdefault("updated_by", None)
        doc.setdefault("shared_category", None)
        doc.setdefault("products", [])
        doc["schema_version"] = 1

    return doc


def _normalise_product(entry: Dict) -> Dict:
    """Fill in a product's fields so a hand-edited or older file still loads."""
    eol = entry.get("eol") or {}
    return {
        "name": str(entry.get("name") or "Product"),
        "category_override_enabled": bool(entry.get("category_override_enabled", False)),
        "category_override": entry.get("category_override"),
        "materials": [
            {
                "name": str(m.get("name", "")),
                "percentage": float(m.get("percentage", 0.0) or 0.0),
            }
            for m in (entry.get("materials") or [])
            if isinstance(m, dict)
        ],
        "eol": {
            key: float(eol.get(key, 0.0) or 0.0)
            for key in ("recycling", "hazardous", "inert", "incineration")
        },
        "origin_pct": float(entry.get("origin_pct", 0.0) or 0.0),
    }


def state_from_doc(doc: Dict) -> Dict:
    """The window-facing state dict: `{shared_category, products}`."""
    return {
        "shared_category": doc.get("shared_category"),
        "products": [_normalise_product(p) for p in (doc.get("products") or [])],
    }


# ──────────────────────────────────────────────────────────────────────────────
# Validation against the currently shipped assets
# ──────────────────────────────────────────────────────────────────────────────
def validate_state(
    state: Dict,
    categories: Iterable[str],
    materials: Iterable[str],
) -> List[str]:
    """
    Human-readable warnings for anything in ``state`` the current build no
    longer recognises.

    Categories are the real risk: the coalescing work renamed them once already
    (LEARNINGS 2026-07-25a) and distributions.json keys must match
    ``InferenceAdapter.categories()`` exactly or the reference lookup silently
    misses (2026-07-25c). A stale name must therefore surface as a message, never
    as a card that quietly looks like the user forgot to pick one.

    Messages name the product, because a scenario can be partly valid — shared
    category fine, one override gone.
    """
    known_categories = set(categories)
    known_materials = set(materials)
    warnings: List[str] = []

    shared = state.get("shared_category")
    if shared and shared not in known_categories:
        warnings.append(f'Category "{shared}" is no longer available.')

    for entry in state.get("products") or []:
        label = entry.get("name") or "Product"

        if entry.get("category_override_enabled"):
            override = entry.get("category_override")
            if override and override not in known_categories:
                warnings.append(f'{label}: category "{override}" is no longer available.')
            elif not override:
                warnings.append(f"{label}: has its own category set, but none was chosen.")

        unknown = [
            m.get("name", "")
            for m in entry.get("materials") or []
            if m.get("name") and m.get("name") not in known_materials
        ]
        for name in unknown:
            warnings.append(f'{label}: material "{name}" is no longer available.')

    return warnings


# ──────────────────────────────────────────────────────────────────────────────
# Scenario CRUD
# ──────────────────────────────────────────────────────────────────────────────
def _path_for(scenario_id: str) -> Path:
    return scenarios_dir() / f"{scenario_id}.json"


def save_scenario(
    name: str,
    state: Dict,
    scenario_id: Optional[str] = None,
    notes: str = "",
) -> Dict:
    """
    Write ``state`` under ``name``.

    ``scenario_id`` given → overwrite that scenario in place, keeping its id and
    created_at (this is "Save"). Omitted → mint a new id ("Save as").
    Returns the stored document.
    """
    now = _utc_now()
    created_at = now
    existing_notes = notes

    if scenario_id:
        try:
            previous = migrate(_read_json(_path_for(scenario_id)))
            created_at = previous.get("created_at", now)
            if not notes:
                existing_notes = previous.get("notes", "")
        except Exception:
            pass       # id points at nothing readable — treat as a fresh save
    else:
        scenario_id = str(uuid.uuid4())

    doc = {
        "schema_version": SCHEMA_VERSION,
        "id": scenario_id,
        "name": name.strip() or "Untitled scenario",
        "notes": existing_notes,
        "created_at": created_at,
        "updated_at": now,
        "updated_by": None,          # reserved for the team-sharing build
        "shared_category": state.get("shared_category"),
        "products": [_normalise_product(p) for p in (state.get("products") or [])],
        "provenance": _provenance(),
    }
    _write_json(_path_for(scenario_id), doc)
    return doc


def load_scenario(scenario_id: str) -> Dict:
    return migrate(_read_json(_path_for(scenario_id)))


def list_scenarios() -> List[Dict]:
    """
    Summaries for the library dialog, newest first.

    The directory is the source of truth — there is no index file to fall out of
    sync with it. Unreadable files are skipped rather than raised: one corrupt
    scenario must not make the library unopenable.
    """
    out: List[Dict] = []
    directory = scenarios_dir()
    if not directory.exists():
        return out

    for path in directory.glob("*.json"):
        try:
            doc = migrate(_read_json(path))
        except Exception:
            continue
        out.append({
            "id":           doc.get("id") or path.stem,
            "name":         doc.get("name") or path.stem,
            "notes":        doc.get("notes") or "",
            "updated_at":   doc.get("updated_at") or "",
            "created_at":   doc.get("created_at") or "",
            "product_count": len(doc.get("products") or []),
        })

    out.sort(key=lambda r: r["updated_at"], reverse=True)
    return out


def rename_scenario(scenario_id: str, name: str) -> Dict:
    doc = load_scenario(scenario_id)
    doc["name"] = name.strip() or doc.get("name") or "Untitled scenario"
    doc["updated_at"] = _utc_now()
    _write_json(_path_for(scenario_id), doc)
    return doc


def duplicate_scenario(scenario_id: str, name: Optional[str] = None) -> Dict:
    doc = load_scenario(scenario_id)
    return save_scenario(
        name=name or f"{doc.get('name', 'Untitled scenario')} (copy)",
        state=state_from_doc(doc),
        notes=doc.get("notes", ""),
    )


def delete_scenario(scenario_id: str) -> None:
    try:
        _path_for(scenario_id).unlink()
    except FileNotFoundError:
        pass


# ──────────────────────────────────────────────────────────────────────────────
# Last-session autosave
# ──────────────────────────────────────────────────────────────────────────────
def session_path() -> Path:
    return root_dir() / _SESSION_FILE


def save_session(
    state: Dict,
    scenario_id: Optional[str],
    scenario_name: Optional[str],
) -> None:
    """
    Autosave the working state. Separate from the library on purpose: this is a
    crash/restart net, not something the user named and chose to keep.
    """
    _write_json(session_path(), {
        "schema_version": SCHEMA_VERSION,
        "saved_at":       _utc_now(),
        "scenario_id":    scenario_id,
        "scenario_name":  scenario_name,
        "state":          {
            "shared_category": state.get("shared_category"),
            "products": [_normalise_product(p) for p in (state.get("products") or [])],
        },
    })


def load_session() -> Optional[Tuple[Dict, Optional[str], Optional[str]]]:
    """
    `(state, scenario_id, scenario_name)`, or None if there is nothing worth
    restoring.

    "Nothing worth restoring" is the same test the window uses for its discard
    prompt: no category and no materials anywhere. A blank card is what a fresh
    launch already shows, so restoring one would just be work with no effect —
    and if the user cleared the screen before quitting, a fresh start is what
    they asked for.
    """
    try:
        doc = _read_json(session_path())
    except Exception:
        return None
    state = doc.get("state") or {}
    products = state.get("products") or []
    if not products:
        return None
    if not state.get("shared_category") and not any(p.get("materials") for p in products):
        return None
    return (
        {
            "shared_category": state.get("shared_category"),
            "products": [_normalise_product(p) for p in state["products"]],
        },
        doc.get("scenario_id"),
        doc.get("scenario_name"),
    )


def clear_session() -> None:
    try:
        session_path().unlink()
    except FileNotFoundError:
        pass
