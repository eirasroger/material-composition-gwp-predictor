"""
Round-trip tests for the scenario library.

Two halves:

* Pure storage — save/load/rename/duplicate/delete and validation. No Tk needed.
* Widget round-trip — real ProductCards through a real MainWindow, because the
  bugs worth catching here live in the widgets, not in the JSON: the category
  override is two independent fields whose combinations are easy to collapse by
  accident, and a load must not leave a card predicting against the wrong
  category.

The Tk half skips itself where no display is available.

    python -m desktop_app.tests.test_scenario_roundtrip
"""

from __future__ import annotations

import os
import sys
import tempfile
import traceback
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from desktop_app import library     # noqa: E402


_failures: list[str] = []
_skipped: list[str] = []


def check(condition: bool, label: str) -> None:
    if condition:
        print(f"  ok    {label}")
    else:
        print(f"  FAIL  {label}")
        _failures.append(label)


def section(title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────
CATEGORIES = ["Concrete", "Cement", "Insulation", "Other: Flooring"]
MATERIALS = ["cement", "sand", "water", "steel rebar", "glass wool"]


def sample_state() -> dict:
    """One card per override combination — the states that are easy to collapse."""
    return {
        "shared_category": "Concrete",
        "products": [
            {   # override off — follows the shared category
                "name": "Baseline mix",
                "category_override_enabled": False,
                "category_override": None,
                "materials": [
                    {"name": "cement", "percentage": 12.5},
                    {"name": "sand", "percentage": 62.5},
                    {"name": "water", "percentage": 25.0},
                ],
                "eol": {"recycling": 60.0, "hazardous": 0.0,
                        "inert": 20.0, "incineration": 20.0},
                "origin_pct": 30.0,
            },
            {   # override on, category chosen
                "name": "Insulated variant",
                "category_override_enabled": True,
                "category_override": "Insulation",
                "materials": [{"name": "glass wool", "percentage": 100.0}],
                "eol": {"recycling": 0.0, "hazardous": 0.0,
                        "inert": 100.0, "incineration": 0.0},
                "origin_pct": 0.0,
            },
            {   # override on, nothing chosen yet — must NOT reload as "off"
                "name": "Undecided",
                "category_override_enabled": True,
                "category_override": None,
                "materials": [{"name": "steel rebar", "percentage": 100.0}],
                "eol": {"recycling": 90.0, "hazardous": 10.0,
                        "inert": 0.0, "incineration": 0.0},
                "origin_pct": 55.0,
            },
        ],
    }


# ──────────────────────────────────────────────────────────────────────────────
# Storage layer
# ──────────────────────────────────────────────────────────────────────────────
def test_storage() -> None:
    section("Storage — save / load / list / rename / duplicate / delete")

    state = sample_state()
    doc = library.save_scenario("Facade options", state)
    check(doc["schema_version"] == library.SCHEMA_VERSION, "schema_version stamped")
    check(bool(doc["id"]), "uuid assigned")
    check(doc["created_at"] == doc["updated_at"], "created_at == updated_at on first save")
    check(doc["updated_by"] is None, "updated_by slot present and empty")

    reloaded = library.state_from_doc(library.load_scenario(doc["id"]))
    check(reloaded == state, "state survives a save/load round trip byte-for-byte")

    products = reloaded["products"]
    check(
        products[0]["category_override_enabled"] is False
        and products[0]["category_override"] is None,
        "override off round-trips",
    )
    check(
        products[1]["category_override_enabled"] is True
        and products[1]["category_override"] == "Insulation",
        "override on with a category round-trips",
    )
    check(
        products[2]["category_override_enabled"] is True
        and products[2]["category_override"] is None,
        "override on with NO category stays distinct from override off",
    )

    # Save in place keeps identity; save-as mints a new one.
    same = library.save_scenario("Facade options", state, scenario_id=doc["id"])
    check(same["id"] == doc["id"], "Save keeps the same id")
    check(same["created_at"] == doc["created_at"], "Save preserves created_at")

    other = library.save_scenario("Facade options v2", state)
    check(other["id"] != doc["id"], "Save as mints a new id")

    rows = library.list_scenarios()
    check(len(rows) == 2, "both scenarios listed")
    check(rows[0]["product_count"] == 3, "product count reported")

    library.rename_scenario(doc["id"], "Renamed")
    check(library.load_scenario(doc["id"])["name"] == "Renamed", "rename persists")

    copy = library.duplicate_scenario(doc["id"])
    check(copy["id"] != doc["id"], "duplicate gets a new id")
    check(copy["name"] == "Renamed (copy)", "duplicate names itself")
    check(
        library.state_from_doc(copy) == state,
        "duplicate carries the same inputs",
    )

    library.delete_scenario(copy["id"])
    check(
        copy["id"] not in {r["id"] for r in library.list_scenarios()},
        "delete removes it from the listing",
    )

    # A corrupt file must not take the library down with it.
    bad = library.scenarios_dir() / "not-json.json"
    bad.write_text("{ this is not json", encoding="utf-8")
    check(len(library.list_scenarios()) == 2, "corrupt file skipped, rest still listed")
    bad.unlink()

    # No derived data on disk.
    raw = library.load_scenario(doc["id"])
    check(
        not any(k in raw for k in ("predictions", "all_preds", "results", "view")),
        "nothing derived is persisted",
    )


def test_migration() -> None:
    section("Storage — migration of an unversioned document")

    legacy = {"products": [], "shared_category": "Cement"}
    migrated = library.migrate(dict(legacy))
    check(migrated["schema_version"] == 1, "unversioned doc migrates to v1")
    check(bool(migrated["id"]), "migration backfills an id")
    check(migrated["shared_category"] == "Cement", "existing fields untouched")

    future = {"schema_version": 99, "id": "x", "name": "n", "products": []}
    check(
        library.migrate(dict(future))["schema_version"] == 99,
        "a newer document is passed through, not rejected",
    )


def test_validation() -> None:
    section("Storage — validation against the shipped assets")

    # "Override on, nothing picked" is itself a reportable state — that card
    # cannot be predicted — so the clean fixture is the first two products only.
    clean = sample_state()
    clean["products"] = clean["products"][:2]
    check(
        library.validate_state(clean, CATEGORIES, MATERIALS) == [],
        "a fully valid scenario produces no warnings",
    )

    state = sample_state()

    stale = sample_state()
    stale["shared_category"] = "Ceramics (removed)"
    stale["products"][1]["category_override"] = "Gone category"
    stale["products"][0]["materials"][0]["name"] = "unobtainium"
    warnings = library.validate_state(stale, CATEGORIES, MATERIALS)

    check(
        any("Ceramics (removed)" in w for w in warnings),
        "a dropped shared category is reported",
    )
    check(
        any("Insulated variant" in w and "Gone category" in w for w in warnings),
        "a dropped override category names the product it belongs to",
    )
    check(
        any("unobtainium" in w and "Baseline mix" in w for w in warnings),
        "an unknown material names the product it belongs to",
    )
    check(
        any("Undecided" in w for w in warnings),
        "override-on-with-nothing-chosen is reported, not silently ignored",
    )


def test_session() -> None:
    section("Storage — last-session autosave")

    state = sample_state()
    library.save_session(state, "sid-1", "Facade options")
    restored = library.load_session()
    check(restored is not None, "session reloads")
    if restored is not None:
        got_state, sid, name = restored
        check(got_state == state, "session state round-trips")
        check(sid == "sid-1" and name == "Facade options", "session keeps scenario identity")

    library.save_session({"shared_category": None, "products": []}, None, None)
    check(library.load_session() is None, "an empty session is not restored")

    # What a fresh launch autosaves: one blank card, nothing typed. Restoring it
    # would recreate exactly what startup already shows.
    library.save_session(
        {"shared_category": None, "products": [{"name": "Product 1", "materials": []}]},
        None, None,
    )
    check(library.load_session() is None, "a blank first-run session is not restored")

    # A category alone is worth keeping even with no materials yet.
    library.save_session(
        {"shared_category": "Concrete", "products": [{"name": "Product 1", "materials": []}]},
        None, None,
    )
    check(library.load_session() is not None, "a chosen category alone is restorable")

    library.clear_session()
    check(library.load_session() is None, "cleared session stays gone")


# ──────────────────────────────────────────────────────────────────────────────
# Widget round-trip
# ──────────────────────────────────────────────────────────────────────────────
def test_widgets() -> None:
    section("Widgets — MainWindow scenario round-trip")

    try:
        import customtkinter as ctk
        from desktop_app.inference_adapter import InferenceAdapter
        from desktop_app.ui.main_window import MainWindow
    except Exception as exc:
        print(f"  skip  imports unavailable ({exc})")
        _skipped.append("widget round-trip")
        return

    try:
        adapter = InferenceAdapter()
    except Exception as exc:
        print(f"  skip  adapter unavailable — run bake_assets.py first ({exc})")
        _skipped.append("widget round-trip")
        return

    try:
        ctk.set_appearance_mode("dark")
        theme = Path(__file__).resolve().parents[1] / "assets" / "theme_dark.json"
        if theme.exists():
            ctk.set_default_color_theme(str(theme))
        window = MainWindow(adapter)
        window.withdraw()
    except Exception as exc:
        print(f"  skip  no display ({exc})")
        _skipped.append("widget round-trip")
        return

    try:
        categories = adapter.categories
        materials = adapter.materials
        state = {
            "shared_category": categories[0],
            "products": [
                {
                    "name": "Shared-category product",
                    "category_override_enabled": False,
                    "category_override": None,
                    "materials": [{"name": materials[0], "percentage": 70.0},
                                  {"name": materials[1], "percentage": 30.0}],
                    "eol": {"recycling": 50.0, "hazardous": 0.0,
                            "inert": 30.0, "incineration": 20.0},
                    "origin_pct": 25.0,
                },
                {
                    "name": "Own-category product",
                    "category_override_enabled": True,
                    "category_override": categories[1],
                    "materials": [{"name": materials[2], "percentage": 100.0}],
                    "eol": {"recycling": 100.0, "hazardous": 0.0,
                            "inert": 0.0, "incineration": 0.0},
                    "origin_pct": 0.0,
                },
                {
                    "name": "Override, nothing picked",
                    "category_override_enabled": True,
                    "category_override": None,
                    "materials": [{"name": materials[3], "percentage": 100.0}],
                    "eol": {"recycling": 0.0, "hazardous": 0.0,
                            "inert": 100.0, "incineration": 0.0},
                    "origin_pct": 10.0,
                },
            ],
        }

        # Go through the real open path — that is what sets scenario identity
        # and the clean baseline, not apply_scenario_state on its own.
        doc = library.save_scenario("Widget round trip", state)
        window._load_scenario(doc["id"])
        window.update()

        check(len(window._cards) == 3, "card count matches the scenario")
        check(window._scenario_id == doc["id"], "opened scenario id is adopted")
        check(window._scenario_name == "Widget round trip", "scenario name is adopted")

        got = window.scenario_state()
        check(got == state, "widget state round-trips exactly")

        cards = window._cards
        check(
            cards[0].has_category_override() is False,
            "card 1 override switch is off",
        )
        check(
            cards[1].has_category_override() is True
            and cards[1].local_category() == categories[1],
            "card 2 keeps its own category",
        )
        check(
            cards[2].has_category_override() is True
            and cards[2].local_category() is None,
            "card 3 stays 'override on, nothing picked'",
        )
        check(
            cards[1]._local_category_panel.winfo_ismapped()
            or cards[1]._local_category_panel.winfo_manager() != "",
            "an overriding card actually shows its category panel",
        )
        check(
            cards[0]._local_category_panel.winfo_manager() == "",
            "a non-overriding card hides its category panel",
        )

        # Effective category, i.e. what actually reached the model.
        check(
            window._effective_category(cards[0]) == categories[0],
            "card 1 predicts against the shared category",
        )
        check(
            window._effective_category(cards[1]) == categories[1],
            "card 2 predicts against its own category",
        )
        check(
            window._effective_category(cards[2]) is None,
            "card 3 has no effective category and is not predicted",
        )

        # Predictions were produced without any further user interaction.
        snap0 = window._predictions.get(id(cards[0]))
        check(snap0 is not None and bool(snap0["all_preds"]),
              "loading a scenario predicts immediately")
        check(
            window._predictions.get(id(cards[2])) is None,
            "the card with no category is left unpredicted, with a status",
        )

        # Dirty tracking.
        check(not window._is_dirty(), "state is clean right after a load")
        cards[0]._origin_panel.set_value(80.0)
        check(window._is_dirty(), "an edit marks the scenario dirty")

        # Shrinking the card count on a subsequent load.
        window.apply_scenario_state(
            {"shared_category": categories[0], "products": [state["products"][0]]}
        )
        window.update()
        check(len(window._cards) == 1, "loading a smaller scenario removes cards")
        check(
            len(window._used_color_indices) == 1,
            "colour slots are reissued, not leaked",
        )
    finally:
        try:
            window.destroy()
        except Exception:
            pass


# ──────────────────────────────────────────────────────────────────────────────
def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        # Redirect the library at a throwaway APPDATA so the real one is untouched.
        os.environ["APPDATA"] = tmp
        print(f"library root: {library.root_dir()}")

        for test in (test_storage, test_migration, test_validation,
                     test_session, test_widgets):
            try:
                test()
            except Exception:
                print(f"  ERROR in {test.__name__}:")
                traceback.print_exc()
                _failures.append(test.__name__)

    print()
    if _skipped:
        print(f"skipped: {', '.join(_skipped)}")
    if _failures:
        print(f"FAILED ({len(_failures)}): {', '.join(_failures)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
