"""
Environmental-impact predictor — training entrypoint.

Usage:
    python main.py                        # trains the default target (ghg_total)
    python main.py --target ghg_a1a3
    python main.py --target ghg_c3
    python main.py --target ghg_c4
    python main.py --target ghg_d
    python main.py --list-targets         # print all registered targets and exit

Each target is defined in src/config.py (TARGET_CONFIGS).  Adding a new impact
category (e.g. eutrophication) requires only a new entry there — no changes here
or in the pipeline.

Output per run:
    models/{target_key}.pt          trained checkpoint
    figures/{target_key}/           training curves, scatter, residuals
    diagnostics_{target_key}.json   full run diagnostics
"""

import argparse
import sys

from src.config import TARGET_CONFIGS
from src.pipeline import run


def _list_targets():
    print("Registered targets:")
    for key, cfg in TARGET_CONFIGS.items():
        print(f"  {key:<20}  {cfg['display_name']}  [{cfg['unit']}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train an environmental-impact predictor for a given target."
    )
    parser.add_argument(
        "--target",
        default="ghg_total",
        choices=list(TARGET_CONFIGS.keys()),
        metavar="TARGET",
        help=(
            f"Target to train. One of: {', '.join(TARGET_CONFIGS.keys())}. "
            "Default: ghg_total"
        ),
    )
    parser.add_argument(
        "--list-targets",
        action="store_true",
        help="Print all registered targets and exit.",
    )
    args = parser.parse_args()

    if args.list_targets:
        _list_targets()
        sys.exit(0)

    run(args.target)
