"""Command-line entry points for the data pipeline.

Used by scripts/data/01-03 and console-script entry points in pyproject.toml.

For deeper modules (04-15), scripts call the modules directly via
src/railrl/data/ (or via the back-compat shim src/railrl/p2_data_eng/).
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

from . import config as C


def inventory_main(argv: list[str] | None = None) -> int:
    """Stage 1 inventory — TD + Movements descriptive statistics."""
    ap = argparse.ArgumentParser(prog="railrl-inventory",
                                  description="Run inventory pass over TD + Movements.")
    ap.add_argument("--nrows", type=int, default=None,
                     help="Limit TD scan to first N rows (for sanity check).")
    ap.add_argument("--show-paths", action="store_true",
                     help="Print resolved paths and exit.")
    args = ap.parse_args(argv)

    if args.show_paths:
        print(C.describe())
        return 0

    from .data.inventory import stream_td_inventory, movements_inventory

    print(f"[inventory] TD scan: {C.TD_CSV}")
    td_summary = stream_td_inventory(nrows=args.nrows)
    C.INVENTORY_TD_JSON.write_text(json.dumps(td_summary, indent=2, default=str))
    print(f"[inventory] -> {C.INVENTORY_TD_JSON}")

    print(f"[inventory] Movements scan: {C.MOVEMENTS_CSV}")
    mv_summary = movements_inventory()
    C.INVENTORY_MOVEMENTS_JSON.write_text(json.dumps(mv_summary, indent=2, default=str))
    print(f"[inventory] -> {C.INVENTORY_MOVEMENTS_JSON}")
    return 0


def decisions_main(argv: list[str] | None = None) -> int:
    """Stage 2 decision-event extraction — PR rows from TD with parsed fields."""
    ap = argparse.ArgumentParser(prog="railrl-decisions",
                                  description="Extract Panel_Request decision events.")
    ap.add_argument("--nrows", type=int, default=None)
    ap.add_argument("--show-paths", action="store_true")
    args = ap.parse_args(argv)

    if args.show_paths:
        print(C.describe())
        return 0

    from .data.decisions import extract, summarize

    print(f"[decisions] extracting from {C.TD_CSV}")
    df = extract(nrows=args.nrows)

    print(f"[decisions] writing {len(df):,} rows to {C.DECISION_EVENTS_PARQUET}")
    df.to_parquet(C.DECISION_EVENTS_PARQUET, index=False)

    summary = summarize(df)
    C.DECISION_EVENTS_SUMMARY.write_text(json.dumps(summary, indent=2, default=str))
    print(f"[decisions] -> {C.DECISION_EVENTS_SUMMARY}")
    return 0


def infrastructure_main(argv: list[str] | None = None) -> int:
    """Stage 3 infrastructure parse — route_to_tc → clean parquet tables."""
    ap = argparse.ArgumentParser(prog="railrl-infrastructure",
                                  description="Parse route_to_tc_all.csv into 4 inventory tables.")
    ap.add_argument("--show-paths", action="store_true")
    args = ap.parse_args(argv)

    if args.show_paths:
        print(C.describe())
        return 0

    from .data.infrastructure import run
    run()
    return 0


# Console-script entry points (resolved via pyproject.toml)
def _entry_inventory():      sys.exit(inventory_main())
def _entry_decisions():      sys.exit(decisions_main())
def _entry_infrastructure(): sys.exit(infrastructure_main())
