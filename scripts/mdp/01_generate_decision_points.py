"""spec 02 §2 — Generate the unified decision_points_v2.parquet table.

Combines SET triggers (~546k from PR events) and WAIT triggers (~181k from
approach events with no PR in [t, t+30s]) into one table.

Output: outputs/decision_points/decision_points_v2.parquet
        outputs/decision_points/decision_points_v2_summary.json

Usage:
    python scripts/mdp/01_generate_decision_points.py
    python scripts/mdp/01_generate_decision_points.py --nrows 1000000  # subset
    python scripts/mdp/01_generate_decision_points.py --no-wait        # skip wait triggers (set only)

Runtime: ~5-10 min on full data (the wait trigger scan iterates ~11M TD events).
"""
from __future__ import annotations
import argparse
import json
import sys
import time as _time
from pathlib import Path

import pandas as pd

# Make src/railrl importable without install
SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC))

from railrl import config as C
from railrl.mdp.trigger import generate_decision_points, summarize


def load_td_for_wait_triggers(nrows: int | None = None) -> pd.DataFrame:
    """Load TD events relevant to wait trigger generation.

    Only Track events with state=1 and non-null trainid_filled are needed.
    Returns DataFrame with cols: time, type, id, state, trainid_filled.
    """
    print(f"[01] Loading TD from cache: {C.TD_PARQUET}")
    if C.TD_PARQUET.exists():
        td = pd.read_parquet(C.TD_PARQUET,
                              columns=["time", "type", "id", "state", "trainid_filled"])
        if nrows:
            td = td.head(nrows)
    else:
        print(f"[01] cache missing; reading TD CSV (slow): {C.TD_CSV}")
        td = pd.read_csv(C.TD_CSV,
                          usecols=["time", "type", "id", "state", "trainid_filled"],
                          parse_dates=["time"],
                          nrows=nrows,
                          low_memory=False)

    print(f"[01]   loaded {len(td):,} rows")
    return td


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nrows", type=int, default=None,
                     help="Limit TD load to first N rows (for testing).")
    ap.add_argument("--no-wait", action="store_true",
                     help="Skip wait trigger generation; SET only.")
    ap.add_argument("--out", type=Path, default=C.DECISION_POINTS_V2_PARQUET,
                     help="Output parquet path.")
    args = ap.parse_args()

    out_dir = args.out.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[01] PROJECT_ROOT = {C.PROJECT_ROOT}")
    print(f"[01] Output       = {args.out}")
    print()

    # 1. Load PR events (set trigger source)
    print(f"[01] Loading decision_events from {C.DECISION_EVENTS_PARQUET}")
    pr = pd.read_parquet(C.DECISION_EVENTS_PARQUET)
    print(f"[01]   {len(pr):,} PR events")

    # 2. Load infrastructure
    print(f"[01] Loading routes_clean from {C.ROUTES_CLEAN_PARQUET}")
    routes = pd.read_parquet(C.ROUTES_CLEAN_PARQUET)
    print(f"[01]   {len(routes):,} routes")

    # 3. Load TD events for wait triggers (skip if --no-wait)
    if args.no_wait:
        print(f"[01] --no-wait set; skipping TD load + wait trigger generation")
        td = pd.DataFrame(columns=["time", "type", "id", "state", "trainid_filled"])
    else:
        td = load_td_for_wait_triggers(nrows=args.nrows)

    # 4. Generate decision points
    print(f"\n[01] Generating decision points (spec 02 §2)")
    t0 = _time.time()
    dp = generate_decision_points(
        decision_events=pr,
        td_events=td,
        routes_clean=routes,
        k_approach=C.APPROACH_K_HOPS,
        delta_wait_seconds=C.DECISION_LOOKAHEAD_SECONDS,
    )
    elapsed_gen = _time.time() - t0

    # 5. Save parquet
    print(f"\n[01] Saving to {args.out}")
    dp.to_parquet(args.out, index=False, compression="zstd")
    print(f"[01]   {len(dp):,} rows written")

    # 6. Summary JSON
    summary = summarize(dp)
    summary["elapsed_seconds"] = round(elapsed_gen, 1)
    summary["spec_constants"] = {
        "K_APPROACH": C.APPROACH_K_HOPS,
        "DELTA_WAIT": C.DECISION_LOOKAHEAD_SECONDS,
    }

    summary_path = args.out.parent / (args.out.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"[01]   summary → {summary_path}")

    # 7. Print headline
    print()
    print("=" * 60)
    print(f"  DECISION POINTS GENERATED")
    print(f"  total: {summary['n_total']:,}  "
          f"(set: {summary['n_set']:,}  wait: {summary['n_wait']:,})")
    print(f"  neg/pos ratio: {summary['neg_pos_ratio']}")
    print(f"  unique trains: {summary['n_unique_trains']:,}")
    print(f"  unique signals: {summary['n_unique_signals']:,}")
    print(f"  per-train decisions median: {summary['per_train_decisions']['median']:.1f}")
    print(f"  elapsed: {summary['elapsed_seconds']:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
