"""spec 02 §3.3 — Validate that ≥99.5% of chosen routes are in candidate sets.

For every SET decision in decision_points_v2.parquet, compute
feasible_actions(focal_train, focal_signal, t, ...) and check that
chosen_route_id is in the result.

Target: coverage ≥ 99.5%. Lower → candidate algorithm too restrictive,
must widen rules before training.

Output: outputs/decision_points/candidate_coverage.json

Usage:
    python scripts/mdp/02_validate_candidates.py
    python scripts/mdp/02_validate_candidates.py --verbose
"""
from __future__ import annotations
import argparse
import json
import sys
import time as _time
from pathlib import Path

import pandas as pd

# Make src/railrl importable
SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC))

from railrl import config as C
from railrl.mdp.action import RouteIndex, build_pass_route_history, validate_candidates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", "-v", action="store_true")
    ap.add_argument("--decision-points", type=Path,
                     default=C.DECISION_POINTS_V2_PARQUET)
    ap.add_argument("--routes", type=Path,
                     default=C.ROUTES_CLEAN_PARQUET)
    ap.add_argument("--out", type=Path,
                     default=C.DECISION_POINTS_DIR / "candidate_coverage.json")
    args = ap.parse_args()

    print(f"[02] PROJECT_ROOT = {C.PROJECT_ROOT}")
    print(f"[02] decision_points = {args.decision_points}")
    print(f"[02] routes          = {args.routes}")
    print()

    # Load
    print("[02] Loading decision_points_v2.parquet")
    dp = pd.read_parquet(args.decision_points)
    n_set = (dp["label"] == "set").sum()
    print(f"[02]   {len(dp):,} total, {n_set:,} set decisions")

    print("[02] Loading routes_clean.parquet")
    routes = pd.read_parquet(args.routes)
    print(f"[02]   {len(routes):,} routes")

    # Build indexes
    print("[02] Building RouteIndex")
    route_index = RouteIndex(routes)
    print(f"[02]   {len(route_index.by_start_signal)} unique start signals indexed")

    # Build per-train PR history (used for prev_routes filter)
    print("[02] Loading PR events for history")
    pr = pd.read_parquet(C.DECISION_EVENTS_PARQUET)
    history = build_pass_route_history(pr)
    print(f"[02]   history for {len(history):,} trains")

    # Validate
    print("\n[02] Validating candidate coverage...")
    t0 = _time.time()
    result = validate_candidates(dp, route_index, history, verbose=args.verbose)
    elapsed = _time.time() - t0

    # Save
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, default=str))

    # Print summary
    print()
    print("=" * 60)
    print(f"  CANDIDATE COVERAGE")
    print(f"  total set decisions:   {result['n_total']:,}")
    print(f"  chosen ∈ candidates:   {result['n_in_candidates']:,}")
    print(f"  chosen ∉ candidates:   {result['n_not_in_candidates']:,}")
    print(f"  coverage:              {result['coverage_pct']:.3f}%")
    print(f"  passes 99.5%:          {'✓ YES' if result['passes_99_5_threshold'] else '✗ NO'}")
    print(f"  elapsed:               {elapsed:.1f}s")
    print(f"  output:                {args.out}")
    print("=" * 60)

    if not result["passes_99_5_threshold"]:
        print("\n  ⚠ Coverage below 99.5% target. Sample mismatches:")
        for m in result["mismatch_sample"][:10]:
            print(f"    t={m['t']} train={m['focal_train']} sig={m['focal_signal']} "
                  f"chosen={m['chosen']} n_cands={m['n_candidates']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
