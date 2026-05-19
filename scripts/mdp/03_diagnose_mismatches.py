"""Diagnose Stage 2 candidate-mismatch + wait-inflation issues.

Two questions:
  1. Is RouteIndex.by_start_signal key format consistent with
     decision_events.signal_no? (n_candidates=0 mystery)
  2. Why is n_wait so high (1.5M vs expected ~181k)?

Usage: python scripts/mdp/03_diagnose_mismatches.py
"""
from __future__ import annotations
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC))

import pandas as pd
from railrl import config as C
from railrl.mdp.action import RouteIndex


def main():
    print("=" * 70)
    print("  Q1: routes_clean.start_signal vs decision_events.signal_no")
    print("=" * 70)

    routes = pd.read_parquet(C.ROUTES_CLEAN_PARQUET)
    print(f"\nroutes_clean columns: {list(routes.columns)}")
    print(f"shape: {routes.shape}")

    # Look at start_signal/signal_no format
    sig_col = "start_signal" if "start_signal" in routes.columns else "signal_no"
    print(f"\nUsing column: {sig_col}")
    print(f"sample values (first 10):  {routes[sig_col].head(10).tolist()}")
    print(f"dtype: {routes[sig_col].dtype}")

    # Check specific signals from mismatch sample
    test_sigs = ["5037", "5040", "5045", "5475", "5302", "5329", "5101"]
    for s in test_sigs:
        m = routes[routes[sig_col].astype(str) == s]
        n_with_cls = m["cls"].value_counts().to_dict() if "cls" in m.columns else {}
        print(f"  routes from '{s}': {len(m)} rows, cls = {n_with_cls}")

    print()
    print("=" * 70)
    print("  Q2: decision_events.signal_no format")
    print("=" * 70)
    de = pd.read_parquet(C.DECISION_EVENTS_PARQUET)
    print(f"signal_no dtype: {de['signal_no'].dtype}")
    print(f"sample: {de['signal_no'].head(10).tolist()}")
    print(f"unique count: {de['signal_no'].nunique()}")

    print()
    print("=" * 70)
    print("  Q3: RouteIndex lookup")
    print("=" * 70)
    ri = RouteIndex(routes)
    print(f"by_start_signal keys (first 10): {list(ri.by_start_signal.keys())[:10]}")
    print(f"key count: {len(ri.by_start_signal)}")

    for s in test_sigs:
        n = len(ri.routes_from(s))
        print(f"  routes_from('{s}'): {n}")

    print()
    print("=" * 70)
    print("  Q4: focal_signal coverage in routes_clean")
    print("=" * 70)
    dp = pd.read_parquet(C.DECISION_POINTS_V2_PARQUET)
    set_dp = dp[dp["label"] == "set"]
    unique_focals = set_dp["focal_signal"].astype(str).unique()
    indexed_keys = set(ri.by_start_signal.keys())
    missing = [s for s in unique_focals if s not in indexed_keys]
    print(f"set decisions cover {len(unique_focals)} unique focal_signals")
    print(f"routes_clean indexes {len(indexed_keys)} unique start_signals")
    print(f"focal_signals NOT in route index: {len(missing)}")
    if missing:
        print(f"  sample missing: {missing[:20]}")
        # how many decisions affected
        affected = set_dp[set_dp["focal_signal"].astype(str).isin(missing)]
        print(f"  affected set decisions: {len(affected):,} ({len(affected)/len(set_dp)*100:.2f}%)")

    print()
    print("=" * 70)
    print("  Q5: wait inflation analysis")
    print("=" * 70)
    wait = dp[dp["label"] == "wait"].copy()
    print(f"total wait: {len(wait):,}")
    print()
    print("per (focal_train, focal_signal) wait count distribution:")
    per_ts = wait.groupby(["focal_train", "focal_signal"]).size()
    print(per_ts.describe().round(2))
    print()
    print(f"  P50: {per_ts.median():.0f}")
    print(f"  P90: {per_ts.quantile(0.9):.0f}")
    print(f"  P99: {per_ts.quantile(0.99):.0f}")
    print(f"  max: {per_ts.max():,}")
    print()
    print("Top 10 (T, S) pairs by wait count:")
    print(per_ts.sort_values(ascending=False).head(10).to_string())

    # Specific suspects: generic IDs like 0S00, OSOO, UNKNOWN
    print()
    print("Wait counts for generic-looking train_ids:")
    for tid in ["0S00", "OSOO", "5C00", "5C05", "UNKNOWN_TRAIN"]:
        n = len(wait[wait["focal_train"] == tid])
        if n > 0:
            print(f"  {tid}: {n:,} wait samples")


if __name__ == "__main__":
    main()
