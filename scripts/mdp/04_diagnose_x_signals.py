"""Diagnose X-prefix signals.

User domain knowledge (2026-05-19):
  X063 is the "predecessor" / approach point of signal 5063.
  X-prefix signals are real concepts (not garbage):
    X063 ↔ 5063, X053 ↔ 5053, etc.

Questions this script answers:
  Q1. How many X-prefix signals exist in routes_clean / decision_points?
  Q2. Do they all have a corresponding numeric main signal?
  Q3. How many wait samples involve X-prefix vs main signal?
  Q4. Where do X-prefix signals come from in our data flow?
      (routes_clean.end_signals? somewhere else?)
  Q5. Recommendation: merge X-prefix → main signal, or keep separate?

Usage: python scripts/mdp/04_diagnose_x_signals.py
"""
from __future__ import annotations
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC))

import pandas as pd
from railrl import config as C


def main():
    print("=" * 70)
    print("  Q1. X-prefix signals in routes_clean")
    print("=" * 70)

    routes = pd.read_parquet(C.ROUTES_CLEAN_PARQUET)
    print(f"\nroutes_clean shape: {routes.shape}")
    print(f"columns: {list(routes.columns)}")

    # Look for X-prefix in all signal-related columns
    for col in ["signal_no", "start_signals", "end_signals", "start_signal", "end_signal"]:
        if col not in routes.columns:
            continue
        sample = routes[col].dropna().head(3).tolist()
        is_list = any(isinstance(v, (list, tuple)) for v in sample)
        print(f"\n  Column '{col}' (is_list={is_list}):")
        print(f"    sample: {sample}")

        # Flatten if list type
        if is_list:
            all_vals = []
            for v in routes[col].dropna():
                if isinstance(v, (list, tuple)):
                    all_vals.extend([str(x) for x in v])
                else:
                    all_vals.append(str(v))
        else:
            all_vals = [str(v) for v in routes[col].dropna()]

        x_prefix = [v for v in set(all_vals) if v.startswith("X")]
        print(f"    unique values: {len(set(all_vals))}")
        print(f"    X-prefix values: {len(x_prefix)}")
        if x_prefix:
            print(f"      sample: {sorted(x_prefix)[:20]}")

    print()
    print("=" * 70)
    print("  Q2. Map X-prefix → main signal")
    print("=" * 70)

    # Gather all X-prefix signals from end_signals (most likely source)
    x_to_main = {}
    if "end_signals" in routes.columns:
        for _, row in routes.iterrows():
            sigs = row["end_signals"]
            if not isinstance(sigs, (list, tuple)):
                sigs = [sigs] if sigs is not None else []
            for s in sigs:
                s = str(s)
                if s.startswith("X") and len(s) >= 2:
                    # X063 → 5063 (replace leading X with 5, common Derby pattern)
                    # OR X063 → 063 (strip X)
                    # Try both interpretations
                    candidate1 = "5" + s[1:]    # X063 → 5063
                    candidate2 = s[1:]          # X063 → 063
                    x_to_main[s] = (candidate1, candidate2)

    print(f"\nFound {len(x_to_main)} X-prefix end_signals")
    for x, (cand1, cand2) in sorted(x_to_main.items())[:20]:
        # Check if either candidate is a real signal in routes_clean
        signals_set = set(routes["signal_no"].astype(str)) if "signal_no" in routes.columns else set()
        marker1 = "✓" if cand1 in signals_set else "✗"
        marker2 = "✓" if cand2 in signals_set else "✗"
        print(f"  {x:6s} → X→5 form: {cand1} {marker1}    strip-X form: {cand2} {marker2}")

    print()
    print("=" * 70)
    print("  Q3. Wait samples involving X-prefix signals")
    print("=" * 70)

    dp = pd.read_parquet(C.DECISION_POINTS_V2_PARQUET)
    print(f"\ndecision_points_v2.parquet shape: {dp.shape}")

    dp_x = dp[dp["focal_signal"].astype(str).str.startswith("X")]
    print(f"  X-prefix focal_signals: {len(dp_x):,} rows")
    print(f"    by label: {dp_x['label'].value_counts().to_dict()}")
    print(f"    by trigger_type: {dp_x['trigger_type'].value_counts().to_dict()}")

    if len(dp_x):
        print(f"\n  Top X-prefix signals by sample count:")
        top_x = dp_x["focal_signal"].value_counts().head(10)
        for sig, cnt in top_x.items():
            print(f"    {sig}: {cnt:,}")

        # Are wait samples on X-prefix ALL approach-triggered? (Expected yes)
        x_set = dp_x[dp_x["label"] == "set"]
        if len(x_set):
            print(f"\n  ⚠ Found {len(x_set)} SET decisions at X-prefix signals:")
            print(f"    sample: {x_set.head(5)[['t','focal_train','focal_signal','chosen_route_id']].to_string()}")
        else:
            print(f"\n  ✓ All X-prefix samples are WAIT (no SET) — as expected for predecessor signals")

    print()
    print("=" * 70)
    print("  Q4. Where do X-prefix signals come from?")
    print("=" * 70)

    # Check if X-prefix appears in approach_tracks (via routes_clean.end_signals)
    if "end_signals" in routes.columns:
        end_sigs_all = []
        for sigs in routes["end_signals"].dropna():
            if isinstance(sigs, (list, tuple)):
                end_sigs_all.extend([str(s) for s in sigs])
            else:
                end_sigs_all.append(str(sigs))
        x_in_end = [s for s in set(end_sigs_all) if s.startswith("X")]
        print(f"\n  X-prefix in routes_clean.end_signals: {len(x_in_end)}")
        if x_in_end:
            print(f"    {sorted(x_in_end)}")

    print()
    print("=" * 70)
    print("  Q5. Recommendation")
    print("=" * 70)
    print("""
Based on user domain knowledge:
  X063 = approach predecessor of signal 5063.

If wait samples at X063 are conceptually "train approaching 5063 area",
they should probably MERGE into 5063's stats.

Options:
  (A) Keep separate    : X063 wait != 5063 wait. Pro: preserves info. Con: 2 events per pass.
  (B) Merge to main    : map X063 → 5063 in focal_signal. Pro: 1 event per pass. Con: lose granularity.
  (C) Filter X-prefix  : skip X-prefix wait samples entirely. Pro: simple. Con: data loss.

Recommendation depends on how X-prefix events show up in TRAINING:
  - If model can learn "X063 wait" as own signal → (A) keep separate
  - If X063 doesn't correspond to a real PR opportunity (signaller never
    acts at X063, only at 5063) → (B) merge OR (C) filter

User decision needed.
""")


if __name__ == "__main__":
    main()
