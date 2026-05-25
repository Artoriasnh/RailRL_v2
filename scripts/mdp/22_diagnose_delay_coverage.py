"""Diagnose WHY delay_change (r_delay) coverage collapses to ~0% in Mar-Jul 2023.

Movements has dense, bracketable, same-day timing data in ALL months (~3 pts/train,
~2 min gaps, >99% gaps <70 min, days match the decision segments). Yet Apr-Jul
decisions get ~0.2% delay_change while Aug+ gets ~9-14%. That points to a
decision↔Movements MATCHING bug, not missing data.

This replays the exact matching from data/reward_features.compute_delay_changes
and tallies the failure reason PER MONTH, so we can see whether Mar-Jul fails at:
  - no_match_headcode : decision focal_train headcode absent from Movements
  - no_trust_in_window: headcode present but no TRUST run within ±70 min of t
  - no_baseline       : matched run has no timing point <= t
  - no_followup       : matched run has no timing point > t
  - out_window        : bracket endpoints exist but > 70 min from t
  - bracketed         : success (delay_change computed)

Read-only. Run on Windows (needs pyarrow for the big snapshot read):
    python scripts/mdp/22_diagnose_delay_coverage.py
    python scripts/mdp/22_diagnose_delay_coverage.py --max 400000   # faster subsample
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from railrl import config as C

WINDOW_S = 4202.0   # matches reward_features.compute_delay_changes default (~70 min)


def build_movements_index(movements_csv, run_gap_s=7200.0):
    """Build per-RUN timing index. run_gap_s splits each train_id's points into
    single passes (FIX). Pass a huge run_gap_s (e.g. 1e12) to DISABLE splitting and
    reproduce the pre-fix buggy behaviour (group by raw train_id) for comparison."""
    mv = pd.read_csv(movements_csv,
                     usecols=["train_id", "actual_timestamp", "planned_timestamp"],
                     low_memory=False)
    mv["headcode"] = mv["train_id"].astype(str).str[2:6]
    mv["actual"] = pd.to_datetime(mv["actual_timestamp"], errors="coerce")
    mv["planned"] = pd.to_datetime(mv["planned_timestamp"], errors="coerce")
    mv = mv.dropna(subset=["actual", "planned", "headcode"])
    mv = mv[mv["headcode"].str.len() == 4]
    mv["actual_ns"] = mv["actual"].astype("datetime64[ns]").astype("int64")
    mv["delay_s"] = (mv["actual"] - mv["planned"]).dt.total_seconds()
    run_gap_ns = int(run_gap_s * 1e9)
    by_run = {}
    headcode_to_runs: dict = {}
    for tid, sub in mv.groupby("train_id"):
        sub = sub.sort_values("actual_ns")
        arr_t = sub["actual_ns"].to_numpy(np.int64)
        arr_d = sub["delay_s"].to_numpy(np.float64)
        hc = str(sub["headcode"].iloc[0])
        splits = np.where(np.diff(arr_t) > run_gap_ns)[0] + 1
        for k, idx in enumerate(np.split(np.arange(arr_t.size), splits)):
            if idx.size == 0:
                continue
            rt, rd = arr_t[idx], arr_d[idx]
            rk = (tid, k)
            by_run[rk] = (rt, rd)
            headcode_to_runs.setdefault(hc, []).append((int(rt[0]), int(rt[-1]), rk))
    for hc in headcode_to_runs:
        headcode_to_runs[hc].sort()
    print(f"  Movements: {len(mv):,} rows, {len(by_run):,} runs "
          f"(gap-split @ {run_gap_s/3600:.1f}h from {mv['train_id'].nunique():,} train_ids), "
          f"{len(headcode_to_runs):,} headcodes")
    return by_run, headcode_to_runs


def classify(hc, t, by_run, headcode_to_runs, window_ns):
    candidates = headcode_to_runs.get(hc)
    if not candidates:
        return "no_match_headcode", np.nan
    best_run, best_dist = None, None
    for t_first, t_last, rk in candidates:
        if (t_first - window_ns) <= t <= (t_last + window_ns):
            center = (t_first + t_last) // 2
            dist = abs(t - center)
            if best_dist is None or dist < best_dist:
                best_dist, best_run = dist, rk
    if best_run is None:
        return "no_trust_in_window", np.nan
    arr_t, _ = by_run[best_run]
    off_min = (int(arr_t[0]) - t) / 6.0e10   # +ve = matched run's 1st point is AFTER decision t
    j = int(np.searchsorted(arr_t, t, side="right"))
    if j == 0:
        return "no_baseline", off_min
    if j >= arr_t.size:
        return "no_followup", off_min
    if (arr_t[j] - t) > window_ns or (t - arr_t[j - 1]) > window_ns:
        return "out_window", off_min
    return "bracketed", off_min


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=None, help="subsample N decisions (speed)")
    ap.add_argument("--movements", type=str, default=str(C.DATA_DIR / "Movements.csv"))
    ap.add_argument("--run-gap", type=float, default=7200.0,
                    help="seconds; split each train_id into runs at gaps > this (FIX, "
                         "default 2h). Pass a huge value (e.g. 1e12) to reproduce the "
                         "pre-fix buggy behaviour for before/after comparison.")
    args = ap.parse_args()

    import pyarrow.parquet as pq
    print("loading decisions (focal_train, t) from snapshots_v2 ...")
    cols = ["focal_train", "t"]
    df = pq.read_table(str(C.SNAPSHOTS_V2_PARQUET), columns=cols).to_pandas()
    if args.max and len(df) > args.max:
        df = df.sample(args.max, random_state=0).reset_index(drop=True)
    df["t_ns"] = pd.to_datetime(df["t"]).astype("datetime64[ns]").astype("int64")
    df["ym"] = pd.to_datetime(df["t"]).dt.strftime("%Y-%m")
    print(f"  {len(df):,} decisions")

    print("building Movements index ...")
    by_run, hc2r = build_movements_index(Path(args.movements), args.run_gap)
    window_ns = int(WINDOW_S * 1e9)

    trains = df["focal_train"].astype(str).to_numpy()
    times = df["t_ns"].to_numpy()
    yms = df["ym"].to_numpy()
    reasons = np.empty(len(df), dtype=object)
    offsets = np.full(len(df), np.nan)
    for i in range(len(df)):
        reasons[i], offsets[i] = classify(trains[i], times[i], by_run, hc2r, window_ns)
        if (i + 1) % 200000 == 0:
            print(f"  ...{i+1:,}")
    df["reason"] = reasons
    df["off_min"] = offsets

    cats = ["bracketed", "no_match_headcode", "no_trust_in_window",
            "no_baseline", "no_followup", "out_window"]
    print("\n=== delay_change match outcome by month (% of decisions) ===")
    print(f"{'month':8s} {'n':>8s} " + " ".join(f"{c[:11]:>12s}" for c in cats))
    for ym, g in df.groupby("ym"):
        vc = g["reason"].value_counts()
        row = " ".join(f"{100*vc.get(c,0)/len(g):11.2f}%" for c in cats)
        print(f"{ym:8s} {len(g):>8,} {row}")
    print("\noverall:")
    vc = df["reason"].value_counts()
    for c in cats:
        print(f"  {c:20s} {vc.get(c,0):>9,}  ({100*vc.get(c,0)/len(df):.2f}%)")
    print("\n=== median offset (matched-run 1st point − decision t), minutes, by month ===")
    print("    +ve ⇒ Movements points come AFTER the decision (decision earlier than timing pts)")
    print("    a large uniform +offset in Mar-Jul but ~0 in Aug+ ⇒ a period-specific CLOCK offset")
    for ym, g in df.groupby("ym"):
        m = g["off_min"].dropna()
        if len(m):
            print(f"  {ym}: n_matched={len(m):>8,}  median={m.median():8.1f}m  "
                  f"p25={m.quantile(.25):7.1f}  p75={m.quantile(.75):7.1f}")
    # per-DAY offset around the boundary months → pin the EXACT start/end of the +1h
    bdf = df[df["ym"].isin(["2023-03", "2023-04", "2023-07", "2023-08"])].copy()
    if len(bdf):
        bdf["ymd"] = pd.to_datetime(bdf["t"]).dt.strftime("%Y-%m-%d")
        print("\n=== per-DAY median offset at boundary months (pin exact +1h start/end) ===")
        for ymd, g in bdf.groupby("ymd"):
            m = g["off_min"].dropna()
            if len(m) >= 20:
                flag = "  <<+1h" if m.median() > 30 else ""
                print(f"  {ymd}: n_matched={len(m):>7,}  median={m.median():7.1f}m{flag}")

    print("\nRead: no_baseline dominating Mar-Jul + a large +median offset ⇒ decisions are "
          "systematically EARLIER than Movements there (Movements +1h, 2nd bug). Per-day "
          "block pins the exact boundary (expected to fall inside the data gaps).")


if __name__ == "__main__":
    raise SystemExit(main())
