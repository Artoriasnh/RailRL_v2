"""Stage 8/10 feasibility probe — can we build AND VALIDATE the P2.6 simulator?

The simulator (spec 05 §14) needs 4 parameter tables (route_running_time /
platform_dwell / min_headway / aspect_clear_lag) and is gated (§14.6) by
Spearman(simulated, actual) > 0.6 on delay_change AND throughput over a held-out
month. Before committing ~3 weeks to a build, this probe quantifies the decisive
unknowns:

  A. Recorded forward-outcome coverage (validation-data density) — esp. the
     delay_change signal, suspected too sparse for the delay-Spearman gate.
  B. Reward composition — how much does r_delay actually contribute to r_total?
     (If ~0, a weak delay-validation is acceptable; lean validation on
     throughput/headway, which derive from the DENSE TD occupancy stream.)
  C. Parameter-source density — do the raw inputs support the 4 tables?
     (Derby_info gap_time fallback / route+track universe / movements dwell pairs /
     TD event density per month.)

Pure pandas/pyarrow, read-only, no torch. Run on Windows:
    python scripts/simulator/00_feasibility_probe.py
Writes outputs/simulator/feasibility_probe.json + prints a report.
"""
from __future__ import annotations
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from railrl import config as C

DECISION_REWARDS_V2 = C.REWARDS_DIR / "decision_rewards_v2.parquet"
ROUTE_TO_TC = C.DATA_DIR / "route_to_tc_all.csv" if hasattr(C, "DATA_DIR") else None
DERBY_INFO = C.REFERENCE_DIR / "Derby_info.csv" if hasattr(C, "REFERENCE_DIR") else None
MOVEMENTS = C.CACHE_DIR / "movements.parquet"
TD_DATA = C.CACHE_DIR / "td_data.parquet"


def pct(x):
    return "n/a" if x != x else f"{100 * x:.2f}%"


def section_A(report):
    """Recorded forward-outcome coverage — the validation-data density crux."""
    print("\n=== A. Recorded forward-outcome coverage (validation-data density) ===")
    want = ["t", "split", "label", "delay_change_seconds", "approach_distance",
            "next_tc_headway_seconds", "outcome"]
    pf = pq.ParquetFile(str(C.SNAPSHOTS_V2_PARQUET))
    avail = [c for c in want if c in pf.schema_arrow.names]
    df = pq.read_table(str(C.SNAPSHOTS_V2_PARQUET), columns=avail).to_pandas()
    n = len(df)
    A = {"n": int(n)}
    print(f"snapshots rows: {n:,}  (columns read: {avail})")
    for c in ["delay_change_seconds", "approach_distance", "next_tc_headway_seconds"]:
        if c in df:
            cov = float(df[c].notna().mean())
            A[c + "_coverage"] = cov
            print(f"  {c:28s} non-null: {pct(cov)}")
    if "split" in df and "delay_change_seconds" in df:
        print("  -- delay_change non-null by split --")
        for sp, g in df.groupby("split"):
            d = float(g["delay_change_seconds"].notna().mean())
            A[f"delay_cov_{sp}"] = d
            print(f"    split={str(sp):5s} n={len(g):>9,}  delay non-null {pct(d)}  "
                  f"approach {pct(float(g['approach_distance'].notna().mean()))}")
    if "t" in df and "delay_change_seconds" in df:
        ts = pd.to_datetime(df["t"], errors="coerce")
        df["_ym"] = ts.dt.strftime("%Y-%m")
        print("  -- delay_change non-null by month (validation-month density) --")
        bym = {}
        for ym, g in df.groupby("_ym"):
            d = float(g["delay_change_seconds"].notna().mean())
            bym[ym] = {"n": int(len(g)), "delay_cov": d}
            print(f"    {ym}: n={len(g):>9,}  delay non-null {pct(d)}")
        A["by_month"] = bym
    report["A_outcome_coverage"] = A


def section_B(report):
    """Reward composition — confirm whether r_delay materially drives r_total."""
    print("\n=== B. Reward composition (does r_delay matter for Tier-3 reward-delta?) ===")
    comp = ["r_delay", "r_throughput", "r_headway", "r_wait", "r_total"]
    src = DECISION_REWARDS_V2 if DECISION_REWARDS_V2.exists() else C.SNAPSHOTS_V2_PARQUET
    pf = pq.ParquetFile(str(src))
    avail = [c for c in comp if c in pf.schema_arrow.names]
    df = pq.read_table(str(src), columns=avail).to_pandas()
    B = {"source": Path(src).name, "n": int(len(df))}
    print(f"source: {Path(src).name}  rows: {len(df):,}")
    for c in avail:
        m = float(df[c].mean())
        B[c + "_mean"] = m
        print(f"  {c:14s} mean: {m:+.4f}")
    if {"r_delay", "r_total"} <= set(avail):
        share = abs(df["r_delay"].mean()) / max(abs(df["r_total"].mean()), 1e-9)
        B["abs_r_delay_over_abs_r_total"] = float(share)
        print(f"  |mean r_delay| / |mean r_total| = {share:.3f}  "
              f"(small ⇒ delay barely drives reward ⇒ weak delay-validation tolerable)")
    report["B_reward_composition"] = B


def section_C(report):
    """Parameter-source density for the 4 simulator tables."""
    print("\n=== C. Parameter-source density (can the 4 tables be built?) ===")
    Cc = {}

    # route_running_time fallback = Derby_info gap_time(s)
    try:
        di = pd.read_csv(DERBY_INFO)
        gcol = next((c for c in di.columns if "gap_time" in c), None)
        cov = float(di[gcol].notna().mean()) if gcol else float("nan")
        Cc["derby_info"] = {"n_routes": int(len(di)), "gap_time_col": gcol,
                            "gap_time_coverage": cov}
        print(f"  Derby_info: {len(di)} routes, gap_time('{gcol}') non-null {pct(cov)} "
              f"→ route_running_time fallback")
    except Exception as e:
        print(f"  [Derby_info] skipped: {e}")

    # route + track universe (route_running_time cells / min_headway cells)
    try:
        rt = pd.read_csv(C.DATA_DIR / "route_to_tc_all.csv")
        nr = rt["route"].nunique() if "route" in rt else None
        ntr = rt["track"].nunique() if "track" in rt else None
        Cc["route_to_tc"] = {"rows": int(len(rt)), "distinct_routes": int(nr or 0),
                             "distinct_tracks": int(ntr or 0)}
        print(f"  route_to_tc: {len(rt)} rows, {nr} routes, {ntr} tracks "
              f"(min_headway universe = {ntr} cells, spec says 249)")
    except Exception as e:
        print(f"  [route_to_tc] skipped: {e}")

    # platform_dwell pairs from movements
    try:
        mv = pq.read_table(str(MOVEMENTS)).to_pandas()
        evcol = next((c for c in mv.columns if "event" in c.lower()), None)
        pcol = next((c for c in mv.columns if c.lower() == "platform"), None)
        info = {"rows": int(len(mv)), "event_col": evcol, "platform_col": pcol}
        if evcol:
            vc = mv[evcol].astype(str).str.upper().value_counts().to_dict()
            info["event_type_counts"] = {k: int(v) for k, v in list(vc.items())[:8]}
            print(f"  movements: {len(mv):,} rows; event types: "
                  f"{ {k: int(v) for k,v in list(vc.items())[:6]} }")
        if pcol:
            info["platform_with_value"] = int(mv[pcol].notna().sum())
            print(f"    rows with platform: {int(mv[pcol].notna().sum()):,} "
                  f"(platform_dwell source)")
        Cc["movements"] = info
    except Exception as e:
        print(f"  [movements] skipped: {e}")

    # TD density (occupancy / headway / running-time raw material — the DENSE signal)
    try:
        tcol_guess = None
        pf = pq.ParquetFile(str(TD_DATA))
        names = pf.schema_arrow.names
        tcol_guess = "time" if "time" in names else None
        td = pq.read_table(str(TD_DATA), columns=[tcol_guess]).to_pandas() if tcol_guess else None
        if td is not None:
            ts = pd.to_datetime(td[tcol_guess], errors="coerce")
            ym = ts.dt.strftime("%Y-%m").value_counts().sort_index()
            Cc["td"] = {"total_events": int(len(td)),
                        "events_by_month": {k: int(v) for k, v in ym.items()}}
            print(f"  TD events: {len(td):,} total (dense occupancy/headway source)")
            print("    by month: " + ", ".join(f"{k}:{v:,}" for k, v in ym.items()))
    except Exception as e:
        print(f"  [td_data] skipped: {e}")

    report["C_parameter_sources"] = Cc


def main():
    report = {}
    for fn in (section_A, section_B, section_C):
        try:
            fn(report)
        except Exception:
            print(f"\n[!] {fn.__name__} failed:\n{traceback.format_exc()}")
    out = C.SIMULATOR_DIR / "feasibility_probe.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n→ wrote {out}")
    print("\nKey reads for go/no-go:")
    print("  - A: delay_change coverage (esp. in val/test months) — feeds the §14.6 "
          "Spearman-on-delay gate.")
    print("  - B: |r_delay|/|r_total| — if tiny, validate on throughput/headway (dense TD) "
          "and treat delay as best-effort.")
    print("  - C: TD event density per month — confirms occupancy/headway validation is dense.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
