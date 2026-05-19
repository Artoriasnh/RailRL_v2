"""P2.4 Iter A — Empirical reward-threshold calibration."""
from __future__ import annotations
import argparse
import json
import sys
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from railrl import config as C
from railrl.p2_data_eng.event_stream        import AssetIndex, EventTokenStream
from railrl.p2_data_eng.snapshot            import StaticGraphView
from railrl.p2_data_eng.reward_calibration  import (
    compute_headway_distribution,
    compute_approach_distance_distribution,
    compute_tiploc_lag_distribution,
    build_train_position_lookup_from_td,
    derive_thresholds,
)


def _maybe_plot_hist(arr, title, xlabel, path, log_x=False, clip=None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    if arr.size == 0:
        return False
    a = arr.copy()
    if clip is not None:
        a = a[a <= clip]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    if log_x:
        a = a[a > 0]
        ax.hist(a, bins=80)
        ax.set_xscale("log")
    else:
        ax.hist(a, bins=80)
    for p, ls in [(5, ":"), (50, "-"), (90, "--"), (99, "-.")]:
        v = float(np.percentile(a, p))
        ax.axvline(v, color="r", linestyle=ls, alpha=0.5,
                   label="P" + str(p) + "=" + str(round(v, 1)))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


def _write_summary_md(thresholds, path):
    h  = thresholds["headway"]
    a  = thresholds["approach_distance"]
    tl = thresholds["tiploc_lag"]
    h_p = h["percentiles"]
    a_p = a["percentiles"]
    t_p = tl["percentiles"]
    bp  = a["d_gate_breakpoints"]

    lines = []
    lines.append("# P2.4 Iter A — Reward Threshold Calibration\n")
    lines.append("Empirically derived from full-year Derby workstation data.\n")
    lines.append("## H_min — minimum acceptable headway (r_headway)\n")
    lines.append("| Statistic | Value |")
    lines.append("|-----------|-------|")
    lines.append("| n pairs | " + format(h["n_pairs"], ",") + " |")
    lines.append("| P1  | " + str(round(h_p["p1"], 1))  + " s |")
    lines.append("| **P5 (= H_min)** | **" + str(round(h["H_min_seconds_used"], 1)) + " s** |")
    lines.append("| P10 | " + str(round(h_p["p10"], 1)) + " s |")
    lines.append("| P50 | " + str(round(h_p["p50"], 1)) + " s |")
    lines.append("| P90 | " + str(round(h_p["p90"], 1)) + " s |")
    lines.append("| P99 | " + str(round(h_p["p99"], 1)) + " s |")
    lines.append("")
    lines.append("Reference values from UK railway standards (paper context):")
    lines.append("- Multi-aspect colour-light mainline minimum signaling headway: 90-120s "
                 "(Network Rail Industry Standard RIS-0786-RIG).")
    lines.append("- Junction headway: 90-150s typical.")
    lines.append("- TPWS overlap clearance: 30-45s.\n")
    lines.append("## d-gate — causal-attribution distance for r_delay\n")
    lines.append("| Statistic | Value |")
    lines.append("|-----------|-------|")
    lines.append("| n decisions sampled | " + format(a["n_decisions_sampled"], ",") + " |")
    lines.append("| n with computable d | " + format(a["n_with_distance"], ",") + " |")
    lines.append("| P10 | " + str(round(a_p["p10"], 1)) + " hops |")
    lines.append("| **P50 (= gate-0.5 boundary)** | **" + str(round(a_p["p50"], 1)) + " hops** |")
    lines.append("| **P90 (= gate-0.1 boundary)** | **" + str(round(a_p["p90"], 1)) + " hops** |")
    lines.append("| P95 | " + str(round(a_p["p95"], 1)) + " hops |")
    lines.append("| P99 | " + str(round(a_p["p99"], 1)) + " hops |\n")
    lines.append("Decided d-gate function:\n")
    lines.append("| Distance d at decision | gate(d) | Interpretation |")
    lines.append("|------------------------|---------|----------------|")
    lines.append("| 0-2 | 1.0 | Train is here NOW; this PR fully responsible |")
    lines.append("| 3-" + str(bp["gate_0.5_max"]) + " | 0.5 | Approaching; partial responsibility |")
    lines.append("| " + str(bp["gate_0.5_max"]+1) + "-" + str(bp["gate_0.1_max"]) + " | 0.1 | Far; minimal responsibility |")
    lines.append("| > " + str(bp["gate_0.1_max"]) + " | 0.0 | Pre-staging; not responsible |\n")
    lines.append("## Reward observation window — TIPLOC-lag P99\n")
    lines.append("| Statistic | Value |")
    lines.append("|-----------|-------|")
    lines.append("| n lags | " + format(tl["n_lags"], ",") + " |")
    if tl["n_lags"] > 0:
        lines.append("| P50 | " + str(round(t_p["p50"], 1)) + " s |")
        lines.append("| P90 | " + str(round(t_p["p90"], 1)) + " s |")
        lines.append("| P95 | " + str(round(t_p["p95"], 1)) + " s |")
        lines.append("| **P99 (= window)** | **" + str(round(tl["window_seconds_used"], 1)) + " s** |")
    lines.append("")
    lines.append("## Final calibrated parameters\n")
    lines.append("```json")
    lines.append(json.dumps({
        "H_min_seconds":         round(h["H_min_seconds_used"], 1),
        "d_gate_0.5_max":        bp["gate_0.5_max"],
        "d_gate_0.1_max":        bp["gate_0.1_max"],
        "reward_window_seconds": round(tl["window_seconds_used"], 1) if tl["n_lags"] > 0 else None,
    }, indent=2))
    lines.append("```")
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--sample-size", type=int, default=50_000)
    args = parser.parse_args()

    print("[1/5] Loading event stream + asset index ...")
    t0 = _time.time()
    es = EventTokenStream.load()
    ai = AssetIndex.load()
    print(f"      {es.n_tokens:,} tokens, {ai.summary()['n_assets']} assets, "
          f"{_time.time()-t0:.1f}s")

    print("[2/5] Computing headway distribution ...")
    t0 = _time.time()
    headway = compute_headway_distribution(es, ai)
    print(f"      {headway.size:,} headway pairs, {_time.time()-t0:.1f}s")

    print("[3/5] Loading TD + building train position lookup ...")
    t0 = _time.time()
    train_pos = build_train_position_lookup_from_td(C.TD_PARQUET)
    print(f"      {len(train_pos):,} train-step events, {_time.time()-t0:.1f}s")

    print("[4/5] Computing approach-distance distribution ...")
    t0 = _time.time()
    sg = StaticGraphView.load()
    decisions = pd.read_parquet(C.DECISION_POINTS_PARQUET)
    pr = decisions[decisions["label"] == "set"].copy()
    print(f"      {len(pr):,} PR decisions, sampling {args.sample_size:,} ...")
    approach = compute_approach_distance_distribution(
        pr, train_pos, sg, sample_size=args.sample_size
    )
    print(f"      {approach['distance'].notna().sum():,} computable, "
          f"{_time.time()-t0:.1f}s")

    print("[5/5] Computing TIPLOC-lag distribution ...")
    t0 = _time.time()
    mv = pd.read_csv(C.MOVEMENTS_CSV, usecols=["train_id", "actual_timestamp"])
    # TRUST train_id chars [2:6] = 4-char headcode (matches PR focal_train)
    mv["train_id"] = mv["train_id"].astype(str).str[2:6]
    tiploc_lags = compute_tiploc_lag_distribution(
        pr, mv, sample_size=args.sample_size
    )
    print(f"      {tiploc_lags.size:,} usable lags, {_time.time()-t0:.1f}s")

    thresholds = derive_thresholds(headway, approach, tiploc_lags)

    C.CALIBRATION_JSON.parent.mkdir(parents=True, exist_ok=True)
    C.CALIBRATION_JSON.write_text(json.dumps(thresholds, indent=2), encoding="utf-8")
    print(f"\nWrote calibration -> {C.CALIBRATION_JSON}")

    if not args.no_plots:
        ok1 = _maybe_plot_hist(
            headway, "Headway distribution - Derby workstation",
            "headway (seconds)", C.CALIBRATION_HEADWAY_HIST_PNG,
            log_x=True, clip=3600.0,
        )
        ok2 = _maybe_plot_hist(
            approach["distance"].dropna().to_numpy(dtype=np.float64),
            "Approach distance at decision",
            "hop distance (TC->focal_signal)",
            C.CALIBRATION_APPROACH_HIST_PNG,
        )
        ok3 = _maybe_plot_hist(
            tiploc_lags, "TIPLOC report lag",
            "seconds (clipped at 1800)",
            C.CALIBRATION_TIPLOC_HIST_PNG, clip=1800.0,
        )
        plotted = sum([ok1, ok2, ok3])
        print(f"Wrote {plotted}/3 histogram PNGs.")

    _write_summary_md(thresholds, C.CALIBRATION_SUMMARY_MD)
    print(f"Wrote markdown summary -> {C.CALIBRATION_SUMMARY_MD}")

    h_min = thresholds["headway"]["H_min_seconds_used"]
    bp    = thresholds["approach_distance"]["d_gate_breakpoints"]
    win   = thresholds["tiploc_lag"]["window_seconds_used"]
    print("\n=== Calibrated thresholds ===")
    print(f"  H_min               = {h_min:.1f} s")
    print(f"  d_gate 0.5/0.1 max  = {bp['gate_0.5_max']} / {bp['gate_0.1_max']} hops")
    print(f"  reward window (P99) = {win:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
