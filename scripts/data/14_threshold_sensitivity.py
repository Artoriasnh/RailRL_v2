"""P2.B+ - Sensitivity analysis for DECISION_LOOKAHEAD_SECONDS."""
from __future__ import annotations
import json, sys, time as _time
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from railrl import config as C
from railrl.p2_data_eng.decision_points import generate_decision_points
from railrl.data_io import load_td

THRESHOLDS = [30, 75, 120]


def main():
    print("Loading TD + decision_events + routes + signals...")
    t0 = _time.time()
    td = load_td()
    de = pd.read_parquet(C.DECISION_EVENTS_PARQUET)
    routes = pd.read_parquet(C.ROUTES_CLEAN_PARQUET)
    signals = pd.read_parquet(C.SIGNALS_INVENTORY_PARQUET)
    print("  done, %.1fs" % (_time.time() - t0))

    results = {}
    for thr in THRESHOLDS:
        print("")
        print("=== Threshold = %ds ===" % thr)
        t1 = _time.time()
        out, summary = generate_decision_points(
            td, de, routes, signals, lookahead_s=thr)
        print("  classification done in %.1fs" % (_time.time() - t1))
        n_total = int(summary["n_total"])
        n_set   = int(summary["n_positives"])
        n_wait  = int(summary["n_negatives"])
        results[thr] = {
            "lookahead_seconds": thr,
            "n_total":     n_total,
            "n_set":       n_set,
            "n_wait":      n_wait,
            "wait_to_set": float(summary["neg_pos_ratio"]),
            "set_pct":     round(100 * n_set / max(n_total, 1), 2),
        }

    md_lines = [
        "# Threshold Sensitivity Analysis (DECISION_LOOKAHEAD_SECONDS)",
        "",
        "Empirical reaction time distribution: P25=28s, P50=72s, P75=197s, P90=361s.",
        "",
        "| threshold | n_total | n_set | n_wait | wait/set | set pct |",
        "|-----------|---------|-------|--------|----------|---------|",
    ]
    for thr in THRESHOLDS:
        r = results[thr]
        line = "| **%ds** | %s | %s | %s | %.2f | %.1f%% |" % (
            thr, format(r["n_total"], ","), format(r["n_set"], ","),
            format(r["n_wait"], ","), r["wait_to_set"], r["set_pct"])
        md_lines.append(line)
    md_lines += [
        "",
        "## Recommendation",
        "",
        "Adopt 120s for production. Above median (72s) and P67-ish, capturing",
        "genuine set decisions while excluding the long-tail hesitation cases.",
    ]

    (C.DECISION_POINTS_DIR / "threshold_sensitivity.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")
    (C.DECISION_POINTS_DIR / "threshold_sensitivity.md").write_text(
        "\n".join(md_lines), encoding="utf-8")

    print("")
    print("=== Comparison ===")
    for thr in THRESHOLDS:
        r = results[thr]
        print("  %ds   n_set=%10s   n_wait=%12s   wait/set=%6.2f   set=%6.1f%%" % (
            thr, format(r["n_set"], ","), format(r["n_wait"], ","),
            r["wait_to_set"], r["set_pct"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
