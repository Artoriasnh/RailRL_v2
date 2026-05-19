"""Stage 1 verification — confirm all existing pipeline outputs match spec 01.

This script runs the §16 verification checklist from spec 01 against the
parquet/JSON files already in outputs/. It does NOT regenerate anything.

Usage:
    python scripts/data/00_verify_pipeline.py
    python scripts/data/00_verify_pipeline.py --verbose

Exit codes:
    0 — all checks passed
    1 — at least one check failed
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

# Make src/railrl importable without install
SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC))

from railrl import config as C


def _ok(msg):    print(f"  ✓ {msg}")
def _fail(msg):  print(f"  ✗ {msg}")
def _info(msg):  print(f"  · {msg}")


class Checker:
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.passed = 0
        self.failed = 0
        self.warnings = 0

    def expect(self, name, condition, detail=""):
        if condition:
            self.passed += 1
            if self.verbose:
                _ok(f"{name}  {detail}")
            else:
                _ok(name)
        else:
            self.failed += 1
            _fail(f"{name}  {detail}")

    def warn(self, name, detail=""):
        self.warnings += 1
        print(f"  ! {name}  {detail}")

    def section(self, title):
        print(f"\n=== {title} ===")


def check_inventory(ck: Checker):
    ck.section("Stage 1 — Inventory")

    if not C.INVENTORY_TD_JSON.exists():
        ck.expect("td_inventory.json exists", False); return
    td = json.load(open(C.INVENTORY_TD_JSON))
    ck.expect("td_inventory.json exists", True)
    ck.expect(
        "TD total_rows >= 11M",
        td.get("total_rows", 0) >= 11_000_000,
        f"got {td.get('total_rows')}")
    ck.expect(
        "TD contains Panel_Request type",
        "Panel_Request" in td.get("type_counts", {}),
        f"types: {list(td.get('type_counts', {}).keys())[:5]}...")

    if not C.INVENTORY_MOVEMENTS_JSON.exists():
        ck.expect("movements_inventory.json exists", False); return
    mv = json.load(open(C.INVENTORY_MOVEMENTS_JSON))
    ck.expect("movements_inventory.json exists", True)
    ck.expect(
        "Movements total_rows >= 240k",
        mv.get("total_rows", 0) >= 240_000,
        f"got {mv.get('total_rows')}")


def check_decisions(ck: Checker):
    ck.section("Stage 2 — Decision events")
    if not C.DECISION_EVENTS_SUMMARY.exists():
        ck.expect("decision_events_summary.json exists", False); return

    s = json.load(open(C.DECISION_EVENTS_SUMMARY))
    ck.expect("decision_events_summary.json exists", True)
    ck.expect(
        "total_decision_events >= 540k",
        s.get("total_decision_events", 0) >= 540_000,
        f"got {s.get('total_decision_events')}")
    ck.expect(
        "all 5 prefixes present",
        set(s.get("by_prefix", {}).keys()) == set(C.DERBY_PREFIXES),
        f"got {sorted(s.get('by_prefix', {}).keys())}")
    ck.expect(
        "headcode_parse_rate >= 99%",
        s.get("headcode_parse_rate_pct", 0) >= 99.0,
        f"got {s.get('headcode_parse_rate_pct')}")
    ck.expect(
        "decision_events.parquet exists",
        C.DECISION_EVENTS_PARQUET.exists(),
        f"path: {C.DECISION_EVENTS_PARQUET}")


def check_infrastructure(ck: Checker):
    ck.section("Stage 3 — Infrastructure")
    expected = {
        C.ROUTES_CLEAN_PARQUET:      "routes_clean",
        C.TRACKS_INVENTORY_PARQUET:  "tracks_inventory",
        C.SIGNALS_INVENTORY_PARQUET: "signals_inventory",
        C.AUXILIARY_PARQUET:         "auxiliary_connections",
        C.INFRASTRUCTURE_GRAPH_JSON: "infrastructure_graph.json",
    }
    for path, name in expected.items():
        ck.expect(f"{name} exists", path.exists())


def check_static_graph(ck: Checker):
    ck.section("Stage 4 — Static heterogeneous graph")
    if not C.STATIC_GRAPH_SUMMARY_JSON.exists():
        ck.expect("static_graph_summary.json exists", False); return

    g = json.load(open(C.STATIC_GRAPH_SUMMARY_JSON))
    ck.expect("static_graph_summary.json exists", True)

    nodes = g.get("nodes", {})
    ck.expect("nodes.track == 249", nodes.get("track") == 249, f"got {nodes.get('track')}")
    ck.expect("nodes.signal == 123", nodes.get("signal") == 123, f"got {nodes.get('signal')}")
    ck.expect("nodes.route == 277",  nodes.get("route") == 277, f"got {nodes.get('route')}")
    ck.expect("nodes.trts == 24",    nodes.get("trts") == 24, f"got {nodes.get('trts')}")

    edges = g.get("edges", {})
    expected_edges = {
        "protects": 100, "connects": 548, "traverses": 1701,
        "starts_at": 279, "ends_at": 290, "same_signal": 1122,
    }
    for ename, expected_n in expected_edges.items():
        ck.expect(
            f"edges.{ename} == {expected_n}",
            edges.get(ename) == expected_n,
            f"got {edges.get(ename)}")

    phys = g.get("physical_features_coverage", {})
    ck.expect(
        "Derby_info physical coverage >= 275 routes",
        phys.get("routes_with_length", 0) >= 275,
        f"got {phys.get('routes_with_length')}")


def check_event_stream(ck: Checker):
    ck.section("Stage 5 — Event token stream")
    ck.expect(
        "event_tokens.parquet exists",
        C.EVENT_STREAM_PARQUET.exists(),
        f"path: {C.EVENT_STREAM_PARQUET}")
    if C.EVENT_STREAM_PARQUET.exists():
        sz_mb = C.EVENT_STREAM_PARQUET.stat().st_size / 1024 / 1024
        ck.expect(
            "event_tokens.parquet > 30 MB",
            sz_mb > 30,
            f"size: {sz_mb:.1f} MB")


def check_calibration(ck: Checker):
    ck.section("Stage 7 — Reward calibration")
    if not C.CALIBRATION_JSON.exists():
        ck.expect("calibration.json exists", False); return

    cal = json.load(open(C.CALIBRATION_JSON))
    ck.expect("calibration.json exists", True)

    h = cal.get("headway", {})
    ck.expect(
        "H_min_seconds == 147.0",
        h.get("H_min_seconds_used") == 147.0,
        f"got {h.get('H_min_seconds_used')}")

    a = cal.get("approach_distance", {}).get("d_gate_breakpoints", {})
    ck.expect(
        "d_gate_0.5_max == 6",
        a.get("gate_0.5_max") == 6,
        f"got {a.get('gate_0.5_max')}")
    ck.expect(
        "d_gate_0.1_max == 16",
        a.get("gate_0.1_max") == 16,
        f"got {a.get('gate_0.1_max')}")

    t = cal.get("tiploc_lag", {})
    win = t.get("window_seconds_used", 0)
    ck.expect(
        "window_seconds ≈ 4202",
        abs(win - 4202) < 1.0,
        f"got {win}")


def check_pr_outcomes(ck: Checker):
    ck.section("Stage 8 — PR outcomes")
    if not C.PR_OUTCOMES_SUMMARY.exists():
        ck.expect("pr_outcomes_summary.json exists", False); return

    s = json.load(open(C.PR_OUTCOMES_SUMMARY))
    ck.expect("pr_outcomes_summary.json exists", True)

    # Outcome distribution: expect >99% used
    by = s.get("outcome_counts", s.get("by_outcome", {}))
    total = sum(by.values()) if by else 1
    used_pct = (by.get("used", 0) / total * 100) if total else 0
    ck.expect(
        "outcome 'used' >= 99%",
        used_pct >= 99.0,
        f"got {used_pct:.2f}%")


def check_decision_rewards(ck: Checker):
    ck.section("Stage 10 — Final reward table")
    if not C.DECISION_REWARDS_SUMMARY.exists():
        ck.expect("decision_rewards_summary.json exists", False); return

    s = json.load(open(C.DECISION_REWARDS_SUMMARY))
    ck.expect("decision_rewards_summary.json exists", True)

    # mean r_total should be modestly positive
    desc = s.get("r_total_describe", {})
    mean_r = desc.get("mean", 0)
    ck.expect(
        "r_total mean > 0.1",
        mean_r > 0.1,
        f"got {mean_r}")
    ck.expect(
        "r_total mean < 0.5",
        mean_r < 0.5,
        f"got {mean_r}")

    weights = s.get("weights", {})
    ck.expect(
        "w_delay == 1.0",
        weights.get("w_delay") == 1.0,
        f"got {weights.get('w_delay')}")
    ck.expect(
        "w_headway == 1.0",
        weights.get("w_headway") == 1.0,
        f"got {weights.get('w_headway')}")
    ck.expect(
        "w_wait == 0.3",
        weights.get("w_wait") == 0.3,
        f"got {weights.get('w_wait')}")

    ck.expect(
        "decision_rewards.parquet exists",
        C.DECISION_REWARDS_PARQUET.exists())


def check_health(ck: Checker):
    ck.section("Stage 10b — Reward health check")
    hp = C.REWARDS_DIR / "health" / "health_summary.json"
    if not hp.exists():
        ck.warn("health_summary.json missing (optional but recommended)")
        return

    h = json.load(open(hp))
    ck.expect("health_summary.json exists", True)

    # Find weight stability spearman if present
    wsp = h.get("weight_spearman", {})
    for key in ("conservative_vs_default", "conservative_vs_default_spearman"):
        if key in wsp:
            ck.expect(
                "weight conservative_vs_default Spearman >= 0.85",
                wsp[key] >= 0.85,
                f"got {wsp[key]}")
            break

    # frac_positive episodes
    if "frac_positive_episodes" in h or "frac_positive" in h:
        fp = h.get("frac_positive_episodes", h.get("frac_positive"))
        ck.expect(
            "frac positive episodes >= 80%",
            fp >= 0.80,
            f"got {fp}")


def check_analyses(ck: Checker):
    ck.section("Optional — Empirical analyses")
    expected = [
        ("conflict_summary.json", C.ANALYSES_DIR / "conflict_summary.json"),
        ("route_class_summary.json", C.ANALYSES_DIR / "route_class_summary.json"),
        ("non_standard_trainids_summary.json", C.ANALYSES_DIR / "non_standard_trainids_summary.json"),
    ]
    for name, path in expected:
        if path.exists():
            ck.expect(f"{name} exists", True)
        else:
            ck.warn(f"{name} missing (optional)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    print(f"verifying v2 pipeline outputs against spec 01 §16")
    print(f"PROJECT_ROOT = {C.PROJECT_ROOT}")
    print(f"DATA_DIR     = {C.DATA_DIR}")
    print(f"OUTPUTS_DIR  = {C.OUTPUTS_DIR}")

    ck = Checker(verbose=args.verbose)

    # Hard pre-check: all required raw inputs present
    try:
        C.verify_paths_resolved()
        _ok("all required raw + reference inputs exist")
        ck.passed += 1
    except FileNotFoundError as e:
        _fail(f"missing inputs: {e}")
        ck.failed += 1

    # Stage-by-stage output checks
    check_inventory(ck)
    check_decisions(ck)
    check_infrastructure(ck)
    check_static_graph(ck)
    check_event_stream(ck)
    check_calibration(ck)
    check_pr_outcomes(ck)
    check_decision_rewards(ck)
    check_health(ck)
    check_analyses(ck)

    print()
    print("=" * 60)
    print(f"  RESULT: {ck.passed} passed, {ck.failed} failed, {ck.warnings} warnings")
    print("=" * 60)

    if ck.failed > 0:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
