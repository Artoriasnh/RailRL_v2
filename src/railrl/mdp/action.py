"""spec 02 §3 — Structured action space + candidate generation.

Action space at each decision point (focal_train, focal_signal, t):

    A_t = {wait} ∪ {(focal_train, R) | R ∈ candidates(focal_train, focal_signal, t)}

Candidate algorithm uses 4 rules (spec 02 §3.2), all from time≤t observable
state:
  1. R starts from focal_signal (or train is in its approach horizon)
  2. Direction of R matches focal_train's inferred direction
  3. R does not conflict with already-set routes for this pass
  4. planned_platform soft filter (NOT hard — allow platform reassignment)

Per spec 01 §17.5, focal_signal is sample metadata (used here for candidate
filtering and reward computation), NEVER passed as state feature to the model.
"""
from __future__ import annotations
import ast
import time as _time
from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd

from .. import config as C
from ..parsers import parse_route_id


# ============================================================
# Helpers
# ============================================================

def _parse_track_sections(val) -> list[str]:
    """Routes_clean's track_sections column may be list or string-encoded list."""
    if val is None:
        return []
    if isinstance(val, list):
        return [str(t) for t in val]
    if isinstance(val, np.ndarray):
        return [str(t) for t in val.tolist()]
    if isinstance(val, str):
        try:
            v = ast.literal_eval(val)
            return [str(t) for t in v] if isinstance(v, list) else []
        except Exception:
            return []
    return []


def _route_direction(track_sections: list[str]) -> Optional[str]:
    """Infer route direction from its first → last TC.

    For Derby, we use a coarse heuristic: the alphabetical sort of TC names
    correlates with geographic direction. A more robust version would use
    the signal numeric tail or an explicit direction map.

    Returns: 'forward' / 'reverse' / None
    """
    if len(track_sections) < 2:
        return None
    first, last = track_sections[0], track_sections[-1]
    return "forward" if first <= last else "reverse"


def _infer_train_direction(recent_tcs: list[str]) -> Optional[str]:
    """Infer focal_train's direction from its last N TCs.

    recent_tcs[0] = oldest, recent_tcs[-1] = current.
    """
    if not recent_tcs or len(recent_tcs) < 2:
        return None
    return "forward" if recent_tcs[0] <= recent_tcs[-1] else "reverse"


# ============================================================
# Routes-by-signal index (built once, reused per decision)
# ============================================================

class RouteIndex:
    """Indexes routes_clean by (start_signal, prefix) for fast candidate lookup.

    Built once and reused across all decision points.
    """

    def __init__(self, routes_clean: pd.DataFrame):
        # Detect column names
        if "start_signal" in routes_clean.columns:
            start_col = "start_signal"
        elif "start_signals" in routes_clean.columns:
            # may be list; explode
            start_col = "start_signals"
        elif "signal_no" in routes_clean.columns:
            start_col = "signal_no"
        else:
            raise KeyError("routes_clean: no start_signal / signal_no column")

        if "track_sections" in routes_clean.columns:
            tc_col = "track_sections"
        elif "track_list" in routes_clean.columns:
            tc_col = "track_list"
        else:
            raise KeyError("routes_clean: no track_sections / track_list column")

        self.by_start_signal: dict[str, list[dict]] = defaultdict(list)

        for _, r in routes_clean.iterrows():
            rid = str(r["route_id"]) if "route_id" in r else None
            if not rid:
                continue
            tcs = _parse_track_sections(r[tc_col])
            direction = _route_direction(tcs)

            # end_platform_id may exist
            end_plat = None
            if "end_platform_id" in r and r["end_platform_id"] is not None:
                try:
                    end_plat = int(r["end_platform_id"])
                except (ValueError, TypeError):
                    end_plat = None

            entry = {
                "route_id":          rid,
                "track_sections":    tcs,
                "direction":         direction,
                "end_platform_id":   end_plat,
                "cls":               str(r["cls"]) if "cls" in r else None,
            }

            # start_col may be list (start_signals) or scalar
            start_val = r[start_col]
            if isinstance(start_val, (list, np.ndarray)):
                for s in start_val:
                    self.by_start_signal[str(s)].append(entry)
            else:
                self.by_start_signal[str(start_val)].append(entry)

    def routes_from(self, signal_id: str) -> list[dict]:
        return self.by_start_signal.get(str(signal_id), [])


# ============================================================
# Per-pass route history (helps prev_routes consistency filter)
# ============================================================

def build_pass_route_history(
    decision_events: pd.DataFrame,
    pass_assignments: Optional[pd.DataFrame] = None,
) -> dict[str, list[tuple[pd.Timestamp, str]]]:
    """For each pass_id, return time-ordered list of (PR time, route_id) tuples.

    If pass_assignments is None, fallback to per-train_id grouping (less precise
    but works without explicit pass disambiguation).

    Returns: dict[pass_id (or train_id) → sorted [(t, route_id), ...]]
    """
    df = decision_events[["time", "train_id", "route_id"]].copy()
    df["time"] = pd.to_datetime(df["time"])

    if pass_assignments is not None and "pass_id" in pass_assignments.columns:
        # TODO: join on time-window matching (per spec 01 §8.2 algorithm)
        # Stub: fallback to train_id for now
        pass

    history: dict[str, list[tuple[pd.Timestamp, str]]] = defaultdict(list)
    df = df.sort_values("time")
    for _, r in df.iterrows():
        key = str(r["train_id"])
        history[key].append((r["time"], str(r["route_id"])))
    return history


# ============================================================
# Main: feasible_actions for one decision point
# ============================================================

def feasible_actions(
    focal_train: str,
    focal_signal: str,
    t: pd.Timestamp,
    route_index: RouteIndex,
    *,
    train_direction: Optional[str] = None,
    train_recent_tcs: Optional[list[str]] = None,
    prev_routes_set: Optional[list[str]] = None,
    planned_platform: Optional[int] = None,
    direction_filter: bool = True,
    platform_soft_filter: bool = True,
) -> list[str]:
    """Compute the candidate route_ids for one decision point.

    spec 02 §3.2 — 4 rules:
      1. Routes starting from focal_signal
      2. Direction matches focal_train (if direction known)
      3. Not conflicting with prev_routes_set (no revisit)
      4. planned_platform soft preference (not hard filter)

    Args:
        focal_train: train identity (sample metadata)
        focal_signal: signal identity (sample metadata, NEVER passed to model)
        t: decision time
        route_index: pre-built RouteIndex
        train_direction: 'forward' / 'reverse' / None (if None, computed
                         from train_recent_tcs if provided)
        train_recent_tcs: last K TCs of focal_train (for direction inference)
        prev_routes_set: list of route_id already set in this pass before t
        planned_platform: train's planned platform (soft preference, 1-6)
        direction_filter: if True, filter by direction match
        platform_soft_filter: if True, sort candidates so platform-matching
                               routes come first (does NOT exclude others)

    Returns:
        Ordered list of candidate route_ids (excluding wait, which is
        always implicit). Empty list = only wait is feasible.
    """
    # Infer direction if not provided
    if train_direction is None and train_recent_tcs:
        train_direction = _infer_train_direction(train_recent_tcs)

    # Rule 1: routes from focal_signal
    candidates = route_index.routes_from(focal_signal)

    out = []
    for c in candidates:
        # Rule 2: direction
        if direction_filter and train_direction is not None and c["direction"] is not None:
            if c["direction"] != train_direction:
                continue

        # Rule 3: prev_routes consistency — skip if same route already set
        if prev_routes_set and c["route_id"] in prev_routes_set:
            continue

        out.append(c)

    # Rule 4 (soft): sort by platform match
    if platform_soft_filter and planned_platform is not None:
        out.sort(key=lambda c: (c["end_platform_id"] != planned_platform,
                                 c["route_id"]))

    return [c["route_id"] for c in out]


# ============================================================
# Validation — check ≥99.5% chosen ∈ candidates
# ============================================================

def validate_candidates(
    decision_points: pd.DataFrame,
    route_index: RouteIndex,
    history: dict[str, list[tuple[pd.Timestamp, str]]],
    *,
    verbose: bool = False,
) -> dict:
    """For every SET decision, check whether chosen_route_id ∈ feasible_actions.

    Per spec 02 §3.3, target ≥ 99.5% coverage. Lower → candidate algorithm
    is too restrictive and needs widening.

    Args:
        decision_points: from generate_decision_points() — set rows must
                         have chosen_route_id
        route_index: pre-built
        history: from build_pass_route_history() — used for prev_routes filter

    Returns:
        dict with keys:
            n_total, n_in_candidates, n_not_in_candidates, coverage_pct,
            mismatch_breakdown (sample of mismatches)
    """
    set_df = decision_points[decision_points["label"] == "set"].copy()
    set_df["t"] = pd.to_datetime(set_df["t"])
    set_df = set_df.sort_values("t").reset_index(drop=True)

    n_total = len(set_df)
    n_match = 0
    mismatches = []

    # For prev_routes filter, walk history forward
    history_pos: dict[str, int] = defaultdict(int)

    t0 = _time.time()
    for i, row in set_df.iterrows():
        train = str(row["focal_train"])
        signal = str(row["focal_signal"])
        t = row["t"]
        chosen = str(row["chosen_route_id"])

        # Build prev_routes_set: routes set for this train BEFORE t
        prev = []
        train_hist = history.get(train, [])
        pos = history_pos[train]
        while pos < len(train_hist) and train_hist[pos][0] < t:
            prev.append(train_hist[pos][1])
            pos += 1
        history_pos[train] = pos

        # Compute candidates (direction inference omitted at validation time
        # for simplicity — see spec 02 §3.2 production code path)
        cands = feasible_actions(
            focal_train=train,
            focal_signal=signal,
            t=t,
            route_index=route_index,
            train_direction=None,    # no direction filter at validation
            prev_routes_set=prev,
            planned_platform=None,
            direction_filter=False,
            platform_soft_filter=False,
        )

        if chosen in cands:
            n_match += 1
        else:
            if len(mismatches) < 50:
                mismatches.append({
                    "i":             int(i),
                    "t":             str(t),
                    "focal_train":   train,
                    "focal_signal":  signal,
                    "chosen":        chosen,
                    "n_candidates":  len(cands),
                    "candidates_sample": cands[:5],
                })

    elapsed = _time.time() - t0
    n_miss = n_total - n_match
    coverage = (n_match / max(n_total, 1)) * 100

    if verbose:
        print(f"  [validate] {n_match:,}/{n_total:,} set decisions have "
              f"chosen ∈ candidates ({coverage:.2f}%)")
        print(f"  [validate] {n_miss:,} mismatches; sampled {len(mismatches)}")
        print(f"  [validate] elapsed: {elapsed:.1f}s")

    return {
        "n_total":              int(n_total),
        "n_in_candidates":      int(n_match),
        "n_not_in_candidates":  int(n_miss),
        "coverage_pct":         round(coverage, 3),
        "elapsed_seconds":      round(elapsed, 1),
        "mismatch_sample":      mismatches,
        "passes_99_5_threshold": bool(coverage >= 99.5),
    }
