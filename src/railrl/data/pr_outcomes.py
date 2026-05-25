"""P2.4 Iter B - PR outcome labelling via route lifecycle scan.

For each PR (focal_signal, chosen_route_id, time), determine the operational
outcome by tracking the route asset's state in event_token_stream:
  - used:               route's traversed TC was occupied during set-period
  - unused_cancelled:   route went 1->0 without any TC occupation, short-lived
  - unused_timeout:     route went 1->0 without any TC occupation, long-lived
  - unknown:            data ends before the route releases

Outputs (per PR row): pr_index, outcome, route_set_duration_seconds,
                      n_route_tc_occupations
"""
from __future__ import annotations
import json
import time as _time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .. import config as C
from .event_stream import AssetIndex, EventTokenStream


# ============================================================
# Lookup helpers
# ============================================================

def build_route_to_tc_idx(asset_index: AssetIndex,
                           edges_traverses_parquet: Path) -> dict[int, np.ndarray]:
    """For each route_id (as asset_idx), return its list of TC asset_idxs."""
    df = pd.read_parquet(edges_traverses_parquet)
    out: dict[int, list[int]] = {}
    for _, r in df.iterrows():
        rid = asset_index.idx(str(r["route_id"]))
        tid = asset_index.idx(str(r["track_id"]))
        if rid is None or tid is None:
            continue
        out.setdefault(rid, []).append(tid)
    return {k: np.array(sorted(set(v)), dtype=np.int16) for k, v in out.items()}


# ============================================================
# Per-PR outcome classification
# ============================================================

def classify_outcome(es: EventTokenStream,
                      route_asset_idx: int,
                      tc_asset_idxs: np.ndarray,
                      pr_time_ns: int,
                      cancelled_threshold_seconds: float = 60.0,
                      ev_by: dict = None) -> dict:
    """Vectorised: find route release, count TC occupations in [pr,release]."""
    if ev_by is None:
        ev_by = es._build_per_asset_index()

    route_pos = ev_by.get(route_asset_idx)
    if route_pos is None or route_pos.size == 0:
        return {"outcome": "unknown", "duration_seconds": float("nan"),
                "n_tc_occupations": 0}
    times_full = es.time_ns
    states_full = es.state

    route_times = times_full[route_pos]
    j = int(np.searchsorted(route_times, pr_time_ns, side="left"))
    if j >= route_pos.size:
        return {"outcome": "unknown", "duration_seconds": float("nan"),
                "n_tc_occupations": 0}
    after = route_pos[j:]
    states_after = states_full[after]
    zero_locs = np.flatnonzero(states_after == 0)
    if zero_locs.size == 0:
        return {"outcome": "unknown", "duration_seconds": float("nan"),
                "n_tc_occupations": 0}
    release_pos = after[zero_locs[0]]
    release_t   = int(times_full[release_pos])
    duration_s  = (release_t - pr_time_ns) / 1e9

    n_occ = 0
    for tc_idx in tc_asset_idxs:
        tc_pos = ev_by.get(int(tc_idx))
        if tc_pos is None or tc_pos.size == 0:
            continue
        tc_times = times_full[tc_pos]
        lo = int(np.searchsorted(tc_times, pr_time_ns, side="left"))
        hi = int(np.searchsorted(tc_times, release_t,  side="right"))
        if lo >= hi:
            continue
        seg_states = states_full[tc_pos[lo:hi]]
        n_occ += int(np.count_nonzero(seg_states == 1))

    if n_occ > 0:
        outcome = "used"
    elif duration_s < cancelled_threshold_seconds:
        outcome = "unused_cancelled"
    else:
        outcome = "unused_timeout"

    return {"outcome": outcome,
            "duration_seconds": float(duration_s),
            "n_tc_occupations": n_occ}


# ============================================================
# Batch driver
# ============================================================

OUTCOME_REWARD = {
    "used":             1.0,
    "unused_cancelled": -1.0,
    "unused_timeout":   -0.5,
    "unknown":          0.0,
}


def label_all_prs(decision_points_parquet: Path,
                   asset_index: AssetIndex,
                   event_stream: EventTokenStream,
                   route_to_tcs: dict[int, np.ndarray],
                   *,
                   limit: int = None,
                   progress_every: int = 50_000) -> pd.DataFrame:
    dp = pd.read_parquet(decision_points_parquet)
    pr = dp[dp["label"] == "set"].copy().reset_index(drop=True)
    if limit is not None:
        pr = pr.head(limit).copy()
    pr["chosen_route_id"] = pr["chosen_route_id"].astype(str)

    # Force PR time to nanoseconds
    pr["t_ns"] = pd.to_datetime(pr["time"]).astype("datetime64[ns]").astype("int64")

    n = len(pr)
    print(f"  classifying {n:,} PR decisions...")
    ev_by = event_stream._build_per_asset_index()
    t0 = _time.time()

    # Map route_id (string) -> asset_idx ONCE
    pr["route_idx"] = pr["chosen_route_id"].map(lambda s: asset_index.idx(str(s)))

    outcomes  = ["unknown"] * n
    durations = [float("nan")] * n
    n_occs    = [0] * n

    times_full = event_stream.time_ns
    states_full = event_stream.state

    n_done = 0
    for rid, group in pr.groupby("route_idx", sort=False):
        if rid is None or pd.isna(rid):
            n_done += len(group); continue
        rid = int(rid)
        tcs = route_to_tcs.get(rid)
        if tcs is None:
            n_done += len(group); continue
        route_pos = ev_by.get(rid)
        if route_pos is None or route_pos.size == 0:
            n_done += len(group); continue
        route_times  = times_full[route_pos]
        route_states = states_full[route_pos]

        # Pre-fetch per-TC times/states once for this route
        tc_data = []
        for tc_idx in tcs:
            tc_pos = ev_by.get(int(tc_idx))
            if tc_pos is None or tc_pos.size == 0:
                continue
            tc_data.append((times_full[tc_pos], states_full[tc_pos]))

        # Process all PRs of this route
        pr_idxs   = group.index.to_numpy()
        pr_times  = group["t_ns"].to_numpy(dtype=np.int64)

        for i_pr, t_ns in zip(pr_idxs, pr_times):
            j = int(np.searchsorted(route_times, t_ns, side="left"))
            if j >= route_pos.size:
                continue   # leave as unknown
            zero_locs = np.flatnonzero(route_states[j:] == 0)
            if zero_locs.size == 0:
                continue
            release_t = int(route_times[j + zero_locs[0]])
            duration_s = (release_t - int(t_ns)) / 1e9

            n_occ = 0
            for tc_times, tc_states in tc_data:
                lo = int(np.searchsorted(tc_times, t_ns, side="left"))
                hi = int(np.searchsorted(tc_times, release_t, side="right"))
                if lo >= hi:
                    continue
                seg = tc_states[lo:hi]
                n_occ += int(np.count_nonzero(seg == 1))

            if n_occ > 0:
                outcomes[i_pr] = "used"
            elif duration_s < 60.0:
                outcomes[i_pr] = "unused_cancelled"
            else:
                outcomes[i_pr] = "unused_timeout"
            durations[i_pr] = float(duration_s)
            n_occs[i_pr] = n_occ

        n_done += len(group)
        if n_done >= progress_every and (n_done % progress_every) < len(group):
            elapsed = _time.time() - t0
            rate = n_done / elapsed if elapsed > 0 else 0
            eta = (n - n_done) / rate / 60 if rate > 0 else 0
            print(f"  {n_done:,}/{n:,} ({100*n_done/n:.1f}%)  "
                  f"{rate:.0f} rows/s  ETA {eta:.1f} min")
    n_route_unknown = int(pr["route_idx"].isna().sum())

    pr["outcome"] = outcomes
    pr["route_set_duration_seconds"] = durations
    pr["n_route_tc_occupations"] = n_occs
    pr["r_thru"] = pr["outcome"].map(OUTCOME_REWARD)
    if n_route_unknown > 0:
        print(f"  WARNING: {n_route_unknown} PRs had no asset_idx for chosen_route_id")
    out_cols = [
        "time", "focal_signal", "focal_train", "chosen_route_id",
        "outcome", "route_set_duration_seconds",
        "n_route_tc_occupations", "r_thru",
    ]
    # Carry sample_id through when the source table has it (v2 rewardfmt).
    # The 4-tuple (time, focal_signal, focal_train, chosen_route_id) is NOT
    # unique — duplicate PRs blow up a downstream merge — so sample_id is the
    # safe 1:1 join key. v1 decision_points lack the column → no-op there.
    if "sample_id" in pr.columns:
        out_cols = ["sample_id"] + out_cols
    return pr[out_cols]


def summarize_outcomes(pr_outcomes: pd.DataFrame) -> dict:
    counts = pr_outcomes["outcome"].value_counts(dropna=False).to_dict()
    total = int(len(pr_outcomes))
    pct = {k: round(100 * v / total, 2) for k, v in counts.items()}
    by_outcome_duration = (pr_outcomes.groupby("outcome")["route_set_duration_seconds"]
                            .describe().to_dict(orient="index"))
    return {
        "total_prs": total,
        "outcome_counts":  {k: int(v) for k, v in counts.items()},
        "outcome_percent": pct,
        "duration_seconds_by_outcome": by_outcome_duration,
        "r_thru_distribution": {
            "mean": float(pr_outcomes["r_thru"].mean()),
            "min":  float(pr_outcomes["r_thru"].min()),
            "max":  float(pr_outcomes["r_thru"].max()),
        },
    }
