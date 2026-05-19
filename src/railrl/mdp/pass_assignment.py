"""spec 01 §17.2 — Materialize pass_assignments.parquet.

A "pass" = one operational journey of one train through Derby.

Algorithm (per spec 01 §8.2):

  1. For each TRUST train_id in Movements, find its time range
     [t_first, t_last] and headcode = chars[2:6].
  2. For each TD event with trainid_filled (4-char headcode), find the
     TRUST train_id whose:
       - chars[2:6] matches trainid_filled
       - time range covers the TD event time (± PASS_LOOKUP_BUFFER_S)
     If multiple match, pick the one whose center is closest to event time.
  3. If no TRUST id matches, fall back to time-gap clustering:
       - Group consecutive (trainid_filled, time) by gap > PASS_FALLBACK_GAP_S
       - Synthetic pass_id = "FB:{trainid}:{cluster_idx}"

Output: outputs/passes/pass_assignments.parquet

Columns:
    time_ns          int64    TD event time
    trainid_filled   string   TD's trainid_filled (4-char headcode)
    pass_id          string   TRUST train_id (matched) or "FB:{tid}:{idx}"
    pass_source      string   'trust_match' or 'fallback_gap'
    pass_t_first_ns  int64    start of this pass's time range
    pass_t_last_ns   int64    end of this pass's time range

Used by:
    - spec 02 §5 episode segmentation (mdp/episode.py::_join_pass_assignments)
    - spec 02 §3.2 prev_routes filter (mdp/action.py::build_pass_route_history)
    - spec 01 §11.3 reward delay attribution (data/reward_features.py)
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional
import time as _time

import numpy as np
import pandas as pd

from .. import config as C


def build_trust_index(
    movements_csv: Path,
) -> tuple[dict, dict]:
    """Read Movements.csv → build TRUST train_id index.

    Returns:
        by_trust:      dict[train_id_10char → (t_first_ns, t_last_ns, headcode)]
        by_headcode:   dict[headcode_4char → list of (t_first_ns, t_last_ns, train_id)]
                       sorted by t_first_ns for fast lookup
    """
    print(f"  [pass] loading Movements: {movements_csv}")
    mv = pd.read_csv(
        movements_csv,
        usecols=["train_id", "actual_timestamp"],
    )
    mv["headcode"] = mv["train_id"].astype(str).str[2:6]
    mv["actual"] = pd.to_datetime(mv["actual_timestamp"], errors="coerce")
    mv = mv.dropna(subset=["actual", "headcode"])
    mv = mv[mv["headcode"].str.len() == 4]
    mv["actual_ns"] = mv["actual"].astype("int64")

    by_trust = {}
    by_headcode: dict[str, list] = {}
    for tid, sub in mv.groupby("train_id"):
        sub = sub.sort_values("actual_ns")
        t_first = int(sub["actual_ns"].iloc[0])
        t_last  = int(sub["actual_ns"].iloc[-1])
        hc = str(sub["headcode"].iloc[0])
        by_trust[tid] = (t_first, t_last, hc)
        by_headcode.setdefault(hc, []).append((t_first, t_last, tid))

    for hc in by_headcode:
        by_headcode[hc].sort()
    print(f"  [pass]   {len(by_trust):,} TRUST train_ids, {len(by_headcode):,} headcodes")
    return by_trust, by_headcode


def match_td_to_trust(
    td_events: pd.DataFrame,
    by_headcode: dict[str, list],
    buffer_s: float = None,
) -> pd.DataFrame:
    """Match each TD event to the best TRUST train_id (or None).

    Args:
        td_events: must have columns ['time', 'trainid_filled'] (only the
                   ones with non-null trainid_filled are matched)
        by_headcode: from build_trust_index()
        buffer_s: ± buffer around TRUST time range; default PASS_LOOKUP_BUFFER_S

    Returns:
        DataFrame with ['time_ns', 'trainid_filled', 'matched_trust_id'].
        matched_trust_id is None if no match within buffer.
    """
    if buffer_s is None:
        buffer_s = C.PASS_LOOKUP_BUFFER_S
    buffer_ns = int(buffer_s * 1e9)

    df = td_events[["time", "trainid_filled"]].copy()
    df["time_ns"] = pd.to_datetime(df["time"]).astype("int64")
    df["trainid_filled"] = df["trainid_filled"].astype(str)

    matched_ids = []
    n_match = n_no_candidate = n_no_match = 0
    for _, row in df.iterrows():
        hc = row["trainid_filled"]
        t_ns = int(row["time_ns"])
        candidates = by_headcode.get(hc, [])
        if not candidates:
            matched_ids.append(None)
            n_no_candidate += 1
            continue
        best_tid = None
        best_dist = None
        for (t_first, t_last, tid) in candidates:
            lo = t_first - buffer_ns
            hi = t_last + buffer_ns
            if lo <= t_ns <= hi:
                center = (t_first + t_last) // 2
                dist = abs(t_ns - center)
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_tid = tid
        if best_tid is None:
            matched_ids.append(None)
            n_no_match += 1
        else:
            matched_ids.append(best_tid)
            n_match += 1

    df["matched_trust_id"] = matched_ids
    print(f"  [pass]   matched: {n_match:,}, no_match: {n_no_match:,}, "
          f"no_candidate: {n_no_candidate:,}")
    return df


def assign_fallback_passes(
    unmatched: pd.DataFrame,
    gap_seconds: float = None,
) -> pd.DataFrame:
    """For TD events with no TRUST match, group by gap to form fallback passes.

    Args:
        unmatched: rows from match_td_to_trust where matched_trust_id is None.
                   Must have ['time_ns', 'trainid_filled'].
        gap_seconds: cluster split threshold; default PASS_FALLBACK_GAP_S (6h)

    Returns:
        Same DataFrame with 'pass_id' column populated as "FB:{tid}:{idx}".
    """
    if gap_seconds is None:
        gap_seconds = C.PASS_FALLBACK_GAP_S
    gap_ns = int(gap_seconds * 1e9)

    df = unmatched.copy().sort_values(["trainid_filled", "time_ns"]).reset_index(drop=True)
    df["gap_ns"] = df.groupby("trainid_filled")["time_ns"].diff().fillna(0).astype("int64")
    df["new_cluster"] = (df["gap_ns"] > gap_ns).astype(int)
    df["cluster_idx"] = df.groupby("trainid_filled")["new_cluster"].cumsum()
    df["pass_id"] = (
        "FB:" + df["trainid_filled"].astype(str) + ":" + df["cluster_idx"].astype(str)
    )
    return df.drop(columns=["gap_ns", "new_cluster", "cluster_idx"])


def build_pass_assignments(
    td_events: pd.DataFrame,
    movements_csv: Path,
    buffer_s: float = None,
    gap_seconds: float = None,
) -> pd.DataFrame:
    """Top-level: assign every TD event with a trainid_filled to a pass_id.

    Returns DataFrame with columns:
        time_ns, trainid_filled, pass_id, pass_source,
        pass_t_first_ns, pass_t_last_ns
    """
    t0 = _time.time()
    by_trust, by_headcode = build_trust_index(movements_csv)

    # Only match TD events with valid trainid_filled (re-use trigger filter)
    mask = (
        td_events["trainid_filled"].notna()
        & td_events["trainid_filled"].astype(str).str.match(r"^[0-9A-Z]{3,4}$", na=False)
        & ~td_events["trainid_filled"].astype(str).isin(
            {"0", "00", "000", "0000", "NULL", "NONE", "NAN", ""})
    )
    eligible = td_events.loc[mask, ["time", "trainid_filled"]]
    print(f"  [pass] eligible TD events with valid trainid: {len(eligible):,}")

    matched_df = match_td_to_trust(eligible, by_headcode, buffer_s=buffer_s)

    # Split by match status
    has_match = matched_df["matched_trust_id"].notna()
    matched = matched_df.loc[has_match].copy()
    unmatched = matched_df.loc[~has_match].copy()

    # Build matched rows
    matched_rows = []
    for _, r in matched.iterrows():
        tid = r["matched_trust_id"]
        t_first, t_last, _ = by_trust[tid]
        matched_rows.append({
            "time_ns":         int(r["time_ns"]),
            "trainid_filled":  r["trainid_filled"],
            "pass_id":         tid,
            "pass_source":     "trust_match",
            "pass_t_first_ns": t_first,
            "pass_t_last_ns":  t_last,
        })
    matched_out = pd.DataFrame(matched_rows)

    # Build fallback rows
    fb_in = unmatched[["time_ns", "trainid_filled"]].copy()
    fb_assigned = assign_fallback_passes(fb_in, gap_seconds=gap_seconds)
    # Compute pass time ranges per fallback pass_id
    fb_ranges = fb_assigned.groupby("pass_id")["time_ns"].agg(["min", "max"]).reset_index()
    fb_ranges = fb_ranges.rename(columns={"min": "pass_t_first_ns", "max": "pass_t_last_ns"})
    fb_assigned = fb_assigned.merge(fb_ranges, on="pass_id", how="left")
    fb_assigned["pass_source"] = "fallback_gap"
    fb_out = fb_assigned[[
        "time_ns", "trainid_filled", "pass_id", "pass_source",
        "pass_t_first_ns", "pass_t_last_ns",
    ]]

    combined = pd.concat([matched_out, fb_out], ignore_index=True)
    combined = combined.sort_values("time_ns").reset_index(drop=True)
    elapsed = _time.time() - t0
    print(f"  [pass] DONE in {elapsed:.1f}s: "
          f"{len(matched_out):,} TRUST + {len(fb_out):,} fallback = {len(combined):,} total")
    return combined


def summarize_pass_assignments(pa: pd.DataFrame) -> dict:
    """Sanity stats for pass_assignments.parquet."""
    return {
        "n_total":            int(len(pa)),
        "n_trust_match":      int((pa["pass_source"] == "trust_match").sum()),
        "n_fallback":         int((pa["pass_source"] == "fallback_gap").sum()),
        "trust_match_pct":    round(
            100 * (pa["pass_source"] == "trust_match").mean(), 2
        ),
        "n_unique_pass_ids":  int(pa["pass_id"].nunique()),
        "n_unique_trains":    int(pa["trainid_filled"].nunique()),
    }
