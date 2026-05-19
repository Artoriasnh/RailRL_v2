"""spec 02 §5 — Episode segmentation.

An episode = the sequence of decision points belonging to one operational
pass of one train through Derby. Episode boundary comes from pass_id
(materialised in outputs/passes/pass_assignments.parquet per spec 01 §17.2).

Each decision sample gets 4 episode-level fields appended:
    pass_id              — episode identifier (TRUST train_id or fallback)
    episode_idx          — global episode index (0..n_episodes-1)
    position_in_episode  — 0-indexed position within the episode
    is_last_in_episode   — terminal flag (for RL discount truncation)

Returns are computed with γ = 0.95 (spec 02 §5.4).
"""
from __future__ import annotations
from typing import Optional

import numpy as np
import pandas as pd

from .. import config as C


# ============================================================
# Pass_id → episode_idx assignment
# ============================================================

def build_episodes(
    decision_points: pd.DataFrame,
    pass_assignments: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Add episode metadata columns to a decision_points DataFrame.

    Args:
        decision_points: from generate_decision_points(); must contain
            columns ['focal_train', 't'] (and optionally 'pass_id').
        pass_assignments: optional pass_assignments.parquet. If provided,
            joins on (focal_train, t) to find pass_id. If None, uses a
            fallback: group consecutive (focal_train, t) by 30-min gap
            (spec 02 §8.2 fallback rule).

    Returns:
        New DataFrame (input + 4 new columns):
            pass_id, episode_idx, position_in_episode, is_last_in_episode
    """
    df = decision_points.copy()
    df["t"] = pd.to_datetime(df["t"])
    df = df.sort_values(["focal_train", "t"]).reset_index(drop=True)

    if pass_assignments is not None and "pass_id" in pass_assignments.columns:
        df = _join_pass_assignments(df, pass_assignments)
    else:
        df = _assign_pass_by_gap(df)

    # Now assign episode_idx globally + position_in_episode within each pass
    df = df.sort_values(["pass_id", "t"]).reset_index(drop=True)

    # Global unique episode index per pass_id
    unique_passes = df["pass_id"].unique()
    pass_to_idx = {p: i for i, p in enumerate(unique_passes)}
    df["episode_idx"] = df["pass_id"].map(pass_to_idx).astype(np.int32)

    # Position within episode
    df["position_in_episode"] = df.groupby("pass_id").cumcount().astype(np.int32)

    # is_last_in_episode
    last_in_group = df.groupby("pass_id")["t"].transform("max")
    df["is_last_in_episode"] = (df["t"] == last_in_group)

    return df


def _join_pass_assignments(df: pd.DataFrame, pa: pd.DataFrame) -> pd.DataFrame:
    """Inner-join decision_points with pass_assignments on (trainid_filled, time).

    pass_assignments rows: time_ns, trainid_filled, pass_id, ...

    Strategy: for each (focal_train, t) in df, find the row in pa with the
    same trainid_filled whose [pass_t_first_ns, pass_t_last_ns] contains t.
    """
    pa = pa.copy()
    if "trainid_filled" not in pa.columns and "focal_train" in pa.columns:
        pa = pa.rename(columns={"focal_train": "trainid_filled"})

    # Build per-train pass intervals
    interval_by_train: dict[str, list[tuple[int, int, str]]] = {}
    for tid, sub in pa.groupby("trainid_filled"):
        intervals = []
        for _, r in sub.iterrows():
            t0 = int(r["pass_t_first_ns"])
            t1 = int(r["pass_t_last_ns"])
            intervals.append((t0, t1, str(r["pass_id"])))
        intervals.sort()
        interval_by_train[str(tid)] = intervals

    # Resolve pass_id for each row
    df["t_ns"] = df["t"].astype("int64")
    pass_ids = []
    for _, row in df.iterrows():
        tid = str(row["focal_train"])
        t_ns = int(row["t_ns"])
        intervals = interval_by_train.get(tid, [])
        matched = None
        for t0, t1, pid in intervals:
            if t0 <= t_ns <= t1:
                matched = pid
                break
        if matched is None:
            # Fallback: form fallback id
            matched = f"FB:{tid}:0"
        pass_ids.append(matched)

    df["pass_id"] = pass_ids
    df = df.drop(columns=["t_ns"])
    return df


def _assign_pass_by_gap(
    df: pd.DataFrame,
    gap_seconds: float = 1800.0,
) -> pd.DataFrame:
    """Fallback when pass_assignments.parquet is not yet materialized.

    Group decisions by (focal_train) and split on gap > gap_seconds.
    Each segment gets a synthetic pass_id "FB:{train}:{segment_idx}".

    gap_seconds = 1800s (30 min) by default — train doesn't typically
    dwell that long in Derby between PRs of the same pass.
    """
    df = df.sort_values(["focal_train", "t"]).reset_index(drop=True)
    df["dt_seconds"] = (
        df.groupby("focal_train")["t"]
        .diff()
        .dt.total_seconds()
        .fillna(0.0)
    )
    df["_segment_break"] = (df["dt_seconds"] > gap_seconds).astype(int)
    df["_segment_idx"] = df.groupby("focal_train")["_segment_break"].cumsum()
    df["pass_id"] = (
        "FB:" + df["focal_train"].astype(str) + ":" + df["_segment_idx"].astype(str)
    )
    return df.drop(columns=["dt_seconds", "_segment_break", "_segment_idx"])


# ============================================================
# Episode returns
# ============================================================

def episode_returns(
    episodes: pd.DataFrame,
    gamma: float = None,
    reward_col: str = "r_total",
) -> pd.Series:
    """Compute Σ_t γ^t · r_t per episode.

    Used by spec 05 §11 (IRL Stage 2) + spec 04 §4.2 for analysis.

    Args:
        episodes: DataFrame with columns ['pass_id', 'position_in_episode', reward_col]
        gamma: discount factor; default spec 02 §5.4 (0.95)
        reward_col: name of the reward column (default 'r_total')

    Returns:
        pd.Series indexed by pass_id, values = discounted returns
    """
    if gamma is None:
        gamma = C.DISCOUNT_GAMMA

    if reward_col not in episodes.columns:
        # If reward not yet computed (decision_points without rewards),
        # return zeros — caller should join with decision_rewards first.
        return pd.Series(dtype=float)

    df = episodes.sort_values(["pass_id", "position_in_episode"])
    df["_discount"] = gamma ** df["position_in_episode"].astype(float)
    df["_discounted_r"] = df[reward_col] * df["_discount"]
    returns = df.groupby("pass_id")["_discounted_r"].sum()
    return returns


# ============================================================
# Summary stats
# ============================================================

def summarize_episodes(episodes: pd.DataFrame) -> dict:
    """Statistics on episode structure (for sanity check)."""
    n_episodes = episodes["pass_id"].nunique()
    per_ep = episodes.groupby("pass_id").size()
    return {
        "n_decisions":        int(len(episodes)),
        "n_episodes":         int(n_episodes),
        "decisions_per_episode": {
            "mean":   float(per_ep.mean()),
            "median": float(per_ep.median()),
            "p25":    float(per_ep.quantile(0.25)),
            "p75":    float(per_ep.quantile(0.75)),
            "p99":    float(per_ep.quantile(0.99)),
            "max":    int(per_ep.max()),
        },
        "n_fallback_passes":  int(
            episodes["pass_id"].astype(str).str.startswith("FB:").sum()
        ),
    }
