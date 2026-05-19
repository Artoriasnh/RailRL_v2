"""P2.4 - Episode partitioning: per-train decision trajectories.

An episode = one train's sequence of decisions while passing through Derby.
Boundary: same focal_train but time gap > gap_minutes since previous decision
(default 30 min). This handles the case where the same headcode is reused
across operational days.
"""
from __future__ import annotations
import pandas as pd


def assign_episodes(df, *, gap_minutes=30,
                     time_col="time", train_col="focal_train"):
    """Add `episode_id` column. Sorts (train, time)."""
    df = df.sort_values([train_col, time_col]).reset_index(drop=True)
    prev_t = df.groupby(train_col)[time_col].shift()
    gap_s  = (df[time_col] - prev_t).dt.total_seconds()
    new_ep = (gap_s.isna() | (gap_s > gap_minutes * 60))
    seq    = new_ep.groupby(df[train_col]).cumsum().astype(int)
    df["episode_id"] = df[train_col].astype(str) + "-" + seq.astype(str)
    return df


def episode_summary(df):
    """Per-episode stats."""
    grp = df.groupby("episode_id", sort=False)
    out = pd.DataFrame({
        "n_decisions":  grp.size(),
        "n_set":        grp["label"].apply(lambda s: int((s == "set").sum())),
        "n_wait":       grp["label"].apply(lambda s: int((s == "wait").sum())),
        "return":       grp["r_total"].sum(),
        "duration_s":   (grp["time"].max() - grp["time"].min()).dt.total_seconds(),
        "focal_train":  grp["focal_train"].first(),
        "first_time":   grp["time"].min(),
    }).reset_index()
    return out
