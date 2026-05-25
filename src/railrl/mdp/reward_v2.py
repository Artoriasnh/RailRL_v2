"""Stage 4.6.5 — v2 reward recompute helpers (spec 01 §9-14, spec 02 §8).

WHY THIS MODULE EXISTS
----------------------
The v1 reward scripts (scripts/data/09_label_pr_outcomes.py and
10_compute_rewards.py) were written against the OLD decision_points.parquet
(~727k rows) and import `railrl.p2_data_eng.snapshot`, which the v2 shim does
NOT re-export. They were therefore never run under v2. The existing
outputs/rewards/decision_rewards.parquet (726,978 rows) and pr_outcomes.parquet
are stale v1 carry-overs — they are keyed to a *different* decision set than the
current decision_points_v2.parquet (≈2.0M rows) and snapshots_v2.parquet
(1,996,572 rows). Training on them would silently mismatch every reward to the
wrong state/action (将错就错). They must be recomputed from decision_points_v2.

KEY CORRECTNESS INVARIANT — the join key
----------------------------------------
scripts/mdp/05_build_snapshots.py assigns, line 121-123:

    dp = build_episodes(dp, pass_assignments=pass_df)   # ⚠ REORDERS rows
    dp = dp.reset_index(drop=True)
    dp["sample_id"] = np.arange(len(dp), dtype=np.int64)   # GLOBAL id

⚠ CRITICAL: build_episodes (mdp/episode.py) sorts by ["focal_train","t"] then
by ["pass_id","t"] BEFORE sample_id is assigned. So sample_id follows the
*episode-sorted* order, NOT the natural decision_points_v2 row order. An earlier
version of build_rewardfmt assigned sample_id on the natural order → the two
sample_id spaces disagreed → the reward↔snapshot merge was silently scrambled
(same sample_id pointed at a 'set' decision in snapshots but a 'wait' in rewards).
We therefore reproduce the build_episodes step EXACTLY (same pass_assignments
file, deterministic sorts) so the integer sample_id is a true 1:1 join. This is
far more robust than joining on (focal_train, focal_signal, t, chosen_route_id)
tuples, which are not unique (duplicate PRs) and risk dtype ambiguity.

COLUMN BRIDGES
--------------
1. decision_points_v2 uses `t` / `trigger_type`; the v1 reward feature/episode
   code expects `time` / `trigger`. build_rewardfmt() renames once.
2. The snapshot schema (mdp/schema.py REWARD_COLS) uses `outcome`,
   `r_throughput_raw`, `r_headway_raw`; the reward_model emits `route_outcome`,
   `r_thru_raw`, `r_head_raw`. REWARD_MERGE_MAP bridges snapshot_col → rewards_col
   at merge time, so decision_rewards_v2.parquet stays schema-identical to the
   v1 reward table (reusable by the v1 health-check scripts).
"""
from __future__ import annotations
from pathlib import Path

from .. import config as C

# ----- v2 reward artifacts (kept separate from the stale v1 files) -----
REWARDFMT_PARQUET     = C.DECISION_POINTS_DIR / "decision_points_v2_rewardfmt.parquet"
PR_OUTCOMES_V2        = C.REWARDS_DIR / "pr_outcomes_v2.parquet"
PR_OUTCOMES_V2_SUMMARY = C.REWARDS_DIR / "pr_outcomes_v2_summary.json"
DECISION_REWARDS_V2   = C.REWARDS_DIR / "decision_rewards_v2.parquet"
DECISION_REWARDS_V2_SUMMARY = C.REWARDS_DIR / "decision_rewards_v2_summary.json"
SNAPSHOTS_V2_REWARDED = C.SNAPSHOTS_DIR / "snapshots_v2_rewarded.parquet"
TIME_LABELS_V2        = C.REWARDS_DIR / "time_labels_v2.parquet"   # 4.7.2b L_time label

# snapshot reward column  ->  decision_rewards_v2 column (name bridge)
REWARD_MERGE_MAP = {
    "outcome":                 "route_outcome",
    "approach_distance":       "approach_distance",
    "delay_change_seconds":    "delay_change_seconds",
    "next_tc_headway_seconds": "next_tc_headway_seconds",
    "gate":                    "gate",
    "r_delay_raw":             "r_delay_raw",
    "r_throughput_raw":        "r_thru_raw",
    "r_headway_raw":           "r_head_raw",
    "r_wait_raw":              "r_wait_raw",
    "r_delay":                 "r_delay",
    "r_throughput":            "r_throughput",
    "r_headway":               "r_headway",
    "r_wait":                  "r_wait",
    "r_total":                 "r_total",
}

# float reward columns (snapshot side) — outcome is the only string one
REWARD_FLOAT_COLS = [c for c in REWARD_MERGE_MAP if c != "outcome"]


def _load_pass_assignments():
    """Mirror scripts/mdp/05_build_snapshots.py::_load_pass_assignments."""
    import pandas as pd
    p = C.PASS_ASSIGNMENTS_PARQUET
    if not p.exists():
        print(f"[reward_v2][warn] {p} not found — falling back to gap-based pass_id "
              f"(MUST match what 05_build_snapshots used!)")
        return None
    return pd.read_parquet(p)


def build_rewardfmt(force: bool = False):
    """Normalise decision_points_v2 → reward-format intermediate parquet.

    Returns the pandas DataFrame (also cached to REWARDFMT_PARQUET).

    sample_id is derived EXACTLY as scripts/mdp/05_build_snapshots.py does:
    build_episodes (which sorts by focal_train/t then pass_id/t) → reset_index
    → arange. This is the ONLY way the integer sample_id lines up 1:1 with the
    snapshots; assigning it on the natural decision_points_v2 order silently
    scrambles the reward↔snapshot merge.

    Columns: sample_id (int64), focal_train, focal_signal, time (ns), label,
    chosen_route_id, trigger, plus episode metadata (pass_id, episode_idx,
    position_in_episode, is_last_in_episode).
    """
    import numpy as np
    import pandas as pd
    from .episode import build_episodes

    if REWARDFMT_PARQUET.exists() and not force:
        return pd.read_parquet(REWARDFMT_PARQUET)

    dp = pd.read_parquet(C.DECISION_POINTS_V2_PARQUET)

    # ⭐ Reproduce 05_build_snapshots.py:121-123 EXACTLY (order-critical).
    pass_df = _load_pass_assignments()
    dp = build_episodes(dp, pass_assignments=pass_df)   # REORDERS
    dp = dp.reset_index(drop=True)
    dp["sample_id"] = np.arange(len(dp), dtype=np.int64)

    # v2 -> v1 column names expected by the reward feature/episode code.
    rename = {}
    if "t" in dp.columns and "time" not in dp.columns:
        rename["t"] = "time"
    if "trigger_type" in dp.columns and "trigger" not in dp.columns:
        rename["trigger_type"] = "trigger"
    dp = dp.rename(columns=rename)

    dp["focal_train"]  = dp["focal_train"].astype(str)
    dp["focal_signal"] = dp["focal_signal"].astype(str)
    # chosen_route_id is None for wait rows; keep as object/str (NaN-safe)
    dp["chosen_route_id"] = dp["chosen_route_id"].astype("string")
    dp["time"] = pd.to_datetime(dp["time"])

    keep = ["sample_id", "focal_train", "focal_signal", "time",
            "label", "chosen_route_id", "trigger",
            "pass_id", "episode_idx", "position_in_episode", "is_last_in_episode"]
    dp = dp[[c for c in keep if c in dp.columns]].copy()

    REWARDFMT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    dp.to_parquet(REWARDFMT_PARQUET, index=False, compression="zstd")
    return dp


def compute_lead_time_buckets(set_decisions, route_first_tc, event_stream, asset_index):
    """spec 03 §7.2 — τ = first-occupy(route's first TC) − t_PR per set decision.

    Returns (tau_s, bucket) numpy arrays aligned to set_decisions rows. τ is the
    lead time from the decision to when the route's FIRST traversed TC is next
    occupied (same occupy-onset scan as reward_features.compute_next_tc_headways).
    bucket = heads.time_bucket(τ) ∈ {0..4}; NaN τ → -1 (excluded from L_time).
    """
    import numpy as np
    import pandas as pd
    from ..policies.heads import time_bucket

    ev_by = event_stream._build_per_asset_index()
    times_full = event_stream.time_ns
    states_full = event_stream.state

    n = len(set_decisions)
    tau = np.full(n, np.nan, dtype=np.float64)
    rows = list(set_decisions.itertuples(index=False))
    for i, r in enumerate(rows):
        rid = str(r.chosen_route_id)
        first_tc = route_first_tc.get(rid)
        if first_tc is None:
            continue
        tc_idx = asset_index.idx(first_tc)
        if tc_idx is None:
            continue
        pos = ev_by.get(int(tc_idx))
        if pos is None or pos.size == 0:
            continue
        t_ns = int(pd.Timestamp(r.time).value)
        tc_t = times_full[pos]
        tc_s = states_full[pos]
        j = int(np.searchsorted(tc_t, t_ns, side="left"))
        seg = tc_s[j:]
        occ = np.flatnonzero(seg == 1)            # first occupy ONSET at/after t
        if occ.size == 0:
            continue
        occ_t = int(tc_t[j + int(occ[0])])
        tau[i] = (occ_t - t_ns) / 1e9
    buckets = np.array(
        [time_bucket(float(t)) if np.isfinite(t) else -1 for t in tau], dtype=np.int64)
    return tau, buckets
