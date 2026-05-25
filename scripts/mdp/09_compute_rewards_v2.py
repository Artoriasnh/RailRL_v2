"""Stage 4.6.5 [2/3] — Compute per-decision rewards on decision_points_v2.

Adapted from scripts/data/10_compute_rewards.py, fixed for v2:
  * reads the rewardfmt intermediate (t→time / trigger_type→trigger, sample_id pinned)
  * imports from railrl.data.* (the v1 `railrl.p2_data_eng.snapshot` import is dead
    in v2 — StaticGraphView now lives in railrl.data.static_graph_view)
  * carries `sample_id` through so the result merges 1:1 into snapshots_v2
  * output schema is otherwise identical to the v1 decision_rewards.parquet
    (route_outcome, r_thru_raw, r_head_raw, ...) so v1 health-check scripts still work

Pipeline (spec 01 §9-14):
  1. load rewardfmt + pr_outcomes_v2
  2. join route_outcome onto set rows
  3. approach_distance (set), delay_change (all), next_tc_headway (set+used)
  4. RewardModel.compute_batch -> 4 components + r_total
  5. assign episodes (gap > 30 min per focal_train)

Outputs:
    outputs/rewards/decision_rewards_v2.parquet
    outputs/rewards/decision_rewards_v2_summary.json

Run on Windows:
    python scripts/mdp/09_compute_rewards_v2.py
    python scripts/mdp/09_compute_rewards_v2.py --limit 20000   # smoke
"""
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
from railrl.mdp import reward_v2 as RV
from railrl.data.event_stream       import AssetIndex, EventTokenStream
from railrl.data.static_graph_view  import StaticGraphView
from railrl.data.reward_calibration import build_train_position_lookup_from_td
from railrl.data.reward_features import (
    compute_delay_changes,
    compute_next_tc_headways,
    build_route_first_tc,
    compute_approach_distance_fast,
)
from railrl.data.reward_model import RewardModel
from railrl.data.episodes     import assign_episodes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="Only process first N decision points (smoke test).")
    args = ap.parse_args()

    if args.limit:
        out_parquet = RV.DECISION_REWARDS_V2.with_name(
            f"decision_rewards_v2_smoke_{args.limit}.parquet")
        out_summary = RV.DECISION_REWARDS_V2_SUMMARY.with_name(
            f"decision_rewards_v2_smoke_{args.limit}_summary.json")
        print(f"  [smoke mode] outputs go to *_smoke_{args.limit}.* "
              f"(NOT overwriting decision_rewards_v2.parquet)")
        pr_path = RV.PR_OUTCOMES_V2.with_name(
            f"pr_outcomes_v2_smoke_{args.limit}.parquet")
        if not pr_path.exists():
            pr_path = RV.PR_OUTCOMES_V2   # fall back to full PR outcomes
    else:
        out_parquet = RV.DECISION_REWARDS_V2
        out_summary = RV.DECISION_REWARDS_V2_SUMMARY
        pr_path = RV.PR_OUTCOMES_V2

    print("[1/7] Loading rewardfmt decision points + pr_outcomes_v2 ...")
    t0 = _time.time()
    dp = RV.build_rewardfmt()              # cached; t→time, trigger, sample_id
    if args.limit:
        dp = dp.head(args.limit).copy()
    # decision_id retained for v1-script parity; sample_id is the real join key.
    dp = dp.reset_index(drop=True)
    dp["decision_id"] = dp["sample_id"]
    print(f"      decision_points: {len(dp):,} rows "
          f"(sample_id 0..{int(dp['sample_id'].max())})")

    if not pr_path.exists():
        raise FileNotFoundError(
            f"pr_outcomes not found: {pr_path}\n"
            f"Run scripts/mdp/08_label_pr_outcomes_v2.py first.")
    pro = pd.read_parquet(pr_path)
    print(f"      pr_outcomes:     {len(pro):,} rows ({pr_path.name})")

    print("[2/7] Joining route_outcome onto set decisions (by sample_id) ...")
    t0 = _time.time()
    # The (time, focal_signal, focal_train, chosen_route_id) 4-tuple is NOT
    # unique (duplicate PRs at the same instant), so joining on it multiplies
    # rows. pr_outcomes_v2 carries sample_id (08_label_pr_outcomes_v2.py) →
    # join on the integer sample_id for a guaranteed 1:1 attach.
    if "sample_id" not in pro.columns:
        raise KeyError(
            "pr_outcomes is missing 'sample_id' — re-run "
            "scripts/mdp/08_label_pr_outcomes_v2.py with the updated "
            "pr_outcomes.py (it now carries sample_id).")
    assert pro["sample_id"].is_unique, "pr_outcomes sample_id not unique!"
    pro_keep = pro[["sample_id", "outcome"]].rename(columns={"outcome": "route_outcome"})

    dp["focal_signal"]    = dp["focal_signal"].astype(str)
    dp["focal_train"]     = dp["focal_train"].astype(str)
    dp["chosen_route_id"] = dp["chosen_route_id"].astype(str).fillna("")

    n_before = len(dp)
    dp = dp.merge(pro_keep, on="sample_id", how="left")
    assert len(dp) == n_before, (
        f"merge changed row count {n_before}->{len(dp)} — sample_id not 1:1!")
    dp["route_outcome"] = dp["route_outcome"].astype("string").fillna(pd.NA)
    n_set = int((dp["label"] == "set").sum())
    n_outcome = int(dp["route_outcome"].notna().sum())
    print(f"      route_outcome attached: {n_outcome:,}/{n_set:,} set decisions, "
          f"{_time.time()-t0:.1f}s")

    print("[3/7] Loading event stream + asset index + static graph ...")
    t0 = _time.time()
    es = EventTokenStream.load()
    ai = AssetIndex.load()
    es._build_per_asset_index()
    sg = StaticGraphView.load()
    print(f"      done, {_time.time()-t0:.1f}s")

    print("[4/7] Approach distance (set decisions) ...")
    t0 = _time.time()
    set_dp = dp[dp["label"] == "set"].copy()
    train_pos = build_train_position_lookup_from_td(C.TD_PARQUET)
    distances = compute_approach_distance_fast(set_dp, train_pos, sg)
    dp["approach_distance"] = np.nan
    dp.loc[set_dp.index, "approach_distance"] = distances
    n_dist = int(dp["approach_distance"].notna().sum())
    print(f"      approach_distance: {n_dist:,}/{n_set:,} "
          f"({100*n_dist/max(n_set,1):.1f}%), {_time.time()-t0:.1f}s")

    print("[5/7] Delay change (all decisions) ...")
    t0 = _time.time()
    dp["delay_change_seconds"] = compute_delay_changes(dp, C.MOVEMENTS_CSV)
    print(f"      {_time.time()-t0:.1f}s")

    print("[6/7] Next-TC headway (set+used decisions) ...")
    t0 = _time.time()
    route_first_tc = build_route_first_tc(ai, C.EDGE_TRAVERSES_PARQUET)
    set_with_outcome = dp[dp["label"] == "set"].copy()
    headways = compute_next_tc_headways(set_with_outcome, route_first_tc, es, ai)
    dp["next_tc_headway_seconds"] = np.nan
    dp.loc[set_with_outcome.index, "next_tc_headway_seconds"] = headways
    print(f"      {_time.time()-t0:.1f}s")

    print("[7/7] Applying RewardModel + assigning episodes ...")
    t0 = _time.time()
    model = RewardModel.from_config()
    rew = model.compute_batch(dp)
    dp = pd.concat([dp, rew], axis=1)
    dp = assign_episodes(dp)               # sorts (focal_train, time); sample_id travels
    print(f"      reward + episodes done, {_time.time()-t0:.1f}s")

    keep_cols = [
        "sample_id", "decision_id", "episode_id", "time",
        "focal_signal", "focal_train", "label", "chosen_route_id", "trigger",
        "approach_distance", "delay_change_seconds",
        "next_tc_headway_seconds", "route_outcome",
        "gate", "r_delay_raw", "r_thru_raw", "r_head_raw", "r_wait_raw",
        "r_delay", "r_throughput", "r_headway", "r_wait", "r_total",
    ]
    dp = dp[keep_cols]

    # sanity: sample_id must be unique + dense (it's the snapshot join key)
    assert dp["sample_id"].is_unique, "sample_id not unique — join would be ambiguous!"

    print("Writing outputs ...")
    dp.to_parquet(out_parquet, index=False, compression="zstd")
    print(f"  decision_rewards_v2 -> {out_parquet}")

    summary = {
        "n_decisions": int(len(dp)),
        "by_label":    {k: int(v) for k, v in dp["label"].value_counts().items()},
        "n_episodes":  int(dp["episode_id"].nunique()),
        "sample_id_range": [int(dp["sample_id"].min()), int(dp["sample_id"].max())],
        "r_total_describe": {k: float(v) for k, v in dp["r_total"].describe().items()},
        "components_mean": {
            "r_delay":      float(dp["r_delay"].mean()),
            "r_throughput": float(dp["r_throughput"].mean()),
            "r_headway":    float(dp["r_headway"].mean()),
            "r_wait":       float(dp["r_wait"].mean()),
        },
        "feature_coverage": {
            "approach_distance":       int(dp["approach_distance"].notna().sum()),
            "delay_change_seconds":    int(dp["delay_change_seconds"].notna().sum()),
            "next_tc_headway_seconds": int(dp["next_tc_headway_seconds"].notna().sum()),
            "route_outcome":           int(dp["route_outcome"].notna().sum()),
        },
        "weights": dict(model.weights),
        "thresholds": {
            "H_min_seconds":  model.thresholds.H_min_seconds,
            "d_gate_05_max":  model.thresholds.d_gate_05_max,
            "d_gate_01_max":  model.thresholds.d_gate_01_max,
            "window_seconds": model.thresholds.window_seconds,
        },
    }
    out_summary.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"  summary -> {out_summary}")

    print("\n=== Reward summary ===")
    print(f"  n_decisions: {summary['n_decisions']:,}  (episodes: {summary['n_episodes']:,})")
    rt = dp["r_total"]
    print(f"  r_total mean/std/min/max: {rt.mean():+.3f} / {rt.std():.3f} / "
          f"{rt.min():+.3f} / {rt.max():+.3f}")
    print("  Component means (weighted):")
    for k, v in summary["components_mean"].items():
        print(f"    {k:<14s} {v:+.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
