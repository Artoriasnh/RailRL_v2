"""P2.4 Iter C - Compute decision-level rewards on full data.

Pipeline:
  1. Load decision_points + pr_outcomes + calibration thresholds.
  2. Compute approach_distance (set decisions only) via Iter A helper.
  3. Compute delay_change_seconds (all decisions) from Movements.
  4. Compute next_tc_headway_seconds (set+used only) from event stream.
  5. Apply RewardModel.compute_batch -> r_total + four components.
  6. Assign episodes per focal_train (gap > 30 min splits a new episode).
  7. Write outputs/p2_data_eng/rewards/decision_rewards.parquet
            outputs/p2_data_eng/rewards/decision_rewards_summary.json

Usage:
  python scripts/p2_data_eng/10_compute_rewards.py
  python scripts/p2_data_eng/10_compute_rewards.py --limit 10000   # smoke
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
from railrl.p2_data_eng.event_stream        import AssetIndex, EventTokenStream
from railrl.p2_data_eng.snapshot            import StaticGraphView
from railrl.p2_data_eng.reward_calibration  import (
    build_train_position_lookup_from_td,
)
from railrl.p2_data_eng.reward_features import (
    compute_delay_changes,
    compute_next_tc_headways,
    build_route_first_tc,
    compute_approach_distance_fast,
)
from railrl.p2_data_eng.reward_model import RewardModel
from railrl.p2_data_eng.episodes     import assign_episodes, episode_summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                     help="Only process first N decision points (smoke test).")
    args = ap.parse_args()

    # Smoke runs (--limit) go to a separate file so they cannot clobber the
    # full 2.64M production parquet.
    if args.limit:
        out_parquet = C.DECISION_REWARDS_PARQUET.with_name(
            f"decision_rewards_smoke_{args.limit}.parquet")
        out_summary = C.DECISION_REWARDS_SUMMARY.with_name(
            f"decision_rewards_smoke_{args.limit}_summary.json")
        print(f"  [smoke mode] outputs go to *_smoke_{args.limit}.* "
              f"(NOT overwriting full decision_rewards.parquet)")
    else:
        out_parquet = C.DECISION_REWARDS_PARQUET
        out_summary = C.DECISION_REWARDS_SUMMARY

    print("[1/7] Loading decision_points + pr_outcomes ...")
    t0 = _time.time()
    dp = pd.read_parquet(C.DECISION_POINTS_PARQUET).reset_index(drop=True)
    if args.limit:
        dp = dp.head(args.limit).copy()
    dp["decision_id"] = dp.index
    print(f"      decision_points: {len(dp):,} rows")

    pro = pd.read_parquet(C.PR_OUTCOMES_PARQUET)
    print(f"      pr_outcomes:     {len(pro):,} rows")
    t0 = _time.time()

    print("[2/7] Joining route_outcome onto set decisions ...")
    # pr_outcomes was generated in PR-row order. Match on (time, focal_signal,
    # focal_train, chosen_route_id).
    keys = ["time", "focal_signal", "focal_train", "chosen_route_id"]
    pro_keep = pro[keys + ["outcome"]].copy()
    pro_keep["focal_signal"] = pro_keep["focal_signal"].astype(str)
    pro_keep["focal_train"]  = pro_keep["focal_train"].astype(str)
    pro_keep["chosen_route_id"] = pro_keep["chosen_route_id"].astype(str)
    pro_keep.rename(columns={"outcome": "route_outcome"}, inplace=True)

    dp["focal_signal"] = dp["focal_signal"].astype(str)
    dp["focal_train"]  = dp["focal_train"].astype(str)
    dp["chosen_route_id"] = dp["chosen_route_id"].astype(str).fillna("")

    dp = dp.merge(pro_keep, on=keys, how="left")
    dp["route_outcome"] = dp["route_outcome"].astype("string").fillna(pd.NA)
    n_set = (dp["label"] == "set").sum()
    n_outcome = dp["route_outcome"].notna().sum()
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
    n_dist = dp["approach_distance"].notna().sum()
    print(f"      approach_distance attached: {n_dist:,}/{n_set:,} ({100*n_dist/max(n_set,1):.1f}%), "
          f"{_time.time()-t0:.1f}s")

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
    dp = assign_episodes(dp)
    print(f"      reward + episodes done, {_time.time()-t0:.1f}s")

    keep_cols = [
        "decision_id", "episode_id", "time", "focal_signal", "focal_train",
        "label", "chosen_route_id", "trigger",
        "approach_distance", "delay_change_seconds",
        "next_tc_headway_seconds", "route_outcome",
        "gate", "r_delay_raw", "r_thru_raw", "r_head_raw", "r_wait_raw",
        "r_delay", "r_throughput", "r_headway", "r_wait", "r_total",
    ]
    dp = dp[keep_cols]

    print("Writing outputs ...")
    dp.to_parquet(out_parquet, index=False, compression="zstd")
    print(f"  decision_rewards.parquet -> {out_parquet}")

    summary = {
        "n_decisions": int(len(dp)),
        "by_label":    dp["label"].value_counts().to_dict(),
        "n_episodes":  int(dp["episode_id"].nunique()),
        "r_total_describe": dp["r_total"].describe().round(3).to_dict(),
        "components_mean": {
            "r_delay":      float(dp["r_delay"].mean()),
            "r_throughput": float(dp["r_throughput"].mean()),
            "r_headway":    float(dp["r_headway"].mean()),
            "r_wait":       float(dp["r_wait"].mean()),
        },
        "feature_coverage": {
            "approach_distance":         int(dp["approach_distance"].notna().sum()),
            "delay_change_seconds":      int(dp["delay_change_seconds"].notna().sum()),
            "next_tc_headway_seconds":   int(dp["next_tc_headway_seconds"].notna().sum()),
            "route_outcome":             int(dp["route_outcome"].notna().sum()),
        },
        "weights": dict(model.weights),
        "thresholds": {
            "H_min_seconds":   model.thresholds.H_min_seconds,
            "d_gate_05_max":   model.thresholds.d_gate_05_max,
            "d_gate_01_max":   model.thresholds.d_gate_01_max,
            "window_seconds":  model.thresholds.window_seconds,
        },
    }
    out_summary.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"  decision_rewards_summary.json -> {out_summary}")

    print("\n=== Reward summary ===")
    print(f"  n_decisions: {summary['n_decisions']:,}  (episodes: {summary['n_episodes']:,})")
    print(f"  r_total mean / std / min / max:")
    rt = dp["r_total"]
    print(f"    {rt.mean():+.3f} / {rt.std():.3f} / {rt.min():+.3f} / {rt.max():+.3f}")
    print(f"  Component means (weighted):")
    for k, v in summary["components_mean"].items():
        print(f"    {k:<14s} {v:+.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
