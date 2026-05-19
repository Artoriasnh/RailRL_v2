"""P2.4 utility - Re-apply RewardModel to existing decision_rewards.parquet."""
from __future__ import annotations
import argparse
import json
import sys
import time as _time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from railrl import config as C
from railrl.p2_data_eng.reward_model import RewardModel
from railrl.p2_data_eng.episodes     import assign_episodes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delay-clip-seconds", type=float, default=1800.0)
    args = ap.parse_args()

    print("Loading existing decision_rewards.parquet ...")
    t0 = _time.time()
    df = pd.read_parquet(C.DECISION_REWARDS_PARQUET)
    print(f"  {len(df):,} rows, {_time.time()-t0:.1f}s")

    drop_cols = ["gate", "r_delay_raw", "r_thru_raw", "r_head_raw", "r_wait_raw",
                  "r_delay", "r_throughput", "r_headway", "r_wait", "r_total",
                  "episode_id"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    print(f"Applying RewardModel (delay_clip={args.delay_clip_seconds}s) ...")
    t0 = _time.time()
    model = RewardModel.from_config(delay_clip_seconds=args.delay_clip_seconds)
    rew = model.compute_batch(df)
    df = pd.concat([df, rew], axis=1)
    df = assign_episodes(df)
    print(f"  done, {_time.time()-t0:.1f}s")

    df.to_parquet(C.DECISION_REWARDS_PARQUET, index=False, compression="zstd")
    print(f"Wrote {C.DECISION_REWARDS_PARQUET}")

    n_eps = int(df["episode_id"].nunique())
    by_label = df["label"].value_counts().to_dict()
    summary = {
        "n_decisions":  int(len(df)),
        "n_episodes":   n_eps,
        "delay_clip_seconds": args.delay_clip_seconds,
        "by_label":     by_label,
        "r_total_describe": df["r_total"].describe().round(3).to_dict(),
        "components_mean": {
            "r_delay":      float(df["r_delay"].mean()),
            "r_throughput": float(df["r_throughput"].mean()),
            "r_headway":    float(df["r_headway"].mean()),
            "r_wait":       float(df["r_wait"].mean()),
        },
        "weights":     dict(model.weights),
        "thresholds": {
            "H_min_seconds":   model.thresholds.H_min_seconds,
            "d_gate_05_max":   model.thresholds.d_gate_05_max,
            "d_gate_01_max":   model.thresholds.d_gate_01_max,
            "window_seconds":  model.thresholds.window_seconds,
        },
    }
    C.DECISION_REWARDS_SUMMARY.write_text(json.dumps(summary, indent=2, default=str),
                                            encoding="utf-8")
    print(f"Wrote {C.DECISION_REWARDS_SUMMARY}")

    rt = df["r_total"]
    n_total = len(df)
    print("")
    print("=== Reward summary ===")
    print(f"  n_decisions: {n_total:,}  (episodes: {n_eps:,})")
    print(f"  r_total mean / std / min / max:")
    print(f"    {rt.mean():+.3f} / {rt.std():.3f} / {rt.min():+.3f} / {rt.max():+.3f}")
    for k, v in summary["components_mean"].items():
        print(f"    {k:<14s} {v:+.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
