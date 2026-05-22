"""Stage 3 — Materialize pass_assignments.parquet (TRUST id pass intervals).

A "pass" = one operational journey of one TRUST train_id through Derby
(spec 01 §17.2). episode.py uses these intervals to assign each decision
point a pass_id → episode_idx → position_in_episode → is_last_in_episode.

Without this file, 05_build_snapshots.py falls back to 30-min gap clustering
(lower quality episode boundaries). Run this ONCE before 05.

Usage:
  python scripts/mdp/00_build_pass_assignments.py          # build
  python scripts/mdp/00_build_pass_assignments.py --force  # rebuild

Algorithm (fast, vectorized):
  For each TRUST train_id in Movements, pass interval =
    [min(actual_timestamp), max(actual_timestamp)] widened by ±30 min.
  headcode (= train_id[2:6]) is the join key to TD focal_train.

Output: outputs/passes/pass_assignments.parquet
        columns: trainid_filled, pass_id, pass_t_first_ns, pass_t_last_ns, pass_source
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from railrl import config as C
from railrl.mdp.pass_assignment import build_pass_intervals


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true",
                    help="Rebuild even if pass_assignments.parquet exists")
    return p.parse_args()


def main():
    args = _parse_args()
    out_path = C.PASS_ASSIGNMENTS_PARQUET
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Build pass_assignments.parquet (TRUST id pass intervals)")
    print("=" * 70)

    if out_path.exists() and not args.force:
        print(f"[skip] {out_path} already exists. Use --force to rebuild.")
        pa = pd.read_parquet(out_path)
    else:
        # Prefer the cached movements.parquet (faster); else read the CSV.
        if Path(C.MOVEMENTS_PARQUET).exists():
            src = C.MOVEMENTS_PARQUET
            print(f"[1/2] source: {src} (cached parquet)")
        elif Path(C.MOVEMENTS_CSV).exists():
            src = C.MOVEMENTS_CSV
            print(f"[1/2] source: {src} (raw csv)")
        else:
            print(f"[ERROR] neither {C.MOVEMENTS_PARQUET} nor {C.MOVEMENTS_CSV} found.")
            sys.exit(1)

        pa = build_pass_intervals(src)
        print(f"[2/2] writing {out_path}")
        pa.to_parquet(out_path, index=False, compression="zstd")

    # ----- Summary -----
    summary = {
        "n_passes":            int(len(pa)),
        "n_unique_headcodes":  int(pa["trainid_filled"].nunique()),
        "n_unique_pass_ids":   int(pa["pass_id"].nunique()),
        "pass_source_counts":  pa["pass_source"].value_counts().to_dict(),
    }
    # Span of intervals + per-headcode pass count distribution
    if len(pa):
        per_hc = pa.groupby("trainid_filled").size()
        summary["passes_per_headcode"] = {
            "mean":  round(float(per_hc.mean()), 2),
            "max":   int(per_hc.max()),
            "min":   int(per_hc.min()),
        }
        t_first = pd.to_datetime(pa["pass_t_first_ns"].min())
        t_last = pd.to_datetime(pa["pass_t_last_ns"].max())
        summary["time_span"] = f"{t_first} → {t_last}"

    summary_path = out_path.parent / "pass_assignments_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print()
    print("-" * 70)
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print("-" * 70)
    print(f"[write] {out_path}")
    print(f"[write] {summary_path}")
    print("=" * 70)
    print("DONE — now re-run 05_build_snapshots.py to use TRUST-based episodes.")
    print("=" * 70)


if __name__ == "__main__":
    main()
