"""Profile SnapshotBuilder.build_snapshot to find the per-snapshot bottleneck.

Builds a few hundred real snapshots under cProfile and prints the hottest
functions. Run this (≈2-3 min incl. TD load) and paste the output so we can
target the real bottleneck instead of guessing.

Usage:
    python scripts/mdp/profile_build.py
    python scripts/mdp/profile_build.py --n 300
"""
from __future__ import annotations
import argparse
import cProfile
import io
import pstats
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from railrl import config as C
from railrl.mdp.state import SnapshotBuilder
from railrl.mdp.episode import build_episodes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300, help="Number of snapshots to profile")
    ap.add_argument("--skip", type=int, default=0, help="Skip the first N decisions (avoid all-early-data)")
    args = ap.parse_args()

    print(f"loading inputs...", flush=True)
    t0 = time.time()
    dp = pd.read_parquet(C.DECISION_POINTS_V2_PARQUET)
    td = pd.read_parquet(C.TD_PARQUET)
    mv = None
    if Path(C.MOVEMENTS_PARQUET).exists():
        mv = pd.read_parquet(C.MOVEMENTS_PARQUET)
    print(f"  loaded dp={len(dp):,} td={len(td):,} in {time.time()-t0:.1f}s", flush=True)

    # Episode metadata (so candidate_route_ids etc. are present)
    pass_df = None
    if Path(C.PASS_ASSIGNMENTS_PARQUET).exists():
        pass_df = pd.read_parquet(C.PASS_ASSIGNMENTS_PARQUET)
    dp = build_episodes(dp, pass_assignments=pass_df).reset_index(drop=True)
    dp["sample_id"] = np.arange(len(dp), dtype=np.int64)

    # Take a mid-corpus slice (strided) so we hit both wait & set, rich & sparse
    sub = dp.iloc[args.skip: args.skip + args.n * 7: 7].head(args.n)
    print(f"  profiling {len(sub):,} snapshots", flush=True)

    print(f"building SnapshotBuilder (loads histories)...", flush=True)
    t0 = time.time()
    sb = SnapshotBuilder.build_default(td, movements=mv)
    sb.run_leak_audit = False   # exclude audit from the profile
    print(f"  builder ready in {time.time()-t0:.1f}s", flush=True)

    decisions = [r.to_dict() for _, r in sub.iterrows()]

    def _run():
        for i, dec in enumerate(decisions):
            sb.build_snapshot(dec, sample_id=int(dec["sample_id"]))

    print(f"profiling...", flush=True)
    t0 = time.time()
    pr = cProfile.Profile()
    pr.enable()
    _run()
    pr.disable()
    wall = time.time() - t0
    rate = len(decisions) / wall
    print(f"\n  {len(decisions)} snapshots in {wall:.2f}s  =  {rate:.1f}/s  "
          f"({1000*wall/len(decisions):.1f} ms/snapshot)\n")

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("tottime")
    ps.print_stats(25)
    print("=" * 70)
    print("TOP 25 BY tottime (self time, excludes sub-calls)")
    print("=" * 70)
    print(s.getvalue())


if __name__ == "__main__":
    main()
