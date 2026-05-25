"""Stage 4.6.5 [3/3] — Merge v2 rewards into snapshots_v2 by sample_id.

snapshots_v2.parquet embeds the reward columns per row (mdp/schema.py
REWARD_COLS), but the builder writes them as NaN placeholders
(state.py:265 "will be joined from decision_rewards.parquet downstream").
This script fills them from decision_rewards_v2.parquet.

JOIN KEY = sample_id (integer). decision_rewards_v2 carries the same sample_id
the snapshot builder assigned (arange over decision_points_v2), so this is an
exact 1:1 fill — no timestamp/tuple matching.

The merge streams snapshots_v2 in row-group batches via pyarrow, so the heavy
nested struct columns (event tokens, node lists) are passed through untouched
and memory stays bounded. Output is a NEW file (non-destructive):

    outputs/snapshots/snapshots_v2_rewarded.parquet

After verifying the report below, point training at the rewarded file (rename
it to snapshots_v2.parquet, or update C.SNAPSHOTS_V2_PARQUET).

Run on Windows:
    python scripts/mdp/10_merge_rewards_into_snapshots.py
    python scripts/mdp/10_merge_rewards_into_snapshots.py --batch-size 4096
"""
from __future__ import annotations
import argparse
import sys
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from railrl import config as C
from railrl.mdp import reward_v2 as RV


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=4096,
                    help="Rows per streamed batch (memory/throughput tradeoff).")
    ap.add_argument("--snapshots", type=Path, default=C.SNAPSHOTS_V2_PARQUET,
                    help="Input snapshots parquet (default snapshots_v2.parquet).")
    ap.add_argument("--rewards", type=Path, default=RV.DECISION_REWARDS_V2,
                    help="Input decision_rewards_v2.parquet.")
    ap.add_argument("--out", type=Path, default=RV.SNAPSHOTS_V2_REWARDED,
                    help="Output rewarded snapshots parquet.")
    args = ap.parse_args()

    import pyarrow as pa
    import pyarrow.parquet as pq

    if not args.rewards.exists():
        raise FileNotFoundError(
            f"{args.rewards} not found — run 09_compute_rewards_v2.py first.")
    if not args.snapshots.exists():
        raise FileNotFoundError(f"{args.snapshots} not found.")

    print(f"[1/3] Loading rewards: {args.rewards.name}")
    t0 = _time.time()
    rew = pd.read_parquet(args.rewards)
    rew_sid = rew["sample_id"].to_numpy(np.int64)
    assert pd.Series(rew_sid).is_unique, "decision_rewards_v2 sample_id not unique!"
    print(f"      {len(rew):,} reward rows, sample_id 0..{rew_sid.max()}, "
          f"{_time.time()-t0:.1f}s")

    # Size dense lookup arrays to cover both files' sample_id ranges.
    pf = pq.ParquetFile(args.snapshots)
    schema = pf.schema_arrow
    n_snap_rows = pf.metadata.num_rows
    # peek max snapshot sample_id (cheap: read only sample_id column)
    snap_sid_max = int(pq.read_table(args.snapshots, columns=["sample_id"])
                       ["sample_id"].to_numpy().max())
    N = int(max(rew_sid.max(), snap_sid_max)) + 1
    print(f"      snapshots: {n_snap_rows:,} rows, max sample_id {snap_sid_max}; "
          f"lookup arrays sized {N:,}")

    # Build sample_id -> value lookup arrays (one per snapshot reward column).
    print("[2/3] Building per-sample_id reward lookups ...")
    float_lut: dict[str, np.ndarray] = {}
    for snap_col in RV.REWARD_FLOAT_COLS:
        src_col = RV.REWARD_MERGE_MAP[snap_col]
        lut = np.full(N, np.nan, dtype=np.float64)
        lut[rew_sid] = rew[src_col].to_numpy(np.float64)
        float_lut[snap_col] = lut
    # outcome (string) — use pd.isna so pandas NA / None / nan all map to a real
    # null (NOT the literal string "<NA>" that str(pd.NA) would produce).
    outcome_lut = np.empty(N, dtype=object)
    outcome_lut[:] = None
    oc_src = rew[RV.REWARD_MERGE_MAP["outcome"]]
    outcome_lut[rew_sid] = [None if pd.isna(v) else str(v)
                            for v in oc_src.to_numpy()]

    name_to_idx = {name: i for i, name in enumerate(schema.names)}
    for c in RV.REWARD_MERGE_MAP:
        if c not in name_to_idx:
            raise KeyError(f"snapshots schema missing reward column '{c}'")

    # ----- stream + fill -----
    print(f"[3/3] Streaming merge -> {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(args.out, schema, compression="zstd")

    n_done = 0
    n_matched = 0          # snapshot rows whose sample_id had a reward row
    n_set = n_wait = 0
    comp_sum = {"r_delay": 0.0, "r_throughput": 0.0, "r_headway": 0.0, "r_wait": 0.0}
    rtot_sum = 0.0
    rtot_finite = 0
    t0 = _time.time()
    try:
        for batch in pf.iter_batches(batch_size=args.batch_size):
            sid = batch.column(name_to_idx["sample_id"]).to_numpy(zero_copy_only=False)
            sid = sid.astype(np.int64)
            arrays = list(batch.columns)

            # which of these snapshot rows actually have a reward row?
            matched = ~np.isnan(float_lut["r_total"][sid])
            n_matched += int(matched.sum())

            for snap_col, lut in float_lut.items():
                vals = lut[sid]
                arrays[name_to_idx[snap_col]] = pa.array(vals, type=pa.float64())
            oc_vals = outcome_lut[sid]
            arrays[name_to_idx["outcome"]] = pa.array(list(oc_vals), type=pa.string())

            new_batch = pa.RecordBatch.from_arrays(arrays, schema=schema)
            writer.write_batch(new_batch)

            # streaming sanity accumulation
            label = batch.column(name_to_idx["label"]).to_numpy(zero_copy_only=False)
            n_set  += int((label == "set").sum())
            n_wait += int((label == "wait").sum())
            rt = float_lut["r_total"][sid]
            fin = ~np.isnan(rt)
            rtot_finite += int(fin.sum())
            rtot_sum    += float(np.nansum(rt))
            for k in comp_sum:
                comp_sum[k] += float(np.nansum(float_lut[k][sid]))

            n_done += len(sid)
            if n_done % (args.batch_size * 50) < args.batch_size:
                rate = n_done / max(_time.time() - t0, 1e-9)
                print(f"      {n_done:,}/{n_snap_rows:,} "
                      f"({100*n_done/n_snap_rows:.1f}%)  {rate:,.0f} rows/s")
    finally:
        writer.close()

    print(f"\n=== Merge report ===")
    print(f"  snapshot rows written : {n_done:,}")
    print(f"  matched to a reward    : {n_matched:,} "
          f"({100*n_matched/max(n_done,1):.2f}%)")
    print(f"  unmatched (left NaN)   : {n_done - n_matched:,}")
    print(f"  label set / wait       : {n_set:,} / {n_wait:,}")
    print(f"  r_total finite         : {rtot_finite:,} "
          f"({100*rtot_finite/max(n_done,1):.2f}%)")
    if rtot_finite:
        print(f"  r_total mean (finite)  : {rtot_sum / rtot_finite:+.4f}")
        print(f"  component means (over all rows):")
        for k, v in comp_sum.items():
            print(f"    {k:<14s} {v / max(n_done,1):+.4f}")
    print(f"\n  wrote -> {args.out}")
    if n_matched < n_done:
        print("  NOTE: unmatched rows had no reward row for their sample_id "
              "(e.g. snapshot built for a decision the reward step dropped). "
              "Investigate before training if this is more than a handful.")
    print("\n  Next: verify the numbers above, then point training at the rewarded\n"
          "  file (rename to snapshots_v2.parquet or update C.SNAPSHOTS_V2_PARQUET).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
