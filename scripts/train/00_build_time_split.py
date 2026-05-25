"""Stage 4.7.1.5 — Build the TIME-BASED train/val/test split (spec 04 §4.1).

Replaces the leaky md5(pass_id) hash split. Each episode (pass_id) is assigned
to a split by its START time (min t over the episode), so:
  * the split is temporal (no future leakage — spec 04 §4.1 + 教训 6), and
  * an episode never spans two splits.

train: start < 2024-02-01 | val: < 2024-03-01 | test: >= 2024-03-01
(boundaries from C.VAL_START / C.TEST_START).

Output: outputs/snapshots/pass_split.parquet  (columns: pass_id, split, t_first)
Both 01_build_normalization_stats.py and SnapshotDataset read this so the split
is identical everywhere.

Run on Windows:
    python scripts/train/00_build_time_split.py
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from railrl import config as C
from railrl.encoders.input_pipeline import time_split_of


def main():
    import pandas as pd
    import pyarrow.parquet as pq

    src = C.SNAPSHOTS_V2_PARQUET
    print(f"[1/3] Reading (pass_id, t) from {src.name} ...")
    tbl = pq.read_table(str(src), columns=["pass_id", "t"])
    df = tbl.to_pandas()
    df["t"] = pd.to_datetime(df["t"])
    print(f"      {len(df):,} snapshot rows, {df['pass_id'].nunique():,} episodes")

    print("[2/3] Assigning each episode by its START time ...")
    first = df.groupby("pass_id", sort=False)["t"].min().reset_index()
    first = first.rename(columns={"t": "t_first"})
    first["split"] = first["t_first"].map(time_split_of)

    # Row-level split counts (what the loader will actually yield)
    pass_to_split = dict(zip(first["pass_id"], first["split"]))
    df["split"] = df["pass_id"].map(pass_to_split)
    row_counts = df["split"].value_counts().to_dict()
    ep_counts = first["split"].value_counts().to_dict()

    print("[3/3] Writing pass_split.parquet ...")
    out = first[["pass_id", "split", "t_first"]]
    out.to_parquet(C.PASS_SPLIT_PARQUET, index=False, compression="zstd")
    print(f"      -> {C.PASS_SPLIT_PARQUET}")

    n = len(df)
    print("\n=== Time-based split (spec 04 §4.1) ===")
    print(f"  date boundaries: train < {C.VAL_START} | val < {C.TEST_START} | test >=")
    print(f"  episodes:  " + "  ".join(
        f"{k}={ep_counts.get(k,0):,}" for k in ("train", "val", "test")))
    for k in ("train", "val", "test"):
        c = row_counts.get(k, 0)
        print(f"  {k:<6s} rows: {c:>10,}  ({100*c/max(n,1):.1f}%)")
    # sanity: every episode is in exactly one split (by construction)
    assert df["split"].notna().all(), "some rows have no split — pass_id mismatch!"
    print("  every row assigned ✓  (episodes never span splits, by construction)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
