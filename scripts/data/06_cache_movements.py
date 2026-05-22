"""One-off: cache Movements.csv → movements.parquet (+ headcode diagnostics).

The raw Movements.csv (~50 MB) is slow to parse on every run. This script
materialises outputs/cache/movements.parquet ONCE so downstream steps
(05_build_snapshots.py via data_io.load_movements()) read the parquet
instantly.

Usage:
  python scripts/data/06_cache_movements.py            # cache + diagnostics
  python scripts/data/06_cache_movements.py --force    # re-cache even if exists

TRUST train_id structure (Table 3.6, ESWA paper §3):
    [AA]   Stanox Prefix  (2 chars) — area where the train starts
    [BBBB] Headcode       (4 chars) — signalling ID  ← what we match to TD focal_train
    [C]    TSPEED         (1 char)  — train status code
    [D]    Call Code      (1 char)  — letter/number based on departure time
    [EE]   Day Indicator  (2 chars) — day of month the train originated
  Example: '851S49ME28' → AA=85, BBBB=1S49, C=M, D=E, EE=28

MovementsLookup.build() extracts the headcode from train_id[2:6]. This script
also reports headcode + platform diagnostics so you can sanity-check the join.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from railrl import config as C


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true",
                    help="Re-cache even if movements.parquet already exists")
    return p.parse_args()


def main():
    args = _parse_args()
    print("=" * 70)
    print("Cache Movements.csv → movements.parquet")
    print("=" * 70)

    if C.MOVEMENTS_PARQUET.exists() and not args.force:
        print(f"[skip] {C.MOVEMENTS_PARQUET} already exists "
              f"({C.MOVEMENTS_PARQUET.stat().st_size / 1e6:.1f} MB). "
              f"Use --force to re-cache.")
        df = pd.read_parquet(C.MOVEMENTS_PARQUET)
    else:
        print(f"[1/2] reading {C.MOVEMENTS_CSV} ...")
        if not Path(C.MOVEMENTS_CSV).exists():
            print(f"[ERROR] {C.MOVEMENTS_CSV} not found.")
            sys.exit(1)
        df = pd.read_csv(
            C.MOVEMENTS_CSV,
            parse_dates=[
                "gbtt_timestamp", "planned_timestamp",
                "actual_timestamp", "msg_queue_timestamp",
            ],
            low_memory=False,
        )
        C.MOVEMENTS_PARQUET.parent.mkdir(parents=True, exist_ok=True)
        print(f"[2/2] writing {C.MOVEMENTS_PARQUET} ...")
        df.to_parquet(C.MOVEMENTS_PARQUET, index=False, compression="zstd")
        print(f"      wrote {len(df):,} rows "
              f"({C.MOVEMENTS_PARQUET.stat().st_size / 1e6:.1f} MB)")

    # ----- Diagnostics -----
    print()
    print("-" * 70)
    print("Diagnostics")
    print("-" * 70)
    print(f"rows: {len(df):,}")
    print(f"columns: {df.columns.tolist()}")

    if "train_id" in df.columns:
        tid = df["train_id"].dropna().astype(str)
        lens = tid.str.len().value_counts().to_dict()
        print(f"\ntrain_id length distribution: {lens}")
        # Headcode = chars [2:6]  (Table 3.6 [BBBB])
        hc = tid.str[2:6]
        hc_valid = hc.str.match(r"^[0-9][A-Z][0-9]{2}$")
        print(f"headcode [2:6] matches NXNN: {hc_valid.sum():,}/{len(tid):,} "
              f"({100*hc_valid.mean():.1f}%)")
        print(f"sample headcodes: {hc[hc_valid].unique()[:12].tolist()}")
        print(f"n unique headcodes: {hc[hc_valid].nunique():,}")

    if "platform" in df.columns:
        pv = df["platform"].value_counts(dropna=False).sort_index()
        print(f"\nplatform distribution: "
              f"{ {('NaN' if pd.isna(k) else k): int(v) for k, v in pv.items()} }")
        n7 = int((df['platform'] == 7).sum())
        print(f"platform 7 (pilot line, EC5487/TECV north + EC5484/TECS south): "
              f"{n7:,} rows")

    if "gbtt_timestamp" in df.columns:
        g = pd.to_datetime(df["gbtt_timestamp"], errors="coerce")
        print(f"\ngbtt_timestamp: {g.notna().sum():,} non-null "
              f"({g.min()} → {g.max()})")

    if "event_type" in df.columns:
        print(f"event_type: {df['event_type'].value_counts().to_dict()}")

    print()
    print("=" * 70)
    print(f"DONE → {C.MOVEMENTS_PARQUET}")
    print("=" * 70)


if __name__ == "__main__":
    main()
