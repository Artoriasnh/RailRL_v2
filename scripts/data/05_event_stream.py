#!/usr/bin/env python3
"""Phase 2.3 Iter 2.A — Build event-token stream parquet.

Reads TD events (parquet cache) → tokenises via change column → persists
the per-asset sorted token stream as event_tokens.parquet (~ 200 MB).

Required by 06_decision_points.py and 07_build_snapshots.py.

Usage:
    python scripts/p2_data_eng/05_event_stream.py            # full TD
    python scripts/p2_data_eng/05_event_stream.py --force    # re-build
"""
from __future__ import annotations
import argparse, time, sys
import pandas as pd

from railrl import config as C
from railrl.p2_data_eng.event_stream import EventTokenStream


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true",
                   help="re-build even if event_tokens.parquet exists")
    args = p.parse_args()

    if C.EVENT_STREAM_PARQUET.exists() and not args.force:
        print(f"event_tokens.parquet already exists at {C.EVENT_STREAM_PARQUET}")
        print("Use --force to re-build.")
        return 0

    print(f"Reading TD events from {C.TD_PARQUET} ...", flush=True)
    t0 = time.time()
    if not C.TD_PARQUET.exists():
        from railrl.data_io import td_to_parquet
        td_to_parquet()
    df = pd.read_parquet(C.TD_PARQUET, columns=["time", "type", "change"])
    print(f"  {len(df):,} rows in {time.time()-t0:.1f}s")

    t1 = time.time()
    es = EventTokenStream.from_td_dataframe(df)
    print(f"Built EventTokenStream in {time.time()-t1:.1f}s · {es.summary()}")

    t2 = time.time()
    es.to_parquet()
    print(f"Wrote {C.EVENT_STREAM_PARQUET} in {time.time()-t2:.1f}s "
          f"(size: {C.EVENT_STREAM_PARQUET.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
