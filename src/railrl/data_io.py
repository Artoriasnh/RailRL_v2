"""I/O utilities — CSV → parquet caching, typed loaders, chunked iteration."""
from __future__ import annotations
import ast
from pathlib import Path
from typing import Iterator, Optional, Sequence

import pandas as pd

from . import config as C


# ===== TD data =====

TD_DTYPES = {
    "descr": "string",
    "from_berth": "string",
    "to_berth": "string",
    "type": "category",
    "id": "string",
    "state": "Int8",
    "change": "string",
    "trainid_filled": "string",
    "timegap": "Float32",
    "Panel_Request": "string",
}


def stream_td(
    chunksize: int = 2_000_000,
    columns: Optional[Sequence[str]] = None,
    nrows: Optional[int] = None,
) -> Iterator[pd.DataFrame]:
    """Yield successive chunks of TD_data.csv with proper dtypes.

    Args:
        chunksize: rows per chunk (memory budget)
        columns:   subset of columns to load (faster if you don't need all)
        nrows:     stop after N rows (useful for sandbox-time-bounded runs)
    """
    use_dtypes = {k: v for k, v in TD_DTYPES.items() if columns is None or k in columns}
    yield from pd.read_csv(
        C.TD_CSV,
        usecols=list(columns) if columns else None,
        dtype=use_dtypes,
        parse_dates=["time"] if columns is None or "time" in columns else False,
        chunksize=chunksize,
        nrows=nrows,
        low_memory=False,
    )


def td_to_parquet(force: bool = False) -> Path:
    """One-off conversion of TD_data.csv → parquet for fast subsequent reads."""
    if C.TD_PARQUET.exists() and not force:
        return C.TD_PARQUET
    print(f"[data_io] Converting TD CSV → parquet: {C.TD_PARQUET}")
    parts = []
    n = 0
    for i, chunk in enumerate(stream_td(chunksize=2_000_000)):
        parts.append(chunk)
        n += len(chunk)
        print(f"  chunk {i}: cumulative rows = {n:,}", flush=True)
    df = pd.concat(parts, ignore_index=True)
    df.to_parquet(C.TD_PARQUET, index=False, compression="zstd")
    print(f"[data_io] Wrote {len(df):,} rows → {C.TD_PARQUET}")
    return C.TD_PARQUET


def load_td(columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Load TD data from parquet, materialising the parquet on first call."""
    if not C.TD_PARQUET.exists():
        td_to_parquet()
    return pd.read_parquet(C.TD_PARQUET, columns=list(columns) if columns else None)


# ===== Movements (TRUST) =====

def load_movements() -> pd.DataFrame:
    """Load Train Movements CSV (50 MB). Caches a parquet on first call."""
    if C.MOVEMENTS_PARQUET.exists():
        return pd.read_parquet(C.MOVEMENTS_PARQUET)
    df = pd.read_csv(
        C.MOVEMENTS_CSV,
        parse_dates=[
            "gbtt_timestamp", "planned_timestamp",
            "actual_timestamp", "msg_queue_timestamp",
        ],
        low_memory=False,
    )
    df.to_parquet(C.MOVEMENTS_PARQUET, index=False, compression="zstd")
    return df


# ===== route_to_tc_all =====

def load_route_to_tc() -> pd.DataFrame:
    """Load route_to_tc_all.csv with its track-list column parsed into Python lists."""
    df = pd.read_csv(C.ROUTE_TO_TC_CSV)
    df["track_list"] = df["track"].apply(
        lambda s: ast.literal_eval(s) if isinstance(s, str) else []
    )
    return df
