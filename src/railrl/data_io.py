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

def correct_movements_bst(
    df: pd.DataFrame,
    time_cols=("actual_timestamp", "planned_timestamp", "gbtt_timestamp"),
    ref_col: str = "actual_timestamp",
) -> pd.DataFrame:
    """Fix the +1h Movements acquisition bug for the 2023-04-04..07-31 runs.

    Subtracts 1h from `time_cols` for rows whose `ref_col` falls in
    [MOVEMENTS_BST_FIX_START, MOVEMENTS_BST_FIX_END) (bounds in data gaps → exact).
    Only the absolute clock is wrong; actual−planned (delay) is invariant under a
    joint shift, so timetable_variation/delay values are preserved. Idempotency is
    NOT guaranteed (applying twice subtracts 2h) — callers apply exactly once on a
    raw frame. See config.MOVEMENTS_BST_FIX_* + IMPLEMENTATION_LOG 2026-05-24 fix #2.
    """
    if ref_col not in df.columns:
        return df
    start = pd.Timestamp(C.MOVEMENTS_BST_FIX_START)
    end = pd.Timestamp(C.MOVEMENTS_BST_FIX_END)
    # Ensure target columns are real datetime BEFORE the masked subtraction. read_csv
    # (compute_delay_changes) yields pyarrow/str-dtype columns that can't hold a
    # Timestamp on masked assignment (TypeError); converting the whole column is a
    # no-op when it's already datetime (load_movements uses parse_dates).
    for c in time_cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    ref = df[ref_col] if ref_col in time_cols else pd.to_datetime(df[ref_col], errors="coerce")
    mask = (ref >= start) & (ref < end)
    n = int(mask.sum())
    if n:
        delta = pd.Timedelta(hours=C.MOVEMENTS_BST_FIX_DELTA_H)
        for c in time_cols:
            if c in df.columns:
                df.loc[mask, c] = df.loc[mask, c] + delta
    return df


def load_movements(correct_bst: bool = True) -> pd.DataFrame:
    """Load Train Movements CSV (50 MB). Caches a RAW parquet on first call; the
    +1h BST correction (correct_movements_bst) is applied on every load so the
    cache stays raw and the fix is never double-applied. Pass correct_bst=False
    only for diagnostics that need the raw feed."""
    if C.MOVEMENTS_PARQUET.exists():
        df = pd.read_parquet(C.MOVEMENTS_PARQUET)
    else:
        df = pd.read_csv(
            C.MOVEMENTS_CSV,
            parse_dates=[
                "gbtt_timestamp", "planned_timestamp",
                "actual_timestamp", "msg_queue_timestamp",
            ],
            low_memory=False,
        )
        df.to_parquet(C.MOVEMENTS_PARQUET, index=False, compression="zstd")  # raw cache
    return correct_movements_bst(df) if correct_bst else df


# ===== route_to_tc_all =====

def load_route_to_tc() -> pd.DataFrame:
    """Load route_to_tc_all.csv with its track-list column parsed into Python lists."""
    df = pd.read_csv(C.ROUTE_TO_TC_CSV)
    df["track_list"] = df["track"].apply(
        lambda s: ast.literal_eval(s) if isinstance(s, str) else []
    )
    return df
