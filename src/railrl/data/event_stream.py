"""P2.3 Iter 2.A — Event token stream from TD `change` column.

Architectural insight (May 2026 redesign):
    The TD `change` column is the canonical event encoding of the entire
    Derby signalling system. Each row is `[(asset_idx, new_state)]` where
    asset_idx ∈ [0, 672] decodes via derby_info_mapping.csv into a
    Track / Signal / Route / TRTS asset.

    By tokenising the stream, the model receives the FULL sequence of
    historical events (not aggregates), which captures temporal causality
    that aggregate statistics destroy.

This module provides:

    AssetIndex          ↔   asset_idx ⇄ asset_name + asset_type
    EventTokenStream    →   load TD events, query last K tokens before t

The K = 256 default (~17 min at typical Derby event density) is tuned to
match Transformer encoder context length (a power of 2).
"""
from __future__ import annotations
import re
import time as _time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .. import config as C


# ============================================================
# AssetIndex — bijection between asset_idx (int) and asset name
# ============================================================

# Asset-type classification by name prefix (mapping CSV uses the full SOP names)
_TYPE_PATTERNS = [
    ("Signal",  re.compile(r"^S(DC|DW|DY|EC|TD)\d+\w*$")),
    ("Route",   re.compile(r"^R(DW|TD|DC|EC|DY)\d+\w*[A-Z]+(?:-\d+)?\((M|C|S|W|PS|SP)\)$")),
    ("TRTS",    re.compile(r"^LPLAT\d+[AB]TRS\([NS]\)$")),
    ("Track",   re.compile(r"^T[A-Z0-9]+$")),
]


def _classify_asset(name: str) -> str:
    if not isinstance(name, str):
        return "Unknown"
    for type_name, pat in _TYPE_PATTERNS:
        if pat.match(name):
            return type_name
    return "Unknown"


@dataclass
class AssetIndex:
    """Forward / reverse lookup over the 672-asset universe."""
    by_idx:  dict[int, str]    = field(default_factory=dict)   # 0 → "SDC5061"
    by_name: dict[str, int]    = field(default_factory=dict)   # "SDC5061" → 0
    type_of: dict[int, str]    = field(default_factory=dict)   # 0 → "Signal"
    df: Optional[pd.DataFrame] = None                          # full dataframe for inspection

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "AssetIndex":
        path = path or C.DERBY_INFO_MAPPING_CSV
        # encoding='utf-8-sig' transparently strips a BOM if present
        df = pd.read_csv(path, encoding="utf-8-sig")
        df.columns = [c.strip() for c in df.columns]
        df["key"]   = df["key"].astype(int)
        df["value"] = df["value"].astype(str).str.strip()
        df["type"]  = df["value"].apply(_classify_asset)
        ai = cls(
            # IMPORTANT: derby_info_mapping.csv uses 0-indexed `key` (0..671)
            # but the TD `change` column emits 1-indexed asset_idx (1..672).
            # The TD feed is the ground truth, so we shift +1 here. Verified
            # on all 638 assets: TD-idx == CSV-key + 1 universally.
            by_idx ={int(r["key"]) + 1: r["value"] for _, r in df.iterrows()},
            by_name={r["value"]: int(r["key"]) + 1 for _, r in df.iterrows()},
            type_of={int(r["key"]) + 1: r["type"]  for _, r in df.iterrows()},
            df     =df,
        )
        return ai

    # ----- query helpers -----

    def name(self, idx: int) -> Optional[str]:
        return self.by_idx.get(int(idx))

    def idx(self, name: str) -> Optional[int]:
        return self.by_name.get(name)

    def asset_type(self, idx: int) -> str:
        return self.type_of.get(int(idx), "Unknown")

    def indices_of_type(self, t: str) -> list[int]:
        return [i for i, kind in self.type_of.items() if kind == t]

    def summary(self) -> dict:
        if self.df is None:
            return {}
        return {
            "n_assets": int(len(self.df)),
            "by_type":  self.df["type"].value_counts().to_dict(),
        }


# ============================================================
# EventTokenStream — full TD event log indexed for K-tail queries
# ============================================================

# Structure-of-arrays packing: 12 M events × (int16, int8, int64) ≈ 130 MB
@dataclass
class EventTokenStream:
    asset_idx: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int16))
    state:     np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int8))
    time_ns:   np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))

    # Per-asset event index, built lazily on first asset_filter query.
    # events_by_asset[a] = np.ndarray of (sorted, ascending) positions into
    # self.asset_idx where the event has asset_idx == a.
    _events_by_asset: Optional[dict[int, np.ndarray]] = field(default=None, repr=False)

    @property
    def n_tokens(self) -> int:
        return int(self.time_ns.size)

    # ---------- per-asset index (built lazily) ----------

    def _build_per_asset_index(self) -> dict[int, np.ndarray]:
        """Group event positions by asset_idx.

        Cost: one stable argsort over the full stream (≈ O(N log N), a few
        seconds for 10 M events), done once. After that every
        last_k_before(asset_filter=...) call is O(|F|·(logN + K)) — independent
        of the total stream size, so the per-snapshot cost no longer scales
        with TD size.
        """
        if self._events_by_asset is not None:
            return self._events_by_asset

        N = int(self.asset_idx.size)
        if N == 0:
            self._events_by_asset = {}
            return self._events_by_asset

        # Stable argsort by asset_idx → equal keys preserve their input order,
        # which here is time-sorted, so each per-asset slice is automatically
        # ascending (= ascending in time too).
        order = np.argsort(self.asset_idx, kind="stable")
        sorted_assets = self.asset_idx[order]

        # Find group boundaries
        change_points = np.concatenate((
            [0],
            np.flatnonzero(np.diff(sorted_assets)) + 1,
            [N],
        ))
        result: dict[int, np.ndarray] = {}
        for s, e in zip(change_points[:-1], change_points[1:]):
            a = int(sorted_assets[s])
            # Cast positions to int32 — fits 2.1 B values, half the memory of int64
            result[a] = order[s:e].astype(np.int32, copy=False)

        self._events_by_asset = result
        return result

    # ---------- builders ----------

    @classmethod
    def from_td_dataframe(cls, td_df: pd.DataFrame) -> "EventTokenStream":
        """Build from a TD DataFrame with 'time' and 'change' columns.

        Keeps only the rows where `change` parses to a single (idx, state) pair.
        Skips berth events (CA/CB/CC) which all have change='[(0, 0)]'.
        """
        if "time" not in td_df.columns or "change" not in td_df.columns:
            raise ValueError("td_df must have columns 'time' and 'change'")

        # Vectorised parse: change is text like '[(444, 1)]'.
        # We extract (idx, state) using a regex.
        pat = re.compile(r"^\[\((\d+),\s*(\d+)\)\]$")
        # Many CA/CB/CC rows have '[(0, 0)]'; we filter them out by also requiring
        # the row's `type` ∈ {Track, Signal, Route, Panel_Request, TRTS} when available.
        mask = pd.Series(True, index=td_df.index)
        if "type" in td_df.columns:
            keep_types = {"Track", "Signal", "Route", "Panel_Request", "TRTS"}
            mask = td_df["type"].isin(keep_types)

        sub = td_df.loc[mask, ["time", "change"]].copy()
        ext = sub["change"].astype(str).str.extract(pat)
        ext.columns = ["idx", "state"]
        valid = ext.dropna()
        sub = sub.loc[valid.index]

        # Pack into arrays
        idx_arr   = valid["idx"].astype(np.int32).to_numpy().astype(np.int16)   # values 0..671 fit safely
        state_arr = valid["state"].astype(np.int32).to_numpy().astype(np.int8)
        # Force NS resolution — parquet may store as us; astype('int64') alone
        # would then produce microseconds, mismatching pd.Timestamp(t).value (ns).
        sub["time"] = pd.to_datetime(sub["time"]).astype("datetime64[ns]")
        time_arr   = sub["time"].astype("int64").to_numpy()

        # Sort by time once — guarantees stable as-of slicing
        order = np.argsort(time_arr, kind="stable")
        return cls(
            asset_idx=idx_arr[order],
            state    =state_arr[order],
            time_ns  =time_arr[order],
        )

    @classmethod
    def load(cls, td_df: Optional[pd.DataFrame] = None) -> "EventTokenStream":
        """Load from cache parquet if available, else build from TD via data_io."""
        if C.EVENT_STREAM_PARQUET.exists() and td_df is None:
            df = pd.read_parquet(C.EVENT_STREAM_PARQUET)
            return cls(
                asset_idx=df["asset_idx"].values.astype(np.int16),
                state    =df["state"].values.astype(np.int8),
                time_ns  =df["time_ns"].values.astype(np.int64),
            )
        if td_df is None:
            from ..data_io import load_td
            td_df = load_td(columns=["time", "type", "change"])
        return cls.from_td_dataframe(td_df)

    def to_parquet(self, path: Optional[Path] = None) -> Path:
        path = path or C.EVENT_STREAM_PARQUET
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "time_ns":   self.time_ns,
            "asset_idx": self.asset_idx,
            "state":     self.state,
        }).to_parquet(path, index=False, compression="zstd")
        return path

    # ---------- queries ----------

    def last_k_before(self, t, K: int = C.EVENT_TOKEN_K,
                       *, asset_filter: Optional[set[int]] = None) -> dict:
        """Return up to K most-recent (asset_idx, state, time_ns) tokens with
        time_ns ≤ t. The result preserves chronological order (oldest → newest).

        If `asset_filter` is given, only events whose asset_idx is in the set
        are eligible (e.g. only TCs in the focal subgraph).
        """
        t_ns = int(pd.Timestamp(t).value)
        upper = int(np.searchsorted(self.time_ns, t_ns, side="right"))

        if asset_filter is None:
            lo = max(0, upper - K)
            return {
                "asset_idx": self.asset_idx[lo:upper].copy(),
                "state":     self.state[lo:upper].copy(),
                "time_ns":   self.time_ns[lo:upper].copy(),
                "n":         upper - lo,
                "padding":   K - (upper - lo),
            }

        # Per-asset index path — cost O(|F|·(logN + K)), no full prefix scans.
        ev_by = self._build_per_asset_index()
        slices: list[np.ndarray] = []
        for a in asset_filter:
            arr = ev_by.get(int(a))
            if arr is None or arr.size == 0:
                continue
            # Rightmost position in arr that is < upper
            j = int(np.searchsorted(arr, upper, side="left"))
            if j == 0:
                continue
            # Take the last min(K, j) candidates — already ascending
            slices.append(arr[max(0, j - K):j])

        if not slices:
            return {
                "asset_idx": np.empty(0, dtype=np.int16),
                "state":     np.empty(0, dtype=np.int8),
                "time_ns":   np.empty(0, dtype=np.int64),
                "n":         0,
                "padding":   K,
            }

        merged = np.concatenate(slices)
        # Each slice is sorted but the union isn't — sort then take last K
        merged.sort(kind="quicksort")
        if merged.size > K:
            merged = merged[-K:]
        abs_idx = merged.astype(np.int64, copy=False)

        n = int(abs_idx.size)
        return {
            "asset_idx": self.asset_idx[abs_idx].copy(),
            "state":     self.state[abs_idx].copy(),
            "time_ns":   self.time_ns[abs_idx].copy(),
            "n":         n,
            "padding":   K - n,
        }

    # ---------- introspection ----------

    def summary(self) -> dict:
        return {
            "n_tokens": self.n_tokens,
            "min_time": str(pd.Timestamp(int(self.time_ns.min())))   if self.n_tokens else None,
            "max_time": str(pd.Timestamp(int(self.time_ns.max())))   if self.n_tokens else None,
            "unique_assets": int(np.unique(self.asset_idx).size),
        }
