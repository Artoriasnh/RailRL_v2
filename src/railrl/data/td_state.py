"""P2.3 Iter 2.2 — As-of dynamic state queries on TD events.

Given any decision moment t, return the latest known state of every requested
Track / Signal / Route / TRTS asset. Powered by per-asset sorted timelines and
binary search; query time is O(log N_per_asset) per asset.

Used downstream by:
    Iter 2.3 — make_dynamic_snapshot(t, focal_signal): fills the spatial
               sub-graph from Iter 2.1 with state values at t.
    Iter 2.5 — per-window history aggregates (uses the same indices).

Memory: 12 M TD rows → ~250 MB parquet → ~150 MB of int64 + int8 numpy arrays
in the index. Loaded once per process; many state_at calls per load.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .. import config as C


# Strip the "S<prefix>" head from a TD signal id ("STD5044" → "5044"; numeric tails of
# real Derby signals don't collide across prefixes per P2.2 verification).
_SIGNAL_PREFIX_RE = re.compile(r"^S(DC|DW|DY|EC|TD)")


def _strip_signal_prefix(td_signal_id: str) -> str:
    if not isinstance(td_signal_id, str):
        return td_signal_id
    return _SIGNAL_PREFIX_RE.sub("", td_signal_id)


# ----------------- core data structure -----------------

# A timeline is a tuple of two parallel numpy arrays:
#   times_ns: int64  (sorted, ascending)
#   states:   int8   (the state value at each event time)
Timeline = tuple[np.ndarray, np.ndarray]


@dataclass
class TDStateView:
    """Loaded-once view of TD events for as-of state queries.

    Internal layout: 4 dicts keyed by canonical asset_id, each value is
    a pre-sorted (timestamps_ns, states) numpy-array pair.
    """
    tracks:  dict[str, Timeline] = field(default_factory=dict)
    signals: dict[str, Timeline] = field(default_factory=dict)
    routes:  dict[str, Timeline] = field(default_factory=dict)   # combined Route(state=0) + Panel_Request(state=1)
    trts:    dict[str, Timeline] = field(default_factory=dict)

    # ---------- builders ----------

    @classmethod
    def load(cls, td_df: Optional[pd.DataFrame] = None) -> "TDStateView":
        """Build the index. If td_df is None, read from C.TD_PARQUET (auto-creating it
        on first call); for tests pass a small synthetic frame."""
        if td_df is None:
            from ..data_io import load_td
            td_df = load_td(columns=["time", "type", "id", "state"])

        v = cls()
        v._build_from(td_df)
        return v

    def _build_from(self, df: pd.DataFrame) -> None:
        # Convert datetime → int64 ns once; needed for fast searchsorted later
        if not pd.api.types.is_datetime64_any_dtype(df["time"]):
            df = df.copy()
            df["time"] = pd.to_datetime(df["time"])
        df = df.assign(time_ns=df["time"].astype("int64"))

        # ---- Track ----
        tracks = df[df["type"] == "Track"][["id", "time_ns", "state"]]
        self.tracks = self._index_by_id(tracks, id_col="id")

        # ---- Signal ----
        sigs = df[df["type"] == "Signal"][["id", "time_ns", "state"]].copy()
        sigs["id"] = sigs["id"].astype(str).map(_strip_signal_prefix)
        self.signals = self._index_by_id(sigs, id_col="id")

        # ---- Route + Panel_Request ----
        # Both touch route_id and toggle state. Concatenate so a single timeline
        # tells us "is route X locked at time t?" regardless of which event fired.
        rt = df[df["type"].isin(["Route", "Panel_Request"])][["id", "time_ns", "state"]]
        self.routes = self._index_by_id(rt, id_col="id")

        # ---- TRTS ----
        trts = df[df["type"] == "TRTS"][["id", "time_ns", "state"]]
        self.trts = self._index_by_id(trts, id_col="id")

    @staticmethod
    def _index_by_id(events: pd.DataFrame, *, id_col: str) -> dict[str, Timeline]:
        """Group events by asset id; per group store sorted (times_ns, states)."""
        out: dict[str, Timeline] = {}
        if events.empty:
            return out
        events = events.sort_values([id_col, "time_ns"], kind="stable")
        # vectorised split via groupby; values are NumPy views (no copy)
        for aid, sub in events.groupby(id_col, sort=False):
            t = sub["time_ns"].to_numpy(dtype=np.int64, copy=True)
            s = sub["state"].to_numpy(dtype=np.int8, copy=True)
            out[str(aid)] = (t, s)
        return out

    # ---------- queries ----------

    @staticmethod
    def _asof(timeline: Optional[Timeline], t_ns: int) -> Optional[int]:
        """Latest state with time ≤ t_ns. None if no events at or before t."""
        if timeline is None:
            return None
        times, states = timeline
        if times.size == 0:
            return None
        pos = int(np.searchsorted(times, t_ns, side="right"))
        if pos == 0:
            return None
        return int(states[pos - 1])

    @staticmethod
    def _to_ns(t) -> int:
        return int(pd.Timestamp(t).value)

    def track_state_at(self, tc_id: str, t) -> Optional[int]:
        return self._asof(self.tracks.get(tc_id), self._to_ns(t))

    def signal_state_at(self, signal_id: str, t) -> Optional[int]:
        return self._asof(self.signals.get(signal_id), self._to_ns(t))

    def route_state_at(self, route_id: str, t) -> Optional[int]:
        return self._asof(self.routes.get(route_id), self._to_ns(t))

    def trts_state_at(self, trts_id: str, t) -> Optional[int]:
        return self._asof(self.trts.get(trts_id), self._to_ns(t))

    def state_at(
        self, t,
        track_ids: Optional[list[str]] = None,
        signal_ids: Optional[list[str]] = None,
        route_ids: Optional[list[str]] = None,
        trts_ids: Optional[list[str]] = None,
    ) -> dict[str, dict[str, Optional[int]]]:
        """Bulk query at time t. If a *_ids list is None → query ALL known assets
        of that type (don't do this in production — pass a filter).
        Returns {'tracks': {tc: state}, 'signals': {...}, 'routes': {...}, 'trts': {...}}."""
        t_ns = self._to_ns(t)

        def _bulk(idx: dict, ids: Optional[list[str]]):
            if ids is None:
                return {k: self._asof(v, t_ns) for k, v in idx.items()}
            return {a: self._asof(idx.get(a), t_ns) for a in ids}

        return {
            "tracks":  _bulk(self.tracks,  track_ids),
            "signals": _bulk(self.signals, signal_ids),
            "routes":  _bulk(self.routes,  route_ids),
            "trts":    _bulk(self.trts,    trts_ids),
        }

    # ---------- introspection ----------

    def summary(self) -> dict:
        return {
            "n_tracks_indexed":  len(self.tracks),
            "n_signals_indexed": len(self.signals),
            "n_routes_indexed":  len(self.routes),
            "n_trts_indexed":    len(self.trts),
            "track_events_total":  int(sum(t[0].size for t in self.tracks.values())),
            "signal_events_total": int(sum(t[0].size for t in self.signals.values())),
            "route_events_total":  int(sum(t[0].size for t in self.routes.values())),
            "trts_events_total":   int(sum(t[0].size for t in self.trts.values())),
        }
