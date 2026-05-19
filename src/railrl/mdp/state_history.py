"""spec 02 §4 — Time-aware history lookups for snapshot construction.

Provides per-asset timeline queries to power:
  - per-window occupancy / aspect aggregates (1, 5, 10, 15, 30 min)
  - K=256 event token slicing
  - schedule_outlook from Movements.gbtt
  - dynamic edges (at_berth, next_signal)
  - multi-train state_nodes_train (other active trains in subgraph)

Design:
  Each "history" loads the relevant TD/Movements rows ONCE at __init__,
  groups them per asset, and sorts ascending by time_ns. Query methods
  use np.searchsorted for O(log n) lookup. This keeps build_snapshot
  fast enough to scale to 2M decision points.

Leak contract (spec 01 §17.5):
  Only events with time_ns < t_ns are visible. All queries clip the
  returned data with binary search. Any query that would expose
  future information returns the safe default (None / 0 / False).
"""
from __future__ import annotations
import bisect
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


# ============================================================
# Constants
# ============================================================

_NS_PER_S = 1_000_000_000


# ============================================================
# Core: per-asset binary-state timeline
# ============================================================

@dataclass
class _StateTimeline:
    """Sorted (time_ns, state ∈ {0,1}, train_id) for one asset."""
    times_ns: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    states:   np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int8))
    train_ids: list[str] = field(default_factory=list)  # parallel to times_ns

    def __post_init__(self):
        # Ensure dtype + sortedness
        if len(self.times_ns) > 0:
            order = np.argsort(self.times_ns, kind="stable")
            self.times_ns = self.times_ns[order]
            self.states   = self.states[order]
            self.train_ids = [self.train_ids[i] for i in order]

    def state_at(self, t_ns: int) -> int:
        """Return the most recent state at or before t_ns (0 if no prior event)."""
        if len(self.times_ns) == 0:
            return 0
        # rightmost index with times_ns <= t_ns
        idx = np.searchsorted(self.times_ns, t_ns, side="right") - 1
        if idx < 0:
            return 0
        return int(self.states[idx])

    def occupier_at(self, t_ns: int) -> Optional[str]:
        """Return train_id of the most recent state=1 transition, if currently
        in state=1 at t_ns. None otherwise.
        """
        if len(self.times_ns) == 0:
            return None
        idx = np.searchsorted(self.times_ns, t_ns, side="right") - 1
        if idx < 0 or int(self.states[idx]) != 1:
            return None
        tr = self.train_ids[idx]
        if not tr or tr in ("0", "00", "000", "0000"):
            return None
        return tr

    def last_change_age_s(self, t_ns: int) -> int:
        """Seconds since last state transition at or before t_ns.

        Returns 0 if no prior event (treat as "always been in default state").
        """
        if len(self.times_ns) == 0:
            return 0
        idx = np.searchsorted(self.times_ns, t_ns, side="right") - 1
        if idx < 0:
            return 0
        delta_ns = int(t_ns - self.times_ns[idx])
        return max(0, delta_ns // _NS_PER_S)

    def window_stats(self, t_ns: int, window_s: float) -> tuple[float, int]:
        """Compute (fraction_in_state_1, n_state_changes) over [t_ns - W, t_ns].

        Returns
        -------
        fraction_in_state_1 : float in [0, 1]
            Time-weighted fraction of the window during which state==1.
        n_state_changes : int
            Count of transitions (state[i-1] != state[i]) strictly inside the
            window (i.e., t_ns - W < event_time <= t_ns).
        """
        if window_s <= 0 or len(self.times_ns) == 0:
            return 0.0, 0
        t_start_ns = t_ns - int(window_s * _NS_PER_S)
        # State at window start
        start_state = self.state_at(t_start_ns)
        # Events strictly inside (t_start, t_end]
        lo = np.searchsorted(self.times_ns, t_start_ns, side="right")
        hi = np.searchsorted(self.times_ns, t_ns, side="right")
        if lo >= hi:
            # No transitions in window
            return float(start_state), 0
        # Walk through transitions and accumulate dwell in state 1
        prev_t = t_start_ns
        prev_s = start_state
        dwell_1 = 0
        n_changes = 0
        for i in range(lo, hi):
            cur_t = int(self.times_ns[i])
            cur_s = int(self.states[i])
            if cur_s != prev_s:
                n_changes += 1
            if prev_s == 1:
                dwell_1 += cur_t - prev_t
            prev_t = cur_t
            prev_s = cur_s
        # Tail segment [last_event, t_ns]
        if prev_s == 1:
            dwell_1 += t_ns - prev_t
        window_ns = t_ns - t_start_ns
        frac = dwell_1 / window_ns if window_ns > 0 else 0.0
        return float(max(0.0, min(1.0, frac))), int(n_changes)


# ============================================================
# Track occupancy
# ============================================================

@dataclass
class TrackOccupancyHistory:
    """Per-track binary occupancy timeline (state 1 = occupied)."""
    timelines: dict[str, _StateTimeline] = field(default_factory=dict)

    @classmethod
    def build(cls, td_events: pd.DataFrame) -> "TrackOccupancyHistory":
        """Group td_events[type=='Track'] by `id` and build timelines."""
        df = td_events[td_events["type"] == "Track"]
        if df.empty:
            return cls()
        tls: dict[str, _StateTimeline] = {}
        for tc_id, sub in df.groupby("id", observed=True, sort=False):
            tc_str = str(tc_id)
            times_ns = sub["time"].astype("int64").to_numpy()
            states = sub["state"].fillna(0).astype("int8").to_numpy()
            tr_ids = sub["trainid_filled"].astype(str).tolist()
            tls[tc_str] = _StateTimeline(times_ns=times_ns, states=states,
                                          train_ids=tr_ids)
        return cls(timelines=tls)

    def _tl(self, tc_id: str) -> _StateTimeline:
        return self.timelines.get(str(tc_id), _StateTimeline())

    def occupied_now(self, tc_id: str, t_ns: int) -> bool:
        return self._tl(tc_id).state_at(t_ns) == 1

    def current_occupier(self, tc_id: str, t_ns: int) -> Optional[str]:
        occ = self._tl(tc_id).occupier_at(t_ns)
        return None if occ in (None, 0) else occ

    def window_stats(self, tc_id: str, t_ns: int, window_s: float) -> tuple[float, int]:
        return self._tl(tc_id).window_stats(t_ns, window_s)

    def last_change_age_s(self, tc_id: str, t_ns: int) -> int:
        return self._tl(tc_id).last_change_age_s(t_ns)


# ============================================================
# Signal aspect
# ============================================================

@dataclass
class SignalAspectHistory:
    """Per-signal binary aspect timeline (state 1 = restrictive/red).

    Keys are bare signal_id strings (e.g., '5044') matching static graph,
    not the TD 'STD5044' full_name. Build accepts an optional
    full_name → signal_id mapping from the signal node table.
    """
    timelines: dict[str, _StateTimeline] = field(default_factory=dict)

    @classmethod
    def build(cls, td_events: pd.DataFrame,
              full_name_to_id: Optional[dict[str, str]] = None
              ) -> "SignalAspectHistory":
        """Group td_events[type=='Signal'] by signal_id.

        Args:
            td_events: TD log.
            full_name_to_id: optional mapping STD5044 -> 5044. If absent,
                we infer by stripping the leading 'S' + 2-char prefix
                (works for STD/SDC).
        """
        df = td_events[td_events["type"] == "Signal"]
        if df.empty:
            return cls()
        tls: dict[str, _StateTimeline] = {}
        fn_to_id = full_name_to_id or {}
        for full_name, sub in df.groupby("id", observed=True, sort=False):
            fn_str = str(full_name)
            sig_id = fn_to_id.get(fn_str)
            if sig_id is None:
                # heuristic: strip leading 'S' + 2-letter prefix
                if fn_str.startswith("S") and len(fn_str) >= 4:
                    sig_id = fn_str[3:]
                else:
                    sig_id = fn_str
            times_ns = sub["time"].astype("int64").to_numpy()
            states = sub["state"].fillna(0).astype("int8").to_numpy()
            tr_ids = sub["trainid_filled"].astype(str).tolist()
            # If duplicate signal_id (multiple prefixes), merge sorted
            if sig_id in tls:
                old = tls[sig_id]
                combined_t = np.concatenate([old.times_ns, times_ns])
                combined_s = np.concatenate([old.states, states])
                combined_tr = old.train_ids + tr_ids
                tls[sig_id] = _StateTimeline(combined_t, combined_s, combined_tr)
            else:
                tls[sig_id] = _StateTimeline(times_ns=times_ns, states=states,
                                              train_ids=tr_ids)
        return cls(timelines=tls)

    def _tl(self, signal_id: str) -> _StateTimeline:
        return self.timelines.get(str(signal_id), _StateTimeline())

    def aspect_restrictive_now(self, signal_id: str, t_ns: int) -> bool:
        return self._tl(signal_id).state_at(t_ns) == 1

    def window_stats(self, signal_id: str, t_ns: int, window_s: float) -> tuple[float, int]:
        """Returns (fraction_red, n_aspect_changes)."""
        return self._tl(signal_id).window_stats(t_ns, window_s)

    def last_change_age_s(self, signal_id: str, t_ns: int) -> int:
        return self._tl(signal_id).last_change_age_s(t_ns)


# ============================================================
# Berth history (per-train) — for current_berth + recent_panel_requests
# ============================================================

@dataclass
class BerthHistory:
    """Per-train berth timeline + panel request log.

    Used for:
      - `current_berth_train_id` on signal nodes (which train sits at berth)
      - `recent_panel_requests_count` on train nodes (PRs in last 5 min)
    """
    # Per-berth: list of (time_ns, train_id, descr) — most recent occupant
    berth_to_events: dict[str, list[tuple[int, str]]] = field(default_factory=dict)
    # Per-train: sorted PR timestamps
    train_to_pr_times: dict[str, np.ndarray] = field(default_factory=dict)

    @classmethod
    def build(cls, td_events: pd.DataFrame) -> "BerthHistory":
        # Berth occupants: any row with non-null to_berth → (time, train_id, berth)
        df_berth = td_events.dropna(subset=["to_berth"])
        b2e: dict[str, list[tuple[int, str]]] = {}
        for berth, sub in df_berth.groupby("to_berth", observed=True, sort=False):
            berth_s = str(berth)
            ev = list(zip(sub["time"].astype("int64").tolist(),
                          sub["trainid_filled"].astype(str).tolist()))
            ev.sort(key=lambda x: x[0])
            b2e[berth_s] = ev

        # Panel requests: type == 'Panel_Request' rows
        df_pr = td_events[td_events["type"] == "Panel_Request"]
        t2pr: dict[str, np.ndarray] = {}
        if not df_pr.empty:
            for train_id, sub in df_pr.groupby("trainid_filled", observed=True, sort=False):
                tr_s = str(train_id)
                if not tr_s or tr_s in ("0", "00", "000", "0000", "None", "nan"):
                    continue
                arr = np.sort(sub["time"].astype("int64").to_numpy())
                t2pr[tr_s] = arr
        return cls(berth_to_events=b2e, train_to_pr_times=t2pr)

    def berth_occupant_at(self, berth: str, t_ns: int) -> tuple[Optional[str], int]:
        """Return (train_id, age_s) — latest train into `berth` at or before t_ns.

        Returns (None, 0) if no prior occupant.
        """
        ev = self.berth_to_events.get(str(berth), [])
        if not ev:
            return None, 0
        times = [e[0] for e in ev]
        idx = bisect.bisect_right(times, t_ns) - 1
        if idx < 0:
            return None, 0
        t, tr = ev[idx]
        if not tr or tr in ("0", "00", "000", "0000", "None", "nan"):
            return None, 0
        age_s = max(0, (t_ns - t) // _NS_PER_S)
        return tr, int(age_s)

    def recent_pr_count(self, train_id: str, t_ns: int, window_s: float) -> int:
        arr = self.train_to_pr_times.get(str(train_id))
        if arr is None or len(arr) == 0:
            return 0
        t_start = t_ns - int(window_s * _NS_PER_S)
        lo = np.searchsorted(arr, t_start, side="right")
        hi = np.searchsorted(arr, t_ns,    side="right")
        return int(max(0, hi - lo))


# ============================================================
# Movements lookup — schedule_outlook + planned_platform
# ============================================================

@dataclass
class MovementsLookup:
    """Per-train gbtt schedule for upcoming arrivals/departures.

    Spec 02 §4.9: schedule_outlook returns top-K=5 upcoming trains in
    `SCHEDULE_LOOKAHEAD_MIN` minutes ahead, using gbtt only (NO actual).
    Per-row leak-safe: drops actual_timestamp, never returns signal IDs.
    """
    # Per-train: sorted list of (gbtt_ns, planned_platform_int_or_none, event_type)
    train_to_schedule: dict[str, list[tuple[int, Optional[int], str]]] = field(default_factory=dict)
    # All upcoming events globally, sorted by gbtt_ns: list of (gbtt_ns, train_id, platform, event_type)
    all_events: list[tuple[int, str, Optional[int], str]] = field(default_factory=list)

    @classmethod
    def build(cls, movements: pd.DataFrame) -> "MovementsLookup":
        if movements is None or movements.empty:
            return cls()
        df = movements.copy()
        # Required columns
        for col in ("gbtt_timestamp", "current_train_id", "platform", "event_type"):
            if col not in df.columns:
                return cls()  # malformed schema → empty lookup
        df = df.dropna(subset=["gbtt_timestamp", "current_train_id"])
        if df.empty:
            return cls()
        # Cast gbtt to int64 ns
        gbtt = pd.to_datetime(df["gbtt_timestamp"], errors="coerce")
        df = df[gbtt.notna()].copy()
        df["__gbtt_ns"] = gbtt.dropna().astype("int64").to_numpy()
        # Parse platform → int 1-6 or None
        def _parse_plat(v):
            if pd.isna(v):
                return None
            try:
                iv = int(float(v))
                if 1 <= iv <= 6:
                    return iv
            except (TypeError, ValueError):
                pass
            return None
        # NOTE: use list comprehension (NOT df.apply().tolist()) — pandas
        # auto-coerces mixed int/None series to float64 dtype, turning
        # `int(3) | None` into `3.0 | nan`. We need true ints + None.
        platforms = [_parse_plat(v) for v in df["platform"]]
        events = df["event_type"].astype(str).tolist()
        trains = df["current_train_id"].astype(str).tolist()
        gbtt_ns = df["__gbtt_ns"].tolist()

        # Per-train index
        t2s: dict[str, list[tuple[int, Optional[int], str]]] = {}
        all_ev = []
        for g, tr, p, e in zip(gbtt_ns, trains, platforms, events):
            tr_s = str(tr).strip()
            if not tr_s or tr_s in ("nan", "None", "0"):
                continue
            # Defensive: ensure p is `None` or pure `int` (never numpy/float/nan).
            # pandas series iteration can yield numpy.int64 + None which the
            # final `Series.tolist()` would coerce to float64; storing as plain
            # int here keeps the leak audit Check 4 (planned_platform must be
            # int 1-6 or None) happy.
            if p is None or (isinstance(p, float) and np.isnan(p)):
                p_norm: Optional[int] = None
            else:
                try:
                    p_norm = int(p)
                    if not (1 <= p_norm <= 6):
                        p_norm = None
                except (TypeError, ValueError):
                    p_norm = None
            t2s.setdefault(tr_s, []).append((int(g), p_norm, e))
            all_ev.append((int(g), tr_s, p_norm, e))
        # Sort
        for tr_s in t2s:
            t2s[tr_s].sort(key=lambda x: x[0])
        all_ev.sort(key=lambda x: x[0])
        return cls(train_to_schedule=t2s, all_events=all_ev)

    def schedule_outlook(self, t_ns: int, k: int = 5, lookahead_s: float = 900.0,
                          exclude_train: Optional[str] = None) -> list[dict]:
        """Return up to k upcoming train events in (t_ns, t_ns + lookahead_s].

        Leak-safe: uses gbtt only, returns planned_platform as int 1-6 or None
        (never a signal ID).
        """
        if not self.all_events:
            return []
        t_end_ns = t_ns + int(lookahead_s * _NS_PER_S)
        times = [e[0] for e in self.all_events]
        lo = bisect.bisect_right(times, t_ns)
        hi = bisect.bisect_right(times, t_end_ns)
        out = []
        for i in range(lo, hi):
            g, tr, p, e = self.all_events[i]
            if exclude_train is not None and tr == str(exclude_train):
                continue
            out.append({
                "train_id":         tr,
                "gbtt_delta_s":     max(0, (g - t_ns) // _NS_PER_S),
                "planned_platform": p,                # int 1-6 or None — NEVER signal_id
                "event_type":       e,
            })
            if len(out) >= k:
                break
        return out

    def planned_platform(self, train_id: str, t_ns: int) -> Optional[int]:
        """Most recent (or imminent) planned_platform from gbtt schedule.

        Picks the schedule entry closest in time to t_ns. Returns None if
        no entry has a valid platform.
        """
        sched = self.train_to_schedule.get(str(train_id), [])
        if not sched:
            return None
        # Pick nearest entry
        times = [e[0] for e in sched]
        idx = bisect.bisect_right(times, t_ns)
        # Look at idx (next) and idx-1 (prev), pick nearest
        candidates = []
        if idx < len(sched):
            candidates.append(sched[idx])
        if idx > 0:
            candidates.append(sched[idx - 1])
        for g, p, e in candidates:
            if p is not None:
                return p
        return None

    def scheduled_delta_s(self, train_id: str, t_ns: int) -> Optional[int]:
        """Seconds until the next gbtt event for this train (or None)."""
        sched = self.train_to_schedule.get(str(train_id), [])
        if not sched:
            return None
        times = [e[0] for e in sched]
        idx = bisect.bisect_right(times, t_ns)
        if idx >= len(sched):
            return None
        return int((sched[idx][0] - t_ns) // _NS_PER_S)


# ============================================================
# Event-token slicing for state_event_tokens (K=256)
# ============================================================

@dataclass
class EventTokenStream:
    """Last-K=256 events with time_ns <= t_ns, restricted to subgraph assets.

    Per spec 02 §4.7 each token is:
        (asset_idx, state ∈ {0, 1}, time_delta_s ∈ [0, ∞))
    The model encoder consumes a fixed K-length sequence (padded with
    sentinel tokens if fewer than K events available).
    """
    # Per-asset (track or signal id) → sorted (time_ns, state) array
    per_asset_times:  dict[str, np.ndarray] = field(default_factory=dict)
    per_asset_states: dict[str, np.ndarray] = field(default_factory=dict)

    @classmethod
    def build(cls, td_events: pd.DataFrame) -> "EventTokenStream":
        df = td_events[td_events["type"].isin(["Track", "Signal"])]
        if df.empty:
            return cls()
        per_t: dict[str, np.ndarray] = {}
        per_s: dict[str, np.ndarray] = {}
        for asset_id, sub in df.groupby("id", observed=True, sort=False):
            per_t[str(asset_id)] = sub["time"].astype("int64").to_numpy()
            per_s[str(asset_id)] = sub["state"].fillna(0).astype("int8").to_numpy()
        return cls(per_asset_times=per_t, per_asset_states=per_s)

    def slice_last_k(self, asset_keys: list[str], t_ns: int, k: int = 256
                      ) -> list[tuple[int, int, float]]:
        """Collect last events from `asset_keys` and return top-k by recency.

        Each tuple: (asset_idx, state, time_delta_s).
        asset_idx is the position in `asset_keys` (not a global asset_index).
        """
        candidates: list[tuple[int, int, int]] = []  # (time_ns, asset_idx, state)
        for idx, key in enumerate(asset_keys):
            times = self.per_asset_times.get(str(key))
            if times is None or len(times) == 0:
                continue
            states = self.per_asset_states[str(key)]
            # Events with time <= t_ns
            hi = np.searchsorted(times, t_ns, side="right")
            if hi == 0:
                continue
            # Take up to k from this asset (saves work; we'll trim globally)
            lo = max(0, hi - k)
            for i in range(hi - 1, lo - 1, -1):
                candidates.append((int(times[i]), idx, int(states[i])))
                if hi - i >= k:
                    break
        if not candidates:
            return []
        # Sort by time descending (most recent first)
        candidates.sort(key=lambda x: -x[0])
        out = []
        for tns, aidx, st in candidates[:k]:
            delta_s = max(0.0, (t_ns - tns) / _NS_PER_S)
            out.append((aidx, st, float(delta_s)))
        return out
