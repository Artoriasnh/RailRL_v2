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

# Platform id range (1-7; platform 7 = Derby pilot line, user domain knowledge
# 2026-05-20). Imported defensively from config.
try:
    from .. import config as _C
    _MIN_PLATFORM = getattr(_C, "MIN_PLATFORM_ID", 1)
    _MAX_PLATFORM = getattr(_C, "MAX_PLATFORM_ID", 7)
except Exception:  # pragma: no cover
    _MIN_PLATFORM, _MAX_PLATFORM = 1, 7

# Headcode is embedded in the 10-char TRUST train_id at chars [2:6]
# e.g. "851S49ME28" → "1S49", "771M99ML28" → "1M99". (user domain knowledge)
_HEADCODE_SLICE = slice(2, 6)


def _to_ns_int64(times) -> np.ndarray:
    """Convert a datetime64 Series/array to int64 NANOSECONDS, regardless of
    the source resolution.

    CRITICAL: td_data.parquet stores `time` as datetime64[**us**] (microseconds),
    so a plain `.astype("int64")` yields MICROSECONDS — but the decision time
    `t_ns` uses `pd.Timestamp.value` (NANOSECONDS). Mixing them makes every
    history time-query wrong (t_ns is ~1000× larger than all event times, so
    queries always return the last-ever event = a future leak). Forcing
    datetime64[ns] here guarantees both sides are in ns.
    """
    arr = times.values if hasattr(times, "values") else times
    return np.asarray(arr).astype("datetime64[ns]").astype("int64")


# ============================================================
# Core: per-asset binary-state timeline
# ============================================================

@dataclass
class _StateTimeline:
    """Sorted (time_ns, state ∈ {0,1}, train_id) for one asset."""
    times_ns: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    states:   np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int8))
    train_ids: list[str] = field(default_factory=list)  # parallel to times_ns

    # Prefix arrays for O(log n) window_stats (built in __post_init__)
    _cum_occ: np.ndarray = field(default=None, repr=False)  # len n: occupied ns from times[0]→times[i]
    _cum_chg: np.ndarray = field(default=None, repr=False)  # len n+1: cumulative state-changes

    def __post_init__(self):
        # Ensure dtype + sortedness
        if len(self.times_ns) > 0:
            order = np.argsort(self.times_ns, kind="stable")
            self.times_ns = self.times_ns[order]
            self.states   = self.states[order]
            self.train_ids = [self.train_ids[i] for i in order]
        self._build_prefix()

    def _build_prefix(self):
        """Precompute prefix sums so window_stats is O(log n), not O(window)."""
        n = len(self.times_ns)
        if n == 0:
            self._cum_occ = np.zeros(0, dtype=np.int64)
            self._cum_chg = np.zeros(1, dtype=np.int64)
            return
        t = self.times_ns.astype(np.int64)
        s = self.states.astype(np.int64)
        # occupied time in each segment [times[i], times[i+1]) == (s[i]==1)*dt
        if n >= 2:
            seg = (s[:-1] == 1).astype(np.int64) * np.diff(t)
            self._cum_occ = np.concatenate([[0], np.cumsum(seg)]).astype(np.int64)  # len n
        else:
            self._cum_occ = np.zeros(1, dtype=np.int64)
        # change indicator: chg[i] = s[i] != (s[i-1] if i>=1 else 0)
        prev = np.concatenate([[0], s[:-1]])
        chg = (s != prev).astype(np.int64)
        self._cum_chg = np.concatenate([[0], np.cumsum(chg)]).astype(np.int64)  # len n+1

    def _occupied_until(self, T: int) -> int:
        """Total state==1 dwell time (ns) from times[0] up to T."""
        idx = int(np.searchsorted(self.times_ns, T, side="right")) - 1
        if idx < 0:
            return 0
        base = int(self._cum_occ[idx])
        if int(self.states[idx]) == 1:
            base += int(T - self.times_ns[idx])
        return base

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
        window_ns = t_ns - t_start_ns
        # Vectorized dwell + change count via prefix arrays (O(log n)).
        # (Numerically identical to the old per-event loop — see test.)
        dwell_1 = self._occupied_until(t_ns) - self._occupied_until(t_start_ns)
        lo = int(np.searchsorted(self.times_ns, t_start_ns, side="right"))
        hi = int(np.searchsorted(self.times_ns, t_ns, side="right"))
        n_changes = int(self._cum_chg[hi] - self._cum_chg[lo])
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
            times_ns = _to_ns_int64(sub["time"])
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
            times_ns = _to_ns_int64(sub["time"])
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
            ev = list(zip(_to_ns_int64(sub["time"]).tolist(),
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
                arr = np.sort(_to_ns_int64(sub["time"]))
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

# Lateness occurrence window: only consider realized records within this much
# BEFORE t (avoids pulling a *previous day's* run for a reused headcode — same
# time-locality issue as the pass bug). 6h matches PASS_FALLBACK_GAP_S.
_LATENESS_WINDOW_NS = 6 * 3600 * _NS_PER_S


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
    # Precomputed sorted gbtt_ns array (built in __post_init__) so schedule_outlook
    # doesn't rebuild a 247k-element list on EVERY call (was ~4ms/snapshot).
    _all_times: np.ndarray = field(default=None, repr=False)
    # Per-headcode realized lateness: headcode -> (sorted actual_ns array, aligned
    # signed_lateness_s array). signed_lateness_s = timetable_variation(min)*60 *
    # sign(variation_status): LATE=+, EARLY=-, ON TIME/OFF ROUTE/other=0.
    # Consumed leak-safely by current_lateness_s() (only actual_ts <= t, within window).
    train_to_lateness: dict = field(default_factory=dict)

    def __post_init__(self):
        self._all_times = np.array([e[0] for e in self.all_events], dtype=np.int64)

    @classmethod
    def build(cls, movements: pd.DataFrame,
              train_id_col: str = "auto") -> "MovementsLookup":
        """Build the lookup, keyed by 4-char headcode (e.g. '1S49').

        Real Movements data has `current_train_id` ~99.9% empty; the headcode
        lives embedded in the 10-char TRUST `train_id` at chars [2:6]
        (e.g. '851S49ME28' → '1S49'). We extract that so schedule_outlook
        keys match TD focal_train headcodes.

        Args:
            movements: Movements DataFrame.
            train_id_col: which column to derive headcode from. "auto" picks
                `train_id` if present (TRUST id → slice [2:6]); else falls back
                to `current_train_id` used verbatim.
        """
        if movements is None or movements.empty:
            return cls()
        df = movements.copy()
        if "gbtt_timestamp" not in df.columns or "platform" not in df.columns \
                or "event_type" not in df.columns:
            return cls()  # malformed schema → empty lookup

        # --- Derive the headcode column ---
        use_col = train_id_col
        if use_col == "auto":
            use_col = "train_id" if "train_id" in df.columns else "current_train_id"
        if use_col not in df.columns:
            return cls()

        if use_col == "train_id":
            # TRUST id → headcode at [2:6]
            raw = df[use_col].astype("string")
            headcodes = raw.str.slice(_HEADCODE_SLICE.start, _HEADCODE_SLICE.stop)
        else:
            headcodes = df[use_col].astype("string")
        df["__headcode"] = headcodes

        # --- Realized lateness lookup (built BEFORE the gbtt filter, since lateness
        #     needs actual_timestamp + variation_status, gbtt optional) ---
        #     signed_lateness_s = timetable_variation(min)*60 * sign(variation_status).
        lateness: dict[str, tuple] = {}
        if {"actual_timestamp", "variation_status", "timetable_variation"} <= set(df.columns):
            a_all = pd.to_datetime(df["actual_timestamp"], errors="coerce")
            lmask = a_all.notna() & df["__headcode"].notna()
            if lmask.any():
                hc = df.loc[lmask, "__headcode"].astype(str).to_numpy()
                a_ns = a_all[lmask].astype("int64").to_numpy()
                status = df.loc[lmask, "variation_status"].astype(str).str.upper().str.strip().to_numpy()
                var_min = pd.to_numeric(df.loc[lmask, "timetable_variation"],
                                        errors="coerce").fillna(0).abs().to_numpy()
                sign = np.where(status == "LATE", 1.0,
                                np.where(status == "EARLY", -1.0, 0.0))
                signed_s = (var_min * 60.0 * sign).astype("int64")
                order = np.argsort(a_ns, kind="mergesort")     # stable global sort by actual_ns
                tmp: dict[str, tuple[list, list]] = {}
                for h, an, sv in zip(hc[order], a_ns[order], signed_s[order]):
                    if h not in tmp:
                        tmp[h] = ([], [])
                    tmp[h][0].append(int(an))
                    tmp[h][1].append(int(sv))
                lateness = {h: (np.array(al, dtype=np.int64), np.array(vl, dtype=np.int64))
                            for h, (al, vl) in tmp.items()}

        df = df.dropna(subset=["gbtt_timestamp", "__headcode"])
        if df.empty:
            return cls()
        # Cast gbtt to int64 ns
        gbtt = pd.to_datetime(df["gbtt_timestamp"], errors="coerce")
        df = df[gbtt.notna()].copy()
        df["__gbtt_ns"] = gbtt.dropna().astype("int64").to_numpy()

        # Parse platform → int MIN..MAX or None
        def _parse_plat(v):
            if pd.isna(v):
                return None
            try:
                iv = int(float(v))
                if _MIN_PLATFORM <= iv <= _MAX_PLATFORM:
                    return iv
            except (TypeError, ValueError):
                pass
            return None
        # NOTE: use list comprehension (NOT df.apply().tolist()) — pandas
        # auto-coerces mixed int/None series to float64 dtype, turning
        # `int(3) | None` into `3.0 | nan`. We need true ints + None.
        platforms = [_parse_plat(v) for v in df["platform"]]
        events = df["event_type"].astype(str).tolist()
        trains = df["__headcode"].astype(str).tolist()
        gbtt_ns = df["__gbtt_ns"].tolist()

        # Per-train (headcode) index
        t2s: dict[str, list[tuple[int, Optional[int], str]]] = {}
        all_ev = []
        for g, tr, p, e in zip(gbtt_ns, trains, platforms, events):
            tr_s = str(tr).strip()
            if not tr_s or tr_s in ("nan", "None", "0", "<NA>"):
                continue
            # Defensive: ensure p is `None` or pure `int` (never numpy/float/nan).
            # pandas series iteration can yield numpy.int64 + None which the
            # final `Series.tolist()` would coerce to float64; storing as plain
            # int here keeps leak audit Check 4 (planned_platform int 1-7 or
            # None) happy.
            if p is None or (isinstance(p, float) and np.isnan(p)):
                p_norm: Optional[int] = None
            else:
                try:
                    p_norm = int(p)
                    if not (_MIN_PLATFORM <= p_norm <= _MAX_PLATFORM):
                        p_norm = None
                except (TypeError, ValueError):
                    p_norm = None
            t2s.setdefault(tr_s, []).append((int(g), p_norm, e))
            all_ev.append((int(g), tr_s, p_norm, e))
        # Sort
        for tr_s in t2s:
            t2s[tr_s].sort(key=lambda x: x[0])
        all_ev.sort(key=lambda x: x[0])
        return cls(train_to_schedule=t2s, all_events=all_ev, train_to_lateness=lateness)

    def schedule_outlook(self, t_ns: int, k: int = 5, lookahead_s: float = 900.0,
                          exclude_train: Optional[str] = None) -> list[dict]:
        """Return up to k upcoming train events in (t_ns, t_ns + lookahead_s].

        Leak-safe: uses gbtt only, returns planned_platform as int 1-6 or None
        (never a signal ID).
        """
        if not self.all_events:
            return []
        t_end_ns = t_ns + int(lookahead_s * _NS_PER_S)
        # Use the precomputed sorted times array (np.searchsorted) instead of
        # rebuilding a 247k-element list every call.
        lo = int(np.searchsorted(self._all_times, t_ns, side="right"))
        hi = int(np.searchsorted(self._all_times, t_end_ns, side="right"))
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

    def current_lateness_s(self, headcode: str, t_ns: int,
                           window_ns: int = _LATENESS_WINDOW_NS) -> int:
        """Leak-safe CURRENT lateness in seconds (signed: + = late, − = early).

        = signed timetable_variation of the train's LATEST realized Movements
        record with `t − window_ns ≤ actual_timestamp ≤ t` (knowable to the
        signaller at t per spec 01 §13.1). 0 if no such record (unknown ≈ 0).
        Window restricts to the current occurrence (avoids a reused headcode's
        previous-day run).
        """
        lk = self.train_to_lateness.get(str(headcode))
        if not lk:
            return 0
        a_arr, v_arr = lk
        idx = int(np.searchsorted(a_arr, t_ns, side="right")) - 1
        if idx < 0 or a_arr[idx] < t_ns - window_ns:
            return 0
        return int(v_arr[idx])

    def scheduled_delta_s(self, train_id: str, t_ns: int) -> Optional[int]:
        """Current SIGNED lateness (sec, + = late) — see current_lateness_s.

        ⚠️ REDEFINED (Stage 4.7.2d lateness fix): was "seconds until NEXT gbtt
        event", which was always ≥0, matched the wrong (far) occurrence for reused
        headcodes (→ 276-day garbage), and never let f_late_train fire. Now uses
        realized timetable_variation ≤ t (leak-safe). f_late_train expects + = late.
        """
        return self.current_lateness_s(str(train_id), t_ns)


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
            per_t[str(asset_id)] = _to_ns_int64(sub["time"])
            per_s[str(asset_id)] = sub["state"].fillna(0).astype("int8").to_numpy()
        return cls(per_asset_times=per_t, per_asset_states=per_s)

    def slice_last_k(self, asset_keys: list[str], t_ns: int, k: int = 256
                      ) -> list[tuple[int, int, float]]:
        """Collect last events from `asset_keys` and return top-k by recency.

        Each tuple: (asset_idx, state, time_delta_s).
        asset_idx is the position in `asset_keys` (not a global asset_index).
        """
        # Vectorized: per asset take the last ≤k events (numpy slice, no Python
        # per-element loop), concatenate, then argpartition for global top-k.
        # The old per-element int() loop over ~25k candidates was ~9ms/call.
        c_times: list = []
        c_idx: list = []
        c_states: list = []
        for idx, key in enumerate(asset_keys):
            times = self.per_asset_times.get(str(key))
            if times is None or len(times) == 0:
                continue
            hi = int(np.searchsorted(times, t_ns, side="right"))
            if hi == 0:
                continue
            lo = max(0, hi - k)
            c_times.append(times[lo:hi])
            c_states.append(self.per_asset_states[str(key)][lo:hi])
            c_idx.append(np.full(hi - lo, idx, dtype=np.int32))
        if not c_times:
            return []
        all_t = np.concatenate(c_times)
        all_s = np.concatenate(c_states)
        all_i = np.concatenate(c_idx)
        # Global top-k most recent (largest time). argpartition is O(n).
        if all_t.shape[0] > k:
            part = np.argpartition(all_t, -k)[-k:]
            order = part[np.argsort(all_t[part])[::-1]]
        else:
            order = np.argsort(all_t)[::-1]
        sel_t = all_t[order]
        sel_i = all_i[order]
        sel_s = all_s[order]
        delta = np.maximum(0.0, (t_ns - sel_t).astype(np.float64) / _NS_PER_S)
        return [(int(sel_i[j]), int(sel_s[j]), float(delta[j]))
                for j in range(order.shape[0])]
