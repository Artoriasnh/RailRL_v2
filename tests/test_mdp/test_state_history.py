"""Unit tests for railrl.mdp.state_history (Stage 3 Round 3)."""
import numpy as np
import pandas as pd
import pytest

from railrl.mdp.state_history import (
    _StateTimeline, TrackOccupancyHistory, SignalAspectHistory,
    BerthHistory, MovementsLookup, EventTokenStream,
)


# ============================================================
# Helpers
# ============================================================

def _td(times, types, ids, states, trains=None, to_berths=None, prs=None):
    """Construct a minimal TD events DataFrame with the schema fields we use."""
    n = len(times)
    if trains is None:
        trains = ["0"] * n
    if to_berths is None:
        to_berths = [None] * n
    if prs is None:
        prs = ["0"] * n
    return pd.DataFrame({
        "time":           pd.to_datetime(times),
        "type":           pd.Categorical(types, categories=[
            "Track", "Signal", "CA", "CB", "CC", "Route", "Panel_Request", "TRTS"]),
        "id":             ids,
        "state":          pd.array(states, dtype="Int8"),
        "trainid_filled": trains,
        "to_berth":       to_berths,
        "from_berth":     [None] * n,
        "descr":          [None] * n,
        "change":         [None] * n,
        "timegap":        [0.0] * n,
        "Panel_Request":  prs,
    })


# ============================================================
# _StateTimeline
# ============================================================

class TestStateTimeline:
    def test_empty_state_at(self):
        tl = _StateTimeline()
        assert tl.state_at(0) == 0
        assert tl.occupier_at(0) is None
        assert tl.last_change_age_s(0) == 0
        assert tl.window_stats(0, 60.0) == (0.0, 0)

    def test_state_at_with_prior_event(self):
        tl = _StateTimeline(
            times_ns=np.array([100, 200, 300], dtype=np.int64),
            states=np.array([1, 0, 1], dtype=np.int8),
            train_ids=["A", "B", "C"],
        )
        assert tl.state_at(50) == 0      # before any event
        assert tl.state_at(100) == 1
        assert tl.state_at(150) == 1
        assert tl.state_at(200) == 0
        assert tl.state_at(300) == 1
        assert tl.state_at(999) == 1

    def test_window_stats_full_in_state_1(self):
        # Event at t=100, state=1; window [50, 200] (150 ns). Inside the
        # window, state was 1 for all 150 ns (event at t=100 enters at 100,
        # held until t=200). Actually: start_state at t=50 is 0 (no prior),
        # then at t=100 transitions to 1, then state=1 dwell is 200-100=100.
        # window_ns=150 → frac = 100/150 ≈ 0.667
        tl = _StateTimeline(
            times_ns=np.array([100], dtype=np.int64),
            states=np.array([1], dtype=np.int8),
            train_ids=["A"],
        )
        frac, n = tl.window_stats(t_ns=200, window_s=150e-9)
        assert abs(frac - (100/150)) < 1e-6
        assert n == 1  # one transition inside window

    def test_window_stats_no_events_in_window(self):
        tl = _StateTimeline(
            times_ns=np.array([100], dtype=np.int64),
            states=np.array([1], dtype=np.int8),
            train_ids=["A"],
        )
        # Window [110, 200] — fully after the event, no transitions inside.
        # start_state at t=110 is 1 (held over).
        frac, n = tl.window_stats(t_ns=200, window_s=90e-9)
        assert frac == 1.0
        assert n == 0


# ============================================================
# TrackOccupancyHistory
# ============================================================

class TestTrackOccupancyHistory:
    def test_empty(self):
        h = TrackOccupancyHistory.build(_td([], [], [], []))
        assert not h.occupied_now("TFBN", 0)
        assert h.current_occupier("TFBN", 0) is None

    def test_basic_occupancy(self):
        df = _td(
            times=["2024-01-01 10:00:00", "2024-01-01 10:00:30"],
            types=["Track", "Track"],
            ids=["TFBN", "TFBN"],
            states=[1, 0],
            trains=["1S49", "0"],
        )
        h = TrackOccupancyHistory.build(df)
        t_during = int(pd.Timestamp("2024-01-01 10:00:15").value)
        t_after = int(pd.Timestamp("2024-01-01 10:01:00").value)
        assert h.occupied_now("TFBN", t_during) is True
        assert h.occupied_now("TFBN", t_after) is False
        assert h.current_occupier("TFBN", t_during) == "1S49"

    def test_window_fraction(self):
        df = _td(
            times=["2024-01-01 10:00:00", "2024-01-01 10:00:30"],
            types=["Track", "Track"],
            ids=["TFBN", "TFBN"],
            states=[1, 0],
            trains=["1S49", "0"],
        )
        h = TrackOccupancyHistory.build(df)
        # At t=10:01:00, 1-minute window covers 10:00:00→10:01:00.
        # State 1 from 10:00:00 to 10:00:30 = 30s; window = 60s.
        # frac = 0.5
        t0 = int(pd.Timestamp("2024-01-01 10:01:00").value)
        frac, n = h.window_stats("TFBN", t0, 60.0)
        assert abs(frac - 0.5) < 0.001
        assert n == 1  # one transition inside the window


# ============================================================
# SignalAspectHistory
# ============================================================

class TestSignalAspectHistory:
    def test_full_name_to_id_mapping(self):
        df = _td(
            times=["2024-01-01 10:00:00", "2024-01-01 10:00:30"],
            types=["Signal", "Signal"],
            ids=["STD5040", "STD5040"],
            states=[1, 0],
        )
        h = SignalAspectHistory.build(df, full_name_to_id={"STD5040": "5040"})
        t0 = int(pd.Timestamp("2024-01-01 10:00:15").value)
        # Should be lookable as "5040", not "STD5040"
        assert h.aspect_restrictive_now("5040", t0) is True
        # heuristic fallback when no map
        h2 = SignalAspectHistory.build(df)
        assert h2.aspect_restrictive_now("5040", t0) is True


# ============================================================
# BerthHistory
# ============================================================

class TestBerthHistory:
    def test_berth_occupant(self):
        df = _td(
            times=["2024-01-01 10:00:00", "2024-01-01 10:00:30"],
            types=["Track", "Track"],
            ids=["TFBN", "TFPJ"],
            states=[1, 1],
            trains=["1S49", "2A28"],
            to_berths=["5040", "5044"],
        )
        h = BerthHistory.build(df)
        t0 = int(pd.Timestamp("2024-01-01 10:01:00").value)
        occ, age = h.berth_occupant_at("5040", t0)
        assert occ == "1S49"
        assert age == 60
        occ2, _ = h.berth_occupant_at("9999", t0)
        assert occ2 is None

    def test_recent_pr_count(self):
        df = _td(
            times=["2024-01-01 10:00:00", "2024-01-01 10:01:00",
                    "2024-01-01 10:02:00", "2024-01-01 10:10:00"],
            types=["Panel_Request"] * 4,
            ids=[None] * 4,
            states=[0] * 4,
            trains=["1S49"] * 4,
        )
        h = BerthHistory.build(df)
        t0 = int(pd.Timestamp("2024-01-01 10:03:00").value)
        # Last 5 min: PRs at 10:00, 10:01, 10:02 → 3 in window (10:00 inclusive)
        # Actually: window = (t-300, t], PRs at 10:00=180s ago (inside),
        # 10:01=120s ago (inside), 10:02=60s ago (inside). 10:10 is in future.
        assert h.recent_pr_count("1S49", t0, 300.0) == 3
        assert h.recent_pr_count("9X99", t0, 300.0) == 0


# ============================================================
# MovementsLookup
# ============================================================

class TestMovementsLookup:
    def test_planned_platform_is_int_not_float(self):
        """Regression test for the pandas Series-coerce-to-float bug."""
        mv = pd.DataFrame({
            "gbtt_timestamp":    pd.to_datetime(["2024-01-01 10:05:00",
                                                   "2024-01-01 10:10:00"]),
            "current_train_id":  ["1S49", "3B99"],
            "platform":          [3, 99],   # 99 → out of range → None
            "event_type":        ["ARRIVAL", "DEPARTURE"],
        })
        ml = MovementsLookup.build(mv)
        p_ok = ml.planned_platform("1S49", int(pd.Timestamp("2024-01-01 10:03").value))
        assert p_ok == 3
        assert isinstance(p_ok, int) and not isinstance(p_ok, bool)
        p_bad = ml.planned_platform("3B99", int(pd.Timestamp("2024-01-01 10:03").value))
        assert p_bad is None  # not nan, not 99 — must be None

    def test_schedule_outlook_excludes_focal(self):
        mv = pd.DataFrame({
            "gbtt_timestamp":   pd.to_datetime(["2024-01-01 10:05:00",
                                                  "2024-01-01 10:08:00"]),
            "current_train_id": ["1S49", "2A28"],
            "platform":         [3, 4],
            "event_type":       ["ARRIVAL", "DEPARTURE"],
        })
        ml = MovementsLookup.build(mv)
        t0 = int(pd.Timestamp("2024-01-01 10:03").value)
        out = ml.schedule_outlook(t0, k=5, lookahead_s=600, exclude_train="1S49")
        assert len(out) == 1
        assert out[0]["train_id"] == "2A28"
        assert out[0]["planned_platform"] == 4
        # Never returns a signal_id-like string
        assert not isinstance(out[0]["planned_platform"], str)


# ============================================================
# EventTokenStream
# ============================================================

class TestEventTokenStream:
    def test_slice_last_k_sorted_descending(self):
        df = _td(
            times=["2024-01-01 10:00:00", "2024-01-01 10:00:10",
                    "2024-01-01 10:00:30", "2024-01-01 10:00:50"],
            types=["Track", "Signal", "Track", "Signal"],
            ids=["TFBN", "STD5040", "TFBN", "STD5040"],
            states=[1, 1, 0, 0],
        )
        es = EventTokenStream.build(df)
        t0 = int(pd.Timestamp("2024-01-01 10:01:00").value)
        toks = es.slice_last_k(["TFBN", "STD5040"], t0, k=10)
        assert len(toks) == 4
        # Most recent first: STD5040 (0, at t+10s), TFBN (0, at t+30s), ...
        # Actually time_delta_s = t0 - event_time so most recent = smallest delta
        deltas = [t[2] for t in toks]
        assert deltas == sorted(deltas)  # ascending delta = descending time

    def test_slice_last_k_caps_at_k(self):
        df = _td(
            times=[f"2024-01-01 10:00:{i:02d}" for i in range(20)],
            types=["Track"] * 20,
            ids=["TFBN"] * 20,
            states=[i % 2 for i in range(20)],
        )
        es = EventTokenStream.build(df)
        t0 = int(pd.Timestamp("2024-01-01 10:01:00").value)
        toks = es.slice_last_k(["TFBN"], t0, k=5)
        assert len(toks) == 5

    def test_no_future_leakage(self):
        df = _td(
            times=["2024-01-01 10:00:00", "2024-01-01 10:05:00"],
            types=["Track", "Track"],
            ids=["TFBN", "TFBN"],
            states=[1, 0],
        )
        es = EventTokenStream.build(df)
        t0 = int(pd.Timestamp("2024-01-01 10:02:00").value)
        toks = es.slice_last_k(["TFBN"], t0, k=10)
        assert len(toks) == 1  # only the 10:00 event, NOT the 10:05 one
        assert toks[0][2] >= 0
