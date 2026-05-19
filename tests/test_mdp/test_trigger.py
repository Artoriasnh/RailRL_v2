"""Unit tests for railrl.mdp.trigger — decision point generation."""
import pytest
import pandas as pd
import numpy as np

from railrl.mdp.trigger import (
    compute_approach_tracks, summarize,
)


# ============================================================
# Approach tracks computation
# ============================================================

class TestApproachTracks:
    def test_basic_single_route(self):
        """Single route ending at signal '5045' with 3 TCs should put last 2
        in approach_TCs('5045') when k_hops=2."""
        routes = pd.DataFrame([{
            "route_id": "RTD5045A(M)",
            "end_signal": "5045",
            "track_sections": ["TC1", "TC2", "TC3"],
        }])
        result = compute_approach_tracks(routes, k_hops=2)
        assert "5045" in result
        # Last 2 TCs
        assert result["5045"] == {"TC2", "TC3"}

    def test_short_route(self):
        """Route with only 1 TC — all 1 should be in approach."""
        routes = pd.DataFrame([{
            "route_id": "RTD5045A(M)",
            "end_signal": "5045",
            "track_sections": ["TC1"],
        }])
        result = compute_approach_tracks(routes, k_hops=2)
        assert result["5045"] == {"TC1"}

    def test_multiple_routes_same_end(self):
        """Two routes both ending at signal '5045' should union their tail TCs."""
        routes = pd.DataFrame([
            {"route_id": "R1", "end_signal": "5045", "track_sections": ["A", "B"]},
            {"route_id": "R2", "end_signal": "5045", "track_sections": ["C", "D"]},
        ])
        result = compute_approach_tracks(routes, k_hops=2)
        assert result["5045"] == {"A", "B", "C", "D"}

    def test_empty_track_sections(self):
        """Routes with empty tracks → not added to result."""
        routes = pd.DataFrame([{
            "route_id": "R", "end_signal": "5045", "track_sections": [],
        }])
        result = compute_approach_tracks(routes, k_hops=2)
        assert "5045" not in result or result["5045"] == set()

    def test_k_hops_zero(self):
        routes = pd.DataFrame([{
            "route_id": "R", "end_signal": "5045", "track_sections": ["A", "B"],
        }])
        # k=0 should return empty tail
        result = compute_approach_tracks(routes, k_hops=0)
        assert result.get("5045", set()) == set()


# ============================================================
# Summarize
# ============================================================

class TestSummarize:
    def test_basic(self):
        dp = pd.DataFrame({
            "focal_train":  ["1S49", "1S49", "2A28"],
            "focal_signal": ["5045", "5040", "5045"],
            "t":            pd.to_datetime(["2024-01-01 10:00",
                                              "2024-01-01 10:05",
                                              "2024-01-01 10:10"]),
            "label":        ["set", "wait", "set"],
            "chosen_route_id": ["RTD5045A(M)", None, "RTD5045B(M)"],
            "trigger_type": ["panel_request", "approach", "panel_request"],
        })
        s = summarize(dp)
        assert s["n_total"] == 3
        assert s["n_set"] == 2
        assert s["n_wait"] == 1
        assert s["n_unique_trains"] == 2
        assert s["n_unique_signals"] == 2
        assert s["by_trigger"]["panel_request"] == 2
        assert s["by_trigger"]["approach"] == 1
