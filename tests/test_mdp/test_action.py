"""Unit tests for railrl.mdp.action — feasible_actions + RouteIndex."""
import pytest
import pandas as pd
import numpy as np

from railrl.mdp.action import (
    RouteIndex, feasible_actions, _route_direction, _infer_train_direction,
)


# ============================================================
# Direction inference helpers
# ============================================================

class TestDirectionHelpers:
    def test_route_direction_forward(self):
        # alphabetical order = forward
        assert _route_direction(["TC_A", "TC_B", "TC_C"]) == "forward"

    def test_route_direction_reverse(self):
        assert _route_direction(["TC_Z", "TC_Y", "TC_X"]) == "reverse"

    def test_route_direction_too_short(self):
        assert _route_direction([]) is None
        assert _route_direction(["only_one"]) is None

    def test_infer_train_direction_forward(self):
        assert _infer_train_direction(["A", "B", "C", "D"]) == "forward"

    def test_infer_train_direction_reverse(self):
        assert _infer_train_direction(["Z", "Y", "X"]) == "reverse"


# ============================================================
# RouteIndex
# ============================================================

class TestRouteIndex:
    @pytest.fixture
    def routes_df(self):
        return pd.DataFrame([
            {"route_id": "RTD5045A(M)", "start_signal": "5045", "cls": "M",
             "track_sections": ["TC1", "TC2", "TC3"], "end_platform_id": 3},
            {"route_id": "RTD5045B(M)", "start_signal": "5045", "cls": "M",
             "track_sections": ["TC1", "TC4"], "end_platform_id": 4},
            {"route_id": "RDC5076A(M)", "start_signal": "5076", "cls": "M",
             "track_sections": ["TC9", "TC8"], "end_platform_id": 5},
        ])

    def test_indexes_by_start_signal(self, routes_df):
        idx = RouteIndex(routes_df)
        # 2 routes from 5045
        assert len(idx.routes_from("5045")) == 2
        # 1 route from 5076
        assert len(idx.routes_from("5076")) == 1
        # 0 routes from unknown
        assert len(idx.routes_from("9999")) == 0

    def test_routes_from_returns_route_dicts(self, routes_df):
        idx = RouteIndex(routes_df)
        routes = idx.routes_from("5045")
        ids = {r["route_id"] for r in routes}
        assert ids == {"RTD5045A(M)", "RTD5045B(M)"}

    def test_direction_inferred(self, routes_df):
        idx = RouteIndex(routes_df)
        for r in idx.routes_from("5076"):
            assert r["direction"] == "reverse"   # TC9 > TC8


# ============================================================
# feasible_actions
# ============================================================

class TestFeasibleActions:
    @pytest.fixture
    def route_index(self):
        df = pd.DataFrame([
            {"route_id": "R_A_fwd", "start_signal": "5045", "cls": "M",
             "track_sections": ["A", "B", "C"], "end_platform_id": 3},
            {"route_id": "R_B_fwd", "start_signal": "5045", "cls": "M",
             "track_sections": ["A", "D"], "end_platform_id": 4},
            {"route_id": "R_C_rev", "start_signal": "5045", "cls": "M",
             "track_sections": ["Z", "Y"], "end_platform_id": 1},
        ])
        return RouteIndex(df)

    def test_returns_routes_from_signal(self, route_index):
        cands = feasible_actions(
            focal_train="1S49", focal_signal="5045", t=None,
            route_index=route_index,
            train_direction=None,
            direction_filter=False, platform_soft_filter=False,
        )
        assert set(cands) == {"R_A_fwd", "R_B_fwd", "R_C_rev"}

    def test_direction_filter(self, route_index):
        # train_direction=forward → only forward routes
        cands = feasible_actions(
            focal_train="1S49", focal_signal="5045", t=None,
            route_index=route_index,
            train_direction="forward",
            direction_filter=True,
        )
        assert set(cands) == {"R_A_fwd", "R_B_fwd"}

    def test_prev_routes_filter_excludes_already_set(self, route_index):
        cands = feasible_actions(
            focal_train="1S49", focal_signal="5045", t=None,
            route_index=route_index,
            train_direction=None,
            prev_routes_set=["R_A_fwd"],
            direction_filter=False, platform_soft_filter=False,
        )
        assert "R_A_fwd" not in cands
        assert "R_B_fwd" in cands

    def test_platform_soft_filter_reorders(self, route_index):
        """planned_platform=3 should put R_A_fwd (end_platform=3) first."""
        cands = feasible_actions(
            focal_train="1S49", focal_signal="5045", t=None,
            route_index=route_index,
            train_direction=None,
            planned_platform=3,
            direction_filter=False, platform_soft_filter=True,
        )
        # R_A_fwd (platform 3) should be first
        assert cands[0] == "R_A_fwd"

    def test_unknown_signal_returns_empty(self, route_index):
        cands = feasible_actions(
            focal_train="1S49", focal_signal="9999", t=None,
            route_index=route_index,
            train_direction=None,
        )
        assert cands == []

    def test_train_direction_inferred_from_recent_tcs(self, route_index):
        # train at A → going forward; only R_A_fwd, R_B_fwd
        cands = feasible_actions(
            focal_train="1S49", focal_signal="5045", t=None,
            route_index=route_index,
            train_direction=None,
            train_recent_tcs=["TC_a", "TC_b", "TC_c"],  # forward
            direction_filter=True,
        )
        forward_routes = {"R_A_fwd", "R_B_fwd"}
        assert set(cands) == forward_routes
