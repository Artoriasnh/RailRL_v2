"""Unit tests for railrl.mdp.schema."""
import pytest

from railrl.mdp.schema import (
    IDENTITY_COLS, REWARD_COLS, STATE_NODE_COLS, STATE_EDGE_COLS,
    STATE_OTHER_COLS, ALL_COLS, validate_row,
)


class TestSchemaConstants:
    def test_identity_cols_present(self):
        # Locked: sample_id, focal_train, focal_signal, t, pass_id, etc.
        for col in ["sample_id", "focal_train", "focal_signal", "t",
                    "pass_id", "label", "chosen_route_id", "n_candidates"]:
            assert col in IDENTITY_COLS

    def test_reward_cols_locked(self):
        for col in ["r_delay", "r_throughput", "r_headway", "r_wait", "r_total",
                    "approach_distance", "delay_change_seconds",
                    "next_tc_headway_seconds", "outcome"]:
            assert col in REWARD_COLS

    def test_state_node_cols_are_4_types(self):
        assert STATE_NODE_COLS == [
            "state_nodes_track", "state_nodes_signal",
            "state_nodes_route", "state_nodes_train",
        ]

    def test_state_edge_cols_6_static_plus_2_dynamic(self):
        assert len(STATE_EDGE_COLS) == 8
        for static in ["connects", "traverses", "starts_at",
                        "ends_at", "protects", "same_signal"]:
            assert f"state_edges_{static}" in STATE_EDGE_COLS
        for dynamic in ["at_berth", "next_signal"]:
            assert f"state_edges_{dynamic}" in STATE_EDGE_COLS

    def test_all_cols_size(self):
        assert len(ALL_COLS) == (
            len(IDENTITY_COLS) + len(REWARD_COLS) +
            len(STATE_NODE_COLS) + len(STATE_EDGE_COLS) +
            len(STATE_OTHER_COLS)
        )


class TestValidateRow:
    def _clean_row(self):
        return {
            "sample_id": 0,
            "focal_train": "1S49",
            "focal_signal": "5040",
            "t": 0,
            "label": "set",
            "chosen_route_id": "RTD5040A(M)",
            "candidate_route_ids": ["RTD5040A(M)"],
            "n_candidates": 1,
            "state_nodes_track": [], "state_nodes_signal": [],
            "state_nodes_route": [], "state_nodes_train": [],
        }

    def test_clean_row_passes(self):
        ok, errs = validate_row(self._clean_row())
        assert ok
        assert errs == []

    def test_set_row_missing_chosen_route_id(self):
        row = self._clean_row()
        del row["chosen_route_id"]
        ok, errs = validate_row(row)
        assert not ok
        assert any("chosen_route_id" in e for e in errs)

    def test_n_candidates_mismatch(self):
        row = self._clean_row()
        row["candidate_route_ids"] = ["A", "B"]
        row["n_candidates"] = 5
        ok, errs = validate_row(row)
        assert not ok

    def test_missing_identity_cols(self):
        ok, errs = validate_row({})
        assert not ok
        assert len(errs) > 0
