"""Unit tests for railrl.mdp.leak_audit — spec 02 §7 contract."""
import pytest

from railrl.mdp.leak_audit import (
    assert_no_leak, collect_violations, LeakAuditError, BANNED_STATE_FIELDS,
)


def _clean_snapshot():
    return {
        "center": {"type": "track", "id": "TFBN"},
        "state_nodes_track":  [{"track_id": "TFBN", "occupied_now": True}],
        "state_nodes_signal": [{"signal_id": "5040", "aspect_restrictive_now": True}],
        "state_nodes_route":  [{"route_id": "RTD5040A(M)"}],
        "state_nodes_train":  [{"train_id": "1S49", "is_focal": True},
                                {"train_id": "2A28", "is_focal": False}],
        "state_event_tokens": [{"asset_idx": 0, "state": 1, "time_delta_s": 30.0}],
        "state_schedule_outlook": [{"train_id": "5C05", "planned_platform": 3}],
        "state_special_flags_meta": {"f_trts_pressed_source": "planned_platform"},
    }


def _meta():
    return {"focal_train": "1S49", "focal_train_current_tc": "TFBN"}


class TestCleanSnapshot:
    def test_clean_passes_all_checks(self):
        assert assert_no_leak(_clean_snapshot(), _meta(), t_ns=0) is True

    def test_collect_violations_empty_on_clean(self):
        assert collect_violations(_clean_snapshot(), _meta(), t_ns=0) == []


class TestCheck1SubgraphCenter:
    def test_signal_centered_fails(self):
        bad = _clean_snapshot()
        bad["center"] = {"type": "signal", "id": "5040"}
        with pytest.raises(LeakAuditError, match="Check 1"):
            assert_no_leak(bad, _meta(), t_ns=0)

    def test_wrong_tc_fails(self):
        bad = _clean_snapshot()
        bad["center"] = {"type": "track", "id": "TWRONG"}
        with pytest.raises(LeakAuditError, match="Check 1"):
            assert_no_leak(bad, _meta(), t_ns=0)


class TestCheck2NoFocalGraphFlags:
    def test_is_focal_signal_fails(self):
        bad = _clean_snapshot()
        bad["state_nodes_signal"] = [{"signal_id": "5040", "is_focal_signal": True}]
        with pytest.raises(LeakAuditError, match="Check 2"):
            assert_no_leak(bad, _meta(), t_ns=0)

    def test_is_focal_route_fails(self):
        bad = _clean_snapshot()
        bad["state_nodes_route"] = [{"route_id": "R", "is_focal_route": True}]
        with pytest.raises(LeakAuditError, match="Check 2"):
            assert_no_leak(bad, _meta(), t_ns=0)

    def test_is_chosen_on_route_fails(self):
        bad = _clean_snapshot()
        bad["state_nodes_route"] = [{"route_id": "R", "is_chosen": True}]
        with pytest.raises(LeakAuditError, match="Check 2"):
            assert_no_leak(bad, _meta(), t_ns=0)


class TestCheck3BannedFields:
    @pytest.mark.parametrize("banned_field", [
        "r_total", "delay_change_seconds", "next_tc_headway_seconds",
        "route_outcome", "chosen_route_id", "focal_signal",
    ])
    def test_banned_field_fails(self, banned_field):
        bad = _clean_snapshot()
        bad["state_nodes_train"] = [
            {"train_id": "1S49", "is_focal": True, banned_field: "leak"},
            {"train_id": "2A28", "is_focal": False},
        ]
        with pytest.raises(LeakAuditError, match="Check 3"):
            assert_no_leak(bad, _meta(), t_ns=0)


class TestCheck4ScheduleOutlook:
    def test_planned_end_signal_fails(self):
        # `planned_end_signal` is in BANNED_STATE_FIELDS, so Check 3 would
        # catch it first. To exercise Check 4's schedule_outlook logic
        # specifically, skip the banned_fields scan.
        bad = _clean_snapshot()
        bad["state_schedule_outlook"] = [{"planned_end_signal": "5040"}]
        with pytest.raises(LeakAuditError, match="Check 4"):
            assert_no_leak(bad, _meta(), t_ns=0, skip_checks={"banned_fields"})

    def test_planned_end_signal_caught_by_banned_fields(self):
        # When all checks are on, Check 3 (banned_fields) catches this
        # before Check 4 — both are valid lines of defense.
        bad = _clean_snapshot()
        bad["state_schedule_outlook"] = [{"planned_end_signal": "5040"}]
        with pytest.raises(LeakAuditError, match="Check 3"):
            assert_no_leak(bad, _meta(), t_ns=0)

    def test_planned_platform_string_fails(self):
        bad = _clean_snapshot()
        bad["state_schedule_outlook"] = [{"planned_platform": "5040"}]   # string!
        with pytest.raises(LeakAuditError, match="Check 4"):
            assert_no_leak(bad, _meta(), t_ns=0)

    def test_planned_platform_out_of_range(self):
        bad = _clean_snapshot()
        bad["state_schedule_outlook"] = [{"planned_platform": 99}]
        with pytest.raises(LeakAuditError, match="Check 4"):
            assert_no_leak(bad, _meta(), t_ns=0)


class TestCheck5FlagsSource:
    def test_focal_signal_source_fails(self):
        bad = _clean_snapshot()
        bad["state_special_flags_meta"] = {"f_trts_pressed_source": "focal_signal_platform"}
        with pytest.raises(LeakAuditError, match="Check 5"):
            assert_no_leak(bad, _meta(), t_ns=0)

    def test_current_platform_source_passes(self):
        snap = _clean_snapshot()
        snap["state_special_flags_meta"] = {"f_trts_pressed_source": "current_platform"}
        assert_no_leak(snap, _meta(), t_ns=0)


class TestCheck6TemporalCausality:
    def test_negative_time_delta_fails(self):
        bad = _clean_snapshot()
        bad["state_event_tokens"] = [{"asset_idx": 0, "state": 1, "time_delta_s": -1.0}]
        with pytest.raises(LeakAuditError, match="Check 6"):
            assert_no_leak(bad, _meta(), t_ns=0)


class TestCheck7OneFocalTrain:
    def test_zero_focal_fails(self):
        bad = _clean_snapshot()
        bad["state_nodes_train"] = [{"train_id": "1S49", "is_focal": False}]
        with pytest.raises(LeakAuditError, match="Check 7"):
            assert_no_leak(bad, _meta(), t_ns=0)

    def test_two_focal_fails(self):
        bad = _clean_snapshot()
        bad["state_nodes_train"] = [
            {"train_id": "1S49", "is_focal": True},
            {"train_id": "2A28", "is_focal": True},
        ]
        with pytest.raises(LeakAuditError, match="Check 7"):
            assert_no_leak(bad, _meta(), t_ns=0)


class TestBannedFieldsContent:
    def test_banned_fields_locked_set(self):
        assert "focal_signal" in BANNED_STATE_FIELDS
        assert "chosen_route_id" in BANNED_STATE_FIELDS
        assert "r_total" in BANNED_STATE_FIELDS
        assert "delay_change_seconds" in BANNED_STATE_FIELDS
        assert "next_tc_headway_seconds" in BANNED_STATE_FIELDS
        assert "is_focal_signal" in BANNED_STATE_FIELDS
        assert "is_focal_route" in BANNED_STATE_FIELDS
        assert len(BANNED_STATE_FIELDS) >= 21


class TestCollectViolations:
    def test_multiple_violations_collected(self):
        bad = {
            "center": {"type": "signal", "id": "X"},   # check 1
            "state_nodes_signal": [{"signal_id": "X", "is_focal_signal": True}],  # check 2
            "state_nodes_route": [{"route_id": "R", "r_total": 1.0}],  # check 3
            "state_nodes_train": [],   # check 7 (no focal)
            "state_event_tokens": [],
            "state_schedule_outlook": [],
            "state_special_flags_meta": {},
            "state_nodes_track": [],
        }
        violations = collect_violations(bad, _meta(), t_ns=0)
        assert len(violations) >= 4
