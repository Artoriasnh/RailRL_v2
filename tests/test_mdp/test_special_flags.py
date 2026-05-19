"""Unit tests for railrl.mdp.special_flags — 8 special-case flag computations."""
import pytest

from railrl.mdp.special_flags import (
    f_advance, f_call_on, f_platform_dev, f_priority_compete,
    f_late_train, f_unusual_id, f_trts_pressed, f_freight_class,
    compute_all_flags, get_flag_sources, FlagSources,
)


class TestUnusualId:
    def test_standard_headcode_not_unusual(self):
        assert f_unusual_id("1S49") is False
        assert f_unusual_id("6M91") is False
        assert f_unusual_id("5K23") is False

    def test_non_standard_format(self):
        assert f_unusual_id("343R") is True   # not <digit><letter><digits>

    def test_empty_or_wrong_length(self):
        assert f_unusual_id("") is True
        assert f_unusual_id("XYZ") is True
        assert f_unusual_id("12345") is True


class TestFreightClass:
    def test_freight_classes(self):
        assert f_freight_class("4") is True   # container intermodal
        assert f_freight_class("6") is True   # heavy freight

    def test_passenger_classes(self):
        assert f_freight_class("1") is False  # express passenger
        assert f_freight_class("2") is False  # stopping
        assert f_freight_class("5") is False  # ECS

    def test_none(self):
        assert f_freight_class(None) is False


class TestLateTrain:
    def test_late_above_threshold(self):
        assert f_late_train(-120.0) == 120  # 120s late
        assert f_late_train(-300.0) == 300

    def test_late_below_threshold(self):
        assert f_late_train(-30.0) == 0   # under 60s threshold
        assert f_late_train(-59.0) == 0

    def test_on_time_or_early(self):
        assert f_late_train(0.0) == 0
        assert f_late_train(60.0) == 0  # ahead of schedule

    def test_none(self):
        assert f_late_train(None) == 0


class TestAdvance:
    def test_other_train_occupies_first_tc(self):
        occ = {"TFBN": "2A28"}
        assert f_advance(["TFBN"], occ, focal_train="1S49") is True

    def test_first_tc_empty(self):
        occ = {"TFBN": None}
        assert f_advance(["TFBN"], occ, focal_train="1S49") is False

    def test_self_occupation(self):
        occ = {"TFBN": "1S49"}
        # Self-occupation doesn't count as advance routing
        assert f_advance(["TFBN"], occ, focal_train="1S49") is False

    def test_multiple_candidates(self):
        occ = {"TFBN": None, "TGAU": "2A28"}
        # At least one is occupied
        assert f_advance(["TFBN", "TGAU"], occ, focal_train="1S49") is True

    def test_no_candidates(self):
        assert f_advance([], {}, focal_train="1S49") is False


class TestCallOn:
    def test_call_on_with_occupied_platform(self):
        plat_occ = {3: True}
        assert f_call_on(["C"], [3], plat_occ) is True

    def test_main_route_not_call_on(self):
        plat_occ = {3: True}
        assert f_call_on(["M"], [3], plat_occ) is False

    def test_call_on_empty_platform(self):
        plat_occ = {3: False}
        assert f_call_on(["C"], [3], plat_occ) is False


class TestPlatformDev:
    def test_no_candidate_matches_planned(self):
        assert f_platform_dev([3, 5], planned_platform=4) is True

    def test_at_least_one_matches(self):
        assert f_platform_dev([3, 4], planned_platform=4) is False

    def test_no_planned_platform(self):
        assert f_platform_dev([3, 5], planned_platform=None) is False


class TestPriorityCompete:
    def test_zero_others(self):
        assert f_priority_compete(0) is False

    def test_one_other(self):
        assert f_priority_compete(1) is True

    def test_many_others(self):
        assert f_priority_compete(10) is True


class TestTrtsPressed:
    def test_planned_platform_trts_on(self):
        assert f_trts_pressed(3, 1, {3: True}) is True

    def test_current_platform_trts_on(self):
        assert f_trts_pressed(3, 1, {1: True}) is True

    def test_no_trts(self):
        assert f_trts_pressed(3, 1, {}) is False

    def test_both_none(self):
        assert f_trts_pressed(None, None, {3: True, 1: True}) is False


class TestComputeAllFlags:
    def test_smoke(self):
        flags = compute_all_flags(
            focal_train="1S49",
            headcode_class_digit="1",
            candidate_routes_first_tc=["TFBN"],
            candidate_route_cls_list=["M"],
            candidate_end_platforms=[3],
            tc_occupancy_now={"TFBN": "2A28"},
            platform_occupancy_now={3: False},
            planned_platform=3,
            current_platform=None,
            trts_state_by_platform={},
            n_other_active_trains=1,
            scheduled_delta_seconds=-120.0,
        )
        assert set(flags.keys()) == {
            "f_advance", "f_call_on", "f_platform_dev", "f_priority_compete",
            "f_late_train", "f_unusual_id", "f_trts_pressed", "f_freight_class",
        }
        assert flags["f_advance"] is True       # other train on first TC
        assert flags["f_call_on"] is False      # M route not C
        assert flags["f_platform_dev"] is False # 3 matches planned
        assert flags["f_priority_compete"] is True
        assert flags["f_late_train"] == 120
        assert flags["f_unusual_id"] is False
        assert flags["f_trts_pressed"] is False
        assert flags["f_freight_class"] is False


class TestFlagSources:
    def test_all_8_have_sources(self):
        sources = get_flag_sources()
        expected = {
            "f_advance", "f_call_on", "f_platform_dev", "f_priority_compete",
            "f_late_train", "f_unusual_id", "f_trts_pressed", "f_freight_class",
        }
        assert set(sources.keys()) == expected

    def test_trts_source_locked_to_planned_or_current(self):
        """spec 01 §17.5.4: f_trts_pressed source MUST be planned/current
        platform, NEVER focal_signal's platform."""
        sources = get_flag_sources()
        trts_src = sources["f_trts_pressed"].lower()
        assert "planned_platform" in trts_src or "current_platform" in trts_src
        assert "focal_signal" not in trts_src
