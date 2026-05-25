"""spec 02 §4.10 — Eight special-case flag computations.

Each flag is computed from time≤t observable state, with explicit `source`
declarations that the leak audit (§7) checks.

The flags are passed to the Q-network (spec 03 §6) and the L2 explanation
template (spec 05 §8); they help the model distinguish ~10% "expert decisions"
from ~90% trivial decisions and provide structured per-flag interpretability.

Per spec 01 §17.5, all flag inputs are auditable:
  - source: which observable data the flag depends on
  - never uses focal_signal directly as state input (only via candidate route)
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .. import config as C


# Standard headcode pattern: <class digit><letter><2 digits>
_HEADCODE_STANDARD = re.compile(r"^[0-9][A-Z][0-9]{2}$")


@dataclass
class FlagSources:
    """Provenance declarations for leak audit (spec 01 §17.5.4)."""
    f_advance:           str = "static_graph(routes_clean.track_sections[0]) + td_events(t<=t)"
    f_call_on:           str = "candidate_routes(cls=C) + platform_tc_map + td_events(t<=t)"
    f_platform_dev:      str = "candidate_routes(end_platform_id) + movements.gbtt"
    f_priority_compete:  str = "decision_points window [t-5s, t+5s]"
    f_late_train:        str = "movements.gbtt - t"
    f_unusual_id:        str = "parsers.HEADCODE_RE (regex check on train_id)"
    f_trts_pressed:      str = "td_events(t<=t) at planned_platform OR current_platform"
    f_freight_class:     str = "parsed headcode_class ∈ {4, 6}"

    def as_dict(self) -> dict:
        return {
            "f_advance":          self.f_advance,
            "f_call_on":          self.f_call_on,
            "f_platform_dev":     self.f_platform_dev,
            "f_priority_compete": self.f_priority_compete,
            "f_late_train":       self.f_late_train,
            "f_unusual_id":       self.f_unusual_id,
            "f_trts_pressed":     self.f_trts_pressed,
            "f_freight_class":    self.f_freight_class,
        }


# ============================================================
# Individual flag computations
# ============================================================

def f_advance(
    candidate_routes_first_tc: list[str],
    tc_occupancy_now: dict[str, Optional[str]],   # tc_id → current train_id (None if empty)
    focal_train: str,
) -> bool:
    """The first TC of any candidate route is currently occupied by a train ≠ focal_train.

    Indicates advance routing — the signaller may be setting before the
    previous train fully clears.
    """
    for tc in candidate_routes_first_tc:
        occupier = tc_occupancy_now.get(tc)
        if occupier and occupier != focal_train:
            return True
    return False


def f_call_on(
    candidate_route_cls_list: list[str],
    candidate_end_platforms: list[Optional[int]],
    platform_occupancy_now: dict[int, bool],     # platform_id → is_occupied
) -> bool:
    """Any candidate route has cls='C' AND its end_platform is currently occupied.

    Indicates call-on (permissive working) scenario.
    """
    for cls, end_plat in zip(candidate_route_cls_list, candidate_end_platforms):
        if cls == "C" and end_plat is not None and platform_occupancy_now.get(end_plat, False):
            return True
    return False


def f_platform_dev(
    candidate_end_platforms: list[Optional[int]],
    planned_platform: Optional[int],
) -> bool:
    """The candidate routes' end platforms are KNOWN and none matches the focal
    train's planned_platform → platform reassignment is likely needed.

    Returns False when:
      - planned_platform is None (no schedule info), OR
      - NO candidate route has a known end_platform_id (can't establish a
        deviation from missing data — same conservative rule as elsewhere).

    ⚠️ FIXED (Stage 4.7.2d diagnostic 19): the old version
    `not any(p == planned for p in cands if p is not None)` returned True when
    EVERY candidate end_platform_id was None (empty generator → any()=False).
    Only ~28% of route nodes have end_platform_id (many routes legitimately don't
    end at a platform — through/depot routes), so it fired on 83% of decisions
    (99.2% of fires were this all-None degenerate case). When candidate platforms
    ARE known they match planned 93.2% of the time, so the true deviation rate is
    ~0.7% (matches spec §4.4 ~1.5%). See IMPLEMENTATION_LOG.
    """
    if planned_platform is None:
        return False
    known = [p for p in candidate_end_platforms if p is not None]
    if not known:
        return False
    return not any(p == planned_platform for p in known)


def f_priority_compete(
    n_other_active_trains: int,
    threshold: int = 1,
) -> bool:
    """≥ `threshold` other trains are active (in approach or with recent PR)
    in [t-5s, t+5s]. Indicates competing trains.
    """
    return n_other_active_trains >= threshold


def f_late_train(scheduled_delta_seconds: Optional[float]) -> int:
    """Seconds late when the train is late by ≥ 60 s (≥ 1 min); else 0.

    ⚠️ REDEFINED (Stage 4.7.2d lateness fix): scheduled_delta_s is now the CURRENT
    SIGNED lateness from realized timetable_variation ≤ t (state_history
    .current_lateness_s), where **positive = late** (timetable_variation×60 with
    sign from variation_status: LATE +, EARLY −, ON TIME/OFF ROUTE 0). So we fire
    when delta ≥ +60 and return the seconds late. (Old convention used gbtt−t with
    negative = late and never triggered — see IMPLEMENTATION_LOG 4.7.2d.)

    Returns int (seconds late, ≥0), not bool.
    """
    if scheduled_delta_seconds is None:
        return 0
    if scheduled_delta_seconds >= 60.0:
        return int(scheduled_delta_seconds)
    return 0


def f_unusual_id(train_id: str) -> bool:
    """True if train_id is NOT a standard 4-char headcode (<digit><letter><digits>).

    Per PROJECT_HANDOFF Ch 5.5, 1.04% of train_ids are non-standard
    (e.g., '343R'); they cluster in depot/sidings.
    """
    if not train_id or len(train_id) != 4:
        return True
    return _HEADCODE_STANDARD.match(train_id) is None


def f_trts_pressed(
    train_planned_platform: Optional[int],
    train_current_platform: Optional[int],
    trts_state_by_platform: dict[int, bool],    # platform_id → TRTS pressed?
) -> bool:
    """TRTS button is currently pressed for planned OR current platform.

    spec 01 §17.5.4 explicitly forbids using focal_signal's platform here;
    must use planned (from gbtt) or current_platform (derived from current_tc).
    """
    for plat in (train_planned_platform, train_current_platform):
        if plat is not None and trts_state_by_platform.get(plat, False):
            return True
    return False


def f_freight_class(headcode_class_digit: Optional[str]) -> bool:
    """True if headcode class ∈ {4, 6} (container intermodal or heavy freight).

    Per PROJECT_HANDOFF Ch 2 — freight decisions follow different priority
    patterns than passenger.
    """
    return str(headcode_class_digit) in {"4", "6"}


# ============================================================
# Batch helper — compute all 8 for one sample
# ============================================================

def compute_all_flags(
    *,
    focal_train: str,
    headcode_class_digit: Optional[str],
    candidate_routes_first_tc: list[str],
    candidate_route_cls_list: list[str],
    candidate_end_platforms: list[Optional[int]],
    tc_occupancy_now: dict[str, Optional[str]],
    platform_occupancy_now: dict[int, bool],
    planned_platform: Optional[int],
    current_platform: Optional[int],
    trts_state_by_platform: dict[int, bool],
    n_other_active_trains: int,
    scheduled_delta_seconds: Optional[float],
) -> dict:
    """Compute all 8 special flags for one snapshot.

    Returns dict with 8 boolean / int values + meta dict containing source
    declarations. The meta is used by `assert_no_leak()` to verify each
    flag's input was from an allowed source.
    """
    flags = {
        "f_advance":          f_advance(candidate_routes_first_tc,
                                         tc_occupancy_now, focal_train),
        "f_call_on":          f_call_on(candidate_route_cls_list,
                                         candidate_end_platforms,
                                         platform_occupancy_now),
        "f_platform_dev":     f_platform_dev(candidate_end_platforms,
                                              planned_platform),
        "f_priority_compete": f_priority_compete(n_other_active_trains),
        "f_late_train":       f_late_train(scheduled_delta_seconds),
        "f_unusual_id":       f_unusual_id(focal_train),
        "f_trts_pressed":     f_trts_pressed(planned_platform, current_platform,
                                              trts_state_by_platform),
        "f_freight_class":    f_freight_class(headcode_class_digit),
    }
    return flags


def get_flag_sources() -> dict:
    """Return the source declarations for all 8 flags (for leak audit)."""
    return FlagSources().as_dict()
