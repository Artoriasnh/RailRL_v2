"""spec 02 §7 — Leak audit framework.

`assert_no_leak(snapshot, sample_meta, t_ns)` is called at snapshot
construction time (dev mode every batch; production every 1000 batches).
Hard-fails on any of 7 checks:

  1. Subgraph centered on focal_train.current_tc (NOT focal_signal)
  2. No is_focal_signal / is_focal_route flags on graph nodes
  3. No BANNED_STATE_FIELDS in any state_* field (21+ items)
  4. Schedule outlook uses planned_platform (int 1-6 or None), never signal_id
  5. f_trts_pressed source ∈ {'planned_platform', 'current_platform'}
  6. All event tokens have time_delta_s >= 0 (no future leak)
  7. Exactly 1 is_focal=True train node per snapshot

References:
  - spec 01 §17.5 (sample-metadata vs state-features separation)
  - spec 02 §7.1 (this implementation)
  - PROJECT_HANDOFF.docx Ch 14 (extended audit)
"""
from __future__ import annotations
from typing import Any, Optional


# ============================================================
# Banned state fields (spec 01 §14.1 + §17.5.4)
# ============================================================

BANNED_STATE_FIELDS: set[str] = {
    # Direct answer fields
    "focal_signal", "focal_signal_id",
    "chosen_route_id", "chosen_action_idx",
    "focal_route", "focal_route_id",

    # Reward intermediates (spec 01 §14.1)
    "delay_change_seconds",
    "route_outcome", "outcome",
    "next_tc_headway_seconds", "headway_seconds",
    "n_tc_occupations_after_t",
    "T_next_occ", "T_clear",
    "arr_delay_future",
    "route_release_time",

    # Reward outputs (should never appear in state)
    "r_delay", "r_throughput", "r_headway", "r_wait", "r_total",
    "r_delay_raw", "r_throughput_raw", "r_headway_raw", "r_wait_raw",

    # Forbidden focal markers on graph nodes
    "is_focal_signal", "is_focal_route",

    # Forbidden schedule details
    "planned_end_signal", "planned_signal",

    # Forbidden future-looking train info
    "actual_next_tc", "next_actual_timestamp",
    "future_delay_seconds",
}


# ============================================================
# Public exception type
# ============================================================

class LeakAuditError(AssertionError):
    """Raised when a snapshot fails any leak audit check.

    Subclasses AssertionError so existing `assert` patterns work, but the
    custom type lets calling code distinguish leak errors from other
    AssertionErrors.
    """
    pass


# ============================================================
# Main audit function
# ============================================================

def assert_no_leak(
    snapshot: dict,
    sample_meta: dict,
    t_ns: int,
    *,
    skip_checks: Optional[set[str]] = None,
) -> bool:
    """Run all 7 spec 02 §7 checks. Raise LeakAuditError on any violation.

    Args:
        snapshot: dict-like with keys
            'center'                : {'type': 'track', 'id': <tc_id>}
            'state_nodes_track'     : list[dict]
            'state_nodes_signal'    : list[dict]
            'state_nodes_route'     : list[dict]
            'state_nodes_train'     : list[dict] (each with is_focal bool)
            'state_event_tokens'    : list of dicts/tuples with time_delta_s
            'state_schedule_outlook': list[dict]
            'state_special_flags_meta': dict (declares f_trts_pressed_source)
        sample_meta: dict containing focal_train, focal_train_current_tc, etc.
        t_ns: decision time in nanoseconds (for completeness; not strictly used)
        skip_checks: optional set of check names to skip (e.g., during
            partial-snapshot dev work). Names: {'subgraph_center',
            'no_focal_graph_flags', 'banned_fields', 'schedule_outlook',
            'flags_source', 'temporal_causality', 'one_focal_train'}

    Returns:
        True on full pass. Raises LeakAuditError otherwise.
    """
    skip_checks = skip_checks or set()

    # ============================================================
    # Check 1: Subgraph centering — must center on focal_train.current_tc
    # ============================================================
    if "subgraph_center" not in skip_checks:
        center = snapshot.get("center", {})
        center_type = center.get("type")
        if center_type != "track":
            raise LeakAuditError(
                f"[Check 1: subgraph_center] subgraph must center on a TRACK node, "
                f"got type={center_type!r}. "
                f"Centering on signal/route would imply focal_signal leakage."
            )
        expected_tc = sample_meta.get("focal_train_current_tc")
        actual_tc = center.get("id")
        if expected_tc is not None and actual_tc != expected_tc:
            raise LeakAuditError(
                f"[Check 1: subgraph_center] expected center on "
                f"focal_train.current_tc={expected_tc!r}, got {actual_tc!r}. "
                f"Spec 01 §17.5.4: subgraph MUST center on focal_train, never on focal_signal."
            )

    # ============================================================
    # Check 2: No is_focal_signal / is_focal_route on graph nodes
    # ============================================================
    if "no_focal_graph_flags" not in skip_checks:
        for sig_node in snapshot.get("state_nodes_signal", []):
            forbidden = {"is_focal_signal", "is_focal", "is_focal_node"} & set(sig_node.keys())
            if forbidden:
                raise LeakAuditError(
                    f"[Check 2: no_focal_graph_flags] signal node "
                    f"{sig_node.get('signal_id', '?')} has forbidden flag(s): {forbidden}. "
                    f"Spec 01 §17.5.4: only is_focal_train allowed; "
                    f"is_focal_signal/is_focal_route are forbidden."
                )
        for r_node in snapshot.get("state_nodes_route", []):
            forbidden = {"is_focal_route", "is_focal", "is_focal_node", "is_chosen"} & set(r_node.keys())
            if forbidden:
                raise LeakAuditError(
                    f"[Check 2: no_focal_graph_flags] route node "
                    f"{r_node.get('route_id', '?')} has forbidden flag(s): {forbidden}. "
                    f"`is_chosen` is the action label, never a state input."
                )
        for t_node in snapshot.get("state_nodes_track", []):
            forbidden = {"is_focal_track"} & set(t_node.keys())
            if forbidden:
                raise LeakAuditError(
                    f"[Check 2: no_focal_graph_flags] track node "
                    f"{t_node.get('track_id', '?')} has forbidden flag(s): {forbidden}."
                )

    # ============================================================
    # Check 3: BANNED_STATE_FIELDS scan
    # ============================================================
    if "banned_fields" not in skip_checks:
        _scan_banned_fields(snapshot)

    # ============================================================
    # Check 4: Schedule outlook uses platform_id (int), not signal IDs
    # ============================================================
    if "schedule_outlook" not in skip_checks:
        for tr in snapshot.get("state_schedule_outlook", []):
            if "planned_end_signal" in tr or "planned_signal" in tr:
                raise LeakAuditError(
                    f"[Check 4: schedule_outlook] schedule outlook row contains "
                    f"forbidden signal field; only planned_platform (int 1-6) allowed. "
                    f"Got: {set(tr.keys())}"
                )
            p = tr.get("planned_platform")
            if p is not None and not isinstance(p, (int, bool)):
                # bool is a subclass of int; reject other types like str signal ID
                raise LeakAuditError(
                    f"[Check 4: schedule_outlook] planned_platform must be int 1-6 or None, "
                    f"got type={type(p).__name__} value={p!r}. "
                    f"Spec 01 §17.5.4: never use signal IDs in schedule outlook."
                )
            if p is not None and not (1 <= int(p) <= 6):
                raise LeakAuditError(
                    f"[Check 4: schedule_outlook] planned_platform must be in 1..6, got {p}"
                )

    # ============================================================
    # Check 5: f_trts_pressed source declaration
    # ============================================================
    if "flags_source" not in skip_checks:
        flags_meta = snapshot.get("state_special_flags_meta", {})
        if "f_trts_pressed_source" in flags_meta:
            src = flags_meta["f_trts_pressed_source"]
            if src not in {"planned_platform", "current_platform", "both"}:
                raise LeakAuditError(
                    f"[Check 5: flags_source] f_trts_pressed must use "
                    f"planned_platform OR current_platform OR both, got source={src!r}. "
                    f"Spec 01 §17.5.4: focal_signal's platform is FORBIDDEN as source."
                )

    # ============================================================
    # Check 6: Temporal causality — all events have time_delta_s >= 0
    # ============================================================
    if "temporal_causality" not in skip_checks:
        for tok in snapshot.get("state_event_tokens", []):
            # tok may be dict or tuple; extract time_delta_s
            if isinstance(tok, dict):
                td = tok.get("time_delta_s", 0.0)
            elif isinstance(tok, (tuple, list)) and len(tok) >= 3:
                td = tok[2]
            else:
                continue
            if td is None:
                continue
            if td < 0:
                raise LeakAuditError(
                    f"[Check 6: temporal_causality] event token has "
                    f"time_delta_s={td} < 0 — event in the FUTURE relative to t. "
                    f"Spec 01 §B.0: state events must have time <= t."
                )

    # ============================================================
    # Check 7: Exactly 1 is_focal=True train node
    # ============================================================
    if "one_focal_train" not in skip_checks:
        train_nodes = snapshot.get("state_nodes_train", [])
        n_focal = sum(1 for tr in train_nodes if tr.get("is_focal", False))
        if n_focal != 1:
            raise LeakAuditError(
                f"[Check 7: one_focal_train] expected exactly 1 is_focal=True train node, "
                f"got {n_focal} (out of {len(train_nodes)} train nodes). "
                f"Spec 02 §4.6: is_focal_train marks the train this decision is about."
            )

    return True


# ============================================================
# Internal: BANNED_STATE_FIELDS recursive scan
# ============================================================

def _scan_banned_fields(snapshot: dict, _path: str = "snapshot") -> None:
    """Recursively scan state_* fields for any banned key.

    Only state_* keys are scanned (sample metadata can legitimately contain
    things like focal_signal, chosen_route_id — they're metadata not state).
    """
    state_only = {k: v for k, v in snapshot.items() if k.startswith("state_") or k == "center"}
    _scan_recursive(state_only, _path)


def _scan_recursive(obj: Any, path: str) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in BANNED_STATE_FIELDS:
                raise LeakAuditError(
                    f"[Check 3: banned_fields] banned field {k!r} found at {path}/{k}. "
                    f"See BANNED_STATE_FIELDS in spec 01 §14.1."
                )
            _scan_recursive(v, f"{path}/{k}")
    elif isinstance(obj, (list, tuple)):
        for i, item in enumerate(obj):
            _scan_recursive(item, f"{path}[{i}]")


# ============================================================
# Diagnostics (no-raise variants for debug runs)
# ============================================================

def collect_violations(
    snapshot: dict,
    sample_meta: dict,
    t_ns: int,
) -> list[str]:
    """Return list of all violations (instead of raising on first).

    Useful for batch debugging: scan many snapshots, collect all error
    messages, fix in one pass.
    """
    violations = []
    all_checks = {
        "subgraph_center", "no_focal_graph_flags", "banned_fields",
        "schedule_outlook", "flags_source", "temporal_causality",
        "one_focal_train",
    }
    for check in all_checks:
        try:
            assert_no_leak(
                snapshot, sample_meta, t_ns,
                skip_checks=all_checks - {check},
            )
        except LeakAuditError as e:
            violations.append(str(e))
    return violations
