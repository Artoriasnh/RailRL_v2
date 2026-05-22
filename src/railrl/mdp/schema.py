"""spec 02 §8 — snapshots_v2.parquet schema definitions.

Defines the pyarrow Schema for the final snapshots_v2.parquet table.
Used by the state builder (state.py) and by spec 04 / spec 05 readers.

Each row of snapshots_v2.parquet has 4 logical sections:

  IDENTITY (sample metadata; NEVER passed to model):
    sample_id, focal_train, focal_signal, t, pass_id,
    episode_idx, position_in_episode, is_last_in_episode,
    label, chosen_route_id, chosen_action_idx, candidate_route_ids,
    n_candidates, trigger_type

  REWARD (per spec 01 §17.1):
    outcome, approach_distance, delay_change_seconds,
    next_tc_headway_seconds, gate, r_*_raw, r_*, r_total

  STATE_NODES (4 lists of structs):
    state_nodes_track, state_nodes_signal, state_nodes_route,
    state_nodes_train

  STATE_GRAPH + SEQUENCE + OUTLOOK + FLAGS:
    state_edges_<type>, state_event_tokens, state_schedule_outlook,
    state_special_flags, state_special_flags_meta,
    state_center  (subgraph centering metadata)

The schema is **forward-extensible**: spec 03's encoder can add columns
without breaking spec 04's loader, as long as required cols are present.
"""
from __future__ import annotations
from typing import Any

# Don't import pyarrow at module load time — defer to functions
# so sandbox / minimal-install environments can still import this file.


# ============================================================
# Required columns (lock contract for downstream)
# ============================================================

# Identity columns — every row MUST have these
IDENTITY_COLS = [
    "sample_id",
    "focal_train",
    "focal_signal",
    "t",
    "pass_id",
    "episode_idx",
    "position_in_episode",
    "is_last_in_episode",
    "label",
    "chosen_route_id",
    "chosen_action_idx",
    "candidate_route_ids",
    "n_candidates",
    "trigger_type",
]

# Reward columns — set rows have all; wait rows have label='wait' with
# r_throughput_raw=0, r_headway_raw=0, r_wait_raw=-1.0 (per spec 01 §17.1)
REWARD_COLS = [
    "outcome",
    "approach_distance",
    "delay_change_seconds",
    "next_tc_headway_seconds",
    "gate",
    "r_delay_raw", "r_throughput_raw", "r_headway_raw", "r_wait_raw",
    "r_delay", "r_throughput", "r_headway", "r_wait",
    "r_total",
]

# State columns — passed to model (after leak audit)
STATE_NODE_COLS = [
    "state_nodes_track",
    "state_nodes_signal",
    "state_nodes_route",
    "state_nodes_train",
]

STATE_EDGE_COLS = [
    "state_edges_connects",
    "state_edges_traverses",
    "state_edges_starts_at",
    "state_edges_ends_at",
    "state_edges_protects",
    "state_edges_same_signal",
    "state_edges_at_berth",      # dynamic
    "state_edges_next_signal",   # dynamic
]

STATE_OTHER_COLS = [
    "state_event_tokens",
    "state_schedule_outlook",
    "state_special_flags",
    "state_special_flags_meta",
    "state_center",
]

ALL_COLS = (
    IDENTITY_COLS + REWARD_COLS + STATE_NODE_COLS +
    STATE_EDGE_COLS + STATE_OTHER_COLS
)


# ============================================================
# pyarrow Schema (built lazily)
# ============================================================

def get_arrow_schema():
    """Return the pyarrow.Schema for snapshots_v2.parquet.

    Lazy import: pyarrow may not be installed in sandbox; users on Windows
    with full deps have it via pyproject.toml.
    """
    import pyarrow as pa

    # ----- Identity -----
    identity_fields = [
        pa.field("sample_id", pa.int64()),
        pa.field("focal_train", pa.string()),
        pa.field("focal_signal", pa.string()),
        pa.field("t", pa.timestamp("ns")),
        pa.field("pass_id", pa.string()),
        pa.field("episode_idx", pa.int32()),
        pa.field("position_in_episode", pa.int32()),
        pa.field("is_last_in_episode", pa.bool_()),
        pa.field("label", pa.string()),
        pa.field("chosen_route_id", pa.string()),
        pa.field("chosen_action_idx", pa.int32()),
        pa.field("candidate_route_ids", pa.list_(pa.string())),
        pa.field("n_candidates", pa.int32()),
        pa.field("trigger_type", pa.string()),
    ]

    # ----- Reward (all floats nullable for wait or unmeasurable) -----
    reward_fields = [
        pa.field("outcome", pa.string()),
        pa.field("approach_distance", pa.float64()),
        pa.field("delay_change_seconds", pa.float64()),
        pa.field("next_tc_headway_seconds", pa.float64()),
        pa.field("gate", pa.float64()),
        pa.field("r_delay_raw", pa.float64()),
        pa.field("r_throughput_raw", pa.float64()),
        pa.field("r_headway_raw", pa.float64()),
        pa.field("r_wait_raw", pa.float64()),
        pa.field("r_delay", pa.float64()),
        pa.field("r_throughput", pa.float64()),
        pa.field("r_headway", pa.float64()),
        pa.field("r_wait", pa.float64()),
        pa.field("r_total", pa.float64()),
    ]

    # ----- State node types (list of structs) -----
    track_node = pa.struct([
        pa.field("track_id", pa.string()),
        pa.field("n_routes_using", pa.int32()),
        pa.field("platform_id", pa.int32()),       # nullable
        pa.field("platform_sub", pa.string()),     # 'A'/'middle'/'B'
        pa.field("occupied_now", pa.bool_()),
        pa.field("current_occupier_train_id", pa.string()),
        pa.field("occupancy_fraction_1m", pa.float32()),
        pa.field("occupancy_fraction_5m", pa.float32()),
        pa.field("occupancy_fraction_10m", pa.float32()),
        pa.field("occupancy_fraction_15m", pa.float32()),
        pa.field("occupancy_fraction_30m", pa.float32()),
        pa.field("n_state_changes_1m", pa.int32()),
        pa.field("n_state_changes_5m", pa.int32()),
        pa.field("n_state_changes_10m", pa.int32()),
        pa.field("n_state_changes_15m", pa.int32()),
        pa.field("n_state_changes_30m", pa.int32()),
        pa.field("last_change_age_s", pa.int64()),
        pa.field("on_focal_train_path", pa.bool_()),
    ])

    signal_node = pa.struct([
        pa.field("signal_id", pa.string()),
        pa.field("prefix", pa.string()),
        pa.field("n_routes_from", pa.int32()),
        pa.field("is_platform_end", pa.bool_()),
        pa.field("platform_id", pa.int32()),
        pa.field("platform_direction", pa.string()),
        pa.field("aspect_restrictive_now", pa.bool_()),
        pa.field("aspect_fraction_red_1m", pa.float32()),
        pa.field("aspect_fraction_red_5m", pa.float32()),
        pa.field("aspect_fraction_red_10m", pa.float32()),
        pa.field("aspect_fraction_red_15m", pa.float32()),
        pa.field("aspect_fraction_red_30m", pa.float32()),
        pa.field("aspect_n_changes_1m", pa.int32()),
        pa.field("aspect_n_changes_5m", pa.int32()),
        pa.field("aspect_n_changes_10m", pa.int32()),
        pa.field("aspect_n_changes_15m", pa.int32()),
        pa.field("aspect_n_changes_30m", pa.int32()),
        pa.field("aspect_last_change_age_s", pa.int64()),
        pa.field("current_berth_train_id", pa.string()),
        pa.field("berth_dwell_age_s", pa.int64()),
    ])

    route_node = pa.struct([
        pa.field("route_id", pa.string()),
        pa.field("prefix", pa.string()),
        pa.field("signal_no", pa.string()),
        pa.field("letter", pa.string()),
        pa.field("sub", pa.string()),
        pa.field("cls", pa.string()),
        pa.field("n_tc", pa.int32()),
        pa.field("end_platform_id", pa.int32()),
        # Derby_info physical features (per spec 03 §3.1)
        pa.field("length_m", pa.float32()),
        pa.field("ave_speed_mps", pa.float32()),
        pa.field("ave_grad", pa.float32()),
        pa.field("gap_time_s", pa.float32()),
        pa.field("n_points", pa.int32()),
        # Dynamic
        pa.field("currently_locked", pa.bool_()),
        pa.field("last_locked_age_s", pa.int64()),
        pa.field("n_tcs_occupied_by_other", pa.int32()),
        pa.field("n_tcs_occupied_by_focal", pa.int32()),
        pa.field("max_relative_position_of_occupied", pa.float32()),
        pa.field("min_age_of_occupation_s", pa.int64()),
        pa.field("in_candidate_set", pa.bool_()),
    ])

    train_node = pa.struct([
        pa.field("train_id", pa.string()),
        pa.field("is_focal", pa.bool_()),           # ⭐ ONLY focal flag allowed
        pa.field("headcode_class", pa.string()),
        pa.field("current_tc", pa.string()),
        pa.field("current_berth", pa.string()),
        pa.field("current_platform", pa.int32()),
        pa.field("planned_platform", pa.int32()),
        pa.field("time_in_current_berth_s", pa.int64()),
        pa.field("scheduled_delta_s", pa.int64()),
        pa.field("recent_panel_requests_count", pa.int32()),
    ])

    node_fields = [
        pa.field("state_nodes_track", pa.list_(track_node)),
        pa.field("state_nodes_signal", pa.list_(signal_node)),
        pa.field("state_nodes_route", pa.list_(route_node)),
        pa.field("state_nodes_train", pa.list_(train_node)),
    ]

    # ----- State edges (each is a list of (src, dst, attr) tuples) -----
    edge_struct = pa.struct([
        pa.field("src", pa.string()),
        pa.field("dst", pa.string()),
        pa.field("order", pa.int32()),  # only meaningful for `traverses`
    ])
    edge_fields = [pa.field(name, pa.list_(edge_struct)) for name in STATE_EDGE_COLS]

    # ----- Event tokens (K=256 per spec 01 §EVENT_TOKEN_K) -----
    event_token = pa.struct([
        pa.field("asset_idx", pa.int16()),
        pa.field("state", pa.int8()),
        pa.field("time_delta_s", pa.float32()),
    ])
    seq_field = pa.field("state_event_tokens", pa.list_(event_token))

    # ----- Schedule outlook (top-5 upcoming) -----
    outlook_struct = pa.struct([
        pa.field("train_id", pa.string()),
        pa.field("headcode_class", pa.string()),
        pa.field("eta_s", pa.int32()),
        pa.field("planned_platform", pa.int32()),
        pa.field("event_type", pa.string()),   # ARRIVAL / DEPARTURE (gbtt)
    ])
    outlook_field = pa.field("state_schedule_outlook", pa.list_(outlook_struct))

    # ----- Special flags (8 per spec 02 §4.10) -----
    flags_struct = pa.struct([
        pa.field("f_advance", pa.bool_()),
        pa.field("f_call_on", pa.bool_()),
        pa.field("f_platform_dev", pa.bool_()),
        pa.field("f_priority_compete", pa.bool_()),
        pa.field("f_late_train", pa.int32()),
        pa.field("f_unusual_id", pa.bool_()),
        pa.field("f_trts_pressed", pa.bool_()),
        pa.field("f_freight_class", pa.bool_()),
    ])
    flags_meta_struct = pa.struct([
        pa.field("f_trts_pressed_source", pa.string()),
        pa.field("audit_passed", pa.bool_()),
    ])

    flags_field = pa.field("state_special_flags", flags_struct)
    flags_meta_field = pa.field("state_special_flags_meta", flags_meta_struct)

    # ----- Subgraph center (for leak audit) -----
    center_struct = pa.struct([
        pa.field("type", pa.string()),   # always 'track'
        pa.field("id", pa.string()),     # track_id (focal_train.current_tc)
    ])
    center_field = pa.field("state_center", center_struct)

    return pa.schema(
        identity_fields + reward_fields + node_fields + edge_fields +
        [seq_field, outlook_field, flags_field, flags_meta_field, center_field]
    )


# ============================================================
# Row validation (cheap structural check)
# ============================================================

def validate_row(row: dict) -> tuple[bool, list[str]]:
    """Cheap row validation before writing to parquet.

    Returns (is_valid, error_list). Caller decides to skip or fail.
    """
    errors = []

    # Identity required
    for col in ["sample_id", "focal_train", "focal_signal", "t", "label"]:
        if col not in row:
            errors.append(f"missing identity col: {col}")

    # State required
    for col in STATE_NODE_COLS:
        if col not in row:
            errors.append(f"missing state col: {col}")

    # If label='set', chosen_route_id must be present
    if row.get("label") == "set":
        if not row.get("chosen_route_id"):
            errors.append("set row missing chosen_route_id")

    # n_candidates must match length of candidate_route_ids
    cands = row.get("candidate_route_ids", [])
    n_cands = row.get("n_candidates", -1)
    if isinstance(cands, list) and n_cands != len(cands):
        errors.append(
            f"n_candidates ({n_cands}) != len(candidate_route_ids) ({len(cands)})"
        )

    return (len(errors) == 0, errors)


# ============================================================
# Helper: write DataFrame → parquet with schema enforcement
# ============================================================

def write_snapshots(df, path, compression: str = "zstd") -> None:
    """Write a DataFrame (or list of dicts) as snapshots_v2.parquet.

    Lazy import pyarrow.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pandas as pd

    if isinstance(df, list):
        df = pd.DataFrame(df)

    schema = get_arrow_schema()
    table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
    pq.write_table(table, path, compression=compression)
