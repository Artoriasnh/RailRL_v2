"""spec 02 §4 — Snapshot builder (main entry).

Builds one snapshot per decision point, producing the per-node feature
vectors + edges + event tokens + schedule outlook + special flags that
the model encoder consumes.

Output: outputs/snapshots/snapshots_v2.parquet (one row per decision point).

Architecture:
    TrainStateLookup    — per-train current_tc / recent_tcs / berth lookups
    SubgraphExtractor   — 3-hop BFS from focal_train.current_tc
    SnapshotBuilder     — orchestrator (this module's main class)

Per spec 01 §17.5 strict separation:
    sample_meta (focal_signal, chosen_route_id, etc.) lives in IDENTITY cols
    state_* lives in STATE cols
    Every built snapshot is checked via assert_no_leak (spec 02 §7).

NOTE on per-window aggregates (Round 3 deferred):
    Track / Signal nodes have per-window aggregates (occupancy_fraction_W,
    n_state_changes_W, etc. for W ∈ {1, 5, 10, 15, 30} min). These require
    per-asset event scans which are expensive. This module's CURRENT
    implementation computes only the "now" features and leaves per-window
    placeholders at 0.0 / 0. The full implementation is a follow-up task
    (see IMPLEMENTATION_LOG Stage 3 Round 3 TODO).
"""
from __future__ import annotations
import time as _time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .. import config as C
from ..data.static_graph_view import StaticGraphView
from .action import RouteIndex, feasible_actions
from .special_flags import compute_all_flags, get_flag_sources
from .state_helpers import TrainStateLookup, SubgraphExtractor
from .leak_audit import assert_no_leak, LeakAuditError


# ============================================================
# Node-feature placeholders for per-window aggregates (Round 3 TODO)
# ============================================================

_NULL_PER_WINDOW_FRACTIONS = {f"occupancy_fraction_{w}m": 0.0
                               for w in (1, 5, 10, 15, 30)}
_NULL_PER_WINDOW_AGGRS_TRACK = {f"n_state_changes_{w}m": 0
                                  for w in (1, 5, 10, 15, 30)}
_NULL_PER_WINDOW_AGGRS_SIGNAL = {f"aspect_n_changes_{w}m": 0
                                   for w in (1, 5, 10, 15, 30)}
_NULL_ASPECT_RED_FRAC = {f"aspect_fraction_red_{w}m": 0.0
                          for w in (1, 5, 10, 15, 30)}


# ============================================================
# Static node tables (loaded once)
# ============================================================

@dataclass
class StaticNodeTables:
    """Per-asset static info loaded from outputs/static_graph/nodes_*.parquet.

    Built once at SnapshotBuilder init.
    """
    track:  dict[str, dict]  # track_id → {n_routes_using, platform_id, platform_sub}
    signal: dict[str, dict]  # signal_id → {prefix, n_routes_from, is_platform_end, ...}
    route:  dict[str, dict]  # route_id  → {prefix, signal_no, letter, sub, cls, n_tc,
                              #               end_platform_id, length_m, ave_speed_mps,
                              #               ave_grad, gap_time_s, n_points}

    @classmethod
    def load(cls) -> "StaticNodeTables":
        ntrack = pd.read_parquet(C.NODE_TRACK_PARQUET)
        nsignal = pd.read_parquet(C.NODE_SIGNAL_PARQUET)
        nroute = pd.read_parquet(C.NODE_ROUTE_PARQUET)
        return cls(
            track={str(r["track_id"]): r.to_dict() for _, r in ntrack.iterrows()},
            signal={str(r["signal_id"]): r.to_dict() for _, r in nsignal.iterrows()},
            route={str(r["route_id"]): r.to_dict() for _, r in nroute.iterrows()},
        )


# ============================================================
# Snapshot builder
# ============================================================

@dataclass
class SnapshotBuilder:
    """Build one snapshot per decision point.

    Init once (loads static graph, static node tables, route index, etc.),
    then call .build_snapshot(decision_row) many times.
    """
    static_view:       StaticGraphView
    static_nodes:      StaticNodeTables
    route_index:       RouteIndex
    train_lookup:      TrainStateLookup
    subgraph:          SubgraphExtractor
    run_leak_audit:    bool = True

    @classmethod
    def build_default(cls, td_events: pd.DataFrame) -> "SnapshotBuilder":
        """Convenience constructor — load all static data and indices."""
        view = StaticGraphView.load()
        nodes = StaticNodeTables.load()
        routes = pd.read_parquet(C.ROUTES_CLEAN_PARQUET)
        route_idx = RouteIndex(routes)
        train_lkp = TrainStateLookup.build(td_events)
        subgraph = SubgraphExtractor(view=view, n_hops=C.SUBGRAPH_HOPS)
        return cls(
            static_view=view, static_nodes=nodes,
            route_index=route_idx, train_lookup=train_lkp,
            subgraph=subgraph,
        )

    # ------------------------------------------------------------
    # Main entry: build one snapshot
    # ------------------------------------------------------------

    def build_snapshot(self, decision: dict, *, sample_id: int = -1) -> Optional[dict]:
        """Build one snapshot from a decision_points row + sample metadata.

        Args:
            decision: dict-like with keys focal_train, focal_signal, t,
                       label, chosen_route_id, candidate_route_ids,
                       n_candidates, trigger_type, pass_id,
                       episode_idx, position_in_episode, is_last_in_episode
            sample_id: assigned sample identifier

        Returns:
            dict matching schema.ALL_COLS, or None if focal_train.current_tc
            cannot be determined (per spec 02 §11 Q4 — skip these).
        """
        focal_train  = str(decision["focal_train"])
        focal_signal = str(decision["focal_signal"])
        t = pd.Timestamp(decision["t"])
        t_ns = int(t.value)
        label = decision["label"]

        # 1. Locate focal_train's current_tc (REQUIRED — no leak)
        current_tc = self.train_lookup.current_tc(focal_train, t_ns)
        if current_tc is None:
            return None  # caller logs to skipped_no_tc.jsonl

        # 2. Extract 3-hop subgraph centered on current_tc
        nodes_by_type = self.subgraph.extract(current_tc)
        edges = self.subgraph.filter_edges(nodes_by_type)

        # 3. Build per-type node feature lists
        nodes_track  = self._build_track_nodes(nodes_by_type["track"], t_ns,
                                                focal_train, decision)
        nodes_signal = self._build_signal_nodes(nodes_by_type["signal"], t_ns)
        nodes_route  = self._build_route_nodes(nodes_by_type["route"], t_ns,
                                                focal_train, decision)
        nodes_train  = self._build_train_nodes(focal_train, t_ns, decision)

        # 4. Edges (cast filtered DataFrames to list of structs)
        state_edges = self._format_edges(edges)

        # 5. Event tokens (placeholder — actual K=256 slicing in Round 3)
        event_tokens = self._build_event_tokens(t_ns)

        # 6. Schedule outlook (placeholder — gbtt Movements join in Round 3)
        schedule_outlook = self._build_schedule_outlook(focal_train, t_ns)

        # 7. Special flags
        flags = self._build_special_flags(
            focal_train, focal_signal, t_ns, decision, nodes_route
        )
        flags_meta = {
            "f_trts_pressed_source": "planned_platform",  # locked per spec 01 §17.5.4
            "audit_passed": False,  # filled after leak audit
        }

        # 8. Center metadata (for leak audit + downstream HGT centering)
        center = {"type": "track", "id": current_tc}

        # 9. Assemble snapshot dict
        snapshot = {
            # Identity
            "sample_id": int(sample_id),
            "focal_train": focal_train,
            "focal_signal": focal_signal,
            "t": t,
            "pass_id": decision.get("pass_id", ""),
            "episode_idx": int(decision.get("episode_idx", -1)),
            "position_in_episode": int(decision.get("position_in_episode", -1)),
            "is_last_in_episode": bool(decision.get("is_last_in_episode", False)),
            "label": label,
            "chosen_route_id": decision.get("chosen_route_id"),
            "chosen_action_idx": int(decision.get("chosen_action_idx", -1)),
            "candidate_route_ids": decision.get("candidate_route_ids", []),
            "n_candidates": int(decision.get("n_candidates", 0)),
            "trigger_type": decision.get("trigger_type", ""),

            # Reward (will be joined from decision_rewards.parquet downstream;
            # leave NaN for now)
            "outcome": decision.get("outcome", None),
            "approach_distance": float("nan"),
            "delay_change_seconds": float("nan"),
            "next_tc_headway_seconds": float("nan"),
            "gate": float("nan"),
            "r_delay_raw": float("nan"),
            "r_throughput_raw": float("nan"),
            "r_headway_raw": float("nan"),
            "r_wait_raw": float("nan"),
            "r_delay": float("nan"),
            "r_throughput": float("nan"),
            "r_headway": float("nan"),
            "r_wait": float("nan"),
            "r_total": float("nan"),

            # State
            "state_nodes_track":  nodes_track,
            "state_nodes_signal": nodes_signal,
            "state_nodes_route":  nodes_route,
            "state_nodes_train":  nodes_train,
            "state_edges_connects":    state_edges["connects"],
            "state_edges_traverses":   state_edges["traverses"],
            "state_edges_starts_at":   state_edges["starts_at"],
            "state_edges_ends_at":     state_edges["ends_at"],
            "state_edges_protects":    state_edges["protects"],
            "state_edges_same_signal": state_edges["same_signal"],
            "state_edges_at_berth":    [],   # dynamic — Round 3 TODO
            "state_edges_next_signal": [],   # dynamic — Round 3 TODO
            "state_event_tokens":       event_tokens,
            "state_schedule_outlook":   schedule_outlook,
            "state_special_flags":      flags,
            "state_special_flags_meta": flags_meta,
            "state_center":             center,
        }

        # 10. Leak audit
        if self.run_leak_audit:
            sample_meta = {
                "focal_train": focal_train,
                "focal_train_current_tc": current_tc,
                "focal_signal": focal_signal,
            }
            try:
                assert_no_leak(snapshot, sample_meta, t_ns)
                flags_meta["audit_passed"] = True
                snapshot["state_special_flags_meta"] = flags_meta
            except LeakAuditError as e:
                # In production, log + skip. Here we re-raise so caller can
                # decide policy.
                raise

        return snapshot

    # ------------------------------------------------------------
    # Per-node feature builders
    # ------------------------------------------------------------

    def _build_track_nodes(self, track_ids: set[str], t_ns: int,
                             focal_train: str, decision: dict) -> list[dict]:
        """Build Track node feature dicts.

        Per spec 02 §4.3, 18 features per node.
        Per-window aggregates are placeholders (Round 3 TODO).
        """
        out = []
        # Compute focal train's candidate route TC set for `on_focal_train_path`
        focal_path_tcs = self._focal_path_tcs(focal_train, decision)

        for tc_id in track_ids:
            static = self.static_nodes.track.get(tc_id, {})
            node = {
                "track_id":          tc_id,
                "n_routes_using":    int(static.get("n_routes_using", 0)),
                "platform_id":       _to_nullable_int(static.get("platform_id")),
                "platform_sub":      _to_str_or_none(static.get("platform_sub")),
                "occupied_now":      False,                       # Round 3 from TD
                "current_occupier_train_id": None,                # Round 3
                **_NULL_PER_WINDOW_FRACTIONS,                     # Round 3
                **_NULL_PER_WINDOW_AGGRS_TRACK,                   # Round 3
                "last_change_age_s": 0,                           # Round 3
                "on_focal_train_path": tc_id in focal_path_tcs,
            }
            out.append(node)
        return out

    def _build_signal_nodes(self, signal_ids: set[str], t_ns: int) -> list[dict]:
        out = []
        for sig_id in signal_ids:
            static = self.static_nodes.signal.get(sig_id, {})
            node = {
                "signal_id":             sig_id,
                "prefix":                _to_str_or_none(static.get("prefix")),
                "n_routes_from":         int(static.get("n_routes_from", 0)),
                "is_platform_end":       bool(static.get("is_platform_end", False)),
                "platform_id":           _to_nullable_int(static.get("platform_id")),
                "platform_direction":    _to_str_or_none(static.get("platform_direction")),
                "aspect_restrictive_now": False,                   # Round 3
                **_NULL_ASPECT_RED_FRAC,                           # Round 3
                **_NULL_PER_WINDOW_AGGRS_SIGNAL,                   # Round 3
                "aspect_last_change_age_s": 0,                     # Round 3
                "current_berth_train_id": None,                    # Round 3
                "berth_dwell_age_s":      0,                       # Round 3
            }
            out.append(node)
        return out

    def _build_route_nodes(self, route_ids: set[str], t_ns: int,
                            focal_train: str, decision: dict) -> list[dict]:
        candidate_set = set(decision.get("candidate_route_ids", []) or [])
        out = []
        for rid in route_ids:
            static = self.static_nodes.route.get(rid, {})
            node = {
                "route_id":        rid,
                "prefix":          _to_str_or_none(static.get("prefix")),
                "signal_no":       _to_str_or_none(static.get("signal_no")),
                "letter":          _to_str_or_none(static.get("letter")),
                "sub":             _to_str_or_none(static.get("sub")),
                "cls":             _to_str_or_none(static.get("cls")),
                "n_tc":            int(static.get("n_tc", 0)),
                "end_platform_id": _to_nullable_int(static.get("end_platform_id")),
                # Derby_info physical features (spec 03 §3.1)
                "length_m":        float(static.get("length_m", 0.0) or 0.0),
                "ave_speed_mps":   float(static.get("ave_speed_mps", 0.0) or 0.0),
                "ave_grad":        float(static.get("ave_grad", 0.0) or 0.0),
                "gap_time_s":      float(static.get("gap_time_s", 0.0) or 0.0),
                "n_points":        int(static.get("n_points", 0) or 0),
                # Dynamic placeholders (Round 3)
                "currently_locked":            False,
                "last_locked_age_s":           0,
                "n_tcs_occupied_by_other":     0,
                "n_tcs_occupied_by_focal":     0,
                "max_relative_position_of_occupied": 0.0,
                "min_age_of_occupation_s":     0,
                "in_candidate_set":            rid in candidate_set,
            }
            out.append(node)
        return out

    def _build_train_nodes(self, focal_train: str, t_ns: int,
                            decision: dict) -> list[dict]:
        """Per spec 02 §4.6 + §17.5 — is_focal=True ONLY for focal_train.

        Round 2 implementation: only the focal_train node. Other active
        trains in subgraph deferred to Round 3.
        """
        current_tc = self.train_lookup.current_tc(focal_train, t_ns)
        current_berth = self.train_lookup.current_berth(focal_train, t_ns)
        time_in_berth = self.train_lookup.time_in_current_berth_s(focal_train, t_ns) or 0

        hc_class = focal_train[0] if focal_train and len(focal_train) >= 4 else "non_standard"
        if hc_class not in {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"}:
            hc_class = "non_standard"
        elif hc_class in {"7", "8"}:
            hc_class = "other"

        focal_node = {
            "train_id":                    focal_train,
            "is_focal":                    True,   # ⭐ spec 02 §4.6
            "headcode_class":              hc_class,
            "current_tc":                  current_tc or "",
            "current_berth":               current_berth or "",
            "current_platform":            None,    # Round 3: derive from current_tc
            "planned_platform":            None,    # Round 3: from Movements gbtt
            "time_in_current_berth_s":     int(time_in_berth),
            "scheduled_delta_s":           0,       # Round 3
            "recent_panel_requests_count": 0,       # Round 3
        }
        return [focal_node]

    def _format_edges(self, edges: dict[str, pd.DataFrame]) -> dict[str, list]:
        """Convert filtered edge DataFrames to lists of (src, dst, order) tuples."""
        out = {}
        for ename, edf in edges.items():
            rows = []
            for _, e in edf.iterrows():
                if ename == "connects":
                    src, dst = str(e["track_a"]), str(e["track_b"])
                elif ename == "traverses":
                    src, dst = str(e["route_id"]), str(e["track_id"])
                elif ename == "starts_at":
                    src, dst = str(e["route_id"]), str(e["signal_id"])
                elif ename == "ends_at":
                    src, dst = str(e["route_id"]), str(e["signal_id"])
                elif ename == "protects":
                    src, dst = str(e["signal_id"]), str(e["track_id"])
                elif ename == "same_signal":
                    src, dst = str(e["route_a"]), str(e["route_b"])
                else:
                    continue
                order = int(e["order"]) if "order" in e and pd.notna(e["order"]) else -1
                rows.append({"src": src, "dst": dst, "order": order})
            out[ename] = rows
        return out

    def _build_event_tokens(self, t_ns: int) -> list[dict]:
        """K=256 last events with time_ns < t (Round 3 TODO: actual implementation).

        For now: empty list. Round 3 will load event_tokens.parquet + slice.
        """
        return []  # Round 3: load + slice last K=256 with time < t

    def _build_schedule_outlook(self, focal_train: str, t_ns: int) -> list[dict]:
        """Top-5 upcoming trains from Movements gbtt (Round 3 TODO).

        Per spec 02 §4.9: gbtt only, no actual.
        """
        return []  # Round 3: load Movements + slice

    def _build_special_flags(self, focal_train: str, focal_signal: str, t_ns: int,
                              decision: dict, nodes_route: list[dict]) -> dict:
        """Compute 8 special flags per spec 02 §4.10 / special_flags.py."""
        candidate_route_ids = decision.get("candidate_route_ids", []) or []
        # Match candidate route_ids → first TC and end_platform
        candidate_first_tc = []
        candidate_cls = []
        candidate_end_plat = []
        for rid in candidate_route_ids:
            static = self.static_nodes.route.get(str(rid), {})
            tcs = static.get("track_sections")
            if isinstance(tcs, (list, np.ndarray)) and len(tcs) > 0:
                candidate_first_tc.append(str(tcs[0]))
            else:
                candidate_first_tc.append("")
            candidate_cls.append(_to_str_or_none(static.get("cls")) or "")
            candidate_end_plat.append(_to_nullable_int(static.get("end_platform_id")))

        flags = compute_all_flags(
            focal_train=focal_train,
            headcode_class_digit=(focal_train[0] if focal_train and len(focal_train) >= 4 else None),
            candidate_routes_first_tc=candidate_first_tc,
            candidate_route_cls_list=candidate_cls,
            candidate_end_platforms=candidate_end_plat,
            tc_occupancy_now={},                  # Round 3: real TC occupancy
            platform_occupancy_now={},            # Round 3
            planned_platform=None,                # Round 3 from Movements
            current_platform=None,                # Round 3 derived
            trts_state_by_platform={},            # Round 3
            n_other_active_trains=0,              # Round 3
            scheduled_delta_seconds=None,         # Round 3
        )
        return flags

    def _focal_path_tcs(self, focal_train: str, decision: dict) -> set[str]:
        """TCs that lie on any of focal_train's candidate routes."""
        out: set[str] = set()
        for rid in (decision.get("candidate_route_ids") or []):
            static = self.static_nodes.route.get(str(rid), {})
            tcs = static.get("track_sections")
            if isinstance(tcs, (list, np.ndarray)):
                out.update(str(t) for t in tcs)
        return out


# ============================================================
# Internal helpers
# ============================================================

def _to_nullable_int(v) -> Optional[int]:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _to_str_or_none(v) -> Optional[str]:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return str(v)
