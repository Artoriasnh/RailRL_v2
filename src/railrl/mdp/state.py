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
from .state_history import (
    TrackOccupancyHistory, SignalAspectHistory, BerthHistory,
    MovementsLookup, EventTokenStream,
)


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

    Init once (loads static graph, static node tables, route index,
    histories, etc.), then call .build_snapshot(decision_row) many times.
    """
    static_view:       StaticGraphView
    static_nodes:      StaticNodeTables
    route_index:       RouteIndex
    train_lookup:      TrainStateLookup
    subgraph:          SubgraphExtractor
    # Round 3 — time-aware histories
    track_history:     Optional[TrackOccupancyHistory] = None
    signal_history:    Optional[SignalAspectHistory] = None
    berth_history:     Optional[BerthHistory] = None
    movements_lookup:  Optional[MovementsLookup] = None
    event_stream:      Optional[EventTokenStream] = None
    run_leak_audit:    bool = True

    @classmethod
    def build_default(cls, td_events: pd.DataFrame,
                       movements: Optional[pd.DataFrame] = None
                       ) -> "SnapshotBuilder":
        """Convenience constructor — load all static data, indices, histories."""
        view = StaticGraphView.load()
        nodes = StaticNodeTables.load()
        routes = pd.read_parquet(C.ROUTES_CLEAN_PARQUET)
        route_idx = RouteIndex(routes)
        train_lkp = TrainStateLookup.build(td_events)
        subgraph = SubgraphExtractor(view=view, n_hops=C.SUBGRAPH_HOPS)
        # Histories
        track_hist = TrackOccupancyHistory.build(td_events)
        # Build signal full_name → bare_id map from static node table
        fn_to_id = {str(r.get("full_name")): str(r.get("signal_id"))
                     for r in nodes.signal.values()
                     if r.get("full_name") is not None}
        signal_hist = SignalAspectHistory.build(td_events,
                                                  full_name_to_id=fn_to_id)
        berth_hist = BerthHistory.build(td_events)
        mv_lookup = MovementsLookup.build(movements) if movements is not None \
                    else MovementsLookup()
        ev_stream = EventTokenStream.build(td_events)
        return cls(
            static_view=view, static_nodes=nodes,
            route_index=route_idx, train_lookup=train_lkp,
            subgraph=subgraph,
            track_history=track_hist,
            signal_history=signal_hist,
            berth_history=berth_hist,
            movements_lookup=mv_lookup,
            event_stream=ev_stream,
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

        # 3. Collect other active trains in subgraph (multi-train support)
        other_train_ids = self._collect_other_active_trains(
            t_ns, focal_train, nodes_by_type["track"],
        )
        all_train_ids = [focal_train] + sorted(other_train_ids)

        # 4. Build per-type node feature lists
        nodes_track  = self._build_track_nodes(nodes_by_type["track"], t_ns,
                                                focal_train, decision)
        nodes_signal = self._build_signal_nodes(nodes_by_type["signal"], t_ns)
        nodes_route  = self._build_route_nodes(nodes_by_type["route"], t_ns,
                                                focal_train, decision)
        nodes_train  = self._build_train_nodes(
            focal_train, t_ns, decision, other_train_ids=other_train_ids,
        )

        # 5. Static edges (cast filtered DataFrames to list of structs)
        state_edges = self._format_edges(edges)

        # 6. Dynamic edges (at_berth, next_signal)
        dyn_edges = self._build_dynamic_edges(
            t_ns, all_train_ids,
            nodes_by_type["track"], nodes_by_type["signal"],
        )

        # 7. Event tokens — K=256 most-recent events restricted to subgraph
        #     assets (tracks + signals, in stable order matching node lists)
        track_keys = [n["track_id"] for n in nodes_track]
        # SignalAspectHistory keys by bare signal_id; EventTokenStream keys by
        # whatever TD `id` column shows (full_name like STD5040). Map bare →
        # full_name from static_nodes.signal so the event lookup hits.
        signal_keys: list[str] = []
        for n in nodes_signal:
            static_sig = self.static_nodes.signal.get(n["signal_id"], {})
            fn = static_sig.get("full_name")
            signal_keys.append(str(fn) if fn else n["signal_id"])
        subgraph_assets = track_keys + signal_keys
        event_tokens = self._build_event_tokens(t_ns, subgraph_assets)

        # 8. Schedule outlook (gbtt only, excludes focal_train)
        schedule_outlook = self._build_schedule_outlook(focal_train, t_ns)

        # 9. Special flags
        flags = self._build_special_flags(
            focal_train, focal_signal, t_ns, decision, nodes_route,
            nodes_track=nodes_track, nodes_signal=nodes_signal,
            all_train_ids=all_train_ids,
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
            "state_edges_at_berth":    dyn_edges["at_berth"],
            "state_edges_next_signal": dyn_edges["next_signal"],
            "state_event_tokens":       event_tokens,
            "state_schedule_outlook":   schedule_outlook,
            "state_special_flags":      flags,
            "state_special_flags_meta": flags_meta,
            "state_center":             center,
            # `center` (no prefix) is what the leak audit reads; we keep both
            # for downstream consumers that expect either name.
            "center":                   center,
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

        Per spec 02 §4.3, 18 features per node:
          static (4): track_id, n_routes_using, platform_id, platform_sub
          now (3): occupied_now, current_occupier_train_id, last_change_age_s
          per-window aggregates (10): occupancy_fraction_W, n_state_changes_W
              for W ∈ {1,5,10,15,30} min
          focal-path (1): on_focal_train_path
        """
        out = []
        focal_path_tcs = self._focal_path_tcs(focal_train, decision)
        th = self.track_history

        for tc_id in track_ids:
            static = self.static_nodes.track.get(tc_id, {})
            occupied = bool(th.occupied_now(tc_id, t_ns)) if th else False
            occupier = th.current_occupier(tc_id, t_ns) if th else None
            last_age = int(th.last_change_age_s(tc_id, t_ns)) if th else 0
            # Per-window aggregates
            win_fracs = {}
            win_changes = {}
            for w in C.TIME_WINDOWS_MINUTES:
                if th is None:
                    frac, n = 0.0, 0
                else:
                    frac, n = th.window_stats(tc_id, t_ns, w * 60.0)
                win_fracs[f"occupancy_fraction_{w}m"] = float(frac)
                win_changes[f"n_state_changes_{w}m"] = int(n)
            node = {
                "track_id":          tc_id,
                "n_routes_using":    int(static.get("n_routes_using", 0)),
                "platform_id":       _to_nullable_int(static.get("platform_id")),
                "platform_sub":      _to_str_or_none(static.get("platform_sub")),
                "occupied_now":      occupied,
                "current_occupier_train_id": occupier,
                **win_fracs,
                **win_changes,
                "last_change_age_s": last_age,
                "on_focal_train_path": tc_id in focal_path_tcs,
            }
            out.append(node)
        return out

    def _build_signal_nodes(self, signal_ids: set[str], t_ns: int) -> list[dict]:
        out = []
        sh = self.signal_history
        bh = self.berth_history
        for sig_id in signal_ids:
            static = self.static_nodes.signal.get(sig_id, {})
            restrictive = bool(sh.aspect_restrictive_now(sig_id, t_ns)) if sh else False
            last_age = int(sh.last_change_age_s(sig_id, t_ns)) if sh else 0
            # Per-window aggregates (fraction_red + n_changes)
            win_fracs = {}
            win_changes = {}
            for w in C.TIME_WINDOWS_MINUTES:
                if sh is None:
                    frac, n = 0.0, 0
                else:
                    frac, n = sh.window_stats(sig_id, t_ns, w * 60.0)
                win_fracs[f"aspect_fraction_red_{w}m"] = float(frac)
                win_changes[f"aspect_n_changes_{w}m"] = int(n)
            # Berth occupant — many signals double as platform-end berths;
            # look up the berth whose name == signal_id.
            berth_train, berth_age = (None, 0)
            if bh is not None:
                berth_train, berth_age = bh.berth_occupant_at(sig_id, t_ns)
            node = {
                "signal_id":             sig_id,
                "prefix":                _to_str_or_none(static.get("prefix")),
                "n_routes_from":         int(static.get("n_routes_from", 0)),
                "is_platform_end":       bool(static.get("is_platform_end", False)),
                "platform_id":           _to_nullable_int(static.get("platform_id")),
                "platform_direction":    _to_str_or_none(static.get("platform_direction")),
                "aspect_restrictive_now": restrictive,
                **win_fracs,
                **win_changes,
                "aspect_last_change_age_s": last_age,
                "current_berth_train_id": berth_train,
                "berth_dwell_age_s":      int(berth_age),
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
                            decision: dict,
                            *,
                            other_train_ids: Optional[set[str]] = None
                            ) -> list[dict]:
        """Per spec 02 §4.6 + §17.5 — is_focal=True ONLY for focal_train.

        Round 3: includes focal_train + any other_train_ids passed in. Order:
        focal first, then others sorted alphabetically (stable ordering).
        """
        nodes = [self._build_one_train_node(focal_train, t_ns, is_focal=True)]
        if other_train_ids:
            for tr in sorted(other_train_ids):
                if tr == focal_train:
                    continue
                nodes.append(self._build_one_train_node(tr, t_ns, is_focal=False))
        return nodes

    def _build_one_train_node(self, train_id: str, t_ns: int,
                                *, is_focal: bool) -> dict:
        """Build one train node feature dict."""
        current_tc = self.train_lookup.current_tc(train_id, t_ns)
        current_berth = self.train_lookup.current_berth(train_id, t_ns)
        time_in_berth = self.train_lookup.time_in_current_berth_s(train_id, t_ns) or 0

        hc_class = train_id[0] if train_id and len(train_id) >= 4 else "non_standard"
        if hc_class not in {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"}:
            hc_class = "non_standard"
        elif hc_class in {"7", "8"}:
            hc_class = "other"

        # Round 3: planned_platform + scheduled_delta_s from Movements gbtt
        planned_plat: Optional[int] = None
        sched_delta: Optional[int] = None
        if self.movements_lookup is not None:
            planned_plat = self.movements_lookup.planned_platform(train_id, t_ns)
            sched_delta = self.movements_lookup.scheduled_delta_s(train_id, t_ns)

        # current_platform — derived from current_tc's platform_id (if any)
        cur_plat: Optional[int] = None
        if current_tc is not None:
            static_tc = self.static_nodes.track.get(current_tc, {})
            cur_plat = _to_nullable_int(static_tc.get("platform_id"))

        # recent_panel_requests_count — PRs in last 5 minutes (300s)
        recent_pr = 0
        if self.berth_history is not None:
            recent_pr = int(self.berth_history.recent_pr_count(train_id, t_ns, 300.0))

        return {
            "train_id":                    train_id,
            "is_focal":                    bool(is_focal),  # ⭐ spec 02 §4.6
            "headcode_class":              hc_class,
            "current_tc":                  current_tc or "",
            "current_berth":               current_berth or "",
            "current_platform":            cur_plat,
            "planned_platform":            planned_plat,
            "time_in_current_berth_s":     int(time_in_berth),
            "scheduled_delta_s":           int(sched_delta) if sched_delta is not None else 0,
            "recent_panel_requests_count": int(recent_pr),
        }

    # ------------------------------------------------------------
    # Edges
    # ------------------------------------------------------------

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
        # Ensure all 6 keys are present (extract may yield empty frames)
        for k in ("connects","traverses","starts_at","ends_at","protects","same_signal"):
            out.setdefault(k, [])
        return out

    # ------------------------------------------------------------
    # Event tokens, schedule outlook, special flags
    # ------------------------------------------------------------

    def _build_event_tokens(self, t_ns: int,
                              subgraph_assets: list[str]) -> list[dict]:
        """K=256 most-recent events restricted to subgraph assets.

        Each token: dict(asset_idx, state, time_delta_s).
        asset_idx is the position in `subgraph_assets` (NOT a global index —
        the encoder will look up the corresponding node embedding by index).
        """
        if self.event_stream is None or not subgraph_assets:
            return []
        tokens = self.event_stream.slice_last_k(
            subgraph_assets, t_ns, k=C.EVENT_TOKEN_K,
        )
        return [{"asset_idx": int(aidx), "state": int(st),
                  "time_delta_s": float(dt)} for aidx, st, dt in tokens]

    def _build_schedule_outlook(self, focal_train: str, t_ns: int) -> list[dict]:
        """Top-K=5 upcoming trains (excluding focal_train) from Movements gbtt.

        Per spec 02 §4.9 + spec 01 §17.5: gbtt only, planned_platform is int
        1-6 or None — NEVER a signal ID.
        """
        if self.movements_lookup is None:
            return []
        return self.movements_lookup.schedule_outlook(
            t_ns,
            k=C.SCHEDULE_OUTLOOK_TOPK,
            lookahead_s=C.SCHEDULE_LOOKAHEAD_MIN * 60.0,
            exclude_train=focal_train,
        )

    # ------------------------------------------------------------
    # Dynamic edges (at_berth, next_signal) + multi-train collection
    # ------------------------------------------------------------

    def _collect_other_active_trains(self, t_ns: int, focal_train: str,
                                       subgraph_tcs: set[str],
                                       cap: Optional[int] = None) -> set[str]:
        """Find other trains whose current_tc lies in the subgraph at t_ns.

        Caps at `cap` (default = C.MAX_TRAINS_PADDED - 1, since focal counts).
        """
        if cap is None:
            cap = max(0, C.MAX_TRAINS_PADDED - 1)
        if self.track_history is None:
            return set()
        candidates: set[str] = set()
        for tc_id in subgraph_tcs:
            occ = self.track_history.current_occupier(tc_id, t_ns)
            if occ is None or occ == focal_train:
                continue
            candidates.add(occ)
            if len(candidates) >= cap:
                break
        return candidates

    def _build_dynamic_edges(self, t_ns: int, all_train_ids: list[str],
                              subgraph_tcs: set[str],
                              subgraph_signals: set[str]
                              ) -> dict[str, list[dict]]:
        """Compute at_berth (train→track) and next_signal (train→signal).

        at_berth: train_id sits in current_tc (track). Edge: (train, tc).
        next_signal: train_id's next likely signal — derived from BerthHistory
            (the signal whose berth this train currently occupies).
        """
        at_berth_edges: list[dict] = []
        next_signal_edges: list[dict] = []
        train_set = set(all_train_ids)
        # at_berth
        for tr in all_train_ids:
            cur_tc = self.train_lookup.current_tc(tr, t_ns)
            if cur_tc is None or cur_tc not in subgraph_tcs:
                continue
            at_berth_edges.append({"src": tr, "dst": cur_tc, "order": -1})
        # next_signal — invert berth occupancy lookup
        if self.berth_history is not None:
            for sig_id in subgraph_signals:
                occ, _age = self.berth_history.berth_occupant_at(sig_id, t_ns)
                if occ is None or occ not in train_set:
                    continue
                next_signal_edges.append({"src": occ, "dst": sig_id, "order": -1})
        return {
            "at_berth":    at_berth_edges,
            "next_signal": next_signal_edges,
        }

    # ------------------------------------------------------------
    # Special flags (8 flags per spec 02 §4.10)
    # ------------------------------------------------------------

    def _build_special_flags(self, focal_train: str, focal_signal: str, t_ns: int,
                              decision: dict, nodes_route: list[dict],
                              *,
                              nodes_track: Optional[list[dict]] = None,
                              nodes_signal: Optional[list[dict]] = None,
                              all_train_ids: Optional[list[str]] = None,
                              ) -> dict:
        """Compute 8 special flags per spec 02 §4.10 / special_flags.py."""
        candidate_route_ids = decision.get("candidate_route_ids", []) or []
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

        # Round 3: real TC occupancy + platform occupancy
        tc_occ_now: dict[str, bool] = {}
        if nodes_track:
            for n in nodes_track:
                tc_occ_now[n["track_id"]] = bool(n.get("occupied_now", False))

        # Platform occupancy: any track in nodes_track with this platform_id occupied
        plat_occ_now: dict[int, bool] = {}
        if nodes_track:
            for n in nodes_track:
                pid = n.get("platform_id")
                if pid is None:
                    continue
                if bool(n.get("occupied_now", False)):
                    plat_occ_now[int(pid)] = True
                plat_occ_now.setdefault(int(pid), False)

        # planned_platform + current_platform of focal_train
        planned_plat = None
        if self.movements_lookup is not None:
            planned_plat = self.movements_lookup.planned_platform(focal_train, t_ns)
        cur_plat = None
        cur_tc = self.train_lookup.current_tc(focal_train, t_ns) if self.train_lookup else None
        if cur_tc is not None:
            cur_plat = _to_nullable_int(
                self.static_nodes.track.get(cur_tc, {}).get("platform_id")
            )

        n_other = max(0, (len(all_train_ids) - 1) if all_train_ids else 0)
        sched_delta = None
        if self.movements_lookup is not None:
            sched_delta = self.movements_lookup.scheduled_delta_s(focal_train, t_ns)

        flags = compute_all_flags(
            focal_train=focal_train,
            headcode_class_digit=(focal_train[0] if focal_train and len(focal_train) >= 4 else None),
            candidate_routes_first_tc=candidate_first_tc,
            candidate_route_cls_list=candidate_cls,
            candidate_end_platforms=candidate_end_plat,
            tc_occupancy_now=tc_occ_now,
            platform_occupancy_now=plat_occ_now,
            planned_platform=planned_plat,
            current_platform=cur_plat,
            trts_state_by_platform={},     # TRTS-by-platform: deferred to a
                                             # Round 4 enhancement; flag
                                             # defaults to 0 safely.
            n_other_active_trains=int(n_other),
            scheduled_delta_seconds=sched_delta,
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
