"""spec 02 §4 helpers — train state lookup + subgraph extraction.

Split out of state.py to keep modules manageable.

Two helper classes:
  TrainStateLookup  — at time t, look up focal_train's current_tc,
                      recent_tcs, current_platform, current_berth, etc.
                      All inputs are time≤t observable (no leak).
  SubgraphExtractor — 3-hop BFS from focal_train.current_tc producing
                      the per-snapshot node set + edge subset.
"""
from __future__ import annotations
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .. import config as C
from ..data.static_graph_view import StaticGraphView


# ============================================================
# TrainStateLookup — per-train time-bounded state queries
# ============================================================

@dataclass
class TrainStateLookup:
    """Per-train indexed lookup of TC/berth events from TD CA/CB/CC stream.

    Build ONCE per training run (loading TD takes minutes); query many times.
    All queries are time-bounded (time ≤ t).

    Internal data structure:
        by_train[trainid_filled] = sorted ndarray of (time_ns, tc_id, berth_id)
    """
    by_train: dict[str, np.ndarray] = field(default_factory=dict)

    @classmethod
    def build(cls, td_events: pd.DataFrame) -> "TrainStateLookup":
        """Build from TD events.

        Accepts the TD DataFrame; filters for CA/CB/CC + Track events
        with non-null trainid_filled.
        """
        # Track + CA/CB/CC events with valid trainid
        ev_mask = (
            td_events["trainid_filled"].notna()
            & (
                td_events["type"].isin(["Track", "CA", "CB", "CC"])
            )
        )
        evs = td_events.loc[ev_mask, ["time", "trainid_filled", "id", "type", "state",
                                       "from_berth", "to_berth"]].copy()
        # Force NANOSECONDS — td time is datetime64[us]; pd.to_datetime/.astype
        # would keep microseconds (pandas 2.x preserves unit), mismatching the
        # ns decision time t_ns. See state_history._to_ns_int64 for the full bug.
        evs["time_ns"] = evs["time"].values.astype("datetime64[ns]").astype("int64")
        evs["trainid_filled"] = evs["trainid_filled"].astype(str)
        evs = evs.sort_values("time_ns")

        by_train: dict[str, list[tuple]] = defaultdict(list)
        for _, r in evs.iterrows():
            t_ns = int(r["time_ns"])
            tid = r["trainid_filled"]
            etype = r["type"]
            # Build a (time_ns, event_type, tc_or_berth, state) tuple
            # Track:  tc_or_berth = id (TC); state from `state`
            # CA/CB/CC: tc_or_berth = to_berth (when state=1) / from_berth (state=0)
            if etype == "Track":
                tc = str(r["id"]) if pd.notna(r["id"]) else None
                state = int(r["state"]) if pd.notna(r["state"]) else None
                by_train[tid].append((t_ns, "Track", tc, state))
            else:
                # CA/CB/CC → use to_berth as "where train is now"
                berth = str(r["to_berth"]) if pd.notna(r["to_berth"]) else None
                by_train[tid].append((t_ns, etype, berth, None))

        # Convert to numpy structured arrays for fast bisect
        by_train_arr = {}
        for tid, evs_list in by_train.items():
            # Just keep a Python list — for now binary search via bisect works fine
            by_train_arr[tid] = sorted(evs_list)

        out = cls(by_train=by_train_arr)
        return out

    def current_tc(self, trainid: str, t_ns: int) -> Optional[str]:
        """Latest Track event with state=1 for this train, at time ≤ t."""
        evs = self.by_train.get(trainid)
        if not evs:
            return None
        # Walk backwards to find last Track state=1 ≤ t
        for (te, etype, val, state) in reversed(evs):
            if te > t_ns:
                continue
            if etype == "Track" and state == 1 and val:
                return val
        return None

    def recent_tcs(self, trainid: str, t_ns: int, n: int = 5) -> list[str]:
        """Last N distinct TCs occupied by this train (oldest → newest)."""
        evs = self.by_train.get(trainid)
        if not evs:
            return []
        out = []
        seen = set()
        for (te, etype, val, state) in reversed(evs):
            if te > t_ns:
                continue
            if etype == "Track" and state == 1 and val and val not in seen:
                out.append(val)
                seen.add(val)
                if len(out) >= n:
                    break
        return list(reversed(out))

    def current_berth(self, trainid: str, t_ns: int) -> Optional[str]:
        """Latest CA/CB/CC to_berth for this train, at time ≤ t."""
        evs = self.by_train.get(trainid)
        if not evs:
            return None
        for (te, etype, val, _state) in reversed(evs):
            if te > t_ns:
                continue
            if etype in ("CA", "CB", "CC") and val:
                return val
        return None

    def time_in_current_berth_s(self, trainid: str, t_ns: int) -> Optional[int]:
        """Seconds since the train entered its current berth."""
        evs = self.by_train.get(trainid)
        if not evs:
            return None
        for (te, etype, val, _state) in reversed(evs):
            if te > t_ns:
                continue
            if etype in ("CA", "CB", "CC") and val:
                return int((t_ns - te) / 1e9)
        return None


# ============================================================
# SubgraphExtractor — 3-hop BFS from focal_train.current_tc
# ============================================================

@dataclass
class SubgraphExtractor:
    """Extracts a 3-hop subgraph from the static heterogeneous graph,
    centered on focal_train.current_tc.

    Per spec 02 §4.2 + §17.5.4: subgraph MUST center on a track node
    (focal_train.current_tc), NEVER on focal_signal.

    Build once with the StaticGraphView; extract many times.
    """
    view: StaticGraphView
    n_hops: int = 3   # spec 02 §4.2 SUBGRAPH_HOPS

    seed_route_hops: int = 2   # how far candidate-route seeds expand (≤ n_hops)
    # Per-type node caps (= spec 03 §2.1 padding caps). The downstream loader
    # truncates to these anyway, so growing the BFS beyond them is pure wasted
    # per-node work (window_stats / event tokens / etc.). BFS is breadth-first
    # so we keep the NEAREST nodes; candidate-route seeds are added before the
    # cap so they're always retained. Set a cap to 0/None to disable it.
    cap_track:  int = 60
    cap_signal: int = 15
    cap_route:  int = 15

    def extract(self, center_tc: str,
                seed_routes: Optional[set[str]] = None) -> dict[str, set[str]]:
        """Multi-seed BFS subgraph.

        The subgraph is grown from TWO kinds of seeds:
          1. the center track (focal_train.current_tc) — expands `n_hops`.
          2. each candidate route in `seed_routes` — expands `seed_route_hops`.

        Seeding from candidate routes is essential because for ~3/4 of
        decisions the focal train sits on an off-network approach/holding
        track (e.g. T938) whose `current_tc` has NO graph neighbours — a
        center-only BFS would yield a degenerate 1-node subgraph. The
        candidate routes (the action set, observable at decision time) carry
        the destination network the train is being routed into.

        The CENTER for the leak audit stays `center_tc` (a track) — seeding
        from candidate routes does NOT change the center, so spec 02 §17.5.4
        (center on focal_train.current_tc, never focal_signal) still holds.

        Returns:
            dict { 'track', 'signal', 'route', 'train' } of id sets.
            center_tc is always in 'track'; seed routes always in 'route'.
        """
        nodes_by_type: dict[str, set[str]] = {
            "track": set(), "signal": set(), "route": set(), "train": set(),
        }
        visited: set[tuple] = set()
        frontier: deque = deque()

        # Seed 1: center track
        start = ("track", center_tc)
        nodes_by_type["track"].add(center_tc)
        visited.add(start)
        frontier.append((start, 0, self.n_hops))

        # Seed 2: candidate routes (shallower expansion)
        if seed_routes:
            for rid in seed_routes:
                key = ("route", str(rid))
                if key in visited:
                    continue
                nodes_by_type["route"].add(str(rid))
                visited.add(key)
                # Start at depth (n_hops - seed_route_hops) so they expand
                # exactly `seed_route_hops` levels under the shared n_hops cap.
                frontier.append((key, max(0, self.n_hops - self.seed_route_hops),
                                 self.n_hops))

        caps = {"track": self.cap_track, "signal": self.cap_signal,
                "route": self.cap_route, "train": 0}
        while frontier:
            (ntype, nid), depth, max_depth = frontier.popleft()
            if depth >= max_depth:
                continue
            for nbr in self.view.neighbours(ntype, nid):
                nb_type, nb_id = nbr[0], nbr[1]
                if (nb_type, nb_id) in visited:
                    continue
                # Cap per-type node count (keep nearest; candidate seeds were
                # added pre-loop so they survive). Bounds per-snapshot work.
                cap = caps.get(nb_type, 0)
                if cap and len(nodes_by_type[nb_type]) >= cap:
                    continue
                visited.add((nb_type, nb_id))
                nodes_by_type[nb_type].add(nb_id)
                frontier.append(((nb_type, nb_id), depth + 1, max_depth))

        return nodes_by_type

    # (src_col, dst_col, src_node_type, dst_node_type) per edge type
    _EDGE_SPEC = {
        "protects":    ("signal_id", "track_id", "signal", "track"),
        "connects":    ("track_a",   "track_b",  "track",  "track"),
        "traverses":   ("route_id",  "track_id", "route",  "track"),
        "starts_at":   ("route_id",  "signal_id","route",  "signal"),
        "ends_at":     ("route_id",  "signal_id","route",  "signal"),
        "same_signal": ("route_a",   "route_b",  "route",  "route"),
    }

    def _ensure_edge_tuples(self):
        """Precompute, ONCE, each edge type as a list of (src, dst, order, src_type,
        dst_type) plain-Python tuples. Avoids per-snapshot pandas .astype/.isin/
        iterrows (was ~10 ms/snapshot across filter_edges + _format_edges)."""
        if getattr(self, "_edge_tuples", None) is not None:
            return
        et: dict[str, list] = {}
        for ename, (sc, dc, _st, _dt) in self._EDGE_SPEC.items():
            e = self.view.edges.get(ename)
            if e is None or len(e) == 0:
                et[ename] = []
                continue
            src = e[sc].astype(str).to_numpy()
            dst = e[dc].astype(str).to_numpy()
            if "order" in e.columns:
                order = e["order"].fillna(-1).astype("int64").to_numpy()
            else:
                order = np.full(len(e), -1, dtype="int64")
            et[ename] = list(zip(src.tolist(), dst.tolist(), order.tolist()))
        self._edge_tuples = et

    def filter_edges(self, nodes: dict[str, set[str]]) -> dict[str, list]:
        """Filter precomputed edge tuples to those with both endpoints in the
        subgraph. Returns dict {edge_type → list[(src, dst, order)]} — pure
        set-membership, no pandas per call.
        """
        self._ensure_edge_tuples()
        out: dict[str, list] = {}
        for ename, (_sc, _dc, st, dt) in self._EDGE_SPEC.items():
            src_set = nodes[st]
            dst_set = nodes[dt]
            out[ename] = [(s, d, o) for (s, d, o) in self._edge_tuples[ename]
                          if s in src_set and d in dst_set]
        return out
