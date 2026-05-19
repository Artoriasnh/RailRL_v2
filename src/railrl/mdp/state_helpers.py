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
        evs["time_ns"] = pd.to_datetime(evs["time"]).astype("int64")
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

    def extract(self, center_tc: str) -> dict[str, set[str]]:
        """BFS k hops from (track, center_tc).

        Returns:
            dict {
                'track':  set of track_ids in subgraph,
                'signal': set of signal_ids,
                'route':  set of route_ids,
                'train':  empty set (trains added separately by SnapshotBuilder),
            }
            and the center itself is always in 'track'.
        """
        nodes_by_type: dict[str, set[str]] = {
            "track": set(),
            "signal": set(),
            "route": set(),
            "train": set(),
        }
        visited: set[tuple] = set()
        frontier: deque = deque()
        start = ("track", center_tc)
        nodes_by_type["track"].add(center_tc)
        visited.add(start)
        frontier.append((start, 0))

        while frontier:
            (ntype, nid), depth = frontier.popleft()
            if depth >= self.n_hops:
                continue
            for nbr in self.view.neighbours(ntype, nid):
                nb_type, nb_id = nbr[0], nbr[1]
                if (nb_type, nb_id) in visited:
                    continue
                visited.add((nb_type, nb_id))
                nodes_by_type[nb_type].add(nb_id)
                frontier.append(((nb_type, nb_id), depth + 1))

        return nodes_by_type

    def filter_edges(self, nodes: dict[str, set[str]]) -> dict[str, pd.DataFrame]:
        """Filter the static edge tables to only edges among the subgraph nodes.

        Returns dict {edge_type → DataFrame with rows in the subgraph}.
        """
        out = {}
        # protects: signal → track
        e = self.view.edges["protects"]
        out["protects"] = e[
            e["signal_id"].astype(str).isin(nodes["signal"])
            & e["track_id"].astype(str).isin(nodes["track"])
        ]
        # connects: track ↔ track
        e = self.view.edges["connects"]
        out["connects"] = e[
            e["track_a"].astype(str).isin(nodes["track"])
            & e["track_b"].astype(str).isin(nodes["track"])
        ]
        # traverses: route → track
        e = self.view.edges["traverses"]
        out["traverses"] = e[
            e["route_id"].astype(str).isin(nodes["route"])
            & e["track_id"].astype(str).isin(nodes["track"])
        ]
        # starts_at: route → signal
        e = self.view.edges["starts_at"]
        out["starts_at"] = e[
            e["route_id"].astype(str).isin(nodes["route"])
            & e["signal_id"].astype(str).isin(nodes["signal"])
        ]
        # ends_at
        e = self.view.edges["ends_at"]
        out["ends_at"] = e[
            e["route_id"].astype(str).isin(nodes["route"])
            & e["signal_id"].astype(str).isin(nodes["signal"])
        ]
        # same_signal: route ↔ route
        e = self.view.edges["same_signal"]
        out["same_signal"] = e[
            e["route_a"].astype(str).isin(nodes["route"])
            & e["route_b"].astype(str).isin(nodes["route"])
        ]
        return out
