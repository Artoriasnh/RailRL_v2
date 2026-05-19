"""StaticGraphView — loaded-once view of the 4-node × 6-edge static graph.

Pure utility: loads all the static_graph/*.parquet edge tables into a dict
of DataFrames and builds an undirected adjacency map for traversal.

This class is reused by:
- `reward_calibration.py` (approach distance BFS)
- `reward_features.py` (approach distance BFS, fast path)
- `static_graph.py` (the builder)
- spec 02 `mdp/state.py` (future — for subgraph extraction)

NOT to be confused with v1's `snapshot.py` which contained subgraph BFS for
the deprecated binary task. The v2 subgraph extractor will live in
`src/railrl/mdp/state.py` per spec 02.
"""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from .. import config as C


# A node reference is a (type, id) pair. Type ∈ {'signal', 'track', 'route', 'train'}.
NodeRef = tuple[str, str]


@dataclass
class StaticGraphView:
    """Loaded-once view of the static graph: adjacency + edge tables.

    Pay the I/O cost ONCE, then reuse for many lookups.
    """
    adj:   dict[NodeRef, list[NodeRef]] = field(default_factory=lambda: defaultdict(list))
    edges: dict[str, pd.DataFrame]       = field(default_factory=dict)

    @classmethod
    def load(cls) -> "StaticGraphView":
        v = cls()

        v.edges["protects"]    = pd.read_parquet(C.EDGE_PROTECTS_PARQUET)
        v.edges["connects"]    = pd.read_parquet(C.EDGE_CONNECTS_PARQUET)
        v.edges["traverses"]   = pd.read_parquet(C.EDGE_TRAVERSES_PARQUET)
        v.edges["starts_at"]   = pd.read_parquet(C.EDGE_STARTS_AT_PARQUET)
        v.edges["ends_at"]     = pd.read_parquet(C.EDGE_ENDS_AT_PARQUET)
        v.edges["same_signal"] = pd.read_parquet(C.EDGE_SAME_SIGNAL_PARQUET)

        # Build undirected adjacency. `connects` and `same_signal` are stored
        # symmetric already (both directions present); protects/traverses/
        # starts_at/ends_at are directed → expand to both directions.
        for _, e in v.edges["protects"].iterrows():
            v.adj[("signal", e["signal_id"])].append(("track",  e["track_id"]))
            v.adj[("track",  e["track_id"])].append(("signal", e["signal_id"]))

        for _, e in v.edges["connects"].iterrows():
            v.adj[("track", e["track_a"])].append(("track", e["track_b"]))

        for _, e in v.edges["traverses"].iterrows():
            v.adj[("route", e["route_id"])].append(("track", e["track_id"]))
            v.adj[("track", e["track_id"])].append(("route", e["route_id"]))

        for _, e in v.edges["starts_at"].iterrows():
            v.adj[("route",  e["route_id"])].append(("signal", e["signal_id"]))
            v.adj[("signal", e["signal_id"])].append(("route", e["route_id"]))

        for _, e in v.edges["ends_at"].iterrows():
            v.adj[("route",  e["route_id"])].append(("signal", e["signal_id"]))
            v.adj[("signal", e["signal_id"])].append(("route", e["route_id"]))

        for _, e in v.edges["same_signal"].iterrows():
            v.adj[("route", e["route_a"])].append(("route", e["route_b"]))

        return v

    def neighbours(self, ntype: str, nid: str) -> list[NodeRef]:
        return self.adj.get((ntype, nid), [])
