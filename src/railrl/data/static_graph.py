"""P2.3 Iteration 1 — Static heterogeneous-graph backbone.

Builds the *time-invariant* skeleton of the heterogeneous graph that every
per-decision-event snapshot will hang off. Inputs:

    routes_clean.parquet      (P2.2)
    tracks_inventory.parquet  (P2.2)
    signals_inventory.parquet (P2.2)
    platform_end_signals.csv  (project data)
    platform_tc_map.csv       (project data)

Outputs (all parquet + one JSON summary, in outputs/p2_data_eng/static_graph/):

    NODES
      nodes_track.parquet     ~ 249 rows  (track_id, n_routes_using, platform_id, platform_sub)
      nodes_signal.parquet    ~ 100 rows  (signal_id, prefix, n_routes_from, is_platform_end, ...)
      nodes_route.parquet     ~ 277 rows  (route_id, prefix, signal_no, letter, sub, cls, n_tc, end_platform_id)

    EDGES (edge tables; every row is one edge)
      edges_protects.parquet     ~  80–100 rows  (signal_id  -> track_id)         K=1 berth-track only
      edges_connects.parquet     ~  varies      (track_a, track_b, undirected)    derived from route TC orderings
      edges_traverses.parquet    ~ 1700 rows    (route_id   -> track_id, order)
      edges_starts_at.parquet    ~  300 rows    (route_id   -> signal_id)
      edges_ends_at.parquet      ~  300 rows    (route_id   -> signal_id)
      edges_same_signal.parquet  ~  varies      (route_a, route_b)                 routes sharing start_signal

    static_graph_summary.json
"""
from __future__ import annotations
import json
import time
from itertools import combinations
from pathlib import Path

import pandas as pd

from .. import config as C
from . import sop_parser
from . import derby_info as _derby_info


# ------------ helpers ------------

def _load_platform_end_signals() -> pd.DataFrame:
    df = pd.read_csv(C.PLATFORM_END_SIGNALS_CSV, comment="#")
    df = df.dropna(subset=["platform_id", "direction", "signal_id"])
    df["platform_id"] = df["platform_id"].astype(int)
    df["signal_id"] = df["signal_id"].astype(int).astype(str)
    return df[["platform_id", "direction", "signal_id"]]


def _load_platform_tc_map() -> pd.DataFrame:
    df = pd.read_csv(C.PLATFORM_TC_MAP_CSV, comment="#")
    df = df.dropna(subset=["tc_id", "platform_id"])
    df["platform_id"] = df["platform_id"].astype(int)
    df["sub_section"] = df["sub_section"].fillna("middle")
    return df[["tc_id", "platform_id", "sub_section"]]


# ------------ NODE TABLES ------------

def build_track_nodes(routes: pd.DataFrame, tracks_inv: pd.DataFrame,
                      platform_tc: pd.DataFrame) -> pd.DataFrame:
    """Track nodes — scoped to route_to_tc TCs (= 249), with SOP cross-check.

    Scoping rationale (user decision): the model predicts routes, and every
    track that is part of any panel route is already in route_to_tc_all.csv.
    Tracks that exist only in the SOP (T886 etc., protected by block signals
    that have no outbound panel routes) are NOT attended to by the policy
    and would just be isolated nodes in the graph — so we exclude them.

    The `source` column is preserved for traceability:
      * both           = TC appears in both route_to_tc and SOP (= 225)
      * routes_only    = TC appears in route_to_tc but not in this SOP version (= 24)
      *  (sop_only is excluded by design — see above)

    Columns: track_id, source, n_routes_using, platform_id, platform_sub.
    """
    sop_set    = set(sop_parser.parse_tracks()["track_id"])
    routes_set = set(tracks_inv["track_id"])

    rows = []
    for tc in sorted(routes_set):                          # 249, the scope
        source = "both" if tc in sop_set else "routes_only"
        rows.append({"track_id": tc, "source": source})
    out = pd.DataFrame(rows)

    nu = tracks_inv[["track_id", "n_routes"]].rename(columns={"n_routes": "n_routes_using"})
    out = out.merge(nu, on="track_id", how="left")
    out["n_routes_using"] = out["n_routes_using"].fillna(0).astype(int)

    out = out.merge(platform_tc[["tc_id", "platform_id", "sub_section"]],
                    left_on="track_id", right_on="tc_id", how="left")
    out = out.drop(columns=["tc_id"]).rename(columns={"sub_section": "platform_sub"})

    cols = ["track_id", "source", "n_routes_using", "platform_id", "platform_sub"]
    return out[cols].sort_values("track_id").reset_index(drop=True)


def build_signal_nodes(routes: pd.DataFrame, signals_inv: pd.DataFrame,
                        platform_end: pd.DataFrame) -> pd.DataFrame:
    """SignalBerth nodes — SOP is the authoritative source (123 signals).

    Approach:
        1. Start from SOP signal list (parse_signals → signal_id, prefix, full_name)
        2. Merge in n_routes_from from signals_inventory (P2.2)
        3. Compute n_routes_to by counting routes that end at each signal
        4. Merge in platform-end metadata
        5. Signals not in SOP (e.g. X-prefix external boundary) are dropped
    """
    # 1) canonical asset list from SOP
    sop_signals = sop_parser.parse_signals()           # 123 rows
    out = sop_signals[["signal_id", "prefix", "full_name"]].copy()

    # 2) n_routes_from from P2.2 signals_inventory
    nf = signals_inv.rename(columns={"signal": "signal_id"})[["signal_id", "n_routes_from"]]
    out = out.merge(nf, on="signal_id", how="left")
    out["n_routes_from"] = out["n_routes_from"].fillna(0).astype(int)

    # 3) n_routes_to — count of end_signals matching this signal
    end_sig_counts = pd.Series(
        [s for lst in routes["end_signals"] for s in lst]
    ).value_counts().rename_axis("signal_id").reset_index(name="n_routes_to")
    out = out.merge(end_sig_counts, on="signal_id", how="left")
    out["n_routes_to"] = out["n_routes_to"].fillna(0).astype(int)

    # 4) platform-end metadata
    pe = platform_end.copy()
    pe["is_platform_end"] = True
    out = out.merge(pe.rename(columns={"direction": "platform_direction"}),
                    on="signal_id", how="left")
    out["is_platform_end"] = out["is_platform_end"].astype("boolean").fillna(False).astype(bool)

    cols = ["signal_id", "prefix", "full_name",
            "n_routes_from", "n_routes_to",
            "is_platform_end", "platform_id", "platform_direction"]
    return out[cols].sort_values("signal_id").reset_index(drop=True)


def build_trts_nodes() -> pd.DataFrame:
    """TRTS (Train Ready To Start) latches per platform sub-section.
    From SOP: 24 entries  (6 platforms × 2 subs × 2 directions = 24)."""
    return sop_parser.parse_trts()


def build_route_nodes(routes: pd.DataFrame, signal_nodes: pd.DataFrame,
                       platform_end: pd.DataFrame) -> pd.DataFrame:
    """Route nodes — propagate end-platform info, merge Derby_info physical features.

    Authoritative sources (per user decision, May 2026):
      * track_sections, n_tc, start/end signals  ←  route_to_tc_all.csv (already in `routes`)
      * length_m, ave_speed_mps, ave_grad, gap_time_s, n_points  ←  Derby_info.csv

    275 / 277 routes get physical features; the 2 ours-only routes
    (RDW5309A(M), RDY572A(M)) have NaN physical columns.
    """
    out = routes.copy()
    out["end_signal"] = out["end_signals"].apply(lambda lst: lst[0] if len(lst) > 0 else None)

    pe_lookup = platform_end.set_index("signal_id")["platform_id"]
    out["end_platform_id"] = out["end_signal"].map(pe_lookup)

    # Merge Derby_info physical features
    phys = _derby_info.load_route_physical_features()
    out = out.merge(phys, on="route_id", how="left")

    cols = ["route_id", "prefix", "signal_no", "letter", "sub", "cls",
            "n_tc", "end_signal", "end_platform_id",
            "length_m", "ave_speed_mps", "ave_grad", "gap_time_s", "n_points"]
    return out[cols].sort_values("route_id").reset_index(drop=True)


# ------------ EDGE TABLES ------------

def build_protects_edges(routes: pd.DataFrame) -> pd.DataFrame:
    """signal -> first TC of any outbound route   (K=1, berth-track only)."""
    rows = []
    for _, r in routes.iterrows():
        if r["track_sections"] is not None and len(r["track_sections"]) > 0:
            rows.append({"signal_id": r["signal_no"],
                         "track_id":  r["track_sections"][0]})
    return (pd.DataFrame(rows).drop_duplicates()
              .sort_values(["signal_id", "track_id"]).reset_index(drop=True))


def build_connects_edges(routes: pd.DataFrame) -> pd.DataFrame:
    """track <-> track adjacency. Two TCs connect iff they appear consecutively
    in any route. Stored as a directed edge in BOTH directions (a -> b and
    b -> a) so downstream graph code doesn't need to symmetrize."""
    pairs = set()
    for _, r in routes.iterrows():
        tcs = r["track_sections"]
        for a, b in zip(tcs[:-1], tcs[1:]):
            if a != b:
                pairs.add((a, b))
                pairs.add((b, a))
    rows = [{"track_a": a, "track_b": b} for (a, b) in pairs]
    return (pd.DataFrame(rows)
              .sort_values(["track_a", "track_b"]).reset_index(drop=True))


def build_traverses_edges(routes: pd.DataFrame) -> pd.DataFrame:
    """route -> track edges, with `order` index along the route."""
    rows = []
    for _, r in routes.iterrows():
        for i, tc in enumerate(r["track_sections"]):
            rows.append({"route_id": r["route_id"], "track_id": tc, "order": i})
    return pd.DataFrame(rows).sort_values(["route_id", "order"]).reset_index(drop=True)


def build_starts_at_edges(routes: pd.DataFrame) -> pd.DataFrame:
    """route -> start signal. Multiple variants may have multiple start signals;
    we record one row per (route_id, signal_id) pair."""
    rows = []
    for _, r in routes.iterrows():
        for s in r["start_signals"]:
            rows.append({"route_id": r["route_id"], "signal_id": str(s)})
    return (pd.DataFrame(rows).drop_duplicates()
              .sort_values(["route_id", "signal_id"]).reset_index(drop=True))


def build_ends_at_edges(routes: pd.DataFrame) -> pd.DataFrame:
    """route -> end signal."""
    rows = []
    for _, r in routes.iterrows():
        for s in r["end_signals"]:
            rows.append({"route_id": r["route_id"], "signal_id": str(s)})
    return (pd.DataFrame(rows).drop_duplicates()
              .sort_values(["route_id", "signal_id"]).reset_index(drop=True))


def build_same_signal_edges(routes: pd.DataFrame) -> pd.DataFrame:
    """route_a <-> route_b for any pair sharing a start signal — these are
    *alternative* routes from one signal (the dynamic action mask Aₜ)."""
    rows = []
    for sig, sub in routes.groupby("signal_no"):
        ids = sub["route_id"].tolist()
        if len(ids) < 2:
            continue
        for a, b in combinations(ids, 2):
            rows.append({"route_a": a, "route_b": b, "shared_signal": sig})
            rows.append({"route_a": b, "route_b": a, "shared_signal": sig})
    return (pd.DataFrame(rows)
              .sort_values(["route_a", "route_b"]).reset_index(drop=True))


# ------------ ENTRY POINT ------------

def run() -> dict:
    t0 = time.time()
    print("=== P2.3 Iter 1 — static graph backbone ===")

    routes      = pd.read_parquet(C.ROUTES_CLEAN_PARQUET)
    tracks_inv  = pd.read_parquet(C.TRACKS_INVENTORY_PARQUET)
    signals_inv = pd.read_parquet(C.SIGNALS_INVENTORY_PARQUET)
    platform_end = _load_platform_end_signals()
    platform_tc  = _load_platform_tc_map()
    print(f"  loaded inputs: {len(routes)} routes, {len(tracks_inv)} tracks, "
          f"{len(signals_inv)} signals, {len(platform_end)} platform-ends, "
          f"{len(platform_tc)} platform-TCs")

    # Nodes
    track_nodes  = build_track_nodes(routes, tracks_inv, platform_tc)
    signal_nodes = build_signal_nodes(routes, signals_inv, platform_end)
    route_nodes  = build_route_nodes(routes, signal_nodes, platform_end)

    # TRTS nodes (new)
    trts_nodes = build_trts_nodes()

    # Edges
    protects     = build_protects_edges(routes)
    connects     = build_connects_edges(routes)
    traverses    = build_traverses_edges(routes)
    starts_at    = build_starts_at_edges(routes)
    ends_at      = build_ends_at_edges(routes)
    same_signal  = build_same_signal_edges(routes)

    # Persist
    track_nodes.to_parquet(C.NODE_TRACK_PARQUET, index=False)
    signal_nodes.to_parquet(C.NODE_SIGNAL_PARQUET, index=False)
    route_nodes.to_parquet(C.NODE_ROUTE_PARQUET, index=False)
    trts_nodes.to_parquet(C.NODE_TRTS_PARQUET, index=False)
    protects.to_parquet(C.EDGE_PROTECTS_PARQUET, index=False)
    connects.to_parquet(C.EDGE_CONNECTS_PARQUET, index=False)
    traverses.to_parquet(C.EDGE_TRAVERSES_PARQUET, index=False)
    starts_at.to_parquet(C.EDGE_STARTS_AT_PARQUET, index=False)
    ends_at.to_parquet(C.EDGE_ENDS_AT_PARQUET, index=False)
    same_signal.to_parquet(C.EDGE_SAME_SIGNAL_PARQUET, index=False)

    summary = {
        "elapsed_s": round(time.time() - t0, 2),
        "nodes": {
            "track":  int(len(track_nodes)),
            "signal": int(len(signal_nodes)),
            "route":  int(len(route_nodes)),
            "trts":   int(len(trts_nodes)),
        },
        "edges": {
            "protects":    int(len(protects)),
            "connects":    int(len(connects)),
            "traverses":   int(len(traverses)),
            "starts_at":   int(len(starts_at)),
            "ends_at":     int(len(ends_at)),
            "same_signal": int(len(same_signal)),
        },
        "track_source_breakdown":
            track_nodes["source"].value_counts().to_dict(),
        "platform_coverage": {
            "tracks_with_platform":      int(track_nodes["platform_id"].notna().sum()),
            "signals_at_platform_end":   int(signal_nodes["is_platform_end"].sum()),
            "routes_ending_at_platform": int(route_nodes["end_platform_id"].notna().sum()),
        },
        "physical_features_coverage": {
            "routes_with_length":     int(route_nodes["length_m"].notna().sum()),
            "routes_with_speed":      int(route_nodes["ave_speed_mps"].notna().sum()),
            "routes_with_gap_time":   int(route_nodes["gap_time_s"].notna().sum()),
            "routes_with_n_points":   int(route_nodes["n_points"].notna().sum()),
            "length_m_describe":   {k: round(float(v),1) for k,v in route_nodes["length_m"].describe().to_dict().items()},
            "ave_speed_mps_describe": {k: round(float(v),2) for k,v in route_nodes["ave_speed_mps"].describe().to_dict().items()},
        },
        "by_prefix_in_signal_nodes":
            signal_nodes["prefix"].value_counts(dropna=False).to_dict(),
        "sop_cross_check": {
            "signals_with_routes_from_only":  int(((signal_nodes["n_routes_from"] >  0) &
                                                    (signal_nodes["n_routes_to"]   == 0)).sum()),
            "signals_with_routes_to_only":    int(((signal_nodes["n_routes_from"] == 0) &
                                                    (signal_nodes["n_routes_to"]   > 0)).sum()),
            "signals_with_both":              int(((signal_nodes["n_routes_from"] >  0) &
                                                    (signal_nodes["n_routes_to"]   > 0)).sum()),
            "signals_with_no_routes":         int(((signal_nodes["n_routes_from"] == 0) &
                                                    (signal_nodes["n_routes_to"]   == 0)).sum()),
            "trts_per_platform":              trts_nodes.groupby("platform_id").size().to_dict(),
        },
    }
    C.STATIC_GRAPH_SUMMARY_JSON.write_text(json.dumps(summary, indent=2, default=str))

    print("\n=== Static graph summary ===")
    print(json.dumps(summary, indent=2, default=str))
    return summary


if __name__ == "__main__":
    run()
