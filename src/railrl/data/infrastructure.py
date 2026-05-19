"""Phase 1.1.5 — Infrastructure graph from route_to_tc_all.csv.

Produces:
    routes_clean.parquet      277 unique named routes (parsed)
    tracks_inventory.parquet  per track section: which routes traverse it
    signals_inventory.parquet per signal: outbound routes (action set)
    auxiliary_connections.parquet  rows where route_id is missing
    infrastructure_graph.json node/edge counts and per-prefix subgraph stats
"""
from __future__ import annotations
import json
import time
from collections import defaultdict

import pandas as pd

from .. import config as C
from ..data_io import load_route_to_tc
from ..parsers import ROUTE_RE_PATTERN


def split_named_and_aux(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separate proper named routes from auxiliary (NaN-route) connections."""
    df = df.copy()
    df["n_tc"] = df["track_list"].apply(len)
    df["is_named_route"] = df["route"].notna()
    named = df[df["is_named_route"]].copy()
    aux = df[~df["is_named_route"]].copy()
    return named.reset_index(drop=True), aux.reset_index(drop=True)


def parse_named_routes(named: pd.DataFrame) -> pd.DataFrame:
    """Add prefix / signal_no / letter / sub / cls columns to the named routes."""
    # Strip stray whitespace from route_id before parsing (some entries have trailing spaces)
    named = named.copy()
    named["route"] = named["route"].astype(str).str.strip()
    ext = named["route"].astype(str).str.extract(ROUTE_RE_PATTERN)
    out = named.copy()
    for col in ("prefix", "signal", "letter", "sub", "cls"):
        out[col] = ext[col].values
    return out.rename(columns={"signal": "signal_no"})


def deduplicate(named_parsed: pd.DataFrame) -> pd.DataFrame:
    """Aggregate route variants by route name (277 unique from 291 raw rows)."""
    rows = []
    for route, sub in named_parsed.groupby("route"):
        track_lists = sub["track_list"].tolist()
        canonical_tcs = max(track_lists, key=len) if track_lists else []
        rows.append({
            "route_id": route,
            "prefix": sub["prefix"].iloc[0],
            "signal_no": sub["signal_no"].iloc[0],
            "letter": sub["letter"].iloc[0],
            "sub": sub["sub"].iloc[0],
            "cls": sub["cls"].iloc[0],
            "start_signals": sorted(set(sub["start"].astype(str))),
            "end_signals": sorted(set(sub["end"].astype(str))),
            "track_sections": canonical_tcs,
            "n_tc": len(canonical_tcs),
            "n_variants": len(sub),
        })
    return pd.DataFrame(rows)


def build_track_inventory(routes: pd.DataFrame) -> pd.DataFrame:
    """For each track section: list routes that traverse it (conflict-detection key)."""
    tc_to_routes: dict[str, list[str]] = defaultdict(list)
    for _, r in routes.iterrows():
        for tc in r["track_sections"]:
            tc_to_routes[tc].append(r["route_id"])
    rows = [
        {"track_id": tc, "n_routes": len(rts), "routes": rts}
        for tc, rts in tc_to_routes.items()
    ]
    return pd.DataFrame(rows).sort_values("n_routes", ascending=False).reset_index(drop=True)


def build_signal_inventory(routes: pd.DataFrame) -> pd.DataFrame:
    """For each signal: outbound routes (defines the dynamic action mask Aₜ)."""
    from_signals: dict[str, list[dict]] = defaultdict(list)
    for _, r in routes.iterrows():
        for s in r["start_signals"]:
            from_signals[s].append({
                "route_id": r["route_id"], "letter": r["letter"],
                "cls": r["cls"], "sub": r["sub"],
            })
    rows = [
        {"signal": s, "n_routes_from": len(rts), "routes_from": rts}
        for s, rts in from_signals.items()
    ]
    return pd.DataFrame(rows).sort_values("n_routes_from", ascending=False).reset_index(drop=True)


def graph_stats(routes: pd.DataFrame, tracks: pd.DataFrame, signals: pd.DataFrame) -> dict:
    by_prefix = {}
    for p in C.DERBY_PREFIXES:
        sub = routes[routes["prefix"] == p]
        by_prefix[p] = {
            "n_routes": int(len(sub)),
            "n_tracks_used": int(len({tc for lst in sub["track_sections"] for tc in lst})),
            "n_distinct_signals": int(len(
                {s for lst in sub["start_signals"] for s in lst}
                | {s for lst in sub["end_signals"] for s in lst}
            )),
            "avg_tcs_per_route": round(sub["n_tc"].mean(), 2) if len(sub) else 0,
        }

    n_routes_per_signal = signals["n_routes_from"]
    return {
        "n_named_routes": int(len(routes)),
        "n_track_sections": int(len(tracks)),
        "n_signals_with_outbound_routes": int(len(signals)),
        "by_prefix": by_prefix,
        "class_distribution": routes["cls"].value_counts().to_dict(),
        "tcs_per_route_describe": {
            k: float(v) for k, v in routes["n_tc"].describe().to_dict().items()
        },
        "routes_per_signal_describe": {
            k: float(v) for k, v in n_routes_per_signal.describe().to_dict().items()
        },
        "signals_with_only_one_route": int((n_routes_per_signal == 1).sum()),
        "signals_with_2plus_routes_decision_required": int((n_routes_per_signal >= 2).sum()),
        "signals_with_3plus_routes": int((n_routes_per_signal >= 3).sum()),
        "signals_with_4plus_routes": int((n_routes_per_signal >= 4).sum()),
        "max_routes_from_one_signal": int(n_routes_per_signal.max()),
        "top10_busiest_signals": signals.head(10)[["signal", "n_routes_from"]]
            .to_dict(orient="records"),
        "top10_most_shared_tracks": tracks.head(10)[["track_id", "n_routes"]]
            .to_dict(orient="records"),
    }


def run() -> dict:
    """End-to-end infrastructure-graph build. Returns the stats dict."""
    print("=== Phase 1.1.5 infrastructure ===")
    t0 = time.time()
    df = load_route_to_tc()
    named, aux = split_named_and_aux(df)
    print(f"  named routes (raw rows)    : {len(named)}")
    print(f"  auxiliary rows (route=NaN) : {len(aux)}")

    named_parsed = parse_named_routes(named)
    routes = deduplicate(named_parsed)
    print(f"  unique named routes        : {len(routes)}")

    tracks = build_track_inventory(routes)
    print(f"  unique track sections      : {len(tracks)}")

    signals = build_signal_inventory(routes)
    print(f"  signals with ≥ 1 outbound  : {len(signals)}")

    routes.to_parquet(C.ROUTES_CLEAN_PARQUET, index=False, compression="zstd")
    tracks.to_parquet(C.TRACKS_INVENTORY_PARQUET, index=False, compression="zstd")
    signals.to_parquet(C.SIGNALS_INVENTORY_PARQUET, index=False, compression="zstd")
    aux.to_parquet(C.AUXILIARY_PARQUET, index=False, compression="zstd")

    stats = graph_stats(routes, tracks, signals)
    C.INFRASTRUCTURE_GRAPH_JSON.write_text(json.dumps(stats, indent=2, default=str))
    print(f"\n=== Infrastructure summary (elapsed {time.time()-t0:.1f}s) ===")
    print(json.dumps(stats, indent=2, default=str))
    return stats
