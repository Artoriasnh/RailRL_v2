"""P2.4 Iter A — Empirical threshold calibration for reward components."""
from __future__ import annotations
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .. import config as C
from .event_stream import AssetIndex, EventTokenStream
from .static_graph_view import StaticGraphView


def compute_headway_distribution(es, asset_index):
    track_idxs = set(asset_index.indices_of_type("Track"))
    ev_by = es._build_per_asset_index()
    gaps_seconds = []
    for a in track_idxs:
        positions = ev_by.get(int(a))
        if positions is None or positions.size < 2:
            continue
        states = es.state[positions]
        times  = es.time_ns[positions]
        last_clear_t = None
        for i in range(len(states)):
            s = states[i]; t = times[i]
            if last_clear_t is None:
                if i > 0 and states[i - 1] == 1 and s == 0:
                    last_clear_t = int(t)
            else:
                if s == 1 and i > 0 and states[i - 1] == 0:
                    gap_ns = int(t) - last_clear_t
                    if gap_ns > 0:
                        gaps_seconds.append(gap_ns / 1e9)
                    last_clear_t = None
    return np.array(gaps_seconds, dtype=np.float64)


def _build_signal_to_tc(view):
    pr = view.edges["protects"]
    out = defaultdict(list)
    for _, r in pr.iterrows():
        out[str(r["signal_id"])].append(str(r["track_id"]))
    return out


def _build_tc_adjacency(view):
    adj = defaultdict(set)
    for _, r in view.edges["connects"].iterrows():
        a, b = str(r["track_a"]), str(r["track_b"])
        adj[a].add(b); adj[b].add(a)
    return adj


def _bfs_hop_distance(src, targets, adj, max_hops=30):
    if src in targets:
        return 0
    visited = {src}; frontier = {src}
    for d in range(1, max_hops + 1):
        nxt = set()
        for n in frontier:
            for nbr in adj.get(n, ()):
                if nbr in visited: continue
                if nbr in targets: return d
                visited.add(nbr); nxt.add(nbr)
        if not nxt: return None
        frontier = nxt
    return None


def compute_approach_distance_distribution(decisions, train_position_lookup, view, *,
                                            sample_size=50_000, random_state=2026):
    if len(decisions) > sample_size:
        decisions = decisions.sample(n=sample_size, random_state=random_state)
    sig_to_tcs = _build_signal_to_tc(view)
    tc_adj     = _build_tc_adjacency(view)
    by_train = {}
    for tid, sub in train_position_lookup.groupby("trainid"):
        sub = sub.sort_values("time_ns")
        by_train[str(tid)] = (sub["time_ns"].to_numpy(np.int64),
                              sub["tc_id"].to_numpy())
    distances = []; signals_out = []; trains_out = []
    for _, r in decisions.iterrows():
        tid    = str(r["focal_train"])
        sigid  = str(r["focal_signal"])
        t_ns   = int(pd.Timestamp(r["time"]).value)
        if tid not in by_train:
            distances.append(None); signals_out.append(sigid); trains_out.append(tid); continue
        times_arr, tcs_arr = by_train[tid]
        j = int(np.searchsorted(times_arr, t_ns, side="right"))
        if j == 0:
            distances.append(None); signals_out.append(sigid); trains_out.append(tid); continue
        train_tc = str(tcs_arr[j - 1])
        target_tcs = set(sig_to_tcs.get(sigid, []))
        if not target_tcs:
            distances.append(None); signals_out.append(sigid); trains_out.append(tid); continue
        d = _bfs_hop_distance(train_tc, target_tcs, tc_adj, max_hops=30)
        distances.append(d); signals_out.append(sigid); trains_out.append(tid)
    return pd.DataFrame({"focal_signal": signals_out, "focal_train": trains_out, "distance": distances})


def compute_tiploc_lag_distribution(decisions, movements, *, sample_size=50_000,
                                     random_state=2026, max_lag_seconds=14400.0):
    if len(decisions) > sample_size:
        decisions = decisions.sample(n=sample_size, random_state=random_state)
    mv = movements.copy()
    mv["actual_timestamp"] = pd.to_datetime(mv["actual_timestamp"], errors="coerce")
    mv = mv.dropna(subset=["actual_timestamp", "train_id"])
    mv["t_ns"] = mv["actual_timestamp"].astype("datetime64[ns]").astype("int64")
    by_train = {}
    for tid, sub in mv.groupby("train_id"):
        by_train[str(tid)] = np.sort(sub["t_ns"].to_numpy(np.int64))
    lags_seconds = []
    for _, r in decisions.iterrows():
        tid  = str(r["focal_train"])
        t_ns = int(pd.Timestamp(r["time"]).value)
        arr = by_train.get(tid)
        if arr is None or arr.size == 0: continue
        j = int(np.searchsorted(arr, t_ns, side="right"))
        if j >= arr.size: continue
        lag = (int(arr[j]) - t_ns) / 1e9
        if 0 < lag <= max_lag_seconds:
            lags_seconds.append(lag)
    return np.array(lags_seconds, dtype=np.float64)


def build_train_position_lookup_from_td(td_path):
    cols = ["time", "type", "id", "state", "trainid_filled"]
    df = pd.read_parquet(td_path, columns=cols)
    mask = ((df["type"] == "Track") & (df["state"] == 1)
            & df["trainid_filled"].notna() & (df["trainid_filled"] != "")
            & df["id"].notna())
    df = df.loc[mask]
    tc_ids   = np.array(df["id"].tolist(),             dtype=object)
    trainids = np.array(df["trainid_filled"].tolist(), dtype=object)
    times    = pd.to_datetime(df["time"]).astype("datetime64[ns]").astype("int64").to_numpy()
    return pd.DataFrame({"trainid": trainids, "time_ns": times, "tc_id": tc_ids})


def percentiles(arr, ps):
    if arr.size == 0:
        return {f"p{p}": float("nan") for p in ps}
    return {f"p{p}": float(np.percentile(arr, p)) for p in ps}


def derive_thresholds(headway_gaps, approach_dist, tiploc_lags):
    headway_pcts  = percentiles(headway_gaps, [1, 5, 10, 50, 90, 99])
    approach_arr  = approach_dist["distance"].dropna().to_numpy(dtype=np.float64)
    approach_pcts = percentiles(approach_arr, [10, 50, 90, 95, 99])
    tiploc_pcts   = percentiles(tiploc_lags, [50, 90, 95, 99])
    H_min_seconds  = headway_pcts.get(f"p{C.HEADWAY_PERCENTILE_FOR_HMIN}", 30.0)
    d_low          = approach_pcts.get(f"p{C.APPROACH_PERCENTILE_LOW}",  3.0)
    d_high         = approach_pcts.get(f"p{C.APPROACH_PERCENTILE_HIGH}", 8.0)
    window_seconds = tiploc_pcts.get(f"p{C.TIPLOC_LAG_PERCENTILE}", 300.0)
    return {
        "headway": {
            "n_pairs": int(headway_gaps.size),
            "percentiles": headway_pcts,
            "H_min_seconds_used": float(H_min_seconds),
            "percentile_used": int(C.HEADWAY_PERCENTILE_FOR_HMIN),
        },
        "approach_distance": {
            "n_decisions_sampled": int(approach_dist.shape[0]),
            "n_with_distance":     int(approach_arr.size),
            "percentiles":         approach_pcts,
            "d_gate_breakpoints": {
                "gate_1.0_max": 2,
                "gate_0.5_max": int(round(d_low)),
                "gate_0.1_max": int(round(d_high)),
            },
        },
        "tiploc_lag": {
            "n_lags": int(tiploc_lags.size),
            "percentiles": tiploc_pcts,
            "window_seconds_used": float(window_seconds),
            "percentile_used": int(C.TIPLOC_LAG_PERCENTILE),
        },
    }
