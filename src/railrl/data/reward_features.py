"""P2.4 Iter C - Per-decision feature builder (delay_change / next_tc_headway).

Approach distance for set decisions is computed via the Iter A helper
compute_approach_distance_distribution (sample_size=full).
"""
from __future__ import annotations
import time as _time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .. import config as C
from .event_stream import AssetIndex, EventTokenStream
from .static_graph_view import StaticGraphView


# ============================================================
# Delay change from Movements/TRUST
# ============================================================

def compute_delay_changes(decision_points: pd.DataFrame,
                           movements_csv: Path,
                           *, window_seconds: float = 4202.0) -> np.ndarray:
    """Per decision: delay_change averaged across decisions sharing one pass.

    Pass = one TRUST train_id. Same headcode on same day can have multiple
    TRUST ids when the train passes Derby twice (e.g. morning + evening run);
    each TRUST id is treated as an independent pass with its own bracket
    scope.

    For each decision (focal_train, time):
      1. Find the TRUST id of same headcode whose time range contains t
         (within +- window_seconds margin).
      2. Within that TRUST id, bracket = (last TIPLOC <= t, first TIPLOC > t).
      3. Both endpoints must be within window_seconds of t.
      4. delay_change = arr_delay[j] - arr_delay[j-1].
      5. Average-attribute across all decisions sharing this bracket
         (within the same TRUST id).
    """
    print(f"  loading Movements ({movements_csv.name}) ...")
    mv = pd.read_csv(movements_csv,
                      usecols=["train_id", "actual_timestamp", "planned_timestamp"])
    # 🔴 fix #2: correct the +1h Apr-Jul 2023 Movements clock so actual_ns aligns with
    # decision times (delay_s = actual-planned is invariant; both shift together).
    from ..data_io import correct_movements_bst
    mv = correct_movements_bst(mv, time_cols=("actual_timestamp", "planned_timestamp"),
                               ref_col="actual_timestamp")
    mv["headcode"] = mv["train_id"].astype(str).str[2:6]
    mv["actual"]   = pd.to_datetime(mv["actual_timestamp"], errors="coerce")
    mv["planned"]  = pd.to_datetime(mv["planned_timestamp"], errors="coerce")
    mv = mv.dropna(subset=["actual", "planned", "headcode"])
    mv = mv[mv["headcode"].str.len() == 4]
    mv["actual_ns"] = mv["actual"].astype("datetime64[ns]").astype("int64")
    mv["delay_s"]   = (mv["actual"] - mv["planned"]).dt.total_seconds()

    # 🔴 FIX (2026-05-24): the TRUST train_id RECURS MONTHLY (EE = day-of-month,
    # Table 3.6), so grouping by raw train_id lumps points from many months into one
    # "run" spanning ~a year (45% of ids span >25 days; 81% of points). That made the
    # per-decision match pick the WRONG day's run → brackets straddled month gaps →
    # out_window crushed delay_change coverage to 6% (and ~0% for Mar-Jul). Same root
    # cause as the episode-cross-month bug. Fix: split each train_id's points into single
    # RUNS (consecutive points with gap <= RUN_GAP_S) and match/bracket within a run.
    # Within-run gaps are ~2 min; monthly reuse (and morning/evening passes) are hours/
    # days apart → cleanly separated. (See IMPLEMENTATION_LOG 2026-05-24 delay bug.)
    RUN_GAP_S = 7200.0                       # 2 h
    run_gap_ns = int(RUN_GAP_S * 1e9)
    # by_run[run_key] = (sorted times, delays); headcode_to_runs[hc] = [(t_first,t_last,run_key)]
    by_run: dict = {}
    headcode_to_runs: dict = {}
    for tid, sub in mv.groupby("train_id"):
        sub = sub.sort_values("actual_ns")
        arr_t = sub["actual_ns"].to_numpy(np.int64)
        arr_d = sub["delay_s"].to_numpy(np.float64)
        hc = str(sub["headcode"].iloc[0])
        splits = np.where(np.diff(arr_t) > run_gap_ns)[0] + 1
        for k, idx in enumerate(np.split(np.arange(arr_t.size), splits)):
            if idx.size == 0:
                continue
            rt, rd = arr_t[idx], arr_d[idx]
            run_key = (tid, k)
            by_run[run_key] = (rt, rd)
            headcode_to_runs.setdefault(hc, []).append((int(rt[0]), int(rt[-1]), run_key))
    # Sort each headcode's runs by t_first for fast lookup
    for hc in headcode_to_runs:
        headcode_to_runs[hc].sort()
    print(f"  Movements: {len(mv):,} rows, {len(by_run):,} runs "
          f"(gap-split from {mv['train_id'].nunique():,} train_ids), "
          f"{len(headcode_to_runs):,} headcodes")

    n = len(decision_points)
    times  = pd.to_datetime(decision_points["time"]).astype("datetime64[ns]").astype("int64").to_numpy()
    trains = decision_points["focal_train"].astype(str).to_numpy()
    window_ns = int(window_seconds * 1e9)

    # Stage 1: per-decision bracket lookup within the matched TRUST id
    out = np.full(n, np.nan, dtype=np.float64)
    bracket = np.full(n, -1, dtype=np.int64)
    matched_run = np.empty(n, dtype=object)
    n_no_match = 0; n_no_baseline = 0; n_no_followup = 0; n_out_window = 0
    for i in range(n):
        hc = trains[i]
        candidates = headcode_to_runs.get(hc)
        if not candidates:
            n_no_match += 1; continue
        t = times[i]
        # Pick the RUN whose [t_first - W, t_last + W] contains t, preferring the
        # one whose center is closest to t. (Runs are ~5-min single passes now, so
        # this window actually discriminates between days — the pre-fix bug was that
        # year-spanning train_ids made every candidate match.)
        best_run = None; best_dist = None
        for t_first, t_last, run_key in candidates:
            if (t_first - window_ns) <= t <= (t_last + window_ns):
                center = (t_first + t_last) // 2
                dist = abs(t - center)
                if best_dist is None or dist < best_dist:
                    best_dist = dist; best_run = run_key
        if best_run is None:
            n_no_match += 1; continue
        arr_t, arr_d = by_run[best_run]
        j = int(np.searchsorted(arr_t, t, side="right"))
        if j == 0:
            n_no_baseline += 1; continue
        if j >= arr_t.size:
            n_no_followup += 1; continue
        if (arr_t[j] - t) > window_ns:
            n_out_window += 1; continue
        if (t - arr_t[j - 1]) > window_ns:
            n_out_window += 1; continue
        out[i] = float(arr_d[j] - arr_d[j - 1])
        bracket[i] = j
        matched_run[i] = best_run

    # Stage 2: average attribution per (trust_id, bracket_j) bucket
    if (bracket >= 0).any():
        valid_mask = ~np.isnan(out)
        df_idx = pd.DataFrame({
            "run":     matched_run,
            "bracket": bracket,
            "valid":   valid_mask,
        })
        counts = (df_idx[df_idx["valid"]]
                    .groupby(["run", "bracket"], sort=False)
                    .size()
                    .rename("n").reset_index())
        df_idx = df_idx.merge(counts, on=["run","bracket"], how="left")
        share = df_idx["n"].fillna(1).to_numpy(dtype=float)
        out = np.where(valid_mask, out / share, np.nan)

    n_attributed = int(np.isfinite(out).sum())
    print(f"  delay_change: avg_attributed={n_attributed:,}; no_match={n_no_match:,} "
          f"no_baseline={n_no_baseline:,} no_followup={n_no_followup:,} "
          f"out_window={n_out_window:,}")
    return out

def build_route_first_tc(asset_index: AssetIndex,
                          edges_traverses_parquet: Path) -> dict[str, str]:
    """For each route_id (str), return its FIRST traversed TC (str)."""
    df = pd.read_parquet(edges_traverses_parquet)
    # `order` column gives the position of TC along route; if absent, use file order
    if "order" in df.columns:
        df = df.sort_values(["route_id", "order"])
    out = (df.groupby("route_id", sort=False)["track_id"]
              .first().astype(str).to_dict())
    return out


def compute_next_tc_headways(set_decisions: pd.DataFrame,
                              route_first_tc: dict[str, str],
                              event_stream: EventTokenStream,
                              asset_index: AssetIndex) -> np.ndarray:
    """For each set+used decision, follow-on headway at the route's first TC.

    Algorithm: from decision_time, find:
      1. first 0->1 (this train occupies the TC)
      2. next 1->0 (this train clears)
      3. next 0->1 (NEXT train occupies)
      headway = (next_occupy - clear) seconds.

    Skipped (NaN) when:
      - outcome != 'used' (no realized passage)
      - any of the three transitions can't be found in the event stream
    """
    ev_by = event_stream._build_per_asset_index()
    times_full  = event_stream.time_ns
    states_full = event_stream.state

    out = np.full(len(set_decisions), np.nan, dtype=np.float64)
    rows = list(set_decisions.itertuples(index=False))
    n_total = len(rows)
    n_ok = 0

    for i, r in enumerate(rows):
        outcome = getattr(r, "route_outcome", "unknown")
        if outcome != "used":
            continue
        rid = str(r.chosen_route_id)
        first_tc = route_first_tc.get(rid)
        if first_tc is None:
            continue
        tc_idx = asset_index.idx(first_tc)
        if tc_idx is None:
            continue
        positions = ev_by.get(int(tc_idx))
        if positions is None or positions.size == 0:
            continue

        t_ns = int(pd.Timestamp(r.time).value)
        tc_t = times_full[positions]
        tc_s = states_full[positions]
        j = int(np.searchsorted(tc_t, t_ns, side="left"))
        if j >= positions.size:
            continue

        # Step 1: find first occupy (state=1) at or after j
        seg_s = tc_s[j:]
        occ_locs = np.flatnonzero(seg_s == 1)
        if occ_locs.size == 0:
            continue
        occ_pos = j + int(occ_locs[0])
        # Step 2: find next clear (state=0) after occupy
        seg2 = tc_s[occ_pos + 1:]
        clear_locs = np.flatnonzero(seg2 == 0)
        if clear_locs.size == 0:
            continue
        clear_pos = occ_pos + 1 + int(clear_locs[0])
        clear_t = int(tc_t[clear_pos])
        # Step 3: find next occupy after clear
        seg3 = tc_s[clear_pos + 1:]
        next_occ_locs = np.flatnonzero(seg3 == 1)
        if next_occ_locs.size == 0:
            continue
        next_occ_pos = clear_pos + 1 + int(next_occ_locs[0])
        next_occ_t = int(tc_t[next_occ_pos])

        headway_s = (next_occ_t - clear_t) / 1e9
        if headway_s > 0:
            out[i] = headway_s
            n_ok += 1

    print(f"  next_tc_headway: {n_ok:,}/{n_total:,} measurable for set decisions")
    return out


# ============================================================
# Fast approach distance: BFS once per focal_signal, then O(1) lookup
# ============================================================

def compute_approach_distance_fast(set_decisions: pd.DataFrame,
                                    train_position_lookup: pd.DataFrame,
                                    view: StaticGraphView) -> np.ndarray:
    """Same semantics as Iter A's helper but ~100x faster on full data.

    Strategy: for each focal_signal (~95 unique), do a multi-source BFS
    from its protected TCs once, building a {tc -> hop_distance} map.
    Then each decision is just a dict lookup on (focal_signal, train_tc).
    """
    from collections import defaultdict

    # Build sig -> list of TCs (signal's protected tracks)
    sig_to_tcs = defaultdict(list)
    for _, r in view.edges["protects"].iterrows():
        sig_to_tcs[str(r["signal_id"])].append(str(r["track_id"]))

    # Build TC adjacency from connects edges
    tc_adj = defaultdict(set)
    for _, r in view.edges["connects"].iterrows():
        a, b = str(r["track_a"]), str(r["track_b"])
        tc_adj[a].add(b); tc_adj[b].add(a)

    # Per focal_signal in this batch: multi-source BFS to fill {tc -> hops}
    unique_signals = set(set_decisions["focal_signal"].astype(str))
    distance_maps: dict[str, dict[str, int]] = {}
    for sig in unique_signals:
        targets = set(sig_to_tcs.get(sig, []))
        if not targets:
            distance_maps[sig] = {}
            continue
        dist = {tc: 0 for tc in targets}
        frontier = set(targets)
        for hop in range(1, 31):
            nxt = set()
            for n in frontier:
                for nbr in tc_adj.get(n, ()):
                    if nbr in dist: continue
                    dist[nbr] = hop
                    nxt.add(nbr)
            if not nxt: break
            frontier = nxt
        distance_maps[sig] = dist

    # Per-(train, date) sorted timeline (for as-of TC lookup, same-day only).
    # Same-day filter avoids matching to a previous-week run of the same headcode
    # which would give a stale "last position" in the presence of data gaps.
    train_position_lookup = train_position_lookup.copy()
    train_position_lookup["date"] = pd.to_datetime(
        train_position_lookup["time_ns"], unit="ns").dt.date

    by_train_date: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
    for (tid, d), sub in train_position_lookup.groupby(["trainid", "date"]):
        sub = sub.sort_values("time_ns")
        by_train_date[(str(tid), d)] = (sub["time_ns"].to_numpy(np.int64),
                                          sub["tc_id"].to_numpy())

    distances = np.full(len(set_decisions), np.nan)
    rows = list(set_decisions.itertuples(index=False))
    for i, r in enumerate(rows):
        tid = str(r.focal_train); sig = str(r.focal_signal)
        ts  = pd.Timestamp(r.time)
        key = (tid, ts.date())
        sub_pair = by_train_date.get(key)
        if sub_pair is None:
            continue
        cache = distance_maps.get(sig)
        if not cache:
            continue
        t_ns = int(ts.value)
        arr_t, arr_tc = sub_pair
        j = int(np.searchsorted(arr_t, t_ns, side="right"))
        if j == 0:
            continue
        train_tc = str(arr_tc[j - 1])
        d = cache.get(train_tc)
        if d is not None:
            distances[i] = d
    return distances
