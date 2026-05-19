"""spec 02 §2 — Decision point trigger logic.

Generates the unified decision_points table containing both:
  - SET triggers (every PR in decision_events.parquet)
  - WAIT triggers (focal_train enters approach horizon of focal_signal,
                   and no PR happens within Δ_WAIT=30s)

Output: outputs/decision_points/decision_points_v2.parquet

Each row = one decision point sample with:
    focal_train, focal_signal, t, label, chosen_route_id, trigger_type

Per spec 01 §17.5, focal_signal is SAMPLE METADATA — used here for trigger
generation and reward computation, NEVER passed as a state feature to the model.
"""
from __future__ import annotations
import ast
import time as _time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .. import config as C
from ..parsers import parse_route_id


# ============================================================
# Approach horizon helpers
# ============================================================

def compute_approach_tracks(routes_clean: pd.DataFrame,
                             k_hops: int = None) -> dict[str, set[str]]:
    """For each signal that is the end_signal of at least one route, return the
    set of TCs that lie within the last k_hops positions of any route ending
    at that signal — i.e. the tracks a train must traverse shortly before
    arriving at the signal.

    Per spec 02 §2.3, K_APPROACH = 2 (locked).

    Args:
        routes_clean: DataFrame from outputs/infrastructure/routes_clean.parquet
                      with at least columns ['end_signals', 'track_sections']
                      (or fallback to 'end_signal', 'track_list')
        k_hops:       how many tail TCs to take from each route's track list

    Returns:
        dict mapping signal_id → set of track_ids
    """
    if k_hops is None:
        k_hops = C.APPROACH_K_HOPS

    approach: dict[str, set[str]] = {}

    # Column names differ between v1 and the spec; handle both
    if "track_sections" in routes_clean.columns:
        tc_col = "track_sections"
    elif "track_list" in routes_clean.columns:
        tc_col = "track_list"
    else:
        raise KeyError("routes_clean missing track_sections / track_list column")

    if "end_signals" in routes_clean.columns:
        sig_col = "end_signals"
    elif "end_signal" in routes_clean.columns:
        sig_col = "end_signal"
    else:
        raise KeyError("routes_clean missing end_signals / end_signal column")

    for _, r in routes_clean.iterrows():
        tcs = r[tc_col]
        # tcs may be list or string-encoded list
        if isinstance(tcs, str):
            try:
                tcs = ast.literal_eval(tcs)
            except Exception:
                tcs = []
        if tcs is None or len(tcs) == 0:
            continue

        # Take last k_hops TCs (or all if shorter).
        # NOTE: Python gotcha — tcs[-0:] == tcs[:] (entire list!), not [].
        # So k_hops=0 must be handled explicitly to mean "no approach".
        if k_hops <= 0:
            tail = []
        elif len(tcs) >= k_hops:
            tail = tcs[-k_hops:]
        else:
            tail = tcs[:]

        # end_signals may be a list or a single string
        sigs = r[sig_col]
        if isinstance(sigs, str):
            sigs = [sigs]
        elif sigs is None:
            continue

        for s in sigs:
            s_str = str(s)
            approach.setdefault(s_str, set()).update(str(tc) for tc in tail)

    return approach


# ============================================================
# Set triggers (from PR events — already in decision_events.parquet)
# ============================================================

def _extract_set_triggers(decision_events: pd.DataFrame) -> pd.DataFrame:
    """Map each PR row to a (set) decision point. spec 02 §2.2.

    Required columns in decision_events: time, route_id, train_id, signal_no
    """
    needed = ["time", "route_id", "train_id", "signal_no"]
    missing = [c for c in needed if c not in decision_events.columns]
    if missing:
        raise KeyError(f"decision_events.parquet missing columns: {missing}")

    df = decision_events[needed].copy()
    df = df.rename(columns={
        "train_id":  "focal_train",
        "signal_no": "focal_signal",
        "route_id":  "chosen_route_id",
        "time":      "t",
    })
    df["label"] = "set"
    df["trigger_type"] = "panel_request"

    # Force string type for focal_signal (downstream code expects str)
    df["focal_signal"] = df["focal_signal"].astype(str)
    df["focal_train"]  = df["focal_train"].astype(str)

    return df[["focal_train", "focal_signal", "t", "label",
               "chosen_route_id", "trigger_type"]]


# ============================================================
# Wait triggers (from TD Track state=1 events with trainid_filled)
# ============================================================

def _extract_wait_triggers(td_events: pd.DataFrame,
                            decision_events: pd.DataFrame,
                            approach_tracks: dict[str, set[str]],
                            delta_wait_seconds: float = None,
                            dedup_window_seconds: float = None) -> pd.DataFrame:
    """Generate wait triggers per spec 02 §2.3.

    Algorithm:
      1. Build reverse map: TC → list of signals it's in the approach horizon of
      2. Scan TD Track events with state=1 and trainid_filled != null
      3. For each (TC, train, t), for each signal S where TC ∈ approach_TCs(S):
         - Check if any PR for (train, S) exists in [t, t + Δ_WAIT]
         - If NO → emit wait sample (focal_train=train, focal_signal=S, t)
      4. Dedup: same (train, S) within `dedup_window_seconds` → keep earliest

    Args:
        td_events: DataFrame with columns time, type, id (TC id), state,
                    trainid_filled. Must contain Track state=1 events.
        decision_events: PR events (used for lookahead check)
        approach_tracks: from compute_approach_tracks()
        delta_wait_seconds: Δ_WAIT, default per spec 02 (30s)
        dedup_window_seconds: per (train, signal) dedup, default 30s

    Returns:
        DataFrame with same columns as _extract_set_triggers but label='wait'
        and chosen_route_id=None.
    """
    if delta_wait_seconds is None:
        delta_wait_seconds = C.DECISION_LOOKAHEAD_SECONDS
    if dedup_window_seconds is None:
        dedup_window_seconds = float(C.DECISION_LOOKAHEAD_SECONDS)

    # Build reverse map: TC → list of signals
    tc_to_signals: dict[str, list[str]] = {}
    for sig, tcs in approach_tracks.items():
        for tc in tcs:
            tc_to_signals.setdefault(tc, []).append(sig)

    # Filter TD events to Track state=1 with VALID trainid_filled.
    # We filter at trigger time (not just downstream) because:
    #   1. TD parses can fill trainid_filled with "0" or "" when train id unknown
    #   2. These garbage IDs cause MASSIVE wait inflation (~1.5M extra waits
    #      from focal_train="0" alone per Stage 2 diagnostics 2026-05-19)
    #   3. Real non-standard headcodes (e.g. "343R", 1.04% of data per
    #      PROJECT_HANDOFF Ch 5.5) ARE preserved — they have ≥3 alphanumeric chars
    # The f_unusual_id flag handles real non-standard IDs as a feature;
    # this filter only drops TD-parse-failure placeholders.
    trainid_str = td_events["trainid_filled"].astype(str)
    valid_id_mask = (
        trainid_str.str.match(r"^[0-9A-Z]{3,4}$", na=False)
        & ~trainid_str.isin({"0", "00", "000", "0000", "NULL", "NONE", "NAN", ""})
    )
    track_mask = (
        (td_events["type"] == "Track")
        & (td_events["state"] == 1)
        & (td_events["trainid_filled"].notna())
        & (td_events["id"].notna())
        & valid_id_mask
    )
    track_events = td_events.loc[track_mask, ["time", "id", "trainid_filled"]].copy()
    track_events.columns = ["t", "tc", "focal_train"]
    track_events["tc"]          = track_events["tc"].astype(str)
    track_events["focal_train"] = track_events["focal_train"].astype(str)
    track_events["t"] = pd.to_datetime(track_events["t"])
    track_events = track_events.sort_values("t").reset_index(drop=True)

    # Restrict to events whose TC is in some signal's approach horizon
    in_approach = track_events["tc"].isin(tc_to_signals.keys())
    track_events = track_events.loc[in_approach].copy()

    # Explode (TC event) → (signal-trigger) — one TC may protect multiple signals
    rows = []
    for _, ev in track_events.iterrows():
        for sig in tc_to_signals[ev["tc"]]:
            rows.append({"focal_train": ev["focal_train"],
                          "focal_signal": sig,
                          "t": ev["t"]})
    triggers = pd.DataFrame(rows)
    if triggers.empty:
        return triggers.assign(label="wait", chosen_route_id=None,
                                trigger_type="approach")

    # Dedup per (focal_train, focal_signal) within dedup_window_seconds
    triggers = triggers.sort_values(["focal_train", "focal_signal", "t"]).reset_index(drop=True)
    keep_mask = np.zeros(len(triggers), dtype=bool)
    last_t: dict[tuple[str, str], pd.Timestamp] = {}
    dedup_delta = pd.Timedelta(seconds=dedup_window_seconds)
    for i, row in triggers.iterrows():
        key = (row["focal_train"], row["focal_signal"])
        last = last_t.get(key)
        if last is None or (row["t"] - last) >= dedup_delta:
            keep_mask[i] = True
            last_t[key] = row["t"]
    triggers = triggers.loc[keep_mask].reset_index(drop=True)

    # Build PR lookahead index: (train, signal) → sorted PR times
    pr_index: dict[tuple[str, str], np.ndarray] = {}
    pr = decision_events[["time", "train_id", "signal_no"]].copy()
    pr["train_id"]  = pr["train_id"].astype(str)
    pr["signal_no"] = pr["signal_no"].astype(str)
    pr["time_ns"]   = pd.to_datetime(pr["time"]).astype("int64")
    for (tr, sig), sub in pr.groupby(["train_id", "signal_no"]):
        pr_index[(tr, sig)] = np.sort(sub["time_ns"].to_numpy())

    # For each trigger, check if any PR for (T, S) falls in [t, t + Δ_WAIT]
    wait_ns_delta = int(delta_wait_seconds * 1e9)
    triggers_t_ns = triggers["t"].astype("int64").to_numpy()
    is_wait = np.ones(len(triggers), dtype=bool)
    for i, row in triggers.iterrows():
        key = (row["focal_train"], row["focal_signal"])
        prs = pr_index.get(key)
        if prs is None:
            continue
        t_ns = triggers_t_ns[i]
        # Find any PR in [t, t + Δ_WAIT]
        lo = np.searchsorted(prs, t_ns, side="left")
        if lo < prs.size and prs[lo] <= t_ns + wait_ns_delta:
            is_wait[i] = False  # PR exists in window → not a wait sample

    waits = triggers.loc[is_wait].copy()
    waits["label"] = "wait"
    waits["chosen_route_id"] = None
    waits["trigger_type"] = "approach"
    return waits[["focal_train", "focal_signal", "t", "label",
                   "chosen_route_id", "trigger_type"]]


# ============================================================
# Public entry point
# ============================================================

def generate_decision_points(
    decision_events: pd.DataFrame,
    td_events: pd.DataFrame,
    routes_clean: pd.DataFrame,
    k_approach: int = None,
    delta_wait_seconds: float = None,
) -> pd.DataFrame:
    """Generate the unified decision_points table per spec 02 §2.

    Args:
        decision_events: outputs/decisions/decision_events.parquet
        td_events: TD events with columns time, type, id, state, trainid_filled
                   (typically outputs/cache/td_data.parquet)
        routes_clean: outputs/infrastructure/routes_clean.parquet
        k_approach: approach horizon depth, default spec 02 §2.3 (2)
        delta_wait_seconds: wait trigger window, default spec 02 §2.3 (30s)

    Returns:
        DataFrame with columns:
          focal_train, focal_signal, t, label, chosen_route_id, trigger_type

        Expected ~727k rows: ~546k set + ~181k wait.
    """
    if k_approach is None:
        k_approach = C.APPROACH_K_HOPS
    if delta_wait_seconds is None:
        delta_wait_seconds = C.DECISION_LOOKAHEAD_SECONDS

    t0 = _time.time()
    print(f"  [trigger] computing approach horizons (k_approach={k_approach})")
    approach_tracks = compute_approach_tracks(routes_clean, k_hops=k_approach)
    n_signals_with_approach = len(approach_tracks)
    print(f"  [trigger]   → {n_signals_with_approach} signals with approach tracks")

    print(f"  [trigger] extracting set triggers (PR events)")
    sets = _extract_set_triggers(decision_events)
    print(f"  [trigger]   → {len(sets):,} set samples")

    print(f"  [trigger] extracting wait triggers "
          f"(Δ_wait={delta_wait_seconds}s)")
    waits = _extract_wait_triggers(
        td_events=td_events,
        decision_events=decision_events,
        approach_tracks=approach_tracks,
        delta_wait_seconds=delta_wait_seconds,
    )
    print(f"  [trigger]   → {len(waits):,} wait samples")

    combined = pd.concat([sets, waits], ignore_index=True)
    combined = combined.sort_values(["t", "focal_train", "focal_signal"]).reset_index(drop=True)

    elapsed = _time.time() - t0
    print(f"  [trigger] total: {len(combined):,} decision points "
          f"(set={len(sets):,}, wait={len(waits):,}) in {elapsed:.1f}s")
    return combined


def summarize(dp: pd.DataFrame) -> dict:
    """Summary stats for the decision_points table."""
    summary = {
        "n_total":             int(len(dp)),
        "n_set":               int((dp["label"] == "set").sum()),
        "n_wait":              int((dp["label"] == "wait").sum()),
        "n_unique_trains":     int(dp["focal_train"].nunique()),
        "n_unique_signals":    int(dp["focal_signal"].nunique()),
        "neg_pos_ratio":       round(
            (dp["label"] == "wait").sum() / max((dp["label"] == "set").sum(), 1), 3),
        "by_trigger":          dp["trigger_type"].value_counts().to_dict(),
        "time_range":          [str(dp["t"].min()), str(dp["t"].max())],
    }
    # Quartile on per-train decision count
    per_train = dp.groupby("focal_train").size()
    summary["per_train_decisions"] = {
        "mean":   float(per_train.mean()),
        "median": float(per_train.median()),
        "p25":    float(per_train.quantile(0.25)),
        "p75":    float(per_train.quantile(0.75)),
        "max":    int(per_train.max()),
    }
    return summary
