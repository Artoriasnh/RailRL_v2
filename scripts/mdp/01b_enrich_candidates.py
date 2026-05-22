"""Stage 2 — Enrich decision_points with the candidate action set.

THE ACTION SPACE.  Each decision point's structured action space is
    A_t = {wait} ∪ {(focal_train, R) | R ∈ candidates(focal_signal)}

01_generate_decision_points.py produced only 6 columns (focal_train,
focal_signal, t, label, chosen_route_id, trigger_type). The candidate routes
(the actual action set the model chooses from) were computed for a coverage
check in 02 but NEVER persisted. This script materialises them so the
snapshot builder (05) and the model have a real action space.

Candidate rule (spec 02 §3.2 Rule 1 — the dominant one):
    candidates = routes that START at focal_signal (route_index.routes_from).
Empirically this covers 99.99% of chosen routes with a clean small set
(mean 2.7, median 2, max 13). Direction / prev-route soft filters (Rules 2-4)
are NOT applied as hard candidate filters — they would risk dropping the
chosen route and only need TD; they can be re-introduced as model features.

Action index convention (LOCKED — Stage 4 reader depends on it):
    action 0      = wait        (always present)
    action 1..K   = candidate_route_ids[0..K-1]  (stored order)
    chosen_action_idx = 0 if label=='wait'
                        else 1 + candidate_route_ids.index(chosen_route_id)
For set rows whose chosen route is somehow not in routes_from(focal_signal)
(~0.01%), we APPEND it so the BC/CQL target is always valid.

Usage:
    python scripts/mdp/01b_enrich_candidates.py
    python scripts/mdp/01b_enrich_candidates.py --out some_other.parquet

Output: rewrites decision_points_v2.parquet with 3 new columns:
    candidate_route_ids : list[str]
    n_candidates        : int32
    chosen_action_idx   : int32
"""
from __future__ import annotations
import argparse
import ast
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from railrl import config as C


def _parse_track_list(v):
    if v is None:
        return []
    if isinstance(v, (list, np.ndarray)):
        return [str(x) for x in v]
    if isinstance(v, str):
        try:
            parsed = ast.literal_eval(v)
            return [str(x) for x in parsed] if isinstance(parsed, (list, tuple)) else []
        except Exception:
            return []
    return []


def _build_routes_from(routes_clean: pd.DataFrame) -> dict[str, list[str]]:
    """signal_id -> ordered list of route_ids that start there (deterministic)."""
    by_sig: dict[str, list[str]] = defaultdict(list)
    start_col = "start_signals" if "start_signals" in routes_clean.columns else "signal_no"
    for _, r in routes_clean.iterrows():
        rid = str(r["route_id"])
        starts = r[start_col]
        if isinstance(starts, (list, np.ndarray)):
            sig_list = [str(s) for s in starts]
        else:
            sig_list = [str(starts)]
        for s in sig_list:
            by_sig[s].append(rid)
    # Deterministic order: sort route_ids per signal
    for s in by_sig:
        by_sig[s] = sorted(set(by_sig[s]))
    return by_sig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default=None,
                    help="Override output path (default: overwrite decision_points_v2.parquet)")
    args = ap.parse_args()
    out_path = Path(args.out) if args.out else C.DECISION_POINTS_V2_PARQUET

    t0 = time.time()
    print("=" * 70)
    print("Enrich decision_points with candidate action set")
    print("=" * 70)

    print(f"[1/4] loading decision_points: {C.DECISION_POINTS_V2_PARQUET}")
    dp = pd.read_parquet(C.DECISION_POINTS_V2_PARQUET)
    print(f"      {len(dp):,} rows, cols={dp.columns.tolist()}")

    print(f"[2/4] loading routes_clean: {C.ROUTES_CLEAN_PARQUET}")
    rc = pd.read_parquet(C.ROUTES_CLEAN_PARQUET)
    by_sig = _build_routes_from(rc)
    print(f"      {len(by_sig):,} start-signals indexed")

    print(f"[3/4] computing candidates for {len(dp):,} decisions...")
    focal_signal = dp["focal_signal"].astype(str).to_numpy()
    label = dp["label"].astype(str).to_numpy()
    chosen = dp["chosen_route_id"].astype("object").to_numpy()

    cand_lists: list[list[str]] = []
    n_cands = np.empty(len(dp), dtype=np.int32)
    chosen_idx = np.empty(len(dp), dtype=np.int32)
    n_chosen_appended = 0
    n_chosen_missing_route = 0

    for i in range(len(dp)):
        sig = focal_signal[i]
        cands = list(by_sig.get(sig, ()))  # copy (deterministic, sorted)
        if label[i] == "set":
            ch = chosen[i]
            ch = None if (ch is None or (isinstance(ch, float) and np.isnan(ch))) else str(ch)
            if ch is None or ch == "" or ch == "None":
                # set row without a usable chosen route — degenerate; idx -1
                chosen_idx[i] = -1
                n_chosen_missing_route += 1
            else:
                if ch not in cands:
                    cands.append(ch)
                    n_chosen_appended += 1
                chosen_idx[i] = 1 + cands.index(ch)   # action 0 = wait
        else:
            chosen_idx[i] = 0  # wait
        cand_lists.append(cands)
        n_cands[i] = len(cands)

    dp["candidate_route_ids"] = cand_lists
    dp["n_candidates"] = n_cands
    dp["chosen_action_idx"] = chosen_idx

    print(f"      candidate-set size: mean={n_cands.mean():.2f} "
          f"median={int(np.median(n_cands))} max={int(n_cands.max())} "
          f"zero%={100*(n_cands==0).mean():.1f}")
    print(f"      chosen appended (not in routes_from): {n_chosen_appended:,} "
          f"({100*n_chosen_appended/max(1,(label=='set').sum()):.3f}% of set)")
    if n_chosen_missing_route:
        print(f"      [warn] set rows with no usable chosen_route_id: {n_chosen_missing_route:,}")

    print(f"[4/4] writing {out_path}")
    dp.to_parquet(out_path, index=False, compression="zstd")

    # Summary
    import json
    summary = {
        "n_decisions":          int(len(dp)),
        "n_set":                int((label == "set").sum()),
        "n_wait":               int((label == "wait").sum()),
        "candidate_size_mean":  round(float(n_cands.mean()), 3),
        "candidate_size_max":   int(n_cands.max()),
        "wait_only_pct":        round(100 * float((n_cands == 0).mean()), 2),
        "chosen_appended":      int(n_chosen_appended),
        "set_no_chosen_route":  int(n_chosen_missing_route),
        "elapsed_seconds":      round(time.time() - t0, 1),
    }
    summary_path = out_path.parent / "decision_points_candidates_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print("-" * 70)
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print("-" * 70)
    print(f"DONE in {time.time() - t0:.1f}s → {out_path}")
    print("Now re-run 05_build_snapshots.py to pick up the action set.")
    print("=" * 70)


if __name__ == "__main__":
    main()
