"""P2.6 simulator — estimate the 4 empirical parameter tables (spec 05 §14.2).

Outputs outputs/simulator/parameters.json with {p25,p50,p75,p95} per cell (rollout
uses p50; ablations p25/p75). Read-only inputs: TD (occupancy/aspect), corrected
Movements (dwell), Derby_info (route running-time fallback). Pure pandas/pyarrow.

Tables (spec 05 §14.2):
  route_running_time(route_id)   — Derby_info gap_time(s) per route (class-agnostic v1;
                                    spec §6.5 fallback. Empirical per-traversal refinement = future).
  platform_dwell_time(platform,class) — Movements ARRIVAL→DEPARTURE delta per train×platform.
  min_headway(track_id)          — gaps between successive DIFFERENT-train occupation onsets per TC.
  aspect_clear_lag(signal_id)    — best-effort: signal red→green (state 1→0) clearing lag per signal.

Run on Windows (TD is 94MB parquet):
    python scripts/simulator/01_estimate_parameters.py
Then: scripts/simulator/02_validate_simulator.py (the PRIMARY gate, spec §14.6.1).
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from railrl import config as C
from railrl.data_io import load_movements


def pctl(arr) -> dict:
    """{n,min,p5,p10,p25,p50,p75,p95}. NOTE: the simulator uses p50 for
    route_running_time/platform_dwell/aspect_clear_lag (typical value), but **p5**
    for min_headway (the physical minimum spacing — p50 is the off-peak-dominated
    typical gap, far too large to use as a headway constraint). See §14.2 / 14.6.1."""
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"n": 0}
    return {"n": int(a.size), "min": float(a.min()),
            "p5": float(np.percentile(a, 5)), "p10": float(np.percentile(a, 10)),
            "p25": float(np.percentile(a, 25)), "p50": float(np.percentile(a, 50)),
            "p75": float(np.percentile(a, 75)), "p95": float(np.percentile(a, 95))}


def route_running_time() -> dict:
    print("[1/4] route_running_time ← Derby_info gap_time(s) (class-agnostic v1)")
    di = pd.read_csv(C.DERBY_INFO_CSV)
    rcol = next((c for c in di.columns if c.lower() == "routeid"), None)
    gcol = next((c for c in di.columns if "gap_time" in c.lower()), None)
    out = {}
    for _, r in di.iterrows():
        rid = str(r[rcol])
        g = r[gcol]
        if pd.notna(g):
            out[rid] = {"p50": float(g)}
    print(f"      {len(out)}/{len(di)} routes with gap_time")
    return out


def platform_dwell() -> dict:
    print("[2/4] platform_dwell_time ← Movements ARRIVAL→DEPARTURE (corrected clock)")
    mv = load_movements()                       # applies the BST fix
    ev = next((c for c in mv.columns if c.lower() == "event_type"), None)
    pcol = next((c for c in mv.columns if c.lower() == "platform"), None)
    mv = mv.dropna(subset=["actual_timestamp", "train_id"]).copy()
    mv["actual"] = pd.to_datetime(mv["actual_timestamp"], errors="coerce")
    mv["plat"] = pd.to_numeric(mv[pcol], errors="coerce")
    mv["cls"] = mv["train_id"].astype(str).str[2]
    mv["evt"] = mv[ev].astype(str).str.upper()
    cells: dict = {}
    for (tid, plat), sub in mv.dropna(subset=["plat"]).groupby(["train_id", "plat"]):
        arr = sub.loc[sub["evt"] == "ARRIVAL", "actual"]
        dep = sub.loc[sub["evt"] == "DEPARTURE", "actual"]
        if len(arr) and len(dep):
            d = (dep.min() - arr.min()).total_seconds()
            if 0 <= d <= 3600:                  # sane dwell (≤1h)
                key = (int(plat), str(sub["cls"].iloc[0]))
                cells.setdefault(key, []).append(d)
    out = {f"{p}|{c}": pctl(v) for (p, c), v in cells.items()}
    print(f"      {len(out)} (platform×class) cells populated")
    return out


def _td_track(td):
    """From TD Track events, per TC: min_headway (different-train onset gaps) +
    tc_traversal_time (occupation duration = 0→1 onset to next 1→0 clear)."""
    print("[3/4] min_headway + tc_traversal_time ← TD Track events")
    tk = td[td["type"] == "Track"]
    headway: dict = {}
    traversal: dict = {}
    for tc, sub in tk.groupby("id", observed=True, sort=False):
        sub = sub.sort_values("time")
        st = sub["state"].fillna(0).astype("int8").to_numpy()
        tr = sub["trainid_filled"].astype(str).to_numpy()
        tns = sub["time"].values.astype("datetime64[ns]").astype("int64")
        onset = np.where((st[1:] == 1) & (st[:-1] == 0))[0] + 1     # 0→1 occupy
        offset = np.where((st[1:] == 0) & (st[:-1] == 1))[0] + 1    # 1→0 clear
        # headway: gaps between successive DIFFERENT-train onsets
        if onset.size >= 2:
            ot, otr = tns[onset], tr[onset]
            gaps = [(ot[k] - ot[k - 1]) / 1e9 for k in range(1, len(ot))
                    if otr[k] != otr[k - 1] and otr[k] not in ("nan", "0", "")]
            gaps = [g for g in gaps if 0 < g < 3600]
            if gaps:
                headway[str(tc)] = pctl(gaps)
        # traversal: each onset → next offset (occupation duration)
        if onset.size and offset.size:
            durs = []
            for on in onset:
                aft = offset[offset > on]
                if aft.size:
                    d = (tns[aft[0]] - tns[on]) / 1e9
                    if 0 < d < 1800:
                        durs.append(d)
            if durs:
                traversal[str(tc)] = pctl(durs)
    print(f"      min_headway: {len(headway)} cells | tc_traversal_time: {len(traversal)} cells")
    return headway, traversal


def _td_aspect_clear_lag(td) -> dict:
    print("[4/4] aspect_clear_lag ← TD Signal red→green (1→0) clearing lag (best-effort)")
    sg = td[td["type"] == "Signal"]
    cells: dict = {}
    for sid, sub in sg.groupby("id", observed=True, sort=False):
        sub = sub.sort_values("time")
        st = sub["state"].fillna(0).astype("int8").to_numpy()
        tns = sub["time"].values.astype("datetime64[ns]").astype("int64")
        red_on = np.where((st[1:] == 1) & (st[:-1] == 0))[0] + 1     # →red
        lags = []
        for k in red_on:
            nxt = np.where((st[k:] == 0))[0]                          # next green
            if nxt.size:
                lag = (tns[k + nxt[0]] - tns[k]) / 1e9
                if 0 < lag < 1800:
                    lags.append(lag)
        if lags:
            cells[str(sid)] = pctl(lags)
    print(f"      {len(cells)} signal_id cells populated")
    return cells


def main() -> int:
    params = {"route_running_time": route_running_time(),
              "platform_dwell_time": platform_dwell()}
    print("loading TD (94MB parquet) ...")
    td = pd.read_parquet(C.TD_PARQUET, columns=["time", "type", "id", "state", "trainid_filled"])
    print(f"      {len(td):,} TD events")
    params["min_headway"], params["tc_traversal_time"] = _td_track(td)
    params["aspect_clear_lag"] = _td_aspect_clear_lag(td)

    out = C.SIMULATOR_DIR / "parameters.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(params, indent=2))
    print("\n=== coverage summary ===")
    for k, tbl in params.items():
        ncells = len(tbl)
        stat = "p5" if k == "min_headway" else "p50"   # min_headway uses the low end
        vals = [v.get(stat) for v in tbl.values()
                if isinstance(v, dict) and v.get(stat) is not None]
        med = float(np.median(vals)) if vals else float("nan")
        note = " (gap_time)" if k == "route_running_time" else (
            " ← headway FLOOR" if k == "min_headway" else "")
        print(f"  {k:22s}: {ncells:>4} cells | median {stat} = {med:.1f} s{note}")
    print(f"\n→ wrote {out}")
    print("Next: scripts/simulator/02_validate_simulator.py (PRIMARY gate: throughput/"
          "occupancy Spearman>0.6 on held-out month — spec §14.6.1).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
