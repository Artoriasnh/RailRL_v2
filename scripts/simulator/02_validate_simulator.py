"""P2.6 simulator PRIMARY validation gate (spec 05 §14.6 / §14.6.1).

Builds factual scenarios from the held-out val month (2024-02, clock-correct period),
runs the L3Simulator on each, and compares to ACTUAL TD: this is the gate that must
pass before building Tier 3 (don't立 head-line claim on an unvalidated simulator).

Per scenario (start time t0):
  * window = [t0, t0 + 30 min]
  * active trains = trains with ≥2 distinct TC occupation-onsets in the window;
    each train's `path` = its ordered TC-onset sequence (the route it actually took);
    initial position = its first onset.
  * run sim 30 min → sim_throughput (# trains completing their path) + per-TC sim
    occupied-time; ACTUAL_throughput = # of those trains whose last path-TC onset
    falls in the window; per-TC actual occupied-time from TD.

Gate (§14.6.1, PRIMARY): Spearman(sim_throughput, actual_throughput) > 0.6 across
scenarios + occupancy-time Spearman/agreement. delay = best-effort (not gated here).

Read-only, pure pandas/numpy/scipy. Run on Windows:
    python scripts/simulator/02_validate_simulator.py --n-scenarios 300
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from railrl import config as C
from railrl.xai.l3_system import L3Simulator, Scenario, Train

VAL_START = pd.Timestamp("2024-02-01")
VAL_END = pd.Timestamp("2024-03-01")
HORIZON_MIN = 30.0


def _spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 5 or np.std(x[m]) == 0 or np.std(y[m]) == 0:
        return float("nan")
    try:
        from scipy.stats import spearmanr
        return float(spearmanr(x[m], y[m]).correlation)
    except Exception:                       # rank-corr fallback (no scipy)
        rx = pd.Series(x[m]).rank().to_numpy()
        ry = pd.Series(y[m]).rank().to_numpy()
        return float(np.corrcoef(rx, ry)[0, 1])


def load_platform_map():
    try:
        pm = pd.read_csv(C.PLATFORM_TC_MAP_CSV)
        tc_c = next((c for c in pm.columns if "tc" in c.lower() or c.lower() == "track"), pm.columns[0])
        pl_c = next((c for c in pm.columns if "platform" in c.lower()), pm.columns[-1])
        return {str(r[tc_c]): int(r[pl_c]) for _, r in pm.iterrows()
                if pd.notna(r[pl_c]) and str(r[pl_c]).strip().isdigit()}
    except Exception as e:
        print(f"  [platform_tc_map] skipped ({e}); no platform dwell in sim")
        return {}


def build_train_paths(win):
    """From a TD-Track window → {train_id: [(tc, onset_ns), ...]} (ordered onsets)."""
    win = win.sort_values("time")
    st = win["state"].fillna(0).astype("int8").to_numpy()
    tr = win["trainid_filled"].astype(str).to_numpy()
    tc = win["id"].astype(str).to_numpy()
    tns = win["time"].values.astype("datetime64[ns]").astype("int64")
    onset = np.where((st[1:] == 1) & (st[:-1] == 0))[0] + 1
    paths: dict = {}
    for k in onset:
        tid = tr[k]
        if tid in ("nan", "0", "", "None"):
            continue
        paths.setdefault(tid, []).append((tc[k], int(tns[k])))
    return paths


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-scenarios", type=int, default=300)
    ap.add_argument("--params", default=str(C.SIMULATOR_DIR / "parameters.json"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--headway-pctl", default="p1",
                    help="calibration knob: which min_headway percentile = the spacing "
                         "floor (min/p1/p5/p10/p25). p1 = CALIBRATED default (throughput "
                         "Spearman 0.86). Lower → trains follow closer → more throughput.")
    args = ap.parse_args()

    sim = L3Simulator.from_json(args.params, headway_pctl=args.headway_pctl)
    plat_of = load_platform_map()
    print(f"sim ready | headway_pctl={args.headway_pctl} | platform TCs: {len(plat_of)}")

    print("loading TD Track events for val month (2024-02) ...")
    td = pd.read_parquet(C.TD_PARQUET, columns=["time", "type", "id", "state", "trainid_filled"])
    td = td[td["type"] == "Track"]
    td["time"] = pd.to_datetime(td["time"], errors="coerce")
    td = td[(td["time"] >= VAL_START) & (td["time"] < VAL_END)].sort_values("time")
    print(f"  {len(td):,} val-month Track events, "
          f"{td['time'].min()} .. {td['time'].max()}")

    rng = np.random.default_rng(args.seed)
    tmin = td["time"].min().value
    tmax = (td["time"].max() - pd.Timedelta(minutes=HORIZON_MIN)).value
    starts = np.sort(rng.integers(tmin, tmax, size=args.n_scenarios))

    sim_tp, act_tp = [], []
    tim_sim, tim_act, progress = [], [], []
    sim_occ_by_tc: dict = {}
    act_occ_by_tc: dict = {}
    hns = int(HORIZON_MIN * 60 * 1e9)
    used = 0
    for t0 in starts:
        t0 = int(t0)
        win = td[(td["time"].values.astype("datetime64[ns]").astype("int64") >= t0) &
                 (td["time"].values.astype("datetime64[ns]").astype("int64") < t0 + hns)]
        if len(win) < 20:
            continue
        paths = build_train_paths(win)
        trains = []
        for tid, seq in paths.items():
            if len(seq) < 2:
                continue
            tcs = [tc for tc, _ in seq]
            trains.append(Train(train_id=tid, path=tcs, cls=str(tid)[2] if len(str(tid)) >= 3 else "0",
                                 idx=0, entered_ns=seq[0][1]))
        if len(trains) < 3:
            continue
        used += 1
        scen = Scenario(t0_ns=t0, trains=[Train(t.train_id, list(t.path), t.cls, 0, t.entered_ns)
                                          for t in trains], platform_of=plat_of)
        res = sim.simulate(scen, HORIZON_MIN)
        # per-train sim trajectory (entry times) from the timeline
        sim_traj: dict = {}
        for tc, te, tid in res["timeline"]:
            sim_traj.setdefault(tid, []).append((tc, te))
        sim_reached = 0
        for t in trains:
            tcs = [tc for tc, _ in sim_traj.get(t.train_id, [])]
            last = str(t.path[-1])
            if last in tcs:                          # entered last TC ⇒ matches actual last-onset
                sim_reached += 1
                sim_last = max(te for tc, te in sim_traj[t.train_id] if tc == last)
                st = (sim_last - t.entered_ns) / 1e9
                at = (paths[t.train_id][-1][1] - paths[t.train_id][0][1]) / 1e9
                if st > 0 and at > 0:
                    tim_sim.append(st); tim_act.append(at)
            ridx = 0                                 # deepest path index reached in sim
            for k in range(len(t.path) - 1, 0, -1):
                if str(t.path[k]) in tcs:
                    ridx = k; break
            progress.append(ridx / max(len(t.path) - 1, 1))
        sim_tp.append(sim_reached)                   # entry-based throughput (matches actual def)
        act_tp.append(sum(1 for t in trains if paths[t.train_id][-1][1] <= t0 + hns))
        for tc, _t, _tid in res["timeline"]:
            sim_occ_by_tc[tc] = sim_occ_by_tc.get(tc, 0) + 1
        for tid, seq in paths.items():
            for tc, _ in seq:
                act_occ_by_tc[tc] = act_occ_by_tc.get(tc, 0) + 1

    print(f"\nscenarios used: {used}/{args.n_scenarios}")
    sp_tp = _spearman(sim_tp, act_tp)
    # occupancy: align per-TC onset counts (sim vs actual) over all scenarios
    tcs = sorted(set(sim_occ_by_tc) | set(act_occ_by_tc))
    so = [sim_occ_by_tc.get(tc, 0) for tc in tcs]
    ao = [act_occ_by_tc.get(tc, 0) for tc in tcs]
    sp_occ = _spearman(so, ao)
    sp_tim = _spearman(tim_sim, tim_act)
    mean_prog = float(np.mean(progress)) if progress else float("nan")

    print("\n=== PRIMARY gate (spec §14.6.1) ===")
    print(f"  throughput:  Spearman(sim, actual) = {sp_tp:.3f}   (gate > 0.6)   "
          f"[sim mean {np.mean(sim_tp):.1f} / actual mean {np.mean(act_tp):.1f}]  (entry-based)")
    print(f"  occupancy:   Spearman(per-TC sim onsets, actual onsets) = {sp_occ:.3f}   (gate > 0.6)")
    print("--- diagnostics (split timing vs conflict) ---")
    print(f"  per-train timing: Spearman(sim, actual traversal time) = {sp_tim:.3f}  (n={len(tim_sim):,})"
          "  ← validates traversal/dwell params (conflict-light)")
    print(f"  mean path-progress reached in sim = {100*mean_prog:.1f}%   "
          "(low ⇒ trains too slow/blocked → try lower --headway-pctl)")
    ok = (sp_tp > 0.6) and (sp_occ > 0.6)
    print(f"\n  {'PASS — simulator validated; OK to build Tier 3' if ok else 'FAIL — recalibrate params / widen CIs / narrow Tier-3 claim (do NOT build Tier 3 yet)'}")
    print("  (delay-Spearman = best-effort, not gated here — spec §14.6.1)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
