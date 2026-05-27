"""Tier 3 — Replicate-AND-Improve 4-way (spec 05 §4) using the L3 simulator.

The paper's head-line claim: when the CQL model DIVERGES from the signaller, are its
choices IMPROVEMENTS? Pipeline:
  1. Forward pass over test set-decisions → model route (Q-argmax candidate) vs
     signaller route (chosen_route_id).  [cheap; prints disagreement breakdown]
  2. For a sample of decisions, build a scenario at t (other active trains' actual
     paths from TD) and roll the focal train under MODEL-route vs SIGNALLER-route
     (route→TCs from route_to_tc) with the validated L3Simulator → l3_delta.
  3. Classify SAFETY-FIRST (v1.2): genuine_unsafe → conflict_indeterminate → delay tier
     (improving/delay_worse/neutral). Safety ≫ delay; an unsafe divergence is intolerable.

⚠️ modelling choices (v1.2, 2026-05-25 — after the asymmetry diagnostic):
  * focal path under route R = route_to_tc track_list[R]; other trains held fixed
    (standard counterfactual — others don't re-route).
  * SAFETY is judged ONLY from simulator-INDEPENDENT signals: route legality (model route
    ∈ candidate set, an action-space guarantee) + alone-feasibility (route completes when
    run with NO others). This is because the fixed-others rollout CANNOT fairly adjudicate
    conflict-safety: the diagnostic showed 100% of with-others non-completions DO complete
    alone → a fixed-others "unsafe" verdict is the de-confliction asymmetry, not the route.
  * DELAY is judged on the FAIR symmetric ALONE finish-Δ (both routes run in isolation),
    so the asymmetry can't bias it; this honestly reports the model's mild intrinsic +14s
    slower tendency rather than the clamp-polluted with-others +33s. δ tiebreak = --delta-s;
    the spec's δ=0.5 *reward units* (r_total on trajectory) is a later refinement.
  * sim absolute throughput is ~73% of actual (timing-conservative) but BIAS CANCELS
    in the a−b delta (validated §14.6.1: occupancy 0.94 / throughput Spearman 0.86).
Run on Windows (needs seed42 model):  python scripts/eval/03_tier3_replicate_improve.py --max-decisions 1500
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch
from torch.utils.data import DataLoader

from railrl import config as C
from railrl.encoders.input_pipeline import NormStats
from railrl.algorithms.transitions import StreamingTransitionDataset
from railrl.data_io import load_route_to_tc
from railrl.model import RailRLModel
from railrl.xai.l3_system import L3Simulator, Scenario, Train, l3_delta

VAL = (pd.Timestamp("2024-03-01"), pd.Timestamp("2024-04-26"))   # test period (clock-correct)
HORIZON_MIN = 30.0


@torch.no_grad()
def forward_pass(model, loader, device):
    """Per decision: sample_id, chosen_action_idx (signaller), model_action_idx (Q-argmax)."""
    sid, chosen, marg = [], [], []
    for bs, _bsp, _done in loader:
        bs = bs.to(device)
        out = model(bs)
        B = bs.num_graphs
        sid.append(bs.sample_id.view(B).cpu().numpy())
        chosen.append(bs.chosen_action_idx.view(B).cpu().numpy())
        marg.append(out["Q"].argmax(1).cpu().numpy())
    return (np.concatenate(sid), np.concatenate(chosen), np.concatenate(marg))


def build_platform_map():
    try:
        pm = pd.read_csv(C.PLATFORM_TC_MAP_CSV)
        tcc = next((c for c in pm.columns if "tc" in c.lower() or c.lower() == "track"), pm.columns[0])
        plc = next((c for c in pm.columns if "platform" in c.lower()), pm.columns[-1])
        return {str(r[tcc]): int(r[plc]) for _, r in pm.iterrows()
                if pd.notna(r[plc]) and str(r[plc]).strip().isdigit()}
    except Exception:
        return {}


def other_trains_at(td_win, exclude_tid):
    """Active trains in a TD-Track window → [Train(path from actual onsets)], excl. focal."""
    td_win = td_win.sort_values("time")
    st = td_win["state"].fillna(0).astype("int8").to_numpy()
    tr = td_win["trainid_filled"].astype(str).to_numpy()
    tc = td_win["id"].astype(str).to_numpy()
    tns = td_win["time"].values.astype("datetime64[ns]").astype("int64")
    onset = np.where((st[1:] == 1) & (st[:-1] == 0))[0] + 1
    paths: dict = {}
    for k in onset:
        tid = tr[k]
        if tid in ("nan", "0", "", "None") or tid == exclude_tid:
            continue
        paths.setdefault(tid, []).append((tc[k], int(tns[k])))
    out = []
    for tid, seq in paths.items():
        if len(seq) >= 2:
            out.append(Train(train_id=tid, path=[c for c, _ in seq],
                             cls=tid[2] if len(tid) >= 3 else "0", idx=0, entered_ns=seq[0][1]))
    return out


def _focal_outcome(res, focal, last_tc, t0_ns, horizon_min):
    """From a sim result: (focal completed its route?, focal finish_ns or horizon)."""
    fin = None
    for tc, te, tid in res["timeline"]:
        if tid == focal and tc == last_tc:
            fin = te if fin is None else max(fin, te)
    if fin is not None:
        return True, fin
    return False, t0_ns + int(horizon_min * 60 * 1e9)   # didn't finish → horizon


def classify(legal, cm1, fm1, cs1, fs1, cm, delta_s):
    """SAFETY-FIRST decomposition (spec §4.3, v1.2). Safety STRICTLY dominates delay and
    is judged ONLY from simulator-INDEPENDENT signals — route legality + alone-feasibility —
    because the fixed-others rollout cannot fairly adjudicate conflict-safety for a
    counterfactual route (diagnostic 2026-05-25: 100% of with-others non-completions DO
    complete when run ALONE → 'unsafe' under fixed-others is an asymmetry artifact, not the
    route). Per Hao: an unsafe divergence is intolerable, so genuine_unsafe must be a clean,
    trustworthy measure — never the conflict-confounded one.
      genuine_unsafe         : model route ILLEGAL (not a candidate) OR infeasible even ALONE
                               (physically can't complete) → intolerable; expected ~0.
      conflict_indeterminate : feasible alone but blocked only WITH the (signaller-
                               de-conflicted) fixed others → cannot adjudicate; reported as-is.
      improving/delay_worse/neutral : both routes safe → FAIR symmetric ALONE finish-Δ
                               (free of the fixed-others bias; honestly shows model's real
                               timing, incl. its mild intrinsic +14s slower tendency)."""
    if not legal or not cm1:
        return "genuine_unsafe"
    if not cm:                                 # feasible alone, fails only WITH fixed others
        return "conflict_indeterminate"
    if not cs1:                                # signaller route not feasible alone (rare) → no fair Δ
        return "neutral"
    fd = (fm1 - fs1) / 1e9                      # FAIR alone finish-Δ (s); negative = model faster
    if fd < -delta_s:
        return "improving"
    if fd > delta_s:
        return "delay_worse"
    return "neutral"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--max-decisions", type=int, default=1500, help="L3 sample cap (sim is ~slow)")
    ap.add_argument("--delta-s", type=float, default=30.0, help="delay tiebreak threshold (s)")
    ap.add_argument("--params", default=str(C.SIMULATOR_DIR / "parameters.json"))
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    try:
        import torch.multiprocessing as _mp
        _mp.set_sharing_strategy("file_system")
    except Exception:
        pass
    device = "cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu"
    ckpt = Path(args.ckpt) if args.ckpt else (C.TRAIN_DIR / f"cql_seed{args.seed}" / "best.pt")

    # ---- 1. model forward pass over test ----
    stats = NormStats.load(C.NORMALIZATION_STATS_JSON)
    model = RailRLModel.build(stats).to(device)
    ck = torch.load(str(ckpt), map_location=device, weights_only=False)
    model.load_state_dict(ck["model"] if isinstance(ck, dict) and "model" in ck else ck)
    model.eval()
    ds = StreamingTransitionDataset(C.SNAPSHOTS_V2_PARQUET, C.NORMALIZATION_STATS_JSON,
                                    split="test", batch_size=C.BATCH_SIZE, shuffle=False, stratified=False)
    print(f"forward pass (test) | ckpt={ckpt.name} ...")
    sid, chosen, marg = forward_pass(model, DataLoader(ds, batch_size=None, num_workers=args.num_workers), device)

    pred = pd.DataFrame({"sample_id": sid, "chosen_idx": chosen, "model_idx": marg})
    import pyarrow.parquet as pq
    ident = pq.read_table(str(C.SNAPSHOTS_V2_PARQUET),
                          columns=["sample_id", "focal_train", "t", "chosen_route_id",
                                   "chosen_action_idx", "candidate_route_ids"]).to_pandas()
    df = pred.merge(ident, on="sample_id", how="left")
    setd = df[df["chosen_action_idx"] > 0].copy()         # set decisions only

    def model_route(row):
        mi = int(row["model_idx"])
        cands = row["candidate_route_ids"]
        cands = list(cands) if cands is not None else []
        if mi <= 0 or mi - 1 >= len(cands):
            return None                                    # model says wait / OOB
        return str(cands[mi - 1])
    setd["model_route"] = setd.apply(model_route, axis=1)
    setd["sig_route"] = setd["chosen_route_id"].astype(str)
    setd["disagree"] = setd["model_route"].astype(str) != setd["sig_route"]

    n_set = len(setd); n_dis = int(setd["disagree"].sum())
    print(f"\n=== disagreement breakdown (set decisions) ===")
    print(f"  set decisions: {n_set:,}")
    print(f"  model≠signaller route: {n_dis:,} ({100*n_dis/max(n_set,1):.1f}%)  "
          f"[of which model=wait: {int((setd['disagree'] & setd['model_route'].isna()).sum()):,}]")

    # ---- 2-3. L3 counterfactual on a sample of disagreements ----
    rt = load_route_to_tc()
    route_tcs = {str(r["route"]): [str(x) for x in r["track_list"]]
                 for _, r in rt.iterrows() if isinstance(r["track_list"], list) and r["track_list"]}
    plat_of = build_platform_map()
    sim = L3Simulator.from_json(args.params)               # headway_pctl=p1 (calibrated)

    dis = setd[setd["disagree"] & setd["model_route"].notna()].copy()
    dis = dis[dis["model_route"].isin(route_tcs) & dis["sig_route"].isin(route_tcs)]
    if len(dis) > args.max_decisions:
        dis = dis.sample(args.max_decisions, random_state=0)
    print(f"\nL3 counterfactual on {len(dis):,} disagreements (both routes have TC lists) ...")

    print("loading TD Track (test period) ...")
    td = pq.read_table(str(C.TD_PARQUET), columns=["time", "type", "id", "state", "trainid_filled"]).to_pandas()
    td = td[td["type"] == "Track"]
    td["time"] = pd.to_datetime(td["time"], errors="coerce")
    td = td[(td["time"] >= VAL[0]) & (td["time"] < VAL[1])]
    td_ns = td["time"].values.astype("datetime64[ns]").astype("int64")
    hns = int(HORIZON_MIN * 60 * 1e9)

    cells = {"genuine_unsafe": 0, "conflict_indeterminate": 0,
             "improving": 0, "delay_worse": 0, "neutral": 0}
    deltas = []
    done = 0
    n_legal = 0                             # model route ∈ candidate set (structural, expect = done)
    # --- asymmetry diagnostic accumulators (separate intrinsic route-length effect
    # from the fixed-others counterfactual bias; both-complete subsets only so the
    # horizon-clamp on non-finishers does NOT pollute the finish-Δ means) ---
    alone_dfin, with_dfin = [], []          # focal-finish Δ (model−sig), s
    comp = {"alone_m": 0, "alone_s": 0, "with_m": 0, "with_s": 0}   # focal completion counts
    uns_total = uns_alone_ok = 0            # completion-unsafe recheck
    # conflict-load: headway-wait Δ (model−sig with-others). Both scenarios share the SAME
    # fixed others & differ ONLY in the focal route → the Δ isolates the EXTRA conflict the
    # model's route introduces vs the signaller's (Hao: 'unsafe' is really conflict-w/-others).
    hw_deltas = []
    for _, row in dis.iterrows():
        t0 = int(pd.Timestamp(row["t"]).value)
        win = td[(td_ns >= t0) & (td_ns < t0 + hns)]
        if len(win) < 20:
            continue
        others = other_trains_at(win, str(row["focal_train"]))
        if not others:
            continue
        f = str(row["focal_train"]); cls = f[2] if len(f) >= 3 else "0"
        scen_m = Scenario(t0, others + [Train(f, list(route_tcs[row["model_route"]]), cls, 0, t0)], plat_of)
        scen_s = Scenario(t0, others + [Train(f, list(route_tcs[row["sig_route"]]), cls, 0, t0)], plat_of)
        last_m, last_s = str(route_tcs[row["model_route"]][-1]), str(route_tcs[row["sig_route"]][-1])
        rm = sim.simulate(scen_m, HORIZON_MIN)
        rs = sim.simulate(scen_s, HORIZON_MIN)
        cm, fm = _focal_outcome(rm, f, last_m, t0, HORIZON_MIN)
        cs, fs = _focal_outcome(rs, f, last_s, t0, HORIZON_MIN)
        deltas.append((rm["throughput"] - rs["throughput"], (fm - fs) / 1e9))
        hw_deltas.append(rm["headway_waits"] - rs["headway_waits"])   # conflict-load Δ
        # focal ALONE (no others) on each route → sim-INDEPENDENT feasibility + symmetric
        # intrinsic timing (the safety-first classifier judges off THESE, not with-others) ---
        rm1 = sim.simulate(Scenario(t0, [Train(f, list(route_tcs[row["model_route"]]), cls, 0, t0)], plat_of), HORIZON_MIN)
        rs1 = sim.simulate(Scenario(t0, [Train(f, list(route_tcs[row["sig_route"]]), cls, 0, t0)], plat_of), HORIZON_MIN)
        cm1, fm1 = _focal_outcome(rm1, f, last_m, t0, HORIZON_MIN)
        cs1, fs1 = _focal_outcome(rs1, f, last_s, t0, HORIZON_MIN)
        # legality: model route must be one of the legal candidates (action-space guarantee) ---
        cands = row["candidate_route_ids"]
        legal = str(row["model_route"]) in {str(x) for x in (cands if cands is not None else [])}
        n_legal += int(legal)
        cells[classify(legal, cm1, fm1, cs1, fs1, cm, args.delta_s)] += 1
        # asymmetry diagnostic accumulators ---
        comp["with_m"] += int(cm); comp["with_s"] += int(cs)
        comp["alone_m"] += int(cm1); comp["alone_s"] += int(cs1)
        if cm and cs:                          # both finish WITH others → clean Δ
            with_dfin.append((fm - fs) / 1e9)
        if cm1 and cs1:                         # both finish ALONE → intrinsic Δ
            alone_dfin.append((fm1 - fs1) / 1e9)
        if cs and not cm:                       # with-others completion-"unsafe"
            uns_total += 1
            if cm1:                             # … but model route finishes fine ALONE
                uns_alone_ok += 1
        done += 1

    gu, ci = cells["genuine_unsafe"], cells["conflict_indeterminate"]
    imp, dw, nt = cells["improving"], cells["delay_worse"], cells["neutral"]
    adj = imp + dw + nt                         # safe & adjudicable (delay tier)
    print(f"\n=== Tier 3 — SAFETY-FIRST decomposition (n={done:,}) ===")
    print("  [SAFETY tier — simulator-independent; an unsafe divergence is INTOLERABLE]")
    print(f"    route legality (model route ∈ candidates): {n_legal:,}/{done:,} "
          f"({100*n_legal/max(done,1):.1f}%)  ← action-space guarantee")
    print(f"    genuine_unsafe (illegal OR infeasible even alone): {gu:>5}  "
          f"({100*gu/max(done,1):.2f}%)  ← MUST be ~0")
    print(f"    conflict_indeterminate (blocked only w/ fixed others): {ci:>5}  "
          f"({100*ci/max(done,1):.1f}%)  ← eval cannot adjudicate (asymmetry)")
    if hw_deltas:
        hw = np.array(hw_deltas, dtype=float)
        more = int((hw > 0).sum()); less = int((hw < 0).sum())
        print(f"    conflict-load: mean headway-wait Δ (model−sig) = {hw.mean():+.2f}  "
              f"[model route causes MORE conflict: {100*more/len(hw):.1f}% | LESS: {100*less/len(hw):.1f}%]")
        print("       (≈0 / symmetric ⇒ model doesn't introduce more inter-train conflict than signaller)")
    print("  [DELAY tier — safe & both complete; FAIR symmetric alone-Δ, δ=±{:.0f}s]".format(args.delta_s))
    for k, v in (("improving", imp), ("delay_worse", dw), ("neutral", nt)):
        print(f"    {k:12s}: {v:>5}  ({100*v/max(done,1):.1f}% of all | "
              f"{100*v/max(adj,1):.1f}% of adjudicable)")
    print("--- head-line metrics (spec §4.3, v1.2) ---")
    print(f"  genuine-unsafe divergence rate = {100*gu/max(done,1):.2f}%   (SAFETY headline; intolerable if >0)")
    print(f"  conditional improvement (adjudicable) = improving/(improving+delay_worse) = "
          f"{100*imp/max(imp+dw,1):.1f}%")
    if deltas:
        tp = np.array([d[0] for d in deltas]); fd = np.array([d[1] for d in deltas])
        print(f"  mean throughput Δ (model−sig) = {tp.mean():+.3f}  | "
              f"mean focal-finish Δ (with-others, clamp-polluted) = {fd.mean():+.1f}s  "
              "[fair alone-Δ ↓ in diagnostic]")

    # ---- asymmetry diagnostic: is the +finish-Δ / unsafe-lean intrinsic or eval-bias? ----
    am = float(np.mean(alone_dfin)) if alone_dfin else float("nan")
    wm = float(np.mean(with_dfin)) if with_dfin else float("nan")
    print(f"\n=== asymmetry diagnostic (n={done:,}) ===")
    print("focal-finish Δ (model−sig), both-complete subsets only (no horizon-clamp):")
    print(f"  alone (no others, intrinsic) : mean {am:+.1f}s  (n={len(alone_dfin):,})  ← pure route length/dwell")
    print(f"  with others   (conflict)     : mean {wm:+.1f}s  (n={len(with_dfin):,})")
    print(f"  conflict-induced (with−alone): {wm-am:+.1f}s")
    print("focal completion rate (reaches last TC within horizon):")
    print(f"  alone:       model {100*comp['alone_m']/max(done,1):.1f}%  | signaller {100*comp['alone_s']/max(done,1):.1f}%")
    print(f"  with others: model {100*comp['with_m']/max(done,1):.1f}%  | signaller {100*comp['with_s']/max(done,1):.1f}%")
    print(f"completion-unsafe recheck (sig finishes, model doesn't, WITH others): {uns_total:,}")
    print(f"  of these, model route finishes when run ALONE: {uns_alone_ok:,} "
          f"({100*uns_alone_ok/max(uns_total,1):.1f}%)")
    print("  → high % ⇒ the 'unsafe' lean is the fixed-others counterfactual asymmetry, not the model's route")

    print("\n(v1 — verify scenario/classify on this run; δ in reward-units is v1.1. "
          "Single-seed42; multi-seed Tier-3 after 43/44.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
