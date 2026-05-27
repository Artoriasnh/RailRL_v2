"""Stage 8 — §12 Selective Override deployment statistics (spec 05 §12).

Per test decision, decide agreement / consider-override / silent (railrl.deploy.selective_override):
  * agreement       : model argmax == signaller logged action.
  * disagreement → three gates:
      gate_l3 : L3 counterfactual reward improvement of model's route over signaller's > 0.5 units
      gate_l4 : L4 rule-compliance of the model's route == 'compliant'
      gate_l2 : L2 SHAP faithfulness > 0.7
    all pass → consider-override; else silent.

L3_DELTA IN REWARD UNITS (the spec refinement 03 deferred): the P2.6 simulator returns
delay_delta_s (model−sig) and throughput_delta (model−sig); we map them through the reward
weights (reward_model: w_delay=1.0, w_throughput=0.5) to a single reward-unit improvement
    l3_reward_delta = -(delay_delta_s/60)*w_delay + throughput_delta*w_throughput
positive ⇒ model's route is better. gate_l3 = l3_reward_delta > δ_L3 (0.5).

COST / SCOPE: the L3 simulator and the 64-coalition L2 are expensive, so (like 03) we run the
gates on a sampled set of disagreements (--max-decisions), and we SHORT-CIRCUIT: L2 faithfulness
(the most expensive) is computed ONLY for disagreements that already pass gate_l3 ∧ gate_l4.
The agreement_rate is reported EXACTLY over all set decisions; override/silent split is reported
over the sampled disagreements (honest scoping, same as Tier-3).

Reuses: model forward (eval/01,03), L3Simulator (xai.l3_system), l4_check (xai.l4_rules),
q_gap_decomposition + l2_faithfulness (xai.l2_qdecomp + deploy.selective_override), encode path
(encoders.input_pipeline). Run on Windows GPU:
    python scripts/eval/13_selective_override.py --seed 42                 # default 1500 sample
    python scripts/eval/13_selective_override.py --seed 42 --max-decisions 50   # smoke
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from railrl import config as C
from railrl.encoders.input_pipeline import NormStats, encode_snapshot, to_heterodata
from railrl.algorithms.transitions import StreamingTransitionDataset
from railrl.data_io import load_route_to_tc
from railrl.model import RailRLModel
from railrl.xai.l3_system import L3Simulator, Scenario, Train, l3_delta
from railrl.xai.l2_qdecomp import q_gap_decomposition
from railrl.xai import l4_rules as L4
from railrl.deploy.selective_override import (
    selective_override, l2_faithfulness, evaluate_selective_override_on_test,
    DELTA_L3, FAITHFULNESS_THRESHOLD)

HORIZON_MIN = 30.0
W_DELAY, W_THROUGHPUT = 1.0, 0.5                     # reward_model weights (delay / throughput)
DELTA_GRID = (0.5, 0.25, 0.1)                        # δ_L3 sensitivity sweep (0.5 = spec primary)
STRATUM_NAMES = {0: "late_train", 1: "advance", 2: "call_on", 3: "platform_dev",
                 4: "priority_compete", 5: "unusual_id", 6: "trivial", -1: "unlabeled"}
# test window (for the TD slice used by the simulator); mirrors 03
VAL = (np.datetime64("2024-03-01"), np.datetime64("2099-01-01"))


def forward_pass(model, loader, device):
    sid, chosen, marg = [], [], []
    with torch.no_grad():
        for bs, _bsp, _done in loader:
            bs = bs.to(device)
            out = model(bs)
            B = bs.num_graphs
            sid.append(bs.sample_id.view(B).cpu().numpy())
            chosen.append(bs.chosen_action_idx.view(B).cpu().numpy())
            marg.append(out["Q"].argmax(1).cpu().numpy())
    return np.concatenate(sid), np.concatenate(chosen), np.concatenate(marg)


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


def l3_reward_delta(l3d: dict) -> float:
    """Convert the simulator's {delay_delta_s, throughput_delta} (model−sig) to reward units."""
    return (-(l3d["delay_delta_s"] / 60.0) * W_DELAY) + (l3d["throughput_delta"] * W_THROUGHPUT)


def fetch_rows(sample_ids: set):
    """sample_id → full snapshot row dict (for L2 encoding). Scans row-groups once."""
    import pyarrow.parquet as pq
    want = set(int(x) for x in sample_ids)
    rows = {}
    pf = pq.ParquetFile(str(C.SNAPSHOTS_V2_PARQUET))
    for rg in range(pf.num_row_groups):
        if not want:
            break
        tb = pf.read_row_group(rg)
        d = tb.to_pydict()
        for i in range(len(d["sample_id"])):
            s = int(d["sample_id"][i])
            if s in want:
                rows[s] = {k: d[k][i] for k in d}
                want.discard(s)
                if not want:
                    break
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="§12 selective override (spec 05 §12).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--max-decisions", type=int, default=1500, help="L3-sim sample cap on disagreements")
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

    # ---- 1. model forward over test ----
    stats = NormStats.load(C.NORMALIZATION_STATS_JSON)
    model = RailRLModel.build(stats).to(device)
    ck = torch.load(str(ckpt), map_location=device, weights_only=False)
    model.load_state_dict(ck["model"] if isinstance(ck, dict) and "model" in ck else ck)
    model.eval()
    ds = StreamingTransitionDataset(C.SNAPSHOTS_V2_PARQUET, C.NORMALIZATION_STATS_JSON,
                                    split="test", batch_size=C.BATCH_SIZE, shuffle=False, stratified=False)
    print(f"forward pass (test) | ckpt={ckpt.name} ...")
    sid, chosen, marg = forward_pass(model, DataLoader(ds, batch_size=None, num_workers=args.num_workers), device)

    import pyarrow.parquet as pq
    ident = pq.read_table(str(C.SNAPSHOTS_V2_PARQUET),
                          columns=["sample_id", "focal_train", "focal_signal", "t",
                                   "chosen_route_id", "chosen_action_idx", "candidate_route_ids"]).to_pandas()
    strata = {}
    sp = C.SNAPSHOTS_DIR / "stratum_labels.parquet"
    if sp.exists():
        t = pq.read_table(str(sp), columns=["sample_id", "stratum"])
        strata = dict(zip(t.column("sample_id").to_pylist(), t.column("stratum").to_pylist()))

    pred = pd.DataFrame({"sample_id": sid, "chosen_idx": chosen, "model_idx": marg})
    df = pred.merge(ident, on="sample_id", how="left")

    # agreement (exact, over ALL decisions incl wait): model argmax == signaller action idx
    df["agree"] = df["model_idx"].astype(int) == df["chosen_action_idx"].astype(int)
    n_all = len(df)
    agreement_rate_all = float(df["agree"].mean())

    setd = df[df["chosen_action_idx"] > 0].copy()

    def model_route(row):
        mi = int(row["model_idx"])
        cv = row["candidate_route_ids"]
        cands = list(cv) if isinstance(cv, (list, np.ndarray)) else []
        return None if (mi <= 0 or mi - 1 >= len(cands)) else str(cands[mi - 1])
    setd["model_route"] = setd.apply(model_route, axis=1)
    setd["sig_route"] = setd["chosen_route_id"].astype(str)
    setd["disagree"] = setd["model_route"].astype(str) != setd["sig_route"]
    n_set = len(setd)
    agreement_rate_set = float((~setd["disagree"]).mean())

    # ---- 2. sample set-disagreements for the gated evaluation ----
    rt = load_route_to_tc()
    route_tcs = {str(r["route"]): [str(x) for x in r["track_list"]]
                 for _, r in rt.iterrows() if isinstance(r["track_list"], list) and r["track_list"]}
    dis = setd[setd["disagree"] & setd["model_route"].notna()].copy()
    dis = dis[dis["model_route"].isin(route_tcs) & dis["sig_route"].isin(route_tcs)]
    if len(dis) > args.max_decisions:
        dis = dis.sample(args.max_decisions, random_state=0)
    print(f"set decisions {n_set:,} | agreement(set) {agreement_rate_set:.3f} | "
          f"gated disagreement sample {len(dis):,}")

    plat_of = build_platform_map()
    sim = L3Simulator.from_json(args.params)
    td = pq.read_table(str(C.TD_PARQUET), columns=["time", "type", "id", "state", "trainid_filled"]).to_pandas()
    td = td[td["type"] == "Track"]
    td["time"] = pd.to_datetime(td["time"], errors="coerce")
    td = td[(td["time"] >= VAL[0]) & (td["time"] < VAL[1])]
    td_ns = td["time"].values.astype("datetime64[ns]").astype("int64")
    hns = int(HORIZON_MIN * 60 * 1e9)

    # ---- 3. gate l3 (sim) + l4 (rule) per sampled disagreement; defer l2 to survivors ----
    records, l2_pending = [], []
    n_simmed = 0
    for _, row in dis.iterrows():
        f = str(row["focal_train"]); cls = f[2] if len(f) >= 3 else "0"
        t0 = int(pd.Timestamp(row["t"]).value)
        win = td[(td_ns >= t0) & (td_ns < t0 + hns)]
        others = other_trains_at(win, f) if len(win) >= 20 else []
        scen_m = Scenario(t0, others + [Train(f, list(route_tcs[row["model_route"]]), cls, 0, t0)], plat_of)
        scen_s = Scenario(t0, others + [Train(f, list(route_tcs[row["sig_route"]]), cls, 0, t0)], plat_of)
        l3d = l3_delta(sim, scen_m, scen_s, HORIZON_MIN)
        rdelta = l3_reward_delta(l3d)
        n_simmed += 1
        _cv = row["candidate_route_ids"]
        sample = dict(focal_signal=row.get("focal_signal"), focal_train=f,
                      candidate_route_ids=[str(x) for x in (list(_cv) if isinstance(_cv, (list, np.ndarray)) else [])])
        l4 = L4.l4_check(sample, audited_route_id=row["model_route"])["hard_status"]
        rec = dict(sample_id=int(row["sample_id"]),
                   signaller_action=int(row["chosen_action_idx"]), model_action=int(row["model_idx"]),
                   l3_delta=rdelta, l4_status=l4, l2_faithfulness=None,
                   focal_train=f, focal_signal=row.get("focal_signal"),
                   signaller_route=row["sig_route"], model_route=row["model_route"],
                   stratum=STRATUM_NAMES.get(int(strata.get(int(row["sample_id"]), 6)), "unlabeled"))
        records.append(rec)
        if (rdelta > min(DELTA_GRID)) and (l4 != "non-compliant"):   # loosest-δ refined survivors → need L2
            l2_pending.append(rec)

    # ---- 4. L2 faithfulness ONLY for l3∧l4 survivors (expensive: encode + decompose) ----
    print(f"L3∧L4 survivors needing L2 faithfulness: {len(l2_pending)}")
    if l2_pending:
        rows = fetch_rows({r["sample_id"] for r in l2_pending})
        for rec in l2_pending:
            row = rows.get(rec["sample_id"])
            if row is None:
                continue
            data = to_heterodata(encode_snapshot(row, stats))
            decomp = q_gap_decomposition(model, data, device)
            rec["l2_faithfulness"] = l2_faithfulness(model, data, decomp, device)["faithfulness"]

    # ---- 5. apply rule + aggregate (δ_L3 sweep × l4 口径) ----
    report = evaluate_selective_override_on_test(records, delta_l3_grid=DELTA_GRID)
    report["meta"] = {
        "seed": args.seed, "ckpt": str(ckpt), "n_all_decisions": n_all, "n_set_decisions": n_set,
        "agreement_rate_all_incl_wait": agreement_rate_all,
        "agreement_rate_set_only": agreement_rate_set,
        "n_set_disagreements_total": int((setd["disagree"]).sum()),
        "n_disagreements_gated(sampled)": n_simmed,
        "delta_l3_grid": list(DELTA_GRID), "faithfulness_threshold": FAITHFULNESS_THRESHOLD,
        "note": "agreement_rate is EXACT (all decisions); consider-override/silent are over the "
                "SAMPLED set-disagreements (L3 sim + L2 expensive). L2 faithfulness computed only "
                "for loosest-δ L3∧L4 survivors. l3_delta in reward units (delay w=1.0/throughput "
                "w=0.5). PRIMARY = δ_L3=0.5 + refined gate_l4; lower δ rows = sensitivity appendix.",
    }
    C.EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out = C.EVAL_DIR / f"selective_override_seed{args.seed}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\n→ wrote {out}")

    # ---- console summary ----
    print(f"\n=== §12 Selective Override (seed{args.seed}) ===")
    print(f"agreement (all decisions, exact): {100*agreement_rate_all:.1f}%  "
          f"| agreement (set-only): {100*agreement_rate_set:.1f}%")
    print(f"on {report['n']:,} sampled set-disagreements  (δ_L3 sensitivity × gate_l4 口径):")
    print(f"  {'δ_L3':>5} {'gate_l4':>9} | {'consider-override':>18} {'silent':>10} | silent gate-fails")
    for d in DELTA_GRID:
        for mode in ("refined", "literal"):
            m = report["sweep"][f"{d:g}"][mode]
            star = " *PRIMARY" if (d == DELTA_L3 and mode == "refined") else ""
            print(f"  {d:>5g} {mode:>9} | {m['counts']['consider-override']:>7,} "
                  f"({100*m['rates']['consider-override']:>4.1f}%)   {m['counts']['silent']:>5,} "
                  f"({100*m['rates']['silent']:>4.1f}%) | {m['silent_gate_failures']}{star}")
    # example override card from the most permissive setting that produced any
    for d in sorted(DELTA_GRID):
        ex = report["sweep"][f"{d:g}"]["refined"]["override_examples"]
        if ex:
            e = ex[0]
            print(f"  e.g. override card (δ={d:g}, refined): sid={e.get('sample_id')} "
                  f"{e.get('signaller_route')}→{e.get('model_route')} "
                  f"l3Δ={e.get('l3_delta'):+.2f} l4={e.get('l4_status')} l2faith={e.get('l2_faithfulness')}")
            break
    else:
        print("  (no consider-override examples at any δ_L3 under refined gate)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
