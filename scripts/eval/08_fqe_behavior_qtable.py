"""L5 step 1 — behavior-policy per-component FQE → per-decision Q-table (spec 05 §11).

Produces the FEATURES the L5 IRL (09_l5_irl.py) consumes: for every TEST decision and every
LEGAL action a, the 4 per-component action-values Q_k(s,a) under the SIGNALLER (behavior)
policy, k ∈ {delay, throughput, headway, wait}.

Behavior-policy FQE (a variant of 05_ope_fqe_decompose): fit Q_e_k by Bellman regression on
logged transitions, bootstrapping with the LOGGED next action (the behavior policy), NOT the
CQL argmax — so Q_e_k ≈ Q^β_k (value of a then following the signaller). No CQL model needed.
    Q_e_k(s, a_β) ← r_k + γ(1−done) Q_e_k_tgt(s', a'_logged)
Then evaluate the 4 nets on the test split and dump one row per (decision, legal action):
    sample_id, action_idx, is_chosen, q_delay, q_throughput, q_headway, q_wait,
    episode_idx (for cluster bootstrap), prefix, headcode_class (for per-subset IRL).

⚠️ The Q_e_k must generalize to the COUNTERFACTUAL legal actions the signaller didn't take
(same offline-RL OOD caveat as our OPE) — a validity note for L5.

Run on Windows GPU (~2-3h like 05):
    python scripts/eval/08_fqe_behavior_qtable.py --epochs 1 --max-batches 4000 --num-workers 2
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch
from torch.utils.data import DataLoader

from railrl import config as C
from railrl.encoders.input_pipeline import NormStats
from railrl.algorithms.transitions import StreamingTransitionDataset
from railrl.model import RailRLModel

COMPONENTS = ["delay", "throughput", "headway", "wait"]
REWARD_COL = {"delay": "r_delay", "throughput": "r_throughput",
              "headway": "r_headway", "wait": "r_wait"}
GAMMA = C.DISCOUNT_GAMMA


def load_reward_arrays():
    import pyarrow.parquet as pq
    cols = ["sample_id"] + [REWARD_COL[k] for k in COMPONENTS]
    t = pq.read_table(str(C.SNAPSHOTS_V2_PARQUET), columns=cols).to_pandas()
    sid = t["sample_id"].to_numpy().astype(np.int64)
    n = int(sid.max()) + 1
    out = {}
    for k in COMPONENTS:
        a = np.full(n, np.nan); a[sid] = t[REWARD_COL[k]].to_numpy().astype(np.float64)
        out[k] = np.nan_to_num(a, nan=0.0)
    print(f"  reward arrays: {len(sid):,} rows")
    return out


def load_meta():
    """sample_id -> (episode_idx, prefix, headcode_class) for test rows (subset/bootstrap meta)."""
    import pyarrow.parquet as pq
    t = pq.read_table(str(C.SNAPSHOTS_V2_PARQUET),
                      columns=["sample_id", "split", "episode_idx",
                               "chosen_route_id", "focal_train"]).to_pandas()
    t = t[t["split"] == "test"]
    meta = {}
    for _, r in t.iterrows():
        rid = str(r["chosen_route_id"]) if r["chosen_route_id"] is not None else ""
        ftr = str(r["focal_train"]) if r["focal_train"] is not None else ""
        meta[int(r["sample_id"])] = (int(r["episode_idx"]),
                                     rid[:2] if rid else "NA",
                                     ftr[2] if len(ftr) >= 3 else "NA")
    print(f"  test meta: {len(meta):,} decisions")
    return meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-batches", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--tau", type=float, default=C.CQL_TARGET_TAU)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    try:
        import torch.multiprocessing as _mp
        _mp.set_sharing_strategy("file_system")
    except Exception:
        pass
    device = "cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu"
    print(f"behavior-FQE Q-table | components={COMPONENTS} | γ={GAMMA} | device={device}")
    stats = NormStats.load(C.NORMALIZATION_STATS_JSON)

    qes, tgts, opts = {}, {}, {}
    for k in COMPONENTS:
        qe = RailRLModel.build(stats).to(device)
        tg = RailRLModel.build(stats).to(device); tg.load_state_dict(qe.state_dict())
        for p in tg.parameters():
            p.requires_grad_(False)
        qes[k] = qe; tgts[k] = tg; opts[k] = torch.optim.AdamW(qe.parameters(), lr=args.lr)
    reward = load_reward_arrays()
    huber = torch.nn.functional.smooth_l1_loss

    print("fitting behavior-FQE per component (bootstrap = LOGGED next action) ...")
    for ep in range(args.epochs):
        ds = StreamingTransitionDataset(C.SNAPSHOTS_V2_PARQUET, C.NORMALIZATION_STATS_JSON,
                                        split="train", batch_size=C.BATCH_SIZE, shuffle=True, seed=ep)
        dl = DataLoader(ds, batch_size=None, num_workers=args.num_workers)
        run = {k: 0.0 for k in COMPONENTS}; nb = 0; t0 = time.time()
        for bs, bsp, done in dl:
            bs = bs.to(device); bsp = bsp.to(device); done = done.to(device).view(-1)
            sids = bs.sample_id.view(-1).cpu().numpy().astype(np.int64)
            a_b = bs.chosen_action_idx.view(-1, 1).long()
            a_sp = bsp.chosen_action_idx.view(-1, 1).long()          # behavior next action
            for k in COMPONENTS:
                r = torch.tensor(reward[k][sids], dtype=torch.float32, device=device)
                with torch.no_grad():
                    q_sp = tgts[k](bsp)["Q"].gather(1, a_sp).squeeze(1)
                    y = r + GAMMA * (1.0 - done) * q_sp
                q_pred = qes[k](bs)["Q"].gather(1, a_b).squeeze(1)
                loss = huber(q_pred, y)
                opts[k].zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(qes[k].parameters(), 10.0)
                opts[k].step()
                with torch.no_grad():
                    for tp, p in zip(tgts[k].parameters(), qes[k].parameters()):
                        tp.mul_(1 - args.tau).add_(args.tau * p)
                run[k] += float(loss.detach())
            nb += 1
            if nb % 250 == 0:
                print(f"    [fit] ep{ep} b{nb:,} | "
                      + " ".join(f"{k[:4]} {run[k]/nb:.3f}" for k in COMPONENTS)
                      + f" | {nb/(time.time()-t0):.1f} b/s", flush=True)
            if args.max_batches and nb >= args.max_batches:
                break
        print(f"  epoch {ep} done ({nb:,} b, {time.time()-t0:.0f}s) | "
              + " ".join(f"{k}:{run[k]/max(nb,1):.4f}" for k in COMPONENTS), flush=True)

    # ---- evaluate Q_k(s,a) for every LEGAL action on the test split → dump Q-table ----
    print("dumping per-decision Q-table (test) ...")
    meta = load_meta()
    for k in COMPONENTS:
        qes[k].eval()
    ds_te = StreamingTransitionDataset(C.SNAPSHOTS_V2_PARQUET, C.NORMALIZATION_STATS_JSON,
                                       split="test", batch_size=C.BATCH_SIZE, shuffle=False)
    dl_te = DataLoader(ds_te, batch_size=None, num_workers=args.num_workers)
    recs = {c: [] for c in ["sample_id", "action_idx", "is_chosen",
                            "q_delay", "q_throughput", "q_headway", "q_wait",
                            "episode_idx", "prefix", "headcode_class"]}
    neb = 0; t0 = time.time()
    with torch.no_grad():
        for bs, _bsp, _done in dl_te:
            bs = bs.to(device)
            B = int(bs.num_graphs) if getattr(bs, "num_graphs", None) else bs.n_candidates.numel()
            qk = {k: qes[k](bs)["Q"].view(B, -1).cpu().numpy() for k in COMPONENTS}   # (B, K+1)
            am = bs.act_mask.view(B, -1).cpu().numpy()                                 # (B, K)
            chosen = bs.chosen_action_idx.view(B).cpu().numpy().astype(int)
            sids = bs.sample_id.view(B).cpu().numpy().astype(np.int64)
            for i in range(B):
                m = meta.get(int(sids[i]))
                if m is None:
                    continue
                eidx, pref, hc = m
                # legal actions: 0 (wait, always) + candidates k where act_mask>0.5
                legal = [0] + [j + 1 for j in range(am.shape[1]) if am[i, j] > 0.5]
                for a in legal:
                    recs["sample_id"].append(int(sids[i])); recs["action_idx"].append(int(a))
                    recs["is_chosen"].append(bool(a == chosen[i]))
                    recs["q_delay"].append(float(qk["delay"][i, a]))
                    recs["q_throughput"].append(float(qk["throughput"][i, a]))
                    recs["q_headway"].append(float(qk["headway"][i, a]))
                    recs["q_wait"].append(float(qk["wait"][i, a]))
                    recs["episode_idx"].append(eidx); recs["prefix"].append(pref)
                    recs["headcode_class"].append(hc)
            neb += 1
            if neb % 200 == 0:
                print(f"    [dump] {neb:,} batches | {len(recs['sample_id']):,} rows | "
                      f"{neb/(time.time()-t0):.1f} b/s", flush=True)

    import pyarrow as pa
    import pyarrow.parquet as pq
    C.EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out = C.EVAL_DIR / "l5_qtable.parquet"
    pq.write_table(pa.table(recs), str(out))
    n_dec = len(set(recs["sample_id"]))
    print(f"\n→ wrote {out}  ({len(recs['sample_id']):,} rows, {n_dec:,} decisions)")
    print("Next: python scripts/eval/09_l5_irl.py  (cheap CPU IRL → Table V)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
