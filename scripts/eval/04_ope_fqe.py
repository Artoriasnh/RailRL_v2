"""Stage 8 OPE — Fitted Q Evaluation (FQE) of the CQL policy vs the signaller.

WHY (the point of this script): the fixed-others L3 simulator CANNOT show delay-reduction.
It measures free-flow TC-traversal time (NOT the real lateness-change the reward optimises)
and the other trains don't react to the model's choice, so system-level improvement is
invisible (diagnostic 2026-05-25: model routes look +14s slower in free-flow, but that's the
wrong instrument). FQE estimates the model policy's expected discounted return DIRECTLY on the
real logged trajectories (real rewards, incl. the real delay component) — the standard
offline-RL way to ask "is the policy better?", needing no reactive simulator.

METHOD (per reward key r ∈ {total, delay, wait, throughput, headway}):
  target policy   π(s) = argmax_a Q_CQL(s,a)             (FROZEN trained model)
  fit evaluator Q_e by Bellman regression on logged transitions (s, a_β, r, s', done):
        Q_e(s, a_β)  ←  r + γ (1-done) Q_e_target(s', π(s'))
  then            V^π(s) = Q_e(s, π(s))                  (return of FOLLOWING π from s)
  behaviour base  V^β(s) = empirical MC discounted return of the LOGGED trajectory
                           (the real outcome under the signaller) — unbiased, model-free.
  report          ΔV = mean_s[ V^π(s) − V^β(s) ]  on the held-out TEST split (+ bootstrap CI).
  ΔV > 0  ⇒  FQE estimates following the model yields higher return.
            For r=delay specifically: ΔV>0 ⇒ the model policy is estimated to REDUCE delay.

⚠️ HONEST LIMITS — offline RL has NO counterfactual ground truth; this is an ESTIMATE:
  * FQE has no conservatism → it can be OPTIMISTIC on the ~4.3% of states where π diverges
    from the signaller (out-of-distribution actions). Coverage is good elsewhere (95.7%
    agreement). Read ΔV WITH that context; treat it as ONE of several imperfect estimators
    (alongside the L3 safety check), never a stand-alone "we beat the human" proof.
  * V^β is the realised logged return (unbiased for the behaviour/signaller policy).
    V^π is a model estimate. Fit on TRAIN, evaluate on TEST. Single seed42 (multi-seed later).
  * Bootstrap CI clusters by episode (states within an episode are correlated).

Run on Windows (GPU):
    python scripts/eval/04_ope_fqe.py --reward-key total  --epochs 3
    python scripts/eval/04_ope_fqe.py --reward-key delay  --epochs 3
"""
from __future__ import annotations
import argparse
import json
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

REWARD_COL = {"total": "r_total", "delay": "r_delay", "wait": "r_wait",
              "throughput": "r_throughput", "headway": "r_headway"}
GAMMA = C.DISCOUNT_GAMMA


def load_reward_array(key: str) -> np.ndarray:
    """sample_id → immediate reward r_<key>, as a dense array indexed by sample_id
    (covers ALL splits; train rows used for the FQE fit, test rows for eval)."""
    import pyarrow.parquet as pq
    col = REWARD_COL[key]
    t = pq.read_table(str(C.SNAPSHOTS_V2_PARQUET), columns=["sample_id", col]).to_pandas()
    sid = t["sample_id"].to_numpy().astype(np.int64)
    r = t[col].to_numpy().astype(np.float64)
    arr = np.full(int(sid.max()) + 1, np.nan, dtype=np.float64)
    arr[sid] = r
    n_nan = int(np.isnan(r).sum())
    print(f"  reward[{key}] ({col}): {len(sid):,} rows | NaN immediate r = {n_nan:,} "
          f"({100*n_nan/max(len(sid),1):.2f}%)  [NaN→0 in fit]")
    return np.nan_to_num(arr, nan=0.0)


def behaviour_mc_returns(key: str):
    """V^β: empirical MC discounted return of the logged trajectory, per TEST state.
    Returns dict sample_id → (episode_idx, G). G_t = r_t + γ G_{t+1} within episode."""
    import pyarrow.parquet as pq
    col = REWARD_COL[key]
    t = pq.read_table(str(C.SNAPSHOTS_V2_PARQUET),
                      columns=["sample_id", "episode_idx", "position_in_episode",
                               "split", col]).to_pandas()
    t = t[t["split"] == "test"].copy()
    t[col] = t[col].fillna(0.0)
    t = t.sort_values(["episode_idx", "position_in_episode"])
    out = {}
    for eidx, sub in t.groupby("episode_idx", sort=False):
        sids = sub["sample_id"].to_numpy().astype(np.int64)
        rs = sub[col].to_numpy().astype(np.float64)
        G = 0.0
        g_arr = np.empty(len(rs), dtype=np.float64)
        for k in range(len(rs) - 1, -1, -1):           # backward discounted sum
            G = rs[k] + GAMMA * G
            g_arr[k] = G
        for s, g in zip(sids, g_arr):
            out[int(s)] = (int(eidx), float(g))
    print(f"  behaviour MC returns (test): {len(out):,} states over "
          f"{len(set(v[0] for v in out.values())):,} episodes")
    return out


@torch.no_grad()
def pi_action(model_cql, batch):
    """π(s) = argmax_a Q_CQL(s,a) (masked) → (B,) long."""
    return model_cql(batch)["Q"].argmax(1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ckpt", default=None, help="CQL model = the policy π to evaluate")
    ap.add_argument("--reward-key", default="total", choices=list(REWARD_COL))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max-batches", type=int, default=0, help=">0 = smoke cap on FIT batches/epoch")
    ap.add_argument("--max-eval-batches", type=int, default=0,
                    help=">0 = cap the test-eval pass (smoke); 0 = FULL test (the real number)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--tau", type=float, default=C.CQL_TARGET_TAU)
    ap.add_argument("--warm-start", action="store_true",
                    help="init Q_e from CQL weights (faster) vs fresh random (cleaner; default)")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    try:
        import torch.multiprocessing as _mp
        _mp.set_sharing_strategy("file_system")
    except Exception:
        pass
    device = "cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu"
    ckpt = Path(args.ckpt) if args.ckpt else (C.TRAIN_DIR / f"cql_seed{args.seed}" / "best.pt")
    print(f"FQE | policy ckpt={ckpt} | reward-key={args.reward_key} | γ={GAMMA} | device={device}")

    stats = NormStats.load(C.NORMALIZATION_STATS_JSON)

    # ---- frozen CQL policy π (provides argmax actions; never trained here) ----
    model_cql = RailRLModel.build(stats).to(device)
    ck = torch.load(str(ckpt), map_location=device, weights_only=False)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    model_cql.load_state_dict(sd)
    model_cql.eval()
    for p in model_cql.parameters():
        p.requires_grad_(False)

    # ---- FQE evaluator Q_e (+ Polyak target) ----
    q_e = RailRLModel.build(stats).to(device)
    if args.warm_start:
        q_e.load_state_dict(sd)
        print("  Q_e warm-started from CQL weights")
    q_e_tgt = RailRLModel.build(stats).to(device)
    q_e_tgt.load_state_dict(q_e.state_dict())
    for p in q_e_tgt.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(q_e.parameters(), lr=args.lr)

    reward_arr = load_reward_array(args.reward_key)

    def reward_of(batch):
        sids = batch.sample_id.view(-1).cpu().numpy().astype(np.int64)
        return torch.tensor(reward_arr[sids], dtype=torch.float32, device=device)

    # ---- fit Q_e by Bellman regression on the TRAIN split ----
    print("fitting Q_e (FQE Bellman regression on train split) ...")
    for ep in range(args.epochs):
        ds = StreamingTransitionDataset(C.SNAPSHOTS_V2_PARQUET, C.NORMALIZATION_STATS_JSON,
                                        split="train", batch_size=C.BATCH_SIZE, shuffle=True, seed=ep)
        dl = DataLoader(ds, batch_size=None, num_workers=args.num_workers)
        q_e.train()
        run_loss = 0.0; nb = 0; t_ep = time.time()
        for bs, bsp, done in dl:
            bs = bs.to(device); bsp = bsp.to(device); done = done.to(device).view(-1)
            r = reward_of(bs)
            with torch.no_grad():
                a_sp = pi_action(model_cql, bsp)                       # π(s')
                q_sp = q_e_tgt(bsp)["Q"].gather(1, a_sp.view(-1, 1)).squeeze(1)
                y = r + GAMMA * (1.0 - done) * q_sp                    # FQE target
            a_b = bs.chosen_action_idx.view(-1, 1).long()              # behaviour action
            q_pred = q_e(bs)["Q"].gather(1, a_b).squeeze(1)
            loss = torch.nn.functional.smooth_l1_loss(q_pred, y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(q_e.parameters(), 10.0)
            opt.step()
            with torch.no_grad():                                     # Polyak target
                for tp, p in zip(q_e_tgt.parameters(), q_e.parameters()):
                    tp.mul_(1 - args.tau).add_(args.tau * p)
            run_loss += float(loss.detach()); nb += 1
            if nb % 500 == 0:
                print(f"    [fit] ep{ep} batch {nb:,} | loss {run_loss/nb:.4f} "
                      f"| {nb/(time.time()-t_ep):.1f} batch/s", flush=True)
            if args.max_batches and nb >= args.max_batches:
                break
        print(f"  epoch {ep}: mean Bellman loss = {run_loss/max(nb,1):.4f}  "
              f"({nb:,} batches, {time.time()-t_ep:.0f}s)", flush=True)

    # ---- evaluate V^π on TEST; compare to behaviour MC V^β ----
    print("evaluating V^π on test split ...")
    mc = behaviour_mc_returns(args.reward_key)
    q_e.eval()
    ds_te = StreamingTransitionDataset(C.SNAPSHOTS_V2_PARQUET, C.NORMALIZATION_STATS_JSON,
                                       split="test", batch_size=C.BATCH_SIZE, shuffle=False)
    dl_te = DataLoader(ds_te, batch_size=None, num_workers=args.num_workers)
    print(f"  (test eval is FULL-split & NOT capped by --max-batches; ~1.3k batches. "
          f"Use --max-eval-batches for a smoke.)", flush=True)
    v_pi, v_beta, ep_id = [], [], []
    t_ev = time.time(); neb = 0
    with torch.no_grad():
        for bs, _bsp, _done in dl_te:
            bs = bs.to(device)
            a_pi = pi_action(model_cql, bs)
            vpi = q_e(bs)["Q"].gather(1, a_pi.view(-1, 1)).squeeze(1).cpu().numpy()
            sids = bs.sample_id.view(-1).cpu().numpy().astype(np.int64)
            for sid, vp in zip(sids, vpi):
                hit = mc.get(int(sid))
                if hit is None:
                    continue
                ep_id.append(hit[0]); v_beta.append(hit[1]); v_pi.append(float(vp))
            neb += 1
            if neb % 200 == 0:
                print(f"    [eval] {neb:,} batches | {len(v_pi):,} matched states | "
                      f"{neb/(time.time()-t_ev):.1f} batch/s", flush=True)
            if args.max_eval_batches and neb >= args.max_eval_batches:
                print(f"    [eval] stopped at --max-eval-batches={args.max_eval_batches} (smoke)")
                break
    v_pi = np.asarray(v_pi); v_beta = np.asarray(v_beta); ep_id = np.asarray(ep_id)
    dv = v_pi - v_beta
    n = len(dv)

    # cluster bootstrap by episode (states within an episode are correlated)
    uniq_ep = np.unique(ep_id)
    ep_to_idx = {e: np.where(ep_id == e)[0] for e in uniq_ep}
    rng = np.random.default_rng(0)
    boot = np.empty(args.n_boot)
    for b in range(args.n_boot):
        pick = rng.choice(uniq_ep, size=len(uniq_ep), replace=True)
        idx = np.concatenate([ep_to_idx[e] for e in pick])
        boot[b] = dv[idx].mean()
    lo, hi = np.percentile(boot, [2.5, 97.5])

    print(f"\n=== FQE OPE result — reward-key={args.reward_key} (n={n:,} test states, "
          f"{len(uniq_ep):,} episodes) ===")
    print(f"  mean V^π (model policy)   = {v_pi.mean():+.4f}")
    print(f"  mean V^β (signaller, MC)  = {v_beta.mean():+.4f}")
    print(f"  ΔV = V^π − V^β            = {dv.mean():+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]")
    sig = "POSITIVE (model est. better" + (" — less delay)" if args.reward_key == "delay" else " return)") \
        if lo > 0 else ("NEGATIVE (model est. worse)" if hi < 0 else "NOT significant (CI spans 0)")
    print(f"  → {sig}")
    print("  (ESTIMATE — no counterfactual ground truth; FQE may be optimistic on the ~4.3% "
          "divergent OOD states. Pair with the L3 safety check; single seed42.)")

    C.EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out = C.EVAL_DIR / f"ope_fqe_seed{args.seed}_{args.reward_key}.json"
    out.write_text(json.dumps({
        "reward_key": args.reward_key, "gamma": GAMMA, "ckpt": str(ckpt),
        "n_states": int(n), "n_episodes": int(len(uniq_ep)),
        "V_pi": float(v_pi.mean()), "V_beta": float(v_beta.mean()),
        "delta_V": float(dv.mean()), "ci95": [float(lo), float(hi)],
        "epochs": args.epochs, "warm_start": bool(args.warm_start),
    }, indent=2))
    print(f"\n→ wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
