"""Stage 8 OPE — FQE reward-COMPONENT decomposition (one data pass, all keys).

WHY: the single-key FQE (04_ope_fqe.py) showed ΔV_total≈0 but ΔV_delay significantly
NEGATIVE (model worse on delay). Since r_total = r_delay + r_throughput + r_headway +
r_wait EXACTLY (stored components are already weighted; their means sum to r_total mean),
the model must be BETTER on the other components to keep total≈0 — i.e. it TRADES delay
for something else. This script fits a separate FQE evaluator Q_e for EVERY reward key in
ONE pass over the data (the data pipeline is the bottleneck; extra Q-heads on the same
batch are comparatively cheap), then reports:
  * per-key ΔV = V^π − V^β (+ episode-clustered 95% CI)  → the trade-off shape
  * the Σ-CHECK: ΔV_total  vs  Σ ΔV_{delay,throughput,headway,wait}. These should match
    if every Q_e is well-fit; a mismatch flags an under-fit / warm-start-biased Q_e
    (the residual mean|V^π_total − ΣV^π_comp| is printed as a fit-quality gauge).

Same FQE method as 04 (full fresh Q_e per key, Bellman regression on train, V^β = real MC
discounted return on test). Fresh init by default (component returns are small-magnitude,
so a ~0-init head is well-scaled — avoids the r_total-scale warm-start mismatch).

⚠️ Same HONEST LIMITS as 04: offline RL has no counterfactual ground truth; FQE is an
estimate, can be optimistic on the ~4.3% divergent OOD states; single seed42. Read the
decomposition alongside the L3 safety result, not as a stand-alone verdict.

Run on Windows (GPU):
    python scripts/eval/05_ope_fqe_decompose.py --epochs 1 --max-batches 4000 --num-workers 2
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

REWARD_COL = {"total": "r_total", "delay": "r_delay", "throughput": "r_throughput",
              "headway": "r_headway", "wait": "r_wait"}
COMPONENTS = ["delay", "throughput", "headway", "wait"]   # r_total = Σ these (exactly)
GAMMA = C.DISCOUNT_GAMMA


def load_reward_arrays(keys):
    """{key: dense array indexed by sample_id} for the immediate reward r_<key>."""
    import pyarrow.parquet as pq
    cols = ["sample_id"] + [REWARD_COL[k] for k in keys]
    t = pq.read_table(str(C.SNAPSHOTS_V2_PARQUET), columns=cols).to_pandas()
    sid = t["sample_id"].to_numpy().astype(np.int64)
    n = int(sid.max()) + 1
    out = {}
    for k in keys:
        r = t[REWARD_COL[k]].to_numpy().astype(np.float64)
        a = np.full(n, np.nan); a[sid] = r
        out[k] = np.nan_to_num(a, nan=0.0)
    print(f"  reward arrays for {keys}: {len(sid):,} rows")
    return out


def behaviour_mc_returns(keys):
    """V^β per TEST state for every key. Returns (sid_to_row, eidx[N], G {key:[N]})."""
    import pyarrow.parquet as pq
    cols = ["sample_id", "episode_idx", "position_in_episode", "split"] + [REWARD_COL[k] for k in keys]
    t = pq.read_table(str(C.SNAPSHOTS_V2_PARQUET), columns=cols).to_pandas()
    t = t[t["split"] == "test"].sort_values(["episode_idx", "position_in_episode"]).reset_index(drop=True)
    for k in keys:
        t[REWARD_COL[k]] = t[REWARD_COL[k]].fillna(0.0)
    N = len(t)
    sid = t["sample_id"].to_numpy().astype(np.int64)
    eidx = t["episode_idx"].to_numpy().astype(np.int64)
    G = {k: np.zeros(N) for k in keys}
    # backward discounted return within each episode (episodes are contiguous after sort)
    ep = t["episode_idx"].to_numpy()
    rcols = {k: t[REWARD_COL[k]].to_numpy().astype(np.float64) for k in keys}
    running = {k: 0.0 for k in keys}
    for i in range(N - 1, -1, -1):
        if i == N - 1 or ep[i] != ep[i + 1]:        # last row of its episode
            for k in keys:
                running[k] = 0.0
        for k in keys:
            running[k] = rcols[k][i] + GAMMA * running[k]
            G[k][i] = running[k]
    sid_to_row = {int(s): i for i, s in enumerate(sid)}
    print(f"  behaviour MC returns (test): {N:,} states over {len(np.unique(eidx)):,} episodes")
    return sid_to_row, eidx, G


@torch.no_grad()
def argmax_action(model, batch):
    return model(batch)["Q"].argmax(1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-batches", type=int, default=4000, help=">0 = cap FIT batches/epoch")
    ap.add_argument("--max-eval-batches", type=int, default=0, help=">0 = cap test eval (smoke)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--tau", type=float, default=C.CQL_TARGET_TAU)
    ap.add_argument("--warm-start", action="store_true", help="init each Q_e from CQL (default fresh)")
    ap.add_argument("--num-workers", type=int, default=2)
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
    keys = ["total"] + COMPONENTS
    print(f"FQE decompose | ckpt={ckpt} | keys={keys} | γ={GAMMA} | device={device}")

    stats = NormStats.load(C.NORMALIZATION_STATS_JSON)
    # frozen CQL policy π
    model_cql = RailRLModel.build(stats).to(device)
    ck = torch.load(str(ckpt), map_location=device, weights_only=False)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    model_cql.load_state_dict(sd); model_cql.eval()
    for p in model_cql.parameters():
        p.requires_grad_(False)

    # one fresh Q_e (+ target + optimizer) per key
    qes, tgts, opts = {}, {}, {}
    for k in keys:
        qe = RailRLModel.build(stats).to(device)
        if args.warm_start:
            qe.load_state_dict(sd)
        tg = RailRLModel.build(stats).to(device)
        tg.load_state_dict(qe.state_dict())
        for p in tg.parameters():
            p.requires_grad_(False)
        qes[k] = qe; tgts[k] = tg; opts[k] = torch.optim.AdamW(qe.parameters(), lr=args.lr)
    print(f"  built {len(keys)} Q_e (+targets) | warm_start={args.warm_start}")

    reward_arrs = load_reward_arrays(keys)
    huber = torch.nn.functional.smooth_l1_loss

    # ---- fit all Q_e in one pass over train ----
    print("fitting Q_e per key (FQE Bellman regression, one data pass) ...")
    for ep in range(args.epochs):
        ds = StreamingTransitionDataset(C.SNAPSHOTS_V2_PARQUET, C.NORMALIZATION_STATS_JSON,
                                        split="train", batch_size=C.BATCH_SIZE, shuffle=True, seed=ep)
        dl = DataLoader(ds, batch_size=None, num_workers=args.num_workers)
        run = {k: 0.0 for k in keys}; nb = 0; t_ep = time.time()
        for bs, bsp, done in dl:
            bs = bs.to(device); bsp = bsp.to(device); done = done.to(device).view(-1)
            sids = bs.sample_id.view(-1).cpu().numpy().astype(np.int64)
            a_b = bs.chosen_action_idx.view(-1, 1).long()
            with torch.no_grad():
                a_sp = argmax_action(model_cql, bsp)                   # π(s') shared across keys
            for k in keys:
                r = torch.tensor(reward_arrs[k][sids], dtype=torch.float32, device=device)
                with torch.no_grad():
                    q_sp = tgts[k](bsp)["Q"].gather(1, a_sp.view(-1, 1)).squeeze(1)
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
                ls = " ".join(f"{k[:4]} {run[k]/nb:.3f}" for k in keys)
                print(f"    [fit] ep{ep} b{nb:,} | {ls} | {nb/(time.time()-t_ep):.1f} b/s", flush=True)
            if args.max_batches and nb >= args.max_batches:
                break
        print(f"  epoch {ep} done ({nb:,} batches, {time.time()-t_ep:.0f}s) | "
              + " ".join(f"{k}:{run[k]/max(nb,1):.4f}" for k in keys), flush=True)

    # ---- evaluate V^π per key on TEST; V^β = MC ----
    print("evaluating V^π on test split ...")
    sid_to_row, eidx_all, G = behaviour_mc_returns(keys)
    for k in keys:
        qes[k].eval()
    ds_te = StreamingTransitionDataset(C.SNAPSHOTS_V2_PARQUET, C.NORMALIZATION_STATS_JSON,
                                       split="test", batch_size=C.BATCH_SIZE, shuffle=False)
    dl_te = DataLoader(ds_te, batch_size=None, num_workers=args.num_workers)
    print("  (full-test eval, NOT capped by --max-batches; ~1.3k batches)", flush=True)
    rows, vpi = [], {k: [] for k in keys}
    t_ev = time.time(); neb = 0
    with torch.no_grad():
        for bs, _bsp, _done in dl_te:
            bs = bs.to(device)
            a_pi = argmax_action(model_cql, bs)
            sids = bs.sample_id.view(-1).cpu().numpy().astype(np.int64)
            qv = {k: qes[k](bs)["Q"].gather(1, a_pi.view(-1, 1)).squeeze(1).cpu().numpy() for k in keys}
            for j, sid in enumerate(sids):
                r = sid_to_row.get(int(sid))
                if r is None:
                    continue
                rows.append(r)
                for k in keys:
                    vpi[k].append(float(qv[k][j]))
            neb += 1
            if neb % 200 == 0:
                print(f"    [eval] {neb:,} batches | {len(rows):,} states | "
                      f"{neb/(time.time()-t_ev):.1f} b/s", flush=True)
            if args.max_eval_batches and neb >= args.max_eval_batches:
                break

    rows = np.asarray(rows)
    ep_id = eidx_all[rows]
    uniq = np.unique(ep_id)
    ep_to_idx = {e: np.where(ep_id == e)[0] for e in uniq}
    rng = np.random.default_rng(0)

    def ci(dv):
        boot = np.empty(args.n_boot)
        for b in range(args.n_boot):
            pick = rng.choice(uniq, size=len(uniq), replace=True)
            boot[b] = dv[np.concatenate([ep_to_idx[e] for e in pick])].mean()
        return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))

    print(f"\n=== FQE decomposition (n={len(rows):,} test states, {len(uniq):,} episodes) ===")
    dV = {}
    vpi_arr = {}
    for k in keys:
        vp = np.asarray(vpi[k]); vb = G[k][rows]
        vpi_arr[k] = vp
        dv = vp - vb
        dV[k] = dv
        lo, hi = ci(dv)
        tag = "≈0" if (lo <= 0 <= hi) else ("POS" if lo > 0 else "NEG")
        print(f"  {k:11s}: V^π {vp.mean():+.4f}  V^β {vb.mean():+.4f}  "
              f"ΔV {dv.mean():+.4f}  95%CI [{lo:+.4f},{hi:+.4f}]  {tag}")

    # Σ-check: ΔV_total vs Σ ΔV_components ; residual = V^π_total − Σ V^π_comp (fit quality)
    sum_comp_dV = np.sum([dV[k] for k in COMPONENTS], axis=0)
    resid = vpi_arr["total"] - np.sum([vpi_arr[k] for k in COMPONENTS], axis=0)
    print("--- Σ-check (r_total = Σ components, so these should match) ---")
    print(f"  ΔV_total              = {dV['total'].mean():+.4f}")
    print(f"  Σ ΔV_components       = {sum_comp_dV.mean():+.4f}")
    print(f"  mean|V^π_total − ΣV^π_comp| (fit residual) = {np.abs(resid).mean():.4f}  "
          "(small ⇒ Q_e fits mutually consistent)")
    print("\n(ESTIMATE — FQE, no counterfactual ground truth; may be optimistic on the 4.3% "
          "divergent OOD states. Pair with L3 safety; single seed42.)")

    C.EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out = C.EVAL_DIR / f"ope_fqe_decompose_seed{args.seed}.json"
    out.write_text(json.dumps({
        "keys": keys, "gamma": GAMMA, "ckpt": str(ckpt), "warm_start": bool(args.warm_start),
        "epochs": args.epochs, "n_states": int(len(rows)), "n_episodes": int(len(uniq)),
        "delta_V": {k: float(dV[k].mean()) for k in keys},
        "V_pi": {k: float(vpi_arr[k].mean()) for k in keys},
        "sum_components_delta_V": float(sum_comp_dV.mean()),
        "fit_residual_abs_mean": float(np.abs(resid).mean()),
    }, indent=2))
    print(f"\n→ wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
