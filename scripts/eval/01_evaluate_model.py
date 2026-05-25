"""Stage 7/8 shared evaluation harness — Tier 1 overall + Tier 2 per-stratum top-1
on the TEST split (spec 05 §2-§3).

Loads a trained checkpoint (default best.pt per seed), runs a forward pass over
the test split using the streaming loader with **stratified=False** (true test
distribution — stratification is a training-only device), collects per-decision
predictions, and computes the metrics in railrl.eval.metrics. Writes one
per-seed JSON to outputs/eval/ that a later step aggregates across seeds.

This is the comparison口径 every method (CQL / baselines) will be scored on.
It also gives CQL's first TEST number (training only logged val).

Usage (local Windows):
    python scripts/eval/01_evaluate_model.py --seed 42
    python scripts/eval/01_evaluate_model.py --seed 42 --ckpt-name final_seed42.pt
Server (HPC sapphire) — prepend the env prefix from docs/IMPLEMENTATION_LOG.md.
Forward-only (no backward), so it is much faster than training; test ≈ 338k rows.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch
from torch.utils.data import DataLoader

from railrl import config as C
from railrl.encoders.input_pipeline import NormStats
from railrl.algorithms.transitions import StreamingTransitionDataset
from railrl.algorithms import trainer as T
from railrl.mdp.reward_v2 import TIME_LABELS_V2
from railrl.model import RailRLModel
from railrl.eval import metrics as Mx


def load_strata(path) -> dict:
    """sample_id (int) → stratum code (int 0..6). See railrl.eval.metrics.STRATUM_NAMES."""
    import pyarrow.parquet as pq
    t = pq.read_table(str(path), columns=["sample_id", "stratum"])
    return dict(zip(t.column("sample_id").to_pylist(), t.column("stratum").to_pylist()))


@torch.no_grad()
def collect_predictions(model, loader, lut, device, max_batches=None):
    """Forward pass over the loader; return aligned numpy arrays."""
    cols = {k: [] for k in ("chosen", "qarg", "rarg", "tpred", "tbk", "sid", "cq", "sbq")}
    for bi, (bs, _bsp, _done) in enumerate(loader):
        bs = bs.to(device)
        out = model(bs)
        B = bs.num_graphs
        a = bs.chosen_action_idx.view(B)
        q = out["Q"]
        sid = bs.sample_id.view(B)
        # Q-gap: chosen vs best OTHER valid action (drop chosen + the -1e9 mask sentinel)
        cq = q.gather(1, a.view(-1, 1)).squeeze(1)
        qtmp = q.clone().scatter_(1, a.view(-1, 1), float("-inf"))
        qtmp = qtmp.masked_fill(qtmp <= -1e8, float("-inf"))
        sb = qtmp.max(dim=1).values                      # -inf if no other valid action
        tb = (lut[sid.clamp(0, lut.size(0) - 1)] if lut is not None
              else torch.full((B,), -1, dtype=torch.long, device=device))

        cols["chosen"].append(a.cpu().numpy())
        cols["qarg"].append(q.argmax(1).cpu().numpy())
        cols["rarg"].append(out["route_scores"].argmax(1).cpu().numpy())
        cols["tpred"].append(out["time_logits"].argmax(1).cpu().numpy())
        cols["tbk"].append(tb.cpu().numpy())
        cols["sid"].append(sid.cpu().numpy())
        cols["cq"].append(cq.cpu().numpy())
        cols["sbq"].append(sb.cpu().numpy())
        if (bi + 1) % 200 == 0:
            print(f"  ...{bi + 1} batches")
        if max_batches and bi + 1 >= max_batches:
            break
    return {k: np.concatenate(v) if v else np.array([]) for k, v in cols.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description="Tier1+Tier2 eval on a split (spec 05 §2-§3).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--run-dir", type=str, default=None,
                    help="dir holding ckpt (default outputs/train/cql_seed{seed})")
    ap.add_argument("--ckpt", type=str, default=None,
                    help="explicit ckpt path (overrides --run-dir/--ckpt-name)")
    ap.add_argument("--ckpt-name", type=str, default="best.pt")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--batch-size", type=int, default=C.BATCH_SIZE)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--max-batches", type=int, default=None, help="debug: cap batches")
    ap.add_argument("--tag", type=str, default=None)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    try:                                  # HPC fd-sharing fix (TOOL_TRAPS §15)
        import torch.multiprocessing as _mp
        _mp.set_sharing_strategy("file_system")
    except (RuntimeError, AttributeError):
        pass

    run_dir = Path(args.run_dir) if args.run_dir else (C.TRAIN_DIR / f"cql_seed{args.seed}")
    ckpt = Path(args.ckpt) if args.ckpt else (run_dir / args.ckpt_name)
    tag = args.tag or f"cql_seed{args.seed}_{ckpt.stem}"
    out_path = Path(args.out) if args.out else (C.EVAL_DIR / f"{tag}_{args.split}_metrics.json")
    device = "cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu"
    print(f"device={device} | ckpt={ckpt} | split={args.split} | out={out_path}")

    if not ckpt.exists():
        print(f"[error] checkpoint not found: {ckpt}", file=sys.stderr)
        return 1

    stats = NormStats.load(C.NORMALIZATION_STATS_JSON)
    model = RailRLModel.build(stats).to(device)
    ck = torch.load(str(ckpt), map_location=device, weights_only=False)  # our trusted ckpt
    state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    model.load_state_dict(state)
    model.eval()
    ckpt_val_acc = ck.get("val_action_acc") if isinstance(ck, dict) else None
    if ckpt_val_acc is not None:
        print(f"[ckpt] {ckpt.name}: val_action_acc={ckpt_val_acc:.4f} "
              f"@ {ck.get('phase', '?')}{ck.get('epoch', '?')}")

    ds = StreamingTransitionDataset(
        C.SNAPSHOTS_V2_PARQUET, C.NORMALIZATION_STATS_JSON, split=args.split,
        batch_size=args.batch_size, shuffle=False, stratified=False)
    loader = DataLoader(ds, batch_size=None, num_workers=args.num_workers)

    time_lut = T.build_time_lut(TIME_LABELS_V2)
    lut = time_lut.to(device) if time_lut.numel() > 1 else None
    strata_map = load_strata(C.SNAPSHOTS_DIR / "stratum_labels.parquet")

    print(f"scanning {args.split} split (stratified=False, batch={args.batch_size}, "
          f"workers={args.num_workers}) ...")
    d = collect_predictions(model, loader, lut, device, args.max_batches)
    n = int(d["chosen"].size)
    if n == 0:
        print("[error] no decisions collected", file=sys.stderr)
        return 1

    stratum = np.array([strata_map.get(int(s), -1) for s in d["sid"]], dtype=np.int64)
    report = Mx.evaluate_all(d["chosen"], d["qarg"], stratum, route_argmax=d["rarg"],
                             time_pred=d["tpred"], time_bucket=d["tbk"],
                             chosen_q=d["cq"], secondbest_q=d["sbq"])
    report["meta"] = {
        "seed": args.seed, "ckpt": str(ckpt), "ckpt_name": ckpt.name,
        "split": args.split, "n": n, "val_action_acc_ckpt": ckpt_val_acc,
        "stratified_sampling": False,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))

    # ---- console table ----
    t1, t2 = report["tier1_overall"], report["tier2_stratified"]
    print(f"\n=== {tag} on {args.split} (n={n:,}) ===")
    print(f"action_top1   all={t1['action_top1_all']:.4f}   set-only={t1['action_top1_set']:.4f}")
    print(f"route_head    {t1['route_head_top1']:.4f}        time_head {t1['time_head_top1']:.4f}")
    print(f"wait_rate     model={t1['wait_rate_model']:.3f} signaller={t1['wait_rate_signaller']:.3f} "
          f"(Δ{t1['wait_rate_delta']:+.3f})  recall={t1['wait_recall']:.3f} prec={t1['wait_precision']:.3f}")
    if "q_gap" in report:
        g = report["q_gap"]
        print(f"Q-gap         mean={g['mean_gap']:+.3f}  frac_chosen_is_argmax={g['frac_chosen_is_argmax']:.3f}")
    print(f"per-stratum top-1   {'stratum':18s} {'all':>7s} {'set-only':>9s}   (n / n_set)")
    for name in ["overall"] + list(Mx.STRATUM_NAMES.values()):
        c = t2[name]
        aa = f"{c['acc_all']:.4f}" if c["acc_all"] == c["acc_all"] else "  n/a "
        aset = f"{c['acc_set']:.4f}" if c["acc_set"] == c["acc_set"] else "   n/a  "
        print(f"  {'':18s} {name:18s} {aa:>7s} {aset:>9s}   ({c['n']:,} / {c['n_set']:,})")
    print(f"\n→ wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
