"""Stage 4.7.2c/d/5 — CQL 3-phase training driver (spec 04 §3-§6, §11 gates).

Phase A (encoder + aux, 5 ep) → Phase B (freeze encoder, CQL, 15 ep, target net
cloned) → Phase C (joint CQL + aux, 20 ep). AdamW + per-phase warmup→cosine LR,
grad-clip 1.0, target soft-update τ=0.005, L_time wired to time_labels_v2.

Loaders:
  * --smoke : OLD map-style TransitionDataset on a tiny Subset (quick loop check, no GPU).
  * --sanity / full : StreamingTransitionDataset (Stage 4.7.2d) — worker-safe,
    decode-once, block-shuffled, STRATIFIED (spec §4.4). Requires the canonical
    snapshots + episodes_v2 + stratum_labels/weights sidecars.

Stage 5 sanity (spec 04 §11), RTX-5070 defaults (batch 96 / 8 workers / ~50k rows/epoch):
    python scripts/train/09_train.py --sanity
Full run (per seed):
    python scripts/train/09_train.py --seed 42 --out outputs/train/cql_seed42
"""
from __future__ import annotations
import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch
from torch.utils.data import DataLoader, Subset

from railrl import config as C
from railrl.encoders.input_pipeline import NormStats
from railrl.algorithms.transitions import (
    TransitionDataset, StreamingTransitionDataset, transition_collate,
)
from railrl.algorithms import trainer as T, losses as L
from railrl.mdp.reward_v2 import TIME_LABELS_V2
from railrl.model import RailRLModel


def set_seed(s: int):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def make_loader(ds, batch_size, shuffle, num_workers):
    """Map-style loader (smoke only)."""
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, collate_fn=transition_collate,
                      drop_last=False)


def count_split_rows(split: str) -> int:
    import pyarrow.parquet as pq
    s = np.asarray(pq.read_table(str(C.SNAPSHOTS_V2_PARQUET), columns=["split"])
                   .column("split").to_pylist())
    return int((s == split).sum())


# spec 04 §11 per-phase success gates (targets; printed, not hard-aborted in sanity)
def check_gates(phase, means, val):
    if not val:   # phase fully completed in a prior (resumed) session → nothing ran
        return []
    g = []
    if phase == "A":
        g.append(("route_acc≥0.50", val["route_acc"] >= 0.50, f"{val['route_acc']:.3f}"))
        g.append(("time_acc≥0.35", val["time_acc"] >= 0.35, f"{val['time_acc']:.3f}"))
        g.append(("loss finite", bool(np.isfinite(means.get("L_total", 0.0))), f"{means.get('L_total', 0):.3f}"))
    elif phase == "B":
        g.append(("Q_top1≥0.55", val["action_acc"] >= 0.55, f"{val['action_acc']:.3f}"))
        g.append(("|Q|<100", val["q_absmax"] < 100.0, f"{val['q_absmax']:.1f}"))
    else:  # C
        g.append(("Q_top1≥0.65", val["action_acc"] >= 0.65, f"{val['action_acc']:.3f}"))
    print(f"  [gate {phase}] " + "  ".join(
        f"{'✓' if ok else '✗'} {name}({v})" for name, ok, v in g))
    return g


def run_phase(phase, model, target, ds_train, val_loader, time_lut, device, *,
              epochs, peak_lr, batch_size, num_workers, warmup, batches_per_epoch,
              ckpt_dir, log, streaming, epoch_base, track_best=False, best_state=None,
              start_epoch=0, gstep0=0, resume_optim_sd=None, seed=42, gates_acc=None):
    optim = torch.optim.AdamW(T.build_param_groups(model), lr=peak_lr,
                              betas=(0.9, 0.999), eps=1e-8)
    if resume_optim_sd is not None:
        try:
            optim.load_state_dict(resume_optim_sd)
            print(f"[resume] optimizer state restored for phase {phase}")
        except Exception as e:  # param groups changed etc. → continue fresh
            print(f"[resume][warn] optimizer not restored ({e}); fresh.")
    total_steps = max(epochs * batches_per_epoch, 1)
    gstep = gstep0
    means = {}; val = {}
    for ep in range(start_epoch, epochs):
        if streaming:
            ds_train.set_epoch(epoch_base + ep)
            loader = DataLoader(ds_train, batch_size=None, num_workers=num_workers,
                                persistent_workers=False)
        else:
            loader = make_loader(ds_train, batch_size, True, num_workers)
        model.train(); run = {}; nb = 0; t0 = time.time(); lr = peak_lr
        for (bs, bsp, done) in loader:
            bs = bs.to(device); bsp = bsp.to(device)
            lr = T.phase_lr(gstep, total_steps, warmup, peak_lr); T.set_lr(optim, lr)
            optim.zero_grad()
            loss, parts = T.compute_loss(model, target, bs, bsp, done, time_lut, phase, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP)
            optim.step()
            if target is not None and phase in ("B", "C"):
                L.soft_update(target, model)
            for k, v in parts.items():
                run[k] = run.get(k, 0.0) + float(v)
            nb += 1; gstep += 1
            if nb >= batches_per_epoch:
                break
        means = {k: run[k] / max(nb, 1) for k in run}
        val = T.evaluate(model, val_loader, time_lut, device, max_batches=50)
        rec = {"phase": phase, "epoch": ep + 1, "lr": lr, **means,
               **{f"val_{k}": v for k, v in val.items()}}
        log.append(rec)
        print(f"[{phase}] ep {ep+1}/{epochs} lr={lr:.2e} "
              + " ".join(f"{k}={means[k]:.4f}" for k in means)
              + f" | val route={val['route_acc']:.3f} time={val['time_acc']:.3f} "
              f"act={val['action_acc']:.3f} |Q|max={val['q_absmax']:.1f}"
              + f" ({time.time()-t0:.0f}s)", flush=True)
        # best-by-val (spec §8.4) — Phase C only; small + worth keeping
        if track_best and best_state is not None and val["action_acc"] > best_state["acc"]:
            best_state.update(acc=val["action_acc"], phase=phase, epoch=ep + 1)
            torch.save({"phase": phase, "epoch": ep + 1, "model": model.state_dict(),
                        "val_action_acc": val["action_acc"]}, ckpt_dir / "best.pt")
        # rolling resume checkpoint (overwrite, 1 file) — epoch-granularity resume for
        # 12h windows. `epoch`=completed epochs IN THIS PHASE; restart loses ≤1 epoch.
        torch.save({"seed": seed, "phase": phase, "epoch": ep + 1, "gstep": gstep,
                    "model": model.state_dict(),
                    "target": target.state_dict() if target is not None else None,
                    "optimizer": optim.state_dict(),
                    "best_state": best_state, "log": log, "gates": gates_acc},
                   ckpt_dir / f"resume_seed{seed}.pt")
    # phase-end checkpoint only (spec §8.5) — NOT per-epoch (server disk is tight)
    torch.save({"phase": phase, "model": model.state_dict(),
                "target": target.state_dict() if target is not None else None},
               ckpt_dir / f"phase_{phase}_end.pt")
    return means, val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--algo", choices=["cql"], default="cql")
    ap.add_argument("--smoke", action="store_true", help="tiny end-to-end loop check (old loader)")
    ap.add_argument("--smoke-n", type=int, default=64)
    ap.add_argument("--sanity", action="store_true",
                    help="Stage 5: streaming+stratified, ~50k rows/epoch, full 3-phase, §11 gates")
    ap.add_argument("--sanity-batches", type=int, default=None,
                    help="batches/epoch in sanity (default ≈ 50k/batch_size)")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--max-batches", type=int, default=None, help="cap batches/epoch (full runs)")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--resume", action="store_true",
                    help="resume from resume_seed{N}.pt in --out dir (12h-window-safe)")
    args = ap.parse_args()

    set_seed(args.seed)
    # HPC fd-sharing fix: PyTorch's default 'file_descriptor' strategy exhausts the
    # per-process fd budget with many workers passing PyG batches → "received 0 items
    # of ancdata". 'file_system' shares via /tmp instead → lets num_workers≥16 work.
    try:
        import torch.multiprocessing as _mp
        _mp.set_sharing_strategy("file_system")
    except (RuntimeError, AttributeError):
        pass
    device = "cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu"
    print(f"device={device} | algo={args.algo} | seed={args.seed} | "
          f"mode={'smoke' if args.smoke else 'sanity' if args.sanity else 'full'}")

    stats = NormStats.load(C.NORMALIZATION_STATS_JSON)
    time_lut = T.build_time_lut(TIME_LABELS_V2)

    if args.smoke:
        # ---- map-style tiny subset (no streaming, no GPU needed) ----
        full = TransitionDataset(C.SNAPSHOTS_V2_PARQUET, C.NORMALIZATION_STATS_JSON, split="val")
        idx = list(range(min(len(full), args.smoke_n)))
        ds_train = Subset(full, idx)
        val_loader = make_loader(Subset(full, idx), 8, False, 0)
        streaming = False
        epochs = (1, 1, 1); batch_size = 8; warmup = 2; bpe = 4
        num_workers = 0; peak_c = 1.5e-4
        out = Path(args.out or (C.TRAIN_DIR / "smoke"))
    else:
        # ---- streaming + stratified (Stage 4.7.2d loader) ----
        streaming = True
        batch_size = args.batch_size or (96 if args.sanity else C.BATCH_SIZE)
        num_workers = args.num_workers if args.num_workers is not None else 8
        ds_train = StreamingTransitionDataset(
            C.SNAPSHOTS_V2_PARQUET, C.NORMALIZATION_STATS_JSON, split="train",
            batch_size=batch_size, shuffle=True, stratified=True, seed=args.seed)
        ds_val = StreamingTransitionDataset(
            C.SNAPSHOTS_V2_PARQUET, C.NORMALIZATION_STATS_JSON, split="val",
            batch_size=batch_size, shuffle=False, stratified=False)
        val_loader = DataLoader(ds_val, batch_size=None, num_workers=0)
        epochs = (C.PHASE_A_EPOCHS, C.PHASE_B_EPOCHS, C.PHASE_C_EPOCHS)
        warmup = C.WARMUP_STEPS; peak_c = 1.5e-4
        train_rows = count_split_rows("train")
        if args.sanity:
            bpe = args.sanity_batches or max(1, 50000 // batch_size)
            out = Path(args.out or (C.TRAIN_DIR / f"sanity_seed{args.seed}"))
        else:
            bpe = args.max_batches or max(1, train_rows // batch_size)
            out = Path(args.out or (C.TRAIN_DIR / f"cql_seed{args.seed}"))
        print(f"train_rows≈{train_rows:,} | batches/epoch={bpe} | batch={batch_size} "
              f"workers={num_workers}")

    out.mkdir(parents=True, exist_ok=True)
    model = RailRLModel.build(stats).to(device)
    print(f"model params: {sum(p.numel() for p in model.parameters()):,} | "
          f"epochs(A,B,C)={epochs} warmup={warmup} out={out}")

    log = []; gates = {}; best_state = {"acc": -1.0}
    # ---- resume (epoch-granularity, 12h-window-safe) ----
    resume_path = out / f"resume_seed{args.seed}.pt"
    start_phase, start_ep, gstep0, optim_sd, target_sd = "A", 0, 0, None, None
    if args.resume and resume_path.exists():
        # weights_only=False: our OWN checkpoint (trusted) holds non-tensor state
        # (best_state/log/gates incl. python+numpy scalars). PyTorch 2.6 defaults
        # weights_only=True which rejects numpy scalars → must pass False here.
        rk = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(rk["model"])
        start_phase, start_ep, gstep0 = rk["phase"], rk["epoch"], rk["gstep"]
        best_state = rk.get("best_state") or best_state
        log = rk.get("log") or []
        gates = rk.get("gates") or {}
        optim_sd, target_sd = rk.get("optimizer"), rk.get("target")
        print(f"[resume] phase={start_phase} done_epochs_in_phase={start_ep} "
              f"gstep={gstep0} | restored best/log/gates")
    elif args.resume:
        print(f"[resume] no {resume_path.name} — starting fresh.")
    common = dict(batch_size=batch_size, num_workers=num_workers, warmup=warmup,
                  batches_per_epoch=bpe, ckpt_dir=out, log=log, streaming=streaming,
                  seed=args.seed, best_state=best_state, gates_acc=gates)
    pidx = {"A": 0, "B": 1, "C": 2}; si = pidx[start_phase]

    # ---- Phase A: encoder + aux heads (no target net) ----
    if si <= 0:
        T.set_encoder_requires_grad(model, True)
        mA, vA = run_phase("A", model, None, ds_train, val_loader, time_lut, device,
                           epochs=epochs[0], peak_lr=C.LR, epoch_base=0,
                           start_epoch=(start_ep if start_phase == "A" else 0),
                           gstep0=(gstep0 if start_phase == "A" else 0),
                           resume_optim_sd=(optim_sd if start_phase == "A" else None),
                           **common)
        gates["A"] = check_gates("A", mA, vA)

    # ---- Phase B: freeze encoder, train Q (CQL). target = restored (B/C resume) or clone ----
    T.set_encoder_requires_grad(model, False)
    target = copy.deepcopy(model)
    if target_sd is not None and start_phase in ("B", "C"):
        target.load_state_dict(target_sd)
    for p in target.parameters():
        p.requires_grad = False
    if si <= 1:
        mB, vB = run_phase("B", model, target, ds_train, val_loader, time_lut, device,
                           epochs=epochs[1], peak_lr=C.LR, epoch_base=epochs[0],
                           start_epoch=(start_ep if start_phase == "B" else 0),
                           gstep0=(gstep0 if start_phase == "B" else 0),
                           resume_optim_sd=(optim_sd if start_phase == "B" else None),
                           **common)
        gates["B"] = check_gates("B", mB, vB)

    # ---- Phase C: unfreeze, joint CQL + aux (half peak LR) ----
    T.set_encoder_requires_grad(model, True)
    mC, vC = run_phase("C", model, target, ds_train, val_loader, time_lut, device,
                       epochs=epochs[2], peak_lr=peak_c, epoch_base=epochs[0] + epochs[1],
                       start_epoch=(start_ep if start_phase == "C" else 0),
                       gstep0=(gstep0 if start_phase == "C" else 0),
                       resume_optim_sd=(optim_sd if start_phase == "C" else None),
                       track_best=True, **common)
    gates["C"] = check_gates("C", mC, vC)

    torch.save({"model": model.state_dict(), "seed": args.seed}, out / f"final_seed{args.seed}.pt")
    (out / f"train_log_seed{args.seed}.json").write_text(json.dumps(log, indent=2))
    print("\n=== §11 gate summary ===")
    for ph in ("A", "B", "C"):
        for name, ok, v in gates.get(ph, []):
            print(f"  {ph}: {'PASS' if ok else 'FAIL'}  {name}  ({v})")
    print(f"\nbest val_action_acc={best_state['acc']:.4f} "
          f"@ {best_state.get('phase','?')} ep{best_state.get('epoch','?')} → best.pt")
    print(f"ckpts: phase_A_end.pt / phase_B_end.pt / phase_C_end.pt / best.pt / final_seed{args.seed}.pt")
    print(f"DONE — final → {out / f'final_seed{args.seed}.pt'}  log → train_log_seed{args.seed}.json")


if __name__ == "__main__":
    main()
