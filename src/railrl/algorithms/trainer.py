"""spec 04 §3/§5/§6 — training infrastructure for the CQL 3-phase protocol.

Reusable pieces (the orchestration lives in scripts/train/09_train.py):
  * build_param_groups  — AdamW groups: wd=0 on embeddings/LayerNorm/bias (§5.3)
  * phase_lr            — per-phase warmup→cosine LR (§5.2), set manually each step
  * set_encoder_requires_grad — freeze/unfreeze encoder for Phase B/C
  * build_time_lut      — sample_id → time_bucket dense lookup (from time_labels_v2)
  * compute_loss        — the right loss per phase (A: aux | B: CQL | C: CQL+aux)
  * evaluate            — val route/action accuracy + mean losses

Phases (spec 04 §3): A encoder+aux (5ep) → B freeze-encoder CQL (15ep, target net
cloned) → C joint CQL+aux (20ep). γ=0.95, α=5, target soft-update τ=0.005.
"""
from __future__ import annotations
import math

from .. import config as C
from . import losses as L

# RailRLModel submodules that constitute the ENCODER (frozen in Phase B)
ENCODER_ATTRS = ["hgt", "seq", "sched", "fusion"]


def build_param_groups(model, weight_decay: float = C.WEIGHT_DECAY):
    """Two AdamW groups: weight_decay on matrix weights only; 0 on embeddings,
    LayerNorm, and biases (spec 04 §5.3, standard practice). Skips frozen params."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        lname = name.lower()
        if p.ndim <= 1 or "emb" in lname or "norm" in lname or name.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]


def phase_lr(step: int, total_steps: int, warmup: int, peak: float,
             final: float = 3e-5) -> float:
    """Linear warmup 0→peak over `warmup` steps, then cosine peak→final (§5.2)."""
    if warmup > 0 and step < warmup:
        return peak * (step + 1) / warmup
    prog = (step - warmup) / max(total_steps - warmup, 1)
    prog = min(max(prog, 0.0), 1.0)
    return final + 0.5 * (peak - final) * (1.0 + math.cos(math.pi * prog))


def set_lr(optimizer, lr: float):
    for g in optimizer.param_groups:
        g["lr"] = lr


def set_encoder_requires_grad(model, flag: bool):
    for attr in ENCODER_ATTRS:
        mod = getattr(model, attr, None)
        if mod is not None:
            for p in mod.parameters():
                p.requires_grad = flag


def build_time_lut(time_labels_path):
    """Dense long tensor: sample_id → time_bucket (-1 where absent). Self-sized to
    max sample_id+1. Returns a length-1 [-1] tensor if the file is missing
    (→ L_time = 0)."""
    import torch
    import pyarrow.parquet as pq
    try:
        tb = pq.read_table(str(time_labels_path), columns=["sample_id", "time_bucket"])
    except Exception:
        print(f"[trainer][warn] {time_labels_path} not found — L_time will be 0.")
        return torch.full((1,), -1, dtype=torch.long)
    sid = tb.column("sample_id").to_numpy()
    bk = tb.column("time_bucket").to_numpy()
    sid_t = torch.from_numpy(sid.copy()).long()
    lut = torch.full((int(sid_t.max().item()) + 1,), -1, dtype=torch.long)
    lut[sid_t] = torch.from_numpy(bk.copy()).long()
    return lut


def _batch_dict(batch_s, done, time_lut, device):
    import torch
    B = batch_s.num_graphs
    a = batch_s.chosen_action_idx.view(B)
    r = batch_s.r_total.view(B)
    sid = batch_s.sample_id.view(B)
    if time_lut is not None and time_lut.numel() > 1:
        idx = sid.clamp(0, time_lut.size(0) - 1)
        tb = time_lut.to(device)[idx]
    else:
        tb = torch.full((B,), -1, dtype=torch.long, device=device)
    return {"chosen_action_idx": a, "r_total": r, "done": done.to(device),
            "set_mask": a > 0, "time_bucket": tb}


def compute_loss(model, target, batch_s, batch_sp, done, time_lut, phase, device):
    """Loss for the given phase. Returns (loss, parts dict)."""
    import torch
    bd = _batch_dict(batch_s, done, time_lut, device)
    out = model(batch_s)
    if phase == "A":
        l_route, l_time = L.aux_losses(out["route_scores"], out["time_logits"],
                                       bd["chosen_action_idx"], bd["set_mask"],
                                       bd["time_bucket"])
        loss = L.W_ROUTE * l_route + L.W_TIME * l_time
        return loss, {"L_route": l_route.detach(), "L_time": l_time.detach(),
                      "L_total": loss.detach()}
    with torch.no_grad():
        out_next = target(batch_sp)
    if phase == "B":
        loss, parts = L.cql_loss(out["Q"], bd["chosen_action_idx"], bd["r_total"],
                                 out_next["Q"], bd["done"])
        parts["L_total"] = loss.detach()
        return loss, parts
    # phase C — joint CQL + aux
    return L.cql_total(out, {"Q": out_next["Q"]}, bd)


def evaluate(model, loader, time_lut, device, max_batches: int = None):
    """Val metrics: route-head acc (set rows) + action acc (Q argmax) + time-head
    acc (valid time_bucket rows) + |Q| magnitude (Phase B gate)."""
    import torch
    model.eval()
    rc = rt = ac = tot = tc = tt = 0
    q_absmax = 0.0
    lut = time_lut.to(device) if (time_lut is not None and time_lut.numel() > 1) else None
    with torch.no_grad():
        for bi, (bs, bsp, done) in enumerate(loader):
            bs = bs.to(device)
            out = model(bs)
            a = bs.chosen_action_idx.view(bs.num_graphs)
            set_mask = a > 0
            if set_mask.any():
                pred = out["route_scores"][set_mask].argmax(1)
                rc += (pred == (a[set_mask] - 1)).sum().item()
                rt += int(set_mask.sum().item())
            ap = out["Q"].argmax(1)
            ac += (ap == a).sum().item()
            tot += a.numel()
            # |Q| over VALID actions only — exclude the -1e9 masked-action sentinel
            # (spec §6.2 fills invalid candidates with -1e9; it's not a real Q value).
            qv = out["Q"]
            qv = qv[qv > -1e8]
            if qv.numel():
                q_absmax = max(q_absmax, float(qv.abs().max().item()))
            if lut is not None:
                sid = bs.sample_id.view(bs.num_graphs)
                tb = lut[sid.clamp(0, lut.size(0) - 1)]
                v = tb >= 0
                if v.any():
                    tp = out["time_logits"][v].argmax(1)
                    tc += (tp == tb[v]).sum().item()
                    tt += int(v.sum().item())
            if max_batches and bi + 1 >= max_batches:
                break
    model.train()
    return {"route_acc": rc / max(rt, 1), "action_acc": ac / max(tot, 1),
            "time_acc": tc / max(tt, 1), "q_absmax": q_absmax}
