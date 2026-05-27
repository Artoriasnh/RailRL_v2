"""spec 04 §2 — offline-RL loss functions (CQL main / IQL alt / BC baseline + aux).

All functions are pure (model outputs + labels → scalar loss tensor); the
training loop (4.7.2) runs the model on s and s' and the target net, then calls
these. Keeping them standalone makes every term unit-testable with synthetic
tensors (scripts/train/07_smoke_losses.py).

Conventions (match spec 03 §6 + the loader):
  Q            (B, K+1)  per-action Q; index 0 = wait, 1..K = candidate routes.
                          Masked (invalid) actions are already -1e9 (QNetwork).
  chosen_action_idx (B,) long in {0..K}; 0 = wait, j = route j (1-indexed).
  r            (B,)      r_total for the transition.
  done         (B,)      1.0 if terminal (last in episode), else 0.0.
  action_mask  (B, K+1)  1 valid / 0 invalid (for the NEXT state's max).
  route_scores (B, K)    RouteHead dot-product scores (masked -1e9).
  time_logits  (B, 5)    TimeHead logits.
  time_bucket  (B,)      long in {0..4}; -1 = excluded (NaN lead time).

γ = 0.95 (spec 02 §5.4), α = 5.0 (CQL), τ = 0.7 (IQL expectile), β = 3.0 (AWR).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .. import config as C

GAMMA = C.DISCOUNT_GAMMA          # 0.95
CQL_ALPHA = C.CQL_ALPHA           # 5.0
IQL_TAU = C.IQL_EXPECTILE_TAU     # 0.7
IQL_BETA = C.IQL_AWR_BETA         # 3.0
W_ROUTE = 0.5                     # spec 03 §7.4 / spec 04 §2.3
W_TIME = 0.2
LAMBDA_WAIT = 0.3                 # spec 04 §2.5 (matches w_wait)


# ============================================================
# Shared helpers
# ============================================================

def _gather_chosen(q: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """Q(s, a_data) → (B,). a is the chosen_action_idx in {0..K}."""
    return q.gather(1, a.view(-1, 1)).squeeze(1)


def _zero(like: torch.Tensor) -> torch.Tensor:
    """A scalar 0 that stays connected to the autograd graph (zero gradient).

    Using `like.new_zeros(())` would DETACH the term — then a Phase-A batch with
    no set rows AND no valid time labels makes the whole loss a constant with no
    grad_fn, and loss.backward() raises "does not require grad". `like.sum()*0`
    keeps the graph edge (gradient is exactly 0), so such a batch is a clean
    no-op instead of a crash.
    """
    return like.sum() * 0.0


# ============================================================
# CQL (main)  — spec 04 §2.1
# ============================================================

def td_target(r, q_target_next, done, gamma: float = GAMMA):
    """y = r + γ·(1-done)·max_{a'} Q_target(s', a').

    Invalid next-actions are -1e9 in q_target_next, so .max ignores them; wait
    (idx 0) is always valid so the max is finite. done rows zero the bootstrap.
    """
    with torch.no_grad():
        q_next_max = q_target_next.max(dim=1).values            # (B,)
        # guard: if a row were fully masked (shouldn't happen), clamp the -1e9
        q_next_max = torch.where(torch.isfinite(q_next_max),
                                 q_next_max, torch.zeros_like(q_next_max))
        return r + gamma * (1.0 - done) * q_next_max


def td_loss(q, a, r, q_target_next, done, gamma: float = GAMMA):
    """L_TD = E[(Q(s,a) - y)²]."""
    q_sa = _gather_chosen(q, a)
    y = td_target(r, q_target_next, done, gamma)
    return F.mse_loss(q_sa, y)


def cql_conservative(q, a):
    """L_cons = E[ logΣ_a exp Q(s,a) − Q(s, a_data) ].

    logsumexp over all actions; masked (-1e9) contribute exp≈0, i.e. it is
    effectively over the valid action set A_s.
    """
    lse = torch.logsumexp(q, dim=1)        # (B,)
    q_data = _gather_chosen(q, a)
    return (lse - q_data).mean()


def cql_loss(q, a, r, q_target_next, done, *, alpha: float = CQL_ALPHA,
             gamma: float = GAMMA):
    """L_CQL = L_TD + α·L_cons. Returns (total, parts dict)."""
    l_td = td_loss(q, a, r, q_target_next, done, gamma)
    l_cons = cql_conservative(q, a)
    total = l_td + alpha * l_cons
    return total, {"L_TD": l_td.detach(), "L_cons": l_cons.detach()}


# ============================================================
# Auxiliary supervised — spec 04 §2.2
# ============================================================

def route_loss(route_scores, a, set_mask):
    """L_route = CE(route_scores[set], (a-1)[set]). a-1 maps action→route idx."""
    if set_mask.any():
        tgt = (a[set_mask] - 1).long()
        return F.cross_entropy(route_scores[set_mask], tgt)
    return _zero(route_scores)


def time_loss(time_logits, time_bucket, valid_mask=None):
    """L_time = CE(time_logits[valid], time_bucket[valid]); -1 buckets excluded."""
    if valid_mask is None:
        valid_mask = time_bucket >= 0
    if valid_mask.any():
        return F.cross_entropy(time_logits[valid_mask], time_bucket[valid_mask].long())
    return _zero(time_logits)


def aux_losses(route_scores, time_logits, a, set_mask, time_bucket,
               time_valid_mask=None):
    l_route = route_loss(route_scores, a, set_mask)
    l_time = time_loss(time_logits, time_bucket, time_valid_mask)
    return l_route, l_time


# ============================================================
# Total (CQL + aux)  — spec 04 §2.3
# ============================================================

def cql_total(out, target_out_next, batch, *, alpha: float = CQL_ALPHA,
              w_route: float = W_ROUTE, w_time: float = W_TIME):
    """L_total = L_CQL + 0.5·L_route + 0.2·L_time.

    out:            dict from model(s)  → 'Q','route_scores','time_logits'
    target_out_next:dict from target_model(s') → 'Q'  (masked)
    batch:          object/dict with chosen_action_idx, r_total, done,
                    set_mask, time_bucket
    """
    q = out["Q"]
    a = batch["chosen_action_idx"]
    l_cql, parts = cql_loss(q, a, batch["r_total"], target_out_next["Q"],
                            batch["done"], alpha=alpha)
    l_route, l_time = aux_losses(out["route_scores"], out["time_logits"], a,
                                 batch["set_mask"], batch["time_bucket"])
    total = l_cql + w_route * l_route + w_time * l_time
    parts.update({"L_CQL": l_cql.detach(), "L_route": l_route.detach(),
                  "L_time": l_time.detach(), "L_total": total.detach()})
    return total, parts


# ============================================================
# IQL (alternative)  — spec 04 §2.4
# ============================================================

def expectile_loss(diff, tau: float = IQL_TAU):
    """L²_τ(u) = |τ - 1[u<0]|·u²,  mean over batch."""
    w = torch.where(diff < 0, 1.0 - tau, tau)
    return (w * diff.pow(2)).mean()


def iql_value_loss(q_target_sa, v, tau: float = IQL_TAU):
    """L_V = expectile_τ( Q_target(s,a) − V(s) ). q_target_sa detached."""
    return expectile_loss(q_target_sa.detach() - v, tau)


def iql_q_loss(q_sa, r, v_next, done, gamma: float = GAMMA):
    """L_Q = E[(r + γ(1-done)V(s') − Q(s,a))²]. V(s') detached."""
    with torch.no_grad():
        y = r + gamma * (1.0 - done) * v_next
    return F.mse_loss(q_sa, y)


def iql_policy_loss(q_sa, v, logp_a, beta: float = IQL_BETA, *, clip: float = 100.0):
    """L_π (AWR) = E[ -exp(β(Q(s,a)−V(s)))·log π(a|s) ], advantage detached."""
    with torch.no_grad():
        w = torch.exp(beta * (q_sa - v)).clamp(max=clip)
    return -(w * logp_a).mean()


def iql_total(out, target_out, batch, *, v, v_next, tau: float = IQL_TAU,
              beta: float = IQL_BETA, w_route: float = W_ROUTE,
              w_time: float = W_TIME):
    """L = L_V + L_Q + L_π + 0.5·L_route + 0.2·L_time. (implicit policy = softmax Q)

    out:        model(s) outputs; target_out: target_model(s) for L_V's Q_target(s,a).
    v, v_next:  value-head V(s), V(s') (B,).
    """
    q = out["Q"]
    a = batch["chosen_action_idx"]
    q_sa = _gather_chosen(q, a)
    q_target_sa = _gather_chosen(target_out["Q"], a)
    logp = F.log_softmax(q, dim=1)
    logp_a = logp.gather(1, a.view(-1, 1)).squeeze(1)

    l_v = iql_value_loss(q_target_sa, v, tau)
    l_q = iql_q_loss(q_sa, batch["r_total"], v_next, batch["done"])
    l_pi = iql_policy_loss(q_sa, v, logp_a, beta)
    l_route, l_time = aux_losses(out["route_scores"], out["time_logits"], a,
                                 batch["set_mask"], batch["time_bucket"])
    total = l_v + l_q + l_pi + w_route * l_route + w_time * l_time
    parts = {"L_V": l_v.detach(), "L_Q": l_q.detach(), "L_pi": l_pi.detach(),
             "L_route": l_route.detach(), "L_time": l_time.detach(),
             "L_total": total.detach()}
    return total, parts


# ============================================================
# BC baseline  — spec 04 §2.5
# ============================================================

def bc_q_loss(q, a):
    """Simple BC-via-Q: CE over the full action set vs chosen_action_idx.

    Used as the Phase-A / sanity behaviour-cloning signal and in 06_smoke_model.
    """
    return F.cross_entropy(q, a.long())


def bc_baseline_loss(route_scores, wait_logits, a, label_is_wait,
                     set_mask, lambda_wait: float = LAMBDA_WAIT):
    """L_BC = CE(route[set], a-1) + λ_wait·BCE(wait_logits, is_wait).

    wait_logits (B,) raw logit from a Linear(s_emb,1); label_is_wait (B,) float.
    """
    l_route = route_loss(route_scores, a, set_mask)
    l_wait = F.binary_cross_entropy_with_logits(
        wait_logits.view(-1), label_is_wait.view(-1).float())
    return l_route + lambda_wait * l_wait, {"L_route": l_route.detach(),
                                            "L_wait": l_wait.detach()}


# ============================================================
# Target network (CQL §6.2) — Polyak soft update
# ============================================================

def soft_update(target: "torch.nn.Module", online: "torch.nn.Module",
                tau: float = C.CQL_TARGET_TAU):
    """θ_target ← (1-τ)·θ_target + τ·θ_online (spec 04 §6.2, τ=0.005)."""
    with torch.no_grad():
        for pt, po in zip(target.parameters(), online.parameters()):
            pt.mul_(1.0 - tau).add_(po, alpha=tau)
        for bt, bo in zip(target.buffers(), online.buffers()):
            bt.copy_(bo)
