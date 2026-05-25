"""Stage 4.7.1 smoke test — offline-RL losses on synthetic tensors.

No model/data needed. Verifies each loss term (CQL/IQL/BC + aux) is finite,
scalar, differentiable, handles masks/edge-cases, and matches hand-computed
values where checkable. Run on Windows (needs torch):
    python scripts/train/07_smoke_losses.py
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from railrl.algorithms import losses as L


def main():
    import torch
    import torch.nn as nn
    torch.manual_seed(0)
    B, K = 6, 14
    NEG = -1e9

    # --- synthetic Q (B,K+1): mask some actions to -1e9; wait(0) always valid ---
    def make_q(requires_grad=False):
        q = torch.randn(B, K + 1)
        # row r: valid actions = wait + first (r%K)+1 routes; rest masked
        mask = torch.zeros(B, K + 1, dtype=torch.bool)
        mask[:, 0] = True
        for r in range(B):
            nv = (r % K) + 1
            mask[r, 1:1 + nv] = True
        q = torch.where(mask, q, torch.full_like(q, NEG))
        q.requires_grad_(requires_grad)
        return q, mask

    q, vmask = make_q(requires_grad=True)
    q_tgt_next, _ = make_q(requires_grad=False)

    # chosen action: pick a VALID action per row (route if available else wait)
    chosen = torch.tensor([min(1, (r % K)) for r in range(B)], dtype=torch.long)
    # ensure chosen is valid (it is: row0 chosen=0 wait; others chosen=1 first route)
    r_total = torch.tensor([0.5, -0.3, 1.0, -0.5, 0.2, -1.0])
    done = torch.tensor([0., 0., 1., 0., 0., 1.])          # rows 2,5 terminal
    set_mask = chosen > 0                                   # rows with a route
    route_scores = torch.where(vmask[:, 1:], torch.randn(B, K), torch.full((B, K), NEG))
    route_scores.requires_grad_(True)
    time_logits = torch.randn(B, 5, requires_grad=True)
    time_bucket = torch.tensor([2, -1, 0, 4, -1, 3])        # rows 1,4 excluded

    # ---- td_target hand-check ----
    y = L.td_target(r_total, q_tgt_next, done)
    qmax = torch.where(torch.isfinite(q_tgt_next.max(1).values),
                       q_tgt_next.max(1).values, torch.zeros(B))
    y_ref = r_total + L.GAMMA * (1 - done) * qmax
    assert torch.allclose(y, y_ref), "td_target mismatch"
    assert torch.allclose(y[done == 1], r_total[done == 1]), "done rows must have y=r"
    print(f"td_target OK (done rows y==r): {y.tolist()}")

    # ---- CQL ----
    l_cql, parts = L.cql_loss(q, chosen, r_total, q_tgt_next, done)
    cons = L.cql_conservative(q, chosen)
    assert torch.isfinite(l_cql), "L_CQL not finite"
    assert cons.item() >= -1e-5, f"L_cons should be >=0 (logsumexp>=q_data), got {cons.item()}"
    # logsumexp must ignore -1e9 masked actions: compare to logsumexp over valid only
    lse_all = torch.logsumexp(q, dim=1)
    lse_valid = torch.stack([torch.logsumexp(q[r, vmask[r]], 0) for r in range(B)])
    assert torch.allclose(lse_all, lse_valid, atol=1e-4), "logsumexp leaked masked actions"
    print(f"CQL OK: L_TD={parts['L_TD']:.4f} L_cons={cons.item():.4f} (>=0) L_CQL={l_cql.item():.4f}")

    # ---- aux losses + masks ----
    l_route = L.route_loss(route_scores, chosen, set_mask)
    l_time = L.time_loss(time_logits, time_bucket)
    assert torch.isfinite(l_route) and torch.isfinite(l_time)
    # edge: empty set_mask -> route loss 0; all-invalid time -> time loss 0
    z = L.route_loss(route_scores, torch.zeros_like(chosen), torch.zeros_like(set_mask))
    zt = L.time_loss(time_logits, torch.full_like(time_bucket, -1))
    assert z.item() == 0.0 and zt.item() == 0.0, "empty-mask losses must be 0"
    print(f"aux OK: L_route={l_route.item():.4f} L_time={l_time.item():.4f}; empty-mask->0 ✓")

    # ---- CQL total + backward (gradients flow to q/route/time) ----
    out = {"Q": q, "route_scores": route_scores, "time_logits": time_logits}
    tgt_next = {"Q": q_tgt_next}
    batch = {"chosen_action_idx": chosen, "r_total": r_total, "done": done,
             "set_mask": set_mask, "time_bucket": time_bucket}
    total, p = L.cql_total(out, tgt_next, batch)
    total.backward()
    assert q.grad is not None and torch.isfinite(q.grad[vmask]).all(), "no grad to Q"
    assert route_scores.grad is not None and time_logits.grad is not None
    print(f"cql_total OK: L_total={p['L_total']:.4f}; grads flow to Q/route/time ✓")

    # ---- IQL ----
    qi, _ = make_q(requires_grad=True)
    qt, _ = make_q(requires_grad=False)
    v = torch.randn(B, requires_grad=True)
    v_next = torch.randn(B)
    # expectile asymmetry: positive diff weighted tau, negative weighted (1-tau)
    d = torch.tensor([1.0, -1.0])
    el = L.expectile_loss(d, tau=0.7)
    el_ref = (0.7 * 1.0 + 0.3 * 1.0) / 2
    assert abs(el.item() - el_ref) < 1e-6, f"expectile {el.item()} != {el_ref}"
    out_i = {"Q": qi, "route_scores": route_scores.detach().requires_grad_(True),
             "time_logits": time_logits.detach().requires_grad_(True)}
    tgt_i = {"Q": qt}
    li, pi = L.iql_total(out_i, tgt_i, batch, v=v, v_next=v_next)
    li.backward()
    assert torch.isfinite(li) and v.grad is not None
    print(f"IQL OK: expectile asym ✓; L_V={pi['L_V']:.4f} L_Q={pi['L_Q']:.4f} "
          f"L_pi={pi['L_pi']:.4f} L_total={pi['L_total']:.4f}")

    # ---- BC ----
    qb, _ = make_q(requires_grad=True)
    lbcq = L.bc_q_loss(qb, chosen)
    lbcq.backward()
    assert torch.isfinite(lbcq) and qb.grad is not None
    wait_logits = torch.randn(B, requires_grad=True)
    is_wait = (chosen == 0).float()
    rs2 = route_scores.detach().requires_grad_(True)
    lbc, pbc = L.bc_baseline_loss(rs2, wait_logits, chosen, is_wait, set_mask)
    lbc.backward()
    assert torch.isfinite(lbc) and wait_logits.grad is not None
    print(f"BC OK: L_BC_q={lbcq.item():.4f}; baseline L_route={pbc['L_route']:.4f} "
          f"L_wait={pbc['L_wait']:.4f}")

    # ---- soft_update (Polyak) ----
    onl = nn.Linear(4, 4); tgt = nn.Linear(4, 4)
    with torch.no_grad():
        for p_ in tgt.parameters(): p_.fill_(0.0)
        for p_ in onl.parameters(): p_.fill_(1.0)
    L.soft_update(tgt, onl, tau=0.005)
    w = next(tgt.parameters())
    assert torch.allclose(w, torch.full_like(w, 0.005)), "soft_update moved wrong amount"
    print("soft_update OK: target moved by tau=0.005 toward online ✓")

    print("\nPASS: all offline-RL losses (CQL/IQL/BC + aux) forward+backward, "
          "masks + edge-cases + hand-checks OK")


if __name__ == "__main__":
    main()
