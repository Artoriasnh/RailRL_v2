"""Stage 4.5 smoke test — fusion + schedule encoder + Q-network + aux heads.

Synthetic tensors (no loader/graph needed). Run on Windows:
    python scripts/train/05_smoke_fusion_q.py
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from railrl import config as C
from railrl.encoders.input_pipeline import NormStats
from railrl.encoders.fusion import ScheduleEncoder, Fusion, SCHED_PER_TRAIN
from railrl.policies.q_network import QNetwork
from railrl.policies.heads import RouteHead, TimeHead, N_TIME_BUCKETS


def main():
    import torch

    stats = NormStats.load(C.NORMALIZATION_STATS_JSON)
    B, K = 4, 14   # batch, max candidates

    # --- schedule encoder ---
    sched = ScheduleEncoder.build(stats)
    hc = torch.randint(0, 3, (B, 5))
    eta = torch.rand(B, 5)
    plat = torch.zeros(B, 5, 8); plat[:, :, 3] = 1
    smask = torch.ones(B, 5); smask[3] = 0   # one row no outlook
    sched_global = sched(hc, eta, plat, smask)
    print(f"schedule_global: {tuple(sched_global.shape)} (expect (4, {SCHED_PER_TRAIN}))")
    assert sched_global.shape == (B, SCHED_PER_TRAIN)

    # --- fusion ---
    h_graph_global = torch.randn(B, 128)
    h_focal = torch.randn(B, 128)
    h_seq_final = torch.randn(B, 128)
    h_seq_pool = torch.randn(B, 128)
    flags = torch.randn(B, 8)
    n_cand = torch.tensor([1.0, 3.0, 5.0, 0.0])
    fusion_in = torch.cat([h_graph_global, h_focal, h_seq_final, h_seq_pool,
                           sched_global, flags, n_cand.unsqueeze(1)], dim=-1)
    in_dim = fusion_in.shape[1]
    print(f"fusion in_dim: {in_dim} (expect 128*4+{SCHED_PER_TRAIN}+8+1)")
    fusion = Fusion.build(in_dim=in_dim)
    s_emb = fusion(fusion_in)
    print(f"s_emb: {tuple(s_emb.shape)} (expect (4, 256))")
    assert s_emb.shape == (B, 256)

    # --- Q-network ---
    q = QNetwork.build()
    h_routes = torch.randn(B, K, 128)
    action_mask = torch.zeros(B, K)
    action_mask[0, :1] = 1; action_mask[1, :3] = 1; action_mask[2, :5] = 1  # row3 = wait-only
    Q = q(h_focal, h_routes, s_emb, n_cand, action_mask, h_seq_final)
    print(f"Q_all: {tuple(Q.shape)} (expect (4, {K+1}))")
    assert Q.shape == (B, K + 1)
    # masked actions should be -1e9
    print(f"row3 (wait-only) max action Q: {float(Q[3,1:].max()):.0f} (expect -1e9), wait Q finite: {torch.isfinite(Q[3,0]).item()}")
    assert (Q[3, 1:] < -1e8).all() and torch.isfinite(Q[3, 0])
    assert torch.isfinite(Q[:, 0]).all()                  # wait Q always finite
    assert torch.isfinite(Q[0, 1]) and torch.isfinite(Q[1, 1])  # valid actions finite

    # argmax action selection
    a = Q.argmax(dim=1)
    print(f"argmax actions: {a.tolist()}")

    # --- aux heads ---
    route_head = RouteHead.build()
    scores = route_head(h_focal, h_routes, action_mask)
    print(f"route scores: {tuple(scores.shape)} (expect (4, {K}))")
    assert scores.shape == (B, K)
    time_head = TimeHead.build()
    tlogits = time_head(h_focal, s_emb)
    print(f"time logits: {tuple(tlogits.shape)} (expect (4, {N_TIME_BUCKETS}))")
    assert tlogits.shape == (B, N_TIME_BUCKETS)

    print("\nPASS: fusion + schedule + Q-network + route/time heads all forward OK")


if __name__ == "__main__":
    main()
