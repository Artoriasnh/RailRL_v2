"""spec 03 §6 — per-action Q-network.

Q(s, a) for a dynamic action set A_t = {wait} ∪ {(focal_train, R_i)}.
  action_in_i = [h_train(128) ⊕ h_route_i(128) ⊕ s_emb(256) ⊕ is_in_cand(1) ⊕ n_cand(1)] = 514
  wait_in     = [h_train(128) ⊕ h_seq_final(128) ⊕ s_emb(256) ⊕ n_cand(1)]            = 513
  Q_all = [q_wait, q_a1, ..., q_aK]  ∈ (B, K+1);  masked actions → -1e9.
"""
from __future__ import annotations

D_MODEL = 128
S_EMB = 256
ACTION_IN = D_MODEL + D_MODEL + S_EMB + 1 + 1   # 514
WAIT_IN = D_MODEL + D_MODEL + S_EMB + 1          # 513


def _torch():
    import torch
    import torch.nn as nn
    return torch, nn


class QNetwork:
    @staticmethod
    def build(dropout: float = 0.1):
        torch, nn = _torch()

        class _Q(nn.Module):
            def __init__(self):
                super().__init__()
                self.mlp_action = nn.Sequential(
                    nn.Linear(ACTION_IN, 512), nn.GELU(), nn.Dropout(dropout),
                    nn.Linear(512, 256), nn.GELU(), nn.Dropout(dropout),
                    nn.Linear(256, 128), nn.GELU(),
                    nn.Linear(128, 1))
                self.mlp_wait = nn.Sequential(
                    nn.Linear(WAIT_IN, 256), nn.GELU(), nn.Dropout(dropout),
                    nn.Linear(256, 128), nn.GELU(),
                    nn.Linear(128, 1))

            def forward(self, h_train, h_routes, s_emb, n_cand, action_mask, h_seq_final):
                # h_train (B,128), h_routes (B,K,128), s_emb (B,256),
                # n_cand (B,), action_mask (B,K) {0,1}, h_seq_final (B,128)
                B, K, _ = h_routes.shape
                s_exp = s_emb.unsqueeze(1).expand(-1, K, -1)
                t_exp = h_train.unsqueeze(1).expand(-1, K, -1)
                nc_exp = n_cand.view(B, 1, 1).expand(-1, K, 1)
                ones = torch.ones(B, K, 1, device=h_train.device)
                action_in = torch.cat([t_exp, h_routes, s_exp, ones, nc_exp], dim=-1)  # (B,K,514)
                q_actions = self.mlp_action(action_in).squeeze(-1)                     # (B,K)
                q_actions = q_actions.masked_fill(action_mask < 0.5, -1e9)
                wait_in = torch.cat([h_train, h_seq_final, s_emb, n_cand.view(B, 1)], dim=-1)  # (B,513)
                q_wait = self.mlp_wait(wait_in).squeeze(-1)                            # (B,)
                return torch.cat([q_wait.unsqueeze(1), q_actions], dim=1)              # (B,K+1)

        return _Q()
