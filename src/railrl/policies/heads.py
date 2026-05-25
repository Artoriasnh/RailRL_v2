"""spec 03 §7 — auxiliary supervised heads (priority head dropped, §7.3).

RouteHead : param-free dot-product over candidates (h_train · h_route_i),
            masked → softmax → CE vs (chosen_action_idx - 1) on set rows.
TimeHead  : 5-bucket lead-time classifier from [h_focal_train ⊕ s_emb].
            (spec called it "MDN" but the locked architecture is a 5-class head.)
"""
from __future__ import annotations

D_MODEL = 128
S_EMB = 256
N_TIME_BUCKETS = 5

# τ (lead-time) bucket edges in seconds — spec §7.2
TIME_BUCKET_EDGES = [5, 15, 30, 60]   # → buckets 0:≤5, 1:≤15, 2:≤30, 3:≤60, 4:>60


def time_bucket(tau_s: float) -> int:
    """Map a lead time (seconds) → bucket 0..4. NaN/None → -1 (excluded)."""
    import math
    if tau_s is None or (isinstance(tau_s, float) and math.isnan(tau_s)):
        return -1
    for i, e in enumerate(TIME_BUCKET_EDGES):
        if tau_s <= e:
            return i
    return N_TIME_BUCKETS - 1


def _torch():
    import torch
    import torch.nn as nn
    return torch, nn


class RouteHead:
    @staticmethod
    def build():
        torch, nn = _torch()

        class _Route(nn.Module):
            def forward(self, h_train, h_routes, action_mask):
                # h_train (B,128), h_routes (B,K,128), action_mask (B,K)
                scores = (h_train.unsqueeze(1) * h_routes).sum(-1)   # (B,K)
                return scores.masked_fill(action_mask < 0.5, -1e9)

        return _Route()


class TimeHead:
    @staticmethod
    def build(dropout: float = 0.1):
        torch, nn = _torch()

        class _Time(nn.Module):
            def __init__(self):
                super().__init__()
                self.mlp = nn.Sequential(
                    nn.Linear(D_MODEL + S_EMB, 128), nn.GELU(), nn.Dropout(dropout),
                    nn.Linear(128, N_TIME_BUCKETS))

            def forward(self, h_focal_train, s_emb):
                return self.mlp(torch.cat([h_focal_train, s_emb], dim=-1))   # (B,5)

        return _Time()
