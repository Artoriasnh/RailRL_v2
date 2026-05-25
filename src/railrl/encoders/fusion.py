"""spec 03 §5 — Fusion + schedule-outlook encoder.

ScheduleEncoder: per upcoming train [hc_emb(8) ⊕ eta(1) ⊕ platform_onehot(8)]
                 → masked mean over 5 → schedule_global (17-d).
                 (spec said 16-d w/ platform 1-6 + None = 7; we have platform
                  1-7 → 8-way one-hot, so 17.)

Fusion: concat([h_graph_global 128, h_focal_train 128, h_seq_final 128,
                h_seq_pool 128, schedule_global 17, special_flags 8,
                n_candidates 1]) → LN → 512 → LN → s_emb(256).  (in_dim 538;
spec's "657" arithmetic was loose — we compute in_dim from actual components.)
"""
from __future__ import annotations

from .input_pipeline import NormStats
from .sequence import sinusoidal_time  # noqa: F401 (re-export convenience)

SCHED_HC_EMB = 8
SCHED_PLATFORM = 8          # 1-7 + None one-hot
SCHED_PER_TRAIN = SCHED_HC_EMB + 1 + SCHED_PLATFORM   # 17
S_EMB = 256


def _torch():
    import torch
    import torch.nn as nn
    return torch, nn


class ScheduleEncoder:
    @staticmethod
    def build(stats: NormStats):
        torch, nn = _torch()
        hc_vocab = stats.vocab_size("train", "headcode_class")

        class _Sched(nn.Module):
            def __init__(self):
                super().__init__()
                self.hc_emb = nn.Embedding(hc_vocab, SCHED_HC_EMB, padding_idx=0)
                self.out_dim = SCHED_PER_TRAIN

            def forward(self, hc, eta, platform, mask):
                # hc (B,5) long, eta (B,5) float, platform (B,5,8), mask (B,5)
                he = self.hc_emb(hc)                       # (B,5,8)
                per = torch.cat([he, eta.unsqueeze(-1), platform], dim=-1)  # (B,5,17)
                m = mask.unsqueeze(-1)                     # (B,5,1)
                denom = m.sum(dim=1).clamp(min=1.0)        # (B,1)
                return (per * m).sum(dim=1) / denom        # (B,17)

        return _Sched()


class Fusion:
    @staticmethod
    def build(in_dim: int, out_dim: int = S_EMB, dropout: float = 0.1):
        torch, nn = _torch()
        import torch.nn.functional as F

        class _Fusion(nn.Module):
            def __init__(self):
                super().__init__()
                self.ln1 = nn.LayerNorm(in_dim)
                self.fc1 = nn.Linear(in_dim, 512)
                self.ln2 = nn.LayerNorm(out_dim)
                self.fc2 = nn.Linear(512, out_dim)
                self.drop = nn.Dropout(dropout)

            def forward(self, x):
                x = self.ln1(x)
                x = F.gelu(self.fc1(x))
                x = self.drop(x)
                x = self.fc2(x)
                return self.ln2(x)            # s_emb (B, 256)

        return _Fusion()
