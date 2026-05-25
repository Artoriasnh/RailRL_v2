"""spec 03 §4 — Sequence branch: Transformer over K=256 event tokens.

Token (per spec §2.3, adapted): each event references a subgraph node (by its
LOCAL index, as stored in state_event_tokens). Instead of a standalone global
asset embedding (spec §2.3 assumed a global asset_idx we don't store), we feed
the **HGT node embedding** of the referenced node — this ties the sequence
branch to the graph branch and is semantically consistent across snapshots.
The gather-by-local-index (with PyG batch offsets) happens in the top-level
model (§8); SeqEncoder receives the already-gathered `node_emb`.

  token = Linear([ node_emb(128) ⊕ state_emb(8) ⊕ time_emb(32) ] → 128)
  4-layer Transformer (4 heads, ff=512), padding-masked.
  outputs: h_seq_final (last unmasked token) + h_seq_pool (masked mean), each 128.

torch required; `build()` constructs the nn.Module lazily.
"""
from __future__ import annotations
import math

D_MODEL = 128
N_LAYERS = 4
N_HEADS = 4
FF_DIM = 512
DROPOUT = 0.1
STATE_VOCAB = 3      # 0=pad, 1=state0, 2=state1
STATE_EMB_DIM = 8
TIME_EMB_DIM = 32
NODE_EMB_DIM = 128   # gathered HGT node embedding


def _torch():
    import torch
    import torch.nn as nn
    return torch, nn


def sinusoidal_time(x, dim: int):
    """Sinusoidal encoding of a continuous scalar (here log1p(time_delta_s)).
    x: (B, K) float → returns (B, K, dim)."""
    import torch
    half = dim // 2
    device = x.device
    freqs = torch.exp(torch.arange(half, device=device, dtype=torch.float32)
                      * -(math.log(10000.0) / max(1, half)))
    ang = x.unsqueeze(-1) * freqs           # (B, K, half)
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)  # (B, K, 2*half)


class SeqEncoder:
    @staticmethod
    def build(n_layers: int = N_LAYERS, n_heads: int = N_HEADS,
              ff_dim: int = FF_DIM, dropout: float = DROPOUT):
        torch, nn = _torch()

        class _Seq(nn.Module):
            def __init__(self):
                super().__init__()
                self.state_emb = nn.Embedding(STATE_VOCAB, STATE_EMB_DIM, padding_idx=0)
                in_dim = NODE_EMB_DIM + STATE_EMB_DIM + TIME_EMB_DIM   # 128+8+32 = 168
                self.proj = nn.Linear(in_dim, D_MODEL)
                layer = nn.TransformerEncoderLayer(
                    d_model=D_MODEL, nhead=n_heads, dim_feedforward=ff_dim,
                    dropout=dropout, batch_first=True, activation="gelu")
                self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)

            def forward(self, node_emb, state, log_dt, mask):
                # node_emb (B,K,128), state (B,K) long, log_dt (B,K) float, mask (B,K) {0,1}
                state_e = self.state_emb(state)                  # (B,K,8)
                time_e = sinusoidal_time(log_dt, TIME_EMB_DIM)   # (B,K,32)
                tok = self.proj(torch.cat([node_emb, state_e, time_e], dim=-1))  # (B,K,128)
                pad = mask < 0.5                                 # True = padding
                # A fully-padded row would make softmax NaN; guard by forcing at
                # least one valid slot (it's masked out of the pooled output anyway).
                all_pad = pad.all(dim=1)
                if all_pad.any():
                    pad = pad.clone()
                    pad[all_pad, 0] = False
                H = self.transformer(tok, src_key_padding_mask=pad)   # (B,K,128)

                m = mask.unsqueeze(-1)                            # (B,K,1)
                denom = m.sum(dim=1).clamp(min=1.0)              # (B,1)
                h_seq_pool = (H * m).sum(dim=1) / denom          # (B,128)
                # last unmasked token index = (#valid - 1), clamped to 0
                last_idx = (mask.sum(dim=1).long() - 1).clamp(min=0)  # (B,)
                h_seq_final = H[torch.arange(H.size(0), device=H.device), last_idx]  # (B,128)
                return h_seq_final, h_seq_pool

        return _Seq()
