"""Stage 4.4 smoke test — Transformer event encoder forward pass (synthetic).

SeqEncoder is standalone (consumes pre-gathered node_emb), so this needs only
torch (no loader/graph). Run on Windows:

    python scripts/train/04_smoke_sequence.py
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from railrl.encoders.sequence import SeqEncoder, D_MODEL


def main():
    import torch

    B, K = 4, 256
    enc = SeqEncoder.build()
    n_params = sum(p.numel() for p in enc.parameters())
    print(f"SeqEncoder params: {n_params:,}")

    node_emb = torch.randn(B, K, D_MODEL)
    state = torch.randint(0, 3, (B, K))               # 0=pad,1,2
    log_dt = torch.rand(B, K) * 10.0
    # variable valid lengths per row (rest padded); row 3 = fully padded (edge case)
    mask = torch.zeros(B, K)
    mask[0, :256] = 1; mask[1, :100] = 1; mask[2, :1] = 1; mask[3, :0] = 1

    enc.eval()
    with torch.no_grad():
        h_final, h_pool = enc(node_emb, state, log_dt, mask)

    print(f"h_seq_final: {tuple(h_final.shape)}  (expect ({B}, {D_MODEL}))")
    print(f"h_seq_pool : {tuple(h_pool.shape)}  (expect ({B}, {D_MODEL}))")
    assert h_final.shape == (B, D_MODEL) and h_pool.shape == (B, D_MODEL)
    assert torch.isfinite(h_final).all(), "h_seq_final non-finite"
    assert torch.isfinite(h_pool).all(), "h_seq_pool non-finite (fully-padded row?)"
    # fully-padded row (3) should give zero pooled (mask sum 0 → denom clamp, m=0)
    print(f"fully-padded row pooled norm: {float(h_pool[3].norm()):.4f} (expect ~0)")
    print("\nPASS: SeqEncoder forward OK (finite incl. fully-padded row)")


if __name__ == "__main__":
    main()
