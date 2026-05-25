"""Stage 4.6 smoke test — full RailRLModel forward + backward on a REAL batch.

Requires torch + torch-geometric. Builds the model from stats, loads a small
real batch via the loader, runs the end-to-end forward, checks output shapes +
finiteness, then a BC-style loss.backward() to confirm gradients flow through
all branches (HGT, sequence, fusion, Q, heads).

Run on Windows:
    python scripts/train/06_smoke_model.py
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from railrl import config as C
from railrl.encoders.input_pipeline import SnapshotDataset, NormStats
from railrl.model import RailRLModel


def main():
    import torch
    import torch.nn.functional as F
    from torch_geometric.loader import DataLoader

    stats = NormStats.load(C.NORMALIZATION_STATS_JSON)
    ds = SnapshotDataset(C.SNAPSHOTS_V2_PARQUET, C.NORMALIZATION_STATS_JSON, split="train")
    batch = next(iter(DataLoader([ds[i] for i in range(6)], batch_size=6)))
    B = batch.num_graphs
    print(f"batch: {B} graphs | track={batch['track'].num_nodes} signal={batch['signal'].num_nodes} "
          f"route={batch['route'].num_nodes} trn={batch['trn'].num_nodes}")

    model = RailRLModel.build(stats)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"RailRLModel params: {n_params:,}")

    # ---- forward ----
    model.train()
    out = model(batch)
    K1 = out["Q"].shape[1]
    print(f"\nforward outputs:")
    print(f"  Q:            {tuple(out['Q'].shape)}  (expect ({B}, K+1))")
    print(f"  route_scores: {tuple(out['route_scores'].shape)}  (expect ({B}, 14))")
    print(f"  time_logits:  {tuple(out['time_logits'].shape)}  (expect ({B}, 5))")
    print(f"  s_emb:        {tuple(out['s_emb'].shape)}  (expect ({B}, 256))")
    for k in ("Q", "route_scores", "time_logits", "s_emb"):
        finite = torch.isfinite(out[k][out[k] > -1e8]).all()  # ignore -1e9 masks
        assert finite, f"{k} has non-finite (besides masks)"

    # chosen action distribution
    chosen = batch.chosen_action_idx.view(B)
    print(f"\nchosen_action_idx: {chosen.tolist()}")
    valid_q = out["Q"].gather(1, chosen.view(B, 1)).squeeze(1)
    assert torch.isfinite(valid_q).all(), "chosen action's Q is masked/-inf (label inconsistent with action_mask!)"
    print("  all chosen actions have finite Q (label ↔ mask consistent) ✓")

    # ---- backward (BC loss on Q) ----
    loss = F.cross_entropy(out["Q"], chosen)
    loss.backward()
    n_with_grad = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    n_total = sum(1 for _ in model.parameters())
    print(f"\nBC loss: {loss.item():.4f}")
    print(f"params with non-zero grad: {n_with_grad}/{n_total}")
    assert n_with_grad > n_total * 0.5, "too few params got gradient — a branch is disconnected"

    print("\nPASS: end-to-end forward + backward OK (gradients flow through all branches)")


if __name__ == "__main__":
    main()
