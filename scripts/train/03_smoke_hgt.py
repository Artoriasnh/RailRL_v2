"""Stage 4.3 smoke test — HGT encoder forward pass on a real batch.

Requires torch + torch-geometric. Builds the encoder from normalization stats,
loads a small batch via the loader, runs forward, and checks output shapes:
  h_dict[ntype] : (N_ntype_total, 128)
  pooled[ntype] : (B, 128) ;  pooled['global'] : (B, 128)

Run on Windows:
    python scripts/train/03_smoke_hgt.py
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from railrl import config as C
from railrl.encoders.input_pipeline import SnapshotDataset, NormStats
from railrl.encoders.hgt import HGTEncoder, node_init_config, D_MODEL


def main():
    import torch
    from torch_geometric.loader import DataLoader

    stats = NormStats.load(C.NORMALIZATION_STATS_JSON)
    print("node_init_config (in_dim per type):")
    for nt, c in node_init_config(stats).items():
        print(f"  {nt}: in_dim={c['in_dim']} ident_vocab={c['ident_vocab']}")

    ds = SnapshotDataset(C.SNAPSHOTS_V2_PARQUET, C.NORMALIZATION_STATS_JSON, split="train")
    items = [ds[i] for i in range(4)]
    batch = next(iter(DataLoader(items, batch_size=4)))

    enc = HGTEncoder.build(stats)
    n_params = sum(p.numel() for p in enc.parameters())
    print(f"\nHGT encoder params: {n_params:,}")

    enc.eval()
    with torch.no_grad():
        h_dict, pooled = enc(batch)

    # NOTE: 'train' node type is keyed 'trn' in PyG (see PYG_NODE_KEY).
    print("\nper-node embeddings:")
    for nt in ("track", "signal", "route", "trn"):
        print(f"  h_dict[{nt}]: {tuple(h_dict[nt].shape)}  (expect (N_{nt}, {D_MODEL}))")
        assert h_dict[nt].shape[1] == D_MODEL
        assert torch.isfinite(h_dict[nt]).all(), f"{nt} has non-finite values"
    print("pooled:")
    for nt in ("track", "signal", "route", "trn", "global"):
        print(f"  pooled[{nt}]: {tuple(pooled[nt].shape)}  (expect (4, {D_MODEL}))")
        assert pooled[nt].shape == (4, D_MODEL)
        assert torch.isfinite(pooled[nt]).all()

    # gather focal-train embedding (is_focal flag = binary[:,0]) per graph — this
    # is what the Q-net will do (sanity that the index plumbing works)
    tr = batch["trn"]
    is_focal = tr.binary[:, 0] > 0.5
    print(f"\nfocal train nodes in batch: {int(is_focal.sum())} (expect 4, one per graph)")

    print("\nPASS: HGT encoder forward OK (finite, correct shapes)")


if __name__ == "__main__":
    main()
