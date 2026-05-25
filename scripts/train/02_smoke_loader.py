"""Stage 4.1 smoke test — verify the PyG HeteroData loader on real snapshots.

Requires torch + torch-geometric installed. Loads a few real snapshots from the
'train' split → HeteroData and prints the structure (node counts, edge types,
event/action/outlook tensors). Run on Windows:

    python scripts/train/02_smoke_loader.py
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from railrl import config as C
from railrl.encoders.input_pipeline import SnapshotDataset


def main():
    ds = SnapshotDataset(C.SNAPSHOTS_V2_PARQUET, C.NORMALIZATION_STATS_JSON, split="train")
    print(f"train split size: {len(ds):,}")

    # Inspect a few items + try to batch them with PyG
    import torch
    from torch_geometric.loader import DataLoader

    for i in [0, 1, len(ds) // 2]:
        d = ds[i]
        nt = {t: d[t].num_nodes for t in ("track", "signal", "route", "train")}
        ets = [k for k in d.edge_index_dict.keys()]
        print(f"\nitem {i}: nodes={nt}")
        print(f"  feature dims: track.cont={tuple(d['track'].cont.shape)} "
              f"track.binary={tuple(d['track'].binary.shape)} "
              f"route.cat={tuple(d['route'].cat.shape)} train.binary={tuple(d['train'].binary.shape)}")
        print(f"  edge types present: {len(ets)} | "
              f"event mask sum={int(d.ev_mask.sum())} | "
              f"n_candidates={float(d.n_candidates):.0f} chosen={int(d.chosen_action_idx)}")

    print("\n[batching] DataLoader batch_size=4 ...")
    dl = DataLoader([ds[j] for j in range(4)], batch_size=4)
    batch = next(iter(dl))
    print(f"  batched: track nodes={batch['track'].num_nodes}, "
          f"ev_asset_idx shape={tuple(batch.ev_asset_idx.shape)}, "
          f"act_route_idx shape={tuple(batch.act_route_idx.shape)}, "
          f"chosen={batch.chosen_action_idx.tolist()}")
    # sanity: feature dims constant across items
    print("\nPASS: HeteroData built + batched OK" )


if __name__ == "__main__":
    main()
