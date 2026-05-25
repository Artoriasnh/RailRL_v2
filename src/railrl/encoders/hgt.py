"""spec 03 §3 — HGT graph encoder.

Pipeline:
  NodeInit  : per node type, [identity_emb ⊕ categorical_embs ⊕ cont ⊕ binary]
              → Linear → d_model (128). Embedding sizes come from the
              normalization-stats vocab (NOT spec's hardcoded estimates).
  HGTConv×3 : PyG heterogeneous graph attention (4 heads, 8 edge types).
  Pooling   : per-type + global mean pool (masked via PyG batch vector).

Output (forward):
  h_dict  : {ntype: (N_ntype_total, 128)}   per-node embeddings (batched)
  pooled  : {ntype: (B, 128), 'global': (B, 128)}

Downstream (Q-net §6) gathers h_focal_train from h_dict['train'] (is_focal flag)
and h_routes from h_dict['route'] (act_route_idx).

torch + torch_geometric required. The config builder `node_init_config(stats)`
is torch-free (sandbox-testable).
"""
from __future__ import annotations

from .input_pipeline import (CONT, BINARY, CAT, PLATFORM, IDENT,
                             N_PLATFORM_SLOTS, NormStats, PYG_NODE_KEY)

# Logical node types (= taxonomy / schema keys). PyG sees PYG_NODE_KEY[nt]
# ('train' → 'trn') because PyG reserves the node-type name 'train'.
NODE_TYPES = ["track", "signal", "route", "train"]
PYG_TYPES = [PYG_NODE_KEY[nt] for nt in NODE_TYPES]   # ['track','signal','route','trn']
# (logical_src, rel, logical_dst) → mapped to PyG keys for HGTConv metadata
_LOGICAL_EDGES = [
    ("track", "connects", "track"),
    ("route", "traverses", "track"),
    ("route", "starts_at", "signal"),
    ("route", "ends_at", "signal"),
    ("signal", "protects", "track"),
    ("route", "same_signal", "route"),
    ("train", "at_berth", "track"),
    ("train", "next_signal", "signal"),
]
EDGE_METADATA = [(PYG_NODE_KEY[s], rel, PYG_NODE_KEY[d]) for (s, rel, d) in _LOGICAL_EDGES]
IDENT_DIM = {"track": 64, "signal": 64, "route": 64, "train": 32}  # spec §3.1
CAT_EMB_DIM = 8       # spec §2.2 (categorical → 8-d learned embedding)
D_MODEL = 128


def node_init_config(stats: NormStats) -> dict:
    """Per-type dims for NodeInit (torch-free; unit-testable).

    Returns {ntype: {cont, binary, cat:[vocab...], ident_vocab, ident_dim, in_dim}}.
    """
    cfg = {}
    for nt in NODE_TYPES:
        cont_dim = len(CONT[nt])
        binary_dim = len(BINARY[nt]) + N_PLATFORM_SLOTS * len(PLATFORM[nt])
        cat_vocabs = [stats.vocab_size(nt, f) for f in CAT[nt]]
        ident_vocab = stats.vocab_size(nt, IDENT[nt])
        ident_dim = IDENT_DIM[nt]
        in_dim = ident_dim + CAT_EMB_DIM * len(cat_vocabs) + cont_dim + binary_dim
        cfg[nt] = {"cont": cont_dim, "binary": binary_dim, "cat": cat_vocabs,
                   "ident_vocab": ident_vocab, "ident_dim": ident_dim, "in_dim": in_dim}
    return cfg


# ============================================================
# torch modules (import torch lazily so config builder stays torch-free)
# ============================================================

def _torch():
    import torch  # noqa
    import torch.nn as nn  # noqa
    return torch, nn


class NodeInit:
    """Factory returning an nn.Module (defined at call time to avoid importing
    torch at module import). Use `NodeInit.build(cfg)`.
    """

    @staticmethod
    def build(cfg: dict):
        torch, nn = _torch()

        class _NodeInit(nn.Module):
            def __init__(self, cfg):
                super().__init__()
                self.ident = nn.ModuleDict()
                self.cats = nn.ModuleDict()
                self.proj = nn.ModuleDict()
                # ModuleDict keys must avoid nn.Module reserved names ('train'!),
                # so prefix every node-type key.
                for nt, c in cfg.items():
                    k = "nt_" + nt
                    self.ident[k] = nn.Embedding(c["ident_vocab"], c["ident_dim"], padding_idx=0)
                    self.cats[k] = nn.ModuleList(
                        [nn.Embedding(v, CAT_EMB_DIM, padding_idx=0) for v in c["cat"]])
                    self.proj[k] = nn.Sequential(
                        nn.Linear(c["in_dim"], D_MODEL), nn.GELU(),
                        nn.LayerNorm(D_MODEL))

            def forward(self, data):
                # x_dict is keyed by PyG node-type ('trn'); ModuleDicts by 'nt_'+logical
                x_dict = {}
                for nt in cfg:                      # logical type
                    k = "nt_" + nt
                    store = data[PYG_NODE_KEY[nt]]  # PyG store
                    parts = [self.ident[k](store.ident)]
                    for j, emb in enumerate(self.cats[k]):
                        parts.append(emb(store.cat[:, j]))
                    parts.append(store.cont)
                    parts.append(store.binary)
                    x = torch.cat(parts, dim=-1)
                    x_dict[PYG_NODE_KEY[nt]] = self.proj[k](x)
                return x_dict

        return _NodeInit(cfg)


class HGTEncoder:
    """Factory for the full HGT encoder nn.Module. Use `HGTEncoder.build(stats)`."""

    @staticmethod
    def build(stats: NormStats, n_layers: int = 3, n_heads: int = 4, dropout: float = 0.1):
        torch, nn = _torch()
        from torch_geometric.nn import HGTConv
        from torch_geometric.utils import scatter

        cfg = node_init_config(stats)
        # metadata node-types MUST be the PyG keys ('trn', not 'train') — HGTConv
        # builds an internal ModuleDict keyed by metadata[0].
        metadata = (PYG_TYPES, EDGE_METADATA)

        class _HGT(nn.Module):
            def __init__(self):
                super().__init__()
                self.node_init = NodeInit.build(cfg)
                self.layers = nn.ModuleList([
                    HGTConv(D_MODEL, D_MODEL, metadata, heads=n_heads)
                    for _ in range(n_layers)])
                self.norm = nn.ModuleDict({"nt_" + nt: nn.LayerNorm(D_MODEL) for nt in NODE_TYPES})
                self.drop = nn.Dropout(dropout)

            def forward(self, data):
                # x_dict / pooled / h_dict are keyed by PyG node-type ('trn').
                x_dict = self.node_init(data)
                edge_index_dict = data.edge_index_dict
                for layer in self.layers:
                    out = layer(x_dict, edge_index_dict)
                    # residual + norm + dropout per type (HGTConv may drop types
                    # with no incoming edges → keep previous repr for those)
                    new_x = {}
                    for nt in NODE_TYPES:                 # logical
                        pk = PYG_NODE_KEY[nt]
                        prev = x_dict[pk]
                        cur = out.get(pk, prev)
                        new_x[pk] = self.norm["nt_" + nt](self.drop(cur) + prev)
                    x_dict = new_x
                # TRUE batch size — must be the SAME for every node type. Using a
                # per-type b.max()+1 truncates the pooled tensor when the LAST graph(s)
                # in the batch have no nodes of that type (→ [95,128] vs [96,128] →
                # stack/fusion size mismatch). scatter with dim_size=B emits B rows
                # (zeros for graphs with no nodes of that type).
                B = getattr(data, "num_graphs", None)
                if B is None:
                    B = 1
                    for nt in NODE_TYPES:
                        bb = getattr(data[PYG_NODE_KEY[nt]], "batch", None)
                        if bb is not None and bb.numel() > 0:
                            B = max(B, int(bb.max().item()) + 1)
                pooled = {}
                feats = []
                for nt in NODE_TYPES:
                    pk = PYG_NODE_KEY[nt]
                    b = getattr(data[pk], "batch", None)
                    if b is None:  # single graph
                        b = torch.zeros(x_dict[pk].size(0), dtype=torch.long,
                                        device=x_dict[pk].device)
                    p = scatter(x_dict[pk], b, dim=0, dim_size=B, reduce="mean")
                    pooled[pk] = p
                    feats.append(p)
                pooled["global"] = torch.stack(feats, dim=0).mean(dim=0)
                return x_dict, pooled

        return _HGT()
