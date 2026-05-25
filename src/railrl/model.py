"""spec 03 §8 / §11.3 — top-level RailRLModel.

Wires: HGT graph branch (§3) + Transformer sequence branch (§4) + fusion (§5)
+ per-action Q-network (§6) + route/time aux heads (§7).

The non-trivial part is THREE gathers from the batched PyG HeteroData (node
embeddings are concatenated across graphs; we use per-type `ptr` offsets):

  1. focal-train  : the is_focal=True train node per graph        → h_focal (B,128)
  2. candidate routes: act_route_idx (local route idx per graph)  → h_routes (B,14,128)
  3. event nodes  : ev_asset_idx (local idx into [track;signal])  → node_emb (B,K,128)

forward(data) → {'Q' (B,K+1), 'route_scores' (B,K), 'time_logits' (B,5), 's_emb'}.
"""
from __future__ import annotations

from .encoders.input_pipeline import NormStats
from .encoders.hgt import HGTEncoder, D_MODEL
from .encoders.sequence import SeqEncoder
from .encoders.fusion import ScheduleEncoder, Fusion
from .policies.q_network import QNetwork
from .policies.heads import RouteHead, TimeHead


def _torch():
    import torch
    import torch.nn as nn
    return torch, nn


def _num_graphs(data) -> int:
    ng = getattr(data, "num_graphs", None)
    if ng is not None:
        return int(ng)
    return int(data.n_candidates.shape[0])


def _ptr(store, num_graphs, device):
    """Per-type node offsets [0, n0, n0+n1, ...] (len B+1). Single graph → [0, N]."""
    import torch
    p = getattr(store, "ptr", None)
    if p is not None:
        return p
    n = store.num_nodes if getattr(store, "num_nodes", None) is not None else store.cont.size(0)
    return torch.tensor([0, n], device=device)


def gather_focal(h_trn, binary, batch, num_graphs):
    """is_focal=True train node per graph → (B, 128) (scatter by graph index)."""
    import torch
    is_focal = binary[:, 0] > 0.5
    out = torch.zeros(num_graphs, h_trn.size(1), device=h_trn.device, dtype=h_trn.dtype)
    if is_focal.any():
        out[batch[is_focal]] = h_trn[is_focal]
    return out


def gather_routes(h_route, route_ptr, act_route_idx):
    """act_route_idx (B,K) local route idx (-1 pad) → (B,K,128)."""
    import torch
    B, K = act_route_idx.shape
    valid = act_route_idx >= 0
    glob = route_ptr[:-1].view(B, 1) + act_route_idx                  # (B,K) (garbage where pad)
    glob = glob.clamp(min=0, max=max(0, h_route.size(0) - 1))
    g = h_route[glob.reshape(-1)].reshape(B, K, -1)                   # (B,K,128)
    return g * valid.unsqueeze(-1).to(g.dtype)


def gather_event_nodes(h_track, h_signal, track_ptr, signal_ptr, asset_idx, mask):
    """ev_asset_idx (B,K) local idx into [track;signal] per graph → (B,K,128)."""
    import torch
    B, K = asset_idx.shape
    n_track = (track_ptr[1:] - track_ptr[:-1]).view(B, 1)             # (B,1)
    is_track = asset_idx < n_track                                   # (B,K)
    tr_glob = (track_ptr[:-1].view(B, 1) + asset_idx).clamp(min=0, max=max(0, h_track.size(0) - 1))
    sg_glob = (signal_ptr[:-1].view(B, 1) + (asset_idx - n_track)).clamp(min=0, max=max(0, h_signal.size(0) - 1))
    tr = h_track[tr_glob.reshape(-1)].reshape(B, K, -1)
    sg = h_signal[sg_glob.reshape(-1)].reshape(B, K, -1)
    node_emb = torch.where(is_track.unsqueeze(-1), tr, sg)
    return node_emb * mask.unsqueeze(-1).to(node_emb.dtype)


class RailRLModel:
    @staticmethod
    def build(stats: NormStats):
        torch, nn = _torch()

        hgt = HGTEncoder.build(stats)
        seq = SeqEncoder.build()
        sched = ScheduleEncoder.build(stats)
        # fusion in_dim = h_graph_global 128 + h_focal 128 + h_seq_final 128
        #                 + h_seq_pool 128 + schedule_global(17) + special_flags 8 + n_cand 1
        fusion_in = D_MODEL * 4 + sched.out_dim + 8 + 1
        fusion = Fusion.build(in_dim=fusion_in)
        qnet = QNetwork.build()
        route_head = RouteHead.build()
        time_head = TimeHead.build()

        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.hgt = hgt
                self.seq = seq
                self.sched = sched
                self.fusion = fusion
                self.q = qnet
                self.route_head = route_head
                self.time_head = time_head

            def forward(self, data):
                B = _num_graphs(data)
                dev = data.n_candidates.device
                h_dict, pooled = self.hgt(data)               # PyG-keyed ('trn')

                track_ptr = _ptr(data["track"], B, dev)
                signal_ptr = _ptr(data["signal"], B, dev)
                route_ptr = _ptr(data["route"], B, dev)
                trn_batch = getattr(data["trn"], "batch", None)
                if trn_batch is None:
                    trn_batch = torch.zeros(h_dict["trn"].size(0), dtype=torch.long, device=dev)

                h_focal = gather_focal(h_dict["trn"], data["trn"].binary, trn_batch, B)
                h_routes = gather_routes(h_dict["route"], route_ptr, data.act_route_idx)
                node_emb = gather_event_nodes(h_dict["track"], h_dict["signal"],
                                              track_ptr, signal_ptr,
                                              data.ev_asset_idx, data.ev_mask)

                h_seq_final, h_seq_pool = self.seq(node_emb, data.ev_state,
                                                   data.ev_log_dt, data.ev_mask)
                sched_global = self.sched(data.ol_hc, data.ol_eta, data.ol_plat, data.ol_mask)
                n_cand = data.n_candidates.view(B)

                fusion_in = torch.cat([
                    pooled["global"], h_focal, h_seq_final, h_seq_pool,
                    sched_global, data.special_flags, n_cand.view(B, 1),
                ], dim=-1)
                s_emb = self.fusion(fusion_in)

                Q = self.q(h_focal, h_routes, s_emb, n_cand, data.act_mask, h_seq_final)
                route_scores = self.route_head(h_focal, h_routes, data.act_mask)
                time_logits = self.time_head(h_focal, s_emb)
                return {"Q": Q, "route_scores": route_scores,
                        "time_logits": time_logits, "s_emb": s_emb,
                        "h_focal": h_focal, "h_routes": h_routes}

        return _Model()
