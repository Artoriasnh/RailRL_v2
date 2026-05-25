"""spec 03 §2 — Snapshot → model tensors (input pipeline).

Two layers:
  1. `encode_snapshot(row, stats)` — PURE NUMPY, torch-free, unit-testable.
     Turns one snapshots_v2 row (dict) into numpy arrays + index maps.
  2. `SnapshotDataset` / `to_heterodata` — thin torch + PyG wrappers (lazy
     import) that wrap the numpy output into a PyG `HeteroData`.

FEATURE LAYOUT (this module is the source of truth; spec 03 §3.1 left the exact
per-field handling open). Per node type each node yields 4 parallel arrays:
  - cont   : z-scored continuous features (clip ±5)            float32
  - binary : 0/1 flags                                         float32
  - cat    : categorical vocab indices (0 = pad/unknown)       int64
  - ident  : identity vocab index (track/signal/route/train)   int64
The HGT node-init MLP (§3.1) consumes [identity_emb ⊕ cat_embs ⊕ cont ⊕ binary].

Nullable platform fields (platform_id / end_platform_id / current_platform /
planned_platform ∈ {1..7, None}) → fixed 8-way one-hot (index 0 = None), no
vocab needed. They are appended to the `binary` block.

Caps (spec 03 §2.1) are already enforced at snapshot-build time, so per-row node
counts are ≤ cap; we still pad to cap for fixed-shape batching of the action /
event-token / outlook tensors (graph nodes use PyG variable-size batching).
"""
from __future__ import annotations
import json
import math
from pathlib import Path
from typing import Optional

import numpy as np

# ============================================================
# Feature taxonomy (must match scripts/train/01_build_normalization_stats.py)
# ============================================================

CONT = {
    "track": ["n_routes_using",
              *[f"occupancy_fraction_{w}m" for w in (1, 5, 10, 15, 30)],
              *[f"n_state_changes_{w}m" for w in (1, 5, 10, 15, 30)],
              "last_change_age_s"],
    "signal": ["n_routes_from",
               *[f"aspect_fraction_red_{w}m" for w in (1, 5, 10, 15, 30)],
               *[f"aspect_n_changes_{w}m" for w in (1, 5, 10, 15, 30)],
               "aspect_last_change_age_s", "berth_dwell_age_s"],
    "route": ["n_tc", "length_m", "ave_speed_mps", "ave_grad", "gap_time_s",
              "n_points", "last_locked_age_s", "n_tcs_occupied_by_other",
              "n_tcs_occupied_by_focal", "max_relative_position_of_occupied",
              "min_age_of_occupation_s"],
    "train": ["time_in_current_berth_s", "scheduled_delta_s",
              "recent_panel_requests_count"],
}
BINARY = {
    "track": ["occupied_now", "on_focal_train_path"],
    "signal": ["is_platform_end", "aspect_restrictive_now"],
    "route": ["currently_locked", "in_candidate_set"],
    "train": ["is_focal"],
}
CAT = {
    "track": ["platform_sub"],
    "signal": ["prefix", "platform_direction"],
    "route": ["prefix", "signal_no", "letter", "sub", "cls"],
    "train": ["headcode_class"],
}
# nullable platform-int fields → fixed 8-way one-hot (0=None, 1..7)
PLATFORM = {
    "track": ["platform_id"],
    "signal": ["platform_id"],
    "route": ["end_platform_id"],
    "train": ["current_platform", "planned_platform"],
}
IDENT = {"track": "track_id", "signal": "signal_id",
         "route": "route_id", "train": "train_id"}
NODE_COL = {"track": "state_nodes_track", "signal": "state_nodes_signal",
            "route": "state_nodes_route", "train": "state_nodes_train"}
# PyG node-type key. 'train' is renamed to 'trn' because PyG's HGTConv
# (HeteroDictLinear) keys an internal ModuleDict by node-type name, and
# nn.Module reserves 'train' (the .train() method) → KeyError. Schema/taxonomy
# keys stay 'train'; only the PyG HeteroData/metadata node-type string changes.
PYG_NODE_KEY = {"track": "track", "signal": "signal", "route": "route", "train": "trn"}
EDGE_TYPES = ["connects", "traverses", "starts_at", "ends_at",
              "protects", "same_signal", "at_berth", "next_signal"]
N_PLATFORM_SLOTS = 8   # index 0 = None, 1..7 = platform


# ============================================================
# Stats loader
# ============================================================

class NormStats:
    """Wraps normalization_stats.json for fast feature lookup."""

    def __init__(self, stats: dict):
        self.continuous = stats["continuous"]
        self.vocab = stats["vocab"]
        self.caps = stats["caps"]

    @classmethod
    def load(cls, path) -> "NormStats":
        with open(path) as f:
            return cls(json.load(f))

    def zscore(self, ntype: str, feat: str, v) -> float:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return 0.0
        s = self.continuous.get(f"{ntype}.{feat}")
        if not s:
            return float(v)
        z = (float(v) - s["mean"]) / (s["std"] or 1.0)
        return float(max(-5.0, min(5.0, z)))

    def cat_index(self, ntype: str, field: str, v) -> int:
        if v is None:
            return 0
        entry = self.vocab.get(f"{ntype}.{field}")
        if not entry:
            return 0
        return int(entry["index"].get(str(v), 0))

    def vocab_size(self, ntype: str, field: str) -> int:
        entry = self.vocab.get(f"{ntype}.{field}")
        return int(entry["size"]) if entry else 1


def _platform_onehot(v) -> list[float]:
    oh = [0.0] * N_PLATFORM_SLOTS
    try:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            oh[0] = 1.0
        else:
            iv = int(v)
            oh[iv if 1 <= iv <= 7 else 0] = 1.0
    except (TypeError, ValueError):
        oh[0] = 1.0
    return oh


# ============================================================
# Per-node-type feature encoding (numpy)
# ============================================================

def encode_nodes(nodes: list, ntype: str, stats: NormStats) -> dict:
    """Encode a list of node dicts → parallel numpy arrays.

    Returns dict with cont (n,Fc), binary (n,Fb+8*nplat), cat (n,Fk), ident (n,).
    n may be 0 (empty subgraph slice) — arrays still have the right 2nd dim.
    """
    cont_f = CONT[ntype]; bin_f = BINARY[ntype]; cat_f = CAT[ntype]
    plat_f = PLATFORM[ntype]; ident_f = IDENT[ntype]
    n = len(nodes)
    Fc, Fb, Fk = len(cont_f), len(bin_f), len(cat_f)
    cont = np.zeros((n, Fc), dtype=np.float32)
    binary = np.zeros((n, Fb + N_PLATFORM_SLOTS * len(plat_f)), dtype=np.float32)
    cat = np.zeros((n, Fk), dtype=np.int64)
    ident = np.zeros((n,), dtype=np.int64)
    for i, node in enumerate(nodes):
        for j, f in enumerate(cont_f):
            cont[i, j] = stats.zscore(ntype, f, node.get(f))
        for j, f in enumerate(bin_f):
            binary[i, j] = 1.0 if node.get(f) else 0.0
        off = Fb
        for f in plat_f:
            binary[i, off:off + N_PLATFORM_SLOTS] = _platform_onehot(node.get(f))
            off += N_PLATFORM_SLOTS
        for j, f in enumerate(cat_f):
            cat[i, j] = stats.cat_index(ntype, f, node.get(f))
        ident[i] = stats.cat_index(ntype, ident_f, node.get(ident_f))
    return {"cont": cont, "binary": binary, "cat": cat, "ident": ident,
            "ids": [str(node.get(ident_f)) for node in nodes]}


# ============================================================
# Edges → node-local index pairs
# ============================================================

# (src_node_type, dst_node_type) per edge type — must match schema/_format_edges
_EDGE_ENDPOINTS = {
    "connects":   ("track", "track"),
    "traverses":  ("route", "track"),
    "starts_at":  ("route", "signal"),
    "ends_at":    ("route", "signal"),
    "protects":   ("signal", "track"),
    "same_signal":("route", "route"),
    "at_berth":   ("train", "track"),
    "next_signal":("train", "signal"),
}


def encode_edges(row, id_index: dict) -> dict:
    """Map each edge's (src,dst) string ids → local node indices.

    `id_index[ntype]` maps node_id → local index (from the node arrays' `ids`).
    Returns {edge_type: int64 array shape (2, n_edges)} (node-local indices).
    Edges referencing a node not in the (capped) subgraph are dropped.
    """
    out = {}
    for et in EDGE_TYPES:
        st, dt = _EDGE_ENDPOINTS[et]
        src_map, dst_map = id_index[st], id_index[dt]
        pairs = []
        for e in (row.get(f"state_edges_{et}") or []):
            s = src_map.get(str(e["src"]))
            d = dst_map.get(str(e["dst"]))
            if s is not None and d is not None:
                pairs.append((s, d))
        out[et] = (np.array(pairs, dtype=np.int64).T
                   if pairs else np.zeros((2, 0), dtype=np.int64))
    return out


# ============================================================
# Event tokens + schedule outlook + action set
# ============================================================

def encode_event_tokens(row, K: int) -> dict:
    """K×(asset_idx, state, time_delta_s). asset_idx is the LOCAL index into the
    snapshot's [track ⊕ signal] node ordering (as built in state.py). Pad to K."""
    toks = row.get("state_event_tokens") or []
    aidx = np.zeros((K,), dtype=np.int64)
    state = np.zeros((K,), dtype=np.int64)      # 0 pad; real state stored +1
    logdt = np.zeros((K,), dtype=np.float32)
    mask = np.zeros((K,), dtype=np.float32)
    for i, t in enumerate(toks[:K]):
        aidx[i] = int(t["asset_idx"])
        state[i] = int(t["state"]) + 1          # {0,1} → {1,2}; 0 = pad
        logdt[i] = math.log1p(max(0.0, float(t["time_delta_s"])))
        mask[i] = 1.0
    return {"asset_idx": aidx, "state": state, "log_dt": logdt, "mask": mask}


def encode_schedule_outlook(row, stats: NormStats, k: int = 5) -> dict:
    """k×(headcode_class_idx, eta_log_z, platform_onehot[8]). Pad with zeros."""
    rows = row.get("state_schedule_outlook") or []
    hc = np.zeros((k,), dtype=np.int64)
    eta = np.zeros((k,), dtype=np.float32)
    plat = np.zeros((k, N_PLATFORM_SLOTS), dtype=np.float32)
    mask = np.zeros((k,), dtype=np.float32)
    for i, r in enumerate(rows[:k]):
        hc[i] = stats.cat_index("train", "headcode_class", r.get("headcode_class"))
        eta[i] = math.log1p(max(0.0, float(r.get("eta_s", 0))))
        plat[i] = _platform_onehot(r.get("planned_platform"))
        mask[i] = 1.0
    return {"headcode_class": hc, "eta_log": eta, "platform": plat, "mask": mask}


FLAG_KEYS = ["f_advance", "f_call_on", "f_platform_dev", "f_priority_compete",
             "f_late_train", "f_unusual_id", "f_trts_pressed", "f_freight_class"]


def encode_special_flags(row) -> "np.ndarray":
    """8 special flags → float vector. Bools → 0/1; f_late_train (seconds late)
    → /600 clipped ±5 so it shares scale with the others under fusion LayerNorm."""
    sf = row.get("state_special_flags") or {}
    out = []
    for k in FLAG_KEYS:
        v = sf.get(k, 0)
        if k == "f_late_train":
            try:
                out.append(max(-5.0, min(5.0, float(v or 0) / 600.0)))
            except (TypeError, ValueError):
                out.append(0.0)
        else:
            out.append(1.0 if v else 0.0)
    return np.array(out, dtype=np.float32)


def encode_actions(row, route_id_index: dict, max_cand: int) -> dict:
    """Candidate routes → their local route-node indices (for h_routes gather).

    action 0 = wait; actions 1..K map to candidate_route_ids[0..K-1]. Returns:
      route_local_idx (max_cand,) int64  — route-node index of each candidate (-1 pad)
      action_mask     (max_cand,) float  — 1 for real candidate
      chosen_action_idx scalar           — label (0=wait, 1..K)
    """
    cands = [str(r) for r in (row.get("candidate_route_ids") or [])]
    idx = np.full((max_cand,), -1, dtype=np.int64)
    mask = np.zeros((max_cand,), dtype=np.float32)
    for i, rid in enumerate(cands[:max_cand]):
        li = route_id_index.get(rid)
        idx[i] = li if li is not None else -1
        mask[i] = 1.0
    chosen = int(row.get("chosen_action_idx", -1))
    return {"route_local_idx": idx, "action_mask": mask,
            "n_candidates": float(len(cands)), "chosen_action_idx": chosen}


# ============================================================
# Top-level: encode one snapshot row (pure numpy)
# ============================================================

def encode_snapshot(row: dict, stats: NormStats) -> dict:
    """One snapshots_v2 row (dict) → encoded numpy structure for the model."""
    nodes_enc = {}
    id_index = {}
    for ntype, col in NODE_COL.items():
        nodes = list(row.get(col) or [])
        enc = encode_nodes(nodes, ntype, stats)
        nodes_enc[ntype] = enc
        id_index[ntype] = {nid: i for i, nid in enumerate(enc["ids"])}
    edges = encode_edges(row, id_index)
    K = int(stats.caps.get("event_tokens", 256))
    max_cand = int(stats.caps.get("candidates", 14))
    events = encode_event_tokens(row, K)
    outlook = encode_schedule_outlook(row, stats)
    actions = encode_actions(row, id_index["route"], max_cand)
    return {
        "nodes": nodes_enc,
        "edges": edges,
        "events": events,
        "outlook": outlook,
        "actions": actions,
        "special_flags": encode_special_flags(row),
        "label": str(row.get("label", "")),
        "sample_id": int(row.get("sample_id", -1)),
        "pass_id": str(row.get("pass_id", "")),
        # reward fields (NaN now; joined from decision_rewards downstream)
        "r_total": float(row["r_total"]) if row.get("r_total") is not None
                   and not (isinstance(row.get("r_total"), float) and math.isnan(row["r_total"]))
                   else float("nan"),
    }


# ============================================================
# Torch / PyG wrappers (lazy import — keeps the core torch-free)
# ============================================================

def to_heterodata(enc: dict):
    """Build a PyG HeteroData from `encode_snapshot` output. torch + PyG required."""
    import torch
    from torch_geometric.data import HeteroData

    data = HeteroData()
    for ntype, e in enc["nodes"].items():
        k = PYG_NODE_KEY[ntype]
        data[k].cont = torch.from_numpy(e["cont"])
        data[k].binary = torch.from_numpy(e["binary"])
        data[k].cat = torch.from_numpy(e["cat"])
        data[k].ident = torch.from_numpy(e["ident"])
        data[k].num_nodes = e["cont"].shape[0]
    for et, ei in enc["edges"].items():
        st, dt = _EDGE_ENDPOINTS[et]
        data[PYG_NODE_KEY[st], et, PYG_NODE_KEY[dt]].edge_index = torch.from_numpy(ei)
    ev = enc["events"]
    data.ev_asset_idx = torch.from_numpy(ev["asset_idx"]).unsqueeze(0)
    data.ev_state = torch.from_numpy(ev["state"]).unsqueeze(0)
    data.ev_log_dt = torch.from_numpy(ev["log_dt"]).unsqueeze(0)
    data.ev_mask = torch.from_numpy(ev["mask"]).unsqueeze(0)
    ol = enc["outlook"]
    data.ol_hc = torch.from_numpy(ol["headcode_class"]).unsqueeze(0)
    data.ol_eta = torch.from_numpy(ol["eta_log"]).unsqueeze(0)
    data.ol_plat = torch.from_numpy(ol["platform"]).unsqueeze(0)
    data.ol_mask = torch.from_numpy(ol["mask"]).unsqueeze(0)
    ac = enc["actions"]
    data.act_route_idx = torch.from_numpy(ac["route_local_idx"]).unsqueeze(0)
    data.act_mask = torch.from_numpy(ac["action_mask"]).unsqueeze(0)
    data.n_candidates = torch.tensor([ac["n_candidates"]], dtype=torch.float32)
    data.chosen_action_idx = torch.tensor([ac["chosen_action_idx"]], dtype=torch.long)
    data.special_flags = torch.from_numpy(enc["special_flags"]).unsqueeze(0)
    data.r_total = torch.tensor([enc["r_total"]], dtype=torch.float32)
    # sample_id travels with the graph so the trainer can join sidecar labels
    # (e.g. time_bucket from time_labels_v2.parquet) by sample_id.
    data.sample_id = torch.tensor([int(enc.get("sample_id", -1))], dtype=torch.long)
    return data


def time_split_of(t) -> str:
    """Map a timestamp → 'train'|'val'|'test' per spec 04 §4.1 (time-based, LOCKED).

    train: t < VAL_START (2024-02-01) | val: < TEST_START (2024-03-01) | test: ≥ that.
    """
    import pandas as pd
    from .. import config as C
    ts = pd.Timestamp(t)
    if ts < pd.Timestamp(C.VAL_START):
        return "train"
    if ts < pd.Timestamp(C.TEST_START):
        return "val"
    return "test"


def load_pass_split(path=None) -> dict:
    """Load pass_id → split mapping (built by 00_build_time_split.py). {} if absent."""
    from .. import config as C
    import pyarrow.parquet as pq
    p = path or C.PASS_SPLIT_PARQUET
    try:
        if not p.exists():
            return {}
    except AttributeError:
        from pathlib import Path as _P
        p = _P(p)
        if not p.exists():
            return {}
    tbl = pq.read_table(str(p), columns=["pass_id", "split"])
    return dict(zip(tbl.column("pass_id").to_pylist(), tbl.column("split").to_pylist()))


def load_episode_split(path=None) -> dict:
    """sample_id → split, from episodes_v2.parquet (built by scripts/mdp/14_resegment_episodes.py).

    AUTHORITATIVE split after the Stage 4.7.2d episode re-segmentation fix: the
    in-file pass_id/position columns are stale; segmentation now lives in the
    sidecar keyed by sample_id. Returns {} if the sidecar is absent (callers then
    fall back to the legacy pass_split-by-pass_id path).
    """
    from .. import config as C
    import pyarrow.parquet as pq
    from pathlib import Path as _P
    p = path or (C.SNAPSHOTS_V2_PARQUET.parent / "episodes_v2.parquet")
    try:
        if not _P(p).exists():
            return {}
    except (AttributeError, OSError):
        return {}
    tbl = pq.read_table(str(p), columns=["sample_id", "split"])
    return dict(zip(tbl.column("sample_id").to_pylist(), tbl.column("split").to_pylist()))


class SnapshotDataset:
    """torch.utils.data.Dataset over snapshots_v2.parquet. Lazy torch import.

    split: 'train'|'val'|'test'|None — filter by the TIME-BASED split (spec 04
    §4.1), via pass_split.parquet (pass_id → split, assigned by episode start
    time). This must match the split used to compute normalization stats so the
    z-score / vocab stay leak-free. If pass_split.parquet is missing it falls
    back to the legacy md5(pass_id) hash split with a warning (NOT spec-compliant).
    """

    def __init__(self, parquet_path, stats_path, split: Optional[str] = None,
                 val_frac: float = 0.15, test_frac: float = 0.15,
                 pass_split_path=None):
        import pyarrow.parquet as pq
        self.pf = pq.ParquetFile(str(parquet_path))
        self.stats = NormStats.load(stats_path)
        self.split = split
        self.val_frac, self.test_frac = val_frac, test_frac
        self._pass_split = load_pass_split(pass_split_path)
        if not self._pass_split:
            print("[SnapshotDataset][warn] pass_split.parquet not found — falling "
                  "back to legacy md5(pass_id) hash split (NOT time-based / spec 04 §4.1). "
                  "Run scripts/train/00_build_time_split.py first.")
        # Build a row-group index of (rg, local_row) for the chosen split.
        self._index = self._build_index()

    def _split_of(self, pass_id: str) -> str:
        # Time-based (spec 04 §4.1) when the mapping is available.
        if self._pass_split:
            return self._pass_split.get(str(pass_id), "train")
        import hashlib
        h = int(hashlib.md5(str(pass_id).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        if h < self.test_frac:
            return "test"
        if h < self.test_frac + self.val_frac:
            return "val"
        return "train"

    def _build_index(self):
        idx = []
        for rg in range(self.pf.num_row_groups):
            pass_ids = self.pf.read_row_group(rg, columns=["pass_id"]).column("pass_id").to_pylist()
            for li, pid in enumerate(pass_ids):
                if self.split is None or self._split_of(pid) == self.split:
                    idx.append((rg, li))
        return idx

    def __len__(self):
        return len(self._index)

    def __getitem__(self, i):
        rg, li = self._index[i]
        row = self.pf.read_row_group(rg).slice(li, 1).to_pylist()[0]
        return to_heterodata(encode_snapshot(row, self.stats))
