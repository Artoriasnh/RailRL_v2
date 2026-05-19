# Spec 03 — Model Architecture

**Document version:** v1.0 · **Last updated:** 2026-05-19
**Status:** 🟡 draft — awaiting sign-off
**Prerequisite:** Spec 02 v1.0 (sign-off implicit by progressing to this spec)
**Scope:** the neural network architecture that consumes `snapshots_v2.parquet`
and produces `Q(s, a)` values + auxiliary supervised predictions. Spec 04 will
define the training algorithm (CQL/IQL) that uses these outputs.

---

## §0 Purpose & scope

### What this spec locks down

- **Encoder** — HGT graph branch + Transformer event-token branch + fusion → `s_emb`
- **Q-network** — per-action MLP, structured-action support (dynamic |A_t|)
- **Auxiliary heads** — 2 supervised heads (route classifier, time MDN). Priority head **dropped** with justification (§7.3).
- **Input pipeline** — snapshot → padded tensors + masks
- **All hyperparameters** — dimensions, layers, dropout, etc.
- **Parameter count** + compute budget
- **Module map** for `src/railrl/encoders/` and `src/railrl/policies/`

### What this spec does NOT cover

- Training algorithm (CQL loss, conservative penalty, 3-stage protocol) → spec **04**
- XAI + evaluation → spec **05**

### Important change from earlier discussion: priority head dropped

PROJECT_HANDOFF.docx Ch 7.5 listed 3 auxiliary heads (route + timing + priority).
This spec **drops the priority head** based on spec 02's decision-point granularity:
each decision point is per `(focal_train, focal_signal, t)`, so within one
decision point there's no priority decision to predict (it's all about
focal_train). Priority emerges **across** decision points at similar t via the
wait-vs-act pattern — already captured by the main Q. See §7.3 for full reasoning.

---

## §1 Overview — from snapshot to action scores

```
┌────────────────────────────────────────────────────────────────────┐
│  snapshots_v2.parquet (one row = one decision point)               │
│   nested: state_nodes_*, state_edges_*, state_event_tokens,        │
│           state_schedule_outlook, state_special_flags              │
└────────────────────────────────────────────────────────────────────┘
                              │
                              ▼ DataLoader
┌────────────────────────────────────────────────────────────────────┐
│  Batched padded tensors + masks (§2)                               │
└────────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼──────────────────┐
              ▼               ▼                  ▼
        ┌──────────┐    ┌──────────┐       ┌──────────┐
        │ HGT      │    │ Transf.  │       │ scalar/  │
        │ graph    │    │ event    │       │ outlook  │
        │ (§3)     │    │ seq (§4) │       │ globals  │
        └────┬─────┘    └────┬─────┘       └────┬─────┘
             │ per-node      │ seq summary      │
             │ embeddings    │ (128-d)          │ (8 flags +
             │ (128-d each)  │                  │  outlook)
             └───────────────┴──────────────────┘
                              │
                              ▼ Fusion (§5)
                       ┌──────────────┐
                       │ s_emb (256)  │
                       └──────┬───────┘
                              │
                ┌─────────────┼──────────────┐
                ▼             ▼              ▼
        ┌───────────┐  ┌───────────┐   ┌───────────┐
        │ Q-network │  │ Route     │   │ Time MDN  │
        │ per-action│  │ classifier│   │ 5-bucket  │
        │ MLP (§6)  │  │ (aux, §7) │   │ (aux, §7) │
        └─────┬─────┘  └───────────┘   └───────────┘
              │
              ▼
       Q(s, a) for each a ∈ A_t
       → argmax (with masking) → action
```

---

## §2 Input pipeline — snapshot to tensors

### 2.1 Padding strategy (resolves spec 02 §11 Q1-Q2-Q5)

Fixed-capacity padding for batched training:

| Item | Cap | Sentinel | Mask |
|------|-----|----------|------|
| Track nodes | **60** | zero feature vec | `mask_track ∈ {0, 1}^60` |
| Signal nodes | **15** | zero feature vec | `mask_signal ∈ {0, 1}^15` |
| Route nodes | **15** | zero feature vec | `mask_route ∈ {0, 1}^15` |
| Train nodes | **8** | zero feature vec, `is_focal=False` | `mask_train ∈ {0, 1}^8` |
| Candidate actions | **14** | sentinel route_emb=0 | `mask_action ∈ {0, 1}^{14+1}` (+1 for wait, always unmasked) |

**Truncation rule** (rare, per spec 02 §11 Q5): if any item exceeds its cap,
keep the top-k by relevance:
- Tracks: keep those on `on_focal_train_path=True` first, then by `n_routes_using` descending
- Routes: keep `in_candidate_set=True` first, then by `n_tcs_occupied_by_focal` descending
- Trains: keep `is_focal=True` first, then by distance to focal_train.current_tc
- Candidates: keep by route `gap_time_s` ascending (shorter routes likely more relevant)

**Edges:** stored as sparse `edge_index ∈ Z^{2 × n_edges}` per edge type. No
padding (PyG/DGL handles variable-length).

### 2.2 Per-node feature normalization

All continuous features z-score normalized using **train-split-only** statistics
(no leakage from val/test). Statistics persisted to:

```
outputs/snapshots/normalization_stats.json
```

| Feature kind | Normalization |
|--------------|---------------|
| Continuous (length_m, ages, fractions) | `(x - μ_train) / σ_train`, clipped to [-5, +5] |
| Binary flags | 0/1, no normalization |
| Categorical (prefix, hc_class) | One-hot or learned embedding (8-d each) |
| Identity (track_id, route_id strings) | NOT a feature — model uses learned embedding indexed by `asset_idx` (see §3.1) |

### 2.3 Event token preparation (resolves spec 02 §11 Q3)

Each token: `(asset_idx ∈ [0, 671], state ∈ {0, 1}, time_delta_s ∈ R^+)`.

Tensor encoding per token:
- `asset_emb = Embedding(673, 64)(asset_idx)` (672 assets + 1 padding idx)
- `state_emb = Embedding(3, 8)(state + 1)` (-1 for padding → 0)
- `time_emb = sinusoidal_pe(log1p(time_delta_s))` → 32-d

Token vector = `[asset_emb, state_emb, time_emb]` ∈ R^104.

Decision: **use log1p(time_delta_s)** for positional encoding — empirical event
intervals span 0.1 s to 17 min (5 orders of magnitude); log-scale gives uniform
resolution.

### 2.4 Schedule outlook tensor

Top-5 upcoming trains. Per-row features:
- `headcode_class` (one of 11 categories) → 8-d embedding
- `eta_s` → log1p + normalize → 1-d
- `planned_platform ∈ {1..6, None}` → 7-d one-hot (extra slot for None)

Total per upcoming train: 16-d. Tensor shape: `(B, 5, 16)`.

Padded with zero rows for snapshots with < 5 upcoming.

---

## §3 Encoder — graph branch (HGT)

### 3.1 Node embedding initialization

Each of 4 node types gets:
1. Per-type **identity embedding** indexed by `asset_idx` (when applicable)
2. Per-type **feature MLP** projecting raw features to common `d_model`

| Node type | Identity emb dim | Feature MLP input | Output |
|-----------|------------------|--------------------|--------|
| Track | `Embedding(250, 64)` | 18 features | Linear(64+18 → 128) |
| Signal | `Embedding(124, 64)` | 18 features | Linear(64+18 → 128) |
| Route | `Embedding(278, 64)` | 18 features (incl. 5 Derby_info physical) | Linear(64+18 → 128) |
| Train | `Embedding(2200, 32)` (~2185 unique trains) | 10 features (incl. `is_focal` 1-bit) | Linear(32+10 → 128) |

All four output 128-d per node (`d_model = 128`).

### 3.2 HGT layers

```
d_model     = 128
n_heads     = 4
n_layers    = 3   (locked)
dropout     = 0.1
n_node_types = 4
n_edge_types = 8  (6 static + 2 dynamic: at_berth, next_signal)
```

Each HGT layer (Hu et al. 2020):
- Per-edge-type attention with **type-specific** Q/K/V projections
- Multi-head attention aggregation per node
- Residual + LayerNorm + 2-layer FFN

Output: per-node embedding ∈ R^128 for each non-padded node.

### 3.3 Edge tensor format

For each edge type t ∈ {connects, traverses, starts_at, ends_at, protects,
same_signal, at_berth, next_signal}:
- `edge_index_t ∈ Z^{2 × n_edges_t}` — source/target node indices (within the
  padded node tensor)
- `edge_attr_t ∈ R^{n_edges_t × d_edge}` — for `traverses` only: `order` attribute
  (1-d), else 0-d

### 3.4 Output of graph branch

`H_graph ∈ R^{N_total × 128}` where N_total = 60+15+15+8 = 98 (padded total).

Plus type-segmented views:
- `H_track ∈ R^{60 × 128}`
- `H_signal ∈ R^{15 × 128}`
- `H_route ∈ R^{15 × 128}`
- `H_train ∈ R^{8 × 128}`

### 3.5 Graph pooling

```
h_graph_global = mean_pool(H_graph, mask=node_mask)  # ∈ R^128
```

Per-type pooled summaries (for fusion / aux heads):
- `h_track_pool`, `h_signal_pool`, `h_route_pool`, `h_train_pool` — each 128-d.

---

## §4 Encoder — sequence branch (Transformer over event tokens)

### 4.1 Token embedding

Per §2.3:
- `asset_emb (64) + state_emb (8) + time_emb (32) = 104-d` token vector

Project to d_model via `Linear(104 → 128)`.

### 4.2 Positional encoding

`time_delta_s` is itself the positional signal (already encoded in time_emb).
**No extra positional encoding** (avoid double-counting).

### 4.3 Transformer config

```
d_model    = 128
n_heads    = 4
n_layers   = 4   (locked; deeper than HGT because sequence is the only temporal source)
dropout    = 0.1
ff_dim     = 512  (= 4 × d_model)
K          = 256  tokens (per spec 01)
```

Padding mask: tokens with `asset_idx = -1` are masked.

### 4.4 Output of sequence branch

`h_seq_final = last_unmasked_token(H_seq)` ∈ R^128.

Alternative: `h_seq_pool = mean_pool(H_seq, mask=token_mask)`. Both computed;
fusion uses both.

---

## §5 Encoder — fusion

### 5.1 Components fused

```python
fusion_input = concat([
    h_graph_global,    # 128  (mean over all nodes)
    h_focal_train,     # 128  (the is_focal=True train node's embedding)
    h_seq_final,       # 128  (last event token)
    h_seq_pool,        # 128  (mean event token)
    schedule_global,   # 16   (mean over 5 upcoming, masked)
    special_flags,     # 8    (binary/numeric flags)
    n_candidates,      # 1    (scalar, normalised)
])
# total: 657-d
```

### 5.2 Fusion module

```python
class Fusion(nn.Module):
    def __init__(self, in_dim=657, out_dim=256):
        self.ln1   = nn.LayerNorm(in_dim)
        self.fc1   = nn.Linear(in_dim, 512)
        self.ln2   = nn.LayerNorm(512)
        self.fc2   = nn.Linear(512, out_dim)
        self.drop  = nn.Dropout(0.1)
    
    def forward(self, x):
        x = self.ln1(x)
        x = F.gelu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        x = self.ln2(x)
        return x   # s_emb ∈ R^256
```

Output: `s_emb ∈ R^256`.

---

## §6 Q-network — per-action MLP

### 6.1 Input per candidate action

For each candidate action `a_i = (focal_train, R_i)`:

```python
action_input_i = concat([
    h_focal_train,              # 128 — from HGT
    h_route_i,                  # 128 — from HGT (R_i's embedding)
    s_emb,                      # 256 — from fusion
    is_in_candidate_set_i,      # 1   (always True for actual candidates; useful for debugging)
    n_candidates,               # 1   (normalised)
])
# total: 514-d
```

For the wait action `a_0`:

```python
wait_input = concat([
    h_focal_train,              # 128
    h_seq_final,                # 128 (proxy for "no route chosen")
    s_emb,                      # 256
    n_candidates,               # 1
])
# total: 513-d (use separate MLP_wait to handle different input dim)
```

### 6.2 MLP architecture

```python
class QNetwork(nn.Module):
    def __init__(self):
        self.mlp_action = nn.Sequential(
            nn.Linear(514, 512), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, 1),    # scalar Q
        )
        self.mlp_wait = nn.Sequential(
            nn.Linear(513, 256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, 1),
        )
    
    def forward(self, h_train, h_routes, s_emb, n_cand, action_mask):
        # h_routes: (B, max_candidates=14, 128)
        # action_mask: (B, max_candidates) — True if real candidate
        B, K, D = h_routes.shape
        
        # Per-action scoring
        s_emb_exp = s_emb.unsqueeze(1).expand(-1, K, -1)         # (B, K, 256)
        h_train_exp = h_train.unsqueeze(1).expand(-1, K, -1)     # (B, K, 128)
        n_cand_exp = n_cand.view(B, 1, 1).expand(-1, K, 1)        # (B, K, 1)
        
        action_in = torch.cat([
            h_train_exp, h_routes, s_emb_exp,
            torch.ones(B, K, 1, device=h_train.device),   # is_in_candidate_set
            n_cand_exp,
        ], dim=-1)                                                  # (B, K, 514)
        
        q_actions = self.mlp_action(action_in).squeeze(-1)         # (B, K)
        # Mask invalid: q for masked actions → -inf
        q_actions = q_actions.masked_fill(~action_mask, -1e9)
        
        # Wait Q
        wait_in = torch.cat([h_train, h_seq_final, s_emb,
                              n_cand.view(B, 1)], dim=-1)            # (B, 513)
        q_wait = self.mlp_wait(wait_in).squeeze(-1)                # (B,)
        
        # Concatenate: A_t = [wait, action_1, ..., action_K]
        Q_all = torch.cat([q_wait.unsqueeze(1), q_actions], dim=1)  # (B, K+1)
        return Q_all
```

### 6.3 Why per-action MLP (not 277-class softmax)

- Action space is **dynamic** (1 to ~14 per snapshot)
- Need to score new routes without retraining (route_emb comes from HGT, which
  generalizes to new route nodes)
- Each Q value is interpretable as "this specific action's expected return"

### 6.4 Action selection (inference)

```python
def select_action(snapshot):
    Q = q_network(...)                # (1, K+1)
    a_idx = Q.argmax(dim=1)           # 0 = wait, 1..K = candidate i-1
    return a_idx.item()
```

---

## §7 Auxiliary supervised heads

### 7.1 Route classifier head

**Purpose:** force encoder to learn "which candidate route fits this train's
trajectory + state" — provides dense gradient even when RL signal is sparse.

**Architecture:**
```python
class RouteHead(nn.Module):
    def forward(self, h_train, h_routes, action_mask):
        # logits over candidates only (excluding wait)
        scores = (h_train.unsqueeze(1) * h_routes).sum(-1)   # (B, K) dot product
        scores = scores.masked_fill(~action_mask, -1e9)
        return scores
```

**Loss:** Cross-entropy over candidate indices. Masked: only set decisions
contribute (wait rows are skipped).

```python
L_route = CrossEntropy(scores[set_mask], chosen_action_idx[set_mask] - 1)
# -1 because chosen_action_idx counts wait at 0
```

**Output metric for §VII eval:** `route_head_top1_acc` (within-candidate
accuracy).

### 7.2 Time MDN head (lead-time prediction)

**Purpose:** force encoder to learn "how soon should this PR happen" — captures
timing decision separately from route choice.

**Target:** for set decisions, the lead time τ between PR (`t_PR`) and the
first TC occupation of the chosen route (`t_first_TC`):

```
τ = t_first_TC - t_PR    (seconds)
```

This is the same `next_tc_headway`-related quantity used in spec 01 §11.4 but
measured for THIS train (not the next train).

**Bucketization** (5-class categorical):

| Bucket | Range | Label |
|--------|-------|-------|
| 0 | τ ≤ 5 s | "immediate" |
| 1 | 5 < τ ≤ 15 s | "quick" |
| 2 | 15 < τ ≤ 30 s | "standard" |
| 3 | 30 < τ ≤ 60 s | "delayed" |
| 4 | τ > 60 s | "long lead" |

**Architecture:**
```python
class TimeHead(nn.Module):
    def __init__(self):
        self.mlp = nn.Sequential(
            nn.Linear(128 + 256, 128), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(128, 5),  # 5 buckets
        )
    def forward(self, h_focal_train, s_emb):
        x = torch.cat([h_focal_train, s_emb], dim=-1)
        return self.mlp(x)   # (B, 5) logits
```

**Loss:** Cross-entropy on 5 buckets. Masked: only set decisions with valid
τ contribute (some have NaN τ — exclude).

```python
L_time = CrossEntropy(logits[time_valid_mask], bucket[time_valid_mask])
```

### 7.3 Priority head — DROPPED (with justification)

**Decision:** No priority head in spec 03.

**Why originally proposed (PROJECT_HANDOFF Ch 7.5):**
- Plan §3.2 lists 3 decision types: route, timing, priority
- Symmetric coverage by 3 heads seemed natural

**Why dropped now:**
1. **Decision-point granularity (spec 02 §2.1)** — each decision point is per
   `(focal_train, focal_signal, t)`. Within ONE decision point, there's no
   priority decision to predict — it's all about the focal_train.
2. **FCFS finding (PROJECT_HANDOFF Ch 2.6 callout)** — Kendall τ ≈ 0.998 already
   from FCFS; supervised prediction is trivial. Priority's value is policy
   improvement (CQL finding non-FCFS optima), which is the Q function's job,
   not a separate head's.
3. **Cross-decision-point priority** — emerges from the wait-vs-act pattern of
   the main Q across simultaneous decision points (e.g., model says set for
   train T_1 at decision point A, wait for T_2 at decision point B at same t →
   priority T_1 > T_2). This is captured by Q without a dedicated head.

**Evaluation still reports priority metrics** (per PROJECT_HANDOFF Ch 2.6):
- Imitation Kendall τ vs signaller (expected ≈ 0.998)
- Counterfactual reward delta (CQL choice vs FCFS, via L3 simulator)

These are **eval metrics computed from Q outputs**, not predictions from a
separate head.

**Future option:** if ESWA reviewers explicitly request a "joint 3-decision
prediction" framing, easy to add a small priority head later (binary pairwise
"should I serve T_i before T_j" from `[h_train_i, h_train_j, s_emb]`).

### 7.4 Loss weights

```python
L_total = L_RL                       # main: CQL/IQL — defined in spec 04
        + 0.5 * L_route               # aux: route classifier
        + 0.2 * L_time                # aux: time bucket
```

Aux loss weights chosen for:
- `L_route` to be ~half of `L_RL` magnitude (helps representation strongly)
- `L_time` lighter (more noise in lead-time labels)

These weights are starting points; spec 04 may tune via Phase A sanity.

---

## §8 Forward pass (end-to-end)

```python
def forward(batch):
    # ====== Encoder ======
    # Graph branch
    H_graph = hgt_encoder(
        node_features=batch.nodes,
        edge_indices=batch.edges,
        node_type_ids=batch.node_types,
    )
    H_track, H_signal, H_route, H_train = segment_by_type(H_graph)
    h_focal_train = H_train.gather(... where is_focal=True ...)  # (B, 128)
    
    # Sequence branch
    H_seq = seq_transformer(batch.event_tokens, mask=batch.token_mask)
    h_seq_final = H_seq[:, -1]
    h_seq_pool = mean_pool(H_seq, batch.token_mask)
    
    # Fusion
    s_emb = fusion(concat([
        mean_pool(H_graph, batch.node_mask),
        h_focal_train,
        h_seq_final,
        h_seq_pool,
        mean_pool(batch.schedule_outlook, batch.outlook_mask),
        batch.special_flags,
        batch.n_candidates,
    ]))                                                            # (B, 256)
    
    # ====== Heads ======
    Q_all = q_network(h_focal_train, H_route, s_emb,
                      batch.n_candidates, batch.action_mask)        # (B, K+1)
    
    route_logits = route_head(h_focal_train, H_route,
                              batch.action_mask)                    # (B, K)
    time_logits = time_head(h_focal_train, s_emb)                   # (B, 5)
    
    return {
        'Q': Q_all,                # for L_RL (spec 04)
        'route_logits': route_logits,
        'time_logits': time_logits,
    }
```

---

## §9 Inference mode

```python
@torch.no_grad()
def predict_action(snapshot):
    model.eval()
    out = model(snapshot.unsqueeze(0))
    action_idx = out['Q'].argmax(dim=1).item()
    if action_idx == 0:
        return 'wait', None
    else:
        route_id = snapshot.candidate_route_ids[action_idx - 1]
        return 'set', route_id
```

Inference cost target: **< 50 ms per decision** on CPU (Q-network is small).

---

## §10 Parameter count + compute budget

### 10.1 Estimated parameters

| Module | Approx params |
|--------|---------------|
| Track identity embedding (250 × 64) | 16 k |
| Signal identity embedding (124 × 64) | 8 k |
| Route identity embedding (278 × 64) | 18 k |
| Train identity embedding (2200 × 32) | 70 k |
| Feature MLPs (4 × Linear) | 50 k |
| HGT 3 layers × (8 edge types × QKV projections + FFN) | ~1.2 M |
| Asset embedding for event stream (673 × 64) | 43 k |
| Sequence Transformer 4 layers | ~800 k |
| Fusion module | 350 k |
| Q-network MLP (514→512→256→128→1) + wait MLP | 350 k |
| Route head | 0 (parameter-free dot product) |
| Time head MLP (384→128→5) | 50 k |
| **Total** | **~3.0 M parameters** |

### 10.2 Compute / memory budget

- **Training**: ~3 M params × FP32 ≈ 12 MB weights + optimizer state ≈ 50 MB
- **Per-batch GPU memory**: B=256 batches × (98 nodes × 128 + 256 tokens × 128) ≈ ~2 GB activations
- **Training time**: ~12 hours per seed on A100 for 40 epochs (spec 04 will tune)
- **Inference**: < 50 ms per decision on CPU; < 5 ms on GPU

### 10.3 Comparison

| Aspect | v1 baseline (B1 BC-MLP) | v2 main model |
|--------|--------------------------|---------------|
| Parameters | ~50 k | ~3 M (60× larger) |
| State input | 115-d flat | ~1,500 numbers (per-node + sequence) |
| Encoder | None (flat MLP) | HGT + Transformer + Fusion |
| Action space | binary | structured, dynamic |

---

## §11 Implementation modules

### 11.1 Module map

```
src/railrl/encoders/
├── __init__.py
├── hgt.py                 # §3: HGT encoder
├── sequence.py            # §4: Transformer over event tokens
├── fusion.py              # §5: fusion module
└── input_pipeline.py      # §2: snapshot → padded tensors + normalization

src/railrl/policies/
├── __init__.py
├── q_network.py           # §6: per-action MLP
└── heads.py               # §7: route + time aux heads (no priority)
```

### 11.2 Module API

```python
# hgt.py
class HGTEncoder(nn.Module):
    def __init__(self, d_model=128, n_layers=3, n_heads=4,
                 n_node_types=4, n_edge_types=8, dropout=0.1): ...
    def forward(self, node_features, edge_indices, node_types) -> Tensor: ...

# sequence.py
class SeqEncoder(nn.Module):
    def __init__(self, d_model=128, n_layers=4, n_heads=4, K=256, dropout=0.1): ...
    def forward(self, tokens, mask) -> Tensor: ...

# fusion.py
class Fusion(nn.Module):
    def __init__(self, in_dim=657, out_dim=256): ...
    def forward(self, x) -> Tensor: ...

# q_network.py
class QNetwork(nn.Module):
    def forward(self, h_train, h_routes, s_emb, n_cand, action_mask) -> Tensor: ...

# heads.py
class RouteHead(nn.Module):
    def forward(self, h_train, h_routes, action_mask) -> Tensor: ...
class TimeHead(nn.Module):
    def forward(self, h_focal_train, s_emb) -> Tensor: ...
```

### 11.3 Top-level model class

```python
# src/railrl/model.py
class RailRLModel(nn.Module):
    def __init__(self, config):
        self.encoder = HGTEncoder(...)
        self.seq = SeqEncoder(...)
        self.fusion = Fusion(...)
        self.q = QNetwork()
        self.route_head = RouteHead()
        self.time_head = TimeHead()
    
    def forward(self, batch): ...
    
    def predict_action(self, snapshot): ...
```

---

## §12 Hyperparameter summary (one-page reference)

```
╔══════════════════════════════════════════════════════════════════╗
║  EMBEDDING DIMENSIONS                                             ║
║    d_model (all encoders)             = 128                       ║
║    s_emb (fusion output)              = 256                       ║
║    asset_emb (event tokens)           = 64                        ║
║    state_emb (event tokens)           = 8                         ║
║    time_emb (event tokens, sinusoidal)= 32                        ║
║    Identity embeddings per node type  = 32-64                     ║
║                                                                    ║
║  ARCHITECTURE                                                     ║
║    HGT layers                         = 3                         ║
║    HGT heads                          = 4                         ║
║    Sequence Transformer layers        = 4                         ║
║    Sequence Transformer heads         = 4                         ║
║    Sequence Transformer ff_dim        = 512                       ║
║    Q-network MLP                      = [514, 512, 256, 128, 1]   ║
║    Wait MLP                           = [513, 256, 128, 1]        ║
║    Route head                         = parameter-free dot product║
║    Time head MLP                      = [384, 128, 5]             ║
║                                                                    ║
║  PADDING CAPS                                                     ║
║    max_tracks                         = 60                        ║
║    max_signals                        = 15                        ║
║    max_routes                         = 15                        ║
║    max_trains                         = 8                         ║
║    max_candidates                     = 14                        ║
║    K (event tokens)                   = 256                       ║
║                                                                    ║
║  AUX LOSS WEIGHTS                                                 ║
║    λ_route                            = 0.5                       ║
║    λ_time                             = 0.2                       ║
║    λ_priority                         = DROPPED                   ║
║                                                                    ║
║  REGULARIZATION                                                   ║
║    Dropout (all)                      = 0.1                       ║
║    Feature clip                       = [-5, +5] after z-score    ║
║                                                                    ║
║  TIME BUCKETS (Time MDN)                                          ║
║    [≤5s, 5-15s, 15-30s, 30-60s, >60s]                            ║
║                                                                    ║
║  PARAM COUNT                                                      ║
║    Total                              = ~3.0 M                    ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## §13 Open questions for spec 04 to inherit

| # | Question | Default proposal |
|---|----------|------------------|
| 1 | Should Phase A (encoder + aux only) use ALL data or balanced sampling? | Balanced: oversample wait + unusual cases (per Ch 10 special-case strata) |
| 2 | CQL alpha (conservative penalty weight) starting value? | α = 5.0 (Kumar 2020 default); tune in Phase B |
| 3 | Optimizer + lr | AdamW, lr=3e-4 with cosine warmup; β_1=0.9, β_2=0.999 |
| 4 | Batch size | 256 (fits A100 with margin); spec 04 may tune |
| 5 | Gradient clipping | 1.0 (standard for Transformer training) |
| 6 | Whether to use separate target network for Q | Yes (CQL standard) — soft update τ=0.005 |
| 7 | Validation/test split | Time-based: train 2023-02-28 to 2024-01-31, val 2024-02-01 to 2024-02-29, test 2024-03-01 to 2024-04-25 |

---

## §14 Changelog

- **v1.0 (2026-05-19)** — Initial draft. Locks encoder architecture (HGT
  d=128 L=3 H=4 + Seq Transformer d=128 L=4 H=4 + Fusion → 256-d s_emb),
  Q-network (per-action MLP), 2 auxiliary heads (route + time MDN —
  **priority head DROPPED** with justification), padding caps (60/15/15/8/14),
  loss weights (λ_route=0.5, λ_time=0.2), full hyperparameter table, and
  ~3 M parameter count.

---

**End of Spec 03.**
**Sign-off:** ☐ Hao  /  Date: ______
