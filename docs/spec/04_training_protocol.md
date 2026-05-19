# Spec 04 — Training Protocol

**Document version:** v1.0 · **Last updated:** 2026-05-19
**Status:** 🟡 draft — awaiting sign-off
**Prerequisite:** Spec 03 v1.0 (signed-off implicitly by progressing here)
**Scope:** the training algorithm (CQL main + IQL alternative + BC baseline),
3-stage protocol, optimizer/scheduler, splits, sampling, target network,
checkpointing, reproducibility, compute budget. Every script in `scripts/train/`
answers to this spec.

---

## §0 Purpose & scope

### What this spec locks down

- **Algorithms**: CQL (main) + IQL (alt) + BC (baseline reference)
- **Loss functions**: complete formulas for each
- **3-stage protocol**: Phase A (5 ep) + Phase B (15 ep) + Phase C (20 ep) = 40 ep
- **Train/val/test split**: time-based (locked dates)
- **Optimizer + scheduler**: AdamW + cosine warmup, lr=3e-4
- **Batch size + sampling strategy** (stratified)
- **Target network** soft update (CQL)
- **Reproducibility**: 3 seeds (42/43/44), deterministic ops where possible
- **Compute budget**: ~12 hours/seed on A100, 3 seeds = ~36 GPU-hours per algorithm

### What this spec does NOT cover

- Eval framework + XAI + selective override → spec **05**
- Specific baseline implementations (B0', B0'') → spec **05**

---

## §1 Algorithm choice and rationale

### 1.1 Main algorithm: CQL (Conservative Q-Learning, Kumar et al. 2020)

**Why CQL is the right choice for this project:**

| Property | Why it matters |
|----------|----------------|
| Conservative — pushes down Q for OOD actions | Prevents "imagining" actions the signaller never tried |
| Stays close to demonstration data when demos are good | Replicate-AND-Improve: replicates when signaller right |
| Allows improvement in support when demos are heterogeneous | Improves when signaller sub-optimal (mixed-quality data) |
| Mature, well-cited in offline-RL literature | ESWA reviewer-friendly (no exotic methods) |
| Works with dynamic discrete action spaces | Our `A_t` has variable size 1-14 |

### 1.2 Alternative algorithm: IQL (Implicit Q-Learning, Kostrikov et al. 2021)

Run as **comparison** to CQL. Different mechanism:
- Uses **expectile regression** of V instead of conservative penalty
- Doesn't need OOD action sampling
- Often less conservative than CQL
- Trains policy via **advantage-weighted regression**

Why include: validates that "CQL beats baselines" isn't an algorithm artifact;
if IQL gives similar performance, both are reported. If IQL ≪ CQL or vice
versa, the gap itself is an empirical finding for §VII.

### 1.3 BC baseline (for §VII Table I)

Pure behavioral cloning — predicts `chosen_action_idx` via cross-entropy on
the route head only (the route head defined in spec 03 §7.1, no Q learning).
This is the **B1 baseline** in evaluation.

### 1.4 Methods explicitly NOT used

| Method | Why not |
|--------|---------|
| IDQL (Diffusion + IQL) | 2023 SOTA but heavy implementation; user said "不需要参考railmind" |
| Decision Transformer | Sequence model approach; awkward to condition on returns in offline setting |
| TD3+BC, AWAC | Designed for continuous actions; ours is discrete structured |
| Vanilla DQN / DDQN | No offline-RL safety guarantees; will diverge with OOD actions |
| BCQ (Batch-Constrained DQN) | Earlier than CQL/IQL; less commonly cited in 2024-26 papers |

---

## §2 Loss functions

### 2.1 CQL loss (main)

```
L_CQL = L_TD + α · L_cons

L_TD  = E_(s,a,r,s',done) [ ( Q(s,a) - y )² ]
        where y = r + γ·(1 - done) · max_{a' ∈ A_{s'}} Q_target(s', a')

L_cons = E_s [ log Σ_{a ∈ A_s} exp(Q(s, a))  -  E_{a ~ π_β(·|s)} Q(s, a) ]
         "push down all Q values; pull up data-action Q value"
```

Where:
- `γ = 0.95` (spec 02 §5.4, locked)
- `α = 5.0` initial (Kumar 2020 default), tunable in Phase B
- `π_β` = behavior policy estimate; for discrete actions with one data action
  per (s, a) pair, π_β simply assigns prob 1 to the observed a
- `A_s, A_s'` = action sets at s and s' (variable size per snapshot, masked
  per spec 03 §6.2)

**Important detail:** `Q_target` is a separate network (target network, §6),
updated softly. `max` over `A_{s'}` uses the action mask.

### 2.2 Auxiliary supervised losses (from spec 03 §7)

```
L_route = CE( route_logits[set_mask],  chosen_action_idx[set_mask] - 1 )
L_time  = CE( time_logits[time_valid_mask], time_bucket[time_valid_mask] )
```

Note: `set_mask` excludes wait rows (wait has no `chosen_action_idx > 0`).
`time_valid_mask` excludes set rows where lead-time τ is NaN.

### 2.3 Total loss

```
L_total = L_CQL + 0.5 · L_route + 0.2 · L_time
```

Coefficients per spec 03 §7.4 (locked unless Phase A tuning finds otherwise).

### 2.4 IQL loss (alternative)

```
L_V    = E_(s,a) [ L²_τ ( Q_target(s, a) - V(s) ) ]
         where L²_τ(u) = |τ - 1[u<0]| · u²  (expectile loss, τ=0.7)

L_Q    = E_(s,a,r,s') [ ( r + γ·(1-done)·V(s') - Q(s, a) )² ]

L_π    = E_(s,a) [ -exp( β · (Q(s,a) - V(s)) ) · log π(a|s) ]
         where π is the policy head (here: the Q-argmax distribution)
         β = 3.0 (temperature)
```

For IQL, the policy is implicit via Q-argmax (no separate policy network).
Aux losses (L_route, L_time) added with same weights as CQL.

### 2.5 BC loss (baseline)

```
L_BC = CE( route_logits[set_mask],  chosen_action_idx[set_mask] - 1 )
     + λ_wait · BCE( wait_logits, label == 'wait' )
       where wait_logits is a sigmoid output from a single Linear(s_emb, 1)
       λ_wait = 0.3 (matches w_wait in reward weights)
```

BC has no Q-network, no target network, no conservative penalty. Trained once
without 3-stage protocol (just standard supervised training, 20 epochs).

---

## §3 Three-stage training protocol

### 3.1 Phase A — encoder + auxiliary heads only (5 epochs)

**Frozen:** Q-network (does not exist yet — random initialization deferred)
**Trained:** encoder (HGT + Seq + Fusion), route head, time head

**Loss:**
```
L_phase_A = 0.5 · L_route + 0.2 · L_time
```

(No L_RL yet — Q-network is not in the graph.)

**Purpose:** get encoder to learn useful representations from supervised
gradient (much denser/cleaner than RL gradient).

**Success criteria** (must pass before proceeding to Phase B):
- `L_route` decreasing monotonically over 5 epochs (≥ 30% reduction)
- `L_time` decreasing monotonically (≥ 20% reduction)
- Validation `route_head_top1_acc` ≥ 50% (chance is ~1/7 = 14%)
- Validation `time_head_top1_acc` ≥ 35% (chance is 1/5 = 20%)
- No NaN losses; gradient norm bounded < 100

### 3.2 Phase B — freeze encoder, train Q only (15 epochs)

**Frozen:** encoder, aux heads
**Trained:** Q-network + wait MLP, target network (soft update)

**Loss:**
```
L_phase_B = L_CQL
```

**Purpose:** Q learns on stable representations. Avoids "moving target" of
both encoder and Q changing simultaneously.

**Initialization:** Q-network initialized with Xavier; target network = Q at
start of Phase B.

**Success criteria** (must pass before proceeding to Phase C):
- `L_TD` decreasing
- `Q(s, chosen_a)` mean increasing toward expected returns
- Validation `Q_argmax_top1_acc` (model's argmax = signaller's chosen action,
  within candidates) ≥ 55%
- No Q-value explosion (max |Q| < 100)
- CQL `L_cons` stable (not blowing up)

### 3.3 Phase C — joint fine-tune (20 epochs)

**Trained:** everything (encoder, aux heads, Q, target)

**Loss:**
```
L_phase_C = L_total = L_CQL + 0.5 · L_route + 0.2 · L_time
```

**Purpose:** jointly fine-tune the full network. Aux heads help maintain
representation quality while RL signal drives the Q.

**Success criteria** (final report metrics):
- Validation `Q_argmax_top1_acc` ≥ 65% on full validation set
- Per-special-case eval (per PROJECT_HANDOFF Ch 10): each of advance/call-on/
  platform_dev/late columns improves vs Phase B (or holds)
- Q magnitudes still bounded
- No catastrophic forgetting of aux head performance

### 3.4 Why 3 stages (not single-stage joint training)

Empirical observation in offline RL literature:
- Single-stage joint training often unstable when encoder is large
- "Moving target": encoder changes alter the input to Q; Q changes alter
  gradients to encoder
- Phased approach (warmup → freeze → fine-tune) typically 5-10× more stable

Our setup has ~3 M params with heavy interconnects; phased training is the
safer choice. If Phase B sanity passes easily, future v1.1 can experiment
with shorter / merged phases.

---

## §4 Data pipeline for training

### 4.1 Train/val/test split — time-based (LOCKED)

| Split | Date range | Approx samples |
|-------|------------|-----------------|
| **train** | 2023-02-28 to 2024-01-31 | ~600 k (~82%) |
| **val** | 2024-02-01 to 2024-02-29 | ~60 k (~8%) |
| **test** | 2024-03-01 to 2024-04-25 | ~70 k (~10%) |

Time-based to avoid temporal leakage (e.g., random shuffle would let model
peek at "the future of similar trains"). **Gap-aware** — no overlap between
splits.

Note: empirical observation (referenced in PROJECT_HANDOFF Ch 11 教训 6) —
random shuffle inflates accuracy 6-8 pp via implicit time leakage. Time-based
split is the only safe choice.

### 4.2 Episode construction for RL

Per spec 02 §5:
- Each (s_i, a_i, r_i, s'_i, done_i) tuple comes from one decision point
- `s'_i` = next decision point's state within the same episode (same pass_id)
- `done_i = True` for the last decision point in each episode
- For `done=True`, `s'_i` is ignored (no bootstrap)

### 4.3 Batch sampling

- **Batch size: 256** (locked; fits A100 with margin)
- **Shuffling**: within-epoch random permutation, **but stratified** (see §4.4)
- **Per-batch composition**: mix of decision points from different episodes
- **NOT episode-based batching**: each item in a batch is one (s, a, r, s')
  tuple. We don't process whole episodes per batch (CQL/IQL don't need it).

### 4.4 Stratified balanced sampling

Per PROJECT_HANDOFF Ch 10.1 (90/5/5 stratification), without rebalancing the
"trivial 90%" decisions dominate gradient and the model never learns special
cases.

**Stratum definition** (5 disjoint strata, per snapshot):

| Stratum | Condition (using `state_special_flags`) | Expected % |
|---------|------------------------------------------|------------|
| `trivial` | No special flag is True | ~85% |
| `advance` | `f_advance` is True | ~3% |
| `call_on` | `f_call_on` is True | ~1.5% |
| `platform_dev` | `f_platform_dev` is True | ~1.5% |
| `priority_compete` | `f_priority_compete` is True | ~2% |
| `late_train` | `f_late_train` > 0 | ~5% |
| `unusual_id` | `f_unusual_id` is True | ~1% |
| (priority order for overlap: late > advance > call_on > platform_dev > priority_compete > unusual_id > trivial) |  |  |

**Sampling weights:**
```
w_stratum = 1.0 / sqrt(stratum_frequency_in_train_split)
```

`sqrt` instead of inverse to avoid extreme oversampling of rare strata.

**Per-batch composition target** (256 samples per batch):
- At least 50 trivial (to maintain baseline pattern)
- At least 20 from each non-trivial stratum if available (otherwise all available)

**Implementation**: PyTorch `WeightedRandomSampler` with stratum weights, then
post-check batch composition.

### 4.5 (s, a, r, s') tuple generation

Tuples are formed at data loading time:

```python
def get_tuple(episode, i):
    s = episode[i].state
    a = episode[i].chosen_action_idx
    r = episode[i].r_total
    if i == len(episode) - 1:
        s_prime, done = None, True
    else:
        s_prime, done = episode[i+1].state, False
    return s, a, r, s_prime, done
```

For batched processing, `s_prime` for done=True rows is replaced with a
dummy state (any state; will be masked out in CQL `y` computation by `(1-done)`).

---

## §5 Optimizer + scheduler

### 5.1 AdamW (locked)

```python
optim = AdamW(
    model.parameters(),
    lr=3e-4,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=1e-4,
)
```

### 5.2 Learning rate schedule

- **Warmup**: linear from 0 → 3e-4 over first 1000 steps
- **Decay**: cosine annealing from 3e-4 to 3e-5 over remaining steps
- Per-phase reset: at Phase B start (after 5 epochs), re-warm to peak
- At Phase C start (after 20 epochs), re-warm to half peak (1.5e-4)

### 5.3 Weight decay

- 1e-4 on linear layers (encoder, MLP)
- **0 on embeddings and LayerNorm** (standard practice)

### 5.4 Gradient clipping

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

Applied at every optimizer step.

### 5.5 Mixed precision

- **bf16 if A100/H100 available** (no loss scaling needed)
- **fp16 + GradScaler** if older GPU
- **fp32 fallback** otherwise

Computational savings: ~2× speedup on A100 with no accuracy degradation
(verified in spec 03 §10.2 budget).

---

## §6 Target network (CQL only)

### 6.1 Architecture clone

Target network = identical clone of Q-network (same architecture and dims).

Created at Phase B start:
```python
q_target = copy.deepcopy(q_network)
for p in q_target.parameters():
    p.requires_grad = False
```

### 6.2 Soft update (Polyak averaging)

After every optimizer step:
```python
tau = 0.005    # locked
for p_t, p in zip(q_target.parameters(), q_network.parameters()):
    p_t.data.copy_(tau * p.data + (1 - tau) * p_t.data)
```

Effective update lag: ~200 steps half-life. Standard for CQL/IQL.

### 6.3 Encoder + target

Target network shares encoder weights with main? **No** — full clone including
encoder. This ensures target is fully stable.

Note: this means encoder forward pass happens TWICE per training step (once
for online Q, once for target Q). Cost is acceptable (~1.5× per-step time vs
no target).

---

## §7 Hyperparameter table (one-page reference)

```
╔═══════════════════════════════════════════════════════════════════╗
║  ALGORITHM                                                         ║
║    main                          = CQL                             ║
║    alt                           = IQL                             ║
║    baseline                      = BC                              ║
║                                                                     ║
║  CQL                                                               ║
║    α (conservative coef)         = 5.0                             ║
║    γ (discount)                  = 0.95   (from spec 02)           ║
║    τ_target (soft update)        = 0.005                           ║
║                                                                     ║
║  IQL                                                               ║
║    expectile τ                   = 0.7                             ║
║    AWR temperature β             = 3.0                             ║
║    τ_target (soft update)        = 0.005                           ║
║                                                                     ║
║  3-STAGE PROTOCOL                                                  ║
║    Phase A (encoder + aux)       = 5 epochs                        ║
║    Phase B (Q only, encoder frz) = 15 epochs                       ║
║    Phase C (joint fine-tune)     = 20 epochs                       ║
║    Total                         = 40 epochs                       ║
║                                                                     ║
║  OPTIMIZER                                                         ║
║    AdamW lr                      = 3e-4                            ║
║    AdamW β                       = (0.9, 0.999)                    ║
║    AdamW ε                       = 1e-8                            ║
║    Weight decay                  = 1e-4 (excl. embed/LN)           ║
║    Warmup steps                  = 1000 linear                     ║
║    LR schedule                   = cosine to 3e-5                  ║
║    Grad clip                     = 1.0                             ║
║                                                                     ║
║  BATCHING                                                          ║
║    Batch size                    = 256                             ║
║    Sampler                       = stratified by special-case      ║
║    Min trivial per batch         = 50                              ║
║    Min per non-trivial stratum   = 20                              ║
║                                                                     ║
║  AUX LOSS WEIGHTS (from spec 03)                                   ║
║    λ_route                       = 0.5                             ║
║    λ_time                        = 0.2                             ║
║                                                                     ║
║  REPRODUCIBILITY                                                   ║
║    Seeds                         = 42, 43, 44                      ║
║    Deterministic ops             = enabled where possible          ║
║    cuDNN benchmark               = False (for determinism)         ║
║                                                                     ║
║  TRAIN/VAL/TEST SPLIT                                              ║
║    train                         = 2023-02-28 to 2024-01-31        ║
║    val                           = 2024-02-01 to 2024-02-29        ║
║    test                          = 2024-03-01 to 2024-04-25        ║
║                                                                     ║
║  PRECISION                                                         ║
║    Training                      = bf16 (A100) or fp16 (older)     ║
║    Loss + gradient accumulation  = fp32                            ║
║                                                                     ║
║  COMPUTE BUDGET                                                    ║
║    Per-seed time (40 epochs)     = ~12 hours on A100               ║
║    3 seeds × 1 algorithm         = ~36 GPU-hours                   ║
║    CQL + IQL + BC × 3 seeds      = ~108 GPU-hours total            ║
╚═══════════════════════════════════════════════════════════════════╝
```

---

## §8 Logging + monitoring

### 8.1 Per-step metrics (every step)

- `loss_total`, `loss_TD`, `loss_cons`, `loss_route`, `loss_time`
- `Q_mean`, `Q_max`, `Q_min` (on current batch)
- `gradient_norm` (after clipping)
- `learning_rate`

### 8.2 Per-epoch metrics (after each epoch on val set)

- `val_Q_argmax_top1_acc` — model's argmax matches signaller's chosen action
- `val_route_head_top1_acc` — route head accuracy on set decisions
- `val_time_head_top1_acc` — time head accuracy
- `val_loss_*` — all loss components
- `val_top1_per_stratum` — top-1 accuracy on each of 7 strata
- `val_Q_distribution` — histogram

### 8.3 Early stopping criteria

- Stop early in Phase B if `val_Q_argmax_top1_acc` doesn't improve for 5 consecutive epochs
- Stop early in Phase C if `val_top1` doesn't improve for 5 consecutive epochs
- Hard abort if any loss becomes NaN

### 8.4 Checkpointing

- Save full state at end of each phase: `outputs/train/run_<seed>/phase_<A,B,C>_end.pt`
- Save best model by `val_Q_argmax_top1_acc` (Phase C only) → `best.pt`
- Save final epoch → `final.pt`
- Include: model weights, optimizer state, lr scheduler state, seed, epoch

### 8.5 Logging output structure

```
outputs/train/cql_seed42/
├── config.yaml          # full hyperparameter snapshot at start
├── git_hash.txt         # repo commit hash
├── phase_A_end.pt
├── phase_B_end.pt
├── phase_C_end.pt
├── best.pt              # best by val_top1
├── final.pt
├── metrics.jsonl        # one JSON per epoch, all metrics
├── train_loss.png       # auto-plot
├── val_acc.png
└── stratum_breakdown.png
```

---

## §9 Reproducibility

### 9.1 Random seeds

3 seeds (locked): **42, 43, 44**

For each seed, set all of:
```python
import random, numpy as np, torch
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

### 9.2 Deterministic ops

- `torch.use_deterministic_algorithms(True)` where possible
- Some PyG / HGT ops may not have deterministic backwards; document any
  exceptions in `outputs/train/run_<seed>/nondeterministic_ops.txt`
- Mixed precision can introduce small non-determinism — acceptable
  (we report mean ± std across seeds)

### 9.3 Reporting protocol

For each algorithm (CQL, IQL, BC):
- Report **mean ± std** across 3 seeds
- Also report **stratified bootstrap 95% CI** (within-class, per PROJECT_HANDOFF
  Ch 10.1 教训 4)
- Both: 3-seed std typically ≫ bootstrap CI half-width; both are reported

---

## §10 Compute budget

### 10.1 Per-seed time estimate

Based on spec 03 §10.2 + this spec's hyperparameters:

| Phase | Epochs | Time/epoch | Total |
|-------|--------|------------|-------|
| Phase A (encoder + aux) | 5 | ~10 min | ~50 min |
| Phase B (Q only) | 15 | ~15 min | ~3.75 hr |
| Phase C (joint) | 20 | ~20 min | ~6.7 hr |
| Eval + checkpointing overhead | n/a | ~5 min total | ~1 hr |
| **Total per seed** |  |  | **~12 hours on A100** |

### 10.2 Memory

- GPU memory: ~6-8 GB peak (model + optimizer + activations + target)
- Disk: ~500 MB per seed (checkpoints + logs)

### 10.3 Total experimental matrix

| Algorithm | Seeds | Time |
|-----------|-------|------|
| CQL (main) | 3 | ~36 GPU-hours |
| IQL (alt) | 3 | ~36 GPU-hours |
| BC (baseline) | 3 | ~12 GPU-hours (20 epochs only, no Q) |
| **Total** | 9 runs | **~84 GPU-hours** |

If A100 24/7: ~3.5 days. If shared cluster: plan ~2 weeks.

### 10.4 Hyperparameter sweep (optional)

If Phase B sanity fails, sweep `α ∈ {1, 5, 10, 25}` with shortened Phase B
(5 epochs each). 4 quick runs = ~5 hours total.

---

## §11 Validation + sanity checks during training

### 11.1 Phase A success gate

After Phase A (5 epochs):

```python
assert val_loss_route < 0.7 * initial_loss_route, "L_route not decreasing"
assert val_loss_time  < 0.85 * initial_loss_time, "L_time not decreasing"
assert val_route_head_top1_acc > 0.50, f"route head too weak: {acc}"
assert val_time_head_top1_acc  > 0.35, f"time head too weak: {acc}"
assert not np.isnan(latest_loss), "NaN loss in Phase A"
```

If any fails → **stop training**, debug (likely: encoder bug, padding bug,
normalization bug).

### 11.2 Phase B success gate

After Phase B (15 epochs):

```python
assert val_Q_argmax_top1_acc > 0.55, f"Q top-1 too weak: {acc}"
assert abs(Q_max) < 100.0, "Q values exploding"
assert abs(Q_min) < 100.0, "Q values exploding"
assert L_cons.mean() < 50.0, "CQL conservative loss blowing up"
```

If Q top-1 < 55%, likely: α too low (model not learning Q), or encoder
frozen with bad representations → revisit Phase A.

### 11.3 Phase C success gate

After Phase C (20 epochs):

```python
assert val_Q_argmax_top1_acc > 0.65, f"final Q top-1 too weak: {acc}"
for stratum in ['advance', 'call_on', 'platform_dev', 'late_train']:
    assert val_top1_by_stratum[stratum] > val_top1_by_stratum_after_B[stratum] - 0.02, \
        f"catastrophic forgetting on {stratum}"
```

If Phase C degrades, the joint fine-tune broke the encoder; reduce lr for
encoder params, or shorten Phase C.

---

## §12 Implementation modules

```
src/railrl/algorithms/
├── __init__.py
├── base.py                # AbstractTrainer class
├── bc.py                  # BC (baseline) — Phase A logic only
├── cql.py                 # CQL — full 3-stage
├── iql.py                 # IQL — full 3-stage
└── replay_buffer.py       # (s, a, r, s') tuple dataset wrapper

src/railrl/training/
├── data_loader.py         # stratified sampler + batch builder
├── lr_schedule.py         # warmup + cosine
├── target_net.py          # soft update helper
└── eval_during_training.py # per-epoch val metrics

scripts/train/
├── 01_train_cql_main.py        # entry: full CQL run, one seed
├── 02_train_iql.py             # entry: full IQL run, one seed
├── 03_train_bc.py              # entry: BC baseline, one seed
├── 04_multi_seed_dispatch.py   # run all 9 (CQL/IQL/BC × 3 seeds)
└── 05_aggregate_results.py     # mean ± std, bootstrap CI
```

### 12.1 Trainer base class

```python
class AbstractTrainer:
    def setup(self, config):       # build model, optimizer, scheduler, loaders
    def train_phase_A(self):       # 5 epochs encoder + aux
    def train_phase_B(self):       # 15 epochs Q only (CQL/IQL) — BC skips
    def train_phase_C(self):       # 20 epochs joint
    def eval_on_val(self):         # per-epoch metrics
    def save_checkpoint(self, name):
    def run_all_phases(self):
```

### 12.2 Config YAML

`configs/exp_cql_main.yaml`:

```yaml
algorithm: cql
seed: 42

data:
  snapshots_path: outputs/snapshots/snapshots_v2.parquet
  train_dates: [2023-02-28, 2024-01-31]
  val_dates:   [2024-02-01, 2024-02-29]
  test_dates:  [2024-03-01, 2024-04-25]
  batch_size: 256
  stratified: true
  min_trivial_per_batch: 50
  min_per_other_stratum: 20

cql:
  alpha: 5.0
  gamma: 0.95
  target_tau: 0.005

training:
  phase_A_epochs: 5
  phase_B_epochs: 15
  phase_C_epochs: 20
  optimizer: adamw
  lr: 3e-4
  weight_decay: 1e-4
  warmup_steps: 1000
  grad_clip: 1.0
  amp: bf16

aux_loss:
  lambda_route: 0.5
  lambda_time:  0.2

output:
  dir: outputs/train/cql_seed42
```

---

## §13 Failure modes + diagnostics

### 13.1 Common failure modes

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Phase A `L_route` stuck high | Padding bug; encoder not seeing actual data | Inspect a batch with `pdb`, check masks |
| Phase A `L_time` doesn't drop | Time labels mostly NaN | Re-check spec 01 §11.3 lead-time computation |
| Phase B Q-values blow up | α too low | Increase α to 10, 25 |
| Phase B Q-values too conservative (model always wait) | α too high | Decrease α to 1, 0.5 |
| Phase C catastrophic forgetting | Joint lr too high for encoder | Reduce encoder lr to 1e-5 for Phase C |
| NaN losses | Mixed precision overflow | Force fp32 for loss computation |
| 3 seeds diverge widely | Encoder unstable; bug | Single-seed deep debug |
| Stratified sampler imbalanced | Bug in stratum assignment | Print per-batch stratum histogram |

### 13.2 When to abort + redesign

- After Phase A: if `route_head_top1_acc < 30%`, encoder design likely wrong
  → revisit spec 03
- After Phase B: if `Q top1 < 40%`, fundamental issue (likely state schema)
  → revisit spec 02
- After Phase C: if best top1 < 50% on val, problem at MDP level
  → revisit spec 02 candidates / leak audit

---

## §14 Open questions for spec 05 to inherit

| # | Question | Default proposal |
|---|----------|------------------|
| 1 | Should we report Q-distribution stats (mean, P5, P95) per stratum in §VII? | Yes — adds interpretability |
| 2 | How to handle ties at argmax (multiple actions with equal Q) | Pick lowest action_idx (deterministic); document. |
| 3 | Should we save model logits for every test sample for ensemble experiments later? | Yes — `outputs/train/<run>/test_logits.npy` (per spec 05 audit Ch 11 教训 8) |
| 4 | When reporting "CQL improves on signaller", what's δ for the divergent-improving threshold? | δ = 0.5 reward units (≈ 30 sec recovered delay or equivalent). Define in spec 05. |

---

## §15 Changelog

- **v1.0 (2026-05-19)** — Initial draft. Locks CQL (α=5.0, γ=0.95, target
  τ=0.005) main + IQL (expectile τ=0.7, AWR β=3.0) alt + BC baseline,
  3-stage protocol (A=5 ep / B=15 ep / C=20 ep = 40 total), AdamW (lr=3e-4,
  cosine warmup), batch 256 with stratified sampler, time-based train/val/test
  split (2023-02-28 ... 2024-04-25), 3 seeds (42/43/44), per-phase success
  gates, ~12 hr/seed budget = ~84 GPU-hours total.

---

**End of Spec 04.**
**Sign-off:** ☐ Hao  /  Date: ______
