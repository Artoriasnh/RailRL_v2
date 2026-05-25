# RailRL v2 — End-to-End Offline RL for UK Railway Signaller Decisions

Research codebase for an interpretable expert system on the Network Rail **Derby
workstation** (14 months of open data, ~2.0M decision points). Target publication: **ESWA**.

The signaller's job is framed as an offline-RL **scheduling game** with a structured
joint action `A_t = {wait} ∪ {(focal_train, route)}`. A heterogeneous-graph + event
sequence state is encoded and scored by a per-action Q-network (CQL), with auxiliary
supervised heads, and decisions are made interpretable across five levels.

> **Status (2026-05): Stages 0–5 complete; Stage 6 (full 3-seed CQL training) in progress.**
> The full ship's log is `docs/IMPLEMENTATION_LOG.md`; a one-page index is `docs/CHANGELOG.md`.

## Status by stage

| Stage | Description | Status |
|---|---|---|
| 0 | Spec lock (5 contract docs) | ✅ done |
| 1 | Data pipeline + environment | ✅ done |
| 2 | Decision points + candidate actions + 8 specialness flags | ✅ done |
| 3 | Snapshot builder (state + leak audit + episodes) | ✅ done |
| 4 | Model (HGT + Transformer + per-action Q + 2 aux heads) + CQL 3-phase training | ✅ done |
| 5 | 50k sanity training (all spec §11 gates passed) | ✅ done |
| **6** | **Full 3-seed CQL training (42/43/44)** | 🔨 **in progress** |
| 7 | Baselines (random / FCFS / BC-MLP / BC) | ⏳ pending |
| 8 | Evaluation (3-tier + Replicate-AND-Improve) | ⏳ pending |
| 9–11 | XAI five levels + rule base + simulator + selective override | ⏳ pending |
| 12 | Paper | ⏳ pending |

## Three contributions (ESWA structure)

| Paper section | Contribution | Code area |
|---|---|---|
| §3 + §4 | Data acquisition + engineering (open feeds → MDP tuples) | `src/railrl/data/`, `src/railrl/mdp/`, `scripts/{data,mdp}/` |
| §5 | End-to-end RL framework: joint route + timing learning | `src/railrl/{encoders,policies,algorithms}/`, `scripts/train/` |
| §6 + §7 + §8 | Five-level interpretability + Replicate-AND-Improve + selective override | `src/railrl/{eval,xai}/`, `scripts/{eval,figures,simulator}/` |

## Model & training (locked, spec 03/04)

- **State** — heterogeneous graph (4 node types: track/signal/route/train × 8 edge types)
  + K=256 event-token sequence + 15-min schedule outlook + 8 specialness flags.
- **Encoder** — 3-layer HGT (d=128) ⊕ 4-layer Transformer over event tokens ⊕ fusion → `s_emb` (256).
- **Heads** — per-action Q-network (`{wait} ∪ candidate routes`) + auxiliary route classifier
  + 5-bucket lead-time head. ~3.0M parameters.
- **Algorithm** — **CQL** (α=5, γ=0.95) with a target network (soft τ=0.005); IQL + BC available.
- **3-phase protocol** — A: encoder + aux (5 ep) → B: freeze encoder, train Q (15 ep)
  → C: joint fine-tune (20 ep). AdamW lr 3e-4, warmup→cosine, grad-clip 1.0, batch 256.
- **Loader** — `StreamingTransitionDataset`: decode-once, block-shuffled, worker-safe,
  block-level stratified sampling (spec §4.4).

## Data artefacts (the canonical training input)

`outputs/snapshots/snapshots_v2.parquet` — **1,996,572 rows**, canonical order (by
`episode_idx, position`), carrying per-row identity / **real 4-component reward** /
episode + split columns / state (graph + sequence + outlook + flags). Sidecars:
`episodes_v2.parquet` (split + episode metadata), `pass_split.parquet`,
`stratum_labels.parquet` + `stratum_weights.json`, `time_labels_v2.parquet`,
`normalization_stats.json`. Large binaries (`*.parquet/*.pt/*.tiff/*.png`) are git-ignored.

**Key invariants** (do not break): `sample_id` is the physical row id reward/state align to;
embedding vocab track 268 / signal 123 / route 278 / train 2184; **time-based split**
(train `<2024-02-01` / val `<2024-03-01` / test `≥`, assigned by episode start time, leak-free).

## Project layout

```
RailRL_v2/
├── data/{raw,reference,domain}/         # TD_data.csv, Movements.csv, route_to_tc, refs (raw not in git)
├── outputs/                             # generated artefacts (big binaries git-ignored)
│   ├── snapshots/                       # snapshots_v2.parquet (+ episodes_v2 / pass_split / stratum / time_labels / norm stats)
│   ├── decisions/ infrastructure/ static_graph/ event_stream/  # Stage 1-3 pipeline outputs
│   ├── rewards/                         # calibration + decision_rewards_v2 + pr_outcomes_v2 + summaries
│   ├── train/                           # checkpoints + train_log (json kept, *.pt ignored)
│   └── _legacy_v1_binary/               # archived v1 (reference only)
├── docs/
│   ├── CHANGELOG.md                     # one-page implementation index (read first)
│   ├── IMPLEMENTATION_LOG.md            # full append-only ship's log (decisions / bugs / lessons)
│   ├── TOOL_TRAPS.md  LEAK_AUDIT.md     # environment traps; leakage audit (all passed)
│   ├── spec/01-05_*.md                  # 5 locked contracts (data / MDP / model / training / XAI+eval)
│   └── manuscript_*.{md,docx}           # paper drafts
├── src/railrl/
│   ├── data/  mdp/                      # ingest/infra/reward; decision points/state/episode/reward_v2/flags
│   ├── encoders/  policies/  algorithms/# HGT+seq+fusion; Q-net+heads; losses+trainer+transitions
│   └── eval/  xai/                      # metrics; (XAI pending)
├── scripts/{data,mdp,train,eval,figures,simulator}/   # numbered CLI entry points
└── pyproject.toml  tests/
```

## Quickstart

Heavy steps run on GPU (RTX 5070 / A100). The sandbox cannot run torch or read the big
parquets — see `docs/TOOL_TRAPS.md`.

```bash
# 1. (re)build the canonical training data — see scripts/mdp/ (decision points → snapshots →
#    reward recompute → episode resegment → canonical resort → lateness/platform_dev patch →
#    stratum labels). Order + commands are in docs/IMPLEMENTATION_LOG.md.
python scripts/train/00_build_time_split.py
python scripts/train/01_build_normalization_stats.py

# 2. Sanity (Stage 5): streaming + stratified, ~50k rows/epoch, full 3-phase, §11 gates
python scripts/train/09_train.py --sanity

# 3. Full run (Stage 6), per seed; --resume is safe across 12h windows
python scripts/train/09_train.py --seed 42 --out outputs/train/cql_seed42 --num-workers 16 --resume
```

## Where to read more

`docs/CHANGELOG.md` (index) → `docs/IMPLEMENTATION_LOG.md` (full log) → `docs/spec/01-05_*.md`
(contracts) → `docs/LEAK_AUDIT.md` (validity threats). New collaborators: start from
`docs/NEW_CONVERSATION_PROMPT.md`.

## Known limitations (paper validity threats)

Data-coverage limits inherited from the open feeds (documented, not bugs):
`approach_distance` defined on ~48% of set decisions, `delay_change` on ~6% (TRUST/Movements
sparsity), `end_platform_id` mapped on ~28% of routes (many through/depot routes legitimately
have none → `f_platform_dev` is conservative / under-detecting). High imitation accuracy is
explained by task imitability (near-FCFS + timetable + small action set), not leakage — the
leak audit passes all checks (`docs/LEAK_AUDIT.md`).
