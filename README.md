# RailRL v2 — End-to-End Offline RL for UK Railway Signaller Decisions

Research codebase for the Derby workstation expert system. Target publication: ESWA.

This is a **clean restart** of the RailRL v1 codebase, with the following changes:

- **Structured joint action space** (replaces v1 binary set/wait): `A_t = {wait} ∪ {(train, route)}`
- **No `focal_signal` leakage** in state (the cause of v1's 91% trivial baseline)
- **Per-node graph state** (replaces v1's mean/std aggregation)
- **Three-decision joint learning** via single Q-function + auxiliary supervised heads
- **Five-level interpretability** as a first-class concern, not afterthought
- **ESWA-aligned section structure**: Data Pipeline → RL Framework → Interpretable Decisions

## Three contributions (ESWA paper structure)

| Section | Contribution | Code area |
|---|---|---|
| **§3 + §4** | Data acquisition + engineering pipeline (open Network Rail feeds → MDP tuples) | `src/railrl/data/` |
| **§5** | End-to-end RL framework jointly learning route + timing + priority | `src/railrl/mdp/`, `encoders/`, `policies/`, `algorithms/` |
| **§6 + §7 + §8** | Five-level interpretability + Replicate-AND-Improve evaluation + selective override | `src/railrl/xai/`, `eval/`, `deploy/` |

## Project layout

```
RailRL_v2/
├── data/
│   ├── raw/                           # TD_data.csv, Movements.csv, route_to_tc_all.csv (copied from v1)
│   ├── reference/                     # Static reference (5 small CSVs + SOP + Derby_all.png)
│   └── domain/                        # Training Plan + signalling PDFs
│
├── outputs/                           # Generated artefacts
│   ├── inventory/                     # TD + Movements statistics
│   ├── decisions/                     # PR events with chosen_route_id (action labels)
│   ├── infrastructure/                # 277 routes / 249 tracks / 100 signals + graph.json
│   ├── static_graph/                  # 4 node types × 6 edge types (parquet)
│   ├── event_stream/                  # K=256 token stream from TD `change` column
│   ├── rewards/                       # H_min=147s calibration + 4-component decision_rewards
│   ├── analyses/                      # 3 empirical analyses (conflict / route-class / non-std IDs)
│   ├── cache/                         # td_data.parquet (parsed cache, 90 MB)
│   └── _legacy_v1_binary/             # Archived v1 binary task outputs (DO NOT USE)
│
├── docs/
│   ├── spec/                          # 5 canonical spec docs (write before code)
│   │   ├── 01_data_pipeline.md        # (TODO)
│   │   ├── 02_mdp_formulation.md      # (TODO) — joint 3-decision action space
│   │   ├── 03_model_architecture.md   # (TODO) — HGT + Transformer + Q-network + aux heads
│   │   ├── 04_training_protocol.md    # (TODO) — CQL 3-stage, IQL alt, BC baseline
│   │   └── 05_xai_and_eval.md         # (TODO) — 5-level XAI + Replicate-AND-Improve
│   ├── phase2_feature_spec.md         # Canonical state-feature spec (carried from v1)
│   └── handoff/                       # Reference docs (v3 proposal + Phase1 inventory)
│
├── src/railrl/
│   ├── __init__.py / config.py / parsers.py / data_io.py    # Shared (carried from v1)
│   ├── data/                          # Layer 1: ingest + infra + reward (carried + extended)
│   ├── mdp/                           # Layer 2: state/action/episode/trigger (NEW)
│   ├── encoders/                      # Layer 3: HGT + sequence + fusion (NEW)
│   ├── policies/                      # Layer 4: Q-network + 3 aux heads (NEW)
│   ├── algorithms/                    # Layer 5: BC, CQL, IQL (NEW)
│   ├── eval/                          # Layer 6: metrics + Replicate-AND-Improve (NEW)
│   └── xai/                           # Layer 7: L1-L5 explanations (NEW)
│
├── scripts/                           # Numbered CLI entry points
│   ├── data/01-15_*.py                # Data pipeline scripts (carried, runnable today)
│   ├── data/analyses/                 # 3 empirical analyses
│   ├── mdp/                           # (NEW) decision-point + snapshot rebuild
│   ├── train/                         # (NEW)
│   ├── eval/                          # (NEW)
│   └── xai/                           # (NEW)
│
├── configs/                           # YAML experiment configs (NEW)
├── tests/                             # pytest suite
└── pyproject.toml                     # Package metadata + deps
```

## What's reusable from v1 (no rewrite)

All P2.1-P2.4 outputs are bit-identical copies from v1:

- `data/raw/TD_data.csv` (713 MB, MD5-verified head + tail)
- `data/raw/Movements.csv` (49 MB)
- `outputs/decisions/decision_events.parquet` — **action labels source**
  (`chosen_route_id` column is the v2 ground truth)
- `outputs/static_graph/` — 4 node types × 6 edge types (HGT input)
- `outputs/event_stream/event_tokens.parquet` — K=256 sequence input
- `outputs/rewards/decision_rewards.parquet` — 4-component reward already calibrated

## What v2 will redo

- `outputs/_legacy_v1_binary/snapshots/` — schema is wrong for v2 (115-dim flat with
  focal_signal leakage). v2 will rebuild snapshots from raw inputs.
- `outputs/_legacy_v1_binary/decision_points/` — binary label `{set, wait}` will be
  replaced with `(train, route)` tuples.
- `outputs/_legacy_v1_binary/p3_modelling/` — B0/B1/B2/B3 trained on binary task,
  archived as paper §VII counter-example only.

## Status

| Phase | Status |
|---|---|
| **Spec docs** | ⏳ in progress (5 docs to write) |
| **Data pipeline** | ✓ runnable (carried from v1) |
| **MDP + state builder** | ⏳ next |
| **Model + training** | ⏳ pending |
| **XAI + evaluation** | ⏳ pending |
| **Paper draft** | ⏳ pending |

See `docs/spec/` for the canonical design once those docs are written.
