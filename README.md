# RailRL v2 — End-to-End Offline RL for UK Railway Signaller Decisions

Research codebase for an interpretable expert system on the Network Rail **Derby
workstation** (14 months of open data, ~2.0M decision points). Target publication: **ESWA**.

The signaller's job is framed as an offline-RL **scheduling game** with a structured
joint action `A_t = {wait} ∪ {(focal_train, route)}`. A heterogeneous-graph + event
sequence state is encoded and scored by a per-action Q-network (CQL), with auxiliary
supervised heads, and decisions are made interpretable across five levels.

> **Status (2026-05-27): modeling + evaluation + interpretability essentially complete.**
> 3-seed CQL training done; all 5 XAI layers + L4 rule-compliance + §12 selective-override built and
> evaluated on seed42. **Single source of truth for results → `docs/RESULTS_SUMMARY.md`**; full ship's
> log `docs/IMPLEMENTATION_LOG.md`; one-page index `docs/CHANGELOG.md`; new-conversation handoff
> `docs/NEW_CONVERSATION_PROMPT.md`.

## Status by stage

| Stage | Description | Status |
|---|---|---|
| 0–5 | Spec lock → data pipeline → decision points + 8 flags → snapshots + leak audit → model + CQL → 50k sanity | ✅ done |
| 6 | Full **3-seed** CQL training (42/43/44; best val .9823 / .9832 / .9830) | ✅ done |
| 7 | Baselines — non-learned B0/B0'/B0'' (Table I) ✅; learned **BC-HG ✅, IQL 🔨 running**; BC-flat deferred | ◑ partial |
| 8 | Evaluation — Tier-1/2 + Tier-3 (safety-first) + OPE/FQE + P2.6 simulator (seed42) | ✅ done |
| 9–11 | XAI — **L1** IG-saliency · **L2** Q-gap SHAP · **L3** counterfactual · **L4** rule-compliance · **L5** MaxEnt-IRL | ✅ done |
| 12 | **§12 Selective Override** (δ_L3 + L4 + L2 gates → agreement / consider-override / silent) | ✅ done |
| — | Paper (drafts only — not AI-written, per user) | ⏳ ongoing |

> **Deferred (recorded in RESULTS_SUMMARY §11, not blocking):** L1 attention-rollout + panel heatmap
> (needs HGTConv attention hook + manual TC→pixel map), **3-seed eval mean±std** (eval currently seed42),
> learned baselines completion (IQL run + BC-flat), L5 reward-recovery refinement.

## Headline results (seed42 test, set-only unless noted)

- **Imitation** — CQL action top-1 **95.7%** (set-only); crushes non-learned baselines on the hard strata
  (call_on 88% vs ≤5%, platform_dev 90% vs 0%). **BC-HG ≈ CQL on accuracy (95.0 vs 95.7)** — on this
  near-deterministic task imitation alone is strong; CQL's value-add is the calibrated Q-function (OPE /
  counterfactual / override / OOD-safety), not raw accuracy (an honest finding; reframes the BC-vs-RL story).
- **Policy value (OPE/FQE)** — total ΔV ≈ 0 (delay-neutral, wait improved, throughput tiny cost): the model
  matches the expert overall and slightly reduces waiting. Sparse reward under-weights delay (reward-design
  finding, corroborated by L5-IRL: the signaller prioritizes delay).
- **Safety (Tier-3)** — 0% genuine-unsafe divergences; conflict-neutral.
- **Rule compliance (L4)** — model 81.0% vs signaller 85.7% on hard Plan rules (concentrated in call_on);
  both deviate ~15-20% (the Plan is guidance, not law).
- **Selective override (§12)** — agreement 95.7% (set); consider-override only ~0.2% (robust to δ_L3) — the
  model rarely has a strong, rule-safe, faithful reason to override the expert → respects experience.

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
│   ├── NEW_CONVERSATION_PROMPT.md       # AI entry point (start here)
│   ├── RESULTS_SUMMARY.md               # single source of truth for all results
│   ├── CHANGELOG.md  IMPLEMENTATION_LOG.md   # one-page index + full append-only ship's log
│   ├── TOOL_TRAPS.md  LEAK_AUDIT.md     # environment traps (§1-23); leakage audit (all passed)
│   ├── spec/01-05_*.md                  # 5 locked contracts (data / MDP / model / training / XAI+eval)
│   └── manuscript_*.{md,docx}  PROJECT_HANDOFF.docx   # paper drafts + high-level handoff
├── src/railrl/
│   ├── data/  mdp/                      # ingest/infra/reward + rule_base; decision points/state/episode/reward_v2/flags
│   ├── encoders/  policies/  algorithms/# HGT+seq+fusion; Q-net+heads; losses(cql/iql/bc)+trainer+transitions
│   └── eval/  xai/  deploy/             # metrics; L1-L5 (l1_attention/l2_qdecomp/l3_system/l4_rules/l5_irl); selective_override
├── scripts/{data,mdp,train,eval,rules,simulator}/     # numbered CLI entry points
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

# 4. Learned baselines (Table I B2/B3): --algo {cql,bc,iql}; --max-batches speeds up the loader-bound epochs
python scripts/train/09_train.py --algo bc  --seed 42                 # B2 BC-HG (20ep supervised)
python scripts/train/09_train.py --algo iql --seed 42 --max-batches 3000 --resume   # B3 IQL (3-phase)

# 5. Evaluation + XAI (after training) — exact commands + expected numbers in docs/RESULTS_SUMMARY.md
python scripts/eval/01_evaluate_model.py --seed 42                    # Tier-1/2 per-stratum (set-only)
#   06 baselines · 03 Tier-3 · 04/05 OPE-FQE · 07 L2 · 08/09 L5 · 10 L1 · 12 L4 · 13 §12-override
#   scripts/rules/03_finalize.py → outputs/rule_base/rules.parquet (19 Hao-approved rules)
```

## Documentation map — read in this order (importance-ranked)

| # | Doc | What it is | Freshness |
|---|---|---|---|
| 1 | `docs/NEW_CONVERSATION_PROMPT.md` | **AI entry point** — paste into a new session; project overview + reading order + work discipline | current |
| 2 | `docs/RESULTS_SUMMARY.md` | **Single source of truth for results** — every verified number, run-status (full vs smoke), remaining-work plan | current |
| 3 | `docs/CHANGELOG.md` | One-page implementation index / roadmap (what was built, when, why) | current |
| 4 | `docs/IMPLEMENTATION_LOG.md` | Full append-only ship's log — every decision / bug / fix / lesson (most detailed) | current |
| 5 | `docs/spec/01-05_*.md` | The 5 locked contracts (data / MDP / model / training / XAI+eval) — design ground truth | locked |
| 6 | `docs/TOOL_TRAPS.md` | Environment / tooling traps (§1–§23) — read before debugging weird failures | current |
| 7 | `docs/LEAK_AUDIT.md` | Leakage audit (all checks pass) — paper validity-threats material | stable |
| — | `docs/manuscript_*.{md,docx}`, `docs/PROJECT_HANDOFF.docx` | Paper drafts + high-level handoff (not AI-maintained) | drafts |

This README is the **human** entry point; `NEW_CONVERSATION_PROMPT.md` is the **AI** entry point. There is
**no separate "working-tree guide" file** — the *Project layout* tree above plus this doc map serve that role.

## Known limitations (paper validity threats)

Data-coverage limits inherited from the open feeds (documented, not bugs):
`approach_distance` defined on ~48% of set decisions, `delay_change` on ~6% (TRUST/Movements
sparsity), `end_platform_id` mapped on ~28% of routes (many through/depot routes legitimately
have none → `f_platform_dev` is conservative / under-detecting). High imitation accuracy is
explained by task imitability (near-FCFS + timetable + small action set), not leakage — the
leak audit passes all checks (`docs/LEAK_AUDIT.md`).
