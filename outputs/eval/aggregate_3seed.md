# 3-seed test-set aggregation

Aggregated across seeds **[42, 43, 44]**.

## Table I ¡ª set-only top-1 (mean ¡À std across seeds)

| stratum | CQL | BC | IQL |
|---|---|---|---|
| **overall** | 96.01% ¡À 0.21 | 91.78% | 94.09% |
| **late_train** | 97.27% ¡À 0.18 | 93.39% | 95.79% |
| **advance** | 93.43% ¡À 1.29 | 84.15% | 86.36% |
| **call_on** | 89.11% ¡À 0.68 | 79.60% | 82.03% |
| **platform_dev** | 89.61% ¡À 0.92 | 77.27% | 86.36% |
| **priority_compete** | 93.09% ¡À 0.42 | 86.33% | 89.91% |
| **unusual_id** | 79.49% ¡À 1.81 | 73.08% | 80.77% |
| **trivial** | 97.56% ¡À 0.06 | 95.14% | 96.85% |

_Missing seeds per algo:_
- **cql**: none ¡ª all present
- **bc**: seed 43, seed 44
- **iql**: seed 43, seed 44

## OPE / FQE ¡ª ¦¤V vs signaller (CQL only)

Primary = fresh-init multi-key FQE (05). Warm-start total (04) shown for transparency but under-converges delay so its total is biased low.

| component | ¦¤V (mean ¡À std) | note |
|---|---|---|
| **total (fresh-init, 05) ¡ª PRIMARY** | 0.042 ¡À 0.025 | headline |
| total (warm-start, 04) | -0.024 ¡À 0.006 | reference |
| delay | -0.008 ¡À 0.020 | per-component |
| throughput | -0.020 ¡À 0.006 | per-component |
| headway | 0.011 ¡À 0.008 | per-component |
| wait | 0.054 ¡À 0.013 | per-component |
| ¦² components | 0.038 ¡À 0.008 | ¦²-check |
| fit_residual | 0.244 ¡À 0.007 | quality |

## L4 ¡ª hard-rule compliance (CQL only)

| | mean ¡À std |
|---|---|
| model | 85.05% ¡À 2.90 |
| signaller | 85.72% ¡À 0.00 |

## ¡ì12 Selective Override (PRIMARY ¦Ä_L3=0.5 + refined gate_l4)

| metric | mean ¡À std |
|---|---|
| agreement (set-only) | 96.01% ¡À 0.21 |
| consider-override | 0.22% ¡À 0.14 |
| silent | 99.78% ¡À 0.14 |

## Raw values per seed

Stored in `aggregate_3seed.json` under `by_algo.<algo>.tier2.<stratum>.acc_set.values`.