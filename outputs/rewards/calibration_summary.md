# P2.4 Iter A — Reward Threshold Calibration

Empirically derived from full-year Derby workstation data.

## H_min — minimum acceptable headway (r_headway)

| Statistic | Value |
|-----------|-------|
| n pairs | 3,284,641 |
| P1  | 91.0 s |
| **P5 (= H_min)** | **147.0 s** |
| P10 | 206.0 s |
| P50 | 852.0 s |
| P90 | 2940.0 s |
| P99 | 21517.0 s |

Reference values from UK railway standards (paper context):
- Multi-aspect colour-light mainline minimum signaling headway: 90-120s (Network Rail Industry Standard RIS-0786-RIG).
- Junction headway: 90-150s typical.
- TPWS overlap clearance: 30-45s.

## d-gate — causal-attribution distance for r_delay

| Statistic | Value |
|-----------|-------|
| n decisions sampled | 50,000 |
| n with computable d | 24,870 |
| P10 | 1.0 hops |
| **P50 (= gate-0.5 boundary)** | **6.0 hops** |
| **P90 (= gate-0.1 boundary)** | **16.0 hops** |
| P95 | 18.0 hops |
| P99 | 23.0 hops |

Decided d-gate function:

| Distance d at decision | gate(d) | Interpretation |
|------------------------|---------|----------------|
| 0-2 | 1.0 | Train is here NOW; this PR fully responsible |
| 3-6 | 0.5 | Approaching; partial responsibility |
| 7-16 | 0.1 | Far; minimal responsibility |
| > 16 | 0.0 | Pre-staging; not responsible |

## Reward observation window — TIPLOC-lag P99

| Statistic | Value |
|-----------|-------|
| n lags | 42,806 |
| P50 | 228.0 s |
| P90 | 3756.0 s |
| P95 | 3864.0 s |
| **P99 (= window)** | **4201.9 s** |

## Final calibrated parameters

```json
{
  "H_min_seconds": 147.0,
  "d_gate_0.5_max": 6,
  "d_gate_0.1_max": 16,
  "reward_window_seconds": 4201.9
}
```