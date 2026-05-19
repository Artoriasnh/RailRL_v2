# Threshold Sensitivity Analysis (DECISION_LOOKAHEAD_SECONDS)

Empirical reaction time distribution: P25=28s, P50=72s, P75=197s, P90=361s.

| threshold | n_total | n_set | n_wait | wait/set | set pct |
|-----------|---------|-------|--------|----------|---------|
| **30s** | 2,638,150 | 545,289 | 2,092,861 | 3.84 | 20.7% |
| **75s** | 2,574,143 | 545,289 | 2,028,854 | 3.72 | 21.2% |
| **120s** | 2,545,001 | 545,289 | 1,999,712 | 3.67 | 21.4% |

## Recommendation

Adopt 120s for production. Above median (72s) and P67-ish, capturing
genuine set decisions while excluding the long-tail hesitation cases.