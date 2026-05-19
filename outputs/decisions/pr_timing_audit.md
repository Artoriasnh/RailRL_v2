# PR timing audit

PRs total: **546,418**
  - with measurable lead time (train seen recently): 543,328
  - no recent track event (likely pre-set very early): 3,090

## Lead time distribution (seconds)
  P10/P50/P90/P99 = 3 / 39 / 85240 / 1726870

## Lead time buckets
  - PR fires AFTER train passed (latency / unusual): 0 (0.0%)
  - 0-60s    (immediate, train in approach): 304,505 (56.0%)
  - 1-10 min (train approaching): 146,274 (26.9%)
  - 10-30 min (early set, in network): 15,922 (2.9%)
  - 30 min - 2h (pre-set, before in-network): 1,687 (0.3%)
  - > 2h     (way pre-set or stale data): 74,940 (13.8%)

## Batch PR (other PRs within ±60s of each PR)
  - alone (no other in 60s):  20,913
  - small burst (1-2):        128,429
  - burst 3-9:                375,031
  - burst 10+:                22,045