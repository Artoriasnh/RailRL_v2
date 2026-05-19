# P2.4 Iter D - Reward Health Check

Run on full 2.64M decisions, 82,429 episodes.

## Reward distribution
- r_total mean: +0.255  std: 0.675
- r_total range: [-30.30, +30.50]
- per-episode return mean: +2.25  std: 3.20
- frac positive episodes: 88.4%

## Component coverage (non-zero count)
- approach_distance: 261,008
- delay_change_seconds: 62,772
- next_tc_headway_seconds: 526,409
- route_outcome: 545,289

## Weight sensitivity

- **conservative** weights={'w_delay': 0.5, 'w_throughput': 0.3, 'w_headway': 1.5, 'w_wait': 0.5}
  ep_return mean/std/P1/P50/P99: +0.37 / 2.59 / -7.26 / +0.90 / +3.30
  frac_positive: 64.1%
- **default** weights={'w_delay': 1.0, 'w_throughput': 0.5, 'w_headway': 1.0, 'w_wait': 0.3}
  ep_return mean/std/P1/P50/P99: +2.25 / 3.20 / -4.90 / +2.50 / +5.50
  frac_positive: 88.4%
- **aggressive** weights={'w_delay': 1.5, 'w_throughput': 1.0, 'w_headway': 0.5, 'w_wait': 0.1}
  ep_return mean/std/P1/P50/P99: +6.09 / 4.80 / -3.38 / +6.50 / +12.51
  frac_positive: 97.8%

**Spearman rank correlations across weight presets** (closer to 1.0 = policy ordering robust to weight choice):
- conservative_vs_default_spearman: 0.908
- conservative_vs_aggressive_spearman: 0.588
- default_vs_aggressive_spearman: 0.842

## Proxy correlations
- Spearman(return, no_cancellation): +0.055
- Spearman(return, use_rate):        +0.117
- Mean return WITH cancellation:    +1.01
- Mean return WITHOUT cancellation: +2.26

**Interpretation of weak proxy correlations**: 99.5% of episodes have
no cancellations and use_rate ~ 1 (signaller is highly effective at
route-setting). The `no_cancel` and `use_rate` proxies are therefore
near-constant across episodes and cannot differentiate them well
(flat distribution -> low Spearman). This does NOT indicate the reward
is mis-specified; rather, our naive proxies are insufficient. A better
proxy would require external operational KPIs (PPM, CaSL, headway
violation reports) which are out of scope for the current dataset.