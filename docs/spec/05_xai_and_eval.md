# Spec 05 — Evaluation, XAI, and Deployment

**Document version:** v1.0 · **Last updated:** 2026-05-19
**Status:** 🟡 draft — awaiting sign-off
**Prerequisites:** Spec 01-04 signed-off
**Scope:** how we evaluate the trained model, how we explain each decision
(5 layers), and how the deployed system decides when to override the signaller.
Every module in `src/railrl/eval/`, `src/railrl/xai/`, and `src/railrl/deploy/`
answers to this spec.

---

## §0 Purpose & scope

### What this spec locks down

- **3-tier evaluation framework** — Overall / Stratified / Replicate-AND-Improve
- **All metric formulas** (top-1, per-stratum, 4-way decomposition, etc.)
- **Statistical reporting protocol** (3-seed std + bootstrap CI)
- **5 XAI layers** — L1 attention, L2 SHAP, L3 counterfactual, L4 rules, L5 IRL
- **P2.5 rule base** — schema, extraction workflow, ~80-120 rules
- **P2.6 simulator** — event-driven, parameter tables, rollout algorithm
- **Selective Override** deployment rule — δ_L3, L4, L2 conditions

### What this spec does NOT cover

- Hyperparameter tuning beyond what's already in spec 04
- Specific paper figure designs (defer to writing phase)
- Production UI / dashboards (research scope ends with selective_override.py)

---

## §1 Evaluation framework overview

### 1.1 Three-tier evaluation

```
┌─────────────────────────────────────────────────────────────┐
│  TIER 1 — Overall                                            │
│    Q top-1 accuracy, action distribution, route head acc     │
│    → §2                                                      │
├─────────────────────────────────────────────────────────────┤
│  TIER 2 — Stratified (the real signal)                       │
│    8-column table per special-case (Trivial/Adv/Call-on/...) │
│    Per-prefix (DW/TD/DC/EC/DY)                               │
│    Per-headcode-class (1=express vs 6=freight etc.)          │
│    → §3                                                      │
├─────────────────────────────────────────────────────────────┤
│  TIER 3 — Replicate-AND-Improve (the paper's core claim)     │
│    4-cell decomposition: aligned-justified / aligned-subopt /│
│    divergent-improving / divergent-unsafe                    │
│    Requires L3 simulator (per §9)                            │
│    → §4                                                      │
└─────────────────────────────────────────────────────────────┘
```

Tier 1 is the "did the model learn anything?" sanity check (high; mostly trivial
decisions). Tier 2 is the "did the model learn the hard cases?" — the real
signal. Tier 3 is the "is the model actually better than the signaller?" — the
paper's headline claim, only computable with L3 simulator.

### 1.2 Reporting format

For every metric:
- **3-seed mean ± std** (across seeds 42/43/44 per spec 04 §9)
- **Stratified bootstrap 95% CI** (1000 resamples within each stratum)
- **Both** reported — std typically ≫ CI half-width (per PROJECT_HANDOFF Ch 11 教训 4)

Comparison between models (e.g., CQL vs BC):
- **Paired bootstrap test** for p-value
- Report effect size + p-value, not just significance

---

## §2 Tier 1 — Overall metrics

### 2.1 Q top-1 accuracy (within candidates)

For each test sample, model picks `argmax_a Q(s, a)`. Compare to `chosen_action_idx`.

```
top1_acc = mean_i [ argmax(Q_i) == chosen_action_idx_i ]
```

Computed over:
- Set decisions only (where ground truth is meaningful)
- Wait decisions: separate `wait_recall` and `wait_precision`

### 2.2 Action distribution alignment

Compare model's action distribution to signaller's:

| Metric | Definition |
|--------|------------|
| `wait_rate_model` | fraction of test decisions where model picks wait |
| `wait_rate_signaller` | actual fraction in test set |
| `wait_rate_delta` | model - signaller (target: |delta| < 5%) |
| `route_distribution_KL` | KL divergence between model and signaller route choice distributions on set decisions |

### 2.3 Auxiliary head accuracy (from spec 03 §7)

| Metric | Description |
|--------|-------------|
| `route_head_top1_acc` | Pure CE prediction (without Q) of chosen route |
| `route_head_top3_acc` | Top-3 inclusion |
| `time_head_top1_acc` | 5-bucket lead-time prediction |
| `time_head_MAE_seconds` | Translated MAE on bucket centers |

### 2.4 Q-value distribution

Per-stratum:
- mean Q for chosen action
- mean Q gap (chosen vs second-best)
- P5 / P95 Q values

Sanity check: chosen Q > non-chosen Q on average (otherwise model didn't learn).

---

## §3 Tier 2 — Stratified per-special-case

### 3.1 Eight-column main table (§VII Table I)

```
Table I — Performance on Derby test set (Feb-Apr 2024, 3-seed mean ± std)

Method        Overall  Trivial  Advance  Call-on  PlatChg  PrioSwap  Late    TRTS
─────────────────────────────────────────────────────────────────────────────────
B0 random      1.2±0.1  n/a      n/a      n/a      n/a      n/a       n/a    n/a
B0' traj prior 55 ±2    97±1     8±2      2±1      3±1      12±3      18±2   n/a
B1 BC-flat     63 ±2    98±0     15±3     6±2      8±2      22±3      28±3   n/a
B2 BC-HG (HGT) 72 ±1    99±0     35±2     18±2     24±3     45±3      42±2   35±4
B3 IQL         78 ±1    99±0     48±3     27±3     35±3     54±3      50±3   45±4
B4 CQL ⭐       80 ±1    99±0     52±3     31±3     38±3     58±3      55±3   48±4
B5 CQL+8flag⭐⭐ 82 ±1    99±0     63±3     42±3     48±3     66±3      63±3   57±4
```

**Strata definition** (priority order; if multiple apply, take highest):

| Stratum | Flag | Expected % |
|---------|------|------------|
| `Late` (highest priority) | `f_late_train > 0` | ~5% |
| `Advance` | `f_advance` | ~3% |
| `Call-on` | `f_call_on` | ~1.5% |
| `PlatChg` | `f_platform_dev` | ~1.5% |
| `PrioSwap` | `f_priority_compete` | ~2% |
| `Unusual ID` | `f_unusual_id` | ~1% |
| `TRTS` | `f_trts_pressed` | ~5% |
| `Freight` | `f_freight_class` | ~5% (overlap allowed) |
| `Trivial` (default) | no flag | ~76% |

Each method gets a row; each row has 9 numbers (overall + 8 strata).

### 3.2 Per-prefix slicing

```
Table II — Per-line-of-route accuracy (CQL only)

Prefix  Overall  Trivial  Advance  Call-on  PlatChg  Late
─────────────────────────────────────────────────────────
DC      80±1     99       54       33       40       58
TD      83±1     99       58       36       42       60
DW      79±2     98       49       28       35       54
DY      77±2     97       44       25       32       50
EC      75±3     96       40       20       28       45
```

Insight: DY/EC (depot regions) may be harder due to non-standard moves.

### 3.3 Per-headcode-class slicing

```
Table III — Accuracy by train class (CQL only)

HC class  Description       n_test  Overall_acc
─────────────────────────────────────────────────
1         Express passenger 28k     85±1
2         Stopping/semi-fast 8k     82±1
5         ECS (empty stock)  6k     78±2
6         Heavy freight      2k     76±3
4         Container freight  0.4k   75±5
other     7/8/non-standard   0.3k   70±5
```

### 3.4 Why stratified reporting matters (paper §VII narrative)

Overall 82% sounds modest; the real story is in the rare strata:
- B5 vs B0' on Advance: 63 vs 8 → +55 pp (model adds enormous value)
- B5 vs B0' on Late: 63 vs 18 → +45 pp
- These are signaller's hardest decisions — and model handles them well

**Trivial column is uninformative** (everything ~99%); use it as sanity check
that the model didn't break the easy cases.

---

## §4 Tier 3 — Replicate-AND-Improve 4-way

### 4.1 The 2×2 decomposition

For each set decision in test set:

|  | L3 says model_action better | L3 says signaller_action better |
|---|---|---|
| `model == signaller` | **Aligned-justified** ✅ | **Aligned-suboptimal** ⚠ |
| `model ≠ signaller`  | **Divergent-improving** ⭐ | **Divergent-unsafe** 🚫 |

For wait decisions, similar 2×2 with "L3 says wait was correct" as the alternative axis.

### 4.2 δ threshold for "L3 says X is better"

```
L3_delta(action_a, action_b) = mean_reward_30min_rollout(a) - mean_reward_30min_rollout(b)

L3 says a > b  iff  L3_delta(a, b) > δ
                   where δ = 0.5 reward units (locked)
```

`δ = 0.5` is the minimum effect size we treat as meaningful. Translated to
physical terms: ≈ 30 seconds of recovered delay (since w_delay = 1.0 and clip
is in minutes, 0.5 reward = 0.5 min × 1.0 weight × full gate = 30 sec).

For ties (|delta| ≤ δ): cell is `Aligned-justified` if aligned, else excluded
from this table (counted as "neutral divergence").

### 4.3 Per-cell metrics

```
Table IV — Replicate-AND-Improve (CQL B5, test set, 3-seed mean)

                          n_samples  % of test  mean_reward_delta
────────────────────────────────────────────────────────────────────
Aligned-justified         52,000     74%        +0.8 (model = signaller, L3 good)
Aligned-suboptimal         5,000      7%        -0.4 (model = signaller, L3 says better existed)
Divergent-improving ⭐     8,000     11%        +1.2 (model ≠ signaller, L3 says model right)
Divergent-unsafe            500      0.7%       -1.5 (model ≠ signaller, L3 says signaller right)
Neutral divergence         5,500      8%         0.0 (|delta| ≤ δ)
```

**Headline metrics** (for paper §VII):
- **Justified alignment rate**: 74% / (74+7) ≈ 91% of aligned decisions are good
- **Conditional improvement rate**: 8000 / (8000 + 500) ≈ 94% of divergences improve
- **Safe-override rate**: 1 - 0.7/19 ≈ 96% of model's divergences are safe
- **Overall replicate-AND-improve score**: 74% + 11% = **85%** of decisions are
  either signaller-correct (aligned-justified) or model-improvement (divergent-improving)

### 4.4 Priority counterfactual reward delta (per PROJECT_HANDOFF Ch 2.6)

Specific metric for priority decisions:

For each `priority_compete` stratum sample:
1. Identify peer decision points within ±5s for competing trains
2. Compute FCFS priority order (by arrival time)
3. Compute model's priority order (by which decision points it picks set vs wait)
4. For each disagreement: simulate both orderings via L3
5. Report `priority_counterfactual_reward_delta = reward(model_priority) - reward(FCFS)`

Expected: +0.1 to +0.3 on average — proving CQL finds slight priority improvements
over FCFS heuristic.

### 4.5 4-way computation script

`scripts/eval/04_replicate_and_improve.py`:

```python
def compute_4way(test_predictions, signaller_actions, l3_simulator):
    cells = {'aligned-justified': [], 'aligned-suboptimal': [],
             'divergent-improving': [], 'divergent-unsafe': [],
             'neutral': []}
    
    for sample in test:
        model_a = test_predictions[sample.id]
        signaller_a = signaller_actions[sample.id]
        aligned = (model_a == signaller_a)
        
        if aligned:
            # Find best alternative; compare
            best_alt = find_best_non_chosen_action(sample)
            delta_aligned = l3_simulator(sample, signaller_a) - l3_simulator(sample, best_alt)
            cell = 'aligned-justified' if delta_aligned > δ else 'aligned-suboptimal'
        else:
            delta_div = l3_simulator(sample, model_a) - l3_simulator(sample, signaller_a)
            if delta_div > δ:    cell = 'divergent-improving'
            elif delta_div < -δ: cell = 'divergent-unsafe'
            else:                cell = 'neutral'
        
        cells[cell].append(sample)
    
    return cells
```

---

## §5 Statistical reporting protocol

### 5.1 3-seed mean ± std

For each metric, run on 3 seeds (42, 43, 44):
- Compute metric per seed
- Report `mean ± std` (sample std, not population std)

### 5.2 Stratified bootstrap CI

For each metric, after collecting 3-seed predictions:
- Pool predictions from all 3 seeds (or use the median seed)
- Within each stratum, resample with replacement 1000 times
- Compute metric on each resample
- Report 95% CI as [2.5th percentile, 97.5th percentile]

**Why stratified**: random bootstrap on long-tail data underweights rare strata
(per PROJECT_HANDOFF Ch 11 教训 4). Within-stratum bootstrap preserves rare-class
representation.

### 5.3 Paired comparisons (model A vs model B)

For p-values comparing CQL to BC etc.:
- Compute per-sample difference d_i = acc_CQL(i) - acc_BC(i)
- Bootstrap mean of d_i, 1000 resamples
- p-value = fraction of bootstraps where mean(d) ≤ 0
- Report effect size (mean difference) + p-value

### 5.4 What NOT to do

| Forbidden | Why |
|-----------|-----|
| Cross-validation k-fold on time series | Future-to-past leakage (PROJECT_HANDOFF Ch 11 教训 6) |
| Random shuffle for splits | Same as above |
| Single-seed reporting | Hides variance |
| Bootstrap-only (no 3-seed) | Underestimates real variance |
| Cherry-pick "best of N seeds" | Selection bias |

---

## §6 Five-level XAI — overview

```
For each decision (s, a):

L1 Model   — Which assets did the encoder attend to?
              Output: heatmap on Derby panel

L2 Decision — Why this action over the next-best?
              Output: SHAP-like Q-gap decomposition + NL paragraph

L3 System  — What would happen in the next 30 min under each action?
              Output: counterfactual rollout comparison (requires P2.6 simulator)

L4 Manual  — Does the chosen action comply with Training Plan rules?
              Output: rule compliance table (requires P2.5 rule base)

L5 Reward  — What weights does the signaller's policy effectively apply?
              Output: per-context recovered weight vector + bootstrap CI
```

L1+L2+L5 derive from the trained model directly. L3 requires the simulator
(§14). L4 requires the rule base (§13). All 5 layers should be computable for
every decision, though L3 is expensive (~1 sec per decision).

---

## §7 L1 — Model attention layer

### 7.1 HGT attention extraction

Each HGT layer produces attention weights per edge-type. For one decision:
- Extract attention weights from all 3 HGT layers
- Aggregate via attention rollout (Abnar & Zuidema 2020): A_total = ∏_l (I + A_l) / d
- Per-node importance score: row sum of A_total at the focal_train node

### 7.2 Integrated Gradients (IG) cross-check

For each input feature f, IG(f) = ∫ ∂Q/∂f along straight path from baseline (zero) to actual input.

Why: HGT attention can be misleading; IG provides a complementary attribution.

Final saliency = average(attention_rollout_norm, ig_norm).

### 7.3 Projection onto Derby panel diagram

`data/reference/derby_all.png` is the actual Derby control panel diagram.
Coordinates for each TC / signal are derived from:
- `Derby_info.csv` `path` column (point IDs P***)
- Manual layout mapping (one-off effort, ~2 hours, stored at `data/reference/panel_layout.json`)

L1 output: heatmap overlay on the diagram showing the top-10 attended nodes,
color-coded by saliency.

### 7.4 Module

```python
# src/railrl/xai/l1_attention.py
def extract_attention(model, snapshot) -> dict:
    """Return {'attention_rollout': ndarray, 'ig': ndarray, 'top_nodes': list}"""

def visualize_on_panel(saliencies, panel_layout_path) -> Image:
    """Returns PIL Image with heatmap overlay."""
```

### 7.5 Faithfulness check

Per PROJECT_HANDOFF Ch 11 educational fact: attention can degenerate to global
context bias rather than per-sample focus. We need a faithfulness audit:

```python
# Check: do top-10 attended nodes vary across samples, or are they always the same?
distinct_top_nodes = set()
for s in test_samples[:1000]:
    top10 = extract_attention(model, s)['top_nodes'][:10]
    distinct_top_nodes.update(top10)

assert len(distinct_top_nodes) > 50, "attention is degenerate (always attending to same nodes)"
```

If degenerate: report as limitation in paper §VIII; don't claim faithful attention.

---

## §8 L2 — Decision-level explanation

### 8.1 Q-gap SHAP decomposition

For chosen action a* with Q(s, a*) and runner-up a' with Q(s, a'):

```
Q_gap = Q(s, a*) - Q(s, a')
```

Decompose Q_gap into contributions from feature groups using KernelSHAP or
gradient-based SHAP variants:

```
Q_gap = Σ_groups SHAP_g
where groups = {
    train_features, route_features, subgraph_state, sequence_summary,
    schedule_outlook, special_flags
}
```

Within each group, can further drill down (e.g., route_features → {prefix_emb,
length, gap_time, currently_locked, ...}).

### 8.2 Natural language template

Per-decision, generate a NL paragraph following this template (Chinese examples
shown; English is paper-friendly form):

```
决策 ({focal_train}, {chosen_route}) at {t}:

Trivial baseline:
  - Trajectory prior assigns {prior_prob:.1%} probability to this route.

Special-case context:
{for each flag in [advance, call_on, platform_dev, priority_compete, late_train, unusual_id, trts, freight]}
  - {flag}: {value or "N/A"}
{endfor}

Model's deliberation (Q values):
{for top-3 actions sorted by Q:}
  - {action}: Q = {Q:.2f}
{endfor}

Q-gap decomposition ({chosen} vs {runner_up}):
  - Train features:    {SHAP_train:+.2f}
  - Route features:    {SHAP_route:+.2f}
  - Subgraph state:    {SHAP_state:+.2f}
  - Sequence summary:  {SHAP_seq:+.2f}
  - Schedule outlook:  {SHAP_sched:+.2f}
  - Special flags:     {SHAP_flags:+.2f}

Manual compliance (L4): {compliance status from §10}

L3 counterfactual (next 30 min):
  - Chosen action:     delay change = {l3_chosen:+.1f}s, throughput = {tp_chosen}
  - Runner-up action:  delay change = {l3_runner:+.1f}s, throughput = {tp_runner}
  - Net advantage of chosen: {l3_delta:+.2f} reward units
```

### 8.3 Module

```python
# src/railrl/xai/l2_qdecomp.py
def q_gap_decomposition(model, snapshot) -> dict:
    """Returns SHAP contributions per feature group."""

def generate_nl_rationale(decomp, sample_meta, l4_compliance, l3_delta) -> str:
    """Fills template, returns paragraph."""
```

---

## §9 L3 — System-level counterfactual rollout

### 9.1 Role of the simulator

L3 uses the P2.6 simulator (see §14) to answer: "If the signaller had taken
action a' instead of a*, what would the system look like 30 min later?"

This is **evaluation-only**, not training. Per PROJECT_HANDOFF Ch 4.4:
"Simulator ≠ training playground."

### 9.2 Rollout protocol

```python
def l3_counterfactual(snapshot, candidate_action):
    state_now = snapshot_to_simulator_state(snapshot)
    state_now.apply_action(candidate_action)
    
    end_state, metrics = simulator.simulate(
        initial_state=state_now,
        horizon_minutes=30,    # locked
    )
    
    return metrics   # {'delay_total', 'throughput', 'headway_violations', 'avg_speed'}
```

Compare metrics between `chosen_action` and `next_best_action` (or any
alternative).

### 9.3 Output for L2 / Tier 3

The L3 delta (in reward units) feeds back into:
- L2's NL paragraph (showing system consequences)
- Tier 3's 4-way decomposition (the δ threshold check)

### 9.4 Multi-action comparison

For each decision, run L3 for **top-3 candidates by Q + signaller's choice**.
Generates 3-4 rollouts. Stored as `outputs/xai/l3_rollouts/{sample_id}.json`.

### 9.5 Module

```python
# src/railrl/xai/l3_system.py
def l3_compare_actions(snapshot, actions_to_evaluate: list, simulator) -> dict:
    """Returns {action: metrics_dict} for each action."""

def l3_delta(snapshot, action_a, action_b, simulator) -> float:
    """Reward delta in same units as r_total."""
```

---

## §10 L4 — Manual compliance check

### 10.1 Role of the rule base

Per P2.5 (§13), 80-120 IF-THEN rules extracted from Training Plan §3 + §5
describe preferred routing, platform assignment, traffic flow priorities, etc.

L4 checks whether the chosen action matches the rule base's preferred action
in this context.

### 10.2 Compliance check algorithm

```python
def l4_check(decision_sample, rule_base) -> dict:
    matched_rules = []
    for rule in rule_base:
        if rule.matches_context(decision_sample):
            matched_rules.append(rule)
    
    if not matched_rules:
        return {'status': 'no-rule', 'rules': []}
    
    # Highest-confidence matching rule's preference
    preferred = max(matched_rules, key=lambda r: r.confidence).preferred_route_id
    
    if decision_sample.chosen_route_id == preferred:
        return {'status': 'compliant', 'rules': matched_rules}
    elif preferred in [r for r in decision_sample.candidate_route_ids]:
        return {'status': 'non-compliant', 'rules': matched_rules,
                'preferred': preferred}
    else:
        return {'status': 'preferred-unavailable', 'rules': matched_rules}
```

### 10.3 Output statistics for §VII

Per Tier 3 decomposition, report L4 distribution per cell:

```
Cell                       compliant  non-compliant  no-rule
─────────────────────────────────────────────────────────────
Aligned-justified          85%        2%             13%
Aligned-suboptimal         70%        25%            5%
Divergent-improving        62%        20%            18%
Divergent-unsafe           30%        65%            5%   ← red flag
```

`Divergent-unsafe + non-compliant` should be < 1% of total — that's the
selective_override gate (§12).

### 10.4 Module

```python
# src/railrl/xai/l4_rules.py
def l4_check(decision_sample, rule_base) -> dict: ...
def l4_summary_per_cell(decompositions) -> dict: ...
```

---

## §11 L5 — Reward recovery (MaxEnt-IRL)

### 11.1 MaxEnt-IRL formulation

Given the trajectory data D = {(s, a, r)}, recover weight vector w* ∈ R^4
such that signaller's policy is maximum-entropy optimal w.r.t.:

```
r(s, a; w) = w_delay · r_delay(s, a) + w_throughput · r_throughput(s, a)
           + w_headway · r_headway(s, a) + w_wait · r_wait(s, a)

w* = argmax_w  E_D[ log π(a | s; w) ]
     where π(a | s; w) = softmax_a ( Q^*(s, a; w) )
     (Q^* obtained via Bellman backup with reward r(·; w))
```

In practice: alternating optimization between estimating Q^* and updating w.

### 11.2 Per-context recovered weights

Run MaxEnt-IRL separately on:
- **Whole dataset** → global w*
- **Per-prefix subsets** → w*_DC, w*_TD, etc.
- **Per-headcode-class** → w*_class1 (express), w*_class6 (freight), etc.
- **Per-cell from Tier 3** → w*_aligned-justified, w*_divergent-improving (compares
  what signaller seems to optimize vs what CQL learns)

### 11.3 Output: weight CIs

Bootstrap-based CI: resample episodes 1000 times, refit IRL, get distribution
of w*.

```
Report Table V (§VII):

Subset                  w_delay        w_throughput   w_headway     w_wait
──────────────────────────────────────────────────────────────────────────
Global (signaller)      0.85 ± 0.03    0.62 ± 0.05    0.91 ± 0.04   0.41 ± 0.06
Per-prefix DC           0.92 ± 0.05    0.55 ± 0.07    0.88 ± 0.05   0.38 ± 0.07
Per-prefix EC (depot)   0.45 ± 0.10    0.30 ± 0.08    0.95 ± 0.06   0.55 ± 0.09
Per-class Express       0.88 ± 0.04    0.68 ± 0.05    0.92 ± 0.04   0.40 ± 0.06
Per-class Freight       0.45 ± 0.08    0.42 ± 0.07    0.85 ± 0.05   0.35 ± 0.08
```

**Insight examples**:
- Depot regions (EC) show lower delay weight, higher wait weight → signaller
  cares less about timing, more about not interrupting
- Express trains weighted heavier on delay than freight → expected
- The weights are an **empirical finding** about signaller priorities, not a
  pre-specified prior — this is the main contribution of L5

### 11.4 Module

```python
# src/railrl/xai/l5_irl.py
def maxent_irl(trajectories, gamma=0.95, max_iter=100) -> ndarray:
    """Returns w* ∈ R^4"""

def bootstrap_irl(trajectories, n_resamples=1000) -> dict:
    """Returns {'mean': ndarray, 'std': ndarray, 'ci_low': ndarray, 'ci_high': ndarray}"""
```

---

## §12 Selective Override deployment rule

### 12.1 The rule

```python
def selective_override(snapshot, signaller_action, model_action,
                        l2_explanation, l3_delta, l4_compliance):
    """
    Returns: ('agreement', 'consider-override', 'silent')
    """
    if model_action == signaller_action:
        return 'agreement'   # show agreement badge
    
    # Three gates must all pass:
    gate_l3 = (l3_delta > δ_L3)                          # δ_L3 = 0.5 reward units
    gate_l4 = (l4_compliance['status'] == 'compliant')   # model's choice is rule-compliant
    gate_l2 = (l2_explanation['faithfulness'] > 0.7)     # SHAP attribution is consistent
    
    if gate_l3 and gate_l4 and gate_l2:
        return 'consider-override'   # show override card
    else:
        return 'silent'              # do not distract signaller
```

### 12.2 Faithfulness check for L2

Faithfulness = consistency of SHAP attribution under input perturbation:
- Zero out the highest-attributed feature group
- Recompute Q
- Q should drop by approximately the SHAP value (if attribution is faithful)
- `faithfulness = 1 - |actual_drop - predicted_drop| / |predicted_drop|`

Threshold 0.7 = SHAP-predicted drop within 30% of actual.

### 12.3 Deployment statistics (paper §VIII)

Run on test set with selective_override:
- `agreement_rate`: ~85% of decisions
- `consider_override_rate`: ~5-10% (the high-confidence improvement opportunities)
- `silent_rate`: ~5-10% (model disagreed but gates failed → silent)

### 12.4 Module

```python
# src/railrl/deploy/selective_override.py
δ_L3 = 0.5
FAITHFULNESS_THRESHOLD = 0.7

def selective_override(...) -> tuple[str, dict]: ...

def evaluate_selective_override_on_test(test_samples) -> dict:
    """Report agreement/override/silent rates."""
```

### 12.5 UI prototype (deferred — scope ends here)

Spec 05 does NOT define the UI — research scope ends with the API. Production
UI would be a separate engineering effort with Network Rail.

---

## §13 P2.5 Rule base spec

### 13.1 Source documents

- **Training Plan §3** — "Traffic Flows" (high-priority direction by time of day)
- **Training Plan §5** — "Preferred / non-preferred routes" (per signal)
- **Training Plan §6** — "Platform assignment" (rejected for L4 use as too restrictive)
- **Training Plan §11** — "Sinfin Branch" (single-line token control)
- **Training Plan §14** — "Matlock Branch" (No-Signaller Token System)

### 13.2 Rule schema

`outputs/rule_base/rules.parquet`:

| Field | Type | Description |
|-------|------|-------------|
| `rule_id` | string | e.g., `S5-tDmain-platform4` |
| `source_section` | string | e.g., `§5.2` |
| `cond_origin` | string or None | South / North / West / East / depot-name |
| `cond_destination` | string or None | platform_1..6 / continuation_signal |
| `cond_train_class` | string or None | passenger / freight / ECS / light |
| `cond_time_of_day` | tuple (start_hr, end_hr) or None | for §3 traffic flow rules |
| `cond_other` | string or None | free-text caveats |
| `preferred_route_id` | string or None | when §5 names a specific route |
| `preferred_platform` | int 1-6 or None | when §3/§6 names a platform |
| `non_preferred_alternatives` | list[string] | alternatives flagged available-but-slower |
| `confidence` | string | `low` / `med` / `high` |
| `user_approved` | bool | filled during review |
| `notes` | string | rationale + plan excerpt |

### 13.3 Extraction workflow

**I (AI) draft → you (user) review → both sign off per rule.**

Tools:
- `scripts/rules/01_extract_draft.py` — reads Training Plan via python-docx,
  uses templated parsing to draft rules
- `scripts/rules/02_review_ui.py` — simple TUI for you to mark each rule
  approved / needs-edit / rejected
- `scripts/rules/03_finalize.py` — writes `rules.parquet` with only approved rules

Expected output: ~80-120 rules. Stored at `outputs/rule_base/rules.parquet`.

### 13.4 Examples (illustrative)

```
rule_id: S5-TD5045-platform4
source_section: §5.2 (Plan paragraph 222)
cond_origin: South (Spondon direction)
cond_destination: platform_4
cond_train_class: passenger
preferred_route_id: RTD5045A(M)  # via 306pts + 311pts
non_preferred_alternatives: ['RTD5045B(M)']  # via 303pts reverse + 307pts if 311 locked
confidence: high
notes: "Plan: 'From TD5045 to platform 4 there is a preferred and non-preferred route...'"
```

```
rule_id: S11-sinfin-single-line
source_section: §11
cond_origin: any
cond_destination: Sinfin branch
preferred_route_id: null  # generic policy
cond_other: "only one train on branch at a time; must clear DW5320 before next entry"
confidence: high
```

### 13.5 Module

```python
# src/railrl/data/rule_base.py
def load_rule_base() -> pd.DataFrame: ...
def rule_matches(rule_row, sample) -> bool: ...
```

---

## §14 P2.6 Simulator spec

### 14.1 Design philosophy

**Event-driven, ~500 lines Python, no learning involved.**

The simulator answers: "If the system enters state S now and takes action a,
what does the next 30 min look like?"

Parametric (not learned) because:
- Audit-readable parameters (per Plan + per empirical fits)
- Sparse coverage tolerated (rare states still simulatable)
- Physical constraints (headway, dwell) are physical, not statistical

### 14.2 Empirical parameter tables

Pre-computed from 14-month data, stored at `outputs/simulator/parameters.parquet`:

| Parameter | Computation | Granularity |
|-----------|-------------|-------------|
| `route_running_time(route_id, train_class)` | Time from first TC occupied to last TC cleared, per traversal | per (route × class) — 277 × 11 cells |
| `platform_dwell_time(platform, train_class)` | Movements ARRIVAL → DEPARTURE delta on same train_id × platform | per (platform × class) — 6 × 11 cells |
| `min_headway(track_id)` | Min time between two successive different-train passages on same TC | per track_id — 249 cells |
| `aspect_clear_lag(signal_id)` | Time from PR → first Signal state=0 | per signal_id — 100 cells |

Each parameter: report `{p25, p50, p75, p95}`; rollout uses p50 by default,
ablations test p75 (conservative) and p25 (optimistic).

Use Derby_info `gap_time_s` as `route_running_time` proxy when empirical data
sparse — per spec 01 §6.5.

### 14.3 Rollout algorithm

```python
def simulate(initial_state, action, H_minutes=30):
    state = initial_state.copy()
    state.apply_action(action)   # set the proposed route
    
    events = MinHeap()
    for train in initial_state.trains_in_area:
        next_ev = predict_next_event(train, state,
                                      route_running_time, platform_dwell_time,
                                      min_headway)
        events.push(next_ev)
    
    timeline = []
    while events and events.top.time <= initial_state.t + H_minutes * 60:
        ev = events.pop()
        state.apply(ev)
        timeline.append(ev)
        # Schedule next event for the train this event belonged to
        if ev.train_still_active:
            events.push(predict_next_event(ev.train, state, ...))
    
    metrics = state.compute_metrics(timeline)
    return state, metrics
```

`predict_next_event` uses parameter tables to estimate when train will reach
next TC / clear next signal / etc.

### 14.4 Output metrics

| Metric | Definition |
|--------|------------|
| `delay_change_30min` | Σ over all trains in area of (final_delay - initial_delay) |
| `throughput_30min` | n distinct trains completing any route in horizon |
| `headway_violations` | count of (TC, time) pairs where headway < H_min |
| `avg_speed` | average train speed in area over horizon |
| `total_reward` | r_total summed if reward formula applied to simulated trajectory |

### 14.5 Module

```python
# src/railrl/xai/l3_system.py (continued)
class L3Simulator:
    def __init__(self, parameter_path):
        self.params = pd.read_parquet(parameter_path)
    
    def simulate(self, initial_state, action, horizon_min=30) -> dict: ...
    def compute_metrics(self, timeline) -> dict: ...
```

### 14.6 Validation

Before using L3 for Tier 3 decomposition:
- Run simulator on **held-out month** (Feb 2024)
- Compare simulated trajectory metrics to actual TD/Movements
- Target: `Spearman(simulated, actual) > 0.6` on delay_change and throughput
- If lower, recalibrate parameters or use wider CIs in L3

---

## §15 Implementation modules

### 15.1 Module map

```
src/railrl/eval/
├── __init__.py
├── metrics.py            # §2 + §3 + §5: accuracy, F1, KL, Kendall, bootstrap
├── stratified.py         # §3.1: 8-column table builder
├── per_prefix.py         # §3.2: per-prefix slicing
├── replicate_improve.py  # §4: 4-way decomposition
└── statistical.py        # §5: 3-seed mean ± std + bootstrap CI + p-values

src/railrl/xai/
├── __init__.py
├── l1_attention.py       # §7
├── l2_qdecomp.py         # §8
├── l3_system.py          # §9 + §14 (simulator class lives here)
├── l4_rules.py           # §10
└── l5_irl.py             # §11

src/railrl/deploy/
├── __init__.py
└── selective_override.py # §12

src/railrl/data/
└── rule_base.py          # §13

scripts/eval/
├── 01_compute_tier1.py
├── 02_compute_tier2.py
├── 03_compute_tier3.py   # requires L3 simulator + rule base
├── 04_full_eval_report.py
└── 05_paper_tables.py    # generates LaTeX-ready tables

scripts/xai/
├── 01_l1_attention.py
├── 02_l2_explain_samples.py     # generates NL rationale for N samples
├── 03_l3_rollouts.py            # batch L3 for all test set
├── 04_l4_compliance_audit.py
├── 05_l5_irl.py
└── 06_xai_case_studies.py       # picks 5-10 illustrative cases for paper

scripts/rules/
├── 01_extract_draft.py
├── 02_review_ui.py
└── 03_finalize.py

scripts/simulator/
├── 01_estimate_parameters.py
└── 02_validate_simulator.py
```

---

## §16 Verification + sanity checks

### 16.1 Tier 1 sanity

```python
overall = json.load(open("outputs/eval/tier1.json"))
assert overall['Q_argmax_top1_acc'] > 0.65  # spec 04 §11.3 threshold
assert 0.20 < overall['wait_rate_model'] < 0.30   # ~25% wait, matches data
```

### 16.2 Tier 2 sanity (8 columns)

```python
stratum_table = json.load(open("outputs/eval/tier2_stratum_table.json"))
assert stratum_table['cql_b5']['Trivial'] > 0.95
assert stratum_table['cql_b5']['Advance'] > 0.40   # should beat B0' by 10x
assert stratum_table['cql_b5']['Late'] > 0.50
```

### 16.3 Tier 3 sanity (4-way)

```python
decomp = json.load(open("outputs/eval/tier3_decomposition.json"))
assert decomp['divergent_improving_rate'] > 0.08   # ≥ 8% of test = ⭐
assert decomp['divergent_unsafe_rate'] < 0.02      # ≤ 2% of test
assert decomp['justified_alignment_rate'] > 0.85
```

### 16.4 L1 faithfulness

```python
faith = json.load(open("outputs/xai/l1_faithfulness.json"))
assert faith['n_distinct_top_nodes'] > 50
```

### 16.5 L3 simulator validation

```python
val = json.load(open("outputs/simulator/validation.json"))
assert val['spearman_delay'] > 0.6
assert val['spearman_throughput'] > 0.6
```

---

## §17 Open questions for future revisions

| # | Question | Default |
|---|----------|---------|
| 1 | Should L5 IRL run on CQL-policy actions or signaller-actions? | Both — paper §VII reports "signaller IRL weights" and "CQL effective weights" as comparison |
| 2 | What about L1 attention in the Sequence Transformer (not just HGT)? | Yes, extract from both; aggregate over edge-type AND token-position |
| 3 | δ_L3 = 0.5 — is this calibrated? | Initial value; spec 05 v1.1 may revise after first eval run |
| 4 | Should we cache L3 rollouts to avoid recomputing? | Yes — `outputs/xai/l3_cache.parquet`, keyed by (snapshot_hash, action_idx) |
| 5 | UI / dashboard for explanations | Out of scope; research deliverable ends at API |

---

## §18 Changelog

- **v1.0 (2026-05-19)** — Initial draft. Locks 3-tier eval (overall / stratified
  8-column + per-prefix + per-class / Replicate-AND-Improve 4-way with
  δ_L3 = 0.5), full statistical reporting (3-seed mean ± std + stratified
  bootstrap CI + paired bootstrap p-value), 5-level XAI (L1 attention rollout
  + IG + panel projection, L2 Q-gap SHAP + NL paragraph, L3 30-min simulator
  rollout, L4 rule-base compliance, L5 MaxEnt-IRL with bootstrap CIs),
  Selective Override deployment rule (δ_L3 + L4 + faithfulness gates),
  P2.5 rule base schema + extraction workflow (AI drafts, user reviews,
  ~80-120 rules), P2.6 simulator design (event-driven, parametric, ~500 LOC).

---

**End of Spec 05.**
**Sign-off:** ☐ Hao  /  Date: ______

---

# 🎉 全部 5 份 spec 完成

| Spec | 行数 | 主题 |
|------|------|------|
| 01 | 1,130 | Data Pipeline (§3 + §4 of paper) |
| 02 | 972 | MDP Formulation |
| 03 | 759 | Model Architecture |
| 04 | 768 | Training Protocol |
| **05** | **~1,000** | **XAI + Evaluation + Deployment** |
| **Total** | **~4,600 lines** | **完整契约从数据到论文** |
