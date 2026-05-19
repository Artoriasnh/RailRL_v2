# Phase 2 — Data Engineering Feature Specification

**Document version:** v2.2 · **Last updated:** 4 May 2026 — empirical patterns added

This document is the canonical reference for **every feature produced or
consumed by Phase 2**. Two parts:

- **Part A** — features already produced (P2.1 + P2.2). Verify against the
  parquet/json under `outputs/p2_data_eng/` to confirm they match.
- **Part B** — features still to be designed and built (P2.3 → P2.6).
  Refined extensively from v1 based on user review (see *Changelog* at end).

Use the **Status** column (✓ done · ⏳ pending · 🟡 design · ❓ open) to track
each item.

---

## What changed in v2 vs v1

| # | Topic | v1 | v2 |
|---|-------|----|----|
| 1 | **Temporal causality** | Mentioned only for Movements | Now governs all data sources (TD, Movements, SCHEDULE) — see new §B.0 |
| 2 | **Time windows** | W = {5m, 30m} | **W = {1, 5, 10, 15, 30} min** — five scales |
| 3 | **Node types** | 5 (Track / Signal / Berth / Route / Train) | **4** — Berth merged into SignalBerth (numeric berth = signal numeric tail; Q3) |
| 4 | **Platform features** | Absent | Added (using `platform_end_signals.csv` + `platform_tc_map.csv`) |
| 5 | **Hop / window distinction** | Conflated | Disentangled — hop sets node count; window sets temporal aggregation |
| 6 | **Conflict** | Hard mask + soft penalty in reward | **Both removed**. Empirical analysis showed 2.1 % of PRs occur with TC occupied (mostly queue routing) — interlocking handles physical safety; signaller's button-press timing is what we model. Occupation kept as state features only |
| 7 | **Reward dimensions** | 5 + 1 (delay, conflict, throughput, switching, priority, schedule_pressure) | **4 + 1** — conflict dropped |
| 8 | **Simulator role** | Implicit "playground" | Explicit: **NOT** a training playground (we use offline RL on demonstration buffer); it is the **L3 counterfactual evaluation engine** |

---

## PART A — Already produced (unchanged from v1)

(Schemas of the 7 already-produced artefacts. See v1 for full detail; only
the headlines repeated here.)

| File | Rows (5 M sample) | Purpose |
|------|-------------------|---------|
| `decisions/decision_events.parquet` | 231,919 | Action labels per Panel_Request |
| `infrastructure/routes_clean.parquet` | 277 | Static route catalogue with TC lists |
| `infrastructure/tracks_inventory.parquet` | 249 | TC → routes traversing it |
| `infrastructure/signals_inventory.parquet` | 100 | Signal → outbound routes (Aₜ source) |
| `infrastructure/auxiliary_connections.parquet` | 156 | Non-panel internal connections (archive) |
| `inventory/td_inventory.json` | n/a | TD descriptive statistics |
| `inventory/movements_inventory.json` | n/a | Movements descriptive statistics |

Verifications you can run against the actual parquets to confirm Part A still
matches reality after a full-data rerun:

```python
import pandas as pd
de = pd.read_parquet('outputs/p2_data_eng/decisions/decision_events.parquet')
de['prefix'].value_counts()                 # 5 prefixes present
de['cls'].value_counts()                    # M ≫ S > C
de['hc_class_digit'].value_counts(dropna=False)  # check 7,8 anomalies
```

---

## PART B — Proposed for P2.3 onwards

### B.0 Temporal causality contract  *(NEW)*

Hard rule that governs every state-feature computation, every reward
computation, and every action-mask query.

**Two kinds of "future" in our datasets:**

| Type | Source | Knowable to signaller at decision time t? | Use in **state** | Use in **reward / Q-target** |
|------|--------|-------------------------------------------|-------------------|-------------------------------|
| **Scheduled** | Movements `gbtt_timestamp`, `planned_timestamp`; SCHEDULE feed | ✅ yes (timetable) | ✅ allowed | ✅ allowed |
| **Realized** | Movements `actual_timestamp` for events at t' > t; TD events at time' > t | ❌ no (hindsight) | ❌ **forbidden** | ✅ allowed (return is computed in hindsight) |

**Practical implementation pattern:**

```python
# State feature for Track 'TFDU' at decision time t
# CORRECT — uses only events with time <= t
state = tracks_df[(tracks_df.id == 'TFDU') & (tracks_df.time <= t)].iloc[-1]

# Forecast feature — incoming train at signal in [t, t+F]
# CORRECT — gbtt_timestamp is scheduled, knowable now
upcoming = movements_df[
    (movements_df.gbtt_timestamp.between(t, t + F)) &
    (movements_df.loc_stanox == focal_stanox)
]

# Reward — delta of timetable_variation from t to t+H
# OK to use actual_timestamp here because reward is hindsight
v_before = latest_variation(focal_train, ≤ t)
v_after  = latest_variation(focal_train, ≤ t + H)  # uses realized events!
reward_delay = -(v_after - v_before)
```

**At deployment (inference) time** the model only needs state, never reward,
so the realized-future signal never leaks into a deployed prediction.

---

### B.1 State features — temporal heterogeneous graph  *(P2.3 🟡)*

Each Panel_Request decision event (≈ 1.5–2 M of them) yields one **state
record** consisting of three components:

```
state(t) = {
    snapshot_graph(t),           # the heterogeneous graph at time t (PRESENT)
    history_features(t, W),      # per-asset temporal aggregates over 5 windows (PAST)
    schedule_outlook(t, F),      # incoming-train forecast from SCHEDULE (FUTURE)
}
```

#### B.1.1 Snapshot graph — four node types  *(reduced from 5)*

| Node type | Source | Typical count per snapshot (3-hop) |
|-----------|--------|-------------------------------------|
| **Track** | `tracks_inventory.parquet` + TD Track events | 5 – 25 |
| **SignalBerth** *(merged)* | `signals_inventory.parquet` + TD Signal events + TD CA/CB/CC events | 1 – 5 |
| **Route** *(candidate routes only)* | `routes_clean.parquet` + TD Route / Panel_Request events | 1 – 13 |
| **Train** | TD `trainid_filled` + Movements joined by train_id | 1 – 6 |

> **SignalBerth merge rule:** TD `from_berth` / `to_berth` purely-numeric IDs
> (e.g. `5045`) refer to the same physical point as the corresponding signal
> (`STD5045`). Merge them into one node carrying both *signal-side* features
> (aspect, n_routes_from) and *berth-side* features (occupied_by_train_id,
> dwell_age). Named berths (`PLT3`, `LDUM`, `ABDT`, etc.) stay as **Train
> attributes** (`train.current_berth='PLT3'`); not graph nodes.

#### B.1.2 Hop definition vs time window — disentangled

Two orthogonal concepts that v1 mistakenly conflated:

| Concept | Determines | Example |
|---------|-----------|---------|
| **Topology hop** *k* | Which nodes are in the snapshot | 1 hop from focal signal = adjacent assets via static infra edges; *k* = 3 chosen as default |
| **Time window** *W* | How far back per-node aggregates look | 5 windows: 1m / 5m / 10m / 15m / 30m |

**Result tensor size:** O(\|3-hop subgraph\|) × O(features-per-node) ≈
**400 numbers per snapshot**, regardless of how busy that 3-hop region was
during *W*. High-frequency events (hundreds per minute on a busy TC)
**get summarised into the per-window aggregates**, not stored individually.

#### B.1.3 Per-window aggregate features  *(applied to every Track and SignalBerth node)*

For each window W ∈ {1m, 5m, 10m, 15m, 30m} and each asset:

| Feature | Type | Computation |
|---------|------|-------------|
| `occupancy_fraction_W` | float [0, 1] | total time in state=1 ÷ W (Track) or fraction at most-restrictive aspect (Signal) |
| `n_state_changes_W` | int ≥ 0 | count of Track / Signal events for this asset within [t-W, t] |
| `last_change_age_s` | int ≥ 0 | seconds since the latest event (single value, window-independent) |

That gives **(2 metrics × 5 windows) + 1 = 11 features per asset**, not 15.
Plus ID, type, position attributes: ≈ 15 total per asset.

#### B.1.4 Track-node features

| Feature | Type | Source / formula |
|---------|------|------------------|
| `track_id` | string | tracks_inventory |
| `n_routes_using` | int | tracks_inventory (static) — conflict potential proxy |
| `platform_id` | int 1–6 \| None | `platform_tc_map.csv` lookup |
| `platform_sub` | A / middle / B \| None | `platform_tc_map.csv` |
| `occupied_now` | bool | latest TD Track event ≤ t with state=1 |
| `current_occupier_train_id` | string \| None | trainid_filled of latest Track event |
| `occupancy_fraction_{1,5,10,15,30}m` | float | aggregates per B.1.3 |
| `n_state_changes_{1,5,10,15,30}m` | int | aggregates per B.1.3 |
| `last_change_age_s` | int | per B.1.3 |
| `on_focal_train_path` | bool | TC ∈ any candidate route's `track_sections` for the focal train |

#### B.1.5 SignalBerth-node features  *(merged Signal + Berth)*

| Feature | Type | Source / formula |
|---------|------|------------------|
| `signal_id` | string | signals_inventory |
| `n_routes_from` | int | signals_inventory (static) |
| `is_focal_signal` | bool | computed |
| `is_platform_end` | bool | `platform_end_signals.csv` lookup |
| `platform_id` | int 1–6 \| None | `platform_end_signals.csv` |
| `platform_direction` | N / S \| None | `platform_end_signals.csv` |
| `aspect_restrictive_now` | bool | latest TD Signal event ≤ t |
| `aspect_fraction_red_{1,5,10,15,30}m` | float | per-window |
| `aspect_n_changes_{1,5,10,15,30}m` | int | per-window |
| `aspect_last_change_age_s` | int | per B.1.3 |
| **Berth-side:** `current_berth_train_id` | string \| None | latest CA/CB/CC into this berth |
| `berth_dwell_age_s` | int \| None | seconds since occupation began |

#### B.1.6 Route-node features  *(candidate routes for focal train)*

| Feature | Type | Source / formula |
|---------|------|------------------|
| `route_id`, `prefix`, `cls`, `letter`, `sub` | various | `routes_clean.parquet` |
| `track_sections` | list[string] | `routes_clean.parquet` (used as ordered embedding by GNN) |
| `n_tc_in_route` | int | `routes_clean.parquet` |
| `currently_locked` | bool | latest TD Route event ≤ t with state=1 |
| `last_locked_age_s` | int | seconds since route was last set |
| `end_platform_id` | int \| None | end_signal → `platform_end_signals.csv` |
| **Occupation features** *(replaces conflict mask)*: |||
| `n_tcs_occupied_by_other` | int | TCs in route currently occupied by ≠ focal train |
| `n_tcs_occupied_by_focal` | int | TCs in route currently occupied by focal train (usually start) |
| `max_relative_position_of_occupied` | float [0, 1] | latest occupied TC's index / (n_tc - 1); >0.7 → likely queue routing |
| `min_age_of_occupation_s` | int \| None | seconds since most-recent occupation began (None if none occupied) |
| `is_chosen` | bool | **the supervised label** — exactly one True per snapshot |

#### B.1.7 Train-node features

| Feature | Type | Source / formula |
|---------|------|------------------|
| `train_id` | string | TD `trainid_filled` |
| `is_focal` | bool | computed — True for the train whose route is being decided |
| `headcode_class` | one of {`0`,`1`,`2`,`3`,`4`,`5`,`6`,`9`,`other`,`non_standard`} | `0–6,9` parsed from train_id; `7`/`8` → `other`; non-4-char IDs (`343R`) → `non_standard` (row kept) |
| `toc_id` | int | Movements joined by train_id |
| `current_berth` | string | latest CA/CB/CC for this train_id |
| `current_platform` | int 1–6 \| None | current_berth → `platform_tc_map.csv` reverse lookup |
| `planned_platform` | int 1–6 \| None | Movements `platform` for this train_id (joined by gbtt_timestamp ≈ now) |
| `time_in_current_berth_s` | int | seconds since last CA/CC event |
| `scheduled_time_delta_s` | int (signed) | gbtt_timestamp − t  (positive = ahead of schedule) |
| `recent_panel_requests_count` | int | Panel_Requests with this `trainid_filled` in [t-5m, t] |

#### B.1.7.1 Empirical observation — route × class correlation

Empirical analysis (`outputs/p2_data_eng/analyses/route_class_summary.json`)
shows that **minority routes at multi-route signals are heavily class-specialised**.
Examples in the 5 M-row sample:

| Route | Share of signal traffic | Top class | Top class % |
|-------|-------------------------|-----------|-------------|
| EC-5475 B(M) | 5.6 % | 5 (ECS) | **98.3 %** |
| DC-5076 A(M) | 3.9 % | 5 (ECS) | 98.1 % |
| EC-5486 D(M) | 12 % | 5 (ECS) | 95.0 % |
| DW-5306 B(C) Call-on | 9.7 % | 1 (Express) | 91.9 % |
| TD-5045 A(C) Call-on | 1.6 % | 1 (Express) | 91.8 % |

Two clear patterns:
1. **EC-prefix minority routes ≈ 95 %+ ECS** — depot moves, Etches Park / Chaddesden
2. **Call-on (C) class routes ≈ 92 % Express** — permissive working for fast trains into occupied platforms

The HG-DT model already has the structural pieces to learn these patterns
(train-node `headcode_class`, route-node `cls` / `letter` / `prefix`); GNN
message passing will pick up the co-occurrence. **No additional feature
engineering required** — but the case-study makes a useful narrative figure
for the paper.

### B.1.8 Future-pane (FUTURE) — schedule outlook only  *(scheduled, not realized)*

| Feature | Type | Source |
|---------|------|--------|
| `n_scheduled_arrivals_in_F` | int | count of Movements rows with gbtt_timestamp ∈ [t, t+F] at focal stanox |
| `top_K_upcoming` | list of dict | per upcoming train (K=5): `{hc_class, planned_platform, scheduled_eta_s, current_loc_proxy}` |
| `upcoming_priority_max` | int | max headcode_class digit among `top_K_upcoming` |

Default F = **15 minutes** (configurable). All values come from
`gbtt_timestamp` / `planned_timestamp` only — never from `actual_timestamp`
at t' > t (per §B.0).

#### B.1.9 TRTS state per platform  *(from new platform_tc_map.csv + TRTS events)*

Per platform sub-section (P1.A, P1.middle, P1.B, …, P6.B):

| Feature | Type | Source |
|---------|------|--------|
| `trts_pressed_age_s` | int \| None | seconds since latest `LPLAT{N}{A,B}TRS` event with state=1 |
| `trts_currently_pressed` | bool | most recent state of that TRTS button |

Used as additional context on the focal-train's home platform.

#### B.1.10 Edge schema  *(8 types)*

| Edge type | From → To | Static / Dynamic | Source |
|-----------|-----------|------------------|--------|
| `connects` | Track ↔ Track | static | adjacent in any `routes_clean.track_sections` |
| `traverses` | Route → Track | static | `routes_clean.track_sections` |
| `starts_at` | Route → SignalBerth | static | `routes_clean.start_signals` |
| `ends_at` | Route → SignalBerth | static | `routes_clean.end_signals` |
| `protects` | SignalBerth → Track | static | inferred (K=1 hop along outbound routes) |
| `same_signal` | Route ↔ Route | static | both have same start_signal — they're alternatives |
| `at_berth` | Train → SignalBerth | dynamic | latest CA/CB/CC |
| `next_signal` | Train → SignalBerth | dynamic | computed projection from current_berth |

Static edges precomputed once in P2.2.5; dynamic edges added per snapshot.

**`protects` edge — resolved:** K = 1, computed directly from `routes_clean`:

```python
# For each route, the "berth track" of its start signal is the first TC.
# A signal protects exactly the set of berth-tracks of its outbound routes.
protects = {(route.signal_no, route.track_sections[0])
            for _, route in routes_clean.iterrows()
            if route.track_sections}
```

This is physically grounded (UK signal "berth" semantics) and entirely data-driven
— no external domain assumption required. Yields ≈ 80–100 unique edges.

---

### B.2 Reward features — 4+1 operational dimensions  *(REVISED)*

Components of `r_t`. Also the feature vector φ(s, a) for MaxEnt-IRL recovery
at L5.

| # | Dim | Computation | Source | Sign |
|---|-----|-------------|--------|------|
| 1 | **delay** | Σ \|Δ-timetable_variation\| over all trains in same prefix sub-area within ±5 min of decision | Movements.timetable_variation | ↓ better |
| 2 | **throughput** | n distinct train_ids with any TD event in same sub-area in 60-s window | TD events | context (state-feature; not directly rewarded) |
| 3 | **switching** | count of Panel_Request events from this workstation in 60-s window before t | TD Panel_Request events | ↓ better (workload proxy) |
| 4 | **priority** | headcode_class one-hot for focal train | decision_events.hc_class_digit | reward shaping by class |
| 5 | **schedule_pressure** *(proposed addition)* | mean & max \|timetable_variation\| over all trains in sub-area at t | Movements.timetable_variation latest per train | ↓ better |

> **Removed in v2:** the `conflict` dimension is **gone** from both reward
> and action mask (§B.0 + empirical justification in
> `outputs/p2_data_eng/analyses/conflict_summary.json`). Track-occupation
> information is preserved as **state features** in §B.1.6 (`n_tcs_occupied_*`,
> `max_relative_position_of_occupied`, `min_age_of_occupation_s`); the model
> will learn from them, IRL will quantify their effective weight at L5.

Reward formula (priors set to 1.0 then revisited by IRL):

```
r_t = -|Δ-delay|                                       # main
      - λ_switching × switching_60s                    # workload
      + λ_priority(focal_hc_class) × (-|Δ-delay_focal|) # priority shaping
      - λ_pressure × schedule_pressure                  # system stress modulator
      # throughput is state-only, NOT a reward term
```

❓ **Resolved open questions:**
- O1 (window for delay) → **±5 min** (default unless ablation shows otherwise)
- O2 (conflict binary or count) → **N/A** (conflict removed)
- O3 (include schedule_pressure) → **YES**, included

---

### B.3 Rule base — Training Plan §3 + §5  *(P2.5 🟡)*

Workflow: **I draft, user reviews** *(confirmed Q1 of round 2)*.

Future schema in `outputs/p2_data_eng/rule_base/rules.parquet`:

| Field | Type | Description |
|-------|------|-------------|
| `rule_id` | string | e.g. `S3-passenger-south-sheffield` |
| `source_section` | string | `§3` or `§5` (§6 not used — geometric only) |
| `cond_origin` | string \| None | South / North / West / East / depot-name |
| `cond_destination` | string \| None | Sheffield / Birmingham / Matlock / etc. |
| `cond_train_class` | string \| None | passenger / freight / ECS / light |
| `cond_other` | string \| None | free-text caveats (e.g. "if 311 pts not locked") |
| `preferred_route_id` | string \| None | when §5 names a specific route |
| `preferred_platform` | int 1–6 \| None | when §3 names a platform |
| `non_preferred_alternatives` | list[string] | alternatives the manual flags as available-but-slower |
| `confidence` | low / med / high | annotator's confidence |
| `user_approved` | bool / None | filled during your review pass |

**Estimated:** 80–120 rules.

---

### B.4 Simulator — *(P2.6 🟡)*

#### What it is — and what it isn't

> **Simulator ≠ training playground.**
>
> We are doing **offline RL**: the policy learns from the demonstration buffer
> (231k+ logged Panel_Requests with their realised next-states from the same
> TD stream). No environment interaction is needed during training.
>
> The simulator's role is to answer **counterfactual** questions at
> evaluation / explanation / deployment time: *"If the signaller had chosen
> route B instead of A at this decision moment, what would the next 30 min
> have looked like?"* This is what L3 system-level explanations require, and
> what the selective-override rule consults.

#### Why parametric (not learned)

1. **Auditability** — L3 explanations need readable parameters
2. **Sparse coverage** — many (state, action) combinations are unseen; a
   parametric model with per-route statistics generalises better than a
   learned forward model
3. **Physical constraints** — headway, minimum running time are physical
   limits, not statistical relationships

#### Empirical parameter tables  *(extracted from past 14 months)*

| Parameter | Computation | Granularity |
|-----------|-------------|-------------|
| `route_running_time(route_id, train_class)` | Time from first TC occupied to last TC cleared, per traversal of this route. Take {p25, p50, p75, p95} | per (route_id × headcode_class) |
| `platform_dwell_time(platform, train_class)` | Movements ARRIVAL → DEPARTURE delta on same train_id × loc_stanox × platform | per (platform × class) |
| `min_headway(track_id)` | Minimum time between two successive different-train passages on same TC | per track_id |
| `aspect_clear_lag(signal_id)` | Time from Panel_Request → first Signal state=0 change | per signal_id |

#### Rollout — event-driven, ~500 lines of Python

```python
def simulate(initial_state, action, H_minutes=30):
    state = initial_state.copy()
    state.apply_action(action)              # set the proposed route
    events = priority_queue()                # min-heap on time
    for train in initial_state.trains_in_area:
        events.push(state.next_event_for(train,
                                         running_time=route_running_time,
                                         dwell=platform_dwell_time,
                                         headway=min_headway))
    while events and events.top.time <= initial_state.t + H_minutes:
        ev = events.pop()
        state.apply(ev)
        events.push(state.next_event_for(ev.train, ...))
    return state, state.compute_metrics()    # delay, conflict count, throughput
```

❓ **Resolved O5:** simulator horizon **H = 30 min** default; ablate to 10/60.

---

## Open questions — final status (v2)

| # | Question | Status | Resolution |
|---|----------|--------|-----------|
| O1 | Reward window for `delay` | **resolved** | ±5 min |
| O2 | `conflict` binary or count | **N/A** | conflict removed entirely |
| O3 | Include `schedule_pressure` | **resolved** | yes, included |
| O4 | Rule extraction ownership | **resolved** | I draft, user reviews |
| O5 | L3 horizon | **resolved** | 30 min default |
| O6 | State snapshot extent | **resolved** | 3 topological hops |
| O7 | `protects` edge inference | **resolved** | K = 1 hop; data-driven from `routes_clean` (each signal protects the first TC of each outbound route) |
| O8 | Headcode 7 / 8 handling | **resolved** | mapped to `"Other (non-standard)"` class |
| O9 | Non-standard train_ids (`343R` 1.04 %) | **resolved** | row kept; `hc_class_digit` → `non_standard` category (distinct from 7/8 `other`). Empirical evidence (`outputs/p2_data_eng/analyses/non_standard_trainids_summary.json`): these IDs cluster heavily in depot / sidings / shunt — EC-prefix 3.5× over-represented, Shunt-class routes 6.7× over-represented, top depot signals (EC-5474 21.6 %, TD-5049 RTC sidings 14.4 %, etc.). Single category sufficient. |
| O10 | Switching window | **resolved** | 60 s |
| **NEW O11** | Hard conflict mask in Aₜ | **resolved** | **dropped** — interlocking handles physical safety, signaller's button-press timing is what we model |
| **NEW O12** | Soft conflict (future schedule overlap) | **resolved** | **dropped** — schedule is real-time mutable |
| **NEW O13** | Time windows W | **resolved** | {1, 5, 10, 15, 30} min × 2 metrics + 1 = 11 features per asset |
| **NEW O14** | SignalBerth merge | **resolved** | merged when berth ID is purely numeric matching signal numeric tail |
| **NEW O15** | Future pane only scheduled | **resolved** | gbtt_timestamp / planned_timestamp only — no actual_timestamp leakage |

---

## Suggested review workflow

1. **Read v2** end-to-end; mark sections you accept vs sections needing change
2. **Resolve the three remaining open questions** — O7, O8, O9 — even
   tentatively. They affect implementation defaults
3. **Run `scripts/p2_data_eng/01_inventory.py` + `02_decisions.py` +
   `03_infrastructure.py` on full data** to confirm Part A still matches
4. **Run `scripts/analyses/conflict_empirical.py` on full data** to confirm
   the 2.1 %-by-other rate doesn't change drastically at full scale (sanity check)
5. After confirmation, P2.3 implementation begins — every line will trace to a
   row in this spec

---

## Changelog

- **v2.2 (4 May 2026)** — added empirical observations:
  (i) §B.1.7.1 route × class correlation findings (minority routes are
  class-specialised at >90 % concentration); (ii) extended O9 row with non-standard
  train_id concentration evidence (3.5× EC-prefix, 6.7× Shunt-class).
  Two new analysis scripts at `scripts/analyses/route_class_skew.py` and
  `non_standard_trainids.py`; both reproduce on full data.
- **v2.1 (4 May 2026)** — closed remaining open questions: O7 `protects` edge
  fixed at K=1 with data-driven definition; O8 headcode 7/8 → `"other"`;
  O9 non-standard train_ids kept (not dropped) as `non_standard` class.
- **v2 (4 May 2026)** — incorporated 8 reframings from kick-off review
  (Q1–Q8 round 1) plus 3 from data-driven review (Q1–Q3 round 2). Conflict
  dropped (both hard and soft); 5-window aggregation; SignalBerth merge;
  platform features integrated; explicit temporal-causality contract;
  simulator role clarified as evaluation engine not training playground.
- **v1 (3 May 2026)** — initial spec following narrative reframe to
  "replicate AND improve". Soft conflict, hard conflict mask, and 6-dim reward
  later removed in v2.
