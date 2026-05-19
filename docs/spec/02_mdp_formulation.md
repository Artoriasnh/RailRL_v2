# Spec 02 — MDP Formulation

**Document version:** v1.0 · **Last updated:** 2026-05-19
**Status:** 🟡 draft — awaiting sign-off
**Prerequisite:** Spec 01 v1.1 (signed-off 2026-05-19)
**Scope:** the MDP that the offline RL algorithm (spec 04) optimises and that the
encoder (spec 03) reads. Every line of code in `src/railrl/mdp/` answers to this spec.

---

## §0 Purpose & scope

### What this spec locks down

This document defines the **Markov Decision Process** that our model learns:
- **State** `s_t` — what the model sees at decision time `t`
- **Action** `a_t` — what the model picks
- **Reward** `r_t` — references spec 01 §12
- **Episode** boundary — when a sequence starts/ends
- **Trigger logic** — how decision points are generated from raw data
- **Sample metadata vs state features** — strict separation (per spec 01 §17.5)
- **`assert_no_leak()`** — runtime contract enforced at snapshot construction

### What this spec does NOT cover

- Encoder architecture (HGT layers, dim, fusion) → spec **03**
- Q-network architecture (per-action MLP, masking implementation) → spec **03**
- Training algorithm (CQL loss, conservative penalty, 3-stage protocol) → spec **04**
- XAI + evaluation → spec **05**

### Outputs this spec produces

- `outputs/snapshots/snapshots_v2.parquet` — one row per decision point, contains
  state features + sample metadata + (s, a, r) tuple components ready for spec
  04 training loop
- `outputs/snapshots/snapshots_v2_summary.json` — sanity counts
- Five new modules in `src/railrl/mdp/`

---

## §1 The MDP (formal definition)

### 1.1 Formal tuple

```
M = (S, A, R, T, γ)
```

| Element | Definition | Defined in |
|---------|------------|------------|
| **S** | All possible state vectors at decision points | §4 |
| **A_t** | **Structured discrete action set** at decision point t; size varies dynamically | §3 |
| **R** | `r_t = w_d·r_delay + w_t·r_thru + w_h·r_head + w_w·r_wait` (LOCKED) | spec 01 §12.2 |
| **T** | Transition kernel — NOT modelled (offline RL on demonstration buffer) | n/a |
| **γ** | Discount factor | **0.95** (§5.4) |

### 1.2 Why offline RL (no T modelled)

We do not learn `T(s' | s, a)`. The demonstration buffer is:

```
D = {(s_i, a_i, r_i, s'_i, done_i)}  i = 1..N (~727 k tuples)
```

s'_i is the **observed** next state after the signaller's recorded action — we do
NOT generate alternative s' via simulation. CQL (spec 04) operates entirely on D.

The L3 simulator in spec 05 is for **counterfactual evaluation** (rolling out
alternative actions for XAI), not for training rollouts.

---

## §2 Decision points and trigger logic

### 2.1 What a decision point is

A **decision point** = a tuple `(focal_train, focal_signal, t)` where the
signaller faces a choice: act for `focal_train` regarding routes from `focal_signal`,
or wait.

Each decision point generates **one** training sample with `label ∈ {set, wait}`.

### 2.2 Set triggers (from PR events)

Every Panel_Request in `decisions/decision_events.parquet` generates one set
decision point:

```
For each PR row p:
    sample = {
        focal_train  = p.train_id,
        focal_signal = parse_start_signal(p.route_id),  # numeric tail, e.g. "5045"
        t            = p.time,
        label        = 'set',
        chosen_route_id = p.route_id,
    }
```

**Expected count:** ~546,418 set samples (matches `decision_events.parquet`).

### 2.3 Wait triggers (from approach events)

For each signal S that is the end_signal of any route, compute its
**approach horizon**:

```
approach_TCs(S) = ⋃ {route.track_sections[-K_APPROACH:]
                     for each route ending at S}

K_APPROACH = 2   (LOCKED; same as v1 APPROACH_K_HOPS)
```

When any TC in `approach_TCs(S)` becomes occupied at time `t` by a train
`focal_train = T` (from TD Track event with `state=1, trainid_filled=T`):

```
Look ahead in PR stream:
    found_pr = exists p ∈ decision_events such that
        p.train_id == T
        AND parse_start_signal(p.route_id) == S
        AND p.time ∈ [t, t + Δ_WAIT]

Δ_WAIT = 30 s   (LOCKED; same as v1 DECISION_LOOKAHEAD_SECONDS)

If NOT found_pr:
    Emit wait sample:
        focal_train  = T,
        focal_signal = S,
        t            = t       (= approach entry time)
        label        = 'wait',
        chosen_route_id = NaN,
```

**Expected count:** ~181 k wait samples (per v1 stats).

### 2.4 Combined decision_points table

After §2.2 + §2.3:

| Quantity | Expected count |
|----------|----------------|
| n_set | ~546,000 |
| n_wait | ~181,000 |
| n_total | ~727,000 |
| neg:pos ratio | ~1:3 |

### 2.5 Trigger time semantics

`t` is the **event time** that triggered the decision point:
- For set: the PR's `time` (signaller's button press)
- For wait: the moment the train entered the approach horizon

State features at `t` use **strict less-than**: events at exactly `time == t`
are NOT included in state (to avoid leaking the trigger event itself).

Exception: the trigger event MUST be visible — TD Track event at `t` for the
train entering approach IS part of state (it's how we know we're at a decision
point). The strict-less-than applies to events of other assets at `t`.

**Implementation rule:** event filter is `event.time < t`, then add the trigger
event explicitly with `state = pre_trigger_state`.

### 2.6 Duplicate decision points

If a train enters multiple TCs in the same approach horizon at near-same times
(adjacent TCs occupied within 1 s of each other), dedupe to the **earliest**
trigger only:

```
For each (T, S):
    keep only the earliest approach entry within any 30-s window
```

This prevents triple-counting the same operational decision.

### 2.7 Module: `src/railrl/mdp/trigger.py`

Public API:

```python
def generate_decision_points(
    pr_df: pd.DataFrame,            # outputs/decisions/decision_events.parquet
    td_events: TDEventStream,
    routes_clean: pd.DataFrame,
    k_approach: int = 2,
    delta_wait_seconds: float = 30.0,
) -> pd.DataFrame:
    """
    Return decision_points table with columns:
        focal_train, focal_signal, t, label, chosen_route_id, trigger_type
    where trigger_type ∈ {'panel_request', 'approach'}
    """
```

**Output:** `outputs/decision_points/decision_points_v2.parquet`

---

## §3 Action space — structured discrete actions

### 3.1 Action definition

At decision point `(focal_train, focal_signal, t)`:

```
A_t = {wait} ∪ {(focal_train, R) | R ∈ candidates(focal_train, focal_signal, t)}
```

|A_t| typically 2-9 (1 wait + 1-8 candidate routes). Maximum observed: ~14.

### 3.2 Candidate set algorithm

```python
def candidates(focal_train, focal_signal, t, snapshot) -> list[Route]:
    """
    Compute the candidate routes for focal_train at focal_signal at time t.
    All inputs are SAMPLE METADATA (focal_signal known) — NOT state features.
    """
    out = []
    # All routes starting from focal_signal:
    for R in routes_starting_from(focal_signal):
        
        # Rule 1: train must be in approach horizon of focal_signal
        train_tc = snapshot.train_current_tc(focal_train, t)
        if not is_in_approach(train_tc, focal_signal, k=2):
            return []  # no candidates if train not actually at this signal
        
        # Rule 2: direction must match train's trajectory
        direction = infer_direction(snapshot.recent_tcs(focal_train, n=5))
        if route_direction(R) != direction:
            continue
        
        # Rule 3: route must not conflict with already-set routes for this pass
        prev_routes = snapshot.routes_already_set(focal_train.pass_id, before=t)
        if R conflicts with any prev_route:
            continue
        
        # Rule 4: planned_platform soft filter (NOT hard — allow platform reassign)
        # Routes ending at planned_platform are preferred but alternatives stay.
        out.append(R)
    
    return out
```

### 3.3 Candidate validation contract

**Hard invariant**: for set decisions, `chosen_route_id ∈ candidates(...)`.

If `chosen_route_id ∉ candidates(...)`:
- Either the candidate algorithm is too restrictive (BUG — log + skip sample)
- Or the data has an unusual case (e.g., manual override) — record in
  `outputs/decision_points/candidate_mismatch.parquet`

Target: ≥ 99.5% of set decisions have `chosen ∈ candidates`. If lower, fix
algorithm before proceeding.

### 3.4 Action representation

In `snapshots_v2.parquet`, each row stores:

| Column | Type | Description |
|--------|------|-------------|
| `n_candidates` | int | Length of A_t (incl. wait) |
| `candidate_route_ids` | list[string] | Ordered list of candidate route_ids; index `i` maps to action `(focal_train, candidate_route_ids[i])` |
| `chosen_action_idx` | int | Index into A_t of the action actually taken; -1 = wait (always at index 0 by convention) |
| `wait_action_idx` | int | Always 0 (wait is canonical first action) |

Action a_i is then identified by its index i ∈ {0, 1, ..., n_candidates-1} where
i=0 means wait and i≥1 means `(focal_train, candidate_route_ids[i-1])`.

### 3.5 Module: `src/railrl/mdp/action.py`

```python
def feasible_actions(focal_train, focal_signal, t, snapshot, static_graph) -> list[str]:
    """Returns list of candidate route_ids (not including wait, which is always implicit)."""

def validate_candidates(decision_points: pd.DataFrame, static_graph) -> dict:
    """Pass over training set, count how many `chosen_route_id` ∈ candidates."""
```

---

## §4 State features — complete schema

### 4.1 High-level state composition

```
s_t = {
    subgraph_nodes  : per-node features (4 types) within 3-hop of focal_train.current_tc
    subgraph_edges  : 6 edge types within the subgraph
    event_tokens    : last K=256 events with time < t (globally)
    schedule_outlook: top-5 upcoming trains in [t, t+15min] from gbtt
    special_flags   : 8 binary/numeric flags (§4.6)
    is_focal_train  : marker on the focal train node only
}
```

### 4.2 Subgraph extraction

**CENTER:** `focal_train.current_tc(t)` — the TC the focal train is currently on
at time t (from latest TD Track event with state=1 trainid_filled=focal_train
and time ≤ t).

**RADIUS:** 3 topological hops via any static graph edge (treating all 6 edge
types as undirected for hop counting).

**EXPECTED SIZE:** ~20-40 nodes typical, max ~60.

**LEAK CHECK:** the centering MUST be on focal_train.current_tc, never on
focal_signal. Spec 01 §17.5.4 enforces this via assert_no_leak().

### 4.3 Track node features

| Field | Type | Source | Notes |
|-------|------|--------|-------|
| `track_id` | string | tracks_inventory | identity |
| `n_routes_using` | int | tracks_inventory | static |
| `platform_id` | int or None | platform_tc_map | static |
| `platform_sub` | str ∈ {A,middle,B,None} | platform_tc_map | static |
| `occupied_now` | bool | TD Track event ≤ t | dynamic |
| `current_occupier_train_id` | string or None | TD trainid_filled of latest Track event | dynamic |
| `occupancy_fraction_1m` | float [0,1] | aggregate over [t-60s, t] | per-window |
| `occupancy_fraction_5m` | float [0,1] | aggregate over [t-300s, t] | per-window |
| `occupancy_fraction_10m` | float | over [t-600s, t] | per-window |
| `occupancy_fraction_15m` | float | over [t-900s, t] | per-window |
| `occupancy_fraction_30m` | float | over [t-1800s, t] | per-window |
| `n_state_changes_1m` | int | count Track events for this TC in [t-60s, t] | per-window |
| `n_state_changes_5m` | int | over [t-300s, t] | per-window |
| `n_state_changes_10m` | int | over [t-600s, t] | per-window |
| `n_state_changes_15m` | int | over [t-900s, t] | per-window |
| `n_state_changes_30m` | int | over [t-1800s, t] | per-window |
| `last_change_age_s` | int | t − last_event_time | single |
| `on_focal_train_path` | bool | TC ∈ any candidate route's track_sections for focal_train | dynamic |

Total: **18 features per Track node.**

### 4.4 Signal node features

| Field | Type | Source | Notes |
|-------|------|--------|-------|
| `signal_id` | string | signals_inventory | identity |
| `prefix` | str ∈ {DW,TD,DC,EC,DY} | signals_inventory | static |
| `n_routes_from` | int | signals_inventory | static |
| `is_platform_end` | bool | platform_end_signals | static |
| `platform_id` | int 1-6 or None | platform_end_signals | static, only if is_platform_end |
| `platform_direction` | str N/S or None | platform_end_signals | static |
| `aspect_restrictive_now` | bool | latest TD Signal event ≤ t | dynamic |
| `aspect_fraction_red_{1,5,10,15,30}m` | 5 × float | per-window aggregates | per-window |
| `aspect_n_changes_{1,5,10,15,30}m` | 5 × int | per-window | per-window |
| `aspect_last_change_age_s` | int | t − last_signal_event | single |
| `current_berth_train_id` | string or None | latest TD CA/CB/CC into this berth | dynamic |
| `berth_dwell_age_s` | int or None | seconds since occupation began | dynamic |

Total: **18 features per Signal node.**

**🚫 PROHIBITED on Signal node** (per spec 01 §17.5.4):
- `is_focal_signal`
- `is_focal`
- Any boolean indicating this signal is the action target

### 4.5 Route node features

| Field | Type | Source | Notes |
|-------|------|--------|-------|
| `route_id` | string | routes_clean | identity (model can read 5045 etc. — OK) |
| `prefix`, `signal_no`, `letter`, `sub`, `cls` | various | routes_clean | static |
| `n_tc` | int | routes_clean | static |
| `end_platform_id` | int or None | routes_clean + platform_end_signals | static |
| `length_m` | float | Derby_info | static, **physical** |
| `ave_speed_mps` | float | Derby_info | static, **physical** |
| `ave_grad` | float | Derby_info | static, **physical** |
| `gap_time_s` | float | Derby_info | static, **physical** (same param L3 simulator uses) |
| `n_points` | int | Derby_info | static, **physical** |
| `currently_locked` | bool | latest TD Route event ≤ t state=1 | dynamic |
| `last_locked_age_s` | int | t − last_route_event | dynamic |
| `n_tcs_occupied_by_other` | int | TCs in route currently occupied by ≠ focal_train | dynamic |
| `n_tcs_occupied_by_focal` | int | TCs occupied by focal_train (start TCs typical) | dynamic |
| `max_relative_position_of_occupied` | float [0,1] or None | latest occupied TC index / (n_tc-1) | dynamic; > 0.7 → likely queue routing |
| `min_age_of_occupation_s` | int or None | seconds since most-recent occupation began | dynamic |
| `in_candidate_set` | bool | is this route in feasible_actions for focal_train at focal_signal? | dynamic |

Total: **18 features per Route node.**

**🚫 PROHIBITED on Route node** (per spec 01 §17.5.4):
- `is_focal_route`
- `is_focal`
- `is_chosen` — that's the label, used only in training loss not as input
- Any boolean indicating this route is the chosen/correct one

### 4.6 Train node features

| Field | Type | Source | Notes |
|-------|------|--------|-------|
| `train_id` | string | TD trainid_filled | identity (4-char headcode or non-standard) |
| `is_focal` | bool | sample metadata | **ALLOWED only on Train nodes** |
| `headcode_class` | one of {0..9, other, non_standard} | parser | static |
| `current_tc` | string | latest TD CA/CB/CC | dynamic |
| `current_berth` | string or None | latest TD CA/CB/CC | dynamic |
| `current_platform` | int or None | current_tc → platform_tc_map | derived |
| `planned_platform` | int or None | Movements gbtt joined by train_id | **schedule** (NOT actual) |
| `time_in_current_berth_s` | int | t − last CA/CC event | dynamic |
| `scheduled_delta_s` | int signed | gbtt − t (positive = ahead of schedule) | derived from gbtt only |
| `recent_panel_requests_count` | int | PRs with this trainid_filled in [t-5min, t] | dynamic |

Total: **10 features per Train node.**

**⚠ Specifically on `is_focal` for trains:**
- It IS allowed (only allowed `is_focal_*` flag in entire state)
- Reasoning: train identity ≠ route action. The model needs to know which
  train to score candidates for.
- Implementation: set `is_focal=True` for exactly one train node per snapshot
  (the focal_train); `False` for all other trains in the subgraph.

### 4.7 Edge features (6 types)

All static edges from `outputs/static_graph/edges_*.parquet`. Within the
3-hop subgraph, include ALL edges (no filtering by focal_signal):

| Edge type | From → To | Static / Dynamic | Source |
|-----------|-----------|------------------|--------|
| `connects` | Track ↔ Track | static | edges_connects |
| `traverses` | Route → Track | static (with `order` attr) | edges_traverses |
| `starts_at` | Route → Signal | static | edges_starts_at |
| `ends_at` | Route → Signal | static | edges_ends_at |
| `protects` | Signal → Track | static | edges_protects |
| `same_signal` | Route ↔ Route | static | edges_same_signal |

Plus optional **dynamic** edges per snapshot:

| Edge type | From → To | Source |
|-----------|-----------|--------|
| `at_berth` | Train → Signal | latest CA/CB/CC of train |
| `next_signal` | Train → Signal | computed projection from current_berth |

### 4.8 Event token sequence (K=256)

From `outputs/event_stream/event_tokens.parquet`:

```
events_t = events[event_tokens.time_ns < t].tail(256)
```

Each token: `(asset_idx: int16, state: int8, time_delta_s: float)` where
`time_delta_s = (t - event.time_ns) / 1e9`.

**Padding:** if fewer than 256 events exist before t (early data), left-pad
with sentinel `(asset_idx=-1, state=0, time_delta_s=99999.0)`.

**Centering / filtering:** NO filtering by focal_signal or focal_train. The
sequence is the globally-last-K events — represents the "panel-wide hum".

### 4.9 Schedule outlook (top-5 upcoming trains)

```
F = 15 * 60   (15-minute lookahead, LOCKED)

upcoming = movements_df[
    (movements_df.gbtt_timestamp.between(t, t + F))
    & (movements_df.loc_stanox.isin(derby_stanoxes))
].sort_values('gbtt_timestamp').head(5)
```

For each upcoming row:

| Field | Type | Notes |
|-------|------|-------|
| `train_id` | string | TRUST id (or its headcode if available) |
| `headcode_class` | str | parsed |
| `eta_s` | int | seconds until arrival = gbtt − t |
| `planned_platform` | int 1-6 or None | **NEVER planned_end_signal — leak risk per §7** |

Padding: if fewer than 5 upcoming, pad with `eta_s=99999, others=None`.

### 4.10 Eight special-case flags

All computed from time ≤ t observable state. Per spec 01 §17.5.4, each
implementation must declare its `source` for leak audit.

| Flag | Type | Definition | Source declaration |
|------|------|------------|--------------------|
| `f_advance` | bool | The first TC of any candidate route is currently occupied by a train ≠ focal_train | `source: 'static_graph + td_events ≤ t'` |
| `f_call_on` | bool | Any candidate route has `cls='C'` AND its `end_platform_id` TC is currently occupied | `source: 'candidate_routes + td_events ≤ t'` |
| `f_platform_dev` | bool | Best-Q-candidate's `end_platform_id` ≠ `focal_train.planned_platform` | `source: 'candidate_routes + movements.gbtt'` |
| `f_priority_compete` | bool | ≥ 2 distinct trains are active (≥ 1 PR or approach in [t-5s, t+5s]) | `source: 'decision_points window'` |
| `f_late_train` | int (signed seconds) | `focal_train.scheduled_delta_s` if < -60s else 0 | `source: 'movements.gbtt − t'` |
| `f_unusual_id` | bool | `focal_train.train_id` does not match standard 4-char format | `source: 'parsers.HEADCODE_RE'` |
| `f_trts_pressed` | bool | TRTS button for `focal_train.planned_platform` OR `focal_train.current_platform` is currently pressed (state=1) | `source: 'td_events ≤ t @ planned|current platform — NEVER focal_signal platform'` |
| `f_freight_class` | bool | `focal_train.headcode_class ∈ {4, 6}` (freight intermodal + heavy freight) | `source: 'parsed headcode'` |

### 4.11 State summary

Per snapshot, total feature count:
- Subgraph: ~20-40 nodes × ~10-18 features each = ~300-700 numbers
- Edges: encoded as adjacency for HGT — not flat features
- Event tokens: 256 × 3 = 768 numbers
- Schedule outlook: 5 × 4 = 20 numbers
- Special flags: 8 numbers
- is_focal_train: 1 flag

**Total state size: ~1,100-1,500 numbers per snapshot.** Spec 03 will define
how the encoder consumes them.

---

## §5 Episode definition

### 5.1 Per-train episodes

An **episode** = the sequence of decision points belonging to one operational
pass of one train through Derby:

```
episode_e = {(s_i, a_i, r_i, s'_i) : sample_i.pass_id == pass_e}
```

`pass_id` comes from spec 01 §17.2 (materialized `pass_assignments.parquet`).

### 5.2 Multi-train overlap

Multiple trains' episodes overlap in time. They are **independent** episodes:
- Train T1's episode and T2's episode can both include decision points at
  similar t with overlapping state
- Each episode is a separate `(s_0, a_0, r_0, ...)` trajectory
- CQL training samples (s, a, r, s') tuples — does not depend on episode boundary
  for the loss itself, only for return computation in some variants

### 5.3 Episode boundary

**Start:** the first decision point with `pass_id == pass_e`
**End:** the last decision point with `pass_id == pass_e`

Between consecutive decision points within the same episode:
- `s'_i = s_{i+1}` (observed next state at the next decision point's t)
- `done_i = (i is the last in this episode)`
- For the final point: `done_T = True`, `s'_T` = ignored

### 5.4 Discount γ

```
γ = 0.95   (LOCKED)
```

Rationale: typical episode length is 5-15 decision points spanning ~5-30 minutes.
γ=0.95 gives effective horizon ~20 steps — covers most episodes' tail without
over-weighting distant terminal effects.

### 5.5 Inter-episode independence

Episodes from different `pass_id` are independent: no s' linkage. The model
does NOT need to learn "after episode A ends, episode B starts" — those are
separate trajectories.

### 5.6 Module: `src/railrl/mdp/episode.py`

```python
def build_episodes(decision_points: pd.DataFrame,
                    pass_assignments: pd.DataFrame) -> pd.DataFrame:
    """Add pass_id, episode_idx, position_in_episode, is_last_in_episode columns."""

def episode_returns(episodes: pd.DataFrame, gamma: float = 0.95) -> pd.Series:
    """Compute Σ_t γ^t · r_t per episode (for IRL Stage 5 + analysis)."""
```

---

## §6 Sample metadata vs. state features (strict separation)

### 6.1 Why this separation matters

The leak audit (§7) enforces a strict wall between:
- **Metadata** — used by reward calc, trigger logic, episode segmentation; lives
  in sample table; NEVER passed to model
- **State** — passed to model encoder; subject to §7 audit

Per spec 01 §17.5, this prevents focal_signal from leaking through any side
channel.

### 6.2 Metadata schema

Stored in `outputs/snapshots/snapshots_v2.parquet`:

| Column | Type | Purpose |
|--------|------|---------|
| `sample_id` | int64 | unique index |
| `focal_train` | string | train identity |
| `focal_signal` | string | signal this sample is about (metadata only!) |
| `t` | datetime | decision time |
| `pass_id` | string | from pass_assignments |
| `episode_idx` | int | per-pass episode number |
| `position_in_episode` | int | 0-indexed |
| `is_last_in_episode` | bool | terminal flag |
| `label` | string | 'set' or 'wait' |
| `chosen_route_id` | string or None | action label for set |
| `chosen_action_idx` | int | index into candidate_route_ids (-1 for wait) |
| `candidate_route_ids` | list[string] | feasible actions (excl. wait) |
| `n_candidates` | int | length of candidate_route_ids |
| `trigger_type` | string | 'panel_request' or 'approach' |
| 4 raw + 4 weighted reward components + `r_total` | float | from spec 01 §12 |

### 6.3 State schema

Stored in **separate columns** of the same parquet (or nested struct):

| Column | Type | Notes |
|--------|------|-------|
| `state_nodes_track` | list[struct] | per §4.3, ~10-25 entries |
| `state_nodes_signal` | list[struct] | per §4.4, ~3-10 entries |
| `state_nodes_route` | list[struct] | per §4.5, ~5-15 entries |
| `state_nodes_train` | list[struct] | per §4.6, ~1-6 entries (incl. is_focal flag) |
| `state_edges_*` | 6 + 2 lists | per §4.7 |
| `state_event_tokens` | array[256, 3] | per §4.8 |
| `state_schedule_outlook` | list[struct, 5] | per §4.9 |
| `state_special_flags` | struct (8 fields) | per §4.10 |
| `state_special_flags_meta` | struct (sources) | for assert_no_leak validation |

### 6.4 What the model loader does

```python
def load_for_model(snapshot_row):
    """Spec 03's dataset loader. Strips metadata, returns ONLY state."""
    state = {k: snapshot_row[k] for k in snapshot_row if k.startswith('state_')}
    label_for_training = (
        snapshot_row['chosen_action_idx'],   # action index in A_t
        snapshot_row['r_total'],              # scalar reward
    )
    # NEVER pass focal_signal, chosen_route_id, etc. to model
    return state, label_for_training
```

### 6.5 What the L3 simulator uses (XAI eval, spec 05)

L3 needs the FULL metadata for counterfactual rollout (e.g., "what if signaller
had chosen R' instead of R?"). The simulator is allowed to read everything —
it's a hindsight evaluation tool, not a model input.

---

## §7 Leak audit — complete `assert_no_leak()` implementation

### 7.1 Module: `src/railrl/mdp/leak_audit.py`

```python
from typing import Any
import json

# All field names that MUST NOT appear in any state_* field
BANNED_STATE_FIELDS = {
    # Direct answer fields
    "focal_signal", "focal_signal_id",
    "chosen_route_id", "chosen_action_idx",
    "focal_route", "focal_route_id",
    
    # Reward intermediates (spec 01 §14.1)
    "delay_change_seconds",
    "route_outcome", "outcome",
    "next_tc_headway_seconds", "headway_seconds",
    "n_tc_occupations_after_t",
    "T_next_occ", "T_clear",
    "arr_delay_future",
    "route_release_time",
    "r_delay", "r_throughput", "r_headway", "r_wait", "r_total",
    "r_delay_raw", "r_throughput_raw", "r_headway_raw", "r_wait_raw",
    
    # Forbidden focal markers on graph nodes
    "is_focal_signal", "is_focal_route",
    
    # Forbidden schedule details
    "planned_end_signal", "planned_signal",
    
    # Forbidden future-looking train info
    "actual_next_tc", "next_actual_timestamp",
}


def assert_no_leak(snapshot: dict, sample_meta: dict, t_ns: int) -> bool:
    """
    Snapshot construction guard. dev mode runs after every snapshot;
    production may skip for speed.
    
    Raises AssertionError on any violation, with detailed message.
    """
    # ============================================================
    # Check 1: subgraph centering
    # ============================================================
    assert snapshot["center"]["type"] == "track", \
        f"subgraph must center on a track node, got {snapshot['center']['type']}"
    expected_tc = sample_meta.get("focal_train_current_tc")
    actual_tc = snapshot["center"]["id"]
    assert actual_tc == expected_tc, \
        (f"subgraph center mismatch: expected focal_train.current_tc={expected_tc}, "
         f"got {actual_tc}. This may indicate centering on focal_signal!")
    
    # ============================================================
    # Check 2: no is_focal_signal / is_focal_route on graph nodes
    # ============================================================
    for sig_node in snapshot["state_nodes_signal"]:
        for k in sig_node:
            assert k not in {"is_focal", "is_focal_signal"}, \
                f"signal node {sig_node['signal_id']} has forbidden flag: {k}"
    
    for r_node in snapshot["state_nodes_route"]:
        for k in r_node:
            assert k not in {"is_focal", "is_focal_route", "is_chosen"}, \
                f"route node {r_node['route_id']} has forbidden flag: {k}"
    
    # ============================================================
    # Check 3: BANNED_STATE_FIELDS scan
    # ============================================================
    def scan(obj, path=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in BANNED_STATE_FIELDS:
                    raise AssertionError(
                        f"banned field '{k}' found at {path}/{k}")
                scan(v, f"{path}/{k}")
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                scan(item, f"{path}[{i}]")
    
    # Only scan state_* keys (metadata is allowed to have these)
    state_only = {k: v for k, v in snapshot.items() if k.startswith("state_")}
    scan(state_only)
    
    # ============================================================
    # Check 4: schedule_outlook does not contain signal-level info
    # ============================================================
    for tr in snapshot["state_schedule_outlook"]:
        assert "planned_end_signal" not in tr
        assert "planned_signal" not in tr
        if "planned_platform" in tr:
            p = tr["planned_platform"]
            assert p is None or (isinstance(p, int) and 1 <= p <= 6), \
                f"planned_platform must be int 1..6 or None, got {p}"
    
    # ============================================================
    # Check 5: f_trts_pressed source declaration
    # ============================================================
    flags_meta = snapshot.get("state_special_flags_meta", {})
    if "f_trts_pressed_source" in flags_meta:
        src = flags_meta["f_trts_pressed_source"]
        assert src in {"planned_platform", "current_platform"}, \
            f"f_trts_pressed must use planned/current platform, got source={src}"
    
    # ============================================================
    # Check 6: temporal causality — all events have time < t
    # ============================================================
    for tok in snapshot["state_event_tokens"]:
        # tok is (asset_idx, state, time_delta_s); time_delta_s ≥ 0 means
        # event happened ≥ 0 sec before t. < 0 would be future leak.
        assert tok["time_delta_s"] >= 0, \
            f"event token has negative time_delta_s ({tok['time_delta_s']}) — future leak"
    
    # ============================================================
    # Check 7: exactly one is_focal=True train (or zero if no focal)
    # ============================================================
    n_focal_train = sum(1 for tr in snapshot["state_nodes_train"]
                         if tr.get("is_focal", False))
    assert n_focal_train == 1, \
        f"expected exactly 1 is_focal=True train node, got {n_focal_train}"
    
    return True
```

### 7.2 When to run

- **dev mode**: every snapshot, every batch (cost ~10-50 ms per snapshot)
- **production training mode**: every 1000-th batch (sampled audit)
- **post-build batch validation**: scripts/mdp/02_validate_snapshots.py runs
  100% on the full snapshots_v2.parquet after build

### 7.3 What to do on violation

`AssertionError` should be HARD — never silently caught. Raise the error,
stop the build, fix the bug.

For schema-level violations during development, snapshot building tools can
catch and log to `outputs/snapshots/leak_violations.jsonl` for batch debugging.

---

## §8 Output schema — `snapshots_v2.parquet`

### 8.1 File location

`outputs/snapshots/snapshots_v2.parquet`

### 8.2 Row schema

Each row = one decision point. Columns:

| Group | Columns |
|-------|---------|
| **Identity** | `sample_id, focal_train, focal_signal, t, pass_id, episode_idx, position_in_episode, is_last_in_episode` |
| **Action** | `label, chosen_route_id, chosen_action_idx, candidate_route_ids, n_candidates, trigger_type` |
| **Reward** | per spec 01 §17.1: `outcome, approach_distance, delay_change_seconds, next_tc_headway_seconds, gate, r_delay_raw..r_wait_raw, r_delay..r_wait, r_total` |
| **State (nested)** | `state_nodes_track, state_nodes_signal, state_nodes_route, state_nodes_train` |
| **State (edges)** | `state_edges_connects, state_edges_traverses, state_edges_starts_at, state_edges_ends_at, state_edges_protects, state_edges_same_signal, state_edges_at_berth, state_edges_next_signal` |
| **State (sequence)** | `state_event_tokens` (256 × 3 array) |
| **State (outlook)** | `state_schedule_outlook` (list of 5 structs) |
| **State (flags)** | `state_special_flags`, `state_special_flags_meta` |

### 8.3 Expected size

~727 k rows × variable-length nested types. Estimated 800 MB - 1.5 GB.

If too large, consider:
- Sharding by month (12 monthly parquets)
- Compressing event_tokens via dictionary encoding

### 8.4 Summary JSON

`outputs/snapshots/snapshots_v2_summary.json`:

```json
{
  "n_decisions": 727432,
  "n_set": 545289,
  "n_wait": 182143,
  "n_episodes": 82429,
  "candidate_set_coverage": {
    "chosen_in_candidates": 543200,
    "chosen_not_in_candidates": 2089,
    "coverage_pct": 99.62
  },
  "subgraph_size_distribution": {
    "mean": 28, "p50": 27, "p90": 42, "p99": 58, "max": 64
  },
  "leak_audit_passed": true,
  "build_seconds": 4823.1
}
```

---

## §9 Implementation modules

### 9.1 Module map

```
src/railrl/mdp/
├── __init__.py
├── trigger.py            # §2: decision point generation
├── action.py             # §3: feasible_actions, validate_candidates
├── state.py              # §4: snapshot builder
├── episode.py            # §5: pass-based episode segmentation
├── leak_audit.py         # §7: assert_no_leak
├── special_flags.py      # §4.10: 8 flag computations + source declarations
└── schema.py             # §8: snapshots_v2.parquet writer + arrow schema
```

### 9.2 Public API contracts

```python
# trigger.py
def generate_decision_points(...) -> pd.DataFrame: ...

# action.py
def feasible_actions(focal_train, focal_signal, t, snapshot, ...) -> list[str]: ...
def validate_candidates(decision_points, static_graph) -> dict: ...

# state.py
def build_snapshot(focal_train, focal_signal, t, ...) -> dict: ...
def build_all_snapshots(decision_points, ...) -> pd.DataFrame: ...

# episode.py
def build_episodes(decision_points, pass_assignments) -> pd.DataFrame: ...

# leak_audit.py
BANNED_STATE_FIELDS: set[str]
def assert_no_leak(snapshot, sample_meta, t_ns) -> bool: ...

# special_flags.py
def compute_all_flags(focal_train, focal_signal, t, snapshot) -> dict: ...

# schema.py
SNAPSHOT_ARROW_SCHEMA: pyarrow.Schema
def write_snapshots(snapshots: pd.DataFrame, path: Path) -> None: ...
```

### 9.3 Build scripts

```
scripts/mdp/
├── 01_generate_decision_points.py    # §2 → outputs/decision_points/decision_points_v2.parquet
├── 02_validate_candidates.py         # §3 sanity: coverage ≥ 99.5%
├── 03_build_snapshots.py             # §4+§8 → outputs/snapshots/snapshots_v2.parquet
└── 04_run_leak_audit_full.py         # §7 full-data audit
```

---

## §10 Verification + sanity checks

### 10.1 Decision point sanity (after §2 build)

```python
dp = pd.read_parquet("outputs/decision_points/decision_points_v2.parquet")

assert len(dp) > 700_000
assert (dp['label'] == 'set').sum() > 540_000
assert (dp['label'] == 'wait').sum() > 175_000
assert dp['trigger_type'].isin({'panel_request', 'approach'}).all()
assert dp.groupby('focal_train').size().median() > 5  # avg ≥ 5 decisions per train
```

### 10.2 Candidate set coverage (after §3)

```python
coverage = json.load(open("outputs/decision_points/candidate_coverage.json"))
assert coverage['chosen_in_candidates_pct'] >= 99.5
```

### 10.3 Snapshot leak audit (after §8 build)

```python
summary = json.load(open("outputs/snapshots/snapshots_v2_summary.json"))
assert summary['leak_audit_passed'] is True
assert summary['n_decisions'] > 700_000
```

### 10.4 Sanity-check single snapshot

```python
import pyarrow.parquet as pq
table = pq.read_table("outputs/snapshots/snapshots_v2.parquet")
sample = table.slice(0, 1).to_pylist()[0]

# Identity present
assert 'focal_train' in sample
assert 'focal_signal' in sample

# State does NOT contain focal_signal as a field
state_keys = [k for k in sample if k.startswith('state_')]
for sk in state_keys:
    val = sample[sk]
    # recursive check
    assert 'focal_signal' not in str(val)[:5000]  # cheap scan
    assert 'chosen_route_id' not in str(val)[:5000]

# Exactly one focal train
trains = sample['state_nodes_train']
n_focal = sum(1 for t in trains if t.get('is_focal'))
assert n_focal == 1
```

---

## §11 Open questions for spec 03 to inherit

| # | Question | Default proposal |
|---|----------|------------------|
| 1 | Should `state_nodes_train` include trains outside the focal train's 3-hop subgraph but within Derby? | Yes, include all trains active in [t-30s, t]. Subgraph determines edges, but train nodes are added if their `current_tc` is anywhere in the graph. |
| 2 | Padding strategy for variable-length lists (different snapshots have different node counts) | Pad each list to a fixed cap (e.g., 60 tracks, 15 signals, 15 routes, 8 trains) with sentinel rows. Encoder mask-attends. |
| 3 | Should `state_event_tokens` use absolute `time_delta_s` or normalized (e.g., log)? | Spec 03 decides — write raw seconds, let encoder normalize. |
| 4 | How to handle samples where `focal_train.current_tc` cannot be determined (no recent CA/CB/CC) | Skip the sample, log to `outputs/snapshots/skipped_no_tc.jsonl`. Expected < 1%. |
| 5 | Should we cap `n_candidates` (e.g., max 14)? | Yes — if A_t exceeds 14 (rare; max observed ~14), truncate to top-14 by some heuristic (e.g., shortest length). Document in §3. |

---

## §12 Changelog

- **v1.0 (2026-05-19)** — Initial draft. Locks decision point trigger logic
  (PR + approach with Δ_wait=30s, K_approach=2), structured action space
  `{wait} ∪ {(focal_train, R)}`, complete state schema (per-node features +
  edges + K=256 sequence + schedule outlook + 8 flags + is_focal_train),
  episode definition (per-pass, γ=0.95), strict sample-metadata vs state
  separation, and full assert_no_leak() implementation. Awaiting sign-off.

---

**End of Spec 02.**
**Sign-off:** ☐ Hao  /  Date: ______
