# Spec 01 — Data Pipeline

**Document version:** v1.1 · **Last updated:** 2026-05-19
**Status:** 🟡 draft — §17 resolved, §17.5 added; awaiting final sign-off
**Scope:** corresponds to ESWA paper §3 (data acquisition) + §4 (data engineering)
**Authority:** this document is the canonical contract for everything between
"raw Network Rail open feeds" and "parquet files ready for MDP construction".
Any code modification touching these stages MUST first update this spec.

---

## §0 Purpose & scope

### What this spec locks down

This document defines the contract for **Stage 1 → Stage 10** of the v2 project:
the transformation from three raw CSV files (TD + Movements + route_to_tc) plus
six reference data files into a set of well-typed parquet tables that downstream
specs (02-05) consume.

### What this spec does NOT cover

- MDP formulation (action space, state schema, candidate mask) → spec **02**
- Model architecture (HGT, Transformer, Q-network) → spec **03**
- Training protocol (CQL, IQL, loss weights) → spec **04**
- Evaluation framework + XAI → spec **05**

### Out-of-scope: Phase 1 (acquisition + SOP decoding)

The raw CSV files (TD_data.csv, Movements.csv) are produced by the acquisition
+ decoding framework from PhD thesis Chapter 3. They are **inputs** to this
pipeline, not outputs. The ESWA paper will cite Chapter 3 for this layer.

---

## §1 Data sources

### 1.1 Raw open data (3 files, ~760 MB total)

Located at: `data/raw/`

| File | Size | Rows | Columns | Description |
|------|------|------|---------|-------------|
| `TD_data.csv` | 713 MB | ~11.91 M | `time, type, area_id, descr, msg_queue_timestamp, id, trainid_filled, change, state, from_berth, to_berth` | Decoded TD S-class + C-class events for Derby area. Spans 2023-02-28 to 2024-04-25. |
| `Movements.csv` | 49 MB | ~247 k | `train_id, event_type, actual_timestamp, planned_timestamp, gbtt_timestamp, loc_stanox, platform, ..., timetable_variation (30 cols total)` | TRUST realised + scheduled events for trains passing Derby area. |
| `route_to_tc_all.csv` | 32 KB | 447 | `route, start, end, track` | Per-route ordered TC list and start/end signals. |

### 1.2 Reference data (6 files)

Located at: `data/reference/`

| File | Used by | Purpose |
|------|---------|---------|
| `Derby_info.csv` | `data/derby_info.py` | Per-route physical features: `length, ave_speed(m/s), ave_grad, gap_time(s), track, path` |
| `derby_info_mapping.csv` | `data/event_stream.py` | `asset_idx (0..671) → asset_name` bidirectional index |
| `platform_end_signals.csv` | `data/static_graph.py` | Signal → platform_id mapping for "is_platform_end" feature |
| `platform_tc_map.csv` | `data/static_graph.py` | TC → platform_id + sub_section (A/middle/B) |
| `TRT1.DY2_2.SOP` | `data/sop_parser.py` | Original SOP dictionary for cross-checking |
| `derby_all.png` | `xai/l1_attention.py` (future) | Derby panel diagram for L1 visualisation |

### 1.3 Domain documents (4 files, reference only)

Located at: `data/domain/`

| File | Used by | Purpose |
|------|---------|---------|
| `Training_Plan_2022.docx` | `data/rule_base.py` (future) | Source for L4 rule base extraction (~80-120 rules) |
| `Signalling_Nomenclature.pdf` | reference | UK signal system naming standards |
| `headcode.pdf` | reference | Headcode encoding rules |
| `S-class.pdf` | reference | S-class byte-level decoding (Chapter 3) |

---

## §2 Pipeline overview (DAG)

```
                ┌─────────────────────────────────────────────────────┐
                │  RAW DATA                                           │
                │   TD_data.csv  +  Movements.csv  +  route_to_tc.csv │
                └──────────────────┬──────────────────────────────────┘
                                   │
        ┌──────────────────────────┼───────────────────────┐
        ▼                          ▼                       ▼
  Stage 1                    Stage 2                  Stage 3
  Inventory pass             PR extraction            Infrastructure parse
  (TD + Movements)           → decisions/             → infrastructure/
       │                       decision_events.        routes_clean /
       │                       parquet                 tracks / signals
       │                          │                       │
       │                          │                       ▼
       │                          │                  Stage 4
       │                          │                  Static graph build
       │                          │                  + Derby_info features
       │                          │                  → static_graph/
       │                          │                    nodes_* + edges_*
       │                          │                       │
       │                          ▼                       │
       │                     Stage 5                      │
       │                     Event token stream           │
       │                     → event_stream/              │
       │                       event_tokens.parquet       │
       │                          │                       │
       │                          ▼                       │
       │                     Stage 6                      │
       │                     Pass disambiguation          │
       │                     (TRUST id matching)          │
       │                          │                       │
       │                          ▼                       │
       └─────────► Stage 7 ◄─── Stage 8 ─── Stage 9 ◄────┘
                  Reward         PR outcome   Per-decision
                  calibration    labelling    feature builder
                  → calibration  → pr_         (delay_change +
                    .json          outcomes      next_tc_headway +
                                   .parquet      approach_distance)
                                       │            │
                                       └─────┬──────┘
                                             ▼
                                          Stage 10
                                          Final reward table
                                          → rewards/
                                            decision_rewards.parquet
                                            (THIS IS THE OUTPUT
                                             SPEC 02 CONSUMES)
```

---

## §3 Stage 1 — Inventory pass

### 3.1 Purpose

Streaming pass over `TD_data.csv` and `Movements.csv` producing descriptive
statistics. These are sanity-check artefacts, not used downstream.

### 3.2 Script: `scripts/data/01_inventory.py`

**Module:** `src/railrl/data/inventory.py`

**Behavior:**
- Stream TD CSV in 3 M-row chunks (memory: < 4 GB peak)
- Count by `type`, `prefix`, `class`, `area_id`
- Monthly profile, time range, state-distribution per type
- Stream Movements CSV (small enough to fit), count by `event_type`, `loc_stanox`

### 3.3 Outputs

| File | Format | Schema (top-level keys) |
|------|--------|-------------------------|
| `outputs/inventory/td_inventory.json` | JSON | `total_rows, time_range, type_counts, by_prefix, by_class, monthly_profile, state_by_type` |
| `outputs/inventory/movements_inventory.json` | JSON | `total_rows, time_range, event_type_counts, by_stanox` |

### 3.4 Verification

```python
import json
td = json.load(open('outputs/inventory/td_inventory.json'))
assert td['total_rows'] >= 11_000_000
assert 'Panel_Request' in td['type_counts']
assert td['type_counts']['Panel_Request'] >= 540_000  # 14 months of PRs
```

---

## §4 Stage 2 — Decision event extraction

### 4.1 Purpose

Extract every `Panel_Request` row from TD and parse its `id` into structured
fields. **This is the source of action labels** (`chosen_route_id` per decision).

### 4.2 Script: `scripts/data/02_decisions.py`

**Module:** `src/railrl/data/decisions.py`

**Algorithm:**

1. Read TD CSV with `usecols=['time', 'type', 'id', 'trainid_filled']`
2. Filter `type == 'Panel_Request'`
3. Parse `id` (e.g., `RTD5045A(M)`) using `ROUTE_RE_PATTERN`:
   - `prefix` ∈ {DW, TD, DC, EC, DY}
   - `signal_no` (numeric tail, e.g., `5045`)
   - `letter` (route letter, e.g., `A`)
   - `sub` (sub-route, optional)
   - `cls` ∈ {M, C, S, W, PS, SP} (route class)
4. Parse `trainid_filled` (4-char headcode) using `HEADCODE_RE_PATTERN`:
   - `hc_class_digit` (first char, mapped to class; 7/8 → `other`; non-4-char → `non_standard`)
   - `hc_dest`, `hc_serial`

### 4.3 Output schema

**File:** `outputs/decisions/decision_events.parquet`

| Column | Type | Description |
|--------|------|-------------|
| `time` | datetime64[ns] | PR event time |
| `route_id` | string | Full id (e.g., `RTD5045A(M)`); **this is `chosen_route_id` for v2** |
| `prefix` | string | DW/TD/DC/EC/DY |
| `signal_no` | string | Numeric signal tail |
| `letter` | string | Route letter |
| `sub` | string \| None | Sub-route |
| `cls` | string | M/C/S/W/PS/SP |
| `train_id` | string | 4-char headcode (may be non-standard) |
| `hc_class_digit` | string | "0".."9" \| "other" \| "non_standard" |
| `hc_dest` | string \| None | Destination chars |
| `hc_serial` | string \| None | Serial chars |

### 4.4 Summary JSON

**File:** `outputs/decisions/decision_events_summary.json`

| Key | Description |
|-----|-------------|
| `total_decision_events` | Should be **≥ 540,000** on full 14-month data |
| `by_prefix` | Dict: DC ~195k / TD ~145k / DW ~121k / DY ~58k / EC ~27k |
| `by_class` | Dict: M ~530k / S ~14k / C ~2.4k / SP 1 |
| `headcode_class_counts` | Dict: 1 ~365k / 2 ~78k / 5 ~62k / ... |
| `unique_train_ids` | ~2,185 |
| `unique_signals` | ~92 |
| `unique_routes` | ~250 (subset of 277 named routes actually used) |
| `time_range` | `[2023-02-28 ..., 2024-04-25 ...]` |
| `headcode_parse_rate_pct` | **≥ 99.0** |

### 4.5 Invariants

- Every row has a non-null `route_id`
- Every row has a parseable `prefix` (one of 5)
- `hc_class_digit` is one of `{0..9, other, non_standard}`
- `time` is monotonically increasing per (`route_id`, `train_id`) within reason (not strictly globally, due to event arrival jitter)

---

## §5 Stage 3 — Infrastructure parsing

### 5.1 Purpose

Parse `route_to_tc_all.csv` into structured infrastructure tables: per-route TC
list, per-track route memberships, per-signal outbound routes.

### 5.2 Script: `scripts/data/03_infrastructure.py`

**Module:** `src/railrl/data/infrastructure.py`

### 5.3 Outputs

#### 5.3.1 `outputs/infrastructure/routes_clean.parquet`

277 unique named routes (after stripping whitespace + deduplicating).

| Column | Type | Description |
|--------|------|-------------|
| `route_id` | string | e.g., `RTD5045A(M)` |
| `prefix` | string | |
| `signal_no` | string | |
| `letter` | string | |
| `sub` | string \| None | |
| `cls` | string | |
| `start_signal` | string | Numeric tail of start signal |
| `end_signal` | string | Numeric tail of end signal |
| `track_sections` | list[string] | Ordered list of TC ids |
| `n_tc` | int | Length of `track_sections` |

#### 5.3.2 `outputs/infrastructure/tracks_inventory.parquet`

249 unique TCs that appear in any named route.

| Column | Type | Description |
|--------|------|-------------|
| `track_id` | string | e.g., `TFBN` |
| `routes_using` | list[string] | All `route_id`s that traverse this TC |
| `n_routes_using` | int | Length of above |

#### 5.3.3 `outputs/infrastructure/signals_inventory.parquet`

100 unique signals with outbound routes.

| Column | Type | Description |
|--------|------|-------------|
| `signal_id` | string | Numeric tail |
| `outbound_routes` | list[string] | All `route_id`s starting from this signal |
| `n_routes_from` | int | |

#### 5.3.4 `outputs/infrastructure/auxiliary_connections.parquet`

156 non-route TC adjacencies (kept for completeness, not used in main pipeline).

#### 5.3.5 `outputs/infrastructure/infrastructure_graph.json`

Summary statistics.

### 5.4 Invariants

- Sum of `n_routes_using` across all tracks = sum of `n_tc` across all routes
- `start_signal` of every route ∈ `signal_id` of `signals_inventory`
- Stripped whitespace in `route_id` (one v1 bug source)

---

## §6 Stage 4 — Static heterogeneous graph

### 6.1 Purpose

Build the time-invariant skeleton of the heterogeneous graph that every snapshot
will hang off. **This is consumed directly by HGT in spec 03.**

### 6.2 Script: `scripts/data/04_static_graph.py`

**Module:** `src/railrl/data/static_graph.py`

### 6.3 Node tables (4 types)

| File | Rows | Key columns |
|------|------|-------------|
| `outputs/static_graph/nodes_track.parquet` | 249 | `track_id, source, n_routes_using, platform_id, platform_sub` |
| `outputs/static_graph/nodes_signal.parquet` | 123 | `signal_id, prefix, n_routes_from, is_platform_end, platform_id, platform_direction` |
| `outputs/static_graph/nodes_route.parquet` | 277 | `route_id, prefix, signal_no, letter, sub, cls, n_tc, end_signal, end_platform_id, length_m, ave_speed_mps, ave_grad, gap_time_s, n_points` |
| `outputs/static_graph/nodes_trts.parquet` | 24 | `trts_id, platform_id, side` |

### 6.4 Edge tables (6 types)

| File | Rows | Schema |
|------|------|--------|
| `edges_protects.parquet` | 100 | `signal_id, track_id` (signal → first TC of each outbound route) |
| `edges_connects.parquet` | 548 | `track_a, track_b` (symmetric; derived from route TC orderings) |
| `edges_traverses.parquet` | 1,701 | `route_id, track_id, order` (route → all its TCs with ordinal position) |
| `edges_starts_at.parquet` | 279 | `route_id, signal_id` |
| `edges_ends_at.parquet` | 290 | `route_id, signal_id` |
| `edges_same_signal.parquet` | 1,122 | `route_a, route_b` (routes sharing the same start signal) |

### 6.5 Derby_info physical-feature integration ⭐

**KEY DESIGN DECISION (documented in v1 derby_info.py docstring, preserved in v2):**

- From `Derby_info.csv` we take **ONLY physical attributes**:
  `length_m, ave_speed_mps, ave_grad, gap_time_s, n_points`
- The track list and start/end signals continue to come from `route_to_tc_all.csv`
  (Derby_info has different conventions in 69% of routes; not a bug in either, just different data sources)
- Coverage: **275 / 277 routes** have physical features (99.3%)
- These features are merged into `nodes_route.parquet` by `route_id`

**Why this matters for spec 03 model architecture:**

`gap_time_s` is the **canonical traversal time** used by the L3 reversed-event
simulator (spec 05). When the same `gap_time_s` also enters `route_emb` via the
HGT route node features, the Q-network's route scoring and the L3 simulator's
counterfactual rollout use a **physically consistent** time parameter. This is
critical for the Replicate-AND-Improve narrative.

### 6.6 Platform features

From `platform_end_signals.csv` + `platform_tc_map.csv`:

- Each signal: `is_platform_end` (bool), `platform_id` (1-6, optional), `platform_direction` (N/S)
- Each TC: `platform_id`, `sub_section` ∈ {A, middle, B}
- 18 tracks have a platform, 12 signals are platform-end, 72 routes end at a platform

### 6.7 Summary JSON

`outputs/static_graph/static_graph_summary.json` — must contain:
- `nodes`: {track: 249, signal: 123, route: 277, trts: 24}
- `edges`: {protects: 100, connects: 548, traverses: 1701, starts_at: 279, ends_at: 290, same_signal: 1122}
- `physical_features_coverage.routes_with_length`: 275

---

## §7 Stage 5 — Event token stream

### 7.1 Purpose

Tokenize the entire TD `change` column into a single time-ordered stream of
events. **This is the K=256 sequence input to the Transformer branch in spec 03.**

### 7.2 Script: `scripts/data/05_event_stream.py`

**Module:** `src/railrl/data/event_stream.py`

### 7.3 Asset index (672 assets, idx ↔ name bijection)

Loaded from `data/reference/derby_info_mapping.csv` (0-indexed `key` column).

**Asset type classification (by name prefix regex):**

| Type | Regex | Count (approx) |
|------|-------|----------------|
| Signal | `^S(DC|DW|DY|EC|TD)\d+\w*$` | ~123 |
| Route | `^R(DW|TD|DC|EC|DY)\d+\w*[A-Z]+(?:-\d+)?\((M|C|S|W|PS|SP)\)$` | ~277 |
| TRTS | `^LPLAT\d+[AB]TRS\([NS]\)$` | 24 |
| Track | `^T[A-Z0-9]+$` | ~249 |

Saved to `outputs/event_stream/asset_index.parquet` for downstream lookup.

### 7.4 Event token format

Each TD row's `change` column yields one or more tokens of form:

```
(asset_idx: int16, new_state: int8, time_ns: int64)
```

- `asset_idx` ∈ [0, 671]
- `new_state` ∈ {0, 1} (typically; some assets may have more states)
- `time_ns` from `time` column (datetime → nanoseconds since epoch)

### 7.5 Output

**File:** `outputs/event_stream/event_tokens.parquet`

| Column | Type | Description |
|--------|------|-------------|
| `time_ns` | int64 | Event timestamp (nanoseconds) |
| `asset_idx` | int16 | Asset index |
| `state` | int8 | New state |

Sorted by `time_ns` ascending. Total size: ~42 MB.

### 7.6 K=256 query semantics

For any decision point `t`, the input to the Transformer is **the last K=256 tokens
with `time_ns < t` (strict)**. The cutoff is strict-less-than to avoid leakage of
events at the exact decision moment.

K=256 chosen as power-of-2 matching Transformer context length. Empirically
covers ~17 min at typical Derby event density.

---

## §8 Stage 6 — Pass disambiguation

### 8.1 Purpose

Assign every PR (and approach event) to a unique `pass_id` = "one operational
journey of one train through Derby". Used by spec 02 for **episode boundary
definition**.

### 8.2 Algorithm

1. For each TD event with a `trainid_filled`, match to a TRUST `train_id` (10-char)
   whose:
   - `chars[2:6] == trainid_filled` (headcode match)
   - Time range `[t_first, t_last] ± PASS_LOOKUP_BUFFER_S` contains the TD event time
   - If multiple match, pick the one whose center is closest to event time
2. If no TRUST id covers a TD event (rare: freight, ECS, trains skipping all
   monitored TIPLOCs), fall back to **time-gap clustering**:
   - Group consecutive events of same `trainid_filled` separated by < `PASS_FALLBACK_GAP_S = 21,600 s` (6 hours)

### 8.3 Constants (locked)

```python
PASS_LOOKUP_BUFFER_S      = 1800   # ±30 min buffer around TRUST id range
PASS_FALLBACK_GAP_S       = 21600  # 6 h gap = new fallback pass
APPROACH_WINDOW_FORWARD_S = 600    # PR within 10 min of approach start = "during window"
```

### 8.4 Output

Pass assignment is computed **inline** during reward feature construction
(`reward_features.compute_delay_changes`). It is also exposed to spec 02 for
episode segmentation.

**TODO for spec 02:** decide whether to materialize a separate
`outputs/passes/pass_assignments.parquet` for downstream consumption.

---

## §9 Stage 7 — Reward calibration

### 9.1 Purpose

Derive the 4 empirical thresholds used by the reward function from the full
14-month dataset (NOT hand-set, NOT from external standards).

### 9.2 Script: `scripts/data/08_calibrate_rewards.py`

**Module:** `src/railrl/data/reward_calibration.py`

### 9.3 Three calibrated thresholds

| Threshold | Source | Calibration percentile | Value |
|-----------|--------|------------------------|-------|
| `H_min_seconds` | Pair-wise headway distribution on same TC (n=3.28 M) | **P5** | **147.0 s** |
| `d_gate_05_max` | Approach distance distribution (n=24,870 sampled) | **P50** | **6** hops |
| `d_gate_01_max` | Approach distance distribution | **P90** | **16** hops |
| `window_seconds` | TIPLOC lag distribution (n=42,806) | **P99** | **4201.9 s** |

### 9.4 Output

**File:** `outputs/rewards/calibration.json`

```json
{
  "headway": {
    "n_pairs": 3284641,
    "percentiles": {"p1": 91.0, "p5": 147.0, ...},
    "H_min_seconds_used": 147.0,
    "percentile_used": 5
  },
  "approach_distance": {
    "n_decisions_sampled": 50000,
    "n_with_distance": 24870,
    "percentiles": {...},
    "d_gate_breakpoints": {
      "gate_1.0_max": 2,
      "gate_0.5_max": 6,
      "gate_0.1_max": 16
    }
  },
  "tiploc_lag": {
    "n_lags": 42806,
    "percentiles": {...},
    "window_seconds_used": 4201.949999999997,
    "percentile_used": 99
  }
}
```

### 9.5 Causal gate function (final)

```
gate(d_hops) = 1.0  if d ≤ 2
             = 0.5  if 3 ≤ d ≤ 6      ← P50 boundary
             = 0.1  if 7 ≤ d ≤ 16     ← P90 boundary
             = 0.0  if d > 16
```

### 9.6 Reference values (for ESWA paper context, NOT used in code)

- Multi-aspect colour-light mainline minimum signaling headway: 90-120 s (Network Rail RIS-0786-RIG)
- Junction headway: 90-150 s typical
- TPWS overlap clearance: 30-45 s

Our empirical P5 = 147 s sits at the **upper bound of UK standard** — consistent
with reality.

---

## §10 Stage 8 — PR outcome labelling

### 10.1 Purpose

For every PR, classify the operational outcome by tracking the route's lifecycle
in the event stream. **Provides `route_outcome` for r_throughput.**

### 10.2 Script: `scripts/data/09_label_pr_outcomes.py`

**Module:** `src/railrl/data/pr_outcomes.py`

### 10.3 Algorithm

For each PR `(focal_signal, chosen_route_id, time t)`:

1. Find first event in the stream where `route_asset.state == 0` and `time > t`
   → this is the **release time** `t_release`
2. Within `[t, t_release]`, count occupations of TCs in this route
3. Classify:

| outcome | Condition | r_throughput raw |
|---------|-----------|------------------|
| `used` | At least one TC was occupied during the lock period | **+1.0** |
| `unused_cancelled` | No TC occupation AND duration < 60 s | **−1.0** |
| `unused_timeout` | No TC occupation AND duration ≥ 60 s | **−0.5** |
| `unknown` | Data ends before route releases | **0.0** |

`cancelled_threshold_seconds = 60.0` (locked).

### 10.4 Output

**File:** `outputs/rewards/pr_outcomes.parquet`

| Column | Type | Description |
|--------|------|-------------|
| `pr_index` | int | Index aligning with `decision_events.parquet` |
| `outcome` | string | One of `used / unused_cancelled / unused_timeout / unknown` |
| `route_set_duration_seconds` | float | `t_release - t` |
| `n_route_tc_occupations` | int | Count of TC state=1 events in window |

**File:** `outputs/rewards/pr_outcomes_summary.json`

Expected distribution (on 14-month data): **~99.5% used**, < 0.5% other.

---

## §11 Stage 9 — Per-decision reward features

### 11.1 Purpose

Compute the three numerical features that feed the reward model:
`delay_change_seconds`, `next_tc_headway_seconds`, `approach_distance`.

### 11.2 Script: `scripts/data/10_compute_rewards.py`

**Module:** `src/railrl/data/reward_features.py`

### 11.3 `delay_change_seconds` (for r_delay)

For each decision `(focal_train, time t)`:

1. Look up matching TRUST id by headcode + time range (§8.2)
2. Find bracket: last TIPLOC with `actual ≤ t` (index j-1), first TIPLOC with `actual > t` (index j)
3. Both endpoints must be within `window_seconds=4201.9` of t
4. `delay_change = arr_delay[j] - arr_delay[j-1]`
   where `arr_delay = actual_timestamp - planned_timestamp`
5. **Average-attribute** across all decisions sharing the same `(trust_id, bracket_j)` bucket

**Coverage:** ~62,772 / 546,418 decisions (~11.5%) have a valid `delay_change_seconds`. The rest are NaN (no matching TIPLOC bracket within window).

### 11.4 `next_tc_headway_seconds` (for r_headway)

For each `set` decision with `outcome == 'used'`:

1. Find the first TC of `chosen_route_id` (via `edges_traverses`, order=0)
2. From `t`, scan the TC's events:
   - Step 1: first `state=1` (this train occupies)
   - Step 2: next `state=0` (this train clears, time = `T_clear`)
   - Step 3: next `state=1` (next train occupies, time = `T_next_occ`)
3. `next_tc_headway_seconds = (T_next_occ - T_clear) / 1e9`

NaN if any of the three transitions can't be found.

**Coverage:** ~526,409 / 546,418 set decisions (~96.3%).

### 11.5 `approach_distance` (for r_delay causal gate)

For each `set` decision:

1. Per `focal_signal`, multi-source BFS from its protected TCs over the
   `connects` graph
2. Result: `{tc_id → hop_distance}` map per signal (cached, ~95 signals total)
3. For each decision, look up `(focal_signal, train_current_tc)` → hop distance

Filter:
- Same-day match only on `(trainid, date)` for "current TC" lookup (avoids
  matching to a previous-week run of the same headcode after data gaps)

**Coverage:** ~261,008 / 546,418 set decisions (~48%). The rest have NaN (train
position not findable on same day).

---

## §12 Stage 10 — Final reward table

### 12.1 Script: `scripts/data/11_reapply_reward_model.py`

**Module:** `src/railrl/data/reward_model.py`

### 12.2 Reward formula (LOCKED)

```python
gate(d) = 1.0 if d ≤ 2 else 0.5 if d ≤ 6 else 0.1 if d ≤ 16 else 0.0

r_delay_raw      = - gate(d) × clip(delay_change_seconds, ±1800) / 60     # in minutes
r_throughput_raw = OUTCOME_REWARD[outcome]                                 # {+1, -1, -0.5, 0}
r_headway_raw    = -1.0 if (head_seconds < 147.0) else 0.0
r_wait_raw       = -1.0 if (label == 'wait') else 0.0

r_total = w_delay      × r_delay_raw
        + w_throughput × r_throughput_raw
        + w_headway    × r_headway_raw
        + w_wait       × r_wait_raw
```

**Default weights (LOCKED):**
```python
w_delay      = 1.0
w_throughput = 0.5
w_headway    = 1.0
w_wait       = 0.3
```

Weights are mutable for IRL Stage 2 (spec 05 §5.5), but the **structure** is fixed.

### 12.3 Output

**File:** `outputs/rewards/decision_rewards.parquet`

| Column | Type | Description |
|--------|------|-------------|
| All columns of `decision_events.parquet` | — | inherited |
| `label` | string | `'set'` or `'wait'` |
| `outcome` | string | from §10 |
| `approach_distance` | float | hops, NaN if unknown |
| `delay_change_seconds` | float | from §11.3, NaN OK |
| `next_tc_headway_seconds` | float | from §11.4, NaN OK |
| `gate` | float | computed gate(d), 1.0 for wait |
| `r_delay_raw`, `r_throughput_raw`, `r_headway_raw`, `r_wait_raw` | float | per §12.2 |
| `r_delay`, `r_throughput`, `r_headway`, `r_wait` | float | weighted |
| `r_total` | float | sum |

### 12.4 Summary

**File:** `outputs/rewards/decision_rewards_summary.json`

Expected statistics (on 14-month data; reproducible):

| Metric | Expected value |
|--------|----------------|
| `n_decisions` | ~727 k (positives + wait negatives if computed via spec 02 trigger; or ~546 k positives only at this stage) |
| `r_total.mean` | ~+0.255 |
| `r_total.std` | ~0.675 |
| `r_total.range` | [-30.30, +30.50] |
| 88.4% positive episodes | (reported in health check, §13) |

### 12.5 Health check (Stage 10b)

**Script:** `scripts/data/13_reward_health_checks.py`

Generates `outputs/rewards/health/health_summary.{md,json}` with:
- Weight sensitivity sweep (conservative / default / aggressive)
- Spearman rank correlations across presets (must be > 0.85)
- Proxy correlations (Movements delay-reduction, no-cancellation)
- Top/bottom episode samples for sanity check

---

## §13 Temporal causality contract (referenced from §B.0)

**This contract governs every feature computation in this pipeline AND in spec 02 (state).**

### 13.1 Two kinds of "future" information

| Type | Source | Knowable to signaller at decision time t? | Use in **state** (spec 02) | Use in **reward** (this spec) |
|------|--------|-------------------------------------------|----------------------------|-------------------------------|
| **Scheduled** | Movements `gbtt_timestamp`, `planned_timestamp` | ✅ yes (timetable) | ✅ allowed | ✅ allowed |
| **Realized** | Movements `actual_timestamp` at t' > t; TD events at time > t | ❌ no (hindsight) | ❌ **forbidden** | ✅ allowed (return is hindsight) |

### 13.2 Pipeline-side implications

**Reward features are allowed to use realized future data:**
- `delay_change_seconds` uses `arr_delay[j]` with `actual_timestamp > t` ✓ OK
- `next_tc_headway_seconds` uses TC events with `time > t` ✓ OK
- `outcome` uses route lifecycle after t ✓ OK

**But these reward intermediates MUST NOT leak into state (spec 02):**
- See spec 02 §2.2 for the banned-feature list
- See PROJECT_HANDOFF.docx Ch 14 for the assert_no_leak() contract

### 13.3 Implementation pattern (for all downstream code)

```python
# CORRECT — state feature uses only time <= t
state_track = tracks_df[(tracks_df.id == 'TFDU') & (tracks_df.time <= t)].iloc[-1]

# CORRECT — schedule outlook uses gbtt (not actual) for t' > t
upcoming = movements[(movements.gbtt_timestamp.between(t, t + F)) &
                     (movements.loc_stanox == focal_stanox)]

# CORRECT (reward only) — uses actual at t' > t in hindsight
v_after = latest_variation(focal_train, ≤ t + H)   # OK for reward; FORBIDDEN in state
```

---

## §14 Output inventory (complete parquet list)

Every parquet file this pipeline produces, in dependency order:

```
outputs/
├── inventory/
│   ├── td_inventory.json
│   └── movements_inventory.json
├── decisions/
│   ├── decision_events.parquet            ← 546k rows, action labels source
│   ├── decision_events_summary.json
│   ├── pr_timing_audit.{json,md}          ← optional: §15 audit script
├── infrastructure/
│   ├── routes_clean.parquet               ← 277 rows
│   ├── tracks_inventory.parquet           ← 249 rows
│   ├── signals_inventory.parquet          ← 100 rows
│   ├── auxiliary_connections.parquet      ← 156 rows
│   └── infrastructure_graph.json
├── static_graph/
│   ├── nodes_track.parquet                ← 249 rows
│   ├── nodes_signal.parquet               ← 123 rows
│   ├── nodes_route.parquet                ← 277 rows (含 Derby_info 物理特征)
│   ├── nodes_trts.parquet                 ← 24 rows
│   ├── edges_protects.parquet             ← 100 rows
│   ├── edges_connects.parquet             ← 548 rows
│   ├── edges_traverses.parquet            ← 1,701 rows
│   ├── edges_starts_at.parquet            ← 279 rows
│   ├── edges_ends_at.parquet              ← 290 rows
│   ├── edges_same_signal.parquet          ← 1,122 rows
│   ├── asset_index.parquet                ← 672 rows (asset_idx ↔ name)
│   └── static_graph_summary.json
├── event_stream/
│   ├── event_tokens.parquet               ← ~3.3 M rows, time-sorted
│   └── asset_index.parquet                ← (duplicate of above for convenience)
├── rewards/
│   ├── calibration.json                   ← H_min=147, d-gate, window=4202
│   ├── headway_distribution.png
│   ├── approach_distance_distribution.png
│   ├── tiploc_lag_distribution.png
│   ├── calibration_summary.md
│   ├── pr_outcomes.parquet                ← per-PR outcome
│   ├── pr_outcomes_summary.json
│   ├── decision_rewards.parquet           ← FINAL REWARD TABLE
│   ├── decision_rewards_summary.json
│   └── health/
│       ├── health_summary.{md,json}
│       ├── component_distributions.png
│       ├── episode_return_distribution.png
│       ├── weight_sensitivity.png
│       └── top_bottom_episodes.csv
├── analyses/                              ← optional: empirical findings
│   ├── conflict_per_pr.parquet
│   ├── conflict_other_occupations.parquet
│   ├── conflict_summary.json
│   ├── route_class_crosstab.parquet
│   ├── route_class_summary.json
│   ├── non_standard_by_signal.parquet
│   └── non_standard_trainids_summary.json
└── cache/
    └── td_data.parquet                    ← 90 MB TD CSV → parquet cache
```

**Total disk:** ~150 MB (excluding raw + cache).

---

## §15 Reproducibility — full pipeline run order

### 15.1 Scripts to run (in order)

From the v2 project root:

```bash
# Stage 1: Inventory pass (~1 min)
python scripts/data/01_inventory.py

# Stage 2: Decision event extraction (~1 min)
python scripts/data/02_decisions.py

# Stage 3: Infrastructure parsing (< 1 s)
python scripts/data/03_infrastructure.py

# Stage 4: Static heterogeneous graph (< 1 s)
python scripts/data/04_static_graph.py

# Stage 5: Event token stream (~30 s)
python scripts/data/05_event_stream.py

# Stage 7: Reward calibration (~5 min, runs full 14-month percentile scans)
python scripts/data/08_calibrate_rewards.py

# Stage 8: PR outcome labelling (~3 min)
python scripts/data/09_label_pr_outcomes.py

# Stage 9 + 10: Per-decision features + final reward (~10 min)
python scripts/data/10_compute_rewards.py

# Stage 10b: Health check (~2 min)
python scripts/data/13_reward_health_checks.py

# Optional empirical analyses (~5 min total)
python scripts/data/analyses/conflict_empirical.py
python scripts/data/analyses/route_class_skew.py
python scripts/data/analyses/non_standard_trainids.py

# Optional: PR timing audit (~2 min)
python scripts/data/15_pr_timing_audit.py
```

**Total wall clock:** ~30-45 minutes on a modern workstation (CUDA not needed at this stage).

### 15.2 Expected counts (sanity-check after each stage)

| After stage | Verification |
|-------------|--------------|
| Stage 1 | `td_inventory.json.total_rows ≥ 11 M` |
| Stage 2 | `decision_events.parquet` has `≥ 540,000` rows; `headcode_parse_rate_pct ≥ 99` |
| Stage 3 | `routes_clean` 277 rows; `tracks_inventory` 249 rows |
| Stage 4 | `nodes_route.parquet` has `length_m` non-null in `≥ 275` rows |
| Stage 5 | `event_tokens.parquet` has `≥ 3 M` rows; sorted by `time_ns` |
| Stage 7 | `calibration.json.headway.H_min_seconds_used == 147.0` |
| Stage 8 | `pr_outcomes_summary.json` shows ≥ 99% `used` |
| Stage 10 | `decision_rewards.parquet.r_total.mean ≈ 0.25 ± 0.05` |

---

## §16 Verification checklist

Run after a full pipeline build:

```python
import json
import pandas as pd

# Stage 4 — static graph
ng = json.load(open('outputs/static_graph/static_graph_summary.json'))
assert ng['nodes']['track']  == 249
assert ng['nodes']['signal'] == 123
assert ng['nodes']['route']  == 277
assert ng['edges']['protects']  == 100
assert ng['edges']['connects']  == 548
assert ng['edges']['traverses'] == 1701
assert ng['physical_features_coverage']['routes_with_length'] >= 275

# Stage 7 — calibration
cal = json.load(open('outputs/rewards/calibration.json'))
assert cal['headway']['H_min_seconds_used'] == 147.0
assert cal['approach_distance']['d_gate_breakpoints']['gate_0.5_max'] == 6
assert cal['approach_distance']['d_gate_breakpoints']['gate_0.1_max'] == 16

# Stage 10 — final rewards
dr = pd.read_parquet('outputs/rewards/decision_rewards.parquet')
assert 'r_total' in dr.columns
assert dr['r_total'].mean() > 0       # majority positive
assert (dr['outcome'] == 'used').mean() > 0.95

# Stage 10b — health check passes weight-stability
hc = json.load(open('outputs/rewards/health/health_summary.json'))
assert hc['weight_spearman']['conservative_vs_default'] >= 0.85

print("✓ all stages verified")
```

---

## §17 Resolutions to open questions (LOCKED 2026-05-19)

These were proposed as open questions; user confirmed answers 2026-05-19.

| Q | Question | Resolution |
|---|----------|------------|
| 1 | Materialize `pass_assignments.parquet` separately, or compute pass_id inline? | **MATERIALIZE** — `outputs/passes/pass_assignments.parquet` is a first-class artefact, computed once during a new Stage 6, consumed by Stage 9 reward features and spec 02 (episode segmentation) |
| 2 | Should `decision_rewards.parquet` include `wait` negative samples? | **YES — merge.** Single table with `label ∈ {'set', 'wait'}` column. Spec 02 trigger logic generates wait samples; reward formula applies uniformly (r_wait_raw = −1.0 for wait) |
| 3 | Compute `approach_distance` for wait too? | **YES, compute (don't hardcode).** Run the same BFS-from-protected-TC for wait. Rationale: spec 02 may evolve approach horizon definition (K=2 → K=3 etc.); hardcoding gate=1.0 would create a hidden assumption. Cost is negligible (per-signal cache + O(1) lookup). **⚠ See §17.5 for the critical leak risk this introduces.** |
| 4 | How should `next_tc_headway_seconds` handle wait? | **NOT DEFINED for wait** (set to NaN). `r_headway_raw = 0` for wait. r_headway measures route-choice tightness; wait has no route choice. Wait's "should I have acted sooner?" failure mode is captured by r_delay (late train cascade). |

### 17.1 Updated `decision_rewards.parquet` schema (per Q2 resolution)

Add `label` column; wait rows have `route_id` set to NaN, `focal_signal` set to
the signal that triggered the wait (in METADATA, not state — see §17.5).

| Column | Type | Set rows | Wait rows |
|--------|------|----------|-----------|
| `label` | string | `'set'` | `'wait'` |
| `route_id` | string | chosen route | NaN |
| `focal_signal` | string | route.start_signal | the signal triggering wait — **metadata only, never state** |
| `focal_train` | string | trainid_filled | trainid_filled in approach |
| `outcome` | string | per §10 | always `'na'` |
| `approach_distance` | float | per §11.5 | computed, typically ≤ 2 (gate ≈ 1.0) |
| `delay_change_seconds` | float | per §11.3 | per §11.3 (wait can still affect downstream delay) |
| `next_tc_headway_seconds` | float | per §11.4 | **NaN (always)** |
| `gate` | float | gate(approach_distance) | gate(approach_distance) |
| `r_delay_raw` | float | per §12.2 | per §12.2 (uses delay_change if available) |
| `r_throughput_raw` | float | per §12.2 | **0.0 (always)** |
| `r_headway_raw` | float | per §12.2 | **0.0 (always)** |
| `r_wait_raw` | float | 0.0 | **−1.0** |

### 17.2 New pipeline stage: Stage 6 — pass assignment

Insert a new stage between current Stage 5 (event stream) and Stage 7 (reward
calibration):

**Script:** `scripts/data/06_assign_passes.py` (new file, to be written)
**Module:** `src/railrl/data/pass_assignment.py` (already in v1, copy to v2)
**Inputs:** `TD_data.csv` + `Movements.csv` + `derby_info_mapping.csv`
**Algorithm:** §8.2 (TRUST id matching + gap-fallback)
**Output:** `outputs/passes/pass_assignments.parquet`

| Column | Type | Description |
|--------|------|-------------|
| `time_ns` | int64 | TD event time |
| `trainid_filled` | string | from TD `trainid_filled` |
| `pass_id` | string | TRUST `train_id` if matched, else `"FB:{trainid}:{cluster_idx}"` |
| `pass_source` | string | `'trust_match'` or `'fallback_gap'` |
| `pass_t_first_ns` | int64 | start time of the pass |
| `pass_t_last_ns` | int64 | end time of the pass |

**Updated run order:**

```
Stage 1: 01_inventory.py
Stage 2: 02_decisions.py
Stage 3: 03_infrastructure.py
Stage 4: 04_static_graph.py
Stage 5: 05_event_stream.py
Stage 6: 06_assign_passes.py     ← NEW
Stage 7: 08_calibrate_rewards.py
Stage 8: 09_label_pr_outcomes.py
Stage 9: 10_compute_rewards.py   ← now consumes pass_assignments.parquet
Stage 10b: 13_reward_health_checks.py
```

---

## §17.5 NEW LEAK AUDIT — critical risks introduced by Q3 resolution

User flagged 2026-05-19: **"决策 signal 必须是从轨迹算/学出来的，不能给"**.
Even computing `approach_distance` for wait (Q3) requires knowing which signal
the wait was about — which is implicitly the answer. We must rigorously separate
**sample metadata** (where focal_signal lives, used for reward computation) from
**state features** (where focal_signal is FORBIDDEN).

### 17.5.1 The signal/route name overlap risk

UK signalling nomenclature creates inherent overlap:
- Signal `STD5045` ↔ Route `RTD5045A(M)` (start signal of this route = STD5045, "5045" digit shared)
- Route `RDC5063C(M)` starts from signal `SDC5063`

**A model that can parse numeric tails from text IDs can trivially infer:**
- start_signal of any candidate route → "5063"
- focal_signal (if leaked into state) → "5063"
- → if both are "5063" and that's the only matching signal in the snapshot → model knows this candidate IS the chosen one

**Defence (mandatory in spec 02):** state features MUST NOT contain ANY of:
- `focal_signal` text or numeric ID
- `is_focal_signal` boolean on any signal node
- `is_focal_route` boolean on any route node
- `chosen_route_id` (label itself)
- `focal_route` numeric ID
- Any "focus indicator" on route or signal nodes

### 17.5.2 Allowed: focal_train markers

In contrast, **`is_focal_train` on train nodes IS allowed**. Train identity does
not determine route — the model needs to know which train to focus on because
the decision is about that train. The model must INFER the focal signal/route
from focal_train's trajectory + graph topology.

### 17.5.3 Sample metadata vs. state features — strict separation

Going forward, every decision sample has two parts:

**Sample metadata** (used by reward calculation, episode segmentation, trigger
logic; **NEVER passed to the model**):
- `focal_signal` — the signal this sample is about
- `focal_train` — train identity (also in state, but flagged is_focal there)
- `t` — decision timestamp
- `label` — set/wait
- `pass_id` — episode identifier
- `chosen_route_id` (set rows only) — action label
- `approach_distance` — reward input
- `delay_change_seconds`, `next_tc_headway_seconds` — reward inputs
- `outcome` — reward input (set rows only)
- 4 raw + 4 weighted reward components, `r_total`

**State features** (passed to the model encoder; spec 02 §2 will define schema):
- Per-node features (track / signal / route / train) within 3-hop subgraph
  centered on **focal_train.current_tc** (NOT on focal_signal)
- Edge tables (6 types) within that subgraph
- K=256 event tokens (globally last 256 with `time_ns < t`)
- Schedule outlook (gbtt only)
- 8 special-case flags
- `is_focal_train` flag on the train node ONLY

### 17.5.4 Audit checklist for additional leak risks (carry to spec 02)

| Risk | Defence |
|------|---------|
| Subgraph centering on focal_signal | **LOCKED:** center on `focal_train.current_tc`, never on `focal_signal`. Spec 02 §2.3 must enforce. |
| `is_focal_signal` / `is_focal_route` boolean on graph nodes | **FORBIDDEN.** Only `is_focal_train` allowed. |
| Filtering edges/events by focal_signal proximity | **FORBIDDEN.** Use all edges within subgraph; use globally last 256 events. |
| `f_trts_pressed` flag pointing to focal_signal's platform | **LOCKED:** `f_trts_pressed` uses `T.planned_platform` (from schedule) or `T.current_platform` (from current_tc → platform_tc_map). NEVER from focal_signal's platform. |
| Candidate route_id reveals start_signal | **NOT leak** — candidate identity is what the model scores. Model learns to score (candidate-T pair) based on context. Different from giving focal_signal as "the answer". |
| `pass_id` revealing future episode end time | **NOT leak in feature** — pass_id used only for episode segmentation at training time, not as a state feature. Spec 02 must enforce. |
| Schedule outlook leaking "this train is going to platform 3" | **OK if** `planned_platform` is used (it's public schedule). NOT OK if `planned_end_signal` (specific signal ID of platform end) — signal ID is too close to focal_signal. Use only `platform_id ∈ {1..6}`, never signal ID. |

### 17.5.5 Implementation contract

`src/railrl/mdp/leak_audit.py` (to be created in spec 02) MUST enforce all of
§17.5.4 at snapshot build time. Pseudocode:

```python
def assert_no_leak(snapshot, sample_meta, t):
    # Existing checks (from Ch 14 of PROJECT_HANDOFF.docx)
    # ...
    
    # NEW (from spec 01 §17.5):
    
    # 1. Subgraph centering check
    assert snapshot.center_node_type == 'train', \
        f"subgraph must center on train, got {snapshot.center_node_type}"
    assert snapshot.center_node_id == sample_meta.focal_train, \
        "subgraph must center on focal_train"
    
    # 2. No is_focal_signal / is_focal_route
    for sig_node in snapshot.nodes_signal:
        assert 'is_focal' not in sig_node and 'is_focal_signal' not in sig_node
    for r_node in snapshot.nodes_route:
        assert 'is_focal' not in r_node and 'is_focal_route' not in r_node
    
    # 3. focal_signal numeric ID must not appear in any top-level state field
    fs = str(sample_meta.focal_signal)
    # Allow fs to appear as part of candidate route_id (RTD5045A(M) — necessary),
    # but NOT as standalone signal_id field
    assert 'focal_signal' not in snapshot.flat_field_names()
    assert 'focal_signal_id' not in snapshot.flat_field_names()
    
    # 4. Schedule outlook uses platform_id, not signal_id
    for tr in snapshot.schedule_outlook:
        assert 'planned_end_signal' not in tr
        if 'planned_platform' in tr:
            assert isinstance(tr['planned_platform'], int)  # 1..6
    
    # 5. f_trts_pressed source check
    for flag_meta in snapshot.special_flags_meta:
        if flag_meta['flag'] == 'f_trts_pressed':
            assert flag_meta['source_platform'] in {'planned', 'current'}, \
                "f_trts_pressed must use planned_platform or current_platform"
    
    return True
```

### 17.5.6 Why this matters for ESWA

ESWA reviewers will scrutinize "how did you avoid label leakage in offline RL?"
This expanded leak audit gives a defensible, code-enforced answer. The specific
signal/route name overlap risk is a Derby-domain idiosyncrasy that demonstrates
careful attention to the real data — exactly the rigour ESWA values.

---

## §18 Changelog

- **v1.0 (2026-05-19 早)** — Initial draft. Locks all parquet schemas,
  calibration thresholds (147 s / 6 / 16 / 4201.9 s), reward weights
  (1.0/0.5/1.0/0.3), pipeline run order, and verification checklist.
- **v1.1 (2026-05-19 晚)** — User resolved §17 open questions:
  - Q1: materialize pass_assignments.parquet (new Stage 6)
  - Q2: merge wait negatives into decision_rewards.parquet
  - Q3: compute approach_distance for wait too
  - Q4: next_tc_headway undefined for wait (NaN, r_headway=0)
  - **§17.5 added** — critical leak audit triggered by Q3 resolution:
    signal/route naming overlap risk, sample-metadata vs state-features strict
    separation, mandatory `assert_no_leak()` enforcement for spec 02.

---

**End of Spec 01.**
**Sign-off:** ☐ Hao  /  Date: ______
