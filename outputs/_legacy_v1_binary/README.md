# Legacy v1 Binary-Task Outputs (archived, do not use for training)

These outputs were generated under the **v1 binary action framing** (set vs wait, 2-class),
which has since been deprecated due to:

- **`focal_signal` leakage**: state contained the signal at which the decision occurred,
  which is part of the action — per-signal majority alone reached 91% accuracy.
- **Lost route information**: 277-route choice compressed into 2-class label.
- **No multi-train priority modeling**: binary framing cannot represent who-goes-first
  decisions.
- **Mean/std state aggregation**: topological structure of subgraphs was destroyed.

## What's in here

| Folder | Size | Description |
|---|---|---|
| `snapshots/` | half-copied (incomplete) | v1 binary 115-dim flat snapshots (476 MB total in v1). Not re-copied due to size + uselessness. Original lives at `E:\Claude\RailRL\railrl\outputs\p2_data_eng\snapshots\` if needed. |
| `decision_points/` | 6.5 MB | 726,978 (pass_id, signal, time, label) tuples with binary label. |
| `p3_modelling/` | 13 MB | B1 BC-MLP / B2 BCQ / B3 IQL trained on binary task. Final accuracy 0.91-0.93 — same as per-signal majority baseline. |

## What's reused in v2 (not in this directory)

These v1 artefacts are KEPT and live in `../`:

- `../decisions/decision_events.parquet` — PR events with `chosen_route_id` (v2's action label)
- `../infrastructure/` — 277 routes / 249 tracks / 100 signals
- `../static_graph/` — 4 node types × 6 edge types (will be re-used directly)
- `../event_stream/event_tokens.parquet` — K=256 token stream
- `../rewards/` — H_min=147s calibration + 4-component reward (re-usable)
- `../analyses/` — 3 empirical analyses (gold for paper §3 audit findings)

## How v2 differs

| Aspect | v1 binary | v2 structured |
|---|---|---|
| Action space | {set, wait} | {wait} ∪ {(train, route)} |
| `focal_signal` in state | Yes (95-dim one-hot) — **leak** | No |
| Subgraph state | mean/std aggregation | per-node feature vectors |
| Priority decision | Not modeled | Emerges from train choice in (T, R) |
| Timing decision | Not modeled | Emerges from sequence of wait vs (T, R) |
| Trivial baseline | per-signal majority = 0.91 | per-trajectory majority ≈ 0.30-0.50 |
| HG-DT-CQL ceiling | ~0.93 (1.4pp over trivial) | 0.65-0.85 expected (large room) |

See `../../docs/spec/02_mdp_formulation.md` for the v2 MDP formulation.
