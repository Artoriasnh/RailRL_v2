"""railrl.mdp — MDP formulation per spec 02.

Modules:
    trigger          — decision point generation (PR + approach)
    action           — feasible_actions + candidate validation
    special_flags    — 8 flag computations for state features
    episode          — per-pass episode segmentation (spec 02 §5)
    leak_audit       — assert_no_leak with 7 checks (spec 02 §7)
    schema           — snapshots_v2.parquet schema (spec 02 §8)
    pass_assignment  — TRUST id matching → pass_id (spec 01 §17.2)
    state_helpers    — TrainStateLookup + SubgraphExtractor
    state            — SnapshotBuilder (spec 02 §4)
"""
from . import (
    trigger, action, special_flags,
    episode, leak_audit, schema,
    pass_assignment, state_helpers, state,
)

__all__ = [
    "trigger", "action", "special_flags",
    "episode", "leak_audit", "schema",
    "pass_assignment", "state_helpers", "state",
]
