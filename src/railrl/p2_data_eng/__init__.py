"""Back-compat shim — old v1 scripts import from railrl.p2_data_eng.*

In v2, the modules live in railrl.data.* (cleaner naming).
This shim re-exports them so existing scripts/data/04-15 work unchanged.

Migration path: when scripts are eventually rewritten to use railrl.data.*
directly, this shim can be removed.
"""
from __future__ import annotations
from importlib import import_module

# All modules that v1 scripts reference
_MODULES = [
    "decisions",
    "derby_info",
    "episodes",
    "event_stream",
    "infrastructure",
    "inventory",
    "pr_outcomes",
    "reward_calibration",
    "reward_features",
    "reward_model",
    "sop_parser",
    "static_graph",
    "td_state",
]

# Re-export each module under railrl.p2_data_eng.<name>
import sys as _sys
for _name in _MODULES:
    _mod = import_module(f"railrl.data.{_name}")
    _sys.modules[f"railrl.p2_data_eng.{_name}"] = _mod

# Modules that don't exist yet in v2 — let import fail with a helpful message
# (decision_points, snapshot, batch_snapshots, pass_assignment, event_snapshot
#  are spec 02 scope, will be added in Stage 2-3)

