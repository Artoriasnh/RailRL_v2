"""railrl.mdp — MDP formulation per spec 02.

Modules:
    trigger        — decision point generation (PR + approach)
    action         — feasible_actions + candidate validation
    special_flags  — 8 flag computations for state features

The state schema, snapshot builder, leak audit, episode + schema modules
will be added in subsequent stages:
    state.py        (snapshot construction, spec 02 §4)
    leak_audit.py   (assert_no_leak, spec 02 §7)
    episode.py      (spec 02 §5)
    schema.py       (parquet schemas, spec 02 §8)
"""
from . import trigger, action, special_flags

__all__ = ["trigger", "action", "special_flags"]
