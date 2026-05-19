"""Smoke tests — all v2 modules importable, no syntax errors, no missing deps."""
import pytest


def test_import_railrl_root():
    import railrl
    assert hasattr(railrl, "__version__")


def test_import_config():
    from railrl import config
    assert config.PROJECT_ROOT.exists()


def test_import_parsers():
    from railrl.parsers import parse_route_id
    r = parse_route_id("RTD5045A(M)")
    assert r is not None
    assert r.prefix == "TD"
    assert r.signal_no == "5045"
    assert r.letter == "A"
    assert r.cls == "M"


def test_import_data_io():
    from railrl import data_io
    assert hasattr(data_io, "load_route_to_tc")


# ============================================================
# All data/ modules
# ============================================================

DATA_MODULES = [
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
    "static_graph_view",  # v2 new (extracted from v1 snapshot.py)
    "td_state",
]


@pytest.mark.parametrize("name", DATA_MODULES)
def test_import_data_module(name):
    """Each railrl.data.<name> module imports without error."""
    mod = __import__(f"railrl.data.{name}", fromlist=[name])
    assert mod is not None


# ============================================================
# Back-compat shim for v1 script paths
# ============================================================

def test_p2_data_eng_shim_static_graph():
    """v1 scripts import via railrl.p2_data_eng.* — shim must work."""
    from railrl.p2_data_eng.static_graph import run
    assert callable(run)


def test_p2_data_eng_shim_reward_model():
    from railrl.p2_data_eng.reward_model import RewardModel, RewardThresholds
    assert RewardModel is not None


def test_p2_data_eng_shim_event_stream():
    from railrl.p2_data_eng.event_stream import AssetIndex, EventTokenStream
    assert AssetIndex is not None


# ============================================================
# CLI entry points
# ============================================================

def test_cli_imports():
    from railrl.cli import (
        inventory_main, decisions_main, infrastructure_main
    )
    assert callable(inventory_main)
    assert callable(decisions_main)
    assert callable(infrastructure_main)


def test_cli_show_paths(capsys):
    """railrl-inventory --show-paths runs without errors."""
    from railrl.cli import inventory_main
    rc = inventory_main(["--show-paths"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "PROJECT_ROOT" in out
    assert "DATA_DIR" in out
