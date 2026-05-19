"""Tests for src/railrl/config.py — path resolution + existence checks."""
import os
import pytest
from pathlib import Path

from railrl import config as C


# ============================================================
# Path resolution
# ============================================================

def test_project_root_is_v2_root():
    """PROJECT_ROOT should resolve to the RailRL_v2 directory."""
    assert C.PROJECT_ROOT.exists()
    assert C.PROJECT_ROOT.name == "RailRL_v2"


def test_data_dir_resolves_to_raw():
    """DATA_DIR should point to data/raw/ (v2 layout)."""
    assert C.DATA_DIR.exists()
    assert C.DATA_DIR.name == "raw"
    assert C.DATA_DIR.parent.name == "data"


def test_reference_dir_resolves_to_reference():
    """REFERENCE_DIR should point to data/reference/ (v2 layout)."""
    assert C.REFERENCE_DIR.exists()
    assert C.REFERENCE_DIR.name == "reference"


def test_domain_dir_resolves_correctly():
    assert C.DOMAIN_DIR.exists()
    assert C.DOMAIN_DIR.name == "domain"


def test_outputs_dir_exists():
    assert C.OUTPUTS_DIR.exists()


# ============================================================
# Required input files
# ============================================================

def test_td_csv_exists():
    assert C.TD_CSV.exists(), f"TD_data.csv missing at {C.TD_CSV}"


def test_movements_csv_exists():
    assert C.MOVEMENTS_CSV.exists(), f"Movements.csv missing at {C.MOVEMENTS_CSV}"


def test_route_to_tc_csv_exists():
    assert C.ROUTE_TO_TC_CSV.exists()


def test_derby_info_csv_exists():
    assert C.DERBY_INFO_CSV.exists()


def test_derby_info_mapping_exists():
    assert C.DERBY_INFO_MAPPING_CSV.exists()


def test_sop_file_exists():
    assert C.SOP_FILE.exists()


def test_platform_csvs_exist():
    assert C.PLATFORM_END_SIGNALS_CSV.exists()
    assert C.PLATFORM_TC_MAP_CSV.exists()


def test_derby_panel_image_exists():
    """L1 attention visualization will need this."""
    assert C.DERBY_PANEL_PNG.exists()


def test_training_plan_exists():
    """L4 rule base extraction (spec 05 §13) needs this."""
    assert C.TRAINING_PLAN_DOCX.exists()


# ============================================================
# Verify helper
# ============================================================

def test_verify_paths_resolved_passes():
    """C.verify_paths_resolved() should pass on a well-set-up v2 repo."""
    C.verify_paths_resolved()


# ============================================================
# Locked constants (these are CONTRACTS — changing them breaks downstream)
# ============================================================

class TestLockedConstants:
    """These values are locked in spec 01-05; any change here is a spec amendment."""

    def test_reward_weights(self):
        assert C.REWARD_WEIGHTS_DEFAULT == {
            "w_delay":      1.0,
            "w_throughput": 0.5,
            "w_headway":    1.0,
            "w_wait":       0.3,
        }

    def test_event_token_k(self):
        assert C.EVENT_TOKEN_K == 256

    def test_approach_k_hops(self):
        assert C.APPROACH_K_HOPS == 2

    def test_decision_lookahead(self):
        assert C.DECISION_LOOKAHEAD_SECONDS == 30

    def test_pass_constants(self):
        assert C.PASS_LOOKUP_BUFFER_S == 1800
        assert C.PASS_FALLBACK_GAP_S == 21600
        assert C.APPROACH_WINDOW_FORWARD_S == 600

    def test_calibration_percentiles(self):
        assert C.HEADWAY_PERCENTILE_FOR_HMIN == 5
        assert C.APPROACH_PERCENTILE_LOW == 50
        assert C.APPROACH_PERCENTILE_HIGH == 90
        assert C.TIPLOC_LAG_PERCENTILE == 99

    def test_mdp_constants(self):
        assert C.DISCOUNT_GAMMA == 0.95
        assert C.SUBGRAPH_HOPS == 3
        assert C.TIME_WINDOWS_MINUTES == (1, 5, 10, 15, 30)
        assert C.SCHEDULE_LOOKAHEAD_MIN == 15
        assert C.SCHEDULE_OUTLOOK_TOPK == 5

    def test_padding_caps(self):
        assert C.MAX_TRACKS_PADDED == 60
        assert C.MAX_SIGNALS_PADDED == 15
        assert C.MAX_ROUTES_PADDED == 15
        assert C.MAX_TRAINS_PADDED == 8
        assert C.MAX_CANDIDATES_PADDED == 14

    def test_training_constants(self):
        assert C.CQL_ALPHA == 5.0
        assert C.CQL_TARGET_TAU == 0.005
        assert C.IQL_EXPECTILE_TAU == 0.7
        assert C.IQL_AWR_BETA == 3.0
        assert C.SEEDS == (42, 43, 44)
        assert C.BATCH_SIZE == 256
        assert C.LR == 3e-4
        assert C.WARMUP_STEPS == 1000
        assert C.GRAD_CLIP == 1.0
        assert C.PHASE_A_EPOCHS == 5
        assert C.PHASE_B_EPOCHS == 15
        assert C.PHASE_C_EPOCHS == 20

    def test_split_dates(self):
        assert C.TRAIN_START == "2023-02-28"
        assert C.TRAIN_END == "2024-01-31"
        assert C.VAL_START == "2024-02-01"
        assert C.VAL_END == "2024-02-29"
        assert C.TEST_START == "2024-03-01"
        assert C.TEST_END == "2024-04-25"

    def test_xai_eval_constants(self):
        assert C.L3_HORIZON_MIN == 30
        assert C.L3_DELTA_THRESH == 0.5
        assert C.L2_FAITHFULNESS_THRESH == 0.7

    def test_domain_constants(self):
        assert C.DERBY_PREFIXES == ["DW", "TD", "DC", "EC", "DY"]
        assert C.HEADCODE_NON_STANDARD == "non_standard"
        assert "1" in C.HEADCODE_CLASS and "Express" in C.HEADCODE_CLASS["1"]
        assert C.ROUTE_CLASS["M"] == "Main route"
        assert C.ROUTE_CLASS["C"] == "Call-on route (permissive working)"


# ============================================================
# Env var overrides
# ============================================================

def test_env_var_override_data_dir(tmp_path, monkeypatch):
    """RAILRL_DATA_DIR env var should override path resolution."""
    monkeypatch.setenv("RAILRL_DATA_DIR", str(tmp_path))
    # Reload config to pick up env var
    import importlib
    from railrl import config as _c
    importlib.reload(_c)
    try:
        assert _c.DATA_DIR == tmp_path
    finally:
        # Restore
        monkeypatch.delenv("RAILRL_DATA_DIR")
        importlib.reload(_c)
