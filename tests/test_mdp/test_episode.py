"""Unit tests for railrl.mdp.episode."""
import pandas as pd
import pytest

from railrl.mdp.episode import (
    build_episodes, _assign_pass_by_gap, episode_returns, summarize_episodes,
)


class TestAssignPassByGap:
    def test_single_train_no_gap(self):
        df = pd.DataFrame({
            "focal_train": ["1S49"] * 3,
            "t": pd.to_datetime(["2024-01-01 10:00", "2024-01-01 10:05",
                                  "2024-01-01 10:10"]),
        })
        out = _assign_pass_by_gap(df, gap_seconds=1800.0)
        # All within 30-min → 1 pass
        assert out["pass_id"].nunique() == 1
        assert out["pass_id"].iloc[0] == "FB:1S49:0"

    def test_single_train_with_gap_splits(self):
        df = pd.DataFrame({
            "focal_train": ["1S49"] * 3,
            "t": pd.to_datetime(["2024-01-01 10:00", "2024-01-01 10:05",
                                  "2024-01-01 22:00"]),  # 12-hour gap
        })
        out = _assign_pass_by_gap(df, gap_seconds=1800.0)
        # 12h gap > 30min → 2 passes
        assert out["pass_id"].nunique() == 2

    def test_two_trains_independent(self):
        df = pd.DataFrame({
            "focal_train": ["1S49", "2A28", "1S49"],
            "t": pd.to_datetime(["2024-01-01 10:00", "2024-01-01 10:05",
                                  "2024-01-01 10:10"]),
        })
        out = _assign_pass_by_gap(df, gap_seconds=1800.0)
        # 2 unique trains → 2 unique pass_ids
        assert out["pass_id"].nunique() == 2


class TestBuildEpisodes:
    def test_fallback_assigns_pass_and_episode_idx(self):
        df = pd.DataFrame({
            "focal_train": ["1S49", "1S49", "2A28"],
            "t": pd.to_datetime(["2024-01-01 10:00", "2024-01-01 10:05",
                                  "2024-01-01 10:10"]),
        })
        out = build_episodes(df, pass_assignments=None)
        assert "pass_id" in out.columns
        assert "episode_idx" in out.columns
        assert "position_in_episode" in out.columns
        assert "is_last_in_episode" in out.columns
        assert out["episode_idx"].nunique() == 2  # 2 trains → 2 episodes

    def test_position_in_episode_starts_at_0(self):
        df = pd.DataFrame({
            "focal_train": ["1S49"] * 3,
            "t": pd.to_datetime(["2024-01-01 10:00", "2024-01-01 10:05",
                                  "2024-01-01 10:10"]),
        })
        out = build_episodes(df, pass_assignments=None)
        out = out.sort_values("position_in_episode")
        assert out["position_in_episode"].tolist() == [0, 1, 2]

    def test_is_last_marks_last_only(self):
        df = pd.DataFrame({
            "focal_train": ["1S49"] * 3,
            "t": pd.to_datetime(["2024-01-01 10:00", "2024-01-01 10:05",
                                  "2024-01-01 10:10"]),
        })
        out = build_episodes(df, pass_assignments=None)
        assert int(out["is_last_in_episode"].sum()) == 1


class TestJoinPassAssignments:
    """spec 01 §17.2 — TRUST interval matching + gap-clustered fallback."""

    def _pa(self):
        # One TRUST pass for 1S49 covering 10:00–10:20 on 2024-01-01.
        t0 = int(pd.Timestamp("2024-01-01 10:00").value)
        t1 = int(pd.Timestamp("2024-01-01 10:20").value)
        return pd.DataFrame({
            "trainid_filled":  ["1S49"],
            "pass_id":         ["851S49ME01"],
            "pass_t_first_ns": [t0],
            "pass_t_last_ns":  [t1],
            "pass_source":     ["trust_match"],
        })

    def test_decision_inside_interval_gets_trust_pass(self):
        df = pd.DataFrame({
            "focal_train": ["1S49", "1S49"],
            "t": pd.to_datetime(["2024-01-01 10:05", "2024-01-01 10:15"]),
        })
        out = build_episodes(df, pass_assignments=self._pa())
        assert (out["pass_id"] == "851S49ME01").all()
        assert out["episode_idx"].nunique() == 1  # one TRUST pass → one episode

    def test_unmatched_decisions_gap_cluster_not_collapse(self):
        # Two decisions far OUTSIDE the TRUST interval, 2 days apart →
        # must become TWO fallback episodes (not one giant FB:1S49:0).
        df = pd.DataFrame({
            "focal_train": ["1S49", "1S49"],
            "t": pd.to_datetime(["2024-02-01 10:00", "2024-02-03 10:00"]),
        })
        out = build_episodes(df, pass_assignments=self._pa())
        assert out["pass_id"].str.startswith("FB:").all()
        # 2-day gap > PASS_FALLBACK_GAP_S (6h) → 2 distinct fallback passes
        assert out["pass_id"].nunique() == 2

    def test_mixed_matched_and_fallback(self):
        df = pd.DataFrame({
            "focal_train": ["1S49", "1S49"],
            "t": pd.to_datetime(["2024-01-01 10:05",   # inside interval
                                  "2024-06-01 10:00"]),  # way outside
        })
        out = build_episodes(df, pass_assignments=self._pa())
        ids = set(out["pass_id"])
        assert "851S49ME01" in ids                       # TRUST match
        assert any(p.startswith("FB:") for p in ids)     # fallback for the other


class TestEpisodeReturns:
    def test_zero_rewards_yield_zero(self):
        df = pd.DataFrame({
            "pass_id": ["P1", "P1"],
            "position_in_episode": [0, 1],
            "r_total": [0.0, 0.0],
        })
        ret = episode_returns(df, gamma=0.95)
        assert ret["P1"] == 0.0

    def test_basic_discounted(self):
        df = pd.DataFrame({
            "pass_id": ["P1", "P1"],
            "position_in_episode": [0, 1],
            "r_total": [1.0, 1.0],
        })
        ret = episode_returns(df, gamma=0.95)
        # γ^0·1 + γ^1·1 = 1 + 0.95 = 1.95
        assert abs(ret["P1"] - 1.95) < 1e-9


class TestSummarize:
    def test_basic_stats(self):
        df = pd.DataFrame({
            "pass_id": ["P1", "P1", "P2"],
            "focal_train": ["1S49", "1S49", "2A28"],
            "t": pd.to_datetime(["2024-01-01 10:00", "2024-01-01 10:05",
                                  "2024-01-01 10:10"]),
        })
        s = summarize_episodes(df)
        assert s["n_decisions"] == 3
        assert s["n_episodes"] == 2
        assert s["decisions_per_episode"]["mean"] == 1.5
