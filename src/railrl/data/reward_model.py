"""P2.4 Iter C - Reward model: assemble four components into r_total.

The model is a pure function of pre-computed per-decision features:
    label, approach_distance, route_outcome,
    delay_change_seconds, next_tc_headway_seconds.

Decoupling features from weights lets us swap weights at IRL Stage 2
without recomputing features.
"""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config as C


# Outcome from Iter B route lifecycle classification -> r_thru raw value
OUTCOME_REWARD = {
    "used":             1.0,
    "unused_cancelled": -1.0,
    "unused_timeout":   -0.5,
    "unknown":          0.0,
}


@dataclass
class RewardThresholds:
    H_min_seconds: float
    d_gate_05_max: int
    d_gate_01_max: int
    window_seconds: float

    @classmethod
    def from_calibration(cls, path=None):
        path = Path(path or C.CALIBRATION_JSON)
        cal = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            H_min_seconds=float(cal["headway"]["H_min_seconds_used"]),
            d_gate_05_max=int(cal["approach_distance"]["d_gate_breakpoints"]["gate_0.5_max"]),
            d_gate_01_max=int(cal["approach_distance"]["d_gate_breakpoints"]["gate_0.1_max"]),
            window_seconds=float(cal["tiploc_lag"]["window_seconds_used"]),
        )


@dataclass
class RewardModel:
    """Stateless reward model.  weights are mutable; swap them for IRL Stage 2.

    delay_clip_seconds caps |delay_change_seconds| to suppress data-artefact
    outliers (cancelled trains, multi-day repositioning, malformed Movements
    records). Default 1800s (30 min) keeps ~99.5% of the empirical delay
    distribution intact and bounds r_delay magnitude.
    """
    weights: dict
    thresholds: RewardThresholds
    delay_clip_seconds: float = 1800.0

    @classmethod
    def from_config(cls, *, weights=None, calibration_path=None,
                     delay_clip_seconds: float = 1800.0):
        w = dict(weights or C.REWARD_WEIGHTS_DEFAULT)
        thr = RewardThresholds.from_calibration(calibration_path)
        return cls(w, thr, delay_clip_seconds)

    def causal_gate_array(self, distances):
        """Vectorised gate(d).  distances: array of float (NaN = unknown)."""
        d = np.asarray(distances, dtype=float)
        out = np.zeros_like(d, dtype=float)
        valid = ~np.isnan(d)
        # Order matters: tighter buckets overwrite looser ones.
        out[valid & (d <= self.thresholds.d_gate_01_max)] = 0.1
        out[valid & (d <= self.thresholds.d_gate_05_max)] = 0.5
        out[valid & (d <= 2)] = 1.0
        return out

    def compute_batch(self, df):
        """Apply model to a feature DataFrame and return raw + weighted columns.

        Required columns:
            label                       'set' | 'wait'
            approach_distance           float (NaN OK)
            route_outcome               str or NaN
            delay_change_seconds        float (NaN = unmeasurable)
            next_tc_headway_seconds     float (NaN = unmeasurable)

        Returns DataFrame (index aligned with df).
        """
        is_set = (df["label"].values == "set")

        # Gate: set -> from approach_distance; wait -> by construction d <= 2 -> 1.0
        gate_set  = self.causal_gate_array(df["approach_distance"].values)
        gate_wait = np.ones(len(df), dtype=float)
        gate = np.where(is_set, gate_set, gate_wait)

        # r_delay (raw, in minutes for stability) — clip extreme outliers
        delay_s = df["delay_change_seconds"].values.astype(float)
        delay_s_clipped = np.clip(delay_s, -self.delay_clip_seconds,
                                              self.delay_clip_seconds)
        delay_min = delay_s_clipped / 60.0
        r_delay_raw = -gate * delay_min
        r_delay_raw = np.where(np.isnan(delay_s), 0.0, r_delay_raw)

        # r_thru (raw): from outcome lookup; only for set decisions
        outcome_arr = df["route_outcome"].astype("string").fillna("unknown")
        r_thru_arr = outcome_arr.map(OUTCOME_REWARD).fillna(0.0).values.astype(float)
        r_thru_raw = np.where(is_set, r_thru_arr, 0.0)

        # r_head (raw): -1 if measured headway < H_min, else 0
        head_s = df["next_tc_headway_seconds"].values.astype(float)
        r_head_raw = np.where(np.isnan(head_s), 0.0,
                               np.where(head_s < self.thresholds.H_min_seconds,
                                         -1.0, 0.0))

        # r_wait (raw): -1 for wait, 0 for set
        r_wait_raw = np.where(is_set, 0.0, -1.0)

        out = pd.DataFrame({
            "gate":           gate,
            "r_delay_raw":    r_delay_raw,
            "r_thru_raw":     r_thru_raw,
            "r_head_raw":     r_head_raw,
            "r_wait_raw":     r_wait_raw,
            "r_delay":        self.weights["w_delay"]      * r_delay_raw,
            "r_throughput":   self.weights["w_throughput"] * r_thru_raw,
            "r_headway":      self.weights["w_headway"]    * r_head_raw,
            "r_wait":         self.weights["w_wait"]       * r_wait_raw,
        }, index=df.index)
        out["r_total"] = (out["r_delay"] + out["r_throughput"]
                           + out["r_headway"] + out["r_wait"])
        return out
