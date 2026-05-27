"""P2.6 event-driven simulator (spec 05 §14) — L3 counterfactual + Tier-3 engine.

Parametric, NO learning, EVAL-ONLY (spec §9.1 "simulator ≠ training playground").
Given an initial Scenario (active trains positioned on TC paths) + the empirical
parameter tables (scripts/simulator/01_estimate_parameters.py → parameters.json),
rolls the system forward H minutes via an event queue → occupancy timeline + metrics.

Model (v1 — validated/iterated via scripts/simulator/02_validate_simulator.py):
  * A train advances along its `path` (ordered TC ids) one TC at a time.
  * Time on a TC = tc_traversal_time[tc] p50 (TD occupation duration) + platform_dwell
    p50 when the TC is a platform TC (fallback (plat,cls)→plat→global median).
  * A train may ENTER the next TC only when it is FREE and ≥ min_headway[tc] p5 since
    the previous train entered it (physical spacing). Else it waits and is woken when
    the TC frees (event-driven, no polling).
  * Completes when it clears the last TC of its path → throughput++.
Metrics (spec §14.4): throughput, occupancy timeline (for validation Spearman /
agreement), mean finish-delay vs planned, headway_waits.

The core `simulate()` is pure-python (no I/O) → unit-testable. Scenario construction
from real data lives in 02_validate_simulator.py + the Tier-3 driver. See spec §14.6.1
for the validation gate (throughput/occupancy primary; delay best-effort).
"""
from __future__ import annotations
import heapq
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class Train:
    train_id: str
    path: list                       # ordered TC ids the train will traverse
    cls: str = "0"
    idx: int = 0                     # index of CURRENT TC in path
    entered_ns: int = 0              # when it entered the current TC
    planned_finish_ns: Optional[int] = None
    done: bool = False


@dataclass
class Scenario:
    t0_ns: int
    trains: list                     # list[Train], each on path[idx] at t0
    platform_of: dict = field(default_factory=dict)   # tc(str) -> platform int


def _median(vals, default):
    v = [x for x in vals if x is not None]
    return float(np.median(v)) if v else float(default)


class L3Simulator:
    GENERIC_TRAVERSAL = 35.0         # s, per-TC fallback
    GENERIC_HEADWAY = 120.0          # s, fallback
    GENERIC_DWELL = 120.0            # s, fallback

    def __init__(self, params: dict, headway_pctl: str = "p1"):
        # headway_pctl="p1" = CALIBRATED default (02_validate 2026-05-25): lowering the
        # spacing floor from p5→p1 lifted throughput Spearman 0.62→0.86 + path-progress
        # 79.5%→88.5% (p5 was over-restrictive). p1 ≈ min but robust to per-TC outliers.
        self.headway_pctl = headway_pctl          # calibration knob: min/p1/p5/p10/...
        mh = params.get("min_headway", {})
        tt = params.get("tc_traversal_time", {})
        dw = params.get("platform_dwell_time", {})
        self.headway = {tc: c[headway_pctl] for tc, c in mh.items()
                        if isinstance(c, dict) and c.get(headway_pctl) is not None}
        self.traversal = {tc: c["p50"] for tc, c in tt.items()
                          if isinstance(c, dict) and c.get("p50") is not None}
        self.dwell_pc, _byp = {}, {}
        for key, c in dw.items():
            if not (isinstance(c, dict) and c.get("p50") is not None):
                continue
            plat, cls = key.split("|")
            self.dwell_pc[(plat, cls)] = c["p50"]
            _byp.setdefault(plat, []).append(c["p50"])
        self.dwell_p = {p: float(np.median(v)) for p, v in _byp.items()}
        self._def_trav = _median(self.traversal.values(), self.GENERIC_TRAVERSAL)
        self._def_hw = _median(self.headway.values(), self.GENERIC_HEADWAY)
        self._def_dwell = _median(self.dwell_p.values(), self.GENERIC_DWELL)

    @classmethod
    def from_json(cls, path, headway_pctl: str = "p1"):
        return cls(json.loads(Path(path).read_text()), headway_pctl=headway_pctl)

    def _trav(self, tc):
        return self.traversal.get(str(tc), self._def_trav)

    def _hw(self, tc):
        return self.headway.get(str(tc), self._def_hw)

    def _dwell(self, plat, cls):
        if plat is None:
            return 0.0
        plat = str(plat)
        v = self.dwell_pc.get((plat, cls))
        if v is None:
            v = self.dwell_p.get(plat)
        return float(v) if v is not None else self._def_dwell

    def _time_on_tc(self, tc, scen, cls):
        t = self._trav(tc)
        plat = scen.platform_of.get(str(tc))
        if plat is not None:
            t += self._dwell(plat, cls)
        return t

    def simulate(self, scen: Scenario, horizon_min: float = 30.0) -> dict:
        end_ns = scen.t0_ns + int(horizon_min * 60 * 1e9)
        occ: dict = {}               # tc -> train_id
        last_enter: dict = {}        # tc -> ns (headway reference)
        wait: dict = {}              # tc -> [train_id,...] waiting to enter
        # work on COPIES — simulate must NOT mutate the input Train objects, because
        # l3_delta reuses the same `others` list across the two scenarios (else the 2nd
        # run sees the 1st run's done/idx → garbage delta, e.g. +6.3 throughput artifact).
        trains = {tr.train_id: replace(tr) for tr in scen.trains}
        timeline = []                # (tc, enter_ns, train_id)
        heap = []                    # (ready_ns, train_id) advance attempts

        for tr in trains.values():
            cur = str(tr.path[tr.idx])
            occ[cur] = tr.train_id
            last_enter[cur] = tr.entered_ns or scen.t0_ns
            ready = (tr.entered_ns or scen.t0_ns) + int(self._time_on_tc(cur, scen, tr.cls) * 1e9)
            heapq.heappush(heap, (ready, tr.train_id))

        completed = headway_waits = steps = 0
        finish_delays = []
        max_steps = 500_000

        def _free_and_wake(tc, t):
            occ.pop(tc, None)
            q = wait.get(tc)
            if q:
                heapq.heappush(heap, (t, q.pop(0)))    # re-checks headway when it fires

        while heap and heap[0][0] <= end_ns and steps < max_steps:
            steps += 1
            t, tid = heapq.heappop(heap)
            tr = trains[tid]
            if tr.done:
                continue
            cur_tc = str(tr.path[tr.idx])
            if tr.idx + 1 >= len(tr.path):               # clears last TC → completes
                _free_and_wake(cur_tc, t)
                tr.done = True
                completed += 1
                if tr.planned_finish_ns:
                    finish_delays.append((t - tr.planned_finish_ns) / 1e9)
                continue
            nxt = str(tr.path[tr.idx + 1])
            if nxt in occ:                               # blocked: wait, woken on free
                wait.setdefault(nxt, [])
                if tid not in wait[nxt]:
                    wait[nxt].append(tid)
                continue
            since = (t - last_enter.get(nxt, -(10 ** 18))) / 1e9
            hw = self._hw(nxt)
            if since < hw:                               # headway not yet satisfied
                headway_waits += 1
                heapq.heappush(heap, (last_enter[nxt] + int(hw * 1e9), tid))
                continue
            _free_and_wake(cur_tc, t)                    # leave current TC
            occ[nxt] = tid
            last_enter[nxt] = t
            timeline.append((nxt, t, tid))
            tr.idx += 1
            tr.entered_ns = t
            heapq.heappush(heap, (t + int(self._time_on_tc(nxt, scen, tr.cls) * 1e9), tid))

        return {
            "throughput": completed,
            "n_trains": len(scen.trains),
            "headway_waits": headway_waits,
            "mean_finish_delay_s": float(np.mean(finish_delays)) if finish_delays else None,
            "timeline": timeline,
            "occupied_tc_count_end": len(occ),
            "steps": steps,
        }


def l3_delta(sim: L3Simulator, scen_a: Scenario, scen_b: Scenario,
             horizon_min: float = 30.0) -> dict:
    """Tier-3 counterfactual: metrics(action_a) − metrics(action_b). Positive
    throughput delta / negative delay delta ⇒ a is better. (scen_a/scen_b differ
    only in the one re-routed train's path.)"""
    ma = sim.simulate(scen_a, horizon_min)
    mb = sim.simulate(scen_b, horizon_min)
    return {
        "throughput_delta": ma["throughput"] - mb["throughput"],
        "delay_delta_s": ((ma["mean_finish_delay_s"] or 0.0) - (mb["mean_finish_delay_s"] or 0.0)),
        "a": ma, "b": mb,
    }
