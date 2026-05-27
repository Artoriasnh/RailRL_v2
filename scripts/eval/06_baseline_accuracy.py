"""Stage 7 — non-learned baseline IMITATION ACCURACY (spec 05 §3 Table I rows B0/B0').

The paper's Table I compares each method's top-1 match with the signaller, per stratum.
This fills the NON-LEARNED baseline rows — no model, no GPU, no FQE; pure pandas/pyarrow
over snapshots_v2. The learned rows (CQL = B4/B5) already come from eval/01_evaluate_model.

Baselines (all computed self-contained from the snapshot itself):
  * B0  random         : uniform over the legal action set {wait} ∪ candidates.
                         Reported as the ANALYTIC expected top-1 accuracy mean[1/(n_cand+1)]
                         (deterministic — no sampling noise).
  * B0' planned-platform-preferring ("traj prior", Hao's choice): prefer the candidate whose
                         end_platform_id == focal planned_platform; if none matches, fall back
                         to the FIRST candidate (always routes). Mimics "follow the timetable".
                         focal planned_platform ← state_nodes_train (is_focal node); candidate
                         end_platforms ← state_nodes_route by route_id. (else-FIRST, not
                         else-WAIT — the latter is degenerate on sparse end_platform data.)
  * B0'' first-candidate: always set the first candidate (greedy, never wait); wait only if
                         no candidate. The simplest "always act" reference.

Metric (matches eval/01 conventions): top-1 action match vs signaller (chosen_action_idx),
overall + per stratum, BOTH all-decisions and set-only. Strata from stratum_labels.parquet
(0 late_train/1 advance/2 call_on/3 platform_dev/4 priority_compete/5 unusual_id/6 trivial).

Read-only, streaming by row-group (bounded memory). Run on Windows:
    python scripts/eval/06_baseline_accuracy.py
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from railrl import config as C

STRATUM_NAMES = {0: "late_train", 1: "advance", 2: "call_on", 3: "platform_dev",
                 4: "priority_compete", 5: "unusual_id", 6: "trivial"}
TRIVIAL = 6


def load_strata():
    """sample_id -> stratum (int). Empty dict if sidecar missing (→ overall only)."""
    import pyarrow.parquet as pq
    p = C.SNAPSHOTS_DIR / "stratum_labels.parquet"
    if not p.exists():
        print(f"  [strata] {p} missing → overall only")
        return {}
    t = pq.read_table(str(p), columns=["sample_id", "stratum"])
    return dict(zip(t.column("sample_id").to_pylist(), t.column("stratum").to_pylist()))


def focal_planned_platform(train_nodes):
    """planned_platform of the is_focal train node (None if absent)."""
    for nd in (train_nodes or []):
        if nd.get("is_focal"):
            return nd.get("planned_platform")
    return None


def route_end_platforms(route_nodes):
    """{route_id(str): end_platform_id(int or None)} from the route nodes."""
    out = {}
    for nd in (route_nodes or []):
        rid = nd.get("route_id")
        if rid is not None:
            out[str(rid)] = nd.get("end_platform_id")
    return out


def b0p_action(cands, ep_map, pp):
    """B0' planned-platform-PREFERRING action (1-based into cands; 0 = wait).
    Prefer the first candidate whose end_platform == focal planned_platform; if none
    matches (or pp unknown), fall back to the FIRST candidate (always route when one
    exists). The else-FIRST fallback avoids the degenerate else-WAIT version that, on the
    sparse end_platform data (~28% of routes known), collapsed to ~always-wait (9% set rate,
    0.5% set-only) — a wait-majority mirage rather than a routing heuristic."""
    if pp is not None:
        for i, rid in enumerate(cands):
            if ep_map.get(str(rid)) == pp:
                return i + 1                  # action idx: 0=wait, 1..K=candidates
    return 1 if cands else 0                  # fallback: first candidate, else wait


class Acc:
    """Per-stratum accumulator: sums of match (or expected match) + counts, all & set-only."""
    def __init__(self):
        self.s = {}                            # stratum -> dict of running sums

    def add(self, stratum, is_set, b0r_p, b0p_m, b0pp_m):
        for key in (stratum, "overall"):
            d = self.s.setdefault(key, dict(n=0, n_set=0,
                                            b0r=0.0, b0p=0, b0pp=0,
                                            b0r_set=0.0, b0p_set=0, b0pp_set=0))
            d["n"] += 1
            d["b0r"] += b0r_p; d["b0p"] += int(b0p_m); d["b0pp"] += int(b0pp_m)
            if is_set:
                d["n_set"] += 1
                d["b0r_set"] += b0r_p; d["b0p_set"] += int(b0p_m); d["b0pp_set"] += int(b0pp_m)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-rows", type=int, default=0, help=">0 = smoke cap")
    args = ap.parse_args()
    import pyarrow.parquet as pq

    strata = load_strata()
    pf = pq.ParquetFile(str(C.SNAPSHOTS_V2_PARQUET))
    cols = ["sample_id", "split", "label", "chosen_action_idx", "candidate_route_ids",
            "n_candidates", "state_nodes_train", "state_nodes_route"]
    acc = Acc()
    n_seen = 0
    pp_known = 0                               # focal planned_platform known
    b0p_set_cnt = 0                            # how often B0' actually sets (not wait)
    print(f"streaming snapshots_v2 ({pf.num_row_groups} row groups), split={args.split} ...")
    for rg in range(pf.num_row_groups):
        tb = pf.read_row_group(rg, columns=cols)
        d = tb.to_pydict()
        m = len(d["sample_id"])
        for i in range(m):
            if d["split"][i] != args.split:
                continue
            sid = int(d["sample_id"][i])
            chosen = int(d["chosen_action_idx"][i])
            cands = [str(x) for x in (d["candidate_route_ids"][i] or [])]
            ncand = int(d["n_candidates"][i]) if d["n_candidates"][i] is not None else len(cands)
            is_set = chosen > 0
            pp = focal_planned_platform(d["state_nodes_train"][i])
            ep_map = route_end_platforms(d["state_nodes_route"][i])
            if pp is not None:
                pp_known += 1
            # B0 random: expected top-1 match = 1/(#legal actions) = 1/(n_cand+1)
            b0r_p = 1.0 / (ncand + 1)
            # B0' planned-platform
            a_b0p = b0p_action(cands, ep_map, pp)
            if a_b0p > 0:
                b0p_set_cnt += 1
            b0p_m = (a_b0p == chosen)
            # B0'' first-candidate (greedy; wait only if no candidate)
            a_b0pp = 1 if ncand >= 1 else 0
            b0pp_m = (a_b0pp == chosen)
            st = strata.get(sid, TRIVIAL)
            acc.add(st, is_set, b0r_p, b0p_m, b0pp_m)
            n_seen += 1
            if args.max_rows and n_seen >= args.max_rows:
                break
        if args.max_rows and n_seen >= args.max_rows:
            break

    print(f"\nrows: {n_seen:,} | focal planned_platform known: {pp_known:,} "
          f"({100*pp_known/max(n_seen,1):.1f}%) | B0' sets a route: {b0p_set_cnt:,} "
          f"({100*b0p_set_cnt/max(n_seen,1):.1f}%)")

    def row(d, which):
        n, ns = d["n"], d["n_set"]
        if which == "all":
            return d["b0r"]/max(n,1), d["b0p"]/max(n,1), d["b0pp"]/max(n,1), n
        return d["b0r_set"]/max(ns,1), d["b0p_set"]/max(ns,1), d["b0pp_set"]/max(ns,1), ns

    order = ["overall", 0, 1, 2, 3, 4, 5, 6]
    print("\n=== Table I — non-learned baseline top-1 accuracy (vs signaller) ===")
    for mode in ("all", "set"):
        print(f"\n--- {mode}-decisions top-1 ---")
        print(f"  {'stratum':16s} {'n':>8} | {'B0 random':>10} {'B0 plat':>9} {'B0 first':>9}")
        for k in order:
            if k not in acc.s:
                continue
            b0r, b0p, b0pp, n = row(acc.s[k], mode)
            name = "overall" if k == "overall" else STRATUM_NAMES[k]
            print(f"  {name:16s} {n:>8,} | {100*b0r:>9.1f}% {100*b0p:>8.1f}% {100*b0pp:>8.1f}%")

    out = {"split": args.split, "n": n_seen, "pp_known_frac": pp_known/max(n_seen,1),
           "b0_definitions": {"B0_random": "uniform over {wait}+candidates (analytic 1/(ncand+1))",
                              "B0p_planned_platform": "first candidate w/ end_platform==focal planned_platform else wait",
                              "B0pp_first_candidate": "always first candidate else wait"},
           "table": {}}
    for k in order:
        if k not in acc.s:
            continue
        name = "overall" if k == "overall" else STRATUM_NAMES[k]
        ba = row(acc.s[k], "all"); bs = row(acc.s[k], "set")
        out["table"][name] = {
            "n": ba[3], "n_set": bs[3],
            "B0_random": {"all": ba[0], "set": bs[0]},
            "B0p_planned": {"all": ba[1], "set": bs[1]},
            "B0pp_first": {"all": ba[2], "set": bs[2]},
        }
    C.EVAL_DIR.mkdir(parents=True, exist_ok=True)
    p = C.EVAL_DIR / "baseline_accuracy_table.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\n→ wrote {p}")
    print("(model rows = CQL from eval/01_evaluate_model; combine for the full Table I.)")
    print("NOTE: compare methods on SET-only — all-decisions is confounded by each method's "
          "wait/set propensity (signaller waits ~73%; a wait-heavy baseline scores high on 'all' "
          "for the wrong reason). SET-only = the real routing comparison.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
