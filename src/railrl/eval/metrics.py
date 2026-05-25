"""spec 05 §2-§3 — evaluation metrics (Tier 1 overall + Tier 2 per-stratum).

Pure numpy: every function takes arrays of collected predictions + labels and
returns plain-python dicts (JSON-friendly). No torch / pyarrow, so it is
unit-testable in the sandbox. The driver (scripts/eval/01_evaluate_model.py)
runs a forward pass over the test split, gathers the arrays, and calls these.

Array conventions (one entry per test decision, aligned by position):
  chosen        (N,) int  chosen_action_idx in {0..K}; 0 = wait, j = route j.
  q_argmax      (N,) int  argmax_a Q(s,a) over the (masked) action set.
  route_argmax  (N,) int  RouteHead argmax in {0..K-1} (route slot).
  time_pred     (N,) int  TimeHead argmax bucket in {0..4}.
  time_bucket   (N,) int  ground-truth bucket in {0..4}; -1 = no label (excluded).
  stratum       (N,) int  priority stratum 0..6 (see STRATUM_NAMES); -1 = unknown.
  chosen_q      (N,) float Q(s, chosen)        — for Q-gap sanity.
  secondbest_q  (N,) float max_{a != chosen, valid} Q(s, a); -inf/NaN if none.

Two top-1 definitions are reported side by side because spec 04 (training) and
spec 05 §2.1 differ:
  action_top1_all  — over ALL decisions (== trainer.evaluate's action_acc; this is
                     the number comparable to the .984 val figure).
  action_top1_set  — over SET decisions only (spec 05 §2.1 "where ground truth is
                     meaningful"); wait handled separately via recall/precision.
"""
from __future__ import annotations

import numpy as np

# int code → stratum name (locked in scripts/mdp/16_build_stratum_labels.py).
STRATUM_NAMES = {
    0: "late_train",
    1: "advance",
    2: "call_on",
    3: "platform_dev",
    4: "priority_compete",
    5: "unusual_id",
    6: "trivial",
}


def _acc(pred: np.ndarray, truth: np.ndarray) -> float:
    """Mean exact-match accuracy; NaN on empty."""
    if pred.size == 0:
        return float("nan")
    return float((pred == truth).mean())


def overall_metrics(chosen, q_argmax, route_argmax=None, time_pred=None,
                    time_bucket=None) -> dict:
    """Tier 1 overall metrics (spec 05 §2)."""
    chosen = np.asarray(chosen)
    q_argmax = np.asarray(q_argmax)
    n = int(chosen.size)
    set_mask = chosen > 0
    pred_wait = q_argmax == 0
    true_wait = chosen == 0

    tp = int((pred_wait & true_wait).sum())
    out = {
        "n": n,
        "n_set": int(set_mask.sum()),
        "n_wait": int(true_wait.sum()),
        # both top-1 definitions (see module docstring)
        "action_top1_all": _acc(q_argmax, chosen),
        "action_top1_set": _acc(q_argmax[set_mask], chosen[set_mask]),
        # wait as positive class
        "wait_rate_signaller": float(true_wait.mean()) if n else float("nan"),
        "wait_rate_model": float(pred_wait.mean()) if n else float("nan"),
        "wait_rate_delta": float(pred_wait.mean() - true_wait.mean()) if n else float("nan"),
        "wait_recall": tp / max(int(true_wait.sum()), 1),
        "wait_precision": tp / max(int(pred_wait.sum()), 1),
    }
    # route head (set rows only): route_argmax vs chosen-1
    if route_argmax is not None and set_mask.any():
        ra = np.asarray(route_argmax)[set_mask]
        out["route_head_top1"] = _acc(ra, (chosen[set_mask] - 1))
    else:
        out["route_head_top1"] = float("nan")
    # time head (valid-label rows only)
    if time_pred is not None and time_bucket is not None:
        tb = np.asarray(time_bucket)
        tvalid = tb >= 0
        out["n_time_valid"] = int(tvalid.sum())
        out["time_head_top1"] = _acc(np.asarray(time_pred)[tvalid], tb[tvalid]) \
            if tvalid.any() else float("nan")
    return out


def _stratum_cell(chosen_s, qarg_s) -> dict:
    """Per-group action top-1, reported BOTH over all decisions and set-only.

    `acc_set` is the honest hard-case signal: per-stratum `acc_all` is inflated
    by the wait majority within the stratum (a stratum that is mostly wait gets a
    high acc_all from trivially-correct waits). `acc_set` isolates the route-choice
    accuracy on the decisions where the signaller actually set a route.
    """
    chosen_s = np.asarray(chosen_s)
    qarg_s = np.asarray(qarg_s)
    set_mask = chosen_s > 0
    return {
        "n": int(chosen_s.size),
        "n_set": int(set_mask.sum()),
        "n_wait": int((~set_mask).sum()) if chosen_s.size else 0,
        "acc_all": _acc(qarg_s, chosen_s),
        "acc_set": _acc(qarg_s[set_mask], chosen_s[set_mask]),
    }


def stratified_top1(chosen, q_argmax, stratum) -> dict:
    """Tier 2: per-stratum action top-1 (acc_all + acc_set + set/wait split).

    Mirrors the spec 05 §3.1 Table-I row structure (overall + 7 priority strata).
    Strata are the single priority-assigned code; TRTS/Freight overlap slices are
    deferred (they need flag re-derivation from state_special_flags).
    """
    chosen = np.asarray(chosen)
    q_argmax = np.asarray(q_argmax)
    stratum = np.asarray(stratum)

    out = {"overall": _stratum_cell(chosen, q_argmax)}
    for code, name in STRATUM_NAMES.items():
        m = stratum == code
        out[name] = _stratum_cell(chosen[m], q_argmax[m])
    n_unknown = int((stratum < 0).sum())
    if n_unknown:
        out["_unknown_stratum_n"] = n_unknown
    return out


def qgap_sanity(chosen_q, secondbest_q) -> dict:
    """Q-value sanity (spec 05 §2.4): is chosen-action Q above the best alternative?

    Over rows where a finite second-best exists, report mean gap (chosen − 2nd)
    and the fraction where chosen ≥ 2nd (i.e. the chosen action IS the argmax).
    A healthy model has positive mean gap and a high fraction.
    """
    cq = np.asarray(chosen_q, dtype=float)
    sq = np.asarray(secondbest_q, dtype=float)
    finite = np.isfinite(cq) & np.isfinite(sq)
    if not finite.any():
        return {"n": 0, "mean_gap": float("nan"), "frac_chosen_is_argmax": float("nan")}
    gap = cq[finite] - sq[finite]
    return {
        "n": int(finite.sum()),
        "mean_gap": float(gap.mean()),
        "p5_gap": float(np.percentile(gap, 5)),
        "p95_gap": float(np.percentile(gap, 95)),
        "frac_chosen_is_argmax": float((gap >= 0).mean()),
    }


def evaluate_all(chosen, q_argmax, stratum, *, route_argmax=None, time_pred=None,
                 time_bucket=None, chosen_q=None, secondbest_q=None) -> dict:
    """Assemble the full report (Tier 1 + Tier 2 + Q-gap) from collected arrays."""
    report = {
        "tier1_overall": overall_metrics(chosen, q_argmax, route_argmax,
                                         time_pred, time_bucket),
        "tier2_stratified": stratified_top1(chosen, q_argmax, stratum),
    }
    if chosen_q is not None and secondbest_q is not None:
        report["q_gap"] = qgap_sanity(chosen_q, secondbest_q)
    return report
