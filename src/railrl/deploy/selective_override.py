"""§12 Selective Override deployment rule (spec 05 §12).

The deployment-facing decision rule that decides, per signaller decision, whether to:
  * 'agreement'        -- model agrees with the signaller (show an agreement badge)
  * 'consider-override'-- model disagrees AND all three confidence gates pass (show a card)
  * 'silent'           -- model disagrees but a gate fails (stay quiet, don't distract)

Three gates (all must pass to surface an override, spec §12.1):
  gate_l3 : L3 counterfactual reward improvement of model's action over signaller's > δ_L3 (0.5)
  gate_l4 : L4 rule-compliance of the model's action == 'compliant'
  gate_l2 : L2 SHAP faithfulness > 0.7

L2 FAITHFULNESS (spec §12.2): zero the highest-attributed feature group, recompute the Q-gap,
the actual drop should approximate that group's SHAP value:
    faithfulness = 1 - |actual_drop - predicted_drop| / |predicted_drop|
NOTE: our L2 is EXACT Shapley (64 coalitions), so the SHAP value is the AVERAGE marginal over
all coalitions; the "drop when removing the top group from the FULL set" is the marginal at one
specific coalition. They coincide only when the group's marginal is coalition-independent (no
interactions). So this check is still meaningful for exact Shapley -- it measures how
interaction-free the top group's contribution is. (Empirically expected high for route-dominated
gaps.) Reported honestly either way.

The gate rule + evaluator are PURE PYTHON (sandbox-tested). `l2_faithfulness` needs the model
forward (torch/PyG) and reuses railrl.xai.l2_qdecomp; run via scripts/eval/13_selective_override.py.
"""
from __future__ import annotations

from typing import Optional

DELTA_L3 = 0.5                  # spec §12.4: reward-unit threshold for L3 improvement
FAITHFULNESS_THRESHOLD = 0.7    # spec §12.4

VERDICTS = ("agreement", "consider-override", "silent")


def _gate_l4(l4_status: Optional[str], mode: str) -> bool:
    """gate_l4 under two口径:
      'refined' (DEFAULT, Hao-approved 2026-05-27): pass unless the model's route VIOLATES a
        rule — i.e. status != 'non-compliant'. Rationale: ~99% of decisions are 'no-rule'
        (the 19 Plan rules cover only specific signals); the gate's intent is "don't suggest
        an override that breaks a rule", so 'no-rule'/'compliant'/'policy-applies' should pass.
      'literal' (spec §12.1 verbatim): pass only if status == 'compliant'. Reported for contrast
        (degenerates to ~0 here because 'no-rule' dominates)."""
    if mode == "literal":
        return l4_status == "compliant"
    return (l4_status is not None) and (l4_status != "non-compliant")


def selective_override(signaller_action: int,
                       model_action: int,
                       l2_faithfulness_val: Optional[float],
                       l3_delta: Optional[float],
                       l4_status: Optional[str],
                       l4_mode: str = "refined",
                       delta_l3: float = DELTA_L3) -> tuple:
    """spec §12.1 rule. Returns (verdict, detail_dict).

    signaller_action / model_action: action indices (0=wait, 1..K=candidate slot).
    l2_faithfulness_val: float in [~0,1] or None.  l3_delta: reward-unit improvement of the
    model's action over the signaller's (None if not simulated).  l4_status: l4_check hard_status.
    l4_mode: 'refined' (non-compliant blocks; default) or 'literal' (only 'compliant' passes).
    delta_l3: L3 reward-improvement threshold (default 0.5 = spec; lower = sensitivity contrast).
    """
    if model_action == signaller_action:
        return "agreement", {"reason": "model matches signaller"}

    gate_l3 = (l3_delta is not None) and (l3_delta > delta_l3)
    gate_l4 = _gate_l4(l4_status, l4_mode)
    gate_l2 = (l2_faithfulness_val is not None) and (l2_faithfulness_val > FAITHFULNESS_THRESHOLD)
    detail = {"gate_l3": bool(gate_l3), "gate_l4": bool(gate_l4), "gate_l2": bool(gate_l2),
              "l3_delta": l3_delta, "l4_status": l4_status, "l2_faithfulness": l2_faithfulness_val,
              "l4_mode": l4_mode}

    if gate_l3 and gate_l4 and gate_l2:
        return "consider-override", detail
    return "silent", detail


def l2_faithfulness(model, data, decomp, device="cpu") -> dict:
    """spec §12.2 faithfulness of the L2 SHAP attribution.
    decomp = output of l2_qdecomp.q_gap_decomposition (has 'shap', 'q_gap', a_star, a_runner).
    Zero the highest-|SHAP| group, recompute the Q-gap (same a*/a'), compare the drop to that
    group's SHAP value. Returns {top_group, predicted_drop, actual_drop, faithfulness}."""
    import torch
    from ..xai.l2_qdecomp import GROUPS, _baseline_subset

    shap = decomp["shap"]
    top = max(GROUPS, key=lambda g: abs(shap[g]))
    a_star, a_runner = decomp["a_star"], decomp["a_runner"]
    present = set(GROUPS) - {top}                       # all groups EXCEPT the top one
    model.eval()
    with torch.no_grad():
        q = model(_baseline_subset(data, present).to(device))["Q"].view(-1)
    qgap_without_top = float(q[a_star] - q[a_runner])
    actual_drop = decomp["q_gap"] - qgap_without_top     # how much the gap fell w/o top group
    predicted_drop = float(shap[top])                    # SHAP says this group contributes this
    denom = abs(predicted_drop)
    faith = (1.0 - abs(actual_drop - predicted_drop) / denom) if denom > 1e-9 else 1.0
    return {"top_group": top, "predicted_drop": predicted_drop,
            "actual_drop": actual_drop, "faithfulness": float(faith)}


def _tally(records: list, mode: str, delta_l3: float = DELTA_L3) -> dict:
    """One pass of the rule under a given l4_mode + δ_L3 → counts/rates/gate-failures/examples."""
    n = len(records)
    counts = {v: 0 for v in VERDICTS}
    silent_fail = {"l3": 0, "l4": 0, "l2": 0}   # which gate(s) failed among silent disagreements
    examples = []
    for r in records:
        verdict, detail = selective_override(
            r["signaller_action"], r["model_action"],
            r.get("l2_faithfulness"), r.get("l3_delta"), r.get("l4_status"),
            l4_mode=mode, delta_l3=delta_l3)
        counts[verdict] += 1
        if verdict == "silent":
            if not detail["gate_l3"]:
                silent_fail["l3"] += 1
            if not detail["gate_l4"]:
                silent_fail["l4"] += 1
            if not detail["gate_l2"]:
                silent_fail["l2"] += 1
        elif verdict == "consider-override" and len(examples) < 20:
            examples.append({**detail, **{k: r.get(k) for k in
                            ("sample_id", "focal_train", "focal_signal",
                             "signaller_route", "model_route", "stratum")}})
    disagree = counts["consider-override"] + counts["silent"]
    return {
        "counts": counts,
        "rates": {v: (counts[v] / n if n else 0.0) for v in VERDICTS},
        "disagreement_n": disagree,
        "override_rate_of_disagreements": (counts["consider-override"] / disagree) if disagree else 0.0,
        "silent_gate_failures": silent_fail,
        "override_examples": examples,
    }


def evaluate_selective_override_on_test(records: list, l4_mode: str = "refined",
                                        delta_l3_grid=(0.5, 0.25, 0.1)) -> dict:
    """spec §12.3 deployment statistics, with a δ_L3 SENSITIVITY sweep.
    records: list of dicts, each with keys signaller_action, model_action, l2_faithfulness,
    l3_delta, l4_status (+ optional meta).
    For each δ_L3 in the grid × each gate_l4 口径 ('refined' primary / 'literal' contrast) we
    tally agreement/consider-override/silent. PRIMARY result = (δ_L3=0.5, refined). The lower
    δ rows are an appendix sensitivity (more, lower-confidence override suggestions)."""
    sweep = {}
    for d in delta_l3_grid:
        sweep[f"{d:g}"] = {"refined": _tally(records, "refined", d),
                           "literal": _tally(records, "literal", d)}
    return {
        "n": len(records),
        "primary": {"delta_l3": DELTA_L3, "l4_mode": l4_mode},
        "thresholds": {"delta_l3_default": DELTA_L3, "faithfulness": FAITHFULNESS_THRESHOLD,
                       "delta_l3_grid": list(delta_l3_grid)},
        "sweep": sweep,
    }
