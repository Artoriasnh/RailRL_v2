"""L4 — manual rule-compliance check (spec 05 §10).

"Does the audited action comply with the Training Plan rule base?" The rule base (19
Hao-approved rules) lives in `railrl.data.rule_base`. We audit a *chosen action* (the model's
argmax route, or the signaller's logged route) against the rules whose context matches.

EXTENSION OF SPEC §10.2 — the spec sketch only compares `preferred_route_id`. Our rules are
mostly platform-PREFERENCES (a set of acceptable platforms), so `l4_check` handles three kinds:
  * route_choice  (R1): compare audited route_id to the named preferred route.
  * platform_set  (C8/C10/C11/R4/R5 + soft §3): compare the audited route's END PLATFORM to a
    set of acceptable platforms.
  * policy_fact   (C9/C12/R6/R7): context-matchable safety/fact policy with no per-decision
    "preferred action" → reported as 'policy-applies' (informational; never non-compliant).

HARD (confidence=high) rules produce the gating status. SOFT (med) rules are reported
separately and NEVER gate (§12). When two soft rules with disjoint platform sets both match,
status is 'ambiguous' (the Plan itself contradicts, e.g. Sheffield→5 vs →6) — we do NOT pick
one. When the audited route's end platform is unknown (~72% of routes lack end_platform), the
platform_set verdict is 'undetermined' rather than guessed.

Pure-python (no torch / pyarrow) → sandbox-testable. The GPU driver
(`scripts/eval/12_l4_compliance.py`) computes the model's argmax route per decision and feeds
decision samples here.
"""
from __future__ import annotations

from typing import Optional

from ..data import rule_base as RB

# gating-relevant hard verdicts vs reference-only soft verdicts
HARD_STATUSES = ("compliant", "non-compliant", "preferred-unavailable",
                 "policy-applies", "undetermined", "no-rule", "wait")


def _platform_verdict(pref_platforms, audited_route_id, candidate_route_ids):
    """Compare audited route's end platform to an acceptable set (spec §10.2, platform variant)."""
    ep = RB.route_end_platform(audited_route_id) if audited_route_id else None
    if ep is None:
        return "undetermined", None
    if ep in pref_platforms:
        return "compliant", ep
    # preferred available among candidates?
    for rid in (candidate_route_ids or []):
        cep = RB.route_end_platform(rid)
        if cep in pref_platforms:
            return "non-compliant", ep
    return "preferred-unavailable", ep


def _route_verdict(pref_route_id, non_pref, audited_route_id, candidate_route_ids):
    """spec §10.2 route variant, with non-preferred-acceptable note."""
    cands = [str(x) for x in (candidate_route_ids or [])]
    if str(audited_route_id) == str(pref_route_id):
        return "compliant", None
    if str(pref_route_id) in cands:
        # preferred was available but not chosen → non-compliant, unless audited is the
        # sanctioned non-preferred alternative (still flagged, since preferred WAS available)
        return "non-compliant", None
    # preferred unavailable: choosing the sanctioned non-preferred is acceptable
    if str(audited_route_id) in [str(x) for x in (non_pref or [])]:
        return "compliant", "non-preferred-acceptable (preferred route unavailable)"
    return "preferred-unavailable", None


def l4_check(decision_sample: dict, rule_base=None, audited_route_id: Optional[str] = None) -> dict:
    """Compliance verdict for one decision.

    decision_sample needs: focal_signal, focal_train, candidate_route_ids (list[str]),
      and audited_route_id (the route under audit). If audited_route_id is None we read
      decision_sample['audited_route_id']; if still None the action is WAIT.
    Returns dict with hard_status (gating), soft_status (reference), matched ids, and detail.
    """
    rules = RULES_LIST if rule_base is None else (
        rule_base if isinstance(rule_base, list) else rule_base.to_dict("records"))
    if audited_route_id is None:
        audited_route_id = decision_sample.get("audited_route_id")
    cands = [str(x) for x in (decision_sample.get("candidate_route_ids") or [])]

    matched = [r for r in rules if RB.rule_matches(r, decision_sample, audited_route_id)]
    hard = [r for r in matched if r.get("confidence") == "high"]
    soft = [r for r in matched if r.get("confidence") in ("med", "low")]

    # ---- HARD (gating) ----
    if audited_route_id is None:
        hard_status, hard_detail = "wait", None
    elif not hard:
        hard_status, hard_detail = "no-rule", None
    else:
        actionable = [r for r in hard if r["kind"] in ("route_choice", "platform_set")]
        if not actionable:
            hard_status, hard_detail = "policy-applies", [r["rule_id"] for r in hard]
        else:
            # priority: route_choice first, then most-specific platform_set (smallest set)
            actionable.sort(key=lambda r: (r["kind"] != "route_choice",
                                           len(r["pref"].get("preferred_platforms") or [9]*9)))
            r = actionable[0]
            if r["kind"] == "route_choice":
                hard_status, hard_detail = _route_verdict(
                    r["pref"]["preferred_route_id"], r["pref"].get("non_preferred_route_ids"),
                    audited_route_id, cands)
            else:
                hard_status, hard_detail = _platform_verdict(
                    r["pref"]["preferred_platforms"], audited_route_id, cands)

    # ---- SOFT (reference only; never gates) ----
    soft_status, soft_detail = "no-soft-rule", None
    if soft:
        sets = [tuple(r["pref"].get("preferred_platforms") or []) for r in soft]
        # conflicting if two matched soft rules have disjoint non-empty platform sets
        nonempty = [set(s) for s in sets if s]
        conflict = any(a and b and a.isdisjoint(b) for i, a in enumerate(nonempty)
                       for b in nonempty[i + 1:])
        if conflict:
            soft_status, soft_detail = "ambiguous", [r["rule_id"] for r in soft]
        else:
            union = sorted(set().union(*nonempty)) if nonempty else []
            soft_status, soft_detail = _platform_verdict(union, audited_route_id, cands)

    return {
        "hard_status": hard_status,
        "hard_detail": hard_detail,
        "soft_status": soft_status,
        "soft_detail": soft_detail,
        "matched_hard": [r["rule_id"] for r in hard],
        "matched_soft": [r["rule_id"] for r in soft],
        "audited_route_id": audited_route_id,
        "audited_end_platform": RB.route_end_platform(audited_route_id) if audited_route_id else None,
    }


def l4_summary_per_cell(decompositions: list) -> dict:
    """spec §10.3 — L4 hard-status distribution per Tier-3 cell (or stratum).
    decompositions: list of dicts each with 'cell' (str) and 'l4' (l4_check output).
    Returns per-cell counts + fractions, plus the §12 gate (divergent-unsafe ∩ non-compliant).
    """
    cells: dict = {}
    gate_unsafe_noncompliant = 0
    total = 0
    for d in decompositions:
        cell = d.get("cell", "all")
        st = d["l4"]["hard_status"]
        total += 1
        for key in (cell, "overall"):
            c = cells.setdefault(key, {s: 0 for s in HARD_STATUSES})
            c[st if st in c else "no-rule"] += 1
        if "unsafe" in str(cell).lower() and st == "non-compliant":
            gate_unsafe_noncompliant += 1

    out = {"per_cell": {}, "n": total,
           "gate_divergent_unsafe_noncompliant": gate_unsafe_noncompliant,
           "gate_frac": (gate_unsafe_noncompliant / total) if total else 0.0,
           "gate_pass(<1%)": (gate_unsafe_noncompliant / total < 0.01) if total else True}
    for cell, c in cells.items():
        n = sum(c.values())
        out["per_cell"][cell] = {"n": n, "counts": c,
                                 "frac": {k: (v / n if n else 0.0) for k, v in c.items()}}
    return out


# materialise once (list[dict]) for the default rule base
RULES_LIST = list(RB.RULES)
