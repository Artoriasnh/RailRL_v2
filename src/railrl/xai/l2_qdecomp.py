"""L2 — decision-level explanation (spec 05 §8).

Two pieces:
  1. q_gap_decomposition(model, data): decompose the Q-gap between the model's chosen
     action a* and the runner-up a' into 6 feature-GROUP contributions via EXACT Shapley
     values (2^6 = 64 coalitions) using input-ablation.
         Q_gap = Q(s, a*) − Q(s, a')  =  base_value + Σ_g SHAP_g            (completeness)
     An 'absent' group has its inputs set to the designed NEUTRAL baseline — continuous /
     binary → 0 (features are z-scored so 0 = mean) and categorical / identity → index 0
     (the pad/unknown token, per encoders/input_pipeline). base_value = Q_gap with ALL
     groups baselined (the model's residual preference with no informative input).
  2. generate_nl_rationale(...): fill the §8.2 natural-language template (top-3 Q, the
     special-case flags, the Q-gap decomposition, and — if supplied — the L3 counterfactual
     and L4 compliance).

The 6 groups (spec §8.1): train_features, route_features, subgraph_state, sequence_summary,
schedule_outlook, special_flags. a* = argmax masked Q (within candidates); a' = 2nd-highest.

NOTE on the focal node: 'is_focal' lives in the train binary block, so ablating the train
group zeroes is_focal → gather_focal yields h_focal = 0. That IS the intended "train
features absent" semantics (no focal representation), documented as a modelling choice.

The Shapley combinatorics + NL template are PURE PYTHON (sandbox-tested). The model forward
/ HeteroData ablation need torch + PyG → run on Windows GPU (scripts/eval/07_l2_explain.py).
"""
from __future__ import annotations
from itertools import combinations
import math

# 6 feature groups (spec §8.1)
GROUPS = ["train", "route", "state", "sequence", "schedule", "flags"]
GROUP_LABEL = {
    "train": "Train features", "route": "Route features", "state": "Subgraph state",
    "sequence": "Sequence summary", "schedule": "Schedule outlook", "flags": "Special flags",
}
# group -> node-types whose .cont/.binary/.cat/.ident get baselined
_NODE_GROUPS = {"train": ["trn"], "route": ["route"], "state": ["track", "signal"]}
# group -> graph-level tensors that get zeroed
_GRAPH_GROUPS = {"sequence": ["ev_state", "ev_log_dt"],
                 "schedule": ["ol_hc", "ol_eta", "ol_plat"],
                 "flags": ["special_flags"]}


def _baseline_subset(data, present):
    """Clone `data`; every group NOT in `present` → neutral baseline
    (node cont/binary→0, cat/ident→0 pad; graph tensors→0)."""
    import torch
    d = data.clone()
    for g, ntypes in _NODE_GROUPS.items():
        if g in present:
            continue
        for nt in ntypes:
            st = d[nt]
            st.cont = torch.zeros_like(st.cont)
            st.binary = torch.zeros_like(st.binary)
            st.cat = torch.zeros_like(st.cat)
            st.ident = torch.zeros_like(st.ident)
    for g, tensors in _GRAPH_GROUPS.items():
        if g in present:
            continue
        for tn in tensors:
            setattr(d, tn, torch.zeros_like(getattr(d, tn)))
    return d


def shapley_from_values(values):
    """Exact Shapley φ_g for each group from a {frozenset(coalition): v} table.
    φ_g = Σ_{S⊆G\\{g}} |S|!(n-|S|-1)!/n! · [v(S∪g) − v(S)]. Pure python (testable)."""
    n = len(GROUPS)
    shap = {}
    for g in GROUPS:
        rest = [x for x in GROUPS if x != g]
        phi = 0.0
        for r in range(len(rest) + 1):
            w = math.factorial(r) * math.factorial(n - r - 1) / math.factorial(n)
            for S in combinations(rest, r):
                s = frozenset(S)
                phi += w * (values[s | {g}] - values[s])
        shap[g] = phi
    return shap


def q_gap_decomposition(model, data, device="cpu"):
    """Exact Shapley decomposition of Q_gap(a*, a') over the 6 groups. Returns dict."""
    import torch
    model.eval()
    with torch.no_grad():
        q_full = model(data.to(device))["Q"].view(-1)
    order = torch.argsort(q_full, descending=True)
    a_star, a_runner = int(order[0]), int(order[1])

    def qgap(d):
        with torch.no_grad():
            q = model(d.to(device))["Q"].view(-1)
        return float(q[a_star] - q[a_runner])

    # value table over all 64 coalitions
    values = {}
    for r in range(len(GROUPS) + 1):
        for combo in combinations(GROUPS, r):
            present = set(combo)
            values[frozenset(present)] = qgap(_baseline_subset(data, present))

    shap = shapley_from_values(values)
    full = values[frozenset(GROUPS)]
    base = values[frozenset()]
    return {
        "a_star": a_star, "a_runner": a_runner,
        "q_gap": full, "base_value": base,
        "shap": shap, "shap_sum": sum(shap.values()),
        "completeness_resid": full - (base + sum(shap.values())),  # ≈0 by construction
        "q_full": [float(x) for x in q_full.tolist()],
    }


FLAG_NAMES = ["f_advance", "f_call_on", "f_platform_dev", "f_priority_compete",
              "f_late_train", "f_unusual_id", "f_trts_pressed", "f_freight_class"]


def generate_nl_rationale(decomp, meta, l3=None, l4=None):
    """Fill the §8.2 NL template. meta: {focal_train, chosen_route, t, candidate_route_ids,
    flags(dict or list of 8)}. l3 (optional): {chosen:{delay,tp}, runner:{delay,tp}, delta}.
    l4 (optional): compliance string."""
    shap = decomp["shap"]
    lines = []
    fr = meta.get("focal_train", "?"); rt = meta.get("chosen_route", "?"); t = meta.get("t", "?")
    lines.append(f"Decision ({fr}, route {rt}) at {t}:")
    lines.append("")
    # special-case context
    flags = meta.get("flags", {})
    if isinstance(flags, (list, tuple)):
        flags = {FLAG_NAMES[i]: flags[i] for i in range(min(len(FLAG_NAMES), len(flags)))}
    active = [k for k in FLAG_NAMES if float(flags.get(k, 0) or 0) > 0]
    lines.append(f"Special-case context: {', '.join(active) if active else 'none (trivial)'}")
    lines.append("")
    # model deliberation: top-3 Q
    q = decomp["q_full"]
    cands = meta.get("candidate_route_ids") or []

    def act_label(idx):
        if idx == 0:
            return "wait"
        if 1 <= idx <= len(cands):
            return f"route {cands[idx - 1]}"
        return f"action#{idx}"
    top3 = sorted(range(len(q)), key=lambda i: q[i], reverse=True)[:3]
    lines.append("Model deliberation (top-3 Q):")
    for i in top3:
        star = "  ⟵ chosen" if i == decomp["a_star"] else (
            "  (runner-up)" if i == decomp["a_runner"] else "")
        lines.append(f"  - {act_label(i):22s} Q = {q[i]:+.2f}{star}")
    lines.append("")
    # Q-gap decomposition
    lines.append(f"Q-gap decomposition ({act_label(decomp['a_star'])} vs "
                 f"{act_label(decomp['a_runner'])}) = {decomp['q_gap']:+.2f}:")
    lines.append(f"  - base (no informative input): {decomp['base_value']:+.2f}")
    for g in sorted(GROUPS, key=lambda g: -abs(shap[g])):
        lines.append(f"  - {GROUP_LABEL[g]:18s} {shap[g]:+.2f}")
    lines.append("")
    # L4 compliance (optional)
    lines.append(f"Manual compliance (L4): {l4 if l4 is not None else 'N/A (rule base pending)'}")
    # L3 counterfactual (optional)
    if l3 is not None:
        lines.append("")
        lines.append("L3 counterfactual (next 30 min):")
        c, r = l3.get("chosen", {}), l3.get("runner", {})
        lines.append(f"  - chosen   : finish Δ {c.get('delay', float('nan')):+.1f}s, "
                     f"throughput {c.get('tp', '?')}")
        lines.append(f"  - runner-up: finish Δ {r.get('delay', float('nan')):+.1f}s, "
                     f"throughput {r.get('tp', '?')}")
        lines.append(f"  - net advantage of chosen: {l3.get('delta', float('nan')):+.2f}")
    return "\n".join(lines)
