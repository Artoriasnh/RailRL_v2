"""L1 — model-level attribution (spec 05 §7).

Answers "which assets did the model attend to for this decision?" via node saliency.

We use INTEGRATED GRADIENTS as the primary, robust attribution (spec §7.2 explicitly
positions IG as the complementary, more faithful signal because HGT attention "can be
misleading"). For a decision, the target is the Q-value of the model's chosen action; we
integrate ∂Q/∂(node continuous + binary features) along a straight path from a zero baseline
to the actual input, giving a per-node saliency:

    IG_f = (x_f − 0) · (1/S) Σ_{s=1..S} ∂Q/∂x_f |_{x = (s/S)·x}
    node saliency = Σ_{f ∈ node} |IG_f|              (sum over that node's cont+binary dims)

Attention rollout (spec §7.1) is NOT extracted: the encoder uses PyG `HGTConv`, which does
not expose per-edge attention weights cleanly (unlike `GATConv`), so attention_rollout()
returns None and L1 saliency is IG-only. This is recorded as a limitation; the panel-diagram
heatmap (spec §7.3) is also deferred because `data/reference/panel_layout.json` (the manual
TC/signal → pixel-coordinate map) does not yet exist.

A faithfulness audit (spec §7.5) checks that the top attributed nodes vary across decisions
rather than degenerating to a fixed global-context set.

torch + PyG required → runs on Windows GPU (driver: scripts/eval/10_l1_saliency.py). The
IG-accumulation arithmetic and the faithfulness distinct-count are pure-python (sandbox-tested).
"""
from __future__ import annotations

from ..encoders.input_pipeline import PYG_NODE_KEY

PYG_TYPES = [PYG_NODE_KEY[nt] for nt in ["track", "signal", "route", "train"]]


def attention_rollout(model, data):
    """NOT available: PyG HGTConv does not cleanly expose per-edge attention weights in
    this build, so attention rollout is not extracted. Returns None (documented limitation;
    IG is the L1 saliency per spec §7.2)."""
    return None


def integrated_gradients(model, data, device="cpu", steps=32, target="argmax"):
    """IG node saliency for one decision. target='argmax' (model's chosen action) or
    'chosen' (signaller's logged action). Returns dict with per-type saliency + top nodes."""
    import torch

    model.eval()
    data = data.to(device)
    with torch.no_grad():
        q_all = model(data)["Q"].view(-1)
    a = int(q_all.argmax()) if target == "argmax" else int(data.chosen_action_idx.view(-1)[0])
    q_target = float(q_all[a])

    # actual cont/binary per node type (the IG endpoints; baseline = 0)
    actual = {nt: (data[nt].cont.detach().clone(), data[nt].binary.detach().clone())
              for nt in PYG_TYPES}
    grad_sum = {nt: [torch.zeros_like(actual[nt][0]), torch.zeros_like(actual[nt][1])]
                for nt in PYG_TYPES}

    for s in range(1, steps + 1):
        alpha = s / steps
        leaves = {}
        for nt in PYG_TYPES:
            c = (actual[nt][0] * alpha).requires_grad_(True)
            b = (actual[nt][1] * alpha).requires_grad_(True)
            data[nt].cont = c
            data[nt].binary = b
            leaves[nt] = (c, b)
        q = model(data)["Q"].view(-1)[a]
        flat = [leaves[nt][0] for nt in PYG_TYPES] + [leaves[nt][1] for nt in PYG_TYPES]
        grads = torch.autograd.grad(q, flat, retain_graph=False, allow_unused=True)
        for i, nt in enumerate(PYG_TYPES):
            gc = grads[i]
            gb = grads[i + len(PYG_TYPES)]
            if gc is not None:
                grad_sum[nt][0] += gc.detach()
            if gb is not None:
                grad_sum[nt][1] += gb.detach()
    # restore actual inputs (we mutated data in-place)
    for nt in PYG_TYPES:
        data[nt].cont = actual[nt][0]
        data[nt].binary = actual[nt][1]

    sal = {}                                   # nt -> (N_nt,) per-node saliency
    for nt in PYG_TYPES:
        ig_c = actual[nt][0] * (grad_sum[nt][0] / steps)      # (N,Fc)
        ig_b = actual[nt][1] * (grad_sum[nt][1] / steps)      # (N,Fb)
        sal[nt] = (ig_c.abs().sum(-1) + ig_b.abs().sum(-1)).cpu().numpy()   # (N,)

    # top nodes across all types (with identity + focal markers where available)
    top = []
    for nt in PYG_TYPES:
        ident = data[nt].ident.view(-1).cpu().numpy() if hasattr(data[nt], "ident") else None
        is_focal = None
        if nt == "trn" and hasattr(data["trn"], "binary"):
            is_focal = (data["trn"].binary[:, 0] > 0.5).cpu().numpy()
        for j in range(len(sal[nt])):
            top.append({"type": nt, "local_idx": j,
                        "ident_vocab_idx": int(ident[j]) if ident is not None else None,
                        "is_focal": bool(is_focal[j]) if is_focal is not None else None,
                        "saliency": float(sal[nt][j])})
    top.sort(key=lambda d: -d["saliency"])
    return {"target_action": a, "q_target": q_target,
            "saliency_by_type": {nt: sal[nt].tolist() for nt in PYG_TYPES},
            "top_nodes": top, "attention_rollout": None}


def top_node_keys(decomp, k=10):
    """Stable identity keys (type + ident vocab idx) of the top-k nodes — for faithfulness."""
    keys = []
    for d in decomp["top_nodes"][:k]:
        keys.append((d["type"], d["ident_vocab_idx"]))
    return keys


def faithfulness_verdict(distinct_count, threshold=50):
    """spec §7.5: top nodes should vary across decisions; degenerate if too few distinct."""
    return {"distinct_top_nodes": int(distinct_count),
            "threshold": threshold,
            "faithful": bool(distinct_count > threshold),
            "note": ("attention varies across decisions (not degenerate)" if distinct_count > threshold
                     else "DEGENERATE: top nodes nearly constant — report as limitation (spec §7.5)")}
