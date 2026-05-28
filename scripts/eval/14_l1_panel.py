"""L1 panel-heatmap driver (spec 05 §7.3, panel-figure surrogate).

Renders the existing IG node saliency (`outputs/eval/l1_saliency.json`) onto a self-drawn
Derby schematic — no manual `panel_layout.json` needed (per Hao 2026-05-28: abstract the
infrastructure adjacencies into our own diagram instead of overlaying on the real panel image).

Outputs (all saved to `outputs/figures/l1_panel/` for reuse):
  * `decision_<sample_id>_<stratum>.png` — one figure per example decision (12 total).
  * `decision_<sample_id>_<stratum>.json` — the rendering data (positions/saliency_by_name/
    route_tcs/top route-and-train saliency) for later reuse without re-rendering.
  * `aggregate_panel.png` — single panel with mean saliency across all decisions.
  * `aggregate_panel.json` — the mean saliency / coverage by TC.
  * `adjacency_matrix.png` — TC×TC adjacency heatmap (top-80 by saliency).
  * `derby_layout.json` — the hand-coded operational anchor positions (reusable across runs).
  * `tc_adjacency.json` — TC→adjacent-TC map derived from route track_lists (reusable).
  * `INDEX.md` — catalog with sample_id / stratum / focal_train / focal_route / top-5 nodes.

Pure-python (matplotlib only) — no torch/pyarrow needed; runs everywhere `01_evaluate_model`'s
sibling does.  Re-run by: `python scripts/eval/14_l1_panel.py`.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from railrl import config as C
from railrl.xai.panel_schematic import (
    load_vocabs, build_tc_adjacency, DERBY_LAYOUT, render_adjacency_matrix,
)
from railrl.xai.panel_schematic_plotly import (
    render_decision_plotly, render_aggregate_plotly,
)

OUT_DIR = C.OUTPUTS_DIR / "figures" / "l1_panel"


def chosen_route_id_of(example: dict, vocabs: dict) -> str | None:
    """Best-effort: pull the focal/chosen route_id from the example's top_nodes (the route
    flagged is_focal=True is the model's argmax route)."""
    # In l1_saliency.json the route node with is_focal=True is the target action's route.
    # If none flagged, fall back to the highest-saliency route node.
    routes_focal = [n for n in example.get("top_nodes", [])
                    if n.get("type") == "route" and n.get("is_focal")]
    chosen = routes_focal[0] if routes_focal else None
    if chosen is None:
        routes_sorted = sorted(
            [n for n in example.get("top_nodes", []) if n.get("type") == "route"],
            key=lambda d: -d.get("saliency", 0))
        chosen = routes_sorted[0] if routes_sorted else None
    if chosen is None:
        return None
    return vocabs.get("route", {}).get(int(chosen.get("ident_vocab_idx", -1)))


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"loading saliency from {C.EVAL_DIR / 'l1_saliency.json'}")
    sal = json.loads((C.EVAL_DIR / "l1_saliency.json").read_text())
    print(f"  n examples: {len(sal.get('examples', []))} | seed: {sal.get('seed')}")

    vocabs = load_vocabs()
    print(f"  vocab sizes: " + " | ".join(f"{nt}:{len(v)}" for nt, v in vocabs.items()))
    adj = build_tc_adjacency()
    print(f"  TC adjacency: {len(adj)} nodes, "
          f"{sum(len(v) for v in adj.values())//2} edges")

    # save reusable data artefacts up-front
    (OUT_DIR / "derby_layout.json").write_text(json.dumps(DERBY_LAYOUT, indent=2))
    (OUT_DIR / "tc_adjacency.json").write_text(json.dumps(adj, indent=2))

    # ---- per-decision figures ----
    per_decision = []
    index_rows = []
    for ex in sal.get("examples", []):
        sid = ex.get("sample_id"); st = ex.get("stratum") or "unknown"
        focal_train = ex.get("focal_train")
        focal_route_id = chosen_route_id_of(ex, vocabs)
        data = render_decision_plotly(
            top_nodes=ex.get("top_nodes", []), vocabs=vocabs, adj=adj,
            focal_route_id=focal_route_id, candidate_route_ids=None,
            sample_id=sid, stratum=st, output_dir=OUT_DIR, top_k=15)
        data["focal_train"] = focal_train
        (OUT_DIR / f"decision_{sid}_{st}.json").write_text(json.dumps(data, indent=2))
        per_decision.append(data)
        top5 = sorted(data["sal_by_name"].items(), key=lambda kv: -kv[1])[:5]
        top5s = ", ".join(f"{n}={s:.2f}" for n, s in top5)
        index_rows.append({"sample_id": sid, "stratum": st, "focal_train": focal_train,
                           "focal_route": focal_route_id, "top5_spatial": top5s,
                           "html": Path(data["html_path"]).name,
                           "png": Path(data["png_path"]).name if data.get("png_path") else None})
        print(f"  [{st:18s}] sid={sid:7d} → {Path(data['html_path']).name}  | top5: {top5s}")

    # ---- aggregate panel (plotly, HTML + PNG) ----
    agg = render_aggregate_plotly(per_decision, OUT_DIR)
    (OUT_DIR / "aggregate_panel.json").write_text(json.dumps(
        {"n_decisions": agg["n_decisions"],
         "mean_sal": agg["mean_sal"], "coverage": agg["coverage"]}, indent=2))
    print(f"  aggregate → {Path(agg['html_path']).name}"
          + (f" (+ {Path(agg['png_path']).name})" if agg.get("png_path") else ""))

    # ---- adjacency matrix (matplotlib, top-80 by aggregate saliency among TCs) ----
    adj_png = OUT_DIR / "adjacency_matrix.png"
    render_adjacency_matrix(adj, adj_png, saliency_per_tc=agg["mean_sal"], top_k=80)
    print(f"  adjacency  → {adj_png.name}")
    agg_png_name = Path(agg['png_path']).name if agg.get("png_path") else "aggregate_panel.html"

    # ---- INDEX.md ----
    lines = ["# L1 panel-schematic figures — index",
             "",
             f"Generated from `outputs/eval/l1_saliency.json` (seed {sal.get('seed')}, "
             f"{len(per_decision)} example decisions across 7 strata).",
             "",
             "**Layout convention** (self-drawn schematic, replaces manual `panel_layout.json`): "
             "platforms 1-6 as horizontal lanes (top to bottom), north (Duffield) ← x → east (Spondon). "
             "TC adjacency derived from route track-lists. Hand-coded anchor positions for ~60 "
             "operational TCs/signals (platforms, L4-rule junction signals, branches); other TCs "
             "placed via BFS-radial fallback near their adjacent anchors.",
             "",
             "## Per-decision figures",
             "",
             "| sample_id | stratum | focal_train | focal_route | top-5 spatial nodes | figure |",
             "|---|---|---|---|---|---|"]
    for r in index_rows:
        fig_cell = f"[html](./{r['html']})"
        if r.get("png"):
            fig_cell += f" · [png](./{r['png']})"
        lines.append(f"| {r['sample_id']} | {r['stratum']} | {r['focal_train']} | "
                     f"`{r['focal_route']}` | {r['top5_spatial']} | {fig_cell} |")
    lines += ["",
              "## Aggregate panel",
              "",
              "Interactive: [aggregate_panel.html](./aggregate_panel.html) · "
              f"Static: ![aggregate]({agg_png_name})",
              "",
              "Mean IG saliency across all decisions, on the same schematic. Larger/redder = more "
              "consistently salient across decisions. The companion adjacency matrix:",
              "",
              f"![adjacency matrix]({adj_png.name})",
              "",
              "## Reusable data files",
              "",
              "- `derby_layout.json` — hand-coded anchor positions (TC/signal → [x,y]).",
              "- `tc_adjacency.json` — TC → list of adjacent TCs (from route track-lists).",
              "- `decision_<sid>_<stratum>.json` — per-decision render data (positions, "
              "saliency_by_name, route_tcs, top route/train saliency).",
              "- `aggregate_panel.json` — mean saliency + coverage per node across decisions.",
              ""]
    (OUT_DIR / "INDEX.md").write_text("\n".join(lines))
    print(f"\n→ wrote {OUT_DIR / 'INDEX.md'}")
    print(f"  total files in {OUT_DIR}: "
          f"{len(per_decision)*2 + 5} (figures + json + layout/adj/INDEX)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
