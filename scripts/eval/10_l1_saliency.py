"""L1 — node saliency for example decisions + faithfulness audit (spec 05 §7).

For a few example test decisions, computes Integrated-Gradients node saliency (which assets
the model relied on) and reports the top nodes. Then runs the faithfulness audit (spec §7.5):
over N test decisions, checks that the top-attributed nodes vary across decisions rather than
collapsing to a fixed global-context set (degenerate attribution).

Attention rollout and the Derby-panel heatmap are not produced (PyG HGTConv attention is not
cleanly extractable; the panel coordinate map data/reference/panel_layout.json does not exist).

Run on Windows GPU:
    python scripts/eval/10_l1_saliency.py --seed 42 --n-per-stratum 1 --faith-n 50   # smoke
    python scripts/eval/10_l1_saliency.py --seed 42 --n-per-stratum 2 --faith-n 300  # full
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from railrl import config as C
from railrl.xai.l1_attention import integrated_gradients, top_node_keys, faithfulness_verdict

STRATUM_NAMES = {0: "late_train", 1: "advance", 2: "call_on", 3: "platform_dev",
                 4: "priority_compete", 5: "unusual_id", 6: "trivial"}
NAME_STRATUM = {v: k for k, v in STRATUM_NAMES.items()}
DEFAULT_STRATA = ["platform_dev", "call_on", "advance", "priority_compete", "late_train", "trivial"]


def load_strata():
    import pyarrow.parquet as pq
    t = pq.read_table(str(C.SNAPSHOTS_DIR / "stratum_labels.parquet"),
                      columns=["sample_id", "stratum"])
    return dict(zip(t.column("sample_id").to_pylist(), t.column("stratum").to_pylist()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--n-per-stratum", type=int, default=2)
    ap.add_argument("--strata", default=",".join(DEFAULT_STRATA))
    ap.add_argument("--steps", type=int, default=32, help="IG steps for example decisions")
    ap.add_argument("--faith-n", type=int, default=300, help="# decisions for faithfulness audit")
    ap.add_argument("--faith-steps", type=int, default=8, help="IG steps for the audit (cheaper)")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    import torch
    import pyarrow.parquet as pq
    from railrl.encoders.input_pipeline import NormStats, encode_snapshot, to_heterodata
    from railrl.model import RailRLModel

    device = "cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu"
    ckpt = Path(args.ckpt) if args.ckpt else (C.TRAIN_DIR / f"cql_seed{args.seed}" / "best.pt")
    stats = NormStats.load(C.NORMALIZATION_STATS_JSON)
    model = RailRLModel.build(stats).to(device)
    ck = torch.load(str(ckpt), map_location=device, weights_only=False)
    model.load_state_dict(ck["model"] if isinstance(ck, dict) and "model" in ck else ck)
    model.eval()
    print(f"L1 saliency | ckpt={ckpt.name} | device={device}")

    strata = load_strata()
    want = {NAME_STRATUM[s] for s in args.strata.split(",") if s in NAME_STRATUM}
    need = {s: args.n_per_stratum for s in want}

    pf = pq.ParquetFile(str(C.SNAPSHOTS_V2_PARQUET))
    # one streaming pass: collect example rows (per stratum) + first faith_n test set-decisions
    examples = []           # (row, stratum)
    faith_rows = []         # rows for the audit
    for rg in range(pf.num_row_groups):
        if not any(v > 0 for v in need.values()) and len(faith_rows) >= args.faith_n:
            break
        tb = pf.read_row_group(rg)
        rows = tb.to_pylist()
        for row in rows:
            if row.get("split") != "test" or (row.get("chosen_action_idx") or 0) <= 0:
                continue
            st = strata.get(int(row["sample_id"]), 6)
            if st in need and need[st] > 0:
                examples.append((row, st)); need[st] -= 1
            elif len(faith_rows) < args.faith_n:
                faith_rows.append(row)
        if not any(v > 0 for v in need.values()) and len(faith_rows) >= args.faith_n:
            break
    print(f"examples: {len(examples)} | faithfulness sample: {len(faith_rows)}")

    # ---- example decisions: full IG, top nodes ----
    results = []
    for row, st in examples:
        data = to_heterodata(encode_snapshot(row, stats))
        dec = integrated_gradients(model, data, device, steps=args.steps, target="argmax")
        top = dec["top_nodes"][:10]
        results.append({"sample_id": int(row["sample_id"]), "stratum": STRATUM_NAMES[st],
                        "focal_train": row.get("focal_train"), "t": str(row.get("t")),
                        "target_action": dec["target_action"], "q_target": dec["q_target"],
                        "top_nodes": top})
        print(f"\n----- [{STRATUM_NAMES[st]}] sample_id={row['sample_id']} "
              f"(focal {row.get('focal_train')}) -----")
        print(f"  target action idx {dec['target_action']} (Q={dec['q_target']:+.2f}); top-5 attributed nodes:")
        for d in top[:5]:
            foc = " [FOCAL]" if d["is_focal"] else ""
            print(f"    {d['type']:6s} #{d['local_idx']:<3d} ident={d['ident_vocab_idx']} "
                  f"sal={d['saliency']:.4f}{foc}")

    # ---- faithfulness audit ----
    print(f"\nfaithfulness audit over {len(faith_rows)} decisions (IG steps={args.faith_steps}) ...")
    distinct = set(); t0 = time.time()
    for i, row in enumerate(faith_rows):
        data = to_heterodata(encode_snapshot(row, stats))
        dec = integrated_gradients(model, data, device, steps=args.faith_steps, target="argmax")
        distinct.update(top_node_keys(dec, k=10))
        if (i + 1) % 50 == 0:
            print(f"    [faith] {i+1}/{len(faith_rows)} | distinct top-nodes so far {len(distinct)} | "
                  f"{(i+1)/(time.time()-t0):.1f}/s", flush=True)
    verdict = faithfulness_verdict(len(distinct))
    print(f"\n=== L1 faithfulness (spec §7.5) ===")
    print(f"  distinct top-10 nodes over {len(faith_rows)} decisions: {verdict['distinct_top_nodes']} "
          f"(threshold {verdict['threshold']}) → {'PASS' if verdict['faithful'] else 'DEGENERATE'}")
    print(f"  {verdict['note']}")

    C.EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out = {"seed": args.seed, "faithfulness": verdict,
           "attention_rollout": "not extracted (PyG HGTConv); IG-only saliency (spec §7.2)",
           "panel_heatmap": "deferred (data/reference/panel_layout.json missing)",
           "examples": results}
    (C.EVAL_DIR / "l1_saliency.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    md = ["# L1 — node saliency (Integrated Gradients), seed%d\n" % args.seed,
          f"Faithfulness: {verdict['distinct_top_nodes']} distinct top-10 nodes over "
          f"{len(faith_rows)} decisions → {'PASS' if verdict['faithful'] else 'DEGENERATE'}.\n"]
    for r in results:
        md.append(f"## [{r['stratum']}] sample_id={r['sample_id']} (focal {r['focal_train']})\n")
        md.append(f"target action {r['target_action']} (Q={r['q_target']:+.2f}). Top nodes:\n")
        for d in r["top_nodes"][:10]:
            md.append(f"- {d['type']} #{d['local_idx']} ident={d['ident_vocab_idx']} "
                      f"sal={d['saliency']:.4f}{' [FOCAL]' if d['is_focal'] else ''}")
        md.append("")
    (C.EVAL_DIR / "l1_saliency.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\n→ wrote {C.EVAL_DIR / 'l1_saliency.md'} (+ .json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
