"""L2 — generate per-decision explanations for example decisions (spec 05 §8).

For a handful of EXAMPLE test decisions (sampled across the interesting strata), runs the
exact-Shapley Q-gap decomposition (src/railrl/xai/l2_qdecomp) + the natural-language
rationale, and writes them to outputs/eval/l2_explanations.md (+ .json). These worked
examples are the paper's L2 interpretability artifact.

Two-pass over snapshots_v2 (canonical order; test episodes are late in the file):
  pass 1 (light): pick N example test SET-decisions per target stratum (sample_id + locator).
  pass 2: decode just those rows in full, encode → HeteroData → decompose → NL.

Each decomposition = 64 model forwards (2^6 coalitions); N is small so this is minutes.
Run on Windows GPU:
    python scripts/eval/07_l2_explain.py --seed 42 --n-per-stratum 2
    python scripts/eval/07_l2_explain.py --seed 42 --n-per-stratum 1 --strata platform_dev,call_on  # smoke
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from railrl import config as C
from railrl.xai.l2_qdecomp import q_gap_decomposition, generate_nl_rationale, FLAG_NAMES

STRATUM_NAMES = {0: "late_train", 1: "advance", 2: "call_on", 3: "platform_dev",
                 4: "priority_compete", 5: "unusual_id", 6: "trivial"}
NAME_STRATUM = {v: k for k, v in STRATUM_NAMES.items()}
DEFAULT_STRATA = ["platform_dev", "call_on", "advance", "priority_compete", "late_train", "trivial"]


def load_strata():
    import pyarrow.parquet as pq
    p = C.SNAPSHOTS_DIR / "stratum_labels.parquet"
    t = pq.read_table(str(p), columns=["sample_id", "stratum"])
    return dict(zip(t.column("sample_id").to_pylist(), t.column("stratum").to_pylist()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--n-per-stratum", type=int, default=2)
    ap.add_argument("--strata", default=",".join(DEFAULT_STRATA),
                    help="comma list of stratum names to sample example decisions from")
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
    print(f"L2 explain | ckpt={ckpt.name} | device={device}")

    strata = load_strata()
    want = {NAME_STRATUM[s] for s in args.strata.split(",") if s in NAME_STRATUM}
    need = {s: args.n_per_stratum for s in want}

    pf = pq.ParquetFile(str(C.SNAPSHOTS_V2_PARQUET))
    # ---- pass 1: locate example test SET-decisions per target stratum ----
    picks = []                       # (rg, local_idx, sample_id, stratum)
    for rg in range(pf.num_row_groups):
        if not any(v > 0 for v in need.values()):
            break
        tb = pf.read_row_group(rg, columns=["sample_id", "split", "chosen_action_idx"])
        sid = tb.column("sample_id").to_pylist()
        sp = tb.column("split").to_pylist()
        ch = tb.column("chosen_action_idx").to_pylist()
        for li in range(len(sid)):
            if sp[li] != "test" or (ch[li] or 0) <= 0:
                continue
            st = strata.get(int(sid[li]), 6)
            if st in need and need[st] > 0:
                picks.append((rg, li, int(sid[li]), st))
                need[st] -= 1
                if not any(v > 0 for v in need.values()):
                    break
    print(f"picked {len(picks)} example decisions: "
          + ", ".join(f"{STRATUM_NAMES[st]}" for _, _, _, st in picks))

    # ---- pass 2: decode full rows, decompose, explain ----
    by_rg = {}
    for rg, li, sid, st in picks:
        by_rg.setdefault(rg, []).append((li, sid, st))
    results = []
    for rg in sorted(by_rg):
        tb = pf.read_row_group(rg)              # full row group (all columns)
        rows = tb.to_pylist()
        for li, sid, st in by_rg[rg]:
            row = rows[li]
            data = to_heterodata(encode_snapshot(row, stats))
            decomp = q_gap_decomposition(model, data, device)
            flags = row.get("state_special_flags") or {}
            meta = {"focal_train": row.get("focal_train"),
                    "chosen_route": row.get("chosen_route_id"),
                    "t": str(row.get("t")),
                    "candidate_route_ids": [str(x) for x in (row.get("candidate_route_ids") or [])],
                    "flags": {k: flags.get(k, 0) for k in FLAG_NAMES}}
            nl = generate_nl_rationale(decomp, meta)
            results.append({"sample_id": sid, "stratum": STRATUM_NAMES[st],
                            "decomp": decomp, "nl": nl})
            print(f"\n----- [{STRATUM_NAMES[st]}] sample_id={sid} -----")
            print(nl)
            print(f"  (completeness resid = {decomp['completeness_resid']:+.4f})")

    C.EVAL_DIR.mkdir(parents=True, exist_ok=True)
    md = ["# L2 — per-decision explanations (seed%d)\n" % args.seed]
    for r in results:
        md.append(f"## [{r['stratum']}] sample_id={r['sample_id']}\n\n```\n{r['nl']}\n```\n")
    (C.EVAL_DIR / "l2_explanations.md").write_text("\n".join(md), encoding="utf-8")
    (C.EVAL_DIR / "l2_explanations.json").write_text(
        json.dumps([{k: v for k, v in r.items()} for r in results], indent=2), encoding="utf-8")
    print(f"\n→ wrote {C.EVAL_DIR / 'l2_explanations.md'} (+ .json)  [{len(results)} explanations]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
