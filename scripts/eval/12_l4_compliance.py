"""Stage 8 — L4 manual rule-compliance over the test split (spec 05 §10).

For every test decision we audit TWO actions against the 19 Hao-approved Training Plan rules
(railrl.data.rule_base / railrl.xai.l4_rules):
  * the MODEL's chosen action   (argmax Q over legal actions)   — the headline L4 audit
  * the SIGNALLER's logged action (chosen_route_id)             — human baseline for contrast

Pipeline:
  pass A (pyarrow, CPU): stream snapshots_v2 test split → per sample_id:
        focal_signal, focal_train, candidate_route_ids, chosen_route_id, chosen_action_idx, stratum.
  pass B (torch, GPU):   forward the checkpoint → per sample_id: model argmax action index.
  join + l4_check both actions → aggregate hard-status distribution per stratum + overall,
        model-vs-signaller compliance contrast, soft-preference adherence (reference only),
        and the §12 gate proxy. Writes outputs/eval/l4_compliance_seed{seed}.json.

Tier-3-cell decomposition (spec §10.3) is available by joining sample_id with the Tier-3
classifier output (scripts/eval/03) if present; otherwise we report per-stratum (the robust,
self-contained cell set). The l4_check/aggregation logic is sandbox-verified; this driver
needs GPU only for the argmax forward (mirrors eval/01).

Run on Windows:
    python scripts/eval/12_l4_compliance.py --seed 42
    python scripts/eval/12_l4_compliance.py --seed 42 --max-batches 20   # smoke
"""
from __future__ import annotations
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from railrl import config as C
from railrl.data import rule_base as RB
from railrl.xai import l4_rules as L4

STRATUM_NAMES = {0: "late_train", 1: "advance", 2: "call_on", 3: "platform_dev",
                 4: "priority_compete", 5: "unusual_id", 6: "trivial", -1: "unlabeled"}


def load_meta(split: str, max_rows: int = 0) -> dict:
    """sample_id → dict(focal_signal, focal_train, cands[list[str]], chosen_route_id,
    chosen_action_idx, stratum). Streamed by row-group (bounded memory)."""
    import pyarrow.parquet as pq
    strata = {}
    sp = C.SNAPSHOTS_DIR / "stratum_labels.parquet"
    if sp.exists():
        t = pq.read_table(str(sp), columns=["sample_id", "stratum"])
        strata = dict(zip(t.column("sample_id").to_pylist(), t.column("stratum").to_pylist()))
    cols = ["sample_id", "split", "focal_signal", "focal_train",
            "candidate_route_ids", "chosen_route_id", "chosen_action_idx"]
    pf = pq.ParquetFile(str(C.SNAPSHOTS_V2_PARQUET))
    meta, seen = {}, 0
    for rg in range(pf.num_row_groups):
        d = pf.read_row_group(rg, columns=cols).to_pydict()
        for i in range(len(d["sample_id"])):
            if d["split"][i] != split:
                continue
            sid = int(d["sample_id"][i])
            cands = [str(x) for x in (d["candidate_route_ids"][i] or [])]
            meta[sid] = dict(
                focal_signal=str(d["focal_signal"][i]) if d["focal_signal"][i] is not None else None,
                focal_train=str(d["focal_train"][i]) if d["focal_train"][i] is not None else None,
                cands=cands,
                chosen_route_id=(str(d["chosen_route_id"][i]) if d["chosen_route_id"][i] is not None else None),
                chosen_action_idx=int(d["chosen_action_idx"][i]),
                stratum=int(strata.get(sid, 6)))
            seen += 1
            if max_rows and seen >= max_rows:
                return meta
    return meta


@torch.no_grad()
def model_argmax_by_sid(model, loader, device, max_batches=None) -> dict:
    """sample_id → model argmax action index (0=wait, 1..K=candidate slot)."""
    out = {}
    for bi, (bs, _bsp, _done) in enumerate(loader):
        bs = bs.to(device)
        q = model(bs)["Q"]
        B = bs.num_graphs
        arg = q.argmax(1).cpu().numpy()
        sid = bs.sample_id.view(B).cpu().numpy()
        for s, a in zip(sid, arg):
            out[int(s)] = int(a)
        if (bi + 1) % 200 == 0:
            print(f"  ...{bi + 1} batches, {len(out):,} decisions")
        if max_batches and bi + 1 >= max_batches:
            break
    return out


def idx_to_route(idx: int, cands: list) -> str | None:
    """action idx (0=wait, 1..K) → route_id (None if wait or out of range)."""
    if idx is None or idx <= 0 or (idx - 1) >= len(cands):
        return None
    return cands[idx - 1]


class CellAcc:
    """Per-cell hard/soft status counters for one audited policy (model or signaller)."""
    def __init__(self):
        self.hard = defaultdict(lambda: defaultdict(int))
        self.soft = defaultdict(lambda: defaultdict(int))

    def add(self, cell, hard_status, soft_status):
        for key in (cell, "overall"):
            self.hard[key][hard_status] += 1
            self.soft[key][soft_status] += 1

    def export(self):
        def pack(counter):
            o = {}
            for cell, c in counter.items():
                n = sum(c.values())
                o[cell] = {"n": n, "counts": dict(c),
                           "frac": {k: v / n for k, v in c.items()}}
            return o
        return {"hard": pack(self.hard), "soft": pack(self.soft)}


def main() -> int:
    ap = argparse.ArgumentParser(description="L4 rule compliance (spec 05 §10).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--run-dir", type=str, default=None)
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--ckpt-name", type=str, default="best.pt")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--batch-size", type=int, default=C.BATCH_SIZE)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--max-batches", type=int, default=None, help="smoke cap")
    ap.add_argument("--max-rows", type=int, default=0, help="smoke cap on meta rows")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--dump-per-decision", action="store_true",
                    help="also write per-decision parquet (large) for Tier-3 join")
    args = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader
    from railrl.encoders.input_pipeline import NormStats
    from railrl.algorithms.transitions import StreamingTransitionDataset
    from railrl.model import RailRLModel

    try:
        import torch.multiprocessing as _mp
        _mp.set_sharing_strategy("file_system")
    except (RuntimeError, AttributeError):
        pass

    run_dir = Path(args.run_dir) if args.run_dir else (C.TRAIN_DIR / f"cql_seed{args.seed}")
    ckpt = Path(args.ckpt) if args.ckpt else (run_dir / args.ckpt_name)
    device = "cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu"
    print(f"device={device} | ckpt={ckpt} | split={args.split}")
    if not ckpt.exists():
        print(f"[error] checkpoint not found: {ckpt}", file=sys.stderr)
        return 1

    print("pass A: reading snapshot metadata (focal_signal / candidates / chosen) ...")
    meta = load_meta(args.split, args.max_rows)
    print(f"  {len(meta):,} {args.split} decisions")

    print("pass B: model forward (argmax route per decision) ...")
    stats = NormStats.load(C.NORMALIZATION_STATS_JSON)
    model = RailRLModel.build(stats).to(device)
    ck = torch.load(str(ckpt), map_location=device, weights_only=False)
    model.load_state_dict(ck["model"] if isinstance(ck, dict) and "model" in ck else ck)
    model.eval()
    ds = StreamingTransitionDataset(C.SNAPSHOTS_V2_PARQUET, C.NORMALIZATION_STATS_JSON,
                                    split=args.split, batch_size=args.batch_size,
                                    shuffle=False, stratified=False)
    loader = DataLoader(ds, batch_size=None, num_workers=args.num_workers)
    model_arg = model_argmax_by_sid(model, loader, device, args.max_batches)
    print(f"  {len(model_arg):,} model decisions")

    print("join + l4_check (model & signaller) ...")
    macc, sacc = CellAcc(), CellAcc()
    agree_compliant = both_n = 0
    per_decision = []
    n_audited = 0
    for sid, mrow in meta.items():
        if sid not in model_arg:
            continue
        cands = mrow["cands"]
        sample = dict(focal_signal=mrow["focal_signal"], focal_train=mrow["focal_train"],
                      candidate_route_ids=cands)
        m_route = idx_to_route(model_arg[sid], cands)
        s_route = mrow["chosen_route_id"] or idx_to_route(mrow["chosen_action_idx"], cands)
        m = L4.l4_check(sample, audited_route_id=m_route)
        s = L4.l4_check(sample, audited_route_id=s_route)
        cell = STRATUM_NAMES.get(mrow["stratum"], "unlabeled")
        macc.add(cell, m["hard_status"], m["soft_status"])
        sacc.add(cell, s["hard_status"], s["soft_status"])
        n_audited += 1
        if m["hard_status"] in ("compliant", "non-compliant") and \
           s["hard_status"] in ("compliant", "non-compliant"):
            both_n += 1
            if m["hard_status"] == "compliant" and s["hard_status"] == "compliant":
                agree_compliant += 1
        if args.dump_per_decision:
            per_decision.append(dict(sample_id=sid, stratum=cell,
                                     model_route=m_route, sig_route=s_route,
                                     model_hard=m["hard_status"], sig_hard=s["hard_status"],
                                     model_soft=m["soft_status"], sig_soft=s["soft_status"]))

    # headline: compliant-rate among decisions where a HARD verdict was rendered (compliant|non-compliant)
    def compliant_rate(acc):
        c = acc.hard["overall"]
        denom = c.get("compliant", 0) + c.get("non-compliant", 0)
        return (c.get("compliant", 0) / denom) if denom else None, denom

    m_rate, m_denom = compliant_rate(macc)
    s_rate, s_denom = compliant_rate(sacc)
    report = {
        "meta": {"seed": args.seed, "ckpt": str(ckpt), "split": args.split,
                 "n_audited": n_audited, "n_rules": len(RB.RULES),
                 "note": "hard rules (high) gate; soft (med) reference-only; soft 'no-soft-rule' "
                         "dominates because destination direction is hidden by leak audit "
                         "(resolve_direction→None) — supply a headcode→direction map to activate."},
        "headline": {
            "model_hard_compliant_rate": m_rate, "model_hard_denom": m_denom,
            "signaller_hard_compliant_rate": s_rate, "signaller_hard_denom": s_denom,
            "both_rendered_n": both_n, "both_compliant_n": agree_compliant,
        },
        "model": macc.export(),
        "signaller": sacc.export(),
    }
    C.EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = C.EVAL_DIR / f"l4_compliance_seed{args.seed}.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\n→ wrote {out_path}")

    if args.dump_per_decision and per_decision:
        import pandas as pd
        pp = C.EVAL_DIR / f"l4_per_decision_seed{args.seed}.parquet"
        pd.DataFrame(per_decision).to_parquet(pp, index=False)
        print(f"→ wrote {pp}  ({len(per_decision):,} rows, for Tier-3 join)")

    # ---- console summary ----
    mh = macc.hard["overall"]
    print(f"\n=== L4 compliance (overall, n={n_audited:,}) ===")
    print(f"MODEL hard-status: {dict(mh)}")
    print(f"  → hard compliant-rate (of {m_denom:,} rendered): "
          f"{100*m_rate:.1f}%" if m_rate is not None else "  → no hard verdicts")
    print(f"SIGNALLER hard-status: {dict(sacc.hard['overall'])}")
    print(f"  → hard compliant-rate (of {s_denom:,} rendered): "
          f"{100*s_rate:.1f}%" if s_rate is not None else "  → no hard verdicts")
    print("(soft §3 traffic-flow rules report 'no-soft-rule' until a headcode→direction map "
          "is supplied — destination is hidden from state by the leak audit. They never gate.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
