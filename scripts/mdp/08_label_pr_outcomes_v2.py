"""Stage 4.6.5 [1/3] — Label PR outcomes from decision_points_v2.

Regenerates pr_outcomes against the CURRENT decision_points_v2 (≈2.0M rows),
replacing the stale v1 pr_outcomes.parquet. Uses the proven v1 route-lifecycle
scan (railrl.data.pr_outcomes.label_all_prs) on a reward-format intermediate
that renames t→time / trigger_type→trigger and pins sample_id.

Outputs:
    outputs/decision_points/decision_points_v2_rewardfmt.parquet   (intermediate)
    outputs/rewards/pr_outcomes_v2.parquet
    outputs/rewards/pr_outcomes_v2_summary.json

Run on Windows (sandbox can't read the full event stream):
    python scripts/mdp/08_label_pr_outcomes_v2.py
    python scripts/mdp/08_label_pr_outcomes_v2.py --limit 20000   # smoke
"""
from __future__ import annotations
import argparse
import json
import sys
import time as _time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from railrl import config as C
from railrl.mdp import reward_v2 as RV
from railrl.data.event_stream import AssetIndex, EventTokenStream
from railrl.data.pr_outcomes import (
    build_route_to_tc_idx, label_all_prs, summarize_outcomes,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="Only label first N set decisions (smoke test).")
    ap.add_argument("--force-rewardfmt", action="store_true",
                    help="Rebuild the rewardfmt intermediate even if it exists.")
    args = ap.parse_args()

    print(f"[0/4] Normalising decision_points_v2 -> rewardfmt "
          f"({C.DECISION_POINTS_V2_PARQUET.name})")
    t0 = _time.time()
    dp = RV.build_rewardfmt(force=args.force_rewardfmt)
    n_set = int((dp["label"] == "set").sum())
    n_wait = int((dp["label"] == "wait").sum())
    print(f"      {len(dp):,} decision points  (set={n_set:,}  wait={n_wait:,}), "
          f"sample_id 0..{int(dp['sample_id'].max())}, {_time.time()-t0:.1f}s")
    print(f"      rewardfmt -> {RV.REWARDFMT_PARQUET}")

    print("[1/4] Loading event stream + asset index...")
    t0 = _time.time()
    es = EventTokenStream.load()
    ai = AssetIndex.load()
    es._build_per_asset_index()
    print(f"      {es.n_tokens:,} tokens, per-asset index ready, {_time.time()-t0:.1f}s")

    print("[2/4] Building route->TCs index from edges_traverses...")
    t0 = _time.time()
    r2tc = build_route_to_tc_idx(ai, C.EDGE_TRAVERSES_PARQUET)
    sizes = [v.size for v in r2tc.values()]
    mean_tcs = sum(sizes) / max(len(sizes), 1) if sizes else 0
    print(f"      {len(r2tc)} routes, mean {mean_tcs:.1f} TCs/route, {_time.time()-t0:.1f}s")

    print("[3/4] Classifying PR outcomes (set rows)...")
    t0 = _time.time()
    out = label_all_prs(RV.REWARDFMT_PARQUET, ai, es, r2tc, limit=args.limit)
    print(f"      done, {_time.time()-t0:.1f}s")

    print("[4/4] Writing outputs...")
    out_path = (RV.PR_OUTCOMES_V2 if not args.limit
                else RV.PR_OUTCOMES_V2.with_name(f"pr_outcomes_v2_smoke_{args.limit}.parquet"))
    out.to_parquet(out_path, index=False, compression="zstd")
    summary = summarize_outcomes(out)
    if not args.limit:
        RV.PR_OUTCOMES_V2_SUMMARY.write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"      pr_outcomes_v2 -> {out_path}")

    print("\n=== Outcome distribution ===")
    for k, v in summary["outcome_counts"].items():
        pct = summary["outcome_percent"][k]
        print(f"  {k:<20s}  {v:>9,}  ({pct:5.2f}%)")
    print(f"  TOTAL: {summary['total_prs']:,}")
    print(f"  mean r_thru = {summary['r_thru_distribution']['mean']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
