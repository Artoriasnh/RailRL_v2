"""P2.4 Iter B - Label PR outcomes via route lifecycle scan."""
from __future__ import annotations
import json
import sys
import time as _time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from railrl import config as C
from railrl.p2_data_eng.event_stream import AssetIndex, EventTokenStream
from railrl.p2_data_eng.pr_outcomes  import (
    build_route_to_tc_idx, label_all_prs, summarize_outcomes,
)


def main():
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
    mean_tcs = sum(sizes)/max(len(sizes),1) if sizes else 0
    max_tcs = max(sizes) if sizes else 0
    print(f"      {len(r2tc)} routes, mean {mean_tcs:.1f} TCs/route, max {max_tcs}, {_time.time()-t0:.1f}s")

    print("[3/4] Classifying PR outcomes...")
    t0 = _time.time()
    out = label_all_prs(C.DECISION_POINTS_PARQUET, ai, es, r2tc)
    print(f"      done, {_time.time()-t0:.1f}s")

    print("[4/4] Writing outputs...")
    out.to_parquet(C.PR_OUTCOMES_PARQUET, index=False, compression="zstd")
    summary = summarize_outcomes(out)
    C.PR_OUTCOMES_SUMMARY.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"      pr_outcomes.parquet -> {C.PR_OUTCOMES_PARQUET}")
    print(f"      pr_outcomes_summary -> {C.PR_OUTCOMES_SUMMARY}")

    print("")
    print("=== Outcome distribution ===")
    for k, v in summary["outcome_counts"].items():
        pct = summary["outcome_percent"][k]
        print(f"  {k:<20s}  {v:>8,}  ({pct:5.2f}%)")
    print(f"  TOTAL: {summary['total_prs']:,}")
    print(f"  mean r_thru = {summary['r_thru_distribution']['mean']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
