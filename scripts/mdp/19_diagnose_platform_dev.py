"""Diagnose why f_platform_dev fires on ~51% of decisions (spec §4.4 expected ~1.5%).

f_platform_dev = (planned_platform is set) AND (no candidate route's end_platform_id
== planned_platform). It ALSO fires when every candidate end_platform_id is None
(generator empty → any()=False → True). So the question is: are the fires REAL
platform deviations, or just unmapped end_platform_id?

Recomputes the cause per snapshot from what the flag actually saw (stored in the
row): focal train's planned_platform + candidate routes' end_platform_id
(in_candidate_set). Streams row groups (memory-bounded).

Categories per row:
  no_planned          planned_platform None            → flag False
  degenerate_allNone  planned set, ALL cand end_plat None → flag True  (← bug-like)
  match               planned set, a candidate ends at planned → flag False
  genuine_dev         planned set, candidates have platforms, none == planned → True (← real)

Run on Windows:
    python scripts/mdp/19_diagnose_platform_dev.py
"""
from __future__ import annotations
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from railrl import config as C


def main():
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(C.SNAPSHOTS_V2_PARQUET)
    cols = ["state_special_flags", "state_nodes_train", "state_nodes_route"]
    print(f"scanning {C.SNAPSHOTS_V2_PARQUET}  ({pf.metadata.num_rows:,} rows)")

    cat = Counter()
    n = 0
    mism = 0                       # recompute vs stored f_platform_dev
    fired_stored = 0
    n_planned_set = 0
    n_route_total = 0
    n_route_endplat = 0            # route nodes with end_platform_id != None
    # among rows with planned set AND ≥1 candidate platform: did any match?
    n_planned_with_candplat = 0
    n_match = 0
    planned_vals = Counter()
    endplat_vals = Counter()

    for bi in range(pf.num_row_groups):
        b = pf.read_row_group(bi, columns=cols)
        flags = b.column("state_special_flags").to_pylist()
        trains = b.column("state_nodes_train").to_pylist()
        routes = b.column("state_nodes_route").to_pylist()
        for fl, trs, rts in zip(flags, trains, routes):
            n += 1
            stored = bool(fl.get("f_platform_dev"))
            fired_stored += int(stored)

            planned = None
            for tr in trs:
                if tr.get("is_focal"):
                    planned = tr.get("planned_platform")
                    break
            if planned is not None:
                n_planned_set += 1
                planned_vals[int(planned)] += 1

            cand = []
            for r in rts:
                n_route_total += 1
                ep = r.get("end_platform_id")
                if ep is not None:
                    n_route_endplat += 1
                    endplat_vals[int(ep)] += 1
                if r.get("in_candidate_set"):
                    cand.append(ep)
            cand_nonnull = [p for p in cand if p is not None]

            if planned is None:
                recomp = False; cat["no_planned"] += 1
            elif len(cand_nonnull) == 0:
                recomp = True; cat["degenerate_allNone"] += 1
            else:
                n_planned_with_candplat += 1
                if any(p == planned for p in cand_nonnull):
                    recomp = False; cat["match"] += 1; n_match += 1
                else:
                    recomp = True; cat["genuine_dev"] += 1
            if recomp != stored:
                mism += 1
        if bi % 50 == 0:
            print(f"  rg {bi}/{pf.num_row_groups}", flush=True)

    fired = cat["degenerate_allNone"] + cat["genuine_dev"]
    print("\n" + "=" * 64)
    print(f"rows: {n:,}")
    print(f"recompute vs stored f_platform_dev MISMATCH: {mism:,} "
          f"({100*mism/max(n,1):.3f}%)  (should be ~0 — confirms recompute is faithful)")
    print(f"\nf_platform_dev fired (stored): {fired_stored:,} ({100*fired_stored/n:.1f}%)")
    print(f"f_platform_dev fired (recomp): {fired:,} ({100*fired/n:.1f}%)")
    print("\n--- cause breakdown ---")
    for k in ("no_planned", "match", "degenerate_allNone", "genuine_dev"):
        print(f"  {k:<20s} {cat[k]:>10,}  ({100*cat[k]/max(n,1):5.1f}%)")
    print(f"\n  of FIRES: degenerate(all-None)={cat['degenerate_allNone']:,} "
          f"({100*cat['degenerate_allNone']/max(fired,1):.1f}%)  "
          f"genuine={cat['genuine_dev']:,} ({100*cat['genuine_dev']/max(fired,1):.1f}%)")
    print(f"\nplanned_platform set: {n_planned_set:,} ({100*n_planned_set/n:.1f}%)")
    print(f"route nodes with end_platform_id: {n_route_endplat:,}/{n_route_total:,} "
          f"({100*n_route_endplat/max(n_route_total,1):.1f}%)")
    print(f"rows w/ planned AND ≥1 candidate platform: {n_planned_with_candplat:,}; "
          f"of those matched planned: {n_match:,} "
          f"({100*n_match/max(n_planned_with_candplat,1):.1f}%)")
    print(f"\nplanned_platform value dist: {dict(planned_vals.most_common())}")
    print(f"route end_platform_id value dist: {dict(endplat_vals.most_common())}")
    print("=" * 64)
    print("\nVERDICT GUIDE:")
    print("  - degenerate(all-None) dominates fires  → end_platform_id unmapped = BUG (over-fires)")
    print("  - genuine dominates + match rate sane    → real platform deviations")
    print("  - match rate ~0% despite candidates having platforms → numbering mismatch (BUG)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
