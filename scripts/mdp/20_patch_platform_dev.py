"""Stage 4.7.2d fix — surgical patch of state_special_flags.f_platform_dev.

f_platform_dev over-fired on 83% of decisions (99.2% were the degenerate
all-None-candidate-platforms case; see scripts/mdp/19 diagnostic). The fix lives
in special_flags.f_platform_dev (returns False when no candidate end platform is
known). This script recomputes the flag for every row USING THAT FIXED FUNCTION,
from the snapshot's own stored nodes (focal train planned_platform + candidate
routes' end_platform_id) — no external data — and rewrites only the
state_special_flags struct column, preserving row order + schema.

Only this one binary flag changes → NO normalization re-run needed (flags aren't
z-scored). DO re-run 16_build_stratum_labels (stratum depends on it) + 10 smoke [D].

Run on Windows:
    python scripts/mdp/20_patch_platform_dev.py
then: rename snapshots_v2.platdev.parquet → snapshots_v2.parquet (keep backup),
      python scripts/mdp/16_build_stratum_labels.py
      python scripts/train/10_smoke_streaming.py
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from railrl import config as C
from railrl.mdp.special_flags import f_platform_dev


def main():
    import pyarrow as pa
    import pyarrow.parquet as pq

    src = C.SNAPSHOTS_V2_PARQUET
    out = src.with_name("snapshots_v2.platdev.parquet")
    pf = pq.ParquetFile(src)
    schema = pf.schema_arrow
    flags_field = schema.field("state_special_flags")
    flags_type = flags_field.type
    fi = schema.names.index("state_special_flags")
    print(f"scanning {src}  ({pf.metadata.num_rows:,} rows, {pf.num_row_groups} row groups)")

    writer = pq.ParquetWriter(out, schema, compression="zstd")
    n = changed = fired_before = fired_after = 0
    try:
        for bi in range(pf.num_row_groups):
            tbl = pf.read_row_group(bi)
            flags = tbl.column("state_special_flags").to_pylist()
            trains = tbl.column("state_nodes_train").to_pylist()
            routes = tbl.column("state_nodes_route").to_pylist()
            new_flags = []
            for fl, trs, rts in zip(flags, trains, routes):
                old = bool(fl["f_platform_dev"]); fired_before += int(old)
                planned = None
                for tr in trs:
                    if tr.get("is_focal"):
                        planned = tr.get("planned_platform")
                        break
                cand = [r.get("end_platform_id") for r in rts if r.get("in_candidate_set")]
                new = bool(f_platform_dev(cand, planned))
                fired_after += int(new)
                if new != old:
                    changed += 1
                d = dict(fl); d["f_platform_dev"] = new
                new_flags.append(d)
                n += 1
            new_arr = pa.array(new_flags, type=flags_type)
            writer.write_table(tbl.set_column(fi, flags_field, new_arr))
            if bi % 50 == 0:
                print(f"  rg {bi}/{pf.num_row_groups}", flush=True)
    finally:
        writer.close()

    print("\n" + "=" * 60)
    print(f"rows: {n:,}")
    print(f"f_platform_dev BEFORE: {fired_before:,} ({100*fired_before/n:.1f}%)")
    print(f"f_platform_dev AFTER : {fired_after:,} ({100*fired_after/n:.1f}%)  "
          f"(expect ~0.7%)")
    print(f"changed: {changed:,} ({100*changed/n:.1f}%)")
    print(f"wrote -> {out}")
    print("=" * 60)
    print("Next: rename .platdev → snapshots_v2.parquet (keep backup), "
          "then re-run 16_build_stratum_labels + 10_smoke_streaming.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
