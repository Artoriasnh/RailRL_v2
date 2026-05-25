"""Stage 3 / Round 3 — Full-corpus leak audit on snapshots_v2.parquet.

Usage:
  python scripts/mdp/06_run_leak_audit_full.py             # audit ALL rows
  python scripts/mdp/06_run_leak_audit_full.py --sample 10000  # audit random N
  python scripts/mdp/06_run_leak_audit_full.py --first-fail   # stop at first

Outputs:
  - outputs/snapshots/leak_audit_report.json    (summary + violations)
  - outputs/snapshots/leak_violations.jsonl     (per-row violations)

Per spec 02 §7 — runs all 7 leak audit checks on each snapshot. The build
script (05) audits every N=1000th row by default; this script gives the
"final say" on the full corpus before training.
"""
from __future__ import annotations
import argparse
import json
import math
import sys
import time
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from railrl import config as C
from railrl.mdp.leak_audit import assert_no_leak, collect_violations, LeakAuditError


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, default=None,
                    help="Override input parquet (default: snapshots_v2.parquet)")
    p.add_argument("--sample", type=int, default=None,
                    help="Audit only N random rows (default: all)")
    p.add_argument("--first-fail", action="store_true",
                    help="Stop at first violation (raise instead of collect)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _row_to_snapshot(row: dict) -> tuple[dict, dict]:
    """Re-hydrate snapshot + sample_meta dicts from a parquet row (to_pylist dict).

    Row comes from pyarrow .to_pylist() → nested values are Python lists/dicts and
    nulls are None, so we just keep non-None state_*/center keys. (The old version
    used pd.notna() on pandas Series cells, which raised "truth value of an array is
    ambiguous" when a nested cell decoded to a numpy array.)
    """
    snap = {k: v for k, v in row.items()
            if (k.startswith("state_") or k == "center") and v is not None}
    # leak_audit Check 1 reads snapshot["center"], but the parquet stores it as
    # "state_center" (bare "center" is dropped at write time / not in the Arrow
    # schema). Alias so Check 1 sees the real center {type:'track', id:...}.
    if "center" not in snap and snap.get("state_center") is not None:
        snap["center"] = snap["state_center"]
    sample_meta = {
        "focal_train":            row.get("focal_train"),
        "focal_train_current_tc": (snap.get("state_center") or {}).get("id"),
        "focal_signal":           row.get("focal_signal"),
    }
    return snap, sample_meta


def main():
    args = _parse_args()
    inp = Path(args.input) if args.input else C.SNAPSHOTS_V2_PARQUET
    out_report = inp.parent / "leak_audit_report.json"
    out_violations = inp.parent / "leak_violations.jsonl"

    t0 = time.time()
    print("=" * 70)
    print("Stage 3 R3 — full-corpus leak audit")
    print("=" * 70)

    # STREAM row-group by row-group (memory-bounded). The old `pd.read_parquet(inp)`
    # loaded the WHOLE 573MB file (nested state_* cols decode to ~15-20GB) → OOM.
    pf = pq.ParquetFile(str(inp))
    nrg = pf.num_row_groups
    total_rows = pf.metadata.num_rows
    per_group = None
    if args.sample is not None and args.sample < total_rows:
        per_group = max(1, math.ceil(args.sample / nrg))   # spread sample across groups
    print(f"[1/3] streaming {inp.name} ({total_rows:,} rows, {nrg} row groups)"
          + (f"; sampling ~{per_group}/group ≈ {args.sample:,}" if per_group else "; ALL rows"))

    n_pass = 0
    n_fail = 0
    violations = []          # capped (keep first 1000 to avoid OOM)
    VIO_CAP = 1000
    seen = 0
    stop = False
    import random as _random
    print(f"[2/3] running audits...")
    for rg in range(nrg):
        rows = pf.read_row_group(rg).to_pylist()      # one group only (~5k rows) → dicts
        if per_group is not None and per_group < len(rows):
            rows = _random.Random(args.seed + rg).sample(rows, per_group)
        for row in rows:
            snap, meta = _row_to_snapshot(row)
            t_ns = int(pd.Timestamp(row["t"]).value) if row.get("t") is not None else 0
            if args.first_fail:
                try:
                    assert_no_leak(snap, meta, t_ns)
                    n_pass += 1
                except LeakAuditError as e:
                    n_fail += 1
                    print(f"\n[FAIL] sample_id={row.get('sample_id')} {e}\n")
                    violations.append({"sample_id": int(row.get("sample_id", -1)),
                                       "focal_train": row.get("focal_train"),
                                       "t": str(row.get("t")), "violations": [str(e)]})
                    stop = True
                    break
            else:
                row_v = collect_violations(snap, meta, t_ns)
                if row_v:
                    n_fail += 1
                    if len(violations) < VIO_CAP:
                        violations.append({"sample_id": int(row.get("sample_id", -1)),
                                           "focal_train": row.get("focal_train"),
                                           "t": str(row.get("t")), "violations": row_v})
                else:
                    n_pass += 1
            seen += 1
        del rows
        if stop:
            break
        if rg % 25 == 0:
            print(f"  ... rg {rg}/{nrg}  audited={seen:,}  pass={n_pass:,}  fail={n_fail:,}",
                  flush=True)

    # Write violations
    if violations:
        with open(out_violations, "w") as f:
            for v in violations:
                f.write(json.dumps(v) + "\n")
        print(f"\n[write] {out_violations}  ({len(violations):,} rows)")

    # Summary
    pct_pass = 100.0 * n_pass / max(1, n_pass + n_fail)
    report = {
        "input":              str(inp),
        "n_total_audited":    int(n_pass + n_fail),
        "n_passed":           int(n_pass),
        "n_failed":           int(n_fail),
        "pct_passed":         round(pct_pass, 4),
        "first_fail_mode":    bool(args.first_fail),
        "elapsed_seconds":    round(time.time() - t0, 1),
        "first_20_failures":  violations[:20],
    }
    with open(out_report, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[write] {out_report}")

    print()
    print("=" * 70)
    if n_fail == 0:
        print(f"PASS — all {n_pass:,} snapshots cleared leak audit "
              f"({time.time() - t0:.1f}s)")
    else:
        print(f"FAIL — {n_fail:,}/{n_pass + n_fail:,} snapshots failed "
              f"({pct_pass:.2f}% pass rate)")
    print("=" * 70)
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
