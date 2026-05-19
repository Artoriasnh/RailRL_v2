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
import sys
import time
from pathlib import Path

import pandas as pd

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


def _row_to_snapshot(row: pd.Series) -> tuple[dict, dict]:
    """Re-hydrate snapshot + sample_meta dicts from a parquet row."""
    # The build script flattens snapshot dict → row. Reconstruct the bits
    # the audit needs.
    state_keys = [c for c in row.index if c.startswith("state_") or c == "center"]
    snap = {k: row[k] for k in state_keys if pd.notna(row.get(k, None)) or
             isinstance(row.get(k), (list, dict))}
    # Lists/dicts may have come back as numpy arrays — leave as-is, the audit
    # functions handle both.
    sample_meta = {
        "focal_train":            row.get("focal_train"),
        "focal_train_current_tc": (snap.get("state_center", {}) or {}).get("id"),
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

    print(f"[1/3] loading {inp}")
    df = pd.read_parquet(inp)
    print(f"      {len(df):,} snapshots")

    if args.sample is not None and args.sample < len(df):
        df = df.sample(n=args.sample, random_state=args.seed)
        print(f"      sampled {len(df):,}")

    print(f"[2/3] running {len(df):,} audits...")
    n_pass = 0
    n_fail = 0
    violations = []
    log_every = max(1000, len(df) // 50)

    for i, (_, row) in enumerate(df.iterrows()):
        snap, meta = _row_to_snapshot(row)
        t_ns = int(pd.Timestamp(row["t"]).value) if "t" in row.index else 0

        if args.first_fail:
            try:
                assert_no_leak(snap, meta, t_ns)
                n_pass += 1
            except LeakAuditError as e:
                n_fail += 1
                print(f"\n[FAIL] sample_id={row.get('sample_id')} {e}\n")
                violations.append({
                    "sample_id":   int(row.get("sample_id", -1)),
                    "focal_train": row.get("focal_train"),
                    "t":           str(row.get("t")),
                    "violations":  [str(e)],
                })
                break
        else:
            row_v = collect_violations(snap, meta, t_ns)
            if row_v:
                n_fail += 1
                violations.append({
                    "sample_id":   int(row.get("sample_id", -1)),
                    "focal_train": row.get("focal_train"),
                    "t":           str(row.get("t")),
                    "violations":  row_v,
                })
            else:
                n_pass += 1

        if (i + 1) % log_every == 0:
            print(f"  ... {i+1:,}/{len(df):,}  "
                  f"pass={n_pass:,}  fail={n_fail:,}")

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
