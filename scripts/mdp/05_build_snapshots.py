"""Stage 3 / Round 3 — Build snapshots_v2.parquet from decision_points + TD + Movements.

Usage:
  python scripts/mdp/05_build_snapshots.py            # full corpus
  python scripts/mdp/05_build_snapshots.py --limit 100   # smoke
  python scripts/mdp/05_build_snapshots.py --dev          # +verbose progress

Inputs:
  - outputs/decision_points/decision_points_v2.parquet  (Stage 2 output)
  - outputs/cache/td_data.parquet                       (TD event log)
  - outputs/cache/movements.parquet                     (Movements gbtt — optional)
  - outputs/static_graph/*.parquet                      (loaded by StaticGraphView/StaticNodeTables)
  - outputs/passes/pass_assignments.parquet             (optional — TRUST id matching)

Output:
  - outputs/snapshots/snapshots_v2.parquet
  - outputs/snapshots/snapshots_v2_summary.json
  - outputs/snapshots/skipped_no_tc.jsonl                (decisions where focal_train.current_tc
                                                            could not be located)

Per spec 02 §11 Q4: if a decision_point's focal_train has no recent TD trace
to anchor current_tc, we skip it and log to skipped_no_tc.jsonl. These are
typically TD parse failures from upstream.

This script:
  1. Loads decision_points + TD + Movements (optional)
  2. Optionally joins pass_assignments (TRUST id matched pass_id)
  3. Constructs SnapshotBuilder with all five histories
  4. Loops over each decision, calls build_snapshot, collects results
  5. Builds episode metadata via episode.build_episodes (per-pass split)
  6. Writes snapshots_v2.parquet
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd

# Allow running as script
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from railrl import config as C
from railrl.mdp.state import SnapshotBuilder
from railrl.mdp.episode import build_episodes
from railrl.mdp.leak_audit import LeakAuditError


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None,
                    help="Build only first N decision points (smoke)")
    p.add_argument("--dev", action="store_true",
                    help="Verbose progress and run leak audit on EVERY snapshot")
    p.add_argument("--audit-every", type=int, default=1000,
                    help="In non-dev mode, run leak audit every N snapshots "
                         "(otherwise audit_passed=None for un-audited rows)")
    p.add_argument("--out", type=str, default=None,
                    help="Override output parquet path")
    return p.parse_args()


def _load_pass_assignments() -> Optional[pd.DataFrame]:
    p = C.PASS_ASSIGNMENTS_PARQUET
    if not Path(p).exists():
        print(f"[warn] {p} not found — falling back to gap-based pass_id")
        return None
    return pd.read_parquet(p)


def main():
    args = _parse_args()
    out_path = Path(args.out) if args.out else C.SNAPSHOTS_V2_PARQUET
    out_path.parent.mkdir(parents=True, exist_ok=True)
    skipped_path = out_path.parent / "skipped_no_tc.jsonl"
    summary_path = C.SNAPSHOTS_V2_SUMMARY

    t0 = time.time()
    print("=" * 70)
    print("Stage 3 R3 — snapshots_v2 builder")
    print("=" * 70)

    # 1. Inputs
    print(f"[1/6] loading decision_points... ({C.DECISION_POINTS_V2_PARQUET})")
    dp = pd.read_parquet(C.DECISION_POINTS_V2_PARQUET)
    print(f"      {len(dp):,} decision points")
    if args.limit is not None:
        dp = dp.head(args.limit)
        print(f"      limited to first {len(dp):,}")

    print(f"[2/6] loading TD events... ({C.TD_PARQUET})")
    td = pd.read_parquet(C.TD_PARQUET)
    print(f"      {len(td):,} TD events")

    print(f"[3/6] loading Movements... ({C.MOVEMENTS_PARQUET})")
    if Path(C.MOVEMENTS_PARQUET).exists():
        mv = pd.read_parquet(C.MOVEMENTS_PARQUET)
        print(f"      {len(mv):,} movement rows")
    else:
        print("      [warn] not found — schedule_outlook will be empty")
        mv = None

    # 2. Episode metadata
    print(f"[4/6] building episode metadata...")
    pass_df = _load_pass_assignments()
    dp = build_episodes(dp, pass_assignments=pass_df)
    print(f"      {dp['episode_idx'].nunique():,} episodes / "
          f"{dp['pass_id'].nunique():,} passes")

    # 3. Snapshot builder
    print(f"[5/6] constructing SnapshotBuilder (loads histories)...")
    sb = SnapshotBuilder.build_default(td, movements=mv)
    # In production we audit selectively; in dev mode audit always.
    if not args.dev:
        sb.run_leak_audit = False  # will audit every N below

    # 4. Loop
    print(f"[6/6] building {len(dp):,} snapshots...")
    rows = []
    skipped = []
    audit_failures = []
    n_audited = 0
    n_built = 0
    t_loop_start = time.time()
    log_every = max(1000, len(dp) // 50)

    for sample_id, (_, dec) in enumerate(dp.iterrows()):
        try:
            run_audit = args.dev or (sample_id % args.audit_every == 0)
            sb.run_leak_audit = run_audit
            decision_dict = dec.to_dict()
            snap = sb.build_snapshot(decision_dict, sample_id=sample_id)
        except LeakAuditError as e:
            audit_failures.append({"sample_id": sample_id, "error": str(e)})
            if args.dev:
                print(f"[LEAK] sample_id={sample_id} {e}")
            continue
        if snap is None:
            skipped.append({
                "sample_id": sample_id,
                "focal_train": dec.get("focal_train"),
                "t": str(dec.get("t")),
                "reason": "no_current_tc",
            })
            continue
        if run_audit:
            n_audited += 1
        n_built += 1
        rows.append(snap)
        if n_built % log_every == 0:
            elapsed = time.time() - t_loop_start
            rate = n_built / max(elapsed, 0.001)
            eta = (len(dp) - sample_id) / max(rate, 0.001)
            print(f"  ... {n_built:,}/{len(dp):,} built  "
                  f"skip={len(skipped):,}  audit_fail={len(audit_failures):,}  "
                  f"{rate:.1f}/s  ETA {eta/60:.1f}min")

    # 5. Output
    print(f"\n[write] {out_path}")
    df_out = pd.DataFrame(rows)
    df_out.to_parquet(out_path, index=False)
    print(f"        {len(df_out):,} snapshots written")

    if skipped:
        with open(skipped_path, "w") as f:
            for s in skipped:
                f.write(json.dumps(s) + "\n")
        print(f"[write] {skipped_path}  ({len(skipped):,} skipped)")

    # 6. Summary
    summary = {
        "n_decision_points":   int(len(dp)),
        "n_snapshots_built":   int(n_built),
        "n_skipped_no_tc":     int(len(skipped)),
        "n_audit_failures":    int(len(audit_failures)),
        "n_audited":           int(n_audited),
        "audit_every":         int(args.audit_every),
        "elapsed_seconds":     round(time.time() - t0, 1),
        "rate_per_second":     round(n_built / max(time.time() - t_loop_start, 0.001), 1),
        "dev_mode":            bool(args.dev),
        "audit_failure_samples": audit_failures[:20],  # top 20 for inspection
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[write] {summary_path}")

    print()
    print("=" * 70)
    print(f"DONE: {n_built:,} snapshots / "
          f"{len(skipped):,} skipped / "
          f"{len(audit_failures):,} audit fails  "
          f"({(time.time() - t0):.1f}s total)")
    print("=" * 70)


if __name__ == "__main__":
    main()
