"""Stage 3 — Parallel driver for 05_build_snapshots.py (multi-core).

Launches N independent OS processes, each running
    python scripts/mdp/05_build_snapshots.py --shard K --nshards N
on a strided 1/N slice of the decision points, then merges the N part files
into the final snapshots_v2.parquet (streaming, memory-safe).

Each worker is a fully independent process (no shared Python objects / no
pickling), so this is robust on Windows. Each worker loads TD + Movements and
builds its own histories — so peak RAM ≈ N × (TD + histories). Tune --workers
to your machine (4-8 is typical on a workstation).

Usage:
    python scripts/mdp/05b_build_snapshots_parallel.py --workers 6
    python scripts/mdp/05b_build_snapshots_parallel.py --workers 4 --limit 40000  # smoke
    python scripts/mdp/05b_build_snapshots_parallel.py --workers 6 --keep-parts

Output: outputs/snapshots/snapshots_v2.parquet (+ merged summary + skipped)
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from railrl import config as C

THIS_DIR = Path(__file__).resolve().parent
BUILD_SCRIPT = THIS_DIR / "05_build_snapshots.py"


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=6, help="Number of parallel shards/processes")
    p.add_argument("--limit", type=int, default=None, help="Pass-through: cap total decisions (smoke)")
    p.add_argument("--audit-every", type=int, default=1000, help="Pass-through to workers")
    p.add_argument("--batch-size", type=int, default=5000, help="Pass-through to workers")
    p.add_argument("--keep-parts", action="store_true", help="Keep per-shard part files after merge")
    return p.parse_args()


def main():
    args = _parse_args()
    n = max(1, args.workers)
    out_path = C.SNAPSHOTS_V2_PARQUET
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stem = out_path.with_suffix("")

    t0 = time.time()
    print("=" * 70)
    print(f"Parallel snapshot build — {n} workers")
    print("=" * 70)

    # 1. Launch N workers. `-u` = unbuffered stdout so the per-shard progress
    #    lines flush to the log files in real time (otherwise block-buffering
    #    makes the logs look empty and the run look "stuck").
    procs = []
    for k in range(n):
        cmd = [sys.executable, "-u", str(BUILD_SCRIPT),
               "--shard", str(k), "--nshards", str(n),
               "--audit-every", str(args.audit_every),
               "--batch-size", str(args.batch_size)]
        if args.limit is not None:
            cmd += ["--limit", str(args.limit)]
        log_path = out_path.parent / f"build_shard{k}.log"
        print(f"  launching shard {k}/{n} → log {log_path.name}")
        lf = open(log_path, "w")
        procs.append((k, subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT), lf))

    # 2. Wait
    print(f"\n  waiting for {n} workers (tail the build_shard*.log files for progress)...")
    failed = []
    for k, p, lf in procs:
        rc = p.wait()
        lf.close()
        status = "OK" if rc == 0 else f"FAILED rc={rc}"
        print(f"  shard {k}: {status}")
        if rc != 0:
            failed.append(k)
    if failed:
        print(f"\n[ERROR] shards {failed} failed — see build_shard*.log. NOT merging.")
        sys.exit(1)

    # 3. Merge part parquets (streaming → constant memory)
    import pyarrow.parquet as pq
    part_files = [Path(f"{stem}.part{k}.parquet") for k in range(n)]
    missing = [str(p) for p in part_files if not p.exists()]
    if missing:
        print(f"[ERROR] missing part files: {missing}")
        sys.exit(1)

    print(f"\n[merge] merging {n} parts → {out_path}")
    writer = None
    total_rows = 0
    try:
        for pf_path in part_files:
            # `with` ensures the OS file handle is released BEFORE we try to
            # unlink the part file in cleanup (Windows refuses to delete an
            # open file → WinError 32).
            with pq.ParquetFile(pf_path) as pf:
                if writer is None:
                    writer = pq.ParquetWriter(out_path, pf.schema_arrow, compression="zstd")
                for rg in range(pf.num_row_groups):
                    tbl = pf.read_row_group(rg)
                    writer.write_table(tbl)
                    total_rows += tbl.num_rows
    finally:
        if writer is not None:
            writer.close()
    print(f"        merged {total_rows:,} snapshots")

    # 4. Merge per-shard summaries + skipped logs
    agg = {"n_snapshots_built": 0, "n_skipped_no_tc": 0, "n_audit_failures": 0, "n_audited": 0}
    for k in range(n):
        sp = out_path.parent / f"snapshots_v2_summary.part{k}.json"
        if sp.exists():
            d = json.loads(sp.read_text())
            for key in agg:
                agg[key] += int(d.get(key, 0))
    merged_skip = out_path.parent / "skipped_no_tc.jsonl"
    with open(merged_skip, "w") as out:
        for k in range(n):
            skp = out_path.parent / f"skipped_no_tc.part{k}.jsonl"
            if skp.exists():
                out.write(skp.read_text())

    summary = {
        **agg,
        "n_total_rows_merged": int(total_rows),
        "n_workers":           n,
        "elapsed_seconds":     round(time.time() - t0, 1),
    }
    (Path(C.SNAPSHOTS_V2_SUMMARY)).write_text(json.dumps(summary, indent=2))

    # 5. Cleanup parts (best-effort — the merged output is already written, so
    #    a lingering Windows file handle must NOT fail the whole run).
    if not args.keep_parts:
        import gc
        gc.collect()  # release any pyarrow file handles before unlinking
        leftover = []
        for k in range(n):
            paths = [Path(f"{stem}.part{k}.parquet"),
                     out_path.parent / f"snapshots_v2_summary.part{k}.json",
                     out_path.parent / f"skipped_no_tc.part{k}.jsonl"]
            for fp in paths:
                if fp.exists():
                    try:
                        fp.unlink()
                    except OSError:
                        leftover.append(fp.name)
        if leftover:
            print(f"  [warn] could not delete {len(leftover)} part file(s) "
                  f"(safe to remove manually): {leftover[:6]}")

    print()
    print("=" * 70)
    for key, v in summary.items():
        print(f"  {key}: {v}")
    print("=" * 70)
    print(f"DONE → {out_path}  ({time.time()-t0:.1f}s, {n} workers)")


if __name__ == "__main__":
    main()
