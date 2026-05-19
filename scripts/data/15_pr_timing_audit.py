"""P2.B+ - Audit how signallers actually set routes in time:
  (a) Lead time: how long before the train enters approach does PR fire?
  (b) Batch set: how often do multiple PRs fire together (>= 3 within 60s)?
  (c) Pre-set: what fraction of PRs fire when focal_train is nowhere in
      the monitored area for the past N minutes?

Inputs:  decision_events.parquet, td_data.parquet
Outputs: outputs/p2_data_eng/decisions/pr_timing_audit.{json,md}
"""
from __future__ import annotations
import json, sys, time as _time
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from railrl import config as C


def main():
    t0 = _time.time()
    print("[1/4] Loading decision_events + TD ...")
    de = pd.read_parquet(C.DECISION_EVENTS_PARQUET)
    de["t_ns"] = pd.to_datetime(de["time"]).astype("datetime64[ns]").astype("int64")
    print(f"  PRs: {len(de):,}")

    # Stream-load TD to find each train's TC entries
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(C.TD_PARQUET)
    print("  scanning TD for train TC entries ...")
    by_train = defaultdict(list)  # train_id -> sorted list of t_ns
    for batch in pf.iter_batches(batch_size=200_000,
                                   columns=["time","type","state","trainid_filled"]):
        df = batch.to_pandas()
        df = df[(df["type"]=="Track") & (df["state"]==1)
                 & df["trainid_filled"].notna() & (df["trainid_filled"]!="")]
        if df.empty: continue
        ts = pd.to_datetime(df["time"]).astype("datetime64[ns]").astype("int64").to_numpy()
        ts_arr = ts; tids = df["trainid_filled"].astype(str).to_numpy()
        for tid, t in zip(tids, ts_arr):
            by_train[tid].append(int(t))
    for tid in by_train:
        by_train[tid] = np.array(sorted(by_train[tid]), dtype=np.int64)
    print(f"  trains tracked: {len(by_train):,}  ({_time.time()-t0:.1f}s)")

    print("[2/4] Lead time analysis: PR_time - last_TC_event_time ...")
    leads = []  # negative = train was already in network before PR
    no_recent = 0  # train never seen in past 1 hour
    HOUR = int(3600 * 1e9)
    for r in de.itertuples(index=False):
        tid = str(r.train_id) if hasattr(r, 'train_id') else None
        if tid is None or tid not in by_train:
            no_recent += 1; continue
        arr = by_train[tid]
        t_pr = int(r.t_ns)
        # Last TC event STRICTLY BEFORE PR
        j = int(np.searchsorted(arr, t_pr, side="left"))
        if j == 0:
            no_recent += 1; continue
        last_tc = int(arr[j-1])
        delta_s = (t_pr - last_tc) / 1e9
        leads.append(delta_s)

    leads = np.array(leads)
    print(f"  PRs with measurable lead time: {len(leads):,}")
    print(f"  PRs with no recent TC events:  {no_recent:,}")

    print("[3/4] Lead-time bucket distribution ...")
    bins = [(-np.inf, 0,        "PR fires AFTER train passed (latency / unusual)"),
            (0,        60,      "0-60s    (immediate, train in approach)"),
            (60,       600,     "1-10 min (train approaching)"),
            (600,      1800,    "10-30 min (early set, in network)"),
            (1800,     7200,    "30 min - 2h (pre-set, before in-network)"),
            (7200,     np.inf,  "> 2h     (way pre-set or stale data)")]
    bucket_counts = []
    for lo, hi, name in bins:
        n = int(((leads >= lo) & (leads < hi)).sum())
        bucket_counts.append((name, n, round(100*n/max(len(leads),1), 1)))

    print("[4/4] Batch-PR analysis: # PRs fired within 60s of each PR ...")
    de_sorted = de.sort_values("t_ns").reset_index(drop=True)
    t_ns = de_sorted["t_ns"].to_numpy()
    batch_count = []
    WIN = int(60 * 1e9)
    for i in range(len(t_ns)):
        # Count PRs within ±60s
        lo = int(np.searchsorted(t_ns, t_ns[i] - WIN, side="left"))
        hi = int(np.searchsorted(t_ns, t_ns[i] + WIN, side="right"))
        batch_count.append(hi - lo - 1)  # exclude self
    batch_count = np.array(batch_count)
    print(f"  median other-PR-count within 60s: {int(np.median(batch_count))}")
    print(f"  PRs that fire alone (no other in 60s):     {int((batch_count==0).sum()):,} ({100*(batch_count==0).mean():.1f}%)")
    print(f"  PRs fired in burst (>=3 others in 60s):    {int((batch_count>=3).sum()):,} ({100*(batch_count>=3).mean():.1f}%)")
    print(f"  PRs fired in burst (>=10 others in 60s):   {int((batch_count>=10).sum()):,} ({100*(batch_count>=10).mean():.1f}%)")

    summary = {
        "n_PRs":                  int(len(de)),
        "n_with_lead_time":       int(len(leads)),
        "n_no_recent_track_event": int(no_recent),
        "lead_time_seconds": {
            "P10":  float(np.percentile(leads, 10))  if len(leads) else None,
            "P50":  float(np.percentile(leads, 50))  if len(leads) else None,
            "P90":  float(np.percentile(leads, 90))  if len(leads) else None,
            "P99":  float(np.percentile(leads, 99))  if len(leads) else None,
            "max":  float(leads.max())               if len(leads) else None,
        },
        "lead_buckets": [
            {"range": name, "n": n, "pct": pct} for name, n, pct in bucket_counts
        ],
        "batch_PR": {
            "alone":         int((batch_count==0).sum()),
            "small_burst":   int(((batch_count>=1)&(batch_count<3)).sum()),
            "burst_3_to_9":  int(((batch_count>=3)&(batch_count<10)).sum()),
            "burst_10plus":  int((batch_count>=10).sum()),
        },
    }
    out_dir = C.DECISIONS_DIR
    (out_dir / "pr_timing_audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md = ["# PR timing audit", ""]
    md.append(f"PRs total: **{summary['n_PRs']:,}**")
    md.append(f"  - with measurable lead time (train seen recently): {summary['n_with_lead_time']:,}")
    md.append(f"  - no recent track event (likely pre-set very early): {summary['n_no_recent_track_event']:,}")
    md.append("")
    md.append("## Lead time distribution (seconds)")
    md.append(f"  P10/P50/P90/P99 = {summary['lead_time_seconds']['P10']:.0f} / "
              f"{summary['lead_time_seconds']['P50']:.0f} / "
              f"{summary['lead_time_seconds']['P90']:.0f} / "
              f"{summary['lead_time_seconds']['P99']:.0f}")
    md.append("")
    md.append("## Lead time buckets")
    for b in summary["lead_buckets"]:
        md.append(f"  - {b['range']}: {b['n']:,} ({b['pct']}%)")
    md.append("")
    md.append("## Batch PR (other PRs within ±60s of each PR)")
    bp = summary["batch_PR"]
    md.append(f"  - alone (no other in 60s):  {bp['alone']:,}")
    md.append(f"  - small burst (1-2):        {bp['small_burst']:,}")
    md.append(f"  - burst 3-9:                {bp['burst_3_to_9']:,}")
    md.append(f"  - burst 10+:                {bp['burst_10plus']:,}")
    (out_dir / "pr_timing_audit.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nWrote {out_dir/'pr_timing_audit.json'} and .md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
