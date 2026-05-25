"""Stage 4.7.2b' — recalibrate time_bucket edges to the data (train-split τ quintiles).

The spec edges [5,15,30,60]s put 94% of τ in bucket 4 (signallers set routes well
in advance), making the L_time head degenerate. We instead set the 4 edges to the
p20/p40/p60/p80 percentiles of τ computed on the TRAIN split ONLY (no val/test
leak — same discipline as reward thresholds + normalization), giving 5 ≈balanced
buckets. Edges are then applied to ALL splits.

Reuses the saved tau_s in time_labels_v2.parquet — no event-stream re-scan.

Outputs (overwrites buckets, keeps tau_s):
    outputs/rewards/time_labels_v2.parquet      (sample_id, tau_s, time_bucket)
    outputs/rewards/time_bucket_edges.json      (calibrated edges + provenance)

Run on Windows:
    python scripts/mdp/12_recalibrate_time_buckets.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from railrl import config as C
from railrl.mdp import reward_v2 as RV
from railrl.encoders.input_pipeline import load_pass_split

EDGES_JSON = C.REWARDS_DIR / "time_bucket_edges.json"
PERCENTILES = [20, 40, 60, 80]   # → 5 buckets


def main():
    print("[1/4] Loading time_labels_v2 (tau_s) + split mapping ...")
    tl = pd.read_parquet(RV.TIME_LABELS_V2)
    rf = RV.build_rewardfmt()[["sample_id", "pass_id"]]
    ps = load_pass_split()
    if not ps:
        raise RuntimeError("pass_split.parquet missing — run 00_build_time_split.py")
    rf["split"] = rf["pass_id"].map(lambda p: ps.get(str(p), "train"))
    m = tl.merge(rf[["sample_id", "split"]], on="sample_id", how="left")
    assert len(m) == len(tl), "sample_id join changed row count"

    print("[2/4] Calibrating edges on TRAIN-split τ only (no leak) ...")
    train_tau = m.loc[(m["split"] == "train") & m["tau_s"].notna(), "tau_s"].to_numpy()
    n_train = train_tau.size
    if n_train == 0:
        raise RuntimeError("no labelled train τ — cannot calibrate")
    edges = np.round(np.percentile(train_tau, PERCENTILES)).astype(np.int64)
    # guard against duplicate edges (if τ is very concentrated) → make strictly increasing
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1
    print(f"      train labelled τ: {n_train:,} | p{PERCENTILES} edges = {edges.tolist()} s")

    print("[3/4] Reassigning buckets (all splits) via searchsorted ...")
    tau = tl["tau_s"].to_numpy(np.float64)
    bucket = np.full(len(tau), -1, dtype=np.int64)
    fin = np.isfinite(tau)
    # bucket(τ): 0 if τ<=edges[0], 1 if <=edges[1], ..., 4 if >edges[3]
    bucket[fin] = np.searchsorted(edges, tau[fin], side="left").astype(np.int64)
    tl["time_bucket"] = bucket

    print("[4/4] Writing recalibrated labels + edges ...")
    tl.to_parquet(RV.TIME_LABELS_V2, index=False, compression="zstd")

    # distribution overall + per-split
    lab = bucket[bucket >= 0]
    bc = np.bincount(lab, minlength=5)
    per_split = {}
    for sp in ("train", "val", "test"):
        sids = m.loc[m["split"] == sp, "sample_id"]
        sub = tl.set_index("sample_id").loc[sids, "time_bucket"]
        sub = sub[sub >= 0]
        per_split[sp] = np.bincount(sub.to_numpy(), minlength=5).tolist()

    meta = {
        "edges_seconds": edges.tolist(),
        "percentiles": PERCENTILES,
        "calibrated_on": "train split τ only",
        "n_train_labelled": int(n_train),
        "bucket_definition": (f"0: τ≤{edges[0]}s | 1: ≤{edges[1]}s | 2: ≤{edges[2]}s "
                               f"| 3: ≤{edges[3]}s | 4: >{edges[3]}s"),
        "distribution_all": bc.tolist(),
        "distribution_by_split": per_split,
        "note": ("spec 03 §7.2 fixed [5,15,30,60] left 94% in bucket 4; recalibrated "
                 "to train-τ quintiles (user decision 2026-05-22)."),
    }
    EDGES_JSON.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"      -> {RV.TIME_LABELS_V2}")
    print(f"      -> {EDGES_JSON}")

    print(f"\n=== recalibrated time_bucket distribution ({len(lab):,} labelled) ===")
    print(f"  edges (s): {edges.tolist()}   →  {meta['bucket_definition']}")
    for b in range(5):
        c = int(bc[b])
        print(f"  bucket {b}: {c:>8,}  ({100*c/max(len(lab),1):.1f}%)")
    print(f"  per-split (train/val/test): {per_split}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
