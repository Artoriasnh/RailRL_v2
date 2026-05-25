"""Stage 4.7.2b — compute the τ / time_bucket label for the L_time aux head.

τ = first-occupy(route's first TC) − t_PR  (spec 03 §7.2), per SET decision.
bucket = heads.time_bucket(τ) ∈ {0..4}; wait rows / unmeasurable τ → -1 (excluded
from L_time). Output is a sidecar keyed by sample_id (the trainer joins it onto
each snapshot via data.sample_id):

    outputs/rewards/time_labels_v2.parquet  (sample_id, tau_s, time_bucket)

Run on Windows (needs the event stream):
    python scripts/mdp/11_compute_time_labels_v2.py
"""
from __future__ import annotations
import sys
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from railrl import config as C
from railrl.mdp import reward_v2 as RV
from railrl.data.event_stream import AssetIndex, EventTokenStream
from railrl.data.reward_features import build_route_first_tc
from railrl.policies.heads import TIME_BUCKET_EDGES


def main():
    print("[1/4] Loading rewardfmt (sample_id-aligned) ...")
    rf = RV.build_rewardfmt()
    set_mask = (rf["label"] == "set").to_numpy()
    set_dp = rf[set_mask].copy()
    print(f"      {len(rf):,} decisions ({int(set_mask.sum()):,} set)")

    print("[2/4] Loading event stream + asset index + route→first-TC ...")
    t0 = _time.time()
    es = EventTokenStream.load()
    ai = AssetIndex.load()
    es._build_per_asset_index()
    route_first_tc = build_route_first_tc(ai, C.EDGE_TRAVERSES_PARQUET)
    print(f"      done, {_time.time()-t0:.1f}s")

    print("[3/4] Computing τ = first-occupy(route first TC) − t_PR ...")
    t0 = _time.time()
    tau, buckets = RV.compute_lead_time_buckets(set_dp, route_first_tc, es, ai)
    n_tau = int(np.isfinite(tau).sum())
    print(f"      τ measurable: {n_tau:,}/{len(set_dp):,} set decisions "
          f"({100*n_tau/max(len(set_dp),1):.1f}%), {_time.time()-t0:.1f}s")

    # full sidecar over all sample_ids (default bucket -1 = excluded)
    full_tau = np.full(len(rf), np.nan, dtype=np.float64)
    full_bucket = np.full(len(rf), -1, dtype=np.int64)
    set_idx = np.flatnonzero(set_mask)
    full_tau[set_idx] = tau
    full_bucket[set_idx] = buckets
    out = pd.DataFrame({
        "sample_id": rf["sample_id"].to_numpy(np.int64),
        "tau_s": full_tau,
        "time_bucket": full_bucket,
    })

    print("[4/4] Writing time_labels_v2.parquet ...")
    out.to_parquet(RV.TIME_LABELS_V2, index=False, compression="zstd")
    print(f"      -> {RV.TIME_LABELS_V2}")

    # bucket distribution (labelled set rows only)
    lab = full_bucket[full_bucket >= 0]
    edges = TIME_BUCKET_EDGES  # [5,15,30,60]
    names = [f"≤{edges[0]}s", f"≤{edges[1]}s", f"≤{edges[2]}s", f"≤{edges[3]}s", f">{edges[3]}s"]
    print("\n=== time_bucket distribution (labelled set decisions) ===")
    bc = np.bincount(lab, minlength=5)
    for b in range(5):
        c = int(bc[b])
        print(f"  bucket {b} ({names[b]:>6s}): {c:>8,}  ({100*c/max(len(lab),1):.1f}%)")
    print(f"  labelled: {len(lab):,} | excluded (-1): {int((full_bucket<0).sum()):,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
