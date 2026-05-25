"""Stage 4.2 — Build normalization stats + categorical vocabularies.

Scans snapshots_v2.parquet ONCE (streaming over row groups, memory-safe) and
produces outputs/snapshots/normalization_stats.json with:

  continuous[feature] = {mean, std, n}          # z-score, clip [-5,5] at load
  vocab[field]        = {value: index, ...}      # learned-embedding indices
  meta                = {caps, dims, ...}

Per spec 03 §2.2 + §3.1:
  - Continuous (fractions, ages, counts, Derby_info physical) → z-score.
  - Binary flags → left as 0/1.
  - Categorical (prefix, headcode_class, platform_sub, ...) → embedding index.
  - Identity (track_id, signal_id, route_id, train_id) → embedding index
    (vocab sized per spec: track~250, signal~124, route~278, train~2200).

IMPORTANT (spec 03 §2.2): stats use the TRAIN SPLIT ONLY (no val/test leak).
Split is TIME-BASED (spec 04 §4.1) via pass_split.parquet — each episode is
assigned by its start time, so train stats never see future (val/test) data.
Run scripts/train/00_build_time_split.py first.

Usage:
    python scripts/train/01_build_normalization_stats.py
    python scripts/train/01_build_normalization_stats.py --val-frac 0.15 --test-frac 0.15
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from railrl import config as C
from railrl.encoders.input_pipeline import load_pass_split, load_episode_split

# ----- Feature taxonomy (from schema.py node structs) -----
# continuous features per node type (z-scored)
CONT = {
    "track": ["n_routes_using",
              *[f"occupancy_fraction_{w}m" for w in (1, 5, 10, 15, 30)],
              *[f"n_state_changes_{w}m" for w in (1, 5, 10, 15, 30)],
              "last_change_age_s"],
    "signal": ["n_routes_from",
               *[f"aspect_fraction_red_{w}m" for w in (1, 5, 10, 15, 30)],
               *[f"aspect_n_changes_{w}m" for w in (1, 5, 10, 15, 30)],
               "aspect_last_change_age_s", "berth_dwell_age_s"],
    "route": ["n_tc", "length_m", "ave_speed_mps", "ave_grad", "gap_time_s",
              "n_points", "last_locked_age_s", "n_tcs_occupied_by_other",
              "n_tcs_occupied_by_focal", "max_relative_position_of_occupied",
              "min_age_of_occupation_s"],
    "train": ["time_in_current_berth_s", "scheduled_delta_s",
              "recent_panel_requests_count"],
}
# categorical features → vocab (small cardinality, learned 8-d embedding)
CAT = {
    "track": ["platform_sub"],
    "signal": ["prefix", "platform_direction"],
    "route": ["prefix", "signal_no", "letter", "sub", "cls"],
    "train": ["headcode_class"],
}
# identity fields → per-type asset embedding vocab
IDENT = {"track": "track_id", "signal": "signal_id",
         "route": "route_id", "train": "train_id"}
NODE_COL = {"track": "state_nodes_track", "signal": "state_nodes_signal",
            "route": "state_nodes_route", "train": "state_nodes_train"}


def _split_of(pass_id: str, val_frac: float, test_frac: float) -> str:
    h = int(hashlib.md5(str(pass_id).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    if h < test_frac:
        return "test"
    if h < test_frac + val_frac:
        return "val"
    return "train"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    args = ap.parse_args()

    path = C.SNAPSHOTS_V2_PARQUET
    pf = pq.ParquetFile(path)
    print(f"scanning {path}  ({pf.metadata.num_rows:,} rows, {pf.num_row_groups} row groups)")

    # Split source priority (Stage 4.7.2d):
    #   1. episodes_v2.parquet (sample_id → split) — AUTHORITATIVE after the
    #      episode re-segmentation fix (in-file pass_id is stale).
    #   2. pass_split.parquet (pass_id → split) — legacy time-based.
    #   3. md5(pass_id) hash — legacy, NOT spec-compliant.
    # split_key = the row column used to look up the split.
    episode_split = load_episode_split()
    if episode_split:
        print(f"using episodes_v2.parquet split (sample_id → split, {len(episode_split):,} rows) "
              "[post episode re-segmentation 4.7.2d]")
        split_method = "time-based via episodes_v2.parquet (sample_id → split, post-resegment 4.7.2d)"
        split_key = "sample_id"
        def _split(k): return episode_split.get(int(k), "train")
    else:
        pass_split = load_pass_split()
        if pass_split:
            print(f"[warn] episodes_v2.parquet MISSING — using pass_split.parquet "
                  f"(pass_id → split, {len(pass_split):,} episodes). If you ran "
                  "14_resegment_episodes.py, check the sidecar path.")
            split_method = "time-based via pass_split.parquet (pass_id → split)"
            split_key = "pass_id"
            def _split(p): return pass_split.get(str(p), "train")
        else:
            print("[warn] no split sidecar — falling back to md5(pass_id) hash "
                  "split (NOT spec 04 §4.1). Run 14_resegment_episodes.py first!")
            split_method = "md5(pass_id) hash (LEGACY — not time-based)"
            split_key = "pass_id"
            def _split(p): return _split_of(p, args.val_frac, args.test_frac)

    # streaming accumulators (TRAIN split only for continuous stats)
    cnt = defaultdict(int)        # (ntype, feat) -> n
    s1 = defaultdict(float)       # sum
    s2 = defaultdict(float)       # sum of squares
    vocab = defaultdict(set)      # (ntype, field) -> set of values
    n_train = n_val = n_test = 0

    cols = list(NODE_COL.values()) + [split_key]
    for rg in range(pf.num_row_groups):
        df = pf.read_row_group(rg, columns=cols).to_pandas()
        splits = [_split(k) for k in df[split_key]]
        for ntype, col in NODE_COL.items():
            cont_feats = CONT[ntype]
            cat_feats = CAT.get(ntype, [])
            ident = IDENT[ntype]
            for row_nodes, split in zip(df[col], splits):
                for node in row_nodes:
                    # identity + categorical vocab from ALL splits (so unseen
                    # val/test ids still map; embedding is not a leak)
                    iv = node.get(ident)
                    if iv is not None:
                        vocab[(ntype, ident)].add(str(iv))
                    for cf in cat_feats:
                        v = node.get(cf)
                        if v is not None and str(v) != "nan":
                            vocab[(ntype, cf)].add(str(v))
                    # continuous stats from TRAIN only
                    if split == "train":
                        for f in cont_feats:
                            v = node.get(f)
                            if v is None:
                                continue
                            fv = float(v)
                            if not np.isfinite(fv):
                                continue
                            key = (ntype, f)
                            cnt[key] += 1
                            s1[key] += fv
                            s2[key] += fv * fv
        for sp in splits:
            if sp == "train": n_train += 1
            elif sp == "val": n_val += 1
            else: n_test += 1
        if rg % 50 == 0:
            print(f"  rg {rg}/{pf.num_row_groups}", flush=True)

    # finalize
    continuous = {}
    for (ntype, f), n in cnt.items():
        mean = s1[(ntype, f)] / n
        var = max(0.0, s2[(ntype, f)] / n - mean * mean)
        std = float(np.sqrt(var)) or 1.0
        continuous[f"{ntype}.{f}"] = {"mean": round(mean, 6), "std": round(std, 6), "n": n}

    vocabs = {}
    for (ntype, field), values in vocab.items():
        # index 0 reserved for padding/unknown
        idx = {v: i + 1 for i, v in enumerate(sorted(values))}
        vocabs[f"{ntype}.{field}"] = {"size": len(idx) + 1, "index": idx}

    stats = {
        "split": {"train": n_train, "val": n_val, "test": n_test,
                  "method": split_method},
        "continuous": continuous,
        "vocab": vocabs,
        "caps": {"track": C.MAX_TRACKS_PADDED, "signal": C.MAX_SIGNALS_PADDED,
                 "route": C.MAX_ROUTES_PADDED, "train": C.MAX_TRAINS_PADDED,
                 "candidates": C.MAX_CANDIDATES_PADDED, "event_tokens": C.EVENT_TOKEN_K},
    }
    out = C.NORMALIZATION_STATS_JSON
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(stats, f, indent=2)

    print()
    print("=" * 70)
    print(f"split: train={n_train:,} val={n_val:,} test={n_test:,}")
    print(f"continuous features: {len(continuous)}")
    print(f"vocabs: " + ", ".join(f"{k}={v['size']}" for k, v in vocabs.items()
                                    if ".track_id" in k or ".signal_id" in k
                                    or ".route_id" in k or ".train_id" in k))
    print(f"[write] {out}")
    print("=" * 70)


if __name__ == "__main__":
    main()
