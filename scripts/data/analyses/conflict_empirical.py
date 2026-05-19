#!/usr/bin/env python3
"""
Empirical analysis of "conflict" — does a Panel_Request ever fire on a route
whose track sections are currently occupied?

Run:
    python scripts/analyses/conflict_empirical.py            # full TD file
    python scripts/analyses/conflict_empirical.py --nrows 5000000

Outputs:
    outputs/p2_data_eng/analyses/conflict_summary.json
    outputs/p2_data_eng/analyses/conflict_per_pr.parquet
    outputs/p2_data_eng/analyses/conflict_other_occupations.parquet

Result interpreted:  if "% of PRs with ≥1 TC occupied by another train" is >0.5%,
the binary hard-conflict mask is wrong — drop it and use granular features instead.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import pandas as pd

from railrl import config as C
from railrl.data_io import load_route_to_tc


def main(nrows: int | None) -> None:
    out_dir = C.OUTPUTS_DIR / "p2_data_eng" / "analyses"
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"Loading TD CSV (nrows={nrows}) ...", flush=True)
    df = pd.read_csv(
        C.TD_CSV,
        usecols=["time", "type", "id", "state", "trainid_filled"],
        parse_dates=["time"], nrows=nrows, low_memory=False,
    )
    print(f"  loaded {len(df):,} rows in {time.time()-t0:.1f}s")

    tracks = df[df["type"] == "Track"][
        ["time", "id", "state", "trainid_filled"]
    ].rename(columns={"id": "tc_id", "trainid_filled": "occupier",
                      "time": "tc_event_time"})
    prs = df[df["type"] == "Panel_Request"][
        ["time", "id", "trainid_filled"]
    ].rename(columns={"id": "route_id", "trainid_filled": "focal_train"}).copy()

    # Dedupe routes
    routes = load_route_to_tc()
    routes = routes[routes["route"].notna()].copy()
    routes["n_tc"] = routes["track_list"].apply(len)
    routes = (routes.sort_values("n_tc", ascending=False)
                    .drop_duplicates("route", keep="first"))
    route_to_tcs = dict(zip(routes["route"], routes["track_list"]))

    prs["tcs"] = prs["route_id"].map(route_to_tcs)
    prs = prs.dropna(subset=["tcs"])
    prs["pr_id"] = range(len(prs))
    prs["tc_pos"] = prs["tcs"].apply(lambda lst: list(enumerate(lst)))

    exploded = prs[["pr_id", "time", "focal_train", "route_id", "tc_pos"]].explode("tc_pos")
    exploded[["tc_position", "tc_id"]] = pd.DataFrame(
        exploded["tc_pos"].tolist(), index=exploded.index)
    exploded = exploded.drop(columns=["tc_pos"]).reset_index(drop=True)
    exploded["n_tc_in_route"] = exploded.groupby("pr_id")["tc_id"].transform("count")
    exploded["relative_position"] = (
        exploded["tc_position"] / (exploded["n_tc_in_route"] - 1).clip(lower=1)
    )

    print(f"Joining {len(exploded):,} (PR×TC) pairs to latest Track state ...")
    exploded = exploded.sort_values("time").reset_index(drop=True)
    tracks = tracks.sort_values("tc_event_time").reset_index(drop=True)
    matched = pd.merge_asof(
        exploded, tracks, by="tc_id",
        left_on="time", right_on="tc_event_time", direction="backward",
    )

    matched["is_occupied"] = (matched["state"] == 1).astype("Int8")
    matched["by_focal"] = (matched["occupier"] == matched["focal_train"]).astype("Int8")
    matched["age_seconds"] = (matched["time"] - matched["tc_event_time"]).dt.total_seconds()

    # Per-PR aggregation
    per_pr = matched.groupby("pr_id").agg(
        n_tc=("tc_id", "count"),
        n_occupied=("is_occupied", "sum"),
        n_occupied_by_focal=("by_focal", "sum"),
    ).reset_index()
    per_pr["n_occupied_by_other"] = per_pr["n_occupied"] - per_pr["n_occupied_by_focal"]
    per_pr.to_parquet(out_dir / "conflict_per_pr.parquet", index=False)

    # Other-occupation drill-down
    other = matched[(matched["is_occupied"] == 1) & (matched["by_focal"] == 0)].copy()
    other.to_parquet(out_dir / "conflict_other_occupations.parquet", index=False)

    n_pr = len(per_pr)
    summary = {
        "n_panel_requests_analysed": int(n_pr),
        "n_with_zero_occupied_tcs": int((per_pr["n_occupied"] == 0).sum()),
        "pct_with_zero_occupied_tcs": round(
            100 * (per_pr["n_occupied"] == 0).mean(), 2),
        "n_with_occupation_by_focal_only": int(
            ((per_pr["n_occupied"] > 0) & (per_pr["n_occupied_by_other"] == 0)).sum()),
        "n_with_occupation_by_other": int((per_pr["n_occupied_by_other"] > 0).sum()),
        "pct_with_occupation_by_other": round(
            100 * (per_pr["n_occupied_by_other"] > 0).mean(), 2),
        "occupied_pos_distribution": pd.cut(
            other["relative_position"],
            bins=[-0.01, 0.05, 0.25, 0.5, 0.75, 1.01],
            labels=["start", "early", "mid", "late", "end"],
        ).value_counts().to_dict(),
        "occupation_age_seconds_describe": {
            k: float(v) for k, v in other["age_seconds"].describe(
                percentiles=[.1, .25, .5, .75, .9, .99]).items()
        },
        "top10_violating_routes": other["route_id"].value_counts().head(10).to_dict(),
    }
    summary["occupied_pos_distribution"] = {
        str(k): int(v) for k, v in summary["occupied_pos_distribution"].items()
    }
    summary_path = out_dir / "conflict_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))

    print()
    print(f"Wrote {out_dir}/conflict_summary.json")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nrows", type=int, default=None,
                        help="limit TD scan; default = full file")
    args = parser.parse_args()
    main(args.nrows)
