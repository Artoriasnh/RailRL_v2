#!/usr/bin/env python3
"""
Empirical analysis: route × headcode-class correlation.

Verifies the hypothesis that minority routes at a signal are class-specialised
(e.g. EC-5475 B(M) is 98% ECS class-5).

Run:
    python scripts/analyses/route_class_skew.py             # full TD file
    python scripts/analyses/route_class_skew.py --nrows 5000000

Outputs:
    outputs/p2_data_eng/analyses/route_class_crosstab.parquet
    outputs/p2_data_eng/analyses/route_class_summary.json
"""
from __future__ import annotations
import argparse, json
import pandas as pd
from railrl import config as C
from railrl.parsers import ROUTE_RE_PATTERN, HEADCODE_RE_PATTERN


def class_label(c):
    if pd.isna(c): return "non_standard"
    if c in {"7", "8"}: return "other"
    return c


def main(nrows: int | None) -> None:
    out_dir = C.OUTPUTS_DIR / "p2_data_eng" / "analyses"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(C.TD_CSV, usecols=["type", "id", "trainid_filled"],
                     nrows=nrows, low_memory=False)
    pr = df[df["type"] == "Panel_Request"].copy()

    ext_r = pr["id"].astype(str).str.extract(ROUTE_RE_PATTERN)
    pr = pr.assign(prefix=ext_r["prefix"].values, signal_no=ext_r["signal"].values,
                   letter=ext_r["letter"].values, cls=ext_r["cls"].values)
    pr = pr[pr["prefix"].notna()].copy()

    ext_h = pr["trainid_filled"].astype(str).str.extract(HEADCODE_RE_PATTERN)
    pr["hc_label"] = ext_h["hc_cls"].apply(class_label)

    sig_route = pr.groupby(["prefix", "signal_no", "letter", "cls"]).agg(
        n=("hc_label", "count"),
        top_class=("hc_label", lambda x: x.value_counts().index[0]),
        top_class_pct=("hc_label", lambda x: x.value_counts(normalize=True).iloc[0] * 100),
    ).reset_index()
    sig_route["signal_id"] = sig_route["prefix"] + "-" + sig_route["signal_no"]
    sig_total = sig_route.groupby("signal_id")["n"].transform("sum")
    sig_route["route_share_pct"] = sig_route["n"] / sig_total * 100

    sig_route.to_parquet(out_dir / "route_class_crosstab.parquet", index=False)

    minority = sig_route[(sig_route["route_share_pct"] < 20) & (sig_route["n"] >= 100)]
    majority = sig_route[(sig_route["route_share_pct"] >= 50) & (sig_route["n"] >= 100)]

    summary = {
        "n_panel_requests": int(len(pr)),
        "n_unique_routes": int(len(sig_route)),
        "minority_routes_count": int(len(minority)),
        "majority_routes_count": int(len(majority)),
        "minority_top_class_pct_describe": minority["top_class_pct"].describe(
            percentiles=[.25, .5, .75, .9]).to_dict(),
        "majority_top_class_pct_describe": majority["top_class_pct"].describe(
            percentiles=[.25, .5, .75, .9]).to_dict(),
        "top10_class_concentrated_minority_routes": (
            minority.sort_values("top_class_pct", ascending=False)
                    .head(10)
                    [["signal_id", "letter", "cls", "n", "route_share_pct",
                      "top_class", "top_class_pct"]]
                    .to_dict(orient="records")),
    }
    (out_dir / "route_class_summary.json").write_text(
        json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--nrows", type=int, default=None)
    main(p.parse_args().nrows)
