#!/usr/bin/env python3
"""
Empirical analysis: where do non-standard train_ids cluster?

Tests the hypothesis that non-standard ids (e.g. '343R', 'PARK', 'WAGS') are
concentrated in depot / sidings / shunt operations rather than mainline traffic.

Run:
    python scripts/analyses/non_standard_trainids.py               # full TD file
    python scripts/analyses/non_standard_trainids.py --nrows 5e6
"""
from __future__ import annotations
import argparse, json
import pandas as pd
from railrl import config as C
from railrl.parsers import ROUTE_RE_PATTERN, HEADCODE_RE_PATTERN


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
    pr["is_standard"] = ext_h["hc_cls"].notna()

    ns = pr[~pr["is_standard"]].copy()
    ns["signal_id"] = ns["prefix"] + "-" + ns["signal_no"]
    pr["signal_id"] = pr["prefix"] + "-" + pr["signal_no"]

    sig_total = pr.groupby("signal_id").size().rename("total_pr")
    sig_ns = ns.groupby("signal_id").size().rename("ns_pr")
    sig_concentrate = pd.concat([sig_total, sig_ns], axis=1).fillna(0)
    sig_concentrate["ns_pct"] = sig_concentrate["ns_pr"] / sig_concentrate["total_pr"] * 100
    sig_concentrate = sig_concentrate[sig_concentrate["total_pr"] >= 100].sort_values(
        "ns_pct", ascending=False)
    sig_concentrate.to_parquet(out_dir / "non_standard_by_signal.parquet")

    summary = {
        "n_total_pr": int(len(pr)),
        "n_non_standard_pr": int(len(ns)),
        "pct_non_standard": round(100 * len(ns) / max(len(pr), 1), 3),
        "prefix_distribution_overall": (pr["prefix"].value_counts(normalize=True) * 100)
            .round(2).to_dict(),
        "prefix_distribution_non_standard": (ns["prefix"].value_counts(normalize=True) * 100)
            .round(2).to_dict(),
        "class_distribution_overall": (pr["cls"].value_counts(normalize=True) * 100)
            .round(2).to_dict(),
        "class_distribution_non_standard": (ns["cls"].value_counts(normalize=True) * 100)
            .round(2).to_dict(),
        "top20_id_formats": ns["trainid_filled"].value_counts().head(20).to_dict(),
        "top10_signals_by_non_standard_fraction": sig_concentrate.head(10).round(2)
            .reset_index().to_dict(orient="records"),
    }
    (out_dir / "non_standard_trainids_summary.json").write_text(
        json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--nrows", type=int, default=None)
    main(p.parse_args().nrows)
