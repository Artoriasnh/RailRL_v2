"""Phase 1.1 — Extract decision events from Panel_Request rows.

Each Panel_Request becomes one (state, action) candidate row:
    (time, route_id, prefix, signal_no, letter, sub, cls,
     train_id, hc_class_digit, hc_dest, hc_serial)

State features themselves are added in Phase 1.2; this module produces the
action-label table and the train-identity context.
"""
from __future__ import annotations
import json
import time
from typing import Optional

import pandas as pd

from .. import config as C
from ..parsers import ROUTE_RE_PATTERN, HEADCODE_RE_PATTERN


def extract(nrows: Optional[int] = None) -> pd.DataFrame:
    """Read TD CSV, extract Panel_Request rows, parse route + headcode."""
    t0 = time.time()
    df = pd.read_csv(
        C.TD_CSV,
        usecols=["time", "type", "id", "trainid_filled"],
        parse_dates=["time"],
        nrows=nrows,
        low_memory=False,
    )
    print(f"  loaded {len(df):,} rows in {time.time() - t0:.1f}s", flush=True)

    pr = df[df["type"] == "Panel_Request"].copy()
    print(f"  Panel_Request rows: {len(pr):,}", flush=True)

    ext = pr["id"].astype(str).str.extract(ROUTE_RE_PATTERN)
    pr = pr.assign(
        prefix=ext["prefix"].values,
        signal_no=ext["signal"].values,
        letter=ext["letter"].values,
        sub=ext["sub"].values,
        cls=ext["cls"].values,
    )
    n_unparsed = pr["prefix"].isna().sum()
    pr = pr[pr["prefix"].notna()].copy()
    print(f"  unparsed dropped: {n_unparsed}", flush=True)

    hc = pr["trainid_filled"].astype(str).str.extract(HEADCODE_RE_PATTERN)
    pr["hc_class_digit"] = hc["hc_cls"].values
    pr["hc_dest"] = hc["hc_dest"].values
    pr["hc_serial"] = hc["hc_serial"].values

    out = pr.rename(columns={"id": "route_id", "trainid_filled": "train_id"})[[
        "time", "route_id", "prefix", "signal_no", "letter", "sub", "cls",
        "train_id", "hc_class_digit", "hc_dest", "hc_serial",
    ]].reset_index(drop=True)
    return out


def summarise(df: pd.DataFrame) -> dict:
    return {
        "total_decision_events": int(len(df)),
        "by_prefix": df["prefix"].value_counts().to_dict(),
        "by_class": df["cls"].value_counts().to_dict(),
        "with_headcode_parsed": int(df["hc_class_digit"].notna().sum()),
        "headcode_parse_rate_pct": round(100 * df["hc_class_digit"].notna().mean(), 2),
        "headcode_class_counts": df["hc_class_digit"].value_counts(dropna=False).to_dict(),
        "unique_train_ids": int(df["train_id"].nunique()),
        "unique_signals": int(df.groupby(["prefix", "signal_no"]).ngroups),
        "unique_routes": int(df["route_id"].nunique()),
        "time_range": [str(df["time"].min()), str(df["time"].max())],
    }


def run(nrows: Optional[int] = None) -> pd.DataFrame:
    """Extract, summarise, persist. Returns the decision-event dataframe."""
    print("=== Phase 1.1 decisions ===")
    df = extract(nrows=nrows)
    df.to_parquet(C.DECISION_EVENTS_PARQUET, index=False, compression="zstd")
    print(f"Wrote {len(df):,} decision events → {C.DECISION_EVENTS_PARQUET}")

    summary = summarise(df)
    C.DECISION_EVENTS_SUMMARY.write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))
    return df
