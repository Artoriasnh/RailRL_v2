"""Phase 1.1 — Streaming inventory of TD_data.csv and Movements.csv."""
from __future__ import annotations
import json
import time
from collections import Counter
from typing import Optional

import pandas as pd

from .. import config as C
from ..data_io import stream_td
from ..parsers import ROUTE_RE_PATTERN


def stream_td_inventory(
    chunksize: int = 3_000_000,
    nrows: Optional[int] = None,
) -> dict:
    """Single streaming pass — counts, types, prefixes, classes, monthly profile."""
    type_counts = Counter()
    pr_by_prefix = Counter()
    pr_by_class = Counter()
    rt_by_prefix = Counter()
    pr_by_signal = Counter()
    monthly_by_type: dict[str, dict[str, int]] = {}
    state_by_type: dict[str, dict[int, int]] = {}
    unparsed_pr = Counter()

    total_rows = 0
    min_t, max_t = None, None
    cols = ["time", "type", "id", "state"]

    t0 = time.time()
    for i, chunk in enumerate(stream_td(chunksize=chunksize, columns=cols, nrows=nrows)):
        total_rows += len(chunk)
        chunk["type"] = chunk["type"].astype(str)
        type_counts.update(chunk["type"].values)

        # state breakdown by type
        for (typ, st), n in chunk.groupby(["type", "state"]).size().items():
            state_by_type.setdefault(typ, {})
            state_by_type[typ][int(st) if pd.notna(st) else -1] = (
                state_by_type[typ].get(int(st) if pd.notna(st) else -1, 0) + int(n)
            )

        # monthly counts
        mo = chunk["time"].dt.strftime("%Y-%m")
        for (m, typ), n in chunk.groupby([mo, "type"]).size().items():
            monthly_by_type.setdefault(m, {})
            monthly_by_type[m][typ] = monthly_by_type[m].get(typ, 0) + int(n)

        # Panel_Request: vectorised parse + aggregations
        pr = chunk.loc[chunk["type"] == "Panel_Request", "id"].astype(str)
        ext = pr.str.extract(ROUTE_RE_PATTERN)
        good = ext.dropna(subset=["prefix"])
        pr_by_prefix.update(good["prefix"].values)
        pr_by_class.update(good["cls"].values)
        for (p, s), n in good.groupby(["prefix", "signal"]).size().items():
            pr_by_signal[(p, s)] += int(n)
        unp = pr[ext["prefix"].isna()]
        for v, c in unp.value_counts().head(50).items():
            unparsed_pr[v] += int(c)

        # Route(release) prefix breakdown
        rt = chunk.loc[chunk["type"] == "Route", "id"].astype(str)
        rt_ext = rt.str.extract(ROUTE_RE_PATTERN).dropna(subset=["prefix"])
        rt_by_prefix.update(rt_ext["prefix"].values)

        cmin, cmax = chunk["time"].min(), chunk["time"].max()
        min_t = cmin if min_t is None else min(min_t, cmin)
        max_t = cmax if max_t is None else max(max_t, cmax)
        print(f"  chunk {i}: rows={total_rows:,}  elapsed={time.time()-t0:.1f}s", flush=True)

    return {
        "total_rows": int(total_rows),
        "elapsed_s": round(time.time() - t0, 2),
        "min_time": str(min_t),
        "max_time": str(max_t),
        "type_counts": dict(type_counts),
        "state_by_type": state_by_type,
        "panel_request_by_prefix": dict(pr_by_prefix),
        "route_release_by_prefix": dict(rt_by_prefix),
        "panel_request_by_class": dict(pr_by_class),
        "panel_request_top20_signals": dict(
            Counter({f"{p}-{s}": n for (p, s), n in pr_by_signal.items()}).most_common(20)
        ),
        "unparsed_panel_request_top10": dict(unparsed_pr.most_common(10)),
        "monthly_by_type": monthly_by_type,
    }


def movements_inventory() -> dict:
    """Full pass over Movements.csv (50 MB, 247 k rows)."""
    df = pd.read_csv(
        C.MOVEMENTS_CSV,
        usecols=[
            "event_type", "actual_timestamp", "platform",
            "timetable_variation", "variation_status",
            "train_id", "toc_id", "loc_stanox",
        ],
        parse_dates=["actual_timestamp"],
        low_memory=False,
    )
    return {
        "total_rows": int(len(df)),
        "min_time": str(df["actual_timestamp"].min()),
        "max_time": str(df["actual_timestamp"].max()),
        "event_type_counts": df["event_type"].value_counts().to_dict(),
        "variation_status_counts": df["variation_status"].value_counts().to_dict(),
        "platform_counts": {
            str(k): int(v)
            for k, v in df["platform"].value_counts(dropna=False).head(15).items()
        },
        "toc_id_counts": df["toc_id"].value_counts().head(15).to_dict(),
        "timetable_variation_describe": {
            k: float(v) for k, v in df["timetable_variation"].describe().to_dict().items()
        },
        "unique_train_ids": int(df["train_id"].nunique()),
        "unique_loc_stanox": int(df["loc_stanox"].nunique()),
        "on_time_pct": round(100 * (df["variation_status"] == "ON TIME").mean(), 2),
        "late_pct": round(100 * (df["variation_status"] == "LATE").mean(), 2),
        "early_pct": round(100 * (df["variation_status"] == "EARLY").mean(), 2),
    }


def run(nrows: Optional[int] = None) -> tuple[dict, dict]:
    """Run both inventories and persist JSON. Returns the (td, movements) dicts."""
    print("=== Phase 1.1 inventory ===")
    print("Streaming TD ...")
    td = stream_td_inventory(nrows=nrows)
    print("Reading Movements ...")
    mv = movements_inventory()

    C.INVENTORY_TD_JSON.write_text(json.dumps(td, indent=2, default=str))
    C.INVENTORY_MOVEMENTS_JSON.write_text(json.dumps(mv, indent=2, default=str))
    print(f"Wrote {C.INVENTORY_TD_JSON}")
    print(f"Wrote {C.INVENTORY_MOVEMENTS_JSON}")
    return td, mv
