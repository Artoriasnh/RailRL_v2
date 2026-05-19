"""Loader for Derby_info.csv — per-route physical features.

User decision: from this file we ONLY take physical attributes
    length_m, ave_speed_mps, ave_grad, gap_time_s, n_points
the track list and start/end signals continue to come from route_to_tc_all.csv
because the two sources differ on track-list conventions in 69 % of routes
(documented in spec v2.2; not a bug in either source).
"""
from __future__ import annotations
import ast
import pandas as pd

from .. import config as C


def load_route_physical_features() -> pd.DataFrame:
    """Return per-route physical features keyed by route_id (with leading R).

    Columns:
        route_id           string  e.g. 'RDC5061A(M)'  (R prefix added)
        length_m           float   physical route length in metres
        ave_speed_mps      float   average traversal speed (m/s)
        ave_grad           float   average gradient (Network Rail units)
        gap_time_s         float   canonical traversal time (s) — used by L3 simulator
        n_points           Int64   number of turnouts on the route's path
    """
    df = pd.read_csv(C.DERBY_INFO_CSV)

    # Strip stray whitespace in routeid (Derby_info has at least one trailing-space bug,
    # 'TD5049B(M) ' — same upstream issue we hit in route_to_tc_all)
    df["routeid"] = df["routeid"].astype(str).str.strip()
    df["route_id"] = "R" + df["routeid"]

    # Keep rows with all physical features present (drop the 7 NaN rows)
    keep = df.dropna(subset=["length", "ave_speed(m/s)", "ave_grad", "gap_time(s)"]).copy()

    # n_points from path list (if present)
    def _count_points(s):
        if not isinstance(s, str):
            return 0
        try:
            return len(ast.literal_eval(s))
        except Exception:
            return 0
    keep["n_points"] = keep["path"].apply(_count_points).astype("Int64")

    out = keep[[
        "route_id", "length", "ave_speed(m/s)", "ave_grad", "gap_time(s)", "n_points",
    ]].rename(columns={
        "length":          "length_m",
        "ave_speed(m/s)":  "ave_speed_mps",
        "gap_time(s)":     "gap_time_s",
    })
    return out.sort_values("route_id").reset_index(drop=True)
