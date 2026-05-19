"""SOP file parser — Network Rail Signal Output Plan.

The .SOP file is the canonical Derby asset registry. Each line has:
    <byte_addr_or_blank>  <bit:0-7>  <element_name>

Element name prefixes:
    S  Signal           (e.g. SDC5061)
    R  Route            (e.g. RDC5061A(M))
    T  Track circuit    (e.g. TFDU, T868)
    L  Latch — TRTS, etc. (e.g. LPLAT1ATRS(N))
    P  Points
"""
from __future__ import annotations
import re
from typing import NamedTuple

import pandas as pd

from .. import config as C


_SOP_SIGNAL_RE = re.compile(r"^S(?P<prefix>DC|DW|DY|EC|TD)(?P<signal_no>\d+\w*)$")
_SOP_ROUTE_RE  = re.compile(
    r"^R(?P<prefix>DW|TD|DC|EC|DY)(?P<signal_no>\d+\w*)"
    r"(?P<letter>[A-Z]+)(?:-(?P<sub>\d+))?\((?P<cls>M|C|S|W|PS)\)$")
_SOP_TRTS_RE   = re.compile(
    r"^LPLAT(?P<platform>\d+)(?P<sub>[AB])TRS\((?P<direction>[NS])\)$")


def parse_sop_assets(sop_path = None) -> dict[str, list[str]]:
    """Walk the .SOP file and return a dict of asset-prefix → sorted asset names."""
    if sop_path is None:
        sop_path = C.SOP_FILE
    by_prefix: dict[str, set[str]] = {}
    with open(sop_path, "r", encoding="ascii", errors="ignore") as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            name = parts[-1]
            if not name or name[0] not in "SRTLP":
                continue
            by_prefix.setdefault(name[0], set()).add(name)
    return {k: sorted(v) for k, v in by_prefix.items()}


def parse_signals(sop_path = None) -> pd.DataFrame:
    """Return a DataFrame of all Derby signals in the SOP.

    Columns: signal_id (numeric tail), prefix (DW/TD/DC/EC/DY), full_name (SDC5061).
    """
    assets = parse_sop_assets(sop_path)
    rows = []
    for full in assets.get("S", []):
        m = _SOP_SIGNAL_RE.match(full)
        if m:
            rows.append({"signal_id": m.group("signal_no"),
                         "prefix":    m.group("prefix"),
                         "full_name": full})
    return pd.DataFrame(rows).sort_values("signal_id").reset_index(drop=True)


def parse_routes(sop_path = None) -> pd.DataFrame:
    """Return a DataFrame of all routes named in the SOP (cross-check against
    route_to_tc_all.csv — should match)."""
    assets = parse_sop_assets(sop_path)
    rows = []
    for full in assets.get("R", []):
        m = _SOP_ROUTE_RE.match(full)
        if m:
            rows.append({"route_id":  full,
                         "prefix":    m.group("prefix"),
                         "signal_no": m.group("signal_no"),
                         "letter":    m.group("letter"),
                         "sub":       m.group("sub"),
                         "cls":       m.group("cls")})
    return pd.DataFrame(rows).sort_values("route_id").reset_index(drop=True)


def parse_tracks(sop_path = None) -> pd.DataFrame:
    """Return a DataFrame of all track circuits in the SOP.

    Columns: track_id, naming_pattern (T+letters, T+digits, etc.).
    """
    assets = parse_sop_assets(sop_path)
    rows = []
    for full in assets.get("T", []):
        # Skip TRTS-related (those start with L not T anyway, but be safe)
        rows.append({"track_id": full,
                     "naming_pattern": "alpha" if any(c.isalpha() for c in full[1:]) else "numeric"})
    return pd.DataFrame(rows).sort_values("track_id").reset_index(drop=True)


def parse_trts(sop_path = None) -> pd.DataFrame:
    """Return a DataFrame of TRTS (Train Ready To Start) latches.

    Columns: trts_id, platform_id, sub_section (A/B), direction (N/S).
    Format: LPLAT<n><A|B>TRS(<N|S>) e.g. LPLAT1ATRS(N).
    """
    assets = parse_sop_assets(sop_path)
    rows = []
    for full in assets.get("L", []):
        m = _SOP_TRTS_RE.match(full)
        if m:
            rows.append({"trts_id":     full,
                         "platform_id": int(m.group("platform")),
                         "sub_section": m.group("sub"),
                         "direction":   m.group("direction")})
    return (pd.DataFrame(rows)
              .sort_values(["platform_id", "sub_section", "direction"])
              .reset_index(drop=True))
