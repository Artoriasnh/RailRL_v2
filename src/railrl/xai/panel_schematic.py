"""L1 — Derby schematic for IG node saliency (spec 05 §7.3, panel-heatmap surrogate).

Replaces the deferred manual `panel_layout.json` (TC/signal → pixel coordinates on the real
Derby panel image) with an **abstracted operational schematic** built from data we already
have. Per Hao (2026-05-28): "we don't need the actual panel — we can render our own diagram
from the infrastructure adjacencies".

Pipeline:
  * `load_vocabs()` — invert the normalization-stats vocab (`track.track_id`, etc.) to map
    `ident_vocab_idx` → real TC/signal/route names.
  * `build_tc_adjacency()` — derive TC↔TC adjacency from route track-lists (consecutive TCs in
    a route's track are physically adjacent). Pure CSV; no parquet/torch needed.
  * `DERBY_LAYOUT` — semi-hand-coded **operational** (x,y) positions for ~60 anchor TCs/signals
    covering the 6 platforms + the L4-rule signals (TD5045/TD5049/DC5076/DC5061/DC5065/DW5302/
    DW5310) + the major branches (Sinfin DW5320, Matlock DY572, RTC sidings). Other TCs are
    placed near their adjacent anchors via BFS-radial fallback.
  * `render_decision(...)` — matplotlib figure for one decision: top-K IG-salient nodes drawn
    in the schematic, with the focal train + candidate routes' TCs highlighted; node color =
    saliency (Reds), edges = TC adjacency.
  * `render_aggregate(...)` — single panel with all anchored TCs, color = cross-decision mean
    saliency. The companion adjacency-matrix heatmap is `render_adjacency_matrix(...)`.

All generated data files and figures are saved to `outputs/figures/l1_panel/` for reuse:
PNG figures, the layout JSON, the adjacency JSON, the per-decision render data JSON, and an
INDEX.md catalog.

Pure-python + matplotlib (no networkx / torch / pyarrow) → sandbox-testable.
"""
from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional

from .. import config as C

# ============================================================
# Vocab + topology loaders
# ============================================================

def load_vocabs() -> dict:
    """ident_vocab_idx → name for each node type, derived from normalization_stats.json."""
    s = json.loads((C.NORMALIZATION_STATS_JSON).read_text())
    v = s["vocab"]
    out = {}
    # vocab key naming: track.track_id / signal.signal_id / route.route_id / train.headcode etc.
    # The IG node types are track/signal/route/train. Find the *_id vocab per type.
    for nt in ("track", "signal", "route", "train"):
        candidates = [k for k in v if k.startswith(f"{nt}.") and ("_id" in k or "id" in k.lower())]
        chosen = candidates[0] if candidates else None
        # fall back to any vocab under this prefix
        if chosen is None:
            chosen = next((k for k in v if k.startswith(f"{nt}.")), None)
        if chosen is None:
            out[nt] = {}
            continue
        idx2name = {int(i): n for n, i in v[chosen]["index"].items()}
        out[nt] = idx2name
    return out


def build_tc_adjacency() -> dict:
    """TC name → set of adjacent TC names. From route_to_tc_all.csv: consecutive TCs in a
    route's track_list are physically adjacent (a train passing through traverses them in
    order)."""
    adj: dict = defaultdict(set)
    with open(C.ROUTE_TO_TC_CSV, newline="") as f:
        for r in csv.DictReader(f):
            tcs = re.findall(r"'([A-Z0-9]+)'", r.get("track", "") or "")
            for a, b in zip(tcs[:-1], tcs[1:]):
                adj[a].add(b)
                adj[b].add(a)
    return {k: sorted(v) for k, v in adj.items()}


def route_tc_path(route_id: str) -> list:
    """route_id → ordered list of TCs it traverses (for showing the chosen route on the panel)."""
    with open(C.ROUTE_TO_TC_CSV, newline="") as f:
        for r in csv.DictReader(f):
            if r["route"].strip() == route_id:
                return re.findall(r"'([A-Z0-9]+)'", r.get("track", "") or "")
    return []


# ============================================================
# Operational hand-coded layout (Derby anchors)
# ============================================================
# Coordinates are in [0,100]×[0,100]. Convention:
#   X: 0 = west/Pear-Tree/Burton end → 100 = east/Spondon end
#   Y: bottom (10) = platform 6, top (90) = platform 1
# Platforms are 6 horizontal lanes; each platform has sub-sections A (west/south end) /
# middle / B (east/north end). North-end signals (DC50xx) on the LEFT, south-end (DW53xx /
# TD50xx) on the RIGHT.
# This was hand-positioned to mimic the operational topology — not geographic; the goal is
# readability for a railway audience.

PLAT_Y = {1: 92, 2: 80, 3: 66, 4: 52, 5: 38, 6: 24}
PLAT_X = {"A": 32, "middle": 50, "B": 68}   # sub-section → x

DERBY_LAYOUT = {
    # Platform 1 (TPSL=A, TPSM=middle, TPSU=B)
    "TPSL": (PLAT_X["A"], PLAT_Y[1]), "TPSM": (PLAT_X["middle"], PLAT_Y[1]), "TPSU": (PLAT_X["B"], PLAT_Y[1]),
    # Platform 2
    "TYTW": (PLAT_X["A"], PLAT_Y[2]), "TYTV": (PLAT_X["middle"], PLAT_Y[2]), "TYTS": (PLAT_X["B"], PLAT_Y[2]),
    # Platform 3
    "TNGR": (PLAT_X["A"], PLAT_Y[3]), "TNGS": (PLAT_X["middle"], PLAT_Y[3]), "TNGU": (PLAT_X["B"], PLAT_Y[3]),
    # Platform 4
    "TRJY": (PLAT_X["A"], PLAT_Y[4]), "TRJW": (PLAT_X["middle"], PLAT_Y[4]), "TRJV": (PLAT_X["B"], PLAT_Y[4]),
    # Platform 5
    "TDPG": (PLAT_X["A"], PLAT_Y[5]), "TDPJ": (PLAT_X["middle"], PLAT_Y[5]), "TDPK": (PLAT_X["B"], PLAT_Y[5]),
    # Platform 6
    "TFMP": (PLAT_X["A"], PLAT_Y[6]), "TFMN": (PLAT_X["middle"], PLAT_Y[6]), "TFML": (PLAT_X["B"], PLAT_Y[6]),
    # East approach TCs (TD5045 / Spondon-Nottingham side) — x=78-92, spread vertically
    "TDMZ": (88, 50), "TDPA": (82, 48), "TDPC": (78, 50), "TRKC": (74, 50), "TRKA": (72, 48),
    "TNGK": (78, 56), "TNGM": (74, 60),
    "TDPB": (82, 36), "TDPE": (78, 36), "TDPF": (74, 36),
    "TFPB": (90, 28), "TFMY": (86, 26), "TFMW": (82, 24), "TFMU": (78, 24), "TFMR": (75, 22),
    "TFPV": (94, 18),                    # Nottingham (TD5030)
    "TDMC": (92, 50),                    # Spondon→Derby
    # West/South approach (Pear Tree, Birmingham/Crewe/Stenson side)
    "TYVR": (8, 18),                     # peartree (B3/4/5/6 anchor)
    "TYVN": (12, 22), "TYVM": (16, 24), "TYVL": (20, 26),
    "TYWB": (14, 28), "TYWC": (18, 30),
    # North end (Duffield / Chesterfield corridor)
    "T884": (50, 99), "T883": (50, 96),
    # Etches Park / Chaddesden
    "TECF": (84, 8), "TECJ": (88, 10), "TECK": (80, 10), "TECL": (76, 10), "TECM": (72, 10),
    "TECR": (68, 8), "TECS": (64, 8), "TECV": (60, 8),
    # RTC / Sinfin / Matlock branches (drawn off to the right edge for clarity)
    # Signals — placed near their primary TC. Signals don't have their own TCs in vocab,
    # but the IG can attribute to them; we render them with a different marker.
    "DC5061": (28, 92), "DC5062": (28, 80), "DC5063": (28, 66), "DC5064": (28, 52),
    "DC5065": (28, 38), "DC5066": (28, 24),                  # north-end exit signals
    "DW5301": (72, 92), "DW5302": (72, 80),                  # south-end exits (plat 1=Burton, plat 2=West)
    "DW5310": (4, 50),                                       # Litchurch
    "DW5319": (4, 18),                                       # Birmingham/Crewe/Stenson peartree
    "DW5320": (4, 4),                                        # Sinfin North
    "DY572": (50, 5), "DY571": (44, 5),                      # Matlock branch
    "TD5045": (96, 48), "TD5043": (96, 12), "TD5049": (96, 60), "DC5076": (28, 60),
    # ---- Etches Park / Chaddesden sidings corridor (south-east of station, y=4-16) ----
    "TEDA": (78, 14), "TEDB": (74, 14), "TEDC": (70, 14), "TEDG": (66, 14), "TEDH": (62, 12),
    "TEDJ": (60, 10), "TEDK": (66, 8), "TEDM": (70, 8), "TEDN": (74, 8),
    "TECA": (84, 6), "TECB": (82, 6), "TECD": (80, 6), "TECG": (76, 6), "TECW": (72, 6),
    # already had TECF/TECJ/TECK/TECL/TECM/TECR/TECS/TECV above (y=8-10)
    # Chaddesden Y-area (further east)
    "TGAJ": (92, 16), "TGAK": (94, 14), "TGAL": (96, 14), "TGAM": (96, 12),
    "TGAR": (94, 10), "TGAS": (92, 8), "TGAU": (90, 6), "TGAW": (88, 4),
    # ---- Junction TCs north of platforms (TRK*, TNJ*) ----
    "TRKB": (76, 56), "TRKD": (72, 52), "TRKE": (70, 50), "TRKG": (68, 48), "TRKH": (66, 46),
    "TRKJ": (64, 50), "TRKL": (62, 48), "TRKM": (60, 50), "TRKN": (58, 52), "TRKS": (62, 54),
    "TRKU": (64, 56), "TRKV": (66, 58),
    "TNJB": (30, 78), "TNJC": (28, 76), "TNJH": (26, 74), "TNJP": (24, 72),
    "TNJR": (22, 70), "TNJS": (20, 68),
    # ---- North end TCs (Duffield/Chesterfield corridor, T8xx/T9xx) ----
    "T868": (40, 99), "T871": (42, 99), "T872": (44, 99), "T874": (46, 99),
    "T878": (48, 99), "T895": (52, 99), "T896": (54, 99), "T897": (56, 99),
    "T898": (58, 99), "T899": (60, 99), "T901": (38, 97), "T902": (40, 97), "T903": (42, 97),
    "T904": (44, 97), "T905": (46, 97), "T906": (48, 97), "T907": (50, 97), "T908": (52, 97),
    "T909": (54, 97), "T910": (56, 97), "T911": (58, 97), "T912": (60, 97),
    "T915": (44, 95), "T917": (46, 95), "T918": (48, 95), "T920": (50, 95),
    "T921": (52, 95), "T922": (54, 95), "T925": (56, 95), "T927": (58, 95),
    # ---- Pear Tree area + Sinfin south (TYV*, TPR*, TFE*) ----
    "TYVA": (5, 14), "TYVB": (7, 14), "TYVC": (9, 14), "TYVD": (11, 14), "TYVG": (4, 16),
    "TYVJ": (6, 16), "TYVK": (8, 18), "TYVU": (10, 20), "TYVW": (12, 22), "TYVS": (12, 18),
    "TYWE": (16, 28), "TYWG": (18, 30), "TYWH": (20, 32),
    "TPRB": (8, 28), "TPRC": (10, 28), "TPRD": (10, 30), "TPRE": (12, 30), "TPRG": (14, 30),
    "TPRH": (16, 32), "TPRJ": (14, 32), "TPRM": (16, 30), "TPRU": (18, 32),
    "TPSB": (28, 88), "TPSC": (30, 88), "TPSD": (32, 88), "TPSF": (34, 88),
    "TPSG": (36, 90), "TPSH": (40, 90), "TPSJ": (42, 90),
    "TFEV": (4, 4), "TFEY": (6, 4),
    # ---- Pear Tree / Stenson approach (TFD*, TFP* not already placed, TFA*, TFB*) ----
    "TFDB": (22, 30), "TFDC": (20, 32), "TFDE": (18, 32), "TFDG": (16, 30),
    "TFDK": (24, 28), "TFDM": (22, 26), "TFDP": (20, 26), "TFDR": (18, 26),
    "TFDS": (16, 26), "TFDU": (14, 26), "TFDW": (16, 24), "TFDY": (18, 22),
    "TFAB": (20, 22), "TFAE": (22, 22), "TFAG": (24, 22), "TFAH": (26, 22),
    "TFAJ": (28, 22), "TFAK": (30, 22), "TFAL": (32, 24), "TFAM": (28, 24),
    "TFAS": (26, 26), "TFAW": (24, 28),
    "TFBD": (32, 30), "TFBE": (30, 30), "TFBG": (28, 30), "TFBK": (26, 30),
    "TFBN": (24, 32), "TFBP": (26, 32), "TFBR": (28, 34), "TFBU": (30, 32),
    "TFMB": (40, 22), "TFMC": (42, 22), "TFMD": (44, 22), "TFME": (46, 22),
    "TFMH": (52, 24), "TFMJ": (54, 24),
    # ---- Spondon approach details (more TDM*) ----
    "TDMB": (90, 56), "TDMC": (92, 50), "TDME": (90, 52), "TDMF": (88, 54),
    "TDMH": (86, 54), "TDML": (84, 52), "TDMN": (88, 48), "TDMP": (90, 46),
    "TDMS": (92, 44), "TDMU": (94, 42), "TDMV": (90, 40), "TDMW": (88, 42),
}


# bare-number → prefixed signal alias (signal vocab in normalization_stats uses bare numbers
# like "5045", but DERBY_LAYOUT uses prefixed "TD5045"). Build a quick lookup at module load.
_BARE_TO_PREFIXED_SIGNAL = {}
for _k in list(DERBY_LAYOUT.keys()):
    _m = re.match(r"^[A-Z]{2}(\d{3,4})$", _k)
    if _m:
        _BARE_TO_PREFIXED_SIGNAL[_m.group(1)] = _k

# A type marker for the layout entries (TC vs signal) — used for plotting glyphs.
def node_kind(name: str) -> str:
    """Heuristic: TC names start with 'T', signals usually 'D'/'TD'/'DC'/'DW'/'DY'/'EC'."""
    if not name:
        return "other"
    if name.startswith("T") and not re.match(r"^TD\d", name):
        # TXXXX track circuit (e.g. TPSL, TNGK, T884)
        return "track"
    if re.match(r"^[A-Z]{2}\d", name) or name.startswith("TD"):
        return "signal"
    return "other"


# ============================================================
# BFS-radial fallback layout for non-anchored TCs
# ============================================================

def _resolve_anchor(name: str, anchors: dict):
    """Direct hit on DERBY_LAYOUT, or — for bare-number signal names like '5045' — try the
    prefixed alias ('TD5045'). Returns (x,y) or None."""
    if name in anchors:
        return anchors[name]
    aliased = _BARE_TO_PREFIXED_SIGNAL.get(name)
    if aliased and aliased in anchors:
        return anchors[aliased]
    return None


def fill_positions(saliency_names: list, adj: dict, anchors: dict = DERBY_LAYOUT) -> dict:
    """Position non-anchored TCs near their nearest anchored neighbor (BFS from anchors).
    Bare-number signal names (e.g. '5045') resolve via _BARE_TO_PREFIXED_SIGNAL."""
    import math
    pos = {}; remaining = []
    for n in saliency_names:
        p = _resolve_anchor(n, anchors)
        if p is not None:
            pos[n] = p
        else:
            remaining.append(n)
    # BFS distance from any anchor
    dist = {a: 0 for a in pos}
    frontier = list(pos.keys())
    parent = {a: None for a in pos}
    while frontier:
        nxt = []
        for u in frontier:
            for v in adj.get(u, []):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    nxt.append(v)
        frontier = nxt
    # place each remaining node at offset from its parent (radial small jitter)
    used = set(pos)
    for n in remaining:
        if n in dist and parent.get(n) in pos:
            px, py = pos[parent[n]]
            # offset by a small radial step depending on the hop distance
            theta = (hash(n) % 360) * (math.pi / 180)
            r = 3 + 1.5 * (dist[n] - 1)
            pos[n] = (px + r * math.cos(theta), py + r * math.sin(theta))
        else:
            # unreachable from any anchor → drop off-canvas to the right
            pos[n] = (102, 50 + (hash(n) % 20 - 10))
        used.add(n)
    return pos


# ============================================================
# Rendering
# ============================================================

def render_decision(top_nodes: list, vocabs: dict, adj: dict,
                    focal_route_id: Optional[str] = None,
                    candidate_route_ids: Optional[list] = None,
                    sample_id: Optional[int] = None,
                    stratum: Optional[str] = None,
                    title: Optional[str] = None,
                    output_path: Optional[Path] = None,
                    top_k: int = 20) -> dict:
    """Render one decision's IG saliency on the Derby schematic.

    top_nodes: list of dicts as produced by l1_attention.integrated_gradients (each has
      'type','local_idx','ident_vocab_idx','is_focal','saliency').
    Returns the rendering data dict (positions + saliency_map + chosen route TCs) for archival.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, FancyBboxPatch

    # ---- 1. resolve top_nodes ident_vocab_idx → real names per type, keep top_k ----
    top = sorted([n for n in top_nodes if n.get("saliency", 0) > 0],
                 key=lambda d: -d["saliency"])[:top_k]
    sal_by_name = {}
    sal_by_type_top = {"track": [], "signal": [], "route": [], "train": []}
    for nd in top:
        nt = nd["type"]
        # the IG output uses PyG type keys: trn for train; map back
        nt_logical = {"trn": "train", "track": "track", "signal": "signal", "route": "route"}.get(nt, nt)
        ivi = nd.get("ident_vocab_idx")
        name = vocabs.get(nt_logical, {}).get(int(ivi)) if ivi is not None else None
        sal_by_type_top[nt_logical].append({"name": name, "saliency": float(nd["saliency"]),
                                            "is_focal": bool(nd.get("is_focal") or False)})
        if name and nt_logical in ("track", "signal"):
            sal_by_name[name] = max(sal_by_name.get(name, 0.0), float(nd["saliency"]))

    # ---- 2. add chosen route's TCs as low-saliency context ----
    route_tcs = route_tc_path(focal_route_id) if focal_route_id else []
    for tc in route_tcs:
        sal_by_name.setdefault(tc, 0.0)

    # ---- 3. position all nodes (anchored or via BFS) ----
    spatial_names = list(sal_by_name.keys())
    pos = fill_positions(spatial_names, adj)

    # ---- 4. figure ----
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(-2, 105)
    ax.set_ylim(0, 102)
    ax.set_aspect("equal")
    ax.set_axis_off()

    # ground: faint platform lanes + sub-section markers
    for p, y in PLAT_Y.items():
        ax.plot([22, 78], [y, y], color="#e0e0e0", linewidth=12, zorder=0)
        ax.text(-1, y, f"P{p}", ha="right", va="center", fontsize=9, color="#888")
    ax.text(20, 96, "← North (Duffield)", color="#888", fontsize=9)
    ax.text(80, 96, "East (Spondon) →", color="#888", fontsize=9, ha="right")

    # adjacency edges among rendered nodes (only edges where BOTH ends are positioned)
    drawn = set()
    for u in spatial_names:
        for v in adj.get(u, []):
            if v in pos and (v, u) not in drawn:
                xu, yu = pos[u]; xv, yv = pos[v]
                ax.plot([xu, xv], [yu, yv], color="#cccccc", linewidth=0.6, zorder=1)
                drawn.add((u, v))

    # chosen route path: thick highlighted line
    if route_tcs:
        path_xy = [pos[t] for t in route_tcs if t in pos]
        if len(path_xy) >= 2:
            xs, ys = zip(*path_xy)
            ax.plot(xs, ys, color="#ffaa00", linewidth=3.0, alpha=0.7, zorder=2,
                    label=f"chosen route {focal_route_id}")

    # nodes: color by saliency (Reds), size by salience too
    smax = max((s for s in sal_by_name.values()), default=1.0) or 1.0
    cmap = plt.get_cmap("Reds")
    for name, sal in sal_by_name.items():
        x, y = pos[name]
        norm = sal / smax
        kind = node_kind(name)
        if kind == "signal":
            # signals: diamond marker
            ax.plot(x, y, marker="D", markersize=6 + 9 * norm, color=cmap(0.25 + 0.7 * norm),
                    markeredgecolor="black", markeredgewidth=0.4, zorder=3)
        else:
            # TC: circle
            ax.plot(x, y, marker="o", markersize=6 + 11 * norm, color=cmap(0.25 + 0.7 * norm),
                    markeredgecolor="black", markeredgewidth=0.4, zorder=3)

    # label only the top-8 salient + focal
    top_labels = sorted(sal_by_name.items(), key=lambda kv: -kv[1])[:8]
    for name, sal in top_labels:
        x, y = pos[name]
        ax.annotate(name, (x, y), xytext=(4, 4), textcoords="offset points",
                    fontsize=8, color="#333", zorder=4,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))

    # title + side panel: non-spatial top nodes (route, train) listed numerically
    ttl = title or f"L1 IG saliency — sample_id={sample_id}  stratum={stratum}"
    ax.set_title(ttl, fontsize=11, loc="left")
    side_lines = []
    for nt in ("route", "train"):
        if sal_by_type_top[nt]:
            side_lines.append(f"top {nt}s:")
            for d in sal_by_type_top[nt][:5]:
                mark = " ★focal" if d["is_focal"] else ""
                side_lines.append(f"  {d['name'] or '?'}  {d['saliency']:.3f}{mark}")
    if side_lines:
        ax.text(102, 90, "\n".join(side_lines), fontsize=8, va="top", ha="left",
                family="monospace", color="#333",
                bbox=dict(boxstyle="round,pad=0.3", fc="#f5f5f5", ec="#ccc"))

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "sample_id": sample_id, "stratum": stratum,
        "focal_route_id": focal_route_id,
        "positions": {n: list(p) for n, p in pos.items()},
        "sal_by_name": sal_by_name,
        "sal_by_type_top": sal_by_type_top,
        "route_tcs": route_tcs,
        "output_path": str(output_path) if output_path else None,
    }


def render_aggregate(per_decision_data: list, output_path: Path, top_k: int = 40) -> dict:
    """Aggregate saliency across all decisions → one panel.
    per_decision_data: list of dicts returned by render_decision (each has sal_by_name).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    agg: dict = defaultdict(list)
    for d in per_decision_data:
        for name, sal in d.get("sal_by_name", {}).items():
            agg[name].append(sal)
    # mean saliency per node across decisions where it appears
    mean_sal = {n: sum(v) / len(v) for n, v in agg.items()}
    coverage = {n: len(v) / max(1, len(per_decision_data)) for n, v in agg.items()}

    spatial_names = list(mean_sal.keys())
    adj = build_tc_adjacency()
    pos = fill_positions(spatial_names, adj)

    fig, ax = plt.subplots(figsize=(15, 9))
    ax.set_xlim(-2, 105); ax.set_ylim(0, 102); ax.set_aspect("equal"); ax.set_axis_off()
    for p, y in PLAT_Y.items():
        ax.plot([22, 78], [y, y], color="#e8e8e8", linewidth=12, zorder=0)
        ax.text(-1, y, f"P{p}", ha="right", va="center", fontsize=9, color="#888")

    smax = max(mean_sal.values()) if mean_sal else 1.0
    cmap = plt.get_cmap("Reds")
    for name, sal in mean_sal.items():
        x, y = pos[name]
        norm = sal / smax if smax > 0 else 0
        kind = node_kind(name)
        mk = "D" if kind == "signal" else "o"
        ax.plot(x, y, marker=mk, markersize=5 + 12 * norm,
                color=cmap(0.2 + 0.75 * norm), markeredgecolor="black",
                markeredgewidth=0.3, zorder=3, alpha=0.85)
    # label top-K by mean saliency
    for name, sal in sorted(mean_sal.items(), key=lambda kv: -kv[1])[:top_k]:
        x, y = pos[name]
        ax.annotate(f"{name}", (x, y), xytext=(3, 3), textcoords="offset points",
                    fontsize=7, color="#222", zorder=4,
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7))

    ax.set_title(f"L1 IG saliency aggregate ({len(per_decision_data)} decisions; "
                 f"node size+color = mean saliency)", fontsize=11, loc="left")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return {"n_decisions": len(per_decision_data),
            "mean_sal": mean_sal, "coverage": coverage,
            "output_path": str(output_path)}


def render_adjacency_matrix(adj: dict, output_path: Path,
                            saliency_per_tc: Optional[dict] = None,
                            top_k: int = 80) -> None:
    """TC×TC adjacency heatmap (binary 0/1), with optional row/col reordering by saliency."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # restrict to the most salient TCs if given, otherwise all TCs in adj
    if saliency_per_tc:
        tcs = sorted([t for t in adj if t in saliency_per_tc],
                     key=lambda t: -saliency_per_tc.get(t, 0))[:top_k]
    else:
        tcs = sorted(adj.keys())[:top_k]
    n = len(tcs); idx = {t: i for i, t in enumerate(tcs)}
    M = np.zeros((n, n))
    for u in tcs:
        for v in adj.get(u, []):
            if v in idx:
                M[idx[u], idx[v]] = 1.0
    fig, ax = plt.subplots(figsize=(max(8, n * 0.12), max(8, n * 0.12)))
    ax.imshow(M, cmap="Greys", aspect="equal", interpolation="nearest")
    ax.set_xticks(range(n)); ax.set_xticklabels(tcs, rotation=90, fontsize=6)
    ax.set_yticks(range(n)); ax.set_yticklabels(tcs, fontsize=6)
    ax.set_title(f"TC adjacency (top-{n}{' by saliency' if saliency_per_tc else ''})", fontsize=10)
    plt.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
