"""L1 panel-schematic — plotly version (interactive HTML + static PNG/SVG).

Replaces the matplotlib renderer in `panel_schematic.py` (which Hao called "ugly"). This
module produces:
  * an **interactive HTML** per decision — hover for TC name + saliency + type, zoom in;
  * a static **PNG/SVG** (via kaleido) — paper-ready;
  * with **explicit region labels** (Duffield North, Spondon East, Etches Park, Chaddesden
    Sidings, Pear Tree South, Sinfin, Matlock, Litchurch, Derby Station P1–P6) so the figure
    self-explains without needing the reader to know the layout convention;
  * with **tight zoom per decision** — bounds set to the salient nodes' bounding box + padding,
    so empty platforms/regions aren't shown when irrelevant. Aggregate panel keeps the full
    Derby view.

Reuses helpers from `panel_schematic` (DERBY_LAYOUT, build_tc_adjacency, route_tc_path,
fill_positions, vocab loaders) — only the rendering changes.
"""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
from typing import Optional

from .panel_schematic import (
    DERBY_LAYOUT, PLAT_Y, PLAT_X, build_tc_adjacency, fill_positions,
    load_vocabs, node_kind, route_tc_path,
)

# Region label rectangles — (name, x0,y0, x1,y1, fill_rgba). Drawn UNDER nodes.
REGIONS = [
    # Top: north approach
    ("↑ Duffield (North)",       30, 92, 70, 100, "rgba(180,200,255,0.15)"),
    # Derby station bracket — encompasses all 6 platforms
    ("Derby Station — Platforms 1–6 (TPS/TYT/TNG/TRJ/TDP/TFM)",
                                  20, 14, 78, 94, "rgba(250,250,250,0.55)"),
    # East: Spondon approach
    ("Spondon (East) → · TD5045 · Nottingham (TD5030)",
                                  72, 38, 100, 64, "rgba(255,220,170,0.18)"),
    # Lower-right sidings
    ("Etches Park sidings",      58, 4, 80, 18,  "rgba(200,240,200,0.18)"),
    ("Chaddesden Sidings",       82, 2, 100, 18, "rgba(255,200,220,0.18)"),
    # Left: south / west
    ("↓ Pear Tree · Birmingham · Crewe", 0, 14, 22, 36, "rgba(180,240,250,0.18)"),
    ("Sinfin branch",             0, 0, 12, 7,    "rgba(230,230,230,0.30)"),
    ("Matlock branch",           38, 0, 58, 9,    "rgba(230,230,230,0.30)"),
    ("Litchurch Ln",              0, 44, 8, 56,   "rgba(230,230,230,0.30)"),
]


def _resolve_name(nd: dict, vocabs: dict) -> Optional[str]:
    nt = nd["type"]
    nt_logical = {"trn": "train", "track": "track", "signal": "signal", "route": "route"}.get(nt, nt)
    ivi = nd.get("ident_vocab_idx")
    if ivi is None:
        return None
    return vocabs.get(nt_logical, {}).get(int(ivi))


def render_decision_plotly(top_nodes: list, vocabs: dict, adj: dict,
                           focal_route_id: Optional[str],
                           candidate_route_ids: Optional[list],
                           sample_id, stratum, output_dir: Path,
                           top_k: int = 15, write_static: bool = True) -> dict:
    """One decision → HTML (always) + PNG/SVG (if write_static)."""
    import plotly.graph_objects as go

    # ---- 1) resolve top-K spatial nodes + side-panel route/train top ----
    top = sorted([n for n in top_nodes if n.get("saliency", 0) > 0],
                 key=lambda d: -d["saliency"])[:top_k]
    sal_by_name: dict = {}                # spatial nodes (track + signal)
    sal_meta: dict = {}                   # name → (type, is_focal)
    side: dict = {"route": [], "train": []}
    for nd in top:
        nm = _resolve_name(nd, vocabs)
        nt = nd["type"]
        if nm is None:
            continue
        nt_logical = {"trn": "train"}.get(nt, nt)
        if nt_logical in ("track", "signal"):
            sal_by_name[nm] = max(sal_by_name.get(nm, 0.0), float(nd["saliency"]))
            sal_meta[nm] = (nt_logical, bool(nd.get("is_focal") or False))
        else:
            side[nt_logical].append({"name": nm, "saliency": float(nd["saliency"]),
                                     "is_focal": bool(nd.get("is_focal") or False)})

    # add chosen route's TCs as low-sal context (so the orange path renders)
    route_tcs = route_tc_path(focal_route_id) if focal_route_id else []
    for tc in route_tcs:
        sal_by_name.setdefault(tc, 0.0)
        sal_meta.setdefault(tc, ("track", False))

    # ---- 2) positions ----
    pos = fill_positions(list(sal_by_name.keys()), adj)

    # ---- 3) figure with region shapes + nodes + edges ----
    fig = go.Figure()
    # region rectangles + their labels
    for (name, x0, y0, x1, y1, fill) in REGIONS:
        fig.add_shape(type="rect", x0=x0, y0=y0, x1=x1, y1=y1, line=dict(width=0),
                      fillcolor=fill, layer="below")
        fig.add_annotation(x=(x0 + x1) / 2, y=y1 - 1, text=f"<b>{name}</b>",
                           showarrow=False, font=dict(size=10, color="#444"),
                           bgcolor="rgba(255,255,255,0.6)", borderpad=2)
    # platform lane markers (subtle horizontal lines)
    for p, y in PLAT_Y.items():
        fig.add_shape(type="line", x0=22, y0=y, x1=78, y1=y,
                      line=dict(color="rgba(180,180,180,0.6)", width=10),
                      layer="below")
        fig.add_annotation(x=20, y=y, text=f"P{p}", showarrow=False,
                           xanchor="right", font=dict(size=11, color="#666"))

    # edges among rendered nodes (light gray)
    edge_xs, edge_ys = [], []
    drawn = set()
    for u in sal_by_name:
        for v in adj.get(u, []):
            if v in pos and (v, u) not in drawn:
                xu, yu = pos[u]; xv, yv = pos[v]
                edge_xs += [xu, xv, None]
                edge_ys += [yu, yv, None]
                drawn.add((u, v))
    if edge_xs:
        fig.add_trace(go.Scatter(x=edge_xs, y=edge_ys, mode="lines",
                                 line=dict(color="rgba(150,150,150,0.5)", width=1),
                                 hoverinfo="skip", showlegend=False))

    # chosen route path (orange, thick)
    if route_tcs:
        rxs = [pos[t][0] for t in route_tcs if t in pos]
        rys = [pos[t][1] for t in route_tcs if t in pos]
        if len(rxs) >= 2:
            fig.add_trace(go.Scatter(x=rxs, y=rys, mode="lines",
                                     line=dict(color="rgba(255,140,0,0.8)", width=4),
                                     hoverinfo="skip",
                                     name=f"chosen route {focal_route_id}"))

    # nodes (TCs + signals separately, different markers)
    smax = max(sal_by_name.values()) if sal_by_name else 1.0
    smax = smax if smax > 0 else 1.0
    tc_x, tc_y, tc_s, tc_text, tc_hover = [], [], [], [], []
    sg_x, sg_y, sg_s, sg_text, sg_hover = [], [], [], [], []
    for name, sal in sal_by_name.items():
        x, y = pos[name]
        kind, is_focal = sal_meta.get(name, ("track", False))
        if kind == "signal":
            sg_x.append(x); sg_y.append(y); sg_s.append(sal)
            sg_text.append(name if sal > 0 else "")
            sg_hover.append(f"<b>{name}</b><br>signal<br>IG saliency: {sal:.3f}")
        else:
            tc_x.append(x); tc_y.append(y); tc_s.append(sal)
            tc_text.append(name if sal > 0 else "")
            tc_hover.append(f"<b>{name}</b><br>track-circuit{' ★focal' if is_focal else ''}"
                            f"<br>IG saliency: {sal:.3f}")
    # TC = circles
    if tc_x:
        fig.add_trace(go.Scatter(
            x=tc_x, y=tc_y, mode="markers+text",
            marker=dict(size=[8 + 20 * (s / smax) for s in tc_s],
                        color=tc_s, colorscale="Reds", cmin=0, cmax=smax,
                        line=dict(color="rgba(0,0,0,0.55)", width=0.8),
                        showscale=True,
                        colorbar=dict(title="IG saliency", thickness=12, len=0.4,
                                      x=1.02, y=0.5, tickfont=dict(size=9))),
            text=tc_text, textposition="top right",
            textfont=dict(size=10, color="#222"),
            hovertext=tc_hover, hoverinfo="text",
            name="TC (track-circuit)"))
    # signals = diamonds
    if sg_x:
        fig.add_trace(go.Scatter(
            x=sg_x, y=sg_y, mode="markers+text",
            marker=dict(size=[8 + 18 * (s / smax) for s in sg_s],
                        color=sg_s, colorscale="Reds", cmin=0, cmax=smax,
                        symbol="diamond",
                        line=dict(color="rgba(0,0,0,0.55)", width=0.8),
                        showscale=False),
            text=sg_text, textposition="bottom right",
            textfont=dict(size=10, color="#222"),
            hovertext=sg_hover, hoverinfo="text",
            name="Signal"))

    # side panel: top routes + trains as a fixed annotation
    side_lines = []
    for nt in ("route", "train"):
        if side[nt]:
            side_lines.append(f"<b>top {nt}s</b>")
            for d in side[nt][:5]:
                star = " ★focal" if d["is_focal"] else ""
                side_lines.append(f"  {d['name']}  {d['saliency']:.3f}{star}")
    if side_lines:
        fig.add_annotation(x=1.18, y=0.95, xref="paper", yref="paper",
                           text="<br>".join(side_lines), showarrow=False,
                           align="left", xanchor="left", yanchor="top",
                           font=dict(size=10, family="monospace", color="#333"),
                           bgcolor="rgba(245,245,245,0.95)",
                           bordercolor="rgba(180,180,180,0.6)", borderwidth=1, borderpad=8)

    # ---- 4) tight bounding box: zoom to salient nodes + padding ----
    if pos:
        xs = [p[0] for p in pos.values()]
        ys = [p[1] for p in pos.values()]
        # always include platforms area for orientation
        x_lo = max(-2, min(min(xs) - 6, 18))
        x_hi = min(105, max(max(xs) + 6, 82))
        y_lo = max(-2, min(min(ys) - 6, 12))
        y_hi = min(105, max(max(ys) + 6, 96))
    else:
        x_lo, x_hi, y_lo, y_hi = -2, 105, -2, 102

    title = (f"<b>L1 IG node-saliency</b> · sample_id={sample_id} · stratum={stratum}"
             f"<br><span style='font-size:11px;color:#666'>"
             f"chosen route = <b>{focal_route_id or '?'}</b> · "
             f"node size+color = IG saliency · orange line = chosen route's TCs</span>")
    fig.update_layout(
        title=dict(text=title, x=0.02, xanchor="left", font=dict(size=14)),
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(range=[x_lo, x_hi], visible=False, scaleanchor="y", scaleratio=1),
        yaxis=dict(range=[y_lo, y_hi], visible=False),
        margin=dict(l=40, r=180, t=70, b=20),
        width=1200, height=720,
        showlegend=False, hoverlabel=dict(bgcolor="white"),
        font=dict(family="Inter, Helvetica, Arial, sans-serif"),
    )

    # ---- 5) write outputs ----
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    base = f"decision_{sample_id}_{stratum}"
    html_path = output_dir / f"{base}.html"
    fig.write_html(html_path, include_plotlyjs="cdn", config={"displaylogo": False})
    png_path = None
    if write_static:
        png_path = output_dir / f"{base}.png"
        try:
            fig.write_image(png_path, scale=2)
        except Exception as e:
            print(f"  [warn] static export failed for {base}: {e}")
            png_path = None

    return {
        "sample_id": sample_id, "stratum": stratum,
        "focal_route_id": focal_route_id,
        "positions": {n: list(p) for n, p in pos.items()},
        "sal_by_name": sal_by_name,
        "side_top": side,
        "route_tcs": route_tcs,
        "html_path": str(html_path),
        "png_path": str(png_path) if png_path else None,
    }


def render_aggregate_plotly(per_decision_data: list, output_dir: Path,
                            write_static: bool = True, top_k_labels: int = 30) -> dict:
    """Aggregate panel: mean saliency across all decisions, on the FULL Derby view (no zoom)."""
    import plotly.graph_objects as go

    agg: dict = defaultdict(list)
    for d in per_decision_data:
        for name, sal in d.get("sal_by_name", {}).items():
            agg[name].append(sal)
    mean_sal = {n: sum(v) / len(v) for n, v in agg.items()}
    coverage = {n: len(v) / max(1, len(per_decision_data)) for n, v in agg.items()}
    smax = max(mean_sal.values()) if mean_sal else 1.0
    smax = smax if smax > 0 else 1.0

    adj = build_tc_adjacency()
    pos = fill_positions(list(mean_sal.keys()), adj)

    fig = go.Figure()
    for (name, x0, y0, x1, y1, fill) in REGIONS:
        fig.add_shape(type="rect", x0=x0, y0=y0, x1=x1, y1=y1, line=dict(width=0),
                      fillcolor=fill, layer="below")
        fig.add_annotation(x=(x0 + x1) / 2, y=y1 - 1, text=f"<b>{name}</b>",
                           showarrow=False, font=dict(size=10, color="#444"),
                           bgcolor="rgba(255,255,255,0.6)", borderpad=2)
    for p, y in PLAT_Y.items():
        fig.add_shape(type="line", x0=22, y0=y, x1=78, y1=y,
                      line=dict(color="rgba(180,180,180,0.6)", width=10),
                      layer="below")
        fig.add_annotation(x=20, y=y, text=f"P{p}", showarrow=False,
                           xanchor="right", font=dict(size=11, color="#666"))

    # nodes: split by kind, label only top-K
    tops = set(n for n, _ in sorted(mean_sal.items(), key=lambda kv: -kv[1])[:top_k_labels])
    tx, ty, ts, ttext, thov = [], [], [], [], []
    sx, sy, ss, stext, shov = [], [], [], [], []
    for name, sal in mean_sal.items():
        x, y = pos[name]
        hov = (f"<b>{name}</b><br>mean IG saliency: {sal:.3f}"
               f"<br>coverage: {100*coverage[name]:.0f}% of decisions")
        label = name if name in tops else ""
        if node_kind(name) == "signal":
            sx.append(x); sy.append(y); ss.append(sal); stext.append(label); shov.append(hov)
        else:
            tx.append(x); ty.append(y); ts.append(sal); ttext.append(label); thov.append(hov)
    if tx:
        fig.add_trace(go.Scatter(
            x=tx, y=ty, mode="markers+text",
            marker=dict(size=[6 + 22 * (s / smax) for s in ts],
                        color=ts, colorscale="Reds", cmin=0, cmax=smax,
                        line=dict(color="rgba(0,0,0,0.55)", width=0.6),
                        showscale=True, colorbar=dict(title="mean IG", thickness=12, len=0.4,
                                                     x=1.02, y=0.5, tickfont=dict(size=9))),
            text=ttext, textposition="top right",
            textfont=dict(size=9, color="#222"),
            hovertext=thov, hoverinfo="text", name="TC"))
    if sx:
        fig.add_trace(go.Scatter(
            x=sx, y=sy, mode="markers+text",
            marker=dict(size=[6 + 20 * (s / smax) for s in ss],
                        color=ss, colorscale="Reds", cmin=0, cmax=smax,
                        symbol="diamond", line=dict(color="rgba(0,0,0,0.55)", width=0.6),
                        showscale=False),
            text=stext, textposition="bottom right",
            textfont=dict(size=9, color="#222"),
            hovertext=shov, hoverinfo="text", name="Signal"))

    fig.update_layout(
        title=dict(text=f"<b>L1 IG saliency — aggregate across {len(per_decision_data)} decisions</b>"
                   f"<br><span style='font-size:11px;color:#666'>"
                   f"node size + color = mean saliency where the node appeared in top-K · "
                   f"hover for coverage</span>",
                   x=0.02, xanchor="left", font=dict(size=14)),
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(range=[-2, 105], visible=False, scaleanchor="y", scaleratio=1),
        yaxis=dict(range=[-2, 102], visible=False),
        margin=dict(l=40, r=140, t=80, b=20),
        width=1400, height=860,
        showlegend=False, hoverlabel=dict(bgcolor="white"),
        font=dict(family="Inter, Helvetica, Arial, sans-serif"),
    )

    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / "aggregate_panel.html"
    fig.write_html(html_path, include_plotlyjs="cdn", config={"displaylogo": False})
    png_path = None
    if write_static:
        png_path = output_dir / "aggregate_panel.png"
        try:
            fig.write_image(png_path, scale=2)
        except Exception as e:
            print(f"  [warn] aggregate static export failed: {e}")
            png_path = None

    return {"n_decisions": len(per_decision_data),
            "mean_sal": mean_sal, "coverage": coverage,
            "html_path": str(html_path),
            "png_path": str(png_path) if png_path else None}
