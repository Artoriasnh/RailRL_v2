"""Generate an editable SVG framework figure for the RailRL paper.

The figure is intentionally publication-style and vector-only: all labels are
SVG text, all objects are editable shapes, and confidential local signalling
details are not shown.
"""
from __future__ import annotations

from html import escape
from pathlib import Path
import textwrap


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "figures" / "railrl_framework"
SVG = OUT / "railrl_end_to_end_framework.svg"

W, H = 1900, 1060


COL = {
    "ink": "#1f2937",
    "muted": "#5f6b7a",
    "line": "#91a2b6",
    "paper": "#fbfcff",
    "panel_stroke": "#9fb0c5",
    "blue": "#dfeeff",
    "blue_dark": "#2f6fb2",
    "green": "#e6f5e8",
    "green_dark": "#2f7d54",
    "peach": "#fff0e4",
    "peach_dark": "#bd6b35",
    "yellow": "#fff7cc",
    "yellow_dark": "#9b7a13",
    "violet": "#efe9ff",
    "violet_dark": "#7158b6",
    "cyan": "#dff7f4",
    "cyan_dark": "#247f86",
    "rose": "#fde8ed",
    "rose_dark": "#a84d63",
    "rail": "#f7e5d6",
    "white": "#ffffff",
}


def svg_header() -> str:
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">
  <defs>
    <marker id="arrow" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto" markerUnits="strokeWidth">
      <path d="M 0 0 L 12 6 L 0 12 z" fill="#3f5268"/>
    </marker>
    <marker id="arrowBlue" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto" markerUnits="strokeWidth">
      <path d="M 0 0 L 12 6 L 0 12 z" fill="{COL['blue_dark']}"/>
    </marker>
    <style>
      .title {{ font: 700 31px Arial, Helvetica, sans-serif; fill: {COL['ink']}; }}
      .subtitle {{ font: 400 17px Arial, Helvetica, sans-serif; fill: {COL['muted']}; }}
      .panel-title {{ font: 700 19px Arial, Helvetica, sans-serif; fill: {COL['ink']}; }}
      .panel-tag {{ font: 700 16px Arial, Helvetica, sans-serif; fill: {COL['white']}; }}
      .box-title {{ font: 700 15px Arial, Helvetica, sans-serif; fill: {COL['ink']}; }}
      .small {{ font: 400 13px Arial, Helvetica, sans-serif; fill: {COL['muted']}; }}
      .tiny {{ font: 400 11px Arial, Helvetica, sans-serif; fill: {COL['muted']}; }}
      .caption {{ font: 400 14px Arial, Helvetica, sans-serif; fill: {COL['ink']}; }}
      .rail-title {{ font: 700 16px Arial, Helvetica, sans-serif; fill: {COL['ink']}; }}
      .arrow {{ stroke: #3f5268; stroke-width: 2.2; fill: none; marker-end: url(#arrow); }}
      .arrow-blue {{ stroke: {COL['blue_dark']}; stroke-width: 2.1; fill: none; marker-end: url(#arrowBlue); }}
      .feedback {{ stroke: #d48a3a; stroke-width: 2; stroke-dasharray: 8 7; fill: none; marker-end: url(#arrow); }}
      .dash {{ stroke-dasharray: 7 6; }}
    </style>
  </defs>
'''


def rect(x, y, w, h, fill, stroke=COL["panel_stroke"], rx=14, sw=1.5, cls="") -> str:
    return (
        f'<rect class="{cls}" x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'
    )


def line(x1, y1, x2, y2, cls="arrow") -> str:
    return f'<path class="{cls}" d="M {x1} {y1} L {x2} {y2}"/>'


def path(d, cls="arrow") -> str:
    return f'<path class="{cls}" d="{d}"/>'


def text(x, y, s, cls="small", anchor="start") -> str:
    return f'<text class="{cls}" x="{x}" y="{y}" text-anchor="{anchor}">{escape(s)}</text>'


def wrapped_text(x, y, s, width=25, cls="small", line_h=16, anchor="start") -> str:
    lines = textwrap.wrap(s, width=width, break_long_words=False)
    tspans = []
    for i, ln in enumerate(lines):
        dy = 0 if i == 0 else line_h
        tspans.append(f'<tspan x="{x}" dy="{dy}">{escape(ln)}</tspan>')
    return f'<text class="{cls}" x="{x}" y="{y}" text-anchor="{anchor}">' + "".join(tspans) + "</text>"


def panel(x, y, w, h, tag, title, fill, tag_fill) -> str:
    parts = [
        rect(x, y, w, h, fill, rx=18, sw=1.6),
        f'<circle cx="{x + 27}" cy="{y + 27}" r="17" fill="{tag_fill}"/>',
        text(x + 27, y + 33, tag, "panel-tag", "middle"),
        text(x + 54, y + 32, title, "panel-title"),
    ]
    return "\n".join(parts)


def module_box(x, y, w, h, title, body, fill="#ffffff", stroke=COL["line"], icon=None) -> str:
    parts = [rect(x, y, w, h, fill, stroke=stroke, rx=10, sw=1.2)]
    if icon:
        parts.append(icon(x + 15, y + 15))
        tx = x + 47
    else:
        tx = x + 15
    parts.append(wrapped_text(tx, y + 24, title, width=max(12, int((w - (tx - x) - 10) / 7)), cls="box-title", line_h=16))
    parts.append(wrapped_text(x + 15, y + 54, body, width=max(18, int((w - 28) / 7)), cls="small", line_h=15))
    return "\n".join(parts)


def icon_feed(x, y) -> str:
    return (
        f'<rect x="{x}" y="{y}" width="22" height="27" rx="3" fill="#ffffff" stroke="{COL["blue_dark"]}" stroke-width="1.5"/>'
        f'<path d="M {x+5} {y+8} H {x+17} M {x+5} {y+14} H {x+17} M {x+5} {y+20} H {x+13}" stroke="{COL["blue_dark"]}" stroke-width="1.5"/>'
    )


def icon_db(x, y) -> str:
    return (
        f'<ellipse cx="{x+14}" cy="{y+6}" rx="14" ry="6" fill="#ffffff" stroke="{COL["green_dark"]}" stroke-width="1.5"/>'
        f'<path d="M {x} {y+6} V {y+27} C {x} {y+35}, {x+28} {y+35}, {x+28} {y+27} V {y+6}" fill="#ffffff" stroke="{COL["green_dark"]}" stroke-width="1.5"/>'
        f'<path d="M {x} {y+17} C {x} {y+25}, {x+28} {y+25}, {x+28} {y+17}" fill="none" stroke="{COL["green_dark"]}" stroke-width="1.2"/>'
    )


def icon_graph(x, y) -> str:
    pts = [(x + 4, y + 18), (x + 18, y + 5), (x + 31, y + 18), (x + 18, y + 31)]
    segs = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]
    out = [f'<path d="{" ".join([f"M {pts[a][0]} {pts[a][1]} L {pts[b][0]} {pts[b][1]}" for a,b in segs])}" stroke="{COL["peach_dark"]}" stroke-width="1.3" fill="none"/>']
    for px, py in pts:
        out.append(f'<circle cx="{px}" cy="{py}" r="4.7" fill="#ffffff" stroke="{COL["peach_dark"]}" stroke-width="1.5"/>')
    return "".join(out)


def icon_model(x, y) -> str:
    parts = []
    for i, c in enumerate(["#cfe0ff", "#d8f0dc", "#efe0ff"]):
        parts.append(f'<rect x="{x + i*8}" y="{y + i*6}" width="25" height="25" fill="{c}" stroke="{COL["violet_dark"]}" stroke-width="1"/>')
    return "".join(parts)


def icon_eval(x, y) -> str:
    return (
        f'<rect x="{x}" y="{y+4}" width="32" height="28" rx="3" fill="#ffffff" stroke="{COL["cyan_dark"]}" stroke-width="1.4"/>'
        f'<path d="M {x+5} {y+25} L {x+12} {y+17} L {x+19} {y+21} L {x+27} {y+11}" fill="none" stroke="{COL["cyan_dark"]}" stroke-width="2"/>'
        f'<path d="M {x+5} {y+29} H {x+29}" stroke="{COL["cyan_dark"]}" stroke-width="1.2"/>'
    )


def mini_event_stream(x, y) -> str:
    parts = [text(x, y, "event sequence", "tiny")]
    colors = [COL["blue_dark"], COL["green_dark"], COL["peach_dark"], COL["violet_dark"], COL["cyan_dark"]]
    for i in range(18):
        parts.append(f'<rect x="{x + i*8}" y="{y+10}" width="5" height="{12 + (i % 4)*5}" fill="{colors[i % len(colors)]}" opacity="0.75"/>')
    return "\n".join(parts)


def mini_graph(x, y) -> str:
    pts = [(x + 18, y + 8), (x + 52, y + 8), (x + 86, y + 8), (x + 36, y + 42), (x + 70, y + 42)]
    out = []
    for a, b in [(0, 1), (1, 2), (0, 3), (1, 3), (1, 4), (2, 4), (3, 4)]:
        out.append(f'<path d="M {pts[a][0]} {pts[a][1]} L {pts[b][0]} {pts[b][1]}" stroke="#7990aa" stroke-width="1.5"/>')
    for i, (px, py) in enumerate(pts):
        fill = ["#cfe0ff", "#d8f0dc", "#ffe0c4", "#efe0ff", "#dff7f4"][i]
        out.append(f'<circle cx="{px}" cy="{py}" r="9" fill="{fill}" stroke="#58708c" stroke-width="1.2"/>')
    out.append(text(x + 8, y + 64, "heterogeneous railway graph", "tiny"))
    return "\n".join(out)


def mini_q_network(x, y) -> str:
    parts = []
    for i, lab in enumerate(["wait", "R1", "R2", "R3"]):
        h = [30, 48, 38, 20][i]
        bx = x + i * 35
        parts.append(f'<rect x="{bx}" y="{y + 55 - h}" width="21" height="{h}" fill="#8db6e8" stroke="{COL["blue_dark"]}" stroke-width="1.2"/>')
        parts.append(text(bx + 10.5, y + 70, lab, "tiny", "middle"))
    parts.append(text(x, y + 10, "per-action Q(s,a)", "tiny"))
    return "\n".join(parts)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    s: list[str] = [svg_header(), rect(0, 0, W, H, COL["paper"], stroke="none", rx=0)]

    s.append(text(58, 54, "RailRL end-to-end framework for explainable railway route-setting decisions", "title"))
    s.append(text(60, 82, "From operational feed acquisition to leak-controlled offline RL, evaluation, and decision-support evidence", "subtitle"))

    # Panel geometry
    top_y, ph = 122, 690
    gap = 24
    pw = [330, 330, 365, 390, 360]
    xs = [58]
    for i in range(1, 5):
        xs.append(xs[i - 1] + pw[i - 1] + gap)

    panels = [
        ("1", "Operational acquisition", COL["blue"], COL["blue_dark"]),
        ("2", "Structured data basis", COL["green"], COL["green_dark"]),
        ("3", "Decision MDP dataset", COL["peach"], COL["peach_dark"]),
        ("4", "Offline RL model", COL["violet"], COL["violet_dark"]),
        ("5", "Evaluation and XAI", COL["cyan"], COL["cyan_dark"]),
    ]
    for i, (tag, title_, fill, tag_fill) in enumerate(panels):
        s.append(panel(xs[i], top_y, pw[i], ph, tag, title_, fill, tag_fill))

    # Panel 1
    x = xs[0]
    s.append(module_box(x + 24, top_y + 70, 282, 118, "Operational feed sources", "Train Describer, Train Movements, SCHEDULE/VSTP, RTPPM, and static reference files.", "#ffffff", COL["blue_dark"], icon_feed))
    s.append(module_box(x + 24, top_y + 212, 282, 132, "Application-based collector", "Feed selection, broker subscription, connection checks, runtime logging, and acquisition monitoring.", "#ffffff", COL["blue_dark"]))
    s.append(module_box(x + 24, top_y + 370, 282, 122, "Continuity support", "Heartbeat detection, safe cleanup, bounded retry, subscription recovery, and resumed collection.", "#ffffff", COL["blue_dark"]))
    s.append(module_box(x + 24, top_y + 522, 282, 112, "Persistent raw archive", "Feed-specific records are preserved for traceability and reproducible reconstruction.", "#ffffff", COL["blue_dark"], icon_db))
    s.append(line(x + 165, top_y + 188, x + 165, top_y + 212))
    s.append(line(x + 165, top_y + 344, x + 165, top_y + 370))
    s.append(line(x + 165, top_y + 492, x + 165, top_y + 522))

    # Panel 2
    x = xs[1]
    s.append(module_box(x + 24, top_y + 70, 282, 110, "Canonical event store", "Berth transitions, route events, track occupancy, signal state, TRTS, and movement records.", "#ffffff", COL["green_dark"], icon_db))
    s.append(module_box(x + 24, top_y + 202, 282, 126, "Static railway graph", "Track, signal, route, platform, and train-context tables become a typed infrastructure graph.", "#ffffff", COL["green_dark"], icon_graph))
    s.append(mini_graph(x + 84, top_y + 350))
    s.append(module_box(x + 24, top_y + 456, 282, 124, "Data quality gates", "Temporal alignment, continuity checks, duplicate handling, coverage summaries, and provenance metadata.", "#ffffff", COL["green_dark"]))
    s.append(module_box(x + 24, top_y + 604, 282, 52, "Confidential local details are not shown.", "", "#ffffff", COL["rose_dark"]))

    # Panel 3
    x = xs[2]
    s.append(module_box(x + 24, top_y + 70, 317, 108, "Decision-point extraction", "SET decisions from panel requests; WAIT decisions from approach observations without imminent route setting.", "#ffffff", COL["peach_dark"]))
    s.append(module_box(x + 24, top_y + 200, 317, 112, "Dynamic action space", "A_t = {wait} plus feasible route candidates for the focal train and signal context.", "#ffffff", COL["peach_dark"]))
    s.append(module_box(x + 24, top_y + 334, 317, 124, "Leak-controlled state snapshot", "Centered railway subgraph, recent event tokens, schedule outlook, other active trains, and special-case flags.", "#ffffff", COL["peach_dark"], icon_graph))
    s.append(module_box(x + 24, top_y + 480, 317, 116, "Reward and transitions", "Delay, throughput, headway, and wait components; adjacent decisions in each episode define s to s'.", "#ffffff", COL["peach_dark"]))
    s.append(mini_event_stream(x + 74, top_y + 620))

    # Panel 4
    x = xs[3]
    s.append(module_box(x + 24, top_y + 70, 342, 92, "Graph branch", "HGT encoder over tracks, signals, routes, and trains.", "#ffffff", COL["violet_dark"], icon_model))
    s.append(module_box(x + 24, top_y + 182, 342, 92, "Event branch", "Transformer encoder over the last K=256 event tokens.", "#ffffff", COL["violet_dark"]))
    s.append(module_box(x + 24, top_y + 294, 342, 92, "Context branch", "Schedule outlook, scalar state, and special operational flags.", "#ffffff", COL["violet_dark"]))
    s.append(rect(x + 72, top_y + 414, 246, 88, "#fff9e6", stroke=COL["yellow_dark"], rx=12, sw=1.3))
    s.append(text(x + 195, top_y + 442, "Fusion representation z", "box-title", "middle"))
    s.append(text(x + 195, top_y + 466, "HGT + Transformer + context", "small", "middle"))
    s.append(line(x + 195, top_y + 386, x + 195, top_y + 414))
    s.append(mini_q_network(x + 122, top_y + 524))
    s.append(module_box(x + 24, top_y + 612, 342, 50, "CQL training with route/time auxiliary heads", "", "#ffffff", COL["violet_dark"]))

    # Panel 5
    x = xs[4]
    s.append(module_box(x + 24, top_y + 70, 312, 108, "Overall metrics", "Action top-1, set-only route choice, wait precision/recall, route head, time head, and Q-gap.", "#ffffff", COL["cyan_dark"], icon_eval))
    s.append(module_box(x + 24, top_y + 202, 312, 118, "Stratified evaluation", "Late trains, call-on, platform deviation, priority competition, unusual IDs, and trivial cases.", "#ffffff", COL["cyan_dark"]))
    s.append(module_box(x + 24, top_y + 344, 312, 112, "Replicate and improve", "Compare signaller action, model action, rule checks, and counterfactual operational delta.", "#ffffff", COL["cyan_dark"]))
    s.append(module_box(x + 24, top_y + 480, 312, 116, "Explainability outputs", "Attention/attribution, Q-gap rationale, rule evidence, simulator trace, and decision-support view.", "#ffffff", COL["cyan_dark"]))
    s.append(module_box(x + 24, top_y + 620, 312, 42, "Paper evidence: ESWA / T-ITS ready evaluation package", "", "#ffffff", COL["cyan_dark"]))

    # Cross-panel arrows
    for i in range(4):
        x1 = xs[i] + pw[i]
        x2 = xs[i + 1]
        y = top_y + 350
        s.append(path(f"M {x1 + 5} {y} C {x1 + 18} {y}, {x2 - 18} {y}, {x2 - 5} {y}", "arrow"))

    # Training feedback loop
    s.append(path(f"M {xs[4] + 180} {top_y + 620} C {xs[4] + 160} {top_y + 770}, {xs[3] + 190} {top_y + 770}, {xs[3] + 190} {top_y + 662}", "feedback"))
    s.append(text(xs[3] + 250, top_y + 762, "training / audit feedback", "tiny"))

    # Bottom evidence rail
    rail_y = 842
    s.append(rect(58, rail_y, W - 116, 150, COL["rail"], stroke="#e2b896", rx=18, sw=1.4))
    s.append(text(86, rail_y + 35, "Cross-cutting research controls", "rail-title"))
    rail_items = [
        ("Traceability", "raw messages -> canonical records -> sample_id"),
        ("Temporal causality", "state uses information available at decision time"),
        ("Leak control", "chosen action, rewards, and future outcomes excluded from state"),
        ("Reproducibility", "versioned scripts, summaries, audits, and checkpoints"),
        ("Confidentiality", "only aggregate event representation is reported"),
    ]
    item_w = 320
    for i, (a, b) in enumerate(rail_items):
        ix = 86 + i * 348
        s.append(rect(ix, rail_y + 54, item_w, 70, "#fffaf5", stroke="#d2a47f", rx=12, sw=1.1))
        s.append(text(ix + 18, rail_y + 81, a, "box-title"))
        s.append(wrapped_text(ix + 18, rail_y + 104, b, width=36, cls="tiny", line_h=13))

    # Dataset and figure note
    s.append(text(70, 1024, "Dataset scale: 14 months | 11.8M TD records | 247k movement records | about 2.0M decisions | 1.996M usable MDP snapshots", "caption"))
    s.append(text(W - 70, 1024, "Editable SVG: shapes and labels can be modified directly", "caption", "end"))

    s.append("</svg>\n")
    SVG.write_text("\n".join(s), encoding="utf-8")
    print(SVG)


if __name__ == "__main__":
    main()
