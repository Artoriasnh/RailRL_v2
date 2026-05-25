"""Visual-first RailRL framework figure.

This version follows the visual grammar of IEEE/ESWA framework schematics:
large panels, compact labels, arrows, mini-graphs, tensor stacks, bar charts,
and dashboard-like evaluation outputs. It deliberately avoids confidential
local signalling details.
"""
from __future__ import annotations

from html import escape
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "figures" / "railrl_framework"
SVG = OUT / "railrl_end_to_end_framework_visual.svg"

W, H = 2100, 1180

COL = {
    "ink": "#1f2937",
    "muted": "#667085",
    "line": "#52677f",
    "soft_line": "#9aa9bb",
    "paper": "#fbfcff",
    "blue": "#e7f0ff",
    "blue2": "#b8d4f6",
    "blue3": "#2f6fb2",
    "green": "#e8f6ed",
    "green2": "#bde5c8",
    "green3": "#2f7d54",
    "peach": "#fff0e5",
    "peach2": "#ffd2b5",
    "peach3": "#bd6b35",
    "violet": "#efeaff",
    "violet2": "#d5c7ff",
    "violet3": "#7158b6",
    "cyan": "#e3f7f7",
    "cyan2": "#b9e6e8",
    "cyan3": "#247f86",
    "yellow": "#fff7cf",
    "yellow2": "#f7dd78",
    "yellow3": "#9b7a13",
    "rose": "#fde7ed",
    "rose2": "#f6b8c6",
    "rose3": "#a84d63",
    "white": "#ffffff",
    "rail": "#f7e6d7",
}


def header() -> str:
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">
  <defs>
    <marker id="arrow" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto" markerUnits="strokeWidth">
      <path d="M 0 0 L 12 6 L 0 12 z" fill="{COL['line']}"/>
    </marker>
    <marker id="arrowWarm" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto" markerUnits="strokeWidth">
      <path d="M 0 0 L 12 6 L 0 12 z" fill="{COL['peach3']}"/>
    </marker>
    <style>
      .title {{ font: 700 32px Arial, Helvetica, sans-serif; fill: {COL['ink']}; }}
      .sub {{ font: 400 17px Arial, Helvetica, sans-serif; fill: {COL['muted']}; }}
      .panel {{ font: 700 20px Arial, Helvetica, sans-serif; fill: {COL['ink']}; }}
      .tag {{ font: 700 15px Arial, Helvetica, sans-serif; fill: {COL['white']}; }}
      .label {{ font: 700 14px Arial, Helvetica, sans-serif; fill: {COL['ink']}; }}
      .mini {{ font: 400 12px Arial, Helvetica, sans-serif; fill: {COL['muted']}; }}
      .tiny {{ font: 400 10px Arial, Helvetica, sans-serif; fill: {COL['muted']}; }}
      .math {{ font: 600 13px Arial, Helvetica, sans-serif; fill: {COL['ink']}; }}
      .arrow {{ stroke: {COL['line']}; stroke-width: 2.1; fill: none; marker-end: url(#arrow); }}
      .thin {{ stroke: {COL['soft_line']}; stroke-width: 1.3; fill: none; }}
      .dash {{ stroke-dasharray: 8 7; }}
      .feedback {{ stroke: {COL['peach3']}; stroke-width: 2; stroke-dasharray: 8 7; fill: none; marker-end: url(#arrowWarm); }}
    </style>
  </defs>
'''


def rect(x, y, w, h, fill, stroke=None, rx=8, sw=1.2, extra="") -> str:
    stroke = stroke or COL["soft_line"]
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}" {extra}/>'
    )


def circle(x, y, r, fill, stroke=None, sw=1.2) -> str:
    stroke = stroke or COL["soft_line"]
    return f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'


def text(x, y, s, cls="mini", anchor="start") -> str:
    return f'<text class="{cls}" x="{x}" y="{y}" text-anchor="{anchor}">{escape(s)}</text>'


def line(x1, y1, x2, y2, cls="arrow") -> str:
    return f'<path class="{cls}" d="M {x1} {y1} L {x2} {y2}"/>'


def path(d, cls="arrow") -> str:
    return f'<path class="{cls}" d="{d}"/>'


def panel(x, y, w, h, n, title, fill, accent) -> str:
    return "\n".join([
        rect(x, y, w, h, fill, "#afbdd0", rx=18, sw=1.5),
        circle(x + 28, y + 28, 17, accent, accent, sw=1),
        text(x + 28, y + 33, n, "tag", "middle"),
        text(x + 55, y + 34, title, "panel"),
    ])


def feed_card(x, y, name, color) -> str:
    return "\n".join([
        rect(x, y, 74, 50, COL["white"], color, rx=8, sw=1.4),
        f'<path d="M {x+17} {y+13} H {x+57} M {x+17} {y+24} H {x+52} M {x+17} {y+35} H {x+44}" stroke="{color}" stroke-width="2"/>',
        text(x + 37, y + 68, name, "tiny", "middle"),
    ])


def database(x, y, w, h, color, label) -> str:
    return "\n".join([
        f'<ellipse cx="{x+w/2}" cy="{y+12}" rx="{w/2}" ry="12" fill="{COL["white"]}" stroke="{color}" stroke-width="1.5"/>',
        f'<path d="M {x} {y+12} V {y+h-12} C {x} {y+h+4}, {x+w} {y+h+4}, {x+w} {y+h-12} V {y+12}" fill="{COL["white"]}" stroke="{color}" stroke-width="1.5"/>',
        f'<path d="M {x} {y+42} C {x} {y+58}, {x+w} {y+58}, {x+w} {y+42}" fill="none" stroke="{color}" stroke-width="1.2"/>',
        text(x + w / 2, y + h + 22, label, "tiny", "middle"),
    ])


def mini_table(x, y, w, h, color, title) -> str:
    rows = []
    rows.append(rect(x, y, w, h, COL["white"], color, rx=6, sw=1.2))
    rows.append(rect(x, y, w, 22, color, color, rx=6, sw=0))
    rows.append(text(x + 10, y + 16, title, "tiny"))
    for i in range(1, 5):
        rows.append(f'<path d="M {x} {y+22+i*(h-22)/5} H {x+w}" stroke="{color}" stroke-width="0.7" opacity="0.55"/>')
    for j in range(1, 4):
        rows.append(f'<path d="M {x+j*w/4} {y+22} V {y+h}" stroke="{color}" stroke-width="0.7" opacity="0.55"/>')
    return "\n".join(rows)


def rail_graph(x, y, scale=1.0, accent=COL["green3"]) -> str:
    pts = [
        (x + 10*scale, y + 42*scale),
        (x + 55*scale, y + 18*scale),
        (x + 105*scale, y + 36*scale),
        (x + 152*scale, y + 14*scale),
        (x + 58*scale, y + 74*scale),
        (x + 118*scale, y + 78*scale),
        (x + 174*scale, y + 58*scale),
    ]
    edges = [(0, 1), (1, 2), (2, 3), (1, 4), (2, 5), (5, 6), (4, 5), (2, 6)]
    out = []
    for a, b in edges:
        out.append(f'<path d="M {pts[a][0]} {pts[a][1]} L {pts[b][0]} {pts[b][1]}" stroke="{accent}" stroke-width="{2*scale}" fill="none" opacity="0.8"/>')
    fills = [COL["blue2"], COL["green2"], COL["yellow2"], COL["peach2"], COL["violet2"], COL["cyan2"], COL["rose2"]]
    for i, (px, py) in enumerate(pts):
        out.append(circle(px, py, 8*scale, fills[i % len(fills)], accent, sw=1.2*scale))
    return "\n".join(out)


def tensor_stack(x, y, color) -> str:
    out = []
    for i in range(5):
        out.append(rect(x + i*7, y - i*5, 60, 52, "#ffffff", color, rx=3, sw=1))
        for r in range(3):
            for c in range(4):
                shade = ["#f4f7fb", "#d9e7fb", "#b8d4f6", "#8db6e8"][(r+c+i) % 4]
                out.append(rect(x + i*7 + 8 + c*10, y - i*5 + 8 + r*11, 8, 8, shade, "none", rx=0, sw=0))
    return "\n".join(out)


def transformer_blocks(x, y, color) -> str:
    out = []
    for i in range(4):
        out.append(rect(x + i*38, y, 26, 92, "#ffffff", color, rx=8, sw=1.4))
        out.append(text(x + i*38 + 13, y + 51, "T", "label", "middle"))
    out.append(text(x + 72, y + 112, "Transformer", "tiny", "middle"))
    return "\n".join(out)


def heatmap(x, y, w=90, h=72, color=COL["violet3"]) -> str:
    out = [rect(x, y, w, h, COL["white"], color, rx=6, sw=1.1)]
    cols = ["#f4efff", "#ddd1ff", "#bba7f1", "#9274de", "#7055b8"]
    cell_w, cell_h = w / 6, h / 5
    for r in range(5):
        for c in range(6):
            out.append(rect(x + c*cell_w + 2, y + r*cell_h + 2, cell_w - 4, cell_h - 4, cols[(r*2 + c) % len(cols)], "none", rx=1, sw=0))
    return "\n".join(out)


def bar_chart(x, y, color=COL["blue3"]) -> str:
    vals = [45, 75, 58, 32]
    labs = ["W", "R1", "R2", "R3"]
    out = [f'<path d="M {x} {y+80} H {x+145}" stroke="{COL["soft_line"]}" stroke-width="1.2"/>']
    for i, v in enumerate(vals):
        out.append(rect(x + 14 + i*31, y + 80 - v, 18, v, color, "none", rx=2, sw=0))
        out.append(text(x + 23 + i*31, y + 96, labs[i], "tiny", "middle"))
    out.append(text(x + 72, y + 15, "Q scores", "tiny", "middle"))
    return "\n".join(out)


def timeline(x, y, color=COL["peach3"]) -> str:
    out = [f'<path d="M {x} {y} H {x+170}" stroke="{color}" stroke-width="3"/>']
    for i, lab in enumerate(["s", "a", "r", "s'"]):
        px = x + i*52
        out.append(circle(px, y, 10, COL["white"], color, sw=2))
        out.append(text(px, y + 30, lab, "tiny", "middle"))
    out.append(path(f"M {x+160} {y} L {x+190} {y}", "arrow"))
    return "\n".join(out)


def app_collector(x, y) -> str:
    out = [
        rect(x, y, 190, 130, COL["white"], COL["blue3"], rx=12, sw=1.5),
        rect(x, y, 190, 26, COL["blue2"], COL["blue3"], rx=12, sw=0),
        text(x + 18, y + 19, "collector", "label"),
    ]
    for i, lab in enumerate(["TD", "TM", "SCH"]):
        yy = y + 48 + i*24
        out.append(rect(x + 18, yy - 12, 14, 14, "#ffffff", COL["blue3"], rx=2, sw=1))
        out.append(f'<path d="M {x+21} {yy-5} L {x+25} {yy} L {x+31} {yy-10}" stroke="{COL["blue3"]}" stroke-width="1.8" fill="none"/>')
        out.append(text(x + 42, yy, lab, "mini"))
        out.append(rect(x + 84, yy - 13, 80, 10, "#edf4ff", "none", rx=5, sw=0))
    out.append(f'<path d="M {x+154} {y+95} C {x+194} {y+70}, {x+194} {y+117}, {x+158} {y+112}" stroke="{COL["blue3"]}" stroke-width="1.6" fill="none" marker-end="url(#arrow)"/>')
    return "\n".join(out)


def mdp_state_icon(x, y) -> str:
    out = [rect(x, y, 180, 130, COL["white"], COL["peach3"], rx=12, sw=1.4)]
    out.append(rail_graph(x + 16, y + 20, scale=0.75, accent=COL["peach3"]))
    for i in range(16):
        out.append(rect(x + 18 + i*8, y + 98, 5, 18 - (i % 5)*2, [COL["blue3"], COL["green3"], COL["peach3"], COL["violet3"]][i % 4], "none", rx=1, sw=0))
    out.append(text(x + 90, y + 120, "state snapshot", "tiny", "middle"))
    return "\n".join(out)


def action_set(x, y) -> str:
    out = [rect(x, y, 180, 92, COL["white"], COL["peach3"], rx=12, sw=1.4)]
    out.append(text(x + 90, y + 22, "A_t", "math", "middle"))
    labels = ["wait", "R1", "R2", "R3"]
    for i, lab in enumerate(labels):
        out.append(rect(x + 16 + i*39, y + 42, 30, 28, [COL["yellow"], COL["blue"], COL["green"], COL["violet"]][i], COL["peach3"], rx=6, sw=1))
        out.append(text(x + 31 + i*39, y + 60, lab, "tiny", "middle"))
    return "\n".join(out)


def eval_dashboard(x, y) -> str:
    out = [rect(x, y, 185, 128, COL["white"], COL["cyan3"], rx=12, sw=1.4)]
    out.append(text(x + 20, y + 24, "metrics", "label"))
    for i in range(3):
        out.append(rect(x + 20, y + 40 + i*24, 60 + i*26, 12, [COL["cyan2"], COL["green2"], COL["yellow2"]][i], "none", rx=6, sw=0))
    for r in range(2):
        for c in range(3):
            out.append(rect(x + 112 + c*18, y + 39 + r*24, 12, 12, [COL["blue2"], COL["violet2"], COL["rose2"], COL["green2"]][(r+c) % 4], COL["white"], rx=2, sw=0.4))
    out.append(text(x + 92, y + 112, "overall + strata", "tiny", "middle"))
    return "\n".join(out)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    s: list[str] = [header(), rect(0, 0, W, H, COL["paper"], "none", rx=0, sw=0)]

    s.append(text(62, 55, "RailRL: end-to-end framework from railway feeds to explainable route-setting decisions", "title"))
    s.append(text(64, 84, "Visual schematic for ESWA / IEEE T-ITS style manuscripts", "sub"))

    y0, ph, gap = 132, 770, 25
    widths = [360, 360, 400, 440, 405]
    xs = [60]
    for i in range(1, len(widths)):
        xs.append(xs[-1] + widths[i - 1] + gap)

    panel_specs = [
        ("1", "Data acquisition", COL["blue"], COL["blue3"]),
        ("2", "Canonical data layer", COL["green"], COL["green3"]),
        ("3", "MDP construction", COL["peach"], COL["peach3"]),
        ("4", "Offline RL core", COL["violet"], COL["violet3"]),
        ("5", "Evaluation & XAI", COL["cyan"], COL["cyan3"]),
    ]
    for i, spec in enumerate(panel_specs):
        s.append(panel(xs[i], y0, widths[i], ph, *spec))

    # Panel 1: acquisition graphics
    x = xs[0]
    s.append(feed_card(x + 32, y0 + 92, "TD", COL["blue3"]))
    s.append(feed_card(x + 128, y0 + 92, "TM", COL["green3"]))
    s.append(feed_card(x + 224, y0 + 92, "SCH", COL["violet3"]))
    s.append(path(f"M {x+180} {y0+192} C {x+180} {y0+226}, {x+180} {y0+226}, {x+180} {y0+248}", "arrow"))
    s.append(app_collector(x + 84, y0 + 246))
    s.append(line(x + 180, y0 + 376, x + 180, y0 + 430))
    s.append(database(x + 110, y0 + 430, 140, 112, COL["blue3"], "raw archive"))
    s.append(path(f"M {x+282} {y0+300} C {x+330} {y0+338}, {x+315} {y0+438}, {x+250} {y0+472}", "thin dash"))
    s.append(text(x + 273, y0 + 404, "resilience", "tiny", "middle"))
    for i in range(8):
        s.append(rect(x + 58 + i*30, y0 + 600 + (i % 2)*8, 18, 48 - (i % 3)*8, COL["blue2"], "none", rx=3, sw=0))
    s.append(text(x + 180, y0 + 690, "continuous collection", "label", "middle"))

    # Panel 2: data basis graphics
    x = xs[1]
    s.append(mini_table(x + 35, y0 + 90, 135, 110, COL["green3"], "events"))
    s.append(mini_table(x + 190, y0 + 90, 135, 110, COL["blue3"], "movements"))
    s.append(line(x + 170, y0 + 145, x + 190, y0 + 145))
    s.append(rail_graph(x + 78, y0 + 248, scale=1.0, accent=COL["green3"]))
    s.append(text(x + 180, y0 + 360, "typed infrastructure graph", "label", "middle"))
    s.append(database(x + 112, y0 + 400, 136, 104, COL["green3"], "canonical store"))
    s.append(path(f"M {x+180} {y0+200} C {x+180} {y0+230}, {x+180} {y0+230}, {x+180} {y0+248}", "arrow"))
    s.append(line(x + 180, y0 + 374, x + 180, y0 + 400))
    for i, lab in enumerate(["time", "identity", "coverage", "quality"]):
        bx = x + 38 + i*77
        s.append(rect(bx, y0 + 604, 58, 58, COL["white"], COL["green3"], rx=12, sw=1.2))
        s.append(circle(bx + 29, y0 + 626, 9, [COL["blue2"], COL["green2"], COL["yellow2"], COL["rose2"]][i], COL["green3"], sw=1))
        s.append(text(bx + 29, y0 + 681, lab, "tiny", "middle"))
    s.append(text(x + 180, y0 + 714, "provenance + checks", "label", "middle"))

    # Panel 3: MDP construction
    x = xs[2]
    s.append(rect(x + 34, y0 + 88, 330, 88, COL["white"], COL["peach3"], rx=12, sw=1.3))
    s.append(text(x + 62, y0 + 122, "SET", "label"))
    s.append(text(x + 62, y0 + 152, "WAIT", "label"))
    s.append(f'<path d="M {x+120} {y0+118} H {x+322}" stroke="{COL["peach3"]}" stroke-width="3"/>')
    s.append(f'<path d="M {x+120} {y0+148} H {x+250}" stroke="{COL["soft_line"]}" stroke-width="3"/>')
    s.append(circle(x + 324, y0 + 118, 7, COL["peach2"], COL["peach3"], sw=1.4))
    s.append(text(x + 199, y0 + 72, "decision triggers", "label", "middle"))
    s.append(line(x + 199, y0 + 176, x + 199, y0 + 220))
    s.append(action_set(x + 109, y0 + 220))
    s.append(line(x + 199, y0 + 312, x + 199, y0 + 354))
    s.append(mdp_state_icon(x + 109, y0 + 354))
    s.append(line(x + 199, y0 + 484, x + 199, y0 + 526))
    s.append(timeline(x + 70, y0 + 556, COL["peach3"]))
    s.append(text(x + 199, y0 + 656, "reward + transition", "label", "middle"))
    for i, c in enumerate([COL["rose2"], COL["green2"], COL["yellow2"], COL["blue2"]]):
        s.append(rect(x + 70 + i*61, y0 + 690, 42, 24, c, COL["peach3"], rx=5, sw=0.8))
    s.append(text(x + 199, y0 + 738, "delay | flow | headway | wait", "tiny", "middle"))

    # Panel 4: offline RL core
    x = xs[3]
    branch_y = [y0 + 96, y0 + 278, y0 + 460]
    branch_lab = ["HGT graph encoder", "Event Transformer", "Context encoder"]
    branch_col = [COL["green3"], COL["blue3"], COL["violet3"]]
    for by, lab, c in zip(branch_y, branch_lab, branch_col):
        s.append(rect(x + 30, by, 245, 132, COL["white"], c, rx=14, sw=1.3))
        s.append(text(x + 52, by + 28, lab, "label"))
    s.append(rail_graph(x + 72, branch_y[0] + 45, scale=0.8, accent=COL["green3"]))
    s.append(tensor_stack(x + 78, branch_y[1] + 58, COL["blue3"]))
    s.append(transformer_blocks(x + 160, branch_y[1] + 36, COL["blue3"]))
    for i in range(5):
        s.append(rect(x + 72 + i*32, branch_y[2] + 56, 22, 46, [COL["violet2"], COL["yellow2"], COL["cyan2"]][i % 3], COL["violet3"], rx=4, sw=0.8))
    s.append(text(x + 154, branch_y[2] + 116, "schedule + flags", "tiny", "middle"))
    # fusion and heads
    s.append(rect(x + 316, y0 + 294, 92, 142, COL["yellow"], COL["yellow3"], rx=18, sw=1.5))
    s.append(text(x + 362, y0 + 333, "gated", "label", "middle"))
    s.append(text(x + 362, y0 + 353, "fusion", "label", "middle"))
    s.append(circle(x + 362, y0 + 392, 21, COL["white"], COL["yellow3"], sw=1.4))
    s.append(text(x + 362, y0 + 398, "+", "panel", "middle"))
    for by in branch_y:
        s.append(path(f"M {x+275} {by+66} C {x+298} {by+66}, {x+300} {y0+365}, {x+316} {y0+365}", "arrow"))
    s.append(line(x + 408, y0 + 365, x + 455, y0 + 365))
    s.append(rect(x + 455, y0 + 245, 150, 250, COL["white"], COL["violet3"], rx=16, sw=1.4))
    s.append(text(x + 530, y0 + 275, "CQL policy", "label", "middle"))
    s.append(bar_chart(x + 462, y0 + 305, COL["violet3"]))
    s.append(rect(x + 478, y0 + 425, 104, 22, COL["green"], COL["green3"], rx=6, sw=1))
    s.append(text(x + 530, y0 + 441, "route head", "tiny", "middle"))
    s.append(rect(x + 478, y0 + 457, 104, 22, COL["blue"], COL["blue3"], rx=6, sw=1))
    s.append(text(x + 530, y0 + 473, "time head", "tiny", "middle"))
    s.append(path(f"M {x+530} {y0+495} C {x+530} {y0+610}, {x+245} {y0+610}, {x+245} {y0+536}", "feedback"))
    s.append(text(x + 365, y0 + 628, "loss feedback", "tiny", "middle"))

    # Panel 5: evaluation and XAI
    x = xs[4]
    s.append(eval_dashboard(x + 34, y0 + 92))
    s.append(heatmap(x + 262, y0 + 100, 96, 78, COL["violet3"]))
    s.append(text(x + 310, y0 + 196, "attention / attribution", "tiny", "middle"))
    s.append(line(x + 220, y0 + 156, x + 262, y0 + 156))
    s.append(rect(x + 42, y0 + 270, 136, 116, COL["white"], COL["cyan3"], rx=12, sw=1.3))
    for r in range(3):
        for c in range(3):
            fill = [COL["cyan2"], COL["green2"], COL["yellow2"], COL["rose2"]][(r + c) % 4]
            s.append(rect(x + 62 + c*34, y0 + 300 + r*25, 24, 16, fill, "none", rx=3, sw=0))
    s.append(text(x + 110, y0 + 410, "strata", "label", "middle"))
    s.append(rect(x + 220, y0 + 270, 140, 116, COL["white"], COL["cyan3"], rx=12, sw=1.3))
    s.append(f'<path d="M {x+245} {y0+342} C {x+270} {y0+308}, {x+316} {y0+312}, {x+337} {y0+292}" stroke="{COL["cyan3"]}" stroke-width="2.2" fill="none"/>')
    s.append(f'<path d="M {x+245} {y0+362} C {x+278} {y0+348}, {x+310} {y0+374}, {x+340} {y0+336}" stroke="{COL["peach3"]}" stroke-width="2.2" fill="none"/>')
    s.append(text(x + 290, y0 + 410, "counterfactual", "label", "middle"))
    s.append(rect(x + 58, y0 + 485, 298, 132, COL["white"], COL["cyan3"], rx=14, sw=1.3))
    s.append(text(x + 207, y0 + 515, "decision-support evidence", "label", "middle"))
    s.append(bar_chart(x + 80, y0 + 536, COL["cyan3"]))
    s.append(heatmap(x + 235, y0 + 535, 82, 64, COL["cyan3"]))
    s.append(rect(x + 70, y0 + 660, 274, 42, COL["yellow"], COL["yellow3"], rx=10, sw=1.2))
    s.append(text(x + 207, y0 + 687, "ESWA / T-ITS evaluation package", "label", "middle"))

    # Cross-panel arrows
    for i in range(4):
        x1 = xs[i] + widths[i]
        x2 = xs[i + 1]
        mid = y0 + 385
        s.append(path(f"M {x1+4} {mid} C {x1+17} {mid}, {x2-17} {mid}, {x2-4} {mid}", "arrow"))

    # Bottom visual evidence rail
    ry = 940
    s.append(rect(60, ry, W - 120, 165, COL["rail"], "#e0b896", rx=20, sw=1.3))
    s.append(text(90, ry + 36, "Cross-cutting controls", "panel"))
    controls = [
        ("trace", COL["blue3"]),
        ("causal time", COL["green3"]),
        ("leak audit", COL["rose3"]),
        ("3 seeds", COL["violet3"]),
        ("confidential", COL["peach3"]),
    ]
    for i, (lab, c) in enumerate(controls):
        bx = 120 + i * 380
        s.append(rect(bx, ry + 62, 250, 74, "#fffaf5", "#d3a783", rx=14, sw=1.1))
        # simple icon
        s.append(circle(bx + 38, ry + 99, 20, "#ffffff", c, sw=2))
        if lab == "trace":
            s.append(f'<path d="M {bx+27} {ry+99} H {bx+49} M {bx+38} {ry+88} V {ry+110}" stroke="{c}" stroke-width="2"/>')
        elif lab == "causal time":
            s.append(f'<path d="M {bx+26} {ry+103} C {bx+36} {ry+85}, {bx+50} {ry+90}, {bx+51} {ry+103}" stroke="{c}" stroke-width="2" fill="none"/>')
        elif lab == "leak audit":
            s.append(f'<path d="M {bx+28} {ry+99} L {bx+36} {ry+107} L {bx+50} {ry+89}" stroke="{c}" stroke-width="2.6" fill="none"/>')
        elif lab == "3 seeds":
            for k in range(3):
                s.append(circle(bx + 31 + k*8, ry + 99, 4, c, c, sw=1))
        else:
            s.append(f'<path d="M {bx+28} {ry+96} H {bx+48} V {ry+110} H {bx+28} Z M {bx+33} {ry+96} V {ry+90} C {bx+33} {ry+80}, {bx+43} {ry+80}, {bx+43} {ry+90} V {ry+96}" stroke="{c}" stroke-width="2" fill="none"/>')
        s.append(text(bx + 80, ry + 94, lab, "label"))
        s.append(text(bx + 80, ry + 116, ["sample lineage", "no future state", "no answer leakage", "robustness", "public-safe view"][i], "tiny"))

    s.append(text(70, 1145, "Scale: 14 months | 11.8M TD records | 247k movements | about 2.0M decisions | 1.996M usable MDP snapshots", "mini"))
    s.append(text(W - 70, 1145, "Editable SVG: all panels are shapes and text", "mini", "end"))

    s.append("</svg>\n")
    SVG.write_text("\n".join(s), encoding="utf-8")
    print(SVG)


if __name__ == "__main__":
    main()
