"""Reference-style RailRL system / architecture figure.

This figure follows the visual grammar of common engineering / AI manuscript
architecture figures: large pastel panels, explicit process arrows, compact
module diagrams, a bottom validation rail, and a small legend. It is generated
locally as vector-style matplotlib output so that all labels remain controlled.

Public figure rule:
    The diagram shows only manuscript-facing system components. Protected
    operational internals are not represented as modules or labels.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "figures" / "system_architecture_reference_style" / "output"
OUT.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7.0,
})

COL = {
    "ink": "#1F2933",
    "muted": "#64748B",
    "line": "#AAB7C8",
    "blue": "#0F4D92",
    "teal": "#227C70",
    "gold": "#B7791F",
    "violet": "#6656A4",
    "red": "#B0444C",
    "green": "#27834F",
    "orange": "#E07025",
    "blue_bg": "#D9E9FC",
    "teal_bg": "#DEEEEC",
    "gold_bg": "#FFF4C7",
    "violet_bg": "#E9DFF7",
    "red_bg": "#F9E5E4",
    "green_bg": "#E1EFD9",
    "peach_bg": "#F8CAAC",
    "soft": "#F8FAFC",
    "white": "#FFFFFF",
}


def rounded(ax, x, y, w, h, fc, ec="#CBD5E1", r=0.018, lw=0.9, ls="-"):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.008,rounding_size={r}",
        linewidth=lw, linestyle=ls, edgecolor=ec, facecolor=fc,
    )
    ax.add_patch(patch)
    return patch


def panel(ax, x, y, w, h, number, title, fc, color):
    rounded(ax, x, y, w, h, fc, ec="#CBD5E1", r=0.020, lw=0.9)
    ax.text(x + 0.014, y + h - 0.035, f"{number}  {title}",
            ha="left", va="top", fontsize=7.8, fontweight="bold", color=color)


def label(ax, x, y, text, size=5.8, color=None, weight="normal", ha="left", va="center"):
    ax.text(x, y, text, fontsize=size, color=color or COL["ink"],
            fontweight=weight, ha=ha, va=va)


def arrow(ax, start, end, color="#475569", lw=1.1, ms=9, style="-|>", ls="-", rad=0):
    arr = FancyArrowPatch(
        start, end, arrowstyle=style, mutation_scale=ms, linewidth=lw,
        color=color, linestyle=ls, shrinkA=2.5, shrinkB=2.5,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arr)
    return arr


def big_arrow(ax, x0, y, x1, text):
    arrow(ax, (x0, y), (x1, y), color="#93A4BA", lw=1.35, ms=15)
    label(ax, (x0 + x1) / 2, y + 0.033, text, size=5.4, color=COL["muted"],
          weight="bold", ha="center")


def draw_feed_icon(ax, x, y, w, h):
    rounded(ax, x, y, w, h, COL["white"], "#B8C6D8", r=0.010)
    for i, color in enumerate([COL["blue"], COL["teal"], COL["gold"], COL["violet"]]):
        yy = y + h - 0.028 - i * 0.025
        ax.add_line(Line2D([x + 0.018, x + w - 0.018], [yy, yy], color=color, lw=1.3, alpha=0.75))
        ax.add_patch(Circle((x + 0.025 + 0.025 * i, yy), 0.0045, facecolor=color, edgecolor="none"))


def draw_database(ax, x, y, w, h, color):
    for i in range(3):
        rounded(ax, x + i * 0.012, y + i * 0.010, w, h, "#FFFFFF", color, r=0.008, lw=0.8)
        for j in range(4):
            ax.add_line(Line2D([x + i * 0.012 + 0.012, x + i * 0.012 + w - 0.012],
                               [y + i * 0.010 + 0.018 + j * 0.017] * 2,
                               color=color, lw=0.8, alpha=0.55))


def draw_grid(ax, x, y, w, h, rows=5, cols=6, colors=None):
    colors = colors or ["#D9E9FC", "#DEEEEC", "#FFF4C7", "#E9DFF7"]
    cw, rh = w / cols, h / rows
    for r in range(rows):
        for c in range(cols):
            ax.add_patch(Rectangle(
                (x + c * cw, y + r * rh), cw * 0.92, rh * 0.85,
                facecolor=colors[(r + c) % len(colors)], edgecolor="#7B8794", linewidth=0.35,
            ))


def draw_graph(ax, x, y, scale=1.0):
    pts = [
        (x, y), (x + 0.030 * scale, y + 0.035 * scale),
        (x + 0.065 * scale, y + 0.020 * scale),
        (x + 0.050 * scale, y - 0.030 * scale),
        (x + 0.010 * scale, y - 0.035 * scale),
    ]
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (1, 3)]
    for a, b in edges:
        ax.add_line(Line2D([pts[a][0], pts[b][0]], [pts[a][1], pts[b][1]],
                           color=COL["blue"], lw=0.8, alpha=0.8))
    for px, py in pts:
        ax.add_patch(Circle((px, py), 0.008 * scale, facecolor="#9FC3E6",
                            edgecolor=COL["blue"], linewidth=0.7))


def draw_mini_bars(ax, x, y, w, h):
    vals = [0.45, 0.72, 0.60, 0.85]
    for i, v in enumerate(vals):
        bx = x + i * w / 5 + 0.010
        bh = h * v
        ax.add_patch(Rectangle((bx, y), w / 7, bh, facecolor="#8B6BCB",
                               edgecolor="#6A54A3", linewidth=0.45))
    ax.add_line(Line2D([x, x + w], [y, y], color="#7B8794", lw=0.6))
    ax.add_line(Line2D([x, x], [y, y + h], color="#7B8794", lw=0.6))


def draw_attention_map(ax, x, y, w, h):
    vals = [
        [0.20, 0.35, 0.55, 0.30],
        [0.28, 0.65, 0.88, 0.42],
        [0.18, 0.36, 0.72, 0.58],
        [0.12, 0.25, 0.42, 0.33],
    ]
    rows, cols = 4, 4
    cw, rh = w / cols, h / rows
    for r in range(rows):
        for c in range(cols):
            v = vals[r][c]
            color = (0.20 + 0.45 * (1 - v), 0.36 + 0.22 * (1 - v), 0.74, 1)
            ax.add_patch(Rectangle((x + c * cw, y + r * rh), cw * 0.94, rh * 0.90,
                                   facecolor=color, edgecolor="#FFFFFF", linewidth=0.25))
    rounded(ax, x - 0.002, y - 0.002, w + 0.004, h + 0.004, "none", "#AAB7C8", r=0.003, lw=0.55)


def draw_legend(ax, x, y):
    rounded(ax, x, y, 0.150, 0.108, COL["white"], "#CBD5E1", r=0.012, lw=0.8, ls=(0, (4, 3)))
    items = [
        (COL["blue"], "data flow", "-"),
        (COL["orange"], "training feedback", (0, (3, 2))),
        (COL["violet"], "explainability", (0, (3, 2))),
    ]
    for i, (color, text, ls) in enumerate(items):
        yy = y + 0.082 - i * 0.030
        arrow(ax, (x + 0.017, yy), (x + 0.050, yy), color=color, lw=0.9, ms=8, ls=ls)
        label(ax, x + 0.058, yy, text, size=5.2, color=COL["ink"])


def draw_architecture(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    label(ax, 0.035, 0.955, "RailRL System / Architecture", size=13.5,
          color=COL["ink"], weight="bold")
    label(ax, 0.037, 0.918,
          "Traceable railway data acquisition -> leak-safe decision reconstruction -> conservative offline RL -> explainable route-setting support",
          size=6.6, color=COL["muted"])

    rounded(ax, 0.800, 0.932, 0.075, 0.033, COL["violet_bg"], ec="none", r=0.016)
    rounded(ax, 0.884, 0.932, 0.075, 0.033, COL["red_bg"], ec="none", r=0.016)
    label(ax, 0.837, 0.949, "ESWA", size=6.1, color=COL["violet"], weight="bold", ha="center")
    label(ax, 0.922, 0.949, "T-ITS", size=6.1, color=COL["red"], weight="bold", ha="center")

    y, h, w = 0.410, 0.410, 0.210
    xs = [0.035, 0.275, 0.515, 0.755]

    panel(ax, xs[0], y, w, h, "1", "Data Layer", COL["blue_bg"], COL["blue"])
    panel(ax, xs[1], y, w, h, "2", "Decision Layer", COL["green_bg"], COL["green"])
    panel(ax, xs[2], y, w, h, "3", "Learning Layer", COL["violet_bg"], COL["violet"])
    panel(ax, xs[3], y, w, h, "4", "Evaluation + XAI", COL["red_bg"], COL["red"])

    # Panel 1: operational acquisition.
    draw_feed_icon(ax, xs[0] + 0.020, y + 0.235, 0.075, 0.105)
    label(ax, xs[0] + 0.107, y + 0.315, "Operational feeds", size=6.0, weight="bold")
    for i, txt in enumerate(["TD events", "movement records", "schedule/performance", "infrastructure graph"]):
        label(ax, xs[0] + 0.107, y + 0.286 - i * 0.031, txt, size=5.35)
    arrow(ax, (xs[0] + 0.060, y + 0.225), (xs[0] + 0.060, y + 0.168), color=COL["blue"])
    draw_database(ax, xs[0] + 0.028, y + 0.085, 0.075, 0.065, COL["blue"])
    label(ax, xs[0] + 0.120, y + 0.139, "Provenance archive", size=5.8, weight="bold")
    label(ax, xs[0] + 0.120, y + 0.108, "versioned records", size=5.2, color=COL["muted"])

    # Panel 2: decision dataset.
    label(ax, xs[1] + 0.020, y + 0.315, "Canonical decision samples", size=5.9, weight="bold")
    draw_grid(ax, xs[1] + 0.024, y + 0.225, 0.085, 0.070, rows=4, cols=6)
    rounded(ax, xs[1] + 0.127, y + 0.235, 0.055, 0.052, COL["white"], "#B8C6D8", r=0.010)
    label(ax, xs[1] + 0.154, y + 0.268, "wait", size=5.0, ha="center")
    label(ax, xs[1] + 0.154, y + 0.248, "route_i", size=5.0, ha="center")
    arrow(ax, (xs[1] + 0.110, y + 0.260), (xs[1] + 0.127, y + 0.260), color=COL["green"])
    rounded(ax, xs[1] + 0.024, y + 0.120, 0.070, 0.057, "#FFFFFF", "#B8C6D8", r=0.010)
    label(ax, xs[1] + 0.059, y + 0.151, "reward", size=5.4, weight="bold", ha="center", color=COL["gold"])
    label(ax, xs[1] + 0.059, y + 0.132, "delay / risk", size=4.8, ha="center")
    rounded(ax, xs[1] + 0.112, y + 0.120, 0.075, 0.057, "#FFFFFF", "#B8C6D8", r=0.010)
    label(ax, xs[1] + 0.149, y + 0.151, "state", size=5.4, weight="bold", ha="center", color=COL["green"])
    label(ax, xs[1] + 0.149, y + 0.132, "leak-safe", size=4.8, ha="center")

    # Panel 3: model.
    draw_graph(ax, xs[2] + 0.037, y + 0.282, scale=1.0)
    label(ax, xs[2] + 0.103, y + 0.300, "graph encoder", size=5.4, weight="bold")
    draw_grid(ax, xs[2] + 0.032, y + 0.178, 0.078, 0.045, rows=3, cols=5,
              colors=["#D9E9FC", "#E9DFF7", "#FFF4C7"])
    label(ax, xs[2] + 0.126, y + 0.205, "event encoder", size=5.4, weight="bold")
    arrow(ax, (xs[2] + 0.095, y + 0.270), (xs[2] + 0.143, y + 0.245), color=COL["violet"])
    arrow(ax, (xs[2] + 0.107, y + 0.198), (xs[2] + 0.143, y + 0.225), color=COL["violet"])
    rounded(ax, xs[2] + 0.132, y + 0.218, 0.055, 0.055, COL["white"], "#AFA3D5", r=0.012)
    label(ax, xs[2] + 0.159, y + 0.248, "fusion", size=5.2, weight="bold", ha="center")
    label(ax, xs[2] + 0.159, y + 0.228, "+ mask", size=4.8, ha="center")
    rounded(ax, xs[2] + 0.047, y + 0.085, 0.125, 0.056, "#FFFFFF", "#AFA3D5", r=0.012)
    label(ax, xs[2] + 0.109, y + 0.119, "CQL per-action Q", size=5.5, weight="bold", ha="center")
    label(ax, xs[2] + 0.109, y + 0.099, "Q(wait), Q(route_i)", size=4.8, ha="center")
    arrow(ax, (xs[2] + 0.160, y + 0.218), (xs[2] + 0.110, y + 0.143), color=COL["violet"])

    # Panel 4: evaluation and outputs.
    rounded(ax, xs[3] + 0.020, y + 0.220, 0.074, 0.105, COL["white"], "#D9B6BB", r=0.010)
    label(ax, xs[3] + 0.057, y + 0.304, "metrics", size=5.3, weight="bold", ha="center", color=COL["red"])
    draw_mini_bars(ax, xs[3] + 0.038, y + 0.238, 0.040, 0.043)
    rounded(ax, xs[3] + 0.114, y + 0.220, 0.074, 0.105, COL["white"], "#D9B6BB", r=0.010)
    label(ax, xs[3] + 0.151, y + 0.304, "explanations", size=5.3, weight="bold", ha="center", color=COL["violet"])
    draw_attention_map(ax, xs[3] + 0.135, y + 0.243, 0.033, 0.040)
    rounded(ax, xs[3] + 0.034, y + 0.088, 0.145, 0.070, "#FFFFFF", "#D9B6BB", r=0.012)
    label(ax, xs[3] + 0.106, y + 0.136, "Decision-support output", size=5.6,
          weight="bold", color=COL["blue"], ha="center")
    label(ax, xs[3] + 0.106, y + 0.112, "ranked wait/set actions", size=5.0, ha="center")

    big_arrow(ax, xs[0] + w + 0.003, y + 0.235, xs[1] - 0.006, "canonicalize")
    big_arrow(ax, xs[1] + w + 0.003, y + 0.235, xs[2] - 0.006, "train")
    big_arrow(ax, xs[2] + w + 0.003, y + 0.235, xs[3] - 0.006, "validate")

    # Bottom evidence band.
    band_y, band_h = 0.140, 0.190
    rounded(ax, 0.035, band_y, 0.930, band_h, "#FFF8ED", "#F0C7A4", r=0.020, lw=0.9)
    label(ax, 0.055, band_y + band_h - 0.040, "Validation and publication-evidence rail",
          size=7.0, color=COL["orange"], weight="bold")
    label(ax, 0.055, band_y + 0.130,
          "evidence must separate behaviour replication from operational-improvement claims",
          size=5.4, color=COL["muted"])
    rail_y = band_y + 0.070
    ax.add_line(Line2D([0.090, 0.925], [rail_y, rail_y], color="#C7A484", lw=1.2))
    checkpoints = [
        (0.105, "temporal split"),
        (0.245, "schema tests"),
        (0.385, "leak audit"),
        (0.525, "training gates"),
        (0.670, "baselines"),
        (0.815, "counterfactuals"),
        (0.925, "paper figures"),
    ]
    for x, text in checkpoints:
        ax.add_patch(Circle((x, rail_y), 0.010, facecolor="#FFFFFF",
                            edgecolor=COL["green"], linewidth=0.9))
        label(ax, x, rail_y - 0.036, text, size=5.2, weight="bold", ha="center")
    arrow(ax, (0.055, band_y + 0.025), (0.935, band_y + 0.025),
          color=COL["orange"], lw=0.9, ms=9, ls=(0, (3, 2)))

    draw_legend(ax, 0.055, 0.815)
    label(ax, 0.035, 0.060,
          "Figure 1. Reference-style system architecture of RailRL for railway route-setting decision support.",
          size=5.8, color=COL["muted"])


def main():
    width_mm, height_mm = 183, 128
    fig = plt.figure(figsize=(width_mm / 25.4, height_mm / 25.4), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    draw_architecture(ax)
    stem = OUT / "railrl_system_architecture_reference_style"
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(stem)


if __name__ == "__main__":
    main()
