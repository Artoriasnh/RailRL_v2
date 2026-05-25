"""RailRL system architecture figure.

Engineering-figure-agent brief
------------------------------
Goal:
    Draw a manuscript-ready System / Architecture diagram for the RailRL paper.
Claim:
    RailRL links operational inputs, traceable acquisition, canonical research
    storage, leak-safe MDP reconstruction, conservative offline RL, and
    evaluation/explanation outputs for railway route-setting decision support.
Mode:
    Conceptual image mode, implemented as a local vector-style matplotlib
    schematic to keep labels reproducible and reusable.
Public-disclosure rule:
    Show only public-facing system components. Protected operational details
    are intentionally absent from visible labels and figure text.
"""
from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle, Circle


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "figures" / "system_architecture" / "output"
OUT.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7.4,
})

COL = {
    "ink": "#1F2933",
    "muted": "#667085",
    "line": "#B8C3D1",
    "blue": "#0F4D92",
    "teal": "#227C70",
    "gold": "#B7791F",
    "violet": "#6656A4",
    "red": "#B0444C",
    "green": "#27834F",
    "slate_bg": "#F7FAFC",
    "blue_bg": "#EAF1F8",
    "teal_bg": "#E7F4F0",
    "gold_bg": "#FFF3D8",
    "violet_bg": "#ECE8F7",
    "red_bg": "#F8E5E8",
    "green_bg": "#E7F3EA",
    "white": "#FFFFFF",
}


def wrap_lines(lines: list[str], width: int = 25) -> list[str]:
    out: list[str] = []
    for line in lines:
        out.extend(textwrap.wrap(line, width=width, break_long_words=False) or [""])
    return out


def rounded_box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    body: list[str],
    fc: str,
    ec: str,
    title_color: str,
    title_size: float = 7.0,
    body_size: float = 5.7,
    wrap: int = 24,
):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.010,rounding_size=0.022",
        linewidth=0.9,
        edgecolor=ec,
        facecolor=fc,
        mutation_aspect=1,
    )
    ax.add_patch(patch)
    ax.text(
        x + 0.016, y + h - 0.030, title,
        ha="left", va="top", fontsize=title_size, fontweight="bold",
        color=title_color,
    )
    yy = y + h - 0.070
    for line in wrap_lines(body, wrap):
        ax.text(x + 0.017, yy, line, ha="left", va="top",
                fontsize=body_size, color=COL["ink"])
        yy -= 0.032
    return patch


def arrow(ax, start, end, color=COL["muted"], lw=1.15, ms=10):
    arr = FancyArrowPatch(
        start, end,
        arrowstyle="-|>",
        mutation_scale=ms,
        linewidth=lw,
        color=color,
        shrinkA=4,
        shrinkB=4,
        connectionstyle="arc3,rad=0",
    )
    ax.add_patch(arr)
    return arr


def small_tag(ax, x: float, y: float, text: str, fc: str, color: str):
    patch = FancyBboxPatch(
        (x, y), 0.095, 0.034,
        boxstyle="round,pad=0.004,rounding_size=0.015",
        linewidth=0,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(x + 0.0475, y + 0.017, text, ha="center", va="center",
            fontsize=5.8, fontweight="bold", color=color)


def draw_system_architecture(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Outer manuscript/system boundary.
    boundary = FancyBboxPatch(
        (0.020, 0.075), 0.960, 0.815,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        linewidth=0.9,
        edgecolor="#D4DCE8",
        facecolor="#FBFCFE",
    )
    ax.add_patch(boundary)

    ax.text(0.042, 0.955, "RailRL System Architecture", ha="left", va="top",
            fontsize=13.5, fontweight="bold", color=COL["ink"])
    ax.text(
        0.044, 0.907,
        "From operational railway feeds to leak-audited offline RL and explainable route-setting decision support",
        ha="left", va="top", fontsize=6.9, color=COL["muted"],
    )
    small_tag(ax, 0.820, 0.925, "ESWA", "#ECE8F7", COL["violet"])
    small_tag(ax, 0.920, 0.925, "T-ITS", "#F8E5E8", COL["red"])

    # Main row: the technical system.
    y = 0.585
    h = 0.220
    w = 0.137
    xs = [0.045, 0.200, 0.355, 0.510, 0.665, 0.820]

    rounded_box(
        ax, xs[0], y, w, h, "Operational inputs",
        ["TD event stream", "movement records", "schedule and performance", "static infrastructure graph"],
        COL["blue_bg"], "#B1BDCB", COL["blue"], wrap=24,
    )
    rounded_box(
        ax, xs[1], y, w, h, "Acquisition service",
        ["feed listeners", "runtime monitoring", "timestamp alignment", "versioned raw archive"],
        COL["teal_bg"], "#A8CDBF", COL["teal"], wrap=24,
    )
    rounded_box(
        ax, xs[2], y, w, h, "Canonical store",
        ["normalized datasets", "provenance index", "reproducible splits", "schema contracts"],
        COL["slate_bg"], "#CBD5E1", COL["ink"], wrap=24,
    )
    rounded_box(
        ax, xs[3], y, w, h, "Decision dataset",
        ["route-setting points", "dynamic actions", "reward + episode builder", "leak-safe state"],
        COL["gold_bg"], "#E0BD75", COL["gold"], wrap=23,
    )
    rounded_box(
        ax, xs[4], y, w, h, "Offline RL core",
        ["graph encoder", "event encoder", "action mask", "CQL per-action Q"],
        COL["violet_bg"], "#B8ADD8", COL["violet"], wrap=23,
    )
    rounded_box(
        ax, xs[5], y, w, h, "Evaluation + XAI",
        ["imitation and baselines", "counterfactual tests", "multi-level explanations", "audit reports"],
        COL["red_bg"], "#E0B2B8", COL["red"], wrap=23,
    )

    for i in range(5):
        arrow(ax, (xs[i] + w, y + h * 0.56), (xs[i + 1], y + h * 0.56))

    # Decision-support interface beneath model/evaluation blocks.
    rounded_box(
        ax, 0.622, 0.305, 0.333, 0.190, "Decision-support interface",
        ["ranked feasible wait/set actions", "Q-gap, trade-off and uncertainty summaries", "operator-facing explanation package"],
        "#FFFFFF", "#CBD5E1", COL["ink"], title_size=7.2, body_size=5.8, wrap=44,
    )
    arrow(ax, (xs[4] + w * 0.75, y), (0.708, 0.495), color=COL["violet"], lw=1.1)
    arrow(ax, (xs[5] + w * 0.55, y), (0.842, 0.495), color=COL["red"], lw=1.1)

    # MDP contract callout under the dataset builder.
    rounded_box(
        ax, 0.195, 0.305, 0.342, 0.190, "MDP contract",
        ["transition tuple",
         r"($s_t$, $a_t$, $r_t$, $s_{t+1}$, done)",
         "training uses observed next states only",
         "counterfactual rollout: evaluation only"],
        "#FFFFFF", "#CBD5E1", COL["ink"], title_size=7.2, body_size=5.8, wrap=44,
    )
    arrow(ax, (xs[3] + w * 0.45, y), (0.368, 0.495), color=COL["gold"], lw=1.1)

    # Governance rail.
    rail_y = 0.155
    ax.add_line(Line2D([0.090, 0.915], [rail_y, rail_y], color=COL["line"], lw=1.2))
    checkpoints = [
        (0.105, "temporal split"),
        (0.275, "schema tests"),
        (0.445, "leak audit"),
        (0.615, "training gates"),
        (0.785, "baseline tests"),
        (0.915, "counterfactual evidence"),
    ]
    for x, label in checkpoints:
        ax.add_patch(Circle((x, rail_y), 0.0105, facecolor=COL["green_bg"],
                            edgecolor=COL["green"], linewidth=0.9))
        ax.text(x, rail_y - 0.034, label, ha="center", va="top",
                fontsize=5.9, color=COL["ink"], fontweight="bold")

    ax.text(0.055, 0.225, "Validation and publication-evidence rail",
            ha="left", va="center", fontsize=6.5, fontweight="bold", color=COL["green"])

    fig_caption = (
        "Figure 1. System architecture of RailRL for route-setting decision support."
    )
    ax.text(0.025, 0.035, fig_caption, ha="left", va="center",
            fontsize=5.8, color=COL["muted"])


def main():
    width_mm, height_mm = 183, 112
    fig = plt.figure(figsize=(width_mm / 25.4, height_mm / 25.4), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    draw_system_architecture(ax)

    stem = OUT / "railrl_system_architecture"
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(stem)


if __name__ == "__main__":
    main()
