"""Figure 1: RailRL end-to-end architecture and MDP mechanism.

Figure contract
---------------
Core conclusion:
    RailRL connects operational railway feeds to leak-audited structured offline
    RL and explainable route-setting decision support.
Figure archetype:
    schematic-led composite.
Target journal/output:
    ESWA/T-ITS manuscript figure; double-column vector plus high-DPI raster.
Backend:
    Python/matplotlib only.
Final size:
    183 mm x 125 mm.
Panel map:
    a. End-to-end system architecture.
    b. Single decision-point MDP mechanism.
Evidence hierarchy:
    hero evidence: complete flow from feeds to decision support.
    validation evidence: leak audit, time split, sanity gates.
    controls/robustness: baselines/counterfactual evaluation shown as final gate.
Reviewer risk:
    Avoid implying that protected local operational references are disclosed;
    avoid claiming operational improvement before baseline/counterfactual evaluation.
"""
from __future__ import annotations

from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7.2,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "legend.frameon": False,
})

COL = {
    "ink": "#1F2933",
    "muted": "#687385",
    "line": "#B8C2CC",
    "feed": "#E8EEF5",
    "acq": "#E3F3EE",
    "mdp": "#FFF1D6",
    "model": "#E9E6F6",
    "eval": "#F7E4E6",
    "audit": "#DDEFE3",
    "blue": "#0F4D92",
    "teal": "#2D8C8C",
    "gold": "#C78318",
    "violet": "#6E5AA8",
    "red": "#B64342",
    "green": "#2E9E44",
    "white": "#FFFFFF",
}


def panel_label(ax, label: str, x: float, y: float) -> None:
    ax.text(x, y, label, transform=ax.transAxes, ha="left", va="top",
            fontsize=9.5, fontweight="bold", color=COL["ink"])


def rounded_box(ax, xy, wh, title, body=(), fc="#FFFFFF", ec=None, title_color=None,
                lw=0.9, radius=0.06, pad=0.012, title_size=7.5, body_size=6.2):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad={pad},rounding_size={radius}",
        linewidth=lw, edgecolor=ec or COL["line"], facecolor=fc,
        mutation_aspect=1,
    )
    ax.add_patch(patch)
    ax.text(x + 0.018, y + h - 0.026, title, ha="left", va="top",
            fontsize=title_size, fontweight="bold", color=title_color or COL["ink"])
    yy = y + h - 0.070
    for line in body:
        ax.text(x + 0.020, yy, line, ha="left", va="top",
                fontsize=body_size, color=COL["ink"])
        yy -= 0.036
    return patch


def arrow(ax, start, end, color=COL["muted"], lw=1.1, ms=10, connectionstyle="arc3,rad=0"):
    arr = FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=ms, linewidth=lw,
        color=color, shrinkA=4, shrinkB=4, connectionstyle=connectionstyle
    )
    ax.add_patch(arr)
    return arr


def small_badge(ax, x, y, text, fc, ec=None, color=None):
    rounded_box(ax, (x, y), (0.098, 0.036), text, (), fc=fc, ec=ec or fc,
                radius=0.018, pad=0.004, title_size=5.8, title_color=color or COL["ink"])


def draw_architecture(ax):
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    panel_label(ax, "a", 0.005, 0.985)
    ax.text(0.045, 0.965, "End-to-end RailRL system", fontsize=10.5,
            fontweight="bold", color=COL["ink"], va="top")
    ax.text(0.045, 0.888,
            "Public operational feeds are transformed into leak-audited offline RL transitions and explainable route-setting recommendations.",
            fontsize=6.7, color=COL["muted"], va="top")

    y = 0.565
    h = 0.245
    xs = [0.035, 0.205, 0.380, 0.555, 0.730]
    w = 0.145

    boxes = []
    boxes.append(rounded_box(ax, (xs[0], y), (w, h), "Data feeds",
                             ["TD events", "Movements", "Timetable/planning",
                              "Performance + graph"],
                             fc=COL["feed"], ec="#AEB8C6", title_color=COL["blue"]))
    boxes.append(rounded_box(ax, (xs[1], y), (w, h), "Acquisition",
                             ["Broker subscription", "Runtime monitoring",
                              "Resilient collection", "Feed storage"],
                             fc=COL["acq"], ec="#A7CDBF", title_color=COL["teal"]))
    boxes.append(rounded_box(ax, (xs[2], y), (w, h), "MDP builder",
                             ["Decision points", "Candidate actions",
                              "Rewards + episodes", "Leak-safe state"],
                             fc=COL["mdp"], ec="#E0BD75", title_color=COL["gold"]))
    boxes.append(rounded_box(ax, (xs[3], y), (w, h), "Offline RL",
                             ["HGT graph encoder", "Event transformer",
                              "Per-action Q", "CQL training"],
                             fc=COL["model"], ec="#B8ADD8", title_color=COL["violet"]))
    boxes.append(rounded_box(ax, (xs[4], y), (w, h), "Evaluation + XAI",
                             ["Imitation metrics", "Baselines", "Counterfactuals",
                              "L1-L5 explanations"],
                             fc=COL["eval"], ec="#E0B2B8", title_color=COL["red"]))

    for i in range(len(boxes) - 1):
        arrow(ax, (xs[i] + w, y + h * 0.55), (xs[i + 1], y + h * 0.55))

    # Validation rails beneath the pipeline
    rail_y = 0.435
    ax.add_line(Line2D([0.050, 0.852], [rail_y, rail_y], color=COL["line"], lw=1.0))
    checkpoints = [
        (0.105, "Traceability", "feed origin preserved"),
        (0.290, "Temporal cut", "train < val < test"),
        (0.475, "Leak audit", "no answer/future state"),
        (0.660, "Training gates", "bounded Q, no NaN"),
        (0.845, "Operational test", "pending counterfactuals"),
    ]
    for x, title, desc in checkpoints:
        c = Circle((x, rail_y), 0.0105, facecolor=COL["audit"], edgecolor=COL["green"], lw=0.8)
        ax.add_patch(c)
        ax.text(x, rail_y - 0.032, title, ha="center", va="top",
                fontsize=6.2, fontweight="bold", color=COL["ink"])
        ax.text(x, rail_y - 0.060, desc, ha="center", va="top",
                fontsize=5.6, color=COL["muted"])

    # Output callout
    rounded_box(ax, (0.045, 0.205), (0.905, 0.110), "Manuscript-ready contribution",
                ["Traceable feed capture, leak-audited offline RL training, and interpretable route-setting decision support."],
                fc="#F8FAFC", ec="#CBD5E1", radius=0.035, title_size=7.2, body_size=6.1)
    small_badge(ax, 0.800, 0.274, "ESWA", "#E9E6F6", color=COL["violet"])
    small_badge(ax, 0.905, 0.274, "T-ITS", "#F7E4E6", color=COL["red"])


def draw_mdp(ax):
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    panel_label(ax, "b", 0.005, 0.980)
    ax.text(0.045, 0.955, "Decision-point mechanism", fontsize=10.0,
            fontweight="bold", color=COL["ink"], va="top")
    ax.text(0.045, 0.875,
            "Each historical signalling decision becomes an observed transition for conservative offline RL.",
            fontsize=6.6, color=COL["muted"], va="top")

    # State block
    rounded_box(ax, (0.050, 0.585), (0.245, 0.235), r"State $s_t$",
                ["3-hop infrastructure graph", "recent TD event sequence",
                 "schedule outlook", "special-case flags"],
                fc=COL["feed"], ec="#AEB8C6", title_color=COL["blue"])

    # Action set
    rounded_box(ax, (0.380, 0.590), (0.265, 0.225), r"Dynamic action set $A_t$",
                ["0: wait", "1..K: set candidate route",
                 "mask invalid/padded actions"],
                fc=COL["mdp"], ec="#E0BD75", title_color=COL["gold"])

    # Reward
    rounded_box(ax, (0.720, 0.590), (0.235, 0.225), r"Reward $r_t$",
                ["delay change", "route use", "headway risk", "waiting cost"],
                fc=COL["eval"], ec="#E0B2B8", title_color=COL["red"])

    arrow(ax, (0.295, 0.700), (0.380, 0.700), color=COL["gold"])
    arrow(ax, (0.645, 0.700), (0.720, 0.700), color=COL["red"])

    # Model computation row
    rounded_box(ax, (0.085, 0.260), (0.185, 0.150), "Graph + events",
                ["HGT + Transformer"], fc="#F8FAFC", ec="#CBD5E1", title_size=6.6, body_size=5.8)
    rounded_box(ax, (0.330, 0.260), (0.185, 0.150), "Fusion",
                ["state embedding"], fc="#F8FAFC", ec="#CBD5E1", title_size=6.6, body_size=5.8)
    rounded_box(ax, (0.575, 0.260), (0.185, 0.150), "Per-action Q",
                ["Q(s, wait)", "Q(s, route_i)"], fc="#F8FAFC", ec="#CBD5E1", title_size=6.6, body_size=5.8)
    rounded_box(ax, (0.805, 0.260), (0.145, 0.150), "Policy",
                ["argmax Q", "or explain"], fc="#F8FAFC", ec="#CBD5E1", title_size=6.6, body_size=5.8)
    arrow(ax, (0.270, 0.335), (0.330, 0.335), color=COL["violet"])
    arrow(ax, (0.515, 0.335), (0.575, 0.335), color=COL["violet"])
    arrow(ax, (0.760, 0.335), (0.805, 0.335), color=COL["violet"])

    # Transition line
    ax.text(0.055, 0.125, "Observed transition:", fontsize=6.8,
            fontweight="bold", color=COL["ink"], ha="left", va="center")
    ax.text(0.238, 0.125,
            r"$(s_t,\ a_t,\ r_t,\ s_{t+1},\ done_t)$ from canonical episode order",
            fontsize=6.8, color=COL["ink"], ha="left", va="center")
    ax.text(0.055, 0.070,
            "Training uses observed next states only; counterfactual rollouts are reserved for evaluation and explanation.",
            fontsize=5.9, color=COL["muted"], ha="left", va="center")


def main():
    width_mm, height_mm = 183, 125
    fig = plt.figure(figsize=(width_mm / 25.4, height_mm / 25.4), facecolor="white")
    gs = fig.add_gridspec(2, 1, height_ratios=[1.12, 1.0], hspace=0.08)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0])
    draw_architecture(ax_a)
    draw_mdp(ax_b)

    fig.text(0.012, 0.012,
             "Figure 1. End-to-end RailRL architecture for route-setting decision support.",
             ha="left", va="bottom", fontsize=5.8, color=COL["muted"])

    stem = OUT / "fig1_architecture"
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(stem)


if __name__ == "__main__":
    main()
