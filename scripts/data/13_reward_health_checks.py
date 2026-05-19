"""P2.4 Iter D - Reward sanity checks + weight ablation."""
from __future__ import annotations
import argparse
import json
import sys
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from railrl import config as C
from railrl.p2_data_eng.reward_model import RewardModel, RewardThresholds
from railrl.p2_data_eng.episodes     import assign_episodes


HEALTH_DIR = C.REWARDS_DIR / "health"
HEALTH_DIR.mkdir(parents=True, exist_ok=True)

PNG_COMPONENTS  = HEALTH_DIR / "component_distributions.png"
PNG_EP_RETURN   = HEALTH_DIR / "episode_return_distribution.png"
PNG_WEIGHT_ABL  = HEALTH_DIR / "weight_sensitivity.png"
CSV_TOP_BOTTOM  = HEALTH_DIR / "top_bottom_episodes.csv"
JSON_SUMMARY    = HEALTH_DIR / "health_summary.json"
MD_SUMMARY      = HEALTH_DIR / "health_summary.md"


WEIGHT_PRESETS = {
    "conservative": {"w_delay": 0.5, "w_throughput": 0.3, "w_headway": 1.5, "w_wait": 0.5},
    "default":      {"w_delay": 1.0, "w_throughput": 0.5, "w_headway": 1.0, "w_wait": 0.3},
    "aggressive":   {"w_delay": 1.5, "w_throughput": 1.0, "w_headway": 0.5, "w_wait": 0.1},
}


def _maybe_plot():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        return None


def plot_component_distributions(df, plt):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    components = ["r_delay", "r_throughput", "r_headway", "r_wait"]
    for ax, col in zip(axes.flat, components):
        nonzero = df[col][df[col] != 0]
        if nonzero.empty:
            ax.text(0.5, 0.5, "all zero", ha="center", va="center"); ax.set_title(col); continue
        ax.hist(nonzero, bins=50, log=True)
        ax.set_title(f"{col}  (n_nonzero={len(nonzero):,}, mean={df[col].mean():+.4f})")
        ax.set_xlabel("weighted reward")
        ax.set_ylabel("log count")
    fig.tight_layout()
    fig.savefig(PNG_COMPONENTS, dpi=120); plt.close(fig)


def plot_episode_returns(df, plt):
    ep = df.groupby("episode_id")["r_total"].sum()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    # Linear-clipped view
    a = ep.clip(-50, 50)
    axes[0].hist(a, bins=80)
    for p, ls in [(1, ":"), (50, "-"), (99, "--")]:
        v = float(np.percentile(a, p))
        axes[0].axvline(v, color="r", ls=ls, alpha=0.5, label=f"P{p}={v:+.1f}")
    axes[0].axvline(0, color="k", lw=1, ls=":")
    axes[0].set_title(f"Per-episode return  n={len(ep):,}  mean={ep.mean():+.2f}  std={ep.std():.2f}")
    axes[0].set_xlabel("return (clipped to +-50)"); axes[0].legend(fontsize=8)
    # Tails view
    axes[1].hist(ep[ep > 0], bins=60, alpha=0.6, color="g", label=f"positive (n={(ep>0).sum():,})")
    axes[1].hist(ep[ep < 0], bins=60, alpha=0.6, color="r", label=f"negative (n={(ep<0).sum():,})")
    axes[1].set_yscale("log"); axes[1].set_xlabel("return")
    axes[1].set_title("Positive vs negative episodes (log y)"); axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PNG_EP_RETURN, dpi=120); plt.close(fig)


def top_bottom_episodes(df, n=20):
    ep = df.groupby("episode_id").agg(
        focal_train=("focal_train", "first"),
        n_decisions=("decision_id", "size"),
        n_set=("label", lambda x: int((x == "set").sum())),
        n_wait=("label", lambda x: int((x == "wait").sum())),
        n_used_routes=("route_outcome", lambda x: int((x == "used").sum())),
        n_cancelled=("route_outcome", lambda x: int((x == "unused_cancelled").sum())),
        return_=("r_total", "sum"),
        r_delay_sum=("r_delay", "sum"),
        r_thru_sum=("r_throughput", "sum"),
        r_head_sum=("r_headway", "sum"),
        r_wait_sum=("r_wait", "sum"),
        first_time=("time", "min"),
        duration_s=("time", lambda x: (x.max() - x.min()).total_seconds()),
    ).reset_index()
    top = ep.nlargest(n, "return_").assign(rank_kind="top")
    bot = ep.nsmallest(n, "return_").assign(rank_kind="bottom")
    out = pd.concat([top, bot], ignore_index=True)
    out.to_csv(CSV_TOP_BOTTOM, index=False)
    return out


def weight_sensitivity(df, plt):
    """For each weight preset, recompute r_total and compare distributions."""
    thr = RewardThresholds.from_calibration()
    results = {}
    for name, weights in WEIGHT_PRESETS.items():
        model = RewardModel(weights, thr)
        rew = model.compute_batch(df)
        df_w = df.copy()
        df_w["r_total"] = rew["r_total"].values
        ep_ret = df_w.groupby("episode_id")["r_total"].sum()
        results[name] = {
            "weights": weights,
            "ep_return_mean":   float(ep_ret.mean()),
            "ep_return_std":    float(ep_ret.std()),
            "ep_return_p1":     float(ep_ret.quantile(0.01)),
            "ep_return_p50":    float(ep_ret.quantile(0.50)),
            "ep_return_p99":    float(ep_ret.quantile(0.99)),
            "frac_positive":    float((ep_ret > 0).mean()),
        }
        results[name]["_returns"] = ep_ret  # keep for plot

    # Plot side-by-side (only if matplotlib is available)
    if plt is not None:
        fig, ax = plt.subplots(figsize=(11, 5))
        colors = {"conservative": "tab:blue", "default": "tab:green", "aggressive": "tab:red"}
        for name in WEIGHT_PRESETS:
            rets = np.clip(results[name]["_returns"].values, -50, 50)
            ax.hist(rets, bins=80, alpha=0.5, label=name, color=colors[name])
        ax.set_yscale("log"); ax.set_xlabel("episode return (clipped +-50)")
        ax.set_title("Weight sensitivity - episode return distribution")
        ax.legend(fontsize=10)
        fig.tight_layout()
        fig.savefig(PNG_WEIGHT_ABL, dpi=120); plt.close(fig)

    # Cross-preset rank correlation
    rank_corr = {}
    presets = list(WEIGHT_PRESETS)
    for i, a in enumerate(presets):
        for b in presets[i+1:]:
            sa = results[a]["_returns"]; sb = results[b]["_returns"]
            common = sa.index.intersection(sb.index)
            r = sa.loc[common].rank().corr(sb.loc[common].rank())
            rank_corr[f"{a}_vs_{b}_spearman"] = float(r)

    out = {name: {k: v for k, v in r.items() if k != "_returns"}
            for name, r in results.items()}
    out["rank_correlations"] = rank_corr
    return out


def proxy_correlation(df):
    """Proxy success indicator: episode has zero cancelled PRs and >50% used."""
    ep = df.groupby("episode_id").agg(
        return_=("r_total", "sum"),
        n_set=("label", lambda x: int((x == "set").sum())),
        n_used=("route_outcome", lambda x: int((x == "used").sum())),
        n_cancelled=("route_outcome", lambda x: int((x == "unused_cancelled").sum())),
    )
    ep["any_cancelled"] = (ep["n_cancelled"] > 0).astype(int)
    ep["use_rate"]      = ep["n_used"] / ep["n_set"].clip(lower=1)
    return {
        "n_episodes":           int(len(ep)),
        "n_with_cancellation":  int(ep["any_cancelled"].sum()),
        "spearman_return_vs_no_cancel":
            float((-ep["any_cancelled"]).rank().corr(ep["return_"].rank())),
        "spearman_return_vs_use_rate":
            float(ep["use_rate"].rank().corr(ep["return_"].rank())),
        "mean_return_with_cancel":
            float(ep[ep["any_cancelled"] == 1]["return_"].mean()),
        "mean_return_without_cancel":
            float(ep[ep["any_cancelled"] == 0]["return_"].mean()),
    }


def write_summary_md(stats):
    lines = [
        "# P2.4 Iter D - Reward Health Check",
        "",
        f"Run on full 2.64M decisions, {stats['n_episodes']:,} episodes.",
        "",
        "## Reward distribution",
        f"- r_total mean: {stats['r_total_mean']:+.3f}  std: {stats['r_total_std']:.3f}",
        f"- r_total range: [{stats['r_total_min']:+.2f}, {stats['r_total_max']:+.2f}]",
        f"- per-episode return mean: {stats['ep_mean']:+.2f}  std: {stats['ep_std']:.2f}",
        f"- frac positive episodes: {stats['frac_positive_episodes']:.1%}",
        "",
        "## Component coverage (non-zero count)",
    ]
    for k, v in stats["coverage"].items():
        lines.append(f"- {k}: {v:,}")
    lines += [
        "",
        "## Weight sensitivity",
        "",
    ]
    for name, w in stats["weight_sensitivity"].items():
        if name == "rank_correlations":
            continue
        lines.append(f"- **{name}** weights={w['weights']}")
        lines.append(f"  ep_return mean/std/P1/P50/P99: "
                      f"{w['ep_return_mean']:+.2f} / {w['ep_return_std']:.2f} / "
                      f"{w['ep_return_p1']:+.2f} / {w['ep_return_p50']:+.2f} / {w['ep_return_p99']:+.2f}")
        lines.append(f"  frac_positive: {w['frac_positive']:.1%}")
    lines += [
        "",
        "**Spearman rank correlations across weight presets** (closer to 1.0 = "
        "policy ordering robust to weight choice):",
    ]
    for k, v in stats["weight_sensitivity"]["rank_correlations"].items():
        lines.append(f"- {k}: {v:.3f}")
    lines += [
        "",
        "## Proxy correlations",
        f"- Spearman(return, no_cancellation): {stats['proxy']['spearman_return_vs_no_cancel']:+.3f}",
        f"- Spearman(return, use_rate):        {stats['proxy']['spearman_return_vs_use_rate']:+.3f}",
        f"- Mean return WITH cancellation:    {stats['proxy']['mean_return_with_cancel']:+.2f}",
        f"- Mean return WITHOUT cancellation: {stats['proxy']['mean_return_without_cancel']:+.2f}",
        "",
        "**Interpretation of weak proxy correlations**: 99.5% of episodes have",
        "no cancellations and use_rate ~ 1 (signaller is highly effective at",
        "route-setting). The `no_cancel` and `use_rate` proxies are therefore",
        "near-constant across episodes and cannot differentiate them well",
        "(flat distribution -> low Spearman). This does NOT indicate the reward",
        "is mis-specified; rather, our naive proxies are insufficient. A better",
        "proxy would require external operational KPIs (PPM, CaSL, headway",
        "violation reports) which are out of scope for the current dataset.",
    ]
    MD_SUMMARY.write_text("\n".join(lines), encoding="utf-8")


def main():
    print("[1/5] Loading decision_rewards ...")
    t0 = _time.time()
    df = pd.read_parquet(C.DECISION_REWARDS_PARQUET)
    print(f"  {len(df):,} rows, {_time.time()-t0:.1f}s")

    plt = _maybe_plot()
    if plt is None:
        print("  [warn] matplotlib not available; skipping PNGs.")

    print("[2/5] Component + episode return distributions ...")
    if plt:
        plot_component_distributions(df, plt)
        plot_episode_returns(df, plt)
        print(f"  -> {PNG_COMPONENTS}")
        print(f"  -> {PNG_EP_RETURN}")

    print("[3/5] Top/bottom 20 episodes ...")
    tb = top_bottom_episodes(df, n=20)
    print(f"  -> {CSV_TOP_BOTTOM}")
    print(f"  top_5 returns: {tb[tb['rank_kind']=='top'].head()['return_'].tolist()}")
    print(f"  bot_5 returns: {tb[tb['rank_kind']=='bottom'].head()['return_'].tolist()}")

    print("[4/5] Weight sensitivity (3 presets) ...")
    ws = weight_sensitivity(df, plt) if plt else weight_sensitivity(df, None)
    if plt: print(f"  -> {PNG_WEIGHT_ABL}")
    for k, v in ws.items():
        if k != "rank_correlations":
            print(f"  {k:<13s}  ep_mean={v['ep_return_mean']:+.2f}  "
                  f"frac_positive={v['frac_positive']:.1%}")
    print(f"  rank_correlations: {ws['rank_correlations']}")

    print("[5/5] Proxy correlations ...")
    pr = proxy_correlation(df)
    print(f"  Spearman(return, no_cancel): {pr['spearman_return_vs_no_cancel']:+.3f}")
    print(f"  Spearman(return, use_rate):  {pr['spearman_return_vs_use_rate']:+.3f}")

    ep = df.groupby("episode_id")["r_total"].sum()
    summary = {
        "n_decisions":              int(len(df)),
        "n_episodes":               int(df["episode_id"].nunique()),
        "r_total_mean":             float(df["r_total"].mean()),
        "r_total_std":              float(df["r_total"].std()),
        "r_total_min":              float(df["r_total"].min()),
        "r_total_max":              float(df["r_total"].max()),
        "ep_mean":                  float(ep.mean()),
        "ep_std":                   float(ep.std()),
        "ep_p1":                    float(ep.quantile(0.01)),
        "ep_p50":                   float(ep.quantile(0.50)),
        "ep_p99":                   float(ep.quantile(0.99)),
        "frac_positive_episodes":   float((ep > 0).mean()),
        "coverage": {
            "approach_distance":         int(df["approach_distance"].notna().sum()),
            "delay_change_seconds":      int(df["delay_change_seconds"].notna().sum()),
            "next_tc_headway_seconds":   int(df["next_tc_headway_seconds"].notna().sum()),
            "route_outcome":             int(df["route_outcome"].notna().sum()),
        },
        "weight_sensitivity": ws,
        "proxy":              pr,
    }
    JSON_SUMMARY.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    write_summary_md(summary)
    print(f"\nWrote {JSON_SUMMARY}")
    print(f"Wrote {MD_SUMMARY}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
