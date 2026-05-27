"""L5 step 2 — recover signaller's effective reward weights (Table V, spec 05 §11).

Cheap CPU analysis: load the behavior-FQE Q-table (scripts/eval/08_fqe_behavior_qtable.py),
fit conditional-logit MaxEnt-IRL (src/railrl/xai/l5_irl) globally and per subset
(per-prefix, per-headcode-class), with episode-clustered bootstrap CIs → Table V, compared
to the TRAINED reward weights (w_delay=1.0, w_throughput=0.5, w_headway=1.0, w_wait=0.3).

The headline: does the signaller weight DELAY more than the trained reward effectively did?
(Our OPE found the trained model under-prioritises delay because r_delay is sparse/small.)

Run on Windows (CPU, minutes):
    python scripts/eval/09_l5_irl.py --n-boot 300
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from railrl import config as C
from railrl.xai.l5_irl import maxent_irl, bootstrap_irl, normalize_w, COMPONENTS, TRAINED_W

QCOLS = ["q_delay", "q_throughput", "q_headway", "q_wait"]


def build_design(df):
    """df (rows = decision×legal-action, grouped by sample_id) → X, offsets, chosen, ep, meta."""
    df = df.sort_values("sample_id", kind="stable").reset_index(drop=True)
    sid = df["sample_id"].to_numpy()
    # contiguous decision groups
    starts = np.r_[0, np.where(np.diff(sid) != 0)[0] + 1]
    offsets = np.r_[starts, len(sid)]
    X = df[QCOLS].to_numpy(float)
    is_ch = df["is_chosen"].to_numpy(bool)
    chosen_rows = np.where(is_ch)[0]
    D = len(offsets) - 1
    if len(chosen_rows) != D:
        # fall back: take first chosen per group (or group start if none)
        chosen_rows = np.array([
            (np.where(is_ch[offsets[d]:offsets[d + 1]])[0][:1] + offsets[d]).tolist() or [offsets[d]]
            for d in range(D)]).ravel()
    ep = df["episode_idx"].to_numpy()[starts]
    prefix = df["prefix"].to_numpy()[starts]
    hc = df["headcode_class"].to_numpy()[starts]
    return X, offsets, chosen_rows, ep, prefix, hc


def subset_design(X, offsets, chosen, ep, mask_dec):
    """Restrict to decisions where mask_dec is True → rebuilt (X,offsets,chosen,ep)."""
    keep = np.where(mask_dec)[0]
    blocks, ch, off = [], [], [0]
    for d in keep:
        s, e = offsets[d], offsets[d + 1]
        blocks.append(X[s:e]); ch.append(off[-1] + (chosen[d] - s)); off.append(off[-1] + (e - s))
    return np.concatenate(blocks, 0), np.array(off), np.array(ch), ep[keep]


def fmt_row(name, w, ci=None):
    cells = []
    for i, k in enumerate(COMPONENTS):
        if ci is not None:
            cells.append(f"{w[i]:.2f}±{(ci['ci_high'][i]-ci['ci_low'][i])/2:.2f}")
        else:
            cells.append(f"{w[i]:.2f}")
    return f"  {name:22s} " + "   ".join(f"{c:>12}" for c in cells)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qtable", default=str(C.EVAL_DIR / "l5_qtable.parquet"))
    ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--l2", type=float, default=1e-2)
    ap.add_argument("--max-iter", type=int, default=600)
    ap.add_argument("--min-decisions", type=int, default=2000, help="min decisions for a subset row")
    args = ap.parse_args()
    import pyarrow.parquet as pq

    df = pq.read_table(args.qtable).to_pandas()
    X, offsets, chosen, ep, prefix, hc = build_design(df)
    D = len(offsets) - 1
    print(f"L5 IRL | {len(X):,} action-rows over {D:,} decisions, "
          f"{len(np.unique(ep)):,} episodes | normalize=l1")
    print(f"trained reward weights: " + ", ".join(f"{k}={TRAINED_W[k]}" for k in COMPONENTS))

    irl_kw = dict(l2=args.l2, max_iter=args.max_iter)
    out_rows = {}

    # ---- global (signaller), with bootstrap CI ----
    g = bootstrap_irl(X, offsets, chosen, ep, n_resamples=args.n_boot, **irl_kw)
    gw = g["w"]; gwn = normalize_w(gw, "l1")
    out_rows["global"] = {"w_raw": gw.tolist(), "w_norm": gwn.tolist(),
                          "ci_low": g["ci_low"].tolist(), "ci_high": g["ci_high"].tolist(),
                          "n_dec": int(D)}

    print("\n=== Table V — signaller's effective reward weights (l1-normalized) ===")
    print(f"  {'subset':22s} " + "   ".join(f"{k:>12}" for k in COMPONENTS))
    print(fmt_row("Global (ALL—wait-conf.)", normalize_w(gw, "l1"), None))
    # raw + CI line for global
    print(fmt_row("  └ raw ± half-CI", gw, g))

    # ---- per-subset point estimates (prefix, headcode_class) ----
    def subset_rows(label_arr, tag):
        for val in sorted(set(label_arr.tolist())):
            if val in ("NA", "", None):
                continue
            mask = (label_arr == val)
            if mask.sum() < args.min_decisions:
                continue
            Xs, os_, chs, eps = subset_design(X, offsets, chosen, ep, mask)
            w = maxent_irl(Xs, os_, chs, **irl_kw)
            wn = normalize_w(w, "l1")
            out_rows[f"{tag}:{val}"] = {"w_raw": w.tolist(), "w_norm": wn.tolist(),
                                        "n_dec": int(mask.sum())}
            print(fmt_row(f"{tag} {val} (n={int(mask.sum())})", wn, None))

    print("--- per prefix (line of route) ---")
    subset_rows(prefix, "prefix")
    print("--- per headcode-class ---")
    subset_rows(hc, "class")

    # ---- Q-feature diagnostic: exposes the wait confound behind the negative global w ----
    print("\n--- Q-feature diagnostic (mean component-Q over action rows) ---")
    ch = df["is_chosen"].to_numpy(bool); wta = (df["action_idx"].to_numpy() == 0)
    for qc in QCOLS:
        q = df[qc].to_numpy(float)
        print(f"  {qc:14s}: chosen {q[ch].mean():+.3f} | not-chosen {q[~ch].mean():+.3f} | "
              f"wait-act {q[wta].mean():+.3f} | route-act {q[~wta].mean():+.3f}")
    print("  (if 'wait-act' Q ≪ 'route-act', and signaller waits ~73%, the global IRL "
          "mistakes wait-propensity for a negative reward weight → use SET-only below)")

    # ---- SET-only, ROUTES-only IRL (the well-posed 'which route does the signaller prefer') ----
    set_sids = set(df.loc[(df["action_idx"] > 0) & df["is_chosen"], "sample_id"])
    dfr = df[(df["action_idx"] > 0) & df["sample_id"].isin(set_sids)]
    Xr, offr, chr_, epr, prefr, hcr = build_design(dfr)
    gr = bootstrap_irl(Xr, offr, chr_, epr, n_resamples=args.n_boot, **irl_kw)
    print(f"\n=== Table V — SET-only, routes-only (well-posed route-choice IRL; "
          f"{len(offr)-1:,} decisions) ===")
    print(f"  {'subset':22s} " + "   ".join(f"{k:>12}" for k in COMPONENTS))
    print(fmt_row("Signaller (set,routes)", normalize_w(gr["w"], "l1"), None))
    print(fmt_row("  └ raw ± half-CI", gr["w"], gr))
    out_rows["set_routes_only"] = {"w_raw": gr["w"].tolist(),
                                   "w_norm": normalize_w(gr["w"], "l1").tolist(),
                                   "ci_low": gr["ci_low"].tolist(), "ci_high": gr["ci_high"].tolist(),
                                   "n_dec": int(len(offr) - 1)}

    print("\ninterpretation: compare the signaller's NORMALIZED w to the trained "
          f"({', '.join(f'{k}={TRAINED_W[k]}' for k in COMPONENTS)}). If signaller w_delay is "
          "relatively higher → the signaller prioritises delay more than the trained reward "
          "effectively did (consistent with the OPE 'delay under-weighted' finding).")
    print("(L5 ESTIMATE: feature-matching IRL on behavior-FQE Q; relies on FQE generalizing "
          "to counterfactual candidate actions — the offline-RL OOD caveat. Single seed42.)")

    C.EVAL_DIR.mkdir(parents=True, exist_ok=True)
    p = C.EVAL_DIR / "l5_irl_weights.json"
    p.write_text(json.dumps({"components": COMPONENTS, "trained_w": TRAINED_W,
                             "n_boot": args.n_boot, "rows": out_rows}, indent=2))
    print(f"\n→ wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
