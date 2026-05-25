"""Stage 6 — aggregate multi-seed CQL training results (spec 04 §11/§12).

Reads the per-seed `train_log_seed{N}.json` files produced by 09_train.py,
extracts each seed's FINAL-epoch and BEST val metrics plus its §11 gate
pass/fail, then reports mean ± std (+ bootstrap CI) across seeds.

Pure stdlib + numpy: no torch / pyarrow, so it runs anywhere (incl. sandbox).
The final checkpoint stores only weights, so all metrics come from the JSON log.

Usage:
    python scripts/train/11_aggregate_results.py
    python scripts/train/11_aggregate_results.py --glob "outputs/train/cql_seed*" \
        --algo cql --out outputs/train/aggregate_cql.json

Note on scope: this aggregates the per-epoch VAL metrics logged during training
(route/action/time top-1, |Q|max, losses). The §11.3 per-stratum
"no-catastrophic-forgetting" check and the 3-tier / counterfactual evaluation
(spec 05) are NOT in the training log — those belong to Stage 8 and are
reported separately. Bootstrap CIs here are SEED-LEVEL (n = #seeds, typically 3)
and are weak by construction; per-decision bootstrap CIs are a Stage 8 artefact.
"""
from __future__ import annotations
import argparse
import datetime as _dt
import glob as _glob
import json
import re
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# spec 04 §11 gate thresholds (single source of truth, mirror of 09_train.check_gates)
GATE = {
    "A_route_acc_min": 0.50,
    "A_time_acc_min": 0.35,
    "A_route_loss_ratio_max": 0.70,   # L_route(final) < 0.70 * L_route(initial)
    "A_time_loss_ratio_max": 0.85,    # L_time(final)  < 0.85 * L_time(initial)
    "B_action_acc_min": 0.55,
    "B_qabsmax_max": 100.0,
    "B_cons_max": 50.0,
    "C_action_acc_min": 0.65,
}

# Final-epoch metrics we aggregate across seeds (log key -> short label).
FINAL_METRICS = {
    "val_action_acc": "Q_top1 (action)",
    "val_route_acc": "route_top1",
    "val_time_acc": "time_top1",
    "val_q_absmax": "|Q|max",
    "L_total": "L_total",
    "L_TD": "L_TD",
    "L_cons": "L_cons",
    "L_CQL": "L_CQL",
    "L_route": "L_route",
    "L_time": "L_time",
}


def _seed_from_path(p: Path) -> int:
    m = re.search(r"seed(\d+)", p.name)
    if not m:
        raise ValueError(f"cannot parse seed number from {p}")
    return int(m.group(1))


def _phase_rows(log: list[dict], phase: str) -> list[dict]:
    return [r for r in log if r.get("phase") == phase]


def summarize_seed(log: list[dict]) -> dict:
    """Extract final-epoch metrics, best action_acc, and §11 gate results."""
    if not log:
        raise ValueError("empty train log")

    A = _phase_rows(log, "A")
    B = _phase_rows(log, "B")
    C = _phase_rows(log, "C")
    final = log[-1]   # canonical: last logged epoch == Phase C ep20

    # best val_action_acc anywhere in the log (mirrors best.pt selection)
    best_row = max(log, key=lambda r: r.get("val_action_acc", float("-inf")))

    # ---- §11 gates (per spec 04 §11.1-11.3) ----
    gates: dict[str, dict] = {}
    if A:
        a0, a1 = A[0], A[-1]
        r_ratio = (a1["L_route"] / a0["L_route"]) if a0.get("L_route") else float("nan")
        t_ratio = (a1["L_time"] / a0["L_time"]) if a0.get("L_time") else float("nan")
        finite = bool(np.isfinite(a1.get("L_total", float("nan"))))
        gates["A"] = {
            "route_acc>0.50": (a1["val_route_acc"] > GATE["A_route_acc_min"], round(a1["val_route_acc"], 4)),
            "time_acc>0.35": (a1["val_time_acc"] > GATE["A_time_acc_min"], round(a1["val_time_acc"], 4)),
            "L_route_ratio<0.70": (r_ratio < GATE["A_route_loss_ratio_max"], round(r_ratio, 4)),
            "L_time_ratio<0.85": (t_ratio < GATE["A_time_loss_ratio_max"], round(t_ratio, 4)),
            "loss_finite": (finite, finite),
        }
    if B:
        b1 = B[-1]
        gates["B"] = {
            "Q_top1>0.55": (b1["val_action_acc"] > GATE["B_action_acc_min"], round(b1["val_action_acc"], 4)),
            "|Q|<100": (b1["val_q_absmax"] < GATE["B_qabsmax_max"], round(b1["val_q_absmax"], 2)),
            "L_cons<50": (b1.get("L_cons", float("nan")) < GATE["B_cons_max"], round(b1.get("L_cons", float("nan")), 4)),
        }
    if C:
        c1 = C[-1]
        gates["C"] = {
            "Q_top1>0.65": (c1["val_action_acc"] > GATE["C_action_acc_min"], round(c1["val_action_acc"], 4)),
            # §11.3 per-stratum forgetting check is NOT in the training log:
            "per_stratum_forgetting": (None, "needs Stage 8 stratified eval"),
        }

    all_pass = all(ok for ph in gates.values() for (ok, _v) in ph.values() if ok is not None)

    return {
        "n_epochs": len(log),
        "final_phase_epoch": f'{final["phase"]}{final["epoch"]}',
        "final": {k: final[k] for k in FINAL_METRICS if k in final},
        "best_action_acc": {
            "value": best_row["val_action_acc"],
            "at": f'{best_row["phase"]}{best_row["epoch"]}',
        },
        "gates": gates,
        "all_gates_pass": all_pass,
    }


def bootstrap_ci(vals: np.ndarray, n_boot: int = 10000, alpha: float = 0.05,
                 rng: np.random.Generator | None = None) -> tuple[float, float]:
    """Seed-level percentile bootstrap CI for the mean. NaN if <2 samples."""
    vals = np.asarray(vals, dtype=float)
    if vals.size < 2:
        return (float("nan"), float("nan"))
    rng = rng or np.random.default_rng(0)
    boot = rng.choice(vals, size=(n_boot, vals.size), replace=True).mean(axis=1)
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))


def aggregate(per_seed: dict[int, dict]) -> dict:
    """mean ± std (+ bootstrap CI) across seeds for each final metric, plus best."""
    rng = np.random.default_rng(0)
    seeds = sorted(per_seed)
    out: dict[str, dict] = {}

    # collect arrays
    cols: dict[str, list[float]] = {k: [] for k in FINAL_METRICS}
    best_col: list[float] = []
    for s in seeds:
        fin = per_seed[s]["final"]
        for k in FINAL_METRICS:
            if k in fin:
                cols[k].append(float(fin[k]))
        best_col.append(float(per_seed[s]["best_action_acc"]["value"]))

    def _stats(vals: list[float]) -> dict:
        a = np.asarray(vals, dtype=float)
        n = a.size
        lo, hi = bootstrap_ci(a, rng=rng)
        return {
            "n": int(n),
            "mean": float(a.mean()) if n else float("nan"),
            "std": float(a.std(ddof=1)) if n >= 2 else (0.0 if n == 1 else float("nan")),
            "min": float(a.min()) if n else float("nan"),
            "max": float(a.max()) if n else float("nan"),
            "ci95_lo": lo,
            "ci95_hi": hi,
            "values": [round(v, 6) for v in vals],
        }

    for k in FINAL_METRICS:
        out[k] = _stats(cols[k])
    out["best_action_acc"] = _stats(best_col)
    return out


def _fmt(stat: dict, pct: bool = False) -> str:
    if stat["n"] == 0:
        return "—"
    scale = 100.0 if pct else 1.0
    unit = "%" if pct else ""
    m, sd = stat["mean"] * scale, stat["std"] * scale
    lo, hi = stat["ci95_lo"] * scale, stat["ci95_hi"] * scale
    base = f"{m:7.3f}{unit} ± {sd:5.3f}"
    if np.isfinite(lo):
        base += f"  CI95[{lo:.3f}, {hi:.3f}]"
    return base


def main() -> int:
    ap = argparse.ArgumentParser(description="Aggregate multi-seed CQL training results (Stage 6).")
    ap.add_argument("--glob", default="outputs/train/cql_seed*",
                    help="dir glob (relative to project root) holding per-seed runs")
    ap.add_argument("--algo", default="cql")
    ap.add_argument("--out", default="outputs/train/aggregate_cql.json")
    ap.add_argument("--expect-seeds", type=int, default=3,
                    help="warn if fewer than this many seeds are found")
    args = ap.parse_args()

    pattern = str((PROJECT_ROOT / args.glob))
    run_dirs = sorted(Path(p) for p in _glob.glob(pattern) if Path(p).is_dir())
    per_seed: dict[int, dict] = {}
    for d in run_dirs:
        logs = sorted(d.glob("train_log_seed*.json"))
        if not logs:
            print(f"[skip] {d.name}: no train_log_seed*.json")
            continue
        log = json.loads(logs[0].read_text())
        seed = _seed_from_path(logs[0])
        per_seed[seed] = summarize_seed(log)

    if not per_seed:
        print(f"[error] no seed runs matched {pattern}", file=sys.stderr)
        return 1

    n = len(per_seed)
    seeds = sorted(per_seed)
    if n < args.expect_seeds:
        print(f"[warn] found {n} seed(s) {seeds}; expected {args.expect_seeds}. "
              f"mean±std/CI are weak/undefined with few seeds.\n")

    agg = aggregate(per_seed)

    # ---- console report ----
    print(f"=== {args.algo.upper()} multi-seed aggregate — seeds {seeds} (n={n}) ===\n")
    print("Per-seed §11 gate summary:")
    for s in seeds:
        ss = per_seed[s]
        verdict = "PASS" if ss["all_gates_pass"] else "CHECK"
        print(f"  seed {s}: gates={verdict:5s} | final {ss['final_phase_epoch']} "
              f"Q_top1={ss['final']['val_action_acc']:.4f} "
              f"|Q|max={ss['final']['val_q_absmax']:.1f} | "
              f"best Q_top1={ss['best_action_acc']['value']:.4f} @ {ss['best_action_acc']['at']}")
    print()

    print(f"{'metric':<18} {'mean ± std (across seeds)':<44} {'[min, max]'}")
    print("-" * 86)
    pct_keys = {"val_action_acc", "val_route_acc", "val_time_acc", "best_action_acc"}
    order = ["val_action_acc", "best_action_acc", "val_route_acc", "val_time_acc",
             "val_q_absmax", "L_total", "L_TD", "L_cons", "L_CQL", "L_route", "L_time"]
    for k in order:
        st = agg[k]
        if st["n"] == 0:
            continue
        pct = k in pct_keys
        scale = 100.0 if pct else 1.0
        rng_s = f"[{st['min']*scale:.3f}, {st['max']*scale:.3f}]"
        print(f"{k:<18} {_fmt(st, pct):<44} {rng_s}")

    # ---- JSON artefact ----
    payload = {
        "algo": args.algo,
        "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "n_seeds": n,
        "seeds": seeds,
        "gate_thresholds": GATE,
        "per_seed": {str(s): per_seed[s] for s in seeds},
        "aggregate": agg,
        "notes": [
            "Bootstrap CIs are seed-level (n=#seeds); weak by construction.",
            "§11.3 per-stratum forgetting + 3-tier/counterfactual eval are Stage 8 (not in train log).",
        ],
    }
    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\n→ wrote {out_path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
