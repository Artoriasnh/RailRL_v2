"""3-seed test-set aggregation (spec 05 §3, publication-grade mean ± std).

Aggregates the per-seed test-set evaluation JSONs (from `01_evaluate_model.py` etc.) into
publication-ready **mean ± std (across seeds)** tables. Unlike `11_aggregate_results.py`
(which aggregates the per-epoch *training-log* val metrics), this script aggregates the
*test-set* numbers that go in the paper's §VII tables.

Inputs (auto-discovered by seed glob; missing seeds reported, never crashed):
  * `cql_seed{N}_best_test_metrics.json` (eval/01) — Tier-1 + Tier-2 (per-stratum) accuracies,
    plus wait_rate / route_head / time_head. **The headline Table I rows.**
  * `ope_fqe_seed{N}_total.json` / `ope_fqe_decompose_seed{N}.json` (eval/04/05) — total ΔV +
    per-component decomposition. Aggregated if all seeds present.
  * `l4_compliance_seed{N}.json` (eval/12) — hard-rule compliance rate.
  * `selective_override_seed{N}.json` (eval/13) — agreement / consider-override / silent.
  * `bc_seed{N}_test_metrics.json` / `iql_seed{N}_test_metrics.json` (eval/01 for learned
    baselines) — Table I rows B2 / B3.

Outputs:
  * `outputs/eval/aggregate_3seed.json` — all aggregated metrics with mean / std / n / values.
  * `outputs/eval/aggregate_3seed.md` — paper-ready markdown tables.

Pure-stdlib + numpy (no torch / pyarrow / matplotlib) → runs in sandbox + on Windows.
Run:
    python scripts/eval/15_aggregate_3seed.py                       # all algos, seeds 42/43/44
    python scripts/eval/15_aggregate_3seed.py --seeds 42 43 44 --algos cql bc iql
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from railrl import config as C

STRATA = ("overall", "late_train", "advance", "call_on", "platform_dev",
          "priority_compete", "unusual_id", "trivial")


def _agg(values: list) -> dict:
    """mean / std / min / max / n / values from a list of floats (None entries dropped)."""
    xs = [float(v) for v in values if v is not None]
    if not xs:
        return {"mean": None, "std": None, "min": None, "max": None, "n": 0, "values": []}
    a = np.asarray(xs, dtype=float)
    return {"mean": float(a.mean()), "std": float(a.std(ddof=0)),
            "min": float(a.min()), "max": float(a.max()),
            "n": len(xs), "values": [round(x, 4) for x in xs]}


def _load_json(p: Path) -> Optional[dict]:
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


# ---------- per-algo loaders (eval/01 — Tier-1/2) ----------

def _basics_for_algo(algo: str, seeds: list) -> dict:
    """Tier-1/2 set-only top-1 + key Tier-1 scalars across seeds for one algo."""
    if algo == "cql":
        files = {s: C.EVAL_DIR / f"cql_seed{s}_best_test_metrics.json" for s in seeds}
    else:                                            # bc / iql / etc.
        files = {s: C.EVAL_DIR / f"{algo}_seed{s}_test_metrics.json" for s in seeds}
    loaded = {s: _load_json(p) for s, p in files.items()}
    present_seeds = [s for s, d in loaded.items() if d is not None]
    missing = sorted(set(seeds) - set(present_seeds))

    # tier-1 scalars
    t1_keys = ["action_top1_all", "action_top1_set", "wait_rate_model",
               "wait_rate_signaller", "wait_rate_delta", "wait_recall",
               "wait_precision", "route_head_top1", "time_head_top1"]
    tier1 = {k: _agg([loaded[s]["tier1_overall"].get(k) if loaded[s] else None
                      for s in seeds]) for k in t1_keys}

    # tier-2 per-stratum acc_set (the publication numbers)
    tier2 = {}
    for st in STRATA:
        tier2[st] = {
            "acc_set": _agg([loaded[s]["tier2_stratified"].get(st, {}).get("acc_set")
                             if loaded[s] else None for s in seeds]),
            "acc_all": _agg([loaded[s]["tier2_stratified"].get(st, {}).get("acc_all")
                             if loaded[s] else None for s in seeds]),
            "n_set": [(loaded[s] or {}).get("tier2_stratified", {}).get(st, {}).get("n_set")
                      for s in seeds if loaded[s]],
        }
    return {"algo": algo, "seeds_requested": seeds, "seeds_present": present_seeds,
            "seeds_missing": missing, "files": {s: str(p) for s, p in files.items()},
            "tier1": tier1, "tier2": tier2}


# ---------- OPE / L4 / §12 (CQL only) ----------

def _ope_for_seeds(seeds: list) -> dict:
    """OPE/FQE ΔV (3-seed). Three numbers:
      * total_dv_warmstart — from 04 (r_total FQE w/ warm-start). LESS RELIABLE: r_total-scale
        warm init under-converges delay; reported only for transparency.
      * total_dv_fresh — from 05's `delta_V.total` (fresh-init multi-key FQE). PRIMARY.
      * per-component — from 05's `delta_V.{delay,throughput,headway,wait}`.
    """
    totals = [_load_json(C.EVAL_DIR / f"ope_fqe_seed{s}_total.json") for s in seeds]
    decomps = [_load_json(C.EVAL_DIR / f"ope_fqe_decompose_seed{s}.json") for s in seeds]
    out = {
        "total_dv_warmstart": _agg([(d or {}).get("delta_V") for d in totals]),
        "total_dv_fresh":     _agg([(d or {}).get("delta_V", {}).get("total") for d in decomps]),
    }
    for k in ("delay", "throughput", "headway", "wait"):
        out[f"{k}_dv"] = _agg([(d or {}).get("delta_V", {}).get(k) for d in decomps])
    # consistency check (Σ-check + fit residual)
    out["sum_components_delta_V"] = _agg(
        [(d or {}).get("sum_components_delta_V") for d in decomps])
    out["fit_residual_abs_mean"] = _agg(
        [(d or {}).get("fit_residual_abs_mean") for d in decomps])
    return out


def _l4_for_seeds(seeds: list) -> dict:
    files = {s: _load_json(C.EVAL_DIR / f"l4_compliance_seed{s}.json") for s in seeds}
    return {
        "model_hard_compliant_rate":
            _agg([(files[s] or {}).get("headline", {}).get("model_hard_compliant_rate")
                  for s in seeds]),
        "signaller_hard_compliant_rate":
            _agg([(files[s] or {}).get("headline", {}).get("signaller_hard_compliant_rate")
                  for s in seeds]),
    }


def _so_for_seeds(seeds: list) -> dict:
    """§12 selective override rates (primary = δ=0.5 refined)."""
    files = {s: _load_json(C.EVAL_DIR / f"selective_override_seed{s}.json") for s in seeds}

    def pull(s, k):
        d = files[s]
        if not d:
            return None
        m = (d.get("meta") or {})
        if k == "agreement_set_only":
            return m.get("agreement_rate_set_only")
        sweep = d.get("sweep", {}).get("0.5", {}).get("refined", {})
        return (sweep.get("rates") or {}).get(k)

    return {
        "agreement_set_only": _agg([pull(s, "agreement_set_only") for s in seeds]),
        "consider_override_rate_primary":
            _agg([pull(s, "consider-override") for s in seeds]),
        "silent_rate_primary":
            _agg([pull(s, "silent") for s in seeds]),
    }


# ---------- markdown writer ----------

def _ms(d: dict, pct: bool = True, prec: int = 3) -> str:
    if d["mean"] is None:
        return "—"
    s = "%" if pct else ""
    sc = 100 if pct else 1
    if d["n"] == 1:
        return f"{sc*d['mean']:.{prec-1}f}{s}"
    return f"{sc*d['mean']:.{prec-1}f}{s} ± {sc*d['std']:.{prec-1}f}"


def _md(report: dict) -> str:
    L = ["# 3-seed test-set aggregation",
         "",
         f"Aggregated across seeds **{report['seeds']}**.",
         ""]
    # Table I — per-stratum set-only top-1 across algos
    L += ["## Table I — set-only top-1 (mean ± std across seeds)",
          "",
          "| stratum | " + " | ".join(f"{a.upper()}" for a in report["algos_with_data"]) + " |",
          "|---|" + "|".join(["---"] * len(report["algos_with_data"])) + "|"]
    for st in STRATA:
        row = [f"**{st}**"]
        for a in report["algos_with_data"]:
            d = report["by_algo"][a]["tier2"][st]["acc_set"]
            row.append(_ms(d, pct=True, prec=3))
        L.append("| " + " | ".join(row) + " |")
    # missing-seed callouts
    L += ["", "_Missing seeds per algo:_",
          *[f"- **{a}**: " + (", ".join(f"seed {x}" for x in report['by_algo'][a]['seeds_missing'])
                              if report['by_algo'][a]['seeds_missing'] else "none — all present")
            for a in report["by_algo"]]]
    # OPE
    if report.get("ope") and any(report["ope"][k]["n"] for k in report["ope"]):
        L += ["", "## OPE / FQE — ΔV vs signaller (CQL only)", "",
              "Primary = fresh-init multi-key FQE (05). Warm-start total (04) shown for "
              "transparency but under-converges delay so its total is biased low.",
              "",
              "| component | ΔV (mean ± std) | note |", "|---|---|---|",
              f"| **total (fresh-init, 05) — PRIMARY** | {_ms(report['ope']['total_dv_fresh'], pct=False, prec=4)} | headline |",
              f"| total (warm-start, 04) | {_ms(report['ope']['total_dv_warmstart'], pct=False, prec=4)} | reference |",
              f"| delay | {_ms(report['ope']['delay_dv'], pct=False, prec=4)} | per-component |",
              f"| throughput | {_ms(report['ope']['throughput_dv'], pct=False, prec=4)} | per-component |",
              f"| headway | {_ms(report['ope']['headway_dv'], pct=False, prec=4)} | per-component |",
              f"| wait | {_ms(report['ope']['wait_dv'], pct=False, prec=4)} | per-component |",
              f"| Σ components | {_ms(report['ope']['sum_components_delta_V'], pct=False, prec=4)} | Σ-check |",
              f"| fit_residual | {_ms(report['ope']['fit_residual_abs_mean'], pct=False, prec=4)} | quality |"]
    # L4
    if report.get("l4") and report["l4"]["model_hard_compliant_rate"]["n"]:
        L += ["", "## L4 — hard-rule compliance (CQL only)", "",
              "| | mean ± std |", "|---|---|",
              f"| model | {_ms(report['l4']['model_hard_compliant_rate'])} |",
              f"| signaller | {_ms(report['l4']['signaller_hard_compliant_rate'])} |"]
    # §12
    if report.get("so") and report["so"]["agreement_set_only"]["n"]:
        L += ["", "## §12 Selective Override (PRIMARY δ_L3=0.5 + refined gate_l4)", "",
              "| metric | mean ± std |", "|---|---|",
              f"| agreement (set-only) | {_ms(report['so']['agreement_set_only'])} |",
              f"| consider-override | {_ms(report['so']['consider_override_rate_primary'])} |",
              f"| silent | {_ms(report['so']['silent_rate_primary'])} |"]
    L += ["", "## Raw values per seed",
          "",
          "Stored in `aggregate_3seed.json` under `by_algo.<algo>.tier2.<stratum>.acc_set.values`."]
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description="3-seed test-set aggregation (spec 05 §3).")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--algos", nargs="+", default=["cql", "bc", "iql"])
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    seeds = sorted(set(args.seeds))

    by_algo = {a: _basics_for_algo(a, seeds) for a in args.algos}
    algos_with_data = [a for a in args.algos if by_algo[a]["seeds_present"]]
    print(f"seeds: {seeds} | algos: {args.algos}")
    for a in args.algos:
        ba = by_algo[a]
        print(f"  [{a}] present {ba['seeds_present']} | missing {ba['seeds_missing']}")

    # OPE / L4 / §12 only meaningful for CQL (the main model)
    ope = _ope_for_seeds(seeds) if "cql" in algos_with_data else None
    l4 = _l4_for_seeds(seeds) if "cql" in algos_with_data else None
    so = _so_for_seeds(seeds) if "cql" in algos_with_data else None

    report = {"seeds": seeds, "algos_with_data": algos_with_data,
              "by_algo": by_algo, "ope": ope, "l4": l4, "so": so}

    out_json = Path(args.out) if args.out else (C.EVAL_DIR / "aggregate_3seed.json")
    out_md = out_json.with_suffix(".md")
    out_json.write_text(json.dumps(report, indent=2))
    out_md.write_text(_md(report))
    print(f"\n→ wrote {out_json}")
    print(f"→ wrote {out_md}")

    # console summary
    print("\n=== Table I (set-only top-1, mean ± std) ===")
    print(f"{'stratum':18s} | " + " | ".join(f"{a.upper():>16s}" for a in algos_with_data))
    for st in STRATA:
        row = [f"{st:18s}"]
        for a in algos_with_data:
            d = report["by_algo"][a]["tier2"][st]["acc_set"]
            row.append(_ms(d, pct=True, prec=3).rjust(16))
        print(" | ".join(row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
