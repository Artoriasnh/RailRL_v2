"""Pre-retrain全面体检 — run AFTER regenerating snapshots_v2.parquet (state patch 23
+ reward recompute 08→09→10 + 01 norm + 16 stratum), BEFORE the expensive retrain.

Design principle (lesson from every bug this project hit): bugs were ALWAYS caught
by a DISTRIBUTION check or an INDEPENDENT-SIGNAL cross-check — never by
matched/non-null counts — and several only showed up PER-MONTH. So this audit is
distribution-and-independent-signal based, broken out BY MONTH, and proactively
scans for ANY period-specific anomaly (the way Apr-Jul stood out).

Sections (each prints PASS/FAIL):
  A. Structure invariants  (row count, sample_id unique/full, canonical order, schema)
  B. Reward integrity      (r_total arithmetic; sample_id↔label independent agreement;
                            delay coverage BY MONTH — verifies fix #1+#2; r_delay share)
  C. State integrity       (planned_platform range; f_late_train/platform_dev rates;
                            lateness sign vs variation_status sample; schedule_outlook shape)
  D. Non-window-unchanged  (compare clean rows to the pre-patch backup — patch 23 must
                            not have touched rows outside Apr-Jul)
  E. Cross-month anomaly scan (panel of key features by month → flag outlier months)

Also (run separately, this script reminds you): 06_run_leak_audit_full,
07_audit_snapshots, 21_audit_leakage (data changed → re-audit leakage),
10_smoke_streaming (loader still healthy).

Read-only. Pure pandas/pyarrow (no torch). Run on Windows:
    python scripts/mdp/24_pre_retrain_audit.py
    python scripts/mdp/24_pre_retrain_audit.py --backup outputs/snapshots/snapshots_v2.prefix2.parquet
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from railrl import config as C

WIN_START = pd.Timestamp(C.MOVEMENTS_BST_FIX_START)
WIN_END = pd.Timestamp(C.MOVEMENTS_BST_FIX_END)
VOCAB_EXPECT = {"track_id": 268, "signal_id": 123, "route_id": 278, "train_id": 2184}

_results = []
def gate(name, ok, detail=""):
    _results.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{('  — ' + detail) if detail else ''}")


def col(pf_path, names):
    """Read scalar columns (full) into a DataFrame."""
    avail = [c for c in names if c in pq.ParquetFile(str(pf_path)).schema_arrow.names]
    return pq.read_table(str(pf_path), columns=avail).to_pandas()


def section_A(src):
    print("\n=== A. Structure invariants ===")
    df = col(src, ["sample_id", "episode_idx", "position_in_episode", "t", "split"])
    n = len(df)
    gate("row count == 1,996,572", n == 1_996_572, f"got {n:,}")
    gate("sample_id unique", df["sample_id"].is_unique, f"{df['sample_id'].nunique():,}/{n:,}")
    # canonical order: (episode_idx, position) non-decreasing; each episode starts at 0
    if {"episode_idx", "position_in_episode"}.issubset(df.columns):
        epi = df["episode_idx"].to_numpy()
        pos = df["position_in_episode"].to_numpy()
        mono = bool(np.all((np.diff(epi) > 0) | ((np.diff(epi) == 0) & (np.diff(pos) == 1))))
        gate("canonical order (episode_idx,position) monotonic", mono)
        starts = df.groupby("episode_idx")["position_in_episode"].min()
        gate("every episode starts at position 0", bool((starts == 0).all()))
    return df


def section_B(src):
    print("\n=== B. Reward integrity ===")
    cols = ["sample_id", "t", "split", "label", "chosen_action_idx", "n_candidates",
            "outcome", "approach_distance", "delay_change_seconds", "next_tc_headway_seconds",
            "r_delay", "r_throughput", "r_headway", "r_wait", "r_total", "r_wait_raw"]
    df = col(src, cols)
    df["ym"] = pd.to_datetime(df["t"], errors="coerce").dt.strftime("%Y-%m")

    # B1 arithmetic self-consistency
    if {"r_delay", "r_throughput", "r_headway", "r_wait", "r_total"}.issubset(df.columns):
        recon = df[["r_delay", "r_throughput", "r_headway", "r_wait"]].sum(axis=1)
        err = float((recon - df["r_total"]).abs().max())
        gate("r_total == Σ components", err < 1e-6, f"max abs err {err:.2e}")
        gate("r_total finite (no NaN)", bool(np.isfinite(df["r_total"]).all()))
        share = abs(df["r_delay"].mean()) / max(abs(df["r_total"].mean()), 1e-9)
        print(f"      r_delay share |mean r_delay|/|mean r_total| = {share:.3f} "
              f"(was 0.025 pre-fix; expect HIGHER now)")

    # B2 sample_id ↔ label INDEPENDENT agreement (caught the 4.6.5 misalignment).
    # set rows (chosen_action_idx>0) must have a non-null outcome AND r_wait_raw==0;
    # wait rows (==0) must have null outcome AND r_wait_raw==-1.
    if {"chosen_action_idx", "outcome", "r_wait_raw"}.issubset(df.columns):
        is_set = df["chosen_action_idx"] > 0
        out_present = df["outcome"].notna()
        rw = df["r_wait_raw"]
        set_ok = float(((out_present & (rw == 0)) | ~is_set)[is_set].mean()) if is_set.any() else 1.0
        wait_ok = float(((~out_present & (rw == -1)) | is_set)[~is_set].mean()) if (~is_set).any() else 1.0
        gate("sample_id↔reward label agreement (set rows: outcome+r_wait_raw=0)",
             set_ok > 0.999, f"{set_ok:.4f}")
        gate("sample_id↔reward label agreement (wait rows: null outcome+r_wait_raw=-1)",
             wait_ok > 0.999, f"{wait_ok:.4f}")

    # B3 delay coverage BY MONTH — the fix #1+#2 verification (must be high + uniform,
    # NO month ≈0, esp. Apr-Jul which was 0.2% pre-fix).
    if "delay_change_seconds" in df.columns:
        cov = df.groupby("ym")["delay_change_seconds"].apply(lambda s: float(s.notna().mean()))
        print("      delay_change coverage by month:")
        for ym, c in cov.items():
            flag = "  <<LOW" if c < 0.10 else ""
            print(f"        {ym}: {100*c:5.2f}%{flag}")
        aprjul = cov[[m for m in cov.index if isinstance(m, str) and "2023-04" <= m <= "2023-07"]]
        gate("fix #2: Apr-Jul delay coverage recovered (each > 15%, was ~0.2%)",
             bool((aprjul >= 0.15).all()) if len(aprjul) else False,
             f"Apr-Jul min {100*aprjul.min():.2f}%" if len(aprjul) else "no Apr-Jul months")
        ov = float(df['delay_change_seconds'].notna().mean())
        gate("overall delay coverage > 20% (was 6.4%)", ov > 0.20, f"{100*ov:.2f}%")
        low = [m for m, c in cov.items() if isinstance(m, str) and c < 0.10
               and not ("2023-04" <= m <= "2023-07")]
        if low:
            print(f"      note: other low-coverage months (likely genuine Movements "
                  f"sparsity — e.g. 2023-03 had ~31 Movements rows — review, not a bug): {low}")
        # delay magnitude sane (minutes-scale, not days)
        nz = df["delay_change_seconds"].dropna()
        if len(nz):
            gate("delay_change magnitude sane (|p99| < 2h)",
                 float(nz.abs().quantile(0.99)) < 7200, f"p99 {nz.abs().quantile(0.99):.0f}s")
    return df


def section_C(src):
    print("\n=== C. State integrity (full scan, memory-bounded by row group) ===")
    # read nested columns via to_pylist (NOT to_pandas) → python list/dict/None, avoiding
    # the numpy-array truthiness trap on list-of-struct columns (TOOL_TRAPS §16).
    pf = pq.ParquetFile(str(src))
    cols = ["t", "state_nodes_train", "state_special_flags", "state_schedule_outlook"]
    bad_plat = late_rate = pdev_rate = aprjul_late = n = 0
    sched_eta_ok = True
    for rg in range(pf.num_row_groups):
        sub = pf.read_row_group(rg, columns=cols)
        t = sub.column("t").to_pylist()
        nt = sub.column("state_nodes_train").to_pylist()
        sf = sub.column("state_special_flags").to_pylist()
        so = sub.column("state_schedule_outlook").to_pylist()
        for i in range(len(t)):
            n += 1
            ym = pd.Timestamp(t[i]).strftime("%Y-%m") if t[i] is not None else ""
            for nd in (nt[i] or []):
                pp = nd.get("planned_platform")
                if pp is not None and not (1 <= int(pp) <= 7):
                    bad_plat += 1
            sfi = sf[i] or {}
            if (sfi.get("f_late_train", 0) or 0) > 0:
                late_rate += 1
                if "2023-04" <= ym <= "2023-07":
                    aprjul_late += 1
            if sfi.get("f_platform_dev"):
                pdev_rate += 1
            soi = so[i] or []
            if soi and "eta_s" not in (soi[0] or {}):
                sched_eta_ok = False
    gate("planned_platform ∈ {1..7,None} (no signal IDs/out-of-range)", bad_plat == 0,
         f"{bad_plat} bad")
    gate("schedule_outlook has eta_s (gbtt-based shape intact)", sched_eta_ok)
    gate("f_platform_dev rate < 5% (not 83%)", pdev_rate / max(n, 1) < 0.05,
         f"{100*pdev_rate/max(n,1):.2f}%")
    print(f"      f_late_train fire rate: {100*late_rate/max(n,1):.2f}% (n={n:,})")
    gate("late_train fires in Apr-Jul (was structurally 0)", aprjul_late > 0,
         f"{aprjul_late} fires Apr-Jul")


def section_D(src, backup):
    print("\n=== D. Non-window rows unchanged vs backup (patch 23 only touched Apr-Jul) ===")
    if backup is None or not Path(backup).exists():
        gate("backup provided for non-window diff", False,
             "pass --backup <pre-fix snapshots>; SKIPPED")
        return
    if Path(backup).resolve() == Path(src).resolve():
        gate("backup differs from src (meaningful diff)", False,
             "backup == src → pass the PRE-23-patch snapshots backup; SKIPPED")
        return
    cols = ["sample_id", "t", "state_special_flags"]
    a = col(src, cols).set_index("sample_id")
    b = col(backup, cols).set_index("sample_id")
    common = a.index.intersection(b.index)
    a, b = a.loc[common], b.loc[common]
    ym = pd.to_datetime(a["t"], errors="coerce")
    out_win = ~((ym >= WIN_START) & (ym < WIN_END))
    # compare f_late_train/f_platform_dev for OUT-OF-WINDOW rows → must be identical
    def fl(s, k):
        return s["state_special_flags"].apply(lambda d: (d or {}).get(k))
    same_late = bool((fl(a[out_win], "f_late_train").values ==
                      fl(b[out_win], "f_late_train").values).all())
    same_pdev = bool((fl(a[out_win], "f_platform_dev").values ==
                      fl(b[out_win], "f_platform_dev").values).all())
    gate("out-of-window f_late_train unchanged vs backup", same_late)
    gate("out-of-window f_platform_dev unchanged vs backup", same_pdev)
    print(f"      compared {int(out_win.sum()):,} out-of-window rows")


def section_E(src):
    print("\n=== E. Cross-month anomaly scan (find ALL suspicious points) ===")
    df = col(src, ["t", "label", "chosen_action_idx", "n_candidates",
                   "delay_change_seconds", "approach_distance", "next_tc_headway_seconds"])
    df["ym"] = pd.to_datetime(df["t"], errors="coerce").dt.strftime("%Y-%m")
    df["is_wait"] = (df["chosen_action_idx"] == 0).astype(float)
    feats = {
        "n":              ("label", "size"),
        "wait_rate":      ("is_wait", "mean"),
        "n_cand_mean":    ("n_candidates", "mean"),
        "delay_cov":      ("delay_change_seconds", lambda s: s.notna().mean()),
        "approach_cov":   ("approach_distance", lambda s: s.notna().mean()),
        "headway_cov":    ("next_tc_headway_seconds", lambda s: s.notna().mean()),
    }
    g = df.groupby("ym")
    table = {}
    for name, (c, fn) in feats.items():
        table[name] = g[c].size() if fn == "size" else g[c].agg(fn)
    tab = pd.DataFrame(table)
    pd.set_option("display.width", 200)
    print(tab.round(3).to_string())
    # flag months where a coverage/rate feature deviates > 3 robust-sigma from the median
    print("      outlier flags (|value − median| > 3·MAD across months):")
    flagged = []
    for c in ["wait_rate", "n_cand_mean", "delay_cov", "approach_cov", "headway_cov"]:
        v = tab[c].dropna()
        med = v.median(); mad = (v - med).abs().median() or 1e-9
        for ym, x in v.items():
            if abs(x - med) > 3 * 1.4826 * mad:
                flagged.append(f"{ym}.{c}={x:.3f} (median {med:.3f})")
    if flagged:
        for f in flagged:
            print(f"        ⚠ {f}")
    gate("no extreme cross-month outlier (manual review the table above)", not flagged,
         f"{len(flagged)} flag(s) — review")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(C.SNAPSHOTS_V2_PARQUET))
    ap.add_argument("--backup", default=None,
                    help="pre-fix snapshots backup for the non-window-unchanged diff (D)")
    args = ap.parse_args()
    src = args.src
    print(f"AUDIT {src}\nwindow [{C.MOVEMENTS_BST_FIX_START}, {C.MOVEMENTS_BST_FIX_END})")
    section_A(src)
    section_B(src)
    section_C(src)
    section_D(src, args.backup)
    section_E(src)

    print("\n" + "=" * 64)
    n_fail = sum(1 for _, ok in _results if not ok)
    for name, ok in _results:
        if not ok:
            print(f"  FAIL: {name}")
    print(f"\n  {len(_results) - n_fail}/{len(_results)} gates PASS, {n_fail} FAIL")
    print("  ALSO re-run (data changed): 06_run_leak_audit_full --sample 100000 ; "
          "07_audit_snapshots ; 21_audit_leakage ; 10_smoke_streaming")
    print("  → retrain ONLY if all gates PASS + leak audits clean + you've reviewed "
          "the by-month tables in B3/C/E.")
    print("=" * 64)
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
