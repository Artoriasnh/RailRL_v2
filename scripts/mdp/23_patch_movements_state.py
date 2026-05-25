"""fix #2 — patch Movements-derived STATE for the Apr-Jul 2023 +1h window.

After load_movements() corrects the +1h Apr-Jul Movements clock (config
MOVEMENTS_BST_FIX_*), this re-derives the Movements-dependent snapshot fields for
the affected rows ONLY (decision t in [MOVEMENTS_BST_FIX_START, _END)), reusing
the SAME MovementsLookup methods + flag functions + schedule_outlook transform
that state.build_snapshot uses (so values match exactly). Rows outside the window
are left byte-for-byte equivalent (values untouched).

Recomputed (5 of the 6 Movements-derived fields):
  - state_nodes_train[].planned_platform   (lk.planned_platform)
  - state_nodes_train[].scheduled_delta_s  (lk.current_lateness_s; signed lateness)
  - state_schedule_outlook                 (lk.schedule_outlook + _build_schedule_outlook xform)
  - state_special_flags.f_late_train       (f_late_train of focal scheduled_delta_s)
  - state_special_flags.f_platform_dev     (f_platform_dev of candidate end_platforms vs corrected focal planned_platform)

NOT recomputed: state_special_flags.f_trts_pressed — needs TD-derived
trts_state_by_platform which is NOT stored in the snapshot (only build_snapshot
has it). Its residual error is tiny (1/8 flag, only planned_platform input
affected, TRTS-pressed rare) and is documented as a known limitation. See
IMPLEMENTATION_LOG 2026-05-24 "fix #2".

Writes snapshots_v2.movstate.parquet (保序). After verifying, rename to
snapshots_v2.parquet (keep backup), then re-run reward (08→09→10) + 01(norm) +
16(stratum). Run on Windows:
    python scripts/mdp/23_patch_movements_state.py
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from railrl import config as C
from railrl.data_io import load_movements
from railrl.mdp.state_history import MovementsLookup
from railrl.mdp.special_flags import f_late_train, f_platform_dev

_HC_DIGITS = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"}


def build_outlook(lk, t_ns: int, focal_train: str) -> list[dict]:
    """Verbatim copy of state.SnapshotBuilder._build_schedule_outlook (so the
    patched struct matches a fresh build exactly)."""
    raw = lk.schedule_outlook(
        t_ns, k=C.SCHEDULE_OUTLOOK_TOPK,
        lookahead_s=C.SCHEDULE_LOOKAHEAD_MIN * 60.0,
        exclude_train=focal_train,
    )
    out = []
    for r in raw:
        tr = str(r.get("train_id", ""))
        hc = tr[0] if tr and len(tr) >= 4 else "non_standard"
        if hc not in _HC_DIGITS:
            hc = "non_standard"
        elif hc in {"7", "8"}:
            hc = "other"
        out.append({
            "train_id":         tr,
            "headcode_class":   hc,
            "eta_s":            int(r.get("gbtt_delta_s", 0)),
            "planned_platform": r.get("planned_platform"),
            "event_type":       str(r.get("event_type", "")),
        })
    return out


def main() -> int:
    src = C.SNAPSHOTS_V2_PARQUET
    out = src.with_name("snapshots_v2.movstate.parquet")
    win_start = pd.Timestamp(C.MOVEMENTS_BST_FIX_START)
    win_end = pd.Timestamp(C.MOVEMENTS_BST_FIX_END)

    print("[1/4] load CORRECTED Movements + build MovementsLookup")
    mv = load_movements()                      # applies the −1h Apr-Jul fix
    lk = MovementsLookup.build(mv, train_id_col="auto")
    print(f"      lateness headcodes: {len(lk.train_to_lateness):,}")

    pf = pq.ParquetFile(str(src))
    schema = pf.schema_arrow
    nt_type = schema.field("state_nodes_train").type
    so_type = schema.field("state_schedule_outlook").type
    sf_type = schema.field("state_special_flags").type
    nt_idx = schema.get_field_index("state_nodes_train")
    so_idx = schema.get_field_index("state_schedule_outlook")
    sf_idx = schema.get_field_index("state_special_flags")

    print(f"[2/4] recompute window rows only ({pf.num_row_groups} row groups; "
          f"window [{C.MOVEMENTS_BST_FIX_START}, {C.MOVEMENTS_BST_FIX_END}))")
    writer = None
    n_total = n_win = 0
    pp_changed = so_changed = 0
    fl_before = fl_after = pd_before = pd_after = 0
    for rg in range(pf.num_row_groups):
        tbl = pf.read_row_group(rg)
        t_dt = pd.to_datetime(tbl.column("t").to_pylist())
        t_ns = t_dt.values.astype("datetime64[ns]").astype("int64")
        in_win = (t_dt >= win_start) & (t_dt < win_end)
        focal = tbl.column("focal_train").to_pylist()
        nt = tbl.column("state_nodes_train").to_pylist()
        so = tbl.column("state_schedule_outlook").to_pylist()
        sf = tbl.column("state_special_flags").to_pylist()
        rt = tbl.column("state_nodes_route").to_pylist()
        for i in range(len(nt)):
            n_total += 1
            if not bool(in_win[i]):
                continue
            n_win += 1
            ti = int(t_ns[i])
            focal_pp = None
            focal_delta = 0
            for nd in (nt[i] or []):
                tid = nd.get("train_id")
                pp_new = lk.planned_platform(tid, ti)
                if nd.get("planned_platform") != pp_new:
                    pp_changed += 1
                nd["planned_platform"] = pp_new
                nd["scheduled_delta_s"] = int(lk.current_lateness_s(tid, ti))
                if nd.get("is_focal"):
                    focal_pp = pp_new
                    focal_delta = int(nd["scheduled_delta_s"])
            new_so = build_outlook(lk, ti, focal[i])
            if new_so != so[i]:
                so_changed += 1
            so[i] = new_so
            if sf[i] is not None:
                fl_before += int(bool(sf[i].get("f_late_train")))
                pd_before += int(bool(sf[i].get("f_platform_dev")))
                sf[i]["f_late_train"] = int(f_late_train(focal_delta))
                cand = [r.get("end_platform_id") for r in (rt[i] or [])
                        if r.get("in_candidate_set")]
                sf[i]["f_platform_dev"] = bool(f_platform_dev(cand, focal_pp))
                fl_after += int(bool(sf[i]["f_late_train"]))
                pd_after += int(bool(sf[i]["f_platform_dev"]))
                # f_trts_pressed deliberately untouched (needs TD trts_state, not stored)
        tbl = tbl.set_column(nt_idx, "state_nodes_train", pa.array(nt, type=nt_type))
        tbl = tbl.set_column(so_idx, "state_schedule_outlook", pa.array(so, type=so_type))
        tbl = tbl.set_column(sf_idx, "state_special_flags", pa.array(sf, type=sf_type))
        if writer is None:
            writer = pq.ParquetWriter(str(out), tbl.schema, compression="zstd")
        writer.write_table(tbl, row_group_size=5000)
        if rg % 50 == 0:
            print(f"      rg {rg}/{pf.num_row_groups}  ({n_total:,} rows, {n_win:,} in-window)",
                  flush=True)
    if writer is not None:
        writer.close()

    print("[3/4] self-check (window rows only)")
    print(f"      rows total={n_total:,}  in-window recomputed={n_win:,} "
          f"({100*n_win/max(n_total,1):.1f}%)")
    print(f"      focal/train planned_platform values changed: {pp_changed:,}")
    print(f"      schedule_outlook structs changed:            {so_changed:,}")
    print(f"      f_late_train (in-window)  before {fl_before:,} → after {fl_after:,}")
    print(f"      f_platform_dev (in-window) before {pd_before:,} → after {pd_after:,}")
    print("      (non-window rows untouched; f_trts_pressed untouched everywhere)")

    print(f"[4/4] -> {out.name}  (保序, {n_total:,} rows)")
    print("\nVerify, then: rename → snapshots_v2.parquet (keep backup) → re-run "
          "reward 08→09→10 + 01_build_normalization_stats + 16_build_stratum_labels.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
