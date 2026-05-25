"""Stage 3 — Numerical sanity audit of snapshots_v2.parquet.

Checks VALUE DISTRIBUTIONS (not just "non-null/shape"), per the lesson that
shape-only checks missed both the empty action space and the us/ns time bug.
Run after building snapshots; paste the output.

Usage:
    python scripts/mdp/07_audit_snapshots.py
"""
from __future__ import annotations
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from railrl import config as C

BANNED = {"focal_signal", "chosen_route_id", "is_focal_signal", "is_focal_route",
          "r_total", "delay_change_seconds", "next_tc_headway_seconds", "is_chosen"}


def main():
    path = C.SNAPSHOTS_V2_PARQUET
    pf = pq.ParquetFile(path)
    n_rows = pf.metadata.num_rows
    n_rg = pf.num_row_groups
    print("=" * 72)
    print(f"AUDIT {path}")
    print(f"rows={n_rows:,}  row_groups={n_rg}  columns={len(pf.schema_arrow.names)}")
    print("=" * 72)

    # Sample ~every 8th row group
    rgs = list(range(0, n_rg, 8))
    deltas = []
    occ_frac = []          # occupancy_fraction_5m
    asp_frac = []          # aspect_fraction_red_5m
    nchg = []              # n_state_changes_5m
    last_age = []          # last_change_age_s
    tk = []; sg = []; rt = []; tn = []
    nc = []; so = []; cidx = []
    occ_now = []           # occupied_now
    occupier_nonnull = []
    bad_center = bad_focal = bad_plat = banned_hit = 0
    labels = Counter()

    for rg in rgs:
        df = pf.read_row_group(rg, columns=[
            "state_event_tokens", "state_nodes_track", "state_nodes_signal",
            "state_nodes_route", "state_nodes_train", "n_candidates",
            "state_schedule_outlook", "label", "chosen_action_idx",
            "state_center"]).to_pandas()
        labels.update(df["label"].tolist())
        tk.append(df["state_nodes_track"].apply(len).to_numpy())
        sg.append(df["state_nodes_signal"].apply(len).to_numpy())
        rt.append(df["state_nodes_route"].apply(len).to_numpy())
        tn.append(df["state_nodes_train"].apply(len).to_numpy())
        nc.append(df["n_candidates"].to_numpy())
        so.append(df["state_schedule_outlook"].apply(len).to_numpy())
        cidx.append(df["chosen_action_idx"].to_numpy())
        for toks in df["state_event_tokens"]:
            for t in toks:
                deltas.append(t["time_delta_s"])
        for tracks in df["state_nodes_track"]:
            for n in tracks:
                occ_frac.append(n.get("occupancy_fraction_5m", 0.0))
                nchg.append(n.get("n_state_changes_5m", 0))
                last_age.append(n.get("last_change_age_s", 0))
                occ_now.append(1 if n.get("occupied_now") else 0)
                occupier_nonnull.append(1 if n.get("current_occupier_train_id") else 0)
        for sigs in df["state_nodes_signal"]:
            for n in sigs:
                asp_frac.append(n.get("aspect_fraction_red_5m", 0.0))
        # leak checks
        for i in range(len(df)):
            c = df["state_center"].iloc[i]
            if not isinstance(c, dict) or c.get("type") != "track":
                bad_center += 1
            tnodes = df["state_nodes_train"].iloc[i]
            if sum(1 for t in tnodes if t.get("is_focal")) != 1:
                bad_focal += 1
            for s in df["state_schedule_outlook"].iloc[i]:
                p = s.get("planned_platform")
                if p is not None and not (1 <= int(p) <= 7):
                    bad_plat += 1
            for node in list(df["state_nodes_signal"].iloc[i]) + list(df["state_nodes_route"].iloc[i]):
                if BANNED & set(node.keys()):
                    banned_hit += 1
                    break

    tk = np.concatenate(tk); sg = np.concatenate(sg); rt = np.concatenate(rt); tn = np.concatenate(tn)
    nc = np.concatenate(nc); so = np.concatenate(so); cidx = np.concatenate(cidx)
    d = np.array(deltas); of = np.array(occ_frac); af = np.array(asp_frac)
    ch = np.array(nchg); la = np.array(last_age)
    n = len(tk)

    print(f"\nsampled {n:,} snapshots  | label={dict(labels)}")

    print("\n--- ⭐ TIME features (the us/ns bug) ---")
    print(f"event time_delta_s: min={d.min():.0f} med={np.median(d):.0f} max={d.max():.0f}  "
          f"%<1h={100*(d<3600).mean():.1f}% %<1d={100*(d<86400).mean():.1f}% %>1e8(garbage)={100*(d>1e8).mean():.3f}%")
    print(f"occupancy_fraction_5m: min={of.min():.3f} med={np.median(of):.3f} max={of.max():.3f} "
          f"%==0={100*(of==0).mean():.1f}% %in(0,1)={100*((of>0)&(of<1)).mean():.1f}%")
    print(f"aspect_fraction_red_5m: med={np.median(af):.3f} max={af.max():.3f} %>0={100*(af>0).mean():.1f}%")
    print(f"n_state_changes_5m: med={np.median(ch):.0f} max={ch.max()} %>0={100*(ch>0).mean():.1f}%")
    print(f"last_change_age_s: med={np.median(la):.0f}s max={la.max():.0f}s")
    print(f"occupied_now: {100*np.mean(occ_now):.1f}% of track-nodes occupied")
    print(f"current_occupier non-null: {100*np.mean(occupier_nonnull):.1f}% of track-nodes")

    print("\n--- subgraph + action space ---")
    print(f"track nodes: mean={tk.mean():.1f} max={tk.max()} | ==1 degenerate={100*(tk==1).mean():.1f}%")
    print(f"signal nodes: mean={sg.mean():.1f} max={sg.max()} | route: mean={rt.mean():.1f} max={rt.max()}")
    print(f"train nodes: mean={tn.mean():.2f} max={tn.max()} | >1 multi={100*(tn>1).mean():.1f}%")
    print(f"caps ok: track<=60={tk.max()<=60} signal<=15={sg.max()<=15} route<=15={rt.max()<=15} train<=8={tn.max()<=8}")
    print(f"n_candidates: mean={nc.mean():.2f} max={nc.max()} | ==0 wait-only={100*(nc==0).mean():.1f}%")
    print(f"chosen_action_idx: wait(0)={100*(cidx==0).mean():.1f}% set(>=1)={100*(cidx>=1).mean():.1f}% invalid(-1)={(cidx==-1).sum()}")
    print(f"schedule_outlook nonzero: {100*(so>0).mean():.1f}%")

    print("\n--- leak spot-check ---")
    print(f"center not track: {bad_center} | !=1 focal: {bad_focal} | platform∉1-7: {bad_plat} | banned in state: {banned_hit}")
    leak_ok = (bad_center == 0 and bad_focal == 0 and bad_plat == 0 and banned_hit == 0)
    time_ok = (d > 1e8).mean() < 0.001 and ((of > 0) & (of < 1)).mean() > 0.01

    print("\n" + "=" * 72)
    print(f"LEAK: {'PASS' if leak_ok else 'FAIL'}   |   TIME features: {'PASS' if time_ok else 'FAIL'}")
    print("VERDICT:", "READY FOR STAGE 4 ✅" if (leak_ok and time_ok) else "NOT READY ❌")
    print("=" * 72)


if __name__ == "__main__":
    main()
