"""Stage 4.7.2d — patch lateness（scheduled_delta_s + f_late_train）on canonical 文件。

修复见 IMPLEMENTATION_LOG 4.7.2d lateness bug：旧 scheduled_delta_s = "到下一个未来 gbtt
事件的秒数"（恒≥0、reused headcode 取到远 occurrence → 276 天垃圾、f_late_train 永不触发）。

本脚本用**修正后的** MovementsLookup（state_history.scheduled_delta_s 现走 current_lateness_s：
realized timetable_variation×60×sign(variation_status), actual_ts≤t, 6h 窗口, leak-safe）
重算两个字段，**保序重写**，其余列/episode/reward/sample_id 全不动：
  - state_nodes_train[].scheduled_delta_s（每个 train 节点）
  - state_special_flags.f_late_train（focal 节点的 lateness 经 f_late_train 阈值）

输出新文件 snapshots_v2.lateness.parquet（不覆盖）；验证后改名为 snapshots_v2.parquet。
之后重跑 01(normalization) + 16(stratum) + smoke。内存有界（逐 row group）。

Windows:
    python scripts/mdp/18_patch_lateness.py
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
from railrl.mdp.special_flags import f_late_train


def main() -> int:
    src = C.SNAPSHOTS_V2_PARQUET
    out = src.parent / "snapshots_v2.lateness.parquet"

    print("[1/4] load Movements + build MovementsLookup（含修正后的 lateness 查表）")
    mv = load_movements()
    lk = MovementsLookup.build(mv, train_id_col="auto")
    print(f"      lateness headcodes: {len(lk.train_to_lateness):,}")
    if not lk.train_to_lateness:
        raise SystemExit("[ERROR] lateness 查表为空——检查 Movements 是否有 "
                         "actual_timestamp/variation_status/timetable_variation 列。")

    pf = pq.ParquetFile(str(src))
    schema = pf.schema_arrow
    nt_type = schema.field("state_nodes_train").type
    sf_type = schema.field("state_special_flags").type
    nt_idx = schema.get_field_index("state_nodes_train")
    sf_idx = schema.get_field_index("state_special_flags")

    print(f"[2/4] 逐 row group 重算 scheduled_delta_s + f_late_train（{pf.num_row_groups} 组）")
    writer = None
    n_total = n_late = 0
    focal_deltas: list[int] = []
    for rg in range(pf.num_row_groups):
        tbl = pf.read_row_group(rg)
        t_ns = pd.to_datetime(tbl.column("t").to_pylist()).values.astype(
            "datetime64[ns]").astype("int64")
        nt = tbl.column("state_nodes_train").to_pylist()
        sf = tbl.column("state_special_flags").to_pylist()
        for i in range(len(nt)):
            ti = int(t_ns[i])
            focal_delta = 0
            for nd in (nt[i] or []):
                d = int(lk.current_lateness_s(nd.get("train_id"), ti))
                nd["scheduled_delta_s"] = d
                if nd.get("is_focal"):
                    focal_delta = d
            if sf[i] is not None:
                fl = int(f_late_train(focal_delta))
                sf[i]["f_late_train"] = fl
                if fl > 0:
                    n_late += 1
            focal_deltas.append(focal_delta)
            n_total += 1
        tbl = tbl.set_column(nt_idx, "state_nodes_train", pa.array(nt, type=nt_type))
        tbl = tbl.set_column(sf_idx, "state_special_flags", pa.array(sf, type=sf_type))
        if writer is None:
            writer = pq.ParquetWriter(str(out), tbl.schema, compression="zstd")
        writer.write_table(tbl, row_group_size=5000)
        if rg % 50 == 0:
            print(f"      rg {rg}/{pf.num_row_groups}  ({n_total:,} 行)", flush=True)
    if writer is not None:
        writer.close()

    print("[3/4] 验证（focal current_lateness_s 分布）")
    d = np.array(focal_deltas, dtype=np.int64)
    print(f"      focal lateness(秒)  >0(晚点) {np.mean(d > 0):.4f} / <0(早到) {np.mean(d < 0):.4f} "
          f"/ ==0 {np.mean(d == 0):.4f}")
    if (d != 0).any():
        nz = d[d != 0]
        print(f"      非零绝对值 min/median/max = {np.abs(nz).min()} / "
              f"{int(np.median(np.abs(nz)))} / {np.abs(nz).max()} 秒（应是分钟×60、远小于旧 23.8M）")
    print(f"      f_late_train(focal)>0 占比: {n_late/max(n_total,1):.4f}（修复前 0.0000）")

    print(f"[4/4] -> {out.name}（{n_total:,} 行，保序）")
    print(f"\n核对无误后：改名 {out.name} → snapshots_v2.parquet（原文件留备份），"
          "再重跑 01_build_normalization_stats + 16_build_stratum_labels + 10_smoke_streaming。")
    return 0


if __name__ == "__main__":
    main()
