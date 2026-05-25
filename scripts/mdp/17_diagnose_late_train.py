"""Stage 4.7.2d 旁路诊断 — 查 late_train=0.00% 的根因。

smoke [D] 发现 stratum `late_train`（f_late_train>0）占比 0.00%。本脚本查清是：
  (1) f_late_train（state_special_flags 里的标量）—— 是没填(全 null)？还是填了但从不>0？
  (2) scheduled_delta_s（focal train 节点特征）—— 上游"计划-实际"差是否有数据/有正值？
      （即便 flag 恒 0，看 train 节点是否仍带晚点信息。）

据此判断：是 flag 计算 bug（该修），还是上游 scheduled/TRUST 数据本就稀疏（已知局限）。
只读，不改文件。Windows:
    python scripts/mdp/17_diagnose_late_train.py
    python scripts/mdp/17_diagnose_late_train.py --sample-rows 200000
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import pyarrow.parquet as pq
import pyarrow.compute as pc

from railrl import config as C


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-rows", type=int, default=200000,
                    help="抽样多少行做 focal scheduled_delta_s 检查（前 N 行）")
    args = ap.parse_args()
    src = C.SNAPSHOTS_V2_PARQUET
    print(f"读 {src}")

    # ---- (1) f_late_train（全量，标量 struct 字段，便宜）----
    print("\n[1] f_late_train（state_special_flags 标量）")
    sf = pq.read_table(str(src), columns=["state_special_flags"]).column("state_special_flags")
    fields = set(sf.type.names) if hasattr(sf.type, "names") else set(sf.combine_chunks().type.names)
    print(f"    special_flags 字段: {sorted(fields)}")
    if "f_late_train" not in fields:
        print("    [!] 没有 f_late_train 字段——flag 未写进 schema。")
    else:
        lt = pc.struct_field(sf, "f_late_train")
        nnull = lt.null_count
        vals = np.array([x for x in lt.to_pylist() if x is not None], dtype=float)
        n = len(lt)
        print(f"    null: {nnull:,}/{n:,} ({100*nnull/n:.2f}%)")
        if len(vals):
            print(f"    非null n={len(vals):,}  min={vals.min():.1f}  max={vals.max():.1f}  "
                  f"mean={vals.mean():.2f}")
            print(f"    >0 (晚点) 占比(非null): {(vals>0).mean():.4f}；占全体: {(vals>0).sum()/n:.4f}")
            print(f"    ==0 占比(非null): {(vals==0).mean():.4f}；<0 占比: {(vals<0).mean():.4f}")
        else:
            print("    全部为 null —— f_late_train 从没被写入（flag bug）。")

    # ---- (2) focal train 的 scheduled_delta_s（抽样，嵌套需迭代）----
    print(f"\n[2] focal train 的 scheduled_delta_s（前 {args.sample_rows:,} 行抽样）")
    pf = pq.ParquetFile(str(src))
    got = 0
    deltas = []
    focal_field_ok = None
    for rg in range(pf.num_row_groups):
        if got >= args.sample_rows:
            break
        tb = pf.read_row_group(rg, columns=["state_nodes_train"]).column("state_nodes_train").to_pylist()
        for nodes in tb:
            got += 1
            if not nodes:
                continue
            focal = None
            for nd in nodes:
                if nd.get("is_focal"):
                    focal = nd
                    break
            if focal is None:
                continue
            if focal_field_ok is None:
                focal_field_ok = "scheduled_delta_s" in focal
                print(f"    focal train 节点字段示例: {sorted(focal.keys())}")
            deltas.append(focal.get("scheduled_delta_s"))
            if got >= args.sample_rows:
                break
    arr = np.array([d for d in deltas if d is not None], dtype=float)
    nnone = sum(1 for d in deltas if d is None)
    print(f"    抽样 focal 行 {len(deltas):,}；scheduled_delta_s null: {nnone:,} "
          f"({100*nnone/max(len(deltas),1):.2f}%)")
    if len(arr):
        print(f"    非null n={len(arr):,}  min={arr.min():.1f}  max={arr.max():.1f}  "
              f"mean={arr.mean():.2f}  >0(晚点)占比={ (arr>0).mean():.4f}")
    else:
        print("    全部 null —— train 节点也没有 scheduled_delta_s（上游没算）。")

    print("\n判读：")
    print("  · f_late_train 全 null → flag 计算没接上（bug，可修）。")
    print("  · f_late_train 有值但从不>0 + scheduled_delta_s 也无正值 → 上游 scheduled/TRUST 稀疏（已知局限，非 bug）。")
    print("  · scheduled_delta_s 有正值但 f_late_train 恒 0 → flag 逻辑/取数错（bug，可修）。")
    return 0


if __name__ == "__main__":
    main()
