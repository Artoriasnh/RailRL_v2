"""Stage 4.7.2d 旁路诊断 — 对比两个 snapshot 文件的 8 个 special flag 原始触发率。

用途：smoke [D] 显示 patch(18) 后 stratum 分布大变（platform_dev 67%→0.15%、
priority_compete 2.45%→16.79%、trivial 5.9%→44%）。若只改了 f_late_train，这些**不可能
上升**——怀疑 18 的 struct 重写 pa.array(sf, type=sf_type) 改坏了其它 flag。本脚本逐 flag
报原始触发率，对比 patched 文件 vs 未动过的备份，定位是否 patch 误改。

只读。Windows:
    python scripts/mdp/19_diagnose_flags.py                       # 默认对比 snapshots_v2 vs prereward
    python scripts/mdp/19_diagnose_flags.py A.parquet B.parquet   # 自定两个文件
    python scripts/mdp/19_diagnose_flags.py A.parquet             # 只报一个
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import pyarrow.parquet as pq
import pyarrow.compute as pc

from railrl import config as C

FLAGS = ["f_advance", "f_call_on", "f_platform_dev", "f_priority_compete",
         "f_late_train", "f_unusual_id", "f_trts_pressed", "f_freight_class"]


def rates(path: str) -> dict:
    sf = pq.read_table(path, columns=["state_special_flags"]).column("state_special_flags")
    present = set(sf.type.names) if hasattr(sf.type, "names") else set(
        sf.combine_chunks().type.names)
    n = len(sf)
    out = {"_n": n}
    for f in FLAGS:
        if f not in present:
            out[f] = None
            continue
        a = pc.struct_field(sf, f)
        a = pc.fill_null(a, 0)
        arr = np.asarray(a.to_pylist())
        # bool flags → True 率；f_late_train(int 秒) → >0 率
        if f == "f_late_train":
            out[f] = float(np.mean(np.asarray(arr, dtype=float) > 0))
        else:
            out[f] = float(np.mean(np.asarray(arr, dtype=bool)))
    return out


def main() -> int:
    args = sys.argv[1:]
    if len(args) == 0:
        a = str(C.SNAPSHOTS_V2_PARQUET)
        b = str(C.SNAPSHOTS_V2_PARQUET.parent / "snapshots_v2.prereward.parquet")
    elif len(args) == 1:
        a, b = args[0], None
    else:
        a, b = args[0], args[1]

    print(f"A = {a}")
    ra = rates(a)
    rb = rates(b) if (b and Path(b).exists()) else None
    if b and rb is None:
        print(f"[warn] B 不存在: {b}（只报 A）")
    if rb is not None:
        print(f"B = {b}")

    print(f"\n{'flag':<20s} {'A':>10s}" + (f" {'B':>10s} {'Δ(A−B)':>10s}" if rb else ""))
    for f in FLAGS:
        va = ra.get(f)
        sa = "n/a" if va is None else f"{100*va:7.3f}%"
        line = f"{f:<20s} {sa:>10s}"
        if rb is not None:
            vb = rb.get(f)
            sb = "n/a" if vb is None else f"{100*vb:7.3f}%"
            dd = "" if (va is None or vb is None) else f"{100*(va-vb):+8.3f}pp"
            line += f" {sb:>10s} {dd:>10s}"
        print(line)
    print(f"\nA n={ra['_n']:,}" + (f"  B n={rb['_n']:,}" if rb else ""))
    if rb is not None:
        print("\n判读：除 f_late_train 外，A 与 B 的 flag 触发率应**几乎相同**（patch 只该改 "
              "f_late_train + scheduled_delta_s）。若 f_platform_dev/f_priority_compete 等"
              "明显不同 → 18 的 struct 重写改坏了别的 flag（patch bug，需修重写逻辑）。")
    return 0


if __name__ == "__main__":
    main()
