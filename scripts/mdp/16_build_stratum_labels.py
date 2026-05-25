"""Stage 4.7.2d — 分层采样标签 sidecar（spec 04 §4.4）。

按 state_special_flags 给每个决策打一个 stratum 标签，供流式 loader 做"块级近似分层
采样"——否则 ~85% 的 trivial 决策会淹没稀有但关键的特殊情况（call-on / 晚点 / 改站台 …）。

stratum（优先级 late > advance > call_on > platform_dev > priority_compete > unusual_id > trivial）：
    0 late_train       f_late_train > 0
    1 advance          f_advance
    2 call_on          f_call_on
    3 platform_dev     f_platform_dev
    4 priority_compete f_priority_compete
    5 unusual_id       f_unusual_id
    6 trivial          以上都不满足
（f_trts_pressed / f_freight_class 不作为 stratum。）

输出：
  - stratum_labels.parquet : [sample_id, stratum]（loader 按 sample_id 查）
  - stratum_weights.json   : 每 stratum 在 **train split** 的频数 + 采样权重 1/sqrt(freq)

只读 [sample_id, state_special_flags, split]（小列），不改任何文件。
Windows:
    python scripts/mdp/16_build_stratum_labels.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import pyarrow.parquet as pq
import pyarrow.compute as pc

from railrl import config as C

# stratum 名 → code（按优先级，0 最高）
NAMES = ["late_train", "advance", "call_on", "platform_dev",
         "priority_compete", "unusual_id", "trivial"]
TRIVIAL = 6


def main() -> int:
    src = C.SNAPSHOTS_V2_PARQUET
    out_labels = src.parent / "stratum_labels.parquet"
    out_weights = src.parent / "stratum_weights.json"

    print(f"[1/4] 读 [sample_id, state_special_flags, split] ← {src.name}")
    tbl = pq.read_table(str(src), columns=["sample_id", "state_special_flags", "split"])
    n = tbl.num_rows
    sid = tbl.column("sample_id").to_numpy()
    split = np.asarray(tbl.column("split").to_pylist(), dtype=object)
    sf = tbl.column("state_special_flags")
    present = set(sf.type.names) if hasattr(sf.type, "names") else set(
        sf.combine_chunks().type.names)
    print(f"      {n:,} 行；special_flags 字段: {sorted(present)}")

    def boolfield(name):
        if name not in present:
            return np.zeros(n, dtype=bool)
        a = pc.struct_field(sf, name)
        a = pc.fill_null(a, False)
        return np.asarray(a.to_pylist(), dtype=bool)

    def intfield(name):
        if name not in present:
            return np.zeros(n, dtype=np.float64)
        a = pc.struct_field(sf, name)
        a = pc.fill_null(a, 0)
        return np.asarray(a.to_pylist(), dtype=np.float64)

    print("[2/4] 计算 stratum（按优先级覆盖）")
    late = intfield("f_late_train") > 0
    adv = boolfield("f_advance")
    callon = boolfield("f_call_on")
    pdev = boolfield("f_platform_dev")
    pc_ = boolfield("f_priority_compete")
    unus = boolfield("f_unusual_id")

    stratum = np.full(n, TRIVIAL, dtype=np.int8)   # 默认 trivial
    # 从低优先级到高优先级赋值，高优先级最后覆盖
    stratum[unus] = 5
    stratum[pc_] = 4
    stratum[pdev] = 3
    stratum[callon] = 2
    stratum[adv] = 1
    stratum[late] = 0

    print("[3/4] 写 stratum_labels.parquet + stratum_weights.json")
    import pyarrow as pa
    lab = pa.table({"sample_id": pa.array(sid), "stratum": pa.array(stratum)})
    pq.write_table(lab, str(out_labels), compression="zstd")

    # 权重在 TRAIN split 上算（采样只发生在 train；val/test 不重采样）
    is_train = (split == "train")
    weights = {}
    print("\n      stratum 分布（全体 / train）+ 权重 1/sqrt(train_freq)：")
    for code, name in enumerate(NAMES):
        all_cnt = int((stratum == code).sum())
        tr_cnt = int(((stratum == code) & is_train).sum())
        w = 1.0 / np.sqrt(tr_cnt) if tr_cnt > 0 else 0.0
        weights[str(code)] = {"name": name, "train_freq": tr_cnt, "weight": w}
        print(f"        {code} {name:<16s} all={all_cnt:>9,} ({100*all_cnt/n:5.2f}%)  "
              f"train={tr_cnt:>9,}  w={w:.3e}")
    out_weights.write_text(json.dumps(weights, indent=2))

    print(f"\n[4/4] -> {out_labels.name}  + {out_weights.name}")
    print("下一步：流式 loader 用 stratified=True 读这两个文件做块级近似分层采样。")
    return 0


if __name__ == "__main__":
    main()
