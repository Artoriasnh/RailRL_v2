"""Stage 4.7.2d — 把 snapshots 重排成 canonical 文件（内存有界的流式外排）。

把两件事烤进一个干净文件：
  (1) 用 episodes_v2.parquet（14 产出）**替换** snapshots 过时的 4 个 episode 列
      (pass_id / episode_idx / position_in_episode / is_last_in_episode) + 新增 split 列。
  (2) 按新 episode 顺序 **(episode_idx, position_in_episode)** 全局重排行（= (focal_train, t)）。
      → 每个 episode 的行连续、按 position 升序 → 流式 loader 顺序读即得 (s, s')。

state / reward / sample_id / 其余所有列原样 carried（只换 4 列 + 加 split + 改行序）。
⚠️ 写到**新文件** snapshots_v2.canonical.parquet（不覆盖原文件）；验证后由 Hao 改名。

== 内存（重要，第一版在此翻车，见 TOOL_TRAPS §14）==
snapshots 嵌套列多，整表 read_table + sort_by 解码后膨胀十几 GB、再复制一份 → 31GB 机器
OOM（连 PyCharm JVM 一起崩）。本版改 **流式 bucket 外排**，峰值仅几百 MB：
  Pass 1：逐 row group 读 → 换列 → 按 episode_idx 分到 N 个 bucket 临时文件。
  Pass 2：按 bucket 顺序读回 → 桶内排序 → 追加写最终文件。
临时文件写到 outputs/snapshots/_resort_tmp/（本地 4TB 盘够）。

Windows（建议用独立 PowerShell 终端跑，别在 PyCharm 内置运行器，少和 IDE 抢内存）：
    python scripts/mdp/15_resort_snapshots_canonical.py
    python scripts/mdp/15_resort_snapshots_canonical.py --buckets 96   # 更省内存（桶更小）
"""
from __future__ import annotations
import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from railrl import config as C

OLD_EP_COLS = ["pass_id", "episode_idx", "position_in_episode", "is_last_in_episode"]
NEW_COLS = ["pass_id", "episode_idx", "position_in_episode", "is_last_in_episode", "split"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--buckets", type=int, default=64,
                    help="外排桶数；越多每桶越小越省内存（默认 64）")
    args = ap.parse_args()
    NB = max(1, args.buckets)

    src = C.SNAPSHOTS_V2_PARQUET
    side_path = src.parent / "episodes_v2.parquet"
    out = src.parent / "snapshots_v2.canonical.parquet"
    tmp = src.parent / "_resort_tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    print(f"[1/4] 读 sidecar {side_path.name}（小表，可整载）")
    side = pq.read_table(str(side_path))          # sample_id, pass_id, episode_idx, position, is_last, split
    n_side = side.num_rows
    sid_side = side.column("sample_id").to_numpy()
    lut = {int(s): i for i, s in enumerate(sid_side)}     # sample_id → sidecar 行号
    max_eidx = int(side.column("episode_idx").to_numpy().max())
    denom = max_eidx + 1
    print(f"      {n_side:,} 行, max episode_idx={max_eidx:,}, 桶数={NB}")

    pf = pq.ParquetFile(str(src))
    n = pf.metadata.num_rows
    assert n == n_side, f"snapshot 行数 {n} != sidecar {n_side}"
    # 丢掉将要重建的列（含 split）——这样脚本对"已含 split 的 canonical 文件"也幂等，
    # 重跑不会产生重复 split 列。
    keep_cols = [c for c in pf.schema_arrow.names if c not in NEW_COLS]

    print(f"[2/4] Pass 1：逐 row group 换列 + 分桶 → {tmp.name}/ "
          f"（{pf.num_row_groups} 组）")
    writers: dict[int, pq.ParquetWriter] = {}
    out_schema = None
    seen = 0
    for rg in range(pf.num_row_groups):
        t = pf.read_row_group(rg)
        s = t.column("sample_id").to_numpy()
        try:
            gi = pa.array(np.fromiter((lut[int(x)] for x in s), np.int64, len(s)))
        except KeyError as e:
            raise SystemExit(f"[ERROR] sample_id {e} 不在 sidecar——先重跑 14。")
        b = t.select(keep_cols)
        for c in NEW_COLS:
            b = b.append_column(c, side.column(c).take(gi))
        if out_schema is None:
            out_schema = b.schema
        eidx = b.column("episode_idx").to_numpy()
        bk = (eidx.astype(np.int64) * NB) // denom        # 连续 episode 区间 → 桶
        for bv in np.unique(bk):
            sub = b.filter(pa.array(bk == bv))
            w = writers.get(int(bv))
            if w is None:
                w = pq.ParquetWriter(str(tmp / f"b{int(bv):04d}.parquet"),
                                     out_schema, compression="zstd")
                writers[int(bv)] = w
            w.write_table(sub)
        seen += t.num_rows
        if rg % 50 == 0:
            print(f"      rg {rg}/{pf.num_row_groups}  ({seen:,} 行)", flush=True)
    for w in writers.values():
        w.close()
    print(f"      Pass 1 完成：{seen:,} 行 → {len(writers)} 个桶文件")

    print(f"[3/4] Pass 2：按桶顺序排序 + 追加写 {out.name}")
    writer = None
    total = 0
    for bv in range(NB):
        bf = tmp / f"b{bv:04d}.parquet"
        if not bf.exists():
            continue
        bt = pq.read_table(str(bf)).sort_by(
            [("episode_idx", "ascending"), ("position_in_episode", "ascending")])
        if writer is None:
            writer = pq.ParquetWriter(str(out), bt.schema, compression="zstd")
        # row_group_size=5000：让 canonical 文件有 ~400 个 5000 行的 row group
        # （而非每桶一个 ~31k 行的大组）→ 流式 loader 内存有界 + worker 均衡 + 洗牌粒度好。
        writer.write_table(bt, row_group_size=5000)
        total += bt.num_rows
        bf.unlink()
    if writer is not None:
        writer.close()
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"      写出 {total:,} 行")

    print("[4/4] 验证 canonical（只读 5 个小列，内存轻）")
    chk = pq.read_table(str(out), columns=[
        "sample_id", "pass_id", "episode_idx", "position_in_episode", "is_last_in_episode"])
    assert chk.num_rows == n, f"行数变了 {chk.num_rows} != {n}"
    eidx = chk.column("episode_idx").to_numpy()
    pos = chk.column("position_in_episode").to_numpy()
    sid = chk.column("sample_id").to_numpy()
    de, dp = np.diff(eidx), np.diff(pos)
    ok = (de > 0) | ((de == 0) & (dp == 1))
    assert ok.all(), f"顺序非 (episode_idx,position) 单调：{int((~ok).sum())} 处违例"
    first = np.concatenate([[True], de > 0])
    assert (pos[first] == 0).all(), "有 episode 首行 position != 0"
    assert len(np.unique(sid)) == n, "sample_id 不唯一/缺失"
    n_ep = int((de > 0).sum()) + 1
    print(f"      ✓ {n:,} 行；(episode_idx,position) 单调；每 episode 从 0 起；"
          f"sample_id 唯一全覆盖；episodes={n_ep:,}")
    print(f"\n核对无误后：把 {out.name} 改名为 snapshots_v2.parquet（原文件留备份），再做流式 loader。")
    return 0


if __name__ == "__main__":
    main()
