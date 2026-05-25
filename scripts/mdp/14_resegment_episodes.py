"""Stage 4.7.2d 修复 — 重分段 episode（修跨月 pass）→ sidecar episodes_v2.parquet。

问题（见 IMPLEMENTATION_LOG 4.7.2d）：当前 pass_id=完整 TRUST train_id，其 EE 段是
"当月几号"（day-of-month）→ 同 headcode 每月复用 → groupby min/max 把跨数月的同名车
塌缩成一个 episode。后果：max 跨度 397 天、84.7% 行在 >1 天 episode、12.9 万行 test 期
决策泄露进 train、6.7 万跨 gap 假转移。

修复：按 (focal_train, gap>G, split-date 边界) 重新切分 episode。
  - G 默认 7200s (2h)，数据驱动（13_episode_gap_analysis：gap 分布双峰，空谷 [30min,12h]，
    密度最低 1-2h；对 G∈[30min,6h] 不敏感）。
  - 额外在 split 日期边界 (VAL_START / TEST_START) 切 → 无 episode/转移跨 split → 泄露归零。

**只读 [sample_id, focal_train, t]；不动 sample_id、不动 reward、不重建 state。**

输出：
  - episodes_v2.parquet : [sample_id, pass_id, episode_idx, position_in_episode,
                           is_last_in_episode, split]   ← loader/normalization 按 sample_id 用
  - pass_split.parquet  : [pass_id, split, t_first]      ← 新 pass_id（覆盖旧）

自验证（独立信号，全部 assert）：
  - 每个 episode 内的步长 ≤ G（零跨-G 转移）
  - 无 episode 跨 split（零跨-split 转移）→ 行 split == episode split（泄露 0）
  - sample_id 唯一且覆盖全部 snapshot 行
  - is_last 每 episode 恰一个（按 position，非按 t）
打印：episode 数、各 split 行/episode 数、跨度分位、对比修复前后泄露。

Windows:
    python scripts/mdp/14_resegment_episodes.py
    python scripts/mdp/14_resegment_episodes.py --gap-hours 1   # 改 G
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import pandas as pd

from railrl import config as C

SPLIT_NAMES = ["train", "val", "test"]


def _to_ns(series: pd.Series) -> np.ndarray:
    """强制纳秒 int64（防 TOOL_TRAPS §12 的 us/ns 陷阱）。"""
    return pd.to_datetime(series).values.astype("datetime64[ns]").astype("int64")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gap-hours", type=float, default=2.0, help="episode 切分 gap 阈值 G（小时）")
    args = ap.parse_args()
    G_ns = int(args.gap_hours * 3600 * 1e9)

    src = C.SNAPSHOTS_V2_PARQUET
    episodes_out = src.parent / "episodes_v2.parquet"
    pass_split_out = C.PASS_SPLIT_PARQUET

    VAL_ns = pd.Timestamp(C.VAL_START).value
    TEST_ns = pd.Timestamp(C.TEST_START).value

    def split_codes(ns_arr: np.ndarray) -> np.ndarray:
        return np.where(ns_arr < VAL_ns, 0, np.where(ns_arr < TEST_ns, 1, 2)).astype(np.int8)

    print(f"[1/5] 读 [sample_id, focal_train, t] ← {src.name}")
    df = pd.read_parquet(str(src), columns=["sample_id", "focal_train", "t"])
    n = len(df)
    df["focal_train"] = df["focal_train"].astype(str)
    df["t_ns"] = _to_ns(df["t"])
    assert df["sample_id"].is_unique, "sample_id 不唯一——snapshot 本身有问题"
    print(f"      {n:,} 行, {df['focal_train'].nunique():,} 个 focal_train")

    # 排序：episode 在 (focal_train, t) 内连续
    df = df.sort_values(["focal_train", "t_ns"], kind="mergesort").reset_index(drop=True)
    sid = df["sample_id"].to_numpy()
    ft = df["focal_train"].to_numpy()
    tns = df["t_ns"].to_numpy()
    rsplit = split_codes(tns)                      # 每行按自身时间的"应属"split

    # ---- episode 边界：换 train | gap>G | split 变化（同 train 内）----
    same = np.empty(n, bool); same[0] = False; same[1:] = ft[1:] == ft[:-1]
    gap = np.empty(n, np.int64); gap[0] = 1 << 62; gap[1:] = tns[1:] - tns[:-1]
    split_chg = np.empty(n, bool); split_chg[0] = True; split_chg[1:] = rsplit[1:] != rsplit[:-1]
    new_ep = (~same) | (gap > G_ns) | (same & split_chg)

    print(f"[2/5] 重分段：G={args.gap_hours}h, 额外切 split 边界 "
          f"({C.VAL_START} / {C.TEST_START})")
    epid = (np.cumsum(new_ep) - 1).astype(np.int64)   # 0-indexed 全局 episode id

    out = pd.DataFrame({
        "sample_id": sid,
        "focal_train": ft,
        "t_ns": tns,
        "episode_idx": epid,
        "_new": new_ep,
    })
    # 每 train 内的 episode 序号（让 pass_id 既唯一又可读）
    seg = out.groupby("focal_train")["_new"].cumsum().astype(np.int64) - 1
    out["pass_id"] = out["focal_train"].values + ":" + seg.astype(str).values
    # episode 内位置 + 终止（按 position，每 episode 恰一个 is_last）
    out["position_in_episode"] = out.groupby("episode_idx").cumcount().astype(np.int32)
    ep_maxpos = out.groupby("episode_idx")["position_in_episode"].transform("max")
    out["is_last_in_episode"] = out["position_in_episode"].to_numpy() == ep_maxpos.to_numpy()
    # episode 的 split = 起始时间的 split（切了 split 边界 → 全 episode 同 split）
    ep_start = out.groupby("episode_idx")["t_ns"].transform("min").to_numpy()
    esplit = split_codes(ep_start)
    out["split"] = pd.Categorical.from_codes(esplit, categories=SPLIT_NAMES)

    # ---- (3) 自验证 ----
    print("[3/5] 自验证（独立信号）...")
    n_ep = int(new_ep.sum())
    # A) episode 内步长 ≤ G
    within_step = gap[~new_ep]
    assert within_step.size == 0 or within_step.max() <= G_ns, \
        f"有 episode 内步长 > G：max={within_step.max()/1e9:.0f}s"
    # B) 无 episode 跨 split：行 split == episode split → 泄露 0
    leak = int((rsplit != esplit).sum())
    assert leak == 0, f"仍有 {leak} 行 split 泄露（不应发生，已切 split 边界）"
    # C) is_last 每 episode 恰一个
    n_last = int(out["is_last_in_episode"].sum())
    assert n_last == n_ep, f"is_last 数 {n_last} != episode 数 {n_ep}"
    # D) sample_id 覆盖且唯一
    assert out["sample_id"].is_unique and len(out) == n, "sample_id 覆盖/唯一性破坏"
    # E) pass_id 唯一性（每个 episode 一个 pass_id）
    assert out["pass_id"].nunique() == n_ep, "pass_id 与 episode 不是一一对应"
    print(f"      ✓ 零跨-G 转移  ✓ 零跨-split 泄露  ✓ is_last 唯一  "
          f"✓ sample_id 全覆盖  ✓ pass_id↔episode 一一对应")

    # ---- (4) 写 sidecar + pass_split ----
    print("[4/5] 写 episodes_v2.parquet + pass_split.parquet ...")
    side = out[["sample_id", "pass_id", "episode_idx", "position_in_episode",
                "is_last_in_episode", "split"]].sort_values("sample_id").reset_index(drop=True)
    side.to_parquet(episodes_out, index=False, compression="zstd")
    ps = (out.groupby("pass_id")
            .agg(split=("split", "first"), t_first=("t_ns", "min"))
            .reset_index())
    ps["t_first"] = pd.to_datetime(ps["t_first"])
    ps.to_parquet(pass_split_out, index=False, compression="zstd")
    print(f"      -> {episodes_out}")
    print(f"      -> {pass_split_out}")

    # ---- (5) 摘要 ----
    span_s = (out.groupby("episode_idx")["t_ns"].agg(lambda s: (s.max() - s.min()) / 1e9))
    row_split_counts = out["split"].value_counts()
    ep_split_counts = (out.drop_duplicates("episode_idx")["split"].value_counts())
    print("[5/5] 摘要")
    print(f"      episodes: {n_ep:,}（修复前 14,494）")
    print(f"      跨度(秒) p50/p90/p99/max = "
          f"{span_s.quantile(.5):.0f} / {span_s.quantile(.9):.0f} / "
          f"{span_s.quantile(.99):.0f} / {span_s.max():.0f}  "
          f"(max={span_s.max()/3600:.1f}h)")
    for k in SPLIT_NAMES:
        r = int(row_split_counts.get(k, 0)); e = int(ep_split_counts.get(k, 0))
        print(f"      {k:<5s}: rows {r:>10,} ({100*r/n:5.1f}%)  episodes {e:>8,}")
    print(f"      test 期决策泄露进 TRAIN: 0（修复前 129,021）")
    print("\n下一步：改 01_build_normalization_stats.py 读 sidecar split 并重跑 → 新 normalization_stats.json。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
