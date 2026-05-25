"""Stage 4.7.2d 修复前置 — 用数据选 episode 重分段的 gap 阈值 G。

背景：当前 snapshots_v2.parquet 的 pass_id（完整 TRUST train_id，EE=当月几号 →
每月复用）把同 headcode、跨数月的决策塌缩成一个 episode（max 跨度 397 天，p99
345 天，84.7% 的行落在 >1 天的 episode 里，12.9 万行 test 期决策泄露进 train）。
修复 = 按 (focal_train, gap>G) 重新分段。本脚本**用数据决定 G**，不靠拍脑袋：

  (1) inter-decision gap 分布（按 focal_train 分组、按 t 排序后相邻决策的时间差）——
      预期是**双峰**：行程内步长（秒~分钟）+ 复现间隔（小时~天），中间有"空谷"。
      G 取在空谷里。
  (2) G 敏感性扫描：把候选 G 各跑一遍，看 episode 数 / 跨度分位 / >1天 episode 行占比 /
      被切断的转移数 / split 泄露行数 如何随 G 变。**好的 G 落在指标稳定的平台上。**

只读 [focal_train, t] 两列，不改任何文件。Windows 跑：
    python scripts/mdp/13_episode_gap_analysis.py
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import pandas as pd

from railrl import config as C


def _to_ns(series: pd.Series) -> np.ndarray:
    """强制转纳秒 int64（防 TOOL_TRAPS §12 的 us/ns 陷阱）。"""
    return pd.to_datetime(series).values.astype("datetime64[ns]").astype("int64")


def main() -> int:
    src = C.SNAPSHOTS_V2_PARQUET
    print(f"[1/4] 读 (focal_train, t) ← {src.name}")
    df = pd.read_parquet(str(src), columns=["focal_train", "t"])
    df["t_ns"] = _to_ns(df["t"])
    df["focal_train"] = df["focal_train"].astype(str)
    df = df.sort_values(["focal_train", "t_ns"]).reset_index(drop=True)
    n = len(df)
    print(f"      {n:,} 行, {df['focal_train'].nunique():,} 个 headcode(focal_train)")

    ft = df["focal_train"].to_numpy()
    tns = df["t_ns"].to_numpy()
    same_train = np.empty(n, dtype=bool)
    same_train[0] = False
    same_train[1:] = ft[1:] == ft[:-1]
    dt_s = np.full(n, np.inf)                       # 距同 train 上一条决策的秒数
    dt_s[1:] = (tns[1:] - tns[:-1]) / 1e9
    dt_s[~same_train] = np.inf                      # 换 train 处不算 gap

    # ---- (1) inter-decision gap 分布（只看同 train 的相邻间隔） ----
    gap = dt_s[np.isfinite(dt_s)]
    edges = [0, 1, 2, 5, 10, 20, 30, 60, 120, 300, 600, 1800, 3600, 7200,
             21600, 43200, 86400, 259200, 604800, 2592000, 34560000]
    labels = ["0-1s", "1-2s", "2-5s", "5-10s", "10-20s", "20-30s", "30-60s",
              "1-2m", "2-5m", "5-10m", "10-30m", "30-60m", "1-2h", "2-6h",
              "6-12h", "12-24h", "1-3d", "3-7d", "7-30d", "30d+"]
    cnt, _ = np.histogram(gap, bins=edges)
    mx = max(cnt.max(), 1)
    print(f"\n[2/4] inter-decision gap 分布（同 focal_train 相邻决策时间差, N={len(gap):,}）")
    print("      预期双峰：行程内(秒~分) ←空谷→ 复现间隔(时~天)。G 取空谷。")
    for lab, c in zip(labels, cnt):
        bar = "#" * int(60 * c / mx)
        print(f"      {lab:>7s}: {c:>10,} ({100*c/len(gap):5.2f}%) {bar}")

    # 自动找"空谷"：在 [60s, 86400s] 区间里密度最低的桶边界作为候选 G
    valley_lo, valley_hi = 60.0, 86400.0
    band = [(labels[i], edges[i + 1], cnt[i]) for i in range(len(cnt))
            if edges[i] >= valley_lo and edges[i + 1] <= valley_hi]
    if band:
        vmin = min(band, key=lambda x: x[2])
        print(f"\n      [auto] 1min~24h 间密度最低的桶: {vmin[0]} (count={vmin[2]:,}) "
              f"→ 候选 G ≈ 该桶上界 {int(vmin[1])}s")

    # ---- (2) G 敏感性扫描 ----
    VAL_ns = pd.Timestamp(C.VAL_START).value
    TEST_ns = pd.Timestamp(C.TEST_START).value

    def split_of(ns_arr: np.ndarray) -> np.ndarray:
        return np.where(ns_arr < VAL_ns, 0, np.where(ns_arr < TEST_ns, 1, 2))

    row_split = split_of(tns)                       # 每行按自身时间的"应属"split

    grid = [300, 900, 1800, 3600, 7200, 21600, 43200, 86400]
    gnames = {300: "5m", 900: "15m", 1800: "30m", 3600: "1h", 7200: "2h",
              21600: "6h", 43200: "12h", 86400: "1d"}
    print(f"\n[3/4] G 敏感性扫描（按 (focal_train, gap>G) 重分段）")
    print(f"      {'G':>5} {'#episodes':>10} {'span_p50_s':>11} {'span_p99_s':>11} "
          f"{'span_max_d':>11} {'%rows_>1d_ep':>12} {'cut_trans':>10} {'leak_rows':>10}")
    results = []
    for G in grid:
        new_ep = (~same_train) | (dt_s > G)         # 该行起一个新 episode
        epid = np.cumsum(new_ep)                     # 排序后的 episode id（连续）
        s = pd.Series(tns, copy=False).groupby(epid)
        ep_min = s.transform("min").to_numpy()
        ep_max = s.transform("max").to_numpy()
        span_s = (ep_max - ep_min) / 1e9             # 每行所属 episode 的跨度(秒)
        # 每个 episode 取一次：用 first-occurrence 掩码
        first = new_ep
        ep_span = span_s[first]
        n_ep = int(first.sum())
        # >1天 episode 的行占比
        pct_big = 100.0 * (span_s > 86400).mean()
        # 会被切断的转移：同 train、上一条 gap>G（即新 episode 边界中"非换 train"的）
        cut_trans = int((same_train & (dt_s > G)).sum())
        # split 泄露：本行所属 episode 的 split（按 episode 起始时间）≠ 本行自身 split
        ep_split = split_of(ep_min)
        leak_rows = int((ep_split != row_split).sum())
        results.append((G, n_ep, leak_rows, pct_big))
        print(f"      {gnames[G]:>5} {n_ep:>10,} "
              f"{np.median(ep_span):>11.0f} {np.quantile(ep_span,0.99):>11.0f} "
              f"{ep_span.max()/86400:>11.2f} {pct_big:>11.2f}% "
              f"{cut_trans:>10,} {leak_rows:>10,}")

    # ---- (4) 推荐 ----
    print(f"\n[4/4] 解读")
    print("      • gap 分布若双峰且有空谷 → G 取空谷（上面 [auto] 的候选）。")
    print("      • 扫描表里 leak_rows 应在某个 G 后≈0、#episodes 趋稳 → 取那个"
          "稳定平台的起点。")
    zero_leak = [G for (G, _, lk, _) in results if lk == 0]
    if zero_leak:
        print(f"      • leak_rows==0 的最大 G = {gnames[max(zero_leak)]}"
              f"（更大 G 也零泄露说明很稳）。")
    else:
        lo = min(results, key=lambda r: r[2])
        print(f"      • 没有 G 完全零泄露；最低泄露 G={gnames[lo[0]]} "
              f"({lo[2]:,} 行)。需进一步看（可能跨午夜的合法 episode）。")
    print("\n把本输出贴回，我据此锁定 G 再写 patch（sidecar，sample_id/reward 不动）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
