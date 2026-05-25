"""Leak-context audit — "is the high val accuracy a LEAK or just an EASY task?".

Stage 6 全量 ep1 出现 val route=0.915 / time=0.653。这脚本不重复 06(assert_no_leak)/
07(数值分布)，而是补上**可解释性基线**：如果"傻基线"已经能拿到接近的精度，则高精度是
任务可模仿性（FCFS + planned_platform 强预测 + 小动作集），而非泄露；若傻基线很低却模型很
高，才该深查泄露。

只读 snapshots_v2.parquet（本地，无 torch）。配合 06+07 构成完整泄露审计。Windows/本地：
    python scripts/mdp/21_audit_leakage.py                 # 默认 val, 抽 50k set 行
    python scripts/mdp/21_audit_leakage.py --split test
"""
from __future__ import annotations
import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import pyarrow.parquet as pq

from railrl import config as C


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--max-rows", type=int, default=50000, help="抽多少目标-split 行")
    args = ap.parse_args()

    src = C.SNAPSHOTS_V2_PARQUET
    pf = pq.ParquetFile(str(src))
    cols = ["label", "chosen_action_idx", "n_candidates", "candidate_route_ids",
            "split", "state_nodes_train", "state_nodes_route"]
    print(f"读 {src.name}，抽 {args.split} 行（上限 {args.max_rows:,}）...")
    rows = []
    for rg in range(pf.num_row_groups):
        if len(rows) >= args.max_rows:
            break
        for r in pf.read_row_group(rg, columns=cols).to_pylist():
            if r["split"] == args.split:
                rows.append(r)
                if len(rows) >= args.max_rows:
                    break
    n = len(rows)
    print(f"  抽到 {n:,} 行")

    # ---- (1) 动作空间 / 类别分布 ----
    labels = Counter(r["label"] for r in rows)
    ncand = np.array([int(r["n_candidates"] or 0) for r in rows])
    wait_rate = labels.get("wait", 0) / max(n, 1)
    print("\n[1] 动作空间 / 类别")
    print(f"    label: {dict(labels)}   wait 占比={wait_rate:.3f}  "
          f"→ '永远 wait' 的 action_acc 基线 ≈ {wait_rate:.3f}")
    print(f"    n_candidates: mean={ncand.mean():.2f} median={np.median(ncand):.0f} "
          f"max={ncand.max()}  → 候选越少，route 越易猜")

    # ---- (2) set 行的 route 基线 ----
    set_rows = [r for r in rows if r["label"] == "set" and (r["chosen_action_idx"] or 0) > 0]
    ns = len(set_rows)
    print(f"\n[2] route 基线（{ns:,} 个 set 行）")
    if ns == 0:
        print("    无 set 行，跳过。")
        return 0
    chosen_idx = []           # 0-based route index within candidates
    first_correct = 0         # 总选第一个候选 的命中
    follow_planned = 0        # 选中的路线 end_platform == focal planned_platform（已知）
    planned_known = 0
    plannedpred_correct = 0   # "选 end_platform 匹配 planned 的候选" 这个预测器命中
    plannedpred_applicable = 0
    for r in set_rows:
        cands = r["candidate_route_ids"] or []
        ci = int(r["chosen_action_idx"]) - 1
        if ci < 0 or ci >= len(cands):
            continue
        chosen_idx.append(ci)
        if ci == 0:
            first_correct += 1
        # focal planned platform
        planned = None
        for nd in (r["state_nodes_train"] or []):
            if nd.get("is_focal"):
                planned = nd.get("planned_platform"); break
        # route_id -> end_platform_id
        rmap = {nd.get("route_id"): nd.get("end_platform_id")
                for nd in (r["state_nodes_route"] or [])}
        chosen_ep = rmap.get(cands[ci])
        if planned is not None:
            planned_known += 1
            if chosen_ep is not None and chosen_ep == planned:
                follow_planned += 1
            # planned-predictor：候选里 end_platform==planned 的 → 预测它；命中=chosen 是它
            match = [i for i, c in enumerate(cands) if rmap.get(c) == planned]
            if match:
                plannedpred_applicable += 1
                if ci in match:
                    plannedpred_correct += 1
    cnt = Counter(chosen_idx)
    maj_idx, maj_n = cnt.most_common(1)[0]
    print(f"    chosen route-index 分布(top5): {cnt.most_common(5)}")
    print(f"    基线A『总选第一个候选』 acc = {first_correct/ns:.3f}")
    print(f"    基线B『总选最常见 index={maj_idx}』 acc = {maj_n/ns:.3f}")
    print(f"    基线C『选中路线 end_platform == planned_platform』占比(planned 已知 {planned_known:,}) "
          f"= {follow_planned/max(planned_known,1):.3f}")
    print(f"    基线D『预测 end_platform 匹配 planned 的候选』acc(可用 {plannedpred_applicable:,}) "
          f"= {plannedpred_correct/max(plannedpred_applicable,1):.3f}")

    print("\n[判读]")
    print("  · 若基线 A/B/C/D 已接近模型的 val route_acc(0.915) → 高精度=任务可模仿(planned_platform")
    print("    +FCFS+小候选集)，**不是泄露**。")
    print("  · 若傻基线都很低(~0.3)但模型 0.9+ → 怀疑泄露，深查 state 是否含答案相关字段。")
    print("  · 泄露的最终保证仍靠：06(assert_no_leak 无 banned 字段) + 时间划分干净(episode 不跨 split)")
    print("    + 实现态特征只用 ≤t（lateness/event token/occupancy 已修）。见 docs/LEAK_AUDIT.md 清单。")
    return 0


if __name__ == "__main__":
    main()
