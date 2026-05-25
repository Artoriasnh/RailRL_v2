"""Stage 4.7.2d — smoke：流式 loader 正确性 + 吞吐 + worker 安全。

前提：snapshots_v2.parquet 已是 canonical（15 重排后改名）。验证三件事：
  (A) 正确性：流式产出的转移集合 == 从文件直接推出的 ground-truth（仅差块边界丢的极少几条）。
  (B) 吞吐：流式 transitions/s（对比旧 loader ~16/s）。
  (C) worker 安全：num_workers=2（Windows spawn）能跑、且产出集合与单进程一致。

Windows（必须有 __main__ 保护，num_workers>0 才安全）：
    python scripts/train/10_smoke_streaming.py
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import pyarrow.parquet as pq

from railrl import config as C


def ground_truth(split: str) -> set:
    """从 canonical 文件直接推 (sid_s, sid_sp, done) 集合（权威对照）。"""
    t = pq.read_table(str(C.SNAPSHOTS_V2_PARQUET),
                      columns=["sample_id", "episode_idx", "is_last_in_episode", "split"])
    sid = t.column("sample_id").to_numpy()
    eidx = t.column("episode_idx").to_numpy()
    islast = np.asarray(t.column("is_last_in_episode").to_pylist(), dtype=bool)
    sp = np.asarray(t.column("split").to_pylist(), dtype=object)
    n = len(sid)
    val = (sp == split)
    term = val & islast
    nt = np.zeros(n, dtype=bool)
    nt[:-1] = val[:-1] & (~islast[:-1]) & val[1:] & (eidx[1:] == eidx[:-1])
    gt = set()
    for i in np.where(term)[0]:
        gt.add((int(sid[i]), int(sid[i]), 1))
    for i in np.where(nt)[0]:
        gt.add((int(sid[i]), int(sid[i + 1]), 0))
    return gt


def collect(ds, num_workers: int) -> set:
    from torch.utils.data import DataLoader
    dl = DataLoader(ds, batch_size=None, num_workers=num_workers)
    got = set()
    nb = 0
    for bs, bsp, done in dl:
        s_ids = bs.sample_id.view(-1).tolist()
        sp_ids = bsp.sample_id.view(-1).tolist()
        dn = done.view(-1).tolist()
        for a, b, d in zip(s_ids, sp_ids, dn):
            got.add((int(a), int(b), int(round(d))))
        nb += 1
    return got, nb


def throughput(num_workers: int, max_batches: int, warmup: int) -> float:
    """warmup 后计时（避开 worker spawn 启动开销）→ transitions/s。"""
    from railrl.algorithms.transitions import StreamingTransitionDataset
    from torch.utils.data import DataLoader
    ds = StreamingTransitionDataset(C.SNAPSHOTS_V2_PARQUET, C.NORMALIZATION_STATS_JSON,
                                    split="train", shuffle=True, batch_size=256, seed=42)
    dl = DataLoader(ds, batch_size=None, num_workers=num_workers,
                    persistent_workers=(num_workers > 0))
    t0 = None
    ntr = nb = 0
    for bs, bsp, done in dl:
        nb += 1
        if nb == warmup:
            t0 = time.time()
        elif nb > warmup:
            ntr += int(done.numel())
        if nb >= max_batches:
            break
    dt = (time.time() - t0) if t0 else 1e-9
    return (ntr / dt) if dt > 0 else 0.0


def main() -> int:
    from railrl.algorithms.transitions import StreamingTransitionDataset

    path = C.SNAPSHOTS_V2_PARQUET
    stats = C.NORMALIZATION_STATS_JSON
    print(f"canonical snapshots: {path}")

    # ---- (A) 正确性（val split，确定性 shuffle=False，num_workers=0）----
    print("\n[A] 正确性对照（val split）...")
    gt = ground_truth("val")
    ds = StreamingTransitionDataset(path, stats, split="val", shuffle=False,
                                    batch_size=256)
    got, nb = collect(ds, num_workers=0)
    missing = gt - got
    extra = got - gt
    print(f"    ground-truth 转移: {len(gt):,}")
    print(f"    流式产出转移:     {len(got):,}  (batches={nb})")
    print(f"    extra (流式多出, 应=0):     {len(extra)}")
    print(f"    missing (块边界丢, 应很少): {len(missing)}  "
          f"(≤ 超块数 {len(ds.blocks)})")
    okA = (len(extra) == 0) and (len(missing) <= len(ds.blocks))
    # 抽查 missing 都是非终止（done=0）的边界条目
    if missing:
        miss_done = {d for (_, _, d) in missing}
        print(f"    missing 的 done 取值: {miss_done}（应都是 0=非终止边界）")
        okA = okA and (miss_done <= {0})
    print(f"    [A] {'PASS' if okA else 'FAIL'}")

    # ---- (B) 吞吐：单进程 vs 多 worker（真训练用多 worker）----
    print("\n[B] 吞吐（train；编码是 CPU-bound，靠 num_workers 并行）...")
    r0 = throughput(num_workers=0, max_batches=120, warmup=20)
    print(f"    num_workers=0 : {r0:,.0f} transitions/s（单核编码上限）")
    r8 = throughput(num_workers=8, max_batches=400, warmup=40)
    print(f"    num_workers=8 : {r8:,.0f} transitions/s（真训练用这个；旧 loader ~16/s）")
    rate = r8
    # 判据=多 worker 确实并行提速（绝对吞吐随机器空闲度波动，仅供参考，不作硬阈值）。
    okB = (r8 > r0 * 1.3)
    print(f"    （绝对值随机器负载波动 —— 判据只看并行是否生效：{r8:,.0f} vs 单核 {r0:,.0f}）")

    # ---- (C) worker 安全（num_workers=2，val，shuffle=False）----
    print("\n[C] worker 安全（num_workers=2, val, shuffle=False）...")
    okC = True
    try:
        ds3 = StreamingTransitionDataset(path, stats, split="val", shuffle=False,
                                         batch_size=256)
        got3, nb3 = collect(ds3, num_workers=2)
        same = (got3 == got)
        print(f"    num_workers=2 产出 {len(got3):,} 转移, 与单进程集合一致: {same}")
        okC = same
    except Exception as e:
        print(f"    [C] 异常: {e}")
        okC = False

    # ---- (D) 分层采样（块级近似；需先跑 16 生成 sidecar）----
    print("\n[D] 分层采样（块级近似, train）...")
    import json as _json
    lab = C.SNAPSHOTS_V2_PARQUET.parent / "stratum_labels.parquet"
    wj = C.SNAPSHOTS_V2_PARQUET.parent / "stratum_weights.json"
    okD = True
    if not (lab.exists() and wj.exists()):
        print("    跳过：未找到 stratum_labels.parquet / stratum_weights.json（先跑 scripts/mdp/16）")
    else:
        from torch.utils.data import DataLoader
        lt = pq.read_table(str(lab))
        strata = dict(zip(lt.column("sample_id").to_pylist(), lt.column("stratum").to_pylist()))
        wmeta = _json.loads(wj.read_text())
        names = {int(k): v["name"] for k, v in wmeta.items()}
        natural = np.array([float(wmeta[str(k)]["train_freq"]) for k in range(7)])
        natural = natural / max(natural.sum(), 1)
        ds4 = StreamingTransitionDataset(path, stats, split="train", shuffle=True,
                                         stratified=True, batch_size=256, seed=1)
        hist = np.zeros(7)
        nb = 0
        for bs, _bsp, _done in DataLoader(ds4, batch_size=None, num_workers=0):
            for sd in bs.sample_id.view(-1).tolist():
                hist[int(strata.get(int(sd), 6))] += 1
            nb += 1
            if nb >= 200:
                break
        frac = hist / max(hist.sum(), 1)
        print("    stratum            自然占比   分层后占比")
        for k in range(7):
            print(f"      {k} {names.get(k, ''):<16s} {100*natural[k]:6.2f}%   {100*frac[k]:6.2f}%")
        # 分层是否生效：主导 stratum 占比下降 + 最稀有(非空)stratum 占比上升（趋于均衡）。
        # （不假设 trivial 是多数——本数据 trivial 反而很少，platform_dev 才是主导。）
        dom = int(np.argmax(natural))
        nz = [k for k in range(7) if natural[k] > 0]
        rare = min(nz, key=lambda k: natural[k])
        okD = (frac[dom] < natural[dom] - 0.03) and (frac[rare] > natural[rare])
        print(f"    主导 {dom}={names[dom]}: {100*natural[dom]:.1f}%→{100*frac[dom]:.1f}%（应降）；"
              f"最稀有 {rare}={names[rare]}: {100*natural[rare]:.2f}%→{100*frac[rare]:.2f}%（应升）；"
              f"[D] {'PASS' if okD else 'FAIL'}")

    print("\n" + "=" * 60)
    print(f"  [A] 正确性 : {'PASS' if okA else 'FAIL'}")
    print(f"  [B] 吞吐   : {'PASS' if okB else 'FAIL'}  ({rate:,.0f}/s)")
    print(f"  [C] worker : {'PASS' if okC else 'FAIL'}")
    print(f"  [D] 分层   : {'PASS' if okD else 'FAIL'}")
    print("=" * 60)
    return 0 if (okA and okB and okC and okD) else 1


if __name__ == "__main__":
    sys.exit(main())
