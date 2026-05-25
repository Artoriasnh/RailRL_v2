"""spec 04 §4.2 / §4.5 — (s, a, r, s', done) transition datasets for offline RL.

Two datasets:
  * TransitionDataset (map-style) — random access via (pass_id, position) successor
    map. SLOW for big runs; kept as the CORRECTNESS REFERENCE (parity vs streaming).
  * StreamingTransitionDataset (IterableDataset, Stage 4.7.2d) — the FAST loader for
    real training; relies on the canonical (episode-ordered) snapshots file so a
    sequential read yields transitions directly.

Each item is a transition built from consecutive decision points within ONE episode:
    s        = snapshot i
    a, r     = i.chosen_action_idx, i.r_total  (carried inside the HeteroData)
    s'       = next position in the same episode
    done     = True iff i is the last position in its episode (then s' is a dummy
               = self, masked by (1-done))

Terminality is POSITION-based (is_last_in_episode is correct after 14_resegment:
exactly one per episode). After the 4.7.2d fix + canonical re-sort, s' is the next
file row within the same episode_idx, and every episode lies entirely within one
split (no transition crosses train/val/test).

NOTE: `import torch` at module top is required so StreamingTransitionDataset is a
*module-level* IterableDataset subclass — only module-level classes are picklable,
which Windows `spawn` DataLoader workers (num_workers>0) need. torch_geometric /
numpy / pyarrow stay lazy (imported inside methods).
"""
from __future__ import annotations
from typing import Optional

import torch

from .. import config as C
from ..encoders.input_pipeline import (
    NormStats, encode_snapshot, to_heterodata, load_pass_split,
)


class TransitionDataset:
    """Map-style reference loader (random access). SLOW (reads a whole row group per
    item) — used for smoke / correctness parity, NOT real training."""

    def __init__(self, parquet_path, stats_path, split: Optional[str] = "train",
                 pass_split_path=None):
        import pyarrow.parquet as pq
        self.pf = pq.ParquetFile(str(parquet_path))
        self.stats = NormStats.load(stats_path)
        self.split = split
        self._pass_split = load_pass_split(pass_split_path)
        if not self._pass_split:
            print("[TransitionDataset][warn] pass_split.parquet missing — run "
                  "scripts/train/00_build_time_split.py first (split would be degenerate).")

        self._locs: list[tuple[int, int]] = []          # (row_group, local_row)
        self._meta: list[tuple[str, int]] = []           # (pass_id, position)
        cols = ["pass_id", "position_in_episode"]
        for rg in range(self.pf.num_row_groups):
            tb = self.pf.read_row_group(rg, columns=cols)
            pids = tb.column("pass_id").to_pylist()
            poss = tb.column("position_in_episode").to_pylist()
            for li, (pid, pos) in enumerate(zip(pids, poss)):
                if split is None or self._split_of(pid) == split:
                    self._locs.append((rg, li))
                    self._meta.append((str(pid), int(pos)))

        key_to_idx = {(pid, pos): i for i, (pid, pos) in enumerate(self._meta)}
        n = len(self._meta)
        self._succ = list(range(n))
        self._done = [0.0] * n
        for i, (pid, pos) in enumerate(self._meta):
            j = key_to_idx.get((pid, pos + 1))
            if j is None:                       # max position in this episode → terminal
                self._succ[i] = i
                self._done[i] = 1.0
            else:
                self._succ[i] = j
                self._done[i] = 0.0
        self.n_missing_successor = 0

    def _split_of(self, pass_id) -> str:
        if self._pass_split:
            return self._pass_split.get(str(pass_id), "train")
        return "train"

    def __len__(self):
        return len(self._locs)

    def _load(self, i):
        rg, li = self._locs[i]
        row = self.pf.read_row_group(rg).slice(li, 1).to_pylist()[0]
        return to_heterodata(encode_snapshot(row, self.stats))

    def __getitem__(self, i):
        return {"s": self._load(i),
                "s_prime": self._load(self._succ[i]),
                "done": float(self._done[i])}


def transition_collate(items):
    """List[{s, s_prime, done}] → (batch_s, batch_s_prime, done) for the trainer."""
    from torch_geometric.data import Batch
    batch_s = Batch.from_data_list([it["s"] for it in items])
    batch_sp = Batch.from_data_list([it["s_prime"] for it in items])
    done = torch.tensor([it["done"] for it in items], dtype=torch.float32)
    return batch_s, batch_sp, done


# ============================================================
# Stage 4.7.2d — 流式转移数据集（canonical 文件）
# ============================================================

class StreamingTransitionDataset(torch.utils.data.IterableDataset):
    """流式 (s,a,r,s',done) —— 解决旧 loader「每取 1 行解码整组、还取两次」的性能阻塞。

    前提：snapshots 已被 15_resort_snapshots_canonical.py 重排成 **canonical 顺序**（按
    (episode_idx, position_in_episode) 全局排序、episode 列已修正、带 split 列）。于是顺序
    读即得转移：s=行 i；s'=行 i+1（若 episode_idx 相同）；done=is_last_in_episode[i]（终止 s'=自身）。

    要点（docs/4_7_2d_loader_design_DRAFT.md v2）：
    - **超块** = 连续 block_groups 个 row group；每 epoch 打乱超块顺序 + 块内打乱转移（块洗牌）。
    - **每 row group 只解码一次**（整块一次 read_row_groups）；行编码按需 + 块内缓存，内存有界。
    - **split 过滤**：episode 整段同 split → 丢非目标 split 的整 episode；配对用 episode_idx 相等
      判定（过滤后跨 episode 不误配）。
    - **worker 安全**：每 worker 取 超块[wid::nw]，各自开 pyarrow 句柄（不 pickle）；本类是
      module-level 可 pickle（Windows spawn 需要）。
    - 边界：超块末尾若是非终止行（episode 跨块）→ 丢这一条转移（每块边界 ~1 条，可忽略）。

    DataLoader 约定（dataset 直接产出已 collate 的 batch）：
        ds = StreamingTransitionDataset(path, stats_path, split="train")
        dl = DataLoader(ds, batch_size=None, num_workers=4)
        for batch_s, batch_sp, done in dl: ...
    """

    TRIVIAL_STRATUM = 6

    def __init__(self, parquet_path, stats_path, split: str = "train",
                 block_groups: int = 2, batch_size: int = 256,
                 shuffle: bool = True, seed: int = 0,
                 stratified: bool = False,
                 stratum_labels_path=None, stratum_weights_path=None):
        # block_groups=2 → 超块 ~10k 行（row_group=5000）；编码缓存 ~10k×~40KB≈0.4GB/worker。
        # 调大=洗牌更充分但更吃内存；调小=更省内存。
        # stratified=True：块级近似分层采样（spec §4.4）——块内按 1/sqrt(freq) 权重有放回抽样，
        #   稀有 stratum 过采样；需先跑 16_build_stratum_labels.py 生成 sidecar。
        super().__init__()
        from pathlib import Path as _P
        import pyarrow.parquet as pq
        self.path = str(parquet_path)
        self.stats_path = str(stats_path)
        self.split = split
        self.block_groups = max(1, int(block_groups))
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        self.stratified = bool(stratified)
        _sdir = _P(self.path).parent
        self.stratum_labels_path = str(stratum_labels_path or (_sdir / "stratum_labels.parquet"))
        self.stratum_weights_path = str(stratum_weights_path or (_sdir / "stratum_weights.json"))
        num_rg = pq.ParquetFile(self.path).num_row_groups
        self.blocks = [(b, min(b + self.block_groups, num_rg))
                       for b in range(0, num_rg, self.block_groups)]
        self._stats = None                       # lazy per-worker
        self._strata = None                       # sample_id → stratum (lazy)
        self._weights = None                      # stratum → weight (lazy)

    def set_epoch(self, e: int):
        self.epoch = int(e)

    def _get_stats(self):
        if self._stats is None:
            self._stats = NormStats.load(self.stats_path)
        return self._stats

    def _get_strata(self):
        if self._strata is None:
            import pyarrow.parquet as pq
            t = pq.read_table(self.stratum_labels_path, columns=["sample_id", "stratum"])
            self._strata = dict(zip(t.column("sample_id").to_pylist(),
                                    t.column("stratum").to_pylist()))
        return self._strata

    def _get_weights(self):
        if self._weights is None:
            import json
            w = json.loads(open(self.stratum_weights_path).read())
            self._weights = {int(k): float(v["weight"]) for k, v in w.items()}
        return self._weights

    def _worker_blocks(self):
        import random
        wi = torch.utils.data.get_worker_info()
        wid, nw = (0, 1) if wi is None else (wi.id, wi.num_workers)
        order = list(range(len(self.blocks)))
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(order)
        return [self.blocks[order[j]] for j in range(len(order)) if j % nw == wid]

    def __iter__(self):
        import random
        import numpy as np
        import pyarrow.parquet as pq
        stats = self._get_stats()
        rng = random.Random(self.seed * 1009 + self.epoch * 7 + 1)
        pf = pq.ParquetFile(self.path)           # 每 worker 自开句柄（不 pickle）
        for (rg0, rg1) in self._worker_blocks():
            tbl = pf.read_row_groups(list(range(rg0, rg1)))    # 整块一次解码
            eidx = tbl.column("episode_idx").to_numpy()
            islast = np.asarray(tbl.column("is_last_in_episode").to_pylist(), dtype=bool)
            split_b = np.asarray(tbl.column("split").to_pylist(), dtype=object)
            m = len(eidx)
            specs = []                            # (i_s, i_sp, done)
            for i in range(m):
                if split_b[i] != self.split:
                    continue
                if islast[i]:
                    specs.append((i, i, 1.0))
                elif i + 1 < m and split_b[i + 1] == self.split and eidx[i + 1] == eidx[i]:
                    specs.append((i, i + 1, 0.0))
                # else: 块边界非终止行 → 丢弃（可忽略）
            if not specs:
                continue
            cache: dict[int, object] = {}

            def enc(li, _cache=cache, _tbl=tbl, _stats=stats):
                e = _cache.get(li)
                if e is None:
                    row = _tbl.slice(li, 1).to_pylist()[0]
                    e = to_heterodata(encode_snapshot(row, _stats))
                    _cache[li] = e
                return e

            def emit(chunk):
                items = [{"s": enc(a), "s_prime": enc(b), "done": d}
                         for (a, b, d) in chunk]
                return transition_collate(items)

            if self.stratified:
                # 块级近似分层：块内按 1/sqrt(freq) 权重有放回抽样（≈ WeightedRandomSampler）
                strata = self._get_strata()
                wmap = self._get_weights()
                sid_arr = tbl.column("sample_id").to_numpy()
                spec_w = [wmap.get(int(strata.get(int(sid_arr[a]), self.TRIVIAL_STRATUM)), 1.0)
                          for (a, _b, _d) in specs]
                n_batches = max(1, len(specs) // self.batch_size)
                for _ in range(n_batches):
                    chunk = rng.choices(specs, weights=spec_w, k=self.batch_size)
                    yield emit(chunk)
            else:
                if self.shuffle:
                    rng.shuffle(specs)
                for c in range(0, len(specs), self.batch_size):
                    yield emit(specs[c:c + self.batch_size])
            cache.clear()
