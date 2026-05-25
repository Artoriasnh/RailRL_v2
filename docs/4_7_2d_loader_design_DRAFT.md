# 4.7.2d Loader 设计草案 v2（待 Hao 审核）

> v2（2026-05-22）。v1 是 episode 跨月 bug 暴露前写的，已作废。本版反映：
> episode 已修（14 重分段 → sidecar；15 重排 → canonical 文件）+ Hao 选定**路径甲（canonical 重排）**。
> 尽量大白话。定稿后把最终方案摘要追加进 IMPLEMENTATION_LOG。

---

## 0. 现在到哪了

- **episode 跨月 bug 已修**：`14_resegment_episodes.py` 按 (focal_train, gap>2h, split边界) 重分段 →
  `episodes_v2.parquet` sidecar；normalization 已在修正 train 上重算（vocab 不变）。
- **canonical 重排**：`15_resort_snapshots_canonical.py` 把 sidecar 的正确 episode 列烤回 snapshots、
  并按 (episode_idx, position) 重排 → `snapshots_v2.canonical.parquet`（验证后改名为 snapshots_v2.parquet）。
- 所以进 loader 时，文件已是**干净 + 按 episode 顺序**的，下面的设计因此极简。

## 1. Loader 要干的事

把 `(s, a, r, s', done)` 一条条喂给 CQL/IQL trainer，要满足：
1. **快**（解决阻塞）——每个 row group 每个 epoch 只解码一次（现在是每取 1 行解码整组、还取两次）。
2. **转移完整**——s' = 同 episode 的下一个决策；episode 末尾 done=True。
3. **块洗牌**——去相关相邻 batch（SGD 稳定）+ 给分层采样腾空间；不跨 train/val/test。
4. **分层采样**（spec §4.4）——稀有 stratum 过采样，否则 85% trivial 淹没梯度。
5. **worker 安全**——多进程喂数据不 pickle pyarrow 句柄（解决现有 num_workers=0）。
6. **省盘省内存**——服务器盘紧（空 10-12GB），只流式读 ~573MB 文件，不产生大中间物。

## 2. canonical 文件让流式变简单（核心）

重排后，文件物理顺序 = (episode_idx, position)：**每个 episode 的行连续、按时间升序**。于是：

- **顺序读 = episode 顺序**。流里相邻两行，只要 pass_id 相同，就是一对 (s=前, s'=后)——
  **转移白送，不用为 s' 再随机翻文件**。
- 一个 episode 的最后一行（下一行换了 pass_id）= 终点（done=True，s' 用 dummy，被 (1-done) 掩掉）。
- s' 必在 s 的下一行 → 同/邻 row group → 一个很小的缓存就够。

> 对比 v1 的纠结（文件 6-shard 交错、s' 散在别处）——重排后这些全没了。这就是选路径甲的回报。

## 3. 流式 loader 设计（StreamingTransitionDataset）

`IterableDataset`，按 row-group 顺序流，配 shuffle buffer：

1. **超块（super-block）**：把连续的 ~B 个 row group（B≈8-16，约 4-8 万行）当一个超块。
2. 每个 epoch **打乱超块顺序**（块级洗牌）。
3. 读进一个超块 → 解码这些 row group（每组只解码一次）→ 在块内**按 (pass_id, position) 把相邻行配成
   转移** `(s, s', done)`（用 position+1 判定，不靠"流里下一条"——防 sample_id 有洞误配；判定逻辑
   与现有 `TransitionDataset` 一致）。
4. 跨超块边界的 episode（很少，episode ≤ ~80 行、多数 <30）用一个**小 carryover** 接上。
5. 把本超块的转移列表**打乱** → 切成 batch 吐出。
6. **split 过滤**：canonical 文件已带 `split` 列，只流式 train（或按需 val/test）。

**worker 安全**：每个 DataLoader worker 在自己进程里开文件、各分一段超块区间
（`超块[worker_id :: num_workers]`），不共享/不 pickle 句柄。A100 上 `num_workers≈4`。

**内存**：一个超块的解码缓冲（~8-16 组 × ~5000 行 ≈ 4-8 万行）+ carryover，几百 MB，可调。

## 4. 分层采样（spec §4.4，块级近似）

- **一次性预扫** `state_special_flags` → 每行 stratum 标签（优先级 late>advance>call_on>
  platform_dev>priority_compete>unusual_id>trivial），存小 sidecar `stratum_labels.parquet`
  （sample_id→stratum，几 MB）。
- 算各层频率 → 权重 `1/√freq`。
- 在 shuffle buffer 里**按权重过采样**（稀有 stratum 的转移多放几份/优先抽），让每个 batch 大致
  满足 spec：≥50 trivial + 每个非 trivial 层 ≥20。
- 每 batch 打印 stratum 直方图自检（spec §13.1）。
- "块级近似" 对 Stage 5（50k sanity，目标只是 loss 降 + 阶段判据过）够用；**精确分层**（全局加权 +
  严格配额）留到 Stage 6 全量前再加固。

## 5. 模块落点 + smoke

- `src/railrl/algorithms/transitions.py`：新增 `StreamingTransitionDataset(IterableDataset)`；
  **保留**现有 `TransitionDataset` 作正确性对照（小数据上两者转移集合应一致）。
- `src/railrl/algorithms/strata.py`（新，小）：special_flags → stratum + 权重。
- `scripts/mdp/16_build_stratum_labels.py`：预扫 → `stratum_labels.parquet`。
- 改 `scripts/train/09_train.py` 用流式 loader（`--smoke` 仍可跑）。
- `scripts/train/10_smoke_streaming.py`（新）：对比流式 vs 旧 loader 的转移集合一致、吞吐（行/秒）、
  stratum 直方图、worker 安全。
- 范围外（本阶段不做）：IQL value head、BC baseline（Stage 7）。

## 6. 待你拍 / 待确认

1. 先跑 `15_resort_snapshots_canonical.py` + 验证 → 改名为 snapshots_v2.parquet（原文件留备份）。
2. 超块大小 B（默认 ~12 组）、shuffle buffer 行数——这些是性能旋钮，smoke 后按吞吐调，先给默认值。
3. 分层先做块级近似（推荐，先解 Stage 5 阻塞）——已默认，若你要这轮就上精确分层告诉我。
