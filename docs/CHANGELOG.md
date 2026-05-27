# RailRL v2 实现路径速览（CHANGELOG）

> **目的**：从头到尾一眼看清"做了什么、改了什么、为什么"，方便随时 check。
> 这是**索引/路线图**（按时间顺序、每步一两行）；**完整细节**见 `IMPLEMENTATION_LOG.md`
> （按时间追加的航海日志），**逐次代码变更**见 git history，**契约**见 `spec/01-05`。
> 维护：每完成一个子步骤，在对应阶段末尾追加一行。

---

## 阶段进度总览

| 阶段 | 内容 | 状态 |
|------|------|------|
| Stage 0 | Spec 锁定（5 份契约） | ✅ |
| Stage 1 | 数据 pipeline 验证 + 环境 | ✅ |
| Stage 2 | 决策点 + 候选动作 + 8 特殊 flag | ✅ |
| Stage 3 | snapshot builder（state + leak audit + episodes） | ✅ |
| Stage 4.1-4.6 | 模型（HGT + Transformer + Q + 2 aux heads） | ✅ |
| Stage 4.6.5 | v2 真 reward 重算 + 填回 | ✅ |
| Stage 4.7.1 / .1.5 | 损失模块 + 时间划分 | ✅ |
| Stage 4.7.2a-c | transitions + trainer（smoke 端到端过） | ✅ |
| **Stage 4.7.2d** | episode 跨月修 + canonical 重排 + 流式 loader + 块级分层 + lateness 修 + platform_dev 修 | ✅ |
| **Stage 5** | 50k sanity（全 §11 gate PASS）+ 泄露审计 06/07/21 全过 | ✅ |
| Stage 4.6.5b | 🔴 delay reward 两 bug 修复（train_id 跨月 + Movements +1h）→ 全量重算 reward | ✅ |
| **Stage 6** | 全量 3-seed CQL（修复数据上重训：seed 42 ✅ / 43 ✅ / 44 待；§11 gate 全过） | 🔨 |
| **Stage 7** | 非学习 baseline Table I ✅(B0 随机/B0' 计划站台/B0'' 首候选;BC/IQL 待) | 🔨 |
| **Stage 8** | 评估：Tier1/2 口径 ✅ · P2.6 模拟器 ✅验证 · Tier-3 安全优先 ✅ · OPE/FQE ✅ | ✅ |
| **Stage 9-11** | XAI：L3✅(反事实) · L2✅(Q-gap Shapley) · L5✅(IRL 定性);L1(显著性)/L4(规则) 进行中 | 🔨 |
| Stage 12 | 论文（不由 AI 代写,只记录结果;结果汇总 `RESULTS_SUMMARY.md`） | ⏳ |
| — | **多 seed**：seed44 → 3-seed mean±std（发表硬要求） | ⏳ |

---

## 详细路径（按时间）

### Stage 0-3（2026-05-19 ~ 05-20）数据 → snapshot
- **Stage 0**：锁定 5 份 spec（task framing、CQL α=5/γ=0.95、3 阶段 5+15+20ep、编码器架构…）。
- **Stage 1**：`config.py` 重写 v2 路径 + 锁定常量；`00_verify_pipeline.py`（49 项 sanity 过）。
- **Stage 2**：`mdp/trigger.py`(决策点) + `action.py`(候选) + `special_flags.py`(8 flag)。跑出 ~2M 决策点。
  关键修：train_id="0" 占位符过滤；X-prefix signals = signal 前置点（保持独立）。
- **Stage 3**：`mdp/state.py` + `state_history.py`（5 个 history）+ `schema.py`(45 列) + `leak_audit.py`。
  大返工：动作空间持久化（`01b_enrich_candidates.py`）、子图候选种子化、性能 12x、**us/ns 单位 bug**（TOOL_TRAPS §12）。
  最终：`snapshots_v2.parquet` 1,996,572 行，审计全过（退化子图 0%、leak 0、caps 遵守）。

### Stage 4.1-4.6（05-20）模型
- `encoders/hgt.py`(HGT) + `sequence.py`(Transformer) + `fusion.py` + `policies/q_network.py` + `heads.py`（route+time）+ `model.py`（串起来，3 个 gather）。
- 关键修：PyG ModuleDict 'train' 撞名 → 改 'trn'（TOOL_TRAPS §13）。模型 3,020,903 参数。

### Stage 4.6.5（05-22）v2 真 reward
- 发现 snapshot 的 reward 列是 NaN 占位 + 旧 v1 reward 不对应 → 从 decision_points_v2 重算。
- `reward_v2.py` + `08_label_pr_outcomes_v2.py` + `09_compute_rewards_v2.py` + `10_merge_rewards_into_snapshots.py`。
- 关键修：**sample_id 必须复刻 build_episodes 重排序**，否则 reward↔snapshot 静默错位（matched 100% 也会错指）→ 用 label 独立信号验证对齐 100%。

### Stage 4.7.1 ~ 4.7.2c（05-22）训练管线
- `algorithms/losses.py`（CQL/IQL/BC + aux + totals）；smoke 过、算术自洽。
- `4.7.1.5`：发现哈希划分有时间泄露 → 改**按时间·整 episode 划分**（`00_build_time_split.py` + `pass_split.parquet`）；重算 normalization。
- `algorithms/transitions.py`（TransitionDataset，按 position 判终止）+ `trainer.py`（3 阶段 A/B/C）+ `09_train.py`；`--smoke` 端到端过。
- 收尾留下阻塞项 **4.7.2d**：loader 性能 + 分层采样。

### Stage 4.7.2d（2026-05-22，本会话）—— episode 修复 + loader

> 路径：先查现状 → 发现 loader 性能阻塞 → 顺带验证时**挖出更严重的 episode 定义 bug** → 停下来先修 episode → 再回 loader。

| # | 做了什么 | 文件 | 关键发现/决策 |
|---|---------|------|--------------|
| 1 | 通读 log/traps/spec，核对源码与产物 | — | 确认 4.7.2c 状态；发现项目根在 RailRL_v2（非挂载的 RailRL） |
| 2 | 写 loader 设计草案 v1 | `4_7_2d_loader_design_DRAFT.md` | 推荐流式；硬件（服务器 A100 40GB/盘紧，本地 RTX5070/4TB）→ 排除 memmap |
| 3 | 验证文件顺序（Win） | `scripts/1.py` | 🔴 文件是 **6-shard strided 交错、非 sample_id 顺序**；sample_id 有 3,051 个洞 |
| 4 | 验证 pass 时间跨度（Win） | — | 🔴🔴 **pass 跨度最长 397 天**（应是分钟级）→ episode 定义有 bug |
| 5 | 定位根因 | 读 `pass_assignment.py`/`episode.py` | TRUST train_id 的 EE=当月几号 → 每月复用 → groupby min/max 跨数月 |
| 6 | 量化危害（Win 诊断脚本） | — | 🔴 85% 行在 >1天 episode；**12.9 万行 test 泄露进 train**；6.7 万假转移 |
| 7 | 确认修复范围 | 读 `01b`/`state.py`/`action.py` | pass_id 是纯 identity（不进 state/候选/reward）→ 修复 = 列 patch，不重建 |
| 8 | 数据驱动选 gap 阈值 G（Win） | `scripts/mdp/13_episode_gap_analysis.py` | gap 分布**双峰**，空谷 [30min,12h] → **锁定 G=2h** |
| 9 | 重分段 episode（Win，✅验证） | `scripts/mdp/14_resegment_episodes.py` | → `episodes_v2.parquet` sidecar + 新 `pass_split.parquet`；泄露 129,021→0；80,210 episodes；sample_id/reward 不动 |
| 10 | normalization 改读 sidecar（Win，✅） | `input_pipeline.py`(+`load_episode_split`)、`01_build_normalization_stats.py` | split 计数 == 14；**vocab 不变**（268/123/278/2184）→ 编码器不重建 |
| 11 | 选 loader 路径 | — | Hao 选**路径甲（canonical 重排）**；草案刷成 v2 |
| 12 | canonical 重排（Win，✅验证） | `scripts/mdp/15_resort_snapshots_canonical.py` | 替换 4 个 episode 列+加 split+按 (episode_idx,position) 重排 → `snapshots_v2.canonical.parquet` |
| 12b | 修 15 的 OOM | 同上 | 第一版整表 sort → 31GB RAM 爆（连 PyCharm 崩，TOOL_TRAPS §14）；改**流式 bucket 外排**，峰值几百 MB ✅ |
| 13 | 改名 canonical → snapshots_v2.parquet ✅ | （Hao 手动） | 原文件留备份 |
| 14 | 流式 loader 实现 ✅（smoke A/B/C PASS） | `transitions.py`(+`StreamingTransitionDataset`)、`scripts/train/10_smoke_streaming.py` | module-level IterableDataset（可 pickle→worker 安全）；超块顺序流+块洗牌+s'=下一行。smoke：正确性 PASS、num_workers=8 → 930/s、worker 一致。修：15 改 row_group=5000、loader block_groups=2 |
| 15 | 分层采样（spec §4.4）✅ | `scripts/mdp/16_build_stratum_labels.py`、`transitions.py`(stratified)、smoke [D] | stratum 标签 sidecar + 块级近似分层（1/√freq 有放回抽样）。smoke [D] PASS |
| 16 | 🔴→✅ lateness bug | `state_history.py`(current_lateness_s)、`special_flags.py`(f_late_train)、`scripts/mdp/17,18` | scheduled_delta_s 旧=gbtt-next（恒≥0、远 occurrence 垃圾）→ 改 realized timetable_variation×60×sign(status) ≤t（leak-safe）。late_train 0→21% |
| 17 | 🔴→✅ platform_dev 过宽 | `special_flags.py`(f_platform_dev)、`scripts/mdp/19,20`（Hao 另对话修） | 空生成器 bug：候选 end_platform 全 None→误触发 83%。修=要求≥1 已知候选平台。83%→0.7%（spec ~1.5%）。印证任务#8 |
| 18 | **✅✅ Stage 4.7.2d 完成** | — | episode+canonical+流式loader+分层+lateness+platform_dev 全部修通，数据干净，smoke A/B/C/D PASS |

### Stage 5 — 50k sanity（2026-05-22）✅ 全 §11 gate PASS
- trainer 接流式+分层+§11 gate（`09_train.py` 重写，`trainer.evaluate` 加 time_acc/q_absmax）。修 |Q| 指标 bug（排除 -1e9 掩码哨兵）+ HGT pooling batch bug（dim_size=num_graphs，末尾空类型节点截断）。
- 真 sanity（50k/epoch）：Phase A route .73/time .41/loss↓；B Q-top1 .87/|Q|有界；**C Q-top1 .946**。全损失↓、全精度↑、|Q| 有界、无 NaN → 框架健全。0.946 是模仿精度（FCFS+timetable），"是否优于信号员"留 Stage 8。

### 泄露审计（Stage 6 训练中复审）✅ 06+07+21 全过
- `06_run_leak_audit_full.py`（assert_no_leak 全 7-check，修流式 OOM + center 别名 bug）+ `07_audit_snapshots.py`（数值/结构，242k：banned=0/center track/time 干净）+ **新 `21_audit_leakage.py`**（基线判据）。**无泄露**；val route .915 高出平凡基线 ~30pp=学真实 state→决策映射。`docs/LEAK_AUDIT.md` 锁定结论。

### Stage 6 — 全量 3-seed CQL（2026-05-22 训练中）
- 全量模式（batch 256，batches/epoch≈5,750）；`--sanity`(50k) 模式保留；checkpoint 精简(phase-end+best+final)；**`--resume`** 滚动 ckpt 每 epoch（12h 窗口安全）；HPC fd-sharing fix（`set_sharing_strategy('file_system')` → num_workers≥16）。
- seed 42 全量训练中（~17h）；ep1 已 val route .915/time .65（高=步数多+任务可模仿，已审计无泄露）。43/44 next。
- **（2026-05-24）seed 42 跑完 ✅ §11 gate 全过**（A/B/C；final C20 Q-top1 .984、best .9846@C17、|Q| C 阶段 ~124 无闸但有界）。新 **`scripts/train/11_aggregate_results.py`**（= spec §12 的 05_aggregate_results，05 号被占改用 11）：跨 seed mean±std + bootstrap CI，沙盒已测 n=1/n=3。详见 IMPLEMENTATION_LOG「Stage 6 — seed 42 跑完 + 聚合脚本」。**待删**临时合成测试数据 `outputs/train/_SYNTHETIC_TEST_DELETE_ME/`。

### Stage 7/8 准备 — 共享评估口径（2026-05-24）🔨
- Hao 锁定：Stage 7 baseline = B0 随机 / B0' FCFS / B1 BC / **IQL**；**先建共享评估口径**；eval 用 best.pt；分层先做现有 7 类 + overall。
- 新 **`src/railrl/eval/metrics.py`**（纯 numpy，沙盒单测过）：Tier1 整体（action top-1 all & set-only 并列 / route·time head / wait recall·precision / Q-gap）+ Tier2 per-stratum top-1（7 类 + overall）。
- 新 **`scripts/eval/01_evaluate_model.py`**（torch，Hao 跑）：best.pt + test 集（stratified=False）→ 指标 → `outputs/eval/{tag}_{split}_metrics.json`。AST/符号核对过。
- **待 Hao**：`python scripts/eval/01_evaluate_model.py --seed 42` 出 CQL 的首个 test 数。详见 IMPLEMENTATION_LOG「Stage 7/8 准备 — 共享评估口径」。

### 🔴 delay_change (r_delay) train_id 跨月 bug — 已查实 + 已修（2026-05-24）
- Hao 质疑 r_delay 仅占奖励 2.5% + Mar-Jul delay 覆盖≈0。查实：Movements 延误数据每月密集可用，**根因是 `reward_features.compute_delay_changes` 按完整 TRUST train_id 分组（EE=当月几号→每月复用，与 episode 跨月 bug 同源）→ 81% 计时点落在跨年 train_id → 决策匹配挑错日的行程 → out_window 77% → delay 覆盖压到 6%**。
- **修**：按 gap(2h) 把 train_id 切成单次行程再匹配/夹取（`reward_features.py` + 诊断 `scripts/mdp/22_diagnose_delay_coverage.py`，后者带 `--run-gap` 可 before/after 对比）。/tmp 逻辑验证通过。
- **待 Hao**：跑 22 预览新覆盖 → 确认大升 → **重算 decision_rewards_v2(08→09→10) + 重并 snapshots + 重训 CQL 42/43/44**（当前模型在 r_delay 大面积缺失的奖励上训的）。详见 IMPLEMENTATION_LOG「delay_change bug」。

### 🔴🔴 fix #2 — Movements Apr-Jul +1h 时钟 bug（已查实 + 源头修 + 状态 patch，2026-05-24）
- 22 预览：fix#1 后 out_window 77%→0.13%、delay 覆盖 6%→24%、Aug+ 各月~35%；但 Mar-Jul 仍≈0（no_baseline 95%）。
- 定位：**Movements actual/planned/gbtt 在 2023-04-04..07-31 整体 +1h**（双重 BST 采集 bug；per-day offset +58.6m、Movements 自身节律晚 1h 佐证；delay 值不变因 actual−planned 抵消）。仅绝对时钟错位 → 与 TD 决策时间错配。**val/test(2024) 在正常期不受影响 → 之前评估有效。**
- 修：`config.MOVEMENTS_BST_FIX_*` + `data_io.correct_movements_bst()`（窗口 [3/17,8/5) −1h）；`load_movements`/`compute_delay_changes` 统一应用。blast radius = reward + 6 个 Movements 派生状态字段（41% train）；**episodes 不受影响**（来自决策-gap 重分段）。
- 新 `scripts/mdp/23_patch_movements_state.py`：仅对 Apr-Jul 行重算 5 个状态字段（planned_platform/scheduled_delta_s/schedule_outlook/f_late_train/f_platform_dev，复用真实函数）；f_trts_pressed 需 TD trts_state（未存）无法 patch、留微小残差记局限。
- **完整跑序**：23 → reward 08→09→10 → 01 norm → 16 stratum → 重训 42/43/44 → 重评估。详见 IMPLEMENTATION_LOG。

### ✅ 数据修复落地 + 重训前体检 + seed42/43 重训（2026-05-24/25）
- 两 delay bug 修复后全量重算 reward(08→09→10)+重并+重派生 Apr-Jul 状态(`23_patch_movements_state.py`)；新增**重训前 5 段体检 gate `24_pre_retrain_audit.py`**（结构 / 奖励+label 独立一致 / 状态 / 非窗口行不变 / 跨月 MAD 异常扫描）。修 fix#2 reward 路径 dtype bug（TOOL_TRAPS §18）。
- **seed42 在修正数据上重训 ✅ §11 gate 全过**（C Q-top1 .981 / best .9823@C8）；test set-only .957；r_delay 2.5%→9.6%。**seed43 ✅ 全过**（best .9832@C20，与 42 高度一致；|Q| C 末 105 有界、比 42 更受控）。seed44 待。

### ✅ Stage 8 评估 — 模拟器 + Tier-3 + OPE/FQE（2026-05-25/26）
- **P2.6 模拟器**（`xai/l3_system.py` + `simulator/01_estimate_parameters`,`02_validate_simulator`）：4 参数表 + 事件驱动 rollout（eval-only）；spec §14.6.1 PRIMARY 门 occupancy Spearman **.94** / throughput **.86**（headway 分位 p5→p1 校准后）→ PASS。
- **Tier-3 安全优先 v1.2**（`eval/03_tier3_replicate_improve.py`）：先修共享可变状态 artifact（simulate 用 `replace` 副本）+ 反事实不对称诊断（被判 unsafe 的 100% 单独跑能跑完 → fixed-others 偏置）→ 改 genuine_unsafe 用**模拟器无关信号**（候选合法 + 单独可行）。结果 **genuine_unsafe 0% · 冲突负荷 headway-wait Δ≈+0.07(≈0) · 复制 95.7%**；delay 层模型偏离 intrinsic +14s 略慢。
- **OPE/FQE**（`eval/04_ope_fqe.py` + `05_ope_fqe_decompose.py`）：真实 logged 轨迹估 V^π(FQE) vs V^β(真实 MC 折扣回报)。**total ΔV≈0（与人持平）；delay ΔV=−0.24 显著为负（模型 delay 更差）**，与模拟器 +14s 互证。根因诊断（**非架构**）：r_delay 稀疏 34% + 量级小 → 有效占比 ~10%，权重 1.0 最大却被 wait/throughput 淹没 → 模型拿 delay 换少等待/高吞吐（待 05 分解 + Σ-check 证实）。
- **框架转向（与 Hao 商定）**：headline 从"超越人类"改为 **"高保真(95.7%)+ 安全(0 unsafe)复制 · 与人持平 · 胜过 FCFS 等 baseline"** —— baselines 成为关键对照（可用 FQE 换 π 直接评估）。
- 详见 IMPLEMENTATION_LOG 对应日期条目。

**本会话新增/改动文件清单（截至 Stage 6）**
- 新脚本：`scripts/mdp/` 13(gap 分析) 14(重分段) 15(canonical 重排) 16(stratum 标签) 17(late 诊断) 18(late patch) 19(flag/platform 诊断) 20(platform_dev patch) 21(泄露基线)；`scripts/train/10_smoke_streaming.py`、`scripts/train/11_aggregate_results.py`(Stage 6 多 seed 聚合)
- 改代码：`input_pipeline.py`(+`load_episode_split`)、`01_build_normalization_stats.py`(split 改 sidecar)、`transitions.py`(+`StreamingTransitionDataset`+stratified)、`state_history.py`(`current_lateness_s`)、`special_flags.py`(`f_late_train`/`f_platform_dev`)、`hgt.py`(pooling dim_size 修)、`09_train.py`(流式+分层+§11 gate+resume+fd fix)、`trainer.py`(time_acc/q_absmax+resume)、`06_run_leak_audit_full.py`(流式+center 别名)
- 新产物：`episodes_v2.parquet`、`stratum_labels.parquet`+`stratum_weights.json`、`snapshots_v2.parquet`(canonical+lateness+platform_dev patched)、重生成 `pass_split.parquet`/`normalization_stats.json`
- 新文档：`4_7_2d_loader_design_DRAFT.md`(v2)、`CHANGELOG.md`、`LEAK_AUDIT.md`、`NEW_CONVERSATION_PROMPT.md`(刷新)；`IMPLEMENTATION_LOG.md`+`TOOL_TRAPS.md` 持续追加

---

## 关键不变量（贯穿，别破坏）
- **sample_id** 是行的物理 id（reward 按它对齐，4.6.5 label agreement 100%）——episode 重分段/重排都**不改 sample_id↔reward/state 的对应**。
- **vocab**：track_id 268 / signal 123 / route 278 / train 2184（编码器据此，勿改）。
- **时间划分**：train<2024-02-01 / val<2024-03-01 / test≥（按 episode 起始时间，episode 已时间局部，零泄露）。
- **锁定超参**：CQL α=5、γ=0.95、3 阶段 5+15+20ep、batch 256、AdamW lr3e-4。

## 2026-05-27 — L4 rule-compliance (P2.5 + spec §10) 建成
- rule_base.py（19 条 Hao-approved 规则 + 路由目录 + matcher）、l4_rules.py（l4_check/summary，扩展 §10.2 含 platform_set/policy_fact/ambiguous）、03_finalize.py（→rules.parquet）、12_l4_compliance.py（模型&信号员合规 GPU 驱动）。
- 方向锚点经 Hao 回填 + AI 核验（更正 TFPY→TFPV、弃用 0命中的 TDWV→改 TD5043）。
- 纯逻辑沙盒全过；torch/pyarrow 实跑待 Hao。

## 2026-05-27 — L4 全量结果 + focal_signal 格式修复
- 修 focal_signal 裸数字格式 bug（rule_base.decision_signal 从候选 route_id 还原 prefixed 决策信号）→ L4 首跑 0 判定 → 修后正常。
- L4 全量(seed42 test)：硬规则合规 模型 81.0% / 信号员 85.7%；rules.parquet(19 行)已生成。
- 新坑记 TOOL_TRAPS §20（磁盘满→NUL 损坏+不可删 pyc，用 PYTHONPYCACHEPREFIX）、§21（virtiofs mount 冻结在截断版本）。

## 2026-05-27 — §12 Selective Override 建成
- deploy/selective_override.py(三闸规则+L2 faithfulness+评估器) + scripts/eval/13_selective_override.py(模型+L3 reward-unit delta+l4+gated L2 驱动)。纯逻辑沙盒全过;torch/sim 待 Hao。spec 最后一个未建模块完成。

## 2026-05-27 — §12 gate_l4 改"非 non-compliant 即放行"(双口径)
- selective_override 加 l4_mode(refined 默认/literal 对照)；evaluate 并列报两口径。修因:no-rule 占 ~99%,字面 compliant 闸使 override≈0。

## 2026-05-27 — B2 BC-HG + B3 IQL 学习型 baseline 建成
- trainer.compute_loss 加 alg(cql/bc/iql)+value_head；09_train.py 加 --algo bc/iql 分支(BC 20ep监督；IQL 3-phase+expectile-V/AWR+外置value_head)。复用 losses.py 现成 bc_q_loss/iql_total。CQL 路径不变。训练待 Hao GPU×3seed。
