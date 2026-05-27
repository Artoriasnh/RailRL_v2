# 新对话开场 Prompt（RailRL v2）

> 把这份贴给新对话的 AI 助手。它会让助手了解项目在做什么、有哪些材料、进度到哪、接下来做什么。
> **最后更新：2026-05-27（训练 3-seed 完成；Stage 7/8 评估 + XAI L1–L5 + L4 规则合规 全部完成。剩：§12 Selective Override、L1 attention-rollout+面板热图、3-seed eval 聚合、BC/IQL baseline、L5 奖励恢复改进）。结果汇总见 `docs/RESULTS_SUMMARY.md`。**

---

## 一、项目是什么

我在做 **RailRL v2**（项目根目录 `E:\Claude\RailRL_v2`）——一个用于英国铁路信号员调度决策的
**离线强化学习**系统（Network Rail Derby 工作站，14 个月数据，约 200 万决策点），目标投 **ESWA 期刊**。三大贡献：
1. **数据采集管线**（TD + Movements + 静态路网 → 状态/动作/奖励）；
2. **端到端 RL 框架**（HGT 图编码 + Transformer 事件序列 + per-action Q 网络，结构化动作
   `{wait} ∪ {(focal_train, route)}`，**CQL** 主算法 + 3 阶段训练）；
3. **五级可解释决策**（XAI，spec 05；尚未开工）。

---

## 二、开工前必读（按顺序）——这些是项目的"航海日志"与契约

0. **`docs/RESULTS_SUMMARY.md`** —— **最先读**。所有已验证结果集中一处（数据/训练/Tier1-3/OPE/IRL/L2/L5/headline/待办）。
1. **`docs/CHANGELOG.md`** —— 实现路径速览（索引/路线图），一眼看清做了什么、改了什么、为什么。
2. **`docs/IMPLEMENTATION_LOG.md`** —— **最详细**。完整执行记录（append-only）：每个 stage 的交付、关键决策、踩过的坑+修复、教训、下一步。**通读末尾**（Stage 6 重训 / 评估 / 模拟器 / Tier-3 / OPE / L2 / L5 各节）。
3. **`docs/TOOL_TRAPS.md`** —— 工具/环境陷阱（§11 沙盒磁盘满截断[反复出现]、§12 us/ns、§13 ModuleDict 'train'、§14 pyarrow OOM、§15 HPC fd、§16 pd.notna 嵌套、§17 torch2.6 weights_only、§18 pyarrow-string dtype、§19 Grep/Glob 须传 path）。
4. **`docs/spec/01-05_*.md`** —— 5 份**契约**（数据管线 / MDP / 模型架构 / 训练协议 / XAI+评估）。锁定值在此（CQL α=5、γ=0.95、3 阶段 5+15+20ep、按时间划分、batch 256、padding caps 60/15/15/8/14 等）。
5. **`docs/LEAK_AUDIT.md`** —— 泄露审计清单（直接答案/时间/候选三类）+ 复验工具 + 现状（已全过）。论文"效度威胁"素材。
6. **`docs/PROJECT_HANDOFF.docx`** —— 高层领域 + 框架 + 路线图总览。

读完跟我确认你掌握了现状，再开工。所有架构决策已锁定（见 spec + CHANGELOG），不要建议改 task framing。

---

## 三、当前状态（2026-05-26）

**Stage 0–6 完成 ✅（数据修正后重训, seed 42/43/44）；Stage 7 baselines + Stage 8 评估 + XAI L1–L5 + L4 规则合规 全部完成 ✅；剩：§12 Override / L1 缺件 / 3-seed eval 聚合 / BC-IQL baseline。**
**👉 所有数字/结果集中在 `docs/RESULTS_SUMMARY.md`（单一事实来源，先读它）。** 要点：

- **数据干净 + 奖励两 bug 修正后重训**：snapshots_v2(1,996,572)。本阶段修了 delay 的 **train_id 跨月** + **Apr–Jul Movements +1h** 两个 bug，重算 reward(08→09→10) + 状态 patch(23) + 重训前体检(24)全过。（更早的 episode 跨月 / lateness / platform_dev 三 bug 亦已修。）
- **Stage 6 全量训练完成**（修正数据, 3-seed 一致）：seed42 best .9823 / seed43 .9832 / seed44 .9830。**但 eval(Table I/OPE/L1/L2/L4/L5) 目前仅 seed42 → 3-seed mean±std 待**。
- **Stage 7/8 评估（单 seed42）**：Tier-1 set-only **.957**；Table I 分层 vs baseline(难 strata 碾压:call_on .881 vs ≤.05、platform_dev .903 vs 0)；P2.6 模拟器验证(occ **.94**/tp **.86**)；Tier-3 安全优先 **genuine_unsafe 0%**;OPE/FQE total≈0、delay 持平、wait 改进。
- **XAI 五层全部完成 ✅**：L1(IG 显著性+忠实度,全量)、L2(Q-gap Shapley)、L3(反事实)、**L4(规则合规:19 条 Hao 审签规则,模型 81.0% vs 人 85.7% 硬规则合规,全量)**、L5(IRL,定性:信号员选路 delay 居首)。**L1 仍缺 attention rollout + 面板热图**(记账);**§12 Selective Override 未做**。
- **诚实 headline**：专家级、安全、可解释复制 + 胜过朴素 baseline + 与人持平(不主张超越人类)。OPE+IRL 双向解释 delay 中性 = 稀疏奖励低估 delay(reward-design,非架构)。

---

## 四、材料清单（写论文用的"原料"在哪）

| 类别 | 在哪 | 备注 |
|------|------|------|
| **方法/契约** | `docs/spec/01-05_*.md` | 数据/MDP/模型/训练/XAI+评估 = 方法学章节 |
| **执行/决策/踩坑** | `IMPLEMENTATION_LOG.md` + `CHANGELOG.md` | 每个决策+bug+修复+为什么 |
| **效度威胁** | `LEAK_AUDIT.md` + 各泄露修复记录 | 论文 validity threats |
| **领域知识** | log 内（X-prefix 信号、platform 7 pilot line、TRUST id 结构 Table 3.6、数据 10 段时间 gap…） | 数据描述 |
| **代码** | `src/railrl/`（mdp/encoders/policies/algorithms）、`scripts/`（data/mdp/train） | git 有历史 |
| **结果（汇总）** | **`docs/RESULTS_SUMMARY.md`** + `outputs/eval/*.json`（Tier1/2、baseline Table I、OPE、IRL、L2）+ `outputs/train/cql_seed42&43/` | 单 seed42；mean±std 待 seed44 |
| **待产出** | seed44→3-seed mean±std / 学习型 baseline(BC/IQL) / XAI L1+L4 / docx + 图表 | 收尾 |

---

## 五、路线/计划（spec 04 §10、spec 05）

**Stage 6** ✅(seed42/43/44) → **Stage 7** ✅(B0/B0'/B0'' 非学习 baseline Table I;BC/IQL 待) → **Stage 8** ✅(Tier-1/2 + P2.6 模拟器验证 + Tier-3 安全优先 + OPE/FQE) → **Stage 9-11 XAI**：L1✅/L2✅/L3✅/L4✅/L5✅;**§12 Selective Override 待、L1 attention-rollout+面板热图 待** → **Stage 12** 论文（不由 AI 代写,只记录结果）。

---

## 六、我的工作纪律（请严格延续）

- **记录**：计划、进度、**每一次修改/每一个教训**都追加进 `IMPLEMENTATION_LOG.md`（**只追加、不删旧**，扛住 context 压缩）；工具坑记 `TOOL_TRAPS.md`；路径索引同步 `CHANGELOG.md`。用任务列表（TodoList）跟踪。
- **核心原则："不妥协，将错就错的行为要杜绝"**：绝不在错的/旧的/错位的数据上往下建；进下一步前验证数据正确性，且用**独立信号**交叉验证（栽过："matched 100% 但 sample_id 指向错的决策"，靠 label 才发现；这次 episode 跨月也是靠 pass 时间跨度这个独立信号查出）。遇到岔路（设计取舍、可能偏离 spec、需重算）**先停下**，用 **AskUserQuestion** 把选项+推荐摆给我，让我定。核对结果时把**算术逐项对一遍**。
- **回应**：用中文；简洁；核对显式 reconcile；主动指出正确性隐患；岔路用 AskUserQuestion 给带"(推荐)"标记的选项。

---

## 七、协作方式 + 运行环境

- 所有代码在 `E:\Claude\RailRL_v2`。**我在 Windows 端 git commit / 跑脚本，把控制台输出贴给你核对**——你别假设脚本跑过。
- **沙盒（Linux）跑不了 torch/torch-geometric、装不了 pyarrow（`/sessions` 磁盘 100% 满 → virtiofs 会截断刚写的文件、`ast.parse` 假报 SyntaxError，§11）**。所以：纯 numpy/pandas 逻辑可沙盒小数据测；**改完文件用 Read 工具（Windows 真文件）确认**，别信沙盒截断视图；torch 代码靠 Windows/服务器跑。
- **三处硬件**：本地 **RTX 5070 8GB + 32 核 Ryzen + 4TB**（sanity / 审计 / 数据 patch，batch 调小）；**A100 40GB 服务器**（全量训练 batch 256，但系统盘紧 ~10-12GB 空 + **单次租用仅 12h** → 用 `--resume`）。
- **服务器命令前缀**（HPC sapphire）：
  `cd /rds/homes/h/hxn886/ondemand/RailRL_v2 && env PYTHONUNBUFFERED=1 PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MPLBACKEND=Agg /rds/homes/h/hxn886/virtual-environments/my-virtual-env-sapphire/bin/python <script> ...`

---

## 八、关键不变量（别破坏）

- **sample_id** 是行物理 id（reward 按它对齐，4.6.5 label agreement 100%）——任何重分段/重排都不改 sample_id↔reward/state 对应。
- **vocab**：track_id 268 / signal 123 / route 278 / train 2184（编码器据此，勿改）。
- **时间划分**：train<2024-02-01 / val<2024-03-01 / test≥（按 episode 起始时间，episode 已时间局部 + 切 split 边界，零泄露）。
- **锁定超参**：CQL α=5、γ=0.95、3 阶段 5+15+20ep、batch 256、AdamW lr3e-4、warmup→cosine、grad clip 1.0、target soft τ=0.005、aux 权重 λ_route=0.5/λ_time=0.2、time bucket 边界 [98,121,153,204]s。

---

## 九、已知小项（记着，别忘）

- `is_last_in_episode` 旧定义按 t==max 会在时间戳并列时标多行；**终止性按 position 判**（transitions/canonical 已正确）。
- Stage 6 完成后：跑 seed 43/44；聚合 mean±std（`05_aggregate_results` 待写）。
- loader 吞吐 encode-bound（~5ms/行）；若 Stage 6 太慢可加 worker 或 profile `encode_snapshot`。
- `end_platform_id` 仅 28% 路线有映射（含合理 through/depot）→ platform_dev 保守欠检测，记入论文局限（同 approach_distance 48% / delay_change 6% 一类）。

请先读 `docs/RESULTS_SUMMARY.md` + 按"二"顺序读文档，确认现状后继续（当前剩余：§12 Selective Override / L1 attention-rollout+面板热图 / 3-seed eval 聚合 / BC-IQL baseline / L5 奖励恢复改进；论文不由 AI 代写，只记录结果）。
