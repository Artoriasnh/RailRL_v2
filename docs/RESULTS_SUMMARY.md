# RailRL v2 — 已验证结果汇总（单一事实来源）

> 目的：把所有**已验证**的结果集中一处，便于随时查看、防记忆丢失。**仅追加**。
> 时序/细节见 `IMPLEMENTATION_LOG.md`；路线图见 `CHANGELOG.md`。
> ⚠️ 除非注明，**所有结果均为单 seed（seed 42）**；**seed 43 仅有训练 val gate、未跑 test 评估**；**seed 44 未训 → 暂无 3-seed mean±std**。
> 更新于 2026-05-26。

## ⚠️ 运行状态（full vs smoke）—— 重要，别把"模块就绪"当"全量结果"
| 组件 | 实跑 | 状态 |
|---|---|---|
| 训练 seed42/43/44 | 全 40-epoch,§11 全过,best val .9823/.9832/.9830 | ✅ 全量(3-seed 训练一致) |
| eval/01 seed42 test | 全 test | ✅ 全量 |
| eval/01 seed43 test | 未跑 | ❌（seed43 只有训练 gate） |
| 06 baseline Table I | 全 test | ✅ 全量 |
| 02 模拟器验证 | 全 | ✅ 全量 |
| 03 Tier-3 | --max-decisions 1500 | ✅ 全量（配置上限） |
| 04/05 OPE | --max-batches 4000 --warm | ✅ 全量（拟合按设计 cap、eval 全 test） |
| 08 L5 qtable | **全量 fit `--max-batches 4000` ✅（Hao 确认）** | ✅ 全量 |
| 09 L5 IRL | --n-boot 300（基于全量 qtable） | ✅ 全量 |
| **07 L2** | **全量 12 决策**(completeness 全 ±0.0000) | ✅ 全量 |
| **10 L1** | **全量 12 例 + 300 决策审计**(忠实度 448 distinct) | ✅ 全量 |
| **L4 规则合规** | **全量 test(seed42)：模型 81.0% vs 信号员 85.7% 硬规则合规** | ✅ 全量（19 条规则 Hao 审签；修了 focal_signal 格式 bug） |
| **3-seed eval 聚合 (Tier-1/2 + L4 + §12 + OPE)** | seed42/43/44 全 | ✅ **全量**（CQL overall 96.0±0.2、L4 模型 85.0±2.9% vs 人 85.7%、§12 agreement 96.0±0.2 / consider-override 0.22±0.14%、OPE wait ΔV +0.054±0.014/delay -0.008±0.020/total +0.04±0.03） |

---

## 0. 训练/数据设置
- 1,996,572 可用 snapshot；set/wait = 546,418 / 1,453,205。
- 划分（时间+episode-local）：train 1,472,064 / val 186,145 / test 338,363（train<2024-02-01、val<2024-03-01、test≥2024-03-01）。
- vocab track/signal/route/train = 268/123/278/2184。
- CQL α=5、γ=0.95、3 阶段(A5/B15/C20)、AdamW 3e-4、target τ=0.005、batch 256。
- 奖励(修正后)：`r_total = 1.0·r_delay + 0.5·r_throughput + 1.0·r_headway + 0.3·r_wait`（r_total 精确=四分量和）。均值：total −0.106(std .587)/ delay −0.010 / thru +0.136 / head −0.013 / wait −0.218。delay-change 覆盖 685,715(~34%)。**r_delay 有效占比~10%（稀疏+量级小，尽管名义权重 1.0）**。

## 1. 奖励两 bug 修正（2026-05-24）
- **Bug#1**：`compute_delay_changes` 按原始 train_id 分组(每月复用)→ out_window 77% → delay 覆盖 6%。修：按 2h gap 切单次行程。→ 覆盖 24%。
- **Bug#2**：Apr–Jul 2023 Movements +1h(双重 BST)。修：`correct_movements_bst` 在 [2023-03-17,2023-08-05) −1h。影响~41% train(reward+Movements 派生状态)；**val/test(2024) 不受影响 → 之前 test 评估有效**。
- 重算 reward(08→09→10) + 状态 patch(23, 5 字段) + 重训前体检(24, 5 段全过)。

## 2. Stage 6 全量训练（修正数据）— §11 gate 全过
- **seed42**：A route .925/time .699；B Q-top1 .963/|Q| 57.8/L_cons .089；C Q-top1 **.981**；best val action top-1 **.9823@C8**。|Q| C 末 ~105–124 有界(|Q|<100 仅 B 闸)。
- **seed43**：best **.9832@C20**(复现；|Q| 比 42 更受控)。**seed44 待**。

## 3. Tier-1（test, seed42）
- action top-1 all **.9882** / **set-only .9572**。wait_rate 信号员 .7272 = 模型 .7274。route_head .9498。time_head .667。Q-gap 均值 17.2、frac_argmax .987。

## 4. Table I — 分层 set-only top-1（模型 vs 非学习 baseline）
| stratum | n_set | CQL | plat-pref(B0') | first(B0'') | random(B0) |
|---|---|---|---|---|---|
| overall | 92,280 | **.957** | .531 | .528 | .322 |
| late_train | 24,964 | **.970** | .616 | .612 | .353 |
| advance | 1,760 | **.917** | .709 | .702 | .320 |
| call_on | 6,373 | **.881** | .048 | .028 | .093 |
| platform_dev | 308 | **.903** | .000 | .000 | .133 |
| priority_compete | 16,066 | **.925** | .511 | .511 | .314 |
| trivial | 42,807 | **.975** | .557 | .557 | .344 |
| (unusual_id) | 26 | .808 | .615 | .615 | .299 |
- **难 strata(call_on/platform_dev)上启发式连随机都不如**(信号员的正确动作=非默认)→ 模型在最难决策上加大价值。**只用 set-only**(all-decisions 被各方法 wait/set 倾向污染)。
- baseline 定义：B0 随机(合法动作均匀)、B0' 计划站台-否则首候选、B0'' 首候选。注：候选顺序是 route_id 字母序(非站台序)→ B0'≈B0''。

## 5. P2.6 模拟器（eval-only）— 已验证
- occupancy-onset Spearman **.94**、throughput Spearman **.86**(headway 分位 p5→p1 校准)。绝对吞吐~73%(保守、delta 抵消)。验证门 PASS。
- 参数：`outputs/simulator/parameters.json`（route_running/platform_dwell/min_headway/tc_traversal/aspect_clear_lag）。

## 6. Tier-3 安全优先（seed42, 1489 个偏离）
- 偏离率 **4.3%**(3954/92280)。**genuine_unsafe 0.0%**(路线合法 100% + 单独可行 100%)。conflict_indeterminate 9.3%。**冲突负荷 headway-wait Δ ≈ +0.07(≈0)**。
- 不对称诊断：alone intrinsic finish Δ **+14.2s**、conflict +3.3s；87/87 "completion-unsafe" 单独跑都能跑完 → fixed-others 伪影、非模型。
- v1.2 classify：安全(合法+单独可行,模拟器无关)≫ delay。原始 throughput +6.29 是共享可变状态伪影(已用 replace() 修)。

## 7. OPE / FQE（seed42）
- **total ΔV ≈ 0**(分解 +0.041[−0.022,+0.090])→ 与人持平。
- **delay ΔV +0.020[−0.040,+0.070] ≈ 0(中性)**。【⚠️ 早先单跑 −0.238 是 **warm-start 尺度失配伪影**;fresh-init 复核 = +0.020。】
- **wait ΔV +0.035[+0.027,+0.044] 显著正**(模型减少等待)。throughput −0.012[−0.018,−0.007](微负)。headway +0.005(≈0)。
- Σ-check：total +0.041 ≈ Σ分量 +0.048(自洽)。
- 与 sim +14s 自洽：模型选"稍长但更不拥堵"的路 → 换更少等待 → 真实 delay 打平。
- 效度：FQE 估计、对 4.3% 偏离 OOD 可能乐观;单 seed。

## 7b. L1 模型显著性 — **全量完成 ✅（12 例 + 300 决策审计）**
- `src/railrl/xai/l1_attention.py` + `scripts/eval/10_l1_saliency.py`。**Integrated Gradients** 对节点 cont+binary 特征(零基线积分)→ 每节点显著性 + top 节点。
- **忠实度审计(spec §7.5)全量 PASS**：**448 distinct top-10 节点 / 300 决策**(阈值 50;稳步增长 50→177、300→448)→ 强非退化、归因随决策变化。
- top 节点合理：chosen route + focal 车高位;train-centric strata(late/platform_dev/advance/call_on)focal 车进 top-5;priority 靠 tracks+signals;trivial 低弥散。
- **局限**：attention rollout 未提取(PyG HGTConv 不暴露,spec 自认 IG 更可靠);面板热图延后(缺 `data/reference/panel_layout.json`,需手工 TC→像素坐标)。`outputs/eval/l1_saliency.{md,json}`。

## 8. L2 决策解释 — **全量完成 ✅（12 决策）**
- Q-gap(chosen vs runner-up)6 特征组**精确 Shapley**,**12 决策完备性全 ±0.0000**。route-vs-route 决策由 Route features 主导;route-vs-**wait** 决策(trivial/late_train)由 Subgraph state + Sequence summary 主导 + 大负 base(="是否动手"由当前占用/事件驱动,合理)。NL 理由生成。`outputs/eval/l2_explanations.{md,json}`(12 条)。
- **cosmetic 注**:偏离决策上 NL 标题显示信号员路线、`⟵chosen` 标的是模型 argmax("chosen"一词重载),数字无误。

## 9. L5 IRL — 定性 ✅
- **global IRL 全负 = wait 混淆伪影**(wait 动作分量-Q 低、信号员 73% 选 wait → 条件 logit 误读)。**global 作废**。
- **SET-only(routes-only, 92,280)**：归一化 w **delay +1.45(最高)** > throughput +0.91 > wait +0.61；**headway −1.03 不可解读**(共线性+弱特征)。
- 结论(定性)：信号员选路 **delay 居首** → 与 OPE 互证(信号员重视准点,但稀疏奖励低估 delay → 模型 delay-中性)。`outputs/eval/l5_irl_weights.json`。

## 10. 诚实 headline
**专家级(95.7% set-only)、安全(0% genuine-unsafe、冲突中性)、可解释的复制**,胜过朴素 baseline(尤其难 strata),与人持平(总回报+delay)、小幅减少等待。**不主张超越人类**。OPE+IRL 双向解释 delay 中性 = 稀疏奖励低估 delay(reward-design finding,非架构)。

## 11. 待办
**进行中 / 下一步**
- ~~**L4 规则合规**~~ ✅ **完成(2026-05-27, seed42 全量)**：19 条规则 Hao 审签 → `rule_base.py`+`l4_rules.py`+`03_finalize.py`(已生成 `rules.parquet` 19 行)+`12_l4_compliance.py`。**结果：硬规则合规率 模型 81.0%(2376/2933) vs 信号员 85.7%(2515/2934)**；判定 68% 集中在 call_on(模型 84.75% vs 人 89.5%)；软规则全 inert(目的地被 leak-audit 隐藏)。`outputs/eval/l4_compliance_seed42.json`。修过 focal_signal 裸数字格式 bug(decision_signal 从候选 route_id 还原)。**L4 增强(可选,未做)**:按 rule_id 拆 557 不合规、L4×Tier-3 交叉表+§12 闸、headcode→方向映射激活软规则。

**🔴 必须在后面完成(记账,别遗漏)**
- **L1 缺失部分**:① attention rollout(需 hook PyG HGTConv 内部 / 换支持 attention 的实现);② 面板热图(需手工建 `data/reference/panel_layout.json` = TC/信号→`derby_all.png` 像素坐标)。当前 L1 只有 IG + 忠实度。
- ~~**§12 Selective Override**~~ ✅ **完成(2026-05-27, seed42 全量)**:`deploy/selective_override.py`(三闸+L2 faithfulness+δ_L3 敏感性+双 gate_l4 口径)+`scripts/eval/13_selective_override.py`。**结果**：agreement set-only **95.7%**/全 98.8%；consider-override 极罕见 **0.2%**(δ=0.5+refined,3/1500;δ↓0.1→0.4%,对阈值稳健);瓶颈=gate_l3(模型少有 >0.5 reward单位改进,与 Tier-3/OPE 自洽);3 张 PRIMARY 覆盖卡(2 在 call_on)可作论文例子。gate_l4 改"非 non-compliant 即放行"(refined 主口径,literal 对照退化为~0)。`outputs/eval/selective_override_seed42.json`。
- **L5 奖励恢复改进**:当前 IRL 受 FQE-OOD + 特征共线性限制只能定性;未来想办法拿干净权重(更密/去相关特征、或换 IRL 形式)。

**多 seed / baseline / 收尾**
- eval(Table I/OPE/L1/L2/L5)目前**仅 seed42** → 在 seed43/44 跑 eval → **3-seed mean±std**(发表硬要求;训练已 3-seed 一致 .9823/.9832/.9830)。
- **学习型 baseline ✅** —— BC-HG (seed42): **0.9178** ; IQL (seed42): **0.9409** ; CQL (**3-seed mean ± std**): **0.9601 ± 0.0021**(seed42/43/44 = .9572/.9613/.9619)。**Table I 完整(set-only top-1)**：非学习 ~53% ≪ BC-HG 91.8 < IQL 94.1 < CQL **96.0 ± 0.2**。难 strata gap 仍最大：platform_dev BC 77→IQL 86→CQL **89.6 ± 0.9**；call_on BC 80→IQL 82→CQL **89.1 ± 0.7**；advance BC 84→IQL 86→CQL **93.4 ± 1.3**。3-seed CQL **std<1pp 占绝大多数 → 模型极稳定**。结论：**BC→offline-RL 大跳跃 + CQL≈IQL（封住 spec §1.2 "非算法假象"）+ 增量集中难 strata**。B1 BC-flat 暂缓。
- **L4 hard-rule 合规 (CQL, 3-seed)**: 模型 **85.05 ± 2.90%**(values .8101/.8646/.8768);信号员 **85.72%(精确同值跨 seed ✓**)。**结论强化**:模型与人**在 Plan 遵守率上统计无差**——原单 seed "模型 81 vs 人 86" 实际处在 seed 噪声范围内。
- docx 重建(可选)。

## 12. 关键输出文件
- `outputs/eval/cql_seed42_best_test_metrics.json`（Tier-1/2）
- `outputs/eval/baseline_accuracy_table.json`（Table I）
- `outputs/eval/ope_fqe_seed42_{total,delay}.json`、`ope_fqe_decompose_seed42.json`（OPE）
- `outputs/eval/l5_irl_weights.json`（IRL）、`outputs/eval/l5_qtable.parquet`（FQE 分量 Q 表）
- `outputs/eval/l2_explanations.{md,json}`（L2）
- `outputs/train/cql_seed42/`、`cql_seed43/`（模型 + train_log）
- 脚本：`scripts/eval/01`(Tier1/2) `03`(Tier3) `04/05`(OPE) `06`(baseline) `07`(L2) `08/09`(L5)；`scripts/simulator/01/02`；`src/railrl/xai/{l3_system,l2_qdecomp,l5_irl}.py`、`src/railrl/eval/metrics.py`
