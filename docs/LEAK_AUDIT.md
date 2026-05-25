# 泄露审计清单（Leak Audit Checklist）

> 目的：把"模型会不会偷看答案/未来"的所有途径列清楚 + 每条的防御 + 如何复验 + 现状。
> 触发：Stage 6 全量 ep1 出现 val route_acc=0.915 / time_acc=0.653（高），需确认是
> **任务可模仿**而非**泄露**。本清单也是论文"效度威胁(validity threats)"一节的素材。
> 复验工具：`scripts/mdp/06_run_leak_audit_full.py`（assert_no_leak）、`07_audit_snapshots.py`
> （数值分布）、`21_audit_leakage.py`（可解释性基线）。契约见 `spec/01 §13`、`spec/02 §7`。

---

## A. 直接答案泄露（state 不得含"答案"字段）
| # | 途径 | 防御 | 复验 | 状态 |
|---|------|------|------|------|
| A1 | focal_signal 进 state | spec 01 §17.5 永久禁止；BANNED_STATE_FIELDS 含 focal_signal/focal_signal_id | 06 (Check3 banned scan) | ✅ |
| A2 | chosen_route_id / chosen_action_idx 进 state | 在 BANNED_STATE_FIELDS | 06 | ✅ |
| A3 | is_focal_signal / is_focal_route 图节点标记 | 禁止；只允许 is_focal_train | 06 | ✅ |
| A4 | reward 中间量(outcome/delay_change/headway/r_*) 进 state | 21 项 BANNED_STATE_FIELDS | 06 | ✅ |
| A5 | 候选顺序泄露(chosen 总在固定位) | route head 是 h_train·h_route 点积**按嵌入打分、不看位置**；候选由 routes_from(focal_signal) 生成 | 21 基线A/B（首候选/最常见 index acc） | ✅ 复验 |

## B. 时间泄露（未来信息不得进 state；t'>t 禁用）
| # | 途径 | 防御 | 复验 | 状态 |
|---|------|------|------|------|
| B1 | train/val/test 划分时间泄露 | 按时间·**整 episode** 划分（spec 04 §4.1）；4.7.2d 修了"跨月 pass→整段灌进 train"的泄露（129,021 行→0） | 14 输出泄露=0；21 split 重检 | ✅ |
| B2 | 实现态 Movements(actual/timetable_variation) at t'>t | spec 01 §13.1：scheduled(gbtt/planned)允许；realized 仅 **actual_ts≤t** 可用 | lateness 用 ≤t 窗口（current_lateness_s） | ✅ 4.7.2d 修 |
| B3 | TD 事件 time>t 进 state（占用/信号/event token） | 各 history 按 t_ns 二分只取 ≤t；**us/ns 单位 bug 已修**（否则总取最后一次事件=未来泄露） | 07（event time_delta_s ≥0、%>1e8=0） | ✅ |
| B4 | schedule_outlook 用 actual | 只用 gbtt_timestamp；planned_platform 强制 int 1-7 | 06 (Check4) | ✅ |
| B5 | aux 标签(τ/time_bucket)、reward 用 hindsight | **允许**（标签/奖励是 hindsight）；但只经 sample_id join 给 loss，**不进 state** | 设计：time_labels_v2/reward 是 sidecar，非 state 列 | ✅ |

## C. 候选/掩码泄露
| # | 途径 | 防御 | 状态 |
|---|------|------|------|
| C1 | 候选 mask 用 t>t 可见信息 | action.py 从 focal_train.current_tc BFS（time≤t）+ routes_from(focal_signal) | ✅ |
| C2 | f_trts_pressed 用 focal_signal 的 platform | 只用 planned/current platform（spec 02 §4.10） | ✅ |

## D. "高精度=泄露还是易任务？" 的判据（本轮重点）
- **基线对照**（`21_audit_leakage.py`）：若『总选第一候选/最常见 index/匹配 planned_platform』等傻基线已接近 0.9，则 val route_acc 0.915 是**任务可模仿**（planned_platform 强预测 route + 近 FCFS + 候选集小 mean≈2.7），非泄露。
- **action_acc 0.729 @ ep1** ≈ wait 占比(~73%) → 未训练 Q 偏好 wait 的基线，非泄露信号；Phase B/C 升过基线（sanity 达 0.946）。
- **time_acc**：5 桶均衡 chance=0.20；state 含占用/接近信息天然预测 τ-bucket，高于 chance 合理。
- 决定性仍是 A/B/C 项（无 banned 字段 + 时间干净 + 实现态 ≤t）。

## 现状结论（Hao 本地复验，2026-05-22）
- [x] **07** `07_audit_snapshots.py`（抽 242,620）→ **LEAK PASS / TIME PASS**：banned in state=0、center 全 track、!=1 focal=0、platform∉1-7=0；event time_delta_s %>1e8 garbage=**0.000%**、deltas≥0；退化子图 0%、caps 遵守。**→ 直接答案泄露(A1-4) + 时间泄露(B3) 排除。**
- [x] **21** `--split val`：set 14,254/wait 35,746（wait 0.715=act 基线）；route 基线 **总选第一候选=0.550**、planned 匹配候选预测器=0.974（但仅覆盖 12% set 行，因 end_platform 仅 28% 映射）。**test 一致**（首候选 0.599、预测器 0.966）。
- [x] **判读**：模型 val route_acc 0.915 **高出傻基线(~0.55-0.60)约 30pp** → 确在用丰富 state（非位置/平凡捷径；route head 是位置无关点积，无法利用"候选0"规律）。结合 07 零 banned 字段 + 时间划分干净 → **无泄露迹象**；0.915 与"可学习的模仿任务 + 干净划分"一致。
- [x] **06** `06_run_leak_audit_full.py --sample 100000` → **ALL PASS**（全 7-check assert_no_leak，pct_passed=100）。修了两个潜伏 bug：整表读 OOM→流式；`_row_to_snapshot` 用 `center` 而文件存 `state_center`（别名修复）。
- **✅ 泄露审计三件套（06+07+21）全通过 → 无泄露，结论锁定。**
- **论文"效度威胁"可写**：已系统审计直接/时间/候选三类泄露（06+07 通过）；高模仿精度由任务结构（timetable + 近 FCFS + 小动作集 mean≈2.6）+ 干净时间划分解释，模型显著高于平凡基线说明在学真实 state→决策映射。残留局限：end_platform 仅 28% 映射 → platform_dev/平台基线覆盖窄（如实记）。

---
## 更新日志
- 2026-05-22 v1.0 — 初版（Stage 6 ep1 高精度触发；汇总 4.7.2d 各项泄露修复 + 基线判据）
