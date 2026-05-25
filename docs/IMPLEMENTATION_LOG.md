# RailRL v2 Implementation Log

> **目的**：跨对话、跨 context-compression 的**持久实施记录**。
> 记录每个 stage 实际交付的内容、关键决策、踩过的坑、下一步。
>
> **使用方法**：新对话开场让 AI 助手先读这份文档 + `PROJECT_HANDOFF.docx`，
> 它就知道项目现在在哪、可以做什么。
>
> **维护规则**：每完成一个 stage（或重大里程碑），在文档末尾**追加**一节，
> 不要删旧记录。这是项目的"航海日志"，不是计划。

---

## 文档关系一览

| 文档 | 用途 | 更新频率 |
|------|------|---------|
| `docs/spec/01-05_*.md` | **契约**：应该做什么、长什么样 | 仅在重大设计变更时（v1.1, v1.2 ...） |
| `docs/PROJECT_HANDOFF.docx` | **总览**：领域 + 框架 + 路线图 | 仅在大块结论变化时 |
| `docs/IMPLEMENTATION_LOG.md` | **执行**：实际做了什么、什么时候、踩过哪些坑 | **每个 stage** 完成时追加 |
| `README.md` | **入口**：5 行说明 + 路径 | 极少 |
| Git commit history | **原始记录**：每次代码变更的最小单元 | 每次 commit |

---

## 整体进度速查表

| Stage | 描述 | 状态 | 完成日期 | Spec 依据 |
|-------|------|------|---------|----------|
| Stage 0 | Spec 锁定（5 份） | ✅ done | 2026-05-19 | n/a |
| Stage 1 | 数据 pipeline 验证 + 环境 setup | ✅ done | 2026-05-19 | spec 01 |
| **Stage 2** | **决策点 + 候选动作 + 8 特殊性 flag** | ✅ **done** | **2026-05-19** | spec 02 §2-§4.10 |
| Stage 3 | 新 snapshot builder (state + leak audit + episodes) | ✅ done（数据已重建+审计 READY）| 2026-05-20 | spec 02 §4-§8 |
| Stage 4 | 主模型（HGT + Transformer + Q + 2 aux heads + CQL） | ✅ **done**（4.1-4.6 模型 + 4.6.5 真 reward + 4.7.1 losses + 4.7.1.5 时间划分 + 4.7.2a-c transitions/trainer + **4.7.2d** 流式 loader+分层；含 episode 跨月修、lateness 修、platform_dev 修） | 2026-05-22 | spec 03 + 04 |
| Stage 5 | Sanity 训练 50k subset | ✅ **done**（全 §11 gate PASS：route .73/time .41/Q-top1 .87→.95；泄露审计 06/07/21 全过） | 2026-05-22 | spec 04 §11 |
| Stage 6 | 全数据训练 3 seeds | 🔨 **in progress**（seed 42 全量训练中 ~17h；trainer 已接流式+分层+§11 gate+resume；43/44 next） | — | spec 04 §10 |
| Stage 7 | Baselines (B0/B0'/B1/BC) | ⏳ pending | — | spec 04 §1.3 |
| Stage 8 | 评估框架（3-tier + Replicate-AND-Improve） | ⏳ pending | — | spec 05 §1-§5 |
| Stage 9 | XAI L1/L2/L5 | ⏳ pending | — | spec 05 §7-§11 |
| Stage 10 | P2.5 规则库 + P2.6 仿真器 | ⏳ pending | — | spec 05 §13-§14 |
| Stage 11 | XAI L3/L4 集成 + Selective Override | ⏳ pending | — | spec 05 §9-§12 |
| Stage 12 | 论文撰写 | ⏳ pending | — | n/a |

---

## Stage 0 — Spec 锁定（2026-05-19）

### 交付物
5 份 spec 文档（共 4,668 行）：
- `docs/spec/01_data_pipeline.md` (1,130 行) — §3+§4 of ESWA paper
- `docs/spec/02_mdp_formulation.md` (972 行) — MDP + 状态 schema + leak audit
- `docs/spec/03_model_architecture.md` (759 行) — HGT + Transformer + Q net
- `docs/spec/04_training_protocol.md` (768 行) — CQL/IQL/BC + 3 阶段
- `docs/spec/05_xai_and_eval.md` (1,039 行) — 5 层 XAI + 评估 + 部署

### 关键决策（一并锁进 v2 各模块）

| 决策 | 值 | 出处 |
|---|---|---|
| Task framing | 调度游戏 + 结构化动作 `{wait} ∪ {(T, R)}` | spec 02 §1.2 |
| State 不含 focal_signal | 永久禁止 | spec 01 §17.5 |
| 主算法 | CQL（IQL 对照、BC 基线） | spec 04 §1 |
| 编码器 | HGT (3 层 × 4 head) + Transformer (4 层 × 4 head) | spec 03 §3-§5 |
| Q 网络 | per-action MLP（动态 |A_t| 友好） | spec 03 §6 |
| 辅助监督头 | 2 个（route + time）—— **priority head 已 drop**（详见 spec 03 §7.3）| spec 03 §7 |
| 训练 protocol | 3 阶段 5 + 15 + 20 = 40 epochs | spec 04 §3 |
| Derby_info 物理特征 | 进入 route_emb（length / speed / grad / gap_time / n_points）| spec 03 §3.1 |
| Leak 防御 | `assert_no_leak()` 每 batch 跑 + 21 个 banned field | spec 02 §7 |
| 主模型 vs Baselines 顺序 | **主模型先**（用户 2026-05-19 决策） | PROJECT_HANDOFF §15 |
| Priority head 定位 | 为 improvement 保留，不为 imitation accuracy | PROJECT_HANDOFF Ch 2.6 |

### Spec 01 §17 已锁定的 4 个开放问题
1. `pass_assignments.parquet` 单独物化（新 Stage 6）
2. `decision_rewards.parquet` 含 wait 负样本（label 列区分）
3. `approach_distance` 也对 wait 算（注意 leak 见 §17.5）
4. `next_tc_headway` for wait → NaN（r_headway=0）

### Spec 01 §17.5 新增 leak 审计点
- 子图必须 centered on focal_train.current_tc（不是 focal_signal）
- 没有 `is_focal_signal` / `is_focal_route` flag（只有 `is_focal_train`）
- 候选 mask 推导只能用 time≤t 可见信息
- `f_trts_pressed` 必须用 planned/current platform，不能用 focal_signal's platform

---

## Stage 1 — 数据 pipeline 验证 + 环境 setup（2026-05-19）

### 交付物（10 个文件，~1,200 行）

| 文件 | 行数 | 作用 |
|------|------|------|
| `src/railrl/config.py` | 351 | 重写：v2 layout 路径解析 + 所有 spec 常量锁定 |
| `src/railrl/cli.py` | 90 | 新写：3 个 console 入口（inventory/decisions/infrastructure） |
| `src/railrl/data/static_graph_view.py` | 78 | 新写：从 v1 snapshot.py 抽出 `StaticGraphView` 工具类 |
| `src/railrl/data/reward_calibration.py` | 修 | 改 import 用 static_graph_view |
| `src/railrl/data/reward_features.py` | 修 | 同上 |
| `src/railrl/p2_data_eng/__init__.py` | 38 | 兼容 shim：v1 scripts 04-15 透明用 `railrl.data.*` |
| `scripts/data/00_verify_pipeline.py` | 344 | spec 01 §16 自动验证（49 项 sanity） |
| `tests/conftest.py` | 8 | pytest 路径配置 |
| `tests/test_config.py` | 193 | 路径 + 锁定常量测试 |
| `tests/test_imports.py` | 99 | 14 个 data 模块 + cli + shim 全部 import |

### 关键决策
- **路径解析**：DATA_DIR → `data/raw/`，REFERENCE_DIR → `data/reference/`，outputs 扁平（drop `p2_data_eng/` 包装）
- **StaticGraphView 拆分**：v1 `snapshot.py` 包含 BFS 子图提取（binary task 用）+ 加载工具类。后者抽到 `static_graph_view.py`，前者 spec 02 重写
- **兼容 shim**：保持 v1 scripts 04-15 不改源码也能跑

### 踩过的坑
- **Edit 工具截断**：长字符串多次出现写入截断，需 bash heredoc / cp 修补
- **pyc 缓存**：mtime 不更新时旧字节码仍生效，需 touch + 删 __pycache__

### 验证结果（用户本地）
- `scripts/data/00_verify_pipeline.py` → **49 passed, 0 failed**

### Git 提交建议
```
Stage 1: data pipeline verification + env setup
- Rewrite src/railrl/config.py for v2 directory layout
- Add cli.py + static_graph_view.py + back-compat shim
- 00_verify_pipeline.py runs 49 spec 01 §16 checks
- pytest sanity suite (test_config + test_imports)
```

---

## Stage 2 — MDP trigger + action + special_flags（2026-05-19）

### 交付物（9 个文件，~1,500 行）

#### `src/railrl/mdp/`

| 文件 | 行数 | 作用 |
|------|------|------|
| `__init__.py` | 17 | 包入口 + 模块说明 |
| `trigger.py` | 337 | spec 02 §2: 决策点生成（PR + approach）|
| `action.py` | 344 | spec 02 §3: 结构化动作 + RouteIndex + feasible_actions + validate_candidates |
| `special_flags.py` | 212 | spec 02 §4.10: 8 个 flag 计算 + FlagSources 数据源声明 |

#### `scripts/mdp/`

| 文件 | 行数 | 作用 |
|------|------|------|
| `01_generate_decision_points.py` | 135 | 输出 `decision_points_v2.parquet` |
| `02_validate_candidates.py` | 102 | 覆盖率检查（target ≥99.5%） |

#### `tests/test_mdp/`

| 文件 | 行数 | 测试 |
|------|------|------|
| `test_trigger.py` | 88 | compute_approach_tracks + summarize |
| `test_action.py` | 148 | RouteIndex + feasible_actions（5 个 case） |
| `test_special_flags.py` | 175 | 8 个 flag 全测 + compute_all_flags + FlagSources |

**测试总计：51 cases**

### 关键决策 / 实现细节

1. **trigger.py 的 wait trigger 算法**（spec 02 §2.3）：
   - 输入：approach_tracks (per-signal) + TD Track state=1 events with trainid_filled
   - Explode：一个 TC 事件可能触发多个 signal 的 wait 触发
   - Dedup：同 (T, S) 30 秒窗口内只留最早
   - Lookahead：用 (train, signal) → sorted PR times 索引做 O(log n) 二分

2. **action.py 候选规则**（spec 02 §3.2）：
   - 4 条规则：起点匹配 + 方向匹配 + prev_routes 不冲突 + platform 软优先
   - 方向用"TC 字符顺序"做粗 heuristic（forward = first ≤ last alphabetically）—— 后续可换显式 direction 表
   - `planned_platform` 是**软优先级**（reorder），不硬过滤——允许 platform 改派

3. **special_flags.py 防 leak 设计**：
   - 每个 flag 都有 `source` 声明，记录所用数据源
   - `f_trts_pressed` 显式禁止用 focal_signal's platform，只用 planned/current
   - `f_late_train` 返回 int（秒）而非 bool，给模型更多信号

### 踩过的坑

1. **`tcs[-0:]` 不是空 list**（Python 切片陷阱）—— 第一版 `compute_approach_tracks` 在 k_hops=0 时返回全部 TC，单元测试发现。修复：显式 `if k_hops <= 0: tail = []`。
2. **Edit 工具尾部截断**：trigger.py 末尾 `summary["per_train_decisions"]` dict 被截断，用 bash 重新拼回。
3. **pyc cache 引发的虚假错误**：`from .static_graph_view import StaticGraphView` 报 "No module named 'railrl.data.snapshot'" —— 是 pyc 还指向旧 import。touch + 清缓存解决。

### 验证状态
- ✅ Sandbox：51 unit tests 全部通过（自实现 assert）
- ✅ 用户本地 (Windows)：`pytest tests/test_mdp/ -v` → **51 passed, 0 failed**（含修 k_hops=0 的 bug）

### 等用户本地跑
- `python scripts/mdp/01_generate_decision_points.py` （~5-10 min, 出 ~727k 决策点）
- `python scripts/mdp/02_validate_candidates.py` （目标 coverage_pct ≥ 99.5%）

### Git 提交建议
```
Stage 2: MDP trigger + action + special_flags (spec 02 §2-§4.10)
- mdp/trigger.py: decision point generation (PR + approach + dedup)
- mdp/action.py: RouteIndex + feasible_actions + validate_candidates
- mdp/special_flags.py: 8 flags + FlagSources for leak audit
- scripts/mdp/01_generate_decision_points.py + 02_validate_candidates.py
- tests/test_mdp/: 51 unit tests
```

### Spec 02 §11 留给后续 stage 的问题状态

| # | 问题 | 处理 |
|---|------|------|
| Q1 | state_nodes_train 是否含子图外的活跃车 | **延后到 Stage 3** spec 03 已建议 yes，加 mask |
| Q2 | 变长 list 怎么 padding | Stage 3 / spec 03 §2.1 已答（cap+mask） |
| Q3 | event_tokens 时间归一 | spec 03 §2.3 已答（log1p） |
| Q4 | 找不到 focal_train.current_tc 的样本 | Stage 3 实现时跳过 + log |
| Q5 | n_candidates 是否硬封顶 | spec 03 §2.1 已答（14） |

---

## 项目结构速查（Stage 2 完成后）

```
RailRL_v2/
├── docs/
│   ├── PROJECT_HANDOFF.docx     ← 高层总览（v1.1）
│   ├── phase2_feature_spec.md   ← v2 状态契约
│   ├── IMPLEMENTATION_LOG.md    ← 本文档（每 stage 追加）
│   ├── spec/                    ← 5 份契约文档
│   └── handoff/
│
├── data/                        ← 760 MB，全部就绪
│   ├── raw/      3 个 CSV
│   ├── reference/ 6 个文件
│   └── domain/   4 个 PDF/DOCX
│
├── outputs/                     ← 已有 v1 产物 + Stage 1-2 新建空目录
│   ├── inventory/ decisions/ infrastructure/ static_graph/
│   ├── event_stream/ rewards/ analyses/ cache/
│   ├── passes/         (Stage 2 待生成)
│   ├── decision_points/ (Stage 2 脚本待跑)
│   ├── snapshots/      (Stage 3 生成)
│   └── _legacy_v1_binary/  (归档)
│
├── src/railrl/
│   ├── __init__.py / config.py / parsers.py / data_io.py / cli.py
│   ├── data/          13 个数据模块 + static_graph_view.py
│   ├── mdp/           ✅ Stage 2: trigger / action / special_flags
│   ├── p2_data_eng/   (v1 兼容 shim)
│   ├── encoders/      ⏳ Stage 4
│   ├── policies/      ⏳ Stage 4
│   ├── algorithms/    ⏳ Stage 4-7
│   ├── eval/          ⏳ Stage 8
│   └── xai/           ⏳ Stage 9-11
│
├── scripts/
│   ├── data/    15 个数据 pipeline 脚本 + 00_verify_pipeline.py
│   ├── mdp/     ✅ Stage 2: 01 + 02
│   ├── train/   ⏳ Stage 4+
│   ├── eval/    ⏳ Stage 8
│   └── xai/     ⏳ Stage 9+
│
├── tests/
│   ├── conftest.py
│   ├── test_config.py / test_imports.py
│   ├── test_data/test_parsers.py
│   └── test_mdp/   ✅ Stage 2: 3 个 test 文件，51 cases
│
├── configs/             (待写, Stage 4+)
└── pyproject.toml + .gitignore + README.md
```

---

## 新对话开场建议（如果 context 被压缩或新会话）

> "请按此顺序阅读：
> 1. `docs/IMPLEMENTATION_LOG.md`（本文档，了解现在做到哪）
> 2. `docs/PROJECT_HANDOFF.docx` §13-§15（业务领域 + reward + leak + 路线图）
> 3. 当前 stage 涉及的 `docs/spec/0?_*.md`
>
> 然后再回答我的问题。所有架构决策已锁定（见上面"整体进度速查表"+ spec），不要建议改 task framing。"

---


---

## Stage 2 修订与教训（2026-05-19 evening）

经历 3 个 bug fix 后总结。**所有 bug 修后单元测试 51/51 通过**，
但过程暴露了几个**工具系统性问题**——专门记录在 `docs/TOOL_TRAPS.md`。

### 本 session 3 个 bug 修订表

| ID | 文件 | 症状 | 根因 | 修复 |
|---|---|---|---|---|
| **BUG-S2-1** | `src/railrl/mdp/trigger.py` | `test_k_hops_zero` 失败：k=0 返回 {'A','B'} 而非空集 | Python 切片陷阱 `tcs[-0:]` 等价于 `tcs[:]` 返回整个 list | 显式 `if k_hops <= 0: tail = []` 分支 |
| **BUG-S2-2** | `src/railrl/mdp/trigger.py` | `test_basic` 失败：`NameError: name 'per_train' is not defined` | Edit 工具沉默截断了 `per_train = dp.groupby(...)` 那一行 | bash heredoc 重写 summarize() 尾部 |
| **BUG-S2-3** | `src/railrl/config.py` | `01_generate_decision_points.py` 失败：`AttributeError: 'config' has no attribute 'TD_PARQUET'` | Edit 工具沉默截断 `TD_PARQUET = CACHE_DIR / "td_data.parquet"` 那行 | Edit 重新插入；用户 `Remove-Item __pycache__` 清缓存 |

### 教训（按重要程度）

#### L1 — Edit 工具长内容沉默截断（最危险）

**发生过 ≥ 3 次**：trigger.py 尾部两次、config.py 一次（可能更多次未发现）。

**症状**：Write/Edit 报告 "successfully" 但实际写入的内容比预期短，
末尾被无声切掉。Python 报 SyntaxError 或 NameError（如果幸运），
否则可能很久才发现。

**对策**（→ `docs/TOOL_TRAPS.md` 详述）：
- 用 bash heredoc (`cat >> file << 'EOF'`) 写超过 200 行或含复杂字符串的内容
- 每次 Edit 长文件后立即 `wc -l` + `tail -n 5` + `python3 -c "import ast; ast.parse(...)"`
- 严重的配置类文件（如 config.py）单元测试覆盖**每一个常量存在**

#### L2 — Python `tcs[-0:]` 切片陷阱

`tcs[-0:]` ≡ `tcs[0:]` ≡ `tcs[:]` ≡ **整个 list**（不是空 list！）。

任何接受 k 作为"取最后 N 个"的代码，都要检查 k=0 边界。

```python
# WRONG
tail = tcs[-k:] if len(tcs) >= k else tcs[:]

# RIGHT
if k <= 0:
    tail = []
elif len(tcs) >= k:
    tail = tcs[-k:]
else:
    tail = tcs[:]
```

#### L3 — pyc 缓存 mtime 不一致

修改 .py 后 .pyc 有时不重编译，旧 import 行为持续。

**对策**：
- Windows: `Get-ChildItem -Recurse -Filter __pycache__ | Remove-Item -Recurse -Force`
- Linux: `find . -name __pycache__ -type d -exec rm -rf {} +`
- 或 `touch <file>.py` 强制更新 mtime

### 加固措施（going forward）

1. **每次完成阶段性修订**（修 bug、改逻辑、加特征）→ **必须**追加到 IMPLEMENTATION_LOG.md 对应 stage 章节末尾
2. **工具陷阱**（不是项目 bug 本身）→ `docs/TOOL_TRAPS.md`
3. **每次新对话开场**让 AI 助手按顺序读：
   - `docs/IMPLEMENTATION_LOG.md`（项目状态）
   - `docs/TOOL_TRAPS.md`（避免重蹈覆辙）
   - 当前 stage 的 `docs/spec/0?_*.md`

### Stage 2 最终状态确认（2026-05-19 evening）

- ✅ 51/51 unit tests pass（user 待最终本地确认）
- ✅ config.py: 350 行, 所有路径 + 所有锁定常量
- ✅ trigger.py: 339 行, summarize() 完整闭合
- ✅ action.py, special_flags.py 不受影响
- ⏳ 待用户跑：`scripts/mdp/01_generate_decision_points.py` (~5-10 min)
- ⏳ 待用户跑：`scripts/mdp/02_validate_candidates.py` (~30s-2 min)


## Stage 2 数据跑通 + 关键发现（2026-05-19 evening 后续）

Stage 2 代码完成后，跑完 `01_generate_decision_points.py` + `02_validate_candidates.py`
的真实数字，以及关键的领域知识收获。

### 实际跑出的数字（修复后）

第一版（含"0" garbage train_ids）：

| 指标 | 值 |
|---|---|
| n_total | 2,093,120 |
| n_set | 546,418 |
| n_wait | 1,546,702 |
| neg_pos_ratio | 2.83 |
| coverage_pct | **99.646%** ✅ |

加 train_id 过滤后（要求 `^[0-9A-Z]{3,4}$` 且非"0"/"00"等占位符）：

| 指标 | 值 | 评论 |
|---|---|---|
| n_total | **1,999,623** | |
| n_set | 546,418 | 不变 ✓ |
| n_wait | **1,453,205** | 仅减 93k（6%）|
| neg_pos_ratio | **2.66** | 比 v1 sample 高 8× |
| max per-train decisions | **12,727** | 从 91,869 大降，单 train 极端值消失 |
| coverage_pct | **99.646%** ✅ | 不变 |

### 关键领域知识收获 ⭐⭐⭐

#### KH-1: X-prefix signals 是 signal 前置点（已用 04 脚本确认 + 决策）

**用户 2026-05-19 输入**：
> "X063等X-prefix signals 都应该是某个 SIGNAL 的前置，比如 X063 是 5063 的前置。"

**04_diagnose_x_signals.py 跑出来**：

- **8 个 X-prefix signals**，全部在 wait 样本里（**0 个 set**） — 信号员永不在 X063 按按钮
- 总占比 81,040 wait（约 5.5% of all wait）
- 映射规则：`X{nnn} ↔ 5{nnn}`（首字符 X 替换 5）

| X-prefix | Main signal | wait count |
|---|---|---|
| X056 | 5056 | 14,487 |
| X054 | 5054 | 13,910 |
| X484 | 5484 | 12,890 |
| X064 | 5064 | 12,434 |
| X065 | 5065 | 11,967 |
| X061 | 5061 | 6,369 |
| X063 | 5063 | 6,050 |
| X480 | 5480 | 2,933 |

**LOCKED 决策（2026-05-19）：(A) Keep separate**

- X063 和 5063 在 `focal_signal` 列里**保持独立**，不 merge
- 理由：5.5% 占比非主导但非边缘；包含"早期接近"时序信号；spec 02 §17.5 已规定 focal_signal 是 metadata 不进 state，所以 leak 风险 zero
- 候选 mask 自动正确（action.py 从 train.current_tc 做 BFS，不依赖 focal_signal 名字）

**遗留诊断 bug 记录**：

- `04_diagnose_x_signals.py` 的 Q1/Q4 检测 X-prefix 时用 `isinstance(v, (list, tuple))`
  漏判了 numpy.ndarray（routes_clean.end_signals 实际是 numpy 数组）
- 报告 "X-prefix in end_signals: 0" 是误报；实际 trigger.py 用 `for s in sigs:` 直接迭代正确处理 numpy
- **生产代码 OK，diagnostic 报告字段有 false negative**——下次重写 04 时修

#### KH-2: 5063/5053 是站台繁忙信号 — 高 wait 数合法

- 5063、5053 是 Derby 站台上的主信号
- 每天数百次列车进 approach 但未立即按 PR → 大量合法 wait
- 1.45M wait（neg_pos_ratio=2.66）**很可能是合理的真实数字**，不是 bug

#### KH-3: v1 sample 的 0.33 比率不能直接外推

- v1 用的是 5M 行 TD sample（占 11.91M 的 ~42%）+ 较短时间窗
- v2 是 14 个月全量数据
- v2 的 wait 比率高 ≠ v2 有 bug；只是 v1 sample 量小 + Derby 真的有很多 wait

### Spec 02 §2.3 的 trigger 算法**不改**

讨论后决定：**保持当前算法**（每 (T, S) 进 approach 内 30s dedup）。

不加 per-pass dedup 的理由：
- 一次 pass 内同 train 多次进同 signal 的 approach 是合法操作模式
- 缩 wait 数不该靠改 trigger 逻辑，而该靠**训练时 stratified sampling**
- spec 04 §4.4 已经设计了"每 batch 至少 50 trivial + 20 个 non-trivial stratum"

### Stage 2 修订总结表（含本次）

| ID | 文件 | 症状 | 根因 | 修复 |
|---|---|---|---|---|
| BUG-S2-1 | trigger.py | k=0 返回全部 TCs | `tcs[-0:]` 切片陷阱 | `if k <= 0: tail = []` |
| BUG-S2-2 | trigger.py | NameError per_train | Edit 截断丢行 | bash heredoc 重写 |
| BUG-S2-3 | config.py | TD_PARQUET 丢失 | Edit 截断丢行 | Edit 补 + 用户清 pyc |
| **FIX-S2-4** | **trigger.py** | **wait 1.5M 过多** | **TD parse 失败时 trainid="0" 占位符大量触发** | **`valid_id_mask` 过滤非标 IDs** |
| KH-1 | (insight) | X063 出现在 focal_signal | X-prefix = signal 前置点（用户领域知识）| 写 04_diagnose_x_signals.py 进一步分析 |
| KH-2 | (insight) | 5063/5053 wait 极高 | 它们是站台繁忙信号，wait 合法 | 保持当前算法 |

### 已增加的诊断脚本

- `scripts/mdp/03_diagnose_mismatches.py` (167 行) — 候选 mismatch + wait 分布
- `scripts/mdp/04_diagnose_x_signals.py` (165 行) — X-prefix signals 来源 + 处理建议

### Stage 2 最终交付状态（2026-05-19 evening 第二次更新）

| 文件 | 行数 | 状态 |
|---|---|---|
| `src/railrl/mdp/trigger.py` | 352 | ✅ 含 valid_id_mask 过滤 |
| `src/railrl/mdp/action.py` | 344 | ✅ |
| `src/railrl/mdp/special_flags.py` | 212 | ✅ |
| `scripts/mdp/01_generate_decision_points.py` | 135 | ✅ |
| `scripts/mdp/02_validate_candidates.py` | 102 | ✅ |
| `scripts/mdp/03_diagnose_mismatches.py` | 167 | ✅ 新增 |
| `scripts/mdp/04_diagnose_x_signals.py` | 165 | ✅ 新增 |
| `tests/test_mdp/*` | 411 | ✅ 51 cases pass |

### 给 Stage 3 的输入契约

下游（spec 02 §4 snapshot builder）从 `decision_points_v2.parquet` 读：
- ~2M 行（546k set + 1.45M wait）
- 7 列：focal_train, focal_signal, t, label, chosen_route_id, trigger_type, (potentially pass_id)
- focal_signal **包含一些 X-prefix 值**（待 04 脚本确认）
- 在 stratified training 中需平衡 set vs wait

## 更新日志（本文档自身的）

- **2026-05-19** v1.0 — 初版，记录 Stage 0-2 完成状态
- 未来 Stage 完成时在末尾追加新 section（不删旧记录）


---

## Stage 3 — MDP snapshot builder (2026-05-19 → 2026-05-20)

### 完成范围

5 个 module + 3 个 round 的迭代，最终能从 (decision_points + TD + Movements + static_graph) 端到端构造一个通过 leak audit 的 snapshot。

### Round 1 — 基础设施（episode + schema + leak_audit）

| 文件 | 行数 | 内容 |
|---|---|---|
| `src/railrl/mdp/episode.py` | 203 | build_episodes / _assign_pass_by_gap / episode_returns(γ=0.95) / summarize |
| `src/railrl/mdp/leak_audit.py` | 295 | 32 BANNED_STATE_FIELDS + assert_no_leak (7 checks) + collect_violations |
| `src/railrl/mdp/schema.py` | 355 | 45 ALL_COLS = 14 identity + 14 reward + 4 node + 8 edge + 5 other + arrow_schema + validate_row |
| `tests/test_mdp/test_episode.py` | 109 | 9 cases |
| `tests/test_mdp/test_leak_audit.py` | 167 | 22 cases (7 checks parametrized) |
| `tests/test_mdp/test_schema.py` | 82 | 9 cases |

### Round 2 — 中层组件（pass_assignment + state_helpers + state skeleton）

| 文件 | 行数 | 内容 |
|---|---|---|
| `src/railrl/mdp/pass_assignment.py` | 247 | TRUST id matching + fallback gap-based + summarize |
| `src/railrl/mdp/state_helpers.py` | 239 | TrainStateLookup (current_tc/recent_tcs/berth) + SubgraphExtractor (3-hop BFS + filter_edges) |
| `src/railrl/mdp/state.py` (v1) | 471 | SnapshotBuilder skeleton with placeholders |

测试结果：**51 + 41 = 92 cases pass** (Stage 2 + 3 R2)。

### Round 3 — 完整实现（per-window aggregates + K=256 + schedule_outlook + dynamic edges + multi-train）

| 文件 | 行数 | 新增 |
|---|---|---|
| `src/railrl/mdp/state_history.py` | 521 | _StateTimeline + TrackOccupancyHistory + SignalAspectHistory + BerthHistory + MovementsLookup + EventTokenStream |
| `src/railrl/mdp/state.py` (R3) | 724 | 5 个 history wired in，所有 Round 2 placeholder 替换为真实计算 |
| `scripts/mdp/05_build_snapshots.py` | 200 | 全语料 driver (`--limit N`, `--dev`, `--audit-every`) |
| `scripts/mdp/06_run_leak_audit_full.py` | 156 | 全语料 leak audit (`--sample N`, `--first-fail`) |
| `tests/test_mdp/test_state_history.py` | 282 | 15 cases |

**最终测试结果：~108 cases pass**。

### Round 3 关键设计决策

- **时间窗口聚合算法**：每 asset 用 (time_ns, state) 数组 + `np.searchsorted` 二分查找。窗口 W 内：`start_state = state_at(t_start)`, 逐 transition 累加 state=1 的 dwell，最后 / window_ns。O(log n + transitions_in_window) per query。
- **K=256 event token**：按全局 time descending 排序，取 top-256，每个 token = (asset_idx in subgraph, state, time_delta_s)。Asset_idx 是 subgraph 内 [track_keys + signal_keys] 的位置，下游 encoder 用它索引 node embedding。
- **schedule_outlook 防泄露**：MovementsLookup 只用 gbtt_timestamp，actual_timestamp 永不出现。`planned_platform` 强制 int 1-6 或 None（绝不返回 signal_id 字符串）。
- **Dynamic edges**：`at_berth(train→track)` 由 train_lookup.current_tc 推出；`next_signal(train→signal)` 由 BerthHistory 反查得到（哪个 signal 的 berth 当前由这个 train 占用）。
- **Multi-train state_nodes_train**：从 subgraph 内 tracks 的当前 occupier 收集（去重，排除 focal，按字母序排，cap = MAX_TRAINS_PADDED - 1 = 7）。

### Round 3 修订与陷阱

| ID | 文件 | 症状 | 根因 | 修复 |
|---|---|---|---|---|
| BUG-S3-1 | state_history.py | `planned_platform=3.0` 而非 3 | pandas Series mixed int/None 自动 coerce 到 float64 | `_parse_plat` 后再在存储点强制 `int(p)` |
| BUG-S3-2 | state.py / state_history.py | 多次被 Edit/heredoc 截断（行尾 + 中部）| virtiofs 同步问题；touch 居然也会截断 | 用 `head -N` + `cat >> EOF` 重建，每次后立即 `ast.parse` 验证 |
| BUG-S3-3 | state.py | `state_center` 在 snapshot dict 但 leak audit 读 `center` | 命名不一致 | snapshot 同时存 `state_center` 和 `center`（向后兼容）|
| BUG-S3-4 | state.py | 同份代码出现两次（line 712 vs line 725 重复）| 用户/linter 修了一次同时 heredoc 又追加了一次 | `head -724` 截断重写 |
| KH-3 | (insight) | sandbox pyc 缓存常常导致测试看上去没生效 | virtiofs 上 pyc 的 mtime 跟 py 同步，Python 优先 pyc | 写完 .py 后用 `open(p).write(open(p).read())` 强制更新 mtime |

### Stage 3 端到端验证（合成数据）

输入：2 个 track + 2 个 signal + 1 个 route + 6 条 TD events + 2 行 Movements + 1 个 decision row。

输出（关键字段）：
- `center: {'type':'track', 'id':'TFBN'}` ✅
- `audit_passed: True` ✅
- Per-window aggregates 数值正确（TFBN frac_5m=0.10，TFPJ frac_5m=0.25，signal 5040 red_5m=0.267）
- `planned_platform=3`（int 非 float）✅
- Dynamic edges: 2 at_berth + 1 next_signal ✅
- Multi-train: 1S49 (focal=True) + 2A28 (focal=False) ✅
- Schedule outlook 排除 focal_train ✅

### 给 Stage 4 的输入契约

下游（model + training）从 `snapshots_v2.parquet` 读取一行约 45 列：
- **identity (14)**: sample_id, focal_train, focal_signal, t, pass_id, episode_idx, position_in_episode, is_last_in_episode, label, chosen_route_id, chosen_action_idx, candidate_route_ids, n_candidates, trigger_type
- **reward (14)**: outcome, approach_distance, delay_change_seconds, next_tc_headway_seconds, gate, r_*_raw, r_*, r_total（注意：snapshot 阶段填 NaN，由 decision_rewards.parquet 在训练前 join）
- **state (4 node lists)**: state_nodes_track / signal / route / train，每个是 list of struct
- **state edges (8)**: 6 static + 2 dynamic (at_berth, next_signal)
- **state aux**: state_event_tokens (list of {asset_idx,state,time_delta_s}), state_schedule_outlook, state_special_flags, state_special_flags_meta, state_center

Stage 4 loader 需要把这些 list-of-struct 转成 padded tensors + PyG HeteroData。

### Stage 3 给后续 Stage 的建议

1. **Edit tool 截断仍未根除** — 任何对 `state.py` / `state_history.py` 的修改后必须 `ast.parse + wc -l + tail` 三件套检查。
2. **pyc 缓存陷阱** — virtiofs 下 .py 修改后 mtime 不变。补救：`open(p,'w').write(open(p).read())` 强制 bump。
3. **数值类型严格检查** — leak_audit Check 4 要求 planned_platform 是 int。pandas-cast 容易把 int 升级为 float。
4. **dev/prod audit 比例** — 05 driver 默认每 1000 行 audit 一次；06 driver 做全量。如果性能允许，05 可改成全量 audit + 离线汇总。

---

## Stage 3 Hotfixes（2026-05-20，全量跑 05 时发现）

### Hotfix-1：OOM（rows 列表堆 2M 行）

**症状**：`05_build_snapshots.py` 跑到 ~400k/2M 时 `MemoryError`。
**根因**：把所有 snapshot dict 累积到一个 Python list 再一次性 `to_parquet`，2M 行 × ~5KB ≈ 20GB。
**修复**：改成 `pyarrow.parquet.ParquetWriter` 流式写。每 `--batch-size`（默认 5000）行 flush 一个 row group 并 `batch.clear()`。内存稳定在 ~25MB。加了 `try/finally` 保证 writer.close()，schema 不一致时 `table.cast(safe=False)` 兜底。

### Hotfix-2：Movements headcode 提取 + platform 1-7（领域知识）

**症状**：05 报 `[warn] Movements not found — schedule_outlook will be empty`。即便加载了，`current_train_id` 列也 **99.88% 为空**。
**根因 + 领域知识**：
- Movements 真正可用的列是 TRUST `train_id`（10 字符，如 `851S49ME28`），headcode 嵌在 `[2:6]`（→ `1S49`）。99.7% 的行匹配 `NXNN` 格式。
- **platform 7 是 Derby 的 pilot line**（用户领域知识 2026-05-20）：北行从 **EC5487** 发车、南行从 **EC5484** 发车（两信号机方向相反），它们的 TC 分别是 **TECV** 和 **TECS**。所以 platform 7 是合法站台，不是噪声。
**修复**：
- `MovementsLookup.build(train_id_col="auto")`：auto 时优先用 `train_id` 切 `[2:6]` 提 headcode，否则回退 `current_train_id`。
- platform 范围 `1-6 → 1-7`：新增 `config.MIN_PLATFORM_ID=1 / MAX_PLATFORM_ID=7`；`leak_audit.py` Check 4 + `state_history.py` MovementsLookup 都从 config 读这两个常量（defensive import，缺省回退 1/7）。
- `05_build_snapshots.py` 改用 `data_io.load_movements()`（自动从 `Movements.csv` 缓存 parquet）。
**验证**（真实 Movements 前 10 万行）：提取出 1280 个 headcode，TD 已知 headcode 全部命中（1S49→plat 1, 1M99→4, 1K69→3, 2A28→4, 2A31→6），platform 分布 `{1:4542, 2:4786, 3:10297, 4:10128, 5:9002, 6:10753, 7:1494}`，platform 7 保留，全部是纯 int。

### Platform 7 物理拓扑（领域知识存档）

| 方向 | 发车信号机 | TC |
|---|---|---|
| 北行 (north) | EC5487 | TECV |
| 南行 (south) | EC5484 | TECS |

（两信号机方向相反；platform 7 = pilot line。后续若需在 static graph 里特别处理 platform 7 的 route/signal 关联，参考此表。）

### TRUST train_id 结构（Table 3.6，ESWA paper §3 — 权威定义存档）

10 字符 TRUST train_id = `[AA][BBBB][C][D][EE]`：

| Part | Component | 长度 | 含义 |
|---|---|---|---|
| `[AA]` | Stanox Prefix | 2 | 列车始发区域 |
| `[BBBB]` | **Headcode** | 4 | **信令 ID（headcode）** ← 与 TD focal_train 匹配的就是这段 |
| `[C]` | TSPEED | 1 | 列车状态码 |
| `[D]` | Call Code | 1 | 基于始发出发时间的字母/数字 |
| `[EE]` | Day Indicator | 2 | 列车始发当月的日期 |

例：`851S49ME28` → AA=`85`, BBBB=`1S49`, C=`M`, D=`E`, EE=`28`。

**这正式确认了 `train_id[2:6]` = BBBB headcode 的提取是正确的**（`MovementsLookup.build` 用的就是 `slice(2,6)`）。
**重要推论**：headcode 在不同日期 / 不同 Call Code 下会复用（同一个 `1S49` 可能对应多趟车）。`MovementsLookup` 的 schedule_outlook 按时间窗过滤，`planned_platform` 取时间上最近的条目，所以 headcode 复用问题被时间窗口自然处理。但**如果 Stage 4+ 要做精确的 TRUST↔TD pass 对齐，必须用完整 train_id（含 Day Indicator EE）而不是裸 headcode**（这也是 pass_assignment.py 的设计前提）。

### Movements 缓存脚本

新增 `scripts/data/06_cache_movements.py`：一次性把 Movements.csv → movements.parquet（zstd），并打印 headcode/platform/日期诊断。用户 4TB 盘，永久缓存避免每次 re-parse。`load_movements()` 也会在首次调用时自动缓存。

### 环境提示：src-layout 的 import

`railrl` 是 src-layout（包在 `src/railrl/`）。`pyproject.toml` 的 `[tool.pytest.ini_options] pythonpath=["src"]` 让 pytest 自动加 `src/` 到路径，但**裸 `python -c "import railrl"` 不会**，会报 `ModuleNotFoundError`。
解决：`pip install -e . --no-deps`（注册包，不拉重依赖）；或临时 `$env:PYTHONPATH="src"`（PowerShell）。所有 scripts/ 下脚本自己 `sys.path.insert(0, .../src)`，所以直接 `python scripts/...` 不受影响。

### ⚠️ 重要：本次 hotfix 改动了 leak_audit 的契约

`leak_audit.py` Check 4 的 platform 上限从 6 → 7。意味着**旧的 test_leak_audit 里如果有断言 platform 6 是上界、7 应该失败的用例需要更新**。如果 pytest 报 Check 4 相关失败，检查 `tests/test_mdp/test_leak_audit.py::TestCheck4ScheduleOutlook` 是否有 `planned_platform=7` 应当通过的新语义。（目前的测试用例只测了 99 越界，仍然失败，所以应该不受影响。）

### Hotfix-3：pass_assignments.parquet 缺失 + episode fallback 修正

**症状**：05 在 `[4/6] building episode metadata` 报 `pass_assignments.parquet not found — falling back to gap-based pass_id`，得到 82,858 episodes（gap-based 质量较低）。
**根因**：Stage 3.4 写了 `pass_assignment.py` 模块但**从没写驱动脚本**生成 `pass_assignments.parquet`。

**修复**：
1. 新增 `pass_assignment.build_pass_intervals(movements_source)`：**快路径**，一行 per TRUST train_id（向量化 groupby min/max actual_timestamp，±30min buffer）。这正是 `episode.py::_join_pass_assignments` 实际需要的（它按 trainid_filled 分组，把每个 decision 的 t 落到包含它的 interval）。绕开了原 `build_pass_assignments` 对每条 TD 事件 iterrows 的慢路径。输出列：`trainid_filled, pass_id, pass_t_first_ns, pass_t_last_ns, pass_source`。
2. 新增驱动 `scripts/mdp/00_build_pass_assignments.py`：读 movements.parquet（或 csv）→ build_pass_intervals → 写 `outputs/passes/pass_assignments.parquet` + summary。
3. **修正 `episode.py::_join_pass_assignments` 的 fallback bug**：原来未匹配 TRUST interval 的 decision 全部塌缩成 `FB:{tid}:0`（一个 train 的所有未匹配 decision 跨 14 个月合并成一个 episode，γ-discount 会算错）。改成对未匹配集合做 **gap-based 聚类**（>PASS_FALLBACK_GAP_S=6h 拆分），每个 fallback episode 时间局部。同时把 matching 从 `df.iterrows()` 换成 `to_numpy()` 索引（200k 行 0.2s，全量 2M ~2s，原来要分钟级）。

**真实数据验证**（200k decision points + 全量 Movements）：
- TRUST 匹配率 **93.7%**，fallback 6.3%
- fallback episode 大小：mean=2.8, max=24（不再是 1 个跨月巨型 episode ✓）
- headcode 缺失 1.1% / 时间不在任何 interval 5.2%
- join 耗时：200k → 0.2s（全量 2M 估计 ~2s）

**新增测试**：`tests/test_mdp/test_episode.py::TestJoinPassAssignments`（3 cases：interval 内匹配 TRUST、未匹配 gap-cluster 不塌缩、混合）。

### Pass interval 设计要点

- 用 **actual_timestamp**（非 gbtt）定义 pass 时间范围。pass_id/episode 是 IDENTITY metadata（离线 episode 边界），不是 state feature，所以用 actual 时间不违反 spec 01 §17.5 leak 契约（leak audit 只扫 state_* 字段）。
- merge_asof **不能**用于此匹配：buffered intervals 会嵌套（短 pass 落在长 pass 内），merge_asof 只看「最近开始」的一个 interval 会漏（实测掉到 54%）。必须扫所有 interval 查 containment（iterrows/to_numpy 循环，93.7%）。
- 运行顺序：先 `00_build_pass_assignments.py`（一次性）→ 再 `05_build_snapshots.py`（自动 join TRUST episodes）。

### Hotfix-4：流式 parquet schema 推断崩溃（ArrowNotImplemented int64→null）

**症状**：05 流式写到某批 `_flush` 报 `pyarrow.lib.ArrowNotImplementedError: Unsupported cast from int64 to null using function cast_null`。
**根因**：原 `_flush` 用 `pa.Table.from_pandas(df_batch)` **逐批推断** schema。第一批 5000 行里某个 nullable 嵌套字段（如 `state_nodes_train[].planned_platform`）全是 None → pyarrow 推断成 `null` 类型并写进 writer schema；后面某批该字段出现 int → `cast int64 → null` 崩溃。
**修复**：用 **固定显式 schema**（`schema.get_arrow_schema()`）建 ParquetWriter，每批用 `pa.Table.from_pylist(batch, schema=FIXED)` 写。`from_pylist(schema=...)` 三大好处：(1) 不做逐批推断（消除 null-type 问题）；(2) 自动忽略 dict 里多余的 key（如冗余的 `center`）；(3) 缺失的 key 填 null。
**连带修复 schedule_outlook 字段对齐**：schema 的 `outlook_struct` 期望 `{train_id, headcode_class, eta_s, planned_platform}`，但 `MovementsLookup.schedule_outlook` 产出 `{train_id, gbtt_delta_s, planned_platform, event_type}`。不对齐的话 from_pylist 会把 eta/headcode_class 填成 null（丢失 ETA 信息）。修法：
  - schema `outlook_struct` 加 `event_type` 字段（5 字段）。
  - `state._build_schedule_outlook` 转换成 `{train_id, headcode_class, eta_s(=gbtt_delta_s), planned_platform, event_type}`。
**验证**（独立脚本，完整 45-field schema + 嵌套 struct + 混合 null/int 批次）：45 列全部正确，batch-2 的 int planned_platform 存活，schedule_outlook 5 字段正确，多余 `center` key 被忽略。

### ⚠️ snapshot dict ↔ schema 对齐契约（Stage 4 reader 必读）

`state.build_snapshot` 产出的 dict 必须与 `schema.get_arrow_schema()` 字段名**逐一对齐**（嵌套 struct 内字段名也是）。已核对全部对齐：identity 14 + reward 14 + 4 node lists + 8 edges + event_tokens + schedule_outlook(5字段) + flags(8) + flags_meta(2) + center = 45 顶层字段。改任何一边都要同步另一边，否则 from_pylist 静默填 null 丢数据。`center` key 是例外（dict 有、schema 无，被 from_pylist 忽略；leak audit 读它）。

---

## 第一次全量 6.58h 跑后的数据审计 + 大返工（2026-05-20）

### 全量跑结果（已废弃，需重跑）

1,999,611 / 1,999,623 built（99.999%），12 skipped，0 audit fails，**23,703s = 6.58h @ 84.9/s**。

**好消息**：schedule_outlook 97.3% 有数据（Movements/headcode/platform-1-7 都对），TRUST episode 100% 命中，流式写稳定。

**两个致命问题（需重建 snapshots）**：
1. **动作空间为空**：`decision_points_v2.parquet` 只有 6 列，`candidate_route_ids` 在 Stage 2 coverage check 算过但**从没持久化**。所有 snapshot 的 `n_candidates=0`、`in_candidate_set=False`、无 `chosen_action_idx` → 结构化动作 `{wait}∪{(train,R)}` 没有路线可选，RL 根本没法训。
2. **77.7% 退化子图**：`current_tc` 解析到不在 249-track 路网里的 approach/holding track（`T938`/`TFPW`/`TYWH`），3-hop 子图只剩 1 个孤立 track（0 signal、0 route）。模型对 3/4 的决策几乎看不到任何状态。

（另外发现：`nodes_route.parquet` 没有 `track_sections`，所以 `on_focal_train_path` flag 一直是空的——返工一并修了。）

### 返工 T1-T4（全部完成，已在真实数据/独立脚本验证）

**T1 — `scripts/mdp/01b_enrich_candidates.py`**：对 2M decision 算 `candidates = routes_from(focal_signal)`（spec 02 §3.2 Rule 1，主规则），写回 decision_points。验证：**0.8s/2M**，candidate-set mean=2.72/median=2/max=13，12% wait-only；**99.99% coverage**（chosen 在 routes_from 里），只有 34 条（0.0062%）需 append chosen。动作索引约定锁定：**action 0=wait, 1..K=candidate_route_ids[0..K-1], chosen_action_idx=0(wait) 或 1+idx(set)**。

**T2 — 子图候选种子化**：`SubgraphExtractor.extract(center_tc, seed_routes=...)`，从 current_tc + 候选路线一起 BFS（候选路线扩 `seed_route_hops=2` 层）。中心仍是 current_tc（leak audit Check 1 不变）。验证：退化中心 T938 从 track=1/signal=0/route=0 变成 track=11/signal=3/route=9；全 build_snapshot 端到端测试通过（off-network 中心 + audit pass + on_focal_train_path/in_candidate_set 都对）。新增 `state.route_tracks` map（route_id→ordered track_sections，来自 routes_clean，因 nodes_route 没有）。

**T3 — 向量化 `window_stats`**：`_StateTimeline` 在 `__post_init__` 预算 prefix-sum（occupied-time `_cum_occ` + change-count `_cum_chg`），每次窗口查询 O(log n) 而非 O(window 内事件数)。验证：2000 个随机查询与旧循环**数值完全一致（0 mismatch）**，118k calls/s（与窗口大小无关）。

**T4 — 并行 + sharding**：
- `05_build_snapshots.py` 加 `--shard K --nshards N`：在 FULL dp 上建 episode（保证 episode_idx 跨 shard 一致）+ 赋全局 `sample_id`，再 strided 切片 `dp.iloc[K::N]`，写 `.partK.parquet`。
- `scripts/mdp/05b_build_snapshots_parallel.py`：subprocess 起 N 个独立进程（Windows-spawn 安全，无 pickling），完成后流式合并 part 文件 + 汇总 summary/skipped。
- 验证：strided 切分是完美分区（每行恰好一次，无重叠/缺失）；wrapper AST-OK。每 worker 独立加载 TD+histories，峰值内存 ≈ N×(TD+histories)，`--workers` 建议 4-8。

### 重跑顺序（用户在 Windows 执行）

```
python scripts/mdp/01b_enrich_candidates.py            # 1) 动作空间写回（~秒级）
python scripts/mdp/05b_build_snapshots_parallel.py --workers 6   # 2) 并行重建 snapshots
```
（先用 `--workers 6 --limit 40000` 跑 smoke 验证，再全量。）

### 待办：snapshots 重建后重新审计

重跑后应看到：n_candidates mean≈2.7、退化子图 ≪77.7%（应降到接近 12% 的纯 wait-only signal）、train_nodes 多于 1 的比例上升、schedule_outlook 仍 97%+。然后跑 `06_run_leak_audit_full.py --sample 10000` 确认 pct_passed=100。


---

## 性能返工（2026-05-20，profile 驱动）

第一次返工（T1-T4 候选+种子化）修对了数据，但**子图变丰富后每个 snapshot 慢了 ~12x**。profile_build.py（cProfile，真实数据）定位：
- **startup 151.8s**：SnapshotBuilder.build_default 从 11.7M TD 事件建 5 个 history（一次性，每 worker 付一次）。用户误读成"300 个要 151s"，其实 300 snaps 只花 7.8s。
- **每 snapshot 26ms (38.5/s)**，三大热点：
  1. `_format_edges` iterrows → 每 snapshot 创建 ~187 个 pandas Series（8.7ms）
  2. `MovementsLookup.schedule_outlook` 每次重建 247k 元素 list（line 326，~4ms）
  3. `slice_last_k` 最终 256-tuple listcomp（5ms）

**已修（独立脚本验证，0 mismatch）**：
- **filter_edges/_format_edges**：StaticGraphView 边预计算成 (src,dst,order) tuples（`_ensure_edge_tuples`），filter 用 set 成员判断，`_format_edges` 只 wrap。**55x faster**（3.5ms→0.064ms），返回类型 DataFrame→list[tuple]（仅 build_snapshot 调用，安全）。
- **schedule_outlook**：`MovementsLookup.__post_init__` 预计算 `_all_times` numpy 数组，用 np.searchsorted。**1454x faster**（3.5ms→0.002ms）。
- **node caps**：`SubgraphExtractor` 加 cap_track=60/cap_signal=15/cap_route=15（= padding caps，loader 本就截断）。候选路线先种子化所以保留。TFMP 72/29/118→46/15/15，route 工作 8x 少。
- **slice_last_k**：numpy 化（per-element int() 循环 → 切片+argpartition），9.4→~5ms。

预期：26ms→~12ms/snapshot ≈ 80/s 单进程；6 workers 全量 ~70min build + 一次性 151s setup。

**待确认**：用户重跑 `profile_build.py --n 300` 验证新 rate（应 ~80/s），再全量。若 startup 151s 仍痛，可加 history 磁盘缓存（build 一次，workers load）。

### profile_build.py 关键认知（存档）
- 151s 是 history 构建（11.7M 事件 → per-asset 查找表），**一次性**，不是 per-snapshot。
- 子图 node 数直接决定 per-snapshot 成本（window_stats/event tokens/edges 都 scale）。padding caps 必须在 extract 时就 enforce，否则白做被截断的 node 的活。

---

## Stage 3 完成 ✅（2026-05-20）— snapshots_v2.parquet 全量构建 + 审计通过

性能修复后全量重跑：**1,999,611 snapshots / 12 skipped / 0 audit fails / 1992s (33 分钟) / 6 workers**。文件 573 MB（比旧 165MB 大，因为子图现在真正填充了，不再 77.7% 退化）。

### 全量审计结果（采样 253k）

| 指标 | 返工前（坏） | 现在 |
|---|---|---|
| 退化子图（1 track） | 77.7% | **8.8%** |
| n_candidates（动作空间） | 0（空！） | mean **2.70**, max 13, invalid **0** |
| schedule_outlook 有数据 | — | **94.2%**（7.1% 是 7 天 smoke 切片假象） |
| 多车 snapshot | 9.9% | **31.4%** |
| padding caps (60/15/15/8) | 无界(max 98/118) | **全部遵守** |
| label | — | 27% set / 73% wait（与 Stage 2 一致） |
| leak 审计 | — | build 内 2000/2000 + 独立 103k 抽查 **全 PASS** |

leak 独立抽查（103k snapshots）：center 永远是 track、每个 snapshot 恰好 1 个 is_focal train、planned_platform 永远 1-7 或 None、state 里 0 个 banned field。

### 这次彻底解决的问题（给后续 stage 的教训）

1. **动作空间必须持久化**：decision_points 原来只有 6 列，candidate_route_ids 算过没存。01b_enrich_candidates.py 补上（routes_from(focal_signal)，99.99% 覆盖）。
2. **子图必须候选种子化**：current_tc 77.7% 落在路网外的 approach track（T938/TFPW/TYWH）→ 用候选路线 seed BFS。
3. **padding caps 要在 extract 时 enforce**：否则白建被截断的 node（route 118→15，8x 浪费）。
4. **per-snapshot 严禁 pandas iterrows / 重建大 list**：filter_edges 预计算 tuple（55x）、schedule_outlook 预计算 times 数组（1454x）。
5. **子进程 stdout 要 `-u`**：否则 block-buffering 让日志看起来空的、像卡住。

### Stage 3 最终交付文件清单

- `src/railrl/mdp/`: trigger, action, special_flags, episode, leak_audit, schema, pass_assignment, state_helpers, state, state_history（10 模块）
- `scripts/mdp/`: 00_build_pass_assignments, 01_generate_decision_points, 01b_enrich_candidates, 02_validate_candidates, 03/04_diagnose, 05_build_snapshots, 05b_build_snapshots_parallel, 06_run_leak_audit_full, profile_build
- `scripts/data/06_cache_movements.py`
- 输出（不进 git）：`outputs/snapshots/snapshots_v2.parquet` (573MB, 2M 行)

### → 下一步 Stage 4：HGT + Transformer + Q-net + CQL（spec 03 + 04）

snapshots_v2.parquet 是 Stage 4 data loader 的输入。45 列 schema 见 schema.py / 上文"对齐契约"。

---

## ⚠️ Stage 3 数据验证发现严重 bug（2026-05-20）—— us/ns 单位不匹配，需重建

**进 Stage 4 前的数据验证**（用户要求"确定数据没问题"）抓到一个会让整个时间维度失效的 bug。**上一节的"Stage 3 完成"作废，需用修复后的代码重跑。**

### bug：td_data.parquet time 是 datetime64[us]，代码当 ns 用

详见 TOOL_TRAPS §12。一句话：history builder 用 `sub["time"].astype("int64")` 得到**微秒**，但 `t_ns` 是**纳秒**，差 1000 倍 → 所有 TD 时间查询永远返回"最后一次事件"（未来泄露 + 全错）。

**审计如何发现的**：检查 event token `time_delta_s` 的**数值分布**（不只看非空），发现 100% 是 ~1.69e9（明显是 UNIX 时间戳不是 delta）。

**影响**（全错，需重建）：current_tc（子图中心）、occupied_now、current_occupier、occupancy_fraction_*、n_state_changes_*、aspect_*、last_change_age_s、berth、recent_panel_requests、event tokens。
**不受影响**：candidates、Derby_info、静态属性、episodes/pass_id、schedule_outlook（都是 ns，已验证 decision t / movements actual+gbtt 都是 datetime64[ns]）。

**修复**：`state_history._to_ns_int64()` 强制转 ns，用在 5 个 TD-time 转换点 + state_helpers TrainStateLookup 1 处。standalone 用 us-dtype 合成数据验证：event delta 从 1.69e9 → 正确的 30/60/90 秒。

**重建**：用户重跑 `python scripts/mdp/05b_build_snapshots_parallel.py --workers 6`（~33 min）。decision_points / pass_assignments / movements 都不用重跑（没受影响）。

---

## 本轮大返工 session 的教训总结（2026-05-20，用户要求记录）

从"第一次全量 6.58h 跑完"到"发现数据全错需返工"再到"性能 12x 慢"再到"us/ns bug"，一连串问题。核心教训：

### 1. 数据 pipeline 的"完整性"必须端到端验证，不能只验单元
- **动作空间整段缺失**（candidate_route_ids 没持久化）跑完 6.58h 才发现 —— 因为单元测试都过、脚本都"成功"，但没人检查"snapshot 里 n_candidates 是不是真的有值"。
- **77.7% 退化子图**同理 —— 各组件都对，但组合起来 current_tc 落在路网外没人验。
- **us/ns bug** —— 所有合成测试都过（合成数据是 ns），真实数据（us）才暴露。
- **教训**：每个 stage 产物都要**用真实数据抽样、看特征数值分布**（不只是"非空/形状对"）。建议 Stage 4 起，每个产物配一个"数值合理性审计"脚本。

### 2. 合成测试必须复刻真实数据的 dtype/分布
- 合成 TD 用 `pd.to_datetime` 给 ns，真实是 us → 盲区。
- **教训**：合成 fixture 要么从真实数据切一小片，要么显式匹配 dtype（含 datetime resolution、object vs category、nullable int 等）。

### 3. 性能要 profile 驱动，不要猜
- 我前期反复猜瓶颈（filter_edges？window_stats？）浪费了时间。`profile_build.py`（cProfile）一次就定位了真凶（_format_edges iterrows 8.7ms + schedule_outlook 重建 list 4ms）。
- **教训**：性能问题先写 profiler 拿真实 breakdown，再动手。

### 4. per-snapshot 代码严禁 pandas 慢操作
- `iterrows`（每行造 Series）、每次重建大 list、`.astype(str).isin` 在全表上 —— 这些在 per-snapshot 热路径上是致命的。
- 修复后：filter_edges 用预计算 tuple + set（55x），schedule_outlook 预计算 numpy times（1454x）。
- **教训**：热路径用 numpy/dict/预计算，杜绝 per-call pandas。

### 5. padding caps 要在生成时 enforce，别白做被截断的活
- 子图建 118 个 route 但只留 15 个 → 8x 浪费。caps 提前到 extract 的 BFS 里。

### 6. 沙盒环境的硬约束（贯穿全程，拖慢验证）
- `/sessions` 磁盘满 → bash 看到的是**冻结/截断的旧文件缓存**，无法 import 最新代码、无法 AST 校验。
- 对策：用 Read 工具（=Windows 真实文件）逐段确认；逻辑抽出来在 /tmp 独立验证（不 import 挂载包）；真实端验证靠用户跑。
- 子进程 stdout 要 `-u`，否则日志缓冲看着像卡住。

### 7. 重大数据产物跑之前，先小样本 smoke（--limit）确认不崩 + 抽查质量
- 全量跑动辄 30min-数小时，崩在最后/数据错都很贵。


---

## Stage 3 真正完成 ✅ + Stage 4.2 完成（2026-05-20，us/ns 修复后重建 + 验证）

### Stage 3 重建结果（修复 us/ns 后）

`05b_build_snapshots_parallel.py --workers 6` 重跑：**1,996,572 snapshots / 3,051 skipped / 0 audit fails / 6157s (~103 min)**。
- skipped 从 12 → 3,051：因为 current_tc 现在**正确**（决策时刻的位置，不再是"最后一次事件"），无 TD 前置轨迹的决策被正确跳过（0.15%）。
- 时间 33min → 103min：时间特征现在真正被计算（之前 us/ns 错配让 window_stats 提前 return 几乎不干活）。

### 审计 READY ✅（`scripts/mdp/07_audit_snapshots.py`，采样 252k）

| 项 | 值 | 判定 |
|---|---|---|
| event time_delta_s | min 0 / med 2482 / max 3.69M, %>1e8 garbage=**0.000%**（之前 100%）| ✅ us/ns 修好 |
| occupancy_fraction_5m | 29% 在 (0,1), max 1.0, 70% 空 | ✅ window stats 真在算 |
| aspect_fraction_red_5m | %>0=23.3% | ✅ |
| occupied_now | 8.8% track 占用 | ✅ 物理合理 |
| 退化子图 | **0.0%**（之前 8.8%）| ✅ current_tc 正确→都在路网 |
| caps | track≤60/signal≤15/route≤15/train≤8 全遵守 | ✅ |
| n_candidates | mean 2.70, invalid 0 | ✅ |
| schedule_outlook | 94.3% 非空 | ✅ |
| leak | center=track / 1 focal / platform 1-7 / 0 banned | ✅ PASS |

**VERDICT: READY FOR STAGE 4。** snapshots_v2.parquet（~2M 行）是 Stage 4 的输入。

### 新增审计脚本（教训落地）
`scripts/mdp/07_audit_snapshots.py` —— 检查特征**数值分布**（不只形状/非空）。这是"每个产物配数值合理性审计"教训的落地，今后每个 stage 产物都该有。

### Stage 4.2 完成 — normalization stats + vocab

`scripts/train/01_build_normalization_stats.py` → `outputs/snapshots/normalization_stats.json`。
- split（按 pass_id 哈希，episode 不跨 split）：train 1,419,965 / val 286,991 / test 289,616。
- 39 个连续特征 z-score（**仅 train split** 统计，无泄露）。
- vocab（learned embedding 索引，0=pad）：track_id=268, signal_id=123, route_id=278, train_id=2184 + 类别字段（prefix/cls/headcode_class...）。
- ⚠️ **embedding 尺寸用 stats 里的 vocab size，不要用 spec 03 §3.1 的硬编码**（track_id 实测 268 > spec 写的 250）。

### Stage 4 依赖提醒
- 需 `pip install torch torch-geometric`（之前 `--no-deps` 没装）。
- **沙盒已无法读 573MB+ 的 snapshots 文件**（/sessions 满 → virtiofs 服务旧/坏视图）。Stage 4 碰数据的代码靠 Windows 端验证；模型逻辑可用小合成 tensor 在沙盒单测。

### → 下一步 Stage 4.1：PyG HeteroData loader
读 normalization_stats.json + snapshots_v2.parquet → 每行一个 padded HeteroData（4 节点类型 + 8 边）+ K=256 event token tensor + 动作集（wait + candidates）+ chosen_action_idx 标签。

---

## Stage 4.1 — PyG HeteroData loader（2026-05-20，core 已验证，torch 包装待 Windows 测）

`src/railrl/encoders/input_pipeline.py`。设计：**numpy core 与 torch 分离**——
- `encode_snapshot(row, stats)`（纯 numpy，沙盒可测）：一行 snapshot → 4 节点类型的 (cont z-score / binary / cat vocab idx / ident idx) + 8 边的本地索引对 + K=256 event tokens + schedule outlook + 动作集（候选路线→route 节点本地索引，padding 到 14）+ chosen_action_idx 标签。
- `to_heterodata(enc)` / `SnapshotDataset`（lazy import torch + PyG）：包成 PyG HeteroData；按 pass_id 哈希 split 过滤（与 normalization 一致，无泄露）。

**特征布局决策**（spec 03 §3.1 没完全定死，本模块为准）：
- 连续特征 z-score 裁剪 ±5；binary 0/1；categorical → vocab 索引（0=pad，编码器再 embed）；identity（track/signal/route/train_id）→ vocab 索引。
- nullable platform（platform_id/end_platform_id/current_platform/planned_platform ∈ {1..7,None}）→ 固定 8-way one-hot（idx0=None），不进 vocab，拼到 binary 块。
- ⚠️ 编码器 embedding 尺寸用 normalization_stats 的 vocab size（track_id 268 等），不要用 spec 硬编码。

**沙盒验证（numpy core，合成 snapshot）**：节点特征 shape/zscore（occ_frac_5m→1.0）、platform-7 one-hot、边本地索引重映射（connects/traverses/at_berth/next_signal）、event token（state {0,1}→{1,2}+log1p）、候选→route 索引 [0,1,-1,-1]、chosen_action_idx、outlook eta log1p + platform onehot —— **全部正确**。

**待 Windows 测**：`python scripts/train/02_smoke_loader.py`（需 torch+torch_geometric）—— 验证 to_heterodata + SnapshotDataset 产出可批处理的 HeteroData。

**下一步**：4.3 HGT encoder（PyG HGTConv，节点 init embedding 用 stats vocab size）+ 4.4 Transformer event encoder + 4.5 fusion/Q-head。

---

## Stage 4.3 — HGT graph encoder（2026-05-20，config 已验证，torch forward 待 Windows 测）

`src/railrl/encoders/hgt.py`。
- `node_init_config(stats)`（torch-free，沙盒已验证）：per-type 维度。track in_dim=94（64 ident+8 platform_sub+12 cont+10 binary），signal=103，route=125（5 个 cat 字段），train=60。**embedding 尺寸全用 stats vocab size**（track_id 269 等）。
- `NodeInit`：per-type [identity_emb(64/32) ⊕ cat_embs(8 each) ⊕ cont ⊕ binary] → Linear+GELU+LayerNorm → d_model=128。
- `HGTEncoder`：NodeInit + 3× PyG HGTConv(heads=4, 8 edge types) + 残差/LayerNorm/dropout；输出 per-node h_dict + per-type & global mean pool（scatter by batch）。HGTConv 对无入边的类型可能 drop → 用 `out.get(nt, x_dict[nt])` 保留。
- edge metadata（PyG (src,rel,dst)）与 loader 的 HeteroData edge key 一致。

**待 Windows 测**：`python scripts/train/03_smoke_hgt.py` —— 前向检查 h_dict[(N,128)]、pooled[(B,128)]、finite、focal train 提取。
**下一步**：4.4 sequence(Transformer over event tokens) + 4.5 fusion + Q-head + aux heads。

---

## Stage 4.3 ✅ + 4.4（2026-05-20）

### 4.3 HGT encoder — done（Windows 验证）
`encoders/hgt.py`：1.15M params；h_dict[track/signal/route/**trn**] (N,128)，pooled (B,128)，finite，focal train 提取正确。
- **关键 bug**：PyG `HGTConv`（HeteroDictLinear）按节点类型名建内部 ModuleDict，节点类型 `'train'` 撞 `nn.Module.train()` → KeyError。修：节点类型在 PyG 边界改名 `'train'→'trn'`（`PYG_NODE_KEY`，loader 的 to_heterodata + hgt 的 metadata/x_dict/pooling 全用 PyG key；taxonomy/stats/schema key 仍是 'train'）。坑：第一次只改了我自己的 ModuleDict key，漏了传给 HGTConv 的 `metadata=(NODE_TYPES,...)` 仍含 'train' → 必须 `metadata=(PYG_TYPES,...)`。见 TOOL_TRAPS §13。

### 4.4 Transformer event encoder — built（待 Windows 测）
`encoders/sequence.py`：token = Linear([node_emb(128) ⊕ state_emb(8) ⊕ sinusoidal_time(log1p dt,32)] → 128) → 4 层 Transformer(4 head, ff512) → h_seq_final（最后非 pad token）+ h_seq_pool（masked mean）。
- **asset 设计决策**：event token 存的是**子图节点本地索引**（非全局 asset id）。不用 spec §2.3 的全局 Embedding(673,64)，改为喂**该节点的 HGT embedding**（gather by local idx，在 §8 top-level model 里做，带 PyG batch offset）→ 序列↔图绑定、跨 snapshot 一致、无需重建数据。SeqEncoder 接收已 gather 的 node_emb (B,K,128)。
- 全 pad 行（无 event）做了 NaN 保护（强制留 1 个 slot，pooled 仍为 0）。
- 验证：sinusoidal shape (B,K,32)、AST OK。待 `python scripts/train/04_smoke_sequence.py`（合成输入）确认 forward。

### 下一步 4.5：fusion（§5，concat 7 组件→s_emb 256）+ Q-head（§6 per-action MLP）+ aux heads（§7 route + time MDN）。
注意：fusion 需要 special_flags（8）—— loader 目前没编码 state_special_flags，4.5 要补进 encode_snapshot + to_heterodata。schedule_global 用 platform 1-7 → 8-d onehot（spec 写的 7 是 1-6）。

---

## Stage 4.5 — fusion + Q-network + aux heads（2026-05-20，built，待 Windows 测）

- `encoders/fusion.py`：`ScheduleEncoder`（per upcoming train [hc_emb8 ⊕ eta1 ⊕ platform_onehot8]=17，masked mean over 5 → schedule_global 17-d；spec 写 16 是 platform 1-6，我们 1-7 → 8）；`Fusion`（concat 7 组件 → LN→512→LN→s_emb 256，in_dim=538 动态算，spec 的 657 是松散算术）。
- `policies/q_network.py`：per-action MLP。action_in=514（h_train128+h_route128+s_emb256+is_in_cand1+n_cand1），wait_in=513（h_train+h_seq_final+s_emb+n_cand），Q_all (B,K+1)，masked action→-1e9。
- `policies/heads.py`：`RouteHead`（param-free dot product h_train·h_route，masked）；`TimeHead`（[h_focal⊕s_emb]→Linear(384,128)→5 buckets；spec 叫 MDN 其实是 5 类分类器）；`time_bucket()` τ→{0:≤5,1:≤15,2:≤30,3:≤60,4:>60}。
- loader 扩展：`encode_special_flags`（8-d：7 bool + f_late_train/600 clip±5）加进 encode_snapshot + to_heterodata（data.special_flags）。
- 沙盒验证（torch-free）：time_bucket、special_flags、fusion in_dim 538、Q 514/513 全对。
- 待 Windows：`python scripts/train/05_smoke_fusion_q.py`（合成）确认 fusion/schedule/Q/heads forward + masked action=-1e9。

### 下一步 4.6：top-level model.py —— 把 HGT+Seq+Fusion+Q+heads 串起来，关键是 **event token 的 node_emb gather**（asset_idx 本地索引 → concat[h_track;h_signal] by PyG batch ptr offset，向量化）+ focal-train gather（is_focal）+ candidate h_routes gather（act_route_idx）。然后 end-to-end 真实 batch → Q 的 smoke。再 4.7 CQL/aux losses + 训练循环。

---

## Stage 4.6 — top-level RailRLModel（2026-05-20，built + gather 已验证，待 Windows end-to-end）

`src/railrl/model.py`：RailRLModel 串 HGT + Seq + Fusion + Q + route/time heads。
- **3 个 gather**（PyG batch ptr offset 向量化，沙盒 numpy 验证全对）：
  1. `gather_focal`：is_focal train 节点 per graph（scatter by trn batch）→ (B,128)。
  2. `gather_routes`：act_route_idx（本地 route idx，-1 pad）+ route_ptr → (B,14,128)。
  3. `gather_event_nodes`：ev_asset_idx（本地 [track;signal] idx）+ track/signal ptr → (B,K,128)，喂给 SeqEncoder。
- forward(data) → {Q (B,K+1), route_scores (B,14), time_logits (B,5), s_emb, h_focal, h_routes}。
- fusion in_dim = 128*4 + 17 + 8 + 1 = 538（动态算）。
- 待 Windows：`python scripts/train/06_smoke_model.py` —— 真实 batch forward + backward，检查 shape/finite + **chosen action 的 Q 不被 mask**（label↔mask 一致性）+ 梯度流过所有分支。

### 模型搭完里程碑
spec 03 §2-§8 全部实现并逐件单测。4.6 end-to-end 过了之后，Stage 4 模型部分完成，进 4.7：CQL(α=5) + IQL + BC + aux losses（spec 04 §3-4）+ 3 阶段训练循环（5+15+20 ep）。

---

## Stage 4.6.5 — v2 reward 重算（2026-05-22，脚本已建+逻辑单测，待 Windows 跑）

### 决策：重算，不将错就错（用户「以不妥协的原则…将错就错的行为要杜绝」）
进 4.7 训练前必须有**和当前 state/action 对齐的 reward**。现状的 reward 是 v1 残留，与 v2 决策集不对应——若拿去训 CQL/IQL/BC，每个 reward 都会静默错配到另一组 state/action（典型「将错就错」）。故：**从 `decision_points_v2` 重新计算 PR outcomes + per-decision rewards，并填回 snapshots**。

### 关键发现（动手前先查清，避免又一次将错就错）
1. **reward 是嵌在 snapshot 行里的**（`mdp/schema.py` REWARD_COLS 14 列），不是独立表 join。但 builder 写的是 **NaN 占位**（`state.py:265`「will be joined from decision_rewards.parquet downstream」）→ 当前 `snapshots_v2.parquet` 的 reward 全是 NaN，训练读 `r_total`（`input_pipeline.py:317`）会拿到 NaN。**训练前必须填。**
2. `outputs/rewards/decision_rewards.parquet`（726,978 行）+ `pr_outcomes.parquet` 是 **v1 残留**（旧 727k 决策点），与 v2 的 1,996,572 行 snapshots 不对应、不可 join。
3. **v1 reward 脚本（`scripts/data/09,10`）从未在 v2 跑通**：它们 `from railrl.p2_data_eng.snapshot import StaticGraphView`，而 v2 shim（`p2_data_eng/__init__.py`）不导出 `snapshot`（StaticGraphView 已搬到 `railrl.data.static_graph_view`）。所以那份 decision_rewards 是直接拷贝过来的，不是 v2 重算的。
4. **JOIN KEY = `sample_id`（整数）**：`05_build_snapshots.py:122-123` 用 `dp.reset_index(drop=True); sample_id = arange(len)` over `decision_points_v2`。只要重算时**完全复刻这一步**，就能拿 sample_id 做干净的 1:1 填充——比用 (focal_train,focal_signal,t,route) 四元组 join 稳健得多（无 timestamp/dtype/重复歧义）。
5. **列名桥**：snapshot 用 `outcome`/`r_throughput_raw`/`r_headway_raw`；reward_model 产出 `route_outcome`/`r_thru_raw`/`r_head_raw`。`reward_v2.REWARD_MERGE_MAP` 在合并时桥接，让 `decision_rewards_v2.parquet` 保持与 v1 同 schema（v1 health-check 脚本可复用）。v2 决策点用 `t`/`trigger_type`，v1 feature/episode 代码要 `time`/`trigger` → `build_rewardfmt()` 一次性改名。

### 交付（3 脚本 + 1 helper，均 numpy-core 可沙盒单测、重活靠 Windows）
- `src/railrl/mdp/reward_v2.py`：路径常量 + `build_rewardfmt()`（复刻 sample_id 派生 + t→time/trigger_type→trigger）+ `REWARD_MERGE_MAP`。
- `scripts/mdp/08_label_pr_outcomes_v2.py` → `pr_outcomes_v2.parquet`（复用 v1 `label_all_prs` 跑在 rewardfmt 上）。
- `scripts/mdp/09_compute_rewards_v2.py` → `decision_rewards_v2.parquet`（修 import 到 `railrl.data.*`；带 sample_id；schema 同 v1）。
- `scripts/mdp/10_merge_rewards_into_snapshots.py` → `snapshots_v2_rewarded.parquet`（pyarrow 按 row-group **流式**填 reward 列，struct 列原样透传、内存有界；**不覆盖原文件**）。

### 沙盒单测（pandas/numpy，无需 torch/parquet）—— 全过
schema↔map 14 列对齐；build_rewardfmt 改名+sample_id=arange+chosen_route_id NaN-safe；merge LUT 按 sample_id 填充（**乱序无关**、未匹配→NaN、`r_thru_raw→r_throughput_raw`/`r_head_raw→r_headway_raw`/`route_outcome→outcome` 桥接、outcome 字符串+None）。4 文件 `py_compile` OK。
- ⚠️ 沙盒 `/sessions` 仍 100% 满（18M 空闲）→ 装不了 pyarrow、读不了大 parquet。pyarrow 流式写 + v1 feature 函数读大文件只能 Windows 端验。

### Windows 运行顺序 + 要回传的报告
1. `python scripts/mdp/08_label_pr_outcomes_v2.py`（先 `--limit 20000` 冒烟）→ 看 outcome 分布（used/unused_cancelled/unused_timeout/unknown）≈ v1。
2. `python scripts/mdp/09_compute_rewards_v2.py`（先 `--limit 20000`）→ 看 r_total mean/std、4 分量 mean、feature coverage、`sample_id is_unique` assert。
3. `python scripts/mdp/10_merge_rewards_into_snapshots.py` → 看 **matched %（应≈100%）**、r_total finite %、set/wait 数。
4. 核对无误后，把 `snapshots_v2_rewarded.parquet` 改名为 `snapshots_v2.parquet`（或改 `C.SNAPSHOTS_V2_PARQUET` 指向它），训练即读到真 reward。

### 修正 1（2026-05-22）：pr_outcomes join 必须走 sample_id，不能用四元组
跑 `09` 时 `assert sample_id is_unique` 触发。根因：第 2 步把 `route_outcome` 用 `(time, focal_signal, focal_train, chosen_route_id)` 四元组 left-join 到 dp 上，而**该四元组不唯一**（decision_events 里有同一瞬间、同信号、同进路的重复 PR）→ 一对多 join 行翻倍（set 546,418 → 548,676，多 2,258）→ sample_id 重复。这是 v1 `10_compute_rewards.py` 沿用的脆弱写法（v1 没这个 assert，是**静默**带病写出的）。
- 修：`pr_outcomes.py` 输出加 `sample_id`（加性、有 guard，v1 决策点没这列就跳过）；`09` 改成 `merge(on="sample_id")` + 行数不变断言。沙盒对照测确认旧法膨胀(5→7)、新法 1:1(5→5)，且 sample_id join 对重复四元组也精确。
- ⚠️ **要先重跑 `08`**（重新生成带 sample_id 的 pr_outcomes_v2），再跑 `09`。

### 已知局限（2026-05-22 与用户确认：保留，非 bug，按不妥协原则**不修改**）
`09` 跑出两个低覆盖率，讨论后判定为**数据本身的天花板 + 设计内的保守归因**，不是缺陷：

1. **approach_distance 覆盖 47.8%**（261,178/546,418 set）。算它要先知道决策时刻列车在哪个 TC。查不到的两类原因：(a) TD 占用没带车次号；(b) **信号员常按 schedule 提前排路，列车此刻还没进入该区域，本就没有 TC 可查**（用户领域补充）。距离未知 → `gate=0` → 该决策 r_delay 记 0。这是**保守不归因**（无法确认列车在跟前就不把延误算到这次排路头上），符合 spec §9.5 门控语义。
2. **delay_change 覆盖 6.4%**（128,697/1,999,623）。靠 TRUST/Movements，**只在有 TIPLOC/STANOX 的计时点才上报**（用户补充），而 TD 是实时逐状态。决策每几秒一个、计时点十几分钟一个 → 多数决策前后 70min 窗口内没有计时点夹住（out_window=153.7 万）。**spec 第 603 行本就预期 ~11.5%**，我们实际还略好。

**关键判断**：缺失值一律保守留 0（不归因），**不是塞错值**——所以不是「将错就错」。反之若为刷覆盖率而放宽 70min 窗口 / 未知距离默认 gate=1.0，等于把对不上因果的延误硬安到决策上，那才是往奖励灌错误信号。故**保持现状**。
**下游影响（记着，留到后面处理）**：r_delay 需 approach∩delay 两者都有才非零（set 里 ~3%），故虽 w_delay=1.0 最高，实际贡献极小（均值 −0.0025）；奖励现由 r_wait(−0.218)、r_throughput(+0.136) 主导。→ 在**论文「局限/效度威胁」**如实写明；**spec 04 IRL 阶段**权重重学时它会自然降权。瓶颈是 TRUST 稀疏，管线层面无法无中生有。

### `09` 全量结果（2026-05-22，校验通过）
decision_rewards_v2.parquet：1,999,623 行（无膨胀），episodes 82,858，sample_id 唯一。r_total mean/std/min/max = −0.098 / 0.481 / −30.3 / +30.5。四分量均值 r_delay −0.0025 / r_throughput +0.1356 / r_headway −0.0130 / r_wait −0.2180（和 = −0.098 ✓）。逐项算术核对自洽：r_wait=0.3×(wait占比0.727)、r_throughput=0.5×0.992×(set占比0.273)、r_headway≈4.9% 可测 headway<H_min（与 p5 标定自洽）、±30 来自 delay clip 1800s。

### 修正 2（严重 / 2026-05-22）：sample_id 必须复刻 build_episodes 的重排序，否则 reward↔snapshot 静默错位
合并后抽查发现：同一个 `sample_id`，snapshot 里是 `set`、reward 表里是 `wait` —— 两套 sample_id 指向的**不是同一个决策**，整张 reward 按 sample_id 填进了错的 snapshot。`merge report` 的 matched 100% 完全没报警（每个 id 都能在 LUT 找到，但指向错了），**这正是最隐蔽的「将错就错」**。
- **根因**：`05_build_snapshots.py:121-123` 是先 `build_episodes(dp)` **再** `reset_index`+`arange` 分配 sample_id。而 `episode.build_episodes` 内部 `sort_values(["focal_train","t"])` → 分 pass → `sort_values(["pass_id","t"])`，**重排了行序**。我最初的 `build_rewardfmt` 在 decision_points_v2 **自然行序**上分配 sample_id → 与 snapshot 的 sample_id 空间错位。
- **修**：`reward_v2.build_rewardfmt` 改为**逐字复刻** 05 的派生（同样 `_load_pass_assignments()` 读 `PASS_ASSIGNMENTS_PARQUET` + `build_episodes` + reset_index + arange）。build_episodes 是纯函数、排序对相同输入确定 → 复刻后 sample_id 与 snapshot 完全对齐。合成数据验证：两次 build_episodes 顺序一致（确定性 ✓）、自然序 vs episode 序确实错位（复现 bug ✓）、复刻后 100% 对齐（修复 ✓）。
- **教训**（写入 TOOL_TRAPS 候选）：**两个产物若靠"复刻同一派生"对齐，必须复刻到最后一步、且要用独立信号（这里是 label）验证对齐，不能只看 join 的 matched%。** matched% 只证明 key 存在，不证明 key 指向同一实体。
- **顺带修**：`10_merge` 里 `str(pd.NA)` 会把空 outcome 写成字面 `"<NA>"`，改用 `pd.isna(v)` → 真 null。
- ⚠️ 沙盒此刻 `/sessions` 满 → virtiofs **截断**了 reward_v2.py（只看到 109 行），py_compile 假报 IndentationError；Windows 端真文件完整（§11 trap 复发）。校验改在 Windows 跑。

**重跑顺序（Windows）**：先跑对齐校验（rewardfmt 的 sample_id→label 对 snapshot 的 sample_id→label，应 100% 一致）；过了再 `08`（重生成带正确 sample_id 的 pr_outcomes_v2）→ `09` → `10`（输入指向 `snapshots_v2.prereward.parquet` 干净备份）。原合并产物（已被改名成 snapshots_v2.parquet）是错的，重做后覆盖。

### ✅ 4.6.5 完成（2026-05-22）
对齐校验 `label agreement 100.0000% / mismatches 0`（独立信号）。重跑 08→09→10（从 prereward 干净备份合并）：matched 100%、r_total finite 100%。逐行语义校验全 0：set 行 `r_wait_raw=0`+真 outcome、wait 行 `r_wait_raw=-1`+null outcome，set outcome 分布 `used 540,871 / unused_timeout 1,587 / unused_cancelled 900 / unknown 9`（无 `<NA>`）。四分量与 09 的差异完全由「snapshot 比 reward 表少 3,051 个 set」解释（占比算术自洽）。
产物：`snapshots_v2_rewarded.parquet` 改名覆盖为 `snapshots_v2.parquet`（loader 读 `C.SNAPSHOTS_V2_PARQUET` 直接拿到真 reward）；`snapshots_v2.prereward.parquet` 保留为无-reward 备份。normalization_stats.json 无需重算（仅 reward 列变化）。

---

## Stage 4.7 — 训练（in progress，2026-05-22）

### 4.7.1 ✅ 损失模块（Windows smoke 过）
`src/railrl/algorithms/losses.py`（spec 04 §2）：CQL（L_TD + α·L_cons，α=5；target-net bootstrap，掩码动作 -1e9 → logsumexp/max 自动只算合法集）、IQL（expectile τ=0.7 / Q / AWR β=3）、BC（CE-on-Q + route-CE + wait-BCE λ=0.3）、aux（L_route=CE(set,a-1)、L_time=CE(valid bucket)）、totals（L_total=L_CQL+0.5·L_route+0.2·L_time）、`soft_update`(τ=0.005)。`07_smoke_losses.py` 全过，算术自洽（L_CQL=L_TD+5·L_cons、L_total 等逐项核对）。

### 4.7.1.5 ✅ 时间划分 + 重算 normalization（**修了时间泄露**）
**发现**：旧划分是 `md5(pass_id)` 哈希=随机，违反 spec 04 §4.1 锁定的**按时间划分**，且项目「教训 6」明说随机划分因时间泄露虚高 6–8pp；normalization 还在哈希-train 上算（偷看未来）。用户选「按时间·整 episode 划分」。
**修**：`config.PASS_SPLIT_PARQUET`；`input_pipeline.time_split_of(t)`（train<2024-02-01 / val<2024-03-01 / test≥）+ `load_pass_split()`；`00_build_time_split.py` 按**每个 episode 起始时间**整体归类 → `pass_split.parquet`；`SnapshotDataset` 与 `01_build_normalization_stats.py` 都改读它（缺失才回退哈希并告警）。
**结果**：14,494 episodes → train 8,576 / val 992 / test 4,926；行 86.6% / 5.3% / 8.0%（时间密度不均，量级合理）。重算 normalization：split 一致、**vocab 不变**（track_id 268 / signal 123 / route 278 / train 2184）→ 编码器无需重建，仅 z-score 变。**关键性质：按整 episode 切 → s 与 s' 必在同一 split。**

### 4.7.2a ✅ 转移数据集（待 Windows smoke 复跑）
`src/railrl/algorithms/transitions.py`：`TransitionDataset` 产 (s, a, r, s', done)——s'=同 pass_id 的 position+1，**done=「position+1 不存在」（基于位置，不是 is_last）**。单扫一遍建后继映射；因 episode 不跨 split，每 pass 恰一个 max-position 终止。`transition_collate` 把 s / s' 各拼成一个 PyG Batch + done。`08_smoke_transitions.py`：验后继/done（终止数==episode 数）、collate、model(s)+target(s')+cql_total+backward。
- ⚠️ **数据 quirk（已绕过）**：snapshot 的 `is_last_in_episode = (t == max t in pass)`，当末尾几个决策时间戳相同会把**多行**都标 True（val 里 992 episodes 却有 1,216 个 is_last），且被标的 tie 行可能仍有真后继。终止性必须**按 position**判（position_in_episode 是干净的 cumcount，唯一）。→ **下游 eval/discount 截断也应按 position，不要信 is_last**；未来重建 snapshot 时可顺手修 is_last 定义。

### 待办（4.7 收尾前）
1. **τ/time_bucket 标签**：L_time 需 τ = t_first_TC − t_PR（spec 03 §7.2），目前数据没有。复用 reward_features 的占用扫描算 τ，做成 sidecar（sample_id→time_bucket）给 trainer。在此之前 L_time=0（占位、可插拔）。
2. **loader 读放大**：`SnapshotDataset/_load` 每取 1 行要读整个 row group（~4096 行）→ 训练会很慢。trainer 阶段加 row-group LRU 缓存或顺序/分片读。
3. **4.7.2c trainer**：AdamW(lr3e-4,wd1e-4,emb/LN 不衰减) + warmup1000+cosine（每阶段重热）+ grad clip1.0 + target net（Phase B clone, soft τ=0.005）+ 3 阶段（A 仅 encoder+aux 5ep / B 冻 encoder 训 Q 15ep / C 联合 20ep）+ 分层采样(§4.4) + ckpt + 3 seeds。

### 4.7.2b ✅ τ/time_bucket 标签（Windows 跑完）
`reward_v2.compute_lead_time_buckets`（τ = 路线首 TC 首次占用 − t_PR，spec 03 §7.2，复用占用扫描）+ `scripts/mdp/11_compute_time_labels_v2.py` → `time_labels_v2.parquet`(sample_id, tau_s, time_bucket)。`to_heterodata` 加了 `data.sample_id` 供 trainer 按 sample_id 挂标签。
- τ 覆盖 **97.0%**（530,021/546,418 set）。
- ⚠️ **bucket 高度倾斜**：bucket0 0.1% / b1 0.2% / b2 1.5% / b3 4.3% / **b4(>60s) 93.9%**。说明 spec §7.2 锁的边界 `[5,15,30,60]` 对本数据没标定（94% 落一类）→ L_time 头会退化成常猜 b4。likely 真实（信号员提前排路）。**待用户定**：保持 spec 边界（L_time 弱 aux）vs 按 τ 分位重标定边界（让 L_time 有信息量、撑可解释叙事，但偏离 locked spec）。

### 4.7.2b' ✅ time_bucket 重标定（用户选「按训练集 τ 分位」）
`12_recalibrate_time_buckets.py`：在 **train split** τ 上取 p20/40/60/80 → 边界 **[98,121,153,204] s**（中位 lead ≈2 分钟，证实提前排路），5 桶均衡（~20% 各）。写 `time_bucket_edges.json`（边界 + provenance，供论文/XAI）。复用 tau_s，无需重扫事件。

### 4.7.2c ✅ trainer（**Windows smoke 端到端过 — Stage 4 管线打通**）
`src/railrl/algorithms/trainer.py`（param groups: emb/LN/bias 不衰减；phase_lr warmup→cosine；冻结 encoder；build_time_lut；compute_loss per-phase；evaluate route/action acc）+ `scripts/train/09_train.py`（3 阶段编排 + ckpt + `--smoke`）。
- smoke 三阶段全过，损失公式逐项核对：A `0.5·route+0.2·time`、B `L_TD+5·L_cons`、C `CQL+0.5·route+0.2·time` 全对；LR 调度/grad clip/soft update τ=0.005/ckpt+log 正常。
- **bug 修**（smoke 抓到）：`losses._zero` 原用 `new_zeros()` 脱离计算图 → Phase A 全-wait batch（route+time 都 0）的 loss 无 grad_fn，backward 报错。改 `like.sum()*0`（图内零，梯度 0）→ 全-wait batch 变 no-op 不崩。
- `nested tensors prototype` 警告无害（PyTorch Transformer 内部）。
- algo：trainer 目前只做 CQL（主）。IQL 需加 value head；BC 是独立 baseline（Stage 7）。

### ⚠️ 真训练（Stage 5/6）前的阻塞项 —— 4.7.2d
1. **loader 性能（阻塞）**：`SnapshotDataset/TransitionDataset._load` 每取 1 行 `read_row_group` 整组（~5000 行）→ ~16 _load/s，1.7M×2×40ep 完全不可行。**关键发现**：snapshots 按 (pass_id,t) 顺序存（05_build：build_episodes 排序后 sample_id=arange 顺序写），所以 **s' = sample_id i+1 = 文件里紧邻的下一行 → 与 s 几乎同 row group**。故可写**流式 IterableDataset**（按 row-group 顺序/块洗牌，每组只解码一次，2-组缓存供 s/s'）→ ~数千倍加速，转移保持完整。备选：预编码成定长 memmap 张量随机访问。
2. **分层采样（spec §4.4）**：trivial 90% 会淹没梯度，需按 special_flags 分层加权采样。与流式块洗牌如何结合需设计。
3. （小）IQL value head；BC baseline。

### 下一步
4.7.2d：loader 性能 + 分层采样（Stage 5 sanity 前必须）。**因对话已极长**，建议在新对话里基于本 log 接着做这块（设计空间较大）。

---

## Stage 4.7.2d — 环境/数据事实 + loader 设计前的关键发现（2026-05-22，新会话）

新会话开场通读了 log/TOOL_TRAPS/spec03+04，并用 Read 工具核对了真实源码与产物（未在沙盒读大 parquet——/sessions 100% 满，§11 复发）。记录本次确认的事实 + 一个会改设计的关键发现。

### 运行环境（存档，直接影响 loader 选型）
- **服务器**：A100 40GB（真训练 Stage 5/6 靠它）；系统盘 20GB 总，RailRL_v2 项目约 8GB，平时空闲 **~10-12GB**（偶尔降到 2-4GB）。
- **本地**：RTX 5070 8GB（只够 smoke / 弱 GPU）；磁盘 4TB（管够）。
- **矛盾**：大 GPU 在缺盘机器、大盘在弱 GPU 机器。→ memmap 预编码（定长 dense ≈ 30-40GB，padding 到 caps）**放不进服务器**、即便烤到本地 4TB 服务器 A100 也读不到 → **排除 memmap，定为「流式读 573MB parquet」**。
- smoke 在 RTX 5070 上 batch 调小（64-128）；A100 上才用 spec 锁的 256。不影响 loader 设计。

### 数据时间覆盖有 gap（用户领域知识存档）
日历跨度 2023-02-28 ~ 2024-04-25，但**实际有数据的是 10 段**（段间为空）：
2023: 02-28~03-01 / 03-07 / 03-11~03-16 / 04-04~04-17 / 05-05~05-26 / 05-30~07-31 / 08-10~08-19 / 09-01~12-07；2024: 01-18~02-27 / 03-08~04-25。
- **影响转移正确性？不影响**：pass 由 TRUST interval 或 `PASS_FALLBACK_GAP_S=6h`（config.py:216）切分，绝不跨多日 gap；s'=同 pass `position+1`，故 gap 前最后一个决策天然是该 pass 终止（done=True），**不会产生跨 gap 的假转移**。
- **好性质**：val/test 边界（2024-02-27→03-08 有 9 天数据空档）正好落在 gap 里 → val 与 test 时间上天然隔离。train→val 边界（01-31→02-01）在连续段内，但按整 episode 归类，至多个别 pass 跨午夜，可忽略。

### shuffle 分析结论（"按时间序列数据是否需要打乱"）
- **需要 shuffle，但只需块级；不丢时间信息、不造成泄露。**
- 时间信息装在**每个样本内部**（event tokens / 窗口占用率），不靠 batch 顺序体现。CQL/IQL 逐条独立回填 `y=r+γ·max Q_target(s',·)`，spec 04 §4.3 明确**不按 episode 顺序成批**。
- shuffle 的真实理由：(1) 去相关相邻 batch（SGD 稳定，否则相邻时刻高度相似把梯度带偏）；(2) 分层采样必须打破"特殊情况成时间簇"。
- 泄露已由"按时间·整 episode 划分"在 split 层解决；**train 段内部怎么打乱无泄露含义**。
- 块洗牌（打乱块顺序 + 缓冲区内打乱）足够，是流式训练标准做法（TF shuffle buffer / WebDataset）。纯顺序读会伤收敛且没法分层。

### ⚠️ 关键发现：snapshots_v2.parquet **不是 sample_id 顺序**（4.7.2d 原前提作废）
- log 里 4.7.2d 线索写的"s' = sample_id i+1 = 文件里紧邻下一行 → 与 s 几乎同 row group"，**对当前生产文件不成立**。
- **根因**：文件是 `05b_build_snapshots_parallel.py` 6-shard 产物。`05_build_snapshots.py:123` 在**全量 dp**（已 `build_episodes` 按 (pass_id,t) 排序）上赋 `sample_id=arange`，**然后** line 129 做 **strided 切片 `dp.iloc[K::6]`**；`05b` line 100-110 合并是 **part0 ++ part1 ++ … ++ part5 直接拼接（无重排）**。
- 所以文件物理顺序 = **6 段各自按 sample_id 升序的 strided run 首尾拼接**，全局**非** sample_id 顺序。sample_id i 的后继 i+1 落在另一个 part 区域、文件里相隔数十万行。
- **正确性不受影响**：`TransitionDataset`/`SnapshotDataset` 用 `(pass_id, position)` 建后继映射，与文件物理顺序无关（这就是 08_smoke 能过的原因——smoke 数据小，性能问题暴露不出来）。受影响的只是"s' 与 s 同 row group"这个**性能前提**。
- **待独立验证（Windows）**：读 `sample_id` 列查单调性 / run 结构（验证脚本见 `docs/4_7_2d_loader_design_DRAFT.md`）。符合"用独立信号交叉验证、不凭假设"。
- **设计影响**：流式 loader 二选一——(a) 一次性把文件**重排成 sample_id 顺序**（之后 2-组缓存即可，最简单/canonical）；或 (b) 用 `sample_id→(rg,li)` 索引 + ~8 组 LRU 做 **K 路归并式顺序流**（不动已验证文件）。详见草案文档。

> 本节为事实/决策记录（append-only）。loader 具体设计写在 `docs/4_7_2d_loader_design_DRAFT.md`（待 Hao 审核），定稿后再把最终方案摘要追加到本 log。

### 验证结果（Windows，2026-05-22）+ 🔴 episode 定义红旗
- **脚本1（文件顺序）**：rows=1,996,572；全局单调=False；下降处=5（→6 段）；步长众数=6；min/max sample_id=0 / 1,999,622。
  - → **证实文件是 6-shard strided 交错**（path-甲 关于"文件非 sample_id 顺序"的判断成立）。
  - → 且 **sample_id 非连续**：max 1,999,622 = 全量决策 1,999,623−1，与 rows 之差 3,051 = 跳过的无-TC 决策数。**含义**：(a) 重排后 sample_id 是"递增但有 3,051 个洞"，不是 0..N-1 连续（草案 §3.2 校验描述需更正）；(b) 流式配对**必须按 `position+1` 判定**（不能只靠"流里下一条"，否则跳过造成的洞会配出错误 (s,s')）。
- **脚本2（pass 时间跨度）**：**max=34,302,283s ≈ 397 天，p99≈345 天**。🔴 **pass 不是时间局部的**。
- **根因**（读 pass_assignment.py + episode.py 确认）：`build_pass_intervals`（实际被 05 用的快路径）按**完整 TRUST train_id** groupby 取 min/max actual_timestamp 作为 pass 区间。但 TRUST train_id 的 `EE` 段是**当月日期（day-of-month）**（见上文 Table 3.6 存档），同一 train_id **每月复用** → groupby min/max **跨越整个数据集** → TRUST-matched pass（占 93.7%）的区间动辄横跨数月。fallback（6.3%）有 6h gap 切分，**TRUST 路径没有**。
- **后果**（将错就错风险，进训练前必须先解）：
  1. **跨月假转移**：build_episodes 按 (pass_id,t) 排序赋 position，跨月决策被串成 position 相邻 → CQL `y=r+γ·max Q_target(s',·)` 把"一个月后的状态"当作下一状态。
  2. **时间划分泄露**：00_build_time_split 按 episode 起始时间归类；跨月 pass 起于 2023 → 整段（含 2024-03/04 test 期决策）落入 train split。**这恰好抵消了 4.7.1.5 本想修的时间泄露**——whole-episode 归类在 episode 非时间局部时反而把未来灌进 train。
  3. γ-return 失真。
- **episode 计数不一致待查**：4.7.1.5 报 14,494 episodes（00_build_time_split 读 snapshots），09 报 82,858（decision_rewards_v2）。同一 pass 定义不应差 5.7×。
- **结论**：path 甲（重排）机械上没问题，但**先别做**——重排只会忠实保留错的 episode 结构（"在错位数据上往下建"）。应先：(1) 用诊断脚本量化（跨 gap 转移数 / split 泄露行数 / 跨度分位 / 计数对账）；(2) 修 TRUST pass 的 gap 切分（让一个 pass = 一段物理行程、时间局部）；(3) 重算 episode 元数据 + pass_split（**sample_id 不变、reward 按 sample_id 无需重 join**，仅需 patch pass_id/episode_idx/position/is_last + 重生成 pass_split）；episode 干净后再做 loader。

### Hao 决策 + 修复范围确认（2026-05-22）
- **Hao 选：暂停 loader，先修 episode 定义。**
- **修复范围 = 列 patch，不需重建 snapshot**（已读码确认 pass_id 是纯 identity）：
  - `state.py` 构建 state **不引用 pass_id** → state 特征与 episode 分组无关。
  - `01b_enrich_candidates.py`（实际持久化候选的脚本）只用 `routes_from(focal_signal)`（Rule 1），**不用 pass_id / prev_routes**；`build_pass_route_history`/`prev_routes`（action.py）仅被诊断脚本 `02_validate_candidates.py` 调用，**从未写进数据**。
  - 逐步 reward 按 `sample_id` 对齐（不变）、且只依赖 outcome/delay/headway，与 episode 边界无关。
  - → 故修复只需重算并 patch `pass_id/episode_idx/position_in_episode/is_last_in_episode` + 重生成 `pass_split.parquet`；**无需 100min 重建、无需 reward 重 join**。
- **下一步**：Hao 在 Windows 跑诊断脚本量化规模（泄露行数 / 跨 gap 转移数 / 跨度分位 / episode 计数对账），再据此定 gap 切分阈值与是否保留 TRUST id 作子标签。

### 诊断结果（Windows，2026-05-22）—— 确认严重，必须修
- **跨度(天) p50/p90/p99/max = 0.01 / 1.01 / 345 / 397**。pass 总数 14,494；>1天 1,559（10.8%）；>6h 1,772。
- 🔴 **>1天 的 pass 装了 84.7% 的行** —— 时间破碎的 episode 恰恰是决策最多的（同 headcode 跨月累积）→ **全数据 85% 都在跨天 episode 里**，不是尾部小问题。
- pass 内步长(秒) p50/p90/p99/max = 21 / 129 / 86,892(~24h) / 20,905,220(~242天)。**步长>1h 转移 66,981（3.35%）**、>1天 20,947（1.05%）= 假转移。
- 🔴 **时间泄露：129,021 行 test 期（≥2024-03-01）决策被划进 TRAIN split**（test split 本应 ~16万行，泄露量级巨大；4.7.1.5 想修的泄露被 whole-episode 归类反向放大）。
- **episode 计数对账**：当前 snapshots = **14,494**（与 4.7.1.5 一致）。09 报的 82,858 是**修正2 之前**（自然序、未对齐 sample_id）的旧数；修正2 后 reward 已复刻 05 派生、label agreement 100% 确认按 sample_id 对齐 → **82,858 作废，现况 14,494**，无活跃不一致。
- **判定**：三个问题（85% 行在跨天 episode / 129k 泄露 / 67k 假转移）任一都足以让 Stage 5/6 结果失效。**必须先修 episode 定义再训。**

### 修复方案（待 Hao 定 2 个参数）
- **做法**：按 `(focal_train, gap>G)` 重新分段（即把现有的 `_assign_pass_by_gap` 逻辑施加到全体决策，不只 fallback），使每个 episode = 同 headcode、内部无 >G 间隔的时间局部连续段。重算 `pass_id/episode_idx/position_in_episode/is_last` + 重生成 `pass_split`。**sample_id 与 reward 全程不动**（reward 按 sample_id 对齐，不受影响）。
- **待定参数 1 — gap 阈值 G**：数据上正常行程步长是秒~分钟（p90=2min），复现间隔是 24h+，故 G 取 [30min, 6h] 任意值都能干净切分；倾向 **1h**。
- **待定参数 2 — 落地方式**：sidecar（写 `episodes_v2.parquet`: sample_id→新 episode 列，loader/split 读它，**不动 573MB 验证文件**，推荐）vs 原地 patch（重写 snapshots 的 4 列，canonical 但要改验证文件）。
- **修后复验**：跨度全部 <G；零跨-G 转移；零 test→train 泄露；episode 计数合理；label↔sample_id 仍 100%。

### Hao 决策（2026-05-22）：G 由数据算、落地走 sidecar
- **落地 = sidecar**（不动 573MB 验证文件）。
- **G 不拍脑袋，用数据选**：新增 `scripts/mdp/13_episode_gap_analysis.py`——(1) 按 focal_train 分组算 inter-decision gap 分布（预期双峰：行程内秒~分 / 复现间隔时~天，中间空谷，G 取空谷）；(2) 候选 G 网格 [5m,15m,30m,1h,2h,6h,12h,1d] 敏感性扫描（episode 数 / 跨度分位 / >1天行占比 / 被切转移数 / split 泄露行数），G 取指标稳定平台起点。Hao 在 Windows 跑、回传输出后锁定 G，再写 patch（重算 4 列 → `episodes_v2.parquet` sidecar + 重生成 pass_split；sample_id/reward 不动）。

### 13_episode_gap_analysis 结果（Windows，2026-05-22）—— 漂亮的双峰，G 有据可依
- **gap 分布（同 focal_train，N=1,994,390）强双峰**：
  - 行程内模式 0s~5min，峰在 30-60s（17.5%），**累计 ~99% 的 gap < 5min**。
  - **空谷 [30min, 12h]**：30-60m 0.09% / 1-2h **0.04%（最低）** / 2-6h 0.05% / 6-12h 0.04%。
  - 复现模式：12-24h 回升到 **2.42%（次日复现）**，再 1-3d/3-7d/7-30d/30d+ 拖尾。
  - → 空谷仅约 2,600 个 gap，G 取空谷内任意值几乎等价。
- **敏感性扫描**：G∈[15min,6h] 稳定平台（episodes ~79-86k、span_max <1d、>1天行占比 **0%**、泄露 6~85）；**G=1d 崩**（episodes 30,011、span_max 45.04d、77.67% 行跨天、泄露 14,444——没切开 12-24h 次日复现）；12h 开始轻微漏（span_max 1.5d）。
- **泄露 129,021 → 个位/二十几行**：残留是 **train/val 边界(2024-02-01)跨午夜的合法单次行程**；val/test 边界(2024-03-01)落在数据 gap 内无跨越。→ patch 里**额外按 split 日期边界切一刀** → 泄露精确归零。
- **推荐 G=2h（7200s）**：数据驱动的密度最低点（auto 选中 1-2h 桶上界），落稳定平台；结果对 G∈[30min,6h] 不敏感。G=2h 时：episodes 80,207、median span 927s、p99 5215s(~87min)、max 0.29d(~7h)、泄露 24。**待 Hao 确认 G。**

### Hao 锁定 G=2h（2026-05-22）+ 修复全范围（3 步，泄露归零设计）
- **G = 2h（7200s）锁定**。patch 额外在 split 日期边界（VAL_START 2024-02-01 / TEST_START 2024-03-01）也切，保证无 episode/转移跨 split → **泄露精确归零**。
- **关键认识：修 episode 会改变 split 成员**（129k 行离开 train）→ **train 集变了** → **`normalization_stats.json` 必须按修正后的 train 重算**（旧 z-score 偷看了泄露的未来行）。vocab 不受影响（从全 split 建 → 编码器不动），只 z-score μ/σ 变。
- **3 步修复**：
  1. `scripts/mdp/14_resegment_episodes.py`（新）：按 (focal_train, gap>2h, split边界) 重分段 → `episodes_v2.parquet` sidecar [sample_id, pass_id, episode_idx, position_in_episode, is_last_in_episode, split] + 重生成 `pass_split.parquet`（新 pass_id）。**sample_id/reward/state 全不动**。自验证：跨度、零跨-G 转移、零跨-split 转移、零泄露、sample_id 覆盖==snapshot、各 split 计数。
  2. `01_build_normalization_stats.py`：改成按 sidecar（sample_id→split）定 split，重跑 → 新 normalization_stats.json（z-score 在修正 train 上；vocab 应不变）。
  3. loader（4.7.2d）：从 sidecar 按 sample_id 读 pass_id/position/is_last/split（文件内旧列作废）。
- **`00_build_time_split.py` 被 14 取代**（14 直接产出 pass_split，且按新 pass_id）。
- 顺序：先写+跑+验证 14（自验证通过再往下），再改+跑 01，最后 loader。**先验证 14 干净，再在它上面建。**

### 14_resegment_episodes 结果（Windows，2026-05-22）✅ 验证全过
- episodes **80,210**（= 扫描预测 80,207 + 3 个 split 边界额外切的跨午夜行程）。
- 自验证 5 项 assert 全过：零跨-G 转移 / 零跨-split 泄露 / is_last 每 episode 唯一（按 position）/ sample_id 全覆盖+唯一 / pass_id↔episode 一一对应。
- 跨度(秒) p50/p90/p99/max = 926 / 1827 / 5215 / **24783（6.9h）** —— 全 <12h，时间局部。
- split（行/episode）：train 1,472,064(73.7%)/59,121 · val 186,145(9.3%)/7,350 · test 338,363(16.9%)/13,739。行/episode 求和均自洽。
- **🔴→✅ 泄露 129,021 → 0**。split 由 train 86.6%/test 8.0% 变为 73.7%/16.9%（12.9 万行回归 test）——正是泄露修复方向。test 16.9% 比 spec ~10% 大，是真实数据密度（test 窗 03-08~04-25 数据密集 ~7 周）+ 锁定 split 日期的自然结果，时间干净无泄露，对评估更充分，非问题。
- 产物：`episodes_v2.parquet`（sample_id→新 episode 列）+ `pass_split.parquet`（新 pass_id）。**sample_id/reward/state/4.6.5 的 label↔sample_id 对齐全不受影响**（sidecar 仅从 snapshot 自身 sample_id/focal_train/t 派生，无跨文件 join）。
- **下一步**：01 改读 sidecar split + 重跑 normalization。

### 4.7.2d 步骤2：normalization 改读 sidecar（代码已改，待 Hao 重跑）
- `input_pipeline.py`：新增 `load_episode_split()`（读 `episodes_v2.parquet` → sample_id→split；缺失返回 {}）。
- `01_build_normalization_stats.py`：split 来源优先级改为 **episodes_v2.parquet（sample_id→split，权威）→ pass_split.parquet（pass_id，legacy）→ md5 hash**；`split_key` 驱动读哪列（sidecar 用 sample_id，legacy 用 pass_id）。
- **预期重跑结果**：split 计数 == 14 的输出（train 1,472,064 / val 186,145 / test 338,363）；**vocab 不变**（268/123/278/2184，vocab 从全 split 建，与 split 变化无关 → 编码器不用重建）；仅连续特征 z-score μ/σ 在修正后的 train 上变化。
- 待 Hao 跑 `python scripts/train/01_build_normalization_stats.py` 回传，核对 split 计数 + vocab 不变。

### 4.7.2d 步骤2 结果（Windows，2026-05-22）✅ normalization 重算干净
- 用 episodes_v2 split（1,996,572 行全覆盖）。split: train=1,472,064 / val=186,145 / test=338,363 —— **与 14 输出逐字一致** ✓（normalization 现在在修正后无泄露的 train 上算）。
- **vocab 不变**：track_id 268 / signal 123 / route 278 / train 2184 ✓（编码器无需重建）。continuous features 39 不变。文件 488 row groups（~4,091 行/组）。
- **✅ episode 修复（步骤 1+2）完成**：episode 时间局部（max 6.9h）、零泄露、零跨-gap 假转移、normalization 在干净 train 上重算、vocab 不变。
- **下一步**：回到 loader（4.7.2d 本体）。loader 从 `episodes_v2.parquet` 按 sample_id 读 pass_id/position/is_last 配 (s,a,r,s',done)；流式效率仍要解（文件是 6-shard 交错，path 甲重排 vs 乙 索引+LRU 待定）。先刷新 `4_7_2d_loader_design_DRAFT.md`（纳入 sidecar + 交错事实）再实现。

### 4.7.2d 步骤3：Hao 选路径甲（canonical 重排）+ 草案 v2（2026-05-22）
- **Hao 选 path 甲**：把 snapshots 重排成 canonical 文件，loader 极简。
- `scripts/mdp/15_resort_snapshots_canonical.py`（新）：用 sidecar 覆盖 4 个旧 episode 列 + 加 split 列 → 按 (episode_idx, position) 重排 → **写新文件 `snapshots_v2.canonical.parquet`（不覆盖原文件）**。state/reward/sample_id 原样 carried。自验证：行数不变、顺序 (episode_idx,position) 单调、每 episode 从 position 0 起、sample_id 唯一全覆盖。**Hao 跑+验证后改名为 snapshots_v2.parquet（原文件留备份）。**
- 重排后好处：每个 episode 行连续按 position 升序 → 流式顺序读即得 (s,s')（s'=下一行，2-行缓存）、消除"文件旧列 vs sidecar"隐患。
- `docs/4_7_2d_loader_design_DRAFT.md` **刷成 v2**（v1 作废）：StreamingTransitionDataset（超块顺序流 + 块洗牌 buffer + position+1 配对 + carryover + worker 分片）、块级近似分层（stratum sidecar + 1/√freq 过采样）、模块落点 + smoke 计划。
- **待 Hao**：跑 15 验证 → 改名；之后实现流式 loader + stratum 标签 + smoke。

### 4.7.2d 步骤3 修订：15 第一版 OOM（整表 sort）→ 改流式外排（2026-05-22）
- **现象**：Hao 跑 15 第一版崩溃，上传的是 `java_error_in_pycharm_*.log`（PyCharm JVM OOM：`malloc failed ... system out of physical RAM`）——**不是 Python traceback**，是同机 Python 进程把 31GB RAM 吃光、连带 PyCharm 被 OS 杀。
- **根因**：第一版 `pq.read_table(整个 573MB)` + `Table.sort_by`。snapshots 嵌套列多，解码后膨胀十几 GB，sort 再复制一份 → 超 31GB。机器是本地 Ryzen 9 9955HX / **31GB RAM**（之前误以为 RAM 充裕）。记入 TOOL_TRAPS §14。
- **修复**：15 改成**内存有界流式 bucket 外排**：Pass1 逐 row group 换列+按 episode_idx 分桶到临时文件；Pass2 按桶顺序读回、桶内 sort_by、追加写最终文件。峰值几百 MB。临时文件在 `outputs/snapshots/_resort_tmp/`（本地 4TB 够）。`--buckets` 可调（默认 64）。
- **建议**：用独立 PowerShell 终端跑（别在 PyCharm 内置运行器抢内存）。
- **待 Hao**：重跑 15 → 验证（行数 1,996,572、episodes≈80,210、顺序单调）→ 改名为 snapshots_v2.parquet。

### 4.7.2d 步骤3 结果（Windows，2026-05-22）✅ canonical 重排成功
- 流式 bucket 外排（64 桶）跑通，内存稳。写出 **1,996,572 行**；验证：(episode_idx,position) 单调、每 episode 从 0 起、sample_id 唯一全覆盖、**episodes=80,210**（与 14 一致）。
- 产物 `snapshots_v2.canonical.parquet`，**待 Hao 改名为 snapshots_v2.parquet**（原文件留备份，如 `snapshots_v2.preresort.parquet`）。
- **新增 `docs/CHANGELOG.md`**：从头到尾的实现路径速览（索引/路线图），指向本 log 细节。
- **下一步**：实现流式 loader（`StreamingTransitionDataset`）+ stratum 标签 + smoke（设计见 `4_7_2d_loader_design_DRAFT.md` v2）。

### 4.7.2d 步骤4：流式 loader 实现（代码已写，待 Windows smoke）
- `src/railrl/algorithms/transitions.py`：新增 **`StreamingTransitionDataset(IterableDataset)`**（module-level，可 pickle → Windows spawn 的 num_workers>0 安全）。
  - 超块=连续 `block_groups`(默认8) 个 row group；每 epoch 打乱超块顺序 + 块内打乱转移；每块一次 `read_row_groups` 解码；行编码按需+块内缓存。
  - 转移：canonical 顺序下 s'=行 i+1（episode_idx 相等）、done=is_last；split 过滤丢非目标 split 整 episode；块边界非终止行丢 1 条/块（可忽略）。
  - worker：每 worker 取 超块[wid::nw]、各开 pyarrow 句柄、lazy load stats（不 pickle 句柄/stats）。
  - **设计修订**：最初写成"工厂函数内定义类"——局部类不可 pickle，Windows spawn 会崩；改为 **module-level 类 + 顶部 `import torch`**（py_compile 不执行 import，沙盒仍可编译）。保留旧 `TransitionDataset` 作正确性对照。
- `scripts/train/10_smoke_streaming.py`：验证 (A) 正确性=流式转移集合 vs 文件直推 ground-truth（仅差块边界极少条）、(B) 吞吐 transitions/s（旧 ~16/s）、(C) num_workers=2 与单进程集合一致。
- ⚠️ 沙盒 `/sessions` 100% 满 → bash `ast.parse` 对刚写的 transitions.py 报假 SyntaxError（§11 截断视图）；Read 工具看真文件完整。Windows 端正常。
- **分层采样（spec §4.4）仍待做**（下一子步：`16_build_stratum_labels.py` + 块级近似分层）。先验证流式 loader 正确+快，再加分层。
- **待 Hao**：Windows 跑 `python scripts/train/10_smoke_streaming.py`，回传 A/B/C 结果。

### 4.7.2d 步骤4 smoke 结果（Windows，2026-05-22）+ 修订
- **[A] 正确性 PASS**：流式转移 186,145 == ground-truth，0 extra / 0 missing。**[C] worker PASS**：num_workers=2 与单进程集合一致。流式逻辑正确。
- **[B] 160 transitions/s（单进程）** —— 关键认知：**瓶颈从 parquet 解码移到了特征编码**（`encode_snapshot`+`to_heterodata` ≈5ms/行，CPU-bound）。流式确实解决了 I/O（比旧 16/s 快 10×，即便单进程）；其余靠 **num_workers 并行编码**（[C] 已证 worker 安全）。我最初的 smoke 只测了单进程 → 误判 FAIL。
- **发现 canonical 文件 row group 过大**：15 第一版每桶一个 ~31k 行的 row group（`≤超块数 8` 暴露）→ 流式超块巨大、内存/均衡差。
- **两处修订**：
  1. `15`：`write_table(bt, row_group_size=5000)` → canonical 文件 ~400 个 5000 行组；并改 `keep_cols` 排除 `NEW_COLS`（含 split）→ **幂等**，可在已含 split 的文件上重跑不产生重复列。
  2. `transitions.py`：`StreamingTransitionDataset` 默认 `block_groups` 8→**2**（超块 ~10k 行，编码缓存 ~0.4GB/worker，多 worker 内存有界）。
  3. `10_smoke_streaming.py`：[B] 改测 num_workers=0 **和** =8（真训练用多 worker；warmup 跳过 spawn 启动开销）。
- **重跑顺序（Windows）**：`python scripts/mdp/15_resort_snapshots_canonical.py` →（验证）改名 canonical→snapshots_v2.parquet（覆盖旧 64-组版；preresort 备份保留）→ `python scripts/train/10_smoke_streaming.py`。
- **吞吐预期**：单进程 ~160/s；num_workers=8 在 32 核机上约 ~1000-1300/s（够 Stage 5 50k sanity；全量 Stage 6 若不够再加 worker 或优化 encode_snapshot）。
- ⚠️ §11 复发：沙盒 `/sessions` 满 → bash 对刚写/改的 .py 报假 SyntaxError（13/14/15/transitions/smoke 都中招）；Read 工具看真文件均完整。

### 4.7.2d 步骤4 重跑结果（Windows）✅ A/B/C 全 PASS
- [A] 正确性 PASS：流式 186,126 vs ground-truth 186,145，0 extra / **19 missing 全是 done=0**（块边界丢，0.01%，设计内）。
- [B] PASS：单进程 213/s → **num_workers=8 → 930/s**（~4.4×；其余受 collated batch 跨进程 IPC 限制，可后续调）。930/s = ~3.6 batch/s，够 Stage 5 50k sanity。
- [C] PASS：num_workers=2 与单进程集合一致。
- canonical 文件重排后 row group 已是 ~5000 行（216 超块/block_groups=2）。

### 4.7.2d 步骤5：分层采样（spec §4.4，代码已写，待 Windows）
- `scripts/mdp/16_build_stratum_labels.py`：按 state_special_flags 打 stratum（优先级 late>advance>call_on>platform_dev>priority_compete>unusual_id>trivial）→ `stratum_labels.parquet`(sample_id→stratum) + `stratum_weights.json`(train split 频数 + 权重 1/sqrt(freq))。
- `transitions.py`：`StreamingTransitionDataset` 加 `stratified=True`——块内按 1/sqrt(freq) 权重**有放回抽样**（≈ WeightedRandomSampler，块级近似），稀有 stratum 过采样；lazy load labels+weights（per worker）。
- `10_smoke_streaming.py`：加 [D] 分层检查（跑 stratified 200 batch，打印自然 vs 分层后 stratum 占比，trivial 占比应明显下降）。
- **待 Hao**：跑 `python scripts/mdp/16_build_stratum_labels.py`（看 stratum 分布，trivial 应 ~85%）→ 再跑 `python scripts/train/10_smoke_streaming.py`（[D] 应 PASS：trivial 占比下降）。
- 完成后 **4.7.2d 全部结束**，进 Stage 5（50k sanity）。

### 4.7.2d 步骤5 smoke 结果（Windows）+ 两个数据发现
- **[A] PASS / [B] PASS（num_workers=8 → 1,563/s！）/ [C] PASS**。吞吐这次更高（1563/s，机器更空闲），全量 Stage 6 也够了。
- **[D] FAIL —— 是测试判据写错，不是 loader bug**：我假设 trivial 是多数（spec §4.4 说 ~85%）、应被压低；但实测 trivial 只占 5.9%。分层采样其实**正确生效**（主导 platform_dev 67.45%→46.56% 降、稀有全升：call_on 3.8→9.7 / priority_compete 2.5→5.6 / unusual_id 0.25→2.4 / trivial 5.9→10.7）。已把判据改成「主导 stratum 降 + 最稀有升」。
- **🔴 数据发现 1：实际 stratum 分布与 spec §4.4 假设完全不同**。spec 设想 trivial ~85%，实测：
  `platform_dev 67.45% / advance 20.16% / trivial 5.90% / call_on 3.79% / priority_compete 2.45% / unusual_id 0.25% / late_train 0.00%`。
  即 ~94% 决策带某个特殊 flag、platform_dev 才是主导。含义：「trivial 淹梯度」前提不成立；但分层仍合理（且更重要——否则 platform_dev 主导）。稀有但关键的 call_on/priority_compete/unusual_id 已被有效提升。
- **🔴 数据发现 2：late_train = 0.00%（f_late_train>0 从不触发）**——可疑。f_late_train 喂模型 special_flags（一维）+ late_train stratum。若恒 0：该特征维死、late stratum 空。**疑似 f_late_train / scheduled_delta_seconds 在 snapshot 里没填**（与已知 TRUST/scheduled 稀疏一致，但 0.00% 太干净）。不阻塞 sanity（死特征维无害、空 stratum 无害），但「晚点」是项目关心的操作类型 → **值得查**。
- **结论**：loader + 分层**机械上完工且正确**；[D] 重跑应 PASS。两个数据发现待 Hao 定：是否在进 Stage 5 前查 late_train（及确认 platform_dev 67% 是真实而非 flag 过宽）。

### 4.7.2d 步骤5 重跑（[D] 修复后）+ [B] 判据修正（Windows）
- **[A] PASS / [C] PASS / [D] PASS**（主导 platform_dev 67.4%→46.6% 降、最稀有 unusual_id 0.25%→2.40% 升 → 分层生效）。
- **[B] 吞吐随机器负载波动**：三次跑 930 / 1,563 / **699** transitions/s（loader 代码没变、A/C/D 每次完全一致，只有 B 变）——是**并发跑其他大程序的 CPU 抢占**，非回归。已把 [B] 判据从「绝对 >800/s」改成「**多 worker 是否并行提速**（r8 > r0×1.3）」——绝对吞吐只作参考。185/s 单核 → 699/s 8worker 仍是 ~3.8× 并行有效。
- **Hao 选「先查 late_train」**：新增 `scripts/mdp/17_diagnose_late_train.py`（查 f_late_train 是全 null/还是从不>0；+ focal train 的 scheduled_delta_s 覆盖）。**待 Hao 跑 17 回传** → 判 flag bug vs 上游稀疏 → 决定修否 → 收尾 4.7.2d 进 Stage 5。任务 #8 跟踪。

### 🔴 数据 bug 确认（17 诊断，2026-05-22）：scheduled_delta_s / f_late_train（lateness 坏）
- **17 结果**：f_late_train 0% null 但 **min=max=mean=0**（恒 0）；focal `scheduled_delta_s` 0% null、**94.83% >0、min=0、max=23,839,604s(≈276 天)、mean≈45h**（从不为负）。
- **根因**（读 state_history.py:530 + special_flags.py:116 确认）：
  - `scheduled_delta_s(train_id,t) = sched[bisect_right(times,t)][0] − t` = "到该 headcode **下一个未来**排程事件的秒数" → **恒 ≥0**；且 headcode 跨天复用 → 排程表跨整个数据集 → "下一个未来事件"常是**另一天的同名车** → 巨大（276 天）。
  - `f_late_train` 期望 `scheduled_delta_s = gbtt−t、负=晚点`，触发条件 `< −60`。生产者恒非负 → flag **永不触发** → late_train 恒 0。**生产者/消费者语义不一致 + 取了错的（远）occurrence**（与 pass bug 同源：headcode 复用需时间局部匹配）。
  - 对比：`planned_platform` 用 nearest（idx-1 & idx），正确；`scheduled_delta_s` 只用 bisect_right/next，错。
- **影响**：f_late_train 死维（恒 0）；scheduled_delta_s 节点特征是垃圾（巨值，z-score+clip 后近噪声）→ **模型几乎拿不到"晚点"信号**（项目关心的操作类型）。**不阻塞 Stage 5 sanity 的机制验证**（单特征坏不影响 loss 是否下降）。
- **修复**：`scheduled_delta_s` 改成对 **nearest occurrence（窗口内）** 算**带符号** delta（负=晚点），与 f_late_train 对齐。⚠️ 是 Stage 3 烤进 snapshot 的特征 → 落实需外科 patch（只重算 2 字段重写文件）或重建。**lateness 的精确定义需 Hao（领域）确认**（拿哪个排程点？下一停站 gbtt？）。
- **待 Hao 定**：(A) 先修代码+进 Stage 5 sanity（当前数据，Stage 6 前再落实重建）；(B) 修代码+现在外科 patch 两字段；(C) 修代码+全量重建。

### Hao 决策：早修（不将错就错）+ 用 timetable_variation 而非 gbtt（2026-05-22）
- **Hao：以不妥协原则，早修——不在已知坏的 lateness 上跑 sanity；反正要改就趁早，避免重建两次。** 采纳。
- **Hao 指出应用 `timetable_variation`/`variation_status`/`planned_timestamp`/`actual_timestamp`（已多次提及），不是我提的 gbtt−t。** 核对：
  - 这些字段**已记录**：spec 01 §45（Movements 30 列含 timetable_variation）；load_movements() 读**整个 CSV**（字段都在，只是没用）。leak 规则 spec 01 §13.1：scheduled(gbtt/planned) state 允许；realized(actual/variation) at **t'>t** 禁；but **actual_timestamp ≤ t（已发生、t 时可知）允许**。`timetable_variation` **不在 BANNED_STATE_FIELDS**（banned 的是 delay_change_seconds/arr_delay_future 等未来窗口 reward 中间量）。
  - **差距**：state 侧 lateness 特征（scheduled_delta_s/f_late_train）被写成 gbtt−t（且 buggy），而非 timetable_variation；reward 侧已用 timetable_variation。我应先查 spec 可用字段，是我的疏忽。
- **修复范围 = 外科 patch（不全量重建）**：scheduled_delta_s/f_late_train 不影响 episode/order/reward/sample_id → 只重算这两个字段、重写文件保序、再重跑 01(norm)+16(stratum)，loader/canonical/episode 全不动。
- **拟定义（leak-safe，待 Hao 确认精确形式）**：current_lateness = 该 train **actual_timestamp ≤ t 的最近一条** Movements 记录的 `timetable_variation`（TRUST 实测延误，t 时可知）；f_late_train 在晚点超阈值时触发。是 **spec 02 状态特征的契约细化**，需 sign-off。待确认：用 numeric timetable_variation（确认符号约定）/ 是否加 variation_status 类别 / 阈值 / 字段命名（保留 scheduled_delta_s 改义 vs 新名）。

### 数据核对（Movements 真实值，2026-05-22）+ Hao 确认定义
- 实测 Movements：`variation_status ∈ {LATE, EARLY, ON TIME, OFF ROUTE}`（**4 值**，不止 3）；`timetable_variation` 是**整数分钟**（abs，0..69，e.g. actual 15:21−planned 15:19=2→`var=2,LATE`）。
- **Hao 锁定定义**：`current_lateness_s = var_min×60 × sign`，sign=+1(LATE)/−1(EARLY)/0(ON TIME, OFF ROUTE)；取该 headcode **t−W≤actual_ts≤t** 的最近一条（W=6h，避开 reused headcode 的上一日 run）；**positive=late**；f_late_train 在 **≥60s（≥1min）晚点** 时触发返回秒数；OFF ROUTE→0；variation_status 仅用于定符号→**不新增模型输入、编码器/vocab 不变**。

### ✅ lateness fix 实现（代码改完，待 Windows patch）
- `state_history.py`：`MovementsLookup` 加 `train_to_lateness`（headcode→(actual_ns 排序, signed_var_s)）+ `current_lateness_s(headcode,t,window)`（leak-safe，actual≤t & 在窗口内，取最近一条 signed variation；无→0）。`scheduled_delta_s` **改为转调 current_lateness_s**（保留方法名/签名 → state.py/schema 不动；旧 gbtt-next 逻辑废弃）。
- `special_flags.py`：`f_late_train` 改新约定（positive=late，≥60 触发返回秒数）。
- `scripts/mdp/18_patch_lateness.py`：用修正后的 MovementsLookup 重算 `state_nodes_train[].scheduled_delta_s`（所有节点）+ `state_special_flags.f_late_train`（focal），**保序重写** → `snapshots_v2.lateness.parquet`，其余列/episode/reward/sample_id 不动。内存有界（逐 row group）。
- **修复范围确认**：仅 2 个派生字段变；不动 episode/order/reward/sample_id/canonical；故只需重跑 01(norm，scheduled_delta_s z-score 变) + 16(stratum，late_train 现非空) + smoke。**无需 05b 全量重建**。
- **⚠️ spec 契约细化**：scheduled_delta_s 语义由「gbtt−t」改为「realized timetable_variation ≤ t」。应在 spec 02 状态 schema + f_late_train 文档加 changelog（待办，附 sign-off）。
- **Windows 运行序**：`18_patch_lateness.py` →（验证 focal 晚点占比>0、量级合理）改名 snapshots_v2.lateness→snapshots_v2.parquet（留备份）→ `01_build_normalization_stats` → `16_build_stratum_labels` → `10_smoke_streaming`（[D] late_train 现应非空）。任务 #8。

### ✅ lateness fix 跑通 + 🔴→✅ platform_dev 过宽 bug（Hao 在另一对话修，2026-05-22）
- **lateness（18）跑通**：focal lateness >0(晚点)21.28% / <0(早到)11.91% / ==0 66.81%；非零量级 min/median/max = 60/180/**42720s(11.9h 极端值)** —— 中位 3min 合理、远小于旧 23.8M；f_late_train>0 0→**21.28%**。01 重算（vocab 不变 268/123/278/2184）。
- **🔴 platform_dev 过宽 bug 确认+修（印证任务 #8 的怀疑）**：
  - **诊断（`19_diagnose_platform_dev.py`）**：f_platform_dev 触发 **83.2%**，其中 **99.2% 是 degenerate(all-None) ——候选路线 end_platform_id 全 None**；genuine 仅 0.7%。route 节点仅 **27.9%** 有 end_platform_id（很多 route 本就不在站台终结：through/depot）。
  - **根因**：空生成器陷阱 —— 旧 `not any(p==planned for p in cands if p is not None)`，当所有候选 end_platform 为 None 时 filtered 生成器为空 → `any()=False` → `not`=**True** → 误触发。
  - **修（`special_flags.f_platform_dev`）**：planned_platform None 或**无任一已知候选 end_platform** → False（缺数据不归因，同 reward 覆盖率的保守原则）；已知平台才比对。→ **0.7%**（与 spec §4.4 ~1.5% 吻合）。
  - **patch（`20_patch_platform_dev.py`）**：用修正函数+snapshot 自身节点（focal planned_platform + 候选 end_platform_id）重算 f_platform_dev，保序重写。f_platform_dev 83.2%→**0.7%**。仅改这一 binary flag → **不重算 normalization**（flag 不进 z-score）；重跑 16+smoke。
- **优先级级联自洽**：platform_dev(prio 3) 从 51% 掉到 0.15% → 原被它遮蔽的行正确下沉到 priority_compete(2.16→15.96)+trivial(5.19→42.63)，增量和 ≈ 51 完美对账 → 两次 patch 均无附带损坏。
- **最终 stratum 分布（健康，spec §4.4 意图达成）**：late_train 21.28% / advance 16.34% / call_on 3.40% / **platform_dev 0.15%** / priority_compete 15.96% / unusual_id 0.24% / **trivial 42.63%（多数=例行）**。
- **smoke A/B/C/D 全 PASS**（1,132/s @8worker；[D] 主导 trivial 44.2→33.1 降、最稀有 platform_dev 0.15→1.08 升）。
- **已知局限（记入论文效度威胁）**：end_platform_id 仅 28% 路线有（含合理的 through/depot 不终于站台）→ platform_dev 只在已映射处可判，保守不过报（同 approach_distance 47.8%/delay_change 6.4% 一类）。
- **教训**：`any(f(x) for x in xs if cond)` 在 filtered 为空时返 False → `not any(...)` 误为 True。任何"无反例即成立"的 flag 要先判"有没有可比的数据"。

### ✅✅ Stage 4.7.2d 全部完成（2026-05-22）
episode 重分段 + canonical 重排 + 流式 loader（A/B/C PASS）+ 块级近似分层（D PASS）+ lateness 修复 + platform_dev 修复。数据干净、normalization/stratum 当前、loader 验证通过。**→ 进 Stage 5（50k sanity 训练，spec 04 §11）。**

---

## Stage 5 — 50k sanity 训练（2026-05-22，trainer 已接线，待 Windows 跑）

### trainer 接入流式 loader + 分层 + §11 gates
- `scripts/train/09_train.py` 重写：`--smoke`（旧 map-style 小 Subset，回归检查）保留；`--sanity`/full 改用 **`StreamingTransitionDataset`**（train: stratified=True；val: plain）。`run_phase` 流式化（每 epoch `set_epoch`+新 DataLoader(batch_size=None)，按 `batches_per_epoch` 截断；epoch_base 让 A/B/C 的 shuffle 不重复）。
- `--sanity`：RTX-5070 默认 **batch 96 / num_workers 8 / ~50k 行每 epoch**（`50000//batch`，可 `--sanity-batches` 调），跑满 3 阶段（5+15+20ep），打印 **spec §11 gates**（A: route≥.50 & time≥.35 & loss finite；B: Q_top1≥.55 & |Q|<100；C: Q_top1≥.65）+ 末尾 gate 汇总（sanity 只打印不硬中止）。
- `trainer.evaluate` 增 `time_acc` + `q_absmax`（喂 A 的 time gate、B 的 |Q| gate）。
- 不变量：CQL α=5/γ=0.95/3阶段5+15+20/AdamW lr3e-4/warmup→cosine/grad clip1.0/target τ=0.005。

### Windows 分级运行（先快验证再跑满，省得 1h 跑到一半崩）
1. `python scripts/train/09_train.py --smoke` —— 验证重写没破坏基本 loop（map-style，秒级，CPU）。
2. `python scripts/train/09_train.py --sanity --sanity-batches 30` —— 验证**流式+分层训练路径**在 RTX 5070 上端到端能跑（几分钟；欠拟合 → gates 多半 FAIL，正常，只看不崩/不 NaN/不 OOM）。若 8GB OOM：`--batch-size 64`。
3. `python scripts/train/09_train.py --sanity` —— 真 sanity（~1h），看 §11 gate：**关键是 loss 下降 + route/time/action acc 随 epoch 上升、无 NaN、|Q| 不爆**（欠 50k 子集 + 短 epoch 未必满 gate 阈值，看趋势）。
- 待 Hao 回传三步输出。任务 #9。

### Step 1 (--smoke) 结果 + |Q| gate 指标 bug 修（2026-05-22）
- smoke loop **跑通**：A/B/C 三阶段、损失有限（L_route 9.6→5.6 / L_TD~0.18 / L_cons~1.6）、action_acc 0.30→0.67、ckpt+log 落盘。**重写没破坏 loop。**
- 两个 gate FAIL：(1) Phase A route_acc 0.486<0.50 —— 64 样本×4 batch 的 smoke 噪声（37 个 set 行），无意义、真 sanity 会涨；(2) **Phase B |Q|=1e9** —— **是我 evaluate 的指标 bug**：Q 网络对非法候选动作填哨兵 **−1e9**（spec §6.2），我 `out["Q"].abs().max()` 把哨兵也算进去 → 报 1e9。真实有效 Q 正常。**重要**：不修则真 sanity 的 Phase B |Q| gate 会一直假 FAIL。
- **修**：`trainer.evaluate` 的 q_absmax 改为 **`out["Q"][out["Q"]>-1e8].abs().max()`**（排除 −1e9 掩码哨兵）。
- 下一步：Hao（可选重跑 smoke 确认 |Q| 正常）→ step 2 `--sanity --sanity-batches 30`（验流式训练路径在 RTX 5070 跑通）→ step 3 真 sanity。

### Step 2 (--sanity --sanity-batches 30) 结果 + 🔴 HGT pooling batch bug 修
- **训练信号好**（即便 30 batch/epoch）：Phase A L_route 8.2→5.1↓、route_acc 0.56→0.58（✓≥.50）、time_acc ~.19-.26（<.35 gate，欠拟合+time 是弱 aux）；Phase B action_acc 0.65→0.73（✓≥.55）、L_cons 1.19→0.78↓、**|Q|max 0.2→2.6（有界、不爆；|Q| 指标修生效，不再 1e9）**。机制正确、在学。
- **🔴 崩在 Phase B ep7**：`hgt.py:171 torch.stack` 报 `[96,128] vs [95,128]`。**根因**：HGT 池化对每个节点类型用**该类型自己的 `b.max()+1`** 作 scatter dim_size；当 batch 里**最后一个 graph 没有该类型节点**（如末尾是稀疏/纯 wait snapshot 缺 route/signal 节点）→ 该类型 pooled 少一行 → stack/fusion 尺寸不齐。batch 组成相关，故 330 batch 后才撞上。
- **修（hgt.py）**：用**真 batch 大小** B=`data.num_graphs`（缺则各类型 batch 向量 max+1 取最大）对**所有**类型 `scatter(dim_size=B)` → 每类型恒 [B,128]（空 graph→0 行）。与 model.py 的 `_num_graphs`/`gather_focal`（已预分配 num_graphs 行，robust）一致。核查 model.py 的 3 个 gather 无同类 bug。
- **教训**：PyG 池化/scatter 永远显式传 `dim_size=num_graphs`，别用 per-type `batch.max()+1`（末尾空 graph 会截断）。
- 待 Hao 重跑 step 2（应跑完 3 阶段不崩、|Q| 有界）→ step 3 真 sanity。

### Step 2 重跑（pooling 修后）✅ 全 40 epoch 跑通，强学习信号（2026-05-22）
- HGT pooling 修生效：**40 epoch（A5+B15+C20）全跑完，无崩、无 NaN、|Q| 始终有界**。
- 即便 30 batch/epoch（~2,880 样本，远少于 50k）已强学习：route_acc 0.56→**0.649**、action_acc（Q top1）0.48→0.78→**0.858**、time_acc 0.19→0.31；损失全降（L_route 8.2→2.6 / L_cons 1.19→0.54 / L_CQL 4.9→3.1 / L_total 8.0→4.7）；|Q|max 0.2→11.3（远 <100，健康增长不爆）。
- gate（欠拟合小跑）：A route✓(.58)/loss✓、time✗(.19,但 C 升到 .31)；B Q_top1✓(.78)/|Q|✓(7.7)；**C Q_top1✓✓(.858)**。
- **待观察（真 sanity）**：Phase A time_acc<0.35（弱 aux 头，5 桶均衡 chance .20，现 .31 已 >chance 在学）；50k 全跑应清 0.35，否则查 time head/labels。Phase B route/time 冻结不变=设计内（B 冻 encoder+aux）。
- **结论：框架健全，放心跑真 sanity** `python scripts/train/09_train.py --sanity`（~1h），看 5+15+20 epoch 的趋势 + §11 gate。

### ✅✅✅ Stage 5 真 sanity 全 §11 gate PASS（2026-05-22，50k/epoch，RTX 5070）
- **Phase A**：route_acc **0.728**(≥.50✓) / time_acc **0.408**(≥.35✓——step2 担心的现清了) / L_route 3.06→0.30(−90%✓) / L_time 1.36→0.99(✓) / 无 NaN。
- **Phase B**：Q_top1 **0.867**(≥.55✓) / |Q| 6.8→28(<100✓有界) / L_cons 0.67→0.19(✓)。route/time 冻结=设计内。
- **Phase C**：Q_top1 **0.946**(≥.65✓✓✓) / route_acc 0.795(↑) / time_acc 0.509(↑) / L_CQL 2.22→0.75 / L_TD→0.34 / L_route→0.40 / L_cons→0.08 / |Q|~28 有界 / **无灾难性遗忘（aux 头在 C 反升）**。
- **全部损失下降、全部精度上升、|Q| 有界、无 NaN** —— **整个框架（数据+episode+lateness+platform_dev 修复+流式 loader+分层+model+CQL 3 阶段）验证健全**。
- **诚实解读**：action_acc 0.946 是**模仿**精度（Q argmax=信号员实选，候选内）；数据 FCFS 规则性高（Kendall τ≈0.998）+ 动作集小（~2.7 候选+wait）→ 高模仿精度本就预期，证明"学会复现信号员"。**"是否优于信号员"是 Stage 8 反事实评估**，非此处。sanity 是验证机制，不是最终战果。
- **→ Stage 6（全量 3 seeds 42/43/44，spec 04 §10），跑在 A100。**

## Stage 5 完成 ✅（2026-05-22）。下一步 Stage 6。

---

## Stage 6 — 全量 3-seed CQL（2026-05-22，准备中）

### Hao 选：先精简 ckpt + 跑 seed 42 验证，再 43/44
- **checkpoint 精简（09_train.py，spec §8.5/§8.4）**：去掉每-epoch 存盘；改为 **phase_A/B/C_end.pt + best.pt（按 val_action_acc，Phase C 跟踪）+ final.pt** ≈ ~120MB/seed（原 ~1GB/seed），适配服务器紧盘。`run_phase(track_best, best_state)`；末尾打印 best + ckpt 清单。
- **全量模式**：`09_train.py`（不带 --sanity）→ batches/epoch = train_rows//batch ≈ 1,472,064//256 ≈ 5,750；total_steps=40×5750=230k；warmup 1000。
- **吞吐**：encode-bound（~1500 trans/s @8worker）→ 全量 ~11h/seed，与 GPU 无关（A100 不必更快，除非 host CPU 核更多）。batch 256（spec）需 >8GB → A100。`--num-workers` 调高（按服务器核数）。
- **需拷到 A100 服务器的文件**：`snapshots_v2.parquet`（canonical 含 episode/split/flags）、`normalization_stats.json`、`stratum_labels.parquet`+`stratum_weights.json`、`time_labels_v2.parquet`；代码 git pull。
- **运行序**：(1) 廉价预检 `--seed 42 --num-workers 16 --max-batches 20`（验全量模式+batch256 在 A100 不 OOM+测 sec/batch 估总时）；(2) 真 seed42 全量 `--seed 42 --out outputs/train/cql_seed42 --num-workers 16`（~11h，验全规模 gate+盘）；(3) 过了再 43/44。任务 #10。

### 服务器（HPC sapphire）运行环境 + fd-sharing fix（2026-05-22）
- **服务器命令前缀**（所有 server 命令前加）：
  `cd /rds/homes/h/hxn886/ondemand/RailRL_v2 && env PYTHONUNBUFFERED=1 PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MPLBACKEND=Agg /rds/homes/h/hxn886/virtual-environments/my-virtual-env-sapphire/bin/python <script> ...`
- **🔴 num_workers=16 报错 `received 0 items of ancdata`**：PyTorch DataLoader 默认 file_descriptor 共享策略在 HPC 上超 fd 预算（多 worker 传 PyG batch）。**非代码 bug**。
- **修**：`09_train.py` 加 `torch.multiprocessing.set_sharing_strategy("file_system")`（main 开头，spawn 前）→ 用 /tmp 共享、绕开 fd 限制 → num_workers≥16 可用。（备：也可 `ulimit -n 4096`。）
- **num_workers=8 已能跑**（无需 fix），预检 `--max-batches 20` 约 **87s/epoch**——但这含 8-worker spawn + 每 epoch 50-batch val eval + 仅 20 train batch，**spawn/eval 主导、不代表稳态**。**先标定再压全量**：跑 `--max-batches 200`（+ fix 后 num_workers 16）报每-epoch 时间 → 由 (200-batch 与 20-batch 的差/180) 得稳态 sec/batch → 外推全 epoch(5750 batch)+总时，再决定启全量（避免盲启 10-40h）。

### checkpoint/resume（Hao 服务器单次仅 12h）（2026-05-22）
- **需求**：服务器单次租用 12h；单 seed 全量训练可能 >12h → 需 resume 跨会话续训。
- **实现（09_train.py）**：`--resume` 标志 + **滚动 resume checkpoint** `resume_seed{N}.pt`（每 epoch 末覆盖写、仅 1 个文件，含 model+target+optimizer+phase+epoch+gstep+best_state+log+gates，~50MB）。epoch 粒度：被杀最多丢当前 1 个 epoch（~20-30min）。
  - `run_phase`：加 `start_epoch/gstep0/resume_optim_sd`，从指定 epoch 续、恢复 optimizer 与 gstep（LR 调度连续）；每 epoch 存 resume.pt。
  - `main`：`--resume` 时读 resume.pt → 恢复 model/best/log/gates + (start_phase, start_ep, gstep, optim_sd, target_sd)；按 phase 跳过已完成、当前 phase 从 start_ep 续（含 optimizer/target 恢复）、后续 phase 全新。Phase B target：B/C 续训时从 resume.pt 恢复（已 soft-update 的），否则克隆。`check_gates` 对空 val（边界续训）no-op 防 KeyError。
  - 数据顺序可复现（shuffle 由 seed+epoch 确定）；torch RNG/dropout 续训后微异（无害）。
- **用法**：始终带 `--resume` + 固定 `--out outputs/train/cql_seed{N}`；首跑无 resume 文件→从头（打印提示），后续跑→自动续。
- **必须先测 resume**（别等真跑 12h 才发现不工作）：丢弃目录跑 `--max-batches 5 --resume`，Ctrl-C，再同命令 `--resume` → 应打印 `[resume] phase=... done_epochs...` 且 epoch 号接着涨（非重头）。

### Stage 6 训练中：泄露审计（Hao 本地复审）+ ep1 高精度分析（2026-05-22）
- Stage 6 全量启动（~17h 预计）。Hao 注意到 **ep1 val route=0.915 / time=0.653 / act=0.729**，担心泄露。
- **分析**：全量 ep1=5,750 batch≈1.47M 样本≈sanity 整个 5-epoch Phase A 的 ~11×（≈30× sanity 的 ep1 步数）→ 高精度部分来自步数多 + 任务高度可模仿（近 FCFS、planned_platform 强预测 route、候选集小 mean≈2.7）。act 0.729≈wait 占比(~73%)=未训练 Q 偏好 wait 的基线（非泄露）。**但需复验**。
- **审计三件套（本地跑）**：`06_run_leak_audit_full.py`（assert_no_leak 无 banned）+ `07_audit_snapshots.py`（数值分布）+ **新 `21_audit_leakage.py`**（可解释基线：总选第一候选/最常见 index/匹配 planned_platform 的 acc——若已近 0.9 则高精度=易任务非泄露）。
- **新 `docs/LEAK_AUDIT.md`**：泄露途径清单（直接答案 A1-5 / 时间 B1-5 / 候选 C1-2）+ 防御 + 复验 + 现状，兼做论文"效度威胁"素材。
- 待 Hao 跑 06/07/21 填 LEAK_AUDIT.md 结论。
- **复验结果（Hao 本地）**：07 PASS（242k：banned=0/center 全 track/time 干净）；21 val/test 基线（首候选 0.55-0.60、planned 预测器 0.97 但仅覆盖 12%、wait 基线 0.72）→ 模型 0.915 高出平凡基线 ~30pp=在用丰富 state、route head 位置无关无法利用候选序 → **无泄露迹象**。
- **06 修两个潜伏 bug**（之前能跑是旧文件嵌套列解码成 list；现解码成 numpy array）：(1) OOM——`pd.read_parquet(整表)` 改**逐 row group 流式**；(2) `pd.notna(嵌套 numpy array)` truth ambiguous——改 `to_pylist()` 取**行 dict**、`_row_to_snapshot` 用 `is not None`。(3) 还有 center 列名 bug——文件存 `state_center`，leak_audit Check1 读 `center`→别名修复。
- **✅ 06 `--sample 100000` ALL PASS**（pct_passed=100）。**→ 泄露审计 06+07+21 全过，无泄露，结论锁定（LEAK_AUDIT.md）。** 教训记 TOOL_TRAPS §15(HPC fd ancdata)/§16(pd.notna 嵌套 + center 别名)。

### 文档全量更新到当前版本（2026-05-22）
- 更新：顶部进度速查表（Stage 4/5 ✅、Stage 6 🔨）、`CHANGELOG.md`（Stage 5/6/泄露审计 + 文件清单）、`LEAK_AUDIT.md`（06 box 全过）、`TOOL_TRAPS.md`(§15/§16)、`NEW_CONVERSATION_PROMPT.md`（**全面刷新到当前状态**：项目/必读顺序/当前状态/材料清单/路线/纪律/环境/不变量/已知小项）。
- 新对话用 `docs/NEW_CONVERSATION_PROMPT.md` 开场即可全面 onboard。
- **下一步**：等 Stage 6 seed 42 跑完（~17h，可 resume），核对 §11 gate + train_log → 启 seed 43/44 → Stage 7 baselines。

### Stage 6 resume 修：torch 2.6 weights_only（2026-05-22）
- **现象**：全量训练已进 Phase B、12h 窗口结束后 `--resume` 报 `UnpicklingError: Weights only load failed ... numpy._core.multiarray.scalar`。
- **根因**：torch 2.6 `torch.load` 默认 `weights_only=True`；resume ckpt 的 `gates["A"]` "loss finite" 项是 `np.isfinite()`→numpy.bool_，被拒。`--max-batches 5` 测试时在 Phase A 中途被杀、gates 还空，故没暴露。（TOOL_TRAPS §17）
- **修**：(1) `09_train.py` resume 加载 `torch.load(..., weights_only=False)`（自己写的可信 ckpt）；(2) check_gates 的 `np.isfinite` 包 `bool(...)`，keep ckpt 干净。
- **重要：现有 resume_seed42.pt（Phase A+B 进度）完好**，修后重跑同一 `--resume` 命令即从断点续，不丢进度。

### Stage 6 — seed 42 跑完 + 聚合脚本（2026-05-24）
- **seed 42 全量训练完成**（40 ep：A5+B15+C20，resume 续训正常收尾）。逐项核对 spec 04 §11 gate（末-epoch val）：A route .933(>.50)/time .714(>.35)/L_route 降到初值 28.7%(<70%)/L_time 71.8%(<85%)；B Q-top1 .970(>.55)/|Q|max 78.4(<100)/L_cons .077(<50)；C Q-top1 **.984**(>.65)。**三 gate 全过**。损失全程↓（C：L_total .80→.42、L_CQL .568→.222、L_TD .27→.165），无 NaN。best val_action_acc=**.9846 @ C-ep17**（best.pt），final(C20)=.9839。
- **两点记录（非 gate 失败，但记下）**：(1) **|Q| 在 C 阶段涨过 100**——B 末 78.4(<100 达标)，C 末爬到 ~120–131（峰 130.9 @C-ep18，final 123.8）。spec/trainer 对 **C 阶段未设 |Q| 闸**（仅 B 有），且为 120–130 区间震荡非单调爆炸 + L_TD 低（Q 与 target 自洽）→ 多半是 CQL 抬高 in-dist Q 的正常表现；留 Stage 8 看 Q 校准。(2) **§11.3 per-stratum 不退化半条无法从 train_log 验**——trainer `check_gates` 只查整体 C Q-top1≥.65、未算分层；整体 acc B→C 升(.970→.984) 无整体遗忘，但分层(advance/call_on/platform_dev/late_train)要等 Stage 8 分层评估坐实。
- **新 `scripts/train/11_aggregate_results.py`**（= spec §12 的 05_aggregate_results；05 号已被 `05_smoke_fusion_q` 占用，故改用 11）：纯 stdlib+numpy（无 torch/pyarrow，沙盒可跑）。glob `outputs/train/cql_seed*/train_log_seed*.json` → 每 seed 取 final-epoch + best 指标 + §11 gate pass/fail → 跨 seed mean±std + seed 级 bootstrap CI（n=#seeds，弱，已注明）→ 写 `outputs/train/aggregate_cql.json` + 打印表。CLI：`--glob/--algo/--out/--expect-seeds`。final ckpt 只存权重，故指标全取自 JSON log。
- **沙盒已测**：n=1（真 seed42）+ n=3（合成 43/44，仅改末-epoch action_acc=.97/.99）→ 手工核对 mean=98.131%、std(ddof=1)=1.025% 与脚本输出**完全一致**；gate/CI/best 选取均正确。
- **🔴 待 Hao 手动删**：合成测试数据已移出 glob 路径到 `outputs/train/_SYNTHETIC_TEST_DELETE_ME/`（含假 seed43/44 log + 测试 aggregate_cql.json）——**绝不可参与真聚合**（沙盒无删除权限、Hao 拒绝授权删除，故只移未删）。
- **下一步**：Hao 删 `_SYNTHETIC_TEST_DELETE_ME` → 服务器跑 seed 43/44（同 `--resume`）→ 真聚合 `python scripts/train/11_aggregate_results.py` → Stage 7 baselines。

### Stage 7/8 准备 — 共享评估口径（eval harness，2026-05-24）
- **决策（Hao）**：Stage 7 baseline 范围 = B0 随机 / B0' FCFS / B1 BC / **IQL**（spec 04 §1 要 CQL+IQL 对照）；**先建共享评估口径**再做 baseline（baseline 单独训没法比，且评估口径顺手给 CQL 的第一个 test 数）；eval 用每 seed 的 **best.pt**；Table I 分层先做**现有 7 类优先级 stratum + overall**（TRTS/Freight 作后续，需从 state_special_flags 单独切 overlap 片）。
- **盘点**：`losses.py` 已有 CQL/IQL/BC 三套损失；但 `bc.py`/`iql.py` 从未建、`trainer.compute_loss` 只 dispatch CQL；模型缺 IQL 的 value head V(s) 与 BC 的 wait_logits 头（实现 baseline 时要加）；`scripts/eval/` 与 `src/railrl/eval/` 空（Stage 8 未开工）。
- **新 `src/railrl/eval/metrics.py`（纯 numpy，沙盒已单测）**：Tier1 整体（action top-1 **all & set-only 两种定义并列**——spec04 训练口径 vs spec05 §2.1 set-only；route/time head top-1；wait rate/recall/precision；Q-gap sanity）+ Tier2 per-stratum top-1（overall + 7 类，int→name 映射 0..6 取自 16_build_stratum_labels：0 late_train/1 advance/2 call_on/3 platform_dev/4 priority_compete/5 unusual_id/6 trivial）。合成数据手算核对全对（各 stratum n 求和=总数=完整划分）。
- **新 `scripts/eval/01_evaluate_model.py`（torch，Hao 跑）**：载 best.pt + `RailRLModel.build(stats)` → `StreamingTransitionDataset(split=test, **stratified=False**)` 扫 test → 收 chosen/Q-argmax/route/time/sample_id/Q-gap → join `stratum_labels.parquet` + time_lut → metrics → 写 `outputs/eval/{tag}_{split}_metrics.json` + 打印表。AST 过、符号核对（RailRLModel.build / NormStats.load / out 的 route_scores+time_logits / C.EVAL_DIR / bs.sample_id 全存在）。
- **关键设计点**：评估 **stratified=False**（分层只用于训练，评估要真实 test 分布）；per-stratum 在不分层预测上按 stratum 分组；两种 action top-1 并列，避免静默选错口径（.984 那个是 all 口径）。
- **待 Hao（Windows/A100）**：`python scripts/eval/01_evaluate_model.py --seed 42` → 拿 CQL seed42 的 **test** 数（overall + 7 类 + wait/Q-gap），回传核对。之后：43/44 同样 eval → 跨 seed 聚合（扩 11_aggregate 或新 eval-aggregate）→ baseline（B0/B0' eval 时直接算、BC/IQL 加头+训练）。

### CQL seed42 首个 test 结果（2026-05-24，best.pt）+ 核对
- **跑通**（n=338,329；test 应 338,363，差 **34**=流式块边界丢弃 0.01%，设计内）。`outputs/eval/cql_seed42_best_test_metrics.json`。
- **指标**：action_top1 **all=0.9897 / set-only=0.9622**；route_head 0.9520；time_head 0.6803（弱 aux，5 桶）；wait_rate model=signaller=0.727（Δ+0.0002）；**wait recall 0.99998 / precision 0.99970**（命令行印 1.000 是取整，真值非完美，~81 错分）；Q-gap mean +30.49（p5 +8.25 全正）/ frac_chosen_is_argmax 0.988。
- **per-stratum top-1（全决策口径）**：overall .9897 / late_train .9937(n=103,133) / advance .9977(49,231) / call_on .9458(11,712) / platform_dev .9355(496) / priority_compete .9776(46,335) / unusual_id .9892(278) / trivial .9919(127,144)。**n 求和=338,329 ✓ 完整划分**。
- **算术核对全过**：all-acc = recall×n_wait + set_acc×n_set = 0.98966 ✓；Q-gap 排除 41,518 行(12.3%)=纯 wait-only（对上 log 的~12%）；time 标签 97.1%（对上 τ 覆盖 97.0%）；val_action_acc_ckpt 0.98456 对上 train_log best。
- **关键解读**：(1) **test all-acc 0.9897 > val 0.9846** 非过拟合——"all"口径被 wait 多数(72.7%)主导 + test stratum 分布更有利；honest 信号是 set-only 0.962。(2) **残差错误几乎全是 route-vs-route，极少 wait-vs-act**（precision~1）。(3) 🔴 **per-stratum 模仿精度 94–99% 全线高**，远超 spec05 Table I 示意值（call_on 31–42%/platChg 38–48%）——印证任务高度可模仿（near-FCFS, τ≈0.998, 动作集 mean≈2.7）。**含义：论文价值主张须落在 Tier3 反事实改进 + XAI，不能靠难case模仿精度（已近天花板）。**
- **两个指标口径待改进**：(a) per-stratum 现为"全决策"口径→被 stratum 内 wait 多数稀释拉高，难case诚实数字需补 **per-stratum set-only / wait-act 分开**；(b) test stratum 分布异于整体（late_train 30.5% vs ~21%，test 期晚点多）→ 记入论文。
- **(a) 已实现（2026-05-24）**：`metrics.stratified_top1` 每 stratum 改出 `acc_all / acc_set / n / n_set / n_wait`（新 `_stratum_cell`）；驱动打印加 set-only 列。验证：§11 沙盒磁盘满(/sessions 100%)→bash 读到截断视图假报 SyntaxError；**Read 工具确认真文件完整**，逻辑抽到 /tmp 独立测全对（overall .75/.75；trivial acc_all .667 但 acc_set 1.0；late_train acc_set=NaN/n_set=0）。**待 Hao 重跑 `python scripts/eval/01_evaluate_model.py --seed 42`** 看各难case的 set-only 真实精度（JSON schema 变：per-stratum 由 {acc,n} → {acc_all,acc_set,n,n_set,n_wait}，无下游依赖）。
- **set-only 结果（2026-05-24 重跑，seed42 best）** acc_all → **acc_set** (n_set)：overall .9897→**.9622**(92,298) / late_train .9937→**.9739**(24,961) / advance .9977→**.9381**(1,760，96%是wait故all虚高) / call_on .9458→**.9005**(6,373) / platform_dev .9355→**.8961**(308，小样本) / priority_compete .9776→**.9355**(16,064) / unusual_id .9892→**.8846**(26，噪声) / trivial .9919→**.9760**(42,806)。n_set 求和=92,298 ✓。**解读**：剥掉 wait 后难case真实路线精度 ~89–97%（最难的 call_on/platform_dev/priority 落 ~90–94%）——仍高，"模仿近天花板"成立；6–10% 的分歧正是 Tier3 反事实要验证"是否改进"的金矿。错误几乎全集中在 set 路线选择（advance 113 错里 ~109 在 set）。

### Stage 8/10 — P2.6 仿真器可行性勘探（2026-05-24）
- **Hao 选"先做可行性勘探"而非直接全量建仿真器**。理由（我提的，Hao 采纳）：反事实无 ground truth 可验→在验不了的仿真器上立 Tier3"模型更优"头号结论=本项目最大"将错就错"风险；spike 先量化"能不能验"，de-risk 核心又不先砸 3 周。
- **读完 spec 05 §14**：仿真器 = event-driven rollout（MinHeap 事件，predict_next_event 用参数表）+ **4 参数表**（route_running_time(route×class) / platform_dwell(platform×class) / min_headway(track,249) / aspect_clear_lag(signal,100)）；**验证门 §14.6** = held-out 月 **Spearman(sim, actual) > 0.6 on delay_change AND throughput**，否则重标定/放宽 CI。
- **盘点（数据全在）**：`td_data.parquet`（11.9M 事件，密）、`movements.parquet`、`Derby_info.gap_time(s)`（route_running_time fallback，spec §6.5）、`route_to_tc_all.csv`（route/track 全集）、`snapshots/decision_rewards_v2`（录到的 outcome/delay/headway）。→ **建参数表低风险**（3 表源自密集 TD，gap_time fallback 恒可用）。
- **真正风险=验证**：`delay_change` 录得稀疏（~6%/决策，TRUST 计时点稀）→ 难独力达 delay-Spearman>0.6；但 throughput/occupancy/headway 来自**密集 TD**，可密验。
- **缓解洞察**：r_delay 对 r_total 贡献极小（4.6.5 实测 r_delay −0.0025 vs r_wait −0.218 / r_throughput +0.136）→ Tier3 比较的 reward-delta **主要由 throughput/wait/headway 驱动**（可密验），非 delay → **弱 delay 验证可接受**，前提 probe 确认 r_delay 权重确实小 + 验证主轴改 throughput/headway。
- **新 `scripts/simulator/00_feasibility_probe.py`**（纯 pandas/pyarrow，只读，Hao 跑）：A 录到前向结果覆盖（按 split+月，delay/approach/headway non-null %）/ B reward 组成（r_* 均值 + |r_delay|/|r_total|）/ C 参数源密度（Derby_info gap_time / route_to_tc 轨道数 / movements dwell / TD 按月事件数）。AST 过、config 符号（DATA_DIR/REFERENCE_DIR/CACHE_DIR/REWARDS_DIR/SIMULATOR_DIR）核对存在。
- **go/no-go 判据**：GO = 参数源密 + |r_delay|/|r_total| 小 + throughput/occupancy 可密验（→ 建仿真器、验证主轴用 occupancy/throughput、delay-Spearman 作 best-effort 并注明覆盖局限）；CAUTION/rescope = 若 r_delay 实际占比大且 delay 覆盖仍 ~6%（则头号 reward-delta 验不了，需窄化 Tier3 claim 或改用其他改进证据）。
- **待 Hao 跑 `python scripts/simulator/00_feasibility_probe.py`** 回传 → 我据数定 go/no-go + 调整后验证策略。

### 🔴 delay_change (r_delay) 覆盖 bug 调查（2026-05-24，Hao 质疑触发）
- **Hao 质疑** r_delay 只占 2.5% + Mar-Jul≈0 不合理，要求核查（"绝对是 lateness 出了问题，可能需重训，先核查后决定"）。
- **核查算法**（`data/reward_features.compute_delay_changes`）：r_delay=delay_change=`arr_delay[j]−arr_delay[j-1]`（决策后/前 TIPLOC 的 `actual−planned` 延误之差），要 ① headcode 匹配 TRUST train_id[2:6] ② 决策 t 落在该 run [t_first−W, t_last+W] ③ 前后点都 ≤W=4202s(~70min)。
- **读 Movements.csv 全量（247,310 行，未截断）**：timetable_variation 100% 有值、actual/planned 99.6%+、日期 2023-02-28~2024-04-25；**每月 ~3 pt/train、中位 consec-gap 2min、>99% gap<70min** → 70min 夹取在**所有月份（含 Mar-Jul）都可行**；**日覆盖匹配决策段**（2023-04 在 04-04~04-17 ~900/天 hours 0-23；2023-08 在 08-09~08-19 ~900/天）——**April 与 August 数据形态几乎一致**。
- **🔴 结论**：April delay 覆盖 0.17% vs August 8.99%，**同密度/同日/同小时 → 50× 差异不可能来自数据**，是**决策↔Movements join 对 Mar-Jul 系统性失败**（real bug，印证 Hao 直觉；与 episode 跨月、lateness 取错 occurrence 同一 join-bug 家族）。
- **影响**：Apr-Jul ≈ 30% 数据（605k 决策），修好覆盖或从 ~0.2%→~9%（类比 Aug）→ delay 覆盖整体可能翻倍+，r_delay 不再 2.5%。**当前 CQL 是在 r_delay 大面积缺失的奖励上训的 → 修复后需重算 reward + 很可能重训**（Hao 已预期）。
- **新 `scripts/mdp/22_diagnose_delay_coverage.py`**（只读，Hao 跑）：复刻匹配，**按月报失败原因**（no_match_headcode / no_trust_in_window / no_baseline / no_followup / out_window / bracketed）→ 定位 Mar-Jul 是 headcode 不匹配（TD vs TRUST 格式）还是时窗错位。AST 过、config 符号核对。
- **澄清两个"延误"量**：`current_lateness_s`（状态特征，最近 timetable_variation ≤t，33% 覆盖，已修）≠ `delay_change_seconds`（奖励，前后夹的"改善量"，6% 且有此 bug）。Hao 提议或可用更密的 lateness 直接定义 r_delay（=重开 4.6.5 奖励设计，待定）。
- **暂停 simulator go/no-go + Stage 7 baseline**，先核查（Hao 指示）。
- **✅ 根因确认（22 输出 + 独立验证，2026-05-24）**：失败主导 = **out_window 76.99%**（no_match_headcode 仅 1.16%→非 headcode 问题）；区分好坏月份的偏态 = 坏月（Apr-Jul）`no_baseline` 高(Apr 29%)、`no_followup`≈0（决策总落在匹配车次的点**之前**）。独立验证 Movements 全量：**45.5% 的 train_id 跨度>25 天（p95=366 天，max 397），81% 的计时点属于跨度>1 天的 train_id**。**根因 = `compute_delay_changes` 按完整 TRUST `train_id` 分组（EE=当月几号→每月复用，与 episode 跨月 bug 同源），导致每个候选车次跨大半年 → "t 落在车次时段"判据对所有候选都真、区分不开 → 按"中心最近"挑错日期的行程 → 在错行程内夹到相隔数周的两点 → >70min 被拒 → out_window。** Apr-Jul 更惨是月度簇排布使然。
- **修复方向**：像 episode 那样**按 gap（2–6h）把每个 train_id 的点切成"单次行程"**，决策匹配到正确当天行程后在行程内夹取。预期 out_window 大部分转 bracketed → **delay 覆盖从 6% 大升且各月均匀**。当前已 bracketed 的 6% 值本身没错（70min 闸挡住跨月配对）→ 是"覆盖被压垮"非"值算错"。
- **影响**：r_delay 不再 2.5% → 奖励实质改变 → **重算 decision_rewards_v2（08→09→10）+ 重并 snapshots + 重训 CQL（42/43/44）**。修复前先改 `compute_delay_changes` + 重跑 22 预览新覆盖，确认大升再投入全量重算+重训。

### ✅ 修复实现 — compute_delay_changes run-segmentation（2026-05-24，Hao 选"先修+预览"）
- **机制再确认（/tmp 复现）**：buggy 的真正失败不是"t 落在跨月 gap"，而是**同 headcode 有多个 train_id（不同 EE 日），每个都跨年 → "[t_first−W,t_last+W] 含 t" 对所有候选都成立 → "中心最近"挑错那个日的 train_id → 在错 train_id 的月度簇间夹取 → out_window**。/tmp 双-train_id 合成：决策在 tidA 当天行程，no-split→out_window，2h-split→bracketed ✅。
- **改 `reward_features.compute_delay_changes`**：建索引时按 `RUN_GAP_S=2h` 把每个 train_id 的点切成单次行程 → `by_run` + `headcode_to_runs`；匹配/夹取/Stage2 归因全改 run（trust→run）。行程仅 ~5min 窗 → 窗口判据真正能区分日期。
- **镜像进 `22_diagnose_delay_coverage.py`**（build/classify 改 run-based）+ 加 `--run-gap`（默认 7200=修复；传 1e12 复现旧 buggy 做 before/after）。
- **验证**：reward_features.py AST PASS；Grep 无残留 trust 引用；/tmp 逻辑测通过；§11（/sessions 满）对 22 报假 SyntaxError@129，Read 确认真文件完整（行 129 = `df["reason"]=reasons`）。
- **待 Hao 预览**：`python scripts/mdp/22_diagnose_delay_coverage.py`（默认=修复）看新 bracketed%，与 `--run-gap 1e12`（旧）对比。确认大升 → 全量重算 08→09→10 + 重并 snapshots + 重训 CQL 42/43/44。

### 22 预览结果（fix #1 后，2026-05-24）：fix #1 生效，但暴露 fix #2（Mar-Jul 时钟偏移）
- **fix #1 生效**：`out_window` 76.99%→**0.13%**（跨月 bracket 消失）；overall `bracketed` 6.44%→**23.87%**（3.7×）；切出 75,730 runs（从 25,357 train_id）。
- **Aug 2023 起大升**：各月 bracketed ~34–37%（原 8–14%），bracketed/no_baseline/no_followup ≈ 35/33/29 平衡 → 健康。
- **🔴 Mar-Jul 仍≈0%，换了失败原因**：Apr-Jul bracketed 0.18–0.45%、**no_baseline 94–96%**（2023-03 是 no_trust_in_window 91%）。no_baseline=决策 t 在匹配行程首点**之前** → **Mar-Jul 决策系统性早于 Movements = 第二个 bug（时钟/时区偏移），与 train_id bug 无关**。
- **加 offset 测量到 22**：classify 返回 `(reason, off_min=行程首点−t 分钟)`，按月报 median offset → ~+60min⇒BST/UTC 时区，其他幅度⇒别的采集问题。/tmp 验证 offset 语义（前30min→+30、内部→−3）。
- **待 Hao 重跑 22** 看 by-month median offset 定位 fix #2。fix #1 已是大胜（6→24%，Aug+ 翻 3–4 倍）；fix #2 修好 Mar-Jul（~30% 数据）后覆盖还会再升一截。

### ✅ fix #2 定位 — Movements 时间戳 Apr-Jul 整体 +1h（2026-05-24，结论性）
- **22 by-month median offset**：**Apr-Jul = +58.6min**（= 正常 −1.4 + 60），其余月 −1.3~−1.6min → 干净的 **+1h，仅 Apr-Jul**（非 DST 边界：BST 到 10/29，但偏移 7/31 止 → 上游采集期特异）。
- **哪一侧 = Movements**：Movements 自身 `actual` 日节律 Apr-Jul 比 Aug+/Nov-Feb **晚 ~1h**（晨峰 hr4→hr5；hr23-01 夜间偏多）；`planned_timestamp` 晨峰同样晚 1h。→ Movements 时间戳整体 +1h，TD/决策侧正常（Movements 自身节律变了说明是 Movements 而非 TD）。
- **delay 值未损**：`actual−planned` 中位 ≈0（两期一致）、`timetable_variation` 中位 2min（一致）→ **actual/planned/gbtt 一起 +1h，内部延误 cancel 不受影响**，仅绝对时钟错位（疑上游 Apr-Jul 双重 BST）。
- **🔴 blast radius（比 fix #1 大）**：Apr-Jul 决策 = Apr 88k+May 138k+Jun 175k+Jul 204k = **604,972 ≈ 41% of train split**（全在 train，因 train<2024-02-01）。受损 = 所有 Movements↔决策t 的 join：**reward delay_change + 状态 current_lateness_s/f_late_train + schedule_outlook + planned_platform/platform_dev**。**TD 派生状态（占用/aspect/event token/current_tc）不受影响**（TD 内部一致）。**关键：val(Feb24)/test(Mar-Apr24) 落在 Aug+ 正常期 → 不受 bug #2 影响 → 之前 test 评估在干净数据上、方法学有效**；受损的是 ~41% 训练数据的 Movements 派生输入 + reward。
- **修复方向**：源头修——`load_movements()` 对受影响日期范围（约 BST起~07/31，待精确定边界）Movements 时间戳 **−1h** → 重派生 Movements 依赖列（schedule_outlook/lateness/platform_dev，可能需对 Apr-Jul 部分重建 snapshots）+ 重算 reward（08→09→10）+ 重训 42/43/44。**待 Hao 定范围 + 先精确定边界（per-day offset）。**

### ✅ fix #2 边界确认 + 源头修正实现（2026-05-24）
- **边界（22 per-day offset）**：2023-03-11 正常(−1.6m)→**2023-04-04 起 +58.6m**→…→2023-07-31(+58.6m)→2023-08-10 恢复(−1.4m)。+1h 精确覆盖 **4/4–7/31 全部数据**，两端落在数据空档（3/17-4/3、8/1-8/9）。锁定窗口 **[2023-03-17, 2023-08-05)**。
- **episodes 不受影响**：当前 episodes 来自 14 的决策-gap 重分段（TD 时钟），非 Movements 区间 → bug #2 不碰 episode。blast radius = reward delay + Movements 派生状态（schedule_outlook/lateness/platform_dev）。
- **源头修实现**：`config.py` 加 `MOVEMENTS_BST_FIX_START/END/DELTA_H`；`data_io.correct_movements_bst(df)`（对窗口内行 actual/planned/gbtt −1h，delay 不变）；`load_movements()` 缓存保持 RAW、每次 load 应用修正（不双重应用）；`reward_features.compute_delay_changes` read_csv 后调用同一修正（actual_ns 对齐、delay_s 不变）。/tmp 验证：窗口行 −1h 且 delay 保持(120s/60s)、窗外不动。
- **剩余步骤（待 Hao 定 step3 做法）**：重派生 Movements 派生状态——拟新增 `23_patch_movements_state.py` 一次性重算 Apr-Jul 行的 schedule_outlook+lateness+platform_dev（复用 MovementsLookup+修正 Movements，保序重写；比全量 05b 重建安全，沿用 18/20 的 patch 模式）→ 重算 reward(08→09→10)+重并 → 重跑 01(norm)/16(stratum) → 重训 42/43/44 → 重评估。
- **🔴 写 patch 23 时发现状态 blast radius 是 6 个字段（非 3）**：读 state.py 确认所有经 `movements_lookup` 的字段在 Apr-Jul 都 +1h 受损：(1) `state_nodes_train[].planned_platform`（`lk.planned_platform`）(2) `state_nodes_train[].scheduled_delta_s`（18 已覆盖）(3) `state_schedule_outlook` 整 struct（`lk.schedule_outlook`）(4) `f_late_train`（由 #2，18）(5) `f_platform_dev`（由 #1+候选，20，但依赖先修 #1）(6) `f_trts_pressed`（用 planned_platform #1）。现有 18/20 只覆盖 #2/#4/#5；**#1 planned_platform / #3 schedule_outlook / #6 f_trts 无 patch**。→ patch 23 要复算 6 字段、跨 3 nested struct、须与 build_snapshot 完全一致，且沙盒无法测（§11+无 pyarrow）→ 风险升高。**待 Hao 定：(A) 写全面 patch 23（仅改 Apr-Jul 行、复用真实函数、带前后分布自检，Windows 验证）vs (B) 对 Apr-Jul 行走 build_snapshot 局部重建（用真实代码，零偏差，但重）。**
- **🔴 写 patch 23 时再发现：6 个字段里 `f_trts_pressed` 无法用 stored-field patch 修**。`f_trts_pressed(planned_platform, current_platform, trts_state_by_platform)` 需要 **TD 派生的 `trts_state_by_platform`（每站台 TRTS 是否按下），snapshot 没存**（build 时从 TD 算）。→ 纯 patch 只能干净修 **5/6**（planned_platform / scheduled_delta_s / schedule_outlook / f_late_train / f_platform_dev，全 Movements-only），f_trts 需 builder（TD histories）才能正确重算。
- **f_trts 残差很可能极小**：它是 8 个 flag bit 之一，只有 planned_platform 输入受 +1h 影响（current_platform 来自 TD、trts_state 来自 TD，都没坏），且 TRTS-pressed 本就少见 → 只在"TRTS 恰好按在错 vs 对的站台"时才翻转。**待 Hao 定 f_trts 处理**：(A') patch 5 字段 + f_trts 记为已知微小残差（最快，近完整）；(B) 局部/全量 build_snapshot 重建（修全 6 含 f_trts，重）。
- **Hao 选 A'（patch 5 + f_trts 留残差）**。判断依据（诚实比较）：全量重建反而**更危险**——重跑 05b 会解开 05b→14(episode)→15(canonical)→reward合并→18→20 整条链，sample_id/顺序须全重接（当初花一整个 Stage 才调对）；patch 是局部微创、只动 Apr-Jul 行、其余 59% 值不变、沿用 18/20 已验证机制。f_trts 是唯一真妥协，影响≈0，记论文局限。
- **✅ 新 `scripts/mdp/23_patch_movements_state.py`**（Hao 跑）：load_movements()（已修正）+ MovementsLookup → **仅对 decision t ∈ [3/17,8/5) 的行**重算 5 字段（planned_platform/scheduled_delta_s via lk；schedule_outlook 逐字照搬 _build_schedule_outlook 转换；f_late_train/f_platform_dev via 真实 flag 函数，用修正后 focal planned）；**f_trts + 窗外行不碰**；保序写 snapshots_v2.movstate.parquet + 前后分布自检。AST PASS；/tmp mock 逻辑测全对（planned 4/lateness 180/f_late 180/f_platdev True/outlook xform 正确/f_trts 保持）。
- **完整重算+重训跑序（Hao，Windows/服务器）**：① `23_patch_movements_state.py`→验证→改名 snapshots_v2.parquet（留备份）；② 重算 reward `08→09→10`（compute_delay_changes 现含 fix#1 train_id + fix#2 BST，全量 delay 覆盖应大升且各月均匀）；③ `01_build_normalization_stats`（lateness/schedule eta z-score 变）；④ `16_build_stratum_labels`（late_train/platform_dev 变）；**⑤ 🔴 重训前全面体检 gate `24_pre_retrain_audit.py`（全绿 + 06/07/21 leak 复审干净才往下）**；⑥ 重训 `09_train --seed 42/43/44`；⑦ 重评估 `eval/01_evaluate_model`。

### ✅ 重训前全面体检 gate — `scripts/mdp/24_pre_retrain_audit.py`（2026-05-24，Hao 要求）
- **动机**：重训代价极大，重训前必须把重算后的 snapshots 彻底体检、找全可疑点。设计原则（本项目每个 bug 的共性）：**靠"分布"和"独立信号"发现，且常按月才暴露** → 体检围绕这两点 + 按月 + 主动扫异常月。
- **5 段（各有 PASS/FAIL gate）**：A 结构不变量（行数 1,996,572 / sample_id 唯一 / canonical (episode_idx,position) 单调 / 每 episode 起于 0）；B 奖励完整性（r_total=Σ分量算术自洽；**sample_id↔label 独立一致**——set 行 outcome 非空+r_wait_raw=0、wait 行 null+−1，复刻 4.6.5 抓错位的判据；**delay 覆盖按月**——验 fix#1+#2，无月<10%、整体>20%、量级<2h；r_delay 占比应较 2.5% 升）；C 状态完整性（planned_platform∈{1..7,None}；schedule_outlook 有 eta_s；platform_dev 率~0.7%；**late_train 在 Apr-Jul 现非零**）；D **非窗口行 vs 备份不变**（patch 23 只该动 Apr-Jul → 窗外 f_late/f_platdev 与备份逐行相同）；E **跨月异常扫描**（按月打 wait_rate/n_cand/delay&approach&headway 覆盖表 + MAD>3 自动标异常月 → 找"还有没有别的 Apr-Jul"）。
- **验证**：AST PASS；/tmp 测 label-agreement（干净 1.0；注入 set-row-null-outcome 错位 →0.667 触发 FAIL，证明能抓 4.6.5 类 bug）+ MAD 扫描（正确标出 0.002 的异常月）。
- **用法**：`python scripts/mdp/24_pre_retrain_audit.py --backup outputs/snapshots/<pre-fix备份>.parquet`。全绿 + 06/07/21 + 10 smoke 干净，才重训。

### 🐛 fix #2 reward 路径 dtype bug（2026-05-24，Hao 跑 09 时崩）
- **23 跑通**（用 load_movements 的 datetime 列；输出正确：in-window 604,972、planned_platform 改 110,566、schedule_outlook 改 599,038、**f_late_train 1,835→179,715**=Apr-Jul 晚点终于被算出）。**08 跑通**（outcome used 99.53%）。
- **09 在 compute_delay_changes 崩**：`correct_movements_bst` 往 read_csv 产生的 **pyarrow-string dtype** 列掩码赋 Timestamp → `TypeError: Invalid value for dtype 'str'`。load_movements 用 parse_dates(datetime)→没事；read_csv(usecols)→string→崩。（TOOL_TRAPS §18；合成测试用 object-dtype 没复现，同 §12 dtype 教训。）
- **修**：`data_io.correct_movements_bst` 掩码减法前先 `df[c]=pd.to_datetime(df[c])` 整列转 datetime（已 datetime 时 no-op），再 `df.loc[mask,c]+=delta`。/tmp 用 string-dtype 复测：输入 string→输出 datetime64、Apr 行 −1h、delay 保持 120s、Aug/Feb 不动。**23 无需重跑**（它走 datetime 路径本就对）；**Hao 重跑 09→10 即可**继续跑序。

### ✅ Stage 6 seed 42 全量完成（2026-05-22，全 §11 gate PASS）
- resume 跨窗口续训成功，log 连续 40 epoch（A5/B15/C20），无断点异常。
- **gate 全 PASS**：A route **0.933**(≥.50)/time **0.712**(≥.35)/L_route 0.63→0.18↓；B Q-top1 **0.970**(≥.55)/|Q| 78.4(<100)；C Q-top1 **0.984**(≥.65)。
- 全损失↓（L_CQL 0.57→0.22 / L_TD→0.16 / L_cons→0.011）、精度高且稳（route ~.93 / action .95→**.984** / time ~.71）、无 NaN。比 50k sanity 更强（更多数据）。
- **⚠️ 注意 1：|Q| 在 Phase C 涨过 100**（93→130，平台 ~120-130）。spec 的 <100 是 Phase B 阈值（B 78✓）；C 只要求"有界"，130 有界不发散，且在理论 return 尺度内（γ=.95、reward±30 → |return|可达~600），L_cons 仍降、L_TD 平 → 是 Q 稳定在自然量级、**非 α 太低爆炸**。跨 seed 观察，非阻塞。
- **⚠️ 注意 2：action_acc 0.984 = 模仿精度**（Q argmax=信号员实选，候选内），证明高度复现近-FCFS 信号员；**"是否优于信号员"是 Stage 8 反事实**，且已审计无泄露。
- **下一步**：同命令启 **seed 43 / 44**（`--seed 43 --out outputs/train/cql_seed43 --resume` 等）→ 3 seed 齐后聚合 mean±std（`05_aggregate_results` 待写）→ Stage 7 baselines。

### ✅ lateness patch + 全部重跑（Windows，2026-05-22）—— 4.7.2d 完成
- `18_patch_lateness`：focal lateness >0(晚)21.28% / <0(早)11.91% / ==0 66.81%；非零 |秒| min/median/max=60/180/42720（量级合理，远小于旧 23.8M 坏值）；**f_late_train(focal)>0 = 21.28%（修前 0.0000）** → 死特征复活。1,996,572 行保序 → 改名为 canonical snapshots_v2.parquet。
- `01_build_normalization_stats`：用 episodes_v2 split（无泄露重分段），train=1,472,064/val=186,145/test=338,363；vocab 不变（268/123/278/2184）→ 编码器不重建。
- `16_build_stratum_labels`：late_train 现 **21.28%（非空）**；platform_dev 51.42%（late 经优先级从 platform_dev 拿走 ~21% → 由 67% 降到 51%）；trivial 5.19%；权重 1/√(train_freq)。
- `10_smoke_streaming` **[A][B][C][D] 全 PASS**：[A] 流式 186,126/186,145，块边界丢 19（≤216 超块、全 done=0、=0.01% 可忽略）、extra 0；[B] 1 worker 294/s → 8 worker **1,576/s**（旧 ~16/s，~100×）；[C] num_workers=2 集合一致；[D] 主导 platform_dev 53.8%→35.9% 降、最稀有 unusual_id 0.25%→2.26% 升 + late_train 18.2%→21.5%。
- **逐项核对自洽**：lateness 量级、split 计数与 14 一致、stratum 含 late_train、流式正确性/吞吐/worker/分层。**4.7.2d 全部完成。**

### 🔴 f_platform_dev bug 诊断 + 修复（2026-05-22）
查 platform_dev 51% 异常（`scripts/mdp/19_diagnose_platform_dev.py`，只读、重算 vs stored 0% 误差）：
- f_platform_dev 原始触发 **83.2%**（stratum 51% 是被更高优先级 late/advance/call_on 拿走后的剩余）。
- **触发原因拆分：99.2% 是 `degenerate_allNone`** —— focal 有 planned_platform，但候选路线 `end_platform_id` 全 None → `not any(... if p is not None)` 对空生成器返回 True，**缺数据被误判成"偏离"**。
- 根因：**route 节点仅 27.9% 有 end_platform_id**（许多路线本就不终结于站台——直通/进库，None 合理）。
- **铁证**：候选确实带平台时（仅 ~10% 行），与 planned **匹配率 93.2%** → 真实偏离 `genuine_dev` 仅 0.7%（合 spec §4.4 ~1.5%）。
- **修复**：`special_flags.f_platform_dev` —— 候选 end_platform 全未知时返回 False（保守，不从缺失断言偏离，与 planned=None→False 一致）。触发从 83.2%→~0.7%。
- **落地（外科 patch，同 lateness 套路）**：`scripts/mdp/20_patch_platform_dev.py` 用修正后的函数从 snapshot 自身的 route/train 节点重算该 flag，**只重写 state_special_flags 列、保序**。**无需重跑 01**（flag 是 binary，不进 normalization）；**要重跑 16 stratum + 10 smoke [D]**。
- **Windows 序**：`20_patch_platform_dev` →（验证 after ~0.7%）改名 snapshots_v2.platdev→snapshots_v2.parquet（留备份）→ `16_build_stratum_labels` → `10_smoke_streaming`。任务 #20。

### 下一步
patch + 重跑 16/10 核对后，进 **Stage 5 — 50k sanity 训练**（spec 04 §11）：流式 loader + 分层在 50k 子集跑 3 阶段，验证 loss 下降 + 各阶段成功判据（L_route ≥30% 降、L_TD 降、L_cons 稳、无 NaN、grad norm 有界）。需先把 `09_train.py` 接到 `StreamingTransitionDataset`（当时用的是旧 TransitionDataset）。

### Manuscript draft v0（2026-05-24）
- 使用 `nature-writing` 写作流程，基于当前 RailRL v2 已验证材料 + 私有 Chapter 3 数据获取文稿，新增 `docs/manuscript_draft_v0.md`。
- 稿件定位：ESWA 主投、T-ITS 作为 Stage 8 operational/counterfactual 结果充分后的备选。
- 关键写作纪律：只把已验证事实写成结果（1,996,572 usable snapshots；Stage 5 sanity Q-top1=0.946；泄露审计已过）；Stage 6 三 seed、Stage 7 baselines、Stage 8 Replicate-and-Improve/XAI 仍以占位符标注，避免未完成结果被写成事实。

### Manuscript DOCX export（2026-05-24）
- 新增 `scripts/docs/build_manuscript_docx.py`，将 `docs/manuscript_draft_v0.md` 转成干净的 Word 稿件 `docs/manuscript_draft_v0.docx`。
- 结构检查通过：`python-docx` 可打开，182 paragraphs，DOCX zip 包含 `word/document.xml`/styles/numbering/footer 等核心部件。
- render QA 尝试使用 Documents skill 的 `render_docx.py`，但本机/环境未找到 `soffice`/LibreOffice，可视化 PNG 渲染未完成；交付时需如实说明。

### Academic-paper DOCX draft v1（2026-05-24）
- 使用用户指定的 `academic-paper` skill 重新组织稿件，新增 `docs/manuscript_academic_paper_draft_v1.md` 与 `docs/manuscript_academic_paper_draft_v1.docx`。
- 这版采用 journal article / IMRaD 风格：Paper Configuration Record、英文 abstract、中文工作摘要、Introduction、Related Work、Data、MDP、Model、Experiments、Preliminary Results、Discussion、Conclusion、Declarations、References-to-verify。
- 转换脚本 `scripts/docs/build_manuscript_docx.py` 改成通用 `--src/--out`，并支持简单 Markdown 表格转 Word 真表格。
- 结构检查通过：`python-docx` 可打开，143 paragraphs，1 table，DOCX zip 核心部件完整。环境仍缺 `soffice`/LibreOffice，视觉渲染 QA 未完成。

### Figure 1 architecture schematic (2026-05-24)
- Used the `nature-figure` workflow with the Python/matplotlib backend only, per user selection.
- Added `scripts/figures/fig1_architecture.py` to generate a two-panel manuscript mechanism figure: (a) acquisition-to-evaluation RailRL pipeline; (b) single decision-point MDP/offline-RL mechanism.
- Exported `outputs/figures/fig1_architecture.svg`, `.pdf`, `.tiff` at 600 dpi, and `.png` preview. Visual QA passed after reducing title collisions and callout overlap.
- Public figure text intentionally omits SOP decoding/table details and does not expose protected operational reference material.

### RailRL System / Architecture figure package (2026-05-24)
- Used the user-provided `engineering-figure-agent` workflow from `C:\Users\92588\.codex\skills\engineering-figure-agent\adapters\claude-code\skills\engineering-figure-agent\SKILL.md`.
- Added `outputs/figures/system_architecture/figure-brief.md` and `outputs/figures/system_architecture/prompt.txt` to capture the figure goal, claim, public labels, style constraints, and QA checklist.
- Added `scripts/figures/fig_system_architecture.py` to generate a single-panel System / Architecture diagram: operational inputs -> acquisition service -> canonical store -> decision dataset -> offline RL core -> evaluation/XAI -> decision-support interface, plus a validation/publication-evidence rail.
- Exported `outputs/figures/system_architecture/output/railrl_system_architecture.svg`, `.pdf`, `.tiff` at 600 dpi, and `.png` preview.
- QA passed: output files exist, PNG/TIFF dimensions verified, TIFF dpi verified, visible SVG text scan contains no SOP/decoding/table-sensitive wording, and visual preview checked for text overlap.

### Reference-style System / Architecture figure (2026-05-24)
- User provided example architecture figures with pastel panels, internal module drawings, arrowed workflow, legends, and a bottom application/evidence band.
- Added `scripts/figures/fig_system_architecture_reference_style.py` to generate a closer reference-style RailRL architecture figure with four panels: Data Layer, Decision Layer, Learning Layer, and Evaluation + XAI.
- Exported `outputs/figures/system_architecture_reference_style/output/railrl_system_architecture_reference_style.svg`, `.pdf`, `.tiff` at 600 dpi, and `.png` preview.
- QA passed: output files exist, PNG/TIFF dimensions verified, TIFF dpi verified, SVG/text scan contains no SOP/decoding/table-sensitive wording, and visual preview checked after fixing bottom evidence-rail overlap.
