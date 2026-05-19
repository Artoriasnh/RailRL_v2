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
| Stage 3 | 新 snapshot builder (state + leak audit + episodes) | ⏳ next | — | spec 02 §4-§8 |
| Stage 4 | 主模型（HGT + Transformer + Q + 2 aux heads + CQL） | ⏳ pending | — | spec 03 + 04 |
| Stage 5 | Sanity 训练 50k subset | ⏳ pending | — | spec 04 §11 |
| Stage 6 | 全数据训练 3 seeds | ⏳ pending | — | spec 04 §10 |
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
