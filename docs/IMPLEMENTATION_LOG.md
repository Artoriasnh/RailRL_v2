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
| Stage 3 | 新 snapshot builder (state + leak audit + episodes) | ✅ done | 2026-05-20 | spec 02 §4-§8 |
| Stage 4 | 主模型（HGT + Transformer + Q + 2 aux heads + CQL） | ⏳ next | — | spec 03 + 04 |
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
