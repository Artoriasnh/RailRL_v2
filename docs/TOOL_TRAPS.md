# Tool Traps & Workarounds

> **目的**：记录"AI 工具的系统性陷阱" — 不是项目本身的 bug，而是
> Claude / Edit / Write / pyc / virtiofs 这一类**工具底层**带来的问题。
>
> **使用方法**：新对话开场让 AI 助手先读这份文档，避免重蹈覆辙。
> 项目 bug 记录在 `docs/IMPLEMENTATION_LOG.md`；本文件只记录**工具陷阱**。
>
> **维护规则**：每次遇到新的工具坑，追加一节。仅追加不删旧记录。

---

## 总览

| ID | 陷阱 | 严重度 | 频率 | 见 § |
|----|------|--------|------|------|
| T1 | Edit/Write 工具沉默截断长内容 | 🔴 高 | 多次/会话 | §1 |
| T2 | pyc 缓存 mtime 不一致 | 🟡 中 | 偶尔 | §2 |
| T3 | virtiofs 删除文件 Operation not permitted | 🟡 中 | 偶尔 | §3 |
| T4 | sandbox /sessions/sdc 磁盘小（~10 GB）| 🟡 中 | 装包时 | §4 |
| T5 | Python `tcs[-0:]` 切片返回整个 list | 🟡 中 | 一次 | §5 |
| T6 | mv 跨 virtiofs/sandbox 失败 | 🟢 低 | 偶尔 | §6 |

---

## §1 — Edit / Write 工具沉默截断长内容 🔴

### 症状

- Write 报 `File created successfully` 但实际文件比预期短
- Edit 报 `updated successfully` 但替换块末尾被切断
- Python 解析时报 `SyntaxError: unterminated string literal`
  / `'{' was never closed` / `unmatched '}'` 等
- 或更隐蔽：`NameError` 因为某行变量赋值被悄悄删了

### 已确认发生的位置（本项目）

| 时间 | 文件 | 截断处 | 后果 |
|------|------|--------|------|
| 2026-05-19 早 | `_build_handoff.py` | `print(f"  Paragraphs: ..."` 缺收尾 | docx 生成失败 |
| 2026-05-19 中 | `spec/01_data_pipeline.md` | §17 后半段 + §18 全没了 | spec 缺章 |
| 2026-05-19 中 | `_build_handoff.py` | callout 字符串里的曲引号炸 | SyntaxError |
| 2026-05-19 晚 | `src/railrl/config.py` | `TD_PARQUET = ...` 那一行没写出 | 脚本 AttributeError |
| 2026-05-19 晚 | `src/railrl/mdp/trigger.py` | summarize() 末尾 `per_train` 那行丢 | NameError |
| 2026-05-19 晚 | `src/railrl/mdp/trigger.py` | dict literal `"max":  ` 收尾被切 | SyntaxError |
| 2026-05-19 晚 | `src/railrl/data/reward_calibration.py` | 文件最后的 `}` 收尾被切 | unmatched `}` |

**至少 7 次**，跨 3 个 session。这是**系统性问题**，不是偶发。

### 根因（猜测）

- Write 工具对长内容（> ~200 行 或含复杂转义）有沉默截断
- 字符串里的 `"` / `\` / 中文曲引号 `""` 容易触发
- 不是单次 token 限制，因为偶尔 Write 400+ 行能成功
- 跟 mount 类型（virtiofs）+ 大写盘符 + Windows 行尾可能有关

### 对策（按优先级）

#### A. 用 bash heredoc 写长内容（最稳）

```bash
cat > path/to/file.py << 'EOF'
# ... 任意长度任意字符的内容 ...
echo "writing complete content here"
EOF
```

bash heredoc 把字符当字面值，不解释 `\` / `"` / 中文引号。

#### B. 每次 Edit/Write 完做 3 步验证

```bash
wc -l <file>                                                  # 行数对吗
tail -10 <file>                                               # 末尾完整吗
python3 -c "import ast; ast.parse(open('<file>').read())"     # 语法 OK 吗
```

90% 的截断在这 3 步里能发现。

#### C. 重要常量加单测

`tests/test_config.py` 已经实现了"每一个常量存在"的测试。这种测试**就是**用来捕捉
"config 文件被截断丢了某个常量"这类隐蔽 bug。

### 检测

- Python 报错信息和当前代码内容**矛盾**时 → 第一反应是 Edit 截断 + pyc 缓存
- 用 `head -N <报错行号> <文件>` 看真实内容；如果跟 traceback 不一样 → 确认

### 修复 protocol（截断后怎么救）

1. `wc -l` + `tail -10` 找截断点
2. `head -N file > /tmp/clean.py && cat >> /tmp/clean.py << 'EOF' ... EOF`
3. `cp /tmp/clean.py file.py`（virtiofs 不允许跨设备 mv）
4. `python3 -c "import ast; ast.parse(...)"` 验证语法
5. `touch file.py && rm -rf __pycache__` 清缓存

---

## §2 — pyc 缓存 mtime 不一致 🟡

### 症状

- 修改 .py 后旧的 .pyc 仍被使用
- 报错指向已删除的旧 import 或旧行号
- 同样代码 sandbox 和 Windows 端行为不一致

### 已确认发生

- 改 trigger.py 后仍报旧 `from .snapshot import` 错（pyc 没重编译）
- 改 config.py TD_PARQUET 后仍报 AttributeError

### 根因

Python `.pyc` 重编译依赖 `.py` mtime > `.pyc` mtime。virtiofs 跨
Windows ↔ Linux 时 mtime 可能错乱；Edit 工具有时不更新 mtime。

### 对策

```bash
# 改完任何 .py 后立刻
touch file.py                        # 强制更新 mtime

# Linux
find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null

# Windows PowerShell
Get-ChildItem -Recurse -Filter __pycache__ | Remove-Item -Recurse -Force
```

⚠️ `python3 -B` 只防写新 pyc，**不防读旧 pyc**——不够。

---

## §3 — virtiofs 删文件 Operation not permitted 🟡

### 症状

```
rm: cannot remove '...': Operation not permitted
mv: inter-device move failed: '/tmp/x' to '/mnt/...';
    unable to remove target: Operation not permitted
```

### 根因

virtiofs 的权限模型不完全允许 unlink / chmod，
特别是 sandbox 写入的 pyc 文件、跨设备 mv。

### 对策

- 删 pyc：用 `find ... -exec rm -rf {} +` 而不是 `rm -rf` 直接打
- 不能 mv 时用 `cp` 替换：
  ```bash
  # WRONG
  mv /tmp/clean.py src/railrl/config.py
  # RIGHT
  cp /tmp/clean.py src/railrl/config.py
  ```

---

## §4 — sandbox 磁盘空间紧张 🟡

### 症状

```
No space left on device
ENOSPC ... pip / npm install failed
```

### 根因

sandbox `/sessions` (`/dev/sdc`) 通常只有 ~10 GB，且共享给所有 process。
TD_data.csv (713 MB) + npm 缓存 + pip 缓存能很快填满。

### 对策

- **不在 sandbox 装新依赖**——用 Windows 侧 `pip install -e .[dev]`
- sandbox 只跑 syntax check / 验证 import 路径，不跑训练
- 大文件 copy 用 `dd bs=1M count=200 conv=notrunc skip=N seek=N` 分块
  （每块 < 45s 避免 bash timeout）

### 检测

```bash
df -h /sessions
```

---

## §5 — Python `tcs[-0:]` 切片返回整个 list 🟡

### 症状

```python
>>> tcs = ['A', 'B']
>>> tcs[-2:]
['A', 'B']
>>> tcs[-0:]
['A', 'B']    # ← 不是 []!
```

任何接受 k 作为"取最后 N 个"的代码都要检查 k=0 边界。

### 已确认发生

- `trigger.compute_approach_tracks(routes, k_hops=0)` 返回全部 TCs 而非空集
- 单元测试 `test_k_hops_zero` 抓到

### 对策

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

### 检测

任何带 `[-k:]` 的代码加 k=0 单元测试。

---

## §6 — bash 工具 45 秒 hard timeout 🟢

### 症状

```
bash failed on resume: Command timed out after 45000ms
```

### 根因

bash 工具有 45s timeout。后续 bash 调用可能被锁，要等几秒。

### 对策

- 单次 bash 调用 < 30s 留余量
- 长操作分块（如 dd 分 200 MB chunks，每块 ~30s）
- 不用 `du -sh` 扫大目录（用 `du -sh --max-depth=1`）
- 后台进程 `nohup ... &` 不可靠—— sandbox 在 bash session 结束后会杀掉

---

## 通用调试 protocol — 4 步发现 90% 的工具陷阱

任何 bash / python 错误的标准调试序：

```bash
# 1. 文件长度对吗？（截断检测）
wc -l <file>

# 2. 末尾完整吗？（截断检测）
tail -10 <file>

# 3. 语法 OK 吗？（截断检测）
python3 -c "import ast; ast.parse(open('<file>').read())"

# 4. 缓存清了吗？（pyc 检测）
find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null
touch <file>
```

90% 的 SyntaxError / NameError / ImportError 在这 4 步里能查出来。

---

## 更新日志（本文档自身的）

- **2026-05-19** v1.0 — 初版，记录 Stage 1-2 期间发现的 6 个工具陷阱
- 未来发现新陷阱：在末尾追加 §7、§8 ...


---

## §7 — `touch` 在 virtiofs 上可能截断文件（Stage 3 R3）

**症状**：跑 `touch src/railrl/mdp/state_history.py` 期望仅 bump mtime，结果文件被截断到 ~500 行（原本 503+ 行），最后一个方法 `slice_last_k` 的 body 被切掉，引发 `TypeError: 'NoneType' object is not iterable`（缺 `return out`）。

**原因推断**：Windows ↔ Linux virtiofs 的同步层。`touch` 在某些情况下会触发 fsync-like 操作，跨同步层时如果有 in-flight write 被 drop，文件就裸露 partial 状态。

**避免**：
- **不要用 `touch` 在 virtiofs 挂载下 bump mtime**。
- 用 `python -c "p='...'; open(p,'w').write(open(p).read())"` 替代 — 显式 read + write，full content guaranteed。
- 改完后立即 3 件套：`ast.parse + wc -l + tail -10`。

---

## §8 — heredoc 追加遇到 linter 的"自动修复"会产生重复块（Stage 3 R3）

**症状**：state.py 末尾的 `_to_nullable_int` helper 因为 Edit 截断只剩 `except` 关键字断掉 → `try: return int(v); exc`。用 heredoc 追加 `ept (TypeError, ValueError):\n    return None\n...`。但 linter 已经先一步把整个文件修好了（自动补全 `except`）。结果文件里出现两份内容：lines 712-724 是 linter 版本，lines 725-732 是我的 heredoc 版本。后者 `ept (TypeError, ValueError):` 是 orphan，触发 `SyntaxError: invalid syntax`。

**根因**：cowork 环境里有自动 linter（system reminder 形式提示）会修文件。如果你已经规划好 heredoc 追加，但 linter 在你 heredoc 前就修好了，你的 heredoc 会落到修好版本后面，造成重复。

**避免**：
- **改前用 `tail -5` 检查文件结尾**。如果看到看似已经完整的代码（比如 `except` 块完整），不要再 heredoc 追加。
- 如果不确定，用 `head -N` 截断 + 整段重写比 heredoc 追加更安全。

---

## §9 — pandas `Series.apply()` 自动 coerce mixed int/None 到 float64（Stage 3 R3）

**症状**：`_parse_plat` 函数返回 int 或 None。在隔离测试中 `[_parse_plat(v) for v in ...]` 给出 `[3, None]`（types: int, NoneType）。但同样的代码放进 `MovementsLookup.build()` 的实际执行路径，存进 dict 后变成 `[3.0, nan]`（types: float, float）。差异来源：实际路径里中间经过了一次 `pd.Series → list` 之类的转换。

**测试**：
```python
s = pd.Series([3, 4, 99]).apply(lambda v: int(v) if 1<=v<=6 else None)
print(s.dtype)  # float64 ！
print(s.tolist())  # [3.0, 4.0, nan]
```

**避免**：
- **不要依赖 pandas Series 的元素类型保留**。在最终存储点强制 `int(p) if 1 <= p <= 6 else None`。
- 这条对 leak audit 也重要：Check 4 严格要求 `planned_platform` 是 int（不是 float）。

---

## §10 — pyc 缓存在 virtiofs 上挡住代码更新（Stage 3 R3 重灾区）

**症状**：改了 `state.py` 的方法签名（如新加 `_collect_other_active_trains`），但运行测试时 `AttributeError: 'SnapshotBuilder' object has no attribute '_collect_other_active_trains'`。`inspect.getsource()` 显示的源码确实有这个方法（说明 .py 是新的），但 dataclass `__dataclass_fields__` 还是旧的（说明加载的是 stale pyc）。

**原因**：virtiofs 上 .py 和 .pyc 的 mtime 同步性微妙。Edit 工具改 .py 但 mtime 不一定 bump。Python 用 mtime 比较来判断 pyc 是否过期，若 .py 的 mtime <= .pyc 内嵌的 mtime，pyc 被认为是新鲜的，被加载。

**避免**：
- **永远的最后一招**：`python -c "p='<file>'; open(p,'w').write(open(p).read())"` — 读完整内容再写回，强制 bump mtime。
- 不要用 `find . -name __pycache__ -exec rm -rf {} +` — virtiofs 上很多情况下没有写权限删 pyc。
- 不要用 `touch` —— 见 §7。

---

## §11 — `/sessions` 沙盒磁盘满 → bash 看到的挂载文件被冻结在旧版本（Stage 3 hotfix）

**症状**：用 Edit 工具改了 4 个文件，Read 工具确认改动都在（Windows 侧文件完整），但 sandbox 的 `bash` 里 `wc -l` / `cat` / `ast.parse` 看到的是**截断的旧版本**，且反复 retry + sleep 都不刷新。`stat` 显示的 mtime 是几天前的旧时间戳。

**根因**：`df -h /sessions` 显示 **100% 满**（9.8G 用满，仅剩几 MB）。bash 沙盒通过 virtiofs 挂载读 `E:\` 的文件，需要把更新后的副本写进沙盒缓存。磁盘满 → 写不进 → bash 一直读旧的冻结副本。罪魁通常是之前缓存的大数据文件（td_data.parquet 11.7M 行、Movements.csv 50MB 等）撑满了 virtiofs cache。

**关键认知**：
- **Read/Edit/Write 工具走 Windows 文件 API，不受沙盒磁盘影响** —— 它们读写的就是用户在 Windows 上跑的真实文件。
- **bash 走 virtiofs 挂载，磁盘满时会读到冻结的旧副本**。
- **用户在 Windows 上跑 pytest/脚本，拿到的是 Read 工具看到的版本（正确完整的）**。

**避免 / 应对**：
1. 当 bash 的 `wc -l` 跟 Read 工具结果不一致时，先 `df -h /sessions`，如果满了就知道是这个问题。
2. **用 Read 工具逐段验证编辑区域**（它读 Windows 真实文件），不要依赖 bash 的 AST 检查。
3. 清缓存：`rm -rf ~/.cache/* ~/.local/...`，但大头是 virtiofs 缓存的挂载数据文件（内核应自动回收，满的时候可能回收不及时）。
4. 验证逻辑正确性时，把逻辑抽出来在 `/tmp`（在 `/` 而非 `/sessions`，通常还有空间）写独立脚本测，不 import 挂载的包。
5. 不要 `pip install` 大包到 `/sessions`（用 `--target=/tmp/...`）。

---

## 更新日志（增量）

- **2026-05-20** v1.1 — 增加 §7-§10（Stage 3 R3 期间发现的 4 个新陷阱）
- **2026-05-20** v1.2 — 增加 §11（/sessions 磁盘满导致 bash 挂载文件冻结）


---

## §12 — datetime64[us] vs ns 单位不匹配（Stage 3 数据 bug，最严重）

**症状**：全量构建 2M snapshots 后审计，发现 event token 的 `time_delta_s` 100% 都是 ~1.69e9（看着像 UNIX 秒时间戳，不是"距决策多少秒"的 delta）。

**根因**：`td_data.parquet` 的 `time` 列是 **datetime64[us]（微秒）**。各 history builder 用 `sub["time"].astype("int64")` 得到的是**微秒**（~1.69e15）；但 `build_snapshot` 里 `t_ns = pd.Timestamp(decision["t"]).value` 是**纳秒**（~1.69e18）。两者差 1000 倍。结果：`t_ns >> 所有事件时间` → `np.searchsorted` 永远返回末尾 → 所有时间查询都返回"该资产最后一次事件"（既是未来泄露，又完全错误）。

**影响范围**（全错，需重建）：current_tc（子图中心！）、occupied_now、current_occupier、occupancy_fraction_*、n_state_changes_*、aspect_*、last_change_age_s、berth 占用、recent_panel_requests、event token deltas。
**不受影响**：candidates、Derby_info、静态属性、episodes/pass_id、schedule_outlook（Movements 走 ns，一致）。

**为什么所有单元测试 + 合成测试都没抓到**：合成数据用 `pd.to_datetime([...])` 创建的是 datetime64[**ns**]，`.astype("int64")` 给 ns，恰好和 t_ns 一致 → 测试全过。真实 parquet 是 us → 暴露。**这是合成测试和真实数据 dtype 不一致导致的盲区。**

**修复**：加 `_to_ns_int64(times)` helper：`np.asarray(times.values).astype("datetime64[ns]").astype("int64")` —— 不管源是 us 还是 ns，强制转 ns。在所有 TD-time 转换点用它（state_history 5 处 + state_helpers TrainStateLookup 1 处）。

**教训 / 检查清单**：
1. **任何时间→int 的转换都要显式声明单位**。`datetime64.astype("int64")` 的结果单位 = 该列的 resolution（us/ns/ms），不一定是 ns！
2. **`pd.to_datetime(series)` 在 pandas 2.x 保留源单位**（us 进 → us 出），不会自动转 ns。要强制 ns 用 `.values.astype("datetime64[ns]")`。
3. **合成测试必须复刻真实数据的 dtype**（包括 datetime resolution）。用 `pd.to_datetime(...).astype("datetime64[us]")` 复刻 td 的 us。
4. **验证产物时，看特征的数值范围/分布，不只是"非空/有值"**。time_delta_s 全是 1.69e9 一眼就能看出是时间戳不是 delta —— 但只检查"schedule_outlook 94% 非空"这种就漏了。

---

## 更新日志（增量）

- **2026-05-20** v1.3 — 增加 §12（datetime64 us/ns 单位 bug，Stage 3 最严重数据 bug）


---

## §13 — nn.ModuleDict 不能用 'train'/'eval'/'type' 等保留名做 key（Stage 4.3）

**症状**：`HGTEncoder.build` 报 `KeyError: "attribute 'train' already exists"`，在 `self.ident["train"] = nn.Embedding(...)`。

**根因**：`nn.ModuleDict[key]=mod` 调 `Module.add_module(key, mod)`，而 `nn.Module` 已有 `train()` / `eval()` / `type()` / `to()` / `cpu()` / `forward` 等方法，同名 key 被拒。我们用节点类型名（track/signal/route/**train**）做 ModuleDict key，"train" 撞车。

**修复**：所有按节点类型 key 的 ModuleDict（ident/cats/proj/norm）一律加前缀 `"nt_"+nt`，forward 里同样。

**避免**：任何用"业务名字"做 ModuleDict/ Module 属性名时，避开 PyTorch 保留名（train/eval/type/to/cpu/cuda/forward/training/...）。统一加前缀最省心。policies/heads.py 若也按节点类型 key 要注意。

---

## §14 — pyarrow 整表载入 + sort_by/take 在嵌套列上内存爆炸（Stage 4.7.2d）

**症状**：`15_resort_snapshots_canonical.py` 第一版 `pq.read_table(整个 573MB snapshots)` + `Table.sort_by(...)`，在 **31GB RAM** 的机器上把系统物理内存吃光，连带把 **PyCharm 的 JVM 也 OOM 杀掉**（`java_error_in_pycharm_*.log`：`malloc failed to allocate 1046512 bytes ... system is out of physical RAM`）。注意：崩溃日志是 **PyCharm/JVM 的**，不是 Python traceback——容易误判成 IDE 问题，实则是同机的 Python 进程把 RAM 吃干。

**根因**：snapshots_v2.parquet 虽然 zstd 压缩后只有 573MB，但**列是大量 list-of-struct 嵌套**（state_nodes_* / event_tokens 256 / edges 8 类 …）。`read_table` 解码成 Arrow in-memory 后**膨胀到十几 GB**；`sort_by`（以及 `take`/`join`）会再**整体复制一份** → 峰值 ×2，轻松超过 31GB。

**修复**：改成**内存有界的流式外排（bucket sort）**：
- Pass 1：逐 row group 读（~5000 行/次）→ 用 sidecar 替换 episode 列 + 加 split → 按 episode_idx 分到 N 个 bucket 临时文件（每 bucket 是连续 episode_idx 区间）。峰值 = 一个 row group。
- Pass 2：按 bucket 顺序逐个读回（~几万行）→ 桶内 `sort_by(episode_idx, position)` → 追加写最终文件。峰值 = 一个 bucket。
- 全程峰值几百 MB，且天然顺序写。

**避免 / 检查清单**：
1. **嵌套列的大 parquet 永远不要"整表 read_table + sort_by/take"**——哪怕压缩后看着不大。先按行组流式处理，需要全局排序就走 bucket/外排。
2. 估内存按**解码后**算（嵌套可膨胀 10-30×），不是按 parquet 文件大小。
3. 机器是 31GB（本地 Ryzen 9 9955HX）；重活在它上面跑要留意峰值。崩溃若是 `java_error_in_pycharm`，先怀疑同机 Python 进程吃光 RAM，而非 IDE bug。
4. 跑大内存脚本尽量用**独立终端**（PowerShell）而非 PyCharm 内置运行器，少和 IDE 抢内存。

---

## §15 — HPC DataLoader 多 worker 报 `received 0 items of ancdata`（Stage 6）

**症状**：服务器（HPC sapphire）上 `num_workers=16` 跑 `09_train.py` 报
`RuntimeError: received 0 items of ancdata`（在 `recvfds`）。`num_workers=8` 正常。

**根因**：PyTorch DataLoader 默认 `file_descriptor` 张量共享策略，多 worker 传 PyG Batch
时超出进程的 fd 预算（cluster 上 `ulimit -n` 常较低）。**非代码 bug**。

**修复**：主进程 spawn worker 前设 `torch.multiprocessing.set_sharing_strategy("file_system")`
（改用 /tmp 文件共享，绕开 fd 限制）→ num_workers≥16 可用。备选：`ulimit -n 4096`。

**避免**：任何在 HPC 上多 worker 的 DataLoader，开头就设 file_system 策略。

---

## §16 — pandas `pd.notna()` 作用在"嵌套列单元（numpy array）"上 truth ambiguous（Stage 6 审计）

**症状**：`06_run_leak_audit_full.py` 用 `pd.read_parquet→iterrows`，
`{k: row[k] for k in keys if pd.notna(row.get(k)) or isinstance(...)}` 报
`ValueError: The truth value of an array with more than one element is ambiguous`。

**根因**：`to_pandas()` 把 list-of-struct 列解码成 **numpy object array** 的单元；
`pd.notna(np.array([...]))` 返回数组，`数组 or ...` → `bool(数组)` → 多元素时报错。

**避免**：处理嵌套 parquet 的行，用 **`ParquetFile.read_row_group(rg).to_pylist()`** 拿
**Python dict 行**（嵌套为 list/dict、null 为 None），判空用 `v is not None`，别用 pandas
Series + `pd.notna`。（顺带也内存有界、避开 §14 的整表 read。）

**连带**：从 parquet 重建 snapshot 给 leak_audit 时，注意列名——文件存 `state_center`，
而 `leak_audit` Check1 读 `snapshot["center"]` → 需别名 `center=state_center`，否则 100% 假 fail。

---

## §17 — PyTorch 2.6 `torch.load` 默认 `weights_only=True` 拒载 numpy 标量（Stage 6 resume）

**症状**：服务器（torch 2.6）resume 时 `torch.load(resume_path)` 报
`UnpicklingError: Weights only load failed ... Unsupported global numpy._core.multiarray.scalar`。

**根因**：PyTorch 2.6 把 `torch.load` 的 `weights_only` 默认从 False 改成 **True**（只允许张量/基本类型）。
我们的 resume checkpoint 含**非张量状态**（best_state/log/**gates**），其中 `gates["A"]` 的
"loss finite" 项是 `np.isfinite(...)` → **numpy.bool_** 标量 → 被拒。

**为什么 `--max-batches 5` 测试时没事**：那次在 **Phase A 中途**被 Ctrl-C，gates 还是 `{}`（A 的 gate 在 run_phase 返回后才算）→ checkpoint 无 numpy 标量 → 加载成功。全量跑进了 **Phase B**，gates["A"] 已含 numpy bool → 加载失败。

**修复**：(1) 加载**自己写的可信** checkpoint 用 `torch.load(..., weights_only=False)`；
(2) 写入端把 numpy 标量转 python（`bool(np.isfinite(...))`），keep checkpoint 干净。
**避免**：torch≥2.6 下，凡 `torch.load` 自己存的含非张量对象的 ckpt，显式 `weights_only=False`；
存 ckpt 时别塞 numpy 标量（统一 `float()/bool()/int()`）。

## 更新日志（增量）
- **2026-05-20** v1.4 — 增加 §13（nn.ModuleDict 保留名冲突）
- **2026-05-22** v1.5 — 增加 §14（pyarrow 整表 sort_by 在嵌套列上 OOM；改用流式 bucket 外排）
- **2026-05-22** v1.6 — 增加 §15（HPC DataLoader fd-sharing ancdata→file_system）+ §16（pd.notna 嵌套数组 + center 列名别名）
- **2026-05-22** v1.7 — 增加 §17（torch 2.6 weights_only=True 拒载 numpy 标量；resume 加载用 weights_only=False）

---

## §18 — pandas pyarrow/string-dtype 列不能被掩码赋 Timestamp + 合成测试 dtype 不匹配（2026-05-24 fix #2）

**症状**：`correct_movements_bst` 里 `df.loc[mask, c] = pd.to_datetime(...) + delta` 报
`TypeError: Invalid value for dtype 'str'. Value should be a string or missing value`。

**根因**：`pd.read_csv(usecols=[...])`（compute_delay_changes 用）在 pandas 2.x 下把那两列读成 **pyarrow-backed string dtype**；往 string-dtype 列**掩码赋值一个 Timestamp** 不被允许。而 `load_movements` 用 `parse_dates=[...]` → datetime64 列，赋值 OK → **同一函数在两条读取路径下，一条过一条崩**（23 走 load_movements 过了，09 走 read_csv 崩了）。

**为什么 /tmp 测试没抓到**：合成 DataFrame 用的是 **object-dtype** 字符串列（普通 python str），object 列能装 Timestamp（混合类型），所以测试通过；真实 read_csv 是 **pyarrow-string dtype**，不能。又是"合成 dtype ≠ 真实 dtype"（同 §12 us/ns 教训）。

**修**：掩码减法前先把整列转 datetime（`df[c] = pd.to_datetime(df[c], errors="coerce")`，已是 datetime 时为 no-op），再 `df.loc[mask,c] = df.loc[mask,c] + delta`。两种 dtype 都稳。

**教训**：(1) 往 pandas 列写值要确认列 dtype 能容纳该类型，尤其 pyarrow-backed dtype 严格；(2) **合成测试必须复刻真实读取路径的 dtype**（read_csv 的 string-backend vs parse_dates 的 datetime vs object），否则盲区——这是本项目第二次栽在 dtype 不匹配上（见 §12）。

## 更新日志（增量）
- **2026-05-24** v1.8 — 增加 §18（pyarrow/string-dtype 掩码赋 Timestamp 失败 + 合成测试 dtype 不匹配，fix #2 reward 路径）

---

## §19 — Grep/Glob 工具不传 `path` 时搜的是 scratchpad cwd（不是项目仓库）（2026-05-26）

**症状**：Grep `pattern="r_wait|r_throughput|..."` 不带 `path` → 返回 "No files found"；Glob `**/*reward*.py` 不带 path → 空。差点据此误判"奖励分量列名不存在 / 文件不存在"。同一 Grep 带 `path=E:\Claude\RailRL_v2` → 立刻命中 47 个文件。

**根因**：cowork 的工作目录(cwd)是 outputs scratchpad（`...\local_*\outputs`），**不是**用户挂载的项目仓库 `E:\Claude\RailRL_v2`。Grep/Glob 省略 `path` 时默认搜 cwd → 搜了个几乎空的暂存目录 → 假"无结果"。

**避免**：
- **对项目代码的 Grep/Glob 一律显式传 `path="E:\Claude\RailRL_v2"`（或更精确的子目录）。** 别依赖默认 cwd。
- "No files found" 先怀疑 path 错了，再下"不存在"的结论——尤其当你明知某符号本应存在时。

## 更新日志（增量）
- **2026-05-26** v1.9 — 增加 §19（Grep/Glob 默认 cwd 是 scratchpad 非仓库，须显式传 path）

## §20 磁盘满→文件 NUL 损坏 + 不可删的陈旧 .pyc（2026-05-27）
- **现象**：Edit 改了 `C.RAW_DIR`→`C.ROUTE_TO_TC_CSV`，但 python 仍报 `AttributeError: module 'railrl.config' has no attribute 'RAW_DIR'`，且 traceback 行号指向**新**代码行。`grep` 把 .py 当 **binary file**。
- **根因**：sandbox 磁盘 100%（pip install pyarrow 触发 "No space left on device"）时，Edit 写入把文件插入了 **16 个 NUL 字节**（损坏），且 `__pycache__/*.pyc` 变得**不可删**（`rm: Operation not permitted`）→ python 加载陈旧 .pyc（含旧 RAW_DIR）。
- **修法**：① 去 NUL：`tr -d '\000' < f > f.tmp && mv f.tmp f`（核验 `open(f,'rb').read().count(b'\x00')==0`）；② 绕过不可删的本地 .pyc：`export PYTHONPYCACHEPREFIX=/tmp/pyc_$RANDOM` → python 改从 /tmp 读写字节码、忽略本地陈旧 __pycache__、从源码重编译。两招合用后导入正常。
- **预防**：沙盒里别 `pip install` 大包（无空间）；改文件后若行为诡异先 `grep -c $'\x00'` 查 NUL + 设 PYTHONPYCACHEPREFIX。torch/pyarrow 实跑交 Hao 的 Windows 端。

## §21 虚拟文件系统 mount 冻结在截断版本（2026-05-27，§11/§20 续）
- **现象**：Edit/Write 改 rule_base.py 后，Read 工具(Windows 侧)显示完整 402 行、语法正确；但 sandbox 里 `wc -l`/`cp`/`python compile` 始终看到**截断的 358-359 行**(triple-quote 数为奇数→unterminated docstring SyntaxError)，`sleep`+重读、Write 完整重写都**无法刷新** sandbox 的 mount 缓存。
- **根因**：磁盘满引发的 virtiofs mount 缓存冻结——sandbox 的文件视图卡在某个中间(截断)状态，与 Windows 实际文件不一致。
- **影响/对策**：① **Windows 文件是对的**(Read 工具 + Write 成功)，**Hao 在 Windows 跑不受影响**；② 但 sandbox **无法 import 该模块自测**。绕法：把**纯逻辑 + 真实 CSV 数据**(数据文件读得到、只有 .py 模块被冻结)拷到 /tmp 写独立测试，复刻待验逻辑跑通 → 既验证又不依赖冻结的 mount。③ 真验证交 Hao Windows 端实跑(import 失败会立刻 SyntaxError 暴露截断)。

## §22 pandas list 列 → numpy array，`arr or []` 触发 ambiguous-truth（2026-05-27）
- **现象**：13_selective_override smoke 报 `ValueError: The truth value of an array with more than one element is ambiguous`，在 `cands = list(row["candidate_route_ids"] or [])`。
- **根因**：pyarrow→pandas（to_pandas）把 list 列变成 **numpy object array**；对 >1 元素的 array 求布尔（`arr or []`）非法。03 当年用的是 `list(cv) if cv is not None else []`（避开 `or`），我图省事用了 `or` 才踩。
- **修**：`cv=row["col"]; cands = list(cv) if isinstance(cv,(list,np.ndarray)) else []`（同时挡住 None 和 float nan）。沙盒验证 numpy array/None/nan/wait 全 OK。
- **预防**：从 DataFrame 行取 list/array 列，**永远别用 `x or default`**；用 `isinstance` 或 `x if x is not None else default`（且 nan 也要挡）。

## §23 结构化动作空间：跨状态 gather 动作 idx 撞 masked −1e9（2026-05-27）
- IQL 的 L_V 误用 next-state Q 作 Q_target(s,a)：当前动作 idx 在 s' 的 mask 下常指向 masked 槽(−1e9)→ loss≈1e18→发散。**凡用 chosen_action_idx 去 gather Q，必须 gather 的是该 idx 所属状态的 Q**(当前动作→当前状态 Q)。masked 槽用 −1e9 sentinel,跨状态 gather 必爆。
