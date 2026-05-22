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
