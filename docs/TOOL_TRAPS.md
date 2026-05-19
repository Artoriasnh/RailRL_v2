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
