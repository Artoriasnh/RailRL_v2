# L4 规则库 — 全量草稿（按 spec 05 §13.2 schema，送 Hao 终审）

> 来源 `data/domain/Training_Plan_2022.docx`；方向→signal/TC 锚点来自 Hao 回填的 `destinations_to_map.md`（已核验，TFPY→TFPV 已更正）。
> schema 字段（§13.2）：`rule_id, source_section, cond_origin, cond_destination, cond_train_class, cond_time_of_day, cond_other, preferred_route_id, preferred_platform, non_preferred_alternatives, confidence, user_approved, notes`。
> **R1–R7 = Hao 已于 2026-05-27 批准**（原样落 schema）；**C8–C17 = 新起草，请逐条审**（每条 `审核:` 行填 approved/改/拒）。
> ⚠️ 审过的才写 `rules.parquet`（`user_approved=true`）。L4 统计中 **med 软偏好只作参考、不计入 §12 override 闸**。

> **审核进度（2026-05-27）**：**全部 19 条已批准**（R1–R7 + C8–C17，含 C15b "没问题"）。可写 `rules.parquet`。
> **缺口已全部闭合**：RTC South 检测改锚 **TD5043**（Hao 给的 TC TDWV 经核验不存在）；Etches=TECF/TECJ✓、Litchurch 出口=DW5310✓；**C9 的 A 线** = 平台1 TC{TPSL,TPSM,TPSU} 经 DC5061 北向引出（Hao 图示确认，无独立 A 线 TC）。
> **可检性已确认**：destination 可由 headcode `hc_dest` 或所设路由的终到信号判定、train_class 可由 `hc_class_digit`(HEADCODE_CLASS) 判定 → 规则可逐决策匹配（精确检测逻辑在 l4_rules.py 落定）。

---

## 方向 → 锚点 图例（Hao 回填 + AI 核验，全部 TC 在路由数据中存在）

| 方向/终点 | 连续信号 / TC 锚点 | 备注 |
|---|---|---|
| North（含 Chesterfield, Barrow Hill, Erewash, South Wingfield, Sheffield 经 Chesterfield） | Duffield：**T884**(出 Derby/北行) / T883(入 Derby) | Derby 往北最近站 Duffield |
| South（含 Birmingham, Crewe, Stenson） | Pear Tree 方向：signal **DW5319** / TC **TYVR** | TYVR 见 RDW5311/5313/5315A(M) |
| West | **DW5302** / 平台2 TC{TYTW,TYTV,TYTS} | |
| East / Nottingham（Spondon 侧） | Nottingham：**TD5030 / TFPV**；Derby 向：TD5029/TDMC | TFPY 不存在→更正 TFPV |
| Sheffield / Matlock（服务终点） | 平台5 **DC5065**；Matlock 分支 DY572/DY571、912TC | 连续方向=North |
| pilot line | TC **TECS, TECV**；signal EC5487 / EC5484 |  |
| RTC North sidings | 出口信号 **TD5049** | |
| RTC South sidings | 出口信号 **TD5043**（有效✓）；~~TC TDWV~~ ⚠️AI核验：TDWV 在路由数据 **0 命中**→不用，C11 改用 TD5043 锚点 | |
| Litchurch Lane | **202pts**（平台3/4/Chad 已 gauge-cleared） | 出口信号DW5310 |
| Etches Park | **305pts** | EC5474，TECF和TECJ |
| Chaddesden Sidings | **EC5491/EC5493** 及其后一串 TC | |
| Sinfin branch | **DW5320**(Sinfin North) / DW5323(stop-await 牌) | |
| Matlock branch | **DY572/DY571 / 912TC / Ambergate** | |

---

# 第一组：硬规则 — preferred/non-preferred 路由（§5），confidence=high

### R1 — `S5-TD5045-platform4` ✅已批准(2026-05-27)
- source_section: §5（"Preferred and Non-preferred Routing"；zip-extract para~659 + 表区 para~423）
- cond_origin: East/Spondon（TD5045 进站，自 Spondon-Nottingham 侧下行）
- cond_destination: platform_4
- cond_train_class: any ｜ cond_time_of_day: None
- preferred_route_id: **RTD5045B-1(M)**
- preferred_platform: 4
- non_preferred_alternatives: **['RTD5045B-2(M)']**
- cond_other: "非首选触发：311pts(TNGK) 定位被锁或已设他进路 → 走 303 反位(TFPB)+307(TFMW)"
- confidence: high ｜ user_approved: **true**
- notes: "para659 'From TD5045 to platform 4 ... via 306pts and 311pts. non-preferred if 311pts ... via 303pts reverse and 307pts.' 四重核验：306=TDPA,311=TNGK;B-1 轨道含 TDPA+TNGK 终 TRJV(平台4);B-2 含 TFPB(303B)+TFMW(307) 不含 TNGK;303A=TDMZ 两路共用=303 定位/反位分叉。修正 spec §13.4 笔误(原写 A(M)/B(M))。"

---

# 第二组：硬规则 — 访问约束 / 唯一路由（§5/§6/§9），confidence=high

### C8 — `S5-TD5049-platform3or4only`
- source_section: §5（zip-extract para~426, ~434）
- cond_origin: RTC North sidings（出口信号 TD5049）
- cond_destination: platform_3 或 platform_4（**只此二者；不得进平台5**）
- cond_train_class: any ｜ cond_time_of_day: None
- preferred_route_id: None
- preferred_platform: None（约束=可达集合 {3,4}）
- non_preferred_alternatives: []
- cond_other: "TD5049 进平台3/4 得 MAF；**不能从此信号进平台5**；可 permissive；另有 headshunt 移动"
- confidence: high ｜ user_approved: ____
- notes: "para426 'TD5049 ... Trains from here can be signalled into platform 3 or 4 only ... MAF'；para434 'Trains cannot be signalled into platform 5 from this signal.'"
- 审核: 没问题

### C9 — `S5-platform1-north-aline`
- source_section: §5（zip-extract para~478）
- cond_origin: platform_1（TC: **TPSL/TPSM/TPSU**；北端出口 **DC5061**=往北、南端 **DW5301**=往 Pear Tree/南）
- cond_destination: North（出 Derby 北行经 DC5061；A 线=平台1 引出的 down fast）
- cond_train_class: any ｜ cond_time_of_day: None
- preferred_route_id: None（平台1 北向唯一路由=A 线/down fast；route_id 取自平台1经 DC5061 北向出站路由）
- preferred_platform: None
- non_preferred_alternatives: []
- cond_other: "平台1 出 Derby 往北：经 A 线(down fast) 为**唯一**路由（Hao 已确认：A 线即平台1 TPSL/TPSM/TPSU 经 DC5061 北向引出，无独立 A 线 TC，缺口已补）"
- confidence: high ｜ user_approved: **true**
- notes: "para478 'Trains signalled out of Derby along the A line which becomes the down fast is the preferred route (only route from platform 1)'。Hao 图示确认平台1 TC=TPSL/TPSM/TPSU、DC5061 往北、DW5301 往 Pear Tree。"
- 审核:可以 ✅approved（A 线缺口已由 Hao 图示补齐）

### C10 — `S9-rtcnorth-platform3or4`
- source_section: §9（zip-extract para~749）
- cond_origin: North
- cond_destination: RTC North sidings（出/入口锚 TD5049）
- cond_train_class: any ｜ cond_time_of_day: None
- preferred_route_id: None
- preferred_platform: None（约束=经平台 {3,4}）
- non_preferred_alternatives: []
- cond_other: "北向来车进 RTC North sidings 须经平台3或4"
- confidence: high ｜ user_approved: ____
- notes: "para749 'Access to the RTC North sidings by trains from the North is via platform 3 or 4.'"
- 审核:可以

### C11 — `S9-rtcsouth-access`
- source_section: §9（zip-extract para~751）
- cond_origin: any
- cond_destination: **RTC South sidings**（锚点=出口信号 **TD5043**；Hao 给的 TC "TDWV" 路由数据 0 命中→弃用，改用 TD5043）
- cond_train_class: any ｜ cond_time_of_day: None
- preferred_route_id: None
- preferred_platform: None（约束=可达集合 {pilot, 6, 5, 3, 4}）
- non_preferred_alternatives: []
- cond_other: "进 RTC South 可经 pilot line / 平台6 / 5 / 3 / 4"
- confidence: high ｜ user_approved: ____
- notes: "para751 'Access to the RTC sidings South can be via the pilot line; platform 6; platform 5; platform 3 or 4.' 检测锚点=出口信号 TD5043（Hao 给）；TC TDWV 经 AI 核验在路由数据 0 命中、弃用。"
- 审核:TD5043是RTC South sidings的出口信号 ✅approved（AI核验：TDWV 不存在，已改用 TD5043）

### C12 — `S9-platform5-depart-301`
- source_section: §9（zip-extract para~753）
- cond_origin: platform_5
- cond_destination: None（发车一般约束）
- cond_train_class: any ｜ cond_time_of_day: None
- preferred_route_id: None ｜ preferred_platform: None
- non_preferred_alternatives: []
- cond_other: "平台5 发车须沿 down main 反方向行至 301pts（合法但**较慢**）= 次选事实"
- confidence: high ｜ user_approved: ____
- notes: "para753 'Trains departing Derby from platform 5 have to run along the down main in the up direction until 301 points whilst this is valid movement it is a slower move.'"
- 审核:可以

### R4 — `S6-platform6-callon-north` ✅已批准
- source_section: §6（call-on，zip-extract para~711）
- cond_origin: North ｜ cond_destination: platform_6
- cond_train_class: any ｜ cond_time_of_day: None
- preferred_route_id: None ｜ preferred_platform: None
- non_preferred_alternatives: []
- cond_other: "自 DC5076 signalled 进**已占用**平台6，需先停车告知司机（信号视距限制）；**例外**：EC5486/EC5488 经 Chad curve 不适用"
- confidence: high ｜ user_approved: **true**
- notes: "para711；与 f_call_on stratum 直接相关"

### R5 — `S9-litchurch-platform3or4` ✅已批准
- source_section: §9（zip-extract para~757）
- cond_origin/destination: Litchurch Lane（Bombardier，经 202pts）
- cond_train_class: any ｜ cond_time_of_day: None
- preferred_route_id: None ｜ preferred_platform: None（约束=经平台 {3,4}）
- non_preferred_alternatives: []
- cond_other: "Litchurch Lane 进/出经平台3或4；平台3/4/Chad 已 gauge-cleared"
- confidence: high ｜ user_approved: **true**
- notes: "para757 'Trains for Litchurch Lane in or out are via platform 3 or 4...'"

---

# 第三组：硬规则 — 分支安全策略（§11/§14），confidence=high

### R6 — `S11-sinfin-singleline` ✅已批准
- source_section: §11（zip-extract para~775-779）
- cond_origin: any ｜ cond_destination: Sinfin branch
- cond_train_class: any ｜ cond_time_of_day: None
- preferred_route_id: None ｜ preferred_platform: None
- non_preferred_alternatives: []
- cond_other: "上分支后'自行其是'；**一次只许一列**（不得在另一列已 signalled 朝 DW5323 stop-await 牌时驶近 DW5320）；关键信号 DW5320(Sinfin North)；run-round loop standage=232m"
- confidence: high ｜ user_approved: **true**
- notes: "para779；已删草稿无出处的 '15 mph'"

### R7 — `S14-matlock-token` ✅已批准
- source_section: §14（zip-extract para~806-808；para~324 'Token Block'）
- cond_origin: any ｜ cond_destination: Matlock branch
- cond_train_class: any ｜ cond_time_of_day: None
- preferred_route_id: None ｜ preferred_platform: None
- non_preferred_alternatives: []
- cond_other: "No-Signaller Token(TS7)；token 在 Ambergate 由 912TC 占用释放；得 token 依 DY572 进往返；归还后才可从 DY571 设进路；Token Block 非 TCB"
- confidence: high ｜ user_approved: **true**
- notes: "para806-808"

---

# 第四组：软偏好 — §3 traffic-flow 平台偏好，confidence=med（只作参考、不计入 §12 闸）

### R2 — `S3-sheffieldmatlock-platform5` ✅已批准
- source_section: §3（zip-extract para~364）
- cond_origin: None ｜ cond_destination: Sheffield/Matlock（连续 North；锚 平台5 DC5065 / T884）
- cond_train_class: passenger ｜ cond_time_of_day: None
- preferred_route_id: None ｜ preferred_platform: **5**
- non_preferred_alternatives: []
- confidence: **med** ｜ user_approved: **true**
- notes: "para364 'trains to Sheffield and Matlock trains will use platform 5 predominantly' = 软偏好。⚠️与 C17(para370,→平台6) 冲突，见 C17 注。"

### R3 — `S3-west-platform2` ✅已批准
- source_section: §3（zip-extract para~370/379）
- cond_origin: None ｜ cond_destination: West（锚 DW5302 / 平台2 TC{TYTW,TYTV,TYTS}）
- cond_train_class: passenger ｜ cond_time_of_day: None
- preferred_route_id: None ｜ preferred_platform: **2**
- non_preferred_alternatives: ['platform_3','platform_4']
- confidence: **med** ｜ user_approved: **true**
- notes: "para370/379 'all passenger trains to the West will use platform 2 ... option to use platform 3 or 4'"

### C13 — `S3-north-fastest-platform5`
- source_section: §3（zip-extract para~376）
- cond_origin: None ｜ cond_destination: North（锚 Duffield T884）
- cond_train_class: any ｜ cond_time_of_day: None
- preferred_route_id: None ｜ preferred_platform: **5**（最快）
- non_preferred_alternatives: ['platform_3','platform_4','pilot_line','platform_6']（platform_6 更慢）
- confidence: **med** ｜ user_approved: ____
- notes: "para376 'For the North may be signalled through platform 3, 4, 5, or the pilot line ... quickest option is through platform 5, platform 6 ... slower move.'"
- 审核: 可以，要注意是软规则。

### C14 — `S3-south-optimum-platform6`
- source_section: §3（zip-extract para~382）
- cond_origin: None ｜ cond_destination: South（锚 Pear Tree DW5319 / TYVR）
- cond_train_class: any ｜ cond_time_of_day: None
- preferred_route_id: None ｜ preferred_platform: **6**（最优）
- non_preferred_alternatives: ['platform_3','platform_4','pilot_line']
- confidence: **med** ｜ user_approved: ____
- notes: "para382 'Trains to the South may be signalled ... via platform 3 or 4 or the pilot line however the optimum route is via platform 6.'"
- 审核:可以，要注意是软规则。

### C15a — `S3-north-passenger-platform1`
- source_section: §3（zip-extract para~367）
- cond_origin: None ｜ cond_destination: North（锚 Duffield T884）
- cond_train_class: passenger ｜ cond_time_of_day: None
- preferred_route_id: None ｜ preferred_platform: **1**
- non_preferred_alternatives: []
- confidence: **med** ｜ user_approved: ____
- notes: "para367 'trains for the North will use platform 1'。⚠️与 C13(North→平台5 最快) 并存——Plan 两段对 North 偏好不同(客车→1 vs 最快→5)；均软偏好、不计闸，重叠决策标'歧义'。"
- 审核:可以，要注意是软规则。

### C15b — `S3-nottingham-platform3or4`
- source_section: §3（zip-extract para~367）
- cond_origin: None ｜ cond_destination: Nottingham（锚 TD5030 / TFPV，Spondon 侧）
- cond_train_class: passenger ｜ cond_time_of_day: None
- preferred_route_id: None ｜ preferred_platform: None（偏好集合 {3,4}）
- non_preferred_alternatives: []
- confidence: **med** ｜ user_approved: ____
- notes: "para367 'Trains for Nottingham will use platform 3 or 4'"
- 审核:可以，要注意是软规则。

### C15c — `S3-crewe-layover-platform3b4b`
- source_section: §3（zip-extract para~367）
- cond_origin: None ｜ cond_destination: Crewe（锚 Pear Tree DW5319 / TYVR）
- cond_train_class: passenger ｜ cond_time_of_day: None
- preferred_route_id: None ｜ preferred_platform: None（偏好 3B 或 4B，layover）
- non_preferred_alternatives: []
- cond_other: "Crewe service layover 用平台 3B 或 4B（子段，TC：3B=TNGU/4B=TRJV）"
- confidence: **med** ｜ user_approved: ____
- notes: "para367 'The Crewe service will use either platform 3B or 4B for the layover.'"
- 审核:可以

### C16 — `S3-birmingham-platform3or4`
- source_section: §3（zip-extract para~364）
- cond_origin: None ｜ cond_destination: Birmingham（锚 Pear Tree DW5319 / TYVR）
- cond_train_class: passenger ｜ cond_time_of_day: None
- preferred_route_id: None ｜ preferred_platform: None（偏好集合 {3,4}）
- non_preferred_alternatives: []
- confidence: **med** ｜ user_approved: ____
- notes: "para364 'trains for Birmingham using platform 3 or 4'"
- 审核:可以

### C17 — `S3-sheffieldmatlocknorthern-platform6`
- source_section: §3（zip-extract para~370）
- cond_origin: None ｜ cond_destination: Sheffield/Matlock/Northern（连续 North）
- cond_train_class: passenger ｜ cond_time_of_day: None
- preferred_route_id: None ｜ preferred_platform: **6**
- non_preferred_alternatives: ['platform_3','platform_4']
- confidence: **med** ｜ user_approved: ____
- notes: "para370 'the services from Matlock and Sheffield services (and Northern) will use platform 6, however there is an option to use platform 3 or 4'。⚠️**与 R2 直接冲突**(R2 para364 说 Sheffield/Matlock→平台5)——Plan 两段本身矛盾。处理：两条均 med 软偏好、不计入 §12 闸；遇此二者同时匹配的决策，L4 标 'ambiguous/规则不适用' 而非 non-compliant（不妥协：不强行二选一）。"
- 审核:可以，要注意是软规则。

---

## 计数与诚实说明
- 本草稿共 **19 条**（R1–R7 已批 7 条 + C8–C17 新 12 条；其中 C15 拆成 a/b/c）。spec §13.1 估 80–120 条是**乐观估计**；Plan 里**显式可判定**的 preferred/access/branch 规则就这些，**不为凑数硬造**。若你要更细粒度（如把每个方向×时段、每条 §5 信号的 MAF/MAR 拆开），可再扩，但需你点头。
- **硬规则 9 条**（R1,C8–C12,R4,R5,R6,R7）+ **软偏好 10 条**（R2,R3,C13,C14,C15a/b/c,C16,C17）。
- **待补的检测缺口**（不阻塞审核，但影响 L4 能否逐决策匹配）：C-4 RTC South TC、C9 的 A 线 TC（D4）。其余锚点齐备。
- **Plan 自身矛盾**已显式标注（R2↔C17 平台5/6；C13↔C15a North 平台5/1）——按"软偏好+歧义标记"处理，绝不静默选一条。

## 给 Hao
1. 审 **C8–C17**（每条 `审核:` 填 approved/改/拒）。
2. 上面两处检测缺口（RTC South TC、A 线 TC）要补吗？不补就把 C11/C9 降为"文档规则、L4 不逐决策匹配"。
3. 计数 19 条够不够？要不要按方向×时段再细拆？
4. 审过后我写 `scripts/rules/03_finalize.py` → `rules.parquet`（只落 user_approved=true），再建 `l4_rules.py`（§13.5 load_rule_base + rule_matches + l4_check + l4_summary_per_cell）。
