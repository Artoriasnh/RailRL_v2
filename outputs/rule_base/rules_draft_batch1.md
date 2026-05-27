# L4 规则库 — 第一批草稿（校准用，待 Hao 逐条审核）

> 来源：`data/domain/Training_Plan_2022.docx`。schema 见 spec 05 §13.2。
> ⚠️ AI 起草、**必须 Hao 逐条审**（approved / 改 / 拒）后才入 `rules.parquet` 用于 L4。
> 本批是**校准批**：先确认格式 + route_id 解析方式，再批量起草全部 ~80–120 条。
> 审核请在每条的 `审核:` 行填 approved / 改(写明) / 拒。

> **核验状态（2026-05-27 更新）**：R1 已**三重核验**（Hao 点位→TC + route_to_tc_all 轨道串 + 词表命名），route_id 唯一锁定且校准问题①已闭环；R6 已按原文收紧并删除无出处的 "15 mph"；R7 原文核对无误。R2–R5 仍为初稿，待 Hao 逐条审。文末新增**剩余硬规则清单(R8+)**，请 Hao 先圈定范围再批量起草。

> **✅ 审核结论（2026-05-27，Hao 已逐条审）**：R1–R7 **全部通过**（R1 行；R2 行+给平台5锚点；R3 行+给平台2锚点；R4/R5/R6 没问题；R7 行）。校准答复：② 软偏好"不要太硬"→ 入库标 confidence=med、只作参考、不计入 §12 闸；③ 访问约束也做（可以）；④ 格式 OK；范围 = **C8–C17 全做**。
> **R1 第④重核验（Hao 补点位）**：303A=TDMZ、303B=TFPB、307=TFMW → B-1(preferred) 仅含 TDMZ(303A)=303 定位；B-2(non-preferred) 含 TFPB(303B)+TFMW(307)=303 反位+307。**与 para659 / para423 完全吻合，非 preferred 路径已逐点闭合。**
> **Hao 补充的 signal/TC 锚点**：平台5(Sheffield/Matlock)=DC5065 进出，TC{TDPG,TDPJ,TDPK}；平台2(West)=DW5302 进出，TC{TYTW,TYTV,TYTS}。
> **下一步**：方向/终点→signal/TC 仍多数缺映射 → 见同目录 `destinations_to_map.md`，Hao 回填后我把 R1–R7+C8–C17 全量起草成 schema 送终审，审过才写 `rules.parquet`。

---

### R1 — S5-TD5045-platform4（preferred/non-preferred 路由）
- source_section: §5 "Preferred and Non-preferred Routing"（zip-extract para ~659；§5 表区 para ~422-423 为第二处佐证）
- cond_origin: TD5045 ｜ cond_destination: platform_4 ｜ cond_train_class: any ｜ cond_time_of_day: None
- **preferred**: 经 306pts(TDPA) + 311pts(TNGK)（= 303pts 定位 / MAF）｜ **non_preferred_alternatives**: 当 311pts(TNGK) 定位被锁或已设他进路 → 经 303pts 反位 + 307pts（MAR 替代路径）
- preferred_route_id: **RTD5045B-1(M)** ｜ non_preferred_alternatives: **['RTD5045B-2(M)']**
- **三重核验（已确认，route_id 由轨道串唯一确定，不再依赖人工补点）**：
  - (1) Hao 点位→TC：**306=TDPA、311=TNGK**。
  - (2) route_to_tc_all.csv 轨道串：**RTD5045B-1(M)** = TDMZ→**TDPA**→TDPC→TRKC→**TNGK**→TNGM→TRKA→TRJY→TRJW→**TRJV(平台4B)**，含 306+311 ✓=preferred；**RTD5045B-2(M)** = TDMZ→TFPB→TFMY→TFMW→TDPB→TDPE→TRKC→TRKA→…→**TRJV(平台4B)**，**不含 TNGK**、走替代路径 ✓=non-preferred（311 锁时走它）。
  - (3) 命名核验：路由词表/parsers.py + route_to_tc_all.csv 均用 **R 前缀**形式 RTD5045B-1(M)/RTD5045B-2(M)，与本条 preferred_route_id 一致 ✓。
- 排除项：RTD5045A(M) 含 TDPA+TNGK 但终到 **TNGU=平台3** → 属平台3 路线、非本规则。**spec §13.4 示例写 preferred=A(M) 有误；"RTD5045B(M)"（无 -1/-2）词表不存在。**
- **第二处佐证（§5 表区 para~423）**：原文 "There are 2 routes into platform 4 from TD5045 MAF via 303 points normal, which is the preferred route, also available is MAR 303 points reverse this is non-preferred." → 与 para659 一致：**preferred = 303 定位(MAF)、non-preferred = 303 反位(MAR)**，两处独立互证。
- confidence: high ｜ notes: para659 原文 "From TD5045 to platform 4 there is a preferred and non-preferred route. The route is via 306pts and 311pts. The non-preferred route if 311pts are normal (either locked or by a route set through) then the route is via 303pts reverse and 307pts."
- 待 Hao 复核签核：B-1=preferred / B-2=non-preferred 是否通过（route_id 已由轨道串唯一锁定，**不再需要**补 303/307 点位）。
- 审核: 行，303点位有303A对应的TC为TDMZ, 303B的对应的TC是TFPB，307对应的TC是TFMW。如果还有需要我确认的点位和TC跟我说。

### R2 — S3-Sheffield/Matlock-platform5（traffic flow 平台偏好）
- source_section: §3（para 268）
- cond_origin: None ｜ cond_destination: Sheffield 或 Matlock 方向 ｜ cond_train_class: passenger
- preferred_platform: **5** ｜ preferred_route_id: None
- confidence: **med**（原文"envisaged ... predominantly" = 软偏好,非硬规则）
- notes: "trains to Sheffield and Matlock trains will use platform 5 predominantly"
- 审核:行，如果目标是Sheffield 或 Matlock 方向，那么platform 5 就是从DC5065出发的，或者是到DC5065。platform5的TC包含了TDPG,TDPJ,TDPK

### R3 — S3-West-platform2（traffic flow）
- source_section: §3（para 274）
- cond_origin: None ｜ cond_destination: West 方向 ｜ cond_train_class: passenger
- preferred_platform: **2** ｜ preferred_route_id: None
- confidence: med ｜ notes: "all passenger trains to the West will use platform 2"
- 审核:行，如果目标是West 方向，那么就是从DW5302出发，或者到DW5302，platform2的TC范围是，TYTW,TYTV,TYTS

### R4 — S6-platform6-from-North-callon（call-on/许可进路）
- source_section: §6（para 599）
- cond_origin: North ｜ cond_destination: platform_6 ｜ cond_other: "从 DC5076 signalled 进**已占用**平台,需先停车告知司机;**例外**:EC5486/EC5488 经 Chad curve 不适用"
- preferred_route_id: None（这是 call-on 条件,非 preferred 路由）
- confidence: high ｜ notes: 与 f_call_on stratum 直接相关
- 审核: 没问题

### R5 — S9-Litchurch-platform3or4
- source_section: §9（para 645）
- cond_origin/destination: Litchurch Lane（Bombardier）｜ cond_train_class: any
- preferred_platform: **3 或 4** ｜ preferred_route_id: None
- confidence: high ｜ notes: "Trains for Litchurch Lane in or out are via platform 3 or 4"（platform 3/4/Chad 已 gauge cleared）
- 审核: 没问题

### R6 — S11-Sinfin-single-line（分支安全策略）
- source_section: §11 "The Sinfin Branch"（zip-extract para ~775-779）
- cond_destination: Sinfin branch ｜ preferred_route_id: None（通用安全策略）
- cond_other: "列车 signalled 上 Sinfin 分支后'自行其是'(look after itself)；**不得**在另一列车已 signalled 朝 DW5323 stop-and-await 指示牌时驶近 DW5320（即分支上一次只许一列）；关键信号 DW5320(Sinfin North 站台,NR 维护)；run-round loop standage = 232m"
- confidence: high ｜ notes: 安全策略类(非 preferred 路由)。原文 para779 "Once train is signalled onto the Sinfin branch it will look after itself it is not permitted for the train to approach DW 5320 signal, with another train signal towards the stop and await instruction boards DW 5323."。**已删除草稿里未在原文找到的 "15 mph"（不妥协/将错就错：无出处不写）。**
- 审核: 没问题

### R7 — S14-Matlock-token（No-Signaller Token）
- source_section: §14 "The Matlock Branch"（zip-extract para ~806-808；另 para~324 "Track Circuit Block ... apart from the Matlock Branch which is Token Block"）
- cond_destination: Matlock branch ｜ preferred_route_id: None
- cond_other: "No-Signaller Token system(Rule Book TS7)；token 在 Ambergate 由 **912TC** 占用释放；得 token 后司机依 DY572 进分支往返；归还 token 后才可从 DY571 设进路；Token Block 非 TCB"
- confidence: high ｜ notes: 安全策略类。原文 para806-808 "The Matlock Branch is operated under the No-Signaller Token system (TS7 of the Rule Book). The token instrument at Ambergate is released upon occupation of 912 TC in Ambergate Station..."
- 审核:行

---

## 校准问题（请 Hao 定，定了我才能批量推进）

**① route_id 解析**：Plan 用"经 306pts+311pts"这种**点位**描述路线,但我们模型/数据里是 **route_id**(如 RTD5045A(M))。要填 `preferred_route_id`,我需要把点位↔route_id 对上。可行做法:用 `route_to_tc` / `Derby_info`(含每条 route 的 track_list/points)反查"哪条 route 经过 306+311pts"。**你确认走这条自动映射(我做、你抽查)?还是你直接告诉我关键路线的 route_id?**  前面已经回答了。

**② 软偏好(§3 traffic flow)要不要入库**：§3 多是"envisaged ... predominantly/all ... will use platform X"这类**软**平台偏好(confidence med)。L4 是"是否符合规则"的硬检查——软偏好算 non-compliant 会不会太严?**建议:软偏好单独标 confidence=med,L4 统计里 med 规则只作参考、不计入 §12 override 闸。你同意?** 不要太硬。

**③ 规则粒度/数量**：Plan 里**显式 preferred-route 很少**(§5 就 1 条),大量是 §6-§10 的**访问约束/事实**(如"Litchurch 经 3/4""RTC South 经 TD5045")。spec 说 80-120 条——**是否把这些访问约束也逐条做成规则**(能凑到几十条),还是只做明确的 preferred/non-preferred + 分支策略(更少但更硬)?**你定范围。**可以。

**④ 格式**：上面每条的字段/写法 OK 吗?要不要加字段、或改中文/英文？  没问题

---

## 剩余硬规则清单（R8+，待 Hao 圈定范围；尚未起草成正式规则）

> 下面是从 §3/§5/§9 原文里再抽出的、可做成**硬规则**的候选。我**先不**逐条 schema 化，等你圈定要哪些、以及软偏好怎么处理(校准问题②)，再批量起草并送审。每条标了 **硬/软** 与原文出处。

| 候选 | 内容 | 硬/软 | 原文出处(zip-extract para) |
|---|---|---|---|
| C8 | TD5049（RTC sidings North 出口）只能进 **平台3或4**；不能进平台5；进3/4 得 MAF | 硬(访问约束) | para~426, ~434 |
| C9 | 平台1 出 Derby 往北：唯一/首选 = 经 A 线(down fast) | 硬(唯一路由) | para~478 |
| C10 | 北向来车进 **RTC North sidings** → 经平台3或4 | 硬(访问约束) | para~749 |
| C11 | 进 **RTC South sidings** → 可经 pilot line / 平台6 / 5 / 3 / 4 | 硬(可达集合) | para~751 |
| C12 | 平台5 发车需沿 down main 反向行至 301pts（合法但较慢） | 硬(事实/次选) | para~753 |
| C13 | 往**北**：可走平台3/4/5/pilot；**最快=平台5**；平台6 可但慢 | 软(速度偏好) | para~376 |
| C14 | 往**南**：可经平台3/4/pilot；**最优=平台6** | 软(速度偏好) | para~382 |
| C15 | 往**北**的客车 → 平台1；Nottingham → 平台3/4；Crewe layover → 3B/4B | 软("envisaged") | para~367 |
| C16 | Birmingham → 平台3/4（与 R2 同段） | 软("envisaged") | para~364 |
| C17 | Sheffield/Matlock/Northern 服务 → 平台6（亦可3/4）（与 R3 同段对照） | 软("envisaged"+option) | para~370 |

说明：C8–C12 是**硬**访问约束/唯一路由（建议纳入）；C13–C17 是**软**速度/流量偏好（取决于校准问题②的结论）。R4(平台6 call-on)、R5(Litchurch→3/4) 已在上面的校准批里。

---

## 给 Hao 的下一步（请回我）

1. **R1 签核**：B-1(M)=preferred / B-2(M)=non-preferred，approved？（已三重核验，无需再补点位）
2. **R2–R7 逐条审**：在每条 `审核:` 行填 approved / 改 / 拒（R6 已删 15mph，R7 已对原文）。
3. **校准问题②（软偏好）**：§3 这类 "envisaged…predominantly/all…will use platform X" 要不要入库？建议：**入库但标 confidence=med、L4 只作参考、不计入 §12 override 闸**。同意？   可以
4. **范围**：C8–C12（硬）要不要全做？C13–C17（软）做不做（取决于②）？全做，做
5. **格式**：字段/写法 OK 吗？ok

你定了这 5 点，我再一次性把全部硬规则起草成 schema、连同 R1–R7 一起送你终审，审过才写 `rules.parquet` 并建 `l4_rules.py`。
