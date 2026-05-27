# 方向/终点 → signal/TC 映射清单（请 Hao 一一回填）

> 用途：L4 规则里很多 `cond_destination` 是**地理方向/终点**（如 "Sheffield 方向"、"West 方向"），模型状态里没有这种字段，只能靠**boundary signal / TC**来判定"这趟车是去哪个方向"。
> 下面把 Training Plan 里出现的全部命名方向/终点/侧线列出，**已知的我预填了**（标 `[已知]`），**其余请你在「signal / TC（Hao 填）」列回填**。判定口径建议：**该车被 signalled 朝向 / 终到某 boundary signal（或终到某站台 TC 集合）⟺ 属该方向**。
> 你已给的：West→DW5302、Sheffield/Matlock→DC5065（平台5）。已并入下表。
> 回填方式同规则草稿：在对应行后面直接写signal/TC即可；不确定的写"不确定/需查"。

---

## A. 罗盘方向（line-of-route，traffic-flow 规则主键）

| # | 方向 | 用在哪条规则 | 原文出处(para) | signal / TC（Hao 填） |
|---|---|---|---|---|
| A1 | **North**（往北） | C13(最快平台5)、C15(客车→平台1) | ~367, ~373, ~376 | 367, 373, 376 似乎已经超出了Derby的范围，可以确认的是Derby往北最近的一个station是DUFFIELD STATION, 对应的TC是T884（出Derby方向）和T883（入Derby方向）。 |
| A2 | **South**（往南） | C14(最优平台6) | ~382 | 382，应该也已经在Derby地区外了。 |
| A3 | **West**（往西） | R3(→平台2) | ~370, ~379 | `[已知]` DW5302（进出平台2；平台2 TC=TYTW,TYTV,TYTS） |
| A4 | **East / 往 Spondon-Nottingham 侧**（如适用） | — | TD5045 为该侧下行进站信号 | Spondon station往Derby方向的站台是TD5029对应TC是TDMC，往Nottingham的站台是TD5030对应TC是TFPY。 ⚠️AI核验：TFPY 在路由数据里**不存在**(0条)，正确应为 **TFPV**(见B4，在 RTD5032A(M) 中)；已按 TFPV 采用。 |

## B. 命名服务终点（service destinations）

| # | 终点 | 用在哪条规则 | 原文出处(para) | signal / TC（Hao 填） |
|---|---|---|---|---|
| B1 | **Sheffield** | R2(→平台5) | ~364, ~370 | `[已知]` DC5065（进出平台5；平台5 TC=TDPG,TDPJ,TDPK） |
| B2 | **Matlock**（分支，token） | R2(→平台5)、R7(token) | ~364, ~806 | `[已知]` 平台5=DC5065；分支 token 信号 DY572/DY571、912TC |
| B3 | **Birmingham** | C16(→平台3/4) | ~364 | 往peartree方向去birmingham，即DW5319, TYVR |
| B4 | **Nottingham** | C15(→平台3/4) | ~367 | 往spondon方向去nottingham，即TD5030, TFPV |
| B5 | **Crewe**（layover 3B/4B） | C15 | ~367 | 往peartree方向去crewe，即DW5319, TYVR |
| B6 | **Stenson**（往 Stenson Jcn） | — | ~392 | 同样是往peartree方向 |
| B7 | **Chesterfield**（经 Ambergate） | — | ~436 | 同样是往duffield方向 |
| B8 | **Burton** | 平台1 S 朝向 | platform_end_signals | `[已知]` DW5301（平台1 南端） |
| B9 | **Duffield** | 平台1 N 朝向 | platform_end_signals | `[已知]` DC5061（平台1 北端 5061） |
| B10 | **Barrow Hill / Erewash Valley**（gauge-cleared 北线） | R5 注 | ~757 | 同样是往duffield方向 |
| B11 | **South Wingfield**（Matlock 线沿途） | — | ~331 | 同样是往duffield方向 |
| B12 | **Ambergate (Jcn)**（往北 + token 机位） | R7 | ~390, ~806 | 912TC（token 释放） |

## C. 侧线 / 货场 / 分支 / 车辆段（sidings / yards / branches / depots）

| # | 终点 | 用在哪条规则 | 原文出处(para) | signal / TC（Hao 填） |
|---|---|---|---|---|
| C-1 | **Chaddesden Sidings (Chad)** | C9/C10 相关；R5 注 gauge | ~373, ~390 | EC5491/EC5493后面的一堆都是 |
| C-2 | **St Andrews Sidings** | §11 相关 | ~333, ~798 | DW5309（出 St Andrews 触发 train-waiting）、DW5321 |
| C-3 | **RTC North sidings** | C10(经平台3/4)；TD5049 出口 | ~426, ~749 | TD5049（出口信号） |
| C-4 | **RTC South sidings** | C11(经 pilot/6/5/3/4) | ~751 | |
| C-5 | **Litchurch Lane**（Bombardier 车厂，经 202pts） | R5(→平台3/4) | ~390, ~757 | 202pts；平台3/4/Chad gauge-cleared |
| C-6 | **Etches Park**（车辆段，经 305pts） | — | ~390 | 305pts |
| C-7 | **Sinfin branch**（卖给 Rolls-Royce，单线 token-like） | R6 | ~775 | `[已知]` DW5320(Sinfin North)、DW5323(stop-await 牌) |
| C-8 | **Matlock branch**（No-Signaller Token） | R7 | ~806 | `[已知]` DY572/DY571、912TC、Ambergate |
| C-9 | **sheet stores**（反向经 Tamworth slow） | — | ~390 | |

## D. 内部路由目标（不是地理终点，但规则会引用；多数已有 TC）

| # | 目标 | 说明 | signal / TC（Hao 填 / 确认） |
|---|---|---|---|
| D1 | **pilot line** | C11/C13/C14 备选路由 | 对应TC是TECS和TECV，signal EC5487和EC5484是两个不同方向的signal |
| D2 | **service platform** | TD5045 可达集合之一 | |
| D3 | **Tamworth 线（up/down fast/slow）** | 站南主干，平台1/2=fast、3/4=slow 对应 | up/down Tamworth fast↔平台1/2；slow↔平台3/4（请确认） |
| D4 | **A / B / C / D 线**（北端引出线） | C9 平台1→A线(down fast) | platform 1，signal DC5061往北，signal DQ5301往peartree方向。platform1的TC包括TPSL,TPSM,TPSU |
| D5 | **down Sunnyhill loop** | Sinfin 相关 | ~798 |

---

## 给 Hao：只要回填上面缺的 signal/TC（尤其 A1/A2 North·South、B3–B7、C-1/C-4/C-6/C-9），我就能把 R1–R7 + C8–C17 全量起草成 schema（硬规则 high、软偏好 med、不计入 §12 闸），连同已批的 7 条一起送你终审，审过才写 `rules.parquet` 并建 `l4_rules.py`。

> 注：标 `[已知]` 的来自 `data/reference/platform_end_signals.csv` / `platform_tc_map.csv` 或你已给的回复，请顺便核一下对不对；其余空格请回填或标"不确定"。
