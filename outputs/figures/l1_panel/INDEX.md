# L1 panel-schematic figures ！ index

Generated from `outputs/eval/l1_saliency.json` (seed 42, 12 example decisions across 7 strata).

**Layout convention** (self-drawn schematic, replaces manual `panel_layout.json`): platforms 1-6 as horizontal lanes (top to bottom), north (Duffield) ○ x ★ east (Spondon). TC adjacency derived from route track-lists. Hand-coded anchor positions for ~60 operational TCs/signals (platforms, L4-rule junction signals, branches); other TCs placed via BFS-radial fallback near their adjacent anchors.

## Per-decision figures

| sample_id | stratum | focal_train | focal_route | top-5 spatial nodes | figure |
|---|---|---|---|---|---|
| 1875901 | call_on | **** | `REC5488B(M)` | TNGS=0.40, T915=0.39, TGAR=0.37, 569=0.31, T909=0.20 | [html](./decision_1875901_call_on.html) ， [png](./decision_1875901_call_on.png) |
| 1875902 | priority_compete | **** | `REC5487C(C)` | TGAM=1.57, 5076=1.07, TEDA=0.96, 5045=0.80, TECV=0.55 | [html](./decision_1875902_priority_compete.html) ， [png](./decision_1875902_priority_compete.png) |
| 1875903 | advance | **** | `REC5484A(S)` | TECS=0.69, TGAS=0.20, TECR=0.11, TGAR=0.11, TEDC=0.07 | [html](./decision_1875903_advance.html) ， [png](./decision_1875903_advance.png) |
| 1875904 | advance | **** | `REC5484A(S)` | TECS=0.41, 5045=0.07, TECR=0.07, TECV=0.07, 5477=0.06 | [html](./decision_1875904_advance.html) ， [png](./decision_1875904_advance.png) |
| 1875910 | call_on | **** | `REC5488B(M)` | TNGS=1.58, TFDR=1.33, TEDC=0.78, TRKB=0.50, TNGR=0.45 | [html](./decision_1875910_call_on.html) ， [png](./decision_1875910_call_on.png) |
| 1875911 | priority_compete | **** | `RTD5045E(M)` | TFDY=1.01, TECV=0.93, TGAM=0.86, 5045=0.76, TFML=0.32 | [html](./decision_1875911_priority_compete.html) ， [png](./decision_1875911_priority_compete.png) |
| 469372 | trivial | 0D01 | `RDW5328B(M)` | TPRE=0.13, TPRJ=0.11, TPRG=0.09, 5324=0.08, TPRH=0.05 | [html](./decision_469372_trivial.html) ， [png](./decision_469372_trivial.png) |
| 469373 | trivial | 0D01 | `RDW5324A(M)` | TPRC=0.05, TPRJ=0.04, TPRM=0.03, TPRU=0.03, 5316=0.03 | [html](./decision_469373_trivial.html) ， [png](./decision_469373_trivial.png) |
| 883215 | late_train | 0D10 | `RTD5043A(S)` | TEDC=0.14, TRKG=0.13, TGAS=0.11, TFDR=0.08, TRKE=0.05 | [html](./decision_883215_late_train.html) ， [png](./decision_883215_late_train.png) |
| 883220 | late_train | 0D10 | `None` | TRJY=2.16, TFMW=1.57, TRJW=0.91, TNGS=0.86, TFML=0.77 | [html](./decision_883220_late_train.html) ， [png](./decision_883220_late_train.png) |
| 1883047 | platform_dev | 0R57 | `REC5488B(M)` | 5502=0.73, TGAJ=0.37, 5056=0.37, 5488=0.26, TEDA=0.18 | [html](./decision_1883047_platform_dev.html) ， [png](./decision_1883047_platform_dev.png) |
| 608284 | platform_dev | 0Z11 | `RTD5045E(M)` | TDMC=2.52, TDMF=1.53, 5029=0.45, TDMH=0.40, 5033=0.39 | [html](./decision_608284_platform_dev.html) ， [png](./decision_608284_platform_dev.png) |

## Aggregate panel

Interactive: [aggregate_panel.html](./aggregate_panel.html) ， Static: ![aggregate](aggregate_panel.png)

Mean IG saliency across all decisions, on the same schematic. Larger/redder = more consistently salient across decisions. The companion adjacency matrix:

![adjacency matrix](adjacency_matrix.png)

## Reusable data files

- `derby_layout.json` ！ hand-coded anchor positions (TC/signal ★ [x,y]).
- `tc_adjacency.json` ！ TC ★ list of adjacent TCs (from route track-lists).
- `decision_<sid>_<stratum>.json` ！ per-decision render data (positions, saliency_by_name, route_tcs, top route/train saliency).
- `aggregate_panel.json` ！ mean saliency + coverage per node across decisions.
