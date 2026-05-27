# L2 — per-decision explanations (seed42)

## [call_on] sample_id=1875901

```
Decision (****, route REC5488B(M)) at 2024-03-15 17:57:20:

Special-case context: f_call_on, f_priority_compete, f_unusual_id

Model deliberation (top-3 Q):
  - route REC5488B(M)      Q = -0.32  ⟵ chosen
  - route REC5488A(M)      Q = -0.67  (runner-up)
  - route REC5486A(M)      Q = -2.96

Q-gap decomposition (route REC5488B(M) vs route REC5488A(M)) = +0.35:
  - base (no informative input): +0.00
  - Route features     +0.76
  - Train features     -0.54
  - Subgraph state     +0.28
  - Schedule outlook   -0.25
  - Sequence summary   +0.14
  - Special flags      -0.04

Manual compliance (L4): N/A (rule base pending)
```

## [priority_compete] sample_id=1875902

```
Decision (****, route REC5487C(C)) at 2024-03-15 18:02:43:

Special-case context: f_priority_compete, f_unusual_id

Model deliberation (top-3 Q):
  - route REC5487C(C)      Q = -0.17  ⟵ chosen
  - route REC5487A(M)      Q = -0.78  (runner-up)
  - route REC5487B(M)      Q = -1.08

Q-gap decomposition (route REC5487C(C) vs route REC5487A(M)) = +0.61:
  - base (no informative input): +0.00
  - Train features     +0.77
  - Route features     -0.32
  - Sequence summary   +0.28
  - Subgraph state     -0.20
  - Schedule outlook   +0.06
  - Special flags      +0.02

Manual compliance (L4): N/A (rule base pending)
```

## [advance] sample_id=1875903

```
Decision (****, route REC5484A(S)) at 2024-03-18 17:39:37:

Special-case context: f_advance, f_priority_compete, f_unusual_id

Model deliberation (top-3 Q):
  - route REC5484A(S)      Q = +0.19  ⟵ chosen
  - route REC5484C(S)      Q = -1.26  (runner-up)
  - route REC5484B(S)      Q = -2.36

Q-gap decomposition (route REC5484A(S) vs route REC5484C(S)) = +1.45:
  - base (no informative input): +0.00
  - Sequence summary   -1.65
  - Route features     +1.51
  - Train features     +0.93
  - Subgraph state     +0.74
  - Schedule outlook   -0.08
  - Special flags      +0.00

Manual compliance (L4): N/A (rule base pending)
```

## [advance] sample_id=1875904

```
Decision (****, route REC5484A(S)) at 2024-03-23 23:17:56:

Special-case context: f_advance, f_unusual_id

Model deliberation (top-3 Q):
  - route REC5484A(S)      Q = +0.11  ⟵ chosen
  - route REC5484C(S)      Q = -3.12  (runner-up)
  - route REC5484B(S)      Q = -3.62

Q-gap decomposition (route REC5484A(S) vs route REC5484C(S)) = +3.23:
  - base (no informative input): +0.00
  - Route features     +1.96
  - Subgraph state     +1.25
  - Sequence summary   -1.09
  - Train features     +0.99
  - Special flags      +0.23
  - Schedule outlook   -0.10

Manual compliance (L4): N/A (rule base pending)
```

## [call_on] sample_id=1875910

```
Decision (****, route REC5488B(M)) at 2024-04-11 17:58:34:

Special-case context: f_call_on, f_priority_compete, f_unusual_id

Model deliberation (top-3 Q):
  - route REC5488B(M)      Q = -0.23  ⟵ chosen
  - route REC5488A(M)      Q = -0.43  (runner-up)
  - route REC5488B(C)      Q = -7.10

Q-gap decomposition (route REC5488B(M) vs route REC5488A(M)) = +0.20:
  - base (no informative input): +0.00
  - Route features     +0.80
  - Train features     -0.44
  - Sequence summary   +0.42
  - Subgraph state     -0.27
  - Schedule outlook   -0.20
  - Special flags      -0.12

Manual compliance (L4): N/A (rule base pending)
```

## [priority_compete] sample_id=1875911

```
Decision (****, route REC5487C(C)) at 2024-04-11 18:05:21:

Special-case context: f_priority_compete, f_unusual_id

Model deliberation (top-3 Q):
  - route REC5487A(M)      Q = +0.27  ⟵ chosen
  - route REC5487B(M)      Q = -0.33  (runner-up)
  - route REC5487C(M)      Q = -0.63

Q-gap decomposition (route REC5487A(M) vs route REC5487B(M)) = +0.59:
  - base (no informative input): +0.00
  - Sequence summary   +0.56
  - Subgraph state     +0.34
  - Train features     -0.26
  - Route features     -0.07
  - Special flags      +0.03
  - Schedule outlook   -0.02

Manual compliance (L4): N/A (rule base pending)
```

## [trivial] sample_id=469372

```
Decision (0D01, route RDW5328B(M)) at 2024-03-29 10:31:25:

Special-case context: none (trivial)

Model deliberation (top-3 Q):
  - route RDW5328B(M)      Q = -0.04  ⟵ chosen
  - route RDW5328A(M)      Q = -2.98  (runner-up)
  - wait                   Q = -20.42

Q-gap decomposition (route RDW5328B(M) vs route RDW5328A(M)) = +2.94:
  - base (no informative input): +0.00
  - Route features     +4.18
  - Sequence summary   -1.24
  - Train features     +0.37
  - Subgraph state     -0.29
  - Schedule outlook   -0.08
  - Special flags      +0.00

Manual compliance (L4): N/A (rule base pending)
```

## [trivial] sample_id=469373

```
Decision (0D01, route RDW5324A(M)) at 2024-03-29 10:32:44:

Special-case context: none (trivial)

Model deliberation (top-3 Q):
  - route RDW5324A(M)      Q = -0.77  ⟵ chosen
  - wait                   Q = -29.94  (runner-up)
  - action#2               Q = -1000000000.00

Q-gap decomposition (route RDW5324A(M) vs wait) = +29.17:
  - base (no informative input): -11.25
  - Subgraph state     +17.50
  - Sequence summary   +12.51
  - Route features     +9.45
  - Train features     +1.40
  - Schedule outlook   -0.44
  - Special flags      +0.00

Manual compliance (L4): N/A (rule base pending)
```

## [late_train] sample_id=883215

```
Decision (0D10, route RTD5043A(S)) at 2024-04-15 06:39:08:

Special-case context: f_advance, f_priority_compete, f_late_train

Model deliberation (top-3 Q):
  - route RTD5043A(S)      Q = -0.91  ⟵ chosen
  - wait                   Q = -19.38  (runner-up)
  - action#2               Q = -1000000000.00

Q-gap decomposition (route RTD5043A(S) vs wait) = +18.47:
  - base (no informative input): -10.61
  - Sequence summary   +24.80
  - Train features     +6.25
  - Special flags      -4.13
  - Route features     +2.98
  - Schedule outlook   -0.70
  - Subgraph state     -0.12

Manual compliance (L4): N/A (rule base pending)
```

## [late_train] sample_id=883220

```
Decision (0D10, route RTD5045E(M)) at 2024-04-15 06:41:07:

Special-case context: f_call_on, f_priority_compete, f_late_train

Model deliberation (top-3 Q):
  - route RTD5045A(M)      Q = -1.57  ⟵ chosen
  - route RTD5045C(M)      Q = -2.19  (runner-up)
  - route RTD5045E(M)      Q = -2.24

Q-gap decomposition (route RTD5045A(M) vs route RTD5045C(M)) = +0.62:
  - base (no informative input): +0.00
  - Sequence summary   +1.66
  - Train features     -1.64
  - Subgraph state     +0.50
  - Special flags      +0.25
  - Schedule outlook   -0.17
  - Route features     +0.02

Manual compliance (L4): N/A (rule base pending)
```

## [platform_dev] sample_id=1883047

```
Decision (0R57, route REC5488A(M)) at 2024-03-18 05:41:29:

Special-case context: f_platform_dev, f_priority_compete

Model deliberation (top-3 Q):
  - route REC5488B(M)      Q = -3.65  ⟵ chosen
  - route REC5488F(M)      Q = -4.28  (runner-up)
  - route REC5488C(M)      Q = -4.77

Q-gap decomposition (route REC5488B(M) vs route REC5488F(M)) = +0.63:
  - base (no informative input): +0.00
  - Route features     +2.22
  - Sequence summary   -1.43
  - Special flags      +0.21
  - Subgraph state     -0.13
  - Train features     -0.12
  - Schedule outlook   -0.12

Manual compliance (L4): N/A (rule base pending)
```

## [platform_dev] sample_id=608284

```
Decision (0Z11, route RTD5045E(M)) at 2024-04-01 16:43:05:

Special-case context: f_platform_dev

Model deliberation (top-3 Q):
  - route RTD5045E(M)      Q = -3.04  ⟵ chosen
  - route RTD5045C(M)      Q = -7.05  (runner-up)
  - route RTD5045A(M)      Q = -7.06

Q-gap decomposition (route RTD5045E(M) vs route RTD5045C(M)) = +4.01:
  - base (no informative input): +0.00
  - Sequence summary   +1.61
  - Train features     +0.87
  - Special flags      +0.63
  - Schedule outlook   +0.46
  - Route features     +0.42
  - Subgraph state     +0.03

Manual compliance (L4): N/A (rule base pending)
```
