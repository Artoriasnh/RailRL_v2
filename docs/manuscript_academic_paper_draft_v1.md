# Academic-paper manuscript draft v1

## Paper Configuration Record

| Item | Configuration |
|---|---|
| Manuscript type | Research article / applied AI system paper |
| Target journals | Primary: Expert Systems with Applications; alternative: IEEE Transactions on Intelligent Transportation Systems |
| Field | Railway traffic management; safety-critical intelligent decision support; offline reinforcement learning |
| Citation style | Numeric placeholder style for drafting; final style to be adapted to ESWA or IEEE T-ITS author guidelines |
| Language | English main text, with a Chinese working abstract for author review |
| Current evidence status | Full-data Conservative Q-Learning training verified (seed 42; seed 43 reproducibility check). Completed: non-learned baseline comparison, counterfactual simulator validation, safety-first divergence analysis, off-policy value evaluation, and decision/reward explanation layers (L2 Q-gap, L5 inverse RL). Pending: three-seed mean ± std (seed 44), learned baselines (behavioural cloning, IQL), and the attention/rule explanation layers (L1, L4) |
| Claim boundary | The evidence supports expert-level, safe, explainable replication that outperforms non-learned baselines and matches the signaller on overall return and delay while modestly reducing waiting. It does NOT claim super-human improvement, which is neither supported nor required. All quantitative claims are single-seed (42) pending multi-seed confirmation |

---

# An End-to-End Explainable Offline Reinforcement Learning Framework for Railway Route-Setting Decision Support

## Abstract

Railway route setting is a safety-critical traffic management activity in which signallers decide whether to wait or to set a feasible route for a train under changing infrastructure occupation, timetable intent and local operating constraints. Existing data-driven railway decision-support studies often begin from already structured datasets, while many railway optimisation studies rely on simplified state descriptions or simulated scenarios. This leaves a gap between live operational data acquisition and deployable intelligent decision support. This paper presents RailRL, an end-to-end framework that connects live railway operational feeds, traceable data acquisition, leak-safe decision reconstruction and offline reinforcement learning for route-setting decisions. The framework first converts heterogeneous Network Rail feeds into a structured empirical data basis through feed-specific acquisition, decoding and storage. It then reconstructs route-setting decision points as a Markov decision process with dynamic action sets, where each action is either `wait` or `(focal train, candidate route)`. Rewards combine realised delay change, route utilisation, headway risk and waiting cost, while strict leakage audits prevent future outcomes or answer variables from entering the model state. The proposed model combines a heterogeneous graph transformer for infrastructure and train state, a transformer encoder for recent signalling events and a per-action Q-network trained with Conservative Q-Learning. On 14 months of Derby workstation data, the current pipeline constructs 1,996,572 usable decision snapshots and passes systematic leakage checks. Full-data training reaches 0.957 set-only top-1 agreement with the signaller on a held-out test month and substantially outperforms non-learned baselines on the signaller's hardest decisions (for example 0.88–0.90 versus near-zero on call-on and platform-deviation cases). On an event-driven simulator validated against the realised record (occupancy and throughput rank correlations of 0.94 and 0.86), 0.0% of the model's divergences from the signaller are genuinely unsafe. Off-policy value evaluation finds the learned policy statistically indistinguishable from the signaller on total return and on delay while modestly reducing waiting, and inverse reinforcement learning shows that the signaller's routing prioritises delay — a priority the sparse engineered delay reward under-represented, which explains the model's delay-neutrality. The framework thus delivers expert-level, safe and explainable route-setting support that beats naive rules; results are single-seed (seed 42) pending three-seed confirmation, and improvement beyond expert level is neither claimed nor required.

**Keywords:** railway traffic management; route setting; offline reinforcement learning; Conservative Q-Learning; explainable AI; heterogeneous graph transformer; operational data acquisition.

## 中文摘要（工作版）

铁路进路设置是信号员在安全关键环境中持续进行的调度决策。信号员不仅需要决定为列车设置哪一条进路，还需要判断是否应立即办理、是否应等待前车出清、以及在多列车竞争资源时如何体现优先级。本文提出 RailRL，一个从真实铁路运营数据获取到可解释离线强化学习决策支持的端到端框架。该框架首先通过面向 Network Rail 多源运营 feed 的采集、解码和结构化存储流程，形成可追溯的数据基础；随后将 Derby 工作站 14 个月数据重构为泄露审计通过的 MDP，包括动态动作集合、时间局部 episode、候选进路、奖励函数和状态表示；最后使用异构图 Transformer、事件序列 Transformer 和 per-action Q 网络，并以 Conservative Q-Learning 训练离线策略。当前管线生成 1,996,572 个可用决策 snapshot 并通过泄露审计。全量训练在留出测试月达到 set-only top-1 一致率 0.957，并在信号员最难的决策（如 call-on、平台偏离）上大幅超过非学习 baseline（约 0.88–0.90 对近 0）。在用真实记录校准的事件驱动模拟器（占用与吞吐秩相关 0.94 与 0.86）上，模型相对信号员的偏离中 genuine-unsafe 占 0.0%。离线策略价值评估显示模型在总回报与延误上与信号员统计无异、并小幅减少等待；逆强化学习显示信号员的选路把准点放在首位，而稀疏的工程化延误奖励未能充分体现这一点，解释了模型为何延误中性。该框架因此提供专家级、安全、可解释且胜过朴素规则的进路设置支持；结果为单 seed（42），待三 seed 确认，本文不主张也不需要超越专家水平。

---

## 1. Introduction

Railway signalling decisions are local, time-dependent and safety-critical. In a busy workstation, a signaller observes train movements, track-circuit occupation, signal states, route availability, timetable intent and local operating conditions. A route-setting action is therefore not a simple route classification problem. At each decision moment, the signaller must decide whether to set a route now, which feasible route to set, or whether to wait because immediate action may create operational conflict or reduce downstream performance.

This decision structure creates a difficult modelling problem. Classical optimisation and rescheduling approaches can produce high-quality solutions when the state, objective and constraints are formalised in advance. However, route-setting practice also contains local knowledge, tacit operational judgement and short-horizon control behaviour that are difficult to encode completely by hand. Behavioural learning from historical signaller actions is attractive because it learns from real operations. Yet pure imitation is also insufficient: historical actions are demonstrations, not proof of optimality. A decision-support model should therefore learn the structure of signaller behaviour while preserving the possibility of identifying actions that may improve operational outcomes.

A second difficulty is data provenance. Route-setting decisions are embedded in heterogeneous railway data streams. Train Describer messages provide signalling and infrastructure events; Train Movement records describe train-level operational progression; timetable and planning feeds provide intended movement context; and performance feeds record realised outcomes. These records differ in temporal resolution, operational meaning and analytical use. If a railway AI system starts only from a cleaned table, the link between live operations, data interpretation and model input is weakened. For safety-critical decision support, this link is part of the method.

A third difficulty is information leakage. In railway route setting, the variables needed for labels and rewards are often close to the answer. The focal signal, chosen route, realised delay change and route outcome are necessary for reconstructing decisions and evaluating outcomes, but they must not be passed as model state. Likewise, future realised events may be valid for hindsight reward calculation but invalid as input features. Without strict separation between sample metadata, state features and reward labels, high validation accuracy may reflect leakage rather than learning.

This paper addresses these challenges with RailRL, an end-to-end explainable offline reinforcement learning framework for railway route-setting decision support. The framework integrates four layers: a live data acquisition and traceability layer; a decision reconstruction layer that converts operational data into route-setting snapshots; an offline RL layer that learns Q-values over dynamic structured actions; and an evaluation and explanation layer designed to separate imitation from operational improvement. The current study focuses on the Derby workstation, using 14 months of Network Rail operational data.

The paper makes four contributions. First, it describes a traceable acquisition framework for converting live heterogeneous railway feeds into a structured research data basis. Second, it formulates railway route setting as a leak-audited offline RL problem with dynamic action sets and observed transitions. Third, it proposes a graph-sequence neural architecture that combines local infrastructure state, recent signalling events and schedule context. Fourth, it defines an evaluation pathway that distinguishes behavioural replication from counterfactual improvement, which is required before making claims about decision-support value.

## 2. Related Work

### 2.1 Railway traffic management and decision support

Railway traffic management has traditionally been studied through optimisation, rescheduling, simulation and decision-support methods. These approaches formalise conflicts, delay objectives and resource constraints, and they can provide strong performance when the operational setting is well specified. However, practical railway control also depends on local operating knowledge and context-sensitive trade-offs [3,4]. In route setting, decisions are often made under incomplete or rapidly changing information, where the signaller must balance safety constraints, punctuality, train priority, headway and platform use.

The present work differs from classical rescheduling in its decision granularity. Rather than optimising a complete timetable adjustment, it models the local route-setting choices made at individual signalling decision points. This granularity makes it possible to learn from the operational traces of signallers while still evaluating whether alternative actions may have improved local outcomes.

### 2.2 Operational railway data and traceability

Railway organisations increasingly generate large volumes of operational data, but data quality and interpretability remain major concerns [1]. For AI systems, traceability is also central to trust: the data, transformations and artefacts used to build a model should be accountable and reproducible [2]. These concerns are amplified in railway signalling because raw messages are heterogeneous and often require local reference knowledge before they can be interpreted.

This study therefore treats data acquisition as part of the research contribution. The upstream collector does not merely store raw messages. It performs feed-specific acquisition, decoding, monitoring and structured preservation, thereby maintaining a link between live feed context and downstream decision modelling.

### 2.3 Offline reinforcement learning for decision support

Offline reinforcement learning learns policies from fixed historical datasets without online interaction. This is appropriate for safety-critical railway applications, where exploratory online trial-and-error is not acceptable. Conservative Q-Learning is particularly relevant because it penalises unsupported high Q-values and is designed to reduce overestimation for actions not well covered by the behaviour data [11]. In the route-setting setting, this conservative property is important because the model scores alternative actions that were available but not necessarily taken.

The present framework uses offline RL not to replace domain constraints, but to learn action values over a structured operational action set. The model is trained only on observed transitions. Counterfactual simulation is reserved for evaluation and explanation.

### 2.4 Explainable AI in safety-critical transport

Safety-critical AI requires more than predictive performance. Explanations must support validation, verification and operator trust [5,6]. In railway signalling, an explanation should identify the operational context behind a recommendation, compare it with feasible alternatives and relate it to rules or system consequences. The planned RailRL explanation stack therefore includes model-level attribution, decision-level Q-gap analysis, counterfactual operational evaluation, rule compliance and reward trade-off interpretation. This multi-layer design is intended to support both technical validation and operational review.

## 3. Data Description and Acquisition Framework

### 3.1 Data sources

The empirical dataset is built from Network Rail operational feeds and static infrastructure references for the Derby workstation. The main feed types are Train Describer (TD), Train Movements, Very Short-Term Planning (VSTP), schedule data and RTPPM-related performance records. TD data provide high-frequency signalling and infrastructure events, including berth movements, track-circuit states and signalling-related messages. Train Movement data provide train-level progression and performance records. Planning and timetable feeds provide intended movement context, and performance records support outcome interpretation.

These feeds are complementary. A Panel Request indicates that a route was requested, but it does not by itself define the full decision context. The state also depends on recent signalling events, train positions, route availability, timetable expectations and downstream realised outcomes. The dataset therefore requires integration across feed types while preserving their distinct meanings.

### 3.2 Acquisition and storage framework

The acquisition framework uses a broker-based subscription layer and an application-based collector. Before a collection session begins, the collector validates feed access credentials and database connectivity. During acquisition, the user can select feed categories and observe runtime state, message counts, active subscriptions and recent errors. This design supports long-running collection, which is necessary because the dataset is assembled from live operational feeds rather than from a single static export.

The collector also includes continuity-oriented recovery. Heartbeat monitoring detects broker disruption, guarded cleanup closes incomplete sessions, bounded retry avoids uncontrolled reconnection loops and subscription recovery restores feed selection after reconnection. Durable subscription options can strengthen continuity when required. These mechanisms improve acquisition robustness and make interruptions visible in logs.

After receipt, messages are handled according to feed type and written to persistent structured storage. Selected TD messages are decoded using locally supplied signalling reference information, because their operational meaning depends on area-specific knowledge. This feed-specific handling preserves both the original feed context and the structured interpretation needed for later analysis.

### 3.3 Processed Derby decision dataset

The downstream RailRL pipeline converts the acquired data into route-setting decision samples. The current processed dataset contains 1,999,623 generated decision points, including 546,418 set decisions and 1,453,205 wait decisions before snapshot-level exclusions. After excluding cases where the focal train's current track circuit cannot be determined, the final snapshot table contains 1,996,572 usable decision snapshots.

Each snapshot contains sample metadata, candidate actions, reward fields, nested graph state, event tokens, schedule outlook and special-case flags. The final canonical file is ordered by corrected episode index and position. The final split contains 1,472,064 training snapshots, 186,145 validation snapshots and 338,363 test snapshots. The split is time-based and episode-local, with training before 2024-02-01, validation before 2024-03-01 and testing from 2024-03-01 onward.

## 4. MDP Reconstruction

### 4.1 Decision point definition

A decision point is defined as `(focal_train, focal_signal, t)`. It asks whether a route should be set for the focal train from the focal signal at time `t`, or whether the appropriate action is to wait. Set decision points are generated from observed Panel Request events. Wait decision points are generated from approach events where a train enters the approach horizon of a relevant signal and no corresponding Panel Request occurs within the configured short look-ahead window.

### 4.2 Dynamic action set

The action set is:

```text
A_t = {wait} union {(focal_train, R) | R in candidates(focal_train, focal_signal, t)}.
```

Action index 0 always denotes wait. Route-setting actions are indexed from 1 to `K`, where `K` is the number of candidate routes. Candidate routes are generated from static route definitions and time-local operational context. The route must start from the focal signal, direction consistency is applied when inferable, and routes already set for the same train are excluded. Planned platform information is used as a soft ordering signal rather than as a hard filter.

This design avoids fixed global route classification. The model scores only the routes that are feasible at a given decision point, plus the wait action.

### 4.3 State representation and leakage control

The model state includes a local heterogeneous graph, a recent event sequence, schedule outlook and special-case flags. The graph is centred on the focal train's current track circuit, not the focal signal. This design prevents the focal signal from being embedded as an answer-like state feature. Exactly one train node is marked as focal; focal signal and focal route markers are forbidden.

The event sequence contains the most recent TD tokens before the decision time. Schedule outlook uses planned information only. Reward intermediates, chosen action labels, future realised outcomes and answer-like fields are banned from state. Leakage audits check this separation directly.

### 4.4 Reward design

The reward combines delay, throughput, headway and waiting components:

```text
r_total = 1.0 r_delay + 0.5 r_throughput + 1.0 r_headway + 0.3 r_wait.
```

`r_delay` is based on realised delay change, clipped to 30 minutes, scaled into minutes and weighted by an approach-distance causal gate. `r_throughput` rewards route requests that were used and penalises unused or cancelled requests. `r_headway` penalises measured headways below the empirical threshold of 147 seconds. `r_wait` applies a fixed raw penalty to wait actions. For wait decisions, headway and throughput components are zero, while delay and wait components remain applicable.

Reward construction uses hindsight information, which is valid for return calculation. The same variables are forbidden from state input.

### 4.5 Episode construction

Episodes represent time-local passages of a train through the Derby workstation. A critical data issue was identified in early episode construction: train identifiers can be reused across months, which caused cross-month episodes and split leakage. The corrected procedure splits by focal train, a two-hour gap threshold and train/validation/test boundaries. This produces time-local episodes and prevents transitions from crossing split boundaries.

## 5. Model and Training Method

### 5.1 Neural architecture

The model has three representation branches. A heterogeneous graph transformer encodes track, signal, route and train nodes. A transformer sequence encoder processes recent TD event tokens. A fusion module combines graph summaries, the focal-train embedding, event-sequence summaries, schedule context, special flags and candidate-set size into a state embedding.

The Q-network is action-structured. For each route candidate, it combines the focal-train embedding, route embedding and fused state embedding. The wait action is scored by a separate MLP using the focal train, event summary and fused state. Invalid padded actions are masked.

Two auxiliary heads support representation learning. The route head predicts the historical route among candidates for set decisions. The time head predicts a calibrated five-bucket lead-time label. A separate priority head is not used; priority is represented through the Q-values assigned to wait and set actions across competing decision points.

### 5.2 Conservative offline RL

The main training objective is Conservative Q-Learning:

```text
L_CQL = L_TD + alpha L_cons, alpha = 5.0, gamma = 0.95.
```

The TD target uses the target network's maximum Q-value for the observed next state, with bootstrapping disabled for terminal transitions. The conservative term penalises high Q-values over the valid action set relative to the demonstrated action. This discourages overconfident values for poorly supported actions.

Training follows three phases. Phase A trains the encoder and auxiliary heads. Phase B freezes the encoder and trains the Q-function with CQL. Phase C jointly trains CQL and auxiliary losses. The optimiser is AdamW with warmup-to-cosine scheduling, gradient clipping and Polyak target updates.

### 5.3 Streaming transition loader

The dataset is too large and nested for inefficient row-wise loading. The final loader streams from the canonical snapshot file, where adjacent rows in the same episode define `(s, s')`. It performs block shuffling and approximate stratified sampling by operational case. This design enables full-dataset training without materialising all transitions in memory.

## 6. Experimental Design

### 6.1 Data validity checks

The first experiment layer verifies the dataset before interpreting model performance. The checks include snapshot construction audits, split leakage checks, banned-field leakage scans, temporal-order checks and baseline shortcut analysis. These checks are necessary because high validation accuracy could otherwise be caused by answer leakage, candidate ordering or future information.

### 6.2 Model training and sanity gates

Training proceeds in two stages. A 50,000-sample-per-epoch sanity run (seed 42) first verifies the training loop, representation learning, Q-value stability and phase gates. Full-data training is then run on the complete training set; the results below are for seed 42 with seed 43 as a reproducibility check, while seed 44, required for a three-seed mean ± standard deviation, is pending.

### 6.3 Baselines

The final evaluation should include:

1. random valid action;
2. first-come-first-served or timetable-rule baseline;
3. flat behavioural cloning MLP;
4. structured behavioural cloning;
5. CQL without special flags;
6. full CQL model.

All baselines should use the same time split and candidate action sets.

### 6.4 Metrics

The first metric group measures behavioural agreement: action top-1 agreement, route accuracy on set decisions, wait/set F1 and timing bucket accuracy. The second group measures stratified performance across late train, advance routing, call-on, platform deviation, priority competition, unusual train identifier and trivial cases. The third group will measure operational value through counterfactual evaluation, including reward delta, delay component, headway component and divergent-unsafe rate.

## 7. Results

All results below are from a single training seed (seed 42) on the full corrected dataset, with a second seed (seed 43) used as a reproducibility check; full three-seed mean ± standard deviation (seed 44) remains pending and all claims are stated accordingly.

### 7.1 Data reconstruction

The pipeline builds 1,996,572 usable decision snapshots, skipping 3,051 samples with no determinable focal-train track circuit and recording zero snapshot audit failures in the sampled build audit. The final canonical snapshot file includes corrected episode labels, rewards, split labels and patched lateness and platform-deviation features.

### 7.2 Reward correction and summary

Two systematic delay-attribution defects were identified and corrected before training. First, train identifiers are reused across months, which caused the delay-change computation to match decisions to the wrong monthly run and discard most pairs as out-of-window. Second, the Movements timestamps for April–July 2023 were offset by one hour (a double-applied British Summer Time correction at the source), which mis-aligned roughly 41% of the training-period decisions against the Train Describer clock. Both were repaired at source, and the rewards, the Movements-derived state fields and the affected snapshots were recomputed, after which a five-section pre-retraining audit (structure, reward–label agreement, state integrity, unchanged out-of-window rows, and a per-month anomaly scan) passed.

After correction, the total reward has mean −0.106 and standard deviation 0.587, with weighted component means of −0.010 for delay, +0.136 for throughput, −0.013 for headway and −0.218 for wait, where `r_total` is exactly the sum of the four weighted components. Delay-change coverage rose to 685,715 decisions (about 34%) and became uniform across months. A property that proves important for the later analysis is that, even after correction, the delay component remains small and sparse relative to the dense and large waiting and throughput components: the effective contribution of delay to the realised return is on the order of one tenth, despite its nominal unit weight.

### 7.3 Leakage audit

The leakage audit excludes direct answer fields, reward intermediates and forbidden focal markers from the model state. A separate baseline analysis indicates that high route accuracy is not explained by trivial candidate-position baselines, supporting the interpretation that the model learns state-to-decision structure rather than exploiting a shortcut.

### 7.4 Full offline RL training

Full-dataset Conservative Q-Learning training on the corrected data passes every predefined Stage-§11 gate. Phase A reaches validation route accuracy 0.925 and time accuracy 0.699 with the auxiliary losses falling to the required fraction of their initial values. Phase B reaches validation action top-1 agreement 0.963 with bounded Q-values and a conservative loss far below threshold. Phase C reaches validation action top-1 agreement 0.981, with the best validation action top-1 agreement of 0.982. The independent seed 43 reproduces this closely (best 0.983), and its phase-C Q-magnitude is in fact more tightly bounded than seed 42, indicating that the result is not seed-specific. Losses decrease monotonically across phases with no numerical instability.

### 7.5 Behavioural fidelity on the test split

On the held-out test split of 338,363 decisions, action top-1 agreement with the signaller is 0.988 over all decisions and 0.957 on set (route-choice) decisions; the all-decisions figure is inflated by the large wait majority, so the set-only figure is the honest measure of routing fidelity. The model reproduces the signaller's wait/set propensity almost exactly (0.727 versus 0.727), and the route head agrees with the historical route on 0.950 of set decisions. These figures indicate high-fidelity replication of signaller behaviour, consistent with the near-first-come-first-served regularity of routine route setting.

### 7.6 Stratified comparison against non-learned baselines

Table 3 reports set-only top-1 agreement by operational case for the model and three non-learned baselines: a uniform-random valid action, a planned-platform-preferring heuristic, and a first-candidate (greedy) heuristic. Because the baselines differ in their wait/set propensity, the all-decisions figure is not comparable across methods; the set-only figure isolates the routing decision. The model dominates the baselines overall (0.957 versus at most 0.531) and, crucially, on the hard strata where the signaller departs from the default route. On call-on and platform-deviation decisions the model reaches 0.881 and 0.903, while both heuristics fall to 0.048 and 0.000 respectively — below the random baseline — because the signaller's correct action on these strata is precisely the non-default one that a default-routing rule systematically misses. This stratified gap, rather than the near-saturated trivial cases, is the core evidence that the learned model captures decision skill that simple rules do not.

### 7.7 Counterfactual simulator validation

The event-driven P2.6 simulator, used only for evaluation and never for training, was validated on a held-out month against the actual Train Describer record. After calibrating the minimum-headway percentile, the simulator reaches a per-track-circuit occupancy-onset rank correlation of 0.94 and a throughput rank correlation of 0.86 with the realised data. It is conservative in absolute throughput (about 73% of the realised count, because the slowest trains do not complete within the horizon), but the relative dynamics required for counterfactual comparisons are validated, and this absolute bias cancels in action-versus-action deltas.

### 7.8 Safety-first divergence analysis

The model diverges from the signaller on only 4.3% of set decisions. Divergences are classified safety-first using simulator-independent signals — route legality and feasibility of the route when the focal train is simulated alone — because a fixed-others counterfactual cannot fairly adjudicate conflict-safety: every divergence that fails to complete with the other trains held fixed completes when the focal train is run in isolation, showing that such "failures" reflect the de-confliction asymmetry of the frozen world rather than an unsafe route. Under this analysis the genuine-unsafe divergence rate is 0.0%: every route the model proposes is a legal candidate that completes feasibly. A direct conflict-load measure (the change in headway-wait events introduced by the model's route relative to the signaller's) is essentially zero (mean ≈ +0.07), so the model introduces no meaningful additional inter-train conflict. The model's divergent routes are intrinsically about 14 seconds longer in free-flow traversal.

### 7.9 Off-policy value of the learned policy

Fitted Q-Evaluation on the logged trajectories estimates the value of following the model's policy relative to the signaller's, reported as a discounted-return difference ΔV with an episode-clustered 95% confidence interval (Table 4). On total return the policies are statistically indistinguishable (ΔV ≈ 0), so the model matches the expert overall. A reward-component decomposition that is internally consistent (the component differences sum to the total difference) shows where the policies differ: the model significantly reduces waiting (ΔV_wait = +0.035, CI [+0.027, +0.044]) at a small throughput cost (−0.012, CI [−0.018, −0.007]), and is statistically neutral on delay (ΔV_delay = +0.020, CI [−0.040, +0.070]) and on headway. Reconciling this with the +14-second free-flow figure of the divergence analysis, the model selects routes that are slightly longer in isolation but less congested in context, trading marginal path length for reduced waiting and netting no change in realised delay. These estimates carry the standard offline-RL caveat that the evaluator must extrapolate to the small fraction of out-of-distribution divergent actions and may be optimistic there; they are single-seed.

### 7.10 Decision- and reward-level explanation

The decision-level layer decomposes the Q-gap between the model's chosen route and the runner-up into contributions from six feature groups using exact Shapley values, which satisfy completeness exactly. The gap is dominated by route features, as expected, because the two competing actions share the same train, infrastructure, event and flag context and differ only in the route itself; the shared-context groups therefore influence the absolute action values more than the route-versus-route gap.

The reward-recovery layer applies maximum-entropy inverse reinforcement learning to the signaller's route choices, using behaviour-policy component action-values as features. Restricted to route-choice decisions — the well-posed setting, since including the dominant wait action confounds the recovery — the recovered weights place the largest positive weight on delay, followed by throughput. In other words, the signaller's revealed routing priorities lead with punctuality. This corroborates the off-policy finding from the opposite direction: the signaller prioritises delay, but because the engineered reward represented delay only sparsely and weakly, the trained model did not learn to prioritise it and is consequently delay-neutral rather than delay-improving. The individual recovered weights are reported qualitatively only: the headway weight in particular is not interpretable in isolation because the component value-features are collinear, and the recovery inherits the same out-of-distribution caveat as the off-policy evaluation.

### 7.11 Summary of findings

Taken together, the single-seed evidence describes the model as a high-fidelity (0.957 set-only), safe (0.0% genuine-unsafe, conflict-neutral) replicator of the expert signaller that matches the human on overall return and on delay while modestly reducing waiting, and that substantially outperforms non-learned baselines on the signaller's hardest decisions. Two independent estimators — off-policy value evaluation and inverse reinforcement learning — converge on a coherent account of the delay objective: the signaller prioritises it, the engineered reward under-represented it, and the model is therefore delay-neutral rather than delay-improving. The headline claim is thus expert-level, safe, explainable replication that beats naive baselines, not super-human improvement; the latter is neither supported nor required by the present evidence.

## 8. Discussion

The current evidence supports three conclusions. First, live railway operational feeds can be transformed into a traceable decision dataset suitable for offline reinforcement learning. This is a non-trivial contribution because acquisition, decoding, storage, decision reconstruction and leakage control all affect the validity of the final learning task.

Second, route setting can be represented as a dynamic-action MDP. The wait action and feasible route-setting actions are placed in the same action set, which allows timing and route choice to be learned jointly. This formulation better matches signaller practice than a fixed global classifier.

Third, under full-data conservative offline RL the model replicates historical route setting with high fidelity (0.957 set-only) and, more importantly, does so on the signaller's hardest decisions where naive default-routing rules fail. The stratified comparison locates the model's value precisely in the non-routine cases — call-on, platform deviation, priority competition — rather than in the near-saturated trivial cases.

The counterfactual and value analyses sharpen this into an honest claim. The model matches the signaller on overall return and is statistically neutral on delay while modestly reducing waiting, and none of its divergences are genuinely unsafe. It is therefore best characterised as expert-level, safe replication rather than super-human improvement. The inverse-RL and off-policy analyses jointly explain why the model does not improve delay: the signaller's revealed routing priorities lead with punctuality, but the engineered reward represented delay only sparsely and weakly, so the model optimised the dense waiting and throughput signals and treated delay as near-noise. This is a reward-design finding rather than a model failure; materially improving delay would require a denser, episodically-credited or re-scaled delay reward and re-training, not a change of architecture.

Several limitations remain. The results are single-seed (seed 42) with a second seed as a reproducibility check; full three-seed mean ± standard deviation is pending. Learned baselines (behavioural cloning, IQL) and two explanation layers (attention/saliency and rule-compliance) are not yet implemented. The off-policy and inverse-RL estimates rely on a fitted evaluator extrapolating to counterfactual actions — the standard offline-RL out-of-distribution caveat — and the recovered reward weights are reported qualitatively only. Platform-deviation detection is conservative because route-to-platform mapping is available for only a subset of routes. The dataset covers the Derby workstation, so transfer to other areas would require new local decoding and infrastructure mapping.

## 9. Conclusion

This paper presents RailRL, an end-to-end framework for explainable offline reinforcement learning in railway route-setting decision support. The framework connects live operational data acquisition with leak-safe MDP reconstruction, structured-action Conservative Q-Learning and a planned multi-level explanation stack. On 14 months of Derby workstation data, the current pipeline constructs nearly two million usable snapshots and passes data validity, leakage and sanity-training checks. These results support the feasibility of traceable offline RL for railway route-setting behaviour modelling. Final claims about improvement over historical signaller behaviour require the pending full-data multi-seed training, baseline comparison and counterfactual operational evaluation.

## Declarations

### Data availability

The raw operational feeds are obtained from Network Rail Open Data sources and local acquisition records. Processed research artefacts are stored within the RailRL project workspace. Public release of processed data may be restricted by licensing, operational sensitivity and data provenance considerations. A final submission should provide a precise data availability statement after confirming what can be shared.

### Code availability

The implementation is maintained in the RailRL v2 project repository. A final submission should identify which scripts, trained checkpoints and derived artefacts can be archived or shared.

### Funding

[To be completed by the author.]

### Competing interests

The author declares no competing interests, subject to final confirmation.

### Author contributions

[To be completed using CRediT roles.]

### AI-assisted writing disclosure

This draft was prepared with AI-assisted writing support under author direction. The author remains responsible for the accuracy of all claims, the verification of citations and the final manuscript content.

## References to verify

[1] Q. Fu and J. M. Easton, "Understanding data quality: Ensuring data quality by design in the rail industry," 2017 IEEE International Conference on Big Data, 2017.

[2] M. Mora-Cantallops, S. Sanchez-Alonso, E. Garcia-Barriocanal, and M.-A. Sicilia, "Traceability for Trustworthy AI: A Review of Models and Tools," Big Data and Cognitive Computing, 2021.

[3] D. Golightly and M. S. Young, "Local knowledge in rail signalling and balancing trade-offs," Applied Ergonomics, 2022.

[4] J. Tornquist Krasemann, "Computational decision-support for railway traffic management and associated configuration challenges: An experimental study," Journal of Rail Transport Planning & Management, 2015.

[5] J. Perez-Cerrolaza et al., "Artificial Intelligence for Safety-Critical Systems in Industrial and Transportation Domains: A Survey," ACM Computing Surveys, 2024.

[6] J. Wiggerthale and C. Reich, "Explainable Machine Learning in Critical Decision Systems: Ensuring Safe Application and Correctness," AI, 2024.

[7] Network Rail, "Network Rail Open Data Feeds," official web resource.

[8] Open Rail Data Wiki, "TD - Train Describer Messages."

[9] Open Rail Data Wiki, "Train Movements."

[10] Open Rail Data Wiki, "VSTP."

[11] A. Kumar, A. Zhou, G. Tucker, and S. Levine, "Conservative Q-Learning for Offline Reinforcement Learning," NeurIPS, 2020.

[12] Z. Hu et al., "Heterogeneous Graph Transformer," WWW, 2020.

## Tables

**Table 1. Dataset construction and split (Derby workstation, 14 months).**

| Item | Value |
|---|---|
| Usable decision snapshots | 1,996,572 |
| Set / wait decisions | 546,418 / 1,453,205 |
| Train / validation / test snapshots | 1,472,064 / 186,145 / 338,363 |
| Split rule | time-based, episode-local (train < 2024-02-01; val < 2024-03-01; test ≥ 2024-03-01) |
| Vocabulary (track / signal / route / train) | 268 / 123 / 278 / 2,184 |
| Delay-change coverage (post-correction) | 685,715 decisions (≈34%) |

**Table 2. Training gates (full-data Conservative Q-Learning; seed 42, with seed 43 as reproducibility check).** All §11 gates pass.

| Phase | Gate | Seed 42 | Seed 43 |
|---|---|---|---|
| A | route > 0.50 / time > 0.35 | 0.925 / 0.699 | 0.916 / 0.690 |
| B | Q-top1 > 0.55 / \|Q\| < 100 | 0.963 / 57.8 | 0.964 / 62.1 |
| C | Q-top1 > 0.65 | 0.981 | 0.983 |
| — | best validation action top-1 | 0.982 | 0.983 |

**Table 3. Stratified set-only top-1 agreement with the signaller, model vs non-learned baselines (test split, seed 42).** Set-only isolates the routing decision (the all-decisions figure is confounded by each method's wait/set propensity).

| Stratum | n (set) | CQL model | Planned-platform | First-candidate | Random |
|---|---|---|---|---|---|
| Overall | 92,280 | **0.957** | 0.531 | 0.528 | 0.322 |
| Late train | 24,964 | **0.970** | 0.616 | 0.612 | 0.353 |
| Advance | 1,760 | **0.917** | 0.709 | 0.702 | 0.320 |
| Call-on | 6,373 | **0.881** | 0.048 | 0.028 | 0.093 |
| Platform deviation | 308 | **0.903** | 0.000 | 0.000 | 0.133 |
| Priority competition | 16,066 | **0.925** | 0.511 | 0.511 | 0.314 |
| Trivial | 42,807 | **0.975** | 0.557 | 0.557 | 0.344 |

**Table 4. Safety and off-policy value of the learned policy (seed 42).** Simulator validated on a held-out month (occupancy rank-corr 0.94; throughput rank-corr 0.86). ΔV = FQE discounted-return difference (model − signaller); 95% CI episode-clustered.

| Quantity | Value |
|---|---|
| Divergence rate (set decisions) | 4.3% |
| Genuine-unsafe divergence rate | 0.0% |
| Conflict-load Δ (headway-wait events, model − signaller) | ≈ +0.07 |
| ΔV total | ≈ 0 (CI spans 0) |
| ΔV delay | +0.020 [−0.040, +0.070] |
| ΔV wait | **+0.035 [+0.027, +0.044]** |
| ΔV throughput | −0.012 [−0.018, −0.007] |
| ΔV headway | +0.005 (≈ 0) |
| Signaller IRL routing weights (set-only, l1-norm) | delay 1.45 > throughput 0.91 > wait 0.61 (headway not interpreted) |

## Figures to add before submission

Figure 1. End-to-end RailRL pipeline. Figure 2. Dynamic route-setting action formulation. Figure 3. HGT + event-transformer + per-action Q-network architecture. Figure 4. Leakage audit and data-repair summary. Figure 5. Training curves and stratified baseline comparison (Table 3). Figure 6. Explainability case study (L2 Q-gap decomposition; L5 recovered weights).

## Internal quality note

This draft follows an IMRaD-style applied AI structure. It is intentionally conservative about claims. The strongest current contribution is the traceable end-to-end system and leak-safe MDP reconstruction. The strongest future contribution depends on Stage 8 operational improvement evidence.
