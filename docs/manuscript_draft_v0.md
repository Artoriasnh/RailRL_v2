# Manuscript draft v0

Working title:

**An explainable offline reinforcement learning framework for railway route-setting decisions from live operational data**

Target journal positioning:

- Primary target: Expert Systems with Applications (ESWA), because the paper is strongest as an end-to-end intelligent decision-support system with real operational data, traceable acquisition, structured decision modelling, offline reinforcement learning, and explainability.
- Alternative target: IEEE Transactions on Intelligent Transportation Systems (T-ITS), if the final Stage 8 operational and counterfactual evaluation demonstrates clear transport-system benefit over strong baselines.

Status note:

This draft uses only evidence currently available in the RailRL v2 project. Results that are not yet complete are marked as placeholders rather than written as findings.

---

## Abstract

Railway route setting is a safety-critical traffic management task in which signallers select whether and when to set routes for trains under changing infrastructure occupation, timetable constraints and local operating practice. Although data-driven decision support has become increasingly feasible in railway operations, route-setting models are often limited by two difficulties: operational data are heterogeneous and temporally fragile, and historical signaller actions are demonstrations rather than guaranteed optimal decisions. We introduce RailRL, an end-to-end framework for reconstructing, modelling and explaining railway route-setting decisions from live operational data. The framework combines a traceable acquisition layer for Network Rail operational feeds, a leak-audited Markov decision process that converts signalling and movement records into structured decision snapshots, and a conservative offline reinforcement learning model that scores a dynamic action set consisting of waiting and feasible route-setting actions. The model uses a heterogeneous graph transformer to encode local railway infrastructure and train state, a transformer encoder for recent panel-wide event history, and per-action Q-functions trained with Conservative Q-Learning and auxiliary route and timing objectives. On the Derby workstation dataset, the pipeline produces 1,996,572 usable decision snapshots from 14 months of operational data, with time-based train, validation and test splits and systematic leakage audits. A 50k-sample sanity training run passes all predefined training gates, reaching 0.946 validation action top-1 agreement with bounded Q-values and no numerical instability. Final multi-seed training, operational baselines and counterfactual evaluation remain to be completed before making claims of improvement over human signallers. These results establish a traceable, leak-safe and interpretable foundation for offline reinforcement learning in railway signalling decision support.

Keywords:

railway traffic management; route setting; offline reinforcement learning; Conservative Q-Learning; explainable AI; heterogeneous graph transformer; operational data acquisition; signaller decision support.

---

## 1. Introduction

Railway traffic management depends on local, time-sensitive decisions made under operational constraints. In signalled railway networks, route setting determines whether a train may proceed through a physical path protected by signals, points and track circuits. A signaller does not merely choose a route in isolation. The decision involves whether to act now or wait, which train should receive scarce infrastructure first, and which feasible route should be set given timetable intent, current occupation, headway, platform use and local operating practice. These decisions affect punctuality, throughput and operational robustness, but they must also remain explainable because they occur in a safety-critical control environment.

Most algorithmic approaches to railway traffic management treat the problem as optimisation, rescheduling, simulation or prediction. These approaches provide useful abstractions, but they rarely capture the full empirical decision context observed by signallers at the level of route-setting actions. Conversely, behavioural modelling from historical records can reproduce signaller actions, but imitation alone risks treating every historical action as optimal. This is problematic in railway operations, where historical decisions reflect a mixture of routine practice, local knowledge, disruption management, information delays and occasional suboptimal choices. A useful decision-support system should therefore not only replicate historical behaviour, but also provide a principled basis for identifying when an alternative action may improve operational outcomes.

Developing such a system requires more than a neural model. First, the data must be acquired and preserved from operational feeds in a way that is traceable and usable for later reconstruction. Live railway feeds differ in structure and meaning: Train Describer messages describe signalling and infrastructure events, Train Movement records describe train-level progression and performance, timetable feeds provide planning context, and performance feeds describe service outcomes. Second, the raw data must be converted into decision points without leaking the answer or the future into the state representation. In route setting this is a major risk, because the focal signal, chosen route, realised delay and route outcome are useful for labels and rewards but must not be exposed as state features. Third, the model must operate over a dynamic action set, because each decision point has a different set of feasible routes plus the option to wait.

This paper addresses these requirements through RailRL, an end-to-end framework for explainable offline reinforcement learning in railway route setting. The framework starts from live Network Rail operational feeds and converts them into a structured research data resource through a feed-specific acquisition and storage process. It then reconstructs decision points, feasible actions, rewards and episodes for a Derby workstation route-setting task. Finally, it trains a structured-action offline reinforcement learning model that scores both wait and route-setting actions using graph, event-sequence and schedule context.

The central insight is that railway route setting can be represented as an offline reinforcement learning problem without simulating alternative next states during training. Each historical decision yields an observed transition `(s, a, r, s')`, where the state captures the local infrastructure and temporal context, the action is either wait or set a feasible route, and the reward summarises delay, route use, headway and waiting costs. Conservative Q-Learning is then used to learn a policy from these historical transitions while discouraging unsupported high values for actions outside the behaviour distribution. Counterfactual simulation is reserved for evaluation and explanation, rather than used as a training rollout model.

The paper makes four contributions.

1. It presents a traceable railway operational data acquisition framework that converts heterogeneous live Network Rail feeds into a structured empirical basis for downstream decision modelling.
2. It defines a leak-audited MDP formulation for railway route-setting decisions, including dynamic structured actions, time-local episode segmentation, reward construction and strict separation between sample metadata and model state.
3. It implements an offline reinforcement learning architecture combining a heterogeneous graph transformer, an event-sequence transformer and a per-action Q-network for `{wait} union {(train, route)}` action sets.
4. It proposes an evaluation and explainability pathway based on imitation, stratified special-case analysis, counterfactual operational comparison and multi-level explanation.

The current evidence supports the validity of the data pipeline, MDP reconstruction and training mechanics. The final claim that the learned policy improves on signaller behaviour will depend on the pending multi-seed baseline and counterfactual evaluation stages.

---

## 2. Data acquisition and operational dataset

### 2.1 Operational feeds

The empirical basis of this study is a 14-month collection of railway operational data for the Network Rail Derby workstation. The data originate from multiple live operational feed types rather than from a static benchmark. Train Describer (TD) data provide fine-grained signalling, berth and track-circuit events. Train Movement records provide train-level movement and performance information. Very Short-Term Planning (VSTP) and schedule feeds provide planning context and short-term timetable changes. RTPPM and related performance records provide service outcome context. Together, these feeds allow route-setting decisions to be studied as operational actions embedded in a dynamic railway environment.

This distinction matters for the modelling task. Route-setting decisions cannot be inferred from one feed alone. A Panel Request indicates that a route was requested, but the decision context also depends on where the train was, which track circuits were occupied, which routes were physically available, which trains were nearby, what the timetable expected, and what happened after the action. The acquisition framework therefore treats feed integration as part of the research design rather than as a preprocessing detail.

### 2.2 Acquisition framework

The acquisition framework converts live feed messages into a structured and traceable research data basis. Incoming messages are received through a broker-based subscription layer and then handled according to feed type. This feed-specific design preserves the operational meaning of each message stream: TD messages are not treated as interchangeable with Train Movement records, and timetable updates are not stored as generic event logs. For selected TD data, locally supplied signalling reference information is used to decode raw signalling-state messages into interpretable operational states before structured storage.

The collector was designed as an application-based acquisition environment rather than a single background script. It supports explicit configuration of feed credentials and database connectivity, selective subscription to feed types, live monitoring of message counts and errors, and controlled start, stop and recovery behaviour. This design was chosen because the dataset was assembled through long-running live collection rather than one-off download. The collector includes heartbeat-based disconnection detection, guarded cleanup, bounded retry, subscription recovery and runtime logging. These functions improve the traceability and inspectability of the collection process, although they do not guarantee loss-free acquisition under all external failures.

After acquisition, records are written to persistent feed-specific storage in PostgreSQL. The purpose of this storage layer is not only to save messages, but also to maintain a queryable relation between feed origin, decoded operational meaning and later analytical use. In the manuscript figures, this layer should be shown as the upstream part of the end-to-end system: live feeds -> collector -> decoded feed tables -> decision reconstruction.

### 2.3 Derby route-setting dataset

The downstream RailRL v2 pipeline constructs route-setting decision samples from the acquired operational data and static infrastructure references. The current processed dataset contains 1,999,623 generated decision points, of which 1,996,572 usable snapshots remain after excluding cases where the focal train's current track circuit could not be determined. The decision set includes 546,418 set decisions and 1,453,205 wait decisions before snapshot-level exclusions. The final snapshot table is sorted into canonical episode order and contains corrected episode labels, time splits, rewards, candidate actions and nested state representations.

The final time split is episode-based and leak-safe: training samples occur before 2024-02-01, validation samples before 2024-03-01, and test samples from 2024-03-01 onward. After correcting cross-month train identifier reuse and cutting at split boundaries, the canonical data contain 80,210 time-local episodes. The row counts are 1,472,064 training snapshots, 186,145 validation snapshots and 338,363 test snapshots. These figures should be reported together with an explicit note that `sample_id` remains the physical alignment key between state, action and reward throughout resegmentation and sorting.

---

## 3. MDP formulation

### 3.1 Decision points

A decision point is defined as a tuple `(focal_train, focal_signal, t)`. It asks whether the signaller should wait or set one of the feasible routes for the focal train from the focal signal at time `t`. Two sources produce decision points. Set samples are generated from observed Panel Request events. Wait samples are generated from approach events when a train enters the approach horizon of a signal and no corresponding route request occurs within the configured short look-ahead window.

The state uses only information available at or before the decision time. TD event history is cut strictly before the trigger time to avoid including the triggering event itself. Realised future outcomes, such as later delay change or route usage, are permitted only for reward construction and never enter the state.

### 3.2 Structured action space

The action space is dynamic:

```text
A_t = {wait} union {(focal_train, R) | R in candidates(focal_train, focal_signal, t)}.
```

In the implementation, action index 0 is always wait. Route-setting actions are indexed from 1 to `K`, where each index maps to a candidate route in `candidate_route_ids`. Candidate routes are generated from static route definitions and time-local operational context. A candidate must start from the focal signal, should be directionally consistent with recent focal-train movement when this can be inferred, and must not simply repeat a previously set route for the same train. Planned platform information is used as a soft ordering preference rather than a hard filter, because platform reassignment is itself an operationally meaningful case.

This structured action formulation avoids treating route setting as a fixed global classification problem. The model scores only the actions that are operationally meaningful at a given decision point, plus wait. It also allows the Q-function to compare waiting against immediate action within the same decision context.

### 3.3 State representation

Each state consists of four components. The first is a heterogeneous railway graph centred on the current track circuit of the focal train, not on the focal signal. This is an important leakage control: centring on the focal signal would provide information too close to the route-setting answer. The graph includes track, signal, route and train nodes, with static and dynamic edges describing infrastructure and current operational relations. Exactly one train node may be marked as focal; focal signal and focal route markers are forbidden.

The second component is a panel-wide event sequence containing the most recent TD event tokens before the decision time. The third component is a schedule outlook containing upcoming timetable context. The fourth component is a set of special-case operational flags, including late train, advance routing, call-on, platform deviation, priority competition, unusual train identifier, TRTS and freight class. These flags are used both to improve representation and to support stratified evaluation and explanation.

### 3.4 Reward

The scalar reward combines four operational components:

```text
r_total = 1.0 * r_delay
        + 0.5 * r_throughput
        + 1.0 * r_headway
        + 0.3 * r_wait.
```

`r_delay` rewards reductions in realised delay and penalises increases in delay. It uses `delay_change_seconds`, clipped to 30 minutes and scaled into minutes, with a causal gate based on approach distance. `r_throughput` rewards route requests that were actually used and penalises route requests that were cancelled or timed out without use. `r_headway` penalises route-setting decisions that lead to a measured headway below the empirical minimum threshold of 147 seconds. `r_wait` applies a fixed raw penalty to wait actions to discourage indefinite waiting while still allowing waiting to be optimal when immediate route setting would be operationally worse.

The reward is a hindsight quantity and can use future realised outcomes. However, reward components and their intermediate variables are explicitly banned from the model state. This separation is enforced through leakage audits and is central to the validity of the offline RL setting.

### 3.5 Episodes and transitions

Episodes represent time-local passages of a focal train through the Derby area. An earlier episode definition based directly on train identifiers was found to be unsafe because some TRUST identifiers are reused across months. The corrected episode construction splits by focal train, time gaps greater than two hours and train/validation/test boundaries. Transitions are then read from the canonical snapshot order:

```text
s_i     = snapshot at position i
a_i     = chosen_action_idx
r_i     = r_total
s'_i    = next snapshot in the same episode
done_i  = true if position i is the final position of the episode
```

The offline RL model therefore learns from observed historical transitions and does not generate alternative next states during training.

---

## 4. Offline reinforcement learning model

### 4.1 Encoder architecture

The model has three representation branches. A heterogeneous graph transformer encodes the local railway graph with track, signal, route and train node types. A transformer sequence encoder processes the recent event-token history. A fusion network combines graph summaries, the focal-train embedding, sequence summaries, schedule context, special-case flags and candidate-set size into a 256-dimensional state embedding.

This architecture is motivated by the structure of the task. Route-setting decisions depend simultaneously on physical infrastructure, current occupation, nearby trains and recent event history. A flat feature vector would obscure the relational structure of routes and track circuits, while a purely graph-based model would underuse short-term temporal evidence from the panel event stream.

### 4.2 Per-action Q-network

The Q-network scores the dynamic action set directly. For each route candidate, the input combines the focal-train embedding, the candidate route embedding, the fused state embedding and candidate-set size. The wait action is scored by a separate wait MLP using the focal-train embedding, the event-sequence summary, the fused state embedding and candidate-set size. Invalid padded actions are masked with a large negative sentinel, so the policy cannot select them.

The model outputs `Q(s,a)` for wait and all candidate routes. At inference time, `argmax_a Q(s,a)` returns either wait or a specific route-setting action.

### 4.3 Auxiliary heads

Two auxiliary heads support representation learning. The route head predicts the historical route among candidate routes for set decisions using a dot product between focal-train and route embeddings. The time head predicts a calibrated five-bucket lead-time label for route-setting timing. The original idea of a separate priority head is not used in the current architecture. Priority is represented through the Q-values assigned to wait and set actions across competing decision points rather than through a separate supervised label.

### 4.4 Training protocol

The main training algorithm is Conservative Q-Learning with discount factor 0.95 and conservative penalty weight 5.0. Training follows a three-stage protocol. Phase A trains the encoder and auxiliary heads to learn route and timing representations. Phase B freezes the encoder and trains the Q-function with CQL. Phase C unfreezes the model and jointly trains CQL and auxiliary losses. A target network is updated by Polyak averaging with tau 0.005. The implementation uses AdamW, warmup-to-cosine learning rate scheduling and gradient clipping.

The training loader streams transitions from the canonical snapshot file without loading the full dataset into memory. It performs block shuffling and approximate block-level stratified sampling using special-case strata. This was necessary because the dataset contains nearly two million nested snapshots and because full-table decoding was memory-inefficient.

---

## 5. Experiments and evaluation plan

### 5.1 Validation of data reconstruction

The first evaluation layer checks whether the data reconstruction is valid before model performance is interpreted. The current pipeline passes the following checks:

- 1,996,572 usable snapshots are built with zero recorded snapshot audit failures in the build summary.
- Episode resegmentation removes cross-split leakage caused by reused train identifiers.
- The final train, validation and test splits are time-based and episode-local.
- Leakage audit scripts check banned state fields, focal markers, temporal ordering and baseline shortcuts.
- A separate leakage analysis shows that high validation route accuracy is not explained by trivial candidate position baselines.

These checks should be reported as part of the main paper, not only as supplementary implementation details, because they support the validity of all subsequent learning results.

### 5.2 Sanity training result

A 50k-per-epoch sanity run with seed 42 was used to validate the training mechanics before full multi-seed training. The run passes the predefined success gates. Phase A reaches validation route accuracy 0.728 and timing accuracy 0.408. Phase B reaches validation action top-1 agreement 0.867 with bounded Q-values. Phase C reaches validation action top-1 agreement 0.946, with Q-values remaining below the predefined bound and no NaN failures. Losses decrease across the phases and auxiliary accuracies do not collapse during joint training.

This result should be framed carefully. It demonstrates that the model, data pipeline and training protocol are mechanically sound and that the behaviour policy is highly learnable. It does not yet prove that the learned policy improves railway operations relative to signallers or rule-based baselines.

### 5.3 Main comparisons

The final manuscript should compare the full CQL model against the following baselines:

- B0: random valid action.
- B0-prime: first-come-first-served or timetable/route-rule baseline.
- B1: flat behavioural cloning MLP.
- BC: structured behavioural cloning using the same candidate action set.
- CQL without special flags.
- CQL full model with graph, sequence, schedule and special flags.

Placeholder for final table:

```text
Table 1. Imitation and action-alignment metrics on validation and test sets.
Rows: B0, B0-prime, B1, BC, CQL-no-flags, CQL-full.
Columns: action top-1, route accuracy on set rows, wait/set F1, time-bucket accuracy, Q-value stability.
Report mean +/- std over seeds 42/43/44.
```

### 5.4 Stratified operational cases

The second evaluation layer should report performance by special-case stratum. The current training strata include late train, advance routing, call-on, platform deviation, priority competition, unusual train identifier and trivial cases. The final paper should report whether the model only performs well on routine cases or also handles rare and operationally important cases. This is especially important for ESWA and T-ITS reviewers, because a high aggregate imitation score can hide weak behaviour on rare safety-critical or disruption-related cases.

Placeholder for final table:

```text
Table 2. Stratified action agreement by operational case.
Columns: late_train, advance, call_on, platform_dev, priority_compete, unusual_id, trivial.
Rows: baselines and model variants.
Report test mean +/- std over three seeds.
```

### 5.5 Replicate-and-improve evaluation

The decisive operational evaluation is a Replicate-and-Improve analysis. Historical signaller actions are not treated as the gold standard. Instead, each decision is placed into one of four categories using model agreement and counterfactual operational assessment:

- aligned-justified: model matches the signaller and the action is operationally supported;
- aligned-suboptimal: model matches the signaller but a better alternative appears available;
- divergent-improving: model differs from the signaller and the counterfactual evaluation supports the model action;
- divergent-unsafe: model differs from the signaller and the signaller action is supported.

This evaluation requires the Stage 8 counterfactual or operational simulator. Until that stage is complete, claims about improvement over signallers should remain provisional.

Placeholder for final table:

```text
Table 3. Replicate-and-Improve decomposition on the test set.
Columns: count, percentage, mean counterfactual reward delta, delay component, headway component.
Rows: aligned-justified, aligned-suboptimal, divergent-improving, divergent-unsafe.
```

### 5.6 Explainability outputs

The explainability evaluation should be presented as a multi-level explanation stack:

1. model-level attention or attribution over graph nodes and event tokens;
2. decision-level Q-gap decomposition between the chosen and runner-up actions;
3. system-level counterfactual consequences over a short rollout horizon;
4. rule-level compliance against extracted operating rules;
5. reward-level interpretation of the objective trade-offs.

The paper should avoid presenting explanation as a cosmetic add-on. Each explanation layer should answer a different operational question: which assets mattered, why this action beat the alternative, what system consequence was expected, whether the action was rule-consistent, and which reward trade-off dominated.

---

## 6. Discussion

This work reframes railway route setting as an end-to-end decision-support problem that begins with live operational data acquisition and ends with explainable action scoring. The current pipeline demonstrates that heterogeneous railway feeds can be transformed into a leak-audited offline RL dataset with dynamic action sets, structured rewards and time-local episodes. The sanity training result further indicates that the route-setting behaviour observed in the Derby dataset is highly learnable using a graph-sequence representation and a conservative offline RL objective.

The main methodological contribution is the combination of traceability and decision modelling. Many learning pipelines begin from a cleaned table and treat data acquisition as external. In contrast, this framework treats acquisition, decoding, storage, state reconstruction, leakage control, reward construction and learning as one connected system. This matters in railway signalling because small data alignment errors can create large validity problems. The project encountered and corrected several such issues, including reward-to-state misalignment, train identifier reuse across months, incorrect lateness semantics and over-broad platform-deviation flags. These corrections are not incidental implementation details; they are part of what makes the resulting decision model trustworthy enough to evaluate.

The model architecture is also aligned with the operational structure of the task. The heterogeneous graph branch represents infrastructure and train relations, the event transformer represents recent temporal context, and the per-action Q-network respects the dynamic candidate set. This avoids the artificial framing of route setting as fixed-label classification and allows wait to compete directly with feasible route-setting actions.

Several limitations remain. First, final multi-seed full-data CQL results are still pending. Second, imitation agreement cannot establish operational improvement. The strongest paper claim will require the planned baseline comparisons and Replicate-and-Improve counterfactual evaluation. Third, some operational features have limited coverage. For example, platform deviation can only be assessed where route-to-platform mapping is available, and delay-change reward features are only available for a subset of decisions. Fourth, the framework is currently validated on the Derby workstation; generalisation to other signalling areas will require new SOP decoding, infrastructure mapping and acquisition checks.

If the pending evaluations show that CQL improves counterfactual operational reward while maintaining low divergent-unsafe rates, the paper can make a stronger T-ITS-style claim about railway traffic management improvement. If the operational improvement is modest or context-specific, the paper remains strong as an ESWA-style contribution: a complete, traceable and explainable expert decision-support pipeline for a safety-critical railway task.

---

## 7. Conclusion

This paper presents an end-to-end framework for explainable offline reinforcement learning in railway route-setting decision support. The framework links live operational feed acquisition, feed-specific decoding and traceable storage with leak-safe MDP reconstruction, structured reward design and Conservative Q-Learning over dynamic route-setting actions. On the Derby workstation dataset, the current pipeline constructs nearly two million usable decision snapshots and passes data leakage and sanity-training checks. These results establish a credible foundation for modelling route-setting behaviour from real railway operations. The final assessment of operational improvement will depend on forthcoming multi-seed baseline comparison and counterfactual evaluation, which are required before claiming that the learned policy improves on historical signaller decisions.

---

## Suggested manuscript structure

1. Introduction
2. Data acquisition framework and Derby operational dataset
3. Route-setting MDP formulation
4. Explainable offline RL model
5. Experiments
6. Results
7. Discussion
8. Conclusion
9. Data and code availability
10. Supplementary material

For ESWA, Sections 2-4 should be central because the system contribution is important. For T-ITS, Sections 5-6 must become stronger and should foreground operational improvements, safety constraints and transport-system metrics.

---

## Figure plan

Figure 1. End-to-end RailRL pipeline: live feeds -> acquisition framework -> structured database -> decision reconstruction -> MDP snapshots -> offline RL -> XAI/evaluation.

Figure 2. Derby route-setting decision formulation: focal train, focal signal, candidate routes and wait action.

Figure 3. State and model architecture: heterogeneous graph, event transformer, schedule/flags, fusion, Q-network and auxiliary heads.

Figure 4. Data validity and leakage audit summary: time split, episode repair, banned-field audit and baseline leakage checks.

Figure 5. Training and evaluation results: phase-wise sanity curve, final multi-seed comparison and stratified performance.

Figure 6. Explainability case study: selected decision with Q-gap, operational context, counterfactual outcome and rule-compliance interpretation.

---

## Claim-evidence map

Claim: The work is end-to-end from live operational feeds to decision support.
Evidence: Chapter 3 acquisition framework plus RailRL v2 reconstruction, MDP, training and XAI pipeline.
Status: supported as a system claim; final paper should include an integrated pipeline figure.

Claim: The dataset is leak-audited and time-local.
Evidence: corrected episode segmentation, time-based splits, leakage audit documents, final split counts.
Status: supported.

Claim: The model can learn historical route-setting behaviour.
Evidence: Stage 5 sanity run reaches 0.946 validation action top-1 agreement with bounded Q-values.
Status: supported for sanity/imitation mechanics; needs full 3-seed confirmation.

Claim: The model improves railway operations over signaller behaviour.
Evidence: Stage 8 counterfactual evaluation not yet complete.
Status: not yet supported; keep as future evaluation objective.

Claim: The framework is suitable for safety-critical decision support.
Evidence: leak audit, metadata/state separation, planned XAI layers, selective override concept.
Status: partially supported; requires L3/L4/L2 evaluation before strong deployment claims.

---

## Missing inputs before a submission-ready manuscript

1. Full Stage 6 results for seeds 42, 43 and 44.
2. Stage 7 baseline results under the same split and preprocessing.
3. Stage 8 counterfactual or operational evaluation results.
4. Final XAI case studies with at least several representative decisions.
5. Final reference list aligned with ESWA or T-ITS citation style.
6. Final decision on whether the first submission target is ESWA or T-ITS.

---

## Notes for the author

The safest current title is system-oriented rather than performance-oriented. Avoid titles claiming "improved railway traffic management" until the Replicate-and-Improve results are complete. For the current evidence state, "explainable offline reinforcement learning framework" is accurate and strong.

The strongest current paper narrative is ESWA-style: a complete intelligent decision-support system with real data acquisition, traceable processing, leak-safe MDP design, structured offline RL and explanation. The T-ITS version should be prepared only after operational improvement is quantified.
