# Medical and Regulatory Constraints

> Module 14B — Efficient AI and On-Device Inference
> Topic 09 of 11
> Estimated study time: 4 hours
> Prerequisites: 01, 05
> Version 2026.3

## 1. Why this topic exists

Compression is an engineering activity until the moment it changes a clinical output. At that moment it becomes a regulatory one. The same INT8 conversion that a machine-learning team calls a build setting is, for a regulated device, a modification that must be assessed against the approved claim, the risk file, and the change-control pathway.

Teams handle this badly in three ways.

**They treat compression as an implementation detail.** The model is quantized, the accuracy is checked, and the change is shipped without a regulatory review. The device now contains a model that differs from the one in the submission, and the difference is discoverable in an audit. The consequence ranges from a documentation finding to a recall, depending on the change and the jurisdiction.

**They treat every change as a new submission.** Every quantization tweak triggers a full 510(k), which is expensive and slow. The team concludes that compression is impractical and ships an uncompressed model that misses its latency budget. The device is compliant and unusable.

**They mistake a statistical test for a clinical argument.** A paired comparison shows no significant difference, so the team declares equivalence. But the acceptance margin was chosen after seeing the results, the subgroup analysis was omitted, and the calibration shift was never assessed. The test is not the argument; the argument is that the change is within a pre-specified, clinically justified envelope.

The regulatory framework is not a single document. It is a set of processes (software lifecycle, risk management, usability, cybersecurity), a set of pathways (510(k), De Novo, premarket approval, the European Medical Devices Regulation), and a set of expectations for how AI-enabled devices change over time (the predetermined change control plan, or PCCP).

This topic teaches the map: which requirement applies to a compression change, what evidence each requires, how to classify the change, and how to write the impact assessment. It does not provide legal advice, and the regulatory content is educational; a real submission requires professional review.

### 1.1 Three questions that decide the pathway

Every compression change reduces to three questions, and the answers determine the work.

| Question | If yes | If no |
|---|---|---|
| Does the change alter outputs beyond the approved tolerance? | It is a change, not an optimization | Documentation update |
| Does the approved plan cover this method and these bounds? | PCCP pathway | New submission or redesign |
| Is the cumulative device still within its claim? | Implement and monitor | Re-claim or revert |

The questions are deliberately ordered. A "no" to the first ends the assessment. A "yes" to the first and a "no" to the second moves the work from evidence generation to submission preparation. A "yes" to the first two and a "no" to the third means the individual change was covered but the device has drifted.

### 1.2 What this topic is not

- It is not legal advice. The classification depends on the approved plan and the jurisdiction.
- It is not a substitute for a regulatory professional. It teaches the vocabulary and the evidence structure.
- It is not a guarantee that a particular change will be cleared. It teaches how to build the argument.

The boundary matters because regulatory content is often presented as rules. Most of it is judgement, structured by process.

## 2. Learning outcomes

By the end of this topic you can:

1. **Classify** a compression change as no change, PCCP-covered, or requiring a new submission.
2. **Map** a change to the lifecycle processes in IEC 62304, ISO 14971, and IEC 62366-1.
3. **Write** an equivalence argument with a pre-specified non-inferiority margin.
4. **Design** the evidence package for a quantized model: analytical, statistical, and target.
5. **Assess** cumulative impact across multiple changes.
6. **Update** the risk file and labeling for a compressed model.
7. **Define** post-market monitoring that would detect a compression-induced drift.
8. **Document** the change so that a reviewer can reconstruct the decision.

| Outcome | Practised in | Evidenced by |
|---:|---|---|
| 1 | Exercise 1 | A classification with a rationale |
| 2 | Exercise 2 | A process map |
| 3 | Exercise 3 and Section 6 | A margin with a CI |
| 4 | Exercise 4 | An evidence table |
| 5 | Exercise 5 | A cumulative impact analysis |
| 6 | Exercise 6 | Risk-file entries |
| 7 | Exercise 7 | A monitoring plan |
| 8 | Exercise 8 | A change record |

## 3. The core mental model

A regulated device is defined by its **approved configuration**: the intended use, the model, the data, the runtime, the hardware, the labeling, and the evidence. A change is a difference between the approved configuration and the proposed one.

```text
   approved configuration
   ┌─────────────────────────────────────────────┐
   │ intended use │ model │ data │ runtime │ HW  │
   │ labeling     │ evidence │ risk file         │
   └─────────────────────────────────────────────┘
                       │
                       │ proposed change
                       ▼
   ┌─────────────────────────────────────────────┐
   │ classify: no change / PCCP / new submission │
   └─────────────────────────────────────────────┘
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
       document     protocol     submission
```

The classification depends on the **effect**, not the intent. A quantization pass that changes no output on any input is a documentation change. A quantization pass that changes the output on a borderline case is a model change, even if the average accuracy is identical.

### 3.1 The three regulatory questions

Every compression change must answer three questions.

1. **Is the change within the approved intended use?** Compression does not usually change the intended use, but it can change the population in which the device performs adequately. If the quantized model is validated only on adults and the approved use includes adolescents, the change is outside the evidence.
2. **Is the change within the approved performance envelope?** The submission states performance claims, often with confidence intervals and subgroup results. A change that moves a claim outside its interval is a change to the claim.
3. **Is the change covered by a pre-authorized pathway?** A PCCP can pre-authorize specified modifications with a protocol and an impact assessment. A change outside the plan requires a new submission or a redesign.

A "yes" to the first two and a "yes" to the third means the change can proceed under the plan. Any "no" means the change needs a different pathway or a different design.

### 3.2 Equivalence is an argument, not a test

Statistical equivalence and clinical equivalence are different claims.

- **Analytical equivalence:** the change is mathematically bounded. For example, quantization error is bounded by half a step per value, and the bound propagates through the network.
- **Statistical equivalence:** the performance difference is within a pre-specified margin, with a confidence interval that excludes the margin.
- **Clinical equivalence:** the difference is small enough that the clinical decision, the workflow, and the risk profile are unchanged.

A submission needs all three, in that order. A statistical test without an analytical bound is a measurement without a mechanism. A clinical claim without a statistical margin is an assertion.

### 3.3 The change surface

Compression changes different parts of the configuration depending on the method.

| Method | What changes | Typical classification |
|---|---|---|
| Post-training quantization | Weights, activation ranges | Model change; often PCCP-eligible |
| Quantization-aware training | Weights, training procedure | Model change; larger evidence |
| Distillation | Architecture, training procedure | New model; usually new submission |
| Structured pruning | Architecture, weights | New model; new submission |
| Unstructured pruning | Weights | Model change; PCCP-eligible if anticipated |
| Runtime upgrade | Execution, fusion, kernels | Software change; re-verification |
| Layout change | Execution only | Software change if outputs differ |

The table is a starting point, not a rule. The classification depends on the effect and the approved plan, and the plan may be narrower or broader than the typical case.

### 3.4 The evidence ladder

Evidence accumulates from the cheapest and most general to the most expensive and specific.

```text
   level 1: analytical bound
      │  "the quantization error is at most half a step"
      ▼
   level 2: numerical diff on a reference set
      │  "outputs differ by at most 0.003 in probability"
      ▼
   level 3: statistical non-inferiority
      │  "sensitivity difference is -0.4 pp, 95% CI [-1.14, +0.34]"
      ▼
   level 4: subgroup and calibration
      │  "worst subgroup -0.5 pp, CI [-1.42, +0.42]; ECE +0.008"
      ▼
   level 5: target measurement
      │  "p99 45.7 ms on the production board at 40 °C"
      ▼
   level 6: clinical review
         "the change does not alter the clinical decision or workflow"
```

A change that passes levels 1–3 but fails level 4 is not equivalent. A change that passes levels 1–5 but has no level 6 review is not clinically justified.

### 3.5 The protocol is the contract

A PCCP's modification protocol is a contract between the manufacturer and the regulator. It says: if we make this kind of change, and it meets these criteria, we may implement it without a new submission. The protocol must therefore be specific enough to be testable and general enough to cover the changes the product will need.

A protocol has five parts:

- **Data management:** what data may be used, how it is curated, and how independence is maintained.
- **Re-development:** what may be changed and what may not.
- **Performance evaluation:** the metrics, margins, subgroups, and target measurements.
- **Update procedure:** how the change is deployed, verified, and recorded.
- **Reporting:** what is communicated to the regulator and when.

A protocol without a data-management section is a protocol that cannot prevent leakage. A protocol without a target-measurement section is a protocol that cannot prevent a latency regression.

### 3.6 The difference between a plan and a promise

A plan describes what will be done and how it will be validated. A promise describes an outcome. Regulators accept plans, not promises, because a plan can be audited and a promise cannot.

For compression, the plan should say "8-bit quantization with per-channel weight scales, validated by the protocol, with a sensitivity margin of 1.5 percentage points", not "the model will remain accurate". The first is testable; the second is not.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Intended use | Regulator's description of the device's purpose and population | Confused with marketing description | Statement |
| Indication for use | Specific clinical condition and population | Confused with intended use | Statement |
| Substantial modification | Change that alters safety or effectiveness | Confused with any change | Classification |
| PCCP | Predetermined change control plan | Confused with a change log | Document |
| Modification protocol | Pre-specified method to develop, validate, and implement a change | Confused with a test plan | Document |
| Impact assessment | Analysis of benefits, risks, and cumulative effects | Confused with a risk assessment | Document |
| 510(k) | Premarket notification pathway in the United States | Confused with premarket approval | Pathway |
| De Novo | Pathway for novel low-to-moderate-risk devices | Confused with 510(k) | Pathway |
| PMA | Premarket approval for high-risk devices | Confused with 510(k) | Pathway |
| Special 510(k) | 510(k) for a device modification | Confused with a new 510(k) | Pathway |
| MDR | European Medical Devices Regulation | Confused with the AI Act | Regulation |
| AI Act | European regulation for AI systems | Confused with MDR | Regulation |
| GMLP | Good Machine Learning Practice principles | Treated as binding | Guidance |
| IEC 62304 | Medical device software lifecycle standard | Confused with IEC 60601-1 | Standard |
| IEC 60601-1 | General safety and essential performance standard | Confused with IEC 62304 | Standard |
| ISO 14971 | Risk management standard | Confused with a risk matrix | Standard |
| IEC 62366-1 | Usability engineering standard | Confused with human factors testing | Standard |
| Essential performance | Performance necessary to keep risk acceptable | Confused with accuracy | Property |
| Non-inferiority margin | Largest acceptable loss, pre-specified | Chosen after results | Proportion |
| Equivalence | Similarity within a justified margin | Confused with statistical non-significance | Claim |
| Confidence interval | Range of plausible values | Confused with a tolerance interval | Interval |
| Paired analysis | Comparison using matched inputs | Confused with unpaired comparison | Method |
| Subgroup | Population defined by a clinically relevant attribute | Treated as optional | Group |
| Calibration | Agreement between probability and frequency | Confused with quantization calibration | Metric |
| ECE | Expected calibration error | Compared across bin counts | Proportion |
| Slope | Calibration slope | Ignored after compression | Unitless |
| Decision curve | Net benefit across thresholds | Confused with ROC | Analysis |
| Risk file | Record of hazards, controls, and residual risk | Confused with a bug list | Document |
| Hazard | Potential source of harm | Confused with a failure mode | Risk term |
| Hazardous situation | Circumstance in which a hazard can cause harm | Confused with harm | Risk term |
| Residual risk | Risk remaining after controls | Assumed zero | Risk term |
| Labeling | Instructions, warnings, and claims | Confused with a sticker | Document |
| Post-market surveillance | Ongoing monitoring after launch | Confused with customer support | Process |
| Change control | Process for managing changes | Confused with version control | Process |
| Configuration item | Item under version control | Confused with a file | Quality term |
| Traceability | Ability to link requirements to evidence | Confused with logging | Property |
| Cumulative impact | Combined effect of multiple changes | Ignored in incremental updates | Analysis |
| Analytical bound | Mathematical limit on a change's effect | Confused with an empirical result | Derivation |
| Reference standard | The truth used to judge outputs | Confused with the model's label | Clinical term |

Two definitions are essential.

**Substantial modification** is not defined by the size of the code change. It is defined by the effect on safety and effectiveness. A one-line change to a threshold can be substantial; a large refactor that preserves outputs can be non-substantial. The assessment is about the effect, and the effect must be demonstrated, not asserted.

**Non-inferiority margin** must be pre-specified and clinically justified. A margin chosen after seeing the result is not a margin; it is a rationalization. The margin should reflect the smallest difference that would matter clinically, not the largest difference the data can support.

### 4.1 The evidence ladder in one table

| Level | Question | Evidence | Failure mode |
|---|---|---|---|
| 1 | What does the method change? | Analytical bound | Bound exceeds the margin |
| 2 | Do the outputs differ? | Numerical diff | Diff beyond tolerance |
| 3 | Is the difference small? | Non-inferiority test | Interval crosses the margin |
| 4 | Is it small everywhere? | Subgroup and calibration | One group fails |
| 5 | Does it hold on the device? | Target measurement | Latency or energy miss |
| 6 | Does it matter clinically? | Clinical review | Decision or workflow changes |

A package that skips a level is incomplete, and the missing level is usually the one a reviewer asks about.

## 5. The mechanics, step by step

The procedure produces a change assessment and an evidence plan. It assumes an approved device and a proposed compression change.

### Step 1. Define the approved configuration

**Input:** The submission, the design history file, and the current device.

**Output:** A configuration record with the model, data, runtime, hardware, labeling, and claims.

**Rationale:** A change is a difference from a baseline. Without a precise baseline, the difference cannot be described or assessed.

The record includes:

- Model architecture, weights hash, and training data version.
- Preprocessing, thresholds, and postprocessing rules.
- Runtime and toolchain versions.
- Target hardware and memory configuration.
- Intended use, indications, and contraindications.
- Performance claims, with confidence intervals and subgroups.
- Labeling and instructions for use.
- Risk file version.

**Branch point:** If the approved configuration cannot be reconstructed, the device is not in a state to assess a change. Reconstruct it before proceeding.

**Edge case:** A device cleared years ago may have a model that no longer matches the training pipeline. The approved configuration is the one in the submission, not the one in the repository.

### Step 2. Describe the proposed change

**Input:** The engineering change.

**Output:** A description in regulatory terms: what changes, what does not, and why.

**Rationale:** The description determines the classification. A vague description ("optimize the model") cannot be classified and will be rejected by a reviewer.

A good description names:

- The method (INT8 post-training quantization).
- The scope (all convolution and linear layers; layer normalization stays FP32).
- The parameters (per-channel symmetric weights, per-tensor asymmetric activations, 8 bits).
- The data (calibration set version and composition).
- What does not change (architecture, training data, intended use, output interface).
- The reason (latency and memory budget).

**Branch point:** If the description cannot say what does not change, the change is not understood well enough to classify.

**Edge case:** A change that bundles several modifications (quantization plus a threshold change) must describe each, because the cumulative effect may differ from the sum of the parts.

### Step 3. Classify the change

**Input:** The description and the approved PCCP, if any.

**Output:** A classification: no change, PCCP-covered, or new submission.

**Rationale:** The classification determines the pathway and the evidence. Getting it wrong in either direction is costly: an uncovered change is a compliance risk, and an over-classified change is a schedule risk.

The decision:

| Condition | Classification |
|---|---|
| Outputs identical on all inputs within tolerance, no claim change | Documentation update |
| Outputs differ within a pre-specified tolerance, plan covers the method | PCCP-covered |
| Outputs differ beyond tolerance, or plan does not cover the method | New submission or redesign |
| Intended use or population changes | New submission |
| Performance claim changes | New submission |

**Branch point:** If the plan is ambiguous about the method, treat the change as uncovered until the regulatory function confirms otherwise. Ambiguity is not authorization.

**Edge case:** A plan that covers "re-training" may not cover "quantization", even though both change weights. The wording matters.

### Step 4. Map the change to the lifecycle processes

**Input:** The classification and the device's quality system.

**Output:** A list of affected processes and records.

**Rationale:** A change is not a single document. It touches requirements, design, verification, risk, usability, cybersecurity, and labeling. A process map prevents an omission.

The map:

| Process | What the change affects | Record |
|---|---|---|
| Requirements (IEC 62304) | Performance requirements, latency, memory | Requirement update |
| Architecture and design | Runtime, precision, memory | Design description |
| Verification | Accuracy, calibration, subgroup, target | Test report |
| Risk management (ISO 14971) | New hazards, changed residual risk | Risk file update |
| Usability (IEC 62366-1) | Alert timing, confidence display | Usability assessment |
| Cybersecurity | Model integrity, update path | Security assessment |
| Labeling | Performance claims, limitations | Labeling update |
| Post-market surveillance | New monitoring signals | Monitoring plan |

**Branch point:** If a process is marked "not affected" without a rationale, the map is incomplete. Record the rationale.

**Edge case:** A change that affects only the runtime can still affect usability, because the latency changes. The map must follow the effect, not the file.

### Step 5. Build the equivalence argument

**Input:** The change, the reference data, and the clinical claim.

**Output:** An argument with an analytical bound, a statistical margin, and a clinical justification.

**Rationale:** The reviewer needs a reason to believe the change is safe, not only a measurement that it is small.

The argument has three parts.

- **Analytical:** what the method does and what bound it implies. For quantization, the per-value error is at most half a step, and the step is set by the calibrated range. The bound is a property of the method, not of the sample.
- **Statistical:** the observed difference on a held-out set, with a confidence interval and a pre-specified non-inferiority margin.
- **Clinical:** why the margin is the right one, in terms of the decision, the workflow, and the harm model.

**Branch point:** If the analytical bound exceeds the clinical margin, the change cannot be justified by statistics alone. The method or the bit width must change.

**Edge case:** A paired analysis is more powerful than an unpaired one when the same inputs are used for both models. Report the method and the correlation assumption.

### Step 6. Pre-specify the acceptance criteria

**Input:** The clinical claim and the risk analysis.

**Output:** A table of metrics, margins, and decision rules.

**Rationale:** Criteria chosen before the results are evidence; criteria chosen after are rationalization. The table is the difference.

An example:

| Metric | Margin | Decision rule |
|---|---|---|
| Sensitivity | ≥ -1.5 pp | Lower 95% CI bound above -1.5 |
| Specificity | ≥ -1.5 pp | Lower 95% CI bound above -1.5 |
| Worst subgroup sensitivity | ≥ -1.5 pp | Lower 95% CI bound above -1.5 |
| ECE | ≤ +0.015 | Point estimate below +0.015 |
| Calibration slope | ≥ -0.08 | Point estimate above -0.08 |
| p99 latency | ≤ 250 ms | Measured on target at 40 °C |

**Branch point:** If a metric has no margin, it cannot be assessed. Either define one or explain why it is not relevant.

**Edge case:** A margin that is larger than the clinical effect is not a margin; it is a permission to degrade. Justify the number clinically.

### Step 7. Assemble the evidence

**Input:** The criteria and the available data.

**Output:** An evidence package with analytical, statistical, and target results.

**Rationale:** The reviewer must be able to reproduce the conclusion. The package is the reproduction.

The package includes:

- The reference model and the compressed model, with hashes.
- The calibration data description and the range method.
- The held-out data description, including subgroups.
- The statistical analysis plan and the results.
- The target measurements: latency, energy, memory, thermal.
- The numerical diff on a reference set.
- The risk-file changes.
- The labeling changes.

**Branch point:** If the evidence uses a dataset that was also used to select the model or the threshold, the result is optimistic. Use a disjoint set.

**Edge case:** A retrospective dataset may not represent the deployment population. The evidence should state the limitation and, where possible, include a prospective or external set.

### Step 8. Assess cumulative impact

**Input:** The current change and the changes since the last approval.

**Output:** A cumulative impact statement.

**Rationale:** A sequence of small changes can produce a large cumulative effect. Each individual change may be within its margin while the total is not.

The assessment:

- List the changes since approval.
- Sum their effects on each metric, or measure the combined model.
- Compare with the approved claim.
- Identify any metric that has drifted beyond its original interval.

**Branch point:** If the cumulative effect exceeds the claim, the device is no longer the approved device, even if each change was individually covered. A new submission or a claim update is required.

**Edge case:** A change that was reverted still belongs in the history, because the evidence package may have been built on it.

### Step 9. Update the risk file and labeling

**Input:** The evidence and the risk analysis.

**Output:** Updated risk-file entries and labeling.

**Rationale:** A change that alters performance alters risk. The risk file is where that alteration is recorded and controlled.

The risk file entries:

- The hazard (subgroup degradation, calibration shift).
- The hazardous situation (a missed alert in a subgroup).
- The control (stratified calibration, monitoring, fallback).
- The residual risk and its acceptability.
- The link to the evidence.

The labeling changes:

- The performance claims, if they changed.
- The limitations, if new ones were found.
- The instructions, if the workflow changed.

**Branch point:** If the residual risk is not acceptable, the change is rejected, regardless of its engineering benefit.

**Edge case:** A labeling change can be a larger regulatory event than the model change. Plan both.

### Step 10. Implement, monitor, and record

**Input:** The approved change.

**Output:** A deployed change with a monitoring plan and a record.

**Rationale:** The change is not finished at deployment. The monitoring plan is how the device's continued safety is demonstrated.

The monitoring plan includes:

- The metrics to track: sensitivity proxies, alert rate, fallback rate, saturation rate.
- The subgroups to stratify.
- The thresholds for investigation.
- The review cadence.
- The escalation path.

The record includes:

- The change description and classification.
- The evidence package.
- The approval.
- The deployment date and version.
- The monitoring plan.
- The cumulative impact statement.

**Branch point:** If the monitoring cannot detect the change's effect, the plan is not adequate. Add a signal or increase the review frequency.

**Edge case:** A device without connectivity cannot report telemetry. The monitoring plan must use the channels that exist, such as service visits or returned devices.

### 5.1 Decision table for change classification

| Evidence | Plan coverage | Classification | Action |
|---|---|---|---|
| Outputs identical within tolerance | Any | Documentation | Update the record |
| Outputs differ, within margins | Method explicitly covered | PCCP-covered | Implement per protocol |
| Outputs differ, within margins | Method not covered | New submission | Prepare a submission |
| Outputs differ, beyond margins | Any | New submission or redesign | Redesign or re-claim |
| Subgroup fails, overall passes | Any | Not equivalent | Redesign or restrict the claim |
| Calibration fails at the threshold | Any | Not equivalent | Recalibrate or redesign |
| Cumulative drift beyond claim | Any | New submission | Re-claim or revert |

### 5.2 Edge cases in regulatory assessment

**Outlier channels.** An outlier-driven scale collapse is an analytical failure that can be bounded: the step is known, and the affected weights can be identified. The bound belongs in the analytical section, not only in the statistical result.

**Per-tensor versus per-channel scales.** Per-channel quantization changes the metadata and the arithmetic. If the plan anticipated per-tensor, per-channel may be a different modification. Check the wording.

**Activation clipping.** Clipping changes the tail behavior. A statistical result on average may hide a tail effect that matters clinically. Evaluate the tail explicitly.

**Operator fallback.** A fallback changes the execution path and may change outputs. If the runtime version changes the fallback behavior, the change is not covered by a model-only plan.

**Unsupported operators.** A model that runs partly in float is a hybrid. The plan must describe the hybrid, not only the quantized part.

**Zero-point drift.** A drift after deployment is a post-market signal, not a pre-market one. The monitoring plan must include saturation or a proxy.

**Calibration-set bias.** A calibration set that under-represents a subgroup is a design defect, not only a statistical limitation. The risk file should record it.

**Thermal throttling.** A latency claim measured at 22 °C may not hold at 40 °C. The target evidence must state the condition, and the claim must match it.

### 5.3 What a PCCP does and does not cover

A PCCP authorizes a **class of modifications**, not a model. It does not authorize:

- A change outside the described method, even if the effect is smaller.
- A change outside the stated bounds, such as a bit width below the described minimum.
- A change to the intended use, the population, or the output interface.
- A change that the impact assessment did not consider.
- A cumulative drift beyond the approved claim, even if each change was covered.

It does authorize:

- The described method within the stated bounds.
- The data and validation procedure in the protocol.
- The update procedure in the protocol.
- The labeling changes described in the plan.

The distinction matters because teams often read a PCCP as a general permission to improve the model. It is a permission to make a specific kind of change under specific conditions. A change that is better than the plan anticipated is still outside the plan.

### 5.4 Does quantization require a new submission?

The question has no universal answer, and the honest response is a decision procedure.

1. **Is the model part of the approved device?** If the model is a software item that produces clinical output, yes.
2. **Does the change alter outputs beyond the approved tolerance?** If no, it is a documentation update. If yes, continue.
3. **Does the approved plan describe this method and these bounds?** If yes, the change may be PCCP-covered. If no, continue.
4. **Is the change within the approved performance claim and its confidence intervals?** If yes and the plan covers it, implement under the plan. If no, a new submission or a redesign is required.

The common error is to skip question 2 by asserting that INT8 is "numerically equivalent". It is not; it is bounded, and the bound must be compared with the tolerance. The second common error is to skip question 3 by assuming that a plan about "model updates" covers quantization. The plan's wording is the authority.

A useful test: if the change required a new calibration dataset and a new numerical comparison, it is a model change, not a build change. Whether it needs a submission depends on the plan and the claim.

### 5.5 The evidence package for a numerically different model

A model that differs numerically from the cleared one needs an evidence package that lets a reviewer conclude that the difference is acceptable. The package has six parts.

- **Configuration:** the cleared model and the proposed model, with hashes, runtime versions, and calibration data versions.
- **Analytical bound:** what the method changes and the maximum per-value effect.
- **Numerical diff:** the output difference on a frozen reference set, including tails and borderline cases.
- **Statistical comparison:** pre-specified metrics, margins, confidence intervals, and subgroups.
- **Target evidence:** latency, energy, memory, and thermal behaviour on the production configuration.
- **Risk and labeling:** the updated hazards, controls, residual risks, and claims.

The package is not a single study. It is a chain of evidence, and the reviewer can reject it at any link. A missing analytical bound makes the numerical diff anecdotal; a missing subgroup makes the statistical comparison incomplete; a missing target measurement makes the latency claim unsupported.

### 5.6 Change-control documentation

The change record is the artifact that survives the project. It should contain:

| Section | Content |
|---|---|
| Identification | Change ID, date, owner, device versions |
| Description | Method, scope, parameters, data, what does not change |
| Rationale | Why the change is needed |
| Classification | Covered, not covered, or documentation only, with the plan clause |
| Evidence | Analytical, numerical, statistical, target |
| Risk impact | New hazards, changed residual risks |
| Labeling impact | Claims, limitations, instructions |
| Cumulative impact | Changes since approval and their combined effect |
| Approval | Who approved, on what basis |
| Deployment | Version, date, rollback plan |
| Monitoring | Metrics, thresholds, cadence |

A record that omits the classification or the cumulative impact is incomplete, even if the evidence is strong. The reviewer needs to know not only what was measured but why the measurement is sufficient.

### 5.7 Performance-neutral versus behaviour-shifting compression

Two compression changes can have the same average accuracy and different regulatory meanings.

**Performance-neutral compression** changes the representation without changing the decisions. The outputs are identical within a tight tolerance on the reference set, the calibration is unchanged, and the subgroup results are unchanged. The change is a candidate for a documentation update or a narrow PCCP.

**Behaviour-shifting compression** preserves the average metric but changes where the errors fall. Calibration may shift, a subgroup may degrade, or the tail may change. The average is unchanged; the device is not.

The distinction is not about the method. The same INT8 conversion can be performance-neutral on one model and behaviour-shifting on another, depending on the weight distribution, the calibration data, and the clinical threshold.

The test is a table, not a metric:

| Dimension | Neutral | Shifting |
|---|---|---|
| Overall sensitivity | Within 0.2 pp | Within 0.2 pp |
| Worst subgroup | Within 0.2 pp | Falls 1.0 pp or more |
| ECE | Within 0.003 | Rises 0.010 or more |
| Slope | Within 0.02 | Falls 0.05 or more |
| Decision agreement | 99.9% | 98–99% with clustering in one class |

A change that is neutral on all five is a strong candidate for a light pathway. A change that is shifting on any one needs the full evidence package, regardless of its average.


## 6. Worked example

This is a constructed case. It is not legal advice, and the classification depends on the actual approved plan, which is hypothetical here.

### 6.1 The approved device and the proposed change

The approved device is the ECG patch from earlier topics: a 41,216-parameter FP32 model, 164,864 bytes of weights, 138 ms p99, 3.395 mJ per inference, sensitivity 0.942, ECE 0.021, worst subgroup 0.931. The approved claim is sensitivity at least 0.925 overall, with a subgroup floor of 0.915 and an ECE ceiling of 0.040.

The proposed change is INT8 post-training quantization: per-channel symmetric weights, per-tensor asymmetric activations, 8 bits, calibration on 512 windows from three sites. The architecture, training data, intended use, and output interface are unchanged.

The PCCP describes "precision reduction of the approved architecture to 8-bit integer arithmetic, with per-channel weight scales, validated by the protocol in Appendix B." The proposed change matches that description.

### 6.2 Scale and zero point by hand

The first convolution has 40 weights with a minimum of -0.061 and a maximum of 0.412. The affine parameters:

```text
S = 0.473 / 255 = 0.00185490
Z = round(0.061 / 0.00185490) = 33
q(0.412) = 222 + 33 = 255
q(-0.061) = -33 + 33 = 0
```

The symmetric parameters:

```text
S = 0.412 / 127 = 0.00324409
Z = 0
```

The affine step is 1.75 times finer, and the SQNR is 42.04 dB against 37.54 dB. For the plan, the relevant fact is that the per-value error is bounded by half a step, or 0.000927 in affine and 0.001622 in symmetric. That bound is the analytical basis of the equivalence argument.

### 6.3 Memory and latency before and after

| Quantity | FP32 approved | INT8 proposed | Change |
|---|---:|---:|---:|
| Weight bytes | 164,864 | 41,816 | -74.6% |
| KiB | 161.0 | 40.8 | -120.2 KiB |
| p99 latency (ms) | 138.0 | 45.7 | -66.9% |
| Energy (mJ) | 3.395 | 1.124 | -66.9% |

The memory and latency changes are large and favorable. They are also irrelevant to the equivalence argument: a faster model is not automatically an equivalent model, and the regulatory question is about outputs, not speed.

### 6.4 Accuracy and calibration

| Metric | FP32 approved | INT8 proposed | Difference | Margin | Result |
|---|---:|---:|---:|---:|---|
| Sensitivity | 0.942 | 0.938 | -0.4 pp | -1.5 pp | Pass |
| Specificity | 0.961 | 0.958 | -0.3 pp | -1.5 pp | Pass |
| Worst subgroup | 0.931 | 0.926 | -0.5 pp | -1.5 pp | Pass |
| ECE | 0.021 | 0.029 | +0.008 | +0.015 | Pass |
| Slope | 0.97 | 0.93 | -0.04 | -0.08 | Pass |
| p99 latency | 138 ms | 45.7 ms | -92.3 ms | ≤ 250 ms | Pass |

The point estimates all pass. The confidence intervals decide whether the evidence is sufficient.

### 6.5 Confidence intervals

With 8,000 positive cases for sensitivity and 6,000 for the subgroup, and an unpaired normal approximation for illustration:

```text
Sensitivity:
  difference = -0.0040
  SE = sqrt(0.942×0.058/8000 + 0.938×0.062/8000) = 0.00375
  95% CI = -0.0040 ± 1.96×0.00375 = [-0.0114, +0.0034]

Worst subgroup:
  difference = -0.0050
  SE = sqrt(0.931×0.069/6000 + 0.926×0.074/6000) = 0.00470
  95% CI = -0.0050 ± 1.96×0.00470 = [-0.0142, +0.0042]
```

Both lower bounds are above the -1.5 pp margin (-1.14 and -1.42). The sensitivity result passes comfortably; the subgroup result passes with 0.08 pp of margin, which is thin.

A paired analysis would usually produce a narrower interval, because the same inputs are used for both models. The case uses the unpaired approximation to be conservative and to keep the arithmetic simple; the analysis plan should state which method is used.

### 6.6 Diagnostic one: outlier-driven scale collapse

The fully connected layer's 32,768 weights have a median absolute value of 0.0610 and a maximum of 1.2912. The symmetric per-tensor scale is 0.01016693, or 16.7% of the median.

```text
step / median = 0.1667
weights rounded to zero = 1,411 (4.3%)
distinct levels used = 102 of 255
SQNR = 30.00 dB
```

For the regulatory argument, the diagnostic matters because it identifies which weights are affected. The 1,411 zeroed weights are the smallest 4.3%, and their contribution to the output is bounded by the step times the number of weights per output. That bound belongs in the analytical section.

The remedy, per-channel quantization, raises SQNR to 44.67 dB and adds 2,560 bytes of metadata. The plan anticipated per-channel scales, so the remedy is within the approved modification.

### 6.7 Diagnostic two: per-channel skew

The 512 channels have ranges from 0.2756 to 2.3202, a ratio of 8.4. Under a per-tensor scale, a typical channel gets about 28 levels; per-channel gives every channel 255 levels.

The diagnostic supports the choice of per-channel quantization, which is what the plan describes. If the plan had specified per-tensor, the skew would be evidence that the proposed change is different from the approved one, and a plan amendment or a new submission would be required.

### 6.8 Varying one assumption

**Calibration set: 512 to 64 windows.** The activation ranges shrink by about 6%, clipping increases, and sensitivity falls to 0.934. The difference is -0.8 pp, and the 95% CI is [-1.55, -0.05], which crosses the -1.5 pp margin. The change fails the pre-specified criterion. The lesson is that calibration set size is a regulatory variable, not only an engineering one.

**Bit width: 8 to 4.** Sensitivity falls to 0.907, a difference of -3.5 pp, with a CI of [-4.32, -2.68]. The change fails by a wide margin. INT4 is outside the approved modification, and the plan's description of 8-bit arithmetic does not cover it.

**Layer excluded: the first convolution kept in FP32.** The model grows by 120 bytes (0.29%), and sensitivity rises to 0.939, a difference of -0.3 pp. The change passes with more margin. Excluding a layer is a different modification from the plan's "all convolution and linear layers", so the plan must describe the exclusion explicitly.

**Cumulative change: quantization plus a threshold adjustment.** If a threshold change of -0.2 pp is added, the combined sensitivity effect is -0.6 pp, still within the margin, but the calibration slope may shift by -0.06, close to the -0.08 limit. The cumulative assessment is required because the individual changes pass.

### 6.9 The impact assessment

The impact assessment has three parts.

**Benefits:** The model fits the device's flash and latency budgets, which enables the on-device alert. The reduced energy extends battery life. These are benefits to the patient and to the product.

**Risks:** The subgroup margin is thin (0.08 pp above the criterion). The calibration slope falls from 0.97 to 0.93, which may change the alarm rate at the threshold. The rare-class sensitivity falls by 0.6 pp. Each risk has a control: stratified calibration, threshold re-selection on validation data, and monitoring of the subgroup alert rate.

**Cumulative:** The change is the first since approval. The cumulative impact equals the individual impact. The assessment records that future changes must be evaluated against the post-change baseline, not the original.

### 6.10 The classification decision

The change matches the plan's description of 8-bit precision reduction with per-channel weight scales. The evidence meets the protocol's acceptance criteria. The cumulative impact is within the approved claim.

The classification is therefore PCCP-covered, with two conditions: the subgroup margin must be monitored, and any future change to the calibration set size or the bit width is a new modification that requires a plan amendment.

If the plan had not included the method, the classification would be a new submission, likely a Special 510(k) for a device modification, with the same evidence package plus the submission format.

### 6.11 What the case does not show

- **The actual approved plan.** The case assumes a plan that covers the method. A real plan may be narrower.
- **The clinical review.** The statistical result is not a clinical judgement; a clinician must confirm that the change does not alter the decision or the workflow.
- **The labeling change.** The performance claims may need to be updated to reflect the new confidence intervals.
- **The post-market data.** The monitoring plan is described but not executed.
- **The jurisdiction-specific requirements.** The case is generic; the United States and the European Union have different pathways and different evidence expectations.

### 6.12 A case that passes on average and fails a subgroup

The per-tensor INT8 model has an overall sensitivity of 0.934 against the approved 0.942, a difference of -0.8 pp. The approved claim floor is 0.925, so the model passes the claim with 0.9 pp of margin. A team that reports only the overall metric would classify the change as acceptable.

The subgroup table tells a different story.

| Group | FP32 approved | INT8 per-tensor | Difference | 95% CI | Margin | Result |
|---|---:|---:|---:|---|---:|---|
| Overall | 0.942 | 0.934 | -0.8 pp | [-1.6, 0.0] | -1.5 pp | Pass, narrowly |
| Group A | 0.945 | 0.940 | -0.5 pp | [-1.3, 0.3] | -1.5 pp | Pass |
| Group B | 0.931 | 0.918 | -1.3 pp | [-2.2, -0.4] | -1.0 pp | Fail |
| Group C | 0.938 | 0.933 | -0.5 pp | [-1.4, 0.4] | -1.5 pp | Pass |

Group B fails its pre-specified margin. The overall result passes because Group B is a minority of the sample, and its loss is diluted by the larger groups. This is the exact pattern that aggregate reporting hides.

The regulatory consequence is not "the model is slightly worse". It is:

- The change is **not equivalent** for the intended population, because the intended population includes Group B.
- The PCCP's performance-evaluation criterion is not met, so the change cannot be implemented under the plan.
- The options are: improve the method (per-channel quantization raises Group B to 0.926), restrict the intended use to exclude Group B (a new submission and a labeling change), or abandon the change.
- If the team implements the change anyway and reports only the overall metric, the submission contains a misleading analysis, which is a compliance problem independent of the model's performance.

Per-channel quantization recovers Group B to 0.926, a difference of -0.5 pp with a CI of [-1.4, 0.4], which passes the -1.0 pp margin. The remedy is within the plan's description if the plan says "per-channel weight scales". If the plan says "per-tensor", the remedy is a different modification and requires an amendment.

### 6.13 What the subgroup failure teaches

- The overall metric is not the claim. The claim is about the population, and the population includes the subgroup.
- A margin must be pre-specified per subgroup, not only overall. A single overall margin lets a small group fail silently.
- The remedy must be within the approved modification. A better method that is outside the plan is still outside the plan.
- The regulatory consequence of a hidden subgroup failure is larger than the performance loss. The misleading analysis is the bigger problem.


## 7. Code, query, or calculation

The script compares an approved model's metrics with a proposed model's metrics, computes confidence intervals for the differences, applies pre-specified margins, and classifies each metric. It also implements the quantization arithmetic.

```python
import math


def difference_ci(p_approved, p_proposed, n_approved, n_proposed, z=1.96):
    """Unpaired normal approximation for a difference in proportions."""
    diff = p_proposed - p_approved
    se = math.sqrt(p_approved * (1 - p_approved) / n_approved
                   + p_proposed * (1 - p_proposed) / n_proposed)
    return {"diff": diff, "se": se,
            "low": diff - z * se, "high": diff + z * se}


def classify(metric, result, margin, lower_is_worse=True):
    """A metric passes if the relevant bound is inside the margin."""
    bound = result["low"] if lower_is_worse else result["high"]
    passed = bound > margin if lower_is_worse else bound < margin
    return {"metric": metric, "diff": round(result["diff"], 4),
            "low": round(result["low"], 4), "high": round(result["high"], 4),
            "margin": margin, "pass": passed}
```

Evaluate the proposed change:

```python
checks = []
checks.append(classify("sensitivity",
                       difference_ci(0.942, 0.938, 8000, 8000), -0.015))
checks.append(classify("specificity",
                       difference_ci(0.961, 0.958, 8000, 8000), -0.015))
checks.append(classify("worst subgroup",
                       difference_ci(0.931, 0.926, 6000, 6000), -0.015))
checks.append(classify("calibration slope",
                       difference_ci(0.97, 0.93, 8000, 8000), -0.08))

for c in checks:
    print("%-18s diff %+.4f CI [%+.4f, %+.4f] margin %+.3f pass %s"
          % (c["metric"], c["diff"], c["low"], c["high"],
             c["margin"], c["pass"]))
```

Expected output:

```text
sensitivity        diff -0.0040 CI [-0.0114, +0.0034] margin -0.015 pass True
specificity        diff -0.0030 CI [-0.0103, +0.0043] margin -0.015 pass True
worst subgroup     diff -0.0050 CI [-0.0142, +0.0042] margin -0.015 pass True
calibration slope  diff -0.0400 CI [-0.0437, -0.0363] margin -0.080 pass True
```

The calibration slope interval is narrow because the code applies the proportion formula to a continuous quantity, which is not appropriate. For a continuous metric, use a paired bootstrap or a normal interval with the metric's own standard error. The example is included to show the structure, not to endorse the formula for every metric.

Variations:

```python
print("calibration 64:",
      classify("sensitivity",
               difference_ci(0.942, 0.934, 8000, 8000), -0.015))
print("int4:",
      classify("sensitivity",
               difference_ci(0.942, 0.907, 8000, 8000), -0.015))
print("layer excluded:",
      classify("sensitivity",
               difference_ci(0.942, 0.939, 8000, 8000), -0.015))
```

Expected output:

```text
calibration 64: {'metric': 'sensitivity', 'diff': -0.008, 'low': -0.0155, 'high': -0.0005, 'margin': -0.015, 'pass': False}
int4: {'metric': 'sensitivity', 'diff': -0.035, 'low': -0.0432, 'high': -0.0268, 'margin': -0.015, 'pass': False}
layer excluded: {'metric': 'sensitivity', 'diff': -0.003, 'low': -0.0103, 'high': 0.0043, 'margin': -0.015, 'pass': True}
```

Quantization arithmetic:

```python
def round_half_away(value):
    if value >= 0:
        return int(math.floor(value + 0.5))
    return int(math.ceil(value - 0.5))


def affine_params(values, bits=8):
    lo, hi = min(values), max(values)
    scale = (hi - lo) / (2 ** bits - 1)
    zero_point = min(2 ** bits - 1, max(0, round_half_away(-lo / scale)))
    return scale, zero_point


def symmetric_scale(values, bits=8):
    return max(abs(v) for v in values) / (2 ** (bits - 1) - 1)


def quantize(values, scale, zero_point=0, bits=8):
    qmax = 2 ** bits - 1
    return [min(qmax, max(0, round_half_away(v / scale) + zero_point))
            for v in values]


def dequantize(codes, scale, zero_point=0):
    return [(q - zero_point) * scale for q in codes]


weights = [-0.061, -0.034, 0.012, 0.021, 0.045, 0.089, 0.412]
s, z = affine_params(weights, 8)
print("affine S=%.8f Z=%d" % (s, z))
print("quantized", quantize(weights, s, z, 8))
print("reconstructed", [round(v, 6) for v in dequantize(quantize(weights, s, z, 8), s, z)])
print("symmetric S=%.8f" % symmetric_scale(weights, 8))
```

Expected output:

```text
affine S=0.00185490 Z=33
quantized [0, 15, 39, 44, 57, 81, 255]
reconstructed [-0.06121, -0.03339, 0.01113, 0.0204, 0.04452, 0.08903, 0.41195]
symmetric S=0.00324409
```

Assumptions: the confidence intervals use an unpaired normal approximation; the margins are pre-specified in the protocol; the metrics are proportions except for the slope; and the quantization uses round-half-away with per-tensor scales.

## 8. Data, standards, and regulatory context

This section is the reference map. It states what each instrument requires, what it does not, and how it applies to compression.

### 8.1 IEC 62304

IEC 62304, "Medical device software — Software life cycle processes", Edition 1.1 (2006 plus Amendment 1:2015), is the software lifecycle standard recognized by the United States Food and Drug Administration (FDA) under recognition number 13-79. It requires software requirements, architecture, detailed design, unit verification, integration testing, and configuration management, with rigour scaled by safety class A, B, or C.

For a compression change:

- The model, runtime, and toolchain are software items.
- The compression procedure is a design activity with an input, output, and rationale.
- Verification must demonstrate the requirements, including accuracy, calibration, latency, and memory.
- The change must be under configuration management, with the old and new configurations identified.

IEC 62304 does not define how much evidence is enough for a compression change. It requires that the requirements be verified; the clinical and statistical judgement is supplied by the other processes.

### 8.2 ISO 14971

ISO 14971, "Medical devices — Application of risk management to medical devices", requires hazards to be identified, risks estimated and evaluated, controls implemented, and residual risks assessed.

Compression-specific hazards:

- A subgroup's sensitivity falls below the clinical boundary.
- Calibration shifts at the operating threshold, changing the alarm rate.
- A rare class loses the feature that distinguished it.
- A quantized model saturates on out-of-distribution inputs.
- A fallback path produces a false negative.

Each hazard needs a hazardous situation, a control, and a residual-risk judgement. The risk file is where the compression evidence meets the clinical harm model.

### 8.3 IEC 60601-1 and essential performance

IEC 60601-1, the general standard for basic safety and essential performance of medical electrical equipment, defines essential performance as the performance necessary to keep risk acceptable. If the device's essential performance depends on the model's output within a deadline, the compressed model's latency and accuracy are part of essential performance, and the verification must cover the environmental conditions the standard requires.

### 8.4 IEC 62366-1 and usability

IEC 62366-1, "Application of usability engineering to medical devices", requires the device to be designed for safe and effective use. Compression can affect usability in two ways: a change in alert timing, and a change in the confidence information presented to the user. A model that is 93 ms faster does not change the workflow; a model that is 93 ms slower may.

### 8.5 FDA Good Machine Learning Practice

The FDA, Health Canada, and the United Kingdom Medicines and Healthcare products Regulatory Agency published ten guiding principles for Good Machine Learning Practice (GMLP) in October 2021. The principles are non-binding, but they describe the expectations a reviewer brings.

- Principle 3: clinical study participants and data sets are representative of the intended patient population.
- Principle 4: training data sets are independent of test sets.
- Principle 6: model design is tailored to the available data and reflects the intended use.
- Principle 8: testing demonstrates device performance during clinically relevant conditions.
- Principle 10: deployed models are monitored for performance and re-training risks are managed.

A quantization calibration set that misses a subgroup violates principle 3. A calibration set drawn from the test set violates principle 4. A target measurement taken only at room temperature may not satisfy principle 8. A model without monitoring violates principle 10.

### 8.6 FDA PCCP

The FDA's final guidance, "Marketing Submission Recommendations for a Predetermined Change Control Plan for Artificial Intelligence-Enabled Device Software Functions", was issued on 3 December 2024, with a Federal Register notice on 4 December 2024, under section 515C of the Food, Drug, and Cosmetic Act as added by the Food and Drug Omnibus Reform Act of 2022.

The guidance describes a PCCP with three components:

- **Description of Modifications:** what may change, within what bounds.
- **Modification Protocol:** how the change is developed, validated, and implemented, including the data management, re-training, performance evaluation, and update procedures.
- **Impact Assessment:** the benefits and risks of the modifications, including bias and cumulative impact.

The guidance is non-binding, but the statutory pathway it implements is binding. A PCCP is reviewed as part of a 510(k), De Novo, or PMA submission, and modifications within the plan can be implemented without a new submission, subject to the plan's protocol and any reporting requirements.

For compression, the description should name the method, the precision, the granularity, the calibration data, and the acceptance criteria. A description that says "model optimization" is too vague to authorize a specific change.

### 8.7 EU Medical Devices Regulation and AI Act

Regulation (EU) 2017/745, the Medical Devices Regulation, governs the device: classification, conformity assessment, clinical evaluation, post-market surveillance, and vigilance.

Regulation (EU) 2024/1689, the Artificial Intelligence Act, adds obligations for high-risk AI systems. A medical device that is a safety component, or is itself a device requiring third-party conformity assessment, is high-risk under Article 6(1).

The Annex I applicability date was originally 2 August 2027. The Commission's AI Omnibus, adopted on 19 November 2025 with political agreement reported on 7 May 2026, moves the Annex I date to 2 August 2028. Verify the current date before relying on it; the Medical Device Coordination Group guidance MDCG 2025-6 (June 2025) remains the reference for the interaction between the two regulations.

For compression, the relevant obligations are transparency about the model and its limitations, human oversight, and post-market monitoring. A compressed model whose subgroup performance differs is a transparency issue, not only an engineering one.

### 8.8 Cybersecurity

The FDA's cybersecurity guidance and the IEC 62443 series establish expectations for secure boot, signed updates, vulnerability monitoring, and an SBOM. A compressed model is part of the attack surface: a modified model file can change outputs without changing the application. The integrity of the model file must be protected by the same mechanisms as the firmware.

### 8.9 What is binding and what is convention

| Instrument | Status | Compression relevance |
|---|---|---|
| IEC 62304 | Binding where required | Model and toolchain are software items |
| ISO 14971 | Binding where required | Compression hazards enter the risk file |
| IEC 60601-1 | Binding where required | Timing may be essential performance |
| IEC 62366-1 | Binding where required | Alert timing and confidence display |
| FDA GMLP | Non-binding guidance | Representativeness and independence |
| FDA PCCP guidance | Non-binding; statute binding | Change must be pre-authorized |
| EU MDR | Binding regulation | Clinical evaluation and surveillance |
| EU AI Act | Binding regulation, dates moving | Transparency and oversight |
| IEC 62443 | Binding where applicable | Secure update and monitoring |
| Statistical margins | Convention | Must be pre-specified and justified |

### 8.10 The documents a reviewer expects

| Document | Content |
|---|---|
| Change description | Method, scope, parameters, data, what does not change |
| Classification | Covered, not covered, or documentation only |
| Equivalence argument | Analytical bound, statistical margin, clinical justification |
| Analysis plan | Metrics, margins, subgroups, methods |
| Results | Point estimates, intervals, target measurements |
| Risk update | Hazards, controls, residual risk |
| Labeling update | Claims, limitations, instructions |
| Monitoring plan | Metrics, subgroups, thresholds, cadence |

### 8.11 What the guidance does not settle

- Whether a specific quantization change is substantial. The guidance gives a process, not a list.
- How large a subgroup margin should be. That is a clinical judgement.
- Whether a performance-neutral change needs any submission at all. It depends on the plan and the jurisdiction.
- How cumulative impact should be measured when changes interact. The guidance requires the assessment, not the method.
- Whether a runtime upgrade is a device change when the model file is unchanged. It depends on the effect.

The gaps are not defects in the guidance; they are the space in which professional judgement operates. A team that expects a lookup table will be disappointed, and a team that documents its judgement will be prepared.

## 9. Industry practice and case evidence

### 9.1 What the guidance actually says

The FDA PCCP guidance is the most direct public statement about how AI-enabled devices are expected to change. Its three-part structure, with a description of modifications, a modification protocol, and an impact assessment, is a template for the compression dossier.

The guidance was finalized after a draft in April 2023 and a period of public comment. The final version clarifies scope, aligns definitions with the executive order in force at the time, adds recommendations on labeling and public decision summaries, and discusses the Special 510(k) pathway for PCCP modifications. These are documented changes, not interpretations.

The GMLP principles, published in October 2021, are the other public statement. They are non-binding, and they describe the expectations that reviewers and auditors bring to a model's development and monitoring.

### 9.2 What is not public

Most compression changes in cleared devices are not individually public. The FDA's AI-enabled device list identifies devices with AI/ML functions, but it does not describe the quantization of each model. The absence of public case data means the evidence for "what works" comes from the guidance and from general regulatory practice, not from a catalogue of cleared quantization changes.

This is an important limit. A team cannot point to a precedent and say "this quantization was cleared". It must build its own argument, and the argument's quality is what a reviewer assesses.

### 9.3 What the evidence supports

| Claim | Status |
|---|---|
| A PCCP can cover planned model changes | FDA guidance, December 2024 |
| The plan needs a description, protocol, and impact assessment | FDA guidance |
| GMLP expects representative data and independent test sets | Joint guidance, October 2021 |
| IEC 62304 requires verification of software requirements | Standard |
| ISO 14971 requires residual-risk assessment | Standard |
| Compression changes should be classified by effect | Inference from the guidance, not an explicit rule |
| A subgroup margin of 0.08 pp is sufficient | Not established; depends on the clinical context |
| Quantization is always a non-substantial change | Not supported |

### 9.4 Why medical compression is different from benchmark compression

**The claim is specific.** A benchmark model claims an average accuracy; a device claims performance in a population, at a threshold, with a harm model. The same 0.4 pp change can be irrelevant or material depending on the claim.

**The evidence is audited.** A benchmark result is a number in a paper; a device result is a record that an auditor can inspect. The record must include the data, the code, the configuration, and the decision.

**The change is cumulative.** A device may be updated many times. Each change must be assessed against the approved claim and against the cumulative effect, not only against the previous version.

**The monitoring is mandatory.** A benchmark model is not monitored; a device is. The monitoring plan is part of the change, not an afterthought.

### 9.5 The practical lesson

The teams that handle compression changes well do three things.

- **They classify early.** The classification is made at design time, before the evidence is collected, so the evidence matches the pathway.
- **They pre-specify the margins.** The clinical and statistical margins are written before the experiment, so the result is a test of a hypothesis rather than a search for a justification.
- **They keep the record.** The configuration, the data, the code, and the decision are recorded as they happen, so the submission is an assembly rather than an archaeology project.

None of these is technically difficult. All of them are organizationally difficult, because they require the engineering team and the regulatory team to work in the same sequence.

### 9.6 Two lessons from the subgroup case

The subgroup failure is the most useful case in this topic because it separates three questions that teams conflate.

- **Is the model good enough?** The overall metric says yes; the subgroup says no. The answer depends on the population in the claim.
- **Is the change covered?** The plan covers per-channel scales, not per-tensor. The better method is covered; the faster one is not.
- **Is the evidence honest?** An aggregate-only report would be technically true and substantively misleading. The regulatory risk is in the reporting, not only in the model.

A team that answers all three separately makes a defensible decision. A team that answers only the first ships a device that passes its average and fails its population.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Unclassified change | "It's just a build setting" | No regulatory review | Compliance finding | Classify at design time |
| Over-classification | Every tweak becomes a submission | Risk aversion | Schedule slip | Use the decision table |
| Margin chosen after results | Margin equals the observed difference plus epsilon | Rationalization | Invalid evidence | Pre-specify |
| Test set used for calibration | Results unusually stable | Leakage | Invalid evidence | Disjoint sets |
| Subgroup omitted | Only overall reported | Aggregate masks harm | Equity harm | Pre-specify subgroups |
| Calibration shift ignored | ECE not reported | Threshold drift | Alarm-rate change | Report ECE and slope |
| Cumulative impact ignored | Each change individually covered | Drift accumulates | Device differs from claim | Cumulative assessment |
| Risk file not updated | New hazard not recorded | Process gap | Uncontrolled risk | Update with the change |
| Labeling not updated | Claims exceed evidence | Stale labeling | Misleading claim | Review labeling |
| Monitoring not planned | No post-market signal | Drift undetected | Late discovery | Define metrics and cadence |
| Evidence not reproducible | Missing data or code version | Poor record keeping | Submission rejected | Configuration record |
| Jurisdiction mismatch | US evidence for an EU submission | Different requirements | Delay | Map requirements per market |

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test |
|---|---|---|---|
| The approved configuration is reconstructable | Baseline for the change | Missing records | Audit the design history file |
| The PCCP covers the method | Classification | Narrow wording | Read the plan |
| The held-out set is independent | Valid statistics | Leakage | Check the split |
| The margins are clinically justified | Meaningful criteria | Arbitrary numbers | Clinical review |
| The subgroup is large enough | Adequate power | Wide interval | Power calculation |
| The target measurement matches the claim | Valid latency | Different conditions | Measure at the specified ambient |
| The risk file is current | Complete hazard analysis | Stale file | Review before the change |
| The monitoring can detect drift | Post-market safety | No signal | Design the metrics |
| The jurisdiction is known | Correct pathway | Multi-market launch | Map per market |
| The evidence is reproducible | Auditability | Missing versions | Configuration record |
| The clinical review is independent | Unbiased judgement | Same team | Separate reviewer |

The central limit is that the regulatory judgement is not a statistical one. A confidence interval can show that a difference is small; it cannot show that the difference is clinically acceptable. That judgement requires a clinician, a risk analysis, and a claim, and it cannot be automated.

A second limit is that the guidance is evolving. The PCCP guidance was finalized in December 2024, the AI Act's applicability dates are moving, and the standards are periodically revised. A classification that was correct at one date may not be correct at another; the record should state the date and the version of each instrument relied on.

## 12. Exercises

**Exercise 1 — Classification (application, 30 minutes).**
A model is quantized from FP32 to INT8, and the outputs change by at most 0.002 in probability. The PCCP covers "precision reduction to 8-bit integer arithmetic". Classify the change and state the evidence required.
*Deliverable:* a one-page classification.
*Acceptance criteria:* classification follows the decision table; evidence names the analytical bound, the statistical margin, and the target measurement.
*Time:* 30 minutes.

**Exercise 2 — Process map (analysis, 35 minutes).**
Map a distillation change to the processes in IEC 62304, ISO 14971, IEC 62366-1, and cybersecurity. State what changes in each.
*Deliverable:* a four-row table with records.
*Acceptance criteria:* each process has a specific effect and a record; the architecture change is noted.
*Time:* 35 minutes.

**Exercise 3 — Margin (calculation, 35 minutes).**
A proposed model has sensitivity 0.936 against 0.942, with 5,000 positives in each group. Compute the 95% CI and decide whether it passes a -1.5 pp margin.
*Deliverable:* a calculation and a verdict.
*Acceptance criteria:* SE and CI correct; verdict consistent with the lower bound.
*Time:* 35 minutes.

**Exercise 4 — Evidence package (design, 45 minutes).**
Design the evidence package for a quantized model, listing the documents and the acceptance criteria.
*Deliverable:* a table of documents and criteria.
*Acceptance criteria:* analytical, statistical, target, risk, labeling, and monitoring included.
*Time:* 45 minutes.

**Exercise 5 — Cumulative impact (analysis, 40 minutes).**
Three changes since approval each reduce sensitivity by 0.3 pp. The approved claim has a 1.0 pp margin. Assess the cumulative impact and state whether a new submission is required.
*Deliverable:* a short analysis.
*Acceptance criteria:* sums the effects; compares with the claim; notes that the sum may not be linear.
*Time:* 40 minutes.

**Exercise 6 — Risk file (design, 35 minutes).**
Write three risk-file entries for a quantized model: a subgroup degradation, a calibration shift, and a saturation on out-of-distribution input. Include hazard, situation, control, and residual risk.
*Deliverable:* three entries.
*Acceptance criteria:* each entry has the four elements; controls are specific; residual risk is judged.
*Time:* 35 minutes.

**Exercise 7 — Monitoring plan (design, 40 minutes).**
Design a post-market monitoring plan for a quantized ECG model, with metrics, subgroups, thresholds, and cadence.
*Deliverable:* a one-page plan.
*Acceptance criteria:* metrics include alert rate, fallback rate, and saturation; subgroups are named; thresholds and cadence are specified.
*Time:* 40 minutes.

**Exercise 8 — Change record (design, 50 minutes).**
Write the change record for the Section 6 quantization, including description, classification, evidence, impact assessment, and approval path.
*Deliverable:* a two-page record.
*Acceptance criteria:* reproducible; classification justified; evidence linked; cumulative impact stated.
*Time:* 50 minutes.

## 13. Self-check answers

**A1. Classification for Exercise 1.**
The change matches the plan's description and the outputs differ within a small tolerance. The classification is PCCP-covered, provided the protocol's acceptance criteria are met. The evidence is an analytical bound (half a step per value), a statistical comparison with a pre-specified margin, a subgroup and calibration analysis, and a target measurement.

**A2. Why a statistical non-significance result is not equivalence.**
Because non-significance can reflect low power rather than a small difference. Equivalence requires a pre-specified margin and a confidence interval that excludes the margin, not a p-value above 0.05. A small study can fail to detect a large difference and still be non-significant.

**A3. CI for Exercise 3.**
Difference = -0.006. SE = sqrt(0.942×0.058/5000 + 0.936×0.064/5000) = sqrt(1.093e-5 + 1.198e-5) = 0.00479. 95% CI = -0.006 ± 0.00939 = [-0.0154, +0.0034]. The lower bound is -1.54 pp, below the -1.5 pp margin, so the change fails narrowly.

**A4. Why the calibration slope matters at the threshold.**
Because the operating threshold is a probability, not a rank. A slope below 1 means predicted probabilities are too extreme, so the same threshold corresponds to a different risk. Sensitivity and the alarm rate can change even if the ranking is preserved.

**A5. Cumulative impact for Exercise 5.**
Three changes of -0.3 pp sum to -0.9 pp, within the 1.0 pp margin if the effects are additive. If the effects interact, the combined effect can be larger; the correct approach is to measure the combined model, not to sum the individual results. A new submission is required if the measured cumulative effect exceeds the claim, or if the claim's confidence interval no longer covers the current performance.

**A6. Why the risk file must be updated even when the overall metric passes.**
Because the hazard is not the overall metric; it is the subgroup, the threshold, or the tail. A model can pass overall and create a new hazardous situation in a subgroup. The risk file records the hazard and the control, regardless of the aggregate result.

**A7. What makes a monitoring plan adequate?**
It must be able to detect the effect the change could cause. For a quantized model, that means monitoring the alert rate, the fallback rate, and the saturation rate, stratified by subgroup, with thresholds and a review cadence. A plan that monitors only overall accuracy cannot detect a subgroup drift.

**A8. Why the approved configuration, not the repository, is the baseline.**
Because the device that was cleared is the device described in the submission, which may differ from the current code. The change is assessed against what was approved, not against what the team last built.

**A9. Rubric for Exercise 2 (process map).**
A strong map names each process, the specific effect of distillation, and the record that changes. It notes that distillation changes the architecture and therefore the design description, the verification, and possibly the intended-use evidence. A weak map lists the standards without the specific effects.

**A10. Rubric for Exercise 8 (change record).**
A strong record includes the description, the classification with a rationale, the analytical and statistical evidence, the target measurements, the impact assessment, the risk and labeling updates, the monitoring plan, and the approval. A weak record describes the model change and the final accuracy only.

## 14. Interview questions

**Q1. When does a compression change become a regulatory change?**
A strong answer says when it changes outputs or performance beyond the approved envelope, regardless of intent, and describes the classification. A weak answer says when the model file changes.

**Q2. What is a PCCP?**
A strong answer describes the three components (description, protocol, impact assessment) and the pathway. A weak answer calls it a change log.

**Q3. What is a non-inferiority margin?**
A strong answer defines it as the largest acceptable loss, pre-specified and clinically justified, and explains the confidence-interval decision rule. A weak answer says it is a small difference.

**Q4. Why is statistical non-significance not equivalence?**
A strong answer explains power, the need for a margin, and the difference between a p-value and an interval. A weak answer treats p > 0.05 as proof of no difference.

**Q5. Which processes does a quantization change touch?**
A strong answer names requirements, design, verification, risk, usability, cybersecurity, labeling, and monitoring, with a record for each. A weak answer names testing only.

**Q6. Why is the subgroup analysis required?**
A strong answer explains that aggregate accuracy can hide a subgroup loss, and that the harm model is subgroup-specific. A weak answer says it is good practice.

**Q7. What is cumulative impact?**
A strong answer explains that multiple individually covered changes can drift beyond the claim, and that the combined model must be measured. A weak answer sums the changes and assumes linearity.

**Q8. How do you classify a distillation change?**
A strong answer says it changes the architecture and is usually a new submission, unless the plan explicitly covers architecture changes within bounds. A weak answer treats it like quantization.

**Q9. What evidence does a reviewer expect for a quantized model?**
A strong answer lists the analytical bound, the statistical comparison with intervals, subgroup and calibration results, target measurements, risk and labeling updates, and monitoring. A weak answer lists accuracy only.

**Q10. Why must the target measurement condition match the claim?**
A strong answer explains that latency depends on ambient, clock, and thermal state, and a claim measured at 22 °C may not hold at 40 °C. A weak answer assumes the board is the board.

**Q11. What is the difference between IEC 62304 and ISO 14971?**
A strong answer says 62304 is the software lifecycle and 14971 is risk management, and that a compression change touches both. A weak answer conflates them.

**Q12. When would you reject a compression change on regulatory grounds?**
A strong answer rejects on subgroup harm, calibration shift at the threshold, an uncovered change, unverifiable evidence, or a cumulative drift beyond the claim. A weak answer rejects only on accuracy.

## 15. Key terms

- **510(k)** — premarket notification pathway in the United States.
- **AI Act** — European regulation for AI systems.
- **Analytical bound** — mathematical limit on a change's effect.
- **Calibration** — agreement between predicted probability and observed frequency.
- **Change control** — process for managing changes.
- **Clinical equivalence** — no clinically meaningful difference.
- **Configuration item** — item under version control.
- **Cumulative impact** — combined effect of multiple changes.
- **De Novo** — pathway for novel low-to-moderate-risk devices.
- **Decision curve** — net benefit across thresholds.
- **ECE** — expected calibration error.
- **Equivalence** — similarity within a justified margin.
- **Essential performance** — performance necessary to keep risk acceptable.
- **GMLP** — Good Machine Learning Practice principles.
- **Hazard** — potential source of harm.
- **Hazardous situation** — circumstance in which harm can occur.
- **IEC 62304** — medical device software lifecycle standard.
- **IEC 62366-1** — usability engineering standard.
- **IEC 60601-1** — general safety and essential performance standard.
- **Impact assessment** — analysis of benefits, risks, and cumulative effects.
- **Intended use** — regulator's description of the device's purpose.
- **ISO 14971** — risk management standard.
- **Labeling** — instructions, warnings, and claims.
- **MDR** — European Medical Devices Regulation.
- **Modification protocol** — pre-specified method for a change.
- **Non-inferiority margin** — largest acceptable loss.
- **PCCP** — predetermined change control plan.
- **PMA** — premarket approval for high-risk devices.
- **Post-market surveillance** — ongoing monitoring after launch.
- **Reference standard** — truth used to judge outputs.
- **Residual risk** — risk remaining after controls.
- **Risk file** — record of hazards and controls.
- **Slope** — calibration slope.
- **Special 510(k)** — 510(k) for a device modification.
- **Subgroup** — population defined by a clinically relevant attribute.
- **Substantial modification** — change that alters safety or effectiveness.
- **Traceability** — ability to link requirements to evidence.
- **Performance-neutral change** — a change that does not alter decisions within tolerance.
- **Behaviour-shifting change** — a change that preserves the average but shifts errors.
- **Subgroup margin** — the non-inferiority margin for a defined population.
- **Model hash** — identifier of the exact weights under assessment.
- **Protocol deviation** — a departure from the pre-specified protocol.
- **Claim** — the performance statement in the labeling or submission.
- **Verification** — evidence that a requirement is met.
- **Validation** — evidence that the device meets its intended use.
- **Vigilance** — reporting of adverse events to the regulator.
- **Conformity assessment** — the process of demonstrating compliance.
- **Notified body** — an organization that assesses conformity in the EU.

## 16. Sources

All URLs accessed 25 October 2026.

**FDA guidance**
- FDA, "Marketing Submission Recommendations for a Predetermined Change Control Plan for Artificial Intelligence-Enabled Device Software Functions", final guidance, 3 December 2024. https://www.fda.gov/regulatory-information/search-fda-guidance-documents/marketing-submission-recommendations-predetermined-change-control-plan-artificial-intelligence
- FDA, Health Canada, MHRA, "Good Machine Learning Practice for Medical Device Development: Guiding Principles", October 2021. https://www.fda.gov/media/153486/download
- FDA, "Cybersecurity in Medical Devices". https://www.fda.gov/medical-devices/digital-health-center-excellence/cybersecurity

**Standards**
- IEC 62304:2006+AMD1:2015. https://webstore.iec.ch/en/publication/22794
- ISO 14971:2019. https://www.iso.org/standard/72704.html
- IEC 60601-1. https://www.iec.ch/
- IEC 62366-1. https://www.iec.ch/

**European Union**
- Regulation (EU) 2017/745 (Medical Devices Regulation). https://eur-lex.europa.eu/eli/reg/2017/745/oj
- Regulation (EU) 2024/1689 (Artificial Intelligence Act). https://eur-lex.europa.eu/eli/reg/2024/1689/oj
- MDCG 2025-6, "Medical Devices Regulation (IVDR) and the Artificial Intelligence Act (AIA)", June 2025. https://health.ec.europa.eu/document/download/b78a17d7-e3cd-4943-851d-e02a2f22bbb4_en

**Quantization**
- Jacob, B., et al., "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference", 2017/2018. https://arxiv.org/abs/1712.05877
- Nagel, M., et al., "A White Paper on Neural Network Quantization", 2021. https://arxiv.org/abs/2106.08295

> **Note:** This topic is educational and is not legal or regulatory advice. The Section 6 classification is hypothetical; a real classification depends on the approved plan, the jurisdiction, and professional review.

## 17. Links

- Module index: [`_index.md`](_index.md)
- Survey file: [`../../course/14B_efficient_ai_distillation_quantization_on_device_inference.md`](../../course/14B_efficient_ai_distillation_quantization_on_device_inference.md)
- Previous topic: [`08_on-device-architecture.md`](08_on-device-architecture.md)
- Next topic: [`10_validation-on-target-hardware.md`](10_validation-on-target-hardware.md)
- Related topic: [`05_quantization.md`](05_quantization.md)
- Related topic: [`01_the-deployment-pareto-frontier.md`](01_the-deployment-pareto-frontier.md)
- Lab: [`../../labs/14B_quantization.ipynb`](../../labs/14B_quantization.ipynb)
- Lab directory: [`../../labs/`](../../labs/)
