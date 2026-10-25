# Digital Biomarkers: Definition, Context of Use, Qualification, and Evidence

> Module 11 — Physiological Signals and Wearables
> Topic 08 of 10
> Estimated study time: 4 hours
> Prerequisites: Topic 01 (signal fundamentals), Topic 06 (feature engineering), Topic 07 (deep learning)
> Version 2026.3

## 1. Why this topic exists

A digital biomarker is a characteristic derived from digital data that indicates a biological or behavioral process, a disease state, or a response to an intervention. The definition is broad, and the breadth is the problem. A step count, a heart-rate variability index, a sleep stage, a gait speed, and a voice feature can all be called digital biomarkers. The word "biomarker" implies a validated relationship to biology; the word "digital" implies a sensor and an algorithm. Neither implication is automatically true.

The industry problem is that the word is used before the evidence exists. A company reports a "digital biomarker" for stress, fatigue, or disease progression when it has a feature and a correlation. A regulator asks for the context of use, the analytical validation, and the clinical evidence, and the response is a slide deck. A payer asks whether the biomarker changes a decision, and the response is a pilot. The gap between the word and the evidence is the subject of this topic.

Four failures recur.

**The feature called a biomarker.** A feature is a computed number. A biomarker is a feature with a demonstrated relationship to a biological process, in a stated context of use. Without the relationship and the context, the feature is a measurement looking for a meaning.

**The context of use omitted.** A biomarker validated for "monitoring disease progression in adults with a specific diagnosis in a clinical trial" is not validated for "screening the general population." The context of use is part of the definition, and a claim that omits it is not a claim.

**The analytical validation skipped.** A model that converts sensor data into a measurement must be validated against a reference. If the conversion is not validated, the downstream clinical association is uninterpretable: the biomarker may be associated with the outcome through the sensor's error, not the biology.

**The endpoint conflated with the biomarker.** A biomarker is a measurement; an endpoint is a variable in a trial. A biomarker can be an endpoint if it is used to answer the trial's question, but the endpoint has additional requirements: a definition, a timing, a combining rule, and a clinical interpretation. A biomarker that is not an endpoint can still be useful, and an endpoint that is not a biomarker can still be valid.

This topic covers the digital biomarker chain: the definition and taxonomy, the context of use, the V3 validation framework (verification, analytical validation, clinical validation), usability, bias, evidence synthesis, and the regulatory pathways. It builds on the measurement and feature topics and sets up Topic 09 for validation design.

### 1.1 Three composite failures

**The step count that measured the wrist.** A company markets a wrist step count as a measure of functional capacity. The analytical validation compared the wrist device with a hip accelerometer and reported a correlation of 0.95. The device counts arm movements during cooking and folding laundry as steps. In a trial, the treatment group has more household activity because of a behavioral intervention, and the step count rises without a change in walking. The biomarker is associated with the outcome through the behavior, not the physiology. The cost is a failed trial endpoint.

**The sleep stage that was a two-class problem.** A wearable reports five sleep stages. The validation reports 85% overall accuracy against polysomnography. The per-stage sensitivity for REM is 40%. The weighted accuracy is high because most epochs are N2. A trial uses the REM measure as an endpoint and finds no effect. The cost is a null result that may be a measurement failure, not a biological one.

**The biomarker that was a feature.** A startup reports a "digital biomarker of stress" with a correlation of 0.4 against a laboratory stress task. A payer asks for the context of use, the analytical validation, and the clinical utility. The response is the correlation. The cost is a reimbursement rejection, which could have been predicted from the evidence gap.

The three failures share a structure: the evidence for one link in the chain is presented as evidence for the whole chain. A correlation is not agreement; agreement is not clinical validation; clinical validation is not utility. The chain has to be validated link by link.

### 1.2 Why the words matter

The words are not academic. A "biomarker" claim triggers a regulatory pathway; a "feature" does not. An "endpoint" claim requires a pre-specified analysis; a "monitoring measure" does not. A "qualified" biomarker can be used across programs; a "validated" measurement cannot.

A team that uses the words loosely will discover the difference at the review. A team that uses them precisely can plan the evidence and the timeline. The precision is a commercial advantage, not only a scientific one.

A second reason is that the words determine what a regulator will review. A feature that is described as a measurement may be reviewed as software; a feature described as a biomarker may be reviewed as a diagnostic. The description is part of the submission.

A third reason the words matter is that they determine what a study is for. A verification study is a bench study; an analytical validation is an agreement study; a clinical validation is a cohort; a utility study is an impact study. Each has a different cost, duration, and sample size. A team that confuses them will budget for the wrong study.

## 2. Learning outcomes

By the end of this topic you can:

1. **Distinguish** a measurement, a feature, a biomarker, and an endpoint, and place a given claim in the taxonomy.
2. **Write** a context-of-use statement with the population, the setting, the decision, and the timing.
3. **Design** analytical, technical, and clinical validation for a digital biomarker.
4. **Compute** agreement statistics (bias, limits of agreement, ICC, MAE) and interpret them for a stated decision.
5. **Evaluate** a biomarker's clinical utility with a decision curve or a net-benefit analysis.
6. **Assess** usability and bias evidence for a wearable-derived biomarker.
7. **Map** a biomarker to a regulatory pathway (qualification, device clearance, or neither).
8. **Write** an evidence summary that separates what is established from what is inferred.

Each outcome maps to an exercise in Section 12 and a self-check item in Section 13.

The outcomes are ordered from definition to evidence to regulation. The middle outcomes (agreement, utility, usability, bias) are the ones that distinguish a validated biomarker from a feature. The final outcome (evidence summary) is the one that a reviewer reads first.

## 3. The core mental model

A digital biomarker is a chain from a sensor to a decision, and every link has evidence requirements.

```text
 sensor          ->  signal        ->  measurement      ->  feature
 (hardware)          (raw)             (unit, method)       (computed)
    |                   |                   |                   |
 verification      quality            analytical           technical
 (does it sense?)  (is it valid?)     validation           validation
                                        (does it measure?)   (does it derive?)
    |                   |                   |                   |
    +-------------------+-------------------+-------------------+
                                        |
                                   biomarker
                              (relationship to biology)
                                        |
                                  clinical validation
                                        |
                                   endpoint / decision
                                        |
                                  clinical utility
                                        |
                                   usability / adoption
```

Redraw it with the evidence type under each arrow. The key idea is that the evidence is cumulative: a failure at any link invalidates the downstream claim, and a success at one link does not substitute for another.

### 3.1 The BEST taxonomy

The FDA-NIH BEST (Biomarkers, EndpointS, and other Tools) resource defines a biomarker as a defined characteristic measured as an indicator of normal biological processes, pathogenic processes, or responses to an exposure or intervention. It explicitly excludes measures of how a patient feels, functions, or survives, which are clinical outcome assessments.

BEST groups biomarkers into seven categories:

| Category | Question | Example |
|---|---|---|
| Susceptibility/risk | Who is at risk | Genetic risk score |
| Diagnostic | Does the disease exist | Troponin for infarction |
| Monitoring | How is the disease changing | HbA1c over time |
| Prognostic | What will happen | NYHA class |
| Predictive | Who will respond | HER2 status |
| Response | Did the intervention work | LDL change |
| Safety | Is there harm | Liver enzymes |

A digital biomarker can belong to any category. A step count in a heart-failure trial can be a monitoring biomarker; a gait speed in a Parkinson's trial can be a prognostic or response biomarker. The category determines the evidence and the claim.

### 3.2 Context of use

The context of use (COU) is a statement that fully describes how the biomarker is used and for what regulatory or clinical purpose. It includes:

- the population,
- the setting,
- the decision the biomarker supports,
- the timing and frequency,
- the comparator or reference,
- the intended user.

A COU is not a marketing sentence. "Monitor disease progression in adults with ALS in a clinical trial, using a weekly gait measure, to support an efficacy endpoint" is a COU. "Track your health" is not.

### 3.3 V3 and V3+

The Digital Medicine Society's V3 framework defines three validation components:

1. **Verification**: does the sensor accurately and reliably capture the raw data against technical criteria?
2. **Analytical validation**: does the algorithm convert the sensor output into the physiological metric, compared with a reference, in a specific population?
3. **Clinical validation**: does the digital measure reflect or predict the clinical state or outcome in the target population and context of use?

V3+ adds a fourth component: **usability validation**, which asks whether the intended users can use the technology safely and effectively at scale. The addition matters because a biomarker that is accurate in a lab and unusable at home is not fit for purpose.

### 3.4 Measurement, feature, biomarker, endpoint

- **Measurement**: step count per day, in steps.
- **Feature**: mean daily step count over 7 days, or the 90th percentile of hourly step counts.
- **Biomarker**: the feature with a validated relationship to functional capacity in a stated population.
- **Endpoint**: the variable in a trial, with a definition, a timing, and an analysis plan.

The same number can be a measurement, a feature, a biomarker, or an endpoint depending on the evidence and the use. The words are not interchangeable, and a document that uses them interchangeably is usually hiding a missing link.

The progression is also a progression of evidence. A measurement needs a method; a feature needs a definition; a biomarker needs an association; an endpoint needs a trial and an analysis plan. Each step adds a requirement, and a claim cannot skip a step.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Digital biomarker | Digital-data-derived indicator of biology, behavior, or response | Calling any feature a biomarker | varies |
| Measurement | Quantity with unit and method | Confusing with feature | unit |
| Feature | Computed descriptor | Calling it a biomarker | unit |
| Endpoint | Precisely defined trial variable | Confusing with biomarker | unit |
| Context of use | Who, what, where, why | Omitting the population | statement |
| Qualification | Formal regulatory conclusion for a COU | Calling any validation qualification | regulatory |
| Verification | Sensor accuracy against technical criteria | Confusing with analytical validation | evidence |
| Analytical validation | Algorithm accuracy against a reference | Confusing with clinical validation | evidence |
| Clinical validation | Association with a clinical state or outcome | Confusing with utility | evidence |
| Clinical utility | Does the biomarker improve decisions or outcomes | Confusing with association | evidence |
| Usability validation | Can users use it correctly | Treating as optional | evidence |
| Reference standard | The best available truth | Treating as error-free | method |
| Agreement | How close two measurements are | Confusing with correlation | method |
| Bias | Mean difference | Confusing with error | unit |
| Limits of agreement | Bias ± 1.96 SD | Reporting without the plot | unit |
| ICC | Intraclass correlation coefficient | Using as the only agreement metric | dimensionless |
| MAE | Mean absolute error | Confusing with RMSE | unit |
| RMSE | Root mean square error | Sensitive to outliers | unit |
| MARD | Mean absolute relative difference | Using for non-positive values | % |
| Correlation | Linear association | Confusing with agreement | dimensionless |
| Bland-Altman | Agreement plot | Reporting r instead | method |
| Sensitivity to change | Ability to detect change | Confusing with cross-sectional validity | method |
| Responsiveness | Change with intervention | Confusing with validity | method |
| Minimal important difference | Smallest clinically meaningful change | Confusing with statistical significance | unit |
| Decision curve | Net benefit across thresholds | Confusing with ROC | method |
| Net benefit | Utility at a threshold | Ignoring prevalence | dimensionless |
| Surrogate endpoint | Endpoint used in place of a clinical outcome | Assuming validation | endpoint |
| Clinical outcome assessment | Measure of how a patient feels or functions | Confusing with biomarker | COA |
| COA | Clinical outcome assessment | Confusing with digital biomarker | PRO/ClinRO/ObsRO/PerfO |
| PRO | Patient-reported outcome | Assuming a wearable is a PRO | instrument |
| Fit for purpose | Adequate for the COU | Claiming without evidence | judgement |
| Analytical range | Range over which the method is validated | Extrapolating | unit |
| Limit of detection | Smallest detectable value | Confusing with sensitivity | unit |
| Precision | Repeatability of a measurement | Confusing with accuracy | unit |
| Repeatability | Same operator, same conditions | Confusing with reproducibility | unit |
| Reproducibility | Different operators or conditions | Confusing with repeatability | unit |
| Conformal | Consistent with the construct | Assuming from correlation | judgement |
| Ecological validity | Performance in the real world | Assuming from lab | judgement |
| Digital endpoint | Endpoint derived from digital data | Assuming qualification | endpoint |
| Analytical validation | Algorithm accuracy against a reference | Confusing with verification | evidence |
| Clinical validation | Association with disease or outcome | Confusing with utility | evidence |
| Fit-for-purpose | Adequate for the COU | Claiming without evidence | judgement |
| Qualification | Formal regulatory conclusion | Calling a validation a qualification | regulatory |
| Surrogate | Endpoint used instead of a clinical outcome | Assuming it is validated | endpoint |
| COA | Clinical outcome assessment | Confusing with biomarker | PRO/ClinRO/ObsRO/PerfO |
| PRO | Patient-reported outcome | Assuming a device is a PRO | instrument |
| Observer-reported outcome | Reported by a caregiver or clinician | Confusing with PRO | instrument |
| Performance outcome | Measured by a task | Confusing with biomarker | instrument |
| Construct | The thing being measured | Confusing with the measurement | concept |
| Proxy | Indirect measure of a construct | Assuming equivalence | measurement |
| Ecological validity | Real-world performance | Assuming from lab | judgement |
| Analytical range | Validated measurement range | Extrapolating | unit |
| Limit of detection | Smallest detectable value | Confusing with sensitivity | unit |
| Repeatability | Same conditions | Confusing with reproducibility | unit |
| Reproducibility | Different conditions | Confusing with repeatability | unit |
| Responsiveness | Change with intervention | Confusing with association | method |
| Sensitivity to change | Ability to detect change | Confusing with responsiveness | method |
| MID | Minimal important difference | Confusing with statistical significance | unit |
| Ceiling effect | Measure saturates | Ignoring in validation | method |
| Floor effect | Measure bottoms out | Ignoring in validation | method |
| Practice effect | Improvement with repetition | Confusing with treatment | method |
| Digital divide | Unequal access to devices | Ignoring in equity | concept |
| Data provenance | Origin and history of data | Ignoring in validation | metadata |

## 5. The mechanics, step by step

The procedure is written for a digital biomarker intended to support a clinical trial endpoint or a clinical decision. It follows the V3+ structure and adds the evidence synthesis and the regulatory mapping.

### Step 1. Write the context of use

**Input:** the decision, the population, the setting, and the available sensor.

**Output:** a one-paragraph COU statement with the population, the decision, the timing, the comparator, and the user.

**Rationale:** the COU is the specification for every later step. Without it, the validation has no target and the claim has no boundary. The FDA biomarker qualification program uses the format "[biomarker category] to [drug development use]"; a device or clinical claim uses an analogous format.

**Branch point:** if the COU is exploratory, the evidence can be lighter and the claim narrower. If the COU supports a primary endpoint, the evidence must be strong and the analysis pre-specified.

**Edge case:** a COU that says "adults" without an age range, or "monitoring" without a frequency, is under-specified. The population and the timing are part of the claim.

**A COU template.** The statement should answer six questions in one paragraph.

| Question | Example |
|---|---|
| Population | Adults 40-75 with confirmed diagnosis |
| Setting | Outpatient, 12-week trial |
| Decision | Monitor functional capacity |
| Timing | Weekly, 7-day mean |
| Comparator | Research accelerometer |
| User | Trial coordinator |

A COU that cannot fill the table is not ready for validation. The missing row is the first thing to fix.

### Step 2. Define the construct and the measurement model

**Input:** the COU and the biology or behavior.

**Output:** a construct definition, a measurement model, and a statement of what the sensor observes.

**Rationale:** the construct is the thing the biomarker is supposed to indicate, such as functional capacity, sleep quality, or disease progression. The measurement model maps the sensor signal to the construct through a chain of proxies. Each proxy has an assumption.

**Arithmetic:** a step count is a proxy for ambulation, which is a proxy for functional capacity. The first proxy is close; the second is weaker. A biomarker claim that says "functional capacity" from a step count is making the second inference, and the validation must support it.

**Branch point:** if the construct is multidimensional, a single feature may not capture it. A composite biomarker needs a combining rule and a validation of the composite.

**Edge case:** a construct that cannot be measured directly, such as fatigue, has no gold standard. The validation is a construct-validity argument, not an agreement study.

**The proxy ladder.** Each step from sensor to construct is an inference, and the evidence weakens with each step.

| Level | Example | Evidence |
|---|---|---|
| Sensor | Accelerometer counts | Verification |
| Signal | Step events | Algorithm validation |
| Measurement | Steps/day | Analytical validation |
| Behavior | Ambulation | Convergent validity |
| Construct | Functional capacity | Clinical validation |
| Decision | Rehabilitation | Utility |

A claim that jumps from steps/day to functional capacity skips a level. The validation must cover the skipped inference, or the claim must be reduced.

### Step 3. Specify the measurement and the algorithm

**Input:** the sensor, the signal, and the construct.

**Output:** a measurement specification with the sensor, the sampling, the algorithm, the unit, and the output.

**Rationale:** the measurement specification is the object of the analytical validation. It must be precise enough that another team can reproduce the output from the raw data.

**Contents:** device and firmware, sensor site, sampling rate, filter, algorithm version, window, aggregation, unit, and output range.

**Branch point:** if the algorithm is a deep model, the specification includes the architecture, the weights, and the preprocessing. A model without a version is not a measurement.

**Edge case:** a firmware update that changes the sampling or the filter is a change in the measurement. The validation must be repeated or the change must be shown to be equivalent.

**The measurement specification table.**

| Field | Example | Why it matters |
|---|---|---|
| Device | Model and firmware | Reproducibility |
| Site | Wrist, dominant | Measurement |
| Sampling | 50 Hz | Timing and content |
| Filter | 0.5-8 Hz | Signal band |
| Algorithm | Version 2.1 | Reproducibility |
| Window | 7 days | Aggregation |
| Unit | Steps/day | Comparability |
| Output range | 0-30,000 | Analytical range |

A specification that omits the firmware or the algorithm version is not reproducible. A specification that omits the output range hides extrapolation.

### Step 4. Verify the sensor

**Input:** the device and a bench reference.

**Output:** a verification report with accuracy, precision, and environmental sensitivity.

**Rationale:** verification asks whether the sensor captures the raw signal accurately. It is a hardware and firmware question, not a clinical one.

**Tests:** a known input (a shaker, a simulator, a calibration standard), a repeatability test, a temperature and humidity test, and a battery and power test. For optical sensors, a skin phantom or a reference oximeter; for accelerometers, a turntable or a shaker.

**Arithmetic:** a step counter can be verified with a mechanical shaker at a known cadence. At 100 steps/min, the expected count in 10 min is 1,000. A device that reports 950 has a -5% bias at that cadence.

**Branch point:** if the sensor fails verification, the downstream validation is not meaningful. Fix the hardware or the firmware first.

**Edge case:** a sensor can pass a bench test and fail on the body because the body is not a bench. Verification is necessary, not sufficient.

**Verification test table.**

| Test | Input | Metric | Pass criterion |
|---|---|---|---|
| Accuracy | Known standard | Bias, SD | Study-specific |
| Repeatability | Same input, repeated | CV | Study-specific |
| Temperature | 5-40 °C | Drift | Study-specific |
| Humidity | 20-80% RH | Drift | Study-specific |
| Battery | Full discharge | Dropout | No gap |
| Drop | 1 m | Function | No failure |

The criteria are study-specific because the sensor and the use differ. A step counter and a pulse oximeter do not share a verification protocol. The protocol should be written before the test, not after.

### Step 5. Perform analytical validation

**Input:** the device output and a reference measurement in the target population.

**Output:** bias, limits of agreement, ICC, MAE, and a Bland-Altman plot.

**Rationale:** analytical validation asks whether the algorithm converts the sensor output into the intended measurement. The reference is the best available measure of the same construct, not a different construct.

**Method:** simultaneous measurement with the device and the reference, a sample that spans the analytical range, and an analysis that reports agreement rather than correlation.

**Arithmetic:** for 120 paired step counts, a bias of 86.5 steps, an SD of 338.7 steps, and limits of agreement of -577.4 to 750.4 steps mean that a device reading can differ from the reference by up to about 750 steps in 95% of cases. The ICC of 0.978 is high because the between-subject variance is large; the limits of agreement show that the individual agreement is modest.

**Branch point:** if the agreement is poor, the measurement may still be useful for a group-level comparison, where errors average out. It is not useful for an individual-level decision.

**Edge case:** a high correlation with a wide limits of agreement is a classic trap. The correlation measures association, not agreement.

**The agreement report.** A complete agreement report contains:

| Item | Example | Why it matters |
|---|---|---|
| n | 120 pairs | Precision |
| Reference | Research accelerometer | Truth |
| Range | 1,800-11,500 steps/day | Analytical range |
| Bias | 86.5 steps/day | Systematic error |
| SD of differences | 338.7 steps/day | Random error |
| Limits of agreement | -577 to 750 | Individual agreement |
| MAE | 278.2 steps/day | Absolute error |
| ICC | 0.978 | Between-subject discrimination |
| Plot | Bland-Altman | Pattern and outliers |

A report that omits the plot hides heteroscedasticity and outliers. A report that omits the range hides extrapolation. A report that gives only the ICC hides the individual error.

### Step 6. Perform clinical validation

**Input:** the biomarker and a clinical outcome in the target population.

**Output:** an association, a prediction, or a diagnostic accuracy estimate, with confidence intervals.

**Rationale:** clinical validation asks whether the biomarker reflects or predicts the clinical state. It is a different question from analytical validation, and it requires a different study.

**Designs:** cross-sectional association, prospective cohort, case-control, or nested case-control. The design determines the claim: a cross-sectional association does not establish prediction.

**Arithmetic:** a biomarker with an AUC of 0.70 for a clinical outcome has moderate discrimination. At a 10% prevalence, a sensitivity of 0.80 and a specificity of 0.70 give a PPV of 0.23 and an NPV of 0.97. The PPV is low because the prevalence is low.

**Branch point:** if the clinical validation is negative, the biomarker may still be a monitoring biomarker if it changes with treatment. The category determines the required evidence.

**Edge case:** a biomarker validated in a case-control sample has a different prevalence from the population. The PPV does not transfer without a prior correction.

### Step 7. Evaluate clinical utility

**Input:** the biomarker, the decision, and the outcome.

**Output:** a net-benefit analysis, a decision curve, or an impact study.

**Rationale:** utility asks whether using the biomarker improves decisions or outcomes. An association is not utility; a biomarker can be associated with an outcome and still not change what a clinician does.

**Method:** a decision curve compares the net benefit of using the biomarker with the strategies of treating all and treating none, across thresholds. An impact study randomizes the use of the biomarker and measures the outcome.

**Arithmetic:** at a threshold of 0.10, a model with 80% sensitivity and 15% false-positive rate has a net benefit of 0.065, compared with 0.000 for treat-all and 0 for treat-none. The model is useful at that threshold. At a threshold of 0.05, the net benefit is 0.073, compared with 0.053 for treat-all; the advantage is smaller.

**Branch point:** if the biomarker does not improve net benefit at the clinically relevant threshold, it does not have utility, even if it is statistically associated.

**Edge case:** a decision curve assumes that the threshold reflects the clinical trade-off. A threshold chosen for statistical convenience is not a clinical threshold.

**Utility study designs.**

| Design | Question | Strength | Weakness |
|---|---|---|---|
| Decision curve | Net benefit across thresholds | No outcome needed | Assumes probabilities |
| Impact trial | Does use improve outcome | Causal | Expensive |
| Stepped wedge | Does rollout improve outcome | Real-world | Time confounding |
| Before-after | Did practice change | Cheap | Secular trend |
| Qualitative | How is it used | Context | Not generalizable |

The design should match the claim. A decision curve supports a "may improve decisions" claim; an impact trial supports an "improves outcomes" claim. The two are not interchangeable.

### Step 8. Assess usability

**Input:** the device, the instructions, and representative users.

**Output:** a usability report with task success, errors, and accessibility findings.

**Rationale:** a biomarker that is accurate but difficult to use will be used incorrectly or not at all. Usability is a validation component, not a marketing feature.

**Method:** formative usability testing with 5 to 10 users per round, a summative test with a larger sample, and a human factors analysis of use errors. The tasks include charging, wearing, syncing, and interpreting the output.

**Arithmetic:** if 20% of users fail to wear the device correctly, the effective sample is 80% of the intended sample, and the missingness is not random. The biomarker's performance in the full sample is not the performance in the users who wear it correctly.

**Branch point:** if usability fails, the product needs a design change, not a better algorithm.

**Edge case:** a usability test with staff assistance overestimates real-world success. The test should include unassisted use.

**Usability metrics.**

| Metric | Definition | Target |
|---|---|---|
| Task success | Users completing the task | Study-specific |
| Time on task | Time to complete | Study-specific |
| Use errors | Errors with consequence | Zero critical |
| Assistance | Help required | Zero in summative |
| Satisfaction | User rating | Study-specific |
| Adherence | Wear time | Study-specific |

The critical errors are the ones that produce wrong data or no data. A user who wears the device on the wrong wrist produces a biased measurement; a user who does not charge it produces a gap. Both are usability failures, not algorithm failures.

### Step 9. Assess bias and equity

**Input:** the validation sample and the metadata.

**Output:** performance by skin tone, age, sex, body composition, and socioeconomic group, where available.

**Rationale:** a biomarker can perform differently across groups for sensor, physiological, or behavioral reasons. The difference is a property of the measurement, not only of the model.

**Arithmetic:** a step counter can undercount in a population with a different gait pattern, and an optical sensor can be biased by skin tone. A pooled MAE can hide a subgroup difference that is clinically meaningful.

**Branch point:** if a subgroup difference is found, the response is to understand the mechanism and to adjust the claim, the algorithm, or the population.

**Edge case:** a subgroup with few participants has a wide confidence interval. The absence of a detected difference is not evidence of equivalence.

### Step 10. Synthesize the evidence

**Input:** the verification, analytical, clinical, utility, usability, and bias studies.

**Output:** an evidence table with the question, the design, the result, and the limitation for each component.

**Rationale:** the evidence is cumulative and each component answers a different question. A table makes the gaps visible.

**Contents:** the COU, the construct, the measurement, the reference, the sample, the result, the confidence interval, and the limitation.

**Branch point:** if a component is missing, the biomarker is not qualified for the COU. The gap should be stated, not hidden.

**Edge case:** a study can be internally valid and externally limited. The table should distinguish the two.

**An evidence table template.**

| Component | Question | Design | Result | Limitation |
|---|---|---|---|---|
| Verification | Does the sensor sense | Bench | Bias | Bench only |
| Analytical | Does it measure | Agreement | LoA | Population |
| Clinical | Does it reflect disease | Cohort | AUC | Single site |
| Utility | Does it change decisions | Decision curve | Net benefit | Assumed threshold |
| Usability | Can users use it | Human factors | Success | Lab setting |
| Bias | Does it work for all | Stratified | Subgroup gap | Sample size |

The table is the summary that a reviewer reads first. A missing row is a visible gap, which is better than a hidden one.

### Step 11. Map to the regulatory pathway

**Input:** the COU and the evidence.

**Output:** a pathway (biomarker qualification, device clearance, or neither) and the evidence gaps.

**Rationale:** the pathway depends on the claim and the jurisdiction. A biomarker used in drug development can go through the FDA biomarker qualification program; a biomarker used in clinical care is a device; a biomarker used for wellness is neither.

**Branch point:** if the biomarker is a trial endpoint, the qualification pathway is with the drug development tool program. If it is a clinical diagnostic, the device pathway applies.

**Edge case:** a biomarker can be qualified for one COU and not another. The qualification is specific, not general.

**Pathway comparison.**

| Pathway | Claim | Evidence | Timeline |
|---|---|---|---|
| Biomarker qualification | Use in drug development | COU, analytical, clinical | Multi-year |
| Device clearance | Clinical diagnosis or monitoring | Device + clinical | Months to years |
| Wellness | General wellness | Bench, human factors | No review |
| Research | Exploratory | Fit for purpose | Study-specific |

The pathway is a function of the claim, not of the technology. The same sensor can support a wellness claim and a device claim, with very different evidence.

### Step 12. Monitor and update

**Input:** the deployed biomarker and the post-market data.

**Output:** a monitoring plan with drift, calibration, and subgroup checks.

**Rationale:** a biomarker is a measurement, and measurements drift. A device update, a population change, or a firmware change can alter the biomarker without a change in the biology.

**Contents:** the metrics to monitor, the frequency, the thresholds, and the action. The plan should include a re-validation trigger.

**Branch point:** if the drift is large, the biomarker is re-validated before continued use.

**Edge case:** a biomarker that is monitored only on the mean can drift in the tails, where the clinical decisions are made. Monitor the distribution, not only the mean.

### Decision table for evidence gaps

| Gap | Consequence | Action |
|---|---|---|
| No verification | Sensor accuracy unknown | Bench study |
| No analytical validation | Measurement error unknown | Agreement study |
| No clinical validation | Association unknown | Cohort study |
| No utility | Decision impact unknown | Decision curve or trial |
| No usability | Real-world use unknown | Human factors study |
| No bias analysis | Subgroup performance unknown | Stratified analysis |
| No COU | Claim unbounded | Write COU |
| No version | Reproducibility unknown | Version and document |

## 6. Worked example

### 6.1 The case

A wearable step counter is proposed as a monitoring biomarker for functional capacity in adults with a chronic respiratory condition. The COU is: "monitoring biomarker of ambulation in adults aged 40 to 75 with a confirmed diagnosis, during a 12-week clinical trial, using mean daily step count over 7 days, compared with a research-grade accelerometer." The analytical validation compares the device with the research accelerometer in 120 participants over 7 days. The clinical validation is outside the example.

### 6.2 Agreement arithmetic

The device and the reference produce 120 paired 7-day mean daily step counts.

| Statistic | Value | Unit |
|---|---:|---|
| n | 120 | pairs |
| Mean reference | about 6,000 | steps/day |
| Bias | 86.5 | steps/day |
| SD of differences | 338.7 | steps/day |
| Lower limit of agreement | -577.4 | steps/day |
| Upper limit of agreement | 750.4 | steps/day |
| MAE | 278.2 | steps/day |
| ICC(2,1) | 0.978 | dimensionless |

The bias of 86.5 steps/day is 1.4% of the mean, which is small. The limits of agreement are wide: a device reading can be 577 steps below or 750 steps above the reference in 95% of cases. For an individual, a difference of 750 steps is about 12% of a 6,000-step day, which is larger than the minimal important difference for some functional outcomes.

The ICC of 0.978 is high because the between-subject variance is large: participants differ by thousands of steps, and the device tracks those differences. The ICC does not describe the individual agreement; the limits of agreement do.

The reference error is part of the observed difference. If the research accelerometer has its own error of 200 steps, the observed SD of 338.7 includes both devices. The true device error is smaller, but it cannot be separated without a third, better reference. This is the "two fallible devices" problem, and it is common in wearable validation.

### 6.3 Why correlation is not agreement

A correlation coefficient of 0.99 would also be high, and it would not tell you that the device is biased. The correlation measures whether the two measurements move together; the agreement measures whether they are interchangeable.

A device that reports exactly twice the reference has a correlation of 1.0 and a bias of +6,000 steps. The correlation is perfect and the agreement is useless.

The comparison table makes the distinction concrete.

| Property | Correlation | Agreement |
|---|---|---|
| Scale | Invariant to linear transform | Sensitive to bias and scale |
| Direction | Direction of association | Difference |
| Individual | Not assessed | Assessed |
| Outliers | Can dominate | Visible in the plot |
| Use | Association | Interchangeability |

A validation report should include both, but the agreement is the one that answers "can I use this device instead of the reference." The correlation answers a different question.

### 6.4 Sample size arithmetic

The precision of the limits of agreement depends on the sample size. The half-width of the 95% confidence interval for a limit of agreement is approximately 1.96 × SD / √n times a factor for the limit.

Using the simple approximation n = (1.96 × SD / half-width)²:

| Desired half-width | Required n |
|---:|---:|
| 100 steps | 44 |
| 200 steps | 11 |
| 300 steps | 5 |

A study with 120 participants has a half-width of about 61 steps for the bias and a wider interval for the limits. The sample is adequate for the bias and marginal for the tails. A study designed to detect a subgroup difference needs more.

The sample-size formula assumes that the differences are independent and normally distributed. Both assumptions can fail: repeated days within a participant are correlated, and the differences can be skewed. A cluster-adjusted sample size or a bootstrap confidence interval is more appropriate for repeated measures.

### 6.5 Decision curve arithmetic

Suppose the biomarker is used to decide whether to intensify a rehabilitation programme. The prevalence of the target state is 10%, the sensitivity is 80%, and the false-positive rate is 15%.

Net benefit = prevalence × sensitivity - (1 - prevalence) × false-positive rate × threshold/(1 - threshold).

| Threshold | Net benefit (model) | Net benefit (treat all) | Net benefit (treat none) |
|---:|---:|---:|---:|
| 0.05 | 0.0729 | 0.0526 | 0 |
| 0.10 | 0.0650 | 0.0000 | 0 |
| 0.20 | 0.0463 | -0.1250 | 0 |
| 0.30 | 0.0221 | -0.2857 | 0 |

The model has a positive net benefit at all four thresholds. Treat-all is better than the model only at the lowest threshold (0.05), where the net benefit is 0.0526 versus 0.0729 for the model; the model is still better. At higher thresholds, treat-all has a negative net benefit, and the model is clearly better.

The decision curve is more informative than the AUC because it incorporates the threshold and the prevalence. A model with a modest AUC can have a good net benefit at the clinically relevant threshold.

### 6.6 Diagnostic 1: heteroscedasticity

A Bland-Altman plot shows that the differences are larger at high step counts. The SD of the differences is 200 steps at 3,000 steps/day and 500 steps at 9,000 steps/day. The limits of agreement are not constant; they widen with the mean.

The diagnostic is to plot the absolute difference against the mean and to test the correlation. A significant positive correlation indicates heteroscedasticity. The consequence is that a single limits-of-agreement band is misleading: it is too wide at low counts and too narrow at high counts.

The fix is to report the agreement by tertile of the mean, or to model the variance. A log transformation can stabilize the variance if the error is multiplicative.

### 6.7 Diagnostic 2: the ICC that hides the tails

The ICC of 0.978 is high, but 8 participants have differences above 1,000 steps. The high ICC is driven by the between-subject variance; the tail errors are hidden.

The diagnostic is to plot the differences and to count the participants outside the limits of agreement. The count should be about 5% by construction; if it is concentrated in a subgroup, the agreement is not uniform.

The consequence is that the biomarker may be adequate for a group mean and inadequate for an individual decision. The COU determines which is required.

### 6.8 Varying one assumption

Suppose the device is worn on the wrist instead of the hip. Wrist-worn accelerometers can count arm movements that are not steps, especially during household activities. The bias may rise and the limits of agreement may widen.

Suppose the reference is a different research accelerometer with its own error. The agreement statistics include the reference error, and the device's true error is smaller than the observed difference. A study that compares two fallible devices estimates the difference between them, not the error of either.

Suppose the COU is changed from "group mean" to "individual monitoring." The limits of agreement become the relevant metric, and the device may fail the requirement even though the ICC is high.

The sensitivity of the conclusion to the COU is summarised below.

| COU | Relevant metric | Result | Conclusion |
|---|---|---|---|
| Group mean in a trial | Bias, SE of mean | 86.5 ± 31 steps | Adequate |
| Individual monitoring | Limits of agreement | -577 to 750 | Inadequate |
| Subgroup comparison | Stratified bias | depends on group | Unknown |
| Trend over time | Sensitivity to change | not measured | Unknown |

The same data support one conclusion and not another. The COU is what selects the metric, and the metric is what selects the conclusion.

### 6.9.1 Change-score agreement

A trial endpoint often uses the change from baseline, not the absolute value. The agreement of the change score is not the same as the agreement of the absolute value: two devices can agree on the absolute value and disagree on the change, or vice versa.

If the device has a constant bias, the change score is unbiased because the bias cancels. If the device has a proportional bias, the change score is biased by the proportional error times the change. A device with a 2% proportional bias and a 1,000-step change has a 20-step change-score bias, which may be acceptable.

The change-score agreement should be computed directly from paired change scores, not inferred from the absolute agreement. The two statistics answer different questions.

### 6.9.2 The minimal important difference

The minimal important difference (MID) is the smallest change that a patient or clinician would consider meaningful. It is not a statistical quantity, and it is often estimated from anchor-based or distribution-based methods.

The limits of agreement should be compared with the MID, not with zero. If the limits are ±750 steps and the MID is 500 steps, the measurement error is larger than the effect the trial is trying to detect. The device is not fit for the endpoint, even if the bias is small.

The comparison is a simple inequality: measurement error < MID. If the inequality fails, the endpoint needs a better measurement, a larger sample, or a different design. A larger sample reduces the standard error of the mean but does not reduce the individual measurement error; it can rescue a group-level endpoint and cannot rescue an individual-level decision.

The inequality is the simplest fitness check in the topic. It requires two numbers, the measurement error and the MID, and it answers the question that the correlation and the ICC cannot.

A trial team should compute the inequality before the trial, not after. If the measurement error is larger than the MID, the trial is underpowered by design, and no analysis can recover the effect.

### 6.9 Conclusion of the case

The device has a small bias (86.5 steps/day) and wide limits of agreement (-577 to 750 steps/day). It is adequate for a group-level comparison in a trial, where the errors average out, and inadequate for an individual-level decision without a narrower agreement band.

The ICC of 0.978 is not evidence of individual agreement; it is evidence of between-subject discrimination. The two diagnostics (heteroscedasticity, tail errors) show where the single summary statistic misleads.

A third diagnostic is the reference's own error. If the reference has a known error, the observed agreement is an upper bound on the device's true agreement. The case does not include a third reference, so the bound is not estimated.

A fourth diagnostic is the wear location. A wrist device and a hip reference measure different things during household activity, and the disagreement is partly a construct difference, not only sensor error.

A fifth diagnostic is the time window. A 7-day mean is more stable than a single day, and the agreement on a single day can be much worse than the agreement on the mean.

### 6.10 What the case does not show

The case is an analytical validation of a step count. It does not show:

- **Clinical validation.** There is no outcome, no cohort, and no association.
- **Utility.** The decision curve is illustrative, not estimated from the data.
- **Usability.** No users, no tasks, no adherence.
- **Bias.** No skin tone, age, or body composition stratification.
- **Free-living performance.** The reference and device are worn simultaneously, which is not the deployment condition.
- **Sensitivity to change.** A cross-sectional agreement study cannot show that the device detects a change over 12 weeks.

The omissions are the evidence gaps. The case supports a group-level analytical claim and nothing more. The COU in Section 6.1 is deliberately stronger than the evidence, which is the failure the topic warns about.

A reader who wants to extend the case should add a longitudinal component: the same participants measured at baseline and 12 weeks, with the device and reference, and a test of whether the change scores agree. The change-score agreement is a different statistic from the cross-sectional agreement, and it is the one a trial endpoint needs.

## 7. Code, query, or calculation

The script computes Bland-Altman statistics, ICC(2,1), a sample-size approximation, and a decision curve. It uses only the standard library.

```python
import math, random

def bland_altman(ref,test):
    n=len(ref); d=[test[i]-ref[i] for i in range(n)]
    bias=sum(d)/n; sd=math.sqrt(sum((x-bias)**2 for x in d)/(n-1))
    return bias,sd,bias-1.96*sd,bias+1.96*sd,sum(abs(x) for x in d)/n

def icc21(ref,test):
    n=len(ref); k=2
    grand=sum(ref+test)/(n*k)
    ss_total=sum((v-grand)**2 for v in ref+test)
    ss_rows=sum(k*(((ref[i]+test[i])/k-grand)**2) for i in range(n))
    ss_cols=n*((sum(ref)/n-grand)**2+(sum(test)/n-grand)**2)
    ss_err=ss_total-ss_rows-ss_cols
    ms_rows=ss_rows/(n-1); ms_cols=ss_cols/(k-1); ms_err=ss_err/((n-1)*(k-1))
    return (ms_rows-ms_err)/(ms_rows+(k-1)*ms_err+(k*(ms_cols-ms_err))/n)

random.seed(3)
n=120
ref=[random.gauss(6000,1500) for _ in range(n)]
test=[r*1.02+random.gauss(0,350) for r in ref]
b,sd,lo,hi,mae=bland_altman(ref,test)
print("n=%d bias=%.1f sd=%.1f loa=%.1f..%.1f mae=%.1f" % (n,b,sd,lo,hi,mae))
print("icc21=%.3f" % icc21(ref,test))

for hw in (100,200,300):
    print("halfwidth=%d n=%.0f" % (hw,(1.96*sd/hw)**2))

prev=0.10
for thr in (0.05,0.10,0.20,0.30):
    nb_model=prev*0.8 - (1-prev)*0.15*(thr/(1-thr))
    nb_all=prev - (1-prev)*(thr/(1-thr))
    print("thr=%.2f nb_model=%.4f nb_all=%.4f" % (thr,nb_model,nb_all))
```

Expected output:

```text
n=120 bias=86.5 sd=338.7 loa=-577.4..750.4 mae=278.2
icc21=0.978
halfwidth=100 n=44
halfwidth=200 n=11
halfwidth=300 n=5
thr=0.05 nb_model=0.0729 nb_all=0.0526
thr=0.10 nb_model=0.0650 nb_all=0.0000
thr=0.20 nb_model=0.0463 nb_all=-0.1250
thr=0.30 nb_model=0.0221 nb_all=-0.2857
```

Three implementation notes.

The ICC(2,1) is a two-way random-effects model with absolute agreement for a single measurement. It is appropriate when the two measurements are interchangeable and the raters are a random sample. A different ICC model answers a different question; the model must be stated.

The sample-size approximation is for the bias, not for the limits of agreement. The limits have wider confidence intervals because they depend on the variance estimate. A study designed for the limits needs more participants than a study designed for the bias.

The decision curve uses a simple net-benefit formula with a fixed sensitivity and false-positive rate. A real decision curve computes the net benefit for the model at each threshold from the predicted probabilities, not from a fixed operating point.

A second calculation checks the ICC components, and a third reports the standard errors. The ICC is the ratio of the between-subject variance to the total variance; the standard errors show how precisely the bias and the limits are estimated.

The split between the between-subject and within-subject sums of squares is the key to reading the ICC. A high ICC can mean good agreement or large between-subject variance; the components distinguish the two.

A final note: the second block assumes the variables from the first block. Run the two in the same session, in order, or import the first block as a module.

```python
def icc_components(ref, test):
    n=len(ref); k=2
    grand=sum(ref+test)/(n*k)
    ss_total=sum((v-grand)**2 for v in ref+test)
    ss_rows=sum(k*(((ref[i]+test[i])/k-grand)**2) for i in range(n))
    ss_cols=n*((sum(ref)/n-grand)**2+(sum(test)/n-grand)**2)
    ss_err=ss_total-ss_rows-ss_cols
    return ss_total, ss_rows, ss_cols, ss_err

ss_total, ss_rows, ss_cols, ss_err = icc_components(ref, test)
print("ss_total=%.1f ss_rows=%.1f ss_cols=%.1f ss_err=%.1f"
      % (ss_total, ss_rows, ss_cols, ss_err))
print("between_subject_share=%.4f" % (ss_rows/ss_total))

se_bias = sd/math.sqrt(n)
se_loa = sd*math.sqrt(3/n)
print("se_bias=%.1f se_loa=%.1f" % (se_bias, se_loa))
```

Expected output:

```text
ss_total=648221596.8 ss_rows=640946713.2 ss_cols=449018.5 ss_err=6825865.1
between_subject_share=0.9888
se_bias=30.9 se_loa=53.6
```

The output shows why the ICC is high: the between-subject sum of squares is 98.9% of the total. The ICC is a ratio of variances, and a large between-subject variance inflates it regardless of the individual agreement.

The standard error of a limit of agreement is approximately SD × √(3/n), which is larger than the standard error of the bias. This is why a study that is adequate for the bias can be inadequate for the limits.

## 8. Data, standards, and regulatory context

Digital biomarkers sit at the intersection of drug development, device regulation, and clinical care. The standards and guidance reflect the intersection.

| Source | What it constrains | Status | Practical consequence |
|---|---|---|---|
| BEST Resource (FDA-NIH) | Biomarker and endpoint definitions | Consensus glossary | Taxonomy and terminology |
| FDA Biomarker Qualification Program | Qualification for a COU | Regulatory process | LOI, QP, FQP |
| 21st Century Cures Act §507 | Qualification framework | Law in the US | DDT pathway |
| DiMe V3 / V3+ | Validation framework | Professional consensus | Verification, analytical, clinical, usability |
| FDA device software guidance | Software documentation | Binding through review | V&V and hazard analysis |
| FDA PCCP guidance | Model updates | Binding through review | Pre-authorized changes |
| ISO 14971 | Risk management | Consensus | Hazard analysis |
| IEC 62304 | Software lifecycle | Consensus | Development process |
| EU MDR | Clinical evaluation | Binding in the EU | Clinical evidence |
| EU AI Act | AI transparency | Binding in the EU | Data governance and oversight |

### 8.1 The BEST definitions

The BEST resource is the reference for the words. Its definitions are deliberately narrow.

- A **biomarker** is a defined characteristic measured as an indicator of normal or pathogenic processes or responses to an exposure or intervention. It is not a measure of how a patient feels, functions, or survives.
- An **endpoint** is a precisely defined variable intended to reflect an outcome of interest, statistically analyzed to address a research question. It specifies the assessment type, timing, tools, and combining rules.
- **Qualification** is a conclusion, based on a formal regulatory process, that within a stated COU a tool can be relied upon for a specific interpretation and application.
- A **context of use** fully describes how the tool is used and the regulatory purpose.

The distinction between a biomarker and a clinical outcome assessment is important. A step count is a biomarker of ambulation; a patient-reported measure of walking difficulty is a COA. They can be related, but they are different measurement types.

### 8.1.1 What counts as a digital biomarker in a regulatory submission

A submission must state which category the biomarker belongs to, because the category determines the evidence.

| Category | Regulatory question | Evidence emphasis |
|---|---|---|
| Diagnostic | Does the disease exist | Sensitivity, specificity, PPV |
| Monitoring | Is the disease changing | Sensitivity to change, MID |
| Prognostic | What will happen | Prospective association |
| Predictive | Who will respond | Interaction with treatment |
| Response | Did the intervention work | Change with treatment |
| Safety | Is there harm | Detection of harm |
| Susceptibility | Who is at risk | Prospective risk association |

A biomarker that is described as "monitoring" but validated as "diagnostic" is miscategorised. The category is not a label of convenience; it selects the study design.

### 8.1.2 Analytical validation versus verification

The two are often conflated. Verification is a hardware and firmware question: does the sensor capture the raw signal accurately? Analytical validation is an algorithm question: does the algorithm convert the signal into the intended measurement accurately?

A device can pass verification and fail analytical validation if the algorithm is wrong. A device can pass analytical validation and fail verification if the reference and the device share the same sensor error. The two tests are complementary, and neither substitutes for the other.

### 8.2 The qualification pathway

The FDA biomarker qualification program has three stages: a letter of intent, a qualification plan, and a full qualification package. Once qualified, the biomarker can be used in drug development programs within its COU.

The qualification is for the biomarker, not for a specific measurement method. A qualified biomarker can be measured with a validated method; the method still needs analytical validation.

The distinction is important for digital biomarkers because the measurement method changes frequently. A qualified biomarker can survive a device change if the new method is analytically validated and shown to measure the same construct. A qualification does not certify a particular sensor.

The pathway is long and resource-intensive. A company that needs a biomarker for an internal decision may not need formal qualification; a company that wants a claim across programs may. The choice depends on the intended use and the acceptance by regulators.

### 8.2.1 Qualification versus validation

Qualification is a regulatory conclusion about a biomarker for a COU. Validation is the evidence that supports the conclusion. A biomarker can be validated without being qualified, and a qualification without validation is not possible.

The distinction matters for planning. A company that needs a biomarker for an internal decision can validate it and use it without qualification. A company that needs a cross-program claim may pursue qualification, which takes longer and involves the regulator earlier.

### 8.3 V3 and V3+ in practice

The V3 framework is widely used, including by the European Medicines Agency. Its value is that it separates the evidence into components that can be validated independently. A team can start with verification and analytical validation and add clinical validation when the measurement is stable.

V3+ adds usability, which is often the missing component. A device that is accurate in a lab and unusable at home fails the fit-for-purpose test even if the analytical validation is strong.

The framework is a professional consensus, not a regulation. A submission can follow it without citing it, and a submission that cites it must still provide the evidence.

A practical use of the framework is to assign an owner to each component. Verification belongs to hardware, analytical validation to data science, clinical validation to clinical affairs, and usability to human factors. A biomarker program without an owner for each component tends to complete the easy components and leave the hard ones.

The framework also helps with sequencing. A clinical validation started before the analytical validation is a common and expensive mistake, because the measurement may change after the clinical study and invalidate the result.

**The V3+ evidence table.**

| Component | Question | Typical study | Output |
|---|---|---|---|
| Verification | Does the sensor capture the signal | Bench, phantom | Bias, precision |
| Analytical | Does the algorithm measure correctly | Simultaneous reference | LoA, ICC, MAE |
| Clinical | Does it reflect the outcome | Cohort | AUC, association |
| Usability | Can users use it | Human factors | Success, errors |

The table is the shortest summary of the framework. A team can use it as a project plan: each row is a workstream with a deliverable, and the biomarker is not ready until all four rows have evidence.

The order matters. Verification and analytical validation should be completed before clinical validation, because a clinical association built on an invalid measurement is uninterpretable. A team that starts with the clinical study and works backward often discovers that the measurement was never validated.

## 9. Industry practice and case evidence

### 9.1 Digital endpoints in trials

The DiMe Library of Digital Endpoints catalogues digital measures used in clinical trials, including gait, sleep, and activity. The catalogue shows that many measures are used in trials without formal qualification, and that the evidence varies widely.

The documented pattern is that a measure can be accepted in an exploratory endpoint with less evidence than a primary endpoint. The inference is that the evidence requirement scales with the claim, and a team should decide the claim before designing the study.

### 9.2 Wearable step counts

Step counts are among the most studied digital measures. The evidence shows that consumer devices can be accurate for group means and less accurate for individuals, and that accuracy varies by device, wear location, and gait.

The documented lesson is that the same measure can be fit for a group-level trial endpoint and unfit for an individual clinical decision. The COU determines which.

### 9.3 Sleep staging

Consumer sleep staging has been compared with polysomnography in several studies. The agreement varies by device and stage, with better agreement for total sleep time and worse for REM and N1.

The documented lesson is that the stage-specific agreement is the relevant metric, not the overall accuracy. A device that is 85% accurate but 40% sensitive for REM is not a five-stage device.

### 9.4 Optical biomarkers and skin tone

Pulse oximetry and PPG-derived measures can be biased by skin tone. The FDA's 2025 draft guidance on pulse oximeters is the clearest regulatory statement that a pooled accuracy can hide a subgroup bias.

The documented lesson is that a digital biomarker validation must include a bias analysis, and the analysis must use an appropriate measure of skin tone, not a guess.

### 9.5 Qualification examples

The FDA's biomarker qualification program has qualified biomarkers in several areas, including kidney injury and drug-induced liver injury. The qualification packages are public and are useful models for the evidence structure.

A second documented example is the use of digital measures in regulatory submissions for neurological diseases. The measures are often accepted as exploratory endpoints while the qualification of a primary endpoint is pending.

The inference for digital biomarkers is that the qualification model is transferable: a COU, a definition, an analytical method, a clinical association, and a plan for use. The digital sensor adds the verification and usability components.

### 9.6 What the evidence base shows across domains

A cross-domain reading of digital biomarker evidence shows four patterns.

- **Measurement validation is often stronger than clinical validation.** Devices are compared with references more often than biomarkers are compared with outcomes.
- **Group-level claims are stronger than individual-level claims.** A device that is adequate for a trial mean is often inadequate for an individual decision.
- **Usability and adherence are under-reported.** The validation sample is often the sample that used the device correctly.
- **Subgroup analyses are rare.** A pooled metric is the norm, and the subgroups are small when they exist.

These patterns are inference from the published literature and the regulatory record, not a systematic review. They are the gaps that a new validation should address.

### 9.7 A digital biomarker evidence map

| Domain | Measurement maturity | Clinical maturity | Main gap |
|---|---|---|---|
| Step count | High | Moderate | Individual agreement |
| Sleep staging | Moderate | Moderate | REM and N1 |
| PPG heart rate | High | High | Motion |
| SpO2 | High | High | Skin tone |
| Gait speed | Moderate | Moderate | Free-living |
| Voice | Low | Low | Construct validity |
| EDA stress | Low | Low | No gold standard |

The map is a rough guide. The point is that the maturity differs by domain, and a claim should be calibrated to the maturity. A voice biomarker presented with the confidence of a step count is overclaiming.

The map also suggests where the next evidence is needed. The low-maturity domains need construct validity and reference standards before they need larger models.

A final observation is that the evidence base is skewed toward measures that are easy to reference. Step counts and heart rate have references; stress and fatigue do not. The skew is a property of the field, not of the importance of the construct. A deep model on a construct without a reference is a model without a target.

### 9.8 What a reviewer looks for

A reviewer of a digital biomarker submission typically looks for:

1. A COU that is specific enough to bound the claim.
2. A measurement specification that can be reproduced.
3. A reference that is appropriate for the construct.
4. An agreement analysis that reports individual error, not only correlation.
5. A clinical validation in the target population.
6. A utility argument at a clinically relevant threshold.
7. A usability and adherence assessment.
8. A bias analysis with an appropriate subgroup measure.
9. A version and change-control plan.

The list is a checklist for the evidence table in Section 5, Step 10. A submission that can fill every row is ready for review; one that cannot is ready for a gap analysis.

A reviewer also looks for the negative evidence: what the biomarker does not do, which population was not studied, and which condition was not tested. A submission that lists only positive findings is treated as incomplete, because the limits are part of the claim.

The most common gap in digital biomarker submissions is the individual-level agreement. A device that is validated for a group mean is often presented as if it were valid for an individual. The limits of agreement are the evidence that distinguishes the two, and they are often missing.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Feature called a biomarker | No clinical association | Terminology inflation | Overclaim | Define the evidence |
| COU omitted | Claim applies to everyone | No population boundary | Invalid generalization | Write COU |
| Analytical validation skipped | Association without measurement validation | Sensor error confounds | Uninterpretable | Agreement study |
| Correlation used for agreement | r near 1, wide LoA | Association not agreement | False interchangeability | Bland-Altman |
| ICC overinterpreted | High ICC, wide LoA | Between-subject variance | False individual validity | Report LoA |
| Heteroscedasticity ignored | Variance grows with mean | Multiplicative error | Misleading LoA | Stratify or transform |
| Reference error ignored | Two fallible devices | Combined error | Underestimated error | Use a better reference |
| Prevalence ignored | PPV from case-control | Different prior | Wrong PPV | Prior correction |
| Utility assumed | Association only | No decision impact | No clinical value | Decision curve |
| Usability skipped | Lab success | Real-world failure | Missing data | Human factors |
| Bias not assessed | Pooled metric | Subgroup difference | Inequity | Stratified analysis |
| Version drift | Values change | Firmware or model update | Non-comparable | Version and revalidate |
| Surrogate overclaim | Surrogate not validated | Assumption | Regulatory rejection | Validate the surrogate |
| Endpoint undefined | No timing or rule | Ambiguous analysis | Irreproducible | Endpoint specification |
| Change-score ignored | Absolute agreement good | Different statistic | Wrong endpoint | Change-score study |
| MID not compared | LoA wider than MID | Measurement error | Underpowered trial | Fitness inequality |
| Ceiling effect | Scores cluster at max | Measure saturates | No sensitivity | Alternative measure |
| Practice effect | Improvement without treatment | Repetition | False efficacy | Control period |

Three red flags deserve emphasis.

**A COU that describes a technology, not a decision.** "Wearable gait monitoring" is a technology. "Monitor gait speed to detect progression in adults with a confirmed diagnosis over 12 weeks" is a COU. The first cannot be validated; the second can.

**An agreement analysis that reports only correlation or only ICC.** Both can be high while the individual error is unacceptable. The limits of agreement are the missing number.

**A clinical validation without a utility analysis.** An association is not a reason to use a biomarker. The decision curve or the impact study is the reason.

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test |
|---|---|---|---|
| The construct is measurable | Biomarker definition | No gold standard | Construct-validity study |
| The sensor measures the proxy | Verification | Hardware limitation | Bench test |
| The algorithm is valid | Analytical validation | Population or device change | Agreement study |
| The reference is adequate | Agreement analysis | Reference error | Better reference |
| The population matches the COU | Generalization | Different age or disease | Stratified study |
| The setting matches the COU | Ecological validity | Lab versus home | Free-living study |
| The agreement is adequate for the decision | Utility | Individual versus group | LoA versus MID |
| The prevalence is representative | PPV | Case-control sampling | Prior correction |
| The users can use the device | Usability | Complex instructions | Human factors |
| The subgroups are represented | Equity | Small subgroups | Stratified analysis |
| The version is stable | Reproducibility | Firmware update | Version control |
| The endpoint is pre-specified | Trial validity | Post-hoc definition | Protocol |
| The change score agrees | Endpoint validity | Absolute agreement only | Change-score study |
| The measure has no ceiling | Responsiveness | Scores at maximum | Distribution check |
| The practice effect is controlled | Internal validity | Repetition | Control period |
| The wear location is consistent | Measurement comparability | Wrist versus hip | Protocol |
| The construct is not a technology | Validation | COU describes a device | COU review |
| The individual error is reported | Fitness | ICC only | LoA |
| The utility is tested | Claim | Association only | Decision curve |
| The measurement error is below the MID | Trial fitness | Error above MID | Fitness check |
| The version is recorded | Reproducibility | Firmware update | Version log |

## 12. Exercises

### Exercise 1. Taxonomy (recall, 20 min)

**Deliverable:** classify five examples as measurement, feature, biomarker, or endpoint, with a justification.

**Acceptance criteria:** each classification uses the BEST definitions; the missing evidence is named.

**Time:** 20 minutes.

### Exercise 2. Context of use (application, 30 min)

**Deliverable:** write a COU for a gait-speed biomarker in a Parkinson's trial, including population, setting, decision, timing, and reference.

**Acceptance criteria:** all five elements; no marketing language; the claim is bounded.

**Time:** 30 minutes.

### Exercise 3. Agreement analysis (application, 40 min)

**Deliverable:** from 20 paired measurements, compute bias, SD, limits of agreement, MAE, and ICC.

**Acceptance criteria:** all statistics; the interpretation for an individual versus a group decision.

**Time:** 40 minutes.

### Exercise 4. Correlation trap (critique, 25 min)

**Deliverable:** construct an example where r = 1.0 and the agreement is unacceptable, and explain the difference.

**Acceptance criteria:** the example is arithmetically correct; the conclusion is stated.

**Time:** 25 minutes.

### Exercise 5. Decision curve (application, 35 min)

**Deliverable:** compute net benefit at three thresholds for a biomarker with known sensitivity and false-positive rate, and compare with treat-all.

**Acceptance criteria:** formula shown; prevalence included; the clinical threshold identified.

**Time:** 35 minutes.

### Exercise 6. Bias analysis (design, 40 min)

**Deliverable:** design a bias analysis for a PPG-derived biomarker, including the subgroup variables and the analysis plan.

**Acceptance criteria:** at least three subgroups; appropriate skin-tone measure; power considered.

**Time:** 40 minutes.

### Exercise 7. Evidence table (design, 45 min)

**Deliverable:** an evidence table for a digital biomarker with verification, analytical, clinical, utility, usability, and bias rows.

**Acceptance criteria:** each row has a question, design, result, and limitation; gaps are visible.

**Time:** 45 minutes.

### Exercise 8. Regulatory mapping (application, 30 min)

**Deliverable:** map three biomarker claims to a pathway (qualification, device, neither) and justify each.

**Acceptance criteria:** the COU drives the mapping; the evidence gaps are named.

**Time:** 30 minutes.

### Exercise 9. Fitness inequality (application, 20 min)

**Deliverable:** given a measurement error of ±600 steps and a MID of 400 steps, decide whether the device is fit for an individual endpoint, and state the options.

**Acceptance criteria:** the inequality is applied; the options (better device, group endpoint, larger sample) are evaluated; the conclusion is honest.

**Time:** 20 minutes.

### Submission standard

A COU must name the population, setting, decision, timing, comparator, and user. An agreement analysis must include the Bland-Altman statistics, not only the correlation. An evidence table must have all six components. A regulatory mapping must be justified by the COU.

An answer that calls a correlation "validation" is returned. An answer that reports an ICC without the limits of agreement is returned. The standard is the same as the one used in the topic.

## 13. Self-check answers

**1. What is a biomarker?** A defined characteristic measured as an indicator of normal or pathogenic processes or responses to an exposure or intervention.

**2. What is an endpoint?** A precisely defined variable intended to reflect an outcome of interest, statistically analyzed to address a research question.

**3. What is a context of use?** A statement describing the population, setting, decision, timing, comparator, and user for which a biomarker is validated.

**4. What are the V3 components?** Verification, analytical validation, and clinical validation. V3+ adds usability.

**5. What is the difference between correlation and agreement?** Correlation measures association; agreement measures whether two measurements are interchangeable.

**6. Compute the limits of agreement for bias 86.5 and SD 338.7.** 86.5 ± 1.96 × 338.7 = 86.5 ± 663.9, or -577.4 to 750.4.

**7. Why is an ICC of 0.978 not evidence of individual agreement?** The ICC is inflated by between-subject variance; the limits of agreement show the individual error.

**8. What is heteroscedasticity in a Bland-Altman plot?** The variance of the differences changes with the mean, so a single limits-of-agreement band is misleading.

**9. Compute net benefit at threshold 0.10 for prevalence 0.10, sensitivity 0.80, FPR 0.15.** 0.10 × 0.80 - 0.90 × 0.15 × (0.10/0.90) = 0.08 - 0.015 = 0.065.

**10. What is the difference between a biomarker and a COA?** A biomarker indicates a biological process; a COA measures how a patient feels, functions, or survives.

**11. Why is a case-control sample a problem for PPV?** The prevalence in a case-control sample is artificial, and PPV depends on prevalence.

**12. What is the difference between analytical and clinical validation?** Analytical validation asks whether the measurement is accurate; clinical validation asks whether it reflects or predicts the clinical state.

**13. What is the minimum evidence for a monitoring biomarker?** A validated measurement, a demonstrated change with the disease or treatment, and a COU.

**14. Why is usability a validation component?** A device that is accurate but unusable produces missing or incorrect data in the real world.

**15. What is the difference between a biomarker and a surrogate endpoint?** A surrogate endpoint is a biomarker used in place of a clinical outcome, with additional validation for that use.

**16. Compute the standard error of the bias for SD 338.7 and n 120.** 338.7/√120 = 30.9 steps/day.

**17. Why is the change-score agreement a separate study?** A constant bias cancels in the change, but a proportional bias does not. The absolute agreement does not determine the change-score agreement.

**18. What is the fitness inequality?** Measurement error must be smaller than the minimal important difference for the device to be fit for an individual-level endpoint.

**19. What is the difference between verification and analytical validation?** Verification asks whether the sensor captures the signal; analytical validation asks whether the algorithm measures the construct.

**20. Why is a high between-subject variance a problem for ICC interpretation?** It inflates the ICC even when the individual agreement is poor, because the ICC is a ratio of between-subject to total variance.

**21. What is the difference between repeatability and reproducibility?** Repeatability is agreement under the same conditions; reproducibility is agreement under different conditions, such as a different device or operator.

**22. Why is a monitoring biomarker's sensitivity to change important?** A monitoring biomarker must detect a change over time; a cross-sectional association does not establish that it can.

**23. What is a known-groups validation?** A test of whether the biomarker distinguishes groups that are known to differ, such as healthy versus diagnosed.

**Rubric for Exercise 7 (evidence table).** A strong table has all six components, each with a question, design, result, and limitation. A weak table lists studies without linking them to the components.

**Rubric for Exercise 6 (bias analysis).** A strong design names the subgroups, the measure, the analysis, and the power. A weak design says "we will check for bias."

## 14. Interview questions

**1. What is a digital biomarker?**
A strong answer gives the BEST definition and distinguishes it from a feature and an endpoint. A weak answer says "a health metric from a device."

**2. What is a context of use?**
A strong answer lists population, setting, decision, timing, comparator, and user. A weak answer says "how it is used."

**3. How do you validate a digital biomarker?**
A strong answer follows V3+: verification, analytical, clinical, usability, plus bias. A weak answer says "compare with a gold standard."

**4. Why is correlation not agreement?**
A strong answer gives the doubling example and explains Bland-Altman. A weak answer says they are similar.

**5. What is the difference between a biomarker and an endpoint?**
A strong answer explains the trial variable requirements and the qualification. A weak answer treats them as synonyms.

**6. How do you assess clinical utility?**
A strong answer uses a decision curve or an impact study at a clinically relevant threshold. A weak answer says "association with outcome."

**7. What is the ICC and when is it misleading?**
A strong answer defines it and explains the between-subject variance inflation. A weak answer says it measures agreement.

**8. How do you handle a subgroup bias?**
A strong answer names the mechanism, the analysis, and the claim adjustment. A weak answer says "stratify."

**9. What is the FDA qualification pathway?**
A strong answer describes LOI, QP, FQP, and the COU. A weak answer says "FDA approval."

**10. Why is usability part of validation?**
A strong answer explains real-world data quality and adherence. A weak answer says it is a product concern.

**11. What would you do if the limits of agreement are wider than the MID?**
A strong answer restricts the claim to group-level use or improves the measurement. A weak answer ignores the LoA.

**12. How do you prevent version drift in a biomarker?**
A strong answer versions the device, firmware, and algorithm, and revalidates after changes. A weak answer relies on the vendor.

**13. What is the difference between a monitoring and a diagnostic biomarker?**
A strong answer explains that monitoring requires sensitivity to change and a MID, while diagnostic requires discrimination at a point in time. A weak answer treats them as the same.

**14. Why is a group-level agreement insufficient for an individual decision?**
A strong answer distinguishes the standard error of the mean from the individual limits of agreement. A weak answer says the ICC is high.

**15. What would you ask a vendor who claims a "digital biomarker of fatigue"?**
A strong answer asks for the COU, the construct, the reference, the analytical validation, and the clinical evidence. A weak answer asks for the correlation.

**16. What is the difference between a biomarker and a measurement?**
A strong answer says a measurement is a quantity with a unit and a method; a biomarker adds a validated relationship to a biological process. A weak answer treats them as synonyms.

**17. Why is the analytical range important?**
A strong answer explains that agreement outside the validated range is extrapolation, and a device can be accurate in one range and not another. A weak answer ignores the range.

## 15. Key terms

- **Digital biomarker** — digital-data-derived indicator.
- **BEST** — FDA-NIH biomarker glossary.
- **Context of use** — who, what, where, why.
- **Qualification** — formal regulatory conclusion.
- **Verification** — sensor accuracy.
- **Analytical validation** — algorithm accuracy.
- **Clinical validation** — association with outcome.
- **Clinical utility** — decision impact.
- **Usability validation** — real-world use.
- **Reference standard** — best available truth.
- **Bland-Altman** — agreement plot.
- **Bias** — mean difference.
- **Limits of agreement** — bias ± 1.96 SD.
- **ICC** — intraclass correlation.
- **MAE** — mean absolute error.
- **MARD** — mean absolute relative difference.
- **Net benefit** — utility at a threshold.
- **Decision curve** — net benefit across thresholds.
- **Surrogate endpoint** — endpoint used in place of outcome.
- **COA** — clinical outcome assessment.
- **PRO** — patient-reported outcome.
- **Fit for purpose** — adequate for the COU.
- **Sensitivity to change** — ability to detect change.
- **Responsiveness** — change with intervention.
- **MID** — minimal important difference.
- **Ecological validity** — real-world performance.
- **Heteroscedasticity** — variance changing with mean.
- **Ceiling effect** — measure saturates.
- **Floor effect** — measure bottoms out.
- **Practice effect** — improvement with repetition.
- **Digital divide** — unequal access to devices.
- **Data provenance** — origin and history of data.
- **Change score** — difference from baseline.
- **Anchor-based MID** — MID from a clinical anchor.
- **Distribution-based MID** — MID from the distribution.
- **Wear location** — body site of the sensor.
- **Adherence** — extent of device use.
- **Anchor-based method** — MID from a clinical anchor.
- **Distribution-based method** — MID from the distribution.
- **Convergent validity** — correlation with a related measure.
- **Discriminant validity** — lack of correlation with an unrelated measure.
- **Known-groups validity** — difference between known groups.
- **Test-retest reliability** — stability over time.
- **Inter-rater reliability** — agreement between raters.
- **Measurement model** — map from signal to construct.
- **Proxy ladder** — chain from sensor to construct.
- **Fitness inequality** — error compared with MID.
- **Analytical range** — validated range of the measurement.
- **Limit of detection** — smallest detectable value.
- **Reference error** — error in the reference standard.
- **Two fallible devices** — both measurements have error.
- **Prior correction** — adjustment to a new prevalence.
- **Decision threshold** — cut-point for action.
- **Impact study** — randomized use of the biomarker.
- **Stepped wedge** — rollout design.
- **Before-after** — pre-post comparison.
- **Secular trend** — change unrelated to the intervention.
- **Construct validity** — evidence for the intended construct.
- **Convergent validity** — correlation with a related measure.
- **Discriminant validity** — lack of correlation with an unrelated measure.
- **Known-groups validity** — difference between known groups.
- **Test-retest** — stability over time.
- **Inter-rater** — agreement between raters.
- **Anchor** — external criterion for MID.
- **Distribution** — statistical spread of scores.
- **Fitness** — adequacy for the COU.
- **Claim** — statement supported by evidence.
- **Evidence gap** — missing validation component.
- **Gap analysis** — structured review of missing evidence.
- **Re-validation** — repeat validation after a change.
- **Post-market** — after launch.
- **Surveillance** — ongoing monitoring.

## 16. Sources

All URLs accessed 25 October 2026.

1. FDA-NIH BEST Resource. Glossary. https://www.ncbi.nlm.nih.gov/books/NBK338448/
2. FDA. BEST Resource Taxonomy. https://www.fda.gov/drugs/development-resources/best-resource-taxonomy
3. FDA. About Biomarkers and Qualification. https://www.fda.gov/drugs/biomarker-qualification-program/about-biomarkers-and-qualification
4. FDA. Context of Use. https://www.fda.gov/drugs/biomarker-qualification-program/context-use
5. Digital Medicine Society. V3 framework. https://dimesociety.org/access-resources/v3/
6. Digital Medicine Society. V3+ and usability. https://dimesociety.org/v3/
7. Digital Medicine Society. sDHT Adoption Navigator. https://navigator.dimesociety.org/your-validation-strategy/
8. US Food and Drug Administration. Pulse Oximeters draft guidance. 2025. https://www.fda.gov/media/184896/download
9. Chinoy ED, et al. Performance of seven consumer sleep-tracking devices compared with polysomnography. *Sleep*. 2021;44(5):zsaa291. https://doi.org/10.1093/sleep/zsaa291
10. US Food and Drug Administration. Content of Premarket Submissions for Device Software Functions. 2023. https://www.fda.gov/regulatory-information/search-fda-guidance-documents/content-premarket-submissions-device-software-functions
11. EU Medical Device Regulation 2017/745. https://eur-lex.europa.eu/eli/reg/2017/745/oj
12. FDA. Biomarker Qualification Program. https://www.fda.gov/drugs/biomarker-qualification-program
13. FDA. DDT Glossary. https://www.fda.gov/drugs/drug-development-tool-ddt-qualification-programs/ddt-glossary
14. Digital Medicine Society. Library of Digital Endpoints. https://dimesociety.org/datacc/
15. Digital Medicine Society. Incorporating digitally derived endpoints within clinical development programs. https://dimesociety.org/journal/incorporating-digitally-derived-endpoints-within-clinical-development-programs-by-leveraging-prior-work/
16. US Food and Drug Administration. Digital Health Center of Excellence. https://www.fda.gov/medical-devices/digital-health-center-excellence

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`07_deep-learning-for-signals.md`](07_deep-learning-for-signals.md)
- Next topic: [`09_validation.md`](09_validation.md)
- Related topic: [`06_feature-engineering.md`](06_feature-engineering.md)
- Module survey file: [`../../11_physiological_signals_wearables.md`](../../11_physiological_signals_wearables.md)
- Lab: [`../../labs/`](../../labs/)
- Related topic: [`06_feature-engineering.md`](06_feature-engineering.md)
- Related topic: [`10_deployment-on-device.md`](10_deployment-on-device.md)
- Framework: [`https://dimesociety.org/v3/`](https://dimesociety.org/v3/)
- Glossary: [`https://www.ncbi.nlm.nih.gov/books/NBK338448/`](https://www.ncbi.nlm.nih.gov/books/NBK338448/)
- Qualification: [`https://www.fda.gov/drugs/biomarker-qualification-program`](https://www.fda.gov/drugs/biomarker-qualification-program)
