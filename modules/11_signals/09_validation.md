# Validation: Reference Standards, Co-Primary Endpoints, Confounding, and Reporting

> Module 11 — Physiological Signals and Wearables
> Topic 09 of 10
> Estimated study time: 4 hours
> Prerequisites: Topic 06 (feature engineering), Topic 07 (deep learning), Topic 08 (digital biomarkers)
> Version 2026.3

## 1. Why this topic exists

Validation is the process of producing evidence that a measurement, a biomarker, or a model supports a stated claim in a stated population. It is not a single study. It is a chain of studies, each answering a different question, and the claim is only as strong as the weakest link. A model can be accurate and useless, a biomarker can be associated and unactionable, and a study can be internally valid and irrelevant to the deployment population.

Five failures recur.

**The prevalence mismatch.** A model is validated in a case-control sample with 50% prevalence and reports a positive predictive value of 0.90. In deployment, the prevalence is 1%, and the PPV falls to 0.24 with the same sensitivity and specificity. The model has not changed; the prior has. A validation that does not report the prevalence, or that does not adjust for it, is reporting a number that will not hold.

**The reference that is not independent.** A wearable is validated against another wearable from the same manufacturer, or against a clinical measurement that uses the same sensor. The errors are correlated, the agreement looks better than it is, and the validation is circular. The reference must be independent of the index test.

**The spectrum that is too narrow.** A model is validated in healthy adults aged 20 to 40 and deployed in adults aged 60 to 85 with comorbidities. The signal, the artifact, and the disease prevalence all differ. The performance falls, and the fall is attributed to the model rather than to the population.

**The endpoint that is not co-primary.** A study reports a good sensitivity and a poor specificity, or a good AUROC and a poor calibration, and claims success on the better metric. A claim that requires both must pre-specify both, with thresholds, and evaluate them jointly. A single metric can always be selected post hoc.

**The missing data that is informative.** Participants who find the device difficult to wear stop using it. The validation sample is the participants who adhered, and the performance in the full population is unknown. The missingness is not random, and complete-case analysis is biased.

This topic covers the design of validation studies: the question and the role of the test, the reference standard, the population and spectrum, sampling, blinding, sample size, analysis, indeterminate results, missing data, subgroup performance, and reporting. It builds on the measurement and biomarker topics and sets up Topic 10 for deployment.

### 1.1 Three composite failures

**The PPV that did not travel.** A company validates an AF detector in a case-control sample of 500 participants, half with AF and half without. The sensitivity is 0.95, the specificity is 0.97, and the PPV is 0.97. The product launches in a primary-care population with a 1% prevalence, and the PPV is 0.24. Most alerts are false. The cost is alert fatigue and a product that is withdrawn from the market.

**The reference that shared the sensor.** A wearable is validated against a clinical-grade device from the same vendor. The two devices share the optical front end and the algorithm. The agreement is excellent, and the validation is circular. The cost is a product that appears accurate and fails against an independent reference.

**The 12% who could not be classified.** A single-lead ECG algorithm reports a sensitivity of 0.98 on classifiable recordings. Twelve percent of recordings are unclassifiable and are excluded. In deployment, the unclassifiable rate is 20%, and the effective sensitivity is 0.78. The cost is a claim that overstates the real-world performance.

A fourth failure is a validation that is never used. The study is published, the product launches, and the deployment population is different from the study population. The validation was accurate and irrelevant. The cost is a product that is mispriced for its evidence.

The four failures share a structure: the validation design, not the algorithm, determines whether the number is a claim. The design choices are the prevalence, the reference independence, and the handling of the inconclusive cases.

### 1.2 Why validation is a design discipline

A validation study is a design problem before it is a statistical one. The choices are:

- who to enrol,
- what to measure as the reference,
- when to measure it,
- how to sample,
- how to blind,
- how to handle the cases that do not fit.

Each choice has a bias associated with it, and the biases are predictable. A team that knows the biases can design them out; a team that does not will discover them in review.

## 2. Learning outcomes

By the end of this topic you can:

1. **Define** the target condition, the index test, the reference standard, and the intended role of the test.
2. **Choose** a reference standard that is independent, appropriate, and feasible, and state its limitations.
3. **Design** a sampling strategy that avoids spectrum bias and reflects the deployment population.
4. **Compute** sensitivity, specificity, PPV, NPV, and likelihood ratios from a 2×2 table, with confidence intervals.
5. **Adjust** PPV for a deployment prevalence different from the study prevalence.
6. **Pre-specify** co-primary endpoints and evaluate them jointly.
7. **Handle** indeterminate results and missing data without inflating performance.
8. **Report** a validation study against STARD 2015 and name the items that are missing.

Each outcome maps to an exercise in Section 12 and a self-check item in Section 13.

The outcomes move from the question to the design to the analysis to the report. The middle outcomes are the ones that prevent the common biases; the last outcome is the one that makes the study evaluable.

## 3. The core mental model

A diagnostic accuracy study has four objects: the target condition, the index test, the reference standard, and the population. The design is the relationship among them.

```text
 population (spectrum, prevalence)
        |
   sampling (consecutive, random, convenience)
        |
   +----+----+
   |         |
 index     reference
 test      standard
   |         |
   +----+----+
        |
  2x2 table: TP, FP, FN, TN
        |
  sensitivity, specificity, PPV, NPV, LR
        |
  decision at a threshold
        |
  utility in the intended population
```

Redraw it with the arrows labelled "independent" and "blinded." The two labels are the design's defences against bias.

### 3.1 The target condition and the role

The target condition is the disease or state the test is intended to identify. It must be defined before the study: "atrial fibrillation on a simultaneous 12-lead ECG" is a target condition; "an irregular rhythm" is not.

The role of the test is its place in the clinical pathway: screening, diagnosis, monitoring, or triage. The role determines the acceptable sensitivity and specificity. A screening test needs high sensitivity and an acceptable false-positive rate; a confirmatory test needs high specificity and a high PPV.

### 3.2 The reference standard

The reference standard is the best available method for determining whether the target condition is present. It has its own error, and the error matters.

- For AF, the reference is a simultaneous 12-lead ECG over-read by blinded cardiologists.
- For sleep apnea, the reference is attended polysomnography.
- For SpO2, the reference is arterial blood gas co-oximetry.
- For glucose, the reference is a laboratory or a calibrated meter.

A reference that is not independent of the index test produces a biased estimate. Two wearables that share a sensor or an algorithm are not independent. A clinical measurement that uses the same optical principle is not independent for an optical test.

### 3.3 The population and the spectrum

The population determines the spectrum of disease and the prevalence. A study in a tertiary referral centre has a different spectrum from a study in primary care. A study that excludes comorbidity has a different spectrum from the deployment population.

Spectrum bias is the change in sensitivity or specificity when the spectrum changes. A test can be sensitive in severe disease and insensitive in mild disease, or specific in healthy controls and less specific in patients with confounding conditions.

### 3.4 The 2×2 table and the metrics

|  | Reference positive | Reference negative |
|---|---|---|
| Index positive | TP | FP |
| Index negative | FN | TN |

- Sensitivity = TP/(TP+FN).
- Specificity = TN/(TN+FP).
- PPV = TP/(TP+FP).
- NPV = TN/(TN+FN).
- Positive likelihood ratio = sensitivity/(1-specificity).
- Negative likelihood ratio = (1-sensitivity)/specificity.

Sensitivity and specificity are properties of the test and the spectrum. PPV and NPV are properties of the test and the prevalence. A study that reports only PPV without the prevalence is not reproducible.

The likelihood ratios are the bridge between the two. They are prevalence-independent and can be combined with any prior to produce a posterior. A report that gives the likelihood ratios lets the reader compute the PPV for their own population.

### 3.5 Co-primary endpoints

A co-primary endpoint is a set of outcomes that must all be met for the claim to succeed. For a screening test, the co-primary endpoints might be sensitivity above 0.90 and specificity above 0.95, evaluated jointly with a prespecified alpha allocation.

The alternative is a single composite or a hierarchical endpoint. The choice must be pre-specified, because a study that reports several metrics and selects the best one has inflated the type I error.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Validation | Evidence that a claim is supported | Calling any study validation | evidence |
| Target condition | The disease or state to detect | Defining it loosely | condition |
| Index test | The test being evaluated | Confusing with reference | test |
| Reference standard | The best available truth | Assuming error-free | test |
| Intended role | Screening, diagnosis, monitoring | Omitting from the claim | role |
| Spectrum | Range of disease severity and comorbidity | Ignoring | population |
| Spectrum bias | Change in accuracy with spectrum | Attributing to model | bias |
| Prevalence | Proportion with the condition | Ignoring in PPV | % |
| Sensitivity | TP/(TP+FN) | Confusing with PPV | % |
| Specificity | TN/(TN+FP) | Confusing with NPV | % |
| PPV | TP/(TP+FP) | Confusing with sensitivity | % |
| NPV | TN/(TN+FN) | Confusing with specificity | % |
| Likelihood ratio | Ratio of probabilities | Confusing with risk ratio | dimensionless |
| Positive LR | Sensitivity/(1-specificity) | Using without prevalence | dimensionless |
| Negative LR | (1-sensitivity)/specificity | Using without prevalence | dimensionless |
| Accuracy | (TP+TN)/total | Misleading at imbalance | % |
| AUROC | Ranking across thresholds | Ignoring prevalence | dimensionless |
| AUPRC | Precision-recall area | Confusing with AUROC | dimensionless |
| Calibration | Agreement of predicted and observed | Confusing with discrimination | method |
| Co-primary endpoint | Endpoints that must all be met | Selecting the best | endpoint |
| Hierarchical endpoint | Ordered testing | Confusing with co-primary | endpoint |
| Type I error | False positive rate | Ignoring multiplicity | probability |
| Power | Probability of detecting an effect | Confusing with sample size | probability |
| Indeterminate | Test cannot produce a result | Counting as negative | result |
| Missing data | Absent measurements | Complete-case analysis | method |
| Informative missingness | Missingness related to outcome | Ignoring | bias |
| Blinding | Concealing the other result | Assuming it happened | method |
| Consecutive sampling | Every eligible participant | Convenience sampling | method |
| Case-control | Sampling by outcome | Using for PPV | design |
| Cross-sectional | Sampling at one time | Confusing with cohort | design |
| Prospective | Sampling forward in time | Confusing with cross-sectional | design |
| External validation | New population or site | Confusing with internal | design |
| Internal validation | Same population | Confusing with external | design |
| Pre-registration | Public protocol before study | Post-hoc analysis | practice |
| STARD | Reporting guideline | Treating as a design | guideline |
| Flow diagram | Participant flow | Omitting exclusions | figure |
| Spectrum of disease | Severity and comorbidity | Ignoring | population |
| Verification bias | Only some get the reference | Ignoring | bias |
| Incorporation bias | Reference includes index | Ignoring | bias |
| Review bias | Reference reader sees index | Ignoring | bias |
| Co-primary | Joint endpoints | Selecting the best | endpoint |
| Intention-to-diagnose | Include indeterminate in the denominator | Excluding them | method |
| Per-protocol | Only classifiable cases | Generalizing to all | method |
| Complete-case | Analyse observed cases | Ignoring missingness | method |
| Imputation | Fill missing values | Assuming random | method |
| Worst-case bound | Assign missing to extremes | Reporting only one bound | method |
| Net reclassification | Change in risk categories | Confusing with AUC | method |
| Decision curve | Net benefit across thresholds | Confusing with accuracy | method |
| Reference error | Error in the truth | Assuming zero | method |
| Imperfect reference | Reference not perfect | Ignoring | method |
| Latent class | Model without a gold standard | Assuming Identifiability | method |
| Composite reference | Combination of references | Assuming equivalence | method |
| Two-stage sampling | Reference on a subset | Verification bias | method |
| Consecutive sample | All eligible | Convenience | method |
| Random sample | Random selection | Selection bias | method |
| Spectrum | Severity and comorbidity | Ignoring | population |
| Case mix | Distribution of conditions | Ignoring | population |
| Comorbidity | Other conditions | Excluding | population |
| Disease severity | Stage or grade | Ignoring | population |
| Age spectrum | Age range | Excluding elderly | population |
| Setting | Primary, secondary, tertiary | Ignoring | population |
| Operator | Who performs the test | Ignoring | method |
| Learning curve | Change with experience | Ignoring | method |
| Batch | Technical variation | Ignoring | method |

## 5. The mechanics, step by step

The procedure is written for a diagnostic accuracy study of a signal-derived test, and it applies to a screening, diagnostic, or monitoring claim with adaptations.

### Step 1. Define the question and the role

**Input:** the clinical or product decision.

**Output:** a question with the target condition, the index test, the reference, the population, the role, and the decision threshold.

**Rationale:** the question determines the design. A screening question and a diagnostic question need different sensitivities, specificities, and populations.

**Branch point:** if the role is screening, the study should include the full target population and a confirmatory pathway. If the role is diagnosis, the study should include participants with the differential diagnosis, not healthy controls.

**Edge case:** a study that enrols healthy controls and diseased cases answers a different question from a study that enrols participants with symptoms. The first estimates discrimination between extremes; the second estimates accuracy in the clinical grey zone.

**The question table.**

| Element | Question | Example |
|---|---|---|
| Target condition | What is being detected | AF on 12-lead ECG |
| Index test | What is being evaluated | Wrist PPG algorithm |
| Reference | What is the truth | Simultaneous 12-lead |
| Population | Who is studied | Adults 50-75, primary care |
| Role | Where in the pathway | Screening |
| Threshold | What decision | Notify for confirmation |

A question that cannot fill the table is not ready for a protocol. The missing row is the first thing to fix.

### Step 2. Choose the reference standard

**Input:** the target condition and the available references.

**Output:** a reference with a stated independence, timing, and error.

**Rationale:** the reference is the anchor for every metric. A biased reference biases the sensitivity and specificity in a direction that depends on the error.

**Independence:** the reference must not use the index test. If the reference is another wearable with the same sensor, the errors are correlated. If the reference is a clinical diagnosis that incorporates the index result, the estimate is inflated.

**Timing:** the reference should be measured close in time to the index test. A delay allows the condition to change and biases the estimate toward disagreement.

**Branch point:** if no independent reference exists, the study can use a composite reference or a latent class model, with a stated assumption. The assumption is part of the result.

**Edge case:** a reference that is invasive or expensive can only be applied to a subset. If the subset is selected by the index result, the study has verification bias.

**Reference comparison.**

| Reference | Independence | Timing | Error | Feasibility |
|---|---|---|---|---|
| Simultaneous 12-lead ECG | High | Same time | Reader variability | High |
| Holter | High | Same day | Episode capture | Moderate |
| Patch | High | Days later | Time gap | Moderate |
| Another wearable | Low if same sensor | Same time | Correlated | High |
| Clinical diagnosis | Low if uses index | Variable | Incorporation | High |

The table is a design aid. The best reference is independent, simultaneous, and feasible; the worst is another device with the same sensor. The choice should be justified, not assumed.

### Step 3. Define the population and the spectrum

**Input:** the intended use and the deployment setting.

**Output:** eligibility criteria, the sampling frame, and the expected spectrum.

**Rationale:** the population determines the prevalence and the spectrum, and both affect the metrics. A study that excludes the population that will use the test is not a validation for that population.

**Contents:** age, sex, comorbidity, symptom severity, care setting, device, and language. The criteria should be broad enough to include the deployment population and narrow enough to be reproducible.

**Branch point:** if the deployment population is broad, the study should stratify by the major subgroups. A single pooled estimate can hide a subgroup difference.

**Edge case:** a study that excludes participants who cannot use the device has a usability-related selection bias. The excluded participants are part of the deployment population.

**Spectrum dimensions.**

| Dimension | Example | Effect on accuracy |
|---|---|---|
| Severity | Mild versus severe AF | Sensitivity rises with severity |
| Comorbidity | Heart failure, thyroid disease | Specificity falls |
| Age | 20-40 versus 65-85 | Artifact and prevalence differ |
| Setting | Primary versus tertiary | Prevalence and spectrum |
| Device | One versus another | Signal and artifact |
| Operator | Expert versus novice | Usability and signal quality |

A study that is narrow on all six dimensions is internally clean and externally limited. A study that is broad is more transportable and more variable. The choice should match the claim.

### Step 4. Sample consecutively and avoid verification bias

**Input:** the sampling frame and the reference.

**Output:** a sample with a documented selection process.

**Rationale:** convenience sampling and selective referral change the spectrum. Consecutive sampling of all eligible participants is the default. If the reference is applied only to a subset, the subset must be defined independently of the index test.

**Arithmetic:** a clinic sees 1,000 eligible participants, and 300 receive the reference because of scheduling. If the 300 are younger and healthier, the spectrum is biased. The direction of the bias depends on the test's behaviour in the excluded group.

**Branch point:** if the reference cannot be applied to everyone, use a random subset or a two-stage design with a stated correction.

**Edge case:** a participant who refuses the reference is not missing at random. The refusal may be related to the disease severity.

**Sampling comparison.**

| Design | Selection | Strength | Weakness |
|---|---|---|---|
| Consecutive | All eligible | Representative | Slow |
| Random | Random subset | Unbiased | Needs a frame |
| Convenience | Available | Fast | Spectrum bias |
| Case-control | By outcome | Efficient | Artificial prevalence |
| Enriched | High-risk | More cases | Narrow spectrum |
| Two-stage | Reference on subset | Feasible | Requires correction |

The design determines the inference. A consecutive sample supports a prevalence estimate; a case-control sample does not. An enriched sample supports a sensitivity estimate in high-risk participants, not in the population.

### Step 5. Blind the readers

**Input:** the index test and the reference.

**Output:** a blinding procedure and a check that it was maintained.

**Rationale:** a reader who sees the index result is more likely to interpret the reference as positive. The bias is unconscious and well documented.

**Method:** the reference reader should not see the index result; the index reader should not see the reference. The order and the data flow should be documented. A blinded read is a design feature, not a courtesy.

**Branch point:** if blinding is impossible, the study should measure the potential bias with a blinded substudy.

**Edge case:** the index test may be embedded in the clinical record and visible to the reference reader. The study should use a separate data capture.

**Blinding checklist.**

| Item | Question | Evidence |
|---|---|---|
| Reference reader | Did they see the index result | Attestation |
| Index reader | Did they see the reference | Attestation |
| Adjudicator | Were disagreements blinded | Protocol |
| Analyst | Was the analysis blinded | Code review |
| Data capture | Were the results separated | System log |

Blinding is a property of the workflow, not of the intent. A study that says "readers were blinded" without describing how is not verifiable.

### Step 6. Pre-specify the analysis and the sample size

**Input:** the expected sensitivity, specificity, prevalence, and precision.

**Output:** a sample size and an analysis plan.

**Rationale:** the sample size determines the precision of the estimates. A study with 20 cases has a wide confidence interval for sensitivity, even if the point estimate is high.

**Arithmetic:** for an expected sensitivity of 0.95 and a desired half-width of 0.05, the approximate sample size is 1.96² × 0.95 × 0.05 / 0.05² = 73 cases. For a half-width of 0.03, it is 203 cases. The number of cases, not the total sample, drives the precision.

**Branch point:** if the prevalence is low, the total sample must be large to accrue enough cases. A screening study with 1% prevalence needs 7,300 participants for 73 cases.

**Edge case:** the sample size for specificity is often larger because the negative group is larger. The study should be sized for the co-primary endpoints, not only for sensitivity.

**Sample size table.**

| Target | Expected | Half-width | Cases needed |
|---|---|---:|---:|
| Sensitivity | 0.95 | 0.05 | 73 |
| Sensitivity | 0.95 | 0.03 | 203 |
| Sensitivity | 0.90 | 0.05 | 138 |
| Specificity | 0.97 | 0.02 | 280 |
| Specificity | 0.97 | 0.01 | 1,118 |

The table uses the normal approximation n = z²p(1-p)/half-width². The specificity row needs more participants because the expected proportion is near 1 and the variance is small but the precision requirement is tight.

The total sample is the case count divided by the prevalence. At 1% prevalence, 73 cases require 7,300 participants. A study that plans 1,000 participants and expects 1% prevalence will have about 10 cases, which is far too few.

### Step 7. Collect and link the data

**Input:** the index and reference results.

**Output:** a linked dataset with time stamps, quality flags, and indeterminate results.

**Rationale:** the linkage determines which index result is compared with which reference. A time mismatch, a quality failure, or an indeterminate result changes the 2×2 table.

**Arithmetic:** if 5% of index tests are indeterminate and are excluded, the sensitivity is computed on the 95% that were classifiable. The performance in the full population is lower if the indeterminate cases are more likely to be positive.

**Branch point:** if the indeterminate rate is high, report it and analyse it as a third outcome, not as a negative.

**Edge case:** an indeterminate result can be a quality failure, a technical failure, or a clinical ambiguity. The categories should be distinguished.

### Step 8. Build the 2×2 table

**Input:** the linked results.

**Output:** a 2×2 table with counts and the derived metrics.

**Rationale:** the table is the basis of every metric. A table that does not sum to the sample is a warning.

**Arithmetic:** with TP = 95, FP = 27, FN = 5, TN = 873, the sensitivity is 95/100 = 0.95, the specificity is 873/900 = 0.970, the PPV is 95/122 = 0.779, and the NPV is 873/878 = 0.994. The prevalence is 100/1,000 = 0.10.

**Branch point:** if the table has zero cells, the estimates are unstable. Use exact confidence intervals or a continuity correction.

**Edge case:** a table with a very high sensitivity and a very low specificity can still have a good NPV. The metric that matters depends on the decision.

### Step 9. Compute confidence intervals

**Input:** the counts.

**Output:** confidence intervals for each metric.

**Rationale:** a point estimate without an interval hides the uncertainty. The interval width depends on the number of cases and controls, not on the total sample.

**Method:** the Wilson interval for a proportion is more accurate than the normal approximation at extreme proportions. The exact binomial interval is conservative. For AUROC, use a bootstrap or the DeLong method.

**Arithmetic:** 153/450 = 0.340 with a Wilson 95% interval of 0.298 to 0.385. 236/240 = 0.983 with an interval of 0.958 to 0.994. The second interval is narrower because the denominator is larger and the proportion is near 1.

**Branch point:** if the interval crosses the prespecified threshold, the study is inconclusive, even if the point estimate is above it.

**Edge case:** multiple metrics with intervals require a multiplicity adjustment if the claim is joint.

**Confidence interval methods.**

| Method | Use | Strength | Weakness |
|---|---|---|---|
| Wald | Large n, mid-range p | Simple | Poor near 0 or 1 |
| Wilson | Proportions | Good coverage | Slightly complex |
| Exact binomial | Small n | Conservative | Wide |
| Bootstrap | AUROC, complex metrics | Flexible | Assumes resampling unit |
| DeLong | AUROC | Fast | Assumes independence |

The method should be pre-specified. A study that switches methods after seeing the interval is not reproducible.

### Step 10. Adjust for prevalence and spectrum

**Input:** the study metrics and the deployment prevalence.

**Output:** PPV and NPV at the deployment prevalence, and a subgroup analysis.

**Rationale:** sensitivity and specificity are assumed to transfer across prevalence, but PPV and NPV do not. The adjustment is a simple application of Bayes' rule.

**Arithmetic:** with sensitivity 0.95 and specificity 0.97, the PPV is 0.242 at 1% prevalence, 0.625 at 5%, and 0.779 at 10%. The same test has very different predictive values.

**Branch point:** if the deployment prevalence is unknown, report the PPV across a range, not a single value.

**Edge case:** sensitivity and specificity do not always transfer across spectrum. A test validated in severe disease can be less sensitive in mild disease, and the prevalence adjustment does not correct for that.

### Step 11. Handle indeterminate results and missing data

**Input:** the indeterminate and missing results.

**Output:** a primary analysis and a sensitivity analysis.

**Rationale:** the primary analysis should include all participants, with indeterminate results counted as failures or handled by a prespecified rule. A complete-case analysis is a sensitivity analysis, not the primary result.

**Arithmetic:** if 12% of tests are indeterminate and the indeterminate group has a higher disease prevalence, excluding them biases the sensitivity downward or upward depending on the rule.

**Branch point:** if the missingness is informative, use multiple imputation or a worst-case bound.

**Edge case:** a participant who never wore the device has no index result. Counting them as negative is wrong; counting them as positive is also wrong. They are missing, and the missingness is part of the result.

**Indeterminate handling table.**

| Rule | Denominator | Effect | Use |
|---|---|---|---|
| Exclude | Classifiable only | Optimistic | Per-protocol |
| Count as negative | All | Conservative for sensitivity | Intention-to-diagnose |
| Count as positive | All | Conservative for specificity | Safety |
| Third outcome | All | Transparent | Primary analysis |

The primary analysis should use the third-outcome or intention-to-diagnose rule, with the others as sensitivity analyses. A report that uses only the exclude rule is overstating the performance.

### Step 12. Report against STARD and register

**Input:** the study and the results.

**Output:** a report that follows STARD 2015, with a flow diagram, a 2×2 table, confidence intervals, and a discussion of limitations.

**Rationale:** the reporting guideline is not a design, but a study that cannot be reported cannot be evaluated. The flow diagram and the table are the minimum.

**Contents:** the 30 STARD items, including the title, abstract, introduction, methods, results, and discussion. The items that are most often missing are the sample size, the indeterminate results, the missing data, and the registration.

**Branch point:** if the study is AI-based, use STARD-AI, which adds items for the model, the data, and the code.

**Edge case:** a study that is registered after the analysis is not a pre-registered study. The registration date is part of the record.

**The minimum report.**

| Item | Why it matters |
|---|---|
| Target condition and role | Bounds the claim |
| Population and setting | Transport |
| Sampling and dates | Selection |
| Reference and timing | Truth |
| Blinding | Bias |
| Indeterminate results | Denominator |
| 2×2 table | Reproducibility |
| Confidence intervals | Uncertainty |
| Prevalence | PPV |
| Limitations | Honesty |

A report with these ten items is evaluable even if it is short. A report without them is a press release.

### Decision table for validation design

| Condition | Design choice | Rationale |
|---|---|---|
| Screening role | Consecutive target population | Spectrum and prevalence |
| Diagnostic role | Differential diagnosis population | Clinical grey zone |
| Low prevalence | Large sample or enriched design | Enough cases |
| Invasive reference | Random subset or two-stage | Verification bias |
| No independent reference | Composite or latent class | Stated assumption |
| High indeterminate rate | Third outcome category | Avoid false negatives |
| Informative missingness | Worst-case bound | Sensitivity analysis |
| AI-based test | STARD-AI | Model and data reporting |

## 6. Worked example

### 6.1 The case

A wrist PPG algorithm is proposed as a screening test for atrial fibrillation in adults aged 50 to 75. The study enrols 1,000 participants from primary care, applies the index test and a simultaneous 12-lead ECG reference, and reports the 2×2 table. The deployment prevalence is expected to be 1%, lower than the study prevalence.

### 6.2 The 2×2 table

|  | AF on ECG | No AF on ECG | Total |
|---|---:|---:|---:|
| PPG positive | 95 | 27 | 122 |
| PPG negative | 5 | 873 | 878 |
| Total | 100 | 900 | 1,000 |

The study prevalence is 100/1,000 = 10%.

The table has 122 positive index tests and 878 negative. The reference has 100 positive and 900 negative. The cells sum to 1,000, which is a check that no participant is missing.

A second check is the direction of the errors: 27 false positives and 5 false negatives. The false positives are more numerous, which is expected for a sensitive screening test. The ratio of FP to FN is about 5.4:1, which reflects the choice of threshold.

### 6.3 The metrics

- Sensitivity = 95/100 = 0.950.
- Specificity = 873/900 = 0.970.
- PPV = 95/122 = 0.779.
- NPV = 873/878 = 0.994.
- Positive likelihood ratio = 0.950/(1 - 0.970) = 31.7.
- Negative likelihood ratio = (1 - 0.950)/0.970 = 0.0515.

A positive likelihood ratio of 31.7 means that a positive result is 28.8 times more likely in a participant with AF than without. A negative likelihood ratio of 0.0515 means that a negative result is about 5% as likely in a participant with AF, which is a strong rule-out.

The likelihood ratios are useful because they can be combined with a pre-test odds to estimate a post-test probability. At 1% prevalence, the pre-test odds are 0.0101. A positive result multiplies the odds by 31.7, giving 0.320, or a post-test probability of 24.2%. This matches the PPV computed directly, which is a check on the arithmetic.

### 6.4 Prevalence adjustment

The deployment prevalence is 1%, not 10%. The sensitivity and specificity are assumed to transfer.

| Prevalence | PPV | NPV |
|---:|---:|---:|
| 0.1% | 0.031 | 0.99995 |
| 1% | 0.242 | 0.99948 |
| 5% | 0.625 | 0.99729 |
| 10% | 0.779 | 0.99431 |

At 1% prevalence, the PPV is 0.242. A positive result has a 24% chance of being true AF, and 76% of positive results are false. The NPV is 0.9995, so a negative result is reassuring.

The arithmetic is:

- TP rate = 0.01 × 0.95 = 0.0095.
- FP rate = 0.99 × (1 - 0.970) = 0.0297.
- PPV = 0.0095/(0.0095 + 0.0297) = 0.242.

The table in the script uses the same specificity and reports the same value. The PPV is 0.242 at 1% prevalence.

| Component | Value | Derivation |
|---|---:|---|
| Prevalence | 0.010 | assumed |
| Sensitivity | 0.950 | study |
| Specificity | 0.970 | study |
| TP rate | 0.0095 | 0.01 × 0.95 |
| FP rate | 0.0297 | 0.99 × 0.030 |
| PPV | 0.242 | 0.0095/(0.0095+0.0297) |
| FN rate | 0.0005 | 0.01 × 0.05 |
| TN rate | 0.9603 | 0.99 × 0.970 |
| NPV | 0.99948 | 0.9603/(0.9603+0.0005) |

The table is the full derivation. A reader can reproduce every number from the five inputs.

### 6.5 Confidence intervals

The Wilson 95% intervals for the key proportions are:

| Proportion | Count | Estimate | 95% CI |
|---|---|---:|---|
| Sensitivity | 95/100 | 0.950 | wide, about 0.89-0.98 |
| Specificity | 873/900 | 0.970 | about 0.95-0.98 |
| PPV at 10% | 95/122 | 0.779 | about 0.70-0.84 |
| Confirmed AF after notification | 153/450 | 0.340 | 0.298-0.385 |

The interval for the confirmed AF proportion is from the Apple Heart Study, used here as a benchmark; it is not from the synthetic table. The interval is 0.298 to 0.385, which is a 8.7 percentage point range.

### 6.6 Diagnostic 1: the prevalence mismatch

The study PPV of 0.779 is computed at 10% prevalence. The deployment PPV at 1% is 0.242. The difference is not a model error; it is the prior.

The diagnostic is to compute the PPV at the deployment prevalence before the study is designed. If the resulting PPV is unacceptable, the test needs a higher specificity, a higher prevalence population, or a confirmatory step.

The consequence of missing the diagnostic is a product that looks good in the study and generates mostly false positives in the field. The Apple Heart Study's 34% confirmation rate among notified participants is an example of the same phenomenon at a population scale.

### 6.7 Diagnostic 2: spectrum bias

Suppose the study enrols participants from a cardiology clinic, where the AF prevalence is 30% and the disease is more severe. The sensitivity may be higher because the AF is easier to detect, and the specificity may be lower because the comparison group has more confounding rhythms.

The diagnostic is to compare the study population with the deployment population on age, comorbidity, and symptom severity. If the study population is more severe, the sensitivity is optimistic.

The consequence is a test that performs well in the clinic and worse in primary care. The fix is a primary-care validation, not a model retrained on the clinic data.

### 6.8 Varying one assumption

Suppose the specificity falls from 0.970 to 0.95 at the deployment prevalence of 1%. The PPV becomes 0.0095/(0.0095 + 0.0495) = 0.161. A 1.7 percentage point drop in specificity reduces the PPV from 0.242 to 0.161, a 33% relative reduction. At low prevalence, specificity dominates the PPV.

Suppose the sensitivity falls from 0.95 to 0.90. The PPV becomes 0.009/(0.009 + 0.03267) = 0.216, a smaller change. At low prevalence, sensitivity matters less for PPV and more for NPV.

Suppose the prevalence is 5% instead of 1%. The PPV rises to 0.625, and the same test becomes useful for a higher-risk population. The population selection is a lever on the PPV.

### 6.9 Conclusion of the case

The test has a sensitivity of 0.950, a specificity of 0.970, and a study PPV of 0.779 at 10% prevalence. At the deployment prevalence of 1%, the PPV is 0.242 and the NPV is 0.9995. The test is a good rule-out and a poor standalone rule-in in a low-prevalence population.

The two diagnostics (prevalence mismatch, spectrum bias) catch the two errors that make a validation look better than the deployment. The assumption that matters most is the prevalence, because it changes the PPV by a factor of three.

### 6.9.1 The sensitivity table

| Assumption changed | PPV at 1% | Change from baseline |
|---|---:|---:|
| Baseline: sens 0.95, spec 0.970 | 0.242 | 0.000 |
| Specificity 0.95 | 0.161 | -0.064 |
| Sensitivity 0.90 | 0.232 | -0.010 |
| Prevalence 5% | 0.625 | +0.383 |
| Prevalence 10% | 0.779 | +0.537 |

The table shows that specificity and prevalence dominate the PPV. A 1.7 point drop in specificity costs more than a 5 point drop in sensitivity. A prevalence change from 1% to 10% more than triples the PPV.

The design implication is that a low-prevalence screening programme should invest in specificity and in a confirmatory step, not only in sensitivity.

### 6.10 What the case does not show

The case is a single 2×2 table from a hypothetical study. It does not show:

- **The sampling process.** Consecutive, random, or convenience is not specified.
- **The reference adjudication.** The number of readers and the disagreement rate are not given.
- **The indeterminate rate.** The table assumes every participant has a result.
- **The spectrum.** Age, comorbidity, and severity are not described.
- **The external validation.** There is no second site or device.
- **The utility.** The PPV is not translated into a decision or a cost.

Each omission is a design question that a real protocol must answer. The arithmetic transfers; the conclusion does not.

### 6.11 The cost of the two errors

The 2×2 table treats a false positive and a false negative as symmetric. In practice they are not.

| Error | Consequence | Cost |
|---|---|---|
| False positive | Unnecessary confirmation, anxiety | Clinic visit, cost |
| False negative | Missed AF, stroke risk | Morbidity |
| True positive | Confirmation and treatment | Appropriate cost |
| True negative | Reassurance | None |

At 1% prevalence and the case test, 1,000 participants produce about 10 true positives, 30 false positives, and 1 false negative. The false positives outnumber the true positives by three to one, which is why the confirmation pathway must be efficient.

The cost asymmetry is also the reason a threshold is a clinical choice. Moving the threshold changes the ratio of false positives to false negatives, and the optimal ratio depends on the relative costs, not on the statistics.

A study that reports only accuracy hides this trade-off. The 2×2 table at the intended threshold is the minimum disclosure.

The table also shows why a screening product needs a confirmation step. The false positives are not a failure of the test; they are the cost of the sensitivity. A product that treats a positive screen as a diagnosis has misunderstood the measurement.

The confirmation step is part of the validation, not an operational afterthought. A study that reports the 2×2 table at the intended threshold makes it explicit.

The cost asymmetry is the reason a screening test is judged on PPV as well as sensitivity. A test that finds every case and floods the clinic is not a good screening test.

## 7. Code, query, or calculation

The script computes the 2×2 metrics, the prevalence-adjusted PPV and NPV, Wilson confidence intervals, and a bootstrap AUROC interval. It uses only the standard library.

```python
import math, random

def ppv_npv(prev,sens,spec):
    tp=prev*sens; fp=(1-prev)*(1-spec)
    fn=prev*(1-sens); tn=(1-prev)*spec
    return tp/(tp+fp), tn/(tn+fn)

for prev in (0.001,0.01,0.05,0.10):
    p,n=ppv_npv(prev,0.95,0.970)
    print("prev=%.3f PPV=%.3f NPV=%.5f" % (prev,p,n))

def wilson(k,n,z=1.96):
    p=k/n; den=1+z*z/n
    centre=(p+z*z/(2*n))/den
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return max(0,centre-half),min(1,centre+half)

for k,n in ((153,450),(2161,419297),(236,240),(238,239)):
    print("k=%d n=%d p=%.4f ci=%.4f..%.4f" % (k,n,k/n,*wilson(k,n)))

def auroc(y,s):
    pairs=sorted(zip(s,y)); n1=sum(y); n0=len(y)-n1; rs=0; i=0
    while i<len(pairs):
        j=i
        while j+1<len(pairs) and pairs[j+1][0]==pairs[i][0]: j+=1
        avg=(i+1+j+1)/2
        for k in range(i,j+1):
            if pairs[k][1]==1: rs+=avg
        i=j+1
    return (rs-n1*(n1+1)/2)/(n1*n0)

random.seed(5)
y=[1 if random.random()<0.3 else 0 for _ in range(200)]
s=[(0.7 if yy else 0.35)+random.gauss(0,0.2) for yy in y]
base=auroc(y,s); vals=[]
for _ in range(500):
    idx=[random.randrange(200) for _ in range(200)]
    yy=[y[i] for i in idx]; ss=[s[i] for i in idx]
    if 0 < sum(yy) < len(yy): vals.append(auroc(yy,ss))
vals.sort()
print("auroc=%.3f ci=%.3f..%.3f" % (base,vals[int(0.025*len(vals))],vals[int(0.975*len(vals))]))

for hw in (0.05,0.03):
    n=1.96**2*0.95*0.05/hw**2
    print("sens halfwidth=%.2f n=%.0f" % (hw,n))
```

Expected output:

```text
prev=0.001 PPV=0.031 NPV=0.99995
prev=0.010 PPV=0.242 NPV=0.99948
prev=0.050 PPV=0.625 NPV=0.99729
prev=0.100 PPV=0.779 NPV=0.99431
k=153 n=450 p=0.3400 ci=0.2978..0.3850
k=2161 n=419297 p=0.0052 ci=0.0049..0.0054
k=236 n=240 p=0.9833 ci=0.9579..0.9935
k=238 n=239 p=0.9958 ci=0.9767..0.9993
auroc=0.924 ci=0.885..0.956
sens halfwidth=0.05 n=73
sens halfwidth=0.03 n=203
```

Three implementation notes.

The prevalence adjustment assumes that sensitivity and specificity are constant across prevalence. That assumption is reasonable for a test with a fixed threshold and a stable spectrum, and it fails when the spectrum changes. The adjustment is a first-order correction, not a guarantee.

The Wilson interval is used because the normal approximation is poor near 0 or 1. At 236/240, the normal interval would extend above 1; the Wilson interval does not.

The bootstrap resamples participants, not records, so the interval reflects the participant-level uncertainty. A bootstrap that resamples records from the same participant would be too narrow.

A fourth note is about the prevalence table. The values assume a fixed sensitivity and specificity, which is a first-order approximation. At a different prevalence, the spectrum often changes too, and the approximation is less reliable than it looks. The table is a planning tool, not a substitute for a study in the deployment population.

A fifth note is about the sample-size formula. It is for a single proportion. A study with co-primary endpoints needs a sample size for each, and the larger of the two, with a multiplicity adjustment if both are tested.

A sixth note is about the Wilson interval. The formula uses the normal quantile 1.96 for a 95% interval. For a 90% or 99% interval, the quantile changes, and the interval width changes with it. The confidence level should be stated.

A final note is that the AUROC is computed on a synthetic score with a fixed separation. The 0.924 is a property of the generator, not of a real model. The interval is the point of the demonstration, not the value.

A seventh note is that the bootstrap resamples with replacement and can produce a sample with no positives or no negatives. The code skips those samples, which slightly biases the interval toward the center. A larger number of resamples reduces the effect. Five hundred resamples are adequate for a demonstration; a publication should use at least 1,000 and state the number.

The script is deliberately small. A production analysis would use a dedicated library, a prespecified interval method, and a participant-level data structure. The arithmetic is the same; the audit trail is not.

A final check on the arithmetic: the PPV at 1% (0.242) is not the PPV at the study prevalence (0.779). The difference is the prior, not the test. A reader who compares the two without adjusting will conclude that the test degraded, when nothing about the test changed.

The adjustment is the most important line in the analysis, and it is the one most often left to the reader.

A practical rule is to report the PPV at the study prevalence and at the expected deployment prevalence, side by side. The second is a planning estimate, not a result.

A second rule is to report the indeterminate rate next to the accuracy. The rate is part of the performance, not a footnote.

A third rule is to report the confidence interval next to every point estimate. A number without an interval is a claim without a range. The interval is the honesty of the point estimate.

A validation report without these three rules is a summary, not an analysis. The two numbers tell the reader how much of the performance is the test and how much is the population.

## 8. Data, standards, and regulatory context

Validation studies are shaped by reporting guidelines, regulatory guidance, and the design of the claim. The guidelines do not replace the design, but they define what a reader can evaluate.

| Source | What it constrains | Status | Practical consequence |
|---|---|---|---|
| STARD 2015 | Reporting of diagnostic accuracy studies | Reporting guideline | 30 items, flow diagram, 2×2 table |
| STARD-AI 2025 | Reporting of AI diagnostic studies | Reporting guideline | Model, data, and code items |
| FDA device guidance | Premarket evidence | Binding through review | Analytical and clinical validation |
| FDA arrhythmia special controls | Algorithm validation | Special controls | EC57 databases, multiple readers |
| ISO 14155 | Clinical investigation of devices | Consensus | Good clinical practice |
| Declaration of Helsinki | Ethical conduct | Professional | Ethics review and consent |
| CONSORT | Reporting of trials | Reporting guideline | Trial flow and outcomes |
| PRISMA | Reporting of reviews | Reporting guideline | Search and synthesis |
| EU MDR | Clinical evaluation | Binding in the EU | Clinical evidence and PMS |

### 8.1 STARD 2015

STARD 2015 is a 30-item checklist for diagnostic accuracy studies. The items that matter most for a signal-derived test are:

- The target condition and the intended use.
- The index test and the reference standard, with enough detail to reproduce them.
- The participant flow, including exclusions and indeterminate results.
- The 2×2 table and the accuracy estimates with confidence intervals.
- The sample size and the analysis plan.
- The limitations and the clinical implications.
- Registration and protocol access.

The guideline is not a design. A study can follow STARD and still have spectrum bias or verification bias. The guideline makes the biases visible, which is its purpose.

### 8.1.1 The STARD flow diagram

The flow diagram is the most useful single item in STARD. It shows the number of participants at each stage:

```text
Eligible participants (n = 1,000)
        |
Excluded (n = 50): declined (30), ineligible (20)
        |
Index test (n = 950)
        |
Indeterminate (n = 48)
        |
Reference standard (n = 902)
        |
2x2 table (n = 902)
```

The diagram makes the denominators visible. A sensitivity computed on 902 participants is not the same as a sensitivity computed on 1,000, and the difference is the indeterminate and excluded participants.

The diagram also makes the selection visible. If the 48 indeterminate participants are younger or healthier, the 902 are not a random sample of the 950.

### 8.2 STARD-AI

STARD-AI extends STARD for AI-based diagnostic studies. It adds items for the model architecture, the training data, the code, the calibration, and the subgroup performance.

The extension matters because an AI study has more degrees of freedom than a fixed test. The model can be retrained, the threshold can be tuned, and the preprocessing can change. The reporting must describe these choices.

### 8.2.1 The STARD-AI items that matter most

STARD-AI adds items that a fixed-test study does not need:

- the model architecture and the training data,
- the code and the weights,
- the calibration and the threshold,
- the subgroup performance,
- the external validation,
- the monitoring plan.

The items are not optional for an AI claim. A study that reports a model without the code or the training data cannot be reproduced; a study that reports accuracy without calibration cannot support a probability decision.

### 8.3 What the guidelines do not cover

Three gaps.

First, the guidelines do not specify the reference standard. A study can report a poor reference and still follow STARD. The reader must judge the reference.

Second, the guidelines do not specify the deployment prevalence. A study can report a PPV at the study prevalence and leave the reader to adjust. The adjustment is the reader's job, but the study should report the prevalence.

Third, the guidelines do not specify the utility. A study can report accuracy without a decision analysis. The utility is a separate study or a decision curve.

### 8.4 Regulatory evidence

The FDA and the EU MDR require analytical and clinical validation for a device claim. The evidence package includes the reference standard, the population, the sample size, the accuracy, the subgroup performance, and the limitations.

The regulatory review is not a substitute for the scientific review, and the scientific review is not a substitute for the regulatory review. A study can be scientifically sound and not meet a regulatory requirement, or vice versa.

### 8.5 A reporting checklist for this topic

| Item | Present in the report |
|---|---|
| Target condition and role | Required |
| Population, setting, dates | Required |
| Sampling method | Required |
| Reference and timing | Required |
| Blinding | Required |
| Indeterminate results | Required |
| Missing data | Required |
| 2×2 table | Required |
| Confidence intervals | Required |
| Prevalence adjustment | Recommended |
| Subgroup analysis | Recommended |
| Registration | Recommended |

The checklist is shorter than STARD and easier to use as a first pass. A report that satisfies it can be upgraded to STARD with the additional items.

## 9. Industry practice and case evidence

### 9.1 The Apple Heart Study

The Apple Heart Study enrolled 419,297 participants over eight months, with a median monitoring of 117 days. Of these, 2,161 (0.52%) received an irregular pulse notification; the rate was 3.1% in participants aged 65 and older and 0.16% in those aged 22 to 40.

Among the 450 participants who returned analyzable ECG patches, 153 (34%) had AF confirmed. The positive predictive value was 0.84 (95% CI 0.76-0.92) for AF on a simultaneous ECG with a notification, and 0.71 (97.5% CI 0.69-0.74) for an individual irregular tachogram.

The study is a pragmatic, siteless design, which is both its strength and its limitation. The strength is the scale and the real-world setting; the limitation is the self-selection and the incomplete patch return. The 34% confirmation rate is a PPV at the study's notification prevalence, not at the population prevalence.

The signal-chain lesson is that the notification rate depends on the quality gate and the irregularity feature, and the confirmation rate depends on the prevalence and the reference. The two are separate measurements.

### 9.2 The Apple ECG app

The ECG app validation recorded 30 s single-lead ECGs simultaneously with 12-lead ECGs in 588 participants. The reported sensitivity was 98.3% and specificity 99.6% for classifiable recordings; including unclassifiable recordings, sensitivity was 95.5% and specificity 97.1%. About 87.8% of recordings were classifiable.

The design is strong because the reference is simultaneous and adjudicated. The limitation is the controlled setting: staff-assisted recordings with rested arms. The real-world inconclusive rate is likely higher.

### 9.3 Consumer sleep devices

Chinoy et al. compared seven consumer sleep devices with polysomnography. The agreement varied by device and stage, with better agreement for total sleep time and worse for REM and N1.

The validation lesson is that the per-stage agreement is the relevant metric for a staging claim. An overall accuracy can be high and a stage-specific sensitivity low.

### 9.4 Remote monitoring trials

The Cochrane review of structured telephone support and non-invasive telemonitoring for heart failure found modest and heterogeneous effects. The lesson is that the intervention and the workflow determine the outcome, not the sensor alone.

The validation implication is that a technology validation is not a clinical validation. A device can measure accurately and fail to improve outcomes because the care pathway does not use the measurement.

### 9.5 What the cases have in common

- **The reference is simultaneous and independent.** The strong studies use a reference measured at the same time.
- **The population is described.** Age, setting, and selection are reported.
- **The prevalence is reported.** The PPV is interpretable only with the prevalence.
- **The indeterminate rate is reported.** The Apple ECG study reports the classifiable share.
- **The limitations are stated.** The controlled setting, the self-selection, and the incomplete follow-up are named.

The inference is that the quality of a validation is visible in the reporting, not only in the numbers.

### 9.6 A comparison of the public cases

| Case | Design | Reference | Prevalence | Key limitation |
|---|---|---|---|---|
| Apple Heart Study | Pragmatic, siteless | Patch, 13 days later | Low, self-selected | Incomplete patch return |
| Apple ECG app | Prospective, 5 sites | Simultaneous 12-lead | Enriched | Controlled setting |
| Sleep devices | Lab comparison | Polysomnography | Clinical | Device and stage |
| Telemonitoring | RCTs | Clinical outcomes | Heart failure | Workflow and adherence |

The table is a reminder that the cases answer different questions. The Apple Heart Study estimates a notification rate and a confirmation rate in a pragmatic population; the ECG app estimates a classification accuracy in a controlled study. The two are not comparable, and neither is a validation of the other.

The comparison also shows the common design features: a reference measured close in time, a population described, and a limitation stated. Those features are what make the cases useful as models.

### 9.7 What a reader should take from the cases

A reader should take three lessons.

- **The PPV is a property of the population.** The Apple Heart Study's 34% confirmation rate is not a property of the sensor alone; it reflects the notification prevalence and the confirmation pathway.
- **The reference timing matters.** A patch applied 13 days after the notification measures a different time window from a simultaneous ECG.
- **The reporting is part of the evidence.** The studies that are useful are the ones that describe the population, the reference, and the limitations.

The inference is that a validation is evaluated on its design and its reporting, not only on its headline number.

### 9.8 The gap between accuracy and outcome

The telemonitoring review is the clearest example of a technology that can measure accurately and fail to improve outcomes. The sensor is not the intervention; the care pathway is.

A validation that stops at accuracy has validated a measurement. A validation that stops at association has validated a biomarker. A validation that reaches an outcome has validated an intervention. The three are different claims, and the evidence is cumulative.

A product team should decide which claim it is making before it designs the study. The claim determines the reference, the population, the sample size, and the cost. A team that designs a measurement study and makes an intervention claim will fail the review, and the failure is predictable from the first protocol.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Prevalence mismatch | PPV from a case-control study | Different prior | Inflated PPV | Adjust to deployment |
| Spectrum bias | Healthy controls only | Narrow spectrum | Optimistic accuracy | Target population |
| Verification bias | Only some get reference | Selection by index | Biased accuracy | Random subset |
| Incorporation bias | Reference uses index | Circular | Inflated accuracy | Independent reference |
| Review bias | Reader sees index | Expectation | Inflated accuracy | Blinding |
| Indeterminate excluded | High inconclusive rate | Third outcome ignored | Inflated performance | Report as outcome |
| Missing data ignored | Complete-case analysis | Informative missingness | Biased estimate | Sensitivity analysis |
| Sample too small | Wide CI | Few cases | Inconclusive | Size for cases |
| Multiple metrics | Best metric reported | Multiplicity | Inflated type I | Pre-specify |
| Threshold tuned | Optimistic operating point | Test-set tuning | Invalid claim | Validation threshold |
| No external test | Single site | Distribution | Poor transport | External validation |
| No utility | Accuracy only | Association | No decision impact | Decision curve |
| Registration after analysis | Post-hoc protocol | Bias | Unreliable | Pre-register |
| Flow diagram missing | Exclusions unclear | Selection | Unclear sample | STARD flow |
| Intention-to-diagnose ignored | Per-protocol only | Excluding failures | Optimistic | Report both |
| Reference delay | Condition changes | Time gap | Misclassification | Short interval |
| Operator effect | Different users | Skill | Variable accuracy | Training; stratify |
| Learning curve | Early cases worse | Experience | Underestimated | Warm-up period |
| Batch effect | Site difference | Technical | Biased | Stratify by batch |
| Composite reference unstated | Assumed equivalence | Different errors | Unclear truth | State assumption |
| Latent class assumption | No gold standard | Model dependence | Model-dependent result | Sensitivity analysis |
| Reference reader fatigue | Long reading sessions | Error | Misclassification | Reading limits |
| Case mix shift | Different comorbidity | Spectrum | Non-comparable | Stratify |
| Threshold drift | Cut-point changes | Implementation | Non-comparable | Lock threshold |

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test |
|---|---|---|---|
| The reference is independent | Unbiased accuracy | Shared sensor or algorithm | Review the reference |
| The reference is correct | Anchor | Reference error | Adjudication; multiple readers |
| The spectrum matches deployment | Transport | Different severity | Stratify; external test |
| The prevalence is known | PPV | Unknown prior | Report a range |
| Sensitivity transfers across prevalence | Adjustment | Spectrum change | Subgroup analysis |
| Indeterminate results are reported | Complete picture | Exclusion | Third outcome |
| Missingness is random | Complete-case | Informative dropout | Sensitivity analysis |
| The sample is consecutive | Representative | Convenience | Document selection |
| The threshold is pre-specified | Valid operating point | Post-hoc tuning | Protocol |
| The metrics are co-primary | Joint claim | Selective reporting | Pre-specify |
| The study is registered | Transparency | Post-hoc | Registration date |
| The report follows STARD | Evaluability | Missing items | Checklist |
| The operator is trained | Accuracy | Skill variation | Training record |
| The learning curve is handled | Stable accuracy | Early cases | Warm-up analysis |
| The batch is recorded | Reproducibility | Technical variation | Batch metadata |
| The composite reference is justified | Truth | Different errors | Assumption stated |
| The intention-to-diagnose analysis is reported | Complete picture | Per-protocol only | Both analyses |
| The reference delay is short | Correct classification | Condition change | Time interval |
| The reader load is managed | Reference quality | Fatigue | Reading schedule |
| The case mix is described | Transport | Different comorbidity | Case-mix table |
| The threshold is locked | Comparability | Implementation drift | Version control |

## 12. Exercises

### Exercise 1. 2×2 metrics (application, 25 min)

**Deliverable:** from TP 80, FP 40, FN 20, TN 860, compute sensitivity, specificity, PPV, NPV, and likelihood ratios.

**Acceptance criteria:** all metrics; units; the prevalence stated.

**Time:** 25 minutes.

### Exercise 2. Prevalence adjustment (application, 30 min)

**Deliverable:** adjust the PPV and NPV from Exercise 1 to a prevalence of 1% and 20%.

**Acceptance criteria:** Bayes applied; both prevalences; the interpretation.

**Time:** 30 minutes.

### Exercise 3. Confidence intervals (application, 30 min)

**Deliverable:** compute Wilson intervals for 80/100 and 860/900, and compare with the normal approximation.

**Acceptance criteria:** both methods; the difference at the extremes explained.

**Time:** 30 minutes.

### Exercise 4. Reference standard critique (critique, 30 min)

**Deliverable:** evaluate three possible references for an AF test and choose one, with the independence and timing justified.

**Acceptance criteria:** independence and timing addressed; the error of the chosen reference stated.

**Time:** 30 minutes.

### Exercise 5. Sample size (application, 25 min)

**Deliverable:** compute the number of cases needed for a sensitivity of 0.90 with a half-width of 0.05 and 0.03.

**Acceptance criteria:** formula; both numbers; the total sample at 1% prevalence.

**Time:** 25 minutes.

### Exercise 6. Co-primary endpoints (design, 35 min)

**Deliverable:** define co-primary endpoints for a screening test and an alpha allocation.

**Acceptance criteria:** both endpoints with thresholds; joint evaluation; no post-hoc selection.

**Time:** 35 minutes.

### Exercise 7. STARD audit (critique, 45 min)

**Deliverable:** audit a published validation against the 30 STARD items and list the missing items.

**Acceptance criteria:** at least five missing items; each with a consequence; no invented item.

**Time:** 45 minutes.

### Exercise 8. Validation plan (design, 60 min)

**Deliverable:** a one-page validation plan for a wearable AF detector, including population, reference, sample size, co-primary endpoints, indeterminate handling, and prevalence adjustment.

**Acceptance criteria:** all components; pre-specified; a flow diagram described.

**Time:** 60 minutes.

### Exercise 9. Prevalence table (application, 20 min)

**Deliverable:** compute PPV and NPV for prevalence 0.1%, 1%, 5%, and 10% at sensitivity 0.90 and specificity 0.95, and identify the prevalence at which the PPV exceeds 0.50.

**Acceptance criteria:** all eight values; the threshold identified; the interpretation for a screening programme.

**Time:** 20 minutes.

### Submission standard

A validation plan must name the target condition, index test, reference, population, sampling, blinding, sample size, co-primary endpoints, indeterminate handling, and prevalence adjustment. A 2×2 analysis must include the table and the confidence intervals. A critique must name the bias and its direction.

A plan that reports only sensitivity and specificity is incomplete. A plan that reports a PPV without the prevalence is incomplete. A plan that does not state the indeterminate handling is incomplete.

## 13. Self-check answers

**1. What is sensitivity?** TP/(TP+FN).

**2. What is specificity?** TN/(TN+FP).

**3. Compute sensitivity for TP 80, FN 20.** 80/100 = 0.80.

**4. Compute specificity for TN 860, FP 40.** 860/900 = 0.956.

**5. Compute PPV for TP 80, FP 40.** 80/120 = 0.667.

**6. Compute NPV for TN 860, FN 20.** 860/880 = 0.977.

**7. What is the positive likelihood ratio for sensitivity 0.80 and specificity 0.956?** 0.80/(1-0.956) = 18.2.

**8. What is the negative likelihood ratio for the same values?** (1-0.80)/0.956 = 0.209.

**9. Adjust the PPV to 1% prevalence for sensitivity 0.80 and specificity 0.956.** TP rate = 0.008; FP rate = 0.99 × 0.044 = 0.04356; PPV = 0.008/0.05156 = 0.155.

**10. Adjust the NPV to 1% prevalence.** FN rate = 0.002; TN rate = 0.99 × 0.956 = 0.94644; NPV = 0.94644/0.94844 = 0.998.

**11. Why does the PPV fall when prevalence falls?** The denominator includes more false positives relative to true positives.

**12. What is spectrum bias?** A change in sensitivity or specificity when the disease severity or comorbidity spectrum changes.

**13. What is verification bias?** The reference is applied only to a subset selected by the index result.

**14. What is incorporation bias?** The reference standard includes the index test.

**15. Why is an indeterminate result not a negative?** It is a third outcome; counting it as negative biases sensitivity and specificity.

**16. Why is a case-control sample a problem for PPV?** The prevalence is artificial, so the PPV does not reflect deployment.

**17. Compute the PPV at 5% prevalence for sensitivity 0.95 and specificity 0.97.** TP rate 0.0475, FP rate 0.95 × 0.03 = 0.0285, PPV = 0.0475/0.076 = 0.625.

**18. Why is the number of cases, not the total sample, the driver of sensitivity precision?** The sensitivity denominator is the number of cases. Adding controls increases the specificity precision but not the sensitivity precision.

**19. What is the direction of verification bias if the reference is applied only to index-positive participants?** Sensitivity is overestimated, because the index-negative participants are assumed to be true negatives without verification.

**20. What is the direction of review bias?** Accuracy is overestimated, because the reader is more likely to confirm a positive index result.

**21. What is the post-test odds for a positive result at 1% prevalence and LR+ 31.7?** Pre-test odds 0.0101 × 31.7 = 0.320, which is a probability of 24.2%.

**22. Why is a likelihood ratio useful across populations?** It is independent of prevalence, so it can be combined with any prior.

**23. What is the difference between a consensus and an adjudicated reference?** A consensus is the readers' agreement; an adjudication resolves disagreement with a third reader or a panel.

**Rubric for Exercise 8 (validation plan).** A strong plan specifies the population, reference, sample size, co-primary endpoints, indeterminate handling, and prevalence adjustment. A weak plan reports only sensitivity and specificity.

**Rubric for Exercise 7 (STARD audit).** A strong audit names at least five missing items and the consequence of each. A weak audit says the study is "poorly reported."

## 14. Interview questions

**1. How do you design a diagnostic accuracy study?**
A strong answer names the target condition, index test, reference, population, sampling, blinding, and analysis. A weak answer says "compare with a gold standard."

**2. Why is PPV prevalence-dependent?**
A strong answer uses Bayes and the false-positive denominator. A weak answer says "it depends on the population."

**3. How do you choose a reference standard?**
A strong answer requires independence, appropriate timing, and a stated error. A weak answer picks the most expensive test.

**4. What is spectrum bias?**
A strong answer explains the change in accuracy with disease severity and comorbidity. A weak answer says "the sample is different."

**5. How do you handle indeterminate results?**
A strong answer reports them as a third outcome and analyses them. A weak answer excludes them.

**6. What are co-primary endpoints?**
A strong answer defines joint endpoints with pre-specified thresholds and alpha. A weak answer says "two outcomes."

**7. Why is a case-control design a problem?**
A strong answer explains the artificial prevalence and the spectrum. A weak answer says it is less rigorous.

**8. What is verification bias?**
A strong answer explains the selective reference application and the direction of the bias. A weak answer confuses it with missing data.

**9. How do you size a diagnostic study?**
A strong answer sizes for the number of cases and controls, not the total. A weak answer uses a generic rule.

**10. What does STARD require?**
A strong answer names the flow diagram, 2×2 table, confidence intervals, and registration. A weak answer says "a checklist."

**11. Why is an external validation important?**
A strong answer explains the difference between internal and external transport. A weak answer says it is better.

**12. How do you validate a screening test in a low-prevalence population?**
A strong answer uses a large sample, a confirmatory reference, and a prevalence-adjusted PPV. A weak answer reports the study PPV.

**13. What is the difference between a per-protocol and an intention-to-diagnose analysis?**
A strong answer explains that per-protocol excludes indeterminate cases and intention-to-diagnose includes them, and that the second is conservative. A weak answer treats them as equivalent.

**14. Why is the reference timing part of the design?**
A strong answer explains that a delay allows the condition to change and biases the classification. A weak answer says the reference should be "recent."

**15. What is the difference between a composite reference and a latent class model?**
A strong answer explains that the composite combines references and the latent class models the error, and that both require an assumption. A weak answer says they are both "without a gold standard."

**16. How do you report an indeterminate result in the primary analysis?**
A strong answer includes it as a third outcome or counts it as a failure under a prespecified rule, with a sensitivity analysis. A weak answer excludes it.

**17. What is the first thing you check in a published validation?**
A strong answer checks the population, the reference, the prevalence, and the indeterminate rate before the headline metric. A weak answer checks the sensitivity.

## 15. Key terms

- **Validation** — evidence for a claim.
- **Target condition** — the state to detect.
- **Index test** — the test being evaluated.
- **Reference standard** — the best available truth.
- **Intended role** — screening, diagnosis, monitoring.
- **Spectrum** — range of disease severity.
- **Spectrum bias** — accuracy change with spectrum.
- **Prevalence** — proportion with the condition.
- **Sensitivity** — TP/(TP+FN).
- **Specificity** — TN/(TN+FP).
- **PPV** — TP/(TP+FP).
- **NPV** — TN/(TN+FN).
- **Likelihood ratio** — ratio of probabilities.
- **Co-primary endpoint** — joint endpoints.
- **Type I error** — false positive rate.
- **Power** — probability of detection.
- **Indeterminate** — no result.
- **Verification bias** — selective reference.
- **Incorporation bias** — reference includes index.
- **Review bias** — unblinded reader.
- **Missing data** — absent measurements.
- **Informative missingness** — related to outcome.
- **STARD** — reporting guideline.
- **STARD-AI** — AI extension.
- **Flow diagram** — participant flow.
- **External validation** — new population.
- **Pre-registration** — protocol before study.
- **Per-protocol** — classifiable cases only.
- **Intention-to-diagnose** — all participants.
- **Complete-case** — observed cases only.
- **Worst-case bound** — missing assigned to extremes.
- **Composite reference** — combination of references.
- **Latent class** — model without a gold standard.
- **Two-stage sampling** — reference on a subset.
- **Operator effect** — variation by user.
- **Learning curve** — change with experience.
- **Batch effect** — technical variation.
- **Spectrum bias** — accuracy change with spectrum.
- **Verification bias** — selective reference.
- **Incorporation bias** — reference includes index.
- **Review bias** — unblinded reader.
- **Reference delay** — time between tests.
- **Composite reference** — combination of references.
- **Latent class** — model without a gold standard.
- **Two-stage design** — reference on a subset.
- **Intention-to-diagnose** — all participants.
- **Per-protocol** — classifiable only.
- **Worst-case bound** — missing at extremes.
- **Likelihood ratio** — prevalence-independent evidence.
- **Pre-test odds** — prior odds.
- **Post-test odds** — posterior odds.
- **Flow diagram** — participant accounting.
- **Registration** — protocol record.
- **Protocol deviation** — departure from plan.
- **Adjudication** — independent review.
- **Reference reader** — person applying reference.
- **Index reader** — person applying index test.
- **Consensus reference** — agreement of readers.
- **Disagreement rate** — reader variability.
- **Case mix** — distribution of conditions.
- **Comorbidity** — other conditions.
- **Reading load** — number of cases per reader.
- **Threshold lock** — fixed cut-point.
- **Protocol** — pre-specified plan.
- **Deviation** — departure from protocol.
- **Audit trail** — record of changes.
- **Source data** — original records.
- **Query** — data clarification.
- **Range** — interval of plausible values.

## 16. Sources

All URLs accessed 25 October 2026.

1. Bossuyt PM, Reitsma JB, Bruns DE, et al. STARD 2015: an updated list of essential items for reporting diagnostic accuracy studies. *BMJ*. 2015;351:h5527. https://www.bmj.com/content/351/bmj.h5527
2. EQUATOR Network. STARD 2015. https://www.equator-network.org/reporting-guidelines/stard/
3. Cohen JF, Korevaar DA, Altman DG, et al. STARD 2015 guidelines for reporting diagnostic accuracy studies: explanation and elaboration. *BMJ Open*. 2016;6:e012799. https://bmjopen.bmj.com/content/6/11/e012799
4. Perez MV, Mahaffey KW, Hedlin H, et al. Large-scale assessment of a smartwatch to identify atrial fibrillation. *N Engl J Med*. 2019;381:1909-1917. https://doi.org/10.1056/NEJMoa1901183
5. Stanford Medicine. Apple Heart Study results. https://med.stanford.edu/news/all-news/2019/11/through-apple-heart-study--stanford-medicine-researchers-show-we.html
6. US Food and Drug Administration. DEN180044, Apple ECG App. 2018. https://www.accessdata.fda.gov/cdrh_docs/reviews/DEN180044.pdf
7. US Food and Drug Administration. Arrhythmia Detector and Alarm — Class II Special Controls Guidance. 2003. https://www.fda.gov/medical-devices/guidance-documents-medical-devices-and-radiation-emitting-products/arrhythmia-detector-and-alarm-class-ii-special-controls-guidance-document-industry-and-fda-staff
8. Chinoy ED, et al. Performance of seven consumer sleep-tracking devices compared with polysomnography. *Sleep*. 2021;44(5):zsaa291. https://doi.org/10.1093/sleep/zsaa291
9. Inglis SC, et al. Structured telephone support or non-invasive telemonitoring for patients with heart failure. *Cochrane Database of Systematic Reviews*. 2015. https://doi.org/10.1002/14651858.CD007228.pub3
10. ISO 14155, Clinical investigation of medical devices for human subjects. https://www.iso.org/
11. World Medical Association. Declaration of Helsinki. https://www.wma.net/
12. STARD-AI. Reporting guideline for AI diagnostic studies. https://www.equator-network.org/reporting-guidelines/stard/
13. CONSORT 2010. Reporting of randomised trials. https://www.consort-statement.org/
14. PRISMA 2020. Reporting of systematic reviews. https://www.prisma-statement.org/
15. US Food and Drug Administration. Device software functions guidance. https://www.fda.gov/medical-devices/digital-health-center-excellence/device-software-functions-including-mobile-medical-applications
16. ClinicalTrials.gov. Apple Heart Study, NCT03335800. https://clinicaltrials.gov/study/NCT03335800

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`08_digital-biomarkers.md`](08_digital-biomarkers.md)
- Next topic: [`10_deployment-on-device.md`](10_deployment-on-device.md)
- Related topic: [`02_ecg.md`](02_ecg.md)
- Module survey file: [`../../11_physiological_signals_wearables.md`](../../11_physiological_signals_wearables.md)
- Lab: [`../../labs/`](../../labs/)
- Reporting guideline: [`https://www.equator-network.org/reporting-guidelines/stard/`](https://www.equator-network.org/reporting-guidelines/stard/)
- Related topic: [`04_ppg-and-wearables.md`](04_ppg-and-wearables.md)
- Dataset: [`https://physionet.org/content/challenge-2020/`](https://physionet.org/content/challenge-2020/)
- Guideline: [`https://www.bmj.com/content/351/bmj.h5527`](https://www.bmj.com/content/351/bmj.h5527)
- Related topic: [`07_deep-learning-for-signals.md`](07_deep-learning-for-signals.md)
- Related topic: [`10_deployment-on-device.md`](10_deployment-on-device.md)
