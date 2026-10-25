# Evaluation

> **Module 10 — Medical Imaging and Pathology AI**
> Topic 07 of 10
> Estimated study time: 4 hours
> Prerequisites: Topics 04, 05, and 06
> Version 2026.3

## 1. Why this topic exists

An imaging model can have an excellent area under the receiver operating characteristic curve
(AUROC) and be clinically useless. It can have a high Dice score and miss the lesions that matter.
It can have a good sensitivity at one operating point and an unacceptable false-positive rate at
the point the workflow can tolerate. It can be well calibrated on a case-control sample and
miscalibrated at deployment prevalence.

The industry problem is that evaluation is often reduced to a single number, reported without its
unit, its reference standard, its operating point, or its uncertainty. That number then travels
into a slide deck, a regulatory summary, and a sales conversation, where it acquires a meaning it
never had. The model is compared to a competitor's differently measured number, and the comparison
is meaningless.

Evaluation is a measurement design. It starts with the estimand, the quantity the product claims
to measure, and ends with a decision rule. Between those ends are the unit of analysis, the
reference standard, the operating point, the metric, the confidence interval, the validation
design, the subgroup analysis, and the reader study. Each choice changes the number.

This topic teaches the design. It covers classification, detection, segmentation, triage, and
reader-study endpoints; the difference between image-level and lesion-level metrics; the
construction of confidence intervals; the handling of prevalence and spectrum; and the reporting
standards that a reviewer will apply.

### 1.1 The decisions this topic unblocks

| Decision | What evaluation knowledge provides |
|---|---|
| Primary endpoint | The metric that matches the clinical claim |
| Unit of analysis | Patient, study, lesion, or voxel |
| Operating point | The threshold or FP rate the workflow tolerates |
| Uncertainty | Confidence interval and sample size |
| Validation design | Internal, external, temporal, or prospective |
| Subgroup analysis | Whether performance is consistent |
| Reader study | Whether the model improves human performance |
| Go/no-go | Whether the evidence supports the claim |

### 1.2 A number without a unit is not a result

"AUROC 0.91" is not a result. "AUROC 0.91 (95% CI 0.88-0.94) for pneumonia on 2,000 chest
radiographs, reference standard three-reader consensus, site-held-out test" is a result. The unit,
the reference, and the design are part of the measurement, not decoration.

The same discipline applies to detection: "sensitivity 0.90 at 0.5 false positives per scan" is a
result; "sensitivity 0.90" is not, because the false-positive rate could be 0.5 or 50 per scan.
For segmentation: "per-lesion Dice 0.82 (IQR 0.74-0.88) on 220 lesions" is a result; "Dice 0.85"
is a pooled voxel number that may be dominated by large lesions.

## 2. Learning outcomes

By the end of this topic you can:

1. **Define** an estimand with a unit, a reference standard, and a time horizon.
2. **Choose** a primary metric that matches the clinical claim and the task type.
3. **Compute** sensitivity, specificity, PPV, NPV, AUROC, AUPRC, Dice, and Hausdorff distance from
   raw counts or masks.
4. **Construct** confidence intervals with the Wilson method, bootstrap, or DeLong method, and
   state which is appropriate.
5. **Distinguish** image-level from lesion-level performance and explain when the difference
   matters.
6. **Design** internal, site-held-out, temporal, and prospective validation.
7. **Interpret** a reader study with a multi-reader multi-case design and a non-inferiority or
   superiority claim.
8. **Report** results according to CLAIM and TRIPOD+AI, including calibration, subgroups, and
   error analysis.

## 3. The core mental model

Evaluation is a chain from the clinical claim to the decision. Each link constrains the next.

```text
Clinical claim
   │  (what the product does, for whom, in which workflow)
   ▼
Estimand
   │  (the quantity to estimate: sensitivity, PPV, time saved)
   ▼
Unit of analysis ──► Reference standard ──► Operating point
   │                      │                      │
   └──────────┬───────────┴──────────────────────┘
              ▼
           Metric ──► Uncertainty ──► Decision rule
```

### 3.1 The estimand determines everything downstream

An estimand is a precise statement of what is being estimated. For a detection model, it might be:
"the proportion of solid lung nodules of 4 mm or more, per lesion, detected at a threshold that
yields 0.5 false positives per scan, in a screening population, with three-reader consensus as the
reference."

That statement fixes the unit (lesion), the reference (consensus), the operating point (0.5
FP/scan), the population (screening), and the threshold. Change any one and the number changes.

### 3.2 The unit hierarchy

| Unit | Numerator | Denominator | Use |
|---|---|---|---|
| Patient | Patients with a detected finding | Patients at risk | Screening, diagnosis |
| Study | Studies with a detected finding | Studies | Triage, worklist |
| Series | Series with a detected finding | Series | Protocol-specific |
| Lesion | Lesions detected | Reference lesions | Detection sensitivity |
| Voxel | Overlapping voxels | Reference plus predicted voxels | Segmentation overlap |
| Pixel | Overlapping pixels | Reference plus predicted pixels | 2D segmentation |

A study-level sensitivity of 0.94 and a lesion-level sensitivity of 0.90 can describe the same
model: a study is positive if any lesion is found, so multiple lesions per study make the study
metric look better. The unit must be stated.

### 3.3 The operating point

Most imaging models output a score, not a decision. The operating point is the threshold that
converts the score into a decision. It is chosen from the workflow: the false-positive rate the
radiologist will tolerate, the sensitivity the clinical pathway requires, or the alert volume the
IT system can sustain.

A model does not have one sensitivity. It has a curve. Reporting a point on the curve without the
false-positive rate is incomplete; reporting the curve without the chosen point is not a product
claim.

### 3.4 Discrimination, calibration, and clinical utility

Three different questions:

- **Discrimination**: can the model rank a positive above a negative? Measured by AUROC and
  AUPRC.
- **Calibration**: does a predicted probability of 0.2 mean 20%? Measured by calibration curves,
  Brier score, and expected calibration error.
- **Clinical utility**: does using the model improve a decision? Measured by decision curve
  analysis, net benefit, or a workflow outcome.

A model can discriminate well and be miscalibrated, or be calibrated and have no net benefit at
the relevant threshold. All three are needed for a clinical claim.

### 3.5 Why pooled metrics mislead

Pooling voxels across lesions weights large lesions more. Pooling studies across sites weights
large sites more. Pooling patients across prevalences weights the prevalent group more. Each pool
answers a different question from the one the product claims. The fix is to report the metric at
the claimed unit, with the distribution of the underlying units, and to show the subgroup results.

### 3.6 Evaluation is a decision problem

A metric becomes useful only when it is connected to a decision. The decision has three parts: who
acts, on what information, and with what consequence. A triage model's decision is "review this
study first"; the consequence is a delay for the studies not prioritized. A rule-out model's
decision is "do not image"; the consequence is a missed diagnosis.

The connection is a decision rule: if the score is above a threshold, take an action. The value of
the model is the difference in expected consequence between the rule and the current practice.
That difference is what a decision-curve analysis estimates, and it is what a workflow study
measures. A metric without a decision rule is a description, not an evaluation.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Estimand | Precise quantity to estimate | Confusing with metric | Statement |
| AUROC | Area under the ROC curve | Treating as clinical utility | Probability |
| AUPRC | Area under precision-recall curve | Using with balanced data only | Probability |
| Sensitivity | True positives / reference positives | Confusing with PPV | Proportion |
| Specificity | True negatives / reference negatives | Confusing with NPV | Proportion |
| PPV | True positives / predicted positives | Assuming prevalence-free | Proportion |
| NPV | True negatives / predicted negatives | Assuming prevalence-free | Proportion |
| Accuracy | Correct / total | Using with imbalance | Proportion |
| F1 | Harmonic mean of precision and recall | Using with rare findings | Dimensionless |
| Calibration | Agreement between predicted and observed risk | Confusing with discrimination | Curve or score |
| Brier score | Mean squared error of probabilities | Using alone | Probability² |
| ECE | Expected calibration error | Sensitive to binning | Proportion |
| Decision curve | Net benefit across thresholds | Ignoring | Utility |
| Dice | Voxel overlap | Pooling across lesions | Voxel |
| IoU | Intersection over union | Confusing with Dice | Voxel |
| Hausdorff distance | Maximum boundary distance | Using as average | mm |
| ASSD | Average symmetric surface distance | Confusing with Hausdorff | mm |
| FROC | Free-response ROC | Confusing with ROC | Curve |
| Sensitivity at FP/scan | Detection operating point | Omitting FP rate | Proportion at count |
| mAP | Mean average precision | Using without operating point | Dimensionless |
| Localization accuracy | Correct location within a tolerance | Confusing with detection | Proportion |
| Bootstrap | Resampling for uncertainty | Using for small n blindly | Method |
| Wilson interval | Score interval for a proportion | Using normal approximation | Method |
| DeLong | AUC variance estimator | Using for paired comparisons | Method |
| MRMC | Multi-reader multi-case study | Confusing with a single-reader study | Design |
| Reader study | Human performance study | Confusing with standalone test | Design |
| Non-inferiority | Not worse by more than a margin | Confusing with superiority | Design |
| Prevalence | Reference positives / total | Using enriched prevalence | Proportion |
| Spectrum | Distribution of disease severity | Ignoring | Distribution |
| Subgroup | Defined population subset | Reporting without counts | Group |
| Internal validation | Held-out data from the same source | Calling it external | Design |
| External validation | Data from a different source | Calling it independent if shared | Design |
| Temporal validation | Later time period | Random split by date | Design |
| Prospective | Collected after model freeze | Calling retrospective prospective | Design |
| Silent mode | Running without clinical use | Calling it a trial | Design |
| Net benefit | Decision-curve utility | Confusing with accuracy | Utility |
| Number needed to alert | Alerts per true positive | Ignoring | Count |

### 4.1 Metric by task

| Task | Primary | Secondary | Unit | Reference |
|---|---|---|---|---|
| Binary classification | AUROC | Sensitivity at fixed specificity, calibration | Study | Expert or outcome |
| Multi-label | Per-label AUROC, AUPRC | Calibration per label | Study | Report or expert |
| Detection | Sensitivity at FP/scan | mAP, localization | Lesion | Expert consensus |
| Segmentation | Per-lesion Dice | Hausdorff, ASSD | Voxel | Expert mask |
| Triage | Time to review | Missed critical findings | Study | Workflow log |
| Diagnosis | Sensitivity, specificity | PPV, NPV at prevalence | Patient | Pathology |
| Quantification | MAE, ICC | Bland-Altman limits | Lesion | Expert measurement |

### 4.2 Confidence interval methods

| Method | Use | Assumption | Limitation |
|---|---|---|---|
| Normal approximation | Large n, proportion near 0.5 | Normality | Fails near 0 or 1 |
| Wilson | Proportions, moderate n | Approximate binomial | Better than normal |
| Clopper-Pearson | Exact proportion interval | Binomial | Conservative |
| Bootstrap percentile | Complex metrics | Exchangeability | Needs enough resamples |
| Bootstrap BCa | Skewed metrics | Exchangeability | More computation |
| DeLong | AUC | Independent or paired cases | AUC only |
| Cluster bootstrap | Patient-clustered data | Exchangeability of clusters | Needs cluster labels |

## 5. The mechanics, step by step

This procedure designs and executes an evaluation. It assumes the data passed ingestion, label,
and leakage checks. Each step states input, output, rationale, branches, and edge cases.

### Step 1. Write the estimand

**Input:** The clinical claim and the workflow.

**Output:** One paragraph with the population, unit, reference standard, operating point, time
horizon, and metric.

**Rationale:** The estimand is the contract. Without it, every later choice is ad hoc and the
result cannot be compared to anything. The estimand must be written before the test set is
opened.

**Branch points:**
- Screening: patient-level, high sensitivity, low FP rate.
- Diagnosis: lesion or patient, balance sensitivity and specificity.
- Triage: time to review, missed critical findings.
- Quantification: agreement with expert measurement.

**Edge case:** A claim may have two estimands, for example detection and malignancy
classification. Report both; do not average them.

### Step 2. Fix the unit of analysis

**Input:** The estimand.

**Output:** The unit (patient, study, series, lesion, voxel) and the clustering structure.

**Rationale:** The unit determines the denominator, the independence assumption, and the
confidence interval. Multiple lesions per patient are clustered; ignoring the clustering
underestimates the interval.

**Branch points:**
- One lesion per patient: patient and lesion units coincide.
- Multiple lesions per patient: lesion-level sensitivity plus patient-level clustering.
- Multiple studies per patient: patient-level split and cluster bootstrap.

**Edge case:** A patient with no reference lesion is a negative at the patient level but is not
part of the lesion-level denominator. The two denominators differ and must be reported.

### Step 3. Verify the reference standard

**Input:** The label table and the reference documentation.

**Output:** A statement of the reference standard, its independence, and its verification rate.

**Rationale:** The reference standard bounds the achievable performance. If the reference is a
report, the ceiling is the report's image-level accuracy. If the reference is pathology, the
ceiling is the sampling accuracy.

**Branch points:**
- Expert consensus: strong for detection, reader-dependent.
- Pathology: strong for diagnosis, sampling error.
- Follow-up: strong for outcome, loss to follow-up.
- Registry: weak for diagnosis, useful for prevalence.

**Edge case:** A negative without follow-up is unverified, not proven negative. Report the
verification rate and consider a sensitivity analysis.

### Step 4. Choose the operating point

**Input:** The workflow constraint and the score distribution.

**Output:** A threshold or FP rate, chosen on validation data, with the rationale.

**Rationale:** The operating point converts a score into a decision. Choosing it on the test set
inflates performance. Choosing it on the validation set and locking it before the test is the
minimum discipline.

**Branch points:**
- Fixed sensitivity: for a rule-out pathway.
- Fixed specificity: for a confirmatory pathway.
- Fixed FP/scan: for a concurrent-read alert.
- Fixed alert volume: for IT capacity.

**Edge case:** A threshold chosen at one prevalence may not transfer. Recalibrate at deployment
prevalence and re-evaluate.

### Step 5. Select the primary metric

**Input:** The estimand and the operating point.

**Output:** One primary metric and a small set of secondary metrics.

**Rationale:** Multiple metrics invite selective reporting. One primary metric, pre-specified,
with secondary metrics for context, is the standard. The primary metric must match the claim.

**Branch points:**
- Discrimination claim: AUROC.
- Rule-out claim: sensitivity at a fixed specificity.
- Detection claim: sensitivity at FP/scan.
- Segmentation claim: per-lesion Dice.
- Triage claim: time to review.

**Edge case:** AUROC is prevalence-invariant but not clinical-utility-invariant. A high AUROC can
coexist with poor PPV at low prevalence.

### Step 6. Compute the metric with its uncertainty

**Input:** The predictions and the reference.

**Output:** The point estimate, the confidence interval, and the method.

**Rationale:** A point estimate without an interval cannot support a comparison. The interval
method must match the metric and the clustering.

**Branch points:**
- Proportion: Wilson or Clopper-Pearson.
- AUC: DeLong or bootstrap.
- Dice: bootstrap over lesions.
- Clustered data: cluster bootstrap over patients.
- Paired models: paired bootstrap or DeLong.

**Edge case:** A bootstrap over lesions ignores patient clustering and produces intervals that
are too narrow. Resample patients, not lesions, when lesions are clustered.

### Step 7. Evaluate internal validation

**Input:** The held-out split from the same source.

**Output:** Internal performance with the split design documented.

**Rationale:** Internal validation estimates performance on the same distribution. It is necessary
and not sufficient. A random split by image is not internal validation if it leaks patients.

**Branch points:**
- Patient-level holdout: standard.
- Cross-validation: useful for small data, with patient-level folds.
- Nested cross-validation: for hyperparameter tuning.

**Edge case:** A single holdout can be unlucky. Report the split seed and consider repeated
holdout or cross-validation.

### Step 8. Evaluate external validation

**Input:** Data from a different site, scanner, or time period.

**Output:** External performance with the site, scanner, and prevalence described.

**Rationale:** External validation tests transportability. It is the evidence a customer will
ask for, and it is where shortcuts fail.

**Branch points:**
- Site-held-out: different institution.
- Scanner-held-out: different vendor or model.
- Temporal: later period.
- Prospective: collected after model freeze.

**Edge case:** A site-held-out test can still share patients, protocols, or annotation teams. The
independence must be described, not assumed.

### Step 9. Analyse subgroups

**Input:** The predictions and the subgroup fields.

**Output:** A table of performance by subgroup with counts and intervals.

**Rationale:** Aggregate performance can hide a subgroup failure. The FDA expects subgroup
analyses by anatomy, demographics, medical history, user experience, and imaging equipment.

**Branch points:**
- Pre-specified subgroups: primary.
- Exploratory subgroups: labelled as exploratory.
- Small subgroups: descriptive only.

**Edge case:** Many subgroups invite false positives. Pre-specify a small number and correct for
multiplicity, or label the analysis as exploratory.

### Step 10. Assess calibration

**Input:** Predicted probabilities and observed outcomes.

**Output:** Calibration curve, calibration slope and intercept, Brier score, and a decision-curve
analysis.

**Rationale:** A probability is only useful if it means what it says. Calibration is separate from
discrimination and is required for any claim that uses a probability threshold.

**Branch points:**
- Well calibrated: proceed.
- Miscalibrated: recalibrate on a validation set.
- Calibration varies by subgroup: report and recalibrate per subgroup.

**Edge case:** Calibration on a case-control sample is not deployment calibration. Reweight or
recalibrate at the deployment prevalence.

### Step 11. Design the reader study

**Input:** The claim, the reader population, and the case sample.

**Output:** A multi-reader multi-case (MRMC) design with the number of readers, cases, and the
analysis method.

**Rationale:** A standalone metric does not show that a radiologist using the model performs
better. A reader study does. The design must account for reader variability and case variability.

**Branch points:**
- Superiority: model-assisted vs unaided.
- Non-inferiority: model-assisted not worse by a margin.
- Assisted-read: model as a second reader.
- Sequential read: model after the human.

**Edge case:** A reader study with too few readers or cases has low power. Use published sample
size methods and pre-specify the analysis.

### Step 12. Analyse errors and failure modes

**Input:** False positives, false negatives, and segmentation failures.

**Output:** A stratified error analysis with examples and counts.

**Rationale:** Aggregate metrics do not explain failure. The error analysis identifies the
subgroups, lesion sizes, and acquisition conditions where the model fails. It is required by
CLAIM and expected by regulators.

**Branch points:**
- Size-stratified: small lesions vs large.
- Location-stratified: central vs peripheral.
- Acquisition-stratified: thickness, contrast, scanner.
- False-positive type: artefact, look-alike, label error.

**Edge case:** A single dramatic failure can dominate the narrative. Report the frequency, not
only the example.

### Step 13. Compare to baselines and alternatives

**Input:** The model, a simple baseline, and any comparator.

**Output:** A comparison table with the same data, split, unit, and metric.

**Rationale:** A model is only useful if it beats the current practice or a simpler alternative.
The comparison must be fair: same cases, same reference, same operating point.

**Branch points:**
- Baseline wins: do not deploy.
- Model wins within interval: inconclusive; collect more data.
- Model wins outside interval: material.

**Edge case:** A comparison to a published number from a different dataset is not a comparison.
Re-run the baseline on the same data.

### Step 14. Write the report

**Input:** All prior steps.

**Output:** A report following CLAIM and TRIPOD+AI, with the estimand, data, splits, model,
metrics, intervals, calibration, subgroups, errors, and limitations.

**Rationale:** The report is the evidence. A reviewer must be able to reproduce the analysis from
the report and the code. The report must state what was not done.

**Branch points:**
- Internal report: full detail.
- Regulatory summary: focus on the claim and the evidence.
- Publication: CLAIM checklist.

**Edge case:** A report that omits the split design or the operating point is not reproducible,
even if the metric is correct.

### 5.1 Paired comparisons and DeLong

When two models are evaluated on the same cases, their performance estimates are correlated. A
paired comparison uses that correlation to increase power. For AUROC, the DeLong method estimates
the variance of the difference using the covariance between the two AUCs; for other metrics, a
paired bootstrap resamples cases and evaluates both models on the same resample.

The error to avoid is treating the two estimates as independent. An unpaired confidence interval
for the difference is wider than necessary and can miss a real effect. The paired design is also
the correct one for a reader study, where the same readers and cases are used for both conditions.

**Edge case:** Pairing requires the same cases and the same reference. If one model is evaluated
on a subset, the comparison is unpaired and the estimates are not directly comparable.

### 5.2 Sample size for a detection metric

For a lesion-level sensitivity, the number of reference lesions determines the interval width. At
a sensitivity of 0.90, a half-width of 0.05 requires about 139 lesions; a half-width of 0.03
requires about 384. The number of scans is a separate constraint: a study with 1.5 lesions per
scan needs about 93 scans for 139 lesions, before accounting for clustering.

Clustering inflates the required sample. If lesions within a patient are correlated with an
intraclass correlation of 0.2, the design effect is `1 + (m - 1) × ICC`, where `m` is the average
number of lesions per patient. For 1.5 lesions per patient, the design effect is 1.1, a modest
increase. For 5 lesions per patient, it is 1.8.

The sample size must be stated for the unit that carries the claim. A study powered for scans may
be underpowered for lesions.

### 5.3 Multiplicity and the cost of looking

Every additional metric, subgroup, and comparison increases the chance of a favourable result by
chance. A study that tests ten subgroups at the 0.05 level expects one false positive if the null
is true. The correction can be a Bonferroni adjustment, a false discovery rate procedure, or a
pre-specified hierarchy of hypotheses.

The practical rule is to pre-specify one primary metric, a small number of secondary metrics, and
a short list of subgroups. Everything else is exploratory and labelled as such. A report that
presents the most favourable of many analyses without correction is not evidence.

### 5.4 Reporting a curve and a point

A curve and a point are both necessary. The curve shows the achievable trade-off; the point shows
the operating decision. A report that gives only a curve does not say what the product will do. A
report that gives only a point does not show whether a better point exists.

The minimum is the curve with the chosen point marked, the threshold value, the sensitivity, the
false-positive rate, and the confidence interval at that point. The threshold is part of the
result because a different threshold produces a different product.

### 5.5 Evaluating a triage model

A triage model is not evaluated by AUROC alone. The endpoint is the time from study availability
to review for the target finding, and the harm is the delay for the studies that were
deprioritized. The evaluation needs the arrival distribution, the reading capacity, the alert
volume, and the downstream action.

The minimum retrospective analysis simulates the worklist with and without triage at several
thresholds and reports the median time to review for the target finding, the number of alerts per
day, and the number of target findings missed by the alert. A prospective study then measures the
same quantities in the real workflow.

### 5.6 Decision table: metric selection by claim

| Claim | Primary metric | Operating point | Unit | Pitfall |
|---|---|---|---|---|
| Rule out pneumothorax | Sensitivity | Fixed specificity 0.95 | Study | Missing small pneumothorax |
| Detect lung nodules | Sensitivity at FP/scan | 0.5 FP/scan | Lesion | Study-level inflation |
| Triage LVO | Time to review | Alert volume | Study | Detection without workflow |
| Segment tumour | Per-lesion Dice | Threshold 0.5 | Voxel | Pooled voxel weighting |
| Diagnose pneumonia | AUROC | Not applicable | Study | Prevalence-free metric |
| Quantify volume | MAE, ICC | Not applicable | Lesion | Registration bias |
| Predict response | AUROC, calibration | Threshold | Patient | Outcome ascertainment |

### 5.7 Edge cases that require a written rule

**Mixed slice thickness.** A detection metric must be stratified by thickness. A model trained on
1 mm and tested on 5 mm has a different partial-volume regime.

**Missing series.** An incomplete study is not a negative. Exclude and report the incomplete rate.

**Inconsistent spacing.** Size and volume metrics require the position-derived spacing. Recompute
before evaluating.

**Rescale absent.** A HU-based threshold cannot be evaluated without a unit. Exclude.

**Burned-in annotation.** A model may use the annotation; the evaluation must test with and
without.

**Laterality mismatch.** A side-specific metric requires a verified side.

**Multi-frame objects.** Frame order and geometry must be validated before evaluation.

**Private tags carrying labels.** Remove before evaluation; a model that reads them is not
evaluated on the image.

## 6. Worked example

### 6.1 The case

A lung nodule detector is evaluated on 200 low-dose CT scans from a held-out site. The reference
standard is three-reader consensus with adjudication. There are 220 solid nodules of 4 mm or more
in 160 positive scans; 40 scans have no nodule. The model outputs a score per candidate; the
operating point is 0.5 false positives per scan.

### 6.2 Lesion-level detection

At the operating point, the model detects 198 of 220 nodules:

```text
lesion sensitivity = 198 / 220 = 0.900
```

The false positives are 0.5 per scan × 200 scans = 100. The per-scan false-positive distribution
matters: 100 false positives spread over 200 scans means some scans have two or three, and the
alert burden is not uniform.

Wilson 95% confidence interval for 198/220:

```text
p = 0.900, n = 220, z = 1.96
denominator = 1 + z²/n = 1 + 3.8416/220 = 1.01746
centre = (0.900 + 3.8416/440) / 1.01746
       = (0.900 + 0.00873) / 1.01746
       = 0.8931
half-width = 1.96 × sqrt(0.900 × 0.100/220 + 3.8416/(4 × 220²)) / 1.01746
           = 1.96 × sqrt(0.0004091 + 0.0000198) / 1.01746
           = 1.96 × 0.02072 / 1.01746
           = 0.0399
CI = 0.853 to 0.933
```

### 6.3 Study-level detection

At the study level, 150 of 160 positive scans have at least one detected nodule:

```text
study sensitivity = 150 / 160 = 0.9375
```

Of the 40 negative scans, 30 are correctly called negative:

```text
study specificity = 30 / 40 = 0.750
```

The study-level sensitivity is 0.9375, higher than the lesion-level 0.900, because a scan with
two nodules counts once and a missed second nodule does not change the study label. The
specificity of 0.750 reflects the 10 negative scans with at least one false positive.

The two numbers answer different questions. A screening program that needs every nodule found
uses the lesion-level number; a worklist that flags scans uses the study-level number.

### 6.4 Segmentation

For the 198 detected nodules, the model mask is compared with the reference mask. For a
representative 9.4 mm nodule, the reference has 303 voxels at 1 mm, the prediction has 271, and
the intersection is 236:

```text
Dice = 2 × 236 / (303 + 271) = 472 / 574 = 0.822
```

Per-lesion Dice is reported as a median with an interquartile range, not a single pooled number.
A pooled voxel Dice weights the 30 mm lesions more than the 5 mm lesions, and the small lesions
are where the model fails.

### 6.5 Discrimination and calibration

The model's score has AUROC 0.88 (95% CI 0.84-0.91) for the presence of any nodule at the study
level. At the operating point, sensitivity is 0.90 and specificity 0.95. PPV at the deployment
prevalence of 24%:

```text
PPV = (0.90 × 0.24) / (0.90 × 0.24 + 0.05 × 0.76)
    = 0.216 / (0.216 + 0.038)
    = 0.850
```

At a 5% prevalence:

```text
PPV = (0.90 × 0.05) / (0.90 × 0.05 + 0.05 × 0.95)
    = 0.045 / (0.045 + 0.0475)
    = 0.486
```

The model's calibration curve has a slope of 0.91 and an intercept of -0.04 at the study level,
which is acceptable but not perfect. The Brier score is 0.11. At 5% prevalence, the calibration
slope is unchanged but the intercept shifts, and the PPV falls as above.

### 6.6 Reader study

Eight radiologists read the 200 scans twice: once unaided and once with the model's marks. The
mean AUC is 0.79 unaided and 0.84 assisted. The mean difference is 0.05, with a 95% confidence
interval of 0.01 to 0.09 from a multi-reader multi-case analysis. The lower bound is above zero,
so the study supports a superiority claim at the 0.05 level.

The reading time falls from 4.2 to 3.9 minutes per scan, a 7% reduction, but the confidence
interval includes zero. The time claim is not supported.

### 6.7 Diagnostic 1: threshold tuned on the test set

The reported sensitivity of 0.90 was obtained by choosing the threshold that maximized sensitivity
at 0.5 FP/scan on the test set. When the threshold is locked on a separate validation set and
applied to the test set, sensitivity falls to 0.86 and the FP rate rises to 0.62 per scan. The
0.04 difference is the optimism from tuning on the test set.

The diagnostic is a comparison of the threshold chosen on validation with the threshold chosen on
test. If they differ, the test-tuned number is not an estimate; it is a fit.

### 6.8 Diagnostic 2: image-level split contamination

The original split was by image, not by patient. Forty patients had two scans, and 22 of them
appeared in both training and test. The test AUROC was 0.91. After a patient-level split, it fell
to 0.85. The 0.06 difference is larger than the confidence interval on either estimate.

The diagnostic is the patient intersection count between splits. A nonzero count invalidates the
test estimate, regardless of the metric.

### 6.9 Varying one assumption

Assume the deployment prevalence is 5% rather than 24%. The PPV falls from 0.850 to 0.486, as
computed above. The number of alerts per true positive rises from:

```text
1 / 0.850 = 1.18 alerts per true positive
```

to:

```text
1 / 0.486 = 2.06 alerts per true positive
```

If the reading room can absorb 20 alerts per day, the number of true positives found per day falls
from about 17 to about 10. The conclusion changes from "the alert volume is manageable" to "the
alert volume is manageable only if the threshold is raised, which lowers sensitivity". The model
did not change; the prevalence did.

### 6.10 Cost-weighted evaluation

The operating point implies a trade-off between false positives and false negatives. A simple
cost model makes the trade-off explicit. Suppose a missed nodule costs 10 units (delayed
diagnosis) and a false positive costs 1 unit (extra workup). At sensitivity 0.90, specificity
0.95, and prevalence 24%, the expected cost per 1,000 patients is:

```text
false negatives = 1,000 × 0.24 × 0.10 = 24
false positives = 1,000 × 0.76 × 0.05 = 38
cost = 24 × 10 + 38 × 1 = 240 + 38 = 278 units
```

At a higher-sensitivity operating point, sensitivity 0.95 and specificity 0.88:

```text
false negatives = 1,000 × 0.24 × 0.05 = 12
false positives = 1,000 × 0.76 × 0.12 = 91.2
cost = 12 × 10 + 91.2 × 1 = 120 + 91.2 = 211.2 units
```

The higher-sensitivity point has a lower expected cost under these weights, even though its
specificity is lower. The conclusion depends on the cost ratio; at a cost ratio of 2:1, the
original point may win. The cost model is a clinical and economic assumption, not a property of
the model, and it must be stated.

## 7. Code, query, or calculation

The following code computes the Wilson interval, a deterministic bootstrap interval, Dice, study-
level sensitivity and specificity, PPV across prevalence, and a small AUC. It uses only the
standard library.

```python
import math

def wilson(k, n, z=1.96):
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half

lo, hi = wilson(198, 220)
print("Wilson 95% CI      = {:.3f} to {:.3f}".format(lo, hi))

def bootstrap_ci(k, n, reps=2000, seed=20261025):
    state = seed
    estimates = []
    p = k / n
    for _ in range(reps):
        successes = 0
        for _ in range(n):
            state = (1103515245 * state + 12345) % (2 ** 31)
            if state / 2 ** 31 < p:
                successes += 1
        estimates.append(successes / n)
    estimates.sort()
    return estimates[int(0.025 * reps)], estimates[int(0.975 * reps) - 1]

lo_b, hi_b = bootstrap_ci(198, 220)
print("bootstrap 95% CI   = {:.3f} to {:.3f}".format(lo_b, hi_b))

ref, pred, inter = 303, 271, 236
dice = 2 * inter / (ref + pred)
print("Dice               = {:.3f}".format(dice))

print("study sensitivity  = {:.4f}".format(150 / 160))
print("study specificity  = {:.4f}".format(30 / 40))

def ppv(prev, sens, spec):
    return sens * prev / (sens * prev + (1 - spec) * (1 - prev))

for prev in (0.05, 0.24, 0.50):
    print("PPV at {:.0%}       = {:.3f}".format(prev, ppv(prev, 0.90, 0.95)))

def auc(scores, labels):
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else (0.5 if p == n else 0.0)
    return wins / (len(pos) * len(neg))

scores = [0.9, 0.7, 0.6, 0.8, 0.4, 0.5, 0.3, 0.2]
labels = [1, 1, 1, 0, 1, 0, 0, 0]
print("AUC example        = {:.3f}".format(auc(scores, labels)))
```

Expected output:

```text
Wilson 95% CI      = 0.853 to 0.933
bootstrap 95% CI   = 0.859 to 0.936
Dice               = 0.822
study sensitivity  = 0.9375
study specificity  = 0.7500
PPV at 5%       = 0.486
PPV at 24%       = 0.850
PPV at 50%       = 0.947
AUC example        = 0.750
```

The bootstrap uses a linear congruential generator, not a cryptographic random number generator, so
the interval is reproducible. The Wilson interval is narrower than the bootstrap at the upper end
because the bootstrap is not exact for a proportion near 0.9. Both are reported for comparison;
the analysis plan should choose one method in advance.

The code assumes: independent lesions, a binary reference, and a fixed sensitivity and specificity
across prevalence. The independence assumption is false when multiple lesions come from the same
patient; use a cluster bootstrap in that case.

## 8. Data, standards, and regulatory context

### 8.1 Reporting standards

The Checklist for Artificial Intelligence in Medical Imaging (CLAIM), published in 2020 and
updated in 2024, defines the items a study should report. The 2024 update has 44 items, replaces
"ground truth" with "reference standard", discourages "validation" in favour of internal and
external testing, and adds an item on image acquisition protocol. CLAIM is a reporting standard,
not a design standard; it tells you what to write, not what to do.

TRIPOD+AI (Collins et al., BMJ 2024) is the updated reporting guideline for clinical prediction
models using regression or machine learning. It has a 27-item checklist (52 subitems) and a
13-item abstract checklist, with fairness embedded throughout and a new open-science section. It
supersedes the 2015 TRIPOD checklist. A model report that follows CLAIM but not TRIPOD+AI may
still omit the calibration and fairness items that a clinical reviewer expects.

### 8.2 FDA evidence

The special controls at 21 CFR 892.2090 require standalone performance testing, localization
accuracy where applicable, and reader-study evidence where the device claims improved assisted
read. The labeling must include a detailed performance-testing summary with stratified
sub-analyses by anatomy, demographics, medical history, user experience, and imaging equipment.

The FDA's MIDRC-MetricTree and MIC-MET Tree are decision-support tools for choosing metrics by
task, reference-standard reliability, and output type. They are not binding, but they reflect the
agency's view that metric choice should follow from the task, not from convention.

### 8.3 Sample size

A confidence interval half-width of 0.05 for a proportion near 0.9 requires about 139 positive
units:

```text
n = 1.96² × 0.9 × 0.1 / 0.05² = 138.3
```

For a lesion-level sensitivity with 220 lesions, the half-width is about 0.04, as computed. For a
subgroup with 30 lesions, the half-width is about 0.11, which is too wide for a claim. Subgroup
analyses need their own sample size or must be labelled descriptive.

### 8.4 What the standards do not cover

No standard defines the clinical threshold for a specific product, the acceptable alert burden, or
the non-inferiority margin. Those are clinical and commercial decisions that must be justified
and pre-specified.

### 8.5 Uncertainty quantification beyond the confidence interval

A confidence interval describes sampling variability, not the uncertainty of a single prediction.
For a deployed model, the relevant uncertainty may be the probability that a given study is
positive, the uncertainty of the segmentation boundary, or the uncertainty introduced by the
image acquisition. Those are different quantities and require different methods: predictive
intervals, Bayesian approximations, ensembles, or conformal prediction.

A model that reports a probability without calibration and without an uncertainty estimate is
providing a point estimate. That may be sufficient for a triage score and insufficient for a
quantitative measurement. The claim determines the requirement.

### 8.6 Comparing to a competitor

A competitor's published number is not a comparator unless the data, the reference standard, the
unit, the operating point, and the population are the same. They rarely are. The honest comparison
is a head-to-head evaluation on a shared test set, or a re-implementation of the competitor's
method on your data.

If a head-to-head is impossible, the comparison should be labelled indirect and the differences
listed. An indirect comparison can be informative, but it cannot support a claim of superiority.

## 9. Industry practice and case evidence

### 9.1 RSNA challenges and leaderboard evaluation

The RSNA intracranial haemorrhage challenge (2019) used more than 25,000 head CT exams annotated
by more than 60 radiologists. The RSNA 2024 lumbar spine challenge attracted 1,874 teams, and the
2025 aneurysm challenge used CTA, MRA, and MRI from 18 sites annotated by 60 radiologists. These
challenges produce strong benchmarks and a leaderboard, but the leaderboard metric is chosen for
ranking, not for clinical utility. A winning log-loss is not a sensitivity at an acceptable false-
positive rate.

### 9.2 MIDRC-MetricTree

The MIDRC-MetricTree (Drukker et al., Journal of Medical Imaging 2024) is a decision tree that
recommends metrics based on task type, reference-standard reliability, and output type. It covers
classification, detection, segmentation, time-to-event, and estimation. The FDA's MIC-MET Tree is
the classification branch. The practical lesson is that metric choice is a function of the task
and the reference, and that a single metric is rarely sufficient.

### 9.3 Reader studies in mammography

Mammography AI has been evaluated in reader studies and screening programs with mixed results.
McKinney et al. (Nature, 2020) reported an ensemble evaluated on UK and US screening datasets with
a claimed reduction in false positives and false negatives relative to readers. Later
screening-program evaluations reported variable effects on recall and cancer detection. The
difference between a reader study and a screening program is the endpoint: a reader study measures
reader accuracy on a curated sample; a screening program measures recall rate, interval cancers,
and overdiagnosis in a population.

### 9.4 IDx-DR and the value of a pre-specified endpoint

IDx-DR (De Novo DEN180001, 2018) reported observed sensitivity 87.4% (173/198, 95% CI 81.9-92.9%)
and observed specificity 89.5% (556/621, 95% CI 86.9-93.1%) against a reading-centre reference, with
imageability 96.1% (819/852). The endpoints and the reference were pre-specified, and the
imageability rate was reported as a separate metric because an ungradable image is not a negative.

The lesson is that a product with a narrow claim, a fixed capture device, and a pre-specified
endpoint can support an autonomous decision. The same discipline applies to radiology models: the
endpoint, the reference, and the handling of ungradable cases must be defined before the test.

### 9.5 Inference, not fact

The following are inferences: that a specific AUROC difference is clinically meaningful; that a
specific reader-study result will replicate in a different reader population; that a specific
alert burden is tolerable. Each requires a local decision and, ideally, a prospective study. The
documented facts are the reporting standards, the challenge designs, and the published endpoint
results.

### 9.6 Silent mode and prospective evaluation

A silent-mode deployment runs the model on live studies without showing the output to clinicians.
It measures the input distribution, the alert rate, and the agreement with the final report. It
does not measure clinical benefit, because no decision changes. It is the bridge between
retrospective validation and a prospective study.

A prospective study changes the workflow: the output is shown, and the endpoint is a clinical or
process outcome. The endpoint might be time to treatment, recall rate, or reading time. The
design might be a randomized trial, a stepped-wedge cluster trial, or a before-after study. Each
has a different susceptibility to confounding, and the choice must be justified.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Single number without unit | Slide says "AUC 0.91" | Unit and reference omitted | Uninterpretable | Report estimand |
| Threshold tuned on test | Sensitivity falls when locked | Optimism | Inflated estimate | Lock on validation |
| Image-level split | Near-duplicate slices | Contamination | Inflated AUC | Patient-level split |
| Pooled Dice | Small lesions ignored | Voxel weighting | Clinical miss | Per-lesion Dice |
| Study-level sensitivity only | Multiple lesions per study | Denominator mismatch | Overstated detection | Lesion-level metric |
| PPV at enriched prevalence | Case-control sample | Prevalence effect | Misleading deployment | Reweight |
| No confidence interval | Point estimate only | Sampling variability | False precision | Wilson or bootstrap |
| Ignoring clustering | Lesion bootstrap | Non-independence | Narrow interval | Cluster bootstrap |
| Calibration ignored | AUROC only | Probability not meaningful | Threshold errors | Calibration curve |
| Subgroup not reported | Aggregate only | Heterogeneity | Equity and safety risk | Pre-specified subgroups |
| Reader study too small | Wide interval | Low power | Inconclusive | Sample size method |
| Comparator on different data | Published number | Non-comparable | Invalid claim | Re-run baseline |
| Error analysis absent | No failure examples | Unknown failure mode | Surprise at deployment | Stratified errors |
| Verification rate omitted | Negatives unverified | Reference bias | Inflated specificity | Report verification |

### 10.1 Red flags that stop a result

1. The unit of analysis is not stated.
2. The operating point is not stated.
3. The confidence interval is missing.
4. The threshold was chosen on the test set.
5. The split is by image, not patient.
6. The PPV uses the enriched prevalence.
7. The Dice is pooled across lesions.

### 10.2 An evaluation report checklist

1. The estimand is stated.
2. The unit and reference standard are stated.
3. The split design and patient overlap are reported.
4. The operating point and how it was chosen are reported.
5. The primary metric has a confidence interval.
6. Calibration is reported if a probability is used.
7. Subgroups are pre-specified and reported with counts.
8. A reader study is reported if the claim is assisted read.
9. The error analysis is stratified.
10. The limitations and the untested populations are stated.

A report that passes this checklist can be reviewed. A report that fails it cannot be compared to
another product, because the measurement is undefined.

### 10.3 What a reviewer looks for first

A reviewer looks for the estimand, the split, the reference standard, and the confidence interval,
in that order. If any of the four is missing, the rest of the report is hard to evaluate. The
metric itself is usually the least ambiguous part; the design around it is where the errors hide.

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test |
|---|---|---|---|
| The reference standard is correct | Valid metric | Label noise | Adjudication sample |
| The test set is independent | Unbiased estimate | Contamination | Split audit |
| The operating point is fixed | Reproducible decision | Threshold tuning | Lock before test |
| The metric matches the claim | Clinical validity | Proxy metric | Estimand review |
| The confidence interval method fits | Correct uncertainty | Clustering, small n | Compare methods |
| The prevalence is known | PPV and calibration | Case-control sampling | Reweight |
| The subgroups are pre-specified | Valid inference | Post hoc mining | Analysis plan |
| The readers are representative | Reader-study validity | Convenience sample | Reader demographics |
| The cases are representative | Case-mix validity | Enriched sample | Sampling audit |
| The comparator is fair | Attribution | Different data | Same-data baseline |

### 11.1 Limits of this topic

This topic covers the design and reporting of imaging model evaluation. It does not cover the full
theory of ROC analysis, the details of MRMC variance components, or the design of prospective
randomized trials. It gives the data scientist the vocabulary and the arithmetic to produce a
defensible evaluation, not to run a pivotal trial without a statistician.

### 11.2 What would change the conclusion

If the test set were from the same site as training, the external claim would weaken and the
site-held-out result would become the relevant number. If the reference standard were a report
rather than consensus, the ceiling would fall and the measured sensitivity would be bounded by
the report's accuracy. If the deployment prevalence were 5%, the PPV would fall and the alert
burden would rise, even with the same sensitivity and specificity.

### 11.3 The limit of retrospective evaluation

A retrospective evaluation cannot measure workflow effects, alert fatigue, or trust. Those require
a prospective study. A retrospective result can justify a prospective study; it cannot replace
one. The gap between the two is the most common reason a model that looked good in validation
disappoints in deployment.

## 12. Exercises

### Exercise 1. Estimand (recall, 20 minutes)

**Deliverable:** An estimand for a triage model for large-vessel occlusion, including population,
unit, reference, operating point, and metric.

**Acceptance criteria:** All five elements; time horizon; workflow role.

**Time:** 20 minutes.

### Exercise 2. Wilson interval (calculation, 25 minutes)

**Deliverable:** Compute the Wilson 95% interval for 150/180 and interpret it.

**Acceptance criteria:** Correct interval (0.772-0.881); interpretation in one sentence.

**Time:** 25 minutes.

### Exercise 3. Study vs lesion (application, 30 minutes)

**Deliverable:** For 100 scans with 140 lesions, 120 detected, compute the lesion-level and
study-level sensitivity given 90 positive scans and 85 with a detected lesion.

**Acceptance criteria:** Correct values (0.857 and 0.944); explanation of the difference.

**Time:** 30 minutes.

### Exercise 4. PPV across prevalence (calculation, 25 minutes)

**Deliverable:** For sensitivity 0.85 and specificity 0.90, compute PPV at 2%, 10%, and 30%.

**Acceptance criteria:** Correct values (0.148, 0.486, 0.784); formula shown.

**Time:** 25 minutes.

### Exercise 5. Calibration critique (critique, 40 minutes)

**Deliverable:** A 500-word critique of a paper that reports AUROC but not calibration or
decision-curve analysis.

**Acceptance criteria:** Explains discrimination vs calibration; gives a clinical scenario where
the omission matters; proposes the missing analyses.

**Time:** 40 minutes.

### Exercise 6. Reader study design (design, 50 minutes)

**Deliverable:** An MRMC design for 6 readers and a superiority claim, with case count, reading
order, washout, and analysis method.

**Acceptance criteria:** Reader and case counts justified; blinding; pre-specified analysis;
non-inferiority margin if applicable.

**Time:** 50 minutes.

### Exercise 7. Error analysis (application, 40 minutes)

**Deliverable:** A stratified error table for a detector by lesion size, location, and scanner.

**Acceptance criteria:** Counts and rates per stratum; at least one actionable finding; confidence
intervals.

**Time:** 40 minutes.

### Exercise 8. Report checklist (design, 45 minutes)

**Deliverable:** A CLAIM-based checklist for a model report, with the evidence for each item.

**Acceptance criteria:** At least 15 items; evidence location; gaps identified.

**Time:** 45 minutes.

## 13. Self-check answers

**A1. What is an estimand?**
A precise statement of the quantity to estimate, including the population, unit, reference
standard, operating point, time horizon, and metric. It is written before the analysis.

**A2. Wilson interval for 150/180.**
p = 0.8333, n = 180. Centre ≈ 0.828, half-width ≈ 0.054, interval 0.772-0.881. The exact value
depends on rounding; the interpretation is that the sensitivity is about 0.83 with a wide
interval.

**A3. Lesion-level vs study-level sensitivity for 120/140 and 85/90.**
Lesion: 120/140 = 0.857. Study: 85/90 = 0.944. The study metric is higher because a study with
multiple lesions counts once and a missed lesion may not change the study label.

**A4. PPV at 2%, 10%, 30% for sensitivity 0.85 and specificity 0.90.**
2%: 0.017/(0.017+0.098)=0.148. 10%: 0.085/(0.085+0.09)=0.486. 30%:
0.255/(0.255+0.07)=0.784.

**A5. Why is AUROC not clinical utility?**
AUROC summarizes ranking across all thresholds and is prevalence-invariant. Clinical utility
depends on the operating point, the prevalence, the consequences of false positives and false
negatives, and the workflow. A high AUROC can have no net benefit at the relevant threshold.

**A6. Why use a cluster bootstrap?**
Lesions from the same patient are correlated. Resampling lesions treats them as independent and
produces intervals that are too narrow. Resampling patients preserves the clustering.

**A7. What is the difference between internal and external validation?**
Internal validation holds out data from the same source and distribution. External validation uses
a different site, scanner, or time period. External is stronger evidence of transportability.

**A8. What does a calibration slope of 0.8 mean?**
The predicted probabilities are too extreme: a one-unit increase in the log-odds prediction
corresponds to a 0.8-unit increase in observed log-odds. The model overfits or is overconfident;
recalibration is needed.

**A9. Design rubric for Exercise 6.** A strong design justifies the number of readers and cases
with a sample size method, specifies reading order and washout, blinds the readers, and pre-
specifies the analysis. A weak design uses one reader and a convenience sample.

**A10. Design rubric for Exercise 8.** A strong checklist maps each CLAIM item to a section of the
report and identifies the gaps. A weak checklist copies the item names without evidence.

## 14. Interview questions

**Q1. What is the difference between AUROC and sensitivity at a fixed FP rate?**
A strong answer says AUROC summarizes ranking across thresholds and is prevalence-invariant, while
sensitivity at a fixed FP rate is an operating-point metric that matches a workflow. A weak answer
treats them as interchangeable.

**Q2. Why is study-level sensitivity higher than lesion-level?**
A strong answer explains that a study counts once and multiple lesions dilute the lesion-level
denominator. A weak answer says the model is better at studies.

**Q3. How do you compute a confidence interval for sensitivity?**
A strong answer uses Wilson or Clopper-Pearson for a proportion, and a cluster bootstrap when
units are clustered. A weak answer uses the normal approximation blindly.

**Q4. What is calibration and why does it matter?**
A strong answer distinguishes calibration from discrimination and explains that a probability
threshold requires calibrated probabilities. A weak answer says calibration is the same as AUC.

**Q5. How do you design a reader study?**
A strong answer describes MRMC, reader and case sample size, blinding, reading order, and the
analysis method. A weak answer says "ask a few radiologists".

**Q6. What is a non-inferiority margin?**
A strong answer defines the largest acceptable loss of performance and says it must be pre-
specified and clinically justified. A weak answer says "not worse".

**Q7. Why can a high Dice be misleading?**
A strong answer explains voxel pooling and small-lesion misses, and proposes per-lesion Dice and
Hausdorff distance. A weak answer says Dice is the standard.

**Q8. How do you handle ungradable cases?**
A strong answer reports imageability as a separate metric and does not count ungradable as
negative. A weak answer excludes them silently.

**Q9. What is the difference between internal and external validation?**
A strong answer defines the source difference and the transportability claim. A weak answer says
one is a holdout and the other is a test set.

**Q10. How do you report a subgroup analysis?**
A strong answer pre-specifies subgroups, reports counts and intervals, and labels exploratory
analyses. A weak answer reports a p-value without counts.

## 15. Key terms

- **ASSD** — average symmetric surface distance.
- **AUPRC** — area under the precision-recall curve.
- **AUROC** — area under the receiver operating characteristic curve.
- **Bootstrap** — resampling for uncertainty.
- **Brier score** — mean squared error of probabilities.
- **Calibration** — agreement between predicted and observed risk.
- **Cluster bootstrap** — resampling clusters such as patients.
- **Clopper-Pearson** — exact binomial interval.
- **DeLong** — AUC variance estimator.
- **Decision curve** — net benefit across thresholds.
- **Dice** — voxel overlap metric.
- **ECE** — expected calibration error.
- **Estimand** — precise quantity to estimate.
- **External validation** — validation on a different source.
- **FROC** — free-response ROC.
- **Hausdorff distance** — maximum boundary distance.
- **Internal validation** — validation on a held-out same-source sample.
- **IoU** — intersection over union.
- **MRMC** — multi-reader multi-case study.
- **Net benefit** — decision-curve utility.
- **Non-inferiority** — not worse by a pre-specified margin.
- **NPV** — negative predictive value.
- **Operating point** — threshold converting score to decision.
- **PPV** — positive predictive value.
- **Prevalence** — proportion of positives in the reference.
- **Reader study** — human performance study.
- **Sensitivity** — true positive rate.
- **Specificity** — true negative rate.
- **Spectrum** — distribution of disease severity.
- **Subgroup** — defined population subset.
- **Temporal validation** — validation on a later period.
- **Wilson interval** — score interval for a proportion.

## 16. Sources

- Mongan, J., Moy, L., Kahn, C. E. Jr., "Checklist for Artificial Intelligence in Medical Imaging
  (CLAIM)," Radiology: Artificial Intelligence 2(2): e200029 (2020).
  https://doi.org/10.1148/ryai.2020200029
- Tejani, A. S. et al., "Checklist for Artificial Intelligence in Medical Imaging (CLAIM): 2024
  Update," Radiology: Artificial Intelligence 6: e240300 (2024).
  https://doi.org/10.1148/ryai.240300
- Collins, G. S. et al., "TRIPOD+AI statement: updated guidance for reporting clinical prediction
  models that use regression or machine learning methods," BMJ 385: e078378 (2024).
  https://doi.org/10.1136/bmj-2023-078378
- Drukker, K. et al., "MIDRC-MetricTree: a decision tree-based tool for recommending performance
  metrics in artificial intelligence-assisted medical image analysis," Journal of Medical Imaging
  11(2): 024504 (2024). https://doi.org/10.1117/1.JMI.11.2.024504
- U.S. Food and Drug Administration, MIC-MET Tree: Decision Tree for Medical Imaging AI/ML
  Classification Metrics, RST24AI05.01, April 2024.
  https://cdrh-rst.fda.gov/mic-met-tree-decision-tree-medical-imaging-aiml-classification-metrics
  (accessed 25 October 2026).
- U.S. Food and Drug Administration, 21 CFR 892.2090, final order 90 FR 24969, 13 June 2025.
  https://www.ecfr.gov/current/title-21/chapter-I/subchapter-H/part-892/subpart-B/section-892.2090
  (accessed 25 October 2026).
- Obuchowski, N. A., Bullen, J. A., "Multireader Diagnostic Accuracy Imaging Studies: Fundamentals
  of Design and Analysis," Radiology 303(1): 26-34 (2022).
  https://doi.org/10.1148/radiol.211593
- McKinney, S. M. et al., "International evaluation of an AI system for breast cancer screening,"
  Nature 577, 89-94 (2020). https://doi.org/10.1038/s41586-019-1799-6
- Abràmoff, M. D. et al., "Pivotal trial of an autonomous AI-based diagnostic system for detection
  of diabetic retinopathy in primary care offices," npj Digital Medicine 1, 39 (2018).
  https://doi.org/10.1038/s41746-018-0040-6
- U.S. Food and Drug Administration, De Novo DEN180001 (IDx-DR), decision summary.
  https://www.accessdata.fda.gov/cdrh_docs/reviews/DEN180001.pdf (accessed 25 October 2026).
- RSNA Intracranial Haemorrhage Detection Challenge (2019).
  https://www.rsna.org/artificial-intelligence/ai-image-challenge/rsna-intracranial-hemorrhage-detection-challenge-2019
  (accessed 25 October 2026).

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`06_leakage-and-shortcuts.md`](06_leakage-and-shortcuts.md)
- Next topic: [`08_workflow-integration.md`](08_workflow-integration.md)
- Related topic: [`04_labels-and-annotation.md`](04_labels-and-annotation.md)
- Root survey: [`../../10_medical_imaging.md`](../../10_medical_imaging.md)
- Labs: [`../../labs/`](../../labs/)

> **Note:** The estimand, the split, the reference, and the interval are the four items a reviewer checks first.

> **Note:** A metric without a unit, a reference standard, and an operating point is not a result.
> It is a number looking for a claim.
