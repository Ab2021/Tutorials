# Cross-Specialty Comparison

> Module 23 — Specialty Electives
> Topic 07 of 07
> Estimated study time: 3 hours
> Prerequisites: topics 01-06 of this module
> Version 2026.3

> **Note:** This file is educational. It is not medical advice and not a regulatory
> determination. The comparison below is about measurement and evidence, not about
> which specialty matters more.

## 1. Why this topic exists

Every specialty in this module has the same analytic skeleton: an eligible population, a
time zero, a set of features available at the decision, a predicted state, an action, an
endpoint, and a payer. The nouns differ. The skeleton does not. That similarity is the
trap.

A model that works in one specialty looks portable. It has an area under the receiver
operating characteristic curve (AUROC), a calibration plot, and a validation cohort. A
team moving from cardiology to oncology, or from critical care to paediatrics, can
reasonably ask why the same architecture should not apply. Sometimes it does. Often it
does not, and the failure is not in the architecture. It is in four boundaries that the
model crosses without noticing:

- **Measurement.** The predictor means something different. An ejection fraction is a
  vendor-dependent ultrasound estimate; a tumour measurement is a reader-dependent
  caliper; a creatinine is a muscle-mass-dependent proxy for kidney function.
- **Endpoint.** The event means something different. A readmission in cardiology, a
  progression in oncology, and a late report in public health are all "events", and none
  is ascertained the same way.
- **Coding.** The same clinical concept has different codes, different granularity, and
  different rules across specialties and years.
- **Workflow.** The action, the actor, and the timing differ. A model that recommends a
  biopsy and a model that recommends a fluid bolus have different costs of a false
  positive, different confirmation pathways, and different regulatory classes.

This topic exists because the most expensive mistake in a multi-specialty organisation
is assuming transfer. A health system builds a deterioration model in the intensive care
unit (ICU), deploys it on the wards, and finds that the alert rate triples because the
case mix and the monitoring intensity changed. A vendor sells a sepsis model validated
in one academic centre to a community hospital and finds an operating characteristic
that no one recognises. A pharma company reuses an oncology real-world pipeline for a
rare disease and finds that the endpoint cannot be constructed at all.

### 1.1 Who is blocked when transfer fails

| Stakeholder | What they lose | Why they cannot fix it alone |
|---|---|---|
| Patient | A decision made on a probability calibrated for someone else | Cannot see the model or its provenance |
| Clinician | Trust in a tool that fires at the wrong rate | Cannot recalibrate it locally |
| Health system | The cost of a deployment that is withdrawn after a pilot | Depends on the vendor for updates |
| Vendor | A reference site that becomes a cautionary tale | Needs multi-site evidence to sell |
| Payer | Payment for a tool whose benefit was demonstrated elsewhere | Contracts on the vendor's evidence |
| Regulator | A cleared product used outside its intended population | Clearance is tied to a stated intended use |
| Analyst | A result that cannot be defended in a second setting | Inherited a model without its assumptions |
| Research community | A literature that reports development and calls it validation | Reporting norms reward novelty over confirmation |

The last two rows are the professional stakes. Cross-specialty transfer is not a
theoretical concern; it is the normal condition of a product that succeeds in one market
and is then sold into another.

The cost structure explains why the mistake persists. A transfer review costs a few weeks
of analyst time and delays a launch. A failed deployment costs a pilot, a clinical
champion's credibility, and sometimes a contract. The review is cheap and the failure is
expensive, and yet the review is the step that gets skipped when a launch date is fixed.
The reason is that the review produces a document that says "not yet", and "not yet" is
harder to defend in a steering committee than a pilot that has already started.

### 1.2 The industry question

Which decision does this work change, who pays for it, and which specialty-specific bias
will invalidate the evidence first? For this topic the third clause becomes a comparison:
which bias is shared across specialties, and which is specific to one. A shared bias
(time-zero leakage, overfitting) is a methods problem with a general fix. A
specialty-specific bias (stage migration in oncology, prevalence-dependent positive
predictive value in cardiology, slide leakage in pathology, label leakage in critical
care, weight-based dosing in paediatrics, reporting delay in public health) requires a
domain audit that a general methods reviewer will not perform.

### 1.3 What this topic covers

This topic builds a transfer test: a procedure for deciding whether an analytic artifact
from one specialty can be used in another, and what must change if it can. It compares
the six specialties on measurement, endpoint, coding, workflow, and regulation. It uses
one worked example of a mortality model transported across three settings, with the
arithmetic of prevalence, calibration, and net benefit.

It does not re-teach the specialties. It assumes you have read at least one of topics 01
to 06, and it links back to each.

## 2. Learning outcomes

By the end of this topic you can:

| # | Outcome | Evidence of attainment |
|---:|---|---|
| 1 | Compare two specialties on measurement, endpoint, coding, and workflow using a structured table | Exercise 1; self-check 1 |
| 2 | Compute positive predictive value (PPV) at a deployment prevalence from sensitivity and specificity, and show how it changes | Exercise 2; self-check 2 |
| 3 | Distinguish recalibration from refitting and say which is required when discrimination transports but calibration does not | Exercise 3; interview question 3 |
| 4 | Detect pooled calibration that masks opposite miscalibration across sites | Exercise 4; self-check 4 |
| 5 | Identify the specialty-specific bias most likely to invalidate a transfer, and design the audit that would reveal it | Exercise 5; interview question 5 |
| 6 | Decide between adopt, recalibrate, refit, rebuild, and reject for a stated transfer | Exercise 6; self-check 6 |
| 7 | Map a transferred product's regulatory class and intended use, and identify when transfer changes the class | Exercise 7; interview question 7 |
| 8 | Write a transportability memo that a clinical champion and a regulator would both accept | Exercise 8 |

Outcomes 1 and 5 are comparative judgement. Outcomes 2, 3, and 4 are arithmetic and are
the ones that can be marked objectively. Outcomes 6 and 8 are decisions and are assessed
by rubric, not by a single right answer.

The outcomes are deliberately ordered from description to decision. A learner who can
fill in the four-boundary table but cannot compute PPV at a deployment prevalence has
learned vocabulary without the test that makes the vocabulary useful. A learner who can
compute PPV but cannot name the workflow constraint has learned a formula without the
context that determines whether the number matters.

Outcome 7 is included because transfer is a regulatory event as well as a statistical
one. A model that moves from a research cohort to a clinical claim, or from one intended
population to another, may change class. That is not a detail to be handled by legal
after the fact; it changes the evidence the project must produce.

### 2.1 How the outcomes map to the module

Outcome 1 synthesises topics 01 to 06. Outcome 2 is the cardiology lesson applied
generally: the same sensitivity and specificity produce different PPV at different
prevalences, and every specialty has a different prevalence. Outcome 3 is the distinction
between the two things people call "validation". Outcome 4 is the statistical version of
the same mistake. Outcomes 5 to 7 connect to module 14 (device regulation) and module 18
(quality).

If you retain one outcome, retain outcome 2. It is the cheapest transfer test in
existence: before reading a single validation paper, compute what the reported
sensitivity and specificity imply at your prevalence.

## 3. The core mental model

A model crosses four boundaries when it moves between specialties. Each boundary can
break it independently.

```text
  SOURCE SPECIALTY                              TARGET SPECIALTY
  ------------------                            ----------------
  population  ------ (1) case mix / spectrum ------>  population
  measurement ------ (2) instrument / definition --->  measurement
  endpoint    ------ (3) ascertainment / window ---->  endpoint
  workflow    ------ (4) action / actor / timing --->  workflow
  coding      ------ (5) vocabulary / version ------>  coding
  regulation  ------ (6) intended use / class ------>  regulation
```

The first three boundaries determine whether the model's numbers mean the same thing.
The last three determine whether the model can be used, paid for, and defended. A
transfer that passes the first three and fails the last three is technically valid and
commercially dead.

### 3.1 Boundary 1: case mix and spectrum

Discrimination is a property of the population as much as of the model. A model
developed in patients who all had a test because a clinician suspected disease will
perform differently in a population screened without suspicion. This is spectrum bias,
and it changes sensitivity, specificity, and calibration simultaneously.

```text
  high-prevalence, symptomatic cohort  ->  sensitivity high, specificity lower
  low-prevalence, screened cohort      ->  sensitivity lower, specificity higher
```

The direction is not universal, but the point is: sensitivity and specificity are not
constants of the model. They are estimated in a population, and they move when the
population moves.

### 3.2 Boundary 2: measurement

Every specialty has a measurement whose value depends on the instrument, the operator,
or the protocol.

| Specialty | Measurement | What it depends on | Transfer consequence |
|---|---|---|---|
| Oncology | Tumour diameter, stage | Reader, slice thickness, staging edition | Response and stage are not comparable across sites |
| Cardiology | Ejection fraction, ECG intervals | Vendor, lead placement, filter settings | Thresholds move with device |
| Pathology | Grade, biomarker score | Stain batch, scanner, inter-pathologist agreement | Labels have irreducible noise |
| Critical care | SOFA components, fluid balance | Charting practice, unit protocol | Score distribution shifts by unit |
| Paediatrics | Weight, height, gestational age | Scale, stadiometer, dating method | Z-scores change with reference |
| Public health | Case counts, test positivity | Testing access, case definition, reporting delay | Trends change with the reporting system |

The transfer test is not "is the measurement standardised" but "is the measurement
standardised in the same way at the target". A DICOM-conformant scanner can still produce
a different ejection fraction from a different vendor.

### 3.3 Boundary 3: endpoint ascertainment

The same endpoint name can mean different things in different specialties, and the same
specialty can ascertain the same endpoint differently in two settings.

| Endpoint | Oncology | Cardiology | Critical care | Public health |
|---|---|---|---|---|
| Death | Registry-linked, complete | Registry or claims, incomplete | In-hospital, discharge disposition ambiguous | Death certificate, coded cause |
| Event of interest | Progression on imaging | Myocardial infarction, adjudicated | Deterioration, escalation | Case, by definition |
| Time zero | Diagnosis or randomisation | Admission or procedure | Suspicion time | Symptom onset |
| Censoring | Last adequate assessment | Last follow-up | Discharge | Report vintage |

A model trained to predict "death" in a critical care cohort is predicting in-hospital
death, because that is what the data capture. Deployed in a primary care setting, where
most patients are not in hospital, the endpoint no longer exists in the same form.

### 3.4 Boundary 4: workflow and regulation

A prediction becomes a product only when it changes an action. The action determines the
threshold, the acceptable PPV, and the regulatory class.

```text
  prediction -> action -> confirmation -> consequence -> regulation
```

A high-sensitivity, low-PPV screening alert is acceptable when the confirmation test is
cheap and safe. The same alert is unacceptable when the confirmation is an invasive
biopsy or an anticoagulant. The model can be identical; the transfer fails because the
downstream action changed.

### 3.5 The transfer decision

There are five outcomes, not two. The decision table in Step 9 formalises them. The
mental model is that recalibration fixes the average, refitting fixes the shape, and
rebuilding fixes the measurement. Choosing the cheapest option that resolves the failure
is the discipline.

A way to remember the ladder is to ask what is broken:

| What is broken | Cheapest repair | What stays the same |
|---|---|---|
| The average risk | Recalibration in the large | Predictors, coefficients, ranking |
| The confidence of the predictions | Recalibration with slope | Predictors, ranking |
| The predictor weights | Refitting | Predictors, measurement |
| The predictors or the endpoint | Rebuilding | Nothing |
| The action or the harm | Rejection | Nothing, and the model is irrelevant |

Most transfer failures are at the first or second rung. Most transfer arguments are about
the fourth, because rebuilding is visible and expensive. The analytic skill is to
establish which rung is actually broken before the argument starts.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| Transportability | Whether a model's performance carries to a new population or setting | Treated as a property of the algorithm | Judgement with evidence |
| External validation | Evaluation of a fixed model in data not used for development | Called validation when the model was refit | Study design |
| Temporal validation | Evaluation in a later period from the same site | Treated as external | Study design |
| Geographical validation | Evaluation at a different site | Treated as transport to a different case mix | Study design |
| Spectrum bias | Performance changes because the case mix changed | Treated as model degradation | Effect on sens/spec |
| Case mix | Distribution of severity, comorbidity, and presentation | Ignored when comparing sites | Distribution |
| Prevalence | Proportion with the outcome in the target population | Taken from the development cohort | Proportion |
| PPV | Probability that a positive result is a true positive | Reported only at development prevalence | Proportion |
| NPV | Probability that a negative result is a true negative | Assumed constant across prevalences | Proportion |
| Sensitivity | True positive rate at a threshold | Treated as a fixed model property | Proportion |
| Specificity | True negative rate at a threshold | Treated as a fixed model property | Proportion |
| Calibration | Agreement between predicted probabilities and observed frequencies | Confused with discrimination | Slope and intercept |
| Calibration slope | Regression slope of outcome on the linear predictor | Reported without the intercept | Dimensionless, ideal 1 |
| Calibration intercept | Calibration-in-the-large on the logit scale | Reported without the slope | Logit units, ideal 0 |
| O:E ratio | Observed divided by expected events | Computed without a confidence interval | Ratio, ideal 1 |
| Recalibration | Adjusting intercept and slope without changing predictors | Confused with refitting | Model update |
| Refitting | Re-estimating predictor coefficients in the target data | Called recalibration | Model update |
| Rebuilding | Developing a new model with target-specific predictors | Avoided when measurement changed | Model development |
| Discrimination | Ability to rank patients by risk | Treated as sufficient for deployment | AUROC, c-statistic |
| AUROC | Area under the receiver operating characteristic curve | Compared across populations as if constant | 0.5 to 1 |
| AUPRC | Area under the precision-recall curve | Omitted when prevalence is low | 0 to 1 |
| Net benefit | Decision-analytic value at a threshold probability | Replaced by accuracy | Proportion |
| Decision curve | Net benefit across thresholds | Reported without the treat-all comparator | Curve |
| Intended use | The stated purpose, population, user, and setting | Written after the model is built | Regulatory document |
| Regulatory class | Risk-based category determining the pathway | Assumed unchanged by transfer | Jurisdiction-specific |
| Software as a medical device | Software with a medical intended use | Confused with general software | Regulatory category |
| Clinical decision support | Software that supports, rather than directs, a clinician | Assumed to be exempt | Regulatory category |
| Estimand | The precise quantity a study estimates | Left implicit | ICH E9(R1) framework |
| Dataset shift | Change in the joint distribution of inputs and outcome | Treated as covariate shift only | Distribution |
| Label shift | Change in the outcome prevalence with stable conditional distributions | Ignored in calibration | Distribution |

### 4.1 Discrimination versus calibration

These are the two properties most often conflated, and the confusion is the single most
common reason a transferred model fails silently.

Discrimination asks whether the model ranks a patient who has the outcome above one who
does not. It is summarised by AUROC, which is prevalence-invariant by construction.
Calibration asks whether a predicted 10 percent means 10 percent in this population. It
is prevalence-dependent.

A model can have an AUROC of 0.85 and be badly calibrated. In that case, every ranking
is right and every number is wrong, which means every threshold-based action is wrong
while every audit of discrimination passes. A transfer that reports AUROC and omits
calibration has not tested the property that matters most for the action.

### 4.2 Recalibration, refitting, rebuilding

| Operation | What changes | When it is sufficient | What it cannot fix |
|---|---|---|---|
| Recalibration in the large | Intercept only | Prevalence differs, slope is 1 | Any change in predictor effects |
| Recalibration with slope | Intercept and slope | Over- or under-confidence | Changed predictor-outcome relationships |
| Refitting | Coefficients re-estimated | Same predictors, different weights | Measurement definition changes |
| Rebuilding | Predictors and coefficients | Measurement or endpoint changed | Nothing, but it is a new product |
| Rejecting transfer | Nothing | The action or harm profile changed | Nothing |

The table is ordered by cost. The error is to jump to rebuilding when recalibration would
suffice, or to accept recalibration when the measurement itself changed.

## 5. The mechanics, step by step

This section is a procedure for deciding whether an analytic artifact transfers. It is
written as ten steps. Each has an input, an output, a rationale, branch points, and edge
cases. The procedure assumes the artifact is a prediction model, but the same steps apply
to an endpoint definition, a cohort builder, or a feature pipeline.

### Step 1 — Characterise the source

**Input.** The development study, the model card, and the source dataset documentation.

**Output.** A source profile: population, setting, time zero, outcome definition,
predictors and their measurement, prevalence, and performance with confidence intervals.

**Rationale.** Transfer cannot be assessed without knowing what was developed. A model
card that reports only AUROC has not described the artifact.

**Procedure.**

1. Record the eligibility criteria and the setting: admitted patients, outpatients,
   screened volunteers, or a registry subset.
2. Record the outcome definition and the ascertainment source.
3. Record the prevalence of the outcome in the development data.
4. Record each predictor's measurement instrument and version.
5. Record the performance metrics with confidence intervals and the data split used.
6. Record the intended use as stated, not as inferred.

**Branch point.** If the source study reports only discrimination, the transfer
assessment must treat calibration as unknown, which is a material gap rather than a
neutral omission.

**Edge case.** A model developed on a convenience sample with a high outcome prevalence
will appear well calibrated in the development report and will over-predict in a
lower-prevalence target. The source profile must record prevalence explicitly for this
reason.

### Step 2 — Characterise the target decision

**Input.** The target workflow, the decision, the actor, and the harm of each error.

**Output.** A target profile: population, decision point, action, threshold, and
acceptable error rates.

**Rationale.** Transfer is only meaningful relative to a decision. The same model can be
adequate for a low-harm triage action and inadequate for a high-harm treatment decision.

**Procedure.**

1. Name the decision and the actor.
2. Name the action that follows a positive result and the confirmation pathway.
3. State the harm of a false positive and a false negative in the target workflow.
4. State the operating threshold implied by the workflow, not by statistical convention.
5. State the capacity constraint: how many positives the workflow can absorb per day.

**Branch point.** If the target has no confirmation pathway, a low-PPV alert is not
actionable regardless of the model's discrimination.

**Edge case.** A target where the action is mandatory once triggered, such as an
infection-control isolation order. The threshold must be much higher because the cost of
a false positive is borne by the patient and the system.

### Step 3 — Compare outcome definitions

**Input.** Source and target outcome definitions.

**Output.** A concordance judgement: identical, mappable, or not comparable.

**Rationale.** If the outcome means something different, the model predicts a different
thing. No amount of recalibration fixes a different target.

**Procedure.**

1. Compare the event definition word by word.
2. Compare the ascertainment source: registry, EHR, claims, adjudication.
3. Compare the time window. A 30-day mortality model and an in-hospital mortality model
   are different products.
4. Compare the handling of competing events. Death competes with discharge in critical
   care and with progression in oncology.
5. Decide concordance and record the reason.

**Branch point.** If the definitions are mappable but not identical, estimate the
mapping error on a sample with both definitions recorded. If the mapping error exceeds
the effect size of interest, the transfer fails at this step.

**Edge case.** A target where the outcome is not observable at all, such as a
progression endpoint in a setting without protocolised imaging. The transfer is not
weak; it is impossible, and the correct decision is to rebuild around an observable
endpoint.

**Worked check for this step.** Take the source definition "30-day all-cause mortality"
and the target definition "in-hospital mortality". They agree for patients who die in
hospital within 30 days and disagree for two groups: patients discharged to hospice who
die within 30 days, and patients who die in hospital after day 30. If the target records
discharge disposition, the first group can be partially recovered by linkage; if it does
not, the two endpoints differ by a fraction that is unknown and probably related to
access to hospice care. That relation makes the difference a confounder, not noise.

### Step 4 — Compare time zero

**Input.** Source and target index dates.

**Output.** A time-zero mapping and a list of features that become unavailable.

**Rationale.** Every prediction is conditional on information available at a moment.
Moving the moment changes the admissible feature set and the meaning of the prediction
horizon.

**Procedure.**

1. Identify the source time zero.
2. Identify the target time zero.
3. List every feature and classify it as available before both, before one, or after
   one.
4. Remove or re-time features that are unavailable at the target time zero.
5. Recompute the horizon relative to the new time zero.

**Branch point.** If the target time zero is earlier, features that were available in
the source become leakage in the target. If it is later, the model loses predictive
signal and may need new features.

**Edge case.** A target where time zero is defined by a process that itself depends on
the outcome, such as a sepsis suspicion time derived from a culture order placed after
deterioration. The transfer inherits the leakage.

### Step 5 — Compare measurement instruments

**Input.** Source and target measurement definitions, devices, and protocols.

**Output.** A measurement concordance table with a direction of bias for each
non-concordant item.

**Rationale.** Measurement differences change the predictor distribution and the
predictor-outcome relationship. They are the most common cause of a transfer that passes
statistical checks and fails clinically.

**Procedure.**

1. For each predictor, record the instrument, the protocol, and the version.
2. Identify differences: vendor, calibration, units, timing, operator.
3. Estimate the direction and size of the resulting distribution shift on a small
   paired sample if available.
4. Decide whether the difference is tolerable, correctable by harmonisation, or fatal.

**Branch point.** If several predictors shift in the same direction, the model's linear
predictor shifts systematically and recalibration may absorb it. If they shift in
different directions, recalibration cannot.

**Edge case.** A unit change that looks cosmetic. Milligrams per decilitre and
millimoles per litre are convertible, but a pipeline that assumes the wrong unit
produces a predictor that is wrong by a constant factor, and the model will appear to
have a shifted distribution rather than a unit error.

### Step 6 — Compare case mix and estimate target prevalence

**Input.** Target population data, even a small sample.

**Output.** An estimated target prevalence with an interval, and a comparison of
severity distribution.

**Rationale.** Prevalence drives PPV, NPV, and calibration. It cannot be assumed to equal
the source value.

**Procedure.**

1. Estimate the outcome prevalence in the target from the best available data.
2. Compare severity, age, comorbidity, and presentation with the source.
3. Compute PPV and NPV at the target prevalence using the source sensitivity and
   specificity as a first approximation.
4. State the approximation and its assumption: that sensitivity and specificity
   transport even though prevalence does not.

**Branch point.** If the target prevalence is an order of magnitude lower, the PPV will
fall substantially even with unchanged sensitivity and specificity. Decide whether the
workflow can tolerate it before proceeding.

**Edge case.** A target prevalence estimated from a selected sample, such as patients who
received the confirmatory test. That estimate is biased by verification and is usually
higher than the population prevalence.

### Step 7 — Test calibration and discrimination in the target

**Input.** A target validation sample with the outcome.

**Output.** Calibration slope, intercept, O:E ratio, AUROC, and AUPRC, with intervals.

**Rationale.** This is the empirical test. It requires target data, and it cannot be
replaced by a literature comparison.

**Procedure.**

1. Apply the model unchanged and compute the linear predictor.
2. Fit a logistic model of outcome on the linear predictor. The slope is the calibration
   slope and the intercept is calibration-in-the-large.
3. Compute the O:E ratio with a confidence interval.
4. Compute AUROC and AUPRC with intervals.
5. Plot calibration by decile with counts.
6. Report performance by relevant subgroup.

**Branch point.** If discrimination is preserved and calibration is not, recalibration is
candidate. If discrimination is degraded, recalibration will not help and refitting or
rebuilding is required.

**Edge case.** A small target sample. Calibration slope estimates are unstable below
about 100 events, and a "failed" calibration may be noise. Report intervals and avoid
binary pass/fail language.

### Step 8 — Recalibrate or update, then re-test

**Input.** The target validation sample and the chosen update method.

**Output.** An updated model with re-estimated performance, ideally in held-out target
data.

**Rationale.** Updating is a modelling step and must be validated like any other. An
update fitted and evaluated on the same sample is not evidence.

**Procedure.**

1. Split the target sample or use cross-validation.
2. Fit the update on one part: intercept only, intercept and slope, or full refit.
3. Evaluate on the held-out part.
4. Report the update method, the coefficients, and the performance change.
5. State the sample size and the events per variable for any refit.

**Branch point.** If the update requires predictors that are unavailable at the target
time zero, the update is not deployable.

**Edge case.** An update that improves calibration and worsens discrimination, which can
happen with a full refit in a small sample. Prefer the simpler update when performance is
similar.

### Step 9 — Make and document the transfer decision

**Input.** The evidence from Steps 1 to 8.

**Output.** One of five decisions, with conditions and a monitoring plan.

**Rationale.** The decision is a judgement under uncertainty, and it must be recorded so
that it can be revisited when the target data accumulate.

**Decision table.** More than two outputs, and each has a different evidentiary
requirement.

| Evidence pattern | Decision | What must be true | Monitoring |
|---|---|---|---|
| Outcome, time zero, and measurement concordant; calibration slope 0.9-1.1 and intercept near 0 | Adopt | Target validation meets pre-specified criteria | Quarterly calibration check |
| Discrimination preserved, calibration slope near 1, intercept shifted | Recalibrate in the large | Target prevalence differs but predictor effects hold | Recheck after 6 months |
| Discrimination preserved, slope and intercept both off | Recalibrate with slope, or refit | Enough target events to estimate coefficients | Recheck per update |
| Discrimination degraded, measurement concordant | Refit or rebuild | Target events per variable adequate | Full revalidation |
| Measurement or endpoint not concordant | Rebuild | New predictors or new endpoint can be defined | Development study |
| Action or harm profile changed | Reject | No statistical result can justify the new harm | None |
| Target data insufficient to test | Conditional pilot | Prospective data collection with a stopping rule | Pre-specified review |

**Edge case.** A transfer that is statistically adequate and operationally impossible.
The decision is still reject, because adoption requires a workflow that can act on the
output.

The decision record should be short enough to read in a meeting and specific enough to
be falsified. A useful template has six lines: the source model and version, the target
population and decision, the boundary that carries the most risk, the empirical evidence
available today, the decision and its conditions, and the evidence that would reverse
the decision. A record without the last line is a preference, not a decision.

Two failure modes in the documentation are common. The first is a decision recorded
without a version, so that a later reader cannot tell which model was approved. The
second is a conditional approval with no condition that can be evaluated, such as
"monitor performance". A condition should name the metric, the threshold, the cadence,
and the owner.

### Step 10 — Monitor after deployment

**Input.** The deployed model, the target data stream, and the decision rule.

**Output.** A monitoring plan with leading indicators, review cadence, and pre-specified
triggers for suspension.

**Rationale.** Transport is not a one-time test. Devices change, coding changes, case mix
drifts, and the standard of care moves. A model that transported at launch can fail
within a year, and the failure is gradual unless the monitoring is designed to detect it.

**Procedure.**

1. Track the alert rate per 1,000 patients. A change in the alert rate is the earliest
   signal of case-mix or measurement drift.
2. Track the positive predictive value using confirmed outcomes. This requires a
   confirmation pathway, which is also what makes the alert actionable.
3. Track calibration-in-the-large quarterly, or more often if the outcome is common.
4. Track missingness for every predictor. A rise in missingness changes the effective
   model even when the coefficients are unchanged.
5. Track subgroup performance at the same cadence. An aggregate that is stable can hide a
   subgroup that has failed.
6. Pre-specify suspension triggers, such as a PPV below a stated floor for two
   consecutive quarters, and name the person authorised to suspend.

**Branch point.** If the confirmation pathway is incomplete, PPV cannot be measured
directly. Use a sampled audit and state the sampling fraction, rather than substituting
the alert rate for PPV.

**Edge case.** A change in the outcome definition by a payer or regulator. The model's
target moves without any change in the model, and the calibration check will register a
failure that is administrative. Record the definition version alongside the metric.

## 6. Worked example

A vendor has a 30-day mortality model developed in an admitted cardiology population. It
reports AUROC 0.82, sensitivity 0.72, and specificity 0.88 at the recommended threshold,
with a development prevalence of 8 percent. A health system wants to use it in two other
settings: an elective surgical service with an expected prevalence of 1 percent, and an
inpatient oncology service with an expected prevalence of 20 percent.

### 6.1 PPV and NPV at three prevalences

PPV is computed from sensitivity, specificity, and prevalence:

```text
PPV = (sens * p) / (sens * p + (1 - spec) * (1 - p))
NPV = (spec * (1 - p)) / (spec * (1 - p) + (1 - sens) * p)
```

At p = 0.08, the development setting:

```text
PPV = (0.72 * 0.08) / (0.72 * 0.08 + 0.12 * 0.92)
    = 0.0576 / (0.0576 + 0.1104)
    = 0.0576 / 0.1680
    = 0.3429 -> 34.3%
```

At p = 0.01, the surgical service:

```text
PPV = (0.72 * 0.01) / (0.72 * 0.01 + 0.12 * 0.99)
    = 0.0072 / (0.0072 + 0.1188)
    = 0.0072 / 0.1260
    = 0.0571 -> 5.7%
```

At p = 0.20, the oncology service:

```text
PPV = (0.72 * 0.20) / (0.72 * 0.20 + 0.12 * 0.80)
    = 0.1440 / (0.1440 + 0.0960)
    = 0.1440 / 0.2400
    = 0.6000 -> 60.0%
```

| Setting | Prevalence | Sensitivity | Specificity | PPV | NPV |
|---|---:|---:|---:|---:|---:|
| Cardiology (development) | 8% | 0.72 | 0.88 | 34.3% | 97.3% |
| Elective surgery | 1% | 0.72 | 0.88 | 5.7% | 99.7% |
| Inpatient oncology | 20% | 0.72 | 0.88 | 60.0% | 92.6% |

The model is identical in all three rows. The meaning of a positive result is not.

### 6.2 Calibration in the large

The model's mean predicted risk in the development cohort was 8 percent. Suppose the
observed 30-day mortality is 1 percent in the surgical service and 20 percent in the
oncology service. The observed-to-expected ratio is:

```text
Surgical:  O:E = 0.01 / 0.08 = 0.125
Oncology:  O:E = 0.20 / 0.08 = 2.50
```

On the logit scale, calibration-in-the-large is:

```text
Surgical:  logit(0.01) - logit(0.08) = -4.595 - (-2.442) = -2.153
Oncology:  logit(0.20) - logit(0.08) = -1.386 - (-2.442) = +1.056
```

The model over-predicts by a factor of eight in surgery and under-predicts by a factor of
2.5 in oncology. Both are large. Averaged with equal weight, the intercept is
approximately -0.55, which looks like mild miscalibration and is not.

### 6.3 Two diagnostics that reveal a broken analysis

**Diagnostic 1: report PPV only at the development prevalence.** A vendor summary that
says "PPV 34 percent" is true in the development cohort and false everywhere else. The
diagnostic is to recompute PPV at the prevalence of each deployment setting. In the
surgical service, the PPV is 5.7 percent, which means roughly 17 of every 18 alerts are
false. A workflow that responds to every alert with a costly intervention will spend most
of its capacity on patients who would not have died. The size of the error is the ratio
34.3 / 5.7 = 6.0, so the summary overstates the probability that an alert is real by a
factor of six.

**Diagnostic 2: pool the calibration across sites.** A dashboard that combines the three
services reports mean predicted risk near 8 percent and observed risk near 8 percent,
because the over-prediction in surgery and the under-prediction in oncology cancel. The
pooled calibration intercept is close to zero. The diagnostic is to plot calibration by
site. If the site-specific intercepts are -2.15 and +1.06 while the pooled intercept is
-0.55, the pooled statistic is an artefact of aggregation, and the same artefact can hide
a subgroup inversion in discrimination.

### 6.4 Vary one assumption

The PPV calculations above assume that sensitivity and specificity transport unchanged.
That is a strong assumption, and it fails whenever the case mix changes the spectrum of
disease.

Suppose that in the surgical service the model's sensitivity falls to 0.55 because the
patients who die after elective surgery present differently from cardiology admissions,
while specificity stays at 0.88. At a prevalence of 1 percent:

```text
PPV = (0.55 * 0.01) / (0.55 * 0.01 + 0.12 * 0.99)
    = 0.0055 / (0.0055 + 0.1188)
    = 0.0055 / 0.1243
    = 0.0442 -> 4.4%
```

The PPV falls from 5.7 to 4.4 percent. More importantly, the number of alerts needed to
find one true case rises:

```text
At sensitivity 0.72: 1 / 0.0571 = 17.5 alerts per true case
At sensitivity 0.55: 1 / 0.0442 = 22.6 alerts per true case
```

If the service can investigate 15 alerts per week, the model at 5.7 percent PPV would
require about 262 patients screened per true case, and the workload is already beyond
capacity. The transfer decision changes from "recalibrate and pilot" to "do not deploy
at this threshold". The arithmetic, not the AUROC, drives the decision.

### 6.5 Applying the decision table

The three settings resolve differently, and the difference is driven by the workflow, not
by the model.

| Setting | Evidence | Decision | Condition |
|---|---|---|---|
| Cardiology (source) | Development and internal validation | Adopt | Quarterly calibration review |
| Inpatient oncology | Prevalence 20%, O:E 2.50, discrimination untested | Recalibrate with slope after a target sample; refit if discrimination falls | At least 100 target events before refitting |
| Elective surgery | Prevalence 1%, PPV 5.7%, capacity 15 alerts per week | Reject at the current threshold, or pilot at a higher threshold | A higher threshold and a low-harm confirmation step |

The surgery decision deserves the arithmetic. Raising the operating threshold from the
recommended value to a more specific one changes sensitivity and specificity together.
Suppose the higher threshold gives sensitivity 0.40 and specificity 0.96. At a prevalence
of 1 percent:

```text
PPV = (0.40 * 0.01) / (0.40 * 0.01 + 0.04 * 0.99)
    = 0.0040 / (0.0040 + 0.0396)
    = 0.0040 / 0.0436
    = 0.0917 -> 9.2%
```

The PPV rises from 5.7 to 9.2 percent, and alerts per true case fall from 17.5 to 10.9.
That may be enough for a service that can absorb about 11 investigations per true case
and whose confirmation is a chart review rather than a procedure. It is not enough for a
service whose confirmation is invasive. The same model, the same population, and a
different action produce a different decision.

This is the practical content of cross-specialty comparison. The transfer question is
rarely "is the model good". It is "is this model, at this threshold, in this workflow,
with this confirmation pathway, good enough for this action". Four of those five clauses
change when the specialty changes.

### 6.6 What the case does not establish

The case does not show that the model is useless. It shows that the model's output means
something different in each setting and that the deployment decision depends on the
workflow's capacity and the harm of a false positive. A service that can absorb 20 alerts
per week for a low-harm confirmation step may still benefit at a 4.4 percent PPV. A
service whose confirmation is invasive may not benefit at 34 percent.

The case also does not establish transportability of discrimination. Sensitivity and
specificity were assumed, not measured. The next step in a real project is a target
validation sample, and the case is the pre-test that decides whether that sample is worth
collecting.

## 7. Code, query, or calculation

The block below computes PPV, NPV, O:E, calibration-in-the-large, and net benefit across
settings. It is standard-library Python and runs as written.

### 7.1 Transport calculator

```python
"""Cross-specialty transport calculator. Standard library only."""

import math


def ppv(sens, spec, p):
    return (sens * p) / (sens * p + (1 - spec) * (1 - p))


def npv(sens, spec, p):
    return (spec * (1 - p)) / (spec * (1 - p) + (1 - sens) * p)


def logit(x):
    return math.log(x / (1 - x))


def oe(observed, expected):
    return observed / expected


def net_benefit(sens, spec, p, threshold):
    tp = sens * p
    fp = (1 - spec) * (1 - p)
    return tp - fp * (threshold / (1 - threshold))


settings = [
    ("cardiology (development)", 0.08, 0.72, 0.88),
    ("elective surgery", 0.01, 0.72, 0.88),
    ("inpatient oncology", 0.20, 0.72, 0.88),
    ("elective surgery, spectrum shift", 0.01, 0.55, 0.88),
]

print(f"{'setting':34s} {'prev':>5s} {'PPV':>7s} {'NPV':>7s} {'alerts/case':>12s}")
for name, p, se, sp in settings:
    v = ppv(se, sp, p)
    print(f"{name:34s} {p:5.2f} {v:7.3f} {npv(se, sp, p):7.3f} {1 / v:12.1f}")

print()
print("O:E surgery  ", round(oe(0.01, 0.08), 3))
print("O:E oncology ", round(oe(0.20, 0.08), 3))
print("CIL surgery  ", round(logit(0.01) - logit(0.08), 3))
print("CIL oncology ", round(logit(0.20) - logit(0.08), 3))
print("CIL pooled   ", round((logit(0.01) + logit(0.20)) / 2 - logit(0.08), 3))

print()
for p in (0.01, 0.08, 0.20):
    nb_model = net_benefit(0.72, 0.88, p, 0.10)
    nb_all = p - (1 - p) * (0.10 / 0.90)
    print(f"p={p:.2f} net benefit model={nb_model:+.4f} treat-all={nb_all:+.4f}")
```

Expected output:

```text
setting                             prev     PPV     NPV  alerts/case
cardiology (development)            0.08   0.343   0.973          2.9
elective surgery                    0.01   0.057   0.997         17.5
inpatient oncology                  0.20   0.600   0.926          1.7
elective surgery, spectrum shift    0.01   0.044   0.995         22.6

O:E surgery   0.125
O:E oncology  2.5
CIL surgery   -2.153
CIL oncology  1.056
CIL pooled    -0.548

p=0.01 net benefit model=-0.0060 treat-all=-0.1000
p=0.08 net benefit model=+0.0453 treat-all=-0.0222
p=0.20 net benefit model=+0.1333 treat-all=+0.1111
```

The last block is the decision-relevant one. At a 10 percent threshold and a 1 percent
prevalence, the model's net benefit is negative, which means that acting on the model is
worse than acting on no one. At 8 and 20 percent, the model beats both treat-none and
treat-all. The same model, the same sensitivity and specificity, and opposite deployment
conclusions.

Note that the oncology net benefit of 0.1333 is only modestly above treat-all at 0.1111.
At a 20 percent prevalence, treating everyone already captures most of the benefit, and a
model that adds 0.02 in net benefit may not justify its cost, its maintenance, or the
risk of automation bias. The prevalence that makes a model look necessary is also the
prevalence at which the treat-all comparator is strongest.

### 7.2 What the code does not do

The code assumes sensitivity and specificity are known constants. It does not compute
confidence intervals, does not adjust for the sampling variation in prevalence, and does
not model the correlation between the two. It treats net benefit as if the threshold were
fixed, when in practice the threshold is a clinical and economic choice.

The calibration-in-the-large values are computed from a single observed proportion, not
from a fitted model, so they carry the same uncertainty as the observed rate. The pooled
value is an unweighted average of logits, which is not the correct pooled estimate for
unequal sample sizes.

## 8. Data, standards, and regulatory context

### 8.1 Reporting standards

TRIPOD+AI (Collins et al., BMJ, 2024) is the current reporting guideline for clinical
prediction models using regression or machine learning. It has 27 main items and 52
subitems and covers development and evaluation. It deliberately uses "evaluation" rather
than "validation", because a model is never finally validated; it is evaluated in a
stated population at a stated time. Items 16, 20c, 23, and 24 require reporting the
differences between development and evaluation data, the predictor and outcome
distributions in both, performance with confidence intervals and by subgroup, and any
updating.

PROBAST (Wolff et al., Annals of Internal Medicine, 2019) is the risk-of-bias and
applicability tool, with four domains (participants, predictors, outcome, analysis) and
20 signalling questions. PROBAST+AI extends it to machine learning. A meta-review of
PROBAST-based systematic reviews reported high risk of bias most often in the analysis
domain, which is where overfitting, leakage, and inappropriate performance reporting
live.

What these constrain: reporting and appraisal, not the transfer decision itself. What
they do not constrain: whether a given target has the workflow to act on the model.

### 8.2 Data model standards and their limits

FHIR, OMOP, and CDISC make data exchangeable across specialties. They do not make
measurements comparable. A serum creatinine in FHIR is the same structure everywhere and
a different biological signal in a patient with low muscle mass. An ejection fraction in
DICOM is the same structure and a different number across vendors. Interoperability is a
necessary condition for transfer, not a sufficient one.

ICD-10-CM, ICD-10-PCS, and ICD-11 differ in granularity across countries and years. A
cohort definition built on one code set will not port to another without a mapping, and
the mapping error is often larger than the effect being studied. LOINC and SNOMED CT
reduce but do not eliminate the problem, because local mapping decisions remain.

### 8.3 Regulatory context

In the United States, a model with a medical intended use is generally device software,
regulated through 510(k), De Novo, or premarket approval. Clinical decision support that
meets the four criteria in the FDA guidance (not the sole basis for a decision, capable
of independent review, and so on) may be exempt. In the European Union, the Medical
Device Regulation and the In Vitro Diagnostic Regulation set the pathway, and the AI Act
(Regulation (EU) 2024/1689) adds obligations for high-risk systems. The AI Act's
application dates are phased, and the classification depends on the intended use.

The transfer question is regulatory as well as statistical. A model cleared for "adults
admitted to a cardiology service" is not cleared for "all inpatients". Changing the
intended population can change the class, the evidence required, and the post-market
obligations. A transfer memo that does not address this is incomplete.

### 8.4 Estimands

ICH E9(R1) requires the estimand to be defined explicitly: the population, the variable,
the treatment or exposure contrast, the intercurrent event handling, and the population
summary. Two specialties can run the same model on the same data and answer different
estimands because they handle intercurrent events differently. In oncology, subsequent
therapy is an intercurrent event; in critical care, discharge is; in public health,
revision is. A transfer that does not align estimands is comparing different questions.

### 8.5 Coding across specialties

The same clinical concept is coded differently in each specialty, and the differences are
not cosmetic. A cohort definition is a program that reads codes, so a coding difference is
a cohort difference.

| Concept | Oncology | Cardiology | Critical care | Public health |
|---|---|---|---|---|
| Diagnosis | ICD-O-3 morphology plus site | ICD-10-CM | ICD-10-CM, often free text | ICD-10-CM or ICD-11, plus case definition |
| Procedure | CPT plus ICD-10-PCS | CPT plus ICD-10-PCS | ICD-10-PCS, bed-side procedures often uncoded | Rarely used |
| Lab | LOINC plus tumour markers | LOINC | LOINC, unit-dependent | LOINC plus test positivity |
| Stage or severity | AJCC TNM, versioned | NYHA class, ejection fraction | SOFA, APACHE | Not applicable |
| Therapy | Regimen names, J-codes | Drug-eluting stent codes | Infusions, often uncodeable | Vaccination codes |

A transfer that keeps the code list and changes the population will produce a cohort
that is not the intended one. The audit is to take ten patients from the target, abstract
the concept manually, and compare with the code-based definition. A disagreement rate
above the pre-specified tolerance means the definition must be rebuilt for the target.

## 9. Industry practice and case evidence

### 9.1 A widely deployed sepsis model that did not transport

Wong et al. (JAMA Internal Medicine, 2021) externally validated the Epic Sepsis Model at
Michigan Medicine and reported an AUROC of about 0.63, well below the vendor's reported
performance, with poor calibration and a high alert volume. Documented fact: the
independent evaluation found substantially lower discrimination and a large number of
alerts. Inference: widespread deployment is not evidence of validity, and a model
validated at one site cannot be assumed to transport to another. What did not transfer:
the vendor's performance claims, and the assumption that a commercial model's user base
constitutes validation.

### 9.2 A model designed for transport

TREWS (Adams et al., Nature Medicine, 2022) was implemented with a prospective evaluation
that measured outcomes after alerts were acted on within a defined window. Documented
fact: the study reported improved outcomes in the implemented setting. Inference: the
intervention, including the alert and the workflow, was the unit of evaluation, not the
model alone. What did not transfer: the single health system, the evolving model, and the
residual confounding that any before-after design carries.

### 9.3 A device-derived measurement that does not port

AI-ECG models for low ejection fraction (Attia et al., Nature Medicine, 2019) perform well
within the health system where they were developed. Documented fact: the model detected
left ventricular systolic dysfunction from a 12-lead ECG in a Mayo Clinic cohort.
Inference: the signal is real, and the model is a screening pathway for echocardiography
rather than a replacement. What did not transfer: performance across ECG vendors,
lead-placement practices, and populations with different prevalence of structural heart
disease.

### 9.4 A pathology model and the scanner boundary

The CAMELYON16 reader study (Ehteshami Bejnordi et al., JAMA, 2017) showed that deep
learning systems could detect lymph node metastases with performance comparable to
pathologists in a controlled setting. Documented fact: the systems performed at a level
comparable to the pathologists in that study. Inference: the result is a detection task
on curated slides, not a full diagnostic workflow. What did not transfer: scanner
diversity, stain protocols, and the clinical consequence of a false negative in a
screening pathway.

### 9.5 A public health model whose target moved

The COVID-19 Forecast Hub (Cramer et al., PNAS, 2022) demonstrated that an ensemble of
independent models outperformed most individual models for mortality forecasting.
Documented fact: the ensemble had better average scores. Inference: the result depends on
a stable target definition and on continued data collection; when the case definition or
the reporting system changes, the forecast target changes with it. What did not transfer:
the assumption that a forecasting pipeline can be moved to a new pathogen without
re-specifying the target and the delay distribution.

### 9.6 The common pattern

In each case, the failure or success was determined by a boundary in Section 3, not by
the algorithm. The sepsis model failed on case mix and calibration. TREWS succeeded
because it evaluated the workflow. The ECG model is limited by the device boundary. The
pathology model is limited by the scanner and task boundary. The forecast hub is limited
by target definition. None of these is a story about architecture.

The cases also show why a vendor's reference list is weak evidence. A list of sites
proves that the software installed, not that it transported. The evidence that matters is
a site-specific evaluation that the buyer can inspect, with the prevalence, the
calibration, and the alert burden reported for that site's own population. A vendor that
cannot supply it has not done the work, regardless of how many logos appear on the slide.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Assuming prevalence transports | PPV quoted without a prevalence | Development prevalence differs from target | Alert burden and harm misjudged | Recompute PPV at target prevalence |
| Reporting AUROC only | No calibration plot in the validation report | Discrimination is prevalence-invariant | Threshold actions are wrong while audits pass | Require calibration slope and intercept |
| Pooled calibration masking sites | Pooled intercept near zero, site intercepts large | Opposite miscalibration cancels | Deployment looks safe, is not | Report calibration by site |
| Refitting mistaken for validation | Coefficients changed, called external validation | Model updated on the evaluation data | Inflated performance | Separate update and evaluation samples |
| Outcome definition drift | Event rate implausible for the setting | Different ascertainment or window | Model predicts a different event | Compare definitions word by word |
| Time-zero move | Features unavailable at the target moment | Index date differs | Silent leakage or lost signal | Re-audit feature availability |
| Measurement unit error | Predictor distribution shifted by a constant factor | Unit mismatch in the pipeline | Model appears miscalibrated, is actually wrong | Verify units and ranges |
| Scanner or device boundary | Performance drops at one site only | Vendor-specific acquisition | Site-level inequity | Stratify performance by device |
| Spectrum shift | Sensitivity falls, specificity stable | Case mix changed | PPV falls, workload rises | Report performance by severity |
| Endpoint not observable | Large missingness in the target outcome | No protocolised assessment | Cohort restricted to monitored patients | Choose an observable endpoint |
| Regulatory class change | Intended population broadened | New intended use | Unclear legal status | Re-map the pathway before launch |
| Workflow mismatch | High PPV, low action rate | No confirmation pathway or capacity | No clinical effect | Design the pathway with the model |
| Denominator mismatch | Rates differ by a round factor | Population versus patient-days | Comparison invalid | Match numerator and denominator |
| Transfer without monitoring | No post-deployment calibration check | Drift over time | Gradual failure, unnoticed | Quarterly calibration and alert-rate review |

The two structural red flags are a validation report without calibration and a
deployment plan without a prevalence estimate. Either one alone is enough to stop a
transfer review until it is resolved.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| Sensitivity and specificity transport | PPV pre-test uses source values | Spectrum shift changes both | Measure in a target sample |
| Prevalence can be estimated before deployment | PPV is prevalence-dependent | No target data exist | Use a pilot or registry estimate |
| Outcome definitions are concordant | The model predicts a specific event | Ascertainment or window differs | Side-by-side definition audit |
| Time zero is comparable | Features must be admissible | Index date differs | Timestamp audit |
| Measurement instruments are comparable | Predictor distributions must match | Vendor or protocol change | Paired measurement sample |
| Calibration slope of 1 means correct | It is the standard criterion | Case mix differences change slope | Report with intervals |
| Pooled performance represents sites | Aggregation is convenient | Site effects are heterogeneous | Stratify by site |
| Updating improves performance | It is the standard remedy | Small samples cause overfitting | Held-out evaluation |
| The action is available in the target | A prediction needs a response | No confirmation pathway | Workflow mapping |
| Regulatory class is unchanged | It determines the pathway | Intended use broadened | Pre-submission meeting |

The largest limit is that transportability is not a property that can be established once.
A model that transports in 2026 may fail in 2028 after a device change, a coding change,
or a shift in the standard of care. The evidence for transfer is always time-stamped, and
the monitoring plan is part of the model, not an optional extra.

A second limit is that the comparison in this topic is structural, not empirical. The six
specialties were compared on the properties that determine transfer, not on a common
dataset. That is the right level for a decision framework and the wrong level for a
quantitative claim about which specialty is hardest.

A third limit is that the framework assumes the target can be described before the
transfer is evaluated. In practice, the target population is often described by the
vendor, the workflow is described by the clinical champion, and the prevalence is
estimated from a convenience sample. Each of those descriptions is a source of error, and
the transfer decision inherits all of them. The mitigation is to have the target
description signed by someone who owns the operational data, not by someone who owns the
business case.

## 12. Exercises

### Exercise 1 — Build a four-boundary comparison (application, 40 minutes)

**Deliverable.** A table comparing two specialties of your choice on measurement,
endpoint, coding, and workflow, with one transfer risk per cell.

**Acceptance criteria.** Each cell names a specific measurement or code, not a generic
category; each risk states the direction of bias; at least one cell identifies a boundary
that would be fatal.

### Exercise 2 — PPV at deployment prevalence (application, 30 minutes)

**Deliverable.** PPV and NPV for a stated sensitivity and specificity at three
prevalences, with the arithmetic shown.

**Acceptance criteria.** The formula is stated; all three calculations are correct to two
decimal places; the write-up names the number of alerts per true case.

### Exercise 3 — Recalibrate or refit (critique, 35 minutes)

**Deliverable.** A one-page recommendation for a model with AUROC 0.80 in the target,
calibration slope 0.95, and intercept −1.2.

**Acceptance criteria.** The recommendation distinguishes recalibration from refitting;
it states the sample size required; it names what the update cannot fix.

### Exercise 4 — Detect pooled masking (application, 40 minutes)

**Deliverable.** A site-level calibration table and a pooled estimate showing how
opposite miscalibration cancels.

**Acceptance criteria.** The arithmetic is shown on the logit scale; the write-up explains
why the pooled intercept is not a summary of site performance.

### Exercise 5 — Specialty-specific bias audit (design, 45 minutes)

**Deliverable.** An audit plan for the signature bias of one specialty from topics 01 to
06, applied to a model transferred into that specialty.

**Acceptance criteria.** The plan names the bias, the diagnostic, the expected signal, and
the decision rule if the signal appears.

### Exercise 6 — Make the transfer decision (application, 35 minutes)

**Deliverable.** A completed decision-table row for a supplied scenario, with the chosen
option and its conditions.

**Acceptance criteria.** The choice is one of adopt, recalibrate, refit, rebuild, reject;
the conditions are testable; the monitoring plan is specified.

### Exercise 7 — Regulatory remap (design, 30 minutes)

**Deliverable.** A memo mapping a transferred product's intended use and regulatory class
before and after transfer.

**Acceptance criteria.** The memo names the jurisdiction and pathway; it identifies
whether the class changes; it states the evidence the new class requires.

### Exercise 8 — Transportability memo (design, 60 minutes)

**Deliverable.** A two-page memo for a clinical champion and a regulator covering source,
target, boundary analysis, empirical evidence, decision, and monitoring.

**Acceptance criteria.** Every claim is sourced or labelled as inference; the decision
follows from the evidence; the memo states what would change the decision.

## 13. Self-check answers

**1. Four boundaries.** Measurement, endpoint, coding, and workflow. Case mix and
spectrum sit inside measurement and population; regulation follows from workflow and
intended use.

**2. PPV at 1 percent.** With sensitivity 0.72 and specificity 0.88, PPV is
(0.72 × 0.01) / (0.72 × 0.01 + 0.12 × 0.99) = 0.0072 / 0.1260 = 5.7 percent. NPV is
(0.88 × 0.99) / (0.88 × 0.99 + 0.28 × 0.01) = 0.8712 / 0.8740 = 99.7 percent.

**3. Recalibrate or refit.** A calibration slope near 1 with a shifted intercept indicates
that the predictor effects are preserved and the average risk is wrong. Recalibration in
the large is sufficient. Refitting is not indicated and risks overfitting.

**4. Pooled masking.** Site intercepts of −2.15 and +1.06 average to about −0.55, which
looks like mild miscalibration. The pooled value is an artefact of cancellation and
should never be reported without site-level values.

**5. Signature biases.** Oncology: immortal time and stage migration. Cardiology:
prevalence-dependent PPV and device-specific signal. Pathology: slide and scanner
leakage, label noise. Critical care: label leakage and competing risks. Paediatrics and
obstetrics: weight-based dosing, rare events, and maternal-fetal confounding. Public
health: denominators and surveillance lag.

**6. Transfer decision.** If discrimination is preserved and calibration is off,
recalibrate. If discrimination is degraded with concordant measurement, refit or rebuild.
If the measurement or endpoint is not concordant, rebuild. If the action or harm profile
changed, reject.

**7. Regulatory class.** A model that moves from a detection aid to a diagnostic claim,
or from a clinician-facing support tool to an autonomous alert, can change class. The
intended use, not the algorithm, determines the pathway.

**8. What would change the decision.** A target validation showing preserved
discrimination, acceptable calibration after update, and a workflow that can absorb the
alert volume. The memo should state this in advance, so the decision is falsifiable.

**Rubric for Exercises 5, 7, and 8.** A strong answer names the mechanism, quantifies the
consequence, and states the condition under which the decision would change. A weak
answer lists generic limitations without connecting them to the transfer.

A note on marking the arithmetic items. The correct PPV at 1 percent prevalence is 5.7
percent, and an answer of 34 percent indicates that the development prevalence was used.
That error is worth more feedback than an arithmetic slip, because it is the error the
topic exists to prevent. Mark the reasoning, not only the number.

## 14. Interview questions

**1. What is the difference between external validation and transportability?**
Strong answer: external validation is an empirical evaluation in new data;
transportability is a judgement about whether that result will hold in a stated target,
which depends on the four boundaries. Weak answer: treats them as synonyms.

**2. A model has AUROC 0.85 in the target. Can you deploy it?**
Strong answer: not without calibration and a prevalence-specific PPV. Weak answer: yes,
because discrimination is high.

**3. When is recalibration enough?**
Strong answer: when the slope is near 1 and the intercept is off, indicating a prevalence
shift rather than a change in predictor effects. Weak answer: "always try recalibration
first".

**4. Why can pooled calibration be misleading?**
Strong answer: opposite site-level miscalibration can cancel, producing a near-zero
pooled intercept. Weak answer: "pooling always loses information".

**5. Which bias would you audit first when moving a model from ICU to the ward?**
Strong answer: label leakage and time-zero differences, then case mix and monitoring
intensity. Weak answer: "overfitting".

**6. How does prevalence affect the value of a screening model?**
Strong answer: PPV falls with prevalence at fixed sensitivity and specificity, raising
alerts per true case and changing the workflow's capacity requirement. Weak answer:
"prevalence does not affect the model".

**7. When does transfer change the regulatory class?**
Strong answer: when the intended use, population, or claim changes, for example from
detection aid to diagnosis or from clinician support to autonomous action. Weak answer:
"never, the algorithm is the same".

**8. How would you validate a model across ten hospitals?**
Strong answer: site-specific calibration and discrimination, case-mix description,
pre-specified pooling, and a plan for sites with too few events. Weak answer: pool
everything and report one AUROC.

**9. What is spectrum bias?**
Strong answer: performance changes because the case mix and disease severity
distribution change, affecting sensitivity and specificity. Weak answer: "the data are
different".

**10. What would make you reject a transfer outright?**
Strong answer: a changed harm profile with no confirmation pathway, or an endpoint that
cannot be observed in the target. Weak answer: "a lower AUROC".

## 15. Key terms

- **Transportability** — whether a model's performance carries to a new population or setting.
- **External validation** — evaluation of a fixed model in data not used for development.
- **Temporal validation** — evaluation in a later period at the same site.
- **Geographical validation** — evaluation at a different site.
- **Spectrum bias** — performance change caused by a change in case mix.
- **Case mix** — distribution of severity, comorbidity, and presentation.
- **Prevalence** — proportion with the outcome in the target population.
- **PPV** — probability that a positive result is a true positive.
- **NPV** — probability that a negative result is a true negative.
- **Calibration** — agreement between predicted probabilities and observed frequencies.
- **Calibration slope** — regression slope of outcome on the linear predictor.
- **Calibration intercept** — calibration-in-the-large on the logit scale.
- **O:E ratio** — observed divided by expected events.
- **Recalibration** — adjusting intercept and slope without changing predictors.
- **Refitting** — re-estimating predictor coefficients in target data.
- **Rebuilding** — developing a new model with target-specific predictors.
- **Discrimination** — ability to rank patients by risk.
- **AUROC** — area under the receiver operating characteristic curve.
- **AUPRC** — area under the precision-recall curve.
- **Net benefit** — decision-analytic value at a threshold probability.
- **Decision curve** — net benefit plotted across thresholds.
- **Intended use** — stated purpose, population, user, and setting.
- **Dataset shift** — change in the joint distribution of inputs and outcome.
- **Label shift** — change in outcome prevalence with stable conditional distributions.
- **Estimand** — precise quantity a study estimates.
- **TRIPOD+AI** — reporting guideline for prediction models, 2024.
- **PROBAST** — risk-of-bias and applicability tool for prediction models.
- **Calibration-in-the-large** — the intercept term on the logit scale.
- **Number needed to alert** — alerts required to find one true case.
- **Alert burden** — alerts per unit of clinical activity.
- **Conditional approval** — deployment allowed only under stated, testable conditions.
- **Boundary analysis** — structured comparison of source and target on the six boundaries.

## 16. Sources

- Collins GS, Moons KGM, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. BMJ. 2024;385:e078378. https://pmc.ncbi.nlm.nih.gov/articles/PMC11019967/ (accessed 26 October 2026).
- Wolff RF, Moons KGM, Riley RD, et al. PROBAST: a tool to assess the risk of bias and applicability of prediction model studies. Ann Intern Med. 2019;170(1):51–58. https://doi.org/10.7326/M18-1376 (accessed 26 October 2026).
- Moons KGM, Wolff RF, Riley RD, et al. PROBAST: explanation and elaboration. Ann Intern Med. 2019;170(1):W1–W33. https://doi.org/10.7326/M18-1377 (accessed 26 October 2026).
- Riley RD, et al. Evaluation of clinical prediction models, parts 1–3. BMJ. 2024;384:e074819, e074820, e074821. https://doi.org/10.1136/bmj-2023-074819 (accessed 26 October 2026).
- Wong A, et al. External validation of a widely implemented proprietary sepsis prediction model in hospitalized patients. JAMA Intern Med. 2021. https://doi.org/10.1001/jamainternmed.2021.2626 (accessed 26 October 2026).
- Adams R, et al. Prospective, multi-site study of patient outcomes after implementation of the TREWS machine learning-based early warning system for sepsis. Nat Med. 2022. https://doi.org/10.1038/s41591-022-01894-0 (accessed 26 October 2026).
- Attia ZI, et al. Screening for cardiac contractile dysfunction using an artificial intelligence-enabled electrocardiogram. Nat Med. 2019. https://doi.org/10.1038/s41591-018-0240-2 (accessed 26 October 2026).
- Ehteshami Bejnordi B, et al. Diagnostic assessment of deep learning algorithms for detection of lymph node metastases in women with breast cancer. JAMA. 2017. https://doi.org/10.1001/jama.2017.14585 (accessed 26 October 2026).
- Cramer EY, et al. Evaluation of individual and ensemble probabilistic forecasts of COVID-19 mortality in the United States. PNAS. 2022. https://doi.org/10.1073/pnas.2112532119 (accessed 26 October 2026).
- US Food and Drug Administration. Clinical decision support software guidance. https://www.fda.gov/regulatory-information/search-fda-guidance-documents/clinical-decision-support-software (accessed 26 October 2026).
- European Union. Artificial Intelligence Act, Regulation (EU) 2024/1689. https://eur-lex.europa.eu/eli/reg/2024/1689/oj (accessed 26 October 2026).
- ICH. E9(R1) addendum on estimands and sensitivity analysis. https://www.ich.org/page/efficacy-guidelines (accessed 26 October 2026).

## 17. Links

- Module index: [`_index.md`](_index.md)
- Survey file: [`../../course/23_specialty_electives.md`](../../course/23_specialty_electives.md)
- Previous topic: [`06_public-health.md`](06_public-health.md)
- First topic: [`01_oncology.md`](01_oncology.md)
- Measurement foundations: [`02_cardiology.md`](02_cardiology.md), [`03_pathology.md`](03_pathology.md)
- Labs and datasets: [`../../labs/`](../../labs/)
- Regulation: [`../14_devices_regulation/`](../14_devices_regulation/)
- Quality and validation: [`../18_quality/`](../18_quality/)
