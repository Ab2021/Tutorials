# Labels and Outcomes

> **Module 06 — Machine Learning Foundations**
> Topic 02 of 13
> Estimated study time: 3.5 hours
> Prerequisites: Topic 01
> Version 2026.3

## 1. Why this topic exists

A supervised model learns to reproduce the label it is given. If the label is a poor stand-in for
the clinical outcome, the model learns the label's error structure as faithfully as it learns the
biology. The result is a model that performs well against the label and poorly against the patient.
This is not a theoretical concern. Sepsis, acute kidney injury, readmission, and deterioration are
all routinely operationalised with diagnosis codes whose sensitivity is far below what a clinical
team would assume, and the resulting models inherit that gap.

The problem is structural, not a matter of careless coding. The clinical record is generated for
billing, communication, and legal documentation, not for measurement. A clinician documents what
was considered, what was ruled out, and what the service was paid for. A diagnosis code may be
assigned to justify a treatment, to capture a comorbidity, or to satisfy a quality measure. The
timestamp of a code reflects when it was entered, not when the disease began. A laboratory value
is measured because someone ordered it, and the ordering decision is itself a function of the
patient's severity, the clinician's suspicion, and the time of day. Every one of these processes
leaves a signature in the label, and a model trained without understanding them will learn the
signature.

The consequences are concrete. A sepsis model trained on ICD-coded sepsis learns to predict
whether a clinician will document sepsis, which depends on the clinician's specialty, the shift,
and the hospital's coding culture. A readmission model trained on claims data learns to predict
which readmissions are billable to the index hospital, not which patients will return. An acute
kidney injury model trained on creatinine criteria learns to predict who gets a creatinine test,
because patients without a test cannot meet the criterion. In each case the model may discriminate
well against the label and still fail the clinical decision, and the failure is invisible unless
the label itself has been validated.

Label problems also interact with equity. If a group is less likely to be tested, less likely to
receive a diagnosis code, or more likely to be admitted to a hospital that codes differently, the
label is differentially ascertained. A model trained on that label can appear to perform equally
well across groups while learning a different decision boundary for each. Obermeyer et al. (2019)
showed this for a cost proxy: at the same risk score, Black patients were sicker because the label
tracked spending rather than need. The label choice was the mechanism of bias, and no amount of
fairness-aware model tuning would have fixed it.

This topic treats the label as an object of study with its own measurement properties. You will
learn to define a construct, operationalise it, validate the operational definition against a
reference standard, quantify the error, propagate that error into downstream metrics, and monitor
the label after deployment. The output is a label specification and a validation report that can
be attached to the model development report.

The label is also a moving target. Coding guidelines change annually, laboratory assays are
replaced, documentation templates are redesigned, and hospitals merge. A label validated in 2022
may be invalid in 2026 for reasons that have nothing to do with the model. Teams that treat the
label as a fixed asset discover the drift only when performance falls, and they usually blame the
model. The monitoring plan in Step 13 exists to catch this earlier.

Three questions separate a label that is fit for purpose from one that is merely available. First,
what is the error direction, and does it matter for this decision? A label that misses mild cases
may be adequate for a rule-in decision and inadequate for a rule-out decision. Second, is the error
differential across the groups the model will serve? A label that is 80 percent sensitive overall
but 60 percent sensitive in one group will produce apparently unequal model performance even if the
model is equally good. Third, is the label stable over the deployment period? A label that shifts
with a coding update will make a stable model look unstable. If the team cannot answer these three
questions, it does not yet know what it is predicting.

> **Note:** A label is not the truth. It is a measurement of the truth, with sensitivity,
> specificity, and a process that generated it. Every downstream metric inherits those properties.

## 2. Learning outcomes

By the end of this topic you can:

1. **Distinguish** a clinical construct from its operational label and **state** the gap between
   them in one paragraph.
2. **Write** a computable label definition with code sets, timing, washout, and competing-event
   rules.
3. **Design** a chart-review validation sample that avoids verification bias, including sampling,
   blinding, and adjudication.
4. **Compute** label sensitivity, specificity, PPV, NPV, and kappa with confidence intervals from a
   two-by-two table.
5. **Correct** an observed prevalence for label error using the Rogan-Gladen estimator, and
   **explain** when the correction is unstable.
6. **Quantify** the attenuation of a model's sensitivity and specificity when the reference label
   is imperfect.
7. **Detect** differential ascertainment across sites, time periods, and subgroups.
8. **Design** a label monitoring plan that triggers revalidation when coding practice or testing
   practice changes.

Outcomes 4, 5, and 6 are computational and are assessed in Exercises 2, 3, and 7. Outcomes 3 and
8 are design outcomes assessed by rubric.

The outcomes are deliberately weighted toward measurement rather than modelling. A learner who can
fit a classifier but cannot say what its label measures is not ready to deploy a clinical model.
A learner who can characterise a label but cannot fit a model is ready to contribute to a
multidisciplinary team, because the measurement question is the one that determines whether the
model's results mean anything.

## 3. The core mental model

A label is produced by a chain that starts with a patient state and ends with a value in a
database. Each link is a measurement or a decision, and each link can distort the signal.

```text
Patient state          Clinical action           Documentation          Data capture          Label
(true outcome)    ->   (test, diagnosis,    ->   (note, order,      ->  (code, lab value, ->  (value used
                        admission)                problem list)         registry)             by the model)
     |                      |                        |                     |                    |
     |                      |                        |                     |                    |
   severity            test-ordering            clinician              coding rules,        sensitivity,
   biology             thresholds,              attention,             dictionaries,        specificity,
   timing              access                   language               interfaces           PPV, NPV
```

Read left to right, the chain is ascertainment. Read right to left, it is inference: given the
label, what can be said about the patient state? The model is trained on the rightmost value and
deployed to act on the leftmost state. The distance between the two is the label gap.

Four properties of the chain matter.

**The label is a measurement with operating characteristics.** Treat it as a diagnostic test for
the construct. It has sensitivity (the probability that a true case is labelled positive) and
specificity (the probability that a true non-case is labelled negative). These are properties of
the ascertainment process, not of the model. A label with sensitivity 0.35 and specificity 0.98,
which is roughly what meta-analysis finds for ICD-coded sepsis, is a rule-in test: a positive code
is informative, a negative code is nearly uninformative.

**The measurement error is usually differential.** It varies by site, era, specialty, payer, and
patient group. A label that is 80 percent sensitive overall may be 90 percent sensitive at a
teaching hospital and 55 percent sensitive at a community hospital. A model trained across sites
will learn the site-specific documentation pattern as a feature if site or its correlates are
available, and will fail when the pattern changes.

**The label defines the population as well as the outcome.** Requiring a creatinine test to define
acute kidney injury restricts the cohort to patients who were tested. Requiring a diagnosis code
restricts the cohort to patients who were coded. The restriction is not random, and it changes
prevalence, case mix, and the meaning of a negative. This is selection at the label level, and it
is distinct from selection at the cohort level.

**The label has a time structure.** The outcome window, the washout period for prior episodes, the
distinction between incident and prevalent cases, and the handling of competing events all change
what the label means. A readmission label with a 30-day window and no washout counts a planned
return for a staged procedure as an unplanned readmission. A sepsis label anchored to the first
blood culture counts a culture drawn for a different reason as onset. The time structure must be
written down, because two teams using the same code set with different windows will produce
different labels and incomparable models.

The chain is not a criticism of clinicians or coders. It is a description of how clinical data are
produced. A record that supports care, billing, and legal defence will not also be a research
instrument unless someone designs the measurement deliberately.

A useful way to remember the model is the "two-by-two inside the dataset" idea. For every patient
there is a true state (case or not) and an observed label (positive or not). The dataset contains
only the observed column. The validation sample is the small subset for which both columns exist.
Everything the team knows about label quality comes from that subset, and the credibility of the
whole project depends on how that subset was chosen.

### Four kinds of label error

The distinction matters because the remedies differ.

| Error type | Mechanism | Effect on model metrics | Remedy |
|---|---|---|---|
| Random misclassification | Label flipped independently of patient and model | Attenuates discrimination and calibration toward chance | Larger sample; better label |
| Systematic under-ascertainment | True cases missed at a constant rate | Sensitivity underestimated; prevalence underestimated | Correction with validated sensitivity |
| Differential ascertainment | Error varies by group, site, or era | Apparent subgroup differences; unfairness | Stratified validation; site-specific correction |
| Temporal misalignment | Label timestamp before or after the true event | Leakage or misclassified horizon | Timing audit; re-anchor to clinical event |

Random misclassification is the least damaging and the easiest to describe. Systematic and
differential errors are more dangerous because they can look like real clinical signal. Temporal
misalignment is the most dangerous because it can produce impressive performance that disappears
at deployment. A label validation report should name which of the four types it has assessed.

> **Note:** If the validation sample is drawn only from patients who tested positive, the label's
> sensitivity cannot be estimated. The sample design is part of the label definition.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Construct | The clinical phenomenon of interest, independent of how it is measured | Treated as identical to the label | Conceptual definition |
| Operational definition | The rules that convert data into a label | Written after seeing the data | Code set plus logic |
| Reference standard | The best available ascertainment of the construct, used to validate the label | Assumed to be perfect | Adjudication protocol |
| Label | The value used as the supervised target | Called "ground truth" | Binary, ordinal, or time-to-event |
| Proxy label | A label that stands in for a construct it does not directly measure | Presented as the construct | Named proxy plus validation |
| Ascertainment | The process by which a patient comes to be labelled | Confused with coding | Process description |
| Computable phenotype | An algorithm that identifies a clinical condition from EHR data | Assumed portable across sites | Executable definition |
| Code set | The diagnosis, procedure, or medication codes used in a label | Version not recorded | ICD-10-CM, SNOMED CT, LOINC, RxNorm |
| Chart review | Manual abstraction of the clinical record as a reference standard | Single reviewer without blinding | Protocol plus form |
| Adjudication | Resolution of disagreements between reviewers | Majority vote without discussion | Committee plus rule |
| Inter-rater agreement | Agreement between reviewers beyond chance | Reported as accuracy | Cohen's kappa, Krippendorff's alpha |
| Label sensitivity | Probability a true case is labelled positive | Confused with model sensitivity | Proportion |
| Label specificity | Probability a true non-case is labelled negative | Confused with model specificity | Proportion |
| Label PPV | Probability a labelled positive is a true case | Treated as fixed across prevalences | Proportion |
| Label NPV | Probability a labelled negative is a true non-case | Assumed high without checking | Proportion |
| Verification bias | Bias when the reference standard is obtained only for a selected subset | Ignored when sampling is enriched | Design flaw |
| Differential ascertainment | Label error that varies by group, site, or time | Detected only by subgroup validation | Stratified metrics |
| Label noise | Random or systematic error in the label | Assumed to be random | Misclassification rates |
| Label leakage | Outcome information present in features | Confused with label noise | Audit finding |
| Surveillance bias | More testing or monitoring in one group produces more labels | Mistaken for higher incidence | Detection-rate difference |
| Length-time bias | Screening preferentially detects slow-progressing cases | Mistaken for survival benefit | Screening evaluation |
| Lead-time bias | Earlier detection appears to prolong survival without changing course | Mistaken for benefit | Screening evaluation |
| Immortal time bias | Follow-up time during which the outcome cannot occur is misclassified | Common in treatment labels | Time-to-event analysis |
| Competing risk | An event that prevents the outcome of interest | Treated as censoring | Cause-specific hazard |
| Censoring | End of observation before the outcome is known | Assumed non-informative | Administrative or clinical |
| Outcome window | Time interval after time zero in which the outcome counts | Chosen for convenience | Days |
| Washout period | Interval before time zero used to exclude prior episodes | Omitted | Days |
| Incident case | A new episode after a disease-free interval | Confused with prevalent case | Definition plus lookback |
| Prevalent case | An existing condition at time zero | Included in incidence labels | Definition plus lookback |
| Lookback window | Period before time zero searched for history | Too short misses chronic disease | Days |
| Data cutoff | Last date with complete outcome ascertainment | Ignored for recent patients | Date |
| Informative missingness | Missingness that depends on the outcome or its predictors | Treated as random | Mechanism analysis |
| Composite endpoint | A label combining several outcomes | Components reported separately only after the fact | Definition plus hierarchy |
| Weak supervision | Labels generated by rules or heuristics rather than adjudication | Called validated | Rule plus validation sample |
| Silver label | A probabilistic or rule-based label used at scale | Treated as gold | Probability plus error model |
| Label shift | Change in the relationship between label and construct after deployment | Confused with covariate shift | Monitoring metric |
| Positive predictive value of the label | Synonym for label PPV | Used interchangeably with precision | Proportion |
| Negative predictive value of the label | Synonym for label NPV | Assumed | Proportion |
| Reference-standard bias | Error in the reference standard itself | Assumed away | Adjudication quality |
| Spectrum bias | Label performance differs by case severity | Ignored in validation sampling | Stratified metrics |
| Case definition | The complete rule for identifying a case | Only the code list | Document |
| Control definition | The rule for identifying a non-case | Absence of the code | Document |
| Eligibility algorithm | The rule for including a patient at all | Confused with case definition | Document |
| Phenotype portability | Whether a phenotype performs similarly across sites | Assumed from development site | Multi-site validation |
| Label drift | Change in label prevalence or error over time | Mistaken for clinical change | Time series |

Three terms need more than a table row.

**Construct versus label.** The construct is what the clinician means by "sepsis": a
life-threatening organ dysfunction caused by a dysregulated host response to infection. The label
might be "ICD-10 code A41.9 present in the first three diagnosis fields". The two are related but
not identical. A model trained on the label learns the coding behaviour associated with the
construct, including the behaviour of clinicians who code suspected sepsis before confirmation.
The validation question is not "is the code correct?" but "how well does the code measure the
construct in this population, and for what decisions is that good enough?"

**Reference-standard bias.** Chart review is often treated as ground truth, but reviewers disagree
with each other and with the underlying biology. A two-reviewer adjudication with a kappa of 0.75
is not a perfect standard. The validation report should state the reference standard's own
reliability and, where possible, the direction of its likely error. If the reference standard is
more sensitive than the label, the label's sensitivity will be underestimated; if it is less
sensitive, the opposite.

**Verification bias.** If the validation sample is enriched for label-positive patients (a common
practical choice because true cases are rare), PPV and NPV can be estimated from the enriched
sample with weights, but sensitivity and specificity cannot be estimated without reweighting or
correcting. Reporting naive sensitivity from an enriched sample is one of the most common errors in
phenotype validation reports. The sample design determines which metrics are estimable.

> **Note:** Every metric in this topic is reported with the prevalence of the construct in the
> validation sample and the sampling design. A label sensitivity of 0.78 means little without
> knowing whether the sample was random, enriched, or restricted.

## 5. The mechanics, step by step

Label development is a measurement project nested inside the prediction project. It has thirteen
steps and three artefacts: the label specification, the validation report, and the monitoring plan.
Steps 1 through 7 can be completed before any model is fitted. Steps 8 through 11 require access to
the chart and a reviewer with clinical training. Steps 12 and 13 connect the label to the model and
to deployment.

### Step 1. State the construct and the decision it serves

**Input:** the prediction task specification from Topic 01.

**Output:** a two-sentence construct definition: what the clinical phenomenon is, and what decision
the label must support. Example: "Acute kidney injury is a sudden decline in renal function
meeting KDIGO criteria. The label must support a decision to order a nephrology consult within 24
hours."

**Rationale:** The construct determines which measurement is adequate. A label used to trigger a
nephrology consult needs to capture clinically significant injury, not every transient creatinine
bump. A label used for epidemiological incidence needs to capture all episodes, including
subclinical ones. The same code set can be adequate for one decision and inadequate for another.

**Branch point:** if the decision is a rule-out (for example, avoiding a contrast study), the label
must be sensitive; a false negative is the costly error. If the decision is a rule-in (for example,
starting dialysis), the label must be specific. State which error dominates before choosing the
ascertainment source.

### Step 2. Choose the ascertainment strategy

**Input:** the construct, the available data sources, and the decision's error asymmetry.

**Output:** a primary source and a reference standard, with a justification for each.

**Rationale:** Different sources have different error structures, and no source is uniformly best.
Diagnosis codes are cheap and specific but insensitive; laboratory values are reproducible but
conditional on testing; chart review is accurate but expensive and reviewer-dependent; registries
have validated definitions but limited coverage. The choice should follow the decision, not the
convenience of the data warehouse.

The decision table below summarises the common options.

| Source | Use when | Strength | Limitation |
|---|---|---|---|
| Diagnosis codes | Large cohorts, administrative data, rule-in labels | Cheap, specific, standardised | Low and variable sensitivity; coding drift |
| Laboratory values | An objective threshold defines the construct | Reproducible, timestamped | Testing selection; assay and unit changes |
| Chart review | Reference standard for a validation sample | Captures clinical reasoning | Cost, reviewer bias, non-scalable |
| Disease registry | A validated registry covers the population | Standard definitions, follow-up | Coverage and consent bias |
| Claims data | Multi-site capture with long follow-up | Complete within payer | Billing incentives; out-of-network gaps |
| Device or wearable signal | Continuous physiological outcome | Temporal resolution | Artifact, adherence, calibration |
| Patient-reported outcome | Symptoms and function matter to the patient | Captures lived experience | Recall bias, missingness |
| Death registry | Mortality is the outcome | Complete and objective | Cause-of-death attribution is weak |

The source choice is also a governance choice. A code-based label can be recomputed by an auditor
from the same claims extract; a chart-review label depends on access to records and on reviewer
judgement that may not be reproducible years later. If the model will be regulated or audited,
prefer a label whose provenance can be reconstructed, and keep the validation sample identifiers
so the review can be repeated.

**Branch point:** if no single source covers the construct, consider a composite label with an
explicit hierarchy (for example, laboratory criterion OR chart-review confirmation). Composites
must state whether any component suffices or all are required, and must report component-level
performance.

### Step 3. Draft the case definition

**Input:** the chosen source and the construct.

**Output:** a computable rule with code sets, value thresholds, logical operators, and timing
constraints. Example: "ICD-10-CM N17.0, N17.1, N17.2, N17.9 in any diagnosis field, OR a creatinine
rise of at least 0.3 mg/dL within 48 hours, OR a rise to at least 1.5 times baseline within 7
days."

**Rationale:** The case definition is the operational core of the label. It must be specific enough
that two analysts obtain the same result from the same data, and transparent enough that a
clinician can see what is being counted. Code sets should be listed by version; a code that was
valid in FY2022 may be retired or split in FY2025.

**Branch point:** if the definition combines codes and laboratory values, decide whether they are
ORed or ANDed, and whether the code can substitute for a missing lab. An OR rule increases
sensitivity and decreases specificity; an AND rule does the opposite. Report both versions if the
choice is contested.

### Step 4. Draft the control and eligibility definitions

**Input:** the case definition and the cohort specification.

**Output:** rules for who is a non-case and who is excluded entirely.

**Rationale:** A label has two halves. The control definition determines specificity, and the
eligibility definition determines the population in which the label is meaningful. "Absence of the
code" is a weak control definition because it conflates "did not have the condition" with "was not
tested" and "was not coded". A stronger control requires evidence of the relevant tests or
encounters.

**Branch point:** if the control definition requires a test, the cohort is restricted to tested
patients, and the label becomes a label of tested-and-negative rather than of disease-free. That
restriction must be stated, because it changes prevalence and transportability.

### Step 5. Fix the outcome window and washout

**Input:** the clinical course of the construct.

**Output:** a start time, an end time, and a lookback period before the start.

**Rationale:** The window determines which events count. A 30-day readmission window excludes
day-31 returns; a 7-day AKI window excludes slower declines. The washout period excludes patients
with a prior episode, converting prevalence into incidence. Without a washout, a patient with
chronic kidney disease may be counted as a new AKI case on every admission.

**Branch point:** if the outcome can recur, decide whether the unit of analysis is the patient, the
admission, or the episode. Recurrent events require a model that accounts for within-patient
correlation, and the label definition must state the unit.

### Step 6. Handle competing events and censoring

**Input:** the outcome window and the clinical reality that some patients die or leave the system.

**Output:** a rule for deaths, transfers, discharges to hospice, and loss to follow-up.

**Rationale:** A patient who dies before the outcome window closes cannot have a readmission. If
deaths are treated as censoring, the analysis assumes the patient could have been readmitted,
which is false. If deaths are treated as non-events, the label undercounts the severity of the
course. The choice depends on the decision: for a resource-planning model, death is a competing
event; for a quality metric, it may be an exclusion.

**Branch point:** if the competing event is common (more than roughly 10 percent of the cohort),
a naive binary label will misstate the outcome for a meaningful fraction of patients. Consider a
composite endpoint or a cause-specific analysis, and state the choice.

### Step 7. Build and test the extraction

**Input:** the case, control, and eligibility rules.

**Output:** a query or script, a count of labelled patients, and a set of unit tests on known
cases.

**Rationale:** Extraction bugs are common and silent. A join that duplicates rows, a timezone
mismatch, a unit conversion error, or a code set that includes history codes will change the label
without changing the code's appearance. Unit tests should include patients known to be cases,
known non-cases, and edge cases such as a single creatinine value or a code entered after
discharge.

**Branch point:** if the extraction produces a prevalence far outside the expected range, treat it
as a bug until proven otherwise. A sudden prevalence change after a database migration is almost
always a data issue, not a clinical change.

### Step 8. Design the validation sample

**Input:** the cohort size, the expected prevalence, and the precision required.

**Output:** a sampling design, a sample size, and a weighting scheme.

**Rationale:** The sample design determines which metrics are estimable. A simple random sample
supports sensitivity, specificity, PPV, and NPV. An enriched sample (all positives plus a random
sample of negatives) supports PPV and specificity well but requires weighting for sensitivity. A
sample drawn only from positives supports PPV only. The sample size should be chosen so that the
confidence interval around the metric of interest is narrow enough for the decision; a common
practical target is at least 100 cases and 100 non-cases in the validation sample, though the
required number depends on the precision needed.

A rough sample-size rule for a proportion uses `n = z^2 * p * (1-p) / e^2`. For a sensitivity near
0.8, a 95 percent interval half-width of 0.05 requires about `1.96^2 * 0.8 * 0.2 / 0.05^2 = 246`
cases; a half-width of 0.10 requires about 61 cases. Because sensitivity is estimated among true
cases, the number of cases in the validation sample, not the total sample, is the relevant
quantity. If the construct has 33 percent prevalence, 246 cases require about 745 reviewed
patients. That is why full-cohort review is common for rare constructs in moderate-sized cohorts.

**Branch point:** if the cohort is small (fewer than about 200 patients), review all of them
rather than sampling. If the cohort is large but the construct is rare, enrich on positives and
weight the estimates, and state the weighting.

### Step 9. Abstract with blinding and adjudication

**Input:** the validation sample.

**Output:** a structured abstraction form, reviewer assignments, and adjudication records.

**Rationale:** Reviewers who know the label value are biased toward confirming it. Blinding to the
label, using a structured form, and adjudicating disagreements with a third reviewer reduces this
bias. Inter-rater agreement should be reported, because a reference standard with low agreement
cannot support a precise estimate of label performance.

**Branch point:** if agreement is low (kappa below roughly 0.6), the construct or the abstraction
form is ambiguous. Revise the form and re-review a subsample before proceeding; do not average
disagreements into a false precision.

### Step 10. Estimate label operating characteristics

**Input:** the two-by-two table from the validation sample.

**Output:** sensitivity, specificity, PPV, NPV, and kappa with confidence intervals, plus the
sampling design and prevalence.

**Rationale:** These are the measurement properties that propagate into every downstream metric.
Reporting them without intervals invites overconfidence, especially in small samples. Wilson
intervals are preferable to normal-approximation intervals when counts are small or near 0 or 1.

The validation report should also state whether the estimates are design-weighted. In a simple
random sample, no weighting is needed. In a stratified or enriched sample, each reviewed patient
represents a number of patients in the full cohort, and the weights must be applied before the
metrics are computed. Reporting unweighted metrics from a stratified sample is a common error that
makes the label look more sensitive than it is, because the enriched sample over-represents
positives.

**Branch point:** if any cell is zero, the corresponding metric is 0 or 1 with a wide interval.
Do not report a point estimate of 1.0 as if it were certain; report the interval and the sample
size. If both sensitivity and specificity are estimated from an enriched sample without weighting,
label the estimates as biased.

### Step 11. Test differential ascertainment

**Input:** the validation sample with site, era, and subgroup identifiers.

**Output:** stratified label metrics with counts and intervals.

**Rationale:** A label that is 78 percent sensitive overall may be 85 percent sensitive at one
site and 62 percent at another. If the model is deployed across sites, the site with lower label
sensitivity will show lower observed model performance even if the model is equally good. Testing
differential ascertainment separates label artefacts from model artefacts.

**Branch point:** if a subgroup has fewer than about 20 cases, report the counts and refrain from
a subgroup-specific conclusion. Pool across related subgroups or collect more data.

### Step 12. Propagate the error into downstream metrics

**Input:** the label operating characteristics and the model's confusion matrix against the label.

**Output:** a bias-corrected estimate or a bound on the model's true sensitivity and specificity,
plus a statement of the direction of bias.

**Rationale:** When the reference label is imperfect, the observed sensitivity and specificity of
a model are attenuated toward the label's own error rates. The observed sensitivity is a weighted
average of the model's true sensitivity and its false positive rate, with weights determined by
the label's sensitivity and specificity. Reporting the observed value without correction
overstates the model's apparent modesty or, in some configurations, its apparent skill.

The correction has three possible outcomes, which should be reported explicitly.

| Situation | What the observed metric means | What to report |
|---|---|---|
| Label sensitivity and specificity both above 0.90 | Attenuation is small | Observed and corrected estimates; note that correction is stable |
| One metric high, one moderate | Attenuation is asymmetric | Corrected estimate with interval; state which metric is distorted |
| Either metric below 0.70 | Observed metric is dominated by label error | Bounds only; do not report a corrected point estimate without a sensitivity analysis |

**Branch point:** if the label's sensitivity is below roughly 0.6, the observed model sensitivity
is dominated by label error, and correction is essential before any claim about model performance.
If the label's specificity is very high (above 0.95), observed specificity is less distorted but
observed sensitivity may still be substantially attenuated.

### Step 13. Freeze, version, document, and monitor

**Input:** the validated label specification and the validation report.

**Output:** a versioned label definition, a change log, and a monitoring plan with triggers.

**Rationale:** Labels drift when coding guidelines change, when a new assay is introduced, when a
documentation template is updated, or when a hospital joins a new network. A label that was valid
in 2023 may not be valid in 2026. The monitoring plan should track prevalence, code-mix, testing
rates, and a periodic revalidation sample, with explicit triggers for revalidation.

The change log should record the date, the reason, the person who approved the change, and the
effect on prevalence in a fixed cohort. A change without a fixed-cohort comparison is not auditable.

**Branch point:** if the label changes, the model's performance metrics are no longer comparable
across the change. Re-baseline the metrics and record the date of the change in the model card.

### Label edge cases

**Zero cells.** A validation sample with no false positives produces a specificity of 1.0 and an
uninformative confidence interval. Increase the sample or report the one-sided bound.

**Perfect agreement and the kappa paradox.** When one label value is very rare, kappa can be low
even when agreement is high, because chance agreement is high. Report both raw agreement and
kappa, and the prevalence of each label value.

**Tiny validation samples.** A sample of 30 patients cannot support a precise estimate of
sensitivity. Report the interval, and do not use the point estimate to justify a deployment
decision.

**Prevalence shift.** When the construct becomes more or less common, label PPV and NPV change even
if sensitivity and specificity are stable. A label validated at 33 percent prevalence may have a
PPV of 0.62 at 15 percent prevalence. Recompute PPV for the deployment prevalence rather than
transporting the validation value.

**Coding version changes.** ICD-10-CM updates annually; a code may be added, retired, or split. A
label that spans the update without accounting for it can show a spurious prevalence jump. Record
the code-set version and test for a discontinuity at the update date.

**Differential testing.** If one group receives more laboratory testing, the laboratory-based label
will detect more cases in that group. This is surveillance bias, and it can masquerade as a true
difference in incidence. Compare testing rates before interpreting prevalence differences.

**Empty strata.** A site with no cases in the validation sample cannot support a site-specific
estimate. Report the count and either pool the site with a similar site or exclude it from the
stratified claim.

**Labels that arrive after time zero.** A diagnosis code assigned at discharge may encode an
outcome that was not known at admission. If such a code is used as a feature, it is label leakage;
if it is used as the label, the timing must be checked so that it does not encode pre-time-zero
information.

## 6. Worked example

### The setting

A cardiac surgery service wants a model that predicts acute kidney injury (AKI) within 7 days of
surgery, to trigger an early nephrology consult. The construct is AKI by KDIGO criteria: a
creatinine rise of at least 0.3 mg/dL within 48 hours, or a rise to at least 1.5 times baseline
within 7 days, or urine output below 0.5 mL/kg/h for 6 hours. The decision is a rule-in decision:
the consult team has limited capacity, and a false positive consumes a consult that another patient
needs. Specificity matters, but a missed severe AKI is worse.

The cohort is 1,850 adults who had isolated coronary artery bypass grafting or valve surgery
between 2022 and 2025. The candidate label is the presence of an ICD-10-CM N17.x code in any
diagnosis field during the index admission. A second candidate is the KDIGO laboratory criterion
computed from creatinine values. The team chooses the ICD code as the scalable label and uses
adjudicated chart review as the reference standard, because the code is what will be available at
deployment scale and the laboratory criterion misses patients without serial creatinine.

### The label definition

The code label is: any of N17.0, N17.1, N17.2, or N17.9 in any diagnosis field of the index
admission, with the admission ending within the 7-day outcome window. The control definition is:
no N17.x code and at least two creatinine measurements in the window, so that a patient without
testing is not silently classified as a non-case. Patients with end-stage renal disease on chronic
dialysis, prior kidney transplant, or fewer than two creatinine measurements are excluded from the
validation cohort. The washout is 90 days: patients with an N17.x code in the 90 days before
surgery are excluded as prevalent cases.

After exclusions, 1,850 admissions remain. The ICD label is positive in 579 (31.3 percent).
Adjudicated review of the full record estimates the true AKI prevalence at 611 of 1,850, or 33.0
percent.

### The validation sample

The team draws a simple random sample of 300 admissions and reviews all 300. This avoids
verification bias: every sampled patient has both the label and the reference standard, regardless
of label value. Two trained reviewers abstract a structured form; disagreements are adjudicated by
a nephrologist blinded to the ICD code. Inter-rater kappa before adjudication is 0.78, which is
adequate for a reference standard but not perfect.

The two-by-two table is:

| | Adjudicated AKI | No adjudicated AKI | Total |
|---|---:|---:|---:|
| ICD label positive | 77 | 17 | 94 |
| ICD label negative | 22 | 184 | 206 |
| Total | 99 | 201 | 300 |

### The arithmetic

Label sensitivity:

```text
77 / 99 = 0.7778   (95% Wilson CI 0.686 to 0.848)
```

Label specificity:

```text
184 / 201 = 0.9154   (95% Wilson CI 0.869 to 0.947)
```

Label PPV:

```text
77 / 94 = 0.8191   (95% Wilson CI 0.729 to 0.884)
```

Label NPV:

```text
184 / 206 = 0.8932
```

Cohen's kappa between the label and the adjudicated reference:

```text
observed agreement = (77 + 184) / 300 = 0.8700
expected agreement = (94 * 99 + 206 * 201) / 300^2
                   = (9,306 + 41,406) / 90,000
                   = 0.5635
kappa = (0.8700 - 0.5635) / (1 - 0.5635) = 0.702
```

The label is specific and moderately sensitive. About one in five true AKI cases is missed, and
about one in five labelled cases is not AKI on adjudication. For a rule-in decision, the PPV of
0.82 is acceptable but not excellent: five of every 27 consults would go to a patient who does not
meet the construct.

### Correcting the prevalence

The observed label prevalence in the full cohort is 579 / 1,850 = 0.3130. The Rogan-Gladen
correction uses the validation-sample sensitivity and specificity:

```text
true prevalence = (observed prevalence - (1 - specificity)) / (sensitivity + specificity - 1)
                = (0.3130 - 0.0846) / (0.7778 + 0.9154 - 1)
                = 0.2284 / 0.6932
                = 0.3295
```

The corrected prevalence is 33.0 percent, which matches the adjudicated prevalence of 611 / 1,850 =
33.0 percent. The agreement is expected because the validation sample was random; it confirms that
the correction is implemented correctly and that the sample is representative.

### Attenuation of the model's metrics

The team fits a model and, against the ICD label, observes sensitivity 0.80 and specificity 0.85
at the chosen threshold. Those are the model's true operating characteristics only if the label
were perfect. With a label sensitivity of 0.7778 and specificity of 0.9154, the observed metrics
against the label are attenuated. For a true prevalence of 33 percent, per 1,000 patients:

```text
true cases: 330; true non-cases: 670
model true positives: 330 * 0.80 = 264
model false positives: 670 * 0.15 = 100.5

label-positive among model-positive true cases: 264 * 0.7778 = 205.3
label-positive among model-positive non-cases: 100.5 * 0.0846 = 8.5
label-positive among model-negative true cases: 66 * 0.7778 = 51.3
label-positive among model-negative non-cases: 569.5 * 0.0846 = 48.2

observed sensitivity = (205.3 + 8.5) / (205.3 + 8.5 + 51.3 + 48.2)
                     = 213.8 / 313.3
                     = 0.682
```

The observed sensitivity is 0.68, not 0.80. The model is better than it looks. Observed
specificity is computed among label-negative patients:

```text
label-negative among model-negative non-cases: 569.5 * 0.9154 = 521.3
label-negative among model-negative true cases: 66 * 0.2222 = 14.7
label-negative among model-positive non-cases: 100.5 * 0.9154 = 92.0
label-negative among model-positive true cases: 264 * 0.2222 = 58.7

observed specificity = (521.3 + 14.7) / (521.3 + 14.7 + 92.0 + 58.7)
                     = 536.0 / 686.7
                     = 0.781
```

The observed specificity is 0.78, not 0.85. Both metrics are biased toward the label's error rates.
A team that reports only the observed values will conclude that the model is mediocre and may
abandon it or over-tune it to compensate for a measurement artefact.

### Two diagnostics that would reveal a broken analysis

**Diagnostic 1: the label validation table itself.** If the team had skipped validation and assumed
the code was correct, the observed sensitivity of 0.68 would be interpreted as the model's true
sensitivity. The 95 percent confidence interval around the label sensitivity (0.686 to 0.848) shows
that even the label's own sensitivity is uncertain. A report that presents model performance
without the label's operating characteristics is incomplete, and the incompleteness is detectable
by asking for the two-by-two table.

**Diagnostic 2: prevalence by year and by site.** The team plots label prevalence by year: 24.1
percent in 2022, 26.8 percent in 2023, 31.0 percent in 2024, and 34.2 percent in 2025. The clinical
case mix, measured by mean creatinine and bypass time, is stable. A rising label prevalence with a
stable case mix suggests a coding change, not a clinical change. The team finds that a new
documentation template introduced in 2024 prompts clinicians to add an N17.x code when creatinine
rises, which increased labelling without changing incidence. The validation sample was drawn across
all years, so the pooled sensitivity hides a difference: sensitivity is 0.85 in 2025 and 0.68 in
2022. A model trained on the pooled label learns the documentation change as a signal.

### Varying one assumption: prevalence shift

Suppose the model is deployed in a lower-risk population where the true AKI prevalence is 15
percent rather than 33 percent. The label's sensitivity and specificity are assumed unchanged.
Label PPV becomes:

```text
PPV = 0.7778 * 0.15 / (0.7778 * 0.15 + 0.0846 * 0.85)
    = 0.1167 / (0.1167 + 0.0719)
    = 0.619
```

A labelled case now has only a 62 percent chance of being true AKI, down from 82 percent. More
importantly, the observed model sensitivity falls further. With true model sensitivity 0.80 and
specificity 0.85, per 1,000 patients at 15 percent prevalence:

```text
true cases: 150; true non-cases: 850
model true positives: 120; model false positives: 127.5
label-positive among model-positive true cases: 120 * 0.7778 = 93.3
label-positive among model-positive non-cases: 127.5 * 0.0846 = 10.8
label-positive among model-negative true cases: 30 * 0.7778 = 23.3
label-positive among model-negative non-cases: 722.5 * 0.0846 = 61.1

observed sensitivity = (93.3 + 10.8) / (93.3 + 10.8 + 23.3 + 61.1)
                     = 104.1 / 188.5
                     = 0.552
```

The same model now appears to have sensitivity 0.55 against the label. A team comparing this
deployment to the development cohort would see a large performance drop and might conclude that
the model has drifted, when in fact the label's PPV has fallen and the label error dominates the
observed metric. The correct response is to revalidate the label in the new population, not to
retrain immediately.

### What to report

The primary label artefact is the two-by-two table with sensitivity, specificity, PPV, NPV, and
kappa, each with a confidence interval, plus the sampling design and the reference-standard
reliability. The primary model figure is the calibration plot against the adjudicated reference in
the validation subsample, with the label-based metrics shown alongside for comparison. The
secondary figure is label prevalence and testing rate by year and site, which makes drift visible.
The report should state the label's sensitivity and specificity in the model card, because every
downstream metric depends on them.

## 7. Code, query, or calculation

The script implements the validation arithmetic with the standard library. It assumes a completed
two-by-two table from a simple random sample. For a stratified or enriched sample, replace the raw
counts with design-weighted counts before calling `label_metrics`, or apply the weights in a
separate step and document them.

```python
"""Label validation arithmetic. Standard library only."""
import math


def wilson(k, n, z=1.96):
    """Wilson score interval for a proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def label_metrics(tp, fp, tn, fn):
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    ppv = tp / (tp + fp)
    npv = tn / (tn + fn)
    n = tp + fp + tn + fn
    observed = (tp + tn) / n
    expected = (((tp + fp) * (tp + fn)) + ((tn + fn) * (tn + fp))) / (n * n)
    kappa = (observed - expected) / (1 - expected)
    return {"sens": sens, "spec": spec, "ppv": ppv, "npv": npv, "kappa": kappa}


def rogan_gladen(observed_prev, sens, spec):
    """Correct an observed prevalence for imperfect sensitivity and specificity."""
    denominator = sens + spec - 1
    if abs(denominator) < 1e-9:
        raise ValueError("label carries no information: sens + spec = 1")
    return (observed_prev - (1 - spec)) / denominator


def attenuation(true_sens, true_spec, label_sens, label_spec, prevalence):
    """Observed model sensitivity/specificity when the reference label is imperfect."""
    tp = prevalence * true_sens
    fn = prevalence * (1 - true_sens)
    fp = (1 - prevalence) * (1 - true_spec)
    tn = (1 - prevalence) * true_spec
    observed_sens = (tp * label_sens + fp * (1 - label_spec)) / \
                    (prevalence * label_sens + (1 - prevalence) * (1 - label_spec))
    observed_spec = (tn * label_spec + fn * (1 - label_sens)) / \
                    ((1 - prevalence) * label_spec + prevalence * (1 - label_sens))
    return observed_sens, observed_spec


m = label_metrics(tp=77, fp=17, tn=184, fn=22)
print(f"sensitivity {m['sens']:.4f} CI {wilson(77, 99)[0]:.4f}-{wilson(77, 99)[1]:.4f}")
print(f"specificity {m['spec']:.4f} CI {wilson(184, 201)[0]:.4f}-{wilson(184, 201)[1]:.4f}")
print(f"PPV {m['ppv']:.4f} CI {wilson(77, 94)[0]:.4f}-{wilson(77, 94)[1]:.4f}")
print(f"NPV {m['npv']:.4f} kappa {m['kappa']:.4f}")
print(f"corrected prevalence {rogan_gladen(94 / 300, m['sens'], m['spec']):.4f}")

for prevalence in (0.33, 0.15):
    obs_sens, obs_spec = attenuation(0.80, 0.85, m["sens"], m["spec"], prevalence)
    print(f"prevalence {prevalence:.2f}: observed sens {obs_sens:.4f} spec {obs_spec:.4f}")
```

Expected output:

```text
sensitivity 0.7778 CI 0.6864-0.8484
specificity 0.9154 CI 0.8687-0.9465
PPV 0.8191 CI 0.7293-0.8839
NPV 0.8932 kappa 0.7022
corrected prevalence 0.3300
prevalence 0.33: observed sens 0.6824 spec 0.7806
prevalence 0.15: observed sens 0.5522 spec 0.8233
```

Reading the output:

- The label's sensitivity interval is wide (0.686 to 0.848) because only 99 cases were reviewed.
  A team that needs a precision of plus or minus 0.05 would need about 246 cases, as Section 5
  shows.
- Kappa of 0.70 is substantial but not excellent. It reflects both the label's error and the
  reviewers' disagreement.
- The corrected prevalence matches the adjudicated prevalence because the sample was random. In an
  enriched sample, the correction would be wrong unless the sampling weights were applied first.
- At 15 percent prevalence, observed sensitivity falls to 0.552 against the same model. This is the
  attenuation effect, not model drift.

The function `rogan_gladen` raises an error when `sens + spec = 1`, because the label then carries
no information and the corrected prevalence is undefined. That case occurs with a label that is
essentially a coin flip and should be treated as a failed label, not as a numerical edge case.

## 8. Data, standards, and regulatory context

Label definitions are constrained by terminology standards and by the reporting requirements of
the studies that use them. The table separates binding requirements from conventions.

| Instrument | Status | What it constrains | What it does not do |
|---|---|---|---|
| ICD-10-CM and ICD-10-PCS | Binding for US billing and many quality measures | Code set, annual update on 1 October | Does not define clinical truth or timing |
| ICD-11 | WHO classification, adoption varies by country | Coding of diagnoses and functioning | Not yet used for US billing |
| SNOMED CT | Terminology standard in many EHRs | Clinical concepts and relationships | Does not provide a cohort definition |
| LOINC | Terminology for observations and labs | Analyte identity and units | Does not specify a case threshold |
| RxNorm | Normalised medication names | Drug identity and ingredients | Does not encode indication |
| OMOP CDM | Common data model for observational research | Table structure and concept identifiers | Does not mandate a phenotype |
| FHIR | Interoperability standard | Resource structure and timestamps | Does not validate label accuracy |
| CDISC SDTM and ADaM | Binding for many regulatory submissions | Trial data structure and analysis datasets | Does not cover EHR phenotyping |
| STARD 2015 | Reporting guideline for diagnostic accuracy | Reporting of reference standard and flow | Does not assess label quality directly |
| TRIPOD+AI items 6, 7, 10, 14 | Reporting convention | Outcome definition, timing, and measurement | Does not prescribe a validation sample |
| PROBAST+AI outcome domain | Appraisal tool | Risk of bias from outcome definition | Does not supply a better label |
| FDA GMLP principle on reference standards | Guidance, not binding | Independent, representative reference standard | Does not specify a code set |
| HIPAA Privacy Rule | Binding in the US | Use and disclosure of protected health information | Does not govern label validity |

Three practical consequences follow.

**The code set must be versioned.** ICD-10-CM changes every October. A label that says "N17.x"
without a fiscal year is ambiguous after a code split. The specification should record the code
set, the version, the source (CMS, WHO, or a national modification), and the date range over which
the definition was validated. The same applies to LOINC and SNOMED CT releases, which are updated
more frequently than most teams expect.

**The reference standard must be described as a measurement.** STARD 2015 requires the reference
standard, its rationale, and the blinding of readers to be reported. TRIPOD+AI item 6 requires the
outcome definition and timing. Neither requires the reference standard to be perfect, but both
require the reader to be able to judge it. A validation report that says "chart review confirmed
the diagnosis" without a protocol, reviewer training, or agreement statistic is not sufficient.

**De-identification constrains chart review.** Under the HIPAA Privacy Rule, a limited data set or
de-identified data may not contain the dates needed to establish the outcome window. A label that
depends on precise timing may require identifiable data under a data use agreement or an
institutional review board approval. This is a legal constraint on the validation design, not a
statistical one, and it should be resolved before sampling.

### A label specification template

A label specification that can survive an audit contains the following fields. The template is a
convention, not a regulatory requirement, but it maps cleanly onto TRIPOD+AI items 6 and 7.

```text
construct:          one sentence, clinical language
decision served:    what action the label supports
case definition:    codes, values, logical operators
control definition: how a non-case is established
eligibility:        inclusion and exclusion rules with counts
time zero:          event and timestamp field
outcome window:     start, end, and boundary rule
washout:            lookback period and prior-episode rule
competing events:   death, transfer, hospice, loss to follow-up
data source:        tables, extracts, and date ranges
code set version:   ICD-10-CM FY, LOINC release, SNOMED CT release
reference standard: reviewer protocol and adjudication
validation sample:  design, size, and weights
operating characteristics: sens, spec, PPV, NPV, kappa with CIs
known error direction: under- or over-ascertainment
monitoring:         metrics, cadence, triggers, owner
```

**Provenance in FHIR and OMOP.** A label defined on FHIR resources should cite the resource type,
the code system, and the timestamp element used for the window. A label defined in OMOP should cite
the concept set identifiers and the cohort definition version. In both cases the provenance record
is what allows a second analyst to reproduce the label. This is a convention, but it is the
convention that distinguishes a data product from an ad hoc extract.

> **Note:** The FDA's GMLP principles call for a reference standard that is independent of the
> model's output and representative of the intended population. That is a quality expectation, not
> a statutory requirement, but it is the standard a reviewer will apply.

## 9. Industry practice and case evidence

**ICD-coded sepsis.** A systematic review and meta-analysis of 13 studies found pooled sensitivity
of 35 percent (95 percent CI 22 to 48) and pooled specificity of 98 percent (98 to 99) for ICD-10
codes against chart review. Median PPV was 72 percent and median NPV 95.9 percent. The authors
concluded that sepsis is substantially undercoded and recommended local validation before using
codes for research. The practical implication is that a sepsis model trained on codes is trained
on a label that captures roughly one-third of true cases, and the missed two-thirds are not a
random sample: they are the patients whose clinicians did not document the diagnosis, which
correlates with atypical presentation, non-ICU location, and coding practice.

**eMERGE phenotype validation.** The eMERGE network validated 13 electronic phenotype algorithms
across five sites with manual chart review. Most algorithms performed well, with positive predictive
values from 67.7 to 100 percent and about three-quarters of 51 reviews at or above 90 percent. The
lessons reported by the network include validating case and control definitions separately,
specifying time windows, handling repeated measures and transient states, and iterating the
algorithm after review. The variability across sites is as important as the median: a phenotype
that performs at 95 percent PPV at one site can perform at 70 percent at another with different
coding and testing patterns.

**Cost as a proxy for need.** Obermeyer et al. (2019) studied a commercial population-health
algorithm that used health care costs as a proxy for clinical need. At the same predicted risk
score, Black patients had more chronic conditions and worse biomarkers than White patients. The
authors estimated that replacing the cost label with a health-based label would increase the
proportion of Black patients identified for additional care from 17.7 percent to 46.5 percent. The
label choice, not the model class, produced the disparity.

**Portability across sites.** The PhEMA project executed a benign prostatic hyperplasia phenotype
at eight sites and found case PPV at or above 90 percent wherever chart review was performed, but
only about 90 percent case overlap with the original eMERGE implementations at three of five
sites. The phenotype travelled, but not identically. For a model, that means the label can be
reused across sites with local validation, not with the assumption that the same code produces the
same cohort.

**Laboratory-defined acute kidney injury.** KDIGO criteria require a baseline creatinine and a
follow-up value. Patients without serial creatinine cannot be classified, and the probability of
serial testing depends on severity, comorbidity, and ward. A KDIGO-based label therefore has a
built-in surveillance bias: sicker patients are tested more and labelled more. The label is still
useful for a clinical decision, but its prevalence cannot be interpreted as incidence without
accounting for testing.

**Readmission labels.** The CMS readmission measure uses an algorithm to exclude planned
readmissions, because a planned staged procedure is not a quality failure. The algorithm is
imperfect, and different implementations produce different labels. A model trained on a label that
does not exclude planned readmissions learns to predict surgical scheduling as much as unplanned
return. The lesson is that the control and exclusion rules are part of the label, not a
post-processing detail.

**Reference standards in imaging.** The IDx-DR pivotal trial used a reading centre with
ETDRS-graded widefield photographs as the reference standard, not the treating clinician's
impression. That choice made the performance estimate comparable across sites. A trial that used
local reads as the reference would have produced a different, and likely more optimistic, estimate
of accuracy.

**Label timing in sepsis models.** The Epic Sepsis Model validation reported a hospitalisation-
level AUROC of 0.63, but an analysis that included scores up to three hours after sepsis onset
raised the AUC to 0.80. The difference is label timing: if the prediction is evaluated after the
outcome has begun to manifest, the model appears better than it is at the moment it must act. The
framing lesson is that the evaluation must anchor predictions at the intended decision time, not at
whatever timestamp makes the numbers look strong. This is a documented sensitivity analysis in the
paper; the interpretation as a label-timing effect is an inference.

**Planned versus unplanned readmission.** The CMS Hospital Readmissions Reduction Program uses a
planned-readmission algorithm to exclude staged procedures and other expected returns. Different
implementations of that algorithm produce different labels, and a model trained on a label without
the exclusion learns scheduling patterns. The lesson generalises: the control and exclusion rules
are part of the outcome definition, and two teams that use different exclusions are not training on
the same task.

> **Note:** The cases above are documented. The generalisation that "labels should be validated
> locally" is an inference from them, not a finding of any single study.

## 10. Failure modes and red flags

Label failures are unusual in that they are often invisible in the model report. The model's
metrics look reasonable, the calibration plot is acceptable, and the problem only appears when
someone compares the label to a better standard. The table is therefore organised around the
evidence that would expose each failure, not around the failure's statistical description.

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Code label treated as truth | No validation sample exists | Coding captures documentation, not disease | Model learns coding behaviour | Validate against chart review |
| Control defined as absence of code | No testing requirement in the control definition | Untested patients counted as non-cases | Specificity inflated, prevalence distorted | Require evidence of testing or encounters |
| Enriched sample analysed naively | Sensitivity reported from a case-enriched sample | Sampling weights ignored | Sensitivity biased upward or downward | Weight or correct for the design |
| Verification bias | Reference standard obtained only for label positives | Non-random verification | Sensitivity and specificity biased | Review a random sample |
| No washout period | Chronic conditions counted as incident | Prevalent cases enter the cohort | Prevalence overstated; model learns chronicity | Add a lookback window |
| Outcome window chosen for convenience | Window not tied to clinical course | Events outside the window ignored | Label misses the outcome it claims to measure | Justify the window clinically |
| Death treated as non-event | Mortality not in the label definition | Competing event ignored | Severity understated | Composite or competing-risk design |
| Coding version change | Prevalence jump at fiscal year boundary | Codes added, split, or retired | Spurious trend; model drift | Version the code set; test discontinuity |
| Differential testing | Testing rates differ by group | Surveillance bias | Prevalence difference mistaken for biology | Report testing rates by group |
| Single reviewer, unblinded | No agreement statistic | Reviewer confirms the label | Reference standard biased | Blind and adjudicate |
| Tiny validation sample | Wide intervals ignored | Sampling noise | False precision | Report intervals; increase sample |
| Label leakage into features | A code or note after time zero is predictive | Outcome encoded in features | Inflated performance | Timing audit; ablation |
| Label drift after deployment | Prevalence or code mix changes | Documentation or policy change | Model performance degrades | Monitor and revalidate |
| Composite endpoint hides components | Only the composite is reported | One component drives the result | Misleading conclusion | Report component-wise performance |

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The reference standard measures the construct | Validation requires a comparison | Reviewers disagree or use a different definition | Report agreement and adjudication |
| The validation sample is representative | Estimates are transported to the cohort | Sample restricted by site, era, or label | Compare sample and cohort characteristics |
| Label sensitivity and specificity are stable | Correction and attenuation formulas assume constancy | Coding or testing practice changes | Stratify by era and site |
| Error is non-differential with respect to the model | Attenuation formulas assume independence | Clinicians code differently when the model is used | Prospective validation |
| The outcome window captures the event | Label must count what it claims | Events occur after the window | Clinical justification; sensitivity to window |
| Competing events are handled explicitly | Binary label assumes two states | Death or transfer prevents the outcome | Compare definitions with and without |
| Controls are disease-free, not untested | Specificity depends on the control definition | Untested patients counted as controls | Require testing evidence |
| The code set is complete for the population | Label must capture all presentations | Atypical cases are coded differently | Chart review of label negatives |
| Prevalence in deployment is known | PPV and net benefit depend on it | Case mix shifts | Monitor prevalence |
| Reviewers are blinded to model output | Reference standard must be independent | Reviewers see predictions | Blinding audit |

The deepest limit is that a label validation study measures the label against a reference standard
that is itself imperfect. The result is a estimate of agreement between two measurements, not a
measurement of truth. That is acceptable if the reference standard is better than the label and
the direction of its error is understood. It is not acceptable to present the result as if the
reference standard were infallible. The report should state what the reference standard itself
likely misses.

## 12. Exercises

Exercises 1 through 3 are recall and calculation; 4 through 6 are design and critique; 7 is
calculation applied to a model comparison; 8 is a design deliverable that can be reused in the
module assignment.

**Exercise 1. Construct to label.** Choose a clinical outcome with which you are familiar. Write
the construct definition, three candidate operational definitions, and the error asymmetry each
would introduce. Deliverable: a one-page table. Acceptance criteria: each definition is computable;
each error direction is stated; the recommended definition is justified by the decision. Time: 45
minutes.

**Exercise 2. Two-by-two arithmetic.** A label validation sample of 400 has TP 120, FP 30, FN 40,
TN 210. Compute sensitivity, specificity, PPV, NPV, and kappa, with Wilson intervals for the first
four. Deliverable: the calculations and a short interpretation. Acceptance criteria: all values
correct to three decimals; interpretation names the dominant error direction. Time: 45 minutes.

**Exercise 3. Prevalence correction.** Using the label from Exercise 2, the full cohort has 1,200
labelled positives out of 6,000 patients. Correct the prevalence with Rogan-Gladen and compare it
with the observed prevalence. Deliverable: the arithmetic and a statement of whether the correction
is stable. Acceptance criteria: corrected prevalence correct; stability judged by the denominator
`sens + spec - 1`. Time: 30 minutes.

**Exercise 4. Design a validation sample.** For a cohort of 20,000 with an expected construct
prevalence of 4 percent, design a sampling scheme that estimates sensitivity with a 95 percent
interval half-width of 0.07 and PPV with a half-width of 0.05. State the number of cases and
controls, the sampling method, and the weighting. Deliverable: a sampling protocol. Acceptance
criteria: sample size justified by arithmetic; design avoids verification bias; weighting
explained. Time: 60 minutes.

**Exercise 5. Audit a published label.** Choose a published clinical prediction model and find its
outcome definition. Assess whether the definition was validated, whether the code set was versioned,
and whether the timing was specified. Deliverable: a 500-word audit. Acceptance criteria: at least
four of the five label properties from Section 3 are addressed; claims are cited to the paper.
Time: 60 minutes.

**Exercise 6. Detect differential ascertainment.** Using a dataset with site identifiers (real or
simulated), compute label prevalence and testing rate by site. Identify whether differences are
likely due to case mix, testing, or coding. Deliverable: a table and a 300-word analysis.
Acceptance criteria: testing and coding are separated; no causal claim without evidence. Time: 75
minutes.

**Exercise 7. Attenuation calculation.** A model has true sensitivity 0.75 and specificity 0.90. The
label has sensitivity 0.60 and specificity 0.95. At prevalence 25 percent, compute the observed
sensitivity and specificity. Repeat at prevalence 8 percent. Deliverable: the arithmetic and a
one-paragraph interpretation. Acceptance criteria: both prevalences computed correctly; the
interpretation distinguishes label artefact from model drift. Time: 45 minutes.

**Exercise 8. Write a label monitoring plan.** For a deployed model, specify the label metrics to
monitor, the cadence, the triggers for revalidation, and the owner. Deliverable: a one-page plan.
Acceptance criteria: at least four metrics; triggers are quantitative; owner and escalation path
named. Time: 60 minutes.

## 13. Self-check answers

**Answer to Exercise 2.** Sensitivity = 120 / 160 = 0.750. Specificity = 210 / 240 = 0.875. PPV =
120 / 150 = 0.800. NPV = 210 / 250 = 0.840. Observed agreement = 330 / 400 = 0.825. Expected
agreement = ((150 * 160) + (250 * 240)) / 400^2 = (24,000 + 60,000) / 160,000 = 0.525. Kappa =
(0.825 - 0.525) / (1 - 0.525) = 0.632. The dominant error is false negatives: 40 true cases are
missed, which is 25 percent of cases.

**Answer to Exercise 3.** Observed prevalence = 1,200 / 6,000 = 0.200. Corrected prevalence =
(0.200 - (1 - 0.875)) / (0.750 + 0.875 - 1) = (0.200 - 0.125) / 0.625 = 0.075 / 0.625 = 0.120. The
denominator 0.625 is comfortably above zero, so the correction is stable, but the result depends on
the assumption that sensitivity and specificity are constant between the validation sample and the
full cohort.

**Answer to Exercise 7.** At 25 percent prevalence, per 1,000: true cases 250, non-cases 750. Model
true positives 187.5, false positives 75. Label-positive among model-positive cases =
187.5 * 0.60 = 112.5; among model-positive non-cases = 75 * 0.05 = 3.75. Label-positive among
model-negative cases = 62.5 * 0.60 = 37.5; among model-negative non-cases = 675 * 0.05 = 33.75.
Observed sensitivity = (112.5 + 3.75) / (112.5 + 3.75 + 37.5 + 33.75) = 116.25 / 187.5 = 0.620.
Observed specificity = (675 * 0.95 + 62.5 * 0.40) / (750 * 0.95 + 250 * 0.40) = (641.25 + 25) /
(712.5 + 100) = 666.25 / 812.5 = 0.820. At 8 percent prevalence: true cases 80, non-cases 920.
Model true positives 60, false positives 92. Observed sensitivity = (60 * 0.60 + 92 * 0.05) /
(80 * 0.60 + 920 * 0.05) = (36 + 4.6) / (48 + 46) = 40.6 / 94 = 0.432. The observed sensitivity
falls from 0.62 to 0.43 as prevalence falls, even though the model and label are unchanged. This is
attenuation, not drift.

**Answer to Exercise 4.** For sensitivity with a half-width of 0.07 around 0.80, `n` cases is about
`1.96^2 * 0.8 * 0.2 / 0.07^2 = 125`. For PPV with a half-width of 0.05 around an expected 0.70,
`n` labelled positives is about `1.96^2 * 0.7 * 0.3 / 0.05^2 = 323`. The binding constraint is
labelled positives, so the sample needs about 323 label-positive patients. At 4 percent prevalence,
that requires reviewing roughly 8,075 patients, which is 40 percent of the cohort. A more practical
design is to review all 800 expected positives plus a random sample of 1,000 negatives, then weight
the estimates: this gives adequate precision for PPV and enough controls to estimate specificity,
but sensitivity still rests on the 800 positives. The protocol must state that sensitivity is
estimated from all positives rather than from a random sample, which avoids verification bias
because the positives are not selected on anything other than their label value.

**Rubric for Exercises 1, 4, 5, 6, and 8.** A strong submission (1) names the construct and the
decision, (2) makes the label computable and versioned, (3) validates against a described reference
standard, (4) reports intervals and design, (5) checks differential ascertainment, and (6) states
the direction of residual bias. A competent submission meets four of the six. A weak submission
reports a single PPV without a sampling design or treats a code as ground truth.

## 14. Interview questions

**1. What is the difference between a construct and a label?** A strong answer defines the
construct as the clinical phenomenon and the label as its measurement, and gives an example where
the two diverge. A weak answer uses the terms interchangeably.

**2. How would you validate a diagnosis-code label?** A strong answer describes a random or
properly weighted chart-review sample, a structured abstraction form, blinding, adjudication, and
reporting of sensitivity, specificity, PPV, NPV, and kappa with intervals. A weak answer says
"check a few charts".

**3. Why is a code-based sepsis label problematic?** A strong answer cites the low sensitivity of
ICD codes for sepsis and explains that the model learns documentation behaviour. A weak answer says
codes are "unreliable" without quantifying or explaining the direction.

**4. What is verification bias?** A strong answer explains that it arises when the reference
standard is obtained only for a selected subset, and that sensitivity and specificity are biased
while PPV may remain estimable. A weak answer confuses it with selection bias in the cohort.

**5. How does label error affect a model's measured sensitivity?** A strong answer explains
attenuation, gives the intuition that the observed metric is pulled toward the label's error rates,
and notes that the direction depends on the label's sensitivity and specificity. A weak answer says
label noise "adds randomness" without direction.

**6. When would you use a proxy label?** A strong answer says when the construct is expensive or
slow to measure, the proxy has been validated against the construct, and the proxy's failure modes
are monitored. A weak answer says proxies are acceptable if data are scarce.

**7. How do you handle deaths in a readmission label?** A strong answer treats death as a competing
event or an exclusion with a stated rationale, and shows how the choice changes the estimand. A
weak answer counts death as a non-readmission without discussion.

**8. What is label drift and how do you detect it?** A strong answer monitors prevalence, code mix,
testing rates, and periodic revalidation, and distinguishes drift from case-mix change. A weak
answer says "retrain the model" without a detection method.

**9. How large should a validation sample be?** A strong answer bases the size on the precision
needed for sensitivity among true cases and for PPV among labelled positives, and gives the
arithmetic. A weak answer says "a few hundred".

**10. What is the difference between label drift and dataset shift?** A strong answer says label
drift is a change in the measurement process (coding, testing, documentation), while dataset shift
is a change in the distribution of features or the feature-outcome relationship; the two can occur
independently and require different responses. A weak answer uses the terms interchangeably.

**11. How would you label an outcome that is only observed in patients who are tested?** A strong
answer describes the selection mechanism, proposes a tested-and-negative control definition, and
discusses inverse-probability weighting or a sensitivity analysis for untested patients. A weak
answer treats untested patients as non-cases.

**12. Can a model outperform its label?** A strong answer says no in expectation against the same
label, but yes against a better reference standard, and explains that apparent outperformance
usually indicates leakage or a different evaluation sample. A weak answer says yes if the model is
better.

## 15. Key terms

- **Adjudication** — resolution of reviewer disagreements by a defined rule or committee.
- **Ascertainment** — the process by which a patient comes to be labelled.
- **Case definition** — the complete rule for identifying a case.
- **Censoring** — end of observation before the outcome is known.
- **Chart review** — manual abstraction of the record as a reference standard.
- **Code set** — the diagnosis, procedure, or medication codes used in a label.
- **Competing risk** — an event that prevents the outcome of interest.
- **Computable phenotype** — an executable algorithm that identifies a condition from EHR data.
- **Construct** — the clinical phenomenon of interest, independent of measurement.
- **Control definition** — the rule for identifying a non-case.
- **Differential ascertainment** — label error that varies by group, site, or time.
- **Eligibility algorithm** — the rule for including a patient in the cohort.
- **Immortal time bias** — misclassified follow-up during which the outcome cannot occur.
- **Incident case** — a new episode after a disease-free interval.
- **Informative missingness** — missingness that depends on the outcome or its predictors.
- **Inter-rater agreement** — agreement between reviewers beyond chance.
- **Label** — the value used as the supervised target.
- **Label drift** — change in label prevalence or error over time.
- **Label noise** — random or systematic error in the label.
- **Label PPV** — probability a labelled positive is a true case.
- **Label sensitivity** — probability a true case is labelled positive.
- **Label specificity** — probability a true non-case is labelled negative.
- **Lead-time bias** — earlier detection appears to prolong survival without changing course.
- **Length-time bias** — screening preferentially detects slow-progressing cases.
- **Lookback window** — period before time zero searched for history.
- **Operational definition** — the rules that convert data into a label.
- **Outcome window** — interval after time zero in which the outcome counts.
- **Phenotype portability** — whether a phenotype performs similarly across sites.
- **Prevalent case** — an existing condition at time zero.
- **Proxy label** — a label that stands in for a construct it does not directly measure.
- **Reference standard** — the best available ascertainment of the construct.
- **Rogan-Gladen correction** — prevalence correction for imperfect sensitivity and specificity.
- **Spectrum bias** — label performance differs by case severity.
- **Surveillance bias** — more testing in one group produces more labels.
- **Verification bias** — bias from obtaining the reference standard only for a selected subset.
- **Washout period** — interval before time zero used to exclude prior episodes.
- **Weak supervision** — labels generated by rules or heuristics rather than adjudication.

## 16. Sources

All URLs accessed 25 October 2026.

1. Liu S, et al. Accuracy of International Classification of Diseases, 10th Revision codes for
   identifying sepsis: a systematic review and meta-analysis. *Critical Care Explorations*
   2022;4(11):e0788. https://doi.org/10.1097/CCE.0000000000000788
2. Newton KM, Peissig PL, Kho AN, et al. Validation of electronic medical record-based phenotyping
   algorithms: results and lessons learned from the eMERGE network. *Journal of the American
   Medical Informatics Association* 2013;20(e1):e147-e154.
   https://doi.org/10.1136/amiajnl-2012-000896
3. Bhasin M, et al. Verification bias in phenotype validation. *ACI Open* 2024.
   https://doi.org/10.1055/a-2402-5937
4. Obermeyer Z, Powers B, Vogeli C, Mullainathan S. Dissecting racial bias in an algorithm used to
   manage the health of populations. *Science* 2019;366(6464):447-453.
   https://doi.org/10.1126/science.aax2342
5. Agniel D, Kohane IS, Weber GM. Biases in electronic health record data due to processes within
   the healthcare system. *BMJ* 2018;361:k1479. https://doi.org/10.1136/bmj.k1479
6. Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement. *BMJ* 2024;385:e078378.
   https://doi.org/10.1136/bmj-2023-078378
7. Moons KGM, Damen JAA, Kaul T, et al. PROBAST+AI. *BMJ* 2025;388:e082505.
   https://doi.org/10.1136/bmj-2024-082505
8. Bossuyt PM, Reitsma JB, Bruns DE, et al. STARD 2015: an updated list of essential items for
   reporting diagnostic accuracy studies. *BMJ* 2015;351:h5527. https://doi.org/10.1136/bmj.h5527
9. US Food and Drug Administration. Good Machine Learning Practice for Medical Device Development:
   Guiding Principles. 2021.
   https://www.fda.gov/medical-devices/artificial-intelligence-enabled-medical-devices/good-machine-learning-practice-medical-device-development-guiding-principles
10. US Centers for Medicare and Medicaid Services. ICD-10-CM official guidelines for coding and
    reporting. https://www.cms.gov/medicare/coding-billing/icd-10-codes
11. World Health Organization. International Classification of Diseases 11th Revision.
    https://icd.who.int/
12. Observational Health Data Sciences and Informatics. OMOP Common Data Model.
    https://ohdsi.github.io/CommonDataModel/
13. HL7. FHIR Release 4 specification. https://hl7.org/fhir/R4/
14. Clinical Data Interchange Standards Consortium. SDTM and ADaM implementation guides.
    https://www.cdisc.org/standards
15. KDIGO. Clinical Practice Guideline for Acute Kidney Injury. *Kidney International Supplements*
    2012;2(1):1-138. https://kdigo.org/guidelines/acute-kidney-injury/
16. Wong A, Otles E, Donnelly JP, et al. External validation of a widely implemented proprietary
    sepsis prediction model. *JAMA Internal Medicine* 2021;181(8):1065-1070.
    https://doi.org/10.1001/jamainternmed.2021.2626
17. Abràmoff MD, Lavin PT, Birch M, Shah N, Folk JC. Pivotal trial of an autonomous AI-based
    diagnostic system for detection of diabetic retinopathy. *npj Digital Medicine* 2018;1:39.
    https://doi.org/10.1038/s41746-018-0040-6
18. US Department of Health and Human Services. HIPAA Privacy Rule: guidance on de-identification.
    https://www.hhs.gov/hipaa/for-professionals/privacy/special-topics/de-identification/index.html

A final check before using any label: write the construct, the operational definition, and the
validation sample on one page. If the three cannot be read together without contradiction, the label
is not ready, and no model trained on it will be.

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic: [`01_problem-framing.md`](01_problem-framing.md)
- Next topic: [`03_splitting-and-validation.md`](03_splitting-and-validation.md)
- Metrics and prevalence: [`07_metrics.md`](07_metrics.md)
- Survey file: [`../../06_machine_learning_foundations.md`](../../06_machine_learning_foundations.md)
- Labs: [`../../labs/`](../../labs/)
- Cohort pipeline lab: [`../../labs/03_cohort_pipeline.ipynb`](../../labs/03_cohort_pipeline.ipynb)
