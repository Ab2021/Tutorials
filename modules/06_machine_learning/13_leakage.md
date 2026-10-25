# Leakage

> **Module 06 — Machine Learning Foundations**
> Topic 13 of 13
> Estimated study time: 4 hours
> Prerequisites: Topics 02, 03, 04, and 09
> Version 2026.3

## 1. Why this topic exists

Leakage is the presence of information about the outcome that will not be available at the moment
of prediction. It is the most common reason a clinical model looks excellent in development and
fails in deployment. The model has not learned the clinical signal; it has learned the outcome's
paper trail. A discharge disposition, a follow-up call note, a billing code, a repeat admission, or
a site identifier can all encode the answer, and the model will use them if they are in the table.

The failure is not a modelling error in the narrow sense. The algorithm is doing what it was asked:
finding the features that predict the label. The error is in the data flow, the split, or the
feature inventory. Kaufman et al. (2012) formalised leakage as the introduction of information about
the target that should not be legitimately available, and described it as one of the top data-mining
mistakes. They argued that the most reliable defence is "learn-predict separation": a clear
boundary between the information available at training time and the information available at
prediction time.

Clinical data have a particular density of leakage opportunities. Davis et al. (2023) propose a
framework with three families: aggregation-based leakage (site-level and patient-level, including
the same patient in training and validation), timing-based leakage (post-outcome predictors that
violate the "no time machine" rule), and association-based leakage (deterministic proxies such as
antibiotics for sepsis or chest tubes for pneumothorax). The taxonomy is useful because the
detection test differs by family.

The performance inflation can be enormous. A model with an honest AUROC of 0.70 can reach 0.95 with
a single leaked feature, and the improvement is often larger than the difference between any two
legitimate models. The inflation is not a small bias to be noted in limitations; it is a different
model, trained on a different task. The deployment model does not have the leaked feature, and its
performance reverts to the honest value.

Leakage also has a way of hiding behind a good story. A feature called "follow-up completed" sounds
like a care-process variable; it is actually a post-discharge event that occurs because someone was
worried about the patient. A feature called "discharge disposition" sounds administrative; it
encodes the clinical course. A site identifier sounds neutral; it encodes the case mix and the
coding practice. The audit must look at the timestamp and the generating process, not the name.

Leakage also distorts the comparison between models. A flexible model is better at exploiting a
leaked feature, so a gradient-boosted model can appear superior to a logistic regression solely
because it found the leak more efficiently. The comparison is not about the algorithm; it is about
which model found the back door. The audit should precede any model comparison, because a leaky
feature can reverse the ranking.

A final practical point: leakage is often introduced by a well-intentioned feature. The team should
ask of every new feature: when was this recorded, by whom, and for what purpose? If the answer is
"after the outcome" or "because of the outcome", the feature is a leak.

The audit should also be repeated when the data source changes. A new feed, a new coding system, or
a new site can introduce a feature that was not in the original inventory. The audit is a lifecycle
activity, not a one-time gate. A model that passed the audit in 2023 may fail it in 2025, not
because the model changed but because the data did.

The audit should be documented with the date, the auditor, and the version of the pipeline. A
leakage section without those is not reproducible.


A final practical point: leakage is often introduced by a well-intentioned feature. A team adds a
"care coordination" flag because it sounds like a process improvement; the flag is recorded after
the outcome. A team adds a "prior utilization" count because it is clinically plausible; the count
includes the readmission itself. The audit must inspect the generating process, not the intent.

This topic is the module's catalogue and audit. You will learn the nine common leakage mechanisms,
the detection test for each, the fix, and the revalidation that proves the fix. The output is a
leakage audit that a reviewer can reproduce and a performance waterfall that shows how much of the
original result was leakage.

> **Note:** A performance estimate that is much better than the clinical literature is a leakage
> suspect, not a triumph. The first response is an audit, not a celebration.

## 2. Learning outcomes

By the end of this topic you can:

1. **Define** the prediction-time information set and **classify** features by availability at time
   zero.
2. **Identify** the nine leakage mechanisms in a feature inventory and a data pipeline.
3. **Apply** a detection test to each mechanism, including duplicate detection, temporal checks,
   single-feature screens, and ablation.
4. **Quantify** the inflation with a performance waterfall from the naive to the honest estimate.
5. **Fix** each leak at the data, split, or pipeline level, and **revalidate** the model.
6. **Recognise** leakage in published models and **explain** why the reported performance is not
   deployable.
7. **Design** a leakage audit that runs as part of the validation, not after it.
8. **Write** a leakage section that states the tests, the findings, and the residual risk.

Outcomes 1 through 5 are computational and audit-based; 6 through 8 are critique and design.

## 3. The core mental model

Leakage is an information-flow problem. The model is trained on a set of features, and the
prediction is made at a time zero with a smaller information set. The gap between the two sets is
where leakage lives.

```text
All data in the warehouse
        |
        +--> available before time zero  ---> legitimate features
        |
        +--> created at or after time zero ---> leakage candidates
                 |
                 +--> the outcome itself
                 +--> a consequence of the outcome
                 +--> a note or order written because of the outcome
                 +--> a code assigned after the outcome
                 +--> a repeat record of the same patient
```

The diagram is the first audit: for every feature, ask when it was created and why. A feature
created after time zero is a candidate. A feature created before time zero but updated after it is
also a candidate. A feature that is a deterministic consequence of the outcome is a candidate even
if its timestamp is early.

The three families of Davis et al. map onto three failure points.

**Timing-based leakage.** The feature exists at prediction time in the table but was created later.
Examples: discharge diagnosis codes, final laboratory results, radiology reports signed after the
prediction, notes that mention the outcome. Detection: compare the feature timestamp with time zero;
ablate the feature and measure the performance change.

**Aggregation-based leakage.** The split does not respect the unit of independence, so the same
patient or site appears in training and test. Examples: multiple admissions per patient, duplicate
records, a site identifier that is constant within a site. Detection: count patients and sites
across folds; check duplicate feature vectors.

**Association-based leakage.** A feature is a deterministic proxy for the outcome, even if it is
recorded before the outcome. Examples: antibiotics ordered for suspected sepsis, a chest tube for a
pneumothorax, a comfort-care order for imminent death. Detection: a single-feature AUC near 1, a
clinician review of the top predictors, and an ablation.

The families are also a reminder that leakage is not one bug. A team can have a clean timing audit
and a broken split, or a clean split and a leaked encoding. The audit is a set of checks, not a
single verdict. The audit is a set of independent
checks, not a single gate. A team that fixes the post-outcome
features and stops has fixed one family and left two. The audit should cover all three, and the
waterfall should show the contribution of each.

The performance waterfall is the quantitative audit:

```text
Naive estimate (all features, random split)          0.94
  minus post-outcome features                        0.84
  minus duplicate records                            0.81
  minus patient overlap (grouped split)              0.76
  minus target-encoding leakage                      0.74
  minus site signature                               0.73
  temporal validation                                0.70
```

Each step removes one mechanism and shows its contribution. The final value is the honest estimate.
The waterfall is more useful than a single "corrected" number because it shows which leak mattered
and how much.

A practical way to assign the families is to ask three questions about each feature: when, from
what, and for whom. When was the value created? From what source or event? For whom is it recorded
(the patient, the encounter, the site)? The answers usually place the feature in one or more
families and suggest the test.

The families are not mutually exclusive. A feature can be a post-outcome variable and a site
signature at the same time, if the site's coding practice determines when the variable is recorded.
The audit should record both families and both tests.
The audit should assign all applicable labels, not one.

The families are also not exhaustive. A leak can enter through the label, the cohort, the
preprocessing, or the evaluation, and the catalogue in Step 14 is the operational list. The
three-family framework is a way to organise the catalogue, not a substitute for it.

The three families suggest three different audit questions. For timing, ask when the information
became available; for aggregation, ask whether the unit crossed the split; for association, ask
whether the feature is a consequence. A single audit checklist should have a row for each family. For aggregation, ask whether the same unit appears on both sides of the split.
For association, ask whether the feature is a consequence of the outcome rather than a cause. The
questions are different, and a single "leakage check" that does not separate them will miss one of
the families.

A useful table for the audit is a matrix of features by family, with a check mark for each test
applied and a result. The matrix should be part of the audit report, because it shows what was
tested and what was not. A feature with no check mark is a gap, not a pass. The matrix makes the coverage visible: a feature that was not tested for
timing is a gap, even if it passed the association test.

A useful way to remember the audit is three questions, and a useful way to remember the fix is one
word: remove. Remove the feature, remove the duplicate, or remove the future from the window. The
alternative, a more complex model, usually makes the leak harder to see. Remove the feature, remove the duplicate, or remove the future from the window. The
fix is usually subtraction, not a more complex model. When was the information created? Which unit
crossed the split? Is the feature a consequence of the outcome? The questions map to the three
families and to the three tests.

A useful rule is the plausibility check. If the model's AUROC is far above the published range for
the same outcome, or if a single feature has an AUC above 0.95, or if the performance drops sharply
when one feature is removed, the burden of proof is on the team to show that the feature is
legitimate. The clinical literature is a prior, not a verdict, but a large discrepancy is a signal.

> **Note:** Leakage can be in the label as well as the features. A label that is defined after the
> outcome, or that encodes the outcome's consequences, produces the same inflation. The label audit
> in Topic 02 is part of the leakage audit.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Leakage | Information about the outcome not available at prediction time | Confused with overfitting | Defect |
| Time zero | The moment the prediction is made and features are frozen | Vague or missing | Timestamp |
| Information set | The features legitimately available at time zero | Assumed from the table | Set |
| Post-outcome feature | A feature created after the outcome | Called a process variable | Feature |
| Target leakage | The outcome or its cause present in the features | Confused with class imbalance | Defect |
| Label leakage | A label that encodes the features or the future | Confused with label noise | Defect |
| Temporal leakage | Using future information to predict the past | Confused with temporal validation | Defect |
| Patient leakage | The same patient in training and test | Confused with a random split | Defect |
| Site leakage | The same site or a site signature in both splits | Confused with site validation | Defect |
| Duplicate record | The same encounter entered twice | Confused with a repeated measure | Row |
| Preprocessing leakage | Fitting a transform outside the fold | Confused with preprocessing | Defect |
| Target encoding leakage | Encoding categories with outcomes from the test fold | Confused with target encoding | Defect |
| Feature selection leakage | Selecting features on the full data | Confused with feature selection | Defect |
| Hyperparameter leakage | Tuning on the test set | Confused with tuning | Defect |
| Threshold leakage | Choosing the threshold on the test set | Confused with threshold analysis | Defect |
| Outcome proxy | A feature that is a deterministic consequence of the outcome | Confused with a risk factor | Feature |
| Association leakage | Leakage through a proxy recorded before the outcome | Confused with a strong predictor | Defect |
| Site signature | A feature that identifies the site and its practice | Confused with a clinical variable | Feature |
| Metadata leakage | Identifiers, timestamps, or file names that encode the outcome | Ignored | Defect |
| Label-derived metadata | A code or flag created from the label | Confused with a feature | Defect |
| No-time-machine rule | Features must predate the prediction | Ignored | Rule |
| Learn-predict separation | A clear boundary between training and prediction information | Confused with a split | Principle |
| Ablation | Removing a feature and measuring the change | Confused with feature importance | Test |
| Single-feature AUC | The AUC of one feature alone | Confused with model AUC | Metric |
| Performance waterfall | The sequence of estimates as leaks are removed | Confused with a sensitivity analysis | Table |
| Plausibility check | Comparing the result with the literature | Confused with a benchmark | Heuristic |
| Duplicate detection | Finding identical or near-identical records | Confused with deduplication | Test |
| Near-duplicate | A record that differs only in an identifier or timestamp | Confused with a duplicate | Row |
| Entity overlap | The same patient or site across splits | Confused with a group split | Count |
| Temporal overlap | Outcome windows that cross the split boundary | Confused with a temporal split | Count |
| Future information | Data created after the prediction | Confused with recent data | Defect |
| Point-in-time correctness | A feature is valid at time zero | Confused with data freshness | Property |
| Backfill | A value entered later for an earlier time | Confused with a real-time value | Process |
| Late-arriving data | Data that arrive after the event | Confused with missing data | Process |
| Immortal time | Follow-up during which the outcome cannot occur | Confused with censoring | Bias |
| Informative missingness | Missingness that depends on the outcome | Confused with random missingness | Mechanism |
| Label timing | When the label is ascertained | Confused with feature timing | Timestamp |
| Index date | The date used to anchor the cohort | Confused with time zero | Date |
| Washout | A period before time zero to exclude prior events | Confused with follow-up | Days |
| Outcome window | The period after time zero in which the outcome counts | Confused with follow-up | Days |
| Data cutoff | The last date with complete data | Confused with time zero | Date |
| Audit trail | The record of feature creation and transformation | Omitted | Log |
| Provenance | The source and history of a feature | Confused with a data dictionary | Record |
| Revalidation | Re-running the validation after a fix | Confused with retraining | Process |
| Residual risk | Leakage that remains after the audit | Assumed to be zero | Statement |
| Leakage audit | A structured review of the mechanisms | Confused with a data quality check | Process |
| Clinical review | A clinician's review of top predictors | Confused with a statistical test | Review |
| Plausibility range | The published performance range for the outcome | Confused with a target | Range |
| Data dictionary | The description of each feature | Confused with provenance | Document |
| Feature inventory | The list of features with timing and source | Confused with a data dictionary | Table |
| Extraction query | The code that builds the feature | Confused with the feature | Query |
| Join duplication | Row multiplication from a join | Confused with a true duplicate | Defect |
| Many-to-many join | A join that multiplies rows | Ignored | Defect |
| Aggregation window | The lookback period for a feature | Confused with the outcome window | Days |
| Look-ahead bias | Using information from the future | Confused with hindsight | Defect |
| Survivorship bias | Conditioning on survival to the outcome | Confused with a complete-case analysis | Bias |
| Selection leakage | Selecting the cohort using future information | Confused with eligibility | Defect |

Three definitions deserve more than a table row.

**Leakage versus overfitting.** Overfitting is fitting noise in the training data; leakage is using
information that will not be available at prediction time. A model can overfit without leakage and
leak without overfitting. The remedies differ: overfitting is addressed by regularisation and more
data; leakage is addressed by removing the information or changing the split.

**Leakage versus a strong predictor.** A feature can be strongly predictive and legitimate, such as
a prior admission count for readmission. The test is not the strength of the association but the
availability of the information at time zero and the causal ordering. A strong predictor with the
right timing is a signal; a strong predictor with the wrong timing is a leak.

**Leakage versus distribution shift.** Distribution shift changes the relationship between features
and outcome after deployment; leakage is present at development and absent at deployment. The
performance patterns can look similar, but the diagnosis differs: shift is detected by monitoring,
leakage by auditing the feature timing and the split.

> **Note:** The most dangerous leaks are the ones that look like plausible clinical variables. A
> feature named "follow-up" or "disposition" does not announce its timing. The audit must inspect
> the generating process.

## 5. The mechanics, step by step

A leakage audit has fourteen steps. Steps 1 through 4 define the information set and inventory the
features. Steps 5 through 9 apply the mechanism-specific tests. Steps 10 through 14 quantify, fix,
revalidate, and report.

### Step 1. Fix time zero and the information set

**Input:** the prediction task specification.

**Output:** a timestamped time zero and a written statement of what is available at that moment.

**Rationale:** Leakage is defined relative to time zero. Without a timestamped boundary, every
feature is ambiguous. "At discharge" is not a time zero; "at the signature of the discharge order"
is. The information set should be written before the feature inventory, so that the inventory can
be checked against it.

The information set should include the negative space: what is not available. A model that assumes a
laboratory result is available at time zero, when the result typically takes four hours, is leaking
the future even if the specimen was drawn earlier. The negative space is part of the specification.

**Branch point:** if the model updates over time, there are multiple time zeros. The audit must
check each update, not only the first. The audit must
check each update, not only the first.

### Step 2. Inventory every feature with provenance

**Input:** the feature list and the data dictionary.

**Output:** a table with name, source table, generating event, timestamp field, lookback window, and
expected availability at time zero.

**Rationale:** A feature without provenance cannot be audited. The generating event is the key: a
laboratory value is generated by an order and a specimen; a note is generated by a clinician; a code
is generated by a coder. The timestamp field tells when the value became available, not when the
event occurred.

The inventory should include the extraction query, because the query is the only reliable evidence
of what the feature actually contains. A data dictionary describes the intent; the query describes
the implementation, and the two often differ.

**Branch point:** if the provenance is unknown, treat the feature as a leakage suspect until it is
traced. A feature that cannot be traced cannot be defended.

### Step 3. Classify features by timing

**Input:** the provenance table.

**Output:** each feature labelled pre-time-zero, at-time-zero, post-time-zero, or unknown.

**Rationale:** The classification is the first filter. Post-time-zero features are removed or
replaced with a pre-time-zero version. Unknown features are investigated. At-time-zero features are
allowed only if the value is available before the prediction is made, not merely on the same day.

The classification should be done on the feature's value, not on the column name. A column called
"admission_lab" may contain the latest lab of the stay, including one drawn after the prediction.
The extraction query is the evidence, and it should be read, not assumed.

**Branch point:** if a feature is a summary over a window that extends past time zero, the window
must be truncated at time zero. A 30-day average that includes the outcome day is a leak.

### Step 4. Classify features by generating process

**Input:** the provenance table and clinical knowledge.

**Output:** each feature labelled as a clinical measurement, a process variable, an administrative
code, or an identifier.

**Rationale:** Process variables and administrative codes are the most common leaks. A process
variable such as "follow-up call completed" is generated after discharge; an administrative code
such as a discharge disposition is assigned at the end of the stay. The generating process reveals
the timing even when the timestamp is missing.

**Branch point:** if a process variable is a legitimate predictor (for example, a pre-admission
medication order), keep it; if it is generated by the outcome, remove it.

### Step 5. Apply the temporal test

**Input:** the features and time zero.

**Output:** a list of features with a timestamp after time zero, or a generating event after time
zero, with counts.

**Rationale:** The temporal test is the most direct. It compares the feature's availability
timestamp with time zero and flags anything later. The test should use the timestamp of
availability, not the timestamp of the clinical event, because a lab drawn before time zero but
resulted after it is not available.

**Branch point:** if the timestamp is missing, use the generating event and the workflow to infer
availability. A missing timestamp is not evidence of availability.

### Step 6. Apply the patient and site overlap test

**Input:** the split assignments and the patient and site identifiers.

**Output:** the number of patients and sites in more than one fold, and the duplicate record count.

**Rationale:** The overlap test catches aggregation-based leakage. A patient with multiple
admissions can appear in training and test; a site can contribute to both. The duplicate count
catches the same encounter entered twice. The test is a count, not a modelling choice.

The overlap test should be run on the actual fold assignments, not on the intended design. A
grouped split can be implemented incorrectly, and the count is the evidence that it was not. The
test is a few lines of code and should be part of the validation report.

**Branch point:** if the split is temporal, patients can still cross periods. Group by patient
within the temporal split, or exclude the crossing patients.

### Step 7. Apply the single-feature screen

**Input:** each feature and the outcome.

**Output:** the univariate AUC or association for each feature, sorted.

**Rationale:** A single feature with an AUC above about 0.95 is a leakage suspect. The screen is
cheap and catches deterministic proxies that the timestamp test might miss, because the proxy can be
recorded before the outcome while still encoding it.

The screen should be run on the held-out data or on the full cohort before modelling. A feature's
univariate AUC is a property of the feature and the outcome, not of the model, so the screen does
not require a split. It is a first-pass filter, and it should be reported in the audit.

**Branch point:** if a feature has a very high univariate AUC, review it with a clinician before
keeping it. A legitimate feature can be strong; a deterministic proxy cannot be legitimate.

### Step 8. Apply the ablation test

**Input:** the fitted model and the feature set.

**Output:** the change in performance when each suspect feature is removed, one at a time and in
groups.

**Rationale:** Ablation shows how much the model relies on a feature. A large drop when a suspect
feature is removed is evidence that the feature is carrying the result. The test should be done on
held-out data, and the model should be refitted without the feature, not merely scored without it.

The ablation should be run in groups as well as individually, because two leaks can mask each
other. A model that uses two post-outcome features may show little change when one is removed,
because the other carries the signal.

**Branch point:** if removing a feature improves performance, the feature may be adding noise or
the model may be unstable. Investigate rather than assuming the feature is good.

### Step 9. Apply the preclinical review

**Input:** the top predictors and a clinical reviewer.

**Output:** a written review of whether each top predictor is plausible at time zero and whether it
could be a proxy for the outcome.

**Rationale:** The statistical tests can miss a leak that a clinician recognises immediately. A
review of the top 10 to 20 predictors by someone who knows the workflow is one of the most
efficient audits. The review should ask: when is this recorded, who records it, and why?

The review should be documented with the reviewer's role and the date. A review without a record is
an opinion; a review with a record is evidence that the audit was done. The record should name the
features reviewed and the conclusion for each.

**Branch point:** if the clinician cannot explain why a top feature would be known at time zero,
treat it as a leak until proven otherwise.

### Step 10. Quantify the inflation with a waterfall

**Input:** the naive estimate and the estimates after each fix.

**Output:** a waterfall table from the naive to the honest estimate.

**Rationale:** The waterfall shows which leak mattered and how much. It also prevents the team from
claiming that a single fix solved the problem: removing one leak can expose another, and the
waterfall makes the sequence visible.

**Branch point:** if the honest estimate is close to the naive one, the leakage was minor. If the
drop is large, the original result was not a model of the clinical task.

### Step 11. Fix at the right level

**Input:** the audit findings.

**Output:** a fix for each mechanism: remove the feature, truncate the window, group the split,
deduplicate, move the transform inside the fold, or change the label.

**Rationale:** The fix must match the mechanism. Removing a post-outcome feature fixes timing
leakage; grouping the split fixes patient leakage; moving the imputer inside the fold fixes
preprocessing leakage. A fix at the wrong level leaves the leak in place.

The fix should be documented in the pipeline, not only in the analysis notebook. A leak that is
removed in one script but present in the production pipeline will return at deployment. The
pipeline should have a test that fails if a post-time-zero feature is included.

**Branch point:** if the label is the leak, a feature fix will not help. The label must be changed
and the model redeveloped.

### Step 12. Revalidate after the fix

**Input:** the fixed pipeline.

**Output:** a new validation with the corrected split and features, and a comparison with the
naive result.

**Rationale:** A fix that is not revalidated is a hypothesis. The revalidation should use the same
design as the original, with the leak removed, so the comparison is direct. The new estimate is the
one that goes in the model card.

**Branch point:** if the revalidated performance is below the decision requirement, the project may
not be viable. Report that rather than reverting to the leaky result.

### Step 13. Audit the label and the cohort

**Input:** the label definition and the cohort construction.

**Output:** a statement of whether the label or the cohort uses future information.

**Rationale:** Leakage can enter through the label (a code assigned after the outcome) or the cohort
(selecting patients who survived to the outcome). The label audit from Topic 02 and the cohort flow
diagram are the evidence. A cohort that requires a post-outcome test is selected on the future.

**Branch point:** if the cohort is selected on a post-outcome variable, the model is not deployable
for the intended population. Redefine the cohort at time zero.

### Step 14. Report and monitor

**Input:** the audit and the revalidation.

**Output:** a leakage section with the tests, the findings, the fixes, the waterfall, and the
residual risk, plus a monitoring plan.

**Rationale:** The report should let a reviewer reproduce the audit and see what was checked. The
residual risk statement is important: an audit cannot prove the absence of leakage, only the
absence of the leaks that were tested. The monitoring plan should watch for a performance drop that
would suggest a residual leak.

The report should include the audit queries or scripts, so that a reviewer can rerun them. A
leakage section that describes the tests but cannot be reproduced is a narrative, not an audit.

**Branch point:** if the model is updated, rerun the audit. A new feature or a new data source can
introduce a new leak.

### The leakage catalogue

The table below is the core of the topic. Each row is a mechanism, an example, a detection test, and
a fix.

| Mechanism | Example | Detection test | Fix |
|---|---|---|---|
| Target leakage | The outcome is a feature, or a deterministic function of it | Single-feature AUC near 1; ablation | Remove the feature |
| Post-outcome feature | Discharge disposition, follow-up note, final culture | Timestamp after time zero; clinician review | Remove or truncate |
| Patient overlap | Same patient in train and test | Count patients spanning folds | Group split by patient |
| Preprocessing outside the fold | Median, scaler, or selector fitted on all data | Audit the fit scope | Fit inside the fold |
| Target encoding | Category means computed with test outcomes | Encoding fitted before the split | Fold-safe encoding with smoothing |
| Temporal leakage | Future labs, notes, or codes used for a past prediction | Timestamp and outcome-window overlap | Point-in-time extraction |
| Duplicate records | Same encounter entered twice | Exact and near-duplicate matching | Deduplicate before splitting |
| Site signature | Site code or site-specific practice pattern | Site in top predictors; site-held-out validation | Remove the identifier; validate by site |
| Label-derived metadata | A flag or code created from the label | Provenance of the code | Remove the metadata |

### Leakage edge cases

**A feature with no timestamp.** Treat it as a suspect until the generating event is traced. The
absence of a timestamp is not evidence of availability. The
absence of a timestamp is common in derived features, and it is not evidence of availability.

**A feature whose timestamp is the update time.** A problem list entry may have an update timestamp
that is after time zero even though the condition began earlier. The relevant time is when the
information became available, not when it was last edited.

**A feature that is backfilled.** A value entered later for an earlier time is not available at the
earlier time. The backfill flag, if it exists, should be used; if not, the extraction should be
checked against the source audit log.

**A feature that is legitimate but strong.** A prior admission count can have a high AUC and be
legitimate. The test is timing, not strength.

**A feature created by the billing process.** A code assigned to justify a payment may be created
after the outcome. The generating process, not the code's name, determines the timing.

**A feature that is a note section.** A note's conclusion may mention the outcome even if the note
was started before it. The note timestamp is not enough; the content and the signing time matter.

**A duplicate that differs only in an identifier.** Exact matching misses it. Use a near-duplicate
test on the feature vector.

**A feature whose missingness predicts the outcome.** The missingness may be informative, and a
missing indicator can leak the outcome if the reason for testing is the outcome. Check the
indicator's univariate association and the reason for the test.

**A feature that is constant in a subgroup but varies overall.** A site-specific variable can be
constant within a site and carry the site's outcome rate. It is a site signature in disguise.

**A join that multiplies rows.** The row count grows and the same encounter appears multiple times.
Check the join cardinality.

**A site identifier that is predictive.** It may encode case mix or coding. Validate by site before
keeping it.

**A label that is assigned at discharge.** The label timing may be after time zero but still valid
for the outcome; the question is whether the features use the label or its consequences.

**A post-outcome feature that is also a legitimate predictor.** A pre-admission medication is
legitimate; a medication ordered because of the outcome is not. The generating reason matters.

## 6. Worked example

### The setting

A readmission model reports an AUROC of 0.941 on a random split of 40,000 discharges. The clinical
team is pleased; the data team is suspicious, because the published readmission models are in the
0.60 to 0.75 range. The audit begins.

### The audit findings

The feature inventory has 34 features. Three are post-time-zero: discharge disposition, a
follow-up-call completion flag, and a note-derived "care transition" indicator. The dataset has
1,240 duplicate records (3.1 percent), because a data migration loaded some encounters twice. The
split is random by row, and 18 percent of patients have more than one admission. A physician
identifier is target-encoded, and the encoding was fitted on the full dataset. A site code is in the
feature set.

### The performance waterfall

| Step | AUROC | Change |
|---|---:|---:|
| Naive: all features, random split | 0.941 | - |
| Remove post-outcome features | 0.842 | -0.099 |
| Deduplicate records | 0.806 | -0.036 |
| Group split by patient | 0.762 | -0.044 |
| Fold-safe target encoding | 0.741 | -0.021 |
| Remove site code | 0.728 | -0.013 |
| Temporal validation | 0.701 | -0.027 |

The honest estimate is 0.701. The naive estimate was 0.240 higher. The largest single contribution
is the post-outcome features (0.099), followed by patient overlap (0.044) and duplicates (0.036).
The target encoding and site code add smaller but material inflation.

### The arithmetic of the inflation

The naive AUROC is 0.941; the honest is 0.701. The difference is 0.240, which is larger than the
difference between the best and worst legitimate model in the comparison. A team that used the naive
number to choose the model would have chosen the wrong model; a team that used it to justify
deployment would have overpromised by a wide margin.

The single-feature screen finds that discharge disposition alone has an AUC of 0.890, and the
follow-up-call flag has 0.874. Both are post-outcome. The site code has 0.612, which is modest but
material at the margin.

### Two diagnostics that would reveal the leakage

**Diagnostic 1: the single-feature screen.** A feature with an AUC of 0.890 is a leakage suspect,
especially when its name suggests a process variable. The screen is cheap and should run before any
modelling. A report that gives only the model AUROC would not show it.

**Diagnostic 2: the duplicate and overlap counts.** The audit finds 1,240 duplicate records and
18 percent of patients with multiple admissions. A random row split puts those records in both
training and test. The counts are the evidence; the performance drop after deduplication and
grouping is the consequence.

### The label audit

The label is an unplanned readmission within 30 days, identified from the admission feed. The audit
checks whether the label uses a post-discharge source that is unavailable at time zero. It does:
the readmission is ascertained after discharge, which is correct for the outcome, but the features
must not use the same feed. The team finds that one feature, "days to next admission", was derived
from the feed and is effectively the label. It is removed. The lesson is that a legitimate label can
still leak through a feature derived from it.

### The preprocessing audit

The imputer was fitted on the full dataset, the scaler was fitted on the full dataset, and the
physician target encoding was fitted on the full dataset. All three are preprocessing leaks. The
first two inflate the estimate by a small amount; the third inflates it by 0.021, because the
encoding carries the test patients' outcomes. The fix is to fit every transform inside the training
fold and to use cross-fitting for the encoding.

### The target-encoding arithmetic

The physician identifier has 180 levels, and the encoding was computed on all 40,000 records. A
physician with 12 patients and 6 events gets an encoding near 0.5, which includes the test patients.
When the encoding is refitted inside the fold with smoothing `m = 20`, the same physician's encoding
is `(12 * 0.5 + 20 * 0.19) / 32 = 0.306`, much closer to the global rate. The change is small per
physician but material across the model: the AUROC drops from 0.762 to 0.741.

### The site signature

The site code has a univariate AUC of 0.612. It is not a deterministic proxy, but it encodes the
site's case mix and coding practice, and the model uses it. At a new site, the code is unseen and the
model's behaviour changes. The fix is to remove the identifier and validate by site. The AUROC drops
from 0.741 to 0.728, and the site-held-out estimate is lower still.

### The cost of the audit, in context

The audit is two days against a redevelopment that can take months. The ratio is the business case
for the audit.

### The cost of the audit

The audit takes two days: one for the inventory and the timing test, one for the split, duplicates,
encoding, and clinician review. The cost is small relative to the cost of deploying a model with a
0.24 inflation. The audit is not a delay; it is the difference between a model and a demonstration.

### The order of the audit

The team audits the timing first, because it is the cheapest and the largest. It then audits the
split, because patient overlap is the second largest. The preprocessing and encoding audits follow,
and the site signature is last. The order is deliberate: the largest and cheapest checks first, so
that the team does not spend a week on a small leak while a large one remains.

### The clinician review

A heart failure nurse reviews the top 15 predictors. She flags three: the follow-up-call flag
("someone called because they were worried"), the discharge disposition ("hospice patients are not
readmitted"), and the care-transition note ("written after the handover"). She also flags a
medication count as possibly post-discharge medication reconciliation. The team checks the
medication count's timestamp and finds it is pre-discharge, so it stays. The review is the fastest
way to catch the leaks that the timestamps miss, and it should be a required step, not an optional
one.

### The cohort audit

The cohort is adults discharged alive with a heart failure diagnosis. The diagnosis is identified
from codes in the index admission, which is legitimate because the admission is before time zero.
The cohort excludes patients discharged to hospice, which is a post-discharge disposition and could
select on the future. The team reviews the exclusion: hospice discharge is known at the discharge
order, so it is available at time zero. The exclusion is legitimate, but the timing should be
documented.

### What the waterfall does not show

The waterfall is sequential, so the order of the fixes changes the individual contributions. If the
duplicates are removed before the post-outcome features, the duplicate contribution may be smaller
or larger. The total drop is stable; the decomposition is order-dependent. The report should state
the order and should not over-interpret a single row.

### The residual risk

The audit tested nine mechanisms and found four with material effects. It did not test whether the
note-derived features contain outcome language that the timestamp test misses, because the notes
were not available for review. That is a residual risk, and the report states it. A residual risk
statement is not a confession; it is the boundary of the evidence.

### What would change the conclusion

The conclusion would change if the post-outcome features were legitimately available at time zero,
which they are not. It would change if the duplicates were true repeated measures rather than data
entry errors, which the record identifiers show they are not. It would change if the site code were
a clinical variable rather than an identifier, which the data dictionary shows it is not. The audit
is a sequence of specific claims, each with evidence.

### Varying one assumption: the split is temporal

If the split is temporal (train 2019-2022, test 2023) with a 30-day gap, the patient overlap is
reduced but not eliminated, because a patient can be admitted in both periods. Grouping by patient
within the temporal split reduces the AUROC further, to 0.681. The temporal design is the most
realistic, and it gives the lowest estimate. The conclusion changes from "the model is excellent"
to "the model is modest but potentially useful at the staffable threshold".

### What to report

The primary figure is the performance waterfall, with each fix labelled and the honest estimate at
the bottom. The secondary table is the leakage catalogue with the test result for each mechanism.
The report should state the residual risk: the audit tested the nine mechanisms, but it cannot prove
the absence of an untested leak. The model card should use the honest estimate, not the naive one.

## 7. Code, query, or calculation

The script demonstrates three audit checks on a synthetic cohort: duplicate detection, a
single-feature screen, and an ablation of a leaky feature. The honest model uses one feature; the
leaky model adds a post-outcome feature. The score is a fixed linear combination, so the difference
is attributable to the leak.

```python
"""Duplicate, single-feature, and ablation checks for leakage. Standard library only."""
import math
import random


def auroc(labels, scores):
    pairs = sorted(zip(scores, labels))
    positives, negatives = sum(labels), len(labels) - sum(labels)
    wins, seen_negatives = 0.0, 0
    for _, label in pairs:
        if label == 1:
            wins += seen_negatives
        else:
            seen_negatives += 1
    return wins / (positives * negatives)


rng = random.Random(3)
rows = []
for i in range(2000):
    x = rng.gauss(0, 1)
    true_risk = 1 / (1 + math.exp(-(-0.8 + 1.0 * x)))
    y = 1 if rng.random() < true_risk else 0
    leak = y + rng.gauss(0, 0.45)          # a post-outcome feature
    rows.append({"pid": i, "x": x, "leak": leak, "y": y})

duplicates = [dict(row) for row in rows[:200]]
all_rows = rows + duplicates

unique_patients = len({row["pid"] for row in all_rows})
print(f"records={len(all_rows)} unique={unique_patients} "
      f"dup_rate={1 - unique_patients / len(all_rows):.3f}")

print(f"single_feature_leak_auc={auroc([r['y'] for r in all_rows], [r['leak'] for r in all_rows]):.3f}")
print(f"honest_auc={auroc([r['y'] for r in all_rows], [0.8 * r['x'] for r in all_rows]):.3f}")
print(f"leaky_auc={auroc([r['y'] for r in all_rows], [0.8 * r['x'] + 3 * r['leak'] for r in all_rows]):.3f}")
```

Expected output:

```text
records=2200 unique=2000 dup_rate=0.091
single_feature_leak_auc=0.941
honest_auc=0.744
leaky_auc=0.958
```

Reading the output:

- The duplicate rate is 9.1 percent, because 200 records were copied. A random split would put
  many of those copies in both training and test.
- The post-outcome feature alone has an AUC of 0.941, which is well above the honest model's 0.744.
  The single-feature screen would flag it immediately.
- The leaky model's AUC is 0.958, against the honest 0.744. The difference of 0.214 is the
  inflation from one feature.
- The honest model is not a good model, but it is an honest one. The audit's job is to find the
  0.214, not to make the model look good.

The reusable checks are the duplicate count, the single-feature AUC, and the ablation. In a real
pipeline, the same checks run on the actual data, and the clinician review covers the features that
the statistics cannot judge.

## 8. Data, standards, and regulatory context

Leakage is addressed by reporting guidelines, appraisal tools, and regulatory expectations about
data provenance and testing. The table distinguishes binding requirements from conventions.

| Instrument | Status | What it says about leakage | What it does not do |
|---|---|---|---|
| TRIPOD+AI | Reporting convention | Requires predictor timing, data sources, and validation design to be reported | Does not audit the pipeline |
| PROBAST+AI | Appraisal tool | Assesses predictors, analysis, and validation for risk of bias, including leakage | Does not run the tests |
| FDA GMLP principles | Guidance, not binding | Calls for representative data, independent test sets, and reference standards | Does not define leakage |
| FDA PCCP guidance | Guidance, not binding | Requires planned data and model changes to be described | Does not audit the initial pipeline |
| EU AI Act | Binding in the EU, phased | Requires data governance and testing for high-risk systems | Does not specify a leakage test |
| EU MDR | Binding in the EU | Requires clinical evaluation and performance evaluation | Does not define leakage |
| ISO 14971 | Voluntary standard | Risk management across the lifecycle | Does not address data leakage |
| NIST AI RMF | Voluntary framework | Measure and manage data and model risks | Not sector-specific |

Three points matter for practice.

**TRIPOD+AI expects the timing to be reported.** The checklist asks for the predictors, their
measurement timing, and the validation design. A report that lists features without their timing
has not given the reader the information needed to judge leakage. The timing is part of the
predictor definition.

**PROBAST+AI treats leakage as a risk of bias.** The analysis domain asks whether the predictors
were available at the intended prediction time and whether the validation was independent. A study
with post-outcome predictors is at high risk of bias even if the statistics are correct.

**Regulators expect the data flow to be documented.** The FDA's GMLP principles and the EU AI Act's
data governance requirements both imply that the provenance and timing of the inputs should be
known. A submission that cannot explain when a feature becomes available cannot support the claimed
intended use. The leakage audit is part of the evidence.

### A leakage audit checklist

A leakage section that can be reproduced contains the following.

```text
time zero and the information set
feature inventory with generating event and timestamp
timing classification, including unknown features
patient and site overlap counts
duplicate and near-duplicate counts
preprocessing fit scope
target encoding fit scope
single-feature AUC table
ablation waterfall, with the order stated
clinician review of the top predictors
label and cohort audit
residual risk statement
monitoring plan
```

The checklist maps onto the TRIPOD+AI predictor and validation items and the PROBAST+AI analysis
domain. A section that reports the performance without the audit has not given the reader the
evidence to judge whether the performance is real.

> **Note:** No regulation prescribes a leakage test. The requirement is that the model's inputs be
> available at the intended use time and that the validation be independent.

## 9. Industry practice and case evidence

**The formal treatment.** Kaufman et al. (2012) define leakage as the introduction of information
about the target that should not be legitimately available. They describe the "no-time-machine"
condition for features and the "learn-predict separation" principle for training and evaluation.
They argue that fixing leakage without separation is often hard or impossible, and that removing
one leak can expose another.

**The clinical taxonomy.** Davis et al. (2023) propose three families: aggregation-based,
timing-based, and association-based leakage. They distinguish proactive models, which intentionally
use information about planned events, from reactive models, which must respect the no-time-machine
rule. The framework is useful because it separates the mechanisms and the tests.

**The COVID-19 models.** The living systematic review by Wynants et al. (2020) found many COVID-19
prognostic models at high risk of bias, with few externally validated. Leakage was one of the
recurring problems, particularly the use of post-outcome information and the absence of independent
validation. The case is a reminder that rapid model proliferation is not evidence.

**The sepsis label timing.** The external validation of the Epic Sepsis Model reported that
including scores up to three hours after sepsis onset raised the AUROC from 0.63 to 0.80. The
difference is label timing: evaluating the model after the outcome has begun to manifest makes it
look better. The audit should anchor the prediction at the intended decision time.

**The KDD-Cup patient-ID case.** Kaufman et al. describe the KDD-Cup 2008 patient-ID case, where
the identifier itself carried information about the outcome. The lesson is that metadata and
identifiers are leakage candidates, not neutral columns.

**The pattern across domains.** Leakage is not unique to health care. The Kaufman paper describes
competitions in finance and social networks where the identifier or the timestamp carried the
answer. The health care cases are a subset of a general problem: any dataset generated by a process
that also generates the outcome can leak the process. The audit is a general discipline, and the
clinical version is the one this module teaches.

**The cost of a missed leak.** A model with a 0.24 inflation can pass internal review, be deployed,
and fail. The failure is not only the model's; it is the team's credibility and the programme's
trust. The audit is the cheapest insurance against that failure.

**A note on the evidence base.** Most of the leakage literature is methodological or case-based;
there is no randomised evidence that an audit improves outcomes. The argument for the audit is
mechanical: it removes information that will not be available at deployment, and it prevents a
performance claim that cannot be reproduced. The report should not overstate the evidence as
clinical.

**The audit in a regulated submission.** A sponsor that submits a model with a post-outcome feature
may be asked to justify the timing. The justification is usually impossible, and the submission is
delayed while the model is redeveloped. The audit is cheaper before submission than after.

**The audit and the incentive.** A team rewarded for a high AUROC has an incentive to keep a leaky
feature. A team rewarded for a reproducible audit has an incentive to remove it. The governance of
the metric matters as much as the metric itself.

**The audit and the culture.** A team that treats leakage as a personal failure will hide it; a
team that treats it as a routine check will find it. The audit should be blameless and routine, like
a code review. The report should describe the findings without attributing them to an individual.

**The audit and the update.** A model update can reintroduce a leak that was fixed. The audit
should be rerun on every version, and the version history should record the result. A model that
passed once is not certified forever.

**The audit and the timeline.** A leakage audit done after the model is built is a rescue; done
before, it is a design. The cost is lower and the result is better when the audit is part of the
protocol. The report should state when the audit was performed relative to the model development.

**The audit and the team.** The audit is a team exercise. The data engineer knows the joins, the
clinician knows the workflow, and the statistician knows the split. A leakage audit done by one
person is a partial audit. The report should name the contributors and the checks each performed.

**The audit and the reader.** A reader of a model report cannot see the pipeline. The audit section
is the reader's only evidence that the features were available at time zero and that the split was
independent. A report without that section asks the reader to trust the number, which is the
opposite of what a validation report should do.

**The audit and the report.** The leakage section should be written before the performance section,
because the honest estimate is the performance number. A report that writes the performance first
and the audit second tends to present the naive number as the headline and the audit as a caveat.
The order should be reversed.

**The audit and the label.** A label that is defined after the outcome, or that encodes the
outcome's consequences, is a leak even if the features are clean. The label audit in Topic 02 is
the evidence, and the leakage audit should cite it rather than duplicate it. The two audits are
complements, not alternatives.

**The audit and the model comparison.** If two models are compared on a leaky feature set, the more
flexible model often wins because it exploits the leak more efficiently. The comparison is not
evidence about the algorithms. The audit should precede the comparison, and the honest waterfall
should be the basis for any model choice.

**The audit in a research publication.** A paper that reports a leaked model can be cited for
years, and the leak is often discovered only when another team tries to reproduce the result. The
audit is part of reproducible research, not only commercial due diligence. The report should
include the audit so that the reader can judge the claim.

**The audit in a commercial due-diligence.** A buyer that acquires a model without a leakage audit
inherits the risk. The audit is part of the technical due diligence, and the absence of one is a
red flag. The buyer should ask for the feature inventory, the split design, and the waterfall.

**What the cases have in common.** In each case, the leak was a plausible variable that a domain
expert could have flagged if asked. The sepsis label timing, the patient identifier, the discharge
disposition, and the cost proxy are all variables that look like clinical or administrative data.
The common thread is not a technical bug; it is the absence of a timing audit and a clinician
review.

**How a team should use the cases.** A team should read the cases as a checklist, not as a set of
anecdotes. Each case maps to a mechanism and a test. The team should ask which of the mechanisms are
present in its own pipeline, and run the corresponding test. The cases are most useful when they
change the team's prior about its own data.

**The industry pattern.** Most leakage audits find at least one post-outcome feature and at least
one split problem. The audits are inexpensive relative to the cost of a failed deployment, and they
are increasingly part of the model development checklist. The practice is still uneven: many model
cards report the performance without the audit, and the reader cannot tell whether the audit was
done.

> **Note:** The cases are documented. The claim that a specific published model was leaked is not
> made here unless the cited source reports it; the cases illustrate the mechanisms.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Post-outcome feature | Top predictor is a discharge variable | Timing | Inflated performance | Remove or truncate |
| Patient overlap | Patients span folds | Aggregation | Inflated performance | Group split |
| Duplicate records | Row count exceeds unique encounters | Aggregation | Inflated performance | Deduplicate |
| Preprocessing outside fold | Transform fitted on all data | Pipeline | Small to moderate inflation | Fit inside fold |
| Target encoding leak | Encoding fitted before split | Pipeline | Moderate inflation | Fold-safe encoding |
| Temporal leakage | Future labs or notes | Timing | Large inflation | Point-in-time extraction |
| Site signature | Site code in top features | Aggregation | Transport failure | Remove and validate by site |
| Label-derived metadata | A flag created from the label | Association | Large inflation | Remove metadata |
| Outcome proxy | A deterministic consequence | Association | Large inflation | Remove or relabel |
| Cohort selected on the future | Post-outcome eligibility | Selection | Biased cohort | Redefine at time zero |
| Join duplication | Row multiplication | Data engineering | Inflated performance | Check cardinality |
| Single-feature AUC near 1 | One feature dominates | Any | Suspicious result | Clinician review |
| No leakage audit | Performance much above literature | Process | Undetected leak | Run the audit |
| Audit without revalidation | The fix is a hypothesis | Process | Claim not supported | Revalidate |
| Feature with no timestamp | Provenance unknown | Timing | Undetected leak | Trace the query |
| Backfilled value | Entered later for an earlier time | Timing | Leak | Check the audit log |
| Cohort selected on the future | Post-outcome eligibility | Selection | Biased cohort | Redefine at time zero |
| Near-duplicate missed | Exact match only | Aggregation | Inflated performance | Distance matching |
| Feature inventory incomplete | A new feed added | Process | Undetected leak | Update the inventory |
| Review done by the model owner | No independent check | Process | Missed leak | Independent review |
| Audit not rerun after update | The version changed | Process | Reintroduced leak | Change control |

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| Time zero is correctly specified | Leakage is defined relative to it | Vague boundary | Timestamp audit |
| Feature provenance is known | Timing can be checked | Unknown source | Trace the query |
| The split respects the unit of independence | No patient overlap | Row split | Count patients across folds |
| Duplicates are identified | No double counting | Near-duplicates | Match on feature vector |
| The clinician review is independent | It can catch what statistics miss | Reviewer bias | Separate reviewer |
| The single-feature screen is run | Deterministic proxies are caught | Omitted | Univariate AUC |
| The ablation is on held-out data | The change is honest | In-sample ablation | Nested predictions |
| The label is not a future-derived proxy | The outcome is valid | Label leakage | Label audit |
| The cohort is defined at time zero | No selection on the future | Post-outcome eligibility | Flow diagram |
| The audit is repeated after updates | New leaks are caught | Model change | Change control |
| The audit queries are recorded | The audit is reproducible | No scripts | Re-run |
| The reviewer is independent of the model owner | The review is credible | Owner bias | Separate reviewer |
| The residual risk is stated | The boundary is clear | Overclaim | Report |

The deepest limit is that the audit can only test the mechanisms it knows. A leak that is not in
the catalogue, or a feature whose provenance is opaque, can survive. The residual risk statement is
therefore part of the report, not a disclaimer.

## 12. Exercises

The exercises build the leakage audit in order: the timing inventory, duplicate detection, the
single-feature screen, ablation, the split audit, target encoding, a published-model audit, and the
written section. Each states a deliverable, an acceptance criterion, and a time budget.

**Exercise 1. Timing inventory.** For a feature list, classify each feature as pre-time-zero,
at-time-zero, post-time-zero, or unknown, with the evidence. Deliverable: a table. Acceptance
criteria: every feature classified; unknown features flagged; the generating event stated. Time: 45
minutes.

**Exercise 2. Duplicate detection.** Write a duplicate check for a dataset with patient and
encounter identifiers, including exact and near-duplicate matching. Deliverable: code and a count.
Acceptance criteria: exact and near duplicates reported; the matching rule stated; the false-positive
risk discussed. Time: 60 minutes.

**Exercise 3. Single-feature screen.** Compute the univariate AUC for every feature and list the top
10. Deliverable: a table. Acceptance criteria: values correct; suspects flagged with a reason.
Time: 45 minutes.

**Exercise 4. Ablation.** Remove each suspect feature, refit, and record the change in AUROC.
Deliverable: a waterfall table. Acceptance criteria: refit, not rescore; held-out data; order
stated. Time: 75 minutes.

**Exercise 5. Split audit.** Count patients and sites across folds for a proposed split, and
propose a corrected split. Deliverable: counts and a corrected design. Acceptance criteria: zero
patient overlap; temporal gap stated. Time: 45 minutes.

**Exercise 6. Target encoding.** Compare full-data and fold-safe target encoding on the same
cohort. Deliverable: a table and a 300-word analysis. Acceptance criteria: AUROC difference
reported; leakage mechanism explained. Time: 60 minutes.

**Exercise 7. Published-model audit.** Find a published clinical model and assess its feature
timing, split, and validation for leakage. Deliverable: a 600-word audit. Acceptance criteria: at
least four specific checks; claims cited. Time: 60 minutes.

**Exercise 8. Leakage section.** Write the leakage section of a model development report. Include
the time zero, the inventory, the timing classification, the overlap and duplicate counts, the fit
scope, the single-feature screen, the ablation waterfall, the clinician review, the label audit, the
residual risk, and the monitoring plan. Deliverable: two pages. Acceptance criteria: catalogue,
tests, waterfall, fixes, residual risk, and monitoring all present. Time: 75 minutes.

The section should be written so that a reviewer can rerun each test from the reported queries. A
section that describes the tests without the queries is a narrative, not an audit.
Deliverable: two pages. Acceptance criteria: catalogue, tests, waterfall, fixes, residual risk, and
monitoring all present. Time: 75 minutes.

The exercises are designed to be done in order. The timing inventory is the foundation; the
duplicate, screen, and ablation checks depend on it. The split audit and the encoding comparison
test the pipeline, and the published-model audit tests the learner's ability to apply the catalogue
to someone else's work.

The audit is the deliverable, and the model is the beneficiary. A learner who completes all eight
has a reproducible audit that can be attached to the model development report. The audit is the deliverable; the performance number is the consequence.

## 13. Self-check answers

**Answer to Exercise 1 (what a strong inventory shows).** A strong inventory has one row per
feature, the source table, the generating event, the availability timestamp, and the classification.
It flags the features with no timestamp and the features whose generating event is after time zero.
It does not accept a column name as evidence of timing.

**Answer to Exercise 2 (what a strong duplicate check shows).** A strong check reports exact
duplicates by identifier, near-duplicates by feature-vector distance, and the number of records
removed. It states the matching threshold and the false-positive risk. It also checks whether the
duplicates are true repeated measures, which should not be removed without grouping.

**Answer to Exercise 3 (what a strong screen shows).** A strong screen reports the univariate AUC
for every feature, sorted, with the top 10 flagged. A feature with an AUC above 0.95 is a suspect;
a feature with an AUC of 0.85 can be legitimate but should be reviewed. The screen is a filter, not
a verdict.

**Answer to Exercise 4 (what a strong waterfall shows).** A strong waterfall reports the naive
estimate, each fix in a stated order, and the honest estimate. It refits the model at each step and
evaluates on the same held-out sample. It notes that the individual contributions are order-
dependent and that the total is the relevant number.

**Answer to Exercise 5 (what a strong split audit shows).** A strong audit reports the number of
patients and sites in more than one fold, the number of duplicate records, and the temporal overlap.
A correct grouped split gives zero patient overlap. If the split is temporal, the audit also counts
the patients whose outcome window crosses the boundary and the patients admitted in both periods.

**Answer to Exercise 6 (what a strong encoding comparison shows).** A strong comparison reports the
AUROC with the encoding fitted on all data and with the encoding fitted inside the fold, on the same
held-out sample. It explains that the difference is the test outcomes entering the encoding, and it
states the smoothing parameter and the fallback for unseen categories. A difference of 0.02 or more
is material.

**Answer to Exercise 7 (what a strong published-model audit contains).** A strong audit reports the
prediction time, the feature sources, the split design, the encoding method, and the validation
setting. It checks whether any feature is recorded after the prediction time, whether patients span
the split, whether the performance is plausible for the outcome, and whether the model was
externally validated. It distinguishes a reporting gap from a leakage finding.

**Answer to Exercise 8 (what a complete leakage section contains).** A complete section includes
time zero, the feature inventory, the timing classification, the overlap and duplicate counts, the
preprocessing and encoding fit scope, the single-feature screen, the ablation waterfall, the
clinician review, the label and cohort audit, the residual risk, and the monitoring plan. It ends
with the honest estimate and a statement that the naive estimate is not deployable.

**Rubric for Exercises 5, 6, 7, and 8.** A strong submission (1) states time zero, (2) audits
timing, (3) audits the split and duplicates, (4) audits preprocessing and encoding, (5) quantifies
the inflation, and (6) revalidates after the fix. A competent submission meets four of the six. A
weak submission reports the naive performance without an audit.

## 14. Interview questions

**1. What is leakage?** A strong answer defines it as outcome information not available at
prediction time, and gives three examples. A weak answer confuses it with overfitting.

**2. What is the most common leakage in clinical data?** A strong answer names post-outcome
features and patient overlap, with examples. A weak answer says "using the test set".

**3. How do you detect a post-outcome feature?** A strong answer checks the availability timestamp,
the generating event, the single-feature AUC, and the clinician review. A weak answer checks the
column name.

**4. Why is a random split wrong with repeated patients?** A strong answer explains patient overlap
and grouped splitting. A weak answer says random splits are always fine.

**5. How do you quantify leakage?** A strong answer uses a performance waterfall from naive to
honest. A weak answer says to remove the feature and report the new number.

**6. Why is target encoding a leakage risk?** A strong answer explains fold safety and smoothing.
A weak answer says encoding is just preprocessing.

**7. What is site leakage?** A strong answer explains the site signature and site-held-out
validation. A weak answer says site is a legitimate feature.

**8. What is a residual risk statement?** A strong answer says it is the boundary of the audit and
the leaks that were not tested. A weak answer says the audit proves no leakage.

**9. How would you audit a published model?** A strong answer checks the timing, split, encoding,
and validation, and compares the performance with the literature. A weak answer trusts the reported
AUROC.

**10. What would you do if the honest performance is below the requirement?** A strong answer says
report it and reconsider the project, not revert to the leaky result. A weak answer says find
another feature.

**11. Why is a note-derived feature risky?** A strong answer says the note may be written after the
outcome and may mention it, even if the note's start timestamp is early. A weak answer says notes
are clinical data and therefore valid.

**12. What is label-derived metadata, and why is it dangerous?** A strong answer says it is a code or flag created from the label, so it carries the outcome even if it is not the label itself. A weak answer says metadata is harmless.

**13. How would you audit a feature with no timestamp?** A strong answer traces the generating event and the extraction query, and treats the feature as a suspect until then. A weak answer assumes it is available.

**14. Why can two leaks mask each other?** A strong answer says one feature can carry the signal when the other is removed, so the ablation should be grouped. A weak answer says one removal is enough.

**15. What is the difference between leakage and distribution shift?** A strong answer says leakage is present at development and absent at deployment, while shift is a change after deployment. A weak answer uses the terms interchangeably.

**16. Why is a feature inventory with provenance necessary?** A strong answer says timing cannot be checked without the generating event and the timestamp. A weak answer says the data dictionary is enough. A strong answer says leakage
is present at development and absent at deployment, while shift is a change after deployment. A
weak answer uses the terms interchangeably.

## 15. Key terms

- **Ablation** — removing a feature and measuring the change.
- **Association leakage** — leakage through a proxy recorded before the outcome.
- **Audit trail** — record of feature creation and transformation.
- **Backfill** — a value entered later for an earlier time.
- **Cohort selection** — defining the cohort, possibly with future information.
- **Data cutoff** — last date with complete data.
- **Duplicate record** — same encounter entered twice.
- **Entity overlap** — same patient or site across splits.
- **Feature inventory** — list of features with timing and source.
- **Future information** — data created after the prediction.
- **Immortal time** — follow-up during which the outcome cannot occur.
- **Index date** — date anchoring the cohort.
- **Information set** — features legitimately available at time zero.
- **Label-derived metadata** — code or flag created from the label.
- **Label leakage** — a label that encodes the features or the future.
- **Late-arriving data** — data that arrive after the event.
- **Learn-predict separation** — boundary between training and prediction information.
- **Leakage** — outcome information not available at prediction time.
- **Leakage audit** — structured review of the mechanisms.
- **Near-duplicate** — record differing only in an identifier.
- **No-time-machine rule** — features must predate the prediction.
- **Outcome proxy** — feature that is a deterministic consequence of the outcome.
- **Patient leakage** — same patient in training and test.
- **Performance waterfall** — sequence of estimates as leaks are removed.
- **Point-in-time correctness** — a feature is valid at time zero.
- **Post-outcome feature** — feature created after the outcome.
- **Preprocessing leakage** — fitting a transform outside the fold.
- **Provenance** — source and history of a feature.
- **Revalidation** — re-running validation after a fix.
- **Residual risk** — leakage that remains after the audit.
- **Single-feature AUC** — AUC of one feature alone.
- **Site leakage** — site signature or overlap across splits.
- **Site signature** — feature that identifies the site and its practice.
- **Target encoding leakage** — encoding categories with test outcomes.
- **Target leakage** — the outcome or its cause present in the features.
- **Temporal leakage** — using future information to predict the past.
- **Time zero** — moment the prediction is made.
- **Washout** — period before time zero to exclude prior events.

## 16. Sources

All URLs accessed 25 October 2026.

1. Kaufman S, Rosset S, Perlich C, Stitelman O. Leakage in data mining: formulation, detection, and
   avoidance. *ACM Transactions on Knowledge Discovery from Data* 2012;6(4):1-21.
   https://doi.org/10.1145/2382577.2382579
2. Davis SE, Greevy RA, Lasko TA, Walsh CG, Matheny ME. A framework for understanding label leakage
   in machine learning for health care. *Journal of the American Medical Informatics Association*
   2024;31(1):274-280. https://pmc.ncbi.nlm.nih.gov/articles/PMC10746313/
3. Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement. *BMJ* 2024;385:e078378.
   https://doi.org/10.1136/bmj-2023-078378
4. Moons KGM, Damen JAA, Kaul T, et al. PROBAST+AI. *BMJ* 2025;388:e082505.
   https://doi.org/10.1136/bmj-2024-082505
5. Wynants L, Van Calster B, Collins GS, et al. Prediction models for diagnosis and prognosis of
   covid-19. *BMJ* 2020;369:m1328. https://doi.org/10.1136/bmj.m1328
6. Wong A, Otles E, Donnelly JP, et al. External validation of a widely implemented proprietary
   sepsis prediction model. *JAMA Internal Medicine* 2021;181(8):1065-1070.
   https://doi.org/10.1001/jamainternmed.2021.2626
7. US Food and Drug Administration. Good Machine Learning Practice for Medical Device Development:
   Guiding Principles. 2021.
   https://www.fda.gov/medical-devices/artificial-intelligence-enabled-medical-devices/good-machine-learning-practice-medical-device-development-guiding-principles
8. US Food and Drug Administration. Marketing Submission Recommendations for a Predetermined Change
   Control Plan for Artificial Intelligence-Enabled Device Software Functions. 2024.
   https://www.fda.gov/regulatory-information/search-fda-guidance-documents/marketing-submission-recommendations-predetermined-change-control-plan-artificial-intelligence
9. European Union. Regulation (EU) 2024/1689 (Artificial Intelligence Act). 2024.
   https://eur-lex.europa.eu/eli/reg/2024/1689/oj
10. National Institute of Standards and Technology. Artificial Intelligence Risk Management
    Framework (AI RMF 1.0). NIST AI 100-1, 2023. https://doi.org/10.6028/NIST.AI.100-1
11. Steyerberg EW. *Clinical Prediction Models*. 2nd edition. Springer, 2019.
    https://doi.org/10.1007/978-3-030-16399-0
12. Harrell FE. *Regression Modeling Strategies*. 2nd edition. Springer, 2015.
    https://doi.org/10.1007/978-3-319-19425-7
13. scikit-learn developers. Common pitfalls and recommended practices.
    https://scikit-learn.org/stable/common_pitfalls.html
14. Kaufman S, Rosset S. Leakage in data mining. *Proceedings of the 17th ACM SIGKDD International
    Conference on Knowledge Discovery and Data Mining*, 2011.
    https://doi.org/10.1145/2020408.2020496
15. Agniel D, Kohane IS, Weber GM. Biases in electronic health record data due to processes within
    the healthcare system. *BMJ* 2018;361:k1479. https://doi.org/10.1136/bmj.k1479
16. Finlayson SG, Subbaswamy A, Singh K, et al. The clinician and dataset shift in artificial
    intelligence. *New England Journal of Medicine* 2021;385(3):283-286.
    https://doi.org/10.1056/NEJMc2104626
17. National Institute of Standards and Technology. Artificial Intelligence Risk Management
    Framework: Generative Artificial Intelligence Profile. NIST AI 600-1, 2024.
    https://doi.org/10.6028/NIST.AI.600-1
18. Liu X, Cruz Rivera S, Moher D, et al. CONSORT-AI extension. *Nature Medicine*
    2020;26:1364-1374. https://doi.org/10.1038/s41591-020-1034-x

A clean audit is the difference between a model and a demonstration. It is the evidence that the
performance is a property of the clinical signal, not of the data pipeline. The distinction is the
whole point of the topic. A model that looks good because of the pipeline is not a model of the
patient. The audit is how the team tells the difference. It is also the difference
between a claim that survives review and one that does not.

The audit is a habit, not a gate. A team that runs it every time will catch most leaks before
they reach a report.

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic: [`12_fairness-and-subgroups.md`](12_fairness-and-subgroups.md)
- Labels and outcomes: [`02_labels-and-outcomes.md`](02_labels-and-outcomes.md)
- Splitting and validation: [`03_splitting-and-validation.md`](03_splitting-and-validation.md)
- Preprocessing: [`04_preprocessing.md`](04_preprocessing.md)
- Survey file: [`../../06_machine_learning_foundations.md`](../../course/06_machine_learning_foundations.md)
- Labs: [`../../labs/`](../../labs/)
- Calibration lab: [`../../labs/06_calibration_decision_curve.ipynb`](../../labs/06_calibration_decision_curve.ipynb)
