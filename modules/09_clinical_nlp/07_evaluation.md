# Evaluation

> **Module 09 — Clinical NLP**
> Topic 07 of 10
> Estimated study time: 4 hours
> Prerequisites: Topics 03, 05, and 06
> Version 2026.3

## 1. Why this topic exists

A clinical NLP system can report a good number and still be unsafe to deploy. The number may be a
span F1 that says nothing about the cohort, a document accuracy that hides a subgroup failure, or a
positive predictive value computed at a prevalence that does not exist in practice. Evaluation is
the discipline of choosing the number that matches the decision and stating the conditions under
which it holds.

The problem is structural. Clinical NLP has at least five evaluation levels, and each answers a
different question.

- Span-level: did the system find the right text?
- Assertion-level: did it assign the right status?
- Concept-level: did it map to the right code?
- Document-level: did it label the document correctly?
- Patient-level: did it identify the right patient?
- Downstream: did the decision improve?

A vendor can report any of these, and the numbers are not comparable. A span F1 of 0.90 and a
patient-level sensitivity of 0.90 describe different systems. The first says the extractor is
accurate on text; the second says the cohort is complete. A buyer who compares them has made a
category error.

The unit problem is compounded by prevalence. Sensitivity and specificity are properties of the
system at a given operating point; PPV and NPV depend on prevalence. A system with sensitivity 0.80
and specificity 0.874 has PPV 0.667 at 24 percent prevalence and PPV 0.356 at 8 percent. Both are
correct. A report that presents only the first number has described a setting, not the system.

The reference standard problem compounds it again. Every metric is measured against labels with
their own error. If annotation disagreement produces a 3-point band, a 1-point model difference is
not interpretable. Topic 06 covers the annotation; this topic covers how to carry its uncertainty
into the evaluation.

The clinical review problem is the last layer. Automated metrics cannot detect an error that is
correct by the metric and harmful in context. A system that extracts "no resuscitation" as a
positive code status note may have perfect span F1 and a clinically dangerous output. Clinician
review of a sample, with a severity rating, is the only way to catch that class of error.

This topic builds an evaluation design from the decision backward. It covers span matching rules,
assertion accuracy, patient-level aggregation, confidence intervals, calibration, decision curves,
subgroup analysis, error severity, and reporting. The output is an evaluation plan and a results
table that a reviewer can interpret without asking what the denominator was.

The order matters. The decision determines the unit, the unit determines the reference standard,
the reference standard determines the match rule, and the match rule determines the metric. A team
that starts with a model and looks for a metric has reversed the chain, and the reversal is visible
in the report: the metric will be the one the model produces most naturally, not the one the
decision needs.

> **Note:** Every metric in this topic carries its unit. A sentence that says "F1 0.87" without a
> unit is incomplete. A sentence that says "span F1 0.87 with exact boundaries on 2,000 notes" is
> evaluable.

## 2. Learning outcomes

By the end of this topic you can:

1. **Select** the evaluation level that matches a stated decision and **justify** why lower levels
   are insufficient.
2. **Define** an exact, overlap, and concept-level span match rule, and **compute** precision,
   recall, and F1 under each.
3. **Compute** assertion accuracy conditioned on correctly extracted spans, and **explain** why
   conditioning matters.
4. **Aggregate** span results to document and patient levels under an explicit rule, and **report**
   the metric at each level.
5. **Compute** Wilson confidence intervals for proportions and **interpret** overlapping intervals.
6. **Recalculate** PPV at deployment prevalence and **explain** the difference from the sample
   value.
7. **Evaluate** calibration and decision-curve net benefit for a probabilistic output.
8. **Design** a clinician review with a severity scale and **report** error severity alongside error
   counts.

Outcomes 2 through 7 are computational; 1, 8, and the reporting components are design.

## 3. The core mental model

Evaluation is a chain from text to decision, with a metric at each link and a unit at each metric.

```text
  text
   |
   |  span match rule
   v
  SPAN metrics        precision, recall, F1        unit: span
   |
   |  conditioning on matched spans
   v
  ASSERTION metrics   accuracy, kappa               unit: span
   |
   |  concept mapping
   v
  CONCEPT metrics     accuracy, top-k               unit: concept
   |
   |  aggregation rule
   v
  DOCUMENT metrics    sensitivity, PPV              unit: document
   |
   |  patient aggregation
   v
  PATIENT metrics     sensitivity, PPV, NPV         unit: patient
   |
   |  decision analysis
   v
  DECISION metrics    net benefit, harm, cost       unit: decision
```

The chain is not a required pipeline; it is a map of claims. A report that jumps from span to
decision without the intermediate steps has skipped the links where error accumulates.

A second model is the **confusion matrix as a function of prevalence**.

```text
                 truth +     truth -
  predict +        TP          FP
  predict -        FN          TN

  sensitivity = TP / (TP + FN)      fixed by the system
  specificity = TN / (TN + FP)      fixed by the system
  PPV         = TP / (TP + FP)      depends on prevalence
  NPV         = TN / (TN + FN)      depends on prevalence
```

Sensitivity and specificity travel with the system. PPV and NPV travel with the setting. A report
that gives only PPV has described one setting and cannot be extrapolated without the other three
numbers.

A third model is the **uncertainty stack**.

```text
  sampling uncertainty      -> confidence interval
  reference-standard error  -> adjudication band
  distribution shift        -> subgroup and site analysis
  temporal drift            -> era analysis
```

Each layer widens the interval around the point estimate. A 95 percent confidence interval that
ignores the reference-standard band understates the uncertainty. A subgroup analysis that ignores
sampling uncertainty over-interprets a small difference.

A fourth model is the **evaluation funnel**, which shows how many cases survive each condition.

```text
  all patients                1000
    eligible for analysis      940   (60 excluded, reasons reported)
      reference positive       240
        span correct           192   (patient-level true positive)
          assertion correct    181   (11 attribution or status errors)
            documented in time 175   (6 temporal errors)
```

The funnel is useful because it localizes loss. A drop at "eligible" is a data or protocol problem.
A drop at "span correct" is extraction. A drop at "assertion correct" is status. A drop at "in time"
is temporality. The final number is the decision-level truth, and every step is reportable.

A fifth model is the **decision curve**. A model's net benefit at a threshold probability is the
share of true positives minus the weighted share of false positives.

```text
net benefit = TP/N - (FP/N) x (p_t / (1 - p_t))
```

The threshold p_t is the probability at which a clinician would act. At low thresholds, the model
should behave like "treat all"; at high thresholds, like "treat none." A model is useful if its
curve is above both reference strategies across the range of plausible thresholds. The curve
converts a metric into a decision, and it is the most direct answer to "should we use this?"

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| Evaluation level | The object a metric is computed over | Left unstated | Span, document, patient, decision |
| Span match | The rule for deciding whether a predicted span equals a gold span | Assumed to be exact | Rule |
| Exact match | Identical start and end offsets | Called strict when it includes type | Rule |
| Overlap match | Any character overlap between spans | Called partial credit | Rule |
| Boundary error | A span that differs at start or end | Counted as a complete miss | Error |
| Precision | TP / (TP + FP) | Called PPV at patient level | Proportion |
| Recall | TP / (TP + FN) | Called sensitivity at patient level | Proportion |
| F1 | Harmonic mean of precision and recall | Reported without a match rule | Proportion |
| Sensitivity | TP / (TP + FN) at the decision unit | Confused with recall at span level | Proportion |
| Specificity | TN / (TN + FP) | Omitted from reports | Proportion |
| PPV | TP / (TP + FP) at the decision unit | Called precision | Proportion |
| NPV | TN / (TN + FN) | Omitted | Proportion |
| Prevalence | Share of the unit that is positive | Confused with incidence | Proportion |
| Wilson interval | A binomial confidence interval | Confused with a standard error | Interval |
| Calibration | Agreement between predicted probability and observed frequency | Confused with discrimination | Curve |
| Discrimination | Ability to rank positives above negatives | Called calibration | AUROC |
| AUROC | Area under the receiver operating characteristic curve | Reported as clinical utility | Probability |
| Net benefit | True positive share minus weighted false positive share | Confused with accuracy | Proportion |
| Decision threshold | The probability at which action is taken | Chosen arbitrarily | Probability |
| Subgroup analysis | Metrics computed within a stratum | Confused with a post-hoc test | Stratum |
| Error severity | Clinical harm potential of an error | Counted as a binary error | Scale |
| Reference-standard band | Metric range across defensible standards | Confused with a CI | Range |
| Abstention rate | Share of cases where the system does not answer | Confused with error rate | Proportion |
| Coverage | Share of cases the system handles | Confused with recall | Proportion |
| External validation | Evaluation on a different site or era | Confused with a held-out split | Study |
| Temporal validation | Evaluation on later data | Confused with random split | Study |
| Conjunction fallacy | Reporting a component metric as a system metric | Common in vendor claims | Error |
| Denominator | The unit count a metric divides by | Left unstated | Count |
| Evaluation funnel | Staged counts from eligible cases to decision-level hits | Confused with a flow diagram | Count |
| Spectrum bias | Performance change due to case mix | Confused with sampling error | Bias |
| Prospective validation | Evaluation during live deployment | Confused with temporal validation | Study |
| Missing-label sensitivity | Metric range under extreme treatment of unlabelled items | Omitted | Range |

Three definitions require emphasis.

**Precision versus PPV.** At span level, precision is the share of predicted spans that match gold.
At patient level, PPV is the share of flagged patients who are true positives. The formulas look
similar and the units differ. A report that says "precision 0.85" without a unit is ambiguous, and
the ambiguity usually favors the vendor.

**Discrimination versus calibration.** AUROC measures ranking. A model can rank well and be badly
calibrated, meaning its predicted probabilities do not match observed frequencies. A clinical
decision that uses a probability threshold needs calibration; a decision that uses a ranking needs
discrimination. Most clinical NLP papers report AUROC and not calibration, which is a gap.

**Statistical versus clinical significance.** A difference can be statistically significant and
clinically irrelevant, or clinically important and statistically uncertain. The evaluation should
report both the interval and the decision consequence. A 0.02 F1 improvement with a 0.05-wide
interval is not a finding; a 0.02 improvement in a safety-critical recall with a narrow interval may
be.

## 5. The mechanics, step by step

This section is the procedure for designing and executing an evaluation that matches a decision.

### Step 1. State the decision and the unit

**Input:** The intended use.

**Output:** A decision statement and a unit.

**Rationale:** The unit is the denominator. A decision about a patient requires a patient-level
metric. A decision about a document requires a document-level metric. Starting from the metric
instead of the decision produces a report that answers a question nobody asked.

**Branch point:** If the decision has multiple stages, each stage needs its own unit. A triage
system that flags a message and then routes it has a message-level flag metric and a routing metric.

### Step 2. Define the reference standard

**Input:** The unit and the construct.

**Output:** A reference standard with a stated construction, sample, and uncertainty.

**Rationale:** The reference standard bounds every metric. Its error must be estimated, not assumed
away.

**Branch point:** If the reference standard is a single annotator, the evaluation should report the
metric against a second annotator as a sensitivity analysis. If it is adjudicated, the
pre-adjudication labels provide the band.

### Step 3. Define the span match rule

**Input:** Span annotations and predictions.

**Output:** A written rule: exact, overlap, token-level, or concept-level.

**Rationale:** The rule changes the metric, sometimes by 5 to 10 points. It must be fixed before
the model is evaluated and stated in the report.

**Branch point:** Match rules have different biases.

| Rule | Definition | Bias | Use |
|---|---|---|---|
| Exact + type | Identical offsets and entity type | Penalizes boundary differences | Strict extraction |
| Exact, type-agnostic | Identical offsets, any type | Ignores type errors | Boundary-focused tasks |
| Overlap + type | Any overlap and same type | Rewards over-long spans | Screening |
| Token-level | Per-token labels | Rewards partial spans | Model diagnostics |
| Concept-level | Same normalized concept | Ignores boundaries | Phenotyping |

### Step 4. Compute span metrics with intervals

**Input:** Matched spans.

**Output:** Precision, recall, F1, and confidence intervals.

**Rationale:** A point estimate without an interval hides sampling uncertainty. The interval is
especially important for subgroups and rare entities.

**Branch point:** If the interval is wide, either collect more annotations or narrow the claim.
A wide interval on a safety-critical metric is a finding in itself.

### Step 5. Compute assertion accuracy conditioned on spans

**Input:** Matched spans with gold and predicted assertion labels.

**Output:** Assertion accuracy, and accuracy by assertion category.

**Rationale:** Assertion accuracy computed over all mentions mixes extraction errors with
classification errors. Conditioning on correctly extracted spans isolates the assertion task.

**Branch point:** Report both conditioned and end-to-end accuracy. The conditioned number answers
"how good is the assertion classifier"; the end-to-end number answers "how good is the system."

### Step 6. Compute concept-level accuracy

**Input:** Normalized concepts.

**Output:** Exact-match accuracy, accuracy at the parent level, and top-k accuracy.

**Rationale:** Concept mapping has a granularity problem. A system that maps to a parent concept
may be clinically adequate and technically wrong. Reporting only exact match understates it;
reporting only parent match overstates it.

**Branch point:** If the downstream use tolerates parent concepts, report parent accuracy as the
primary metric and exact accuracy as secondary. State the choice.

### Step 7. Aggregate to the document level

**Input:** Span and assertion output per document.

**Output:** Document labels under a stated rule.

**Rationale:** The aggregation rule is a policy. "Any affirmed mention" and "affirmed mention in the
assessment" produce different document labels from the same spans.

**Branch point:** If the document-level metric is much lower than the span metric, the aggregation
or the entity distribution is the cause. Investigate before tuning the model.

### Step 8. Aggregate to the patient level

**Input:** Document labels or spans per patient.

**Output:** Patient labels and a confusion matrix.

**Rationale:** Patient-level aggregation is where cohort decisions are made. It must be explicit and
tested against alternatives.

**Branch point:** The choice among rules changes sensitivity and PPV in opposite directions. Report
the rule and, if feasible, a sensitivity analysis across two rules.

### Step 9. Recompute PPV at deployment prevalence

**Input:** Sensitivity, specificity, and deployment prevalence.

**Output:** PPV and NPV at the deployment setting.

**Rationale:** Sample prevalence is rarely deployment prevalence. A model evaluated in an enriched
sample will have an inflated PPV.

**Branch point:** If deployment prevalence is unknown, report PPV across a range of plausible
prevalences. A table is more honest than a point estimate.

### Step 10. Assess calibration

**Input:** Predicted probabilities and outcomes.

**Output:** A reliability diagram, calibration slope, and intercept.

**Rationale:** A threshold on an uncalibrated score is arbitrary. Calibration makes the score
interpretable and the threshold defensible.

**Branch point:** If calibration is poor, recalibrate on a held-out set or restrict the output to a
ranking. Do not apply a threshold to an uncalibrated score without stating the risk.

### Step 11. Compute decision-curve net benefit

**Input:** Predicted probabilities, outcomes, and a range of thresholds.

**Output:** Net benefit curves for the model, treat-all, and treat-none.

**Rationale:** Net benefit answers whether the model improves decisions relative to simple
strategies. It incorporates the cost trade-off that F1 hides.

**Branch point:** If the model's curve is below treat-all at clinically plausible thresholds, the
model does not add value at those thresholds, regardless of its AUROC.

### Step 12. Stratify by site, note type, and subgroup

**Input:** Metrics and metadata.

**Output:** Stratified metrics with intervals and sample sizes.

**Rationale:** An aggregate can hide a stratum where the model fails. Equity and transportability
both require stratified reporting.

**Branch point:** If a stratum has fewer than about 30 positive cases, its interval will be wide.
Report the count and the interval rather than a point estimate alone.

### Step 13. Conduct a clinician severity review

**Input:** A sample of errors and a severity scale.

**Output:** Error counts by severity and category.

**Rationale:** Not all errors are equal. A missed historical mention and a missed active
contraindication are both false negatives and have different consequences.

**Branch point:** Severity scales vary; a three-level scale (no harm, moderate, severe) is usually
enough. The scale must be defined before review and applied blind to the system identity.

### Step 14. Estimate the reference-standard band

**Input:** Metrics against different defensible reference standards.

**Output:** A range of metric values.

**Rationale:** The band prevents overinterpretation of small differences. It is a sensitivity
analysis, not a confidence interval.

**Branch point:** If the band is wider than the difference the study reports, the difference is not
established. State that plainly.

### Step 15. Report with units and conditions

**Input:** All metrics.

**Output:** A results table with unit, denominator, prevalence, interval, and conditions.

**Rationale:** The table is the artifact a reviewer uses. If it does not state the unit and the
denominator, the reviewer cannot interpret it.

**Branch point:** The report should include a limitations paragraph for each metric, not one
generic paragraph at the end. A limitation attached to the metric is harder to ignore.

### Step 16. Model error propagation explicitly

**Input:** Component metrics and the aggregation rule.

**Output:** A projected decision-level metric under independence and a comparison with the observed
value.

**Rationale:** Component metrics do not combine independently. Errors cluster by patient, note type,
and site. A projection that assumes independence is a lower bound, and a report that presents it as
an estimate overstates certainty.

**Branch point:** The projection has three uses.

| Use | Method | Caveat |
|---|---|---|
| Sanity check | Multiply component accuracies | Understates clustering |
| Scenario analysis | Vary correlation assumptions | Requires a correlation estimate |
| Decision threshold | Plug lower bound into net benefit | Conservative, not realistic |

In the worked example, cell-level accuracy of 0.866 across 13 criteria would predict 0.154
all-correct patients under independence, while the observed value was 0.535. The ratio of 3.5 is the
clustering factor. A report that omitted the observed value would have understated patient-level
accuracy by a factor of three.

### Step 16a. Fix the denominator for every reported number

**Input:** The metric list.

**Output:** A denominator and a unit for each metric, written into the results table.

**Rationale:** Many evaluation errors are denominator errors. A recall computed over gold mentions
and a recall computed over documents are different quantities with the same name. The table forces
the distinction.

**Branch point:** The denominator choices below cover most clinical NLP metrics.

| Metric | Numerator | Denominator | Unit |
|---|---|---|---|
| Span precision | Matched predicted spans | All predicted spans | Span |
| Span recall | Matched gold spans | All gold spans | Span |
| Assertion accuracy | Correct assertion labels | Matched spans | Span |
| Concept accuracy | Correct normalized concepts | Matched spans with a gold concept | Concept |
| Document sensitivity | Positive documents flagged | All positive documents | Document |
| Document PPV | True positive documents | All flagged documents | Document |
| Patient sensitivity | Positive patients flagged | All positive patients | Patient |
| Patient PPV | True positive patients | All flagged patients | Patient |
| Abstention rate | Cases with no output | All cases | Case |
| Coverage | Cases with an output | All eligible cases | Case |

A report that uses "accuracy" for assertion and "accuracy" for the patient label has used one word
for two denominators. The table prevents the ambiguity, and it makes the results section longer by
one column in exchange for being interpretable.

### Step 17. Design external and temporal validation

**Input:** The deployment setting and the available data.

**Output:** A validation plan with a site split, a time split, or both.

**Rationale:** A random split within one site and one era measures generalization to new documents
from the same distribution, not to a new distribution. Deployment changes the distribution.

**Branch point:** Validation designs answer different questions.

| Design | Split | Question answered | Limitation |
|---|---|---|---|
| Random holdout | Random documents | Performance on similar documents | Same site and era |
| Patient-level holdout | Random patients | Performance on new patients | Same site and era |
| Temporal | Later period | Performance after drift | One site |
| External site | Different site | Transportability | Site-specific differences confounded |
| Prospective | Live deployment | Real-world performance | Expensive; requires governance |

The strongest evidence is prospective, and it is rarely available before deployment. The practical
sequence is patient-level holdout, then temporal, then external, then prospective monitoring. Each
step should be reported separately; pooling them hides which distribution changed.

### Edge cases that decide evaluation quality

**Negation scope.** A span-level metric counts a negated mention as correctly extracted if the span
matches. The assertion metric then judges the status. A report that combines them into one "accuracy"
hides which stage failed.

**Hypotheticals.** A hypothetical mention extracted as present is an assertion error, not an
extraction error. The evaluation should separate the two, because the fixes differ.

**Family history.** A family mention extracted as a patient finding is an attribution error. At the
patient level, it can create a false positive that no span metric reveals.

**Copy-forward duplicates.** If the evaluation sample contains duplicated notes, a single mention
can be counted many times, inflating both gold and predicted counts. Deduplicate before sampling or
weight by patient.

**Template boilerplate.** A template phrase extracted as a finding inflates precision if the gold
standard also counted it. The reference standard's treatment of template text must match the
system's.

**Abbreviation ambiguity.** An abbreviation expanded wrongly produces a concept error that a
type-level span metric does not catch. Concept-level evaluation is required.

**Section boundaries.** A mention extracted from the wrong section may still match the gold span and
receive credit, while the downstream aggregation uses it incorrectly. The evaluation should include
a section-stratified error review.

**Unicode and OCR noise.** If the gold standard was annotated on normalized text and the system runs
on raw text, exact matching fails for reasons unrelated to the model. Normalize both or report the
mismatch.

## 6. Worked example

A health system is evaluating a phenotype for "documented heart failure" across 1,000 patients and
2,000 notes. The decision is cohort inclusion for a care-management program. The unit is the
patient. The team evaluates at span, document, and patient levels to see where error enters.

### Reference standard

Two clinicians review all 2,000 notes for each patient, with adjudication. The patient-level
reference standard is 240 positive patients (24.0 percent). The span reference standard contains
6,000 mentions.

### Span results

The system predicts 5,400 spans. Exact matching with entity type yields 4,620 true positives.

Precision = 4,620 / 5,400 = 0.856.
Recall = 4,620 / 6,000 = 0.770.
F1 = 2 × 0.856 × 0.770 / 1.626 = 1.318 / 1.626 = 0.810.

Wilson 95 percent intervals, with n = 5,400 for precision and n = 6,000 for recall:

| Metric | Estimate | 95% Wilson interval |
|---|---:|---|
| Span precision | 0.856 | 0.846 to 0.865 |
| Span recall | 0.770 | 0.759 to 0.780 |
| Span F1 | 0.810 | not a simple proportion; report components |

F1 is a ratio of two proportions, so a Wilson interval on F1 is not directly defined. The correct
practice is to report intervals for precision and recall and treat F1 as a summary. A bootstrap
interval on F1 is also acceptable if the resampling unit is stated.

### Document results

There are 2,000 notes, of which 700 are positive in the reference standard. The system flags 760
notes, with 560 true positives.

Document sensitivity = 560 / 700 = 0.800.
Document PPV = 560 / 760 = 0.737.
Document F1 = 2 × 0.737 × 0.800 / 1.537 = 1.179 / 1.537 = 0.767.

The document F1 (0.767) is lower than the span F1 (0.810). The difference is expected: a document
is positive if any mention is affirmed and in the right section, so a single missed or misasserted
mention can flip the document.

### Patient results

At the patient level, a patient is positive if at least one document is positive.

| | Reference positive | Reference negative |
|---|---:|---:|
| System positive | 192 | 96 |
| System negative | 48 | 664 |

Sensitivity = 192 / 240 = 0.800. Wilson interval with n = 240: 0.745 to 0.846.
Specificity = 664 / 760 = 0.874.
PPV = 192 / 288 = 0.667. Wilson interval with n = 288: 0.610 to 0.719.
NPV = 664 / 712 = 0.933.
Accuracy = 856 / 1,000 = 0.856.

The patient-level PPV of 0.667 is the number that matters for the care-management program: one in
three flagged patients is not in the reference cohort. The span F1 of 0.810 and the patient PPV of
0.667 are both true, and they answer different questions.

### Where the patient-level errors come from

The 48 false negatives and 96 false positives are reviewed.

| Error category | FN | FP |
|---|---:|---:|
| No mention in the selected notes | 22 | 0 |
| Assertion error (negated or historical) | 14 | 26 |
| Family history attributed to patient | 0 | 40 |
| Abbreviation or synonym miss | 8 | 0 |
| Section or indication text | 4 | 18 |
| Other | 0 | 12 |
| Total | 48 | 96 |

Two categories dominate: source coverage (22 FN) and family history (40 FP). Neither is a modeling
problem in the narrow sense. The first requires more note types; the second requires an attribution
filter. A report that presented only F1 would not have surfaced either.

### PPV at deployment prevalence

The evaluation sample has 24 percent prevalence because it was enriched for likely positives. The
care-management program's population has 8 percent prevalence. Using sensitivity 0.800 and
specificity 0.874:

PPV at 8 percent = (0.800 × 0.080) / ((0.800 × 0.080) + (0.126 × 0.920))
= 0.0640 / (0.0640 + 0.1159)
= 0.0640 / 0.1799
= 0.356.

At 8 percent prevalence, 64.4 percent of flagged patients would not be in the reference cohort. The
program would screen 1,000 patients, flag about 180, and find about 64 true cases. The operational
consequence is a review workload dominated by false positives. The sample PPV of 0.667 would have
understated that workload by a factor of nearly two.

A prevalence table is more useful than a single recalculation:

| Prevalence | PPV | Flagged per 1,000 | True positives per 1,000 |
|---:|---:|---:|---:|
| 5% | 0.250 | 164 | 41 |
| 8% | 0.356 | 180 | 64 |
| 15% | 0.523 | 214 | 112 |
| 24% | 0.667 | 288 | 192 |

The table makes the setting dependence explicit. A program operating at 5 percent prevalence should
not use a system with this specificity without a second-stage review.

### Calibration and net benefit

The system also outputs a probability. The team bins predictions and compares them with observed
frequencies.

| Predicted bin | Patients | Observed positive rate |
|---|---:|---:|
| 0.0 to 0.2 | 540 | 0.06 |
| 0.2 to 0.4 | 210 | 0.24 |
| 0.4 to 0.6 | 130 | 0.47 |
| 0.6 to 0.8 | 80 | 0.69 |
| 0.8 to 1.0 | 40 | 0.88 |

The observed rates track the bins reasonably well. The calibration slope, estimated by regressing
the outcome on the logit of the predicted probability, is 0.94, and the intercept is 0.03. A slope
below 1 indicates slight overconfidence at the extremes; the intercept near 0 indicates no overall
bias.

Net benefit at a threshold of 0.24, the sample prevalence:

Model: TP/N − (FP/N) × (p_t / (1 − p_t)) = 192/1,000 − (96/1,000) × (0.24/0.76)
= 0.192 − 0.096 × 0.3158
= 0.192 − 0.0303
= 0.162.

Treat all: 240/1,000 − (760/1,000) × 0.3158 = 0.240 − 0.240 = 0.000.
Treat none: 0.000.

At a threshold of 0.08:

Model: 64/1,000 − (116/1,000) × (0.08/0.92) = 0.064 − 0.116 × 0.0870 = 0.064 − 0.0101 = 0.054.
Treat all: 80/1,000 − (920/1,000) × 0.0870 = 0.080 − 0.080 = 0.000.

The model has positive net benefit at both thresholds, and it exceeds treat-all. The net benefit is
smaller at the lower prevalence, which is consistent with the PPV drop.

### Two diagnostics that reveal a broken evaluation

**Diagnostic 1: PPV versus prevalence.** Plot PPV against prevalence for the observed sensitivity
and specificity. If the reported PPV is far above the curve at the deployment prevalence, the
evaluation sample was enriched and the report is describing the sample, not the setting. In this
example, 0.667 at 24 percent versus 0.356 at 8 percent is the diagnostic. A report that does not
show the curve invites the reader to assume the sample value applies.

**Diagnostic 2: subgroup sensitivity with intervals.** Stratify by site.

| Site | Patients | Reference positive | True positive | Sensitivity | 95% Wilson |
|---|---:|---:|---:|---:|---|
| A | 700 | 140 | 129 | 0.921 | 0.865 to 0.956 |
| B | 300 | 100 | 63 | 0.630 | 0.532 to 0.718 |

The intervals do not overlap. The aggregate sensitivity of 0.800 hides a site where the system
misses 37 percent of positive patients. A report with only the aggregate would support a deployment
that fails at site B. The diagnostic is the stratified table with intervals, not a significance
test.

### Vary one assumption: use overlap matching

Suppose the team changes the span rule from exact to overlap. An additional 330 spans match,
raising true positives to 4,950 with the same 5,400 predictions.

Precision = 4,950 / 5,400 = 0.917.
Recall = 4,950 / 6,000 = 0.825.
F1 = 2 × 0.917 × 0.825 / 1.742 = 1.513 / 1.742 = 0.868.

Span F1 rises by 5.8 points, from 0.810 to 0.868. The patient-level confusion matrix is unchanged,
because the boundary differences do not change which patients are flagged.

| Metric | Exact | Overlap |
|---|---:|---:|
| Span F1 | 0.810 | 0.868 |
| Document F1 | 0.767 | 0.767 |
| Patient sensitivity | 0.800 | 0.800 |
| Patient PPV | 0.667 | 0.667 |

The conclusion is that the match rule changes the span claim and not the decision claim. A report
that switched rules without saying so would appear to improve the system by 5.8 points while
changing nothing clinically. The rule must be fixed and disclosed.

## 7. Code, query, or calculation

The following module implements span matching under exact and overlap rules, patient aggregation,
Wilson intervals, prevalence-adjusted PPV, and net benefit. It uses only the standard library.

```python
import math


def match_spans(gold, pred, mode="exact"):
    """gold and pred are lists of (start, end, type). Greedy one-to-one."""
    used, tp = set(), 0
    for g in gold:
        for j, p in enumerate(pred):
            if j in used or g[2] != p[2]:
                continue
            if mode == "exact" and g[:2] == p[:2]:
                tp += 1
                used.add(j)
                break
            if mode == "overlap" and max(g[0], p[0]) < min(g[1], p[1]):
                tp += 1
                used.add(j)
                break
    return tp, len(pred) - len(used), len(gold) - tp


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r)) if p + r else 0.0


gold = [(10, 22, "PROBLEM"), (30, 41, "PROBLEM"), (50, 60, "DRUG")]
pred = [(10, 22, "PROBLEM"), (30, 38, "PROBLEM"), (50, 60, "DRUG")]
for mode in ("exact", "overlap"):
    p, r, f1 = prf(*match_spans(gold, pred, mode))
    print(f"{mode}: p={p:.3f} r={r:.3f} f1={f1:.3f}")


def wilson(k, n, z=1.96):
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


for k, n in [(192, 240), (192, 288)]:
    lo, hi = wilson(k, n)
    print(f"{k}/{n} = {k/n:.3f} CI {lo:.3f} to {hi:.3f}")


def ppv_at(sens, spec, prev):
    tp = sens * prev
    fp = (1 - spec) * (1 - prev)
    return tp / (tp + fp)


for prev in (0.05, 0.08, 0.24):
    print(f"prev {prev:.2f}: PPV {ppv_at(0.8, 0.874, prev):.3f}")


def net_benefit(tp, fp, n, pt):
    return tp / n - (fp / n) * (pt / (1 - pt))


print("NB model @0.24:", round(net_benefit(192, 96, 1000, 0.24), 4))
print("NB treat all @0.24:", round(net_benefit(240, 760, 1000, 0.24), 4))
```

Expected output:

```text
exact: p=0.667 r=0.667 f1=0.667
overlap: p=1.000 r=1.000 f1=1.000
192/240 = 0.800 CI 0.745 to 0.846
192/288 = 0.667 CI 0.610 to 0.719
prev 0.05: PPV 0.250
prev 0.08: PPV 0.356
prev 0.24: PPV 0.667
NB model @0.24: 0.1617
NB treat all @0.24: 0.0
```

The first two lines show the match-rule effect on a single example: one boundary-shifted span moves
the score from 0.667 to 1.000. The Wilson intervals reproduce Section 6. The PPV table shows the
prevalence dependence. The net benefit shows the model above treat-all at the sample threshold.

A second block computes the subgroup comparison and the reference-standard band:

```python
import math


def wilson(k, n, z=1.96):
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


site_a = wilson(129, 140)
site_b = wilson(63, 100)
print(f"site A: {129/140:.3f} CI {site_a[0]:.3f} to {site_a[1]:.3f}")
print(f"site B: {63/100:.3f} CI {site_b[0]:.3f} to {site_b[1]:.3f}")
print("overlap:", site_a[0] <= site_b[1])

band = 0.834 - 0.802
print("reference-standard band:", round(band, 3))
```

Expected output:

```text
site A: 0.921 CI 0.865 to 0.956
site B: 0.630 CI 0.532 to 0.718
overlap: False
reference-standard band: 0.032
```

The subgroup intervals do not overlap, which supports the claim of a site difference. The band of
0.032 sets the minimum difference that the evaluation can resolve against its reference standard.

## 8. Data, standards, and regulatory context

**FDA device validation expectations.** The FDA's January 2025 draft guidance on AI-enabled device
software functions recommends describing validation study protocols, endpoints, reference standards,
and results, including subgroup performance. It distinguishes internal validation from external
validation and asks for repeatability and reproducibility evidence. The guidance is a draft, not a
final rule. For a text-based device, the evaluation plan in this topic maps directly onto the
guidance's validation section.

**ONC HTI-1 performance attributes.** The HTI-1 final rule's 31 predictive DSI source attributes
include quantitative performance measures for validity and fairness in test and external data. A
vendor that reports a single F1 without a unit will struggle to populate those attributes. The
practical implication is that the evaluation design should be built to produce the attributes, not
retrofitted after certification.

**TRIPOD and prediction-model reporting.** The TRIPOD+AI statement (2024) extends the TRIPOD
reporting guidance to prediction models that use artificial intelligence. It asks for a description
of the data source, the predictors, the outcome, the sample size, the missing data handling, the
model development, the performance measures, and the fairness assessment. The statement is a
reporting convention, not a regulation, and it is a useful checklist for a clinical NLP validation
report.

**STARD and diagnostic accuracy.** The Standards for Reporting of Diagnostic Accuracy Studies
(STARD) guidance is relevant when the NLP system is evaluated as a diagnostic test. It asks for the
reference standard, the flow of participants, the indeterminate results, and the time interval
between index test and reference standard. A phenotype evaluated against chart review is a
diagnostic accuracy study in this sense, even if the word "diagnosis" is not used.

**n2c2 evaluation scripts.** The n2c2 challenges distribute evaluation scripts that define exact
match rules, including relaxed matching for some tracks. Using a published script makes the result
comparable to the literature; using a custom script without disclosure makes it incomparable. The
script version is part of the method.

**Confidence intervals.** There is no regulatory requirement to use Wilson rather than Wald
intervals. The practical requirement is to state the method and to avoid the Wald interval when
proportions are near 0 or 1, where it can produce bounds outside [0, 1]. Wilson or exact intervals
are safer defaults.

**Calibration reporting.** Regulatory guidance for prediction models increasingly asks for
calibration, not only discrimination. A model with a high AUROC and poor calibration can still
mislead at a threshold. The evaluation should report a calibration plot or slope and intercept, and
state the sample used for calibration.

**Subgroup reporting.** FDA guidance and HTI-1 both ask for subgroup performance. The statistical
caution is that subgroup analyses are underpowered and can produce spurious differences. The report
should give counts and intervals, pre-specify the subgroups, and avoid claiming a difference when
the intervals overlap.

**What standards do not settle.** No standard sets an acceptable F1 or PPV for a clinical NLP task.
The acceptable value depends on the decision, the cost of errors, and the alternatives. The
evaluation should state the threshold that the decision requires and whether the system meets it.

**Sample size and precision.** A validation sample should be sized for the smallest difference the
decision needs to detect. For a proportion near 0.8, a sample of 100 has a 95 percent interval width
of roughly 0.16; a sample of 400 has a width of roughly 0.08. A subgroup analysis multiplies the
requirement by the number of subgroups that must be compared. The calculation is a design input,
not a post-hoc excuse.

**Multiple comparisons.** A stratified report with many subgroups will produce some spurious
differences at conventional significance levels. The report should pre-specify the subgroups that
matter for the decision, report intervals rather than p-values, and treat exploratory differences as
hypotheses. This is a convention, not a regulatory requirement, and it is widely ignored in
published clinical NLP.

**Handling missing reference labels.** Some items will be unlabelled because the annotator could
not determine the answer. Excluding them biases the metric toward easy cases. The report should
state the count and the reason, and should present a sensitivity analysis that treats them as
positive and as negative. A missing label is information, not a nuisance.

**Evaluation of generated text.** For summarization and question answering, the unit is the claim
or the answer, and the reference standard is often a clinician's judgment of support. Exact-match
metrics such as ROUGE correlate poorly with that judgment. The evaluation should report claim-level
support, omission, and severity, and should state the decomposition rule used to segment the output
into claims.

> **Note:** A regulatory submission is not a journal article. It requires the protocol, the
> analysis plan, and the results to be consistent, and it requires the reference standard and the
> subgroups to be pre-specified. A post-hoc subgroup finding is a hypothesis, not evidence.

## 9. Industry practice and case evidence

**Shared-task evaluation.** The n2c2 2018 cohort-selection track reported a best micro F1 of 0.91,
a mean of 0.799, and a median of 0.8227 across 86 test patients and 13 criteria, with inter-annotator
Cohen's kappa averaging 0.54. The i2b2 2010 challenge reported best F1 of 0.936 for assertion, 0.852
for concept extraction, and 0.737 for relation extraction on its test set. These are benchmark
results on specific corpora and units; they are not deployment estimates.

**Label noise in report-derived datasets.** CheXpert extracted labels from radiology reports,
including uncertainty labels, and the resulting dataset enabled large-scale supervision. The labels
inherit the reporting conventions of the source institution, and the dataset's own validation
showed that some labels are noisier than others. The lesson for evaluation is that a reference
standard derived from text has the same problem as a model derived from text: it is an
interpretation, not a measurement.

**Intrinsic versus extrinsic evaluation.** Velupillai et al. (2018) argued that clinical NLP is
typically evaluated with intrinsic, word- or document-level metrics, while health outcomes research
operates at the patient or population level. The gap between the two is the subject of this topic.
The paper proposes a minimal reporting protocol covering data, approach, development, and
evaluation, and it is one of the few sources that treats the reporting gap as a methodological
problem rather than a formatting issue.

**Summarization evaluation at the claim level.** Recent clinical summarization studies evaluate
claims rather than documents. A 2024 blinded study of discharge summaries compared physician- and
model-generated summaries and reported similar hallucination rates (0.23 versus 0.31 per summary)
but more total errors for the model, driven by omissions. The unit is the claim or the summary, and
the evaluation distinguishes omission from fabrication. Inference: the claim-level design is
becoming the standard for generative outputs because document-level similarity metrics correlate
poorly with clinical judgment.

**De-identification evaluation.** The 2014 i2b2/UTHealth de-identification challenge reported a
best entity-based micro F1 of 0.964 and a mean of 0.872 across systems, with token-level
inter-annotator F1 of 0.927. The evaluation reported per-category results, not only an aggregate,
because the identifier types have different base rates and different consequences. The design is a
model for any safety-critical evaluation.

**Industry practice.** Operational teams increasingly report a "model card" style summary with
intended use, training data, performance by subgroup, and limitations. The practice is uneven, and
the performance section often lacks the unit and the prevalence. A buyer should ask for the
evaluation plan, not only the summary. Inference: the teams that can produce the plan are the ones
whose numbers are interpretable.

**Benchmark reporting in vendor materials.** A common pattern is a single number described as
"accuracy" or "F1" without a unit, a corpus description, or a reference standard. The number is
usually the best result on a public benchmark, which is a legitimate technical achievement and not a
deployment estimate. A buyer can test the claim by asking for the same metric on the buyer's note
types and decision unit. A vendor that cannot produce it has not evaluated the deployment.

**Evaluation of generative outputs.** Studies of clinical summarization increasingly report
claim-level support and omission rates rather than similarity to a reference summary. The shift
reflects the finding that n-gram overlap metrics correlate poorly with clinician judgment. The
practical implication for an NLP team is that a generative feature needs a claim-level reference
standard, which is expensive, and a document-level similarity metric is not a substitute. Inference:
the evaluation cost of generative features is higher than the evaluation cost of extraction
features, and it should be budgeted before the feature is built.

**External validation in the literature.** Published external validations of clinical NLP models
are less common than internal validations, and the gap is larger for text models than for
structured-data models because note templates and documentation practices differ by site. A
systematic review of clinical prediction models has repeatedly found that most models are validated
internally only. The pattern is a caution for buyers: "validated" usually means "validated where it
was built." This is an inference from the general prediction-model literature, not a count specific
to NLP.

**Post-deployment monitoring.** Operational teams monitor positive rates, override rates, and
periodic audit samples. A change in the positive rate can indicate a template change, a case-mix
change, or model drift. The monitoring design should distinguish these by tracking the input
distribution alongside the output. Inference: most published evaluations stop at deployment, and the
monitoring evidence is thin.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Metric substitution | Span F1 in the executive summary | Component metric for a system claim | Overstated capability | Report at the decision unit |
| Unit omitted | "F1 0.87" with no denominator | Ambiguity | Uninterpretable | Require unit and n |
| PPV at sample prevalence | PPV much higher than clinical experience | Enriched sample | Wrong workload estimate | Recompute at deployment |
| No confidence intervals | Point estimates only | Sampling uncertainty hidden | Overinterpretation | Wilson intervals |
| Single aggregate | No subgroup table | Stratum failure hidden | Inequitable deployment | Stratify |
| Exact match undisclosed | Score changes when rule changes | Boundary sensitivity | Incomparable results | State rule |
| Assertion accuracy over all mentions | Low accuracy attributed to classifier | Extraction errors mixed in | Wrong fix | Condition on matched spans |
| AUROC only | No calibration | Threshold arbitrary | Wrong operating point | Calibration curve |
| Accuracy on imbalanced data | High accuracy, low recall | Majority class dominance | Missed cases | Report P/R and confusion matrix |
| Reference standard treated as truth | No band | Measurement error ignored | Overinterpretation | Sensitivity analysis |
| Test set leakage | Suspiciously high score | Duplicate patients or notes | Invalid estimate | Patient-level split |
| Temporal leakage | Post-outcome features | Prediction time violated | Invalid estimate | Temporal validation |
| Error counts without severity | All errors equal | Clinical consequence ignored | Misprioritized fixes | Severity review |
| No abstention analysis | System answers all cases | Coverage hidden | Confident errors | Report coverage and abstention |
| Spectrum bias | Sample from a specialty clinic | Case mix differs from deployment | Wrong sensitivity | Describe selection; external validation |
| Missing labels excluded silently | Denominator smaller than sample | Non-random missingness | Biased metric | Report count; sensitivity analysis |
| Subgroup p-hacking | Many comparisons reported | No pre-specification | Spurious differences | Pre-specify; intervals |
| Generative output scored by ROUGE | Similarity metric for clinical summary | Weak correlation with judgment | Misleading quality claim | Claim-level review |

The red flags cluster around denominators and conditions. A metric without a denominator, a PPV
without a prevalence, a subgroup without a sample size, and a difference without an interval are all
the same failure: the number is presented without the conditions that make it mean something. A
reviewer can detect the pattern quickly by asking "of what?" after every metric.

A second cluster is evaluation leakage. A random split that puts the same patient in training and
test, a reference standard that was influenced by the model, or a test set drawn from the same
templated era all inflate performance. The diagnostic is the split audit and the reference-standard
construction description.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test |
|---|---|---|---|
| The reference standard measures the construct | Grounds all metrics | Guideline ambiguity | Agreement study |
| The sample represents deployment | Supports transportability | Enrichment or site skew | Compare strata to population |
| The match rule is fixed before evaluation | Comparability | Rule chosen after seeing results | Pre-register the rule |
| The aggregation rule matches the decision | Validity at the unit | Rule counts documents | Sensitivity analysis |
| Sensitivity and specificity are stable | Prevalence adjustment | Spectrum shift | Subgroup analysis |
| Deployment prevalence is known | PPV calculation | Unknown or changing setting | Prevalence table |
| Probabilities are calibrated | Threshold use | Uncalibrated output | Reliability diagram |
| Errors are independent across units | Simple projections | Clustering by patient or site | Compare projection with observed |
| The test set is independent | Unbiased estimate | Patient overlap or duplication | Split audit |
| Subgroup sample sizes support inference | Fair comparison | Small strata | Report counts and intervals |
| The reference-standard band is small | Supports comparisons | Large disagreement | Band calculation |
| Clinical severity is captured by labels | Prioritizes fixes | Metric-correct but harmful errors | Clinician review |

The strongest assumption is that the reference standard measures the construct. In the worked
example, the 40 family-history false positives were not extraction errors; they were attribution
errors that the guideline did not fully resolve. A different guideline would have produced a
different reference standard and a different PPV. The evaluation is only as stable as the construct.

The second strongest is that the sample represents deployment. The 24 percent prevalence in the
worked example was an enrichment choice, and it changed the PPV by 31 points. Any evaluation that
does not state its sampling design cannot be extrapolated, and any reader who assumes the sample
prevalence applies is making an error the report invited.

A limit of decision-curve analysis is that it assumes a single threshold probability and a
well-defined cost ratio. Clinical decisions often have heterogeneous thresholds across clinicians
and patients. The curve is a summary, not a prescription. It is most useful when the threshold
range is plausible and the model's advantage is robust across it.

Spectrum bias is another limit. Sensitivity and specificity depend on the case mix. A system
evaluated on patients who were tested because a clinician suspected the condition will perform
differently on a screening population with a different spectrum of disease. The evaluation should
describe how the sample was selected, not only its size. A sample drawn from a specialty clinic is
not a screening sample, even if both are called "patients."

A final limit is that evaluation measures the system against a reference standard, and the
reference standard is a snapshot. A model that meets the standard today may fail after a template
change, a coding change, or a shift in the patient population. The monitoring plan is part of the
evaluation, not an operational afterthought.

## 12. Exercises

**Exercise 7.1 — Unit selection (application, 30 minutes).**
Deliverable: for three decisions (readmission risk, message triage, quality measure), state the
evaluation unit and the metric that matches it.
Acceptance criteria: three decisions; unit and metric per decision; one sentence on why a span
metric would mislead.

**Exercise 7.2 — Match rule (calculation, 40 minutes).**
Deliverable: using the Section 7 example, compute exact and overlap precision, recall, and F1 by
hand, showing the matching.
Acceptance criteria: exact 0.667/0.667/0.667; overlap 1.0/1.0/1.0; a sentence on why the rule must
be fixed in advance.

**Exercise 7.3 — Wilson interval (calculation, 30 minutes).**
Deliverable: compute the Wilson interval for 192/240 and 192/288, showing the formula and each term.
Acceptance criteria: 0.745 to 0.846 and 0.610 to 0.719; a sentence on interpreting overlap.

**Exercise 7.4 — Prevalence adjustment (calculation, 35 minutes).**
Deliverable: compute PPV at 5, 8, 15, and 24 percent prevalence and build the table.
Acceptance criteria: all four values; a statement of the workload implication at 8 percent.

**Exercise 7.5 — Calibration critique (critique, 30 minutes).**
Deliverable: critique a report that gives AUROC 0.88 and applies a 0.5 threshold, with no
calibration plot.
Acceptance criteria: three problems; the plot and statistics you would request.

**Exercise 7.6 — Subgroup plan (design, 45 minutes).**
Deliverable: a pre-specified subgroup analysis plan with four subgroups, sample sizes, and a rule
for interpreting overlapping intervals.
Acceptance criteria: four subgroups; counts; interpretation rule.

**Exercise 7.7 — Severity review (design, 40 minutes).**
Deliverable: a three-level severity scale for extraction errors in a medication list, with examples
and a review procedure.
Acceptance criteria: three levels defined; two examples per level; blinding procedure.

**Exercise 7.8 — Results table (application, 50 minutes).**
Deliverable: a results table with columns for metric, unit, n, estimate, 95% interval, prevalence,
and condition, populated from Section 6.
Acceptance criteria: at least eight rows; every row has a unit and n; conditions stated.

**Exercise 7.9 — Evaluation funnel (application, 40 minutes).**
Deliverable: draw an evaluation funnel for a named task with at least four stages, and give a
hypothetical count at each stage. Identify which stage loses the most cases and what you would
investigate.
Acceptance criteria: four stages; counts; the largest loss identified with a diagnostic.

## 13. Self-check answers

**7.2 (match rule).** Gold has three spans; predictions match the first and third exactly. Exact:
TP 2, FP 1, FN 1; precision 2/3 = 0.667, recall 2/3 = 0.667, F1 0.667. The second predicted span
(30, 38) overlaps the gold (30, 41) but does not match exactly. Overlap: TP 3, FP 0, FN 0;
precision 1.0, recall 1.0, F1 1.0. The example is small enough that one boundary error moves the
score by 33 points, which is why the rule must be fixed before evaluation and stated in the report.

**7.3 (Wilson).** For 192/240: p = 0.8, n = 240, z = 1.96. Denominator 1 + 3.8416/240 = 1.0160.
Centre = (0.8 + 3.8416/480)/1.0160 = 0.8080/1.0160 = 0.7953. Half-width =
1.96 × sqrt(0.8 × 0.2/240 + 3.8416/(4 × 240²))/1.0160 = 1.96 × sqrt(0.0006667 + 0.0000167)/1.0160
= 1.96 × 0.02614/1.0160 = 0.0504. Interval 0.745 to 0.846. For 192/288: p = 0.667, denominator
1.0133, centre 0.6645, half-width 0.0541, interval 0.610 to 0.719. The intervals do not overlap with
the site B interval in Section 6, which supports the difference.

**7.4 (prevalence).** Using sensitivity 0.800 and specificity 0.874:
5%: 0.040/(0.040 + 0.1197) = 0.250.
8%: 0.064/(0.064 + 0.1159) = 0.356.
15%: 0.120/(0.120 + 0.1071) = 0.528.
24%: 0.192/(0.192 + 0.0958) = 0.667.
At 8 percent, about 180 patients per 1,000 are flagged and 64 are true positives; 116 reviews per
1,000 are false positives.

**7.1 (unit selection) rubric.** Full credit: readmission — patient or admission, sensitivity/PPV at
the decision unit; triage — message, sensitivity and false reassurance rate; quality measure —
eligible encounter, numerator and denominator agreement. Half credit: units without metrics.

**7.5 (calibration critique).** Problems: AUROC does not establish calibration; a 0.5 threshold may
not match the clinical cost ratio; without a calibration plot the predicted probabilities may be
over- or under-confident. Request: reliability diagram, calibration slope and intercept, and a
decision-curve analysis across plausible thresholds.

**7.6 (subgroup) rubric.** Full credit: four pre-specified subgroups with counts, intervals, and a
rule that a difference is claimed only when intervals do not overlap and the count is adequate.
Half credit: subgroups without the interpretation rule.

**7.7 (severity) rubric.** Full credit: three levels defined by clinical consequence, two examples
per level, blinded review, and an adjudication rule for disagreements. Half credit: scale without
examples.

**7.8 (results table) rubric.** Full credit: eight rows covering span, document, and patient levels;
unit and n per row; prevalence for PPV; intervals; conditions. Half credit: six rows or missing
units.

**7.9 (funnel) rubric.** Full credit: four stages with counts, a stated exclusion reason at each
stage, and a diagnostic for the largest loss. Half credit: stages without counts. Zero: a single
accuracy number.

## 14. Interview questions

**1. "What is the difference between span F1 and patient-level sensitivity?"**
Strong answer: unit and commitment differ; span measures text matching, patient measures cohort
completeness; gives a numeric example of the gap. Weak answer: "both measure accuracy."

**2. "Why is PPV prevalence-dependent?"**
Strong answer: the false-positive count grows with the negative population; gives the formula and an
example. Weak answer: "PPV is precision."

**3. "How do you choose a span match rule?"**
Strong answer: from the downstream use; exact for strict extraction, overlap for screening; fix
before evaluation and disclose. Weak answer: "exact match is standard."

**4. "What is a Wilson interval and why use it?"**
Strong answer: a binomial interval that behaves near 0 and 1; avoids out-of-range Wald bounds.
Weak answer: "a confidence interval."

**5. "What is calibration and why does it matter?"**
Strong answer: agreement between predicted probability and observed frequency; needed for
thresholds; AUROC does not imply it. Weak answer: "accuracy of probabilities."

**6. "What is net benefit?"**
Strong answer: TP share minus weighted FP share at a threshold; compares with treat-all and
treat-none; ties metric to decision. Weak answer: "a profit calculation."

**7. "How do you evaluate a subgroup difference?"**
Strong answer: pre-specify, report counts and intervals, avoid claiming when intervals overlap,
discuss mechanisms. Weak answer: runs a t-test on accuracy.

**8. "What is metric substitution and why is it a problem?"**
Strong answer: reporting a component metric as a system metric; inflates capability claims; gives
an example. Weak answer: "using the wrong metric."

**9. "How do you handle a reference standard with known disagreement?"**
Strong answer: report the band, evaluate against multiple standards, avoid claiming differences
smaller than the band. Weak answer: "use the adjudicated labels and move on."

**10. "Why are error counts insufficient?"**
Strong answer: severity differs; a missed historical mention and a missed contraindication are both
false negatives; review with a severity scale. Weak answer: "counts are objective."

**11. "How would you detect evaluation leakage?"**
Strong answer: audit the split at the patient level, check duplicate content, check temporal
ordering, compare with a temporal validation. Weak answer: "use a random split."

**12. "What would you put in a results table?"**
Strong answer: metric, unit, n, estimate, interval, prevalence, and conditions; one row per level.
Weak answer: a list of F1 scores.

## 15. Key terms

- **Abstention rate** — share of cases where the system does not answer.
- **AUROC** — area under the receiver operating characteristic curve.
- **Boundary error** — a span that differs at start or end.
- **Calibration** — agreement between predicted probability and observed frequency.
- **Conjunction fallacy** — reporting a component metric as a system metric.
- **Coverage** — share of cases the system handles.
- **Decision curve** — net benefit across thresholds.
- **Decision threshold** — the probability at which action is taken.
- **Denominator** — the unit count a metric divides by.
- **Discrimination** — ability to rank positives above negatives.
- **Error severity** — clinical harm potential of an error.
- **Evaluation level** — the object a metric is computed over.
- **Exact match** — identical start and end offsets.
- **External validation** — evaluation on a different site or era.
- **F1** — harmonic mean of precision and recall.
- **Net benefit** — true positive share minus weighted false positive share.
- **NPV** — negative predictive value.
- **Overlap match** — any character overlap between spans.
- **PPV** — positive predictive value.
- **Precision** — TP / (TP + FP).
- **Prevalence** — share of the unit that is positive.
- **Recall** — TP / (TP + FN).
- **Reference-standard band** — metric range across defensible standards.
- **Sensitivity** — TP / (TP + FN) at the decision unit.
- **Span match** — the rule for deciding whether spans are equal.
- **Specificity** — TN / (TN + FP).
- **Subgroup analysis** — metrics computed within a stratum.
- **Temporal validation** — evaluation on later data.
- **Wilson interval** — a binomial confidence interval.
- **Evaluation funnel** — staged counts showing where cases are lost.
- **Spectrum bias** — performance change due to case mix.
- **Missing-label sensitivity** — metric range when unlabelled items are counted both ways.
- **Prospective validation** — evaluation during live deployment.
- **Reference standard** — the labels used as truth for evaluation.
- **Conjunction error** — treating a component result as a system result.

## 16. Sources

All URLs accessed 25 October 2026.

1. Stubbs A, Filannino M, Soysal E, Henry S, Uzuner O. "Cohort selection for clinical trials:
   n2c2 2018 shared task track 1." J Am Med Inform Assoc. 2019;26(11):1163-1171.
   https://doi.org/10.1093/jamia/ocz163
2. Uzuner O, South BR, Shen S, DuVall SL. "2010 i2b2/VA challenge on concepts, assertions, and
   relations in clinical text." J Am Med Inform Assoc. 2011;18(5):552-556.
   https://doi.org/10.1136/amiajnl-2011-000203
3. Velupillai S, Suominen H, Liakata M, et al. "Using clinical Natural Language Processing for
   health outcomes research." J Biomed Inform. 2018;88:11-19.
   https://doi.org/10.1016/j.jbi.2018.10.005
4. Irvin J, Rajpurkar P, Ko M, et al. "CheXpert: A large chest radiograph dataset with uncertainty
   labels and expert comparison." AAAI 2019. https://doi.org/10.1609/aaai.v33i01.3301590
5. Stubbs A, Uzuner O. "Annotating longitudinal clinical narratives for de-identification: The 2014
   i2b2/UTHealth corpus." J Biomed Inform. 2015;58 Suppl:S20-S29.
   https://pmc.ncbi.nlm.nih.gov/articles/PMC4978170/
6. Collins GS, Moons KGM, Dhiman P, et al. "TRIPOD+AI statement: updated guidance for reporting
   clinical prediction models that use regression or machine learning methods." BMJ.
   2024;385:e078378. https://doi.org/10.1136/bmj-2023-078378
7. Bossuyt PM, Reitsma JB, Bruns DE, et al. "STARD 2015: an updated list of essential items for
   reporting diagnostic accuracy studies." BMJ. 2015;351:h5527.
   https://doi.org/10.1136/bmj.h5527
8. Vickers AJ, Elkin EB. "Decision curve analysis: a novel method for evaluating prediction models."
   Med Decis Making. 2006;26(6):565-574. https://doi.org/10.1177/0272989X06295361
9. Wilson EB. "Probable inference, the law of succession, and statistical inference." J Am Stat
   Assoc. 1927;22(158):209-212.
10. U.S. Food and Drug Administration. "Artificial Intelligence-Enabled Device Software Functions:
    Lifecycle Management and Marketing Submission Recommendations." Draft guidance, January 2025.
    https://www.fda.gov/media/184856/download
11. ONC. "HTI-1 Decision Support Interventions (DSI) Fact Sheet."
    https://healthit.gov/wp-content/uploads/2023/12/HTI-1_DSI_fact-sheet_508.pdf
12. Williams CYK, et al. "Physician- and large language model-generated hospital discharge
    summaries: a blinded, comparative quality and safety study." medRxiv. 2024.
    https://doi.org/10.1101/2024.09.29.24314562

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic: [`06_annotation.md`](06_annotation.md)
- Next topic: [`08_deidentification.md`](08_deidentification.md)
- Task units and aggregation: [`03_task-taxonomy.md`](03_task-taxonomy.md)
- Reference standards and agreement: [`06_annotation.md`](06_annotation.md)
- Grounded summarization evaluation: [`09_summarization-and-grounding.md`](09_summarization-and-grounding.md)
- Module survey: [`../../09_clinical_nlp.md`](../../course/09_clinical_nlp.md)
- Labs: [`../../labs/`](../../labs/)
