# Metrics

> **Module 06 — Machine Learning Foundations**
> Topic 07 of 13
> Estimated study time: 3.5 hours
> Prerequisites: Topics 02, 03, and 06
> Version 2026.3

## 1. Why this topic exists

A metric is a compressed statement about a model's behaviour. Compression always discards
something, and the choice of metric determines what is discarded. Accuracy discards the difference
between a missed event and a false alarm. AUROC discards the probability scale and the prevalence.
F1 discards the true negatives and imposes an implicit equal weighting of precision and recall. A
single number can be useful, but only if the team knows what it hides.

The failure mode is not that teams use metrics; it is that they use the wrong one for the decision
and then treat it as a property of the model. A readmission model with an AUROC of 0.78 may have a
positive predictive value of 0.34 at the planned threshold, which means two-thirds of the patients
enrolled in a programme would not have been readmitted. A sepsis model with an AUROC of 0.63 may
still be useful if the threshold is set to a reviewable alert volume. The metric that matters
depends on the action, the prevalence, the threshold, and the capacity.

Metric choice also has a social dimension. A model that reports accuracy of 97 percent on a 3
percent prevalence cohort appears excellent to a reader who does not know the base rate, and the
same reader may not notice that the model has zero sensitivity. A model that reports AUROC of 0.82
appears strong even when its precision is 0.10 and its alert volume is unmanageable. The reporting
conventions in this module exist to prevent that kind of misreading.

The evidence is clear that the standard trio of discrimination, calibration, and clinical utility
is the minimum. Steyerberg et al. (2010) proposed a framework with overall performance (Brier
score), discrimination (c-statistic), calibration (intercept, slope, plot), reclassification, and
clinical usefulness (net benefit). The framework has been widely adopted, and later work has
sharpened it: calibration and decision curves are now treated as primary evidence, not optional
extras. At the same time, some metrics have been shown to be misleading. The net reclassification
improvement (NRI) can be positive for an uninformative marker, even on independent test data
(Pepe et al., 2014), and its event and non-event components can move in opposite directions. F1
imposes a utility assumption that most clinical decisions do not make.

The metric set also shapes the conversation after deployment. If the monitoring plan tracks AUROC,
it will not detect a prevalence shift that halves the PPV, because AUROC is prevalence-independent.
If it tracks PPV and alert volume, it will. The metrics chosen at development become the metrics
that govern the model's life, which is another reason to choose them deliberately.

This topic teaches the metric set as a decision aid. You will learn to construct the two-by-two
table at a threshold, compute the prevalence-dependent and prevalence-independent metrics, report
discrimination and calibration together, compute net benefit, and avoid the metrics that mislead.
The output is a performance section that a clinician, a statistician, and a regulator can all read
without being misled.

There is also a governance dimension. A metric quoted in a regulatory submission becomes part of
the device's performance claim. If the claim is "sensitivity of at least 0.80 at the intended
threshold", the sponsor must show the threshold, the population, and the interval that supports it.
A claim that shifts from sensitivity to AUROC after the results are in is a change of claim, and it
requires a justification. The metric set should therefore be written with the same care as the
intended-use statement.

A further complication is that the same metric can be computed on different denominators. A
sensitivity computed among all patients is not the same as a sensitivity computed among patients
with complete data. A PPV computed at the validation prevalence is not the same as a PPV at the
deployment prevalence. The report should name the denominator, not only the metric.

The practical consequence is that metric selection is not a statistical afterthought. It is part of
the argument that the model is fit for its purpose. A team that cannot state why a metric was
chosen, or what it hides, is not ready to defend the model to a regulator, a clinical committee, or
a patient.

> **Note:** Every metric in this topic is reported with its prevalence, its threshold (if
> threshold-dependent), and a confidence interval. A metric without those three is a number, not
> evidence.

## 2. Learning outcomes

By the end of this topic you can:

1. **Construct** a two-by-two table at a specified threshold and compute sensitivity, specificity,
   PPV, NPV, accuracy, and likelihood ratios.
2. **Explain** why PPV and NPV depend on prevalence while sensitivity and specificity do not, and
   **compute** the change across prevalences.
3. **Compute** AUROC and AUPRC, and **explain** what each discards.
4. **Compute** the Brier score and calibration slope and intercept, and **interpret** them together.
5. **Compute** net benefit at a threshold and **compare** a model with treat-all and treat-none.
6. **Choose** a primary metric that matches the decision, and **justify** the choice before seeing
   results.
7. **Critique** the use of accuracy, F1, and NRI in a clinical model report.
8. **Report** a metric set with prevalence, threshold, and confidence intervals.

Outcomes 1, 3, 4, and 5 are computational; 6 and 7 are design and critique.

## 3. The core mental model

Metrics fall into four families, and each answers a different question. The two-by-two table at a
threshold is the root of the threshold-dependent family.

```text
                        Outcome positive        Outcome negative
Model positive               TP                      FP
Model negative               FN                      TN

Threshold-dependent:  sensitivity = TP/(TP+FN)      specificity = TN/(TN+FP)
                      PPV = TP/(TP+FP)              NPV = TN/(TN+FN)
                      accuracy = (TP+TN)/N           F1 = 2*PPV*sens/(PPV+sens)

Threshold-free:       AUROC = P(score_pos > score_neg)
                      AUPRC = area under precision-recall curve

Calibration:          observed vs predicted risk; slope, intercept, Brier score

Utility:              net benefit = sens*prev - (1-spec)*(1-prev)*t/(1-t)
```

The diagram is read from the table outward. The table depends on the threshold. The
threshold-free metrics integrate over all thresholds. The calibration metrics compare the predicted
probability with the observed frequency. The utility metric combines the table with the prevalence
and the harm ratio.

The prevalence is the hinge between the families. Sensitivity and specificity are properties of the
model at a threshold, conditional on the true status, so they do not depend on prevalence. PPV and
NPV are properties of the model and the population, because they condition on the model's output
rather than on the truth. AUROC is prevalence-independent; AUPRC is prevalence-dependent because
precision depends on the number of false positives relative to the number of positives. Net benefit
depends on prevalence directly.

A useful way to remember the relationships is Bayes' theorem:

```text
PPV = sensitivity * prevalence / (sensitivity * prevalence + (1 - specificity) * (1 - prevalence))
```

Every prevalence-dependent metric can be derived from the sensitivity, the specificity, and the
prevalence. That means a report can give sensitivity and specificity and let the reader compute PPV
for their own population. It also means a report that gives only PPV for one prevalence has not
described the model; it has described one deployment.

The families are also a communication device. A clinician can read the two-by-two table; a
statistician can read the AUROC and the calibration slope; a programme manager can read the net
benefit and the alert burden; a regulator can read the intended use and the intervals. The same
report serves four audiences because it reports four kinds of evidence, not because it repeats the
same number in four formats.

The four families answer four questions.

- **Discrimination:** can the model rank a random positive above a random negative? (AUROC, AUPRC)
- **Threshold performance:** at this cut-off, how many events are caught and how many alarms are
  false? (sensitivity, specificity, PPV, NPV)
- **Calibration:** when the model says 20 percent, does 20 percent of such patients have the event?
  (slope, intercept, plot)
- **Utility:** does using the model at this threshold do more good than harm? (net benefit)

A report that answers only one of the four is incomplete. A report that answers all four gives the
reader what they need to decide.

The four families are not independent, and that is a source of both insight and confusion. A model
with poor discrimination cannot have good calibration across the whole range, because it cannot
separate low-risk from high-risk patients. A model with good discrimination can still be
miscalibrated, because the ranking can be right while the scale is wrong. A model with good
discrimination and good calibration can still have poor utility, because the threshold may be in a
region where the harms outweigh the benefits. The families form a chain of necessary but not
sufficient conditions: utility requires calibration, calibration requires some discrimination, but
none of them guarantees the next.

The chain also explains why a single "best" metric does not exist. A metric that emphasises one link
in the chain necessarily underweights the others. The team's job is to report the links, not to
find a number that stands for all of them.

A useful diagnostic sequence follows the chain. Start with the two-by-two table and ask whether the
flagged fraction is feasible. If it is not, the decision is already impossible and the rest of the
metrics describe a decision that will not be made. If it is feasible, ask whether the PPV is
acceptable for the action. If it is not, the model may still be useful at a different threshold or
for a different action. If it is acceptable, ask whether the calibration supports the probability
statements that will be made to patients and clinicians. If it does not, recalibrate. Finally, ask
whether the net benefit is positive at the threshold and prevalence that will actually be used.

> **Note:** A metric is not a property of the model alone. It is a property of the model, the
> population, the threshold, and the outcome definition. Changing any of the four changes the
> number.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| True positive (TP) | A positive prediction with a positive outcome | Counted without the threshold | Count |
| False positive (FP) | A positive prediction with a negative outcome | Called a false alarm | Count |
| True negative (TN) | A negative prediction with a negative outcome | Ignored in F1 | Count |
| False negative (FN) | A negative prediction with a positive outcome | Called a miss | Count |
| Prevalence | Proportion of positives in the population | Confused with the flagged fraction | Proportion |
| Sensitivity | TP / (TP + FN) | Called recall | Proportion |
| Specificity | TN / (TN + FP) | Ignored at low prevalence | Proportion |
| PPV | TP / (TP + FP) | Assumed prevalence-independent | Proportion |
| NPV | TN / (TN + FN) | Assumed high without checking | Proportion |
| Accuracy | (TP + TN) / N | Used for rare events | Proportion |
| Balanced accuracy | Mean of sensitivity and specificity | Confused with accuracy | Proportion |
| F1 | Harmonic mean of precision and recall | Treated as a utility measure | 0 to 1 |
| Matthews correlation coefficient | Correlation between prediction and outcome | Assumed to solve imbalance | -1 to 1 |
| False positive rate | FP / (FP + TN) = 1 - specificity | Confused with false discovery rate | Proportion |
| False discovery rate | FP / (FP + TP) = 1 - PPV | Confused with FPR | Proportion |
| Positive likelihood ratio | Sensitivity / (1 - specificity) | Reported without prevalence | Ratio |
| Negative likelihood ratio | (1 - sensitivity) / specificity | Reported without prevalence | Ratio |
| Diagnostic odds ratio | LR+ / LR- | Treated as a clinical effect size | Ratio |
| Youden index | Sensitivity + specificity - 1 | Used to choose a threshold without harms | Proportion |
| AUROC | Probability a random positive ranks above a random negative | Reported without prevalence | 0.5 to 1.0 |
| AUPRC | Area under the precision-recall curve | Compared across prevalences | 0 to 1 |
| Average precision | A step-wise estimate of AUPRC | Confused with interpolated AUPRC | 0 to 1 |
| Precision | TP / (TP + FP) | Called PPV | Proportion |
| Recall | TP / (TP + FN) | Called sensitivity | Proportion |
| Brier score | Mean squared difference between predicted probability and outcome | Confused with AUROC | 0 to 1 |
| Log loss | Negative log-likelihood of the predicted probabilities | Confused with Brier | Positive |
| Calibration slope | Coefficient of the logit of predicted risk in a logistic recalibration | Confused with discrimination | Slope |
| Calibration intercept | Calibration-in-the-large | Confused with slope | Log-odds |
| Calibration plot | Observed risk versus predicted risk | Read without the distribution | Curve |
| Hosmer-Lemeshow test | Goodness-of-fit test for calibration | Treated as a calibration measure | p-value |
| Net benefit | Prevalence-weighted true positives minus threshold-weighted false positives | Confused with profit | Per patient |
| Decision curve | Net benefit across thresholds | Read as discrimination | Curve |
| NRI | Net reclassification improvement | Interpreted as a percentage | -2 to 2 |
| IDI | Integrated discrimination improvement | Treated as a proper score | Difference |
| Number needed to treat | Patients treated for one benefit | Confused with number needed to evaluate | Count |
| Number needed to evaluate | Patients reviewed per true case found | Confused with NNT | Count |
| Alert burden | Alerts per unit of clinical work | Measured only after launch | Alerts per 100 |
| Precision at k | Precision among the top k ranked patients | Confused with PPV at a threshold | Proportion |
| Recall at k | Recall among the top k ranked patients | Confused with sensitivity at a threshold | Proportion |
| Lift | Precision at k divided by prevalence | Confused with PPV | Ratio |
| Confidence interval | Uncertainty in a metric | Confused with standard deviation | Interval |
| Bootstrap interval | Resampling-based interval | Used without enough events | Interval |
| DeLong interval | Analytic interval for AUROC | Assumes independence | Interval |
| Wilson interval | Interval for a proportion | Replaced by normal approximation | Interval |
| Prevalence shift | Change in outcome prevalence | Confused with model drift | Proportion |
| Spectrum effect | Sensitivity varies by case severity | Ignored in subgroup analysis | Effect |
| Metric shopping | Choosing a metric after seeing results | Called exploratory analysis | Bias |
| Proper scoring rule | A metric that is optimised by the true probability | Confused with any metric | Property |
| Composite metric | A single number combining several components | Hides the components | Metric |
| Calibration-in-the-large | Mean predicted versus observed risk | Confused with the slope | Log-odds |
| Risk distribution | Distribution of predicted risks | Omitted from reports | Histogram |
| Decision threshold | Synonym for the operating threshold | Confused with the harm ratio | Probability |
| Test prevalence | Prevalence in the evaluation sample | Assumed to equal deployment | Proportion |
| Calibration curve | Synonym for calibration plot | Read without the distribution | Curve |
| Discrimination slope | Mean predicted risk difference between outcome groups | Confused with AUROC | Proportion |
| Explanatory power | Variance explained by the model | Confused with discrimination | Proportion |
| Overall performance | A proper score such as Brier | Confused with accuracy | Score |
| Reclassification table | Cross-tabulation of risk categories before and after a marker | Confused with NRI | Table |
| Delta AUROC | Change in AUROC from adding a marker | Confused with clinical value | Difference |
| Delta net benefit | Change in net benefit from adding a marker | Preferred over NRI | Difference |

Three definitions deserve more than a table row.

**Proper scoring rules.** A scoring rule is proper if it is minimised in expectation by the true
probability. The Brier score and log loss are proper; accuracy, F1, and AUROC are not. A model that
optimises a proper score is incentivised to report the truth, not a convenient threshold. That is
why calibration assessment relies on proper scores and why a model can have a good AUROC and a poor
Brier score at the same time.

**Threshold-free versus threshold-dependent.** AUROC and AUPRC summarise performance across all
thresholds. They are useful for comparing models before a threshold is chosen, but they do not tell
the team what will happen at the operating point. The two-by-two table at the threshold does. A
report needs both: the curve for the comparison, the table for the decision.

**Prevalence independence.** AUROC is prevalence-independent in the sense that it is unchanged when
the ratio of positives to negatives changes, as long as the conditional score distributions are
unchanged. That is a mathematical property, not a clinical virtue. It means AUROC can be compared
across cohorts with different prevalence, but it also means AUROC cannot tell you whether the model
will be useful in a low-prevalence cohort. AUPRC and PPV answer that question, at the cost of being
prevalence-dependent.

> **Note:** The choice between AUROC and AUPRC is not a preference. If the decision is a ranking
> with a fixed capacity, AUROC is a reasonable summary. If the decision is a threshold with a
> precision requirement, AUPRC and PPV at the threshold are the relevant metrics.

## 5. The mechanics, step by step

The metric procedure has fourteen steps. Steps 1 through 6 build the threshold-dependent table.
Steps 7 through 10 add the threshold-free, calibration, and utility evidence. Steps 11 through 14
make the result transportable and auditable.

### Step 1. Fix the decision and the threshold

**Input:** the prediction task specification.

**Output:** the action, the operating threshold, and the capacity.

**Rationale:** Almost every metric in this topic depends on the threshold or on the decision. A
sensitivity reported without a threshold is undefined; a PPV reported without a prevalence is
uninterpretable. Fixing the threshold before computing the table prevents the common practice of
searching thresholds for the most flattering number.

The threshold also determines which error is called a false positive. At a low threshold, many
patients are flagged and the false positive count is large; at a high threshold, the false negative
count is large. The same model can look excellent at one threshold and poor at another. That is why
a threshold-free metric alone cannot answer the decision question, and why a threshold-dependent
metric without the threshold is undefined.

**Branch point:** if the threshold is not yet chosen, report the full curve and a table at two or
three candidate thresholds, and state that the operating point is provisional. Do not report a
single threshold as if it were fixed.

### Step 2. Build the two-by-two table

**Input:** the held-out predictions, the outcomes, and the threshold.

**Output:** TP, FP, TN, FN, and the total.

**Rationale:** The table is the root of every threshold-dependent metric. It also makes the
arithmetic auditable: a reader can recompute every metric from the four counts. Reporting metrics
without the counts makes it impossible to check whether the denominator was the right one.

**Branch point:** if the model outputs a continuous score, define the positive rule explicitly
(`score >= threshold` or `score > threshold`). The choice changes the counts when scores are tied.
Report the tie count at the threshold.

### Step 3. Report prevalence and flagged fraction

**Input:** the table.

**Output:** prevalence `(TP + FN) / N` and flagged fraction `(TP + FP) / N`.

**Rationale:** Prevalence is the context for every other metric. The flagged fraction is the
operational burden. Reporting both makes the base-rate dependence visible and connects the table to
the workflow. A model that flags 35 percent of patients to catch 62 percent of events is a different
operational proposition from one that flags 10 percent.

The flagged fraction is also the denominator for the number needed to evaluate. At a flagged
fraction of 0.345 and a PPV of 0.342, the number needed to evaluate is `1 / 0.342 = 2.9`. That is a
manageable workload. At a prevalence of 5 percent and the same sensitivity and specificity, the
flagged fraction is `0.031 + 0.266 = 0.297`, the PPV is `0.031 / 0.297 = 0.104`, and the number
needed to evaluate is 9.6. The workload triples even though the flagged fraction falls.

**Branch point:** if the flagged fraction exceeds the programme's capacity, report the capacity gap
before reporting the metrics. The operating point is not feasible, and the metrics describe a
decision the programme cannot make. The operating point is not feasible, and the metrics describe a
decision the programme cannot make.

### Step 4. Compute sensitivity and specificity

**Input:** the table.

**Output:** sensitivity `TP / (TP + FN)` and specificity `TN / (TN + FP)`, with confidence intervals.

**Rationale:** These are the two conditional probabilities that do not depend on prevalence. They
are the most transportable summary of the model's threshold behaviour, and they are the inputs to
Bayes' theorem for any deployment prevalence. Reporting them with intervals makes the uncertainty
visible.

**Branch point:** if either denominator is small (fewer than about 20), the interval will be wide.
Report the interval and avoid a point-estimate claim.

### Step 5. Compute PPV and NPV, and show the prevalence dependence

**Input:** sensitivity, specificity, and the deployment prevalence.

**Output:** PPV and NPV at the validation prevalence and at the deployment prevalence, with the
arithmetic.

**Rationale:** PPV and NPV are what the clinician experiences. A PPV of 0.34 means two-thirds of
flagged patients do not have the event. The same model at a lower prevalence will have a lower PPV,
often dramatically so. Showing both prevalences prevents the reader from transporting the
validation PPV to a different population.

The Bayes calculation is worth doing by hand once. At 19 percent prevalence, a sensitivity of 0.62
and a specificity of 0.72 give a PPV of 0.342. At 5 percent prevalence, the same sensitivity and
specificity give `0.031 / (0.031 + 0.266) = 0.104`. The model has not changed; the prior has. A
clinician who sees a PPV of 0.342 in a paper and assumes it applies to their 5 percent clinic will
overestimate the yield by a factor of three.

**Branch point:** if the deployment prevalence is unknown, report a table of PPV across a plausible
range (for example, 5, 10, 20, and 30 percent). A single PPV without a prevalence is not a metric.

### Step 6. Compute likelihood ratios

**Input:** sensitivity and specificity.

**Output:** positive likelihood ratio `sens / (1 - spec)` and negative likelihood ratio
`(1 - sens) / spec`.

**Rationale:** Likelihood ratios are prevalence-independent and can be combined with a local prior
to estimate post-test probability. They are useful when the deployment prevalence differs from the
validation prevalence and the team wants a transportable summary. They are less intuitive than PPV
for a clinical audience, so they are usually a secondary metric.

A likelihood ratio of 2.2 means a positive result roughly doubles the odds of the outcome. A
likelihood ratio of 0.53 means a negative result roughly halves the odds. Those statements are
prevalence-independent and can be combined with a local prior, which makes them useful for a
multi-site deployment. They are less useful for a single-site decision, where the PPV is more
directly interpretable.

**Branch point:** if the likelihood ratio is close to 1, the model does not change the probability
materially at that threshold. That is a useful negative result, not a failure of reporting.

### Step 7. Compute threshold-free discrimination

**Input:** the scores and outcomes.

**Output:** AUROC and AUPRC, with intervals and the AUPRC baseline.

**Rationale:** AUROC summarises ranking across all thresholds and is useful for model comparison.
AUPRC summarises precision-recall across all thresholds and is more informative when the prevalence
is low. Reporting both, with the prevalence, prevents the reader from assuming that a good ranking
implies good precision.

AUPRC should be computed with a stated method. Average precision, which sums precision at each
positive and divides by the number of positives, differs from trapezoidal interpolation of the
precision-recall curve, especially when the curve is steep. The choice matters when two models are
compared by a small margin. Report the method and use the same one for both models.

**Branch point:** if AUROC and AUPRC disagree about which model is better, the prevalence is doing
the work. Report both and explain which one matches the decision. Do not choose the one that
flatters the preferred model.

### Step 8. Compute calibration

**Input:** the predicted probabilities and outcomes.

**Output:** calibration slope, calibration intercept, Brier score, and a calibration plot.

**Rationale:** Discrimination is about ranking; calibration is about the probability scale. A model
can rank well and be miscalibrated, which matters whenever the probability drives a threshold or a
conversation with a patient. The slope and intercept locate the miscalibration; the plot shows its
shape.

The Brier score is a proper scoring rule that combines discrimination and calibration into one
number. Its decomposition into reliability, resolution, and uncertainty shows why two models with
the same Brier score can fail differently: one may be well calibrated but poorly discriminating,
the other well discriminating but miscalibrated. The Brier score alone does not tell which, so it
is reported with the calibration plot and AUROC, not instead of them.

**Branch point:** if the model is miscalibrated, recalibrate on a held-out set and re-report the
metrics. Do not report the uncalibrated probabilities as if they were the model's final output.

### Step 9. Compute net benefit

**Input:** sensitivity, specificity, prevalence, and the threshold.

**Output:** net benefit at the threshold, and a decision curve across thresholds.

**Rationale:** Net benefit translates the table into the decision's terms. It allows a comparison
with treat-all and treat-none, and it makes the harm ratio explicit. A model with a higher AUROC can
have a lower net benefit if its errors are concentrated at the operating point.

Net benefit is expressed per patient, so a value of 0.06 means 60 net true positives per 1,000
patients. Multiplying by the cohort size gives the net true positives per year, which is the
quantity a programme manager can compare with the cost of the intervention. The conversion from
per-patient to per-cohort is simple but often omitted, leaving a number that nobody can act on.

**Branch point:** if net benefit is negative at the planned threshold, the decision rule is harmful.
Report that result, even if the AUROC is good. The model may still be useful at a different
threshold or for a different action.

### Step 10. Compute confidence intervals

**Input:** the counts and the metric definitions.

**Output:** intervals for every primary metric.

**Rationale:** A point estimate without an interval invites overconfidence, especially in small
validation sets. Wilson intervals are appropriate for proportions; bootstrap intervals are
appropriate for AUROC, AUPRC, and net benefit; DeLong intervals are appropriate for AUROC under
independence assumptions. The interval method should be stated.

The interval method should be appropriate to the metric and the design. Proportions from a simple
random sample use Wilson intervals. AUROC from a single test set uses DeLong or bootstrap. AUPRC
uses bootstrap because its sampling distribution is not well approximated by a simple formula. Net
benefit uses bootstrap over patients. A clustered design (patients within sites) requires a
cluster bootstrap; treating rows as independent will understate the interval.

**Branch point:** if the interval is so wide that the decision is uncertain, report that. A wide
interval is a finding, not a formatting problem.

### Step 11. Report subgroup metrics

**Input:** the table and subgroup identifiers.

**Output:** prevalence, sensitivity, specificity, PPV, and event counts by subgroup.

**Rationale:** A model that performs well overall can fail in a subgroup. The subgroup table
separates a real disparity from a prevalence difference, because PPV depends on the subgroup's
prevalence as well as on the model's sensitivity and specificity.

**Branch point:** if a subgroup has fewer than about 20 events, report the count and avoid a
subgroup-specific claim.

### Step 12. Recompute at the deployment prevalence

**Input:** sensitivity, specificity, and the deployment prevalence.

**Output:** PPV, NPV, net benefit, and flagged fraction at the deployment prevalence.

**Rationale:** The validation prevalence is rarely the deployment prevalence. A model validated at
19 percent and deployed at 8 percent will have a much lower PPV and may have negative net benefit.
Recomputing is arithmetic, not revalidation, and it is the minimum due diligence for a deployment
claim.

**Branch point:** if the deployment prevalence is outside the range observed in development,
treat the transport as an assumption and recommend local validation.

### Step 13. Avoid misleading composite metrics

**Input:** the metric set.

**Output:** a statement of which metrics are reported and which are excluded, with reasons.

**Rationale:** Accuracy, F1, and NRI are common but misleading in different ways. Accuracy is
dominated by the majority class; F1 imposes an equal weighting of precision and recall; NRI can be
positive for an uninformative marker. Excluding them is a deliberate choice that should be stated.

The NRI deserves a separate warning. It was designed to quantify how many patients move into a
higher or lower risk category when a new marker is added. Pepe et al. (2014) showed that it can be
positive on average even when the new marker is uninformative, because the reclassification is
driven by overfitting rather than by new information. Kerr et al. (2014) showed that its event and
non-event components can move in opposite directions, that it is not a proportion (its maximum is
2), and that published variance formulas are unreliable. The recommended alternative is the change
in net benefit, which is a proper utility measure. If the NRI is reported, both components should
be reported separately, with a bootstrap interval, and the result should be interpreted alongside
the decision curve.

**Branch point:** if a stakeholder requires F1, report it alongside precision, recall, and the
prevalence, and explain the implicit weighting. Do not let it become the primary metric.

### Step 14. Freeze the metric set and document it

**Input:** the analysis.

**Output:** a performance section with the metric definitions, the threshold, the prevalence, the
intervals, and the software.

**Rationale:** A metric set chosen after seeing results is a form of analyst degrees of freedom. The
primary metric should be pre-specified; secondary metrics can be added but should be labelled. The
documentation should allow a reader to recompute every number from the counts.

The metric definitions should be written in the protocol, not only in the results. A reader should
be able to tell whether "sensitivity" was computed among all events or among events with complete
data, and whether "PPV" refers to the validation prevalence or the deployment prevalence. Those
details change the number, and they are easy to lose when the analysis is handed to a new analyst.

**Branch point:** if the metric set changes after the results, record the change and the reason. A
change made after seeing results is a new analysis, not a correction.

### Metric edge cases

**Zero cells.** A table with no false positives gives a specificity of 1.0 and an infinite positive
likelihood ratio. Report the interval and the one-sided bound; do not report infinity as a point
estimate.

**Perfect separation.** A model that separates the classes perfectly in validation is usually
leaking or overfitting. Check the split and the features before celebrating the metrics.

**Ties in scores.** Ties at the threshold change the counts depending on the comparison rule.
Report the tie count and use a consistent rule.

**Tiny subgroups.** A subgroup with 12 events has a PPV interval wide enough to include almost any
value. Report the count and the interval.

**Prevalence shift.** A model validated at 19 percent and deployed at 5 percent may have a PPV
below 0.10 and negative net benefit. Recompute before deployment. A prevalence shift is not a model
failure, but it is a decision failure if the model is deployed without recomputing the metrics.

**A threshold with no ties but no capacity.** A threshold can produce a feasible flagged count on
average and an infeasible count on a busy day. The capacity analysis should use the upper end of the
daily volume, not the annual mean.

**Empty folds.** A fold with no events has undefined sensitivity. Report the fold as not evaluable.

**A metric computed on the wrong denominator.** A sensitivity computed among flagged patients is a
PPV, not a sensitivity. Check each denominator against the definition. The confusion is common in
reports where the table is not shown.

**Multiple thresholds.** Reporting the best threshold after seeing the curve is a form of selection.
Pre-specify the threshold or report the whole curve.

## 6. Worked example

### The setting

A heart failure programme uses a readmission model at a threshold of 0.20. The validation cohort has
40,000 discharges and 7,600 events, a prevalence of 19.0 percent. At the threshold, the model has
sensitivity 0.62 and specificity 0.72. The programme's deployment population is expected to have a
12.0 percent prevalence because it serves a lower-risk group.

### The two-by-two table at 19 percent

```text
TP = 0.62 * 7,600 = 4,712
FN = 7,600 - 4,712 = 2,888
FP = 0.28 * 32,400 = 9,072
TN = 32,400 - 9,072 = 23,328
```

The flagged fraction is `(4,712 + 9,072) / 40,000 = 13,784 / 40,000 = 0.3446`, or 34.5 percent.

### The metrics at 19 percent

```text
sensitivity = 4,712 / 7,600 = 0.620
specificity = 23,328 / 32,400 = 0.720
PPV = 4,712 / 13,784 = 0.3418
NPV = 23,328 / 26,216 = 0.8898
accuracy = (4,712 + 23,328) / 40,000 = 0.7010
LR+ = 0.62 / 0.28 = 2.214
LR- = 0.38 / 0.72 = 0.528
F1 = 2 * 0.3418 * 0.62 / (0.3418 + 0.62) = 0.4407
```

### The metrics at 12 percent

The sensitivity and specificity are assumed unchanged. The counts change because the population
changes:

```text
events = 0.12 * 40,000 = 4,800
non-events = 35,200
TP = 0.62 * 4,800 = 2,976
FN = 1,824
FP = 0.28 * 35,200 = 9,856
TN = 25,344
PPV = 2,976 / (2,976 + 9,856) = 2,976 / 12,832 = 0.2319
NPV = 25,344 / (25,344 + 1,824) = 25,344 / 27,168 = 0.9329
accuracy = (2,976 + 25,344) / 40,000 = 0.7080
```

The PPV falls from 0.342 to 0.232, a 32 percent relative decline. The accuracy rises slightly, from
0.701 to 0.708, because the majority class is larger and easier to predict. A report that gives
only accuracy would show an improvement; a report that gives only PPV would show a deterioration.
Both are true, and neither is sufficient.

The AUPRC at 19 percent prevalence is 0.34, against a baseline of 0.19. At 12 percent prevalence,
with the same sensitivity and specificity, the AUPRC falls to about 0.22, against a baseline of
0.12. The ratio of AUPRC to baseline falls from 1.8 to 1.8, which is a coincidence of the assumed
distributions; in general, the ratio is not constant. The absolute value is not comparable across
prevalences, which is why AUPRC is always reported with its baseline.

### The calibration

The model's calibration slope is 0.78 and its intercept is 0.09. A slope below 1 means the
predictions are overdispersed: low risks are too low and high risks are too high. An intercept above
0 means the model underpredicts on average. At a predicted risk of 0.20, the recalibrated risk is:

```text
logit(0.20) = -1.3863
recalibrated logit = 0.09 + 0.78 * (-1.3863) = -0.9913
recalibrated risk = 1 / (1 + exp(0.9913)) = 0.2706
```

A patient told they are at 20 percent risk is, on average, at about 27 percent risk. The threshold
of 0.20 on the nominal scale corresponds to a true risk of about 0.27 for the patients near the
cut-off. The calibration error shifts the operating point.

The calibration plot also shows the distribution of predicted risks. A model that predicts between
0.10 and 0.30 for almost everyone will have a calibration curve that is hard to estimate at the
extremes. The distribution belongs alongside the curve, because it shows where the model is
actually operating.

### The utility

Net benefit at the threshold of 0.20, at 19 percent prevalence:

```text
0.62 * 0.19 - 0.28 * 0.81 * (0.20 / 0.80)
= 0.1178 - 0.2268 * 0.25
= 0.1178 - 0.0567
= 0.0611
```

At 12 percent prevalence:

```text
0.62 * 0.12 - 0.28 * 0.88 * 0.25
= 0.0744 - 0.2464 * 0.25
= 0.0744 - 0.0616
= 0.0128
```

Net benefit falls from 0.0611 to 0.0128 per patient, an 79 percent decline. At 5 percent prevalence:

```text
0.62 * 0.05 - 0.28 * 0.95 * 0.25
= 0.0310 - 0.2660 * 0.25
= 0.0310 - 0.0665
= -0.0355
```

Net benefit is negative. At 5 percent prevalence, the model at this threshold harms more than it
helps, despite an AUROC of 0.68 and a sensitivity of 0.62. The decision changes from "deploy" to
"do not deploy without a different threshold or a different intervention".

### Two diagnostics that would reveal a broken report

**Diagnostic 1: accuracy versus PPV across prevalences.** The accuracy is 0.701 at 19 percent and
0.708 at 12 percent, while the PPV falls from 0.342 to 0.232. A report that shows only accuracy
would conclude that the model is stable; a report that shows only PPV would conclude that it
degrades. The diagnostic is a table with both, plus the prevalence, so the reader can see that the
apparent stability of accuracy is a majority-class effect.

**Diagnostic 2: the calibration slope.** A slope of 0.78 with an intercept of 0.09 means the
nominal threshold does not correspond to the stated risk. A report that gives AUROC and PPV without
calibration would not reveal that the patients near the cut-off are being told 20 percent when their
average risk is 27 percent. The diagnostic is a calibration plot with the identity line and the
slope and intercept annotated.

### Varying one assumption: the deployment prevalence

The worked example already varies prevalence from 19 to 12 to 5 percent. The conclusion changes at 5
percent, where net benefit becomes negative. If the programme can raise the threshold to 0.30 at 5
percent prevalence, with sensitivity 0.40 and specificity 0.90:

```text
PPV = 0.40 * 0.05 / (0.40 * 0.05 + 0.10 * 0.95)
= 0.020 / (0.020 + 0.095)
= 0.174

net benefit = 0.40 * 0.05 - 0.10 * 0.95 * (0.30 / 0.70)
= 0.020 - 0.095 * 0.4286
= 0.020 - 0.0407
= -0.0207
```

Even at the higher threshold, net benefit is negative at 5 percent prevalence under the assumed harm
ratio. The model is not useful for this decision at this prevalence, regardless of the threshold.
That is a finding about the decision, not about the model's discrimination.

### What to report

The primary table is the two-by-two at the operating threshold, with TP, FP, TN, FN, prevalence,
sensitivity, specificity, PPV, NPV, and net benefit. The primary figure is the calibration plot with
the slope and intercept. The secondary figure is the decision curve at the deployment prevalence,
with the operating point marked. AUROC and AUPRC belong in the comparison table with their
intervals and the AUPRC baseline. The report should state the prevalence for every metric and the
threshold for every threshold-dependent metric.

A reader should be able to take the two-by-two table and recompute every number in the report.
That is the simplest test of whether the performance section is complete. If a metric cannot be
recomputed from the table and the stated prevalence, either the denominator is missing or the
metric has been defined in a non-standard way. Both are reasons to revise the section before it
leaves the team.

## 7. Code, query, or calculation

The script computes the threshold-dependent metrics from counts, the threshold-free metrics from
labels and scores, and net benefit at three prevalences. It uses only the standard library.

```python
"""Threshold-dependent and threshold-free metrics. Standard library only."""


def confusion(tp, fp, tn, fn):
    n = tp + fp + tn + fn
    prevalence = (tp + fn) / n
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    ppv = tp / (tp + fp)
    npv = tn / (tn + fn)
    accuracy = (tp + tn) / n
    lr_pos = sensitivity / (1 - specificity)
    lr_neg = (1 - sensitivity) / specificity
    f1 = 2 * ppv * sensitivity / (ppv + sensitivity)
    return {"prevalence": prevalence, "sensitivity": sensitivity,
            "specificity": specificity, "ppv": ppv, "npv": npv,
            "accuracy": accuracy, "lr_pos": lr_pos, "lr_neg": lr_neg, "f1": f1}


def auroc(labels, scores):
    positives = [s for s, y in zip(scores, labels) if y == 1]
    negatives = [s for s, y in zip(scores, labels) if y == 0]
    wins = sum(1.0 if p > n else (0.5 if p == n else 0.0)
               for p in positives for n in negatives)
    return wins / (len(positives) * len(negatives))


def average_precision(labels, scores):
    ordered = sorted(zip(scores, labels), reverse=True)
    tp = fp = 0
    total = 0.0
    for _, label in ordered:
        if label == 1:
            tp += 1
            total += tp / (tp + fp)
        else:
            fp += 1
    return total / sum(labels)


def net_benefit(sensitivity, specificity, prevalence, threshold):
    return (sensitivity * prevalence
            - (1 - specificity) * (1 - prevalence) * threshold / (1 - threshold))


at_19 = confusion(tp=4712, fp=9072, tn=23328, fn=2888)
at_12 = confusion(tp=2976, fp=9856, tn=25344, fn=1824)
for name, m in (("19%", at_19), ("12%", at_12)):
    print(f"{name}: prev={m['prevalence']:.2f} sens={m['sensitivity']:.3f} "
          f"spec={m['specificity']:.3f} ppv={m['ppv']:.4f} npv={m['npv']:.4f} "
          f"acc={m['accuracy']:.3f} f1={m['f1']:.4f}")

print(f"LR+={at_19['lr_pos']:.3f} LR-={at_19['lr_neg']:.3f}")

labels = [1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0]
scores = [0.9, 0.7, 0.6, 0.8, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.2, 0.1]
print(f"AUROC={auroc(labels, scores):.4f} AUPRC={average_precision(labels, scores):.4f}")

print(f"NB 19%={net_benefit(0.62, 0.72, 0.19, 0.20):.4f} "
      f"NB 12%={net_benefit(0.62, 0.72, 0.12, 0.20):.4f} "
      f"NB 5%={net_benefit(0.62, 0.72, 0.05, 0.20):.4f}")
```

Expected output:

```text
19%: prev=0.19 sens=0.620 spec=0.720 ppv=0.3418 npv=0.8898 acc=0.701 f1=0.4407
12%: prev=0.12 sens=0.620 spec=0.720 ppv=0.2319 npv=0.9329 acc=0.708 f1=0.3376
LR+=2.214 LR-=0.528
AUROC=0.7812 AUPRC=0.7153
NB 19%=0.0611 NB 12%=0.0128 NB 5%=-0.0355
```

Reading the output:

- Sensitivity and specificity are identical across the two prevalences, by construction. PPV and
  NPV change because the population changes. Accuracy rises slightly while PPV falls, which is the
  majority-class effect.
- F1 falls from 0.441 to 0.338 even though the model's conditional performance is unchanged,
  because F1 depends on PPV and recall and PPV depends on prevalence.
- The likelihood ratios are unchanged, which is the property that makes them transportable.
- AUROC is 0.781 on the small example, and AUPRC is 0.715 at a prevalence of 4/12 = 0.333. The
  AUPRC baseline is 0.333, so the model is well above chance on this toy set.
- Net benefit falls from 0.0611 to 0.0128 to -0.0355 as prevalence falls from 19 to 12 to 5
  percent. The sign change is the decision-relevant result.

The functions are deliberately simple. `auroc` is O(n_pos * n_neg), which is fine for a teaching
example but not for a large cohort; in practice use a rank-based or library implementation. The
point of the code is the arithmetic, not the efficiency.

## 8. Data, standards, and regulatory context

Metrics are addressed by reporting guidelines and appraisal tools, not by a binding statistical
standard. The table distinguishes what is required from what is conventional.

| Instrument | Status | What it says about metrics | What it does not do |
|---|---|---|---|
| TRIPOD+AI | Reporting convention | Requires discrimination, calibration, and performance at thresholds to be reported | Does not prescribe a metric set |
| PROBAST+AI | Appraisal tool | Assesses whether the performance evaluation was appropriate, including calibration | Does not compute metrics |
| FDA GMLP principles | Guidance, not binding | Calls for testing on representative data and reporting of performance | Does not specify metrics |
| FDA PCCP guidance | Guidance, not binding | Requires performance criteria for planned changes | Does not define the criteria |
| STARD 2015 | Reporting convention | Requires sensitivity, specificity, and intervals for diagnostic accuracy | Does not cover prediction model utility |
| CONSORT-AI | Reporting convention | Requires the AI intervention's performance to be reported in trials | Does not cover development studies |
| TRIPOD+AI for Abstracts | Reporting convention | Requires key performance numbers in the abstract | Does not prescribe which |
| NIST AI RMF | Voluntary framework | Measure and manage performance, including subgroup performance | Not sector-specific |
| EU AI Act | Binding in the EU, phased | Requires technical documentation and testing for high-risk systems | Does not specify metrics |

Three points matter for practice.

**The reporting standard expects more than AUROC.** TRIPOD+AI requires the model's performance to
be reported in a way that reflects the intended use, including calibration and, where relevant,
threshold-based performance. A report that gives only AUROC has not met the item in substance,
even if it has filled the field. The same is true of a report that gives calibration without
discrimination or utility.

**STARD's accuracy metrics are not enough for a prediction model.** STARD was designed for
diagnostic accuracy studies, where sensitivity and specificity are the primary quantities. A
prediction model that informs a threshold decision also needs calibration and net benefit. Using
STARD's items without adding the prediction-specific ones produces a report that is accurate but
not decision-relevant.

**No regulator prescribes a metric.** The FDA and the EU expect the sponsor to justify the
performance measures against the intended use. A sponsor that reports AUROC only, for a device
whose output drives a threshold, will be asked why calibration and utility were not assessed. The
justification is part of the submission, not an optional discussion.

### A reporting checklist for a performance section

A performance section that satisfies the guidelines contains the following. The list is a
convention, but it maps onto TRIPOD+AI's performance items and PROBAST+AI's analysis domain.

```text
population and prevalence, with exclusions
validation design and date range
threshold and how it was chosen
two-by-two table at the threshold
sensitivity, specificity, PPV, NPV with intervals
AUROC and AUPRC with intervals and the AUPRC baseline
calibration slope, intercept, plot, and Brier score
net benefit at the threshold and a decision curve
subgroup performance with counts
alert burden and capacity
software and version
```

A report that omits the two-by-two table but gives the derived metrics is hard to audit. A report
that gives the table but not the prevalence is hard to interpret. The checklist is designed so that
each item supports the next.

**The metric set and the monitoring plan.** The metrics chosen for development should be the
metrics monitored after deployment, with the same definitions and the same prevalence context. A
monitoring plan that tracks a different metric from the one in the performance claim cannot detect
whether the claim still holds. The continuity between development and monitoring is part of the
metric choice, not an operational detail. A metric that cannot be monitored is not a useful primary
metric, even if it is statistically ideal at development time. The monitoring plan and the performance
section should use the same definitions.

**The metric set and the model card.** A model card summarises the performance section for a
non-specialist. It should carry the prevalence, the threshold, the PPV, the sensitivity, the
calibration slope, and the alert burden, in plain language. It should not carry a single composite
number without the components, because the reader cannot interpret it without the context the card
is supposed to provide.

**The metric set and the intended use statement.** A device's intended use names the population,
the user, the decision, and the output. The metric set should map onto each element. Population:
prevalence and subgroup performance. User: interpretability and alert burden. Decision: threshold,
PPV, and net benefit. Output: calibration. If a performance section does not address one of the
four elements, the intended-use statement and the evidence are misaligned.

**A note on subgroup reporting.** A subgroup table that reports only AUROC does not answer the
fairness question. The relevant quantities are the subgroup prevalence, sensitivity, PPV, and net
benefit at the same threshold. A subgroup with the same AUROC but a lower prevalence will have a
lower PPV and a higher number needed to evaluate, which is a different operational burden even if
the ranking is equally good.

> **Note:** The choice of metric is a scientific judgement, but the requirement to justify it is a
> regulatory and reporting expectation. Write the justification before the results.

## 9. Industry practice and case evidence

**The performance framework.** Steyerberg et al. (2010) proposed a framework with overall
performance (Brier score), discrimination (c-statistic), calibration (intercept, slope, plot),
reclassification, and clinical usefulness (net benefit). The framework is the basis for most
current guidance and for the structure of this topic. Its key claim is that no single measure is
sufficient, because each answers a different question.

**AUROC versus precision-recall.** Saito and Rehmsmeier (2015) show that the ROC curve is unchanged
when the class ratio changes, while the precision-recall curve changes and its baseline equals the
prevalence. In their example, a classifier with an ROC AUC of 0.886 had a precision-recall AUC of
0.054, close to the baseline, exposing near-random precision that the ROC curve concealed.

**NRI critiques.** Pepe et al. (2014) showed that the NRI can be positive on average for an
uninformative marker, even on independent test data, because reclassification is driven by
overfitting. Kerr et al. (2014) showed that the NRI is not a proportion, that its components can
move in opposite directions, and that its variance formulas are unreliable. Leening et al. (2014)
reviewed 67 publications and found incomplete reporting, incorrect calculation, and
misinterpretation. The recommended alternative is the change in net benefit.

**Sepsis alert burden.** The external validation of the Epic Sepsis Model reported an AUROC of 0.63,
a sensitivity of 33 percent, a PPV of 12 percent, and 18 percent of hospitalisations flagged at the
vendor's threshold. The case shows why AUROC alone is insufficient: the model's discrimination was
modest, but the operational failure was the combination of low sensitivity, low PPV, and high alert
volume. A report with only AUROC would not have revealed the alert burden.

**Readmission discrimination.** Kansagara et al. (2011) found c-statistics of 0.55 to 0.65 for
administrative readmission models in large US populations, and 0.68 to 0.83 for models usable at
discharge. The range illustrates that a c-statistic is not a universal quality label; the same
value can be acceptable for one decision and inadequate for another.

**The accuracy paradox in practice.** A model that predicts "no readmission" for every patient in a
19 percent prevalence cohort has an accuracy of 81 percent and a sensitivity of zero. The number is
arithmetically correct and clinically useless. Reports that lead with accuracy are common in
vendor materials because the number looks strong to a reader who does not compute the base rate.

**Why the metric set matters commercially.** A vendor that reports AUROC in a sales deck and PPV in
a contract is making two different claims. The buyer who signs the contract on the PPV and later
discovers that it referred to a 50 percent prevalence has a legitimate grievance, even if the
number was technically correct for the validation sample. The metric set should be part of the
commercial documentation, with the prevalence stated in the same sentence as the number.

**What a good performance section looks like.** It starts with the population and prevalence, states
the threshold and how it was chosen, gives the two-by-two table, reports discrimination and
calibration with intervals, reports net benefit and the decision curve, and ends with the subgroup
and capacity analysis. It is not a list of every metric the software can compute. It is a structured
argument that the model is fit for a specific decision.

**A note on the Brier score.** Two models can have the same Brier score with different calibration
and discrimination. The score is a sum of squared errors, so a model that is well calibrated but
poorly discriminating and a model that discriminates well but is miscalibrated can land at the same
value. The Brier score is useful as a proper overall measure, but it does not replace the
calibration plot or the AUROC.

**A note on the reporting burden.** More metrics are not always better. A performance section with
thirty metrics is harder to audit than one with eight well-chosen ones, and the extra numbers invite
selective reading.

**A note on external comparisons.** A c-statistic of 0.70 in one cohort and 0.75 in another does
not establish that the second model is better, because the cohorts differ in case mix, outcome
definition, and prevalence. A comparison across studies is only interpretable if the populations and
definitions are comparable. Within a study, the same split and the same outcome definition make the
comparison meaningful.

> **Note:** The cases are documented. The claim that a specific reported metric was chosen to
> flatter a model is not made here; the cases show the mechanisms, not the intentions.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Reporting AUROC alone | No calibration or threshold metrics | Ranking treated as the goal | Hidden miscalibration and alert burden | Report the four families |
| Reporting accuracy | Accuracy above 90 percent at low prevalence | Majority-class dominance | Useless model looks good | Report sensitivity and PPV |
| PPV without prevalence | PPV quoted as a model property | Base-rate neglect | Wrong deployment expectation | Report prevalence and PPV table |
| Threshold not stated | Sensitivity without a cut-off | Undefined operating point | Unreproducible metrics | State the threshold |
| F1 as primary metric | F1 without components | Implicit equal weighting | Wrong utility assumption | Report components and net benefit |
| NRI as evidence of value | NRI positive for a weak marker | Overfitting-driven reclassification | False improvement | Use change in net benefit |
| Calibration ignored | Good AUROC, unknown slope | Discrimination confused with calibration | Wrong probabilities | Calibration plot and slope |
| No confidence intervals | Single point estimates | Sampling uncertainty hidden | Overconfidence | Wilson or bootstrap intervals |
| Subgroup metrics omitted | Only overall performance | Heterogeneity hidden | Harm in a subgroup | Subgroup table with counts |
| Prevalence assumed constant | Validation prevalence only | Base rate changes | PPV falls at deployment | Recompute at deployment prevalence |
| Multiple thresholds reported | Best threshold highlighted | Selection after results | Optimistic operating point | Pre-specify the threshold |
| Metric chosen after results | Primary metric appears in results only | Analyst degrees of freedom | Inflated performance | Pre-register the metric |
| Wrong denominator | Sensitivity computed among flagged | Definition error | Misleading metric | Show the two-by-two table |
| Ties ignored | Flagged count changes with comparison rule | Coarse scores | Unstable operating point | Report tie count and rule |
| Metric computed on the training set | Numbers higher than validation | Apparent performance | Optimistic report | Use held-out predictions |
| Calibration assessed without a plot | Slope reported alone | Nonlinear miscalibration hidden | Wrong correction | Plot by decile |
| AUPRC compared across prevalences | Model A 0.30, model B 0.25 in different cohorts | Baseline differs | Wrong ranking | Compare within the same cohort |
| Alert burden omitted | No alerts per period | Workflow ignored | Unusable deployment | Compute alerts and capacity |
| Composite score without components | One number only | Components hidden | Misleading decision | Report components |
| Calibration plot omitted | Only slope reported | Nonlinear error hidden | Wrong correction | Plot by decile |
| Risk distribution omitted | No histogram | Operating range unknown | Threshold poorly chosen | Report the distribution |
| Metric computed on a subset | Denominator not stated | Selection hidden | Non-comparable | State the denominator |
| Calibration assessed only at the mean | Intercept reported without slope | Overdispersion hidden | Wrong threshold | Report slope and plot |
| Discrimination reported without utility | AUROC only | Decision not analysed | Harmful deployment | Add net benefit |
| Subgroup counts omitted | Only subgroup AUROC | Small-sample noise | False disparity | Report counts |
| Interval method unstated | No method described | Uncertainty unknown | Overconfidence | Name the method |

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The validation sample represents deployment | Metrics transport | Case-mix or site differences | Compare characteristics |
| The threshold is the deployment threshold | Threshold metrics apply | Different operating point | State and verify the threshold |
| The prevalence is known | PPV and net benefit depend on it | Prevalence shift | Monitor and recompute |
| The outcome definition is stable | Metrics compare like with like | Label drift | Label validation |
| The predictions are independent | Intervals are valid | Clustered data | Cluster bootstrap |
| The model is calibrated for probability claims | Probability scale is meaningful | Miscalibration | Calibration plot |
| The harm ratio is known | Net benefit is interpretable | Unknown utilities | Sensitivity analysis |
| Subgroup sizes are adequate | Subgroup metrics are stable | Fewer than 20 events | Report counts |
| The metric is pre-specified | Selection bias is controlled | Post-hoc choice | Protocol audit |
| The comparison uses the same split | Differences reflect models | Different validation sets | Reuse the split |
| The metric is appropriate to the decision | Interpretation is valid | AUROC for a precision requirement | Map metric to decision |
| The threshold is implementable | Operating point is real | Score not available at the decision time | Workflow test |
| The alert volume is manageable | Programme can respond | Capacity below flagged count | Capacity analysis |
| The subgroup prevalence is reported | PPV differences are interpretable | Prevalence omitted | Subgroup table |
| The calibration is checked at the threshold | The operating point is meaningful | Calibration assessed only at the mean | Plot near the threshold |
| The net benefit uses the deployment prevalence | Decision analysis is valid | Validation prevalence used | Recompute |
| The metric set is pre-specified | Selection bias is controlled | Post-hoc choice | Protocol audit |
| The alert volume is computed on the deployment population | Workload estimate is valid | Validation volume used | Recompute |
| The decision threshold is implementable in the workflow | The metric describes a real decision | Score not shown at the decision time | Workflow test |

The deepest limit is that a metric summarises a distribution of patients and decisions. Two models
with the same AUROC can have different net benefit, different subgroup behaviour, and different
calibration. The metric set is a way to make the comparison honest, not a way to reduce it to one
number. A report that gives the set lets the reader make their own judgement; a report that gives
one number asks the reader to accept the author's judgement without seeing the evidence.

## 12. Exercises

Answers to the calculation items are in Section 13; the critique and writing items are assessed by
rubric. A learner who completes all eight has a draft performance section for the module assignment.
The sequence is: the table, the prevalence table, the threshold-free metrics, calibration, utility,
then critique and writing. Each states a
deliverable, an acceptance criterion, and a time budget.

**Exercise 1. Two-by-two arithmetic.** At 8 percent prevalence, a model has sensitivity 0.75 and
specificity 0.85. Compute PPV, NPV, accuracy, LR+, LR-, and the flagged fraction. Deliverable: the
arithmetic. Acceptance criteria: all values correct to three decimals; interpretation states the
alert burden. Time: 30 minutes.

**Exercise 2. Prevalence table.** For the model in Exercise 1, compute PPV at prevalences of 2, 8,
and 20 percent. Deliverable: a table and a sentence on the direction of change. Acceptance criteria:
values correct; explanation names Bayes' theorem. Time: 30 minutes.

**Exercise 3. AUROC and AUPRC.** For a small labelled score list, compute AUROC and average
precision by hand. Deliverable: the working. Acceptance criteria: both values correct; explanation
of why AUPRC depends on prevalence. Time: 45 minutes.

**Exercise 4. Calibration interpretation.** A model has slope 0.75 and intercept 0.12. Compute the
recalibrated risk at a predicted 0.30. Deliverable: the arithmetic and a sentence on the direction
of miscalibration. Acceptance criteria: value correct; direction stated. Time: 30 minutes.

**Exercise 5. Net benefit.** At 15 percent prevalence, sensitivity 0.60, specificity 0.80, compute
net benefit at thresholds 0.10, 0.20, and 0.30. Deliverable: a table and a decision curve sketch.
Acceptance criteria: values correct; the threshold where net benefit becomes negative identified.
Time: 45 minutes.

**Exercise 6. Metric critique.** Find a clinical model paper and assess whether it reports
discrimination, calibration, and utility with prevalence and intervals. Deliverable: a 600-word
critique. Acceptance criteria: at least four specific gaps; claims cited. Time: 60 minutes.

**Exercise 7. NRI critique.** Read a paper that reports NRI and assess whether the components,
intervals, and calibration are reported, and whether the conclusion would survive a net benefit
analysis. Deliverable: a 500-word critique. Acceptance criteria: at least three of the known NRI
problems addressed. Time: 60 minutes.

**Exercise 8. Performance section.** Write the performance section of a model development report.
Before writing, decide which of the four families is primary for your decision and state why. Then
write the section so that a reader who disagrees with your choice can see the evidence for the other
three. Include the two-by-two table, the prevalence table, discrimination with
using the four families. Include the two-by-two table, the prevalence table, discrimination with
intervals, calibration with a plot, net benefit, subgroup counts, and alert burden. Deliverable: two
pages. Acceptance criteria: prevalence, threshold, and intervals for every metric; calibration and
net benefit included; no accuracy, F1, or NRI as primary evidence. Time: 75 minutes. Write the performance section of a model development report
using the four families. Deliverable: two pages. Acceptance criteria: prevalence, threshold, and
intervals for every metric; calibration and net benefit included. Time: 75 minutes.

## 13. Self-check answers

**Answer to Exercise 1.** Per 10,000: 800 events, 9,200 non-events. TP = 600, FN = 200, FP = 1,380,
TN = 7,820. PPV = 600 / 1,980 = 0.303. NPV = 7,820 / 8,020 = 0.975. Accuracy = 8,420 / 10,000 =
0.842. LR+ = 0.75 / 0.15 = 5.0. LR- = 0.25 / 0.85 = 0.294. Flagged fraction = 1,980 / 10,000 =
0.198. About one in five patients is flagged, and one in three flagged patients has the event.

**Answer to Exercise 2.** At 2 percent: TP = 150, FP = 1,470, PPV = 150 / 1,620 = 0.093. At 8
percent: 0.303. At 20 percent: TP = 1,500, FP = 1,200, PPV = 1,500 / 2,700 = 0.556. The PPV rises
with prevalence because the same false positive rate is applied to a larger positive class.

**Answer to Exercise 4.** `logit(0.30) = -0.8473`. Recalibrated logit = `0.12 + 0.75 * (-0.8473) =
-0.5155`. Recalibrated risk = `1 / (1 + exp(0.5155)) = 0.374`. The model under-predicts at this
risk: a patient told 30 percent is closer to 37 percent. The slope below 1 also means the
predictions are overdispersed across the range.

**Answer to Exercise 5.** At 15 percent prevalence: NB(0.10) = `0.60 * 0.15 - 0.20 * 0.85 * (0.10 /
0.90) = 0.090 - 0.0189 = 0.0711`. NB(0.20) = `0.090 - 0.170 * 0.25 = 0.090 - 0.0425 = 0.0475`.
NB(0.30) = `0.090 - 0.170 * 0.4286 = 0.090 - 0.0729 = 0.0171`. Net benefit remains positive across
these thresholds but declines as the threshold rises. It becomes negative when the threshold
exceeds the point where the false positive penalty outweighs the true positive gain.

**Answer to Exercise 6 (what a strong critique contains).** A strong critique names the metric set,
checks whether prevalence and threshold are stated, and asks whether calibration and net benefit are
reported. It distinguishes a reporting omission from a modelling error, and it identifies whether
the model's claimed use is supported by the evidence. It ends with two or three specific,
actionable corrections.

**Answer to Exercise 3 (method).** To compute AUROC by hand, list every positive-negative pair,
count 1 if the positive has the higher score, 0.5 for a tie, and divide by the number of pairs. For
the example, there are 4 positives and 8 negatives, or 32 pairs. To compute average precision, sort
the scores descending, walk down the list, and at each positive record the current precision
(`tp / (tp + fp)`), then average those precisions over the number of positives. The two metrics
answer different questions: AUROC asks how often a positive outranks a negative; average precision
asks how precise the model is at the recalls where it finds positives.

**Answer to Exercise 7 (what a strong NRI critique contains).** A strong critique reports the event
and non-event components separately, checks whether the reclassification categories were
pre-specified, checks the calibration of the new model, and asks whether the NRI's sign is
consistent with the change in net benefit. It notes that the NRI can be positive for an
uninformative marker and that its maximum is 2, so it is not a percentage. It recommends a decision
curve or a change in net benefit as the primary evidence.

**Answer to Exercise 8 (what a complete section contains).** A complete section states the
population and prevalence, the validation design, the threshold and its derivation, the two-by-two
table, sensitivity and specificity, PPV and NPV at both the validation and deployment prevalences,
AUROC and AUPRC with intervals and the AUPRC baseline, calibration slope and intercept with a plot,
Brier score, net benefit at the threshold and a decision curve, subgroup performance with counts,
alert burden, and the software version. It does not lead with accuracy or F1. If a stakeholder
requires those, they appear in a clearly labelled secondary table.

**Rubric for Exercises 6, 7, and 8.** A strong submission (1) states prevalence and threshold, (2)
reports discrimination with intervals, (3) reports calibration with a plot, (4) reports net benefit
or a decision curve, (5) checks subgroups, and (6) avoids accuracy, F1, and NRI as primary
evidence. A competent submission meets four of the six. A weak submission reports AUROC alone or
quotes PPV without prevalence.

## 14. Interview questions

**1. Why is AUROC not enough?** A strong answer explains that it ignores calibration, prevalence,
threshold, and utility, and gives an example where AUROC is stable while PPV halves. A weak answer
says AUROC is "just one metric" without a mechanism.

**2. What is the difference between discrimination and calibration?** A strong answer says
discrimination is ranking and calibration is the probability scale, and gives an example of a model
that ranks well but is miscalibrated. A weak answer uses the terms interchangeably.

**3. How does prevalence affect PPV?** A strong answer derives the Bayes relation and shows the
direction of change. A weak answer says PPV depends on prevalence without the arithmetic.

**4. What is net benefit?** A strong answer defines it, gives the formula, and explains the
comparison with treat-all and treat-none. A weak answer confuses it with profit or accuracy.

**5. Why is F1 problematic?** A strong answer explains the implicit equal weighting, the omission of
true negatives, and the lack of a decision interpretation. A weak answer says F1 is fine for
imbalanced data.

**6. Why is NRI controversial?** A strong answer cites the uninformative-marker result, the
component directions, and the variance problems, and recommends net benefit. A weak answer says NRI
measures reclassification without the caveats.

**7. How do you report uncertainty for AUROC?** A strong answer names DeLong or bootstrap and
mentions clustering. A weak answer says "report a confidence interval" without a method.

**8. When would you report accuracy?** A strong answer says almost never for a clinical prediction
model, and only alongside the full table if a stakeholder requires it. A weak answer says accuracy
is always useful.

**9. How do you choose a primary metric?** A strong answer maps the metric to the decision: net
benefit for a threshold decision, AUPRC for a precision requirement, AUROC for a ranking. A weak
answer picks the highest number.

**10. What is the difference between AUROC and AUPRC?** A strong answer explains that AUROC is prevalence-independent and AUPRC is not, and names the decision context for each. A weak answer says AUPRC is "better for imbalance" without a mechanism.

**11. Why might a model with good discrimination have poor utility?** A strong answer points to miscalibration, the threshold, and the harm ratio. A weak answer says utility and discrimination are the same thing.

**12. How do you compare two models when their AUROCs overlap?** A strong answer compares the difference with its interval, then uses calibration and net benefit to break the tie. A weak answer picks the higher point estimate.

**13. What is the number needed to evaluate and why does it matter?** A strong answer defines it as 1/PPV and connects it to workload. A weak answer confuses it with number needed to treat.

**14. What is a proper scoring rule?** A strong answer defines it as a metric minimised in expectation by the true probability, and names Brier and log loss as examples. A weak answer says any metric that is "properly calculated" is a proper scoring rule.

**15. How do you report metrics for a model that will be used at a different prevalence?** A strong answer reports sensitivity and specificity, a PPV table across prevalences, and net benefit at the deployment prevalence. A weak answer reports the validation PPV.

**16. What does a calibration slope of 0.7 mean?** A strong answer says the predictions are
overdispersed and gives the recalibration arithmetic. A weak answer says the model is "poorly
calibrated" without direction.

## 15. Key terms

- **Accuracy** — proportion of correct predictions.
- **Alert burden** — alerts per unit of clinical work.
- **AUPRC** — area under the precision-recall curve.
- **AUROC** — area under the receiver operating characteristic curve.
- **Average precision** — a step-wise estimate of AUPRC.
- **Balanced accuracy** — mean of sensitivity and specificity.
- **Brier score** — mean squared error of predicted probabilities.
- **Calibration intercept** — calibration-in-the-large.
- **Calibration plot** — observed versus predicted risk.
- **Calibration slope** — over- or under-dispersion of predictions.
- **Decision curve** — net benefit across thresholds.
- **DeLong interval** — analytic interval for AUROC.
- **Diagnostic odds ratio** — LR+ divided by LR-.
- **False discovery rate** — 1 minus PPV.
- **False positive rate** — 1 minus specificity.
- **F1 score** — harmonic mean of precision and recall.
- **Hosmer-Lemeshow test** — goodness-of-fit test for calibration.
- **IDI** — integrated discrimination improvement.
- **Lift** — precision at k divided by prevalence.
- **Likelihood ratio negative** — (1 - sensitivity) / specificity.
- **Likelihood ratio positive** — sensitivity / (1 - specificity).
- **Log loss** — negative log-likelihood.
- **Matthews correlation coefficient** — correlation between prediction and outcome.
- **Net benefit** — prevalence-weighted true positives minus threshold-weighted false positives.
- **NPV** — negative predictive value.
- **NRI** — net reclassification improvement.
- **Number needed to evaluate** — patients reviewed per true case.
- **PPV** — positive predictive value.
- **Precision** — proportion of flagged patients who are positive.
- **Precision at k** — precision among the top k ranked.
- **Prevalence** — proportion of positives.
- **Proper scoring rule** — a metric minimised by the true probability.
- **Recall** — proportion of positives flagged.
- **Sensitivity** — proportion of positives flagged.
- **Specificity** — proportion of negatives not flagged.
- **Spectrum effect** — sensitivity varies by case severity.
- **Wilson interval** — interval for a proportion.
- **Youden index** — sensitivity plus specificity minus one.

## 16. Sources

All URLs accessed 25 October 2026.

1. Steyerberg EW, Vickers AJ, Cook NR, et al. Assessing the performance of prediction models: a
   framework for traditional and novel measures. *Epidemiology* 2010;21(1):128-138.
   https://doi.org/10.1097/EDE.0b013e3181c30fb2
2. Steyerberg EW, Vergouwe Y. Towards better clinical prediction models: seven steps for
   development and an ABCD for validation. *European Heart Journal* 2014;35(29):1924-1932.
   https://doi.org/10.1093/eurheartj/ehu207
3. Saito T, Rehmsmeier M. The precision-recall plot is more informative than the ROC plot when
   evaluating binary classifiers on imbalanced datasets. *PLoS ONE* 2015;10(3):e0118432.
   https://doi.org/10.1371/journal.pone.0118432
4. Pepe MS, Fan J, Feng Z, Gerds T, Hilden J. The net reclassification index (NRI): a misleading
   measure of prediction improvement even with independent test data sets. *Statistics in
   Biosciences*. https://pmc.ncbi.nlm.nih.gov/articles/PMC4615606/
5. Kerr KF, Wang Z, Janes H, McClelland RL, Psaty BM, Pepe MS. Net reclassification indices for
   evaluating risk-prediction instruments: a critical review. *Epidemiology* 2014;25(1):114-121.
   https://pmc.ncbi.nlm.nih.gov/articles/PMC3918180/
6. Leening MJG, Vedder MM, Witteman JCM, Pencina MJ, Steyerberg EW. Net reclassification
   improvement: computation, interpretation, and controversies. *Annals of Internal Medicine*
   2014;160(2):122-131. https://doi.org/10.7326/M13-1522
7. Vickers AJ, Elkin EB. Decision curve analysis. *Medical Decision Making* 2006;26(6):565-574.
   https://doi.org/10.1177/0272989X06295361
8. Van Calster B, McLernon DJ, van Smeden M, et al. Calibration: the Achilles heel of predictive
   analytics. *BMC Medicine* 2019;17:230. https://doi.org/10.1186/s12916-019-1466-7
9. Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement. *BMJ* 2024;385:e078378.
   https://doi.org/10.1136/bmj-2023-078378
10. Moons KGM, Damen JAA, Kaul T, et al. PROBAST+AI. *BMJ* 2025;388:e082505.
    https://doi.org/10.1136/bmj-2024-082505
11. Bossuyt PM, Reitsma JB, Bruns DE, et al. STARD 2015. *BMJ* 2015;351:h5527.
    https://doi.org/10.1136/bmj.h5527
12. Wong A, Otles E, Donnelly JP, et al. External validation of a widely implemented proprietary
    sepsis prediction model. *JAMA Internal Medicine* 2021;181(8):1065-1070.
    https://doi.org/10.1001/jamainternmed.2021.2626
13. Kansagara D, Englander H, Salanitro A, et al. Risk prediction models for hospital readmission.
    *JAMA* 2011;306(15):1688-1698. https://doi.org/10.1001/jama.2011.1515
14. US Food and Drug Administration. Good Machine Learning Practice for Medical Device Development:
    Guiding Principles. 2021.
    https://www.fda.gov/medical-devices/artificial-intelligence-enabled-medical-devices/good-machine-learning-practice-medical-device-development-guiding-principles
15. Liu X, Cruz Rivera S, Moher D, et al. CONSORT-AI extension. *Nature Medicine*
    2020;26:1364-1374. https://doi.org/10.1038/s41591-020-1034-x
16. National Institute of Standards and Technology. Artificial Intelligence Risk Management
    Framework (AI RMF 1.0). NIST AI 100-1, 2023. https://doi.org/10.6028/NIST.AI.100-1
17. scikit-learn developers. Metrics and scoring.
    https://scikit-learn.org/stable/modules/model_evaluation.html
18. Steyerberg EW. *Clinical Prediction Models*. 2nd edition. Springer, 2019.
    https://doi.org/10.1007/978-3-030-16399-0

A final check before reporting a metric: state the prevalence, the threshold, and the interval. If
any of the three is missing, the number is not yet evidence.

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic: [`06_class-imbalance.md`](06_class-imbalance.md)
- Next topic: [`08_calibration.md`](08_calibration.md)
- Labels and outcomes: [`02_labels-and-outcomes.md`](02_labels-and-outcomes.md)
- Survey file: [`../../06_machine_learning_foundations.md`](../../course/06_machine_learning_foundations.md)
- Labs: [`../../labs/`](../../labs/)
- Calibration lab: [`../../labs/06_calibration_decision_curve.ipynb`](../../labs/06_calibration_decision_curve.ipynb)
