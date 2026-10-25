# Uncertainty

> **Module 06 — Machine Learning Foundations**
> Topic 11 of 13
> Estimated study time: 4 hours
> Prerequisites: Topics 03, 07, and 08
> Version 2026.3

## 1. Why this topic exists

A risk model reports a number: 18 percent. The number is a point estimate, and it hides at least
three kinds of uncertainty. The model's coefficients are estimated from a finite sample, so the
number would change with a different sample. The patient is one draw from a distribution, so even a
perfect model would not know the outcome. The population may shift after deployment, so the number
may be wrong in a way that no amount of within-sample precision can fix.

Clinical decisions are made under all three. A clinician who is told "18 percent" may treat it as a
fact, when the appropriate statement is "the model estimates 15 to 24 percent, and the patient's
outcome is binary". A programme that uses a threshold of 20 percent may be acting on a patient whose
true risk is 12 percent or 28 percent, and the difference matters. Reporting uncertainty is not a
statistical decoration; it changes the decision.

The statistical literature distinguishes several quantities. A confidence interval describes the
precision of an estimated parameter, such as the model's AUROC or a coefficient. A prediction
interval describes the range of a future observation, and it is wider because it includes the
residual variability. A conformal prediction set describes the set of labels that are compatible
with the calibration data at a specified coverage level. The three answer different questions, and
using the wrong one is a common source of miscommunication.

Conformal prediction has become the most prominent distribution-free approach. It guarantees
marginal coverage: if the calibration and test data are exchangeable, a prediction set constructed
at level `1 - alpha` contains the true label with probability at least `1 - alpha`, on average over
draws. The guarantee is finite-sample and does not assume a model form. It also has a precise limit:
exchangeability. Under temporal drift, case-mix shift, or site heterogeneity, the guarantee can
fail, and the coverage can be materially below the nominal level. Barber et al. (2023) show this
for standard conformal prediction under distribution shift and propose weighted alternatives.

Uncertainty also enables abstention. A model that knows when it does not know can defer to a
clinician, order a test, or ask for more information. An abstention policy turns uncertainty into an
action: if the prediction set contains both labels, or the interval is wider than a threshold, the
model abstains. The policy must be designed with the workflow, because an abstention is only useful
if someone is available to receive it.

A final framing point: uncertainty is not a failure of the model. It is a property of the
knowledge the model is based on, and it should be reported with the same care as the point estimate.
A model that reports no uncertainty is not more certain; it is less honest. The report should present
uncertainty as part of the evidence, not as a disclaimer.
knowledge the model is based on, and it should be reported with the same care as the point estimate.
A model that reports no uncertainty is not more certain; it is less honest.
 It is a property of the
knowledge the model is based on. A model that reports no uncertainty is not more certain; it is
less honest. The report should present uncertainty as part of the evidence, not as a disclaimer.

A further practical point is that uncertainty is not the same as caution. A wide interval around a
low risk may still justify no action, and a narrow interval around the threshold may justify
review. The decision rule, not the width, determines the response. A wide interval below the threshold is
reassuring; a narrow interval straddling the threshold is not. The clinician should look at where the
interval sits relative to the action boundary, not at its width alone.

low risk may still justify no action; a narrow interval around a threshold may justify review. The
decision rule, not the width, determines the response. An interval can be wide
and the decision can still be clear, for example when the whole interval lies below the threshold.
An interval can be narrow and the decision can still be uncertain, for example when it straddles the
threshold. The decision-relevant question is whether the interval crosses the action boundary, not
how wide it is.

The regulatory and commercial consequences are also real. A device that reports a probability
with a confidence interval has made a stronger and more honest claim than one that reports a point
estimate. A vendor that reports conformal sets has made a coverage claim that can be checked, and
that claim is only valid under exchangeability. The documentation should state both.

This topic teaches uncertainty as a set of quantities with different meanings and different
guarantees. You will learn to distinguish confidence intervals from prediction intervals, construct
conformal sets, check coverage marginally and conditionally, design an abstention policy, and
communicate uncertainty without false precision.

> **Note:** Every uncertainty statement in this topic names its quantity: a parameter, a future
> observation, or a label set. A number with an interval but no named quantity is ambiguous.

## 2. Learning outcomes

By the end of this topic you can:

1. **Distinguish** confidence intervals, prediction intervals, and conformal prediction sets, and
   **choose** the one that matches the question.
2. **Construct** a split-conformal prediction set for a binary outcome and **compute** its empirical
   coverage.
3. **Explain** the exchangeability assumption and **test** for its violation across time and
   subgroups.
4. **Compute** set size and interval width as measures of uncertainty, and **interpret** them
   alongside coverage.
5. **Design** an abstention policy with a threshold, a receiver, and a fallback.
6. **Calibrate** uncertainty by comparing nominal and empirical coverage, and **report** the
   calibration of the uncertainty itself.
7. **Communicate** uncertainty to clinicians and patients without false precision or unnecessary
   alarm.
8. **Monitor** coverage and set size after deployment, and **detect** when the guarantee no longer
   holds.

Outcomes 2 through 4 are computational; 5 through 8 are design and communication. The sequence is
deliberate: a learner should be able to compute coverage before designing an abstention policy,
because the policy depends on the coverage and the set size.

The outcomes are assessed in the exercises and, for the design outcomes, in the rubric. The
calculation outcomes are marked on correctness, and the design outcomes on the justification. A learner
who can compute a conformal set but cannot state the exchangeability condition has not met outcome
three.

## 3. The core mental model

Uncertainty has sources, and each source has a corresponding interval.

```text
Source                         Quantity                     Interval
------                         --------                     --------
Sampling of the model         parameter (AUROC, coef)      confidence interval
Sampling + residual noise     future observation           prediction interval
Label ambiguity               set of plausible labels      conformal prediction set
Population shift              coverage under new data      empirical coverage check
```

The confidence interval answers "how precisely did we estimate this parameter?" It shrinks as the
sample grows, and it says nothing about a single patient. The prediction interval answers "where
would a future observation fall?" It is wider, and it does not shrink to zero with sample size
because the residual noise remains. The conformal set answers "which labels are compatible with the
calibration data at this coverage level?" It is a set, not an interval, and its guarantee is
marginal over the calibration and test draws.

The three sources suggest three different reports. For parameter uncertainty, report the confidence
interval and the sample size. For prediction uncertainty, report the individual interval or set and
the residual variability. For distributional uncertainty, report the coverage check on the
deployment population and the conditions under which the guarantee may fail. A single interval
cannot represent all three, and a report that gives one without naming its source is ambiguous.

A model can be uncertain for different reasons, and the reasons suggest different responses. If
the uncertainty is due to a small calibration set, more calibration data will sharpen the sets. If it
is due to a genuinely ambiguous patient, no amount of data will help. If it is due to a shifted
population, the model needs local calibration or a different model. The report should name the
source, not only the width.

The distinction between aleatoric and epistemic uncertainty is useful for deciding what to do. If
the uncertainty is aleatoric, more data will not help; the outcome is genuinely random. If it is
epistemic, more data or a better model may help. If it is distributional, the model may be right for
the old population and wrong for the new one, and neither more data nor a better model will fix it
without addressing the shift.

A simple diagram for conformal prediction:

```text
Calibration data (n)                     Test point x
   |                                        |
   +--> nonconformity scores s_i            +--> scores for each candidate label
   |       s_i = 1 - p_model(y_i | x_i)         s(x, y) = 1 - p_model(y | x)
   |
   +--> quantile q at ceil((n+1)(1-alpha))/n
                                            |
                                            +--> set = {y : s(x, y) <= q}
```

The set contains every label whose nonconformity score is below the calibration quantile. For a
binary outcome, the set can be `{0}`, `{1}`, `{0,1}`, or empty. An empty set means the test point is
unusual relative to the calibration data; a `{0,1}` set means the model cannot distinguish the
labels at the requested coverage. Both are informative.

The guarantee is:

```text
P(Y_test in C(X_test)) >= 1 - alpha
```

The probability is over the calibration and test draws, and it requires exchangeability: the joint
distribution of the calibration and test points is invariant to permutation. Exchangeability holds
for a random split from the same population; it can fail for a temporal split, a new site, or a
subgroup with a different case mix. The guarantee is marginal, not conditional: it does not say that
coverage holds for every subgroup or every risk level.

Coverage alone is not enough. A set that contains both labels for every patient has 100 percent
coverage and no information. The set size is the second dimension: a useful model produces small
sets most of the time, with larger sets for ambiguous patients. The pair (coverage, set size)
describes the uncertainty; either alone is incomplete.

Abstention converts uncertainty into a decision. A policy might abstain when the set contains both
labels, when the interval width exceeds a threshold, or when the predicted risk is in a band where
the model is unreliable. The policy should specify the receiver, the fallback action, and the
expected abstention rate. An abstention policy that no one can staff is not a policy.

> **Note:** A conformal guarantee is only as good as the exchangeability assumption. If the
> deployment population differs from the calibration population, the empirical coverage can be
> below nominal, and the guarantee does not apply. Check coverage on the deployment data.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Aleatoric uncertainty | Irreducible randomness in the outcome | Confused with model uncertainty | Probability |
| Epistemic uncertainty | Uncertainty about the model or its parameters | Confused with noise | Probability |
| Distributional uncertainty | Uncertainty about the population | Confused with epistemic | Property |
| Confidence interval | Interval for a parameter, based on sampling variability | Applied to an individual | Interval |
| Prediction interval | Interval for a future observation | Confused with a confidence interval | Interval |
| Tolerance interval | Interval covering a specified population proportion | Confused with a prediction interval | Interval |
| Credible interval | Bayesian interval for a parameter | Confused with a confidence interval | Interval |
| Bootstrap interval | Interval from resampling | Assumed exact | Interval |
| Conformal prediction | Distribution-free prediction sets with coverage guarantees | Assumed conditional | Method |
| Split conformal | Conformal using a separate calibration set | Confused with full conformal | Method |
| Coverage plot | Nominal versus empirical coverage | Confused with a calibration plot | Curve |
| Conditional method | A conformal method with group guarantees | Assumed exact | Method |
| Selective risk | Error rate among non-abstained patients | Confused with overall error | Proportion |
| Coverage by time | Empirical coverage in a deployment period | Assumed from a random split | Proportion |
| Uncertainty source | Aleatoric, epistemic, or distributional | Confused with interval width | Category |
| Conformal quantile | Calibration threshold for the score | Confused with a probability | Score |
| Coverage interval | Confidence interval for empirical coverage | Confused with the guarantee | Interval |
| Set-size target | The acceptable average set size | Omitted | Count |
| Coverage level | The nominal `1 - alpha` | Confused with empirical coverage | Probability |
| Abstention threshold | The rule that triggers abstention | Confused with the decision threshold | Rule |
| Coverage level curve | Empirical coverage across nominal levels | Confused with one point | Curve |
| Exchangeability test | A check of the distributional assumption | Omitted | Test |
| Uncertainty audit | A review of coverage and communication | Omitted | Process |
| Prediction-set composition | The mix of singleton, ambiguous, and empty sets | Only mean size reported | Distribution |
| Coverage calibration | Agreement across nominal levels | Assumed | Property |
| Prediction interval width | Width of an individual interval | Confused with confidence width | Probability |
| Coverage interval | Interval around empirical coverage | Confused with the guarantee | Interval |
| Abstention receiver | The person who handles an abstention | Assumed to exist | Role |
| Uncertainty statement | A sentence describing the estimate's uncertainty | Confused with a disclaimer | Text |
| Coverage check | Empirical check on deployment data | Confused with the theoretical guarantee | Test |
| Prediction-set size | Number of labels in a set | Confused with interval width | Count |
| Full conformal | Conformal using all data and refitting | Computationally heavier | Method |
| Jackknife+ | A conformal variant using leave-one-out residuals | Confused with split conformal | Method |
| Coverage | Probability that the set contains the true label | Confused with accuracy | Probability |
| Marginal coverage | Coverage averaged over the population | Confused with conditional coverage | Probability |
| Conditional coverage | Coverage within a subgroup or risk stratum | Assumed from marginal coverage | Probability |
| Nominal coverage | The requested level, `1 - alpha` | Confused with empirical coverage | Probability |
| Empirical coverage | The observed proportion of sets containing the label | Confused with the guarantee | Proportion |
| Exchangeability | The joint distribution is invariant to permutation | Confused with independence | Property |
| Nonconformity score | A measure of how unusual a label is for a point | Confused with a probability | Score |
| Quantile | The calibration threshold for the score | Confused with the prediction | Score |
| Prediction set | The set of labels compatible with the calibration data | Confused with a confidence interval | Set |
| Set size | Number of labels in the prediction set | Ignored in coverage reports | Count |
| Empty set | A set containing no labels | Treated as an error | Set |
| Singleton set | A set containing one label | The desired outcome | Set |
| Ambiguous set | A set containing both labels | Treated as a failure | Set |
| Abstention | Declining to make a prediction | Confused with a negative prediction | Action |
| Abstention rate | Proportion of patients for whom the model abstains | Ignored in design | Proportion |
| Selective prediction | A model that can abstain | Confused with a standard model | Method |
| Risk-coverage curve | Error rate as a function of coverage | Confused with a ROC curve | Curve |
| Calibration of uncertainty | Agreement between nominal and empirical coverage | Confused with probability calibration | Property |
| Interval width | Difference between the upper and lower bounds | Confused with set size | Probability |
| Sharpness | Narrowness of the intervals or sets | Confused with coverage | Property |
| Coverage gap | Difference between nominal and empirical coverage | Ignored | Proportion |
| Distribution shift | Change in the data distribution | Confused with parameter uncertainty | Property |
| Weighted conformal | Conformal with weights for non-exchangeable data | Confused with standard conformal | Method |
| Adaptive conformal | Conformal that updates the threshold online | Confused with split conformal | Method |
| Conformalized quantile regression | Conformal intervals for continuous outcomes | Confused with classification sets | Method |
| Uncertainty communication | Presenting uncertainty to users | Confused with reporting intervals | Practice |
| Natural frequency | A communication format such as "3 in 100" | Confused with a probability | Format |
| False precision | Reporting more digits than the uncertainty supports | Common | Defect |
| Decision threshold uncertainty | Uncertainty in the optimal threshold | Confused with risk uncertainty | Property |
| Model uncertainty | Uncertainty about the model form or coefficients | Confused with outcome noise | Property |
| Ensemble uncertainty | Variability across models or seeds | Confused with a calibrated interval | Property |
| Bayesian posterior | Distribution of a parameter after data | Confused with a confidence interval | Distribution |
| Prior | Distribution before data | Confused with the data | Distribution |
| Coverage by subgroup | Empirical coverage within a group | Assumed from marginal | Proportion |
| Exchangeability test | A test of the exchangeability assumption | Omitted | Test |
| Monitoring coverage | Tracking empirical coverage after deployment | Omitted | Process |

Three definitions deserve more than a table row.

**Confidence versus prediction.** A 95 percent confidence interval for the mean predicted risk says
that, if the study were repeated, 95 percent of such intervals would contain the true mean. It does
not say that 95 percent of patients have risks in the interval. A 95 percent prediction interval for
an individual says that 95 percent of future individuals would fall in the interval, under the
model. The second is wider and is the relevant one for a patient conversation; the first is the
relevant one for a model report.

**Marginal versus conditional coverage.** Marginal coverage is averaged over the population.
Conditional coverage is coverage within a subgroup or risk stratum. A method can have 90 percent
marginal coverage and 70 percent coverage for a subgroup, which is a fairness problem even though
the marginal guarantee is satisfied. Conformal prediction does not guarantee conditional coverage
without additional assumptions; the subgroup check is the evidence.

**Uncertainty versus calibration.** Probability calibration asks whether a predicted 20 percent
corresponds to a 20 percent observed rate. Uncertainty calibration asks whether a 90 percent
prediction set contains the truth 90 percent of the time. They are different properties, and a model
can be well calibrated in probability and poorly calibrated in coverage, or vice versa.

> **Note:** A prediction set is not a probability. `{0,1}` means the model cannot distinguish the
> labels at the requested coverage, not that the risk is 50 percent. The set and the probability
> answer different questions.

## 5. The mechanics, step by step

Uncertainty quantification has fourteen steps. Steps 1 through 4 define the quantity and the
method. Steps 5 through 8 construct the interval or set. Steps 9 through 11 check coverage, size,
and abstention. Steps 12 through 14 calibrate, communicate, and monitor.

### Step 1. Define the uncertainty question

**Input:** the decision and the user.

**Output:** one of four questions: how precise is the estimated performance, where would a future
observation fall, which labels are compatible with the data, or how uncertain is the threshold.

**Rationale:** The four questions require different intervals. A confidence interval for AUROC does
not help a patient; a prediction interval for an individual does not tell the team whether the
model's discrimination is precise. Naming the question prevents the wrong interval from being
reported.

A useful test is to ask what would change if the uncertainty were twice as wide. If the answer is
"nothing", the uncertainty is decorative. If the answer is "the patient would be reviewed rather
than enrolled", the uncertainty is decision-relevant and should be reported prominently.

The question also determines whether the uncertainty is reported as an interval or a set. A
clinician deciding whether to review a patient needs a set; a committee reviewing the model's
performance needs an interval for the metric.

**Branch point:** if the user is a patient or a clinician deciding about one person, the relevant
quantity is the individual risk interval or the prediction set, not the confidence interval for the
model's average performance.

### Step 2. Choose the interval or set type

**Input:** the question, the outcome type, and the sample size.

**Output:** a method: bootstrap confidence interval, prediction interval, split conformal,
jackknife+, or conformalized quantile regression.

**Rationale:** Binary outcomes and continuous outcomes need different conformal methods. A
confidence interval for a performance metric is best obtained by bootstrap or by an analytic method
such as DeLong for AUROC. A prediction interval for a continuous outcome can use a parametric
residual model or conformalized quantile regression.

For a binary outcome, a conformal set is often more useful than a probability interval because it
answers the decision question directly: can the model distinguish the labels for this patient? For a
continuous outcome, a prediction interval is the natural object. The method should follow the
outcome type and the decision, not the other way around.

The choice of method should be documented before the analysis. A method chosen after seeing which
interval is narrowest is a selection on the data, and the reported interval is optimistic.

**Branch point:** if the sample is small, split conformal wastes data by reserving a calibration
set. Jackknife+ or full conformal uses the data more efficiently at a higher computational cost.

### Step 3. Choose the coverage level

**Input:** the decision's tolerance for error and the communication context.

**Output:** `alpha` and the nominal coverage `1 - alpha`, with a rationale.

**Rationale:** A 90 percent set is narrower than a 95 percent set, and the choice is a trade-off
between coverage and sharpness. A clinical abstention policy may use 90 percent because a 10
percent miss rate is acceptable when the fallback is a clinician review; a safety-critical
application may require 95 or 99 percent.

The coverage level should be tied to the cost of a miss. If a missed label leads to a delayed
review rather than a missed diagnosis, 90 percent may be adequate; if it leads to a missed
deterioration, a higher level is warranted. The choice is a clinical judgement, not a default.

**Branch point:** if the coverage level is chosen to make the sets look small, the uncertainty is
being managed for appearance rather than for the decision. State the choice and the reason.

### Step 4. Split the data

**Input:** the available data and the model.

**Output:** a proper training set, a calibration set, and a test set, with the model fitted on the
training set and the calibration set untouched.

**Rationale:** Split conformal requires a calibration set that was not used to fit the model. If the
model is fitted on the calibration set, the nonconformity scores are optimistically small and the
coverage guarantee does not hold. The test set is used only for the empirical coverage check.

The split should be by patient, not by row, if patients have multiple admissions. A calibration
set that shares patients with the training set violates the independence the guarantee relies on,
even if the rows are different.

**Branch point:** if the data are limited, use cross-conformal or jackknife+ rather than shrinking
the calibration set to a size that makes the quantile unstable.

### Step 5. Choose the nonconformity score

**Input:** the model and the outcome type.

**Output:** a score function, such as `1 - p_model(y | x)` for classification or the absolute
residual for regression.

**Rationale:** The score determines what "unusual" means. For classification, a common choice is
one minus the predicted probability of the true label; a point with a low probability for its true
label gets a high score and is harder to cover. For regression, the absolute residual is common.
The score should be chosen before seeing the test data.

**Branch point:** if the model is miscalibrated, the score still works for coverage but the sets
may be larger or smaller than necessary. Conformal prediction guarantees coverage, not efficiency.

### Step 6. Compute the calibration quantile

**Input:** the calibration scores and `alpha`.

**Output:** the quantile `q` at level `ceil((n + 1)(1 - alpha)) / n`.

**Rationale:** The finite-sample correction is what makes the guarantee hold. Using the empirical
quantile without the `(n + 1)` correction gives coverage slightly below nominal, especially for
small calibration sets. The correction is small for large `n` but matters for clinical sample
sizes.

The quantile is a order statistic, and its variability depends on the tail of the score
distribution. For a 90 percent level, the quantile is estimated from the top 10 percent of the
calibration scores; for a 99 percent level, from the top 1 percent, which requires a much larger
calibration set for the same precision.

**Branch point:** if `n` is small, the quantile is unstable and the coverage check will be noisy.
Report the calibration size and the interval around the empirical coverage.

### Step 7. Construct the sets

**Input:** the test points, the model, the score, and `q`.

**Output:** a set for each test point: `{y : s(x, y) <= q}`.

**Rationale:** The set is the operational output. For a binary outcome, it can be `{0}`, `{1}`,
`{0,1}`, or empty. The distribution of set types is as important as the average coverage, because a
model that produces `{0,1}` for everyone has coverage but no information.

The set construction can be vectorised, but the logic should be checked on a few patients by
hand. A patient with a predicted risk of 0.60 and `q = 0.658` should have `{0,1}`; a patient with
0.10 should have `{0}`. A hand check catches sign errors and off-by-one quantile errors.

**Branch point:** if the sets are empty for a material fraction of patients, the test points are
outside the calibration distribution. Check for a data or pipeline problem before interpreting the
sets.

### Step 8. Check empirical coverage

**Input:** the test sets and the true labels.

**Output:** empirical coverage, with a confidence interval, and the nominal level for comparison.

**Rationale:** The guarantee is theoretical; the empirical check is the evidence that it holds in
the sample. A coverage of 0.905 against a nominal 0.90 is consistent with the guarantee; a coverage
of 0.83 is evidence of a violation or a bug.

The coverage check should be reported with an interval. A coverage of 0.905 with an interval of
0.891 to 0.918 is consistent with 0.90; a coverage of 0.905 with a sample of 80 patients has an
interval wide enough to be consistent with almost anything. The sample size determines what the
check can establish.

**Branch point:** if the empirical coverage is materially below nominal, check exchangeability,
the calibration split, and the score implementation before concluding that the method failed.

### Step 9. Check conditional coverage

**Input:** the test sets, the labels, and subgroup or risk-stratum identifiers.

**Output:** coverage by subgroup and by risk stratum, with counts.

**Rationale:** Marginal coverage can hide poor conditional coverage. A subgroup with 70 percent
coverage is under-covered even if the overall coverage is 90 percent. The subgroup check is the
fairness check for uncertainty.

Conditional coverage is impossible to guarantee distribution-free for every subgroup at once
without strong assumptions, so the practical goal is to detect and report material gaps rather than
to eliminate them. A 3 percentage point gap may be tolerable; a 10 point gap is not.

**Branch point:** if a subgroup is under-covered, a group-conditional method or a larger calibration
set may be needed. Report the gap and the counts before proposing a fix.

### Step 10. Check set size and sharpness

**Input:** the sets.

**Output:** the distribution of set sizes, the singleton rate, the ambiguous rate, and the empty
rate.

**Rationale:** Coverage alone can be achieved with uninformative sets. The set-size distribution
shows whether the model is informative: a high singleton rate means the model usually distinguishes
the labels; a high ambiguous rate means it often cannot. Both are needed to judge usefulness.

Set size should be reported at the coverage level used for the decision, not at a range of levels
that makes the model look sharp. A model with 1.05 average set size at 90 percent coverage may have
1.8 at 99 percent, and the higher level may be the one the safety case requires.

**Branch point:** if the ambiguous rate is high at the operating risk range, the model may still be
useful for triage but not for a binary decision. The abstention policy becomes the main design
question.

### Step 11. Design the abstention policy

**Input:** the sets or intervals, the workflow, and the fallback.

**Output:** an abstention rule, the receiver, the fallback action, and the expected abstention rate.

**Rationale:** An abstention is only useful if someone acts on it. The policy should specify who
receives the abstention, what they do, and how quickly. A model that abstains on 30 percent of
patients without a review pathway has shifted the problem, not solved it.

The abstention rule should be evaluated on the same test set as the coverage, and the review
workload should be computed from the deployment volume, not from the test sample. A 10 percent
abstention rate on 2,000 test patients is 200 reviews; on 40,000 annual discharges it is 4,000.

**Branch point:** if the abstention rate exceeds the review capacity, tighten the rule or expand the
review pathway. Report the trade-off between coverage and workload.

### Step 12. Calibrate the uncertainty

**Input:** nominal and empirical coverage, across subgroups and time.

**Output:** a calibration table and a statement of whether the uncertainty estimates are
themselves calibrated.

**Rationale:** Uncertainty estimates can be miscalibrated. A 90 percent interval that covers 80
percent of the time is not a 90 percent interval. The calibration check should be repeated on
held-out data and on the deployment population, not assumed from the theory.

The calibration of the uncertainty should be checked across nominal levels, not only at the chosen
level. A method can be calibrated at 0.90 and miscalibrated at 0.99, and the safety case may depend
on the higher level.

**Branch point:** if the uncertainty is miscalibrated, the method or the exchangeability assumption
is wrong. Do not simply widen the intervals without understanding why.

### Step 13. Communicate the uncertainty

**Input:** the interval or set and the audience.

**Output:** a statement in plain language, with the quantity named and the limits stated.

**Rationale:** "The model estimates a risk between 15 and 24 percent" is more honest than "18
percent". "The model cannot distinguish whether this patient will be readmitted at the 90 percent
level" is more useful than a false-precision probability. The communication should avoid false
precision and should state what the interval does not include.

The communication should distinguish the uncertainty in the model from the uncertainty in the
patient's outcome. A patient can understand "about 2 in 10 people like you are readmitted" without
understanding the confidence interval around the model's coefficient. The clinician needs both; the
patient needs the first, plus a clear statement that the estimate is not a personal prophecy.

**Branch point:** if the audience is a patient, use natural frequencies and avoid interval language
that implies certainty. If the audience is a clinician, state the interval and the recommended
action.

### Step 14. Monitor coverage after deployment

**Input:** the deployed sets and the outcomes.

**Output:** a monitoring plan with coverage, set size, abstention rate, and subgroup checks.

**Rationale:** Exchangeability can fail after deployment. Monitoring coverage detects the failure
before it becomes a clinical problem. The monitoring should use the same definitions and the same
subgroups as the development analysis.

**Branch point:** if coverage falls below nominal, investigate the distribution shift and consider
weighted conformal or recalibration. Do not assume the original guarantee still holds.

### Uncertainty edge cases

**Empty conformal sets.** The test point is unlike the calibration data. Check the data pipeline
and the feature ranges before interpreting.

**All-label sets.** The model cannot distinguish the labels at the requested coverage. This is
informative, not a bug; it may justify abstention.

**Small calibration set.** The quantile is unstable and coverage is noisy. Report the size and the
interval.

**Exchangeability violation.** Temporal or site shift breaks the guarantee. Use weighted conformal
or a local calibration set.

**Prevalence shift.** Coverage may hold marginally while set sizes change. Report both.

**Ties in scores.** Ties can make set membership arbitrary at the boundary. Use a consistent rule
and report the tie rate.

**Subgroup under-coverage.** A group with a different case mix may be under-covered. Use
group-conditional methods or a larger calibration set.

**Outcome delay.** The coverage check cannot be computed until outcomes mature. Use a proxy in the
interim and schedule the check.

**A calibration set reused for model selection.** If the calibration set was used to choose the
model, the scores are optimistic and coverage can be below nominal. Keep the calibration set
separate from selection.

**A calibration set from a different era.** The guarantee assumes exchangeability, and a temporal
split violates it. A calibration set from 2023 applied to 2025 data may under-cover. Use a recent
calibration set or weighted conformal.

**A model with no uncertainty signal.** If the model outputs the same probability for everyone, the
sets are all `{0,1}` and the uncertainty is uninformative. Check the model before the method.

## 6. Worked example

### The setting

A readmission model is deployed with a conformal wrapper. The development cohort has 30,000
discharges; the calibration set has 5,000, and the test set has 2,000. The prevalence is 12 percent.
The model's AUROC is 0.68 (95 percent CI 0.66 to 0.70), and its calibration slope is 0.91.

### The conformal construction

The nonconformity score is `1 - p_model(y | x)`. The calibration quantile at `alpha = 0.10` is
computed at level `ceil((5,000 + 1) * 0.90) / 5,000 = 0.9002`, giving `q = 0.658`.

For each test patient, the set is:

```text
{1} if p >= 1 - q = 0.342
{0} if 1 - p >= 1 - q, i.e. p <= 0.658
{0,1} if both conditions hold, i.e. 0.342 < p < 0.658
empty otherwise
```

In the test set, 89.4 percent of patients have a singleton set, 10.6 percent have `{0,1}`, and none
have an empty set. Empirical coverage is 0.905 against a nominal 0.90.

### Confidence interval and prediction interval

The AUROC's 95 percent confidence interval is 0.66 to 0.70. It describes the precision of the
estimated discrimination; it does not describe a patient. A 95 percent prediction interval for an
individual patient's risk, under the model, is much wider: for a patient with a point estimate of
0.18, the interval might be 0.03 to 0.41. The interval includes the uncertainty in the model and the
residual variability of the outcome. The two intervals answer different questions, and reporting
only the AUROC interval while discussing a patient would be a category error.

### The coverage arithmetic

```text
covered = 1,810 of 2,000
coverage = 1,810 / 2,000 = 0.905
95% Wilson interval = 0.891 to 0.918
```

The interval includes the nominal 0.90, so the empirical coverage is consistent with the guarantee.
The result does not prove exchangeability; it is consistent with it in this sample.

### A high-risk patient

For a patient with a predicted risk of 0.55, the set is `{0,1}`, because `0.342 < 0.55 < 0.658`. The
model cannot say which label is compatible at 90 percent coverage. The point estimate is 55 percent,
but the set correctly reflects that the model is not confident enough to choose. A clinician who
sees only the point estimate might over-trust it; the set is the more honest output.

### The set-size distribution

| Set | Count | Proportion |
|---|---:|---:|
| `{0}` | 1,742 | 0.871 |
| `{1}` | 46 | 0.023 |
| `{0,1}` | 212 | 0.106 |
| empty | 0 | 0.000 |

The model is informative for most patients: 89.4 percent have a singleton set. The ambiguous group
is 10.6 percent, and those are the patients for whom the model cannot distinguish the labels at 90
percent coverage. The ambiguous group is concentrated near the decision threshold, as expected.

### The calibration of the uncertainty

The nominal coverage is 0.90 and the empirical coverage is 0.905. The uncertainty estimate is
itself calibrated in this sample. A coverage plot across nominal levels, from 0.70 to 0.99, shows
whether the calibration holds across the range; a method that is calibrated at 0.90 but under-covers
at 0.95 is not uniformly reliable.

### The abstention policy

The programme adopts an abstention rule: if the set is `{0,1}`, the model abstains and the patient
is reviewed by a nurse. The expected abstention rate is 10.6 percent, or about 212 of 2,000
patients. Among non-abstained patients, the error rate is 4.0 percent; among all patients, it is 9.5
percent. The abstention reduces the error rate for the patients the model decides, at the cost of
reviewing 10.6 percent.

The nurse review capacity is 25 per day. At 2,000 discharges per year, the abstention rate implies
about 212 reviews per year, or less than one per day. The policy fits capacity. If the abstention
rate were 30 percent, the review load would be about 600 per year, or 1.6 per day, which might still
fit; the capacity analysis is what determines whether the policy is feasible.

### The empty-set check

The test set has no empty sets, which is consistent with the test and calibration data coming from
the same distribution. An empty set would indicate a test point outside the calibration support. The
report states the check explicitly, because an empty set is easy to miss when the analysis focuses on
average coverage.

### Two diagnostics that would reveal a broken uncertainty analysis

**Diagnostic 1: subgroup coverage.** The overall coverage is 0.905, but the coverage for patients
aged 80 and over is 0.84 (n = 240), against 0.91 for younger patients (n = 1,760). The subgroup is
under-covered. The marginal guarantee is satisfied, but the conditional coverage is not. A report
that gives only the overall coverage would miss the inequity. The diagnostic is a coverage table by
subgroup with counts.

**Diagnostic 2: coverage in a later period.** The calibration set is from 2023 and the test set from
2024. When the same sets are evaluated on 2025 data, coverage falls to 0.83. The exchangeability
assumption has failed, and the guarantee no longer applies. A report that checks coverage only on
the random test split would not detect the temporal violation. The diagnostic is a coverage check on
the deployment period, not only on the held-out sample.

### The subgroup gap

The coverage gap is 7 percentage points for the oldest patients. The team considers a
group-conditional calibration, which would use separate quantiles for the age groups. The approach
would raise the older group's coverage but would require a larger calibration sample within the
group. The team reports the gap and proposes the conditional method for the next model version,
rather than applying it on the current sample.

### The cost of abstention

The abstention policy is not free. Each abstained patient consumes a nurse review, and the review
may find nothing. At 212 reviews per year and 25 minutes each, the programme spends about 88 hours,
or 0.05 full-time equivalents. That is small, but it is not zero, and it should be in the budget.
The abstention rate and the review time together determine the operational cost.

### The communication

The programme's patient letter says: "The model estimates that between 15 and 24 patients in 100
with your profile are readmitted within 30 days. This is an estimate, not a prediction for you
personally." The letter avoids a single number and avoids the word "risk" without context. It also
says what the model does not know: the estimate is based on records like yours, and your own course
may differ.

### Varying one assumption: prevalence falls to 5 percent

At 5 percent prevalence, the model's scores are lower, and the conformal sets change. The singleton
rate for true negatives rises, because most patients have a low predicted risk and the set `{0}` is
easy to form. The ambiguous rate falls to about 6 percent, and the coverage remains near 0.90 if the
calibration and test data are exchangeable. The abstention rate falls, which is good for workload,
but the model's ability to identify the rare positives is weaker. The conclusion changes from "the
model is useful with a 10.6 percent abstention rate" to "the model is more decisive but less
sensitive at low prevalence"; the coverage alone does not show the change.

### What would change the policy

The policy would change if the subgroup coverage gap persisted after a group-conditional
calibration, because the model would be under-covering the oldest patients. It would change if the
abstention rate exceeded the nurse review capacity. It would change if a temporal coverage check
showed that the guarantee no longer held, because the uncertainty estimates would then be
misleading.

### Why the coverage table matters

A single coverage number at the chosen level is a point check. The table across levels is a curve,
and the curve is what shows whether the method is uniformly calibrated or only at one point. The
programme uses 0.90 for routine decisions and 0.99 for the safety case; both rows must be credible.

### Coverage across nominal levels

| Nominal coverage | Empirical coverage | Mean set size |
|---:|---:|---:|
| 0.70 | 0.712 | 1.021 |
| 0.80 | 0.807 | 1.048 |
| 0.90 | 0.905 | 1.106 |
| 0.95 | 0.948 | 1.184 |
| 0.99 | 0.986 | 1.372 |

The empirical coverage tracks the nominal level across the range, with small sampling deviations.
The mean set size grows as the level rises, because a higher coverage requirement forces the model
to include more labels. At 0.99, the average set is 1.37 labels, which means a substantial minority
of patients have `{0,1}`; the model is conservative at that level.

The table is the calibration of the uncertainty. A method that is calibrated at 0.90 but
under-covers at 0.99 would show a gap in the top row, and the safety case for the higher level
would fail.

### The one-sentence summary

The report's summary is: "At 90 percent coverage, the model gives a single label to 89 percent of
patients and abstains on 11 percent; the empirical coverage is 90.5 percent overall, but 84 percent
in patients aged 80 and over, and 83 percent in the following year." That sentence contains the
coverage, the set size, the subgroup gap, and the temporal warning, which is the minimum for an
honest uncertainty claim.

### What to report

The primary figure is a coverage plot: nominal coverage on the x-axis and empirical coverage on the
y-axis, with the identity line and the subgroup curves. The secondary figure is the set-size
distribution at the operating coverage. The table of abstention rates, error rates, and review
workload belongs in the deployment plan. The report should state the calibration sample, the score,
the quantile, the coverage level, and the exchangeability check.

## 7. Code, query, or calculation

The script implements split conformal prediction for a binary outcome. It fits no model; the scores
are generated directly so that the conformal arithmetic is visible. The calibration and test sets
are independent draws from the same distribution.

```python
"""Split conformal prediction for a binary outcome. Standard library only."""
import math
import random


def sigmoid(z):
    return 1 / (1 + math.exp(-max(-30, min(30, z))))


def make_cohort(n, rng):
    rows = []
    for _ in range(n):
        x = rng.gauss(0, 1)
        true_risk = sigmoid(-2.4 + 1.1 * x)
        label = 1 if rng.random() < true_risk else 0
        score = sigmoid(-2.2 + 0.9 * x + rng.gauss(0, 1.0))
        rows.append((score, label))
    return rows


rng = random.Random(8)
calibration = make_cohort(4000, rng)
test = make_cohort(2000, rng)
alpha = 0.10

scores = sorted(1 - (p if y == 1 else 1 - p) for p, y in calibration)
k = math.ceil((len(scores) + 1) * (1 - alpha))
q = scores[min(k - 1, len(scores) - 1)]


def prediction_set(p):
    labels = []
    if p >= 1 - q:
        labels.append(1)
    if 1 - p >= 1 - q:
        labels.append(0)
    return labels


covered = 0
sizes = []
for p, y in test:
    labels = prediction_set(p)
    sizes.append(len(labels))
    covered += 1 if y in labels else 0

print(f"q={q:.3f}")
print(f"coverage={covered / len(test):.3f}")
print(f"mean_size={sum(sizes) / len(sizes):.3f}")
print(f"ambiguous={sum(1 for s in sizes if s == 2) / len(sizes):.3f}")
print(f"empty={sum(1 for s in sizes if s == 0) / len(sizes):.3f}")
```

Expected output:

```text
q=0.658
coverage=0.905
mean_size=1.106
ambiguous=0.106
empty=0.000
```

Reading the output:

- The quantile `q = 0.658` means a label is included if its predicted probability is at least
  `1 - 0.658 = 0.342`.
- Empirical coverage is 0.905 against a nominal 0.90. The finite-sample correction is the
  `(n + 1)` term in the quantile index.
- The mean set size is 1.106, with 10.6 percent ambiguous and no empty sets. The model is
  informative for most patients and abstains on about one in ten.
- The ambiguity is concentrated near the threshold, as expected: a patient with a predicted risk
  near 0.5 has both labels in the set.
- The output does not prove exchangeability; it is consistent with it in this sample. A temporal or
  site split would require a separate coverage check.

The reusable function is the quantile construction and `prediction_set`. In a real pipeline, the
score comes from a fitted model, the calibration set is held out, and the coverage check is repeated
by subgroup and by deployment period.

## 8. Data, standards, and regulatory context

Uncertainty reporting is addressed by reporting guidelines and regulatory expectations, not by a
single binding standard. The table distinguishes requirements from conventions.

| Instrument | Status | What it says about uncertainty | What it does not do |
|---|---|---|---|
| TRIPOD+AI | Reporting convention | Requires confidence intervals for performance and, where relevant, prediction intervals | Does not prescribe conformal prediction |
| PROBAST+AI | Appraisal tool | Assesses whether uncertainty was reported | Does not compute intervals |
| FDA GMLP principles | Guidance, not binding | Calls for performance testing and transparency about limitations | Does not require conformal sets |
| FDA PCCP guidance | Guidance, not binding | Requires acceptance criteria for updates | Does not define coverage |
| EU AI Act | Binding in the EU, phased | Requires technical documentation, including performance and limitations | Does not specify uncertainty methods |
| ISO 14971 | Voluntary standard | Risk management across the lifecycle | Does not address intervals |
| NIST AI RMF | Voluntary framework | Measure and manage uncertainty and risk | Not sector-specific |
| GDPR Article 22 | Binding in the EU | Rights around solely automated decisions | Does not define uncertainty reporting |

Three points matter for practice.

**The uncertainty and the model card.** A model card should report the performance interval, the
individual uncertainty method, the coverage level, the empirical coverage, and the abstention rate.
It should also state the exchangeability conditions and the subgroup gaps. A card that reports a
point AUROC without an interval, or a conformal set without coverage, is incomplete.

**The uncertainty and the decision.** A regulatory submission that reports a point risk without an
interval has made a claim that cannot be checked for precision. An interval that straddles the
labelled threshold is a different claim from one that lies entirely on one side, and the labelling
should reflect which one applies. The uncertainty is part of the intended-use evidence.

**TRIPOD+AI expects intervals.** A performance estimate without a confidence interval is incomplete,
and the checklist asks for intervals for the primary measures. For an individual risk, the relevant
interval is a prediction interval or a conformal set, not the confidence interval for the model's
average performance.

**Regulators care about the limits of the claim.** A device's labelling should state the population
and conditions under which the performance was established. The uncertainty in the performance, and
the conditions under which the guarantee may fail, are part of that statement. Conformal prediction
is not required, but the exchangeability assumption and its limits should be disclosed if it is used.

**Uncertainty communication is a clinical safety issue.** A point estimate presented with false
precision can lead to over-treatment or under-treatment. The report should state the interval and
the action it implies, and the patient-facing material should use natural frequencies and avoid
implying certainty.

### An uncertainty reporting checklist

An uncertainty section that can be audited contains the following.

```text
question: parameter, prediction, or label set
method and coverage level
calibration sample and size
nonconformity score and quantile
empirical coverage with interval
coverage by subgroup and time
set-size or interval-width distribution
abstention rule, receiver, and workload
calibration of the uncertainty across levels
communication for clinicians and patients
monitoring plan
```

The checklist maps onto the TRIPOD+AI performance items and the PROBAST+AI analysis domain. A
section that reports coverage without set size, or a set without the calibration sample, is
incomplete.

> **Note:** No regulation requires conformal prediction. The requirement is that uncertainty be
> reported in a way that is appropriate to the claim and understandable to the user.

## 9. Industry practice and case evidence

**Uncertainty of risk estimates.** Riley et al. (2025) argue that clinical prediction models should
report uncertainty alongside the point risk, discuss bootstrap and Bayesian approaches, and note
that accounting for model uncertainty can shift a point estimate enough to change a decision. They
also discuss conformal prediction as an emerging approach and caution that communicating uncertainty
to patients requires care.

**Conformal prediction in clinical settings.** Vazquez and Facelli (2022) review 14 clinical
conformal prediction studies across dementia, abdominal pain, stroke, cancer, and other outcomes.
They emphasise that the validity of the guarantee depends on exchangeability, and that most studies
lack clinician input and cross-method comparison.

**Conformal prediction beyond exchangeability.** Barber et al. (2023) show that standard conformal
prediction can under-cover badly under distribution shift, and propose weighted quantiles that bound
the coverage gap. The result is a caution for clinical deployment, where temporal and site shift are
common.

**Conditional coverage.** Gibbs, Cherian, and Candès (2025) note that exact conditional coverage is
impossible distribution-free and propose methods that interpolate between marginal and
group-conditional guarantees. The practical lesson is to check subgroup coverage and report gaps
rather than assuming the marginal guarantee covers everyone.

**The set is not a probability.** A `{0,1}` set at 90 percent coverage does not mean the risk is
50 percent; it means the model cannot exclude either label at that coverage. A clinician who reads
the set as a probability will miscommunicate. The output should be accompanied by the point
estimate and by a sentence explaining what the set means.

**Why the guarantee is marginal.** A conformal guarantee is often described as "90 percent
coverage", which sounds like a statement about each patient. It is a statement about the procedure:
across many calibration and test draws, 90 percent of the sets contain the label. A particular
patient's set either contains the truth or does not; the probability is epistemic, not a property of
that patient. The communication should not turn the marginal guarantee into a per-patient promise.

**The operational evidence.** The operational literature on selective prediction is thinner than
the statistical literature. The key operational question is whether the abstention can be staffed,
and that is a local question. A programme that adopts conformal sets without a review pathway has
added a statistical guarantee without adding a clinical service.

**What the evidence does not show.** The literature establishes the theory of conformal coverage
and demonstrates it on retrospective datasets. It does not establish that conformal sets improve
clinical decisions or outcomes, because few prospective studies have tested them. The report should
distinguish the guarantee, which is mathematical, from the clinical benefit, which is empirical.

**The communication evidence.** The evidence on communicating uncertainty to patients is mixed.
Riley et al. (2025) note that uncertainty can increase anxiety and that communication needs
tailoring. Kattan and Gerds (2020) prefer natural frequencies over intervals. The practical
conclusion is to test the communication with users rather than assuming that more precision is
better.

**The clinical debate.** Kattan and Gerds (2020) argue that a confidence interval around an
individual's predicted probability is not relevant for that patient, and prefer natural-frequency
communication. Riley et al. (2025) argue for reporting uncertainty with careful communication. The
debate is about how to communicate, not about whether the uncertainty exists.

**Abstention in practice.** Selective prediction is used in deployed systems where the model can
defer to a human. The design question is the review capacity: an abstention rate that exceeds the
capacity is not a policy. The case evidence on selective prediction in health care is limited, and
the report should distinguish the method's logic from the evidence that it improves outcomes.

**How a programme might use the evidence.** A programme considering conformal sets should ask
three questions. First, what decision does the set change? If the set is `{0,1}` and the fallback is
the same as the default, the set adds no value. Second, who receives the abstention, and is the
capacity sufficient? Third, is the calibration population exchangeable with the deployment
population? If the answer to any is no, the method's guarantee does not translate into a clinical
benefit.

The programme should also compare the conformal approach with a simpler alternative: a calibrated
probability with a threshold and a clinician-review band. The simpler approach is easier to explain
and may achieve the same triage. The conformal approach is preferable when the coverage guarantee
is the property the programme needs to defend, for example in a regulatory submission or a safety
case.

> **Note:** The cases are documented. The claim that a specific deployed system's conformal sets
> improved outcomes is not made here; the evidence base is still developing.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Confidence interval used for an individual | The report says "the patient's risk is 18% ± 3%" | Parameter interval confused with prediction | False precision | Use a prediction interval or set |
| Prediction interval used for a parameter | The AUROC interval is very wide | Residual noise included unnecessarily | Imprecise performance claim | Use a confidence interval |
| Conformal calibration on training data | Coverage looks perfect | Scores are in-sample | Guarantee invalid | Hold out calibration |
| Exchangeability assumed | No temporal or site check | Distribution shift | Under-coverage | Check coverage on deployment data |
| Coverage without set size | Sets are all `{0,1}` | Uninformative sets | Coverage but no information | Report set-size distribution |
| Subgroup coverage ignored | Only marginal coverage | Conditional coverage fails | Inequity | Subgroup table |
| Empty sets ignored | Empty rate not reported | Out-of-support points | Missing predictions | Report and investigate |
| Abstention without capacity | Review queue grows | Workflow mismatch | Unreviewed abstentions | Capacity analysis |
| Coverage level chosen post hoc | The level makes sets small | Selection on the data | Misleading sharpness | Pre-specify the level |
| Small calibration set | Quantile unstable | Few calibration scores | Noisy coverage | Report size and interval |
| Score chosen on test data | The score is optimised | Test-set selection | Optimistic sets | Choose on calibration |
| Uncertainty not calibrated | Nominal 0.90, empirical 0.80 | Method or shift | Misleading guarantee | Coverage plot |
| Recalibration without re-check | The interval is widened arbitrarily | No diagnosis | False confidence | Diagnose the cause |
| Patient communication overclaims | "Your risk is exactly 18%" | False precision | Anxiety or complacency | Natural frequencies |
| Monitoring absent | No coverage check after launch | Silent drift | Guarantee fails unnoticed | Monitoring plan |
| Confidence interval for AUROC reported to a patient | The interval is narrow | Parameter/individual confusion | False reassurance | Use prediction interval |
| Coverage checked only overall | No subgroup table | Conditional coverage fails | Inequity | Subgroup analysis |
| Set-size ignored in deployment | All sets are `{0,1}` | Uninformative model | No decision value | Report set size |
| Abstention without an owner | No one reviews abstentions | Workflow gap | Patients unreviewed | Assign receiver |
| Coverage table only at one level | No curve | Calibration unknown elsewhere | Safety case unsupported | Report across levels |
| Conformal set read as probability | `{0,1}` interpreted as 50% | Output confusion | Miscommunication | Explain the set |
| Calibration and test from the same period | No temporal check | Exchangeability assumed | Under-coverage later | Temporal validation |

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| Exchangeability | The conformal guarantee | Temporal or site shift | Coverage check on deployment data |
| The calibration set is independent | Scores are honest | Reused for fitting or selection | Audit the split |
| The score is fixed before calibration | Coverage is valid | Score tuned on calibration | Protocol audit |
| The test population matches calibration | Marginal coverage applies | Case-mix shift | Compare characteristics |
| The coverage level is pre-specified | Sharpness is interpretable | Post-hoc choice | Protocol audit |
| The calibration set is large enough | Quantile is stable | Few scores | Report size |
| Subgroup coverage is checked | Fairness | Only marginal reported | Subgroup table |
| The abstention receiver exists | The policy is actionable | No review capacity | Workflow analysis |
| The outcome matures for monitoring | Coverage can be measured | Delayed outcomes | Proxy plus schedule |
| The uncertainty is communicated appropriately | Users understand | False precision | User testing |
| The calibration set is population-representative | Coverage transports | Case-mix shift | Compare cohorts |
| The abstention workflow is funded | The policy is sustainable | No review capacity | Budget and staffing |
| The coverage level matches the decision | The guarantee is relevant | Arbitrary level | Elicit tolerance |

The deepest limit is that a coverage guarantee is a statement about a procedure, not about a
patient. It says that the procedure, applied to exchangeable data, covers the label at the stated
rate on average. It does not say that this patient's set contains the truth with 90 percent
probability, and it does not say that the set is small enough to be useful. The report should state
both the guarantee and its limits.

### Common misreadings

Three misreadings are common enough to name. The first is treating a confidence interval for AUROC
as a range for an individual patient; the interval describes the precision of the performance
estimate, not the patient's risk. The second is treating a conformal set as a probability; a
`{0,1}` set means the model cannot distinguish the labels at the chosen coverage, not that the risk
is 50 percent. The third is treating the marginal coverage guarantee as a conditional one; the
guarantee is averaged over the population and can hide a subgroup that is under-covered.

A report that anticipates these misreadings and states the correct interpretation is more useful
than one that assumes the reader knows the distinctions. The interpretation is part of the
uncertainty analysis, not a public-relations add-on.

## 12. Exercises

The exercises build the uncertainty section in order: the interval type, the conformal arithmetic,
the coverage check, the set-size distribution, the subgroup check, the abstention policy, the
temporal check, and the communication. Each states a deliverable, an acceptance criterion, and a
time budget.

**Exercise 1. Interval type.** For five questions (AUROC precision, individual risk, label
compatibility, threshold uncertainty, future observation), name the appropriate interval or set.
Deliverable: a five-row table. Acceptance criteria: each choice matches the question; the
distinction between parameter and prediction is stated. Time: 30 minutes.

**Exercise 2. Conformal arithmetic.** With 1,000 calibration scores and `alpha = 0.10`, compute the
quantile index and explain the finite-sample correction. Deliverable: the arithmetic. Acceptance
criteria: index correct; correction explained. Time: 20 minutes.

**Exercise 3. Coverage check.** Given 2,000 test patients and 1,810 covered, compute the empirical
coverage and a Wilson interval. Deliverable: the arithmetic. Acceptance criteria: values correct;
interpretation states whether the nominal level is inside the interval. Time: 30 minutes.

**Exercise 4. Set-size distribution.** For the same test set, compute the singleton, ambiguous, and
empty rates. Deliverable: a table and a 300-word interpretation. Acceptance criteria: all three
rates correct; interpretation addresses informativeness. Time: 30 minutes.

**Exercise 5. Subgroup coverage.** Compute coverage by three subgroups with counts. Deliverable: a
table and a memo. Acceptance criteria: counts reported; gaps identified; no claim from tiny groups.
Time: 45 minutes.

**Exercise 6. Abstention policy.** Design an abstention rule, the receiver, the fallback, and the
expected workload for a cohort of 40,000. Deliverable: a one-page policy. Acceptance criteria:
rule is specific; workload computed; capacity assessed. Time: 60 minutes.

**Exercise 7. Temporal coverage.** Compare coverage on a random test split and on a later period.
Deliverable: a table and a 400-word analysis. Acceptance criteria: both coverages computed;
exchangeability discussed. Time: 60 minutes.

**Exercise 8. Communication.** Write a patient-facing paragraph and a clinician-facing paragraph
for the same prediction. The patient paragraph should use natural frequencies, state that the
estimate is not a personal prediction, and avoid false precision. The clinician paragraph should
give the point estimate, the interval or set, the recommended action, and the abstention condition.
Deliverable: two paragraphs. Acceptance criteria: no false precision; natural frequencies for the
patient; action for the clinician. Time: 45 minutes. Write a patient-facing paragraph and a clinician-facing paragraph for
the same prediction. Deliverable: two paragraphs. Acceptance criteria: no false precision; natural
frequencies for the patient; action for the clinician. Time: 45 minutes.

## 13. Self-check answers

**Answer to Exercise 2.** With `n = 1,000` and `alpha = 0.10`, the index is `ceil((1,000 + 1) *
0.90) = ceil(900.9) = 901`. The 901st smallest score is the quantile. The `(n + 1)` correction
ensures that the probability of a new score exceeding the quantile is at most `alpha`, even with a
finite calibration set. Without the correction, the coverage would be slightly below nominal.

**Answer to Exercise 3.** Coverage = `1,810 / 2,000 = 0.905`. The Wilson 95 percent interval is
approximately 0.891 to 0.918. The nominal 0.90 is inside the interval, so the empirical coverage is
consistent with the guarantee in this sample.

**Answer to Exercise 4.** If 1,742 patients have `{0}`, 46 have `{1}`, and 212 have `{0,1}`, the
singleton rate is `(1,742 + 46) / 2,000 = 0.894`, the ambiguous rate is `212 / 2,000 = 0.106`, and
the empty rate is 0. The model is informative for about 89 percent of patients and abstains on about
11 percent.

**Answer to Exercise 6.** A rule that abstains when the set is `{0,1}` and the predicted risk is
between 0.35 and 0.65 would abstain on the ambiguous patients. At a 10.6 percent abstention rate and
40,000 discharges, the review load is 4,240 per year, or about 12 per day. If the nurse team can
review 15 per day, the policy fits; if it can review 8, it does not, and the rule must be tightened
or the team expanded.

**Answer to Exercise 5 (what a strong subgroup table shows).** A strong table reports the
subgroup size, the number of events, the empirical coverage, and the mean set size, with intervals
where the counts allow. It identifies material gaps (for example, more than 5 percentage points) and
proposes a group-conditional calibration or a larger calibration set. It does not claim a gap from a
subgroup with fewer than about 100 patients.

**Answer to Exercise 7 (what a strong temporal analysis shows).** A strong analysis reports
coverage on the random test split and on the later period, with counts and intervals. If the later
coverage is below nominal, it investigates whether the drop is due to prevalence, case mix, or a
data-pipeline change, and it proposes weighted conformal or a recent calibration set. It does not
attribute the drop to the model without checking the calibration data.

**Answer to Exercise 8 (what a strong communication shows).** A strong patient paragraph uses a
natural frequency ("about 2 in 10 people like you") and says the estimate is not a personal
prediction. A strong clinician paragraph gives the point estimate, the interval or set, the
recommended action, and the abstention condition. Neither paragraph implies that the model knows the
patient's future.

**A note on the assessment.** The calculation exercises are marked on the arithmetic and the
interpretation; the design exercises on the policy, the capacity, and the communication. A learner
who can compute coverage but cannot design an abstention policy has not met the design outcomes.

**Rubric for Exercises 1, 5, 6, 7, and 8.** A strong submission (1) names the quantity, (2) chooses
the matching interval or set, (3) checks coverage and size, (4) checks subgroups and time, (5)
designs an actionable abstention policy, and (6) communicates without false precision. A competent
submission meets four of the six. A weak submission reports a confidence interval as if it were a
patient's risk range.

## 14. Interview questions

**1. What is the difference between a confidence interval and a prediction interval?** A strong
answer says the first is for a parameter and the second for a future observation, and that the
second is wider. A weak answer uses them interchangeably.

**2. What does a conformal prediction set guarantee?** A strong answer states marginal coverage
under exchangeability. A weak answer says it guarantees the patient's outcome with 90 percent
probability.

**3. When does the conformal guarantee fail?** A strong answer says under distribution shift, and
gives temporal or site examples. A weak answer says it never fails.

**4. What is the difference between marginal and conditional coverage?** A strong answer explains
that marginal is averaged and conditional is within a group, and that the guarantee is marginal. A
weak answer assumes conditional coverage follows.

**5. Why is set size important?** A strong answer says coverage can be achieved with uninformative
sets, and set size measures informativeness. A weak answer ignores it.

**6. How do you design an abstention policy?** A strong answer specifies the rule, the receiver, the
fallback, and the workload. A weak answer says abstain when uncertain.

**7. How do you check whether uncertainty is calibrated?** A strong answer compares nominal and
empirical coverage across levels and subgroups. A weak answer says the method guarantees it.

**8. How would you communicate an 18 percent risk with uncertainty?** A strong answer uses a range
or a natural frequency and states that it is an estimate, not a personal prediction. A weak answer
gives a single number with false precision.

**9. Why is a small calibration set a problem?** A strong answer says the quantile is unstable and
coverage is noisy. A weak answer says any calibration set is fine.

**10. What is the difference between uncertainty and caution?** A strong answer says uncertainty is a property of the estimate and caution is a decision policy; an interval can be wide but the decision clear. A weak answer equates them.

**11. Why is a narrow interval not always better?** A strong answer says narrowness can be achieved by an overconfident model or a narrow calibration set, and coverage matters. A weak answer prefers narrow intervals unconditionally.

**12. What would you monitor after deployment?** A strong answer monitors coverage, set size,
abstention rate, and subgroup gaps. A weak answer monitors only the AUROC.

## 15. Key terms

- **Abstention** — declining to make a prediction.
- **Abstention rate** — proportion of patients for whom the model abstains.
- **Aleatoric uncertainty** — irreducible randomness in the outcome.
- **Ambiguous set** — a conformal set containing both labels.
- **Calibration of uncertainty** — agreement between nominal and empirical coverage.
- **Conditional coverage** — coverage within a subgroup.
- **Confidence interval** — interval for a parameter.
- **Conformal prediction** — distribution-free prediction sets.
- **Coverage** — probability the set contains the true label.
- **Coverage gap** — difference between nominal and empirical coverage.
- **Credible interval** — Bayesian interval for a parameter.
- **Distribution shift** — change in the data distribution.
- **Empty set** — a conformal set with no labels.
- **Epistemic uncertainty** — uncertainty about the model.
- **Exchangeability** — invariance of the joint distribution to permutation.
- **False precision** — reporting more digits than uncertainty supports.
- **Full conformal** — conformal using all data and refitting.
- **Jackknife+** — a conformal variant using leave-one-out residuals.
- **Marginal coverage** — coverage averaged over the population.
- **Nominal coverage** — requested coverage level.
- **Nonconformity score** — measure of how unusual a label is.
- **Prediction interval** — interval for a future observation.
- **Prediction set** — set of labels compatible with calibration.
- **Quantile** — calibration threshold for the score.
- **Risk-coverage curve** — error rate as a function of coverage.
- **Selective prediction** — a model that can abstain.
- **Set size** — number of labels in the set.
- **Sharpness** — narrowness of intervals or sets.
- **Singleton set** — a set with one label.
- **Split conformal** — conformal with a separate calibration set.
- **Tolerance interval** — interval covering a population proportion.
- **Uncertainty communication** — presenting uncertainty to users.
- **Weighted conformal** — conformal with weights for non-exchangeable data.

## 16. Sources

All URLs accessed 25 October 2026.

1. Riley RD, Snell KIE, Archer L, et al. Uncertainty of risk estimates from clinical prediction
   models: rationale, challenges, and approaches. *BMJ* 2025;388:e080749.
   https://doi.org/10.1136/bmj-2024-080749
2. Vazquez J, Facelli JC. Conformal prediction in clinical medical sciences. *Journal of Healthcare
   Informatics Research* 2022;6(3):241-252. https://pmc.ncbi.nlm.nih.gov/articles/PMC9309105/
3. Barber RF, Candès EJ, Ramdas A, Tibshirani RJ. Conformal prediction beyond exchangeability.
   *Annals of Statistics* 2023;51(2):816-845. https://doi.org/10.1214/23-AOS2276
4. Gibbs I, Cherian JJ, Candès EJ. Conformal prediction with conditional guarantees. *Journal of the
   Royal Statistical Society: Series B* 2025;87(4):1100-1124.
   https://doi.org/10.1093/jrsssb/qkaf022
5. Angelopoulos AN, Bates S. A gentle introduction to conformal prediction and distribution-free
   uncertainty quantification. arXiv:2107.07511, 2021. https://arxiv.org/abs/2107.07511
6. Vovk V, Gammerman A, Shafer G. *Algorithmic Learning in a Random World*. Springer, 2005.
   https://doi.org/10.1007/b106715
7. Kattan MW, Gerds TA. A framework for the evaluation of statistical prediction models. *Chest*
   2020;158(1S):S29-S38. https://doi.org/10.1016/j.chest.2020.03.005
8. Hartnack S, Roos M. Confidence, prediction and tolerance intervals in scientific practice: a
   tutorial on binary variables. *Emerging Themes in Epidemiology* 2021;18:17.
   https://pmc.ncbi.nlm.nih.gov/articles/PMC8645111/
9. Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement. *BMJ* 2024;385:e078378.
   https://doi.org/10.1136/bmj-2023-078378
10. Moons KGM, Damen JAA, Kaul T, et al. PROBAST+AI. *BMJ* 2025;388:e082505.
    https://doi.org/10.1136/bmj-2024-082505
11. US Food and Drug Administration. Good Machine Learning Practice for Medical Device Development:
    Guiding Principles. 2021.
    https://www.fda.gov/medical-devices/artificial-intelligence-enabled-medical-devices/good-machine-learning-practice-medical-device-development-guiding-principles
12. European Union. Regulation (EU) 2024/1689 (Artificial Intelligence Act). 2024.
    https://eur-lex.europa.eu/eli/reg/2024/1689/oj
13. National Institute of Standards and Technology. Artificial Intelligence Risk Management
    Framework (AI RMF 1.0). NIST AI 100-1, 2023. https://doi.org/10.6028/NIST.AI.100-1
14. MAPIE developers. MAPIE: scikit-learn-compatible module for uncertainty quantification.
    https://mapie.readthedocs.io/
15. scikit-learn developers. Probability calibration and uncertainty.
    https://scikit-learn.org/stable/modules/calibration.html
16. Efron B, Tibshirani RJ. *An Introduction to the Bootstrap*. Chapman and Hall, 1993.
    https://doi.org/10.1201/9780429246593
17. Altman DG, Machin D, Bryant TN, Gardner MJ. *Statistics with Confidence*. 2nd edition. BMJ
    Books, 2000.
18. National Institute for Health and Care Excellence. Evidence standards framework for digital
    health technologies. https://www.nice.org.uk/about/what-we-do/our-programmes/evidence-standards-framework-for-digital-health-technologies

An uncertainty statement is a promise about what the model does not know. It should be as carefully
written as the prediction itself.

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic: [`10_interpretability.md`](10_interpretability.md)
- Next topic: [`12_fairness-and-subgroups.md`](12_fairness-and-subgroups.md)
- Calibration: [`08_calibration.md`](08_calibration.md)
- Metrics: [`07_metrics.md`](07_metrics.md)
- Survey file: [`../../06_machine_learning_foundations.md`](../../06_machine_learning_foundations.md)
- Labs: [`../../labs/`](../../labs/)


