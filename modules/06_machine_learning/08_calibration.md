# Calibration

> **Module 06 — Machine Learning Foundations**
> Topic 08 of 13
> Estimated study time: 3.5 hours
> Prerequisites: Topics 03, 05, and 07
> Version 2026.3

## 1. Why this topic exists

Discrimination tells you whether a model can rank patients. Calibration tells you whether the
number attached to the rank is true. A model can have an AUROC of 0.80 and predict a 40 percent
risk for a patient whose actual risk is 10 percent. The ranking is useful for allocating a fixed
number of slots; the number is useless, and dangerous, for a conversation with a patient or a
threshold derived from a harm ratio.

Van Calster et al. (2019) called calibration "the Achilles heel of predictive analytics" for a
reason. Calibration is harder to estimate than discrimination, harder to report, and easier to
ignore. It is also the property that determines whether a probability can be used as a probability.
A model that is miscalibrated will over-treat or under-treat at a threshold, and the error is
invisible in an AUROC. Van Calster and Vickers (2015) showed that miscalibration always reduces net
benefit and can make a model clinically harmful even when its discrimination is good.

The harm is concrete. A sepsis model that predicts 30 percent risk for patients whose true risk is
12 percent will generate three times as many alerts as the threshold implies. A readmission model
that predicts 15 percent for patients whose true risk is 25 percent will miss the patients who most
need the programme. A risk calculator shown to a patient will misstate their odds. In each case the
model's ranking may be excellent, and the decision is still wrong.

Calibration is also the property most likely to drift after deployment. Case mix changes, coding
changes, treatment changes, and data-pipeline changes can all shift the relationship between the
predicted probability and the observed frequency, even when the ranking is stable. A model that was
calibrated at development can become miscalibrated at the new site or in the new year. Monitoring
calibration is therefore part of maintaining the model, not an optional validation step.

The statistical literature has converged on a small set of measures. The calibration plot shows the
shape; the calibration slope and intercept locate the error; the Brier score summarises overall
performance as a proper scoring rule; and the integrated calibration index (ICI) or expected
calibration error (ECE) summarises the average absolute deviation. The Huang et al. (2020) tutorial
provides a practical guide to these measures and to the recalibration methods (Platt scaling,
isotonic regression, Bayesian binning). The consensus is that a calibration plot and a slope and
intercept should accompany every probability model, not only the ones that "look miscalibrated".

This topic teaches calibration as a measurement and a repair. You will learn to plot it, quantify
it, interpret the slope and intercept, decompose the Brier score, recalibrate on held-out data,
validate the recalibration, and monitor drift. The output is a calibration section that a
statistician can check and a clinician can understand.

A final reason to care is that calibration is the property most amenable to ongoing monitoring. It
is also the property that most directly connects the model to the patient conversation, because the
number is what the patient hears. A patient who is told "20 percent" and later learns that the
clinic's observed rate for similar patients is 10 percent has been misinformed, even if the model
ranked them correctly.

The outcome is observed, the predicted probability is recorded, and the comparison can be repeated on a
rolling window. The comparison is simple arithmetic: mean predicted risk against observed
prevalence, and a slope on the logit scale when enough events have accrued. Discrimination monitoring requires enough events to estimate a ranking; calibration
monitoring requires only the mean prediction and the observed rate, which are available sooner and
with less data. That makes calibration a practical early-warning signal.

The calibration problem is also a communication problem. A clinician who is told that a patient has
a 20 percent risk will act on that number, not on the model's AUROC. If the number is wrong, the
clinician's action is wrong even if the ranking is right. A model card that reports AUROC and omits
calibration gives the clinician no way to know whether the number can be trusted.

There is a regulatory dimension as well. A device that outputs a probability is making a claim about
that probability. The claim must be supported by evidence that the probability is calibrated in the
intended population. A submission that reports discrimination and not calibration leaves the
probability claim unsupported, even if the discrimination is excellent.

The practical consequence is that calibration should be designed into the validation from the
start, not added after a reviewer asks. The calibration sample, the measure, the recalibration
method, and the monitoring trigger should all be specified before the model is fitted, because
each of them affects what the validation can claim. That means reserving a calibration sample, choosing the
measure in advance, and planning the recalibration and monitoring steps. The cost is modest; the
cost of retrofitting is a new validation.

> **Note:** A probability model without a calibration assessment is incomplete. An AUROC without a
> calibration plot is a ranking claim, not a probability claim.

## 2. Learning outcomes

By the end of this topic you can:

1. **Construct** a calibration plot by decile or smoother, with the distribution of predicted risks.
2. **Estimate** calibration slope and intercept by regressing the outcome on the logit of the
   predicted risk, and **interpret** their direction.
3. **Compute** the Brier score and its reliability, resolution, and uncertainty components.
4. **Explain** why AUROC is invariant to monotone recalibration and why that makes it insufficient.
5. **Recalibrate** a model with Platt scaling or isotonic regression on held-out data, and
   **validate** the recalibration.
6. **Quantify** the effect of miscalibration on net benefit at a threshold.
7. **Monitor** calibration over time and **distinguish** calibration drift from case-mix change.
8. **Report** calibration with the plot, the slope, the intercept, the Brier score, and the
   recalibration method.

Outcomes 2, 3, and 5 are computational; 6 and 7 are applied; 8 is a writing outcome.

## 3. The core mental model

Calibration is a relationship between two probabilities: the one the model outputs and the one the
data reveal. The relationship is usually summarised on the logit scale, because that is where a
logistic model is linear.

```text
Predicted risk p  --->  logit(p)  --->  calibration model  --->  observed risk
                                            |
                                            +-- intercept: average over/under-prediction
                                            +-- slope: over/under-dispersion
```

The calibration model is a logistic regression of the outcome on the logit of the predicted risk:

```text
logit(P(Y = 1)) = a + b * logit(p)
```

The intercept `a` is calibration-in-the-large. A value of 0 means the average prediction matches
the average outcome. A positive value means the model under-predicts; a negative value means it
over-predicts. The slope `b` is the calibration slope. A value of 1 means the predictions are
neither too extreme nor too compressed. A value below 1 means the predictions are overdispersed:
high risks are too high and low risks are too low. A value above 1 means the predictions are
underdispersed: the model is too conservative.

The two parameters describe different failures, and a model can have one without the other.

```text
slope = 1, intercept = 0     perfect calibration
slope = 1, intercept > 0     uniform under-prediction
slope = 1, intercept < 0     uniform over-prediction
slope < 1, intercept = 0     overdispersion (extremes too extreme)
slope > 1, intercept = 0     underdispersion (extremes too mild)
slope < 1, intercept < 0     overdispersion plus over-prediction
```

The calibration plot is the visual version. Predicted risk is on the x-axis, observed frequency on
the y-axis, and the identity line is perfect calibration. A curve above the identity means
under-prediction at that risk; below means over-prediction. A curve that is steeper than the
identity means overdispersion; flatter means underdispersion. The plot should include the
distribution of predicted risks, because a calibration curve estimated from few patients at the
extremes is unreliable.

The Brier score is the mean squared difference between the predicted probability and the outcome:

```text
Brier = (1/N) * sum((p_i - y_i)^2)
```

It is a proper scoring rule: it is minimised in expectation by the true probability. Lower is
better. Its decomposition separates reliability (calibration), resolution (discrimination), and
uncertainty (the base rate). A model can have the same Brier score as another and fail differently;
the decomposition, or the plot, is needed to know which.

AUROC is invariant to any monotone transformation of the scores. If you replace every predicted
probability with its square, or with a logistic recalibration, the ranking is unchanged and the
AUROC is unchanged. That is why AUROC cannot detect miscalibration. It is also why a model can
"look fine" on AUROC while its probabilities are wrong by a factor of two.

The calibration model can be fitted on the logit scale or on the probability scale; the logit
scale is preferred because it matches the logistic model and keeps the slope interpretable. The
intercept and slope are estimated with the same software that fits a logistic regression, using the
logit of the predicted risk as the only predictor.

The calibration model is usually fitted by maximum likelihood on a sample that is independent of
the model development. The slope and intercept are estimated with uncertainty, and the uncertainty
is larger when the sample has few events or when the predicted risks are concentrated in a narrow
range. A slope of 0.8 with an interval from 0.6 to 1.0 is not evidence of a problem; a slope of 0.8
with an interval from 0.75 to 0.85 is.

The distinction between weak and moderate calibration is not academic. A model can have a slope of
1.0 and an intercept of 0.0 and still be miscalibrated at the operating threshold if the errors at
low and high risk cancel. The slope and intercept are a summary; the plot is the evidence.

The hierarchy also explains why a single calibration number is not enough. A model can pass a test
of weak calibration (slope and intercept near their ideal values) and still be miscalibrated in a
region that matters, because the errors cancel in the aggregate. The plot, or a measure that
averages absolute deviations such as ICI, is needed to detect that.

A second way to remember the distinction is to imagine two models with the same ranking. Model A
outputs the rank percentile as a "risk", so its top-ranked patients get 0.99 and its bottom-ranked
get 0.01. Model B outputs a calibrated probability, so its top-ranked patients get 0.40 and its
bottom-ranked get 0.02. The AUROC is identical, because the ordering is identical. The first model
is useless for a probability conversation and may be harmful at a threshold; the second is usable.
The difference is entirely calibration.

> **Note:** Calibration is population-specific. A model calibrated in a 30 percent prevalence cohort
> may be miscalibrated in an 8 percent cohort, even if its slope and intercept were perfect in
> development. The calibration assessment must be repeated in the deployment population.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Calibration | Agreement between predicted probabilities and observed frequencies | Confused with discrimination | Property |
| Perfect calibration | Predicted equals observed at every risk | Assumed from a good AUROC | Ideal |
| Calibration-in-the-large | Average predicted versus average observed | Confused with the slope | Log-odds |
| Calibration intercept | The intercept in the recalibration model | Interpreted as a risk difference | Log-odds |
| Calibration slope | The slope in the recalibration model | Interpreted as correlation | Slope |
| Calibration plot | Observed risk versus predicted risk | Read without the risk distribution | Curve |
| Reliability diagram | Synonym for calibration plot | Binned without stating bins | Curve |
| Smooth calibration curve | A smoothed estimate of observed risk | Assumed to be unbiased at the extremes | Curve |
| Loess calibration | Locally weighted smoothing of the calibration curve | Confused with a parametric fit | Curve |
| Brier score | Mean squared error of predicted probabilities | Confused with AUROC | 0 to 1 |
| Brier decomposition | Reliability, resolution, and uncertainty | Ignored | Components |
| Reliability | Calibration component of the Brier score | Confused with the calibration slope | Component |
| Resolution | Discrimination component of the Brier score | Confused with AUROC | Component |
| Uncertainty | Base-rate component of the Brier score | Ignored | Component |
| Log loss | Negative log-likelihood | Confused with Brier | Positive |
| ECE | Expected calibration error; weighted absolute deviation by bin | Sensitive to binning | Proportion |
| MCE | Maximum calibration error | Sensitive to small bins | Proportion |
| ICI | Integrated calibration index; average absolute deviation from a smooth curve | Less bin-dependent than ECE | Proportion |
| Spiegelhalter z | Test of calibration-in-the-large | Confused with a calibration measure | z-score |
| Hosmer-Lemeshow test | Goodness-of-fit test for calibration | Treated as a calibration measure | p-value |
| Calibration drift | Change in calibration over time or site | Confused with case-mix shift | Property |
| Recalibration | Adjusting probabilities to restore calibration | Confused with retraining | Method |
| Platt scaling | Logistic recalibration of scores | Can worsen an already calibrated model | Method |
| Isotonic regression | Monotonic, nonparametric recalibration | Needs enough data | Method |
| Beta calibration | Parametric recalibration with two shape parameters | Less common than Platt | Method |
| Temperature scaling | Single-parameter recalibration for neural networks | A special case of Platt | Method |
| Bayesian binning | Calibration by averaging over binning models | Computationally heavier | Method |
| Recalibration intercept | The intercept added during recalibration | Confused with the original intercept | Log-odds |
| Recalibration slope | The slope applied during recalibration | Confused with the original slope | Slope |
| Held-out calibration set | Data not used to fit the model or the calibrator | Reused for both | Design |
| Nested calibration | Calibration fitted inside cross-validation folds | Confused with fitting on all data | Design |
| Monotone transformation | A transformation that preserves order | Assumed to preserve calibration | Function |
| Rank preservation | The ordering of scores is unchanged | Confused with calibration preservation | Property |
| Overdispersion | Predictions too extreme | Confused with overfitting | Property |
| Underdispersion | Predictions too compressed | Confused with underfitting | Property |
| Prevalence | Proportion of positives | Confused with mean predicted risk | Proportion |
| Mean predicted risk | Average of the model's probabilities | Assumed to equal prevalence | Proportion |
| Calibration by subgroup | Calibration assessed within a group | Ignored in fairness analysis | Property |
| Net benefit | Decision-analytic utility | Confused with calibration | Per patient |
| Decision curve | Net benefit across thresholds | Read without calibration | Curve |
| Calibration hierarchy | Weak, moderate, and strong calibration | Ignored | Framework |
| Weak calibration | Calibration-in-the-large and slope | Confused with strong calibration | Level |
| Moderate calibration | Observed equals predicted on average within risk strata | Confused with weak calibration | Level |
| Strong calibration | Observed equals predicted for every covariate pattern | Rarely achievable | Level |
| Recalibration drift | Change in the recalibration parameters over time | Confused with model drift | Property |
| Updating | Changing the model, not only the calibration | Confused with recalibration | Method |
| Refitting | Re-estimating the model coefficients | Confused with recalibration | Method |
| Model revision | Changing the predictors or the model form | Confused with updating | Method |
| Calibration sample size | Number of events used to estimate calibration | Confused with total sample | Count |
| Calibration by decile | Binned calibration with ten groups | Assumed to be unbiased | Curve |
| Calibration by quantile | Binned calibration with equal counts | Confused with equal-width bins | Curve |
| Smoothing bandwidth | Amount of smoothing in a calibration curve | Chosen without reporting | Parameter |
| Calibration uncertainty band | Interval around the calibration curve | Omitted | Interval |
| Net reclassification | Movement between risk categories | Confused with calibration | Count |
| Risk stratification | Grouping patients by predicted risk | Confused with thresholding | Category |
| Probability of event | The quantity being calibrated | Confused with a score | Probability |
| Score | An uncalibrated numeric output | Called a probability | Arbitrary |
| Calibration map | The function from score to probability | Confused with the model | Function |
| Recalibration curve | Predicted versus recalibrated risk | Confused with the calibration plot | Curve |

Three definitions deserve more than a table row.

**Weak, moderate, and strong calibration.** Van Calster et al. (2016) defined a hierarchy. Weak
calibration means the intercept and slope are adequate; it is what a logistic recalibration model
estimates. Moderate calibration means the observed risk equals the predicted risk on average within
strata of predicted risk; it is what a calibration plot shows. Strong calibration means the observed
risk equals the predicted risk for every combination of covariate values; it is rarely achievable
and usually unnecessary. A model can be weakly calibrated and moderately miscalibrated if the
miscalibration has a nonlinear shape. The plot is needed to detect that.

**Calibration versus discrimination.** Discrimination is a property of the ranking; calibration is a
property of the scale. A monotone transformation preserves discrimination and destroys calibration.
That is why a model can be excellent for triage and useless for shared decision-making. The
decision context determines which property matters more, and most clinical decisions need both.

**Recalibration versus refitting.** Recalibration changes the mapping from the model's score to a
probability, usually with one or two parameters. Refitting re-estimates the model's coefficients on
new data. Recalibration is preferred when the ranking is stable and only the scale has drifted;
refitting is needed when the relationship between features and outcome has changed. The choice
should be based on whether the slope and intercept alone can restore calibration, or whether the
discrimination has also degraded.

> **Note:** A calibration slope below 1 is not always a defect of the model. It can reflect genuine
> heterogeneity in the population that the model does not capture, or overfitting in development.
> The interpretation depends on whether the slope is estimated in the development data, in a
> temporal validation, or at a new site.

## 5. The mechanics, step by step

Calibration assessment has fourteen steps. Steps 1 through 3 define the claim and the sample.
Steps 4 through 7 measure calibration. Steps 8 through 11 repair it. Steps 12 through 14 transport
and monitor it.

### Step 1. Define the probability claim

**Input:** the prediction task specification and the model output.

**Output:** a statement of what the probability is supposed to mean, for whom, and at what horizon.

**Rationale:** A probability claim is specific: "the risk of unplanned readmission within 30 days
of discharge for adults discharged alive with a heart failure diagnosis". A model output that is
used as a score, a rank, or a triage category does not make a probability claim and does not need
the same calibration standard. Defining the claim prevents the team from spending effort on
calibrating a score that will only be used for ranking.

The horizon is part of the claim. A 30-day risk and a 90-day risk are different probabilities, and a
model calibrated for one is not calibrated for the other. The population is part of the claim too:
a model calibrated for adults discharged alive is not necessarily calibrated for patients
transferred to a rehabilitation facility. The claim should be as specific as the intended use.

**Branch point:** if the model output will be shown to a patient or used in a threshold derived
from a harm ratio, calibration is required. If it will only rank patients for a fixed capacity,
calibration is secondary but still useful for monitoring.

### Step 2. Choose the calibration sample

**Input:** the validation design and the available data.

**Output:** a held-out sample that was not used to fit the model or to tune its hyperparameters.

**Rationale:** Calibration estimated on the development data is optimistic, because the model was
fitted to those outcomes. Calibration estimated on a sample that was used for tuning is also
optimistic. The calibration sample must be independent of both. In a nested cross-validation, the
calibration assessment uses the outer-fold predictions, not the inner-fold predictions.

**Branch point:** if no independent sample exists, use bootstrap or cross-validation to estimate
optimism, and state that the calibration estimate is internal. Do not present an internal estimate
as external.

### Step 3. Plot the calibration curve with the risk distribution

**Input:** the predicted probabilities and outcomes in the calibration sample.

**Output:** a calibration plot with a smooth curve, the identity line, and a histogram or rug of
predicted risks.

**Rationale:** The plot shows the shape of the miscalibration, which the slope and intercept may
miss. A model can have a slope near 1 and an intercept near 0 while being miscalibrated in the tails.
The distribution shows where the curve is estimated from enough patients and where it is
extrapolated.

The plot should state the number of patients and the number of events in each bin or region. A
smooth curve fitted to 400 patients with 60 events will be uncertain in the tails, and the
uncertainty should be shown with a band or with the counts. A calibration plot without counts is
easy to over-read.

**Branch point:** if the predicted risks are concentrated in a narrow range, the plot will be
uninformative at the extremes. Report the range and avoid claims about calibration outside it.

### Step 4. Estimate the calibration slope and intercept

**Input:** the predicted probabilities and outcomes.

**Output:** the intercept and slope from a logistic regression of the outcome on the logit of the
predicted risk, with confidence intervals.

**Rationale:** The two parameters are the standard summary of weak calibration. The intercept
locates the average error; the slope locates the dispersion. They are interpretable, comparable
across studies, and easy to report. They do not capture nonlinear miscalibration, which is why the
plot is also required.

The intercept is sometimes called calibration-in-the-large when the slope is fixed at 1. The
two-parameter model estimates both, which is preferable because a slope different from 1 changes
the interpretation of the intercept. Reporting an intercept from a model with a free slope as if it
were calibration-in-the-large is a common error. Report which model was used and whether the
slope was free. The distinction matters for comparing calibration across studies, because a fixed
slope changes the estimate of the intercept, sometimes substantially.
State the model form in the table footnote, not only in the methods.

**Branch point:** if the slope is estimated with few events, its interval will be wide. Report the
interval and avoid a strong claim about over- or under-dispersion.

### Step 5. Compute the Brier score and its components

**Input:** the predicted probabilities and outcomes.

**Output:** the Brier score, and where possible the reliability, resolution, and uncertainty
components.

**Rationale:** The Brier score is a proper scoring rule that combines calibration and
discrimination. It is useful for comparing models and for monitoring, because it is a single number
that responds to both miscalibration and loss of discrimination. The decomposition explains why a
model's Brier score changed: a rise in reliability (worse calibration) is different from a fall in
resolution (worse discrimination).

The Brier score's scale depends on the prevalence: a population with a 50 percent event rate has a
higher maximum Brier score than one with a 5 percent rate. Comparing Brier scores across
populations with different prevalence is therefore misleading unless the score is scaled or the
decomposition is reported. Within the same population, the comparison is valid.

**Branch point:** if the Brier score changes but the AUROC does not, the change is calibration. If
both change, the model's discrimination has also drifted, and recalibration alone may not be enough.

### Step 6. Test calibration-in-the-large

**Input:** the mean predicted risk and the observed prevalence.

**Output:** the difference, a confidence interval, and a test such as the Spiegelhalter z.

**Rationale:** The simplest calibration check is whether the average prediction equals the average
outcome. A model that predicts a mean of 0.24 in a population with 0.18 prevalence is
over-predicting on average, regardless of the slope. The test is a screening check; a
non-significant result does not establish calibration at every risk.

**Branch point:** if the mean prediction differs materially from the prevalence, the intercept will
be nonzero and recalibration will shift the probabilities. Check whether the difference is due to
case mix or to the model.

### Step 7. Interpret the shape and the parameters together

**Input:** the plot, the slope, the intercept, and the Brier score.

**Output:** a written interpretation of the calibration failure, with its direction and its
practical consequence.

**Rationale:** A slope of 0.64 and an intercept of -0.28 describe overdispersion plus
over-prediction, but the practical consequence depends on where the threshold sits. If the
threshold is in a region where the curve crosses the identity, the miscalibration may have little
effect on the decision. If the threshold is in a region where the curve is far from the identity,
the effect can be large.

A useful summary is to compute the recalibrated risk at the operating threshold and at the
threshold implied by the harm ratio. If the two differ materially, the miscalibration changes the
decision. If they are close, the decision is robust even if the model is miscalibrated elsewhere.

**Branch point:** if the calibration curve crosses the identity near the operating threshold, the
decision may be robust to the miscalibration. If it does not, recalibration is required before the
threshold can be trusted.

### Step 8. Decide between recalibration and refitting

**Input:** the calibration results and the discrimination results.

**Output:** a decision, with a rationale.

**Rationale:** Recalibration changes the probability scale; refitting changes the model. If the
AUROC is stable and only the calibration has drifted, recalibration is sufficient and safer. If the
AUROC has also fallen materially, the relationship between features and outcome has changed, and
refitting or model revision is needed. The distinction matters because recalibration preserves the
validated ranking, while refitting requires a new validation.

A practical rule is to compare the change in AUROC with its confidence interval. If the interval
overlaps the development value, the ranking is stable enough for recalibration. If the interval
lies below the development value, the model's discrimination has degraded, and recalibration will
restore the scale but not the ranking. The distinction should be stated explicitly in the report.

**Branch point:** if the AUROC has fallen by more than the uncertainty in the estimate, do not rely
on recalibration alone. Investigate the cause and consider refitting.

### Step 9. Fit the recalibrator on separate data

**Input:** a calibration set that was not used to fit the model.

**Output:** a recalibration function (Platt, isotonic, beta, or temperature scaling) and its
parameters.

**Rationale:** A recalibrator fitted on the same data used to assess calibration will appear to fix
the problem by construction. It must be fitted on one sample and evaluated on another. In a
cross-validation design, the recalibrator is fitted inside each training fold and evaluated on the
held-out fold.

Platt scaling fits a two-parameter logistic model to the score and is stable with modest samples.
Isotonic regression fits a monotone step function and is more flexible but needs more data; with
few events it can produce a step function that is calibrated in-sample and unstable out of sample.
Beta calibration and temperature scaling are alternatives with different flexibility. The choice
should be reported with the sample size and the number of events used to fit it.

**Branch point:** if the recalibration sample is small, prefer a one- or two-parameter method
(Platt, temperature) over isotonic regression, which can overfit with few events.

### Step 10. Validate the recalibration

**Input:** the recalibrator and an independent evaluation sample.

**Output:** the post-recalibration slope, intercept, Brier score, and plot.

**Rationale:** Recalibration can fail if the calibration set is not representative or if the method
is too flexible. The post-recalibration assessment is the evidence that the repair worked. It
should use the same metrics as the pre-recalibration assessment, so the comparison is direct.

**Branch point:** if the post-recalibration slope is still far from 1, the problem is not a simple
scale shift. Consider a more flexible calibrator or a model revision.

### Step 11. Recompute the decision metrics

**Input:** the recalibrated probabilities and the threshold.

**Output:** sensitivity, specificity, PPV, net benefit, and the flagged fraction after
recalibration.

**Rationale:** Recalibration changes which patients are above the threshold, so the operating
characteristics change even though the ranking does not. The decision metrics must be recomputed on
the recalibrated scale, not carried over from the pre-recalibration analysis.

The recomputation should also be done at the deployment prevalence, not only at the validation
prevalence, because recalibration changes the relationship between the threshold and the flagged
fraction in a way that interacts with prevalence. A recalibrated model at a lower prevalence can
flag a similar number of patients while finding fewer true cases.

**Branch point:** if the recalibrated threshold flags more patients than capacity allows, adjust the
threshold and report the new operating point. Recalibration is not a substitute for capacity
planning.

### Step 12. Assess calibration by subgroup

**Input:** the recalibrated probabilities, outcomes, and subgroup identifiers.

**Output:** slope, intercept, and plot by subgroup, with counts.

**Rationale:** A model can be calibrated overall and miscalibrated in a subgroup. That is a fairness
issue as well as a statistical one, because a subgroup whose risk is under-predicted may be
under-served. The subgroup assessment should use the same calibration sample and the same method.

A subgroup can be well calibrated and still disadvantaged if its threshold is set for the overall
population. The calibration analysis answers whether the probabilities are right; the threshold
analysis answers whether the decision is fair. Both are needed, and they are separate questions.

**Branch point:** if a subgroup has fewer than about 50 events, the slope and intercept will be
unstable. Report the plot and the count, and avoid a subgroup-specific recalibration unless there
is a strong rationale.

### Step 13. Monitor calibration after deployment

**Input:** the deployed model, the outcome data, and a monitoring cadence.

**Output:** a monitoring plan with metrics, triggers, and an owner.

**Rationale:** Calibration drifts when the population, the coding, or the treatment changes. The
monitoring should track the mean predicted risk against the observed prevalence, the calibration
slope on a rolling window, and the Brier score. A trigger should specify the magnitude of change
that warrants recalibration, not only statistical significance.

**Branch point:** if the outcome is delayed, use a proxy (for example, the score distribution) in
the interim and schedule the outcome-based check when the data mature. Do not assume stability
because the proxy is stable.

### Step 14. Report the calibration

**Input:** the full analysis.

**Output:** a calibration section with the plot, the slope, the intercept, the Brier score, the
recalibration method, and the post-recalibration results.

**Rationale:** The report should let a reader see the shape and the summary, and should state
whether the calibration was assessed internally or externally. A slope without a plot hides
nonlinear miscalibration; a plot without a slope hides the magnitude. Both are needed.

**Branch point:** if the calibration is poor and cannot be repaired, report that the model should
not be used for probability-based decisions. A ranking use may still be defensible.

### Calibration edge cases

**Good discrimination, poor calibration.** Recalibrate and re-evaluate the decision metrics. Do not
discard the model; the ranking may still be valuable.

**Perfect calibration, poor resolution.** A model that predicts the prevalence for everyone is
perfectly calibrated and useless. Calibration is necessary but not sufficient for a useful model. The Brier decomposition and the AUROC reveal the lack of
resolution.

**Sparse bins at the extremes.** A decile plot with 12 patients in the top decile is unreliable.
Use a smoother, report the count, and avoid claims about the extreme risk range.

**Prevalence shift.** A model calibrated at 18 percent and deployed at 8 percent will over-predict.
Recalibrate to the local prevalence or shift the intercept.

**Calibration set reused for fitting.** The recalibrator will look perfect in-sample. Use a
separate set or cross-fitting.

**A calibration curve fitted with too few bins.** A decile plot with 10 patients per bin is a
scatter of noise. Use fewer bins, a smoother, or a larger sample, and report the bin counts.

**Ties in predicted risk.** Ties can make bin assignment arbitrary. Use a method that handles ties
or report the tie rate.

**Empty folds.** A fold with no events cannot estimate a slope. Report the fold as not evaluable.

**Calibration assessed on the development data.** The slope will be near 1 by construction if the
model was fitted by maximum likelihood. Report the sample and the fit scope; an apparent
calibration of 1.0 is not evidence of calibration.

**A recalibrator fitted on the evaluation data.** The post-recalibration metrics will look perfect
in-sample and may be worse out of sample. Use a separate set or cross-fitting.

**Outcome delay.** A 30-day outcome is not observed for recent patients. Restrict the monitoring
window to patients with complete follow-up.

## 6. Worked example

### The setting

A heart failure readmission model was developed on 30,000 discharges from 2019-2021, with a
prevalence of 22 percent. It was temporally validated on 12,000 discharges from 2022, with a
prevalence of 18 percent, and deployed in 2023. By 2025, the programme notices that the model seems
to be over-alerting. A monitoring sample of 11,400 discharges from 2025 has a prevalence of 14
percent.

### The calibration results

| Period | Prevalence | Mean predicted risk | Slope | Intercept | Brier | AUROC |
|---|---:|---:|---:|---:|---:|---:|
| 2022 validation | 18% | 0.181 | 0.95 | 0.02 | 0.142 | 0.70 |
| 2025 monitoring | 14% | 0.196 | 0.64 | -0.28 | 0.168 | 0.66 |

The 2022 validation is well calibrated: the mean prediction of 0.181 is close to the prevalence of
0.18, the slope is near 1, and the intercept is near 0. The 2025 sample is not: the mean prediction
of 0.196 is well above the prevalence of 0.14, the slope is 0.64, and the intercept is -0.28.

### The arithmetic at the threshold

At a threshold of 0.20, the 2022 operating characteristics are sensitivity 0.60 and specificity
0.80. PPV is:

```text
0.60 * 0.18 / (0.60 * 0.18 + 0.20 * 0.82)
= 0.108 / (0.108 + 0.164)
= 0.397
```

Net benefit is:

```text
0.60 * 0.18 - 0.20 * 0.82 * (0.20 / 0.80)
= 0.108 - 0.164 * 0.25
= 0.067
```

In 2025, at the same threshold, sensitivity is 0.58 and specificity 0.76. PPV is:

```text
0.58 * 0.14 / (0.58 * 0.14 + 0.24 * 0.86)
= 0.0812 / (0.0812 + 0.2064)
= 0.282
```

Net benefit is:

```text
0.58 * 0.14 - 0.24 * 0.86 * 0.25
= 0.0812 - 0.0516
= 0.0296
```

PPV falls from 0.397 to 0.282 and net benefit falls from 0.067 to 0.0296, a 56 percent decline.
Part of the decline is the lower prevalence, and part is the miscalibration.

### The recalibration arithmetic

The calibration model is:

```text
logit(P(Y = 1)) = -0.28 + 0.64 * logit(p)
```

At a nominal predicted risk of 0.20:

```text
logit(0.20) = -1.3863
recalibrated logit = -0.28 + 0.64 * (-1.3863) = -1.1672
recalibrated risk = 1 / (1 + exp(1.1672)) = 0.237
```

A patient the model puts at 20 percent is actually at about 24 percent. The model is
under-predicting at the threshold, not over-predicting, even though its mean prediction is above the
prevalence. The pattern is overdispersion: low risks are too low and high risks are too high, and
the intercept shifts the whole curve down.

At a nominal risk of 0.10:

```text
logit(0.10) = -2.1972
recalibrated logit = -0.28 - 1.4062 = -1.6862
recalibrated risk = 0.156
```

A patient the model puts at 10 percent is actually at about 16 percent. The under-prediction at the
low end is larger than at the threshold. That is why the slope matters: the model compresses the
risk range, and the compression is not visible in the mean.

After recalibration, the Brier score falls from 0.168 to 0.151, and the calibration slope on a
held-out sample returns to 0.97 with an intercept of 0.01. The AUROC is unchanged at 0.66, because
recalibration is a monotone transformation.

### The decision after recalibration

Recalibration changes which patients are above 0.20. The recalibrated 0.20 threshold corresponds to
a pre-recalibration risk of about 0.151:

```text
solve -0.28 + 0.64 * logit(p) = logit(0.20)
logit(p) = (-1.3863 + 0.28) / 0.64 = -1.7286
p = 1 / (1 + exp(1.7286)) = 0.151
```

At the pre-recalibration risk of 0.151, the operating characteristics are sensitivity 0.68 and
specificity 0.72. Net benefit at 14 percent prevalence is:

```text
0.68 * 0.14 - 0.28 * 0.86 * 0.25
= 0.0952 - 0.0602
= 0.0350
```

The recalibrated model has a net benefit of 0.0350, against 0.0296 before recalibration. The
improvement is 0.0054 per patient, or 5.4 net true positives per 1,000. That is modest but real. The
larger effect is on the probability statements: a patient told 20 percent is now closer to 20
percent than to 24 percent.

### Two diagnostics that would reveal a broken calibration analysis

**Diagnostic 1: the calibration plot by year.** Plotting observed against predicted for 2022 and
2025 on the same axes shows the curve rotating and shifting. The 2022 curve is close to the identity;
the 2025 curve is flatter and below the identity at low risks. A report that gives only the Brier
score would show a rise from 0.142 to 0.168 but would not show the shape or the direction. The plot
is the diagnostic.

**Diagnostic 2: the subgroup calibration.** The 2025 slope for patients aged 80 and over is 0.51
with an intercept of -0.41, against 0.68 and -0.24 for younger patients. The older subgroup is more
severely over-predicted. A report that gives only the overall slope of 0.64 would miss that the
model is substantially worse for the group with the highest event rate. The diagnostic is a
subgroup table with counts.

### Varying one assumption: prevalence falls to 8 percent

Suppose the 2025 prevalence is 8 percent rather than 14 percent, with the same sensitivity and
specificity at the threshold. PPV becomes:

```text
0.58 * 0.08 / (0.58 * 0.08 + 0.24 * 0.92)
= 0.0464 / (0.0464 + 0.2208)
= 0.174
```

Net benefit becomes:

```text
0.58 * 0.08 - 0.24 * 0.92 * 0.25
= 0.0464 - 0.0552
= -0.0088
```

Net benefit is negative. At 8 percent prevalence, the model at this threshold harms more than it
helps, even after recalibration. Recalibration fixes the probability scale; it does not fix a
prevalence at which the decision itself is unfavorable. The conclusion changes from "recalibrate and
continue" to "reconsider the threshold or the intervention before deploying at this prevalence".

### What to report

The primary figure is the calibration plot at the deployment site, with the identity line, a smooth
curve, and the distribution of predicted risks. The primary table is the slope and intercept before
and after recalibration, with the Brier score. The decision analysis is reported before and after
recalibration at the deployment prevalence. The report should state whether the calibration was
internal or external, and which sample was used to fit the recalibrator.

## 7. Code, query, or calculation

The script generates a cohort with a known miscalibration, estimates the calibration slope and
intercept by regressing the outcome on the logit of the predicted risk, computes the Brier score
and the expected calibration error, and shows the effect of recalibration. It uses only the
standard library.

```python
"""Calibration slope, intercept, Brier score, and ECE. Standard library only."""
import math
import random


def fit_calibration(p, y, iterations=800, learning_rate=0.5):
    """Logistic regression of y on logit(p); returns intercept and slope."""
    x = [math.log(v / (1 - v)) for v in p]
    a = b = 0.0
    n = len(y)
    for _ in range(iterations):
        ga = gb = 0.0
        for xi, yi in zip(x, y):
            q = 1 / (1 + math.exp(-max(-30, min(30, a + b * xi))))
            error = q - yi
            ga += error
            gb += error * xi
        a -= learning_rate * ga / n
        b -= learning_rate * gb / n
    return a, b


def brier(p, y):
    return sum((pi - yi) ** 2 for pi, yi in zip(p, y)) / len(y)


def ece(p, y, bins=10):
    buckets = [[] for _ in range(bins)]
    for pi, yi in zip(p, y):
        buckets[min(bins - 1, int(pi * bins))].append((pi, yi))
    total = 0.0
    for bucket in buckets:
        if bucket:
            mean_p = sum(pi for pi, _ in bucket) / len(bucket)
            mean_y = sum(yi for _, yi in bucket) / len(bucket)
            total += len(bucket) / len(y) * abs(mean_p - mean_y)
    return total


rng = random.Random(4)
p, y = [], []
for _ in range(20000):
    v = rng.betavariate(2.2, 5.0) * 0.9 + 0.02
    q = 1 / (1 + math.exp(-(-0.28 + 0.64 * math.log(v / (1 - v)))))
    p.append(v)
    y.append(1 if rng.random() < q else 0)

a, b = fit_calibration(p, y)
print(f"prev={sum(y) / len(y):.3f} mean={sum(p) / len(p):.3f} "
      f"intercept={a:.3f} slope={b:.3f} Brier={brier(p, y):.4f} ECE={ece(p, y):.4f}")

calibrated = [1 / (1 + math.exp(-(a + b * math.log(v / (1 - v))))) for v in p]
print(f"recal mean={sum(calibrated) / len(calibrated):.3f} "
      f"Brier={brier(calibrated, y):.4f} ECE={ece(calibrated, y):.4f}")
```

Expected output:

```text
prev=0.303 mean=0.296 intercept=-0.246 slope=0.643 Brier=0.2034 ECE=0.0374
recal mean=0.303 Brier=0.2013 ECE=0.0036
```

Reading the output:

- The estimated intercept of -0.246 and slope of 0.643 are close to the generating values of -0.28
  and 0.64. The small differences are sampling and optimisation error.
- The mean predicted risk of 0.296 is slightly below the observed prevalence of 0.303, which is
  consistent with the small negative intercept. The model is not badly miscalibrated on average,
  but its slope is far from 1, so it is overdispersed.
- The Brier score of 0.2034 is high because the prevalence is 30 percent and the model is not
  highly discriminating. The ECE of 0.0374 is the average absolute deviation across bins.
- After recalibration, the mean is 0.303, the Brier score falls to 0.2013, and the ECE falls to
  0.0036. The recalibration improved calibration without changing the ranking.
- The Brier improvement is modest because the model's discrimination is modest. Recalibration can
  fix the scale but cannot create resolution.

The function `fit_calibration` is a two-parameter logistic regression and should be fitted on a
held-out sample. In this script it is fitted on the same sample used to generate the data, which is
acceptable for demonstrating the arithmetic but not for a real calibration assessment. In a real
analysis, split the sample or use cross-fitting.

## 8. Data, standards, and regulatory context

Calibration is addressed by reporting guidelines, appraisal tools, and regulatory expectations.
The table distinguishes binding requirements from conventions.

| Instrument | Status | What it says about calibration | What it does not do |
|---|---|---|---|
| TRIPOD+AI | Reporting convention | Requires calibration to be reported for prediction models | Does not prescribe a measure |
| PROBAST+AI | Appraisal tool | Assesses whether calibration was evaluated | Does not compute it |
| FDA GMLP principles | Guidance, not binding | Calls for performance testing representative of the intended use | Does not specify calibration metrics |
| FDA PCCP guidance | Guidance, not binding | Requires performance criteria for planned updates | Does not define recalibration triggers |
| EU AI Act | Binding in the EU, phased | Requires technical documentation and testing for high-risk systems | Does not specify calibration |
| EU MDR | Binding in the EU | Requires clinical evaluation and performance evaluation | Does not specify metrics |
| ISO 14971 | Voluntary standard | Risk management across the lifecycle | Does not address calibration |
| NIST AI RMF | Voluntary framework | Measure and manage performance, including drift | Not sector-specific |

Three points matter for practice.

**TRIPOD+AI expects calibration, not only discrimination.** The checklist asks for the model's
performance to be reported in a way that reflects its intended use. For a model that outputs a
probability, that includes calibration. A report with AUROC and no calibration has not described
the probability scale, and a reviewer will ask for it.

**PROBAST+AI treats calibration as part of the analysis domain.** The tool asks whether the
performance evaluation included calibration and whether the calibration was assessed on appropriate
data. A study that reports calibration on the development data, or that reports only the
Hosmer-Lemeshow p-value, is at risk of bias or at least of incomplete reporting.

**Regulators expect calibration to be maintained.** A device whose output is a probability must
remain calibrated after deployment. The FDA's PCCP guidance allows pre-specified updates, including
recalibration, provided the protocol and acceptance criteria are described. The EU AI Act's
post-market obligations similarly require monitoring. A calibration monitoring plan is therefore
part of the regulatory lifecycle, not only a statistical nicety.

### A calibration reporting checklist

A calibration section that can be checked contains the following.

```text
probability claim: outcome, horizon, population
calibration sample: source, dates, size, events
fit scope: whether the sample was used for model fitting or tuning
plot: smooth curve, identity line, risk distribution, counts
slope and intercept with intervals
Brier score and, where possible, its decomposition
ECE or ICI with the binning or smoothing rule
recalibration method and fitting sample
post-recalibration slope, intercept, and Brier score
decision metrics before and after recalibration
subgroup calibration with counts
monitoring plan and triggers
```

The checklist is a convention, but it maps onto the TRIPOD+AI performance items and the PROBAST+AI
analysis domain. A section that omits the fit scope or the post-recalibration validation is not
reproducible.

**Calibration and the intended-use population.** A device approved for adults may be used off-label
in adolescents, where the calibration is unknown. The calibration claim is bounded by the
population in which it was assessed, and the labelling should state that bound. A model that is
well calibrated in the trial population is not thereby calibrated in every population that shares
the diagnosis.

**The minimum defensible set.** A plot, a slope, an intercept, and a Brier score are the minimum.
The plot shows the shape, the slope and intercept summarise weak calibration, and the Brier score
provides a proper overall measure that can be monitored. Additional measures (ECE, ICI,
Spiegelhalter z, Hosmer-Lemeshow) are useful but not substitutes for the minimum set.

> **Note:** The choice of calibration measure is a scientific judgement, but the requirement to
> assess calibration is a reporting and regulatory expectation. A plot plus slope and intercept is
> the minimum defensible set.

## 9. Industry practice and case evidence

**Calibration as the Achilles heel.** Van Calster et al. (2019) argue that calibration is
under-reported and under-appreciated, that discrimination alone is insufficient for clinical use,
and that calibration should be assessed with a plot and with slope and intercept. They note that
many models with good discrimination are miscalibrated, and that the miscalibration can reverse a
clinical recommendation.

**Miscalibration harms net benefit.** Van Calster and Vickers (2015) show that miscalibration
always reduces net benefit and can make a model clinically harmful. The result is important because
it links a statistical property to a decision consequence: a model that over-predicts risk will
over-treat at a threshold, and the over-treatment is a harm even if the ranking is good.

**The calibration hierarchy.** Van Calster et al. (2016) defined weak, moderate, and strong
calibration. Weak calibration is the intercept and slope; moderate calibration is the agreement
within risk strata; strong calibration is agreement for every covariate pattern. The hierarchy
clarifies what a slope and intercept can and cannot establish. A model can be weakly calibrated and
moderately miscalibrated if the error is nonlinear.

**Practical measurement guidance.** Huang et al. (2020) provide a tutorial on calibration
measurements (Brier score, Spiegelhalter z, Hosmer-Lemeshow, reliability diagrams, ECE, MCE, ICI)
and calibration models (Platt scaling, isotonic regression, Bayesian binning). They caution that
Platt scaling can worsen an already calibrated model and that the choice of measure depends on the
sample size and the shape of the miscalibration.

**Calibration drift.** Davis et al. (2021) propose dynamic calibration curves and a drift detector
for deployed clinical prediction models, and argue for drift-triggered updating rather than fixed
schedules. Tanner et al. (2024) describe proactive and reactive updating pipelines with
noninferiority criteria, and Feng et al. (2022) propose Bayesian online recalibration with
performance guarantees. The common finding is that recalibration is usually preferred to full
refitting when only the calibration has drifted.

**A note on the evidence quality.** Much of the calibration literature is methodological rather than
empirical: it proposes measures and demonstrates them on simulated or single-cohort data. That is
appropriate for a measurement problem, but it means the evidence for "calibration monitoring
improves outcomes" is thinner than the evidence for "miscalibration harms net benefit", which is
analytic. The report should distinguish the two kinds of claim.

**The practical pattern.** In deployed clinical models, the most common drift is a shift in
prevalence or case mix that changes the intercept, followed by a change in the slope as the
population becomes more or less heterogeneous. The ranking often survives longer than the scale.
That is why monitoring calibration is a leading indicator of model degradation, and why a
calibration check can detect a problem before the AUROC moves.

**What the drift evidence means for a programme.** The methods literature converges on three
practices. First, monitor calibration rather than only discrimination, because the scale usually
drifts before the ranking. Second, prefer recalibration to refitting when the ranking is stable,
because recalibration preserves the validated ordering and requires less new evidence. Third, use
triggers rather than fixed schedules, because a model can remain stable for years and then shift
quickly after a guideline change or a new site goes live.

**A note on the choice of recalibration set.** The recalibration set should be large enough to
estimate two parameters with reasonable precision and representative of the deployment population.
A small recalibration set can produce a correction that is worse than the original miscalibration.
The set should be distinct from the model-fitting and evaluation sets, and its size and event count
should be reported.

**A note on recalibration frequency.** There is no universal schedule. A model in a stable inpatient
population may need recalibration every few years; a model in a quickly changing outpatient
population may need it annually. The trigger should be based on observed drift and on the outcome
window, not on a calendar. A fixed annual recalibration can be too late for a fast drift and
unnecessary for a stable one.

**The organisational side.** A calibration monitoring plan requires an owner, a data feed with
matured outcomes, and a decision rule for when to recalibrate. Without those, the statistical method
has no effect. The most common failure is not the absence of a method; it is the absence of a person
who looks at the calibration plot after launch.

> **Note:** The cases are documented. The claim that a specific deployed model failed because of
> calibration drift is not made here; the cases describe methods and mechanisms.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| AUROC only | No calibration plot | Discrimination confused with calibration | Wrong probabilities | Plot and slope |
| Calibration on development data | Slope near 1 by construction | In-sample fit | Optimistic calibration | Held-out sample |
| Recalibrator fitted on evaluation data | Post-recalibration metrics perfect | In-sample calibration | Worse out of sample | Separate set or cross-fitting |
| Slope interpreted as correlation | Slope reported without intercept | Misunderstanding of the model | Wrong correction | Report both |
| Brier score without decomposition | Single number only | Calibration and discrimination conflated | Wrong diagnosis | Decompose or plot |
| Hosmer-Lemeshow as the only measure | p-value reported | Test depends on binning and sample size | False reassurance | Plot and slope |
| ECE with sparse bins | ECE changes with bin count | Binning sensitivity | Unstable summary | Report bin counts or ICI |
| No risk distribution | Calibration curve without histogram | Extrapolation at the extremes | Over-reading the tails | Add distribution |
| Recalibration without re-evaluation | Threshold metrics unchanged | Operating point shifted | Wrong decision | Recompute metrics |
| Prevalence shift ignored | Mean prediction drifts from prevalence | Base rate changed | Over- or under-treatment | Recalibrate to local prevalence |
| Subgroup calibration omitted | Overall slope only | Heterogeneity hidden | Inequity | Subgroup plots |
| Drift monitoring absent | No post-deployment check | Silent degradation | Harm after months | Monitoring plan |
| Calibration claim without a horizon | Probability claim vague | Different horizons | Wrong interpretation | State the horizon |
| Recalibration mistaken for refitting | AUROC decline ignored | Ranking degraded | Recalibration cannot fix | Check AUROC first |
| Calibration curve without uncertainty | A smooth line with no band | Small-sample noise | Over-reading the shape | Add a band or counts |
| Bin count chosen after results | ECE changes with bins | Binning sensitivity | Flattering summary | Pre-specify the rule |
| Mean prediction vs prevalence ignored | Intercept not checked | Average miscalibration hidden | Wrong scale | Report both |
| Recalibration applied without versioning | The calibrator is not in the model artefact | Deployment mismatch | Wrong probabilities | Version the calibrator |
| Calibration assessed only overall | No subgroup table | Heterogeneity hidden | Inequity | Subgroup plots |
| Drift trigger undefined | No threshold for action | Monitoring without response | Silent harm | Define quantitative triggers |
| Calibration claim outside the validated range | No stated range | Extrapolation | Wrong probabilities | Bound the claim |
| Recalibration without a version | Artefact mismatch | Deployed model differs | Wrong output | Version the calibrator |
| Calibration curve read outside the data | No counts shown | Extrapolation | Over-reading | Mark the supported range |
| Recalibration method not reported | Only post-recalibration numbers | Irreproducible | Cannot audit | Report method and sample |

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The calibration sample is independent | Calibration estimate is honest | In-sample fitting | Audit fit scope |
| The sample represents deployment | Calibration transports | Case-mix or site shift | Compare characteristics |
| The outcome is ascertained consistently | Observed risk is comparable | Label drift | Label validation |
| The probability claim is specific | Calibration is defined | Vague horizon or population | State the claim |
| The slope and intercept capture the error | Weak calibration is adequate | Nonlinear miscalibration | Plot |
| The recalibrator is fitted on enough events | Stable correction | Few events | Report sample size |
| The recalibration is validated | Correction works out of sample | In-sample evaluation | Independent check |
| The threshold is on the recalibrated scale | Decision is valid | Threshold carried over | Recompute |
| The monitoring outcome is available | Drift can be detected | Delayed outcomes | Proxy plus scheduled check |
| Subgroup sizes are adequate | Subgroup calibration is stable | Few events per group | Report counts |
| The recalibrator is part of the deployed artefact | Deployment matches validation | Calibrator omitted | Wrong probabilities | Version the full pipeline |
| The monitoring outcome matures | Drift can be measured | Delayed outcomes | Late detection | Proxy plus scheduled check |
| The calibration measure matches the sample size | Stable estimate | ECE with sparse bins | Misleading summary | Use ICI or fewer bins |
| The calibration claim is bounded by the validated range | No extrapolation | Claims outside the data | Wrong probabilities at extremes | State the range |
| The recalibration sample is representative | Correction transports | Selected sample | Wrong correction | Compare characteristics |
| The calibration plot covers the operating range | The threshold is supported | Plot ends below the threshold | Unsupported claim | Mark the threshold |
| The monitoring window has mature outcomes | Drift estimate is valid | Immature outcomes | False stability | Restrict the window |

The deepest limit is that calibration is a property of a model in a population at a time. It is not
a fixed attribute that can be certified once. A model that is calibrated today may be miscalibrated
next year, and the only defence is monitoring. The calibration report should therefore include a
plan for rechecking, not only a result. The plan should name the metric, the window, the trigger,
and the action. A plan without an action is a monitoring intention, not a monitoring system. A plan without an
owner is not even that. Name the owner and the escalation path in the calibration report.

A second limit is that calibration cannot be assessed without outcomes. In a deployment where the
outcome is rare or delayed, the calibration check lags the model's behaviour by the outcome window.
The programme may be operating with a miscalibrated model for months before the data are available
to detect it. That lag is a reason to monitor leading indicators, such as the score distribution and
the alert volume, even though they do not measure calibration directly.

## 12. Exercises

The exercises move from arithmetic to reporting. Each asks for the calibration sample, the number
of events, and the fit scope to be stated, because a calibration number without that context is not interpretable. Exercises 1 and 2 can be done by hand; 3 through 5
require predictions and outcomes; 6 and 7 require a dataset with dates or subgroups; 8 produces the
calibration section for the module assignment.

**Exercise 1. Calibration arithmetic.** A model has slope 0.70 and intercept -0.20. Compute the
recalibrated risk at predicted risks of 0.10, 0.30, and 0.50. Deliverable: three values and a
sentence on the direction. Acceptance criteria: values correct to three decimals; direction stated.
Time: 30 minutes.

**Exercise 2. Brier score.** For five patients with predicted risks 0.1, 0.2, 0.6, 0.8, 0.9 and
outcomes 0, 0, 1, 1, 1, compute the Brier score. Deliverable: the arithmetic. Acceptance criteria:
value correct; interpretation states that lower is better. Time: 20 minutes.

**Exercise 3. Calibration plot.** Using a dataset with predictions and outcomes, plot the
calibration curve by decile with the risk distribution. Deliverable: the plot and a 300-word
interpretation. Acceptance criteria: identity line; bin counts; interpretation of shape. Time: 60
minutes.

**Exercise 4. Recalibration.** Split a dataset into model-fitting, calibration, and evaluation
sets. Fit a Platt calibrator and report pre- and post-recalibration slope, intercept, and Brier
score on the evaluation set. Deliverable: a table and a plot. Acceptance criteria: three-way split;
post-recalibration metrics on held-out data. Time: 75 minutes.

**Exercise 5. Decision impact.** For a model with and without recalibration, compute net benefit at
the operating threshold and the deployment prevalence. Deliverable: a table and a decision curve.
Acceptance criteria: both models evaluated on the same sample; the difference interpreted. Time: 60
minutes.

**Exercise 6. Drift diagnosis.** Given calibration results from two years, distinguish a prevalence
shift from a slope change and propose a response. Deliverable: a 400-word memo. Acceptance criteria:
intercept and slope separated; response matches the diagnosis. Time: 45 minutes.

**Exercise 7. Subgroup calibration.** Compute slope and intercept by subgroup for a model you have
built, with counts. Deliverable: a table and a plot. Acceptance criteria: at least three subgroups;
counts reported; no claim from groups with fewer than 50 events. Time: 60 minutes.

**Exercise 8. Calibration section.** Write the calibration section of a model development report,
including the plot, the slope and intercept, the Brier score, the recalibration method, the
post-recalibration validation, and the monitoring plan. Deliverable: two pages. Acceptance criteria:
all six elements present; fit scope stated; no claim beyond the supported risk range. Time: 60
minutes. Write the calibration section of a model development report.
Deliverable: two pages. Acceptance criteria: plot, slope, intercept, Brier, recalibration method,
and monitoring plan all present. Time: 60 minutes.

## 13. Self-check answers

**Answer to Exercise 1.** At 0.10: `logit = -2.1972`; recalibrated logit = `-0.20 + 0.70 *
(-2.1972) = -1.7380`; risk = `1 / (1 + exp(1.7380)) = 0.1495`. At 0.30: `logit = -0.8473`;
recalibrated logit = `-0.20 - 0.5931 = -0.7931`; risk = `0.3115`. At 0.50: `logit = 0`; recalibrated
logit = `-0.20`; risk = `0.4502`. The model under-predicts at low risks and slightly under-predicts
at 0.30, while over-predicting at 0.50. The pattern is overdispersion plus a downward shift.

**Answer to Exercise 2.** Squared errors: `(0.1-0)^2 = 0.01`, `(0.2-0)^2 = 0.04`, `(0.6-1)^2 =
0.16`, `(0.8-1)^2 = 0.04`, `(0.9-1)^2 = 0.01`. Sum = 0.26; mean = 0.052. The Brier score is 0.052.

**Answer to Exercise 5 (method).** Compute sensitivity and specificity at the threshold for both
the original and recalibrated probabilities, then apply the net benefit formula at the deployment
prevalence. Recalibration preserves the ranking but changes which patients are above the threshold,
so the operating characteristics must be recomputed. The comparison should use the same evaluation
sample and the same threshold.

**Answer to Exercise 2 (interpretation).** The predicted risks are 0.1, 0.2, 0.6, 0.8, 0.9 and the
outcomes are 0, 0, 1, 1, 1, so the model is reasonably calibrated on this tiny sample: the average
prediction is 0.52 and the observed frequency is 0.60. The Brier score of 0.052 is lower than the
score of a model that predicts the prevalence (0.48) for everyone, which is 0.24, so the model has
resolution as well as calibration.

**Answer to Exercise 3 (what a strong plot shows).** A strong plot has predicted risk on the x-axis,
observed risk on the y-axis, the identity line, a smooth curve, and a histogram or rug of predicted
risks along the bottom or the top. It states the number of patients and events, the smoothing method,
and the range over which the curve is supported. It interprets the shape in words: "the curve is
below the identity above 0.3 and above the identity below 0.1, which indicates overdispersion".

**Answer to Exercise 1 (interpretation).** The model under-predicts at 0.10 and 0.30 and slightly
over-predicts at 0.50. The pattern is not a uniform shift; it is overdispersion plus a downward
intercept. A recalibration with these parameters would raise low risks, leave the middle close to
unchanged, and lower the highest risks. That is why the slope and intercept must be interpreted
together.

**Answer to Exercise 5 (what a strong decision comparison shows).** A strong comparison reports net
benefit for the original and recalibrated models at the same threshold and the same deployment
prevalence, with the operating characteristics (sensitivity, specificity, flagged fraction) for
each. It shows the decision curve for both, and it states whether the recalibration changed the
ranking (it should not) and whether the change in net benefit is larger than the uncertainty. It
does not compare a recalibrated model at one threshold with an original model at another.

**Answer to Exercise 7 (what a strong subgroup table shows).** A strong table reports the subgroup
size, the number of events, the slope, the intercept, and the Brier score, with intervals where the
counts allow. It shows the calibration plot for each subgroup or for the subgroups with enough
events, and it states which subgroups were pooled because of small counts. It avoids a
subgroup-specific recalibration unless the subgroup is large enough to support one.

**Answer to Exercise 4 (what a strong recalibration report shows).** A strong report states the
three-way split, the size and event count of each part, the recalibration method, and the
pre-recalibration and post-recalibration slope, intercept, and Brier score on the evaluation set. It
shows the calibration plot before and after, and it reports whether the AUROC changed (it should
not, for a monotone method). It also states the risk range over which the recalibration was
estimated.

**Answer to Exercise 6 (a strong drift memo).** A strong memo separates the intercept from the
slope. A change in the intercept with a stable slope suggests a prevalence or case-mix shift; a
change in the slope suggests a change in the strength or form of the relationship. It proposes
recalibration for an intercept shift and refitting or model revision for a slope change, and it
states the evidence needed to distinguish them. It also names the monitoring trigger and the owner.

**Rubric for Exercises 3, 4, 6, 7, and 8.** A strong submission (1) uses a held-out sample, (2)
plots the curve with the distribution, (3) reports slope and intercept with intervals, (4) reports
the Brier score, (5) validates any recalibration out of sample, and (6) states a monitoring plan. A
competent submission meets four of the six. A weak submission reports a Hosmer-Lemeshow p-value
alone or fits the recalibrator on the evaluation data.

## 14. Interview questions

**1. What is calibration?** A strong answer defines agreement between predicted and observed risk
and distinguishes it from discrimination. A weak answer says calibration is "accuracy".

**2. Why can AUROC not detect miscalibration?** A strong answer explains monotone invariance. A
weak answer says AUROC measures calibration.

**3. What does a slope of 0.7 mean?** A strong answer explains overdispersion and gives the
recalibration arithmetic. A weak answer says the model is "not calibrated" without direction.

**4. What does a negative intercept mean?** A strong answer says over-prediction on average. A weak
answer confuses it with a low AUROC.

**5. How do you recalibrate a model?** A strong answer describes Platt or isotonic on held-out
data, with validation. A weak answer says "retrain the model".

**6. When would you refit instead of recalibrate?** A strong answer says when discrimination has
also degraded. A weak answer says always refit.

**7. How do you monitor calibration after deployment?** A strong answer tracks mean prediction
versus prevalence, slope on a rolling window, and Brier score, with triggers. A weak answer says
"check the AUROC".

**8. What is the Brier score?** A strong answer defines it as mean squared error of probabilities
and notes it is a proper scoring rule. A weak answer confuses it with accuracy.

**9. Why is calibration population-specific?** A strong answer explains prevalence and case-mix
dependence. A weak answer says calibration is a model property that travels.

**10. What is the difference between ECE and ICI?** A strong answer explains that ECE depends on binning while ICI averages the absolute deviation from a smooth curve, and that both are summaries rather than substitutes for the plot. A weak answer treats them as interchangeable.

**11. Why is a calibration slope of 1.2 possible?** A strong answer explains underdispersion, where predictions are too compressed, and notes that it can occur with over-regularised models. A weak answer says a slope above 1 is impossible.

**12. What would you monitor if outcomes are delayed by 90 days?** A strong answer uses score distribution and process proxies in the interim, with a scheduled outcome-based check. A weak answer says monitoring is impossible.

**13. What is the difference between recalibration and updating a model?** A strong answer says recalibration changes the probability map, updating changes the model or its coefficients, and revision changes the predictors or form. A weak answer uses the terms interchangeably.

**14. When is a calibration slope above 1 a concern?** A strong answer says underdispersion can indicate over-regularisation or a compressed risk range, and that the practical concern depends on the threshold. A weak answer says a slope above 1 is always fine.

**15. What is the role of the Brier score in monitoring?** A strong answer says it is a proper score that responds to both calibration and discrimination, so a rise can signal either, and the decomposition or the plot is needed to tell which. A weak answer treats it as a calibration-only measure.

**16. What is the difference between a calibration curve and a decision curve?** A strong answer says the calibration curve compares predicted and observed risk, while the decision curve compares net benefit across thresholds. A weak answer treats them as two names for the same plot.

**17. How would you report calibration to a clinical committee?** A strong answer uses a plot with
the identity line and a plain-language explanation of over- or under-prediction at the threshold. A
weak answer gives a p-value.

## 15. Key terms

- **Brier score** — mean squared error of predicted probabilities.
- **Brier decomposition** — reliability, resolution, and uncertainty.
- **Calibration** — agreement between predicted and observed risk.
- **Calibration drift** — change in calibration over time or site.
- **Calibration hierarchy** — weak, moderate, and strong calibration.
- **Calibration-in-the-large** — average predicted versus observed risk.
- **Calibration intercept** — intercept of the recalibration model.
- **Calibration plot** — observed versus predicted risk.
- **Calibration slope** — slope of the recalibration model.
- **Decision curve** — net benefit across thresholds.
- **ECE** — expected calibration error.
- **Held-out calibration set** — data not used to fit the model.
- **Hosmer-Lemeshow test** — goodness-of-fit test for calibration.
- **ICI** — integrated calibration index.
- **Isotonic regression** — monotone nonparametric recalibration.
- **Log loss** — negative log-likelihood.
- **MCE** — maximum calibration error.
- **Mean predicted risk** — average of the model's probabilities.
- **Moderate calibration** — agreement within risk strata.
- **Monotone transformation** — a transformation that preserves order.
- **Net benefit** — decision-analytic utility.
- **Overdispersion** — predictions too extreme.
- **Perfect calibration** — predicted equals observed at every risk.
- **Platt scaling** — logistic recalibration.
- **Prevalence** — proportion of positives.
- **Recalibration** — adjusting the probability scale.
- **Refitting** — re-estimating the model coefficients.
- **Reliability diagram** — synonym for calibration plot.
- **Resolution** — discrimination component of the Brier score.
- **Spiegelhalter z** — test of calibration-in-the-large.
- **Strong calibration** — agreement for every covariate pattern.
- **Temperature scaling** — one-parameter recalibration.
- **Underdispersion** — predictions too compressed.
- **Uncertainty component** — base-rate component of the Brier score.
- **Weak calibration** — intercept and slope adequate.

## 16. Sources

All URLs accessed 25 October 2026.

1. Van Calster B, McLernon DJ, van Smeden M, Wynants L, Steyerberg EW. Calibration: the Achilles
   heel of predictive analytics. *BMC Medicine* 2019;17:230.
   https://doi.org/10.1186/s12916-019-1466-7
2. Van Calster B, Nieboer D, Vergouwe Y, De Cock B, Pencina MJ, Steyerberg EW. A calibration
   hierarchy for risk models was defined: from utopia to empirical data. *Journal of Clinical
   Epidemiology* 2016;74:167-176. https://doi.org/10.1016/j.jclinepi.2015.12.005
3. Van Calster B, Vickers AJ. Calibration of risk prediction models: impact on decision-analytic
   performance. *Medical Decision Making* 2015;35(2):162-169.
   https://doi.org/10.1177/0272989X14547233
4. Huang Y, Li W, Macheret F, Gabriel RA, Ohno-Machado L. A tutorial on calibration measurements
   and calibration models for clinical prediction models. *Journal of the American Medical
   Informatics Association* 2020;27(4):621-633. https://doi.org/10.1093/jamia/ocz228
5. Davis SE, Greevy RA, Lasko TA, Walsh CG, Matheny ME. Detection of calibration drift in clinical
   prediction models to inform model updating. *Journal of Biomedical Informatics*
   2021. https://pmc.ncbi.nlm.nih.gov/articles/PMC8627243/
6. Tanner KT, Diaz-Ordaz K, Keogh RH. Dynamic model updating pipeline for clinical prediction
   models. *Journal of Clinical Epidemiology* 2024. https://www.jclinepi.com/article/S0895-4356(24)00287-7/fulltext
7. Feng J, Gossmann A, Sahiner B, Pirracchio R. Bayesian logistic regression for online
   recalibration and revision of risk prediction models with performance guarantees. *Journal of
   the American Medical Informatics Association* 2022.
   https://pmc.ncbi.nlm.nih.gov/articles/PMC9006691/
8. Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement. *BMJ* 2024;385:e078378.
   https://doi.org/10.1136/bmj-2023-078378
9. Moons KGM, Damen JAA, Kaul T, et al. PROBAST+AI. *BMJ* 2025;388:e082505.
   https://doi.org/10.1136/bmj-2024-082505
10. Vickers AJ, Elkin EB. Decision curve analysis. *Medical Decision Making* 2006;26(6):565-574.
    https://doi.org/10.1177/0272989X06295361
11. US Food and Drug Administration. Good Machine Learning Practice for Medical Device Development:
    Guiding Principles. 2021.
    https://www.fda.gov/medical-devices/artificial-intelligence-enabled-medical-devices/good-machine-learning-practice-medical-device-development-guiding-principles
12. US Food and Drug Administration. Marketing Submission Recommendations for a Predetermined Change
    Control Plan for Artificial Intelligence-Enabled Device Software Functions. 2024.
    https://www.fda.gov/regulatory-information/search-fda-guidance-documents/marketing-submission-recommendations-predetermined-change-control-plan-artificial-intelligence
13. European Union. Regulation (EU) 2024/1689 (Artificial Intelligence Act). 2024.
    https://eur-lex.europa.eu/eli/reg/2024/1689/oj
14. National Institute of Standards and Technology. Artificial Intelligence Risk Management
    Framework (AI RMF 1.0). NIST AI 100-1, 2023. https://doi.org/10.6028/NIST.AI.100-1
15. Steyerberg EW. *Clinical Prediction Models*. 2nd edition. Springer, 2019.
    https://doi.org/10.1007/978-3-030-16399-0
16. Niculescu-Mizil A, Caruana R. Predicting good probabilities with supervised learning.
    *Proceedings of the 22nd International Conference on Machine Learning*, 2005.
    https://doi.org/10.1145/1102351.1102430
17. Murphy AH. A new vector partition of the probability score. *Journal of Applied Meteorology*
    1973;12(4):595-600.
18. scikit-learn developers. Probability calibration.
    https://scikit-learn.org/stable/modules/calibration.html

## 17. Links

- Class imbalance: [`06_class-imbalance.md`](06_class-imbalance.md)
- Model families: [`05_model-families.md`](05_model-families.md)
- Module map: [`_index.md`](_index.md)
- Previous topic: [`07_metrics.md`](07_metrics.md)
- Next topic: [`09_thresholds-and-decision-curves.md`](09_thresholds-and-decision-curves.md)
- Metrics: [`07_metrics.md`](07_metrics.md)
- Survey file: [`../../06_machine_learning_foundations.md`](../../06_machine_learning_foundations.md)
- Labs: [`../../labs/`](../../labs/)
- Calibration lab: [`../../labs/06_calibration_decision_curve.ipynb`](../../labs/06_calibration_decision_curve.ipynb)
