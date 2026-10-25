# Probability and Statistics Basics

> Module 01A — Foundations, Assumptions, and Units
> Topic 02 of 10
> Estimated study time: 4 hours
> Prerequisites: 01
> Version 2026.3

## 1. Why this topic exists

A number computed from data is an estimate, not a fact. The distinction is the whole of this topic.
An estimate has a sampling distribution: if the same study were repeated, the number would move. The
width of that movement determines whether a difference is distinguishable from noise, whether a
threshold is stable, and whether a deployment will reproduce the result.

Industry failures in this area are rarely caused by ignorance of formulas. They are caused by
reporting a point estimate as if it were exact, by testing a hypothesis nobody needed tested, and by
interpreting a p-value as something it is not.

- A quality dashboard reports that mortality "improved" from 8.1% to 7.9%. The denominator is 420
  admissions. The difference is 0.2 percentage points, well inside the sampling error of a rate based
  on 34 deaths. The dashboard shows a green arrow.
- A vendor validates a model on 1,100 held-out encounters and reports an area under the curve of
  0.91 with no confidence interval. The effective sample size, after accounting for repeated
  encounters per patient, is closer to 300. The interval is three times wider than the one implied by
  the naive count.
- A clinical team runs 14 subgroup comparisons and finds one with p = 0.03. The probability of at
  least one such result under the global null is 1 − 0.95^14 = 0.512. The finding is presented as a
  signal.
- A screening program with 85% sensitivity and 90% specificity is deployed in a population with 4%
  prevalence. Clinicians are surprised that only 26% of positive results are true positives. The
  arithmetic was never in dispute; nobody did it.
- A model is deployed with a threshold chosen to maximise the F1 score on the development set. In
  deployment the threshold produces three times the expected alert volume because the score
  distribution shifted.

Each of these is a statistics failure that a competent analyst can catch in an hour. None requires
advanced theory. All require the habit of asking: what is the unit of observation, what is the
sampling variation, what was the comparison, and what does the interval look like?

There is a second reason the topic exists. Later modules use statistical language as shorthand.
Module 06 assumes you know what calibration means. Module 07 assumes you can compute a rate with
person-time and attach an interval. Module 09 assumes you know why a missing value is not a zero.
Module 10 assumes you can write a statistical assumption into a ledger and design a test for it. A
learner who cannot state what a confidence interval covers will write ledger entries that cannot be
tested.

The commercial consequence is direct. Decisions about deployment, reimbursement, and trial design are
made on estimates with uncertainty. An analyst who quantifies that uncertainty changes the decision
from "does the number look good" to "is the range of plausible values compatible with the value we
need." That is a different, and much more defensible, conversation.

> **Note:** This topic teaches the frequentist and Bayesian ideas needed to read applied biomedical
> work. It does not derive estimators, prove asymptotic results, or cover likelihood theory in depth.
> Those belong to a dedicated statistics course.

## 2. Learning outcomes

By the end of this topic you can:

1. **Identify** the unit of observation in a dataset and detect when clustering invalidates an
   independence assumption. Assessed in Exercise 1 and self-check item 3.
2. **Compute** a point estimate, a standard error, and a confidence interval for a proportion, a mean,
   and a difference in proportions, and choose between the normal, Wilson, and bootstrap intervals.
   Assessed in Exercises 2 and 3.
3. **Interpret** a 95% confidence interval correctly and explain what it does and does not cover.
   Assessed in Exercise 4 and self-check item 5.
4. **Explain** why a p-value is not the probability that the null hypothesis is true, and state what
   it does measure. Assessed in Exercise 5 and interview question 4.
5. **Compute** positive and negative predictive value from sensitivity, specificity, and prevalence,
   and show how PPV changes when prevalence changes. Assessed in Exercise 6.
6. **Apply** likelihood ratios and Bayes' theorem to update a probability, and state the prior used.
   Assessed in Exercise 6 and self-check item 7.
7. **Diagnose** the statistical errors in a published abstract: missing intervals, multiplicity,
   cluster ignored, p-value misinterpreted, and significance confused with importance. Assessed in
   Exercise 7.
8. **Design** a minimal analysis plan for a stated decision, including the estimand, the interval
   method, the multiplicity policy, and the stopping rule. Assessed in Exercise 8.

## 3. The core mental model

The model is a funnel. A population parameter passes through a sampling process to produce data, and
the data produce an estimate. Every inference runs backward through the funnel, and every inference is
uncertain because many different populations could have produced the same data.

```text
        POPULATION
        parameter theta (fixed, unknown)
             |
             |  sampling process (random, biased, clustered)
             v
        SAMPLE
        n observations, possibly dependent
             |
             |  estimator (mean, proportion, regression coefficient)
             v
        ESTIMATE  theta_hat
             |
             |  sampling distribution of theta_hat
             v
        UNCERTAINTY
        standard error -> interval -> decision
```

Two properties of the funnel drive everything.

**The estimate is a random variable.** Before the data are collected, the estimate has a distribution.
After the data are collected, the realised estimate is one draw from that distribution. The confidence
interval is a statement about the procedure that produced the draw, not about the parameter.

**The sampling process determines the width.** More observations narrow the interval, but the rate of
narrowing depends on the design. Independent observations narrow it as 1/√n. Clustered observations
narrow it more slowly, because each additional observation inside a cluster adds less information than
an observation from a new cluster. A dataset with 10,000 rows from 200 patients has an effective
sample size closer to 200 than to 10,000 for any patient-level quantity.

### 3.1 The three questions of inference

Every statistical statement answers one of three questions:

| Question | Tool | Output |
|---|---|---|
| How large is the quantity? | Estimation | Point estimate plus interval |
| Is there a difference? | Hypothesis test | Test statistic and p-value |
| What should we believe now? | Bayesian updating | Posterior distribution |

The questions are not interchangeable. A test tells you whether a difference is distinguishable from
a specified null; it does not tell you how large the difference is. An estimate tells you the size but
requires a decision rule to act on. A posterior combines prior belief with data and requires the prior
to be stated.

### 3.2 The two errors and the asymmetry of testing

| Reality | Fail to reject H0 | Reject H0 |
|---|---|---|
| H0 true | Correct (1 − α) | Type I error (α) |
| H0 false | Type II error (β) | Correct (power, 1 − β) |

The conventional α = 0.05 is a convention, not a law. It means that a testing procedure applied to a
true null will produce a false positive 5% of the time. In a portfolio of 100 null experiments, about
five will be "significant." That is the arithmetic behind the multiplicity problem, and it is why the
number of tests matters as much as any single p-value.

### 3.3 Why intervals are more informative than tests

A p-value collapses the estimate and its uncertainty into one number and loses the scale. An interval
keeps both. "The difference is 3.3 percentage points, 95% CI 0.8 to 5.8" tells the reader the
direction, the magnitude, and the precision. "p = 0.01" tells the reader only that the interval
excludes zero. The interval is the better report in almost every applied setting, and regulatory and
reporting guidelines increasingly require it.

### 3.4 Bayes in one line

```text
posterior odds = prior odds x likelihood ratio
```

The likelihood ratio is the bridge between what a test can do (sensitivity and specificity) and what a
clinician needs (the probability of disease given a result). The prior is the prevalence in the
population being tested. Change the population, and the posterior changes even though the test has not
changed.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Population | The complete set about which a claim is made | Using "population" for the sample | none |
| Parameter | A fixed but unknown population quantity, θ | Calling an estimate a parameter | parameter scale |
| Sample | The observed subset | Treating a convenience sample as random | count |
| Unit of observation | The entity a row represents | Counting visits as independent patients | none |
| Independent observations | Observations whose values do not influence each other | Assuming independence within a patient | none |
| Clustering | Grouping of observations that induces correlation | Ignoring it in standard errors | none |
| Estimator | A rule for computing an estimate from data | Confusing the rule with the value | none |
| Estimate | The realised value from an estimator | Reporting it without uncertainty | parameter scale |
| Sampling distribution | The distribution of an estimator over repeated samples | Confusing with the data distribution | none |
| Standard error | The standard deviation of the sampling distribution | Calling it the standard deviation of the data | parameter unit |
| Confidence interval | An interval produced by a procedure with stated long-run coverage | Saying "95% probability the parameter is in this interval" | parameter unit |
| Coverage | The long-run proportion of intervals containing the parameter | Assuming coverage holds in every design | proportion |
| Bias | Expected estimate minus true value | Confusing with imprecision | parameter unit |
| Consistency | Convergence of an estimator to the parameter as n grows | Assuming it holds for any sampling scheme | none |
| Efficiency | Precision of an estimator relative to another | Treating it as unbiasedness | variance ratio |
| Null hypothesis (H0) | A specified value or model used as a reference | Believing it is probably true | parameter scale |
| Alternative hypothesis | The values considered under the test | Using a two-sided test for a one-sided claim | parameter scale |
| Test statistic | A quantity whose distribution is known under H0 | Interpreting its sign without the scale | depends |
| p-value | Probability, under H0, of data at least as extreme as observed | Probability that H0 is true | probability |
| Significance level (α) | The threshold for rejecting H0 | Treating it as the probability of error after the test | probability |
| Power | Probability of rejecting H0 when a specified alternative is true | Assuming post-hoc power is informative | probability |
| Type I error | Rejecting a true H0 | Confusing with a false positive in prediction | probability |
| Type II error | Failing to reject a false H0 | Ignoring it when planning | probability |
| Effect size | The magnitude of a difference or association | Reporting only its p-value | parameter unit |
| Multiplicity | Testing many hypotheses, inflating the family-wise error | Reporting the smallest p-value | count |
| Family-wise error rate | Probability of at least one false rejection across a family | Confusing with per-test α | probability |
| False discovery rate (FDR) | Expected proportion of false discoveries among rejections | Confusing with family-wise error | proportion |
| Prior | Belief distribution before data | Choosing a prior after seeing data | parameter distribution |
| Likelihood | Probability of the data as a function of the parameter | Confusing with posterior | none |
| Posterior | Updated belief after data | Reporting it without the prior | parameter distribution |
| Sensitivity | TP/(TP + FN) | Calling it PPV | proportion |
| Specificity | TN/(TN + FP) | Calling it NPV | proportion |
| PPV | TP/(TP + FP) | Assuming it is a property of the test alone | proportion |
| NPV | TN/(TN + FN) | Assuming it is a property of the test alone | proportion |
| Prevalence | Proportion with the condition at a time | Using it interchangeably with incidence | proportion |
| Likelihood ratio positive | Sensitivity/(1 − specificity) | Computing it as a ratio of risks | dimensionless |
| Likelihood ratio negative | (1 − sensitivity)/specificity | Using LR+ for a negative result | dimensionless |
| Pre-test probability | Probability before the test | Using the study prevalence when the local one differs | probability |
| Post-test probability | Probability after the test | Reporting it without the pre-test value | probability |
| Standard deviation | Spread of individual observations | Confusing with standard error | observation unit |
| Standard error of the mean | SD/√n for independent observations | Using SD/√n with clustered data | parameter unit |
| Bootstrap | Resampling with replacement to approximate a sampling distribution | Resampling rows when clusters are the unit | none |
| Percentile interval | Interval from the empirical bootstrap distribution | Using it when the statistic is heavily biased | parameter unit |
| Central limit theorem | Approximate normality of sums and means under conditions | Applying it to extreme quantiles with small n | none |
| Binomial distribution | Count of successes in n independent trials | Using it with clustered or varying-risk trials | count |
| Poisson distribution | Counts in a fixed exposure with a constant rate | Using it when counts are overdispersed | count |
| Normal distribution | Symmetric bell curve defined by mean and variance | Assuming all biomedical data are normal | parameter unit |
| Overdispersion | Variance greater than the nominal model implies | Ignoring it in confidence intervals | none |

Two definitions are worth isolating because they are the most frequently misstated in industry
material.

**Standard deviation versus standard error.** The standard deviation describes how spread out the
observations are. The standard error describes how spread out the estimate would be across repeated
studies. A clinical report that shows "mean ± SD" and a model report that shows "AUC ± SE" are
reporting different things, and a reader who treats them as the same will misjudge precision by a
factor of √n.

**PPV is not a property of the test.** Sensitivity and specificity are properties of the test in a
specified population. PPV and NPV depend on prevalence. A test with 85% sensitivity and 90%
specificity has a PPV of 26% at 4% prevalence and 54% at 12% prevalence, with identical test
performance. Any statement of the form "our test is 90% accurate" is incomplete without the
prevalence.

## 5. The mechanics, step by step

The procedure below takes a decision question through to a defensible statistical statement. It is
ordered so that each choice constrains the next: the estimand fixes the unit, the unit fixes the
independence structure, and the structure fixes the interval method.

### Step 1 — State the estimand before looking at the data

**Input.** A decision question, for example "should we adopt the sepsis alert."

**Output.** A one-sentence estimand: population, outcome, comparator, time window, and summary
measure.

**Reasoning.** The estimand determines every later choice. "The difference in 30-day mortality between
alert-exposed and unexposed adults admitted with suspected sepsis, estimated as a risk difference,"
is an estimand. "Does the alert work" is not. If the estimand is written after the analysis, the
analysis will be chosen to fit a preferred conclusion.

**Branch point.** If the decision does not require a comparison, no hypothesis test is needed. An
estimation-only analysis is often the right answer.

### Step 2 — Identify the unit of observation

**Input.** The dataset.

**Output.** A statement of what one row represents and how rows group.

**Reasoning.** The unit determines n. If one row is a patient-day and patients contribute many days,
the rows are not independent. If one row is an encounter and patients have multiple encounters, the
same. Virtually every clinical dataset has a grouping structure: patient, clinician, ward, site,
calendar period. The unit is the first thing to check and the most common thing to get wrong.

**Branch point.** If rows group, the effective sample size is closer to the number of groups than to
the number of rows for group-level quantities. Go to Step 3.

### Step 3 — Choose the independence structure and the variance method

**Input.** The unit and grouping.

**Output.** A decision: independent, clustered, or longitudinal, with the corresponding variance
estimator.

**Reasoning.** Independent observations allow the simple variance formulas. Clustered observations
require cluster-robust standard errors, a mixed model, or aggregation to the cluster level.
Longitudinal observations require methods that account for within-subject correlation. The choice
changes the interval width, sometimes by a factor of two or more.

**Branch point.** If the design is clustered and the number of clusters is small (below about 30),
cluster-robust standard errors can be anticonservative; use a small-sample correction or a
randomisation-based method.

### Step 4 — Specify the outcome distribution

**Input.** The outcome variable and its scale.

**Output.** A distributional family: Bernoulli for a binary outcome, binomial for a count of successes,
Poisson for event counts in exposure, normal for a continuous approximately symmetric outcome,
log-normal for a positive skewed outcome.

**Reasoning.** The distribution determines the mean-variance relationship and therefore the standard
error formula. A proportion has variance p(1 − p)/n. A count has variance equal to its mean under
Poisson, and greater than its mean under overdispersion. A positive skewed outcome analysed on the
raw scale will produce intervals that cross zero.

**Branch point.** If the observed variance exceeds the nominal (overdispersion), do not use the
Poisson standard error. Estimate the dispersion or use a negative binomial model.

### Step 5 — Compute the point estimate

**Input.** The data and the chosen summary.

**Output.** A single number with its unit.

**Reasoning.** For a proportion, the estimate is x/n. For a mean, it is the sample mean. For a
difference, it is the difference of the two estimates. For a rate, it is events/person-time. Report
the numerator and denominator alongside, because the estimate alone does not convey the information
needed to judge its precision.

**Branch point.** If the numerator is zero or equals the denominator, the normal approximation fails.
Go to Step 7 for an exact or Wilson method.

### Step 6 — Compute the standard error

**Input.** The estimate and the sample size.

**Output.** A standard error in the unit of the estimate.

**Reasoning.** For a proportion, SE = √(p(1 − p)/n). For a mean, SE = s/√n. For a difference in
proportions, SE = √(p1(1 − p1)/n1 + p2(1 − p2)/n2). For a rate, SE = √(events)/person-time under
Poisson. The formula encodes the design; using the independent formula on clustered data understates
the SE.

**Branch point.** If the formula's assumptions are doubtful, compute a bootstrap standard error
instead (Step 7b) and compare. A large discrepancy is evidence against the formula.

### Step 7 — Choose the interval method

**Input.** The estimate, the SE, and the sample size.

**Output.** A confidence interval with a stated method.

**Reasoning.** Three methods cover most applied cases:

- **Wald (normal approximation):** estimate ± z·SE. Simple, symmetric, and poor when p is near 0 or 1
  or n is small.
- **Wilson score interval:** solves the score equation and behaves well for proportions near the
  boundaries. Preferred for a single proportion with moderate n.
- **Bootstrap percentile:** resample the units with replacement, recompute the estimate many times,
  and take the 2.5th and 97.5th percentiles. Preferred when the estimator is complex or the
  distribution is unknown, provided the resampling respects the clustering.

**Branch point.** If p is below 0.05 or above 0.95 and n is below 100, use Wilson or exact. If the
estimator is a ratio of fitted quantities, use the bootstrap.

### Step 8 — Check the interval against the decision threshold

**Input.** The interval.

**Output.** A judgement: entirely below, entirely above, or spanning the threshold.

**Reasoning.** The decision-relevant question is usually not "is the effect non-zero" but "is the
effect large enough." If the threshold is a 2 percentage-point reduction and the interval is 0.8 to
5.8, the data are compatible with an effect too small to justify the program. That is a more useful
statement than "significant."

**Branch point.** If the interval spans the threshold, the decision is not resolved by these data.
State what additional sample or evidence would resolve it.

### Step 9 — Interpret the interval in plain language

**Input.** The interval.

**Output.** A sentence that is literally true.

**Reasoning.** The correct interpretation is procedural: "If we repeated this study many times, about
95% of intervals constructed this way would contain the true value." The incorrect interpretation is
"there is a 95% probability that the true value is in this interval," which assigns a probability to a
fixed parameter. The practical shorthand "we cannot rule out values between X and Y" is acceptable
because it describes compatibility rather than probability.

**Branch point.** If the audience needs a probability statement about the parameter, use a Bayesian
credible interval and state the prior. Do not relabel a confidence interval.

### Step 10 — Decide whether a hypothesis test is needed at all

**Input.** The decision question.

**Output.** A yes or no, with a reason.

**Reasoning.** A test is needed when a specific null value must be rejected as a gate for a decision,
for example a regulatory non-inferiority margin. A test is not needed when the decision is a
cost-benefit comparison, which requires the estimate and its interval. Many industry analyses run
tests by habit and then act on the p-value, which inverts the logic: the test was never the decision
rule.

**Branch point.** If no decision depends on rejecting a specific null, report the estimate and
interval and skip the test.

### Step 11 — State the null and alternative before computing

**Input.** The estimand.

**Output.** H0 and H1 with the direction and the margin.

**Reasoning.** The null must be specific enough to define a test statistic. "No difference" is
specific for a superiority test. "Difference worse than −5 percentage points" is the null for a
non-inferiority test. Choosing the null after seeing the data allows the analyst to pick the test that
produces the desired answer.

**Branch point.** If the question is equivalence rather than superiority, the null is a difference
outside a margin, and the test is two one-sided tests. A standard two-sided test does not answer an
equivalence question.

### Step 12 — Compute the test statistic and the p-value

**Input.** The estimate, the null value, and the SE under the null.

**Output.** A test statistic and a p-value.

**Reasoning.** For a difference in proportions, z = (p1 − p2)/SE_pooled under the null of no
difference. The p-value is the probability, under H0, of a statistic at least as extreme. For a
two-sided test it includes both tails. The p-value is a property of the data and the null, not of the
hypothesis.

**Branch point.** If the expected cell count is below 5 in a 2×2 table, use an exact test. If the
outcome is a count with overdispersion, use a quasi-Poisson or negative binomial test.

### Step 13 — Adjust for multiplicity if a family was tested

**Input.** The number of tests and their p-values.

**Output.** Adjusted p-values or a stated error rate.

**Reasoning.** Testing k independent null hypotheses at α = 0.05 gives a family-wise error rate of
1 − 0.95^k. For k = 14, that is 0.512. Bonferroni controls the family-wise rate by testing each at
α/k. Benjamini-Hochberg controls the false discovery rate, which is less conservative when many tests
are expected to be real. The choice depends on whether a single false positive is costly (family-wise)
or whether a proportion of false discoveries is tolerable (FDR).

**Branch point.** If the tests were not pre-specified as a family, the adjustment is post hoc and must
be labelled as such. Exploratory findings should be reported as hypotheses, not conclusions.

### Step 14 — Distinguish statistical from clinical significance

**Input.** The p-value and the interval.

**Output.** A statement about magnitude relative to a decision threshold.

**Reasoning.** With a large enough sample, a trivial difference becomes statistically significant. A
0.1 percentage-point mortality difference can have p < 0.001 in a registry of two million admissions
and still be irrelevant to care. The decision threshold must come from clinical or operational
reasoning, not from the test.

**Branch point.** If the interval excludes zero but lies entirely within the region of clinical
equivalence, the correct conclusion is "no clinically important difference detected," not "effective."

### Step 15 — Update a probability with Bayes when the question is diagnostic

**Input.** Pre-test probability, sensitivity, specificity.

**Output.** Post-test probability.

**Reasoning.** Convert the pre-test probability to odds, multiply by the likelihood ratio, and convert
back. Equivalently, use the 2×2 table with the local prevalence. The two routes agree exactly. The
likelihood ratio is preferable when the result is continuous or multi-level, because each level has
its own LR.

**Branch point.** If the local prevalence is unknown, do not substitute the trial prevalence without
stating the assumption. Report the post-test probability as a function of prevalence.

### Step 16 — Sensitivity-test the prior

**Input.** The post-test probability at one prevalence.

**Output.** A table across plausible prevalences.

**Reasoning.** PPV is approximately linear in prevalence over moderate ranges, so a table of
prevalences is easy to produce and highly informative. The table shows the clinician how much the
answer depends on the population, which is exactly the quantity that changes between the emergency
department and the intensive care unit.

**Branch point.** If the decision flips across the plausible prevalence range, the test alone cannot
support the decision. Additional information or a different test is required.

### Step 17 — Plan the sample size against the interval width

**Input.** The desired precision and the expected proportion.

**Output.** The required n, with the formula and assumptions stated.

**Reasoning.** For a proportion, the half-width of a Wald interval is z·√(p(1 − p)/n). Solving for n
gives n = z²p(1 − p)/w². For a 95% interval with w = 0.03 and p = 0.10, n = 3.8416 × 0.09/0.0009 =
384.2, so 385. Power calculations for tests use a related formula with the alternative and β.

**Branch point.** If the design is clustered, inflate n by the design effect 1 + (m − 1)·ICC, where m
is the cluster size and ICC the intraclass correlation. Ignoring the design effect is a common cause
of underpowered cluster trials.

### Step 18 — Handle missing data explicitly

**Input.** The analysis dataset with missingness.

**Output.** A statement of the missingness mechanism and the handling method.

**Reasoning.** Complete-case analysis is valid only if missingness is independent of the outcome given
the covariates (missing completely at random, or missing at random with the model correctly
specified). When sicker patients are less likely to be tested, complete-case analysis selects a
healthier sample and biases the estimate. Multiple imputation under a stated model is the default for
missing at random; sensitivity analysis under missing not at random is required when the mechanism is
suspect.

**Branch point.** If the missingness is likely informative, report the complete-case and imputed
estimates side by side, and state the direction of the likely bias.

### Step 19 — Check calibration, not only discrimination

**Input.** A risk model and validation data.

**Output.** A calibration plot and a calibration slope or intercept.

**Reasoning.** Discrimination (AUROC) measures ranking; calibration measures whether predicted
probabilities match observed frequencies. A model can rank well and be badly calibrated, which makes
every threshold-based decision wrong even though the AUROC looks good. Report both, and report them in
the deployment population.

**Branch point.** If calibration is poor but discrimination is good, recalibrate (intercept and slope)
before deployment rather than discarding the model.

### Step 20 — Run a sensitivity analysis on the assumptions that matter

**Input.** The primary analysis and the assumption ledger.

**Output.** A table of estimates under alternative assumptions.

**Reasoning.** The purpose is not to find the analysis that gives the desired answer but to show how
much the answer depends on assumptions that cannot be tested. Typical dimensions: the independence
structure, the missingness mechanism, the prior, the outcome definition, and the time window.

**Branch point.** If the conclusion changes qualitatively across plausible assumptions, the analysis
does not support a decision. Report the range and the conditions under which each conclusion holds.

### Step 21 — Report the estimate, the interval, the n, and the method

**Input.** All of the above.

**Output.** A reporting sentence and a table row.

**Reasoning.** A complete report contains: the estimand, the point estimate with unit, the interval
with its method and level, the sample size and number of events, and the software or method used.
Anything less cannot be reproduced or critiqued.

**Branch point.** If any element is missing, the analysis is not ready for a decision meeting.

### 5.1 The branch map

```text
Decision question
  |
  +-- no comparison needed? ----------> estimate + interval only
  |
  +-- rows grouped? ------------------> cluster-robust / mixed / aggregate
  |
  +-- binary outcome near 0 or 1? ----> Wilson or exact interval
  |
  +-- count with variance > mean? ----> negative binomial / quasi-Poisson
  |
  +-- many tests? --------------------> pre-specify family; Bonferroni or BH
  |
  +-- diagnostic question? -----------> Bayes with local prevalence
  |
  +-- decision threshold? ------------> compare interval to threshold, not to zero
  |
  +-- missing outcome data? ----------> state mechanism; impute + sensitivity
  |
  +-- model with threshold? ----------> calibration before deployment
```

### 5.2 The four checks that catch most statistical errors

1. **Count the unit, not the rows.** Report n as patients or clusters, and state the rows-per-unit
   ratio.
2. **Put an interval on every estimate.** A point estimate without an interval is a claim without a
   precision.
3. **Write the null before the test.** If the null was chosen after seeing the data, the p-value is
   descriptive, not inferential.
4. **Convert the p-value into a statement about magnitude.** If the magnitude is not clinically
   meaningful, significance is irrelevant.

## 6. Worked example

An emergency department (ED) is evaluating a sepsis screening alert. The vendor reports sensitivity
0.85 and specificity 0.90. The ED sees 12,000 admissions per year with a sepsis prevalence of 4.0%.
The department also wants to know whether its 30-day mortality, 34 deaths in 420 admissions, differs
from a peer hospital reporting 46 deaths in 410 admissions.

The case carries four calculations end to end: the 2×2 table at two prevalences, the interval for a
single proportion by three methods, the two-proportion comparison, and the multiplicity correction for
a 14-test dashboard.

### 6.1 Build the 2×2 table at the local prevalence

```text
Population         = 12,000 admissions
Prevalence         = 4.0% -> 12,000 x 0.040 = 480 sepsis cases
Non-cases          = 12,000 - 480 = 11,520

True positives     = 480 x 0.85 = 408
False negatives    = 480 - 408 = 72
False positives    = 11,520 x 0.10 = 1,152
True negatives     = 11,520 - 1,152 = 10,368

Total positives    = 408 + 1,152 = 1,560
Total negatives    = 72 + 10,368 = 10,440
Check              = 1,560 + 10,440 = 12,000
```

```text
PPV = 408 / 1,560 = 0.26154 = 26.2%
NPV = 10,368 / 10,440 = 0.99310 = 99.3%
```

The alert is correct 26% of the time when it fires. It misses 72 cases per year. It generates 1,152
false alerts per year, or about 3.2 per day. The sensitivity and specificity are unchanged from the
vendor report; the PPV is a property of the deployment population, not of the test.

### 6.2 Recompute at the intensive care prevalence

The same test used in an intensive care unit (ICU) with 12% prevalence:

```text
Cases        = 12,000 x 0.12 = 1,440
Non-cases    = 10,560
TP           = 1,440 x 0.85 = 1,224
FN           = 216
FP           = 10,560 x 0.10 = 1,056
TN           = 9,504

PPV = 1,224 / (1,224 + 1,056) = 1,224 / 2,280 = 0.53684 = 53.7%
NPV = 9,504 / (9,504 + 216) = 9,504 / 9,720 = 0.97778 = 97.8%
```

The PPV more than doubles with no change in the test. The false-positive count falls from 1,152 to
1,056 while the true-positive count triples. This is the base-rate effect, and it is the single most
important quantitative fact for anyone deploying a classifier.

### 6.3 The same answer by likelihood ratios

```text
LR+ = sensitivity / (1 - specificity) = 0.85 / 0.10 = 8.50
LR- = (1 - sensitivity) / specificity = 0.15 / 0.90 = 0.1667

At 4% prevalence:
  pre-test odds = 0.04 / 0.96 = 0.041667
  post-test odds = 0.041667 x 8.50 = 0.354167
  post-test probability = 0.354167 / 1.354167 = 0.26154

At 12% prevalence:
  pre-test odds = 0.12 / 0.88 = 0.136364
  post-test odds = 0.136364 x 8.50 = 1.159091
  post-test probability = 1.159091 / 2.159091 = 0.53684
```

Both routes give the same PPV. The likelihood-ratio route generalises to tests with more than two
results and to continuous markers, where each result level carries its own LR.

### 6.4 Interval for the mortality proportion

The ED observed 34 deaths in 420 admissions.

```text
p  = 34 / 420 = 0.080952 = 8.10%
SE = sqrt(p(1 - p)/n)
   = sqrt(0.080952 x 0.919048 / 420)
   = sqrt(0.00017713)
   = 0.013309 = 1.33 percentage points
```

**Wald interval.**

```text
p +/- 1.96 x SE = 0.080952 +/- 1.96 x 0.013309
                = 0.080952 +/- 0.026086
                = 0.054866 to 0.107038
                = 5.49% to 10.70%
```

**Wilson score interval.** The Wilson interval solves the score equation rather than approximating
symmetrically.

```text
z^2 = 3.8416
centre = (p + z^2/(2n)) / (1 + z^2/n)
       = (0.080952 + 0.004573) / 1.009147
       = 0.084751

half-width = z/(1 + z^2/n) x sqrt(p(1-p)/n + z^2/(4n^2))
           = 1.94223 x sqrt(0.00017713 + 0.00000545)
           = 1.94223 x 0.013512
           = 0.026244

interval = 0.058507 to 0.110994 = 5.85% to 11.10%
```

**Bootstrap percentile interval.** Resampling 420 Bernoulli observations with p = 0.080952, 4,000
times with a fixed seed, gives 5.71% to 10.71%. The three intervals agree to within about 0.4
percentage points, which is expected at this sample size. The Wilson interval is the best default for
a single proportion; the Wald interval is the narrowest and the least trustworthy near the
boundaries.

### 6.5 Compare two hospitals

```text
Hospital A: 34/420  = 0.080952
Hospital B: 46/410  = 0.112195
Difference = 0.080952 - 0.112195 = -0.031243 = -3.12 percentage points

Pooled p = (34 + 46)/(420 + 410) = 80/830 = 0.096386
SE_diff  = sqrt(0.096386 x 0.903614 x (1/420 + 1/410))
         = sqrt(0.096386 x 0.903614 x 0.004823)
         = sqrt(0.0004200)
         = 0.020489

z = -0.031243 / 0.020489 = -1.5249
two-sided p = 0.1273
```

The difference is not distinguishable from zero at the conventional threshold. More importantly, the
difference is 3.1 percentage points with a standard error of 2.0, so the data are compatible with
everything from a 7.1-point lower mortality at hospital A to a 0.9-point higher mortality. A
dashboard that colours hospital A green and hospital B red on this evidence is reporting noise.

The relative difference is −27.8%, which sounds large. The absolute difference is 3.1 percentage
points. Both are correct, and the second is the one that matters for patients.

### 6.6 Multiplicity on a 14-metric dashboard

A quality dashboard tracks 14 independent mortality metrics, each tested at α = 0.05.

```text
P(at least one false positive) = 1 - (1 - 0.05)^14
                               = 1 - 0.95^14
                               = 1 - 0.4877
                               = 0.5123
```

There is a 51% chance that at least one metric turns red purely by chance in a period with no real
change. A Bonferroni correction tests each metric at 0.05/14 = 0.00357, which controls the family-wise
error at 5% but makes each individual signal harder to detect. A false discovery rate procedure is
preferable when the goal is to triage metrics for review rather than to make a definitive claim about
any one.

### 6.7 Sample size for a desired precision

The department wants a mortality estimate with a 95% interval half-width of 3 percentage points,
assuming p ≈ 0.08.

```text
n = z^2 p(1-p) / w^2
  = 3.8416 x 0.080952 x 0.919048 / 0.03^2
  = 3.8416 x 0.074391 / 0.0009
  = 0.28577 / 0.0009
  = 317.5 -> 318 admissions
```

At 420 admissions the achieved half-width is 2.6 percentage points, slightly better than requested.
A target of 2 percentage points would require 715 admissions, which is why precision targets should be
set against the cost of collecting more data.

### 6.8 What the example demonstrates

| Calculation | Quantity | Why it matters |
|---|---|---|
| 2×2 at 4% | PPV 26.2% | Test performance is not a property of the test alone |
| 2×2 at 12% | PPV 53.7% | Deployment population changes the meaning of a positive |
| Likelihood ratio | LR+ 8.5 | One number transports across prevalences |
| Wald interval | 5.49% to 10.70% | Symmetric approximation, narrowest |
| Wilson interval | 5.85% to 11.10% | Better behaviour near boundaries |
| Bootstrap | 5.71% to 10.71% | Design-agnostic check |
| Two-proportion test | p = 0.127 | A 28% relative difference is not distinguishable |
| Multiplicity | FWER 0.512 | Dashboard colouring is mostly noise |
| Sample size | 318 for ±3 pp | Precision has a price |

## 7. Code, query, or calculation

The script reproduces every number in section 6 using only the standard library. The bootstrap uses a
fixed seed so the output is reproducible.

```python
"""Topic 02 verification: PPV, Wilson interval, z-test, bootstrap, multiplicity."""
from math import sqrt, erf
import random

def phi(z):
    return 0.5 * (1 + erf(z / sqrt(2)))

def two_sided_p(z):
    return 2 * (1 - phi(abs(z)))

# --- A. Diagnostic test at two prevalences -----------------------------------
sens, spec = 0.85, 0.90
def bayes_table(n, prev):
    cases = round(n * prev)
    non = n - cases
    tp = round(cases * sens)
    fn = cases - tp
    fp = round(non * (1 - spec))
    tn = non - fp
    ppv = tp / (tp + fp)
    npv = tn / (tn + fn)
    return cases, non, tp, fn, fp, tn, ppv, npv

print("A. Diagnostic table")
for n, prev in ((12000, 0.04), (12000, 0.12)):
    cases, non, tp, fn, fp, tn, ppv, npv = bayes_table(n, prev)
    print(f"  n={n} prev={prev:.0%} cases={cases} non={non} "
          f"TP={tp} FN={fn} FP={fp} TN={tn}")
    print(f"    PPV={ppv:.4f} NPV={npv:.4f}")

lr_pos = sens / (1 - spec)
lr_neg = (1 - sens) / spec
print(f"  LR+ = {lr_pos:.4f}  LR- = {lr_neg:.4f}")

def post_from_odds(prev, lr):
    odds = prev / (1 - prev) * lr
    return odds / (1 + odds)

print(f"  post-test at 4%  = {post_from_odds(0.04, lr_pos):.4f}")
print(f"  post-test at 12% = {post_from_odds(0.12, lr_pos):.4f}")

# --- B. Mortality proportion, three intervals --------------------------------
n_deaths, n_adm = 34, 420
p = n_deaths / n_adm
se = sqrt(p * (1 - p) / n_adm)
z = 1.959963985
wald = (p - z * se, p + z * se)

z2 = z * z
centre = (p + z2 / (2 * n_adm)) / (1 + z2 / n_adm)
half = z / (1 + z2 / n_adm) * sqrt(p * (1 - p) / n_adm + z2 / (4 * n_adm ** 2))
wilson = (centre - half, centre + half)

print("\nB. Mortality proportion")
print(f"  p = {p:.6f}  SE = {se:.6f}")
print(f"  Wald   {wald[0]:.6f} to {wald[1]:.6f}")
print(f"  Wilson {wilson[0]:.6f} to {wilson[1]:.6f}")

random.seed(20261024)
B = 4000
boot = []
for _ in range(B):
    draws = sum(1 for _ in range(n_adm) if random.random() < p)
    boot.append(draws / n_adm)
boot.sort()
lo = boot[int(0.025 * B)]
hi = boot[int(0.975 * B) - 1]
print(f"  bootstrap {lo:.6f} to {hi:.6f}")

# --- C. Two-proportion z test ------------------------------------------------
x1, n1 = 34, 420
x2, n2 = 46, 410
p1, p2 = x1 / n1, x2 / n2
pooled = (x1 + x2) / (n1 + n2)
se_diff = sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
z_stat = (p1 - p2) / se_diff
pval = two_sided_p(z_stat)
print("\nC. Two-proportion test")
print(f"  p1={p1:.6f} p2={p2:.6f} diff={p1-p2:.6f}")
print(f"  pooled={pooled:.6f} SE={se_diff:.6f}")
print(f"  z={z_stat:.4f} p={pval:.4f}")

# --- D. Multiplicity ---------------------------------------------------------
for k in (1, 5, 14, 20):
    print(f"  k={k:2d} FWER at alpha .05 = {1 - 0.95 ** k:.4f}")

# --- E. Sample size for a proportion ----------------------------------------
for w in (0.05, 0.03, 0.02):
    n_req = z2 * p * (1 - p) / (w * w)
    print(f"  half-width {w:.2f} -> n = {n_req:.1f} -> {int(n_req) + 1}")
```

**Expected output.**

```text
A. Diagnostic table
  n=12000 prev=4% cases=480 non=11520 TP=408 FN=72 FP=1152 TN=10368
    PPV=0.2615 NPV=0.9931
  n=12000 prev=12% cases=1440 non=10560 TP=1224 FN=216 FP=1056 TN=9504
    PPV=0.5368 NPV=0.9778
  LR+ = 8.5000  LR- = 0.1667
  post-test at 4%  = 0.2615
  post-test at 12% = 0.5368

B. Mortality proportion
  p = 0.080952  SE = 0.013309
  Wald   0.054866 to 0.107038
  Wilson 0.058507 to 0.110994
  bootstrap 0.057143 to 0.107143

C. Two-proportion test
  p1=0.080952 p2=0.112195 diff=-0.031243
  pooled=0.096386 SE=0.020489
  z=-1.5249 p=0.1273
  k= 1 FWER at alpha .05 = 0.0500
  k= 5 FWER at alpha .05 = 0.2262
  k=14 FWER at alpha .05 = 0.5123
  k=20 FWER at alpha .05 = 0.6415
  half-width 0.05 -> n = 114.3 -> 115
  half-width 0.03 -> n = 317.6 -> 318
  half-width 0.02 -> n = 714.5 -> 715
```

**Assumptions stated by the script.**

1. The trials for the 2×2 table are independent admissions. If patients have repeat admissions, the
   effective n is smaller and the PPV interval is wider.
2. Sensitivity and specificity are constant across prevalences. In reality they can vary with case
   severity, so the table is a transportability assumption, not a fact.
3. The two hospitals have comparable case mix. If hospital B admits sicker patients, the comparison
   is confounded.
4. The bootstrap resamples individual admissions. For clustered data the cluster must be resampled.
5. The 14 dashboard metrics are independent. Correlated metrics reduce the effective number of tests
   and make the Bonferroni correction conservative.

## 8. Data, standards, and regulatory context

Statistics enters regulation through two doors: the standards that govern laboratory measurement, and
the guidelines that govern trial design and reporting. Both constrain what an analyst may claim.

### 8.1 Laboratory measurement standards

**CLSI EP05-A3**, *Evaluation of Precision of Quantitative Measurement Procedures*, third edition,
published October 2014, defines how repeatability and within-laboratory precision are estimated. The
standard single-site design is 20 days × 2 runs per day × 2 replicates per run, giving 80 results per
sample. A two-way nested random-effects analysis of variance (ANOVA) separates three variance
components:

- S_R = √V_error, the repeatability (within-run) standard deviation.
- S_WL = √(V_day + V_run + V_error), the within-laboratory precision.
- Reproducibility adds site and instrument components in the multisite design.

The worked example in the standard's appendix reports glucose at a mean of 244.2 mg/dL with
S_R = 2.81 mg/dL (1.2% coefficient of variation) and S_WL = 3.60 mg/dL (1.5%). The standard notes
that within-laboratory precision from such a study is a lower bound for routine measurement
uncertainty, because long-term reagent and calibrator lot effects are not fully represented. That
caveat is the statistical version of the assumption ledger: the design measures what it measures, and
the analyst must state the gap.

**CLSI EP15-A3** is the companion standard for end-user laboratories verifying a manufacturer's
precision claims, harmonised with the EP05 single-site design. A laboratory that verifies precision
with the wrong design will produce a number that cannot be compared with the claim.

**CLSI EP28-A3c**, *Defining, Establishing, and Verifying Reference Intervals in the Clinical
Laboratory*, third edition, recommends a minimum of 120 reference individuals per partition for a
nonparametric reference interval. The reason is statistical: the 90% confidence interval for the 2.5th
and 97.5th percentiles requires at least 120 values, and with exactly 120 the interval spans roughly
the 1.9th to 3.1st percentile. Laboratories that cannot recruit 120 typically verify a transferred
interval using as few as 20 reference individuals under the standard's validation procedure. This is a
direct example of a sample-size rule derived from an interval-width requirement, which is the logic of
Step 17.

### 8.2 Trial design and reporting standards

**ICH E9(R1)**, the addendum on estimands and sensitivity analysis, requires the sponsor to define the
estimand before analysis: population, treatment condition, variable, handling of intercurrent events,
and population-level summary. It is binding for submissions in the ICH regions. The statistical
consequence is that the primary estimate and its interval must correspond to a pre-specified estimand,
not to whichever analysis produces the smallest p-value.

**ICH E9**, *Statistical Principles for Clinical Trials*, establishes the conventional α = 0.05,
two-sided testing, the role of intention-to-treat, and the requirement for a pre-specified analysis
plan. It is the regulatory basis for the "write the null before the test" rule in Step 11.

**The ASA Statement on p-Values** (Wasserstein and Lazar, *The American Statistician* 70(2), 2016)
sets out six principles that are now widely cited in industry guidance:

1. p-values can indicate how incompatible the data are with a specified model.
2. They do not measure the probability that the hypothesis is true, or that the data arose by chance.
3. Conclusions should not be based only on whether p passes a threshold.
4. Full reporting and transparency are required.
5. A p-value does not measure the size or importance of an effect.
6. A p-value alone is not a good measure of evidence.

The statement is not binding regulation. It is the consensus position of the professional body, and
it is the standard against which an analysis will be judged in review.

**CONSORT** and **STROBE** require intervals, effect sizes, and the number of participants at each
stage. **TRIPOD** (Transparent Reporting of a multivariable prediction model for Individual Prognosis
Or Diagnosis) requires the reporting of calibration and discrimination, which is the basis for Step 19.

### 8.3 What the standards do not constrain

- They do not specify which interval method to use for a given estimator. Wilson versus Wald versus
  bootstrap is an analyst choice, and the choice must be stated.
- They do not resolve multiplicity across exploratory analyses. They require pre-specification, not a
  particular correction.
- They do not define clinical importance. A statistically significant 0.5 mmHg blood pressure
  difference is compatible with every standard and still clinically trivial.
- They do not validate a model in a new population. TRIPOD requires reporting, not external
  validation.

> **Note:** The regulatory content in this section is educational. A submission-grade statistical
> analysis plan requires review by a qualified statistician and, where applicable, prior agreement
> with the regulator.

## 9. Industry practice and case evidence

### 9.1 What organisations actually do

**Clinical laboratories** run internal quality control daily and external quality assessment
proficiency testing periodically. The precision claims on a reagent package insert are established
under EP05-A3 and verified under EP15-A3. A laboratory that reports a result without a method-specific
uncertainty is relying on the manufacturer's claim, which is a documented assumption in the ledger.

**Regulated prediction models** are validated with a pre-specified protocol that separates development
from validation by time or site. The FDA's guidance on clinical decision support and the IMDRF
frameworks for machine-learning-enabled devices expect a description of the validation population and
of the model's limitations. A random split within one site is not external validation, and reviewers
increasingly say so.

**Payers and health technology assessment bodies** require intervals and sensitivity analyses. NICE's
methods manual and ICER's value assessment framework both require probabilistic sensitivity analysis
for cost-effectiveness models, which is a Monte Carlo propagation of uncertainty through the model
inputs. A single-point incremental cost-effectiveness ratio is routinely rejected.

**Pharmaceutical sponsors** pre-register the primary estimand and the analysis plan, and they report
the primary result with a confidence interval. The estimand framework in ICH E9(R1) has made the
"which analysis answers the question" problem an explicit regulatory artefact rather than an analyst
convention.

### 9.2 Case evidence

**The reproducibility project.** The Open Science Collaboration (2015, *Science* 349, aac4716)
replicated 100 psychology studies. Of the original studies, 97% reported significant results; of the
replications, 36% did. Mean effect sizes in the replications were roughly half those in the originals.
The result is not specific to psychology; it is a demonstration that a single significant result is
weak evidence, which is principle 6 of the ASA statement in operational form.

**Diabetic retinopathy screening.** Gulshan et al. (2016, *JAMA* 316(22):2402–2410) reported a deep
learning algorithm for referable diabetic retinopathy with high sensitivity and specificity in two
validation sets. The documented lesson is methodological: the validation sets were drawn from specific
imaging sources, and subsequent work examined performance across cameras, populations, and grading
standards. A high AUROC in a source-matched validation set is discrimination, not deployment
readiness; calibration and transportability remain separate questions.

**Sepsis alert fatigue.** Health systems that deploy sepsis alerts report high override rates and
limited specificity in practice, which is the PPV problem of section 6.1 at scale: a test with
reasonable sensitivity and specificity applied at low prevalence produces mostly false positives. The
operational response in several systems has been to narrow the trigger population, which raises
prevalence and therefore PPV, rather than to improve the test.

**Laboratory reference intervals.** A reference interval transferred from a manufacturer's population
to a local population without verification is a documented source of misclassification, particularly
for analytes with strong age or ancestry dependence. EP28-A3c exists because the statistical problem,
estimating the 2.5th and 97.5th percentiles with adequate precision, is separable from the clinical
problem of choosing the reference population.

> **Note:** The organisations named above are cited for documented standards or publications. The
> operational interpretation in each paragraph is inference from the cited material, not a claim about
> a specific institution's internal results.

### 9.3 The practical pattern

Teams that get this right do four things consistently:

1. They report n as units, not rows.
2. They attach intervals to every estimate that enters a decision.
3. They pre-specify the primary comparison and label everything else exploratory.
4. They recompute PPV in the deployment population before launch.

None of these requires advanced methods. All four are cheap, and each prevents a class of failure that
is expensive to reverse after a product is in use.

## 10. Failure modes and red flags

| # | Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---:|---|---|---|---|---|
| 1 | Rows treated as independent observations | n is in the tens of thousands but the cohort is hundreds of patients | Repeated measures per patient inflate the apparent sample size | Intervals too narrow by a factor of √(design effect); false significance | Report units and rows; cluster-robust or mixed-model variance |
| 2 | Point estimate reported without an interval | Slide shows "8.1%" and nothing else | Precision omitted for brevity | Reader cannot judge whether the change is noise | Require intervals on every estimate entering a decision |
| 3 | p-value read as the probability the null is true | "There is a 5% chance the drug does nothing" | Inverting the conditional probability | Belief updated in the wrong direction | State that p = P(data \| H0), not P(H0 \| data) |
| 4 | Significance confused with importance | p < 0.001 on a 0.1 percentage-point difference | Large n makes trivial effects significant | Resources reallocated for no clinical benefit | Report the effect size and compare with a decision threshold |
| 5 | Multiplicity ignored | One of 14 metrics is highlighted as improved | Family-wise error grows with the number of tests | Roughly half of dashboards will show a spurious signal | Pre-specify the family; Bonferroni or Benjamini-Hochberg |
| 6 | PPV computed from the study prevalence | "90% accurate" claimed for a low-prevalence screen | Base-rate neglect | Clinicians over-trust positive results | Recompute PPV at local prevalence; report LR+ |
| 7 | Confidence interval method mismatched to the estimand | Symmetric interval around a proportion of 2% | Wald approximation fails near boundaries | Interval includes impossible values or has poor coverage | Use Wilson or exact for extreme proportions |
| 8 | Post-hoc power calculated from the observed effect | "The study was underpowered" after a null result | Observed power is a function of the p-value | Circular reasoning; no new information | Report the interval, which already conveys precision |
| 9 | Subgroup analysis presented as primary | Forest plot with one significant subgroup highlighted | Multiplicity plus effect modification | False claim of differential benefit | Pre-specify subgroups; test interaction, not within-subgroup p |
| 10 | Missing data handled by complete-case analysis without comment | Analysis n is lower than enrolment and no reason given | Informative missingness selects a subsample | Biased estimate, usually toward the healthy | State the mechanism; multiple imputation; sensitivity analysis |
| 11 | Calibration not assessed | AUROC reported, no calibration plot | Discrimination and calibration conflated | Thresholds produce wrong alert volumes | Report calibration slope, intercept, and plot |
| 12 | Poisson variance assumed for overdispersed counts | Variance exceeds the mean in the count data | Unmodelled heterogeneity | Intervals too narrow; spurious rate differences | Estimate dispersion; negative binomial or quasi-Poisson |
| 13 | Change score analysed without regression to the mean | Extreme baseline group improves | Selection on baseline value | Programme credited for natural fluctuation | Randomise or use a concurrent control; model baseline |
| 14 | Odds ratio interpreted as a risk ratio | Common outcome and OR of 2.5 | Odds exceed risk at high incidence | Effect overstated | Convert with baseline risk or report risk difference |
| 15 | Effective sample size ignored in cluster trials | 20 clinics, 2,000 patients, analysed as 2,000 | Design effect not applied | Underpowered study reported as adequately powered | Inflate sample size by 1 + (m − 1)·ICC |

Three rows deserve emphasis.

**Row 1, clustering.** The design effect is 1 + (m − 1)·ICC, where m is the average cluster size and
ICC the intraclass correlation. With m = 20 visits per patient and ICC = 0.3, the design effect is
1 + 19 × 0.3 = 6.7. A nominal n of 10,000 has an effective n of about 1,500. The interval is √6.7 =
2.6 times wider than the naive one.

**Row 6, base rate.** The PPV at 4% prevalence was 26% in section 6.1. A clinician told "90%
accurate" will interpret a positive result as a 90% probability of disease. The gap between 26% and
90% is the cost of not doing the arithmetic.

**Row 13, regression to the mean.** Any group selected for an extreme baseline value will improve on
average without any intervention. A before-after evaluation of the worst-performing wards will always
show improvement. This is arithmetic, not a clinical effect, and it is the most common explanation for
a successful pilot that fails to replicate.

## 11. Assumptions and limits

| # | Assumption | Why it is needed | What breaks it | How to test |
|---:|---|---|---|---|
| 1 | The sample is representative of the target population | Generalisation of the estimate | Convenience sampling, referral bias, survivorship | Compare sample characteristics with the target; weight if needed |
| 2 | Observations are independent within the analysis unit | Standard error formulas | Repeated measures, clustering, contagion | Inspect the grouping; compare naive and cluster-robust SEs |
| 3 | The outcome model matches the data type | Correct mean-variance relationship | Overdispersion, zero inflation, censoring | Residual diagnostics; dispersion estimate |
| 4 | The estimand is stable across the study period | A single number summarises the period | Secular trend, policy change, seasonality | Plot the estimate by period; test for a level shift |
| 5 | Missingness is ignorable given the model | Complete-case or imputation validity | Sicker patients untested; informative dropout | Compare observed characteristics by missingness status |
| 6 | The prior is stated and defensible | Bayesian inference | Prior chosen after seeing data; undeclared prior | Sensitivity analysis across priors |
| 7 | The test null was pre-specified | p-value interpretation | Null chosen after inspection | Check the analysis plan and registration |
| 8 | Multiplicity is controlled within a stated family | Family-wise or FDR control | Undeclared tests; data-dependent families | Count the tests; pre-register the family |
| 9 | Sensitivity and specificity transport to the new population | PPV computation | Spectrum bias, different case mix, different operators | Stratify by severity; local validation |
| 10 | The model is calibrated in the deployment population | Threshold-based decisions | Case-mix shift, prevalence shift, coding change | Calibration plot in deployment data |
| 11 | The measurement is unbiased | Estimation of a true quantity | Assay drift, observer bias, device bias | Validation substudy; blinded comparison |
| 12 | The interval method has adequate coverage at this n | Interval interpretation | Small n, extreme proportion, skewed statistic | Simulation study for the specific design |
| 13 | The comparison group is exchangeable with the treated group | Causal interpretation of a difference | Confounding by indication, era effects | Randomisation; negative control; sensitivity analysis |
| 14 | The analysis does not depend on the analyst's choices | Reproducibility | Researcher degrees of freedom in preprocessing | Multiverse analysis; pre-registration |

The limits of the topic are explicit:

- It does not cover causal identification. A difference in an observational comparison is an
  association until an identification strategy is stated; that is Module 07 and Module 10.
- It does not cover likelihood theory, generalised linear models, or survival analysis in depth.
- It does not cover Bayesian computation. It covers the updating rule and the role of the prior, not
  Markov chain Monte Carlo.
- It does not cover machine-learning validation. Discrimination, calibration, and decision-curve
  analysis in a predictive modelling context are Module 06.
- It does not make any claim about which prior is correct. It requires that the prior be stated.

## 12. Exercises

### Exercise 1 — Count the unit (application)

Take a dataset description with 48,000 rows described as "patient encounters from 6,200 patients
across 14 clinics." State the unit of observation, the number of independent units at the patient
level, the number at the clinic level, and the effective sample size for a patient-level outcome if
the intraclass correlation within patient is 0.45 and the average encounters per patient is 7.7.

**Deliverable.** A short memo with the four numbers and the design-effect calculation.

**Acceptance criteria.** Design effect computed as 1 + (7.7 − 1) × 0.45 = 4.015; effective n =
6,200/4.015 = 1,544, reported to the nearest whole patient; the memo states that the clinic level has
only 14 units and that clinic-level inference requires different methods.

**Time.** 20 minutes.

### Exercise 2 — Interval for a proportion (calculation)

A clinic reports 12 adverse events in 180 procedures.

Compute the point estimate, the standard error, the Wald 95% interval, and the Wilson 95% interval.
State which you would report and why.

**Deliverable.** Four values and a two-sentence justification.

**Acceptance criteria.** p = 6.67%; SE = 1.86 percentage points; Wald 3.02% to 10.31%; Wilson
approximately 3.86% to 11.27%; the justification references the moderate n and the asymmetry of the
Wilson interval.

**Time.** 25 minutes.

### Exercise 3 — Bootstrap by hand (calculation and design)

For the data in Exercise 2, describe how to construct a bootstrap percentile interval. State the
resampling unit, the number of resamples you would use, and one reason the bootstrap might disagree
with the Wilson interval.

**Deliverable.** A procedure of no more than 150 words plus the resampling unit.

**Acceptance criteria.** Resampling unit is the procedure, not the event; resamples at least 2,000;
the disagreement reason references small-sample bias or discreteness of a binary outcome.

**Time.** 20 minutes.

### Exercise 4 — Interpret the interval (critique)

A report states: "There is a 95% probability that the true mortality rate lies between 5.9% and
11.1%."

Explain why this sentence is not a correct frequentist interpretation, write a correct version, and
state the condition under which a probability statement of this form would be legitimate.

**Deliverable.** A 150-word critique with the corrected sentence.

**Acceptance criteria.** Identifies that the parameter is fixed, not random, under the frequentist
framework; the corrected sentence refers to the procedure's long-run coverage; the legitimate
condition names a Bayesian credible interval with a stated prior.

**Time.** 20 minutes.

### Exercise 5 — What a p-value is (recall and critique)

A colleague says: "p = 0.03 means there is a 3% chance the null hypothesis is true, so we should
reject it."

Identify two distinct errors in the statement and write a correct interpretation of p = 0.03 for a
two-sided test of no difference.

**Deliverable.** Two labelled errors and a corrected sentence.

**Acceptance criteria.** Error one is the inversion of the conditional probability; error two is the
conflation of the test result with a decision; the corrected sentence states that under the null, 3%
of studies would produce a statistic at least this extreme.

**Time.** 20 minutes.

### Exercise 6 — PPV across prevalences (calculation)

A test has sensitivity 0.92 and specificity 0.95. Compute PPV and NPV at prevalences of 1%, 5%, 20%,
and 50%, assuming 10,000 people at each prevalence. Compute LR+ and LR−.

**Deliverable.** A table with prevalence, TP, FP, FN, TN, PPV, NPV; plus the two likelihood ratios.

**Acceptance criteria.** At 5%: TP 460, FP 475, PPV 49.2%; at 20%: TP 1,840, FP 400, PPV 82.1%;
LR+ = 18.4; LR− = 0.0842; the table shows PPV rising monotonically with prevalence.

**Time.** 30 minutes.

### Exercise 7 — Audit an abstract (critique)

Find an abstract of a clinical prediction model. Identify: the estimand, the unit of observation, the
validation design, whether an interval is reported for the primary performance measure, whether
calibration is reported, and whether any subgroup claim is pre-specified.

**Deliverable.** A one-page audit with a finding for each of the six items and an overall verdict on
whether the abstract supports deployment.

**Acceptance criteria.** Each item has a quoted or paraphrased finding; the verdict distinguishes
"supports further evaluation" from "supports deployment"; at least one missing item is named
explicitly rather than inferred.

**Time.** 45 minutes.

### Exercise 8 — Write an analysis plan (design)

For a decision about whether to adopt a remote monitoring programme, write a one-page analysis plan
containing: the estimand, the unit of observation, the primary estimate and interval method, the
multiplicity policy for secondary outcomes, the handling of missing data, and the sensitivity
analyses.

**Deliverable.** A one-page plan with six labelled sections.

**Acceptance criteria.** The estimand names population, outcome, comparator, and window; the interval
method is justified for the outcome type; the multiplicity policy names the family; the missing-data
section states a mechanism and a sensitivity analysis; no section is left as "to be determined."

**Time.** 50 minutes.

## 13. Self-check answers

### Answer to Exercise 1

```text
Unit of observation: patient encounter (one row)
Independent units: 6,200 patients; 14 clinics
Average encounters per patient: 48,000 / 6,200 = 7.742

Design effect = 1 + (m - 1) x ICC
              = 1 + (7.742 - 1) x 0.45
              = 1 + 6.742 x 0.45
              = 1 + 3.034
              = 4.034

Effective n = 6,200 / 4.034 = 1,537 patients
```

The effective sample size is about 1,537, not 48,000. Any interval computed with n = 48,000 is too
narrow by a factor of √4.034 = 2.01. At the clinic level there are only 14 units, so a clinic-level
comparison has very low power and requires methods designed for few clusters.

### Answer to Exercise 2

```text
p = 12/180 = 0.066667 = 6.67%
SE = sqrt(0.066667 x 0.933333 / 180) = sqrt(0.00034568) = 0.018593 = 1.86 pp

Wald: 0.066667 +/- 1.96 x 0.018593
    = 0.066667 +/- 0.036442
    = 0.030225 to 0.103109 = 3.02% to 10.31%

Wilson: centre = (0.066667 + 3.8416/360) / (1 + 3.8416/180)
              = (0.066667 + 0.010671) / 1.021342
              = 0.075723
        half  = 1.96/1.021342 x sqrt(0.00034568 + 3.8416/129600)
              = 1.91904 x sqrt(0.00034568 + 0.00002964)
              = 1.91904 x 0.019386
              = 0.037203
        interval = 0.038520 to 0.112926 = 3.85% to 11.29%
```

Report the Wilson interval. At 12 events in 180 procedures the proportion is low enough that the
Wald interval is asymmetric in truth but symmetric in its formula, and the Wilson interval has better
coverage near the boundary. The two differ by about 0.8 percentage points at the lower limit, which
matters when the decision threshold is close to that limit.

### Answer to Exercise 3

Resample 180 procedures with replacement, compute the proportion of events in each resample, and
repeat at least 2,000 times. Take the 2.5th and 97.5th percentiles of the resampled proportions.

The resampling unit is the procedure, because procedures are the independent observations. Resampling
events and non-events separately would destroy the denominator and produce a meaningless interval.

The bootstrap may disagree with Wilson because with only 12 events the bootstrap distribution is
discrete and skewed, and the percentile method can under-cover at small n. A bias-corrected and
accelerated (BCa) interval is the usual remedy.

### Answer to Exercise 4

The sentence treats the parameter as a random variable. Under the frequentist framework the true
mortality rate is a fixed unknown number; it is either inside the realised interval or outside, and
the probability statement applies to the procedure, not to this interval.

Correct version: "The interval 5.9% to 11.1% was constructed by a procedure that contains the true
rate in about 95% of repeated samples of this design and size."

A probability statement about the parameter is legitimate in a Bayesian framework, where the
parameter has a posterior distribution given the data and a stated prior. It is then called a
credible interval, not a confidence interval.

### Answer to Exercise 5

Error one: p = 0.03 is not the probability that the null is true. It is the probability, computed
under the assumption that the null is true, of observing data at least as extreme as these.

Error two: rejecting the null is a decision, and the p-value alone does not determine it. The
decision depends on the effect size, the interval, the cost of errors, and the prior plausibility.

Correct interpretation: "If there were truly no difference, about 3% of studies of this design and
size would produce a difference at least this large in either direction. The observed difference is
therefore unusual under the null, but the p-value says nothing about the probability that the null is
true or about the size of the effect."

### Answer to Exercise 6

Sensitivity 0.92, specificity 0.95, LR+ = 0.92/0.05 = 18.4, LR− = 0.08/0.95 = 0.0842.

| Prevalence | Cases | Non-cases | TP | FN | FP | TN | PPV | NPV |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1% | 100 | 9,900 | 92 | 8 | 495 | 9,405 | 15.7% | 99.9% |
| 5% | 500 | 9,500 | 460 | 40 | 475 | 9,025 | 49.2% | 99.6% |
| 20% | 2,000 | 8,000 | 1,840 | 160 | 400 | 7,600 | 82.1% | 97.9% |
| 50% | 5,000 | 5,000 | 4,600 | 400 | 250 | 4,750 | 94.8% | 92.2% |

Check the 5% row: PPV = 460/(460 + 475) = 460/935 = 0.4920. The PPV moves from 15.7% to 94.8% across
prevalences with identical sensitivity and specificity. This is why a test's "accuracy" cannot be
quoted without the population.

### Answer to Exercise 7 (rubric)

| Criterion | Weight | What earns credit |
|---|---:|---|
| Estimand identified | 20% | Names population, outcome, comparator, and window or notes their absence |
| Unit of observation | 15% | States whether encounters, patients, or images are the unit and whether clustering is handled |
| Validation design | 20% | Distinguishes random split, temporal split, and external site |
| Interval reported | 15% | Quotes the interval or notes its absence for the primary measure |
| Calibration reported | 15% | Names calibration slope, intercept, or plot, or notes absence |
| Subgroup claims | 15% | States whether subgroups were pre-specified or exploratory |

A weak audit lists the model's AUROC and declares it good without checking the unit, the validation
design, or calibration.

### Answer to Exercise 8 (rubric)

| Criterion | Weight | What earns credit |
|---|---:|---|
| Estimand | 25% | Population, outcome, comparator, window, summary measure all present |
| Unit and variance | 20% | Unit stated; clustering addressed; variance method named |
| Interval method | 15% | Justified for the outcome type, with the level stated |
| Multiplicity | 15% | Family defined; correction named |
| Missing data | 15% | Mechanism stated; primary method and sensitivity analysis named |
| Sensitivity analyses | 10% | At least two named, each tied to an assumption in the ledger |

A weak plan states the outcome and the model but leaves the comparator, the interval method, and the
missing-data handling unspecified.

## 14. Interview questions

### Question 1 — "How many observations does this dataset have?"

**Strong answer.** "I would first ask what a row represents. If there are 48,000 encounters from
6,200 patients, the independent unit for a patient-level outcome is 6,200, and the effective sample
size is lower still once within-patient correlation is accounted for. I would report rows, units, and
the design effect."

**Weak answer.** "48,000." The row count is a property of the file, not of the evidence.

### Question 2 — "What does a 95% confidence interval mean?"

**Strong answer.** "It means the procedure that produced the interval covers the true value in about
95% of repeated samples. It does not mean there is a 95% probability that the parameter lies in this
particular interval. If the decision needs a probability statement, I would use a Bayesian credible
interval and state the prior."

**Weak answer.** "There is a 95% chance the true value is in the interval." This is the single most
common statistical misstatement in industry material.

### Question 3 — "The p-value is 0.04. What do you conclude?"

**Strong answer.** "That the data are somewhat unusual under the null. I would want the effect size,
the interval, the number of tests, and the decision threshold before concluding anything. A p of 0.04
on the fifteenth subgroup is not evidence of an effect."

**Weak answer.** "The result is significant, so the treatment works." Significance is a property of the
test, not of the treatment.

### Question 4 — "Why is p not the probability that the null is true?"

**Strong answer.** "Because the p-value is computed conditional on the null: P(data at least this
extreme | H0). The quantity people want is P(H0 | data), which requires a prior and Bayes' theorem.
Confusing the two is the prosecutor's fallacy."

**Weak answer.** "Because the null is either true or false." That is true under the frequentist
framework but does not explain the conditional inversion, which is the actual error.

### Question 5 — "A screening test is 95% sensitive and 95% specific. Is it good?"

**Strong answer.** "It depends on prevalence. At 1% prevalence the PPV is about 16%, so five of six
positives are false. At 20% prevalence the PPV is about 83%. I would report LR+ = 19 and recompute the
PPV in the population where it will be used."

**Weak answer.** "Yes, 95% is high." The question is unanswerable without the base rate.

### Question 6 — "What is the difference between a standard deviation and a standard error?"

**Strong answer.** "The standard deviation describes the spread of observations. The standard error
describes the spread of an estimate across repeated studies and is approximately SD/√n for
independent observations. A figure labelled mean ± SD and a figure labelled mean ± SE look similar and
mean different things."

**Weak answer.** "They are both measures of spread." True but useless; the distinction determines
whether a reader can recover the sample size and the precision.

### Question 7 — "You have 14 quality metrics and one improved with p = 0.03. What now?"

**Strong answer.** "The family-wise error rate across 14 independent tests is about 51%, so one
apparent improvement is expected. I would check whether the metric was pre-specified as primary, apply
a false discovery rate correction if the purpose is triage, and look for a dose-response or a
corroborating metric before acting."

**Weak answer.** "Investigate it as a possible improvement." Investigation is reasonable, but the
statement should be labelled exploratory, not a finding.

### Question 8 — "When would you use a bootstrap instead of a formula?"

**Strong answer.** "When the estimator is a complex function of the data, such as a ratio of fitted
quantities, a model-based AUC, or a cost-effectiveness ratio, or when the distribution is unknown and
the sample is small. The bootstrap must resample the independent unit, not the row, or the interval
will be too narrow."

**Weak answer.** "When the data are not normal." Normality of the data is not the criterion; the
criterion is whether the sampling distribution of the estimator is tractable and well approximated.

### Question 9 — "How do you handle multiple comparisons?"

**Strong answer.** "First, by pre-specifying the primary comparison and labelling the rest exploratory.
If a family of confirmatory tests is required, Bonferroni controls the family-wise error and is
appropriate when any false positive is costly. Benjamini-Hochberg controls the false discovery rate
and is appropriate when a proportion of false discoveries is tolerable. The choice and the family must
be stated in advance."

**Weak answer.** "I use Bonferroni." Bonferroni is often unnecessarily conservative for screening
applications, and applying it after seeing the data does not restore validity.

### Question 10 — "What is calibration and why does it matter more than AUROC for a threshold?"

**Strong answer.** "Calibration is whether predicted probabilities match observed frequencies. A
threshold converts a probability into an action, so a model that ranks well but predicts 0.3 when the
true rate is 0.1 will fire three times too often. I would report calibration slope, intercept, and a
reliability plot in the deployment population, and recalibrate before setting a threshold."

**Weak answer.** "AUROC is the main metric; calibration is a nice-to-have." For any decision that uses
a probability threshold, calibration is primary.

### Question 11 — "What sample size do you need for a 3-point margin of error on a proportion near 10%?"

**Strong answer.** "Using n = z²p(1−p)/w² with p = 0.10 and w = 0.03, n = 3.8416 × 0.09/0.0009 = 384,
so 385. If the design is clustered, I inflate by the design effect."

**Weak answer.** "About a thousand, to be safe." A number without a derivation cannot be checked or
defended.

### Question 12 — "What would make you distrust a reported effect?"

**Strong answer.** "No interval, an undeclared unit of observation, a random split presented as
external validation, an undeclared number of comparisons, or a calibration claim without a calibration
plot. Any one of these makes the effect uninterpretable for a decision."

**Weak answer.** "A small sample." Small samples widen intervals, which is visible; the failures that
matter are the ones that are invisible.

## 15. Key terms

- **ANOVA (analysis of variance)** — a decomposition of variance into components attributed to
  specified sources.
- **Base rate** — the prevalence of the condition in the population being tested.
- **Bayes' theorem** — P(D|T) = P(T|D)P(D)/P(T).
- **Bernoulli distribution** — a binary outcome with probability p.
- **Bias (statistical)** — expected estimate minus true value.
- **Binomial distribution** — the count of successes in n independent Bernoulli trials.
- **Bootstrap** — resampling with replacement to approximate a sampling distribution.
- **Calibration** — agreement between predicted probabilities and observed frequencies.
- **Clustering** — grouping of observations that induces correlation.
- **Confidence interval** — an interval with stated long-run coverage for the parameter.
- **Coverage** — the long-run proportion of intervals containing the true value.
- **Credible interval** — a Bayesian interval with stated posterior probability.
- **Design effect** — 1 + (m − 1)·ICC, the variance inflation from clustering.
- **Discrimination** — the ability of a model to rank cases above non-cases.
- **Effect size** — the magnitude of a difference or association in its natural unit.
- **Estimand** — the precise quantity a study intends to estimate.
- **False discovery rate (FDR)** — expected proportion of false rejections among rejections.
- **Family-wise error rate (FWER)** — probability of at least one false rejection in a family.
- **Intraclass correlation (ICC)** — the proportion of total variance attributable to clustering.
- **Likelihood ratio** — the probability of a result under disease divided by under no disease.
- **Marginal structural model** — a model for the marginal effect of a time-varying treatment.
- **Missing at random (MAR)** — missingness explained by observed variables.
- **Missing completely at random (MCAR)** — missingness independent of all variables.
- **Missing not at random (MNAR)** — missingness depends on the unobserved value.
- **Multiple imputation** — repeated imputation and pooling to account for missing-data uncertainty.
- **Negative predictive value (NPV)** — TN/(TN + FN).
- **Null hypothesis** — the reference value or model under which a test statistic is distributed.
- **Overdispersion** — variance greater than the nominal model implies.
- **p-value** — P(data at least as extreme | H0).
- **Poisson distribution** — counts in a fixed exposure with variance equal to the mean.
- **Posterior distribution** — the updated distribution of a parameter given data and prior.
- **Power** — probability of rejecting H0 under a specified alternative.
- **PPV (positive predictive value)** — TP/(TP + FP).
- **Prevalence** — the proportion of a population with the condition at a point or period.
- **Prior distribution** — belief about a parameter before observing data.
- **Regression to the mean** — extreme values tending toward the average on remeasurement.
- **Sensitivity** — TP/(TP + FN).
- **Specificity** — TN/(TN + FP).
- **Standard error** — the standard deviation of an estimator's sampling distribution.
- **Test statistic** — a quantity with a known distribution under H0.
- **TRIPOD** — reporting guideline for prediction models.
- **Type I error** — rejecting a true null.
- **Type II error** — failing to reject a false null.
- **Wilson interval** — a score-based confidence interval for a proportion.
- **Wald interval** — estimate ± z·SE; a symmetric normal approximation.

## 16. Sources

All URLs were accessed on 24 October 2026.

### Core statistical statements

- Wasserstein, R. L., Lazar, N. A., "The ASA Statement on p-Values: Context, Process, and Purpose",
  *The American Statistician* 70(2), 129–133, 2016. https://doi.org/10.1080/00031305.2016.1154108
- Open Science Collaboration, "Estimating the reproducibility of psychological science", *Science*
  349(6251), aac4716, 2015. https://doi.org/10.1126/science.aac4716
- Wilson, E. B., "Probable inference, the law of succession, and statistical inference", *Journal of
  the American Statistical Association* 22(158), 209–212, 1927.
  https://www.tandfonline.com/doi/abs/10.1080/01621459.1927.10502953
- Efron, B., "Bootstrap methods: another look at the jackknife", *Annals of Statistics* 7(1), 1–26,
  1979. https://doi.org/10.1214/aos/1176344552

### Laboratory standards

- CLSI EP05-A3, *Evaluation of Precision of Quantitative Measurement Procedures; Approved
  Guideline—Third Edition*, 2014. https://clsi.org/standards/products/method-evaluation/documents/ep05/
- CLSI EP15-A3, *User Verification of Precision and Estimation of Bias; Approved Guideline—Third
  Edition*, 2014. https://clsi.org/standards/products/method-evaluation/documents/ep15/
- CLSI EP28-A3c, *Defining, Establishing, and Verifying Reference Intervals in the Clinical
  Laboratory; Approved Guideline—Third Edition*, 2010. https://clsi.org/standards/products/method-evaluation/documents/ep28
- FDA recognition record for CLSI EP28-A3c, standard identification number 31787.
  https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfStandards/detail.cfm?standard__identification_no=31787

### Regulatory and reporting

- ICH, *E9(R1) Addendum on Estimands and Sensitivity Analysis in Clinical Trials*, 2019.
  https://www.ich.org/page/efficacy-guidelines
- ICH, *E9 Statistical Principles for Clinical Trials*, 1998. https://www.ich.org/page/efficacy-guidelines
- CONSORT Group, *CONSORT 2010 Statement*. https://www.consort-statement.org/consort-2010
- STROBE Initiative, *STROBE Statement*. https://www.strobe-statement.org/
- TRIPOD Group, *TRIPOD Statement*. https://www.tripod-statement.org/
- National Institute for Health and Care Excellence, *NICE health technology evaluations: the manual*.
  https://www.nice.org.uk/process/pmg36
- Institute for Clinical and Economic Review, *Value Assessment Framework*.
  https://icer.org/our-approach/methods-process/value-assessment-framework/

### Clinical evidence

- Gulshan, V. et al., "Development and Validation of a Deep Learning Algorithm for Detection of
  Diabetic Retinopathy in Retinal Fundus Photographs", *JAMA* 316(22), 2402–2410, 2016.
  https://doi.org/10.1001/jama.2016.17216

### Background texts

- Hernán, M. A., Robins, J. M., *Causal Inference: What If*.
  https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/
- Rothman, K. J., Greenland, S., Lash, T. L., *Modern Epidemiology*, 3rd edition.

> **Note:** CLSI documents are sold standards; the URLs are product pages. The content summarised here
> is from the publicly available front matter and preview material, and should be checked against the
> purchased document before use in a laboratory validation.

## 17. Links

- Up to the module map: [`_index.md`](_index.md)
- Previous topic: [`01_mathematical-basics.md`](01_mathematical-basics.md)
- Next topic: [`03_units-and-metrology.md`](03_units-and-metrology.md)
- Rates and measures of association: [`07_epidemiology-basics.md`](07_epidemiology-basics.md)
- Measurement error and precision: [`03_units-and-metrology.md`](03_units-and-metrology.md)
- Assumption testing: [`10_the-assumptions-ledger.md`](10_the-assumptions-ledger.md)
- Root survey file: [`../../01A_foundations_assumptions_and_units.md`](../../01A_foundations_assumptions_and_units.md)
- Labs: [`../../labs/`](../../labs/) — `06_calibration_decision_curve.ipynb` applies calibration and
  decision-curve analysis to the concepts in sections 6 and 9.
