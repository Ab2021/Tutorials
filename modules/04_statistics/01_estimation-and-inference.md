# Estimation and Inference

> Module 04 — Statistics and Causal Inference
> Topic 01 of 10
> Estimated study time: 3 hours
> Prerequisites: 01A
> Version 2026.3

---

## 1. Why this topic exists

A vendor presents a slide: "Our heart-failure program reduced 30-day readmissions by 31%." The number is a point estimate. It comes from a comparison of 412 enrolled patients and 388 patients who received usual care. The slide does not show the confidence interval, the denominator, the follow-up window, or how the 800 patients were selected. A hospital executive is being asked to commit USD 2.4 million per year for three years.

The decision cannot be made from the point estimate. It requires three separate quantities:

1. The size of the effect, on a scale that maps to the decision (readmissions avoided per 100 patients, or per USD spent).
2. The uncertainty around that effect, expressed as an interval that would be stable if the study were repeated.
3. The probability that an effect at least this large would have appeared even if the program did nothing.

These are the objects of estimation and inference. They are frequently conflated. A p-value below 0.05 is treated as proof of benefit; a confidence interval that includes zero is treated as proof of no benefit; a large relative reduction is reported without its absolute counterpart. Each of these errors changes what an organization buys, stops, or funds.

The industry cost is concrete:

- **False adoption.** A program with a 6-percentage-point absolute reduction and a confidence interval from 1.2 to 12.0 percentage points may be worth funding. A program with the same point estimate and an interval from -4 to 16 may not. The point estimate is identical; the decisions differ.
- **False rejection.** A trial with 120 events can miss a real 25% relative reduction because the interval is wide. Reporting "no effect" rather than "not enough evidence to rule out an important effect" misleads the next team.
- **Scale confusion.** An odds ratio of 0.64 and a risk ratio of 0.70 can describe the same data when the outcome is common. Presenting the odds ratio as if it were a risk ratio overstates the benefit.
- **Silent multiplicity.** A dashboard tests 40 hospital-level metrics; two cross p < 0.05. Without a multiplicity plan, the organization investigates noise.
- **Unstable intervals.** Ignoring clustering by hospital, or treating repeated admissions as independent, produces intervals that are too narrow and decisions that are too confident.

Regulators have made the first of these errors a review issue. The International Council for Harmonisation (ICH) guideline E9(R1), adopted at Step 4 on 20 November 2019, requires a trial to define its estimand before analysis: the treatment, population, endpoint, handling of intercurrent events, and population-level summary. The United States Food and Drug Administration (FDA) issued its version of the addendum on 11 May 2021. A sponsor that reports an estimate without a defined estimand cannot tell a reviewer what quantity the estimate refers to. The same logic applies to internal business cases, even when no regulator is watching.

This topic builds the minimum toolkit: define the quantity, estimate it, quantify its uncertainty, and connect the interval to a decision. Later topics assume this vocabulary. Regression (Topic 02) is a machine for producing estimates and standard errors. Survival analysis (Topic 03) changes the outcome scale. Causal topics (05 to 08) change the identification assumptions, not the arithmetic of uncertainty.

> **Note:** Estimation and inference are not the same activity. Estimation produces a best guess. Inference quantifies how wrong that guess could be under a model and a sampling design. A number without an interval is an estimate with the uncertainty hidden, not an estimate without uncertainty.

## 2. Learning outcomes

By the end of this topic you can:

1. **Write** a complete estimand for a comparative question, naming the five ICH E9(R1) attributes: treatment condition, population, endpoint, intercurrent-event strategy, and population-level summary.
2. **Compute** a standard error for a mean, a proportion, a difference in proportions, and a difference in means from summary data, and state the unit of each.
3. **Construct** and **interpret** a 95% confidence interval, including the correct frequentist statement about coverage and the incorrect statement about a fixed parameter.
4. **Explain** what a p-value is and **state** at least four things it is not, using the American Statistical Association (ASA) statement as the reference.
5. **Convert** between relative and absolute effect measures, and **calculate** the number needed to treat (NNT) with its interval.
6. **Choose** among Wald, Wilson, Newcombe, bootstrap, and cluster-robust intervals for a given data structure, and justify the choice.
7. **Critique** a reported effect estimate for missing estimand, missing interval, scale error, or ignored clustering.
8. **Design** a minimum detectable effect calculation for a planned study and explain how it changes the decision.

Each outcome maps to an exercise in Section 12 or a self-check item in Section 13. Outcome 4 maps to Exercise 3 and self-check items 4 and 5. Outcome 5 maps to Exercise 2 and the worked example.

## 3. The core mental model

The mental model has five layers. Draw it from the bottom up.

```text
  SUPERPOPULATION
  Distribution of (Y, T, X) in the population about which you want to speak
        |
        |  sampling / eligibility / observation
        v
  SAMPLE  (n patients, observed outcomes, measured covariates)
        |
        |  estimator (a formula chosen before seeing the data)
        v
  ESTIMATE  (one number, e.g. risk difference = -6.6 percentage points)
        |
        |  sampling distribution (hypothetical repetition of the whole study)
        v
  STANDARD ERROR  ->  INTERVAL  ->  DECISION
                         |
                         +-- sensitivity: what if the model or design is wrong?
```

Read the diagram as a chain of substitutions. The population quantity is unknown. The sample is one draw. The estimator is a rule that maps any sample to a number. The estimate is what the rule produced on this sample. The sampling distribution is the distribution of the rule's output across repeated samples. The standard error is its standard deviation. The interval is a region constructed so that, across repeated samples, it covers the population quantity at a stated rate.

### 3.1 The three questions before any number

Every estimate in this course answers three questions in order:

| Question | What it fixes | Example |
|---|---|---|
| What quantity, in what population, over what time? | The estimand | 30-day readmission risk difference among adults discharged with heart failure, program vs usual care |
| What identifies that quantity from these data? | The design and assumptions | Enrollment is not randomized; we assume no unmeasured confounding given age, sex, ejection fraction, renal function |
| How precise is the resulting number? | The standard error and interval | Risk difference -6.6 percentage points, 95% CI -12.0 to -1.2 |

Skipping question 1 makes question 3 meaningless, because there is no parameter to be uncertain about. This is the most common failure in industry analytics: a precise interval around an undefined quantity.

### 3.2 The sampling distribution is a thought experiment

The sampling distribution is not observed. It is the distribution you would see if you repeated the entire study, including eligibility, enrollment, treatment assignment, follow-up, and measurement. This matters because the standard error depends on the design, not only on the sample size:

- A simple random sample of 800 patients has a different sampling distribution from a cluster sample of 20 hospitals with 40 patients each.
- A trial with 800 patients and 20% loss to follow-up has a different sampling distribution from a trial with 800 patients and 2% loss.
- An estimate adjusted for a strongly prognostic baseline covariate has a narrower sampling distribution than an unadjusted estimate, even with the same n.

The formula for the standard error is a claim about that thought experiment. If the thought experiment does not match the data-generating process, the interval is wrong even when the arithmetic is right.

### 3.3 Coverage, not probability about a fixed parameter

A 95% confidence interval is constructed by a procedure that covers the true parameter in 95% of repeated samples. After you compute one interval, the parameter is either inside it or not; the 95% refers to the procedure, not to this interval.

Correct: "If we repeated this study many times, 95% of intervals built this way would contain the true risk difference."

Incorrect: "There is a 95% probability that the true risk difference lies in this interval."

The incorrect statement is a Bayesian statement that requires a prior. It is not what the frequentist interval means. In practice the difference is often small, but the distinction matters when the interval is used in a regulatory or contractual argument.

### 3.4 The p-value is a tail area, not a verdict

A p-value is the probability of observing a test statistic at least as extreme as the one computed, assuming the null hypothesis and the statistical model are true. It is a measure of compatibility between the data and the model, not a measure of the probability that the null is true, not the probability that the result will replicate, and not the size of the effect.

The ASA statement, released on 7 March 2016, lists six principles. Three are especially relevant to industry decisions:

- Principle 3: scientific conclusions and business or policy decisions should not be based only on whether a p-value passes a specific threshold.
- Principle 5: a p-value, or statistical significance, does not measure the size of an effect or the importance of a result.
- Principle 6: by itself, a p-value does not provide a good measure of evidence regarding a model or hypothesis.

The mental model is therefore asymmetric: the interval carries the effect size and the uncertainty; the p-value carries only a compatibility statement. If a report contains a p-value but no interval, the report has removed the information needed for the decision and kept the information that is easiest to threshold.

### 3.5 Where the decision enters

Statistical inference ends with an interval. The decision adds three things the interval does not contain:

1. **A threshold.** The smallest effect that would justify the cost, the risk, or the change in workflow. For a readmission program costing USD 2.4 million per year and preventing readmissions at USD 12,000 each, the break-even absolute reduction is 2.4M / (12,000 × annual discharges). If the program serves 3,000 discharges per year, break-even is 6.7 percentage points.
2. **A loss function.** The cost of adopting a useless program differs from the cost of rejecting a useful one. A decision rule that treats both errors as equally bad is a choice, not a default.
3. **A time horizon.** Effects that appear at 30 days may vanish or grow at one year. The interval must be computed on the horizon that matches the decision.

The worked example in Section 6 carries all three through.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or scale |
|---|---|---|---|
| Estimand | The precise quantity to be estimated, defined by treatment, population, endpoint, intercurrent-event strategy, and summary measure | Used as a synonym for "outcome" | Not a number; a specification |
| Estimator | The rule or formula that maps data to an estimate | Confused with the estimate itself | A function |
| Estimate | The numerical value produced by an estimator on one sample | Reported as if it were the truth | Same unit as the estimand |
| Standard error (SE) | The standard deviation of the estimator's sampling distribution | Used interchangeably with standard deviation of the outcome | Same unit as the estimate |
| Standard deviation (SD) | The spread of individual observations around their mean | Reported as SE, making the study look more precise | Same unit as the observation |
| Confidence interval (CI) | An interval produced by a procedure with a stated coverage probability | Interpreted as a probability that the parameter is inside it | Same unit as the estimate |
| Coverage | The long-run proportion of intervals that contain the true parameter | Assumed to hold exactly in small samples | Proportion, 0 to 1 |
| Null hypothesis | The model under which the p-value is computed, usually no effect | Treated as "the effect is zero" in the population without stating the model | A constraint on a parameter |
| Alternative hypothesis | The set of parameter values against which power is computed | Left vague, making power uninterpretable | A set of parameter values |
| p-value | Probability of a test statistic at least as extreme as observed, under the null and the model | Interpreted as P(null is true) or P(replication) | Probability, 0 to 1 |
| Significance level (alpha) | The pre-specified Type I error rate used for a decision rule | Changed after seeing the result | Probability, usually 0.05 |
| Type I error | Rejecting a true null hypothesis | Called "false positive" without stating the null | Probability |
| Type II error | Failing to reject a false null hypothesis | Ignored in power calculations | Probability |
| Power | Probability of rejecting the null when a specified alternative is true | Computed with an unrealistically large effect | Probability, usually 0.80 or 0.90 |
| Test statistic | A summary of the data whose distribution is known under the null | Chosen after inspecting the data | Dimensionless or in SE units |
| Sampling distribution | Distribution of an estimator across repeated samples from the same design | Treated as the distribution of the data | A distribution |
| Bootstrap | Resampling observed units to approximate the sampling distribution | Resampling rows when the unit is the hospital | Empirical distribution |
| Credible interval | Interval with stated posterior probability under a prior | Called a confidence interval | Same unit as the parameter |
| Risk difference (RD) | Difference in outcome proportions between groups | Reported without the baseline risk | Proportion, often percentage points |
| Risk ratio (RR) | Ratio of outcome proportions | Called "relative risk reduction" when the complement is meant | Unitless, positive |
| Odds ratio (OR) | Ratio of odds between groups | Interpreted as a risk ratio when the outcome is common | Unitless, positive |
| Hazard ratio (HR) | Ratio of instantaneous event rates under proportional hazards | Interpreted as a ratio of cumulative risks | Unitless, positive |
| Number needed to treat (NNT) | Reciprocal of the absolute risk reduction | Reported without a time horizon or interval | Number of patients |
| Minimal clinically important difference (MCID) | The smallest effect a patient or clinician would consider important | Confused with statistical significance | Same unit as the effect |
| Multiplicity | Testing several hypotheses without controlling error rates | Ignored when many subgroups are examined | A property of the testing plan |
| Clustering | Correlation of observations within hospitals, patients, or sites | Ignored in standard errors | Intraclass correlation, 0 to 1 |
| Design effect | Ratio of the variance under clustering to the variance under simple random sampling | Forgotten in sample-size calculations | Unitless, at least 1 |
| Non-inferiority margin | The largest loss of effect acceptable to declare non-inferiority | Set without clinical justification | Same unit as the effect |
| Delta method | First-order approximation of the variance of a function of estimates | Used where the function is highly nonlinear and n is small | Approximation |

Two conventions are used throughout this module:

- **Percentage points** are the unit of a risk difference. A change from 21.6% to 15.0% is a difference of 6.6 percentage points, which is a 31% relative reduction. Mixing the two is a common source of overstated benefit.
- **Unitless** measures (RR, OR, HR) are reported with their reference category and their scale. "OR 0.64" is incomplete without "program vs usual care, 30-day readmission."

> **Note:** The ICH E9(R1) glossary distinguishes an estimand (the quantity) from an estimate (its value on the data) and from an estimator (the method). Using these words precisely is not pedantry; it is the difference between a protocol that a regulator can review and a results slide that cannot be audited.

## 5. The mechanics, step by step

The procedure below runs from the decision to the reported interval. Each step states its input, its output, and the reason it exists. Steps 1 to 5 define the quantity. Steps 6 to 9 estimate it and quantify uncertainty. Steps 10 to 13 test the result and translate it into a decision.

Do not reorder steps 1 to 5 after seeing the data. The order is the protection against choosing a summary measure because it produced a favourable number.

### Step 1. Write the decision and the loss function

**Input:** A decision memo naming the action, the alternatives, the budget, and the time window. For a program: adopt, expand, continue, or stop, with annual cost and the number of eligible patients per year.

**Output:** One sentence of the form "We will choose between A and B for population P over horizon H, and we will act if the absolute effect exceeds X." Plus the cost of a false adoption and the cost of a false rejection.

**Why it exists:** The decision fixes the scale, the horizon, and the threshold. Without it, the analyst cannot tell whether a 1.5-percentage-point reduction is large or small, or whether a 30-day or one-year endpoint is relevant.

**Branch points:**

| Decision type | Summary measure that fits | What to avoid |
|---|---|---|
| Adopt or reject a program | Absolute risk difference, NNT | A relative measure alone |
| Compare two drugs for formulary | Risk difference plus cost per event avoided | Odds ratio presented as risk |
| Rank sites for quality improvement | Shrunken site effects with intervals | Raw ranking without uncertainty |
| Price a diagnostic | Incremental cost per true positive | Sensitivity alone |
| Set a monitoring threshold | Predictive value at the operating point | AUC alone |

**Edge case:** When the decision is irreversible and the cost of a false adoption is high, the threshold should be stricter than the break-even effect. State this explicitly rather than letting a default 0.05 stand in for a loss function.

### Step 2. Specify the estimand's five attributes

**Input:** The decision statement, the clinical question, and the planned data source.

**Output:** A five-row estimand specification that a reviewer could apply to a different dataset.

**Why it exists:** ICH E9(R1) requires that the quantity be defined before analysis, especially how intercurrent events are handled. An intercurrent event is an event after treatment initiation that affects either the interpretation or the existence of the measurements associated with the clinical question, such as discontinuation, rescue medication, or death.

| Attribute | Question to answer | Example |
|---|---|---|
| Treatment condition | Which strategies are compared, including dose and duration? | Structured transition program vs usual care |
| Population | Who is eligible, in which setting, with which prior treatments? | Adults discharged alive with a primary heart-failure diagnosis |
| Endpoint | Which variable, measured how, at what time? | All-cause readmission within 30 days of discharge |
| Intercurrent-event strategy | How are discontinuation, rescue, and death handled? | Treatment policy for discontinuation; death counts as failure |
| Population-level summary | Which summary, on which scale? | Difference in cumulative incidence at 30 days |

The five strategies for intercurrent events are not interchangeable. Choose one per event and justify it against the decision.

| Strategy | Definition | When it answers the decision | Effect on the estimate |
|---|---|---|---|
| Treatment policy | Use the outcome regardless of whether treatment was received or continued | The decision is "offer the strategy" and adherence is part of real-world performance | Usually conservative for efficacy, realistic for effectiveness |
| Hypothetical | Estimate the outcome that would be observed if the event had not occurred | The decision concerns the mechanism or the efficacy of full adherence | Requires assumptions that cannot be fully tested |
| Composite | The intercurrent event is counted as part of the outcome | The event is itself a meaningful failure | Can conflate efficacy with tolerability |
| While on treatment | Compare outcomes during treatment only | The decision concerns the effect while the drug is taken | Censoring at discontinuation can introduce selection bias |
| Principal stratum | Estimate the effect in the subgroup that would not experience the event | The decision targets a definable subpopulation | Requires strong assumptions; often wide intervals |

**Decision table for an intercurrent event with more than two possible strategies:**

| Event | If the event is a failure the decision wants to avoid | If the event is a tolerated side effect | If the event makes the outcome undefined |
|---|---|---|---|
| Treatment discontinuation | Treatment policy | While on treatment, with censoring analysis | Hypothetical |
| Rescue medication | Composite including rescue | Treatment policy for the primary endpoint | Hypothetical |
| Death | Composite (death = failure) | Not applicable | Principal stratum or while on treatment, with caution |
| Misdiagnosis discovered later | Exclude, but report the exclusion | Exclude | Exclude |

**Edge case:** Death is never a neutral censoring event for a non-fatal endpoint. If death prevents the endpoint, a treatment-policy estimand that counts death as failure and a while-on-treatment estimand that censors at death answer different questions and can reverse the sign of the effect.

### Step 3. Fix the population, time zero, and follow-up

**Input:** Eligibility criteria, the index date, the follow-up window, and the censoring rules.

**Output:** A cohort definition with an explicit time zero and a rule for each way a patient can leave the cohort.

**Why it exists:** Time zero is where the sampling distribution starts. If time zero differs between treatment groups, the comparison includes time during which one group could not have experienced the outcome. This is immortal time bias, and no standard error can repair it.

**Checks:**

- Every patient has exactly one time zero.
- Eligibility is assessed at or before time zero, using information available then.
- Treatment status is defined at time zero, not by a future prescription.
- Follow-up ends at the outcome, at the end of the window, at death (if a competing event), or at the last observed contact.

**Edge cases:**

| Edge case | Symptom | Remedy |
|---|---|---|
| Prevalent users | Patients already on treatment at cohort entry | New-user design; exclude or define a washout period |
| Immortal time | Treated patients must survive to receive treatment | Align time zero to treatment initiation or use time-varying exposure |
| Informative censoring | Censoring related to prognosis and treatment | Inverse probability of censoring weights; report the censoring model |
| Administrative censoring | Follow-up ends at a fixed calendar date | Censoring is unrelated to prognosis if the date is fixed; still report |
| Left truncation | Patients enter after the start of the risk period | Account for delayed entry in the risk set |

### Step 4. Choose the endpoint and the scale

**Input:** The endpoint definition, the measurement schedule, and the plausible range of the outcome.

**Output:** A variable type (binary, continuous, count, time-to-event) and a scale for analysis.

**Why it exists:** The scale determines which effect measures are valid and which standard-error formulas apply.

| Outcome type | Analysis scale | Default effect measure | Notes |
|---|---|---|---|
| Binary, common (at least 10% in a group) | Risk | Risk difference and risk ratio | Odds ratio diverges from risk ratio |
| Binary, rare (under 5%) | Risk | Risk ratio, odds ratio | Odds ratio approximates risk ratio |
| Continuous, roughly symmetric | Mean | Mean difference | Report SD and units |
| Continuous, skewed positive (cost, length of stay) | Log or mean with bootstrap | Ratio of geometric means or difference in means | A mean difference in cost can be dominated by outliers |
| Count | Rate | Rate ratio | Check overdispersion; negative binomial if variance exceeds mean |
| Time-to-event | Hazard or cumulative incidence | Hazard ratio or risk difference at a horizon | Proportional hazards must be checked |
| Ordinal | Cumulative logit or proportion in category | Odds ratio or difference in proportions | Report the full distribution |

**Why the scale matters:** An odds ratio is not collapsible. Even in a randomized trial with no confounding, the conditional odds ratio from a model with a prognostic covariate differs from the marginal odds ratio. This is a property of the measure, not a bias. The risk difference and the risk ratio are collapsible on their own scales. If the decision is about population impact, choose a collapsible measure or standardize the model predictions.

### Step 5. Choose the summary measure

**Input:** The decision threshold, the baseline risk, and the outcome scale.

**Output:** A primary measure and one supporting measure, declared before analysis.

**Why it exists:** Each measure answers a different question. Reporting one without the other hides either the population impact or the relative strength of the effect.

| Measure | Answers | Decision use | Failure mode |
|---|---|---|---|
| Risk difference (RD) | How many patients are affected per 100? | Budget, NNT, capacity planning | Sensitive to baseline risk; not portable across settings |
| Risk ratio (RR) | How many times more likely? | Etiology, mechanism, transportability | Hides the absolute size |
| Odds ratio (OR) | Ratio of odds | Case-control designs, logistic models | Inflated relative to RR when the outcome is common |
| Hazard ratio (HR) | Ratio of instantaneous rates | Time-to-event comparisons under proportional hazards | Not a ratio of cumulative risks; time-averaged |
| Mean difference | Absolute change in a continuous outcome | Clinical importance when the MCID is known | Unit-dependent; not comparable across scales |
| Standardized mean difference | Effect in SD units | Meta-analysis across scales | Not a clinical unit; easily over-interpreted |
| NNT | Patients treated per event avoided | Communication and budgeting | Direction is easy to misstate; time horizon must accompany it |

**Rule:** For a binary outcome, report the RD and the RR. Add the OR only when the analysis model is logistic and the OR is the model's natural output. Never report the OR alone for a common outcome and call it a risk.

### Step 6. Choose the estimator

**Input:** The estimand, the data structure, and the identification assumptions.

**Output:** A named estimator with its target parameter.

**Why it exists:** Different estimators target different quantities. A crude proportion difference, a regression coefficient, and an inverse-probability-weighted estimate can all be computed on the same data and answer different questions.

| Data structure | Estimator | Target | Key assumption |
|---|---|---|---|
| Randomized, no loss, binary outcome | Difference in proportions | Marginal RD | Randomization |
| Randomized, baseline covariates | Regression adjustment or standardization | Marginal or conditional RD | Model for the outcome |
| Observational, measured confounding | IPTW, matching, doubly robust | ATE or ATT | No unmeasured confounding, positivity |
| Observational, instrument available | Two-stage least squares | Local average treatment effect | Exclusion restriction, monotonicity |
| Clustered | Mixed model, GEE, or cluster bootstrap | Conditional or marginal | Correct cluster structure |
| Time-to-event | Kaplan-Meier, Cox, parametric | Survival or hazard | Censoring independent of prognosis |

The estimator must match the estimand's summary measure. A Cox model produces a hazard ratio; if the estimand is a 30-day risk difference, the hazard ratio must be standardized into absolute risks before it answers the question.

### Step 7. Compute the standard error

**Input:** The estimate, the sample size, the design, and the outcome distribution.

**Output:** A standard error in the same unit as the estimate, plus the method used.

**Why it exists:** The interval is the estimate plus or minus a multiple of the standard error. An incorrect standard error produces a confident wrong answer.

| Quantity | Standard error | Assumptions |
|---|---|---|
| Mean | s / sqrt(n) | Independent observations; finite variance |
| Proportion | sqrt(p(1-p)/n) | Independent Bernoulli observations |
| Difference in proportions | sqrt(p1(1-p1)/n1 + p0(1-p0)/n0) | Independent groups |
| Difference in means | sqrt(s1^2/n1 + s0^2/n0) | Independent groups; Welch degrees of freedom if variances differ |
| Log risk ratio | sqrt(1/a - 1/n1 + 1/c - 1/n0) | Non-zero cells; large-sample normal approximation |
| Log odds ratio | sqrt(1/a + 1/b + 1/c + 1/d) | Non-zero cells |
| Regression coefficient | Model-based or robust sandwich | Correct model or correct mean function plus independence |
| Any statistic, cluster design | Cluster bootstrap or design effect | Clusters sampled; enough clusters |

**Cluster adjustment:** When observations are clustered, multiply the naive variance by the design effect 1 + (m - 1) × ICC, where m is the average cluster size and ICC is the intraclass correlation coefficient. With 40 patients per hospital and an ICC of 0.02, the design effect is 1 + 39 × 0.02 = 1.78, and the standard error is 1.33 times larger. The interval is 33% wider, and a result that was significant can become inconclusive.

### Step 8. Construct the interval

**Input:** The estimate, the standard error, the degrees of freedom or resampling scheme, and the confidence level.

**Output:** A two-sided interval on the reporting scale, with the method named.

**Why it exists:** The interval is the decision-relevant object. It states the range of effects compatible with the data and the model.

| Situation | Method | Reason |
|---|---|---|
| Mean, n at least 30, roughly symmetric | t interval with n-1 degrees of freedom | Accounts for estimated SD |
| Mean, n under 30, roughly symmetric | t interval; check symmetry | Normal approximation may fail |
| Mean, skewed | Bootstrap percentile or BCa; or log transform | Asymmetry in the sampling distribution |
| Proportion, n at least 40, p between 0.1 and 0.9 | Wald or Wilson | Wald is simple; Wilson is better near the boundaries |
| Proportion, p near 0 or 1, or n small | Wilson or exact (Clopper-Pearson) | Wald can extend outside 0 to 1 |
| Difference in proportions | Newcombe hybrid score or Wald | Newcombe behaves better with sparse cells |
| Ratio (RR, OR, HR) | Compute on the log scale, exponentiate | Symmetry and positivity |
| NNT | Invert the bounds of the RD interval | Preserves coverage; direction must be handled |
| Clustered | Cluster bootstrap | Robust to within-cluster correlation |
| Survival probability | Greenwood variance with log-log transform | Bounded and asymmetric |

**Wilson score interval for a proportion:**

```text
centre = (2np + z^2) / (2(n + z^2))
half   = z * sqrt(4np(1-p) + z^2) / (2(n + z^2))
interval = centre +/- half
```

**Newcombe hybrid score interval for a difference in proportions:** compute Wilson intervals for p1 and p0, then

```text
lower = (p1 - p0) - sqrt((p1 - l1)^2 + (u0 - p0)^2)
upper = (p1 - p0) + sqrt((u1 - p1)^2 + (p0 - l0)^2)
```

where (l1, u1) and (l0, u0) are the Wilson limits for each group. This interval is not exactly symmetric around the point estimate, which is a feature, not an error.

**Edge cases:**

- **Zero events in one arm:** the log risk ratio is undefined. Add 0.5 to each cell only as a sensitivity analysis, and report the risk difference, which remains defined.
- **Zero events in both arms:** report the RD and its interval; do not report a ratio.
- **Proportion at 0 or 1:** Wald collapses to a zero-width interval. Use Wilson or exact.
- **Small n with a continuous outcome:** use t, not z. The difference between 1.96 and 2.2 at 10 degrees of freedom is material.
- **Ratio measures with sparse cells:** the log-scale normal approximation is poor; use bootstrap or exact methods.

### Step 9. Compute a p-value only if a null is meaningful

**Input:** The estimate, the standard error, and the null value.

**Output:** A two-sided p-value, with the test named.

**Why it exists:** A p-value is a compatibility measure. It is optional. Many decisions are better served by the interval and the threshold.

The relation between the interval and the test is exact for Wald intervals: a 95% interval excludes the null exactly when the two-sided p-value is below 0.05. If a report shows a significant p-value and an interval that includes the null, one of the two is wrong.

**When to omit the p-value:** when the null is not a plausible decision state (for example, no effect of a program that must be compared against a cost threshold), or when the estimand is a descriptive quantity with no null.

**When the p-value adds information:** when the null is a regulatory or contractual boundary, such as superiority at 0, or non-inferiority at a pre-specified margin.

### Step 10. Run diagnostics before interpreting

**Input:** The data, the fitted model, and the interval.

**Output:** A list of checks with pass, fail, or unresolved, and the action taken.

**Why it exists:** Every interval depends on assumptions. Diagnostics detect the cases where the assumptions fail badly enough to change the conclusion.

| Check | What it detects | Action if it fails |
|---|---|---|
| Counts sum to denominators | Data-entry or definition error | Reconcile before analysis |
| Interval width equals 2 × z × SE | Arithmetic or software error | Recompute |
| Point estimate lies inside the interval | Coding error | Recompute |
| Two interval methods agree in large samples | Implementation error | Investigate the discrepancy |
| Events per variable in a regression | Overfitting | Reduce covariates or use penalization |
| Residual plots or calibration | Model misspecification | Add terms or change the model |
| Influence of individual observations | Results driven by a few patients | Report with and without |
| Sensitivity to missing-data assumption | Informative missingness | Report a tipping-point analysis |
| Cluster-robust versus naive SE | Ignored clustering | Use the cluster-robust result |
| Multiplicity | False discoveries | Pre-specify or adjust |

**Edge cases that change the arithmetic:**

- **Sparse events:** with fewer than 10 events per covariate, regression estimates are unstable. Use fewer covariates, Firth penalization, or exact methods.
- **Separation:** a covariate perfectly predicts the outcome, so the logistic coefficient diverges. Report the separation, use penalization, or drop the covariate.
- **Zero cells:** see Step 8.
- **Ties in time:** survival methods need a tie-handling rule (Breslow, Efron, exact). The choice matters when many events occur at the same recorded time.
- **Informative censoring:** censoring that depends on prognosis biases survival estimates. Use inverse probability of censoring weights or a sensitivity analysis.
- **Positivity violations:** in causal comparisons, some covariate strata have no treated or no control patients. The effect in those strata is not estimable. Trim, overlap-weight, or change the target population.

### Step 11. Translate the estimate into decision units

**Input:** The RD, the baseline risk, the size of the eligible population, and the unit costs.

**Output:** Events avoided per year, NNT, budget impact, and break-even effect.

**Why it exists:** Decision-makers allocate budgets, not log-odds.

| Quantity | Formula | Example |
|---|---|---|
| NNT | 1 / absolute RD | 1 / 0.066 = 15.2, reported as 16 |
| Events avoided per year | RD × eligible patients per year | 0.066 × 3,000 = 198 readmissions |
| Gross saving | Events avoided × cost per event | 198 × USD 12,000 = USD 2.376 million |
| Net benefit | Gross saving - program cost | 2.376M - 2.4M = -USD 24,000 |
| Break-even RD | Program cost / (eligible × cost per event) | 2.4M / (3,000 × 12,000) = 0.0667 |
| Break-even NNT | 1 / break-even RD | 15.0 |

The example is deliberately at the margin: the point estimate is almost exactly the break-even effect. That is the situation in which uncertainty matters most, and in which a binary "significant or not" decision rule is least useful.

### Step 12. Pre-specify sensitivity analyses and multiplicity control

**Input:** The primary analysis, a list of plausible departures from its assumptions, and the number of hypotheses.

**Output:** A short list of sensitivity analyses, each tied to one assumption, and a multiplicity strategy.

**Why it exists:** A result that survives only one assumption set is fragile. A result that was found after testing many outcomes is likely to be a selection artifact.

| Departure | Sensitivity analysis | What a change in conclusion means |
|---|---|---|
| Unmeasured confounding | E-value or bias adjustment | The effect could be explained by a moderate confounder |
| Missing outcome | Tipping-point analysis | A small amount of informative missingness reverses the result |
| Clustering | Cluster-robust or cluster bootstrap | The interval was too narrow |
| Model misspecification | Alternative functional form | The result depends on the model, not the data |
| Outcome definition | Second definition from a different code set | The result is definition-dependent |
| Population definition | Broader or narrower eligibility | The result does not transport |

For multiplicity, pre-specify the family of hypotheses. If the family is one primary endpoint, no adjustment is needed. If the family is 12 secondary endpoints, use a gatekeeping or Holm procedure and report adjusted intervals. If the analysis is exploratory, label it exploratory and do not attach confirmatory language.

### Step 13. Report the estimate, the interval, the estimand, and the decision

**Input:** The primary result, the diagnostics, the sensitivity analyses, and the decision threshold.

**Output:** A results paragraph and a one-page decision summary.

**Why it exists:** A number without its estimand and interval cannot be audited, reproduced, or used.

The reporting checklist:

1. The estimand, in one sentence with all five attributes.
2. The estimator and the software, with version.
3. The estimate, with units.
4. The 95% interval, with the method named.
5. The number of observations and the number of events per group.
6. The absolute and relative measures, both.
7. The p-value only if a null is meaningful, labelled as two-sided or one-sided.
8. The diagnostics that passed and the ones that failed.
9. The sensitivity analyses, with the assumption each tests.
10. The decision threshold and whether the interval crosses it.

### Branch points at a glance

| If the situation is | Then the default is | Because |
|---|---|---|
| Binary outcome, common | RD + RR, Newcombe interval | OR overstates the ratio |
| Binary outcome, rare | RR or OR, log-scale interval | RD is tiny and hard to communicate |
| Continuous, symmetric | Mean difference, t interval | Efficient and exact under normality |
| Continuous, skewed | Median or geometric mean, bootstrap | Mean may not represent a typical patient |
| Count, overdispersed | Negative binomial rate ratio | Poisson variance is too small |
| Time-to-event, proportional hazards | Cox HR plus standardized risks | HR alone is not a risk |
| Time-to-event, non-proportional | Restricted mean survival time difference | HR is time-averaged and misleading |
| Clustered data | Cluster-robust or cluster bootstrap | Independence is false |
| Many hypotheses | Pre-specified family with Holm or gatekeeping | Type I error inflates |
| Decision at a threshold | Interval against the threshold, not against zero | Zero is rarely the decision boundary |

## 6. Worked example

### 6.1 The decision and the estimand

A health system is deciding whether to continue a heart-failure transition program. The program assigns a nurse navigator, a pharmacist medication review, and a 7-day follow-up call. It costs USD 2.4 million per year and serves about 3,000 discharges per year. The alternative is usual care. The decision is whether to renew for three years.

The break-even absolute reduction is 2.4M / (3,000 × 12,000) = 0.0667, or 6.7 percentage points, assuming each avoided readmission saves USD 12,000.

The estimand:

| Attribute | Specification |
|---|---|
| Treatment condition | Structured transition program initiated before discharge vs usual care |
| Population | Adults discharged alive with a primary diagnosis of heart failure, 2019 to 2024, in 20 hospitals |
| Endpoint | All-cause inpatient readmission within 30 days of discharge |
| Intercurrent-event strategy | Treatment policy: readmission counts regardless of program adherence; death before readmission counts as a failure |
| Population-level summary | Difference in 30-day cumulative incidence, program minus usual care |

The analysis is observational. Patients were not randomized. The identification assumption is no unmeasured confounding given age, sex, ejection fraction, estimated glomerular filtration rate (eGFR), prior-year admissions, and hospital. That assumption is examined in Sections 6.5 and 9, not assumed away.

### 6.2 The data

| Group | Readmitted within 30 days | Not readmitted | Total | 30-day risk |
|---|---:|---:|---:|---:|
| Program | 62 | 350 | 412 | 15.05% |
| Usual care | 84 | 304 | 388 | 21.65% |
| Total | 146 | 654 | 800 | 18.25% |

### 6.3 Arithmetic, step by step

**Step 1. Group risks.**

```text
p1 = 62 / 412 = 0.150485... = 15.05%
p0 = 84 / 388 = 0.216495... = 21.65%
```

**Step 2. Risk difference.**

```text
RD = p1 - p0 = 0.150485 - 0.216495 = -0.066009
   = -6.60 percentage points
```

The program is associated with 6.60 fewer readmissions per 100 patients. The point estimate is just below the 6.67-percentage-point break-even threshold.

**Step 3. Standard error of the risk difference.**

```text
SE = sqrt( p1(1-p1)/n1 + p0(1-p0)/n0 )
   = sqrt( 0.150485 × 0.849515 / 412 + 0.216495 × 0.783505 / 388 )
   = sqrt( 0.0003103 + 0.0004372 )
   = sqrt( 0.0007475 )
   = 0.027340
```

The standard error is 2.73 percentage points. This is the unit of the sampling variability, not the spread of individual patients.

**Step 4. 95% Wald interval.**

```text
RD +/- 1.96 × SE = -0.066009 +/- 1.96 × 0.027340
                 = -0.066009 +/- 0.053586
                 = (-0.119596, -0.012423)
                 = (-11.96, -1.24) percentage points
```

The interval excludes zero. The data are compatible with a benefit as large as 11.96 percentage points and as small as 1.24 percentage points.

**Step 5. Test statistic and p-value.**

```text
z = RD / SE = -0.066009 / 0.027340 = -2.4144
two-sided p = 2 × (1 - Phi(2.4144)) = 0.0158
```

The chi-square test without continuity correction gives 5.836 and p = 0.0157. The two p-values differ in the third decimal because the z test uses unpooled variances and the chi-square test uses pooled variances. The difference is a useful diagnostic: large discrepancies signal sparse data or a coding error.

**Step 6. Risk ratio.**

```text
RR = p1 / p0 = 0.150485 / 0.216495 = 0.6951
SE(log RR) = sqrt(1/62 - 1/412 + 1/84 - 1/388) = 0.15175
95% CI = exp( log(0.6951) +/- 1.96 × 0.15175 )
       = exp( -0.36375 +/- 0.29745 )
       = (0.5163, 0.9359)
```

The program is associated with a 30.5% relative reduction (RR 0.695), with an interval from 6.4% to 48.4% relative reduction.

**Step 7. Odds ratio.**

```text
OR = (62/350) / (84/304) = 0.177143 / 0.276316 = 0.6411
SE(log OR) = sqrt(1/62 + 1/350 + 1/84 + 1/304) = 0.18488
95% CI = exp( -0.44451 +/- 1.96 × 0.18488 )
       = exp( -0.44451 +/- 0.36237 )
       = (0.4462, 0.9211)
```

The OR of 0.641 is 7.8% further from 1 than the RR of 0.695, because the outcome is common (21.65% in the control group). Reporting the OR as "a 36% reduction" would overstate the risk reduction by about 5.5 percentage points on the relative scale.

**Step 8. NNT and its interval.**

```text
NNT = 1 / |RD| = 1 / 0.066009 = 15.15, reported as 16 patients
95% CI for NNT = (1 / 0.119596, 1 / 0.012423) = (8.4, 80.5)
```

Report the NNT as "16 patients (95% CI 9 to 80)" with the 30-day horizon. An NNT without a horizon is uninterpretable, because the same program may have a different NNT at 90 days.

**Step 9. Decision arithmetic.**

```text
Events avoided per year = 0.066009 × 3,000 = 198.0
Gross saving = 198.0 × USD 12,000 = USD 2,376,000
Net benefit = USD 2,376,000 - USD 2,400,000 = -USD 24,000
```

At the point estimate, the program is USD 24,000 per year short of break-even. At the lower confidence limit (1.24 percentage points), the gross saving is 0.0124 × 3,000 × 12,000 = USD 446,400, far below cost. At the upper limit (11.96 percentage points), the gross saving is USD 4.31 million, a clear net benefit.

### 6.4 Diagnostic checks

**Check 1. Counts reconcile.** 62 + 350 = 412; 84 + 304 = 388; 146 + 654 = 800. The overall risk is 146/800 = 18.25%, which lies between the group risks. If it did not, the table would be wrong.

**Check 2. Interval width equals 2 × z × SE.** (0.119596 - 0.012423) / 3.92 = 0.027340, which equals the computed SE. This check detects a mis-specified standard error or a software default that used a different confidence level.

**Check 3. Alternative interval agrees.** The Newcombe hybrid score interval is (-0.119718, -0.012381), a difference of 0.0001 from the Wald interval. With 800 patients and cell counts above 60, agreement is expected. A material disagreement would indicate sparse cells and would favour the Newcombe interval.

**Check 4. Point estimate lies inside the interval and the interval is on the correct side of zero.** The RD is negative, and both limits are negative. If the point estimate were outside its own interval, the software output would be misread.

**Check 5. Effect measure consistency.** The RR and OR both lie below 1, and the OR is farther from 1 than the RR, as expected for a common outcome. If the OR were closer to 1 than the RR, the coding of the outcome or the group variable would be suspect.

### 6.5 Vary one assumption: 5% of the program arm is lost to follow-up

Suppose 5% of the 412 program patients (20.6 patients) have no 30-day readmission data, and the analysis counts all of them as readmissions. This is a worst-case imputation for the program arm.

```text
New program events = 62 + 0.05 × 412 = 62 + 20.6 = 82.6
New p1 = 82.6 / 412 = 0.200485 = 20.05%
New RD = 0.200485 - 0.216495 = -0.016009 = -1.60 percentage points
New SE = sqrt(0.200485 × 0.799515 / 412 + 0.216495 × 0.783505 / 388)
       = sqrt(0.000389 + 0.000437) = 0.028744
New 95% CI = -0.016009 +/- 1.96 × 0.028744
           = (-0.072348, 0.040329) = (-7.23, +4.03) percentage points
New p = 0.58
```

The conclusion changes from "statistically significant benefit" to "inconclusive." The interval now includes both a clinically important benefit and a clinically important harm. The arithmetic is transparent: counting 5% of the program arm as events raises the program risk by exactly 5 percentage points, which is the entire point estimate minus 1.6 points.

This is not a reason to distrust the original result. It is a reason to report the amount of missing outcome data and to pre-specify how it will be handled. If the missingness is plausibly unrelated to readmission, the original analysis stands; if the missing patients are systematically sicker or healthier, the tipping point matters.

### 6.6 Vary a second assumption: clustering by hospital

The 800 patients are treated at 20 hospitals, with an average of 40 patients per hospital. Readmission practices differ by hospital, so observations within a hospital are correlated. Assume an ICC of 0.02, a modest value for a process measure.

```text
Design effect = 1 + (40 - 1) × 0.02 = 1.78
Adjusted SE = 0.027340 × sqrt(1.78) = 0.027340 × 1.334 = 0.036476
95% CI = -0.066009 +/- 1.96 × 0.036476
       = (-0.137502, +0.005483) = (-13.75, +0.55) percentage points
p = 0.070
```

The interval now crosses zero. The point estimate is unchanged, but the uncertainty is larger because the effective sample size is smaller. With an ICC of 0.05, the design effect is 2.95, the interval is (-15.80, +2.60) percentage points, and p = 0.16.

The correct analysis uses a cluster-robust standard error or a cluster bootstrap, and the sample-size calculation should have included the design effect. The naive interval is not wrong because the arithmetic failed; it is wrong because the sampling distribution was mis-specified.

### 6.7 What to report and why

Report the risk difference as the primary measure: **-6.6 percentage points (95% CI -12.0 to -1.2), p = 0.016**, with the 30-day horizon. Report the RR as **0.70 (95% CI 0.52 to 0.94)** and the NNT as **16 (95% CI 9 to 80)**. State the estimand and the identification assumption. Report the cluster-robust interval as the primary interval if the trial was cluster-based or if the hospital effect is part of the design; report the naive interval as a sensitivity analysis only if clustering was ignored in the design.

The estimate to report is the risk difference because the decision is a budget decision at a 6.7-percentage-point threshold. The RR alone cannot be compared with that threshold. The NNT is the communication form of the same quantity. The p-value is secondary and should not appear without the interval.

The honest conclusion is conditional: "Under the assumption of no unmeasured confounding, the program is associated with a 6.6-percentage-point absolute reduction in 30-day readmission. The confidence interval includes effects below the break-even threshold and effects well above it. The decision depends on the cost of the program, the cost per readmission, and the credibility of the confounding assumption. A cluster-robust analysis that accounts for hospital-level correlation widens the interval to include no effect."

## 7. Code, query, or calculation

The script below reproduces every quantity in Section 6 using only the Python standard library. It also implements a t interval from the density function, so the reader can see that the 1.96 multiplier is not universal, and it runs a small coverage simulation to show why the Wald interval fails near the boundary.

**Assumptions of the script:**

- The two groups are independent simple random samples.
- The outcome is binary and measured without error.
- The normal approximation is adequate for the Wald interval at these cell counts.
- The simulation uses a fixed seed, so the printed coverage values are reproducible.
- The t interval assumes an approximately normal underlying distribution for the paired differences.

```python
import math, random
from statistics import NormalDist

ND = NormalDist()
z = ND.inv_cdf(0.975)

# --- 1. Two-by-two table ---
n1, e1 = 412, 62
n0, e0 = 388, 84
p1, p0 = e1 / n1, e0 / n0
rd = p1 - p0
se = math.sqrt(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0)
lo, hi = rd - z * se, rd + z * se
pval = 2 * (1 - ND.cdf(abs(rd / se)))
print(f"p1={p1:.4f} p0={p0:.4f} rd={rd:.4f} se={se:.4f}")
print(f"RD 95% CI: ({lo:.4f}, {hi:.4f})  p={pval:.4f}")

# --- 2. Wilson intervals and Newcombe difference ---
def wilson(k, n, z=z):
    p = k / n
    den = 2 * (n + z * z)
    centre = (2 * n * p + z * z) / den
    half = z * math.sqrt(4 * n * p * (1 - p) + z * z) / den
    return centre - half, centre + half

l1, u1 = wilson(e1, n1)
l0, u0 = wilson(e0, n0)
nlo = rd - math.sqrt((p1 - l1) ** 2 + (u0 - p0) ** 2)
nhi = rd + math.sqrt((u1 - p1) ** 2 + (p0 - l0) ** 2)
print(f"Wilson p1: ({l1:.4f}, {u1:.4f})  Wilson p0: ({l0:.4f}, {u0:.4f})")
print(f"Newcombe RD 95% CI: ({nlo:.4f}, {nhi:.4f})")

# --- 3. Ratio measures on the log scale ---
rr = p1 / p0
se_rr = math.sqrt(1 / e1 - 1 / n1 + 1 / e0 - 1 / n0)
orr = (e1 / (n1 - e1)) / (e0 / (n0 - e0))
se_or = math.sqrt(1 / e1 + 1 / (n1 - e1) + 1 / e0 + 1 / (n0 - e0))
print(f"RR={rr:.4f} CI=({math.exp(math.log(rr)-z*se_rr):.4f}, {math.exp(math.log(rr)+z*se_rr):.4f})")
print(f"OR={orr:.4f} CI=({math.exp(math.log(orr)-z*se_or):.4f}, {math.exp(math.log(orr)+z*se_or):.4f})")

# --- 4. NNT ---
print(f"NNT={1/abs(rd):.2f} CI=({1/abs(lo):.2f}, {1/abs(hi):.2f})")

# --- 5. t interval implemented from the density (no scipy) ---
def t_pdf(x, df):
    c = math.exp(math.lgamma((df + 1) / 2) - math.lgamma(df / 2)) / math.sqrt(df * math.pi)
    return c * (1 + x * x / df) ** (-(df + 1) / 2)

def t_cdf(x, df):
    # integrate the density on a grid; adequate for a teaching check
    if x == 0:
        return 0.5
    sign = 1 if x > 0 else -1
    a, b, m = 0.0, abs(x), 4000
    h = (b - a) / m
    s = t_pdf(a, df) + t_pdf(b, df)
    for i in range(1, m):
        s += (4 if i % 2 else 2) * t_pdf(a + i * h, df)
    return 0.5 + sign * (h / 3) * s

def t_quantile(q, df):
    lo, hi = 0.0, 20.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if t_cdf(mid, df) < q:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2

diffs = [4.1, 6.2, 3.3, 5.5, 7.0, 2.8, 5.9, 4.4, 6.6, 3.9]
m = sum(diffs) / len(diffs)
s = math.sqrt(sum((x - m) ** 2 for x in diffs) / (len(diffs) - 1))
se_m = s / math.sqrt(len(diffs))
tq = t_quantile(0.975, len(diffs) - 1)
print(f"mean diff={m:.3f} sd={s:.3f} se={se_m:.3f} t_.975={tq:.3f}")
print(f"t 95% CI: ({m-tq*se_m:.3f}, {m+tq*se_m:.3f})  z-based: ({m-z*se_m:.3f}, {m+z*se_m:.3f})")

# --- 6. Coverage simulation: Wald vs Wilson, n=50, p=0.2 ---
random.seed(20261025)
true_p, n, reps = 0.2, 50, 2000
cov_wald = cov_wilson = 0
for _ in range(reps):
    k = sum(random.random() < true_p for _ in range(n))
    phat = k / n
    se_hat = math.sqrt(phat * (1 - phat) / n)
    if phat - z * se_hat <= true_p <= phat + z * se_hat:
        cov_wald += 1
    wl, wh = wilson(k, n)
    if wl <= true_p <= wh:
        cov_wilson += 1
print(f"coverage n=50 p=0.2 over {reps} reps: Wald={cov_wald/reps:.3f} Wilson={cov_wilson/reps:.3f}")

# --- 7. Design effect ---
for icc in (0.02, 0.05):
    de = 1 + 39 * icc
    print(f"ICC={icc:.2f} design effect={de:.2f} SE multiplier={math.sqrt(de):.3f} "
          f"CI=({rd-z*se*math.sqrt(de):.4f}, {rd+z*se*math.sqrt(de):.4f})")
```

**Expected output:**

```text
p1=0.1505 p0=0.2165 rd=-0.0660 se=0.0273
RD 95% CI: (-0.1196, -0.0124)  p=0.0158
Wilson p1: (0.1192, 0.1882)  Wilson p0: (0.1784, 0.2601)
Newcombe RD 95% CI: (-0.1197, -0.0124)
RR=0.6951 CI=(0.5163, 0.9359)
OR=0.6411 CI=(0.4462, 0.9211)
NNT=15.15 CI=(8.36, 80.49)
mean diff=4.970 sd=1.459 se=0.461 t_.975=2.262
t 95% CI: (3.926, 6.014)  z-based: (4.066, 5.874)
coverage n=50 p=0.2 over 2000 reps: Wald=0.933 Wilson=0.950
ICC=0.02 design effect=1.78 SE multiplier=1.334 CI=(-0.1375, 0.0055)
ICC=0.05 design effect=2.95 SE multiplier=1.718 CI=(-0.1580, 0.0260)
```

**What the output shows:**

- The Wald and Newcombe intervals agree to four decimal places, as they should with 62 to 84 events per arm. The agreement is a check, not evidence that either method is always correct.
- The OR is farther from 1 than the RR (0.641 versus 0.695), which is the expected direction when the control risk is 21.65%.
- The NNT interval is asymmetric: 8.4 to 80.5. NNT intervals are always asymmetric when inverted from a symmetric RD interval, and they must be reported as a range, never as a single number with a symmetric plus-or-minus.
- The t multiplier for 9 degrees of freedom is 2.262, not 1.96. The t interval (3.93, 6.01) is 21% wider than the z interval (4.07, 5.87). With small samples, using 1.96 is an avoidable overconfidence.
- The coverage simulation shows the Wald interval covering 93.3% of the time and the Wilson interval covering 95.0% at n = 50 and p = 0.2. The nominal 95% is not achieved by Wald at this sample size. This is the empirical justification for the method table in Step 8.
- The design-effect lines reproduce the clustering sensitivity in Section 6.6.

> **Note:** The simulation seed is fixed so that the printed coverage is exactly reproducible on any Python 3.8 or later interpreter. Changing the seed changes the third decimal, not the conclusion.

## 8. Data, standards, and regulatory context

### 8.1 Binding requirements and conventions

| Instrument | Status | What it constrains | What it does not constrain |
|---|---|---|---|
| ICH E9, Statistical Principles for Clinical Trials (Step 4, 5 February 1998) | Binding in the EU, Japan, and US through regional adoption | Pre-specification, analysis sets, confidence intervals, multiplicity, subgroup handling | Specific estimators or software |
| ICH E9(R1) addendum on estimands and sensitivity analysis (Step 4, 20 November 2019) | Binding through regional adoption; EU effective 30 July 2020 | The five estimand attributes, intercurrent-event strategies, sensitivity versus supplementary analyses | The choice of statistical model |
| FDA guidance, E9(R1) addendum (11 May 2021) | FDA guidance; describes review expectations | How sponsors should document estimands in protocols and statistical analysis plans (SAPs) | New legal obligations beyond the regulation |
| FDA guidance, Adjusting for Covariates in Randomized Clinical Trials (May 2023) | FDA guidance | Prospective specification of prognostic covariates, robust standard errors, estimand definition for nonlinear models | Observational confounding control |
| CONSORT 2010 | Journal editorial policy, not law | Items 17a and 17b: effect size with precision, and both absolute and relative effects for binary outcomes | The analysis itself |
| STROBE and RECORD | Journal editorial policy | Reporting of observational studies and studies using routinely collected data | Design quality |
| SAMPL | Journal editorial policy | Basic statistical reporting: numbers, tests, intervals, survival analyses | Advanced methods |

The distinction matters in a submission. A sponsor that ignores CONSORT will not be prosecuted, but a manuscript may be rejected or returned. A sponsor that ignores ICH E9(R1) in a confirmatory trial faces a review question that can delay approval, because the estimand defines what the trial is claiming.

### 8.2 What ICH E9(R1) actually requires

The addendum defines an estimand through five attributes and requires the sponsor to describe how intercurrent events are handled. It distinguishes:

- **Main estimation:** the primary analysis, targeting the estimand.
- **Sensitivity analysis:** repeated analyses under departures from the assumptions of the main analysis, to test robustness.
- **Supplementary analysis:** analyses that target a different estimand or provide additional insight, not a robustness check.

The addendum does not prescribe a method. It prescribes that the question, the handling of events, and the summary be stated before the analysis, and that sensitivity analyses address specific assumptions rather than repeat the main analysis with a different model.

For an industry analyst outside a regulated trial, the practical translation is a one-page estimand sheet attached to every analysis plan. The sheet has five rows and one paragraph on intercurrent events. If the team cannot fill it in, the analysis is not ready to run.

### 8.3 Regulatory treatment of p-values and intervals

No FDA or EMA guidance bans p-values. The expectation is that confirmatory claims are supported by a pre-specified hypothesis, a controlled Type I error rate, and an effect estimate with a confidence interval. The ASA statement is a professional-society position, not a regulation, but it is widely cited in reviewer comments and in journal editorial policy.

The practical consequences:

- A primary endpoint reported only as "p < 0.05" without an effect size is incomplete under CONSORT item 17a and under general review expectations.
- A subgroup claim supported only by a within-subgroup p-value, without an interaction test, is not a confirmatory claim.
- A non-inferiority claim requires an interval compared with a pre-specified margin, not a p-value compared with 0.05.

### 8.4 Real-world evidence

The FDA's real-world evidence (RWE) program accepts observational studies under specific conditions. The August 2023 final guidance, "Considerations for the Use of Real-World Data and Real-World Evidence to Support Regulatory Decision-Making for Drug and Biological Products," emphasizes reliability and relevance of the data source, pre-specified protocols and SAPs, and transparency. The estimand framework applies to these studies as well: a real-world study without a defined estimand is a database query, not evidence.

The FDA's RWE submissions page reports that in fiscal year 2025, CDER received 10 new drug application or biologics license application submissions containing RWE, and four approvals reflected an RWE contribution to the decision. Those counts are small relative to the total number of submissions, and they do not imply that any observational analysis is acceptable. They show that the pathway exists and that its gate is design quality.

### 8.5 What the standards do not cover

- They do not tell you the decision threshold. That is a business or clinical judgment.
- They do not validate the data. A perfect estimand computed on misclassified outcomes is still wrong.
- They do not resolve multiplicity across an entire program of analyses, only within a stated family.
- They do not make an observational estimate causal. Identification comes from design and assumptions, not from reporting compliance.

## 9. Industry practice and case evidence

### 9.1 Documented practice: pre-specification and double programming

Pharmaceutical sponsors routinely write a SAP before unblinding and often before the last patient is enrolled. The SAP states the primary estimand, the analysis population, the primary model, the handling of intercurrent events, the multiplicity strategy, and the sensitivity analyses. Independent statistical groups and double programming are common for pivotal trials. This is documented in ICH E9 and in company transparency policies, and it is visible in trial registries and in the clinical study reports that regulators publish.

**Inference (not a documented universal statistic):** organizations with mature analytics functions apply the same discipline to internal program evaluations, but many do not. The absence of a pre-specified analysis plan is one of the most reliable predictors that an internal evaluation will produce a result that cannot be reproduced.

### 9.2 Documented case: RCT-DUPLICATE

The RCT-DUPLICATE initiative emulated 32 randomized clinical trials using claims data, with protocols registered on ClinicalTrials.gov before analysis. The main results were published by Wang, Schneeweiss, and colleagues in JAMA in 2023 (JAMA 2023;329(16):1376-1385).

Reported agreement metrics:

| Metric | Result |
|---|---|
| Pearson correlation of effect estimates across 32 emulations | r = 0.82 (95% CI 0.64 to 0.91) |
| Intraclass correlation | 0.81 |
| Cohen kappa for significance agreement | 0.57 |
| Trials meeting significance agreement | 75% (56% full, 19% partial) |
| Estimate agreement | 66% |
| Close-emulation subgroup (n = 16) | r = 0.93 (95% CI 0.79 to 0.97) |
| Poorer-emulation subgroup (n = 16) | r = 0.53 (95% CI 0.00 to 0.83) |

The lesson for estimation and inference is not that observational studies reproduce trials. It is that the emulation quality, which includes the estimand, the time zero, the outcome definition, and the confounding control, predicts the agreement. When the design is closely emulated, the estimates align; when it is not, they diverge. The divergence is not only sampling error; it includes differences in the quantity being estimated.

### 9.3 Documented case: hormone replacement therapy

Observational studies suggested that hormone replacement therapy reduced coronary heart disease in postmenopausal women. The Women's Health Initiative randomized trial, reported by Manson and colleagues in the New England Journal of Medicine in 2003, found an increased risk of coronary heart disease in the estrogen-plus-progestin arm. The observational association was confounded by socioeconomic status, health behaviors, and the characteristics of women who chose therapy.

The estimation lesson is that a precise observational estimate can be precisely wrong when the identification assumption fails. The standard error describes sampling variability; it does not describe bias. An interval that excludes zero can still be centred on a biased quantity.

### 9.4 Documented case: the reimbursement of a diagnostic

Health technology assessment (HTA) bodies in several countries evaluate diagnostics using cost per true positive or cost per quality-adjusted life year (QALY). These calculations require the sensitivity, specificity, and prevalence, and they propagate the uncertainty in each. A diagnostic with a sensitivity of 0.90 (95% CI 0.84 to 0.94) and a specificity of 0.95 (95% CI 0.92 to 0.97) can produce a cost-effectiveness interval wide enough to change a coverage decision. The estimation requirement is not a single number; it is the joint uncertainty across the inputs.

### 9.5 Practice pattern: relative versus absolute reporting

Marketing decks for programs and devices tend to report relative reductions ("31% fewer readmissions") because the number is larger. Clinical and HTA reviewers tend to require absolute reductions because the number is decision-relevant. Both numbers come from the same table. The reporting failure is a choice, not a computational error.

**Inference:** the frequency of this pattern in industry materials is not known from a published census. It is consistent with the reporting gaps documented in reviews of trial abstracts, where relative effects are reported more often than absolute effects. The remedy is a reporting standard that requires both, as CONSORT item 17b does for binary outcomes.

### 9.6 What mature teams do differently

1. They write the estimand before the analysis and attach it to the results.
2. They compute the standard error with the design in mind, including clustering and repeated measures.
3. They report an interval on the decision scale, not only a p-value.
4. They pre-specify one or two sensitivity analyses tied to named assumptions.
5. They separate the statistical result from the decision recommendation, so that the recommendation can be debated without re-running the analysis.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| No estimand | The results slide names an outcome but not a population or time horizon | The quantity was never defined; different team members assume different targets | The interval is precise around the wrong quantity | Write the five-attribute estimand sheet before analysis |
| Odds ratio reported as risk | "36% reduction" from an OR of 0.64 with a common outcome | OR diverges from RR as the outcome becomes common | Benefit overstated | Report RR and RD; label the OR |
| No confidence interval | Only a point estimate and a p-value | Interval omitted for space or impact | Decision-makers cannot see the range | Require intervals in the reporting template |
| Standard deviation reported as standard error | Very small SE with visibly variable data | SD divided by sqrt(n) or software default confused | False precision | Check that SE shrinks with n and SD does not |
| Ignoring clustering | Naive SE much smaller than a site-level analysis | Independence assumed for correlated observations | Interval too narrow; false significance | Design effect or cluster-robust SE |
| Multiplicity ignored | Many comparisons, two significant, no adjustment | Type I error inflates with the number of tests | False discoveries enter the decision | Pre-specify the family; Holm or gatekeeping |
| Endpoint switched after seeing data | Primary outcome differs between protocol and report | The reported outcome was chosen for its result | Inflated Type I error; unusable evidence | Lock the SAP; report all outcomes |
| Small-sample z interval | n under 30 and a 1.96 multiplier | Normal approximation used where t is required | Interval too narrow | t interval or bootstrap |
| Zero cell handled by adding 0.5 without disclosure | A ratio appears with no events in one arm | Log-scale approximation fails | Misleading ratio and interval | Report RD; disclose any continuity correction |
| Immortal time | Treated group has guaranteed survival to treatment | Time zero misaligned | Large spurious benefit | New-user design; time-varying exposure |
| Confidence interval misinterpreted | "95% probability the true value is in this interval" | Bayesian language applied to a frequentist procedure | Overconfidence in a specific interval | State coverage correctly |
| Continuous variable dichotomized | "High risk" split at the median | Information lost; cut-point selected post hoc | Reduced power; inflated effect | Keep continuous; use splines |
| Alpha changed after interim look | The reported significance level is not the protocol level | Optional stopping without adjustment | Type I error above nominal | Pre-specify sequential boundaries |
| Only significant results reported | The report has no null results | Publication or selection bias | Effect overstated in the summary | Report all pre-specified analyses |
| Baseline risk ignored | Relative effect transported to a different setting | RD depends on baseline risk | Wrong absolute impact | Recalculate RD from local baseline |

The common thread is that none of these failures is a mathematical error in the narrow sense. Each is a mismatch between the quantity computed and the quantity needed, or between the uncertainty reported and the uncertainty that the design implies.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| Independent observations | Standard errors and intervals assume independent sampling units | Clustering by hospital, repeated measures per patient, matched pairs | Compare naive and cluster-robust SEs; estimate the ICC |
| Representative sample | The estimate generalizes to the target population | Selection into the cohort, survival bias, referral filtering | Compare cohort characteristics with the target population |
| Correct outcome measurement | The estimate refers to the defined endpoint | Misclassification, differential coding by group | Validation study; chart review sample; negative control outcome |
| No unmeasured confounding (causal claims) | The association is interpreted as an effect | Prognostic variables that affect treatment choice are unmeasured | E-value; negative control outcome; sensitivity analysis |
| Positivity | Every covariate stratum has both treated and untreated patients | A subgroup receives only one treatment | Overlap plots; trimmed weights; effective sample size |
| Consistency | The observed outcome under treatment equals the potential outcome under the defined treatment | Multiple versions of the treatment, varying adherence | Define the treatment precisely; report versions |
| Non-informative censoring (survival) | The risk set remains representative | Censoring related to prognosis and group | Inverse probability of censoring weights; sensitivity analysis |
| Adequate sample size for the normal approximation | Wald intervals and z tests rely on the central limit theorem | Small n, extreme proportions, sparse events | Wilson or t intervals; bootstrap; exact methods |
| Correct model specification (regression) | The estimator targets the parameter only if the model is right | Nonlinearity, interaction, omitted terms | Residual plots; calibration; alternative specifications |
| Missing at random (multiple imputation) | The imputation model can recover the missing values from observed data | Missingness depends on the unobserved value itself | Sensitivity analysis under MNAR; tipping point |
| No differential measurement error | The observed group difference reflects the true difference | One group is measured more intensively | Validation substudy; chart review |
| Stable treatment definition over time | Pooled estimates apply to one intervention | Protocol changes, product reformulation, learning curve | Analyze by period; test for interaction with time |

**Limits of this topic.** Estimation and inference do not identify causal effects. The interval around an observational estimate describes sampling variability under a model; it does not include uncertainty about the identification assumptions unless a sensitivity analysis is added. The methods here are large-sample approximations; exact methods exist for many of the edge cases and should be used when the data are sparse. The decision arithmetic in Section 6 uses a single cost per readmission; real budgets have fixed and variable components, and the break-even calculation changes accordingly.

## 12. Exercises

Each exercise states the type, the deliverable, the acceptance criteria, and the estimated time. Work in a single Markdown or notebook file so that the arithmetic can be checked.

### Exercise 1. Write an estimand (design, 60 minutes)

**Scenario.** A payer wants to know whether a remote blood-pressure monitoring program reduces hospitalization over 12 months. Enrolled patients receive a connected cuff and a nurse call if readings are high. Some patients stop using the cuff after three months; some start an additional antihypertensive prescribed by their own physician.

**Deliverable.** A five-row estimand table (treatment condition, population, endpoint, intercurrent-event strategy, summary) plus one paragraph justifying the strategy for each of the two intercurrent events: discontinuation of monitoring and addition of a new drug.

**Acceptance criteria.**

- All five attributes are present and specific enough to be applied to a second dataset.
- Each intercurrent event is named and assigned one of the five ICH strategies, with a reason tied to the payer's decision.
- The summary measure is on a scale the payer can use (for example, hospitalizations per 100 patient-years or risk difference at 12 months).
- The population and time zero are unambiguous.

### Exercise 2. Compute the interval and the NNT (application, 45 minutes)

**Data.**

| Group | Events | No event | Total |
|---|---:|---:|---:|
| Monitoring | 30 | 220 | 250 |
| Usual care | 48 | 212 | 260 |

**Deliverable.** The group risks, risk difference, 95% Wald interval, risk ratio with interval, odds ratio with interval, NNT with interval, and a two-sided p-value. State the units of each quantity. Then answer: does the interval exclude the null, and how far is the upper confidence limit from the null?

**Acceptance criteria.**

- Arithmetic is shown to at least four decimal places.
- The interval width equals 2 × 1.96 × SE.
- The NNT interval is reported with the correct direction and an explicit time horizon.
- The answer notes that the upper confidence limit is close to zero and explains what that means for a decision.

### Exercise 3. Critique a p-value claim (critique, 45 minutes)

**Claim.** "The program reduced readmissions (p = 0.03), proving that the intervention works. The effect was not significant in the small-hospital subgroup (p = 0.21), showing that the program does not work in small hospitals."

**Deliverable.** A 400-word critique using at least four of the six ASA principles and one interaction-test argument.

**Acceptance criteria.**

- Distinguishes statistical significance from effect size and from proof.
- Identifies the subgroup claim as a within-subgroup comparison rather than an interaction test.
- Notes that absence of significance is not evidence of absence.
- Proposes a correct analysis: an interaction term with a pre-specified subgroup, reported with an interval.
- Does not use the phrase "proves" in the corrected claim.

### Exercise 4. Choose the interval method (application, 60 minutes)

For each situation, name the method and justify it in one sentence.

1. 18 patients, continuous length of stay, strongly right-skewed.
2. 400 patients, binary outcome, 4% risk in one arm, 2% in the other.
3. 12 hospitals, 300 patients, outcome measured at the patient level.
4. A risk ratio with 3 events in one arm.
5. A proportion of 0 out of 40.
6. A 30-day survival probability from a Kaplan-Meier curve with 25 events.

**Deliverable.** A six-row table: situation, method, reason, and the quantity that would signal the method is failing.

**Acceptance criteria.**

- At least four different methods are used across the six situations.
- Each justification refers to a property of the data, not to software convenience.
- The zero-event and small-event cases do not use a log-scale Wald interval without comment.

### Exercise 5. Design effect (application, 45 minutes)

A quality program is evaluated in 30 clinics with an average of 60 patients per clinic. The ICC for the outcome is estimated at 0.03. The naive risk difference is -4.0 percentage points with a naive SE of 1.8 percentage points.

**Deliverable.** The design effect, the adjusted SE, the adjusted 95% interval, and a sentence for a non-statistical audience explaining why the interval changed. Then recompute with an ICC of 0.08 and state whether the conclusion changes.

**Acceptance criteria.**

- Design effect formula is written explicitly.
- Adjusted SE and interval are correct to two decimal places.
- The explanation avoids the word "bias" for the change in interval width; it is a variance issue, not a bias issue unless the cluster sampling is informative.
- The conclusion is stated separately for each ICC.

### Exercise 6. Coverage simulation (application and programming, 90 minutes)

Write a Python script using only the standard library that:

1. Simulates 2,000 studies with n = 30 and true proportion p = 0.10.
2. Computes a Wald 95% interval and a Wilson 95% interval for each.
3. Reports the coverage of each.

**Deliverable.** The script, its printed output, and a three-sentence interpretation.

**Acceptance criteria.**

- The seed is fixed and the output is reproducible.
- The script does not import numpy, scipy, or pandas.
- The reported coverage is compared with the nominal 0.95.
- The interpretation explains why the Wald interval undercovers and why the Wilson interval does not.

### Exercise 7. Audit a vendor deck (critique, 75 minutes)

Use a real or constructed vendor deck with at least six slides. If you use a constructed deck, label it as constructed in your answer.

**Deliverable.** A red-flag table with columns: slide, claim, missing information, why it matters, question to ask the vendor. At least eight rows.

**Acceptance criteria.**

- At least three distinct failure modes from Section 10 are identified.
- Each question is answerable by the vendor and would change the decision if answered badly.
- The audit separates what is wrong from what is merely unclear.
- The final recommendation states what evidence would be sufficient to proceed.

### Exercise 8. Design a decision rule (design, 120 minutes)

**Scenario.** A hospital must decide whether to buy a sepsis alert. The alert costs USD 600,000 per year. The eligible population is 4,000 admissions per year. Each avoided septic shock case saves an estimated USD 18,000, and the alert is expected to reduce cases by between 1 and 6 percentage points.

**Deliverable.** A one-page decision rule that states:

1. The estimand.
2. The break-even absolute reduction.
3. The interval that would justify adoption, the interval that would justify rejection, and the interval that would require a pilot.
4. The sensitivity analysis that would most change the decision.
5. The cost of each type of error.

**Acceptance criteria.**

- Break-even arithmetic is correct and shown.
- The rule uses interval bounds, not only the point estimate.
- The three-region rule is explicit and mutually exclusive.
- The sensitivity analysis is tied to a named assumption.
- The recommendation is conditional on the evidence, not on the vendor's framing.

## 13. Self-check answers

**Answer 1 (Exercise 2 arithmetic).**

```text
p1 = 30/250 = 0.120000
p0 = 48/260 = 0.184615
RD = -0.064615 = -6.46 percentage points
SE = sqrt(0.12 × 0.88/250 + 0.184615 × 0.815385/260)
   = sqrt(0.0004224 + 0.0005789)
   = sqrt(0.0010013) = 0.031644
95% CI = -0.064615 +/- 1.96 × 0.031644
       = (-0.126637, -0.002593) = (-12.66, -0.26) percentage points
z = -2.042; two-sided p = 0.0412
RR = 0.120000/0.184615 = 0.6500 (95% CI 0.4263 to 0.9911)
OR = (30/220)/(48/212) = 0.136364/0.226415 = 0.6023 (95% CI 0.3676 to 0.9867)
NNT = 1/0.064615 = 15.48, reported as 16
NNT 95% CI = (7.9, 385.6), reported as 8 to 386
```

The interval excludes zero, but the upper limit is -0.26 percentage points, which is a reduction of about 2.6 readmissions per 1,000 patients. A decision threshold of 5 percentage points would not be met at the point estimate of 6.46 only if the lower bound were used; the point estimate exceeds 5, the interval includes values below 5, and the interval also includes values above 6. The honest statement is that the data are compatible with a benefit ranging from negligible to 12.7 percentage points. The NNT interval of 8 to 386 makes the uncertainty visible in the units a budget officer uses.

**Answer 2 (Exercise 4 method selection).**

| Situation | Method | Reason | Failure signal |
|---|---|---|---|
| 18 patients, skewed length of stay | Bootstrap percentile or BCa | The sampling distribution of the mean is skewed; t assumes symmetry | Bootstrap interval very different from t; heavy influence of one patient |
| 400 patients, 4% vs 2% | Newcombe or Wilson-based RD; log RR for the ratio | Sparse cells and proportions near zero | Wald interval crosses 0 or 1, or is much narrower than Newcombe |
| 12 hospitals, 300 patients | Cluster bootstrap | The unit of independence is the hospital, not the patient | Naive SE much smaller than cluster SE |
| Risk ratio with 3 events | Exact or bootstrap; report RD as well | Log-scale normal approximation is poor with 3 events | Interval spans implausible values; asymmetric on the ratio scale |
| 0 out of 40 | Wilson or exact one-sided upper bound | Wald interval has zero width | Any interval that excludes 0 |
| 30-day survival, 25 events | Greenwood variance with log-log transform | Survival is bounded and its sampling distribution is skewed | Interval outside 0 to 1; symmetric interval around a probability |

**Answer 3 (Exercise 5 design effect).**

```text
Design effect = 1 + (60 - 1) × 0.03 = 1 + 1.77 = 2.77
Adjusted SE = 1.8 × sqrt(2.77) = 1.8 × 1.664 = 3.00 percentage points
Adjusted 95% CI = -4.0 +/- 1.96 × 3.00 = (-9.87, +1.87) percentage points
```

With an ICC of 0.08, the design effect is 1 + 59 × 0.08 = 5.72, the SE is 1.8 × 2.392 = 4.31 percentage points, and the interval is (-12.44, +4.44). The conclusion changes from "significant benefit" (the naive interval excludes zero) to "inconclusive" at both ICC values. The interval widens because the effective sample size is 300/2.77 = 108 patients at ICC 0.03, not because the point estimate is biased. The point estimate is unchanged.

**Answer 4 (Exercise 6 expected result).**

At n = 30 and p = 0.10, the Wald interval covers the true proportion in roughly 91% to 93% of replications, while the Wilson interval covers in roughly 94% to 96%. The exact value depends on the seed. The mechanism is that the Wald interval uses the estimated p in the variance and collapses when the sample proportion is 0 or 1; with n = 30 and p = 0.10, zero events occur in about 4% of samples, and those samples produce a zero-width interval that cannot cover. The Wilson interval uses a score test that remains defined at the boundary and has coverage closer to nominal.

**Answer 5 (Exercise 3 critique rubric).**

A strong critique contains:

- The p-value is the probability of data at least as extreme under the null, not the probability that the program works.
- Statistical significance at p = 0.03 does not establish clinical or economic importance; the effect size and interval are missing.
- The subgroup comparison uses a within-subgroup test, which is not a test of interaction; a non-significant result in a small subgroup is expected when power is low.
- The correct test is treatment-by-subgroup interaction, pre-specified, with the subgroup defined at baseline.
- Absence of evidence in the subgroup is not evidence of absence.
- The revised claim reports the effect with an interval and states the uncertainty.

A weak critique states only "p-values are bad" or "the sample is small" without connecting the error to the decision.

**Answer 6 (Exercise 1 rubric).**

A strong estimand:

- Defines the treatment as the offer of monitoring plus the nurse response, not as "using the cuff."
- Defines the population by age, diagnosis, and care setting, with a stated time zero at enrollment.
- Defines the endpoint as unplanned hospitalization within 12 months, with an explicit rule for deaths.
- Handles discontinuation with a treatment-policy strategy if the payer is buying the offer, or a while-on-treatment strategy if the payer is buying the monitoring itself; either is defensible if justified.
- Handles the new antihypertensive with a treatment-policy strategy, because the payer cannot control prescribing and the decision concerns the program as delivered.
- States the summary as a risk difference at 12 months, with hospitalizations per 100 patient-years as a secondary measure.

A weak estimand uses "improves outcomes" as the endpoint, leaves time zero implicit, or does not mention either intercurrent event.

**Answer 7 (Exercise 7 rubric).**

A strong audit:

- Names at least three mechanisms, not just "the number is wrong."
- Distinguishes missing information (cannot evaluate) from incorrect information (can evaluate and reject).
- Asks questions that the vendor can answer with data, not with reassurance.
- Identifies the decision threshold that the deck omits.
- States what a sufficient evidence package would contain: estimand, cohort flow, baseline table, unadjusted and adjusted estimates, interval, and sensitivity analysis.

**Answer 8 (Exercise 8 arithmetic).**

```text
Break-even absolute reduction = 600,000 / (4,000 × 18,000)
                              = 600,000 / 72,000,000
                              = 0.00833 = 0.83 percentage points
```

The break-even effect is 0.83 percentage points, far below the vendor's expected 1 to 6 percentage points. The decision rule:

- Adopt if the lower 95% confidence limit exceeds 0.83 percentage points.
- Reject if the upper 95% confidence limit is below 0.83 percentage points.
- Pilot if the interval contains 0.83 percentage points.

The error costs are asymmetric: adoption at USD 600,000 per year for a null effect wastes the full cost, while rejection of a 3-percentage-point effect forgoes 120 avoided cases per year, valued at USD 2.16 million. The rule should therefore require stronger evidence against adoption than against rejection, unless the budget is fixed and the opportunity cost of the 600,000 is high.

## 14. Interview questions

**1. What is an estimand, and why does it come before the method?**

A strong answer names the five attributes and explains that the method is chosen to target the quantity, not the reverse. It mentions intercurrent events and gives an example where treatment-policy and hypothetical estimands yield different numbers. A weak answer defines an estimand as "the outcome" or says it is a regulatory formality.

**2. What does a 95% confidence interval mean?**

A strong answer describes coverage across repeated samples and explicitly rejects the statement that there is a 95% probability the parameter lies in this interval. A weak answer gives the Bayesian interpretation without noticing the difference.

**3. What does a p-value not tell you?**

A strong answer lists at least four ASA principles: not the probability the null is true, not the probability of replication, not the effect size, not the importance, and not a substitute for the model. A weak answer says only "it does not tell you the effect size."

**4. Why can an odds ratio mislead when the outcome is common?**

A strong answer explains non-collapsibility and gives the numerical example: with a control risk of 21.65%, OR 0.641 versus RR 0.695. It notes that the OR is not wrong, but it answers a different question. A weak answer says the OR is "biased" or confuses the OR with the RR.

**5. When would you use a cluster-robust standard error?**

A strong answer identifies the sampling unit, gives examples (patients within hospitals, repeated measures within patients), and explains the design effect. It notes that clustering affects variance, not necessarily the point estimate. A weak answer says "when the data are grouped" without connecting it to the interval.

**6. How do you decide between a Wald and a Wilson interval?**

A strong answer refers to the cell counts, the proximity of the proportion to 0 or 1, and the observed undercoverage of Wald in small samples. It may mention that Newcombe's method extends the Wilson idea to differences. A weak answer says "use Wald because it is standard."

**7. What is the difference between a standard deviation and a standard error?**

A strong answer describes the SD as the spread of observations and the SE as the spread of the estimator across samples, and notes that SE shrinks with the square root of n while SD does not. A weak answer treats them as interchangeable or says the SE is "the SD divided by n."

**8. How would you handle a primary analysis with 30% missing outcomes?**

A strong answer states the estimand, distinguishes missingness mechanisms, proposes multiple imputation under MAR with the outcome included, and pre-specifies a tipping-point or MNAR sensitivity analysis. It reports the amount and pattern of missingness. A weak answer proposes complete-case analysis or last-observation-carried-forward without discussion.

**9. Why is "p = 0.03 in the subgroup" not evidence of a subgroup effect?**

A strong answer explains that a within-subgroup test does not test the difference between subgroups, that power is lower in the subgroup, and that the correct test is an interaction with a pre-specified subgroup. A weak answer says the subgroup is too small without explaining the inferential error.

**10. What would make you distrust a significant result?**

A strong answer lists pre-specification, multiplicity, effect magnitude, interval width, clustering, missing data, and sensitivity to adjustment. It distinguishes a result that is fragile from one that is biased. A weak answer says "a small sample" and stops.

**11. How do you translate a risk difference into a budget impact?**

A strong answer multiplies the RD by the eligible population and the cost per event, computes the break-even effect, and compares the interval with the threshold. It notes that baseline risk may differ locally. A weak answer reports the relative reduction without the absolute calculation.

**12. What is the difference between a sensitivity analysis and a supplementary analysis under ICH E9(R1)?**

A strong answer says a sensitivity analysis tests robustness of the main estimate to departures from its assumptions, while a supplementary analysis targets a different estimand or adds insight and is not a robustness check. A weak answer treats every additional analysis as a sensitivity analysis.

## 15. Key terms

- **Absolute risk reduction:** the difference in outcome risk between groups, in percentage points.
- **Alpha:** the pre-specified Type I error rate for a decision rule.
- **Bootstrap:** resampling observed units to approximate a sampling distribution.
- **Break-even effect:** the effect at which the benefit equals the cost.
- **Cluster-robust standard error:** a variance estimate that allows correlation within clusters.
- **Confidence interval:** an interval from a procedure with a stated coverage probability.
- **Coverage:** the long-run proportion of intervals containing the true parameter.
- **Design effect:** the ratio of clustered variance to simple-random-sample variance.
- **Estimand:** the precise quantity to estimate, defined by five attributes.
- **Estimate:** the numerical value produced by an estimator on one sample.
- **Estimator:** the rule that maps data to an estimate.
- **Intraclass correlation:** the proportion of total variance attributable to cluster membership.
- **Minimal clinically important difference:** the smallest effect that matters to patients or clinicians.
- **Multiplicity:** the inflation of Type I error when many hypotheses are tested.
- **Newcombe interval:** a hybrid score interval for a difference in proportions.
- **NNT:** the number of patients treated per additional event avoided.
- **Non-inferiority margin:** the largest acceptable loss of effect.
- **Odds ratio:** the ratio of odds of the outcome between groups.
- **p-value:** the probability of a test statistic at least as extreme as observed under the null.
- **Power:** the probability of rejecting the null when a specified alternative is true.
- **Risk difference:** the absolute difference in outcome risk, in percentage points.
- **Risk ratio:** the ratio of outcome risks between groups.
- **Sampling distribution:** the distribution of an estimator across repeated samples.
- **Standard error:** the standard deviation of an estimator's sampling distribution.
- **Statistical significance:** a decision that a p-value is below a pre-specified threshold.
- **Tipping point:** the amount of bias or missingness that would change the conclusion.
- **Type I error:** rejecting a true null hypothesis.
- **Wilson interval:** a score-based confidence interval for a proportion.
- **Wald interval:** an interval of the form estimate plus or minus a multiple of the standard error.

## 16. Sources

All URLs were accessed on 25 October 2026.

1. International Council for Harmonisation. *ICH Harmonised Tripartite Guideline E9: Statistical Principles for Clinical Trials*. Step 4, 5 February 1998. https://database.ich.org/sites/default/files/E9_Guideline.pdf
2. International Council for Harmonisation. *ICH E9(R1) Addendum on Estimands and Sensitivity Analysis in Clinical Trials to the Guideline on Statistical Principles for Clinical Trials*. Step 4, 20 November 2019. https://database.ich.org/sites/default/files/E9-R1_Step4_Guideline_2019_1203.pdf
3. United States Food and Drug Administration. *E9(R1) Statistical Principles for Clinical Trials: Addendum: Estimands and Sensitivity Analysis in Clinical Trials: Guidance for Industry*. 11 May 2021. https://www.hhs.gov/guidance/document/e9r1-statistical-principles-clinical-trials-addendum-estimands-and-sensitivity-analysis
4. United States Food and Drug Administration. *Adjusting for Covariates in Randomized Clinical Trials for Drugs and Biological Products: Guidance for Industry*. May 2023. https://www.fda.gov/media/148910/download
5. Wasserstein, R. L., Lazar, N. A. "The ASA Statement on p-Values: Context, Process, and Purpose." *The American Statistician* 2016;70(2):129-133. https://doi.org/10.1080/00031305.2016.1154108
6. NIST/SEMATECH. *e-Handbook of Statistical Methods*. https://www.itl.nist.gov/div898/handbook/ (last updated 27 April 2022).
7. Schulz, K. F., Altman, D. G., Moher, D., for the CONSORT Group. "CONSORT 2010 Statement: Updated Guidelines for Reporting Parallel Group Randomised Trials." *PLoS Medicine* 2010;7(3):e1000251. https://doi.org/10.1371/journal.pmed.1000251
8. Lang, T. A., Altman, D. G. "Basic Statistical Reporting for Articles Published in Biomedical Journals: The SAMPL Guidelines." EQUATOR Network, 20 March 2013. https://www.equator-network.org/2013/03/20/basic-statistical-reporting-for-articles-published-in-biomedical-journals-the-statistical-analyses-and-methods-in-the-published-literature-or-the-sampl-guidelines/
9. STROBE Initiative. *STROBE Statement: Strengthening the Reporting of Observational Studies in Epidemiology*. https://www.strobe-statement.org/
10. Benchimol, E. I., Smeeth, L., Guttmann, A., et al. "The REporting of studies Conducted using Observational Routinely-collected health Data (RECORD) Statement." *PLoS Medicine* 2015;12(10):e1001885. https://doi.org/10.1371/journal.pmed.1001885
11. Hernán, M. A., Robins, J. M. "Using Big Data to Emulate a Target Trial When a Randomized Trial Is Not Available." *American Journal of Epidemiology* 2016;183(8):758-764. https://doi.org/10.1093/aje/kwv254
12. Wang, S. V., Schneeweiss, S., and the RCT-DUPLICATE Initiative. "Emulation of Randomized Clinical Trials With Nonrandomized Database Analyses: Results of 32 Clinical Trials." *JAMA* 2023;329(16):1376-1385. https://doi.org/10.1001/jama.2023.4221
13. United States Food and Drug Administration. *Real-World Evidence Submissions to the Center for Drug Evaluation and Research*. https://www.fda.gov/science-research/real-world-evidence/real-world-evidence-submissions-center-drug-evaluation-and-research
14. United States Food and Drug Administration. *Considerations for the Use of Real-World Data and Real-World Evidence to Support Regulatory Decision-Making for Drug and Biological Products: Guidance for Industry*. August 2023. https://www.fda.gov/media/171667/download
15. Manson, J. E., Hsia, J., Johnson, K. C., et al. "Estrogen plus Progestin and the Risk of Coronary Heart Disease." *New England Journal of Medicine* 2003;349:523-534. https://doi.org/10.1056/NEJMoa030808

> **Note:** Sources 1 to 4 are regulatory or standards documents. Sources 5 to 10 are reporting and methods references. Sources 11 to 15 are empirical or methodological studies used in Section 9. The RCT-DUPLICATE figures are quoted from the published abstract and main results; the fiscal-year 2025 submission counts are quoted from the FDA page and change as the agency updates it.

## 17. Links

- Module index: [`_index.md`](_index.md)
- Next topic: [`02_regression-models.md`](02_regression-models.md) — regression as an estimator factory.
- Causal assumptions: [`05_causal-inference-and-dags.md`](05_causal-inference-and-dags.md) — identification before estimation.
- Reporting: [`10_reporting-statistical-results.md`](10_reporting-statistical-results.md) — how to present the interval and the absolute effect.
- Propensity weighting: [`06_propensity-scores-and-weighting.md`](06_propensity-scores-and-weighting.md) — standard errors under weighting.
- Survey file: [`../../04_statistics_causal_inference.md`](../../course/04_statistics_causal_inference.md)
- Labs: [`../../labs/`](../../labs/) — Lab 4.6 covers bootstrap confidence intervals; the exercises in this topic feed directly into it.
- Foundations: [`../../01A_foundations_assumptions_and_units.md`](../../course/01A_foundations_assumptions_and_units.md) — units, assumptions ledger, and notation.
