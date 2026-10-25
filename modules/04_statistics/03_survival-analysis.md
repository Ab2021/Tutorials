# Survival Analysis

> Module 04 — Statistics and Causal Inference
> Topic 03 of 10
> Estimated study time: 4 hours
> Prerequisites: 01, 02
> Version 2026.3

---

## 1. Why this topic exists

Time-to-event outcomes are different from every other outcome in this module because the observation can end before the event happens. A patient discharged after a heart-failure admission may be readmitted on day 9, may die on day 12 without a readmission, may move to another region on day 20, or may still be event-free when the dataset is frozen at day 30. The last three patients carry information, but not the same information. Ignoring the difference produces numbers that look precise and are wrong.

Four errors recur in industry analyses:

1. **Ignoring time.** Analysts compute the proportion readmitted within 30 days among patients with 30 days of follow-up. This conditions on survival to 30 days and drops the patients who died or left. The denominator becomes a selected population, and the comparison between groups is no longer randomized or exchangeable.
2. **Ignoring censoring.** Analysts count censored patients as event-free and divide by the full cohort. A patient censored on day 2 contributes the same to the numerator as a patient followed event-free for 30 days. The risk is underestimated, and the underestimation differs by group when follow-up differs.
3. **Immortal time.** A program is defined as "enrolled within 7 days of discharge." Patients who die on day 2 cannot be enrolled, so the enrolled group is guaranteed to survive to day 7. The comparison includes a period during which the outcome could not occur in one group. This is immortal time bias, documented by Suissa in the American Journal of Epidemiology in 2008, and it can turn a null drug into an apparently protective one.
4. **Competing risks.** In older adults, death prevents readmission. A Kaplan-Meier curve that censors death estimates the readmission risk in a hypothetical world where patients cannot die, which is not the world the decision concerns. The complement of Kaplan-Meier overestimates the cumulative incidence of readmission when death is common.

The decisions that depend on these quantities are concrete:

- A hospital at risk for a readmission penalty needs the 30-day readmission risk under the program, not a hazard ratio.
- A dialysis provider comparing catheter types needs the rate of catheter-related bloodstream infection per 1,000 catheter-days, with censoring at transplant and death.
- A pharmaceutical sponsor needs an overall survival (OS) hazard ratio for a label and a restricted mean survival time (RMST) difference for a health technology assessment.
- A device manufacturer needs the probability of revision at 5 years, with death as a competing event.

Regulators have become more specific about the time-to-event estimand. The United States Food and Drug Administration (FDA) draft guidance "Approaches to Assessment of Overall Survival in Oncology Clinical Trials," issued in August 2025, recommends that randomized oncology trials be designed to assess OS as a pre-specified safety endpoint even when OS is not primary, using intent-to-treat Cox or log-rank analyses and explicit handling of crossover and subsequent therapy through the estimand framework. The International Council for Harmonisation (ICH) E9(R1) addendum applies: the time horizon, the handling of death, and the summary measure must be defined before analysis.

This topic builds the mechanics: the risk set, the hazard, the Kaplan-Meier estimator, the log-rank test, the Cox model, the proportional hazards assumption, competing risks, and the alternatives when proportional hazards fails.

> **Note:** A hazard ratio is not a risk ratio. It compares instantaneous event rates among those still at risk. When the event is common or the hazard changes over time, the hazard ratio can be far from the ratio of cumulative risks that a patient or payer experiences.

## 2. Learning outcomes

By the end of this topic you can:

1. **Define** time zero, the event, the censoring rule, and the competing events for a given question, and **explain** how each choice changes the estimand.
2. **Compute** a Kaplan-Meier curve by hand from a small risk-set table, including the Greenwood standard error at a stated time.
3. **Perform** a log-rank test and **interpret** the observed-minus-expected count in each group.
4. **Fit** and **interpret** a Cox model, stating the proportionality assumption and the scale of the coefficient.
5. **Diagnose** non-proportional hazards using scaled Schoenfeld residuals or a time interaction, and **choose** among stratification, time-varying coefficients, and RMST.
6. **Distinguish** cause-specific from subdistribution hazards, and **compute** a cumulative incidence function in the presence of a competing risk.
7. **Detect** immortal time and **redesign** the analysis with a time-varying exposure or an aligned time zero.
8. **Report** a time-to-event result with the estimand, the number at risk, the cumulative incidence or survival at named times, and the assumption checks.

Outcome 2 maps to Exercise 2 and the worked example. Outcome 5 maps to Exercise 4 and self-check item 5. Outcome 6 maps to Exercise 5.

## 3. The core mental model

Survival analysis is a sequence of risk sets. At every event time, the method compares what happened among the people still under observation with what would have been expected if the groups were exchangeable.

```text
Patient A  |---- event (readmission) day 9
Patient B  |--------- censored day 20 (moved away)
Patient C  |------------------- event day 27
Patient D  |---- death day 6 (competing event)
Patient E  |------------------------------ administratively censored day 30

Time        0    6    9        20   27   30
Risk set    5    4    3        2    1    0
                 ^    ^        ^    ^
              death  event   censor event
```

The risk set at time t is the set of patients with follow-up time at least t. It shrinks when an event or a censoring occurs. The hazard at t is the instantaneous rate of the event among those still at risk. The survival function is the probability of remaining event-free past t.

### 3.1 The three quantities and how they relate

| Quantity | Definition | Scale | What it answers |
|---|---|---|---|
| Hazard h(t) | Instantaneous event rate among those at risk at t | Events per unit time | How fast is the event occurring now? |
| Cumulative hazard H(t) | Integral of h from 0 to t | Unitless | How much accumulated risk? |
| Survival S(t) | Probability of being event-free at t | Probability, 0 to 1 | What fraction remains event-free? |

Under the continuous-time model, S(t) = exp(-H(t)). The Kaplan-Meier estimator estimates S(t) without assuming a functional form for h(t). The Cox model estimates the ratio of hazards between groups without estimating the baseline hazard.

### 3.2 Kaplan-Meier is a product of conditional probabilities

At each event time t_j, with n_j at risk and d_j events, the conditional probability of surviving that instant given survival to just before it is 1 - d_j/n_j. The Kaplan-Meier estimate is the product:

```text
S(t) = product over event times t_j <= t of (1 - d_j / n_j)
```

Censoring times do not change S(t); they reduce the risk set for later event times. This is why a Kaplan-Meier curve is a step function that falls only at events and stays flat at censoring times.

The Greenwood variance is

```text
Var(S(t)) = S(t)^2 × sum over t_j <= t of d_j / (n_j (n_j - d_j))
```

and the standard error is its square root. Confidence intervals are usually built on the log-log scale to keep the limits between 0 and 1.

### 3.3 The Cox model compares risk sets

The Cox model writes the hazard as

```text
h(t | X) = h0(t) × exp(beta1 X1 + ... + betap Xp)
```

The baseline hazard h0(t) is unspecified. The coefficient beta is estimated from the partial likelihood, which at each event time compares the covariate of the patient who had the event with the covariates of everyone still at risk:

```text
Partial likelihood contribution at t_j = exp(beta X_event) / sum over risk set of exp(beta X_i)
```

This is why the Cox model is called semiparametric: the baseline hazard is left free, and only the ratio is estimated. It is also why the coefficient has a conditional interpretation: it is the log hazard ratio for a one-unit change in X among patients who are still at risk at the same time.

### 3.4 Proportional hazards is a modelling assumption, not a fact

The Cox model assumes that the ratio of hazards between two covariate patterns is constant over time. If the treatment effect is strong early and disappears later, the hazard ratio changes, and a single coefficient is a time-average that can mislead in both directions. The assumption can be checked with scaled Schoenfeld residuals, with a treatment-by-time interaction, or by comparing Kaplan-Meier curves. It cannot be checked by looking at whether the curves cross, because non-proportional hazards can occur without crossing.

### 3.5 Competing risks change the quantity, not just the method

With a competing event, each patient can experience one of several mutually exclusive events. There are two different hazards:

- **Cause-specific hazard:** the instantaneous rate of the event of interest among those still at risk, treating the competing event as censoring. It answers "how fast does the event occur among those still event-free?"
- **Subdistribution hazard:** the instantaneous rate of the event of interest among those still at risk plus those who already had the competing event (who remain in the risk set with weight). It answers "how does the cumulative incidence function change?"

The corresponding regression models are the cause-specific Cox model and the Fine-Gray subdistribution hazard model. They answer different questions and can give different coefficients. Neither is universally correct.

The cumulative incidence function is

```text
CIF_k(t) = sum over event times t_j <= t of S(t_j-) × d_kj / n_j
```

where S(t_j-) is the event-free survival just before t_j and d_kj is the number of cause-k events at t_j. The sum of the CIFs over all causes equals 1 minus the probability of being event-free.

### 3.6 Time zero is the most consequential choice

Every survival analysis starts the clock somewhere. The choice defines the risk set, the eligible population, and the meaning of the estimate. Common time zeros:

- Date of randomization in a trial.
- Date of first prescription in a new-user cohort.
- Date of discharge for a readmission study.
- Date of diagnosis for a cancer cohort.
- Date of device implantation for a revision study.

If treatment is defined by an event after time zero, such as enrollment in a program or a second prescription, the period between time zero and treatment is immortal in the treated group. The remedy is to move time zero to treatment initiation, or to model treatment as a time-varying exposure that starts at the date it was received.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or scale |
|---|---|---|---|
| Time origin | The date follow-up starts | Left implicit | Calendar date or study day |
| Event | The outcome of interest | Any bad outcome | A defined indicator |
| Censoring | End of observation without the event | Treated as event-free | Indicator |
| Right censoring | Observation ends after the last contact | Confused with truncation | Indicator |
| Administrative censoring | Follow-up ends at a fixed calendar date | Treated as informative | Indicator |
| Left truncation | Entry into the risk period after time zero | Ignored, biasing survival upward | Delayed entry |
| Risk set | Patients under observation and event-free at time t | Computed with the wrong time rule | Count |
| Person-time | Total time at risk contributed by the cohort | Replaced by the number of patients | Person-days or person-years |
| Incidence rate | Events divided by person-time | Called a risk | Events per 1,000 person-years |
| Hazard | Instantaneous event rate among those at risk | Called a probability | Events per unit time |
| Cumulative hazard | Integral of the hazard | Called a risk | Unitless |
| Survival function | Probability of remaining event-free | Called "survival rate" without a time | Probability, 0 to 1 |
| Kaplan-Meier estimator | Product-limit estimator of the survival function | Described as assumption-free | Probability |
| Greenwood variance | Variance estimator for Kaplan-Meier | Used without the log-log transform near 0 or 1 | Variance |
| Median survival time | Smallest time at which S(t) is at most 0.5 | Reported when the curve never reaches 0.5 | Time |
| Log-rank test | A test comparing observed and expected events across groups | Used when hazards cross | Chi-square statistic |
| Hazard ratio | Ratio of hazards between groups | Interpreted as a risk ratio | Unitless, positive |
| Proportional hazards | The hazard ratio is constant over time | Assumed without a check | Assumption |
| Schoenfeld residuals | Residuals used to test proportional hazards | Plotted without scaling | Residual scale |
| Time-varying coefficient | A coefficient that changes with time | Confused with a time-varying covariate | Model scale |
| Time-varying covariate | A covariate whose value changes during follow-up | Entered as a baseline value | Covariate unit |
| Immortal time | Follow-up during which the outcome cannot occur | Ignored in exposure definition | Time |
| Landmark analysis | Analysis starting at a fixed time after baseline | Used to fix immortal time without aligning exposure | Time |
| Competing risk | An event that prevents the event of interest | Censored as if it were unrelated | Event type |
| Cause-specific hazard | Hazard of the event among those still at risk | Compared with subdistribution hazard | Events per unit time |
| Subdistribution hazard | Hazard in a risk set that retains prior competing events | Interpreted as a cause-specific hazard | Events per unit time |
| Cumulative incidence function | Probability of the event by time t in the presence of competing risks | Replaced by 1 minus Kaplan-Meier | Probability, 0 to 1 |
| Fine-Gray model | Regression model for the subdistribution hazard | Used for etiologic questions | Subdistribution hazard ratio |
| Restricted mean survival time (RMST) | Area under the survival curve up to a horizon | Interpreted beyond the horizon | Time units |
| Non-proportional hazards | A hazard ratio that changes over time | Ignored because the Cox model ran | Assumption violation |
| Ties | Multiple events at the same recorded time | Handled inconsistently | Count |
| Efron method | A tie-handling approximation in Cox models | Confused with Breslow | Method |
| Informative censoring | Censoring related to prognosis | Assumed independent | Assumption |

Two conventions in this topic:

- Time is reported in a named unit (days, months, years) with the horizon. A hazard ratio without a time unit is incomplete because the baseline hazard has units.
- Cumulative incidence is reported as a probability with a horizon: "28.4% at 24 months," not "28.4%."

> **Note:** The complement of Kaplan-Meier, 1 - S(t), is not the cumulative incidence of the event when a competing risk exists. It is the cumulative incidence in a hypothetical population in which the competing event never occurs. That quantity is sometimes useful, but it is not the probability a patient experiences the event.

## 5. The mechanics, step by step

The procedure runs from the estimand to the reported curve. Steps 1 to 5 define the cohort and the clock. Steps 6 to 9 estimate and test. Steps 10 to 13 handle the assumptions that fail most often. Step 14 reports.

### Step 1. Choose the estimand and the summary measure

**Input:** The decision, the population, the treatment contrast, the event, and the time horizon.

**Output:** A named target: risk at a horizon, hazard ratio, RMST difference, median time, or incidence rate.

**Why it exists:** Each target requires a different estimator and answers a different question. A hazard ratio is not a risk, and a median is not defined when the curve never reaches 0.5.

| Decision question | Target | Estimator | Reporting unit |
|---|---|---|---|
| What fraction has the event by 24 months? | Cumulative incidence at 24 months | Kaplan-Meier or Aalen-Johansen | Probability with horizon |
| How much faster does the event occur? | Hazard ratio | Cox model | Unitless ratio |
| How many events per 1,000 patient-years? | Incidence rate ratio | Poisson or negative binomial with person-time | Events per 1,000 person-years |
| How many event-free months are gained? | RMST difference | Area under Kaplan-Meier to a horizon | Months |
| When does the typical patient have the event? | Median survival time | Kaplan-Meier | Months |
| What is the probability of being event-free at a fixed time? | Survival probability | Kaplan-Meier | Probability |

**Edge case:** If the survival curve does not reach 0.5 by the end of follow-up, the median is not estimable. Report "not reached" with the number at risk and the last observed survival, not a fabricated median.

**Edge case:** If the decision needs both a relative and an absolute summary, pre-specify both and their relationship. A hazard ratio and a 24-month risk difference are not redundant; they answer different questions and can point in different directions when the baseline hazard changes. Reporting both is not hedging; it is giving the decision-maker the two quantities the decision requires.

### Step 2. Fix time zero and eligibility

**Input:** The clinical question, the data source, and the treatment definition.

**Output:** One time zero per patient, eligibility assessed at or before that time, and a treatment status defined at that time.

**Why it exists:** The risk set is defined relative to time zero. A misaligned time zero creates immortal time or prevalent-user bias, and no estimator can repair it.

| Design problem | Symptom | Remedy |
|---|---|---|
| Treatment defined by an event after baseline | Treated patients survive to treatment | Move time zero to treatment initiation |
| Prevalent users | Patients already on treatment at cohort entry | New-user design with a washout period |
| Eligibility uses future information | Criteria applied after the outcome could occur | Assess eligibility at time zero only |
| Delayed entry | Patients enter the risk set after the start of follow-up | Account for left truncation in the risk set |
| Immortal time from a grace period | A 30-day window to start treatment | Time-varying exposure or landmark with treatment defined at the landmark |

**Immortal time arithmetic.** Suppose a treated group must survive 7 days to be enrolled, and the 30-day mortality is 10% in the untreated. If 3% of patients die in the first 7 days, the treated group's observed mortality is diluted by the 3% who could never be counted. The apparent hazard ratio is biased downward even if the treatment does nothing. The bias grows with the length of the immortal period and with the event rate during it.

### Step 3. Define the event and the competing events

**Input:** The clinical endpoint, the data source, and the ascertainment method.

**Output:** A hierarchy of event types, with the event of interest and every competing event named.

**Why it exists:** Death, transplant, or discharge can prevent the event. Whether these are censoring, competing events, or part of a composite changes the estimand.

| Event type | Role | If ignored |
|---|---|---|
| Readmission | Event of interest | — |
| Death before readmission | Competing event | 1 - Kaplan-Meier overestimates readmission |
| Elective admission | Not an event; censoring or exclusion | Misclassified outcome |
| End of insurance enrollment | Censoring | Informative if related to health |
| Study end | Administrative censoring | Usually non-informative |

**Ascertainment:** Define the event with codes, a validation sample, and a rule for multiple events. For recurrent events, decide whether the analysis is for the first event, the rate of events, or the probability of at least one event. These are different estimands.

### Step 4. Specify the censoring rule and its assumption

**Input:** Every way a patient can leave observation.

**Output:** A censoring indicator and a statement of the assumption required for each type.

**Why it exists:** Kaplan-Meier and Cox assume that censoring is independent of the event time conditional on the covariates. If censoring is informative, the risk set is selected.

| Censoring type | Assumption | Check |
|---|---|---|
| Administrative (study end) | Independent of prognosis | Calendar date is fixed in advance |
| Loss to follow-up | Independent of prognosis given covariates | Compare censored and observed patients |
| Competing event treated as censoring | Independent of the event of interest | Use a competing-risk method instead |
| Treatment discontinuation | Independent of prognosis | Time-varying exposure or IPCW |
| Withdrawal for toxicity | Informative | Sensitivity analysis; composite endpoint |

**Edge case:** Censoring at treatment discontinuation in a trial of a drug that prevents a progressive disease is rarely independent. Patients who stop because of adverse effects may have different prognosis from those who continue. Report the discontinuation rate by arm and run a sensitivity analysis.

### Step 5. Build and validate the analysis dataset

**Input:** The raw cohort, the time zero, the event dates, and the censoring dates.

**Output:** One row per patient (or one row per time interval for time-varying analyses) with time, event indicator, and covariates.

**Why it exists:** Most survival errors are data errors: negative follow-up, events after censoring, duplicate records, or a time scale that mixes days and months.

Checks:

- Follow-up time is positive for every patient.
- No event occurs after the censoring date.
- No patient has two conflicting event dates without a rule for the first event.
- The time unit is consistent across the cohort.
- The number at risk at time zero equals the cohort size.
- The number at risk never increases in a single-event analysis.
- Censoring dates are not missing; a missing date becomes a data error, not a censoring event.

**Time scale.** Use the scale that matches the clinical process: days for 30-day readmission, months for cancer survival, years for device revision. The scale affects the tie structure, the baseline hazard, and the interpretability of the hazard ratio. Do not mix scales within a cohort. If the analysis uses age as the time scale, patients are compared at the same age rather than the same time since diagnosis, which is a different and sometimes better adjustment for age.

**Zero events in a group.** If one group has no events, the Kaplan-Meier curve is flat at 1 and the hazard ratio is not estimable. Report the risk difference with an exact or Wilson interval, the number at risk, and the follow-up time. A zero-event arm is not evidence of no effect; it is evidence that the study was too small to observe the event.

### Step 6. Describe the cohort before modelling

**Input:** The analysis dataset.

**Output:** Person-time, event counts, incidence rates, and unadjusted Kaplan-Meier or cumulative incidence curves by group.

**Why it exists:** The descriptive table reveals implausible event rates, unequal follow-up, and early separation or crossing that a model would hide.

```text
Incidence rate = number of events / total person-time at risk
```

Report the rate with its unit, for example 18.4 events per 1,000 person-years. Report the number at risk at each plotted time point under the curve. A Kaplan-Meier plot without a number-at-risk table is not interpretable after the first few censoring times.

### Step 7. Choose the estimator

**Input:** The estimand, the event type, the number of events, and the shape of the hazard.

**Output:** A named estimator with its assumptions.

**Why it exists:** The estimator determines the target and the assumptions; a Cox model and a cumulative incidence function answer different questions.

| Situation | Estimator | Target | Assumption |
|---|---|---|---|
| Unadjusted risk by group | Kaplan-Meier | S(t) | Independent censoring |
| Unadjusted risk with competing events | Aalen-Johansen | CIF(t) | Independent censoring |
| Adjusted hazard ratio, proportional hazards plausible | Cox | HR | Proportional hazards |
| Adjusted hazard ratio, competing event, etiologic question | Cause-specific Cox | Cause-specific HR | Independent censoring |
| Adjusted subdistribution hazard, prognostic question | Fine-Gray | Subdistribution HR | Independent censoring |
| Small number of events | Cox with few covariates, or parametric | HR | Proportional hazards; stable model |
| Hazard known to follow a shape | Weibull, exponential, Gompertz | HR and S(t) | Parametric form |
| Non-proportional hazards | RMST difference, or time-varying coefficient | Difference in means | Follow-up to the horizon |
| Recurrent events | Andersen-Gill or frailty model | Rate ratio | Model-specific |

**How many events?** A common heuristic is at least 10 events per covariate in a Cox model. With 15 events, a model with treatment and two confounders is already at the limit. Sparse-event Cox models produce biased coefficients and optimistic intervals.

### Step 8. Fit the Cox model and handle ties

**Input:** The time, event, and covariate data.

**Output:** Coefficients, standard errors, and the partial likelihood.

**Why it exists:** Ties are common when time is recorded in days or months. The tie method changes the coefficient slightly and the standard error more.

| Tie method | When | Consequence |
|---|---|---|
| Breslow | Default in many packages; few ties | Slight bias when many ties |
| Efron | Many ties; recommended default | Closer to the exact partial likelihood |
| Exact | Very few tied events | Computationally expensive |

The coefficient is a log hazard ratio. Exponentiate it for the hazard ratio, and report the confidence interval on the hazard ratio scale. A hazard ratio of 0.57 means the instantaneous event rate is 43% lower among those still at risk, not that 43% fewer patients have the event.

### Step 9. Check proportional hazards

**Input:** The fitted Cox model and the event times.

**Output:** A global test, a per-covariate test, and a plot of the scaled Schoenfeld residuals against time.

**Why it exists:** The Cox coefficient is a weighted average of time-specific effects. If the effect changes, the average can be close to 1 while the early and late effects are in opposite directions.

Checks:

1. **Global and per-covariate tests** based on scaled Schoenfeld residuals. A small p-value suggests non-proportional hazards.
2. **Plot of scaled residuals against time** with a smoothed line. A slope suggests a changing coefficient.
3. **Treatment-by-time interaction** in an extended Cox model. A significant interaction is direct evidence.
4. **Kaplan-Meier curves** for visual context. Crossing curves are a strong signal; parallel curves are supportive but not proof.
5. **Landmark comparison.** Fit the model before and after a landmark time and compare the hazard ratios. A large change suggests non-proportionality.

**Power caveat:** With few events, the test has low power. Failing to reject is not evidence that the assumption holds. Report the test and the number of events, and state the limitation.

### Step 10. Respond to non-proportional hazards

**Input:** The diagnostic result.

**Output:** A revised analysis that answers the decision question without relying on a constant hazard ratio.

**Why it exists:** A time-averaged hazard ratio can hide a crossing effect; the remedy must match the pattern.

| Pattern | Remedy | What is reported | Limitation |
|---|---|---|---|
| Early benefit that wanes | Time-varying coefficient; piecewise HR | HR by interval | Intervals may be wide |
| Curves cross | RMST difference, or a milestone risk difference | Difference in event-free time or risk at a horizon | Horizon-dependent |
| Delayed benefit | Landmark analysis at the delay | HR after the delay | Excludes early events |
| Harm early, benefit late | Composite of harm and benefit, or net benefit | Both components | Value judgment required |
| Non-proportionality from a strong prognostic covariate | Stratify the baseline hazard by that covariate | Stratified Cox HR | No estimate for the stratifying variable |

**RMST** is the area under the survival curve up to a horizon τ:

```text
RMST(τ) = integral from 0 to τ of S(t) dt
```

It is interpreted as the average event-free time within the horizon. The difference between groups is in months or days, which is easier to communicate than a hazard ratio and does not require proportional hazards. It requires that follow-up extends to τ for enough patients.

### Step 11. Handle competing risks according to the question

**Input:** The event hierarchy and the decision.

**Output:** A cause-specific analysis, a subdistribution analysis, or both, with a statement of which question each answers.

**Why it exists:** The two models answer different questions and can disagree in direction; the choice must follow the decision.

| Question | Model | Target | Interpretation |
|---|---|---|---|
| Does the treatment affect the rate of the event among those still at risk? | Cause-specific Cox | Cause-specific HR | Etiologic |
| Does the treatment affect the probability of the event by time t? | Fine-Gray | Subdistribution HR | Prognostic and absolute |
| What is the probability of the event by time t? | Aalen-Johansen | CIF | Descriptive |
| Does the treatment affect the combined burden of event or death? | Composite endpoint | HR for the composite | Decision-relevant if both are bad |
| Is the treatment's effect on the event mediated by the competing event? | Both models plus sensitivity | Two estimates | Requires interpretation |

**The coefficients can disagree in direction.** A treatment that reduces the cause-specific hazard of readmission but increases mortality can have a subdistribution hazard ratio above 1, because patients who would have died are kept in the risk set for readmission. Neither model is wrong; they answer different questions.

### Step 12. Model time-varying exposure when treatment starts after time zero

**Input:** The exposure start date for each patient.

**Output:** A start-stop dataset where each patient contributes unexposed person-time before initiation and exposed person-time after.

**Why it exists:** This is the direct fix for immortal time. The patient is counted as unexposed until the treatment starts, then as exposed. No time is excluded, and no group is guaranteed to survive.

Data structure:

| Patient | Start | Stop | Event | Exposed |
|---|---:|---:|---:|---:|
| 1 | 0 | 7 | 0 | 0 |
| 1 | 7 | 19 | 1 | 1 |
| 2 | 0 | 30 | 0 | 0 |

**Landmark alternative:** Choose a fixed landmark (for example, day 7), classify treatment status at the landmark, and start follow-up at the landmark for both groups. This is simpler but discards patients who had the event before the landmark and changes the estimand to the landmark population.

### Step 13. Run sensitivity analyses

**Input:** The primary analysis and a list of assumptions.

**Output:** A short table of sensitivity estimates, each tied to one assumption.

**Why it exists:** The primary estimate is conditional on assumptions; the table shows how much the conclusion depends on them.

| Assumption | Sensitivity analysis | What a change means |
|---|---|---|
| Independent censoring | Inverse probability of censoring weights; worst-case imputation | Censoring is informative |
| Proportional hazards | RMST difference; piecewise HR | The HR is time-averaged |
| No competing risk | CIF versus 1 - KM | The competing event matters |
| Correct event definition | Alternative code set; validation sample | Misclassification |
| No immortal time | Time-varying exposure versus naive | The naive design is biased |
| Correct functional form | Restricted cubic spline for continuous covariates | The model is misspecified |
| No unmeasured confounding | E-value or negative control outcome | Confounding could explain the result |

### Step 14. Report

**Input:** The estimates, diagnostics, and sensitivity analyses.

**Output:** A results section with a curve, a table, and a statement of the estimand.

**Why it exists:** A time-to-event result cannot be interpreted without the number at risk, the horizon, and the estimand.

The reporting checklist:

1. The estimand: population, treatment contrast, event, horizon, and summary.
2. Time zero and the censoring rule.
3. The number of patients, events, and person-time by group.
4. The Kaplan-Meier or cumulative incidence curve with a number-at-risk table.
5. The estimate at named horizons, with intervals.
6. The hazard ratio with its interval and the proportionality check.
7. The competing-risk method, if used, and the reason.
8. The sensitivity analyses.
9. The median or RMST, if reported, with the horizon.
10. The software and the tie method.

A report that satisfies items 1 to 10 can be reproduced by a second analyst. A report that omits item 2 cannot, because the risk set is undefined. A report that omits item 7 cannot be compared with another study, because the competing-risk method changes the number.

### Edge cases that change the estimate

| Edge case | What happens | Detection | Remedy |
|---|---|---|---|
| Sparse events | Coefficients biased away from the null; intervals too narrow | Fewer than 10 events per covariate | Reduce covariates; Firth-Cox; parametric model |
| Ties in time | Breslow and Efron give different SEs | Many events recorded on the same day | Use Efron; report the method |
| Left truncation | Survival is overestimated if delayed entry is ignored | Entry time later than time zero | Start the risk set at entry |
| Informative censoring | Risk set selected on prognosis | Censoring related to group and outcome | IPCW; sensitivity analysis |
| Competing event | 1 - Kaplan-Meier overstates the event probability | Death or transplant is common | Aalen-Johansen or Fine-Gray |
| Non-proportional hazards | The HR is a time-average that hides crossing | Schoenfeld test; landmark comparison | RMST; time-varying coefficient |
| Time-varying treatment | Immortal time if baseline-defined | Treatment start after time zero | Start-stop data; landmark |
| Recurrent events | First-event analysis ignores later events | Multiple events per patient | Andersen-Gill, frailty, or rate model |
| Outcome misclassification | Events attributed to the wrong cause | Validation sample | Chart review; sensitivity analysis |
| Calendar-time trend | The baseline hazard changes across enrollment periods | Enrollments span a practice change | Stratify by period; test interaction |

**Sparse events, in more detail.** A Cox model with 15 events and three covariates has 5 events per covariate. The coefficient is unstable, and the Wald interval is too narrow because the partial likelihood is far from quadratic. Options are to reduce to one covariate, to use a parametric model with few parameters, or to use a penalized Cox estimator. The choice must be reported, because it changes the estimand slightly and the interval materially.

**Left truncation, in more detail.** A patient who enters the cohort at month 6 is not at risk from month 0. If the analysis starts everyone at month 0, the patient contributes six months of event-free time that was never observed, and survival is overestimated. Start-stop data with entry at month 6 is the correct structure. This situation is common in claims data when enrollment begins after diagnosis.

### Branch points at a glance

| If the situation is | Then the default is | Because |
|---|---|---|
| Decision needs a probability by a horizon | Kaplan-Meier or Aalen-Johansen | The decision is on the risk scale |
| Decision needs a relative rate | Cox hazard ratio | The rate ratio is the natural parameter |
| Curves cross or the effect wanes | RMST difference or milestone risk | The HR is not constant |
| Death is common and prevents the event | Cumulative incidence with competing risk | 1 - KM overestimates |
| Treatment starts after time zero | Time-varying exposure | Avoids immortal time |
| Few events | Few covariates or a parametric model | Stability |
| Censoring is informative | IPCW or a sensitivity analysis | The risk set is selected |
| Recurrent events are the outcome | Rate model or Andersen-Gill | First-event analysis loses information |

**Recurrent events, in more detail.** A readmission study that counts only the first event answers "time to first readmission." A study that counts every readmission answers "rate of readmission." The first-event analysis discards information and changes the estimand; the rate analysis requires a model for within-patient correlation. Neither is universally preferred. The decision determines which is relevant: a penalty program that counts every readmission needs the rate, while a patient-facing decision about the first event needs the first-event analysis.

**Calendar time, in more detail.** If enrollment spans a period when the standard of care changed, the baseline hazard is not common across the cohort. Stratifying by calendar period, or including period as a covariate, prevents a time trend from being attributed to the exposure. This is a form of confounding that is easy to miss because the calendar date is not usually treated as a covariate.

The branch table is not a substitute for the estimand. A team that wants the probability of readmission by 30 days and reports a hazard ratio has answered a different question, even if every row of the table is satisfied.

## 6. Worked example

### 6.1 The question and the estimand

A heart-failure program is compared with usual care for the time to first unplanned readmission over 24 months. Twenty patients are available: ten in the program and ten in usual care. Patients who die before readmission are treated as a competing event in a sensitivity analysis; the primary analysis treats death as censoring to illustrate the difference.

The estimand is the cumulative incidence of first readmission at 24 months, program minus usual care, with a hazard ratio as a secondary summary. The population is adults discharged alive after a heart-failure admission. Time zero is discharge.

### 6.2 The data

| Patient | Arm | Time (months) | Event |
|---:|---|---:|---|
| 1 | Program | 4 | Readmission |
| 2 | Program | 6 | Readmission |
| 3 | Program | 8 | Readmission |
| 4 | Program | 10 | Censored |
| 5 | Program | 12 | Readmission |
| 6 | Program | 14 | Readmission |
| 7 | Program | 16 | Readmission |
| 8 | Program | 18 | Censored |
| 9 | Program | 20 | Readmission |
| 10 | Program | 24 | Censored |
| 11 | Usual care | 2 | Readmission |
| 12 | Usual care | 3 | Readmission |
| 13 | Usual care | 5 | Readmission |
| 14 | Usual care | 7 | Readmission |
| 15 | Usual care | 9 | Readmission |
| 16 | Usual care | 11 | Readmission |
| 17 | Usual care | 13 | Censored |
| 18 | Usual care | 15 | Readmission |
| 19 | Usual care | 17 | Readmission |
| 20 | Usual care | 19 | Censored |

### 6.3 Kaplan-Meier by hand

Program arm, 10 patients, 7 events and 3 censored:

| t | n at risk | d | 1 - d/n | S(t) | Greenwood SE |
|---:|---:|---:|---:|---:|---:|
| 4 | 10 | 1 | 0.9000 | 0.9000 | 0.0949 |
| 6 | 9 | 1 | 0.8889 | 0.8000 | 0.1265 |
| 8 | 8 | 1 | 0.8750 | 0.7000 | 0.1449 |
| 12 | 6 | 1 | 0.8333 | 0.5833 | 0.1610 |
| 14 | 5 | 1 | 0.8000 | 0.4667 | 0.1658 |
| 16 | 4 | 1 | 0.7500 | 0.3500 | 0.1602 |
| 20 | 2 | 1 | 0.5000 | 0.1750 | 0.1474 |

The risk set drops from 8 at t = 8 to 6 at t = 12 because patient 3 had the event at 8 and patient 4 was censored at 10. The curve is flat at t = 10, 18, and 24 because those are censoring times.

Usual-care arm, 10 patients, 8 events and 2 censored:

| t | n at risk | d | 1 - d/n | S(t) | Greenwood SE |
|---:|---:|---:|---:|---:|---:|
| 2 | 10 | 1 | 0.9000 | 0.9000 | 0.0949 |
| 3 | 9 | 1 | 0.8889 | 0.8000 | 0.1265 |
| 5 | 8 | 1 | 0.8750 | 0.7000 | 0.1449 |
| 7 | 7 | 1 | 0.8571 | 0.6000 | 0.1549 |
| 9 | 6 | 1 | 0.8333 | 0.5000 | 0.1581 |
| 11 | 5 | 1 | 0.8000 | 0.4000 | 0.1549 |
| 15 | 3 | 1 | 0.6667 | 0.2667 | 0.1501 |
| 17 | 2 | 1 | 0.5000 | 0.1333 | 0.1205 |

At 24 months:

```text
Program:     S(24) = 0.1750, so 1 - S = 0.8250 = 82.5% readmitted
Usual care:  S(24) = 0.1333, so 1 - S = 0.8667 = 86.7% readmitted
Difference:  82.5% - 86.7% = -4.2 percentage points
```

The median time to readmission is 14 months in the program arm and 9 months in usual care, the first times at which S(t) is at most 0.5.

### 6.4 Log-rank test by hand

For each event time, the expected number of events in the program arm is d × n1/n, where d is the total number of events at that time, n1 is the number at risk in the program arm, and n is the total at risk.

| t | n1 | n0 | d1 | d0 | Expected in program |
|---:|---:|---:|---:|---:|---:|
| 2 | 10 | 10 | 0 | 1 | 0.500 |
| 3 | 10 | 9 | 0 | 1 | 0.526 |
| 4 | 10 | 8 | 1 | 0 | 0.556 |
| 5 | 9 | 8 | 0 | 1 | 0.529 |
| 6 | 9 | 7 | 1 | 0 | 0.562 |
| 7 | 8 | 7 | 0 | 1 | 0.533 |
| 8 | 8 | 6 | 1 | 0 | 0.571 |
| 9 | 7 | 6 | 0 | 1 | 0.538 |
| 11 | 6 | 5 | 0 | 1 | 0.545 |
| 12 | 6 | 4 | 1 | 0 | 0.600 |
| 14 | 5 | 3 | 1 | 0 | 0.625 |
| 15 | 4 | 3 | 0 | 1 | 0.571 |
| 16 | 4 | 2 | 1 | 0 | 0.667 |
| 17 | 3 | 2 | 0 | 1 | 0.600 |
| 20 | 2 | 0 | 1 | 0 | 1.000 |

```text
Observed in program = 7
Expected in program = 8.926
Variance = 3.413
Chi-square = (7 - 8.926)^2 / 3.413 = 3.709 / 3.413 = 1.086
p = 0.297
```

The program arm had 1.93 fewer readmissions than expected if the arms were exchangeable. The direction favours the program, but the difference is compatible with chance.

### 6.5 Cox model

```text
log HR = -0.5580
HR = exp(-0.5580) = 0.5724
SE(log HR) = 0.5422
95% CI = exp(-0.5580 +/- 1.96 × 0.5422)
       = exp(-1.6207, 0.5047)
       = (0.1978, 1.6564)
Wald p = 0.303
```

The point estimate suggests a 43% lower instantaneous readmission rate in the program arm, but the interval runs from a 80% reduction to a 66% increase. The estimate is too imprecise for a decision.

### 6.6 Diagnostic checks

**Check 1. Risk-set arithmetic.** The program risk set goes 10, 9, 8, 6, 5, 4, 2. Each drop is accounted for by one event or one censoring: 10 to 9 at t = 4 (event), 9 to 8 at t = 6 (event), 8 to 6 at t = 12 (event at 8 and censoring at 10), and so on. If the risk set dropped without an event or censoring, the data would have a missing record.

**Check 2. Kaplan-Meier monotonicity.** S(t) never increases, and it is flat at t = 10, 18, and 24. A rise in S(t) indicates a coding error in the risk set or a negative time.

**Check 3. Log-rank consistency.** The observed count in the program arm is 7, which equals the number of program events in the table. The expected count is between the observed counts of the two arms. The chi-square is positive, as it must be.

**Check 4. Direction agreement.** The log-rank observed-minus-expected is negative in the program arm, and the Cox hazard ratio is below 1. Both say the program arm had fewer events than expected. If they disagreed, the group coding or the time variable would be suspect.

**Check 5. Naive proportion versus Kaplan-Meier.** The naive proportions readmitted are 7/10 = 70% in the program arm and 8/10 = 80% in usual care, a difference of -10 percentage points. The Kaplan-Meier risks at 24 months are 82.5% and 86.7%, a difference of -4.2 points. The naive analysis counts censored patients as event-free and underestimates both risks; it also overstates the difference. This is the clearest demonstration of why censoring cannot be ignored.

**Check 6. Proportional hazards.** A landmark split at 12 months gives a hazard ratio of 0.553 before 12 months and 0.622 after 12 months. The two are similar, which is consistent with proportional hazards, but the late estimate has a standard error of 1.003 and its interval includes a wide range. The test has almost no power with 15 events. Report the check and the limitation; do not claim the assumption is verified.

**Check 7. Competing-risk arithmetic.** The cumulative incidences should sum to 1 minus the probability of being event-free at 24 months:

```text
CIF readmission 0.6083 + CIF death 0.2167 + event-free 0.1750 = 1.0000
```

If the three do not sum to 1, the cause coding or the risk set is wrong. The event-free probability of 0.1750 is the Kaplan-Meier estimate of remaining free of both events.

**Check 8. Independence of censoring.** The six censored patients are assumed to be censored independently of readmission. This cannot be verified from the data. The sensitivity analysis would compare the primary result with a worst-case analysis that treats all censored program patients as readmitted, and with one that treats them as event-free. If the conclusion changes, the censoring assumption is carrying the result.

### 6.7 Vary one assumption: death as a competing event

Suppose two program patients who were readmitted at 6 and 14 months actually died first, at those same times, without a readmission. The event of interest is still readmission, and death is now a competing event.

Using the Aalen-Johansen estimator for the program arm:

```text
CIF readmission at 24 months = 0.6083 = 60.8%
CIF death at 24 months        = 0.2167 = 21.7%
Sum                           = 0.8250 = 82.5%
```

The naive Kaplan-Meier that censors death gives 1 - S = 0.7539 = 75.4% readmission. The Kaplan-Meier overestimates the readmission probability by 14.6 percentage points because it removes patients who died from the risk set and then treats the remaining patients as if they could only be readmitted.

For a like-for-like comparison, reclassify two usual-care readmissions as deaths at days 5 and 11. The Aalen-Johansen estimates in the usual-care arm are 66.7% readmission and 20.0% death, so the 24-month readmission risk difference is 60.8% - 66.7% = -5.9 percentage points. The program's advantage is smaller under the competing-risk analysis than under the naive 1 - Kaplan-Meier comparison (82.5% vs 86.7%, -4.2 points), and the death risks are similar (21.7% vs 20.0%).

The conclusion changes materially. Under the primary analysis, the program arm has an 82.5% readmission risk by 24 months. Under the competing-risk analysis, the risk of readmission is 60.8% and the risk of death is 21.7%. The composite of readmission or death is still 82.5%. Which number answers the decision depends on whether the decision concerns readmission burden or the combined burden of readmission and death.

### 6.8 What to report

Report the cumulative incidence of readmission at 24 months with death as a competing event: **60.8% in the program arm versus 66.7% in usual care (RD -5.9 percentage points)**, with the cause-specific hazard ratio as a secondary measure. Report the composite of readmission or death alongside it, because a program that reduces readmission by increasing death is not a success.

For this small example, the honest conclusion is: "The program was associated with a lower instantaneous readmission rate (HR 0.57, 95% CI 0.20 to 1.66), but the estimate is imprecise. The 24-month readmission cumulative incidence was 60.8% with death as a competing event, and the composite of readmission or death was 82.5%. The data are compatible with a substantial benefit and with harm; a larger cohort with adequate follow-up is required."

The estimate to report is the cumulative incidence at 24 months with the competing risk, not the hazard ratio alone, because the decision is about the probability of readmission. The hazard ratio is reported as a secondary summary with its proportionality caveat.

## 7. Code, query, or calculation

The script implements the Kaplan-Meier estimator with Greenwood standard errors, the log-rank test, a Cox model with Breslow ties, and the Aalen-Johansen cumulative incidence estimator. It uses only the standard library.

**Assumptions:**

- Censoring is independent of the event time within each arm.
- The event times are recorded in whole months; no two patients share an event time in this dataset.
- The Cox model includes a single binary covariate, so the proportional hazards assumption is the only structural assumption.
- The competing-risk example applies to the program arm only, with two readmissions reclassified as deaths at the same times.

```python
import math

# time, event (1 = readmission, 0 = censored), arm (1 = program, 0 = usual care)
data = [
    (4,1,1),(6,1,1),(8,1,1),(10,0,1),(12,1,1),(14,1,1),(16,1,1),(18,0,1),(20,1,1),(24,0,1),
    (2,1,0),(3,1,0),(5,1,0),(7,1,0),(9,1,0),(11,1,0),(13,0,0),(15,1,0),(17,1,0),(19,0,0)
]

def km(rows):
    rows = sorted(rows)
    times = sorted(set(t for t, e in rows if e == 1))
    S, var, out = 1.0, 0.0, []
    for t in times:
        n = sum(1 for tt, e in rows if tt >= t)
        d = sum(1 for tt, e in rows if tt == t and e == 1)
        S *= (1 - d / n)
        if n - d > 0:
            var += d / (n * (n - d))
        out.append((t, n, d, S, S * math.sqrt(var)))
    return out

for arm in (1, 0):
    print(f"KM arm {arm}:")
    for t, n, d, s, se in km([(t, e) for t, e, a in data if a == arm]):
        print(f"  t={t:2d} n={n:2d} d={d} S={s:.4f} se={se:.4f}")

def logrank(data):
    times = sorted(set(t for t, e, a in data if e == 1))
    O1 = E1 = V = 0.0
    for t in times:
        n1 = sum(1 for tt, e, a in data if a == 1 and tt >= t)
        n0 = sum(1 for tt, e, a in data if a == 0 and tt >= t)
        d1 = sum(1 for tt, e, a in data if a == 1 and tt == t and e == 1)
        d0 = sum(1 for tt, e, a in data if a == 0 and tt == t and e == 1)
        n, d = n1 + n0, d1 + d0
        O1 += d1
        E1 += d * n1 / n
        if n >= 2:
            V += d * (n1 / n) * (1 - n1 / n) * (n - d) / (n - 1)
    return O1, E1, V, (O1 - E1) ** 2 / V

O1, E1, V, chi2 = logrank(data)
print(f"log-rank O1={O1:.1f} E1={E1:.3f} V={V:.3f} chi2={chi2:.3f} p={math.erfc(math.sqrt(chi2/2)):.4f}")

def cox(data):
    rows = sorted(data)
    times = sorted(set(t for t, e, a in rows if e == 1))
    beta = 0.0
    for _ in range(50):
        grad = info = 0.0
        for t in times:
            risk = [(tt, e) for tt, e, a in rows if tt >= t]
            d = sum(1 for tt, e in risk if tt == t and e == 1)
            xr = [(a, tt, e) for tt, e, a in rows if tt >= t]
            s0 = sum(math.exp(beta * a) for a, tt, e in xr)
            s1 = sum(a * math.exp(beta * a) for a, tt, e in xr)
            s2 = sum(a * a * math.exp(beta * a) for a, tt, e in xr)
            grad += sum(a for a, tt, e in xr if tt == t and e == 1) - d * s1 / s0
            info += d * (s2 / s0 - (s1 / s0) ** 2)
        step = grad / info
        beta += step
        if abs(step) < 1e-10:
            break
    se = 1 / math.sqrt(info)
    return beta, se

b, se = cox(data)
print(f"Cox logHR={b:.4f} HR={math.exp(b):.4f} se={se:.4f} "
      f"CI=({math.exp(b-1.96*se):.4f}, {math.exp(b+1.96*se):.4f})")

# Competing risks in the program arm: two readmissions become deaths
cr = [(4,1),(6,2),(8,1),(10,0),(12,1),(14,2),(16,1),(18,0),(20,1),(24,0)]
def aalen_johansen(rows, cause):
    times = sorted(set(t for t, e in rows if e != 0))
    S, cum = 1.0, 0.0
    for t in times:
        n = sum(1 for tt, e in rows if tt >= t)
        d = sum(1 for tt, e in rows if tt == t and e != 0)
        dk = sum(1 for tt, e in rows if tt == t and e == cause)
        cum += S * dk / n
        S *= (1 - d / n)
    return cum

cif_readmit = aalen_johansen(cr, 1)
cif_death = aalen_johansen(cr, 2)
km_naive = 1 - km([(t, 1 if e == 1 else 0) for t, e in cr])[-1][3]
print(f"competing risk: CIF readmission={cif_readmit:.4f} death={cif_death:.4f} "
      f"sum={cif_readmit+cif_death:.4f}")
print(f"1-KM readmission ignoring competing death={km_naive:.4f}")
```

**Expected output:**

```text
KM arm 1:
  t= 4 n=10 d=1 S=0.9000 se=0.0949
  t= 6 n= 9 d=1 S=0.8000 se=0.1265
  t= 8 n= 8 d=1 S=0.7000 se=0.1449
  t=12 n= 6 d=1 S=0.5833 se=0.1610
  t=14 n= 5 d=1 S=0.4667 se=0.1658
  t=16 n= 4 d=1 S=0.3500 se=0.1602
  t=20 n= 2 d=1 S=0.1750 se=0.1474
KM arm 0:
  t= 2 n=10 d=1 S=0.9000 se=0.0949
  t= 3 n= 9 d=1 S=0.8000 se=0.1265
  t= 5 n= 8 d=1 S=0.7000 se=0.1449
  t= 7 n= 7 d=1 S=0.6000 se=0.1549
  t= 9 n= 6 d=1 S=0.5000 se=0.1581
  t=11 n= 5 d=1 S=0.4000 se=0.1549
  t=15 n= 3 d=1 S=0.2667 se=0.1501
  t=17 n= 2 d=1 S=0.1333 se=0.1205
log-rank O1=7.0 E1=8.926 V=3.413 chi2=1.086 p=0.2973
Cox logHR=-0.5580 HR=0.5724 se=0.5422 CI=(0.1978, 1.6564)
competing risk: CIF readmission=0.6083 death=0.2167 sum=0.8250
1-KM readmission ignoring competing death=0.7539
```

**What the output demonstrates:**

- The Kaplan-Meier table matches the hand calculation in Section 6.3 at every event time. If a reader's software gives a different value at t = 12, the risk set at 12 has been computed with the wrong time rule.
- The log-rank expected count of 8.926 is the sum of the 15 expected values in Section 6.4. The chi-square of 1.086 matches the hand calculation to three decimals.
- The Cox hazard ratio of 0.5724 matches the reported estimate. The standard error of 0.5422 is large because there are 15 events and one covariate.
- The competing-risk CIF of 0.6083 is lower than the naive 1 - Kaplan-Meier of 0.7539. This is the expected direction: removing deaths from the risk set makes the remaining patients appear to have a higher readmission rate, so the naive method overstates the readmission probability.
- The sum of the CIFs and the event-free probability equals 1. That identity is a check on the Aalen-Johansen implementation.

> **Note:** The script uses Breslow ties, which is exact here because no two patients share an event time. With tied event times, an Efron implementation would differ slightly. Production code should use the tie method specified in the analysis plan and report it.

## 8. Data, standards, and regulatory context

### 8.1 Binding requirements and conventions

| Instrument | Status | What it constrains | What it does not constrain |
|---|---|---|---|
| ICH E9 (1998) | Binding through regional adoption | Analysis populations, pre-specification, censoring conventions | The choice of survival model |
| ICH E9(R1) (2019) | Binding through regional adoption | The estimand, including the time horizon and the handling of death | The estimator |
| FDA draft guidance, Approaches to Assessment of Overall Survival in Oncology Clinical Trials (August 2025) | Draft guidance; not final | OS assessment, pre-specified safety analyses, intent-to-treat methods, crossover handling | Non-oncology endpoints |
| FDA Multiple Endpoints in Clinical Trials (October 2022) | Final guidance | Multiplicity across OS, PFS, and other endpoints | The survival estimator |
| Consolidated Standards of Reporting Trials (CONSORT) 2010 | Journal editorial policy | Reporting of time-to-event outcomes, effect sizes, and precision | The analysis |
| Strengthening the Reporting of Observational Studies in Epidemiology (STROBE) and REporting of studies Conducted using Observational Routinely-collected health Data (RECORD) | Journal editorial policy | Reporting of observational time-to-event studies | Model choice |
| RECORD item 7.1 | Journal editorial policy | Code lists for exposures, outcomes, and confounders | Validation of the codes |

### 8.2 What the FDA OS draft says

The August 2025 draft recommends that overall survival be the primary endpoint when feasible in oncology trials, and that all randomized oncology trials be designed to assess OS as a pre-specified safety endpoint even when a progression-free survival (PFS) endpoint is primary. It recommends intent-to-treat analyses with Cox or log-rank methods, pre-specified harm thresholds, and interim analyses for futility or harm. It also asks sponsors to handle crossover and subsequent therapy through the estimand framework rather than through post hoc censoring.

The document is a draft. Its recommendations are not binding until finalized, and the comment period closed on 20 October 2025. It is cited here because it shows the regulatory direction: the time-to-event estimand, the handling of subsequent events, and the safety analysis are all part of the design, not an afterthought.

### 8.3 What ICH E9(R1) requires for time-to-event endpoints

The estimand must state the time horizon and the handling of intercurrent events. For survival, the intercurrent events typically include treatment discontinuation, subsequent therapy, and death. A treatment-policy strategy counts death as an event for OS; a hypothetical strategy may censor at subsequent therapy; a while-on-treatment strategy censors at discontinuation. These are different estimands, and the hazard ratios can differ.

The addendum also requires sensitivity analyses that target the same estimand under different assumptions, and supplementary analyses that target different estimands. A tipping-point analysis for informative censoring is a sensitivity analysis. A per-protocol analysis is a supplementary analysis if the primary is intention-to-treat.

### 8.4 Data standards

- **CDISC ADaM** time-to-event datasets carry the analysis time, the event indicator, and the censoring reason. The censoring reason is not optional: it distinguishes administrative censoring from loss to follow-up and from competing events.
- **RECORD** requires the code lists for the event, the competing events, and the censoring definitions, and a flow diagram of cohort selection.
- **STROBE** requires the number at risk over time, the median follow-up, and the reasons for censoring.
- **OMOP CDM** represents observation periods and death separately; a survival analysis must join them correctly or it will treat death as loss to follow-up.

### 8.5 What the standards do not cover

- They do not specify the tie method or the software.
- They do not require a competing-risk analysis, although reviewers increasingly expect one when the competing event is common.
- They do not settle the choice between a cause-specific and a subdistribution model.
- They do not validate the event definition. A code-based readmission definition can differ materially from a chart-reviewed one.

### 8.6 Data checks before any survival model

| Check | Question | Consequence if it fails |
|---|---|---|
| Time zero | Is there exactly one per patient, and is it before every event? | Immortal time or negative follow-up |
| Event date | Is the event date after time zero and on or before the censoring date? | Impossible person-time |
| Censoring reason | Is the reason recorded, or only the date? | Cannot distinguish administrative from informative censoring |
| Competing events | Are they coded separately from the event of interest? | 1 - KM bias |
| Death ascertainment | Is death captured through linkage or only through the registry? | Undercounted competing events |
| Time unit | Are all times in the same unit? | Absurd hazard and risk sets |
| Duplicate records | Is there one row per patient per interval? | Inflated risk sets |
| Follow-up completeness | What proportion has at least the planned follow-up? | Unstable estimates at late times |
| Event validation | What is the positive predictive value of the code definition? | Misclassified outcome |
| Calendar period | Does enrollment span a practice change? | Confounded baseline hazard |

The event-validation row is the one most often skipped. A readmission code can capture transfers, planned admissions, and observation stays differently across hospitals, and the error can differ by exposure group. A validation sample of 100 to 200 charts is usually enough to estimate the positive predictive value and to decide whether the definition is fit for the decision.

### 8.7 CDISC and OMOP representations

In CDISC Analysis Data Model (ADaM) time-to-event datasets, the analysis time, the event indicator, and the censoring reason are separate variables. The censoring reason must distinguish "completed follow-up," "lost to follow-up," "death," and "competing event." Collapsing them into a single censoring code destroys the information needed for a competing-risk analysis.

In the Observational Medical Outcomes Partnership (OMOP) Common Data Model, death is a separate table and observation periods are recorded per patient. A survival analysis that joins only the condition occurrence table and ignores the death table will treat death as loss to follow-up. The join must be explicit, and the analysis plan must state how deaths outside the observation period are handled.

### 8.8 Documentation that a reviewer will ask for

A survival analysis submitted for regulatory or payer review is usually accompanied by a request for:

1. The cohort flow diagram with the number excluded at each step and the reason.
2. The definition of time zero, with the source variable and the date logic.
3. The event definition with code lists and the validation result.
4. The censoring definitions and the number censored by reason and arm.
5. The number at risk at each plotted time.
6. The median follow-up, computed by reverse Kaplan-Meier or a stated method.
7. The tie method and the software version.
8. The proportionality test and its result.
9. The competing-risk method and the reason for the choice.
10. The sensitivity analyses for censoring and outcome definition.

The median follow-up deserves a note. The simple average of follow-up times overestimates follow-up when many patients are censored early, because it counts time after the last event. The reverse Kaplan-Meier method estimates the median follow-up by treating the event as censoring and censoring as the event. Whichever method is used, it must be stated, because "median follow-up 18 months" can mean two different things.

## 9. Industry practice and case evidence

### 9.1 Documented case: immortal time in pharmacoepidemiology

Suissa's 2008 paper in the American Journal of Epidemiology documents five cohort designs that produce immortal time bias and quantifies the bias as a function of the proportion of immortal person-time. In a UK General Practice Research Database cohort of 8,176 patients with lung cancer, crude designs that required a prescription or multiple prescriptions produced hazard ratios below 1 for warfarin or clopidogrel, suggesting protection. Time-dependent analyses moved the estimates toward the null: the exposure-based cohort changed from 0.73 to 1.05, and the multiple-event cohort from 0.48 to 0.91.

The lesson is that the bias is a design defect, not a modelling defect. No adjustment in the Cox model repairs it, because the exposed group's person-time before treatment is not in the data as exposed time. The remedy is to include all person-time and to define exposure as time-varying.

### 9.2 Documented practice: trial reporting of OS and PFS

Oncology trials commonly report OS and PFS as co-primary or primary-secondary endpoints, with Kaplan-Meier curves, number-at-risk tables, and Cox hazard ratios. The FDA Multiple Endpoints guidance addresses the multiplicity problem when both are tested. The 2025 OS draft adds the expectation that OS is collected and analyzed even when PFS is the primary endpoint, because a PFS benefit that does not translate into OS benefit, or that comes with a survival harm, changes the risk-benefit assessment.

### 9.3 Documented practice: competing risks in device and geriatric studies

Device registries and geriatric cohorts routinely report cumulative incidence with death as a competing event. The alternative, 1 - Kaplan-Meier, is increasingly rejected by reviewers when the competing event is common. The practical reason is that a 5-year revision risk of 12% computed by Kaplan-Meier can be 9% by cumulative incidence when 20% of patients die first, and the difference matters for informed consent and for pricing.

**Inference:** the direction of the difference is general; the magnitude depends on the death rate and the event rate. The 12% versus 9% example is illustrative, not a published estimate.

### 9.4 Practice pattern: the hazard ratio as a default

Many industry reports default to a hazard ratio because the Cox model is familiar and the software is standard. The hazard ratio is a useful relative measure, but it is not a risk, and it is not the quantity a patient or payer experiences. Mature teams report the hazard ratio with the cumulative incidence or RMST, so the decision-maker sees both the relative rate and the absolute probability.

### 9.5 Documented case: target trial emulation with time-to-event outcomes

The RCT-DUPLICATE initiative emulated 32 trials with claims data and used Cox models with propensity score matching. The reported correlation of 0.82 between emulated and randomized estimates, and 0.93 in the close-emulation subgroup, is evidence that a carefully designed time-to-event analysis can approximate a trial. It is not evidence that any Cox model on claims data is valid. The design elements, including time zero, eligibility, and outcome definitions, carry the result.

### 9.6 What the evidence does and does not establish

Three claims are often made from the cases above, and they differ in support:

| Claim | Status | Reason |
|---|---|---|
| Immortal time can produce a spurious protective effect | Documented | Suissa's worked example and the algebra of person-time |
| A carefully emulated observational analysis can approximate a trial | Documented on average | RCT-DUPLICATE correlation, with wide variation by emulation quality |
| Any Cox model on claims data is valid if it adjusts for enough covariates | Not supported | Confounding, time zero, and outcome definition are design issues |
| Cumulative incidence is always the right summary | Not supported | The estimand may be a rate, an RMST, or a composite |
| A non-significant Schoenfeld test proves proportional hazards | Not supported | The test has low power in small studies |

The distinction between the first two rows and the last three is the difference between evidence and inference. The first two are reported results; the last three are claims that the data do not support.

### 9.7 Payer and health technology assessment use

Payers and health technology assessment bodies increasingly request absolute survival quantities rather than hazard ratios. The reason is arithmetic: a hazard ratio does not determine the number of life-years or event-free months gained without the baseline hazard, and the baseline hazard differs across the populations a payer covers. A restricted mean survival time difference, or a difference in the proportion event-free at a named horizon, can be combined with a cost per life-year without reconstructing the hazard.

**Inference:** the shift toward absolute quantities is visible in methods guidance and in review comments, but the exact proportion of submissions that now include RMST is not documented in a single public statistic. The practical advice is to report both the hazard ratio and one absolute quantity, so the payer can use whichever is required.

### 9.7 What mature teams do differently

1. They draw the timeline before writing the model: time zero, exposure start, event, censoring, competing events.
2. They report the number at risk and the cumulative incidence at named horizons.
3. They check proportional hazards and report the check, including its power limitation.
4. They use a competing-risk method when death is common.
5. They run a time-varying exposure analysis when treatment starts after baseline.
6. They pre-specify the censoring rule and a sensitivity analysis for informative censoring.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Ignoring censoring | The denominator is the full cohort and the risk is low | Censored patients counted as event-free | Risk underestimated | Kaplan-Meier or person-time |
| Immortal time | Treated patients must survive to be classified | Exposure defined after time zero | Spurious benefit | Time-varying exposure; aligned time zero |
| 1 - KM with competing risk | Death is common but the report shows 1 - KM | Competing event censored | Event probability overstated | Aalen-Johansen or Fine-Gray |
| Hazard ratio read as risk ratio | "43% fewer patients" from HR 0.57 | HR is an instantaneous rate ratio | Effect overstated | Report cumulative incidence |
| Proportional hazards assumed | No Schoenfeld test reported | The coefficient is a time-average | Misleading single HR | Test; RMST or time-varying coefficient |
| Curves cross | The HR is near 1 despite a visible crossing | Early and late effects cancel | Null conclusion from a real effect | RMST; milestone analysis |
| Informative censoring | Censoring differs by arm and prognosis | Risk set selected | Biased survival | IPCW; sensitivity analysis |
| Left truncation ignored | Patients enter after time zero | Immortal person-time added | Survival overestimated | Start-stop data |
| Sparse events | 15 events, 5 covariates | Partial likelihood unstable | Biased HR; narrow CI | Fewer covariates; penalization |
| Ties handled inconsistently | Different results across software | Breslow versus Efron | Reproducibility failure | Pre-specify the tie method |
| Time unit mixing | Days and months in the same column | Data-entry error | Absurd hazard | Validate the time variable |
| Median reported when not reached | The curve ends above 0.5 | The median is not estimable | Fabricated summary | Report "not reached" |
| Recurrent events collapsed | Only the first event is analyzed | Information loss; different estimand | Wrong rate | Rate model or Andersen-Gill |
| Event definition not validated | Codes are used without chart review | Misclassification | Attenuated or exaggerated effect | Validation sample |
| Calendar trend ignored | Enrollment spans a practice change | Baseline hazard changes | Confounded exposure effect | Stratify by period |
| Time scale mixed | Days and months in one dataset | Data-entry or unit-conversion error | Absurd hazards and risk sets | Validate the time variable against dates |
| Follow-up truncated at an event | The curve ends when the last patient has the event | No administrative censoring | Survival at late times is unstable | Report the number at risk; truncate the curve |

The first four rows account for most of the serious errors in applied survival analysis. They are all design errors, and none of them produces a software warning.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| Independent censoring | The risk set remains representative | Censoring related to prognosis and group | Compare censored and observed; IPCW sensitivity |
| Correct time zero | The risk set starts at the right moment | Treatment defined after baseline | Timeline review; time-varying exposure |
| Correct event definition | The estimate refers to the intended endpoint | Code misclassification | Validation sample; alternative code set |
| No competing event, or correct handling | The CIF is the probability of the event | Death or transplant prevents the event | Aalen-Johansen; compare with 1 - KM |
| Proportional hazards | A single HR summarizes the effect | Time-varying effect | Schoenfeld residuals; landmark comparison |
| Correct functional form | The adjusted HR is valid | Nonlinear covariate effect | Splines; residual plots |
| No unmeasured confounding | The HR is causal | Prognostic variables affect treatment | DAG; E-value; negative control |
| Positivity | Both groups exist in every covariate stratum | A subgroup receives one treatment | Overlap; propensity distribution |
| Sufficient events | The estimator is stable | Fewer than 10 events per covariate | Events-per-variable count |
| No informative entry | Late entry is unrelated to prognosis | Sick patients enter later | Compare entry times; sensitivity |
| Stable calendar period | The baseline hazard is common | Practice change during enrollment | Stratify by period |
| Correct cause attribution | The competing-risk CIF is valid | Deaths attributed to the wrong cause | Validation sample; sensitivity analysis |
| Complete death ascertainment | Competing events are counted | Deaths outside the registry are missed | Linkage with vital records; capture-recapture |

**Limits of this topic.** The methods here are for a single event time per patient. Recurrent events, multistate models, and joint models for longitudinal and survival data are extensions not covered. The proportional hazards check has low power in small studies, and a non-significant test is not proof of proportionality. The competing-risk models assume independent censoring and a correctly specified subdistribution; neither is testable from the observed data alone. The worked example is a teaching dataset with 20 patients; it is not a basis for a clinical or coverage decision. The estimates in Sections 6 and 7 are reproducible from the printed data, and any difference between a reader's output and the printed output should be resolved before the method is applied to a real cohort.

## 12. Exercises

### Exercise 1. Specify the timeline (design, 50 minutes)

**Scenario.** A sepsis alert fires at the time of order entry for antibiotics. Analysts define exposure as "alert fired within 3 hours of the antibiotic order" and start follow-up at hospital admission. The outcome is 28-day mortality.

**Deliverable.** A timeline diagram with admission, alert, antibiotic order, and outcome; a corrected time zero; and a paragraph explaining whether the original design has immortal time.

**Acceptance criteria.**

- The diagram marks every event used to define exposure.
- The corrected design aligns time zero with the exposure or models exposure as time-varying.
- The explanation distinguishes immortal time from confounding.
- The estimand is stated after the correction.

### Exercise 2. Kaplan-Meier by hand (application, 50 minutes)

**Data.** Six patients with times 2, 4, 4, 6, 8 (censored), and 10 months; all except the patient at 8 had the event.

**Deliverable.** The risk-set table, S(t) at every event time, the median, and the Greenwood standard error at t = 4.

**Acceptance criteria.**

- The risk set at t = 4 is 5, not 6.
- The two events at t = 4 are handled as a single drop of 2/5.
- S(4) = 0.5 and the median is 4 months.
- The Greenwood variance is computed from the terms at t = 2 and t = 4.

### Exercise 3. Redesign an immortal-time analysis (design, 60 minutes)

**Scenario.** A cohort defines "early treatment" as a prescription filled within 14 days of discharge. Patients who die before day 14 are excluded because they could not fill the prescription. The early-treatment group has lower mortality.

**Deliverable.** A new design with a start-stop dataset, the person-time contributions for one example patient, and the estimand under the new design.

**Acceptance criteria.**

- The excluded patients are re-entered as unexposed person-time, not deleted.
- The start-stop table has non-overlapping intervals that cover the full follow-up.
- The estimand is explicit about the treatment contrast and the time zero.
- The answer explains why moving time zero to prescription date changes the population.

### Exercise 4. Interpret a proportional hazards check (critique, 45 minutes)

**Data.** A Cox model reports HR 0.57 (95% CI 0.20 to 1.66). A landmark split at 12 months gives HR 0.553 before 12 months and HR 0.622 after 12 months. The Schoenfeld test p-value is 0.61.

**Deliverable.** A one-paragraph interpretation, including what the test can and cannot establish, and a recommendation for the primary summary.

**Acceptance criteria.**

- The answer does not claim the assumption is proven.
- It notes the low power with 15 events.
- It recommends reporting the cumulative incidence alongside the HR.
- It distinguishes a time-varying effect from sampling variability.

### Exercise 5. Competing risks (application, 50 minutes)

**Data.** In the program arm of the worked example, 1 - Kaplan-Meier readmission at 24 months is 75.4% and the cumulative incidence with death as a competing event is 60.8%. The death CIF is 21.7%.

**Deliverable.** The absolute overstatement, the sum of the CIFs and the event-free probability, and a sentence for a clinical audience explaining which number answers "what is the chance this patient is readmitted?"

**Acceptance criteria.**

- The overstatement is calculated as 75.4 - 60.8 = 14.6 percentage points.
- The sum is shown to equal 1.
- The clinical sentence uses the competing-risk number and states the horizon.
- The answer explains why the Kaplan-Meier number is not wrong, only a different quantity.

### Exercise 6. Events required for a hazard ratio (calculation, 35 minutes)

**Deliverable.** Using the approximation that the required number of events is 4(z_alpha + z_beta)^2 / (ln HR)^2, compute the events required to detect an HR of 0.70 with 80% power at a two-sided alpha of 0.05. Then compute the events for an HR of 0.80 and compare.

**Acceptance criteria.**

- z_alpha = 1.96 and z_beta = 0.84 are stated.
- The arithmetic is shown.
- The comparison quantifies how much the required events grow as the effect shrinks.
- The answer notes that the formula is an approximation and that the actual sample size depends on accrual and follow-up.

### Exercise 7. Audit a survival claim (critique, 75 minutes)

**Claim.** "The program reduced readmissions by 43% (HR 0.57, p = 0.30). Kaplan-Meier curves show a clear benefit. Patients were classified as program participants if they enrolled within 7 days of discharge."

**Deliverable.** A red-flag list with at least six items, a corrected analysis plan, and a revised claim.

**Acceptance criteria.**

- At least three failure modes from Section 10 are identified.
- The immortal-time problem is named and corrected.
- The difference between a hazard ratio and a risk is explained.
- The revised claim reports a cumulative incidence with a horizon and an interval.

### Exercise 8. Write a time-to-event analysis plan (design, 120 minutes)

**Scenario.** A device registry compares two implant types for 5-year revision, with death as a competing event and staggered enrollment.

**Deliverable.** A two-page plan with the estimand, time zero, event and competing-event definitions, censoring rules, the primary estimator, the proportionality check, the competing-risk approach, the sensitivity analyses, and the table shells.

**Acceptance criteria.**

- The estimand states the 5-year cumulative incidence and the treatment contrast.
- The plan distinguishes cause-specific and subdistribution analyses and says which is primary.
- The censoring rules cover administrative end, loss to follow-up, and death.
- The table shells include the number at risk, the CIF at 1 and 5 years, and the HR with its interval.
- At least two sensitivity analyses are tied to named assumptions.

## 13. Self-check answers

**Answer 1 (Exercise 2 Kaplan-Meier).**

| t | n at risk | d | 1 - d/n | S(t) |
|---:|---:|---:|---:|---:|
| 2 | 6 | 1 | 5/6 = 0.8333 | 0.8333 |
| 4 | 5 | 2 | 3/5 = 0.6000 | 0.5000 |
| 6 | 3 | 1 | 2/3 = 0.6667 | 0.3333 |
| 10 | 1 | 1 | 0 | 0.0000 |

The patient censored at 8 reduces the risk set from 3 at t = 6 to 1 at t = 10, because the patient at 6 had the event and the patient at 8 was censored: 3 - 1 - 1 = 1.

The median is 4 months, the first time S(t) is at most 0.5.

Greenwood variance at t = 4:

```text
Var(S) = S(4)^2 × [ 1/(6×5) + 2/(5×3) ]
       = 0.25 × [0.033333 + 0.133333]
       = 0.25 × 0.166667 = 0.041667
SE = 0.2041
```

The 95% confidence interval on the log-log scale is approximately (0.15, 0.84) for S(4). The point estimate of 0.5 is imprecise with six patients.

**Answer 2 (Exercise 4 interpretation).**

The hazard ratio is 0.57, but the interval runs from 0.20 to 1.66. The early and late hazard ratios, 0.553 and 0.622, are similar, so there is no evidence of a time-varying effect. The Schoenfeld p-value of 0.61 is consistent with proportional hazards, but with 15 events the test has very low power and cannot establish the assumption. The appropriate conclusion is that the data are compatible with a constant effect and with a moderate time-varying effect. Report the cumulative incidence at 24 months as the primary summary, with the hazard ratio and the proportionality caveat as secondary. Do not describe the result as a 43% reduction in readmissions; the hazard ratio is a rate ratio, not a risk ratio.

**Answer 3 (Exercise 5 competing risks).**

```text
Overstatement = 75.4% - 60.8% = 14.6 percentage points
Sum = 60.8% + 21.7% + 17.5% = 100.0%
```

The clinical sentence: "Among patients like those in the program arm, about 61 in 100 are readmitted within 24 months and about 22 in 100 die before a readmission." The Kaplan-Meier number of 75.4% answers a different question: the readmission probability if death could not occur. It is not an arithmetic error; it is the complement of a survival curve that censors deaths. The competing-risk number is the one that describes the patients' experience.

**Answer 4 (Exercise 6 events).**

```text
HR = 0.70; ln(0.70) = -0.3567
Events = 4 × (1.96 + 0.84)^2 / (-0.3567)^2
       = 4 × 7.84 / 0.12722
       = 31.36 / 0.12722
       = 246.5, so about 247 events
```

For HR = 0.80:

```text
ln(0.80) = -0.2231
Events = 4 × 7.84 / 0.04979 = 31.36 / 0.04979 = 629.8, so about 630 events
```

A reduction in the hazard ratio from 0.70 to 0.80 more than doubles the required events. The formula is an approximation for a log-rank test with equal allocation and no loss to follow-up. The actual sample size also depends on the event rate, accrual, and follow-up time.

**Answer 5 (Exercise 3 redesign).**

The corrected design uses a start-stop dataset. For a patient discharged on day 0, prescribed on day 10, and readmitted on day 25:

| Start | Stop | Event | Exposed |
|---:|---:|---:|---:|
| 0 | 10 | 0 | 0 |
| 10 | 25 | 1 | 1 |

The patient contributes 10 unexposed days and 15 exposed days. Patients who die before day 10 contribute unexposed person-time and an event. The estimand is the hazard ratio for the time-varying exposure, comparing the period after prescription with the period before it, with time zero at discharge. Moving time zero to the prescription date would answer a different question and would exclude the early deaths, which is the original bias in another form.

**Answer 6 (Exercise 7 revised claim).**

"The program was associated with a lower instantaneous readmission rate (HR 0.57, 95% CI 0.20 to 1.66; p = 0.30), but the estimate is imprecise. The 24-month cumulative incidence of readmission, with death as a competing event, was 60.8% in the program arm; the usual-care comparison is reported in the table. Because exposure was defined by enrollment within 7 days of discharge, the primary analysis uses a time-varying exposure to avoid immortal time. The data are compatible with a substantial benefit and with harm."

**Answer 7 (Exercise 1 rubric).**

A strong answer:

- Marks admission, alert, order, and 28-day outcome on the timeline.
- Notes that exposure defined by alert within 3 hours of the order is a post-baseline variable, and that starting follow-up at admission gives the alert group guaranteed survival to the order.
- Aligns time zero to the antibiotic order or models the alert as time-varying.
- States the estimand as the effect of the alert on 28-day mortality among patients with a suspected infection, with a clear contrast.
- Distinguishes the time-zero problem from confounding by indication.

**Answer 8 (Exercise 8 rubric).**

A strong plan begins with one sentence that a clinician could repeat: "Among patients receiving implant A versus implant B, the probability of revision within five years, with death as a competing event, is X% versus Y%, a difference of Z percentage points." Every methodological choice after that sentence either supports or qualifies it.

A strong plan:

- States the 5-year cumulative incidence of revision as the primary estimand, with death as a competing event.
- Uses a new-user design with time zero at implantation.
- Defines revision by procedure codes with a validation sample, and death by the registry or linkage.
- Uses Aalen-Johansen for the primary descriptive estimate and a Fine-Gray model for the adjusted subdistribution hazard, with a cause-specific Cox model as a secondary analysis.
- Checks proportional hazards for the cause-specific model and reports the check.
- Includes sensitivity analyses for informative censoring, code misclassification, and unmeasured confounding.
- Provides table shells with number at risk, CIF at 1 and 5 years, and hazard ratios with intervals.

## 14. Interview questions

**1. What is censoring, and what assumption does it require?**

A strong answer defines right censoring, distinguishes administrative from loss to follow-up, and states that Kaplan-Meier and Cox require censoring to be independent of the event time given the covariates. A weak answer says censoring is "missing data" without the assumption.

**2. Why is immortal time bias a design problem rather than a modelling problem?**

A strong answer explains that the exposed group's pre-treatment person-time is misclassified or excluded, so no covariate adjustment can recover it; the remedy is a time-varying exposure or an aligned time zero. A weak answer says "adjust for it in the Cox model."

**3. What is the difference between a hazard ratio and a risk ratio?**

A strong answer says the hazard is an instantaneous rate among those at risk, the risk is a cumulative probability, and the HR can differ substantially from the RR when the event is common or the effect changes over time. A weak answer treats them as interchangeable.

**4. How do you test proportional hazards?**

A strong answer describes scaled Schoenfeld residuals, the Grambsch-Therneau test, a treatment-by-time interaction, and landmark comparisons, and notes the power limitation. A weak answer says "plot the Kaplan-Meier curves."

**5. What do you do when proportional hazards fails?**

A strong answer offers RMST, a time-varying coefficient, stratification, or a milestone risk difference, and ties the choice to the decision. A weak answer says "use a different model" without naming one.

**6. When is 1 - Kaplan-Meier wrong?**

A strong answer explains that it estimates the event probability in a world without competing events, and that the cumulative incidence is the correct probability when death or another event prevents the outcome. A weak answer says it is always wrong.

**7. What is the difference between a cause-specific and a subdistribution hazard?**

A strong answer defines both risk sets and explains that the cause-specific model answers an etiologic question while the subdistribution model answers a prognostic one; the coefficients can differ in direction. A weak answer says they are two ways to do the same thing.

**8. How many events do you need for a Cox model?**

A strong answer gives the 10-events-per-covariate heuristic, explains that it is a heuristic, and names remedies for sparse events. A weak answer gives a fixed sample size without reference to events.

**9. What is RMST, and when would you use it?**

A strong answer defines the area under the survival curve to a horizon, explains that it does not require proportional hazards, and notes that it depends on the horizon and requires follow-up to it. A weak answer calls it "the average survival time" without the horizon.

**10. Why is the median not always reportable?**

A strong answer says the median requires the curve to reach 0.5 and that "not reached" is the correct report otherwise. A weak answer extrapolates a median from a curve that never crosses 0.5.

**11. How would you handle recurrent readmissions?**

A strong answer distinguishes first-event, rate, and recurrent-event estimands, and names Andersen-Gill, frailty, or a rate model. A weak answer counts all events as if they were independent.

**12. What would make you distrust a survival analysis?**

A strong answer lists time-zero misalignment, unreported censoring, 1 - KM with competing risks, an unchecked HR, few events, and code-based outcome definitions without validation. A weak answer says "a small sample" and stops.

## 15. Key terms

- **Aalen-Johansen estimator:** a nonparametric estimator of the cumulative incidence function with competing risks.
- **Administrative censoring:** end of follow-up at a fixed calendar date.
- **Cause-specific hazard:** the event rate among those still at risk, treating competing events as censoring.
- **Censoring:** end of observation without the event.
- **Competing risk:** an event that prevents the event of interest.
- **Cox model:** a semiparametric proportional hazards regression model.
- **Cumulative incidence function:** the probability of an event by time t in the presence of competing risks.
- **Cumulative hazard:** the integral of the hazard.
- **Efron method:** a tie-handling approximation for the Cox partial likelihood.
- **Fine-Gray model:** a regression model for the subdistribution hazard.
- **Greenwood variance:** the variance estimator for the Kaplan-Meier curve.
- **Hazard:** the instantaneous event rate among those at risk.
- **Hazard ratio:** the ratio of hazards between groups.
- **Immortal time:** follow-up during which the outcome cannot occur.
- **Incidence rate:** events divided by person-time.
- **Informative censoring:** censoring related to prognosis.
- **Kaplan-Meier estimator:** the product-limit estimator of survival.
- **Landmark analysis:** an analysis starting at a fixed time after baseline.
- **Left truncation:** entry into the risk set after time zero.
- **Log-rank test:** a test comparing observed and expected events across groups.
- **Median survival time:** the first time S(t) is at most 0.5.
- **Non-proportional hazards:** a hazard ratio that changes over time.
- **Person-time:** total time at risk.
- **Proportional hazards:** a constant hazard ratio over time.
- **Restricted mean survival time:** the area under the survival curve to a horizon.
- **Right censoring:** observation ending after the last contact.
- **Risk set:** patients under observation and event-free at time t.
- **Schoenfeld residuals:** residuals used to diagnose proportional hazards.
- **Subdistribution hazard:** the hazard in a risk set that retains prior competing events.
- **Time-varying covariate:** a covariate whose value changes during follow-up.
- **Breslow method:** a tie-handling approximation for the Cox partial likelihood.
- **Reverse Kaplan-Meier:** a method for estimating median follow-up by swapping the event and censoring indicators.
- **Partial likelihood:** the likelihood used to estimate Cox coefficients without specifying the baseline hazard.
- **Baseline hazard:** the unspecified hazard function in a Cox model.
- **Event indicator:** the variable that distinguishes an event from a censoring at the end of follow-up.
- **Study horizon:** the time at which a cumulative incidence or RMST is evaluated.

## 16. Sources

All URLs were accessed on 25 October 2026.

1. International Council for Harmonisation. *ICH E9: Statistical Principles for Clinical Trials*. Step 4, 5 February 1998. https://database.ich.org/sites/default/files/E9_Guideline.pdf
2. International Council for Harmonisation. *ICH E9(R1) Addendum on Estimands and Sensitivity Analysis in Clinical Trials*. Step 4, 20 November 2019. https://database.ich.org/sites/default/files/E9-R1_Step4_Guideline_2019_1203.pdf
3. United States Food and Drug Administration. *Approaches to Assessment of Overall Survival in Oncology Clinical Trials*. Draft guidance, August 2025. https://www.fda.gov/media/188274/download
4. United States Food and Drug Administration. *Multiple Endpoints in Clinical Trials: Guidance for Industry*. October 2022. https://www.fda.gov/media/162416/download
5. Suissa, S. "Immortal Time Bias in Pharmacoepidemiology." *American Journal of Epidemiology* 2008;167(4):492-499. https://doi.org/10.1093/aje/kwm324
6. Grambsch, P. M., Therneau, T. M. "Proportional hazards tests and diagnostics based on weighted residuals." *Biometrika* 1994;81(3):515-526. https://doi.org/10.1093/biomet/81.3.515
7. Fine, J. P., Gray, R. J. "A Proportional Hazards Model for the Subdistribution of a Competing Risk." *Journal of the American Statistical Association* 1999;94(446):496-509. https://doi.org/10.1080/01621459.1999.10474144
8. Kaplan, E. L., Meier, P. "Nonparametric Estimation from Incomplete Observations." *Journal of the American Statistical Association* 1958;53(282):457-481.
9. Cox, D. R. "Regression Models and Life-Tables." *Journal of the Royal Statistical Society, Series B* 1972;34(2):187-220.
10. Schulz, K. F., Altman, D. G., Moher, D., for the CONSORT Group. "CONSORT 2010 Statement." *PLoS Medicine* 2010;7(3):e1000251. https://doi.org/10.1371/journal.pmed.1000251
11. STROBE Initiative. *STROBE Statement*. https://www.strobe-statement.org/
12. Benchimol, E. I., Smeeth, L., Guttmann, A., et al. "The RECORD Statement." *PLoS Medicine* 2015;12(10):e1001885. https://doi.org/10.1371/journal.pmed.1001885
13. Wang, S. V., Schneeweiss, S., and the RCT-DUPLICATE Initiative. "Emulation of Randomized Clinical Trials With Nonrandomized Database Analyses: Results of 32 Clinical Trials." *JAMA* 2023;329(16):1376-1385. https://doi.org/10.1001/jama.2023.4221
14. Hernán, M. A., Robins, J. M. "Using Big Data to Emulate a Target Trial When a Randomized Trial Is Not Available." *American Journal of Epidemiology* 2016;183(8):758-764. https://doi.org/10.1093/aje/kwv254
15. NIST/SEMATECH. *e-Handbook of Statistical Methods*. https://www.itl.nist.gov/div898/handbook/

> **Note:** The FDA OS document is a draft issued in August 2025 with a comment period that closed on 20 October 2025. It is cited to show regulatory direction, not as a final requirement. Kaplan-Meier 1958 and Cox 1972 are cited without DOI because the publisher landing pages are unstable; the journal, volume, and pages are sufficient to locate them.

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`02_regression-models.md`](02_regression-models.md) — the Cox model as a regression model.
- Missing data: [`04_missing-data.md`](04_missing-data.md) — censoring is not the same as missingness.
- Target trial emulation: [`07_target-trial-emulation.md`](07_target-trial-emulation.md) — time zero and immortal time.
- Reporting: [`10_reporting-statistical-results.md`](10_reporting-statistical-results.md) — presenting curves and hazards.
- Survey file: [`../../04_statistics_causal_inference.md`](../../04_statistics_causal_inference.md)
- Labs: [`../../labs/`](../../labs/) — Lab 4.4 covers Kaplan-Meier, Cox, and competing risks.
- Foundations: [`../../01A_foundations_assumptions_and_units.md`](../../01A_foundations_assumptions_and_units.md) — time units and assumptions.
- Estimation: [`01_estimation-and-inference.md`](01_estimation-and-inference.md) — intervals and coverage for survival probabilities.
- Sensitivity analysis: [`08_sensitivity-analysis.md`](08_sensitivity-analysis.md) — informative censoring and unmeasured confounding.
- Study design: [`../../05_study_design_evidence.md`](../../05_study_design_evidence.md) — endpoints and follow-up in trial protocols.
