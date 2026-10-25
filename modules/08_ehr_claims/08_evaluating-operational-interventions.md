# Evaluating Operational Interventions

> **Module 08 — EHR, Claims, and Health System Operations**
> Topic 08 of 09
> Estimated study time: 3.5 hours
> Prerequisites: Topics 04 and 05
> Version 2026.3

## 1. Why this topic exists

Every operational intervention produces a before-and-after story. The readmission rate fell after the navigator programme. The emergency department (ED) boarding time improved after the discharge lounge opened. The no-show rate dropped after the reminder texts. The story is usually true in the narrow sense that the number moved, and usually wrong in the causal sense that the intervention caused the movement. Operational systems are full of secular trends, seasonality, concurrent initiatives, and regression to the mean. A before-and-after comparison attributes all of them to the intervention.

The cost of a false causal claim is a scaled programme that does not work. A hospital that concludes its discharge programme reduced length of stay, when the reduction was a regional trend, will expand the programme, spend the money, and attribute the next year's flat result to poor implementation. The organisation learns the wrong lesson and loses the opportunity to test a different intervention.

Three failure patterns recur.

**The secular trend.** Readmissions, length of stay, and ED visits trend over time for reasons unrelated to the intervention: coding changes, payer policy, the post-acute network, the local economy, and infectious disease seasons. A pre-post comparison cannot separate the trend from the effect. An interrupted time series (ITS) or a difference-in-differences (DiD) design can.

**The concurrent intervention.** Hospitals rarely run one programme at a time. A readmission programme, a new follow-up clinic, and a payer care-management contract may launch in the same quarter. A before-and-after comparison attributes the combined effect to the programme of interest. A comparison group or a design that isolates the timing is needed.

**Regression to the mean.** Programmes are often launched when a metric is unusually bad. The next period is likely to be better even without an intervention, because the extreme value was partly noise. A pre-post comparison will show improvement. A control group or a longer baseline will show whether the improvement is real.

The methodological toolkit is mature. Interrupted time series with segmented regression estimates a level change and a slope change at the intervention, using the pre-intervention trend as the counterfactual. Difference-in-differences compares the change in a treated group with the change in a control group, under the parallel-trends assumption. Stepped-wedge designs randomise the order in which units receive the intervention, so that all units eventually receive it and the comparison is within units over time. Each design has assumptions, and each assumption can be tested.

This topic teaches the three designs as operational tools, with the arithmetic and the diagnostics. The output is an evaluation plan with an estimand, a design, a pre-specified model, a parallel-trends or pre-trend test, a placebo test, and a sensitivity analysis. The plan is written before the intervention launches, not after the data are in.

The same logic applies to quality-improvement programmes. A programme that reports a improvement without a counterfactual is a case study, not an evaluation. The distinction matters because the organisation will use the result to decide whether to continue, expand, or stop, and the decision is only as good as the counterfactual.

> **Note:** An evaluation is not a dashboard. A dashboard shows what happened; an evaluation estimates what would have happened without the intervention. The counterfactual is the product, and it requires a design.

## 2. Learning outcomes

By the end of this topic you can:

1. **State** an estimand (the effect of interest) in terms of a population, an intervention, a comparator, an outcome, and a time horizon.
2. **Choose** between interrupted time series, difference-in-differences, and stepped-wedge designs based on the rollout, the data, and the assumptions.
3. **Specify** a segmented regression with level and slope terms and **interpret** the coefficients.
4. **Test** parallel trends and **run** a placebo test in the pre-period.
5. **Distinguish** a level effect from a slope effect and **explain** why the distinction matters for a programme.
6. **Compute** the counterfactual, the cumulative effect, and the cost per avoided event.
7. **Identify** secular trends, concurrent interventions, and regression to the mean, and **explain** how each biases a naive comparison.
8. **Write** an evaluation plan with a pre-specified primary outcome and a decision rule.

Each outcome maps to an exercise in Section 12 or a self-check item in Section 13. Outcomes 3, 4, and 6 are the quantitative core.

## 3. The core mental model

An evaluation compares what happened with what would have happened. The observed outcome is one potential outcome; the counterfactual is the other. The design is the method for estimating the counterfactual.

```text
OBSERVED OUTCOME
 Y(intervention) for the treated units
    |
    v
COUNTERFACTUAL
 Y(no intervention) for the same units and time
    |
    +---------------------------+---------------------------+
    v              v              v
 BEFORE-AFTER       DIFFERENCE-IN-        STEPPED WEDGE
 (weakest)         DIFFERENCES         (randomised order)
 pre-period as       control group's       units switch at
 counterfactual      change as          randomised times
              counterfactual
```

The three designs differ in how they estimate the counterfactual and in what they assume.

**Before-after** assumes the outcome would not have changed without the intervention. That assumption is almost never true in a dynamic system. The design is useful for generating a hypothesis, not for estimating an effect.

**Interrupted time series** assumes the pre-intervention trend would have continued. It estimates a level change at the intervention and a change in slope after it. The design is strong when there are many pre and post points, no concurrent intervention at the same time, and a stable measurement definition. It is weak when the intervention coincides with a seasonal change or another programme.

**Difference-in-differences** assumes the treated and control groups would have followed parallel trends without the intervention. It estimates the difference between the change in the treated group and the change in the control group. The design is strong when a credible control group exists and the parallel-trends assumption is plausible. It is weak when the groups differ in ways that affect the trend.

**Stepped wedge** randomises the order in which units receive the intervention. Every unit eventually receives it, and the comparison uses the periods before and after each unit's switch. The design is strong when a simultaneous rollout is impossible or unethical, and when there are enough units and periods. It is weak when the number of units is small, when the intervention has spillovers, or when the switch dates are not random.


### The counterfactual in one table

| Design | Counterfactual | Assumption | Falsification test |
|---|---|---|---|
| Before-after | Pre-period mean | No change without intervention | None |
| ITS | Pre-trend extended | Pre-trend continues | Placebo intervention |
| Controlled ITS | Pre-trend plus control trend | Parallel trends | Pre-period difference |
| DiD | Control group change | Parallel trends | Event-study pre-trend |
| Stepped wedge | Units not yet treated | Exchangeability | Baseline balance |
| Regression discontinuity | Units just below threshold | Continuity | Covariate balance |

The table is the core of the topic. The design is a statement about which counterfactual is credible, and the falsification test is the evidence that it is.

> **Note:** The design hierarchy is not absolute. A well-executed ITS with a placebo test can be more credible than a poorly executed DiD with a mismatched control. The credibility comes from the assumptions being tested, not from the label.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Estimand | The effect of interest, defined by population, intervention, comparator, outcome, and horizon | Confused with the estimate | Statement |
| Counterfactual | The outcome that would have occurred without the intervention | Assumed to be the pre-period | Potential outcome |
| Potential outcome | The outcome under a treatment condition | Confused with the observed outcome | Variable |
| Average treatment effect | The mean effect across the population | Confused with the effect on the treated | Effect size |
| Interrupted time series | A design comparing pre and post trends | Used with too few points | Design |
| Segmented regression | A regression with level and slope terms at the intervention | Confused with a simple pre-post t-test | Model |
| Level change | A sudden shift in the outcome at the intervention | Confused with a slope change | Units |
| Slope change | A change in the trend after the intervention | Confused with a level change | Units per period |
| Pre-trend | The trend before the intervention | Assumed to continue | Slope |
| Parallel trends | The assumption that treated and control groups would trend similarly | Untested | Assumption |
| Difference-in-differences | A design comparing change in treated and control groups | Used without a pre-trend check | Design |
| Stepped wedge | A design with randomised staggered rollout | Confused with a before-after | Design |
| Cluster | A unit of randomisation, such as a clinic or hospital | Confused with a patient | Unit |
| Intra-cluster correlation | The similarity of patients within a cluster | Ignored in sample size | ICC |
| Secular trend | A change over time unrelated to the intervention | Attributed to the intervention | Trend |
| Seasonality | Regular variation by time of year | Ignored in short evaluations | Pattern |
| Concurrent intervention | Another change at the same time | Ignored in attribution | Confounder |
| Regression to the mean | The tendency of extreme values to move toward the average | Attributed to the intervention | Statistical phenomenon |
| Placebo test | A test of the design on a period or group with no intervention | Confused with a sensitivity analysis | Test |
| Falsification test | A test of an outcome that should not be affected | Confused with a placebo | Test |
| Lead-lag test | A test of whether the effect appears before the intervention | Ignored | Test |
| Sensitivity analysis | A re-analysis under different assumptions | Confused with a subgroup analysis | Analysis |
| Robust standard error | An error estimate that accounts for clustering or heteroskedasticity | Ignored in cluster designs | Standard error |
| Clustered standard error | A standard error that accounts for within-cluster correlation | Confused with robust | Standard error |
| Confidence interval | The uncertainty around an estimate | Reported without the model | Interval |
| Statistical significance | The probability of the data under the null | Confused with practical significance | p-value |
| Practical significance | Whether the effect is large enough to matter | Ignored in favour of p-values | Effect size |
| Power | The probability of detecting an effect if it exists | Ignored in small evaluations | Probability |
| Minimum detectable effect | The smallest effect the design can detect | Confused with the observed effect | Effect size |
| Attrition | Loss of units or patients from the evaluation | Ignored in the analysis | Percent |
| Spillover | The effect of the intervention on the control group | Ignored in cluster designs | Contamination |
| Contamination | The control group receives part of the intervention | Confused with spillover | Percent |
| Compliance | The degree to which the intervention was delivered as designed | Assumed to be 100 percent | Percent |
| Fidelity | The degree to which the intervention was delivered as intended | Confused with compliance | Assessment |
| Intention-to-treat | Analysis by assigned group, regardless of compliance | Confused with per-protocol | Analysis |
| Per-protocol | Analysis by treatment received | Biased by selection | Analysis |
| As-treated | Analysis by the treatment actually received | Biased by confounding | Analysis |
| Difference-in-differences estimator | The change in treated minus the change in control | Confused with the post-pre difference | Effect |
| Event study | A regression with time dummies around the intervention | Confused with a case study | Model |
| Synthetic control | A weighted combination of controls | Confused with a matched control | Method |
| Interrupted time series with control | An ITS with a comparison series | Confused with DiD | Design |
| Segmented regression coefficient | The estimated level or slope term | Confused with the raw difference | Coefficient |
| Autocorrelation | Correlation of a series with its past | Ignored in ITS standard errors | Statistic |
| Newey-West standard error | A standard error robust to autocorrelation | Confused with clustered | Standard error |
| Seasonality adjustment | Fourier terms or month dummies | Ignored in short series | Adjustment |
| Pre-registration | A plan written before the analysis | Confused with a protocol | Document |
| Primary outcome | The outcome the evaluation is powered for | Chosen after seeing results | Variable |
| Secondary outcome | Other outcomes of interest | Treated as primary | Variable |
| Subgroup analysis | An analysis within a subgroup | Confused with a sensitivity analysis | Analysis |
| Heterogeneity | Variation in effect across groups | Ignored in aggregate results | Effect |
| Decision rule | The rule for continuing or stopping the programme | Written after the result | Rule |

## 5. The mechanics, step by step

The procedure below produces an evaluation plan before the intervention launches. It applies to ITS, DiD, and stepped-wedge designs, with the design chosen at step 4.

### Step 1. State the estimand

**Input:** the intervention and the decision it supports.

**Output:** a sentence defining the population, intervention, comparator, outcome, and horizon.

**Rationale:** "did the programme work" is not an estimand. The estimand is: "for adult medical inpatients discharged to home, what is the effect of the navigator programme, compared with usual care, on the unplanned 30-day readmission rate, over 12 months?" The population, intervention, comparator, outcome, and horizon are all specified. Without them, the analysis can drift to a different question after the data arrive.

The estimand also determines the design. A population-level effect needs a population-level counterfactual; an effect on the treated needs a comparison at the threshold or a randomised assignment.

**Branch point:** if the intervention is expected to have a delayed effect, the horizon must include the delay. A 30-day readmission programme evaluated at 30 days may miss the effect if the navigator contact happens after discharge.

### Step 2. Define the unit and the time granularity

**Input:** the estimand and the data.

**Output:** the unit of analysis (patient, encounter, clinic, hospital) and the time step (day, week, month, quarter).

**Rationale:** the unit determines the standard errors and the power. A clinic-level intervention analysed at the patient level will overstate precision because patients within a clinic are correlated. A monthly series with 12 pre and 12 post points has more power than a quarterly series with 4 and 4.

The time step should match the intervention and the outcome. A daily no-show programme can be evaluated weekly; a readmission programme with a 30-day outcome is usually evaluated monthly. Too fine a step produces noisy zeros; too coarse a step hides the level change.

**Branch point:** if the intervention switches at different times in different units, the time step must be fine enough to distinguish the switch dates. A monthly step with a mid-month switch introduces misclassification.

### Step 3. Plot the series before modelling

**Input:** the outcome by time, overall and by group.

**Output:** a time-series plot with the intervention date, the pre-trend, and any concurrent events marked.

**Rationale:** the plot is the most important diagnostic. It shows whether the pre-trend is stable, whether there is a level change, whether there is a slope change, and whether the intervention coincides with a seasonal shift or another event. A model fitted without looking at the plot can miss the obvious.

Plot at least 12 pre-intervention points and 6 to 12 post points. Mark the intervention date, the data extract date, and any known concurrent events. If the series has a step change before the intervention, the pre-trend is not stable, and the ITS counterfactual is suspect.

**Branch point:** if the pre-series is too short or too noisy, the ITS is not identifiable. Consider a DiD with a control or a stepped wedge instead.

### Step 4. Choose the design

**Input:** the rollout, the data, and the assumptions.

**Output:** a design choice with a justification and a statement of the key assumption.

**Rationale:** the design follows from the rollout and the available comparison. A single-site intervention with a long pre-series supports an ITS. A multi-site rollout with a credible control supports a DiD. A phased rollout with randomisation supports a stepped wedge. A single-site intervention with no control and a short series supports only a descriptive before-after, which should be labelled as such.

The design choice is not a matter of preference. It is a statement about which counterfactual is credible. Write the assumption and the test that would falsify it.

**Branch point:** if no design can support a causal claim, say so. A descriptive analysis is useful; a causal claim without a design is not.

### Step 5. Specify the ITS model

**Input:** the time series and the intervention date.

**Output:** a segmented regression with level and slope terms, a variance specification, and a counterfactual.

**Rationale:** the standard ITS model is:

```text
Y_t = beta0 + beta1 x time_t + beta2 x post_t + beta3 x time_after_t + e_t
```

where `time_t` is the month index, `post_t` is 1 after the intervention, and `time_after_t` is the number of months since the intervention. `beta2` is the level change, `beta3` is the slope change, and the counterfactual is `beta0 + beta1 x time_t` after the intervention.

The model assumes that the pre-trend continues. Test it with a placebo intervention in the pre-period. Account for autocorrelation with Newey-West standard errors or an ARIMA model. Add Fourier terms or month dummies if seasonality is present.

**Branch point:** if the outcome is a count with a small denominator, use a Poisson or negative binomial model rather than a linear model. If the outcome is a proportion, use a logistic or beta regression, or model the logit.

### Step 6. Specify the DiD model

**Input:** the treated and control groups and the pre and post periods.

**Output:** a regression with group, period, and interaction terms, and a parallel-trends test.

**Rationale:** the standard DiD model is:

```text
Y_it = beta0 + beta1 x treated_i + beta2 x post_t + beta3 x treated_i x post_t + e_it
```

`beta3` is the DiD estimate. It equals the change in the treated group minus the change in the control group. The key assumption is parallel trends: without the intervention, the two groups would have changed by the same amount.

Test the assumption with an event-study plot of group differences by period before the intervention. If the pre-period differences are trending, the assumption is not supported. A synthetic control or a matched control can improve credibility.

**Branch point:** if the control group is contaminated by the intervention (spillover), the DiD underestimates the effect. Measure the spillover and report it.

### Step 7. Specify the stepped-wedge model

**Input:** the units, the switch dates, and the outcome.

**Output:** a model with unit and time fixed effects and a treatment indicator, with clustered standard errors.

**Rationale:** the stepped-wedge design randomises the order in which units switch from control to intervention. The model is:

```text
Y_it = alpha_i + gamma_t + beta x treatment_it + e_it
```

where `alpha_i` and `gamma_t` are unit and time fixed effects, and `treatment_it` is 1 after unit i switches. `beta` is the effect. Standard errors must be clustered by unit.

The design requires enough units and periods, randomisation of the switch order, and no spillover between units. The analysis should account for the fact that the intervention effect may vary by time since switch.

**Branch point:** if the number of units is small (fewer than 8 to 10), the stepped wedge has low power and the randomisation may not balance. Consider a DiD with a matched control instead.

### Step 8. Test the assumptions

**Input:** the design and the data.

**Output:** a pre-trend test, a parallel-trends test, and a list of threats.

**Rationale:** each design has a key assumption, and each assumption can be tested. For ITS, test whether the pre-trend is stable and whether a placebo intervention in the pre-period produces a spurious effect. For DiD, test whether the pre-period group differences are constant. For stepped wedge, test whether the switch dates are balanced on baseline characteristics and whether the units are comparable.

A test cannot prove the assumption, but it can falsify it. A failed test is a reason to change the design or to weaken the claim, not a reason to proceed.

**Branch point:** if the pre-trend test fails, do not report the ITS or DiD as a causal estimate. Report the descriptive change and the limitation.

### Step 9. Run placebo and falsification tests

**Input:** the design and the data.

**Output:** placebo estimates on periods or groups with no intervention, and falsification estimates on outcomes that should not be affected.

**Rationale:** a placebo test applies the design to a period before the intervention, using a fake intervention date. If the design produces a large "effect" at the fake date, the model is picking up noise or a trend, and the real estimate is suspect. A falsification test uses an outcome that the intervention should not affect, such as an unrelated admission category. A large effect there suggests confounding.

The tests are cheap and they are the strongest evidence that the design is credible. Report them alongside the main estimate.

**Branch point:** if the placebo estimate is large relative to the main estimate, the main estimate is not credible. Revise the model or the design.

### Step 10. Conduct sensitivity analyses

**Input:** the main model and the assumptions.

**Output:** a table of estimates under alternative specifications.

**Rationale:** the main estimate depends on choices: the time step, the seasonality adjustment, the control group, the outcome definition, the exclusion of outliers, and the standard-error specification. A sensitivity analysis shows whether the conclusion survives those choices. A conclusion that reverses under a plausible alternative is not robust.

Report at least three sensitivities: a different pre-period length, a different control group or seasonality adjustment, and a different outcome definition. If the conclusion is stable, say so; if not, present the range.

**Branch point:** if the sensitivity analysis changes the decision, the decision should be conditional on the specification, not presented as a single number.

### Step 11. Compute the effect size and the cost per avoided event

**Input:** the model estimate, the denominator, and the programme cost.

**Output:** the absolute effect, the relative effect, the events avoided, the cost per avoided event, and the breakeven effect.

**Rationale:** a coefficient is not a decision. Convert the estimate to events avoided: the absolute effect times the population at risk. Then divide the programme cost by the events avoided. Compare the cost per avoided event with the value per event.

For an ITS, the cumulative effect is the sum of the post-period differences between the observed and counterfactual series. For a DiD, the effect is the interaction coefficient times the treated population. For a stepped wedge, the effect is the treatment coefficient times the person-time exposed.

**Branch point:** if the confidence interval for the effect includes zero, the cost per avoided event is undefined or infinite. Report the interval and the decision rule, not a point estimate.

### Step 12. Report and decide

**Input:** the estimate, the diagnostics, and the cost.

**Output:** an evaluation report with the estimand, design, assumptions, results, diagnostics, sensitivity, and decision rule.

**Rationale:** the report should allow a reader to reproduce the estimate and to see how the conclusion depends on the assumptions. It should state the pre-specified primary outcome and whether the result met the decision rule. A report that changes the primary outcome after seeing the results is not an evaluation.

The decision rule should be written before the analysis: continue if the effect is above a threshold and the diagnostics pass; stop or redesign if the effect is below the threshold or the diagnostics fail. The rule protects the organisation from interpreting noise as a signal.

**Branch point:** if the result is inconclusive, the decision is to extend the evaluation or to redesign, not to declare success or failure.


### What the designs cannot do

| Question | Design | Why it cannot answer |
|---|---|---|
| Will it work elsewhere? | Any single-site design | External validity is not identified |
| Why did it work? | ITS or DiD | Mechanism is not estimated |
| What is the long-run effect? | Short post-period | Extrapolation is not evidence |
| Who benefited most? | Aggregate design | Heterogeneity needs subgroup or ML |
| What is the optimal dose? | Binary treatment | Dose-response needs variation |

### Decision table: design choice

| Situation | Design | Key assumption | Test |
|---|---|---|---|
| Single site, long series, no control | ITS | Pre-trend continues | Placebo intervention |
| Single site, long series, control series | Controlled ITS | Parallel trends | Pre-period difference |
| Multi-site, credible control | DiD | Parallel trends | Event-study pre-trend |
| Phased rollout, random order | Stepped wedge | Exchangeable units | Baseline balance |
| Single site, short series, no control | Before-after (descriptive) | None | None |
| Threshold-based eligibility | Regression discontinuity | Continuity at threshold | Covariate balance |

### Decision table: threats and diagnostics

| Threat | Design affected | Diagnostic | Mitigation |
|---|---|---|---|
| Secular trend | Before-after, ITS | Pre-trend plot | Control series or longer baseline |
| Seasonality | ITS, DiD | Month dummies | Fourier terms |
| Concurrent intervention | All | Event timeline | Control group or timing separation |
| Regression to the mean | Before-after | Baseline extremity | Control group |
| Spillover | DiD, stepped wedge | Control contamination | Buffer units or cluster design |
| Attrition | All | Follow-up completeness | Intention-to-treat |
| Definition change | All | Outcome version log | Restate the series |
| Small number of clusters | Stepped wedge, DiD | Cluster count | Randomisation inference |
| Element | Weak report | Strong report |
|---|---|---|
| Counterfactual | Pre-period | Pre-trend or control |
| Diagnostics | None | Placebo and pre-trend |
| Uncertainty | Point estimate | Confidence interval |
| Cost | Not reported | Cost per avoided event |
| Decision rule | After the result | Before the analysis |


### Edge cases that invalidate an evaluation

**The intervention date is unclear.** A programme that launches over three months has no single intervention date. The ITS level change is smeared, and the estimate is biased toward zero. Define the date as the first full month of operation, or model the rollout as a gradual exposure.

**The outcome definition changes.** A readmission definition change in the post-period creates a step change that looks like an effect. Version the outcome and segment the series at the change.

**The control group is contaminated.** A system-wide programme that also affects the control site invalidates the DiD. Measure the exposure in the control and report it.

**The pre-period is too short.** Fewer than 8 to 12 pre points makes the pre-trend unstable and the placebo test weak. Use a longer series or a different design.

**The intervention is targeted at the worst-performing units.** Regression to the mean will produce an improvement even without an effect. Use a control group or a regression discontinuity.

**The outcome is measured with a lag.** Readmissions take 30 days to ascertain, and claims take longer. An evaluation that ends too early will undercount the outcome and bias the effect. Allow for the ascertainment window.

### Interpreting the ITS coefficients

| Coefficient | Meaning | Sign of a good programme | Common misread |
|---|---|---|---|
| beta0 | Baseline level at time zero | Context | Treated as the pre-period mean |
| beta1 | Pre-intervention slope per period | Existing trend | Ignored |
| beta2 | Level change at the intervention | Negative for readmissions | Confused with the total effect |
| beta3 | Change in slope after the intervention | Negative for a sustained decline | Ignored |
| beta2 + beta3 x k | Effect at k periods after the intervention | Depends on k | Reported as a single number |

A programme can have a level effect without a slope effect (a one-time drop), a slope effect without a level effect (a gradual change), or both. The interpretation depends on the mechanism: a reminder programme may produce a level drop, while a care-management programme may produce a gradual slope change.

### A DiD table

| Group | Pre mean | Post mean | Change |
|---|---:|---:|---:|
| Treated | 16.0% | 13.5% | -2.5 pp |
| Control | 15.5% | 14.8% | -0.7 pp |
| Difference-in-differences | | | -1.8 pp |

The naive before-after estimate is -2.5 points. The DiD estimate is -1.8 points, because the control group also improved by 0.7 points. The 0.7-point improvement is the secular trend; attributing it to the programme would overstate the effect by 39 percent.

### Stepped-wedge layout

| Unit | Period 1 | Period 2 | Period 3 | Period 4 |
|---|---|---|---|---|
| Clinic A | Control | Intervention | Intervention | Intervention |
| Clinic B | Control | Control | Intervention | Intervention |
| Clinic C | Control | Control | Control | Intervention |
| Clinic D | Control | Control | Control | Control |

The switch order is randomised. The comparison uses the periods before and after each unit's switch, with unit and period fixed effects. A unit that switches early contributes more intervention periods; a unit that switches late contributes more control periods.

### Power and sample size

| Design | Minimum units | Minimum periods | Key power driver |
|---|---|---|---|
| ITS | 1 | 8-12 pre, 6-12 post | Number of points and effect size |
| DiD | 2 groups | 2 periods | Group size and parallel trends |
| Stepped wedge | 8-10 clusters | 4-6 steps | Number of clusters and ICC |
| Regression discontinuity | 1 threshold | Continuous running variable | Bandwidth and observations near threshold |

A stepped wedge with four clinics and four periods is underpowered. A DiD with two clinics and two periods cannot distinguish the effect from a group-specific shock.

### Standard errors

| Source of correlation | Standard error | Consequence if ignored |
|---|---|---|
| Serial autocorrelation in ITS | Newey-West | Too-narrow intervals |
| Patients within clinics | Clustered by clinic | Too-narrow intervals |
| Repeated measures within patients | Clustered by patient | Too-narrow intervals |
| Few clusters | Randomisation inference | Anti-conservative tests |

The standard error is not a technical detail. An ITS with autocorrelated errors and a naive standard error will report a significant effect that is not. Use Newey-West or an ARIMA model, and report the lag structure.

### Reporting checklist

1. Estimand and population.
2. Intervention and comparator.
3. Design and why it was chosen.
4. Data sources and outcome definition.
5. Pre-period and post-period length.
6. Model specification and standard errors.
7. Pre-trend and placebo tests.
8. Main estimate with confidence interval.
9. Sensitivity analyses.
10. Cost per avoided event and decision rule.

### Converting an estimate to a decision

| Quantity | Formula | Example |
|---|---|---|
| Absolute effect | Post mean - counterfactual mean | -1.8 pp |
| Relative effect | Absolute / baseline | -1.8 / 16.0 = -11.3% |
| Events avoided | Absolute x population at risk | 0.018 x 9,000 = 162 |
| Cost per avoided event | Programme cost / events avoided | $600,000 / 162 = $3,704 |
| Breakeven effect | Cost / (population x value per event) | $600,000 / (9,000 x $16,800) = 0.40 pp |

The breakeven effect is the smallest absolute effect that justifies the programme. If the confidence interval includes the breakeven, the decision is uncertain. If the upper bound is below the breakeven, the programme is not justified; if the lower bound is above it, the programme is justified on the observed effect.

### Designs used in operational work

| Intervention | Feasible design | Why |
|---|---|---|
| Discharge lounge | ITS with a comparison unit | Single-site launch, long series |
| Readmission navigators | Stepped wedge across clinics | Phased rollout |
| No-show reminders | Randomised pilot | Individual assignment is feasible |
| Sepsis alert | ITS with a control hospital | System-wide launch |
| OR scheduling change | DiD with a comparison service | One service changes, another does not |
| Payer contract change | DiD with a non-contracted group | Exposure differs by contract |

The design should follow the rollout, not the analyst's preference. A randomised pilot is strongest when individual or cluster assignment is feasible; an ITS is strongest when the intervention is system-wide and the series is long.

### A note on pre-registration

The evaluation plan should be written before the intervention launches and stored where it cannot be edited silently. The plan fixes the primary outcome, the design, the model, the pre-period, the post-period, and the decision rule. A plan written after the data are in is a rationalisation, not an evaluation.

## 6. Worked example

A health system launches a transitional-care programme for high-risk discharges in January 2026. The outcome is the monthly unplanned 30-day readmission rate for adult medical inpatients, measured for 24 months before the launch and 12 months after. The programme costs $600,000 per year and covers about 9,000 discharges annually.

### Part A: the series

The monthly rates are plotted before modelling. The pre-period runs from January 2024 to December 2025 (24 points), the post-period from January 2026 to December 2026 (12 points). The pre-period mean is 17.58 percent, the post-period mean is 15.28 percent, and the naive before-after difference is -2.30 percentage points.

The naive difference is not the effect. The pre-period has a downward trend of about 0.05 points per month, so the rate would have been about 16.4 percent by December 2026 even without the programme. The before-after comparison attributes the trend to the programme and overstates the effect.

### Part B: segmented regression

The model is:

```text
Y_t = beta0 + beta1 x time_t + beta2 x post_t + beta3 x time_after_t + e_t
```

with `time_t` the month index from 1 to 36, `post_t` 1 from month 25, and `time_after_t` the number of months since the launch. The ordinary least squares estimates are:

| Coefficient | Estimate | Standard error | Interpretation |
|---|---:|---:|---|
| beta0 (intercept) | 18.2015 | 0.0551 | Baseline level |
| beta1 (trend) | -0.0498 | 0.0039 | Pre-trend of -0.05 points per month |
| beta2 (level change) | -1.2372 | 0.0957 | Immediate drop of 1.24 points |
| beta3 (slope change) | -0.0259 | 0.0116 | Additional decline of 0.026 points per month |

The level change is -1.24 points (95 percent CI approximately -1.42 to -1.05), and the slope change is -0.026 points per month (95 percent CI approximately -0.049 to -0.003). Both are statistically significant. The programme produced an immediate drop and a slightly faster decline afterward.

### Part C: the counterfactual and the cumulative effect

The counterfactual at month 36 is the pre-trend extended:

```text
Counterfactual at month 36 = 18.2015 + (-0.0498 x 36) = 16.41
Model-observed at month 36 = 18.2015 - 0.0498 x 36 - 1.2372 - 0.0259 x 12 = 14.24
Effect at month 36 = 14.24 - 16.41 = -2.17 points
```

The average effect over the 12 post months is the mean of `beta2 + beta3 x time_after_t`, which is -1.41 points. The effect grows over time because the slope change accumulates: -1.24 at month 1, -1.55 at month 12, and -2.17 at month 36.

The cumulative effect in readmissions avoided is the sum of the monthly differences between the observed and counterfactual rates applied to the monthly discharge volume. At 9,000 discharges per year, or 750 per month, the average 1.41-point effect corresponds to 750 x 0.0141 = 10.6 avoided readmissions per month, or about 127 per year.

### Part D: cost per avoided readmission

```text
Avoided readmissions = 9,000 x 0.0141 = 126.9
Programme cost = $600,000
Cost per avoided readmission = $600,000 / 126.9 = $4,728
Value per avoided readmission = $16,800
Savings = 126.9 x $16,800 = $2,131,920
Net = $2,131,920 - $600,000 = $1,531,920
Breakeven effect = $600,000 / (9,000 x $16,800) = 0.397 percentage points
```

The programme is worth funding at the estimated effect. The breakeven effect is 0.40 points, far below the estimated 1.41-point average and the 1.24-point immediate level change.

### Part E: placebo test

The same model is fitted to the pre-period only, with a fake intervention at month 13. If the model is picking up noise or a trend, the fake intervention will show a large "effect".

| Placebo coefficient | Estimate | Standard error |
|---|---:|---:|
| Level change at month 13 | 0.013 | 0.108 |
| Slope change after month 13 | 0.003 | 0.016 |

The placebo estimates are essentially zero and are small relative to their standard errors. The model does not manufacture an effect where none exists. The placebo test supports the credibility of the main estimate, although it cannot prove the counterfactual.

### Part F: difference-in-differences cross-check

A comparison hospital did not launch the programme. Its readmission rate fell from 15.5 to 14.8 percent over the same period, a change of -0.7 points. The treated hospital's naive change is -2.3 points, so the DiD estimate is -2.3 - (-0.7) = -1.6 points.

The DiD estimate is larger than the ITS average effect of -1.41 and smaller than the naive -2.3. The difference between the ITS and DiD estimates is within the uncertainty of either design, but the two designs rely on different assumptions. The ITS assumes the pre-trend continues; the DiD assumes the control hospital's trend is the counterfactual. Reporting both is more credible than reporting one.

### Diagnostic 1: the pre-trend is ignored

A naive analysis compares the pre-period mean of 17.58 with the post-period mean of 15.28 and reports a 2.30-point effect. The segmented model attributes 0.89 points of that difference to the pre-existing trend. The naive estimate overstates the effect by 63 percent relative to the model's average effect.

The diagnostic is to plot the pre-trend and to compare the naive difference with the model estimate. If the pre-trend is statistically significant and the naive estimate is larger, the naive estimate is biased upward (for a declining outcome). The same logic applies in reverse for an improving outcome.

### Diagnostic 2: seasonality

Readmission rates have a seasonal pattern, with higher rates in winter. If the post-period contains more winter months than the pre-period, the comparison is confounded. The model can include month dummies or Fourier terms to absorb the seasonality.

A simple check is to compare the same calendar months before and after. If the December rate fell from 18.5 to 16.0 but the July rate was unchanged, the effect may be seasonal rather than programme-related. The diagnostic is to plot the series by month and to test whether the effect is concentrated in the months with the largest seasonal amplitude.

### Varying one assumption

The DiD cross-check suggests that part of the ITS effect may be a regional trend. If the true attributable effect is the DiD estimate of 1.6 points, the cost per avoided readmission is:

```text
Avoided = 9,000 x 0.016 = 144
Cost per avoided = $600,000 / 144 = $4,167
```

The programme is still viable. If the attributable effect is only the difference between the ITS and the control trend, 1.41 - 0.7 = 0.71 points, the avoided readmissions are 63.9 and the cost per avoided is $9,390. The programme is still viable on a variable-cost basis but the margin is much smaller.

| Scenario | Effect (pp) | Avoided | Cost per avoided | Net at $16,800 |
|---|---:|---:|---:|---:|
| Naive before-after | 2.30 | 207 | $2,899 | $2,877,600 |
| ITS average | 1.41 | 127 | $4,728 | $1,531,920 |
| DiD | 1.60 | 144 | $4,167 | $1,819,200 |
| ITS minus control trend | 0.71 | 64 | $9,390 | $473,520 |
| Breakeven | 0.40 | 36 | $16,800 | $0 |

The table is the deliverable. It shows that the conclusion (fund) is robust across the plausible estimates, but the cost per avoided readmission varies by a factor of three. A decision that depends on the cost per avoided event, such as a comparison with an alternative programme, needs the more conservative estimate.

### What the evaluation does not establish

The evaluation does not establish that the programme would work in another hospital, that the effect persists beyond 12 months, or that the mechanism is the navigator contact rather than the attention of the care team. It establishes that, in this hospital and period, the readmission rate fell more than the pre-trend and the control hospital would predict. That is a local causal claim, and it is the right claim for a local decision.

### Part G: model fit and residual diagnostics

The model explains 98.8 percent of the variance in the monthly series (R-squared 0.988), which is expected for a smooth rate series with a strong trend. A high R-squared is not evidence that the model is correct; it is evidence that the series is smooth.

| Diagnostic | Result | Interpretation |
|---|---|---|
| R-squared | 0.988 | Series is smooth; trend dominates |
| Residual standard deviation | 0.131 pp | Small relative to the effect |
| Durbin-Watson | 1.9 | No strong first-order autocorrelation |
| Placebo level | 0.013 pp | No spurious level effect |
| Placebo slope | 0.003 pp/month | No spurious slope effect |

The Durbin-Watson statistic near 2 suggests that first-order autocorrelation is not a major problem in this series. If it were below 1.5, the standard errors would be too narrow, and a Newey-West correction would be needed.

### Part H: effect by month

| Months after launch | Estimated effect (pp) | Cumulative readmissions avoided |
|---:|---:|---:|
| 1 | -1.26 | 9.5 |
| 3 | -1.31 | 29.5 |
| 6 | -1.39 | 62.6 |
| 9 | -1.47 | 99.2 |
| 12 | -1.55 | 139.5 |
| 18 | -1.70 | 229.5 |
| 24 | -1.86 | 334.8 |

The cumulative column multiplies the monthly effect by 750 discharges per month. The effect grows because the slope change accumulates. A programme that is evaluated at month 3 will see only a small part of the benefit; a programme evaluated at month 12 will see most of it.

### Part I: sensitivity to the pre-period

The pre-period includes 2024 and 2025. If the first six months of 2024 are excluded because a coding change affected the series, the pre-trend is estimated on 18 points instead of 24. The level change estimate moves from -1.24 to about -1.30, and the slope change from -0.026 to -0.021. The conclusion is unchanged, but the estimate is less precise.

A second sensitivity excludes the winter months from both periods. The level change is similar, and the slope change is smaller, which suggests that part of the post-period decline is seasonal. The programme's effect is still negative in the non-winter months, but the magnitude is smaller.

The sensitivity table should be reported with the main estimate, not in an appendix. A decision that depends on the exact specification should be presented as a range.

### Part J: the decision

The programme should be continued. The ITS estimate is -1.41 points on average, the DiD cross-check is -1.60, and the placebo test is null. The cost per avoided readmission ranges from $4,167 to $9,390 depending on the estimate, and the programme is net positive under all but the most conservative interpretation at a $16,800 value per readmission.

The decision rule for the next year is: continue if the 12-month average effect remains below -0.8 points and the placebo test remains null; redesign if the effect is between -0.8 and 0; stop if the effect is positive or the control hospital shows a larger decline.

The rule is written before the next year's data arrive, so the decision cannot be redefined after the result.

That is the difference between an evaluation and a narrative.

A pre-specified rule is the strongest protection against hindsight.

It also makes the evaluation auditable by someone who was not in the room.

## 7. Code, query, or calculation

The Python script below fits the segmented regression from Section 6 using only the standard library. It solves the normal equations with Gaussian elimination, reports the coefficients and standard errors, computes the counterfactual, and runs the placebo test.

Assumptions:

- The series is monthly, with 24 pre-intervention and 12 post-intervention points.
- The intervention starts at month 25.
- The model is linear in the level and slope terms.
- The standard errors are ordinary least squares; a production analysis should use Newey-West if autocorrelation is present.

```python
import math

def solve(A, b):
  n = len(A)
  M = [row[:] + [b[i]] for i, row in enumerate(A)]
  for i in range(n):
    p = max(range(i, n), key=lambda r: abs(M[r][i]))
    M[i], M[p] = M[p], M[i]
    d = M[i][i]
    for j in range(i, n + 1):
      M[i][j] /= d
    for r in range(n):
      if r != i and M[r][i] != 0:
        f = M[r][i]
        for j in range(i, n + 1):
          M[r][j] -= f * M[i][j]
  return [M[i][n] for i in range(n)]

def ols(X, y):
  k = len(X[0])
  XtX = [[sum(X[i][a] * X[i][b] for i in range(len(X)))
      for b in range(k)] for a in range(k)]
  Xty = [sum(X[i][a] * y[i] for i in range(len(X))) for a in range(k)]
  beta = solve(XtX, Xty)
  yhat = [sum(X[i][j] * beta[j] for j in range(k)) for i in range(len(X))]
  resid = [y[i] - yhat[i] for i in range(len(X))]
  rss = sum(r * r for r in resid)
  sigma2 = rss / (len(X) - k)
  inv = [[0] * k for _ in range(k)]
  for c in range(k):
    e = [1 if i == c else 0 for i in range(k)]
    sol = solve([row[:] for row in XtX], e)
    for r in range(k):
      inv[r][c] = sol[r]
  se = [math.sqrt(sigma2 * inv[j][j]) for j in range(k)]
  return beta, se, sigma2

months, intervention = 36, 25
rows = []
for t in range(1, months + 1):
  post = 1 if t >= intervention else 0
  ta = max(0, t - intervention + 1)
  base = 18.2 - 0.05 * t
  effect = -1.2 * post - 0.03 * ta
  noise = 0.15 * math.sin(t * 1.7) + 0.08 * math.cos(t * 0.9)
  rows.append((t, post, ta, base + effect + noise))

X = [[1, r[0], r[1], r[2]] for r in rows]
y = [r[3] for r in rows]
beta, se, sigma2 = ols(X, y)
for name, b, s in zip(["intercept", "trend", "level_change", "slope_change"],
           beta, se):
  print(f"{name}: {b:.4f} (se {s:.4f})")
print("sigma2:", round(sigma2, 4))

t = 36
cf = beta[0] + beta[1] * t
obs = beta[0] + beta[1] * t + beta[2] + beta[3] * t
print("counterfactual month 36:", round(cf, 2))
print("model observed month 36:", round(obs, 2))
print("effect:", round(obs - cf, 2))

effs = [beta[2] + beta[3] * r[2] for r in rows if r[1] == 1]
print("mean post effect:", round(sum(effs) / len(effs), 3))

pre = [r for r in rows if r[0] < intervention]
Xp, yp = [], []
for t, _, _, v in pre:
  post = 1 if t >= 13 else 0
  ta = max(0, t - 13 + 1)
  Xp.append([1, t, post, ta])
  yp.append(v)
bp, sep, _ = ols(Xp, yp)
print("placebo level:", round(bp[2], 3), "(se", round(sep[2], 3), ")")
print("placebo slope:", round(bp[3], 3), "(se", round(sep[3], 3), ")")

admits, effect_pp, cost = 9000, 1.41, 600000
avoided = admits * effect_pp / 100
print("avoided:", round(avoided, 1))
print("cost per avoided:", round(cost / avoided, 0))
print("breakeven pp:", round(cost / (admits * 16800) * 100, 3))
```

Expected output:

```text
intercept: 18.2015 (se 0.0551)
trend: -0.0498 (se 0.0039)
level_change: -1.2372 (se 0.0957)
slope_change: -0.0259 (se 0.0116)
sigma2: 0.0171
counterfactual month 36: 16.41
model observed month 36: 14.24
effect: -2.17
mean post effect: -1.406
placebo level: 0.013 (se 0.108 )
placebo slope: 0.003 (se 0.016 )
avoided: 126.9
cost per avoided: 4728.0
breakeven pp: 0.397
```

Read the output in three passes. The coefficient block gives the pre-trend (-0.0498 per month), the level change (-1.2372), and the slope change (-0.0259), with standard errors that are small relative to the estimates. The counterfactual block shows the rate would have been 16.41 at month 36 without the programme and was 14.24 with it, a 2.17-point difference. The placebo block shows a level estimate of 0.013 with a standard error of 0.108, which is consistent with no effect.

> **Note:** The placebo test uses a fake intervention at month 13 within the pre-period. A large placebo estimate would indicate that the model is picking up noise or a non-linear trend. The test is not proof of the counterfactual, but it is a cheap and informative check.

A production analysis should also check the residuals for autocorrelation. A Durbin-Watson statistic below 1.5 suggests positive autocorrelation, which makes the ordinary least squares standard errors too narrow. The correction is Newey-West or an ARIMA model, and the lag structure should be reported.

### Adding seasonality

The model above omits seasonality. A production model would add month dummies or Fourier terms:

```python
# Add 11 month dummies (December as reference)
X = []
for t, post, ta, _ in rows:
  month = (t - 1) % 12 + 1
  dummies = [1 if month == m else 0 for m in range(1, 12)]
  X.append([1, t, post, ta] + dummies)
```

The coefficient of interest remains the level and slope terms. The month dummies absorb the seasonal pattern, and the standard errors should be clustered or Newey-West if the residuals remain autocorrelated.

### Difference-in-differences in Python

A DiD can be estimated with the same `ols` function using a design matrix with group, period, and interaction terms:

```python
X = [[1, treated, post, treated * post] for treated, post in pairs]
```

The coefficient on the interaction is the DiD estimate. The parallel-trends test is a regression of the pre-period outcome on group, time, and group times time; a significant interaction before the intervention indicates that the trends are not parallel.

### PostgreSQL dialect: monthly series with a treatment indicator

The following block is **PostgreSQL** syntax. It assembles the monthly series for a stepped-wedge or DiD analysis.

```sql
-- PostgreSQL 15+
SELECT date_trunc('month', discharge_date)::date AS month,
    unit_id,
    COUNT(*) AS discharges,
    SUM(CASE WHEN readmit_30d = 1 THEN 1 ELSE 0 END) AS readmissions,
    AVG(CASE WHEN readmit_30d = 1 THEN 1.0 ELSE 0 END) AS rate,
    MAX(CASE WHEN intervention_start <= discharge_date THEN 1 ELSE 0 END) AS treated
FROM discharges
WHERE discharge_date >= DATE '2024-01-01'
GROUP BY 1, 2
ORDER BY 1, 2;
```

The `treated` flag is time-varying, which is required for a stepped wedge. A unit that switches in July has `treated = 0` before July and `treated = 1` after. The model uses unit and month fixed effects with this flag.

## 8. Data, standards, and regulatory context

Evaluation methods are governed less by regulation than by reporting standards and by the expectations of funders, journals, and quality committees. The standards do not prescribe a design; they prescribe what must be reported so that a reader can judge the design.

### Reporting standards

| Standard | What it covers | Binding status | What it does not cover |
|---|---|---|---|
| CONSORT | Randomised trials | Journal and funder requirement | Observational designs |
| CONSORT extension for stepped wedge | Stepped-wedge trials | Journal requirement | ITS and DiD |
| TREND | Non-randomised evaluations of interventions | Guideline | Randomised trials |
| STROBE | Observational studies | Guideline | Intervention effects |
| SQUIRE | Quality improvement reporting | Guideline | Causal inference |
| CMS evaluation guidance | Programme evaluations | Contract or programme requirement | Local operational studies |
| AHRQ methods | Comparative effectiveness and quality | Guidance | Specific designs |

The standards share a common requirement: state the design, the population, the intervention, the comparator, the outcome, the analysis, and the limitations. A report that omits the comparator or the pre-period cannot be assessed. A report that presents a before-after comparison as a causal effect violates the spirit of every standard.

### Design-specific reporting

For an ITS, report the number of pre and post points, the model specification, the level and slope terms, the standard-error method, the seasonality adjustment, and the placebo test. For a DiD, report the treated and control groups, the pre-period trend test, the model specification, and the clustering. For a stepped wedge, report the randomisation, the number of clusters and periods, the intra-cluster correlation, and the spillover assessment.

### Regulatory and programme context

CMS and other payers increasingly require evaluations of care-management and quality programmes. The evaluation requirements vary by programme; some require a comparison group, others accept a pre-post design. The programme's evaluation guidance governs, and the analyst should read it before designing the study.

The Common Rule and IRB review apply when the evaluation is research. Quality improvement that is not generalisable may be exempt, but the boundary is determined by the IRB. An evaluation that is intended for publication is usually research.

> **Note:** Reporting standards are not a substitute for a credible design. A study can comply with TREND and still have no counterfactual. The standard makes the weakness visible; it does not remove it.


### Binding versus convention

| Element | Binding? | Source | Local choice |
|---|---|---|---|
| Trial reporting | Journal requirement | CONSORT | Design details |
| Non-randomised reporting | Guideline | TREND | Design details |
| QI reporting | Guideline | SQUIRE | Design details |
| Programme evaluation | Contract-specific | Funder | Design and outcomes |
| IRB review | Federal regulation | Common Rule | Exempt or reviewed |
| Pre-registration | Journal or funder | Policy | Registry |

### What the standards do not settle

1. **The counterfactual.** No standard says which comparison is credible.
2. **The pre-period length.** No standard sets a minimum, although 8 to 12 points is common.
3. **The seasonality adjustment.** No standard requires it, although it is often necessary.
4. **The decision rule.** No standard says when to continue or stop a programme.
5. **The cost per avoided event.** No standard requires it, although funders increasingly ask.

## 9. Industry practice and case evidence

### Documented methodological guidance

Lopez Bernal, Cummins, and Gasparrini published a tutorial on interrupted time series regression in the *International Journal of Epidemiology* in 2017. It describes the segmented regression specification, the level and slope terms, the threats from seasonality and time-varying confounders, and the use of control series to strengthen the design. It is the standard practical reference for ITS in health.

Li et al. published a primer comparing difference-in-differences, segmented regression, and interventional ARIMA for health interventions. It describes the data requirements, the model specifications, and the sample-size rules of thumb, including at least 12 pre and post points for ITS and 50 to 100 pre points for ARIMA.

Feng et al. published a review of modern DiD methods for health policy in *Statistics in Medicine* in 2025. It covers the parallel-trends assumption, doubly robust and synthetic DiD, staggered treatment timing, and inference with few clusters. The review is a useful update for anyone using DiD with multiple rollout dates.

### Documented operational evaluations

The discharge-before-noon literature is the clearest example of the value of design. Artenstein et al. reported a 0.30-day reduction in length of stay and a 2.1-hour reduction in boarding after a sustained programme. Kausar et al. found that the programme increased the share of morning discharges but was not independently associated with length of stay or readmissions, and median boarding increased by 41 minutes. Hodges et al. reported longer order-to-discharge times and increased boarding after a "discharge by 11 a.m." programme. The differences are consistent with differences in implementation and in the counterfactual: a pre-post study in one hospital cannot distinguish the programme from the concurrent changes in transport, pharmacy, and post-acute placement.

The Kaiser Permanente Northern California Transitions Program is the best-documented analytics-targeted readmission evaluation. It used a difference-in-differences design across 21 hospitals and 1.54 million discharges, with an adjusted odds ratio of 0.91 and an absolute risk reduction of about 2.5 points. The study also used regression discontinuity as a cross-check, which is a stronger design than a simple pre-post. The result is credible because the design is credible, not because the sample is large.


### Evidence table

| Study or source | Design | What it shows | Limit |
|---|---|---|---|
| Lopez Bernal et al., 2017 | Tutorial | How to specify and diagnose an ITS | Not an effect estimate |
| Feng et al., 2025 | Review | Modern DiD methods and pitfalls | Methodological |
| Artenstein et al., 2017 | Pre-post | Discharge programme improved LOS and boarding | Single site |
| Kausar et al., 2023 | Pre-post | Morning discharge share rose; boarding rose | Single site |
| Kaiser Transitions, 2021 | DiD and RD | Adjusted OR 0.91; ARR about 2.5 points | Integrated system |

The table separates methodological guidance from empirical results. The methods tell you how to estimate; the studies tell you what happened elsewhere, not what will happen locally.


A common reporting failure is to present the coefficient without the counterfactual. A level change of -1.2 points means little unless the reader can see what the rate would have been without the programme. The counterfactual plot is the most useful single figure in an ITS report.

### Inference

> **Inference:** A local programme should not assume the Kaiser or Geisinger effect size. The designs in those studies are not reproducible in a single hospital without a comparison group. The local evaluation should use the strongest feasible design and report the assumptions, not import the published effect.

### What mature organisations do

They write the evaluation plan before the launch. The plan names the estimand, the design, the primary outcome, the pre-period, the post-period, and the decision rule. It is reviewed by the quality committee and stored with the programme charter.

They run a placebo test on every ITS. The test is cheap and it catches the most common failure: a model that produces an effect at a date where no intervention occurred.

They report the result with the diagnostics, not only the estimate. A committee that sees the estimate, the pre-trend, the placebo, and the sensitivity can make a decision. A committee that sees only the estimate is being asked to trust the analyst.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Before-after presented as causal | No control or pre-trend | Secular trend attributed to programme | Effect overstated | Use ITS or DiD |
| Pre-trend ignored | The pre-period is sloping | Counterfactual is wrong | Level effect biased | Include the trend term |
| Too few pre points | Fewer than 8 | Unstable slope | Placebo test is weak | Extend the series |
| Intervention date smeared | Rollout over months | Level change is diluted | Effect underestimated | Model exposure or use first full month |
| Concurrent intervention | Another launch at the same time | Attribution is split | Effect is confounded | Timeline and control |
| Seasonality ignored | Winter and summer differ | Seasonal change looks like effect | Effect biased | Month dummies or Fourier |
| Definition change | Outcome code set updated | Step change in the series | False effect | Version the outcome |
| Control contaminated | Control also exposed | DiD understates effect | Effect biased toward zero | Measure exposure |
| Regression to the mean | Programme targets worst units | Extreme values revert | False improvement | Control or RD |
| Small number of clusters | Fewer than 8 units | Unstable standard errors | False significance | Randomisation inference |
| Autocorrelation ignored | Residuals correlated | Standard errors too narrow | False significance | Newey-West |
| Attrition | Units drop out | Selection | Effect biased | Intention-to-treat |
| Outcome lag | 30-day outcome measured at 30 days | Incomplete ascertainment | Understated effect | Extend follow-up |
| Subgroup fishing | Many subgroup analyses | Multiplicity | False finding | Pre-specify |
| Decision rule changed after results | Rule not in the plan | Hindsight | Loss of credibility | Pre-register |

Three failures deserve emphasis.

**The ignored pre-trend.** A series that is already declining will continue to decline. A before-after comparison attributes the continuation to the programme. The segmented model includes the trend, and the difference between the naive and model estimates is the size of the bias. In the worked example, the naive estimate was 2.30 points and the model average was 1.41, a 63 percent overstatement.

**The concurrent intervention.** A hospital that launches a readmission programme and a follow-up clinic in the same quarter cannot attribute the effect to either. The evaluation needs a design that separates the timing, a control group, or an explicit statement that the effect is the combined effect of the bundle. The last option is honest but less useful for deciding which component to keep.

**The contaminated control.** A system-wide programme that also changes practice at the control site invalidates the DiD. The control's improvement is attributed to the secular trend, and the DiD underestimates the effect. The mitigation is to measure exposure in the control and to choose a control outside the system.


### Pre-mortem for an evaluation

1. What is the estimand?
2. Which counterfactual, and why is it credible?
3. How many pre and post points?
4. What could change at the same time?
5. What does the placebo or pre-trend test show?
6. What is the decision rule?

An evaluation that cannot answer all six is a narrative, not a design.
> **Note:** A statistically significant effect is not a programme success. The effect must be large enough to justify the cost, and the design must be credible enough to support the decision. A significant coefficient from a weak design is a false confidence.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test |
|---|---|---|---|
| The pre-trend would continue | ITS counterfactual | Structural change, seasonality | Placebo test, pre-trend plot |
| The control is untreated | DiD counterfactual | Spillover, shared programme | Exposure measurement |
| Parallel trends | DiD identification | Different trends | Event-study pre-trend |
| Units are exchangeable | Stepped wedge | Selection, different case mix | Baseline balance |
| The intervention date is sharp | ITS level change | Phased rollout | Exposure model |
| The outcome definition is stable | All designs | Code or definition change | Version log |
| The series is long enough | ITS power | Short pre-period | Minimum point rule |
| Standard errors account for correlation | Inference | Autocorrelation, clustering | Newey-West, cluster |
| No concurrent intervention | Attribution | Another launch | Timeline |
| The effect is homogeneous enough to summarise | Reporting | Subgroup differences | Subgroup analysis |


### Assumption owners

| Assumption | Owner | Review cadence |
|---|---|---|
| Estimand | Programme owner | At protocol |
| Design | Analytics | At protocol |
| Pre-trend | Analytics | At analysis |
| Parallel trends | Analytics | At analysis |
| Outcome definition | Quality or analytics | Each release |
| Decision rule | Steering committee | At protocol |

The limits follow from the assumptions. An evaluation estimates a local effect under a specific counterfactual. It cannot tell you whether the programme would work elsewhere, whether the effect persists, or whether a different implementation would produce a larger effect. It can tell you whether the observed change is larger than the change the counterfactual predicts, with a stated uncertainty.

A second limit is that the counterfactual is unobservable. Every design is an approximation, and the credibility comes from the assumptions being plausible and tested, not from the label. A randomised trial is the strongest design because the counterfactual is generated by randomisation; an ITS is credible when the pre-trend is stable and the placebo is null.

A third limit is that the evaluation is expensive. A stepped wedge across 20 clinics takes years; an ITS with 36 monthly points takes three years. Organisations often choose a weaker design because it is faster. The choice should be explicit, and the claim should be proportional to the design.

## 12. Exercises

### Exercise 1. Write the estimand (recall)

A hospital plans a discharge lounge. Write the estimand in one sentence with the population, intervention, comparator, outcome, and horizon. State which design is feasible and why.

**Deliverable:** an estimand sentence and a design justification.

**Acceptance criteria:** all five elements are present; the design matches the rollout; the key assumption is stated.

**Time:** 30 minutes.

### Exercise 2. Interpret ITS coefficients (application)

A segmented regression gives an intercept of 18.0, a pre-trend of -0.05 per month, a level change of -1.50, and a slope change of -0.04. Compute the counterfactual and the model-observed value at 12 months after the intervention, and the effect at that point. State whether the effect is growing or shrinking.

**Deliverable:** a calculation with the counterfactual, observed, and effect.

**Acceptance criteria:** the counterfactual uses the pre-trend only; the observed includes both post terms; the effect is interpreted as a level-plus-slope effect.

**Time:** 45 minutes.

### Exercise 3. Compute a DiD (application)

A treated hospital's readmission rate falls from 18.0 to 14.0 percent. A control hospital's rate falls from 17.0 to 15.5 percent. Compute the change in each group, the DiD estimate, and the share of the treated change that is attributable to the secular trend.

**Deliverable:** a calculation table.

**Acceptance criteria:** the DiD is the difference in changes; the secular share is the control change divided by the treated change; the interpretation is stated.

**Time:** 30 minutes.

### Exercise 4. Critique a placebo test (critique)

An ITS reports a level change of -1.2 points at the intervention and a placebo level change of -0.9 points at a fake date in the pre-period. The analyst concludes that the programme worked because the main estimate is larger. Critique the conclusion and state what the placebo result implies.

**Deliverable:** a short critique with a corrected interpretation.

**Acceptance criteria:** the placebo is compared with its standard error; the possibility of a trend or noise is discussed; the recommendation is to revise the model or the claim.

**Time:** 45 minutes.

### Exercise 5. Choose a design (design)

A system plans to roll out a sepsis alert to 12 hospitals over 12 months. The rollout order is not random; the largest hospitals go first. Compare the feasibility of ITS, DiD, and stepped wedge, and recommend a design with its assumptions and limitations.

**Deliverable:** a one-page design memo.

**Acceptance criteria:** at least two designs are compared; the non-random rollout is addressed; the recommendation states the key assumption and the fallback.

**Time:** 45 minutes.

### Exercise 6. Implement the ITS in Python or SQL (application)

Using the data from Section 7, add a seasonality term (a 12-month sine and cosine) and refit the model. Report the level and slope coefficients and compare them with the original. Run the code and paste the output.

**Deliverable:** the code, the output, and a comparison table.

**Acceptance criteria:** the code executes; the seasonality terms are included; the comparison states whether the conclusion changes.

**Time:** 90 minutes.

### Exercise 7. Compute the cost per avoided event (application)

An intervention reduces readmissions by 1.2 percentage points. There are 12,000 discharges per year. The programme costs $700,000, and the value per avoided readmission is $15,000. Compute avoided readmissions, cost per avoided readmission, net benefit, and breakeven effect.

**Deliverable:** a calculation table.

**Acceptance criteria:** avoided events use the absolute effect and the denominator; the breakeven uses the value per event; the net is computed correctly.

**Time:** 45 minutes.

### Exercise 8. Write an evaluation plan (design)

Design an evaluation for a no-show reminder programme in 20 clinics. Specify the estimand, the design, the unit, the time step, the primary outcome, the pre-period, the comparison, the diagnostics, and the decision rule.

**Deliverable:** a two-page evaluation plan.

**Acceptance criteria:** the design matches the rollout; the primary outcome is pre-specified; the diagnostics include a placebo or parallel-trends test; the decision rule is explicit.

**Time:** 60 minutes.


### Assessment notes

Exercises 1 to 4 can be completed individually and checked against Section 13. Exercise 6 requires Python or SQL. Exercises 5, 7, and 8 are design tasks assessed with the rubric. Add 20 minutes to Exercise 6 for learners who need to install Python.

## 13. Self-check answers

### Exercise 1

A strong estimand: "For adult medical inpatients discharged to home at Hospital A between January and December 2026, what is the effect of a discharge lounge, compared with usual discharge, on the mean time from discharge order to bed-ready, over the 12 months after launch?" The design is an ITS with a comparison unit, because the lounge launches at one site and a long pre-series is available. The key assumption is that the pre-trend would have continued without the lounge.

### Exercise 2

```text
Counterfactual at 12 months after = 18.0 - 0.05 x 36 = 16.20
Observed at 12 months after = 16.20 - 1.50 - 0.04 x 12 = 14.22
Effect = 14.22 - 16.20 = -1.98 points
```

The effect is growing because the slope change accumulates: -1.50 at month 1, -1.98 at month 12. A programme with only a level change would have a constant effect.

### Exercise 3

```text
Treated change = 14.0 - 18.0 = -4.0 points
Control change = 15.5 - 17.0 = -1.5 points
DiD = -4.0 - (-1.5) = -2.5 points
Secular share = -1.5 / -4.0 = 37.5%
```

The naive before-after estimate is -4.0 points. The DiD estimate is -2.5 points. About 37.5 percent of the treated change is consistent with the secular trend observed in the control, and attributing the full -4.0 to the programme would overstate the effect by 60 percent.

### Exercise 4

The placebo estimate of -0.9 points is large relative to the main estimate of -1.2. If the placebo standard error is small, the pre-period has a break at the fake date, which means the model is picking up a trend or a structural change rather than the intervention. The main estimate is not credible. The analyst should revise the model (longer pre-period, non-linear trend, seasonality) or weaken the claim to a descriptive before-after. A placebo that is 75 percent of the main estimate is a red flag, not a confirmation.

### Exercise 5

An ITS is feasible for each hospital individually but not for the system, because the rollout is staggered and the intervention date differs by site. A DiD with early and late adopters is feasible, but the non-random order (largest hospitals first) means the groups differ in size and case mix, and parallel trends are doubtful. A stepped wedge is the natural design if the rollout order can be randomised; if it cannot, the analysis should adjust for hospital size and case mix and should test whether the early and late groups had parallel pre-trends. The fallback is a controlled ITS at the hospital level with a system-level comparison.

### Exercise 6

Adding sine and cosine terms for seasonality changes the level and slope estimates modestly. If the original level change was -1.24 and the seasonal model gives -1.18, the conclusion is unchanged. If the seasonal model gives -0.60, the original estimate was confounded by seasonality, and the programme's effect is smaller. The comparison table should report both estimates and the seasonal coefficients.

### Exercise 7

```text
Avoided readmissions = 12,000 x 0.012 = 144
Cost per avoided = $700,000 / 144 = $4,861
Savings = 144 x $15,000 = $2,160,000
Net = $2,160,000 - $700,000 = $1,460,000
Breakeven effect = $700,000 / (12,000 x $15,000) = 0.389 percentage points
```

The programme is viable at a 1.2-point effect and breaks even at 0.39 points.

### Exercise 8 rubric

A strong plan specifies the estimand, the cluster (clinic), the time step (month), the primary outcome (show rate), the pre-period (at least 12 months), the comparison (stepped wedge or matched clinics), the diagnostics (parallel trends or placebo), and the decision rule. It also states the power calculation and the spillover risk.

A weak plan uses a before-after comparison with no comparison group and no pre-specified rule.


A note on precision: the worked example rounds coefficients to four decimals and effects to two. A production analysis should report the confidence interval and should not round before computing the cost per avoided event. At 9,000 discharges, a 0.01-point rounding in the effect changes the avoided count by 0.9.

## 14. Interview questions

**1. Why is a before-after comparison weak?**
A strong answer names secular trends, concurrent interventions, and regression to the mean, and explains that the counterfactual is assumed rather than estimated. A weak answer says it is "not randomised".

**2. What is an interrupted time series?**
A strong answer describes the pre-trend, the level and slope terms, the counterfactual, and the placebo test. A weak answer says it is "a line chart with a before and after".

**3. What is the parallel-trends assumption?**
A strong answer explains that the treated and control groups would have followed the same trend without the intervention, and that it is tested with pre-period event studies. A weak answer says the groups are similar.

**4. When would you use a stepped wedge?**
A strong answer says when a simultaneous rollout is impossible or unethical, when there are enough clusters, and when the switch order can be randomised. A weak answer says when you want everyone to get the intervention.

**5. What is a placebo test?**
A strong answer applies the design to a period or group with no intervention and checks whether it produces a spurious effect. A weak answer confuses it with a sensitivity analysis.

**6. How do you distinguish a level effect from a slope effect?**
A strong answer explains that a level effect is an immediate shift and a slope effect is a change in trend, and that the mechanism determines which to expect. A weak answer reports one number.

**7. How do you handle seasonality in an ITS?**
A strong answer uses month dummies or Fourier terms, checks the residual pattern, and compares the same months before and after. A weak answer ignores it.

**8. What is regression to the mean?**
A strong answer explains that extreme values tend to move toward the average, and that programmes targeted at the worst units will show improvement without an effect. A weak answer confuses it with a trend.

**9. How do you compute the cost per avoided event?**
A strong answer multiplies the absolute effect by the population at risk, divides the programme cost by the avoided events, and compares with the value per event. A weak answer divides cost by the number of patients.

**10. What is a decision rule?**
A strong answer is a pre-specified rule for continuing, redesigning, or stopping based on the effect and the diagnostics. A weak answer is a target set after the results.

**11. How do you handle a concurrent intervention?**
A strong answer uses a design that separates the timing, a control group, or an explicit statement that the estimate is a bundle effect. A weak answer ignores it.

**12. What would make you reject your own evaluation?**
A strong answer names a failed placebo, a pre-trend break, a contaminated control, or a definition change, and says it would weaken the claim. A weak answer says nothing would.


**13. What is a falsification test?**
A strong answer applies the design to an outcome the intervention should not affect, and treats a large effect as evidence of confounding. A weak answer confuses it with a placebo.

**14. How do you choose the pre-period length?**
A strong answer balances enough points for a stable trend against changes in the system, and tests sensitivity to dropping early points. A weak answer uses whatever data are available.

## 15. Key terms

- **Estimand:** the effect of interest, defined by population, intervention, comparator, outcome, and horizon.
- **Counterfactual:** the outcome that would have occurred without the intervention.
- **Potential outcome:** the outcome under a treatment condition.
- **Average treatment effect:** the mean effect across the population.
- **Interrupted time series:** a design comparing pre and post trends.
- **Segmented regression:** a regression with level and slope terms at the intervention.
- **Level change:** a sudden shift in the outcome at the intervention.
- **Slope change:** a change in the trend after the intervention.
- **Pre-trend:** the trend before the intervention.
- **Parallel trends:** the assumption that treated and control groups would trend similarly.
- **Difference-in-differences:** a design comparing change in treated and control groups.
- **Stepped wedge:** a design with randomised staggered rollout.
- **Cluster:** a unit of randomisation.
- **Intra-cluster correlation:** the similarity of patients within a cluster.
- **Secular trend:** a change over time unrelated to the intervention.
- **Seasonality:** regular variation by time of year.
- **Concurrent intervention:** another change at the same time.
- **Regression to the mean:** the tendency of extreme values to move toward the average.
- **Placebo test:** a test of the design on a period or group with no intervention.
- **Falsification test:** a test of an outcome that should not be affected.
- **Lead-lag test:** a test of whether the effect appears before the intervention.
- **Sensitivity analysis:** a re-analysis under different assumptions.
- **Robust standard error:** an error estimate that accounts for heteroskedasticity.
- **Clustered standard error:** an error estimate that accounts for within-cluster correlation.
- **Confidence interval:** the uncertainty around an estimate.
- **Statistical significance:** the probability of the data under the null.
- **Practical significance:** whether the effect is large enough to matter.
- **Power:** the probability of detecting an effect if it exists.
- **Minimum detectable effect:** the smallest effect the design can detect.
- **Attrition:** loss of units or patients from the evaluation.
- **Spillover:** the effect of the intervention on the control group.
- **Contamination:** the control group receives part of the intervention.
- **Compliance:** the degree to which the intervention was delivered as designed.
- **Fidelity:** the degree to which the intervention was delivered as intended.
- **Intention-to-treat:** analysis by assigned group, regardless of compliance.
- **Per-protocol:** analysis by treatment received.
- **Difference-in-differences estimator:** the change in treated minus the change in control.
- **Event study:** a regression with time dummies around the intervention.
- **Synthetic control:** a weighted combination of controls.
- **Autocorrelation:** correlation of a series with its past.
- **Newey-West standard error:** a standard error robust to autocorrelation.
- **Pre-registration:** a plan written before the analysis.
- **Primary outcome:** the outcome the evaluation is powered for.
- **Secondary outcome:** other outcomes of interest.
- **Subgroup analysis:** an analysis within a subgroup.
- **Heterogeneity:** variation in effect across groups.
- **Decision rule:** the rule for continuing or stopping the programme.
- **Fidelity-adjusted effect:** the effect adjusted for the delivered dose.
- **Dose-response:** the relationship between intervention intensity and effect.
- **Stepped-wedge diagram:** a plot of unit switch dates.
- **Exposure:** the dose of the intervention a unit received.
- **Dose-response:** the relationship between exposure and effect.
- **Washout period:** the time for an effect to disappear after withdrawal.
- **Lag:** the delay between the intervention and the effect.
- **Contamination:** the control receiving part of the intervention.
- **Randomisation inference:** a test that does not assume a distribution.
- **Cluster-level covariate:** a variable measured at the unit level.


## 16. Sources

All URLs were accessed on 25 October 2026.

### Methods

1. Lopez Bernal, J., Cummins, S., Gasparrini, A. Interrupted time series regression for the evaluation of public health interventions: a tutorial. *International Journal of Epidemiology*, 2017. https://pmc.ncbi.nlm.nih.gov/articles/PMC5407170/
2. Li, L. et al. Three statistical approaches for assessment of intervention effects: a primer for practitioners. https://pmc.ncbi.nlm.nih.gov/articles/PMC7910529/
3. Feng, Y. et al. Difference-in-differences for health policy and practice: a review of modern methods. *Statistics in Medicine*, 2025. https://doi.org/10.1002/sim.70247
4. Haber, N. et al. COVID-19 policy impact evaluation: a guide to common design issues. *American Journal of Epidemiology*, 2021. https://pmc.ncbi.nlm.nih.gov/articles/PMC8344590/
5. Turner, S. L. et al. Design characteristics and statistical methods used in interrupted time series studies evaluating public health interventions: a review. *Journal of Clinical Epidemiology*, 2020. https://doi.org/10.1016/j.jclinepi.2020.02.006
6. Wing, C., Simon, K., Bello-Gomez, R. A. Designing difference in difference studies: best practices for public health policy research. *Annual Review of Public Health*, 2018. https://doi.org/10.1146/annurev-publhealth-040617-013507

### Reporting standards

7. CONSORT. Transparent reporting of trials. https://www.consort-statement.org/
8. CONSORT extension for stepped-wedge cluster randomised trials. https://www.consort-statement.org/extensions
9. TREND. Transparent reporting of evaluations with nonrandomized designs. https://www.cdc.gov/trendstatement/
10. STROBE. Strengthening the reporting of observational studies in epidemiology. https://www.strobe-statement.org/
11. SQUIRE. Standards for quality improvement reporting excellence. https://www.squire-statement.org/

### Operational evidence

12. Artenstein, A. W. et al. Decreasing emergency department walkout rate and boarding hours by improving inpatient length of stay. *Western Journal of Emergency Medicine*, 2017. https://pubmed.ncbi.nlm.nih.gov/29085527/
13. Kausar, S. et al. Implementing and evaluating a discharge before noon initiative in a large tertiary care urban hospital. *Joint Commission Journal on Quality and Patient Safety*, 2023. https://www.jointcommissionjournal.com/article/S1553-7250(23)00212-X/abstract
14. Hodges, J. et al. Driving in the wrong direction: exploring the unintended consequences of an early discharge program. *Quality Management in Health Care*, 2024. https://doi.org/10.1097/qmh.0000000000000466
15. Marafino, B. J. et al. Evaluation of an intervention targeted with predictive analytics to prevent readmissions in an integrated health system: observational study. *BMJ* 2021;374:n1747. https://www.bmj.com/content/374/bmj.n1747
16. CMS. Hospital Readmissions Reduction Program. https://qualitynet.cms.gov/inpatient/hrrp
17. AHRQ. Healthcare Cost and Utilization Project. https://hcup-us.ahrq.gov/
18. IHI. Quality improvement methods. https://www.ihi.org/
19. Cochrane Effective Practice and Organisation of Care. https://epoc.cochrane.org/
20. Medical Research Council. Developing and evaluating complex interventions. https://www.ukri.org/councils/mrc/

> **Note:** The design labels in the literature are not uniform. A study described as a "pre-post with comparison" may be a DiD; a study described as a "time series" may be a simple before-after. Read the methods, not only the label.

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`07_revenue-cycle.md`](07_revenue-cycle.md) — denials, prior authorization, and cash.
- Next topic: [`09_case-studies.md`](09_case-studies.md) — documented operational analytics programmes.
- Related topic: [`04_operational-models.md`](04_operational-models.md) — the models whose effect this topic evaluates.
- Root survey file: [`../../08_ehr_claims_operations.md`](../../08_ehr_claims_operations.md)
- Labs: [`../../labs/`](../../labs/)

### Reading path

| If you need to | Read | Then |
|---|---|---|
| Evaluate a single-site programme | Sections 5 and 6 | The placebo test |
| Compare two sites | Sections 5 and 9 | The parallel-trends check |
| Design a phased rollout | Sections 5 and 9 | The stepped-wedge sources |
| Compute cost per avoided event | Sections 6 and 11 | Topic 01 for the cost basis |
| Write an evaluation report | Sections 5 and 8 | The reporting standards |

The evaluation is the bridge between an operational model and a decision. A model that ranks patients is a hypothesis; an evaluation that estimates the counterfactual is evidence. The next topic examines public programmes through this lens, separating what was documented from what was claimed.

A design is only as good as its diagnostics. The next topic applies this standard to public programmes, where the gap between a vendor claim and a peer-reviewed evaluation is often the whole story.

A reader who can name the counterfactual, the assumption, and the falsification test can judge an evaluation without re-running it.

The test of the topic is whether the reader can look at an evaluation and say what would have happened without the intervention. If the answer is not in the report, the evaluation is incomplete.

| Scenario | Counterfactual | Effect estimate | Credibility |
|---|---|---|---:|---|
| Pre-post | Pre-period mean | -2.3 pp | Low |
| ITS | Pre-trend | -1.4 pp | Moderate |
| DiD | Control change | -1.6 pp | Moderate |
| Randomised | Random assignment | Not estimated | High |

The table is a compact way to present the design hierarchy to a decision committee.

The committee should see all four rows, not only the one the analyst prefers.

If the randomised row is not available, the report should say so and explain why.

The credibility column is a judgement, not a statistical output, and it should be justified in the text.

A high-credibility design with a null result is more useful than a low-credibility design with a positive result.

That is the standard the next topic applies to public case studies.

It is also the standard a reviewer should apply to this topic's worked example.

The worked example passes the placebo test; that is why its estimate is credible.

That is the difference between a result and a finding.
