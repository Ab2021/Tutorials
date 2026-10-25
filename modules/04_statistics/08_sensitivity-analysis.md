# Sensitivity Analysis

> Module 04 — Statistics and Causal Inference
> Topic 08 of 10
> Estimated study time: 3.5 hours
> Prerequisites: 05, 06
> Version 2026.3

---

## 1. Why this topic exists

Every causal estimate rests on assumptions that the data cannot test. The most important is no unmeasured confounding: the claim that the adjustment set blocks every backdoor path, including paths through variables that were never measured. No balance table, no p-value, and no confidence interval can verify that claim. The interval describes sampling variability under the model; it says nothing about bias from an omitted cause.

A sensitivity analysis asks a different question: how wrong would the assumption have to be to change the conclusion? It does not prove the assumption. It bounds the damage if the assumption fails. That is the most honest statement an observational analysis can make.

The need is practical. A vendor reports an adjusted hazard ratio of 0.75 for a program and recommends expansion. The confidence interval excludes 1. The balance table is clean. The question a decision-maker should ask is: "Could an unmeasured difference between the groups, of a plausible size, produce this result?" A sensitivity analysis answers that question with a number: an unmeasured confounder would need a risk ratio of 2.5 with both enrollment and the outcome, above and beyond the measured covariates, to explain away the estimate. The decision-maker can then judge whether a 2.5-fold confounder is plausible.

There are several families of sensitivity analysis, and they answer different questions:

- **E-values** summarize the strength of unmeasured confounding needed to explain away an association. They are quick, transparent, and applicable to any effect measure that can be converted to a risk ratio.
- **Quantitative bias analysis** specifies a confounder's prevalence and associations and recomputes the estimate under that scenario. It is more informative than an E-value when a specific confounder is suspected.
- **Negative controls** test the design for residual bias. An effect on an outcome that the treatment cannot plausibly affect is evidence that the adjustment is incomplete.
- **Delta adjustments and tipping-point analyses** stress-test missing-data and misclassification assumptions.
- **Monte Carlo bias analysis** propagates uncertainty in the bias parameters to produce a bias-adjusted interval.

The common structure is the same: identify one assumption, specify a plausible departure, recompute, and report how the conclusion changes. A sensitivity analysis that recomputes with a different model but the same assumption is not a sensitivity analysis; it is a robustness check of the software.

Regulators and reporting guidelines expect the analysis. The International Council for Harmonisation (ICH) E9(R1) addendum requires sensitivity analyses that target the same estimand under different assumptions. The Strengthening the Reporting of Observational Studies in Epidemiology (STROBE) statement requires a discussion of the limitations, and reviewers increasingly expect the discussion to be quantitative. The E-value has become a common reporting standard in epidemiology and health services research since VanderWeele and Ding introduced it in the Annals of Internal Medicine in 2017.

This topic builds the mechanics: the E-value, the bias-adjustment formula, negative controls, delta adjustment, the tipping point, and the reporting that makes the result usable.

> **Note:** A sensitivity analysis does not make a biased estimate valid. It tells the reader how much bias would be required to change the conclusion. If that amount is small and plausible, the estimate is fragile regardless of its confidence interval.

## 2. Learning outcomes

By the end of this topic you can:

1. **Identify** the assumption that the primary analysis cannot test and **state** what would falsify it.
2. **Compute** an E-value for a risk ratio, an odds ratio, and a hazard ratio, and **interpret** the E-value for the confidence limit.
3. **Perform** a quantitative bias analysis for an unmeasured binary confounder, including a bias-adjusted interval.
4. **Design** a negative control analysis and **explain** what a positive result does and does not establish.
5. **Conduct** a delta-adjusted or tipping-point analysis for missing data or misclassification.
6. **Combine** multiple biases in a Monte Carlo analysis and **report** the resulting interval.
7. **Choose** the sensitivity analysis that matches the decision and the suspected threat.
8. **Report** the sensitivity analysis with the assumption, the parameter range, the result, and the decision implication.

Outcome 2 maps to Exercise 2 and the worked example. Outcome 3 maps to Exercise 3. Outcome 4 maps to Exercise 5.

## 3. The core mental model

A sensitivity analysis is a controlled perturbation of one assumption, followed by a re-estimate and a comparison with the decision threshold.

```text
  PRIMARY ESTIMATE
  (effect, interval, assumption A)
        |
        |  identify the threat to A
        v
  SENSITIVITY PARAMETER
  (confounder prevalence, association, delta, misclassification rate)
        |
        |  choose a plausible range
        v
  RE-ESTIMATE UNDER EACH VALUE
        |
        +--> bias-adjusted estimate
        +--> bias-adjusted interval
        +--> negative control result
        |
        v
  TIPPING POINT  (the value at which the conclusion changes)
        |
        |  compare with plausibility
        v
  DECISION: robust, fragile, or unresolved
```

### 3.1 The four questions a sensitivity analysis answers

| Question | Method | Output |
|---|---|---|
| How strong would an unmeasured confounder have to be? | E-value | A risk ratio |
| What would the estimate be under a specific confounder? | Bias analysis | A corrected estimate and interval |
| Is there evidence of residual bias? | Negative control | An association that should be null |
| How much missingness or misclassification would change the result? | Delta adjustment; tipping point | A threshold value |

A complete report answers at least one of these and states which one. A report that answers none has not done a sensitivity analysis.

### 3.2 The E-value in one paragraph

The E-value is the minimum strength of association, on the risk ratio scale, that an unmeasured confounder would need to have with both the treatment and the outcome, conditional on the measured covariates, to fully explain away the observed association. A large E-value means that only a strong confounder could explain the result; a small E-value means that a weak confounder could.

The formula for a risk ratio above 1 is

```text
E = RR + sqrt(RR × (RR - 1))
```

For a risk ratio below 1, invert the ratio first:

```text
E = 1/RR + sqrt((1/RR) × (1/RR - 1))
```

For an odds ratio with a rare outcome, the risk ratio can be approximated by the square root of the odds ratio, but the approximation must be stated. For a hazard ratio, the same approximation is sometimes used, and it is weaker because the hazard ratio is not a risk ratio.

### 3.3 The bias-adjustment formula

For a binary unmeasured confounder U with prevalence p1 in the treated and p0 in the control, and a confounder-outcome risk ratio RR_UD, the observed risk ratio RR_obs is related to the true risk ratio RR_true by

```text
RR_obs = RR_true × [(1 + p1(RR_UD - 1)) / (1 + p0(RR_UD - 1))]
```

Rearranging gives the bias-adjusted estimate:

```text
RR_true = RR_obs / [(1 + p1(RR_UD - 1)) / (1 + p0(RR_UD - 1))]
```

The term in brackets is the bias factor. It is 1 when the prevalence is equal in both groups or when the confounder has no effect on the outcome. The adjustment assumes that U is binary, that its effect is the same in both groups, and that no other confounder is present. Those assumptions are part of the analysis and must be reported.

### 3.4 Negative controls

A negative control outcome is an outcome that the treatment cannot plausibly affect through the hypothesized mechanism, but that shares the same confounding structure as the primary outcome. A negative control exposure is an exposure that cannot plausibly affect the outcome, but that shares the same confounding structure as the primary exposure.

If the analysis shows an association with the negative control, the design has residual bias. The control does not identify the bias or correct it; it detects it. The magnitude of the bias in the primary analysis cannot generally be inferred from the magnitude of the negative control association without additional assumptions.

The classic example is the influenza vaccine literature. Observational studies found that vaccination in older adults was associated with roughly half the risk of pneumonia hospitalization and all-cause mortality. Jackson and colleagues showed that the vaccine also appeared to protect against injury and trauma hospitalization, an outcome it cannot prevent, and that the apparent protection was largest before the influenza season. Both findings pointed to residual confounding by health status and health-seeking behavior.

### 3.5 Tipping points and delta adjustment

A tipping point is the value of a sensitivity parameter at which the conclusion changes. For missing data, the parameter is delta, the difference between the missing and observed outcomes. For misclassification, it is the error rate. For unmeasured confounding, it is the strength of the confounder.

The tipping-point analysis is not a confidence interval. It answers a counterfactual question: "how wrong would the assumption have to be?" The answer is compared with the plausible range from clinical knowledge. If the tipping point lies outside the plausible range, the conclusion is robust to that threat. If it lies inside, the conclusion is fragile.

### 3.6 Multiple biases

Real analyses face several threats at once. A Monte Carlo bias analysis assigns a distribution to each bias parameter, draws values, recomputes the estimate for each draw, and summarizes the resulting distribution. The output is a bias-adjusted interval that reflects uncertainty in the bias parameters as well as sampling error.

The method is more realistic than a single-scenario analysis, and it is also more assumption-laden: the distributions of the bias parameters are themselves assumptions. Report them and test the sensitivity of the result to the distributional choices.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or scale |
|---|---|---|---|
| Sensitivity analysis | An analysis under a different assumption to test robustness | Repeating the main analysis with a different model | Analysis |
| Bias analysis | A quantitative adjustment for a specific bias | A qualitative limitation paragraph | Analysis |
| E-value | The minimum confounder strength needed to explain away an association | A correction for confounding | Risk ratio |
| Bias factor | The multiplier relating the observed and true effect | Ignored in the interpretation | Unitless |
| Confounder prevalence | The proportion with the unmeasured confounder, by exposure group | Assumed equal between groups | Proportion |
| Confounder-outcome association | The effect of the confounder on the outcome | Assumed absent | Risk ratio |
| Negative control outcome | An outcome the treatment cannot affect through the mechanism | Any unrelated outcome | Outcome |
| Negative control exposure | An exposure that cannot affect the outcome | A variable correlated with treatment | Exposure |
| Tipping point | The sensitivity parameter value at which the conclusion changes | A p-value threshold | Parameter units |
| Delta adjustment | Shifting imputed values by a fixed amount | A correction for MNAR | Outcome units |
| Monte Carlo bias analysis | Propagating bias-parameter uncertainty by simulation | A substitute for a sensitivity analysis | Simulation |
| Quantitative bias analysis | A numerical correction for bias | A qualitative discussion | Analysis |
| Selection bias | Bias from conditioning on a selection variable | Called confounding | Bias |
| Misclassification | Error in measuring exposure, outcome, or confounder | Ignored when non-differential | Error rate |
| Unmeasured confounding | A common cause not in the data | Assumed away | Bias |
| Robustness | The stability of the conclusion across assumptions | The width of the confidence interval | Judgment |
| Fragility | The ease with which the conclusion changes | Statistical significance | Judgment |
| Plausible range | The range of sensitivity parameter values supported by knowledge | The range that preserves significance | Range |
| Bias-adjusted interval | An interval that includes bias-parameter uncertainty | A confidence interval | Interval |
| Probabilistic bias analysis | A bias analysis with distributions on the parameters | A deterministic analysis | Analysis |
| Simple bias analysis | A bias analysis with fixed parameter values | A sensitivity analysis | Analysis |
| Direction of bias | Whether the bias moves the estimate toward or away from the null | Assumed to be toward the null | Direction |
| Magnitude of bias | How far the estimate moves | Assumed to be small | Effect units |
| Residual confounding | Confounding that remains after adjustment | Any confounding | Bias |
| Healthy-user bias | Healthier people are more likely to receive treatment | A form of confounding | Bias |
| Confounding by indication | Sicker patients are more likely to receive treatment | A form of confounding | Bias |
| Immortal time | Follow-up during which the outcome cannot occur | A form of selection bias | Time |
| Quantitative reporting | Reporting the sensitivity result numerically | A narrative caveat | Reporting |
| Assumption ledger | A list of assumptions and their tests | A limitations paragraph | Document |
| Decision threshold | The effect at which the decision changes | The null hypothesis | Effect units |
| Robust conclusion | A conclusion that survives plausible departures | A significant result | Judgment |

Two conventions in this topic:

- An E-value is reported for the point estimate and for the confidence limit closest to the null: "E-value 2.90 for the estimate, 1.69 for the lower confidence limit."
- A bias analysis reports the bias factor, the adjusted estimate, and the adjusted interval, not only the adjusted point.

> **Note:** A sensitivity analysis is only as useful as the plausibility range it uses. An E-value of 3.0 is reassuring if no plausible confounder has a risk ratio of 3.0 with both treatment and outcome; it is not reassuring if the suspected confounder plausibly does.

## 5. The mechanics, step by step

The procedure has twelve steps. Steps 1 to 4 frame the analysis. Steps 5 to 10 compute it. Steps 11 and 12 interpret and report.

### Step 1. Identify the assumption and the threat

**Input:** The primary analysis, its identification assumptions, and the DAG.

**Output:** One named assumption, one named threat, and a statement of what would falsify the assumption.

**Why it exists:** A sensitivity analysis aimed at the wrong threat is a robustness check, not a sensitivity analysis. The threat should be the one that is most plausible and most consequential for the decision.

| Assumption | Threat | Evidence for the threat | Sensitivity method |
|---|---|---|---|
| No unmeasured confounding | An unmeasured prognostic variable affects treatment | Clinical knowledge; prior literature | E-value; bias analysis |
| No selection bias | The cohort is defined by a post-treatment event | Enrollment or restriction on a collider | Negative control; quantitative selection bias |
| No misclassification | The outcome or exposure is coded with error | Validation study; coding changes | Bias analysis for misclassification |
| MAR | Missingness depends on the unobserved value | Dropout for lack of efficacy | Delta adjustment; tipping point |
| Correct time zero | Treatment starts after baseline | Grace period; prevalent users | Time-varying analysis |
| Correct model | The functional form is wrong | Nonlinearity; interaction | Alternative specifications |

### Step 2. Fix the decision threshold

**Input:** The decision and its economics.

**Output:** The effect size at which the decision changes.

**Why it exists:** Robustness is relative to a threshold. A result can be robust at zero and fragile at a cost-effectiveness threshold, or the reverse.

| Decision | Threshold | Why it matters |
|---|---|---|
| Adopt a program | Break-even risk difference | The sensitivity analysis is judged against the break-even |
| Recommend a drug | Minimal clinically important difference | A statistically robust effect can be clinically trivial |
| Reimburse a device | Cost per QALY threshold | The effect must be large enough to justify the price |
| Change a guideline | Net benefit threshold | The threshold combines benefits and harms |

### Step 3. Choose the method

**Input:** The threat and the decision threshold.

**Output:** A named method and the output it will produce.

**Why it exists:** Different threats require different methods; a method aimed at the wrong threat is a robustness check.

| Threat | Method | Output | When to prefer it |
|---|---|---|---|
| Unmeasured confounding, no specific variable | E-value | Minimum confounder strength | Quick, transparent, comparable across studies |
| Unmeasured confounding, specific variable suspected | Bias analysis | Adjusted estimate and interval | A prevalence and association can be defended |
| Residual bias, unknown source | Negative control | A test of the design | A credible control exists |
| Missing outcome, MNAR | Delta adjustment | Tipping point | The missingness mechanism is plausible |
| Misclassification | Bias analysis | Adjusted estimate | Validation data exist |
| Multiple threats | Monte Carlo | Bias-adjusted interval | Several biases act together |

### Step 4. Specify the sensitivity parameter and its plausible range

**Input:** Clinical knowledge, prior studies, and the data.

**Output:** A parameter, a range, and a justification for the endpoints.

**Why it exists:** The plausibility range is the most subjective part of the analysis and the part that determines its value. A range chosen to preserve significance is not a sensitivity analysis.

| Parameter | Range source | Example |
|---|---|---|
| Confounder prevalence difference | Prior studies; clinical judgment | 10 to 40 percentage points |
| Confounder-outcome RR | Published associations | 1.5 to 3.0 |
| Delta for missing outcome | MCID; observed score distribution | -5 to +5 points |
| Misclassification sensitivity | Validation study | 0.85 to 0.95 |
| Selection odds ratio | External data | 1.0 to 2.0 |

**Edge case:** When the range is unknown, report the E-value and let the reader judge the plausibility. The E-value does not require a range; it reports the threshold.

### Step 5. Compute the E-value

**Input:** The effect measure and its confidence interval.

**Output:** An E-value for the estimate and for the confidence limit closest to the null.

**Why it exists:** The E-value is the fastest way to communicate how strong an unmeasured confounder would have to be.

```text
For RR >= 1: E = RR + sqrt(RR × (RR - 1))
For RR < 1:  E = 1/RR + sqrt((1/RR) × (1/RR - 1))
```

For an odds ratio with a rare outcome, use RR ≈ sqrt(OR). For a hazard ratio, the approximation is weaker and should be labelled.

**Decision table for the effect measure:**

| Measure | Conversion | Note |
|---|---|---|
| Risk ratio | Direct | Preferred |
| Odds ratio, outcome under 15% | RR ≈ sqrt(OR) | State the approximation |
| Odds ratio, common outcome | Convert using the control risk | Requires the baseline risk |
| Hazard ratio | RR ≈ sqrt(HR) or use the HR directly with caution | The HR is not a risk ratio |
| Risk difference | Convert to RR using the baseline risk | Requires the baseline risk |
| Mean difference | Standardize to an RR only if a dichotomization is meaningful | Otherwise use a different method |

### E-values for common effect sizes

| Effect measure | Value | E-value | Interpretation |
|---|---|---:|---|
| RR | 1.10 | 1.32 | A weak confounder could explain it |
| RR | 1.25 | 1.71 | A moderate confounder could explain it |
| RR | 1.50 | 2.37 | A fairly strong confounder is needed |
| RR | 2.00 | 3.41 | A strong confounder is needed |
| RR | 0.80 | 1.81 | Invert to 1.25, E = 1.71 |
| RR | 0.50 | 3.41 | Invert to 2.00, E = 3.41 |

The table shows why an E-value below about 2 is usually a warning: many unmeasured confounders in biomedical data have associations of that size. An E-value above 3 is more reassuring, but it is not proof.

### Step 6. Compute the bias-adjusted estimate

**Input:** The observed effect, the confounder prevalence by group, and the confounder-outcome association.

**Output:** A bias factor, an adjusted estimate, and an adjusted interval.

**Why it exists:** The E-value says how strong a confounder would need to be; the bias analysis says what the estimate would be if the confounder had a specific strength.

```text
Bias factor = (1 + p1(RR_UD - 1)) / (1 + p0(RR_UD - 1))
RR_true = RR_obs / Bias factor
Adjusted CI = (CI_lower / Bias factor, CI_upper / Bias factor)
```

The adjusted interval assumes the bias parameters are known. A Monte Carlo analysis relaxes that assumption.

**Edge case:** If the bias factor is larger than the observed RR, the adjusted estimate crosses 1. That is the tipping point for this confounder specification. Report the factor and the adjusted estimate, not only the conclusion.

### A bias-analysis grid

A single scenario is hard to interpret. A grid shows how the adjusted estimate moves across plausible values:

| RR_UD | p1 - p0 = 0.10 | p1 - p0 = 0.20 | p1 - p0 = 0.30 |
|---:|---:|---:|---:|
| 1.5 | 1.67 | 1.60 | 1.53 |
| 2.0 | 1.60 | 1.48 | 1.38 |
| 3.0 | 1.50 | 1.31 | 1.17 |
| 4.0 | 1.42 | 1.20 | 1.03 |

The grid is computed for an observed RR of 1.75 and p0 = 0.10, using the bias formula. It shows that the result is most sensitive to the prevalence difference when the confounder-outcome association is strong. A report can include the grid instead of a single scenario, and the reader can locate the plausible cell.

### Step 7. Design the negative control

**Input:** The DAG and the candidate outcomes or exposures.

**Output:** A control with a stated rationale and a result.

**Why it exists:** A negative control tests the design for bias that the analyst has not specified. It is the only method that can detect an unknown confounder.

| Control type | Definition | Example | What a positive result means |
|---|---|---|---|
| Negative control outcome | Cannot be affected by the treatment through the mechanism | Injury hospitalization for a vaccine | Residual confounding or detection bias |
| Negative control exposure | Cannot affect the outcome | A future exposure | Bias in the design or analysis |
| Pre-period outcome | Occurs before the treatment | Hospitalization before enrollment | Selection or confounding |

**Caveat.** A negative control must share the confounding structure of the primary outcome. An outcome with different confounders is not a valid control, and a null result does not prove the absence of bias.

### Step 8. Run the tipping-point analysis

**Input:** The sensitivity parameter and a grid of values.

**Output:** The value at which the conclusion changes.

**Why it exists:** Decision-makers can judge a threshold value more easily than a distribution.

- For a binary decision (adopt or not), the tipping point is the parameter value at which the confidence interval crosses the decision threshold.
- For a continuous decision, the tipping point is the value at which the expected net benefit changes sign.
- Report the tipping point in the units of the parameter and in clinical terms.

### Step 9. Run a Monte Carlo bias analysis

**Input:** Distributions for each bias parameter.

**Output:** A bias-adjusted distribution and interval.

**Why it exists:** A single-scenario analysis understates uncertainty. Monte Carlo propagates the uncertainty in the bias parameters.

Procedure:

1. Assign a distribution to each bias parameter, justified by prior data or expert judgment.
2. Draw a value from each distribution.
3. Recompute the adjusted estimate.
4. Repeat many times.
5. Report the median, the 2.5th and 97.5th percentiles, and the proportion of draws above the decision threshold.

**Edge case:** The output is only as good as the distributions. Report them, and run the analysis with an alternative distribution as a check.

### Step 10. Combine multiple biases

**Input:** The list of threats.

**Output:** A combined bias-adjusted estimate.

**Why it exists:** Real analyses face several threats at once. Correcting one at a time can understate the total bias, and the biases can interact.

| Combination | Method | Note |
|---|---|---|
| Confounding plus misclassification | Monte Carlo with both parameters | The order of correction matters |
| Confounding plus selection | Bias analysis with a selection model | Requires assumptions about the selection |
| Missing data plus confounding | Delta adjustment plus E-value | Report both |

### Step 11. Judge plausibility

**Input:** The E-value, the bias-adjusted estimate, the negative control, and the tipping point.

**Output:** A judgment: robust, fragile, or unresolved.

**Why it exists:** The numbers do not interpret themselves. The analyst must compare the required bias with the plausible bias.

| Result | Judgment | Report |
|---|---|---|
| E-value far above any plausible confounder | Robust | "The result would require an implausibly strong confounder" |
| E-value within the plausible range | Fragile | "A confounder of plausible strength could explain the result" |
| Negative control shows an effect | Design problem | "The design shows residual bias; the primary estimate is not interpretable" |
| Tipping point inside the plausible range | Fragile | "The conclusion depends on the missing-data assumption" |

### Sensitivity analysis versus robustness check

| Activity | What changes | What it establishes | Example |
|---|---|---|---|
| Sensitivity analysis | One assumption | How much the conclusion depends on it | E-value; delta adjustment |
| Robustness check | The model or estimator | Whether the software or specification matters | Refit with a spline |
| Subgroup analysis | The population | Whether the effect varies | Effect in older patients |
| Negative control | Nothing; the outcome changes | Whether residual bias is present | Injury hospitalization |
| Validation | The data | Whether the measurement is correct | Chart review |

A report that labels a robustness check as a sensitivity analysis is overstating what it did. The distinction matters because only a sensitivity analysis addresses the untestable assumption.

### Choosing between an E-value and a bias analysis

| Situation | E-value | Bias analysis |
|---|---|---|
| No specific confounder identified | Preferred | Not possible |
| A specific confounder is suspected and measured externally | Useful summary | Preferred |
| The confounder is binary and its prevalence is known | Approximate | Exact under assumptions |
| The outcome is common and the measure is an OR | Requires conversion | Can use the risk scale |
| The decision needs a corrected estimate | Not provided | Provided |
| The audience is non-technical | Easy to explain | Harder to explain |

The two are complements. A strong report often includes both: the E-value for the summary and the bias analysis for the specific threat.

### Branch points at a glance

| If the threat is | Then the default is | Because |
|---|---|---|
| Unmeasured confounding, no candidate | E-value | No parameters to specify |
| Unmeasured confounding, candidate known | Bias analysis with a grid | Produces a corrected estimate |
| Unknown residual bias | Negative control | Detects what was not specified |
| Missing outcome, MNAR plausible | Delta adjustment | Transparent tipping point |
| Exposure misclassified | Bias analysis with validation rates | Corrects attenuation |
| Selection on a collider | Redesign or selection bias analysis | The bias is not confounding |
| Several threats at once | Monte Carlo | Propagates joint uncertainty |
| Decision at a non-null threshold | Tipping point against the threshold | Robustness is threshold-relative |

### Reporting table shell

| Assumption | Method | Parameter range | Result | Judgment |
|---|---|---|---|---|
| No unmeasured confounding | E-value | Not applicable | 2.90 (1.69 for lower CI) | Fragile |
| No unmeasured confounding | Bias analysis | RR_UD 2.0 to 3.0 | 1.50 to 1.00 | Fragile |
| No residual bias | Negative control | Not applicable | 1.05 (0.92 to 1.20) | Reassuring |

The shell makes the analysis auditable in three rows. A report with this table and a paragraph of interpretation is more useful than a page of narrative caveats.

### Common mistakes

- Reporting an E-value without saying which effect measure was used.
- Reporting the E-value for the point estimate only.
- Choosing the sensitivity range after seeing the result.
- Calling a different model a sensitivity analysis.
- Reporting a negative control as proof of validity.
- Using the odds ratio as a risk ratio in the E-value formula without conversion.
- Ignoring the direction of bias when combining threats.

### Step 12. Report

**Input:** The primary analysis and the sensitivity results.

**Output:** A results section that states the assumption, the method, the parameters, the result, and the implication.

**Why it exists:** The numbers do not interpret themselves; the judgment must be explicit.

The reporting checklist:

1. The assumption being stressed.
2. The method and the sensitivity parameter.
3. The plausible range and its justification.
4. The E-value for the estimate and the confidence limit.
5. The bias-adjusted estimate and interval.
6. The negative control result.
7. The tipping point.
8. The judgment and its basis.
9. The effect on the decision.

### Edge cases that change the analysis

| Edge case | What happens | Detection | Remedy |
|---|---|---|---|
| Protective effect (RR below 1) | The E-value formula must be inverted | The estimate is below 1 | Use the inverted formula |
| Common outcome with an OR | The OR cannot be converted with sqrt | Control risk above 15% | Convert using the baseline risk |
| Confounder prevalence above 1 after solving | The scenario is impossible | The required prevalence exceeds 1 | Increase the assumed association |
| Differential misclassification | The bias can be away from the null | Validation rates differ by group | Model both error rates |
| Multiple biases in opposite directions | The net bias can be small | The individual biases point in different directions | Monte Carlo with signs |
| Selection on a collider | The bias is not confounding | The cohort is defined by a post-treatment event | Redesign; selection bias analysis |

## 6. Worked example

### 6.1 The estimate and the decision

A health system evaluates a care-coordination program. An adjusted analysis of 4,000 patients gives a risk ratio of 1.75 for 12-month hospitalization among non-enrolled patients compared with enrolled patients. (The direction is chosen so the ratio is above 1; the program is protective if the ratio is below 1, and the same machinery applies after inversion.) The 95% confidence interval is 1.20 to 2.55.

The decision threshold is a risk ratio of 1.30: the program would be cost-saving if non-enrollment were at least 30% more likely to lead to hospitalization. Below that, the program is not justified by the projected savings.

The primary assumption is no unmeasured confounding. The most plausible unmeasured confounders are health literacy and social support, neither of which is measured in the claims data.

### 6.2 The E-value

```text
RR = 1.75
E = 1.75 + sqrt(1.75 × 0.75)
  = 1.75 + sqrt(1.3125)
  = 1.75 + 1.1456
  = 2.8956

Lower confidence limit RR = 1.20
E = 1.20 + sqrt(1.20 × 0.20)
  = 1.20 + 0.4899
  = 1.6899
```

An unmeasured confounder would need a risk ratio of about 2.90 with both enrollment and hospitalization, above and beyond the measured covariates, to explain away the point estimate. To explain away the lower confidence limit, it would need a risk ratio of about 1.69.

Both values are within the range of plausible associations for health literacy and social support. The result is therefore not robust to a single strong confounder; it depends on how strongly those variables are associated with both enrollment and hospitalization.

### 6.3 Bias analysis A: a moderate confounder

Suppose health literacy is the confounder. Assume it is present in 40% of non-enrollees and 20% of enrollees, and that it is associated with a risk ratio of 2.0 for hospitalization.

```text
Bias factor = (1 + 0.40 × (2.0 - 1)) / (1 + 0.20 × (2.0 - 1))
            = (1 + 0.40) / (1 + 0.20)
            = 1.40 / 1.20
            = 1.1667

Adjusted RR = 1.75 / 1.1667 = 1.5000
Adjusted 95% CI = (1.20 / 1.1667, 2.55 / 1.1667)
                = (1.0286, 2.1857)
```

Under this scenario, the effect falls from 1.75 to 1.50. It remains above the decision threshold of 1.30, and the lower confidence limit falls just above 1. The program is still supported, but the margin is smaller.

### 6.4 Bias analysis B: the tipping point

Suppose the confounder is stronger: a risk ratio of 3.0 for hospitalization, present in 55% of non-enrollees and 10% of enrollees.

```text
Bias factor = (1 + 0.55 × 2.0) / (1 + 0.10 × 2.0)
            = 2.10 / 1.20
            = 1.7500

Adjusted RR = 1.75 / 1.75 = 1.0000
Adjusted 95% CI = (1.20 / 1.75, 2.55 / 1.75)
                = (0.6857, 1.4571)
```

The bias factor equals the observed RR exactly, so the adjusted estimate is 1.00 and the interval includes both no effect and a meaningful effect. This is the tipping point for this confounder specification: a confounder with a 3.0-fold association and a 45-percentage-point prevalence difference would explain away the entire result.

Is that plausible? A 3.0-fold association with hospitalization is plausible for severe social isolation. A 45-point prevalence difference between enrollees and non-enrollees is large but possible in a program that recruits through primary care. The tipping point is not outside the plausible range, so the result is fragile.

### 6.4b A Monte Carlo version

A single scenario fixes the bias parameters. A Monte Carlo analysis assigns distributions and propagates the uncertainty. Suppose the confounder-outcome RR is log-normal with a median of 2.5 and a 95% interval of 1.5 to 4.0, and the prevalence difference is uniform between 0.10 and 0.40 with p0 fixed at 0.10.

A simulation with 20,000 draws gives:

```text
Median adjusted RR          1.34
2.5th percentile            1.01
97.5th percentile           1.60
P(adjusted RR < 1.30)       0.409
P(adjusted RR < 1.00)       0.021
```

The output is a distribution, an interval, and a probability that the effect is below the threshold. That format is more useful to a decision-maker than a single adjusted value: it says that under the stated distributions, there is about a 41% chance that the true effect is below the break-even threshold.

The distributions are assumptions. A report must state them, and a sensitivity analysis of the sensitivity analysis (a different distribution for the same parameter) is a reasonable check.

The distributions are assumptions. A report must state them, and a sensitivity analysis of the sensitivity analysis (a different distribution for the same parameter) is a reasonable check.

### 6.5 The negative control outcome

The analysis repeats the comparison with an outcome the program cannot plausibly affect: hospitalization for injury or trauma. The adjusted risk ratio is 1.05 (95% CI 0.92 to 1.20).

The negative control shows no association, which is reassuring: there is no evidence of a large residual bias common to both outcomes. It does not prove the absence of confounding for hospitalization, because the confounders of injury may differ from those of hospitalization. A positive control would have been evidence of a design problem; a null control is consistent with a valid design but is not proof.

### 6.6 Diagnostic checks

**Check 1. The bias factor reduces to 1 when the prevalence is equal.** If p1 = p0 = 0.30, the factor is (1 + 0.30)/(1 + 0.30) = 1, and the adjusted estimate equals the observed. This is a check on the formula.

**Check 2. The bias factor reduces to 1 when the confounder has no effect.** If RR_UD = 1, the factor is 1 for any prevalence. This is the other boundary check.

**Check 3. The bias factor in scenario B equals the observed RR.** 1.75 / 1.75 = 1.00 exactly. The arithmetic is consistent with the definition of the tipping point.

**Check 4. The E-value is consistent with the bias analysis.** The E-value of 2.90 is the strength of a confounder with equal prevalence in both groups that would nullify the result. In scenario B, the confounder has unequal prevalence, so a weaker association (3.0 versus 2.9) with a large prevalence difference nullifies the result. The two analyses are consistent: the E-value assumes equal prevalence, and the bias analysis trades prevalence difference against association.

**Check 5. The confidence interval is adjusted with the same factor.** The adjusted interval is the observed interval divided by the bias factor. If the adjusted point estimate lies outside the adjusted interval, the arithmetic is wrong.

**Check 6. The required prevalence is possible.** With RR_UD = 2.0 and p0 = 0.20, solving for the p1 that would nullify the result gives p1 = 1.10, which is impossible. That is a useful finding: a moderate confounder with a plausible prevalence difference cannot explain the result, so the tipping point requires a stronger confounder. Report the impossible solution rather than forcing a scenario.

### 6.6b Summary of the scenarios

| Scenario | RR_UD | p1 | p0 | Bias factor | Adjusted RR (95% CI) |
|---|---:|---:|---:|---:|---|
| Primary | — | — | — | 1.00 | 1.75 (1.20, 2.55) |
| A: moderate | 2.0 | 0.40 | 0.20 | 1.167 | 1.50 (1.03, 2.19) |
| B: tipping | 3.0 | 0.55 | 0.10 | 1.750 | 1.00 (0.69, 1.46) |
| Negative control | — | — | — | — | 1.05 (0.92, 1.20) |

The table is the report. The reader can see the observed estimate, the adjusted estimates, the factor, and the control in one place.

### 6.6c What would change the decision

The decision threshold is 1.30. The decision changes if:

1. The confounder-outcome association is at least 2.5 and the prevalence difference is at least 25 points.
2. The negative control had shown an association above 1.20, which would suggest a design problem.
3. The outcome were common enough that the odds-ratio conversion lowered the E-value to 2.21, making the result fragile at a weaker confounder.
4. The analysis were restricted to a subgroup with a different baseline risk, changing the threshold.

The first item is the most likely. A confounder with a 2.5-fold association and a 25-point prevalence difference is plausible in a program that recruits through primary care, where social support differs substantially between enrollees and non-enrollees. The result is therefore not robust to the most plausible threat.

The residual uncertainty can be summarized in five statements:

- The observed association is 1.75 (1.20 to 2.55).
- The E-value is 2.90, and the lower-limit E-value is 1.69.
- A moderate confounder reduces the effect to 1.50, still above the threshold.
- A strong confounder with a large prevalence difference nullifies the effect.
- The negative control shows no association.

A decision-maker can weigh these statements without reading the model output.

The report should end with a judgment, not a hedge. In this case the judgment is "fragile at the decision threshold," and the recommendation should say so.

### 6.9 How the threshold changes the judgment

| Decision threshold | Scenario A (RR 1.50) | Scenario B (RR 1.00) | Judgment |
|---|---|---|---|
| 1.10 | Above | At | Fragile |
| 1.30 | Above | Below | Fragile |
| 1.50 | At | Below | Fragile |
| 1.70 | Below | Below | Fragile |

The table shows that the judgment "fragile" is stable across a wide range of thresholds, even though the point estimate exceeds some of them. The fragility comes from the width of the plausible range, not from the choice of threshold.

A different report might conclude "robust" if the decision threshold were 1.10 and the plausible confounder range were narrow. The judgment depends on both the data and the decision, which is why a sensitivity analysis without a threshold is incomplete.

The worked example can be summarized in four sentences:

- The observed association is 1.75, with an interval that excludes 1.
- The E-value is 2.90, which is within the plausible range for social support.
- A plausible confounder would reduce the effect to about 1.5, still above the threshold.
- A stronger confounder would nullify it, so the result is fragile.

The four sentences are the report. The rest of the section is the evidence for them.

## 7. Code, query, or calculation

The script computes E-values, bias factors, a grid of adjusted estimates, a Monte Carlo distribution, and an odds-ratio-to-risk-ratio conversion. It uses only the standard library.

**Assumptions:**

- The effect measure is a risk ratio, or an odds ratio converted to a risk ratio.
- The unmeasured confounder is binary and has the same effect in both groups.
- The bias parameters are known in the grid and distributed in the Monte Carlo analysis.
- The Monte Carlo distributions are log-normal for the confounder-outcome RR and uniform for the prevalence difference.
- The simulation seed is fixed for reproducibility.

```python
import math, random

def e_value(rr):
    if rr < 1:
        rr = 1 / rr
    return rr + math.sqrt(rr * (rr - 1))

def bias_factor(rr_ud, p1, p0):
    return (1 + p1 * (rr_ud - 1)) / (1 + p0 * (rr_ud - 1))

observed, lo, hi = 1.75, 1.20, 2.55
print(f"observed RR={observed:.4f} 95% CI=({lo:.4f}, {hi:.4f})")
print(f"E-value point={e_value(observed):.4f} lower CI={e_value(lo):.4f}")

for name, ru, p1, p0 in [("moderate", 2.0, 0.40, 0.20), ("tipping", 3.0, 0.55, 0.10)]:
    f = bias_factor(ru, p1, p0)
    print(f"{name}: RR_UD={ru} p1={p1} p0={p0} factor={f:.4f} "
          f"adjusted={observed/f:.4f} CI=({lo/f:.4f}, {hi/f:.4f})")

print("grid of adjusted RR (p0=0.10):")
for ru in (1.5, 2.0, 3.0, 4.0):
    row = []
    for d in (0.10, 0.20, 0.30):
        row.append(observed / bias_factor(ru, 0.10 + d, 0.10))
    print(f"  RR_UD={ru}: " + " ".join(f"{v:.2f}" for v in row))

# Monte Carlo over a log-normal RR_UD and a uniform prevalence difference
random.seed(8)
mu = math.log(2.5)
sigma = math.log(4 / 1.5) / (2 * 1.96)
draws = []
for _ in range(20000):
    ru = math.exp(random.gauss(mu, sigma))
    p1 = 0.10 + random.uniform(0.10, 0.40)
    draws.append(observed / bias_factor(ru, p1, 0.10))
draws.sort()
n = len(draws)
print(f"MC median={draws[n//2]:.4f} 2.5%={draws[int(0.025*n)]:.4f} "
      f"97.5%={draws[int(0.975*n)]:.4f}")
print(f"P(adjusted<1.30)={sum(v<1.30 for v in draws)/n:.4f} "
      f"P(adjusted<1.00)={sum(v<1.0 for v in draws)/n:.4f}")

# OR to RR conversion when the outcome is common
orr, p0 = 1.75, 0.30
rr = orr / (1 - p0 + p0 * orr)
print(f"OR={orr} at p0={p0}: RR={rr:.4f} E-value={e_value(rr):.4f}")
```

**Expected output:**

```text
observed RR=1.7500 95% CI=(1.2000, 2.5500)
E-value point=2.8956 lower CI=1.6899
moderate: RR_UD=2.0 p1=0.4 p0=0.2 factor=1.1667 adjusted=1.5000 CI=(1.0286, 2.1857)
tipping: RR_UD=3.0 p1=0.55 p0=0.1 factor=1.7500 adjusted=1.0000 CI=(0.6857, 1.4571)
grid of adjusted RR (p0=0.10):
  RR_UD=1.5: 1.67 1.60 1.53
  RR_UD=2.0: 1.60 1.48 1.38
  RR_UD=3.0: 1.50 1.31 1.17
  RR_UD=4.0: 1.42 1.20 1.03
MC median=1.3421 2.5%=1.0125 97.5%=1.6042
P(adjusted<1.30)=0.4094 P(adjusted<1.00)=0.0210
OR=1.75 at p0=0.3: RR=1.4286 E-value=2.2110
```

**What the output shows:**

- The E-value reproduces the hand calculation (2.8956) and the lower-limit value (1.6899). The lower-limit E-value is the one to report; it is smaller and describes the robustness of the interval.
- The moderate scenario gives an adjusted estimate of 1.5000 with an interval that just excludes 1 (1.0286). The tipping scenario gives exactly 1.0000 because the bias factor equals the observed RR.
- The grid shows that the result crosses the decision threshold of 1.30 when the confounder-outcome association is 3.0 and the prevalence difference is about 20 points. That is the tipping region.
- The Monte Carlo median of 1.3421 is close to the moderate scenario, but the interval (1.0125 to 1.6042) is wider because the bias parameters are uncertain. The probability of being below 1.30 is 0.4094, which is a direct decision input.
- The odds-ratio conversion lowers the E-value from 2.8956 to 2.2110. When the outcome is common, the odds ratio overstates the risk ratio, and the sensitivity analysis must use the converted value.

> **Note:** The Monte Carlo output depends on the assumed distributions. A different distribution for the confounder-outcome association would change the median and the interval. Report the distributions and treat the output as one scenario among several.

## 8. Data, standards, and regulatory context

### 8.1 Binding requirements and conventions

| Instrument | Status | What it constrains | What it does not constrain |
|---|---|---|---|
| ICH E9 (1998) | Binding through regional adoption | Pre-specified analysis; robustness of conclusions | The specific sensitivity method |
| ICH E9(R1) (2019) | Binding through regional adoption | Sensitivity analyses that target the same estimand under different assumptions | The choice of E-value or bias analysis |
| FDA RWE guidance (August 2023) | FDA guidance | Data reliability, pre-specified protocol, transparency about assumptions | A required sensitivity analysis |
| STROBE item 19 | Journal editorial policy | Discussion of limitations | Quantitative bias analysis |
| E-value reporting (VanderWeele and Ding, 2017) | Methodological convention | Reporting the E-value for the estimate and the confidence limit | A regulatory requirement |
| Quantitative bias analysis (Lash et al.) | Methodological reference | The structure of a bias analysis | The specific parameters |

### 8.2 What ICH E9(R1) requires

The addendum distinguishes three types of analysis:

- **Main estimation:** the primary analysis targeting the estimand.
- **Sensitivity analysis:** repeated analyses under departures from the assumptions of the main analysis, to test robustness.
- **Supplementary analysis:** analyses that target a different estimand or provide additional insight.

The addendum requires the sensitivity analyses to be pre-specified and to address specific assumptions, not to repeat the main analysis with a different model. An E-value, a delta adjustment, or a bias analysis is a sensitivity analysis in this sense. A refit with a different covariate set is a robustness check, and it may be a supplementary analysis, but it does not test the untestable assumption.

### 8.3 What the FDA RWE guidance adds

The August 2023 guidance emphasizes transparency about the data source and the design. It does not require a specific sensitivity analysis, but a submission that reports an observational effect without addressing unmeasured confounding is incomplete under the general expectation that the sponsor describe the limitations of the design.

The guidance's emphasis on pre-specification applies to the sensitivity analysis as well. A sensitivity analysis chosen after seeing the primary result is a form of model selection, and reviewers treat it accordingly.

### 8.4 What the reporting guidelines require

STROBE item 19 asks for a discussion of the limitations, including the direction and magnitude of potential bias. A quantitative sensitivity analysis is the strongest way to satisfy that item. A statement that "residual confounding cannot be excluded" is weaker than an E-value or a bias analysis, because it does not say how much confounding would matter.

The E-value is not mandated by any regulator. It has become a convention in epidemiology and health services research because it is simple, comparable across studies, and easy to report. The convention is not a substitute for a bias analysis when a specific confounder is suspected.

### 8.5 What a reviewer will ask

| Question | What a strong answer contains | What a weak answer reveals |
|---|---|---|
| Which assumption did you stress? | A named assumption and its threat | "We did a sensitivity analysis" |
| How strong would the bias have to be? | An E-value or a bias factor | "The result is robust" |
| Is that plausible? | A clinical comparison | No judgment |
| Did you test for residual bias? | A negative control result | No control |
| What is the tipping point? | A value in clinical units | A p-value |

The questions are predictable. A report that answers them in advance is easier to review and harder to misinterpret.

### 8.6 Documentation checklist

A sensitivity analysis is reproducible when the report contains:

1. The assumption and the threat.
2. The method and the effect measure.
3. The parameter values or distributions.
4. The source or justification for the range.
5. The primary and adjusted estimates with intervals.
6. The negative control result, if any.
7. The tipping point and its plausibility.
8. The judgment and the decision implication.

Items 3 and 4 are the ones most often missing. Without them, the reader cannot tell whether the range was chosen before or after the result.

### 8.7 The regulatory conversation in practice

| Question from a reviewer | A strong response | A weak response |
|---|---|---|
| How did you handle unmeasured confounding? | An E-value and a bias analysis | "It is a limitation" |
| What is the direction of the bias? | A DAG-based argument | "It could go either way" |
| Why is the result credible? | A negative control and a robustness table | "The association is strong" |
| What would change the conclusion? | A tipping point | "Nothing" |
| Is the analysis pre-specified? | The protocol and SAP | "The methods are standard" |

The conversation is about assumptions, not about software. A team that has prepared the first column can answer the second.

### 8.8 What the standards do not cover

- They do not require a sensitivity analysis for every observational study.
- They do not specify the plausible range for the sensitivity parameter.
- They do not require the negative control.
- They do not resolve the choice between an E-value and a bias analysis.
- They do not validate the assumptions of the bias model.

## 9. Industry practice and case evidence

### 9.1 Documented case: hormone replacement therapy

Observational studies suggested that hormone replacement therapy reduced coronary heart disease. The Women's Health Initiative randomized trial found an increased risk. The observational association was confounded by socioeconomic status and health behaviors, and the available adjustment sets did not block those paths.

A sensitivity analysis at the time could have shown that a confounder with a moderate association with both therapy and coronary disease would explain the result. The E-value framework was not yet available, but the logic was: the observed association was small enough that a plausible confounder could account for it. The trial settled the question by design.

**Inference:** the counterfactual claim that a sensitivity analysis would have changed the interpretation is an inference, not a documented fact. The trial result is documented.

### 9.2 Documented case: influenza vaccine and negative controls

Jackson and colleagues, in a 2006 paper in the International Journal of Epidemiology, reproduced the finding that influenza vaccination in older adults was associated with about half the risk of pneumonia hospitalization and all-cause mortality. They then showed that the vaccine also appeared to protect against injury and trauma hospitalization, an outcome it cannot prevent, and that the apparent protection was largest before the influenza season.

Both findings are negative-control evidence of residual confounding by health status and health-seeking behavior. The vaccine does not prevent trauma, so the association with trauma cannot be causal. The timing pattern is impossible under the hypothesized mechanism.

The case is the canonical demonstration that a large, precise, adjusted association can be an artifact. It is also the reason negative controls are recommended in observational vaccine studies.

### 9.3 Documented practice: E-values in applied research

The E-value was introduced by VanderWeele and Ding in 2017 and has become a common reporting item in epidemiology and health services research. The E-value calculator website and the R package `EValue` provide implementations. The convention is to report the E-value for the estimate and for the confidence limit closest to the null.

**Inference:** the proportion of observational studies that report an E-value is not documented in a single public statistic. The method's adoption is visible in journal reporting guidelines and in the methods sections of recent cohort studies.

### 9.4 Documented case: RCT-DUPLICATE and design sensitivity

The RCT-DUPLICATE initiative emulated 32 trials and reported a correlation of 0.82 between emulated and randomized estimates, rising to 0.93 in the close-emulation subgroup. The variation across emulations is itself a sensitivity analysis: it shows that the result depends on the design choices, not only on the data.

A single emulation with no sensitivity analysis would have reported one number. The initiative reported the distribution across designs, which is more informative about robustness.

### 9.5 Practice pattern: the limitations paragraph

Many industry reports satisfy the sensitivity requirement with a limitations paragraph: "Residual confounding cannot be excluded because of the observational design." That sentence is true and uninformative. It does not say what confounder, how strong, or how much the conclusion would change.

The alternative is three sentences: the assumption, the method, and the result. "We assessed unmeasured confounding with an E-value. An unmeasured confounder would need a risk ratio of 2.9 with both treatment and outcome to explain the estimate, and 1.7 to explain the lower confidence limit. A confounder of that strength is plausible, so the result is fragile." That is a sensitivity analysis in three sentences.

### 9.6 Documented fact versus inference

| Claim | Status | Basis |
|---|---|---|
| A negative control can detect residual confounding | Documented | Jackson et al. 2006; Lipsitch et al. 2010 |
| The E-value is a valid summary of unmeasured confounding | Documented | VanderWeele and Ding 2017 |
| Emulation results vary with design choices | Documented | RCT-DUPLICATE |
| A sensitivity analysis removes confounding | Not supported | It quantifies dependence, not bias |
| A null negative control proves the design is valid | Not supported | The control is approximate and shares only some confounders |
| An E-value above 3 proves robustness | Not supported | Plausibility is a clinical judgment |

The first three rows are the evidence base. The last three are the interpretations the tools do not support. The distinction is the difference between a sensitivity analysis and a certification.

### 9.7 Questions for a team presenting an observational estimate

- Which assumption is most likely to fail?
- What is the E-value for the estimate and for the confidence limit?
- Is there a negative control, and what did it show?
- What is the tipping point in clinical units?
- Was the sensitivity analysis pre-specified?
- What decision would change if the sensitivity analysis were unfavorable?

The questions are answerable in a five-minute conversation. A team that cannot answer them has not finished the analysis, regardless of the confidence interval.

### 9.8 What mature teams do differently

1. They name the assumption before the analysis.
2. They pre-specify the sensitivity method and the parameter range.
3. They report the E-value for the estimate and the confidence limit.
4. They run a negative control when a credible one exists.
5. They report the tipping point in clinical units.
6. They end with a judgment, not a hedge.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| No sensitivity analysis | Only the primary estimate is reported | The key assumption is untested | Overconfidence | E-value or bias analysis |
| E-value reported for the point only | The lower limit is ignored | The interval is more fragile | Overstated robustness | Report both |
| Odds ratio used as a risk ratio | The outcome is common | The E-value is too large | Understated fragility | Convert to RR |
| Range chosen post hoc | The range preserves significance | Selection on the result | Invalid sensitivity analysis | Pre-specify |
| Negative control with different confounders | The control is unrelated to the exposure | It does not share the bias structure | A null result is uninformative | Choose a U-comparable control |
| Negative control interpreted as proof | "No bias was found" | A null control does not prove validity | Overconfidence | Report as a test, not a proof |
| Bias analysis with impossible parameters | Prevalence above 1 | The scenario is impossible | Misleading adjustment | Report the impossibility |
| Several biases ignored | Only one threat is analyzed | The biases combine | Understated total bias | Monte Carlo |
| Tipping point without plausibility | "The result would change at delta 12" | No clinical judgment | Uninterpretable | Compare with the plausible range |
| Sensitivity analysis labeled but not performed | A different model is called sensitivity | The assumption is unchanged | False reassurance | Name the assumption tested |
| Direction of bias assumed toward the null | The bias could be away | The conclusion could be understated | Missed effect | Analyze both directions |
| Monte Carlo distributions unstated | Only the output is reported | The result is not reproducible | Unverifiable | Report the distributions |
| Sensitivity analysis only if significant | The analysis is conditional on the result | Selection on the outcome | Biased reporting | Run it regardless |
| No threshold | The result is called robust without a decision | Robustness is undefined | Uninterpretable claim | State the threshold |

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The bias model is correctly specified | The adjusted estimate depends on it | The confounder is not binary or effects differ | Alternative models |
| The bias parameters are plausible | The range determines the conclusion | The range is chosen post hoc | Pre-specify; justify |
| The negative control shares confounders | The control tests the same bias | Different confounders | DAG review |
| The E-value conversion is valid | The formula applies to the measure | Common outcome with an OR | Convert using baseline risk |
| No other bias dominates | The analysis isolates one threat | Several biases act together | Monte Carlo |
| The sensitivity parameter is identifiable | The tipping point is meaningful | The parameter is not clinically interpretable | Report in clinical units |
| The decision threshold is stable | Robustness is threshold-relative | The threshold is uncertain | Sensitivity over thresholds |
| The primary estimate is unbiased under the model | The sensitivity analysis starts from it | Model misspecification | Alternative specifications |
| The Monte Carlo distributions are defensible | The interval depends on them | Arbitrary distributions | Report; vary them |
| The conclusion is judged, not computed | The numbers do not interpret themselves | No clinical input | Multidisciplinary review |

**Limits of this topic.** A sensitivity analysis quantifies the dependence on one assumption; it does not identify the true effect. The E-value is a summary that assumes a binary confounder with equal effects and no other bias. The bias-adjustment formula assumes a single binary confounder and a known prevalence; real confounders are continuous and correlated. The negative control detects bias but does not measure it. The worked example is illustrative and the Monte Carlo distributions are assumptions, not estimates from a specific dataset.

## 12. Exercises

### Exercise 1. Compute E-values (application, 30 minutes)

**Deliverable.** The E-value for RR = 1.50, RR = 0.70, and OR = 1.60 with a control risk of 5%. State the conversion used for the odds ratio.

**Acceptance criteria.**

- The formula is written for each case.
- The protective effect is inverted before applying the formula.
- The odds ratio is converted with the rare-outcome approximation.
- The answers are compared with the table in Section 5.

### Exercise 2. Bias-adjust an estimate (application, 45 minutes)

**Data.** Observed RR 2.10 (95% CI 1.40 to 3.15). The suspected confounder has RR_UD = 2.5, prevalence 45% in the exposed and 20% in the unexposed.

**Deliverable.** The bias factor, the adjusted RR, the adjusted interval, and the E-value of the adjusted estimate.

**Acceptance criteria.**

- The bias factor is computed with the formula.
- The adjusted interval is divided by the same factor.
- The adjusted estimate is compared with the null.
- The answer states whether the confounder explains the association.

### Exercise 3. Design a negative control (design, 45 minutes)

**Scenario.** A study compares two diabetes drugs and finds a 20% lower risk of myocardial infarction with drug A. The suspected bias is healthier-user selection.

**Deliverable.** Two candidate negative control outcomes, a DAG justification for each, and the interpretation of a positive and a null result.

**Acceptance criteria.**

- Each control cannot plausibly be affected by the drug.
- Each control shares the selection mechanism.
- The interpretation distinguishes detection from correction.
- The answer states what a positive result would imply for the primary estimate.

### Exercise 4. Tipping point for missing data (application, 40 minutes)

**Data.** A pooled mean difference is 2.5 points (SE 1.0). The decision threshold is 1.0 point. Twenty percent of outcomes are missing.

**Deliverable.** The delta at which the point estimate reaches the threshold, and the delta at which the lower confidence limit reaches it.

**Acceptance criteria.**

- The shift is delta × 0.20.
- The point-estimate tipping point is computed.
- The confidence-limit tipping point is computed.
- The answer states whether each delta is clinically plausible.

### Exercise 5. Choose the method (application, 40 minutes)

For each situation, choose an E-value, a bias analysis, or a negative control, and justify it:

1. No specific confounder is suspected.
2. Health literacy is suspected and its prevalence is known from a survey.
3. The outcome is measured by claims and the exposure by pharmacy fills.
4. A new program has a large apparent benefit and the team is skeptical.
5. The decision threshold is a cost per QALY.

**Deliverable.** A five-row table with the method, the output, and the reason.

**Acceptance criteria.**

- At least two methods are used across the five situations.
- Each choice is justified by the information available.
- The output is named for each method.
- The answer notes when two methods should be combined.

### Exercise 6. Critique a limitations paragraph (critique, 45 minutes)

**Claim.** "This observational study cannot establish causality because of residual confounding. Nevertheless, the association is strong and statistically significant."

**Deliverable.** A 400-word critique with a quantitative alternative.

**Acceptance criteria.**

- The critique identifies the missing quantification.
- It proposes an E-value or a bias analysis.
- It distinguishes the strength of the association from its robustness.
- It states what the decision-maker would need to know.

### Exercise 7. Design a Monte Carlo bias analysis (design, 60 minutes)

**Scenario.** A claims study of a drug and a rare outcome. Two biases are suspected: unmeasured smoking and outcome misclassification.

**Deliverable.** A design with the bias parameters, their distributions, the simulation procedure, and the reporting table.

**Acceptance criteria.**

- Each parameter has a distribution and a justification.
- The simulation draws and recomputes the estimate.
- The output includes a median, an interval, and a probability of crossing the threshold.
- The distributions are reported.

### Exercise 8. Write a sensitivity analysis plan (design, 90 minutes)

**Scenario.** A 5,000-patient observational comparison of two anticoagulants, with claims data and a primary hazard ratio.

**Deliverable.** A two-page plan with the assumptions, the methods, the parameter ranges, the negative controls, the tipping points, and the reporting table.

**Acceptance criteria.**

- Each assumption is paired with a method.
- The ranges are justified.
- At least one negative control is specified.
- The report table is included.
- The decision threshold is stated.
- The plan distinguishes an E-value from a bias analysis.
- The negative control is justified by the DAG.
- The report includes a judgment rule.
- The plan states what would change the recommendation.

## 13. Self-check answers

**Answer 1 (Exercise 1 E-values).**

```text
RR = 1.50: E = 1.50 + sqrt(1.50 × 0.50) = 1.50 + 0.8660 = 2.3660
RR = 0.70: invert to 1/0.70 = 1.4286
           E = 1.4286 + sqrt(1.4286 × 0.4286) = 1.4286 + 0.7824 = 2.2110
OR = 1.60 at 5% control risk: RR ≈ sqrt(1.60) = 1.2649
           E = 1.2649 + sqrt(1.2649 × 0.2649) = 1.2649 + 0.5787 = 1.8436
```

The protective effect has the same E-value as its inverted risk ratio. The odds ratio with a rare outcome gives an E-value of about 1.84, lower than the risk ratio of 1.60 would suggest if it were treated as a risk ratio (which would give 2.20). The conversion matters.

| Measure | Value | E-value |
|---|---:|---:|
| RR | 1.50 | 2.37 |
| RR | 0.70 | 2.21 |
| OR, rare outcome | 1.60 | 1.84 |

A reader who compares the OR and RR rows without the conversion would overstate the robustness of the odds-ratio result.

**Answer 2 (Exercise 2 bias adjustment).**

```text
Bias factor = (1 + 0.45 × 1.5) / (1 + 0.20 × 1.5)
            = (1 + 0.675) / (1 + 0.30)
            = 1.675 / 1.30 = 1.2885
Adjusted RR = 2.10 / 1.2885 = 1.6298
Adjusted 95% CI = (1.40 / 1.2885, 3.15 / 1.2885) = (1.0865, 2.4447)
```

The confounder reduces the estimate from 2.10 to 1.63, and the interval still excludes 1. The confounder does not explain the association. The E-value of the adjusted estimate is 1.63 + sqrt(1.63 × 0.63) = 1.63 + 1.013 = 2.64, which is the strength of a second confounder that would be needed.

**Answer 3 (Exercise 4 tipping points).**

```text
Shift per unit delta = 0.20
Point estimate: 2.5 - 0.20 × delta = 1.0 -> delta = 7.5
Lower confidence limit: 2.5 - 1.96 × 1.0 - 0.20 × delta = 1.0
                        0.54 - 0.20 × delta = 1.0 -> delta = -2.3
```

The point estimate reaches the threshold at a delta of 7.5 points, which is large relative to the standard error. The lower confidence limit is already below the threshold before any delta, so the confidence-limit tipping point is negative (-2.3): the interval includes the threshold even under MAR. The conclusion about the threshold is not robust, but not because of MNAR; it is because the interval is wide.

**Answer 4 (Exercise 5 method choice).**

| Situation | Method | Output | Reason |
|---|---|---|---|
| No specific confounder | E-value | Minimum confounder strength | No parameters to specify |
| Health literacy suspected, prevalence known | Bias analysis | Adjusted estimate and interval | A specific scenario can be defended |
| Claims outcome, pharmacy exposure | Negative control plus misclassification bias analysis | A test and a corrected estimate | Two biases are plausible |
| Large apparent benefit, skeptical team | E-value and negative control | Robustness summary and a bias test | Both are cheap and informative |
| Cost-per-QALY threshold | Tipping point against the threshold | The bias that changes the decision | The decision is threshold-relative |

**Answer 5 (Exercise 6 critique).**

A strong critique:

- "Cannot establish causality" is a conclusion, not a quantification. It does not tell the reader how much confounding would matter.
- "Strong and statistically significant" describes the estimate, not its robustness. A strong association can still be explained by a strong confounder.
- The alternative is an E-value: an unmeasured confounder would need a risk ratio of X with both treatment and outcome to explain the estimate.
- The decision-maker needs the threshold and the probability that the true effect is below it.
- The critique should propose a negative control if a credible one exists.

**Answer 6 (Exercise 3 negative controls).**

Candidate controls for healthier-user selection:

- **Cancer screening uptake:** cannot be caused by the diabetes drug, but shares the healthy-user mechanism.
- **Influenza vaccination:** same logic; a positive association suggests selection.

A positive result (drug A associated with more screening) would suggest that drug A users are healthier or more engaged, which would bias the primary estimate. A null result is consistent with a valid design but does not prove it.

**Answer 7 (Exercise 7 design outline).**

A strong design:

- Specifies smoking prevalence by exposure group with a distribution from survey data.
- Specifies the smoking-outcome association from published relative risks.
- Specifies outcome sensitivity and specificity with distributions from a validation study.
- Draws all parameters, recomputes the adjusted hazard ratio, and repeats 10,000 times.
- Reports the median, the 95% interval, and the probability that the adjusted estimate is below the decision threshold.
- Reports the distributions and a sensitivity analysis with alternative distributions.

**Answer 8 (Exercise 8 rubric).**

A strong plan:

- Names no unmeasured confounding, selection, and outcome misclassification as the threats.
- Uses an E-value for the first, a negative control for the second, and a bias analysis for the third.
- Pre-specifies the parameter ranges.
- States the decision threshold (for example, a hazard ratio of 0.80 for cost neutrality).
- Includes a reporting table with the assumption, method, result, and judgment.
- Commits to a conclusion rule: robust, fragile, or unresolved.

A strong plan also states the order of analyses: the primary estimate first, then the E-value, then the negative control, then the bias analysis. The order matters because the earlier analyses frame the later ones.

A weak plan lists methods without linking them to assumptions, or proposes a sensitivity analysis only if the primary result is significant.

## 14. Interview questions

**1. What is a sensitivity analysis?**

A strong answer says it is an analysis under a different assumption to test how much the conclusion depends on it, and distinguishes it from a robustness check. A weak answer says it is "running the analysis again."

**2. What is an E-value?**

A strong answer defines the minimum confounder strength needed to explain away the association and gives the formula. A weak answer says it is "a measure of robustness" without the definition.

**3. Why report the E-value for the confidence limit?**

A strong answer says the lower limit is closer to the null and therefore easier to explain away; the E-value for the limit is the relevant one for the decision. A weak answer reports only the point estimate.

**4. What is a negative control, and what does it prove?**

A strong answer defines it and says it detects bias but does not correct or prove validity. A weak answer says a null control proves the study is valid.

**5. How do you choose the plausible range for a bias parameter?**

A strong answer uses prior studies, clinical knowledge, and external data, and pre-specifies the range. A weak answer chooses the range that preserves significance.

**6. What is a tipping point?**

A strong answer defines the parameter value at which the conclusion changes and says it must be interpreted in clinical units. A weak answer calls it "the point where it stops being significant."

**7. Why is an odds ratio not a risk ratio for the E-value?**

A strong answer explains that the OR overstates the RR when the outcome is common and gives the conversion formula. A weak answer says they are "close enough."

**8. What is the difference between an E-value and a bias analysis?**

A strong answer says the E-value summarizes the required confounding without specifying it; the bias analysis specifies a confounder and corrects the estimate. A weak answer treats them as alternatives.

**9. When would you use a Monte Carlo bias analysis?**

A strong answer says when several biases act together or when the bias parameters are uncertain, and notes that the distributions are assumptions. A weak answer says "when the data are complex."

**10. What does a positive negative-control result mean?**

A strong answer says it is evidence of residual bias in the design, but not a measure of the bias in the primary estimate. A weak answer says it invalidates the study.

**11. Can a sensitivity analysis prove a causal effect?**

A strong answer says no; it quantifies robustness, not identification. A weak answer says a high E-value proves causation.

**12. What would make you distrust a sensitivity analysis?**

A strong answer lists a post hoc range, an unconverted odds ratio, a negative control with different confounders, and a conclusion without a threshold. A weak answer says "a small E-value."

A follow-up is "what would make you trust one?" The strong answer says a pre-specified range, a method matched to the threat, a reported lower-limit E-value, and a negative control that shares the confounding structure. Trust comes from the design of the sensitivity analysis, not from the size of its output.

## 15. Key terms

- **Bias-adjusted interval:** an interval that includes bias-parameter uncertainty.
- **Bias analysis:** a quantitative adjustment for a specific bias.
- **Bias factor:** the multiplier relating observed and true effects.
- **Confounder-outcome association:** the effect of the confounder on the outcome.
- **Confounder prevalence:** the proportion with the confounder by exposure group.
- **Decision threshold:** the effect at which the decision changes.
- **Delta adjustment:** shifting imputed values by a fixed amount.
- **Direction of bias:** whether the bias moves the estimate toward or away from the null.
- **E-value:** the minimum confounder strength needed to explain away an association.
- **Fragility:** the ease with which the conclusion changes.
- **Healthy-user bias:** healthier people are more likely to receive treatment.
- **Magnitude of bias:** how far the estimate moves.
- **Misclassification:** error in measuring exposure, outcome, or confounder.
- **Monte Carlo bias analysis:** propagating bias-parameter uncertainty by simulation.
- **Negative control exposure:** an exposure that cannot affect the outcome.
- **Negative control outcome:** an outcome the treatment cannot affect through the mechanism.
- **Plausible range:** the range of sensitivity parameter values supported by knowledge.
- **Probabilistic bias analysis:** a bias analysis with distributions on the parameters.
- **Quantitative bias analysis:** a numerical correction for bias.
- **Residual confounding:** confounding that remains after adjustment.
- **Robustness:** the stability of the conclusion across assumptions.
- **Selection bias:** bias from conditioning on a selection variable.
- **Sensitivity analysis:** an analysis under a different assumption.
- **Sensitivity parameter:** the unknown quantity that indexes the departure.
- **Simple bias analysis:** a bias analysis with fixed parameter values.
- **Tipping point:** the parameter value at which the conclusion changes.
- **Unmeasured confounding:** a common cause not in the data.
- **Quantitative reporting:** reporting the sensitivity result numerically.
- **Assumption ledger:** a list of assumptions and their tests.
- **Bias simulation:** a simulation that draws bias parameters.
- **Decision rule:** the rule that maps an estimate to an action.
- **Lower-limit E-value:** the E-value for the confidence limit closest to the null.
- **Bias parameter:** a quantity that describes the bias process.
- **Simulation study:** a study that draws parameters and recomputes the estimate.
- **Directional bias:** bias that moves the estimate in a known direction.
- **Non-differential error:** measurement error unrelated to group.
- **Differential error:** measurement error that differs by group.
- **Assumption:** a claim required for identification.
- **Selection bias analysis:** a quantitative correction for selection.
- **Misclassification bias analysis:** a correction for exposure or outcome error.
- **Bias-adjusted estimate:** the estimate after applying the bias factor.
- **Threshold-relative robustness:** robustness judged against a decision threshold.
- **Qualitative caveat:** a limitation stated without quantification.
- **Bias-adjusted result:** the conclusion after the bias analysis.
- **Sensitivity table:** a table of estimates across sensitivity parameters.
- **Unmeasured variable:** a variable not recorded in the data.
- **Robustness region:** the parameter range over which the conclusion holds.

## 16. Sources

All URLs were accessed on 25 October 2026.

1. VanderWeele, T. J., Ding, P. "Sensitivity Analysis in Observational Research: Introducing the E-Value." *Annals of Internal Medicine* 2017;167(4):268-274. https://doi.org/10.7326/M16-2607
2. Mathur, M. B., Ding, P., Riddell, C. A., VanderWeele, T. J. "Website and R package for computing E-values." *Epidemiology* 2018;29(5):e45-e47.
3. E-value calculator. https://www.evalue-calculator.com/
4. Lipsitch, M., Tchetgen Tchetgen, E., Cohen, T. "Negative Controls: A Tool for Detecting Confounding and Bias in Observational Studies." *Epidemiology* 2010;21(3):383-388. https://doi.org/10.1097/EDE.0b013e3181d61eeb
5. Jackson, L. A., Jackson, M. L., Nelson, J. C., Neuzil, K. M., Weiss, N. S. "Evidence of bias in estimates of influenza vaccine effectiveness in seniors." *International Journal of Epidemiology* 2006;35(2):337-344.
6. Lash, T. L., Fox, M. P., Fink, A. K. *Applying Quantitative Bias Analysis to Epidemiologic Data*. Springer, 2009. https://doi.org/10.1007/978-0-387-87959-8
7. Fox, M. P., MacLehose, R. F., Lash, T. L. *Applying Quantitative Bias Analysis to Epidemiologic Data*. 2nd edition. Springer, 2021. https://doi.org/10.1007/978-3-030-82673-4
8. International Council for Harmonisation. *ICH E9(R1) Addendum on Estimands and Sensitivity Analysis in Clinical Trials*. Step 4, 20 November 2019. https://database.ich.org/sites/default/files/E9-R1_Step4_Guideline_2019_1203.pdf
9. United States Food and Drug Administration. *Considerations for the Use of Real-World Data and Real-World Evidence to Support Regulatory Decision-Making for Drug and Biological Products*. August 2023. https://www.fda.gov/media/171667/download
10. STROBE Initiative. *STROBE Statement*. https://www.strobe-statement.org/
11. Wang, S. V., Schneeweiss, S., and the RCT-DUPLICATE Initiative. "Emulation of Randomized Clinical Trials With Nonrandomized Database Analyses: Results of 32 Clinical Trials." *JAMA* 2023;329(16):1376-1385. https://doi.org/10.1001/jama.2023.4221
12. Manson, J. E., Hsia, J., Johnson, K. C., et al. "Estrogen plus Progestin and the Risk of Coronary Heart Disease." *New England Journal of Medicine* 2003;349:523-534. https://doi.org/10.1056/NEJMoa030808
13. Cro, S., Morris, T. P., Kenward, M. G., Carpenter, J. R. "Sensitivity analysis for clinical trials with missing continuous outcome data using controlled multiple imputation: A practical guide." *Statistics in Medicine* 2020;39(21):2815-2842. https://doi.org/10.1002/sim.8569
14. Rothman, K. J., Greenland, S., Lash, T. L. *Modern Epidemiology*. 3rd edition. Lippincott Williams and Wilkins, 2008.
15. VanderWeele, T. J., Mathur, M. B. "Some desirable properties of the E-value." *Epidemiology* 2019;30(2):e5-e6.
16. Ding, P., VanderWeele, T. J. "Sensitivity Analysis Without Assumptions." *Epidemiology* 2016;27(3):368-377.

> **Note:** The E-value calculator citation is Mathur et al. 2018, not the 2017 methods paper. The Jackson 2006 paper is the source for the influenza vaccine negative-control example; Lipsitch et al. 2010 is the source for the framework. The quantitative bias analysis books are the primary references for the bias-adjustment formulas.

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`07_target-trial-emulation.md`](07_target-trial-emulation.md) — the design that the sensitivity analysis stress-tests.
- Causal graphs: [`05_causal-inference-and-dags.md`](05_causal-inference-and-dags.md) — the unmeasured nodes.
- Missing data: [`04_missing-data.md`](04_missing-data.md) — delta adjustment and tipping points.
- Propensity scores: [`06_propensity-scores-and-weighting.md`](06_propensity-scores-and-weighting.md) — the balance table is not a sensitivity analysis.
- Reporting: [`10_reporting-statistical-results.md`](10_reporting-statistical-results.md) — reporting the sensitivity result.
- Survey file: [`../../04_statistics_causal_inference.md`](../../course/04_statistics_causal_inference.md)
- Labs: [`../../labs/`](../../labs/) — Lab 4.5 covers MNAR sensitivity.
- Foundations: [`../../01A_foundations_assumptions_and_units.md`](../../course/01A_foundations_assumptions_and_units.md) — assumptions ledger.
- Estimation: [`01_estimation-and-inference.md`](01_estimation-and-inference.md) — intervals and effect scales.
- Target trial emulation: [`07_target-trial-emulation.md`](07_target-trial-emulation.md) — the design context.
- Module 18: [`../../18_quality_evaluation_reporting.md`](../../course/18_quality_evaluation_reporting.md) — evaluation standards.

### 6.7 Vary one assumption: the outcome is common and the measure is an odds ratio

Suppose the reported measure is an odds ratio of 1.75, and the control risk is 30%, not rare. Converting to a risk ratio:

```text
RR = OR / (1 - p0 + p0 × OR)
   = 1.75 / (0.70 + 0.30 × 1.75)
   = 1.75 / 1.225
   = 1.4286
E = 1.4286 + sqrt(1.4286 × 0.4286)
  = 1.4286 + 0.7824
  = 2.2110
```

The E-value falls from 2.90 to 2.21. The conclusion is more fragile than the odds-ratio version suggested, because the odds ratio overstates the risk ratio when the outcome is common. This is the material variation: the same odds ratio, interpreted on the correct scale, requires a weaker confounder to explain away.

### 6.8 What to report

Report the primary estimate (RR 1.75, 95% CI 1.20 to 2.55) with the E-value (2.90 for the estimate, 1.69 for the lower limit), the two bias scenarios (adjusted RR 1.50 and 1.00), the negative control (RR 1.05, 95% CI 0.92 to 1.20), and the tipping point (a confounder with RR 3.0 and a 45-point prevalence difference).

The judgment is fragile at the decision threshold of 1.30: scenario A leaves the estimate above the threshold, scenario B nullifies it, and the plausible range of health-literacy and social-support associations spans both. The negative control is reassuring but not decisive.

The honest conclusion: "The adjusted association was 1.75 (95% CI 1.20 to 2.55). An unmeasured confounder would need a risk ratio of about 2.9 with both enrollment and hospitalization to explain the point estimate, and about 1.7 to explain the lower confidence limit. A plausible confounder with a 3.0-fold association and a 45-point prevalence difference would nullify the result. The negative control outcome showed no association. The estimate is suggestive but not robust enough to support a coverage decision without a stronger design."
