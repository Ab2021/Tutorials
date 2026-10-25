# Missing Data

> Module 04 — Statistics and Causal Inference
> Topic 04 of 10
> Estimated study time: 3.5 hours
> Prerequisites: 01, 02
> Version 2026.3

---

## 1. Why this topic exists

Missing data is the normal condition of biomedical analytics, not an exception. Laboratory values are missing because a clinician did not order the test. Insurance claims are missing a diagnosis because the visit happened outside the network. A trial outcome is missing because the patient withdrew. A wearable signal is missing because the device was not worn. The analysis must decide what those absences mean, and that decision changes the estimate.

The default in most analytics code is complete-case analysis: drop every row with a missing value. The default is not neutral. It restricts the sample to patients who were measured, and measurement is often a function of health, access, and treatment. Three consequences follow:

- **Bias.** If sicker patients are less likely to have a lab drawn, or more likely to have it drawn, the complete-case sample is not representative. The direction of the bias depends on the mechanism and cannot be signed from the data alone.
- **Loss of power.** A dataset with 30% missingness across ten variables can lose more than half its rows under complete-case analysis, even when no single variable is missing that often.
- **Irreproducibility.** Different software handles missing values differently by default. A result that depends on a silent deletion rule cannot be reproduced by a reviewer with different settings.

The methods that replace deletion are not interchangeable either. Mean imputation shrinks variance. Last observation carried forward (LOCF) assumes the outcome is flat after dropout and is discouraged by the European Medicines Agency (EMA) for most confirmatory settings. A single regression imputation understates uncertainty because it treats imputed values as observed. A missing-indicator method can encode the missingness mechanism as a covariate and produce impossible interpretations. Multiple imputation (MI) addresses several of these problems, but only under assumptions, and the assumptions must be stated.

Regulators treat the issue as part of the estimand, not as a data-cleaning step. The EMA guideline "Guideline on Missing Data in Confirmatory Clinical Trials" (EMA/CPMP/EWP/1776/99 Rev. 1), adopted on 24 June 2010 and effective 1 January 2011, states that complete-case analysis is not acceptable as a primary analysis, that the method and sensitivity analyses must be pre-specified, and that the reasons for missingness must be reported. The International Council for Harmonisation (ICH) E9(R1) addendum, adopted in 2019, distinguishes an intercurrent event (such as treatment discontinuation) from the missing data that may follow it. The distinction matters: the estimand defines how the event is handled, and the missing-data method defines how the unobserved values are handled. They are separate decisions.

This topic builds the vocabulary and the mechanics: the three mechanisms, the imputation model, Rubin's rules, the diagnostics that reveal a bad imputation, and the sensitivity analyses that test the assumption the primary analysis cannot test.

> **Note:** "Missing at random" does not mean the missingness is random. It means the probability of missingness depends only on observed data, not on the unobserved value itself. That is an assumption about a counterfactual quantity, and it cannot be verified from the observed data.

## 2. Learning outcomes

By the end of this topic you can:

1. **Classify** missingness as missing completely at random (MCAR), missing at random (MAR), or missing not at random (MNAR), and **explain** why the classification is an assumption rather than a test result.
2. **Distinguish** structural missingness from sporadic missingness, and **decide** whether a missing value should be imputed at all.
3. **Describe** a missingness pattern by variable, group, and time, and **identify** patterns that suggest informative missingness.
4. **Build** an imputation model that includes the outcome, auxiliary variables, and interactions, and **explain** congeniality.
5. **Apply** Rubin's rules to pool estimates and standard errors across imputations, and **interpret** the fraction of missing information (FMI).
6. **Choose** the number of imputations from the FMI and the planned sensitivity analyses.
7. **Design** a delta-adjusted or pattern-mixture sensitivity analysis and **report** a tipping point.
8. **Critique** a complete-case, LOCF, or single-imputation analysis and **state** what would be required to replace it.

Outcome 5 maps to Exercise 3 and the worked example. Outcome 7 maps to Exercise 5 and self-check item 6.

## 3. The core mental model

Missing data is a relationship between three objects: the complete data you wish you had, the missingness process that removed some of it, and the observed data you actually have.

```text
  COMPLETE DATA
  (Y_mis, Y_obs, X)
        |
        |  missingness process M: which values are observed?
        |  P(M | Y_mis, Y_obs, X)
        v
  OBSERVED DATA  (Y_obs, X, M)
        |
        |  assumption about P(M | ...)
        v
  MECHANISM
  MCAR: P(M | Y_mis, Y_obs, X) = P(M)
  MAR:  P(M | Y_mis, Y_obs, X) = P(M | Y_obs, X)
  MNAR: P(M | Y_mis, Y_obs, X) depends on Y_mis
        |
        |  imputation model + draws
        v
  M COMPLETED DATASETS
        |
        |  analyze each, pool with Rubin's rules
        v
  ESTIMATE + INTERVAL + FMI
        |
        |  sensitivity: delta adjustment, pattern mixture, tipping point
        v
  DECISION
```

### 3.1 The mechanism is about the missing value, not the variable

The mechanism is defined by the conditional probability of being missing given the full data. It is not a property of a variable in isolation. A lab value can be MCAR in one cohort and MNAR in another, because the ordering of who gets tested differs.

- **MCAR:** missingness is independent of both observed and unobserved data. Complete-case analysis is unbiased, though inefficient. MCAR is rare in clinical data and is usually a simplifying assumption, not a finding.
- **MAR:** missingness depends on observed data but not on the unobserved value, given those observed data. Complete-case analysis is generally biased; MI and inverse probability weighting can be unbiased if the imputation or weighting model is correct.
- **MNAR:** missingness depends on the unobserved value itself, even after conditioning on observed data. No method can identify the estimand from the observed data alone without an additional assumption about the missingness process. Sensitivity analysis is required.

The three categories are not a ladder of quality. They are different assumptions, and the analysis must state which one it relies on and what happens if it is wrong.

### 3.2 MAR is conditional, not absolute

A common mistake is to check whether missingness is associated with an observed variable and conclude "MAR" if it is, or "MCAR" if it is not. That reasoning is backwards. An association with an observed variable is exactly what MAR allows. The question MAR asks is whether, after conditioning on all observed variables in the imputation model, the missing value itself still predicts missingness. That residual dependence is unobservable.

This is why the imputation model must be rich. If a variable that predicts missingness is omitted from the model, the data are MNAR with respect to the model even if they would be MAR with respect to a richer model. The practical instruction is to include all variables associated with missingness and with the outcome, plus the outcome itself.

### 3.3 Multiple imputation has two sources of variance

A single imputed dataset produces a standard error that treats the imputed values as known. That understates uncertainty. MI separates the variance into:

- **Within-imputation variance (U):** the average variance of the estimate across the completed datasets, as if each dataset were real.
- **Between-imputation variance (B):** the variance of the estimate across the completed datasets, which reflects the uncertainty in the imputed values.

Rubin's total variance is

```text
T = U + (1 + 1/m) B
```

where m is the number of imputations. The factor (1 + 1/m) accounts for the fact that B is estimated from a finite number of imputations. As m grows, the factor approaches 1 and the Monte Carlo error of the pooled estimate shrinks.

The **fraction of missing information** is

```text
FMI = (1 + 1/m) B / T
```

It estimates the proportion of the total variance due to missing data. A small FMI means the missing data contribute little to the uncertainty; a large FMI means the result depends heavily on the imputation model. FMI is not the same as the proportion of missing values. A variable with 40% missingness can have a small FMI if the imputation model predicts it well, and a variable with 10% missingness can have a large FMI if it is poorly predicted.

### 3.4 Three ways to think about MNAR

There is no single MNAR model. Three families are common:

1. **Selection models** factor the joint distribution into the outcome model and a missingness model that depends on the outcome. They are natural to describe but sensitive to the missingness model's functional form.
2. **Pattern-mixture models** stratify by missingness pattern and model the outcome within each pattern. They are easy to fit and require an assumption about the unobserved pattern, often that it differs from the observed pattern by a fixed amount.
3. **Delta adjustment** imputes under MAR and then shifts the imputed values by a fixed amount δ, or scales them, before analysis. It is transparent and easy to communicate. The tipping point is the value of δ at which the conclusion changes.

Delta adjustment is the most common sensitivity analysis in industry because it is easy to explain to a clinical team: "we assumed the missing patients did 3 points worse than the model predicts; at 4.5 points the result is no longer significant."

### 3.5 Structural missingness is not a data quality problem

A value can be missing because the question does not apply. A pregnancy test is structurally missing for a male patient. A pediatric dose is structurally missing for an adult. A follow-up visit is structurally missing for a patient who died. Imputing a structural missing value creates a variable that cannot exist.

The rule is: distinguish "not applicable" from "unknown." Not applicable values should be coded as a separate category or excluded from the variable definition, not imputed. Unknown values are candidates for imputation.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or scale |
|---|---|---|---|
| Missing completely at random (MCAR) | Missingness independent of observed and unobserved data | Called "random missingness" without the condition | Assumption |
| Missing at random (MAR) | Missingness depends on observed data but not on the unobserved value given them | Tested with a t-test and declared true | Assumption |
| Missing not at random (MNAR) | Missingness depends on the unobserved value | Treated as a failure rather than a scenario | Assumption |
| Missingness indicator | A variable marking whether a value is observed | Included as a covariate without interpretation | 0/1 |
| Structural missingness | A value that does not apply | Imputed as if unknown | Category |
| Sporadic missingness | An isolated missing value in an otherwise observed record | Ignored | Count |
| Monotone pattern | Once a variable is missing, all later variables are missing | Assumed without checking | Pattern |
| Non-monotone pattern | Missingness can occur and then reverse | Handled by a monotone-only method | Pattern |
| Complete-case analysis | Analysis restricted to rows with no missing values | Called "the analysis" without a sensitivity analysis | Sample |
| Available-case analysis | Each analysis uses all rows observed for its variables | Confused with complete-case | Sample |
| Single imputation | One replacement value per missing value | Treated as observed data | Value |
| Mean imputation | Replacing missing values with the variable mean | Used for inference | Value |
| Last observation carried forward (LOCF) | Replacing a missing follow-up with the last observed value | Assumed conservative | Value |
| Baseline observation carried forward (BOCF) | Replacing a missing follow-up with baseline | Assumed conservative | Value |
| Multiple imputation (MI) | Repeated imputation and pooling | Described as "filling in the blanks" | Method |
| Imputation model | The model used to predict missing values | Omits the outcome or auxiliary variables | Model |
| Congeniality | Compatibility between the imputation model and the analysis model | Ignored when the analysis has interactions | Assumption |
| Rubin's rules | The formulas for pooling estimates and variances across imputations | Applied to a single imputation | Method |
| Within-imputation variance (U) | Average variance across completed datasets | Confused with total variance | Variance |
| Between-imputation variance (B) | Variance of the estimate across completed datasets | Omitted | Variance |
| Total variance (T) | U + (1 + 1/m)B | Computed as U alone | Variance |
| Fraction of missing information (FMI) | The proportion of total variance due to missingness | Replaced by the proportion of missing values | Proportion, 0 to 1 |
| Relative efficiency | The precision of m imputations relative to infinite m | Ignored when m is small | Proportion |
| Auxiliary variable | A variable associated with missingness or with the outcome, included in the imputation model | Omitted because it is not in the analysis model | Variable |
| Delta adjustment | Shifting imputed values by a fixed amount to model MNAR | Applied without a rationale for delta | Outcome units |
| Pattern-mixture model | A model that stratifies by missingness pattern | Confused with a selection model | Model |
| Selection model | A model for the outcome and the missingness process jointly | Sensitive to functional form | Model |
| Tipping point | The value of a sensitivity parameter at which the conclusion changes | Reported without clinical context | Sensitivity parameter units |
| Inverse probability weighting (IPW) | Weighting observed cases by the inverse probability of being observed | Used without checking weight extremes | Weight |
| Missing indicator method | Including the missingness indicator as a covariate | Interpreted as a group effect | Model term |
| Imputation of the outcome | Including the outcome in the imputation model | Considered leakage in prediction but required for effect estimation | Model choice |
| Uncongenial imputation | The imputation model is incompatible with the analysis model | Creates bias toward the null or away from it | Model mismatch |
| Complete-data estimator | The estimator that would be used if no data were missing | Not stated | Estimator |
| Nonresponse weight | A weight that adjusts for the probability of responding | Confused with a sampling weight | Weight |

Two conventions in this topic:

- Missingness is reported as a count and a percentage by variable, group, and time: "eGFR was missing for 84 of 412 program patients (20.4%) and 61 of 388 usual-care patients (15.7%)."
- An MI result is reported with m, the FMI, and the software: "MI with m = 20, FMI = 0.12, using predictive mean matching."

> **Note:** A missing-indicator method is not a substitute for MI. It can produce a coefficient for "missingness" that is a mixture of prognosis, access, and data-entry practice, and it assumes that the missing group has a common offset. It is acceptable for some prediction tasks and is not recommended for effect estimation.

## 5. The mechanics, step by step

The procedure moves from the estimand to a pooled estimate and then to a sensitivity analysis. Steps 1 to 4 define what is missing and why it matters. Steps 5 to 9 impute and pool. Steps 10 to 13 diagnose, stress-test, and report.

### Step 1. Define the estimand and the role of each variable

**Input:** The analysis plan, the estimand, and the variable list.

**Output:** A table classifying each variable as outcome, exposure, confounder, auxiliary, or post-treatment, with its missingness count.

**Why it exists:** The imputation model must be congenial with the analysis model. A variable that is a confounder in the analysis is a predictor in the imputation model. A variable that is a mediator must not be imputed in a way that recovers treatment effects, and a post-treatment variable should not be used to impute a baseline confounder.

| Variable role | Include in imputation model? | Reason |
|---|---|---|
| Outcome | Yes, even if not in the analysis of interest | Preserves the outcome-conditional distribution |
| Exposure | Yes | Preserves the exposure-outcome association |
| Confounder | Yes, with the functional form used in the analysis | Avoids uncongeniality |
| Auxiliary variable | Yes, if it predicts missingness or the outcome | Improves prediction and MAR plausibility |
| Mediator | Yes for a total-effect analysis, with caution | Including it can reintroduce part of the effect |
| Post-treatment variable | Generally no | May carry treatment effects and bias the imputation |
| Identifier or site | Yes, as a fixed effect if the analysis stratifies | Preserves clustering |

### Step 2. Separate structural from sporadic missingness

**Input:** The data dictionary and the clinical logic of each variable.

**Output:** A list of values that are "not applicable" and a list that are "unknown."

**Why it exists:** Imputing a structural missing value creates an impossible patient. The distinction is a data-modeling decision, not a statistical one.

Examples:

- A pregnancy test for a male patient is structural. Code as "not applicable," not missing.
- A pediatric formulation dose for an adult is structural.
- A 12-month outcome for a patient who died at 3 months is structural if the estimand is the 12-month outcome among survivors, and a competing event if the estimand is the 12-month outcome in the whole cohort. The choice is the estimand's, not the imputer's.
- A lab value that was not ordered is unknown and a candidate for imputation.
- A questionnaire item skipped by a respondent is unknown, unless the skip pattern is designed.

### Step 3. Describe the amount, pattern, and timing

**Input:** The analysis dataset.

**Output:** A missingness table by variable, group, and time, plus a pattern plot.

**Why it exists:** The pattern often identifies the mechanism. Missingness that appears only after a treatment switch suggests a post-treatment process. Missingness that differs by arm suggests differential measurement. Missingness concentrated at a specific visit suggests a workflow cause.

| Description | What to report | What it can reveal |
|---|---|---|
| Amount | Count and percentage per variable | Whether complete-case analysis is viable |
| Pattern | Monotone or non-monotone; number of distinct patterns | Which imputation methods are available |
| Timing | Visit or date of missingness | Dropout versus intermittent omission |
| By group | Missingness by treatment arm | Differential missingness |
| By outcome | Missingness by observed outcome category | Informative missingness |
| By site | Missingness by hospital or clinic | Workflow or coding differences |

**Monotone versus non-monotone.** A monotone pattern means that once a variable is missing, all later variables are also missing, as in a longitudinal study with dropout. A non-monotone pattern means missingness can occur at any visit and later reverse. Monotone patterns allow sequential imputation; non-monotone patterns require a method such as fully conditional specification or a multivariate normal model.

**Reasons, not just patterns.** The reason for missingness is more informative than the pattern. A visit missed because the patient moved is different from a visit missed because the patient was too sick to attend, even if both appear as a missing 12-month value. Where the data source records a reason, tabulate it by arm. Where it does not, the clinical team can often reconstruct the main reasons from the workflow. That narrative is the basis for the mechanism assumption in Step 4, and it is more defensible than a test.

### Step 4. State the mechanism assumption and its rationale

**Input:** The missingness description, clinical knowledge, and a causal diagram of the missingness process.

**Output:** A stated assumption (MCAR, MAR, or MNAR), the variables that make MAR plausible, and the sensitivity analysis that will test it.

**Why it exists:** The mechanism cannot be estimated from the observed data. It must be assumed, justified, and tested indirectly. The justification is a narrative about why values are missing, not a p-value.

| Mechanism | Plausible when | Primary method | Required sensitivity |
|---|---|---|---|
| MCAR | Missingness is a random sampling or data-entry loss, unrelated to any variable | Complete case or MI | None beyond a comparison with MI |
| MAR | Missingness is explained by observed variables, including the outcome and auxiliary variables | MI, MMRM, or IPW | Delta adjustment |
| MNAR | Missingness depends on the unobserved value, such as dropout for lack of efficacy | Pattern mixture or selection model | Tipping point, with a plausible delta range |

**A useful test, and its limit.** Little's MCAR test compares the observed means across missingness patterns. A significant result rejects MCAR; a non-significant result does not establish MCAR, because the test has low power and only detects mean differences. It is a screening tool, not a mechanism classifier.

The test is also sensitive to the variables included. Adding a variable with many missing values can change the result without changing the underlying process. Report the test only alongside the missingness table and the clinical narrative, never as the sole justification for the mechanism assumption. A test that rejects MCAR is informative; a test that fails to reject it is not evidence for MCAR.

The practical rule is to assume MAR as the working assumption for any dataset with more than trivial missingness, justify it with the clinical narrative and the observed associations, and test the consequence with a sensitivity analysis. MCAR is a special case that should be demonstrated, not assumed.

### Step 5. Choose the primary analysis method

**Input:** The estimand, the missingness pattern, and the mechanism assumption.

**Output:** A named primary method with its assumptions and its complete-data estimator.

**Why it exists:** The method determines the assumption and the population; the complete-data estimator must be named so the pooling is interpretable.

| Situation | Primary method | Complete-data estimator | Assumption |
|---|---|---|---|
| Longitudinal trial with continuous outcome, MAR | Mixed model for repeated measures (MMRM) | Treatment contrast at the final visit | MAR; correctly specified covariance |
| Cross-sectional or non-longitudinal, MAR | MI | The planned regression or summary | MAR; congenial imputation model |
| Missingness in a confounder, MAR | MI for the covariate | The planned effect model | MAR; outcome included |
| Missing outcome in a trial, MNAR plausible | MI with delta adjustment as primary or sensitivity | The planned contrast | Delta assumption |
| Missingness in a survey with known sampling | IPW or calibration | Weighted estimator | Correct response model |
| Structural missingness | No imputation; recode | Category or exclusion | Not applicable |

**Complete-case analysis** is acceptable as a primary analysis only when MCAR is plausible or when the missingness is so small that the bias is negligible relative to the decision. It is always worth a sensitivity analysis under MI.

**LOCF and BOCF** are single-imputation methods. They are not recommended as primary analyses for most confirmatory trials. They can be conservative in some settings and anti-conservative in others, and the direction cannot be assumed.

**Decision table when the pattern and the variable type interact:**

| Pattern | Continuous | Binary | Count | Time-to-event |
|---|---|---|---|---|
| Monotone dropout | MMRM or sequential MI | Sequential logistic MI | Sequential count MI | Joint model or MI of time and event |
| Non-monotone | Fully conditional specification | Fully conditional specification | Fully conditional specification | Multiple imputation with the event indicator |
| Missing confounder | MI with outcome | MI with outcome | MI with outcome | MI with outcome |
| Missing exposure | MI, then sensitivity to definition | MI, then sensitivity | MI, then sensitivity | MI or time-varying exposure |

The table does not cover every case, but it prevents the common error of applying a monotone-only method to a non-monotone pattern. That error produces imputations that are conditional on the wrong set of variables and a variance that is too small.

### Step 6. Build the imputation model

**Input:** The variable roles, the analysis model, and the available auxiliary variables.

**Output:** A specified imputation model for each variable with missing data.

**Why it exists:** The imputation model is the engine of MI. A misspecified model produces imputed values that are systematically wrong, and no pooling formula repairs it.

Rules:

- Include the outcome, the exposure, and every variable in the analysis model.
- Include auxiliary variables that predict missingness or the outcome, even if they are not in the analysis.
- Match the functional form: if the analysis has an interaction or a spline, the imputation model needs it too. This is congeniality.
- Match the variable type: logistic for binary, ordinal for ordered categories, predictive mean matching or normal for continuous.
- Respect bounds: a percentage must stay between 0 and 100, a count must be non-negative.
- Do not include variables measured after the missing value in a way that leaks future information.
- Include cluster indicators when the data are clustered.

| Variable type | Imputation method | Note |
|---|---|---|
| Continuous, roughly normal | Normal regression with draws | Check bounds and skew |
| Continuous, skewed or bounded | Predictive mean matching (PMM) | Donor values preserve the observed range |
| Binary | Logistic regression with draws | Predict probabilities, then draw |
| Ordinal | Proportional odds model | Preserve the ordering |
| Categorical | Multinomial or polytomous model | Avoid collapsing rare categories |
| Count | Poisson or negative binomial with draws | Check overdispersion |
| Time-to-event | Impute the event time and the event indicator jointly | Censoring complicates the model |
| Longitudinal | Include previous and next visits | Preserves within-person correlation |

### Step 7. Choose the number of imputations

**Input:** The expected FMI and the number of sensitivity analyses.

**Output:** A value of m and a rationale.

**Why it exists:** With m = 5, the Monte Carlo error of the pooled estimate can be material when the FMI is high. The relative efficiency of m imputations is approximately 1 / (1 + FMI/m).

| FMI | Minimum m for stable inference | Common practice |
|---|---|---|
| Under 0.10 | 20 | 20 |
| 0.10 to 0.30 | 50 | 50 to 100 |
| 0.30 to 0.50 | 100 | 100 or more |
| Above 0.50 | 200 or more | Consider whether the imputation model is adequate |

The old rule of m = 5 is obsolete for anything but a quick check. m = 20 is a reasonable default for low FMI, and m = 100 is common when the FMI is high or when a tipping-point analysis is planned.

### Step 8. Impute, check convergence, and diagnose

**Input:** The imputation model and the data.

**Output:** m completed datasets and a diagnostic report.

**Why it exists:** Imputation algorithms can fail silently. A chain that has not converged, or a model that produces impossible values, returns datasets that look complete and are wrong.

Checks:

- The algorithm converged, with trace plots that mix across chains.
- Imputed values are within the plausible range of the variable.
- The distribution of imputed values is similar to the distribution of observed values, conditional on the predictors. A large discrepancy suggests model misspecification.
- The observed and imputed values are compared on auxiliary variables not used in the imputation, if available.
- The number of imputations is adequate for the FMI.
- The completed datasets differ from each other; identical datasets indicate that the draws are not independent.

### Step 9. Analyze each dataset and pool with Rubin's rules

**Input:** m completed datasets and the complete-data estimator.

**Output:** A pooled estimate, standard error, interval, and FMI.

**Why it exists:** Pooling combines the within-imputation and between-imputation information. Analyzing one completed dataset, or averaging the estimates without the variance correction, understates uncertainty.

```text
Qbar = (1/m) sum Q_i
Ubar = (1/m) sum U_i
B    = (1/(m-1)) sum (Q_i - Qbar)^2
T    = Ubar + (1 + 1/m) B
SE   = sqrt(T)
FMI  = (1 + 1/m) B / T
```

For a 95% interval, use a t distribution with degrees of freedom based on the FMI, not 1.96. The Barnard-Rubin degrees of freedom are the standard choice; with large m and small FMI, the t multiplier is close to 1.96.

**Pooling a ratio or a nonlinear function.** Pool on the scale where the estimate is approximately normal, such as the log odds ratio, then transform back. Pooling odds ratios directly and then averaging is not equivalent and can produce an interval that is inconsistent with the estimate.

### Step 10. Report the FMI and interpret the result

**Input:** The pooled output.

**Output:** A result statement that includes the estimate, the interval, m, the FMI, and the imputation model.

**Why it exists:** The FMI tells the reader how much of the answer depends on imputed data. A result with FMI = 0.02 is robust to the imputation model in the sense that the missing data contribute little to the variance; a result with FMI = 0.40 is not.

The FMI is not a measure of bias. A low FMI does not mean the MAR assumption is correct; it means the missing data contribute little to the sampling variance under the model. Bias can be large even when the FMI is small if the missing values are systematically different.

The FMI also depends on the quantity being estimated. A mean of a variable with 25% missingness can have a small FMI, while a subgroup contrast involving that variable can have a large FMI, because the contrast uses fewer observations per parameter. Report the FMI for the primary estimand, not for the variable in general.

A useful diagnostic is to compare the FMI with the proportion of missing values. If 25% of the outcome is missing but the FMI is 0.02, the imputation model is doing most of the work, and the result depends on it. If the FMI is 0.30, the observed data are doing most of the work, and the result is fragile. Neither number is a substitute for the sensitivity analysis.

### Step 11. Run a sensitivity analysis under MNAR

**Input:** The primary MI result and a plausible range for delta.

**Output:** A table of estimates across delta values and a tipping point.

**Why it exists:** MAR cannot be verified. The sensitivity analysis shows how wrong the missing values would have to be to change the conclusion.

Delta adjustment procedure:

1. Impute under MAR to obtain a completed dataset.
2. Add delta to the imputed values of the outcome for a defined group, such as all missing, or missing in one arm, or missing after a certain visit.
3. Analyze the adjusted dataset and pool across imputations.
4. Repeat over a grid of delta values.
5. Report the delta at which the conclusion changes, and whether that delta is clinically plausible.

**Choosing the delta range.** Anchor it in clinical units: a fraction of the observed mean change, the minimal clinically important difference, or a published estimate of the difference between dropouts and completers. A tipping point of 12 points on a scale with an MCID of 3 is not a useful warning; a tipping point of 1.5 points is.

### Step 12. Handle missing covariates separately from missing outcomes

**Input:** The variable roles from Step 1.

**Output:** A statement of whether the missingness is in the outcome, the exposure, a confounder, or several.

**Why it exists:** The consequences differ. Missing outcomes affect power and the estimate of the treatment contrast. Missing confounders affect the validity of the adjustment. Missing exposure values affect the definition of the groups.

| Missing variable | Consequence | Method | Extra care |
|---|---|---|---|
| Outcome | Loss of power; bias if MNAR | MI with the outcome in the model | Delta adjustment by arm |
| Exposure | Misclassification of groups | MI or a sensitivity analysis with alternative definitions | Do not impute exposure from the outcome |
| Confounder | Residual confounding | MI with the outcome and other confounders | Compare with complete case |
| Auxiliary only | No direct effect on the estimate | Optional inclusion | Improves precision |

A missing exposure is the most dangerous case. Imputing the exposure from the outcome creates an association that is partly constructed, and the resulting effect estimate can be biased away from the null. If the exposure is missing, the preferred approach is to define the exposure from a second source, to analyze the observed exposure with a sensitivity analysis, or to treat missing exposure as a separate category with an explicit interpretation. The imputation model must not use the outcome to predict the exposure unless the analysis is explicitly a sensitivity analysis and the direction of the induced bias is discussed.

### Step 13. Report

**Input:** The primary and sensitivity results.

**Output:** A missing-data section that a reviewer can audit.

**Why it exists:** The method, the model, and the sensitivity analysis must be visible; a single sentence cannot be audited.

The reporting checklist:

1. The amount of missingness by variable, group, and time.
2. The pattern and the reasons, where known.
3. The assumed mechanism and its rationale.
4. The primary method and the complete-data estimator.
5. The imputation model, including the outcome and auxiliary variables.
6. The number of imputations and the software.
7. The pooled estimate, interval, and FMI.
8. The sensitivity analyses and the tipping point.
9. The comparison with complete-case analysis.
10. The implications for the decision.

### Edge cases that change the estimate

| Edge case | What happens | Detection | Remedy |
|---|---|---|---|
| Missing outcome only | Complete-case analysis drops patients and biases the contrast | Outcome missingness differs by arm | MI with the outcome; delta by arm |
| Missing confounder | Adjustment is incomplete | Confounder missingness related to exposure | MI including the outcome |
| Non-monotone pattern | Monotone methods cannot be used | Pattern plot | Fully conditional specification |
| Small n | Imputation model unstable; separation | Fewer than 10 events per predictor | Fewer predictors; PMM; penalization |
| Bounded variable | Normal imputation produces impossible values | Values outside the range | PMM or truncated normal |
| Clustered data | Imputation ignores within-cluster correlation | Imputed values vary less than observed | Include cluster effects |
| Missing by design | Structural missingness treated as unknown | Values missing for a whole subgroup | Recode as not applicable |
| Missing after treatment switch | Post-treatment missingness | Missingness concentrated after switch | Time-varying analysis; sensitivity |
| All values missing in a subgroup | No information to impute | A site or arm with 100% missing | Report; do not impute |

### MMRM versus MI

A mixed model for repeated measures uses all observed post-baseline values and assumes MAR. It does not impute, and it handles the within-patient correlation directly. It is often the primary analysis for a continuous longitudinal endpoint in a trial, because it is efficient and uses the partial information in patients who drop out. Its weakness is that it assumes a correctly specified covariance structure and a linear model, and it does not produce a completed dataset for other analyses.

MI is more flexible: it can be used with any complete-data estimator, including a logistic regression, a cost model, or a survival model. It requires a congenial imputation model and more computation. The two methods often give similar answers under MAR; if they disagree substantially, the imputation or covariance model is misspecified, and the discrepancy is a diagnostic.

**Inference:** the choice between them is often driven by regulatory precedent and the analysis plan, not by a statistical theorem. A trial that pre-specifies MMRM should not switch to MI after seeing the data, because the switch changes the estimator and the assumptions.

### Branch points at a glance

| If the situation is | Then the default is | Because |
|---|---|---|
| Under 5% missing, MCAR plausible | Complete case with an MI sensitivity analysis | The bias is likely small; the check is cheap |
| 5% to 40% missing, MAR plausible | MI or MMRM | Uses all observed information |
| Over 40% missing | MI plus a careful MNAR sensitivity analysis | The result depends on the model |
| Missing outcome differs by arm | MI with delta adjustment by arm | Differential missingness is the main threat |
| Missing confounder | MI with the outcome and other confounders | Complete case can leave residual confounding |
| Structural missingness | Recode as not applicable | The value cannot exist |
| Monotone pattern from dropout | Sequential MI or MMRM | The pattern permits it |
| Non-monotone pattern | Fully conditional specification | Monotone methods do not apply |
| Bounded or skewed variable | Predictive mean matching | Donor values preserve the range |
| Clustered data | Imputation with cluster indicators | Preserves within-cluster correlation |

The branch table is a starting point. The estimand decides whether death is a competing event, a structural missing value, or an outcome; the missing-data method follows from that decision, not the reverse.

## 6. Worked example

### 6.1 The question and the estimand

A clinic reports the mean 12-week change in a quality-of-life score for 12 patients. The score is continuous, and higher values indicate improvement. Three patients are missing the 12-week score. The estimand is the mean 12-week change in the clinic population, with a secondary interest in whether the mean exceeds a target of 8 points.

The baseline score is available for all 12 patients and is strongly associated with the change score. It is the natural auxiliary variable for imputation.

### 6.2 The data

| Patient | Baseline score | 12-week change |
|---:|---:|---:|
| 1 | 50 | 2 |
| 2 | 55 | 7 |
| 3 | 60 | 3 |
| 4 | 65 | missing |
| 5 | 70 | 8 |
| 6 | 75 | 6 |
| 7 | 80 | 13 |
| 8 | 85 | missing |
| 9 | 90 | 11 |
| 10 | 95 | 16 |
| 11 | 100 | missing |
| 12 | 105 | 15 |

The missingness is 3 of 12 (25.0%). The three missing patients have baseline scores of 65, 85, and 100, which span the observed range. There is no obvious reason from the baseline distribution alone to suspect MNAR, but the assumption cannot be tested.

### 6.3 Complete-case analysis

```text
Observed n = 9
Mean = (2 + 7 + 3 + 8 + 6 + 13 + 11 + 16 + 15) / 9 = 81 / 9 = 9.0000
SD = 5.0498
SE = 5.0498 / sqrt(9) = 1.6833
95% CI = 9.0000 +/- 1.96 × 1.6833 = (5.7008, 12.2992)
```

The complete-case estimate is 9.00 points, above the target of 8. The interval includes values well below 8 and well above 12.

### 6.4 The imputation model

Regress the observed change on baseline score:

```text
n = 9
mean baseline = 75.5556
mean change = 9.0000
Sxx = sum (x - mean x)^2 = 2555.5556
Sxy = sum (x - mean x)(y - mean y) = 615.0000
slope b = 615.0000 / 2555.5556 = 0.2409
intercept a = 9.0000 - 0.2409 × 75.5556 = -9.2047
residual SD = 2.3952
```

The fitted model is

```text
change = -9.2047 + 0.2409 × baseline
```

Predicted values for the three missing patients:

```text
baseline 65:  -9.2047 + 0.2409 × 65  = 6.4567
baseline 85:  -9.2047 + 0.2409 × 85  = 11.2756
baseline 105: -9.2047 + 0.2409 × 105 = 16.0945
```

A deterministic regression imputation would fill these values and treat them as observed. That is single imputation: it ignores the residual variation around the line and understates the variance. MI draws a value from the predictive distribution, which here is normal with mean equal to the fitted value and standard deviation 2.3952.

### 6.5 Three completed datasets by hand

For a hand calculation, hold the imputation model fixed and use three sets of standard normal draws, one set per completed dataset. In a real analysis the parameters would also be redrawn, which Section 7 does.

| Dataset | Draws for patients 4, 8, 11 | Imputed values | Mean | Variance | SE of mean |
|---:|---|---|---:|---:|---:|
| 1 | -1.0, 0.2, 0.8 | 4.0616, 11.7546, 18.0106 | 9.4685 | 26.7522 | 1.4931 |
| 2 | 0.7, -0.5, 0.2 | 8.1330, 10.0779, 16.5736 | 9.5483 | 22.0788 | 1.3564 |
| 3 | 1.1, 0.6, -0.7 | 9.0908, 12.7127, 14.4178 | 9.6681 | 20.9261 | 1.3205 |

The imputed values differ across datasets because each draw represents a plausible value under the model, not the best guess.

### 6.6 Pooling with Rubin's rules

```text
Qbar = (9.4685 + 9.5483 + 9.6681) / 3 = 28.6849 / 3 = 9.5616
Ubar = (1.4931^2 + 1.3564^2 + 1.3205^2) / 3
     = (2.2293 + 1.8398 + 1.7437) / 3 = 5.8128 / 3 = 1.9376
B    = [(9.4685 - 9.5616)^2 + (9.5483 - 9.5616)^2 + (9.6681 - 9.5616)^2] / 2
     = [0.008668 + 0.000177 + 0.011342] / 2 = 0.010093
T    = 1.9376 + (1 + 1/3) × 0.010093 = 1.9376 + 0.013457 = 1.9511
SE   = sqrt(1.9511) = 1.3968
95% CI = 9.5616 +/- 1.96 × 1.3968 = (6.8238, 12.2994)
FMI  = 0.013457 / 1.9511 = 0.0069
```

> **Note:** The hand calculation holds the imputation model fixed. The script in Section 7 redraws the regression parameters for each imputation, which is the correct MI procedure, and therefore produces a slightly different estimate and a larger standard error. Both are valid; the hand version is for arithmetic, and the script version is for production.

The MI estimate is 9.5616 points, 0.5616 points above the complete-case estimate. The direction is expected: the missing patients had high baseline scores, and the imputation model predicts higher change scores for them. Complete-case analysis understates the mean because it drops the three highest-baseline patients.

The MI standard error (1.3968) is smaller than the complete-case standard error (1.6833), because MI uses 12 patients instead of 9. The interval is therefore narrower, even though the between-imputation variance adds a small amount.

### 6.7 Diagnostic checks

**Check 1. Imputed values are plausible.** The three imputed values in dataset 1 are 4.06, 11.75, and 18.01. The observed range is 2 to 16, so 18.01 is slightly above the maximum. That is not necessarily wrong under a normal model, but it warrants a check of the residual distribution. If the observed data were bounded at 20, a truncated normal or predictive mean matching would be preferable.

**Check 2. Imputed and observed distributions agree conditionally.** The mean of the nine imputed values across the three datasets is (4.0616 + 11.7546 + 18.0106 + 8.1330 + 10.0779 + 16.5736 + 9.0908 + 12.7127 + 14.4178) / 9 = 11.648. The observed mean is 9.000. The difference is expected because the missing patients have higher baseline scores. The correct check is conditional: at baseline 65, 85, and 105, the imputed values should scatter around 6.46, 11.28, and 16.09, which they do.

A marginal comparison of imputed and observed means is therefore not a valid diagnostic on its own. If the missing patients have a different covariate distribution, the imputed mean should differ from the observed mean. The diagnostic is whether the difference is explained by the covariates, which requires comparing the imputed values with the model predictions, not with the overall observed mean.

A practical way to run this check is to hold out 10% of the observed values, impute them with the same model, and compare the imputed values with the held-out truth. If the imputation model cannot recover values that were observed, it is unlikely to recover values that were not.

**Check 3. The completed datasets differ.** The three means are 9.4685, 9.5483, and 9.6681, a range of 0.20 points. If all three were identical, the draws would not be independent and the between-imputation variance would be zero.

**Check 4. FMI is consistent with the amount and predictability of missingness.** The hand calculation gives FMI = 0.0069 because the imputation model is held fixed and the three missing values are well predicted by baseline. A proper MI that also draws the regression parameters gives FMI around 0.12, because the model parameters themselves are uncertain with only nine observed patients. The gap between 0.007 and 0.12 is a diagnostic: holding the model fixed understates uncertainty.

**Check 5. Complete-case and MI estimates are compared.** The difference is 0.56 points, about one third of a standard error. It is not large, but it is in the direction predicted by the baseline difference. A much larger difference would suggest that the missingness is informative or the imputation model is wrong.

**Check 6. The imputation model is congenial with the analysis.** The analysis estimates a mean, and the imputation model includes the outcome and the baseline predictor. If the analysis were a comparison of two groups, the imputation model would need the group indicator and a group-by-baseline interaction if the analysis had one. In this example, the mean is unconditional, so the imputation model is compatible. A model that omitted baseline would still be congenial for an unconditional mean but would predict the missing values worse, and the FMI would be larger.

**Check 7. The number of imputations is adequate for the FMI.** The hand calculation uses m = 3 and reports FMI = 0.007 under a fixed model, which would suggest that m = 3 is enough. The proper implementation reports FMI = 0.15, for which m = 20 is the minimum and m = 100 is safer. The discrepancy is the reason the number of imputations should be chosen from a proper FMI estimate, not from the proportion of missing values or from a fixed rule of five.

**Check 8. The sensitivity analysis is anchored in clinical units.** The delta grid runs from -4 to +3, a range of 7 points on a scale whose observed SD is 5.05. The tipping point of -6.25 lies outside the grid, which is why the table alone does not show it. A sensitivity analysis should extend far enough to find the tipping point, or state that the conclusion survives the entire plausible range.

The delta grid is not a confidence interval, and the tipping point is not a p-value. The grid answers "how wrong would the missing values have to be?" The answer is a clinical judgment: a shift of 6.25 points in a quality-of-life score may be implausible if the score's MCID is 3, and plausible if the missing patients were a selected severely affected group. The analysis provides the number; the clinical team provides the plausibility.

A final check is to ask whether the missingness is plausibly related to the outcome in the direction required by the tipping point. If dropout is known to occur when patients feel better, a negative delta (missing patients worse) is implausible, and the MAR conclusion is strengthened. If dropout occurs when patients deteriorate, the negative delta is plausible, and the conclusion is weakened. The clinical narrative, not the arithmetic, resolves which case applies.

### 6.8 Vary one assumption: MNAR delta adjustment

Suppose the three missing patients are not MAR. They may have dropped out because they were not improving. A delta adjustment adds a fixed amount to each imputed value before analysis. Because delta applies to 3 of 12 patients, the mean shifts by delta × 3/12 = 0.25 × delta.

| Delta per imputed value | Shift in mean | Pooled mean | 95% CI |
|---:|---:|---:|---:|
| 0 (MAR) | 0.00 | 9.56 | (6.82, 12.30) |
| +2 | +0.50 | 10.06 | (7.32, 12.80) |
| +3 | +0.75 | 10.31 | (7.57, 13.05) |
| -2 | -0.50 | 9.06 | (6.32, 11.80) |
| -4 | -1.00 | 8.56 | (5.82, 11.30) |

A negative delta of about -6.25 points per imputed patient would move the mean to 8.0, the target: 9.5616 - 0.25 × 6.25 = 8.00. That is a large shift: it says the three missing patients would have to be 6.25 points worse than the model predicts, on a scale where the observed SD is about 5 points. Such a scenario is possible but not the most plausible; the data cannot rule it out.

The conclusion changes only under an MNAR assumption of that magnitude. The honest report is: "Under MAR, the mean change is 9.6 points (95% CI 6.8 to 12.3). The result would fall to the 8-point target only if the missing patients were at least 6.3 points worse than predicted, a shift larger than the observed standard deviation. We cannot exclude that scenario, but it is not supported by the observed baseline distribution."

### 6.9 What to report

Report the MI estimate as the primary result: **9.56 points (95% CI 6.82 to 12.30), m = 3 in the hand calculation, FMI = 0.007 under a fixed model; m = 20 and FMI = 0.15 in the proper implementation**. Report the complete-case estimate (9.00, 95% CI 5.70 to 12.30) as a comparison. Report the delta-adjustment table and the tipping point.

The MI estimate is primary because complete-case analysis assumes MCAR, and the missing patients differ systematically in baseline score. The MI standard error is smaller because it uses all 12 patients, not because it ignores uncertainty. The delta analysis shows the result is robust to moderate MNAR but not to a large negative shift.

The decision statement: "The estimated mean improvement is 9.6 points, above the 8-point target, but the interval includes 6.8. The conclusion depends on the missing-data assumption: under MAR the target is met at the point estimate, and under a moderate negative delta it is not. A larger sample with better follow-up would resolve the ambiguity."

Three features of this example generalize to real analyses:

1. **The complete-case estimate was not absurd.** It was 9.0, close to the MI estimate of 9.6. The bias was in the direction predicted by the baseline difference, but it was small relative to the standard error. A complete-case analysis is not always catastrophically wrong; it is wrong in a direction that must be diagnosed, not assumed.
2. **The MI interval was narrower, not wider.** This surprises teams who expect imputation to add uncertainty. It adds between-imputation variance, but it also uses more patients. The net effect depends on the amount of missingness and the predictive power of the imputation model.
3. **The tipping point was large in clinical units.** A shift of 6.25 points is larger than the observed SD of 5.05. That is a useful statement: the conclusion would require the missing patients to be more different than the observed patients are from each other. It does not prove MAR, but it bounds the plausibility of the alternative.

A different dataset would produce different numbers, but the workflow is the same: describe, assume, impute, pool, diagnose, stress-test, and report the decision in the units the decision uses.

The choice between reporting the MI estimate and the complete-case estimate is not a choice between a right and a wrong number. Both are estimates under different assumptions. The MI estimate is primary because its assumption (MAR) is more plausible than the complete-case assumption (MCAR), not because it is larger. If the MI estimate had been lower, the same reasoning would apply, and the report would say so.

## 7. Code, query, or calculation

The script implements multiple imputation for a continuous outcome with a fully observed auxiliary variable. It draws the regression parameters for each imputation, imputes the missing values from the predictive normal distribution, pools the mean and variance with Rubin's rules, and runs a delta-adjusted sensitivity analysis. It uses only the standard library.

**Assumptions:**

- The outcome and the auxiliary variable are jointly normal conditional on the model.
- The missingness is MAR given the baseline score.
- The analysis estimand is the unconditional mean of the outcome.
- The parameter draws use the normal approximation to the posterior and a scaled chi-square draw for the variance.
- The delta adjustment adds a fixed amount to every imputed outcome.

```python
import math, random
from statistics import NormalDist

ND = NormalDist()

X = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105]
Y = [2, 7, 3, None, 8, 6, 13, None, 11, 16, None, 15]

obs = [(x, y) for x, y in zip(X, Y) if y is not None]
n_obs = len(obs)
mx = sum(x for x, y in obs) / n_obs
my = sum(y for x, y in obs) / n_obs
sxx = sum((x - mx) ** 2 for x, y in obs)
sxy = sum((x - mx) * (y - my) for x, y in obs)
b = sxy / sxx
a = my - b * mx
res = [y - (a + b * x) for x, y in obs]
sse = sum(r * r for r in res)
df = n_obs - 2
sigma = math.sqrt(sse / df)

mean_cc = my
sd_cc = math.sqrt(sum((y - my) ** 2 for x, y in obs) / (n_obs - 1))
se_cc = sd_cc / math.sqrt(n_obs)
print(f"complete case: n={n_obs} mean={mean_cc:.4f} sd={sd_cc:.4f} se={se_cc:.4f}")
print(f"imputation model: y = {a:.4f} + {b:.4f} x, residual sd={sigma:.4f}")

# Proper-ish MI: draw sigma2 from scaled chi-square, draw beta from its normal approximation.
def draw_params(df, sse, sxx, mx, a, b):
    chi2 = sum(ND.inv_cdf(random.random()) ** 2 for _ in range(df))
    sigma2 = sse / chi2
    sig = math.sqrt(sigma2)
    # covariance of (a, b) for simple linear regression
    var_b = sigma2 / sxx
    var_a = sigma2 * (1 / n_obs + mx * mx / sxx)
    cov_ab = -sigma2 * mx / sxx
    z1, z2 = ND.inv_cdf(random.random()), ND.inv_cdf(random.random())
    b_star = b + math.sqrt(var_b) * z1
    a_star = a + cov_ab / math.sqrt(var_b) * z1 + math.sqrt(var_a - cov_ab ** 2 / var_b) * z2
    return a_star, b_star, sig

def impute(m, delta=0.0, seed=2026):
    random.seed(seed)
    means, variances = [], []
    for _ in range(m):
        aa, bb, sig = draw_params(df, sse, sxx, mx, a, b)
        full = []
        for x, y in zip(X, Y):
            if y is None:
                full.append(aa + bb * x + ND.inv_cdf(random.random()) * sig + delta)
            else:
                full.append(y)
        mu = sum(full) / len(full)
        v = sum((q - mu) ** 2 for q in full) / (len(full) - 1)
        means.append(mu)
        variances.append(v / len(full))
    Q = sum(means) / m
    U = sum(variances) / m
    B = sum((q - Q) ** 2 for q in means) / (m - 1)
    T = U + (1 + 1 / m) * B
    return Q, math.sqrt(T), (1 + 1 / m) * B / T

for m in (20, 100):
    Q, se, fmi = impute(m)
    print(f"MI m={m}: mean={Q:.4f} se={se:.4f} 95% CI=({Q-1.96*se:.4f}, {Q+1.96*se:.4f}) FMI={fmi:.4f}")

Q3, se3, _ = impute(100, delta=3.0)
print(f"delta +3: mean={Q3:.4f} 95% CI=({Q3-1.96*se3:.4f}, {Q3+1.96*se3:.4f})")
```

**Expected output:**

```text
complete case: n=9 mean=9.0000 sd=5.0498 se=1.6833
imputation model: y = -9.2047 + 0.2409 x, residual sd=2.3952
MI m=20: mean=9.5060 se=1.6113 95% CI=(6.3478, 12.6641) FMI=0.1534
MI m=100: mean=9.4429 se=1.5508 95% CI=(6.4033, 12.4825) FMI=0.1216
delta +3: mean=10.1929 95% CI=(6.9487, 13.4370)
```

**What the output demonstrates:**

- The complete-case estimate (9.0000) and the imputation model coefficients (-9.2047, 0.2409) match the hand calculation in Section 6.4 exactly.
- The m = 20 and m = 100 estimates differ by 0.063 points. That difference is Monte Carlo error from the finite number of imputations and the parameter draws, not a change in the data. A stable estimate should not move materially when m doubles; if it does, m is too small or the imputation model is unstable.
- The FMI falls from 0.1534 at m = 20 to 0.1216 at m = 100. The FMI estimate itself has Monte Carlo error, and more imputations give a more stable estimate. It does not converge to zero; the missing information is a property of the data, not of m.
- The delta +3 line shifts the mean by 0.75 points, not 3 points, because only 3 of 12 values are imputed and shifted: 3 × 3/12 = 0.75. Teams that expect a 3-point shift and see 0.75 sometimes conclude the code is wrong; the arithmetic is correct.
- The MI interval is narrower than the complete-case interval at the same confidence level, because the sample size is 12 rather than 9. The between-imputation variance is small relative to the within-imputation variance in this example.

> **Note:** The parameter draw uses a normal approximation to the posterior of (a, b) and a scaled chi-square draw for the variance. This is a simplified Bayesian imputation model, not a full posterior sampler. It is adequate for a teaching example with a normal outcome and a linear model; production MI should use a validated package with predictive mean matching or a full Bayesian model.

## 8. Data, standards, and regulatory context

### 8.1 Binding requirements and conventions

| Instrument | Status | What it constrains | What it does not constrain |
|---|---|---|---|
| ICH E9 (1998) | Binding through regional adoption | Analysis sets, missing values, sensitivity analyses | Specific imputation methods |
| ICH E9(R1) (2019) | Binding through regional adoption | The estimand and the handling of intercurrent events, which is separate from missing data | The imputation model |
| EMA Guideline on Missing Data in Confirmatory Clinical Trials (2010, effective 2011) | Binding in the EU through CHMP | Complete-case analysis not acceptable as primary; pre-specified methods and sensitivity analyses; reporting of reasons | The choice of software |
| FDA guidance on covariates and estimands | Guidance | Estimand definition and pre-specification | Missing-data methods specifically |
| Consolidated Standards of Reporting Trials (CONSORT) 2010 | Journal editorial policy | Reporting of losses, exclusions, and the analysis population | Imputation method |
| Strengthening the Reporting of Observational Studies in Epidemiology (STROBE) and REporting of studies Conducted using Observational Routinely-collected health Data (RECORD) | Journal editorial policy | Reporting of missingness and data-cleaning rules | Method choice |
| NRC report on missing data (2010) | Expert consensus, not binding | Principles for prevention, analysis, and sensitivity | Specific methods |

### 8.2 What the EMA guideline requires

The EMA guideline states that:

- Missing data should be minimized by design and by retrieval of dropouts.
- The amount, timing, pattern, and reasons for missingness should be reported.
- Complete-case analysis is not acceptable as the primary analysis.
- Single imputation methods such as LOCF and BOCF are context-dependent and should not be assumed conservative.
- Multiple imputation is preferred to single imputation.
- Methods that assume MAR, such as MMRM and MI, must be accompanied by sensitivity analyses under departures from MAR.
- Sensitivity analyses should explore different assumptions, not just different implementations of the same assumption.

The guideline is binding in the EU through the CHMP; Australia's Therapeutic Goods Administration has adopted it. In the US, the FDA does not have a standalone missing-data guidance at the time of writing, but ICH E9 and E9(R1) apply, and review divisions expect a pre-specified missing-data plan.

### 8.3 The estimand and the missing-data plan are separate

ICH E9(R1) made a distinction that is easy to miss. An intercurrent event, such as discontinuation of treatment, is handled by the estimand's strategy: treatment policy, hypothetical, composite, while on treatment, or principal stratum. The missing data that follow the event are handled by the missing-data method. A treatment-policy estimand does not imply that the missing outcome should be imputed under MAR; it implies that the outcome should be collected and analyzed regardless of discontinuation, and the missing-data method determines how the uncollected values are handled.

A protocol that says "treatment policy estimand, multiple imputation under MAR" is coherent only if the imputation model is consistent with the treatment-policy question: the imputation model should include the treatment arm and the post-discontinuation outcomes where available.

A common inconsistency is a treatment-policy estimand paired with an imputation model that excludes outcomes observed after discontinuation. That combination effectively changes the estimand to a while-on-treatment quantity without saying so. The reviewer sees a treatment-policy label and a while-on-treatment number. The fix is to align the model with the estimand, or to change the estimand and report the change.

### 8.4 Reporting requirements

- **CONSORT item 13a** requires the number of participants at each stage, including losses and exclusions with reasons.
- **CONSORT item 16** requires the number analyzed in each group and whether the analysis was by original assignment.
- **STROBE item 12c** requires the handling of missing data.
- **RECORD item 13.1** requires the study population selection, including filtering by data quality and availability.

The standards require the missingness to be reported and the method to be stated. They do not require a specific method, and they do not resolve the choice between MI and MMRM.

### 8.5 Missingness by data source

The mechanism and the appropriate method depend on the data source. The same statistical machinery applies, but the clinical narrative differs.

| Data source | Typical missingness | Likely mechanism | Practical approach |
|---|---|---|---|
| Randomized trial | Missed visits, dropout, discontinuation | MAR or MNAR, often differential by arm | MMRM or MI; delta by arm |
| Electronic health record | Labs not ordered, vitals not recorded, notes not coded | MAR given severity and workflow; structural for some tests | MI with severity and visit type; do not impute uncoded diagnoses |
| Insurance claims | Out-of-network care, uncoded diagnoses, enrollment gaps | MAR given enrollment and prior utilization; structural for non-covered services | MI with enrollment history; sensitivity to out-of-network care |
| Device registry | Lost to follow-up, incomplete revision capture | MAR or MNAR; death competes with revision | MI plus competing-risk analysis |
| Wearable | Non-wear time, device failure, sync gaps | MCAR for failure; MAR or MNAR for non-wear | Time-series imputation; sensitivity to non-wear |
| Survey | Item nonresponse, unit nonresponse | MAR given demographics; MNAR for sensitive items | MI plus nonresponse weights |

The EHR row is the one most often mishandled. A missing lab is not a random omission; it is a clinical decision. Imputing it under MAR given severity, visit type, and prior values is often defensible. Imputing a diagnosis that was never coded is not, because the diagnosis may not exist in the record at all.

### 8.6 What to write in the statistical analysis plan

A complete missing-data section has six parts:

1. The variables that can be missing and the expected amount, by arm and visit.
2. The definition of structural missingness and the recoding rule.
3. The assumed mechanism, with the clinical rationale.
4. The primary method, the imputation model, and the number of imputations.
5. The pooling method and the reporting of the FMI.
6. The sensitivity analyses, including the delta range and the tipping-point rule.

The section should be written before unblinding. A missing-data plan written after seeing the results is a description of what was done, not a pre-specification, and reviewers treat it accordingly.

### 8.7 What the standards do not cover

- They do not define "acceptable" amounts of missingness; that depends on the estimand and the decision.
- They do not require a specific number of imputations.
- They do not require a tipping-point analysis, although reviewers increasingly ask for one.
- They do not validate the imputation model. A model that includes the outcome can still be misspecified in its functional form.

The distinction between binding requirement and convention matters when a sponsor is deciding how much evidence to provide. The EMA guideline creates an expectation that a reviewer can enforce. CONSORT and STROBE create an expectation that an editor can enforce. Neither tells the analyst which imputation model to use, and neither relieves the sponsor of the obligation to justify the assumption. A submission that satisfies every reporting item but assumes MCAR without evidence is compliant in form and weak in substance.

## 9. Industry practice and case evidence

### 9.1 Documented practice: MMRM as the default in trials

For continuous longitudinal endpoints, MMRM is a common primary analysis in industry trials because it uses all observed post-baseline data under MAR and handles the within-patient correlation. The EMA guideline discusses MMRM as a method that can be valid under MAR, while noting that it is not a complete solution when the missingness is MNAR. The practical consequence is that a trial protocol often specifies MMRM as primary and MI or a pattern-mixture model as sensitivity.

### 9.2 Documented case: the QRISK cholesterol controversy

Sterne and colleagues, in their 2009 BMJ paper on multiple imputation, describe the QRISK cholesterol example as a cautionary case. A multiple imputation analysis produced a different estimate from the complete-case analysis, and the discrepancy was traced to the imputation model and the handling of a variable with a large amount of missingness. The case is used to illustrate that MI is not a black box: the imputation model, the missingness pattern, and the plausibility of MAR all matter.

**Inference:** the case does not show that MI is unreliable. It shows that an MI analysis without a diagnostic review can produce a result that is difficult to explain, and that the explanation is usually in the imputation model.

### 9.3 Documented evidence: reporting quality

Sterne and colleagues surveyed 59 articles using multiple imputation published in the New England Journal of Medicine, The Lancet, BMJ, and JAMA between 2002 and 2007. They found that reporting was generally inadequate: many articles did not state the number of imputations, the variables in the imputation model, or the software settings. The paper's Box 2 provides a reporting checklist that is still a useful minimum standard.

The implication for industry is that an MI analysis is not complete until the imputation model is documented. A reviewer cannot reproduce an MI result from a point estimate and a p-value.

### 9.4 Documented practice: sensitivity analyses in submissions

The EMA guideline requires sensitivity analyses that explore different assumptions. Common industry practice is a delta-adjusted MI analysis for the primary continuous endpoint, with a tipping-point table. The delta is often anchored to the minimal clinically important difference or to a fraction of the observed treatment effect. The table reports the estimate and interval across a grid of deltas, and the conclusion states the delta at which the result changes.

**Inference:** the specific delta values used in submissions are not publicly standardized. The principle is documented; the calibration is a judgment that should be justified in the protocol.

### 9.5 Case evidence: missing data in real-world databases

In claims and EHR data, missingness is often structural or workflow-driven rather than patient-driven. A lab value is missing because the test was not ordered; a diagnosis is missing because the visit was coded under a different system. Multiple imputation can help with the first, but it cannot recover a diagnosis that was never recorded. The RECORD statement's emphasis on code lists and validation studies exists because the missingness in routinely collected data is often a measurement problem, not a sampling problem.

### 9.6 The cost of getting it wrong

Missing-data errors rarely produce an obviously wrong number. They produce a plausible number with the wrong uncertainty or the wrong sign for a subgroup. The costs appear later:

- A coverage decision based on a complete-case estimate that dropped the sickest patients.
- A trial that fails to replicate because the primary analysis used LOCF and the effect was carried by patients who dropped out for toxicity.
- A quality program that looks effective because patients who left the program early were excluded from the denominator.
- A model that performs well in development and poorly in deployment because the imputation model encoded site-specific workflow.

**Inference:** the frequency of these failures is not captured in a single public statistic. The EMA guideline's explicit rejection of complete-case analysis as primary, and the CONSORT and STROBE reporting items, exist because reviewers encountered them repeatedly.

### 9.7 What vendors and internal decks tend to omit

A missing-data section is often reduced to one sentence: "Missing data were handled using multiple imputation." That sentence omits the variables in the model, the number of imputations, the FMI, and the sensitivity analysis. A reviewer cannot reproduce the result, and a decision-maker cannot judge how much of the answer depends on the imputation.

The minimum defensible sentence is longer: "Missing 12-month outcomes (18.2% overall; 12.1% treatment, 24.3% control) were imputed using a model with arm, baseline score, site, and the 6-month score; m = 50; the pooled estimate had FMI = 0.14; a delta adjustment by arm from -5 to +5 points did not change the conclusion."

The sentence is long because the analysis has four decisions that a reader must be able to audit: how much is missing, what was assumed, how it was imputed, and how sensitive the answer is. A short sentence hides all four.

### 9.8 The NRC recommendations in practice

The 2010 National Research Council report on missing data in clinical trials made three recommendations that shaped current practice:

1. Design trials to minimize missingness, including follow-up of dropouts for the primary outcome.
2. Pre-specify the primary analysis and the sensitivity analyses, and make the primary analysis valid under a plausible assumption.
3. Report the missingness and the assumptions so that a reviewer can assess the sensitivity of the conclusion.

The report also emphasized that no single method is universally valid, and that the analysis should be tailored to the estimand. That position is reflected in the EMA guideline and in the ICH E9(R1) distinction between the estimand and the missing-data handling.

**Inference:** the report is not binding, but its recommendations are visible in the structure of modern protocols: a missing-data section, a pre-specified primary analysis, and a sensitivity analysis with a stated assumption.

### 9.9 Case evidence: pragmatic trials and EHR-based studies

Pragmatic trials embedded in health systems face missingness that is generated by care delivery rather than by a research protocol. An outcome may be missing because the patient was seen at an outside hospital, because the relevant lab was not ordered, or because the encounter was coded under a different system. The missingness is often MAR given the observed utilization and severity, but it can be MNAR if patients who disengage from care are also those with the worst outcomes.

The practical response in this literature is to use the EHR as a secondary source, to link to claims or vital records, and to pre-specify a primary outcome that is observable for everyone (such as death or a registry event). When a complete outcome is not available, the analysis reports the amount of missingness by arm and uses MI with utilization variables as auxiliaries.

**Inference:** the pattern is documented in pragmatic trial methods papers; the specific missingness rates vary by setting and are not summarized by a single statistic.

### 9.10 What mature teams do differently

1. They report missingness before they model it.
2. They separate structural from sporadic missingness.
3. They write the imputation model in the analysis plan, including the outcome and auxiliary variables.
4. They use m = 20 or more and report the FMI.
5. They run a delta-adjusted sensitivity analysis and report the tipping point.
6. They compare complete-case and MI estimates and explain the difference.
7. They archive the imputation model and the completed datasets for audit.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Complete-case analysis as primary | The analysis sample is much smaller than the cohort | Rows with any missing value are deleted | Bias and power loss | MI or MMRM; complete case as sensitivity |
| MCAR assumed without justification | No mechanism discussion in the protocol | Missingness related to health or access | Bias in an unknown direction | State and test the assumption; sensitivity |
| Mean imputation for inference | Imputed values equal to the mean | Variance artificially reduced | Intervals too narrow | MI or MMRM |
| Single regression imputation | Imputed values lie on the regression line | No residual variation | Variance understated | Add draws; MI |
| LOCF as primary | The outcome is assumed flat after dropout | Single imputation with a strong assumption | Bias in an unknown direction | MMRM or MI; LOCF as sensitivity only |
| Missing indicator misinterpreted | A large coefficient on "missing" | The indicator mixes prognosis and workflow | Spurious group effect | Do not use for effect estimation |
| Imputation model omits the outcome | The imputed values are uncorrelated with the outcome | Uncongeniality | Bias toward the null | Include the outcome |
| Imputation model omits an auxiliary variable | The FMI is large and the imputed values are poor | MAR is implausible given the model | Bias | Include variables that predict missingness |
| Too few imputations | m = 5 with high FMI | Monte Carlo error | Unstable estimate and interval | m = 20 or more |
| Bounded variable imputed normally | Values outside the plausible range | Normal model ignores bounds | Impossible values | PMM or truncated normal |
| Structural missingness imputed | A value exists for a patient who cannot have it | The missingness is not "unknown" | Invalid variable | Recode as not applicable |
| No MNAR sensitivity analysis | Only the MAR result is reported | The key assumption is untested | Overconfident conclusion | Delta adjustment; tipping point |
| Pooling on the wrong scale | Averaging odds ratios | Nonlinearity | Inconsistent estimate and interval | Pool on the log scale |
| Ignoring clustering in imputation | Imputed values vary less than observed | Within-cluster correlation omitted | Understated variance | Include cluster effects |
| Imputing a post-treatment variable | The imputation carries treatment effects | Leakage of the exposure | Biased effect | Exclude or handle as a mediator |
| Imputation model does not match the analysis | The analysis has an interaction the imputation model lacks | Uncongeniality | Bias toward the null | Include the interaction in the imputation model |
| Ignoring site in the imputation model | Imputed values are too similar across sites | Between-site heterogeneity omitted | Understated variance | Include site as a fixed or random effect |
| Pooled estimate without the t correction | The interval uses 1.96 with small m | Between-imputation variance is estimated imprecisely | Interval too narrow | Barnard-Rubin degrees of freedom |
| Imputing after an intercurrent event without aligning the estimand | The imputation contradicts the treatment-policy strategy | The estimand and the missing-data method answer different questions | Uninterpretable estimate | Define the estimand first; align the model |
| Treating death as missing | Death is imputed as a score | The outcome cannot exist after death | Impossible values | Competing risk or composite endpoint |
| No comparison with complete case | Only one analysis is reported | The reader cannot see the sensitivity | Hidden dependence on the assumption | Report both |

The first four rows account for most of the avoidable bias in applied work. The last four account for most of the irreproducibility.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| MAR given the imputation model | MI and MMRM are consistent under MAR | Missingness depends on the unobserved value | Delta adjustment; tipping point |
| Congenial imputation model | The imputed distribution matches the analysis | The analysis has interactions or transformations not in the imputation model | Compare imputed and observed distributions; refit with a richer model |
| Correct functional form | The predicted values are unbiased | Nonlinearity or omitted interactions | Residual plots; PMM as a check |
| Correct distributional assumption | Draws have the right spread | Skewness or bounds | Compare imputed and observed ranges |
| Independent draws | Between-imputation variance is valid | Improper imputation with fixed parameters | Check that completed datasets differ |
| Sufficient m | Monte Carlo error is small | High FMI with small m | Re-run with more imputations |
| Structural missingness excluded | Impossible values are not imputed | Not-applicable values treated as unknown | Data dictionary review |
| Clustering represented | Within-cluster correlation is preserved | Cluster effects omitted | Compare within-cluster variance |
| No leakage from the future | Imputation does not use post-outcome information | Future visits used to impute baseline | Variable timing review |
| Stable missingness process | The model applies across groups and time | Missingness differs by arm or period | Stratify the imputation model; interaction terms |
| Correct pooling scale | The pooled estimate is consistent | Pooling odds ratios directly | Pool on the log scale; transform back |
| Valid variance estimator | The interval has nominal coverage | m is small and the t correction is omitted | Barnard-Rubin degrees of freedom |
| Correct treatment of death | The outcome is defined for every patient | Death treated as an ordinary missing value | Impossible imputations | Competing risk or composite |
| Aligned estimand and method | The imputation answers the same question as the analysis | Treatment-policy estimand with a while-on-treatment imputation | Contradictory estimate | State both before analysis |

**Limits of this topic.** Multiple imputation cannot identify an MNAR estimand without an assumption about the missingness process; the sensitivity analysis quantifies the dependence, it does not remove it. The methods here assume a correctly specified imputation model, which is untestable in its entirety. The worked example has 12 patients and is a teaching device; the FMI and the tipping point are illustrative, not estimates for a real program. The choice between MI and MMRM is often settled by regulatory precedent, and this topic does not adjudicate that choice.

## 12. Exercises

### Exercise 1. Classify the mechanism (recall and application, 40 minutes)

For each scenario, name the most plausible mechanism and justify it in two sentences:

1. A sensor fails at random and its readings are lost.
2. Patients with higher blood pressure are more likely to have a follow-up lab drawn.
3. Patients who feel worse stop completing a symptom diary.
4. A pregnancy test is missing for male patients.
5. A site stopped entering data for two weeks during a system migration.

**Deliverable.** A five-row table: scenario, mechanism, justification, and the sensitivity analysis that would test the assumption.

**Acceptance criteria.**

- Structural missingness is identified as not applicable, not as MCAR.
- The justification refers to the missing value, not only to an observed variable.
- At least two scenarios are assigned a sensitivity analysis.
- No scenario is assigned a mechanism solely from a test result.

### Exercise 2. Describe a missingness pattern (application, 50 minutes)

**Data.** A trial has 200 patients. The 12-month outcome is missing for 30 patients: 10 in the treatment arm and 20 in control. Baseline severity is higher among the missing patients in both arms.

**Deliverable.** A missingness table, a statement of the likely mechanism, and a comparison of the complete-case and MI analysis plans.

**Acceptance criteria.**

- Missingness is reported by arm and by baseline severity.
- The answer notes that the difference by arm suggests differential missingness.
- The plan includes the outcome and baseline severity in the imputation model.
- A delta adjustment by arm is proposed.

### Exercise 3. Pool with Rubin's rules (application, 45 minutes)

**Data.** Five imputations give estimates 9, 11, 10, 12, and 8, with within-imputation variances 4.0, 4.4, 3.8, 4.2, and 4.0.

**Deliverable.** Qbar, Ubar, B, T, the standard error, the 95% interval, and the FMI.

**Acceptance criteria.**

- B uses the divisor m - 1.
- T includes the (1 + 1/m) factor.
- The FMI is computed as (1 + 1/m)B/T.
- The interval is interpreted correctly.

### Exercise 4. Choose the number of imputations (calculation, 30 minutes)

**Deliverable.** For an FMI of 0.25, compute the relative efficiency of m = 5, 20, and 50 using 1/(1 + FMI/m). State which value you would use for a primary analysis and why.

**Acceptance criteria.**

- The formula is applied correctly.
- The comparison is expressed as a percentage of the precision of infinite imputations.
- The recommendation is justified by the FMI and the sensitivity analyses.
- The answer notes that the relative efficiency formula is an approximation.

### Exercise 5. Delta adjustment and tipping point (application, 50 minutes)

**Data.** A pooled mean is 10.0 with a standard error of 2.0. Twenty percent of the outcome values are missing. The decision target is a mean of 8.0.

**Deliverable.** The shift in the mean for delta values of -2, -5, and -10, and the delta at which the mean reaches the target.

**Acceptance criteria.**

- The shift is delta × 0.20, not delta.
- The target is reached at delta = -10.
- The answer states whether a 10-point shift is clinically plausible.
- The interval is reported at the tipping point.

### Exercise 6. Critique an LOCF analysis (critique, 45 minutes)

**Claim.** "Missing 12-month scores were carried forward from month 6. The treatment was superior (p = 0.04). LOCF is conservative, so the result is robust."

**Deliverable.** A 400-word critique covering the assumption, the direction of bias, the alternative methods, and the reporting requirements.

**Acceptance criteria.**

- The answer rejects the claim that LOCF is always conservative.
- It explains that LOCF assumes no change after dropout.
- It proposes MMRM or MI as the primary analysis and LOCF as a sensitivity analysis.
- It states what must be reported about the missingness.

### Exercise 7. Write an imputation plan (design, 90 minutes)

**Scenario.** A pragmatic trial of a care program has a continuous primary outcome at 12 months, 18% missingness, and 12 baseline covariates. Missingness is higher in the control arm.

**Deliverable.** A one-page plan with the estimand, the mechanism assumption, the imputation model, m, the pooling method, the diagnostics, and two sensitivity analyses.

**Acceptance criteria.**

- The imputation model includes the outcome, the arm, and the covariates.
- The plan states the functional forms and any interactions.
- m is justified by an anticipated FMI.
- The delta adjustment is by arm.
- The reporting table includes the FMI.

### Exercise 8. Audit an MI result (critique, 60 minutes)

**Claim.** "Multiple imputation was used. The treatment effect was significant (p = 0.03)."

**Deliverable.** A list of at least eight questions you would ask before accepting the result, and the minimum reporting package you would require.

**Acceptance criteria.**

- The questions cover the mechanism, the imputation model, m, the FMI, the pooling, and the sensitivity analysis.
- The reporting package includes the missingness table and the imputation model specification.
- The answer distinguishes an underreported analysis from a wrong one.
- The final judgment is conditional on the answers.

The exercise is deliberately asymmetric: the questions are easy to write and hard to answer. An analyst who can answer all eight has an analysis that can survive review. An analyst who can answer three has a result that may be correct but cannot be defended, and the correct next step is to reconstruct the missing pieces before the result is used.

## 13. Self-check answers

**Answer 1 (Exercise 3 Rubin's rules).**

```text
Qbar = (9 + 11 + 10 + 12 + 8) / 5 = 50 / 5 = 10.0
Ubar = (4.0 + 4.4 + 3.8 + 4.2 + 4.0) / 5 = 20.4 / 5 = 4.08
B = [(9-10)^2 + (11-10)^2 + (10-10)^2 + (12-10)^2 + (8-10)^2] / 4
  = [1 + 1 + 0 + 4 + 4] / 4 = 10 / 4 = 2.5
T = 4.08 + (1 + 1/5) × 2.5 = 4.08 + 3.0 = 7.08
SE = sqrt(7.08) = 2.661
95% CI = 10.0 +/- 1.96 × 2.661 = (4.78, 15.22)
FMI = 3.0 / 7.08 = 0.424
```

The between-imputation variance is large relative to the within-imputation variance, so the FMI is 42.4%. With m = 5 and this FMI, the Monte Carlo error is not negligible; m = 50 or more would be preferable. The interval is wide because the five estimates vary substantially, which is the correct signal that the imputed values are uncertain.

If the five estimates had been nearly identical, B would be small and the FMI would be small. That would not necessarily mean the imputation is good; it could mean the imputations are identical because the model is deterministic. The diagnostic is the combination of a small B and visibly different completed datasets, which indicates a stable imputation, versus a small B and identical datasets, which indicates an improper imputation.

**Answer 2 (Exercise 4 relative efficiency).**

```text
m = 5:  1 / (1 + 0.25/5)  = 1 / 1.05  = 0.952
m = 20: 1 / (1 + 0.25/20) = 1 / 1.0125 = 0.988
m = 50: 1 / (1 + 0.25/50) = 1 / 1.005  = 0.995
```

At m = 5, the estimate has 95.2% of the precision of an infinite number of imputations; at m = 20, 98.8%; at m = 50, 99.5%. For a primary analysis with FMI 0.25 and a planned delta adjustment, m = 50 is a reasonable choice, because the sensitivity analysis multiplies the number of datasets and the Monte Carlo error should be small in both the primary and the sensitivity result. The formula is an approximation and assumes the imputation model is correct.

**Answer 3 (Exercise 5 tipping point).**

```text
Shift = delta × 0.20
delta = -2:  shift = -0.40; mean = 9.60
delta = -5:  shift = -1.00; mean = 9.00
delta = -10: shift = -2.00; mean = 8.00
```

The target is reached at delta = -10 points per missing patient. A 10-point shift is large relative to a standard error of 2.0 and would mean the missing patients were substantially worse than the model predicts. Whether that is plausible depends on the clinical scale and the reasons for missingness. The interval at the tipping point is 8.00 ± 3.92 = (4.08, 11.92), so the target is at the point estimate, not at the lower confidence limit.

A 10-point shift is also larger than the difference between the complete-case and MI estimates in the worked example, which was 0.56 points. That comparison is useful: the sensitivity analysis asks for a shift an order of magnitude larger than the observed discrepancy between methods. It does not prove MAR, but it shows that the conclusion is not fragile to small departures from it.

**Answer 4 (Exercise 1 mechanisms).**

| Scenario | Mechanism | Justification | Sensitivity |
|---|---|---|---|
| Sensor failure at random | MCAR | The failure is unrelated to the reading | Compare with MI; no delta needed |
| Higher blood pressure more likely to be tested | MAR | Missingness depends on observed blood pressure | MI with blood pressure; delta |
| Worse patients stop the diary | MNAR | Missingness depends on the unobserved symptom score | Delta adjustment; tipping point |
| Pregnancy test for male patients | Structural | The value does not apply | Recode; do not impute |
| Site migration gap | MCAR or MAR | Depends on whether the gap is related to patient characteristics | Site indicator; sensitivity |

**Answer 5 (Exercise 6 critique).**

A strong critique:

- LOCF assumes the outcome does not change after the last observation. That is a strong assumption, not a conservative one.
- If patients drop out because they are deteriorating, LOCF freezes them at a better value and biases the treatment effect toward the null or away from it depending on the arm.
- If patients drop out because they improve, LOCF freezes them at a worse value and can exaggerate the effect.
- The direction of bias cannot be signed without knowing why patients dropped out.
- MMRM or MI under MAR should be the primary analysis, with LOCF as a sensitivity analysis if the regulator requests it.
- The report must state the amount, timing, pattern, and reasons for missingness by arm.

**Answer 6 (Exercise 8 minimum package).**

The minimum reporting package:

1. Missingness by variable, arm, and visit.
2. The mechanism assumption and its rationale.
3. The imputation model, including the outcome and auxiliary variables.
4. m and the software.
5. The pooled estimate, interval, and FMI.
6. The complete-case comparison.
7. The delta-adjusted sensitivity analysis and the tipping point.
8. The diagnostics for imputed values.

A result with all eight is auditable. A result with the first five is reproducible but not stress-tested. A result with only a p-value is neither.

The audit question that most often exposes a weak analysis is simple: "What would the result be if the missing patients had a different outcome?" If the analyst cannot answer with a number, the sensitivity analysis was not run. If the analyst can answer but the answer is not in the report, the analysis was run and not reported. Both are fixable, and the fix is the same: report the delta table and the tipping point.

A second useful question is "How many imputations, and why that number?" An answer of "five, because that is the default" reveals that the FMI was not considered. An answer of "50, because the anticipated FMI was 0.2 and the sensitivity analysis uses four delta values" reveals that the analysis was planned.

A third is "What did you do about the patients who died?" The answer distinguishes a missing outcome from a competing event, and it often reveals whether the estimand was defined before the data were examined.

**Answer 7 (Exercise 2 plan outline).**

A strong plan:

- Reports 15.0% missingness overall: 5.0% in treatment and 10.0% in control.
- Notes that baseline severity is higher among missing patients, so MCAR is implausible.
- Assumes MAR given baseline severity, site, and observed intermediate outcomes.
- Imputes the 12-month outcome with a model that includes arm, baseline severity, site, and the 6-month outcome.
- Uses m = 50 because the FMI is expected to be around 0.15.
- Pools with Rubin's rules on the mean difference scale.
- Runs a delta adjustment by arm, with delta from -5 to +5 points.
- Reports the complete-case estimate and the FMI.

**Answer 8 (Exercise 7 rubric).**

A strong plan states the estimand first, then the mechanism, then the model, then the sensitivity. It uses the arm and the baseline covariates in the imputation model, pre-specifies the delta range, and defines the tipping point in clinical units. A weak plan says "use multiple imputation" without specifying the model, the number of imputations, or the sensitivity analysis.

A complete plan also states what will not be imputed. Structural missingness, outcomes after death when the estimand excludes them, and variables that do not exist for a subgroup should be listed and recoded. The list of exclusions is as important as the list of inclusions, because it shows that the analyst has distinguished "unknown" from "not applicable."

Finally, the plan should state who will review the imputed values and what would trigger a change to the model. A pre-specified trigger, such as "if any imputed value lies outside the clinical range, the model will be refitted with predictive mean matching," prevents a post hoc rescue that is really a specification search.

## 14. Interview questions

**1. What is the difference between MCAR and MAR?**

A strong answer says MCAR means missingness is independent of both observed and unobserved data, while MAR means it depends on observed data but not on the unobserved value given them. It notes that MAR is an assumption, not a test result. A weak answer says MAR means "random" and MCAR means "completely random."

**2. Why is complete-case analysis biased under MAR?**

A strong answer explains that the complete cases are a selected subset defined by the missingness process, and that the selection can change the distribution of the outcome and the exposure. A weak answer says it "loses data" without the bias mechanism.

**3. Why does single imputation understate uncertainty?**

A strong answer says the imputed values are treated as known, so the between-imputation variance is zero. It connects this to Rubin's rules and the total variance. A weak answer says single imputation is "less accurate" without the variance mechanism.

**4. What is the FMI, and how does it differ from the proportion missing?**

A strong answer defines the fraction of total variance due to missingness and notes that it depends on the predictability of the missing values, not only on their count. A weak answer treats the two as the same.

**5. How many imputations do you need?**

A strong answer gives m = 20 as a default, scales m with the FMI, and mentions the relative efficiency formula. A weak answer says "five is enough."

The follow-up question is usually "what if you do not know the FMI in advance?" A strong answer says to run a pilot with m = 20, estimate the FMI, and increase m if the FMI is above 0.2 or if a sensitivity analysis will be run. The point is that m is a design choice, not a software default.

**6. When is MI not appropriate?**

A strong answer names structural missingness, missing exposure imputed from the outcome, and settings where the imputation model cannot be specified. A weak answer says "when the data are MNAR" without distinguishing sensitivity from identification.

**7. What is a tipping-point analysis?**

A strong answer defines the delta at which the conclusion changes and says it must be interpreted in clinical units. A weak answer calls it "a sensitivity analysis" without the threshold.

**8. Why must the outcome be in the imputation model?**

A strong answer explains congeniality: omitting the outcome attenuates the exposure-outcome association and biases the effect toward the null. A weak answer says "to improve prediction" without the bias direction.

**9. What is the difference between a pattern-mixture and a selection model?**

A strong answer describes stratification by missingness pattern versus a joint model of outcome and missingness, and notes that both require an untestable assumption. A weak answer says they are two names for the same method.

**10. How would you handle missing baseline covariates in an observational study?**

A strong answer uses MI with the outcome and other confounders, compares with complete case, and discusses residual confounding. A weak answer drops the variable or imputes the mean.

**11. What would you report alongside an MI estimate?**

A strong answer lists m, the FMI, the imputation model, the complete-case comparison, and the sensitivity analysis. A weak answer reports the estimate and p-value only.

**12. Why is LOCF not automatically conservative?**

A strong answer explains that the direction of bias depends on the trajectory after dropout and the reason for dropout, and that freezing a deteriorating patient at a better value can exaggerate the effect. A weak answer repeats the claim that LOCF is conservative.

A strong answer also distinguishes the two dropout scenarios. If patients who improve stop attending, LOCF freezes them at an improved value, which is close to the truth and biases the comparison toward the null when the other arm is observed. If patients who deteriorate stop attending, LOCF freezes them at a better value and biases the comparison away from the null. The sign is a function of the clinical process, not of the method.

## 15. Key terms

- **Auxiliary variable:** a variable included in the imputation model because it predicts missingness or the outcome.
- **Available-case analysis:** an analysis that uses all observed values for each variable separately.
- **Barnard-Rubin degrees of freedom:** the small-sample correction for pooled MI intervals.
- **Baseline observation carried forward:** single imputation using the baseline value.
- **Between-imputation variance:** the variance of the estimate across completed datasets.
- **Complete-case analysis:** an analysis restricted to rows with no missing values.
- **Congeniality:** compatibility between the imputation model and the analysis model.
- **Delta adjustment:** shifting imputed values by a fixed amount to model MNAR.
- **Fraction of missing information:** the proportion of total variance attributable to missingness.
- **Fully conditional specification:** an imputation method that cycles through conditional models for each variable.
- **Imputation model:** the model used to generate replacement values.
- **Inverse probability weighting:** weighting observed cases by the inverse probability of observation.
- **Last observation carried forward:** single imputation using the last observed value.
- **Little's MCAR test:** a test of mean differences across missingness patterns.
- **MAR:** missing at random, conditional on observed data.
- **MCAR:** missing completely at random.
- **Mean imputation:** replacing missing values with the variable mean.
- **Missing indicator:** a variable marking whether a value is missing.
- **Missing not at random:** missingness depends on the unobserved value.
- **Monotone pattern:** once missing, all later variables are missing.
- **Multiple imputation:** repeated imputation and pooling.
- **Non-monotone pattern:** missingness can occur and reverse.
- **Pattern-mixture model:** a model stratified by missingness pattern.
- **Predictive mean matching:** imputation using donor values with similar predicted values.
- **Relative efficiency:** the precision of m imputations relative to infinite m.
- **Rubin's rules:** the formulas for pooling estimates and variances.
- **Selection model:** a joint model of the outcome and the missingness process.
- **Single imputation:** one replacement value per missing value.
- **Structural missingness:** a value that does not apply.
- **Tipping point:** the sensitivity parameter value at which the conclusion changes.
- **Total variance:** within-imputation variance plus the between-imputation component.
- **Uncongeniality:** incompatibility between the imputation and analysis models.
- **Within-imputation variance:** the average variance across completed datasets.
- **Item missingness:** a missing value for one variable in an otherwise observed record.
- **Unit missingness:** a missing record for an entire participant.
- **Dropout:** departure from follow-up before the planned end.
- **Imputation:** the process of replacing missing values with plausible values.
- **Response propensity:** the probability that a participant provides an observed value.
- **Monte Carlo error:** the variability introduced by a finite number of imputations.
- **Ignorable missingness:** a missingness process that can be ignored given MAR and distinctness.
- **Nonresponse bias:** bias from differences between respondents and nonrespondents.
- **Hot-deck imputation:** imputation from a similar observed donor.
- **Cold-deck imputation:** imputation from an external or historical source.
- **Imputation variance:** the component of total variance due to imputed values.
- **Sensitivity parameter:** the unknown quantity that indexes departures from MAR.

## 16. Sources

All URLs were accessed on 25 October 2026.

1. International Council for Harmonisation. *ICH E9: Statistical Principles for Clinical Trials*. Step 4, 5 February 1998. https://database.ich.org/sites/default/files/E9_Guideline.pdf
2. International Council for Harmonisation. *ICH E9(R1) Addendum on Estimands and Sensitivity Analysis in Clinical Trials*. Step 4, 20 November 2019. https://database.ich.org/sites/default/files/E9-R1_Step4_Guideline_2019_1203.pdf
3. European Medicines Agency. *Guideline on Missing Data in Confirmatory Clinical Trials*. EMA/CPMP/EWP/1776/99 Rev. 1, adopted 24 June 2010, effective 1 January 2011. https://www.ema.europa.eu/en/documents/scientific-guideline/guideline-missing-data-confirmatory-clinical-trials_en.pdf
4. Rubin, D. B. "Inference and Missing Data." *Biometrika* 1976;63(3):581-592. https://doi.org/10.1093/biomet/63.3.581
5. Sterne, J. A. C., White, I. R., Carlin, J. B., et al. "Multiple imputation for missing data in epidemiological and clinical research: potential and pitfalls." *BMJ* 2009;338:b2393. https://doi.org/10.1136/bmj.b2393
6. National Research Council. *The Prevention and Treatment of Missing Data in Clinical Trials*. Panel on Handling Missing Data in Clinical Trials, Committee on National Statistics. National Academies Press, 2010.
7. Cro, S., Morris, T. P., Kenward, M. G., Carpenter, J. R. "Sensitivity analysis for clinical trials with missing continuous outcome data using controlled multiple imputation: A practical guide." *Statistics in Medicine* 2020;39(21):2815-2842. https://doi.org/10.1002/sim.8569
8. Little, R. J. A., Rubin, D. B. *Statistical Analysis with Missing Data*. 3rd edition. Wiley, 2019.
9. van Buuren, S. *Flexible Imputation of Missing Data*. 2nd edition. Chapman and Hall/CRC, 2018.
10. Schulz, K. F., Altman, D. G., Moher, D., for the CONSORT Group. "CONSORT 2010 Statement." *PLoS Medicine* 2010;7(3):e1000251. https://doi.org/10.1371/journal.pmed.1000251
11. STROBE Initiative. *STROBE Statement*. https://www.strobe-statement.org/
12. Benchimol, E. I., Smeeth, L., Guttmann, A., et al. "The RECORD Statement." *PLoS Medicine* 2015;12(10):e1001885. https://doi.org/10.1371/journal.pmed.1001885
13. Therapeutic Goods Administration. *Guideline on Missing Data in Confirmatory Clinical Trials*. https://www.tga.gov.au/resources/resource/international-scientific-guidelines/guideline-missing-data-confirmatory-clinical-trials

> **Note:** The EMA guideline is the primary binding source for Section 8 in the EU. The NRC report is expert consensus, not regulation. The Sterne 2009 paper is the source for the QRISK example and the reporting survey described in Section 9.

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`03_survival-analysis.md`](03_survival-analysis.md) — censoring is not missingness.
- Causal inference: [`05_causal-inference-and-dags.md`](05_causal-inference-and-dags.md) — missing confounders and selection.
- Sensitivity analysis: [`08_sensitivity-analysis.md`](08_sensitivity-analysis.md) — E-values and tipping points.
- Reporting: [`10_reporting-statistical-results.md`](10_reporting-statistical-results.md) — reporting missingness and FMI.
- Survey file: [`../../04_statistics_causal_inference.md`](../../04_statistics_causal_inference.md)
- Labs: [`../../labs/`](../../labs/) — Lab 4.5 covers complete case, MI, and MNAR sensitivity.
- Foundations: [`../../01A_foundations_assumptions_and_units.md`](../../01A_foundations_assumptions_and_units.md) — assumptions ledger.
