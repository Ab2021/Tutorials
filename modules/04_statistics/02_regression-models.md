# Regression Models

> Module 04 — Statistics and Causal Inference
> Topic 02 of 10
> Estimated study time: 3.5 hours
> Prerequisites: 01
> Version 2026.3

---

## 1. Why this topic exists

Regression is the workhorse of biomedical analytics, and most of its failures are specification failures rather than arithmetic failures. A team fits a logistic model to predict readmission, reads the coefficient on a program indicator, and reports it as a treatment effect. The coefficient is a conditional log odds ratio: it is adjusted for every other variable in the model, it is not a risk difference, and it can change if a prognostic covariate is added even when that covariate is not a confounder. None of this is visible in the software output.

The decisions that depend on regression are ordinary:

- A pharmacy and therapeutics committee compares two anticoagulants using claims data. The regression must adjust for the variables that drove the choice of drug, not for every variable that happens to be in the table.
- A trial team adjusts for baseline covariates to gain precision. The adjustment must be pre-specified, and the estimator must target the same estimand as the unadjusted analysis.
- A clinical team deploys a risk score. The model must be calibrated in the deployment population, not only discriminative in the development sample.
- An analyst builds a model of length of stay to forecast capacity. A linear model on a skewed outcome produces predictions that are systematically wrong for the longest stays.

The costs are asymmetric and often invisible:

- **Wrong scale.** An odds ratio of 2.0 for a common outcome can correspond to a risk ratio near 1.3. A formulary committee that reads the odds ratio as a risk ratio overestimates the benefit.
- **Collider adjustment.** Adjusting for a variable caused by both treatment and outcome creates a spurious association. The model fits better and the estimate is more biased.
- **Overfitting.** A model with 40 covariates and 60 events produces impressive in-sample statistics and unreliable out-of-sample predictions.
- **Extrapolation.** A linear model fitted to adults aged 40 to 70 is used to predict costs at age 85, where the relationship was never observed.
- **Separation.** A binary covariate perfectly predicts the outcome in a small subgroup; the coefficient diverges and the software either warns or silently returns a large number.
- **Unchecked calibration.** A model with an area under the receiver operating characteristic curve (AUC) of 0.80 can still overpredict risk by a factor of two in the deployment site.

Regulators have taken positions on the parts of this that affect trials. The United States Food and Drug Administration (FDA) final guidance "Adjusting for Covariates in Randomized Clinical Trials for Drugs and Biological Products," issued in May 2023, states that covariate adjustment is generally acceptable, should be prospectively specified, should include randomization strata, and should use model-robust standard errors for linear models. It also addresses the non-collapsibility of odds and hazard ratios and distinguishes conditional from unconditional estimands. The guidance is about randomized trials, but the statistical issues it describes apply to every regression used for an effect estimate.

This topic treats regression as a set of linked choices: the estimand, the outcome scale, the functional form, the adjustment set, the estimator, the uncertainty, and the validation. The arithmetic is the easy part. The choices are where analyses succeed or fail.

> **Note:** A regression coefficient is not automatically an effect. It is an adjusted association under a model. It becomes an effect estimate only when the adjustment set is justified by a causal graph or by randomization, and when the scale of the coefficient matches the scale of the decision.

## 2. Learning outcomes

By the end of this topic you can:

1. **Match** an outcome type to a regression family and link function, and state the scale of the resulting coefficient.
2. **Distinguish** conditional from marginal effects, and **compute** a standardized marginal effect from a fitted nonlinear model.
3. **Select** an adjustment set from a directed acyclic graph (DAG) rather than from univariate screening or stepwise selection.
4. **Specify** functional form for continuous covariates, including splines and interaction terms, and **test** whether the added complexity is supported.
5. **Diagnose** linear, logistic, and count models with the checks appropriate to each: residual structure, influence, collinearity, separation, overdispersion, and calibration.
6. **Explain** why a coefficient changes when a prognostic covariate is added, using the omitted-variable bias formula and the non-collapsibility of the odds ratio.
7. **Validate** a model internally and describe what internal validation does and does not establish.
8. **Report** a regression result with the estimand, the scale, the interval, and the model diagnostics.

Outcome 2 maps to Exercise 4 and self-check item 3. Outcome 5 maps to Exercises 3 and 5. Outcome 6 maps to the worked example and Exercise 2.

## 3. The core mental model

Regression is a map with four layers. Draw it before writing any model code.

```text
  CAUSAL QUESTION
  Population, treatment contrast, outcome, time horizon
        |
        |  DAG: which variables are confounders, mediators, colliders?
        v
  ADJUSTMENT SET  (a design decision, not a data decision)
        |
        |  outcome type -> family and link
        v
  MODEL
  g(E[Y | X]) = beta0 + beta1 X1 + ... + betap Xp
  systematic part (linear predictor) + link g + error distribution
        |
        |  fit: least squares, maximum likelihood, penalized likelihood
        v
  COEFFICIENTS  (conditional on the other covariates, on the model scale)
        |
        |  marginal standardization / average marginal effects
        v
  EFFECT ON THE DECISION SCALE  +  interval  +  validation
```

### 3.1 The model has three separable parts

Every generalized linear model (GLM) has:

1. **A linear predictor** η = β0 + β1X1 + ... + βpXp. This is the part that can be wrong in its functional form: a missing square term, a missing interaction, or a variable that should be log-transformed.
2. **A link function** g that maps the mean of the outcome to the linear predictor: identity for linear regression, logit for logistic, log for Poisson. This is the part that determines the scale of the coefficient.
3. **A variance function** that describes how the variance depends on the mean: constant for linear, p(1-p) for logistic, μ for Poisson. This is the part that determines the standard errors and the need for overdispersion corrections.

Separating the three makes diagnosis tractable. A residual pattern with a constant variance but a curved mean is a functional-form problem. A variance that grows with the mean in a Poisson model is a variance-function problem. A coefficient that is on the wrong scale for the decision is a link problem.

### 3.2 Conditional versus marginal

A coefficient in a model with covariates is conditional: it compares patients with the same values of the other covariates. A marginal effect compares the average outcome under treatment with the average outcome under control, across the whole population.

For linear regression with an identity link, the conditional and marginal effects coincide when the model is correctly specified and there is no interaction between treatment and covariates. This is why linear adjustment is simple.

For logistic regression, they do not coincide. The odds ratio is not collapsible: the conditional odds ratio from a model with a prognostic covariate is farther from 1 than the marginal odds ratio, even in a randomized trial with no confounding. The difference is not bias. It is a property of the measure.

The practical consequence is that a logistic coefficient cannot be read as a population risk difference. To obtain a marginal effect, standardize:

```text
For each patient i, predict p1_i = g^-1(beta0 + beta_T + beta_X X_i)
                    and p0_i = g^-1(beta0 + beta_X X_i)
Marginal RD = mean(p1_i) - mean(p0_i)
Marginal RR = mean(p1_i) / mean(p0_i)
```

This is g-computation under the model. It answers "what would the average risk be if everyone were treated versus if no one were treated," assuming the model is correct.

### 3.3 Omitted-variable bias in one formula

If the true model is

```text
Y = beta0 + beta_T T + beta_C C + error
```

and the fitted model omits C, then the estimated coefficient on T converges to

```text
beta_T_omitted = beta_T + beta_C × delta
```

where delta is the coefficient from regressing C on T. The bias is the product of two quantities: how strongly C affects the outcome, and how strongly C is associated with treatment. If either is zero, there is no bias. This is why a variable that predicts the outcome but is unrelated to treatment is not a confounder, and why a variable related to treatment but unrelated to the outcome is not a confounder either.

The formula also explains why adding a prognostic covariate to a randomized trial changes the estimate slightly even though randomization makes delta zero in expectation: in any finite sample, delta is not exactly zero, and the change is sampling variation, not bias. Adjustment in a randomized trial buys precision, not identification.

### 3.4 Non-collapsibility, in plain terms

Suppose treatment halves the odds of an event in every stratum of a covariate. The marginal odds ratio across strata will be closer to 1 than 0.5, because the strata with different baseline risks are mixed. The risk ratio and risk difference do not have this problem when the stratum-specific effects are homogeneous: the marginal risk ratio equals the common risk ratio, and the marginal risk difference is the weighted average of the stratum-specific differences.

This matters for reporting. A logistic model that adjusts for a strong prognostic covariate will show a larger odds ratio than the unadjusted table, even when the treatment is randomized. Reporting the adjusted odds ratio as "the effect" without saying it is conditional overstates the population impact.

### 3.5 Adjustment is a design decision

The adjustment set comes from the causal graph, not from the data. Three rules:

- **Adjust for confounders:** common causes of treatment and outcome.
- **Do not adjust for mediators:** variables on the causal path from treatment to outcome. Adjustment removes part of the effect.
- **Do not adjust for colliders:** variables caused by both treatment and outcome. Adjustment creates a spurious association.

Data-driven selection, such as stepwise regression or univariate screening at p < 0.10, violates all three rules because it selects on associations in the sample, which can be induced by colliders, mediators, or selection. The result is a model that predicts well in the sample and estimates the wrong quantity.

### 3.6 Prediction and explanation are different jobs

A model built to predict should be judged by calibration and discrimination out of sample. A model built to estimate an effect should be judged by the identification of the adjustment set, the scale of the coefficient, and the sensitivity of the estimate to plausible alternatives. The same data can support both, but not with the same specification choices. Prediction models can include colliders and mediators if they improve forecasting; effect models cannot. Effect models can include a single well-justified covariate; prediction models usually need many.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or scale |
|---|---|---|---|
| Linear predictor | The sum β0 + β1X1 + ... + βpXp | Called "the model" | Depends on the link |
| Link function | The function relating the mean outcome to the linear predictor | Ignored when interpreting coefficients | A function: identity, logit, log |
| Family | The distributional assumption for the outcome | Confused with the link | Distribution |
| Ordinary least squares (OLS) | Estimation by minimizing the sum of squared residuals | Used for binary or count outcomes | Coefficient in outcome units per covariate unit |
| Logistic regression | A GLM with a logit link and Bernoulli variance | Coefficient read as a risk difference | Log odds ratio, or odds ratio after exponentiation |
| Poisson regression | A GLM with a log link and variance equal to the mean | Used when the outcome is overdispersed | Log rate ratio, or rate ratio |
| Negative binomial | A count model with an extra dispersion parameter | Chosen without checking overdispersion | Log rate ratio |
| Cox model | A semiparametric model for the hazard | Coefficient read as a risk ratio | Log hazard ratio |
| Coefficient (β) | The change in the linear predictor per one-unit change in a covariate, holding others fixed | Interpreted as a marginal effect | Model scale per covariate unit |
| Intercept (β0) | The linear predictor when all covariates are zero | Interpreted as a baseline risk when zero is outside the data range | Model scale |
| Reference category | The category against which dummy variables are compared | Left unstated | A level of a categorical variable |
| Dummy variable | A 0/1 indicator for a category | All categories entered, causing collinearity | Unitless |
| Interaction | A product term allowing the effect of one variable to depend on another | Added without pre-specification | Model scale |
| Confounder | A common cause of treatment and outcome | Any variable correlated with the outcome | Not a number |
| Mediator | A variable on the causal path from treatment to outcome | Adjusted as if it were a confounder | Not a number |
| Collider | A common effect of two variables | Adjusted because it predicts the outcome | Not a number |
| Collinearity | High correlation among covariates | Ignored until coefficients flip sign | Variance inflation factor (VIF), unitless |
| Variance inflation factor (VIF) | 1/(1-R²) for a covariate regressed on the others | Compared with a universal threshold without context | Unitless, at least 1 |
| Separation | A covariate or combination perfectly predicts the outcome | Ignored because the software returned a number | A property of the data |
| Overdispersion | Variance greater than the model assumes | Ignored in Poisson models | Dispersion ratio, unitless |
| Deviance | A measure of model fit relative to a saturated model | Interpreted as R² | Model-specific |
| Akaike information criterion (AIC) | A penalized fit statistic for comparing models | Used to compare non-nested models casually | Unitless |
| Calibration | Agreement between predicted and observed risks | Replaced by discrimination | Calibration slope and intercept |
| Discrimination | Ability to rank patients by risk | Reported as if it were calibration | AUC or c-statistic, 0.5 to 1 |
| Average marginal effect | The mean change in predicted outcome per unit change in a covariate | Confused with the coefficient | Outcome scale |
| Marginal standardization | Averaging model predictions over the observed covariate distribution | Called "adjustment" without saying what is averaged | Outcome scale |
| Offset | A variable with a known coefficient of 1, such as log person-time | Omitted, turning a rate into a count | Model scale |
| Spline | A piecewise polynomial that allows a flexible curve | Knots chosen after seeing the fit | Model scale |
| Penalized likelihood | Estimation with a penalty, such as Firth's Jeffreys prior | Treated as a different estimand without comment | Model scale |
| Internal validation | Assessing optimism with bootstrapping or cross-validation | Called external validation | Optimism-corrected performance |
| External validation | Assessing performance in a separate population | Omitted | Performance in the new population |
| Propensity score | The probability of treatment given covariates | Used as a covariate without checking overlap | Probability, 0 to 1 |

Two conventions used in this topic:

- A coefficient is reported with its scale and its covariate unit: "−0.413 mmHg per 1 mmHg of baseline systolic blood pressure," not "−0.413."
- An odds ratio is labelled as conditional when it comes from a model with covariates, and the marginal risk difference is reported alongside it when the decision is about population impact.

> **Note:** Exponentiated logistic coefficients are odds ratios, not risk ratios. The gap between them grows as the outcome becomes more common. At a control risk of 20%, an odds ratio of 0.64 corresponds to a risk ratio of about 0.70; at a control risk of 2%, the two are nearly identical.

## 5. The mechanics, step by step

The procedure has fourteen steps. Steps 1 to 6 are specification. Steps 7 to 10 are fitting and diagnosis. Steps 11 to 14 are interpretation, uncertainty, validation, and reporting.

The order matters because a model cannot repair a specification error. No diagnostic will tell you that you adjusted for a collider if the collider is measured well and the model fits.

### Step 1. Define the estimand and the effect scale

**Input:** The decision, the population, the treatment contrast, the outcome, and the time horizon.

**Output:** A statement of the target effect on a named scale: conditional log odds ratio, marginal risk difference, rate ratio, mean difference, or hazard ratio.

**Why it exists:** The scale determines the model family, the interpretation, and the reporting. A team that wants a population risk difference but fits a logistic model must add a standardization step. A team that wants a conditional odds ratio for an etiologic question can report the coefficient directly.

| Decision question | Target scale | Model | Reporting step |
|---|---|---|---|
| How many events per 100 patients are avoided? | Marginal risk difference | Logistic plus standardization, or a linear probability model | Standardize predictions |
| Is the treatment associated with the outcome after adjustment? | Conditional odds ratio | Logistic | Exponentiate with interval |
| How many times higher is the event rate? | Rate ratio | Poisson or negative binomial | Exponentiate with interval |
| How much does the outcome change in its own units? | Mean difference | Linear | Coefficient with interval |
| Does the effect vary by subgroup? | Ratio of effects or difference of differences | Model with interaction | Report the interaction and stratum-specific effects |
| What is the expected cost per patient? | Mean cost | Linear or gamma GLM with log link | Predict and calibrate |

**Edge case:** In a randomized trial, the unadjusted and adjusted estimates target the same estimand on the risk difference scale, but not on the odds ratio scale. Pre-specify which one is primary and do not switch after seeing the results.

### Step 2. Draw the DAG and fix the adjustment set

**Input:** Subject-matter knowledge, the timing of each variable, and the proposed treatment and outcome definitions.

**Output:** A list of variables to adjust for, a list to exclude, and a justification for each exclusion.

**Why it exists:** Regression adjusts for whatever is on the right-hand side. Including a mediator removes part of the effect; including a collider creates bias. The DAG is the only defensible way to decide.

| Variable role | Example | Action | Reason |
|---|---|---|---|
| Confounder | Baseline blood pressure affects both treatment choice and outcome | Adjust | Blocks a backdoor path |
| Mediator | Adherence lies on the path from program to readmission | Do not adjust for the total effect | Adjustment removes the indirect effect |
| Collider | Hospitalization is caused by treatment and by underlying severity | Do not adjust | Adjustment induces association |
| Instrument | Provider preference affects treatment but not outcome except through treatment | Do not adjust in a regression; use an instrument | Adjustment does not remove confounding |
| Descendant of a confounder | A lab value measured after treatment | Do not adjust unless it is a proxy measured before treatment | May carry treatment effects |
| Pure predictor | A prognostic score unrelated to treatment | Adjust for precision, not identification | Reduces variance; does not change bias |

**Edge case:** A variable measured after treatment can be a confounder if it is a proxy for a pre-treatment cause and is not affected by treatment. The DAG must represent the timing. When in doubt, use the pre-treatment measurement.

### Step 3. Choose the outcome family and link

**Input:** The outcome type, its range, its mean-variance relationship, and the plausible generating process.

**Output:** A family and link, with a reason and a planned diagnostic.

**Why it exists:** The family determines the scale of the coefficient and the variance function; choosing it after seeing the data invites a scale that flatters the result.

| Outcome | Family | Link | Coefficient scale | Default diagnostic |
|---|---|---|---|---|
| Continuous, roughly normal, constant variance | Gaussian | Identity | Units per covariate unit | Residual versus fitted; normal quantile plot |
| Continuous, positive, skewed | Gaussian on log outcome, or gamma with log link | Log | Ratio of geometric means | Residuals on the log scale; influence of high values |
| Binary | Bernoulli | Logit | Log odds ratio | Calibration; separation check |
| Binary, risk difference wanted | Bernoulli with identity link, or logistic plus standardization | Identity or logit | Risk difference | Predicted risks within 0 to 1 |
| Count, variance approximately equal to mean | Poisson | Log | Log rate ratio | Dispersion ratio |
| Count, variance greater than mean | Negative binomial | Log | Log rate ratio | Estimated dispersion parameter |
| Count with known exposure | Poisson with offset | Log | Log rate ratio | Offset correctness |
| Time to event | Cox or parametric | Log hazard | Log hazard ratio | Proportional hazards |
| Ordinal | Cumulative logit | Logit | Cumulative odds ratio | Proportional odds |

**Decision table for a binary outcome when the decision needs a risk difference:**

| Situation | Preferred approach | Reason |
|---|---|---|
| Randomized trial, marginal risk difference is the estimand | Standardization from a logistic model, or a binomial model with identity link and robust standard errors | Both target the marginal RD |
| Observational study with many confounders | Logistic model plus standardization, or IPTW with a marginal model | Separates the treatment model from the outcome model |
| Sparse events | Firth penalization or exact methods, then standardization | Ordinary MLE is unstable |
| Very common outcome | Risk difference or log-binomial model | Odds ratio diverges from the risk ratio |

### Step 4. Specify the functional form of continuous covariates

**Input:** The plausible shape of each relationship, the number of observations, and the number of events.

**Output:** A form for each continuous covariate: linear, log, spline, or categorical, with the knots or cut points pre-specified.

**Why it exists:** A linear term forces the effect to be constant across the range. If the true relationship is a threshold or a curve, the model is misspecified and the treatment coefficient can absorb the error.

| Form | When | Cost | Check |
|---|---|---|---|
| Linear | Roughly linear within the observed range; few events | Bias if the relationship curves | Residual plot; add a quadratic term as a test |
| Log | Positive, right-skewed covariate such as dose or cost | Zero values need a shift | Compare with linear by AIC and residual pattern |
| Restricted cubic spline, 3 to 5 knots | Smooth but nonlinear relationship; enough data | More parameters; knots must be pre-specified | Plot the fitted curve with a reference band |
| Piecewise linear | Known threshold or change point | Threshold must be justified, not chosen from the data | Sensitivity to the threshold |
| Categorical | Clinical cut points with a meaning (for example, eGFR stages) | Loss of information within categories | Check that the cut points are pre-specified |

**Edge case:** Do not choose the form by looking at the fitted curve and moving knots until it looks good. That is overfitting with extra steps. Pre-specify the form from clinical knowledge, then use a global test of nonlinearity as a diagnostic.

### Step 5. Decide whether interactions are needed

**Input:** The estimand, the subgroup hypotheses, and the sample size.

**Output:** A list of pre-specified interactions, or a statement that none are included.

**Why it exists:** An interaction changes the interpretation of every lower-order coefficient. If treatment is interacted with age, the treatment coefficient becomes the effect at age zero, which is usually outside the data range.

| Situation | Interaction | Reporting consequence |
|---|---|---|
| The decision differs by a pre-specified subgroup | Include treatment × subgroup | Report stratum-specific effects and the interaction test |
| The effect is expected to vary continuously | Include treatment × continuous covariate, possibly with a spline | Report the effect curve, not a single number |
| Exploratory search for heterogeneity | Do not include; use a separate pre-specified analysis | Label as exploratory; adjust for multiplicity |
| No hypothesis of heterogeneity | Do not include | Report the average effect |

**Centre continuous variables before forming an interaction.** Otherwise the main-effect coefficient is an extrapolation to zero.

### Step 6. Choose the coding and reference category

**Input:** The categorical variables and their levels.

**Output:** A coding scheme with an explicit reference level for each variable.

**Why it exists:** Every dummy coefficient is relative to the reference. A reference category with few patients produces a wide interval and an unstable comparison. The choice must be clinical, not alphabetical.

- Use the largest clinically meaningful group as the reference, or the standard-of-care group.
- Do not create a dummy for every level plus an intercept; that is collinear.
- For ordinal categories, consider a linear term or cumulative model rather than arbitrary dummies.
- Report the reference category in every table.

### Step 7. Choose the estimator

**Input:** The family, the sample size, the number of events, and the presence of separation or collinearity.

**Output:** An estimator with its properties stated.

**Why it exists:** Different estimators target different quantities and have different small-sample behaviour; the choice must match the estimand and the data.

| Situation | Estimator | Reason | Caution |
|---|---|---|---|
| Gaussian, independent, constant variance | OLS | Efficient and unbiased under the model | Sensitive to outliers and heteroscedasticity |
| Gaussian, heteroscedastic | OLS with robust (sandwich) standard errors | Correct intervals without a variance model | Point estimate unchanged |
| Binary, adequate events | Maximum likelihood | Consistent and efficient | Separation and sparse-data bias |
| Binary, separation or few events | Firth penalized likelihood | Finite estimates; reduces small-sample bias | Estimates are shrunk; report the method |
| Count, overdispersed | Negative binomial or quasi-Poisson | Correct variance | The dispersion model is an assumption |
| Clustered | GEE with robust SE, mixed model, or cluster bootstrap | Accounts for correlation | Different targets: marginal vs conditional |
| Many covariates, prediction | Ridge, lasso, or elastic net | Controls variance | Not for effect estimation without care |

**Events per variable:** a common rule of thumb is at least 10 events per estimated parameter for logistic regression. It is a heuristic, not a law. With 8 events per variable, estimates are biased away from the null and intervals are too narrow; Firth penalization or a smaller model is preferable.

### Step 8. Fit the model and check convergence

**Input:** The design matrix, the outcome, and the estimator.

**Output:** Converged estimates, or a diagnosis of non-convergence.

**Why it exists:** Iterative estimators can fail silently. A logistic model that has not converged may return coefficients from the last iteration with a misleading standard error.

Checks:

- The iterative algorithm reports convergence within the iteration limit.
- The gradient at the solution is near zero.
- The Hessian is positive definite.
- No fitted probability is exactly 0 or 1 in a logistic model.
- No count model has fitted means that are implausibly large.

**Edge case:** Separation produces a warning in some software and a large coefficient with an enormous standard error in others. Treat any coefficient above about 10 on the log-odds scale with suspicion and inspect the covariate.

### Step 9. Run the diagnostics appropriate to the family

**Input:** The fitted model, the residuals, and the covariates.

**Output:** A diagnostic table with pass, fail, or unresolved for each check.

**Why it exists:** Diagnostics detect the failures that the coefficient table cannot show, such as nonlinearity, separation, and influence.

| Family | Diagnostic | What it detects | Remedy |
|---|---|---|---|
| Linear | Residual versus fitted | Nonlinearity, heteroscedasticity | Add a term, transform, or use robust SE |
| Linear | Normal quantile plot of residuals | Heavy tails, outliers | Report influence; consider robust regression |
| Linear | Cook's distance | Single observations driving the fit | Report with and without; investigate |
| Linear | VIF | Collinearity | Combine or drop variables; do not interpret unstable coefficients |
| Logistic | Calibration plot and slope | Systematic over- or underprediction | Recalibrate or add terms |
| Logistic | AUC | Discrimination only | Do not use as a substitute for calibration |
| Logistic | Separation check | Infinite coefficients | Firth or drop the covariate |
| Logistic | Hosmer-Lemeshow test | Lack of fit in deciles of risk | Use with caution; it is sensitive to sample size |
| Poisson | Dispersion ratio | Overdispersion | Negative binomial or quasi-Poisson |
| Poisson | Residual versus fitted | Mean-variance misspecification | Check the offset and the link |
| Cox | Schoenfeld residuals | Non-proportional hazards | Stratify, add a time interaction, or use RMST |
| Any | Influence of individual clusters | One site drives the result | Cluster-robust SE; report site-level estimates |

### Step 10. Handle the pathological cases

**Input:** The diagnostics that failed.

**Output:** A documented remedy and a sensitivity analysis.

**Why it exists:** Pathological cases can produce finite-looking output that is not interpretable; the remedy must be documented so the estimate is not mistaken for a routine one.

| Problem | Symptom | Remedy | What to report |
|---|---|---|---|
| Separation | A coefficient of 8 or more with a huge SE; fitted probabilities of 0 or 1 | Firth penalization; collapse categories; drop the covariate | The remedy and the change in the treatment estimate |
| Zero cells in a contingency table | An odds ratio of 0 or infinity | Exact logistic regression; add 0.5 with disclosure | The exact method and the interval |
| Sparse events | Fewer than 10 events per parameter | Reduce the model; Firth; propensity score as a single covariate | The events-per-variable count |
| Overdispersion | Dispersion ratio above about 1.5 | Negative binomial; quasi-Poisson; robust SE | The estimated dispersion |
| Collinearity | VIF above 5 to 10; coefficients flip sign when a variable is added | Combine variables; use a score; drop one | The correlation matrix and the VIF |
| Nonlinearity | Curved residual pattern | Spline or transformation, pre-specified | The shape of the fitted curve |
| Influential observation | Cook's D above 1, or a large change when removed | Report both estimates; investigate the record | The influence analysis |
| Missing covariates | Complete-case analysis drops 30% | Multiple imputation, then fit and pool | The imputation model and the pooled estimate |

**Edge case: non-proportional hazards.** A Cox model assumes the hazard ratio is constant over time. If Schoenfeld residuals show a trend, the coefficient is a time-averaged summary that can hide a crossing effect. Options are stratification by a covariate that violates the assumption, a time-varying coefficient, or a restricted mean survival time difference. Topic 03 covers the mechanics.

**Edge case: informative censoring.** If censoring depends on prognosis and on treatment, the risk set is no longer representative. Inverse probability of censoring weights or a sensitivity analysis is required. The regression itself cannot repair it.

**Edge case: measurement error.** A covariate measured with random error attenuates its coefficient toward zero, a phenomenon called regression dilution. A binary confounder misclassified at a similar rate in both groups biases the treatment coefficient toward the null if it is a confounder, and the direction is not guaranteed if misclassification differs by group. The remedy is a validation substudy to estimate the error rates, a correction such as regression calibration, or a sensitivity analysis over plausible error rates. The fitted model cannot detect the problem, because measurement error is invisible in the observed data.

**Edge case: the outcome is measured at an irregular time.** If the follow-up window differs by patient, a model of the outcome at a fixed time is comparing patients at different points in their disease course. Either restrict to a common window or include the measurement time in the model, with the functional form pre-specified.

### Step 11. Interpret on the model scale and on the decision scale

**Input:** The coefficients and the covariate distribution.

**Output:** A model-scale statement and a marginal statement, both with intervals.

**Why it exists:** A coefficient is conditional and on the link scale. A decision needs a marginal quantity on the outcome scale.

For a logistic model:

```text
Conditional: OR = exp(beta_T), holding covariates fixed
Marginal:    RD = mean(predicted risk if all treated) - mean(predicted risk if none treated)
             RR = mean(predicted risk if all treated) / mean(predicted risk if none treated)
```

For a linear model with an interaction, the treatment coefficient is the effect at the reference value of the interacting variable. Report the effect at clinically relevant values, or report the average marginal effect.

**Rule:** A coefficient without its scale, its covariate unit, and its reference category is incomplete.

### Step 12. Quantify uncertainty with the design in mind

**Input:** The fitted model and the sampling design.

**Output:** Standard errors and intervals that reflect the design.

**Why it exists:** A standard error that ignores the design produces an interval that is too narrow and a decision that is too confident.

| Design | Standard error | Note |
|---|---|---|
| Independent observations, correct model | Model-based | Can be too small if the variance function is wrong |
| Independent observations, variance misspecified | Heteroscedasticity-robust (HC) | Point estimate unchanged |
| Clustered observations | Cluster-robust or cluster bootstrap | Needs enough clusters, often 30 or more |
| Repeated measures | Mixed model or GEE | Different targets |
| Weighted (IPTW) | Robust SE, or bootstrap | Weights induce correlation |
| Multiple imputation | Rubin's rules | Pool within and between imputation variance |
| Penalized likelihood | Profile likelihood or bootstrap | Wald intervals can be too narrow |

The choice between a robust standard error and a bootstrap is mostly about the number of clusters and the statistic. With 50 or more clusters, a cluster-robust standard error is usually adequate. With 15 clusters, the robust estimator is downward-biased and a cluster bootstrap with a small-sample correction is safer. With a nonlinear statistic such as a standardized risk difference, the bootstrap also captures the uncertainty in the covariate distribution, which the model-based standard error ignores.

### Step 13. Validate the model

**Input:** The fitted model and, ideally, a separate dataset.

**Output:** Calibration and discrimination, with the validation design stated.

**Why it exists:** In-sample fit is optimistic. A model with 20 covariates and 100 events will look better in the development sample than in the next hospital.

| Validation | Method | What it establishes | What it does not |
|---|---|---|---|
| Apparent | Evaluate on the development data | Nothing beyond fit | Generalization |
| Internal, split-sample | Fit on one part, evaluate on the other | Some protection against optimism | Stability of the split |
| Internal, bootstrap | Estimate and correct optimism | Better use of data | Generalization to a new population |
| External, temporal | Evaluate on a later period | Transportability over time | Transportability across sites |
| External, geographic | Evaluate at another site | Transportability across settings | Performance in a third setting |
| Calibration-in-the-large | Compare mean predicted with mean observed | Systematic over- or underprediction | Ranking |
| Calibration slope | Regress outcome on the linear predictor | Whether predictions are too extreme or too compressed | Causes of miscalibration |

For an effect model, validation means checking that the estimate is stable across plausible specifications, not computing an AUC. The relevant checks are the influence analysis, the adjustment-set sensitivity, and the comparison with a design-based estimator such as IPTW.

### Step 14. Report and pre-specify the sensitivity analyses

**Input:** The primary model, the diagnostics, and the validation.

**Output:** A results table and a short sensitivity plan.

**Why it exists:** A result without its estimand, interval, and diagnostics cannot be audited or reproduced.

Report:

1. The estimand and the target scale.
2. The model family, link, and estimator.
3. The adjustment set and its justification.
4. The functional forms and any pre-specified interactions.
5. The sample size, number of events, and events per variable.
6. The coefficient or effect with a 95% interval, on both the model scale and the decision scale.
7. The diagnostics, including the ones that failed.
8. The validation design and results.
9. The sensitivity analyses: alternative adjustment sets, influence, and functional form.

**Sensitivity analyses worth running for almost every effect model:**

- Add or remove one plausible confounder and report the change.
- Refit without the most influential observations.
- Refit with an alternative functional form for the strongest prognostic covariate.
- Refit with a design-based estimator (IPTW or matching) and compare.

If the estimate moves materially under any of these, the primary result is fragile and the report must say so.

### Branch points at a glance

| If the data are | Then the default specification is | Because |
|---|---|---|
| Binary outcome, common | Logistic plus standardization to a risk difference | The conditional odds ratio is not the population effect |
| Binary outcome, rare | Logistic; report the odds ratio and the marginal RD | The odds ratio approximates the risk ratio |
| Count, variance equal to mean | Poisson with an offset for exposure | The rate ratio is the natural parameter |
| Count, overdispersed | Negative binomial or quasi-Poisson | The Poisson variance is too small |
| Clustered | GEE or mixed model with the cluster as the unit | Independence is false |
| Separation present | Firth penalization or exact logistic | The MLE diverges |
| Collinearity high | Combine variables or use a score | Individual coefficients are unstable |
| Nonlinear relationship | Pre-specified restricted cubic spline | A linear term is misspecified |
| Influential observation | Report with and without; investigate the record | One record can drive the estimate |
| Missing covariates | Multiple imputation, then pool | Complete-case analysis selects on completeness |
| Prediction is the goal | Many covariates allowed; calibrate and validate | Prediction and explanation have different rules |
| Effect estimation is the goal | DAG-based adjustment set; sensitivity analysis | Adjustment must block confounding, not maximize fit |

The table is a starting point, not a decision rule. Two rows can apply at once: a clustered count outcome with overdispersion needs a negative binomial mixed model or a GEE with a negative binomial variance, and the analyst must decide which target (conditional or marginal) the decision requires. When rows conflict, the estimand decides.

## 6. Worked example

### 6.1 The question and the estimand

A hypertension clinic compares a new intensification protocol with standard titration. Ten patients are available for a pilot: five received the new protocol and five received standard care. The outcome is the change in systolic blood pressure (SBP) from baseline to 12 weeks, in mmHg. Negative values indicate improvement.

The estimand is the mean difference in 12-week SBP change, new protocol minus standard care, in adults with uncontrolled hypertension, adjusted for baseline SBP. The adjustment is needed because baseline SBP differs between the groups and strongly predicts the change. The target is a conditional mean difference (the model is linear, so it is also the marginal difference under homogeneous effects).

### 6.2 The data

| Patient | Protocol (1 = new) | Baseline SBP (mmHg) | 12-week change (mmHg) |
|---:|---:|---:|---:|
| 1 | 1 | 158 | -18 |
| 2 | 1 | 150 | -14 |
| 3 | 1 | 165 | -20 |
| 4 | 1 | 142 | -13 |
| 5 | 1 | 170 | -22 |
| 6 | 0 | 160 | -12 |
| 7 | 0 | 155 | -7 |
| 8 | 0 | 168 | -15 |
| 9 | 0 | 148 | -6 |
| 10 | 0 | 172 | -18 |

### 6.3 Unadjusted comparison

```text
Mean change, new protocol = (-18 - 14 - 20 - 13 - 22) / 5 = -87 / 5 = -17.4 mmHg
Mean change, standard    = (-12 - 7 - 15 - 6 - 18) / 5 = -58 / 5 = -11.6 mmHg
Unadjusted difference    = -17.4 - (-11.6) = -5.8 mmHg
```

The unadjusted estimate is a 5.8 mmHg greater reduction with the new protocol. The baseline means differ:

```text
Mean baseline, new      = (158 + 150 + 165 + 142 + 170) / 5 = 785 / 5 = 157.0 mmHg
Mean baseline, standard = (160 + 155 + 168 + 148 + 172) / 5 = 803 / 5 = 160.6 mmHg
Delta (new minus standard) = 157.0 - 160.6 = -3.6 mmHg
```

The new-protocol group started 3.6 mmHg lower. Because higher baseline pressure predicts a larger reduction, this imbalance makes the new protocol look worse than it is.

### 6.4 The regression

The model is

```text
change_i = beta0 + beta1 × protocol_i + beta2 × baseline_i + error_i
```

The normal equations are built from these sums:

```text
n = 10
sum(protocol) = 5
sum(baseline) = 1588
sum(baseline^2) = 253090
sum(protocol × baseline) = 785
sum(change) = -145
sum(protocol × change) = -87
sum(baseline × change) = -23339
```

They are:

```text
10 b0 +    5 b1 +   1588 b2 =  -145
 5 b0 +    5 b1 +    785 b2 =   -87
1588 b0 + 785 b1 + 253090 b2 = -23339
```

The solution is

```text
b0 = 54.808
b1 = -7.289 mmHg
b2 = -0.413 mmHg per mmHg
```

The fitted model is

```text
change = 54.808 - 7.289 × protocol - 0.413 × baseline
```

### 6.5 Interpretation and arithmetic checks

**Treatment coefficient.** Holding baseline SBP fixed, the new protocol is associated with a 7.289 mmHg greater reduction (more negative change) than standard care. The unit is mmHg of 12-week change per protocol difference of 1.

**Baseline coefficient.** Each additional 1 mmHg of baseline SBP is associated with a 0.413 mmHg greater reduction, holding protocol fixed. This is regression to the mean plus a real treatment-response relationship; the data cannot separate the two, which is why the coefficient is a descriptive adjustment term, not a causal effect of baseline pressure.

**Confounding arithmetic.** The change from the unadjusted estimate to the adjusted estimate should equal beta2 × delta:

```text
beta2 × delta = (-0.413) × (-3.6) = +1.487 mmHg
Adjusted = unadjusted - beta2 × delta = -5.8 - 1.487 = -7.287 mmHg
```

This matches the fitted -7.289 to rounding. This check is one of the most useful in applied regression: if the adjusted and unadjusted estimates differ by an amount that does not match the covariate imbalance, a coding error or a different sample is present.

**Residual variance and fit.**

```text
Residual sum of squares (SSE) = 13.391
Degrees of freedom = 10 - 3 = 7
Residual standard error = sqrt(13.391 / 7) = sqrt(1.913) = 1.383 mmHg
Total sum of squares (SST) = 248.5
R^2 = 1 - 13.391 / 248.5 = 0.946
```

An R² of 0.946 is high for a biological outcome, but this is a pilot with a strong baseline predictor and a narrow case mix. It is not evidence that the model is correct.

**Standard errors and interval.**

```text
SE(b1) = 0.891 mmHg
t(0.975, 7) = 2.365
95% CI = -7.289 +/- 2.365 × 0.891 = (-9.395, -5.182) mmHg
```

The interval excludes zero and also excludes -5 mmHg: the upper limit is -5.18. If the minimal clinically important difference is 5 mmHg, the data are compatible only with effects larger than 5 mmHg in this pilot.

### 6.6 Diagnostic checks

**Check 1. Residuals sum to zero and the mean residual is zero.** The residuals are approximately (-0.19, 0.51, 0.71, -1.80, 0.78, -0.65, 2.28, -0.34, 0.39, -1.69). Their sum is approximately 0.00. A non-zero sum indicates that the intercept is wrong or that the model was fitted without one.

**Check 2. Leverage and influence.** The maximum leverage is 0.455 (patient 4, baseline 142 in the new arm). Cook's distance is 0.866 for patient 4, below the conventional threshold of 1 but large enough to warrant attention. No single observation exceeds 1.

**Check 3. Collinearity.** The correlation between protocol and baseline is -0.188. Regressing protocol on baseline gives R² = 0.035, so VIF = 1/(1 - 0.035) = 1.04. Collinearity is not a problem. If VIF were above 5, the treatment coefficient would be unstable and its interval would widen.

**Check 4. Functional form.** A quadratic term in baseline adds one parameter and reduces the SSE by less than 1%; the coefficient is not distinguishable from zero. The linear form is adequate for this range. This is a weak test with n = 10 and should be reported as such.

**Check 5. Sign and scale.** The treatment coefficient is negative (benefit), the baseline coefficient is negative (higher baseline, greater reduction), and both are in mmHg. A positive baseline coefficient would suggest that the change score was coded with the wrong sign, a common data error.

### 6.7 Vary one assumption: patient 4 is a non-responder

Patient 4 had a baseline of 142 mmHg and a change of -13 mmHg, which is a smaller improvement than the model predicts (-10.41 predicted, residual -1.80). Suppose the recorded value should have been -2 mmHg: the patient adhered poorly and the protocol had little effect. This is a single-data-point change, but it is exactly the kind of event that occurs in a pilot.

Refitting:

```text
b0 = 84.811
b1 = -5.761 mmHg
b2 = -0.600 mmHg per mmHg
SE(b1) = 1.534 mmHg
95% CI = -5.761 +/- 2.365 × 1.534 = (-9.39, -2.13) mmHg
Unadjusted difference = -3.6 mmHg
```

The point estimate falls from -7.29 to -5.76 mmHg, the interval widens from (-9.40, -5.18) to (-9.39, -2.13), and the conclusion changes at the 5 mmHg threshold: the interval now includes effects smaller than 5 mmHg. The confounding arithmetic still holds: beta2 × delta = (-0.600) × (-3.6) = +2.16, and -3.6 - 2.16 = -5.76.

This is not a reason to remove patient 4. It is a reason to report the influence analysis and to avoid claiming that the effect exceeds 5 mmHg on the basis of ten patients. The design, not the arithmetic, is the limitation.

### 6.8 What to report

Report the adjusted estimate as the primary result: **-7.3 mmHg (95% CI -9.4 to -5.2)**, with the model, the adjustment set (baseline SBP), and the influence analysis. Report the unadjusted estimate (**-5.8 mmHg**) and the one-patient sensitivity (**-5.8 mmHg, 95% CI -9.4 to -2.1**) alongside it.

The adjusted estimate is primary because baseline SBP is a confounder by design: it differs between groups and predicts the outcome. The unadjusted estimate is not wrong; it answers a different question and is biased by the imbalance. The sensitivity analysis is not optional, because with n = 10 a single record changes the conclusion relative to the clinical threshold.

The honest conclusion: "In this ten-patient pilot, the new protocol was associated with a 7.3 mmHg greater reduction in SBP after adjustment for baseline SBP. The estimate is sensitive to a single patient and the confidence interval includes effects below the 5 mmHg clinical threshold. A larger study with pre-specified adjustment is required before the protocol is adopted."

## 7. Code, query, or calculation

The script fits the linear model by normal equations, a logistic model by iteratively reweighted least squares (IRLS), and a Poisson model with an offset by the same method. It uses only the standard library. The logistic and Poisson examples are simulated with fixed seeds so the output is reproducible.

**Assumptions:**

- The linear model assumes independent errors with constant variance.
- The logistic model assumes a Bernoulli outcome and a correctly specified linear predictor on the logit scale.
- The Poisson model assumes a log link, variance equal to the mean, and a correctly specified offset.
- The standard errors are model-based; no clustering is present in these examples.
- The simulated true coefficients are 0.8 and 0.5 on the log-odds scale for the logistic model, and 0.4 on the log-rate scale for the Poisson model.

```python
import math, random
from statistics import NormalDist

ND = NormalDist()

def solve(A, b):
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        d = M[c][c]
        M[c] = [v / d for v in M[c]]
        for r in range(n):
            if r != c and M[r][c] != 0.0:
                f = M[r][c]
                M[r] = [a - f * x for a, x in zip(M[r], M[c])]
    return [M[i][n] for i in range(n)]

def inverse(A):
    n = len(A)
    cols = []
    for j in range(n):
        e = [0.0] * n
        e[j] = 1.0
        cols.append(solve(A, e))
    return [[cols[j][i] for j in range(n)] for i in range(n)]

def ols(X, y):
    n, p = len(X), len(X[0])
    XtX = [[sum(X[i][a] * X[i][b] for i in range(n)) for b in range(p)] for a in range(p)]
    Xty = [sum(X[i][a] * y[i] for i in range(n)) for a in range(p)]
    inv = inverse(XtX)
    beta = [sum(inv[a][b] * Xty[b] for b in range(p)) for a in range(p)]
    res = [y[i] - sum(X[i][a] * beta[a] for a in range(p)) for i in range(n)]
    s2 = sum(r * r for r in res) / (n - p)
    se = [math.sqrt(s2 * inv[a][a]) for a in range(p)]
    ybar = sum(y) / n
    r2 = 1 - sum(r * r for r in res) / sum((v - ybar) ** 2 for v in y)
    return beta, se, r2

def irls(X, y, family, offset=None, maxit=50, tol=1e-10):
    n, p = len(X), len(X[0])
    beta = [0.0] * p
    if offset is None:
        offset = [0.0] * n
    for _ in range(maxit):
        eta = [sum(X[i][a] * beta[a] for a in range(p)) + offset[i] for i in range(n)]
        if family == "logistic":
            mu = [1 / (1 + math.exp(-e)) for e in eta]
            w = [m * (1 - m) for m in mu]
        else:
            mu = [math.exp(e) for e in eta]
            w = mu[:]
        z = [eta[i] - offset[i] + (y[i] - mu[i]) / w[i] for i in range(n)]
        XtWX = [[sum(X[i][a] * w[i] * X[i][b] for i in range(n)) for b in range(p)] for a in range(p)]
        XtWz = [sum(X[i][a] * w[i] * z[i] for i in range(n)) for a in range(p)]
        new = solve(XtWX, XtWz)
        if max(abs(new[a] - beta[a]) for a in range(p)) < tol:
            beta = new
            break
        beta = new
    eta = [sum(X[i][a] * beta[a] for a in range(p)) + offset[i] for i in range(n)]
    mu = [1 / (1 + math.exp(-e)) for e in eta] if family == "logistic" else [math.exp(e) for e in eta]
    w = [m * (1 - m) for m in mu] if family == "logistic" else mu[:]
    XtWX = [[sum(X[i][a] * w[i] * X[i][b] for i in range(n)) for b in range(p)] for a in range(p)]
    inv = inverse(XtWX)
    se = [math.sqrt(inv[a][a]) for a in range(p)]
    return beta, se

# --- OLS: 12-week SBP change, 10 patients ---
T = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
b = [158, 150, 165, 142, 170, 160, 155, 168, 148, 172]
y = [-18, -14, -20, -13, -22, -12, -7, -15, -6, -18]
X = [[1.0, float(t), float(bb)] for t, bb in zip(T, b)]
beta, se, r2 = ols(X, y)
print("OLS beta:", [round(v, 3) for v in beta])
print("OLS se:  ", [round(v, 3) for v in se])
tcrit = 2.365
print(f"treatment effect {beta[1]:.3f} mmHg (95% CI {beta[1]-tcrit*se[1]:.3f} to {beta[1]+tcrit*se[1]:.3f}), R2={r2:.3f}")
print(f"unadjusted difference {sum(y[:5])/5 - sum(y[5:])/5:.3f} mmHg")

# --- Logistic: 2000 simulated patients, true log-OR 0.8 and 0.5 ---
random.seed(11)
XL, yL = [], []
for _ in range(2000):
    x1 = random.gauss(0, 1)
    x2 = 1.0 if random.random() < 0.4 else 0.0
    pr = 1 / (1 + math.exp(-(-1.0 + 0.8 * x1 + 0.5 * x2)))
    XL.append([1.0, x1, x2])
    yL.append(1 if random.random() < pr else 0)
bl, sl = irls(XL, yL, "logistic")
print("logistic beta:", [round(v, 3) for v in bl])
print("logistic OR:  ", [round(math.exp(v), 3) for v in bl])
print("logistic 95% CI x2:", round(math.exp(bl[2]-1.96*sl[2]), 3), round(math.exp(bl[2]+1.96*sl[2]), 3))

# marginal standardization: average predicted risk if everyone treated vs untreated
def mean_risk(beta, x2val):
    s = 0.0
    for row in XL:
        eta = beta[0] + beta[1] * row[1] + beta[2] * x2val
        s += 1 / (1 + math.exp(-eta))
    return s / len(XL)
r1, r0 = mean_risk(bl, 1.0), mean_risk(bl, 0.0)
print(f"standardized risks treated={r1:.3f} untreated={r0:.3f} RD={r1-r0:.3f}")

# --- Poisson: 200 simulated counts with offset ---
random.seed(7)
XP, yP, off = [], [], []
for _ in range(200):
    x = random.gauss(0, 1)
    t = random.choice([1.0, 2.0, 4.0])
    lam = t * math.exp(0.5 + 0.4 * x)
    # Knuth sampler
    L, k, prod = math.exp(-lam), 0, 1.0
    while True:
        prod *= random.random()
        if prod <= L:
            break
        k += 1
    XP.append([1.0, x])
    yP.append(k)
    off.append(math.log(t))
bp, sp = irls(XP, yP, "poisson", offset=off)
print("poisson beta:", [round(v, 3) for v in bp])
print("poisson rate ratio:", [round(math.exp(v), 3) for v in bp])
print("poisson 95% CI x:", round(math.exp(bp[1]-1.96*sp[1]), 3), round(math.exp(bp[1]+1.96*sp[1]), 3))
```

**Expected output:**

```text
OLS beta: [54.808, -7.289, -0.413]
OLS se:   [7.5, 0.891, 0.047]
treatment effect -7.289 mmHg (95% CI -9.395 to -5.182), R2=0.946
unadjusted difference -5.800 mmHg
logistic beta: [-1.11, 0.78, 0.574]
logistic OR:   [0.33, 2.181, 1.775]
logistic 95% CI x2: 1.45 2.175
standardized risks treated=0.382 untreated=0.270 RD=0.112
poisson beta: [0.47, 0.371]
poisson rate ratio: [1.6, 1.449]
poisson 95% CI x: 1.351 1.555
```

**What the output demonstrates:**

- The OLS block reproduces the worked example exactly, including the unadjusted difference and the R². If a reader's own software gives different coefficients on the same ten rows, the data or the coding of the treatment indicator differs.
- The logistic coefficients recover the simulated truth approximately: 0.78 versus 0.80 for the continuous covariate and 0.574 versus 0.50 for the binary covariate. The second estimate is within sampling error but not close; with 2,000 patients, the standard error on that coefficient is about 0.10, so the estimate is about 0.7 standard errors from the truth.
- The standardized risk difference is 0.112, which is 11.2 percentage points. The odds ratio of 1.775 for the binary covariate does not translate into a risk ratio of 1.775; the marginal risks are 0.382 and 0.270, a risk ratio of 1.41. This is the conditional-versus-marginal distinction in one line of output.
- The Poisson block recovers a rate ratio of 1.449 against a simulated truth of 1.492. The offset is what makes the coefficient a rate ratio rather than a count ratio. Removing the offset would change the intercept and the interpretation, not the slope, when exposure is independent of the covariate.
- The script does not compute p-values. That is deliberate: the interval and the effect size carry the decision-relevant information, and the p-value adds nothing to this demonstration.

> **Note:** The IRLS function uses a fixed iteration limit and a tolerance. If a model fails to converge, the function returns the last iterate without a warning. Production code should raise an error or return a convergence flag, and the analyst should inspect it.

## 8. Data, standards, and regulatory context

### 8.1 Binding requirements and conventions

| Instrument | Status | What it constrains | What it does not constrain |
|---|---|---|---|
| International Council for Harmonisation (ICH) E9, Statistical Principles for Clinical Trials (1998) | Binding through regional adoption | Pre-specified analysis, covariates, subgroups, missing values | The choice of model family |
| ICH E9(R1) estimand addendum (2019) | Binding through regional adoption | The estimand that the model must target | The regression specification |
| FDA guidance, Adjusting for Covariates in Randomized Clinical Trials (May 2023) | FDA guidance | Pre-specification, inclusion of stratification variables, robust standard errors, conditional versus marginal estimands | Observational study design |
| Consolidated Standards of Reporting Trials (CONSORT) 2010, items 12a, 12b, 17a, 17b | Journal editorial policy | Reporting of statistical methods, adjusted analyses, effect sizes with precision, absolute and relative effects | The analysis itself |
| Strengthening the Reporting of Observational Studies in Epidemiology (STROBE) and REporting of studies Conducted using Observational Routinely-collected health Data (RECORD) | Journal editorial policy | Reporting of observational studies and routinely collected data | Model selection |
| TRIPOD+AI (2024) | Journal editorial policy | Reporting of prediction models, including calibration and subgroup performance | Effect estimation |
| NIST/SEMATECH e-Handbook | Reference, not binding | Statistical methods and worked examples | Clinical interpretation |

### 8.2 What the FDA covariate guidance changes

The May 2023 final guidance makes four points that affect model specification:

1. **Unadjusted analyses remain acceptable.** Adjustment is an efficiency choice, not a requirement.
2. **Adjustment should be prospective.** Covariates and the model should be specified before unblinding. Adjusting for a covariate chosen because it is unbalanced in the observed data inflates the Type I error rate.
3. **Include randomization strata.** Stratification variables are prognostic by design and should be in the model.
4. **Use robust standard errors for linear models.** The treatment coefficient from OLS remains valid under model misspecification of the covariate effects (a result associated with Lin, 2013), but the usual standard error can be wrong if the variance is misspecified.

The guidance also addresses nonlinear models. It notes that odds ratios and hazard ratios are non-collapsible, distinguishes conditional from unconditional estimands, and recommends model-robust estimators such as standardization or inverse probability weighting when a marginal effect is wanted. This is the regulatory version of the conditional-versus-marginal distinction in Section 3.2.

### 8.3 What TRIPOD+AI requires for prediction models

The TRIPOD+AI statement, published in the BMJ in April 2024, replaces the 2015 TRIPOD statement. It contains a 27-item checklist with 52 subitems and applies to regression and machine-learning prediction models alike. Items relevant to this topic:

- Report the intended use and the target population.
- Report the outcome definition, the predictor definitions, and the timing of measurement.
- Report the sample size and the number of events.
- Report calibration, not only discrimination.
- Report performance in key sociodemographic subgroups.
- Report the model equation or the code, and the model's availability.

A prediction model reported without calibration is incomplete under TRIPOD+AI. A model reported with an AUC and no calibration slope cannot be deployed safely, because a model can rank patients correctly and still assign the wrong absolute risk to everyone.

### 8.4 Standards for effect models

There is no equivalent of TRIPOD for effect models, but CONSORT items 12a and 12b require the statistical methods and any adjusted analyses to be reported, and item 17a requires effect sizes with precision. STROBE item 16 requires the same for observational studies. The common regulatory expectation is that the primary effect model is pre-specified, that the adjustment set is justified, and that sensitivity analyses address the main threats.

### 8.5 What the standards do not cover

- They do not validate the adjustment set. A well-reported model can still adjust for a collider.
- They do not require calibration for effect models, although calibration of standardized predictions is a useful diagnostic.
- They do not specify how to handle separation, overdispersion, or influential observations.
- They do not resolve the choice between a conditional and a marginal estimand; they require the choice to be stated.

### 8.6 Documentation checklist for a regulated or audited model

1. The estimand, in one sentence.
2. The model family, link, and estimator, with software and version.
3. The adjustment set and the DAG or design justification.
4. The functional form of every continuous covariate.
5. The pre-specified interactions and the reference categories.
6. The number of observations, events, and parameters.
7. The convergence status and the iteration count.
8. The diagnostics, including those that failed.
9. The effect on the model scale and the decision scale.
10. The validation design and performance.
11. The sensitivity analyses and what each one tests.
12. The code or a reproducible specification of the model.

Items 1 to 6 are required for a reader to reproduce the estimate. Items 7 to 11 are required for a reader to judge it. Item 12 is required for an auditor to rerun it.

## 9. Industry practice and case evidence

### 9.1 Documented practice: risk functions in clinical use

The Framingham risk functions are among the most widely deployed regression models in medicine. The 1998 paper by Wilson and colleagues in Circulation derived sex-specific models from the Framingham cohort and published score sheets for 10-year coronary heart disease risk. The final models were based on Cox proportional hazards coefficients converted to absolute risk, not on logistic regression, although Framingham models had historically used the logistic function. Reported c-statistics were approximately 0.73 to 0.74 in men and 0.76 to 0.77 in women for the categorical and continuous models.

Two lessons follow. First, the model family is a technical choice that must be reported; assuming "Framingham is logistic" is a factual error. Second, discrimination around 0.75 is useful for ranking but does not establish calibration in a new population. The 2001 validation by D'Agostino and colleagues in JAMA evaluated the functions across several cohorts and found that calibration varied by population, which is why recalibration is standard practice.

### 9.2 Documented practice: covariate adjustment in trials

The FDA guidance describes adjustment as a way to narrow intervals and increase power when covariates are strongly prognostic. The statistical basis includes the result that the OLS treatment coefficient remains consistent under misspecification of the covariate model, and the recommendation to use robust standard errors follows from the possibility that the variance model is wrong. The guidance does not claim that adjustment repairs confounding in observational data; its scope is randomized trials.

### 9.3 Documented case: emulation with regression and propensity scores

The RCT-DUPLICATE initiative emulated 32 trials using claims data and reported a correlation of 0.82 between emulated and randomized effect estimates, rising to 0.93 in the 16 trials with close emulation. The emulations used propensity score matching and Cox models, which illustrates a general industry pattern: regression is rarely used alone for causal effect estimation in observational data; it is combined with a design step that creates comparability.

**Inference:** the correlation is not a validation of any single regression specification. It is an average over trials, databases, and design choices, and it does not tell an analyst whether their particular model is adequate.

### 9.4 Practice pattern: prediction versus explanation

Mature analytics teams separate the two model types in their documentation:

- A **prediction model** has a target population, a prediction horizon, a calibration report, a discrimination report, and a deployment threshold. It may include variables that are not causal.
- An **effect model** has an estimand, an adjustment set justified by a DAG, a pre-specified functional form, and a sensitivity analysis. It does not need an AUC, and its R² is not evidence of correctness.

**Inference:** organizations that use one model for both purposes tend to report an AUC for an effect question or a coefficient for a prediction question. Both reports are incomplete.

### 9.5 Case evidence: model drift after deployment

Prediction models degrade when the case mix, coding practice, or treatment pathways change. A model trained on pre-pandemic admissions and deployed during a period of changed admission thresholds will see a different distribution of predictors and outcomes. This is documented in the general literature on model drift and is the reason TRIPOD+AI requires the intended use and the deployment context to be reported. It is not a defect of regression; it is a defect of treating a fitted model as a permanent asset.

### 9.6 What mature teams do differently

1. They write the estimand and the adjustment set before fitting.
2. They separate prediction models from effect models in the analysis plan.
3. They report calibration for prediction and influence analysis for effects.
4. They pre-specify the functional form of the strongest covariates.
5. They fit a design-based estimator as a check on the regression estimate.
6. They document the software, version, and convergence status.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Adjusting for a collider | A covariate is measured after treatment and predicts the outcome | Conditioning on a common effect induces association | Spurious effect, sometimes in the wrong direction | Draw the DAG; exclude descendants of treatment |
| Adjusting for a mediator | The treatment coefficient shrinks when a post-treatment variable is added | The indirect path is blocked | Total effect underestimated | Define the estimand as total or direct before analysis |
| Odds ratio read as risk ratio | "Two times more likely" from an OR of 2.0 with a common outcome | Non-collapsibility and scale confusion | Benefit overstated | Report marginal risks and RD |
| Stepwise selection | The final model depends on the sample | Selection on associations, including colliders | Invalid intervals; biased effects | Pre-specify the adjustment set |
| Overfitting | Many covariates, few events, impressive in-sample R² | Variance inflation and optimistic fit | Poor out-of-sample performance | Events-per-variable check; penalization; validation |
| Separation | A coefficient above 8 with a huge SE | Perfect prediction in a subgroup | Unstable or infinite estimate | Firth; collapse categories; exact methods |
| Overdispersion ignored | Poisson deviance much larger than its degrees of freedom | Variance underestimated | Intervals too narrow; false significance | Negative binomial or robust SE |
| Collinearity | Coefficients flip sign when a variable is added | Near-linear dependence among covariates | Unstable and uninterpretable coefficients | VIF; combine or drop variables |
| Extrapolation | Predictions at covariate values outside the observed range | The model is only supported where data exist | Wrong predictions and effects | Report the range; truncate predictions |
| No calibration | AUC reported, no calibration plot | Discrimination and calibration are different | Wrong absolute risks in deployment | Calibration slope and intercept |
| Influential observation | One patient changes the estimate | Small n or a high-leverage point | Fragile conclusion | Cook's D; report with and without |
| Wrong link | Predicted probabilities outside 0 to 1, or negative counts | Identity link on a bounded outcome | Impossible predictions | Choose the link from the outcome type |
| Ignoring clustering | Naive SE smaller than a site-level analysis | Independence assumed | Interval too narrow | Cluster-robust or mixed model |
| Missing data handled by deletion | Complete-case n much smaller than the cohort | Selection on completeness | Bias and power loss | Multiple imputation with the outcome included |
| Model reported without the equation | The reader cannot reproduce the predictions | Incomplete reporting | Unusable model | Report coefficients or code |
| Offset omitted | A rate model is interpreted as a count model | Exposure time varies but is not in the model | Wrong rate ratio and wrong predictions | Include log exposure with coefficient fixed at 1 |
| R² compared across families | "The logistic model has a lower R²" | Pseudo-R² is not comparable across links | Wrong model choice | Compare calibration and out-of-sample performance |

The last row is a reporting error rather than an estimation error, but it changes the model that gets deployed. A fit statistic is only comparable within the same family and the same outcome definition.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| Correct functional form | The coefficient is the effect only at the specified shape | Nonlinearity, thresholds, interactions | Residual plots; spline comparison; global test |
| Correct link | The scale of the coefficient depends on the link | A bounded outcome modelled with an identity link | Predicted values within range; calibration |
| Independent observations | Standard errors assume independence | Clustering, repeated measures | Compare naive and cluster-robust SE |
| No unmeasured confounding | The coefficient is causal only if the adjustment set is sufficient | An unmeasured common cause | DAG; E-value; negative control |
| No mediator adjustment for a total effect | The estimand includes the indirect path | A post-treatment variable in the model | DAG; compare models with and without |
| Positivity | Every covariate pattern has both treatment groups | A subgroup receives only one treatment | Overlap; predicted propensity range |
| Correct outcome measurement | The endpoint is what the model thinks it is | Differential coding by group | Validation sample; negative control outcome |
| Constant variance (OLS) | Standard errors are valid | Heteroscedasticity | Residual versus fitted; robust SE |
| Mean equals variance (Poisson) | Standard errors are valid | Overdispersion | Dispersion ratio |
| Proportional hazards (Cox) | The hazard ratio is constant | Time-varying effects | Schoenfeld residuals |
| No influential outliers | The estimate represents the population | One record drives the fit | Cook's D; influence analysis |
| Model stability over time | The model transports to deployment | Case-mix or coding drift | Temporal validation; monitoring |
| Correct offset | A count model estimates a rate, not a count | Exposure time omitted or mis-scaled | Compare predicted with observed counts by exposure stratum |
| No differential measurement error | The coefficient reflects the true covariate | One group's covariate is measured more often | Validation subsample; chart review |
| Stable case mix | The adjustment applies to the target population | Eligibility or referral patterns change | Compare covariate distributions over time |

The last three rows are the ones most often omitted from a regression report. They are not testable with the fitted model alone, which is why external validation and temporal comparison exist. A model that passes every internal diagnostic can still fail the first month after deployment.

**Limits of this topic.** Regression estimates associations under a model. The causal interpretation requires the identification assumptions in Topic 05. The methods here are for independent observations; clustered and longitudinal designs need the extensions in the references for this topic and in Topic 03. Penalized methods change the estimand and should not be used for effect estimation without a clear justification. The worked example has ten patients and is a teaching device, not a basis for a clinical claim.

## 12. Exercises

### Exercise 1. Choose the family and the link (recall and application, 40 minutes)

For each outcome, name the family, the link, the coefficient scale, and one diagnostic:

1. Number of emergency department visits in 12 months.
2. Binary indicator of 30-day readmission.
3. Length of stay in days, right-skewed.
4. Time to discontinuation.
5. Cost in USD, with many zero-cost patients.
6. Ordinal pain score (none, mild, moderate, severe).

**Deliverable.** A six-row table.

**Acceptance criteria.**

- Each family is matched to the outcome's range and mean-variance relationship.
- Each coefficient scale is named precisely (log rate ratio, log odds ratio, and so on).
- The diagnostic is specific to the family.
- The zero-cost case is handled explicitly, not by ignoring the zeros.

### Exercise 2. Convert an odds ratio to a risk difference (application, 40 minutes)

A logistic model gives an adjusted odds ratio of 2.4 for a treatment, with a control risk of 20%.

**Deliverable.** The implied treated risk, the risk ratio, the risk difference, and a sentence explaining why the odds ratio overstates the risk ratio. Then repeat with a control risk of 2% and compare.

**Acceptance criteria.**

- The conversion uses odds = p/(1-p) and the inverse transformation.
- The arithmetic is shown.
- The comparison between the 20% and 2% settings is quantified.
- The answer states that the conversion assumes the odds ratio is the causal effect and that the control risk is known.

### Exercise 3. Diagnose overdispersion (application, 35 minutes)

A Poisson model of admissions has a deviance of 640 on 200 degrees of freedom.

**Deliverable.** The dispersion ratio, the standard error multiplier, and a recommendation. Then state what would change if the model were negative binomial with an estimated dispersion of 0.4.

**Acceptance criteria.**

- The dispersion ratio is computed as deviance/df.
- The SE multiplier is the square root of the ratio.
- The recommendation names a specific alternative and a reporting consequence.
- The answer does not claim that overdispersion biases the point estimate.

### Exercise 4. Standardize a logistic model (application, 60 minutes)

A logistic model has a treatment coefficient of 0.5 and a covariate coefficient of 0.8. The observed covariate is binary with 40% prevalence. Ignore the intercept for the purpose of the exercise and assume the intercept is -1.0.

**Deliverable.** Predicted risks for treated and untreated patients in each covariate stratum, the average predicted risk under each treatment, the marginal risk difference, and the marginal risk ratio.

**Acceptance criteria.**

- The four stratum-specific risks are shown.
- The averages are weighted by the 40/60 prevalence.
- The marginal RD and RR are computed from the averages.
- The answer explains why the marginal odds ratio is closer to 1 than exp(0.5).

### Exercise 5. Events per variable (application, 30 minutes)

A logistic model has 8 covariates and 45 events.

**Deliverable.** The events-per-variable ratio, an assessment against the 10-events rule, and two concrete remedies with their costs.

**Acceptance criteria.**

- The ratio is computed correctly.
- The answer distinguishes a heuristic from a requirement.
- Each remedy states what is lost (variables, sample, or interpretability).
- The answer mentions Firth penalization as an alternative.

### Exercise 6. Find the collider (critique, 50 minutes)

A model of 30-day mortality after surgery adjusts for treatment, age, comorbidity, and "post-operative complications." The treatment coefficient is protective in the unadjusted analysis and harmful after adjustment.

**Deliverable.** A DAG, an explanation of the reversal, and a corrected specification.

**Acceptance criteria.**

- The DAG marks post-operative complications as a descendant of treatment.
- The explanation uses the collider or mediator mechanism, not "confounding."
- The corrected model excludes the post-treatment variable.
- The answer states which estimand the corrected model targets.

### Exercise 7. Influence analysis (application, 45 minutes)

Removing one patient changes a treatment coefficient from -7.3 to -5.5 mmHg and widens the interval from (-9.4, -5.2) to (-9.0, -2.0). The patient has a Cook's distance of 1.4.

**Deliverable.** A recommendation on whether to keep or exclude the patient, the wording for a results paragraph, and the additional analysis you would run.

**Acceptance criteria.**

- The recommendation does not exclude the patient solely because the result weakens.
- The paragraph reports both estimates and the influence statistic.
- The additional analysis is specific (for example, a measurement review or a robust regression).
- The answer distinguishes a data error from a genuine high-leverage observation.

### Exercise 8. Write a regression analysis plan (design, 120 minutes)

**Scenario.** An observational comparison of two diabetes drugs, with claims data, 12-month HbA1c change as the outcome, and 14 candidate covariates.

**Deliverable.** A two-page plan containing the estimand, the DAG-derived adjustment set, the family and link, the functional forms, the pre-specified interactions, the diagnostics, the sensitivity analyses, and the reporting table shell.

**Acceptance criteria.**

- The estimand names the population, contrast, outcome, time zero, and summary.
- Each covariate is classified as confounder, mediator, collider, or predictor, with a reason.
- The family and link match the outcome.
- At least two sensitivity analyses are tied to named assumptions.
- The table shell includes the effect, the interval, and the scale.

## 13. Self-check answers

**Answer 1 (Exercise 2 conversion).**

```text
Control risk p0 = 0.20; odds0 = 0.20/0.80 = 0.25
Treated odds = 2.4 × 0.25 = 0.60
Treated risk p1 = 0.60/1.60 = 0.375
RR = 0.375/0.20 = 1.875
RD = 0.375 - 0.20 = 0.175 = 17.5 percentage points
```

At a control risk of 2%:

```text
odds0 = 0.02/0.98 = 0.02041
odds1 = 2.4 × 0.02041 = 0.04898
p1 = 0.04898/1.04898 = 0.04669
RR = 0.04669/0.02 = 2.335
RD = 0.04669 - 0.02 = 0.02669 = 2.67 percentage points
```

At a control risk of 20%, the odds ratio of 2.4 corresponds to a risk ratio of 1.875, an overstatement of 28%. At a control risk of 2%, it corresponds to a risk ratio of 2.335, close to the odds ratio. The odds ratio is not wrong; it is a ratio of odds, and the gap from the risk ratio depends on the baseline risk.

**Answer 2 (Exercise 3 overdispersion).**

```text
Dispersion ratio = 640 / 200 = 3.2
SE multiplier = sqrt(3.2) = 1.789
```

The point estimates are unchanged under quasi-Poisson or negative binomial; the standard errors are about 79% larger. An interval that was (-0.10, -0.02) on the log scale becomes approximately (-0.18, +0.06), which may cross zero. The recommendation is a negative binomial model or quasi-Poisson with robust standard errors, and a report that states the dispersion estimate. With a negative binomial dispersion of 0.4, the variance is μ + 0.4μ², and the SE multiplier depends on the mean; it is not a single number.

**Answer 3 (Exercise 4 standardization).**

```text
Intercept = -1.0; treatment coefficient = 0.5; covariate coefficient = 0.8.

Covariate = 0 (60%):
  untreated: logit = -1.0;            p = 0.269
  treated:   logit = -0.5;            p = 0.378
Covariate = 1 (40%):
  untreated: logit = -0.2;            p = 0.450
  treated:   logit =  0.3;            p = 0.574

Average untreated = 0.6 × 0.269 + 0.4 × 0.450 = 0.161 + 0.180 = 0.341
Average treated   = 0.6 × 0.378 + 0.4 × 0.574 = 0.227 + 0.230 = 0.457
Marginal RD = 0.116 = 11.6 percentage points
Marginal RR = 0.457/0.341 = 1.34
Conditional OR = exp(0.5) = 1.649
```

The marginal odds ratio implied by these risks is (0.457/0.543)/(0.341/0.659) = 0.842/0.517 = 1.63, close to the conditional 1.65 in this example because the covariate effect is moderate and the baseline risk is not extreme. With a stronger covariate and a more common outcome, the marginal odds ratio moves closer to 1 while the conditional odds ratio stays fixed.

**Answer 4 (Exercise 5 events per variable).**

```text
EPV = 45 / 8 = 5.6
```

This is below the conventional 10-events rule. The rule is a heuristic from simulation studies, not a mathematical requirement, but 5.6 is low enough that the estimates are likely biased away from the null and the intervals too narrow. Remedies: reduce the model to 4 covariates (EPV 11.25), which loses adjustment for possibly important confounders; collect more events, which is usually impossible retrospectively; or use Firth penalization, which reduces small-sample bias but changes the estimator and requires reporting. A propensity score that combines the 8 covariates into one variable is a fourth option, with its own assumptions.

**Answer 5 (Exercise 6 collider).**

The unadjusted analysis shows protection; the adjusted analysis shows harm. Post-operative complications are caused by the treatment (the treatment reduces complications) and are also a cause of mortality. Adjusting for a common effect of treatment and outcome creates a spurious association between treatment and the unobserved causes of mortality, which can reverse the sign. The corrected model excludes post-operative complications and targets the total effect of treatment on mortality. If the question is the direct effect, a mediation analysis is required, with its own assumptions; the naive adjusted model is not that analysis.

**Answer 6 (Exercise 7 influence).**

The patient should not be excluded solely because the estimate weakens. Cook's distance of 1.4 is above the conventional threshold and warrants a data review. If the record is correct, the patient is a genuine high-leverage observation and the primary analysis should retain it, with a sensitivity analysis that reports the estimate without it. The results paragraph: "The adjusted estimate was -7.3 mmHg (95% CI -9.4 to -5.2). One patient had a Cook's distance of 1.4; excluding that patient gave -5.5 mmHg (95% CI -9.0 to -2.0). The conclusion at the 5 mmHg threshold is sensitive to this observation." The additional analysis is a measurement review of the record and a robust regression or a bootstrap of the coefficient.

**Answer 7 (Exercise 1 table).**

| Outcome | Family | Link | Coefficient scale | Diagnostic |
|---|---|---|---|---|
| ED visits | Negative binomial (check dispersion first) | Log | Log rate ratio | Dispersion ratio |
| Readmission | Bernoulli | Logit | Log odds ratio | Calibration; separation |
| Length of stay | Gamma or log-normal | Log | Ratio of geometric means | Residuals on log scale; influence |
| Time to discontinuation | Cox or parametric | Log hazard | Log hazard ratio | Schoenfeld residuals |
| Cost with zeros | Two-part model, or gamma with log link after handling zeros | Logit plus log | Odds of any cost; cost ratio | Calibration of both parts |
| Ordinal pain | Cumulative logit | Logit | Cumulative odds ratio | Proportional odds test |

**Answer 8 (Exercise 8 rubric).**

A strong plan:

- Defines the estimand with all five attributes and a time zero at initiation of the first prescription.
- Classifies each of the 14 covariates with a reason and excludes post-initiation variables.
- Uses a Gaussian model for HbA1c change with a pre-specified spline for baseline HbA1c.
- Pre-specifies no interactions, or one interaction with a clinical rationale.
- Lists diagnostics: residual plots, influence, VIF, and a comparison with IPTW.
- Ties two sensitivity analyses to named assumptions: unmeasured confounding and informative censoring.
- Includes a table shell with the mean difference, 95% CI, and the proportion achieving target.

A weak plan lists covariates without roles, chooses the model by AIC, and reports only a p-value.

## 14. Interview questions

**1. When is a regression coefficient an effect estimate?**

A strong answer says only when the adjustment set is justified by randomization or by a causal graph and the identification assumptions hold. It distinguishes the coefficient from a marginal effect. A weak answer says "when the p-value is significant" or "when the model is adjusted."

**2. Why does the odds ratio change when you add a prognostic covariate to a randomized trial?**

A strong answer explains non-collapsibility and notes that the change is not confounding. It may quantify the difference between conditional and marginal odds ratios. A weak answer calls the change "bias" or says the model is now "more adjusted."

**3. What is the difference between a confounder and a mediator, and why does it matter for adjustment?**

A strong answer defines both, explains that adjusting for a mediator removes part of the total effect, and ties the choice to the estimand. A weak answer says both are "variables to control for."

**4. How do you detect separation, and what do you do about it?**

A strong answer describes coefficients diverging, fitted probabilities of 0 or 1, and huge standard errors; recommends Firth or exact methods. A weak answer says "the model fails" without a remedy.

**5. Why is calibration more important than discrimination for a deployed risk model?**

A strong answer explains that discrimination is ranking and calibration is the correctness of the absolute risk; a well-ranked model can still overpredict everyone. It mentions calibration slope and intercept. A weak answer says "AUC is what everyone reports."

**6. What is overdispersion, and how does it affect inference?**

A strong answer defines variance greater than the mean, notes that point estimates are usually consistent but standard errors are too small, and names negative binomial or quasi-Poisson. A weak answer says "the data are too variable" without the inferential consequence.

**7. How many covariates can you include in a logistic model?**

A strong answer refers to events per variable, the 10-events heuristic, and the alternatives (Firth, propensity score, reducing the model). It notes that the rule is a heuristic. A weak answer gives a fixed number without context.

**8. How would you choose the functional form for a continuous covariate?**

A strong answer pre-specifies from clinical knowledge, uses restricted cubic splines with a fixed number of knots, and tests nonlinearity globally rather than by moving knots. A weak answer says "check the scatterplot and pick the best fit."

**9. What is the difference between internal and external validation?**

A strong answer describes optimism correction by bootstrap or cross-validation versus evaluation in a separate population, and says what each establishes. A weak answer calls a split-sample validation "external."

**10. Why can adjusting for a post-treatment variable reverse an effect?**

A strong answer explains collider stratification or mediation, with a clinical example, and connects it to the DAG. A weak answer says "it introduces bias" without a mechanism.

**11. What would you report alongside a logistic regression odds ratio?**

A strong answer reports the marginal risk difference or risk ratio from standardization, the baseline risk, the interval, and the model diagnostics. A weak answer reports the odds ratio and the p-value.

**12. When would you use a robust standard error?**

A strong answer names heteroscedasticity, clustering, and weighted or imputed data, and notes that the point estimate is unchanged. A weak answer says "when the data are not normal."

## 15. Key terms

- **AIC:** a penalized fit statistic used to compare models.
- **Adjustment set:** the variables included in a model to block confounding paths.
- **Average marginal effect:** the mean change in predicted outcome per unit change in a covariate.
- **Calibration:** agreement between predicted and observed risks.
- **Collider:** a common effect of two variables; adjustment induces association.
- **Collinearity:** near-linear dependence among covariates.
- **Conditional effect:** an effect holding other covariates fixed.
- **Confounder:** a common cause of treatment and outcome.
- **Deviance:** a measure of fit relative to a saturated model.
- **Discrimination:** the ability to rank patients by risk.
- **Dispersion ratio:** deviance divided by degrees of freedom in a count model.
- **Events per variable:** the number of outcome events divided by the number of estimated parameters.
- **External validation:** evaluation of a model in a separate population.
- **Firth penalization:** a penalized likelihood method that reduces small-sample bias and handles separation.
- **Functional form:** the assumed shape of the relationship between a covariate and the linear predictor.
- **Generalized linear model:** a model with a linear predictor, a link, and a variance function.
- **Influence:** the effect of a single observation on the estimate.
- **Interaction:** a term allowing the effect of one variable to depend on another.
- **Internal validation:** optimism-corrected performance within the development data.
- **Link function:** the function mapping the mean outcome to the linear predictor.
- **Marginal effect:** an effect averaged over the covariate distribution.
- **Mediator:** a variable on the causal path from treatment to outcome.
- **Negative binomial:** a count model with an extra dispersion parameter.
- **Odds ratio:** the ratio of odds between groups; non-collapsible.
- **Offset:** a variable with a fixed coefficient of 1, such as log exposure time.
- **Overdispersion:** variance greater than the model's mean-variance relationship assumes.
- **Separation:** a covariate or combination that perfectly predicts the outcome.
- **Standardization:** averaging model predictions over a target covariate distribution.
- **Variance inflation factor:** 1/(1-R²) for a covariate regressed on the others.

## 16. Sources

All URLs were accessed on 25 October 2026.

1. United States Food and Drug Administration. *Adjusting for Covariates in Randomized Clinical Trials for Drugs and Biological Products: Guidance for Industry*. May 2023. https://www.fda.gov/media/148910/download
2. International Council for Harmonisation. *ICH E9: Statistical Principles for Clinical Trials*. Step 4, 5 February 1998. https://database.ich.org/sites/default/files/E9_Guideline.pdf
3. International Council for Harmonisation. *ICH E9(R1) Addendum on Estimands and Sensitivity Analysis in Clinical Trials*. Step 4, 20 November 2019. https://database.ich.org/sites/default/files/E9-R1_Step4_Guideline_2019_1203.pdf
4. Collins, G. S., Moons, K. G. M., Dhiman, P., et al. "TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods." *BMJ* 2024;385:e078378. https://doi.org/10.1136/bmj-2023-078378
5. Schulz, K. F., Altman, D. G., Moher, D., for the CONSORT Group. "CONSORT 2010 Statement." *PLoS Medicine* 2010;7(3):e1000251. https://doi.org/10.1371/journal.pmed.1000251
6. STROBE Initiative. *STROBE Statement*. https://www.strobe-statement.org/
7. NIST/SEMATECH. *e-Handbook of Statistical Methods*. https://www.itl.nist.gov/div898/handbook/
8. Wilson, P. W. F., D'Agostino, R. B., Levy, D., Belanger, A. M., Silbershatz, H., Kannel, W. B. "Prediction of Coronary Heart Disease Using Risk Factor Categories." *Circulation* 1998;97(18):1837-1847. https://doi.org/10.1161/01.CIR.97.18.1837
9. D'Agostino, R. B., Grundy, S., Sullivan, L. M., Wilson, P., for the CHD Risk Prediction Group. "Validation of the Framingham Coronary Heart Disease Prediction Scores." *JAMA* 2001;286(2):180-187. https://doi.org/10.1001/jama.286.2.180
10. Firth, D. "Bias reduction of maximum likelihood estimates." *Biometrika* 1993;80(1):27-38. https://doi.org/10.1093/biomet/80.1.27
11. Heinze, G., Schemper, M. "A solution to the problem of separation in logistic regression." *Statistics in Medicine* 2002;21(16):2409-2419. https://doi.org/10.1002/sim.1047
12. Hosmer, D. W., Lemeshow, S., Sturdivant, R. X. *Applied Logistic Regression*. 3rd edition. Wiley, 2013.
13. McCullagh, P., Nelder, J. A. *Generalized Linear Models*. 2nd edition. Chapman and Hall, 1989.
14. Wang, S. V., Schneeweiss, S., and the RCT-DUPLICATE Initiative. "Emulation of Randomized Clinical Trials With Nonrandomized Database Analyses: Results of 32 Clinical Trials." *JAMA* 2023;329(16):1376-1385. https://doi.org/10.1001/jama.2023.4221
15. Benchimol, E. I., Smeeth, L., Guttmann, A., et al. "The RECORD Statement." *PLoS Medicine* 2015;12(10):e1001885. https://doi.org/10.1371/journal.pmed.1001885

> **Note:** The FDA and ICH documents are the primary regulatory sources for Section 8. Wilson 1998 and D'Agostino 2001 are the primary sources for the Framingham example; the statement that the 1998 models were Cox-based comes from the methods section of Wilson 1998, not from secondary summaries.

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`01_estimation-and-inference.md`](01_estimation-and-inference.md) — intervals and effect scales.
- Survival models: [`03_survival-analysis.md`](03_survival-analysis.md) — Cox and parametric models.
- Causal adjustment: [`05_causal-inference-and-dags.md`](05_causal-inference-and-dags.md) — choosing the adjustment set.
- Propensity methods: [`06_propensity-scores-and-weighting.md`](06_propensity-scores-and-weighting.md) — an alternative to outcome regression.
- Reporting: [`10_reporting-statistical-results.md`](10_reporting-statistical-results.md) — presenting coefficients and marginal effects.
- Survey file: [`../../04_statistics_causal_inference.md`](../../course/04_statistics_causal_inference.md)
- Labs: [`../../labs/`](../../labs/) — Lab 4.2 compares regression, IPTW, and naive estimates.
- Foundations: [`../../01A_foundations_assumptions_and_units.md`](../../course/01A_foundations_assumptions_and_units.md) — notation and units.
