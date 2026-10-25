# Propensity Scores and Weighting

> Module 04 — Statistics and Causal Inference
> Topic 06 of 10
> Estimated study time: 4 hours
> Prerequisites: 02, 05
> Version 2026.3

---

## 1. Why this topic exists

Observational comparisons of treatments are everywhere in industry: claims analyses for formulary decisions, registry studies for device labeling, EHR cohorts for program evaluation, and real-world evidence packages for regulators. In almost every one, the treated and untreated groups differ before treatment starts. The treated patients are sicker, or healthier, or richer, or more engaged, and those differences affect the outcome.

Regression adjustment handles this by modelling the outcome as a function of treatment and covariates. The propensity score handles it by modelling the treatment assignment instead, then using that model to construct a comparison group whose covariate distribution resembles the treated group or the whole population. The two approaches answer the same question from opposite ends, and each has failure modes the other does not.

The propensity score is the probability of receiving treatment given measured covariates:

```text
e(X) = P(A = 1 | X)
```

Its central property, proved by Rosenbaum and Rubin in 1983, is that if treatment assignment is unconfounded given X, then it is also unconfounded given e(X). Conditioning on a single scalar can therefore balance the entire covariate distribution, provided the model is right and the groups overlap.

That property is powerful and easy to misuse. The common failures:

- **Balance is asserted, not shown.** A paper reports a propensity score model and an effect estimate without a balance table. The reader cannot tell whether the score did its job.
- **Extreme weights.** A patient with a propensity score of 0.01 who receives treatment gets a weight of 100. A handful of such patients can dominate the estimate, and the effective sample size collapses.
- **Positivity violations.** Some covariate strata contain only treated or only untreated patients. The effect in those strata is not estimable, and trimming changes the target population without saying so.
- **The score is treated as a confounder.** Adding the propensity score to a regression as a covariate is not the same as weighting or matching, and it can produce a biased estimate when the outcome model is nonlinear.
- **The target population is left implicit.** ATE, ATT, and overlap-weighted estimates answer different questions. A payer deciding whether to expand a program needs the effect in the population that would receive it, which may be the ATT or the overlap population, not the ATE.
- **The variance ignores the weighting.** Weighted estimates require robust or bootstrap standard errors; the model-based variance from an unweighted regression is wrong.

The costs are concrete. A formulary decision based on an inverse probability of treatment weighting (IPTW) estimate with an effective sample size of 40 patients is not supported by 4,000 records. A device registry analysis that matches each treated patient to the nearest control, discarding 60% of controls, answers a question about a selected population and reports it as the population effect. A vendor deck that says "propensity-matched" without a balance table has provided a method name, not evidence.

This topic builds the mechanics: the treatment model, the estimand and target population, matching, weighting, stratification, balance diagnostics, effective sample size, doubly robust estimation, and the sensitivity analyses that address the assumptions the score cannot test.

> **Note:** The propensity score balances measured covariates. It does not balance unmeasured ones. A perfect balance table on 30 covariates is compatible with strong residual confounding by the 31st, and the balance table cannot tell you which case you are in.

## 2. Learning outcomes

By the end of this topic you can:

1. **Define** the propensity score and **state** the balancing property and the assumptions it requires.
2. **Choose** an estimand (ATE, ATT, ATO) that matches the decision, and **explain** how the choice changes the weights.
3. **Fit** a treatment model, **assess** overlap, and **decide** whether trimming or a different target population is required.
4. **Implement** matching, inverse probability of treatment weighting (IPTW), stratification, and overlap weighting, and **state** the target of each.
5. **Compute** standardized mean differences, variance ratios, and effective sample size before and after adjustment.
6. **Estimate** a marginal effect with a weighted outcome model or standardization, and **use** a robust variance.
7. **Apply** a doubly robust estimator and **explain** the sense in which it is robust.
8. **Report** a propensity score analysis with the model, the estimand, the balance table, the weight distribution, and the sensitivity analysis.

Outcome 5 maps to Exercise 3 and the worked example. Outcome 6 maps to Exercise 4. Outcome 7 maps to Exercise 6.

## 3. The core mental model

The propensity score is a one-dimensional summary of a multi-dimensional assignment mechanism. It is used in three steps: model the assignment, use the score to create balance, and estimate the effect in the balanced sample.

```text
  COVARIATES X  (measured, pre-treatment)
        |
        |  treatment model: logistic regression of A on X
        v
  PROPENSITY SCORE e(X) = P(A = 1 | X)
        |
        +--> MATCHING: pair treated and control with similar e(X)
        +--> WEIGHTING: weight by 1/e(X) or 1/(1-e(X))
        +--> STRATIFICATION: compare within strata of e(X)
        +--> OVERLAP WEIGHTS: weight by e(X)(1-e(X))
        |
        v
  BALANCED SAMPLE  (check standardized differences, ESS)
        |
        |  outcome model or weighted difference
        v
  EFFECT IN A TARGET POPULATION
        |
        |  sensitivity: unmeasured confounding, negative control
        v
  DECISION
```

### 3.1 The balancing property

If treatment assignment is unconfounded given X, then

```text
X  independent of A  given e(X)
```

In words: among patients with the same propensity score, the treated and untreated groups have the same distribution of X, in expectation. This is why matching or weighting on the score can balance many covariates at once.

The property is an expectation. In a finite sample, balance is approximate, and it must be checked. A propensity model that is misspecified, or a sample with poor overlap, can produce a score that does not balance the covariates even though the theory says it should.

### 3.2 The estimand determines the weights

Different target populations require different weights. With e = e(X):

| Target | Weight for treated | Weight for control | Interpretation |
|---|---|---|---|
| ATE | 1 | e/(1-e) | The effect if everyone in the population were treated vs if no one were |
| ATT | 1/e | 1 | The effect in the treated population |
| ATC | 1 | 1/(1-e) | The effect in the control population |
| ATO (overlap) | 1-e | e | The effect in the population with clinical equipoise |

The ATE weights are the ones most often taught: 1/e for the treated and 1/(1-e) for the controls. The ATT weights are 1 for the treated and e/(1-e) for the controls. The ATO weights are bounded between 0 and 1 and target the overlap population.

The choice is a decision, not a technicality. A program that is being considered for expansion should be evaluated in the population that would receive it, which is closer to the ATT or the ATO than to the ATE. A regulatory question about a drug's effect in the indicated population is closer to the ATE.

### 3.3 Matching, weighting, and stratification are different designs

- **Matching** selects a subset of controls that resembles the treated. It changes the target population to the matched sample, and it discards patients. It is transparent and easy to audit, and it works well when the treated group is small and the control pool is large.
- **Weighting** keeps all patients and reweights them. It uses more information and can estimate the ATE, but it is sensitive to extreme weights and positivity violations.
- **Stratification** divides the score into strata (often quintiles) and compares within strata. It is simple and robust, but it can leave residual imbalance within strata and it loses information at the boundaries.
- **Overlap weighting** targets the population where both treatments are plausible and avoids extreme weights by construction. It is increasingly used when the groups are very different.

### 3.4 Balance is the diagnostic, not the p-value

The standard balance diagnostic is the standardized mean difference (SMD):

```text
SMD = (mean_treated - mean_control) / sqrt((s_treated^2 + s_control^2) / 2)
```

For a binary covariate, the means are proportions and the denominator is the pooled standard deviation. The conventional threshold is 0.10: an SMD below 0.10 is considered negligible imbalance. The threshold is a convention, not a law, and it should be interpreted with the size of the expected effect.

Hypothesis tests of balance are not recommended. They conflate balance with sample size: a large study can show a "significant" imbalance on a trivial difference, and a small study can fail to detect a meaningful one. Report the SMD, the variance ratio, and the distribution, not a p-value.

### 3.5 Effective sample size

Weights buy balance at the cost of precision. The effective sample size (ESS) for a weighted group is

```text
ESS = (sum of weights)^2 / (sum of squared weights)
```

If all weights are equal, ESS equals the number of patients. If a few patients have large weights, ESS falls below the count and can fall dramatically. An analysis with 4,000 records and an ESS of 40 has the precision of 40 patients, not 4,000.

Report the ESS for each group and the weight distribution: mean, standard deviation, minimum, maximum, and the proportion of weights above a stated threshold. A stabilized weight has a mean near 1 in each group; a mean far from 1 suggests a misspecified or unstable model.

### 3.6 Doubly robust estimation

A doubly robust estimator combines a treatment model and an outcome model. It is consistent if either model is correct, not necessarily both. The most common forms are:

- **Augmented IPTW (AIPTW):** the weighted outcome difference plus a regression adjustment term.
- **Targeted maximum likelihood estimation (TMLE):** a substitution estimator that updates the outcome model using the treatment model.

The appeal is protection against misspecification of one model. The limit is that both models can be wrong, and the estimator can be unstable when the propensity scores are extreme. Doubly robust does not mean doubly correct.

### 3.7 What the score cannot do

- It cannot balance unmeasured confounders. That is the identifying assumption, not a property of the method.
- It cannot create overlap where none exists. Trimming changes the population; it does not create comparability.
- It cannot fix a treatment definition that changed over time.
- It cannot replace a design that aligns time zero. A score computed from post-treatment covariates is not a propensity score; it is a mediator model.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or scale |
|---|---|---|---|
| Propensity score | Probability of treatment given covariates | Called a confounder score | Probability, 0 to 1 |
| Balancing score | A function of covariates such that treatment is independent of covariates given it | Assumed to balance unmeasured variables | Function |
| Treatment model | The model for the probability of treatment given covariates | Specified with post-treatment variables | Model |
| Positivity | Every covariate pattern has positive probability of each treatment | Checked only by the mean score | Assumption |
| Common support | The region where both treatment groups have positive density | Equated with the full range | Region |
| Overlap | The extent to which the score distributions coincide | Reported as a single number without a plot | Distribution |
| Caliper | The maximum distance for a match | Chosen without reference to the logit SD | Score units |
| Matching with replacement | A control can be matched to several treated patients | Assumed to preserve the control distribution | Design |
| Nearest-neighbor matching | Matching each treated to the closest control | Ignoring the caliper | Design |
| Full matching | Matching in variable-ratio strata | Confused with one-to-one matching | Design |
| IPTW | Inverse probability of treatment weighting | Weights not stabilized or trimmed | Weight |
| Stabilized weight | Weight multiplied by the marginal treatment probability | Called "normalized" | Weight |
| Trimming | Removing patients outside a score range | Reported without the population change | Design |
| Overlap weight | Weight proportional to e(1-e) | Confused with IPTW | Weight |
| ATE | Average treatment effect in the population | Estimated by the ATT weights | Effect |
| ATT | Average treatment effect in the treated | Called "the effect" without a target | Effect |
| ATC | Average treatment effect in the controls | Rarely reported | Effect |
| ATO | Average treatment effect in the overlap population | Called "the ATE" | Effect |
| Standardized mean difference | Difference in means divided by the pooled SD | Replaced by a balance p-value | Unitless |
| Variance ratio | Ratio of variances between groups | Omitted | Unitless |
| Effective sample size | (sum w)^2 / sum w^2 | Replaced by the number of records | Count |
| Weight distribution | Mean, SD, range, and extremes of the weights | Reported as a single maximum | Distribution |
| Doubly robust | Consistent if either the treatment or outcome model is correct | Called "doubly correct" | Property |
| AIPTW | Augmented inverse probability of treatment weighting | Implemented without checking overlap | Estimator |
| TMLE | Targeted maximum likelihood estimation | Treated as a black box | Estimator |
| Marginal structural model | A weighted model for the effect of a time-varying treatment | Confused with a baseline PS model | Model |
| Propensity score matching | Matching on the score | Called "adjustment" | Design |
| Balance table | A table of covariates before and after adjustment | Omitted from reports | Table |
| Negative control | An outcome or exposure that should show no effect | Chosen without a causal rationale | Design |
| Positivity violation | A stratum with no treated or no control patients | Trimmed silently | Data feature |
| Weight truncation | Capping weights at a percentile | Reported without the cap value | Design |

Two conventions in this topic:

- A propensity score is reported with its model, its covariates, its estimand, and its balance table. A score without a target is incomplete.
- A weight is reported with its distribution and the ESS. A maximum weight of 120 without the ESS hides the precision loss.

> **Note:** The propensity score is not a causal variable. It is a tool for constructing a comparison. Reporting "the effect of the propensity score" or adding it to a regression as a covariate confuses the tool with the treatment.

## 5. The mechanics, step by step

The procedure has fourteen steps. Steps 1 to 4 define the target and the assignment model. Steps 5 to 9 construct and diagnose the balanced sample. Steps 10 to 14 estimate, quantify uncertainty, stress-test, and report.

### Step 1. Choose the estimand and the target population

**Input:** The decision, the eligible population, and the treatment strategies.

**Output:** A named target: ATE, ATT, ATC, or ATO, with the population it describes.

**Why it exists:** The weights and the interpretation depend on the target. An ATE estimate for a program that would only be offered to the treated population answers the wrong question.

| Decision | Target | Weight for treated | Weight for control |
|---|---|---|---|
| Should the program be offered to everyone eligible? | ATE | 1 | e/(1-e) |
| Should the program be expanded among those who would use it? | ATT | 1/e | 1 |
| Should the program be extended to the unserved? | ATC | 1 | 1/(1-e) |
| Is the program effective where both options are plausible? | ATO | 1-e | e |

**Edge case:** If the treated group is small and the control pool is large, the ATT is often more precise than the ATE, because it uses all treated patients and weights controls to the treated distribution. If the groups are very different, the ATO may be the only target with adequate overlap.

### Step 2. Specify the treatment model from the DAG

**Input:** The adjustment set from the causal graph.

**Output:** A treatment model with pre-treatment covariates only.

**Why it exists:** The propensity score must include the confounders and must not include mediators or colliders. A score that includes a post-treatment variable does not balance the confounding and can induce bias.

Rules:

- Include every confounder in the adjustment set.
- Include variables that predict treatment strongly, even if they are not confounders, if they improve balance.
- Exclude mediators and colliders, for the same reasons as in regression.
- Use the functional form that the data support: splines for continuous variables, interactions where plausible.
- Do not select variables by stepwise regression on the treatment; that is a model of the sample, not of the assignment.

The predictive performance of the treatment model is not the goal. A model with a high area under the receiver operating characteristic curve separates the groups well, which often means poor overlap and extreme weights. A model with modest discrimination but good balance is preferable. Do not report the AUC as evidence that the propensity score is good; report the balance table.

### Step 3. Fit the treatment model and inspect the scores

**Input:** The treatment indicator and the covariates.

**Output:** A fitted model, a score for every patient, and a distribution by group.

**Why it exists:** The score distribution shows whether the groups are comparable. A model that separates the groups almost perfectly produces scores near 0 and 1 and no overlap.

Checks:

- The model converged.
- The score distribution by group overlaps; plot both densities or histograms.
- The mean score in the treated group is higher than in the control group, as it must be.
- No patient has a score of exactly 0 or 1 unless the model is saturated.
- The covariates are balanced before weighting within strata of the score, as a preliminary check.

### Step 4. Assess overlap and positivity

**Input:** The score distribution.

**Output:** A common-support region, a trimming rule if needed, and a statement of the population remaining.

**Why it exists:** A patient with a score outside the other group's range has no comparable counterpart. The effect for that patient is not estimable from these data, and including the patient requires extrapolation.

| Overlap situation | Signal | Action |
|---|---|---|
| Good overlap | Both distributions span most of 0 to 1 | Proceed with the chosen method |
| Partial overlap | One group concentrated near 0 or 1 | Trim to the common support; report the population |
| Minimal overlap | Distributions barely intersect | Change the target to ATO; consider a different design |
| Complete separation | No controls with high scores or no treated with low scores | The effect is not estimable without extrapolation |

**Trimming rules.** Common choices are to remove patients with scores below 0.05 or above 0.95, or to remove patients outside the range of the other group. Every rule changes the target population. Report the rule, the number removed, and the characteristics of the removed patients.

### Step 5. Choose the adjustment method

**Input:** The target, the sample size, the overlap, and the structure of the treated group.

**Output:** A named method and its target.

**Why it exists:** Matching, weighting, and overlap weighting target different populations and have different precision.

| Situation | Method | Target | Reason |
|---|---|---|---|
| Large control pool, small treated group | Nearest-neighbor matching with caliper | ATT | Transparent; uses the best controls |
| Both groups large, good overlap | IPTW with stabilized weights | ATE | Uses all patients |
| Both groups large, moderate overlap | Overlap weighting | ATO | Bounded weights; avoids trimming |
| Many strata, simple analysis | Stratification on the score | ATE or ATT | Easy to explain |
| Extreme scores, concern about weights | Overlap weights or trimming | ATO or trimmed population | Bounded influence |
| Time-varying treatment | Marginal structural model | ATE of a strategy | Handles time-varying confounding |
| Doubly robust protection wanted | AIPTW or TMLE | ATE or ATT | Consistent if one model is right |

### Step 6. Implement matching

**Input:** The score, the treated and control groups, and a caliper.

**Output:** A matched sample and a record of the matches.

**Why it exists:** Matching is a design step. The matched sample is the population for the effect estimate, and the matches must be auditable.

- Match on the logit of the score, not the score itself, because the logit is closer to normally distributed.
- Use a caliper of 0.2 times the standard deviation of the logit of the score. This is a widely used default; it is a convention, not a law.
- Decide whether to match with or without replacement. With replacement can improve balance but reuses controls and complicates the variance.
- Report the number of treated patients unmatched, the number of controls used, and the balance after matching.
- Do not match on the outcome or on post-treatment variables.

**Edge case:** If many treated patients remain unmatched, the matched population is a selected subgroup, and the estimate does not generalize to all treated patients. Report the characteristics of the unmatched patients.

### Step 7. Implement weighting

**Input:** The score and the target.

**Output:** A weight for every patient and a weight distribution.

**Why it exists:** The weights create the pseudo-population in which the covariates are balanced. The distribution shows how much the estimate depends on a few patients.

- ATE: w = 1/e for treated, w = 1/(1-e) for controls.
- ATT: w = 1 for treated, w = e/(1-e) for controls.
- Stabilized ATE: multiply by the marginal probability of the treatment group, so the mean weight is near 1.
- Truncate at a pre-specified percentile if the weights are extreme, and report the change in the estimate and the ESS.
- Do not truncate at a value chosen to make the result significant.

**Weight diagnostics:**

| Statistic | What it shows | Warning sign |
|---|---|---|
| Mean weight | Whether the weights are stabilized | Far from 1 for stabilized weights |
| Maximum weight | The influence of the most extreme patient | Above about 10 to 20 |
| 99th percentile | The tail of the weight distribution | Much larger than the median |
| ESS | The precision after weighting | Much smaller than the sample size |
| Proportion above a threshold | The fraction of highly weighted patients | A few patients dominating |

### Step 8. Check balance

**Input:** The weighted or matched sample.

**Output:** A balance table with SMDs, variance ratios, and distribution plots.

**Why it exists:** Balance is the evidence that the score did its job. Without it, the analysis is a claim, not a demonstration.

- Report the SMD for every covariate before and after adjustment.
- Use a threshold of 0.10 as a convention, and report the actual values.
- Report variance ratios for continuous covariates; a ratio far from 1 indicates a distributional imbalance that the mean difference misses.
- Plot the empirical distributions of the most important covariates.
- Do not use balance p-values as the criterion; they depend on sample size.

A covariate with an SMD of 0.12 after weighting is not automatically a failure. The threshold is a convention, and the decision should depend on how strongly the covariate predicts the outcome and how much the estimate moves when it is added to the outcome model. Report the value, the threshold, and the sensitivity of the estimate to the remaining imbalance.

### Step 9. Refit the treatment model if balance fails

**Input:** The balance table.

**Output:** A revised model or a decision that the target population must change.

**Why it exists:** A score that does not balance the covariates is not a valid balancing score for this sample. Adding interactions or splines can fix it; trimming or changing the target can fix the population.

| Balance failure | Possible cause | Remedy |
|---|---|---|
| One covariate with SMD above 0.10 | The model is misspecified for that variable | Add a spline or interaction; refit |
| Several covariates imbalanced | The model is too simple | Add covariates or interactions |
| Imbalance only in the tail | Positivity problem | Trim; change the target to ATO |
| Balance improves but ESS collapses | Extreme weights | Stabilize, truncate, or use overlap weights |

### Step 10. Estimate the effect

**Input:** The weighted or matched sample and the outcome.

**Output:** A marginal effect on the decision scale.

**Why it exists:** The score is not the estimate. The effect comes from comparing outcomes in the balanced sample or in the weighted pseudo-population.

| Outcome | Estimator | Note |
|---|---|---|
| Continuous | Weighted difference in means, or weighted regression | Report the mean difference in units |
| Binary | Weighted risk difference, or weighted logistic plus standardization | Report both RD and RR |
| Count | Weighted rate ratio with person-time | Check overdispersion |
| Time-to-event | Weighted Cox or weighted Kaplan-Meier | Check proportional hazards |
| Any, with confounding remaining | Doubly robust estimator | Requires both models |

**Do not include the propensity score as a covariate in an outcome regression** unless the regression is a deliberate approximation and the functional form is correct. The score is a summary; the outcome model needs the covariates themselves.

### Step 11. Quantify uncertainty with the design in mind

**Input:** The weighted estimate and the design.

**Output:** A standard error and interval that reflect the weighting or matching.

**Why it exists:** The weights and the matching are part of the design; a model-based variance ignores them.

| Design | Variance method | Note |
|---|---|---|
| IPTW, independent patients | Robust sandwich variance | The default in most software |
| IPTW, clustered patients | Cluster-robust variance | The cluster is the sampling unit |
| Matching without replacement | Paired or Abadie-Imbens variance | The matching is part of the design |
| Matching with replacement | Variance that accounts for reused controls | Often bootstrap the matching process |
| Doubly robust | Influence-function variance or bootstrap | The estimation has two stages |
| Any, small sample | Bootstrap the whole procedure | Includes the treatment-model uncertainty |

A common error is to report the model-based standard error from an unweighted regression on the weighted sample. That variance ignores the weights and is too small.

### Step 12. Consider a doubly robust estimator

**Input:** The treatment model, the outcome model, and the overlap.

**Output:** An estimate that is consistent if either model is correct.

**Why it exists:** The propensity model can be misspecified, and the outcome model can be misspecified. A doubly robust estimator hedges against one of the two failures.

- AIPTW: fit the outcome model, predict the potential outcomes, and combine with the weighted residuals.
- TMLE: fit an initial outcome model, then update it using the treatment model so that the efficient influence function is solved.
- Report both the singly robust and doubly robust estimates. A large discrepancy suggests that one of the models is wrong.

**Edge case:** Doubly robust estimators can be unstable when the propensity scores are extreme, because the weights still enter. Trimming or overlap weights may be needed before the estimator is reliable.

### Step 13. Run sensitivity analyses

**Input:** The estimate, the balance table, and the DAG.

**Output:** A set of estimates under alternative assumptions.

**Why it exists:** The score addresses measured confounding only; the sensitivity analyses address the rest.

| Assumption | Sensitivity analysis | What a change means |
|---|---|---|
| No unmeasured confounding | E-value; negative control outcome | A moderate confounder could explain the result |
| Correct treatment model | Alternative specifications; different covariates | The estimate depends on the model |
| Positivity | Trim at different thresholds; ATO | The target population drives the result |
| Correct outcome model | Doubly robust estimator | One of the models is wrong |
| No differential measurement | Validation subsample | Misclassification differs by group |
| Time zero | Re-run with a different index date | Immortal time or misalignment |

### Step 14. Report

**Input:** The model, the balance table, the weights, the estimate, and the sensitivity analyses.

**Output:** A results section that a reviewer can audit.

**Why it exists:** The model, the balance table, and the ESS are the evidence that the score worked.

The reporting checklist:

1. The estimand and target population.
2. The treatment model, covariates, and functional forms.
3. The score distribution and overlap.
4. The matching or weighting method and any trimming.
5. The balance table before and after.
6. The weight distribution and ESS.
7. The outcome model and the effect with a robust interval.
8. The doubly robust or alternative estimate.
9. The sensitivity analyses.
10. The unmeasured confounding statement.

Items 4 and 5 are the ones that distinguish an auditable analysis from a methods label. A report that includes the model and the estimate but not the balance table has omitted the evidence that the model worked.

The reporting template should be written before the analysis, so that the diagnostics are produced as part of the workflow rather than assembled afterward.

### Edge cases that change the estimate

| Edge case | What happens | Detection | Remedy |
|---|---|---|---|
| Extreme weights | A few patients dominate | Max weight, ESS | Stabilize, truncate, or overlap weights |
| Positivity violation | No comparable patients in a stratum | Score density plot | Trim; change the target |
| Small treated group | Matching discards many controls | Unmatched count | Full matching; ATT; ATO |
| Caliper too wide | Poor matches | Balance after matching | Tighten the caliper |
| Caliper too narrow | Many treated unmatched | Unmatched count | Widen; full matching |
| Score includes a mediator | Bias in the effect | The covariate is post-treatment | Remove it; redraw the DAG |
| Score includes a collider | Bias in the effect | The covariate is a common effect | Remove it |
| Clustered data | Variance too small | Naive versus cluster-robust SE | Cluster-robust or cluster bootstrap |
| Time-varying treatment | Baseline score is insufficient | Treatment changes during follow-up | Marginal structural model |
| Weight truncation undisclosed | The estimate is not reproducible | The cap is not reported | Report the cap and the ESS |

### Stabilized weights, in more detail

An unstabilized ATE weight of 1/e can be large when e is small. Multiplying by the marginal probability of the treatment group, P(A = a), gives a stabilized weight whose mean is approximately 1 in each group. The stabilization does not change the point estimate of the effect in a saturated model, but it reduces the variance of the weights and makes the distribution easier to inspect. A stabilized weight with a mean of 3 indicates that the model or the sample is unusual, and the analyst should investigate before proceeding.

Stabilization is not the same as normalization. Normalizing weights to sum to the sample size in each group also changes the weight distribution, but it changes the estimand in a way that depends on the sample. Report which operation was applied.

### Matching versus weighting, in more detail

Matching and weighting answer the same causal question with different target populations and different precision. Matching discards patients, so it is less efficient, but it is easier to audit and it makes the comparison group concrete. Weighting uses all patients, so it is more efficient, but it can be dominated by a few extreme weights and it requires a variance method that accounts for the weighting.

A useful practice is to run both when the data allow. If the matched and weighted estimates agree, the result is robust to the design choice. If they disagree, the difference is informative: the matched population may be a selected subgroup, or the weights may be unstable. Report both and explain the difference rather than choosing the one that looks better.

### Branch points at a glance

| If the situation is | Then the default is | Because |
|---|---|---|
| Treated group small, control pool large | Matching with caliper, ATT | Transparent and precise for the treated |
| Both groups large, good overlap | Stabilized IPTW, ATE | Uses all patients |
| Extreme scores and no trimming wanted | Overlap weights, ATO | Bounded weights |
| Positivity violation in a subgroup | Trim and report the population | The effect is not estimable there |
| Many covariates, few events | Penalized treatment model or a simpler score | Separation and overfitting |
| Time-varying treatment | Marginal structural model | Baseline score cannot handle it |
| Clustered data | Cluster-robust variance | The cluster is the sampling unit |
| Doubt about the outcome model | AIPTW or TMLE | Consistency if either model is right |
| Balance fails after weighting | Refit with interactions or splines | The score is misspecified |
| Balance fails only in the tail | Trim or change the target | Positivity, not model fit |

The table is a starting point, not a rule. The estimand and the overlap decide, in that order.

**Caliper choice.** The 0.2 × SD(logit score) rule is a default from simulation work, not a requirement. A narrower caliper improves balance but leaves more treated patients unmatched; a wider caliper matches more patients but may match dissimilar ones. Report the caliper and the number unmatched, and check balance after matching. If the unmatched patients differ systematically from the matched ones, the matched estimate does not apply to the full treated population.

**Negative control outcomes.** A negative control outcome is an outcome that should not be affected by the treatment but shares the same confounding structure. If the weighted analysis shows an effect on the negative control, the adjustment is incomplete. The control does not correct the estimate; it tests the design. Choose it before the analysis, and justify it with the same DAG logic as the primary outcome.

**The score as a covariate.** Adding the propensity score to an outcome regression is not equivalent to weighting or matching. In a linear model with a saturated score it can approximate the adjusted estimate, but in nonlinear models it can be biased, and the resulting coefficient is not a marginal effect. The score is a design tool. Use it to build the sample, then estimate the effect in that sample.

**Trimming and the target population.** Trimming removes patients who are outside the common support. The remaining population is not the original population; it is the population of patients for whom both treatments were plausible. That may be exactly the population the decision concerns, but it must be named. A trimmed ATE is an ATE for the trimmed population, not for the eligible population.

**A balance table that a reviewer can use.** The table should have one row per covariate and columns for the treated mean, the control mean, the SMD before, the SMD after, and the variance ratio after. Include the number of patients and the ESS in the header. A table with only the after column cannot show what the weighting accomplished. A table with only the SMD cannot show distributional imbalance in a continuous variable, which is why the variance ratio belongs beside it.

## 6. Worked example

### 6.1 The question and the estimand

A payer compares an intensive case-management program with usual care for 12-month hospitalization. The program is offered to high-risk patients, and the treated group is enriched with prior hospitalizations. The confounder is prior hospitalization (C): patients with a prior hospitalization are both more likely to be enrolled and more likely to be hospitalized again.

The estimand is the average treatment effect (ATE) on the 12-month hospitalization risk difference, in the eligible population. The target is the ATE because the payer is considering offering the program to everyone eligible, not only to those who enrolled.

### 6.2 The data

| Stratum | Program: events / n | Risk | Usual care: events / n | Risk | Risk difference |
|---|---|---:|---|---:|---:|
| Prior hospitalization (C = 1) | 40 / 100 | 40.0% | 30 / 50 | 60.0% | -20.0 pp |
| No prior hospitalization (C = 0) | 10 / 100 | 10.0% | 50 / 250 | 20.0% | -10.0 pp |
| Total | 50 / 200 | 25.0% | 80 / 300 | 26.7% | -1.7 pp |

The crude comparison shows almost no difference. Within each stratum, the program is associated with a 10- to 20-percentage-point lower risk. The treated group is enriched with prior-hospitalization patients, who have a much higher risk, so the crude comparison hides the benefit.

### 6.3 The propensity score

With a single binary confounder, the treatment model is saturated. The fitted probability of treatment in each stratum is the observed treated fraction:

```text
e(C = 1) = 100 / 150 = 0.6667
e(C = 0) = 100 / 350 = 0.2857
```

The score is higher for patients with a prior hospitalization, as expected. The overlap is adequate: both strata contain treated and control patients, and the scores lie between 0.286 and 0.667.

### 6.4 ATE weights

```text
C = 1: treated weight = 1/0.6667 = 1.5000
       control weight = 1/(1 - 0.6667) = 3.0000
C = 0: treated weight = 1/0.2857 = 3.5000
       control weight = 1/(1 - 0.2857) = 1.4000
```

The weights range from 1.4 to 3.5, with no extreme values. The weighted sample sizes are

```text
Treated: 100 × 1.5 + 100 × 3.5 = 150 + 350 = 500
Control:  50 × 3.0 + 250 × 1.4 = 150 + 350 = 500
```

Both groups now represent a pseudo-population of 500 patients.

### 6.5 Weighted risks and the effect

```text
Weighted risk, program:
(40 × 1.5 + 10 × 3.5) / 500 = (60 + 35) / 500 = 95 / 500 = 0.1900

Weighted risk, usual care:
(30 × 3.0 + 50 × 1.4) / 500 = (90 + 70) / 500 = 160 / 500 = 0.3200

Weighted risk difference = 0.1900 - 0.3200 = -0.1300 = -13.0 percentage points
Weighted risk ratio = 0.1900 / 0.3200 = 0.5938
```

The weighted estimate is -13.0 percentage points, compared with the crude -1.7 points. The difference is the confounding removed by the weights.

### 6.6 Balance diagnostics

Before weighting, the prevalence of prior hospitalization is

```text
Treated: 100 / 200 = 0.5000
Control: 50 / 300 = 0.1667
Pooled: (100 + 50) / 500 = 0.3000
Pooled SD = sqrt(0.30 × 0.70) = 0.4583
SMD = (0.5000 - 0.1667) / 0.4583 = 0.7274
```

An SMD of 0.73 is far above the 0.10 threshold. After weighting:

```text
Weighted treated prevalence = (100 × 1.5) / 500 = 0.3000
Weighted control prevalence = (50 × 3.0) / 500 = 0.3000
SMD = 0.0000
```

The weighting balances the confounder exactly, because the model is saturated.

The effective sample sizes are

```text
Treated ESS = 500^2 / (100 × 1.5^2 + 100 × 3.5^2)
            = 250,000 / (225 + 1,225) = 250,000 / 1,450 = 172.4

Control ESS = 500^2 / (50 × 3.0^2 + 250 × 1.4^2)
            = 250,000 / (450 + 490) = 250,000 / 940 = 266.0
```

The 500 weighted patients correspond to an effective sample of 172 treated and 266 control patients. The program estimate is therefore less precise than the raw counts suggest.

### 6.7 Diagnostic checks

**Check 1. Weights reproduce the target population.** The weighted prevalence of C is 0.30 in both groups, which equals the pooled prevalence. This is the expected result for ATE weights when the model is saturated.

**Check 2. Weighted risks equal standardization.** Standardizing directly to the pooled C prevalence of 0.30 gives

```text
Program:    0.30 × 0.40 + 0.70 × 0.10 = 0.12 + 0.07 = 0.19
Usual care: 0.30 × 0.60 + 0.70 × 0.20 = 0.18 + 0.14 = 0.32
```

The weighted risks match the standardized risks exactly. This is a strong arithmetic check: IPTW and standardization are two implementations of the same adjustment formula when the model is saturated.

**Check 3. The weight sums are equal.** Both weighted groups sum to 500, which is the size of the pseudo-population. Unequal sums would indicate a coding error or a missing stratum.

**Check 4. The ESS is below the count.** The treated ESS of 172 is 86% of the treated count of 200, and the control ESS of 266 is 89% of the control count of 300. A modest loss. If the ESS were below 50, the estimate would be too imprecise for the decision.

**Check 5. The estimate is robust to the weight type.** With ATT weights, the treated weight is 1 and the control weight is e/(1-e):

```text
C = 1: control weight = 0.6667/0.3333 = 2.0
C = 0: control weight = 0.2857/0.7143 = 0.4
Weighted control risk = (30×2.0 + 50×0.4) / (50×2.0 + 250×0.4)
                      = (60 + 20) / (100 + 100) = 80/200 = 0.40
ATT RD = 0.25 - 0.40 = -0.15 = -15.0 percentage points
```

The ATT is -15.0 points, larger in magnitude than the ATE of -13.0, because the treated population has more prior-hospitalization patients, in whom the absolute benefit is larger. Both are negative; the target population changes the magnitude.

**Check 6. The weights are not extreme.** The maximum weight is 3.5, which is far below the range where a single patient could dominate the estimate. A maximum above 20 would call for stabilization, truncation, or overlap weights. The ratio of the maximum to the minimum weight is 3.5/1.4 = 2.5, which is modest.

**Check 7. The approximate interval.** Using the ESS as the effective sample size:

```text
SE = sqrt(0.19 × 0.81/172.4 + 0.32 × 0.68/266.0)
   = sqrt(0.000893 + 0.000818)
   = sqrt(0.001711) = 0.04136
95% CI = -0.1300 +/- 1.96 × 0.04136
       = (-0.2111, -0.0489) = (-21.1, -4.9) percentage points
```

The interval excludes zero. It is approximate because it uses the ESS rather than a robust influence-function variance, and it treats the weights as fixed. A production analysis would use a sandwich variance or a bootstrap that repeats the treatment model.

### 6.8 Vary one assumption: the treated group is much smaller

Suppose the program enrolled only 50 patients rather than 200, with the same stratum-specific risks. The ATE weights would be more variable, and the ESS would fall. The estimate would remain -13.0 points in expectation, but the interval would widen substantially.

This is the trade-off the ESS measures. The point estimate is not biased by a smaller treated group; the precision is. A decision rule that requires the lower confidence limit to exclude a cost threshold would fail, even though the point estimate is unchanged.

### 6.9 What to report

Report the ATE as the primary estimate: **-13.0 percentage points (approximate 95% CI -21.1 to -4.9)**, with the risk ratio of 0.59. Report the crude estimate (-1.7 points) and explain that the difference is confounding by prior hospitalization. Report the balance table (SMD 0.73 before, 0.00 after), the weight distribution (1.4 to 3.5), and the ESS (172 treated, 266 control). Report the ATT (-15.0 points) as a secondary estimate for the population that would actually enroll.

Report the assumptions:

- No unmeasured confounding given prior hospitalization. This is the weak point: frailty, social support, and provider preference are not measured.
- Correct treatment model. The saturated model is correct by construction for a single binary confounder, but a real analysis with many covariates would need to check balance and functional form.
- Positivity. Both strata contain both groups.
- Correct outcome measurement.

The honest conclusion: "After weighting for prior hospitalization, the program was associated with a 13.0-percentage-point lower 12-month hospitalization risk (approximate 95% CI -21.1 to -4.9). The crude estimate was -1.7 points because the program enrolled patients with higher prior utilization. The estimate is sensitive to unmeasured confounding by frailty and social support, and the effective sample size is 172 treated and 266 control patients."

The estimate to report is the weighted ATE, not the crude difference and not the ATT alone. The ATE answers the payer's question about offering the program to everyone eligible. The ATT is reported as a secondary estimate because it describes the population that actually enrolled, and the two differ by 2 percentage points, which is material for a budget calculation.

## 7. Code, query, or calculation

The script computes the propensity score from a saturated treatment model, builds ATE and ATT weights, estimates the weighted risks and risk difference, and reports balance and effective sample size. It uses only the standard library.

**Assumptions:**

- The treatment model is saturated, so the fitted probabilities equal the observed treated fraction in each stratum.
- The stratum-specific risks are estimated without error beyond binomial sampling.
- The weights are treated as fixed for the approximate standard error.
- The target population for the primary estimate is the ATE.

```python
import math

# (stratum, treated n, treated events, control n, control events)
rows = [("prior hospitalization", 100, 40, 50, 30),
        ("no prior hospitalization", 100, 10, 250, 50)]

n_t = sum(r[1] for r in rows); n_c = sum(r[3] for r in rows)
e_t = sum(r[2] for r in rows); e_c = sum(r[4] for r in rows)
print(f"crude: treated {e_t/n_t:.4f} control {e_c/n_c:.4f} RD={e_t/n_t - e_c/n_c:.4f}")

# Saturated propensity model: empirical treated fraction within each stratum
ps = {r[0]: r[1] / (r[1] + r[3]) for r in rows}
for s, e in ps.items():
    print(f"e({s}) = {e:.4f}")

# ATE weights and weighted risks
num_t = den_t = num_c = den_c = 0.0
sum_w2_t = sum_w2_c = 0.0
for s, nt, et, nc, ec in rows:
    e = ps[s]
    wt, wc = 1 / e, 1 / (1 - e)
    num_t += et * wt; den_t += nt * wt; sum_w2_t += nt * wt * wt
    num_c += ec * wc; den_c += nc * wc; sum_w2_c += nc * wc * wc
rt, rc = num_t / den_t, num_c / den_c
print(f"ATE weighted risks: treated {rt:.4f} control {rc:.4f} RD={rt-rc:.4f} RR={rt/rc:.4f}")
print(f"weighted sums: treated {den_t:.1f} control {den_c:.1f}")
print(f"ESS: treated {den_t**2/sum_w2_t:.1f} control {den_c**2/sum_w2_c:.1f}")

# Balance before and after
p_t = next(r[1] for r in rows if r[0].startswith("prior")) / n_t
p_c = next(r[3] for r in rows if r[0].startswith("prior")) / n_c
p_pool = (next(r[1] for r in rows if r[0].startswith("prior")) +
          next(r[3] for r in rows if r[0].startswith("prior"))) / (n_t + n_c)
smd_before = (p_t - p_c) / math.sqrt(p_pool * (1 - p_pool))
wt_pre = sum(r[1] * (1 / ps[r[0]]) for r in rows if r[0].startswith("prior"))
wc_pre = sum(r[3] * (1 / (1 - ps[r[0]])) for r in rows if r[0].startswith("prior"))
smd_after = (wt_pre / den_t - wc_pre / den_c) / math.sqrt(p_pool * (1 - p_pool))
print(f"SMD before {smd_before:.4f}; after {smd_after:.4f}")

# ATT
num_c2 = den_c2 = 0.0
for s, nt, et, nc, ec in rows:
    w = ps[s] / (1 - ps[s])
    num_c2 += ec * w; den_c2 += nc * w
print(f"ATT: treated {e_t/n_t:.4f} control {num_c2/den_c2:.4f} RD={e_t/n_t - num_c2/den_c2:.4f}")

# Approximate SE using ESS
se = math.sqrt(rt * (1 - rt) / (den_t**2 / sum_w2_t) + rc * (1 - rc) / (den_c**2 / sum_w2_c))
print(f"approx SE={se:.4f} 95% CI=({rt-rc-1.96*se:.4f}, {rt-rc+1.96*se:.4f})")
```

**Expected output:**

```text
crude: treated 0.2500 control 0.2667 RD=-0.0167
e(prior hospitalization) = 0.6667
e(no prior hospitalization) = 0.2857
ATE weighted risks: treated 0.1900 control 0.3200 RD=-0.1300 RR=0.5938
weighted sums: treated 500.0 control 500.0
ESS: treated 172.4 control 266.0
SMD before 0.7274; after 0.0000
ATT: treated 0.2500 control 0.4000 RD=-0.1500
approx SE=0.0414 95% CI=(-0.2111, -0.0489)
```

**What the output shows:**

- The crude risk difference is -1.7 percentage points, and the weighted difference is -13.0. The difference is the confounding removed by the weights.
- The weighted group sums are both 500, which is the size of the pseudo-population. This is a check on the ATE weights: each group is weighted up to the same total.
- The SMD falls from 0.7274 to 0.0000. Exact zero balance is expected with a saturated model; with a real model, the after-weighting SMD would be small but not zero.
- The ESS of 172 treated and 266 control is below the raw counts. A confidence interval based on 500 and 500 would be too narrow.
- The ATT is -15.0 points, larger than the ATE. The target population matters.
- The approximate interval uses the ESS and treats the weights as fixed. A production analysis should use a robust sandwich variance or a bootstrap that repeats the treatment model.

> **Note:** The saturated model is a teaching device. With many covariates, the propensity model is not saturated, balance is approximate, and the model must be checked and possibly refitted. The script's exact-zero SMD is not a realistic expectation.

## 8. Data, standards, and regulatory context

### 8.1 Binding requirements and conventions

| Instrument | Status | What it constrains | What it does not constrain |
|---|---|---|---|
| International Council for Harmonisation (ICH) E9 (1998) | Binding through regional adoption | Pre-specified analysis, covariates, analysis populations | The use of propensity scores |
| ICH E9(R1) (2019) | Binding through regional adoption | The estimand and the target population | The weighting method |
| FDA RWE guidance (August 2023) | FDA guidance | Data reliability, pre-specified protocol, transparency | The choice of estimator |
| Consolidated Standards of Reporting Trials (CONSORT) 2010 | Journal editorial policy | Reporting of randomized trials, not observational studies | Propensity score reporting |
| Strengthening the Reporting of Observational Studies in Epidemiology (STROBE) item 16a | Journal editorial policy | Methods used to control for confounding | The specific method |
| REporting of studies Conducted using Observational Routinely-collected health Data (RECORD) item 7.1 | Journal editorial policy | Code lists for confounders | The score model |
| Austin and Stuart (2015) | Methodological reference | Balance diagnostics and weight reporting | The substantive model |

### 8.2 What the standards require

STROBE item 16a requires the methods used to control for confounding to be reported. A propensity score analysis should report the treatment model, the covariates, the estimand, the balance table, and the weight distribution. These are not separately mandated items, but they are the minimum content that makes item 16a meaningful.

The FDA RWE guidance emphasizes pre-specification. A propensity score model specified after looking at the outcome is a form of model selection on the result. The model, the covariates, and the target population should be written before the analysis.

ICH E9(R1) requires the estimand to be defined. For a propensity score analysis, the estimand determines whether the weights are ATE, ATT, or ATO. A report that says "propensity score adjusted" without naming the target has not stated the estimand.

### 8.3 Why balance reporting is not optional

A propensity score result without a balance table cannot be distinguished from an unadjusted comparison. The score is a means, not an end. The balance table is the evidence that the means achieved the end. Austin and Stuart (2015) found that among 29 eligible applied articles published in 2014, only about 48% assessed baseline covariate balance after weighting and only about 10% examined the weight distribution. The reporting gap is the reason the balance table is emphasized here.

### 8.4 A reporting template

A propensity score report that a reviewer can audit contains:

1. The estimand and target population.
2. The treatment model, with covariates, functional forms, and interactions.
3. The score distribution by group, with a plot.
4. The overlap assessment and any trimming rule.
5. The matching or weighting method, with the caliper or weight formula.
6. The balance table before and after, with SMDs and variance ratios.
7. The weight distribution and the effective sample size.
8. The outcome model and the effect with a robust interval.
9. The doubly robust or alternative estimate.
10. The sensitivity analysis for unmeasured confounding.

The template is not mandated, but it matches what the methodological literature recommends and what a reviewer will ask for.

### 8.5 Overlap in different data sources

| Data source | Typical overlap problem | Consequence | Practical response |
|---|---|---|---|
| Claims | Treated patients are sicker and have more utilization | Scores concentrated near 1 in the treated | ATT or ATO; trim |
| EHR | Treatment is driven by unrecorded clinical judgment | Scores separate by site or clinician | Include site; check within-site overlap |
| Registry | Eligibility criteria create distinct treated and control groups | Little common support | Restrict to the overlap population |
| Trial | Randomization ensures overlap by design | No propensity model needed | Do not use a score |
| Wearable | Treatment is rare and selected | Small treated group | ATT; report precision |

The registry row is the one where propensity methods are most often misapplied. A registry that enrolls only patients who received a device and compares them with patients who did not is comparing two populations that may have no common support. The score cannot fix the design; it can only describe the lack of overlap.

### 8.6 Three questions a reviewer will ask

1. **What is the target population, and do the weights implement it?** The answer requires the estimand and the weight formula.
2. **What did the weighting balance, and what did it not?** The answer is the balance table, with the unmeasured confounders named.
3. **How precise is the estimate after weighting?** The answer is the ESS and the weight distribution.

A report that answers the first two but not the third has omitted the precision cost. A report that answers the third but not the first has reported a precise number for an unnamed population.

### 8.7 What the standards do not cover

- They do not require a specific balance threshold.
- They do not require the effective sample size.
- They do not require the treatment model's functional form.
- They do not resolve the choice between matching and weighting.
- They do not validate the no-unmeasured-confounding assumption.

## 9. Industry practice and case evidence

### 9.1 Documented practice: propensity scores in real-world evidence

Propensity score methods are standard in claims-based comparative effectiveness research. RCT-DUPLICATE used propensity score matching to emulate 32 randomized trials, with protocols registered before analysis. The reported correlation between emulated and randomized estimates was 0.82 overall, and 0.93 in the 16 trials with close emulation. The design elements, including eligibility, time zero, and outcome definitions, were as important as the score itself.

**Inference:** the correlation is an average across trials and databases. It does not establish that a particular propensity model is adequate for a particular question.

### 9.2 Documented evidence: the reporting gap

Austin and Stuart (2015) reviewed 139 applied articles using IPTW published between 1987 and 2014. Among 29 eligible articles from 2014, about 48% assessed balance after weighting and about 10% examined the weight distribution. The paper provides a checklist of diagnostics: standardized differences for means, higher moments, and interactions; variance ratios; side-by-side boxplots; empirical cumulative distribution functions; and weight summaries including the mean, standard deviation, minimum, and maximum.

The gap matters because an IPTW estimate without balance and weight diagnostics cannot be evaluated. The reader cannot tell whether the weights balanced the covariates or whether a few patients dominated the estimate.

### 9.3 Documented case: matching in device registries

Device registries commonly use propensity score matching to compare implant types. The treated and control groups differ in age, anatomy, and operator preference, and the registry may have hundreds of covariates. Matching is attractive because it produces a concrete comparison group that clinicians can inspect.

The risk is that matching discards a large share of controls and changes the population. A registry analysis that matches 1,000 treated patients to 1,000 of 20,000 controls answers a question about the matched population, not about all treated patients. The report should state the number unmatched and compare their characteristics with the matched patients.

**Inference:** the frequency of this pattern is not documented in a single statistic. The methodological literature consistently recommends reporting the unmatched population, and the recommendation is not always followed.

### 9.4 Documented case: the choice of target population

A program evaluation that uses the ATE answers "what if everyone eligible were enrolled?" A payer considering an expansion may want the ATT or the ATO, because the expansion would reach patients similar to those who enroll. The three targets can differ materially when the treated group is concentrated in a high-risk stratum, as in the worked example, where the ATE is -13.0 points and the ATT is -15.0 points.

**Inference:** the direction of the difference depends on the correlation between the treatment effect and the propensity score. When the effect is larger in the strata that are more likely to be treated, the ATT is larger in magnitude than the ATE. When the effect is larger in the untreated strata, the reverse holds.

### 9.5 Practice pattern: the score as a checkbox

A common industry pattern is to report "propensity score matched" as a methods label without the model, the balance table, or the target. The label signals sophistication, but it does not provide evidence. A reviewer cannot distinguish a well-balanced matched sample from a poorly balanced one.

**Inference:** the pattern is visible in abstracts and vendor decks, but its prevalence is not measured. The remedy is a reporting template that requires the model, the estimand, the balance table, and the ESS.

### 9.6 The cost of a score that is not checked

An unchecked propensity analysis fails in ways that are hard to see:

- The balance table is omitted, so residual imbalance is invisible.
- The weights are extreme, so the estimate is driven by a few patients.
- The target population is unnamed, so the estimate is applied to the wrong decision.
- The variance is model-based, so the interval is too narrow.

The consequence is a confident recommendation that does not survive a second analysis. The cost is not only the wrong decision; it is the loss of credibility for the next analysis.

**Inference:** the specific financial cost depends on the decision and is not documented in a public statistic. The pattern is documented in the reporting-gap literature cited above.

### 9.7 Documented fact versus inference

| Claim | Status | Basis |
|---|---|---|
| Propensity weighting can balance measured covariates | Documented | The Rosenbaum-Rubin theorem and the worked example |
| Balance and weight diagnostics are underreported | Documented | The Austin-Stuart review of 29 articles |
| Extreme weights can dominate an estimate | Documented | The ESS formula and simulation studies |
| Propensity matching can emulate a trial on average | Documented with variation | RCT-DUPLICATE correlation |
| A propensity score removes confounding by unmeasured variables | Not supported | The score balances only measured covariates |
| A matched sample is identical to the treated sample | Not supported | Matching is approximate and changes the population |

The last two rows are the claims most often made in vendor materials. They are not supported by the method, and the reporting template in Section 8.4 is designed to make them visible.

### 9.8 Overlap weighting in practice

Overlap weights were proposed by Li, Morgan, and Zaslavsky in 2018 and extended for extreme scores by Li, Thomas, and Li in 2019. The weights are bounded between 0 and 1, so no single patient can dominate, and the target population is the overlap population, where both treatments are plausible. The method is attractive when the groups are very different and trimming is undesirable.

The cost is that the target population is not the whole population. An overlap-weighted estimate answers "what is the effect among patients for whom either treatment is plausible?" That may be the right question for a clinical guideline, and the wrong question for a coverage decision about everyone eligible. State the target.

**Inference:** the method is increasingly used in applied work, but the frequency of use is not documented in a public census. The properties are established in the cited papers.

### 9.9 A note on software defaults

Most statistical packages offer a one-line propensity score function and a one-line weighting function. The defaults often include unstabilized weights, a model-based standard error, and no balance table. A workflow that accepts the defaults will produce an estimate that looks complete and is missing the diagnostics that make it interpretable.

The practical response is a wrapper function that returns the model, the score distribution, the weights, the balance table, the ESS, and the effect with a robust interval in one object. The wrapper makes the diagnostics part of the estimate rather than an optional extra.

**Inference:** the specific defaults differ by package and version. The principle is general: check what the software did before interpreting the output.

### 9.10 What mature teams do differently

1. They name the target population before fitting the score.
2. They derive the covariates from a DAG and exclude post-treatment variables.
3. They report the score distribution and the overlap.
4. They report the balance table before and after, with SMDs and variance ratios.
5. They report the weight distribution and the ESS.
6. They use a robust variance and, where possible, a doubly robust estimator.
7. They run a negative control outcome and an E-value.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| No balance table | The report says "propensity matched" only | The score's effect is asserted, not shown | Unknown residual imbalance | Require SMDs before and after |
| Extreme weights | Max weight above 20; ESS far below n | A few patients dominate | Unstable estimate; wide interval | Stabilize, truncate, or overlap weights |
| Positivity violation | Score distributions do not overlap | No comparable patients | Extrapolation | Trim; change the target |
| Score includes a mediator | A post-treatment variable is in the model | The score blocks part of the effect | Biased effect | Remove it; redraw the DAG |
| Score includes a collider | A post-treatment common effect is in the model | Conditioning opens a path | Biased effect | Remove it |
| PS as a covariate | The score is added to an outcome regression | The coefficient is not a marginal effect | Biased or uninterpretable | Use weighting or matching |
| Target not stated | "Adjusted" without ATE/ATT | The weights imply a target that is not named | The estimate answers an unnamed question | State the estimand |
| Balance judged by p-values | A table of balance p-values | p-values depend on n | Trivial imbalance called significant | Use SMDs |
| Trimming undisclosed | The sample size is smaller than the cohort | Patients were removed silently | The population changed | Report the rule and the n |
| Matching without a caliper | Poor matches on the score | Distant patients are compared | Bias | Use a caliper |
| Model-based SE on weighted data | The interval is too narrow | The weights are ignored | False precision | Robust or bootstrap variance |
| Doubly robust assumed correct | No comparison with singly robust estimates | Both models can be wrong | Overconfidence | Report both |
| Unmeasured confounding ignored | No E-value or negative control | The key assumption is untested | Overconfident conclusion | Sensitivity analysis |
| Time-varying treatment | Baseline score only | Treatment changes during follow-up | Biased effect | Marginal structural model |
| Clustering ignored | Naive SE smaller than cluster SE | Within-cluster correlation | Interval too narrow | Cluster-robust variance |
| Score model overfit | Many covariates, few treated patients | Separation in the treatment model | Extreme scores | Fewer covariates; penalization |
| Matching without reporting unmatched | The matched sample is a selected subgroup | Treated patients without good matches are dropped | The estimate is for a subgroup | Report unmatched; full matching |
| Propensity model with separation | A coefficient diverges; scores near 0 or 1 | Too many covariates relative to treated patients | Extreme weights | Fewer covariates; penalization |
| Score estimated on the full cohort but analyzed on a subset | The weights refer to a different population | Eligibility and analysis samples differ | Biased target | Refit the score in the analysis sample |
| Balance table only after weighting | No baseline comparison | The improvement cannot be judged | The reader cannot assess the model | Report before and after |
| Weight formula mismatched to the target | ATT weights reported as ATE | The estimand and the method disagree | Wrong interpretation | State the formula and the target together |

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| No unmeasured confounding | The score balances only measured covariates | An unmeasured common cause | E-value; negative control |
| Correct treatment model | The score is a balancing score only if the model is adequate | Misspecified functional form | Balance table; refit |
| Positivity | Every covariate pattern has both groups | Structural zeros | Score distribution; overlap |
| Correctly defined treatment | The treatment is a single intervention | Multiple doses or versions | Consistency review |
| Pre-treatment covariates only | Post-treatment variables would block effects or open paths | A mediator or collider in the model | DAG review |
| Correct outcome measurement | The effect refers to the measured outcome | Differential misclassification | Validation sample |
| SUTVA | One patient's treatment does not affect another | Interference | Cluster design |
| Stable weights | The weights represent a well-defined population | Extreme weights or truncation | Weight distribution; ESS |
| Correct variance method | The interval has nominal coverage | Model-based SE on weighted data | Compare with bootstrap |
| No selection on a collider | The cohort is not defined by a post-treatment event | Enrollment on a collider | Redesign |
| Correct target population | The estimand matches the decision | ATE used for an ATT decision | State and justify the target |
| Correct treatment model | The score balances the covariates | Misspecified functional form | Balance table |
| No model overfitting | The score is stable | Too many covariates relative to treated patients | Cross-validation; penalization |
| Same sample for score and analysis | The weights apply to the analysis population | The score is fit on a different cohort | Refit in the analysis sample |
| Reported target matches the weights | The estimate is interpreted correctly | ATT weights reported as ATE | Check the weight formula against the stated target |
| Balance is checked on the analysis sample | The weights are relevant | Balance computed before trimming | Recompute after trimming |
| Weight distribution is reported | The precision cost is visible | Only the estimate is reported | Report mean, max, ESS |

**Limits of this topic.** Propensity scores address measured confounding only. The balance table cannot detect an unmeasured confounder, and the score cannot create overlap where none exists. The worked example uses a saturated model with one binary confounder, so the balance is exact; real analyses will have residual imbalance and require model checking. The approximate interval in Section 6 uses the effective sample size and is not a substitute for a robust variance. Matching and weighting are designs, and the choice between them changes the target population, not only the precision.

## 12. Exercises

### Exercise 1. Compute the score and the weights (application, 45 minutes)

**Data.** Two strata. In stratum 1, 30 of 50 patients are treated. In stratum 2, 20 of 100 are treated.

**Deliverable.** The propensity score in each stratum, the ATE weights for each of the four cells, the weighted group sums, and the weighted risks if the event counts are 12/30 and 4/20 (treated) and 10/20 and 8/80 (control).

**Acceptance criteria.**

- The score is computed as the treated fraction within each stratum.
- The weights are 1/e and 1/(1-e).
- The weighted sums are equal.
- The weighted risks are reported with units.

### Exercise 2. Compute balance (application, 40 minutes)

**Data.** The treated prevalence of a binary confounder is 0.60 and the control prevalence is 0.25. The pooled prevalence is 0.40.

**Deliverable.** The SMD before weighting. Then compute the SMD after weighting if the weighted prevalence is 0.40 in both groups.

**Acceptance criteria.**

- The pooled SD is sqrt(0.40 × 0.60).
- The SMD is the difference divided by the pooled SD.
- The after-weighting SMD is zero.
- The answer states why exact zero is unusual outside a saturated model.

### Exercise 3. Effective sample size (application, 35 minutes)

**Data.** A treated group of 400 patients has weights: 200 patients at 1.0, 150 at 2.0, and 50 at 6.0.

**Deliverable.** The sum of weights, the sum of squared weights, and the ESS. Compare with the raw count.

**Acceptance criteria.**

- The ESS formula is written explicitly.
- The arithmetic is shown.
- The comparison is expressed as a percentage of the raw count.
- The answer states the implication for the confidence interval.

### Exercise 4. Choose the target (design, 50 minutes)

**Scenario.** A program has 120 treated patients and 4,000 controls. The treated patients are much sicker than the controls. A payer is deciding whether to expand the program to all eligible patients.

**Deliverable.** A recommendation among ATE, ATT, and ATO, with the weights, the expected precision, and the interpretation.

**Acceptance criteria.**

- The recommendation acknowledges the overlap problem.
- The weights for the chosen target are stated.
- The answer explains why the ATE may be imprecise and why the ATT may be more feasible.
- The decision question is matched to the target.

### Exercise 5. Critique a matching claim (critique, 50 minutes)

**Claim.** "We propensity-matched 500 treated patients to 500 controls and found a significant benefit. The analysis is robust because the groups are now identical."

**Deliverable.** A 500-word critique covering balance, the unmatched population, the score model, the variance, and unmeasured confounding.

**Acceptance criteria.**

- The critique rejects "identical" as an unsupported claim.
- It asks for the balance table and the SMDs.
- It asks for the number unmatched and their characteristics.
- It notes that matching addresses measured confounding only.
- It asks for the variance method.

### Exercise 6. Doubly robust estimation (application, 50 minutes)

**Scenario.** An IPTW estimate is -0.08 and an AIPTW estimate is -0.03. The balance table is good.

**Deliverable.** An explanation of why the two estimates differ, which one you would report, and what diagnostic you would run.

**Acceptance criteria.**

- The answer identifies the outcome model as the likely source of the difference.
- It does not assume the doubly robust estimate is automatically correct.
- It proposes refitting the outcome model or checking its calibration.
- It recommends reporting both.

### Exercise 7. Design a propensity analysis (design, 100 minutes)

**Scenario.** A claims database comparison of two anticoagulants, with 8,000 patients and 30 covariates.

**Deliverable.** A two-page plan with the estimand, the DAG-based covariates, the model, the target, the overlap assessment, the balance diagnostics, the outcome model, the variance, and the sensitivity analyses.

**Acceptance criteria.**

- The target population is named.
- The covariates exclude post-treatment variables.
- The balance table and ESS are pre-specified.
- The variance method accounts for weighting.
- At least two sensitivity analyses are listed.

### Exercise 8. Audit a propensity report (critique, 60 minutes)

**Claim.** "Propensity score weighting was used. The effect was significant (p = 0.02)."

**Deliverable.** At least eight questions and the minimum reporting package.

**Acceptance criteria.**

- Questions cover the estimand, the model, the balance, the weights, the ESS, the variance, and the sensitivity.
- The package includes the balance table and the weight distribution.
- The answer distinguishes underreporting from a wrong analysis.
- The final judgment is conditional.

## 13. Self-check answers

**Answer 1 (Exercise 1 weights).**

```text
e(1) = 30/50 = 0.60; e(2) = 20/100 = 0.20
Stratum 1: treated 1/0.60 = 1.6667; control 1/0.40 = 2.5000
Stratum 2: treated 1/0.20 = 5.0000; control 1/0.80 = 1.2500

Weighted treated sum = 30×1.6667 + 20×5.0000 = 50 + 100 = 150
Weighted control sum = 20×2.5000 + 80×1.2500 = 50 + 100 = 150

Weighted treated risk = (12×1.6667 + 4×5.0000)/150 = (20 + 20)/150 = 0.2667
Weighted control risk = (10×2.5000 + 8×1.2500)/150 = (25 + 10)/150 = 0.2333
Weighted RD = 0.0333 = +3.3 percentage points
```

The weighted sums are equal, which is the check that the ATE weights are correctly applied. The weighted risk difference is positive in this example; the crude risks are 16/50 = 32% and 18/100 = 18%, a crude RD of +14 points. The weighting moves the estimate toward the null, which means the treated group was enriched with low-risk patients in this construction.

**Answer 2 (Exercise 2 SMD).**

```text
Pooled SD = sqrt(0.40 × 0.60) = sqrt(0.24) = 0.4899
SMD = (0.60 - 0.25) / 0.4899 = 0.7143
After weighting: (0.40 - 0.40) / 0.4899 = 0
```

An SMD of 0.71 is far above the 0.10 convention. Exact zero after weighting is a property of a saturated model; in a real analysis, the after-weighting SMD would be small but not zero, and the analyst would report the value.

A useful comparison is the SMD before and after as a ratio. An SMD that falls by 90% is a substantial improvement even if it does not reach 0.10. An SMD that falls by 20% suggests the model has not captured the assignment mechanism, and the analyst should add interactions or reconsider the covariates.

The same logic applies to the variance ratio. A variance ratio of 2.0 after weighting means the treated group's covariate distribution is twice as variable as the control's, even if the means agree. For a continuous confounder such as age, that is a distributional imbalance that can affect a nonlinear outcome model.

**Answer 3 (Exercise 3 ESS).**

```text
Sum of weights = 200×1 + 150×2 + 50×6 = 200 + 300 + 300 = 800
Sum of squared weights = 200×1 + 150×4 + 50×36 = 200 + 600 + 1,800 = 2,600
ESS = 800^2 / 2,600 = 640,000 / 2,600 = 246.2
ESS / raw count = 246.2 / 400 = 61.5%
```

The effective sample is 246 patients, not 400. The interval based on 400 would be about 27% too narrow in standard-error terms (sqrt(400/246) = 1.275). The 50 patients with weight 6 account for most of the loss.

**Answer 4 (Exercise 4 target).**

With 120 treated and 4,000 controls and large baseline differences, the ATE weights for the controls in the low-propensity region would be very large, and the ESS would collapse. The ATT is more feasible: it uses all 120 treated patients and weights the controls to the treated distribution. The ATT answers "what is the effect in patients like those who were treated?", which is close to the payer's question if the expansion would reach similar patients. If the payer wants the effect in all eligible patients, the ATE is the right target but may not be estimable with adequate precision; the analysis should report the ATT and state the limitation.

**Answer 5 (Exercise 6 doubly robust).**

The IPTW estimate (-0.08) and the AIPTW estimate (-0.03) differ by 0.05. The balance table is good, so the treatment model is probably adequate. The difference is more likely due to the outcome model: AIPTW uses the outcome model to adjust the weighted estimate, and a misspecified outcome model can move the estimate. The doubly robust estimate is not automatically correct; it is consistent if either model is correct, and here the treatment model looks better supported. Report both, refit the outcome model with the functional forms suggested by the DAG, and check calibration. If the estimates converge, the conclusion is stronger; if they remain apart, the discrepancy is a finding.

**Answer 6 (Exercise 5 critique).**

A strong critique:

- "Identical" is too strong. Matching balances measured covariates approximately; it does not make groups identical, and it does not balance unmeasured ones.
- The balance table and SMDs are missing.
- The number of treated patients unmatched and the characteristics of the unmatched are missing.
- The caliper and matching algorithm are not stated.
- The variance method is not stated; matched data require a paired or Abadie-Imbens variance, or a bootstrap.
- The result is conditional on no unmeasured confounding, which matching cannot establish.

**Answer 7 (Exercise 7 plan outline).**

A strong plan:

- Names the ATT or ATE and the population.
- Uses a DAG to select pre-treatment covariates and exclude post-treatment variables.
- Specifies a logistic treatment model with splines for age and prior utilization.
- Assesses overlap with density plots and a trimming rule.
- Pre-specifies the balance table with SMDs and variance ratios.
- Uses stabilized weights, reports the maximum weight and ESS.
- Uses a robust sandwich variance.
- Runs an E-value and a negative control outcome.

**Answer 8 (Exercise 8 questions).**

1. What is the estimand and target population?
2. What covariates were in the treatment model, and were any post-treatment?
3. What was the score distribution and overlap?
4. What were the SMDs before and after?
5. What were the weights, and what was the maximum?
6. What was the effective sample size?
7. What variance method was used?
8. What sensitivity analysis addressed unmeasured confounding?

A report that answers all eight is auditable. A report that answers only the estimate and the p-value is a claim, not an analysis.

The minimum reporting package is the balance table, the weight distribution with the ESS, and the named target. The full package adds the treatment model, the overlap plot, the variance method, the doubly robust comparison, and the sensitivity analysis. A report can be short and still be auditable if it contains the first three.

The exercises can be answered with a calculator and a table. The point is not the arithmetic; it is the habit of checking the weighted sums, the SMD, and the ESS before interpreting the estimate. An analysis that reports the effect without those three checks is incomplete, regardless of the software.

The eight questions are also a design checklist. If the analyst cannot answer questions 1 to 4 before fitting the model, the analysis is not ready. If the analyst cannot answer questions 5 to 8 after fitting it, the analysis is not finished.

The exercises in this section are ordered from computation to critique to design. A learner who can do 1 to 4 has the mechanics; 5 to 6 have the diagnostics; 7 to 8 have the judgment. The judgment exercises have no single right answer, and they are graded on whether the reasoning is explicit and the assumptions are named.

A final note on the audit: a good report can still be wrong, and a poor report can still be right. The audit assesses whether the analysis can be evaluated, not whether the estimate is correct. The two are different, and both matter.

## 14. Interview questions

**1. What is a propensity score?**

A strong answer defines it as the probability of treatment given covariates, states the balancing property, and notes that it requires no unmeasured confounding. A weak answer calls it "a score that adjusts for confounding."

**2. What is the difference between the ATE and the ATT?**

A strong answer defines both target populations and gives the weights. It explains that the ATT is often more feasible when the treated group is small. A weak answer says they are "the same effect measured differently."

**3. Why can extreme weights be a problem?**

A strong answer explains that a few patients can dominate the estimate, the ESS falls, and the interval widens or becomes unstable; it names trimming, stabilization, and overlap weights. A weak answer says "the weights are too big" without the consequence.

**4. What is the effective sample size?**

A strong answer gives the formula and explains that it measures precision after weighting, not the number of records. A weak answer confuses it with the sample size.

A strong answer also gives an example: 4,000 records with an ESS of 40 have the precision of 40 patients, and the confidence interval should reflect that. The interviewer is listening for the connection between the ESS and the interval width.

**5. How do you assess balance?**

A strong answer describes SMDs, variance ratios, distribution plots, and the 0.10 convention, and explains why balance p-values are not recommended. A weak answer says "compare the means."

**6. What does doubly robust mean?**

A strong answer says the estimator is consistent if either the treatment model or the outcome model is correct, not both, and notes the instability with extreme scores. A weak answer says it is "robust to everything."

A good follow-up is "what if both models are wrong?" The strong answer says the estimator is biased, and the only protection is a design with better identification, such as randomization or an instrument, plus a sensitivity analysis.

**7. Why is positivity important?**

A strong answer defines it and explains that a patient with no comparable counterpart cannot contribute to a causal contrast; trimming changes the population. A weak answer says "the scores must be between 0 and 1."

**8. When would you use matching instead of weighting?**

A strong answer cites a small treated group, a large control pool, a need for transparency, and the ATT target. A weak answer says matching is "more rigorous."

**9. Why should the propensity score not be used as a covariate?**

A strong answer explains that the coefficient is not a marginal effect and can be biased in nonlinear models; the score is a design tool. A weak answer says it is "fine if the model is adjusted."

**10. How do you handle a score with poor overlap?**

A strong answer describes trimming, changing the target to ATO, and reporting the population; it notes that extrapolation is not a fix. A weak answer says "trim at 0.1."

**11. What variance would you use for an IPTW estimate?**

A strong answer says a robust sandwich variance, cluster-robust if clustered, or a bootstrap that repeats the treatment model. A weak answer says "the standard error from the regression."

**12. What would make you distrust a propensity analysis?**

A strong answer lists no balance table, extreme weights, no ESS, an unnamed target, a post-treatment covariate, and no sensitivity analysis. A weak answer says "a small sample."

## 15. Key terms

- **ATE:** average treatment effect in the population.
- **ATT:** average treatment effect in the treated.
- **ATC:** average treatment effect in the controls.
- **ATO:** average treatment effect in the overlap population.
- **Balancing score:** a function of covariates that renders treatment independent of them.
- **Balance table:** a table of covariate comparisons before and after adjustment.
- **Caliper:** the maximum score distance allowed in a match.
- **Common support:** the region where both groups have positive density.
- **Doubly robust:** consistent if either the treatment or outcome model is correct.
- **Effective sample size:** (sum of weights)^2 divided by the sum of squared weights.
- **Full matching:** matching in variable-ratio strata.
- **IPTW:** inverse probability of treatment weighting.
- **Marginal structural model:** a weighted model for a time-varying treatment.
- **Matching with replacement:** allowing a control to match multiple treated patients.
- **Nearest-neighbor matching:** matching each treated patient to the closest control.
- **Negative control outcome:** an outcome that should not be affected by treatment.
- **Overlap:** the extent to which score distributions coincide.
- **Overlap weights:** weights proportional to e(1-e), targeting the ATO.
- **Positivity:** a positive probability of each treatment in every covariate stratum.
- **Propensity score:** the probability of treatment given covariates.
- **Propensity score matching:** matching on the score.
- **Stabilized weight:** a weight multiplied by the marginal treatment probability.
- **Standardized mean difference:** a mean difference divided by the pooled SD.
- **Trimming:** removing patients outside a score range.
- **Variance ratio:** the ratio of variances between groups.
- **Weight distribution:** the summary of weights across patients.
- **Weight truncation:** capping weights at a percentile.
- **AIPTW:** augmented inverse probability of treatment weighting.
- **TMLE:** targeted maximum likelihood estimation.
- **Treatment model:** the model for the probability of treatment.
- **Unmeasured confounding:** a common cause not in the data.
- **Common support region:** the score range where both groups are represented.
- **Weight normalization:** rescaling weights to sum to the sample size.
- **Augmented estimator:** an estimator that combines weighting with outcome regression.
- **Score distribution plot:** a plot comparing the propensity scores of the two groups.
- **Covariate overlap:** the region where the covariate distributions of the groups coincide.
- **Design vs analysis stage:** the separation between building a comparable sample and estimating the effect.

## 16. Sources

All URLs were accessed on 25 October 2026.

1. Rosenbaum, P. R., Rubin, D. B. "The central role of the propensity score in observational studies for causal effects." *Biometrika* 1983;70(1):41-55. https://doi.org/10.1093/biomet/70.1.41
2. Austin, P. C., Stuart, E. A. "Moving towards best practice when using inverse probability of treatment weighting (IPTW) using the propensity score to estimate causal treatment effects in observational studies." *Statistics in Medicine* 2015;34(28):3661-3679. https://doi.org/10.1002/sim.6607
3. Austin, P. C. "Optimal caliper widths for propensity-score matching when estimating differences in means and differences in proportions in observational studies." *Pharmaceutical Statistics* 2011;10(2):150-161. https://doi.org/10.1002/pst.433
4. Li, F., Morgan, K. L., Zaslavsky, A. M. "Balancing Covariates via Propensity Score Weighting." *Journal of the American Statistical Association* 2018;113(521):390-400. https://doi.org/10.1080/01621459.2016.1260466
5. Li, F., Thomas, L. E., Li, F. "Addressing Extreme Propensity Scores via the Overlap Weights." *American Journal of Epidemiology* 2019;188(1):250-257. https://doi.org/10.1093/aje/kwy201
6. International Council for Harmonisation. *ICH E9(R1) Addendum on Estimands and Sensitivity Analysis in Clinical Trials*. Step 4, 20 November 2019. https://database.ich.org/sites/default/files/E9-R1_Step4_Guideline_2019_1203.pdf
7. United States Food and Drug Administration. *Considerations for the Use of Real-World Data and Real-World Evidence to Support Regulatory Decision-Making for Drug and Biological Products*. August 2023. https://www.fda.gov/media/171667/download
8. STROBE Initiative. *STROBE Statement*. https://www.strobe-statement.org/
9. Benchimol, E. I., Smeeth, L., Guttmann, A., et al. "The RECORD Statement." *PLoS Medicine* 2015;12(10):e1001885. https://doi.org/10.1371/journal.pmed.1001885
10. Wang, S. V., Schneeweiss, S., and the RCT-DUPLICATE Initiative. "Emulation of Randomized Clinical Trials With Nonrandomized Database Analyses: Results of 32 Clinical Trials." *JAMA* 2023;329(16):1376-1385. https://doi.org/10.1001/jama.2023.4221
11. Hernán, M. A., Robins, J. M. *Causal Inference: What If*. Chapman and Hall/CRC, 2020.
12. VanderWeele, T. J., Ding, P. "Sensitivity Analysis in Observational Research: Introducing the E-Value." *Annals of Internal Medicine* 2017;167(4):268-274. https://doi.org/10.7326/M16-2607
13. Robins, J. M., Hernán, M. A., Brumback, B. "Marginal Structural Models and Causal Inference in Epidemiology." *Epidemiology* 2000;11(5):550-560.
14. Stuart, E. A. "Matching methods for causal inference: A review and a look forward." *Statistical Science* 2010;25(1):1-21.
15. Glynn, A. N., Quinn, K. M. "An Introduction to the Augmented Inverse Propensity Weighted Estimator." *Political Analysis* 2010;18(1):36-56.
16. Robins, J. M., Rotnitzky, A., Zhao, L. P. "Estimation of Regression Coefficients When Some Regressors Are Not Always Observed." *Journal of the American Statistical Association* 1994;89(427):846-866.
17. Hernán, M. A., Robins, J. M. "Using Big Data to Emulate a Target Trial When a Randomized Trial Is Not Available." *American Journal of Epidemiology* 2016;183(8):758-764. https://doi.org/10.1093/aje/kwv254
18. D'Agostino, R. B. "Propensity score methods for bias reduction in the comparison of a treatment to a non-randomized control group." *Statistics in Medicine* 1998;17(19):2265-2281.

> **Note:** The Austin and Stuart paper is the source for the reporting-gap figures in Section 9. The Austin 2011 caliper paper is in *Pharmaceutical Statistics*, not *Multivariate Behavioral Research*; the journal is stated correctly above. The Rosenbaum and Rubin paper is the primary source for the balancing property.

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`05_causal-inference-and-dags.md`](05_causal-inference-and-dags.md) — choosing the adjustment set.
- Regression: [`02_regression-models.md`](02_regression-models.md) — the outcome-model alternative.
- Target trial emulation: [`07_target-trial-emulation.md`](07_target-trial-emulation.md) — the design that frames the score.
- Sensitivity analysis: [`08_sensitivity-analysis.md`](08_sensitivity-analysis.md) — E-values and negative controls.
- Reporting: [`10_reporting-statistical-results.md`](10_reporting-statistical-results.md) — balance tables and effect scales.
- Survey file: [`../../04_statistics_causal_inference.md`](../../course/04_statistics_causal_inference.md)
- Labs: [`../../labs/`](../../labs/) — Lab 4.2 implements IPTW and balance diagnostics.
- Foundations: [`../../01A_foundations_assumptions_and_units.md`](../../course/01A_foundations_assumptions_and_units.md) — assumptions ledger.
- Estimation: [`01_estimation-and-inference.md`](01_estimation-and-inference.md) — robust intervals and effect scales.
- Missing data: [`04_missing-data.md`](04_missing-data.md) — missing covariates in a score model.
- Study design: [`../../05_study_design_evidence.md`](../../course/05_study_design_evidence.md) — design options before weighting.
- Subgroups: [`09_subgroups-and-heterogeneity.md`](09_subgroups-and-heterogeneity.md) — effects in the target population.
- Module 13: [`../../13_biopharma_clinical_development.md`](../../course/13_biopharma_clinical_development.md) — real-world evidence in development.
- Module 25: [`../../25_trial_operations_hta_global_regulatory.md`](../../course/25_trial_operations_hta_global_regulatory.md) — HTA use of observational estimates.
- Causal graphs: [`05_causal-inference-and-dags.md`](05_causal-inference-and-dags.md) — the covariate selection step.
- Survival outcomes: [`03_survival-analysis.md`](03_survival-analysis.md) — weighted time-to-event analyses.
- Quality reporting: [`../../18_quality_evaluation_reporting.md`](../../course/18_quality_evaluation_reporting.md) — evaluation standards.
