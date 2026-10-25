# Module 04 — Statistics and Causal Inference

> **Program:** Biomedical Data Science for Industry (2026)
> **Time:** 32 hours (12 lecture, 4 case clinic, 16 lab)
> **Track fit:** Common core
> **Prerequisites:** Module 02; basic probability and regression
> **Tools:** Python, statsmodels, scikit-learn, lifelines, DoWhy, R optional
> **Industry question:** "Is this association a cause, a prediction, or an artifact — and what evidence would change the decision?"

---

## 1. Why industry cares

Biomedical decisions are causal:

- Does this drug reduce mortality?
- Does this alert reduce time to antibiotics?
- Does this device prevent readmission?
- Does this screening program save lives?
- Does this benefit design improve adherence?

Prediction answers "who is at risk?" Causal inference answers "what happens if we act?" Confusing the two produces expensive mistakes:

- A model identifies high-risk patients, but the intervention does not help them.
- A treatment looks harmful because sicker patients receive it.
- A biomarker looks predictive because it is measured more often in severe cases.
- A program looks effective because enrolled patients were healthier to begin with.

This module builds the statistical and causal reasoning needed to design, analyze, and critique industry evidence.

## 2. Learning outcomes

By the end of this module, you can:

- Define an estimand and align it with a decision.
- Choose an appropriate regression model for a clinical outcome and interpret coefficients on the relevant scale.
- Explain and quantify confounding, selection bias, and collider bias using causal graphs.
- Design a target trial emulation for an observational comparative-effectiveness question.
- Apply propensity-score weighting, matching, or doubly robust estimation, and diagnose balance.
- Analyze time-to-event data with censoring and competing risks.
- Handle missing data with multiple imputation and sensitivity analysis.
- Use bootstrap and simulation to quantify uncertainty.
- Distinguish statistical significance from clinical importance.
- Critically appraise a causal claim in a paper, vendor deck, or regulatory submission.

## 3. Mental model: the estimand ladder

```
Decision
  → Estimand: which population, treatment contrast, outcome, time horizon?
    → Identification: what assumptions let us estimate it from observed data?
      → Estimation: which statistical method?
        → Sensitivity: how wrong could we be if assumptions fail?
          → Decision: is the effect large and certain enough to act?
```

Most analyses start at "estimation" and skip the first three rungs. That is why they cannot be interpreted. Use the assumptions ledger in Module 01A to record identification, positivity, consistency, and no-unmeasured-confounding assumptions before analysis.

### 3.1 Prediction vs causal inference

| Question | Target | Typical method | Validation |
|---|---|---|---|
| Who is at risk? | Conditional outcome probability | Supervised ML | Discrimination, calibration |
| What happens if we treat? | Counterfactual outcome under treatment | Causal model | Identification, balance, sensitivity |
| What is the association? | Statistical dependence | Regression | Confidence intervals, model fit |
| How does it work? | Mechanism / mediation | Mediation analysis | Assumptions, sensitivity |

**Rule:** a prediction model can be causal only if the prediction is used as a decision rule and the intervention effect is separately established. Do not infer treatment effects from feature importance.

## 4. Statistical foundations

### 4.1 Probability and distributions

- **Binary:** Bernoulli, binomial.
- **Count:** Poisson, negative binomial.
- **Continuous:** normal, log-normal, gamma.
- **Time-to-event:** exponential, Weibull, Cox.
- **Compositional:** Dirichlet, log-ratio.

Clinical data are often skewed, zero-inflated, bounded, or hierarchical. Check the distribution before choosing a model.

### 4.2 Sampling and uncertainty

- **Standard error:** variability of an estimate across repeated samples.
- **Confidence interval:** range of plausible values under a model.
- **Bootstrap:** resample observations (or clusters) to estimate uncertainty without strong distributional assumptions.
- **Bayesian credible interval:** posterior probability interval under a prior.

**Cluster bootstrap** is essential when observations are clustered within patients, hospitals, or sites.

### 4.3 p-values and multiple comparisons

A p-value is the probability of data at least as extreme as observed, assuming the null hypothesis and model are true. It is not the probability that the null is true, nor the probability of replication.

- Pre-specify the primary analysis.
- Control family-wise error or false discovery rate when testing many hypotheses.
- Report effect sizes and confidence intervals.
- Do not use "p < 0.05" as a proxy for clinical importance.

### 4.4 Statistical vs clinical significance

| Concept | Question | Example |
|---|---|---|
| Statistical significance | Is the effect unlikely under the null? | p = 0.03 |
| Clinical significance | Is the effect large enough to matter? | 0.2 mmHg BP reduction |
| Practical significance | Can we deliver it? | Requires 24/7 staffing |
| Economic significance | Is it worth the cost? | $50k per QALY |

## 5. Regression models

### 5.1 Linear regression

\[
Y = \beta_0 + \beta_1 X_1 + \cdots + \beta_p X_p + \epsilon
\]

Use for continuous outcomes. Check linearity, homoscedasticity, independence, and influential points.

### 5.2 Logistic regression

\[
\log\left(\frac{p}{1-p}\right) = \beta_0 + \beta_1 X_1 + \cdots + \beta_p X_p
\]

Use for binary outcomes. Report odds ratios with caution: they are not risk ratios, and they overstate relative risk when the outcome is common.

### 5.3 Poisson and negative binomial

Use for counts (e.g., admissions, ED visits). Negative binomial handles overdispersion.

### 5.4 Mixed-effects models

Use when observations are clustered (patients within hospitals, repeated measures).

\[
Y_{ij} = \beta_0 + \beta_1 X_{ij} + u_j + \epsilon_{ij}
\]

- **Fixed effects:** population-average associations.
- **Random effects:** cluster-specific variation.
- **GEE:** population-average effects with robust standard errors.

### 5.5 Nonlinearity

- Splines for continuous variables.
- Interaction terms for effect modification.
- Piecewise models for thresholds.
- Tree-based models for flexible prediction, but not for easy causal interpretation.

### 5.6 Regularization

- Ridge: shrinks coefficients, handles collinearity.
- Lasso: selects variables.
- Elastic net: combines both.

**Caution:** regularization is for prediction, not for causal effect estimation unless carefully designed. Selection can bias effect estimates.

## 6. Survival analysis

### 6.1 Key concepts

- **Time origin:** when follow-up starts.
- **Event:** the outcome of interest.
- **Censoring:** incomplete observation.
- **Competing risk:** an event that prevents the outcome.

### 6.2 Kaplan-Meier

Estimates survival over time without assuming a distribution. Useful for visualization and median survival.

### 6.3 Cox proportional hazards

\[
h(t|X) = h_0(t)\exp(\beta_1 X_1 + \cdots + \beta_p X_p)
\]

- **Hazard ratio:** relative instantaneous risk.
- **Proportional hazards assumption:** effects constant over time. Test it.
- **Time-varying covariates:** use extended Cox models or landmark analysis.

### 6.4 Competing risks

In older adults, death competes with readmission. Kaplan-Meier and Cox can overestimate the cumulative incidence of the event of interest. Use cumulative incidence functions and Fine-Gray or cause-specific hazards, depending on the question.

### 6.5 Immortal time bias

If treatment must be initiated after time zero, patients who receive it are guaranteed to survive until initiation. This creates immortal time. Use a landmark or time-varying exposure design.

## 7. Missing data

### 7.1 Mechanisms

- **MCAR:** missingness independent of everything.
- **MAR:** missingness depends on observed data.
- **MNAR:** missingness depends on unobserved data.

### 7.2 Methods

| Method | Assumption | When to use | Risk |
|---|---|---|---|
| Complete-case | MCAR | Low missingness, sensitivity analysis | Bias, power loss |
| Mean/median imputation | MCAR | Never for inference | Distorts variance |
| Multiple imputation | MAR | Standard for clinical data | Model misspecification |
| IPW | MAR | Missingness predicted by observed data | Extreme weights |
| Pattern mixture | MNAR | Sensitivity analysis | Complex |
| Missing indicator | Not recommended | Sometimes for prediction | Can encode bias |

### 7.3 Practical guidance

1. Report missingness by variable, site, and subgroup.
2. Distinguish structural, workflow, and biological missingness.
3. Use multiple imputation with auxiliary variables.
4. Include the outcome in the imputation model for prediction tasks, but avoid leakage.
5. Perform sensitivity analyses under MNAR.
6. Do not impute post-decision variables into a pre-decision model.

## 8. Causal inference

### 8.1 Potential outcomes

For each patient, there are potential outcomes under treatment \(Y(1)\) and control \(Y(0)\). The individual treatment effect is \(Y(1) - Y(0)\), which is never observed for the same patient at the same time.

Common estimands:

- **ATE:** average treatment effect in the population.
- **ATT:** average effect in the treated.
- **ATU:** average effect in the untreated.
- **CATE:** conditional average treatment effect.

### 8.2 Causal graphs

A directed acyclic graph (DAG) encodes assumptions about causal relationships.

| Structure | Pattern | Consequence |
|---|---|---|
| Chain | A → B → C | Adjusting for B blocks the effect of A on C |
| Fork | A ← B → C | B is a confounder; adjust for it |
| Collider | A → B ← C | Adjusting for B induces association |
| Descendant | A → B → C, D is descendant of C | Adjusting for D may bias |

**Rule:** use subject-matter knowledge to draw the DAG before seeing the data. Data-driven DAG discovery is not a substitute.

### 8.3 Confounding

A confounder is a common cause of treatment and outcome. Adjusting for confounders requires that they are measured, correctly specified, and not affected by treatment.

**Examples:**

- Sick patients receive more treatment (confounding by indication).
- Healthier patients are more likely to exercise (healthy-user bias).
- Patients with better access are more likely to receive screening.

### 8.4 Propensity scores

The propensity score is the probability of treatment given covariates:

\[
e(X) = P(T=1|X)
\]

Methods:

- **Matching:** pair treated and control patients with similar scores.
- **Stratification:** compare within propensity-score strata.
- **Inverse probability weighting (IPTW):** weight patients by the inverse probability of their observed treatment.
- **Doubly robust:** combine outcome regression and propensity weighting.

**Diagnostics:**

- Standardized mean differences < 0.1 after weighting.
- Overlap/common support.
- Weight distribution; trim extreme weights.
- Sensitivity to unmeasured confounding.

### 8.5 Instrumental variables

An instrument affects treatment but not the outcome except through treatment, and is not associated with unmeasured confounders. Examples: provider preference, geographic variation, lotteries.

**Caution:** instruments are hard to defend. Report the exclusion restriction and monotonicity assumptions.

### 8.6 Regression discontinuity

Use when treatment is assigned by a threshold (e.g., age, score, income). Compare patients just above and below the cutoff.

**Caution:** effects are local to the threshold and sensitive to bandwidth and manipulation.

### 8.7 Difference-in-differences

Compare pre/post changes in a treated group vs a control group.

\[
\hat{\tau} = (\bar{Y}_{T,post} - \bar{Y}_{T,pre}) - (\bar{Y}_{C,post} - \bar{Y}_{C,pre})
\]

**Assumption:** parallel trends absent treatment. Test pre-trends and consider event-study plots.

### 8.8 Synthetic control

Construct a weighted combination of untreated units that resembles the treated unit before intervention. Useful for policy or health-system interventions.

### 8.9 Target trial emulation

Specify the hypothetical randomized trial you would run, then emulate it with observational data.

| Protocol element | Question |
|---|---|
| Eligibility | Who would be enrolled? |
| Treatment strategies | What are the compared strategies? |
| Assignment | How would treatment be assigned? |
| Time zero | When does follow-up start? |
| Outcomes | What is measured, and when? |
| Follow-up | How long? |
| Causal contrast | ITT or per-protocol? |
| Analysis plan | How will confounding be addressed? |

**Common pitfalls:**

- Immortal time bias.
- Prevalent user bias.
- Misaligned time zero.
- Inconsistent eligibility.
- Informative censoring.
- Treatment switching.

## 9. Worked examples

### 9.1 Does a heart-failure program reduce readmission?

**Naive analysis:** compare 30-day readmission among program participants vs non-participants. Participants may be healthier or more engaged.

**Target trial:** eligible patients discharged with heart failure; treatment strategy = enrollment within 7 days vs usual care; time zero = discharge; outcome = 30-day readmission; adjustment for age, sex, comorbidity, EF, renal function, prior utilization, social support.

**Method:** IPTW with balance diagnostics; sensitivity analysis for unmeasured confounding; competing risk of death.

**Decision:** if the effect is small and uncertain, a costly program may not be justified.

### 9.2 Does an alert reduce time to antibiotics?

**Challenge:** alerts are deployed at a specific time; sicker patients may trigger earlier. A pre/post analysis can be confounded by seasonality, staffing, and case mix.

**Design:** interrupted time series with a comparison ward; adjust for season, census, and case mix; examine alert fatigue over time.

### 9.3 Is a biomarker causally related to disease?

**Challenge:** biomarkers are often measured because of symptoms. A cross-sectional association may be reverse causation.

**Design:** prospective cohort with pre-diagnostic samples; Mendelian randomization if a genetic instrument is available; adjust for confounders; replicate in independent cohorts.

## 10. Industry case studies

### 10.1 Hormone replacement therapy

Observational studies suggested that hormone replacement therapy reduced coronary heart disease in postmenopausal women. Randomized trials found increased risk. The observational association was confounded by socioeconomic status and health behaviors.

**Industry lesson:** observational associations can reverse under randomization. Causal claims require design, not just adjustment.

Source: Manson, J. E. et al., "Estrogen plus progestin and the risk of coronary heart disease," *New England Journal of Medicine*, 2003. https://doi.org/10.1056/NEJMoa030808

### 10.2 Beta-blockers in heart failure

Early observational studies suggested beta-blockers might be harmful in heart failure. Randomized trials demonstrated mortality benefit in HFrEF. The observational signal was confounded by indication and disease severity.

**Industry lesson:** confounding by indication can produce apparently harmful effects for effective treatments.

Source: MERIT-HF Study Group, *Lancet*, 1999. https://doi.org/10.1016/S0140-6736(99)04440-2

### 10.3 Target trial emulation in EHR data

Target trial emulation has been used to study the effectiveness of treatments using observational data, including in COVID-19 and chronic disease. The method makes assumptions explicit and reduces avoidable biases.

**Industry lesson:** a protocol-like design improves credibility and supports regulatory conversations.

Source: Hernán, M. A., Robins, J. M., "Using Big Data to Emulate a Target Trial When a Randomized Trial Is Not Available," *American Journal of Epidemiology*, 2016. https://doi.org/10.1093/aje/kwv254

### 10.4 Sepsis treatment bundles

Observational studies of sepsis bundles are vulnerable to confounding by indication and temporal trends. Randomized trials and quasi-experimental designs provide stronger evidence.

**Industry lesson:** quality-improvement analytics should use interrupted time series or difference-in-differences, not naive before/after comparisons.

Source: Seymour, C. W. et al., "Assessment of Clinical Criteria for Sepsis," *JAMA*, 2016. https://doi.org/10.1001/jama.2016.0288

### 10.5 Algorithmic bias in care management

A widely used algorithm predicted health-care costs as a proxy for need. Because spending was lower for Black patients at the same level of need, the algorithm systematically under-referred Black patients to care management.

**Industry lesson:** the choice of label is a causal and ethical decision. A proxy can encode structural inequity.

Source: Obermeyer, Z. et al., *Science*, 2019. https://doi.org/10.1126/science.aax2342

## 11. Labs

### Lab 4.1 — Build a DAG and identify adjustment sets

**Objective:** Translate a clinical question into a DAG and identify confounders, colliders, and mediators.

**Steps:**

1. Choose a question (e.g., statin use and stroke).
2. List variables and relationships.
3. Draw the DAG.
4. Use a graphical criterion or software to find adjustment sets.
5. Explain why adjusting for a collider is harmful.
6. Write the assumptions.

**Deliverable:** DAG and memo.

**Acceptance criteria:** assumptions explicit; adjustment set justified; collider identified.

### Lab 4.2 — Propensity score weighting

**Objective:** Estimate a treatment effect with IPTW and diagnose balance.

**Data:** Simulated or public observational data.

**Steps:**

1. Fit a propensity model.
2. Compute weights and trimmed weights.
3. Check standardized mean differences.
4. Estimate the ATE with a marginal model.
5. Perform a sensitivity analysis for unmeasured confounding.
6. Compare with naive and regression estimates.

**Deliverable:** notebook and results table.

**Acceptance criteria:** balance achieved; weights not extreme; assumptions and limitations stated.

### Lab 4.3 — Target trial emulation

**Objective:** Write a protocol for an observational comparative-effectiveness study.

**Steps:**

1. Define eligibility, treatment strategies, assignment, time zero, outcomes, follow-up, and contrast.
2. Align time zero across arms.
3. Specify confounders and missing-data plan.
4. Pre-specify primary and sensitivity analyses.
5. Identify likely biases.

**Deliverable:** 2-page protocol.

**Acceptance criteria:** time zero consistent; immortal time avoided; estimand clear.

### Lab 4.4 — Survival analysis with competing risks

**Objective:** Compare readmission risk while accounting for death.

**Data:** Public or simulated cohort.

**Steps:**

1. Plot Kaplan-Meier curves.
2. Fit a Cox model and test proportional hazards.
3. Compute cumulative incidence with competing risks.
4. Compare naive and competing-risk estimates.
5. Interpret for a clinical audience.

**Deliverable:** analysis and one-page interpretation.

**Acceptance criteria:** censoring handled; competing risk addressed; interpretation avoids causal overreach.

### Lab 4.5 — Missing data sensitivity

**Objective:** Compare complete-case, multiple imputation, and MNAR sensitivity analyses.

**Steps:**

1. Introduce missingness under MCAR, MAR, and MNAR.
2. Run complete-case and multiple imputation.
3. Compare estimates and confidence intervals.
4. Perform a delta-adjustment sensitivity analysis.
5. Recommend a primary approach.

**Deliverable:** simulation report.

**Acceptance criteria:** mechanisms distinguished; results interpreted correctly; recommendation justified.

### Lab 4.6 — Bootstrap confidence intervals

**Objective:** Estimate uncertainty for a complex metric.

**Steps:**

1. Compute a risk difference or net benefit.
2. Use patient-level bootstrap.
3. Use cluster bootstrap by site.
4. Compare with analytic intervals.
5. Explain why they differ.

**Deliverable:** notebook and explanation.

**Acceptance criteria:** resampling unit correct; percentile and BCa intervals compared.

## 12. Assignment: causal analysis protocol

Write a 2,000-word protocol for a real or realistic comparative-effectiveness question:

1. Decision and estimand.
2. Causal assumptions and DAG.
3. Study design and target trial specification.
4. Data sources and cohort.
5. Confounding control.
6. Missing data.
7. Outcomes and follow-up.
8. Statistical analysis.
9. Sensitivity analyses.
10. Limitations and decision relevance.

## 13. Assessment rubric

| Criterion | Weight | Novice | Competent | Expert |
|---|---:|---|---|---|
| Estimand | 20% | Vague question | Clear population, contrast, outcome, time | Aligned to decision and policy |
| Identification | 25% | Lists confounders | Uses DAG and explicit assumptions | Anticipates colliders and unmeasured confounding |
| Estimation | 20% | Naive regression | Appropriate method with diagnostics | Doubly robust and sensitivity analysis |
| Missing data | 10% | Complete case only | Multiple imputation | MNAR sensitivity |
| Interpretation | 15% | Causal overreach | Cautious, effect sizes and intervals | Decision-ready with uncertainty |
| Communication | 10% | Technical only | Clear to clinicians | Persuasive and balanced |

## 14. Regulatory, ethical, and commercial lens

- **Regulatory evidence:** FDA and EMA accept RWE under specific conditions; pre-specification and data quality matter.
- **HTA:** cost-effectiveness models depend on causal effect estimates. Bias propagates into price and coverage decisions.
- **Equity:** a causal effect may differ across subgroups. Report heterogeneity and avoid assuming average effects apply universally.
- **Transparency:** pre-register the protocol where possible; report all outcomes.
- **Liability:** causal claims can create legal exposure. Be precise about what the data support.

## 15. Failure modes and red flags

- Adjusting for mediators or colliders.
- Using future information to define exposure.
- Immortal time bias.
- Comparing responders to non-responders.
- Ignoring competing risks.
- Treating odds ratios as risk ratios.
- Reporting p-values without effect sizes.
- Multiple testing without correction.
- Assuming missingness is random.
- Using feature importance as a causal effect.
- Extrapolating trial results to populations not studied.

## 16. Interview and discussion questions

1. **What is an estimand?**
   Hint: population, treatment contrast, outcome, time horizon, summary measure.
2. **Prediction vs causation: why does it matter?**
   Hint: different targets, assumptions, validation, and decisions.
3. **What is a collider? Give a clinical example.**
   Hint: hospitalization is a collider between disease and access.
4. **When would you use propensity scores?**
   Hint: observational treatment comparison with measured confounding and overlap.
5. **What is immortal time bias?**
   Hint: follow-up time during which the outcome cannot occur.
6. **How do you handle competing risks?**
   Hint: cumulative incidence, cause-specific or subdistribution hazards.
7. **What is the difference between MCAR and MAR?**
   Hint: missingness depends on observed vs unobserved data.
8. **Why can a randomized trial still be biased?**
   Hint: selection, attrition, adherence, unblinding, measurement, analysis.
9. **How would you test the proportional hazards assumption?**
   Hint: Schoenfeld residuals, interaction with time, graphical checks.
10. **What is a doubly robust estimator?**
    Hint: consistent if either the outcome model or the treatment model is correct.
11. **How do you choose a negative control?**
    Hint: an outcome that should not be affected by the exposure.
12. **What would make you distrust an observational effect estimate?**
    Hint: no pre-specification, poor overlap, implausible magnitude, sensitivity to adjustment.

## 17. Key terms

| Term | Definition |
|---|---|
| Estimand | Precise definition of the quantity to estimate |
| ATE / ATT / CATE | Average treatment effect / in treated / conditional |
| Confounder | Common cause of treatment and outcome |
| Collider | Common effect of two variables; conditioning induces bias |
| Mediator | Variable on the causal path |
| DAG | Directed acyclic graph of assumptions |
| Propensity score | Probability of treatment given covariates |
| IPTW | Inverse probability of treatment weighting |
| Doubly robust | Estimator consistent under one of two models |
| Instrumental variable | Variable affecting treatment but not outcome except through treatment |
| Difference-in-differences | Pre/post comparison with control group |
| Regression discontinuity | Threshold-based identification |
| Target trial emulation | Emulating a hypothetical randomized trial |
| Immortal time | Follow-up during which the outcome cannot occur |
| Censoring | Incomplete observation of the event time |
| Competing risk | Event that prevents the outcome of interest |
| Hazard ratio | Relative instantaneous hazard |
| Multiple imputation | Repeated imputation and pooling |
| MNAR | Missing not at random |
| Sensitivity analysis | Quantifying robustness to assumptions |

## 18. Sources and further reading

### Causal inference
- Hernán, M. A., Robins, J. M., *Causal Inference: What If*. https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/
- Hernán, M. A., Robins, J. M., "Using Big Data to Emulate a Target Trial When a Randomized Trial Is Not Available," *American Journal of Epidemiology*, 2016. https://doi.org/10.1093/aje/kwv254
- Pearl, J., *Causality: Models, Reasoning, and Inference*.
- Imbens, G. W., Rubin, D. B., *Causal Inference for Statistics, Social, and Biomedical Sciences*.
- Rosenbaum, P. R., Rubin, D. B., "The central role of the propensity score in observational studies for causal effects," *Biometrika*, 1983. https://doi.org/10.1093/biomet/70.1.41

### Statistics and survival
- Harrell, F. E., *Regression Modeling Strategies*.
- Steyerberg, E. W., *Clinical Prediction Models*.
- Klein, J. P., Moeschberger, M. L., *Survival Analysis: Techniques for Censored and Truncated Data*.
- Little, R. J. A., Rubin, D. B., *Statistical Analysis with Missing Data*.
- Efron, B., Tibshirani, R. J., *An Introduction to the Bootstrap*.

### Clinical examples
- Manson, J. E. et al., "Estrogen plus progestin and the risk of coronary heart disease," *New England Journal of Medicine*, 2003. https://doi.org/10.1056/NEJMoa030808
- MERIT-HF Study Group, "Effect of metoprolol CR/XL in chronic heart failure," *Lancet*, 1999. https://doi.org/10.1016/S0140-6736(99)04440-2
- Seymour, C. W. et al., "Assessment of Clinical Criteria for Sepsis," *JAMA*, 2016. https://doi.org/10.1001/jama.2016.0288
- Obermeyer, Z. et al., "Dissecting racial bias in an algorithm used to manage the health of populations," *Science*, 2019. https://doi.org/10.1126/science.aax2342

### Tools
- DoWhy: https://py-why.github.io/dowhy/
- EconML: https://econml.azurewebsites.net/
- lifelines: https://lifelines.readthedocs.io/
- statsmodels: https://www.statsmodels.org/

> **Next:** [Module 05](05_study_design_evidence.md) turns causal reasoning into a full evidence-generation strategy.
