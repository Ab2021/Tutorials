# Validation Designs

> Module 05 — Study Design and Evidence Generation
> Topic 04 of 10
> Estimated study time: 4 hours
> Prerequisites: Topics 01 and 02
> Version 2026.3

## 1. Why this topic exists

A model's performance on the data that produced it is not evidence. It is a description of the fitting procedure. The number that matters is performance on patients the model has never seen, in the setting where it will be used, at the moment the decision is made. Validation is the set of designs that produces that number.

The distinction is not academic. A model with an apparent area under the receiver operating characteristic curve (AUC) of 0.85 can fall to 0.65 on external validation, and a model with good discrimination can be badly calibrated. Both failures change the clinical decision. A model that ranks patients well but overestimates risk by a factor of two will trigger unnecessary interventions. A model that is calibrated on average but miscalibrated in the elderly will harm the subgroup that needs it most.

The regulatory and commercial stakes have risen with the number of AI-enabled devices. The US Food and Drug Administration (FDA) has authorised more than 1,600 AI-enabled devices as of September 2026, according to its device centre. The European Union's Medical Device Regulation (MDR) requires clinical evaluation, and the Artificial Intelligence Act adds obligations for high-risk systems. Payers and hospital procurement committees increasingly require local validation before adoption. A vendor that cannot produce a credible validation package is excluded from the market regardless of the model's development performance.

Validation is also where data scientists have the most direct influence. Development is often constrained by data access and computational resources. Validation is a design problem: which populations, which time periods, which sites, which metrics, and which decision thresholds. A data scientist who understands validation design can prevent a launch that would fail, or rescue a product whose development metrics were modest but whose validation is strong.

### The four questions validation must answer

1. **Does the model discriminate?** Can it separate patients who experience the outcome from those who do not?
2. **Is the model calibrated?** Do predicted risks match observed risks, overall and across the range?
3. **Is the model useful?** Does it improve decisions compared with the current pathway, at a plausible threshold?
4. **Does the model transport?** Does performance hold in a different site, population, or era?

A validation study that answers only the first question is incomplete. A study that answers the first two but not the third is a performance study, not an evaluation. A study that answers all four but ignores subgroups may still fail in deployment.

### Why internal validation is not enough

Internal validation, such as cross-validation or bootstrap, estimates how well the model performs on new patients from the same population and data-generating process. It answers the question "did we overfit the development sample?" It does not answer the question "will this model work in another hospital?"

The gap between internal and external performance has three sources.

- **Case-mix differences.** The external population may have a different distribution of predictors or a different baseline risk.
- **Measurement differences.** The same predictor may be measured differently, or the outcome may be ascertained differently.
- **Care-pathway differences.** The clinical action triggered by the model may differ, changing the relationship between prediction and outcome.

Internal validation cannot detect any of these. Only temporal, geographic, or prospective validation can.

### The cost of skipping validation

The cost appears in three forms.

**Clinical harm.** A model that is deployed without external validation may systematically underestimate risk in the population it is applied to, leading to under-treatment of high-risk patients.

**Commercial failure.** A product that fails local validation at a flagship site loses credibility, and the failure is public. Procurement cycles are long, and a failed validation is difficult to reverse.

**Regulatory delay.** A submission that relies on development performance, or on internal validation alone, is likely to receive a request for additional evidence. The delay can exceed the time required to have conducted the validation study in the first place.

### The validation stack in practice

A complete validation package for a deployed model usually contains five layers:

- a development report with apparent performance and optimism correction;
- a temporal validation in the development sites;
- at least one external validation in a different site or population;
- a prospective silent-mode evaluation in the intended workflow;
- a monitoring plan with recalibration triggers.

The layers are cumulative. A package with the first two layers supports a research claim. A package with the first four supports a deployment claim. A package with all five supports a sustained deployment claim.

The package should also contain the model specification, the predictor mapping, the outcome definition, the missing-data plan, and the analysis code. Without these, the validation cannot be reproduced, and a regulator or payer cannot verify it.

### What validation is not

Validation is not a single number. It is not a certificate that a model is safe. It is not a one-time event. And it is not a substitute for clinical judgement.

A model that passes validation can still be harmful if it is used outside its intended population, if the action it triggers is ineffective, or if clinicians over-rely on it. Validation establishes that the model performs as claimed in a defined population; it does not establish that the claim is worth acting on. That is the role of the impact evaluation and the clinical governance process.

## 2. Learning outcomes

By the end of this topic you can:

1. **Distinguish** apparent, internal, temporal, geographic, external, prospective, and impact validation, and select the level appropriate to a claim. (Exercise 1, self-check 1)
2. **Compute** discrimination (AUC, sensitivity, specificity), calibration (slope, intercept, observed-to-expected ratio), and clinical utility (net benefit) from a validation dataset. (Exercises 2 and 3, self-check 3)
3. **Estimate** optimism with bootstrap or cross-validation and explain why apparent performance is optimistic. (Exercise 4, self-check 4)
4. **Size** an external validation study using event-count and precision criteria. (Exercise 5, self-check 5)
5. **Diagnose** spectrum shift, calibration drift, and missing-predictor problems, and propose recalibration or restriction. (Exercise 6, interview 5)
6. **Design** a prospective silent-mode or impact evaluation for a model that has passed external validation. (Exercise 7, interview 7)
7. **Map** a validation package to TRIPOD+AI and PROBAST+AI, and identify the reporting items that a regulator or reviewer will check. (Exercise 8, interview 8)

### How the outcomes map to practice

- Outcomes 1 and 2 are used in the design of a validation protocol.
- Outcomes 3 and 4 are used in the analysis and precision sections.
- Outcome 5 is used in the review of an external validation report.
- Outcome 6 is used in the deployment decision.
- Outcome 7 is used in the regulatory or payer submission.
- Outcome 8 is used in the critical appraisal of a published study.

A learner who can perform outcomes 1 to 4 can design a validation study. A learner who can also perform outcomes 5 to 8 can defend it in review.

## 3. The core mental model

Validation is a ladder, and each rung answers a different question. The ladder is not a hierarchy of prestige; it is a sequence of increasingly strong claims about transportability and impact.

```text
   DEVELOPMENT DATA
        |
        | apparent performance (optimistic)
        v
   INTERNAL VALIDATION
   cross-validation | bootstrap optimism
        |  "Did we overfit?"
        v
   TEMPORAL VALIDATION
   later data, same sites
        |  "Does performance drift over time?"
        v
   GEOGRAPHIC / EXTERNAL VALIDATION
   different sites, different populations
        |  "Does performance transport?"
        v
   PROSPECTIVE VALIDATION
   silent mode, real-time data, intended workflow
        |  "Does it work in live use?"
        v
   IMPACT EVALUATION
   randomised or strong quasi-experiment
        |  "Does it improve outcomes or cost?"
        v
   MONITORING AND RECALIBRATION
   post-deployment surveillance
```

The ladder is cumulative in evidence, not in method. A model can pass external validation and fail prospective validation because the data pipeline differs in live use. A model can pass prospective validation and fail impact evaluation because clinicians ignore it. Each rung tests a different link in the chain from prediction to benefit.

### Three families of metrics

Validation metrics fall into three families. A complete validation reports all three.

**Discrimination.** The ability to separate patients with and without the outcome. The AUC is the most common measure; sensitivity and specificity at a threshold are more decision-relevant. Discrimination is invariant to the base rate, which is both a strength and a limitation: a model can discriminate well and still be badly calibrated.

**Calibration.** The agreement between predicted and observed risk. Calibration-in-the-large (the intercept) measures systematic over- or under-prediction. The calibration slope measures whether the model's predictions are too extreme or too conservative. Calibration plots show the relationship across the risk range.

**Clinical utility.** The net benefit of using the model at a decision threshold, compared with treating everyone or treating no one. Decision curve analysis expresses utility in units of net true positives, accounting for the harm of false positives through the threshold odds.

A model that improves discrimination but worsens calibration may reduce net benefit. A model that improves calibration but not discrimination may still be useful if the current pathway is poorly calibrated. The three families must be interpreted together.

### The transportability problem

Transportability is the question of whether a model developed in one population applies in another. It depends on three things:

1. **The predictor-outcome relationship.** If the relationship is stable, the model transports. If it differs by population, the model does not.
2. **The distribution of predictors.** Even with a stable relationship, a different distribution changes the case mix and the calibration.
3. **The measurement of predictors and outcomes.** If the same variable is measured differently, the model's inputs change.

A model can be transportable in one dimension and not another. A model with a stable predictor-outcome relationship but a different case mix may discriminate well and be badly calibrated. The design response is recalibration, not necessarily redevelopment.

### The validation gap

The difference between apparent and validated performance is the validation gap. It has two components: optimism and transport.

**Optimism** is the inflation caused by evaluating the model on the data that selected it. It is estimated by internal validation and is usually small for large development samples and few predictors, and large for small samples and many predictors.

**Transport** is the change caused by applying the model to a different population or setting. It is estimated by temporal, geographic, or prospective validation and can be large even when optimism is small.

A validation report that reports only the apparent AUC conflates the two and overstates the model's performance. A report that reports internal validation only addresses optimism and leaves transport unaddressed.

### Validation level and question

| Level | Question answered | Data required | Claim supported |
|---|---|---|---|
| Apparent | How well does the model fit? | Development data | None for deployment |
| Internal | Did we overfit? | Development data, resampled | Research finding |
| Temporal | Is performance stable over time? | Later data, same sites | Use with monitoring |
| Geographic | Does it work at other similar sites? | Data from other sites | Multi-site deployment |
| External | Does it work in a different population? | Data from the target population | Target-population deployment |
| Prospective | Does it work in live data and workflow? | Post-deployment data | Live deployment |
| Impact | Does it change outcomes or cost? | Comparison of strategies | Benefit claim |

### The decision threshold

A model's clinical value depends on the threshold at which it triggers an action. The threshold is the probability above which the expected benefit of the action exceeds the expected harm. It is a clinical and economic judgement, not a statistical one.

For a readmission model, the threshold might be 20 percent: above that risk, a transitional-care programme is offered. For a sepsis alert, the threshold might be 5 percent: above that risk, a bedside review is triggered. The threshold determines the sensitivity, specificity, and net benefit. A validation that reports only the AUC hides the threshold choice and the consequences of getting it wrong.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Apparent performance | Performance measured on the development data | Reported as if it were validation | Metric value |
| Internal validation | Performance estimated on held-out or resampled development data | Confused with external validation | Metric value |
| Cross-validation | Repeated splitting of development data into training and test folds | Used with leakage across folds | Procedure |
| Bootstrap optimism | The average difference between bootstrap-apparent and test performance | Estimated with too few replicates | Metric value |
| Temporal validation | Validation on later data from the same sites | Confused with external validation | Design |
| Geographic validation | Validation on data from different sites in the same era | Confused with temporal validation | Design |
| External validation | Validation on a different population or setting | Used loosely for any non-development data | Design |
| Prospective validation | Validation on data collected after model lock, in the intended workflow | Confused with a prospective cohort | Design |
| Silent-mode evaluation | Prospective validation in which the model runs but does not affect care | Confused with impact evaluation | Design |
| Impact evaluation | A study of whether the model changes outcomes or cost | Confused with validation | Design |
| Discrimination | The ability to separate outcome classes | Reported as the only performance measure | AUC, c-statistic |
| AUC | Area under the receiver operating characteristic curve | Interpreted as clinical utility | 0.5 to 1.0 |
| c-statistic | The concordance statistic, equivalent to AUC for binary outcomes | Used interchangeably without definition | 0.5 to 1.0 |
| Sensitivity | The proportion of true positives detected at a threshold | Reported without the threshold | Proportion |
| Specificity | The proportion of true negatives correctly identified | Reported without the threshold | Proportion |
| Positive predictive value (PPV) | The proportion of positive predictions that are true | Confused with sensitivity | Proportion |
| Calibration | Agreement between predicted and observed risk | Assessed only with the Hosmer-Lemeshow test | Calibration plot |
| Calibration-in-the-large | The average predicted risk minus the average observed risk | Confused with the calibration slope | Intercept |
| Calibration slope | The slope of observed outcome on predicted log-odds | Ignored in favour of AUC | Unitless |
| Observed-to-expected ratio (O:E) | Observed events divided by expected events | Reported without a confidence interval | Ratio |
| Recalibration | Updating the intercept or slope of an existing model | Confused with redevelopment | Procedure |
| Net benefit | The clinical utility of a model at a threshold | Confused with accuracy | Probability units |
| Decision curve analysis | A plot of net benefit across thresholds | Used without a plausible threshold range | Curve |
| Spectrum bias | Performance differs because the case mix differs | Ignored when validating in a tertiary centre | Bias category |
| Case-mix shift | A change in the distribution of predictors or baseline risk | Confused with model drift | Distribution change |
| Model drift | Degradation of performance over time | Assumed to be absent | Temporal change |
| Data leakage | Information from the outcome or test set entering development | Undetected in cross-validation | Bias category |
| TRIPOD+AI | Reporting guideline for prediction models using regression or AI | Used as a design checklist | Reporting guideline |
| PROBAST+AI | Risk-of-bias and applicability tool for prediction models | Confused with TRIPOD+AI | Appraisal tool |
| Predetermined change control plan (PCCP) | A pre-authorised plan for model updates | Confused with a post-market change | Regulatory document |
| Good machine learning practice (GMLP) | A set of guiding principles for ML-enabled devices | Treated as a regulation | Principles |

Two distinctions deserve emphasis.

**Validation versus verification.** Verification asks whether the software implements the intended algorithm correctly. Validation asks whether the algorithm achieves the intended clinical performance. A model can be verified and fail validation, or validated and fail verification. Both are necessary, and they are different activities.

**Recalibration versus redevelopment.** Recalibration updates the intercept or slope of an existing model to a new population. Redevelopment refits the model coefficients. Recalibration is preferred when the predictor-outcome relationships are stable and only the baseline risk or case mix has changed. Redevelopment is necessary when the relationships themselves differ.

## 5. The mechanics, step by step

The procedure has fourteen steps. Steps 1 to 4 define the claim and lock the model. Steps 5 to 7 prepare the validation data. Steps 8 to 11 compute and interpret performance. Steps 12 to 14 address prospective use, impact, and monitoring.

### Procedure at a glance

| Step | Input | Output | Failure if skipped |
|---:|---|---|---|
| 1 | Claim and pathway | Intended-use statement | Validation for an undefined use |
| 2 | Intended use | Performance estimand | No target to test |
| 3 | Developed model | Locked model version | Refitting during validation |
| 4 | Claim and data | Validation level | Evidence that cannot support the claim |
| 5 | Event rate and precision | Sample size | Underpowered calibration |
| 6 | Model variables | Variable mapping | Silent predictor mismatch |
| 7 | Validation data | Case-mix table | Misread transportability |
| 8 | Predictions and outcomes | AUC and threshold metrics | Discrimination without decisions |
| 9 | Predictions and outcomes | Calibration slope and plot | Wrong risk thresholds |
| 10 | Predictions and outcomes | Net benefit | Accuracy without utility |
| 11 | Subgroups | Fairness report | Hidden subgroup harm |
| 12 | Locked model | Silent-mode report | Pipeline failure at launch |
| 13 | Validated model | Impact estimate | Unsupported benefit claim |
| 14 | Deployed model | Monitoring plan | Silent drift |

### Metrics: formulas and interpretation

| Metric | Formula or definition | Interpretation | Common error |
|---|---|---|---|
| AUC | Probability that a random case ranks above a random non-case | 0.5 is chance; 1.0 is perfect | Treated as clinical utility |
| Sensitivity | TP / (TP + FN) at a threshold | Case detection | Reported without threshold |
| Specificity | TN / (TN + FP) at a threshold | Non-case identification | Reported without threshold |
| PPV | TP / (TP + FP) | Probability a positive is true | Confused with sensitivity |
| Calibration intercept | Alpha in logit(Y) = alpha + beta * logit(p) | Systematic over- or under-prediction | Ignored when slope is reported |
| Calibration slope | Beta in the same model | 1 is perfect; below 1 is over-fitting or shift | Reported without a plot |
| O:E ratio | Observed events / expected events | 1 is perfect; above 1 is under-prediction | Reported without a CI |
| Net benefit | TP/N - FP/N * (p_t / (1 - p_t)) | Utility at a threshold | Compared without treat-all |
| Brier score | Mean squared difference between predicted risk and outcome | Overall accuracy | Interpreted without a baseline |

The table is a reference for the analysis steps. The formulas are simple, but the interpretation is not: the same AUC can correspond to very different clinical value depending on calibration and threshold.

### Sample-size rules for validation

Use the following rules as a starting point, not as a substitute for a precision calculation.

- At least 100 events and 100 non-events for basic discrimination and calibration.
- At least 200 events and 200 non-events for flexible calibration curves.
- At least 10 events per predictor for a stable calibration model.
- More events for subgroup analyses, with a prespecified minimum per subgroup.
- A simulation-based calculation when the target is a confidence interval width.
- An inflation for missing predictors or outcome data.
- A clustered design effect if the validation is multi-site.
- A separate sample for recalibration, not the sample used to report performance.

### Recalibration methods

| Method | What it updates | When to use | Risk |
|---|---|---|---|
| Intercept-only recalibration | Baseline risk | Case-mix shift with stable slopes | Does not fix slope error |
| Slope and intercept recalibration | Baseline risk and extremeness | Slope away from 1 | Needs a separate sample |
| Model revision | Coefficients of existing predictors | Relationships changed | Requires revalidation |
| Model extension | Adds new predictors | Missing prognostic information | Overfitting; new validation |
| Redevelopment | New model | Population or outcome changed substantially | Full development and validation |

Recalibration is not a substitute for redevelopment when the predictor-outcome relationships have changed. A recalibrated model with the wrong predictors can be well calibrated on average and useless for decision-making.

### Interpreting calibration

Calibration has three common patterns, each with a different response.

- **Intercept shifted, slope near 1.** The model ranks well but the baseline risk is different. Intercept-only recalibration is sufficient.
- **Slope below 1, intercept shifted.** The predictions are too extreme and the baseline is different. Slope and intercept recalibration is needed, with a separate sample.
- **Slope above 1.** The predictions are too conservative; the model may be underfitting. Investigate whether the development sample was small or the predictors were coarse.
- **Non-linear calibration.** The model is calibrated in the middle but not at the extremes. A flexible recalibration or a new model is needed.
- **Subgroup-specific miscalibration.** The overall calibration is acceptable but a subgroup is miscalibrated. Recalibrate by subgroup or restrict the claim.

A calibration plot should accompany every calibration statistic. The slope and intercept summarise the plot, but they can hide a non-linear pattern that matters at the decision threshold.

### Common questions in validation review

A validation review usually asks the following questions. The protocol should answer them before the review.

- What is the intended use, and does the validation population match it?
- Was the model locked before validation?
- How many events were in each validation cohort?
- Was the outcome definition the same in development and validation?
- Were all predictors available in the validation data?
- What was the case mix, and how did it differ?
- What were the discrimination, calibration, and net benefit?
- Were the subgroups prespecified, and what did they show?
- Was there a prospective evaluation?
- Is there a monitoring and recalibration plan?

### Validation data requirements

| Requirement | Why it matters | Evidence to provide |
|---|---|---|
| Consecutive or randomly sampled patients | Avoids selection bias | Sampling description |
| Complete outcome ascertainment | Avoids differential missingness | Linkage and follow-up description |
| Predictors measured at the same time point | Avoids look-ahead bias | Data collection timeline |
| Same predictor definitions | Avoids measurement mismatch | Variable mapping table |
| Sufficient events | Precision | Event count and CI width |
| Representation of the target population | Transportability | Case-mix table |
| Site and calendar coverage | Generality | Site and date listing |
| Data provenance | Reproducibility | Extraction and cleaning documentation |

### Common validation pitfalls by stage

- **Design stage:** choosing a validation cohort because it is available, not because it matches the intended use.
- **Data stage:** mapping predictors by name rather than by definition and unit.
- **Analysis stage:** refitting the model on the validation data.
- **Reporting stage:** reporting the AUC without calibration or net benefit.
- **Interpretation stage:** claiming generalisability from a single external site.
- **Deployment stage:** assuming retrospective performance will hold in live data.
- **Monitoring stage:** having no trigger for recalibration.

### Step 1: Define the intended use and the claim

**Input:** the product claim and the clinical pathway.

**Output:** an intended-use statement with population, user, setting, input data, output, decision, and action.

**Rationale:** Validation is always validation for a use. A model validated for predicting readmission in a tertiary hospital is not validated for predicting readmission in a community hospital, even if the code is the same.

The intended-use statement should specify the decision threshold and the action. "Predicts 30-day readmission risk" is not an intended use. "Predicts 30-day readmission risk at discharge for adults admitted with heart failure, to trigger a transitional-care referral at a threshold of 20 percent" is.

### Step 2: Define the performance estimand

**Input:** the intended use and the decision.

**Output:** a primary performance metric, a decision threshold, and a target performance level.

**Rationale:** The performance estimand determines the validation design. A claim about discrimination requires a different sample size and analysis from a claim about calibration or net benefit.

The estimand should state:

- the population,
- the prediction horizon,
- the outcome definition,
- the metric (AUC, calibration slope, O:E, net benefit),
- the threshold, if applicable,
- the target performance and the acceptable lower bound.

A claim of "good discrimination" is not a performance estimand. A claim of "AUC at least 0.75 with a lower 95 percent confidence bound above 0.70" is.

### Step 3: Lock the model and the preprocessing

**Input:** the developed model, including coefficients, preprocessing, and imputation.

**Output:** a locked model specification with a version identifier.

**Rationale:** Validation must evaluate a fixed model. If the model is refitted during validation, the result is development performance, not validation performance.

The lock should include the predictor definitions, the preprocessing steps, the imputation rules, the coefficient values, and the decision threshold. Any change after the lock creates a new model version and requires a new validation.

### Step 4: Choose the validation level

**Input:** the claim, the available data, and the regulatory context.

**Output:** a validation design with a justification and a list of rejected alternatives.

**Rationale:** The validation level determines what the evidence can support. A claim of local performance needs local validation; a claim of general performance needs multi-site external validation; a claim of impact needs an impact study.

| Claim | Minimum validation | What it supports |
|---|---|---|
| The model fits the development population | Internal | A research finding, not deployment |
| The model is stable over time | Temporal | Use in the same sites, with monitoring |
| The model transports to similar sites | Geographic external | Use across comparable sites |
| The model transports to a different population | External, ideally multi-site | Use in the target population |
| The model works in live use | Prospective silent mode | Deployment with oversight |
| The model improves outcomes | Impact evaluation | A clinical benefit claim |
| The model is safe for autonomous use | Prospective plus impact plus monitoring | A narrow autonomous claim |

The table is a minimum, not a guarantee. A model can meet the minimum and still fail because of workflow, data pipeline, or human factors.

### Step 5: Size the validation study

**Input:** the expected event rate, the target metric, and the desired precision.

**Output:** a sample size with an events-based justification.

**Rationale:** Validation precision depends on the number of events, not the number of records. A study with 10,000 records and 100 events has the precision of a small study for calibration.

The common rules are:

- **100 events and 100 non-events** for a basic validation of discrimination and calibration.
- **200 events and 200 non-events** for flexible calibration curves.
- **Larger samples** for precise calibration slope, O:E ratio, and subgroup analyses.
- **Simulation-based sizing** when the target is a confidence interval width for a specific metric.

A useful check is to compute the expected confidence interval width for the calibration slope at the planned sample size. If the width is too wide to distinguish acceptable from unacceptable calibration, the study is underpowered for its claim.

### Step 6: Define the outcome and the predictors in the validation data

**Input:** the locked model and the validation dataset.

**Output:** a mapping from model variables to validation variables, with a missing-data plan.

**Rationale:** The same variable name in two datasets may not mean the same thing. A model that uses "creatinine" in development may encounter "serum creatinine" measured at a different laboratory, with different units and reference ranges.

The mapping should be documented variable by variable, with the source, the unit, the transformation, and the missingness rate. Any variable that cannot be mapped exactly should be flagged, and the validation should report the performance with and without it.

### Step 7: Assess data quality and case mix

**Input:** the validation dataset and the development dataset.

**Output:** a table comparing predictor distributions, outcome rates, and missingness.

**Rationale:** A validation in a population with a different case mix tests transportability, but it may also test a different question. The comparison table tells the reader whether the validation is a test of the model or a test of a new population.

The table should include the development and validation means or proportions for each predictor, the outcome rate, and the standardised differences. A large standardised difference suggests a case-mix shift that may require recalibration.

### Step 8: Compute discrimination

**Input:** predicted risks and observed outcomes.

**Output:** AUC with a confidence interval, and sensitivity and specificity at the decision threshold.

**Rationale:** Discrimination is the most familiar metric and the easiest to report. It is also the least sufficient. A model can discriminate well and be useless because it is miscalibrated or because the threshold is wrong.

The AUC should be reported with a confidence interval, ideally from a method that accounts for clustering if the data are clustered by site. The threshold-specific metrics should be reported with the threshold stated, because sensitivity and specificity are not intrinsic to the model.

### Step 9: Compute calibration

**Input:** predicted risks and observed outcomes.

**Output:** calibration-in-the-large, calibration slope, O:E ratio, and a calibration plot.

**Rationale:** Calibration determines whether the predicted probabilities can be used as probabilities. A model that predicts 30 percent for a group with a 15 percent observed rate will over-treat that group.

The standard analysis fits a logistic regression of the outcome on the predicted log-odds:

```text
logit(P(Y = 1)) = alpha + beta * logit(predicted risk)
```

The intercept alpha measures calibration-in-the-large; the slope beta measures whether the predictions are too extreme (beta < 1) or too conservative (beta > 1). A perfectly calibrated model has alpha = 0 and beta = 1. The O:E ratio is the observed number of events divided by the expected number.

The calibration plot groups patients by predicted risk and plots observed against predicted, with a smoother and confidence bands. The plot is more informative than a single test, because it shows where the model fails.

### Step 10: Compute clinical utility

**Input:** predicted risks, observed outcomes, and a threshold range.

**Output:** a decision curve with net benefit, and the range of thresholds where the model is better than the alternatives.

**Rationale:** Discrimination and calibration describe the model. Net benefit describes the decision. A model can have good discrimination and calibration and still have no net benefit if the threshold is outside the range where it adds value.

Net benefit at a threshold p_t is:

```text
NB = (TP / N) - (FP / N) * (p_t / (1 - p_t))
```

where TP is the number of true positives, FP the number of false positives, and N the total sample. The decision curve plots NB against p_t for the model, for treat-all, and for treat-none. The model is useful where its curve is above both alternatives.

### Step 11: Analyse subgroups and fairness

**Input:** predicted risks, outcomes, and subgroup variables.

**Output:** subgroup performance with confidence intervals, and a statement of any material differences.

**Rationale:** A model that performs well overall can perform poorly in a subgroup. The subgroup analysis is not a search for significance; it is a check that the model's performance is adequate for the populations it will serve.

The subgroups should be prespecified and justified by clinical or equity concerns: age, sex, race and ethnicity, socioeconomic status, site, and relevant comorbidity. The analysis should report discrimination, calibration, and net benefit by subgroup, not only the AUC.

### Step 12: Prospective silent-mode validation

**Input:** the externally validated model and the deployment data pipeline.

**Output:** a prospective performance report with real-time data, and a comparison with the retrospective validation.

**Rationale:** Retrospective validation uses data that have already been cleaned and curated. Prospective validation uses the data as they arrive, with the missingness and timing of live use. The difference can be substantial.

Silent mode runs the model in the background, records its predictions, and does not show them to clinicians. It tests the data pipeline and the model's performance without changing care. It is the appropriate design before a live pilot.

### Step 13: Impact evaluation

**Input:** a model that has passed prospective validation, and a clinical action.

**Output:** an estimate of the effect on outcomes, cost, or process, with a design that supports causal inference.

**Rationale:** A model that predicts accurately does not necessarily improve outcomes. The action it triggers may be ineffective, or clinicians may ignore it, or the workflow may not support it. An impact evaluation tests the whole pathway.

The strongest design is a randomised trial of the model-plus-action strategy versus usual care. A stepped-wedge or a difference-in-differences design is acceptable when randomisation is infeasible. A before-after study is weak because secular trends can explain the change.

### Step 14: Monitor and recalibrate

**Input:** post-deployment predictions and outcomes.

**Output:** a monitoring plan with triggers for recalibration or redevelopment.

**Rationale:** Performance drifts as populations, coding, and care change. A model that was calibrated at launch may be miscalibrated a year later. Monitoring detects drift before it causes harm.

The plan should specify the metrics, the frequency, the data source, the acceptable range, and the action when the range is exceeded. A common trigger is a calibration slope outside 0.8 to 1.2 or an O:E ratio outside 0.8 to 1.25.

### Edge cases

**Rare outcomes.** A rare outcome produces few events, and calibration becomes imprecise. The validation should report the event count and use methods appropriate for rare events, such as penalised calibration or exact confidence intervals. A validation with 30 events cannot support a claim of good calibration.

**Spectrum shift.** A validation population with a different case mix can change discrimination and calibration even if the model is transportable. The report should compare the case mix and, where possible, reweight the validation sample to the target population.

**Missing predictors.** A predictor that is missing in the validation data cannot be used. The options are to omit the predictor and refit a reduced model, to impute it, or to restrict the validation to patients with complete data. Each option changes the model or the population, and the choice must be reported.

**Cluster effects.** If the validation data are clustered by site, the confidence intervals must account for clustering. A model validated across 10 sites with 100 patients each has less independent information than 1,000 patients from one site.

**Recalibration.** When the calibration slope is far from 1, recalibration may be necessary before deployment. The recalibration should be fitted on a separate dataset or with cross-validation, not on the validation data used to report performance.

**Regulatory updates.** A model that is updated after authorisation may require a new submission unless the update is within a predetermined change control plan. The validation package should state which version is being validated and which changes are covered.

### Decision table: validation design by claim

| Claim | Primary design | Sample size driver | What would falsify it |
|---|---|---|---|
| Local performance | Internal validation | Events per predictor | Optimism-corrected AUC below target |
| Stability over time | Temporal validation | Events in the later period | Calibration drift |
| Transport to similar sites | Geographic external | Events per site | Site-specific miscalibration |
| Transport to a new population | External, multi-site | Events in the target | AUC or calibration below target |
| Live-use performance | Prospective silent mode | Events after deployment | Pipeline failure or drift |
| Clinical impact | Randomised or stepped-wedge | Outcome events | No difference in the outcome |
| Autonomous use | Prospective plus impact plus monitoring | Outcome events and safety events | Any safety signal or subgroup harm |

The table is a reminder that the validation design follows the claim. A claim of impact cannot be supported by a discrimination study, no matter how large.

## 6. Worked example

### The setting

A health system has developed a 30-day readmission model for adults discharged after a heart failure admission. The model uses 12 predictors available at discharge: age, ejection fraction, NYHA class, creatinine, sodium, haemoglobin, length of stay, prior admissions, discharge disposition, diuretic dose, comorbidity count, and follow-up appointment scheduled. The development cohort is 500 admissions with 103 readmissions (20.6 percent).

The vendor wants to deploy the model in three hospitals and has asked for a validation package. The clinical action is a transitional-care referral for patients above a 20 percent predicted risk.

### Step 1: Intended use and claim

> For adults discharged after a heart failure admission, the model predicts the risk of all-cause readmission within 30 days at the point of discharge. The output is a risk score displayed to the discharging clinician. Patients above 20 percent are offered a transitional-care referral. The claim is that the model discriminates with an AUC of at least 0.75 and is calibrated with a slope between 0.8 and 1.2 in the deployment population.

### Step 2: Performance estimand

The primary estimand is the AUC in the deployment population, with a lower 95 percent confidence bound above 0.70. The secondary estimands are the calibration slope (target 0.8 to 1.2), the O:E ratio (target 0.8 to 1.25), and net benefit at the 20 percent threshold.

### Step 3: Model lock

The model is locked at version 1.0. The coefficients, preprocessing, and imputation rules are recorded. The 12 predictors are mapped to the validation data dictionaries. No refitting is permitted during validation.

### Step 4: Validation design

The validation has three stages:

1. **Internal validation** in the development cohort, using 200 bootstrap replicates to estimate optimism.
2. **Temporal validation** in a later cohort from the same hospitals (1,500 admissions, 270 readmissions).
3. **Geographic external validation** in two other hospitals with different case mix and discharge processes.

A prospective silent-mode evaluation is planned after the external validation, and an impact evaluation is planned only if the silent-mode evaluation meets the performance targets.

### Step 5: Sample size

The temporal cohort has 1,500 admissions and 270 events. The external cohorts have 800 and 600 admissions, with about 150 and 110 events. The precision calculation shows that 800 admissions give a calibration slope confidence interval width of about 0.48, and 1,600 admissions reduce it to about 0.34. The external cohort of 800 is adequate for discrimination but marginal for precise calibration. The report will state this limitation.

### Step 6: Outcome and predictor mapping

The outcome is all-cause readmission within 30 days of discharge, ascertained from the health system's admission records and the regional health information exchange. The predictors are mapped variable by variable. Two mappings are imperfect: ejection fraction is recorded as a category in one hospital, and follow-up appointment scheduling is not available in the external data. The validation reports performance with and without those predictors.

### Step 7: Case mix

| Characteristic | Development (n=500) | Temporal (n=1,500) | External A (n=800) | External B (n=600) |
|---|---:|---:|---:|---:|
| Mean age, years | 72.4 | 73.1 | 69.8 | 75.2 |
| Female, percent | 48 | 47 | 44 | 52 |
| Mean ejection fraction, percent | 42 | 41 | 46 | 38 |
| Prior admission, percent | 38 | 40 | 29 | 45 |
| 30-day readmission, percent | 20.6 | 18.0 | 18.8 | 18.3 |

The table shows a case-mix shift: External A is younger with better ejection fraction and fewer prior admissions; External B is older with more prior admissions. The validation must interpret performance in light of these differences.

### Step 8: Discrimination

| Cohort | Events | AUC | 95% CI |
|---|---:|---:|---|
| Development, apparent | 103 | 0.836 | 0.782 to 0.890 |
| Development, optimism-corrected | 103 | 0.806 | Not applicable |
| Temporal | 270 | 0.776 | 0.736 to 0.816 |
| External A | 150 | 0.792 | 0.742 to 0.842 |
| External B | 110 | 0.741 | 0.681 to 0.801 |

The apparent AUC is 0.836. Bootstrap optimism is 0.030, giving a corrected AUC of 0.806. The temporal AUC is 0.776, and the external AUCs are 0.792 and 0.741. All are above the 0.70 lower bound, but External B is close to the target.

### Step 9: Calibration

| Cohort | Calibration-in-the-large | Calibration slope | O:E ratio |
|---|---:|---:|---:|
| Development | 0.00 | 1.00 | 1.00 |
| Temporal | -0.18 | 0.85 | 1.12 |
| External A | -0.10 | 0.92 | 1.07 |
| External B | -0.42 | 0.72 | 1.31 |

The temporal cohort is under-predicted by about 12 percent (O:E 1.12), and External B is under-predicted by 31 percent (O:E 1.31). The calibration slope in External B is 0.72, indicating that the predictions are too extreme: high-risk patients are over-predicted relative to their observed risk, and low-risk patients are under-predicted. The pattern is consistent with a case-mix shift and a different baseline risk.

### Step 10: Clinical utility

At the 20 percent threshold, the net benefit in the temporal cohort is 0.031 for the model, 0.018 for treat-all, and 0 for treat-none. The model adds about 1.3 net true positives per 100 patients compared with treat-all. In External B, the net benefit is 0.012 for the model and 0.015 for treat-all, so the model is not better than treating everyone at that threshold. The result is consistent with the miscalibration: the model under-predicts risk, so it misses patients who would benefit.

### Step 11: Subgroups

The subgroup analysis shows that the model's AUC is lower in patients aged 80 and older (0.69) and in women (0.72) in External B. The calibration slope in women is 0.65, indicating substantial miscalibration. The report flags these findings and recommends recalibration by sex and age band before deployment in External B.

### Step 12: Silent-mode evaluation

A three-month silent-mode evaluation is conducted in External A. The model runs on live data, and predictions are recorded but not shown. The AUC is 0.781, the calibration slope is 0.89, and the O:E ratio is 1.09. The missingness of ejection fraction is 18 percent in live data, higher than in the retrospective validation (8 percent). The performance is acceptable, but the missingness requires a plan.

### Impact evaluation of the model

The health system decides not to proceed to an impact evaluation in External B until recalibration is complete. In External A, a stepped-wedge impact evaluation is planned, with the transitional-care referral as the action and 30-day readmission as the outcome. The evaluation is powered for a 3 percentage point absolute reduction.

### Step 14: Monitoring

The monitoring plan specifies monthly AUC, calibration slope, and O:E ratio, with triggers at slope below 0.8 or above 1.2 and O:E below 0.8 or above 1.25. A trigger leads to a review by the model governance committee and, if confirmed, recalibration.

### Performance summary

| Cohort | Events | AUC | Slope | O:E | Net benefit at 20% | Decision |
|---|---:|---:|---:|---:|---:|---|
| Development, apparent | 103 | 0.836 | 1.00 | 1.00 | 0.041 | Not evidence |
| Development, corrected | 103 | 0.806 | Not estimated | Not estimated | Not estimated | Optimism removed |
| Temporal | 270 | 0.776 | 0.85 | 1.12 | 0.031 | Deploy with monitoring |
| External A | 150 | 0.792 | 0.92 | 1.07 | 0.029 | Deploy with monitoring |
| External B | 110 | 0.741 | 0.72 | 1.31 | 0.012 | Do not deploy |

The table is the validation package in one view. It shows that the model passes discrimination in all cohorts but fails calibration and net benefit in External B. A committee that looked only at the AUC would see 0.741 and conclude that the model is acceptable. A committee that looks at the whole table sees that the model is not decision-ready in that population.

### Why External B failed

External B differs from development in four ways:

- The mean age is 75.2 years versus 72.4 years.
- Prior admission is present in 45 percent versus 38 percent.
- Mean ejection fraction is 38 percent versus 42 percent.
- The discharge process does not include a scheduled follow-up appointment for all patients.

The first three differences shift the case mix toward higher risk. The fourth removes a predictor that is in the model. The combination explains the under-prediction (O:E 1.31) and the slope below 1 (0.72). The model is not wrong about the ranking of patients; it is wrong about the absolute risk, and the missing predictor weakens the ranking.

The lesson is that transportability has two components: the predictor-outcome relationship and the measurement of the predictors. External B changed both. Recalibration can address the first; it cannot recover a predictor that is not measured.

A practical response is to report the validation in two parts: the full-model validation and a reduced-model validation without the unavailable predictor. If the reduced model performs acceptably, it can be deployed in External B while the full model remains the standard elsewhere. If the reduced model fails, the model is not transportable to that site without redevelopment.

### Optimism by hand

The bootstrap optimism is the average difference between performance on the bootstrap sample and performance on the original sample. In the example, the apparent AUC is 0.836 and the average bootstrap difference is 0.030, so the corrected AUC is 0.806.

The arithmetic is simple:

```text
optimism = mean(AUC_bootstrap_apparent - AUC_bootstrap_test)
corrected = AUC_apparent - optimism
          = 0.836 - 0.030
          = 0.806
```

The number of bootstrap replicates matters. With 30 replicates, the Monte Carlo error of the optimism estimate is large enough to change the second decimal. A confirmatory validation should use at least 200 replicates, and the report should state the number and the seed.

Optimism is not the same as overfitting. A model can be overfitted and still have low optimism if the development sample is large. Optimism measures the inflation in the performance estimate, not the quality of the model. A model with low optimism can still fail externally because of transport.

### Two diagnostics that reveal a flawed design

**Diagnostic 1: the calibration slope versus the AUC.** The development report shows an AUC of 0.836 and a calibration slope of 1.00 by construction. If the validation report shows a high AUC but a slope of 0.72, the model discriminates but is miscalibrated. A team that reports only the AUC would miss the problem. The diagnostic is to plot the calibration curve and compute the slope in every validation cohort, not only in development.

**Diagnostic 2: the case-mix table.** External B has a higher mean age, more prior admissions, and a lower ejection fraction than development. The case-mix table explains why the model under-predicts. A validation report without the case-mix table would leave the reader unable to distinguish model drift from population difference. The diagnostic is to compare predictor distributions and baseline risk before interpreting performance.

### Vary one assumption: the event rate

If the external cohort has a 10 percent readmission rate instead of 18 percent, the number of events falls from 110 to 60 in a cohort of 600. The calibration slope confidence interval widens substantially, and the O:E ratio becomes imprecise. The validation may be unable to distinguish acceptable from unacceptable calibration. The design response is to increase the sample size or to pool cohorts with a prespecified adjustment for site.

### Vary one assumption: the sample size

If the external cohort has 300 admissions rather than 800, the calibration slope confidence interval width increases from about 0.48 to about 0.78. The validation can detect a gross miscalibration but not a moderate one. The report should state that the precision is insufficient for the calibration claim, even if the point estimate looks acceptable.

### Vary one assumption: recalibration

If the model is recalibrated in External B by updating the intercept and slope, the O:E ratio moves from 1.31 to 1.00 and the calibration slope from 0.72 to 1.00 in the recalibration sample. The AUC is unchanged, because recalibration is monotonic. The net benefit at the 20 percent threshold improves from 0.012 to 0.026. Recalibration is a legitimate response to case-mix shift, but it must be validated in a separate sample to avoid reporting in-sample performance.

### The decision

The governance committee approves deployment in External A with silent-mode monitoring and a planned impact evaluation. It does not approve deployment in External B until the model is recalibrated and validated in a new sample. The decision is based on the calibration and net benefit, not on the AUC alone.

The example shows the central lesson of validation: discrimination is necessary but not sufficient. A model with an AUC of 0.74 can be clinically useful if it is well calibrated, and a model with an AUC of 0.84 can be harmful if it is miscalibrated at the decision threshold.

## 7. Code, query, or calculation

The code below implements a small logistic model, bootstrap optimism, temporal validation, calibration analysis, and a simulation-based precision check. It uses only the Python standard library. The data are simulated for teaching.

```python
import math
import random


def inv_logit(x):
    return 1 / (1 + math.exp(-x))


def fit_logit(X, y, epochs=120, lr=0.3):
    """Gradient-descent logistic regression; returns intercept and coefficients."""
    n, p = len(y), len(X[0])
    beta = [0.0] * (p + 1)
    for _ in range(epochs):
        grad = [0.0] * (p + 1)
        for xi, yi in zip(X, y):
            z = beta[0] + sum(beta[j + 1] * xi[j] for j in range(p))
            err = inv_logit(z) - yi
            grad[0] += err
            for j in range(p):
                grad[j + 1] += err * xi[j]
        for k in range(p + 1):
            beta[k] -= lr * grad[k] / n
    return beta


def predict(beta, x):
    return beta[0] + sum(beta[j + 1] * x[j] for j in range(len(x)))


def auc(y, score):
    """Rank-based AUC with ties counted as half."""
    order = sorted(range(len(score)), key=lambda i: score[i])
    ranks = [0.0] * len(score)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and score[order[j + 1]] == score[order[i]]:
            j += 1
        r = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = r
        i = j + 1
    n1 = sum(y)
    n0 = len(y) - n1
    s = sum(ranks[i] for i in range(len(y)) if y[i] == 1)
    return (s - n1 * (n1 + 1) / 2) / (n1 * n0)


def calibration(y, lp):
    """Fit logit(y) = a + b * lp; return intercept, slope, SE(slope)."""
    a = b = 0.0
    for _ in range(50):
        g0 = g1 = h00 = h01 = h11 = 0.0
        for yi, li in zip(y, lp):
            p = inv_logit(a + b * li)
            w = p * (1 - p)
            e = yi - p
            g0 += e
            g1 += e * li
            h00 += w
            h01 += w * li
            h11 += w * li * li
        det = h00 * h11 - h01 * h01
        a += (h11 * g0 - h01 * g1) / det
        b += (-h01 * g0 + h00 * g1) / det
    h00 = h01 = h11 = 0.0
    for yi, li in zip(y, lp):
        p = inv_logit(a + b * li)
        w = p * (1 - p)
        h00 += w
        h01 += w * li
        h11 += w * li * li
    det = h00 * h11 - h01 * h01
    return a, b, math.sqrt(h11 / det)


def generate(n, seed, drift=False):
    random.seed(seed)
    X, y = [], []
    for _ in range(n):
        xs = [random.gauss(0, 1) for _ in range(12)]
        if drift:
            xs[0] = random.gauss(0.25, 1)
        z = (-2.1 if drift else -2.0)
        z += (0.85 if drift else 1.0) * xs[0] + 0.7 * xs[1] + 0.5 * xs[2]
        z += 0.3 * xs[3] + 0.2 * xs[4]
        X.append(xs)
        y.append(1 if random.random() < inv_logit(z) else 0)
    return X, y


if __name__ == "__main__":
    X, y = generate(500, seed=1)
    beta = fit_logit(X, y)
    score = [predict(beta, x) for x in X]
    apparent = auc(y, score)
    a, b, se = calibration(y, score)
    print(f"development: events={sum(y)} apparent_auc={apparent:.3f} "
          f"slope={b:.3f}")

    random.seed(9)
    optimism = []
    for _ in range(30):
        idx = [random.randrange(len(y)) for _ in range(len(y))]
        Xb = [X[i] for i in idx]
        yb = [y[i] for i in idx]
        bb = fit_logit(Xb, yb, epochs=100)
        sb = [predict(bb, x) for x in Xb]
        st = [predict(bb, x) for x in X]
        optimism.append(auc(yb, sb) - auc(y, st))
    mean_opt = sum(optimism) / len(optimism)
    print(f"optimism={mean_opt:.3f} corrected_auc={apparent - mean_opt:.3f}")

    Xt, yt = generate(1500, seed=2, drift=True)
    st = [predict(beta, x) for x in Xt]
    at, bt, set_ = calibration(yt, st)
    print(f"temporal: events={sum(yt)} auc={auc(yt, st):.3f} "
          f"slope={bt:.3f} intercept={at:.3f}")
    print(f"mean_predicted={sum(inv_logit(s) for s in st)/len(st):.3f} "
          f"observed={sum(yt)/len(yt):.3f}")

    for n in (400, 800, 1600):
        widths = []
        for r in range(25):
            Xv, yv = generate(n, seed=100 + r, drift=True)
            sv = [predict(beta, x) for x in Xv]
            _, _, sev = calibration(yv, sv)
            widths.append(3.92 * sev)
        print(f"n={n} events={round(n*0.17)} "
              f"slope_ci_width={sum(widths)/len(widths):.3f}")
```

Expected output:

```text
development: events=103 apparent_auc=0.836 slope=1.170
optimism=0.030 corrected_auc=0.806
temporal: events=270 auc=0.776 slope=0.848 intercept=-0.492
mean_predicted=0.225 observed=0.179
n=400 events=68 slope_ci_width=0.681
n=800 events=136 slope_ci_width=0.481
n=1600 events=272 slope_ci_width=0.343
```

The output shows four things.

- The apparent AUC is 0.836, and the optimism-corrected AUC is 0.806. The 0.030 gap is the optimism. It is modest because the development sample has about 100 events and 12 predictors, but it is not zero.
- The temporal AUC is 0.776, lower than the corrected development AUC. The difference is transport, not optimism.
- The calibration slope in the temporal cohort is 0.848, and the mean predicted risk is 0.225 against an observed 0.179. The model under-predicts risk by about 4.6 percentage points.
- The precision check shows that 400 admissions give a calibration slope confidence interval width of 0.68, which is too wide for a calibration claim. 1,600 admissions reduce it to 0.34.

The model in the code is deliberately small. A production validation would use a prespecified model, a proper propensity or regression pipeline, and a clustered variance estimator. The teaching point is the relationship between sample size, events, and precision.

### Decision curve calculation

The net benefit at a threshold can be computed directly from the confusion matrix at that threshold.

```python
def net_benefit(y, risk, threshold):
    """Net benefit at a probability threshold."""
    n = len(y)
    tp = sum(1 for yi, ri in zip(y, risk) if ri >= threshold and yi == 1)
    fp = sum(1 for yi, ri in zip(y, risk) if ri >= threshold and yi == 0)
    return tp / n - (fp / n) * (threshold / (1 - threshold))


if __name__ == "__main__":
    # illustrative: 1,000 patients, 200 events, 200 TP and 0 FP at 20%
    y = [1] * 200 + [0] * 800
    risk = [0.25] * 200 + [0.10] * 800
    nb = net_benefit(y, risk, 0.20)
    treat_all = 200 / 1000 - (800 / 1000) * (0.20 / 0.80)
    print(f"model net benefit={nb:.3f} treat_all={treat_all:.3f}")
```

Expected output:

```text
model net benefit=0.200 treat_all=0.000
```

The calculation shows the model's advantage at the 20 percent threshold: with all 200 events above the threshold and no false positives, the net benefit is 0.200. It also shows why net benefit is threshold-dependent: at a lower threshold, treat-all may have a higher net benefit than the model, and at a higher threshold, the model may be the only strategy with positive net benefit.

### A calculation for the O:E ratio

The O:E ratio is the observed number of events divided by the expected number:

```text
O:E = observed events / sum(predicted risk)
```

For the temporal cohort, if the observed events are 270 and the sum of predicted risks is 241, the O:E ratio is 270 / 241 = 1.12. An O:E above 1 means the model under-predicts; below 1 means it over-predicts. The ratio should be reported with a confidence interval, typically from a Poisson or bootstrap method.

## 8. Data, standards, and regulatory context

### Binding requirements

**EU Medical Device Regulation.** Regulation (EU) 2017/745 requires clinical evaluation for medical devices, including software with a medical purpose. The clinical evaluation must be based on clinical data, and the manufacturer must demonstrate conformity with the general safety and performance requirements. For high-risk devices, a clinical investigation may be required. The regulation is binding.

**EU Artificial Intelligence Act.** Regulation (EU) 2024/1689 establishes a risk-based framework for AI systems. AI systems used in medical devices are generally high-risk, and the Act imposes obligations on providers, including risk management, data governance, technical documentation, and post-market monitoring. The obligations apply in phases; the high-risk provisions apply from August 2026 for some systems and August 2027 for others. The Act is binding, and it interacts with the MDR rather than replacing it.

**US device regulation.** The FDA's premarket approval and De Novo pathways require valid scientific evidence. For AI-enabled devices, the FDA has issued guidance on predetermined change control plans and on lifecycle management. These are guidance under the device statute, not standalone regulations, but they define the review expectations.

### Formal guidance

**TRIPOD+AI.** The TRIPOD+AI statement (BMJ 2024;385:e078378) replaces TRIPOD 2015. It provides a 27-item checklist with 52 subitems for reporting prediction models developed with regression or machine learning. It covers open science, fairness, subgroup performance, and calibration. It is a reporting guideline, not a design requirement, but a validation report that omits its items is difficult to appraise.

**PROBAST+AI.** The PROBAST+AI tool (BMJ 2025;388:e082505) replaces PROBAST 2019. It assesses the quality of model development and the risk of bias in model evaluation, with 16 signalling questions for development and 18 for evaluation across four domains: participants and data sources, predictors, outcome, and analysis. It adds explicit attention to fairness and algorithmic bias. It is an appraisal tool, not a reporting guideline.

**FDA predetermined change control plan.** The final guidance "Marketing Submission Recommendations for a Predetermined Change Control Plan for Artificial Intelligence-Enabled Device Software Functions" was issued in December 2024, with a later listing in August 2025. It describes how a manufacturer can prespecify modifications to an AI-enabled device and the methodology for developing, validating, and implementing them. A PCCP allows certain updates without a new marketing submission. It is guidance, but it is the operative route for model updates.

**FDA AI-enabled device lifecycle guidance.** The final guidance "Artificial Intelligence-Enabled Device Software Functions: Lifecycle Management and Marketing Submission Recommendations" was issued in January 2025. It covers the total product lifecycle, including data management, model development, validation, and post-market monitoring. It is guidance.

**Good machine learning practice.** The FDA, Health Canada, and the UK Medicines and Healthcare products Regulatory Agency jointly published guiding principles for good machine learning practice in 2021. The principles are not binding, but they describe the expected practices for data quality, model development, and monitoring.

### What the standards constrain and what they leave open

| Instrument | Constrains | Leaves open |
|---|---|---|
| MDR | Clinical evaluation and evidence | The validation design and metrics |
| AI Act | Risk management, data governance, documentation | The specific performance thresholds |
| TRIPOD+AI | Reporting content | The study design |
| PROBAST+AI | Appraisal domains | The judgement of overall risk |
| FDA PCCP guidance | The content of a change plan | The model and the validation strategy |
| FDA lifecycle guidance | Lifecycle activities | The specific implementation |

The pattern is familiar: the standards define the documentation and the process, not the scientific judgement. The judgement must be justified in the protocol and the validation report.

### TRIPOD+AI domains and validation evidence

| Domain | Items relevant to validation | Evidence a reviewer expects |
|---|---|---|
| Title and abstract | Model type, intended use, performance | Structured abstract with AUC and calibration |
| Introduction | Rationale and prior evidence | Reference to development study |
| Methods: source of data | Setting, dates, eligibility | Cohort flow and dates |
| Methods: participants | Inclusion and exclusion | Validation population description |
| Methods: outcome | Definition and ascertainment | Code list or adjudication |
| Methods: predictors | Definitions and timing | Variable mapping |
| Methods: sample size | Events and precision | Calculation |
| Methods: missing data | Handling method | Imputation or complete-case plan |
| Methods: analysis | Discrimination, calibration, net benefit | Prespecified metrics |
| Results: participants | Flow and baseline | Table and flow diagram |
| Results: performance | AUC, slope, O:E, net benefit | Estimates with CIs |
| Results: subgroups | Prespecified groups | Forest plot or table |
| Discussion | Limitations and interpretation | Explicit claim boundaries |
| Open science | Data and code availability | Repository or statement |

The table is not a substitute for the full checklist. It shows where validation evidence sits within the reporting structure, and it helps a team prepare the package before submission.

### Documentation checklist

- Locked model specification with version and date.
- Predictor definitions and source variables.
- Outcome definition and ascertainment method.
- Cohort flow diagram with exclusion counts.
- Case-mix table comparing development and validation.
- Analysis code and random seeds.
- Calibration plot and slope.
- Decision curve with the threshold range.
- Subgroup results with confidence intervals.
- Monitoring plan and recalibration triggers.

### Data standards

Validation datasets are often assembled from EHR, registry, or claims sources. The Observational Medical Outcomes Partnership (OMOP) Common Data Model is widely used to standardise observational data, and the Fast Healthcare Interoperability Resources (FHIR) standard is used for exchanging clinical data. The Clinical Data Interchange Standards Consortium (CDISC) standards are used for regulatory submissions. These standards constrain the representation of data, not the validation design.

A validation report should state the data source, the extraction date, the coding system, the outcome definition, and the missingness. Without these, the validation cannot be reproduced.

## 9. Industry practice and case evidence

### The Epic Sepsis Model

Wong and colleagues externally validated a widely implemented proprietary sepsis prediction model at Michigan Medicine. The cohort included 27,697 patients with 38,455 hospitalisations; sepsis was present in 2,552 hospitalisations (7 percent). The hospitalisation-level AUC was 0.63 (95 percent confidence interval 0.62 to 0.64), compared with the vendor's reported 0.76 to 0.83. At the alert threshold of 6 or more, 18 percent of hospitalisations would trigger an alert, with sensitivity 33 percent and positive predictive value 12 percent.

The documented fact is the performance gap. The inference is that the development and validation populations differed in case mix, coding, and care processes, and that the vendor's performance claim was not reproduced in an independent setting. The study did not test whether the model reduced mortality; it tested whether the model performed as claimed.

**Lesson.** External validation is not a confirmatory formality. It can overturn a performance claim, and it should be conducted in the deployment population with the deployment outcome definition.

### IDx-DR

The FDA authorised IDx-DR through the De Novo pathway in 2018 (DEN180001) for autonomous detection of diabetic retinopathy in adults with diabetes. The authorisation was based on a prospective clinical study with a defined intended use, an image-acquisition protocol, and a reference standard. The claim was narrow: detect more than mild diabetic retinopathy and refer.

The documented fact is the authorisation and the intended-use restriction. The inference is that the validation was designed around a specific capture protocol and a specific decision, which made the performance claim credible. A broader claim would have required a broader validation.

**Lesson.** Validation is always validation for an intended use. A narrow intended use makes validation feasible and credible.

### The Apple Heart Study

The Apple Heart Study enrolled 419,297 participants in a siteless, app-based study. Over a median 117 days, 2,161 participants (0.52 percent) received an irregular pulse notification. Among 450 notified participants who returned analyzable ECG patches, atrial fibrillation was present in 34 percent; the positive predictive value for notification concordance with simultaneous patch-detected atrial fibrillation was 0.84.

The documented facts are the scale and the confirmation rate. The inference is that the study validated a detection pathway, not a clinical outcome. The incomplete patch return and the absence of a randomised comparison limited the claim.

**Lesson.** A validation study must match the claim. A detection claim needs a reference standard; an impact claim needs an outcome and a comparison.

### Model updates in practice

AI-enabled devices can change after authorisation. The FDA's PCCP guidance allows manufacturers to prespecify modifications and the validation methodology. The practice is to define the scope of permitted changes, the data and methods for validating them, and the impact assessment. A model update outside the PCCP requires a new submission.

The documented fact is the existence of the PCCP pathway. The inference is that manufacturers who plan for updates have a commercial advantage, because they can improve the model without a new authorisation cycle. The risk is that the PCCP becomes a route for uncontrolled drift; the guidance requires the change to be prespecified and validated.

**Lesson.** Validation is a lifecycle activity, not a one-time event. The design should include monitoring and a change-control plan.

### What companies actually do

The following practices are inferred from public guidance and published validation studies.

1. **Internal validation is standard; external validation is variable.** Many development papers report cross-validation or bootstrap but not external validation. Regulators and payers increasingly require external evidence.
2. **Calibration is under-reported.** Discrimination is reported more often than calibration, even though calibration is more decision-relevant for risk-based actions.
3. **Subgroup analyses are often post hoc.** Fairness analyses are increasingly expected, but they are not always prespecified.
4. **Silent-mode evaluation is used before live pilots.** Health systems often run a model in the background for weeks or months before showing predictions to clinicians.
5. **Impact evaluations are rare.** Few AI products have randomised evidence of clinical or cost impact. The evidence is usually limited to performance.

### Where practice falls short

- **Apparent performance presented as validation.** A development AUC reported without a holdout or external cohort.
- **Internal validation only.** Cross-validation presented as evidence of generalisability.
- **Calibration omitted.** An AUC reported without a calibration plot or slope.
- **Case-mix ignored.** A validation in a different population interpreted without a case-mix comparison.
- **Threshold omitted.** Sensitivity and specificity reported without the threshold.
- **No monitoring plan.** A model deployed without a plan for drift detection.

### What a regulator or payer will check

The following checks recur in device review and payer dossiers.

- The intended use is stated, and the validation population matches it.
- The model version is locked, and the validation was conducted on that version.
- The outcome definition is the same in development and validation, or the difference is justified.
- The validation is independent of the development data.
- Calibration is reported, not only discrimination.
- The decision threshold is stated and clinically justified.
- Subgroup performance is reported for the populations that will be served.
- The sample size is adequate for the precision claimed.
- The prospective evaluation, if any, is described.
- The monitoring and update plan is described.

A package that answers these checks is not guaranteed approval, but a package that omits them invites a request for information.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Apparent performance as validation | No holdout or external cohort | Model evaluated on development data | Overstated performance | Internal and external validation |
| Data leakage | Test data used in preprocessing | Information from test set enters model | Inflated performance | Pipeline separation; audit |
| Optimism ignored | No bootstrap or cross-validation | Many predictors, small sample | Overfitted model | Optimism correction |
| Calibration omitted | Only AUC reported | Calibration not assessed | Wrong risk thresholds | Calibration plot and slope |
| Spectrum bias | Validation in a tertiary centre only | Case mix differs from target | Performance not transportable | Multi-site validation |
| Case-mix shift ignored | Predictor distributions differ | Population change | Miscalibration | Recalibration; restriction |
| Threshold omitted | Sensitivity without threshold | Decision context missing | Unusable report | Report threshold-specific metrics |
| Subgroup harm | Overall AUC acceptable | Subgroup miscalibration | Inequitable performance | Subgroup analysis; recalibration |
| Missing predictors | Predictor unavailable in validation | Model cannot be applied | Reduced performance | Mapping; imputation; reduced model |
| Cluster ignored | Confidence intervals too narrow | Patients clustered by site | False precision | Clustered variance |
| Recalibration in-sample | Recalibration fitted on validation data | Performance reported on fitting data | Optimistic recalibration | Separate sample or cross-validation |
| No monitoring | No post-deployment plan | Drift over time | Silent degradation | Monitoring and triggers |
| Update without validation | Model changed after authorisation | Uncontrolled modification | Unknown performance | PCCP; change control |
| Impact assumed | Performance study only | Prediction does not equal benefit | Unsupported impact claim | Impact evaluation |

### Red flags in a validation report

- The validation cohort is the development cohort.
- The AUC is reported without a confidence interval.
- Calibration is assessed only with the Hosmer-Lemeshow test.
- The threshold is not stated.
- The validation population is not described.
- The case mix is not compared with development.
- Subgroup performance is not reported.
- The model was refitted during validation.
- The recalibration was fitted on the validation data.
- There is no monitoring plan.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The model is locked | Validation must evaluate a fixed model | Refitting during validation | Version control; audit |
| The outcome definition is consistent | Performance depends on the outcome | Coding changes; different ascertainment | Outcome mapping; validation study |
| The predictors are measured consistently | Model inputs must match | Different units, assays, or definitions | Variable mapping; missingness report |
| The validation population is relevant | Transportability requires similarity | Different case mix or setting | Case-mix table; reweighting |
| The events are sufficient | Precision depends on events | Rare outcome; small sample | Event count; CI width |
| The calibration model is correctly specified | Slope and intercept are interpretable | Non-linearity; sparse data | Calibration plot; splines |
| The threshold is clinically justified | Net benefit depends on it | Arbitrary threshold | Decision analysis; clinical input |
| The subgroups are prespecified | Fairness claims require pre-specification | Post hoc selection | Protocol; multiplicity control |
| The deployment pipeline matches validation | Prospective performance depends on it | Data latency; missing fields | Silent-mode evaluation |
| Performance is stable over time | Monitoring assumes a baseline | Drift; population change | Temporal validation; monitoring |
| The report is reproducible | Reviewers must verify the result | Missing code, unspecified seeds | Code and data availability statement |

### Limits of this topic

This topic covers validation design and the core metrics. It does not cover the full theory of prediction-model development, the details of machine-learning validation, or the regulatory pathway for every device class. It also does not cover the statistical theory of net benefit or the design of impact evaluations in detail; those are specialised topics.

The worked example and code use simulated data. The numbers illustrate the relationships among optimism, transport, calibration, and precision; they are not estimates from a real cohort. The regulatory statements are educational summaries, not legal advice. A submission team should verify the current text and date of every guidance it relies on.

## 12. Exercises

### Exercise 1 — Choose the validation level (application, 30 minutes)

**Deliverable:** a validation plan for three claims: (a) a model for local use in one hospital, (b) a model for a national payer, (c) a model marketed as improving outcomes.

**Acceptance criteria:** each claim matched to a validation level; the design justified; the limitations stated.

**Time:** 30 minutes.

### Exercise 2 — Compute discrimination and calibration (calculation, 40 minutes)

**Deliverable:** for a validation cohort of 1,000 patients with 200 events, a predicted-risk sum of 180, and an AUC of 0.74, compute the O:E ratio, state whether the model over- or under-predicts, and explain what the AUC does not tell you.

**Acceptance criteria:** O:E ratio correct; interpretation correct; the limitation of the AUC stated.

**Time:** 40 minutes.

### Exercise 3 — Decision curve (calculation, 35 minutes)

**Deliverable:** at a 20 percent threshold, with 200 events, 800 non-events, 120 true positives, and 80 false positives, compute net benefit and compare it with treat-all and treat-none.

**Acceptance criteria:** net benefit formula correct; arithmetic correct; the comparison interpreted; the threshold odds explained.

**Time:** 35 minutes.

### Exercise 4 — Bootstrap optimism (design, 40 minutes)

**Deliverable:** a plan for estimating optimism in a development cohort of 800 patients with 120 events and 20 predictors, including the number of replicates and the metrics.

**Acceptance criteria:** the resampling procedure described; the optimism definition correct; the number of replicates justified; the reporting plan stated.

**Time:** 40 minutes.

### Exercise 5 — Sample size for validation (calculation, 30 minutes)

**Deliverable:** a sample-size recommendation for an external validation with a target calibration slope confidence interval width of 0.35, given an expected event rate of 15 percent and the precision curve in Section 7.

**Acceptance criteria:** the required sample size estimated; the event count stated; the uncertainty in the estimate acknowledged.

**Time:** 30 minutes.

### Exercise 6 — Diagnose miscalibration (critique, 35 minutes)

**Deliverable:** a diagnosis of a validation report showing AUC 0.80, calibration slope 0.65, O:E 1.40, and a case mix with older patients and more comorbidity than development.

**Acceptance criteria:** the pattern identified; the likely cause stated; a recalibration or restriction plan proposed; the limitation of the AUC explained.

**Time:** 35 minutes.

### Exercise 7 — Silent-mode evaluation (design, 40 minutes)

**Deliverable:** a three-month silent-mode protocol for a sepsis model, including the data pipeline, the metrics, the monitoring, and the criteria for proceeding to a live pilot.

**Acceptance criteria:** the pipeline described; the metrics and thresholds stated; the missing-data plan included; the decision rule clear.

**Time:** 40 minutes.

### Exercise 8 — Map to TRIPOD+AI (application, 45 minutes)

**Deliverable:** a table mapping a validation report you have read to the TRIPOD+AI domains, identifying at least five missing items and their consequences.

**Acceptance criteria:** at least five items mapped; gaps specific; consequences linked to the claim; no generic criticism.

**Time:** 45 minutes.

## 13. Self-check answers

**Answer 1 (validation level).** (a) A model for local use in one hospital needs internal validation plus a local prospective silent-mode evaluation. (b) A model for a national payer needs multi-site external validation with calibration and subgroup analyses. (c) A model marketed as improving outcomes needs prospective validation plus an impact evaluation, ideally randomised. The limitations are that local validation does not generalise, and impact evidence is expensive and slow.

**Answer 2 (discrimination and calibration).** The expected number of events is 180, the observed is 200, so the O:E ratio is 200 / 180 = 1.11. The model under-predicts risk by about 11 percent. The AUC of 0.74 describes discrimination but says nothing about whether the predicted probabilities are accurate. A model can have an AUC of 0.74 and be badly calibrated.

**Answer 3 (decision curve).** Net benefit = (120/1000) - (80/1000) * (0.20/0.80) = 0.12 - 0.08 * 0.25 = 0.12 - 0.02 = 0.10. Treat-all net benefit = (200/1000) - (800/1000) * 0.25 = 0.20 - 0.20 = 0.00. Treat-none = 0. The model has a net benefit of 0.10, which is better than both alternatives at this threshold.

**Answer 4 (bootstrap optimism).** Draw at least 200 bootstrap samples with replacement. For each, fit the model, compute apparent performance on the bootstrap sample, and compute test performance on the original sample. The optimism for that replicate is the difference. Average across replicates and subtract from the apparent performance of the original model. Report the optimism-corrected AUC and calibration slope, with the number of replicates and the random seed.

**Answer 5 (sample size).** The precision curve shows a slope confidence interval width of 0.481 at 800 admissions and 0.343 at 1,600. A target of 0.35 requires about 1,500 admissions, or about 225 events at a 15 percent event rate. The estimate is approximate because the width depends on the distribution of the linear predictor and the calibration of the model in the validation population.

**Answer 6 (miscalibration).** The combination of AUC 0.80, slope 0.65, O:E 1.40, and an older, more comorbid population suggests that the model discriminates but under-predicts risk, and that its predictions are too extreme. The likely cause is a case-mix shift and a different baseline risk. The response is to recalibrate the intercept and slope in a separate sample, or to restrict the model to a population similar to development, and to validate the recalibrated model before deployment.

**Answer 7 (silent-mode rubric).** A strong protocol specifies the data source and latency, the prediction and outcome definitions, the metrics (AUC, slope, O:E, net benefit), the monitoring frequency, the missing-data handling, and the criteria for proceeding to a live pilot. A weak protocol describes the model and says it will be evaluated for three months without metrics or decision rules.

**Answer 8 (TRIPOD+AI rubric).** A strong mapping covers the title and abstract, introduction, methods, open science, results, and discussion, and identifies specific missing items such as calibration, subgroup performance, or data sharing. It links each gap to a consequence for the claim. A weak mapping lists the checklist items without assessing the report.

## 14. Interview questions

**Question 1. What is the difference between internal and external validation?**

A strong answer: internal validation estimates performance on new patients from the same data-generating process, addressing optimism; external validation estimates performance in a different population or setting, addressing transport. A weak answer treats them as synonyms.

**Question 2. Why is calibration important if the AUC is good?**

A strong answer: because the AUC is invariant to the base rate and says nothing about whether predicted probabilities are accurate; a risk-based decision requires calibrated probabilities. A weak answer says calibration is "nice to have."

**Question 3. How do you estimate optimism?**

A strong answer: bootstrap or cross-validation, comparing apparent performance on the resampled data with test performance on the original data, averaged over replicates. A weak answer says "use a holdout set" without the optimism logic.

**Question 4. How many events do you need for external validation?**

A strong answer: at least 100 events and 100 non-events for basic validation, more for precise calibration and subgroups, with a precision calculation for the target metric. A weak answer gives a fixed number without context.

**Question 5. What would you do if the calibration slope is 0.65 in the deployment population?**

A strong answer: investigate case mix and measurement, recalibrate the intercept and slope on a separate sample, validate the recalibrated model, and consider whether the predictor-outcome relationships differ enough to require redevelopment. A weak answer says "recalibrate" without the diagnostic or the validation.

**Question 6. What is net benefit, and why use it?**

A strong answer: net benefit is the clinical utility of a model at a threshold, expressed in net true positives after penalising false positives by the threshold odds; it links performance to decisions. A weak answer confuses it with accuracy.

**Question 7. How would you design an impact evaluation for an AI triage tool?**

A strong answer: a randomised or stepped-wedge comparison of the tool-plus-action strategy versus usual care, with a patient-relevant outcome, a prespecified analysis, and a plan for workflow and human factors. A weak answer proposes a before-after study.

**Question 8. What does TRIPOD+AI require that TRIPOD 2015 did not?**

A strong answer: explicit attention to fairness, subgroup performance, open science, and AI-specific reporting, with 27 items and 52 subitems. A weak answer says it is "for AI."

**Question 9. When is recalibration preferable to redevelopment?**

A strong answer: when the predictor-outcome relationships are stable and only the baseline risk or case mix has changed; redevelopment is needed when the relationships themselves differ. A weak answer says recalibration is always cheaper.

**Question 10. What would make you reject a validation study?**

A strong answer: apparent performance presented as validation, no calibration, no case-mix comparison, refitting during validation, or a population that does not match the claim. A weak answer names a metric it dislikes.

## 15. Key terms

- **Apparent performance:** performance measured on the development data.
- **AUC:** area under the receiver operating characteristic curve.
- **Bootstrap optimism:** the average inflation of performance estimated by resampling.
- **Calibration:** agreement between predicted and observed risk.
- **Calibration-in-the-large:** the intercept of the calibration model.
- **Calibration slope:** the slope of observed outcome on predicted log-odds.
- **Case-mix shift:** a change in the distribution of predictors or baseline risk.
- **c-statistic:** the concordance statistic, equivalent to AUC for binary outcomes.
- **Data leakage:** information from the outcome or test set entering development.
- **Decision curve analysis:** a plot of net benefit across thresholds.
- **Discrimination:** the ability to separate outcome classes.
- **External validation:** validation in a different population or setting.
- **Geographic validation:** validation at different sites in the same era.
- **Good machine learning practice (GMLP):** principles for ML-enabled device development.
- **Impact evaluation:** a study of whether a model changes outcomes or cost.
- **Internal validation:** performance estimated on held-out or resampled development data.
- **Model drift:** degradation of performance over time.
- **Net benefit:** clinical utility at a threshold, in net true positives.
- **Observed-to-expected ratio (O:E):** observed events divided by expected events.
- **Optimism:** inflation of performance from evaluating on development data.
- **Positive predictive value (PPV):** the proportion of positive predictions that are true.
- **Predetermined change control plan (PCCP):** a pre-authorised plan for model updates.
- **Prospective validation:** validation on data collected after model lock.
- **PROBAST+AI:** risk-of-bias and applicability tool for prediction models.
- **Recalibration:** updating the intercept or slope of an existing model.
- **Sensitivity:** the proportion of true positives detected at a threshold.
- **Silent-mode evaluation:** prospective validation without affecting care.
- **Specificity:** the proportion of true negatives correctly identified.
- **Spectrum bias:** performance differences caused by case-mix differences.
- **Temporal validation:** validation on later data from the same sites.
- **TRIPOD+AI:** reporting guideline for prediction models.
- **Verification:** confirmation that software implements the intended algorithm.

## 16. Sources

All URLs accessed 25 October 2026.

### Regulations and guidance

- European Union, Regulation (EU) 2017/745 on medical devices. https://eur-lex.europa.eu/eli/reg/2017/745/oj
- European Union, Regulation (EU) 2024/1689 (Artificial Intelligence Act). https://eur-lex.europa.eu/eli/reg/2024/1689/oj
- FDA, "Marketing Submission Recommendations for a Predetermined Change Control Plan for Artificial Intelligence-Enabled Device Software Functions," final guidance, December 2024. https://www.fda.gov/regulatory-information/search-fda-guidance-documents/marketing-submission-recommendations-predetermined-change-control-plan-artificial-intelligence
- FDA, "Artificial Intelligence-Enabled Device Software Functions: Lifecycle Management and Marketing Submission Recommendations," final guidance, January 2025. https://www.fda.gov/medical-devices/digital-health-center-excellence/artificial-intelligence-enabled-medical-devices
- FDA, Health Canada, MHRA, "Good Machine Learning Practice for Medical Device Development: Guiding Principles," 2021. https://www.fda.gov/medical-devices/software-medical-device-samd/good-machine-learning-practice-medical-device-development-guiding-principles

### Reporting and appraisal

- Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement. BMJ 2024;385:e078378. https://doi.org/10.1136/bmj-2023-078378
- Moons KGM, Damen JAA, Kaul T, et al. PROBAST+AI. BMJ 2025;388:e082505. https://doi.org/10.1136/bmj-2024-082505
- Collins GS, Reitsma JB, Altman DG, Moons KGM. TRIPOD statement. BMJ 2015;350:g7594. https://doi.org/10.1136/bmj.g7594

### Methods

- Riley RD, Debray TPA, Collins GS, et al. Minimum sample size for external validation of a clinical prediction model with a binary outcome. Stat Med 2021;40:4230-4251. https://doi.org/10.1002/sim.9025
- Riley RD, Archer L, Snell KIE, et al. Evaluation of clinical prediction models (part 3): calculating the sample size required for an external validation study. BMJ 2024;384:e074821. https://doi.org/10.1136/bmj-2023-074821
- Snell KIE, Archer L, Ensor J, et al. External validation of clinical prediction models: simulation-based sample size calculations were more reliable than rules-of-thumb. J Clin Epidemiol 2021;135:79-89. https://doi.org/10.1016/j.jclinepi.2021.02.011
- Steyerberg EW. Clinical Prediction Models. Springer, 2nd edition, 2019.

### Case evidence

- Wong A, Otles E, Donnelly JP, et al. External Validation of a Widely Implemented Proprietary Sepsis Prediction Model in Hospitalized Patients. JAMA Intern Med 2021;181:1065-1070. https://doi.org/10.1001/jamainternmed.2021.2626
- Perez MV, Mahaffey KW, Hedlin H, et al. Large-Scale Assessment of a Smartwatch to Identify Atrial Fibrillation. N Engl J Med 2019;381:1909-1917. https://doi.org/10.1056/NEJMoa1901183
- FDA, De Novo classification decision for IDx-DR, DEN180001, 2018. https://www.accessdata.fda.gov/cdrh_docs/reviews/DEN180001.pdf

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic: [`03_real-world-evidence.md`](03_real-world-evidence.md)
- Next topic: [`05_bias-taxonomy.md`](05_bias-taxonomy.md)
- Related topic: [`08_external-validation.md`](08_external-validation.md)
- Module survey file: [`../../05_study_design_evidence.md`](../../05_study_design_evidence.md)
- Labs: [`../../labs/`](../../labs/)
- Calibration and decision curve lab: [`../../labs/06_calibration_decision_curve.ipynb`](../../labs/06_calibration_decision_curve.ipynb)
