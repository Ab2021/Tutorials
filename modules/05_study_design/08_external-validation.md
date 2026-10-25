# External Validation

> Module 05 — Study Design and Evidence Generation
> Topic 08 of 10
> Estimated study time: 4 hours
> Prerequisites: Topics 04 and 05
> Version 2026.3

## 1. Why this topic exists

External validation is the test of whether a model works somewhere it was not built. It is the difference between a research result and a product. A model that performs well in its development data but fails at the first external site is not a generalisable tool; it is a local fit. The external validation is what tells the difference.

The commercial stakes are direct. Hospital procurement committees increasingly require local or independent validation before adoption. Payers ask whether the performance observed in a trial population applies to their members. Regulators ask whether the evidence supports the intended use across the settings where the device will be marketed. A vendor without an external validation package is excluded from many procurement processes, regardless of the development metrics.

The methodological stakes are equally direct. External validation is not a single study; it is a design problem. Which sites? How many? Which populations? Which outcome definitions? How to pool? How to handle heterogeneity? How to recalibrate? How to report? Each choice affects the conclusion, and each has a failure mode.

The US Food and Drug Administration (FDA) has authorised more than 1,600 AI-enabled devices as of September 2026, according to its device centre. The European Union's Medical Device Regulation (MDR) requires clinical evaluation, and the Artificial Intelligence Act adds obligations for high-risk systems. The reporting guideline TRIPOD+AI (2024) and the appraisal tool PROBAST+AI (2025) define the evidence a reviewer expects. These instruments make external validation a documented requirement rather than a methodological preference.

This topic teaches you to design a multi-site external validation, analyse it with appropriate heterogeneity methods, recalibrate where necessary, and report it so that a reviewer can judge transportability.

### What external validation is not

- It is not a random split of the development data. That is internal validation.
- It is not a validation in a later period at the same sites. That is temporal validation.
- It is not a validation in a different country with the same care pathway and the same data system. That is geographic validation, which is useful but not the same as validation in a different population.
- It is not a guarantee of clinical benefit. A model can transport and still fail to improve outcomes.

The distinctions matter because each answers a different question. A temporal validation answers "does performance drift over time?" An external validation answers "does performance transport to a different setting?" A prospective validation answers "does it work in live data?" An impact evaluation answers "does it improve outcomes?" The evidence package needs the level that matches the claim.

### The transportability problem

Transportability is the question of whether a model developed in one population applies in another. It depends on four things:

1. **The predictor-outcome relationship.** If the relationship is stable, the model transports. If it differs by population, the model does not.
2. **The distribution of predictors.** Even with a stable relationship, a different distribution changes the case mix and the calibration.
3. **The measurement of predictors and outcomes.** If the same variable is measured differently, the model's inputs change.
4. **The care pathway.** If the clinical action triggered by the model differs, the relationship between prediction and outcome can change.

A model can be transportable in one dimension and not another. A model with a stable predictor-outcome relationship but a different case mix may discriminate well and be badly calibrated. The design response is recalibration, not necessarily redevelopment.

### Why single-site validation is weak

A single external site is a sample of one. It can demonstrate that the model works somewhere else, but it cannot estimate the between-site variation. A model that works at one site and fails at another may look acceptable in a single-site study and fail in a multi-site deployment.

The design response is to include multiple sites that vary in the dimensions expected to matter: size, case mix, care pathway, data system, and geography. The analysis should estimate the between-site heterogeneity, not only the average performance.

### What external validation can establish

- Whether the model performs outside its development site.
- Whether performance varies across sites, and by how much.
- Whether calibration can be restored by recalibration.
- Whether a subgroup has materially worse performance.
- Whether the claim should be restricted to certain sites or populations.
- Whether the model should be redeveloped.

It cannot establish clinical benefit, cost-effectiveness, or safety in use. Those require an impact evaluation and a safety monitoring plan.

## 2. Learning outcomes

By the end of this topic you can:

1. **Design** a multi-site external validation with a justified site-selection strategy and sample size. (Exercise 1, self-check 1)
2. **Harmonise** predictor and outcome definitions across sites and document the mapping. (Exercise 2, self-check 2)
3. **Compute** site-specific discrimination and calibration, and pool them with a random-effects model. (Exercises 3 and 4, self-check 3)
4. **Quantify** between-site heterogeneity with I-squared and tau-squared, and interpret it for a deployment decision. (Exercise 5, interview 4)
5. **Recalibrate** a model for a site with a case-mix shift, and validate the recalibration in a separate sample. (Exercise 6, self-check 5)
6. **Assess** subgroup and fairness performance across sites. (Exercise 7, interview 6)
7. **Report** an external validation to TRIPOD+AI and defend it to a regulator, payer, or procurement committee. (Exercise 8, interview 8)

### How the outcomes connect

- Outcomes 1 and 2 define the design and the site sample.
- Outcomes 3 and 4 provide the analysis and the heterogeneity assessment.
- Outcomes 5 and 6 provide the recalibration and fairness perspective.
- Outcomes 7 and 8 provide the reporting and defence skills.
- A learner who can perform outcomes 1 to 4 can run a validation.
- A learner who can also perform outcomes 5 to 8 can support a deployment decision.

## 3. The core mental model

External validation is a chain from the target population to the deployment decision. Each link can break.

```text
   TARGET POPULATION AND INTENDED USE
              |
              |  site selection: representativeness
              v
   VALIDATION SITES
              |
              |  harmonisation: predictors and outcomes
              v
   HARMONISED DATA
              |
              |  performance: discrimination and calibration
              v
   SITE-SPECIFIC ESTIMATES
              |
              |  pooling: random effects and heterogeneity
              v
   POOLED ESTIMATE AND I-SQUARED
              |
              |  recalibration: intercept and slope
              v
   RECALIBRATED MODEL
              |
              |  decision rule: deploy, recalibrate, or redevelop
              v
   DEPLOYMENT DECISION
```

The chain is not linear in practice. A site that fails calibration may trigger a recalibration, which requires a new validation sample, which changes the evidence package. The design should anticipate the loop.

### Site selection

The sites should be selected to represent the target population, not to be convenient. A validation in three academic centres does not represent community hospitals. A validation in one country does not represent another.

The dimensions that matter are:

- **Case mix:** age, sex, comorbidity, severity, and prevalence.
- **Care pathway:** referral patterns, discharge processes, and treatment availability.
- **Data system:** coding practices, laboratory assays, and missingness.
- **Setting:** academic versus community, urban versus rural, high- versus low-resource.
- **Geography:** country, region, and regulatory context.

A useful design is a stratified sample of sites across the dimensions that are expected to matter. The stratification should be prespecified, and the analysis should report performance by stratum.

### Performance across sites

A multi-site validation produces a distribution of performance, not a single number. The distribution has three components:

- **Within-site sampling error:** the uncertainty of each site's estimate.
- **Between-site heterogeneity:** the real variation in performance across sites.
- **Covariate effects:** the association between site characteristics and performance.

The analysis should separate the three. A pooled estimate with a confidence interval addresses the first. A random-effects model with tau-squared addresses the second. A meta-regression addresses the third.

### Calibration and case mix

Calibration is more sensitive to case mix than discrimination. A model can rank patients correctly in every site and still predict the wrong absolute risk if the baseline risk differs. The calibration slope and the observed-to-expected ratio are the key diagnostics.

The response to miscalibration depends on the pattern:

- **Intercept shift, slope near 1:** recalibrate the intercept.
- **Slope away from 1:** recalibrate the slope and intercept, or redevelop if the predictor-outcome relationship has changed.
- **Subgroup-specific miscalibration:** recalibrate by subgroup or restrict the claim.

Recalibration must be validated in a separate sample. A recalibration fitted and evaluated on the same data is an in-sample result, not evidence.

### The decision rule

The external validation should end with a decision rule, not only a result. The rule should state:

- the minimum acceptable discrimination,
- the acceptable calibration range,
- the acceptable subgroup performance,
- the action if a site fails (deploy, recalibrate, restrict, or redevelop).

A validation without a decision rule leaves the committee to interpret the numbers, which is where commercial pressure enters.

### Transportability dimensions

| Dimension | What can differ | Effect on performance | Design response |
|---|---|---|---|
| Case mix | Age, severity, comorbidity | Calibration and discrimination | Stratified sites; reweighting |
| Baseline risk | Prevalence of the outcome | Calibration intercept | Recalibration |
| Predictor measurement | Units, assays, coding | Discrimination and calibration | Harmonisation; sensitivity analysis |
| Outcome ascertainment | Codes, adjudication | Apparent performance | Harmonised definition |
| Care pathway | Referral, discharge, treatment | Relationship between prediction and outcome | Site selection; meta-regression |
| Data system | EHR vendor, missingness | Predictor availability | Mapping; imputation |
| Geography | Country, region, regulation | All of the above | Multi-country sites |

The table is a checklist for site selection and for the interpretation of heterogeneity.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| External validation | Validation in a different population or setting | Used for any non-development data | Design |
| Geographic validation | Validation at different sites in the same era | Confused with external validation | Design |
| Temporal validation | Validation on later data from the same sites | Confused with external validation | Design |
| Prospective validation | Validation on data collected after model lock | Confused with a prospective cohort | Design |
| Transportability | Whether a model applies in a different population | Confused with generalisability | Property |
| Case mix | The distribution of predictors and baseline risk | Confused with sample size | Distribution |
| Spectrum effect | Performance varies with case mix | Confused with bias | Phenomenon |
| Site | A hospital, clinic, or data source | Used loosely for a country | Unit |
| Heterogeneity | Real variation in performance across sites | Confused with sampling error | Variance component |
| I-squared | The proportion of variability due to heterogeneity | Confused with the p-value | Percentage |
| Tau-squared | The between-site variance | Confused with the pooled estimate | Variance |
| Random-effects model | A model that estimates the mean and the between-site variance | Confused with a fixed-effect model | Model |
| Fixed-effect model | A model that assumes one true effect | Used when heterogeneity is present | Model |
| Pooled estimate | The combined estimate across sites | Confused with the average | Effect measure |
| Calibration slope | The slope of observed outcome on predicted log-odds | Ignored in favour of AUC | Unitless |
| Calibration-in-the-large | The intercept of the calibration model | Confused with the slope | Intercept |
| Observed-to-expected ratio (O:E) | Observed events divided by expected events | Reported without a CI | Ratio |
| Recalibration | Updating the intercept or slope for a new population | Confused with redevelopment | Procedure |
| Model update | Adding predictors or changing coefficients | Confused with recalibration | Procedure |
| Harmonisation | Aligning predictor and outcome definitions across sites | Confused with data cleaning | Procedure |
| Data mapping | The correspondence between source variables and model variables | Confused with harmonisation | Document |
| Missing predictor | A model variable not available at a site | Ignored in the analysis | Data problem |
| Fairness | Equitable performance across subgroups | Confused with overall performance | Property |
| Subgroup performance | Performance in a defined population subset | Confused with a subgroup claim | Analysis |
| Decision threshold | The probability above which an action is taken | Confused with the AUC | Probability |
| Net benefit | Clinical utility at a threshold | Confused with accuracy | Probability units |
| TRIPOD+AI | Reporting guideline for prediction models | Used as a design checklist | Guideline |
| PROBAST+AI | Risk-of-bias and applicability tool | Confused with TRIPOD+AI | Tool |
| Intended use | The clinical purpose, population, user, and setting | Confused with indication | Regulatory term |
| Predetermined change control plan (PCCP) | A pre-authorised plan for model updates | Confused with a post-market change | Regulatory document |
| Site-level estimate | Performance at a single site | Confused with the pooled estimate | Effect measure |
| Forest plot | A plot of site estimates and confidence intervals | Confused with a funnel plot | Figure |
| Funnel plot | A plot of effect against precision | Confused with a forest plot | Figure |
| Meta-regression | A regression of performance on site characteristics | Confused with subgroup analysis | Method |
| Reweighting | Weighting the validation sample to the target population | Confused with pooling | Method |
| Exchangeability of sites | The assumption that sites are comparable | Assumed without testing | Assumption |
| Case-mix adjustment | Adjusting performance for differences in case mix | Confused with recalibration | Method |
| Clinical utility | Whether the model improves decisions | Confused with discrimination | Property |
| Deployment threshold | The performance below which a site should not deploy | Confused with the decision threshold | Performance level |

Two distinctions deserve emphasis.

**Heterogeneity versus sampling error.** Sampling error is the uncertainty of each site's estimate; it shrinks with sample size. Heterogeneity is the real variation across sites; it does not shrink with sample size, and it determines whether a pooled estimate is meaningful. A pooled estimate with high heterogeneity is an average over different populations, not a generalisable performance.

**Recalibration versus redevelopment.** Recalibration updates the intercept or slope of an existing model. Redevelopment refits the coefficients or adds predictors. Recalibration is appropriate when the predictor-outcome relationship is stable and the case mix or baseline risk has changed. Redevelopment is necessary when the relationship itself differs.

## 5. The mechanics, step by step

The procedure has fourteen steps. Steps 1 to 4 design the validation. Steps 5 to 8 analyse performance. Steps 9 to 12 recalibrate and assess fairness. Steps 13 and 14 decide and report.

### Procedure at a glance

| Step | Input | Output | Failure if skipped |
|---:|---|---|---|
| 1 | Intended use | Target population | Wrong validation population |
| 2 | Target and sites | Site-selection plan | Unrepresentative sample |
| 3 | Event rate and precision | Sample size per site | Imprecise site estimates |
| 4 | Model and data dictionaries | Harmonisation plan | Silent predictor mismatch |
| 5 | Protocol | Prespecified analysis | Post hoc analysis |
| 6 | Site data | Site-specific AUC and calibration | Pooled estimate without detail |
| 7 | Site estimates | Random-effects pooled estimate | False precision |
| 8 | Site estimates | Heterogeneity statistics | Hidden variation |
| 9 | Calibration results | Recalibration plan | Miscalibrated deployment |
| 10 | Recalibration data | Validated recalibration | In-sample result |
| 11 | Subgroups | Fairness report | Hidden subgroup harm |
| 12 | Case mix | Transportability analysis | Unsupported generalisation |
| 13 | All results | Decision rule | Political decision |
| 14 | All evidence | TRIPOD+AI report | Unreviewable package |

### Step 1: Define the target population and intended use

**Input:** the product claim and the clinical pathway.

**Output:** an intended-use statement with population, user, setting, input data, output, decision, and action.

**Rationale:** External validation is validation for a use. The target population determines which sites are relevant and which subgroups must be assessed.

The intended-use statement should specify the decision threshold and the action. "Predicts sepsis risk" is not an intended use. "Predicts 12-hour sepsis deterioration in adults admitted to a medical ward, to trigger a bedside review at a threshold of 5 percent" is.

### Step 2: Select sites

**Input:** the target population and the dimensions expected to matter.

**Output:** a site-selection plan with a rationale and a stratification.

**Rationale:** The sites determine the transportability claim. A convenience sample of academic centres supports a claim about academic centres, not about community hospitals.

The selection should consider:

- size and teaching status,
- case mix and baseline risk,
- care pathway and discharge process,
- data system and coding,
- geography and regulatory context,
- willingness to participate and data access.

A stratified sample with at least two sites per stratum is stronger than a larger convenience sample. The stratification should be prespecified and reported.

A site-selection table should record, for each candidate site:

- the site type and size;
- the case mix and baseline risk;
- the care pathway and discharge process;
- the data system and coding;
- the availability of the predictors and the outcome;
- the event count in a recent period;
- the willingness to share data and the governance route;
- the reason for inclusion or exclusion.

The table is the defence against the criticism that the sites were chosen for convenience. It does not eliminate the criticism, but it makes the choice transparent.

### Step 3: Size the validation

**Input:** the expected event rate, the target precision, and the number of sites.

**Output:** a sample size per site and a total sample size.

**Rationale:** Precision depends on events, not records. A site with 500 patients and 50 events cannot estimate calibration precisely, even if the total sample is large.

The common rules are:

- at least 100 events and 100 non-events per site for basic validation,
- at least 200 events and 200 non-events for flexible calibration,
- more events for subgroup analyses,
- a simulation-based calculation for a target confidence interval width.

A multi-site study should report the precision per site and for the pooled estimate. A pooled estimate can be precise while individual site estimates are imprecise, which matters for the deployment decision.

The precision calculation should state:

- the expected event rate per site;
- the number of events per site;
- the confidence interval width for the AUC and the calibration slope;
- the number of sites needed for a stable heterogeneity estimate;
- the inflation for missing predictors or outcomes.

A design with five sites and 100 events each is stronger than a design with two sites and 500 events each, because the first can estimate heterogeneity and the second cannot.

### Step 4: Harmonise predictors and outcomes

**Input:** the locked model and the site data dictionaries.

**Output:** a variable-by-variable mapping with definitions, units, and transformations.

**Rationale:** The same variable name can mean different things at different sites. A model that uses creatinine in mg/dL will fail at a site that reports micromoles per litre. A model that uses a diagnosis code will fail at a site with different coding practices.

The mapping should state:

- the source variable and the model variable,
- the unit and the conversion,
- the timing of measurement,
- the missingness rate,
- the handling of values outside the expected range.

Any variable that cannot be mapped exactly should be flagged, and the validation should report performance with and without it.

### Step 5: Prespecify the analysis

**Input:** the protocol and the site-selection plan.

**Output:** a statistical analysis plan with the primary performance metrics, the pooling method, the heterogeneity statistics, and the recalibration plan.

**Rationale:** External validation has many degrees of freedom: site inclusion, outcome definitions, subgroup analyses, and recalibration methods. Without prespecification, the result is vulnerable to selective reporting.

The plan should state:

- the primary metric (AUC, calibration slope, O:E, net benefit),
- the pooling method (fixed or random effects),
- the heterogeneity statistic,
- the subgroup analyses,
- the recalibration method,
- the decision rule.

### Step 6: Compute site-specific performance

**Input:** the harmonised data and the locked model.

**Output:** site-specific AUC, calibration slope, calibration-in-the-large, O:E ratio, and net benefit.

**Rationale:** The site-specific estimates are the raw evidence. A pooled estimate without the site-specific estimates hides the variation that matters for deployment.

The analysis should report each site's estimate with a confidence interval. A forest plot is the standard presentation.

For each site, report:

- the number of patients and events;
- the AUC with a confidence interval;
- the calibration slope and intercept;
- the O:E ratio with a confidence interval;
- the net benefit at the decision threshold;
- the missingness for each predictor;
- the proportion of patients outside the model's intended-use population.

The site-level table is the evidence for the deployment decision. A pooled estimate without the table is not enough.

### Step 7: Pool across sites

**Input:** the site-specific estimates and their standard errors.

**Output:** a pooled estimate with a confidence interval, using a fixed-effect or random-effects model.

**Rationale:** The pooling method determines what the pooled estimate means. A fixed-effect model estimates a common effect; a random-effects model estimates the mean of a distribution of effects. When heterogeneity is present, the random-effects model is more appropriate.

The choice should be prespecified. A common approach is to report both and to base the decision on the random-effects estimate when I-squared is above a threshold such as 50 percent.

### Step 8: Quantify heterogeneity

**Input:** the site-specific estimates.

**Output:** Cochran's Q, I-squared, tau-squared, and a prediction interval.

**Rationale:** Heterogeneity determines whether a pooled estimate is meaningful. A pooled AUC of 0.76 with I-squared of 80 percent means that the performance varies substantially across sites, and a new site could fall outside the pooled confidence interval.

The prediction interval is often more useful than the confidence interval for a deployment decision, because it describes the range of performance expected at a new site.

### Meta-analysis methods for validation

| Method | What it estimates | When to use | Limitation |
|---|---|---|---|
| Fixed-effect | A common performance across sites | Sites are exchangeable; heterogeneity low | Understates uncertainty when heterogeneity is present |
| DerSimonian-Laird | Mean performance and between-site variance | Heterogeneity expected; few sites | Variance estimator can be biased with few sites |
| REML | Mean performance and between-site variance | Heterogeneity expected; more sites | Requires iterative estimation |
| Bayesian hierarchical | Posterior distribution of performance | Few sites; prior information available | Prior sensitivity |
| Meta-regression | Association between site characteristics and performance | Explaining heterogeneity | Few sites limit power |

The choice should be prespecified. A common approach is to report the fixed-effect and random-effects estimates and to base the decision on the random-effects estimate when heterogeneity is material.

### Step 9: Recalibrate where necessary

**Input:** the calibration results and the case mix.

**Output:** a recalibration plan for sites with miscalibration.

**Rationale:** Recalibration can restore calibration when the predictor-outcome relationship is stable but the baseline risk or case mix has changed. It cannot restore discrimination.

The methods are:

- intercept-only recalibration, when the slope is near 1,
- slope and intercept recalibration, when the slope is away from 1,
- model revision, when a predictor is missing or mismeasured,
- redevelopment, when the relationship has changed.

### Step 10: Validate the recalibration

**Input:** the recalibration model and a separate sample.

**Output:** performance of the recalibrated model in the separate sample.

**Rationale:** A recalibration fitted and evaluated on the same data is an in-sample result. It will look better than it is.

The separate sample can be a holdout within the site, a later period, or another site with similar characteristics. The validation should report the same metrics as the original validation.

### Step 11: Assess subgroups and fairness

**Input:** the site data and the prespecified subgroups.

**Output:** subgroup performance with confidence intervals and a fairness statement.

**Rationale:** A model that performs well overall can perform poorly in a subgroup. The subgroup analysis is a check that the model's performance is adequate for the populations it will serve.

The subgroups should be prespecified and justified by clinical or equity concerns: age, sex, race and ethnicity, socioeconomic status, site, and relevant comorbidity. The analysis should report discrimination, calibration, and net benefit by subgroup.

### Step 12: Analyse transportability

**Input:** the site characteristics and the performance estimates.

**Output:** a meta-regression or a reweighting analysis that relates performance to case mix.

**Rationale:** The transportability analysis explains why performance varies. It can identify the site characteristics that predict failure and inform the site-selection strategy for future deployments.

The methods are:

- meta-regression of performance on site characteristics,
- reweighting the validation sample to the target population,
- sensitivity analysis excluding sites that are not representative.

### Step 13: Apply the decision rule

**Input:** the pooled estimate, the heterogeneity, the recalibration results, and the subgroup performance.

**Output:** a deployment decision: deploy, deploy with recalibration, restrict, or redevelop.

**Rationale:** The decision rule converts the evidence into an action. Without it, the decision is vulnerable to commercial pressure.

A decision rule might state:

- deploy if the pooled AUC is at least 0.75 and the calibration slope is between 0.8 and 1.2;
- deploy with recalibration if the AUC is adequate but the slope is outside the range;
- restrict if a subgroup has materially worse performance;
- redevelop if the AUC is below 0.70 or the calibration cannot be restored.

### Step 14: Report

**Input:** all of the above.

**Output:** a TRIPOD+AI-compliant report with the site selection, the harmonisation, the performance, the heterogeneity, the recalibration, and the decision.

**Rationale:** The report must be usable by a reviewer who did not design the study. It should include the flow diagram, the site characteristics, the forest plot, the calibration plots, and the decision rule.

The report should also state the limitations: the sites selected, the populations not represented, the missing predictors, and the residual heterogeneity.

### Edge cases

**Rare outcomes.** A rare outcome produces few events per site, and calibration becomes imprecise. The validation should pool sites or use a hierarchical model, and it should report the event count per site.

**Missing predictors.** A predictor that is missing at a site cannot be used. The options are to omit the predictor and refit a reduced model, to impute it, or to restrict the validation to patients with complete data. Each option changes the model or the population, and the choice must be reported.

**Different outcome definitions.** A site may ascertain the outcome differently. The validation should report the definition used at each site and a sensitivity analysis with a harmonised definition.

**Clustered data.** If the validation data are clustered by site, the confidence intervals must account for clustering. A model validated across 10 sites with 100 patients each has less independent information than 1,000 patients from one site.

**Case-mix shift without slope change.** A site with a different baseline risk but a stable predictor-outcome relationship will have an intercept shift and a slope near 1. Intercept-only recalibration is appropriate.

**Case-mix shift with slope change.** A site with a different predictor-outcome relationship will have a slope away from 1. Slope and intercept recalibration may be appropriate, but redevelopment may be necessary.

**Regulatory updates.** A model that is updated after authorisation may require a new submission unless the update is within a predetermined change control plan. The validation package should state which version is being validated and which changes are covered.

### Report template

| Section | Content | Evidence |
|---|---|---|
| Title and abstract | Model, intended use, performance | Structured abstract |
| Introduction | Rationale and prior evidence | Development study |
| Methods: sites | Selection and characteristics | Site table |
| Methods: participants | Eligibility and flow | Flow diagram |
| Methods: predictors | Definitions and harmonisation | Mapping table |
| Methods: outcome | Definition and ascertainment | Code list |
| Methods: sample size | Events and precision | Calculation |
| Methods: analysis | Metrics and pooling | SAP |
| Results: participants | Site-level counts and baseline | Table |
| Results: performance | AUC, slope, O:E, net benefit | Forest plot |
| Results: heterogeneity | Q, I-squared, tau-squared | Table |
| Results: recalibration | Method and validation | Calibration plots |
| Results: subgroups | Prespecified groups | Table |
| Discussion | Limitations and decision | Decision rule |
| Open science | Data and code | Repository |

The template is a guide, not a substitute for TRIPOD+AI. It shows the evidence a reviewer will look for in each section.

### Decision table: site performance pattern to action

| Pattern | Interpretation | Action |
|---|---|---|
| AUC adequate, slope 0.9 to 1.1, O:E 0.9 to 1.1 | Transportable | Deploy |
| AUC adequate, slope near 1, O:E outside range | Baseline risk shift | Intercept recalibration |
| AUC adequate, slope outside 0.8 to 1.2 | Predictor-outcome shift | Slope recalibration or redevelopment |
| AUC below target, calibration adequate | Discrimination failure | Redevelop |
| AUC adequate overall, poor in a subgroup | Fairness failure | Restrict or recalibrate by subgroup |
| High heterogeneity, no site fails | Variable but acceptable | Deploy with monitoring |
| High heterogeneity, one site fails | Local failure | Site-specific recalibration or exclusion |

## 6. Worked example

### The setting

A vendor has developed a model that predicts 12-hour sepsis deterioration in adults admitted to a medical ward. The model was developed at two academic hospitals and validated internally with an optimism-corrected AUC of 0.81. The vendor wants to deploy it in five hospitals across a region.

The five hospitals differ in size, case mix, and data system:

| Site | Type | Beds | Annual admissions | Sepsis prevalence | Data system |
|---|---|---:|---:|---:|---|
| A | Academic | 900 | 42,000 | 17.4% | Epic |
| B | Community | 400 | 18,000 | 12.4% | Cerner |
| C | Academic | 1,100 | 51,000 | 20.2% | Epic |
| D | Community | 250 | 9,000 | 9.2% | Meditech |
| E | Community | 600 | 24,000 | 23.8% | Cerner |

### Step 1: Intended use

> For adults admitted to a medical ward with suspected infection, the model predicts the risk of sepsis deterioration within 12 hours. The output is a risk score displayed to the ward nurse. Patients above 5 percent are reviewed by the attending clinician within 60 minutes. The claim is that the model discriminates with an AUC of at least 0.75 and is calibrated with a slope between 0.8 and 1.2 across the deployment sites.

### Step 2: Site selection

The five sites were selected to represent the region's mix of academic and community hospitals, two data systems, and a range of sepsis prevalence. The selection was stratified by type and size. A sixth site was excluded because it could not provide the predictor data.

The sites are not a random sample of all hospitals. The report states that the validation supports a claim about hospitals with similar characteristics, not about all hospitals.

### Step 3: Sample size

Each site contributed between 600 and 1,200 admissions. The event counts were 55 to 242. Site D had only 55 events, which is below the 100-event rule. The report flags this limitation and uses a hierarchical model to borrow information across sites.

### Step 4: Harmonisation

The model uses 10 predictors: age, temperature, heart rate, respiratory rate, systolic blood pressure, white cell count, creatinine, lactate, platelet count, and a comorbidity count. Eight predictors mapped exactly. Creatinine required a unit conversion at one site. Lactate was missing in 22 percent of admissions at Site D. The comorbidity count was derived differently at Site B.

The report includes a variable-mapping table and a sensitivity analysis excluding lactate.

### Step 5: Analysis plan

The primary metric is the AUC. The secondary metrics are the calibration slope, the O:E ratio, and net benefit at the 5 percent threshold. The pooling method is a random-effects model with the DerSimonian-Laird estimator. Heterogeneity is reported with I-squared and tau-squared. The decision rule is the one in Step 13.

### Step 6: Site-specific performance

| Site | Events | AUC | 95% CI | Calibration slope | O:E ratio |
|---|---:|---:|---|---:|---:|
| A | 174 | 0.751 | 0.706 to 0.796 | 0.95 | 1.11 |
| B | 99 | 0.766 | 0.713 to 0.819 | 1.02 | 0.81 |
| C | 242 | 0.781 | 0.744 to 0.818 | 1.19 | 1.27 |
| D | 55 | 0.677 | 0.594 to 0.760 | 0.64 | 0.57 |
| E | 214 | 0.751 | 0.709 to 0.793 | 0.95 | 1.49 |

The table shows that discrimination is acceptable at four sites and below target at Site D. Calibration varies more than discrimination: the slope ranges from 0.64 to 1.19, and the O:E ratio ranges from 0.57 to 1.49.

### Step 7: Pooling

The random-effects pooled AUC is 0.755, with a 95 percent confidence interval of about 0.73 to 0.78. The fixed-effect estimate is 0.758. The difference is small because the between-site variance is small.

The pooled calibration slope is not simply the average of the site slopes. A random-effects meta-analysis of the slopes gives a pooled slope of about 0.95, with substantial heterogeneity.

### Step 8: Heterogeneity

Cochran's Q is 5.70 on 4 degrees of freedom, with a p-value of about 0.22. I-squared is 29.9 percent, and tau-squared is 0.0002 on the AUC scale. The heterogeneity is moderate but not extreme.

The prediction interval for a new site is approximately 0.70 to 0.81. The lower bound is below the 0.75 target, which means a new site could fall below the target even if the pooled estimate is acceptable.

### Step 9: Recalibration

Sites B, C, and E have O:E ratios outside the 0.9 to 1.1 range. Site B under-predicts risk (O:E 0.81), Site C over-predicts (O:E 1.27), and Site E under-predicts (O:E 1.49). Site D has a slope of 0.64, indicating that the predictions are too extreme.

The recalibration plan is:

- intercept-only recalibration for Sites B, C, and E,
- slope and intercept recalibration for Site D, with a separate validation sample,
- no recalibration for Site A.

### Step 10: Recalibration validation

The recalibration for Sites B, C, and E was fitted on a random half of each site's data and validated on the other half. After recalibration, the O:E ratios were 0.98, 1.03, and 1.01, and the slopes were 0.97, 1.01, and 0.99.

The Site D recalibration was fitted on half the data and validated on the other half. The slope improved from 0.64 to 0.92, but the AUC remained 0.68. The recalibration restored calibration but not discrimination.

### Step 11: Subgroups

The subgroup analysis shows that the AUC is lower in patients aged 80 and older at Site D (0.61) and in patients with chronic kidney disease at Site B (0.66). The calibration slope in patients with chronic kidney disease at Site B is 0.72. The report flags these findings.

### Step 12: Transportability

A meta-regression of AUC on sepsis prevalence shows a modest negative association: sites with higher prevalence have slightly lower discrimination. The association is not statistically significant with five sites, and the report labels it exploratory.

A reweighting analysis to the region's hospital mix changes the pooled AUC by less than 0.01. The result is not sensitive to the site mix.

### Step 13: Decision

The decision rule is:

- deploy if the pooled AUC is at least 0.75 and the pooled slope is between 0.8 and 1.2;
- deploy with recalibration if the AUC is adequate but the slope is outside the range;
- restrict if a subgroup has materially worse performance;
- redevelop if the AUC is below 0.70 or calibration cannot be restored.

The committee decides:

- Site A: deploy without recalibration.
- Sites B, C, and E: deploy with intercept recalibration and monthly monitoring.
- Site D: do not deploy until the model is redeveloped or a site-specific model is validated.
- All sites: restrict the claim to adults under 80 until the subgroup analysis is repeated.

### Reporting the validation

The report follows TRIPOD+AI. It includes the site-selection rationale, the variable mapping, the forest plot, the calibration plots, the heterogeneity statistics, the recalibration results, and the decision rule. It states the limitations: five sites, one region, a small site with 55 events, and a subgroup with worse performance.

### Two diagnostics that reveal a flawed design

**Diagnostic 1: the calibration slope versus the AUC.** Site D has an AUC of 0.677 and a slope of 0.64. The low AUC indicates a discrimination problem; the low slope indicates that the predictions are too extreme. A report that looked only at the pooled AUC of 0.755 would miss the site-level failure. The diagnostic is to plot the site-specific calibration slopes and AUCs, not only the pooled estimate.

**Diagnostic 2: the event count per site.** Site D has 55 events, which is below the 100-event rule. Its calibration estimate is imprecise, and the wide confidence interval (0.594 to 0.760) reflects that. A report that pooled Site D with the others without flagging the precision would overstate the certainty. The diagnostic is to report the event count and the confidence interval for each site.

### Vary one assumption: the number of sites

If the validation had only three sites (A, C, and E), the pooled AUC would be similar, but the heterogeneity estimate would be unstable. With three sites, tau-squared is estimated with very little information, and the prediction interval is unreliable. The design response is to include more sites or to use a Bayesian hierarchical model with a weakly informative prior.

### Vary one assumption: the case mix

If Site E had a sepsis prevalence of 35 percent rather than 23.8 percent, the O:E ratio would rise further above 1, and the calibration slope would move further from 1. The discrimination might improve or worsen depending on the case mix. The design response is to recalibrate and to report the prevalence.

### Vary one assumption: the recalibration sample

If the recalibration were fitted and validated on the same half of the data, the post-recalibration O:E ratio would be closer to 1 than the true out-of-sample value. The in-sample result would overstate the improvement. The design response is to use a separate sample, as the example does.

### The decision

The committee's decision is conditional and site-specific. It is not a single yes or no. The example shows that external validation produces a distribution of performance, and the deployment decision must respect that distribution.

The vendor's claim changes from "the model works" to "the model works at four of five sites after recalibration, and not at the fifth." The narrower claim is more credible and more useful to the buyer.

### What the committee saw

| Site | AUC | Slope | O:E | Decision | Reason |
|---|---:|---:|---:|---|---|
| A | 0.751 | 0.95 | 1.11 | Deploy | Calibration acceptable |
| B | 0.766 | 1.02 | 0.81 | Deploy with recalibration | Under-prediction |
| C | 0.781 | 1.19 | 1.27 | Deploy with recalibration | Over-prediction |
| D | 0.677 | 0.64 | 0.57 | Do not deploy | Discrimination and calibration failure |
| E | 0.751 | 0.95 | 1.49 | Deploy with recalibration | Under-prediction |

The table is the decision. It shows that the pooled AUC of 0.755 is not the basis for the decision; the site-level pattern is.

### Why Site D failed

Site D differs from the development sites in four ways:

- It is smaller (250 beds) and has a lower sepsis prevalence (9.2 percent).
- Its data system is different, and the lactate missingness is 22 percent.
- Its coefficient scale is 0.7, which weakens the predictor-outcome relationship.
- Its case mix includes more patients with chronic kidney disease.

The first two differences affect measurement; the third affects the relationship itself. Recalibration can address the baseline risk but not the weakened relationship. The AUC remains 0.68 after recalibration, which is the signature of a discrimination failure rather than a calibration failure.

### Recalibration arithmetic

For Site E, the observed risk is 23.8 percent and the mean predicted risk is 16.0 percent, so the O:E ratio is 1.49. Intercept-only recalibration adds a constant to the log-odds:

```text
new_intercept = old_intercept + log(O:E)
              = old_intercept + log(1.49)
              = old_intercept + 0.399
```

After recalibration, the mean predicted risk equals the observed risk, and the O:E ratio is 1.00. The slope is unchanged, because the recalibration shifts the intercept only.

For Site D, the slope is 0.638, so slope and intercept recalibration is needed. The recalibrated linear predictor is:

```text
lp_new = a + b * lp_old
```

where b is estimated from the calibration regression. After recalibration, the slope is about 0.92 and the O:E ratio is about 1.00, but the AUC is unchanged because the transformation is monotonic.

## 7. Code, query, or calculation

The code below simulates a five-site external validation, computes site-specific AUC and calibration, pools the AUC with a random-effects model, and reports heterogeneity. It uses only the Python standard library.

```python
import math
import random


def inv_logit(x):
    return 1 / (1 + math.exp(-x))


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
    for _ in range(60):
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


def generate_site(n, seed, prevalence_shift=0.0, coefficient_scale=1.0):
    random.seed(seed)
    y, lp = [], []
    for _ in range(n):
        x1 = random.gauss(0, 1)
        x2 = random.gauss(0, 1)
        z = -2.0 + prevalence_shift + coefficient_scale * (0.9 * x1 + 0.6 * x2)
        lp.append(-2.0 + 0.9 * x1 + 0.6 * x2)
        y.append(1 if random.random() < inv_logit(z) else 0)
    return y, lp


def se_auc(a, n1, n0):
    q1 = a / (2 - a)
    q2 = 2 * a * a / (1 + a)
    return math.sqrt((a * (1 - a) + (n1 - 1) * (q1 - a * a)
                      + (n0 - 1) * (q2 - a * a)) / (n1 * n0))


def random_effects(estimates, ses):
    w = [1 / s ** 2 for s in ses]
    fe = sum(wi * e for wi, e in zip(w, estimates)) / sum(w)
    q = sum(wi * (e - fe) ** 2 for wi, e in zip(w, estimates))
    df = len(estimates) - 1
    c = sum(w) - sum(wi * wi for wi in w) / sum(w)
    tau2 = max(0.0, (q - df) / c) if c > 0 else 0.0
    wr = [1 / (s * s + tau2) for s in ses]
    re = sum(wi * e for wi, e in zip(wr, estimates)) / sum(wr)
    i2 = max(0.0, (q - df) / q) * 100 if q > 0 else 0.0
    return round(fe, 3), round(re, 3), round(q, 2), round(i2, 1), round(tau2, 4)


if __name__ == "__main__":
    specs = [
        (1000, 1, 0.0, 1.0),
        (800, 2, -0.2, 0.85),
        (1200, 3, 0.3, 1.1),
        (600, 4, -0.4, 0.7),
        (900, 5, 0.6, 1.0),
    ]
    estimates, ses = [], []
    for n, seed, shift, scale in specs:
        y, lp = generate_site(n, seed, shift, scale)
        a = auc(y, lp)
        intercept, slope, _ = calibration(y, lp)
        events = sum(y)
        prevalence = events / n
        mean_pred = sum(inv_logit(s) for s in lp) / n
        estimates.append(a)
        ses.append(se_auc(a, events, n - events))
        print(f"site seed={seed}: n={n} events={events} prevalence={prevalence:.3f} "
              f"auc={a:.3f} slope={slope:.3f} O:E={prevalence/mean_pred:.2f}")
    print("pooled (FE, RE, Q, I2, tau2):", random_effects(estimates, ses))
```

Expected output:

```text
site seed=1: n=1000 events=174 prevalence=0.174 auc=0.751 slope=0.947 O:E=1.11
site seed=2: n=800 events=99 prevalence=0.124 auc=0.766 slope=1.019 O:E=0.81
site seed=3: n=1200 events=242 prevalence=0.202 auc=0.781 slope=1.193 O:E=1.27
site seed=4: n=600 events=55 prevalence=0.092 auc=0.677 slope=0.638 O:E=0.57
site seed=5: n=900 events=214 prevalence=0.238 auc=0.751 slope=0.946 O:E=1.49
pooled (FE, RE, Q, I2, tau2): (0.758, 0.755, 5.7, 29.9, 0.0002)
```

The output shows three things.

- Discrimination is moderate and similar across four sites, but Site D is below the target.
- Calibration varies much more than discrimination: the slope ranges from 0.638 to 1.193, and the O:E ratio from 0.57 to 1.49.
- The heterogeneity is moderate (I-squared 29.9 percent), and the between-site variance is small on the AUC scale. A prediction interval would still be wider than the confidence interval.

The simulation is deliberately simple. A production analysis would use a hierarchical model, account for clustering, and adjust for case mix. The teaching point is the difference between a pooled estimate and a site-level decision.

### Intercept recalibration

The code below performs intercept-only recalibration for a site with a baseline-risk shift and reports the O:E ratio before and after.

```python
import math


def inv_logit(x):
    return 1 / (1 + math.exp(-x))


def recalibrate_intercept(y, lp):
    """Shift the intercept so mean predicted risk equals observed risk."""
    n = len(y)
    observed = sum(y) / n
    mean_pred = sum(inv_logit(s) for s in lp) / n
    shift = (math.log(observed / (1 - observed))
             - math.log(mean_pred / (1 - mean_pred)))
    recalibrated = [s + shift for s in lp]
    new_mean = sum(inv_logit(s) for s in recalibrated) / n
    return {
        "observed": round(observed, 3),
        "mean_predicted_before": round(mean_pred, 3),
        "shift": round(shift, 3),
        "mean_predicted_after": round(new_mean, 3),
        "O:E_before": round(observed / mean_pred, 2),
        "O:E_after": round(observed / new_mean, 2),
    }


if __name__ == "__main__":
    # illustrative: 900 patients, 214 events, mean predicted risk 0.160
    y = [1] * 214 + [0] * 686
    lp = [-1.66] * 900  # logit(0.16)
    print(recalibrate_intercept(y, lp))
```

Expected output:

```text
{'observed': 0.238, 'mean_predicted_before': 0.16, 'shift': 0.495, 'mean_predicted_after': 0.238, 'O:E_before': 1.49, 'O:E_after': 1.0}
```

The output shows that intercept recalibration restores the O:E ratio to 1.00 without changing the slope or the discrimination. It is the right response when the predictor-outcome relationship is stable and the baseline risk differs.

### A prediction interval calculation

The prediction interval for a new site is approximately:

```text
pooled_estimate +/- 1.96 * sqrt(tau2 + mean_sampling_variance)
```

For the AUC, with a pooled estimate of 0.755, tau-squared of 0.0002, and a mean sampling variance of about 0.0005, the interval is:

```text
0.755 +/- 1.96 * sqrt(0.0007) = 0.755 +/- 0.052
```

so approximately 0.70 to 0.81. The lower bound is below the 0.75 target. A new site could fall below the target even though the pooled estimate is acceptable.

## 8. Data, standards, and regulatory context

### Binding requirements

**EU Medical Device Regulation.** Regulation (EU) 2017/745 requires clinical evaluation for medical devices, including software with a medical purpose. The clinical evaluation must be based on clinical data, and the manufacturer must demonstrate conformity with the general safety and performance requirements. For high-risk devices, a clinical investigation may be required. The regulation is binding.

**EU Artificial Intelligence Act.** Regulation (EU) 2024/1689 establishes a risk-based framework for AI systems. AI systems used in medical devices are generally high-risk, and the Act imposes obligations on providers, including risk management, data governance, technical documentation, and post-market monitoring. The obligations apply in phases. The Act is binding and interacts with the MDR.

**US device regulation.** The FDA's premarket approval and De Novo pathways require valid scientific evidence. For AI-enabled devices, the FDA has issued guidance on predetermined change control plans and lifecycle management. These are guidance under the device statute, not standalone regulations, but they define the review expectations.

### Formal guidance

**TRIPOD+AI.** The statement (BMJ 2024;385:e078378) replaces TRIPOD 2015. It provides a 27-item checklist with 52 subitems for reporting prediction models developed with regression or machine learning. It emphasises calibration, subgroup performance, fairness, and open science. It is a reporting guideline, not a design requirement.

**PROBAST+AI.** The tool (BMJ 2025;388:e082505) replaces PROBAST 2019. It assesses the quality of model development and the risk of bias in model evaluation, with 16 signalling questions for development and 18 for evaluation across four domains: participants and data sources, predictors, outcome, and analysis. It adds explicit attention to fairness and algorithmic bias.

**FDA predetermined change control plan.** The final guidance "Marketing Submission Recommendations for a Predetermined Change Control Plan for Artificial Intelligence-Enabled Device Software Functions" was issued in December 2024, with a later listing in August 2025. It describes how a manufacturer can prespecify modifications to an AI-enabled device and the methodology for developing, validating, and implementing them.

**FDA lifecycle guidance.** The final guidance "Artificial Intelligence-Enabled Device Software Functions: Lifecycle Management and Marketing Submission Recommendations" was issued in January 2025. It covers the total product lifecycle, including data management, model development, validation, and post-market monitoring.

**Good machine learning practice.** The FDA, Health Canada, and the UK Medicines and Healthcare products Regulatory Agency jointly published guiding principles for good machine learning practice in 2021. The principles are not binding, but they describe the expected practices for data quality, model development, and monitoring.

### What the standards constrain and what they leave open

| Instrument | Constrains | Leaves open |
|---|---|---|
| MDR | Clinical evaluation and evidence | The validation design and metrics |
| AI Act | Risk management and documentation | The performance thresholds |
| TRIPOD+AI | Reporting content | The study design |
| PROBAST+AI | Appraisal domains | The overall judgement |
| FDA PCCP guidance | The content of a change plan | The model and the validation strategy |
| FDA lifecycle guidance | Lifecycle activities | The specific implementation |

The pattern is familiar: the standards define the documentation and the process, not the scientific judgement. The judgement must be justified in the protocol and the validation report.

### Questions a regulator will ask

- What is the intended use, and does the validation population match it?
- How were the sites selected, and are they representative?
- How many events were there at each site?
- Were the predictors and outcomes harmonised?
- What were the site-specific and pooled estimates?
- What was the heterogeneity, and how was it interpreted?
- Was recalibration prespecified, and was it validated separately?
- Were subgroup and fairness analyses prespecified?
- What is the decision rule for a site that fails?
- What is the monitoring and update plan?

The questions are similar to the procurement questions, with more emphasis on the statistical analysis and the update plan.

### Data standards

External validation often uses data from multiple EHR systems. The Observational Medical Outcomes Partnership (OMOP) Common Data Model and the Fast Healthcare Interoperability Resources (FHIR) standard can improve harmonisation. The Clinical Data Interchange Standards Consortium (CDISC) standards are used for regulatory submissions. These standards constrain the representation of data, not the validation design.

A validation report should state the data source, the extraction date, the coding system, the outcome definition, and the missingness at each site. Without these, the validation cannot be reproduced.

## 9. Industry practice and case evidence

### The Epic Sepsis Model

Wong and colleagues externally validated a widely implemented proprietary sepsis prediction model at Michigan Medicine. The cohort included 27,697 patients with 38,455 hospitalisations; sepsis was present in 2,552 hospitalisations (7 percent). The hospitalisation-level AUC was 0.63 (95 percent confidence interval 0.62 to 0.64), compared with the vendor's reported 0.76 to 0.83. At the alert threshold of 6 or more, 18 percent of hospitalisations would trigger an alert, with sensitivity 33 percent and positive predictive value 12 percent.

The documented fact is the performance gap. The inference is that the development and validation populations differed in case mix, coding, and care processes, and that the vendor's performance claim was not reproduced. The study did not test whether the model reduced mortality; it tested whether the model performed as claimed.

**Lesson.** External validation is not a confirmatory formality. It can overturn a performance claim, and it should be conducted in the deployment population with the deployment outcome definition.

### IDx-DR

The FDA authorised IDx-DR through the De Novo pathway in 2018 (DEN180001) for autonomous detection of diabetic retinopathy in adults with diabetes. The authorisation was based on a prospective clinical study with a defined intended use, an image-acquisition protocol, and a reference standard. The claim was narrow: detect more than mild diabetic retinopathy and refer.

The documented fact is the authorisation and the intended-use restriction. The inference is that the validation was designed around a specific capture protocol and a specific decision, which made the performance claim credible. A broader claim would have required a broader validation.

**Lesson.** A narrow intended use makes external validation feasible and credible. A broad claim requires a broad site sample.

### The Apple Heart Study

The Apple Heart Study enrolled 419,297 participants in a siteless, app-based study. Over a median 117 days, 2,161 participants (0.52 percent) received an irregular pulse notification. Among 450 notified participants who returned analyzable ECG patches, atrial fibrillation was present in 34 percent; the positive predictive value for notification concordance with simultaneous patch-detected atrial fibrillation was 0.84.

The documented facts are the scale and the confirmation rate. The inference is that the study validated a detection pathway, not a clinical outcome. The incomplete patch return and the absence of a randomised comparison limited the claim.

**Lesson.** A validation study must match the claim. A detection claim needs a reference standard; an impact claim needs an outcome and a comparison.

### Model updates in practice

AI-enabled devices can change after authorisation. The FDA's PCCP guidance allows manufacturers to prespecify modifications and the validation methodology. The practice is to define the scope of permitted changes, the data and methods for validating them, and the impact assessment. A model update outside the PCCP requires a new submission.

The documented fact is the existence of the PCCP pathway. The inference is that manufacturers who plan for updates have a commercial advantage, because they can improve the model without a new authorisation cycle. The risk is that the PCCP becomes a route for uncontrolled drift; the guidance requires the change to be prespecified and validated.

**Lesson.** External validation is a lifecycle activity, not a one-time event. The design should include monitoring and a change-control plan.

### What companies actually do

The following practices are inferred from public guidance and published validation studies.

1. **Multi-site validation is increasingly expected.** Procurement committees and payers ask for evidence beyond the development site.
2. **Site selection is often convenience-based.** Companies use sites where they have relationships, which can underrepresent community hospitals.
3. **Calibration is under-reported.** Discrimination is reported more often than calibration, even though calibration is more decision-relevant.
4. **Recalibration is used as a deployment strategy.** Vendors recalibrate at the site rather than redevelop the model.
5. **Monitoring is included in contracts.** Deployments often include a performance monitoring clause.

### Where practice falls short

- **Single-site validation.** A model validated at one academic centre is presented as generalisable.
- **Pooled AUC without site detail.** The average hides the site that fails.
- **Calibration omitted.** An AUC reported without a calibration plot or slope.
- **No heterogeneity analysis.** The pooled estimate is presented without I-squared or tau-squared.
- **Recalibration in-sample.** The recalibrated model is evaluated on the fitting data.
- **No monitoring plan.** The model is deployed without a plan for drift detection.

### Practice comparison

| Practice | Mature teams | Less mature teams | Consequence |
|---|---|---|---|
| Site selection | Stratified by case mix | Convenience | Overstated generalisability |
| Events per site | Reported and adequate | Not reported | Imprecise calibration |
| Calibration | Slope and O:E reported | AUC only | Wrong thresholds |
| Heterogeneity | I-squared and prediction interval | Pooled estimate only | Hidden site failure |
| Recalibration | Separate validation sample | In-sample | Optimistic result |
| Subgroups | Prespecified | Post hoc | Hidden inequity |
| Decision rule | Prespecified | Ad hoc | Commercial pressure |
| Monitoring | Contractual | Absent | Silent drift |

The table is an inference from public guidance and validation reports, not a systematic survey.

### What a procurement committee will ask

- Which sites were validated, and how were they selected?
- How many events were there at each site?
- What were the AUC, calibration slope, and O:E ratio at each site?
- What was the heterogeneity across sites?
- What happens at a site that fails?
- Is there a recalibration plan?
- Is there a monitoring plan?
- What are the limitations of the validation?
- Does the validation match our hospital's case mix?
- Can we see the site-level data?

A vendor that cannot answer these questions is not ready for procurement.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Unrepresentative sites | All sites are academic | Convenience selection | Overstated generalisability | Stratified site selection |
| Single-site validation | One external site | No between-site variation | Hidden site failure | Multi-site design |
| Too few events per site | Wide calibration intervals | Small samples | Imprecise site estimates | Pool or increase sample |
| Predictor mismatch | Different units or definitions | Harmonisation failure | Wrong performance | Variable mapping; sensitivity |
| Outcome mismatch | Different codes or ascertainment | Measurement difference | Biased comparison | Harmonised definition |
| Pooled estimate hides failure | High I-squared ignored | Heterogeneity | Deployment at a failing site | Site-specific reporting |
| Calibration omitted | Only AUC reported | Calibration not assessed | Wrong thresholds | Calibration plot and slope |
| Recalibration in-sample | Same data for fit and test | Overfitting | Optimistic recalibration | Separate sample |
| Subgroup harm | Overall AUC acceptable | Differential performance | Inequitable care | Subgroup analysis |
| No decision rule | Committee interprets ad hoc | Commercial pressure | Wrong deployment | Prespecified rule |
| No monitoring | No post-deployment plan | Drift | Silent degradation | Monitoring and triggers |
| Model updated without validation | Version change after launch | Uncontrolled modification | Unknown performance | PCCP; change control |

### Red flags in an external validation report

- The sites are not described.
- The site-selection rationale is absent.
- The event count per site is not reported.
- The calibration slope is not reported.
- The heterogeneity statistics are not reported.
- The pooled estimate is reported without a prediction interval.
- The recalibration is evaluated on the fitting data.
- The subgroup analysis is post hoc.
- The decision rule is absent.
- The monitoring plan is absent.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The sites represent the target population | Transportability claim | Convenience sample | Site characteristics table |
| The model is locked | Validation evaluates a fixed model | Refitting | Version control |
| Predictors are harmonised | Model inputs match | Unit or definition differences | Variable mapping |
| Outcomes are harmonised | Performance is comparable | Coding differences | Outcome definition table |
| Events are sufficient per site | Precision | Rare outcome | Event count; CI width |
| The pooling model is appropriate | Pooled estimate is meaningful | High heterogeneity | I-squared; random effects |
| Recalibration is validated separately | Out-of-sample performance | In-sample evaluation | Holdout or temporal sample |
| Subgroups are prespecified | Fairness claim | Post hoc selection | Protocol |
| The decision rule is stable | Deployment decision | Changing criteria | Protocol lock |
| Performance is monitored | Drift detection | No monitoring | Post-deployment plan |
| The sites are independent | Variance estimation | Shared patients across sites | Site-level audit |
| The recalibration sample is representative | Out-of-sample validity | Selected sample | Random split |
| The decision rule is enforceable | Deployment decision | Commercial override | Governance |

### Limits of this topic

This topic covers the design and analysis of external validation. It does not cover the full theory of transportability, the details of hierarchical modelling, or the regulatory pathway for every device class. It also does not cover impact evaluation in detail; that is a separate design.

The worked example and code use simulated data. The numbers illustrate the relationships among site selection, heterogeneity, calibration, and deployment; they are not estimates from a real cohort. The regulatory statements are educational summaries, not legal advice.

## 12. Exercises

### Exercise 1 — Site selection (design, 35 minutes)

**Deliverable:** a site-selection plan for a model predicting readmission, with five sites stratified by size, teaching status, and data system, and a rationale for each.

**Acceptance criteria:** stratification justified; at least five sites; the target population represented; limitations stated.

**Time:** 35 minutes.

### Exercise 2 — Harmonisation (application, 30 minutes)

**Deliverable:** a variable-mapping table for five predictors across three data systems, including units, definitions, and missingness.

**Acceptance criteria:** each variable mapped; units converted where needed; missingness reported; unmapped variables flagged.

**Time:** 30 minutes.

### Exercise 3 — Site-specific performance (calculation, 35 minutes)

**Deliverable:** for a site with 800 patients, 120 events, an AUC of 0.72, a mean predicted risk of 0.18, and an observed risk of 0.15, compute the O:E ratio and interpret the calibration.

**Acceptance criteria:** O:E correct; interpretation correct; the limitation of the AUC stated.

**Time:** 35 minutes.

### Exercise 4 — Pooling (calculation, 40 minutes)

**Deliverable:** a random-effects pooled AUC for three sites with AUCs of 0.74, 0.78, and 0.70 and standard errors of 0.03, 0.02, and 0.04, including Q, I-squared, and tau-squared.

**Acceptance criteria:** formulas correct; pooled estimate correct; heterogeneity interpreted.

**Time:** 40 minutes.

### Exercise 5 — Heterogeneity (critique, 30 minutes)

**Deliverable:** an interpretation of a pooled AUC of 0.76 with I-squared of 75 percent and a prediction interval of 0.62 to 0.86, for a deployment decision.

**Acceptance criteria:** heterogeneity identified; prediction interval interpreted; decision recommendation stated.

**Time:** 30 minutes.

### Exercise 6 — Recalibration (design, 35 minutes)

**Deliverable:** a recalibration plan for a site with an O:E ratio of 1.4 and a slope of 0.85, including the method and the validation sample.

**Acceptance criteria:** method appropriate; separate validation sample; performance metrics stated.

**Time:** 35 minutes.

### Exercise 7 — Fairness (critique, 30 minutes)

**Deliverable:** a review of a validation report showing an overall AUC of 0.78 but an AUC of 0.66 in patients over 80, with a recommendation.

**Acceptance criteria:** subgroup identified; consequence stated; recommendation specific.

**Time:** 30 minutes.

### Exercise 8 — TRIPOD+AI report (application, 45 minutes)

**Deliverable:** a TRIPOD+AI-compliant outline for the worked example, with at least 15 items addressed.

**Acceptance criteria:** items mapped; evidence specified; gaps identified.

**Time:** 45 minutes.

## 13. Self-check answers

**Answer 1 (site selection).** A strong plan stratifies by size, teaching status, and data system, with at least one site per stratum, and states the target population. It acknowledges that five sites cannot represent all hospitals and that the claim is limited to similar settings. A weak plan lists convenient sites without a rationale.

**Answer 2 (harmonisation).** The table maps each predictor to the source variable, the unit, the conversion, the timing, and the missingness. Creatinine may require conversion from micromoles per litre to mg/dL. A diagnosis-based predictor may have different code sets. Unmapped variables are flagged and a sensitivity analysis is planned.

**Answer 3 (site performance).** The O:E ratio is 0.15 / 0.18 = 0.83, so the model over-predicts risk by about 17 percent. The AUC of 0.72 describes discrimination and says nothing about the absolute risk. The site may need intercept recalibration.

**Answer 4 (pooling).** The fixed-effect weights are 1,111, 2,500, and 625. The fixed-effect estimate is (0.74*1111 + 0.78*2500 + 0.70*625) / 4236 = 0.759. Q is the weighted sum of squared deviations, about 4.0 on 2 degrees of freedom. I-squared is about 50 percent, and tau-squared is positive. The random-effects estimate is slightly different, around 0.75.

**Answer 5 (heterogeneity).** The I-squared of 75 percent indicates substantial heterogeneity. The prediction interval of 0.62 to 0.86 means a new site could fall well below the target. The decision should be site-specific: deploy only after local validation or recalibration.

**Answer 6 (recalibration).** The O:E of 1.4 indicates under-prediction, and the slope of 0.85 indicates mild over-extremeness. Slope and intercept recalibration is appropriate. The recalibration should be fitted on one sample and validated on a separate sample, with O:E, slope, and net benefit reported.

**Answer 7 (fairness).** The overall AUC of 0.78 masks a subgroup AUC of 0.66 in patients over 80. The model may be less useful for the oldest patients, who are often the highest-risk group. The recommendation is to recalibrate or restrict the claim for that subgroup, and to report the subgroup performance.

**Answer 8 (TRIPOD+AI rubric).** A strong outline covers the title, abstract, introduction, methods (source of data, participants, outcome, predictors, sample size, missing data, analysis), results (participants, performance, subgroups), discussion, and open science. It specifies the evidence for each item and identifies the gaps. A weak outline lists the items without evidence.

## 14. Interview questions

**Question 1. What is external validation, and why is it necessary?**

A strong answer: validation in a different population or setting, necessary because development performance does not establish transportability. A weak answer says it is "testing on another dataset."

**Question 2. How do you select sites for an external validation?**

A strong answer: stratify by the dimensions expected to matter (size, case mix, care pathway, data system, geography), justify the sample, and report the limitations. A weak answer says "pick hospitals that will share data."

**Question 3. What is the difference between heterogeneity and sampling error?**

A strong answer: sampling error is within-site uncertainty and shrinks with sample size; heterogeneity is real between-site variation and does not shrink. A weak answer treats them as the same.

**Question 4. When would you use a random-effects model?**

A strong answer: when performance is expected to vary across sites and the goal is the mean of a distribution; report tau-squared and a prediction interval. A weak answer says "when there are many sites."

**Question 5. How do you recalibrate a model for a new site?**

A strong answer: intercept-only recalibration if the slope is near 1; slope and intercept if the slope is away from 1; validate on a separate sample; report O:E, slope, and net benefit. A weak answer says "adjust the intercept."

**Question 6. What would make you recommend against deployment at a site?**

A strong answer: AUC below the target, calibration that cannot be restored, a subgroup with materially worse performance, or an unrepresentative site. A weak answer says "if the AUC is low."

**Question 7. How do you assess fairness in an external validation?**

A strong answer: prespecified subgroup analyses of discrimination, calibration, and net benefit by age, sex, race and ethnicity, and other relevant characteristics, with confidence intervals and a statement of any material differences. A weak answer says "compare the AUC."

**Question 8. What does TRIPOD+AI require for external validation?**

A strong answer: reporting of the source of data, participants, outcome, predictors, sample size, missing data, analysis, performance with confidence intervals, calibration, subgroups, and open science. A weak answer says "a checklist."

**Question 9. Why is a prediction interval more useful than a confidence interval for deployment?**

A strong answer: because it describes the range of performance expected at a new site, including between-site variation; a confidence interval describes the uncertainty of the mean. A weak answer treats them as interchangeable.

**Question 10. How do you handle a missing predictor at a site?**

A strong answer: map it, impute it, or refit a reduced model; report performance with and without it; consider whether the site can deploy the full model. A weak answer says "drop the variable" without a sensitivity analysis.

## 15. Key terms

- **AUC:** area under the receiver operating characteristic curve.
- **Calibration:** agreement between predicted and observed risk.
- **Calibration slope:** the slope of observed outcome on predicted log-odds.
- **Case mix:** the distribution of predictors and baseline risk.
- **Decision threshold:** the probability above which an action is taken.
- **External validation:** validation in a different population or setting.
- **Fairness:** equitable performance across subgroups.
- **Fixed-effect model:** a model that assumes one true effect.
- **Geographic validation:** validation at different sites in the same era.
- **Harmonisation:** aligning predictor and outcome definitions across sites.
- **Heterogeneity:** real variation in performance across sites.
- **I-squared:** the proportion of variability due to heterogeneity.
- **Intended use:** the clinical purpose, population, user, and setting.
- **Model update:** adding predictors or changing coefficients.
- **Net benefit:** clinical utility at a threshold.
- **Observed-to-expected ratio (O:E):** observed events divided by expected events.
- **Pooled estimate:** the combined estimate across sites.
- **Prediction interval:** the range of performance expected at a new site.
- **Prospective validation:** validation on data collected after model lock.
- **PROBAST+AI:** risk-of-bias and applicability tool for prediction models.
- **Random-effects model:** a model that estimates the mean and between-site variance.
- **Recalibration:** updating the intercept or slope for a new population.
- **Site:** a hospital, clinic, or data source.
- **Spectrum effect:** performance varies with case mix.
- **Subgroup performance:** performance in a defined population subset.
- **Tau-squared:** the between-site variance.
- **Temporal validation:** validation on later data from the same sites.
- **Transportability:** whether a model applies in a different population.
- **TRIPOD+AI:** reporting guideline for prediction models.

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
- Debray TPA, Vergouwe Y, Koffijberg H, et al. A new framework to enhance the interpretation of external validation studies of clinical prediction models. J Clin Epidemiol 2015;68:279-289. https://doi.org/10.1016/j.jclinepi.2014.06.018
- Steyerberg EW. Clinical Prediction Models. 2nd edition. Springer, 2019.

### Case evidence

- Wong A, Otles E, Donnelly JP, et al. External Validation of a Widely Implemented Proprietary Sepsis Prediction Model in Hospitalized Patients. JAMA Intern Med 2021;181:1065-1070. https://doi.org/10.1001/jamainternmed.2021.2626
- Perez MV, Mahaffey KW, Hedlin H, et al. Large-Scale Assessment of a Smartwatch to Identify Atrial Fibrillation. N Engl J Med 2019;381:1909-1917. https://doi.org/10.1056/NEJMoa1901183
- FDA, De Novo classification decision for IDx-DR, DEN180001, 2018. https://www.accessdata.fda.gov/cdrh_docs/reviews/DEN180001.pdf

### Further reading

- Steyerberg EW, Vergouwe Y. Towards better clinical prediction models: seven steps for development and an ABCD for validation. Eur Heart J 2014;35:1925-1931. https://doi.org/10.1093/eurheartj/ehu207
- Justice AC, Covinsky KE, Berlin JA. Assessing the generalizability of prognostic information. Ann Intern Med 1999;130:515-524. https://doi.org/10.7326/0003-4819-130-6-199903160-00016
- Debray TPA, Moons KGM, Ahmed I, et al. A framework for developing, implementing, and evaluating clinical prediction models in an individual participant data meta-analysis. Stat Med 2013;32:3158-3180. https://doi.org/10.1002/sim.5732
- Van Calster B, McLernon DJ, van Smeden M, et al. Calibration: the Achilles heel of predictive analytics. BMC Med 2019;17:230. https://doi.org/10.1186/s12916-019-1466-7
- Wynants L, Van Calster B, Collins GS, et al. Prediction models for diagnosis and prognosis of covid-19: systematic review and critical appraisal. BMJ 2020;369:m1328. https://doi.org/10.1136/bmj.m1328

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic: [`07_estimands.md`](07_estimands.md)
- Next topic: [`09_evidence-synthesis.md`](09_evidence-synthesis.md)
- Related topic: [`04_validation-designs.md`](04_validation-designs.md)
- Module survey file: [`../../05_study_design_evidence.md`](../../05_study_design_evidence.md)
- Labs: [`../../labs/`](../../labs/)
- Calibration and decision curve lab: [`../../labs/06_calibration_decision_curve.ipynb`](../../labs/06_calibration_decision_curve.ipynb)
