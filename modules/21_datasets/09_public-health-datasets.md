# Public Health Datasets

> Module 21 — Datasets, Tools, and Benchmarks
> Topic 09 of 12
> Estimated study time: 4 hours
> Prerequisites: Topic 01, Topic 02
> Version 2026.3

## 1. Why this topic exists

Public health data are the main source of population denominators in biomedical data science. They are also the data most often analysed as if they were a simple random sample. A survey with a complex design, analysed without weights and design effects, produces a point estimate that is wrong and a confidence interval that is too narrow. The error is not exotic; it is the default in most tutorials.

The failures are specific:

- A national survey is analysed with an unweighted mean, and the estimate reflects the sample composition rather than the population.
- A confidence interval is computed with the simple random sample formula, and it is half the correct width.
- A state-level prevalence is estimated from a national survey with too few state observations, and the estimate is reported without an interval that reflects the small domain.
- A modelled global estimate is used as if it were a local count.
- A cancer registry's catchment area is treated as the national population.
- A surveillance system's case counts are treated as incidence without a denominator.
- Two surveys with different modes, response rates, and definitions are pooled without harmonisation.

This topic covers NHANES, BRFSS, NHIS, SEER, CDC WONDER, CDC surveillance systems, and the Global Burden of Disease (GBD). It teaches survey design, weighting, design effects, domain estimation, and the difference between a count, a rate, and a model-based estimate.

The central arithmetic is simple and decisive: a design effect of 2.5 means the effective sample size is 40 percent of the nominal sample, and the confidence interval is about 1.6 times wider than the simple random sample interval. A team that ignores it reports precision it does not have.

The practical consequence is that public health analysis has a different default from clinical analysis. A clinical cohort is usually a convenience sample analysed as if it were a population; a survey is a designed population sample analysed as if it were a convenience sample. The first error limits generalization, and the second produces a wrong estimate with a false interval.

A second consequence is that the metadata is the analysis. The weight variable, the strata, the primary sampling units, and the cycle are not background; they determine the estimator. A data card that omits them is not reproducible.

> **Note:** Survey cycles, weights, and definitions change. NHANES releases two-year cycles with sample weights; BRFSS changes its methodology and weighting over time; SEER catchment areas differ. Verify the cycle, the weight variable, and the documentation before analysis.

## 2. Learning outcomes

By the end of this topic you can:

1. **Distinguish** a count, a rate, a prevalence estimate, and a model-based estimate, and state the denominator each requires. Assessed by Exercise 1 and self-check item 1.
2. **Apply** survey weights and compute a weighted prevalence from a stratified sample. Assessed by Exercise 2.
3. **Compute** a design effect, an effective sample size, and a complex-survey confidence interval. Assessed by Exercise 3.
4. **Estimate** a domain (subgroup) and explain why the domain sample size, not the total, determines precision. Assessed by Exercise 4.
5. **Evaluate** a registry or surveillance dataset for catchment, reporting lag, and case definition. Assessed by Exercise 5.
6. **Distinguish** observed data from modelled estimates in GBD and similar products. Assessed by Exercise 6.
7. **Design** a small-area estimate that combines a survey and a model without overstating precision. Assessed by Exercise 7.
8. **Write** a public health data card that states the design, weights, denominator, and prohibited claims. Assessed by Exercise 8.

## 3. The core mental model

Public health data are produced by a measurement system with a design, a denominator, and a reporting process. The analysis must respect all three.

```text
   TARGET POPULATION
        |
        v
   SAMPLING FRAME
   household | telephone | registry catchment | surveillance site
        |
        v
   SAMPLE DESIGN
   strata | primary sampling units | clusters | oversampling
        |
        v
   RESPONSE AND MEASUREMENT
   interview | examination | laboratory | report | claim
        |
        v
   WEIGHTS
   base weight | non-response | post-stratification | raking
        |
        v
   ESTIMATE
   weighted prevalence | rate | count | model-based estimate
        |
        v
   UNCERTAINTY
   design effect | replicate weights | credible interval
```

**The design rule.** The sample design determines the estimator. A stratified multistage sample needs strata, primary sampling units, and weights. A simple random sample formula is wrong for both the point estimate and the variance.

**The weight rule.** A weight is the number of population units the observation represents. Weights correct for unequal selection probability, non-response, and post-stratification. An unweighted analysis estimates the sample, not the population.

**The design-effect rule.** Design effect (DEFF) is the ratio of the variance under the complex design to the variance under a simple random sample of the same size. It captures clustering, stratification, and weighting. Effective sample size is the nominal size divided by DEFF.

**The domain rule.** A subgroup estimate uses only the subgroup's observations. The precision depends on the domain sample size and the domain design effect, not on the total sample size. A national survey with 10,000 respondents may have 300 in a state.

**The model rule.** A modelled estimate is not an observation. GBD, small-area estimates, and nowcasts combine data with assumptions. The uncertainty interval reflects both sampling and model uncertainty, and the estimate cannot be validated against a local count unless one exists.

| Estimate type | Numerator | Denominator | Uncertainty source |
|---|---|---|---|
| Count | Cases | None | Reporting completeness |
| Proportion | Cases | Population at risk | Sampling and design |
| Rate | Events | Person-time | Sampling and model |
| Prevalence | Existing cases | Population | Sampling and design |
| Incidence | New cases | Person-time at risk | Ascertainment and denominator |
| Model-based estimate | Modelled | Modelled | Data and model assumptions |

## 4. Definitions and vocabulary

### 4.1 Survey design terms

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Target population | The population the estimate describes. | Confused with the sample | Definition |
| Sampling frame | The list from which the sample is drawn. | Assumed complete | Frame |
| Stratum | A group sampled separately. | Ignored in analysis | Categorical |
| Primary sampling unit | The first-stage sampled unit, often a county or segment. | Treated as independent | Cluster ID |
| Cluster | A group of observations sampled together. | Ignored in variance | Cluster ID |
| Oversampling | Deliberate over-representation of a group. | Corrected by weights | Design |
| Base weight | Inverse of the selection probability. | Used alone | Weight |
| Non-response weight | Adjustment for differential response. | Omitted | Weight |
| Post-stratification | Calibration to known population totals. | Confused with weighting | Weight |
| Raking | Iterative calibration to marginal totals. | Omitted | Weight |
| Replicate weight | Weights for variance estimation. | Ignored | Weight set |
| Finite population correction | Adjustment for sampling without replacement. | Omitted in large samples | Proportion |

### 4.2 Estimation terms

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Weighted prevalence | Prevalence using survey weights. | Replaced by unweighted | Proportion |
| Design effect | Complex variance divided by SRS variance. | Assumed to be 1 | Ratio |
| Effective sample size | Nominal size divided by design effect. | Equated with nominal size | Count |
| Standard error | Standard deviation of an estimate. | Computed with SRS formula | Proportion |
| Confidence interval | Interval covering the parameter at a stated rate. | Reported without the design | Interval |
| Domain | A subgroup for which an estimate is made. | Treated as a separate sample | Subgroup |
| Small-area estimate | An estimate for a geography with little data. | Treated as observed | Model output |
| Raking | Calibration to marginal totals. | Ignored | Method |
| Imputation | Filling missing values. | Treated as observed | Method |
| Suppression | Hiding small counts for privacy. | Ignored in totals | Rule |

### 4.3 Data source terms

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| NHANES | National Health and Nutrition Examination Survey. | Treated as state-representative | Survey |
| BRFSS | Behavioral Risk Factor Surveillance System. | Treated as measured prevalence | Survey |
| NHIS | National Health Interview Survey. | Treated as clinical data | Survey |
| SEER | Surveillance, Epidemiology, and End Results. | Treated as national | Registry |
| CDC WONDER | CDC query system for mortality and natality. | Treated as complete | Database |
| NWSS | National Wastewater Surveillance System. | Treated as clinical incidence | Surveillance |
| ESSENCE | Electronic Surveillance System for the Early Notification of Community-based Epidemics. | Treated as complete | Surveillance |
| GBD | Global Burden of Disease study. | Treated as observed counts | Model |
| Catchment | The population a registry serves. | Assumed national | Geography |

### 4.4 Surveillance and model terms

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Syndromic surveillance | Monitoring symptoms and codes before diagnosis. | Treated as confirmed cases | Signal |
| Notifiable disease | A disease required by law to be reported. | Assumed complete | List |
| Reporting completeness | The share of cases reported. | Assumed 100 percent | Proportion |
| Nowcast | An estimate of incomplete recent data. | Reported as final | Estimate |
| Backfill | Late addition of reports. | Ignored in trends | Process |
| Revision | A changed estimate. | Treated as error | Version |
| Excess mortality | Deaths above an expected baseline. | Treated as cause-specific | Rate |
| Population attributable fraction | The share of disease due to an exposure. | Treated as causal proof | Proportion |
| Model ensemble | A combination of models. | Treated as observation | Model |
| Credible interval | A Bayesian interval. | Confused with confidence interval | Interval |

### 4.5 Epidemiological terms

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Prevalence | Proportion with a condition at a time. | Confused with incidence | Proportion |
| Incidence | New cases per person-time. | Confused with prevalence | Rate |
| Person-time | Time at risk contributed by the population. | Replaced by headcount | Person-years |
| Age standardization | Adjusting for age structure. | Omitted in comparisons | Method |
| Crude rate | Unadjusted rate. | Compared across populations | Rate |
| Adjusted rate | Rate standardized to a reference. | Treated as the true rate | Rate |
| Reporting lag | Delay between event and report. | Ignored in trends | Days |
| Under-reporting | Cases not captured. | Treated as zero | Proportion |
| Case definition | Criteria for counting a case. | Changed mid-series | Definition |
| Notifiable disease | A disease required to be reported. | Assumed complete | List |

## 5. The mechanics, step by step

The procedure has 18 steps in four phases: define the estimand and the design, prepare the weights, estimate and quantify uncertainty, and validate and report.

### Phase A — Define the estimand and the design

**Step 1. Write the estimand.**
Input: the policy or clinical question.
Output: population, condition, time, geography, and the measure (count, prevalence, rate, or model estimate).
Rationale: the estimand determines the source and the denominator.
Branch: if the estimand is a local count, a national survey is not sufficient.
Edge case: a policy question can require both a national estimate and a local decision.

**Step 2. Identify the target population and the sampling frame.**
Input: the estimand and the source documentation.
Output: the frame, the covered population, and the known under-coverage.
Rationale: a frame that excludes a group cannot estimate that group.
Branch: if the target group is not in the frame, the source fails population fit.
Edge case: telephone surveys under-cover people without landlines or with only mobile phones, depending on the design.

**Step 3. Choose the data source.**
Input: the estimand and the candidate sources.
Output: a source with the right measure, geography, and cycle.
Rationale: measured data, self-report, claims, and registries answer different questions.
Branch: use the source decision table below.
Edge case: combining sources requires a model, not an average.

**Step 4. Read the design documentation.**
Input: the survey or registry documentation.
Output: strata, primary sampling units, weights, and variance method.
Rationale: the design determines the estimator and the variance.
Branch: if the documentation does not identify the design variables, the analysis cannot be design-correct.
Edge case: a public-use file can suppress design variables for privacy, which limits variance estimation.

### Survey profiles

| Source | Custodian | Unit | Coverage | Access and licence | What it cannot answer |
|---|---|---|---|---|---|
| NHANES | CDC NCHS | Person | National, two-year cycles | Open | State or clinic prevalence |
| BRFSS | CDC and states | Person | State and national | Open | Measured prevalence |
| NHIS | CDC NCHS | Person | National | Open | State estimates for small states |
| SEER | NCI | Cancer case | Registry catchment | Registration and DUA | National incidence |
| CDC WONDER | CDC | Death or birth | National and subnational | Open | Case-level clinical detail |
| NWSS | CDC | Site sample | Wastewater sites | Open | Individual incidence |
| ESSENCE | CDC and partners | ED visit | Participating sites | Varies | National completeness |
| GBD | IHME | Country or region | Global | Open aggregate | Local counts |

The coverage column is the first thing to read. A national survey does not become state-representative because the state is named in a subgroup.

### NHANES component weights

| Component | Weight variable | Unit | Common error |
|---|---|---|---|
| Interview | Interview weight | Person | Using the exam weight |
| Examination | Exam weight | Person | Ignoring the exam subsample |
| Laboratory | Lab weight | Person | Using the interview weight |
| Dietary | Dietary weight | Person-day | Treating days as people |
| Combined | Combined weight | Person | Combining without documentation |

Using the wrong weight changes the population represented, not only the precision. The documentation states which weight matches which component.

### Weighted and unweighted arithmetic

| Stratum | n | Prevalence | Weight | Weighted n |
|---|---:|---:|---:|---:|
| Young | 2,000 | 20% | 0.5 | 1,000 |
| Middle | 2,000 | 35% | 1.0 | 2,000 |
| Older | 1,000 | 50% | 2.5 | 2,500 |
| Total | 5,000 | 32% unweighted | | 39.1% weighted |

The unweighted estimate is 32 percent and the weighted estimate is 39.1 percent. The difference is 7.1 percentage points, which is larger than the confidence interval of either estimate. The weights are not a refinement; they change the answer.

### Variance methods

| Method | How it works | Requirement | Limitation |
|---|---|---|---|
| Taylor linearisation | Linearises the estimator | Design variables | Complex for nonlinear estimands |
| Jackknife | Re-estimates leaving out a PSU | Replicate weights or design | Computationally heavier |
| Bootstrap | Resamples PSUs | Replicate weights | Needs many replicates |
| SRS formula | Assumes independent observations | None | Wrong for complex designs |
| Replicate weights | Provided by the source | Weight files | Storage and setup |

The SRS formula is the default in most software and the most common error. A design-based method is required for a design-based estimate.

### Domain estimation

| Domain | Sample size | Design effect | Effective n | Interval width |
|---|---:|---:|---:|---:|
| National | 5,000 | 2.5 | 2,000 | Baseline |
| Age 65+ | 1,200 | 2.0 | 600 | Wider |
| State A | 300 | 1.8 | 167 | Much wider |
| State A, age 65+ | 80 | 2.2 | 36 | Very wide |

The domain sample size and its design effect determine precision. A national sample of 5,000 can contain 80 people in the domain that matters, and the interval for that domain is correspondingly wide.

### Surveillance systems

| System | Data | Strength | Limit |
|---|---|---|---|
| NWSS | Wastewater samples | Early signal | Sewer-shed denominators |
| ESSENCE | ED visits | Timely | Participating sites |
| Notifiable disease | Case reports | Specific diagnoses | Under-reporting |
| Syndromic surveillance | Symptoms and codes | Broad | Definition drift |
| Vital statistics | Deaths and births | Complete registration | Coding delay |

A surveillance count is not an incidence rate without a denominator and a case definition. The system's coverage and reporting lag are part of the estimate.

### Registry and GBD limits

| Source | Measure | Strength | Limit |
|---|---|---|---|
| SEER | Cancer incidence and survival | High-quality abstraction | Catchment, not national |
| NPCR | Cancer incidence | National coverage | Variable quality |
| GBD | Burden estimates | Comparable | Modelled, not observed |
| WHO | Country aggregates | Standardized | Reporting gaps |
| CDC WONDER | Mortality | National coverage | Cause coding |

A modelled estimate has an uncertainty interval that reflects model assumptions, not only sampling. It cannot be validated against a local count unless one exists.

### Standardization

| Method | What it removes | When to use | Limit |
|---|---|---|---|
| Direct standardization | Age structure | Comparing populations | Needs age-specific rates |
| Indirect standardization | Age and confounding | Sparse subgroups | Needs a reference |
| Age adjustment | Age only | Trend comparison | Does not remove sex or comorbidity |
| Crude rate | Nothing | Descriptive | Confounded by age |

A crude rate comparison across two states with different age structures measures the age difference as much as the health difference.

### Why the unweighted analysis is wrong

An unweighted analysis answers a different question: what is the prevalence among the people who happened to be sampled? That is not the target population, because the sample was selected with unequal probabilities and differential response. The weights restore the population.

The error has two parts. The point estimate shifts toward over-sampled groups, and the variance is understated because the design effect is ignored. The first makes the number wrong; the second makes the wrong number look precise.

| Error | What happens | Consequence |
|---|---|---|
| No weights | Over-sampled groups dominate | Prevalence biased |
| SRS variance | Clustering ignored | Interval too narrow |
| Wrong weight | Wrong component represented | Population mismatch |
| No domain check | Small subgroup treated as precise | Overconfident subgroup claim |

The correction is not optional and it is not a sensitivity analysis. It is the estimator.

### Reporting checklist for a survey estimate

| Item | What to report | Why |
|---|---|---|
| Source and cycle | Survey name and year | Comparability |
| Design | Strata, PSUs, weights | Estimator |
| Weight variable | Name and component | Population |
| Domain | Definition and sample size | Precision |
| Variance method | Taylor, replicate, or bootstrap | Interval |
| Design effect | Value or range | Effective sample |
| Interval | 95 percent, design-based | Inference |
| Suppression | Rules applied | Privacy |
| Limitations | Coverage and measurement | Interpretation |

A methods paragraph that omits the weight variable and the variance method is not reproducible.

### Weight construction

| Stage | What it corrects | Data needed | Failure if omitted |
|---|---|---|---|
| Base weight | Selection probability | Design | Unrepresentative sample |
| Non-response | Differential response | Response propensity | Bias |
| Post-stratification | Known population totals | Census or frame | Coverage bias |
| Raking | Multiple margins | Population margins | Inconsistent totals |
| Trimming | Extreme weights | Weight distribution | Variance inflation |

The final weight is a product of adjustments, and each adjustment has an assumption. A weight that sums to the population is necessary and not sufficient; the adjustments must match the target population.

### Common comparisons and their pitfalls

| Comparison | Naive method | Problem | Correct method |
|---|---|---|---|
| Two states | Compare crude rates | Age structure differs | Age-standardize |
| Two years | Compare weighted prevalence | Methodology may change | Test for a break |
| Two surveys | Average prevalences | Different constructs | Model-based synthesis |
| Two domains | Compare intervals | Covariance ignored | Design-based test |
| Registry vs survey | Compare counts | Different denominators | Compare rates |
| Model vs count | Treat as validation | Model inputs include the count | Independent validation |

The table is a checklist for the comparisons a reviewer will ask about. Each row has a different failure and a different correction.

### Phase B — Prepare the weights

**Step 5. Select the correct weight variable.**
Input: the cycle, the component, and the analysis unit.
Output: the weight variable for the interview, examination, or laboratory component.
Rationale: NHANES provides different weights for different components, and using the wrong one misstates the population.
Branch: if the analysis combines components, use the appropriate combined weight or a documented approach.
Edge case: a two-year weight is not valid for a four-year combined analysis without adjustment.

**Step 6. Check the weight distribution.**
Input: the weight variable.
Output: the sum, the mean, the maximum, and the design effect of the weights.
Rationale: extreme weights inflate variance and can dominate an estimate.
Branch: if a few weights dominate, consider trimming with a documented rule.
Edge case: trimming changes the estimand slightly and must be reported.

**Step 7. Handle missing data.**
Input: the variables and the missingness pattern.
Output: a complete-case or imputed analysis with a stated method.
Rationale: survey non-response and item missingness are different problems.
Branch: if missingness is differential, complete-case analysis is biased.
Edge case: multiple imputation must incorporate the survey design.

**Step 8. Define the domain.**
Input: the subgroup of interest.
Output: a domain indicator and the domain sample size.
Rationale: the domain, not the total, determines precision.
Branch: if the domain is small, use a model-based small-area estimate or report a wide interval.
Edge case: a domain can cross strata, which complicates variance estimation.

### Phase C — Estimate and quantify uncertainty

**Step 9. Compute the weighted estimate.**
Input: the outcome, the weight, and the domain.
Output: a weighted proportion, mean, or total.
Rationale: the weight converts the sample to the population.
Branch: if the weights do not sum to the population total, check the calibration.
Edge case: a weighted count can be non-integer; report it as an estimate.

**Step 10. Compute the variance with the design.**
Input: the design variables or replicate weights.
Output: a design-based standard error.
Rationale: the SRS formula understates the variance for clustered and weighted samples.
Branch: use Taylor linearisation, jackknife, or bootstrap as documented.
Edge case: a single PSU in a stratum prevents variance estimation; use the documented pooled-stratum approach.

**Step 11. Compute the design effect and effective sample size.**
Input: the design variance and the SRS variance.
Output: DEFF and n/DEFF.
Rationale: the design effect communicates how much information the sample actually provides.
Branch: if DEFF is large, the nominal sample overstates precision.
Edge case: DEFF varies by variable and domain; do not apply one value to all estimates.

**Step 12. Compute the confidence interval.**
Input: the estimate and the design-based standard error.
Output: a confidence interval on the appropriate scale.
Rationale: the interval is the unit of inference.
Branch: for proportions near 0 or 1, use a transformed or exact interval.
Edge case: a domain with a small sample can have an interval that includes implausible values.

**Step 13. Test a subgroup difference with the design.**
Input: two domain estimates and their covariance.
Output: a design-based test or interval for the difference.
Rationale: overlapping intervals are not a test, and independent-sample formulas are wrong for complex designs.
Branch: use a regression model with survey design or a replicate-based test.
Edge case: the covariance between domains can be non-zero.

### Phase D — Validate and report

**Step 14. Compare with an external benchmark.**
Input: the estimate and an independent source.
Output: a direction and magnitude of difference.
Rationale: a large discrepancy signals a design or definition error.
Branch: if the difference is large, audit the weight, the domain, and the case definition.
Edge case: two sources can measure different constructs and differ legitimately.

**Step 15. Check suppression and privacy rules.**
Input: the tables and the source rules.
Output: a disclosure-safe table.
Rationale: small counts can identify people.
Branch: if a cell is below the threshold, suppress or aggregate.
Edge case: complementary suppression can be needed to prevent back-calculation.

**Step 16. Report the design and the uncertainty.**
Input: the analysis and the documentation.
Output: a methods paragraph with the design, weights, variance method, and interval.
Rationale: a reader cannot interpret the estimate without the design.
Branch: if the design is not reported, the estimate is not reproducible.
Edge case: a model-based estimate needs the model and its assumptions.

**Step 17. State what the data cannot support.**
Input: the estimand and the source.
Output: a list of prohibited claims.
Rationale: the limits are part of the result.
Branch: if the claim is local and the source is national, prohibit the local claim.
Edge case: a national estimate can support a national policy claim and not a clinic-level one.

**Step 18. Re-run on cycle, weight, or definition change.**
Input: the release notes and the documentation.
Output: an updated record with a new date.
Rationale: surveys and registries change.
Branch: re-run when the cycle, the weight variable, the case definition, or the geography changes.
Edge case: a mid-series definition change can create a apparent trend.

### Weighted versus unweighted: a decision rule

| Situation | Weighted? | Why |
|---|---|---|
| Population prevalence | Yes | The estimand is the population |
| Sample description | No | The estimand is the sample |
| Trend within one survey | Yes | The population changes |
| Method comparison on a benchmark | Usually no | The benchmark defines the sample |
| Pooled multi-cycle analysis | Yes, with combined weights | The reference population spans cycles |
| Small-area model input | Yes | The model targets the population |

A team that reports both and explains the difference is more credible than a team that reports one and does not say which.

### Source decision table

| Estimand | Source | Strength | Limit |
|---|---|---|---|
| National measured prevalence | NHANES | Physical measurement | Not state-representative |
| State self-reported prevalence | BRFSS | State coverage | Self-report |
| National interview prevalence | NHIS | Broad content | Self-report |
| Cancer incidence and survival | SEER | Registry quality | Catchment, not national |
| Mortality by cause | CDC WONDER | National coverage | Coding and suppression |
| Local outbreak signal | NWSS, ESSENCE | Timely | Denominator and definitions |
| Global burden | GBD | Comparable across countries | Modelled, not observed |
| Claims-based prevalence | Medicaid or commercial claims | Large and local | Diagnosis and access bias |

### Design-effect table

| Design feature | Effect on variance | Typical direction |
|---|---|---|
| Clustering | Observations within clusters are correlated | Increase |
| Stratification | Separate sampling within strata | Decrease |
| Unequal weights | A few observations represent many | Increase |
| Oversampling | Some groups selected more often | Depends on weights |
| Post-stratification | Calibration to known totals | Usually decrease |
| Small domain | Few observations | Increase |

### Edge cases that change the procedure

**Licence forbids commercial use.** Most public health data are open, but SEER and some linked files require a data use agreement, and commercial use can be restricted. Check each source.

**Dataset withdrawn or revised.** A survey cycle can be revised, and a registry can change its catchment. Pin the release and record the revision.

**Credentialing takes months.** SEER and linked mortality files can require approval. A timeline shorter than the access path fails Gate 3.

**Version drift.** NHANES weights are cycle-specific, and BRFSS changes its weighting methodology. A pooled analysis across a methodology change is not comparable without adjustment.

**Label errors discovered.** Self-report and diagnosis codes misclassify the condition. Validate against a measured source and report the error.

**Benchmark contamination.** Public health benchmarks are not training sets, but a model trained on a survey cycle and evaluated on the same cycle is contaminated. Use a later cycle or a different geography.

## 6. Worked example

### The case

A state Medicaid programme wants to size a 12-month hypertension management programme. The programme needs the prevalence of diagnosed and undiagnosed hypertension among adult enrollees, by age band and region, to estimate the number eligible and the staff required.

**Estimand.** Prevalence of hypertension among adult Medicaid enrollees in the state, by age band and region, in the most recent year, with a 95 percent interval.

**Unit.** The person. The estimate is a proportion, and the denominator is the eligible enrollee population.

**Decision.** The programme will be sized to the upper bound of the interval if the lower bound would under-staff it. The cost of under-staffing is a waiting list; the cost of over-staffing is unused capacity.

### Candidate long list

| Candidate | Custodian | Unit | Coverage | Access and licence | Why it is on the list |
|---|---|---|---|---|---|
| NHANES | CDC NCHS | Person | National, two-year cycles | Open | Measured blood pressure |
| BRFSS | CDC and state | Person | State and national | Open | State-level self-report |
| NHIS | CDC NCHS | Person | National | Open | Interview detail |
| SEER | NCI | Cancer case | Registry catchment | Registration and DUA | Not hypertension |
| GBD | IHME | Country or region | Global | Open aggregate | Modelled, not state |
| Medicaid claims | State agency | Enrollee | State | DUA and approval | Diagnosis and utilisation |

### Gate scores for the state Medicaid claim

| Gate | NHANES | BRFSS | NHIS | SEER | GBD | Medicaid claims |
|---|---:|---:|---:|---:|---:|---:|
| 1 Decision fit | 2 | 1 | 1 | 0 | 1 | 1 |
| 2 Population fit | 1 | 2 | 1 | 0 | 1 | 2 |
| 3 Access fit | 2 | 2 | 2 | 1 | 2 | 1 |
| 4 Licence fit | 2 | 2 | 2 | 1 | 2 | 1 |
| 5 Benchmark fit | 1 | 1 | 1 | 1 | 0 | 1 |
| Veto | none | none | none | decision, population | benchmark | none |

**NHANES, Gate 1 and Gate 2.** NHANES measures blood pressure, which supports a prevalence estimate with less misclassification than self-report. It is a national sample and does not provide state or Medicaid-specific estimates with adequate precision. Scores 2 and 1.

**BRFSS, Gate 2.** BRFSS provides state-level self-reported prevalence with large samples. Self-report underestimates undiagnosed hypertension and is affected by question changes. Scores 1 and 2.

**NHIS, Gate 2.** The survey is national and has good interview detail, and it is not state-representative for small states. Score 1.

**SEER.** The registry covers cancer, not hypertension. Veto on decision fit.

**GBD, Gate 5.** The estimates are modelled at country or region level and do not support a state Medicaid claim. Score 0 on benchmark fit for this estimand.

**Medicaid claims, Gate 1 and Gate 2.** Claims identify diagnosed hypertension and utilisation, and they miss undiagnosed disease and people who do not access care. They are state-specific and enrollee-specific. Scores 1 and 2.

### The two diagnostics

**Diagnostic 1: the unweighted estimate is wrong.** In the worked arithmetic, an unweighted analysis of a stratified sample gives 32 percent and the weighted analysis gives 39.1 percent, a difference of 7.1 points. The weights change the estimate because the sample over-represents younger people. An unweighted analysis is not a conservative approximation; it estimates a different population.

**Diagnostic 2: the design effect changes the decision.** With a design effect of 2.5, the effective sample size is 2,000 rather than 5,000, and the 95 percent interval widens from about 37.7 to 40.4 percent to about 37.0 to 41.2 percent. The point estimate is the same, and the precision is not. A programme sized to the narrow interval under-staffs.

A third diagnostic is the self-report versus measurement gap. BRFSS and NHANES measure different constructs. A simple average of the two is not an estimate of either; a model-based synthesis is required, with assumptions stated.

### The selected route

The admissible set is NHANES, BRFSS, NHIS, and Medicaid claims. The plan uses NHANES to calibrate the measurement gap, BRFSS for state-level prevalence, and Medicaid claims for enrollee-specific diagnosed prevalence. The undiagnosed share is modelled, not observed.

| Track | Steps | Owner | Duration | Planning cost |
|---|---|---|---|---|
| NHANES analysis | Weighted prevalence and design effect | Biostatistics | 4 weeks | Staff time |
| BRFSS analysis | State prevalence by age and region | Biostatistics | 4 weeks | Staff time |
| Claims analysis | Diagnosed prevalence and enrolment | Data team | 8 weeks | DUA and staff |
| Synthesis | Measurement-gap model | Biostatistics | 6 weeks | Staff time |
| Review | Programme and clinical sign-off | Programme lead | 3 weeks | Internal |

### Time and cost

| Item | Detail | Calendar time | Planning cost |
|---|---|---|---|
| Public data | NHANES and BRFSS | 1 week | None |
| Medicaid DUA | Approval and transfer | 3 to 6 months | Legal and staff |
| Analysis | Weighted estimates and synthesis | 10 weeks | 0.5 FTE |
| Review | Clinical and programme | 3 weeks | Internal |
| Total to a decision | Critical path is the DUA | About 6 months | Staff-dominated |

### Substitute plan if Medicaid data are unavailable

If the DUA is not approved, the plan changes in three ways:

1. The estimate relies on BRFSS and NHANES, with a modelled enrollee adjustment and a wider interval.
2. The diagnosed and undiagnosed split cannot be estimated for the enrollee population, so the programme uses a range.
3. The programme is sized to the upper bound of the wider interval, with a review trigger after six months.

The fallback is honest about its uncertainty and less precise. It is not equivalent to the claims-based estimate.

### Vary one assumption: commercial use is restricted

Suppose the programme is run by a commercial vendor and the Medicaid DUA prohibits commercial use. The claims track closes, and the plan relies on public data with a model-based adjustment. The estimate is weaker, the interval wider, and the contract must state the limitation.

If the variation is the survey cycle, pooling two cycles changes the weights and the reference year. If the variation is the case definition, a change from self-report to measured hypertension changes the prevalence and the comparison across years.

### Conditions under which the case would fail

- **The Medicaid DUA is denied.** The enrollee-specific estimate relies on a model, and the interval widens.
- **The state domain is too small.** The regional estimate is not precise enough to size the programme.
- **BRFSS changes its methodology.** The trend and the prevalence need re-baselining.
- **The measurement gap is unstable.** The NHANES calibration does not transfer to the enrollee population.
- **The programme needs clinic-level counts.** No public source supports that geography.
- **The undiagnosed share is material and unmodelled.** The programme under-counts need.

The plan cannot answer a clinic-level demand estimate, a within-year trend, or a causal effect of the programme. Those require local data and a different design.

### Gate evidence log

| Gate | Candidate | Evidence inspected | Verdict | Residual uncertainty |
|---|---|---|---|---|
| 1 | NHANES | Measured blood pressure | 2 | Not enrollee-specific |
| 1 | BRFSS | Self-reported diagnosis | 1 | Undiagnosed disease missing |
| 1 | Medicaid claims | Diagnosis codes | 1 | Undiagnosed and access bias |
| 2 | NHANES | National sample | 1 | State and Medicaid gap |
| 2 | BRFSS | State sample | 2 | Mode and response bias |
| 3 | NHANES | Open download | 2 | None |
| 3 | Medicaid claims | DUA required | 1 | Approval time |
| 4 | NHANES | Public domain | 2 | None |
| 4 | Medicaid claims | DUA terms | 1 | Commercial use unclear |
| 5 | NHANES | Weighted analysis possible | 1 | No state benchmark |
| 5 | BRFSS | State estimates possible | 1 | Self-report bias |
| 5 | GBD | Modelled country estimates | 0 | Not state-level |

### Decision timeline

| Month | Public data track | Claims track | Programme track | Deliverable |
|---:|---|---|---|---|
| 1 | NHANES weighted analysis | DUA request | Define age bands | Methods note |
| 2 | BRFSS state analysis | Security review | Staffing model | State estimates |
| 3 | Measurement-gap model | Approval | Regional plan | Synthesis v1 |
| 4-5 | Sensitivity analysis | Data transfer | Budget | Final estimate |
| 6 | Review | Claims analysis | Sign-off | Programme decision |

### What the case demonstrates

- The unweighted estimate was 7.1 points lower than the weighted estimate. The weights changed the answer, not only the precision.
- The design effect widened the interval by 58 percent. The decision changed even though the point estimate did not.
- The state domain interval was more than three times wider than the national interval. The domain sample, not the national sample, set the precision.
- The source with the best measurement was not the source with the right population. The synthesis had to combine them.
- The claims data added diagnosed prevalence and missed undiagnosed disease. The undiagnosed share remained a model assumption.
- The DUA was the critical path, not the analysis.

### Sensitivity of the decision

| Change | Gate affected | Does the source change? | Plan response |
|---|---|---|---|
| Medicaid DUA approved | 3 | Yes, adds claims | Use claims for diagnosed prevalence |
| BRFSS changes its weighting | 2 and 5 | No | Re-run with the new weights |
| NHANES cycle changes | 1 and 5 | No | Update the calibration |
| The programme needs a regional estimate | 2 | Possibly | Model-based small-area estimate |
| The state expands Medicaid | 2 | No | Re-weight the enrollee population |
| A local survey is commissioned | 2 and 5 | Yes | Use it for the domain estimate |

The case is a good example of a decision that is robust to the point estimate and sensitive to the interval. The programme size changes when the upper bound changes, even though the central estimate is stable. That is why the design effect is not a technical detail; it is a decision input.

The case also shows why the analysis plan must be written before the data arrive. The decision rule, the upper bound, and the domain definitions were fixed in advance. If the team had chosen the interval after seeing the estimate, the programme size would have been a post hoc choice.

A final point: the interval, not the point estimate, drove the staffing decision. That is common in public health planning, and it is the reason the design effect belongs in the executive summary rather than the appendix.

The plan is defensible because it states the model, the interval, and the unmeasured share. A single number would have been easier to communicate and harder to defend.

The case is complete when the programme can state its size, its interval, and the assumptions behind the undiagnosed share.

## 7. Code, query, or calculation

The three blocks below use the Python standard library only. They implement a weighted prevalence, a design-effect calculation, and a domain interval.

### 7.1 Weighted prevalence

```python
strata = [
    ("young", 2000, 0.20, 0.5),
    ("middle", 2000, 0.35, 1.0),
    ("older", 1000, 0.50, 2.5),
]
n = sum(row[1] for row in strata)
unweighted_num = sum(row[1] * row[2] for row in strata)
weighted_num = sum(row[1] * row[2] * row[3] for row in strata)
weight_sum = sum(row[1] * row[3] for row in strata)

print(f"{'Stratum':<10}{'n':>7}{'Prevalence':>12}{'Weight':>8}{'Weighted n':>12}")
for name, count, prev, w in strata:
    print(f"{name:<10}{count:>7}{prev:>12.0%}{w:>8.1f}{count*w:>12,.0f}")
print()
print(f"Unweighted prevalence: {unweighted_num/n:.3f}")
print(f"Weighted prevalence:   {weighted_num/weight_sum:.3f}")
print(f"Difference:            {weighted_num/weight_sum - unweighted_num/n:+.3f}")
```

Expected output:

```text
Stratum         n  Prevalence  Weight  Weighted n
young        2000         20%     0.5       1,000
middle       2000         35%     1.0       2,000
older        1000         50%     2.5       2,500

Unweighted prevalence: 0.320
Weighted prevalence:   0.391
Difference:            +0.071
```

The arithmetic is deliberately small. The same calculation on a real survey uses the supplied weight variable and a domain indicator.

### 7.2 Design effect and interval

```python
import math

p = 0.3909
n = 5000
deff = 2.5

se_srs = math.sqrt(p * (1 - p) / n)
se_complex = math.sqrt(deff * p * (1 - p) / n)
n_eff = n / deff

print(f"Prevalence: {p:.4f}")
print(f"Sample size: {n:,}")
print(f"Design effect: {deff}")
print(f"SRS standard error:      {se_srs:.5f} ({se_srs:.2%})")
print(f"Complex standard error:  {se_complex:.5f} ({se_complex:.2%})")
print(f"Effective sample size:   {n_eff:,.0f}")
print()
print(f"SRS 95% CI:     {p-1.96*se_srs:.4f} to {p+1.96*se_srs:.4f}")
print(f"Complex 95% CI: {p-1.96*se_complex:.4f} to {p+1.96*se_complex:.4f}")
print()
print(f"CI width ratio: {(2*1.96*se_complex)/(2*1.96*se_srs):.2f}")
```

Expected output:

```text
Prevalence: 0.3909
Sample size: 5,000
Design effect: 2.5
SRS standard error:      0.00690 (0.69%)
Complex standard error:  0.01091 (1.09%)
Effective sample size:   2,000

SRS 95% CI:     0.3774 to 0.4044
Complex 95% CI: 0.3695 to 0.4123

CI width ratio: 1.58
```

The point estimate is unchanged and the interval is 58 percent wider. A decision sized to the SRS interval is overconfident.

### 7.3 Domain interval

```python
import math

def wilson(k, n, z=1.96):
    p = k / n
    denom = 1 + z*z/n
    centre = (p + z*z/(2*n)) / denom
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
    return centre - half, centre + half

for label, k, n, deff in (
    ("National", 1955, 5000, 2.5),
    ("State A", 117, 300, 1.8),
    ("State A age 65+", 40, 80, 2.2),
):
    lo, hi = wilson(k, n)
    se = math.sqrt(deff * (k/n) * (1 - k/n) / n)
    lo_d = k/n - 1.96*se
    hi_d = k/n + 1.96*se
    print(f"{label:<18} {k}/{n} = {k/n:.1%}  design CI {lo_d:.1%} to {hi_d:.1%}")
print()
print("The domain sample, not the national sample, determines the interval.")
```

Expected output:

```text
National           1955/5000 = 39.1%  design CI 36.9% to 41.2%
State A            117/300 = 39.0%  design CI 34.0% to 44.0%
State A age 65+    40/80 = 50.0%  design CI 37.7% to 62.3%

The domain sample, not the national sample, determines the interval.
```

The state estimate has the same point prevalence and an interval more than three times wider. The age-65 domain in the state is wider still. A programme that needs a regional estimate must budget for the domain sample.

## 8. Data, standards, and regulatory context

Public health data are governed by statistical standards, privacy rules, and reporting guidelines. The table separates binding requirements from conventions.

| Instrument | Status | What it constrains | What it does not do |
|---|---|---|---|
| OMB race and ethnicity standards | Binding for US federal statistics | Minimum categories and reporting | Capture every identity |
| NCHS survey standards | Agency standards | Design, weighting, documentation | Make a survey state-representative |
| BRFSS methodology | Programme standard | Sampling, weighting, questionnaire | Provide measured prevalence |
| SEER registries | Programme standards | Case ascertainment and coding | Cover the whole country |
| CDC suppression rules | Agency policy | Small-cell reporting | Eliminate disclosure risk |
| HIPAA | Binding in the US | PHI handling | Cover all public health data |
| GDPR | Binding in the EU | Personal data processing | Define a survey weight |
| 42 CFR Part 2 | Binding for SUD programmes | Substance use disorder records | Cover all behavioural health |
| STROBE | Reporting guideline | Observational study reporting | Validate an estimate |
| RECORD | Reporting guideline | Routinely collected data | Fix design errors |
| GATHER | Reporting guideline | Global health estimates | Validate a model |

**A survey is not a census.** The weights make it represent the population; they do not eliminate sampling error, non-response, or coverage bias.

**A registry is not national.** SEER covers specific registries and catchment areas. A SEER estimate is a estimate for that coverage, not for the United States.

**A modelled estimate is not a count.** GBD and small-area estimates combine data with assumptions. The interval reflects both, and the estimate should be labelled as modelled.

### Design and reporting standards

| Standard | What it gives | What it does not give |
|---|---|---|
| NCHS analytic guidelines | Weight selection and variance guidance | A ready analysis |
| BRFSS codebook | Variable definitions | Measured blood pressure |
| SEER coding manual | Case definitions | National coverage |
| GBD protocol | Model structure | Local validation |
| STROBE | Reporting structure | Design correction |
| RECORD | Routinely collected data reporting | Data quality |
| GATHER | Estimates reporting | Estimate accuracy |

A report that follows STROBE and ignores the survey design is still wrong. Reporting standards describe what to write, not how to estimate.

### Disclosure and suppression

| Rule | Threshold | Action | Consequence |
|---|---|---|---|
| Minimum cell count | Varies by source | Suppress or aggregate | Small counts hidden |
| Complementary suppression | Secondary cells | Suppress additional cells | Prevents back-calculation |
| Rounding | Nearest 5 or 10 | Round counts | Small differences lost |
| Rate suppression | Small numerator | Suppress rate | Rate unavailable |
| Geographic aggregation | Small area | Aggregate to a larger area | Local detail lost |

A suppressed cell is not zero, and a rounded count is not exact. A total computed from suppressed cells can be wrong, and a table designed without complementary suppression can leak the hidden value.

### Regulatory artefacts for a public health analysis

- **Data management plan** names the sources, the weights, and the retention.
- **Analysis plan** pre-specifies the estimand, the domain, and the variance method.
- **Disclosure review** applies suppression rules.
- **Methods appendix** reports the design and the interval.
- **Limitations** states coverage, measurement, and model assumptions.

## 9. Industry practice and case evidence

### What organisations actually do

- **Use the supplied weights and design variables.** Mature teams never compute an unweighted prevalence for a population claim.
- **Report design effects.** The effective sample size is part of the result.
- **Pre-specify domains.** A subgroup analysis is designed, not discovered.
- **Use measured data to calibrate self-report.** NHANES and BRFSS are combined with a model, not an average.
- **Label modelled estimates.** GBD and small-area estimates are marked as modelled.
- **Apply suppression rules.** Small cells are suppressed or aggregated.
- **Archive the cycle and weight variable.** Reproducibility depends on them.

### Case evidence

| Case | Documented fact | Inference |
|---|---|---|
| NHANES | The survey uses a complex, stratified, multistage design with sample weights and produces two-year cycle files. | An unweighted analysis estimates the sample, not the population. |
| BRFSS | The survey provides state-level self-reported data and has changed its weighting methodology over time. | A trend across a methodology change is not a pure trend. |
| SEER | The registries cover defined catchment areas and require registration for some data. | A SEER estimate is a catchment estimate, not a national one. |
| GBD | The study produces modelled estimates with uncertainty intervals for countries and regions. | A GBD estimate is not a local count and cannot be validated without local data. |
| CDC WONDER | The system provides mortality and natality aggregates with suppression for small counts. | A suppressed cell is not zero. |

The first four rows are stable. Specific cycle counts and methodology versions should be checked at the source.

### What good looks like

- The estimand names the population, geography, time, and measure.
- The weight variable matches the component.
- The variance method matches the design.
- The design effect and effective sample size are reported.
- The domain sample size is reported with the estimate.
- Modelled estimates are labelled as modelled.
- Suppression rules are applied.
- The data card states the prohibited claims.

### Documented fact versus inference

- **Documented fact:** a survey codebook, a registry manual, a methods report, or a data dictionary.
- **Inference:** a planning consequence for another geography or population.
- **Unverified summary:** a secondary account of a prevalence or a cycle.

A case table should carry a source and a status. Survey methods change, and a stale prevalence can misstate both the level and the trend.

### Why public health estimates mislead product teams

A product team wants a local market size. A national survey provides a national prevalence and a wide local interval. The temptation is to multiply the national prevalence by the local population and report a precise number. The result ignores the local age structure, the local coverage, and the survey's domain precision. The correct output is a range with a stated model.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Unweighted analysis | Weights are not mentioned | Sample composition differs from population | Biased prevalence | Use survey weights |
| SRS variance | Interval is narrow | Clustering ignored | Overconfidence | Design-based variance |
| Wrong weight | Component mismatch | Wrong population represented | Biased estimate | Match weight to component |
| Domain neglect | Subgroup estimate from total n | Small domain | False precision | Domain sample size |
| Pooling across cycles | Weights not adjusted | Different reference population | Biased trend | Use combined weights |
| Methodology change | Trend discontinuity | BRFSS weighting change | False trend | Stratify or adjust |
| Catchment error | SEER treated as national | Registry coverage | Wrong geography | State the catchment |
| Suppression ignored | Small cell treated as zero | Privacy rule | Under-count | Aggregate or suppress |
| Modelled as observed | GBD count treated as truth | Model assumptions | False certainty | Label as modelled |
| Case definition drift | Counts change without epidemiology | Definition change | False trend | Version the definition |
| Reporting lag | Recent period looks low | Delayed reports | False decline | Nowcast or truncate |
| Denominator error | Rate uses wrong population | Population estimate | Wrong rate | Match numerator and denominator |
| Age confounding | Crude rates compared | Age structure differs | False difference | Standardize |
| Ecological fallacy | County estimate applied to people | Group-level data | Wrong individual inference | Individual data or a model |
| Weight trimming undisclosed | Extreme weights removed silently | Variance control | Changed estimand | Report the rule |
| Single PSU stratum | Variance cannot be estimated | Design sparsity | Software error | Pooled stratum |
| Imputation ignores design | Variance understated | Missing data | False precision | Design-aware imputation |
| Cycle mismatch | Weights from another cycle | Wrong reference year | Biased estimate | Match cycle |
| Geography mismatch | State estimate used for county | Coverage | False local claim | Small-area model |
| Response bias | Respondents differ from non-respondents | Non-response | Bias | Non-response adjustment |
| Mode effect | Telephone versus web | Measurement | Trend artefact | Mode adjustment |
| Denominator drift | Population estimate changes | Migration | False trend | Same-year denominator |
| Nowcast instability | Recent estimates revised | Reporting lag | False signal | Report revision range |

Red flags that should stop a public health claim:

- The weights are not named.
- The interval uses an SRS formula.
- The domain sample size is not reported.
- A registry estimate is described as national.
- A modelled estimate is described as a count.
- A suppressed cell is treated as zero.
- A trend crosses a methodology change without a note.
- A county estimate is applied to individuals.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test |
|---|---|---|---|
| The weights match the estimand | Population representation | Wrong component | Read the documentation |
| The design variables are available | Variance estimation | Suppressed design | Use replicate weights |
| The domain is large enough | Precision | Small subgroup | Report interval |
| The case definition is stable | Comparability | Definition change | Version the definition |
| The catchment is as stated | Geography | Registry change | Check the documentation |
| The population denominator is current | Rates | Migration or enrolment change | Use the same year |
| Self-report is calibrated | Measurement | Undiagnosed disease | Compare with NHANES |
| The model is valid | Small-area estimates | Assumption violation | Validate where possible |
| Suppression is applied | Privacy | Small cells | Disclosure review |
| The cycle is comparable | Trends | Methodology change | Stratify or adjust |
| The reporting lag is stable | Recent estimates | Backfill | Truncate or nowcast |
| The standard population is appropriate | Age adjustment | Different standard | State the standard |
| The weight trimming is documented | Estimand stability | Silent trimming | Report the rule |
| The single-PSU strata are handled | Variance estimation | Sparse design | Pooled stratum |
| The imputation is design-aware | Interval validity | Naive imputation | Design-aware method |
| The mode effect is adjusted | Trend comparability | Survey mode change | Mode adjustment |
| The denominator is same-year | Rate validity | Population drift | Match years |

The deepest limit is that a population estimate is not a clinical estimate. A prevalence of 39 percent does not tell a clinic how many of its patients have uncontrolled blood pressure, and a county rate does not tell a physician what to do for the next patient.

## 12. Exercises

**Exercise 1. Classify estimates.**
Deliverable: a table of six estimates labelled count, proportion, rate, prevalence, incidence, or model.
Acceptance criteria: six estimates; numerator and denominator named; one mislabelled estimate corrected.
Time: 30 minutes.

**Exercise 2. Compute a weighted prevalence.**
Deliverable: a spreadsheet or script with strata, weights, and weighted and unweighted estimates.
Acceptance criteria: arithmetic correct; difference explained; weights sourced.
Time: 40 minutes.

**Exercise 3. Compute a design effect.**
Deliverable: SRS and complex standard errors, effective sample size, and both intervals.
Acceptance criteria: formulas shown; interval ratio stated; implication for a decision.
Time: 35 minutes.

**Exercise 4. Estimate a domain.**
Deliverable: a domain sample size, design effect, and interval for a named subgroup.
Acceptance criteria: domain defined; interval reported; precision limitation stated.
Time: 40 minutes.

**Exercise 5. Audit a registry.**
Deliverable: a one-page audit of SEER or a state registry covering catchment, case definition, and reporting lag.
Acceptance criteria: three limits; a prohibited claim; a comparison source.
Time: 45 minutes.

**Exercise 6. Critique a modelled estimate.**
Deliverable: a memo separating observed data from model assumptions in a GBD estimate.
Acceptance criteria: names the model; states the interval meaning; proposes a local validation.
Time: 40 minutes.

**Exercise 7. Design a small-area estimate.**
Deliverable: a plan combining a survey and a model, with assumptions and an interval.
Acceptance criteria: data sources; model; validation; prohibited claims.
Time: 50 minutes.

**Assessment note.** Exercises 2, 3, 4, and 5 produce checkable artefacts: a weighted table, a design-effect calculation, a domain interval, and a registry audit. Exercises 1, 6, 7, and 8 are design tasks and are marked against the rubric. Exercise 8 is the integration task.

**Exercise 8. Write a public health data card.**
Deliverable: a two-page card with design, weights, denominator, interval, and prohibited claims.
Acceptance criteria: every field populated or marked unknown; one prohibited claim; cycle recorded.
Time: 60 minutes.

## 13. Self-check answers

**Item 1. What is the difference between a count and a rate?**
A count is the number of events; a rate divides events by person-time at risk. A count without a denominator cannot be compared across populations or times.

**Item 2. Why is an unweighted survey analysis wrong?**
It estimates the sample, not the population. Unequal selection and differential response make the sample composition different from the target population.

**Item 3. What is a design effect?**
The ratio of the variance under the complex design to the variance under a simple random sample of the same size. It captures clustering, stratification, and weighting.

**Item 4. A design effect of 2.5 and n = 5,000 gives what effective sample size?**
2,000. The nominal sample overstates the information by a factor of 2.5.

**Item 5. Why does the domain sample size matter more than the total?**
A subgroup estimate uses only the subgroup's observations. A national sample of 5,000 can contain 80 people in the domain, and the interval reflects those 80.

**Item 6. Why is a SEER estimate not national?**
SEER covers specific registries and catchment areas. The estimate describes that coverage, not the whole country.

**Item 7. What does a GBD uncertainty interval mean?**
It reflects model and data uncertainty for a modelled estimate. It is not a sampling interval from a probability sample.

**Item 8. Why is a suppressed cell not zero?**
Suppression hides a small count to protect privacy. The true count is unknown and may be any value below the threshold.

**Item 9. What is age standardization?**
Adjusting rates to a reference age structure so that comparisons are not confounded by age. It removes age and not other differences.

**Item 10. When is a model-based small-area estimate acceptable?**
When local data are too sparse for a direct estimate, the model is documented, the assumptions are stated, and the interval is reported. It is not a substitute for local data when local data exist.

**Item 11. Why is a weighted count not an integer?**
The weights are inverse probabilities and can be fractional, so the weighted total is an estimate of a population count. Rounding it to an integer implies a precision the estimate does not have.

**Item 12. What is the difference between a stratum and a cluster?**
A stratum is a group sampled separately, often to ensure representation; a cluster is a group sampled together, often for cost. Stratification usually reduces variance, and clustering usually increases it.

**Item 13. Why can a single primary sampling unit in a stratum break variance estimation?**
The variance needs variation between units within a stratum. With one unit, that variation cannot be estimated, and software may drop the stratum or fail. The documented fix is to pool strata.

**Item 14. What is the difference between direct and indirect standardization?**
Direct standardization applies the study population's age-specific rates to a reference age structure. Indirect standardization applies reference rates to the study population and compares observed with expected. Both remove age confounding in different ways.

**Item 15. Why is a nowcast uncertain?**
Recent data are incomplete because reports arrive late. A nowcast estimates the eventual count from partial data and is revised as reports arrive. The revision range is part of the estimate.

**Item 16. Why can two surveys with the same question give different prevalence?**
Mode, sampling frame, response rate, question wording, and weighting differ. A difference is not necessarily a change in the population.

**Item 19. Why is a response rate not a measure of bias?**
A low response rate can produce little bias if non-respondents resemble respondents, and a high rate can produce bias if they do not. The relevant question is the difference between respondents and non-respondents on the outcome.

**Item 20. What is the difference between under-coverage and non-response?**
Under-coverage means the frame misses part of the target population; non-response means sampled units do not participate. Both cause bias, and they require different corrections.

**Item 21. Why is a state-level estimate from a national survey usually imprecise?**
The state domain contains only the state's respondents, which can be a few hundred people, and the domain design effect further reduces the effective sample. The national sample size does not transfer to the domain.

**Item 22. What is the difference between a revision and a revision bias?**
A revision is a later, more complete estimate. Revision bias is the systematic tendency of early estimates to be lower or higher than the final value. Reporting the revision range makes both visible.

**Item 23. Why is a national prevalence not a local market size?**
The local population differs in age, insurance, access, and risk. Multiplying a national prevalence by a local population assumes that the local population is a scaled copy of the national one, which is rarely true.

**Rubric for the design exercises (2, 3, 5, 8).** Full marks require the correct weight, the domain sample size, a design-based interval, and a prohibited-claim statement. Partial marks for a correct estimate without the design effect. No marks for an unweighted population claim.

## 14. Interview questions

**1. How do you estimate a national prevalence from NHANES?**
Use the correct component weight, the design variables, and a design-based variance method; report the interval and the design effect. A strong answer names the weight. A weak answer computes a mean.

**2. Why is BRFSS not a measured prevalence?**
It is self-reported. It misses undiagnosed disease and is affected by question and mode changes. A strong answer suggests NHANES calibration. A weak answer treats it as clinical truth.

**3. What is a design effect and why does it matter?**
The variance ratio between the complex design and SRS. It determines the effective sample size and the interval width. A strong answer computes it. A weak answer ignores it.

**4. How do you estimate a state prevalence from a national survey?**
Use the state domain, report the domain sample size and design-based interval, or use a small-area model. A strong answer does not multiply the national rate by the state population. A weak answer reports a precise state number.

**5. What can SEER tell you and what can it not?**
It can describe cancer incidence and survival in its catchment; it cannot describe national incidence or non-cancer conditions. A strong answer names the catchment. A weak answer calls it national.

**6. What is the difference between GBD and a registry?**
GBD is a modelled estimate; a registry is observed case ascertainment. A strong answer separates model from observation. A weak answer treats them as interchangeable.

**7. How do you handle a survey methodology change?**
Stratify by methodology, adjust with a bridge study, or avoid a continuous trend. A strong answer names the break. A weak answer plots a line through it.

**8. Why is a crude rate comparison misleading?**
Populations differ in age structure. A crude rate reflects both the health difference and the age difference. A strong answer standardizes. A weak answer compares raw rates.

**9. What is the ecological fallacy?**
Applying a group-level association to individuals. A county with high prevalence does not imply that a given patient has the condition. A strong answer names the level. A weak answer uses group data for individual decisions.

**10. When would you refuse a public health dataset?**
When the geography, population, or measure does not match the estimand, or when the design variables are unavailable for the claim. A strong answer treats refusal as a result. A weak answer uses the data because it is open.

**11. How do you report a modelled estimate?**
State the model, the inputs, the assumptions, the interval, and the validation. A strong answer labels it as modelled. A weak answer reports a point estimate as a count.

**12. What belongs in a public health data card?**
Source, cycle, design, weights, denominator, variance method, interval, suppression, and prohibited claims. A strong answer records the weight variable. A weak answer lists the sample size.

**13. How do you combine NHANES and BRFSS?**
Use NHANES to calibrate the measurement gap and BRFSS for state coverage, with a model that states its assumptions. A strong answer does not average the two. A weak answer pools them as if they measured the same thing.

**14. What would you ask before using a state prevalence in a grant?**
The source, the cycle, the weight, the domain sample size, the interval, and the case definition. A strong answer asks for the methods appendix. A weak answer asks for the number.

**15. How do you handle a suppressed cell in a table?**
Aggregate, suppress complementary cells, or use a model. A strong answer does not treat suppression as zero. A weak answer sums the visible cells and reports a total.

**16. Why is a registry catchment a population-fit issue?**
The registry covers a defined population, and a claim about a different population is outside its coverage. A strong answer states the catchment. A weak answer treats the registry as national.

**17. What is the difference between prevalence and incidence?**
Prevalence is the proportion with a condition at a time; incidence is the rate of new cases per person-time. A strong answer names the denominator. A weak answer uses them interchangeably.

**18. When is a modelled estimate the right deliverable?**
When local data are too sparse for a direct estimate and the model assumptions are defensible and documented. A strong answer states the validation. A weak answer presents the model as observed data.

## 15. Key terms

- **Age standardization** — adjusting for age structure.
- **Base weight** — inverse selection probability.
- **BRFSS** — Behavioral Risk Factor Surveillance System.
- **Catchment** — the population a registry serves.
- **Cluster** — a group sampled together.
- **Crude rate** — an unadjusted rate.
- **Design effect** — complex variance divided by SRS variance.
- **Domain** — a subgroup for which an estimate is made.
- **Effective sample size** — nominal size divided by design effect.
- **GBD** — Global Burden of Disease study.
- **Incidence** — new cases per person-time.
- **Model-based estimate** — an estimate produced by a model.
- **NHANES** — National Health and Nutrition Examination Survey.
- **NHIS** — National Health Interview Survey.
- **NWSS** — National Wastewater Surveillance System.
- **Oversampling** — deliberate over-representation.
- **Person-time** — time at risk contributed by a population.
- **Post-stratification** — calibration to known totals.
- **Prevalence** — proportion with a condition at a time.
- **Primary sampling unit** — the first-stage sampled unit.
- **Raking** — iterative calibration to marginal totals.
- **Rate** — events divided by person-time.
- **Replicate weight** — weights for variance estimation.
- **Reporting lag** — delay between event and report.
- **SEER** — Surveillance, Epidemiology, and End Results.
- **Small-area estimate** — an estimate for a geography with little data.
- **Standard error** — standard deviation of an estimate.
- **Stratum** — a group sampled separately.
- **Suppression** — hiding small counts for privacy.
- **Survey weight** — the population units an observation represents.
- **Under-reporting** — cases not captured.
- **Adjusted rate** — a standardized rate.
- **BRFSS** — state-based telephone survey.
- **Case definition** — criteria for counting a case.
- **Complementary suppression** — hiding additional cells to prevent back-calculation.
- **Coverage** — the share of the target population in the frame.
- **Data access request** — an application for controlled data.
- **Direct standardization** — applying study rates to a reference structure.
- **Disclosure control** — rules preventing identification.
- **Domain design effect** — the design effect within a subgroup.
- **Ecological fallacy** — applying group findings to individuals.
- **Finite population correction** — adjustment for sampling without replacement.
- **Frame** — the list from which a sample is drawn.
- **Hot-deck imputation** — filling missing values from similar records.
- **Incidence rate** — new cases per person-time.
- **Indirect standardization** — comparing observed with expected using reference rates.
- **Kish effective sample** — sample size adjusted for unequal weights.
- **Mode effect** — measurement difference by survey mode.
- **Multiple imputation** — repeated imputation for uncertainty.
- **Non-response bias** — bias from differential response.
- **Notifiable disease** — a disease required to be reported.
- **Nowcast** — an estimate of incomplete recent data.
- **Population estimate** — the denominator population.
- **Post-stratification weight** — calibration to known totals.
- **Prevalence ratio** — ratio of two prevalences.
- **Raking weight** — calibrated weight from marginal totals.
- **Registry** — a system collecting cases in a population.
- **Response rate** — the share of sampled units responding.
- **Sample design** — the plan for selecting units.
- **Sampling error** — error from observing a sample.
- **Self-report** — information provided by the respondent.
- **Small-cell suppression** — hiding counts below a threshold.
- **Standard population** — the reference age structure.
- **Survey cycle** — the data collection period.
- **Syndromic surveillance** — monitoring symptoms and codes.
- **Target population** — the population described by the estimate.
- **Telephone coverage** — the share reachable by telephone.
- **Under-coverage** — groups missing from the frame.
- **Variance estimation** — estimating the sampling variance.
- **Weight calibration** — adjusting weights to known totals.
- **Weight trimming** — capping extreme weights.
- **Yield** — the number of usable responses.
- **Zero-dose** — no doses of a vaccine received.
- **ZIP code** — a geographic unit, not a sampling unit.
- **Age-specific rate** — a rate within an age band.
- **All-cause mortality** — deaths from any cause.
- **Cause-specific mortality** — deaths from a named cause.
- **Census tract** — a small geographic unit.
- **Coverage error** — error from incomplete frames.
- **Denominator** — the population or person-time at risk.
- **Epidemiologic transition** — change in disease patterns over time.
- **Excess mortality** — deaths above an expected baseline.
- **Health expectancy** — expected years in good health.
- **Life expectancy** — expected years of life.
- **Mortality rate** — deaths per population per time.
- **Natality** — birth statistics.
- **Population attributable fraction** — share of disease due to an exposure.
- **Prevalence odds ratio** — odds ratio from a prevalence study.
- **Proxy report** — information from a surrogate.
- **Sampling variance** — variance from sampling.
- **Sensitivity analysis** — testing assumptions.
- **Standard deviation** — dispersion of a distribution.
- **Survey mode** — how the survey is administered.
- **Weighted mean** — a mean using weights.
- **Zero-inflated** — a distribution with many zeros.
- **Backfill** — late addition of reports.
- **Revision** — a changed estimate.
- **Vintage** — the release version of a population estimate.
- **Allostatic load** — cumulative physiological wear.
- **Disability-adjusted life year** — years lost to disability or death.
- **Quality-adjusted life year** — years weighted by health utility.
- **Years of life lost** — premature mortality burden.
- **Years lived with disability** — morbidity burden.
- **Risk factor** — an exposure associated with an outcome.
- **Attributable risk** — excess risk from an exposure.
- **Population attributable risk** — excess risk in the population.
- **Sensitivity** — the probability a case is detected.
- **Specificity** — the probability a non-case is excluded.
- **Positive predictive value** — the probability a positive is true.
- **Negative predictive value** — the probability a negative is true.
- **Likelihood ratio** — how much a test changes the odds.
- **Pre-test probability** — the probability before a test.
- **Post-test probability** — the probability after a test.
- **Screening** — testing asymptomatic people.
- **Surveillance** — ongoing systematic data collection.
- **Outbreak** — an excess of cases above expectation.
- **Epidemic** — an outbreak across a wider area.
- **Pandemic** — a global epidemic.
- **Endemic** — a usual level of disease.
- **Herd immunity** — indirect protection from population immunity.
- **Basic reproduction number** — secondary cases per index case.
- **Case fatality ratio** — deaths among diagnosed cases.
- **Infection fatality ratio** — deaths among infected people.
- **Attack rate** — cases among exposed people.
- **Secondary attack rate** — cases among contacts.
- **Generation time** — time between successive infections.
- **Doubling time** — time for cases to double.
- **Excess deaths** — deaths above a baseline.
- **Verification bias** — testing depends on the result of another test.
- **Lead-time bias** — earlier detection without longer survival.
- **Length bias** — slower-growing cases over-represented in screening.

## 16. Sources

All URLs accessed 26 October 2026. Cycles, weights, and definitions change; verify at the source.

1. Centers for Disease Control and Prevention. NHANES. https://www.cdc.gov/nchs/nhanes/
2. Centers for Disease Control and Prevention. NHANES analytic guidelines. https://www.cdc.gov/nchs/nhanes/analyticguidelines.htm
3. Centers for Disease Control and Prevention. BRFSS. https://www.cdc.gov/brfss/
4. Centers for Disease Control and Prevention. NHIS. https://www.cdc.gov/nchs/nhis/
5. National Cancer Institute. SEER. https://seer.cancer.gov/
6. Centers for Disease Control and Prevention. CDC WONDER. https://wonder.cdc.gov/
7. Centers for Disease Control and Prevention. National Wastewater Surveillance System. https://www.cdc.gov/wastewater/
8. Institute for Health Metrics and Evaluation. GBD. https://www.healthdata.org/gbd
9. World Health Organization. Global Health Observatory. https://www.who.int/data/gho
10. Office of Management and Budget. Race and ethnicity standards. https://www.whitehouse.gov/omb/
11. US Department of Health and Human Services. HIPAA. https://www.hhs.gov/hipaa/
12. STROBE. https://www.strobe-statement.org/
13. RECORD. https://www.record-statement.org/
14. GATHER. https://www.gather-statement.org/
15. Centers for Disease Control and Prevention. BRFSS questionnaires and methodology. https://www.cdc.gov/brfss/questionnaires/
16. National Cancer Institute. SEER data access. https://seer.cancer.gov/data-software/
17. Institute for Health Metrics and Evaluation. GBD protocol. https://www.healthdata.org/gbd/about/protocol
18. US Census Bureau. Population estimates. https://www.census.gov/programs-surveys/popest.html
19. Module 21 survey file, section 11. [`../../course/21_datasets_tools_benchmarks.md`](../../course/21_datasets_tools_benchmarks.md)

## 17. Links

- Module map and topic table: [`_index.md`](_index.md)
- Previous topic, text and LLM benchmarks: [`08_text-and-llm-benchmarks.md`](08_text-and-llm-benchmarks.md)
- Pathology datasets: [`07_pathology-datasets.md`](07_pathology-datasets.md)
- Access and licensing: [`02_access-and-licensing.md`](02_access-and-licensing.md)
- Root survey file: [`../../course/21_datasets_tools_benchmarks.md`](../../course/21_datasets_tools_benchmarks.md)
- Labs, including the dataset audit: [`../../labs/`](../../labs/)

