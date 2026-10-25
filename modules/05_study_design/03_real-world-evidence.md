# Real-World Evidence

> Module 05 — Study Design and Evidence Generation
> Topic 03 of 10
> Estimated study time: 4 hours
> Prerequisites: Topics 01 and 02
> Version 2026.3

## 1. Why this topic exists

Real-world evidence (RWE) is evidence derived from real-world data (RWD): data collected outside a conventional randomised trial, such as electronic health records (EHR), insurance claims, registries, death records, pharmacy dispensing, wearable sensors, and patient-reported outcomes. The promise is obvious. These data describe what happens to patients in routine care, at a scale and duration that trials rarely achieve. The risk is equally obvious. They were collected for billing, administration, or care delivery, not for research, and the causal question must be imposed on them after the fact.

The commercial importance of RWE has grown because three decision-makers now ask for it. Regulators accept it for selected label changes, post-marketing requirements, and device submissions. Payers use it to assess comparative effectiveness and budget impact. Health systems use it to decide whether a product works in their own population. A company without an RWE capability is limited to trial populations and trial time horizons, which are often narrower than the market.

The methodological danger is that RWE is easy to produce and hard to make credible. A database query can generate a statistically significant association in an afternoon. Whether that association reflects a causal effect depends on design choices that are invisible in the result: who was eligible, when follow-up began, how exposure was defined, how confounding was handled, and whether the analysis was prespecified. The gap between a database result and credible evidence is the subject of this topic.

The US Food and Drug Administration (FDA) has made the criteria explicit. Its device guidance of December 2025 describes relevance and reliability assessments for RWD, and its drug framework guidance of August 2023 describes the same logic for drugs and biologics. The European Medicines Agency (EMA) uses the same broad concepts in scientific advice and in its qualification pathway. The common structure is that data fitness is question-specific: the same database can be fit for one question and unfit for another.

This topic teaches you to design an RWE study that a regulator, payer, or health system can accept, and to recognise when the data cannot support the claim at all.

### What makes RWE different from a trial

A randomised trial creates the comparison. An RWE study discovers it, or fails to. The difference has four consequences.

1. **Confounding is not controlled by design.** Treatment assignment in routine care reflects clinical judgement, access, preference, and cost. Patients who receive a drug are often sicker, or healthier, than those who do not. The analyst must model the assignment mechanism, and the model may be wrong.
2. **Time zero is a choice.** In a trial, randomisation defines the start of follow-up. In RWD, the analyst chooses the index date, and different choices produce different cohorts and different effects. Immortal time, prevalent-user bias, and depletion of susceptible patients all enter through this choice.
3. **Exposure is measured imperfectly.** A prescription is not a dispensing, a dispensing is not ingestion, and a diagnosis code is not a disease. Measurement error can bias the effect in either direction, and it is rarely random.
4. **The data are reused.** The same database supports many questions, and the analysis is often one of many that could have been run. Without prespecification, the result is vulnerable to selection of the most favourable analysis.

These differences do not make RWE inferior in every case. A well-designed target trial emulation can be more informative than a small, unrepresentative trial. But the burden of proof shifts from the design to the analysis, and the analysis must be transparent enough for a reviewer to test it.

### The decision context

RWE is used for four broad decisions:

- **Regulatory:** a new indication, a label change, a post-marketing requirement, or a device premarket submission.
- **Payer:** coverage, formulary placement, prior authorisation criteria, or a coverage-with-evidence-development arrangement.
- **Health system:** procurement, pathway design, or quality improvement.
- **Clinical:** guideline development, treatment sequencing, or shared decision-making.

Each decision has a different tolerance for uncertainty and a different requirement for data fitness. A regulator may accept an RWE study for a rare disease with no feasible trial, but require a prespecified protocol and a sensitivity analysis for unmeasured confounding. A payer may accept a claims-based comparative effectiveness study, but require a cost-effectiveness model. A health system may accept a local EHR analysis, but require that the result be reproducible in its own data.

The design should name the decision and the decision-maker before the database is opened. This is the same principle as Topic 01, applied to observational data.

## 2. Learning outcomes

By the end of this topic you can:

1. **Assess** whether a real-world data source is fit for a specific regulatory, payer, or clinical question, using the relevance and reliability dimensions. (Exercise 1, self-check 1)
2. **Specify** a target trial with seven components before analysing observational data, and explain how each component maps to an analysis choice. (Exercise 2, self-check 2)
3. **Define** time zero, eligibility, exposure, comparator, and follow-up in a way that avoids immortal time and prevalent-user bias. (Exercise 3, self-check 3)
4. **Choose** a confounding-adjustment method (stratification, propensity score, inverse probability weighting, g-methods) and justify it from the structure of the question. (Exercise 4, interview 3)
5. **Diagnose** positivity violations, measurement error, and missing data, and propose a design or analysis response. (Exercise 5, self-check 5)
6. **Quantify** the sensitivity of an RWE result to unmeasured confounding using an E-value or a quantitative bias analysis. (Exercise 6, self-check 6)
7. **Write** a prespecified RWE protocol and a regulatory pre-submission question that a reviewer can evaluate. (Exercises 7 and 8, interview 8)

The outcomes are cumulative. Outcome 6 assumes outcome 4, and outcome 7 assumes all the preceding outcomes.

## 3. The core mental model

The model is a target trial imposed on observational data. The target trial is the randomised experiment that would answer the question if it were feasible. The emulation is the observational study designed to approximate it. The gap between the two is the assumption ledger.

```text
                    DECISION AND QUESTION
                              |
                              v
                    TARGET TRIAL PROTOCOL
   eligibility | strategies | assignment | follow-up |
   outcome | causal contrast | analysis plan
                              |
                              v
                    DATA FITNESS ASSESSMENT
   relevance: population, exposure, outcome, setting
   reliability: accuracy, completeness, provenance, quality
                              |
                              v
                    EMULATION DESIGN
   time zero | eligibility | exposure | comparator |
   follow-up | outcome | censoring
                              |
                              v
                    CAUSAL ANALYSIS
   confounding | positivity | measurement | missingness
                              |
                              v
                    SENSITIVITY AND BIAS ANALYSIS
   negative controls | E-value | quantitative bias analysis
                              |
                              v
                    DECISION
        support | conditional support | not supportable
```

The vertical order is not optional. A target trial specified after the analysis is a rationalisation, not a design. A data fitness assessment performed after the result is a justification, not an assessment. The model requires the question, the target trial, and the fitness assessment to precede the analysis.

### The seven components of a target trial

Hernán and Robins (2016) specify seven components. Each has a direct analogue in the observational analysis.

| Component | Target trial question | Observational analogue |
|---|---|---|
| Eligibility criteria | Who could be randomised? | Who enters the cohort, and when? |
| Treatment strategies | What are the strategies, including start and stop rules? | How are exposure and comparator defined? |
| Assignment | How is treatment assigned? | How is the observed assignment modelled? |
| Follow-up | When does follow-up start and end? | What is time zero, and what censors follow-up? |
| Outcome | What is measured, how, and when? | Which coded variable, with what validity? |
| Causal contrast | Intention-to-treat or per-protocol? | Which estimand, and which g-method? |
| Analysis plan | How will the effect be estimated? | Which model, weights, and sensitivity analyses? |

The table is the bridge between the design vocabulary of Topic 02 and the data vocabulary of RWE. A study that cannot fill in the right-hand column for every row is not ready to be analysed.

### Time zero and the three biases it controls

Time zero is the moment a participant becomes eligible and is classified into a treatment strategy. It is the single most consequential choice in an RWE study, because it determines which patients are compared and which person-time is counted.

Three biases enter through time zero:

**Immortal time bias.** If exposure is defined by an event that occurs after cohort entry, the exposed group must survive until that event. The period before exposure is "immortal" because the outcome could not have occurred in a person classified as exposed. Counting that period as exposed person-time, or excluding early deaths from the exposed group, makes the exposure look protective.

**Prevalent-user bias.** If the cohort includes people who have already been taking the drug for some time, it excludes those who stopped early, often because of side effects or lack of benefit. The prevalent users are survivors of the initial treatment period, and they are healthier or more tolerant than incident users. The result is a biased estimate of the effect of starting treatment.

**Depletion of susceptibles.** If the drug removes people at high risk from the cohort, the remaining exposed population is lower-risk than the comparison group. The effect appears to weaken over time, even if the drug is effective.

The design response to all three is the same: define a new-user, active-comparator cohort with a single time zero at the start of treatment, and align the comparison group's time zero to the same moment.

### Confounding, positivity, and exchangeability

A causal effect is identifiable if the treated and comparison groups would have had the same outcome distribution in the absence of treatment, conditional on measured covariates. This is the exchangeability assumption. It cannot be tested directly, because the counterfactual is unobserved. It can be supported by design and probed by sensitivity analysis.

Three conditions must hold:

1. **Exchangeability:** no unmeasured confounding, conditional on covariates.
2. **Positivity:** every covariate combination has a nonzero probability of receiving each treatment.
3. **Consistency:** the observed outcome under the received treatment equals the potential outcome under that treatment.

Positivity is testable. A propensity score distribution that piles up near 0 or 1 indicates that some patients have essentially no chance of receiving one of the treatments. In that region, the comparison is extrapolation. The design response is to restrict the population, trim the weights, or redefine the question.

Exchangeability is not testable. The design response is to measure the important confounders, use a new-user active-comparator design, and report an E-value or a bias analysis. A study that claims exchangeability without either is asserting, not demonstrating.

### The data fitness frame

The FDA's RWE guidance organizes data assessment around relevance and reliability. The frame is useful because it separates two questions that are often conflated.

**Relevance** asks whether the data contain the right people, exposures, outcomes, and time period for the question. A claims database with no laboratory results may be relevant for a study of a drug's association with a coded outcome, but irrelevant for a study that requires baseline renal function.

**Reliability** asks whether the data are accurate, complete, and traceable. A registry with voluntary reporting may have good clinical detail but incomplete capture. An EHR may have complete capture within a health system but miss care delivered elsewhere. A claims database may capture all billed care but miscode diagnoses.

The two dimensions are independent. Data can be relevant and unreliable (a registry with the right variables but poor quality control), or reliable and irrelevant (a high-quality database that does not contain the population or outcome).

### The assumption ledger

Every RWE study carries assumptions that cannot be verified from the data. The protocol should list them, state their direction, and describe the sensitivity analysis for each.

| Assumption | Direction if violated | Sensitivity analysis |
|---|---|---|
| No unmeasured confounding | Effect biased toward or away from null | E-value, negative control, bias analysis |
| Positivity | Comparison is extrapolation | Propensity overlap plot, trimming |
| Correct time zero | Immortal time or prevalent-user bias | Alternative index dates |
| Accurate exposure | Misclassification, usually toward null | Active comparator, dose-response |
| Accurate outcome | Misclassification, usually toward null | Validation substudy, positive control |
| Complete follow-up | Informative censoring | Negative control, competing risks |
| Stable standard of care | Secular trend bias | Calendar-time adjustment, era restriction |

The ledger is the honest core of an RWE protocol. A study without one is asking the reader to trust the analyst rather than the design.

### Common data sources and their trade-offs

| Source | Strengths | Weaknesses | Best suited to |
|---|---|---|---|
| Claims | Complete billing capture; large; longitudinal | No clinical detail; coding variation; no lab values | Utilisation, cost, coded outcomes |
| EHR | Clinical detail; labs; vitals; notes | Fragmented across providers; missing outside care | Clinical outcomes, confounders |
| Registry | Disease-specific detail; protocolised data | Voluntary; incomplete; selection | Natural history, device performance |
| Death records | Complete mortality; dates | Cause of death often unreliable | All-cause mortality |
| Pharmacy dispensing | Actual dispensing; dose; days supplied | No ingestion; no clinical outcome | Exposure definition |
| Wearables | Continuous physiological data | Measurement error; adherence; missingness | Digital endpoints, monitoring |
| Patient-reported outcomes | Patient perspective; symptoms | Recall bias; selection; missingness | Quality of life, symptom burden |
| Linkage of claims and EHR | Combines utilisation and clinical detail | Linkage error; coverage gaps | Comparative effectiveness |

The table is a reminder that no single source is fit for every question. A study that needs baseline renal function, drug exposure, and stroke outcomes may require a linked claims-EHR-registry dataset. A study that needs only mortality may be well served by a death registry linked to a claims cohort.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Real-world data (RWD) | Data collected outside a conventional trial, for care or administration | Used as a synonym for "big data" | Data category |
| Real-world evidence (RWE) | Evidence derived from RWD through a defined design and analysis | Used as a synonym for any database study | Evidence category |
| Target trial | The randomised experiment that would answer the question if feasible | Confused with the observational study itself | Protocol |
| Target trial emulation | An observational analysis designed to approximate a target trial | Any retrospective cohort study | Design |
| Time zero | The moment eligibility, exposure, and follow-up begin | Set to the first data availability date | Date |
| Immortal time | Person-time during which the outcome could not have occurred by definition | Ignored or counted as exposed | Person-time |
| Prevalent-user bias | Bias from including people already on treatment at cohort entry | Confused with confounding | Bias category |
| New-user design | Cohort entry at the start of treatment | Used without an active comparator | Design feature |
| Active comparator | A comparator treatment that is used for the same indication | Confused with any comparator | Design feature |
| Positivity | Every covariate pattern has nonzero probability of each treatment | Assumed rather than checked | Assumption |
| Exchangeability | No unmeasured confounding conditional on covariates | Treated as testable | Assumption |
| Consistency | The observed outcome under received treatment equals the potential outcome | Assumed without discussion | Assumption |
| Confounding by indication | Treatment assignment depends on prognosis | Confused with selection bias | Bias category |
| Propensity score | The probability of treatment given covariates | Used without checking overlap | Probability |
| Inverse probability weighting (IPW) | Weighting by the inverse of the treatment probability | Weights not trimmed or checked | Method |
| Standardisation | Weighting stratum-specific estimates to a target population | Confused with regression adjustment | Method |
| g-methods | A family of methods for time-varying treatments and confounding | Used as a synonym for IPW | Method family |
| Marginal structural model | A model fitted with IPW to estimate a marginal effect | Confused with a conditional model | Model |
| Cloning | Creating copies of participants under each strategy | Confused with matching | Method |
| Grace period | A window after time zero during which treatment initiation is allowed | Omitted, creating immortal time | Duration |
| Negative control outcome | An outcome not plausibly affected by the exposure | Used as proof of no confounding | Diagnostic |
| Positive control outcome | An outcome known to be affected by the exposure | Used as proof of data validity | Diagnostic |
| E-value | The minimum strength of unmeasured confounding needed to explain away the result | Treated as a p-value | Ratio |
| Quantitative bias analysis | A model-based correction for a specified bias | Confused with sensitivity analysis | Method |
| Claims data | Billing records from insurers or payers | Assumed to capture clinical detail | Data source |
| Electronic health record (EHR) | Clinical documentation and order data from a care provider | Assumed to capture care outside the system | Data source |
| Registry | A structured collection of data on a defined population | Assumed to be complete | Data source |
| Data provenance | The documented origin and processing history of a data element | Ignored in favour of the value | Metadata |
| Data lineage | The traceable path from source to analysis variable | Confused with provenance | Metadata |
| Fitness for purpose | The match between data characteristics and the question | Assumed to be global | Assessment |
| Coverage with evidence development (CED) | Payer coverage conditioned on further evidence | Treated as approval | Payer mechanism |
| Common data model | A standardised schema for observational data, such as OMOP | Confused with a data quality standard | Schema |

Two distinctions deserve emphasis.

**Confounding versus selection bias.** Confounding is a common cause of both treatment and outcome. Selection bias is a consequence of how participants enter or remain in the analysis. A new-user design addresses selection bias from prevalent use; it does not address confounding by indication. Both must be handled, and they require different diagnostics.

**Negative control versus positive control.** A negative control outcome should show no association with the exposure if confounding is absent. An observed association suggests residual confounding. A positive control outcome should show the expected association if the data are valid. A null association suggests measurement or capture problems. The two diagnostics answer different questions and should be used together.

## 5. The mechanics, step by step

The procedure has fourteen steps. Steps 1 to 3 define the question and assess the data. Steps 4 to 7 build the cohort and the comparison. Steps 8 to 11 address the causal threats. Steps 12 to 14 quantify uncertainty, prespecify, and report.

### Procedure at a glance

| Step | Input | Output | Failure if skipped |
|---:|---|---|---|
| 1 | Decision and claim | Five-attribute estimand | Analysis without a question |
| 2 | Estimand | Seven-component target trial | Design choices made implicitly |
| 3 | Target trial and data | Fitness verdict | Wrong database for the question |
| 4 | Eligibility and data | Cohort with one time zero | Immortal time or prevalent users |
| 5 | Strategies | Exposure and active comparator | Confounding by indication |
| 6 | Outcome and follow-up | Outcome definition and censoring rule | Measurement bias |
| 7 | Confounders and sample | Adjustment method and diagnostics | Residual confounding |
| 8 | Cohort and exposure | Time-aligned analysis | Spurious protective effect |
| 9 | Propensity score | Overlap assessment and restriction | Extrapolation |
| 10 | Variable definitions | Measurement and missing-data plan | Bias of unknown direction |
| 11 | Assumption ledger | Sensitivity analyses | Unquantified fragility |
| 12 | Event rate and effect | Precision and effective sample | Underpowered claim |
| 13 | All of the above | Registered protocol and SAP | Selective analysis |
| 14 | Results and diagnostics | Decision-ready report | Misleading conclusion |

The table is also a review checklist. Steps 3, 4, and 8 are the ones most often skipped in practice, and they are the ones that most often determine whether the result is credible.

### Step 1: State the decision and the estimand

**Input:** the decision, the decision-maker, and the claim.

**Output:** an estimand with the five ICH E9(R1) attributes, written before the database is queried.

**Rationale:** The estimand determines the eligible population, the treatment contrast, the outcome, the handling of intercurrent events, and the summary measure. Without it, the analysis becomes a search for an association.

For RWE, two attributes require particular care.

- **Treatment condition:** the strategies must include start, stop, and switching rules. "Patients on drug X" is not a strategy; "initiate drug X within 30 days of the index date and continue until discontinuation, switch, or death" is.
- **Intercurrent events:** discontinuation, switching, and rescue therapy are common. The estimand must state whether the effect is treatment policy (the effect of initiating the strategy, regardless of later changes) or hypothetical (the effect if everyone adhered). Most regulatory RWE questions use a treatment-policy or a per-protocol analogue.

### Step 2: Specify the target trial

**Input:** the estimand and the clinical question.

**Output:** a seven-component target trial table.

**Rationale:** The target trial makes the design choices explicit and testable. It is the reference against which the emulation is judged.

| Component | Specification for the example |
|---|---|
| Eligibility | Adults aged 65 or older with incident atrial fibrillation, no prior anticoagulant, no severe valve disease |
| Strategies | Initiate a direct oral anticoagulant (DOAC) versus initiate warfarin |
| Assignment | By clinical decision, observed in the data |
| Follow-up | From initiation to stroke, death, disenrolment, or 365 days |
| Outcome | Ischaemic stroke, ascertained from hospital diagnosis codes |
| Contrast | Treatment-policy risk difference at 365 days |
| Analysis | IPW with a propensity score, with a per-protocol sensitivity analysis |

A target trial that cannot be filled in is a sign that the question is not yet specified. A target trial that can be filled in but does not match the data is a sign that the data are not fit for the question.

### Step 3: Assess data relevance and reliability

**Input:** the target trial and the available data sources.

**Output:** a fitness assessment with a verdict for each dimension, and a list of gaps.

**Rationale:** Data fitness is question-specific. The assessment should be documented before the analysis and should state what the data cannot support.

| Dimension | Questions | Evidence required |
|---|---|---|
| Relevance: population | Does the database contain the eligible population, including exclusions? | Counts by age, sex, diagnosis, and setting |
| Relevance: exposure | Are the drugs or devices identifiable, with dose and duration? | Coding dictionary, dispensing records |
| Relevance: outcome | Is the outcome captured with acceptable sensitivity and specificity? | Validation study or positive control |
| Relevance: setting | Does the data reflect the care setting and era of the question? | Coverage dates, site types |
| Reliability: accuracy | Are values plausible and internally consistent? | Range checks, duplicate checks |
| Reliability: completeness | Are key variables missing, and at what rate? | Missingness by variable and site |
| Reliability: provenance | Can each value be traced to a source? | Data lineage documentation |
| Reliability: quality control | Were data entry and coding audited? | Quality reports, audit logs |

The verdict should be one of three: fit, fit with limitations, or unfit. "Fit with limitations" is the most common and requires the limitations to be carried into the sensitivity analyses.

### Step 4: Define the cohort, eligibility, and time zero

**Input:** the target trial eligibility criteria and the data.

**Output:** a cohort definition with a single time zero and a flow diagram.

**Rationale:** Time zero must be the moment at which eligibility is met and treatment is initiated. If time zero is set to the first record in the database, or to the first diagnosis, the cohort includes person-time before treatment and creates immortal time.

A new-user, active-comparator design has three requirements:

1. No prior use of the study drugs before time zero.
2. A measurable baseline period before time zero for covariate assessment.
3. The same time zero definition for both strategies.

If a grace period is used to allow treatment initiation after eligibility, it must be prespecified, short, and applied symmetrically. A grace period that is longer for one strategy creates differential immortal time.

### Step 5: Define exposure and comparator

**Input:** the target trial strategies and the data.

**Output:** an exposure definition with a start rule, a stop rule, and a switching rule; and a comparator definition that is used for the same indication.

**Rationale:** The comparator determines the counterfactual. An active comparator is preferable to a non-user comparator because it reduces confounding by indication: both groups have a reason to be treated.

Exposure definitions should be based on dispensing or administration, not on prescribing alone, where possible. If only prescriptions are available, the definition should state the assumption that a prescription corresponds to use, and the sensitivity analysis should test alternative definitions.

### Step 6: Define the outcome, follow-up, and censoring

**Input:** the estimand and the data.

**Output:** an outcome definition with codes, a follow-up rule, and a censoring rule.

**Rationale:** The outcome must be measured consistently across strategies and over time. Coding changes, reimbursement changes, and shifts in diagnostic practice can create artefacts that look like effects.

Follow-up ends at the outcome, death, disenrolment, or the administrative end of data. Disenrolment is informative if it is related to the outcome; the analysis should treat it as censoring and use IPW for censoring if the assumption is plausible.

### Step 7: Choose the confounding-adjustment method

**Input:** the measured confounders, the sample size, and the estimand.

**Output:** a primary method, a diagnostic plan, and a sensitivity analysis.

**Rationale:** The method must match the structure of the question. A time-fixed exposure with a single time zero can use a propensity score or standardisation. A time-varying exposure requires a marginal structural model or another g-method.

| Method | When it is appropriate | Main limitation |
|---|---|---|
| Stratification | Few categorical confounders, large sample | Sparse strata; difficult with many confounders |
| Regression adjustment | Continuous or many confounders | Model misspecification; conditional vs marginal effect |
| Propensity score matching | Moderate sample, many confounders | Loss of sample; balance not guaranteed |
| Propensity score weighting | Large sample, interest in marginal effect | Extreme weights; positivity violations |
| Standardisation | Interest in a target population | Requires a standard population |
| Doubly robust estimation | Want protection against one model being wrong | Requires both models to be wrong to fail |
| Marginal structural model | Time-varying treatment and confounding | Requires correct weights; complex |
| Instrumental variable | Unmeasured confounding, valid instrument | Instrument validity is untestable |

The choice should be prespecified, and the diagnostic plan should include balance checks, propensity overlap, and a sensitivity analysis with an alternative method.

### Step 8: Handle immortal time and time-varying exposure

**Input:** the cohort and the exposure definition.

**Output:** a time-zero-aligned analysis with no pre-exposure person-time counted as exposed.

**Rationale:** Immortal time is the most common avoidable bias in RWE. It arises whenever exposure status is defined by an event that occurs after follow-up begins.

The design responses are:

- Align time zero to treatment initiation for both strategies.
- Use a new-user design.
- If a grace period is necessary, apply it symmetrically and censor at the end of the grace period.
- For time-varying treatment, use a marginal structural model with time-varying weights.
- Never count pre-initiation person-time as exposed.

A diagnostic is to compute the rate ratio under a null effect with a delayed exposure definition. If the design produces a protective association when the true effect is null, the design has immortal time.

### Step 9: Check positivity and overlap

**Input:** the propensity score or the covariate distribution.

**Output:** an overlap plot, a summary of the common support region, and a decision on trimming or restriction.

**Rationale:** A causal contrast is only defined where both treatments are possible. If some patients have a propensity score near 0 or 1, the comparison is extrapolation.

The response to positivity violations is design, not wishful thinking. Options include restricting the population to the common support, trimming extreme weights, redefining the comparator, or changing the question. A study that reports an effect without checking overlap is reporting an extrapolation.

### Step 10: Address measurement error and missing data

**Input:** the exposure, outcome, and covariate definitions.

**Output:** a measurement-error assessment and a missing-data plan.

**Rationale:** RWD are collected for purposes other than research, and the measurement properties are often unknown. A diagnosis code may have a positive predictive value of 0.7 in one setting and 0.95 in another. A laboratory value may be missing because it was not ordered, which is itself informative.

The plan should state:

- the expected sensitivity and specificity of the outcome definition,
- the expected missingness for each key covariate,
- the assumed missingness mechanism,
- the primary handling method,
- the sensitivity analysis under a plausible departure.

### Step 11: Plan the sensitivity and bias analyses

**Input:** the assumption ledger.

**Output:** a prespecified set of sensitivity analyses, each tied to an assumption.

**Rationale:** RWE cannot prove the absence of confounding. It can quantify how strong a confounder would have to be to change the conclusion.

The core analyses are:

- **E-value:** the minimum strength of association that an unmeasured confounder would need with both treatment and outcome to explain away the observed effect.
- **Negative control outcome:** an outcome that should not be affected by the treatment; an association suggests residual confounding.
- **Positive control outcome:** an outcome that should be affected; a null association suggests measurement problems.
- **Quantitative bias analysis:** a model-based correction for a specified bias, such as outcome misclassification.
- **Alternative exposure or time-zero definitions:** to test the sensitivity of the result to design choices.

### Step 12: Estimate precision and sample size

**Input:** the expected event rate, the effect size, and the variance.

**Output:** a precision calculation or a power calculation, with the effective sample size after weighting.

**Rationale:** In RWE, the number of records is not the effective sample size. Weighting, matching, and restriction reduce the effective sample. A study with 100,000 records and extreme weights may have the precision of a study with 10,000.

The calculation should report the effective sample size and the expected number of events. For a risk difference, the variance can be approximated from the weighted risks; for a hazard ratio, the number of events drives precision.

### Step 13: Prespecify and register

**Input:** all of the above.

**Output:** a protocol and a statistical analysis plan, registered or published before the analysis.

**Rationale:** Prespecification is the main defence against selective analysis. In RWE, the analyst has many degrees of freedom: cohort definitions, exposure windows, covariate sets, model specifications, and subgroup analyses. Without a protocol, the result is one of many possible results.

The protocol should include the target trial, the data fitness assessment, the cohort definition, the estimand, the primary analysis, the sensitivity analyses, and the decision rule. It should be registered in a public repository or submitted to the regulator before the analysis.

### Step 14: Report and engage the decision-maker

**Input:** the results, the diagnostics, and the assumption ledger.

**Output:** a report that states what the data support, what they do not, and what would change the conclusion.

**Rationale:** The report must be usable by a reviewer who did not design the study. It should include the flow diagram, the baseline table, the propensity overlap, the primary estimate with a confidence interval, the sensitivity analyses, and the E-value.

The report should also state the residual uncertainty in plain language. "The data are consistent with a 10 to 25 percent relative reduction in stroke, but an unmeasured confounder of moderate strength could explain the result" is more useful than a p-value.

### Diagnostics by causal threat

Each threat has at least one diagnostic and one design response. The table is the minimum set for a confirmatory RWE study.

| Threat | Diagnostic | Design response | What a positive diagnostic means |
|---|---|---|---|
| Confounding by indication | Baseline table before weighting | Active comparator; IPW | Imbalance before adjustment |
| Unmeasured confounding | Negative control outcome; E-value | Restriction; bias analysis | Result is fragile |
| Immortal time | Null-cohort check; timeline audit | New-user design; time-zero alignment | Spurious benefit |
| Prevalent-user bias | Prior-exposure query | 12-month washout | Healthy-user effect |
| Positivity violation | Propensity overlap plot | Trimming; restriction | Extrapolation |
| Outcome misclassification | Positive control; validation substudy | Validated outcome; bias analysis | Measurement problem |
| Exposure misclassification | Dispensing versus prescription comparison | Dispensing data; dose-response | Attenuation |
| Informative censoring | Censoring rates by arm | IPW for censoring; external linkage | Selection bias |
| Secular trend | Outcome rate by calendar period | Calendar adjustment; era restriction | Time confounding |
| Selective analysis | Protocol versus analysis audit | Registration; SAP lock | Reporting bias |

The diagnostics are not optional appendices. A study that reports an adjusted estimate without the corresponding diagnostic is asking the reviewer to accept the assumption on trust.

### Choosing the grace period

A grace period allows treatment initiation after the eligibility date. It is useful when there is a clinically appropriate delay between diagnosis and treatment. It is dangerous when it is applied asymmetrically or when it is long relative to the outcome risk.

The rules are:

- Prespecify the length and justify it clinically.
- Apply it symmetrically to both strategies.
- Censor follow-up at the end of the grace period for participants who do not initiate.
- Run a sensitivity analysis with no grace period and with a shorter one.
- Report the proportion of participants who initiate within the grace period.

A grace period that is longer for the exposed strategy creates immortal time. A grace period that is longer for the comparator creates the opposite bias. Symmetry is the minimum requirement.

### Minimum data checks before analysis

Before the primary analysis, run the following checks and record the result of each:

- Count of eligible patients at each exclusion step, with reasons.
- Distribution of time from diagnosis to time zero, by strategy.
- Proportion with prior exposure in the washout window.
- Missingness for every covariate, by strategy and site.
- Propensity score distribution and overlap, by strategy.
- Event counts by strategy, calendar period, and site.
- Censoring rates by strategy and reason.
- Duplicate and unmatched linkage records.
- Out-of-range values for age, dates, and doses.
- Code frequency for the outcome, by year and site.

A study that skips these checks may still produce a number, but it cannot explain the number. The checks are cheap, and they often reveal a design problem before the analysis is locked.

### Edge cases

**External control drift.** When RWE supplies the control arm for a single-arm trial, the comparison depends on the comparability of populations, eras, and care pathways. Drift in standard of care, diagnostic criteria, or supportive treatment can bias the estimate. The design should include a concurrent control where possible, a prespecified adjustment for era, and a sensitivity analysis that quantifies the effect of plausible drift.

**Missing endpoint data.** In RWD, the outcome may be missing because the patient received care elsewhere. The missingness is often related to the outcome, which makes complete-case analysis biased. The design should link to external data sources where possible, and the analysis should use a missing-data model with a sensitivity analysis.

**Non-inferiority claims.** An RWE non-inferiority claim requires an active comparator, a justified margin, and assay sensitivity. The observational data must be able to detect the comparator's historical effect, which is often difficult. A non-inferiority claim from RWE without a positive control is weak.

**Multiple treatments.** A study with several comparators or several outcomes has a multiplicity problem. The protocol should prespecify the primary comparison and label the rest as supportive, or use a multiplicity adjustment.

**Adaptive or sequential analyses.** RWE studies are sometimes updated as new data arrive. Repeated analyses without alpha control inflate the false-positive rate. The protocol should state whether the analysis is one-time or sequential, and if sequential, how alpha is controlled.

**Cluster-level interventions.** When the exposure is a policy or a facility-level change, the unit of analysis is the cluster. The analysis must account for clustering, and the design should consider a difference-in-differences or synthetic control approach.

### Decision table: data fitness verdicts

| Situation | Verdict | Design response |
|---|---|---|
| Population, exposure, and outcome all captured; validation study available | Fit | Proceed with primary analysis |
| Outcome captured but not validated; positive control plausible | Fit with limitations | Add validation substudy and bias analysis |
| Exposure captured by prescription only | Fit with limitations | Sensitivity analysis with dispensing data |
| Key confounder not measured | Fit with limitations | E-value and negative control; weaken claim |
| Outcome captured only in one care setting | Unfit for system-level claim | Restrict claim to that setting |
| No active comparator available | Unfit for comparative claim | Reframe as descriptive or single-arm |
| Era of data precedes current standard of care | Unfit for current-practice claim | Restrict to current era or use concurrent control |
| Positivity violated in the target population | Unfit without restriction | Restrict to common support or change question |

The verdict is not a permanent property of the data. The same database can be fit for one question and unfit for another. The assessment must be repeated for each new question.

## 6. Worked example

### The setting

A payer's pharmacy and therapeutics committee is deciding whether to prefer direct oral anticoagulants (DOACs) over warfarin for new starts in adults aged 65 and older with atrial fibrillation. The manufacturer submits an RWE study based on a claims database linked to a hospital discharge registry. The committee's question is comparative effectiveness: among new users, does initiating a DOAC reduce the 1-year risk of ischaemic stroke compared with initiating warfarin?

### Step 1: Estimand

> In adults aged 65 and older with incident atrial fibrillation and no prior anticoagulant use, the risk difference in ischaemic stroke at 365 days between a strategy of initiating a DOAC and a strategy of initiating warfarin, regardless of subsequent discontinuation or switching.

The intercurrent-event strategy is treatment policy for discontinuation and switching. The summary measure is the risk difference, with the odds ratio as a secondary measure. The population is restricted to new users because prevalent users would introduce survivor bias.

### Step 2: Target trial

| Component | Specification |
|---|---|
| Eligibility | Age 65 or older; incident atrial fibrillation diagnosis; no anticoagulant in the prior 12 months; no severe mitral stenosis or mechanical valve; continuous enrolment for 12 months before time zero |
| Strategies | Initiate a DOAC (apixaban, rivaroxaban, edoxaban, or dabigatran) versus initiate warfarin |
| Assignment | Observed clinical decision |
| Follow-up | Time zero to ischaemic stroke, death, disenrolment, or 365 days |
| Outcome | Ischaemic stroke, defined by primary discharge diagnosis codes |
| Contrast | Treatment-policy risk difference at 365 days |
| Analysis | Propensity score with IPW; per-protocol sensitivity analysis |

### Step 3: Data fitness

The database contains 40,000 eligible new users after exclusions. The population is captured through claims, and the outcome is captured through the linked discharge registry. The registry has been used in prior validation studies, but the positive predictive value for ischaemic stroke is not known for this era.

| Dimension | Assessment | Verdict |
|---|---|---|
| Population | New users identifiable; age and diagnosis codes present | Fit |
| Exposure | Dispensing records with drug, dose, and days supplied | Fit |
| Outcome | Discharge diagnosis codes; no local validation | Fit with limitations |
| Confounders | Age, sex, comorbidity, prior bleeding, renal disease, concomitant drugs | Fit with limitations: frailty and over-the-counter NSAID use not measured |
| Era | 2019 to 2023; standard of care stable | Fit |
| Provenance | Claims and registry linkage documented | Fit |

The verdict is fit with limitations. The limitations are unmeasured frailty and over-the-counter non-steroidal anti-inflammatory drug (NSAID) use, and the absence of a local outcome validation study. Both must be carried into the sensitivity analyses.

### Step 4: Cohort and time zero

Time zero is the date of the first dispensing of a DOAC or warfarin after the atrial fibrillation diagnosis. The baseline period is the 12 months before time zero. Patients with any anticoagulant dispensing in that period are excluded. The cohort is restricted to the first eligible time zero for each patient.

A grace period is not used. The exposure definition requires dispensing on the index date. A sensitivity analysis uses a 30-day grace period with symmetric application.

### Step 5: Exposure and comparator

The exposure is the first dispensing of a DOAC, with days supplied used to define continuous use. The comparator is the first dispensing of warfarin. Both groups are new users, and both have the same indication. The active-comparator design reduces confounding by indication because both treatments are plausible choices for the same patient.

Switching is handled by the treatment-policy estimand: a patient who switches remains in the initial strategy. A per-protocol sensitivity analysis censors at switching and uses IPW for censoring.

### Step 6: Outcome and follow-up

The outcome is ischaemic stroke, defined as a primary discharge diagnosis of ischaemic stroke in the linked registry. Follow-up ends at the first stroke, death, disenrolment, or 365 days. Disenrolment is treated as censoring, with a sensitivity analysis using IPW for censoring.

### Step 7: Confounding adjustment

The propensity score includes age, sex, hypertension, diabetes, heart failure, prior stroke, prior bleeding, chronic kidney disease, liver disease, prior NSAID use (from prescription records), antiplatelet use, statin use, and calendar year. The score is estimated with logistic regression, and the analysis uses IPW with stabilised weights.

The balance diagnostic is the standardised mean difference for each covariate, with a target of less than 0.10 after weighting. The overlap diagnostic is a histogram of the propensity score by treatment group.

### Step 8: Immortal time

The design avoids immortal time by aligning time zero to the first dispensing for both strategies and by excluding prior users. A diagnostic is run on a simulated null cohort to confirm that the design does not produce a spurious protective effect. The diagnostic is described in Section 7.

### Step 9: Positivity

The propensity score distributions overlap between 0.05 and 0.95. Patients with a score below 0.05 or above 0.95 are trimmed, and the primary analysis is repeated in the trimmed population. The trimmed population is 97 percent of the original cohort. The result is unchanged.

### Step 10: Measurement error and missing data

The outcome definition has unknown sensitivity and specificity. A positive control analysis uses a known warfarin effect on INR monitoring (a process outcome) to confirm that the data capture warfarin exposure. A negative control uses appendicitis, which should not be affected by anticoagulant choice.

Missing covariates are handled by multiple imputation with 20 imputations. The missingness is below 5 percent for most covariates and about 12 percent for renal function. A sensitivity analysis restricts to complete cases.

### Step 11: Sensitivity analyses

| Analysis | Purpose | Expected direction if confounding is present |
|---|---|---|
| E-value | Quantify robustness to unmeasured confounding | Smaller E-value means more fragile |
| Negative control (appendicitis) | Detect residual confounding | Association suggests bias |
| Positive control (INR monitoring) | Confirm exposure capture | Null suggests measurement problem |
| Alternative time zero (30-day grace) | Test sensitivity to design choice | Attenuation suggests immortal time |
| Per-protocol analysis | Estimate adherence-adjusted effect | Larger effect if non-adherence dilutes |
| Complete-case analysis | Test missing-data sensitivity | Change suggests informative missingness |

### Step 12: Precision

The cohort has 8,921 DOAC initiators and 31,079 warfarin initiators. The crude 1-year stroke risk is 4.93 percent in the DOAC group and 4.41 percent in the warfarin group. The crude odds ratio is 1.125, which appears to suggest harm. This is confounding by indication: DOAC patients are older and have more comorbidity.

After IPW, the weighted risks are 3.89 percent and 4.74 percent, a risk difference of -0.85 percentage points (95 percent confidence interval -1.34 to -0.37). The weighted odds ratio is 0.813 (95 percent confidence interval 0.759 to 0.871). The effective sample sizes after weighting are about 8,026 DOAC and 30,712 warfarin.

The arithmetic for the risk difference is:

```text
RD = 0.0389 - 0.0474 = -0.0085
SE = sqrt(SE_DOAC^2 + SE_WARF)^2
   = sqrt(0.0000022 + 0.0000059)
   = 0.0028
95% CI = -0.0085 +/- 1.96 * 0.0028 = (-0.0134, -0.0037)
```

The weighted odds ratio is computed from the weighted risks:

```text
OR = [0.0389 / (1 - 0.0389)] / [0.0474 / (1 - 0.0474)]
   = 0.04046 / 0.04976
   = 0.813
```

### Step 13: Prespecification

The protocol and SAP were registered before the analysis. The primary estimand, cohort definition, exposure definition, outcome definition, propensity score covariates, weighting method, and sensitivity analyses were fixed in advance. The analysis was conducted once, after the data lock.

### Step 14: Reporting

The report states that the data are consistent with a 0.85 percentage point absolute reduction in 1-year ischaemic stroke, with a confidence interval that excludes no effect. It also states the limitations: unmeasured frailty, over-the-counter NSAID use, and the absence of a local outcome validation study. The E-value for the odds ratio is 1.46, meaning that an unmeasured confounder associated with both treatment and outcome by a risk ratio of 1.46 could explain away the result.

### Crude versus adjusted at a glance

| Quantity | Crude | Adjusted (IPW) | Interpretation |
|---|---:|---:|---|
| Risk in DOAC arm | 4.93% | 3.89% | Confounding inflated the crude risk |
| Risk in warfarin arm | 4.41% | 4.74% | Warfarin patients were healthier |
| Risk difference | +0.52 pp | -0.85 pp | Sign reverses after adjustment |
| Odds ratio | 1.125 | 0.813 | Crude suggests harm; adjusted suggests benefit |
| 95% CI for OR | Not computed | 0.759 to 0.871 | Excludes no effect |
| Effective sample | 40,000 | 38,738 | Weighting costs precision |

The table is the single most important artefact in the report. It shows the reviewer that the crude and adjusted results disagree, and it gives the reason. A report that presents only the adjusted result invites the question of what the crude result was; a report that presents both demonstrates that the analyst understood the confounding.

### What the report does not claim

The report does not claim that DOACs cause a 0.85 percentage point reduction in stroke. It claims that the observed data are consistent with that reduction under the stated assumptions. The distinction is not pedantic. The causal claim would require the exchangeability assumption, and the E-value shows that a moderate unmeasured confounder could overturn it.

The report also does not claim that the result generalises to all patients with atrial fibrillation. The cohort is adults aged 65 and older with incident atrial fibrillation, no prior anticoagulant, and continuous enrolment. The result may not apply to younger patients, patients with valvular disease, or patients outside the database's coverage.

### Computing the E-value by hand

The E-value is a sensitivity threshold, not a probability. For an odds ratio below 1, convert it to an approximate risk ratio by taking the square root, then invert it so that the ratio is above 1, and apply:

```text
E = RR + sqrt(RR * (RR - 1))
```

For the observed odds ratio of 0.813:

```text
RR = sqrt(0.813) = 0.902
RR_inv = 1 / 0.902 = 1.109
E = 1.109 + sqrt(1.109 * 0.109)
  = 1.109 + sqrt(0.1209)
  = 1.109 + 0.348
  = 1.457
```

The interpretation is that an unmeasured confounder would need a risk ratio of at least 1.46 with both treatment and outcome to explain away the observed association. The conversion from odds ratio to risk ratio is approximate and assumes a rare outcome. For common outcomes, the E-value should be computed from the risk ratio directly.

The E-value has two limits. It assumes a single unmeasured confounder, and it does not account for measured confounders that were poorly measured. A study with several moderate unmeasured confounders could be more fragile than the E-value suggests.

### Two diagnostics that reveal a flawed design

**Diagnostic 1: the negative control.** The analysis is repeated with appendicitis as the outcome. If the DOAC group has a lower rate of appendicitis, the association is unlikely to be causal and suggests residual confounding or a selection difference. In the illustrative cohort, the negative control odds ratio is 1.01 (95 percent confidence interval 0.92 to 1.11), which is consistent with no association. If the negative control had shown a protective association of 0.85, the primary result would be suspect.

**Diagnostic 2: the null-cohort immortal time check.** A simulated cohort with a true null effect and a delayed exposure definition is analysed with the same design. If the design produces a protective association, the design has immortal time. In the simulation in Section 7, a true null with a one-month delayed exposure produces a rate ratio of 0.814, a spurious 19 percent benefit. The new-user design with time zero at initiation removes the bias.

### Vary one assumption: the effect size

If the true odds ratio is 0.90 rather than 0.80, the IPW estimate is 0.916 (95 percent confidence interval 0.857 to 0.979), and the risk difference is -0.38 percentage points (95 percent confidence interval -0.89 to +0.12). The confidence interval includes no effect. The payer's decision would shift from "prefer DOAC" to "insufficient evidence for a preference," even though the point estimate still favours DOAC.

### Vary one assumption: unmeasured confounding

If an unmeasured confounder has a risk ratio of 1.5 with both treatment and outcome, it could produce the observed association even if the true effect were null. The E-value of 1.46 is just below that threshold. The result is therefore fragile to moderate confounding. A stronger study would measure frailty, over-the-counter NSAID use, or a validated stroke outcome, or use an active comparator with a different confounding structure.

### Vary one assumption: the outcome definition

If the outcome code has a sensitivity of 0.85 and a specificity of 0.98, and the misclassification is non-differential, the observed odds ratio is biased toward the null. A quantitative bias analysis could adjust the estimate upward. If the misclassification is differential by treatment, the direction is unpredictable. The protocol should prespecify a validation substudy or a bias analysis, not assume non-differential error.

### The decision

The committee reviews the primary estimate, the negative control, the E-value, and the limitations. It concludes that the data support a modest absolute benefit of DOACs, but the evidence is not strong enough for a restrictive formulary preference. It places DOACs on the formulary with prior authorisation for patients at higher stroke risk, and requests a follow-up study with a validated outcome and frailty measurement.

The example shows the central lesson of RWE: the crude association and the causal estimate can point in opposite directions. The crude odds ratio of 1.125 suggested harm; the adjusted estimate of 0.813 suggested benefit. The difference was confounding by indication, not a data error. An analyst who reported the crude result would have been wrong, and an analyst who reported the adjusted result without the diagnostics would have been unconvincing.

## 7. Code, query, or calculation

The code below generates an illustrative cohort, shows the crude and adjusted estimates, and demonstrates immortal time bias in a null cohort. It uses only the Python standard library. The numbers are simulated for teaching; they are not a reproduction of a published study.

```python
import math
import random


def inv_logit(x):
    return 1 / (1 + math.exp(-x))


def age_band(age):
    return min(5, int((age - 60) // 5))


def simulate_cohort(n=40000, seed=42, true_or=0.80):
    """Illustrative cohort with confounding by indication."""
    random.seed(seed)
    rows = []
    for _ in range(n):
        age = random.uniform(60, 90)
        severity = 1 if random.random() < inv_logit(-2.2 + 0.045 * (age - 70)) else 0
        p_treat = inv_logit(-1.6 + 0.035 * (age - 70) + 0.9 * severity)
        treated = random.random() < p_treat
        p_out = inv_logit(-3.6 + 0.055 * (age - 70) + 1.1 * severity
                          + (math.log(true_or) if treated else 0))
        rows.append((age, severity, treated, 1 if random.random() < p_out else 0))
    return rows


def crude(rows):
    t = [r for r in rows if r[2]]
    c = [r for r in rows if not r[2]]
    a = sum(r[3] for r in t)
    b = len(t) - a
    cc = sum(r[3] for r in c)
    d = len(c) - cc
    return {
        "n_treated": len(t),
        "n_control": len(c),
        "risk_treated": round(a / len(t), 4),
        "risk_control": round(cc / len(c), 4),
        "odds_ratio": round((a * d) / (b * cc), 3),
    }


def ipw(rows):
    """Stratum-based propensity score and inverse probability weighting."""
    strata = {}
    for r in rows:
        k = (age_band(r[0]), r[1])
        strata.setdefault(k, [0, 0, 0, 0])
        if r[2]:
            strata[k][0] += 1
            strata[k][2] += r[3]
        else:
            strata[k][1] += 1
            strata[k][3] += r[3]
    p = {k: v[0] / (v[0] + v[1]) for k, v in strata.items()}
    w1 = w0 = y1 = y0 = sw1 = sw0 = 0.0
    for r in rows:
        k = (age_band(r[0]), r[1])
        pr = min(max(p[k], 0.02), 0.98)
        if r[2]:
            w = 1 / pr
            w1 += w
            y1 += w * r[3]
            sw1 += w * w
        else:
            w = 1 / (1 - pr)
            w0 += w
            y0 += w * r[3]
            sw0 += w * w
    r1, r0 = y1 / w1, y0 / w0
    se1 = math.sqrt(r1 * (1 - r1) * sw1 / (w1 * w1))
    se0 = math.sqrt(r0 * (1 - r0) * sw0 / (w0 * w0))
    rd = r1 - r0
    se = math.sqrt(se1 ** 2 + se0 ** 2)
    orr = (r1 / (1 - r1)) / (r0 / (1 - r0))
    se_log = math.sqrt(1 / y1 + 1 / (w1 - y1) + 1 / y0 + 1 / (w0 - y0))
    return {
        "risk_treated": round(r1, 4),
        "risk_control": round(r0, 4),
        "risk_difference": round(rd, 4),
        "rd_ci": (round(rd - 1.96 * se, 4), round(rd + 1.96 * se, 4)),
        "odds_ratio": round(orr, 3),
        "or_ci": (round(math.exp(math.log(orr) - 1.96 * se_log), 3),
                  round(math.exp(math.log(orr) + 1.96 * se_log), 3)),
        "effective_n_treated": round(w1 * w1 / sw1),
        "effective_n_control": round(w0 * w0 / sw0),
    }


def immortal_time_null(n=20000, seed=11, effect=1.0, delay=1):
    """Null cohort with exposure defined by initiation at month `delay`."""
    random.seed(seed)
    events_t = person_t = events_c = person_c = 0.0
    for _ in range(n):
        intended = random.random() < 0.5
        treated = False
        for month in range(12):
            if month == delay and intended:
                treated = True
            hazard = 0.02 * (effect if (treated and month >= delay) else 1.0)
            if random.random() < hazard:
                if treated:
                    events_t += 1
                else:
                    events_c += 1
                person_t += 1
                person_c += 1
                break
            person_t += 1
            person_c += 1
    return round((events_t / person_t) / (events_c / person_c), 3)


def e_value(or_value):
    rr = math.sqrt(or_value)
    if rr < 1:
        rr = 1 / rr
    return round(rr + math.sqrt(rr * (rr - 1)), 2)


if __name__ == "__main__":
    cohort = simulate_cohort()
    print("crude:", crude(cohort))
    print("ipw:", ipw(cohort))
    print("immortal-time null rate ratio:", immortal_time_null())
    print("E-value for OR 0.813:", e_value(0.813))
```

Expected output:

```text
crude: {'n_treated': 8921, 'n_control': 31079, 'risk_treated': 0.0493, 'risk_control': 0.0441, 'odds_ratio': 1.125}
ipw: {'risk_treated': 0.0389, 'risk_control': 0.0474, 'risk_difference': -0.0085, 'rd_ci': (-0.0134, -0.0037), 'odds_ratio': 0.813, 'or_ci': (0.759, 0.871), 'effective_n_treated': 8026, 'effective_n_control': 30712}
immortal-time null rate ratio: 0.814
E-value for OR 0.813: 1.46
```

The output shows three things.

- The crude odds ratio is 1.125, which would be read as harm. The adjusted odds ratio is 0.813, which is consistent with the true simulated effect of 0.80. The difference is confounding by indication.
- The effective sample size after weighting is smaller than the raw count, particularly in the treated group. Precision calculations should use the effective sample size.
- The null cohort with a delayed exposure definition produces a rate ratio of 0.814. That is immortal time bias: a 19 percent spurious benefit when the true effect is null.

The code is deliberately simple. A production analysis would use a validated propensity score model, multiple imputation, and a formal survey or robust variance estimator. The teaching point is the direction and magnitude of the biases, not the specific implementation.

### A query for cohort construction

The SQL below shows the shape of a new-user cohort query in a claims database. It is written in standard SQL and is intended to be adapted, not executed as-is.

```sql
WITH eligible AS (
    SELECT p.person_id,
           MIN(p.index_date) AS index_date,
           p.treatment
    FROM (
        SELECT person_id,
               CASE WHEN drug_class = 'DOAC' THEN 'DOAC'
                    WHEN drug_class = 'WARFARIN' THEN 'WARFARIN' END AS treatment,
               dispensing_date AS index_date,
               ROW_NUMBER() OVER (PARTITION BY person_id ORDER BY dispensing_date) AS rn
        FROM pharmacy_claims
        WHERE drug_class IN ('DOAC', 'WARFARIN')
    ) p
    WHERE p.rn = 1
      AND NOT EXISTS (
          SELECT 1 FROM pharmacy_claims q
          WHERE q.person_id = p.person_id
            AND q.drug_class IN ('DOAC', 'WARFARIN')
            AND q.dispensing_date < p.index_date
            AND q.dispensing_date >= DATEADD('month', -12, p.index_date)
      )
    GROUP BY p.person_id, p.treatment
)
SELECT e.person_id, e.treatment, e.index_date,
       CASE WHEN s.stroke_date IS NOT NULL
             AND s.stroke_date BETWEEN e.index_date
             AND DATEADD('day', 365, e.index_date)
            THEN 1 ELSE 0 END AS stroke_365
FROM eligible e
LEFT JOIN stroke_events s
  ON s.person_id = e.person_id
WHERE e.index_date BETWEEN DATE '2019-01-01' AND DATE '2023-12-31';
```

The query illustrates three design choices: first dispensing only, a 12-month washout, and a 365-day outcome window aligned to the index date. Each corresponds to a target trial component. A query without the washout would include prevalent users; a query without the index-date alignment would introduce immortal time.

## 8. Data, standards, and regulatory context

### Binding requirements

**United States.** The FDA's authority to accept RWE for drug effectiveness derives from section 505F of the Federal Food, Drug, and Cosmetic Act, added by the 21st Century Cures Act. The statute requires the FDA to evaluate RWE and to issue guidance. It does not create an entitlement to approval; the substantial-evidence standard still applies. For devices, the premarket approval standard and the De Novo pathway require valid scientific evidence, and RWE can contribute if it meets the relevance and reliability criteria.

**European Union.** The EMA's scientific advice and qualification procedures provide a route to discuss RWE and novel methodologies. The Medical Device Regulation (EU) 2017/745 requires clinical evaluation, and post-market clinical follow-up can generate RWE. The Clinical Trials Regulation (EU) No 536/2014 governs interventional trials, not observational studies, but the General Data Protection Regulation (EU) 2016/679 constrains the use of personal data.

**Data protection.** In the EU, the GDPR applies to pseudonymised and identifiable data, and the analysis must have a lawful basis and a data protection impact assessment where required. In the US, the Health Insurance Portability and Accountability Act (HIPAA) Privacy Rule governs protected health information, and the Common Rule governs human-subjects research. These are binding legal requirements, not methodological conventions.

### Formal guidance

**FDA device RWE guidance.** "Use of Real-World Evidence to Support Regulatory Decision-Making for Medical Devices" was issued as final guidance in December 2025 (docket FDA-2023-D-4395), superseding the August 2017 version. The Federal Register notice of availability was published on 18 December 2025. It describes relevance and reliability assessments, data accrual and data assurance, and how the assessment fits into benefit-risk review. It also addresses clinical data from devices authorised under an Emergency Use Authorization and information relevant to CLIA waiver determinations.

**FDA drug RWE guidance.** "Considerations for the Use of Real-World Data and Real-World Evidence to Support Regulatory Decision-Making for Drug and Biological Products" was issued in August 2023 (docket FDA-2021-D-1214). It covers the statutory framework, data relevance and reliability, study design and conduct, and the documentation needed for a submission. It is guidance, not regulation.

**FDA externally controlled trials.** The draft guidance "Considerations for the Design and Conduct of Externally Controlled Trials for Drug and Biological Products" (February 2023) applies to drugs and biologics. It describes historical and concurrent external controls, confounding, and sensitivity analyses. It is a draft and does not apply to devices.

**FDA policy announcement, December 2025.** The FDA announced that it will accept certain device RWE submissions without identifiable patient-level data, enabling the use of large de-identified databases. The announcement is a policy statement; the operative requirements remain in the guidance and the device regulations. The practical implication is that de-identified claims and registry data may be more usable in device submissions than before, subject to the guidance's relevance and reliability criteria.

**EMA qualification.** The EMA's qualification of novel methodologies can provide a formal opinion on a data source or method. The PROCOVA qualification opinion (adopted 15 September 2022) is an example in the prognostic-model space; it qualified a prognostic covariate adjustment method for improving precision in randomised trials, not for replacing randomisation. The qualification route is a way to reduce regulatory uncertainty, but it is not a substitute for a well-designed study.

### Reporting conventions

**STROBE.** The Strengthening the Reporting of Observational Studies in Epidemiology statement (2007) provides a 22-item checklist for cohort, case-control, and cross-sectional studies. It is a reporting convention, not a design requirement. It is the minimum for a publishable observational study, and it is increasingly expected in regulatory submissions.

**RECORD.** The RECORD extension to STROBE addresses the reporting of studies using routinely collected health data. It adds items on data linkage, data cleaning, and the validity of codes. It is a convention, but it is the relevant one for claims and EHR studies.

**Target trial emulation reporting.** There is no single reporting guideline for target trial emulations, but the core elements are the seven target trial components, the time-zero definition, the exposure and comparator definitions, the confounding-adjustment method, and the sensitivity analyses. A report that omits these elements cannot be reproduced.

### What the standards do not settle

| Question | Standard | What it leaves open |
|---|---|---|
| Is the data source fit? | FDA relevance and reliability | The specific validation evidence |
| Which estimand? | ICH E9(R1) | Treatment policy versus hypothetical |
| Which comparator? | FDA RWE and external control guidance | Active comparator versus non-user |
| Which confounding method? | None binding | The model and the covariate set |
| How to handle unmeasured confounding? | FDA guidance recommends sensitivity analysis | The method and the threshold |
| How to report? | STROBE and RECORD | The interpretation and the claim |

The pattern is the same as in the previous topics: the standards constrain the process, not the judgement. The judgement must be documented.

### Minimum reporting set for an RWE submission

A submission that includes RWE should contain, at minimum:

- the target trial table and the estimand;
- the data fitness assessment, including validation evidence;
- the cohort flow diagram with exclusion counts;
- the time-zero definition and the exposure and comparator definitions;
- the baseline table before and after weighting;
- the propensity overlap plot;
- the primary estimate with a confidence interval and the effective sample size;
- the negative and positive control results;
- the E-value or bias analysis;
- the list of prespecified versus post hoc analyses.

A submission missing any of these is likely to receive a request for information, which lengthens the review and may delay the decision.

## 9. Industry practice and case evidence

### The FDA's RWE experience

The FDA has reported that, since 2016, more than 250 device premarket authorisations have included RWE. For drugs and biologics, the number is much smaller: the FDA's CDER RWE submissions report for fiscal year 2025 described 31 protocols, 10 new drug application or biologics licence application submissions, and 6 post-marketing requirement final reports, with 4 approvals in which RWE contributed.

The documented fact is the difference in scale between device and drug RWE use. The inference is that device RWE is often used for incremental changes, post-market surveillance, and supplement support, where the question is narrower and the data are closer to the device's use. Drug RWE is more often used for new indications or label changes, where the evidence standard is higher.

**Lesson.** RWE is not a single regulatory product. Its acceptability depends on the question, the pathway, and the strength of the supporting data.

### External control arms in practice

External control arms are used most often in rare diseases, oncology, and device studies where randomisation is difficult. The FDA's 2023 draft guidance on externally controlled trials describes the conditions: a well-characterised natural history, a comparable population, a contemporaneous or stable standard of care, and a prespecified analysis. The guidance is explicit that external controls are not a substitute for randomisation when randomisation is feasible.

A public example is the use of historical controls in device trials, where the FDA's 2013 pivotal-design guidance allows objective performance criteria for some devices. The criteria are based on prior data, and the device must meet a prespecified performance goal. The design is accepted when the performance goal is well established and the device's effect is large relative to the uncertainty.

**Lesson.** External controls work when the benchmark is stable and the effect is large. They fail when the standard of care drifts or the effect is modest.

### The Epic Sepsis Model: a validation case with RWE lessons

Wong and colleagues externally validated a widely implemented proprietary sepsis prediction model at Michigan Medicine. The cohort included 27,697 patients with 38,455 hospitalisations; sepsis was present in 2,552 hospitalisations (7 percent). The hospitalisation-level area under the receiver operating characteristic curve (AUC) was 0.63 (95 percent confidence interval 0.62 to 0.64), compared with the vendor's reported 0.76 to 0.83. At the alert threshold of 6 or more, 18 percent of hospitalisations would trigger an alert, with sensitivity 33 percent and positive predictive value 12 percent.

The documented fact is the performance gap. The inference is that the model's development data and the deployment population differed in case mix, coding, and care processes. The study is a reminder that RWD validation is not a formality; it can overturn a vendor's performance claim.

**Lesson.** An RWE validation must be conducted in the population and setting where the product will be used, with the same outcome definition and the same threshold.

### The Apple Heart Study: RWE at consumer scale

The Apple Heart Study enrolled 419,297 participants in a siteless, app-based study. Over a median 117 days, 2,161 participants (0.52 percent) received an irregular pulse notification. Among 450 notified participants who returned analyzable ECG patches, atrial fibrillation was present in 34 percent; the positive predictive value for notification concordance with simultaneous patch-detected atrial fibrillation was 0.84.

The documented facts are the scale and the confirmation rate. The inference is that the study generated real-world evidence of detection performance, but not of clinical benefit. It did not randomise, did not measure stroke or mortality, and did not capture the full care pathway.

**Lesson.** RWE can establish feasibility and performance at scale. It cannot establish impact without a comparison and an outcome.

### Payer use of RWE

Payers use RWE for coverage decisions, formulary management, and value-based contracts. The evidence requirements are often different from the FDA's: payers care about comparative effectiveness, cost, and subgroup effects in their own population. A study that satisfies the FDA may not satisfy a payer, and vice versa.

The Centers for Medicare and Medicaid Services (CMS) Coverage with Evidence Development (CED) programme is an explicit mechanism for conditional coverage. It allows coverage while a study generates further evidence. For a product team, CED is a signal that the evidence was insufficient for unconditional coverage, and the follow-up study is part of the commercial plan.

**Lesson.** Design the RWE study for the payer question as well as the regulatory question, or expect a second study.

### What companies actually do

The following practices are inferred from public guidance, submissions, and published studies.

1. **Pre-submission meetings.** Companies use the FDA's Q-Submission programme and the EMA's scientific advice to test RWE designs. The meetings are most useful when the company brings a draft protocol and a data fitness assessment.
2. **Feasibility counts before protocol lock.** A common failure is a protocol written before the database is queried. Mature teams run feasibility counts first, then finalise the cohort definition.
3. **Prespecification.** Companies increasingly register RWE protocols and publish SAPs, because payers and journals expect it.
4. **Validation substudies.** When the outcome code is uncertain, teams commission a chart-review validation study. The cost is small relative to the cost of a rejected submission.
5. **Negative controls.** Negative control outcomes are used to probe residual confounding, particularly in claims-based studies.

### Where practice falls short

- **Crude comparisons presented as causal.** A common failure is a table of unadjusted rates with a claim of effectiveness.
- **Immortal time.** Exposure defined by a future event, with pre-exposure person-time counted as exposed.
- **Prevalent users.** Cohorts that include patients already on treatment, with no new-user restriction.
- **No positivity check.** Propensity score analyses with extreme weights and no overlap assessment.
- **No sensitivity analysis.** A single model presented as definitive, with no E-value or negative control.
- **Outcome definition not validated.** Diagnosis codes used without a validation study or a positive control.

### Case-to-lesson map

| Case | Design feature | What it demonstrates | What it does not demonstrate |
|---|---|---|---|
| FDA device RWE, 2025 guidance | Relevance and reliability assessment | Data fitness is question-specific | That any registry is acceptable |
| FDA drug RWE, 2023 guidance | Prespecification and sensitivity analysis | RWE can support selected regulatory decisions | That RWE replaces trials |
| Epic Sepsis Model validation | External validation in the deployment setting | Performance claims can fail outside development data | That the model has no value anywhere |
| Apple Heart Study | Large-scale decentralised cohort | Feasibility and detection performance at scale | Clinical benefit or cost-effectiveness |
| CMS CED | Coverage conditional on evidence | Payers can create an evidence pathway | That coverage is equivalent to approval |
| PROCOVA qualification | EMA qualification of a method | Novel methods can be qualified for a defined use | That qualification replaces randomisation |

The table separates what each case establishes from what it is often cited as establishing. The separation is the difference between evidence and advocacy.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Immortal time | Exposure defined after cohort entry | Pre-exposure person-time counted as exposed | Spurious protective effect | New-user design; time zero at initiation |
| Prevalent-user bias | Cohort includes patients on treatment at entry | Early discontinuers excluded | Healthy-user effect | Restrict to new users |
| Confounding by indication | Treated patients are sicker | Treatment assigned by prognosis | Bias toward or away from null | Active comparator; IPW; E-value |
| Positivity violation | Propensity scores near 0 or 1 | Some patients cannot receive one treatment | Extrapolation | Restrict to common support; trim |
| Outcome misclassification | Diagnosis codes unvalidated | Coding variation by site and era | Bias toward null or unpredictable | Validation substudy; positive control |
| Exposure misclassification | Prescription data only | Prescription is not ingestion | Bias toward null | Dispensing data; dose-response |
| Informative censoring | Disenrolment differs by treatment | Sick patients leave the data | Biased estimate | IPW for censoring; external linkage |
| Secular trend | Standard of care changed during follow-up | Calendar time confounds exposure | Biased estimate | Calendar adjustment; era restriction |
| Negative control positive | Appendicitis associated with treatment | Residual confounding | Causal claim weakened | Report and investigate |
| Multiple analyses | Many cohort definitions tried | Selective reporting | False positive | Prespecification; registration |
| Data linkage error | Duplicate or unmatched records | Wrong person-time or outcome | Biased estimate | Linkage audit; sensitivity analysis |
| Unmeasured frailty | No functional status variable | Confounding by health status | Bias toward benefit | E-value; restrict to homogeneous group |
| Over-adjustment | Adjusting for a mediator | Collider or mediator bias | Attenuated or reversed effect | Causal diagram; prespecified covariates |
| Inappropriate comparison | Non-user comparator | Different indication | Confounding by indication | Active comparator |

### Red flags in an RWE protocol

- The protocol has no target trial table.
- Time zero is the first date in the database.
- The cohort includes patients with prior exposure.
- The comparator is "no treatment" for a drug with a clear indication.
- The analysis has no propensity overlap plot.
- The outcome definition is a single diagnosis code with no validation.
- The results report only a p-value, not an effect size and confidence interval.
- There is no sensitivity analysis for unmeasured confounding.
- The protocol was written after the analysis.
- The conclusion uses causal language for a descriptive result.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| No unmeasured confounding | Causal interpretation | Unmeasured prognosis or frailty | E-value; negative control; bias analysis |
| Positivity | Comparison is defined | Covariate patterns with no treatment variation | Propensity overlap; trimming |
| Consistency | Observed outcome equals potential outcome | Multiple versions of treatment | Define exposure precisely; dose-response |
| Correct time zero | Follow-up aligns with treatment | Delayed initiation; prevalent users | New-user design; null-cohort check |
| Accurate exposure | Effect estimate is valid | Prescription without dispensing | Dispensing data; sensitivity analysis |
| Accurate outcome | Effect estimate is valid | Coding variation; under-capture | Validation study; positive control |
| Complete follow-up | Person-time is accurate | Disenrolment related to outcome | Linkage; IPW for censoring |
| Stable standard of care | Comparison is contemporaneous | Guideline change; new therapy | Calendar adjustment; era restriction |
| No informative measurement | Misclassification is non-differential | Differential testing by treatment | Bias analysis; validation by group |
| Correct model specification | Adjustment removes confounding | Non-linearity; interaction | Doubly robust; alternative model |
| Data provenance is intact | Values are trustworthy | Linkage errors; duplicate records | Audit; lineage documentation |

### Limits of this topic

This topic covers the design and appraisal of RWE for regulatory, payer, and clinical decisions. It does not cover the full theory of causal inference, the details of g-methods, or the implementation of specific estimators. Those are specialised topics with their own literature. It also does not cover the legal requirements for data use, which vary by jurisdiction and require legal review.

The worked example uses a simulated cohort. The numbers are designed to illustrate confounding and immortal time, not to estimate a real treatment effect. The regulatory statements are educational summaries, not legal advice. A submission team should verify the current text and date of every guidance it relies on.

The case evidence is drawn from public sources and is not a systematic review. The inferences about company practice are labelled as inferences.

## 12. Exercises

### Exercise 1 — Data fitness assessment (application, 30 minutes)

**Deliverable:** a fitness assessment for a claims database proposed for a study of a new diabetes drug and heart failure hospitalisation, covering population, exposure, outcome, confounders, and era.

**Acceptance criteria:** each dimension assessed with a specific gap; a verdict of fit, fit with limitations, or unfit; at least two limitations carried into a sensitivity analysis.

**Time:** 30 minutes.

### Exercise 2 — Target trial specification (application, 35 minutes)

**Deliverable:** a seven-component target trial table for the question "does early statin initiation after a myocardial infarction reduce 1-year mortality?"

**Acceptance criteria:** all seven components specified; time zero defined; exposure and comparator are new-user active-comparator; the causal contrast is stated.

**Time:** 35 minutes.

### Exercise 3 — Time zero and immortal time (critique, 30 minutes)

**Deliverable:** a critique of a cohort definition that sets time zero at hospital discharge and defines exposure as "statin dispensed within 90 days," identifying the immortal time and proposing a corrected design.

**Acceptance criteria:** the immortal period is identified with dates; the corrected design aligns time zero to dispensing; the analysis population is stated.

**Time:** 30 minutes.

### Exercise 4 — Confounding method (application, 30 minutes)

**Deliverable:** a justification for choosing IPW over regression adjustment for a study with 40 confounders and a interest in a marginal risk difference.

**Acceptance criteria:** the estimand is named; the method matches the estimand; the diagnostics and sensitivity analysis are stated; the limitations of the alternative are explained.

**Time:** 30 minutes.

### Exercise 5 — Positivity and overlap (calculation and critique, 30 minutes)

**Deliverable:** an interpretation of a propensity score distribution with 8 percent of treated patients above 0.95 and 5 percent of controls below 0.05, and a proposed response.

**Acceptance criteria:** the violation is identified; the consequence for the estimate is explained; at least two responses are proposed; the effect on the estimand is stated.

**Time:** 30 minutes.

### Exercise 6 — E-value calculation (calculation, 25 minutes)

**Deliverable:** the E-value for an observed odds ratio of 0.75 and for 0.90, with a plain-language interpretation of each.

**Acceptance criteria:** formula correct; both values correct; interpretation states the strength of confounding needed to explain away the result; the limits of the E-value are stated.

**Time:** 25 minutes.

### Exercise 7 — Prespecify an RWE study (design, 60 minutes)

**Deliverable:** a two-page protocol for an RWE study of a device in a registry, including the target trial, the data fitness assessment, the estimand, the primary analysis, and the sensitivity analyses.

**Acceptance criteria:** the protocol could be executed by another analyst; the assumptions are listed; the decision rule is stated; the reporting plan is included.

**Time:** 60 minutes.

### Exercise 8 — Regulatory pre-submission question (design, 30 minutes)

**Deliverable:** a one-page question to a regulator asking whether a proposed RWE design would be acceptable for a label expansion, with the target trial, the data source, and the proposed sensitivity analyses.

**Acceptance criteria:** the question is specific and answerable; the design is summarised; the uncertainty is stated; the requested feedback is clear.

**Time:** 30 minutes.

## 13. Self-check answers

**Answer 1 (data fitness).** A claims database may be fit for a study of a diabetes drug and heart failure hospitalisation if it captures the drug, the diagnosis, and the relevant confounders. It is likely fit with limitations because claims lack laboratory values such as HbA1c and renal function, and because heart failure coding may vary. The limitations should be carried into a sensitivity analysis with a negative control outcome and an E-value. If the database does not capture the eligible population or the outcome, the verdict is unfit.

**Answer 2 (target trial).** Eligibility: adults with a first myocardial infarction, no prior statin, no contraindication. Strategies: initiate a high-intensity statin within 30 days versus no statin initiation. Assignment: observed. Follow-up: from time zero to death, disenrolment, or 365 days. Outcome: all-cause mortality. Contrast: treatment-policy risk difference. Analysis: IPW with a propensity score, with a per-protocol sensitivity analysis.

**Answer 3 (immortal time).** If time zero is hospital discharge and exposure is a statin dispensed within 90 days, patients who die before receiving a statin are classified as unexposed, and the exposed group is guaranteed to survive until the dispensing. The immortal period is the time between discharge and dispensing. The corrected design sets time zero at dispensing for the exposed and at a matched eligible date for the unexposed, or uses a new-user design with a 90-day grace period applied symmetrically.

**Answer 4 (confounding method).** IPW is appropriate when the estimand is a marginal risk difference and the propensity score can be estimated well. It balances the covariates without imposing a model on the outcome, and it handles many confounders. The diagnostics are standardised mean differences and overlap. The sensitivity analysis is a doubly robust estimator or a regression-adjusted estimate. Regression adjustment estimates a conditional effect, which may not be the marginal effect of interest.

**Answer 5 (positivity).** The distribution indicates that some treated patients have essentially no chance of being controls and some controls have essentially no chance of being treated. The comparison in those regions is extrapolation. Responses include restricting to the common support, trimming extreme weights, redefining the comparator, or changing the question. The estimand changes if the population is restricted, and the report should state the restricted population.

**Answer 6 (E-value).** For OR 0.75, the approximate risk ratio is the square root of 0.75, or 0.866; the E-value is 1/0.866 + sqrt((1/0.866) * ((1/0.866) - 1)) = 1.155 + 0.429 = 1.58. For OR 0.90, the approximate risk ratio is 0.949; the E-value is 1.054 + 0.245 = 1.30. An unmeasured confounder would need risk ratios of at least 1.58 or 1.30 with both treatment and outcome to explain away the result. The E-value is a sensitivity threshold, not a test of confounding.

**Answer 7 (protocol rubric).** A strong protocol states the decision and estimand, the target trial, the data fitness assessment, the cohort and time zero, the exposure and comparator, the outcome, the confounding method, the diagnostics, the sensitivity analyses, and the decision rule. It is specific enough for another analyst to execute. A weak protocol describes the database and the research question without the design choices.

**Answer 8 (pre-submission rubric).** A strong question names the proposed indication, the target trial, the data source, the primary estimand, the key sensitivity analyses, and the specific feedback requested. It is concise and answerable. A weak question asks "would RWE be acceptable?" without a design.

## 14. Interview questions

**Question 1. What is a target trial emulation, and why does it matter?**

A strong answer: it is an observational analysis designed to approximate the randomised trial that would answer the question, specified through seven components before analysis. It matters because it makes design choices explicit and testable. A weak answer describes it as "using observational data to mimic a trial" without the components.

**Question 2. What is immortal time bias, and how do you prevent it?**

A strong answer: person-time during which the outcome could not occur because exposure is defined by a later event; prevention is a new-user design with time zero at treatment initiation and symmetric grace periods. A weak answer defines it without a design response.

**Question 3. How do you choose between a propensity score and regression adjustment?**

A strong answer: the choice depends on the estimand (marginal versus conditional), the number of confounders, the sample size, and the positivity of the exposure. Propensity scores are useful for many confounders and marginal effects; regression is simpler for few confounders and conditional effects. A weak answer prefers one method categorically.

**Question 4. What is the difference between relevance and reliability of RWD?**

A strong answer: relevance is whether the data contain the right population, exposure, outcome, and era; reliability is whether the data are accurate, complete, and traceable. Both are question-specific. A weak answer treats them as synonyms.

**Question 5. How do you handle unmeasured confounding?**

A strong answer: you cannot eliminate it; you quantify it with an E-value, probe it with negative controls, and possibly correct it with a bias analysis, then weaken the claim accordingly. A weak answer says "adjust for more variables."

**Question 6. When is an external control acceptable?**

A strong answer: when randomisation is infeasible, the natural history is well characterised, the population and era are comparable, the endpoint is objective, and the analysis includes confounding adjustment and sensitivity analyses. A weak answer says "in rare diseases."

**Question 7. What is the role of a negative control outcome?**

A strong answer: it is an outcome that should not be affected by the exposure; an observed association suggests residual confounding or selection bias. It is a diagnostic, not proof of no confounding. A weak answer says it "proves" the study is valid.

**Question 8. What would you ask a regulator before starting an RWE study?**

A strong answer: whether the data source and design are acceptable for the intended claim, what sensitivity analyses are expected, and what documentation is required. A weak answer asks "is RWE allowed?"

**Question 9. Why can an RWE study with 100,000 patients be underpowered?**

A strong answer: because the effective sample size after weighting or matching may be much smaller, and because the event count, not the record count, drives precision. A weak answer says "large data means high power."

**Question 10. What makes an RWE result credible to a payer?**

A strong answer: a relevant population, an active comparator, a validated outcome, a prespecified analysis, a quantified sensitivity to confounding, and a clear statement of uncertainty. A weak answer says "a large database."

## 15. Key terms

- **Active comparator:** a comparator treatment used for the same indication.
- **Claims data:** billing records from insurers or payers.
- **Cloning:** creating copies of participants under each treatment strategy.
- **Common data model:** a standardised schema for observational data, such as OMOP.
- **Confounding by indication:** treatment assignment depends on prognosis.
- **Consistency:** the observed outcome under received treatment equals the potential outcome.
- **Coverage with evidence development (CED):** payer coverage conditioned on further evidence.
- **Data lineage:** the traceable path from source to analysis variable.
- **Data provenance:** the documented origin and processing history of a data element.
- **Depletion of susceptibles:** removal of high-risk people from the exposed group over time.
- **E-value:** the minimum strength of unmeasured confounding needed to explain away a result.
- **Electronic health record (EHR):** clinical documentation and order data from a care provider.
- **Exchangeability:** no unmeasured confounding conditional on covariates.
- **Fitness for purpose:** the match between data characteristics and the question.
- **g-methods:** methods for time-varying treatments and confounding.
- **Grace period:** a window after time zero during which treatment initiation is allowed.
- **Immortal time:** person-time during which the outcome could not have occurred by definition.
- **Inverse probability weighting (IPW):** weighting by the inverse of the treatment probability.
- **Marginal structural model:** a model fitted with IPW to estimate a marginal effect.
- **Negative control outcome:** an outcome not plausibly affected by the exposure.
- **New-user design:** cohort entry at the start of treatment.
- **Positivity:** every covariate pattern has nonzero probability of each treatment.
- **Positive control outcome:** an outcome known to be affected by the exposure.
- **Prevalent-user bias:** bias from including people already on treatment at cohort entry.
- **Propensity score:** the probability of treatment given covariates.
- **Quantitative bias analysis:** a model-based correction for a specified bias.
- **Real-world data (RWD):** data collected outside a conventional trial.
- **Real-world evidence (RWE):** evidence derived from RWD through a defined design.
- **Registry:** a structured collection of data on a defined population.
- **Standardisation:** weighting stratum-specific estimates to a target population.
- **Target trial:** the randomised experiment that would answer the question.
- **Target trial emulation:** an observational analysis designed to approximate a target trial.
- **Time zero:** the moment eligibility, exposure, and follow-up begin.

## 16. Sources

All URLs accessed 25 October 2026.

### Statutes and regulations

- United States, Federal Food, Drug, and Cosmetic Act, section 505F (RWE). https://www.fda.gov/regulatory-information/laws-enforced-fda/federal-food-drug-and-cosmetic-act-fdc-act
- European Union, Regulation (EU) 2017/745 on medical devices. https://eur-lex.europa.eu/eli/reg/2017/745/oj
- European Union, Regulation (EU) 2016/679 (General Data Protection Regulation). https://eur-lex.europa.eu/eli/reg/2016/679/oj
- United States, HIPAA Privacy Rule, 45 CFR Parts 160 and 164. https://www.hhs.gov/hipaa/for-professionals/privacy/index.html

### FDA guidance and policy

- FDA, "Use of Real-World Evidence to Support Regulatory Decision-Making for Medical Devices," final guidance, December 2025 (docket FDA-2023-D-4395). https://www.fda.gov/regulatory-information/search-fda-guidance-documents/use-real-world-evidence-support-regulatory-decision-making-medical-devices
- FDA, "Considerations for the Use of Real-World Data and Real-World Evidence to Support Regulatory Decision-Making for Drug and Biological Products," final guidance, August 2023. https://www.fda.gov/regulatory-information/search-fda-guidance-documents/considerations-use-real-world-data-and-real-world-evidence-support-regulatory-decision-making-drug
- FDA, "Considerations for the Design and Conduct of Externally Controlled Trials for Drug and Biological Products," draft guidance, February 2023. https://www.fda.gov/regulatory-information/search-fda-guidance-documents/considerations-design-and-conduct-externally-controlled-trials-drug-and-biological-products
- FDA, "Advancing Real-World Evidence Program." https://www.fda.gov/drugs/development-resources/advancing-real-world-evidence-program
- FDA, "Real-World Evidence Submissions to the Center for Drug Evaluation and Research." https://www.fda.gov/science-research/real-world-evidence/real-world-evidence-submissions-center-drug-evaluation-and-research
- FDA, "FDA Eliminates Major Barrier to Using Real-World Evidence in Drug and Device Application Reviews," press announcement, 15 December 2025. https://www.fda.gov/news-events/press-announcements/fda-eliminates-major-barrier-using-real-world-evidence-drug-and-device-application-reviews

### EMA

- EMA, Qualification opinion for Prognostic Covariate Adjustment (PROCOVA), adopted 15 September 2022. https://www.ema.europa.eu/en/documents/regulatory-procedural-guideline/qualification-opinion-prognostic-covariate-adjustment-procovatm_en.pdf
- EMA, Scientific advice and protocol assistance. https://www.ema.europa.eu/en/human-regulatory-overview/research-development/scientific-advice-protocol-assistance

### Reporting guidelines

- von Elm E, Altman DG, Egger M, et al. The Strengthening the Reporting of Observational Studies in Epidemiology (STROBE) statement. Lancet 2007;370:1453-1457. https://doi.org/10.1016/S0140-6736(07)61602-X
- Benchimol EI, Smeeth L, Guttmann A, et al. The REporting of studies Conducted using Observational Routinely-collected health Data (RECORD) statement. PLoS Med 2015;12:e1001885. https://doi.org/10.1371/journal.pmed.1001885

### Methods and case evidence

- Hernán MA, Robins JM. Using Big Data to Emulate a Target Trial When a Randomized Trial Is Not Available. Am J Epidemiol 2016;183:758-764. https://doi.org/10.1093/aje/kwv254
- Suissa S. Immortal Time Bias in Pharmacoepidemiology. Am J Epidemiol 2008;167:492-499. https://doi.org/10.1093/aje/kwm324
- Wong A, Otles E, Donnelly JP, et al. External Validation of a Widely Implemented Proprietary Sepsis Prediction Model in Hospitalized Patients. JAMA Intern Med 2021;181:1065-1070. https://doi.org/10.1001/jamainternmed.2021.2626
- Perez MV, Mahaffey KW, Hedlin H, et al. Large-Scale Assessment of a Smartwatch to Identify Atrial Fibrillation. N Engl J Med 2019;381:1909-1917. https://doi.org/10.1056/NEJMoa1901183
- CMS, Coverage with Evidence Development. https://www.cms.gov/medicare/coverage/coverage-with-evidence-development

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic: [`02_trial-designs.md`](02_trial-designs.md)
- Next topic: [`04_validation-designs.md`](04_validation-designs.md)
- Related topic: [`05_bias-taxonomy.md`](05_bias-taxonomy.md)
- Module survey file: [`../../05_study_design_evidence.md`](../../course/05_study_design_evidence.md)
- Labs: [`../../labs/`](../../labs/)
- Cohort pipeline lab: [`../../labs/03_cohort_pipeline.ipynb`](../../labs/03_cohort_pipeline.ipynb)
- Calibration and decision curve lab: [`../../labs/06_calibration_decision_curve.ipynb`](../../labs/06_calibration_decision_curve.ipynb)
