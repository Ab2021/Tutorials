# Data Quality Dimensions

> Module 02 — The Biomedical Data Landscape
> Topic 10 of 10
> Estimated study time: 5 hours
> Prerequisites: all previous topics are helpful; Topics 01 and 02 are the most relevant
> Version 2026.3

---

## 1. Why this topic exists

The previous nine topics taught you what each data source is and what it can answer. This one teaches you how to decide whether a specific dataset is good enough for a specific decision.

That question has no universal answer. "Good quality" is not a property of a dataset; it is a relationship between a dataset and a use. A dataset that is adequate for a prevalence estimate may be inadequate for a safety signal. A dataset that is inadequate for a clinical trial may be perfectly adequate for capacity planning.

The industry problem is that quality is usually assessed either not at all or with a single number. A completeness percentage is reported, the number looks acceptable, and the analysis proceeds. The number hides the pattern, and the pattern is what determines bias.

Three facts organize the topic:

1. **Quality is multidimensional, and the dimensions trade off.** Improving completeness by excluding incomplete records can reduce validity by selecting a biased subgroup. Every fix has a cost in another dimension.
2. **Missingness has mechanisms, and the mechanism determines the bias.** Missing completely at random, missing at random, and missing not at random imply different analyses and different conclusions.
3. **Fitness for purpose is a judgment, and it must be documented.** A quality report that ends with "the data are of high quality" has not made a judgment; it has avoided one.

The commercial stakes:

- A payer that reports a quality measure without checking completeness by site will rank providers by documentation practice rather than by care.
- A sponsor that accepts a real-world data source without a quality assessment will fail a data fitness review.
- A product team that tunes a model on a dataset with 40 percent missingness in a key feature will ship a model that works only where the feature is recorded.
- An analytics team that reports a single composite quality score will make a decision that reverses when the weights change.

The decision this topic supports: whether a dataset is fit for a stated use, what the residual risks are, and what must be reported alongside any result.

### 1.1 Who is blocked, and by what

| Role | Decision blocked | Failure without this topic |
|---|---|---|
| Analytics lead | Whether to release a result | No documented fitness assessment |
| Quality lead | Whether to compare providers | Completeness differences read as performance |
| Regulatory lead | Whether data are fit for an RWE submission | Cannot answer relevance and reliability questions |
| Data engineer | Whether a pipeline is ready | Quality checks are ad hoc |
| Product manager | Whether a model will generalize | Missingness pattern not assessed |
| Biostatistician | Which missing-data method to use | Mechanism not identified |
| Privacy officer | Whether a release is safe | Small-cell risk not assessed |

### 1.2 Why a single number fails

A completeness of 92 percent can describe:

- Every record missing 8 percent of fields.
- 8 percent of records missing every field.
- One site missing 60 percent of one field, with all other sites complete.

The first is a nuisance, the second is a cohort problem, and the third is a site comparability problem. All three produce the same headline number, and only the pattern distinguishes them.

---

## 2. Learning outcomes

By the end of this topic you can:

1. **Apply** the Kahn framework of conformance, completeness, and plausibility to a real dataset, and explain why it avoids the terms accuracy and validity.
2. **Compute** completeness, observed agreement, and Cohen's kappa for a concordance assessment, with correct arithmetic.
3. **Classify** a missingness pattern by mechanism and state the direction of bias each implies.
4. **Build** a multidimensional quality profile stratified by site, year, and subgroup.
5. **Critique** a composite quality score for weight sensitivity and propose a sensitivity analysis.
6. **Decide** whether a dataset is fit for a stated use, with an explicit threshold and a documented rationale.
7. **Design** a quality report that a reviewer, a regulator, and an operator could each act on.
8. **Distinguish** data quality from data integrity and from data validity, and use each term correctly.

Mapping to assessment: outcomes 2, 3, and 5 are assessed by exercises 2, 3, and 5. Outcomes 1, 4, and 8 are assessed by the self-check items. Outcomes 6 and 7 are assessed by the design exercises and the module memo.

---

## 3. The core mental model

Data quality is a set of questions asked of a dataset in the context of a use.

```text
                    FITNESS FOR PURPOSE
                            |
        +-------------------+-------------------+
        |                   |                   |
   CONFORMANCE         COMPLETENESS        PLAUSIBILITY
   (does it follow     (is it there?)      (is it believable?)
    the rules?)
        |                   |                   |
   - value             - missing fields    - uniqueness
   - relational        - missing records   - atemporal
   - computational     - missing visits    - temporal
        |                   |                   |
        +-------------------+-------------------+
                            |
                    TWO CONTEXTS
        VERIFICATION  (internal expectations, metadata)
        VALIDATION    (external gold standard, benchmark)
                            |
                            v
                    QUALITY PROFILE
                    by site, year, subgroup
                            |
                            v
                    FITNESS DECISION
                    (threshold, rationale, residual risk)
```

The framework in the diagram is the one published by Kahn and colleagues in 2016, and it is worth understanding why it is shaped this way.

Kahn and colleagues consolidated terminology from about 100 contributors across roughly 20 US EHR research networks covering more than 540 million patient records. They deliberately **rejected** the terms accuracy, correctness, and validity as top-level categories, because those terms are used inconsistently across the data quality and psychometric literature. They substituted **plausibility**, defined as whether values are believable, assessed against internal expectations (verification) or external standards (validation).

That choice matters analytically. "Accuracy" implies a truth that is usually unavailable. "Plausibility" is a judgment that can be made, tested, and documented. A quality report that says "plausibility was assessed against physiologic ranges and against a chart-reviewed sample" is more useful than one that says "accuracy was high."

### 3.1 The three categories in detail

| Category | Subcategory | Question | Example check |
|---|---|---|---|
| Conformance | Value | Do values conform to the expected format and code set? | Percent of diagnosis codes in the valid FY2026 list |
| Conformance | Relational | Do references resolve? | Orphan encounter identifiers |
| Conformance | Computational | Do derived values satisfy arithmetic rules? | Discharge date before admit date |
| Completeness | — | Are values present? | Percent missing by field and site |
| Plausibility | Uniqueness | Are there duplicates? | Same patient, same day, same service |
| Plausibility | Atemporal | Are values believable ignoring time? | Systolic blood pressure of 400 mmHg |
| Plausibility | Temporal | Are values believable over time? | Weight change of 40 kg in one day |

### 3.2 Verification versus validation

The same check can be run in two contexts:

- **Verification** compares data to internal expectations: metadata constraints, edit rules, known ranges, cross-field logic.
- **Validation** compares data to an external standard: chart review, a registry, a claims file, a published benchmark.

Verification is cheap and catches most structural errors. Validation is expensive and is the only way to assess whether the data reflect reality. A quality program that does only verification will report a clean dataset that is systematically wrong.

### 3.3 The missingness mechanisms

| Mechanism | Definition | Implication | Example |
|---|---|---|---|
| MCAR | Missingness independent of everything | Complete-case analysis is unbiased, less efficient | A tube lost in transit |
| MAR | Missingness depends on observed variables | Multiple imputation is valid given the observables | Labs ordered more for older patients |
| MNAR | Missingness depends on the unobserved value | No method is guaranteed; sensitivity analysis required | A lab not ordered because the clinician judged the patient too well |
| Structural | Field not applicable | Must be excluded from the denominator | Pregnancy field for a male patient |
| Workflow | Not collected because of process | Often related to care quality | A flowsheet row not used at night |
| Device | Sensor failure or non-wear | Related to behaviour and setting | A wearable removed for charging |

The practical point is that most clinical missingness is not MCAR. A lab is ordered because a clinician is worried, so its absence is information. A diagnosis is coded because it was documented, so its absence may reflect documentation practice rather than disease.

### 3.4 The fitness decision

Fitness is a threshold judgment, and it has three parts:

1. **The requirement.** What the use needs, stated in quality terms (for example, "completeness of the exposure field above 90 percent at every site").
2. **The evidence.** The measured quality profile against that requirement.
3. **The residual risk.** What remains uncertain after the assessment, and how the analysis handles it.

A fitness decision without all three is an opinion.

---

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| Data quality | The degree to which data meet requirements for a use | Treated as an absolute property | Multidimensional |
| Data integrity | The completeness, consistency, and accuracy of data over its lifecycle | Confused with data quality | ALCOA+ principles |
| Data validity | Conformance to a rule or format | Confused with accuracy | Rule-based |
| Fitness for purpose | Whether data are adequate for a specific use | Treated as a global judgment | Decision |
| Conformance | Adherence to specified standards and formats | Confused with completeness | Category |
| Completeness | Presence of required values | Confused with correctness | Percent |
| Plausibility | Whether values are believable | Called accuracy | Category |
| Uniqueness | Absence of duplicates | Ignored in record-level analysis | Count or rate |
| Atemporal plausibility | Believability ignoring time | Confused with temporal | Check |
| Temporal plausibility | Believability over time | Confused with atemporal | Check |
| Verification | Assessment against internal expectations | Confused with validation | Context |
| Validation | Assessment against an external standard | Confused with verification | Context |
| Missing completely at random | Missingness independent of all variables | Assumed too often | Mechanism |
| Missing at random | Missingness depends on observed variables | Confused with MCAR | Mechanism |
| Missing not at random | Missingness depends on the unobserved value | Ignored in complete-case analysis | Mechanism |
| Structural missingness | Field not applicable | Counted as missing | Mechanism |
| Workflow missingness | Not collected because of process | Treated as random | Mechanism |
| Concordance | Agreement between two sources | Called accuracy | Kappa or percent |
| Cohen's kappa | Agreement corrected for chance | Confused with percent agreement | −1 to 1 |
| Positive predictive value | Proportion of positives that are true | Confused with sensitivity | Proportion |
| Sensitivity | Proportion of true positives detected | Confused with PPV | Proportion |
| Specificity | Proportion of true negatives detected | Confused with PPV | Proportion |
| Completeness rate | Present values divided by expected values | Computed without a denominator definition | Percent |
| Timeliness | Whether data are available when needed | Confused with currency | Lag |
| Currency | How recent the data are | Confused with timeliness | Date |
| Consistency | Agreement across sources or over time | Confused with concordance | Check |
| Composite score | A weighted combination of dimensions | Treated as objective | Score |
| Weight sensitivity | Change in decision when weights change | Ignored | Analysis |
| Quality profile | A multidimensional assessment | Confused with a single metric | Report |
| Data quality dashboard | An automated set of checks | Confused with a quality assessment | Tool |
| ALCOA+ | Attributable, legible, contemporaneous, original, accurate, complete, consistent, enduring, available | Confused with a statistical framework | Principles |
| Audit trail | A record of changes | Confused with a log | Record |
| Small-cell risk | Re-identification risk from small counts | Ignored in aggregate releases | Count |
| Data provenance | Origin and processing history | Documented after the fact | Record |
| Benchmark | An external reference for comparison | Treated as ground truth | Value |
| Gold standard | A reference standard assumed correct | Assumed error-free | Standard |

### 4.1 Terms that are routinely conflated

- **Quality versus integrity.** Quality is fitness for a use; integrity is the trustworthiness of the record over its lifecycle. A dataset can have high integrity and poor fitness for a specific question.
- **Validity versus accuracy.** Validity is conformance to a rule; accuracy is correspondence to truth. A value can be valid and inaccurate, for example a plausible but wrong blood pressure.
- **Timeliness versus currency.** Currency is how recent the data are; timeliness is whether they arrived when needed. A current dataset can be untimely if it arrives after the decision.
- **Concordance versus accuracy.** Concordance is agreement between two sources; both can be wrong in the same way. Accuracy requires a reference standard.

---

## 5. The mechanics, step by step

This section is the working procedure for a data quality assessment. It has 30 steps in five phases.

### Phase A — Define the requirement (steps 1-6)

**Step 1. State the use and the decision it supports.**
Input: the stakeholder request. Output: one sentence naming the decision and the data's role in it.
Reason: fitness is relative to a use. Without the use, every threshold is arbitrary.
Rule: if the decision cannot be stated, the quality assessment has no target.

**Step 2. List the data elements the use depends on.**
Input: the analysis plan. Output: a short list of critical elements.
Reason: a quality assessment of every field is expensive and unfocused. Most uses depend on 5 to 15 fields.
Rule: distinguish critical (the analysis fails without them) from supporting.

**Step 3. State the requirement per critical element.**
Input: the use. Output: a threshold per element (for example, completeness above 90 percent at every site).
Reason: a threshold converts a judgment into a test, and a test can be passed or failed explicitly.
Rule: state whether the threshold applies overall, by site, or by subgroup.

**Step 4. Identify the reference standard available for validation.**
Input: the landscape. Output: a reference (chart review, registry, claims, benchmark).
Reason: verification alone cannot detect systematic error.
Rule: if no reference exists, state that validation was not possible and treat it as a residual risk.

**Step 5. Identify the decisions the quality result will change.**
Input: the use. Output: a list of actions (proceed, proceed with caveats, remediate, abandon).
Reason: a quality report that cannot change a decision is documentation, not analysis.
Rule: define the action thresholds before measuring.

**Step 6. Write the assessment plan.**
Input: steps 1 to 5. Output: a one-page plan.
Reason: the plan prevents the assessment from becoming a fishing expedition for problems.
Rule: include the elements, thresholds, strata, reference standard, and actions.

### Phase B — Measure conformance (steps 7-11)

**Step 7. Check value conformance.**
Input: the data. Output: the percent of values that conform to the expected format and code set.
Reason: non-conforming values cannot be used and usually indicate a mapping or version problem.
Rule: check against the code set version in force, not a generic pattern.

**Step 8. Check relational conformance.**
Input: the data. Output: the count of unresolved references.
Reason: orphan records break joins and silently drop data.
Rule: report orphans by direction (child without parent, parent without child).

**Step 9. Check computational conformance.**
Input: the data. Output: the count of arithmetic and logical violations.
Reason: dates before birth, discharge before admission, and totals that do not sum are unambiguous errors.
Rule: distinguish violations that are impossible from those that are merely unusual.

**Step 10. Report conformance by stratum.**
Input: the checks. Output: a table by site, year, and domain.
Reason: a conformance problem concentrated at one site is a build problem, not a data problem.
Rule: any stratum above twice the overall rate deserves an explanation.

**Step 11. Record the code set versions used.**
Input: the conformance check. Output: a version record.
Reason: conformance against the wrong version produces false findings.
Rule: state the ICD, LOINC, and terminology versions.

### Phase C — Measure completeness (steps 12-16)

**Step 12. Define the denominator per element.**
Input: the data model. Output: an expected count per element.
Reason: completeness requires knowing what "complete" means. A field that is structurally inapplicable must be excluded from the denominator.
Rule: distinguish structural absence from missing data.

**Step 13. Compute completeness overall and by stratum.**
Input: the data. Output: a completeness matrix.
Reason: the aggregate hides the pattern, and the pattern determines the bias.
Rule: stratify by site, year, age band, sex, and payer at minimum.

**Step 14. Classify missingness by mechanism.**
Input: the completeness matrix and the workflow knowledge. Output: a mechanism per element.
Reason: the mechanism determines whether complete-case analysis is defensible.
Rule: use MCAR only when you have evidence, not as a default.

**Step 15. Test for differential missingness with respect to the outcome.**
Input: the data and an outcome proxy. Output: outcome rates for complete and incomplete records.
Reason: this is the single most informative bias check available without a reference standard.
Rule: report the difference and the direction.

**Step 16. Test for differential missingness with respect to the exposure.**
Input: the data. Output: exposure rates by completeness status.
Reason: if completeness differs by exposure, the association is biased even when prevalence is not.
Rule: run both tests; one can be negative while the other is positive.

### Phase D — Measure plausibility (steps 17-22)

**Step 17. Check uniqueness.**
Input: the data. Output: a duplicate rate.
Reason: duplicates inflate counts and understate variance.
Rule: define the duplicate key before counting, and report the key.

**Step 18. Check atemporal plausibility.**
Input: the data. Output: the percent outside physiologic or logical bounds.
Reason: implausible values indicate entry errors, unit errors, or defaults.
Rule: use bounds that are deliberately wide; the purpose is to catch errors, not to make clinical judgments.

**Step 19. Check temporal plausibility.**
Input: the data. Output: the count of implausible changes over time.
Reason: a weight change of 40 kg in a day is an error, and a lab value that never changes may be a carried-forward default.
Rule: compute the rate of change and flag beyond a threshold.

**Step 20. Validate against a reference standard.**
Input: a sample and a reference. Output: concordance, PPV, sensitivity.
Reason: plausibility against internal rules does not establish correspondence to truth.
Rule: report the sample size, the reference, and the agreement statistic.

**Step 21. Compute concordance with chance correction.**
Input: the cross-tabulation. Output: observed agreement and kappa.
Reason: raw agreement is inflated when one category dominates.
Rule: report both, because kappa can be low when prevalence is extreme even with high agreement.

**Step 22. Report plausibility by stratum.**
Input: the checks. Output: a stratified table.
Reason: an implausible pattern at one site can indicate a unit or device difference.
Rule: compare distributions, not only rates.

### Phase E — Judge fitness (steps 23-30)

**Step 23. Assemble the quality profile.**
Input: all measurements. Output: a single table across dimensions and strata.
Reason: the profile is the evidence base for the decision.
Rule: include the denominator and the sample size for every metric.

**Step 24. Score the dimensions against the requirement.**
Input: the profile and the thresholds. Output: a pass, conditional, or fail per element.
Reason: the score must be traceable to a threshold, not to a feeling.
Rule: state the rule for "conditional."

**Step 25. Test the sensitivity of any composite score.**
Input: the weights. Output: the decision under alternative weights.
Reason: a composite score is a model, and its weights are assumptions. If the decision flips with reasonable weights, the score is not decision-grade.
Rule: run at least equal weights and one alternative.

**Step 26. Identify the residual risks.**
Input: the profile. Output: a list of risks that remain after the assessment.
Reason: every assessment has blind spots, and they must be named.
Rule: include the risks that the assessment could not test.

**Step 27. Decide and document.**
Input: the profile, the scores, the sensitivity, and the risks. Output: a fitness decision with a rationale.
Reason: the decision is the deliverable; the profile is the evidence.
Rule: state the action (proceed, proceed with caveats, remediate, abandon) and the conditions.

**Step 28. Specify the mitigation for each conditional element.**
Input: the decision. Output: a mitigation plan.
Reason: a conditional pass without a mitigation is a pass with extra words.
Rule: each mitigation has an owner, a method, and a date.

**Step 29. Report alongside the result.**
Input: the decision. Output: a quality paragraph in the deliverable.
Reason: a reader who sees the result without the quality context will over-generalize.
Rule: put the quality statement next to the result, not in an appendix.

**Step 30. Re-assess on a schedule.**
Input: the data refresh. Output: a re-assessment.
Reason: quality changes as data arrive and as systems change.
Rule: re-assess at every data refresh and after any system upgrade.

### 5.1 Measuring concordance correctly

Cohen's kappa is the standard chance-corrected agreement statistic for two raters and two categories:

kappa = (p_observed − p_expected) / (1 − p_expected)

where p_observed is the proportion of records on which the two sources agree, and p_expected is the agreement expected from the marginal distributions.

Two cautions:

1. **Kappa is sensitive to prevalence.** When one category is rare, kappa can be low even with high agreement, which is why the prevalence-adjusted bias-adjusted kappa (PABAK) exists.
2. **Kappa does not say which source is right.** Concordance is agreement, not accuracy. A high kappa between two sources that share the same error is not evidence of correctness.

The practical recommendation is to report observed agreement, kappa, and the marginals together, plus PPV and sensitivity against a reference where one exists.

### 5.2 The composite score problem

A composite quality score is a weighted sum of dimension scores. It is convenient for a dashboard and dangerous for a decision, for three reasons:

1. **The weights are assumptions.** A score with completeness weighted at 30 percent and plausibility at 20 percent encodes a judgment that is rarely stated or defended.
2. **The scale is arbitrary.** A 1-to-5 ordinal score assumes equal intervals, which is rarely true.
3. **The aggregation hides failure.** A site that fails one critical dimension can pass on the composite because the other dimensions compensate.

The mitigation is not to abandon composites but to report the components alongside, run a weight sensitivity analysis, and apply a gate: a site that fails a critical dimension fails regardless of the composite.

### 5.3 Quantitative checks before any analysis

| Check | Computation | Passing signal | What failure means |
|---|---|---|---|
| Critical-field completeness | Present / expected per field | Above the stated threshold at every site | Fitness failure or site problem |
| Key uniqueness | Rows versus distinct keys | Equal | Grain error |
| Referential integrity | Orphan count | Zero or documented | Incomplete extract |
| Temporal ordering | Events before birth or after death | Zero | Date or identity error |
| Value conformance | Valid codes / total codes | Above 99% | Code set version mismatch |
| Duplicate records | Duplicate key rate | Low and stable | Registration or feed problem |
| Outcome differential | Outcome rate by completeness | Small difference | Informative missingness |
| Distribution stability | Feature distribution by period | Smooth | Unit or system change |

### 5.4 Derivations and the assumptions they hide

| Derived quantity | Usual construction | Hidden assumption | Challenge question |
|---|---|---|---|
| Completeness rate | Present divided by expected | The denominator is correct | Is structural absence excluded? |
| Missingness rate | 1 minus completeness | Missing means absent | Are defaults counted as present? |
| Kappa | Chance-corrected agreement | The categories are exhaustive and mutually exclusive | Are there unclassified records? |
| PPV | True positives divided by all positives | The reference is correct | What is the reference, and what is its error? |
| Composite score | Weighted sum of dimension scores | The weights and scale are meaningful | Does the decision survive equal weights? |
| Duplicate rate | Duplicates divided by records | The duplicate key is correct | Was the key defined before counting? |
| Timeliness | Median lag from event to availability | The event date is accurate | Which date, and is it complete? |

### 5.5 Edge cases the procedure has to survive

**Structural versus missing.** A field that does not apply to a record must be excluded from the denominator, not counted as missing. The signature is a completeness rate that falls when a new patient group is added to the cohort. The rule is to define eligibility per field and to report both rates.

**Conflicting sources.** The EHR says one diagnosis, claims another, the registry a third. The rule is to carry all three with a source flag, report the concordance, and use the source closest to the clinical event for clinical facts.

**Unit variants across sites.** A lab reported in mg/dL at one site and µmol/L at another produces a plausibility check that fails for the wrong reason. The rule is to normalize units before plausibility testing, and to test the unit field first.

**Code-system versions.** A conformance check against the wrong ICD year produces false failures. The rule is to check the version in force at the service date, not at the extraction date.

**Late-arriving data.** Completeness measured before the run-out is complete will understate the final rate. The rule is to measure completeness at a fixed interval after the period ends, and to state the interval.

**Partial identifiers.** A record that cannot be linked to a patient cannot contribute to longitudinal completeness. The rule is to report cross-sectional and longitudinal completeness separately.

**Duplicate records.** Duplicates inflate completeness and deflate missingness. The rule is to deduplicate before any completeness calculation, and to report the duplicate rate.

**Timezone and date boundaries.** A "same-day" readmission depends on the timezone of the discharge and the admission. The rule is to align to one timezone and to state whether calendar days or 24-hour intervals are used.

### 5.6 Decision table: handling informative missingness

| Evidence | Likely mechanism | Primary analysis | Sensitivity | What to report |
|---|---|---|---|---|
| Missingness unrelated to group and outcome | MCAR | Complete case | None beyond precision | Complete-case rate |
| Missingness predicted by observed covariates | MAR | Multiple imputation | Complete case | Imputation model |
| Outcome rate higher in missing group | MNAR | Include missing indicator | Bounding | Both estimates and the bound |
| Outcome rate lower in missing group | MNAR, opposite direction | Include missing indicator | Bounding | Both estimates |
| Missingness concentrated at one site | Workflow | Site-stratified | Pooled with site effect | Site-specific rates |
| Missingness concentrated in one period | System change | Split the series | Pooled with a period indicator | Change date |

The rule that overrides the table: the direction of the bias must be established from data, not assumed. The variant in section 6.9 shows that reversing the assumption reverses the direction of the bias and the decision.

### 5.7 Field-level validation rules

| Rule | Test | Severity | Action |
|---|---|---|---|
| Denominator defined | Every metric has an explicit denominator | High | Block the report |
| Structural exclusion | Inapplicable records excluded from the denominator | High | Recompute |
| Duplicate check | Duplicate key rate computed before completeness | High | Deduplicate first |
| Unit normalized | Units standardized before plausibility | High | Quarantine unconvertible |
| Code version | Codes validated against the version in force | Medium | Flag; report unmapped |
| Stratification | Every metric reported by site and year | High | Block the comparison |
| Threshold pre-specified | Thresholds recorded before measurement | High | Re-document |
| Reference standard | Validation sample size and method stated | Medium | Flag as unvalidated |
| Small-cell suppression | Counts below the threshold suppressed | Medium | Suppress before release |
| Re-assessment date | Assessment date within the refresh cycle | Low | Re-run |

### 5.8 How to set a threshold that is not arbitrary

A threshold should come from one of four places, in order of preference:

| Basis | Example | Strength | Weakness |
|---|---|---|---|
| Decision requirement | Adjustment variable needed for 90% of patients | Tied to the use | Requires modelling the use |
| Benchmark | Published completeness for the same measure | External | Benchmark may not transfer |
| Prior performance | Site's own best year | Realistic | Institutionalises mediocrity |
| Expert consensus | Clinical informatics group | Available | Not evidence |

The first is the only one that connects the threshold to the decision, and it is the one most often skipped because it requires knowing what the analysis will do with the variable.

A practical method: for each critical variable, ask what fraction of patients can be missing before the adjusted estimate changes by more than the effect size of interest. That fraction is the threshold, and it is computable with a simple sensitivity analysis rather than a debate.

Worked mini-example. If the effect of interest is a 3 percentage point mortality difference, and excluding patients with a missing adjustment variable shifts the estimate by 2 points at 20 percent missing and 4 points at 40 percent missing, the threshold is between 20 and 40 percent, and 30 percent is a defensible choice with the sensitivity analysis documented.

That method produces a threshold with a rationale, which is what a reviewer needs and what a copied benchmark cannot provide.

### 5.9 What a quality profile must contain to change a decision

A profile that lists metrics without connecting them to actions is a report, not an assessment. The minimum structure:

| Element | Content | Consequence if missing |
|---|---|---|
| Critical elements | The fields the use depends on | Assessment is unfocused |
| Thresholds | Pre-specified per element | Decision is post hoc |
| Stratification | Site, year, subgroup | Pattern is hidden |
| Differential test | Outcome by completeness status | Bias direction unknown |
| Reference validation | PPV or agreement, with sample size | Accuracy unknown |
| Decision rule | Pass, conditional, fail | No action follows |
| Residual risk | What could not be tested | Overconfidence |

### 5.10 Branch points summarized

| Branch | Condition | Path A | Path B |
|---|---|---|---|
| Missingness | MCAR supported by tests | Complete-case analysis | Multiple imputation |
| Missingness | MAR plausible | Multiple imputation with observables | Sensitivity analysis |
| Missingness | MNAR suspected | Bounding or tipping-point analysis | State that the estimate is not identified |
| Critical element | Completeness below threshold | Exclude the site or the element | Remediate and re-assess |
| Concordance | High kappa, high PPV | Use the phenotype | Use only for screening |
| Composite | Decision flips with weights | Abandon the composite | Use gates plus components |
| Reference | None available | State the validation gap | Use an internal benchmark as verification only |

### 5.6 How to choose a missing-data method

The method follows the mechanism, not the analyst's preference.

1. **If MCAR is supported**, complete-case analysis is unbiased and simpler. Report the loss of precision.
2. **If MAR is plausible**, multiple imputation using the observed predictors is valid, and the imputation model should include the outcome and auxiliary variables.
3. **If MNAR is suspected**, no method is guaranteed. Use a bounding analysis, a tipping-point analysis, or a pattern-mixture model, and present the result as a range.
4. **If the missingness is structural**, exclude those records from the denominator and do not impute.
5. **If the missingness is workflow-driven**, model it as a site or period effect rather than imputing it away.

The common error is to impute under MAR when the mechanism is MNAR, which produces a precise estimate of the wrong quantity. The second most common error is to delete records to avoid the choice.

A useful discipline is to report the primary analysis under the pre-specified method and the sensitivity analysis under the most plausible alternative. If the conclusion changes, the missing-data assumption is part of the finding.

A final check on any missing-data method: does the imputation model include the variables that predict missingness? If it does not, MAR imputation is invalid even when MAR holds, because the imputation does not condition on the right variables.

That check catches more errors than the choice between methods.

The same logic applies to the outcome differential test in step 15: it is not a formality, it is the fastest way to see whether the missing data are informative, and it costs one query.

If the excluded group differs on the outcome, the remaining analysis should carry that difference as a stated bias or as a sensitivity range, and never as an unremarked exclusion.

---

## 6. Worked example

### 6.1 The question

A four-site health system wants to report a 30-day heart failure readmission measure. The quality team must determine whether the EHR extract is fit for the measure at each site.

Critical elements: heart failure diagnosis, ejection fraction (EF), admission and discharge dates, discharge disposition, and readmission within 30 days.

### 6.2 Completeness of the critical elements

| Site | HF patients | EF recorded | EF completeness | Discharge date complete |
|---|---:|---:|---:|---:|
| A | 640 | 499 | 78.0% | 99.5% |
| B | 520 | 213 | 41.0% | 98.8% |
| C | 580 | 510 | 87.9% | 99.7% |
| D | 400 | 248 | 62.0% | 97.5% |
| Total | 2,140 | 1,470 | 68.7% | 99.0% |

Check: 499/640 = 0.7797. 213/520 = 0.4096. 510/580 = 0.8793. 248/400 = 0.6200.
Total: (499 + 213 + 510 + 248) = 1,470. 1,470/2,140 = 0.687.

Site B's EF completeness of 41.0 percent is below any reasonable threshold. A readmission model that adjusts for EF will effectively exclude site B, and the exclusion is not random.

### 6.3 The bias from requiring EF

Restricting the cohort to patients with an EF recorded:

| Group | Patients | Deaths | Mortality |
|---|---:|---:|---:|
| EF recorded (included) | 1,470 | 141 | 9.6% |
| EF missing (excluded) | 670 | 95 | 14.2% |
| All | 2,140 | 236 | 11.0% |

Check: 1,470 + 670 = 2,140. 141 + 95 = 236. 141/1,470 = 0.0959. 95/670 = 0.1418. 236/2,140 = 0.1103.

The excluded patients have 4.6 percentage points higher mortality (14.2 − 9.6). They are sicker, and they were excluded because a measurement was not recorded, which is itself related to the care pathway: a patient who dies before an echocardiogram never gets an EF.

A complete-case analysis would therefore understate mortality by 1.4 percentage points (11.0 − 9.6) and would be biased toward the patients who survived long enough to be measured.

### 6.4 Concordance between EHR and claims

For 2,000 patients with linked EHR and claims, the heart failure diagnosis was compared:

| | Claims HF+ | Claims HF− | Total |
|---|---:|---:|---:|
| EHR HF+ | 412 | 138 | 550 |
| EHR HF− | 96 | 1,354 | 1,450 |
| Total | 508 | 1,492 | 2,000 |

Observed agreement = (412 + 1,354) / 2,000 = 1,766 / 2,000 = 0.883.

Expected agreement:
- EHR+ proportion = 550/2,000 = 0.275.
- Claims+ proportion = 508/2,000 = 0.254.
- EHR− proportion = 0.725.
- Claims− proportion = 0.746.
- p_expected = (0.275 × 0.254) + (0.725 × 0.746) = 0.06985 + 0.54085 = 0.6107.

kappa = (0.883 − 0.6107) / (1 − 0.6107) = 0.2723 / 0.3893 = 0.6995, or 0.70.

Using claims as the reference:
- PPV of EHR = 412/550 = 0.749.
- Sensitivity of EHR = 412/508 = 0.811.

Check: 412/550 = 0.7491. 412/508 = 0.8110.

A kappa of 0.70 is substantial agreement, and the PPV of 74.9 percent means that a quarter of EHR-identified heart failure patients are not identified in claims. The direction of the discrepancy is informative: EHR captures patients whose condition is documented but not billed, which is plausible for a chronic condition managed in primary care.

### 6.5 Plausibility checks

| Check | Site A | Site B | Site C | Site D | Threshold |
|---|---:|---:|---:|---:|---:|
| Systolic BP outside 60-250 mmHg | 0.4% | 2.1% | 0.3% | 1.8% | below 1% |
| Weight change above 20 kg in 24 h | 0.2% | 0.6% | 0.1% | 0.9% | below 0.5% |
| Discharge before admission | 0 | 0 | 3 | 0 | zero |
| Duplicate encounters | 0.8% | 1.1% | 0.6% | 2.4% | below 1.5% |

Sites B and D fail the plausibility checks, and site C has three computational violations. The pattern suggests a device integration issue at B (blood pressure outliers) and a registration issue at D (duplicates and weight changes).

### 6.6 The composite score and its sensitivity

Dimensions scored 1 to 5 by site: completeness, concordance, plausibility, timeliness, uniqueness.

| Site | Completeness | Concordance | Plausibility | Timeliness | Uniqueness |
|---|---:|---:|---:|---:|---:|
| A | 4 | 4 | 5 | 4 | 5 |
| B | 2 | 3 | 3 | 5 | 5 |
| C | 5 | 4 | 4 | 3 | 4 |
| D | 3 | 3 | 4 | 4 | 4 |

Weighted score with weights 30, 25, 20, 15, 10 percent:

- A: 0.30×4 + 0.25×4 + 0.20×5 + 0.15×4 + 0.10×5 = 1.20 + 1.00 + 1.00 + 0.60 + 0.50 = 4.30.
- B: 0.30×2 + 0.25×3 + 0.20×3 + 0.15×5 + 0.10×5 = 0.60 + 0.75 + 0.60 + 0.75 + 0.50 = 3.20.
- C: 0.30×5 + 0.25×4 + 0.20×4 + 0.15×3 + 0.10×4 = 1.50 + 1.00 + 0.80 + 0.45 + 0.40 = 4.15.
- D: 0.30×3 + 0.25×3 + 0.20×4 + 0.15×4 + 0.10×4 = 0.90 + 0.75 + 0.80 + 0.60 + 0.40 = 3.45.

With a threshold of 3.5: A and C pass, B fails at 3.20, and D fails at 3.45.

Equal weights:

- A: (4+4+5+4+5)/5 = 22/5 = 4.40.
- B: (2+3+3+5+5)/5 = 18/5 = 3.60.
- C: (5+4+4+3+4)/5 = 20/5 = 4.00.
- D: (3+3+4+4+4)/5 = 18/5 = 3.60.

With equal weights and the same 3.5 threshold, all four sites pass, including B at 3.60.

The decision reverses for B and D. The weighted score fails them because it emphasizes completeness, where both are weak; the equal-weight score passes them because timeliness and uniqueness compensate.

This is not a statistical subtlety. It is the difference between excluding two sites and including them, and it is determined entirely by weights that were chosen without justification.

### 6.7 The fitness decision

The defensible decision is not the composite. It is a gate-based decision:

| Site | Completeness gate (EF above 75%) | Plausibility gate | Decision |
|---|---|---|---|
| A | Pass (78.0%) | Pass | Include |
| B | Fail (41.0%) | Fail | Exclude, or report without EF adjustment |
| C | Pass (87.9%) | Pass | Include |
| D | Fail (62.0%) | Fail | Include with caveats, or remediate |

Gates make the decision traceable. A site that fails a critical gate is not rescued by strong performance on a dimension that does not matter for the measure.

The residual risk is stated explicitly: site B's exclusion removes 520 patients, 24.3 percent of the total, and those patients may differ systematically. The report should state the excluded population's characteristics and refrain from claiming system-wide performance.

Check: 520/2,140 = 0.2430.

### 6.8 What the example establishes

- A single completeness number hides a site-level failure.
- Conditioning on a recorded measurement selects survivors and understates mortality.
- Concordance is high and not perfect, and the direction of disagreement is informative.
- A composite score's decision can reverse with plausible weights, so composites need gates and sensitivity analyses.
- The fitness decision must name the excluded population and the residual risk.

### 6.9 Variant case: the missing patients were healthier, not sicker

The primary analysis assumed, from the observed data, that patients without a recorded ejection fraction had higher mortality. Suppose the opposite is true: the ejection fraction was not recorded because those patients were low-risk and the workup was deferred, and their mortality is 6.3 percent rather than 14.2 percent.

| Group | Patients | Deaths | Mortality |
|---|---:|---:|---:|
| EF recorded | 1,470 | 141 | 9.6% |
| EF missing, original assumption | 670 | 95 | 14.2% |
| EF missing, variant assumption | 670 | 42 | 6.3% |
| All, original | 2,140 | 236 | 11.03% |
| All, variant | 2,140 | 183 | 8.55% |

Check: 670 × 0.063 = 42.2, so 42 deaths. 141 + 42 = 183. 183 / 2,140 = 0.0855.

The direction of the bias reverses:

- Under the original assumption, complete-case analysis understates mortality by 1.4 points (9.6 versus 11.03).
- Under the variant, complete-case analysis overstates mortality by 1.0 point (9.6 versus 8.55).

Check: 11.03 − 9.59 = 1.44. 9.59 − 8.55 = 1.04.

The complete-case estimate is identical in both cases. Only the assumption about the unobserved group changed, and the conclusion about the direction of bias changed with it.

### 6.10 What the variant changes, and the threshold sensitivity

The decision about whether to include the missing patients is not the only thing that depends on an assumption. The completeness gate is also a choice, and it moves sites across the boundary:

| Gate | A (78.0%) | B (41.0%) | C (87.9%) | D (62.0%) | Sites included |
|---|---|---|---|---|---|
| 75% | Pass | Fail | Pass | Fail | A, C |
| 60% | Pass | Fail | Pass | Pass | A, C, D |
| 50% | Pass | Fail | Pass | Pass | A, C, D |

Check: D's 62.0 percent clears a 60 percent gate and fails a 75 percent gate. A's 78.0 percent clears both; B's 41.0 percent clears neither.

The gate at 60 percent admits site D, which has 400 patients and a plausibility failure (2.4 percent duplicate encounters). The gate at 75 percent excludes it. Neither gate is obviously right, and the choice should be justified by what the missing EF data would do to the adjustment, which is the method in section 5.8.

### 6.11 Which assumption I would report, and why

Report the measured differential, not either assumed one, and report the range:

- **The measured differential** between included and excluded patients is the only evidence available, and it supports the original direction in this dataset.
- **The range** from the variant (a 1.4 point understatement to a 1.0 point overstatement) is the honest statement of what the data cannot establish.
- **The gate** should be reported with its rationale, and the sensitivity at 60 and 75 percent should be shown, because D's inclusion changes the cohort and the comparison.

The general rule is that an assumption about the unobserved is not a finding. A report that presents one direction as fact, when the opposite is equally consistent with the data, has overstated what the quality assessment can conclude.

### 6.12 What a reviewer would ask

1. **Why 75 percent for EF completeness?** Because below that level, adjustment for EF is effectively a site selection mechanism. The threshold should be justified against the analysis plan, not chosen to fit the data.
2. **Is kappa the right statistic?** For a common condition with balanced marginals it is reasonable, and PPV and sensitivity should be reported alongside because they are what a clinician interprets.
3. **Why not impute EF?** Imputation is defensible under MAR, and EF missingness is likely related to survival, which is MNAR. A sensitivity analysis with imputation is appropriate; a primary analysis that relies on it is not.
4. **Does the gate approach hide a failure elsewhere?** It can. Gates should be paired with a profile of all dimensions, so a site that passes the gates but fails elsewhere is visible.

---

## 7. Code, query, or calculation

### 7.1 Python: concordance, kappa, and completeness

Assumptions: Python 3.11 or later; pandas and numpy installed.

```python
import numpy as np
import pandas as pd

def cohens_kappa(a, b):
    """Cohen's kappa for two binary vectors."""
    a = np.asarray(a, dtype=int)
    b = np.asarray(b, dtype=int)
    n = len(a)
    po = np.mean(a == b)
    pa1, pb1 = a.mean(), b.mean()
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    return po, pe, (po - pe) / (1 - pe)

# 2x2 table from section 6.4
tp, fp, fn, tn = 412, 138, 96, 1354
ehr = np.array([1] * (tp + fp) + [0] * (fn + tn))
clm = np.array([1] * (tp + fn) + [0] * (fp + tn))
po, pe, kappa = cohens_kappa(ehr, clm)
print(f"Observed agreement: {po:.4f}")
print(f"Expected agreement: {pe:.4f}")
print(f"Cohen's kappa:      {kappa:.4f}")
print(f"PPV (claims ref):   {tp/(tp+fp):.4f}")
print(f"Sensitivity:        {tp/(tp+fn):.4f}")
print(f"Specificity:        {tn/(tn+fp):.4f}")
print()

# Completeness by site from section 6.2
comp = pd.DataFrame({
    "site": ["A", "B", "C", "D"],
    "patients": [640, 520, 580, 400],
    "ef_recorded": [499, 213, 510, 248],
})
comp["ef_completeness_pct"] = 100 * comp["ef_recorded"] / comp["patients"]
comp["gate_pass"] = comp["ef_completeness_pct"] >= 75
print(comp.to_string(index=False))
print(f"\nTotal completeness: "
      f"{100*comp['ef_recorded'].sum()/comp['patients'].sum():.2f}%")
```

Expected output:

```text
Observed agreement: 0.8830
Expected agreement: 0.6107
Cohen's kappa:      0.6995
PPV (claims ref):   0.7491
Sensitivity:        0.8110
Specificity:        0.9075

site  patients  ef_recorded  ef_completeness_pct  gate_pass
   A       640          499                77.97       True
   B       520          213                40.96      False
   C       580          510                87.93       True
   D       400          248                62.00      False

Total completeness: 68.69%
```

### 7.2 Python: composite score sensitivity

Assumptions: no external packages beyond numpy.

```python
import numpy as np

sites = ["A", "B", "C", "D"]
scores = np.array([
    [4, 4, 5, 4, 5],
    [2, 3, 3, 5, 5],
    [5, 4, 4, 3, 4],
    [3, 3, 4, 4, 4],
], dtype=float)
dims = ["completeness", "concordance", "plausibility", "timeliness", "uniqueness"]
threshold = 3.5

weight_sets = {
    "weighted_30_25_20_15_10": np.array([0.30, 0.25, 0.20, 0.15, 0.10]),
    "equal_20_each": np.array([0.20, 0.20, 0.20, 0.20, 0.20]),
    "completeness_heavy_50": np.array([0.50, 0.20, 0.15, 0.10, 0.05]),
}

print(f"{'weights':<26}" + "".join(f"{s:>8}" for s in sites) + f"{'pass':>8}")
for name, w in weight_sets.items():
    composite = scores @ w
    n_pass = int((composite >= threshold).sum())
    print(f"{name:<26}" + "".join(f"{v:8.2f}" for v in composite) + f"{n_pass:>8}")

print()
print("Per-dimension scores:")
print(f"{'site':<6}" + "".join(f"{d[:6]:>8}" for d in dims))
for s, row in zip(sites, scores):
    print(f"{s:<6}" + "".join(f"{v:8.1f}" for v in row))
```

Expected output:

```text
weights                          A       B       C       D    pass
weighted_30_25_20_15_10       4.30    3.20    4.15    3.45       2
equal_20_each                 4.40    3.60    4.00    3.60       4
completeness_heavy_50         4.35    3.00    4.30    3.25       2

Per-dimension scores:
site  comple  concor  plausi  timeli  unique
A        4.0     4.0     5.0     4.0     5.0
B        2.0     3.0     3.0     5.0     5.0
C        5.0     4.0     4.0     3.0     4.0
D        3.0     3.0     4.0     4.0     4.0
```

The number of sites passing moves from 2 to 4 and back to 2 as the weights change, which is the evidence that the composite is not decision-grade on its own.

### 7.3 SQL: completeness and outcome differential

Assumptions: PostgreSQL; `hf_patient` at one row per patient with an `ef_recorded` flag and a `died` flag.

```sql
-- Completeness and mortality differential by EF completeness.
SELECT site,
       COUNT(*) AS patients,
       COUNT(*) FILTER (WHERE ef_recorded) AS ef_recorded,
       ROUND(100.0 * COUNT(*) FILTER (WHERE ef_recorded) / COUNT(*), 1)
           AS ef_completeness_pct,
       ROUND(100.0 * COUNT(*) FILTER (WHERE ef_recorded AND died)
             / NULLIF(COUNT(*) FILTER (WHERE ef_recorded), 0), 1)
           AS mortality_ef_recorded_pct,
       ROUND(100.0 * COUNT(*) FILTER (WHERE NOT ef_recorded AND died)
             / NULLIF(COUNT(*) FILTER (WHERE NOT ef_recorded), 0), 1)
           AS mortality_ef_missing_pct
FROM hf_patient
GROUP BY site
ORDER BY site;
```

Expected output:

```text
site | patients | ef_recorded | ef_completeness_pct | mort_recorded | mort_missing
-----+----------+-------------+---------------------+---------------+-------------
A    |      640 |         499 |                78.0 |           8.4 |         13.5
B    |      520 |         213 |                41.0 |          10.1 |         14.7
C    |      580 |         510 |                87.9 |           9.2 |         13.9
D    |      400 |         248 |                62.0 |          11.8 |         14.5
```

The final two columns are the bias check. If the missing group's mortality is consistently higher, complete-case analysis understates mortality at every site, and the understatement is largest where completeness is lowest.

### 7.4 A hand calculation for an interview

Given the 2×2 table with 412, 138, 96, and 1,354:

1. Observed agreement = (412 + 1,354)/2,000 = 0.883.
2. Row margins: 550 and 1,450. Column margins: 508 and 1,492.
3. Expected agreement = (0.275 × 0.254) + (0.725 × 0.746) = 0.6107.
4. Kappa = (0.883 − 0.611)/(1 − 0.611) = 0.272/0.389 = 0.699.

Reporting 0.70 without the marginals and the PPV would hide that a quarter of EHR cases are not in claims.

---

## 8. Data, standards, and regulatory context

### 8.1 The frameworks

| Framework | Origin | Dimensions | Use |
|---|---|---|---|
| Kahn et al. 2016 | eGEMs, US EHR research networks | Conformance, completeness, plausibility | Secondary use of EHR data |
| Weiskopf and Weng 2013 | JAMIA | Completeness, correctness, concordance, plausibility, currency | EHR data quality review |
| Wang and Strong 1996 | Management information systems | Intrinsic, contextual, representational, accessibility | General data quality |
| DAMA-DMBOK | Data Management Association | Multiple dimensions including accuracy, completeness, consistency, timeliness, uniqueness, validity | Data management practice |
| OHDSI Data Quality Dashboard | OHDSI | Conformance, completeness, plausibility at the CDM level | Automated checks on OMOP |
| ALCOA+ | Regulated clinical research | Attributable, legible, contemporaneous, original, accurate, plus completeness, consistency, enduring, available | Data integrity in trials |

The frameworks overlap and are not interchangeable. Kahn is the most useful for secondary analysis of EHR data because it is explicit about verification versus validation and because it deliberately avoids the word accuracy. ALCOA+ is the right framework for a regulated trial record, where the question is whether the record is trustworthy as evidence rather than whether it is fit for a secondary analysis.

### 8.2 What Kahn changed, and why it matters

The Kahn framework's most important move is the rejection of accuracy, correctness, and validity as top-level categories. The authors noted that these terms carry different meanings in data quality and psychometric literature, which made cross-study comparison impossible.

The replacement, plausibility, has two contexts:

- **Verification** against internal expectations: metadata constraints, edit rules, known ranges.
- **Validation** against external standards: chart review, registries, benchmarks.

This structure lets a quality report say precisely what was tested and against what. "Plausibility was verified against physiologic ranges and validated against a 200-record chart review with a PPV of 0.86" is a statement a reviewer can evaluate. "Accuracy was high" is not.

### 8.3 Regulatory expectations

- **FDA real-world evidence.** The framework asks about data relevance and reliability. Reliability includes accuracy, completeness, provenance, and auditability. A quality profile is the evidence for a reliability claim.
- **ICH E6(R2) and ALCOA+.** Trial data must be attributable, legible, contemporaneous, original, and accurate. Source data verification and audit trails are the mechanisms.
- **21 CFR Part 11.** Electronic records must have audit trails, access controls, and validated systems.
- **HIPAA.** Quality assessment often involves identifiable data, and the minimum necessary standard applies.
- **HEDIS and CMS quality programmes.** Measure specifications include data completeness and validity requirements, and providers are audited against them.

### 8.4 Data integrity versus data quality

The two terms are often used interchangeably and should not be:

| Aspect | Data quality | Data integrity |
|---|---|---|
| Question | Is it fit for this use? | Is the record trustworthy over its lifecycle? |
| Framework | Kahn, Weiskopf, DAMA | ALCOA+, GCP |
| Evidence | Quality profile | Audit trail, source verification |
| Failure | Bias in the analysis | Doubt about whether the record reflects what happened |
| Example | 40% missing EF | A value edited after the fact with no audit trail |

A dataset can have perfect integrity and poor quality for a specific use, and it can have good quality for a use and a compromised audit trail. Both assessments are needed for a regulated submission.

### 8.5 The OHDSI Data Quality Dashboard

The OHDSI Data Quality Dashboard runs several thousand automated checks against an OMOP CDM instance, organized into the same conformance, completeness, and plausibility categories. The checks are parameterized by the CDM version and the vocabulary release.

Two practical points:

1. **The dashboard is a screen, not an assessment.** It finds structural problems quickly and does not assess fitness for a specific question.
2. **The thresholds are configurable, and the configuration is a judgment.** A team that lowers a threshold to pass a check has made a decision that should be documented.

### 8.6 Small-cell risk in quality reporting

Quality reports often include counts by subgroup. Small counts can identify individuals, particularly when combined with geography and a rare condition.

The standard mitigations are suppression (reporting "fewer than 11" instead of the count), rounding, and aggregation. The threshold is a policy choice, and it should be stated in the report rather than applied silently.

> **Note:** Small-cell suppression is a disclosure-control practice, not a statistical method. It reduces but does not eliminate re-identification risk when the underlying data are otherwise identifiable.

---

## 9. Industry practice and case evidence

### 9.1 Why quality assessment is usually skipped

Quality assessment is skipped for predictable reasons:

- It delays the analysis and produces no visible output.
- It requires access to metadata and workflow knowledge that the analyst may not have.
- It surfaces problems that the team would then have to address.
- It has no standard deliverable, so it is easy to deprioritize.

The consequence is that quality problems are discovered during validation, when they are expensive, rather than during profiling, when they are cheap.

### 9.2 What the published record shows about EHR data quality

The published evidence is consistent across settings:

- Completeness varies substantially by field, site, and year, and the variation is larger within a health system than between health systems for the same vendor.
- Vital signs and labs are more complete than social and behavioural fields.
- Diagnosis codes are near-complete for billing purposes and unreliable as clinical truth.
- Free-text fields contain information that structured fields lack, and extracting it requires validated methods.

The implication for an analyst is that a quality assessment must be stratified. An aggregate figure describes no site well.

### 9.3 The MIMIC and eICU experience

Public critical care datasets illustrate the same pattern. MIMIC-IV is a single-center dataset with detailed physiology and a documentation culture specific to one hospital. eICU-CRD is multi-site and has different completeness for labs and vitals because ordering and charting practices differ.

Studies that pool or compare them find that the differences in measurement frequency and completeness explain a substantial part of the performance gap between models. That is a data quality finding expressed as a model finding, and it is why the quality profile belongs in the paper rather than the supplement.

### 9.4 How payers assess quality

Payers typically run a data quality programme with four components:

1. **Completeness against a benchmark.** Encounter and diagnosis counts compared to expected values from enrollment and prior years.
2. **Validity checks.** Code validity, date logic, and provider identifiers.
3. **Chart review audits.** A sample of records reviewed to estimate coding accuracy.
4. **Risk adjustment validation.** RADV-style audits of diagnoses submitted for payment.

The fourth is the one with financial consequences, and it is the reason the OIG's MA audit findings in topic 2 matter to a data quality discussion. A diagnosis that cannot be supported by the chart is a quality failure with a payment consequence.

### 9.5 How sponsors assess real-world data

A sponsor evaluating an RWE source typically asks:

- What is the provenance, and is it documented?
- What is the completeness of the exposure, outcome, and covariates?
- What is the validation evidence for the phenotype?
- What is the lag, and is the data current enough for the question?
- What are the known limitations, and are they disclosed?

These map closely to the Kahn categories and to the FDA relevance and reliability questions. A vendor that can answer them with documents is easier to qualify than one that answers with assertions.

### 9.6 The cost of a quality failure

Quality failures are expensive in three ways:

- **Rework.** An analysis that must be re-run after a quality finding costs the original effort plus the re-run.
- **Decision error.** A wrong decision based on biased data can cost far more than the analysis.
- **Reputation.** An organization known for unreliable numbers loses the ability to influence decisions, regardless of the quality of its modeling.

The third is the most expensive and the least measured.

### 9.7 What a mature quality function looks like

- **A quality profile template** applied to every dataset before analysis.
- **Thresholds agreed with the decision owner**, not chosen by the analyst.
- **Automated checks** in the pipeline, with results stored alongside the data.
- **A validation sample** with chart review for the critical phenotypes.
- **A quality statement** in every deliverable, next to the result.

Organizations with these five produce numbers that survive scrutiny. Organizations without them produce numbers that are correct until someone checks.

### 9.8 Quality as a product feature

For a data product, the quality profile is part of the value proposition. A vendor that publishes completeness, validation, and lag figures is easier to buy from than one that publishes only record counts, because the buyer can assess fitness before committing.

The commercial logic is that transparency reduces the buyer's diligence cost, which increases willingness to pay. Vendors that hide quality data are signalling that the data would not survive inspection.

---

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| One-number quality report | Only completeness is reported | Aggregation hides pattern | Bias undetected | Multidimensional profile |
| Aggregate without strata | Overall rate looks fine | Site differences average out | Site comparison invalid | Stratify by site and year |
| Complete-case on informative missingness | Excluded group differs on outcome | Conditioning on a severity-related variable | Biased estimate | Report differential; sensitivity analysis |
| MCAR assumed without evidence | No mechanism analysis | Convenience assumption | Wrong imputation or deletion | Test mechanism |
| Structural missingness counted as missing | A field is "90% missing" | Inapplicable records in denominator | False alarm | Define denominator per element |
| Kappa reported alone | No marginals or PPV | Prevalence sensitivity | Misleading agreement | Report agreement, kappa, PPV |
| Concordance treated as accuracy | Two sources agree | Shared error | False confidence | Validate against a reference |
| Composite without gates | One dimension fails but score passes | Compensation across dimensions | Wrong inclusion decision | Gates plus components |
| Weights unquestioned | Decision flips with equal weights | Arbitrary weights | Decision not robust | Weight sensitivity analysis |
| Plausibility bounds too tight | Many "errors" are real outliers | Clinical judgment imposed by rule | Data deleted | Wide bounds; flag, don't delete |
| Duplicates undetected | Counts higher than expected | Grain or registration issue | Inflated rates | Define duplicate key; count |
| Timeliness ignored | Data arrive after the decision | Lag not measured | Analysis late or wrong | Measure lag; set cutoff |
| Small cells published | Subgroup counts below 11 | Disclosure risk | Privacy breach | Suppression policy |

### 9.9 How a data quality programme is staffed and measured

| Function | Responsibility | Metric | Typical failure |
|---|---|---|---|
| Data steward | Field definitions and code sets | Definition changes per quarter | Dictionary drifts |
| Quality analyst | Profiling and reporting | Profiles delivered per dataset | Reports without decisions |
| Clinical reviewer | Phenotype validation | PPV by phenotype | Validation backlog |
| Engineering | Automated checks | Check pass rate | Checks not maintained |
| Owner | Fitness decision | Decisions documented | Decision deferred |

The programme is measured by decisions, not by reports. A team that produces ten profiles and no fitness decisions has built a dashboard, not a function.

### 10.1 The three red flags

1. **A quality section that reports one number.** Quality is multidimensional, and one number cannot describe it.
2. **A threshold that appears only after the measurement.** A threshold chosen to fit the data is not a threshold.
3. **No statement of the excluded population.** Any analysis that excludes records must say who was excluded and how they differed.

### 10.2 The "the data are clean" trap

"The data are clean" usually means the analyst ran a few checks and found nothing alarming. It does not mean the data are fit for the use, and it does not mean the checks were adequate.

The repair is to replace the phrase with a statement of what was checked, against what, and with what result. That sentence is longer and far more useful.

---

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The critical elements are correctly identified | Focused assessment | The use depends on an unlisted field | Review with the decision owner |
| The thresholds reflect the use | Fitness judgment | Thresholds copied from elsewhere | Justify against the analysis plan |
| The reference standard is adequate | Validation | The reference shares the error | Independent chart review |
| Missingness mechanism is correctly classified | Method choice | Unobserved confounder of missingness | Sensitivity analysis |
| The code set version is correct | Conformance | Version mismatch | Check the version in force |
| Strata are complete | Detecting site effects | An unmeasured stratum | Add strata; review with operators |
| The composite weights are defensible | Scoring | Arbitrary weights | Weight sensitivity |
| The assessment is current | Relevance | System change since assessment | Re-assess on refresh |

### 10.3 The assumption presented as a finding

The variant in section 6.9 is the most common quality-reporting error: a direction of bias stated as fact when the data are consistent with either direction. The signature is a sentence like "missing patients were likely sicker" with no differential test behind it.

The repair is to report the measured difference where it exists, and to state the range where it does not. A sentence with a range is less satisfying and more true.

### 11.1 Limits that are inherent

- **Truth is usually unavailable.** Plausibility and concordance are proxies, and neither is accuracy.
- **Validation samples are small.** A 200-record review estimates PPV with a wide interval.
- **Missingness mechanisms are inferred, not observed.** The mechanism is a hypothesis supported by tests.
- **Quality changes.** An assessment is a snapshot of a moving system.
- **Fitness is use-specific.** A dataset fit for one question can be unfit for the next.
- **Composite scores are models.** Their weights encode judgments that are rarely stated.

### 11.2 How to state the quality limitation

"The extract contained 2,140 heart failure patients across four sites. Ejection fraction was recorded for 68.7 percent overall, ranging from 41.0 percent at site B to 87.9 percent at site C. Patients without a recorded ejection fraction had 14.2 percent mortality against 9.6 percent among those with one, so complete-case analysis would understate mortality. Agreement between EHR and claims diagnosis was 88.3 percent with kappa 0.70, and the EHR phenotype had a PPV of 0.75 against claims. Sites B and D failed the prespecified completeness and plausibility gates and were excluded from the comparative analysis."

That paragraph gives the profile, the bias direction, the concordance, and the decision.

---

## 12. Exercises

### Exercise 1 — Apply the framework (recall, 25 minutes)

For five checks (a valid ICD code, an orphan encounter, a systolic BP of 400, a duplicate record, a discharge before admission), assign the Kahn category and subcategory.

Deliverable: a five-row table.

Acceptance criteria: each assignment is correct; each check is described as verification or validation.

### Exercise 2 — Kappa calculation (calculation, 40 minutes)

Given a 2×2 table with 300, 90, 60, and 1,550, compute observed agreement, expected agreement, kappa, PPV, and sensitivity against the second source.

Deliverable: the calculations with intermediate values.

Acceptance criteria: arithmetic correct to three decimals; the interpretation states whether the agreement is substantial.

### Exercise 3 — Missingness mechanism (application, 40 minutes)

For each of four missing fields, propose a mechanism (MCAR, MAR, MNAR, structural, workflow, device) and a test that would distinguish it from at least one alternative.

Deliverable: a four-row table with mechanism, alternative, and test.

Acceptance criteria: each test uses observable data; structural missingness is not conflated with MNAR.

### Exercise 4 — Completeness and bias (calculation, 45 minutes)

Using the counts in section 6.3, recompute the mortality rates and the bias from complete-case analysis. Then compute the bias if the excluded group's mortality were 11.0 percent rather than 14.2 percent.

Deliverable: two bias calculations and an interpretation.

Acceptance criteria: arithmetic correct; the interpretation states that the bias depends on the difference, not on the completeness alone.

### Exercise 5 — Composite sensitivity (calculation, 45 minutes)

Using the scores in section 6.6, compute the composite under weights 20, 20, 20, 20, 20 and 50, 20, 15, 10, 5. Report the number of sites passing at 3.5.

Deliverable: two sets of composites and a comparison.

Acceptance criteria: arithmetic correct; the conclusion states that the decision is weight-dependent.

### Exercise 6 — Quality report critique (critique, 60 minutes)

Find a published study that reports a data quality assessment. Identify which dimensions were assessed, which were not, and whether the assessment was stratified.

Deliverable: a 400-word critique with two proposed additions.

Acceptance criteria: the critique names the missing dimension and its likely bias; each proposal is feasible with the reported data.

### Exercise 7 — Quality report design (design, 75 minutes)

Design a one-page quality report template for an EHR extract: dimensions, strata, thresholds, reference standards, and decision rules.

Deliverable: a template with a worked example row for each dimension.

Acceptance criteria: every metric has a denominator and a stratum; every threshold has a rationale; the decision rule is explicit.

### Exercise 8 — Fitness decision (design, 45 minutes)

For a stated use (prevalence estimation), decide whether the four-site dataset in section 6 is fit, with gates and residual risks.

Deliverable: a decision with gates, excluded population, and residual risk.

Acceptance criteria: the decision differs from the readmission use if the requirements differ; the residual risk is quantified where possible.

### Exercise 9 — Integrity versus quality (critique, 30 minutes)

Give an example of a dataset with high integrity and poor fitness, and one with good fitness and weak integrity. Explain each.

Deliverable: two examples with explanation.

Acceptance criteria: the examples are distinct; the explanation uses the definitions in section 8.4.

### Exercise 10 — Small-cell policy (design, 30 minutes)

Write a small-cell suppression policy for a quality report with subgroup counts, and state its limits.

Deliverable: a policy with a threshold, a method, and a stated limitation.

Acceptance criteria: the threshold is justified; the limitation acknowledges residual risk.

---

## 13. Self-check answers

**A1. What are the three Kahn categories and their subcategories?**
Conformance (value, relational, computational), completeness (no subcategories), and plausibility (uniqueness, atemporal, temporal).

**A2. Why did Kahn reject "accuracy" as a category?**
Because the term is used inconsistently across the data quality and psychometric literature, which made cross-study comparison impossible. Plausibility was substituted because it is a judgment that can be tested and documented.

**A3. Compute observed agreement for 412, 138, 96, and 1,354.**
(412 + 1,354)/2,000 = 1,766/2,000 = 0.883.

**A4. Compute expected agreement for those marginals.**
Row margins 0.275 and 0.725; column margins 0.254 and 0.746. p_expected = 0.275×0.254 + 0.725×0.746 = 0.06985 + 0.54085 = 0.6107.

**A5. Compute kappa.**
(0.883 − 0.6107)/(1 − 0.6107) = 0.2723/0.3893 = 0.6995, or 0.70.

**A6. What is the PPV of the EHR phenotype against claims?**
412/550 = 0.749, or 74.9 percent.

**A7. Why does complete-case analysis understate mortality in the example?**
Because patients without a recorded ejection fraction had higher mortality (14.2 versus 9.6 percent), so excluding them removes a high-mortality group and lowers the observed rate.

**A8. Compute the bias from complete-case analysis against the full cohort.**
Full cohort 11.0 percent, complete case 9.6 percent, so the understatement is 1.4 percentage points.

**A9. Why can a composite score be misleading?**
Because its weights are assumptions, its scale is arbitrary, and aggregation allows a strong dimension to compensate for a failed critical one. The decision can reverse with plausible alternative weights.

**A10. What is the difference between concordance and accuracy?**
Concordance is agreement between two sources; accuracy is correspondence to truth. Two sources can agree and both be wrong.

**A11. What is structural missingness and why does it matter?**
Missingness because a field is not applicable (for example, a pregnancy field for a male patient). Counting it as missing inflates the missingness rate and can trigger unnecessary remediation.

**A12. Rubric for exercise 7 (quality report template).**
A strong template has a dimension, a metric with a defined denominator, a stratum, a threshold with a rationale, a reference standard where validation is possible, and an explicit decision rule. A weak template lists metrics without denominators or thresholds.

---

## 14. Interview questions

**Q1. How do you decide whether a dataset is fit for purpose?**
A strong answer: state the use and the critical elements, set thresholds per element, measure a stratified profile, validate against a reference where possible, and decide with gates plus residual risk. Weak answers quote a completeness number.

**Q2. What are the missingness mechanisms and why do they matter?**
MCAR, MAR, MNAR, plus structural, workflow, and device. The mechanism determines whether complete-case analysis is unbiased and which imputation method is defensible.

**Q3. Why is a single completeness number insufficient?**
Because the same number can describe uniform missingness, whole-record missingness, or one site failing one field. Only the pattern distinguishes a nuisance from a comparability problem.

**Q4. What is Cohen's kappa and what are its limits?**
A chance-corrected agreement statistic. It is sensitive to prevalence, and it measures agreement rather than accuracy, so it should be reported with marginals and with PPV or sensitivity against a reference.

**Q5. Why did Kahn reject accuracy as a data quality dimension?**
Because the term is used inconsistently and implies a truth that is usually unavailable. Plausibility is testable and documentable.

**Q6. How would you detect informative missingness?**
A strong answer: compare outcome rates between complete and incomplete records, and compare the metric immediately before and after a gap. Weak answers assume MCAR.

**Q7. What is the difference between data quality and data integrity?**
Quality is fitness for a use; integrity is the trustworthiness of the record over its lifecycle. A dataset can have one without the other.

**Q8. What would make you reject a site from a comparative analysis?**
A strong answer: failure of a critical gate, such as completeness of an adjustment variable below the threshold, or a plausibility failure that indicates a data generation problem. The exclusion and its population must be reported.

**Q9. How would you handle a composite quality score?**
A strong answer: report the components, apply gates for critical dimensions, and run a weight sensitivity analysis. Weak answers present the composite as an objective ranking.

**Q10. What is the role of validation versus verification?**
Verification checks against internal expectations and is cheap; validation checks against an external standard and is the only way to detect systematic error.

**Q11. How would you report quality alongside a result?**
A strong answer: a short paragraph next to the result with the profile, the bias direction, the concordance, and the excluded population. Weak answers put a sentence in an appendix.

**Q12. What are the limits of a quality assessment?**
Truth is usually unavailable, validation samples are small, mechanisms are inferred, quality changes, and fitness is use-specific.

---

## 15. Key terms

- **ALCOA+** — data integrity principles for regulated records.
- **Atemporal plausibility** — believability ignoring time.
- **Cohen's kappa** — chance-corrected agreement.
- **Completeness** — presence of required values.
- **Composite score** — a weighted combination of dimensions.
- **Concordance** — agreement between two sources.
- **Conformance** — adherence to standards and formats.
- **Currency** — how recent the data are.
- **Data integrity** — trustworthiness of a record over its lifecycle.
- **Data quality** — fitness of data for a use.
- **Fitness for purpose** — adequacy for a specific use.
- **Gold standard** — an assumed-correct reference.
- **MAR** — missing at random.
- **MCAR** — missing completely at random.
- **MNAR** — missing not at random.
- **Plausibility** — whether values are believable.
- **PPV** — positive predictive value.
- **Quality profile** — a multidimensional assessment.
- **Reference standard** — an external basis for validation.
- **Sensitivity** — proportion of true positives detected.
- **Small-cell risk** — re-identification risk from small counts.
- **Specificity** — proportion of true negatives detected.
- **Structural missingness** — absence because a field is inapplicable.
- **Temporal plausibility** — believability over time.
- **Timeliness** — whether data arrive when needed.
- **Uniqueness** — absence of duplicates.
- **Validation** — assessment against an external standard.
- **Verification** — assessment against internal expectations.
- **Weight sensitivity** — change in decision when weights change.
- **Workflow missingness** — absence due to process.

---

## 16. Sources

All URLs accessed 24 October 2026.

**Frameworks and methods**

- Kahn, M. G., Callahan, T. J., Barnard, J., et al. "A Harmonized Data Quality Assessment Terminology and Framework for the Secondary Use of Electronic Health Record Data." *eGEMs*, 2016;4(1):18. https://doi.org/10.13063/2327-9214.1244
- Open access full text: https://pmc.ncbi.nlm.nih.gov/articles/PMC5051581/
- Kahn, M. G., et al. "A Pragmatic Framework for Single-site and Multisite Data Quality Assessment in Electronic Health Record-based Clinical Research." *Medical Care*, 2012. https://pmc.ncbi.nlm.nih.gov/articles/PMC3833692/ (background)
- Reimer, A. P., Milinovich, A., and Madigan, E. A. "Data Quality Assessment Framework to Assess Electronic Medical Record Data for Use in Research." *International Journal of Medical Informatics*, 2016. https://pmc.ncbi.nlm.nih.gov/articles/PMC4845906/
- Wang, R. Y., and Strong, D. M. "Beyond accuracy: What data quality means to data consumers." *Journal of Management Information Systems*, 1996. (Background.)
- Weiskopf, N. G., and Weng, C. "Methods and dimensions of electronic health record data quality assessment: enabling reuse for clinical research." *JAMIA*, 2013. (Background.)

**Tools and programmes**

- OHDSI. *Data Quality Dashboard.* https://ohdsi.github.io/DataQualityDashboard/
- OHDSI. *Achilles.* https://ohdsi.github.io/Achilles/
- PhysioNet. *MIMIC-IV v3.1.* https://physionet.org/content/mimiciv/3.1/
- PhysioNet. *eICU Collaborative Research Database.* https://eicu-crd.mit.edu/

**Regulators**

- US Food and Drug Administration. *Real-World Evidence.* https://www.fda.gov/science-research/science-and-research-special-topics/real-world-evidence
- ICH. *E6(R2) Good Clinical Practice.* (Cite the guideline.)
- eCFR. *21 CFR Part 11.* (Cite the Part.)
- HHS Office for Civil Rights. *Guidance Regarding Methods for De-identification of Protected Health Information.* https://www.hhs.gov/hipaa/for-professionals/special-topics/de-identification/index.html

> **Note:** The Kahn framework's terminology is used here as published in 2016. It remains the most widely cited harmonized framework for EHR secondary use, and later work has extended rather than replaced it.

---

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`09_licensing-and-access.md`](09_licensing-and-access.md)
- First topic: [`01_ehr-data.md`](01_ehr-data.md)
- Related sibling: [`02_claims-and-billing.md`](02_claims-and-billing.md)
- Related sibling: [`03_registries-and-public-health.md`](03_registries-and-public-health.md)
- Related sibling: [`08_standards-and-terminologies.md`](08_standards-and-terminologies.md)
- Root survey file: [`../../02_biomedical_data_landscape.md`](../../course/02_biomedical_data_landscape.md)
- Labs: [`../../labs/`](../../labs/)
- Lab 2.5 in the survey file (data-quality profile) is the direct practical companion to this topic, and Lab 2.2 (OMOP cohort query) provides the data for the completeness and concordance exercises.

This topic closes the module. The through-line from topic 1 to here is that every data source can answer some questions and no source can answer all of them, and the quality profile is how you tell which is which for the decision in front of you.

The sentence to remember: quality is not a property of a dataset, it is a relationship between a dataset and a decision, and the relationship has to be documented.

### 11.3 The limit that decides whether a comparison is fair

For a cross-site comparison, the binding limit is whether the quality profile is similar across sites. A site with 41 percent completeness on an adjustment variable is not comparable to a site with 88 percent, even if both pass a lenient gate.

The practical statement compares the profiles, not only the rates: "completeness of the ejection fraction differed by 47 percentage points between sites, and the excluded patients' mortality differed by 4.6 points, so the cross-site comparison is reported descriptively and not as a ranking."

### 11.4 The limit that closes the module

Every topic in this module has ended with the same sentence in different words: the data can answer some questions and not others, and the boundary is a property of the source, not of the analysis.

The quality profile is how that boundary is measured rather than asserted. A deliverable that states its profile, its thresholds, its decision rule, and its residual risk is a deliverable a reader can act on.

### 11.5 The three numbers a quality statement needs

- The completeness of each critical element, by site.
- The outcome differential between complete and incomplete records.
- The validation result, with its sample size and reference standard.

A quality statement with those three is auditable. One that says "the data were of high quality" is not.

A final note on the module as a whole: the ten topics describe ten different producers, ten different failure mechanisms, and one shared discipline. That discipline is to state what the data are, what they can measure, what they cannot, and how the boundary was tested.

Everything else in this folder is detail in service of that sentence.

That sentence is also the module's industry question, and it is the one a data-source decision memo has to answer for every source it considers.

Four checks to carry into practice:

- Has the denominator been defined for every metric?
- Has the missingness been tested against the outcome?
- Has the threshold been justified against the decision?
- Has the residual risk been named?
