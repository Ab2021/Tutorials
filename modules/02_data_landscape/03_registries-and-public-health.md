# Registries and Public Health Data

> Module 02 — The Biomedical Data Landscape
> Topic 03 of 10
> Estimated study time: 4 hours
> Prerequisites: none within this module; Topic 01 is helpful
> Version 2026.3

---

## 1. Why this topic exists

Registries and public health systems are the data sources that exist because someone was legally required, professionally obligated, or financially incentivized to report. That origin gives them a property no EHR possesses: a defined denominator and a defined reporting population.

A cancer registry knows which patients in a defined geography were diagnosed with cancer, because hospitals and laboratories are required to report. A vital statistics system knows how many people died and from what coded cause, because death certificates are a legal document. An immunization information system knows which children in a state have received which vaccines, because providers report to it and because schools require it.

That property is worth a great deal. It is also the source of a specific class of error: a registry measures **what was reported under a mandate**, not **what happened**. The gap between the two is ascertainment, and ascertainment varies by geography, by facility type, by disease, and over time. An analyst who treats a registry as a complete enumeration will under-count disease and mis-rank providers.

The decisions this topic supports:

- Whether a registry can serve as the outcome source for a study, and what its completeness implies for the effect estimate.
- Whether a public health dataset is current enough for the decision at hand. Much of it is not.
- Whether a quality registry's benchmark is comparable across the reporting entities.
- Whether a surveillance signal reflects an outbreak or a change in reporting behaviour.

### 1.1 Who is blocked, and by what

| Role | Decision blocked | Failure without this topic |
|---|---|---|
| Epidemiologist | Whether incidence is rising | Confuses reporting completeness change with incidence change |
| Oncology RWE lead | Whether registry data can support an effectiveness claim | Ignores stage completeness and treatment capture |
| Public health analyst | Whether a weekly signal is actionable | Sums weekly counts that cannot be summed |
| Quality lead | Whether a hospital's benchmark is fair | Compares crude rates across case-mix differences |
| Regulatory strategist | Whether a registry endpoint is acceptable | Misunderstands the registry's ascertainment and lag |
| Policy analyst | Whether a program worked | Uses a post-mandate increase in reporting as evidence of effect |
| Data engineer | How to link registry and EHR data | Ignores the registry's own identifier and reporting history |

### 1.2 What is structurally different from EHR and claims

Three differences define the category:

1. **A mandate, not a transaction.** The record exists because reporting is required or incentivized, not because a bill was paid or a note was written. Mandates create completeness and also create reporting artefacts.
2. **A defined population.** The registry has an inclusion rule tied to geography, condition, procedure, or enrollment. The rule is usually stated, which is rare and valuable.
3. **A lag by design.** Case ascertainment, abstraction, deduplication, and reconciliation take time. A registry is almost never real-time, and the lag is part of its definition.

The third point is the one that most often disqualifies a registry for an operational decision and qualifies it for a research one.

---

## 2. Learning outcomes

By the end of this topic you can:

1. **Classify** a registry by its inclusion rule, its reporting mechanism, and its data sources, and state what the inclusion rule excludes.
2. **Estimate** registry completeness using capture-recapture from two overlapping sources, with correct arithmetic.
3. **Compute** a crude and an age-adjusted incidence rate and explain why the two can lead to opposite conclusions.
4. **Explain** why weekly surveillance counts cannot be summed to an annual total, and compute an annual estimate correctly.
5. **Distinguish** a reporting-mandate effect from a true change in disease occurrence using at least two diagnostics.
6. **Critique** a registry-based quality comparison for ascertainment bias, case-mix difference, and definition change.
7. **Choose** between a registry, an EHR, and a claims source for a stated outcome, and state the ascertainment implication of each.
8. **Design** a data provenance record for a registry extract that names the registry, the submission year, the data dictionary version, and the completeness estimate.

Mapping to assessment: outcomes 2, 3, and 5 are assessed by exercises 2, 3, and 5. Outcomes 1, 4, and 6 are assessed by the self-check items. Outcomes 7 and 8 are assessed by the design exercises and the module memo.

---

## 3. The core mental model

A registry is a **reporting funnel with a reconciliation step at the bottom**.

```text
   POPULATION AT RISK
   (geography, condition, procedure, enrollment)
              |
              v
   +----------------------+
   |  CASE OCCURRENCE     |   what actually happened
   +----------+-----------+
              |
     reporting |  (mandate, incentive, abstraction capacity)
              v
   +----------------------+
   |  CASE REPORTED       |   what reached the registry
   +----------+-----------+
              |
    dedup +   |  (linkage, merge, adjudication)
   reconcile  v
   +----------------------+
   |  CASE IN THE FILE    |   what the analyst receives
   +----------+-----------+
              |
   analysis   |  (definitions, exclusions, standardization)
              v
   +----------------------+
   |  CASE IN THE ESTIMATE|
   +----------------------+
```

Every arrow loses cases, and every arrow adds a bias. The four losses are:

- **Occurrence to reporting.** Cases diagnosed or treated outside the reporting network, cases not abstracted because of capacity, cases the patient declined to have reported.
- **Reporting to file.** Duplicates, late submissions not yet loaded, submissions rejected by validation.
- **File to estimate.** Exclusions the analyst applies, definitional mismatches, missing stage or treatment fields.
- **And separately, the population at risk is itself uncertain.** Census denominators are estimates, and intercensal estimates are revised.

### 3.1 The three registry families

| Family | Inclusion rule | Examples | Strength | Fundamental limit |
|---|---|---|---|---|
| Disease registry | Diagnosis of a specified condition in a defined population | Cancer registries, stroke registries, rare disease registries | Condition-specific depth, long follow-up | Ascertainment varies; treatment often incomplete |
| Procedure or device registry | Receipt of a specified procedure or device | Joint replacement, implant, transplant registries | Captures the exposure precisely | Loss to follow-up; voluntary participation |
| Quality registry | Participation by a provider or health system | Cardiac surgery, stroke care, ICU quality registries | Standardized definitions, benchmarking | Selection of participating sites; abstraction burden |

A fourth family, **public health surveillance**, is registry-like but differs in cadence and legal basis: notifiable disease reporting, syndromic surveillance, immunization information systems, and vital statistics. Surveillance prioritizes timeliness over completeness; registries prioritize completeness over timeliness. Confusing the two is a common error.

### 3.2 The completeness triangle

Three quantities are always in tension:

- **Completeness**: the proportion of true cases in the file.
- **Timeliness**: how recently the file reflects reality.
- **Granularity**: how much detail per case.

A system can optimize two at the expense of the third. Death certificates are complete and timely but have almost no clinical granularity. Cancer registries are complete and granular but lag by 18 to 24 months. Syndromic surveillance is timely and granular at the visit level but has poor case definition.

When someone asks for "complete, current, and detailed" data from a registry, the honest answer is that they have asked for three things and can have two.

### 3.3 The denominator problem

Registry rates need a denominator, and the denominator is usually a census estimate rather than a count. Three consequences follow:

1. **Denominator revisions change historical rates.** A census revision can move a published rate without any change in cases.
2. **Small-area denominators are unstable.** A rate for a county of 8,000 people has a wide interval even when the count is exact.
3. **The at-risk population may not be the residential population.** If cases are attributed to the treating facility's geography rather than the patient's residence, the denominator is wrong.

Any registry rate should state the denominator source and the year of the estimate.

---

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| Registry | A systematic collection of data for a defined population or condition | Used for any database | None |
| Ascertainment | The process of identifying cases for inclusion | Confused with completeness | Process |
| Completeness | The proportion of true cases present in the file | Assumed to be 100% | Proportion |
| Capture-recapture | An estimator of the true case count from overlapping sources | Applied without checking source independence | Count |
| Lincoln-Petersen estimator | The two-source capture-recapture formula | Used when sources are dependent | Count |
| Reporting mandate | A legal requirement to report cases | Assumed to guarantee completeness | Legal instrument |
| Notifiable disease | A condition that must be reported to public health authorities | Confused with all reportable conditions | Legal list |
| NNDSS | National Notifiable Diseases Surveillance System | Treated as a real-time count | CDC system |
| Provisional data | Counts subject to revision | Treated as final | Status |
| Final data | Counts reconciled with reporting jurisdictions | Assumed immutable | Status |
| NVSS | National Vital Statistics System | Treated as a clinical data source | NCHS system |
| Vital statistics | Birth, death, marriage, and divorce records | Confused with clinical records | Legal records |
| Death certificate | The legal record of death and coded cause | Treated as an accurate cause of death | Legal document |
| Underlying cause of death | The disease or injury that initiated the chain leading to death | Confused with the immediate cause | ICD-10 code |
| Multiple cause of death | All conditions on the death certificate | Ignored in favour of the underlying cause | ICD-10 codes |
| SEER | Surveillance, Epidemiology, and End Results program | Treated as a national census of cancer | NCI program |
| NPCR | National Program of Cancer Registries | Confused with SEER | CDC program |
| Case report form | The structured form used to abstract a case | Treated as the source record | Form |
| Abstraction | Manual extraction of data from records into a registry | Assumed to be error-free | Process |
| Stage | The extent of disease at diagnosis | Assumed complete in registries | AJCC TNM or summary stage |
| AJCC | American Joint Committee on Cancer | Treated as the only staging system | Staging system |
| ICD-O-3 | International Classification of Diseases for Oncology, third edition | Confused with ICD-10 | Site and morphology codes |
| Age adjustment | Standardizing rates to a reference age distribution | Confused with age-specific rates | Rate per 100,000 |
| Direct standardization | Applying a standard population's weights to stratum-specific rates | Applied without stating the standard | Rate per 100,000 |
| Crude rate | Events divided by population, unadjusted | Compared across populations with different age structures | Rate per 100,000 |
| Standard population | The reference age distribution used in adjustment | Assumed universal | Population |
| Reporting delay | Time between an event and its appearance in the data | Ignored at the end of a period | Days |
| Backfill | Retrospective addition of previously unreported cases | Mistaken for a new outbreak | Count |
| Case definition | The criteria for counting a case | Confused with a diagnosis | Definition |
| Syndromic surveillance | Surveillance using pre-diagnostic signals such as chief complaint | Treated as diagnosis-based | System |
| IIS | Immunization Information System | Assumed complete for all ages | State system |
| HEDIS | Healthcare Effectiveness Data and Information Set | Treated as a clinical measure set | NCQA measure set |
| MIPS | Merit-based Incentive Payment System | Confused with a registry | CMS program |
| Sentinel | FDA's active surveillance system | Treated as a registry | FDA program |

### 4.1 Terms that are routinely conflated

- **Completeness versus accuracy.** A registry can contain every case and code the stage wrongly. Completeness and accuracy are separate dimensions and require separate validation.
- **Reporting delay versus backfill.** Delay is the normal lag. Backfill is the retrospective addition of cases that should have been reported earlier. Backfill looks like an outbreak in a time series and is not one.
- **Underlying cause versus immediate cause.** The death certificate has a chain. The underlying cause is the start of the chain; the immediate cause is the end. Analyses that use the immediate cause will attribute deaths to the final mechanism rather than the disease.
- **Registry participation versus population coverage.** A quality registry may include 90 percent of eligible hospitals and still cover only 60 percent of the population, because the participating hospitals are not representative.

---

## 5. The mechanics, step by step

This section is the working procedure for evaluating and using a registry or public health dataset. It has 27 steps in five phases.

### Phase A — Establish what the registry is (steps 1-6)

**Step 1. Read the inclusion rule and write it in one sentence.**
Input: the registry's data dictionary or program documentation. Output: one sentence naming the population, the condition or procedure, the geography, and the period.
Reason: the inclusion rule defines the denominator and the generalizability. A registry of "patients treated at participating hospitals" is a different population from a registry of "all residents diagnosed in the state."
Test: if you cannot write the sentence without hedging, the rule is not documented well enough to use.

**Step 2. Identify the reporting mechanism and its legal basis.**
Input: program documentation. Output: a statement of whether reporting is mandatory, voluntary, or incentivized, and by what instrument.
Reason: mandatory reporting produces different completeness from voluntary reporting, and incentivized reporting produces different coding behaviour. The mechanism predicts the bias.
Rule: distinguish legal mandate (statute or regulation), payment incentive (quality program), professional obligation, and voluntary participation.

**Step 3. Enumerate the data sources that feed the registry.**
Input: program documentation and the data dictionary. Output: a list of source types (hospital abstract, laboratory report, physician report, death certificate, claims).
Reason: each source has its own ascertainment. A registry fed by hospital abstracts misses cases diagnosed in physician offices and never admitted.
Rule: for each source, note what it can and cannot see.

**Step 4. Determine the reporting unit and the case definition.**
Input: the data dictionary. Output: the unit (patient, tumour, procedure, admission) and the case definition.
Reason: cancer registries count tumours, not patients. A patient with two primaries contributes two records. Procedure registries count procedures, so a revision is a new record.
Rule: state whether the analysis unit is the person, the event, or the lesion, and whether the registry allows multiples.

**Step 5. Identify the data dictionary version and the coding systems.**
Input: the data dictionary. Output: the version, the code sets, and the date of the last change.
Reason: registry definitions change. A stage variable redefined in a given year creates a break that looks like a trend.
Rule: record the version and check the changelog for definitional changes in the study period.

**Step 6. Establish the submission cycle and the data cutoff.**
Input: program documentation. Output: the submission schedule and the last complete year.
Reason: registries operate on annual or semi-annual cycles. The most recent year in the file is usually incomplete.
Rule: never use the most recent year without checking its completeness against prior years.

### Phase B — Measure completeness (steps 7-11)

**Step 7. Obtain an independent second source for the same cases.**
Input: a hospital tumour registry, a claims extract, a death certificate file, or a clinical database. Output: a second case list with identifiers.
Reason: completeness cannot be measured from within a single source. It requires an external reference.
Rule: the second source must overlap the same population and period; otherwise the comparison is meaningless.

**Step 8. Link the two sources and classify the overlap.**
Input: two case lists. Output: counts in A only, B only, and both.
Reason: the overlap is the input to capture-recapture and the direct measure of agreement.
Rule: use a validated linkage method and report the match rate; a poor linkage understates the overlap and overstates the estimated total.

**Step 9. Apply a capture-recapture estimator and check its assumptions.**
Input: the three counts. Output: an estimated true count and a completeness estimate for each source.
Reason: the estimator assumes the sources are independent and that the population is closed. Both assumptions are usually violated, so the estimate is a bound rather than a point.
Rule: report the estimate with the assumption stated, and run a sensitivity analysis with a dependence assumption.

**Step 10. Stratify completeness by facility, geography, and year.**
Input: the linked dataset. Output: completeness by stratum.
Reason: aggregate completeness hides the pattern. A registry that is 95 percent complete overall can be 70 percent complete in rural facilities, which is exactly where a disparity analysis would look.
Rule: stratify by facility type, region, and year at minimum.

**Step 11. Estimate the effect of incomplete ascertainment on the planned analysis.**
Input: the completeness estimate. Output: a statement of the direction and approximate magnitude of bias.
Reason: incomplete ascertainment usually biases prevalence downward and can bias associations in either direction if completeness is related to the exposure.
Rule: if completeness differs by exposure group, the association is biased; if it is uniform, prevalence is biased but the association may be approximately preserved.

### Phase C — Build the analytic dataset (steps 12-17)

**Step 12. Resolve the analysis unit and deduplicate.**
Input: the registry extract. Output: one row per analysis unit.
Reason: duplicate reports are common when multiple facilities report the same case. The registry may or may not have merged them.
Rule: check for duplicates on identifiers plus dates plus site; report the duplicate rate.

**Step 13. Map registry codes to standard concepts.**
Input: ICD-O-3, ICD-10, or registry-specific codes. Output: standard concepts with an unmapped rate.
Reason: registry code sets are specialized. ICD-O-3 morphology codes have no direct equivalent in ICD-10-CM.
Rule: record the mapping and its loss; do not silently drop unmapped records.

**Step 14. Handle missing stage and treatment fields.**
Input: the registry extract. Output: a documented missingness rule per field.
Reason: stage and treatment are the fields most often incomplete, and they are often the fields the analysis needs.
Rule: report missingness by stage and by year; if missingness is related to survival, complete-case analysis is biased.

**Step 15. Build the follow-up and outcome variables.**
Input: registry follow-up fields and a death source. Output: survival time and vital status.
Reason: registries vary in how they obtain vital status. Some link to death certificates, some rely on active follow-up, some use both.
Rule: state the vital status source and the last follow-up date; treat losses to follow-up as censoring, not as alive.

**Step 16. Align the denominator.**
Input: census or enrollment estimates. Output: a person-time or population denominator matched to the case definition.
Reason: the numerator and denominator must cover the same population and period. A registry that includes non-residents paired with a resident population denominator produces a rate that is too high.
Rule: state the denominator source, the year, and whether it is a point estimate or an average.

**Step 17. Freeze and version the extract.**
Input: all rules. Output: a versioned dataset with row counts.
Reason: registries revise historical data. A re-run against a later submission changes the numbers.
Rule: record the submission year and the extract date.

### Phase D — Analyze with the right adjustments (steps 18-22)

**Step 18. Compute crude rates first.**
Input: counts and denominators. Output: crude rates with confidence intervals.
Reason: the crude rate is the transparent starting point and the number a stakeholder will recognize.
Rule: report the interval, especially for small areas.

**Step 19. Compute age-specific rates and inspect the pattern.**
Input: counts by age band and denominators by age band. Output: a rate table.
Reason: the age-specific pattern is the evidence for whether adjustment will matter. If the rates are similar across ages, adjustment changes little.
Rule: use bands that are wide enough to be stable and narrow enough to show the pattern.

**Step 20. Age-adjust and state the standard population.**
Input: age-specific rates and a standard population. Output: a directly standardized rate.
Reason: populations differ in age structure, and age is the strongest predictor of most disease rates. Adjustment removes that contribution.
Rule: name the standard (for example, the 2000 US standard population) and the weights.

**Step 21. Test for definitional and ascertainment breaks.**
Input: the time series. Output: a plot with candidate breakpoints annotated.
Reason: a change in the case definition, a new reporting requirement, or a new facility joining the registry produces a step change that is not epidemiologic.
Rule: annotate every known change on the plot before interpreting any trend.

**Step 22. Run a completeness sensitivity analysis.**
Input: the primary estimate. Output: the estimate under at least two completeness assumptions.
Reason: if the conclusion changes when completeness is 80 rather than 95 percent, the conclusion depends on an unmeasured quantity.
Rule: report both, and say which assumption the conclusion requires.

### Phase E — Document and deliver (steps 23-27)

**Step 23. Record the registry identity and version.**
Input: everything above. Output: a provenance record naming the registry, the submission year, the data dictionary version, and the extract date.
Reason: a registry analysis without a version cannot be reproduced, because the registry will revise the data.
Rule: include the record counts by year.

**Step 24. Document the linkage, if any.**
Input: the linkage method. Output: a statement of the method, the match rate, and the estimated error.
Reason: registry-to-EHR or registry-to-claims linkage is common and its error propagates.
Rule: report false-match and missed-match estimates if available.

**Step 25. State the lag explicitly.**
Input: the data cutoff. Output: a sentence in the methods stating the last complete period and the expected revision.
Reason: readers will otherwise assume the most recent year is complete.
Rule: give the month and year, not "recent."

**Step 26. Reconcile against a published total where one exists.**
Input: the registry's own published aggregate. Output: a reconciliation.
Reason: this catches definitional mismatches and filter errors faster than any other check.
Rule: reconcile case counts for at least one year and one subgroup.

**Step 27. State what the registry cannot answer.**
Input: the question. Output: a limits section naming treatment detail, comorbidity, and care after the reporting period.
Reason: registries are strong on diagnosis and weak on process.
Rule: name at least three limits and say which threatens the conclusion.

### 5.1 Field-level mechanics: cancer registry data

1. **Site and morphology.** ICD-O-3 codes the primary site and the histology. Two tumours with the same site and different morphology are different diseases for most analyses.
2. **Behaviour.** Benign, borderline, in situ, and malignant. Many analyses inadvertently include in situ cases.
3. **Multiple primaries.** Rules determine whether a second tumour is a new primary or a recurrence. The rules changed over time, which creates a break in the series.
4. **Stage.** AJCC TNM, summary stage, or both. Stage completeness varies by site and year, and stage definitions changed with the eighth edition.
5. **First course of treatment.** Surgery, radiation, chemotherapy, and hormonal therapy, recorded as whether given, not as regimen or duration.
6. **Vital status and survival.** Derived from follow-up and death linkage. The last follow-up date is essential for censoring.
7. **Reporting facility.** Which facility reported the case. Useful for ascertainment analysis and for detecting duplicate reports.

### 5.2 Field-level mechanics: notifiable disease surveillance

1. **Case classification.** Confirmed, probable, and suspected. The classification rules are condition-specific and change.
2. **Reporting jurisdiction.** The state or territory that reports. Aggregation across jurisdictions requires reconciling different practices.
3. **Event date versus report date.** The event date is when the case occurred; the report date is when it reached the system. Only the event date is epidemiologically meaningful.
4. **Weekly versus annual tables.** Weekly tables are provisional and cannot be summed. Annual tables are reconciled.
5. **Case definition version.** The Council of State and Territorial Epidemiologists (CSTE) maintains the definitions and revises them, sometimes annually.
6. **Condition list changes.** Conditions are added and removed. COVID-19 was removed from the nationally notifiable list, which changes the 2024 annual tables.

### 5.3 Field-level mechanics: vital statistics

1. **Underlying cause.** Coded with ICD-10 for mortality. The coding rules select one underlying cause from the certificate.
2. **Multiple causes.** Up to 20 conditions are coded. Analyses of comorbidities at death use these.
3. **Place of death and residence.** Different geographies. Rates by residence are the standard; rates by place of death answer a different question.
4. **Race and ethnicity.** Reported by the informant or the funeral director, not self-reported. This matters for disparity analyses.
5. **Age at death.** Recorded in years, with special handling for infants under one year.
6. **Revision of the code set.** A new ICD revision creates a discontinuity in cause-specific trends that is not a real change in mortality.

### 5.5 Quantitative checks to run inside the procedure

Run these after step 17 and before step 18. Each catches a class of error that is expensive to find later.

| Check | Computation | Passing signal | What failure usually means |
|---|---|---|---|
| Duplicate rate | Records per unique case identifier | Low and stable | Multiple facilities reporting the same case |
| Year-on-year count change | Cases by diagnosis year | Smooth except at known changes | Incomplete final year or a real trend |
| Facility contribution | Share of cases by facility | Stable over time | A facility joined or left |
| Stage completeness | Percent with a valid stage by year | Stable, above 90% for recent years | Abstraction backlog or definition change |
| Vital status completeness | Percent with a known vital status | Above 95% | Follow-up loss |
| Denominator consistency | Population by year versus census | Within 2% of the estimate | Wrong denominator vintage |
| Rate plausibility | Rate versus neighbouring regions | Same order of magnitude | Numerator or denominator error |
| Code distribution | Top 20 codes by year | Stable | Code set change or mapping error |

Two deserve emphasis.

**Facility contribution over time.** A registry's case count is a sum over reporters. A single large facility joining can raise the count by several percent in one year, and that increase is ascertainment, not incidence. Plotting each facility's contribution separately is the fastest way to see it.

**Stage completeness by year.** Stage is the field most often used and most often missing. A decline in stage completeness in the most recent years is normal (abstraction backlog), and it means that any stage-specific analysis of recent years is biased toward the cases that were abstracted first, which are usually the simpler ones.

### 5.6 Derivations and the assumptions they hide

| Derived variable | Usual construction | Hidden assumption | Challenge question |
|---|---|---|---|
| Incidence rate | Cases divided by population-years | The denominator covers the same population | Are non-resident cases included? |
| Survival | Diagnosis date to death or last follow-up | Vital status is complete | What is the loss-to-follow-up rate? |
| Stage-specific survival | Survival within stage | Stage is missing at random | Does survival differ by missingness? |
| Completeness | Observed divided by estimated true | The estimate is valid | How was the estimate made? |
| Age-adjusted rate | Weighted sum of age-specific rates | The standard population is appropriate | Which standard, and why? |
| Time trend | Rate by year | The case definition is stable | What changed in the period? |
| Facility comparison | Rate by facility | Case mix is comparable | Was risk adjustment applied? |

### 5.7 Handling the lag without discarding the recent period

The most common practical decision is what to do with the most recent year. Three options, each with a cost:

1. **Exclude it.** The cleanest choice. The series ends at the last complete year, and the report says so. The cost is that the analysis may be 18 months stale by publication.
2. **Include it with a flag.** Keep the year but mark it incomplete in every table and plot, and do not compute a trend across the boundary. The cost is that a reader may still use the number.
3. **Model the lag.** Estimate the expected final count from the historical reporting curve and present the estimate with an interval. The cost is added assumptions, and the model itself needs validation.

Option 3 is defensible only when the historical reporting curve is stable. If the registry changed its submission process, the curve is not stable and the model is extrapolating through a structural break.

A useful diagnostic is the **completeness ratio**: cases reported for year Y as of submission S, divided by cases reported for year Y as of the final submission. Computing this ratio for several historical years shows how much backfill to expect, and whether to expect it at all.

Example: for diagnosis year 2021, the submission-year-2022 file contained 92 percent of the final count; the submission-year-2023 file contained 98 percent; the final file contained 100 percent. For 2024, the submission-year-2025 file contains 88 percent. The expected final count is the observed count divided by 0.88, with an interval informed by the historical variation. That is a defensible estimate, and it is explicitly an estimate.

The same logic applies to surveillance. A weekly provisional count is not a random sample of the eventual annual count; it is the count as of a reporting date. Treating it as a proportion requires knowing the reporting curve, and the curve changes when the reporting process changes.

Four questions decide which lag option is defensible:

- Has the registry's submission process changed in the study period? If yes, option 3 is unavailable.
- Is the historical completeness ratio stable within a few percentage points? If yes, option 3 is reasonable.
- Does the decision depend on the most recent year specifically, or on the trend? If the latter, option 1 is usually acceptable.
- Will a reader be able to tell that the final year is incomplete? If not, option 2 is unsafe regardless of the analysis.

A final check applies to all three options: the completeness ratio must be computed on the same condition and the same facility mix as the year being corrected. A ratio derived from all cancer sites applied to one rare site will over- or under-correct, because reporting completeness varies by site.

If that ratio is unavailable, the honest presentation is the observed count with an explicit statement that it is a lower bound, and a note on the expected revision.

### 5.8 Branch points summarized

| Branch | Condition | Path A | Path B |
|---|---|---|---|
| Completeness measurement | Two independent sources available | Capture-recapture | External benchmark or expert estimate |
| Source dependence | Sources share a facility | Report as upper or lower bound | Apply a dependence model |
| Age adjustment | Populations differ in age structure | Direct standardization | Age-specific reporting |
| Recent year | Last year is incomplete | Exclude or model the lag | Include with an explicit caveat |
| Unit of analysis | Multiple primaries possible | Person-level with a rule | Lesion-level |
| Missing stage | Missingness related to survival | Multiple imputation or bounding | Complete case with a bias statement |

---

## 6. Worked example

### 6.1 The question

A state cancer registry and a university hospital's tumour registry both cover a metropolitan county. A health department analyst is asked: what is the true incidence of colorectal cancer in the county, and how complete is the state registry?

Data: state registry extract for 2022-2024, submission year 2025, data dictionary version current at extraction; hospital tumour registry for the same period; county population estimates from the Census Bureau's Vintage 2024 series.

### 6.2 Capture-recapture

| Source combination | Count |
|---|---:|
| State registry only | 495 |
| Hospital registry only | 719 |
| Both | 4,317 |
| State registry total (A) | 4,812 |
| Hospital registry total (B) | 5,036 |

Check: 4,812 − 4,317 = 495. 5,036 − 4,317 = 719.

Lincoln-Petersen estimate of the true count:

N = (A × B) / M, where M is the overlap.

- A × B = 4,812 × 5,036 = 24,233,232.
- N = 24,233,232 / 4,317 = 5,613.4.

Completeness:

- State registry: 4,812 / 5,613.4 = 0.8573, or 85.7%.
- Hospital registry: 5,036 / 5,613.4 = 0.8971, or 89.7%.
- Cases in neither source: 5,613.4 − 4,812 − 719 = 82.4, or about 82 cases.

Check: 4,812 + 719 = 5,531. 5,613.4 − 5,531 = 82.4.

Interpretation: about 14 percent of colorectal cancers in the county are missing from the state registry, and about 1.5 percent are missing from both sources.

### 6.3 The independence assumption

The estimator assumes the two sources ascertain cases independently. They do not. The hospital reports to the state registry, so a case in the hospital registry is more likely to be in the state registry than a case ascertained elsewhere. This positive dependence means the observed overlap is larger than independence would predict, so the Lincoln-Petersen estimate is **too low**: the true count is higher than 5,613 and true completeness is lower than 85.7 percent.

The direction of the bias is knowable even when the magnitude is not. That is enough to state the finding correctly: "at least 14 percent missing, and probably more, because the sources are positively dependent."

A Chapman estimator, which corrects the small-sample bias, gives:

N = [(A + 1)(B + 1) / (M + 1)] − 1 = (4,813 × 5,037 / 4,318) − 1.

- 4,813 × 5,037 = 24,243,081.
- 24,243,081 / 4,318 = 5,614.4.
- N = 5,613.4.

The Chapman correction changes the estimate by less than one case here, because the counts are large. It matters for small counts, which is exactly where registries are often used.

### 6.4 Crude incidence

County population, 2024 estimate: 1,240,000. Cases over three years: 4,812.

Crude annual rate = 4,812 / (3 × 1,240,000) × 100,000 = 4,812 / 3,720,000 × 100,000 = 129.35 per 100,000 per year.

Check: 4,812 / 3,720,000 = 0.00129355. × 100,000 = 129.35.

### 6.5 Age-specific and age-adjusted rates

| Age band | Population | Cases (3 years) | Annual rate per 100,000 | Standard weight |
|---|---:|---:|---:|---:|
| Under 55 | 806,000 | 1,140 | 47.15 | 0.72 |
| 55-74 | 310,000 | 2,180 | 234.41 | 0.21 |
| 75 and older | 124,000 | 1,492 | 401.08 | 0.07 |

Check the populations: 806,000 + 310,000 + 124,000 = 1,240,000. Check the cases: 1,140 + 2,180 + 1,492 = 4,812.
Check one rate: 1,140 / (3 × 806,000) × 100,000 = 1,140 / 2,418,000 × 100,000 = 47.15.

Age-adjusted rate = 0.72 × 47.15 + 0.21 × 234.41 + 0.07 × 401.08
= 33.948 + 49.226 + 28.076
= 111.25 per 100,000 per year.

| Measure | Rate per 100,000 per year |
|---|---:|
| County crude | 129.35 |
| County age-adjusted | 111.25 |
| State age-adjusted | 118.40 |

The county has a higher crude rate than the state (129.35 versus 118.40) but a lower age-adjusted rate (111.25 versus 118.40). The county's population is older, and the crude comparison reflected age structure rather than risk. A press release based on the crude rate would have reported the opposite conclusion from the one the adjusted rate supports.

This is a different mechanism from the hospital standardization example in topic 1: there, case mix differed across hospitals within one system; here, it differs between a county and a state. The arithmetic is the same and the policy consequence is the same, which is why the method is worth learning once and applying everywhere.

### 6.6 Combining the completeness and age adjustments

The age-adjusted rate of 111.25 uses observed cases. If the true case count is 5,613 rather than 4,812, the completeness correction factor is:

5,613.4 / 4,812 = 1.1665.

Applying it to the age-adjusted rate gives 111.25 × 1.1665 = 129.78 per 100,000.

| Adjustment | Rate per 100,000 |
|---|---:|
| Crude, observed | 129.35 |
| Age-adjusted, observed | 111.25 |
| Age-adjusted, completeness-corrected | 129.78 |
| Crude, completeness-corrected | 150.89 |

Check the last row: 4,812 × 1.1665 = 5,613.2; 5,613.2 / 3,720,000 × 100,000 = 150.89.

The two adjustments partly offset: age adjustment lowers the rate by 18.1 per 100,000, and the completeness correction raises it by 18.5 per 100,000. The net effect is that the fully corrected rate is almost identical to the crude observed rate, by coincidence.

That coincidence is the most dangerous result in the example. An analyst who reports the crude observed rate arrives at nearly the right number for the wrong reasons, and will be wrong the moment either adjustment changes. Reporting both adjustments separately, and the corrected rate with an interval that reflects the uncertainty in completeness, is the only defensible presentation.

### 6.7 Uncertainty

The capture-recapture estimate has its own uncertainty. For the Lincoln-Petersen estimator, an approximate variance is:

Var(N) = (A × B × (A − M) × (B − M)) / M³.

- A × B = 24,233,232.
- A − M = 495.
- B − M = 719.
- M³ = 4,317³ = 80,453,723,013.
- Numerator = 24,233,232 × 495 × 719 ≈ 8.628 × 10^12.
- Var ≈ 8.628 × 10^12 / 8.045 × 10^10 = 107.2.
- Standard error ≈ 10.4 cases.

So the estimate is 5,613 ± 20 (approximate 95 percent interval), before accounting for dependence. The dependence bias is larger than the sampling error, which is the normal situation in capture-recapture: the assumptions dominate the statistics.

Reporting "5,613 (95% CI 5,593-5,634), sensitivity analysis allowing positive dependence suggests 5,800 or more" is honest. Reporting "5,613" alone is not.

### 6.8 What this example establishes

- Completeness is measurable, and the measurement requires a second source.
- Capture-recapture gives a bound, not a truth, because the independence assumption fails in exactly the direction that matters.
- Age adjustment can reverse a comparison, and the reversal is a substantive finding rather than a technicality.
- Two adjustments can offset, producing a right answer for the wrong reason.

### 6.9 A note on the denominator vintage

The county population of 1,240,000 is a Vintage 2024 estimate for 2024, applied to all three years of cases. That is a simplification. If the population grew by 1.2 percent per year over the period, the person-years denominator is overstated for 2022 and understated for 2024, and the three-year average rate is biased slightly downward.

Correcting it: if the 2022 population was 1,211,000 and the 2024 population is 1,240,000, the average population over the period is approximately 1,225,500. Using that average:

4,812 / (3 × 1,225,500) × 100,000 = 4,812 / 3,676,500 × 100,000 = 130.89 per 100,000.

The difference from 129.35 is 1.54 per 100,000, or about 1.2 percent. Small, and larger than the sampling error of the count. Denominator vintage is a real source of error that analysts routinely ignore because it is invisible in the data.

### 6.10 What a reviewer would ask

Four questions, and the answers that hold:

1. **Why capture-recapture rather than a published completeness figure?** Because no published figure exists for this county, and the two-source estimate is local and checkable. The published national figure would be a weaker assumption.
2. **Why is 2000 US standard the right standard?** Because it is the convention in US cancer reporting and makes the county comparable to published state and national rates. A different standard would give a different number and break comparability.
3. **How do you know the hospital registry is independent?** You do not. That is why the estimate is presented as a lower bound and why the dependence sensitivity is included.
4. **Why report a corrected rate at all if it is uncertain?** Because the uncorrected rate is also uncertain, and its uncertainty is hidden rather than stated. Reporting both makes the assumption visible.

---

## 7. Code, query, or calculation

### 7.1 Python: capture-recapture and age adjustment

Assumptions: Python 3.11 or later; no external packages required.

```python
# Capture-recapture completeness and direct age standardization.

A = 4_812          # state registry total
B = 5_036          # hospital registry total
M = 4_317          # overlap

# Lincoln-Petersen
N_lp = (A * B) / M
# Chapman small-sample correction
N_ch = ((A + 1) * (B + 1) / (M + 1)) - 1

# Approximate variance of the Lincoln-Petersen estimator
var = (A * B * (A - M) * (B - M)) / (M ** 3)
se = var ** 0.5

print(f"Lincoln-Petersen N = {N_lp:,.1f}")
print(f"Chapman N          = {N_ch:,.1f}")
print(f"Approx SE          = {se:.1f}")
print(f"95% interval       = {N_lp - 1.96*se:,.0f} to {N_lp + 1.96*se:,.0f}")
print(f"State completeness = {A/N_lp*100:.1f}%")
print(f"Hospital completeness = {B/N_lp*100:.1f}%")
print(f"Missing from both  = {N_lp - A - (B - M):,.0f}")
print()

# Direct age standardization to the 2000 US standard population.
bands = [
    # label, population, cases over 3 years, standard weight
    ("under 55", 806_000, 1_140, 0.72),
    ("55-74",    310_000, 2_180, 0.21),
    ("75+",      124_000, 1_492, 0.07),
]
total_pop = sum(b[1] for b in bands)
total_cases = sum(b[2] for b in bands)
crude = total_cases / (3 * total_pop) * 100_000

adj = 0.0
for label, pop, cases, w in bands:
    rate = cases / (3 * pop) * 100_000
    adj += w * rate
    print(f"{label:>9}: n={cases:5d} pop={pop:7,d} rate={rate:7.2f} w={w:.2f}")

print(f"\nCrude rate        = {crude:,.2f} per 100,000 per year")
print(f"Age-adjusted rate = {adj:,.2f} per 100,000 per year")
factor = N_lp / total_cases
print(f"Completeness factor = {factor:.4f}")
print(f"Adjusted and corrected = {adj*factor:,.2f}")
```

Expected output:

```text
Lincoln-Petersen N = 5,613.4
Chapman N          = 5,613.4
Approx SE          = 10.4
95% interval       = 5,593 to 5,634
State completeness = 85.7%
Hospital completeness = 89.7%
Missing from both  = 82

   under 55: n= 1140 pop=806,000 rate=  47.15 w=0.72
      55-74: n= 2180 pop=310,000 rate= 234.41 w=0.21
        75+: n= 1492 pop=124,000 rate= 401.08 w=0.07

Crude rate        = 129.35 per 100,000 per year
Age-adjusted rate = 111.25 per 100,000 per year
Completeness factor = 1.1665
Adjusted and corrected = 129.78
```

### 7.2 SQL: completeness by facility and year

Assumptions: PostgreSQL; `registry_case` at one row per case; `facility` with a facility type; `expected` table from a prior completeness study.

```sql
-- Case counts and apparent completeness by facility type and diagnosis year.
WITH counts AS (
    SELECT f.facility_type,
           r.dx_year,
           COUNT(*) AS observed
    FROM registry_case r
    JOIN facility f USING (facility_id)
    WHERE r.dx_year BETWEEN 2020 AND 2024
    GROUP BY f.facility_type, r.dx_year
),
expected AS (
    SELECT facility_type, dx_year, expected_cases
    FROM completeness_expectation
)
SELECT c.facility_type,
       c.dx_year,
       c.observed,
       e.expected_cases,
       ROUND(100.0 * c.observed / NULLIF(e.expected_cases, 0), 1) AS pct_complete
FROM counts c
LEFT JOIN expected e USING (facility_type, dx_year)
ORDER BY c.facility_type, c.dx_year;
```

Expected output shape:

```text
facility_type | dx_year | observed | expected_cases | pct_complete
--------------+---------+----------+----------------+--------------
Academic      |    2020 |     1240 |           1290 |         96.1
Academic      |    2024 |     1310 |           1350 |         97.0
Community     |    2020 |      980 |           1180 |         83.1
Community |    2024 | 1010 | 1240 | 81.5
```

The point of the stratification is the last row: community facilities are about 15 percentage points less complete than academic facilities, and the gap is stable. A county-level rate that pools them will understate incidence in the communities served by community facilities.

### 7.3 A hand calculation for an interview

Given A = 4,812, B = 5,036, M = 4,317, compute completeness without a computer:

1. A × B ≈ 4,800 × 5,000 = 24,000,000.
2. Divide by 4,317: 24,000,000 / 4,317 ≈ 5,560.
3. Refine with the exact numerator: 24,233,232 / 4,317 ≈ 5,613.
4. Completeness = 4,812 / 5,613 ≈ 0.857, or about 86 percent.

The approximation is within 1 percent, which is enough to decide whether the registry is usable. Precision comes later.

---

## 8. Data, standards, and regulatory context

### 8.1 The US public health data infrastructure

| System | What it collects | Legal basis | Cadence | Access |
|---|---|---|---|---|
| SEER | Cancer incidence and survival from 17 core registries | NCI program; state reporting mandates | Annual submission | Research data request; SEER*Stat |
| NPCR | Cancer incidence from state registries | CDC program under the National Cancer Act amendments | Annual | State and CDC data products |
| NNDSS | Notifiable infectious diseases | State laws; CSTE definitions | Weekly and annual | CDC WONDER; data.cdc.gov |
| NVSS | Births and deaths | State vital statistics laws | Monthly and annual | NCHS; CDC WONDER |
| IIS | Immunizations | State laws and school requirements | Continuous | State IIS; some regional exchange |
| Syndromic surveillance | ED and urgent care visits | State and federal | Daily to weekly | State health departments |
| Behavioral Risk Factor Surveillance System (BRFSS) | Self-reported health behaviours | CDC, state health departments | Annual | Public microdata |
| National Health and Nutrition Examination Survey (NHANES) | Physical exams, labs, questionnaires | CDC/NCHS | Continuous, released in cycles | Public microdata |

### 8.2 SEER: what the numbers mean

SEER's own published metrics for 2024 describe 17 core registries covering about 48 percent of the US population and collecting roughly 950,000 cancer cases per year. The SEER*Stat documentation for the November 2024 submission gives more precise coverage figures that depend on which database is used:

| Database | Coverage of US population | Period |
|---|---:|---|
| SEER 8 | about 8.3% | 1975-2022 |
| SEER 12 | about 12.2% | 1992-2022 |
| SEER 17 | about 26.5% | 2000-2022 |
| SEER 21 (limited-field) | about 47.9% | 2000-2022 |

The differences matter because a rate computed on SEER 9 and a rate computed on SEER 21 are not the same population. Always name the database, not just "SEER."

The SEER overview page reports approximately 45.9 percent population coverage, with coverage varying substantially by group: 39.6 percent of White residents, 43.5 percent of Black residents, 64.9 percent of Hispanic residents, 59.3 percent of American Indian and Alaska Native residents, 68.2 percent of Asian residents, and 69.9 percent of Native Hawaiian and Pacific Islander residents. Those numbers describe a registry system that over-samples some groups relative to their share of the population, which is a design choice with analytic consequences: a national estimate that weights SEER cases as if they were representative will mis-weight every subgroup.

### 8.3 NNDSS: why weekly counts cannot be summed

CDC's NNDSS documentation states the constraint plainly: weekly tables are provisional, counts can rise or fall as cases are added or deleted after investigation, and weekly totals cannot be summed to produce an annual total. Delayed reports are folded into the cumulative year-to-date figure rather than into the week they arrive.

The practical consequences:

- An annual estimate must come from the annual table, not from summing weekly tables.
- A weekly dip may reflect a reporting interruption, not a decline in disease.
- Final annual data are published roughly 10 months after the year ends. The 2024 annual NNDSS data would therefore be expected around October 2025.
- Condition lists change. COVID-19 is not nationally notifiable as of 2025, so it is excluded from the 2024 annual NNDSS tables. Invasive Cronobacter was added in 2024; Zika without congenital disease was removed; mpox confirmed and probable cases were shown.

An analyst who builds a time series from weekly files without reading the readers' guide will produce a chart with artificial spikes and an annual total that is wrong by a double-digit percentage.

### 8.4 Vital statistics and cause of death

Death certificates are legal documents completed by a physician, medical examiner, or coroner, and coded by a vital records office using ICD-10. The underlying cause of death is selected by coding rules from the chain of conditions on the certificate.

Two limitations are structural:

1. **Cause attribution accuracy varies by condition.** Deaths from some conditions are well coded; deaths from others, particularly in older adults with multiple conditions, are coded to a mechanism rather than the underlying disease.
2. **Race and ethnicity are informant-reported.** The funeral director or informant supplies them, which produces misclassification relative to self-report and affects disparity analyses.

### 8.5 What is binding and what is convention

Binding in the US:

- **State reporting mandates.** Cancer, notifiable disease, and vital event reporting are required by state law, with penalties that vary. The federal role is funding and standardization, not a national mandate for most conditions.
- **HIPAA public health exception.** 45 CFR 164.512(b) permits disclosures to public health authorities for specified purposes without patient authorization. This is why registries can operate.
- **CMS quality reporting.** Hospitals report to the Inpatient Quality Reporting program and clinicians to MIPS. Registry participation is one pathway.
- **21st Century Cures and information blocking.** These affect access to EHR data that feeds registries, not the registries themselves.

Convention:

- Using the 2000 US standard population for age adjustment is a convention. Other standards exist and produce different adjusted rates.
- Reporting completeness estimates in registry publications is a convention that is unevenly followed.
- The 10-month publication lag for NNDSS annual data is administrative practice, not a rule.

### 8.6 European context

The European Health Data Space Regulation (EU) 2025/327, published 5 March 2025 and in force from 26 March 2025, creates a framework for secondary use of health data including registry and cohort data. Chapter IV, on secondary use, applies from 26 March 2029, with some provisions staged to 2031 and 2035. Access runs through health data access bodies under a data permit, processing occurs in secure processing environments, and uses are limited to enumerated purposes.

For a registry holder, this changes the access pathway rather than the registry's own collection rules. For an analyst, it means that cross-border registry analyses in the EU will increasingly run inside approved environments rather than on locally downloaded files.

> **Note:** Regulatory statements here are educational and not legal advice. Registry obligations are set primarily by state and national law and require local counsel.

---

## 9. Industry practice and case evidence

### 9.1 How much SEER is used

SEER's 2024 annual usage metrics report about 25,000 publications using SEER for primary analysis, 16,354 research data requests, and 362,501 SEER*Stat analyses in the year. Those numbers establish SEER as the default US cancer incidence source for research, and they explain why its idiosyncrasies become community-wide conventions.

The practical consequence, stated as inference: a methods choice made for SEER (for example, a particular age-adjustment standard or a particular exclusion of in-situ cases) propagates into the literature as if it were a clinical standard. When you replicate a SEER study on a different registry, you are replicating both the epidemiology and the SEER-specific conventions.

### 9.2 Registry-based quality measurement

Cardiac surgery registries and stroke registries are the clearest examples of registries used for public reporting and payment. The pattern is consistent across them:

- Participation is often voluntary but becomes effectively mandatory because payers and regulators use the benchmark.
- Definitions are standardized and audited, which improves comparability relative to claims.
- Risk models are used to adjust for case mix, and the risk model version matters as much as the data.
- Public reporting creates pressure on documentation and on case selection, which is the registry analogue of upcoding.

The last point is documented in the quality-measurement literature: when outcomes are publicly reported, providers may avoid high-risk cases or document risk factors more completely. Both responses change the data and neither is fraud. An analyst comparing a public-report era to a pre-report era is comparing two data-generating processes.

### 9.3 Registries as trial infrastructure

Registry-based randomized trials use an existing registry as the enrollment and follow-up mechanism, which reduces cost and increases generalizability relative to a conventional trial. The trade-off is that the registry must capture the outcome with sufficient completeness, and the consent and regulatory pathway must fit the registry's legal basis.

The design lesson for a data analyst is that a registry's fitness depends on the specific endpoint. A registry that captures mortality well may capture reintervention poorly, and a trial powered on the former can fail on the latter.

### 9.4 Implant and device registries

Device registries illustrate the exposure-capture strength of the registry model. Because the registry records the device and the procedure, it can answer questions that claims cannot: which implant, which lot, which revision.

Their weakness is follow-up. A patient who moves, changes provider, or dies outside the registry's catchment disappears. Revision rates computed from registry follow-up are therefore lower bounds unless the registry links to death records and to a national hospital dataset.

The transparency problem is also real: registry reports rarely state the loss-to-follow-up rate prominently, and a revision rate without that denominator is not interpretable.

### 9.5 Public health data in industry practice

Three uses are common in industry and each has a characteristic failure:

1. **Burden estimation for a market model.** Uses prevalence and incidence from registries and surveys. The failure is mixing sources with different case definitions and different years, producing a prevalence that is internally inconsistent.
2. **Site selection for a trial.** Uses registry case counts by facility. The failure is counting all cases rather than eligible cases, and ignoring the lag between diagnosis and registry entry.
3. **Safety or epidemiology context.** Uses NNDSS or vital statistics as a background rate. The failure is summing weekly counts, or comparing a provisional year to a final year.

None of these failures requires bad intent. Each requires knowing the data's cadence and definition.

### 9.6 What a mature registry user does

- **Names the database, not the program.** "SEER 21, November 2024 submission" rather than "SEER."
- **States completeness and how it was estimated.** A number and a method.
- **States the standard population** for every adjusted rate.
- **Annotates definitional changes** on every time-series plot.
- **Reports the lag** and the last complete period.

These five habits are the difference between a registry analysis that survives review and one that does not.

### 9.7 How registry data enter a commercial decision

Three commercial decisions routinely depend on registry data, and each has a characteristic failure mode:

| Decision | Registry input | Characteristic failure |
|---|---|---|
| Market sizing | Prevalence and incidence | Mixing sources with different case definitions and years |
| Trial feasibility | Cases by site and year | Counting all cases rather than eligible ones |
| Payer coverage policy | Outcomes and utilization by stage | Using stage-incomplete data as if it were complete |

In each case the failure is definitional, not statistical. A statistically elegant model built on a prevalence that mixes three case definitions is still wrong, and the elegance makes it harder to challenge.

The defence is a source table that names, for every input, the registry, the year, the definition, and the completeness. That table takes an hour to build and prevents the most expensive class of error in commercial epidemiology.

### 9.8 Why public health data rarely support a commercial claim on their own

Public health datasets are aggregated, lagged, and measured at a geographic unit that rarely matches a commercial population. They are excellent for context, for background rates, and for equity analysis. They are poor for the specific, recent, individual-level estimates that a commercial decision usually requires.

A useful rule: use public health data to frame the problem and registry or clinical data to answer it. A market model that rests entirely on public aggregates will be challenged on relevance before it is challenged on method.

---

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Registry treated as complete | Incidence below published national estimates | Unmeasured under-ascertainment | Understated burden; biased associations | Capture-recapture; external benchmark |
| Weekly counts summed | Annual total differs from the published annual table | Delayed reports fold into year-to-date | Wrong annual estimate; artificial spikes | Use annual tables |
| Provisional year treated as final | Last year's count is 30% below trend | Submission cycle incomplete | Spurious decline | Exclude or model the lag |
| Crude rates compared | Conclusion reverses after adjustment | Age structure differs | Wrong policy conclusion | Direct standardization |
| Independence assumed in capture-recapture | Completeness estimate is suspiciously high | Sources are positively dependent | True count under-estimated | Report as a bound; dependence sensitivity |
| Definition change read as trend | Step change at a known revision date | Case definition or code set changed | False epidemiologic signal | Annotate changes; split the series |
| Unit of analysis wrong | Patient counts differ from case counts | Multiple primaries or procedures | Double counting | State and enforce the unit |
| Stage missingness ignored | Survival differs by stage-missing status | Missingness related to outcome | Biased survival estimates | Multiple imputation; bounding |
| Denominator mismatched | Rate implausibly high in border counties | Numerator includes non-residents | Rate inflation | Match numerator and denominator geography |
| Reporting incentive read as disease | Diagnoses rise after a payment change | Documentation response to incentive | False severity trend | Compare to a non-incentivized source |
| Loss to follow-up ignored | Revision rate below clinical expectation | Patients leave the registry catchment | Understated event rate | Link to death and hospital data; report follow-up |
| Cause of death taken literally | Cause distribution implausible for the age group | Coding to mechanism rather than underlying cause | Misattributed mortality | Use multiple cause; validate against medical records |

### 10.1 Three diagnostics for a reporting artefact

When a registry time series shows a change, run these before interpreting it epidemiologically:

1. **Check the revision history.** Did the case definition, the code set, or the reporting requirement change at that date? Registry changelogs answer this in minutes.
2. **Check the facility mix.** Did a large facility join or leave the registry? A new reporter produces a step up that is ascertainment, not incidence.
3. **Check the lag profile.** Is the change confined to the most recent periods? That is a submission artefact, not an epidemiologic one.

If all three are negative, the change is more likely to be real. That is not proof, but it is the minimum due diligence.

### 10.2 The public-reporting feedback loop

A registry used for public reporting changes the behaviour of the entities it reports on. Three documented responses are case selection, documentation improvement, and code reassignment. All three change the data.

The analyst's response is not to avoid public-reporting registries but to compare like with like: restrict to periods with the same reporting regime, or include a regime indicator, and never compare a pre-report period to a post-report period without saying so.

---

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| Cases are ascertained completely | Incidence and prevalence estimates | Reporting gaps, facility non-participation | Capture-recapture; benchmark comparison |
| The population denominator matches the cases | Rate calculation | Non-resident cases, denominator revisions | Compare geography and year |
| The case definition is stable over the period | Trend analysis | Definition or code set change | Changelog review; split series |
| Sources are independent in capture-recapture | True count estimation | One source reports to the other | Report as a bound; dependence model |
| Vital status is complete | Survival analysis | Loss to follow-up | Compare to death registry linkage |
| Stage is missing at random | Stage-specific survival | Missingness related to prognosis | Compare survival by missingness status |
| The registry population is representative | Generalization | Regional practice and screening differences | Compare to a national source |
| The reporting regime is constant | Before-and-after comparison | Public reporting or payment change | Add a regime indicator |

### 11.1 Limits that are inherent

- **Registries capture diagnosis, not process.** They rarely record whether a guideline was followed.
- **Registries capture first course, not full treatment.** A later-line therapy is usually absent.
- **Registries lag by design.** Completeness and timeliness trade off.
- **Registries reflect reporting mandates.** A change in the mandate is a change in the data.
- **Registries are geographically bounded.** A patient who moves is lost or duplicated.
- **Public health data are aggregated.** Small-area analysis often exceeds the resolution available.

### 11.2 The question a registry cannot answer

A registry can tell you how many people were diagnosed and how long they survived. It usually cannot tell you why one treatment was chosen, whether the patient adhered, what the patient's quality of life was, or what care they received outside the reporting catchment.

If the decision depends on any of those, the registry is one input and not the answer.

### 11.3 How to present a registry estimate with its uncertainty

Three numbers belong together in any registry rate:

1. **The point estimate**, with its confidence interval from sampling variation.
2. **The completeness range**, which is usually wider than the sampling interval.
3. **The definitional version**, which determines what the estimate means.

Presenting only the first is the most common error. A rate of 111.25 per 100,000 with a sampling interval of 109.8 to 112.7 looks precise. Add the completeness range of 85.7 to 80 percent, and the plausible range widens to roughly 111 to 139. The sampling interval understates the uncertainty by a factor of five or more.

A practical format for a table is: estimate, sampling interval, completeness assumption, corrected estimate under that assumption. Four columns, and a reader can see what drives the range.

The definitional version is not a number but a sentence: "colorectal cancer, ICD-O-3 malignant behaviour, invasive only, AJCC seventh edition stage available from 2010." Without it, two estimates that look comparable may not be.

A second format, useful for a one-page summary, is a single sentence with all three elements: "Age-adjusted incidence was 111.3 per 100,000 per year (95% CI 109.8-112.7) among observed cases; assuming 86 percent ascertainment, the corrected estimate is 129.8 (plausible range 129.8-139.2), for invasive colorectal cancer, 2022-2024." That sentence is long, and every clause is load-bearing.

---

## 12. Exercises

### Exercise 1 — Write the inclusion rule (recall, 20 minutes)

Choose a registry you can find documentation for. Write its inclusion rule in one sentence, naming population, condition, geography, and period.

Deliverable: one sentence plus a list of three things the rule excludes.

Acceptance criteria: the sentence has all four elements; each exclusion is a real population, not a data quality issue.

### Exercise 2 — Capture-recapture (calculation, 45 minutes)

Using A = 4,812, B = 5,036, M = 4,317, compute the Lincoln-Petersen estimate, both completeness figures, and the number missing from both. Then compute the estimate assuming the sources are positively dependent by 10 percent (that is, the true overlap is 10 percent higher than observed).

Deliverable: two estimates and a one-paragraph interpretation.

Acceptance criteria: arithmetic correct; the interpretation states the direction of the dependence bias and why it makes the first estimate a lower bound.

### Exercise 3 — Age adjustment (calculation, 40 minutes)

Using the county data in section 6.5, recompute the age-adjusted rate with standard weights 0.60, 0.30, and 0.10. Report whether the county still has a lower adjusted rate than the state's 118.40.

Deliverable: the calculation and a two-sentence conclusion.

Acceptance criteria: arithmetic correct to two decimals; the conclusion states that the choice of standard population affects the comparison and names which standard is defensible.

### Exercise 4 — Weekly versus annual (critique, 30 minutes)

Find one NNDSS weekly table and the corresponding annual table for the same condition and year. Compare the sum of weeks to the annual total. Explain the difference.

Deliverable: a table with the two totals and a 200-word explanation.

Acceptance criteria: the difference is quantified; the explanation names delayed reporting and reconciliation, not "data errors."

### Exercise 5 — Reporting artefact or outbreak (critique, 45 minutes)

Given a time series with a step increase, list the three diagnostics from section 10.1 and describe what result in each would indicate an artefact versus a real change.

Deliverable: a three-row table with artefact and real-change criteria.

Acceptance criteria: each diagnostic is testable with available data; the criteria are specific rather than "looks suspicious."

### Exercise 6 — Stage missingness (application, 60 minutes)

Assume 18 percent of Stage III cases have missing stage and their survival is 10 percentage points lower than observed Stage III cases. Compute the bias in the five-year survival estimate for Stage III if missing cases are excluded.

Deliverable: a calculation with assumed counts and a bias statement.

Acceptance criteria: arithmetic correct; the direction of bias is stated; the assumptions are explicit.

### Exercise 7 — Provenance record (design, 60 minutes)

Write a provenance record for a registry extract. Include registry name, submission year, data dictionary version, extract date, completeness estimate and method, and record counts.

Deliverable: a one-page record.

Acceptance criteria: a second analyst could reproduce the extract; every version is named; the completeness estimate has a method attached.

### Exercise 8 — Choose the outcome source (design, 60 minutes)

For three outcomes (all-cause mortality, cancer incidence, 30-day readmission), choose between a registry, an EHR, and claims. Justify each and state what the chosen source cannot measure.

Deliverable: a three-row decision table.

Acceptance criteria: each choice names a specific ascertainment limitation; no source is treated as universally best.

### Exercise 9 — Registry versus surveillance (application, 30 minutes)

Compare a disease registry and a surveillance system for timeliness, completeness, and granularity. Recommend one for an outbreak response and one for a burden estimate.

Deliverable: a three-row comparison table and two recommendations.

Acceptance criteria: the recommendations are justified by the trade-off, not by preference; each names the cost of the choice.

---

## 13. Self-check answers

**A1. What is the difference between completeness and accuracy in a registry?**
Completeness is the proportion of true cases present. Accuracy is whether the recorded values are correct. A registry can be complete and inaccurate, or accurate and incomplete. They require different validation methods: completeness needs a second source, accuracy needs chart review.

**A2. Compute the Lincoln-Petersen estimate for A = 4,812, B = 5,036, M = 4,317.**
N = (4,812 × 5,036) / 4,317 = 24,233,232 / 4,317 = 5,613.4. State completeness = 4,812 / 5,613.4 = 85.7%. Hospital completeness = 5,036 / 5,613.4 = 89.7%. Missing from both = 5,613.4 − 4,812 − 719 = 82.4.

**A3. Why is the capture-recapture estimate a lower bound here?**
Because the hospital reports to the state registry, so the sources are positively dependent. Positive dependence raises the observed overlap above what independence would produce, which lowers the estimated total. The true count is higher and true completeness is lower.

**A4. Compute the crude annual incidence for 4,812 cases over 3 years in a population of 1,240,000.**
4,812 / (3 × 1,240,000) × 100,000 = 129.35 per 100,000 per year.

**A5. Compute the age-adjusted rate with weights 0.72, 0.21, 0.07 and rates 47.15, 234.41, 401.08.**
0.72 × 47.15 = 33.948; 0.21 × 234.41 = 49.2261; 0.07 × 401.08 = 28.0756. Sum = 111.25 per 100,000 per year.

**A6. Why does the county have a higher crude rate but a lower adjusted rate than the state?**
The county population is older. Age is a strong risk factor for colorectal cancer, so an older population produces more cases per person without higher age-specific risk. Adjustment removes the age-structure contribution, and the county's age-specific rates are lower than the state's.

**A7. Why can weekly NNDSS counts not be summed?**
Because delayed reports are added to the cumulative year-to-date figure rather than to the week in which they arrive, and counts can be revised down after investigation. Summing weekly tables therefore double-counts some cases and misses others. Annual tables are reconciled and must be used for annual estimates.

**A8. What is the difference between the underlying and immediate cause of death?**
The underlying cause is the disease or injury that initiated the chain of events leading to death. The immediate cause is the final condition in the chain. Coding rules select the underlying cause, and analyses that use the immediate cause will attribute deaths to mechanisms rather than diseases.

**A9. Why does the choice of SEER database matter?**
Because the databases cover different populations and periods: SEER 8 covers about 8.3 percent of the US population from 1975, while SEER 21 covers about 47.9 percent from 2000. A rate computed on one is not comparable to a rate computed on another without adjustment or restriction.

**A10. Name two documented responses of providers to public reporting.**
Case selection (avoiding high-risk patients) and more complete documentation of risk factors. A third is code reassignment. All three change the data without necessarily changing care.

**A11. Why might a registry-based revision rate be a lower bound?**
Because patients who move, change provider, or are lost to follow-up are not observed for revision. Unless the registry links to death records and to a comprehensive hospital dataset, the observed revision count understates the true count.

**A12. Rubric for exercise 7 (provenance record).**
A strong record names the registry, the submission year, the data dictionary version, the extract date, the completeness estimate and its method, and the record counts by year; it identifies any linkage and its error. A weak record names the registry and the year only, with no version and no completeness statement.

---

## 14. Interview questions

**Q1. How would you measure whether a registry is complete?**
A strong answer: use a second independent source, link the two, and apply capture-recapture; report the estimate as a bound because the independence assumption is usually violated; stratify by facility and year. Weak answers say "compare to published totals" without saying which totals or why they are complete.

**Q2. A registry's incidence rate is falling. What do you check first?**
A strong answer: the revision history, the facility mix, and the lag profile, in that order. A definition change, a facility leaving, or an incomplete final year all produce a decline that is not epidemiologic. Weak answers go to a statistical test.

**Q3. Why is age adjustment necessary?**
Because age is the strongest predictor of most disease rates, and populations differ in age structure. Without adjustment, a comparison reflects demography rather than risk. The standard population must be stated, because the choice changes the adjusted rate.

**Q4. What can a registry tell you that claims cannot?**
Stage, histology, and first-course treatment for cancer; device and procedure detail for implant registries; a defined population denominator. Claims have none of these and cannot substitute for them.

**Q5. What can claims tell you that a registry cannot?**
Comorbidity across all conditions, care outside the registry catchment, cost, and medication dispensing. Registries are condition-specific and geographically bounded.

**Q6. How would you detect a reporting artefact in a surveillance series?**
A strong answer: look for step changes coincident with a definition or mandate change, check whether the change is confined to the most recent periods (a lag artefact), and compare the pattern to a second source with a different reporting mechanism. Weak answers rely on visual inspection alone.

**Q7. Why is the underlying cause of death not always the right variable?**
Because coding selects one cause from a chain, and in patients with multiple conditions the selection can attribute death to a mechanism. Multiple-cause analysis is often more informative, and validation against medical records shows variable accuracy by condition.

**Q8. How would you use a registry for trial site selection?**
A strong answer: count eligible cases, not all cases; adjust for the lag between diagnosis and registry entry; check that the registry captures the outcome the trial needs; and validate the count against a site's own records. Weak answers use raw registry counts.

**Q9. What is the trade-off between completeness and timeliness?**
A system can optimize two of completeness, timeliness, and granularity. Registries prioritize completeness and granularity and accept an 18-24 month lag; surveillance prioritizes timeliness and accepts provisional counts.

**Q10. How do you handle missing stage in a survival analysis?**
A strong answer: report missingness by year and by subgroup, test whether survival differs by missingness status, and use multiple imputation or a bounding analysis rather than complete-case analysis. Weak answers drop missing cases without comment.

**Q11. What is a registry-based randomized trial and what does it require?**
A trial that uses a registry for enrollment and follow-up. It requires that the registry capture the outcome completely and that the consent and regulatory pathway fit the registry's legal basis. It trades some data control for cost and generalizability.

**Q12. What would you put in a registry provenance record?**
Registry name, submission year, data dictionary version, extract date, completeness estimate and method, linkage method and error, record counts by year, and known definitional changes in the period.

**Q13. A registry count rises 12 percent in one year. How do you decide whether it is real?**
A strong answer: check for a new reporting facility, a definition change, and a submission artefact confined to the recent period; then check whether the increase is concentrated in one site, one stage, or one demographic group, which would suggest a case-finding change rather than a true increase. Weak answers compare the number to the prior year and stop.

**Q14. What is the cost of using a provisional year in a published analysis?**
Two costs: the published number will be revised, sometimes materially, and any trend that includes the provisional year will overstate a decline. The revision is not an error by the analyst, but failing to state the provisional status is.

---

## 15. Key terms

- **Abstraction** — manual extraction of data from records into a registry.
- **Age adjustment** — standardizing rates to a reference age distribution.
- **AJCC** — American Joint Committee on Cancer.
- **Ascertainment** — the process of identifying cases for inclusion.
- **Backfill** — retrospective addition of previously unreported cases.
- **Capture-recapture** — estimating a true count from overlapping sources.
- **Case definition** — the criteria for counting a case.
- **Completeness** — the proportion of true cases present.
- **Crude rate** — events divided by population, unadjusted.
- **CSTE** — Council of State and Territorial Epidemiologists.
- **Direct standardization** — applying a standard population's weights to stratum-specific rates.
- **HEDIS** — Healthcare Effectiveness Data and Information Set.
- **ICD-O-3** — International Classification of Diseases for Oncology, third edition.
- **IIS** — Immunization Information System.
- **Lincoln-Petersen estimator** — the two-source capture-recapture formula.
- **MIPS** — Merit-based Incentive Payment System.
- **Multiple cause of death** — all conditions on the death certificate.
- **NNDSS** — National Notifiable Diseases Surveillance System.
- **NPCR** — National Program of Cancer Registries.
- **NVSS** — National Vital Statistics System.
- **Provisional data** — counts subject to revision.
- **Registry** — a systematic collection for a defined population or condition.
- **Reporting delay** — time between an event and its appearance in the data.
- **Reporting mandate** — a legal requirement to report cases.
- **SEER** — Surveillance, Epidemiology, and End Results program.
- **Standard population** — the reference age distribution used in adjustment.
- **Stage** — the extent of disease at diagnosis.
- **Syndromic surveillance** — surveillance using pre-diagnostic signals.
- **Underlying cause of death** — the condition that initiated the chain leading to death.
- **Vital statistics** — birth, death, marriage, and divorce records.

---

## 16. Sources

All URLs accessed 24 October 2026.

**Registries and public health programs**

- National Cancer Institute. *Overview of the SEER Program.* https://seer.cancer.gov/about/overview.html
- National Cancer Institute. *SEER Program Annual Usage Metrics, 2024.* https://seer.cancer.gov/about/factsheets/metrics.pdf
- National Cancer Institute. *SEER*Stat Databases: SEER November 2024 Submission.* https://seer.cancer.gov/data-software/documentation/seerstat/nov2024/
- CDC. *About Notifiable Infectious Diseases and Conditions Data, NNDSS.* https://www.cdc.gov/nndss/notifiable-infectious-disease-tables/about-data.html
- CDC. *Notice to Data Users and Publication Criteria, NNDSS.* https://www.cdc.gov/nndss/infectious-disease/notice-to-data-users.html
- CDC. *Readers' Guide: Understanding Weekly and Annual NNDSS WONDER Tables* (revised 10 December 2024). https://cdc.gov/nndss/docs/Readers-Guide-WONDER-Tables-20210421-508.pdf
- CDC. *NNDSS Data in CDC WONDER.* https://wonder.cdc.gov/nndss.html
- CSTE. *Protocol for Public Health Agencies to Notify CDC about the Occurrence of Nationally Notifiable Conditions, 2024.* https://ndc.services.cdc.gov/wp-content/uploads/NNC_2024_Notification_Requirements_By_Timeframe__FINAL.pdf
- CDC. *USCS Data Dictionary and Data Standards, 2024 Data Submission* (released September 2025). https://www.cdc.gov/rdc/media/pdfs/2026/01/USCS-RDC-DataDictionary-CDC-Nov2024.pdf

**Regulators and frameworks**

- HHS Office for Civil Rights. *Guidance Regarding Methods for De-identification of Protected Health Information.* https://www.hhs.gov/hipaa/for-professionals/special-topics/de-identification/index.html
- eCFR. 45 CFR 164.512, *Uses and disclosures for which an authorization or opportunity to agree or object is not required.* (Cite the Part and section; no separate URL given.)
- European Union. *Regulation (EU) 2025/327 on the European Health Data Space.* https://eur-lex.europa.eu/eli/reg/2025/327/oj/eng
- European Commission. *European Health Data Space Regulation (EHDS).* https://health.ec.europa.eu/ehealth-digital-health-and-care/european-health-data-space-regulation-ehds_en

**Methods and background**

- CDC/NCHS. *National Vital Statistics System* documentation. https://www.cdc.gov/nchs/nvss/
- National Cancer Institute. *SEER*Stat software and documentation.* https://seer.cancer.gov/data-software/documentation/index.html
- Hook, E. B., and Regal, R. R. "Capture-recapture methods in epidemiology: methods and limitations." *Epidemiologic Reviews*, 1995. DOI: 10.1093/oxfordjournals.epirev.a036192 (background, pre-2024).

> **Note:** The 2024 SEER coverage and usage figures come from NCI's published fact sheets and SEER*Stat documentation. Coverage percentages differ by database and by the population group being described; both sets of figures are reported in section 8.2 rather than reconciled into one number.

---

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`02_claims-and-billing.md`](02_claims-and-billing.md)
- Next topic: [`04_imaging-and-dicom.md`](04_imaging-and-dicom.md)
- Related sibling: [`01_ehr-data.md`](01_ehr-data.md)
- Related sibling: [`10_data-quality-dimensions.md`](10_data-quality-dimensions.md)
- Related sibling: [`09_licensing-and-access.md`](09_licensing-and-access.md)
- Root survey file: [`../../02_biomedical_data_landscape.md`](../../02_biomedical_data_landscape.md)
- Labs: [`../../labs/`](../../labs/)
- Lab 2.4 in the survey file (source-selection matrix) and Lab 2.5 (data-quality profile) both draw on the completeness and stratification methods in section 5.

Related reading: [`01_ehr-data.md`](01_ehr-data.md) for the encounter-anchored view that registries complement; [`10_data-quality-dimensions.md`](10_data-quality-dimensions.md) for the formal vocabulary of completeness and plausibility; and [`09_licensing-and-access.md`](09_licensing-and-access.md) for the data use agreements that govern registry research extracts.

A registry is the best available answer to "how much disease is there" and a poor answer to "what happened to this patient." Keeping those two questions separate is the discipline this topic teaches.

The next topic moves to a data type where the file itself is the object of analysis rather than a record about a patient: imaging and DICOM.
