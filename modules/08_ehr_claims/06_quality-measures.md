# Quality Measures

> **Module 08 — EHR, Claims, and Health System Operations**
> Topic 06 of 09
> Estimated study time: 3.5 hours
> Prerequisites: Topics 02 and 03
> Version 2026.3

## 1. Why this topic exists

Quality measures convert clinical care into a number that payers, regulators, and the public use to compare providers. The number is not a direct measurement of care. It is the output of a specification: a denominator, a set of exclusions, a numerator, a data source, and a set of rules for handling missing and conflicting information. When the specification is implemented faithfully, the measure is comparable. When it is implemented loosely, the measure is a local construction that looks like the official one and is not.

The stakes are financial and reputational. Medicare Advantage plans receive a quality bonus based on CMS Star Ratings, and a plan's star rating depends on measures such as Controlling High Blood Pressure, Breast Cancer Screening, Colorectal Cancer Screening, Plan All-Cause Readmissions, and Transitions of Care. In the 2026 Star Ratings, the 5-star cut point for Controlling High Blood Pressure was 86 percent, for Breast Cancer Screening 84 percent, and for Colorectal Cancer Screening 78 percent, according to the CMS technical notes and the published cut-point tables. A provider or plan at 70 percent on a measure with an 86 percent cut point is not close; it is 16 percentage points away, and the distance translates into a payment difference.

The measurement problem is that a measure can improve while care does not. Three mechanisms produce that result.

**Documentation change.** A blood pressure recorded in a note, a home device, or a scanned fax may not be in the structured field that the measure reads. A project that extracts those values and enters them into the structured field raises the measured rate without changing a single patient's blood pressure. The care was always there; the data were not. This is a legitimate improvement in data quality and a misleading improvement in the measure if it is presented as a clinical gain.

**Denominator change.** A measure excludes patients in hospice, with end-stage renal disease, who are pregnant, or who are institutionalized. If the exclusions are applied inconsistently, the denominator changes, and the rate moves without any change in the numerator. A provider that removes sick patients from the denominator can improve the rate while the care for the remaining patients is unchanged, and the care for the removed patients may be worse.

**Numerator rule change.** A measure may specify the most recent blood pressure in the measurement year, the lowest of multiple readings on the same date, or a restriction on the setting. A local implementation that uses the last recorded value, includes ED readings, or averages readings produces a different numerator. The measure is no longer the same measure, even if the label is the same.

The gap between the specification and the computable logic is the central topic. NCQA publishes HEDIS specifications in prose and in value sets; CMS publishes eCQMs in Clinical Quality Language (CQL) and FHIR resources. The prose and the code are not identical. The code resolves ambiguities that the prose leaves open, and a local implementation resolves them again, often differently. The analyst's job is to know which choices were made, to document them, and to test whether they change the conclusion.

This topic teaches measure logic as an engineering artefact. The output is a measure specification with a denominator, exclusions, numerator, data sources, validation results, and a documentation-change log. A measure without that artefact is a number that cannot be defended in an audit or compared with a benchmark.

A measure can also improve because the denominator shrinks. Removing patients with ESRD or hospice from the denominator raises the rate if those patients have lower control, and the care for the remaining patients is unchanged. The removal may be correct under the specification and still produce a misleading improvement story. The analyst must report the exclusion count and the reason, not only the rate.

> **Note:** A quality measure is a payment and accountability construct, not a clinical truth. A patient can receive excellent care and fail a measure because the data are missing; a patient can receive poor care and pass because the documentation is complete. The measure is useful only when its logic and its data are understood.

## 2. Learning outcomes

By the end of this topic you can:

1. **Parse** a measure specification into denominator, exclusions, numerator, measurement period, and data sources.
2. **Compute** a measure rate with the correct denominator after exclusions, and **state** the denominator at each step.
3. **Distinguish** a specification change from a documentation change from a clinical change, and **quantify** each.
4. **Implement** a measure in SQL against an EHR or claims schema and **reconcile** the result to the steward's reported rate.
5. **Identify** the ambiguities in a prose specification and **document** the local resolution.
6. **Evaluate** a measure's sensitivity to data source, exclusion policy, and numerator rule.
7. **Explain** how a measure can improve while care does not, with arithmetic.
8. **Design** a monitoring plan that separates data quality from clinical performance.

Each outcome maps to an exercise in Section 12 or a self-check item in Section 13. Outcomes 2, 3, and 6 are the quantitative core.

## 3. The core mental model

A measure is a funnel with a payment at the end. The funnel starts with the eligible population, removes exclusions, applies the numerator rule, and produces a rate. The rate is compared with a cut point or a benchmark, and the comparison determines a payment or a public rating.

```text
ELIGIBLE POPULATION
 age, condition, enrollment, measurement period
    |
    v
DENOMINATOR
 eligible minus exclusions
    |
    v
NUMERATOR
 patients meeting the clinical criterion
    |
    v
RATE = numerator / denominator
    |
    v
CUT POINT / BENCHMARK -> payment, star rating, public report
```

Four properties determine whether the rate is meaningful.

**The denominator is the measure.** A rate without its denominator is uninterpretable. A 70 percent rate on 11,400 patients is a different statement from 70 percent on 800 patients. The denominator includes the eligibility rules (age, condition, enrollment) and the exclusions (hospice, ESRD, pregnancy, frailty). Each rule is a choice, and each choice changes the population.

**The numerator rule is specific.** "Most recent blood pressure" is not the same as "last blood pressure in the EHR", "lowest blood pressure", or "average blood pressure". The specification may say "most recent" and also "lowest systolic and lowest diastolic on the same date". A local implementation that ignores the second clause will produce a different numerator.

**The data source is part of the measure.** A HEDIS measure may allow claims, EHR, and supplemental data. A CMS eCQM uses EHR data. A Star Rating measure may use claims and administrative data. The same clinical performance can produce different rates under different data sources. A measure that improves because a new data feed was added has not necessarily improved care.

**The cut point is not a clinical target.** A 5-star cut point of 86 percent is a relative ranking of plans, not a statement that 86 percent is the right level of control. A plan at 84 percent may be clinically excellent and still lose a star. The cut point is a payment rule, and it changes from year to year.


### The funnel with numbers

| Stage | Count | Percent of previous |
|---|---:|---:|
| Eligible adults | 50,000 | not applicable |
| With hypertension | 12,000 | 24.0% |
| After exclusions | 11,400 | 95.0% |
| Controlled | 7,980 | 70.0% |
| Additional for 86% | 1,824 | 22.9% of denominator |

The funnel shows that the measure is a small subset of the population and that the numerator is a subset of the denominator. A 1 percent change in the eligible population is 120 patients; a 1 percent change in the rate is 114 patients. The denominators are different, and the difference matters.

> **Note:** The measure rate is a ratio of two counts. The counts depend on the data, the rules, and the implementation. Before comparing two rates, confirm that the denominator, numerator, period, and data sources are the same.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Measure | A specification for quantifying care or outcomes | Treated as a clinical truth | Specification with steward |
| Measure steward | The organisation that owns and maintains the measure | Confused with the regulator | NCQA, CMS, specialty society |
| HEDIS | Healthcare Effectiveness Data and Information Set | Treated as a CMS programme | NCQA measure set |
| Star Rating | CMS's summary rating for Medicare Advantage and Part D | Confused with HEDIS accreditation | One to five stars |
| eCQM | Electronic clinical quality measure | Treated as a HEDIS measure | CQL and FHIR |
| CQL | Clinical Quality Language | Confused with SQL | Expression language |
| Denominator | The eligible population after exclusions | The whole population | Patient count |
| Numerator | Patients meeting the clinical criterion | Patients with any documentation | Patient count |
| Exclusion | A condition that removes a patient from the denominator | A convenience filter | Rule with code and period |
| Denominator exception | A clinical reason the numerator does not apply | Confused with exclusion | Rule |
| Measurement period | The window for eligibility and numerator | Confused with the lookback | Calendar year |
| Continuous enrollment | Enrollment without a gap longer than allowed | Ignored in provider data | Months and gap |
| Value set | The codes that define a clinical concept | Confused with a code set | Codes plus steward |
| VSAC | Value Set Authority Center | Treated as a terminology | Repository |
| Supplemental data | Data beyond claims used for a measure | Treated as optional | Source-specific |
| Attribution | Assignment of a patient to a provider or plan | Confused with enrollment | Member-provider pair |
| Cut point | The threshold for a star or payment tier | Confused with a clinical target | Percent |
| Benchmark | A comparison value, often a percentile | Confused with a target | Percent or rate |
| Performance rate | Numerator divided by denominator | Reported without exclusions | Percent |
| Confidence interval | The uncertainty around a rate | Ignored in small denominators | Interval |
| Small denominator | A denominator too small for a stable rate | Reported without a warning | Count threshold |
| Stratification | Reporting a measure by subgroup | Confused with risk adjustment | Subgroup rates |
| Risk adjustment | Adjusting for patient complexity | Confused with stratification | Model |
| Documentation gap | Care delivered but not recorded in the measure field | Treated as a clinical gap | Count |
| Data quality gap | A missing or mis-mapped data element | Treated as a clinical gap | Count |
| Clinical gap | Care not delivered | Treated as a documentation gap | Count |
| Chart review | Manual verification of a sample | Treated as scalable | Sample |
| Audit | An independent check of the measure calculation | Confused with a chart review | Report |
| Hybrid measure | A measure that combines claims and medical record data | Confused with an eCQM | Specification |
| Electronic measure | A measure computed from electronic data | Confused with a hybrid | Specification |
| Denominator exclusion | A rule that removes a patient | Confused with a numerator exception | Rule |
| Numerator exception | A rule that allows a patient to be counted despite missing data | Confused with an exclusion | Rule |
| Star year | The rating year, based on a measurement year | Confused with the measurement year | Year |
| Measurement year | The year in which care is measured | Confused with the rating year | Year |
| HEDIS MY | HEDIS measurement year | Confused with the reporting year | Year |
| Quality bonus | The payment tied to a star rating | Confused with a base rate | Percent |
| Reward | A payment for performance | Confused with a penalty | Currency |
| Penalty | A payment reduction for poor performance | Confused with a withhold | Percent |
| Withhold | A portion of payment held pending performance | Confused with a penalty | Percent |
| Attribution window | The period used to assign patients | Confused with the measurement year | Months |
| Lookback | The period before the measurement year for eligibility | Confused with the measurement period | Months |
| Most recent | The latest value in the measurement year | Confused with the last value in the EHR | Timestamp |
| Valid reading | A reading that meets the specification's setting and method rules | Any recorded value | Rule |
| Remote reading | A home or device reading | Treated as invalid by default | Method |
| Structured field | A coded or numeric field | Confused with a note | Data element |
| Note extraction | Pulling a value from free text | Treated as structured data | NLP or manual |
| Registry | A list of patients for a condition or programme | Confused with a measure denominator | Patient list |
| Care gap | A patient who fails the numerator | A documentation gap | Patient flag |
| Gap closure | Moving a patient into the numerator | A clinical improvement | Count |
| Measure specification | The official definition | A local query | Document |
| Measure implementation | The local computation | The official measure | Query and data |
| Measure version | The year and revision of the specification | Ignored in trends | Version string |
| Reporting year | The year the measure is reported | Confused with the measurement year | Year |
| Star cut point | The threshold for a star level | A clinical target | Percent |
| Tukey outlier | A method for setting cut points | A clinical rule | Statistical method |

## 5. The mechanics, step by step

The procedure below implements a measure from specification to reported rate. It uses Controlling High Blood Pressure (CBP) as the running example, but the steps apply to any proportion measure.

### Step 1. Select the measure, the steward, and the version

**Input:** the reporting requirement (HEDIS, Star, eCQM, payer contract).

**Output:** the measure identifier, steward, measurement year, and version.

**Rationale:** the same measure name can have different specifications across stewards and years. HEDIS CBP and the CMS MIPS CBP measure share a clinical concept but differ in age rules, exclusions, and data sources. A rate computed under one specification cannot be compared with a rate computed under another. Record the identifier and version before building anything.

**Branch point:** if the measure is used for payment, the payer's specification governs. If it is used for internal improvement, a local adaptation is acceptable only if it is labelled as local and not compared with the official rate.

### Step 2. Parse the specification into computable parts

**Input:** the official specification and value sets.

**Output:** a table with population, denominator, exclusions, numerator, exceptions, period, and data sources.

**Rationale:** the prose specification contains rules that are easy to miss: the age as of a specific date, the enrollment gap, the diagnosis window, the setting restrictions, and the "most recent" rule. Parsing them into a table forces the analyst to resolve each rule explicitly. The table becomes the test plan.

A useful format is one row per rule with four columns: rule, source text, implementation, and test. The implementation column is the SQL or CQL expression; the test column is the expected count or a sample of patients.

**Branch point:** if a rule is ambiguous, document the interpretation and its effect. Do not silently choose the interpretation that produces the higher rate.

### Step 3. Map the measure to data sources

**Input:** the specification's allowed data sources and the local data inventory.

**Output:** a source map with coverage, completeness, and lag for each data element.

**Rationale:** a measure is only as good as its data. HEDIS allows claims, EHR, and supplemental data; a CMS eCQM uses EHR data; a Star measure may use administrative data. The same patient can be in the denominator under one source and not another. The source map states which source is authoritative for each element and how missing values are handled.

For CBP, the critical elements are the hypertension diagnosis, the enrollment, the exclusions, and the blood pressure reading. The diagnosis may come from claims or the problem list; the reading may come from the EHR vitals, a device integration, or a note. The source determines the rate.

**Branch point:** if a data source is incomplete for a subgroup, the measure will understate performance for that subgroup. Report the coverage by subgroup before interpreting a disparity.

### Step 4. Build the eligible population

**Input:** the source map and the measurement period.

**Output:** the eligible count and the attrition at each criterion.

**Rationale:** eligibility includes age, diagnosis, enrollment, and visit requirements. CBP uses ages 18 to 85, a diagnosis of hypertension, continuous enrollment with no more than a one-month gap, and at least two outpatient visits or one visit plus a diagnosis, depending on the version. Each criterion removes patients and changes the denominator.

Report the attrition. A drop of 40 percent at the enrollment step may be correct for a Medicaid population with churn, or it may be a data error. The attrition table is the first diagnostic.

**Branch point:** if the eligible population differs materially from the payer's, reconcile by member before proceeding. A difference of more than 2 percent is usually a definition or data issue.

### Step 5. Apply denominator exclusions

**Input:** the eligible population and the exclusion value sets.

**Output:** the denominator and the exclusion counts by reason.

**Rationale:** exclusions remove patients for whom the measure is not meaningful: hospice, palliative care, ESRD, pregnancy, institutionalized status, and frailty with advanced illness. Each exclusion has a code set and a time window. Applying an exclusion with the wrong window, or missing a code, changes the denominator and the rate.

The exclusion count by reason is diagnostic. A sudden rise in hospice exclusions may be a coding change, not a change in the population. A missing ESRD exclusion will include dialysis patients whose blood pressure is difficult to control, lowering the rate.

**Branch point:** if an exclusion removes more than 10 percent of the eligible population, audit the code set and the window. Large exclusions are often a data artifact.

### Step 6. Build the numerator

**Input:** the denominator and the clinical criterion.

**Output:** the numerator count and the distribution of values.

**Rationale:** CBP uses the most recent blood pressure in the measurement year, on or after the second hypertension diagnosis, with systolic below 140 and diastolic below 90. If there are multiple readings on the same date, the specification uses the lowest systolic and lowest diastolic, which need not come from the same reading. Readings from acute inpatient, ED, or same-day procedures are excluded. The rule is specific, and each clause changes the count.

A common error is to use the last recorded value rather than the most recent valid value. Another is to include ED readings, which are often elevated. A third is to use the first reading rather than the lowest on a date. Each error moves the numerator in a predictable direction.

**Branch point:** if the numerator is built from a note or a device feed, validate the extraction against a chart sample. A 5 percent extraction error can change a star rating.

### Step 7. Apply numerator exceptions

**Input:** the denominator and the exception rules.

**Output:** the numerator with exceptions applied, and the exception count.

**Rationale:** some measures allow a patient to be removed from the numerator for a clinical reason, such as a documented medical exception. The exception is not the same as an exclusion: the patient remains in the denominator but is not counted as a failure. Confusing the two changes the denominator and the rate.

Document each exception, its source, and its frequency. A high exception rate may indicate a documentation problem or a clinical population that the measure does not fit.

**Branch point:** if the exception rate exceeds 5 percent, review a sample. An exception that is applied without documentation is not valid.

### Step 8. Compute the rate and its uncertainty

**Input:** the numerator and the denominator.

**Output:** the performance rate, the confidence interval, and the denominator size.

**Rationale:** the rate is the numerator divided by the denominator. A rate without a confidence interval invites overinterpretation, especially for small denominators. A practice with 40 patients and a 75 percent rate has a wide interval; a plan with 10,000 patients and a 75 percent rate has a narrow one.

Use the Wilson interval for proportions. Report the denominator alongside the rate. Suppress or flag rates below a minimum denominator (often 30) because they are unstable and can mislead.

**Branch point:** if the denominator is below the minimum, report the count and the rate with a suppression flag, not a star or a ranking.

### Step 9. Validate against the steward or an independent recomputation

**Input:** the local rate and the steward's rate or a second implementation.

**Output:** a reconciliation table with differences by step.

**Rationale:** a difference between the local and steward rates can come from the population, the exclusions, the numerator, or the data source. Reconcile step by step: eligible count, exclusion count, denominator, numerator. A 2 percent difference in the denominator and a 0 percent difference in the numerator produce a 2 percent difference in the rate.

If the steward does not publish a rate, recompute with an independent query or a second analyst. The goal is to find the implementation difference before the rate is used.

**Branch point:** if the reconciliation cannot close within 1 percent, do not report the rate as official. Report it as a local estimate with the difference explained.

### Step 10. Attribute gaps: clinical, documentation, or data

**Input:** the patients who fail the numerator.

**Output:** a gap taxonomy with counts and a sampling plan.

**Rationale:** a failed numerator is not necessarily a clinical gap. The patient may have a controlled blood pressure recorded in a note, a home device, or an outside record; the data may be missing; or the care may not have been delivered. The three causes require different actions. A documentation project fixes the first; a data integration fixes the second; a clinical programme fixes the third.

Sample 30 to 50 failures and classify them by chart review. The proportion in each category determines the intervention. A plan that invests in clinical outreach when 60 percent of failures are documentation gaps will waste money.

**Branch point:** if the documentation gap is large, fix the data capture before launching a clinical programme. Otherwise the programme's effect will be invisible in the measure.

### Step 11. Stratify by subgroup

**Input:** the measure numerator, denominator, and patient characteristics.

**Output:** rates by race, ethnicity, language, payer, disability, and geography, with confidence intervals.

**Rationale:** an overall rate can hide a widening gap. A plan that improves from 70 to 74 percent overall while the gap between groups grows has not improved equity. Stratification is required by many programmes and is good practice even when it is not.

Use the same denominator and numerator rules for each subgroup. Do not pool categories to increase the denominator without reporting the pooling. Suppress small cells and report the count.

**Branch point:** if a subgroup rate is based on fewer than 30 patients, report it as exploratory and do not rank providers on it.

### Step 12. Report, monitor, and version

**Input:** the validated rate, the gap taxonomy, and the subgroup results.

**Output:** a measure report with the specification, data sources, rate, interval, exclusions, gaps, and limitations.

**Rationale:** a measure report without the specification cannot be reproduced or compared. The report should state the measure version, the measurement year, the data sources, the denominator, the exclusions, the numerator, the rate, and the confidence interval. It should also state the known limitations and the data-quality issues.

Monitor the rate monthly or quarterly, but do not overreact to small changes. Plot the rate with its interval and mark specification or data-source changes on the timeline. A step change at a specification change is not a clinical change.

**Branch point:** if the measure version changes, restate the prior period under the new version or present the trend with a break. A trend that mixes versions is not a trend.

### Decision table: data source hierarchy

| Data element | Preferred source | Fallback | Risk |
|---|---|---|---|
| Hypertension diagnosis | Claims or problem list | Note extraction | Problem-list coding bias |
| Enrollment | Payer eligibility file | EHR insurance field | Retroactive eligibility |
| Hospice exclusion | Claims or EHR | Note extraction | Missing hospice |
| ESRD exclusion | Claims, labs, or problem list | Note extraction | Missing dialysis |
| Blood pressure | Structured vitals | Device feed or note | Setting and method validity |
| Pregnancy exclusion | Claims or problem list | Note extraction | Missing pregnancy |
| Provider attribution | Payer attribution file | Plurality of visits | Mid-year changes |

### Decision table: exclusion versus exception

| Concept | Effect on denominator | Effect on numerator | Example |
|---|---|---|---|
| Exclusion | Removed | Not applicable | Hospice patient removed |
| Exception | Remains | Removed from failure count | Documented medical exception |
| Missing data | Remains | Counted as failure unless an exception applies | No BP recorded |
| Invalid reading | Remains | Not counted as valid | ED reading excluded |

The distinction is crucial. An exclusion improves the rate by shrinking the denominator; an exception improves the rate by removing a failure; missing data usually lowers the rate. A local implementation that treats missing data as an exception will overstate performance.

### Edge cases that invalidate a measure

**Missing blood pressure.** A patient with no BP in the structured field is a failure unless a valid exception applies. The rate depends on how many patients have missing data and whether the missingness is related to care. A practice that sees healthy patients less often may have more missing data and a lower rate.

**External readings.** A home or device reading may be valid under the specification, but it may not be in the structured field. A documentation project that adds these readings raises the rate without changing care. Report the source of the readings and the change separately.

**Denominator changes.** A change in the enrollment rule, the diagnosis window, or the exclusion set changes the denominator. A rate trend that spans a specification change is not a clinical trend. Segment the trend at the change date.

**Measure version.** HEDIS and CMS measures are updated annually. A measure that was computed under the prior version may not be comparable. Record the version and restate prior periods when possible.

**Attribution changes.** A patient attributed to a different provider changes the provider's denominator and numerator. A mid-year attribution change can create a step change in a provider's rate. Use monthly attribution and report the rule.

**Small denominators.** A provider with 20 patients can swing from 60 to 80 percent with four patients. Ranking providers on small denominators produces noise. Suppress or pool.

**Documentation of exclusions.** An exclusion that is not documented in a coded field may not be applied, which keeps a sick patient in the denominator and lowers the rate. The exclusion code set and the documentation workflow are part of the measure.

### Measure specification worksheet

| Element | CBP example | Implementation question |
|---|---|---|
| Age | 18-85 as of 31 December | Which date of birth source? |
| Condition | Hypertension diagnosis | Claims, problem list, or both? |
| Diagnosis window | Prior year through 30 June | Does the lookback cross systems? |
| Enrollment | Continuous, gap <= 45 days | Which eligibility file? |
| Visits | Two outpatient visits or one plus diagnosis | Telehealth counted? |
| Exclusions | Hospice, palliative, ESRD, pregnancy, frailty | Which code sets and windows? |
| Numerator | Most recent BP <140/90 | Which setting and method? |
| Multiple readings | Lowest systolic and diastolic on a date | Same reading or different? |
| Period | Measurement year | Calendar or contract year? |

### CBP rule detail

| Rule | Specification | Common local deviation | Direction of bias |
|---|---|---|---|
| Age | 18-85 | 18-75 | Excludes older patients, usually lower control |
| Diagnosis | Essential hypertension value set | I10 only | Misses secondary and related codes |
| Enrollment | Continuous with one 45-day gap | No gap | Excludes churn, often sicker |
| Visits | Two outpatient or one plus diagnosis | One visit | Narrows denominator |
| Hospice | Any time in year | Claims only | Misses EHR hospice |
| ESRD | History through year end | Claims only | Includes dialysis patients |
| Pregnancy | Any time in year | Not applied | Includes pregnant patients |
| BP setting | Outpatient, valid method | Any setting | Includes ED readings, lower control |
| Most recent | Latest valid in year | Last in EHR | May use an older value |
| Same-date rule | Lowest systolic and diastolic | First reading | Lower control |

### Gap taxonomy

| Gap type | Definition | Detection | Action |
|---|---|---|---|
| Clinical | Care not delivered or not controlled | Chart review, medication review | Clinical outreach |
| Documentation | Care delivered but not in the measure field | Note review, device feed | Documentation workflow |
| Data | Value missing or mis-mapped | Source reconciliation | Interface fix |
| Eligibility | Patient should not be in the denominator | Exclusion audit | Apply exclusion |
| Attribution | Patient assigned to the wrong provider | Attribution audit | Correct assignment |

A gap taxonomy converts a failed numerator into an action. Without it, every failure looks like a clinical problem, and the programme over-invests in outreach.

### Rate reconciliation

| Step | Local count | Steward count | Difference | Explanation |
|---|---:|---:|---:|---|
| Eligible | 12,000 | 12,000 | 0 | Same source |
| Exclusions | 600 | 640 | -40 | Hospice from EHR not in claims |
| Denominator | 11,400 | 11,360 | +40 | Exclusion difference |
| Numerator | 7,980 | 7,940 | +40 | Same clinical rule |
| Rate | 70.0% | 69.9% | +0.1 pp | Consistent |

The reconciliation isolates the difference at the exclusion step. A difference that appears at the numerator step is a clinical-rule difference; a difference at the eligible step is a population or enrollment difference.

### A worked rate calculation

```text
Eligible adults with hypertension = 12,000
Exclusions: hospice 150 + ESRD 200 + pregnancy 250 = 600
Denominator = 12,000 - 600 = 11,400
Numerator (most recent BP <140/90) = 7,980
Rate = 7,980 / 11,400 = 70.0%
```

Adding 300 controlled patients from a device feed changes the rate to 8,280 / 11,400 = 72.6 percent. The denominator did not change; the numerator did. The care may not have changed at all.

### Data quality checks

| Check | Expected | Action if it fails |
|---|---|---|
| Denominator matches payer | Within 2% | Reconcile member list |
| Exclusion rate | 3-8% | Audit code sets |
| Missing BP rate | Below 15% | Review data capture |
| BP values in range | 50-250 mmHg | Check units and entry |
| Duplicate readings | Low | Deduplicate by timestamp |
| External readings | Stable over time | Check device feed |
| Stratum counts | Sum to total | Check unknown categories |

### Reporting template

A measure report should contain, in this order: measure identifier and version; measurement year; data sources; eligible population; exclusions by reason; denominator; numerator; rate with confidence interval; subgroup rates; gap taxonomy; comparison with the prior period and the benchmark; known limitations; and the name of the person who validated the calculation.

A report without the exclusion counts cannot be audited. A report without the data sources cannot be compared. A report without the validation name cannot be trusted.

### Common measures and their pressure points

| Measure | Denominator pressure | Numerator pressure | Common failure |
|---|---|---|---|
| Controlling High Blood Pressure | Enrollment and visit rules | Most recent valid BP | ED readings included |
| Breast Cancer Screening | Age and mastectomy exclusions | Mammogram in period | Outside imaging missing |
| Colorectal Cancer Screening | Age and hospice exclusions | Any accepted test | FIT kit not returned |
| Plan All-Cause Readmissions | Index admission and planned exclusions | 30-day readmission | Transfers counted |
| Transitions of Care | Discharge and enrollment | Four transition elements | Documentation incomplete |
| Kidney Health Evaluation | Diabetes and enrollment | Two kidney tests | One test only |

### Measure logic versus clinical logic

Measure logic is not clinical logic. A clinician treats the patient in front of them; a measure counts the patients who meet a rule. A patient with a controlled BP at home but no clinic reading may be clinically well and measure as a failure. A patient with a single elevated reading in the ED may be clinically stable and measure as uncontrolled if the ED reading is wrongly included. The analyst must keep both logics in view: the measure is the accountability construct, and the clinical record is the evidence.

### Small denominators

| Denominator | 95% interval half-width at 70% | Use |
|---:|---:|---|
| 30 | +/- 16 pp | Exploratory only |
| 100 | +/- 9 pp | Trend, not ranking |
| 500 | +/- 4 pp | Provider comparison |
| 2,000 | +/- 2 pp | Payment and public reporting |

The interval narrows with the square root of the denominator. A provider with 30 patients cannot be ranked against one with 2,000 on the same scale.

The interval table is also a governance tool. It tells the review committee which comparisons are informative and which are noise, before the ranking is published.

A ranking that ignores the interval is a ranking of sample size as much as performance.

That is why measure reports should show the denominator before the rate.

The next section applies the procedure to a single measure with full arithmetic.
## 6. Worked example

A Medicare Advantage plan is reporting Controlling High Blood Pressure (CBP) for the measurement year. The measure uses ages 18 to 85, a hypertension diagnosis, continuous enrollment with no more than a 45-day gap, at least two outpatient visits, and the exclusions for hospice, palliative care, ESRD, pregnancy, and frailty with advanced illness. The numerator is the most recent valid blood pressure in the measurement year, with systolic below 140 and diastolic below 90.

### Part A: denominator and rate

The plan has 50,000 adult members. Of these, 12,000 have a hypertension diagnosis and meet the age and visit rules. The exclusions are 150 hospice, 200 ESRD, and 250 pregnancy, for 600 total.

| Step | Count | Removed | Percent of previous |
|---|---:|---:|---:|
| Adults 18-85 | 50,000 | not applicable | not applicable |
| Hypertension diagnosis and visits | 12,000 | 38,000 | 76.0% |
| Hospice exclusion | 11,850 | 150 | 1.25% |
| ESRD exclusion | 11,650 | 200 | 1.69% |
| Pregnancy exclusion | 11,400 | 250 | 2.15% |
| Denominator | 11,400 | | |
| Numerator (BP <140/90) | 7,980 | | |
| Performance rate | 70.0% | | |

```text
Denominator = 12,000 - 150 - 200 - 250 = 11,400
Rate = 7,980 / 11,400 = 70.00%
95% Wilson interval approximately 69.16% to 70.83%
```

The rate is 70.0 percent. The 2026 Star Ratings 5-star cut point for Controlling High Blood Pressure was 86 percent, so the plan is 16 percentage points below the top tier. To reach 86 percent on the same denominator, the numerator would need to be 11,400 x 0.86 = 9,804, an increase of 1,824 controlled patients over the current 7,980.

### Part B: a documentation change

The plan discovers that 500 members have a blood pressure recorded in a scanned external document or a home device feed, but not in the structured vitals field. A review finds that 60 percent of those readings are below 140/90, or 300 members. A documentation project enters the readings into the structured field.

```text
New numerator = 7,980 + 300 = 8,280
New rate = 8,280 / 11,400 = 72.63%
Change = +2.63 percentage points
```

No patient's blood pressure changed. The measured rate rose because the data capture improved. The improvement is real for measurement purposes and should be reported as a data-quality gain, not as a clinical improvement. If the plan reports it as clinical, it will overstate the effect of its care management and understate the remaining clinical gap.

The remaining gap to 86 percent is now 9,804 - 8,280 = 1,524 patients, not 1,824. The documentation project closed 300 of the 1,824, or 16.4 percent of the gap, at a cost that is usually far lower than clinical outreach.

### Part C: a clinical programme

The plan considers an outreach programme for members with uncontrolled hypertension. The programme contacts 2,000 members, and 25 percent achieve control, for 500 additional controlled patients.

```text
Additional controlled = 2,000 x 0.25 = 500
New numerator = 8,280 + 500 = 8,780
New rate = 8,780 / 11,400 = 77.02%
Cost = $180,000
Cost per additional controlled patient = $180,000 / 500 = $360
Cost to reach 86% from 72.63% = 1,524 x $360 = $548,640
```

The programme moves the rate to 77.0 percent, still 9 points below the 5-star cut point. Reaching 86 percent through this programme alone would require about 1,524 additional controlled patients, or 6,096 contacts at a 25 percent success rate, at a cost of about $548,640. That may be worthwhile if the star bonus exceeds the cost, but the plan should compare it with the cheaper documentation route and with a targeted programme for the patients most likely to respond.

### Diagnostic 1: the denominator includes members without continuous enrollment

Suppose 300 of the 11,400 members had a coverage gap longer than 45 days and should have been excluded. The denominator becomes 11,100, and the rate becomes 7,980 / 11,100 = 71.89 percent, a 1.89 percentage point increase. The numerator is unchanged.

The error is a denominator error, and it makes the plan look worse than it is. The diagnostic is to recompute the denominator from the eligibility file with the gap rule applied and to reconcile the count to the payer's enrollment report. A difference of more than 2 percent in the denominator is usually an enrollment rule difference.

### Diagnostic 2: the ESRD exclusion is not applied

Suppose the ESRD exclusion is missing because the dialysis codes are in a claims file that was not linked. The 200 ESRD members remain in the denominator. Their blood pressure is harder to control, and assume only 40 percent are controlled, or 80 patients, instead of the 70 percent average.

```text
Denominator without ESRD exclusion = 11,600
Numerator without the 200 ESRD patients removed = 7,980 + 80 = 8,060
Rate = 8,060 / 11,600 = 69.48%
```

The rate falls by 0.52 percentage points relative to the correct 70.0 percent. The direction is predictable: including a group with lower control lowers the rate. The diagnostic is to audit the exclusion code sets against the claims and EHR, and to report the exclusion counts by reason.

### Varying one assumption

The 86 percent cut point is a policy threshold, not a clinical target. If the plan's contract uses a 75 percent target instead, the clinical programme alone (77.0 percent) clears it, while the documentation-only path (72.6 percent) does not. The decision changes with the target, not with the care.

| Scenario | Numerator | Denominator | Rate | Meets 75%? | Meets 86%? |
|---|---:|---:|---:|---|---|
| Base | 7,980 | 11,400 | 70.00% | No | No |
| Documentation only | 8,280 | 11,400 | 72.63% | No | No |
| Clinical programme | 8,780 | 11,400 | 77.02% | Yes | No |
| Both | 9,080 | 11,400 | 79.65% | Yes | No |
| Documentation plus 1,524 clinical | 9,804 | 11,400 | 86.00% | Yes | Yes |

The table shows that the measure can be moved by either route, and that the cheapest route depends on the gap type. A plan that does not separate documentation from clinical gaps will misallocate its improvement budget.

### What the measure does not tell you

The measure does not tell you whether the 7,980 controlled patients are healthy, whether the 3,420 uncontrolled patients received appropriate treatment, or whether the blood pressure readings are accurate. It tells you how many members had a valid reading below the threshold in the measurement year. The clinical questions require chart review, medication review, and patient-level assessment. The measure is a screening and accountability tool, not a clinical audit.

### Part D: provider attribution

The plan attributes members to primary care practices. The practice rates are:

| Practice | Denominator | Numerator | Rate | Flag |
|---|---:|---:|---:|---|
| A | 4,000 | 3,000 | 75.00% | |
| B | 3,500 | 2,600 | 74.29% | |
| C | 2,500 | 1,800 | 72.00% | |
| D | 1,400 | 580 | 41.43% | Outlier, audit |
| E | 80 | 52 | 65.00% | Small denominator, suppress |

Practice D is far below the others. Before ranking it, the plan should check whether its denominator includes patients who should be excluded (for example, ESRD or hospice documented only in the EHR), whether its BP readings are in a different system, and whether its patients are sicker. A 41 percent rate in a practice with 1,400 patients is a signal to audit, not a conclusion about quality.

Practice E has 80 patients and a rate that could move by 10 points with eight patients. It should be reported with a suppression flag and not ranked.

### Part E: stratification

| Subgroup | Denominator | Numerator | Rate | Gap to overall |
|---|---:|---:|---:|---:|
| All members | 11,400 | 7,980 | 70.00% | 0.00 |
| Age 65+ | 6,200 | 4,526 | 73.00% | +3.00 |
| Age 18-64 | 5,200 | 3,454 | 66.42% | -3.58 |
| Black members | 2,100 | 1,344 | 64.00% | -6.00 |
| Hispanic members | 1,500 | 990 | 66.00% | -4.00 |
| Limited English proficiency | 700 | 420 | 60.00% | -10.00 |

The overall rate hides a 10-point gap for members with limited English proficiency. A programme that improves the overall rate by 2 points but does not reach that group will widen the gap. The plan should report the subgroup rates with confidence intervals and make the gap a secondary target.

### Part F: the payment context

A star rating is a payment rule. If the plan's quality bonus is 5 percent of a benchmark of $900 million, the bonus is $45 million, and the difference between 4 and 5 stars can be several million dollars. Against that, $548,640 to close the measure gap is small. The arithmetic explains why plans invest in measure improvement, and why a documentation project with a 300-patient gain can be worth more than a clinical programme with a 100-patient gain.

The comparison is illustrative, not a contract calculation. The actual bonus depends on the plan's rating, the benchmark, and the payment formula. The point is that the value of a measure point is not a clinical value; it is a contract value, and it should be computed with finance before the improvement budget is set.

### What would change the decision

If the measure were used for internal improvement only, with no payment attached, the documentation project would still be worth doing because it improves the data, but the clinical programme would need a clinical justification. If the payment were attached, the documentation project has a high return and the clinical programme has a lower one until the cheap gaps are closed. If the target were 75 percent rather than 86 percent, the clinical programme alone would clear it.

The decision is therefore a function of three things: the target, the payment, and the mix of gap types. The measure rate alone is not enough to decide.

### Part G: reconciliation to the steward

| Step | Local | Steward | Difference | Explanation |
|---|---:|---:|---:|---|
| Eligible | 12,000 | 12,000 | 0 | Same source |
| Exclusions | 600 | 640 | -40 | Hospice in EHR not in claims |
| Denominator | 11,400 | 11,360 | +40 | Exclusion difference |
| Numerator | 7,980 | 7,940 | +40 | Same clinical rule |
| Rate | 70.00% | 69.89% | +0.11 pp | Explained |

The difference is isolated at the exclusion step. The local team found 40 hospice patients in the EHR that the claims-based steward did not see. Adding them would lower the local denominator and raise the rate. The reconciliation shows that the two calculations agree on the numerator and differ only on the exclusion source.

### Part H: confidence intervals by practice

| Practice | Denominator | Rate | 95% interval | Use |
|---|---:|---:|---|---|
| A | 4,000 | 75.00% | 73.57-76.39 | Rank |
| B | 3,500 | 74.29% | 72.78-75.75 | Rank |
| C | 2,500 | 72.00% | 70.24-73.71 | Rank |
| D | 1,400 | 41.43% | 38.87-44.03 | Audit |
| E | 80 | 65.00% | 54.19-74.43 | Suppress |

Practice E's interval spans 20 percentage points and overlaps most of the other practices. A ranking that places E below A is not supported by the data. Practice D's interval does not overlap A, B, or C, so the difference is likely real and warrants an audit.

The interval, not the point estimate, determines whether a provider comparison is defensible.

A point estimate without an interval is a ranking without a margin of error.

That is the difference between a measurement and a claim.

## 7. Code, query, or calculation

The SQLite script below implements the CBP denominator and numerator, first with EHR readings only and then with a device feed. The output matches the base and documentation scenarios in Section 6.

Assumptions:

- Every member has a hypertension diagnosis; the eligible population is 12,000.
- Exclusions are 150 hospice, 200 ESRD, and 250 pregnancy.
- The EHR reading is dated 1 June 2025; the device reading is dated 1 November 2025.
- A reading is valid if the setting is outpatient; the device feed is also outpatient.
- The numerator uses the most recent valid reading per member.

```python
import sqlite3

con = sqlite3.connect(":memory:")
cur = con.cursor()
cur.executescript("""
CREATE TABLE members (
 member_id INTEGER PRIMARY KEY, htn INTEGER,
 hospice INTEGER, esrd INTEGER, pregnancy INTEGER
);
WITH RECURSIVE m(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM m WHERE n < 12000)
INSERT INTO members
SELECT n, 1,
    CASE WHEN n <= 150 THEN 1 ELSE 0 END,
    CASE WHEN n BETWEEN 151 AND 350 THEN 1 ELSE 0 END,
    CASE WHEN n BETWEEN 351 AND 600 THEN 1 ELSE 0 END
FROM m;

CREATE TABLE bp (
 member_id INTEGER, bp_date TEXT, systolic INTEGER, diastolic INTEGER,
 setting TEXT, source TEXT
);
WITH RECURSIVE m(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM m WHERE n < 12000)
INSERT INTO bp
SELECT n, '2025-06-01',
    CASE WHEN n <= 8580 THEN 120 ELSE 150 END,
    CASE WHEN n <= 8580 THEN 78 ELSE 95 END,
    'outpatient', 'ehr'
FROM m WHERE n > 600;

WITH RECURSIVE m(n) AS (SELECT 8581 UNION ALL SELECT n+1 FROM m WHERE n < 9080)
INSERT INTO bp
SELECT n, '2025-11-01',
    CASE WHEN n <= 8880 THEN 122 ELSE 148 END,
    CASE WHEN n <= 8880 THEN 76 ELSE 92 END,
    'outpatient', 'device'
FROM m;
""")

def rate(source_condition):
  q = f"""
  WITH denom AS (
   SELECT member_id FROM members
   WHERE hospice = 0 AND esrd = 0 AND pregnancy = 0
  ), ranked AS (
   SELECT b.member_id, b.systolic, b.diastolic,
       ROW_NUMBER() OVER (PARTITION BY b.member_id
                ORDER BY b.bp_date DESC) AS rn
   FROM bp b JOIN denom d ON b.member_id = d.member_id
   WHERE {source_condition}
  )
  SELECT COUNT(*),
      SUM(CASE WHEN systolic < 140 AND diastolic < 90 THEN 1 ELSE 0 END),
      ROUND(100.0 * SUM(CASE WHEN systolic < 140 AND diastolic < 90
               THEN 1 ELSE 0 END) / COUNT(*), 2)
  FROM ranked WHERE rn = 1;
  """
  return cur.execute(q).fetchone()

print("DENOMINATOR")
print(cur.execute("""
SELECT COUNT(*), SUM(hospice + esrd + pregnancy),
    SUM(CASE WHEN hospice = 0 AND esrd = 0 AND pregnancy = 0
        THEN 1 ELSE 0 END)
FROM members;
""").fetchone())

print("\nEHR ONLY")
print(rate("b.setting = 'outpatient' AND b.source = 'ehr'"))

print("\nWITH DEVICE FEED")
print(rate("b.setting IN ('outpatient','device')"))
con.close()
```

Expected output:

```text
DENOMINATOR
(12000, 600, 11400)

EHR ONLY
(11400, 7980, 70.0)

WITH DEVICE FEED
(11400, 8280, 72.63)
```

Read the output in three passes. The denominator row shows 12,000 eligible, 600 excluded, and 11,400 in the denominator. The EHR-only row shows 7,980 controlled and a 70.0 percent rate. The device row shows 8,280 controlled and a 72.63 percent rate, a 2.63 percentage point gain from documentation alone.

> **Note:** The `ROW_NUMBER` window function selects the most recent reading per member. The specification's same-date rule (lowest systolic and lowest diastolic) is not implemented here; with one reading per date, it does not change the result. A production query must implement the same-date rule explicitly, because a member with multiple readings on the same date can move from uncontrolled to controlled.

### Same-date rule

The specification uses the lowest systolic and lowest diastolic on the most recent date, even if the two values come from different readings. The following query implements that rule:

```sql
WITH ranked AS (
 SELECT member_id, bp_date, systolic, diastolic,
     ROW_NUMBER() OVER (PARTITION BY member_id ORDER BY bp_date DESC) AS date_rank
 FROM bp
 WHERE setting = 'outpatient'
), latest AS (
 SELECT * FROM ranked WHERE date_rank = 1
)
SELECT member_id, MIN(systolic) AS systolic, MIN(diastolic) AS diastolic
FROM latest
GROUP BY member_id;
```

The `MIN` on each column is the same-date rule. A query that takes the first row instead can misclassify a member whose first reading is elevated and whose second is controlled.

### PostgreSQL dialect: measure rate by provider

The following block is **PostgreSQL** syntax. It computes the rate by attributed provider with a confidence interval approximation.

```sql
-- PostgreSQL 15+
SELECT provider_id,
    COUNT(*) AS denominator,
    SUM(CASE WHEN systolic < 140 AND diastolic < 90 THEN 1 ELSE 0 END) AS numerator,
    ROUND(100.0 * SUM(CASE WHEN systolic < 140 AND diastolic < 90
             THEN 1 ELSE 0 END) / COUNT(*), 2) AS rate,
    ROUND(100.0 * SQRT(
     (SUM(CASE WHEN systolic < 140 AND diastolic < 90 THEN 1 ELSE 0 END)::numeric
     / COUNT(*)) *
     (1 - SUM(CASE WHEN systolic < 140 AND diastolic < 90 THEN 1 ELSE 0 END)::numeric
     / COUNT(*)) / COUNT(*)), 2) AS se_pct
FROM measure_population
GROUP BY provider_id
HAVING COUNT(*) >= 30
ORDER BY rate;
```

The `HAVING` clause suppresses providers below 30 patients. The standard error is a normal approximation; for small denominators, use the Wilson interval in Python or R.

### Reconciliation query

A measure implementation should reconcile its denominator to the payer's enrollment file. The query below compares the local denominator to an external count by month.

```sql
SELECT local.month, local.denominator, payer.denominator,
    local.denominator - payer.denominator AS difference
FROM local_denominator local
JOIN payer_denominator payer USING (month)
ORDER BY local.month;
```

A difference above 2 percent in any month is a signal to check the enrollment rule, the diagnosis window, or the exclusion code sets. The reconciliation is the first control in a measure audit.

## 8. Data, standards, and regulatory context

Quality measurement is governed by measure stewards, reporting programmes, and data standards. The steward defines the measure; the programme defines the payment; the standards define how the data are represented.

### Measure and reporting standards

| Standard or programme | What it defines | Binding status | What it does not define |
|---|---|---|---|
| HEDIS (NCQA) | Measure specifications, value sets, audit rules | Binding for NCQA reporting and many contracts | Local workflow |
| CMS Star Ratings | Measure set, weights, cut points, bonus | Binding for Medicare Advantage | Clinical target |
| CMS eCQM (CQL/FHIR) | Electronic measure logic and data requirements | Binding for CMS quality programmes | Local data capture |
| MIPS | Measure set and scoring for clinicians | Binding for MIPS participants | Improvement strategy |
| Medicaid Core Set | Measures for Medicaid reporting | Binding for state programmes | Local delivery model |
| Joint Commission | Accreditation measures | Accreditation condition | Payment |
| VSAC | Value set repository | Required terminology source for eCQMs | Local code mapping |
| USCDI | Minimum data elements for exchange | Required for certified health IT | Measure logic |

HEDIS specifications are published by NCQA in technical volumes and value sets. They define the measurement year, the eligible population, the exclusions, the numerator, and the allowable data sources. The specifications are updated annually; MY2026 introduced seven new measures and retired two, including the Asthma Medication Ratio and Medical Assistance With Smoking and Tobacco Cessation. The MY2026 specifications also reformatted measure logic around FHIR terminology: initial population, denominator, denominator exclusions, measurement period, and person.

CMS eCQMs are expressed in Clinical Quality Language and packaged in FHIR `Measure` and `Library` resources. The `Measure` resource identifies the populations and the CQL expression that defines each; the `Library` resource contains the CQL and the compiled ELM. The logic is computable and testable, which is an improvement over prose, but it still depends on the local data mapping and the value sets.

### The specification-to-implementation gap

The gap between the official specification and the local implementation has four sources.

1. **Terminology mapping.** The value set lists codes; the local system may use a different code system, a local code, or a free-text problem. The mapping determines who enters the denominator.
2. **Data capture.** The measure may require a structured field that the local workflow does not populate. The patient is in the denominator and fails the numerator, even if the care was delivered.
3. **Timing.** The specification may use the most recent value in the measurement year; the local query may use the last value in the extract, which can be from a different period.
4. **Exclusions.** The exclusion code sets and windows may be applied differently, changing the denominator.

Each source is a documented local choice. The measure report should state the choice and its effect. An audit that finds a difference without a documented choice will treat it as an error.

### Binding versus convention

| Element | Binding? | Source | Local choice |
|---|---|---|---|
| Measure definition | Yes, for the programme | Steward | None if reporting officially |
| Value sets | Yes, for eCQMs | VSAC | Local mapping to source codes |
| Measurement year | Yes | Steward | None |
| Exclusion application | Yes | Steward | Data source and code mapping |
| Data source | Programme-specific | Steward | Claims, EHR, supplemental |
| Cut point | Yes, for the rating | CMS | None |
| Internal target | No | Local | Quality committee |
| Equity stratification | Programme-specific | Payer or accreditation | Subgroups |

### Privacy and audit

Measure calculation uses identifiable data and is subject to HIPAA. A payer or provider that reports a measure to CMS or NCQA must comply with the programme's data submission rules and audit requirements. HEDIS audits verify that the measure was calculated according to the specification; an audit finding can require a restatement.

> **Note:** Regulatory citations are dated to 25 October 2026. HEDIS and Star specifications change annually, and the 2027 specifications may differ from the 2026 versions. Verify the current measurement year before using a rate in a payment calculation.


### Binding versus convention

| Element | Binding? | Source | Local choice |
|---|---|---|---|
| Measure definition | Yes, for the programme | Steward | None if official |
| Value sets | Yes, for eCQMs | VSAC | Source-code mapping |
| Measurement year | Yes | Steward | None |
| Exclusion application | Yes | Steward | Data source |
| Data source | Programme-specific | Steward | Claims, EHR, supplemental |
| Cut point | Yes, for the rating | CMS | None |
| Internal target | No | Local | Quality committee |

### What the standards do not settle

1. **The clinical target.** A cut point is a relative threshold, not a treatment goal.
2. **The local workflow.** The specification does not say how to capture the data.
3. **The gap taxonomy.** The measure does not distinguish a clinical gap from a documentation gap.
4. **The equity action.** A subgroup difference is a finding, not a remedy.
5. **The improvement budget.** The measure does not say which intervention is worth funding.

## 9. Industry practice and case evidence

### Documented specification facts

The 2026 Star Ratings technical notes and measures and weights documents define the measure set and cut points. Representative 2026 5-star cut points include Controlling High Blood Pressure at 86 percent, Breast Cancer Screening at 84 percent, Colorectal Cancer Screening at 78 percent, Plan All-Cause Readmissions at 7 percent or lower, Transitions of Care at 79 percent, and Medication Reconciliation at 87 percent. The cut points are set by a clustering and resampling method with guardrails, not by a clinical consensus.

The 2026 Star Ratings fact sheet reports that about 40 percent of MA-PD contracts (207 contracts) earned 4 or more stars, and about 64 percent of MA-PD enrollees were in 4-plus-star contracts. The average overall MA-PD rating was 3.98. These are plan-level facts; they do not establish that a particular measure rate reflects better care.

HEDIS MY2026 introduced seven new measures, including four risk-adjusted acute hospitalization measures after outpatient orthopedic, general, colonoscopy, and urologic surgery (HFO, HFG, HFC, HFU), a Disability Description of Membership measure, Follow-Up After Acute and Urgent Care Visits for Asthma, and Tobacco Use Screening and Cessation Intervention. It retired the Asthma Medication Ratio and Medical Assistance With Smoking and Tobacco Cessation. The changes are documented in NCQA's MY2026 summary; a trend that crosses the change is not a clinical trend.

### Documented measurement-gap evidence

The CBP specification itself documents the conditions under which a patient is excluded or counted: hospice, palliative care, ESRD, pregnancy, institutional status, and frailty with advanced illness. It also specifies the most recent BP, the setting, and the same-date rule. The complexity is the evidence that a local implementation can differ from the official measure without anyone intending it.

The eCQM Logic and Implementation Guidance from CMS describes the population bases, scoring types, and CQL constructs used in electronic measures. It is a practical demonstration that the measure is code, not prose. Two implementations of the same CQL can still differ if the local data mapping differs.


### Evidence table

| Source | What it documents | What it does not establish |
|---|---|---|
| 2026 Star technical notes | Cut points and measure methodology | That a rate reflects care |
| 2026 Star fact sheet | Plan-level rating distribution | Provider-level performance |
| HEDIS MY2026 summary | Seven new measures, two retired | Local implementation |
| CBP specification | Denominator, exclusions, numerator rules | Documentation quality |
| eCQM logic guidance | CQL populations and scoring | Local data mapping |

The table separates the official rules from the local implementation. The rules are necessary but not sufficient for a valid rate.


### Inference

> **Inference:** A plan that improves a measure by 2 percentage points cannot conclude that care improved. The improvement may be documentation, denominator, or data-source change. The only way to separate them is a gap taxonomy and a chart-review sample. This is an inference from the measure logic, not a finding of a specific study.

### What mature organisations do

They maintain a measure library with the specification, the local implementation, the value sets, the data sources, and the validation result. The library is versioned by measurement year. A measure that is not in the library is not reported.

They separate the measure report from the improvement report. The measure report states the official rate; the improvement report states the gap taxonomy, the interventions, and the clinical outcomes. The two are reviewed by different committees. This separation prevents a documentation gain from being presented as a clinical gain.

They audit a sample of failures every year. The audit classifies each failure as clinical, documentation, data, eligibility, or attribution. The mix determines the improvement budget. An organisation that does not audit is guessing.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Using the wrong measure version | The rate changes at a version update | Specification changed | Comparison invalid | Version the measure and restate |
| Denominator includes ineligible patients | Denominator exceeds the payer's | Enrollment or diagnosis rule wrong | Rate understated | Reconcile to payer |
| Exclusion code set incomplete | Exclusion rate is low | Missing hospice or ESRD codes | Sicker patients remain | Audit code sets |
| Missing BP treated as controlled | Rate is implausibly high | Null counted as pass | Performance overstated | Count missing as failure |
| Missing BP treated as exception | Exception rate is high | Local rule not in specification | Performance overstated | Apply only documented exceptions |
| ED readings included | Rate falls after an ED data feed | Setting rule ignored | Performance understated | Filter by setting |
| First reading used instead of most recent | Rate differs from the steward | Timing rule wrong | Comparison invalid | Use the specified rule |
| Same-date rule ignored | A few patients change status | Lowest values not selected | Small rate difference | Implement the rule |
| External readings omitted | Documentation gap looks clinical | Device feed not linked | Outreach over-investment | Link the feed |
| Documentation gain reported as clinical | Rate rises with no clinical change | Data capture improved | Misleading improvement claim | Report gap type separately |
| Small denominators ranked | A small practice tops the list | Random variation | Unfair ranking | Suppress below 30 |
| Attribution changes mid-year | Provider rate jumps | Members reassigned | Comparison invalid | Monthly attribution |
| Measure mixing across years | Trend break at January | Version change | False trend | Segment the trend |
| Cut point treated as clinical target | Teams chase the number | Payment rule, not guideline | Over-treatment or gaming | Separate target from cut point |
| No audit of failures | Gap taxonomy unknown | No chart review | Wrong intervention | Annual audit |

Three failures deserve emphasis.

**Missing data counted as controlled.** A query that uses `COALESCE` or a left join and treats a null BP as a pass will produce an implausibly high rate. The correct rule is that missing data is a failure unless a documented exception applies. The error is easy to make and hard to see because the rate looks good.

**Documentation gain reported as clinical.** A device feed or a note-extraction project can raise the rate by several points without changing care. If the improvement is reported as clinical, the organisation learns the wrong lesson and may scale a programme that did nothing. The mitigation is a gap taxonomy and a clear label on the data-quality gain.

**The wrong denominator.** A rate computed on the eligible population without exclusions is not the measure. It includes hospice and ESRD patients, whose care is different, and it understates performance. The denominator is the measure, and the exclusion counts must be reported.


### Pre-mortem for a measure report

1. Which measure version and measurement year?
2. Which data sources, and what is missing from each?
3. What are the exclusion counts by reason?
4. Is the numerator the most recent valid reading?
5. What share of failures is documentation versus clinical?
6. Is the denominator large enough to report?

A report that cannot answer all six should be labelled provisional.
> **Note:** A measure rate is not a clinical outcome. It is a count of patients who meet a rule in a dataset. The clinical outcome requires a separate measurement, with a clinical definition and a comparison.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test |
|---|---|---|---|
| The specification is the current version | The rate must be comparable | Annual update | Check the steward's year |
| The value sets map to local codes | The denominator depends on it | Local codes, free text | Reconcile code counts |
| Enrollment is accurate | Eligibility depends on it | Retroactive eligibility | Reconcile to payer |
| Exclusions are documented | The denominator depends on it | Missing codes | Audit exclusion counts |
| BP readings are valid | The numerator depends on it | Setting, method, duplicates | Apply the validity rules |
| The most recent rule is applied | The numerator depends on it | Last value in extract | Test on a sample |
| The data source is stable | Trends depend on it | New feed, interface change | Mark source changes |
| The denominator is large enough | Rates must be stable | Small practices | Suppress below 30 |
| Attribution is stable | Provider rates depend on it | Mid-year moves | Monthly attribution |
| The audit sample is representative | Gap taxonomy depends on it | Convenience sample | Random sample |


### Assumption owners

| Assumption | Owner | Review cadence |
|---|---|---|
| Measure version | Quality analytics | Annual |
| Value set mapping | Informatics | Annual |
| Enrollment rule | Eligibility operations | Monthly |
| Exclusion code sets | Quality analytics | Annual |
| Numerator rule | Quality analytics | Each release |
| Gap taxonomy | Quality improvement | Quarterly |

The limits follow from the assumptions. A measure is a specification applied to data, and its validity is bounded by the specification, the data, and the implementation. It cannot tell you whether the patient received good care, only whether the recorded data met the rule. It can tell you how many patients met the rule, with what uncertainty, and how the count changes under different data and exclusion choices.

A second limit is that the measure is a ratio, and ratios are sensitive to the denominator. A small change in the exclusion set can move the rate more than a large change in care. The analyst should report the denominator and the exclusion counts before the rate.

A third limit is that the cut point is a relative ranking. A plan at 84 percent can be clinically excellent and still miss a 5-star cut point of 86 percent. The measure is a payment rule, and the clinical target is a separate decision.

## 12. Exercises

### Exercise 1. Parse a measure specification (recall)

Using the Controlling High Blood Pressure specification in Section 6, write a table with the eligible population, the diagnosis window, the enrollment rule, the visit rule, the exclusions, the numerator rule, the setting rule, and the measurement period.

**Deliverable:** an eight-row table with the rule and the implementation question.

**Acceptance criteria:** every rule has a source; at least three ambiguities are identified; the measurement period is explicit.

**Time:** 30 minutes.

### Exercise 2. Compute the rate with exclusions (application)

A plan has 8,000 members with hypertension. Exclusions are 150 hospice, 100 ESRD, and 150 pregnancy. The numerator is 5,320. Compute the denominator, the rate, and the number of additional controlled patients needed to reach 80 percent.

**Deliverable:** a calculation table.

**Acceptance criteria:** the denominator subtracts all three exclusions; the rate uses the denominator; the additional patients are computed from the target.

**Time:** 45 minutes.

### Exercise 3. Quantify a documentation change (application)

The plan finds 300 members with a device reading that is not in the structured field. 65 percent of the readings are controlled. Compute the new numerator, the new rate, and the change in percentage points. State whether the change is clinical or documentation.

**Deliverable:** a calculation and a one-sentence classification.

**Acceptance criteria:** the numerator increases by 195; the denominator is unchanged; the classification is justified by the source of the reading.

**Time:** 30 minutes.

### Exercise 4. Audit the denominator (application)

The plan discovers that 250 members in the denominator had a coverage gap longer than 45 days. Recompute the denominator and the rate from Exercise 2, and state the direction of the bias.

**Deliverable:** a calculation and a bias statement.

**Acceptance criteria:** the denominator falls by 250; the rate rises; the direction is explained by the enrollment rule.

**Time:** 30 minutes.

### Exercise 5. Implement the measure in SQL (application)

Using the schema from Section 7, add a `continuous` flag and exclude members with `continuous = 0`. Recompute the denominator and rate. Then add a second reading for 100 members and implement the most-recent rule. Run both queries and paste the output.

**Deliverable:** the queries, the output, and a one-paragraph interpretation.

**Acceptance criteria:** the queries execute; the most-recent rule uses a window function; the denominator and numerator are reported separately.

**Time:** 90 minutes.

### Exercise 6. Build a gap taxonomy (critique)

A sample of 50 patients who failed the CBP numerator finds: 20 with a controlled reading in a note, 12 with a reading at an outside clinic, 10 with no reading in the year, 5 with an elevated reading and no treatment change, and 3 with an exclusion that should have applied. Build the gap taxonomy, compute the share of each type, and recommend the first intervention.

**Deliverable:** a taxonomy table and a recommendation.

**Acceptance criteria:** the categories are mutually exclusive; the shares sum to 100 percent; the recommendation matches the largest category.

**Time:** 45 minutes.

### Exercise 7. Build a measure business case (application)

A plan needs to move from 70 percent to 86 percent on a denominator of 7,600. The programme cost is $300 per additional controlled patient. Compute the additional patients needed, the total cost, and the cost per measure point. If the star bonus is worth $2 million, is the programme worth it?

**Deliverable:** a business case with arithmetic and a recommendation.

**Acceptance criteria:** the target numerator is computed from the denominator; the cost per point is computed; the recommendation compares cost and bonus and states the assumptions.

**Time:** 60 minutes.

### Exercise 8. Design an equity report (design)

A measure rate is 72 percent overall, 68 percent for Black members, 70 percent for Hispanic members, and 64 percent for members with limited English proficiency. Design an equity report with subgroups, metrics, suppression rules, and actions.

**Deliverable:** a two-page report template.

**Acceptance criteria:** at least four subgroups; small-cell suppression; actions distinguish data, documentation, and clinical gaps; the review body is named.

**Time:** 60 minutes.


### Assessment notes

Exercises 1 to 4 can be completed individually and checked against Section 13. Exercise 5 requires SQL. Exercises 6 to 8 are assessed with the rubric. Add 20 minutes to Exercise 5 if the learner must create the schema.

## 13. Self-check answers

### Exercise 1

A strong table includes: age 18-85 as of 31 December; hypertension diagnosis in the prior year through 30 June; continuous enrollment with a gap of no more than 45 days; two outpatient visits or one visit plus a diagnosis; exclusions for hospice, palliative care, ESRD, pregnancy, institutional status, and frailty with advanced illness; numerator of the most recent BP below 140/90; outpatient setting only; measurement year. The ambiguities include the diagnosis source (claims or problem list), the visit definition (telehealth), the BP method (manual or device), and the exclusion window.

### Exercise 2

```text
Denominator = 8,000 - 150 - 100 - 150 = 7,600
Rate = 5,320 / 7,600 = 70.00%
Target at 80% = 7,600 x 0.80 = 6,080
Additional controlled = 6,080 - 5,320 = 760
```

The denominator is 7,600, not 8,000. Using 8,000 would produce 66.5 percent and understate performance by 3.5 points.

### Exercise 3

```text
Additional controlled = 300 x 0.65 = 195
New numerator = 5,320 + 195 = 5,515
New rate = 5,515 / 7,600 = 72.57%
Change = +2.57 percentage points
```

The change is documentation, not clinical. The patients' blood pressure did not change; the readings were already controlled but were not in the structured field. The gain is real for measurement and should be labelled as a data-quality improvement.

### Exercise 4

```text
Corrected denominator = 7,600 - 250 = 7,350
Corrected rate = 5,320 / 7,350 = 72.38%
```

The rate rises by 2.38 points because the removed members had a coverage gap and should not have been in the denominator. The direction is predictable: including ineligible members usually lowers the rate, because ineligible members often have less complete data.

### Exercise 5

A strong implementation adds a `continuous` column and filters `continuous = 1` in the denominator. The most-recent query uses `ROW_NUMBER() OVER (PARTITION BY member_id ORDER BY bp_date DESC)` and selects `rn = 1`. The output should show the denominator, numerator, and rate for both the base and the corrected versions. A weak implementation uses `MAX(bp_date)` without joining back to the value, which can pair a date with the wrong reading.

### Exercise 6

| Gap type | Count | Share | First action |
|---|---:|---:|---|
| Documentation (note) | 20 | 40% | Note extraction and structured entry |
| Data (outside clinic) | 12 | 24% | Health information exchange linkage |
| Clinical (no reading) | 10 | 20% | Outreach and testing |
| Clinical (uncontrolled) | 5 | 10% | Treatment intensification |
| Eligibility (exclusion) | 3 | 6% | Apply the exclusion |

The largest category is documentation at 40 percent. The first intervention should be a documentation and data-linkage project, not a clinical outreach programme. After that, the remaining clinical gap is 30 percent of failures.

### Exercise 7

```text
Target numerator = 7,600 x 0.86 = 6,536
Additional controlled = 6,536 - 5,320 = 1,216
Cost = 1,216 x $300 = $364,800
Cost per percentage point = $364,800 / 16 = $22,800
```

If the star bonus is worth $2 million, the programme costs 18 percent of the bonus and is worth doing on a financial basis, provided the bonus is attributable to this measure and the effect is real. If the bonus is shared across many measures, the attribution is weaker and the case should be made at the portfolio level.

### Exercise 8 rubric

A strong report includes subgroups by race, ethnicity, language, payer, age, and disability; rates with confidence intervals; suppression below 30; a comparison to the overall rate and the prior year; and actions that distinguish documentation, data, and clinical gaps. It names the quality committee or health equity committee as the review body and states the cadence.

## 14. Interview questions

**1. What is a quality measure?**
A strong answer defines it as a specification with a denominator, exclusions, numerator, data sources, and period, owned by a steward and used for accountability or payment. A weak answer says it is "a way to measure quality".

**2. How do you compute a measure rate?**
A strong answer computes the denominator after exclusions, the numerator under the specified rule, and the ratio, with a confidence interval and the denominator size. A weak answer divides the numerator by the whole population.

**3. Why do exclusions matter?**
A strong answer explains that exclusions remove patients for whom the measure is not meaningful, that they change the denominator, and that a missing exclusion biases the rate. A weak answer says exclusions make the measure fair.

**4. How can a measure improve while care does not?**
A strong answer gives the three mechanisms (documentation, denominator, numerator rule) and an arithmetic example. A weak answer says "better documentation".

**5. What is the difference between a specification and an implementation?**
A strong answer explains that the specification is the official rule and the implementation is the local query, data mapping, and workflow, and that the two can differ. A weak answer treats them as the same.

**6. How do you choose the data source for a measure?**
A strong answer follows the programme's allowed sources, maps each data element, and reports coverage and missingness. A weak answer uses whatever is easiest.

**7. What is a star cut point?**
A strong answer explains that it is a relative threshold for a rating, set by a statistical method, not a clinical target. A weak answer treats it as a treatment goal.

**8. How do you audit a measure?**
A strong answer reconciles the denominator to the payer, audits the exclusion code sets, samples failures for the gap taxonomy, and recomputes the rate independently. A weak answer recomputes the same query.

**9. How do you handle missing data?**
A strong answer counts it as a failure unless a documented exception applies, and reports the missingness rate. A weak answer excludes missing patients from the denominator.

**10. How do you report a small denominator?**
A strong answer reports the count and the rate with a suppression flag and a confidence interval, and does not rank. A weak answer reports the rate without qualification.

**11. How do you separate a documentation gain from a clinical gain?**
A strong answer uses the source of the new data, a chart-review sample, and a gap taxonomy, and labels the gain. A weak answer assumes any rate increase is clinical.

**12. What would you do if the local rate differs from the payer's by 3 percent?**
A strong answer reconciles step by step (eligible, exclusions, denominator, numerator), identifies the definitional difference, and documents it before reporting. A weak answer averages the two rates.


**13. How do you explain a measure to a clinician?**
A strong answer states the denominator, the exclusions, the numerator rule, and the data source, and gives an example of a patient who would fail despite good care. A weak answer quotes the rate without the rule.

**15. What is a denominator exception, and how does it differ from an exclusion?**
A strong answer explains that an exception keeps the patient in the denominator but removes them from the failure count, while an exclusion removes them from the denominator entirely. A weak answer uses the terms interchangeably.

**14. What is the first thing you check when a rate jumps?**
A strong answer checks the specification version, the data source, the exclusion counts, and the denominator before checking the numerator. A weak answer assumes care improved.

## 15. Key terms

- **Measure:** a specification for quantifying care or outcomes.
- **Measure steward:** the organisation that owns and maintains a measure.
- **HEDIS:** NCQA's Healthcare Effectiveness Data and Information Set.
- **Star Rating:** CMS's summary rating for Medicare Advantage and Part D.
- **eCQM:** electronic clinical quality measure.
- **CQL:** Clinical Quality Language.
- **FHIR Measure:** the FHIR resource that describes a computable measure.
- **FHIR Library:** the FHIR resource that holds CQL and ELM.
- **Denominator:** the eligible population after exclusions.
- **Numerator:** patients meeting the clinical criterion.
- **Exclusion:** a condition that removes a patient from the denominator.
- **Denominator exception:** a clinical reason the numerator does not apply.
- **Measurement period:** the window for eligibility and numerator.
- **Continuous enrollment:** enrollment without a gap longer than allowed.
- **Value set:** the codes that define a clinical concept.
- **VSAC:** Value Set Authority Center.
- **Supplemental data:** data beyond claims used for a measure.
- **Attribution:** assignment of a patient to a provider or plan.
- **Cut point:** the threshold for a star or payment tier.
- **Benchmark:** a comparison value, often a percentile.
- **Performance rate:** numerator divided by denominator.
- **Confidence interval:** the uncertainty around a rate.
- **Small denominator:** a denominator too small for a stable rate.
- **Stratification:** reporting a measure by subgroup.
- **Risk adjustment:** adjusting for patient complexity.
- **Documentation gap:** care delivered but not recorded in the measure field.
- **Data quality gap:** a missing or mis-mapped data element.
- **Clinical gap:** care not delivered.
- **Chart review:** manual verification of a sample.
- **Audit:** an independent check of the measure calculation.
- **Hybrid measure:** a measure that combines claims and medical record data.
- **Electronic measure:** a measure computed from electronic data.
- **Numerator exception:** a rule that allows a patient to be counted despite missing data.
- **Star year:** the rating year, based on a measurement year.
- **Measurement year:** the year in which care is measured.
- **HEDIS MY:** HEDIS measurement year.
- **Quality bonus:** the payment tied to a star rating.
- **Withhold:** a portion of payment held pending performance.
- **Attribution window:** the period used to assign patients.
- **Lookback:** the period before the measurement year for eligibility.
- **Most recent:** the latest value in the measurement year.
- **Valid reading:** a reading that meets the specification's setting and method rules.
- **Remote reading:** a home or device reading.
- **Structured field:** a coded or numeric field.
- **Note extraction:** pulling a value from free text.
- **Registry:** a list of patients for a condition or programme.
- **Care gap:** a patient who fails the numerator.
- **Gap closure:** moving a patient into the numerator.
- **Measure specification:** the official definition.
- **Measure implementation:** the local computation.
- **Measure version:** the year and revision of the specification.

## 16. Sources

All URLs were accessed on 25 October 2026.

### HEDIS and NCQA

1. NCQA. HEDIS Measures and Technical Resources. https://www.ncqa.org/hedis/measures/
2. NCQA. HEDIS MY 2026: What's New, What's Changed, What's Retired. https://www.ncqa.org/blog/hedis-my-2026-whats-new-whats-changed-whats-retired/
3. NCQA. 2026 Health Plan Ratings Required HEDIS, CAHPS and HOS Measures. https://wpcdn.ncqa.org/www-prod/2026-HPR-List-of-Required-Performance-Measures_April-2025-Posting.pdf
4. NCQA. HEDIS MY 2026 Public Reporting Memo. https://wpcdn.ncqa.org/www-prod/wp-content/uploads/HEDIS-MY-2026-Public-Reporting-Memo.pdf
5. NCQA. Controlling High Blood Pressure (CBP) measure summary. https://wpcdn.ncqa.org/www-prod/wp-content/uploads/2020/02/20200212_14_CBP.pdf

### CMS Star Ratings and measures

6. CMS. 2026 Part C & D Star Ratings Technical Notes. https://www.cms.gov/files/document/2026-star-ratings-technical-notes.pdf
7. CMS. 2026 Star Ratings Measures and Weights. https://www.cms.gov/files/document/2026-star-ratings-measures.pdf
8. CMS. 2026 Star Ratings Fact Sheet. https://www.cms.gov/files/document/2026-star-ratings-fact-sheet.pdf
9. CMS. Quality Measures. https://www.cms.gov/medicare/quality/measures
10. CMS. Quality ID #236: Controlling High Blood Pressure (MIPS CQM). https://qpp.cms.gov/docs/QPP_quality_measure_specifications/CQM-Measures/2024_Measure_236_MIPSCQM.pdf

### eCQM and standards

11. CMS eCQI Resource Center. eCQM Logic and Implementation Guidance for 2027 Reporting/Performance. https://ecqi.healthit.gov/sites/default/files/2027-eCQM-Logic-and-Guidance.pdf
12. HL7. Quality Measure STU2 for FHIR R4 Implementation Guide. https://hl7.org/fhir/us/cqfmeasures/STU2/measure-conformance.html
13. HL7. Quality Measure IG STU5. https://hl7.org/fhir/us/cqfmeasures/2024Jan/measure-conformance.html
14. National Library of Medicine. Value Set Authority Center. https://vsac.nlm.nih.gov/
15. ONC. US Core Data for Interoperability. https://www.healthit.gov/isa/us-core-data-interoperability-uscdi
16. HL7. FHIR Release 4. https://hl7.org/fhir/

### Medicaid and programme context

17. Medicaid.gov. Calculating the Controlling High Blood Pressure (CBP) Measure in the Adult and Health Home Core Sets. https://www.medicaid.gov/medicaid/quality-of-care/downloads/CBP-Fact-Sheet.pdf
18. CMS. Core Set of Health Care Quality Measures for Medicaid. https://www.medicaid.gov/medicaid/quality-of-care/performance-measurement/adult-and-child-health-care-quality-measures/index.html
19. Joint Commission. Accreditation and quality measurement. https://www.jointcommission.org/
20. HHS. HIPAA. https://www.hhs.gov/hipaa/index.html

> **Note:** The HEDIS and Star specifications are updated annually. A rate computed under one measurement year is not comparable with a rate computed under another without restatement.

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`05_capacity-and-patient-flow.md`](05_capacity-and-patient-flow.md) — capacity, boarding, and flow.
- Next topic: [`07_revenue-cycle.md`](07_revenue-cycle.md) — denials, prior authorization, and cash.
- Related topic: [`08_evaluating-operational-interventions.md`](08_evaluating-operational-interventions.md) — evaluating whether a measure improved care.
- Root survey file: [`../../08_ehr_claims_operations.md`](../../08_ehr_claims_operations.md)
- Labs: [`../../labs/`](../../labs/)

### Reading path

| If you need to | Read | Then |
|---|---|---|
| Report a HEDIS measure | Sections 5 and 6 | The NCQA specification |
| Build an eCQM | Sections 5 and 8 | The CQL guidance |
| Explain a rate change | Sections 6 and 10 | Topic 08 for the evaluation |
| Design a gap programme | Sections 6 and 9 | Topic 04 for the operational model |
| Report to a quality committee | Sections 6 and 11 | Topic 01 for the payment context |
| Audit a measure | Sections 5, 9, and 11 | The NCQA and CMS sources |
| Separate documentation from care | Sections 6 and 10 | Topic 08 for the evaluation |

Quality measures are the bridge between clinical care and payment. They are also the place where documentation, data, and clinical performance are most easily confused. A reader who can separate the three has learned the main lesson of this topic.

A measure report is a claim about a population, and the claim is only as strong as the specification and the data behind it. The next topic moves from the measure to the cash: how the claim is paid, denied, and appealed.

The measure is a lens on the population, not a verdict on the clinician.

The difference changes both the rate and the interpretation.

| Change | Denominator | Numerator | Rate |
|---|---|---|---|
| Documentation +300 | Same | +300 | +2.63 pp |
| Exclusion +200 | -200 | -80 | +0.52 pp |
| Enrollment +300 | +300 | Same | -1.89 pp |

The table is a reminder that the same rate can move in opposite directions for different reasons.

That is why the denominator and numerator should be reported before the rate.
