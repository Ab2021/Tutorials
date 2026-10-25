# EHR Cohorts

> **Module 08 — EHR, Claims, and Health System Operations**
> Topic 02 of 09
> Estimated study time: 3 hours
> Prerequisites: Topic 01
> Version 2026.3

## 1. Why this topic exists

An EHR cohort is a definition of a population written in code. It decides who is in the denominator of every downstream metric: the readmission rate, the care-gap percentage, the quality bonus, the outreach list, the model's training set. When the definition is wrong, every number that follows is wrong in the same direction, and the error is usually invisible because the cohort looks plausible. A diabetes registry with 24,800 patients is not obviously too small or too large. Only a validation sample, a second data source, or a chart review can reveal that 12 percent of the patients do not have diabetes and 14 percent of the true cases are missing.

The consequences are operational, not academic. A care-gap programme that mails reminders to 24,800 patients spends money on people who do not need the service and misses people who do. A quality measure computed on a cohort that excludes patients with a diagnosis recorded only in a specialist note will overstate performance, because the sickest patients are the ones most likely to have their diagnosis recorded outside the primary care EHR. A readmission model trained on a cohort that excludes transfers will learn a population that does not match the patients it scores at discharge. In each case the code ran, the dashboard rendered, and the decision was wrong.

Three failure patterns account for most cohort defects.

**Code-set drift.** Clinical concepts are represented by sets of diagnosis, procedure, laboratory, and medication codes. The code sets change when ICD-10-CM is updated, when a new drug appears, when a lab switches assays, or when a health system migrates from one EHR vendor to another. A cohort definition written in 2019 and re-run in 2026 may silently lose or gain patients because the underlying code set was never versioned. The eMERGE network, a National Human Genome Research Institute consortium that developed and shared phenotype algorithms across multiple sites, documented this problem in its validation work: performance varied across sites and across implementations even when the algorithm was nominally the same. Newton et al. validated 13 EHR-derived phenotype algorithms across five sites and reported positive predictive values ranging from 67.7 to 100 percent, with about 75 percent of 51 algorithm reviews achieving at least 90 percent and three below 80 percent (JAMIA, 2013).

**Temporal leakage.** A cohort is defined by an index event and a set of windows. If a feature used to define the cohort is recorded after the index event, or if a patient is classified using information that only exists because the outcome occurred, the cohort is contaminated. The classic case is a "sepsis cohort" that includes patients whose blood cultures were ordered after the diagnosis was made, or a "high-risk pregnancy" cohort that includes patients whose risk was identified at delivery. Temporal leakage produces excellent model performance in development and poor performance in deployment, because the deployment setting does not have the future information.

**Absence of evidence treated as evidence of absence.** EHR data record what was documented and coded, not what happened. A patient without an HbA1c result may be healthy, may have been tested at an outside lab, may have declined, or may have been lost to follow-up. A cohort definition that treats a missing result as "not tested" conflates four different states. The resulting care-gap rate is a measure of documentation and data linkage as much as of clinical care.

The scale of the problem is documented. Spratt et al. applied eight EHR-based diabetes phenotype definitions to 173,503 patients at Duke and found that 22,679 met at least one definition, with sensitivity ranging from 62 to 94 percent and specificity from 95 to 99 percent against an expert chart-review standard (JAMIA, 2017). The choice of definition, not the choice of model, determined the size and composition of the cohort. Bhasin et al. showed that validation studies which sample only algorithm-positive patients produce biased sensitivity estimates: a naive sensitivity of 0.942 fell to a bias-adjusted 0.620 when the sampling design was corrected, while specificity rose from 0.979 to 0.999 (ACI Open, 2024). The lesson is that a cohort's reported performance depends on how it was validated, not only on how it was written.

This topic teaches cohort construction as an engineering and validation procedure. The output is a versioned cohort definition with a code set, an index event, explicit windows, exclusions, a validation sample, and a performance estimate. The definition is an artefact that can be reviewed, re-run, and compared across sites. Without it, the cohort is a query that happened to run once.


The cost of a bad cohort is not only the wrong dashboard. A cohort that is too narrow understates the population eligible for a quality programme, which can reduce the measured performance and the payment that depends on it. A cohort that is too broad includes patients who do not have the condition, which increases outreach cost and can expose patients to unnecessary tests. In a value-based contract, both errors affect the benchmark and the shared-savings calculation. The cohort is therefore a financial object as well as a clinical one, and its validation is part of the financial control system.

A second cost is trust. Clinicians who receive a registry list containing patients they know do not have the condition stop using the list. Rebuilding that trust takes longer than building the cohort. The validation sample is the evidence that prevents the first bad list from being sent.
> **Note:** A cohort definition is not a clinical guideline. It is an operational approximation of a clinical concept, built from coded data, and it will misclassify some patients. The relevant question is not "is it correct" but "is the misclassification small enough, and random enough, for the decision at hand".

## 2. Learning outcomes

By the end of this topic you can:

1. **Write** a cohort specification with a base population, index event, eligibility criteria, exclusions, lookback window, follow-up window, and code sets.
2. **Distinguish** an index event from a prevalent case and explain why the distinction changes the denominator.
3. **Build** a code set from ICD-10-CM, CPT, HCPCS, LOINC, RxNorm, and SNOMED CT, and **document** its provenance and version.
4. **Detect** temporal leakage in a cohort definition and **redesign** the windows so that every feature is available at the index time.
5. **Design** a validation sample that estimates positive predictive value, sensitivity, and specificity without verification bias.
6. **Compute** the corrected prevalence and care-gap rate after accounting for misclassification.
7. **Implement** a cohort in SQL against a normalized schema and **reproduce** the counts from a data dictionary.
8. **Critique** a published phenotype algorithm for portability, missing data handling, and version control.

Each outcome maps to an exercise in Section 12 or a self-check item in Section 13. Outcomes 4 and 5 are the ones that most often distinguish a careful analyst from one who has only written queries.

## 3. The core mental model

A cohort is a funnel applied to a base population at a point in time, followed by a longitudinal window. The funnel decides who is eligible; the window decides what is observed. Both must be specified before the code is written.

```text
BASE POPULATION
 All patients with at least one encounter in the data window
    |
    v
ELIGIBILITY
 Age, sex, payer, geography, enrollment, setting
    |
    v
INDEX EVENT
 First qualifying diagnosis / lab / medication / encounter
    |
    v
EXCLUSIONS
 Hospice, pregnancy, prior history, transfers, invalid records
    |
    v
COHORT (prevalent or incident)
    |
    +----------------------+----------------------+
    v           v           v
 LOOKBACK WINDOW     INDEX DATE      FOLLOW-UP WINDOW
 (features, comorbidity) (time zero)     (outcomes, care gaps)
```

The funnel is read from top to bottom as inclusion and from bottom to top as attribution. Every exclusion is a claim about the clinical question: excluding hospice patients from a readmission cohort is appropriate because readmission is not a meaningful quality signal for them; excluding transfers is appropriate because a transfer is not a discharge to the community. An exclusion that is not justified by the clinical question is a source of selection bias.

Three temporal concepts carry most of the analytic weight.

**Index date and time zero.** The index date is the date the patient enters the cohort. Time zero is the instant at which the prediction or measurement is made. For a prevalent cohort, the index date may be the first qualifying diagnosis, and time zero may be the start of the measurement year. For an incident cohort, the index date and time zero are usually the same. For a readmission cohort, the index date is the discharge date and time zero is the moment the discharge order is signed. If time zero is not stated, every feature window is ambiguous.

**Lookback and left truncation.** The lookback window is the period before time zero used to establish history, comorbidity, and prior utilisation. A one-year lookback cannot see a diagnosis recorded two years ago, so a patient with longstanding diabetes may appear incident. The data are left-truncated at the start of the observation period: events before that date are unobserved, and a patient who enters the data as an adult has an unknown history. The choice of lookback is a bias-variance trade-off. A longer lookback captures more history but reduces the cohort to patients with long-enough records and may include stale diagnoses that have resolved.

**Immortal time and follow-up.** Immortal time is the period during which a patient cannot experience the outcome by definition, usually because cohort entry requires surviving to a later event. A cohort that requires a patient to attend a follow-up visit before classifying them as "engaged in care" has excluded everyone who died or moved before the visit, and the engaged group will look healthier. The follow-up window must start at time zero and must handle death, disenrollment, and out-of-network care as censoring or competing events, not as silent exclusions.

A fourth concept, prevalence versus incidence, determines the denominator. A prevalent cohort includes everyone with the condition at a point in time, regardless of when it started. An incident cohort includes only new cases during a period. The same patient can be in a prevalent cohort for years and in an incident cohort once. Quality measures usually use prevalent cohorts; incidence studies and models of disease progression use incident cohorts. Mixing them produces a denominator that is neither.


The cohort is not the population. The population is everyone for whom the question is relevant; the cohort is the subset that the data can identify. A diabetes registry built from a primary care network excludes patients who receive all their care elsewhere, and a readmission cohort built from one hospital excludes patients readmitted to another. The difference between the population and the cohort is the coverage gap, and it should be reported alongside the cohort count. A cohort of 24,800 in a service area with an estimated 35,000 diabetes patients is not a complete registry; it is a registry of the patients the system can see.

Cohorts are also reused. A validated diabetes cohort becomes the base for a care-gap measure, a cost analysis, and a risk model. Reuse is efficient, but it propagates any defect to every consumer. That is why the validation report and the version identifier travel with the cohort, and why a change to the definition requires a review of every downstream metric.
### Cohort designs at a glance

| Design | Index rule | Denominator | Typical use | Main bias |
|---|---|---|---|---|
| Prevalent | Earliest evidence in the lookback | All patients at risk in the base | Registry, quality measure | Under-count of old or external cases |
| Incident | First evidence after a clean period | Patients at risk without prior evidence | Natural history, model training | Left truncation and immortal time |
| Episode | Start of a care episode | Episodes rather than patients | Cost and utilisation | Multiple episodes per patient |
| Cross-sectional | A point-in-time encounter | Patients with an encounter in the period | Utilisation snapshots | Seasonal and access bias |
| Case-control | Cases and matched controls | Matched sample | Rare outcomes, aetiology | Control selection |
| Registry | Entry by report or clinician | Reported cases | Safety and device surveillance | Under-reporting |

The design choice is not a technical detail. It changes who is counted, what the denominator means, and which bias dominates. A prevalent registry and an incident cohort built from the same data can differ by a factor of two in size and by several years in average age.

> **Note:** The funnel is drawn with crisp boundaries for teaching. In practice, the boundaries are probabilistic: a diagnosis code may be a rule-out, a medication may be a sample, a lab may be ordered but not resulted. The cohort definition should state how ambiguous records are handled, and the validation sample should estimate how often the ambiguity changes the classification.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Cohort | A defined set of patients meeting specified criteria in a specified period | Any query result | Patient count with inclusion rules |
| Base population | The universe of patients from which the cohort is drawn | The entire EHR | Patient count with data coverage |
| Index event | The event that qualifies a patient for cohort entry | Any encounter in the database | Diagnosis, lab, medication, or encounter code |
| Index date | The date of the index event | The first date in the record | Date |
| Time zero | The instant at which features are frozen and prediction or measurement begins | The index date, always | Timestamp with source |
| Eligibility criterion | A condition a patient must meet to enter the cohort | An exclusion in disguise | Rule with code and window |
| Exclusion criterion | A condition that removes an otherwise eligible patient | A convenience filter | Rule with code and window |
| Lookback window | The period before time zero used to establish history | The entire available history | Days or months |
| Follow-up window | The period after time zero used to ascertain outcomes | The period until the data end | Days or months |
| Prevalent cohort | All patients with the condition at a point in time | Called incident | Patient count at a date |
| Incident cohort | Patients newly meeting the condition during a period | Called prevalent | Patient count per period |
| Left truncation | Unobserved history before the start of data availability | Ignored in survival analysis | Date of data start |
| Immortal time | Person-time during which the outcome cannot occur by design | Included in follow-up | Days |
| Censoring | End of observation before the outcome, for death, disenrollment, or data end | Treated as no event | Date and reason |
| Competing risk | An event that prevents the outcome, such as death | Treated as censoring without justification | Event type |
| Phenotype | A computable definition of a clinical condition | A single diagnosis code | Algorithm plus code set |
| Computable phenotype | A phenotype expressed as executable logic | A narrative definition | Query or rule set |
| Code set | The diagnosis, procedure, lab, and drug codes that define a concept | A single code | Codes with vocabulary and version |
| Value set | A set of codes from one or more terminologies used in a measure | Confused with code set | Codes plus terminology |
| ICD-10-CM | Diagnosis coding system used in the United States | Used for procedures | Alphanumeric code |
| ICD-10-PCS | Inpatient procedure coding system | Confused with CPT | Seven-character code |
| CPT | Current Procedural Terminology for professional services | Used for inpatient procedures | Five-character code |
| LOINC | Logical Observation Identifiers Names and Codes for labs and observations | Used for diagnoses | Numeric-plus-check-digit code |
| RxNorm | Normalized naming system for medications | Used for procedures | RxCUI |
| SNOMED CT | Clinical terminology for findings, disorders, and procedures | Used as a billing code | SCTID |
| OMOP CDM | Observational Medical Outcomes Partnership Common Data Model | Treated as a database product | Schema specification |
| OHDSI | Observational Health Data Sciences and Informatics collaborative | A software vendor | Research community |
| eMERGE | Electronic Medical Records and Genomics network | A single algorithm | Consortium and phenotype library |
| PheKB | Phenotype KnowledgeBase; a repository of phenotype algorithms | A validated product | Website and algorithm repository |
| PPV | Positive predictive value; proportion of algorithm-positive patients who truly have the condition | Assumed from the literature | Proportion |
| NPV | Negative predictive value; proportion of algorithm-negative patients who truly do not have the condition | Ignored when sampling only positives | Proportion |
| Sensitivity | Proportion of true cases detected by the algorithm | Estimated from a positive-only sample without correction | Proportion |
| Specificity | Proportion of true non-cases correctly excluded | Estimated from a positive-only sample without correction | Proportion |
| Verification bias | Bias from validating only a non-random subset, usually algorithm positives | Ignored in validation reports | Sampling design |
| Chart review | Manual abstraction of the clinical record as a reference standard | Treated as error-free | Sample with adjudication |
| Gold standard | The best available reference definition | Assumed to be perfect | Clinical criteria or adjudication |
| Misclassification | Assignment of a patient to the wrong cohort class | Treated as random noise | Proportion |
| Prevalence | Proportion of the base population with the condition | Confused with cohort size | Proportion at a date |
| Incidence | New cases per population per time | Confused with prevalence | Cases per 1,000 person-years |
| Care gap | Recommended care not received by an eligible patient | Any missing data element | Patient-level flag |
| Denominator | The eligible population for a measure | The cohort without exclusions | Patient count |
| Numerator | The patients who met the measure criterion | The patients with any documentation | Patient count |
| Data dictionary | The documented definition of every field and code | A list of column names | Document with versions |
| Versioning | Assigning an identifier to a cohort definition and its code set | Saving a query with a date | Semantic version |
| Portability | The ability to run a phenotype at another site with comparable performance | Copying a query without validation | Multi-site validation result |
| Validation sample | The subset of patients selected for chart review | A convenience sample | Sampling frame and size |
| PPV threshold | The minimum PPV acceptable for the intended use | A universal 90 percent rule | Proportion with rationale |
| Data coverage | The fraction of the population for which the data are complete | Assumed to be 100 percent | Percent by source |

## 5. The mechanics, step by step

The procedure below builds a cohort that can be reviewed, re-run, and validated. It assumes access to an EHR extract and, where available, claims or a health information exchange. Each step names its input, its output, and the reason it exists. Steps 4 to 8 are the ones that determine most of the cohort's behaviour; steps 10 and 11 are the ones most often skipped.

### Step 1. State the clinical concept and the intended use

**Input:** a request such as "find our diabetes patients" or "build a readmission cohort".

**Output:** one paragraph defining the concept, the population, the period, and the decision the cohort supports.

**Rationale:** the same clinical concept has different operational definitions for different uses. A diabetes registry for care-gap outreach needs high sensitivity, because missing a patient means missing an opportunity for care. A cohort for a randomised trial of a new drug needs high specificity and a confirmed diagnosis, because including a non-case exposes a patient to risk without benefit. A cohort for a prevalence estimate needs both, and the validation sample must support both. The intended use determines which error is worse, and therefore which definition to choose.

Write the concept in clinical language before writing any code. "Adults with type 2 diabetes, defined as a diagnosis code, an HbA1c result, or a glucose-lowering medication, who had at least one encounter in the measurement year" is a concept. "E11" is a code. The code is an implementation of the concept, not the concept itself.

**Branch point:** if the intended use is regulatory or payment-related, the definition may be prescribed by the measure steward. In that case the concept is fixed, and the work is to implement it faithfully and document deviations. Do not invent a local definition when a binding specification exists.

### Step 2. Define the base population

**Input:** the concept and the intended use.

**Output:** a count and a description of the patients from whom the cohort will be drawn, with data coverage stated.

**Rationale:** the base population determines the denominator of prevalence and the pool for incidence. It is not "all patients in the EHR". It is patients with at least one encounter in a defined period, in a defined set of settings, with a defined minimum data history. A base population that includes a patient with a single billing-only encounter and no clinical data will contribute missing values that look like clinical absence.

State the data coverage explicitly: which hospitals, clinics, and services contribute; which payers are included; whether outside records are imported; and whether the health information exchange is complete. A base population drawn from an academic medical centre's primary care network is not representative of the surrounding community, and a cohort built from it will underrepresent patients who receive most care elsewhere.

**Branch point:** if the cohort is for a population health programme, the base population should be the attributed or enrolled population, not the encountered population. Members who have not had an encounter are still part of the denominator, and excluding them inflates prevalence and care-gap rates.

### Step 3. Choose prevalent versus incident design

**Input:** the concept, the intended use, and the available history.

**Output:** a design statement: prevalent at a date, incident during a period, or both.

**Rationale:** a prevalent cohort answers "who has the condition now"; an incident cohort answers "who developed it during the period". The two require different index rules and different denominators. A prevalent cohort can use the earliest evidence in the lookback and does not need a clean baseline. An incident cohort needs a clean period before the index date during which the patient had no evidence of the condition, and it must handle left truncation: a patient who enters the data with an existing diagnosis cannot be classified as incident.

For quality measures, prevalent cohorts are standard, with a measurement year and a lookback for evidence. For disease progression or model training, incident cohorts are usually required, because prevalent cases have different risk profiles and survival. Mixing the two in one cohort produces a hazard ratio that is not interpretable.

**Branch point:** if the data history is shorter than the required clean period, an incident design is not feasible. Either extend the lookback with external data or change the question to a prevalent design with an explicit limitation.

### Step 4. Define the index event and index date

**Input:** the concept and the design.

**Output:** a rule that identifies the first qualifying event, the date assigned, and the precedence when multiple events qualify on the same day.

**Rationale:** the index event is the anchor for all windows. For a diagnosis-based cohort, it is the first encounter with a qualifying diagnosis. For a lab-based cohort, it is the first qualifying result. For a medication-based cohort, it is the first dispensing or order. For a readmission cohort, it is the discharge. The index date must be the date the event occurred, not the date it was entered or billed.

Precedence matters when a patient has several qualifying events. A patient with an outpatient diagnosis on 3 March and an inpatient diagnosis on 10 March has two candidate index dates. The first is usually correct for a prevalent cohort; the inpatient date may be correct for a severity cohort. State the rule and apply it consistently. If the same patient can enter the cohort more than once, decide whether the unit is the patient or the episode, and say so.

**Branch point:** if the index event is a diagnosis code, a rule-out code can create a false index. Use a second confirmatory criterion (a lab, a medication, or a second code) when the concept requires it, and state the confirmatory window.

### Step 5. Build and version the code set

**Input:** the clinical concept and the available terminologies.

**Output:** a code set with codes, descriptions, vocabulary, version, source, and inclusion rationale.

**Rationale:** the code set is the most consequential and least reviewed part of the cohort. A diagnosis concept may require dozens of ICD-10-CM codes, a lab concept may require several LOINC codes for equivalent assays, and a medication concept may require RxNorm ingredients and their descendants. The code set must be built from a terminology server or a published value set, not from memory, and it must record the version.

Document the provenance: whether the codes came from a measure steward, a published phenotype, a clinician panel, or a data-driven search. Document the exclusions: codes that look relevant but were excluded because they represent a different condition, a history code, or a rule-out. Document the mapping: which ICD-10-CM codes were used before the transition from ICD-9-CM, and whether the mapping is one-to-one or approximate.

**Branch point:** if the cohort is multi-site, the code set must be expressed in a common data model and tested at each site. A code set that works at one site may fail at another because of local coding practices, EHR configuration, or formulary differences. The eMERGE portability work found that algorithms often required local adaptation and that performance varied across sites even when the logic was shared.

### Step 6. Apply eligibility criteria

**Input:** the base population, the index event, and the concept.

**Output:** the eligible population count and the attrition at each criterion.

**Rationale:** eligibility criteria define who can be in the cohort at all: age at index, sex, enrollment or attribution, setting, payer, and data availability. Each criterion should be clinically or operationally justified. An age criterion of 18 is justified for an adult cohort; an age criterion of 65 may be a proxy for Medicare, which is a payer criterion in disguise. A requirement for at least one primary care visit may be justified for a care-gap measure, but it excludes patients who receive care only from specialists.

Report the attrition at each step. A cohort that starts with 250,000 patients and ends with 24,800 should show where the 225,200 went. Attrition is diagnostic: an unexpected drop at the eligibility step usually means a join error, a date filter applied to the wrong field, or a code set that is too narrow.

**Branch point:** if an eligibility criterion removes more than 30 percent of otherwise eligible patients, examine it before proceeding. Large attrition is often a data coverage problem rather than a clinical exclusion.

### Step 7. Apply exclusions

**Input:** the eligible population.

**Output:** the final cohort count and a table of exclusions by reason.

**Rationale:** exclusions remove patients for whom the cohort definition or the outcome is not meaningful. Hospice patients are excluded from readmission cohorts because readmission is not a quality failure for them. Patients with a prior history of the condition are excluded from incident cohorts because they are not incident. Transplant patients may be excluded from a diabetes cohort because their diabetes is steroid-induced and managed differently. Each exclusion changes the population and therefore the interpretation.

Exclusions must be applied at the right time. An exclusion based on a diagnosis recorded after the index date is temporal leakage: the patient is excluded because of something that happened later, which can remove exactly the patients who had the outcome. Apply exclusions using data available at or before time zero, unless the exclusion is deliberately prospective and stated as such.

**Branch point:** if an exclusion is based on a code with low sensitivity, it will fail to remove the patients it is meant to remove. Validate the exclusion code set with the same rigour as the inclusion code set.

### Step 8. Set lookback and follow-up windows

**Input:** the index date and the outcome or feature requirements.

**Output:** explicit window definitions in days, with the anchor date and the direction.

**Rationale:** the lookback window establishes history, comorbidity, and prior utilisation. The follow-up window ascertains outcomes, care gaps, and costs. Both must be anchored to time zero, and both must handle the data boundary. A one-year lookback for a patient whose first encounter was six months ago is not a one-year lookback; it is a six-month lookback with 180 days of missing history. A 30-day follow-up for a patient discharged 10 days before the data extract is not a 30-day follow-up; it is a 10-day follow-up with an unknown outcome.

State whether the windows are fixed (every patient gets 365 days of lookback) or available (every patient gets whatever history exists). Fixed windows produce a cleaner comparison but reduce the cohort to patients with long records. Available windows retain more patients but introduce differential history. Report both the fixed-window cohort and the available-window cohort when the difference is material.

**Branch point:** if the outcome can occur after disenrollment or out-of-network transfer, the follow-up window must include those events as censoring or as outcomes, not as silent losses. A readmission that occurs at another hospital is still a readmission; if the data cannot see it, the rate is understated.

### Step 9. Implement in the target data model

**Input:** the specification and the data model (OMOP CDM, i2b2, PCORnet, a local warehouse, or a flat extract).

**Output:** executable SQL or code, a mapping document, and a reproducible run.

**Rationale:** the data model determines how the concept is expressed. In OMOP, diagnoses are in `condition_occurrence`, drugs in `drug_exposure`, labs in `measurement`, and visits in `visit_occurrence`; the cohort is built with concept sets and era logic. In a local warehouse, the tables may be named differently and the codes may be stored as strings. The mapping document records which source table and field implement each criterion, so that a reviewer can trace the logic.

The implementation must be deterministic. The same input data and the same definition should produce the same count. If the query uses `CURRENT_DATE`, a random sample, or a threshold learned from the data, the count will change on every run and the cohort cannot be reproduced. Freeze the run date and record it.

**Branch point:** if the cohort is implemented in more than one model, the counts will differ. Reconcile the differences at the criterion level, not only at the total. A 5 percent difference in the total can hide a 50 percent difference in a subgroup.

### Step 10. Run cohort diagnostics

**Input:** the executed cohort.

**Output:** a diagnostics report with counts, incidence, overlap, attrition, and subgroup distributions.

**Rationale:** diagnostics catch errors that validation cannot. A cohort with an implausible age distribution, a sudden change in incidence at a date, or a count that doubles after a code-set update has a problem. Plot the cohort count by month, the age and sex distribution, the prevalence by clinic, and the overlap with a related cohort. Compare the observed prevalence to a published benchmark or a claims-based estimate.

The OHDSI community has codified this practice in CohortDiagnostics and the Phenotype Library's validity checks: unused concept sets, domain mismatches, non-standard concepts, implausible incidence, and cohort overlap are all signals of a definition problem. The diagnostics are not a substitute for chart review, but they are cheap and they catch gross errors before a validation sample is drawn.

**Branch point:** if the cohort count changes by more than 10 percent after a code or data update, treat the change as a definition change and re-validate. Do not assume the new count is the correct one.

### Step 11. Validate on a sample

**Input:** the cohort and the chart review capacity.

**Output:** PPV, NPV, sensitivity, specificity, and confidence intervals, with the sampling design stated.

**Rationale:** validation estimates how often the algorithm agrees with a reference standard. The reference standard is usually chart review by a clinician, sometimes supplemented by a structured second source. The sampling design determines which metrics are estimable. A sample drawn only from algorithm-positive patients can estimate PPV but not sensitivity or specificity without correction. A random sample of the base population can estimate all four but is expensive when prevalence is low.

Use a stratified sample: a random sample of algorithm positives to estimate PPV, and a random sample of algorithm negatives to estimate NPV. From PPV, NPV, and the algorithm-defined prevalence, sensitivity and specificity can be derived. If the sampling fraction differs between positives and negatives, apply weights. Bhasin et al. show that ignoring the sampling design can inflate sensitivity from 0.620 to 0.942 and deflate specificity from 0.999 to 0.979. The correction is not optional.

**Branch point:** if the PPV is below the threshold required by the intended use, do not proceed to modelling. Tighten the definition, add a confirmatory criterion, or restrict the cohort to a setting where the data are more complete. A model trained on a cohort with a PPV of 0.70 learns the misclassification as signal.

### Step 12. Freeze, document, and monitor drift

**Input:** the validated cohort.

**Output:** a versioned definition, a data dictionary, a validation report, and a monitoring plan.

**Rationale:** a cohort is a living definition in a changing data environment. Codes are added, EHRs are replaced, formularies change, and coding practices shift. Without versioning, a change in the cohort count cannot be attributed to a change in the population, the data, or the definition. Assign a semantic version, record the code-set version and the run date, and archive the exact query.

Monitor drift with three signals: the cohort count per month, the prevalence by site, and the PPV on a small annual sample. A sudden change in any of them triggers a review. The review asks whether the population changed, the data changed, or the definition changed. The answer determines whether to update the definition or to investigate the data pipeline.

**Branch point:** if the cohort is used for payment or quality reporting, a definition change may require re-reporting prior periods. Coordinate with finance and compliance before changing a measure that has already been submitted.

### Decision table: intended use and validation requirement

| Intended use | Primary error to avoid | Minimum validation | Design | Typical PPV target |
|---|---|---|---|---|
| Care-gap outreach | Missing a true case | PPV and sensitivity | Prevalent, high-sensitivity code set | 0.80 or higher |
| Quality measure reporting | Including a non-case | PPV and specificity | Prevalent, measure specification | 0.90 or higher |
| Model training | Label noise | PPV and NPV | Incident or prevalent, temporal split | 0.90 or higher |
| Clinical trial recruitment | Exposing a non-case | PPV and confirmation | Incident, confirmatory criterion | 0.95 or higher |
| Prevalence estimation | Both false positives and false negatives | All four metrics | Prevalent, random sample | 0.85 or higher |
| Signal detection | Missing a safety signal | Sensitivity and NPV | Incident, broad code set | Context-dependent |

The table is a starting point. The actual threshold should be justified by the cost of the two errors. In outreach, a false positive costs a letter and a phone call; a false negative costs a missed screening. In a trial, a false positive costs patient safety. The thresholds follow from those costs, not from a universal rule.

### Edge cases that invalidate a cohort

**Missing data treated as absence.** A patient with no HbA1c result may have been tested elsewhere. A patient with no diagnosis code may have the condition documented only in a note. The cohort should distinguish "no evidence" from "evidence of absence". Where possible, link to claims or a health information exchange to fill the gap, and report the proportion of the cohort with missing data by source.

**Left truncation.** A patient who first appears in the data at age 55 has an unknown history. An incident cohort that treats the first observed diagnosis as incident will include prevalent cases. Require a clean period before the index date, or flag patients with insufficient history and report the cohort with and without them.

**Code-set changes over time.** ICD-10-CM adds and retires codes every year. A code set built in 2021 may not capture a new code introduced in 2024. Version the code set and review the annual changes. For long trends, map the codes to a stable concept or report the trend with a break at the code-set change.

**EHR migration.** A switch from one vendor to another changes documentation templates, order sets, and coding prompts. The cohort count may change without any change in the population. Mark the migration date on every trend and validate the cohort before and after.

**Patients new to the system.** A new patient with a chronic condition may have no historical codes. The first encounter may be an acute event, and the chronic condition may be documented later. A lookback that requires prior evidence will exclude new patients, who are often the most mobile and the most disadvantaged.

**Deceased patients.** A deceased patient cannot have a follow-up visit or a readmission. If death is not recorded, the patient is treated as lost to follow-up, and the outcome rate is understated. Link to mortality data where available and treat death as a competing event.

**Out-of-network care.** A patient may receive a lab, a specialist visit, or an admission outside the network. The EHR will not see it, and the cohort will classify the patient as not tested or not readmitted. Link to claims or a health information exchange, and report the leakage.

### Cohort specification template

The twelve steps produce a specification that fits on two pages. The template below is the minimum field set. A definition missing any of these fields cannot be reproduced by a reviewer.

```text
Cohort name:
Version:
Owner:
Intended use:
Clinical concept (one paragraph):
Base population (settings, dates, coverage):
Design (prevalent / incident):
Index event (codes, setting, precedence):
Index date rule:
Eligibility criteria (each with code and window):
Exclusions (each with code, window, and rationale):
Lookback window (fixed or available):
Follow-up window (start, end, censoring rules):
Code sets (vocabulary, version, source):
Data model and tables:
Validation sample (frame, size, sampling weights):
Performance (PPV, NPV, sensitivity, specificity, confidence intervals):
Known limitations:
Run date and data extract version:
```

The intended-use field is the one that determines the others. A cohort with two intended uses needs two versions, not one definition with a compromise threshold.

### Attrition log

The attrition log is the cohort's audit trail. It records the count after every criterion and the reason for every drop. A reviewer should be able to reproduce the final count by subtracting the exclusions in order.

| Step | Criterion | Patients remaining | Removed | Percent of previous | Reason |
|---|---|---:|---:|---:|---|
| 0 | Base population with an encounter in 2025 | 250,000 | not applicable | not applicable | Data coverage statement |
| 1 | Age 18 or older at index | 231,400 | 18,600 | 7.4% | Paediatric patients excluded |
| 2 | At least one qualifying diagnosis, lab, or drug | 27,300 | 204,100 | 88.2% | No evidence of diabetes |
| 3 | At least 12 months of prior history | 24,800 | 2,500 | 9.2% | Insufficient lookback |
| 4 | Not in hospice at index | 24,650 | 150 | 0.6% | Hospice exclusion |
| 5 | Not pregnant at index | 24,410 | 240 | 1.0% | Pregnancy exclusion |
| 6 | Valid residential address | 23,980 | 430 | 1.8% | Outreach feasibility |

The largest drop is step 2, which is expected: most patients do not have diabetes. A drop of 88 percent at a criterion that should remove 5 percent is a red flag. The attrition log makes that visible before the cohort is used.

### Validation sampling designs

| Design | Sample frame | Metrics estimable without correction | Cost | When to use |
|---|---|---|---|---|
| Algorithm-positive only | Random sample of cohort | PPV | Low | Routine monitoring of an established definition |
| Positive and negative | Random samples of both | PPV, NPV; sensitivity and specificity with prevalence | Medium | New definition or new site |
| Random base population | Random sample of the base | All four directly | High | Prevalence estimation and rare outcomes |
| Stratified by site | Random samples within each site | Site-specific PPV and portability | Medium-high | Multi-site deployment |
| Consecutive clinical review | All patients in a period | PPV; sensitivity only with a negative sample | High | Safety-critical recruitment |

The choice is a budget decision as much as a statistical one. A positive-only sample of 100 patients can estimate PPV with a 95 percent confidence interval of roughly plus or minus 6 percentage points when PPV is 0.90. A positive and negative sample of 100 each can estimate sensitivity and specificity with similar precision, but only if the sampling weights are applied.

### Edge cases: detection and mitigation

| Edge case | Detection method | Mitigation | Residual risk |
|---|---|---|---|
| Missing data treated as absence | Compare EHR to claims or external labs for a sample | Link sources; report missingness | Unlinked care remains invisible |
| Left truncation | Length of prior history by patient | Require a clean period; flag short histories | New patients excluded |
| Code-set change | Count by code and year; review annual updates | Version the code set; map to stable concepts | Mapping is approximate |
| EHR migration | Count and case mix before and after the switch | Mark the date; validate both periods | Workflow changes are persistent |
| New patients | Share of cohort with less than 12 months of history | Report cohort with and without new patients | Selection remains |
| Deceased patients | Mortality linkage; follow-up completeness | Treat death as competing event | Unlinked deaths understate risk |
| Out-of-network care | Compare provider and payer claims | Link claims or HIE; report leakage | Leakage is unmeasured |
| Duplicate records | Same patient with multiple medical record numbers | Probabilistic linkage with review | False merges |
| Test result without order | Lab feeds vs order feeds | Use the lab result as the event | Missing context |
| Diagnosis in note only | NLP or chart review sample | Add NLP-derived concepts; validate | NLP error |

### Worked code-set construction

The table below shows the decisions that go into a small code set for type 2 diabetes. The point is not the specific codes but the reasoning that must be recorded for each inclusion and exclusion.

| Concept component | Candidate code or vocabulary | Include? | Reason |
|---|---|---|---|
| Type 2 diabetes with hyperglycaemia | ICD-10-CM E11.65 | Yes | Confirms the condition and an acute metabolic state |
| Type 2 diabetes with kidney complications | ICD-10-CM E11.22 | Yes | Confirms the condition with a complication |
| Type 2 diabetes without complications | ICD-10-CM E11.9 | Yes | Common uncomplicated code |
| Type 2 diabetes with ophthalmic complications | ICD-10-CM E11.3x | Yes | Confirms the condition with a complication |
| Gestational diabetes | ICD-10-CM O24.4x | No | Different condition and population |
| Abnormal glucose tolerance | ICD-10-CM R73.09 | No | Not a diabetes diagnosis |
| Long-term insulin use | ICD-10-CM Z79.4 | No | A status code, not a diagnosis |
| HbA1c | LOINC 4548-4 | Yes, as supportive evidence | Confirms glycaemic status, not a diagnosis alone |
| Plasma glucose | LOINC 2345-7 | Yes, as supportive evidence | Threshold must be specified |
| Metformin | RxNorm ingredient and descendants | Yes, as supportive evidence | Indicates treatment, not confirmation alone |
| Insulin glargine | RxNorm ingredient and descendants | Yes, as supportive evidence | Indicates treatment, not confirmation alone |

The set is deliberately conservative for a diagnosis-based definition. It excludes R73.09 and Z79.4 because they are commonly used as rule-out or status codes and inflate prevalence. It includes labs and drugs only as supportive evidence, and the combination rule must state whether one supportive criterion is enough or whether two are required.

### Common cohort patterns

| Pattern | Index event | Strength | Weakness | Typical use |
|---|---|---|---|---|
| Diagnosis-based | First qualifying diagnosis code | Simple, portable | Sensitive to coding practice and rule-outs | Registries, prevalence |
| Lab-based | First qualifying result | Objective, reproducible | Misses patients without testing | Chronic disease, risk models |
| Medication-based | First dispensing or order | Reflects treatment | Misses untreated patients; includes off-label use | Adherence, pharmacoepidemiology |
| Procedure-based | First qualifying procedure | Specific | Misses patients managed medically | Surgical outcomes |
| Multi-criteria | Diagnosis plus lab or drug | Higher PPV | Lower sensitivity; complex logic | Trial recruitment, quality |
| Measure specification | Steward-defined | Comparable and auditable | Not locally optimised | HEDIS, CMS reporting |

The multi-criteria pattern is usually the right choice for a high-stakes cohort, but it must be validated against a single-criterion version. If the multi-criteria definition raises PPV from 0.78 to 0.92 while reducing sensitivity from 0.90 to 0.71, the choice depends on whether the programme can tolerate missing 29 percent of cases. That trade-off is a clinical and operational decision, not a statistical one.

### What makes a cohort reviewable

A cohort is reviewable when five conditions hold. First, the specification names every code set and its version. Second, the attrition log reconciles the base population to the final cohort. Third, the validation sample states its sampling frame and weights. Fourth, the run date and data extract version are recorded. Fifth, the known limitations are written in the same document as the definition, not in a separate caveat email.

A reviewer who cannot reproduce the count from the specification has not been given a definition. They have been given a result. The distinction matters when the cohort is used for payment, publication, or a clinical programme, because the reviewer's inability to reproduce the count is the first sign that the organisation cannot defend it.

### Cohort versus measure denominator

A cohort and a measure denominator are not the same object. The cohort is the population with the condition; the measure denominator is the subset of the cohort that is eligible for the measure after measure-specific exclusions, enrollment requirements, and age limits. A diabetes cohort of 24,800 may have a HEDIS denominator of 18,200 after excluding patients without continuous enrollment, patients in hospice, and patients outside the age range. Reporting the cohort size as the measure denominator overstates the population and understates performance. Always report both and label them separately.

The reviewability standard also applies to the code itself. A query with a hard-coded date, a temporary table created outside the version-controlled repository, or a manual Excel adjustment is not reviewable even if the specification is complete. The code, the specification, and the validation report travel together as one artefact. When they diverge, the code wins in production and the specification loses, which is why the two must be reconciled before release.

| Output | Consumer | Refresh | Retention |
|---|---|---|---|
| Cohort definition and code set | Analytics, informatics | On version change | Permanent |
| Attrition log | Reviewers, finance | Each run | Seven years |
| Validation report | Clinical governance | Annual | Permanent |
| Patient-level cohort table | Operations, modelling | Monthly or quarterly | Per data policy |

The refresh cadence is a data governance decision, not an analytics preference. A monthly refresh of a definition that changes annually creates twelve versions of the truth.

## 6. Worked example

An academic health system wants a type 2 diabetes registry for a care-gap outreach programme. The programme will contact patients who have no HbA1c result in the measurement year and offer a point-of-care test or a lab order. The analytics team must build the cohort, estimate how many patients truly have diabetes, correct the care-gap rate for misclassification, and decide whether the outreach capacity is sufficient.

### Part A: cohort construction and attrition

The base population is 250,000 adults with at least one encounter in the health system in 2025. The cohort uses a multi-criteria definition: any qualifying type 2 diabetes diagnosis, or an HbA1c result with a qualifying value, or a glucose-lowering medication, with at least one supportive criterion. The attrition is:

| Step | Criterion | Patients remaining | Removed | Reason |
|---|---|---:|---:|---|
| 0 | Adults with an encounter in 2025 | 250,000 | not applicable | Base population |
| 1 | At least one qualifying diagnosis, lab, or drug | 27,300 | 222,700 | No evidence of diabetes |
| 2 | At least one supportive criterion | 25,600 | 1,700 | Single weak signal only |
| 3 | At least 12 months of prior history | 24,800 | 800 | Insufficient lookback |
| 4 | Not in hospice at index | 24,650 | 150 | Hospice exclusion |
| 5 | Not pregnant at index | 24,410 | 240 | Pregnancy exclusion |
| 6 | Valid address for outreach | 23,980 | 430 | No contactable address |

The final cohort is 23,980 patients. The observed prevalence is 23,980 / 250,000 = 9.59 percent. The prevalence among the 24,410 clinically eligible patients before the address exclusion is 9.76 percent. The address exclusion removes 1.8 percent of the eligible population and is not random: patients without a stable address are more likely to be uninsured and more likely to have gaps in care. The programme should report the cohort with and without the address requirement.

### Part B: validation and corrected prevalence

A validation sample is drawn from the 24,410 clinically eligible patients: 400 randomly selected algorithm positives and 400 randomly selected algorithm negatives. Two clinicians review the charts, with disagreements adjudicated by a third.

| Validation sample | Reviewed | True cases | True non-cases | PPV or NPV |
|---|---:|---:|---:|---:|
| Algorithm positive | 400 | 352 | 48 | PPV = 352 / 400 = 0.88 |
| Algorithm negative | 400 | 24 | 376 | NPV = 376 / 400 = 0.94 |

The sampling fractions differ from the population fractions, so the counts are expanded with weights. Among the 24,410 algorithm positives, the estimated true positives are 24,410 x 0.88 = 21,481. Among the 225,590 algorithm negatives, the estimated false negatives are 225,590 x 0.06 = 13,535. The corrected estimates are:

```text
True cases = 21,481 + 13,535 = 35,016
Corrected prevalence = 35,016 / 250,000 = 14.01%
Sensitivity = 21,481 / 35,016 = 61.35%
False positives = 24,410 - 21,481 = 2,929
Specificity = 1 - 2,929 / (250,000 - 35,016) = 98.63%
```

The algorithm finds 61.4 percent of the true cases and 87.7 percent of the algorithm positives are true cases. The observed prevalence of 9.76 percent understates the true prevalence of 14.01 percent by 4.25 percentage points, a 30 percent relative undercount. The undercount is not a reason to abandon the registry; it is a reason to report the corrected prevalence and to interpret the registry as a high-specificity case-finding tool, not a complete census.

### Part C: care-gap correction

The care gap is defined as no HbA1c result in the measurement year. Among the 24,410 algorithm positives, 16,140 have a result, for an observed testing rate of 16,140 / 24,410 = 66.12 percent and an observed gap of 33.88 percent.

The missed true cases are not a random subset. Patients whose diagnosis is recorded only in a specialist note or an outside record are less likely to have an in-system HbA1c. Assume, for illustration, that true cases detected by the algorithm are tested at the observed rate of 66.12 percent, while missed true cases are tested at 40 percent. The corrected numerator and denominator are:

```text
Tested among detected cases = 21,481 x 0.6612 = 14,203
Tested among missed cases = 13,535 x 0.40 = 5,414
Corrected tested = 14,203 + 5,414 = 19,617
Corrected testing rate = 19,617 / 35,016 = 56.03%
Corrected gap = 43.97%
```

The observed gap of 33.88 percent understates the true gap by 10.09 percentage points. If the missed cases were tested at the same rate as detected cases, the rate would remain 66.12 percent and the gap would be unchanged; the correction matters only because the missed patients differ. The analyst must state the assumption and test it with a sensitivity analysis.

### Part D: outreach capacity and cost per additional test

The outreach team can contact 8,400 patients per year, which is exactly the number of algorithm-positive patients without a result (24,410 - 16,140 = 8,270, plus 130 recontacts). Assume 25 percent of contacted patients complete a test.

```text
Additional tests = 8,270 x 0.25 = 2,068
Outreach cost = 8,270 x $18 = $148,860
Laboratory cost = 2,068 x $12 = $24,816
Total cost = $173,676
Cost per additional test = $173,676 / 2,068 = $84.00
```

If capacity is reduced to 5,000 contacts, additional tests fall to 1,250 and the total cost falls to $105,000, but the cost per additional test remains $84.00 because both costs are proportional to contacts and tests. Capacity affects reach, not unit cost, under these assumptions. The programme's value depends on whether an additional test changes management, which is a clinical question the cost calculation cannot answer.

### Diagnostic 1: the lookback window is too short

The cohort uses a 12-month lookback. A 24-month lookback adds 3,100 patients whose only evidence is between 13 and 24 months before the index date. The algorithm-positive count rises from 24,410 to 27,510, and the corrected true cases rise from 35,016 to 27,510 x 0.88 + 13,535 = 24,209 + 13,535 = 37,744. Corrected prevalence rises from 14.01 percent to 15.10 percent. The 12-month lookback was not wrong; it was a choice that excluded patients with older evidence. The diagnostic is to compare the cohort count and the age distribution under 12, 24, and 36-month lookbacks. If the count is still rising at 36 months, the lookback is truncating real prevalence rather than stabilizing.

### Diagnostic 2: outside labs are missing

The EHR shows 16,140 tested patients. A claims linkage finds that 1,848 of the 8,270 untested patients had an HbA1c at an outside laboratory in the measurement year. The testing rate becomes (16,140 + 1,848) / 24,410 = 73.69 percent, and the gap falls from 33.88 to 26.31 percent. The diagnostic is to compare the EHR testing rate to a claims-based rate for the same patients. A gap of more than 5 percentage points usually means that outside care is material and that the EHR-only measure is a measure of in-system documentation, not of care received.

### Varying one assumption

The validation sample is small, and the PPV estimate has uncertainty. If the true PPV is 0.80 rather than 0.88, the corrected true cases are 24,410 x 0.80 + 13,535 = 19,528 + 13,535 = 33,063. Corrected prevalence falls to 13.23 percent, sensitivity to 59.06 percent, and the corrected testing rate (with the same 40 percent assumption for missed cases) to 55.43 percent. The programme's outreach list does not change, because it is built from algorithm positives, but the reported prevalence and gap change by about 0.8 and 0.6 percentage points respectively. A validation sample of 400 is adequate for a PPV estimate with a confidence interval of roughly plus or minus 3 percentage points; it is not adequate for subgroup estimates, which is why the programme should not report PPV by clinic without a larger sample.

### Part E: subgroup performance

The overall PPV of 0.88 can hide differences between settings. A subgroup review of the same 400 positive charts, stratified by the setting of the index evidence, gives:

| Index evidence setting | Charts reviewed | True cases | PPV |
|---|---:|---:|---:|
| Primary care diagnosis | 200 | 184 | 0.92 |
| Specialist diagnosis | 120 | 103 | 0.86 |
| New patient with lab or drug only | 80 | 59 | 0.74 |
| All positive charts | 400 | 352 | 0.88 |

The new-patient subgroup has a PPV of 0.74, below the 0.80 threshold the programme set for outreach. The likely mechanism is that a single abnormal lab or a medication sample in a new patient is weaker evidence than a longitudinal diagnosis. The programme can respond in three ways: require a second criterion for new patients, exclude new patients from outreach until confirmation, or accept the lower PPV and the extra outreach cost. The choice is operational, and it should be made before the list is generated, not after the first mailer.

The subgroup table also shows why an overall PPV is not a portability guarantee. A site with a higher share of new patients will have a lower PPV even if the algorithm is identical. Reporting only the overall PPV makes the algorithm look more portable than it is.

### Part F: the decision

The outreach capacity of 8,400 contacts exceeds the 8,270 untested algorithm-positive patients, so the programme can contact every untested patient in the cohort once. At a 25 percent completion rate, it generates about 2,068 additional tests at $84 per test. The programme breaks even if the value of an additional completed test, in quality incentive, avoided complication, or contracted performance, exceeds $84. The analytics team cannot set that value; finance and the quality team can.

The cohort decision is therefore conditional on three numbers: the PPV in the population actually contacted, the completion rate, and the value per completed test. The PPV determines how many contacted patients truly have diabetes; the completion rate determines how many tests are generated; the value determines whether the programme is worth running. A dashboard that reports only the number of outreach letters sent has measured activity, not value.

### What the cohort cannot tell you

The cohort cannot tell you whether a patient without an in-system HbA1c received one outside the system, whether a patient with a code has active disease or a resolved history, or whether the patient would benefit from testing. It can tell you who is likely to have diabetes according to the coded record, how confident you can be in that classification, and how the classification error changes the care-gap rate. Those are the questions the registry is designed to answer. Treating the cohort as a clinical truth rather than a coded approximation is the error that the validation sample exists to prevent.

### Sensitivity table

| Assumption changed | Corrected prevalence | Corrected testing rate | Corrected gap | Effect on decision |
|---|---:|---:|---:|---|
| Base case (PPV 0.88, missed tested at 40%) | 14.01% | 56.03% | 43.97% | Fund if value per test exceeds $84 |
| PPV 0.80 | 13.23% | 55.43% | 44.57% | Same decision; prevalence lower |
| Missed tested at 66.12% (same as detected) | 14.01% | 66.12% | 33.88% | Gap smaller; outreach less urgent |
| 24-month lookback | 15.10% | not recomputed | not recomputed | Larger denominator; more outreach |
| Outside labs included | 14.01% | 73.69% | 26.31% | Gap substantially smaller |

The table is the form in which the analysis should be presented. Each row changes one assumption and shows the effect on the two numbers that drive the decision. The base case is not privileged; it is one row among several, and the decision should be robust across the plausible rows or explicitly conditional on which row holds.

### Part G: how precise is the validation estimate

The PPV estimate of 0.88 from 400 charts has a 95 percent confidence interval of approximately 0.88 plus or minus 0.032, or 0.848 to 0.912. The new-patient subgroup estimate of 0.74 from 80 charts has a confidence interval of approximately 0.74 plus or minus 0.096, or 0.644 to 0.836. The subgroup estimate is too imprecise to support a separate threshold without more charts. This is the arithmetic behind the earlier warning: a sample that is adequate for the cohort is not automatically adequate for every subgroup.

The sample size for a desired precision can be approximated as n = z squared times p times (1 - p) divided by the squared margin of error. For a margin of 0.05 and an expected PPV of 0.88, n is approximately 162; for a margin of 0.03, n is approximately 451. The programme's 400 charts give roughly the 0.03 margin for the overall estimate and no useful margin for small subgroups.

### Part H: data dictionary excerpt

The cohort table is only usable if every field is defined. The excerpt below is the minimum for the diabetes registry.

| Field | Source | Definition | Unit or coding |
|---|---|---|---|
| member_id | EHR master patient index | Stable patient identifier after linkage | String |
| index_date | Cohort logic | Earliest qualifying evidence date | Date |
| age_at_index | EHR demographics | Age in years on the index date | Years |
| sex | EHR demographics | Recorded sex | Coded value |
| diabetes_evidence | Cohort logic | Diagnosis, lab, drug, or combination | Categorical |
| hba1c_last | Lab feed | Most recent HbA1c before the measurement year end | Percent |
| hba1c_tested | Lab feed | Any HbA1c in the measurement year | Boolean |
| attribution_clinic | Attribution file | Primary care clinic for the member | Clinic identifier |
| payer | Eligibility file | Payer on the index date | Payer code |
| hospice_flag | EHR or claims | Hospice enrollment at index | Boolean |
| address_valid | Registration | Mailable address at the extract date | Boolean |
| cohort_version | Metadata | Semantic version of the definition | String |

A field without a source and a definition is not a field; it is a guess. The data dictionary is reviewed with the cohort specification, and any field added after validation triggers a re-run of the validation sample if it changes cohort membership.

### Comparison with published phenotypes

The observed sensitivity of 61.4 percent is at the low end of the 62 to 94 percent range reported by Spratt et al. for eight diabetes phenotype definitions, and the PPV of 0.88 sits within the 67.7 to 100 percent range reported by Newton et al. across 13 eMERGE algorithms. The comparison is reassuring but not conclusive: the published ranges come from different populations, different code sets, and different reference standards. The correct use of the literature is to set an expectation and to flag an estimate that falls outside it, not to substitute a published number for a local validation.

A local estimate below the published range usually points to one of three causes: a broader code set, a population with more early or uncertain diagnoses, or a reference standard that counts borderline cases differently. The first two are clinical and data decisions; the third is a measurement decision. All three should be documented before the cohort is released.

The example assumes that the validation sample is drawn after the address exclusion and that the two clinician reviewers are blinded to the algorithm result. If the reviewers see the algorithm flag, they tend to agree with it, which inflates the PPV. The review form should present the chart without the cohort flag, and the agreement between reviewers should be reported. A kappa below 0.6 indicates that the reference standard itself is uncertain, and the PPV estimate should be treated as provisional.

Three checks should be completed before the registry is released:

1. The 400 positive and 400 negative charts are sampled with a documented random seed.
2. The reviewers are blinded to the algorithm result.
3. The cohort version and run date are recorded in the registry metadata.

## 7. Code, query, or calculation

The SQLite script below implements the cohort attrition, the validation correction, the care-gap correction, and the outreach cost for the worked example. It uses aggregate validation counts and a synthetic patient table, so it runs in memory and produces the numbers in Section 6.

Assumptions:

- The cohort is prevalent in 2025 and uses a 12-month lookback.
- `eligible = 1` marks the 24,410 patients who remain after evidence, supportive-criterion, lookback, hospice, and pregnancy exclusions.
- The validation sample is 400 positives and 400 negatives, with the counts from Section 6.
- Missed true cases are tested at 40 percent, while detected cases are tested at the observed rate.
- Outreach contacts every untested patient once; 25 percent complete a test.

```python
import sqlite3

con = sqlite3.connect(":memory:")
cur = con.cursor()
cur.executescript("""
CREATE TABLE attrition (step INTEGER, criterion TEXT, remaining INTEGER);
INSERT INTO attrition VALUES
 (0,'Adults with an encounter in 2025',250000),
 (1,'Any qualifying diagnosis, lab, or drug',27300),
 (2,'At least one supportive criterion',25600),
 (3,'At least 12 months of prior history',24800),
 (4,'Not in hospice at index',24650),
 (5,'Not pregnant at index',24410),
 (6,'Valid address for outreach',23980);

CREATE TABLE cohort (
 patient_id INTEGER PRIMARY KEY, eligible INTEGER, address_valid INTEGER
);
WITH RECURSIVE p(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM p WHERE n < 250000)
INSERT INTO cohort
SELECT n,
    CASE WHEN n <= 24410 THEN 1 ELSE 0 END,
    CASE WHEN n <= 23980 THEN 1 ELSE 0 END
FROM p;

CREATE TABLE labs (patient_id INTEGER, loinc TEXT, lab_date TEXT, value REAL);
WITH RECURSIVE p(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM p WHERE n < 24410)
INSERT INTO labs
SELECT n, '4548-4', '2025-06-15', 7.2
FROM p WHERE n <= 16140;

CREATE TABLE validation (
 sample TEXT, reviewed INTEGER, true_cases INTEGER, true_noncases INTEGER
);
INSERT INTO validation VALUES ('positive',400,352,48), ('negative',400,24,376);

CREATE TABLE assumptions (
 missed_test_rate REAL, contact_rate REAL, test_cost REAL, lab_cost REAL
);
INSERT INTO assumptions VALUES (0.40,0.25,18.0,12.0);
""")

print("ATTRITION")
for row in cur.execute("SELECT step, criterion, remaining FROM attrition ORDER BY step"):
  print(row)
print("final cohort:",
   cur.execute("SELECT remaining FROM attrition WHERE step = 6").fetchone()[0])

print("\nVALIDATION AND CORRECTED PREVALENCE")
print(cur.execute("""
WITH v AS (
 SELECT MAX(CASE WHEN sample = 'positive' THEN 1.0*true_cases/reviewed END) AS ppv,
     MAX(CASE WHEN sample = 'negative' THEN 1.0*true_noncases/reviewed END) AS npv
 FROM validation
), c AS (SELECT SUM(eligible) AS pos, COUNT(*) AS n FROM cohort)
SELECT ROUND(ppv, 2), ROUND(npv, 2),
    ROUND(pos*ppv, 0), ROUND((n-pos)*(1-npv), 0),
    ROUND(pos*ppv + (n-pos)*(1-npv), 0),
    ROUND(100.0*(pos*ppv + (n-pos)*(1-npv))/n, 2),
    ROUND(100.0*pos*ppv/(pos*ppv + (n-pos)*(1-npv)), 2)
FROM v, c;
""").fetchone())

print("\nCARE GAP, OBSERVED AND CORRECTED")
print(cur.execute("""
WITH v AS (
 SELECT MAX(CASE WHEN sample = 'positive' THEN 1.0*true_cases/reviewed END) AS ppv,
     MAX(CASE WHEN sample = 'negative' THEN 1.0*true_noncases/reviewed END) AS npv
 FROM validation
), c AS (SELECT SUM(eligible) AS pos, COUNT(*) AS n FROM cohort),
l AS (SELECT COUNT(*) AS tested FROM labs),
a AS (SELECT missed_test_rate FROM assumptions)
SELECT ROUND(100.0*tested/pos, 2),
    ROUND(100.0*(pos*ppv*(1.0*tested/pos)
       + (n-pos)*(1-npv)*missed_test_rate)/
       (pos*ppv + (n-pos)*(1-npv)), 2)
FROM v, c, l, a;
""").fetchone())

print("\nOUTREACH COST")
print(cur.execute("""
WITH c AS (SELECT SUM(eligible) AS pos FROM cohort),
l AS (SELECT COUNT(*) AS tested FROM labs),
a AS (SELECT contact_rate, test_cost, lab_cost FROM assumptions)
SELECT pos - tested,
    ROUND((pos - tested)*test_cost, 0),
    ROUND((pos - tested)*contact_rate, 0),
    ROUND(((pos - tested)*test_cost
       + (pos - tested)*contact_rate*lab_cost)/
       ((pos - tested)*contact_rate), 2)
FROM c, l, a;
""").fetchone())
con.close()
```

Expected output, produced by Python 3.11 with SQLite 3.50.4:

```text
ATTRITION
(0, 'Adults with an encounter in 2025', 250000)
(1, 'Any qualifying diagnosis, lab, or drug', 27300)
(2, 'At least one supportive criterion', 25600)
(3, 'At least 12 months of prior history', 24800)
(4, 'Not in hospice at index', 24650)
(5, 'Not pregnant at index', 24410)
(6, 'Valid address for outreach', 23980)
final cohort: 23980

VALIDATION AND CORRECTED PREVALENCE
(0.88, 0.94, 21481.0, 13535.0, 35016.0, 14.01, 61.35)

CARE GAP, OBSERVED AND CORRECTED
(66.12, 56.02)

OUTREACH COST
(8270, 148860.0, 2068.0, 84.0)
```

Read the output in four passes. The attrition log reconciles the base population to 23,980 and shows the two clinical exclusions separately. The validation row shows the PPV and NPV, the expanded true positives and false negatives, the corrected prevalence of 14.01 percent, and the sensitivity of 61.35 percent. The care-gap row shows the observed rate of 66.12 percent and the corrected rate of 56.02 percent; the 0.01 difference from the hand calculation is rounding in the true-positive estimate. The outreach row shows 8,270 untested patients, $148,860 of outreach cost, 2,068 additional tests, and $84.00 per additional test.

> **Note:** The `cohort` table is synthetic and the counts are constructed to match the worked example. In a production system, the cohort would be built from `condition_occurrence`, `drug_exposure`, and `measurement` tables in the OMOP CDM or their equivalents, and the validation counts would come from a chart-review database with reviewer identifiers and adjudication status.

### OMOP CDM implementation notes

In the OMOP CDM, the diagnosis criterion is a concept set over `condition_occurrence.condition_concept_id`, the lab criterion is a concept set over `measurement.measurement_concept_id`, and the drug criterion is a concept set over `drug_exposure.drug_concept_id`. Cohort entry is the earliest qualifying event, and the lookback is implemented as a requirement for a prior `observation_period` start date. The cohort is stored in a `cohort` table with `subject_id`, `cohort_start_date`, and `cohort_end_date`, which makes it portable across sites that use the same model.

The portability claim is limited by the concept sets and the local coding. The eMERGE portability study found that an executable phenotype achieved PPV of at least 90 percent at all four sites that completed validation, but that three of five sites had more than 90 percent case overlap with their original implementation, not 100 percent. The difference came from local interpretation and from the inability of the standard format to handle narrative logic at the time. A multi-site cohort should report site-specific PPV and overlap, not only a pooled result.

## 8. Data, standards, and regulatory context

Cohort construction sits at the intersection of three kinds of standards: data model standards that determine where the data live, terminology standards that determine how concepts are coded, and regulatory standards that determine what may be done with the data. None of them defines a cohort. That is the local work.

### Data model standards

| Standard | What it governs | Binding status | What it does not constrain |
|---|---|---|---|
| OMOP CDM | A common schema for observational health data | Voluntary, widely adopted in research networks | Index rules, lookback, exclusions |
| PCORnet CDM | A common schema for patient-centred research networks | Contractual for network participants | Local phenotype logic |
| i2b2 | A query and analytics schema used by academic centres | Voluntary | Terminology normalization |
| FHIR R4 | Exchange of clinical resources through APIs | Binding for CMS-0057-F payers from 2027 | Cohort logic and historical depth |
| US Core | FHIR profiles for US interoperability | Required for certified health IT | Local code sets |
| USCDI | The minimum data elements for exchange | Required for certified health IT | Data completeness |
| C-CDA | Document-level clinical exchange | Required in some programmes | Structured cohort queries |
| Sentinel CDM | A distributed data model for safety surveillance | Contractual for Sentinel | General cohort reuse |

The OMOP CDM is the most important model for cohort work outside a single vendor. It places diagnoses in `condition_occurrence`, drugs in `drug_exposure`, labs in `measurement`, procedures in `procedure_occurrence`, visits in `visit_occurrence`, and patients in `person`. Every source code is mapped to a standard concept identifier, and the concept sets are expressed against those identifiers. The model makes a cohort definition portable in principle: the same concept set and logic can run at any site that has mapped its data.

Portability in principle is not portability in practice. Mapping is local, source codes are incomplete, and the same clinical concept may map to different standard concepts at different sites. The eMERGE network's experience, described in Section 9, is the empirical record of this gap.

### Terminology standards

| Terminology | Domain | Example use in a cohort | Version issue |
|---|---|---|---|
| ICD-10-CM | Diagnoses | Type 2 diabetes code set | Annual code additions and deletions |
| ICD-10-PCS | Inpatient procedures | Procedure-based cohort entry | Annual updates |
| CPT | Professional procedures | Outpatient procedure cohort | Quarterly updates; proprietary |
| HCPCS Level II | Drugs, supplies, services | Infusion cohort | Quarterly updates |
| LOINC | Labs and observations | HbA1c result | New codes for new assays |
| RxNorm | Medications | Glucose-lowering drug exposure | Monthly updates |
| SNOMED CT | Clinical findings | Problem-list concepts | Biannual international release |
| UCUM | Units | Lab values | Stable |

A code set is a set of codes from one or more of these terminologies, with a version. The version is not optional. An ICD-10-CM code set built for FY2024 may not capture a code added in FY2025; an RxNorm ingredient set built before a new drug launch will miss the drug. Record the version and the retrieval date, and review the annual changes for any cohort used in reporting.

### Regulatory context

**HIPAA Privacy and Security Rules (45 CFR Parts 160 and 164).** Cohort work uses protected health information and must comply with the minimum necessary standard, the accounting of disclosures, and the security safeguards. Research use requires an IRB approval, a waiver of authorization, or a de-identification determination. A limited data set may be used under a data use agreement, but it still contains dates and geography that can re-identify.

**The Common Rule (45 CFR Part 46).** Human-subjects research requires IRB review and informed consent unless an exemption applies. Quality improvement that is not generalisable may not be research, but the boundary is determined by the IRB, not by the analytics team. A cohort used to evaluate an intervention and publish the result is usually research.

**21st Century Cures Act and FDA real-world evidence.** The Cures Act directed the FDA to evaluate real-world evidence for regulatory decisions, and the FDA has issued guidance on real-world data and real-world evidence. A cohort intended to support a regulatory submission must meet the evidentiary standards in that guidance, including data relevance and reliability, and must be pre-specified. An operational cohort is not automatically fit for a regulatory purpose.

**GDPR (EU) 2016/679.** For European sites, cohort work involving identifiable data requires a lawful basis, data minimization, and, for special-category health data, an explicit legal basis. Pseudonymisation reduces risk but does not remove the data from scope.

> **Note:** This section is educational and is not legal advice. The regulatory status of a cohort depends on its intended use, the data sources, and the jurisdiction. Consult the privacy office and the IRB before building a cohort that will be used outside operational improvement.

### What the standards do not settle

Four decisions are always local:

1. **The clinical concept.** ICD-10-CM does not define type 2 diabetes; it provides codes that clinicians use. The cohort definition chooses which codes, which combinations, and which confirmatory criteria represent the concept.
2. **The index rule.** The standards do not say whether the first diagnosis, the first lab, or the first medication is the index event. That is a design choice with consequences for prevalence and incidence.
3. **The lookback and follow-up windows.** The standards do not specify how much history is enough. The choice determines who is eligible and how much missing history is tolerated.
4. **The validation threshold.** There is no universal PPV requirement. The threshold follows from the cost of false positives and false negatives in the intended use.

A cohort specification that does not state these four decisions is incomplete, regardless of how well it conforms to OMOP or FHIR.

## 9. Industry practice and case evidence

Cohort building is one of the few areas of health analytics with a substantial peer-reviewed validation literature. The evidence below is documented; the inferences are labelled.

### Documented validation results

**eMERGE validation (Newton et al., JAMIA 2013).** The eMERGE network validated 13 electronic phenotyping algorithms across five sites. Positive predictive values ranged from 67.7 to 100 percent. About 75 percent of 51 algorithm reviews achieved a PPV of at least 90 percent, and three reviews fell below 80 percent. The paper documented that performance depended on the algorithm, the site, and the validation method, and it recommended site-specific validation rather than assuming portability. DOI 10.1136/amiajnl-2012-000896.

**Executable phenotype portability (Pacheco et al., JAMIA 2018).** The PhEMA project converted an eMERGE benign prostatic hyperplasia algorithm into a standards-based executable workflow and deployed it at eight sites; six sites executed it successfully. Blinded chart review found a case PPV of at least 90 percent at all four sites that completed validation, and three of five sites had more than 90 percent case overlap with their original implementation. The authors attributed the remaining differences to local interpretation and to the inability of the standard format to represent narrative logic at the time. DOI 10.1093/jamia/ocy101.

**NLP-enhanced phenotype portability (Pacheco et al., Scientific Reports 2023).** Nine eMERGE sites tested NLP-enhanced versions of six rule-based algorithms. Adding NLP improved or maintained precision and recall for five of the six. For chronic rhinosinusitis, precision rose from 76 to 78-83 percent and recall from 97 to 100 percent; for asthma-COPD overlap, recall rose from 38 to 54 percent. For systemic lupus erythematosus, precision fell from 99 to 96 percent while recall rose from 79 to 91 percent. Negation remained a major failure mode, and development plus validation took at least six months. DOI 10.1038/s41598-023-27481-y.

**Diabetes phenotype definitions (Spratt et al., JAMIA 2017).** Eight EHR-based diabetes definitions were applied to 173,503 Duke patients; 22,679 met at least one definition. Against an expert chart-review standard, sensitivity ranged from 62 to 94 percent and specificity from 95 to 99 percent, depending on the definition and the timing rules. The choice of definition changed the cohort by thousands of patients.

**Verification bias in validation (Bhasin et al., ACI Open 2024).** When validation samples only algorithm-positive patients, naive estimates of sensitivity and specificity are biased. In the reported example, naive sensitivity of 0.942 fell to a bias-adjusted 0.620, and specificity rose from 0.979 to 0.999. PPV and NPV remain valid under case-control sampling but are prevalence-dependent.

### Documented infrastructure

The OHDSI Phenotype Library is a public, version-controlled repository of cohort definitions that conform to the OMOP CDM and the Circe object model. Definitions receive stable identifiers, DOI-versioned releases, and a voluntary open peer review managed by the OHDSI Phenotype Development and Evaluation Workgroup. Accepted definitions become immutable, with deprecated and error statuses available later. The library also publishes diagnostics and validity checks, including unused concept sets, domain mismatches, and non-standard concepts. PheKB, the Phenotype KnowledgeBase, is the eMERGE-associated repository of published phenotype algorithms and their performance data.

These repositories change the economics of cohort building: a team can start from a peer-reviewed definition rather than from a blank query. They do not remove the need for local validation. A definition validated at one site is a hypothesis at another.

### Evidence table

| Study | Algorithm | Sites or population | Reported performance | Lesson for a local build |
|---|---|---|---|---|
| Newton et al., 2013 | 13 eMERGE phenotypes | 5 sites | PPV 67.7-100%; about 75% of 51 reviews at or above 90% | Site-specific validation is required even for a shared algorithm |
| Pacheco et al., 2018 | Executable BPH phenotype | 8 sites, 6 executed | PPV at least 90% at 4 validating sites; 3 of 5 sites above 90% overlap | A common format improves but does not guarantee reproducibility |
| Pacheco et al., 2023 | 6 NLP-enhanced phenotypes | 9 eMERGE sites | Precision and recall improved or held for 5 of 6 | NLP helps, but negation and document heterogeneity remain |
| Spratt et al., 2017 | 8 diabetes definitions | 173,503 patients | Sensitivity 62-94%, specificity 95-99% | Definition choice changes cohort size and composition materially |
| Bhasin et al., 2024 | Validation methodology | Simulation and example | Naive sensitivity 0.942 vs adjusted 0.620 | Sampling design must be corrected before reporting performance |

The table is a warning against treating a published PPV as a property of the algorithm. In every study, the result depends on the population, the reference standard, and the validation design. A local team that adopts an algorithm and reports only its published PPV has imported a number, not a validation.

### What the evidence means for a local build

A local build should start from a published definition, adapt the code set to the local terminology and formulary, run diagnostics, and validate on a local sample. The published definition saves the work of enumerating codes; it does not save the work of confirming that the codes mean the same thing locally. The adaptation should be documented as a deviation, with the reason and the expected effect on sensitivity and specificity.

The evidence also sets expectations for sample size. A PPV of 0.90 estimated from 100 charts has a 95 percent confidence interval of roughly 0.84 to 0.96. A programme that needs to distinguish 0.85 from 0.92 needs several hundred charts. Budget the validation before promising the cohort.

> **Inference:** The published PPV and sensitivity ranges are not benchmarks that a local cohort must meet. They describe algorithms, populations, and reference standards that differ from the local setting. A local PPV below the published range is a signal to investigate the code set, the population, or the chart-review method, not proof that the cohort is wrong.

### How mature organisations operate

Three practices distinguish organisations with reliable cohorts.

First, they maintain a phenotype catalogue. Every cohort has an owner, a version, a code set, a validation report, and a list of consumers. The catalogue is searchable, so a new team does not rebuild a diabetes cohort that already exists with a validated definition.

Second, they run cohort diagnostics on every release. Counts by month, age and sex distributions, prevalence by site, and overlap with related cohorts are reviewed before the cohort is published. A sudden change is treated as a defect until explained.

Third, they separate the cohort from the measure. The cohort is the clinical population; the measure denominator is the subset eligible for a specific quality or payment programme. The two are documented separately, and the measure specification is versioned independently of the cohort. This separation prevents a measure update from silently changing the registry.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Treating a diagnosis code as disease truth | Prevalence is implausible for the population | Codes are used for billing and rule-out, not only diagnosis | Cohort includes non-cases and misses uncoded cases | Validate with chart review and set a PPV threshold |
| Lookback too short | Cohort count keeps rising as lookback lengthens | Prevalent cases with old evidence are excluded | Prevalence understated; incidence overstated | Compare 12, 24, and 36-month lookbacks |
| Time-zero leakage | Model performance is excellent in development | Features recorded after the outcome are included | Performance collapses in deployment | Freeze the feature window at time zero |
| Exclusions applied after the index date | Excluded patients have the outcome more often | Post-index information determines membership | Selection bias removes the sickest patients | Apply exclusions using pre-index data only |
| Missing data treated as absence | A large share of the cohort has no lab or visit | Documentation and linkage gaps look like no care | Care-gap rate overstated | Link external data; report missingness |
| Code set not versioned | Count changes after an EHR or coding update | Codes are added, retired, or remapped | Trend break attributed to the population | Version the code set; mark update dates |
| Validation only on algorithm positives | Sensitivity is reported without a negative sample | Sampling design cannot estimate false negatives | Sensitivity and specificity are biased | Add a negative sample or apply bias correction |
| Verification bias uncorrected | Sensitivity is much higher than published ranges | Positives are oversampled without weights | Performance overstated | Weight by sampling fractions |
| Transfer counted as a readmission | Readmission rate is higher than the benchmark | A transfer creates a second admission | Quality and cost metrics are inflated | Apply a transfer rule and consolidate episodes |
| Death treated as loss to follow-up | Mortality is not linked | Deceased patients cannot have the outcome | Outcome rate understated | Link mortality; treat death as competing risk |
| New patients excluded by lookback | Cohort is older and more established than the population | Clean-period requirement removes new patients | Selection against mobile and disadvantaged patients | Report cohort with and without new patients |
| Out-of-network care invisible | Provider cohort looks healthier than the payer's data | Care outside the network is not recorded | Care gaps and readmissions understated | Link claims or HIE; report leakage |
| Cohort count changes without explanation | A monthly dashboard shows a step change | Data pipeline, code set, or definition changed | Decision made on an artefact | Freeze versions; investigate changes above 10 percent |
| Measure denominator conflated with cohort | Reported performance differs from the official measure | Cohort and measure have different exclusions | Payment or quality report is wrong | Document cohort and denominator separately |
| NLP negation error | "No evidence of" is counted as evidence | The extractor misses negation or family history | False positives in text-derived cohorts | Validate NLP against chart review |

Three failure modes are worth expanding because they are easy to miss and hard to detect.

**The denominator drift.** A cohort used for a monthly dashboard is rebuilt each month with the latest data. If the code set is stable but the underlying EHR changes, the cohort count can drift without any change in the clinical population. The drift is usually gradual and is attributed to "seasonality" or "growth". The control is to hold the definition fixed and to compare the count against an independent source, such as claims or a registry, every quarter.

**The validation sample that validates the wrong thing.** A chart review that selects only patients the clinicians already believe have the disease will produce a high PPV and tell you nothing about the false positives. A review that selects only algorithm positives can estimate PPV, but it cannot estimate sensitivity. The sampling frame must match the metric being reported.

**The code set that encodes a workflow.** A diagnosis code may be added to the problem list only when a care gap is closed, or a lab may be ordered only when a quality measure is due. The code then reflects the workflow, not the disease. This is a form of confounding by indication at the data level, and it is particularly common in pay-for-performance settings.

### Pre-release checklist

Before a cohort is released, eight checks should be signed off. Each has an artefact, so the check is verifiable rather than a matter of opinion.

| Check | Artefact | Owner |
|---|---|---|
| Concept and intended use are written | Specification, first two fields | Clinical lead |
| Code set has a vocabulary and version | Code-set table | Analytics |
| Attrition reconciles base to final | Attrition log | Analytics |
| Every exclusion is applied pre-index | Exclusion audit | Analytics |
| Windows are anchored to time zero | Specification, window fields | Analytics |
| Validation sample and weights are documented | Validation report | Analytics and clinical reviewer |
| PPV meets the threshold for the intended use | Validation report | Clinical governance |
| Cohort and measure denominator are separate | Measure mapping | Quality or finance |

A cohort that passes seven of eight checks is not a cohort with one minor issue. It is a cohort with an unknown error in the missing dimension, and the error is likely to be in the dimension that was hardest to document.

### Five red flags in a cohort review

- The cohort count is reported without a base population.
- The code set is a single code or a single code family.
- The validation report gives PPV but not the sampling frame.
- The lookback is described as "all available history".
- The cohort and the measure denominator are the same number.

Each red flag is cheap to check and each has caused a public correction of a reported rate.

> **Note:** A cohort with a PPV of 0.90 still has 10 false positives per 100 flagged patients. At an outreach cost of $18 per patient, 1,000 flagged patients include about 100 unnecessary contacts, or $1,800 of avoidable cost. That may be acceptable for a low-cost intervention and unacceptable for an expensive one. The threshold is a function of the action, not a universal standard.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test |
|---|---|---|---|
| Coded data reflect the clinical condition | The cohort is built from codes | Coding is incomplete, rule-out codes are used, documentation varies | Chart-review sample |
| The base population is representative | Prevalence and incidence depend on it | The data cover only one setting or payer | Compare to claims or a regional benchmark |
| The lookback is long enough | History and incidence depend on it | Patients have short records or old evidence | Sensitivity analysis at 12, 24, 36 months |
| The code set is complete for the concept | The phenotype depends on it | New codes, local coding practice, code-set drift | Terminology review and validation |
| Exclusions are applied at the right time | Membership must not depend on the outcome | Post-index data used in exclusions | Audit exclusion dates against index dates |
| The validation sample is representative | PPV and sensitivity are estimated from it | Sampling only positives or one site | Document sampling frame and weights |
| The reference standard is accurate | Validation compares against it | Reviewer disagreement, unblinded review | Report kappa and adjudication |
| The data are complete for out-of-network care | Care gaps and outcomes depend on it | Leakage to other providers | Compare EHR to claims |
| The cohort version is stable | Trends compare like with like | Definition changed mid-period | Version control and change log |
| The measure specification is applied faithfully | Reporting must be comparable | Local deviations from the steward's rules | Audit against the specification |

### Cheap tests for expensive assumptions

| Assumption | Cheap test | Time | What it detects |
|---|---|---|---|
| Coded data reflect the condition | Review 30 random positives and 30 random negatives | 3 hours | Gross PPV or NPV failure |
| The lookback is long enough | Recompute the cohort at 12, 24, and 36 months | 1 hour | Truncation of prevalent cases |
| The code set is complete | Compare to a published value set and list the differences | 2 hours | Missing code families |
| Exclusions are pre-index | Check the maximum date used in each exclusion | 1 hour | Temporal leakage |
| Out-of-network care is immaterial | Compare EHR and claims counts for one service | 2 hours | Leakage |
| The cohort is stable | Plot monthly counts and mark data or code changes | 1 hour | Pipeline or definition drift |

These tests do not replace a full validation. They are a triage: if a cheap test fails, the expensive validation is not yet worth commissioning.

The limits of cohort construction follow from these assumptions. A cohort is a coded approximation of a clinical concept, and its performance is bounded by the quality of the coding, the completeness of the data, and the validity of the reference standard. It cannot tell you what a clinician knew, what a patient experienced, or what care was delivered outside the data. It can tell you who is likely to meet a definition, with what error rate, and how that error affects the decision.

A second limit is that validation is expensive. A 400-chart review can cost tens of thousands of dollars in clinician time, and most organisations cannot validate every cohort every year. The practical approach is to validate new definitions, high-stakes definitions, and definitions that show a suspicious change. Routine monitoring can use a smaller sample and a narrower metric, such as PPV on 50 charts per quarter.

A third limit is that portability is not guaranteed by a common data model. The OMOP CDM makes the query portable; it does not make the coding, the population, or the clinical practice portable. A definition validated at an academic centre may perform differently at a community hospital with a different case mix and a different EHR configuration.

## 12. Exercises

### Exercise 1. Write a cohort specification (recall and application)

Choose one of these concepts: adults with heart failure, patients with a first unplanned admission for asthma, or patients overdue for colorectal cancer screening. Write the twelve-field specification from Section 5, including the base population, design, index event, eligibility, exclusions, lookback, follow-up, code sets, data model, validation plan, limitations, and version.

**Deliverable:** a two-page specification.

**Acceptance criteria:** every field is populated; the index date rule is unambiguous; exclusions have a clinical rationale; the code sets name a vocabulary and version.

**Time:** 60 minutes.

### Exercise 2. Compute attrition (application)

A base population of 120,000 adults is reduced by the following criteria, in order: evidence of the condition, 14,200 remain; at least one supportive criterion, 13,600; at least 12 months of prior history, 13,100; not in hospice, 13,040; valid address, 12,880. Compute the number and percentage removed at each step, the final cohort size, and the overall attrition rate. Identify the step with the largest percentage drop and state whether it is plausible.

**Deliverable:** an attrition table with counts and percentages.

**Acceptance criteria:** percentages are computed against the previous step, not the base; the largest drop is identified; the plausibility judgement is justified.

**Time:** 30 minutes.

### Exercise 3. Correct a validation estimate (application)

A cohort has 12,000 algorithm-positive patients out of a base of 120,000. A validation sample of 300 positives finds 264 true cases. A validation sample of 300 negatives finds 18 true cases. Compute PPV, NPV, the expanded true positives and false negatives, corrected prevalence, and sensitivity. Then explain why a validation study that reviewed only the 300 positives could not report sensitivity without an assumption.

**Deliverable:** a calculation table and a short explanation.

**Acceptance criteria:** the expansion uses the correct denominators; the corrected prevalence is expressed as a percentage of the base; the explanation names the missing cell in the 2x2 table.

**Time:** 45 minutes.

### Exercise 4. Critique a code set (critique)

A team proposes a heart failure cohort using only ICD-10-CM I50.9. Identify at least five problems with this choice, including specificity, sensitivity, coding practice, temporality, and the distinction between systolic and diastolic heart failure. Propose a revised code set and a confirmatory criterion.

**Deliverable:** a numbered critique and a revised code-set table.

**Acceptance criteria:** at least one problem concerns false negatives and one concerns false positives; the revised set names specific code families; the confirmatory criterion is clinically justified.

**Time:** 45 minutes.

### Exercise 5. Implement a cohort in SQL (application)

Using the schema from Section 7, write a SQLite query that builds a heart failure cohort with three criteria: a qualifying diagnosis, a natriuretic peptide lab above a threshold, or a loop diuretic. Include a 12-month lookback, exclude hospice patients, and produce the attrition counts at each step. Run the query and paste the output.

**Deliverable:** the query, the output, and a data dictionary for the cohort table.

**Acceptance criteria:** the query executes; each criterion is implemented as a separate step; the attrition is reconcilable; NULL values are handled explicitly.

**Time:** 90 minutes.

### Exercise 6. Audit temporal leakage (critique)

A readmission model uses the following features: prior admissions in the 12 months before discharge, discharge disposition, number of medications at discharge, and a follow-up appointment scheduled within 7 days after discharge. The model reports an AUROC of 0.84. Identify the leakage, explain its mechanism, and propose a corrected feature set. State how the corrected model's performance should be re-estimated.

**Deliverable:** a leakage audit with the affected features, the mechanism, and a corrected specification.

**Acceptance criteria:** the post-discharge feature is identified; the mechanism is explained in terms of time zero; the corrected evaluation uses a temporal split.

**Time:** 45 minutes.

### Exercise 7. Design a validation study (design)

Design a validation study for a new sepsis cohort at a 600-bed hospital. Specify the sampling frame, the sample size for PPV and for sensitivity, the reference standard, the reviewer process, the blinding, the adjudication rule, and the reporting format. Justify the sample size with a precision calculation.

**Deliverable:** a two-page protocol with a sample-size calculation.

**Acceptance criteria:** the sampling frame supports both PPV and sensitivity; the precision target is stated; reviewer blinding and adjudication are described; the reporting includes confidence intervals.

**Time:** 90 minutes.

### Exercise 8. Assess portability (critique and design)

A phenotype algorithm validated at an academic centre with a PPV of 0.93 is proposed for a community hospital. The community hospital has a different EHR, a higher share of new patients, and no onsite chart-review capacity. Identify the threats to portability and propose a local validation plan that fits the available resources.

**Deliverable:** a risk assessment with four to six threats and a validation plan.

**Acceptance criteria:** at least one threat concerns case mix and one concerns data capture; the plan is feasible without onsite chart review; the plan states what would trigger a full validation.

**Time:** 60 minutes.

### Assessment notes

Exercises 1 to 4 can be completed individually and checked against Section 13. Exercises 5 and 6 require a data environment; if none is available, use the synthetic schema in Section 7. Exercises 7 and 8 are group design tasks and should be assessed with the rubric in Section 13, not against a single answer.

The estimated times assume a learner who has read the topic and has access to a laptop. Add 30 minutes to Exercise 5 if the learner must install SQLite and create the schema from scratch.

## 13. Self-check answers

### Exercise 1

A strong specification names the concept in clinical language, states whether the cohort is prevalent or incident, defines the index event with a date rule and a precedence rule, lists eligibility and exclusions with rationale, sets lookback and follow-up windows in days, names the vocabulary and version for each code set, identifies the tables in the target model, describes the validation sample, states the known limitations, and assigns a version. A weak specification lists codes without a concept, omits the index rule, or treats the lookback as "all available history".

### Exercise 2

| Step | Criterion | Remaining | Removed | Percent of previous |
|---|---|---:|---:|---:|
| 0 | Base population | 120,000 | not applicable | not applicable |
| 1 | Evidence of the condition | 14,200 | 105,800 | 88.17% |
| 2 | Supportive criterion | 13,600 | 600 | 4.23% |
| 3 | 12-month history | 13,100 | 500 | 3.68% |
| 4 | Not in hospice | 13,040 | 60 | 0.46% |
| 5 | Valid address | 12,880 | 160 | 1.23% |

The final cohort is 12,880, an overall attrition of 107,120 patients or 89.27 percent. The largest drop is step 1, which is expected when the base population is defined by an encounter rather than by a diagnosis. A drop of 88 percent at a supportive-criterion step would be implausible; a drop of 88 percent at the evidence step is normal. The address exclusion removes 1.23 percent, which is small but not random: patients without a stable address are more likely to be uninsured and mobile.

### Exercise 3

```text
PPV = 264 / 300 = 0.88
NPV = 282 / 300 = 0.94
True positives among all positives = 12,000 x 0.88 = 10,560
False negatives among all negatives = (120,000 - 12,000) x 0.06 = 6,480
Corrected true cases = 10,560 + 6,480 = 17,040
Corrected prevalence = 17,040 / 120,000 = 14.20%
Sensitivity = 10,560 / 17,040 = 61.97%
Observed algorithm prevalence = 12,000 / 120,000 = 10.00%
```

The observed prevalence understates the corrected prevalence by 4.20 percentage points. A study that reviewed only the 300 positives knows the top-left and top-right cells of the 2x2 table but not the bottom row. It can estimate PPV but not sensitivity, because sensitivity requires the number of true cases among algorithm negatives. Assuming zero false negatives would set sensitivity to 100 percent, which is almost never true. Assuming a literature NPV is possible, but then the estimate is only as good as the assumption.

### Exercise 4

A strong critique includes at least five of the following:

1. **I50.9 is unspecified.** It does not distinguish systolic, diastolic, combined, or right heart failure, so the cohort mixes clinically different populations.
2. **It misses common codes.** Hypertensive heart disease with heart failure (I11.0), hypertensive heart and chronic kidney disease with heart failure (I13.0, I13.2), and rheumatic heart failure (I09.81) are common and would be excluded.
3. **It includes rule-out and history codes.** A single code may be recorded during a diagnostic workup and later removed; the cohort needs a confirmatory criterion or a second code.
4. **Coding practice varies.** Specialists code heart failure more specifically than primary care, so a diagnosis-only cohort underrepresents primary care patients.
5. **Temporality is unspecified.** A code from five years ago may represent resolved disease; the lookback must be stated.
6. **There is no severity or ejection fraction information.** A cohort for a clinical programme may need ejection fraction or NYHA class, which are not in the diagnosis code.

A revised set includes I50.1, I50.2x, I50.3x, I50.4x, I50.9, I11.0, I13.0, I13.2, and I09.81, with a confirmatory criterion of a natriuretic peptide result above the local threshold or a loop diuretic order. The confirmatory criterion raises PPV and lowers sensitivity, and the trade-off should be stated.

### Exercises 5 to 8 rubric

Exercise 5 is assessed on whether the query executes, whether each criterion is a separate step, whether the attrition is reconcilable, and whether NULL handling is explicit. A strong submission produces a data dictionary and notes the run date.

Exercise 6 is assessed on identifying the 7-day follow-up appointment as post-index information. A strong answer explains that the feature is only observable after discharge and is therefore unavailable at time zero, proposes removing it or replacing it with a pre-discharge scheduling flag, and re-estimates performance with a temporal split. A weak answer says the feature is "probably fine because it is clinical".

Exercise 7 is assessed on the sampling frame, the sample size justification, the reference standard, blinding, adjudication, and confidence intervals. A strong answer computes the sample size for a desired margin of error, separates the positive and negative samples, and states that the reviewers are blinded to the algorithm result.

Exercise 8 is assessed on the threats (different EHR, case mix, coding practice, new-patient share, no chart review), the feasibility of the local plan (for example, 50 positives and 50 negatives reviewed by a remote clinician, plus a comparison to claims), and the trigger for a full validation (a PPV below 0.80 or a cohort count change above 20 percent).

## 14. Interview questions

**1. How do you define a cohort?**
A strong answer starts with the clinical concept and the intended use, then describes the base population, index event, eligibility, exclusions, windows, and code sets, and ends with the validation plan. A weak answer starts with a diagnosis code.

**2. What is the difference between prevalent and incident cohorts?**
A strong answer explains that prevalent cohorts include existing cases at a point in time, incident cohorts include new cases during a period, and the choice changes the index rule, the lookback, and the denominator. A weak answer says one is "all cases" and the other is "new cases" without implications.

**3. What is time zero, and why does it matter?**
A strong answer defines it as the instant features are frozen and the prediction or measurement begins, and explains that any feature recorded after time zero is leakage. A weak answer equates time zero with the index date without discussing availability.

**4. How do you validate a phenotype?**
A strong answer describes a chart-review reference standard, a sampling frame that supports PPV and sensitivity, blinding, adjudication, and bias correction for stratified sampling. A weak answer says "compare to the problem list".

**5. Why is a PPV of 0.90 not always good enough?**
A strong answer relates the threshold to the cost of false positives and false negatives in the intended use, and notes that a high PPV can coexist with low sensitivity. A weak answer treats 0.90 as a universal standard.

**6. What is verification bias?**
A strong answer explains that sampling only algorithm positives inflates sensitivity and deflates specificity unless the sampling fractions are corrected, and cites the direction of the bias. A weak answer confuses it with selection bias in the clinical population.

**7. How do you handle missing data in a cohort?**
A strong answer distinguishes missing at random, missing because the care happened elsewhere, and missing because the patient did not receive care, and proposes linkage and sensitivity analysis. A weak answer drops patients with missing values.

**8. How do you make a cohort portable?**
A strong answer uses a common data model, standard terminologies, versioned code sets, and site-specific validation, and reports performance by site. A weak answer says "use OMOP" and stops.

**9. What would you do if the cohort count dropped 15 percent after an EHR upgrade?**
A strong answer treats the drop as a definition or data-capture change, compares the code and site distributions before and after, checks the mapping, and re-validates before using the new count. A weak answer assumes the new count is correct.

**10. How do you decide how long the lookback should be?**
A strong answer ties the lookback to the clinical concept and the data availability, tests 12, 24, and 36 months, and reports the cohort with fixed and available windows. A weak answer says "one year is standard".

**11. What is the difference between a cohort and a measure denominator?**
A strong answer explains that the measure denominator applies additional programme-specific exclusions, enrollment requirements, and age limits, and that the two must be reported separately. A weak answer uses the terms interchangeably.

**12. How would you validate a cohort if you have no chart-review capacity?**
A strong answer uses a structured second source (claims, registry, HIE), a small remote review, and a comparison of prevalence to a published benchmark, and states the residual uncertainty. A weak answer says validation is impossible.


**13. How do you know a cohort is fit for purpose?**
A strong answer names the intended use, the error that matters most for that use, the validation metric that captures it, and the threshold that the programme can tolerate. It also states the residual error and the monitoring plan. A weak answer cites a generic PPV target without connecting it to the action.

**14. What is the first thing you check when a cohort count changes unexpectedly?**
A strong answer checks whether the definition, the code set, or the data pipeline changed before checking whether the population changed, and compares the count by site, age, and code to localise the change. A weak answer assumes a real change in disease burden.

## 15. Key terms

- **Base population:** the universe of patients from which a cohort is drawn, with data coverage stated.
- **Cohort:** a defined set of patients meeting specified criteria in a specified period.
- **Index event:** the event that qualifies a patient for cohort entry.
- **Index date:** the date assigned to the index event.
- **Time zero:** the instant at which features are frozen and measurement begins.
- **Eligibility criterion:** a condition a patient must meet to enter the cohort.
- **Exclusion criterion:** a condition that removes an otherwise eligible patient.
- **Lookback window:** the period before time zero used to establish history.
- **Follow-up window:** the period after time zero used to ascertain outcomes.
- **Prevalent cohort:** all patients with the condition at a point in time.
- **Incident cohort:** patients newly meeting the condition during a period.
- **Left truncation:** unobserved history before data availability begins.
- **Immortal time:** person-time during which the outcome cannot occur by design.
- **Censoring:** end of observation before the outcome.
- **Competing risk:** an event that prevents the outcome of interest.
- **Phenotype:** a computable definition of a clinical condition.
- **Computable phenotype:** a phenotype expressed as executable logic.
- **Code set:** the codes from one or more terminologies that define a concept.
- **Value set:** a set of codes used in a measure, often with a steward-defined version.
- **ICD-10-CM:** the US diagnosis coding system.
- **ICD-10-PCS:** the US inpatient procedure coding system.
- **CPT:** Current Procedural Terminology for professional services.
- **LOINC:** the terminology for laboratory tests and observations.
- **RxNorm:** the normalized terminology for medications.
- **SNOMED CT:** the clinical terminology for findings, disorders, and procedures.
- **OMOP CDM:** a common data model for observational health data.
- **OHDSI:** the collaborative that maintains the OMOP CDM and related tools.
- **eMERGE:** the Electronic Medical Records and Genomics network.
- **PheKB:** the Phenotype KnowledgeBase repository.
- **PPV:** positive predictive value.
- **NPV:** negative predictive value.
- **Sensitivity:** the proportion of true cases detected.
- **Specificity:** the proportion of true non-cases excluded.
- **Verification bias:** bias from validating a non-random subset of patients.
- **Chart review:** manual abstraction of the record as a reference standard.
- **Gold standard:** the best available reference definition.
- **Misclassification:** assignment to the wrong cohort class.
- **Prevalence:** the proportion of the base population with the condition.
- **Incidence:** new cases per population per time.
- **Care gap:** recommended care not received by an eligible patient.
- **Denominator:** the eligible population for a measure.
- **Numerator:** the patients who met the measure criterion.
- **Data dictionary:** the documented definition of every field and code.
- **Portability:** the ability to run a phenotype elsewhere with comparable performance.
- **Validation sample:** the subset selected for chart review.

## 16. Sources

All URLs were accessed on 25 October 2026. Validation performance figures are reported as published; they are not local benchmarks.

### Validation studies

1. Newton, K. M. et al. Validation of electronic medical record-based phenotyping algorithms: results and lessons learned from the eMERGE network. *JAMIA*, 2013. https://doi.org/10.1136/amiajnl-2012-000896
2. Pacheco, J. A. et al. A case study evaluating the portability of an executable computable phenotype algorithm across multiple institutions and electronic health record environments. *JAMIA*, 2018. https://doi.org/10.1093/jamia/ocy101
3. Pacheco, J. A. et al. Evaluation of the portability of computable phenotypes with natural language processing in the eMERGE network. *Scientific Reports*, 2023. https://doi.org/10.1038/s41598-023-27481-y
4. Spratt, S. E. et al. Assessing electronic health record phenotypes against gold-standard diagnostic criteria for diabetes mellitus. *JAMIA*, 2017. https://academic.oup.com/jamia/article-pdf/24/e1/e121/34148995/ocw123.pdf
5. Bhasin, A. et al. Measures of electronic phenotyping algorithms. *ACI Open*, 2024. https://www.thieme-connect.com/products/ejournals/pdf/10.1055/a-2402-5937.pdf

### Infrastructure and methods

6. OHDSI. The OHDSI Phenotype Library. https://ohdsi.github.io/PhenotypeLibrary/
7. OHDSI. Observational Health Data Sciences and Informatics. https://ohdsi.org/
8. OHDSI. OMOP Common Data Model. https://ohdsi.github.io/CommonDataModel/
9. OHDSI. ATLAS cohort definition tool. https://atlas.ohdsi.org/
10. eMERGE Network. https://emerge-network.org/
11. PheKB. Phenotype KnowledgeBase. https://phekb.org/
12. OHDSI. Creating cohort definitions tutorial, 2022. https://www.ohdsi.org/wp-content/uploads/2022/10/Creating-Cohorts-Tutorial-2022.pdf

### Terminologies

13. Centers for Disease Control and Prevention. ICD-10-CM. https://www.cdc.gov/nchs/icd/icd-10-cm.htm
14. LOINC. https://loinc.org/
15. National Library of Medicine. RxNorm. https://www.nlm.nih.gov/research/umls/rxnorm/
16. SNOMED International. https://www.snomed.org/
17. American Medical Association. CPT. https://www.ama-assn.org/practice-management/cpt
18. CMS. Healthcare Common Procedure Coding System. https://www.cms.gov/medicare/coding-billing/healthcare-common-procedure-system

### Standards and regulation

19. HL7. FHIR Release 4. https://hl7.org/fhir/
20. Office of the National Coordinator for Health IT. US Core Data for Interoperability. https://www.healthit.gov/isa/us-core-data-interoperability-uscdi
21. US Department of Health and Human Services. HIPAA. https://www.hhs.gov/hipaa/index.html
22. US Department of Health and Human Services. 45 CFR Part 46 (Common Rule). https://www.hhs.gov/ohrp/regulations-and-policy/regulations/45-cfr-46/index.html
23. US Food and Drug Administration. Real-World Evidence. https://www.fda.gov/science-research/science-and-research-special-topics/real-world-evidence
24. 21st Century Cures Act, Public Law 114-255. https://www.congress.gov/bill/114th-congress/house-bill/34
25. GDPR. Regulation (EU) 2016/679. https://gdpr-info.eu/

### Data resources

26. Agency for Healthcare Research and Quality. HCUP. https://hcup-us.ahrq.gov/
27. PCORnet. https://pcornet.org/
28. Sentinel Initiative. https://www.sentinelinitiative.org/

### How to read a phenotype validation paper

Six questions determine whether a reported PPV or sensitivity is usable in a local decision:

1. What was the reference standard, and was it applied to a random sample or only to algorithm positives?
2. What was the base population, and does it resemble the local population?
3. Which code set and vocabulary version were used?
4. Was the validation sample weighted, and were confidence intervals reported?
5. Was the index rule prevalent or incident, and what lookback was used?
6. Did the authors report performance by site or subgroup, or only a pooled estimate?

A paper that answers fewer than four of these questions is a description, not a validation. It may still be useful for the code set, but it should not be cited as evidence that the phenotype will perform locally.

> **Note:** The validation studies report performance in their own populations and against their own reference standards. A local PPV or sensitivity that differs is not necessarily an error. The correct comparison is to the study's population, code set, and validation design, not to the headline number alone.

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`01_health-system-operating-model.md`](01_health-system-operating-model.md) — the operating model that defines the population and the decision.
- Next topic: [`03_claims-analytics.md`](03_claims-analytics.md) — episodes, attribution, risk adjustment, and cost.
- Related topic: [`04_operational-models.md`](04_operational-models.md) — the models that consume cohort labels.
- Root survey file: [`../../08_ehr_claims_operations.md`](../../08_ehr_claims_operations.md)
- Labs: [`../../labs/`](../../labs/)

The cohort is the unit of analysis for the rest of the module. A readmission model, a no-show model, a care-gap dashboard, and a quality measure all begin with a population definition. If a later topic produces a result that seems implausible, return to the cohort specification and check the index rule, the lookback, and the validation sample before questioning the model.

A final note on reuse: the cohort definition in this topic is the input to the claims episode logic in Topic 03 and the operational models in Topic 04. If those topics use a different cohort, the comparison between clinical and claims-based rates is not valid. Keep one definition, version it, and cite the version in every downstream analysis.

### Reading path

| If you need to | Read | Then |
|---|---|---|
| Build a registry | Sections 5 and 6 | Topic 04 for the outreach model |
| Validate a phenotype | Sections 5, 9, and 11 | The eMERGE papers in Section 16 |
| Implement in OMOP | Section 7 and the OHDSI library | The ATLAS tutorial |
| Report a quality measure | Sections 4 and 8 | Topic 06 |
| Link to claims | Sections 5 and 8 | Topic 03 |


The three topics together form a chain: Topic 01 defines the operating model, Topic 02 defines the population, and Topic 03 defines the payment and cost view of that population. A result that cannot be traced through all three is a result without a denominator.

Before moving to Topic 03, confirm three things:

1. You can state the cohort's index rule, lookback, and validation metric without opening the file.
2. You can explain why the cohort count differs from the population estimate.
3. You can name the decision that would change if the PPV were 0.10 lower.

If any answer is missing, the cohort is not yet a usable artefact.

A cohort that cannot survive those three questions will not survive a review by finance, a clinician, or a regulator.
