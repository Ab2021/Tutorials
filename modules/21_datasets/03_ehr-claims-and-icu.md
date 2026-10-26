# EHR, Claims, and ICU Data

> Module 21 — Datasets, Tools, and Benchmarks
> Topic 03 of 12
> Estimated study time: 4 hours
> Prerequisites: Topic 01, Topic 02
> Version 2026.3

## 1. Why this topic exists

Most biomedical data science in industry runs on three record systems: the electronic health record (EHR), the insurance claim, and the intensive care unit (ICU) database. They look similar in a table and answer different questions. The EHR records what clinicians observed and did. The claim records what was billed and paid. The ICU database records a high-resolution slice of a selected, critically ill population. Confusing them is the most common source of claims that survive a review and fail in deployment.

The confusion has a predictable shape:

- A team builds a readmission model on a single-center EHR and reports a 30-day readmission rate far below the published national rate, because readmissions to other hospitals are invisible.
- A team builds a severity model on claims and discovers that the strongest predictor is a billing code that reflects how well the hospital documented, not how sick the patient was.
- A team uses ICU data to claim ward-level performance, because the ICU is where the high-frequency data are.
- A team counts hospitalizations as patients and overstates the cohort by the repeat-admission factor.
- A team trains on an era that spans the ICD-9 to ICD-10 transition and learns the coding system rather than the disease.

Each failure is a mismatch between the unit of the data and the unit of the claim. The fix is not a better model. It is a explicit mapping from the decision to the record system, the unit of analysis, the coverage of the outcome, and the coding era.

This topic covers the major resources in the category: MIMIC-IV and its derivatives, eICU-CRD, HiRID, AmsterdamUMCdb, UK Biobank, All of Us, Epic Cosmos, EHRSHOT, Synthea, CMS synthetic claims, and the OMOP common data model. For each, it states what the resource can support and what it cannot. That second column is the one that prevents expensive mistakes.

The industry stakes are concrete. A payer building a readmission programme needs claims, because readmission is a billing event that crosses hospitals. A hospital building a deterioration alert needs the EHR, because the physiology is clinical and the action is local. A researcher studying sepsis physiology needs an ICU database, because the resolution and the labels are there. A product team selling to community hospitals needs a dataset that contains community hospitals, and none of the academic ICU resources does.

> **Note:** Sizes and versions in this topic were checked on 26 October 2026 against the custodian pages cited in section 16. MIMIC-IV v3.1 was released on 11 October 2024; All of Us Curated Data Repository v9 was released on 26 June 2026. Counts change with releases, and the unit of a count matters as much as the number.

## 2. Learning outcomes

By the end of this topic you can:

1. **Distinguish** EHR, claims, ICU, registry, and survey data by what they measure and what decision they can support. Assessed by Exercise 1 and self-check item 1.
2. **Choose** the unit of analysis for a claim and compute the repeat-admission factor that separates patients from encounters. Assessed by Exercise 2 and self-check item 3.
3. **Audit** outcome capture across settings, including out-of-network readmissions, post-discharge deaths, and administrative censoring. Assessed by Exercise 3.
4. **Validate** a phenotype built from diagnosis codes against a clinical reference and report its positive predictive value. Assessed by Exercise 4.
5. **Detect** coding-era effects and definition changes that make a pooled cohort non-comparable. Assessed by Exercise 5.
6. **Explain** why a single-center ICU dataset cannot support a national incidence or equity claim, with the specific mechanism. Assessed by Exercise 6.
7. **Design** a split and an evaluation for an EHR or claims cohort that respects patient, site, and time. Assessed by Exercise 7.
8. **Write** a data-source memo that names the source, the unit, the coverage limit, and the claim it cannot support. Assessed by Exercise 8.

## 3. The core mental model

Think of the health system as a care continuum and each dataset as a lens with a different field of view and a different blind spot.

```text
   PATIENT
      |
      v
   SYMPTOM ---> PRIMARY CARE ---> ED ---> WARD ---> ICU ---> DISCHARGE
      |              |             |        |         |          |
      |              |             |        |         |          v
      |              |             |        |         |      READMISSION
      |              |             |        |         |      (any hospital)
      |              |             |        |         |
   SURVEY         EHR/CLAIMS     EHR     EHR/ICU    ICU DATA
   (sample)       (episode)     (visit)  (stay)    (high frequency)
      |              |             |        |         |
      +--------------+-------------+--------+---------+
                              |
                              v
                    CLAIM (billing envelope)
                    covers the whole continuum,
                    records little physiology
```

**The lens rule.** Each dataset is sharpest where it was generated and blurry elsewhere. Claims are sharp on utilisation, cost, and cross-hospital events; blurry on severity, physiology, and reason for discontinuation. The EHR is sharp on local clinical detail; blurry on care outside the institution. The ICU database is sharp on high-frequency physiology; blurry on the ward, the community, and long-term outcomes.

**The unit ladder.** A patient has encounters; an encounter has claims; a hospitalization has an ICU stay; an ICU stay has hours. A count at one level is not a count at another. MIMIC-IV v3.1 illustrates the problem: 364,627 patients, 546,028 hospitalizations, and 94,458 ICU stays. The patient count includes 141,175 people with only an emergency department visit, so the hospitalized patient count is 223,452, and the ratio of hospitalizations to hospitalized patients is about 2.44. Any claim about "patients" computed from a hospitalization table is inflated.

**The coverage rule.** An outcome is observable only if the patient remains inside the data. A readmission is observable in a single-center EHR only if the patient returns to that center. A death is observable only if the death is recorded through a linked registry or an in-network event. A negative label is a claim that the event did not happen, and it is only as good as the coverage.

**The coding rule.** Administrative codes are designed for payment and statistics, not for clinical timing. A discharge diagnosis is assigned at the end of the encounter. A present-on-admission flag is a coding judgement, not a clinical measurement. A code can change when the coding rules change, when the documentation improves, or when a financial incentive changes. A model trained across a coding transition can learn the transition.

**The selection rule.** ICU data are selected by acuity; tertiary-center data are selected by referral; volunteer cohort data are selected by willingness; claims data are selected by insurance. None is a probability sample of the population, and each selection changes the answer to a population question.

| Source | Sharp on | Blind to | Typical claim it supports | Claim it cannot support |
|---|---|---|---|---|
| EHR, single centre | Local physiology, orders, notes, workflow | Care outside the institution | Local quality improvement, phenotyping, methods | National incidence, cross-system comparison |
| Claims | Utilisation, cost, cross-hospital events, enrolment | Severity, labs, adherence, out-of-network care | Readmission, cost, treatment patterns | Clinical deterioration, physiological risk |
| ICU database | High-frequency physiology, ICU interventions | Ward, post-discharge, community | Deterioration, ventilation, ICU phenotyping | Ward performance, national ICU quality |
| Registry | Condition-specific incidence and outcomes | Comorbidities outside the registry | Cancer incidence, procedure outcomes | General population risk |
| Survey | Population prevalence and risk factors | Clinical detail, longitudinal follow-up | Prevalence, disparities, trends | Diagnosis-level clinical performance |
| Synthetic | Pipeline shape and schema | Real distributions and fairness | Software testing, teaching | Clinical performance, fairness |

## 4. Definitions and vocabulary

### 4.1 Units and record structures

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Patient | A unique person, usually a `subject_id`. | Confused with a row in an encounter table. | Person |
| Encounter | Any interaction with the health system. | Used as a synonym for admission. | Visit |
| Hospitalization | An inpatient admission with admission and discharge. | Counted as a patient. | Admission |
| ICU stay | A continuous period in an intensive care unit. | Counted as a hospitalization. | Stay |
| Unit stay | A stay in one ICU unit; a patient can have several. | Pooled without deduplication. | Unit encounter |
| ED visit | An emergency department encounter, which may not lead to admission. | Excluded from the denominator by mistake. | Visit |
| Observation stay | A short stay without formal inpatient admission. | Counted inconsistently across payers. | Encounter |
| Claim | A billing request for services. | Treated as a clinical record. | Claim header |
| Claim line | One service within a claim. | Counted as a patient or an admission. | Line item |
| Index admission | The admission that starts the follow-up window. | Chosen after looking at the outcome. | Admission |
| Readmission | A subsequent admission within a defined window. | Counted without checking the hospital. | Event |
| Index date | The anchor date for follow-up. | Set to the outcome date. | Date |

The unit ladder is the first thing to check in any EHR or claims table. Ask what one row represents, then ask what the denominator of the claim represents. If the two differ, the analysis needs a deduplication or aggregation step before any model is trained.

### 4.2 Coding and billing terms

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| ICD-10-CM | US diagnosis coding system for morbidity. | Treated as a clinical ontology. | Code |
| ICD-10-PCS | US procedure coding system for inpatient procedures. | Confused with CPT. | Code |
| CPT | Current Procedural Terminology, used for outpatient and professional billing. | Applied to inpatient procedures. | Code |
| HCPCS | Healthcare Common Procedure Coding System, including supplies and drugs. | Mixed with CPT without a crosswalk. | Code |
| NDC | National Drug Code for packaged drugs. | Treated as a stable identifier over time. | Code |
| DRG | Diagnosis-Related Group, a payment category. | Used as a severity measure. | Category |
| Principal diagnosis | The condition chiefly responsible for admission. | Treated as the only diagnosis. | Code |
| Secondary diagnosis | Additional coded conditions. | Assumed to be present on admission. | Code |
| Present on admission | A flag indicating the condition existed at admission. | Treated as a validated clinical judgement. | Flag |
| Comorbidity | A coexisting condition that affects outcome. | Computed from codes without validation. | Index |
| Upcoding | Assignment of codes that increase payment. | Ignored as a source of drift. | Behaviour |
| Coding depth | The number of codes recorded per encounter. | Assumed constant across sites and years. | Count |

Coding systems are not interchangeable. ICD-10-CM has far more codes than ICD-9-CM, and the transition in the United States on 1 October 2015 changed the measured prevalence of many conditions without changing the patients. A cohort that spans the transition needs either a crosswalk with validation or a restriction to one era.

### 4.3 Clinical and epidemiological terms

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Phenotype | An algorithm that identifies a clinical condition in data. | Treated as a diagnosis. | Algorithm |
| Sepsis-3 | A clinical definition using suspected infection and organ dysfunction. | Replaced by an ICD code. | Definition |
| AKI | Acute kidney injury, defined by creatinine or urine output criteria. | Identified only by diagnosis code. | Definition |
| Comorbidity index | A score summarising coexisting disease burden. | Treated as a severity measure. | Score |
| Immortal time bias | Bias from classifying exposure using future information. | Introduced by cohort definitions. | Bias |
| Competing risk | An event that prevents the outcome of interest. | Ignored in a simple readmission rate. | Statistical concept |
| Administrative censoring | Follow-up ends because the data end, not because the patient is event-free. | Counted as a negative outcome. | Censoring |
| Lookback window | The period before the index date used to define baseline. | Too short to capture chronic disease. | Duration |
| Washout period | A gap used to avoid counting prevalent cases as incident. | Omitted in claims cohorts. | Duration |
| Person-time | The total time at risk contributed by the cohort. | Replaced by a simple count. | Person-years |
| Selection bias | A distortion from how the cohort was chosen. | Ignored when the sample is large. | Bias |
| Transportability | Whether a model trained in one setting works in another. | Asserted from a single test. | Performance delta |

### 4.4 Resource terms

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| MIMIC-IV | De-identified EHR and ICU data from Beth Israel Deaconess Medical Center. | Treated as a US sample. | Dataset |
| eICU-CRD | Multi-hospital ICU data collected through a telehealth programme. | Treated as a random hospital sample. | Dataset |
| All of Us | NIH cohort with surveys, EHR, and genomics. | Treated as a probability sample. | Programme |
| Epic Cosmos | Contributor-only de-identified EHR network with aggregate query access. | Treated as row-level research data. | Product |
| EHRSHOT | Few-shot EHR benchmark from Stanford. | Treated as a clinical validation. | Benchmark |
| OMOP CDM | Common data model for observational health data. | Treated as a data source. | Model |
| Synthea | Open-source synthetic patient generator. | Used to claim clinical performance. | Generator |
| CMS SynPUF | Synthetic Medicare-like claims. | Treated as real utilisation. | Dataset |
| Data dictionary | Documentation of tables, fields, and codes. | Assumed to be complete. | Document |

## 5. The mechanics, step by step

The procedure has 18 steps in four phases: define the question and the unit, audit coverage and selection, build and validate the cohort, and design the evaluation. It assumes the five-gate screen in Topic 01 has already produced a candidate list.

### Phase A — Define the question and the unit

**Step 1. Name the decision and the care setting.**
Input: the product or study question and the workflow.
Output: a decision statement with the setting, the actor, the timing, and the cost of errors.
Rationale: the setting determines which record system contains the event. A readmission decision is a cross-hospital question; a deterioration decision is a local physiology question.
Branch: if the decision is about billing or utilisation, claims are the natural source. If it is about clinical state, the EHR or ICU data are.
Edge case: a decision that spans both, such as discharge planning, needs a linked view and a plan for the unlinked portion.

**Step 2. Choose the unit of analysis.**
Input: the decision statement and the data dictionary.
Output: patient, encounter, hospitalization, ICU stay, claim, or claim line.
Rationale: every count, rate, and model depends on the unit. A patient-level claim needs one row per patient; a stay-level claim needs one row per stay.
Branch: use the unit decision table below.
Edge case: a patient can have multiple ICU stays in one hospitalization. Decide whether the unit is the hospitalization or the stay before aggregating.

**Step 3. Map the care continuum to the data source.**
Input: the unit and the source.
Output: a coverage diagram showing where the source observes the patient and where it stops.
Rationale: the blind spots define the claims the data cannot support.
Branch: if the outcome lies outside the observed continuum, the source cannot measure it without linkage.
Edge case: a patient who moves between hospitals may appear in two claims systems with different identifiers.

**Step 4. Check the denominator.**
Input: the cohort definition and the source's inclusion rules.
Output: the eligible population, the excluded population, and the reason for each exclusion.
Rationale: a rate is only interpretable with a denominator. A readmission rate among survivors is different from a rate among all discharges.
Branch: if the source excludes a group that is central to the claim, the claim fails population fit.
Edge case: the denominator changes when the observation window changes. A 30-day rate and a 90-day rate answer different questions.

### Phase B — Audit coverage and selection

**Step 5. Audit outcome coverage across settings.**
Input: the outcome definition and the source's catchment.
Output: the fraction of expected events that the source can observe, with the mechanism for the rest.
Rationale: a negative label is a claim of absence. If the source cannot see the event, the negative is unverified.
Branch: if out-of-network or out-of-system events are material, the source cannot support the outcome without linkage.
Edge case: death is often recorded through a registry or an in-network event; a death outside the system can appear as a censored observation.

**Step 6. Audit the lookback and washout windows.**
Input: the cohort definition and the available history.
Output: the baseline period, the washout period, and the fraction of the cohort with insufficient history.
Rationale: chronic conditions and prior utilisation need history. A short lookback misclassifies prevalent disease as incident.
Branch: if the source has less history than the lookback requires, restrict the cohort to people with enough observed time.
Edge case: a new enrollee has no claims history, which is different from a healthy person with no claims.

**Step 7. Audit selection by site, acuity, and insurance.**
Input: the cohort and the source metadata.
Output: a coverage table by site, setting, age, sex, race or ethnicity, payer, and year.
Rationale: selection determines transportability and equity.
Branch: if the target population is community hospitals and the source is tertiary academic, the population gate is at most 1.
Edge case: a volunteer cohort can be large and still unrepresentative. Size is not coverage.

**Step 8. Audit coding depth and era.**
Input: the diagnosis and procedure fields by year and site.
Output: codes per encounter, coding system version, and definition changes.
Rationale: coding depth and coding rules change the measured prevalence and the model's learned associations.
Branch: if the cohort spans a coding transition, restrict to one era or validate a crosswalk.
Edge case: a pandemic period can change both coding and care. Model the period or restrict it.

**Step 9. Audit missingness by site and acuity.**
Input: the feature table with site and time.
Output: missingness rates by variable, site, and acuity band, with a mechanism hypothesis.
Rationale: missingness in clinical data is informative. A lab not ordered is a clinical decision.
Branch: if missingness is associated with the outcome, complete-case analysis is biased.
Edge case: a variable that exists only in the ICU is missing by design for ward patients.

### Phase C — Build and validate the cohort

**Step 10. Define the cohort with codes and clinical criteria.**
Input: the phenotype specification and the data dictionary.
Output: an executable cohort definition with codes, thresholds, and time windows.
Rationale: a reproducible cohort is a precondition for a reproducible result.
Branch: if the phenotype relies on a code that changed, use a validated crosswalk or a clinical criterion.
Edge case: a rule-out diagnosis can appear in the record without the disease being present. Exclude or down-weight it.

**Step 11. Validate the phenotype.**
Input: a sample of records and a clinical reference standard.
Output: positive predictive value, sensitivity where estimable, and a confusion matrix.
Rationale: the label is a measurement. Its error bounds the model's attainable performance.
Branch: if positive predictive value is low relative to the decision cost, redefine the phenotype or the claim.
Edge case: the reference standard may itself be imperfect. Report the adjudication process.

**Step 12. Check feature availability at the prediction time.**
Input: the feature list and the timestamp columns.
Output: available, boundary, and leaking features.
Rationale: the EHR contains data recorded after the decision. Using it produces a model that cannot run.
Branch: if a feature is recorded after the prediction time, exclude it or move the prediction time.
Edge case: a result entered late but collected early is still late for a real-time system unless the interface exposes collection time.

**Step 13. Handle competing risks and censoring.**
Input: the outcome and the follow-up structure.
Output: an analysis that distinguishes the event, death, and administrative end of follow-up.
Rationale: a patient who dies cannot be readmitted. Counting death as event-free inflates survival and biases the model.
Branch: if death is common, use a competing-risk framework or a composite outcome with a clear definition.
Edge case: discharge to hospice or a skilled nursing facility changes the readmission risk and the meaning of the outcome.

**Step 14. Build the denominator and person-time.**
Input: the eligible cohort and the follow-up windows.
Output: the denominator, the person-time, and the event count.
Rationale: rates need time at risk. A count without a denominator is not a rate.
Branch: if the follow-up window is truncated by the data cutoff, exclude or censor explicitly.
Edge case: a patient with two eligible admissions contributes two rows unless the unit is the patient.

### Phase D — Design the evaluation

**Step 15. Split by patient, site, and time.**
Input: the unit and the identifiers.
Output: a written split specification with group keys and a hash.
Rationale: a random row split leaks patients across sides and inflates performance.
Branch: if a site identifier exists, hold out sites; if a time field exists, hold out the most recent period.
Edge case: the same patient can have different identifiers across sites. Document the linkage method and its error.

**Step 16. Choose the metric and the calibration plan.**
Input: the decision costs and the capacity.
Output: a primary metric, a secondary metric, and a calibration assessment.
Rationale: discrimination without calibration is insufficient for a threshold-based decision.
Branch: if capacity is fixed, use precision or recall at k; if costs are asymmetric, use net benefit.
Edge case: a metric required by a regulator can be reported alongside, not instead of, the decision metric.

**Step 17. Score the gates and record the evidence.**
Input: the outputs of steps 1 to 16.
Output: a gate table with evidence and residual uncertainty.
Rationale: the decision must be reviewable and reproducible.
Branch: a zero on any gate is a veto for the claim as written.
Edge case: a source can be admissible for a methods claim and inadmissible for a population claim.

**Step 18. Re-run on version, coding, or coverage change.**
Input: the release notes, the coding updates, and the coverage changes.
Output: an updated gate record with a new date.
Rationale: EHR and claims data change with every release, coding update, and network change.
Branch: re-run when the dataset version, the coding system, the site mix, or the outcome definition changes.
Edge case: a vendor can change the underlying network without changing the product name.

### Unit-of-analysis decision table

| Claim | Unit | Required deduplication | Common error |
|---|---|---|---|
| Patient-level risk | Patient | One row per patient, index event chosen | Counting admissions as patients |
| Readmission | Index admission | One row per eligible discharge | Including transfers as readmissions |
| ICU mortality | ICU stay | One row per stay, hospital death linked | Counting unit transfers as new stays |
| Cost per episode | Claim or episode | Episode grouper across claims | Summing claim lines without a header |
| Utilisation rate | Person-time | Enrolment months as denominator | Dividing by the number of people |
| Treatment pattern | Claim line | Drug or procedure identity | Counting refills as new starts |
| Deterioration | Stay-hour | Hourly observations | Using stay-level aggregates |
| Incidence | Person-time | New cases after washout | Counting prevalent cases as incident |

### Source-selection decision table

| Question | First source | Why | Limit |
|---|---|---|---|
| Did the patient return to any hospital? | Claims | Cross-hospital billing | No severity or physiology |
| Is the patient deteriorating now? | EHR or ICU | Local physiology and orders | Only inside the institution |
| What is the national prevalence? | Survey or registry | Population sampling frame | Self-report or condition-specific |
| What is the ICU physiological trajectory? | ICU database | High-frequency measurements | Selected, single or voluntary sites |
| What is the cost of care? | Claims | Payment and utilisation | Charge, not true cost |
| What is the effect of a drug? | EHR plus claims, or trial | Prescribing and outcomes | Confounding by indication |
| What is the patient experience? | Survey | Patient-reported measures | Recall and response bias |
| Does the pipeline work? | Synthetic | No privacy constraint | Not real distributions |

### Missingness decision table

| Pattern | Mechanism | Consequence | Action |
|---|---|---|---|
| Lab not ordered | Clinician judgement | Missingness carries prognosis | Model the missingness or include an order indicator |
| Variable only in ICU | Design | Ward patients have no value | Restrict the cohort or use a setting-specific model |
| Site does not report | Infrastructure | Site effects confounded with missingness | Report by site; exclude sites below a threshold |
| Value below detection | Assay | Censored measurement | Use a censored-data method |
| Field added in a later year | System change | Era effect | Restrict or model the era |
| Patient left the network | Coverage | Outcome unobserved | Linkage or censor explicitly |

### Edge cases that change the procedure

**Licence forbids commercial use.** MIMIC-IV and eICU-CRD are research resources under credentialed agreements. A commercial product needs separate terms. For a commercial claim, score Gate 4 as 0 and keep the dataset for methods or internal research.

**Dataset withdrawn.** If access is withdrawn, stop new work, confirm the deletion requirement, and re-run the source selection. A local copy obtained before withdrawal is not a licence to continue.

**Credentialing takes months.** PhysioNet credentialing is typically reviewed in one to two weeks, but a new project and a new institution can take longer. If the timeline is shorter than the access path, the source fails Gate 3 for that decision.

**Version drift.** MIMIC-IV v3.1 is a bug-fix release over v3.0, but it changed `itemid` values in laboratory tables and removed two `subject_id` values. A pipeline pinned to v3.0 can silently change its cohort under v3.1. Pin the version and re-run the cohort count.

**Label errors discovered.** If the phenotype's positive predictive value is low, the outcome ceiling falls. Re-estimate the ceiling, consider a probabilistic label, and report a sensitivity analysis. Do not relabel only the test set.

**Benchmark contamination.** For EHR benchmarks, contamination often takes the form of patient overlap or temporal leakage rather than pretraining. Check identifiers and dates before trusting a public split.

<!-- APPEND-6 -->
