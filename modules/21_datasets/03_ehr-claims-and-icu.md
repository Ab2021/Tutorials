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

### The selection ladder

Selection happens in layers. Each layer narrows the population, and the layers compound.

| Layer | What is selected | Example | Effect on a population claim |
|---|---|---|---|
| Geography | Catchment area | One city or region | Limits national inference |
| Institution | Hospital type | Tertiary academic center | Over-represents complex cases |
| Setting | Care location | ICU only | Excludes ward and community |
| Insurance | Payer coverage | Commercial or Medicare | Excludes uninsured and other payers |
| Participation | Willingness to contribute | Voluntary network | Non-random hospital set |
| Volunteerism | Willingness to enrol | Research cohort | Healthier and more engaged |
| Era | Calendar period | 2014-2015 | Limits current practice |
| Coding | Documentation practice | One coding culture | Measurement differs from elsewhere |

The layers are not independent. A voluntary network of tertiary centers in one region is selected four times before the first patient is enrolled. Naming the layers makes the population-fit argument concrete.

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

### 4.4 Data quality and linkage terms

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Completeness | The degree to which required fields are present. | Measured as row count | Proportion |
| Accuracy | Agreement between a value and the reference. | Assumed from completeness | Proportion |
| Timeliness | Delay between an event and its availability. | Ignored in offline analysis | Hours |
| Consistency | Agreement of coding across sites and times. | Assumed from a common dictionary | Kappa or distribution |
| Granularity | The finest unit available. | Confused with sample size | Level |
| Provenance | The origin and processing history of a field. | Omitted from the data dictionary | Lineage record |
| Linkage | Joining records across sources at the person level. | Assumed to be exact | Match rate |
| Deterministic linkage | Matching on an exact identifier. | Used when identifiers are absent | Match rule |
| Probabilistic linkage | Matching on a combination of fields with weights. | Treated as certain | Match probability |
| False match rate | The share of linked pairs that are wrong. | Not measured | Proportion |
| Enrolment spine | The table of coverage periods used as the denominator. | Replaced by claim dates | Person-month |
| Attribution | Assigning a member to a provider or programme. | Assumed to be stable | Rule |

### 4.5 Resource terms

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

### Resource profiles

The table is the orientation map for this topic. The last column is the one to read first.

| Resource | Custodian | Version inspected | Unit | Scale | Access | What it cannot answer |
|---|---|---|---|---|---|---|
| MIMIC-IV | Beth Israel Deaconess and MIT | v3.1, October 2024 | Patient, hospitalization, ICU stay | 364,627 patients; 546,028 hospitalizations; 94,458 ICU stays | PhysioNet credentialed | National incidence, community-hospital performance, out-of-network outcomes |
| MIMIC-IV-ED | Same | Current release | ED visit | Hundreds of thousands of visits | PhysioNet credentialed | Post-ED outcomes without linkage to the hospital module |
| MIMIC-IV-Note | Same | Current release | Note | Linked to MIMIC-IV | PhysioNet credentialed | Complete clinical truth; de-identification removes tokens |
| eICU-CRD | Philips and MIT | v2.0, 2019 | ICU unit stay | 208 hospitals; 335 units; 200,859 unit stays; 139,367 patients; 2014-2015 | PhysioNet credentialed | Ward care, notes, imaging, post-discharge outcomes |
| HiRID | Bern University Hospital | Current release | ICU stay | Tens of thousands of stays | Credentialed, Swiss governance | Multi-site transport; US practice |
| AmsterdamUMCdb | Amsterdam UMC | Current release | ICU admission | About 23,000 admissions | Credentialed | US coding and practice; national estimates |
| UK Biobank | UK Biobank | Current release | Participant | 500,000 participants | Application, fee, UKB-RAP | General-population prevalence; causal effects without design |
| All of Us | NIH | CDR v9, June 2026 | Participant | More than 747,000 participants; more than 535,000 with short-read WGS; EHR for more than 481,000 | Registered and controlled tiers | Unbiased national estimates; uniform EHR linkage |
| Epic Cosmos | Epic | 2026 vendor release | Query-level cohort | About 2,358 hospitals and 310 million patients (vendor figure) | Contributor-only | Row-level modeling; reproducible splits |
| EHRSHOT | Stanford | Current release | Patient | 6,739 patients; 15 tasks | PhysioNet plus Stanford agreement | Product performance; multi-site transport |
| Synthea | MITRE | Current release | Synthetic patient | Configurable | Open, Apache 2.0 | Real distributions; clinical performance |
| CMS SynPUF | CMS | 2008-2010 sample | Beneficiary | About 2.3 million beneficiaries (verify) | Open | Current coding; real clinical detail |
| OMOP CDM and Eunomia | OHDSI | Current version | Varies by source | Eunomia is a small synthetic sample | Open tooling | Source data quality; clinical truth |

The scale column mixes units on purpose. A patient count, a hospitalization count, and a unit-stay count are not comparable, and a table that places them side by side without labels invites the wrong comparison.

### MIMIC-IV module map

| Module | Grain | Core content | Typical use | Main limit |
|---|---|---|---|---|
| `hosp` | Hospitalization | Admissions, labs, microbiology, prescriptions, diagnoses, DRGs | Cohort construction, outcomes, treatment | No ICU-only physiology |
| `icu` | ICU stay | Charted vitals, inputs, outputs, procedures, ventilation | Deterioration, organ support | Only ICU time |
| `ed` | ED visit | Triage, vitals, medications, dispositions | Boarding, triage, ED operations | Post-ED outcomes need linkage |
| `note` | Note | Discharge summaries, radiology reports | NLP, summarisation | De-identification loss |
| `cxr` | Image study | Chest radiographs and reports | Imaging models | Separate credentialing |
| `waveform` | Waveform | High-frequency signals for a subset | Signal modeling | Device and alignment issues |
| `ecg` | ECG record | 12-lead ECGs for a subset | Arrhythmia and morphology | Subset selection |
| Demo | Patient | 100-patient subset | Teaching and pipeline tests | Not for scientific claims |

### Common cohort definitions and their traps

| Cohort | Typical definition | Trap | Check |
|---|---|---|---|
| Sepsis | Suspected infection plus organ dysfunction | Culture and antibiotic timing | Verify the window and the antibiotic list |
| AKI | Creatinine rise or urine output fall | Baseline creatinine unknown | Define the baseline and test sensitivity |
| Heart failure | ICD codes plus natriuretic peptide | Code identifies documented disease | Validate positive predictive value |
| Readmission | Any inpatient admission within 30 days | Transfers and observation stays | Exclude planned readmissions per the measure |
| ICU mortality | Death before ICU discharge | Hospital death after transfer | Link to hospital discharge status |
| Ventilation | Procedure codes or charted settings | Non-invasive ventilation mixed in | Separate invasive and non-invasive |
| Delirium | CAM-ICU or ICD codes | Charted assessment is intermittent | Report assessment coverage |
| Comorbidity | Elixhauser or Charlson from codes | Coding depth varies by site | Compare code counts by site |

### Coding transitions and their effects

| Transition | Date | Effect on measurement | Handling |
|---|---|---|---|
| ICD-9-CM to ICD-10-CM/PCS | 1 October 2015 in the US | More codes, different granularity, measured prevalence shifts | Restrict the era or validate a crosswalk |
| Present-on-admission flag adoption | Varies by state and year | Hospital-acquired conditions become distinguishable | Check flag availability before use |
| Sepsis-2 to Sepsis-3 | 2016 publication, gradual adoption | Case definitions and code use change | Use a computable definition and report the version |
| COVID-19 coding and care changes | 2020-2021 | Utilisation, coding, and outcomes shift | Model the period or restrict the cohort |
| Telehealth expansion | 2020 onward | Encounter mix and follow-up change | Check modality fields |
| Observation stay rules | Payer and year specific | Inpatient counts change without clinical change | Use an encounter-level definition |

A model trained across a transition can learn the transition. The safest design restricts the cohort to one coding era and states the restriction. The next safest stratifies by era and reports the difference.

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

### Claims audit checklist

| Check | Question | Evidence |
|---|---|---|
| Enrolment | Is the patient continuously enrolled for the lookback and follow-up? | Eligibility file with start and end dates |
| Setting | Are inpatient, outpatient, and ED claims distinguishable? | Place-of-service and claim type |
| Cross-hospital events | Are claims from other institutions included? | Network and payer coverage |
| Out-of-network | Are out-of-network claims captured? | Network file and claim status |
| Observation stays | Are observation stays separated from admissions? | Revenue codes and claim type |
| Planned readmissions | Are planned returns flagged? | Measure specification or procedure codes |
| Diagnosis timing | Is the diagnosis present on admission? | Present-on-admission flag where available |
| Coding depth | How many codes per encounter? | Counts by year and site |
| Drug capture | Are fills, doses, and days supplied present? | Pharmacy claim fields |
| Death | Is death captured from a registry or a claim? | Linkage and date |
| Denominator | Is the unit a person or a person-month? | Enrolment months |
| Payment versus cost | Is the field payment, charge, or allowed amount? | Data dictionary |

### EHR audit checklist

| Check | Question | Evidence |
|---|---|---|
| Cohort | Who is in the source, and who is excluded? | Inclusion and exclusion counts |
| Setting | Are ED, ward, ICU, and ambulatory separable? | Encounter type |
| Timing | Are order, collection, result, and chart times present? | Timestamp audit |
| Missingness | Which variables are ordered rather than routine? | Order frequency by site |
| Notes | Are notes available, and what was removed? | De-identification report |
| Coding | Which diagnosis and procedure systems are used? | Code system and version |
| Linkage | Can encounters be linked at the person level? | Identifier map |
| Outside care | Can care outside the institution be observed? | Health information exchange or none |
| Era | What years are covered, and what changed? | Release notes |
| Governance | What does the agreement permit? | Clause matrix |

### Cohort-size check

The check is a planning aid, not a power calculation. It asks whether the available cohort can support the intended number of predictors before modeling begins.

| Stratum | Eligible admissions | Outcome rate | Events | Events per 25 predictors |
|---|---:|---:|---:|---:|
| All admissions | 40,000 | 22% | 8,800 | 352 |
| Age 65 and older | 18,000 | 24% | 4,320 | 173 |
| Medicare | 15,000 | 25% | 3,750 | 150 |
| Commercial | 12,000 | 14% | 1,680 | 67 |
| Rural site | 2,000 | 12% | 240 | 10 |
| Smallest site | 600 | 10% | 60 | 2 |

The smallest site has 60 events for 25 predictors, which is below the planning threshold. A site-level metric on that site is uninformative, and the site should be pooled or reported descriptively rather than ranked.

A cohort-size check has three possible conclusions: the stratum supports the model, the stratum supports a descriptive report only, or the stratum is too small to report. Write the conclusion next to the table. A table without a conclusion invites a later analyst to compute the metric because the rows exist.

### Resource-specific traps

| Resource | Trap | Why it happens | Check |
|---|---|---|---|
| MIMIC-IV | Treating the hosp module as a patient table | The grain is hospitalization | Count distinct `subject_id` |
| MIMIC-IV | Using ICU variables for ward patients | ICU tables cover ICU time only | Restrict to ICU stays or use hosp tables |
| MIMIC-IV | Comparing dates with real calendar dates | Dates are shifted into the future | Use intervals, not calendar dates |
| MIMIC-IV | Assuming complete post-discharge follow-up | Only in-network events are visible | Compare with a claims benchmark |
| eICU-CRD | Pooling all hospitals without coverage checks | Variable coverage varies by site | Report coverage by hospital |
| eICU-CRD | Treating unit stays as patients | A patient can have several unit stays | Deduplicate by patient |
| eICU-CRD | Assuming a random hospital sample | Participation is voluntary | State the sampling limitation |
| HiRID | Applying US units and coding | Swiss practice and units differ | Convert and validate units |
| AmsterdamUMCdb | Applying US coding | Dutch coding and practice | Map concepts explicitly |
| UK Biobank | Treating participants as the population | Healthy-volunteer selection | Compare with a national survey |
| All of Us | Treating the cohort as nationally representative | Volunteer cohort and heterogeneous linkage | Weight or state the estimand |
| All of Us | Publishing small subgroup counts | Disclosure rules | Apply the count threshold |
| Epic Cosmos | Requesting row-level export | Query-level product | Use for feasibility only |
| EHRSHOT | Treating few-shot gains as clinical utility | Benchmark tasks are code-derived | Pair with calibration and workflow evidence |
| Synthea | Validating clinical performance | Synthetic distribution is designed | Use for pipelines only |
| CMS SynPUF | Analysing current utilisation | Synthetic and dated | Use for SQL and pipeline teaching |

### Query patterns that prevent unit errors

| Pattern | SQL shape | Purpose |
|---|---|---|
| Distinct patients | `COUNT(DISTINCT subject_id)` | Person-level denominator |
| Admissions per patient | `COUNT(*) GROUP BY subject_id` | Repeat-admission factor |
| First eligible admission | Window function ordered by admission time | One row per patient |
| Last discharge before index | Correlated subquery or window | Baseline definition |
| 30-day window | Join on date range | Outcome ascertainment |
| Continuous enrolment | Enrolment months between dates | Claims denominator |
| Exclusion of transfers | Join to transfer table | Readmission definition |
| Site coverage | `GROUP BY site_id` with counts | Transportability table |
| Coding depth | `AVG(code_count)` by year | Era audit |
| Missingness | `SUM(is_null)/COUNT(*)` by site | Informative missingness check |

The patterns are simple. The discipline is to run them before modeling and to write the result into the evidence ledger.

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

## 6. Worked example

### The case

A Medicare Advantage plan with 420,000 members wants to reduce 30-day heart failure readmissions. It has a transitional care team that can make 40 outreach calls per week. The plan wants a model that ranks patients at discharge so the team calls the highest-risk patients first.

**Decision statement.** "At hospital discharge for a heart failure admission, rank patients for a transitional care call within 7 days; the team can call 40 patients per week, a false alert costs about 20 minutes of nurse time, and a missed readmission costs the plan roughly $15,000 in avoidable care and penalties."

**Unit.** Index admission, with one row per eligible discharge and a 30-day follow-up window. A patient with two eligible discharges contributes two rows unless the programme decides to call once per patient.

**Setting.** Acute care hospitals in the plan's network, including community hospitals. The plan does not own the hospitals, so the data must come from claims or from a data-sharing agreement.

### Candidate long list

| Candidate | Custodian | Version inspected | Unit | Access tier | Why it is on the list |
|---|---|---|---|---|---|
| National commercial claims | Licensed vendor | Current contract year | Claim and enrolment month | Licensed | Cross-hospital readmission and cost |
| MIMIC-IV | Beth Israel Deaconess and MIT | v3.1 | Hospitalization | Credentialed | Clinical depth and a public benchmark |
| eICU-CRD | Philips and MIT | v2.0 | ICU unit stay | Credentialed | Multi-site ICU physiology |
| All of Us | NIH | CDR v9 | Participant | Registered and controlled | Diverse cohort with EHR and genomics |
| Epic Cosmos | Epic | 2026 vendor release | Query-level cohort | Contributor-only | Large network and feasibility counts |

### Gate scores for the commercial claim

| Gate | Claims | MIMIC-IV | eICU-CRD | All of Us | Cosmos |
|---|---:|---:|---:|---:|---:|
| 1 Decision fit | 2 | 1 | 0 | 1 | 2 |
| 2 Population fit | 2 | 0 | 0 | 2 | 2 |
| 3 Access fit | 1 | 2 | 2 | 1 | 0 |
| 4 Licence fit | 1 | 0 | 0 | 0 | 1 |
| 5 Benchmark fit | 1 | 1 | 1 | 1 | 1 |
| Veto | none | population, licence | decision, population | licence | access |

**Claims, Gate 1.** Readmission is a billing event. The claim observes admissions to any in-network hospital and carries the enrolment months needed for a denominator. It does not carry ejection fraction, natriuretic peptide, or renal function, so the model cannot separate compensated from decompensated heart failure. Score 2 for the utilisation outcome, with a clinical-severity limitation recorded.

**Claims, Gate 2.** The plan's members are the target population, and the network covers community and academic hospitals. The coverage table must show enrolment duration and out-of-network leakage. Score 2 with a documented leakage estimate.

**Claims, Gate 3.** A commercial licence is required. The vendor's contract is in place for another use, but the new use needs a field-of-use confirmation and a security review. Score 1 until the amendment is signed.

**Claims, Gate 4.** The contract permits commercial analytics and prohibits re-release of row-level data. Model release is restricted. For an internal care-management model, the use is permitted; for a product sold to other payers, it is not. Score 1 for the internal claim, 0 for a product claim.

**MIMIC-IV, Gate 1 and Gate 2.** The dataset observes readmissions only to Beth Israel Deaconess. A patient readmitted elsewhere appears as event-free. The cohort is a single tertiary academic center, not a Medicare Advantage network. Scores 1 and 0, and the population zero is a veto.

**MIMIC-IV, Gate 4.** The PhysioNet DUA is a research agreement. A commercial care-management product is outside it. Score 0 for the commercial claim.

**eICU-CRD, Gate 1.** The data end at ICU discharge and do not observe 30-day readmission. The outcome is unobservable. Score 0, veto.

**All of Us, Gate 4.** Access is limited to academic, health care, not-for-profit, and government institutions, and the Data User Code of Conduct restricts commercial use. A payer is not an eligible institution in the required sense, and the commercial claim fails. Score 0.

**Epic Cosmos, Gate 3.** Contributor-only and query-level. The plan is not a contributor, and the product does not return row-level records for model training. Score 0, veto.

### The two diagnostics

**Diagnostic 1: outcome capture.** In this planning example, the MIMIC-IV cohort shows a 30-day readmission rate of about 11 percent, while the plan's own claims show about 21 percent for a comparable age band. The gap is consistent with readmissions outside the single center being invisible. The diagnostic is not the difference itself; it is the mechanism. A single-center EHR cannot observe a cross-hospital event, so a negative label is unverified. The same logic applies to post-discharge death, which can appear as a censored observation rather than an event.

**Diagnostic 2: clinical severity.** A chart audit of 120 admissions that the claims-only model ranked in the top decile found that about 38 percent had no documentation of decompensated heart failure during the index admission. The model had learned prior utilisation and chronic codes, not acute severity. This is not a model bug. The claims data do not contain the variables needed to distinguish a stable chronic patient from an acutely decompensated one. The response is to add a clinical data source for the highest-risk patients, or to change the outcome to one the claims can measure, such as a 7-day ED revisit.

A third diagnostic, applied to the benchmark, is the split. The vendor's extract has no patient identifier across the full history in the standard analysis file, so the team must build a person key from the enrolment file. If the key is wrong, a patient can appear on both sides of the split. The team writes the split specification before looking at any model result.

### The selected route

The admissible set is claims only. The plan uses the claims data for development and a held-out time period for evaluation, and it plans a clinical augmentation for the highest-risk tier.

| Track | Steps | Owner | Duration | Planning cost |
|---|---|---|---|---|
| Contract amendment | Field-of-use confirmation, security review, signature | Legal and procurement | 8 to 12 weeks | Legal time plus any fee |
| Data build | Enrolment spine, claim extraction, cohort definition | Data engineering | 6 weeks | Staff time |
| Phenotype validation | Chart audit of 120 admissions | Clinical lead | 3 weeks | About $6,000 |
| Model development | Baseline and strong model, calibration | Data science | 10 weeks | Compute and staff |
| Evaluation | Held-out time period, subgroup and calibration | Data science | 4 weeks | Staff time |
| Clinical augmentation | EHR extract for the top tier | Clinical lead | 3 to 6 months | Partner agreement |

### Time and cost

| Item | Detail | Calendar time | Planning cost |
|---|---|---|---|
| Licence amendment | Field of use and security review | 8 to 12 weeks | Legal, 80 hours |
| Data engineering | Enrolment spine and claim tables | 6 weeks | 0.5 FTE |
| Chart audit | 120 admissions | 3 weeks part-time | About $6,000 |
| Model development | Two baselines and calibration | 10 weeks | Compute, about $5,000 |
| Evaluation | Temporal holdout and subgroups | 4 weeks | 0.25 FTE |
| Total to a decision | Excludes the clinical augmentation | About 20 weeks | About $90,000 internal |

The licence amendment is on the critical path. The data engineering can start once the amendment is drafted, but the model cannot be trained on the full extract until it is signed.

### Substitute plan if the licence amendment fails

If the vendor refuses the care-management use, the plan changes in three ways:

1. The claim narrows to in-hospital mortality or a 7-day ED revisit observed at a partner hospital, which is not the same decision and has a different value.
2. The data source becomes MIMIC-IV plus a partner hospital, which reintroduces the single-center problem and requires a prospective validation.
3. The programme launches with a rules-based risk score and uses the model project as a research study, with no product claim.

The fallback is worse for the business case because the outcome that matters, 30-day readmission across hospitals, is exactly the outcome the alternative sources cannot observe.

### Vary one assumption: the plan wants to sell the model to other payers

Suppose the plan's strategy changes from internal care management to a commercial product sold to other payers. Gate 4 changes from 1 to 0 for the product claim, because the vendor contract prohibits re-release and the model would be a product artefact. The plan has four options:

- Negotiate a commercial redistribution and model-release right, usually at a higher fee.
- Build the product on data the plan owns, such as its own claims, and license the model rather than the data.
- Partner with a provider organisation that can supply data under a commercial agreement.
- Keep the model internal and sell the care-management service rather than the model.

The scientific work does not change. The permission and the product architecture do. This is the normal pattern in payer data: the analytics are feasible, and the commercial route determines whether they become a product.

If the variation is size, a plan with 40,000 members rather than 420,000 would have about 2,000 heart failure discharges per year and 440 readmissions at 22 percent, which is enough for a 25-predictor model but thin for subgroup analysis. If the variation is label quality, a chart audit showing that 30 percent of coded heart failure admissions are not acute decompensation would force a narrower cohort definition and a lower outcome ceiling.

### Gate evidence log

| Gate | Candidate | Evidence inspected | Verdict | Residual uncertainty |
|---|---|---|---|---|
| 1 | Claims | Claim type, place of service, enrolment months | 2 | Out-of-network claims may be incomplete |
| 1 | MIMIC-IV | Discharge and readmission fields; single-center catchment | 1 | Readmissions outside the center are invisible |
| 1 | eICU-CRD | Data end at ICU discharge | 0 | Post-discharge outcome unobservable |
| 2 | Claims | Member coverage and network hospitals | 2 | Network leakage by geography |
| 2 | MIMIC-IV | Single tertiary academic center | 0 | No Medicare Advantage population |
| 3 | Claims | Existing contract, amendment required | 1 | Field-of-use approval time |
| 3 | MIMIC-IV | PhysioNet credentialing | 2 | None for this team |
| 4 | Claims | Redistribution and model-release clauses | 1 | Internal use permitted; product use not |
| 4 | MIMIC-IV | PhysioNet research DUA | 0 | No commercial care-management rights |
| 4 | All of Us | Institutional eligibility and DUCC | 0 | Commercial use restricted |
| 5 | Claims | No frozen split; person key must be built | 1 | Split is ours; comparability limited |
| 5 | MIMIC-IV | Public cohort definitions; no frozen split | 1 | Benchmark reuse across papers |

### Decision timeline

| Month | Contract track | Data track | Clinical track | Deliverable |
|---:|---|---|---|---|
| 1 | Field-of-use request | Enrolment spine design | Chart-audit sampling plan | Request submitted |
| 2 | Security review | Claim extraction | Abstractor training | Draft extract |
| 3 | Amendment signed | Cohort build | Chart audit | Signed amendment |
| 4-5 | Confirmation of renewal | Feature build | Audit results | Analysis dataset v1 |
| 6-7 | Model-release review | Split and baseline | Clinical review of features | Baseline results |
| 8-9 | Product-use review | Strong model and calibration | Workflow review | Internal model v1 |
| 10 | Final approval | Temporal holdout | Programme design | Decision memo |

### What the case demonstrates

- The outcome, not the physiology, determined the source. A readmission model needed claims because readmission is a cross-hospital billing event.
- The clinical dataset was richer and inadmissible. Depth does not compensate for a population veto or a licence veto.
- The claims data had a severity blind spot that no model could fix. The response was a clinical augmentation for the top tier, not a more complex model on the same variables.
- The licence amendment, not the modeling, set the schedule. The same pattern appeared in Topic 02 and will appear again.
- The benchmark was built by the team because no frozen split existed. That is normal, and it means external comparability is limited by construction.
- The fallback changed the outcome, not only the dataset. A fallback that keeps the same outcome is rare in this category.

### What the plan still cannot answer

Three questions remain open after the source selection, and each has a named owner.

1. **Is the readmission preventable?** The claims model ranks risk, not avoidability. A high-risk patient with a planned staged procedure is not a candidate for a transitional care call. The clinical team must define an exclusion list, and the model's target becomes "avoidable readmission" only if that label can be measured.
2. **Does the call change the outcome?** The model predicts readmission; the programme assumes the call reduces it. That is a causal claim, and it requires a design with a comparison group, not a retrospective ranking. The evaluation plan should include a stepped-wedge or a matched comparison, or the programme should be labelled a pilot with an unknown effect.
3. **Does the model transport to the next contract year?** The network, the benefit design, and the coding can change. The evaluation should include a temporal holdout and a re-baseline trigger for the next contract year.

### Conditions under which the case would fail

- **The licence amendment is refused.** The claim narrows to an internal research study, or the plan uses its own claims only and accepts a smaller cohort.
- **The chart audit finds a low predictive value.** The cohort definition narrows, the outcome rate changes, and the required sample size rises.
- **The network leakage is large.** If a third of readmissions occur outside the network, the outcome is not measurable and the project needs a different data source.
- **The programme cannot act on 40 calls per week.** The capacity constraint changes the metric and the threshold; a model ranked for a different capacity is mis-specified.
- **The temporal holdout shows a large drop.** The model is not stable across years, and the deployment plan needs a monitoring and recalibration protocol before launch.

### Sensitivity of the decision

| Change | Gate affected | Does the source change? | Plan response |
|---|---|---|---|
| Outcome becomes 7-day ED revisit | 1 | Possibly, if the ED is out of network | Re-audit coverage and the denominator |
| Population narrows to dual-eligible members | 2 | No, but the subgroup is smaller | Recompute the cohort-size check |
| Contract permits product use | 4 | No | Add model-release and redistribution review |
| Data cutoff moves one quarter earlier | 1 and 5 | No | Re-run censoring and the temporal split |
| A second payer joins the programme | 2 and 3 | Yes, if a second claims source is needed | Harmonise definitions and test transport |
| Chart audit finds 30 percent false positives | 1 | No | Narrow the cohort and lower the ceiling |

## 7. Code, query, or calculation

The three blocks below use the Python standard library only. They implement the unit-of-analysis audit, the follow-up and censoring check, and the cohort-size planning check used above.

### 7.1 Unit-of-analysis audit

```python
COUNTS = {
    "patients (all)": 364627,
    "hospitalizations": 546028,
    "ICU stays": 94458,
    "unique ICU patients": 65366,
    "ED-only patients": 141175,
}

print("MIMIC-IV v3.1 unit-of-analysis audit")
print(f"{'Entity':<24}{'Count':>10}")
for name, n in COUNTS.items():
    print(f"{name:<24}{n:>10,}")

hospitalized = COUNTS["patients (all)"] - COUNTS["ED-only patients"]
print()
print(f"Hospitalized patients (all minus ED-only): {hospitalized:,}")
print(f"Hospitalizations per hospitalized patient: {COUNTS['hospitalizations']/hospitalized:.2f}")
print(f"ICU stays per hospitalization: {COUNTS['ICU stays']/COUNTS['hospitalizations']:.3f}")
print(f"ICU stays per ICU patient: {COUNTS['ICU stays']/COUNTS['unique ICU patients']:.2f}")
print()
print("Interpretation:")
print("  A row in the hosp module is a hospitalization, not a patient.")
print("  Counting rows as patients overstates the cohort by the repeat-admission factor.")
print("  Only about one in six hospitalizations has an ICU stay.")
```

Expected output:

```text
MIMIC-IV v3.1 unit-of-analysis audit
Entity                       Count
patients (all)             364,627
hospitalizations           546,028
ICU stays                   94,458
unique ICU patients         65,366
ED-only patients           141,175

Hospitalized patients (all minus ED-only): 223,452
Hospitalizations per hospitalized patient: 2.44
ICU stays per hospitalization: 0.173
ICU stays per ICU patient: 1.45

Interpretation:
  A row in the hosp module is a hospitalization, not a patient.
  Counting rows as patients overstates the cohort by the repeat-admission factor.
  Only about one in six hospitalizations has an ICU stay.
```

The arithmetic is simple and the consequence is not. A "patient cohort" built by selecting rows from the hospitalization table has 546,028 rows and 223,452 people. A per-patient rate computed on the rows is a per-hospitalization rate with a different denominator.

### 7.2 Follow-up and censoring check

```python
from datetime import date, timedelta

DATA_CUTOFF = date(2020, 12, 31)
FOLLOWUP_DAYS = 30

admissions = [
    {"id": "A1", "discharge": date(2020, 1, 10), "next_admit": date(2020, 1, 25), "died": None},
    {"id": "A2", "discharge": date(2020, 2, 1), "next_admit": None, "died": None},
    {"id": "A3", "discharge": date(2020, 11, 20), "next_admit": date(2020, 12, 15), "died": None},
    {"id": "A4", "discharge": date(2020, 12, 20), "next_admit": None, "died": None},
    {"id": "A5", "discharge": date(2020, 3, 5), "next_admit": None, "died": date(2020, 3, 20)},
]

print(f"{'ID':<4}{'Discharge':>12}{'Next':>12}{'Readmit30':>11}{'Observed':>10}  Note")
for a in admissions:
    observed = (a["discharge"] + timedelta(days=FOLLOWUP_DAYS)) <= DATA_CUTOFF
    if a["next_admit"] is not None:
        gap = (a["next_admit"] - a["discharge"]).days
        readmit = gap <= FOLLOWUP_DAYS
    else:
        readmit = False
    note = ""
    if a["died"] is not None:
        note = "death is a competing event"
    elif not observed:
        note = "administratively censored"
    print(f"{a['id']:<4}{a['discharge'].isoformat():>12}"
          f"{(a['next_admit'].isoformat() if a['next_admit'] else '-'):>12}"
          f"{str(readmit):>11}{str(observed):>10}  {note}")

eligible = [a for a in admissions
            if (a["discharge"] + timedelta(days=FOLLOWUP_DAYS)) <= DATA_CUTOFF
            and a["died"] is None]
print()
print(f"Admissions listed: {len(admissions)}")
print(f"Eligible with complete 30-day follow-up and alive at discharge: {len(eligible)}")
print("Rows without complete follow-up cannot contribute a negative label.")
```

Expected output:

```text
ID     Discharge        Next  Readmit30  Observed  Note
A1    2020-01-10  2020-01-25       True      True
A2    2020-02-01           -      False      True
A3    2020-11-20  2020-12-15       True      True
A4    2020-12-20           -      False     False  administratively censored
A5    2020-03-05           -      False      True  death is a competing event

Admissions listed: 5
Eligible with complete 30-day follow-up and alive at discharge: 3
Rows without complete follow-up cannot contribute a negative label.
```

The check catches two common errors. The censored row would otherwise be labelled a non-readmission, and the death would be treated as event-free follow-up. Both inflate the apparent performance of a model that predicts a low-risk outcome.

### 7.3 Cohort-size planning check

```python
def required_admissions(prevalence, predictors=25, epv=10, inflation=1.0):
    events = predictors * epv
    n = events / prevalence
    return round(n * inflation)

print(f"{'Stratum':<30}{'Prevalence':>12}{'Required n':>12}")
rows = [
    ("All admissions", 0.22),
    ("Age 65 and older", 0.24),
    ("Age under 65", 0.17),
    ("Medicare", 0.25),
    ("Commercial", 0.14),
    ("Rural site", 0.12),
]
for name, prev in rows:
    print(f"{name:<30}{prev:>12.0%}{required_admissions(prev):>12,}")
print()
print("Assumptions: 25 predictors, 10 events per predictor, no design inflation.")
print("Add 20 to 40 percent for subgroup analyses and missing data.")
```

Expected output:

```text
Stratum                         Prevalence  Required n
All admissions                         22%       1,136
Age 65 and older                       24%       1,042
Age under 65                           17%       1,471
Medicare                               25%       1,000
Commercial                             14%       1,786
Rural site                             12%       2,083

Assumptions: 25 predictors, 10 events per predictor, no design inflation.
Add 20 to 40 percent for subgroup analyses and missing data.
```

The check is deliberately crude. Its purpose is to stop a subgroup analysis before it starts, not to replace a power calculation. A stratum that needs 2,083 admissions and has 600 is a descriptive table, not a model.

## 8. Data, standards, and regulatory context

EHR and claims data are governed by coding systems, data models, and payment rules. The table separates the binding from the conventional, because a coding rule can change a cohort as surely as a law.

| Instrument | Status | What it constrains | What it does not do |
|---|---|---|---|
| ICD-10-CM | Binding for US morbidity coding | Diagnosis codes and their definitions | Provide clinical timing or severity |
| ICD-10-PCS | Binding for US inpatient procedure coding | Inpatient procedure codes | Cover outpatient procedures |
| CPT and HCPCS | Binding for US billing in many settings | Procedure and supply codes | Describe clinical outcomes |
| DRG | Binding for Medicare inpatient payment | Payment grouping | Measure severity for research |
| LOINC | Voluntary but widely required | Laboratory and observation identities | Guarantee comparable units or assays |
| SNOMED CT | Voluntary in the US, required in some systems | Clinical terminology | Guarantee consistent clinician use |
| OMOP CDM | Voluntary common data model | Table structure and vocabulary mapping | Preserve every source field |
| FHIR R4 | Voluntary standard, required in some programmes | Exchange resources and APIs | Guarantee data completeness |
| HIPAA | Binding | Use and disclosure of protected health information | Grant research or commercial rights |
| CMS Hospital Readmissions Reduction Program | Binding for participating hospitals | 30-day readmission measure and payment adjustment | Define a research-grade readmission label for all payers |
| CMS SEP-1 | Binding for participating hospitals | Sepsis bundle reporting | Validate a sepsis phenotype for research |
| eCQM specifications | Binding for certain quality programmes | Electronic clinical quality measure logic | Transfer to a different data model without mapping |

**Coding is not measurement.** ICD-10-CM contains many more codes than ICD-9-CM, and the transition changed measured prevalence for conditions whose clinical definition did not change. A model trained on codes learns the coding system, the documentation culture, and the payment incentive along with the disease. Whenever a cohort spans a coding change, the analysis must either restrict the era or validate a crosswalk.

**Payment rules shape the label.** The CMS 30-day readmission measure excludes planned readmissions and uses a risk-adjustment model. A research label built from raw claims without those exclusions measures a different quantity. If the claim is about a CMS programme, use the programme's specification. If the claim is about clinical risk, say so and do not borrow the programme's authority.

**Common data models trade fidelity for portability.** OMOP makes multi-site analysis possible by mapping source codes to standard concepts. The mapping can lose granularity, and a concept that is absent from the vocabulary can disappear. A multi-site study should report the mapping coverage by site and the concepts that failed to map.

### Coding system comparison

| System | Domain | Granularity | Designed for | Research caveat |
|---|---|---|---|---|
| ICD-10-CM | Diagnoses | High | Morbidity coding and payment | Documentation and incentive effects |
| ICD-10-PCS | Inpatient procedures | High | Procedure coding | Not comparable with CPT |
| CPT | Outpatient and professional procedures | Moderate | Billing | Proprietary and updated annually |
| HCPCS | Supplies, drugs, services | Variable | Billing | Codes change and retire |
| NDC | Drugs | Product-level | Pharmacy billing | Package changes over time |
| LOINC | Labs and observations | Test-level | Interoperability | Assay and unit differences |
| SNOMED CT | Clinical concepts | Very high | Clinical terminology | Clinician use varies |
| DRG | Payment groups | Low | Inpatient payment | Not a severity measure |

A crosswalk between coding systems is a research artefact. It needs a version, a validation sample, and a stated error rate. Copying a crosswalk from a paper without testing it locally is a common and quiet source of cohort drift.

### Data quality dimensions for EHR and claims

| Dimension | Question | Typical test | Consequence if poor |
|---|---|---|---|
| Completeness | Are the required fields present? | Missingness by field and site | Bias if missingness is informative |
| Accuracy | Do values match the clinical record? | Chart audit | Label error and ceiling |
| Timeliness | When is the value available? | Timestamp audit | Leakage and unusable features |
| Consistency | Do sites code the same way? | Code distribution by site | Confounded site effects |
| Granularity | Is the unit fine enough? | Grain audit | Wrong denominator |
| Coverage | Are all settings observed? | Coverage diagram | Unverified negative labels |
| Provenance | Where did the value come from? | Field-level lineage | Uninterpretable features |
| Stability | Do definitions change over time? | Year-by-year comparison | Era effect |
| Validity | Does the field measure what it claims? | Reference comparison | Mis-specified model |
| Uniqueness | Are records duplicated? | Identifier and hash checks | Inflated counts |

### Quality-measure logic versus research logic

| Element | CMS-style measure | Research label | Why the difference matters |
|---|---|---|---|
| Index event | Specified admission types | Investigator-defined | Different denominators |
| Planned readmission | Excluded by algorithm | Often not excluded | Higher measured rate |
| Risk adjustment | Specified model | Study-specific | Different comparisons |
| Population | Programme enrollees | Eligibility criteria | Different estimand |
| Time window | 30 days | Often 30 or 90 | Not comparable |
| Data source | Claims or chart | Claims, EHR, or linked | Different capture |
| Use | Payment adjustment | Inference | Different error costs |

A research team that borrows a measure's name without its logic produces a number that is neither a quality measure nor a clean research estimate.

### Data model comparison

| Model | Grain | Strength | Limit | Typical use |
|---|---|---|---|---|
| Raw EHR tables | Local schema | Full fidelity | Site-specific and brittle | Single-site analysis |
| OMOP CDM | Person, visit, condition, drug, measurement | Portability across sites | Mapping loss | Multi-site observational studies |
| FHIR | Resources and references | Exchange and APIs | Not an analytics model | Interoperability and apps |
| Claims warehouse | Claim and enrolment month | Cross-hospital utilisation | No clinical detail | Cost, readmission, utilisation |
| ICU relational database | Stay and chart time | High-frequency physiology | ICU only | Deterioration and organ support |
| Registry | Case | Condition-specific depth | Limited comparators | Incidence and outcomes |

### Regulatory artefacts that depend on these data

- **A readmission measure** needs a defined index admission, a planned-readmission exclusion, a risk-adjustment model, and a denominator of eligible discharges.
- **A sepsis measure** needs a suspected-infection definition, an organ-dysfunction definition, and a time window.
- **A device submission** needs a reference standard, a representativeness analysis, and a validation dataset that matches the intended use.
- **A payer quality programme** needs enrolment continuity, attribution of members to providers, and a consistent network definition.
- **A research publication** needs a reproducible cohort definition, a code list, and a description of the coding era.

## 9. Industry practice and case evidence

### What organisations actually do

- **Use claims for utilisation and EHR for physiology.** Mature teams do not ask one source to answer both.
- **Link sources under a governance agreement.** Linkage is a separate permission and a separate error-prone process.
- **Validate phenotypes before modeling.** A code list is a hypothesis; a chart audit is the test.
- **Report by site and year.** Pooled metrics hide the variation that determines transportability.
- **Keep a code library under version control.** Code lists change with coding updates and clinical guidance.
- **Separate the quality measure from the research label.** A CMS measure and a research outcome can share a name and differ in logic.
- **Plan for re-validation.** Coding updates, network changes, and EHR migrations recur.

### Case evidence

| Case | Documented fact | Inference |
|---|---|---|
| MIMIC-IV | The peer-reviewed descriptor documents a single-center, de-identified EHR and ICU resource from Beth Israel Deaconess, with dates shifted into the future and a credentialed access route. | The resource is strong for methods and phenotyping and not a population sample. A national claim is outside its scope. |
| eICU-CRD | The PhysioNet page documents 208 hospitals, 335 ICU units, 200,859 unit stays, and 139,367 patients from 2014-2015, collected through a telehealth programme. | Multi-site coverage improves transport testing, and voluntary participation means the hospital set is not a probability sample. |
| All of Us v9 | The June 2026 release documents more than 747,000 participants, more than 535,000 with short-read whole-genome sequencing, EHR for more than 481,000, and the first clinical notes for more than 99,000 participants in the controlled tier. | The cohort is large and diverse and still a volunteer cohort with heterogeneous EHR linkage. It supports discovery and not unbiased national prevalence. |
| Epic Cosmos | Vendor materials describe a large de-identified network available to contributing organisations, with query-level access. | The product is useful for feasibility and cohort counts and not for row-level model development by a non-contributor. |
| CMS synthetic claims | CMS publishes a synthetic public use file intended for software testing and analysis development. | Synthetic claims are appropriate for pipelines and not for estimates of real utilisation or clinical performance. |

The first four rows are current and release-dependent. Verify the version and the counts before quoting them. The structural conclusions, single center, voluntary participation, volunteer cohort, and query-level access, are stable across releases.

### Source choice by role

| Role | Primary question | Natural source | What they must not infer |
|---|---|---|---|
| Hospital quality lead | How does our care compare over time? | Local EHR plus a benchmark | National ranking from one site |
| Payer analyst | Who is at risk of a costly event? | Claims and enrolment | Clinical severity from billing |
| Device developer | Will the alert fire in time? | EHR or ICU with timestamps | Ward performance from ICU data |
| Epidemiologist | How common is the condition? | Registry or survey | Incidence from a clinical cohort |
| Clinical researcher | Does the exposure affect the outcome? | Linked EHR and claims, or a trial | Causation from association |
| Operations lead | Where is the bottleneck? | EHR timestamps and flow data | Clinical need from utilisation |

### Documented fact versus inference

- **Documented fact:** a release note, a peer-reviewed descriptor, a custodian page, or a published measure specification.
- **Inference:** a planning consequence drawn from the fact for a different setting.
- **Unverified summary:** a secondary account of a count or a version. Verify before citing.

A case table should carry a source and a status for each row. The distinction matters most for recent releases, where a count can change between the announcement and the next release.

### Why the source choice is an industry decision, not only a scientific one

A hospital that wants to reduce readmissions controls its own EHR and can act locally. A payer that wants to reduce readmissions needs claims because its members move between hospitals. A device company that wants to sell a deterioration alert needs a hospital partner because the alert acts inside the workflow. The source follows the locus of action. Teams that choose a source because it is downloadable usually discover that the action they want to influence is outside its field of view.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Unit confusion | A "patient cohort" has more rows than patients | Hospitalizations counted as people | Inflated cohort and wrong denominator | Unit audit before cohort build |
| Out-of-network loss | Readmission rate far below the payer's rate | Single-center or narrow-network data | Negative labels are wrong | Claims or linkage for the outcome |
| Coding-era pooling | Cohort spans 2015 without a crosswalk | ICD-9 to ICD-10 changes measured prevalence | Model learns the transition | Restrict the era or validate a crosswalk |
| Phenotype without validation | The code list is copied from a paper | Codes identify documentation, not disease | Label error caps performance | Chart audit and positive predictive value |
| Severity blind spot | Claims model ranks prior utilisation | No labs or physiology | High-risk tier is chronically ill, not acutely ill | Clinical augmentation or a different outcome |
| Immortal time | Exposure defined after the index date | Future information enters the cohort | Biased effect estimate | Point-in-time cohort definition |
| Competing risk ignored | Death is treated as event-free | A dead patient cannot be readmitted | Inflated survival and biased model | Competing-risk framework |
| Administrative censoring | The last month has fewer events | Follow-up truncated by the data cutoff | False negatives | Exclude or censor explicitly |
| Observation stays | Admissions and observation mixed | Payer rules differ | Inconsistent denominators | Revenue codes and claim type |
| Transfer counted as readmission | A transfer appears as a new admission | Facility-level billing | Inflated readmission rate | Apply the measure's transfer logic |
| Site effects hidden | Pooled metric with no site table | Practice and coding vary | Transportability fails | Report by site and hold out sites |
| Code list drift | The code library has no version | Annual coding updates | Cohort changes silently | Version control and re-run |
| Synthetic validation | Synthetic data used to claim performance | Convenience | Unsupported claim | Synthetic only for pipelines |
| Vendor network change | The product name is stable, the network is not | Underlying coverage changes | Cohort shifts without notice | Re-baseline on each release |
| Linkage error | Match rate is reported without a false-match estimate | Probabilistic linkage is treated as exact | Biased cohort and duplicated events | Validate a sample of linked pairs |
| Enrolment gap | A member disappears and returns | Coverage lapse looks like an event-free period | Wrong denominator | Build the enrolment spine explicitly |
| Attribution error | A member is assigned to the wrong provider | Attribution rule is unstable | Programme evaluation is biased | Document and test the rule |
| Measure borrowing | A CMS measure name is used with different logic | The label is not the measure | Incomparable results | Cite the specification and match it |

Red flags that should stop a cohort build:

- The data dictionary does not state what one row represents.
- The outcome is observed only inside one institution.
- The code list has no validation and no version.
- The cohort spans a coding transition without a crosswalk.
- The readmission rate is not compared with an external benchmark.
- Death is missing or unlinked.
- The denominator is described as "all patients" without inclusion rules.
- The vendor cannot say which hospitals contribute in which years.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test |
|---|---|---|---|
| The unit is correctly identified | Every count depends on it | Rows are hospitalizations, not patients | Unit audit against the data dictionary |
| The outcome is observable | Negative labels depend on it | Out-of-network events are missing | Compare with a claims benchmark |
| The cohort definition is stable | Reproducibility depends on it | Codes or guidance change | Version the code list and re-run |
| The phenotype is accurate enough | The ceiling depends on it | Positive predictive value is low | Chart audit with a confidence interval |
| Coding depth is comparable | Pooling depends on it | Sites code differently | Codes per encounter by site |
| Missingness is not informative | Complete-case analysis depends on it | Labs are ordered because of illness | Missingness by outcome and site |
| Death is captured | Competing-risk analysis depends on it | Deaths outside the network are missing | Linkage to a death registry |
| Enrolment is continuous | Denominators depend on it | Gaps in coverage | Enrolment-month audit |
| The network is stable | Trend analysis depends on it | Hospitals enter or leave | Network file by year |
| The era is homogeneous | Pooled estimates depend on it | A coding or practice change occurs | Stratify by era |
| The split is patient-level | Benchmark validity depends on it | A patient appears on both sides | Identifier overlap check |
| The licence covers the use | Product claims depend on it | Research-only terms | Clause matrix |
| The network is representative | Comparisons depend on it | Voluntary participation | Compare hospital mix with a frame |
| The linkage is accurate | Cross-source analysis depends on it | False matches and missed matches | Audit a linked sample |
| The attribution rule is stable | Programme evaluation depends on it | Members switch providers | Test sensitivity to the rule |
| The measure specification matches the label | Comparability depends on it | Borrowed name, different logic | Side-by-side specification check |

The deepest limit is that no EHR or claims dataset establishes causal effect. A model that predicts readmission is not a model that reduces it. The intervention, the workflow, and the counterfactual are separate objects of evidence, and a retrospective dataset cannot supply them.

## 12. Exercises

**Exercise 1. Classify six questions by source.**
Deliverable: a table mapping each question to claims, EHR, ICU, registry, or survey, with the reason and the blind spot.
Acceptance criteria: six questions across at least four sources; each row names what the source cannot answer.
Time: 35 minutes.

**Exercise 2. Audit a unit.**
Deliverable: a one-page audit of a table's grain, with the count at each level and the repeat factor.
Acceptance criteria: patient, encounter, and stay counts reconciled; at least one ratio computed; the implication for a claim stated.
Time: 45 minutes.

**Exercise 3. Audit outcome capture.**
Deliverable: a coverage diagram for 30-day readmission in a single-center EHR, with the observable and unobservable portions.
Acceptance criteria: names out-of-network, death, and transfer; estimates the direction of bias; proposes a linkage or a claim change.
Time: 50 minutes.

**Exercise 4. Validate a phenotype.**
Deliverable: a code list, a sampling plan, a chart-audit form, and a positive predictive value with a confidence interval.
Acceptance criteria: reproducible code list with version; at least 50 records; reference standard described; limitations stated.
Time: 75 minutes.

**Exercise 5. Detect a coding-era effect.**
Deliverable: a table of prevalence by year for three conditions around a coding transition, with an interpretation.
Acceptance criteria: at least three conditions; direction of change stated; a handling plan proposed.
Time: 45 minutes.

**Exercise 6. Write the "cannot support" statement.**
Deliverable: 300 words explaining why a single-center ICU dataset cannot support a national incidence claim.
Acceptance criteria: names the selection mechanism; distinguishes incidence from case mix; proposes an alternative source.
Time: 30 minutes.

**Exercise 7. Design a split.**
Deliverable: a split specification with patient, site, and time keys, fractions, and a hash plan.
Acceptance criteria: at least two grouping dimensions; pre-specified; leakage checks named.
Time: 45 minutes.

**Assessment note.** Exercises 2, 3, 4, and 7 produce artefacts that can be checked mechanically: a reconciled unit table, a coverage diagram, an audit form with a confusion matrix, and a split hash. Exercises 1, 5, 6, and 8 are judgement tasks and are marked against the rubric in section 13. Exercise 8 is the integration task.

**Exercise 8. Write a source memo.**
Deliverable: a two-page memo with source, unit, coverage, licence, benchmark, and prohibited claims.
Acceptance criteria: every section present; one quantified limitation; one alternative source; a decision recommendation.
Time: 60 minutes.

## 13. Self-check answers

**Item 1. What is the core difference between EHR and claims data?**
The EHR records clinical observations and actions inside an institution. Claims record billed services across institutions. The EHR is sharper on physiology and local workflow; claims are sharper on cross-hospital utilisation and cost. Neither is a superset of the other.

**Item 2. Why is a claim not a clinical record?**
A claim exists to request payment. Its diagnosis and procedure codes are assigned for billing and statistics, often after the encounter, and its completeness depends on coverage and network. It can support utilisation and cost questions and cannot support bedside severity without a clinical source.

**Item 3. MIMIC-IV v3.1 has 364,627 patients and 546,028 hospitalizations. Which is the denominator for a per-patient rate?**
The patient count, after deciding which patients are eligible. The hospitalization count is the denominator for a per-admission rate. The two rates answer different questions, and mixing them produces a number with no clear interpretation.

**Item 4. Why can a single-center EHR not measure 30-day readmission?**
It observes only returns to that center. A readmission to another hospital is invisible and is coded as a non-event, so the negative labels are wrong. The direction of the bias is downward: the measured rate is lower than the true rate.

**Item 5. What is administrative censoring, and why does it matter?**
Follow-up ends because the data end, not because the patient is event-free. A discharge near the data cutoff has an incomplete 30-day window. Counting it as a non-readmission biases the estimate downward and teaches the model that recent discharges are low risk.

**Item 6. Why is death a competing risk for readmission?**
A patient who dies cannot be readmitted. If death is treated as event-free follow-up, the model learns a false association between survival and non-readmission. A competing-risk analysis or a composite outcome with an explicit definition is required.

**Item 7. What does the ICD-9 to ICD-10 transition change?**
The number and granularity of codes, and therefore the measured prevalence of many conditions. A cohort that spans the transition can change because the coding changed, not because the patients did. Restrict the era or validate a crosswalk.

**Item 8. Why is a random row split dangerous in EHR data?**
The same patient can have multiple encounters, and the same physiology can appear on both sides. A random row split leaks patient identity and inflates performance. Split by patient, and by site or time where possible.

**Item 9. What is the difference between a CMS readmission measure and a research readmission label?**
The CMS measure has a specified index admission, planned-readmission exclusions, risk adjustment, and a payment use. A research label may use a different window, a different exclusion set, and a different population. Sharing a name does not make them the same quantity.

**Item 10. When is a synthetic EHR dataset acceptable?**
For pipeline development, schema testing, interface testing, teaching, and privacy-safe demos. It is not acceptable for clinical performance, fairness, or prevalence claims. The synthetic distribution is a design choice, not an observation.

**Item 11. Why is a claim line a poor unit for a cost analysis?**
A claim line is one service within a claim, and a single episode spans many lines and often many claims. Summing lines without a claim header double-counts adjustments, and without an episode grouper it mixes professional and facility payments. The unit should be the claim or the episode, with a defined payment field.

**Item 12. What is the difference between deterministic and probabilistic linkage, and why does it matter?**
Deterministic linkage matches on an exact identifier; probabilistic linkage matches on a combination of fields with estimated weights. Probabilistic linkage produces false matches and missed matches, and both bias the cohort. A study that links sources must report the match rate and, where possible, the estimated error.

**Item 13. What is attribution, and why can it break a programme evaluation?**
Attribution assigns a member to a provider, hospital, or care team. If members switch providers or the rule changes between years, the denominator changes and a before-and-after comparison mixes a real effect with a reclassification. The rule should be fixed in advance and tested for sensitivity.

**Item 14. Why is an enrolment gap different from a healthy period?**
A gap means the data cannot observe events, not that no events occurred. Treating a coverage gap as event-free follow-up creates false negatives. The enrolment spine should define observable person-time, and gaps should be censored or excluded.

**Rubric for the design exercises (3, 4, 7, 8).** Full marks require a named unit, a stated coverage limit, a validation or leakage check, and a prohibited-claim statement. Partial marks for a complete structure with no quantified limitation. No marks for a claim that the data cannot support.

## 14. Interview questions

**1. Walk me through choosing a source for a readmission model.**
A strong answer defines the decision and the unit, chooses claims for the cross-hospital outcome, checks enrolment and network coverage, validates the cohort, and designs a temporal split. A weak answer starts with MIMIC because it is available.

**2. What can claims never tell you?**
Severity at the bedside, labs, vitals, adherence, out-of-network care, and the reason a therapy stopped. A strong answer adds that diagnosis is a billing concept recorded after the encounter. A weak answer says "less detail" without consequences.

**3. MIMIC-IV or eICU for a sepsis model?**
MIMIC for physiological depth and phenotyping, eICU for multi-site transport, with external validation either way. A strong answer notes that neither is a population sample. A weak answer picks the larger dataset.

**4. How do you validate a phenotype?**
Define the code list and the clinical reference, sample records, adjudicate blind to the code, compute positive predictive value and sensitivity where possible, and report the confusion matrix. A strong answer versions the code list. A weak answer cites a paper's code list without testing it locally.

**5. Why is the unit of analysis a modeling decision?**
It determines the denominator, the independence assumption, and the split. A patient with repeated admissions contributes correlated rows, and a random split leaks. A strong answer connects the unit to the split. A weak answer treats it as a formatting detail.

**6. How do you detect coding-era effects?**
Plot prevalence by year for stable clinical conditions, mark the coding transition, compare with an external benchmark, and test a crosswalk. A strong answer distinguishes coding change from epidemiology. A weak answer assumes the codes are stable.

**7. What is the difference between a hospitalization and an ICU stay?**
A hospitalization is an inpatient admission; an ICU stay is a period within it. A patient can have multiple ICU stays in one hospitalization, and only a fraction of hospitalizations include ICU care. A strong answer uses the correct denominator. A weak answer uses the terms interchangeably.

**8. How would you evaluate an EHR foundation model?**
Few-shot and linear probing, calibration with limited labels, cross-institution transport, subgroup performance, and a simple baseline. A strong answer notes that EHRSHOT and FoMoH are starting points, not validations. A weak answer reports a single benchmark rank.

**9. What does All of Us add, and what does it not?**
It adds a large, diverse volunteer cohort with genomics, surveys, and linked EHR, and it does not provide unbiased national prevalence or uniform linkage. A strong answer names the estimand. A weak answer treats size as representativeness.

**10. When would you refuse to build the model?**
When the outcome is unobservable in the available source, the population is wrong, the licence blocks the use, or the label cannot be validated. A strong answer treats refusal as a valid result. A weak answer builds on a proxy and renames the claim.

**11. How do you handle missing labs in an EHR cohort?**
Distinguish missing by design, missing because not ordered, and missing because not resulted. Include an order indicator, model the missingness, or restrict the cohort. A strong answer tests whether missingness predicts the outcome. A weak answer imputes the mean.

**12. What is the difference between transportability and external validation?**
External validation is an evaluation in another setting. Transportability is the property of retaining performance across settings. One external test is evidence, not proof. A strong answer names the shift being tested. A weak answer treats one site as generalization.

**13. How do you decide whether a vendor's network is adequate?**
Compare the network's hospitals, geography, payer mix, and years with the target population, and estimate out-of-network leakage. A strong answer asks for the network file, not the marketing deck. A weak answer accepts the total patient count.

**14. What is the first check you run on a new EHR extract?**
The grain: what one row represents, and whether the identifier is a patient, an encounter, or an event. Then the count at each level and the missingness by site. A strong answer runs the unit audit before any modeling. A weak answer starts with a descriptive table of the target variable.

**15. What is the difference between incidence and case mix?**
Incidence is the rate of new cases in a defined population over time. Case mix is the composition of patients treated in a setting. A tertiary ICU can have a high case mix of severe disease and a low incidence of the condition in the population. Confusing them produces a claim about the population from a claim about the hospital.

**16. How do you report a source limitation without weakening the whole memo?**
State the limitation, its direction, and the claim it constrains. "Readmissions outside the network are unobserved, so the measured rate is lower than the true rate and the model cannot be used for cross-hospital ranking" is stronger than a general caveat. A strong answer is specific and bounded. A weak answer lists every possible limitation and makes no decision.

## 15. Key terms

- **Administrative censoring** — follow-up ends because the data end.
- **All of Us** — NIH cohort with surveys, EHR, and genomics.
- **Claim** — a billing request for services.
- **Coding depth** — the number of codes recorded per encounter.
- **Competing risk** — an event that prevents the outcome of interest.
- **CPT** — Current Procedural Terminology, used for billing.
- **DRG** — Diagnosis-Related Group, a payment category.
- **EHR** — electronic health record.
- **eICU-CRD** — multi-hospital ICU research database.
- **Encounter** — any interaction with the health system.
- **Epic Cosmos** — contributor-only de-identified EHR network.
- **FHIR** — health data exchange standard.
- **HCPCS** — Healthcare Common Procedure Coding System.
- **Hospitalization** — an inpatient admission.
- **ICD-10-CM** — US diagnosis coding system.
- **ICU stay** — a continuous period in an intensive care unit.
- **Immortal time bias** — bias from using future information to classify exposure.
- **Index admission** — the admission that starts follow-up.
- **LOINC** — laboratory and observation code system.
- **MIMIC-IV** — single-center EHR and ICU research resource.
- **OMOP CDM** — common data model for observational health data.
- **Person-time** — total time at risk in a cohort.
- **Phenotype** — an algorithm that identifies a clinical condition.
- **Present on admission** — a flag for conditions existing at admission.
- **Readmission** — a subsequent admission within a defined window.
- **SNOMED CT** — clinical terminology.
- **Synthea** — open-source synthetic patient generator.
- **Unit stay** — a stay in one ICU unit.
- **Upcoding** — assignment of codes that increase payment.
- **Washout period** — a gap used to avoid counting prevalent cases.
- **Attribution** — assigning a member to a provider or programme.
- **Completeness** — the degree to which required fields are present.
- **Deterministic linkage** — matching on an exact identifier.
- **Enrolment spine** — the table of coverage periods used as a denominator.
- **False match rate** — the share of linked pairs that are wrong.
- **Granularity** — the finest unit available in a dataset.
- **Linkage** — joining records across sources at the person level.
- **Observation stay** — a short stay without formal inpatient admission.
- **Person-time** — total time at risk contributed by the cohort.
- **Probabilistic linkage** — matching on a combination of fields with weights.
- **Provenance** — the origin and processing history of a field.
- **Timeliness** — the delay between an event and its availability.
- **Transfer** — movement between facilities, often excluded from readmission counts.
- **Utilisation** — the use of services, measured in claims.
- **Case mix** — the composition of patients treated in a setting.
- **Incidence** — the rate of new cases in a defined population.
- **Prevalence** — the proportion of a population with a condition at a time.
- **Lookback window** — the period before the index date used for baseline.
- **Index date** — the anchor date for follow-up.
- **Net benefit** — a decision-analytic metric incorporating harms and benefits.
- **Calibration slope** — the agreement between predicted and observed risk across the range.
- **Clinical informatics** — the field concerned with health data systems and their use.
- **Data dictionary** — documentation of tables, fields, and codes.
- **De-identification** — removal or transformation of identifiers under a standard.
- **Encounter type** — the setting of a visit, such as inpatient, ED, or ambulatory.
- **Health information exchange** — infrastructure for sharing records across organisations.
- **Longitudinal record** — a record that follows a person over time.
- **Negative predictive value** — the probability that a negative label is correct.
- **Positive predictive value** — the probability that a positive label is correct.
- **Reference standard** — the best available truth used to validate a label.
- **Sensitivity** — the probability that a true case is identified.
- **Specificity** — the probability that a non-case is correctly excluded.
- **Area under the curve** — a discrimination metric across thresholds.
- **Cohort** — a defined group followed over time.
- **Denominator** — the population at risk used in a rate.
- **Feature** — an input variable used by a model.
- **Label** — the target variable used for supervision or evaluation.
- **Observational study** — a study without an assigned intervention.
- **Target trial** — a design that emulates a randomised trial with observational data.

## 16. Sources

All URLs accessed 26 October 2026. Versions and counts change; verify at the custodian page.

1. Johnson AEW, Bulgarelli L, Shen L, et al. MIMIC-IV, a freely accessible electronic health record dataset. *Scientific Data* 2023;10:1. https://doi.org/10.1038/s41597-022-01899-x
2. PhysioNet. MIMIC-IV v3.1. https://physionet.org/content/mimiciv/3.1/
3. MIT Laboratory for Computational Physiology. MIMIC-IV documentation. https://mimic.mit.edu/
4. PhysioNet. eICU Collaborative Research Database v2.0. https://physionet.org/content/eicu-crd/2.0/
5. Pollard TJ, Johnson AEW, Raffa JD, et al. The eICU Collaborative Research Database, a freely available multi-center database for critical care research. *Scientific Data* 2018. https://doi.org/10.1038/sdata.2018.178
6. All of Us Research Program. Curated Data Repository version 9 announcement. https://support.researchallofus.org/hc/en-us/articles/50653909888788
7. National Institutes of Health. All of Us Research Program genomic and health database release, June 2026. https://www.nih.gov/news-events/news-releases/nihs-all-us-research-program-now-largest-integrated-genomics-health-database-world
8. All of Us Research Program. Data access tiers. https://www.researchallofus.org/data-tools/data-access/
9. Epic. Cosmos. https://cosmos.epic.com/
10. Stanford. EHRSHOT. https://ehrshot.stanford.edu/
11. MITRE. Synthea. https://synthetichealth.github.io/synthea/
12. Centers for Medicare and Medicaid Services. Synthetic Medicare claims public use files. https://data.cms.gov/
13. OHDSI. Common Data Model. https://ohdsi.github.io/CommonDataModel/
14. OHDSI. Eunomia. https://github.com/OHDSI/Eunomia
15. Centers for Medicare and Medicaid Services. Hospital Readmissions Reduction Program. https://www.cms.gov/medicare/payment/prospective-payment-systems/acute-inpatient-pps/hospital-readmissions-reduction-program-hrrp
16. Centers for Medicare and Medicaid Services. SEP-1 sepsis measure. https://www.cms.gov/medicare/quality/measures
17. US Department of Health and Human Services. HIPAA. https://www.hhs.gov/hipaa/
18. HL7. FHIR R4. https://hl7.org/fhir/R4/
19. US National Library of Medicine. ICD-10-CM. https://www.cdc.gov/nchs/icd/icd10cm.htm
20. PhysioNet. eICU Collaborative Research Database project page. https://physionet.org/content/eicu-crd/
21. PhysioNet. HiRID. https://physionet.org/content/hirid/
22. Amsterdam University Medical Centers. AmsterdamUMCdb. https://amsterdammedicaldatascience.nl/
23. UK Biobank. https://www.ukbiobank.ac.uk/
24. Centers for Disease Control and Prevention. National Health and Nutrition Examination Survey. https://www.cdc.gov/nchs/nhanes/
25. Module 21 survey file, sections 5 and 15. [`../../course/21_datasets_tools_benchmarks.md`](../../course/21_datasets_tools_benchmarks.md)

## 17. Links

- Module map and topic table: [`_index.md`](_index.md)
- Previous topic, access and licensing: [`02_access-and-licensing.md`](02_access-and-licensing.md)
- Next topic, imaging datasets: [`04_imaging-datasets.md`](04_imaging-datasets.md)
- Signal datasets: [`05_signal-datasets.md`](05_signal-datasets.md)
- Root survey file: [`../../course/21_datasets_tools_benchmarks.md`](../../course/21_datasets_tools_benchmarks.md)
- Labs, including the dataset audit: [`../../labs/`](../../labs/)

