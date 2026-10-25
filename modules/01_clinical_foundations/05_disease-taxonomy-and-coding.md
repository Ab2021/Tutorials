# Disease Taxonomy and Coding

> Module 01 — Clinical Foundations for Data Scientists
> Topic 05 of 10
> Estimated study time: 4 hours
> Prerequisites: 01, 03
> Version 2026.3

---

## 1. Why this topic exists

Every clinical dataset that spans institutions eventually meets a code. The code may be a
diagnosis, a procedure, a laboratory test, a drug, a symptom, or a billing group. Each comes from
a different system with a different purpose. Treating them as interchangeable is one of the most
common and most expensive mistakes in biomedical data science.

The problems are concrete.

**Problem 1: a code is not a diagnosis.** A diagnosis is a clinical judgment about a patient. A
code is an administrative classification assigned after documentation review, under rules
designed for statistics and payment. The same patient can receive different codes at two
hospitals, and the same code can describe different patients. The correlation is often strong
but never perfect.

**Problem 2: coding is incentive-sensitive.** In inpatient prospective payment, the diagnosis
determines the diagnosis-related group (DRG), and the DRG determines payment. Documentation and
coding improvement programs exist for legitimate reasons — accurate severity capture — and they
also create pressure toward more specific, more severe codes. A dataset built from claims is
therefore a dataset shaped by financial incentives.

**Problem 3: code systems change every year.** ICD-10-CM has an annual release, with new codes,
revised codes, and deleted codes. A phenotype built on a code list from 2019 may silently change
its cohort in 2026. Without versioning, the change is invisible.

**Problem 4: the same concept has many codes.** "Sepsis" is not one code. It is a family of
codes for the organism, the severity, and the associated organ dysfunction. "Heart failure" is a
family distinguished by type, acuity, and cause. A phenotype must decide how wide the family is,
and that decision changes the cohort.

This topic teaches the major classification systems, what each is for, how codes are assigned,
and how to build a phenotype that is explicit about its purpose and its error.

## 2. Learning outcomes

By the end of this topic you can:

1. **Distinguish** diagnosis, procedure, drug, laboratory, and billing classifications and
   **select** the right system for a stated task.
2. **Explain** the structure of an ICD-10-CM code, including the seventh character, laterality,
   and external cause.
3. **Describe** how a principal diagnosis, secondary diagnoses, present-on-admission flags, and
   DRG assignment are produced.
4. **Compare** ICD-10-CM, ICD-11, SNOMED CT, MeSH, Orphanet, and the Human Phenotype Ontology
   (HPO) on granularity, compositionality, and use.
5. **Build** a code-based phenotype with explicit code sets, position rules, and date windows.
6. **Quantify** the difference between two phenotype definitions using sensitivity, positive
   predictive value (PPV), and Cohen's kappa.
7. **Critique** a coding-based label for upcoding, rule-out coding, history codes, and version
   drift.

## 3. The core mental model

Coding is a translation layer between clinical care and administration. The translation loses
information, adds incentives, and changes over time.

```text
   PATIENT CARE
       |
       v
   DOCUMENTATION (notes, problem list, orders, results)
       |
       v
   CLINICAL DOCUMENTATION IMPROVEMENT (queries, clarifications)
       |
       v
   CODING (ICD-10-CM, ICD-10-PCS, CPT)
       |
       +-----------------------+
       |                       |
       v                       v
   BILLING (DRG, claim)    STATISTICS (registry, population health)
       |                       |
       v                       v
   CLAIMS DATA             CODED COHORT
       |                       |
       +-----------+-----------+
                   |
                   v
            RESEARCH PHENOTYPE
```

Each arrow is a lossy transformation. The research phenotype is at the end of the chain, and it
inherits every loss.

### 3.1 The four families of codes

| Family | Answers | Example system | Typical use |
|---|---|---|---|
| Disease classification | What condition does the patient have? | ICD-10-CM, ICD-11 | Billing, mortality, epidemiology |
| Procedure classification | What was done? | ICD-10-PCS, CPT | Billing, utilization |
| Clinical terminology | What is clinically true, in detail? | SNOMED CT | Problem lists, EHR interoperability |
| Specialized terminology | What test, drug, phenotype, or adverse event? | LOINC, RxNorm, HPO, MedDRA | Measurement, medication, research, safety |

A dataset may need all four. A sepsis cohort may use ICD-10-CM for the diagnosis, ICD-10-PCS for
the procedure, LOINC for the lactate, RxNorm for the antibiotic, and SNOMED CT for the problem
list entry. None of them is a substitute for another.

### 3.2 Composition versus enumeration

ICD-10-CM is largely enumerative: each concept has a predefined code, and combinations require
multiple codes or a combination code. ICD-11 and SNOMED CT are compositional: a stem concept can
be refined with extension codes or attributes. This is the deepest structural difference between
the systems.

```text
   Enumerative: one code per pre-defined concept
   A41.9  Sepsis, unspecified organism

   Compositional: stem + extensions
   [infection] + [organism] + [site] + [severity]
```

Compositional systems can express more, but they require more decisions at coding time and more
work at analysis time. Enumerative systems are simpler but force approximations.

### 3.3 The code-to-phenotype gap

```text
   Clinical truth
       |  documentation
       v
   Documented truth
       |  coding rules and incentives
       v
   Coded truth
       |  phenotype algorithm
       v
   Analytic cohort
```

The gap between clinical truth and coded truth is not random. It is larger for conditions that
are hard to document, socially stigmatized, or financially sensitive. It is smaller for
conditions with clear diagnostic criteria and specific codes. A phenotype should state which gap
it is exposed to.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Coding system |
|---|---|---|---|
| ICD | International Classification of Diseases | Treated as a clinical terminology | WHO |
| ICD-10-CM | US clinical modification for diagnoses | Confused with ICD-10-WHO | CDC/CMS |
| ICD-10-PCS | US procedure coding system | Confused with CPT | CMS |
| ICD-11 | WHO's eleventh revision | Assumed to be in use everywhere | WHO |
| SNOMED CT | Systematized Nomenclature of Medicine Clinical Terms | Treated as a billing code | SNOMED International |
| LOINC | Logical Observation Identifiers Names and Codes | Used for diagnoses | Regenstrief |
| MeSH | Medical Subject Headings | Confused with ICD | NLM |
| Orphanet | Rare disease nomenclature and portal | Confused with a general taxonomy | Orphanet/Inserm |
| ORPHAcode | Orphanet identifier for a rare disease | Treated as an ICD code | Orphanet |
| HPO | Human Phenotype Ontology | Confused with a disease classification | HPO consortium |
| UMLS | Unified Medical Language System | Treated as a single terminology | NLM |
| CUI | Concept Unique Identifier in UMLS | Confused with a code | UMLS |
| MedDRA | Medical Dictionary for Regulatory Activities | Used for diagnoses | ICH |
| CPT | Current Procedural Terminology | Used for inpatient procedures | AMA |
| HCPCS | Healthcare Common Procedure Coding System | Confused with CPT | CMS |
| MS-DRG | Medicare severity diagnosis-related group | Treated as a clinical category | CMS |
| Principal diagnosis | The condition chiefly responsible for admission | Confused with the first-listed code | Coding rule |
| Secondary diagnosis | Additional conditions affecting care | Treated as comorbidity only | Coding rule |
| POA | Present on admission | Confused with hospital-acquired | CMS |
| CC | Complication or comorbidity | Confused with severity | CMS |
| MCC | Major complication or comorbidity | Confused with a diagnosis | CMS |
| Query | A coders' request for clarification | Treated as a diagnosis | CDI process |
| Upcoding | Assigning a more severe code than documented | Confused with accurate specificity | Fraud concern |
| Rule-out | A condition considered but not confirmed | Coded as confirmed | Documentation error |
| History code | A code indicating past, not active, disease | Used for active disease | ICD-10-CM Z codes |
| Laterality | Right, left, bilateral, or unspecified | Omitted or defaulted | ICD-10-CM |
| Seventh character | Encounter or episode qualifier | Ignored in cohort definitions | ICD-10-CM |
| External cause | Injury or event cause code | Treated as a diagnosis | ICD-10-CM V-Y codes |
| Combination code | One code for two conditions | Split into two codes | ICD-10-CM |
| Stem code | Core ICD-11 concept | Confused with a full postcoordinated expression | ICD-11 |
| Extension code | ICD-11 qualifier | Ignored in analysis | ICD-11 |
| Postcoordination | Combining stem and extension codes | Confused with precoordination | ICD-11 |
| Linearization | An ordered view of ICD-11 | Treated as the only view | ICD-11 |
| Foundation | The complete ICD-11 content | Confused with the linearization | ICD-11 |
| Refset | A subset of SNOMED CT concepts | Confused with a hierarchy | SNOMED CT |
| Concept model | Rules for SNOMED CT composition | Ignored in mapping | SNOMED CT |
| Crosswalk | A mapping between code systems | Assumed to be one-to-one | Mapping artifact |
| Phenotype | A computable definition of a condition | Treated as a diagnosis | Algorithm |
| Code set | The list of codes used in a phenotype | Treated as complete | Algorithm |
| Validity | Whether a code reflects the condition | Confused with reliability | Assessment |
| Reliability | Agreement between coders or definitions | Confused with validity | Kappa |

Three definitions deserve expansion.

**Principal diagnosis is a coding rule, not a clinical hierarchy.** In inpatient coding, the
principal diagnosis is the condition established after study to be chiefly responsible for the
admission. It is not necessarily the most severe condition, the first one documented, or the one
the patient cares most about. The rule is defined in the official guidelines and can change with
the encounter.

**POA is a flag, not a diagnosis.** The present-on-admission indicator states whether a condition
was present when the patient was admitted. A code with POA = N is hospital-acquired for payment
purposes. A code with POA = Y may still have been documented late. The flag is a billing
construct, not a clinical timeline.

**SNOMED CT is a terminology, not a classification.** It contains concepts and relationships, not
mutually exclusive categories. Two SNOMED concepts can overlap, and a patient can have multiple
concepts for the same condition at different levels of granularity. A SNOMED-based cohort must
decide whether to use a concept alone or its descendants.

## 5. The mechanics, step by step

**Step 1. Define the purpose of the coding.**

- Input: the question the data must answer.
- Operation: choose the classification family — disease, procedure, drug, measurement, phenotype.
- Output: a system choice with a rationale.
- Reasoning: a billing cohort and a research cohort have different requirements. Billing needs
  the codes as submitted; research may need a broader or narrower definition, with chart review
  for validation.
- Branch point: if the purpose is safety surveillance, MedDRA may be required; if it is
  reimbursement, ICD-10-CM and CPT are required.
- Analytic artifact: a data source specification naming the system and version.

**Step 2. Identify the code version and effective date.**

- Input: the dataset documentation.
- Operation: record the code system version and the fiscal year.
- Output: a versioned code set.
- Reasoning: ICD-10-CM changes every 1 October. A code added in FY2026 does not exist in FY2025
  data. A code deleted in FY2025 may still appear in older records.
- Branch point: a multi-year dataset spans multiple versions; the phenotype must handle codes that
  appear and disappear.
- Analytic artifact: `code_system`, `code_version`, `effective_from`, `effective_to`.

**Step 3. Read the documentation, not just the code.**

- Input: the clinical note, problem list, and orders.
- Operation: identify what was documented as confirmed, suspected, ruled out, or historical.
- Output: a clinical concept with a status.
- Reasoning: coders are instructed not to code suspected conditions as confirmed in inpatient
  settings, but documentation is ambiguous and queries are not always answered. "Rule out
  sepsis" may become A41.9 in the data even though the clinician did not confirm it.
- Branch point: outpatient coding rules differ from inpatient rules on suspected conditions.
- Analytic artifact: `documented_status` if available; otherwise a limitation.

**Step 4. Assign the principal diagnosis.**

- Input: the documented conditions and the reason for admission.
- Operation: apply the official guidelines to select the principal diagnosis.
- Output: one principal diagnosis code.
- Reasoning: the principal diagnosis drives the DRG. For a patient admitted with pneumonia who
  develops sepsis, the principal diagnosis may be pneumonia with sepsis as a secondary code, or
  sepsis may be principal depending on the circumstances and the guidelines.
- Branch point: the same clinical course can produce different principal diagnoses depending on
  documentation and query outcomes.
- Analytic artifact: `principal_diagnosis_code`, `principal_diagnosis_poa`.

**Step 5. Assign secondary diagnoses.**

- Input: additional documented conditions.
- Operation: code each condition that affects care or resource use.
- Output: a list of secondary codes.
- Reasoning: not every documented condition is coded. Chronic conditions that do not affect the
  admission may be omitted; complications and comorbidities that increase resource use are coded.
  The set of secondary codes is therefore a severity signal, not a complete problem list.
- Branch point: a comorbidity present on admission may be coded as a CC or MCC, changing the DRG.
- Analytic artifact: `secondary_codes`, `cc_flag`, `mcc_flag`.

**Step 6. Apply the seventh character.**

- Input: the condition code and the encounter context.
- Operation: assign the seventh character for encounter type or episode of care.
- Output: a complete code.
- Reasoning: in ICD-10-CM, many injury, fracture, and obstetric codes require a seventh
  character: A for initial encounter, D for subsequent encounter, S for sequela. The same injury
  has different codes at different stages. Ignoring the seventh character merges initial and
  follow-up care.
- Branch point: a code without the seventh character is invalid for those categories.
- Analytic artifact: `seventh_character`, `encounter_type`.

**Step 7. Apply laterality.**

- Input: the condition and the side.
- Operation: assign right, left, bilateral, or unspecified.
- Output: a complete code.
- Reasoning: laterality matters for procedures, outcomes, and safety. "Unspecified" is a valid
  code but is less specific and often reflects missing documentation.
- Branch point: a high proportion of "unspecified" laterality at a site suggests a documentation
  or coding workflow problem.
- Analytic artifact: `laterality`.

**Step 8. Assign external cause codes.**

- Input: the injury or event.
- Operation: assign the external cause code in addition to the injury code.
- Output: an external cause code.
- Reasoning: external cause codes describe how and where an injury occurred. They are required
  for injury encounters in many settings and are essential for prevention research. They are
  often missing in administrative data.
- Branch point: the external cause may be documented in free text only.
- Analytic artifact: `external_cause_code`, `injury_intent`.

**Step 9. Assign Z codes for history and status.**

- Input: past conditions, devices, medications, social circumstances.
- Operation: assign Z codes where appropriate.
- Output: Z codes.
- Reasoning: Z codes describe factors influencing health status, not current disease. "Personal
  history of myocardial infarction" (Z86.79 or similar) is not the same as an active
  myocardial infarction code. A phenotype that includes history codes will overcount prevalent
  disease.
- Branch point: some Z codes indicate long-term medication use or device presence, which can be
  useful exposure proxies.
- Analytic artifact: `history_code_flag`, `status_code_flag`.

**Step 10. Apply present-on-admission flags.**

- Input: the coded condition and the admission date.
- Operation: assign Y, N, U, or W according to documentation.
- Output: a POA flag per code.
- Reasoning: POA distinguishes conditions present at admission from those acquired during the
  stay. It is required for inpatient claims in the United States and is used in quality and
  payment programs.
- Branch point: POA = U (unknown) is common in some datasets and should not be treated as N.
- Analytic artifact: `poa_flag`.

**Step 11. Query the clinician when documentation is ambiguous.**

- Input: an unclear or incomplete record.
- Operation: send a compliant query asking for clarification, not for a specific code.
- Output: a clarified record or no response.
- Reasoning: queries are a legitimate part of clinical documentation improvement. A leading query
  that suggests the desired answer is non-compliant. The query process introduces its own
  variation: the same record may be queried at one hospital and not another.
- Branch point: query rates vary by site, service, and coder, creating site-level coding
  differences.
- Analytic artifact: `query_flag`, `query_topic`, `query_response`.

**Step 12. Assign procedure codes.**

- Input: the operative report and procedure documentation.
- Operation: assign ICD-10-PCS for inpatient procedures or CPT for outpatient services.
- Output: procedure codes.
- Reasoning: ICD-10-PCS is a seven-character alphanumeric code describing the section, body
  system, root operation, body part, approach, device, and qualifier. CPT uses a different
  structure and is maintained by the American Medical Association.
- Branch point: the same procedure may be coded differently in inpatient and outpatient settings.
- Analytic artifact: `procedure_code`, `procedure_system`, `procedure_date`.

**Step 13. Group to a DRG.**

- Input: principal diagnosis, secondary diagnoses, procedures, discharge status.
- Operation: apply the MS-DRG grouper.
- Output: a DRG code and weight.
- Reasoning: the DRG is a payment category. It is not a clinical category and should not be used
  as a phenotype. A DRG can group clinically heterogeneous patients who share a resource profile.
- Branch point: the same clinical course can group differently depending on the principal
  diagnosis and CC/MCC capture.
- Analytic artifact: `drg_code`, `drg_weight`, `drg_version`.

**Step 14. Validate the coded record.**

- Input: the coded record.
- Operation: audit a sample against the documentation; compute agreement.
- Output: an error rate and error categories.
- Reasoning: coding audits are routine in hospitals. They find errors in specificity,
  principal diagnosis selection, and CC/MCC capture. The error rate is not zero and is not
  uniform across services.
- Branch point: if the audit sample is not random, the error rate is not generalizable.
- Analytic artifact: `audit_result`, `error_category`.

**Step 15. Map to a research terminology.**

- Input: ICD codes.
- Operation: map to SNOMED CT, UMLS, or a phenotype ontology.
- Output: a mapped concept set.
- Reasoning: mapping is not one-to-one. An ICD code may map to multiple SNOMED concepts, and a
  SNOMED concept may map to multiple ICD codes. The mapping is a source of error that should be
  versioned.
- Branch point: automated mapping without review is a common source of phenotype drift.
- Analytic artifact: `map_source`, `map_target`, `map_confidence`.

**Step 16. Build the phenotype.**

- Input: the mapped code set and the clinical question.
- Operation: define inclusion codes, exclusion codes, position rules, and date windows.
- Output: a reproducible phenotype algorithm.
- Reasoning: the phenotype is an argument about what counts as a case. Every choice should be
  justified. For sepsis, do you require a blood culture? An antibiotic? Organ dysfunction? A
  specific code? Each choice changes sensitivity and PPV.
- Branch point: a sensitive phenotype is useful for screening; a specific phenotype is useful for
  confirmation. The choice depends on the decision.
- Analytic artifact: a phenotype specification with code list, logic, and validation.

**Step 17. Validate the phenotype.**

- Input: the phenotype and a reference standard.
- Operation: chart review or comparison with a clinical registry.
- Output: sensitivity, specificity, PPV, NPV, and agreement.
- Reasoning: a phenotype without validation is an assumption. Validation on a sample is the
  minimum; validation across sites is better.
- Branch point: if the reference standard is itself code-based, the validation is circular.
- Analytic artifact: a validation report with a confusion matrix.

**Step 18. Handle history versus active disease.**

- Input: the code set.
- Operation: exclude history codes and require evidence of active disease.
- Output: an active-disease phenotype.
- Reasoning: a patient with a history of heart failure may have no active heart failure. A code
  for "history of" is not a case. A common error is to include all codes in a disease family,
  including history and status codes.
- Branch point: some conditions are chronic and always active (for example, type 1 diabetes);
  others are episodic (for example, asthma).
- Analytic artifact: `active_definition`, `history_exclusion`.

**Step 19. Handle code version changes.**

- Input: a multi-year dataset.
- Operation: build a crosswalk by year; test the phenotype in each year.
- Output: a versioned phenotype with year-specific code sets.
- Reasoning: new codes can split an existing category; deleted codes can leave gaps. A phenotype
  that uses a fixed code list will drift.
- Branch point: if the drift is large, the analysis should be stratified by year.
- Analytic artifact: `year`, `code_set_version`, `drift_report`.

**Step 20. Document provenance.**

- Input: all of the above.
- Operation: record the source, version, mapping, and validation for every code.
- Output: a phenotype package.
- Reasoning: reproducibility requires that another analyst can rebuild the cohort. A code list
  without a version is not reproducible.
- Analytic artifact: a phenotype package with code lists, logic, validation, and changelog.

### 5.1 ICD-10-CM structure examples

| Code | Category | What it expresses | Common confusion |
|---|---|---|---|
| A41.9 | Sepsis | Sepsis, unspecified organism | Confused with severe sepsis |
| A41.51 | Sepsis | Sepsis due to *Escherichia coli* | Requires organism documentation |
| R65.20 | Severe sepsis | Severe sepsis without shock | Not a stand-alone diagnosis |
| R65.21 | Severe sepsis | Severe sepsis with shock | Requires shock documentation |
| N17.9 | Acute kidney failure | Unspecified | Confused with chronic kidney disease |
| N17.0 | Acute kidney failure | With tubular necrosis | Requires pathology or clinical evidence |
| I50.20 | Heart failure | Unspecified systolic | Confused with diastolic |
| I50.30 | Heart failure | Unspecified diastolic | Confused with systolic |
| I50.9 | Heart failure | Unspecified | Loses type and acuity |
| E11.9 | Type 2 diabetes | Without complications | Confused with complication codes |
| E11.65 | Type 2 diabetes | With hyperglycemia | Requires documentation |
| Z79.01 | Long-term medication use | Anticoagulants | Not a disease code |

The seventh character and laterality are not shown in this table. They are required for many
injury, fracture, and obstetric codes, and they change the meaning of the code even when the
first six characters are identical.

A code family is not a concept. The family A41 contains sepsis due to many organisms, and the
family I50 contains heart failure of several types and acuities. A phenotype must decide whether
to use the whole family, a subfamily, or a single code. The decision should be written down,
because the cohort size and the clinical meaning change with it.

The table also shows why a prefix rule is not a phenotype. `LIKE 'I50%'` includes systolic,
diastolic, combined, and unspecified heart failure, and it does not distinguish acute from
chronic. A curated code list with a stated rationale is safer than a pattern.

## 6. Worked example

A health system wants to build a sepsis alert and needs a cohort to train and evaluate it. Three
teams propose three definitions. The question is not which definition is "correct" in the
abstract; it is which definition fits the decision, and what each one costs.

### 6.1 The cohort

The analytic file contains 10,000 inpatient admissions from a single health system between
January 2024 and December 2025. The intended decision is: "Which patients should receive a
sepsis response-team review within 6 hours of admission?"

The three definitions are:

1. **Code-based.** Any discharge diagnosis of sepsis (A41.x), severe sepsis (R65.20 or R65.21),
   or septic shock, in any position.
2. **Clinical-criteria-based.** Suspected infection (blood culture ordered and antibiotic
   administered within a defined window) plus organ dysfunction consistent with Sepsis-3
   (Sequential Organ Failure Assessment, SOFA, increase of 2 or more points).
3. **Treatment-based.** Lactate measured, antibiotics administered, and a fluid bolus given
   within 24 hours.

### 6.2 The counts

| Definition | Admissions meeting criteria | Proportion of cohort |
|---|---:|---:|
| Code-based | 620 | 6.2% |
| Clinical-criteria-based | 940 | 9.4% |
| Treatment-based | 1,250 | 12.5% |
| Code-based AND clinical-criteria-based | 410 | 4.1% |

The code-based cohort is 34% smaller than the clinical-criteria cohort (620 versus 940), and the
treatment-based cohort is 33% larger than the clinical-criteria cohort. A study that reports "the
sepsis cohort" without naming the definition is not reproducible.

### 6.3 The 2x2 table

Using the clinical-criteria definition as the reference standard for this illustration:

|  | Clinical criteria positive | Clinical criteria negative | Total |
|---|---:|---:|---:|
| Code positive | 410 | 210 | 620 |
| Code negative | 530 | 8,850 | 9,380 |
| Total | 940 | 9,060 | 10,000 |

From the table:

```text
Sensitivity of codes = 410 / 940 = 0.436, or 43.6%
Specificity of codes = 8,850 / 9,060 = 0.977, or 97.7%
PPV of codes          = 410 / 620 = 0.661, or 66.1%
NPV of codes          = 8,850 / 9,380 = 0.943, or 94.3%
```

Cohen's kappa:

```text
Observed agreement Po = (410 + 8,850) / 10,000 = 0.926
Expected agreement Pe = (620/10,000 x 940/10,000)
                      + (9,380/10,000 x 9,060/10,000)
                      = 0.062 x 0.094 + 0.938 x 0.906
                      = 0.005828 + 0.849828
                      = 0.855656
kappa = (0.926 - 0.855656) / (1 - 0.855656)
      = 0.070344 / 0.144344
      = 0.487
```

A kappa of 0.487 is moderate agreement. The two definitions are measuring related but distinct
constructs. The code definition is specific (97.7%) and insensitive (43.6%): it misses more than
half of the clinical-criteria cases. For a screening alert, that is a serious limitation. For a
confirmatory cohort, high specificity may be acceptable.

> **Note:** The clinical-criteria definition is not a perfect gold standard. It depends on
> culture ordering, antibiotic timing, and SOFA components, each of which has its own error. The
> comparison shows how two operational definitions differ; it does not establish which one is
> clinically true.

### 6.4 The chart-review correction

A random sample of 100 code-positive admissions is reviewed by two clinicians. They confirm
sepsis in 82 and classify 18 as not sepsis. The reasons are:

| Reason for false positive | Count | Mechanism |
|---|---:|---|
| Rule-out documented as confirmed | 7 | "Sepsis workup" and "possible sepsis" coded as A41.9 |
| History of sepsis | 5 | Prior episode coded in the current record |
| Coding error | 4 | Code assigned without supporting documentation |
| Clinical disagreement | 2 | Clinicians did not accept the criteria |

With 82% PPV, the expected number of true sepsis cases among the 620 code-positive admissions is
approximately 508. The expected number of false positives is approximately 112.

The 18% false-positive rate is not a reason to abandon codes. It is a reason to report the
validation and to avoid treating the code as ground truth.

### 6.5 The cost of the choice

If the model is trained on code labels and deployed on all admissions:

- 112 of 620 labelled positives are not sepsis. The model learns their features as sepsis.
- 530 clinical-criteria cases are labelled negative. The model learns that their features are
  not sepsis.

The label noise is not random. Rule-out coding is more common in patients who are investigated
and found not to have sepsis, who may be sicker than average. History coding is more common in
patients with prior episodes, who may have different baseline risk. The model can learn these
patterns and produce a cohort-specific artifact.

### 6.6 The coding trail for one admission

Consider one admission in the cohort:

| Element | Value | Note |
|---|---|---|
| Admission | 12 March 2025, 21:40 | Emergency department |
| Blood culture | 22:10 | Before antibiotics |
| Antibiotic | 23:05 | Piperacillin-tazobactam |
| Lactate | 4.1 mmol/L | 22:20 |
| SOFA increase | 3 points | Renal and coagulation |
| Discharge diagnosis | A41.9, R65.20, N17.9 | Sepsis, severe sepsis, AKI |
| POA | Y, Y, N | AKI judged hospital-acquired |
| DRG | Septicemia or sepsis with MCC | Grouped after discharge |
| Coding query | None | Documentation considered sufficient |

The AKI code has POA = N, meaning it was not present at admission. If the phenotype requires
"sepsis with organ dysfunction present at admission," this admission may fail the criterion even
though the SOFA increase occurred within the first 24 hours. The timing rule matters as much as
the code list.

### 6.7 The same patient in ICD-11

In ICD-11, the same clinical picture could be expressed with a stem code for sepsis plus
extension codes for the organism, the site, and the severity. The compositional expression is
more precise, but it requires the coder to make more choices. If the extension codes are not
captured, the analysis collapses back to the stem code and loses the detail.

This is the central trade-off: compositional systems can represent more, but only if the
composition is recorded. An analysis cannot recover detail that was never coded.

### 6.8 The phenotype specification

A defensible specification for this cohort would read:

- **Purpose:** training and evaluation cohort for a sepsis response-team alert.
- **Definition:** clinical criteria (suspected infection plus SOFA increase of 2 or more) as the
  primary definition; code-based definition as a secondary, high-specificity definition.
- **Time zero:** the earlier of the first antibiotic administration and the first SOFA-qualifying
  observation, rounded to the hour.
- **Inclusion:** age 18 or older; inpatient admission; at least one blood culture and one
  antibiotic within the window.
- **Exclusion:** admissions with a comfort-care-only plan; transfers with incomplete records.
- **Validation:** 100-chart review; report PPV with a confidence interval; repeat by site.
- **Version:** ICD-10-CM FY2025 and FY2026 code sets; Sepsis-3 logic version 1.0.

The specification is longer than the code, and that is appropriate. The code is the easy part.

## 7. Code, query, or calculation

The block computes the agreement metrics and demonstrates prefix matching for a code family.

```python
"""Phenotype agreement and code-family matching.

The counts are the illustrative cohort from section 6.
"""

from collections import Counter


def agreement(a: int, b: int, c: int, d: int) -> dict:
    """a=both positive, b=code only, c=reference only, d=both negative."""
    n = a + b + c + d
    po = (a + d) / n
    pe = ((a + b) / n) * ((a + c) / n) + ((c + d) / n) * ((b + d) / n)
    kappa = (po - pe) / (1 - pe)
    sensitivity = a / (a + c)
    specificity = d / (b + d)
    ppv = a / (a + b)
    npv = d / (c + d)
    return {
        "n": n, "po": po, "pe": pe, "kappa": kappa,
        "sensitivity": sensitivity, "specificity": specificity,
        "ppv": ppv, "npv": npv,
    }


m = agreement(a=410, b=210, c=530, d=8850)
print(f"n = {m['n']}")
print(f"observed agreement: {m['po']:.3f}")
print(f"expected agreement: {m['pe']:.6f}")
print(f"kappa: {m['kappa']:.3f}")
print(f"sensitivity: {m['sensitivity'] * 100:.1f}%")
print(f"specificity: {m['specificity'] * 100:.1f}%")
print(f"PPV: {m['ppv'] * 100:.1f}%")
print(f"NPV: {m['npv'] * 100:.1f}%")

# Code-family matching: A41.x plus R65.2x.
codes = [
    "A41.9", "A41.51", "A41.01", "R65.20", "R65.21",
    "N17.9", "I50.9", "E11.9", "A40.9", "A02.1",
]
sepsis_prefixes = ("A41", "R65.2")


def in_family(code: str, prefixes: tuple[str, ...]) -> bool:
    return any(code.startswith(p) for p in prefixes)


family = [c for c in codes if in_family(c, sepsis_prefixes)]
print()
print("sepsis-family codes:", family)
print("count:", len(family))

counts = Counter(c.split(".")[0] for c in codes)
print("three-character categories:", dict(sorted(counts.items())))

# Chart-review PPV with a Wilson 95% interval, no external library.
def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    margin = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denominator
    return centre - margin, centre + margin


low, high = wilson(82, 100)
print()
print(f"chart-review PPV: 82.0% (95% CI {low * 100:.1f}-{high * 100:.1f}%)")

assert round(m["kappa"], 3) == 0.487
assert round(m["sensitivity"], 3) == 0.436
assert len(family) == 5
```

Expected output:

```text
n = 10000
observed agreement: 0.926
expected agreement: 0.855656
kappa: 0.487
sensitivity: 43.6%
specificity: 97.7%
PPV: 66.1%
NPV: 94.3%

sepsis-family codes: ['A41.9', 'A41.51', 'A41.01', 'R65.20', 'R65.21']
count: 5
three-character categories: {'A02': 1, 'A40': 1, 'A41': 3, 'E11': 1, 'I50': 1, 'N17': 1, 'R65': 2}

chart-review PPV: 82.0% (95% CI 73.3-88.3%)
```

The prefix match is deliberately simple. It would also match A41.0 (sepsis due to
*Staphylococcus aureus*), which is correct for this family, and it would not match a code that
records sepsis under a different category. Real phenotype code sets are curated lists, not
prefix rules, because prefix rules over-include.

A SQL query for the code family with POA filtering:

```sql
-- Sepsis-family codes, excluding history and requiring present on admission.
SELECT admission_id,
       patient_id,
       admit_ts,
       discharge_ts,
       principal_diagnosis_code,
       ARRAY_AGG(diagnosis_code) AS sepsis_codes
FROM diagnoses
WHERE (diagnosis_code LIKE 'A41%' OR diagnosis_code LIKE 'R65.2%')
  AND poa_flag = 'Y'
  AND diagnosis_code NOT LIKE 'Z%'
GROUP BY admission_id, patient_id, admit_ts, discharge_ts, principal_diagnosis_code;
```

Three details matter. The `LIKE 'A41%'` pattern includes all organisms and the unspecified code.
The `poa_flag = 'Y'` filter removes hospital-acquired cases, which may or may not be desired. The
`Z%` exclusion is redundant for A41 and R65 but is included to show the habit: history codes are
a different family and must be excluded deliberately.

## 8. Data, standards, and regulatory context

**WHO classifications.** The International Classification of Diseases is maintained by the World
Health Organization. ICD-10 is still used for mortality and morbidity in many countries. ICD-11
came into effect on 1 January 2022 and contains more than 17,000 diagnostic categories, with stem
codes, extension codes, and postcoordination. The WHO 2026 fact sheet reports 132 countries at
various stages of implementation, 10 official languages released, and continued growth in API
use. Implementation is country-specific and uneven.

**US diagnosis coding.** ICD-10-CM is maintained by the Centers for Disease Control and
Prevention (CDC), National Center for Health Statistics, with the Centers for Medicare and
Medicaid Services (CMS). The fiscal year 2026 release is effective 1 October 2025 through 30
September 2026. The official guidelines for coding and reporting are published with each release.
The United States has not adopted ICD-11 for routine morbidity coding; ICD-10-CM remains the
required HIPAA code set for diagnoses.

**US procedure coding.** ICD-10-PCS is maintained by CMS and is used for inpatient procedures.
CPT is maintained by the American Medical Association and is used for outpatient and physician
services. HCPCS Level II covers supplies, drugs, and services not in CPT. These systems are not
interchangeable, and a procedure analysis must state which one it uses.

**HIPAA code sets.** The Health Insurance Portability and Accountability Act (HIPAA)
administrative simplification rules adopt standard code sets for electronic transactions,
including ICD-10-CM, ICD-10-PCS, CPT/HCPCS, and NDC. Using a non-standard code set in a covered
transaction is not compliant. This is a binding requirement for covered entities, not a research
convention.

**SNOMED CT.** SNOMED CT is maintained by SNOMED International. It is a compositional clinical
terminology with concepts, descriptions, relationships, and reference sets. The October 2025
International Edition included anatomy reorganization, alignment of the meningioma hierarchy with
the WHO Classification of Tumours, 5th edition, and updated maps to ICD-10. The July 2026 release
raised the description length limit from 255 to 4,096 characters. SNOMED CT is not a billing
classification; it is used in problem lists, documents, and interoperability.

**LOINC.** LOINC identifies laboratory tests and clinical observations. Version 2.80 was released
in February 2025 with 104,672 total terms. LOINC is not a diagnosis system, and using it for
diagnoses is a category error.

**Other terminologies.** MeSH (Medical Subject Headings) is the National Library of Medicine's
indexing vocabulary for biomedical literature. Orphanet maintains the ORPHAcode nomenclature for
rare diseases. The Human Phenotype Ontology (HPO) provides terms for phenotypic abnormalities.
The Unified Medical Language System (UMLS) integrates many terminologies and assigns Concept
Unique Identifiers (CUIs). MedDRA is the International Council for Harmonisation's dictionary for
adverse events, organized into System Organ Classes, Preferred Terms, and Lowest Level Terms.

**Payment.** MS-DRG assignment uses the principal diagnosis, secondary diagnoses, procedures, and
discharge status. The grouper version changes annually. A DRG is a payment category; it is not a
clinical diagnosis and should not be used as a phenotype.

> **Note:** This section is educational and is not medical advice, coding advice, or legal advice. Code
> assignment rules are complex and change annually; consult the current official guidelines and
> a certified coder for operational decisions.

## 9. Industry practice and case evidence

**Annual code churn is routine.** The FY2026 ICD-10-CM release took effect on 1 October 2025. The
CDC publishes the code descriptions, addenda, conversion tables, and official guidelines with
each release. Organizations that do not version their code sets experience silent cohort drift:
a phenotype that used a code deleted in the new release will lose cases, and a phenotype that
does not include a newly added code will miss a newly distinguishable condition. The documented
fact is the annual release cycle. The inference is that every phenotype needs a version and a
yearly regression test.

**ICD-11 adoption is uneven.** As of the WHO's 2026 reporting, 132 countries were at some stage
of ICD-11 implementation. Malaysia transitioned its national health data system beginning in
2023. Germany has pursued a documentation strategy combining SNOMED CT and ICD-11. The
Netherlands estimated implementation by 2031. Sweden targeted mortality coding in 2027 and
admitted care in 2028. The United Kingdom rejected a "big bang" approach in favour of phased
rollout with mappings. Australia had no fixed date, with a business case process underway. These
are documented national positions, not a prediction of when any country will complete the
transition.

**SNOMED CT changes affect phenotypes.** The October 2025 release inactivated duplicate heart
failure concepts, aligned the meningioma hierarchy with the WHO CNS tumour classification, and
updated the ICD-10 map with 398 new mappings and 57 retirements. A phenotype that relies on a
SNOMED concept which was inactivated or remodelled can change without any change in the clinical
population. Concept retirement is a data quality event, not a clinical event.

**Code-based sepsis definitions are known to be insensitive.** The illustrative cohort in section
6 shows a sensitivity of 43.6% against a clinical-criteria reference. This is consistent with the
general finding in validation studies that administrative codes identify a subset of clinically
defined sepsis, with sensitivity varying by code set, position, and setting. The precise value
is site-specific and should not be imported. The inference is that a code-based sepsis label is
a high-specificity, low-sensitivity instrument and should be described as such.

**Documentation and coding improvement changes the data.** Hospitals employ clinical
documentation improvement specialists and query clinicians to improve specificity. This is
legitimate and can improve accuracy. It also creates a secular trend: a hospital that invests in
documentation improvement will code more severity, which can look like worsening case mix. Any
longitudinal analysis of severity or mortality should account for documentation trends. This is
inference from the structure of the payment system, not a finding from a single study.

**Problem lists and claims disagree.** The problem list is a clinical communication tool,
maintained by clinicians; the claim is an administrative artifact, maintained by coders. Studies
comparing them consistently find discordance in both directions. Neither is a gold standard. A
phenotype that uses one should validate against the other and against chart review.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Code treated as diagnosis | Cohort prevalence far from clinical expectation | Administrative artifact | Label noise | Chart validation |
| History codes included | Prevalence rises with age | Past conditions coded as current | Overcount | Exclude Z codes; require active evidence |
| Rule-out coded as confirmed | Sepsis rate tracks culture ordering | Documentation ambiguity | False positives | NLP or chart review |
| Code version drift | Cohort size changes at fiscal year boundary | Annual code release | Non-reproducible cohort | Versioned code sets; yearly test |
| Principal diagnosis assumed first | Order of codes differs from clinical priority | Coding rule | Wrong severity adjustment | Use principal diagnosis field |
| POA ignored | Hospital-acquired cases included | Flag not used | Mixed baseline risk | Filter or stratify by POA |
| DRG used as phenotype | Groups are clinically heterogeneous | Payment grouping | Misclassified disease | Use diagnosis codes, not DRG |
| Laterality unspecified | High proportion of unspecified codes | Documentation gap | Cannot study side-specific outcomes | Report unspecified rate |
| Seventh character ignored | Initial and follow-up merged | Encounter qualifier omitted | Duplicated episodes | Parse the seventh character |
| Crosswalk assumed one-to-one | Mapping counts differ | Many-to-many mapping | Loss or duplication | Versioned map with review |
| SNOMED concept retired | Cohort shrinks without clinical change | Terminology maintenance | Silent case loss | Monitor inactive concepts |
| Prefix matching over-includes | Code set larger than intended | Pattern matches unintended codes | False positives | Curated code lists |
| Upcoding pressure | Severity rises without case-mix change | Financial incentive | Bias in severity analyses | Audit; adjust for documentation trends |
| Query variation by site | Coding differs across hospitals | CDI practice variation | Site confounding | Site-stratified validation |
| Outpatient and inpatient rules mixed | Suspected conditions coded differently | Different coding guidelines | Inconsistent cohort | Specify setting and rule set |

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The code reflects the documented condition | Coding is the only scalable source | Rule-out, history, error | Chart review sample |
| The code set is complete for the condition | Cohort definition depends on it | New codes, rare variants | Compare with clinical registry |
| The code version is stable across the study | Pooling requires it | Annual release | Version field; year-stratified counts |
| The principal diagnosis reflects the admission reason | Severity adjustment depends on it | Coding rule and query variation | Audit principal diagnosis |
| POA flags are accurate | Hospital-acquired separation | Documentation timing | Compare with admission note |
| The mapping to research terminology is valid | Cross-site pooling | Many-to-many maps | Manual mapping review |
| The reference standard is independent | Validation depends on it | Code-based reference | Chart or registry adjudication |
| Documentation practice is stable over time | Trend analysis depends on it | CDI investment, template changes | Documentation trend analysis |
| The DRG is not a clinical category | Secondary analyses | Heterogeneous grouping | Avoid DRG as phenotype |
| The patient is in the coding universe | Cohort completeness | Out-of-network, self-pay, VA | Linkage coverage report |

## 12. Exercises

**Exercise 1 — System selection (recall and application, 40 minutes).**
For five tasks — inpatient mortality reporting, problem-list phenotyping, laboratory result
pooling, adverse event surveillance, and rare disease cohort building — name the most
appropriate code system and explain why. Deliverable: a five-row table. Acceptance criteria: one
system per task; the rationale names the system's purpose. Time: 40 minutes.

**Exercise 2 — Code anatomy (application, 40 minutes).**
Take the code A41.51 and explain each character, the category, the subcategory, and the code. Then
explain what the seventh character would add for a fracture code. Deliverable: a written
explanation with a diagram. Acceptance criteria: correct structure; seventh character explained;
laterality distinguished. Time: 40 minutes.

**Exercise 3 — Phenotype comparison (calculation, 60 minutes).**
Using the 2x2 table in section 6, compute sensitivity, specificity, PPV, NPV, and kappa. Then
recompute PPV if the prevalence of clinical sepsis doubles and the sensitivity and specificity
stay the same. Deliverable: a calculation sheet. Acceptance criteria: all metrics correct;
prevalence effect explained. Time: 60 minutes.

**Exercise 4 — Code list critique (critique, 50 minutes).**
A colleague proposes `WHERE diagnosis_code LIKE 'I50%'` for a heart failure cohort. Critique the
query and propose a revised code set. Deliverable: a 400-word critique and a code list.
Acceptance criteria: identifies history and acuity issues; proposes exclusions; notes that
unspecified codes are common. Time: 50 minutes.

**Exercise 5 — POA analysis (application, 45 minutes).**
Given a table with POA flags, compute the proportion of sepsis codes that are hospital-acquired
by site and year. Explain how the result could be misinterpreted. Deliverable: a table and a
300-word interpretation. Acceptance criteria: denominators explicit; at least two
interpretations; no causal claim. Time: 45 minutes.

**Exercise 6 — ICD-11 design (design, 60 minutes).**
Design a postcoordinated ICD-11 expression for "sepsis due to Escherichia coli with acute kidney
injury, present on admission." Identify the stem code, extension codes, and the information that
would be lost if extensions were not captured. Deliverable: a structured expression and a
loss analysis. Acceptance criteria: stem and extensions distinguished; loss stated; no invented
code numbers. Time: 60 minutes.

**Exercise 7 — Validation plan (design, 75 minutes).**
Write a validation plan for a code-based heart failure phenotype. Specify the sample size, the
reference standard, the adjudication process, and the metrics. Deliverable: a two-page plan.
Acceptance criteria: sample is random or stratified; reference standard is independent; metrics
include PPV and sensitivity; subgroup analysis included. Time: 75 minutes.

**Exercise 8 — Version drift audit (application, 60 minutes).**
Using two years of code data, identify codes that appear in only one year and codes whose counts
change by more than 50%. Propose a versioning strategy. Deliverable: a table and a one-page
strategy. Acceptance criteria: changes distinguished from coding practice changes; strategy
includes a regression test. Time: 60 minutes.

## 13. Self-check answers

**Answer to Exercise 3.** Sensitivity = 410/940 = 43.6%. Specificity = 8,850/9,060 = 97.7%. PPV =
410/620 = 66.1%. NPV = 8,850/9,380 = 94.3%. Kappa = 0.487. If prevalence doubles while
sensitivity and specificity stay fixed, PPV rises. At prevalence 18.8% (double 9.4%), with
sensitivity 43.6% and specificity 97.7%: diseased 1,880, TP 819.7, FP 186.7, PPV = 819.7/1,006.4
= 81.4%. The exact value depends on rounding, but the direction is clear: higher prevalence
raises PPV.

**Answer to Exercise 2.** A41 is the category for sepsis. A41.5 is the subcategory for sepsis due
to other Gram-negative organisms. A41.51 is sepsis due to *Escherichia coli*. The seventh
character in injury codes indicates the encounter: A for initial, D for subsequent, S for
sequela. It changes the meaning of the code even when the injury is the same.

**Answer to self-check on codes versus diagnoses.** A diagnosis is a clinical judgment. A code is
an administrative classification assigned after documentation review under coding rules. The code
can be wrong, incomplete, or influenced by payment. The diagnosis can be uncertain. Neither is
ground truth.

**Answer to self-check on history codes.** History codes (Z codes in ICD-10-CM) indicate past
conditions or status, not active disease. Including them in a disease cohort overcounts
prevalence. A phenotype should exclude them or require evidence of active disease.

**Answer to self-check on DRG.** A DRG is a payment group based on diagnosis, procedures, and
severity. It is not a clinical category. Two patients in the same DRG can have different diseases
and different outcomes.

**Answer to self-check on kappa.** Kappa adjusts observed agreement for chance agreement. A kappa
of 0.487 is moderate; the two definitions agree more than chance but are not interchangeable. A
high observed agreement can coexist with a low kappa when prevalence is very high or very low.

**Rubric for Exercise 6.** Full credit: stem and extension codes distinguished; postcoordination
explained; loss analysis explicit; no invented code numbers. Half credit: stem code only. No
credit: invents ICD-11 codes.

**Rubric for Exercise 7.** Full credit: random or stratified sample; independent reference
standard; adjudication with two reviewers; PPV and sensitivity; subgroup analysis; sample size
justified. Half credit: validation plan without independence. No credit: validation against the
same codes.

## 14. Interview questions

**1. What is the difference between ICD-10-CM and SNOMED CT?**
A strong answer explains classification versus terminology, enumerative versus compositional, and
billing versus clinical use. A weak answer says both are "medical codes."

**2. Why is a diagnosis code not a diagnosis?**
A strong answer explains documentation, coding rules, payment incentives, and validation. A weak
answer says codes can be wrong.

**3. What is a principal diagnosis?**
A strong answer gives the coding rule: the condition chiefly responsible for admission after
study. A weak answer says the first code listed.

**4. What does present-on-admission mean, and why does it matter?**
A strong answer explains the flag, hospital-acquired conditions, and the difference between Y, N,
U, and W. A weak answer says it means the patient had the condition.

**5. How would you build a sepsis phenotype?**
A strong answer compares code-based, clinical-criteria, and treatment-based definitions, and
states the trade-off. A weak answer gives a single code list.

**6. Why does a code-based phenotype drift over time?**
A strong answer explains annual releases, new and deleted codes, and documentation trends. A weak
answer says "coding changes."

**7. What is the difference between ICD-10-CM and ICD-10-PCS?**
A strong answer distinguishes diagnosis and inpatient procedure coding, and notes CPT for
outpatient. A weak answer says both are ICD-10.

**8. How do you validate a phenotype?**
A strong answer proposes chart review, an independent reference standard, PPV and sensitivity,
and subgroup analysis. A weak answer reports agreement with another code set.

**9. What is upcoding, and how would you detect it?**
A strong answer defines it, distinguishes legitimate specificity from fraud, and proposes audit
and case-mix adjustment. A weak answer says it is "coding more."

**10. What is the difference between a crosswalk and a map?**
A strong answer explains that mappings are many-to-many and versioned, and that a crosswalk
implies a direct correspondence that often does not exist. A weak answer treats them as synonyms.

**11. Why is ICD-11 postcoordination important?**
A strong answer explains stem plus extension codes and the loss of detail when extensions are not
captured. A weak answer says ICD-11 is newer.

**12. When would you prefer a narrow code set to a broad one?**
A strong answer weighs sensitivity and specificity against the decision and the cost of false
positives. A weak answer says narrow is more accurate.

## 15. Key terms

- **CC** — complication or comorbidity.
- **CDI** — clinical documentation improvement.
- **Code set** — the list of codes used in a phenotype.
- **Cohen's kappa** — chance-adjusted agreement.
- **Combination code** — one code for two conditions.
- **CPT** — Current Procedural Terminology.
- **Crosswalk** — a mapping between code systems.
- **CUI** — Concept Unique Identifier in UMLS.
- **DRG** — diagnosis-related group.
- **External cause code** — code for the cause of an injury.
- **Foundation** — the complete ICD-11 content.
- **HCPCS** — Healthcare Common Procedure Coding System.
- **History code** — code for a past condition.
- **HPO** — Human Phenotype Ontology.
- **ICD** — International Classification of Diseases.
- **ICD-10-CM** — US clinical modification for diagnoses.
- **ICD-10-PCS** — US inpatient procedure coding system.
- **ICD-11** — WHO's eleventh revision.
- **Laterality** — side of the body.
- **Linearization** — an ordered ICD-11 view.
- **LOINC** — laboratory and observation identifiers.
- **MCC** — major complication or comorbidity.
- **MedDRA** — adverse event dictionary.
- **MeSH** — Medical Subject Headings.
- **MS-DRG** — Medicare severity DRG.
- **Orphanet** — rare disease nomenclature.
- **ORPHAcode** — Orphanet identifier.
- **Phenotype** — computable definition of a condition.
- **POA** — present on admission.
- **Postcoordination** — combining stem and extension codes.
- **Principal diagnosis** — condition chiefly responsible for admission.
- **Query** — coder request for clarification.
- **Refset** — subset of SNOMED CT concepts.
- **Seventh character** — encounter qualifier in ICD-10-CM.
- **SNOMED CT** — clinical terminology.
- **Stem code** — core ICD-11 concept.
- **UMLS** — Unified Medical Language System.
- **Upcoding** — assigning a more severe code than documented.

## 16. Sources

Accessed 24 October 2026 unless otherwise stated.

**Standards and official releases**

- World Health Organization. ICD-11 Fact Sheet, 2026. https://icd.who.int/en/docs/ICD11_Fact_Sheet_2026.pdf
- World Health Organization. ICD-11 implementation FAQ, updated December 2025.
  https://www.who.int/standards/classifications/frequently-asked-questions/icd-11-implementation
- Centers for Disease Control and Prevention. ICD-10-CM FY2026 release and official guidelines,
  effective 1 October 2025. https://www.cdc.gov/nchs/icd/icd-10-cm/files.html
- Centers for Medicare and Medicaid Services. ICD-10 codes.
  https://www.cms.gov/medicare/coding-billing/icd-10-codes
- SNOMED International. October 2025 International Edition release notes.
  https://conf.spaces.snomed.org/wiki/spaces/RMT/pages/149881476/
- Logical Observation Identifiers Names and Codes. Version 2.80 release notes, February 2025.
  https://loinc.org/kb/release-notes-archive/2.80
- National Library of Medicine. Medical Subject Headings. https://www.nlm.nih.gov/mesh/
- National Library of Medicine. Unified Medical Language System. https://www.nlm.nih.gov/research/umls/
- Orphanet. Rare disease nomenclature. https://www.orpha.net/
- Human Phenotype Ontology. https://hpo.jax.org/
- International Council for Harmonisation. MedDRA. https://www.meddra.org/

**Implementation evidence**

- Australian Institute of Health and Welfare. Health classifications update 9, January 2026.
  https://www.aihw.gov.au/about-us/international-collaboration/australian-collaborating-centre-for-who/icd-11/health-classifications-updates/update9-jan26
- World Health Organization. ICD-11 implementation progress level indicator.
  https://www.who.int/data/gho/data/indicators/indicator-details/GHO/icd11-implementation-progress-level

**Background reading**

- O'Malley KJ, Cook KF, Price MD, et al. Measuring diagnoses: ICD code accuracy. *Health
  Services Research*, 2005. https://doi.org/10.1111/j.1475-6773.2005.00444.x

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`04_pharmacology-basics.md`](04_pharmacology-basics.md)
- Next topic: [`06_radiology-and-imaging-basics.md`](06_radiology-and-imaging-basics.md)
- Related topic: [`08_documentation-and-clinical-narratives.md`](08_documentation-and-clinical-narratives.md)
- Related topic: [`10_clinical-plausibility-checks.md`](10_clinical-plausibility-checks.md)
- Survey file: [`../../01_clinical_foundations.md`](../../01_clinical_foundations.md)
- Labs: [`../../labs/`](../../labs/)
- Cohort pipeline lab: [`../../labs/03_cohort_pipeline.ipynb`](../../labs/03_cohort_pipeline.ipynb)

