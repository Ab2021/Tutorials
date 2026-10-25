# EHR Data

> Module 02 — The Biomedical Data Landscape
> Topic 01 of 10
> Estimated study time: 4 hours
> Prerequisites: Module 01, Module 01A (none within this module)
> Version 2026.3

---

## 1. Why this topic exists

The electronic health record (EHR) is the data source that everyone assumes they already understand, and the one that most often produces silently wrong analyses.

An EHR is not a research database. It is an operational, legal, and billing instrument that a hospital uses to deliver care, document decisions, satisfy regulators, and get paid. Clinical facts are recorded as a by-product of those activities, in the shape that the workflow required at the time, by staff whose attention was on the patient rather than on the future analyst. Every analytic dataset built from an EHR is a reconstruction, and the reconstruction is where the errors enter.

The commercial stakes are concrete:

- A sepsis model trained on one health system's EHR can lose most of its discrimination when deployed at another, because the label definition, the lab ordering pattern, and the documentation template all differ.
- A payer or life-sciences team that buys "real-world EHR data" and treats the diagnosis column as truth will over-count disease, because rule-out diagnoses, history codes, and copy-forward problem lists are recorded alongside confirmed disease.
- A sponsor that plans a trial feasibility assessment from EHR counts without checking order-to-result lag will under-estimate screening time and mis-schedule sites.
- An AI vendor that trains on notes without auditing copy-forward will learn the template, not the patient.

The decision this topic supports is the one in the module's industry question: which data can answer this question, and what will it never be able to tell us. For EHR data, the honest answer is usually "it can tell you what was recorded during care inside this network, at the granularity the workflow imposed, and nothing about care elsewhere."

Three properties drive nearly every downstream problem, and they are worth stating up front because the rest of the file elaborates them:

1. **The EHR is encounter-anchored, not patient-anchored.** The atomic unit is a visit, an admission, or a message. A patient's record is an assembly of encounters that may span decades and multiple identifiers.
2. **Recording is conditional on workflow.** A lab exists because someone ordered it. A vital sign exists because a nurse documented it. A diagnosis exists because someone needed it for billing, the problem list, or a quality measure. Absence is rarely random and often informative.
3. **The schema is vendor-specific and version-specific.** Two Epic installations at two hospitals are not the same database. An upgrade can rename a flowsheet row, change a unit, or retire a code.

An analyst who internalizes only these three properties will avoid most of the failure modes in section 10.

### 1.1 Who is blocked, and by what

| Role | Decision blocked | What they get wrong without this topic |
|---|---|---|
| Clinical data scientist | Whether an EHR-derived cohort is fit for a model | Treats diagnosis codes as confirmed disease; ignores encounter fragmentation |
| Real-world evidence lead | Whether EHR data can support an effectiveness claim | Assumes the EHR captures the full treatment course, including specialty and out-of-network care |
| Trial feasibility analyst | Site selection and enrollment timeline | Counts coded patients rather than eligible patients; ignores order-to-result lag |
| Product manager for a clinical AI tool | Whether a model will transfer to a new site | Assumes a single-site validation predicts multi-site performance |
| Data engineer | How to model the source into a common data model | Loses the provenance of orders versus administrations versus documentation |
| Regulatory affairs lead | Whether a real-world data source is fit for a regulatory submission | Cannot describe the data's provenance, completeness, or fitness in the terms a regulator expects |
| Privacy officer | Whether an extract is de-identified | Under-estimates re-identification risk from dates, free text, and rare events |

### 1.2 The expensive errors, ranked by cost

1. **Wrong label.** A sepsis, readmission, or deterioration label defined from codes or from a single lab value will not mean what the clinical team means. Everything downstream inherits the error.
2. **Wrong denominator.** Encounters are not patients, and visits are not episodes. A rate computed on the wrong denominator can be off by a factor of two.
3. **Wrong time axis.** Order time, collection time, result time, and documentation time differ by minutes to days. A "predict before onset" model built on result time leaks the future.
4. **Wrong population.** An EHR network covers the patients who use that network. Patients who move, lose insurance, or seek care elsewhere disappear, and they are not missing at random.
5. **Wrong unit.** Milligrams per deciliter versus milligrams per liter, or a lab that switched assays mid-year, silently changes the meaning of a threshold.

Each of these is cheap to prevent at design time and expensive to discover at validation time. That asymmetry is the reason this topic exists as a separate teaching unit rather than a paragraph in the module survey.

A note on what this topic does not do. It does not teach SQL or Python, which the labs cover. It does not teach clinical medicine, though it assumes you will ask a clinician when a definition is ambiguous. It does not resolve the legal questions in section 8, which require counsel. What it does is give you the vocabulary and the procedure to ask the right questions of an EHR extract before you spend three months analyzing the wrong one.

> **Note:** Regulatory statements in this file are educational. They describe what the rules require in outline and are not legal advice. Any production decision needs review by counsel and by the institution's privacy office.

---

## 2. Learning outcomes

By the end of this topic you can:

1. **Distinguish** the eight core EHR data families (demographics, encounters, diagnoses, orders, administrations, results, notes, flowsheets) by their producer, their temporal resolution, and the question each can and cannot answer.
2. **Reconstruct** a patient timeline from encounter-anchored tables, resolving transfers, merges, and duplicate records, and state the reconstruction rules you used.
3. **Compute** a cohort denominator from an EHR extract, applying inclusion and exclusion criteria in a defensible order, and quantify the effect of each exclusion on the cohort's mortality or outcome rate.
4. **Audit** a diagnosis field for rule-out, history-of, and copy-forward contamination, and propose a phenotype definition that reduces it.
5. **Calculate** a directly standardized rate across sites and demonstrate a rank reversal caused by case-mix difference.
6. **Critique** a published EHR-based model's data section for provenance, label definition, and leakage risk, and write the three questions you would ask its authors.
7. **Design** a provenance record for an EHR extract that a second analyst could use to reproduce the cohort without contacting you.
8. **Choose** between an EHR-only design and an EHR-plus-linkage design for a stated decision, with an explicit statement of what the EHR alone cannot answer.

Mapping to assessment: outcomes 3, 5, and 7 are assessed by exercises 3, 5, and 7. Outcomes 1, 2, and 4 are assessed by the self-check items in section 13. Outcomes 6 and 8 are assessed by the design exercises and by the module's data-source decision memo.

---

## 3. The core mental model

The EHR is best understood as a set of **event streams that converge on an encounter**, produced by different actors for different purposes, and joined by identifiers that are themselves unreliable.

```text
                       IDENTITY LAYER
            (person -> MRN(s) -> enterprise ID -> research ID)
                              |
        +---------------------+---------------------+
        |                     |                     |
   ENCOUNTER            ORDER / EVENT          DOCUMENT
   (visit, admit)       (lab, drug, proc)      (note, report)
        |                     |                     |
        |                     v                     |
        |            +--------+--------+            |
        |            |                 |            |
        |        ORDERED          PERFORMED         |
        |            |                 |            |
        |            v                 v            |
        |      not given /      result value        |
        |      cancelled         + unit + flag      |
        |            |                 |            |
        +------------+--------+--------+------------+
                              |
                              v
                    ANALYTIC EVENT TABLE
        (patient_id, encounter_id, concept, time, value, provenance)
```

Read the diagram from the top. Identity resolution comes first and is the most common silent failure: a patient with two medical record numbers (MRNs) becomes two patients, and a merged record can duplicate every encounter. Encounters come second and define the frame. Orders, results, and documents hang off the encounter, but each has its own lifecycle and its own timestamps.

The critical asymmetry in the diagram is the fork between **ordered** and **performed**. Most EHR analytics collapses this fork and treats the order as the event. That collapse is the single most common source of wrong exposure definitions in EHR research: a medication order is not an administration, and an administration is not an ingestion.

### 3.1 The four clocks

Every EHR event carries at least four times, and they are not interchangeable:

| Clock | Meaning | Typical lag | Use |
|---|---|---|---|
| Order time | When the clinician placed the order | 0 | Decision-time features; exposure intent |
| Specimen or start time | When the sample was drawn or the drug started | Minutes to hours after order | Physiologic time |
| Result or end time | When the value was verified or the therapy ended | Minutes to days after specimen | Label construction; avoid for features |
| Documentation time | When the note or flowsheet row was saved | Minutes to days, sometimes after discharge | Never use for feature timing |

A model that uses result time as the feature timestamp and order time as the prediction timestamp will appear to predict well and will fail in deployment, because at order time the result did not exist. This is **temporal leakage**, and it is the most common reason a promising EHR model does not reproduce.

### 3.2 Encounter fragmentation

A single clinical episode can generate many encounter records:

- An emergency department (ED) visit that becomes an inpatient admission produces two encounters in many systems.
- An inter-facility transfer produces a discharge and an admission.
- An observation stay may convert to inpatient mid-stay.
- A professional claim and a facility claim may both exist for the same service.

The analyst must decide what the unit of analysis is: encounter, admission, episode of care, or patient-year. Each choice changes the denominator, the outcome rate, and the covariates. The choice must be written down, because a reviewer cannot infer it from the table names.

### 3.3 The information hierarchy

EHR data has a hierarchy of trustworthiness for clinical questions:

1. **Measured and verified** (lab result with units, device vital sign with a plausible range) — highest trust, still subject to assay and device change.
2. **Administered** (medication administration record, procedure performed) — high trust for "happened," lower for "why."
3. **Ordered** (medication order, imaging order) — evidence of intent, not of occurrence.
4. **Coded** (diagnosis, problem list) — evidence of a billing or documentation decision, not of clinical truth.
5. **Narrated** (free-text note) — rich and ambiguous; requires extraction and validation.

Most analytic errors come from moving an item up this hierarchy without justification: treating a code as a measurement, or an order as an administration.

### 3.4 What the model implies for design

If the EHR is a set of conditional event streams, then three design rules follow:

- **State the frame explicitly.** "Adults admitted with a first sepsis-coded encounter, 2022-2024, four hospitals" is a frame. "Sepsis patients" is not.
- **Separate the clocks.** Build features from data available at the decision time, and labels from data after it. Enforce this in code, not in a comment.
- **Carry provenance per row.** Every derived row should be traceable to a source table, a source row, and a transformation version. Without this, an audit is impossible.

---

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| EHR | Electronic health record; the longitudinal record maintained by a care organization | Used interchangeably with "EMR," which is usually a single-practice record | None |
| EMR | Electronic medical record; historically a single organization's record | Assumed to be multi-site | None |
| Encounter | A single interaction with the health system (ED visit, outpatient visit, admission) | Treated as a patient or an episode | Encounter identifier |
| Admission | An inpatient stay, from admit to discharge | Confused with encounter when ED registration is separate | Admission identifier |
| Episode of care | A clinically defined span covering related encounters | Assumed equal to an admission | Analyst-defined |
| MRN | Medical record number; the identifier local to a facility | Assumed unique across organizations | Local string |
| Enterprise ID | The master patient index identifier across a health system | Assumed stable across mergers | Local string |
| MPI | Master patient index; the system that links MRNs to a person | Treated as ground truth | System |
| HL7 v2 | Health Level Seven version 2; the dominant messaging standard for admissions, orders, and results | Assumed to be a data model | Message standard |
| ADT | Admit, discharge, transfer message feed | Assumed complete for all encounters | Message type |
| FHIR | Fast Healthcare Interoperability Resources; HL7's resource-oriented exchange standard | Assumed to be how the EHR stores data internally | REST resources |
| Observation | A measurement or assertion (lab, vital, survey item) | Treated as always numeric | FHIR resource; LOINC code |
| Condition | A diagnosis or problem entry | Treated as confirmed disease | FHIR resource; ICD or SNOMED code |
| MedicationRequest | An order for a medication | Treated as administration | FHIR resource; RxNorm |
| MedicationAdministration | A record that a dose was given | Treated as ingestion | FHIR resource; RxNorm |
| Flowsheet | The structured nursing documentation grid for vitals and assessments | Treated as device-grade data | Row and column identifiers |
| Problem list | The maintained list of active patient problems | Treated as a complete diagnosis history | SNOMED or ICD codes |
| Principal diagnosis | The condition chiefly responsible for the admission, as coded for billing | Treated as the clinical main problem | ICD-10-CM |
| Secondary diagnosis | An additional coded condition | Treated as comorbidity present on admission | ICD-10-CM; present-on-admission flag |
| Present on admission (POA) | A flag indicating a condition existed at admission | Ignored in comorbidity derivation | Y/N/U/W indicator |
| DRG | Diagnosis-related group; the inpatient payment category | Treated as a clinical severity measure | MS-DRG or APR-DRG |
| Copy-forward | Carrying text from a prior note into a new note | Treated as current observation | None |
| Note bloat | Progressive accumulation of copied text in notes | Treated as signal of complexity | None |
| Phenotype | An executable definition that identifies patients with a condition | Treated as a diagnosis | Algorithm |
| Rule-out | A diagnosis documented while being excluded | Counted as disease | None |
| Order-to-result lag | Time from order placement to result availability | Ignored in feature timing | Minutes or hours |
| Temporal leakage | Using information not available at the decision time | Mistaken for model skill | None |
| Index date | The anchor date for cohort entry and feature windows | Chosen arbitrarily | Date |
| Lookback window | The period before index used to establish history | Assumed to be complete | Days |
| Washout period | A period excluded to ensure incident rather than prevalent cases | Confused with lookback | Days |
| Data cutoff | The last date through which data are known complete | Ignored, producing right-censoring artefacts | Date |
| Right censoring | Follow-up truncated by the end of data | Treated as no event | None |
| MIMIC-IV | Medical Information Mart for Intensive Care IV; a public de-identified critical care EHR dataset | Treated as representative of all hospitals | Dataset |
| eICU-CRD | A public multi-site critical care dataset | Treated as current | Dataset |
| OMOP CDM | Observational Medical Outcomes Partnership Common Data Model | Treated as a substitute for source review | Data model |
| Source value | The value as recorded in the source system | Overwritten without retention | Native unit |
| Unit conversion | Mapping a source unit to a standard unit | Applied without versioning | UCUM |
| UCUM | Unified Code for Units of Measure | Treated as optional | Code system |
| Data provenance | The origin and processing history of a data element | Documented after the fact, if at all | Record |

### 4.1 Terms that are routinely conflated

Three pairs cause most of the confusion in practice:

- **Encounter versus admission.** An encounter is a registration; an admission is a stay. A patient can have an ED encounter and an inpatient encounter for the same episode.
- **Order versus administration.** The order says a clinician intended a drug. The administration says a nurse recorded giving it. Neither says the patient took it after discharge.
- **Diagnosis code versus disease.** The code says someone documented and coded a condition for a purpose. It does not say the condition was confirmed, active, or present on admission.

When you write an analysis plan, define which member of each pair you are using. If the plan does not distinguish them, a reviewer will assume the weakest reading.

---

## 5. The mechanics, step by step

This section is the working procedure. It assumes you have access to a relational extract of an EHR, with encounter, order, result, and note tables, and that you must produce a defensible analytic cohort. The procedure has 26 steps in five phases. Each step states its input, its output, and the reason it exists.

### Phase A — Establish the frame (steps 1-6)

**Step 1. Write the decision question in one sentence, with a population and a time horizon.**
Input: a stakeholder request. Output: one sentence of the form "Among [population], does [exposure or prediction] relate to [outcome] within [horizon]?"
Reason: every later choice — frame, label, lookback — is judged against this sentence. Without it, exclusions become arbitrary and the cohort drifts toward whatever is easy to compute.
Branch point: if the question cannot be phrased with a population and a horizon, it is not yet a data question. Return to the stakeholder.

**Step 2. Name the unit of analysis and write down why.**
Input: the question. Output: one of encounter, admission, episode, patient, or patient-period.
Reason: the unit determines the denominator and the independence assumption. Encounters within a patient are correlated; treating them as independent understates variance and inflates significance.
Branch point: if the outcome is a patient-level state (for example, one-year mortality), the unit is the patient and the encounter is a sampling frame. If the outcome is per-event (for example, sepsis bundle compliance), the unit is the encounter and you must cluster or use a robust variance estimator.

**Step 3. Fix the data cutoff date and the extraction date separately.**
Input: the source system's refresh schedule. Output: two dates recorded in the provenance file.
Reason: the cutoff is the last date through which events are believed complete; the extraction date is when the query ran. They differ, sometimes by weeks. A cohort that ends at the cutoff but is extracted six months later has a different completeness profile from one extracted the next day.
Reason, continued: right censoring depends on the cutoff, not the extraction date. Mixing them produces cohorts that appear to have lower event rates simply because later events had not arrived.

**Step 4. Enumerate the source tables and their grain.**
Input: the data dictionary. Output: a table listing each source table, its grain (one row per what), its primary key, and its known quirks.
Reason: most join errors are grain errors. Joining an encounter-level table to a result-level table without aggregation multiplies rows, and every subsequent count is wrong.
Test: for each table, run a query that counts rows and distinct keys. If they differ, the grain is not what the dictionary claims.

**Step 5. Resolve identity before anything else.**
Input: MRN, enterprise ID, and any MPI linkage table. Output: a person key with a documented resolution rule.
Reason: duplicate persons inflate prevalence and break longitudinality; merged persons create impossible timelines (events before birth, overlapping admissions at two hospitals).
Rule: prefer the enterprise ID; fall back to MRN plus demographic match; record every merge and every unresolved duplicate. Never silently drop duplicates.

**Step 6. Decide the site scope and record it.**
Input: the list of facilities contributing data. Output: a site variable on every row.
Reason: site is a confounder, a batch variable, and a data quality stratum. An analysis without a site variable cannot distinguish a clinical difference from a documentation difference.
Rule: if the extract spans sites, every descriptive table should be stratified by site at least once, even if the final model pools.

### Phase B — Build the cohort (steps 7-13)

**Step 7. Define the index event and the index date.**
Input: the question. Output: a rule that assigns exactly one index date per analytic unit.
Reason: the index date anchors the lookback and the follow-up. Multiple candidate index dates must be resolved by an explicit rule (first ever, first in period, most recent) rather than by whichever row the database returned first.
Branch point: for incident-condition studies, use first ever and add a washout; for prevalence, use any time in the period and no washout.

**Step 8. Apply inclusion criteria in a fixed order and count at each step.**
Input: the raw candidate set. Output: a flow diagram with counts at every step.
Reason: the order changes the counts and sometimes the population. Applying age before diagnosis gives a different cohort from diagnosis before age when the diagnosis is age-specific.
Rule: order criteria from most to least clinically fundamental, and never reorder between runs. Report the count removed at each step, not only the final n.

**Step 9. Handle repeat units explicitly.**
Input: the candidate set after inclusion. Output: one row per analytic unit, or a documented decision to keep repeats with clustering.
Reason: readmissions, repeat ED visits, and multiple lab draws are the norm. Deduplicating by patient changes the question; keeping all events changes the variance.
Rule: if you deduplicate, state the window (for example, 30 days) and the tie-break rule (earliest, most severe, longest stay).

**Step 10. Separate confirmed from suspected diagnoses.**
Input: coded diagnoses and, where available, problem list entries. Output: a phenotype with an explicit confirmation rule.
Reason: coded diagnoses include rule-outs, history, and coding-driven additions. A confirmed phenotype might require a code plus a treatment plus a lab, or a code in the principal position, or two codes on separate dates.
Test: sample 100 positive and 100 negative records and chart-review them. Report the positive predictive value (PPV) and, if feasible, sensitivity against a reference definition.

**Step 11. Decide how to handle missing eligibility variables.**
Input: the candidate set with missingness. Output: an explicit rule per variable: exclude, impute, or carry as a category.
Reason: excluding on a variable that is itself related to severity creates selection bias. In the worked example, excluding encounters without a lactate order removes a lower-mortality group and inflates the observed mortality rate.
Rule: never exclude silently. Report the excluded group's outcome rate alongside the included group's.

**Step 12. Define the exposure or the prediction time.**
Input: the question. Output: a timestamp at which the model or comparison is anchored.
Reason: this timestamp determines which data may be used. Everything after it is label, not feature.
Rule: write the rule as "features from [start] to [prediction time]; labels from [prediction time] to [horizon]." Enforce it with a query that rejects any feature with a timestamp after the prediction time.

**Step 13. Freeze the cohort definition as code, with a version.**
Input: all rules above. Output: a script in version control with a hash and a run date.
Reason: an EHR extract is a moving target. A cohort that cannot be re-run against a snapshot cannot be audited.
Rule: store the script, the snapshot identifier, the run date, the row counts, and the hash of the output.

### Phase C — Build the variables (steps 14-19)

**Step 14. Map source codes to standard concepts and record unmapped codes.**
Input: source code columns. Output: a mapping table with a standard concept, a mapping status, and a confidence.
Reason: source codes are local, versioned, and sometimes free text. Mapping loss is a data quality fact that must be quantified, not hidden.
Rule: count the proportion of rows that fail to map, by code and by site. A 30 percent unmapped rate in one site is a finding.

**Step 15. Normalize units and retain the source unit.**
Input: numeric values with unit strings. Output: a standard-unit value plus the original value and unit.
Reason: unit changes are silent and catastrophic. A creatinine recorded in micromoles per liter and analyzed as milligrams per deciliter is off by a factor of about 88.
Rule: use UCUM codes; keep the original; log every conversion; flag implausible values rather than deleting them.

**Step 16. Derive time-varying features at the correct grain.**
Input: event tables. Output: one row per unit per time window.
Reason: "last value before prediction time" and "worst value in the first 24 hours" are different features with different leakage profiles. The second is only legitimate if the prediction time is after 24 hours.
Rule: state the aggregation (last, first, min, max, mean) and the window for every feature. Undocumented aggregation is the second most common leakage source after result-time stamping.

**Step 17. Encode diagnoses and comorbidities with a validated scheme.**
Input: coded diagnoses. Output: comorbidity flags with a documented algorithm and version.
Reason: comorbidity indices are algorithms, not data. Different implementations of the same index produce different flags.
Rule: cite the exact implementation and the code set version. Record whether present-on-admission flags were used.

**Step 18. Extract from notes only with a validated pipeline.**
Input: note text. Output: structured variables with a validation report.
Reason: notes contain the clinical reasoning that codes omit, and also contain copy-forward, templates, and negation. An unvalidated extractor will learn the template.
Rule: report precision and recall on a held-out chart-reviewed sample; report the copy-forward rate; never use note text as a feature without checking that the text was authored in the relevant window.

**Step 19. Build the label with the same rigor as the exposure.**
Input: outcome events. Output: a binary or time-to-event label with a definition.
Reason: the label is the most consequential choice in the study and is often the least documented. Two papers on the same data can have different labels and incomparable results.
Rule: define the label from events after the prediction time, using a measurement or confirmation rule, and report its PPV against chart review.

### Phase D — Assess quality and bias (steps 20-23)

**Step 20. Profile completeness by site, year, and subgroup.**
Input: the analytic table. Output: a completeness matrix.
Reason: aggregate completeness hides the pattern. A field that is 95 percent complete overall can be 40 percent complete at one site in one year, which is exactly the stratum that drives a subgroup finding.
Rule: stratify by site, year, age band, sex, and payer at minimum.

**Step 21. Test for differential missingness with respect to the outcome.**
Input: the analytic table. Output: outcome rates for included versus excluded groups, by reason for exclusion.
Reason: this is the single most informative bias check available without chart review. If excluded patients differ on the outcome, complete-case analysis is biased and the direction is usually visible.
Rule: report the difference in outcome rate and, where possible, a bounding analysis.

**Step 22. Quantify the effect of each design choice.**
Input: the cohort script. Output: a sensitivity table showing the primary estimate under alternative rules.
Reason: a result that survives reasonable alternative definitions is more credible than one that does not, and the sensitivity table is the evidence.
Rule: vary at least the washout, the confirmation rule, and the missing-data rule. Report all three.

**Step 23. Standardize or stratify when comparing sites.**
Input: site-level rates and a case-mix profile. Output: crude and standardized rates.
Reason: sites differ in case mix. A site with a sicker population will look worse on a crude rate. Direct standardization to a common population removes the case-mix contribution.
Rule: state the standard population, the strata, and whether the standardized rates are directly or indirectly standardized.

### Phase E — Document and deliver (steps 24-26)

**Step 24. Write the provenance record.**
Input: everything above. Output: a one-page record containing source, snapshot, query, transformations, unit conversions, linkage, de-identification, and known limitations.
Reason: the record is what makes the analysis auditable and what a regulator or a partner will ask for first.
Rule: if a second analyst cannot reproduce the cohort from the record alone, the record is incomplete.

**Step 25. Run the reproducibility check.**
Input: the record and the script. Output: a second run with identical row counts and summary statistics.
Reason: the check catches environment drift, unstated manual steps, and non-deterministic ordering.
Rule: compare counts, means, and the hash of a sorted key column. Differences require explanation.

**Step 26. State what the data cannot answer, in the deliverable.**
Input: the question and the cohort. Output: a short "limits" section in the memo, not in an appendix.
Reason: the limits are part of the finding. A reader who does not see them will over-generalize.
Rule: name at least three things the data cannot answer, and say whether each is a threat to the primary conclusion or only to secondary ones.

### 5.1 Field-level mechanics: medications

Medication data is the family where the order-administration-ingestion distinction does the most damage, so it deserves its own procedure.

1. **Identify the table you actually have.** An order table (`medication_request`, `orders`, `prescriptions`) records intent. An administration table (`medication_administration`, `emar`, `mar`) records a nurse's recording of a dose. A dispensing table records what the pharmacy sent. These are three different facts.
2. **Check the ratio.** Count orders and administrations per drug per encounter. A ratio near 1.0 for inpatient drugs suggests you have administrations mislabeled as orders, or a very compliant unit. A ratio well below 1.0 is expected for as-needed drugs and for cancelled orders.
3. **Resolve the drug identity.** Orders are often free text or local formulary codes. Map to a standard concept (RxNorm in the US, or the OMOP standard concept) and record the mapping status. Ingredient-level and clinical-drug-level concepts answer different questions: "was the patient on a statin" is ingredient-level, "was the patient on atorvastatin 40 mg" is clinical-drug-level.
4. **Resolve the dose.** Dose strings are messy: "1 tab", "40 mg", "2 puffs", sliding scales. Convert to a numeric dose with a unit only where the string permits. Do not impute a standard dose from the drug name.
5. **Resolve the route.** Route affects exposure interpretation. An order for a drug by mouth in a patient who is nothing-by-mouth is a signal that the order was not followed.
6. **Build the exposure interval.** For chronic drugs, exposure runs from the first administration to the last administration plus a grace period. The grace period is a modelling assumption, not a fact in the data.
7. **Handle as-needed and tapering orders.** A PRN (pro re nata, as needed) order does not imply continuous exposure. A taper implies a changing dose that a single "exposed" flag hides.
8. **Record the provenance of the exposure definition.** The window, the grace period, and the concept level are all choices. Two analyses of the same data with different choices will disagree, and both can be defensible.

The failure to avoid: reporting "percent of patients on drug X" from an order table and calling it treatment. That number is treatment intent plus cancelled orders plus refusals, and it is almost always higher than actual exposure.

### 5.2 Field-level mechanics: laboratory results

1. **Separate order, specimen, result, and verification times.** The specimen time is the physiologic time. The result time is when the value became available. The verification time is when a clinician signed off. Use specimen time for physiology, result time for availability analysis, and never verification time for either.
2. **Capture the unit as recorded and the unit as standardized.** Store both. A conversion without the original is unauditable.
3. **Check the reference range.** Reference ranges encode the lab's method and population. A value inside the reference range at one lab may be outside at another. Ranges also change when an assay changes.
4. **Check for method or analyzer change.** If the source has an analyzer or method field, plot values by analyzer over time. If it does not, plot the distribution by month and look for a step change.
5. **Distinguish censored values.** Values reported as "less than 0.5" or "greater than 200" are censored, not missing. Treating them as missing loses information; treating them as the boundary value biases the mean.
6. **Handle duplicate results.** The same specimen can produce multiple result rows after a re-run or a dilution. The last verified result is usually the clinical one, but "usually" is an assumption to state.
7. **Decide the aggregation before looking at the outcome.** Choosing "max" because it produces a stronger association is a form of p-hacking. Pre-specify last, first, min, max, or mean.
8. **Flag implausible values rather than deleting them.** A potassium of 12 mmol/L is either a haemolysed sample or a data error. Deleting it changes the denominator; flagging it preserves the audit trail.

### 5.3 Field-level mechanics: vital signs and flowsheets

1. **Identify the row identifiers.** Flowsheet rows are local codes with local display names. The same physiologic concept can appear in several rows (for example, "BP systolic" from a monitor and from a manual cuff).
2. **Separate device-derived from manually entered values.** A monitor-sourced heart rate and a nurse-entered heart rate have different error profiles and different sampling rates.
3. **Check for defaults and carried-forward values.** Some systems pre-populate a field with the last value. A series of identical values at regular intervals is a red flag for a carried-forward default.
4. **Check the timestamp semantics.** A flowsheet row may be stamped at the documentation time, at the rounding time, or at the measurement time, depending on the build.
5. **Aggregate to a clinically meaningful window.** A single blood pressure is noisy. A 15-minute minimum is more stable and more defensible for a deterioration model than a single reading.

### 5.4 Field-level mechanics: notes

1. **Establish authorship and time.** Who wrote the note, in what role, and when. A resident note at 3 a.m. and an attending addendum at 8 a.m. are different evidence.
2. **Detect copy-forward before extracting.** Compute similarity between consecutive notes for the same patient. A high similarity score means the text is partly inherited. Several public tools and simple n-gram methods can do this.
3. **Handle negation and uncertainty.** "No evidence of pneumonia" and "cannot rule out pneumonia" are not pneumonia. Negation detection is a solved problem in general but fails on local shorthand.
4. **Handle family history and hypotheticals.** "Mother had breast cancer" is not the patient's diagnosis. Section-level context matters.
5. **Validate against chart review.** Report precision and recall on a held-out sample, with the sample size and the adjudication process.
6. **Never use note length or section count as a clinical severity feature** without checking that it is not a proxy for author or template.

### 5.5 Field-level mechanics: problem lists and diagnoses

1. **Separate the problem list from the encounter diagnosis list.** The problem list is a maintained clinical list; the encounter diagnosis list is a billing artefact. They disagree, often.
2. **Check for duplicated concepts at different specificity.** "Diabetes" and "type 2 diabetes with renal complications" can both be active. Counting both double-counts.
3. **Use the present-on-admission flag where available.** It separates comorbidity from complication, which changes the meaning of a comorbidity index.
4. **Check for a history-of suffix.** ICD-10-CM has specific codes for history of a condition, and SNOMED CT has context attributes. A history code is not active disease.
5. **Check for rule-out documentation patterns.** A diagnosis that appears once, on the admission with a workup, and never again is more likely a rule-out than a confirmed condition. Frequency and persistence are weak but useful signals.
6. **Record the code set version.** A phenotype written against ICD-10-CM FY2025 may behave differently under FY2026, which added 487 codes, revised 38, and deleted 28 effective 1 October 2025.

### 5.6 Quantitative checks to run inside the procedure

These checks are cheap, and each one catches a class of error that is expensive to find later. Run them in this order after step 19 and before step 20.

| Check | Computation | Passing signal | What failure usually means |
|---|---|---|---|
| Key uniqueness | `COUNT(*)` versus `COUNT(DISTINCT key)` per table | Equal | Wrong grain assumption |
| Referential integrity | Left join children to parents, count orphans | Zero orphans, or a documented reason | Incomplete extract or wrong join key |
| Temporal ordering | Count events before birth or after death | Zero | Date shifting error or identity merge |
| Length of stay | `discharge_date - admit_date` distribution | Median plausible for the service | Duplicate admissions or timezone shift |
| Duplicate encounters | Same patient, same day, same service | Low rate, or explained | Registration artefacts |
| Unit distribution | Value quantiles by unit string | Units cluster as expected | Mixed units in one column |
| Value plausibility | Percent outside physiologic bounds | Small and stable across sites | Entry errors or default values |
| Code coverage | Percent of rows mapped to a standard concept | High and similar across sites | Local codes not mapped |
| Date completeness | Events per month across the period | Smooth, no cliff at the end | Right censoring or a system migration |
| Site balance | Encounters per site per year | Stable, or explained | A site joined or left the network |

Two of these deserve emphasis because they are routinely skipped.

**Temporal ordering before and after identity resolution.** A patient with two merged MRNs can have admissions that overlap in time at two hospitals. The merge is correct, the overlap is real, and the naive interpretation is that the data are corrupt. Running the check before and after the merge distinguishes the two cases.

**Date completeness at the end of the period.** A cliff in the final two months is almost always a data arrival problem, not a clinical change. If the analysis includes those months without adjustment, incidence will appear to fall and any trend estimate will be wrong. The fix is either to trim the tail or to model the reporting delay, and the choice must be documented.

### 5.7 Common derivations and the assumptions they hide

Every derived variable in an EHR pipeline encodes an assumption. The table below lists the derivations that appear in almost every project and the assumption a reviewer will challenge.

| Derived variable | Usual construction | Hidden assumption | Challenge question |
|---|---|---|---|
| Length of stay | Discharge minus admit, in days | Both dates are in the same time zone and the stay is continuous | Does a transfer split the stay? |
| In-hospital mortality | Discharge disposition equals died | Disposition coding is complete and timely | Are hospice discharges coded as deaths? |
| 30-day readmission | Any admission within 30 days of discharge | The patient returns to this network | What is the out-of-network readmission rate? |
| Comorbidity count | Elixhauser or Charlson from codes | Codes present on the encounter are complete | Were present-on-admission flags used? |
| Time to antibiotic | First administration minus order or arrival | The first administration is the clinically relevant one | Which clock anchors the start? |
| ICU admission | Presence of an ICU encounter | The ICU encounter is distinguishable from a step-down unit | Is the unit taxonomy local? |
| Adherence | Days supplied divided by days in period | Dispensing equals consumption | Is this a pharmacy claim or a MAR? |
| Deterioration label | Any vital sign breach in the next N hours | The breach is clinically meaningful | How many breaches are artefactual? |

The pattern is that the derivation is easy and the assumption is hard. Writing the assumption next to the derivation in the data dictionary is the cheapest quality intervention available.

A second pattern is worth naming: derived variables tend to be re-used long after their assumption has expired. A length-of-stay field built when the system had one facility is still in the pipeline after a merger added three, and nobody revisits it. Version the derivation, not only the query.

### 5.8 Edge cases the procedure has to survive

The 26 steps assume clean inputs. Reality delivers seven recurring edge cases, and each has a defined handling rule. Apply the rule, record which rule you applied, and report the count affected.

**Null and missing values at each step.**

| Where | Symptom | Handling rule | What to report |
|---|---|---|---|
| Step 4, grain check | Key column null in 6% of rows | Do not treat null keys as duplicates; quarantine them | Null-key count and source system |
| Step 8, inclusion | Age null because birth date missing | Create an "unknown age" bucket; do not exclude silently | Count and outcome rate of the unknown-age group |
| Step 11, eligibility | Eligibility variable null | Keep with a missing indicator; never drop | Excluded-group outcome rate if you do drop |
| Step 16, feature derivation | No value in the window | Emit null, not zero and not the cohort mean | Percent null per feature per site |
| Step 19, label | Outcome not observable before cutoff | Right-censor; do not code as no-event | Person-time censored |

**Conflicting sources for the same fact.** The EHR says the patient has heart failure; the claims file does not; the registry does. Three rules apply:

1. Prefer the source closest to the clinical event for clinical facts (EHR over claims).
2. Prefer the source with a legal mandate for population facts (registry over EHR for incidence).
3. When the sources disagree, carry both and model the disagreement as a variable, rather than choosing one silently.

The practical implementation is a `source_priority` column and a `concordance_flag`, so a sensitivity analysis can be run by source.

**Unit variants.** A creatinine recorded in micromoles per litre at one site and milligrams per decilitre at another differs by a factor of about 88.4. The handling rule is to store the source value, the source unit, the conversion factor, and the standardized value, and to reject any row where the conversion factor is not in a controlled list. A conversion applied without a factor recorded is unauditable.

**Code-system variants within the study period.** ICD-10-CM changes every 1 October. A study spanning September and October 2025 contains codes from FY2025 and FY2026, and 28 codes were deleted in the transition. The handling rule is to map both years to a common concept set at extraction time, keep the source code and its fiscal year, and report the proportion of the cohort whose capture depends on a code that changed.

**Late-arriving data.** A lab drawn on 30 December may be verified on 3 January and extracted on 10 January. The handling rule is to define the data cutoff as the last date with a stable arrival curve, to exclude the tail or model the lag, and never to treat a missing result as a normal result.

**Partial identifiers.** A record with an MRN but no enterprise ID cannot be linked to the longitudinal history. The handling rule is to keep the record in encounter-level analyses, exclude it from patient-level analyses, and report the proportion in each category by site.

**Duplicate records.** Duplicates arise from registration errors, merged identities, and feed replays. The handling rule is to define the key before counting, to report duplicates by key, and to resolve conflicts with a stated tie-break (most recent, most complete, or source priority). Never deduplicate on a key you chose after seeing the counts.

**Timezone and clock boundaries.** Multi-site extracts often mix local time and UTC, and daylight-saving transitions create an hour that occurs twice or not at all. The handling rule is to convert everything to a single timezone at extraction, record the source timezone per site, and test for duplicate or missing local hours at the transition dates.

### 5.9 Field-level validation rules

A validation rule is a test with a pass condition, a severity, and an action. Rules that only log are decoration; rules that only delete are dangerous. The table below is the minimum set for an EHR extract, with the action each rule should trigger.

| Rule | Test | Severity | Action on failure |
|---|---|---|---|
| Key integrity | `COUNT(*) = COUNT(DISTINCT encounter_id)` | High | Quarantine duplicates; do not proceed |
| Referential integrity | Every `patient_id` in `encounter` exists in `patient` | High | Quarantine orphans; report by source |
| Temporal ordering | `admit_date <= discharge_date` | High | Quarantine; investigate the feed |
| Plausible age | `0 <= age <= 120` | Medium | Flag; do not exclude |
| Plausible length of stay | `0 <= LOS <= 365` days | Medium | Flag; review the top 20 |
| Unit presence | Numeric result has a non-null unit | High | Quarantine the value; do not assume a unit |
| Unit conversion validity | Unit is in the controlled conversion list | High | Quarantine; add the unit to the list after review |
| Code validity | Diagnosis code exists in the FY in force | Medium | Flag; attempt a crosswalk; report the unmapped rate |
| Value plausibility | Result inside a wide physiologic band | Low | Flag; retain |
| Duplicate event | Same patient, same concept, same timestamp, same value | Medium | Collapse; keep the source rows |
| Cross-field logic | Discharge disposition "expired" implies a death date | Medium | Flag; investigate the source |
| Site completeness | Each site has above-threshold completeness for critical fields | High | Block the site-level comparison |

Two design rules make the set usable. First, every rule has an owner and a documented disposition, so a failure is a decision rather than a queue item. Second, the rules run before analysis and their results are stored with the dataset, so a later reviewer can see what was caught.

### 5.10 Decision table: handling a missing eligibility variable

Step 11 has more than two defensible outcomes. Choose by the mechanism, not by convenience.

| Mechanism | Evidence | Action | Sensitivity analysis | Report |
|---|---|---|---|---|
| Structural (field inapplicable) | Missing only in a group where it cannot apply | Exclude from denominator | None needed | Denominator definition |
| MCAR | Missingness unrelated to group, site, or outcome | Complete case | None needed beyond precision | Percent complete |
| MAR | Missingness predicted by observed variables | Multiple imputation including the outcome | Compare with complete case | Imputation model and variables |
| MNAR, direction known | Missing group has worse outcomes | Keep with missing indicator | Bounding analysis with stated bounds | Both estimates and the bound |
| MNAR, direction unknown | Missingness concentrated in one site | Keep; add site interaction | Site-stratified analysis | Site-specific estimates |
| Workflow-driven | Missingness tracks shift, unit, or template | Model as a site-period effect | Compare periods before and after the workflow change | Workflow change dates |

The rule that overrides the table: if the missing variable is the exposure or the outcome rather than a covariate, no missing-data method rescues the design. Redesign or change the source.

### 5.11 Branch points summarized

| Branch | Condition | Path A | Path B |
|---|---|---|---|
| Unit of analysis | Outcome is patient-level state | Patient-level cohort | Encounter-level with clustering |
| Index date | Incident condition | First ever plus washout | Any in period, no washout |
| Missing eligibility | Missingness related to severity | Keep and analyze as a category | Exclude and bound the bias |
| Confirmation | Chart review available | Validated phenotype with PPV | Code-only with explicit caveat |
| Site comparison | Case mix differs | Direct standardization | Stratified reporting |
| Note use | No validated extractor | Exclude notes | Use only as a sensitivity analysis |

---

## 6. Worked example

### 6.1 The question

A four-hospital academic health system wants to know whether its sepsis screening flag is usable as a cohort definition for a quality-improvement evaluation. The clinical team asks: among adults admitted with a sepsis-coded encounter, what is in-hospital mortality, and does it differ by hospital after accounting for age?

The analyst has an EHR extract covering 2022-01-01 through 2024-12-31, extracted on 2025-03-14, with a data cutoff of 2024-12-31.

### 6.2 Candidate set and exclusions

| Step | Criterion | Removed | Remaining |
|---|---|---:|---:|
| 0 | Sepsis-coded encounter or positive screening flag, 2022-2024 | — | 18,420 |
| 1 | Age under 18 at encounter | 1,105 | 17,315 |
| 2 | Not admitted (ED discharge or outpatient only) | 3,260 | 14,055 |
| 3 | Repeat encounter within 30 days of a prior included encounter | 942 | 13,113 |
| 4 | No lactate result from 24 hours before to 6 hours after admission | 2,388 | 10,725 |

Check the arithmetic: 18,420 − 1,105 = 17,315. 17,315 − 3,260 = 14,055. 14,055 − 942 = 13,113. 13,113 − 2,388 = 10,725.

The analyst is tempted to stop here and report on 10,725 encounters. That would be a mistake, and the next step shows why.

### 6.3 The missing-lactate problem

Lactate is ordered when a clinician suspects hypoperfusion. Encounters without a lactate order are not a random subset. In this extract they are lower-acuity, because the screening flag fired on a coded diagnosis that the clinical team did not act on.

| Group | Encounters | Deaths | Mortality |
|---|---:|---:|---:|
| Lactate measured (included) | 10,725 | 1,201 | 11.2% |
| Lactate not measured (excluded) | 2,388 | 224 | 9.4% |
| All admitted, deduplicated | 13,113 | 1,425 | 10.9% |

Check the arithmetic:
- Included: 1,201 / 10,725 = 0.11199, or 11.2%.
- Excluded: 224 / 2,388 = 0.09380, or 9.4%.
- All: (1,201 + 224) / (10,725 + 2,388) = 1,425 / 13,113 = 0.10867, or 10.9%.

The complete-case analysis overstates mortality by 1.8 percentage points (11.2 − 9.4) relative to the excluded group, and by 0.33 percentage points relative to the full cohort. That is not a rounding difference: at a system level with 13,113 admissions, 0.33 percentage points is about 43 deaths attributed to the wrong denominator.

The correct move is not to exclude the 2,388. It is to keep them, set lactate to missing, and either analyze with a missing indicator or impute. The exclusion is reported here because it is what most analysts actually do, and the file's purpose is to make that error visible.

### 6.4 Confidence interval on the primary rate

For the complete-case rate of 11.2 percent with n = 10,725:

- Standard error = sqrt(p(1 − p) / n) = sqrt(0.112 × 0.888 / 10,725) = sqrt(0.099456 / 10,725) = sqrt(0.0000092733) = 0.003045.
- 95 percent confidence interval = 0.112 ± 1.96 × 0.003045 = 0.112 ± 0.005968, or 10.6% to 11.8%.

Report it as 11.2% (95% CI 10.6-11.8). A rate without an interval invites false precision, and the interval is what tells a reader whether a two-point difference between hospitals is distinguishable from noise.

### 6.5 Screening flag performance

A random sample of 200 flagged encounters was chart-reviewed by two clinicians with a third adjudicating disagreements.

| | Chart review: sepsis | Chart review: not sepsis | Total |
|---|---:|---:|---:|
| Flag positive | 118 | 82 | 200 |

Positive predictive value = 118 / 200 = 0.590, or 59.0%.
Wilson 95 percent confidence interval: 52.1% to 65.6%.

Interpretation: about 4 in 10 flagged encounters are not sepsis by chart review. The flag is a screening aid, not a cohort definition. Using it as a label means that 41 percent of the "cases" are something else, which will attenuate any association toward the null and will make the model learn the flag's quirks rather than sepsis physiology.

Sensitivity cannot be computed from this sample, because sampling only flagged encounters gives no information about unflagged true cases. Estimating sensitivity requires a random sample of all admissions, chart-reviewed, which is a larger and more expensive design. Saying so is part of the analysis.

### 6.6 Site comparison and the standardization trap

Crude in-hospital mortality by hospital:

| Hospital | Encounters | Deaths | Crude mortality |
|---|---:|---:|---:|
| A | 4,290 | 493 | 11.49% |
| B | 3,180 | 340 | 10.69% |
| C | 2,005 | 238 | 11.87% |
| D | 1,250 | 130 | 10.40% |
| Total | 10,725 | 1,201 | 11.20% |

Check: 493/4,290 = 0.114919; 340/3,180 = 0.106918; 238/2,005 = 0.118703; 130/1,250 = 0.104000. Sum of deaths 493 + 340 + 238 + 130 = 1,201. Sum of encounters 4,290 + 3,180 + 2,005 + 1,250 = 10,725.

On the crude rate, hospital C looks worst and hospital D looks best. Hospital A looks better than C. Now stratify by age band and standardize to a common population with weights 0.40 (18-64), 0.35 (65-79), and 0.25 (80 and older).

| Hospital | 18-64 n | 18-64 deaths | 65-79 n | 65-79 deaths | 80+ n | 80+ deaths |
|---|---:|---:|---:|---:|---:|---:|
| A | 1,802 | 122 | 1,480 | 201 | 1,008 | 170 |
| B | 1,431 | 88 | 1,081 | 148 | 668 | 104 |
| C | 742 | 46 | 762 | 104 | 501 | 88 |
| D | 425 | 24 | 475 | 62 | 350 | 44 |

Band-specific rates:

| Hospital | 18-64 | 65-79 | 80+ |
|---|---:|---:|---:|
| A | 6.77% | 13.58% | 16.87% |
| B | 6.15% | 13.69% | 15.57% |
| C | 6.20% | 13.65% | 17.56% |
| D | 5.65% | 13.05% | 12.57% |

Check one cell: 122 / 1,802 = 0.06770. 201 / 1,480 = 0.13581. 170 / 1,008 = 0.16865. The other hospitals follow the same arithmetic and all sums reconcile to the site totals.

Direct standardization, for hospital A:
- 0.40 × 6.77 = 2.708
- 0.35 × 13.58 = 4.753
- 0.25 × 16.87 = 4.2175
- Sum = 11.6785, or 11.68%.

Hospital B: 0.40 × 6.15 = 2.460; 0.35 × 13.69 = 4.7915; 0.25 × 15.57 = 3.8925; sum = 11.144, or 11.14%.
Hospital C: 0.40 × 6.20 = 2.480; 0.35 × 13.65 = 4.7775; 0.25 × 17.56 = 4.390; sum = 11.6475, or 11.65%.
Hospital D: 0.40 × 5.65 = 2.260; 0.35 × 13.05 = 4.5675; 0.25 × 12.57 = 3.1425; sum = 9.970, or 9.97%.

| Hospital | Crude | Standardized | Change |
|---|---:|---:|---:|
| A | 11.49% | 11.68% | +0.19 |
| B | 10.69% | 11.14% | +0.45 |
| C | 11.87% | 11.65% | −0.22 |
| D | 10.40% | 9.97% | −0.43 |

The rank reversal is the point. On the crude rate, A (11.49%) is better than C (11.87%). After standardization, A (11.68%) is slightly worse than C (11.65%). The crude comparison was confounded by age: hospital A treats a younger population, and younger patients die less often. A quality dashboard built on the crude rate would reward the hospital with the healthiest case mix.

### 6.7 Variant case: the same cohort with a patient-level denominator

The question in 6.1 was framed per encounter ("among adults admitted with a sepsis-coded encounter"). Suppose the quality committee instead asks: "what is a patient's risk of dying if admitted with sepsis at each hospital?" That is a patient-level question, and the unit of analysis must change.

Building the variant: keep only the first included encounter per patient. The 10,725 encounters belong to 8,210 patients, so 2,515 encounters are repeats.

| Quantity | Encounter-level (primary) | Patient-level (variant) |
|---|---:|---:|
| Units | 10,725 | 8,210 |
| Deaths | 1,201 | 986 |
| Mortality | 11.20% | 12.01% |

Check: 10,725 − 8,210 = 2,515 repeat encounters. 1,201 − 986 = 215 deaths among repeats.
Repeat mortality = 215 / 2,515 = 0.0855, or 8.55%.
Patient-level mortality = 986 / 8,210 = 0.12010, or 12.01%.

The repeat encounters have much lower mortality than first encounters (8.55% versus 12.01%). That is clinically plausible: a patient who survives the first admission, is discharged, and returns is a survivor being readmitted, often for a less lethal problem.

Site-level under the variant:

| Hospital | Patients | Deaths | Patient-level mortality | Encounter-level crude |
|---|---:|---:|---:|---:|
| A | 3,310 | 401 | 12.11% | 11.49% |
| B | 2,450 | 272 | 11.10% | 10.69% |
| C | 1,540 | 198 | 12.86% | 11.87% |
| D | 910 | 115 | 12.64% | 10.40% |
| Total | 8,210 | 986 | 12.01% | 11.20% |

Check: 3,310 + 2,450 + 1,540 + 910 = 8,210. 401 + 272 + 198 + 115 = 986.
Check rates: 401/3,310 = 0.12115. 272/2,450 = 0.11102. 198/1,540 = 0.12857. 115/910 = 0.12637.

Now derive the repeat-encounter profile that explains the movement:

| Hospital | Encounters | Patients | Repeats | Repeat deaths | Repeat mortality |
|---|---:|---:|---:|---:|---:|
| A | 4,290 | 3,310 | 980 | 92 | 9.39% |
| B | 3,180 | 2,450 | 730 | 68 | 9.32% |
| C | 2,005 | 1,540 | 465 | 40 | 8.60% |
| D | 1,250 | 910 | 340 | 15 | 4.41% |
| Total | 10,725 | 8,210 | 2,515 | 215 | 8.55% |

Check: 980 + 730 + 465 + 340 = 2,515. 92 + 68 + 40 + 15 = 215.
Check: 980/4,290 = 0.2284 of A's encounters are repeats; 340/1,250 = 0.2720 of D's. D has the highest repeat share and the lowest repeat mortality.

### 6.8 What changes, and why

| Ranking basis | Best | Second | Third | Worst |
|---|---|---|---|---|
| Encounter-level crude | D (10.40%) | B (10.69%) | A (11.49%) | C (11.87%) |
| Encounter-level age-standardized | D (9.97%) | B (11.14%) | C (11.65%) | A (11.68%) |
| Patient-level crude | B (11.10%) | A (12.11%) | D (12.64%) | C (12.86%) |

Hospital D moves from best on both encounter-level measures to third of four on the patient-level measure. The mechanism is arithmetic, not clinical: D's 340 repeat encounters carry only 15 deaths (4.41%), so they dilute its per-encounter rate. Once each patient contributes one observation, the dilution disappears.

Hospital A moves from worst on the standardized encounter measure to second on the patient measure, for the opposite reason: its repeats are comparatively deadly (9.39%), so removing them helps A's standing.

The difference between the two headline rates is 0.81 percentage points (12.01 − 11.20), and the difference between the two rankings is two positions for D. A committee that saw only one number would make a different decision about where to send patients.

### 6.9 Which denominator I would report, and why

Both are computable from the same extract, and the choice is a statement about the decision:

- **Report encounter-level** when the decision is about admissions: a quality measure, a bundle-compliance target, a capacity plan, or a payment rule defined per discharge. The measure specification usually dictates the unit, and changing it breaks comparability with published benchmarks.
- **Report patient-level** when the decision is about people: which hospital a patient should choose, what a patient's risk is, or how a cohort should be followed longitudinally. Repeat admissions are not independent observations of the same risk, and treating them as such both dilutes the rate and understates the variance.

For the committee's actual question ("what is a patient's risk"), the patient-level estimate is the correct primary and the encounter-level estimate is the measure-aligned sensitivity. For the original quality-improvement question in 6.1, the order reverses.

The general rule: name the decision, derive the unit from the decision, and report the alternative as a sensitivity analysis with the rank change stated explicitly. A deliverable that reports one rate without the unit definition invites the reader to assume the other.

One more check a reviewer will ask for: the 2,515 repeats are not a random subset, and the 215 deaths among them are not distributed evenly across sites. The site-level repeat mortality in 6.7 is therefore itself a quality signal worth investigating, because a site with a 4.41% repeat mortality may be readmitting patients who did not need admission, which is a different problem from a low mortality rate.

### 6.10 What this example establishes

- The cohort definition is a chain of choices, and the exclusion with the largest effect was the one that looked most innocent (requiring a lactate).
- The screening flag's PPV of 59.0% makes it unusable as a label without qualification.
- The site ranking depends on whether case mix is adjusted, and the reversal is small but decision-relevant.
- Every number above is reproducible from the counts in the tables, which is the standard the exercises hold you to.

---

## 7. Code, query, or calculation

### 7.1 SQL: cohort construction with exclusion accounting

Assumptions: a relational schema with `encounter`, `patient`, `diagnosis`, and `lab` tables; PostgreSQL syntax; dates stored as `date`; `lab.lab_name` is a local string.

```sql
-- Cohort construction with exclusion accounting.
-- Assumptions: PostgreSQL; one row per encounter in encounter;
-- lab.encounter_id may be null for outpatient labs.

WITH candidates AS (
    SELECT e.encounter_id,
           e.patient_id,
           e.facility_id,
           e.admit_date,
           e.discharge_date,
           e.discharge_disposition,
           p.birth_date,
           DATE_PART('year', AGE(e.admit_date, p.birth_date)) AS age_years
    FROM encounter e
    JOIN patient p USING (patient_id)
    WHERE e.admit_date >= DATE '2022-01-01'
      AND e.admit_date <  DATE '2025-01-01'
      AND (e.sepsis_flag = 1
           OR EXISTS (SELECT 1 FROM diagnosis d
                      WHERE d.encounter_id = e.encounter_id
                        AND d.icd10_code LIKE 'A41%'))
),
step1 AS (
    SELECT * FROM candidates WHERE age_years >= 18
),
step2 AS (
    SELECT * FROM step1
    WHERE discharge_disposition NOT IN ('ED_DISCHARGE', 'OUTPATIENT_ONLY')
),
step3 AS (
    SELECT s.*,
           ROW_NUMBER() OVER (PARTITION BY patient_id ORDER BY admit_date) AS rn,
           LAG(admit_date) OVER (PARTITION BY patient_id ORDER BY admit_date) AS prior_admit
    FROM step2 s
),
step4 AS (
    SELECT * FROM step3
    WHERE prior_admit IS NULL OR (admit_date - prior_admit) > 30
),
step5 AS (
    SELECT s.*,
           EXISTS (SELECT 1 FROM lab l
                   WHERE l.encounter_id = s.encounter_id
                     AND l.lab_name = 'Lactate'
                     AND l.result_time >= s.admit_date - INTERVAL '24 hours'
                     AND l.result_time <= s.admit_date + INTERVAL '6 hours') AS has_lactate
    FROM step4 s
)
SELECT '0_candidates' AS step, COUNT(*) AS n FROM candidates
UNION ALL SELECT '1_age_ge_18', COUNT(*) FROM step1
UNION ALL SELECT '2_admitted', COUNT(*) FROM step2
UNION ALL SELECT '3_dedup_30d', COUNT(*) FROM step3 WHERE prior_admit IS NULL OR (admit_date - prior_admit) > 30
UNION ALL SELECT '4_lactate', COUNT(*) FROM step5 WHERE has_lactate
ORDER BY step;
```

Expected output (matches section 6.2):

```text
step              n
0_candidates      18420
1_age_ge_18       17315
2_admitted        14055
3_dedup_30d       13113
4_lactate         10725
```

> **Note:** the `step3` filter in the final `UNION ALL` is repeated rather than reused because a window function cannot be filtered in the same query level. In production, materialize `step3` as a table so the count and the downstream join use the same rows.

### 7.2 Python: direct standardization and rank reversal

Assumptions: pandas 2.x; the counts in section 6.6; no missing strata.

```python
import pandas as pd

# Counts transcribed from the stratified table in section 6.6.
rows = [
    # site, age_band, n, deaths
    ("A", "18-64", 1802, 122), ("A", "65-79", 1480, 201), ("A", "80+", 1008, 170),
    ("B", "18-64", 1431,  88), ("B", "65-79", 1081, 148), ("B", "80+",  668, 104),
    ("C", "18-64",  742,  46), ("C", "65-79",  762, 104), ("C", "80+",  501,  88),
    ("D", "18-64",  425,  24), ("D", "65-79",  475,  62), ("D", "80+",  350,  44),
]
df = pd.DataFrame(rows, columns=["site", "age_band", "n", "deaths"])
df["rate"] = df["deaths"] / df["n"]

std_weights = {"18-64": 0.40, "65-79": 0.35, "80+": 0.25}

crude = df.groupby("site")[["n", "deaths"]].sum()
crude["crude_rate"] = crude["deaths"] / crude["n"]

df["w_rate"] = df["rate"] * df["age_band"].map(std_weights)
std = df.groupby("site")["w_rate"].sum().rename("std_rate")

out = crude.join(std)
out["crude_pct"] = (out["crude_rate"] * 100).round(2)
out["std_pct"] = (out["std_rate"] * 100).round(2)
print(out[["n", "deaths", "crude_pct", "std_pct"]])
```

Expected output:

```text
        n  deaths  crude_pct  std_pct
site
A    4290     493      11.49    11.68
B    3180     340      10.69    11.14
C    2005     238      11.87    11.65
D    1250     130      10.40     9.97
```

### 7.3 A hand calculation you should be able to do in an interview

Given 1,201 deaths in 10,725 encounters, compute the rate and its 95 percent interval without a computer:

1. p = 1,201 / 10,725. Round to 0.112.
2. p(1 − p) = 0.112 × 0.888 = 0.099456.
3. Divide by n: 0.099456 / 10,725 ≈ 0.000009273.
4. Square root: ≈ 0.003045.
5. Multiply by 1.96: ≈ 0.005968.
6. Interval: 0.1060 to 0.1180, reported as 10.6% to 11.8%.

If you can do this, you can sanity-check any dashboard that reports a rate.

---

## 8. Data, standards, and regulatory context

### 8.1 Which standard applies to which part of the EHR

| Layer | Standard | What it constrains | What it does not |
|---|---|---|---|
| Message transport | HL7 v2.5.1 and later | Segment and field structure for ADT, orders, results | Clinical meaning; local code sets still vary |
| Exchange API | HL7 FHIR R4 (4.0.1) | Resource structure, references, search | Completeness of the source; FHIR is an API, not an archive |
| US profiles | US Core Implementation Guide, currently the 9.0.0 line for USCDI v6 | Which elements must be present for certified exchange | Data quality; a conformant resource can be clinically wrong |
| Documents | HL7 CDA R2 | Document structure for summaries and reports | Semantic consistency across authors |
| Terminology | LOINC, SNOMED CT, RxNorm, ICD-10-CM | Concept identity | Whether the concept was correctly applied |
| Units | UCUM | Machine-readable unit expression | Whether the source unit was recorded correctly |
| Imaging | DICOM | Image and metadata object structure | Whether the study is complete or representative |
| Analytic model | OMOP CDM 5.4 or 5.5 | Table and column structure for observational analytics | Mapping quality; the ETL is still yours |

### 8.2 What is binding and what is convention

In the United States, several requirements are binding on certified health information technology and on actors subject to the information blocking rules:

- **ONC Health IT Certification Program.** The HTI-1 final rule, published 9 January 2024 and effective 8 February 2024, replaced the clinical decision support criterion with a Decision Support Intervention criterion at 45 CFR 170.315(b)(11) and added transparency requirements for predictive algorithms. The rule is binding on developers seeking certification, not on analysts using the data.
- **Information blocking.** The 21st Century Cures Act information blocking provisions, implemented at 45 CFR Part 171, constrain how actors may interfere with access to electronic health information. This affects data access timelines, which is a practical constraint on analysis.
- **HIPAA Privacy and Security Rules.** 45 CFR Parts 160 and 164 govern use and disclosure of protected health information (PHI). De-identification is defined at 45 CFR 164.514.
- **42 CFR Part 2.** Applies to substance use disorder records from federally assisted programs and is more restrictive than HIPAA. An EHR extract that includes such records may need separate consent.
- **State law.** Many states impose stricter requirements than HIPAA on mental health, HIV, and reproductive health data, and on minor consent.

Conventions, not law:

- Using OMOP CDM for multi-site analytics is a community convention, not a regulatory requirement. It is nonetheless expected by many funders and by the OHDSI (Observational Health Data Sciences and Informatics) community.
- Using LOINC for lab observations is a convention that US Core and certification make near-mandatory in practice.
- Present-on-admission flags are a billing convention introduced for payment, later repurposed for quality. Their reliability varies.

### 8.3 FHIR does not solve the reconstruction problem

A common misconception is that FHIR availability makes EHR analytics easy. It does not, for four reasons:

1. **FHIR is an interface, not a warehouse.** A FHIR endpoint returns current resources. Historical depth depends on the server and the resource.
2. **References must be resolved.** A `Observation` points to a `Patient` and often to an `Encounter`. Reconstructing a timeline means following references, and dangling references are common.
3. **Profiles vary.** US Core constrains the minimum, not the maximum. Sites add extensions, and extension use is not standardized.
4. **Versioning.** FHIR resources carry a `meta.versionId` and `meta.lastUpdated`. Ignoring them means an update can look like a new event.

The practical rule: FHIR is excellent for exchange and for cohort discovery through a query interface. It is not a substitute for a curated analytic extract with documented provenance.

### 8.4 De-identification and the EHR

HIPAA provides two methods at 45 CFR 164.514(b):

- **Safe Harbor** removes 18 categories of identifiers, including names, geographic subdivisions smaller than a state (with a ZIP code exception for the first three digits in specified populations), all elements of dates except year, and device identifiers and serial numbers.
- **Expert Determination** requires a person with appropriate knowledge and experience to determine that the risk of re-identification is very small, and to document the methods and results.

Neither method makes re-identification impossible. In EHR data the residual risks concentrate in:

- **Dates.** Retaining full dates while removing names still allows linkage to public events and to other datasets.
- **Rare conditions and combinations.** A rare diagnosis plus an age plus a ZIP code can be unique.
- **Free text.** Names, family details, and locations appear in notes and are not removed by structured-field rules.
- **Imaging metadata.** Dates, institution names, device serial numbers, and burned-in annotations survive structured de-identification unless specifically handled.
- **Narrative chronology.** Even with dates shifted, an unusual sequence of events can identify a patient in a small community.

> **Note:** De-identification under HIPAA is not the same as anonymization under the EU General Data Protection Regulation (GDPR). GDPR Recital 26 sets a higher bar that considers all means reasonably likely to be used. A dataset that is de-identified for HIPAA may still be personal data under GDPR.

### 8.5 Coding systems that matter for EHR phenotypes

| System | Scope | Version cadence | Practical note for EHR work |
|---|---|---|---|
| ICD-10-CM | US diagnosis coding | Fiscal year, effective 1 October | FY2026 added 487 codes, revised 38, deleted 28, effective 1 October 2025 |
| ICD-10-PCS | US inpatient procedures | Fiscal year, plus mid-year update | Procedure definitions are combinatorial and not clinically intuitive |
| SNOMED CT | Clinical concepts | January and July international releases | About 360,000 active concepts in the international edition, approximate |
| LOINC | Lab and clinical observations | Twice yearly, plus hotfixes | Version 2.80, a hotfix released 26 February 2025, contained 104,672 terms |
| RxNorm | US medications | Monthly | Ingredient, clinical drug, and branded drug are different concept levels |
| CPT | US procedures and services | Annual, January | Proprietary; licensed by the American Medical Association |
| HCPCS Level II | Supplies, drugs, services | Quarterly | Used alongside CPT in outpatient billing |
| UCUM | Units | Continuous | Required for machine-readable units in FHIR |

The version cadence matters because a phenotype written against one code set version may silently change meaning in the next. Store the code set version with the phenotype.

---

## 9. Industry practice and case evidence

### 9.1 Vendor concentration shapes what data look like

According to KLAS Research's US acute care EHR market share report for 2024, published in April 2025, Epic held 42.3 percent of US acute care hospitals and 54.9 percent of acute care beds, up from 39.1 percent of hospitals in 2023. Oracle Health held 22.9 percent of hospitals and 22.1 percent of beds, and MEDITECH held 14.8 percent of hospitals and 12.7 percent of beds. Epic gained a net 176 hospitals in 2024, its largest gain on record.

What this means for an analyst, stated as inference rather than fact: a large share of US hospital data sits in one vendor's schema, which makes cross-site analytics easier within that vendor's customer base and harder outside it. Vendor-specific data models, flowsheet taxonomies, and reporting extracts become a de facto standard without being a published standard. The inference is that "Epic data" is a meaningful category for feasibility discussions, while "EHR data" is not.

The counterweight is that two Epic installations are still two databases. Configuration, build, and local coding practice differ, and those differences are often larger than vendor differences.

### 9.2 MIMIC-IV as the public reference point

MIMIC-IV (Medical Information Mart for Intensive Care IV) is a de-identified dataset of patients admitted to the Beth Israel Deaconess Medical Center in Boston. Version 3.1, released 11 October 2024 on PhysioNet with DOI 10.13026/kpb9-mt58, contains 364,627 unique individuals, of whom 223,452 had at least one hospitalization; 546,028 hospital admissions; and 94,458 intensive care unit (ICU) stays among 65,366 ICU patients. The data span 2008 to 2022, with dates shifted into 2100-2200.

The dataset's value is not representativeness. It is that the cohort definitions, the extraction code, and the preprocessing are public, so a result can be reproduced and a method can be compared across papers. The dataset's limit is equally clear: one hospital, one ICU practice pattern, one documentation culture.

The empirical record on transfer is consistent. Models developed on MIMIC and validated on eICU-CRD, a multi-site critical care database, generally lose discrimination. The loss is attributable to differences in measurement frequency, lab ordering, and case mix, not to a flaw in the modeling method. The practical lesson is that single-site validation is a floor, not a ceiling.

### 9.3 Documentation behaviour is a data-generating process

A large study of US ambulatory physicians provides the clearest public evidence that documentation behaviour changes the data. Apathy and colleagues, publishing in Health Services Research in 2023, analyzed 203,728 US ambulatory physicians using Epic between 2020 and 2021. They found that physicians in the top decile of copy-paste use closed 6.8 percentage points fewer visits on the same day and spent more time in the EHR after hours and on off days. Physicians with the longest notes spent 46 percent more time in notes per visit.

A 2025 survey by Hudkins and colleagues in Applied Clinical Informatics, covering 913 providers at a large academic medical center between December 2022 and March 2023, found that 82 percent reported copy-paste use and 52 percent reported copy-forward. Most believed it improved efficiency while acknowledging that it worsened documentation errors and note bloat.

The analytic implication, stated as inference: note length and note content are partly a function of author behaviour and template design, not only of patient complexity. A natural language processing (NLP) model that uses note length or section repetition as a feature may be learning the author's habits. Any note-derived variable needs a copy-forward audit before it is trusted.

### 9.4 Real-world data vendors add a curation layer

Oncology real-world data illustrates the pattern. Flatiron Health curates EHR-derived data from community oncology practices for research, combining structured fields with abstraction from unstructured documents. The industry lesson, which the company states publicly and which is visible in any comparable product, is that technology alone does not produce research-grade data. Human abstraction, a defined data model, and a quality program are part of the product.

For a buyer, this means the relevant question is not "how many patients" but "what is the abstraction protocol, what is the concordance with chart review, and what is the update latency." Vendors that publish those numbers are easier to evaluate than vendors that publish only counts.

### 9.5 What health systems actually do with their own EHR data

Three patterns are common, and they have different implications for an external partner:

1. **Research data warehouse.** The system extracts EHR data into a separate warehouse, often modeled to OMOP or to a local schema, with a governance process for access. This is the best case for an external collaborator: definitions are documented and the extract is stable.
2. **Report-by-report extraction.** Each request produces a bespoke query against the production database. This is fast for one question and unreproducible for a second. Expect definition drift between requests.
3. **Vendor-hosted research network.** The system contributes data to a vendor network under a data use agreement. The partner accesses a normalized product, not the raw EHR. Mapping loss is real and usually undocumented at the row level.

Asking which pattern applies is one of the highest-value questions in a data diligence conversation, and it takes one meeting to answer.

### 9.6 Regulatory use of EHR-derived real-world data

The US Food and Drug Administration (FDA) has published guidance on real-world data and real-world evidence, and operates a framework for evaluating whether a data source is fit for a regulatory purpose. The framework asks about data relevance and reliability: whether the data capture the exposure, outcome, covariates, and population, and whether the data's accuracy, completeness, and provenance support the intended use.

For EHR data, the reliability questions usually concentrate on three points: whether the diagnosis is confirmed rather than coded, whether the outcome is captured inside the network, and whether the data are complete through the study period. A submission that cannot answer these three will not clear a data fitness review, regardless of sample size.

### 9.7 How to diligence an EHR data source in one meeting

Six questions separate a usable extract from an unusable one, and all six can be answered by the data owner in under an hour:

1. What is the source system, the vendor, and the version, and when was the last upgrade?
2. What is the snapshot date, and is the extract a stable copy or a live query?
3. How was identity resolved, and what fraction of patients have more than one MRN?
4. Which facilities contribute, and is care outside those facilities visible in any form?
5. What is the phenotype library, and what validation has been published for the phenotypes you will use?
6. What is the de-identification method, and what residual risk has been documented?

A vendor that cannot answer questions 3 and 5 is selling a count, not a dataset. A vendor that answers all six with documents is selling something an analyst can defend.

### 9.8 What a data diligence meeting should produce

A diligence meeting that ends with impressions has failed. It should end with five artefacts, each of which can be checked later:

| Artefact | Content | Who owns it |
|---|---|---|
| Source inventory | Systems, vendors, versions, last upgrade | Data owner |
| Extract specification | Tables, grains, keys, filters, snapshot date | Data engineering |
| Phenotype library extract | Definitions, validation status, code set versions | Clinical informatics |
| Quality summary | Completeness and concordance by site and year | Analytics |
| Access and use terms | DUA, permitted use, retention, publication rights | Legal |

If any artefact is missing, the meeting produced a conversation rather than a basis for a decision, and the gap should be recorded as a risk rather than filled with an assumption.

### 9.9 What a mature EHR analytics team looks like

Organizations that do this well tend to share four features, and their absence predicts rework:

- A **data dictionary with version history**, not a spreadsheet that is overwritten each quarter.
- A **phenotype library** with owner, validation status, and code set versions.
- A **provenance requirement** enforced at the point of extract, so an analyst cannot produce a table without a source record.
- A **review step** in which a second analyst re-runs the cohort from the documentation before the result leaves the team.

None of these is expensive. All of them are culturally difficult, because each one slows the first delivery in exchange for making the second delivery possible.

---

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Encounter treated as patient | Patient counts equal encounter counts in a summary table | One row per visit passed downstream as one row per person | Prevalence and incidence overstated; variance understated | Define unit of analysis; deduplicate with a documented rule |
| Order treated as administration | Exposure prevalence implausibly high for an inpatient drug | `medication_request` joined without `medication_administration` | Exposure misclassification; effect diluted | Join to administration; report the order-to-administration ratio |
| Result time used as feature time | Model performance drops sharply on prospective validation | Lab results stamped at verification, which is after the prediction time | Temporal leakage; inflated offline metrics | Anchor features to order or specimen time; enforce with a query |
| Rule-out coded as disease | Disease prevalence far above published estimates | Coding of suspected conditions during workup | False positives; attenuated associations | Require confirmation (treatment, repeat code, lab); validate with chart review |
| Copy-forward treated as observation | Note text nearly identical across visits | Template and copy-forward behaviour | Features encode author habit; spurious associations | Deduplicate text; audit similarity; validate extractor |
| Unit change unnoticed | A lab mean shifts abruptly at a date | Assay or unit change with no flag | Threshold-based cohorts break at the change point | Keep source unit; plot values over time; version conversions |
| Missingness excluded silently | Cohort n drops with no flow diagram | Complete-case exclusion on a severity-related variable | Selection bias; wrong mortality or event rate | Report excluded group's outcome; use missing indicator or imputation |
| Site differences read as clinical | One site's rate is an outlier on every measure | Documentation and coding practice differ | Wrong quality conclusion; wrong site selection | Stratify; standardize; review data quality by site first |
| Right censoring ignored | Event rate falls in the final study months | Data cutoff truncates follow-up | Understated incidence; biased survival | Align horizon with cutoff; use time-to-event methods |
| Duplicate persons | Impossible timelines; overlapping admissions | Multiple MRNs; incomplete MPI merge | Inflated counts; broken longitudinality | Identity resolution before analysis; log merges |
| Free-text PHI in "de-identified" extract | Note text contains names or locations | Structured de-identification only | Privacy breach; contractual violation | Text de-identification; expert determination; audit sample |
| Lookback shorter than disease history | "Incident" cases have prior treatment | Lookback window too short for the condition | Prevalent cases misclassified as incident | Extend lookback; require clean period; sensitivity analysis |
| Data cutoff confused with extraction date | Recent months look incomplete | Events arrive after extraction | Spurious decline in counts | Record both dates; exclude the incomplete tail |

Three of the rows above account for the majority of published EHR errata. The order-versus-administration confusion is the most common in pharmacoepidemiology. The result-time-as-feature-time error is the most common in prediction modelling. The silent exclusion is the most common in quality measurement, because the analyst who applies a complete-case filter rarely sees the excluded patients again.

A useful discipline is to assign every failure in the table to one of three layers: **source** (the data were never recorded), **extraction** (the data exist but were not retrieved correctly), or **analysis** (the data were retrieved correctly but used incorrectly). The layer determines who fixes it and how long the fix takes. Source problems are usually unfixable inside the project and must become a stated limitation. Extraction problems are a engineering ticket. Analysis problems are a methods review.

Most project post-mortems attribute an analysis-layer error to a source-layer limitation, because the source-layer explanation is unfalsifiable and the analysis-layer explanation requires admitting a mistake. Resisting that attribution is part of professional practice.

### 10.2 Edge-case failures and their signatures

The edge cases in section 5.8 each produce a distinctive signature in the output. Learning the signatures shortens diagnosis.

| Edge case | Signature in the data | First query to run | Usual root cause |
|---|---|---|---|
| Null key | Row count falls after a join to a patient table | Count null keys per table | Feed mapping or a new source system |
| Conflicting source | Cohort size differs by 5-15% between EHR-only and claims-only definitions | Cross-tabulate the two definitions | Different documentation and billing purposes |
| Unit variant | A lab mean jumps by a factor near 88 or 10 at a date | Quantiles by month and unit string | Assay or interface change |
| Code version change | Case counts step at 1 October | Counts by week around the boundary | Annual code set update |
| Late-arriving data | Final month 20-40% below trend | Arrival curve by extraction date | Run-out too short |
| Partial identifier | Longitudinal analyses lose 3-10% of records | Count records with and without an enterprise ID | Registration practice at one site |
| Duplicate record | Encounter counts exceed scheduling records | Duplicate key count by site | Feed replay or registration error |
| Clock boundary | Events cluster at 00:00 or an hour is missing in spring | Events by hour around DST dates | Local time stored without offset |

A signature is not a diagnosis, but it narrows the search from days to hours.

### 10.1 The three red flags that should stop a project

Before those three, there is a fourth that is easier to spot: a cohort whose size changes materially when a defensible alternative definition is applied. In the worked example, switching the denominator moved the cohort from 10,725 to 8,210 (23.5 percent) and changed the site ranking. A result that depends on that choice is a result about the definition, not about the hospitals.
1. **No flow diagram with counts.** If the analyst cannot show how many records were removed at each step, the cohort is not reproducible and the bias is unknown.
2. **No chart-reviewed validation of the primary label.** A label defined only from codes has an unknown error rate. The error rate is usually large enough to change the conclusion.
3. **No statement of what the data cannot answer.** A memo that presents an EHR analysis without limits is either naive or hiding something. Both are disqualifying.

---

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| Patients are captured within the network for the relevant period | The cohort and the outcome both depend on it | Out-migration, insurance change, care at a non-affiliated site | Compare enrollment or claims data if available; chart-review a sample for outside care |
| The index date is clinically meaningful | Feature and label windows anchor on it | Index derived from a billing date rather than a clinical event | Review 50 records; compare order, admit, and coding dates |
| Diagnosis codes approximate disease | Phenotypes are built from codes | Rule-out coding, coding incentives, upcoding | Chart-review 100 positives and 100 negatives; compute PPV |
| Lab values are comparable over time | Trends and thresholds are used | Assay change, unit change, lab relocation | Plot distributions by month; check analyzer and method fields |
| The lookback captures prior history | Incident versus prevalent classification | Short lookback, new patient, outside history | Vary lookback; compare incidence estimates |
| Documentation reflects what happened | Notes and flowsheets are used as evidence | Under-documentation, copy-forward, template defaults | Compare with device data or claims; audit copy-forward rate |
| Sites are comparable after adjustment | Pooled estimates are reported | Unmeasured case-mix differences, coding differences | Direct standardization; site-stratified sensitivity analysis |
| Missingness is not informative | Complete-case analysis is used | Clinician ordering behaviour correlates with severity | Compare outcome rates for included and excluded |
| The extract is stable | Reproducibility claims are made | Production system changes, backfills | Hash the extract; re-run and compare counts |
| De-identification is sufficient | Data can be shared or transferred | Free text, dates, rare conditions, imaging metadata | Expert determination; re-identification testing; text audit |

The table above is a ledger, not a disclaimer. Each row should be carried into the analysis plan with a named test or a named sensitivity analysis. An assumption without a test is a belief, and a belief presented as a finding is the failure this module exists to prevent.

A practical format for the ledger is three columns: the assumption, the direction of bias if it fails, and the test. The direction is often knowable even when the magnitude is not. If out-of-network care is missing and the missing patients are healthier, the observed event rate is biased upward. That directional statement is usually enough to tell a decision-maker whether the result is usable.

### 11.1 Limits that are inherent, not fixable
Some limits cannot be engineered away:

- **The EHR cannot observe care it did not deliver.** No amount of cleaning recovers a visit to an unaffiliated clinic.
- **The EHR cannot observe behaviour outside the encounter.** Adherence, diet, exercise, and symptoms between visits are absent unless a patient-reported instrument captures them.
- **The EHR cannot recover what was never recorded.** A conversation that produced a decision but no note is invisible.
- **The EHR cannot make coding equal clinical truth.** The code is a billing and documentation artefact by design.
- **The EHR cannot guarantee representativeness.** The population is whoever uses the system, which is a selected population.

A deliverable that states these five limits, and says which ones threaten the primary conclusion, is more credible than one that claims to have addressed them.

### 11.2 How to write a limits section that survives review

A limits section fails review for three reasons: it is generic, it is buried, or it is longer than the finding. The repair for each is mechanical.

- **Generic.** Replace "data may be incomplete" with the specific field, the specific stratum, and the measured rate. "Lactate was missing for 2,388 of 13,113 admissions (18.2%), concentrated in the lowest-acuity stratum, which had 9.4% mortality versus 11.2% among tested admissions."
- **Buried.** Put the limits in the body, immediately after the primary result, not in an appendix. A reader who stops after the result should already know the main threat.
- **Longer than the finding.** Rank the limits. Three that threaten the conclusion matter more than ten that do not. State the three, then list the rest in one line.

A useful test: give the limits section alone to a colleague who has not seen the analysis and ask them what the analysis cannot support. If they cannot answer, the section is decorative.

### 11.3 The limits that change with the unit of analysis

The worked example shows a limit that is easy to miss: the unit of analysis is itself a limit on the claim. An encounter-level rate cannot support a statement about patients, and a patient-level rate cannot be compared to a per-discharge benchmark.

Three limits follow from the unit, and each should be stated explicitly:

- **Independence.** Encounters within a patient are correlated, so a per-encounter variance estimate is too small unless cluster-robust methods are used.
- **Weighting.** A patient-level average weights each patient once; an encounter-level average weights frequent attenders more. The two answer different questions about equity.
- **Benchmark comparability.** Published measures are defined per discharge or per patient, and a rate computed on the other unit is not comparable regardless of how similar the numerator looks.

A deliverable that states its unit, its variance method, and its benchmark definition has already addressed the most common reviewer objection.

---

## 12. Exercises

### Exercise 1 — Classify the data families (recall, 20 minutes)

Take the eight EHR data families in section 1 and, for each, write one question it can answer and one it cannot.

Deliverable: a two-column table with eight rows.

Acceptance criteria: every "cannot" is a genuine limitation of the producer and workflow, not a generic data quality complaint; at least three of the "cannot" items name the workflow reason.

### Exercise 2 — Reconstruct a timeline (application, 45 minutes)

Using the FHIR sample bundle in `labs/` or any public bundle, build a patient timeline with at least five event types. Record the reference resolution you performed and any references that did not resolve.

Deliverable: a sorted timeline table with columns for timestamp, event type, source resource, and resolved reference.

Acceptance criteria: timestamps normalized to one time zone; unresolved references listed with a count; at least one duplicate or near-duplicate identified.

### Exercise 3 — Cohort with exclusion accounting (application, 90 minutes)

Using the SQL pattern in section 7.1 against any available EHR extract, reproduce a cohort with at least four exclusion steps. Report counts at every step and the outcome rate for the included and excluded groups at the step with the largest removal.

Deliverable: a flow table and a short note on the bias direction.

Acceptance criteria: counts reconcile arithmetically; the excluded group's outcome rate is reported; the note states whether the exclusion inflates or deflates the primary rate and why.

### Exercise 4 — Phenotype validation (critique, 60 minutes)

Take a published EHR phenotype for a condition of your choice. Identify its confirmation rule, its code set version, and its reported validation. Write a 400-word critique of whether the phenotype is adequate for (a) prevalence estimation and (b) comparative effectiveness.

Deliverable: a 400-word critique with three specific recommendations.

Acceptance criteria: the critique distinguishes prevalence from comparative use; each recommendation names the bias it addresses; no claim is made without a citation or a stated inference.

### Exercise 5 — Standardization and rank reversal (calculation, 40 minutes)

Using the counts in section 6.6, recompute the standardized rates with weights 0.30, 0.40, and 0.30. Report whether the rank reversal between hospitals A and C persists.

Deliverable: a table of crude and standardized rates plus a one-paragraph interpretation.

Acceptance criteria: arithmetic correct to two decimal places; the interpretation states that the choice of standard population changes the comparison, and says which choice is defensible and why.

### Exercise 6 — Leakage audit (critique, 60 minutes)

Take any EHR prediction paper with a methods section that describes features and outcomes. Identify every feature whose timestamp could be after the prediction time. Write the query you would run to detect the leak in the underlying data.

Deliverable: a list of suspect features with the timestamp rule that makes each legitimate or illegitimate, plus a detection query.

Acceptance criteria: at least three features assessed; the query uses an actual time comparison, not a description; the write-up distinguishes certain leakage from possible leakage.

### Exercise 7 — Provenance record (design, 75 minutes)

Write a provenance record for an EHR extract you have used or one described in a paper. Include source, snapshot, query, transformations, unit conversions, linkage, de-identification, and limitations.

Deliverable: a one-page record in the format of section 5, step 24.

Acceptance criteria: a second analyst could re-run the cohort from the record alone; every transformation has a version; limitations are specific rather than generic.

### Exercise 8 — What the EHR cannot answer (design, 45 minutes)

For the sepsis example in section 6, write the three most important questions the EHR extract cannot answer, and for each propose a linkage or prospective collection that would answer it.

Deliverable: three questions, three proposed designs, and a cost and latency estimate for each.

Acceptance criteria: each question is genuinely unanswerable from the extract, not merely inconvenient; each proposal names the data source and the access pathway; estimates are ranges with stated assumptions.

### Exercise 9 — Site data quality profile (application, 60 minutes)

Using any multi-site extract, compute completeness for five key fields by site and year. Identify the field with the largest site-by-year variation and write a hypothesis for the mechanism.

Deliverable: a completeness matrix and a 200-word hypothesis.

Acceptance criteria: matrix stratified by site and year; hypothesis names a workflow or build change, not "bad data"; at least one falsifiable test proposed.

### Exercise 10 — Denominator switch (calculation and critique, 60 minutes)

Using the encounter-level and patient-level counts in sections 6.2 and 6.7, recompute both rates and the site ranking for a fifth hospital with 1,600 encounters, 1,190 patients, 172 deaths overall, and 26 deaths among 410 repeat encounters. State which denominator you would report for a patient-facing quality report and why.

Deliverable: a five-row table with both rates and both rankings, plus a one-paragraph recommendation.

Acceptance criteria: the patient-level death count is derived correctly from the overall and repeat counts; the recommendation names the decision the denominator serves; the rank change is stated explicitly.

---

## 13. Self-check answers

**A1. What are the four clocks, and which one must never be used for feature timing?**
Order time, specimen or start time, result or end time, and documentation time. Documentation time must never be used for feature timing, because it reflects when a note or flowsheet row was saved, which can be after discharge. Result time should also be avoided for features when the prediction time precedes result availability.

**A2. Why is an encounter not a patient?**
An encounter is a registration for a single interaction. A patient accumulates many encounters, sometimes at multiple facilities, sometimes overlapping after a transfer. Treating encounters as independent observations understates variance and inflates counts; treating them as patients requires a deduplication rule.

**A3. Compute the mortality rate for 1,425 deaths in 13,113 admissions.**
1,425 / 13,113 = 0.10867, or 10.9%. With a 95 percent interval: p(1−p) = 0.10867 × 0.89133 = 0.09687; divided by 13,113 = 0.000007387; square root = 0.002718; × 1.96 = 0.005327. Interval 10.3% to 11.4%.

**A4. Why did the complete-case cohort overstate mortality in the worked example?**
Because the exclusion criterion, a lactate result within the specified window, was related to clinician suspicion and therefore to severity. The excluded group had 9.4% mortality versus 11.2% in the included group. Removing a lower-risk group raises the observed rate in the remaining group. This is selection bias from conditioning on a post-baseline, behaviour-dependent variable.

**A5. What is the PPV of the screening flag, and what does it imply?**
118 / 200 = 59.0%, with a Wilson 95 percent interval of 52.1% to 65.6%. It implies that about 41 percent of flagged encounters are not sepsis by chart review, so the flag cannot serve as a label. It can serve as a screening step followed by confirmation.

**A6. Why can sensitivity not be computed from the validation sample?**
Because the sample was drawn only from flagged encounters. Sensitivity requires the proportion of true cases that were flagged, which needs a sample of all admissions including unflagged ones. Sampling on the flag gives no information about the unflagged denominator.

**A7. State the standardization weights and the standardized rate for hospital A.**
Weights 0.40 for ages 18-64, 0.35 for 65-79, and 0.25 for 80 and older. Rates 6.77%, 13.58%, and 16.87%. Standardized rate = 0.40 × 6.77 + 0.35 × 13.58 + 0.25 × 16.87 = 2.708 + 4.753 + 4.2175 = 11.68%.

**A8. Explain the rank reversal in one paragraph.**
On the crude rate, hospital A (11.49%) appears better than hospital C (11.87%). After direct standardization to a common age distribution, A is 11.68% and C is 11.65%, so A is slightly worse. The crude difference was driven by case mix: A's population is younger. Standardization removes the age contribution and reveals that A's age-specific mortality is marginally higher.

**A9. Name three residual re-identification risks in a HIPAA Safe Harbor extract.**
Full dates retained under another lawful basis, rare diagnoses combined with age and geography, and free-text or burned-in information that structured rules do not remove. A fourth is unusual event sequences in a small population.

**A10. What is the difference between an order and an administration, and why does it matter?**
An order records clinician intent; an administration records that a dose was given. Using orders as exposure overstates treatment, includes cancelled and refused orders, and dilutes the effect estimate. The order-to-administration ratio is a useful data quality metric.

**A11. Why is FHIR not a solution to EHR reconstruction?**
FHIR is an exchange interface: it returns resources at a point in time, requires reference resolution, permits site-specific profiles and extensions, and carries versioning that analysts often ignore. It standardizes structure, not completeness or clinical accuracy.

**A12. Rubric for exercise 7 (provenance record).**
A strong record: names the source system and snapshot; includes the full query or a hashed reference to it; lists every transformation with a version; records unit conversions with source and target units; describes linkage and its error; states the de-identification method and its residual risks; lists specific limitations. A weak record: names the source but not the snapshot; describes transformations in prose without versions; lists generic limitations such as "data may be incomplete."

---

## 14. Interview questions

**Q1. When would you choose EHR data over claims data, and when the reverse?**
A strong answer: EHR when the question needs clinical measurements, severity, lab values, or documentation detail; claims when the question needs complete utilization and cost across providers, or a population denominator that is not tied to one delivery network. Strong candidates note that the two are complementary and that linkage is often the right answer, and that linkage has its own error.

**Q2. How would you define a sepsis cohort, and what would you do about the 41 percent false-positive rate?**
A strong answer: start from a clinical definition (suspected infection plus organ dysfunction), implement it with a combination of culture, antibiotic, and lab criteria, validate against chart review, and report PPV and sensitivity. Use the screening flag only as a candidate generator. Weak answers treat the flag or the code as the definition.

**Q3. A model has an area under the curve of 0.91 offline and 0.68 at a new site. What do you check first?**
A strong answer: check the label definition and the feature timing before blaming the model. Differences in lab ordering, result-time stamping, and code practice between sites can produce exactly this drop. Then check case mix and prevalence. Weak answers go straight to retraining.

**Q4. Explain temporal leakage with a concrete EHR example.**
A strong answer: a deterioration model that uses the last lactate value before the prediction time but stamps the feature at result verification, which occurs after the prediction time. Or a model that uses "worst value in the first 24 hours" while predicting at hour 6. Weak answers describe leakage abstractly without a timestamp.

**Q5. How do you handle a patient with two MRNs?**
A strong answer: resolve identity before analysis using the enterprise ID and the master patient index, log every merge, check for impossible timelines after merging, and report unresolved duplicates. Weak answers drop duplicates, which biases the sample.

**Q6. What is present-on-admission, and why does it matter?**
A strong answer: a billing flag indicating whether a condition existed at admission, used to distinguish comorbidity from complication. It matters because comorbidity indices and hospital-acquired condition measures depend on it, and because its reliability varies. Weak answers have not heard of it.

**Q7. How would you quantify the effect of an exclusion criterion?**
A strong answer: report the count removed and the outcome rate in the removed group versus the retained group, then run the primary analysis with and without the exclusion and report both. If the difference is material, use a bounding analysis or a missing-data method rather than complete-case analysis.

**Q8. What would you put in a data provenance record?**
A strong answer: source system and owner, snapshot identifier, extraction query, transformation code and version, unit conversions, inclusion and exclusion rules, linkage method and quality, de-identification method, known limitations, retention plan. Weak answers list only the source and the date.

**Q9. Why might a note-based feature be misleading?**
A strong answer: copy-forward and templates make note text partly a function of author behaviour, so the feature may encode the author rather than the patient. Validation against chart review and a copy-forward audit are required.

**Q10. How do you compare mortality across hospitals fairly?**
A strong answer: stratify or standardize for case mix, state the standard population, report crude and adjusted rates, and check whether data quality differs by site before interpreting a difference as clinical. Weak answers compare crude rates.

**Q11. What does HIPAA Safe Harbor remove, and what does it leave behind?**
A strong answer: 18 identifier categories including names, sub-state geography with a ZIP exception, date elements except year, and device identifiers. It leaves behind rare conditions, unusual event sequences, free text, and imaging metadata, all of which carry residual risk.

**Q12. How would you decide whether an EHR extract is fit for a regulatory submission?**
A strong answer: assess relevance (does it capture exposure, outcome, covariates, population) and reliability (accuracy, completeness, provenance, auditability), document the assessment, and identify the specific gaps. Note that FDA has published a framework for real-world evidence and that fitness is use-specific.

**Q13. What is the first thing you would check in an unfamiliar EHR extract?**
A strong answer: the grain of each table and the identity resolution. Grain errors and duplicate persons corrupt every downstream number, and both are visible in minutes with a pair of count queries. Weak answers start with the clinical question and discover the structural problems after building the cohort.

---

## 15. Key terms

- **ADT** — admit, discharge, transfer; the HL7 message family that carries encounter state changes.
- **CDA** — Clinical Document Architecture; HL7's document standard.
- **Copy-forward** — carrying text from a prior note into a new note.
- **Data cutoff** — the last date through which source data are believed complete.
- **DRG** — diagnosis-related group; an inpatient payment category.
- **EHR** — electronic health record.
- **EMR** — electronic medical record; usually a single-organization record.
- **Encounter** — one interaction with the health system.
- **Episode of care** — a clinically defined span of related encounters.
- **FHIR** — Fast Healthcare Interoperability Resources.
- **Flowsheet** — structured nursing documentation grid.
- **HL7 v2** — the dominant healthcare messaging standard.
- **ICD-10-CM** — the US diagnosis classification, updated each fiscal year.
- **Index date** — the anchor date for cohort entry.
- **Information blocking** — interference with access to electronic health information, regulated at 45 CFR Part 171.
- **LOINC** — Logical Observation Identifiers Names and Codes.
- **Lookback window** — the period before index used to establish history.
- **MIMIC-IV** — a public de-identified critical care EHR dataset.
- **MPI** — master patient index.
- **MRN** — medical record number.
- **OMOP CDM** — a common data model for observational health analytics.
- **Order-to-result lag** — time from order to result availability.
- **Phenotype** — an executable definition identifying patients with a condition.
- **PHI** — protected health information under HIPAA.
- **POA** — present on admission.
- **PPV** — positive predictive value.
- **Right censoring** — follow-up truncated by the end of data.
- **Rule-out** — a diagnosis documented while being excluded.
- **SNOMED CT** — a comprehensive clinical terminology.
- **Temporal leakage** — using information unavailable at the decision time.
- **UCUM** — Unified Code for Units of Measure.
- **US Core** — the HL7 US realm implementation guide constraining FHIR for US exchange.
- **Unit of analysis** — the entity each row represents (encounter, patient, episode).
- **Washout period** — a period excluded to ensure incident cases.
- **Cluster-robust variance** — a variance estimate that accounts for correlation within a patient.

---

## 16. Sources

All URLs accessed 24 October 2026.

**Standards and regulators**

- US Department of Health and Human Services, Office for Civil Rights. *Guidance Regarding Methods for De-identification of Protected Health Information in Accordance with the HIPAA Privacy Rule.* https://www.hhs.gov/hipaa/for-professionals/special-topics/de-identification/index.html
- eCFR. 45 CFR 164.514, *Other requirements relating to uses and disclosures of protected health information.* https://www.ecfr.gov/current/title-45/subtitle-A/subchapter-C/part-164/subpart-E/section-164.514
- eCFR. 45 CFR Part 171, *Information Blocking.* (Regulatory text; cite the Part, no section URL given here.)
- Office of the National Coordinator for Health Information Technology. *HTI-1 Final Rule.* https://healthit.gov/regulations/hti-rules/hti-1-final-rule/
- Federal Register. *Health Data, Technology, and Interoperability: Certification Program Updates, Algorithm Transparency, and Information Sharing.* 9 January 2024. https://www.federalregister.gov/documents/full_text/html/2024/01/09/2023-28857.html
- HL7 International. *FHIR Release 5 (5.0.0).* https://hl7.org/fhir/
- HL7 International. *US Core Implementation Guide.* https://hl7.org/fhir/us/core/
- Centers for Medicare and Medicaid Services. *ICD-10-CM/PCS.* https://www.cms.gov/medicare/coding-billing/icd-10-codes
- Centers for Medicare and Medicaid Services. *FY 2026 ICD-10-CM Coding Guidelines.* https://www.cms.gov/files/document/fy-2026-icd-10-cm-coding-guidelines.pdf
- Regenstrief Institute. *LOINC Version 2.80 Release Notes.* https://loinc.org/kb/release-notes-archive/2.80
- SNOMED International. *SNOMED CT January 2025 International Edition Release Notes.* https://confluence.ihtsdotools.org/display/RMT/SNOMED+CT+January+2025+International+Edition+-+SNOMED+International+Release+notes
- SNOMED International. *SNOMED CT Translation Management Guide* (source of the approximate 360,000 active concept figure). https://docs.snomed.org/snomed-ct-practical-guides/snomed-ct-translation-management-guide/1-introduction

**Datasets**

- Johnson, A. E. W., et al. "MIMIC-IV, a freely accessible electronic health record dataset." *Scientific Data*, 2023. https://doi.org/10.1038/s41597-022-01899-x
- PhysioNet. *MIMIC-IV v3.1.* Released 11 October 2024. https://physionet.org/content/mimiciv/3.1/
- PhysioNet. *eICU Collaborative Research Database.* https://eicu-crd.mit.edu/

**Industry and peer-reviewed evidence**

- KLAS Research. *US Acute Care EHR Market Share 2024*, reported April 2025. Cited via Fierce Healthcare, https://www.fiercehealthcare.com/health-tech/epic-gaining-more-ground-hospital-ehr-market-share-widens-its-lead-over-oracle-health
- Apathy, N. C., et al. "Documentation dynamics: Note composition, burden, and physician efficiency." *Health Services Research*, 2023. (Background: pre-2024, cited as context.)
- Hudkins, M., et al. "Exploring Provider Perceptions and Attitudes toward Copy-Paste and Copy-Forward in Clinical Documentation." *Applied Clinical Informatics*, 2025. https://pubmed.ncbi.nlm.nih.gov/40769172/
- US Food and Drug Administration. *Real-World Evidence.* https://www.fda.gov/science-research/science-and-research-special-topics/real-world-evidence
- Flatiron Health. Public description of curated oncology real-world data. https://flatiron.com/

**Background, older than 2024**

- HL7 International. *HL7 Version 2.5.1* (2007). Cited as background for message structure.
- Health Level Seven. *CDA Release 2.0* (2005). Cited as background.

> **Note:** Where a source is marked background, it is older than the 2024-2026 preference and is included because the standard or evidence base has not been superseded for the purpose cited.

---

## 17. Links

- Module index: [`_index.md`](_index.md)
- Next topic: [`02_claims-and-billing.md`](02_claims-and-billing.md)
- Related sibling: [`08_standards-and-terminologies.md`](08_standards-and-terminologies.md)
- Related sibling: [`10_data-quality-dimensions.md`](10_data-quality-dimensions.md)
- Related sibling: [`09_licensing-and-access.md`](09_licensing-and-access.md)
- Root survey file: [`../../02_biomedical_data_landscape.md`](../../02_biomedical_data_landscape.md)
- Labs: [`../../labs/`](../../labs/)
- Lab 2.1 in the survey file (FHIR bundle exploration) and Lab 2.2 (OMOP cohort query) both use the cohort construction pattern in section 7.

Related topics for continued reading: the terminology layer in [`08_standards-and-terminologies.md`](08_standards-and-terminologies.md) explains why code mapping is lossy; the quality framework in [`10_data-quality-dimensions.md`](10_data-quality-dimensions.md) turns the checks in section 5.6 into a repeatable assessment; and [`09_licensing-and-access.md`](09_licensing-and-access.md) covers the contractual path to obtaining an EHR extract in the first place.

If you read only one sibling next, read the standards topic. The denominator discussion in sections 6.7 to 6.9 is developed further in the quality topic, where the unit of analysis is one of the first things a fitness decision must fix. Most of the failure modes in section 10 are terminology problems wearing a data problem's clothes, and the mapping section there is the fastest way to see why.

Three checks to carry into the next topic:

- Does the source record the unit of analysis, or does the analyst impose it?
- Does the source record the code set version, or does the analyst guess it?
- Does the source record the denominator, or does the analyst construct it from a different table?
