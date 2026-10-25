# Data in Pharma: Domains, Lineage, and the Cost of Disagreement

> Module 13 — Biopharma and Clinical Development
> Topic 02 of 10
> Estimated study time: 4 hours
> Prerequisites: Module 03 (data engineering), Module 08 (EHR and claims); topic 01
> Version 2026.3

## 1. Why this topic exists

A pharmaceutical company holds more data about a single patient than almost any other
organisation, and it holds that data in at least six incompatible forms. The same adverse event
appears as a narrative in the safety database, a coded term in the clinical database, a row in the
submission dataset, a line in a periodic safety report, and a count in an annual report. The same
patient appears as a subject identifier in the trial, a patient identifier in the electronic health
record, a member identifier in claims, and a de-identified token in a real-world dataset. Each
representation is correct within its own system and wrong as a description of the whole.

The cost of this fragmentation is not mainly technical. It is the cost of disagreement. When two
functions report different numbers for the same quantity — the number of serious adverse events,
the number of enrolled patients, the number of sites — the organisation spends weeks reconciling
before it can answer a regulator, a partner, or a board. Worse, some disagreements are never
reconciled, and the wrong number enters a submission, a label negotiation, or a safety review.

This topic maps the data domains of a biopharma company, the grain and standards of each, and the
lineage that connects them. It then teaches the reconciliation discipline that makes disagreement
visible and resolvable. The next topics depend on this map: biomarkers (topic 03) live in
laboratory and omics domains; feasibility (topic 04) lives in EHR, claims, and registry domains;
standards (topic 05) define the tabulation and analysis layers; safety (topic 06) lives in the
pharmacovigilance domain; real-world evidence (topic 07) lives in claims and EHR domains; and
submission (topic 10) is the assembly of the whole chain.

### 1.1 The failure pattern

1. A Phase III study completes. The clinical database contains 398 adverse events in the study
   report.
2. The safety database contains 412 reports for the same study.
3. The safety group presents 412 to the benefit-risk committee. The clinical group presents 398 in
   the clinical study report.
4. Neither number is wrong. The safety database includes 14 events with onset after the clinical
   data cut and 3 duplicate reports; the clinical database includes 5 events never reported to
   pharmacovigilance.
5. The committee spends two meetings on the discrepancy and none on the signal.
6. The submission contains both numbers, in different documents, without an explanation.
7. A reviewer asks why the numbers differ, and the answer is assembled under time pressure.

Steps 4 through 7 are avoidable with a reconciliation that takes hours, not weeks, and that
produces a documented frame: which events are in scope, which identifiers are used to match, and
what the residual differences are.

### 1.2 Who is blocked when the data map is missing

| Stakeholder | What they lose | Why they cannot fix it alone |
|---|---|---|
| Clinical data manager | Weeks reconciling counts with safety | Does not own the safety database |
| Safety physician | Confidence in the case series | Receives coded data without the clinical context |
| Biostatistician | A clean analysis population | Receives datasets with undocumented exclusions |
| Regulatory affairs | A defensible submission narrative | Receives two numbers and no reconciliation |
| Feasibility analyst | An eligible-patient estimate | Cannot see EHR and claims together |
| Commercial analyst | A market estimate | Sees prescription data without diagnosis |
| Patient | A safety signal detected late | Has no view of any of it |

### 1.3 The industry question

> For this decision, which data domain is authoritative, at what grain, under which standard, and
> how do we know the numbers agree?

The question has four parts because a dataset without a stated grain, standard, and authority is
not evidence. Most data disputes in pharma are disputes about one of the four, conducted as if
they were disputes about facts.

## 2. Learning outcomes

By the end of this topic you can:

1. **Map** the principal data domains of a biopharma company to the decisions they support, with
   the grain, owner, and standard of each.
2. **Distinguish** source data, source records, tabulation datasets, analysis datasets, and
   derived outputs, and state the authority of each for a given question.
3. **Trace** a variable from its capture point to its appearance in a table, listing every
   transformation and the metadata that documents it.
4. **Design** a reconciliation between two domains that disagree, including the common frame, the
   matching key, the residual categories, and the acceptance threshold.
5. **Compute** an event-level and subject-level agreement rate, and interpret it with its
   denominator.
6. **Diagnose** a reconciliation that is broken by a data cut, a coding version change, or a
   duplicate-reporting mechanism.
7. **Critique** a data platform proposal that claims to unify domains without addressing grain,
   keys, or governance.

## 3. The core mental model

### 3.1 The supply chain

```text
 Patient encounter            Source record              Tabulation            Analysis           Output
 -----------------            -------------              ----------            --------           ------
 Clinic visit / lab      -->  EHR, eSource, ePRO,   -->  SDTM domains     -->  ADaM datasets -->  Tables,
 imaging / PRO / device       paper, device feed          (DM, AE, LB, VS,      (ADSL, ADAE,        listings,
                              (raw + metadata)             EX, CM, ...)          ADTTE, ...)         figures
                                   |                          |                     |                 |
                                   |  extraction rules        |  mapping spec       |  analysis plan  |
                                   v                          v                     v                 v
                              audit trail               define.xml            analysis metadata   submission
```

Two flows run in parallel with this chain. The **safety flow** sends serious adverse events from
the site to the pharmacovigilance database within regulatory reporting timelines, independent of
the clinical database schedule. The **real-world flow** brings EHR, claims, registry, and
prescription data into feasibility, safety surveillance, and commercial analytics, usually without
any connection to the trial identifiers.

### 3.2 Four axes that make domains incompatible

| Axis | What differs | Example | Consequence |
|---|---|---|---|
| Grain | The unit of a row | One row per patient versus one row per event versus one row per claim line | Counts cannot be compared without aggregation |
| Key | The identifier used to link | Subject ID, patient ID, member ID, de-identified token | Joins lose or duplicate patients |
| Standard | The vocabulary and model | MedDRA, WHO Drug, LOINC, SNOMED CT, ICD-10, CDISC controlled terminology | The same concept has different codes |
| Governance | Who may see, change, and retain | GCP systems, HIPAA-covered data, GDPR personal data, commercially confidential | Access and retention rules conflict |

A data platform that ignores any of these axes produces a join that runs and an answer that is
wrong. The most common failure is joining on a key that is not unique at the required grain — for
example, joining a claims line table to a patient table without deduplicating claim lines, which
multiplies the patient count.

### 3.3 Latency and the data cut

```text
 Site entry ----(days)----> EDC ----(weeks)----> cleaned clinical database ----(months)----> SDTM
      |                                                                                      |
      +----(hours-days)----> safety database (serious events, expedited) --------------------+
                                                                                             |
 EHR/claims ----(1-6 months)----> real-world dataset ----------------------------------------+
```

Each domain has a different latency, and the differences are the source of most reconciliation
gaps. A clinical data cut freezes one domain at a date; the safety database keeps receiving
reports; the real-world dataset lags by months. Comparing a frozen domain with a live one without
aligning the cut produces a discrepancy that is entirely an artefact of timing.

### 3.4 The same patient, five identities

| Domain | Identifier | Typical uniqueness problem |
|---|---|---|
| Clinical trial | Subject identifier within study | Re-screened patients may receive a new identifier |
| Safety database | Case identifier, plus study and subject | Duplicate reports from site and partner |
| EHR | Medical record number within institution | Patients with records in multiple institutions |
| Claims | Member identifier within plan | Plan changes, family members, retroactive eligibility |
| Registry | Registry-specific identifier | Linkage to trial or EHR requires probabilistic matching |

No single identifier spans all five. Linkage is therefore a design decision with an error rate,
not a lookup. Topic 07 covers the methods; this topic establishes that the decision must be
recorded.

### 3.5 Authority is question-specific

There is no "golden record" in pharma. There is an authoritative source for each question:

- For whether an event was reported to a regulator on time, the safety database is authoritative.
- For whether the event was captured in the trial database for analysis, the clinical database is.
- For whether the event occurred during the treatment period, the source record is.
- For the final analysis population, the analysis dataset with its metadata is.

A data strategy that declares one system the single source of truth without specifying the
question will be wrong for some questions and will hide the wrongness inside a pipeline.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Source data | All information in original records and certified copies, including metadata | Used to mean the eCRF only | Record |
| Source record | The original document or electronic record, including relevant metadata | Confused with the transcription | Record |
| eSource | Electronic capture of source data at the point of care | Used for any electronic system | System category |
| Electronic data capture (EDC) | System for entering trial data into case report forms | Treated as the source | System |
| Electronic case report form (eCRF) | The trial form, electronic | Treated as source data | Form |
| Patient-reported outcome (PRO) | Outcome reported directly by the patient | Used for any questionnaire | Instrument |
| Electronic PRO (ePRO) | PRO captured on a device | Assumed equivalent to paper without validation | Instrument |
| Raw data | Data as received, before transformation | Confused with source data | Dataset |
| Tabulation dataset | Structured dataset for regulatory submission, e.g. SDTM | Treated as analysis-ready | Dataset |
| Analysis dataset | Derived dataset structured for a pre-specified analysis, e.g. ADaM | Treated as submission data | Dataset |
| SDTM | Study Data Tabulation Model, CDISC standard for tabulations | Used as an analysis model | Standard |
| ADaM | Analysis Data Model, CDISC standard for analysis datasets | Assumed to contain only raw values | Standard |
| CDASH | Clinical Data Acquisition Standards Harmonisation, CDISC standard for capture | Confused with SDTM | Standard |
| Define-XML | Machine-readable metadata describing datasets and variables | Treated as documentation only | Standard |
| Controlled terminology | CDISC-published code lists for variables | Confused with MedDRA | Code lists |
| MedDRA | Medical Dictionary for Regulatory Activities, for adverse event coding | Used for diagnoses in general | Dictionary, versioned |
| WHO Drug | WHO medicinal product dictionary for concomitant medications | Used for procedures | Dictionary, versioned |
| LOINC | Logical Observation Identifiers Names and Codes for laboratory tests | Assumed to cover all assays | Code system |
| SNOMED CT | Clinical terminology for findings, procedures, and substances | Used as a billing code set | Terminology |
| ICD-10 / ICD-11 | Disease classification for morbidity and mortality coding | Treated as a clinical terminology | Classification |
| FHIR | Fast Healthcare Interoperability Resources, HL7 standard for exchange | Treated as an analytics model | Standard |
| OMOP CDM | Observational Medical Outcomes Partnership Common Data Model | Treated as a regulatory standard | Data model |
| Metadata | Data describing data: definition, origin, derivation, and context | Used to mean column names | Structured description |
| Audit trail | Secure, time-stamped record of creation, modification, and deletion | Treated as a log file | Record |
| ALCOA+ | Attributable, legible, contemporaneous, original, accurate, plus complete, consistent, enduring, available | Reduced to a slogan | Data integrity principle |
| Data lineage | The documented path from source to output | Confused with a data flow diagram | Documentation |
| Data provenance | The origin and history of a data element | Used interchangeably with lineage | Documentation |
| Data cut | The date at which a dataset is frozen for analysis | Confused with database lock | Date |
| Database lock | Formal declaration that the database will no longer be changed | Confused with data cut | Event |
| Query | A request to resolve an inconsistency in entered data | Used for any data request | Record |
| Source data verification (SDV) | Comparison of source records with entered data | Assumed to be the only quality control | Activity |
| Medical coding | Assignment of dictionary terms to verbatim terms | Treated as a data entry step | Activity |
| SEND | Standard for Exchange of Nonclinical Data, CDISC standard for animal studies | Confused with SDTM | Standard |
| Data lake | Central store of raw data in native formats | Assumed to be governed | Architecture |
| Data warehouse | Structured store for query and reporting | Confused with a lake | Architecture |
| Real-world data (RWD) | Data collected outside randomised trials | Treated as real-world evidence | Data category |
| eCTD | Electronic common technical document, the submission format | Confused with the datasets | Format |

### 4.1 Grain, key, and population

Three properties must be stated for every dataset:

- **Grain** — what one row represents. "One row per subject per visit per analyte" is a grain;
  "laboratory data" is not.
- **Key** — the minimal set of columns that uniquely identifies a row at that grain.
- **Population** — who is included and who is excluded, with the rule that produced the exclusion.

A dataset without these three cannot be joined safely and cannot be analysed reproducibly.
Metadata standards capture them, but a surprisingly large number of internal datasets do not.

### 4.2 The domain families and what each is authoritative for

| Domain family | Examples | Grain | Key standard | Authoritative for |
|---|---|---|---|---|
| Preclinical | Toxicology, pharmacokinetics in animals, assay results | One row per animal per timepoint per analyte | SEND | Nonclinical safety and exposure |
| Clinical capture | Demographics, visits, adverse events, concomitant medications, laboratory | One row per subject per event or visit | CDASH, then SDTM | What was recorded at the site |
| Safety | Individual case safety reports, narratives, causality | One row per case, with events nested | E2B(R3), MedDRA, WHO Drug | Regulatory reporting and case narratives |
| Biomarker and translational | Central laboratory, flow cytometry, genomics, PK/PD | One row per sample per analyte per timepoint | CDISC LB, study-specific | The measurement and its assay metadata |
| Imaging | Scans, reads, RECIST assessments | One row per subject per timepoint per reader | DICOM, study-specific | The image and the read |
| Real-world | Claims, EHR, registries, prescriptions | One row per claim line, encounter, or patient | OMOP, FHIR, ICD, CPT/HCPCS | Utilisation and outcomes in routine care |
| Commercial | Sales, prescriptions, market share, pricing | One row per product per geography per period | Company-specific | Revenue and market position |
| Operational | Site activation, enrolment, queries, monitoring | One row per site per period | Company-specific | Trial delivery performance |
| Submission | SDTM, ADaM, define.xml, reviewer's guide | One row per subject per domain record | CDISC, eCTD | What the regulator receives |

The table is the map that most organisations keep in separate documents. Writing it once, with the
authority column, prevents the recurring argument about which system is right. The authority column
is question-specific: the safety database is authoritative for reporting, the clinical database is
authoritative for what the site recorded, and the source record is authoritative for what happened.

## 5. The mechanics, step by step

This procedure builds a domain map for a programme and then reconciles two domains that disagree.
It is written for a clinical data scientist supporting a study team, but applies to any
multi-source evidence problem.

### Step 1. State the decision and the variables it requires

**Input:** the decision, the audience, and the timing.

**Output:** a list of required variables with their definitions, grain, and population.

**Rationale:** Data mapping without a decision produces an inventory, not a design. A safety
committee needs event-level data with causality and seriousness; a feasibility team needs
patient-level counts with diagnosis dates and treatment lines. The same domain serves both at
different grains.

**Branch point:** if the decision requires a variable that no domain captures, stop and record the
gap. A proxy may be acceptable, but it must be labelled as a proxy with its validation status.

### Step 2. Inventory the domains, owners, and access paths

**Input:** the programme's systems and contracts.

**Output:** a table with domain, owner, system, grain, key, standard, latency, access route, and
retention rule.

**Rationale:** Ownership determines who can resolve a discrepancy. Access determines whether the
analysis is feasible within the timeline. Retention determines whether the data will exist when a
regulator asks, which may be years after the analysis.

### Step 3. Fix the grain and the key before any join

**Input:** the inventory.

**Output:** a written statement of grain and key for each domain, plus the join strategy.

**Rationale:** Most silent data errors are join errors. A claims line table joined to a patient
table without deduplication inflates counts; an adverse event table joined to a subject table on
subject identifier alone duplicates events when the same identifier exists in two studies.

**Edge case:** re-screened patients. A patient who fails screening and returns may receive a new
subject identifier. Without a screening log that links the identifiers, the same person is counted
twice, and the safety narrative contains two histories.

### Step 4. Map terminologies and record versions

**Input:** the coding dictionaries and controlled terminology versions in use.

**Output:** a terminology map with the version, the date of the version, and the migration rule for
each domain.

**Rationale:** MedDRA and WHO Drug release new versions periodically. A study that starts on one
version and ends on another contains terms that are not directly comparable. The version is
metadata, not a detail; a submission that does not state it cannot be reproduced.

### Step 5. Build the lineage record

**Input:** the mapping specifications and transformation code.

**Output:** a lineage table with one row per output variable: source variable, transformation,
derivation rule, and the document or code that implements it.

**Rationale:** The lineage record is what allows a reviewer, an auditor, or a successor analyst to
reproduce the number. In a submission, define.xml carries much of this for SDTM and ADaM; internal
dashboards usually have nothing equivalent, which is why their numbers cannot be defended.

### Step 6. Define the reconciliation frame

**Input:** two domains that report the same quantity.

**Output:** a common frame: the same time window, the same population, the same event definition,
and the same unit.

**Rationale:** Almost every apparent discrepancy shrinks when the frame is aligned. The safety
database is live; the clinical database is cut. One includes events after the cut; the other does
not. Aligning the frame is not cosmetic; it determines whether the remaining difference is a real
data quality problem.

### Step 7. Match records and classify residuals

**Input:** the aligned frames.

**Output:** matched records, records only in A, records only in B, and the reasons for each
residual category.

**Rationale:** A count difference is not a finding. A classified difference is. Categories that
recur include timing (after the cut), duplicate reporting, missing reporting to one system,
different coding, different event splitting, and different date conventions.

**Branch point — classification of a residual (more than two outcomes):**

| Residual pattern | Likely cause | Required action | Residual category |
|---|---|---|---|
| Same subject, same term, onset within 1 day | Date convention or data entry | Confirm against source | Timing |
| Same subject, same term, onsets 30 days apart | Event recorded twice | Merge if same event, else keep both | Duplicate |
| Present in safety, absent in clinical | Not entered in eCRF | Query site; add to clinical database | Missing capture |
| Present in clinical, absent in safety | Not reported to pharmacovigilance | Assess seriousness and reporting obligation | Missing report |
| Different term, same event | Dictionary version or coding choice | Re-code with the current version | Coding |
| One event in A, two in B | Split reporting of signs and diagnosis | Link the records | Granularity |
| Same subject identifier, different study | Identifier reuse | Resolve via screening log | Identity |

### Step 8. Compute agreement at the right denominator

**Input:** the classified residuals.

**Output:** event-level and subject-level agreement rates, each with its numerator, denominator,
and population.

**Rationale:** A percentage without its denominator hides the decision. "98% agreement" on 50
events means one discrepancy; on 5,000 events it means 100. Report both the rate and the counts.

### Step 9. Set an acceptance threshold and a resolution path

**Input:** the agreement rates and the criticality of the data.

**Output:** a documented threshold, a list of unresolved items, and an owner and date for each.

**Rationale:** Zero discrepancy is not a realistic target for a live safety database. The threshold
should depend on the decision: a submission-critical endpoint requires near-complete agreement; an
exploratory analysis may tolerate more.

### Step 10. Handle the edge cases that break reconciliation

**Screen failures.** Screen failures appear in the screening log but not in the randomised
population. A reconciliation that includes them inflates the subject count relative to the
analysis dataset.

**Protocol deviations.** A deviation may remove a subject from the per-protocol population while
leaving them in the safety population. Counts by population will differ by design.

**Missing visits.** A missing visit produces a missing row, not a zero. Aggregations that treat
missing as zero understate the denominator.

**Partial dates.** An event with a month-year onset cannot be placed inside a narrow window without
an imputation rule. The rule must be pre-specified and applied consistently in both domains.

**Concomitant medication coding.** A change in WHO Drug version can move a medication between
classes, changing the count of prohibited concomitant medications and therefore the per-protocol
exclusions.

**External controls and data drift.** A historical control cohort collected under a different
standard of care is not comparable on event rates without adjustment. The drift is a data property,
not a statistical nuisance.

**Unblinding.** Reconciliation before unblinding must not reveal treatment allocation. Aggregations
by arm are prohibited until the blind is broken for the analysis.

**Database lock.** After lock, changes require a formal unlock with documented justification. A
reconciliation performed after lock but before submission must record whether it changed any
analysis dataset.

### Step 11. Document, version, and retain

**Input:** the reconciliation and its outputs.

**Output:** a versioned reconciliation report, the code, the input dataset versions, and the
retention location.

**Rationale:** The report is the evidence that the numbers were reconciled. Without a version and a
dataset identifier, the report cannot be tied to the submission it supports.

### Step 12. Re-run at each data cut

**Input:** the new cut and the previous reconciliation.

**Output:** a delta report: what changed, why, and whether the conclusion changed.

**Rationale:** Reconciliation is not a one-time event. Each cut changes the frame, and a
reconciliation that was clean at the interim may not be clean at the final analysis.

### 5.1 Quality checks that belong inside the procedure

The twelve steps describe the flow. The checks below are the ones that catch the errors which
survive the flow. Each is cheap to run and expensive to skip.

| Check | Rule | What it catches | Where it belongs |
|---|---|---|---|
| Key uniqueness | The stated key is unique at the stated grain | Duplicate rows from joins or re-entry | Step 3 |
| Referential integrity | Every subject in an event domain exists in the demographics domain | Orphan records from partial transfers | Step 5 |
| Date ordering | Onset precedes resolution; consent precedes first dose | Entry errors and partial dates | Step 6 |
| Date range | Dates fall within the study window plus a documented allowance | Transposed day and month | Step 6 |
| Unit consistency | Units match the protocol and the normal range source | Milligrams versus micrograms | Step 4 |
| Coded term validity | Every term exists in the stated dictionary version | Coding against a retired version | Step 4 |
| Visit windows | Visits fall within protocol windows or are flagged | Mis-scheduled visits treated as on-time | Step 6 |
| Required variables | No nulls in variables the analysis requires | Structural missingness | Step 5 |
| Cross-domain counts | Event counts by subject reconcile across domains | Systematic capture gaps | Step 7 |
| Duplicate persons | Screening log links re-screened identifiers | Double-counted patients | Step 3 |

Two of these deserve emphasis. Key uniqueness is the check that prevents the most common silent
error: a join that multiplies rows because the key was not unique at the grain. Referential
integrity is the check that catches partial data transfers, which are common when a safety
database and a clinical database are extracted on different schedules.

### 5.2 Populations and their effect on counts

Every count in a reconciliation must name its population. The definitions below follow common
regulatory usage; the protocol defines the exact rule for a given study.

| Population | Definition | Typical size relative to randomised | Common use |
|---|---|---|---|
| Screened | All patients who signed consent | Larger by the screen-failure fraction | Feasibility and recruitment |
| Randomised | All patients allocated to a study intervention | Reference denominator | Disposition reporting |
| Safety | All patients who received at least one dose | Slightly smaller than randomised | Adverse event rates |
| Intention-to-treat (ITT) | All randomised patients, analysed as randomised | Equal to randomised | Primary efficacy |
| Modified ITT (mITT) | ITT with a pre-specified restriction, e.g. biomarker-positive | Smaller, sometimes much smaller | Efficacy in a defined subset |
| Per-protocol (PP) | Patients who completed without major deviation | Smaller and selected | Sensitivity analysis |
| Completers | Patients who reached the end of the planned treatment | Smallest | Supportive analyses |

A reconciliation that compares a safety population count with an ITT count will always show a
difference, and the difference is not a data error. The 1,204 randomised patients in the worked
example may correspond to 1,198 dosed (safety), 1,204 ITT, 1,150 PP, and 1,090 completers. Four
different denominators, four different rates for the same event count.

### 5.3 The reconciliation report

A reconciliation that is not written down did not happen, as far as a reviewer is concerned. The
report should be short and should contain:

1. **Scope** — the domains, the study, the data cut dates, and the dataset versions.
2. **Frame** — the population, time window, event definition, and unit used on each side.
3. **Matching rule** — the key and the tolerance, stated precisely enough to be re-run.
4. **Results** — matched, residual by category, union, and agreement rates at event and subject
   level, each with its denominator.
5. **Unresolved items** — owner, action, and date.
6. **Conclusion** — whether the domains are fit for the intended decision, and any limitation the
   decision must carry.

The report is an appendix to the clinical study report or the submission, not a working document
that disappears into a shared drive. Its version must match the dataset version it describes.

### 5.4 When the domains cannot be reconciled

Sometimes the disagreement is real and cannot be resolved before the decision. The procedure then
requires a documented limitation rather than a forced match:

- State the residual and its likely cause.
- State the direction of the bias, if it can be determined.
- State the sensitivity of the decision to the residual: would the conclusion change if all
  residual events fell on one side?
- Record who accepted the limitation and when.

A forced match is worse than an acknowledged gap, because it destroys the evidence that the gap
exists.

### 5.5 Lineage in practice: one variable, five representations

Take a single variable, the start date of an adverse event. Its journey illustrates every
transformation the lineage record must capture.

| Layer | Representation | Rule applied | Common failure |
|---|---|---|---|
| Source | Site entry in the eCRF: day, month, year | Entered as recorded, including partial dates | Day and month transposed |
| Clinical database | Raw field plus query status | Partial date flag set; no imputation yet | Imputation applied silently at entry |
| SDTM AE | AESTDTC in ISO 8601 | Partial dates represented as YYYY-MM or YYYY | Imputed date written as complete |
| ADaM ADAE | ASTDT and ASTDTF | Imputation rule from the SAP; flag records imputation | Flag missing, so the imputation is invisible |
| Output | Table of time to event | Derived from ASTDT with the analysis window | Window applied differently from the SAP |

The failure modes are not independent. A transposed date at source propagates to the table. An
imputation applied at entry means the SDTM date looks complete and the ADaM flag says no imputation
occurred, so the analysis cannot distinguish a real date from a constructed one. The lineage record
must therefore capture not only the transformation but also the *flag* that records it.

### 5.6 Retention, access, and the life of the data after the trial

Data governance does not end at database lock. Three obligations continue:

- **Retention.** Records must be kept for the period required by the jurisdiction. As examples,
  US investigational drug regulations require records for two years after a marketing application
  is approved or the investigation is discontinued (21 CFR 312.57), while the EU Clinical Trials
  Regulation sets a longer archiving period for trial master file content. The exact period is a
  legal question for the sponsor's regulatory function, not a data management preference.
- **Access.** Investigators must retain access to their source records, and sponsors must be able
  to provide data to regulators on request. A data platform that stores only derived values cannot
  satisfy either obligation.
- **Destruction.** Destruction must be documented and must not occur while an investigation, a
  litigation hold, or a regulatory request is open.

A data strategy that treats retention as an infrastructure cost will eventually delete the only
copy of a source record during a period when it is required.

### 5.7 Choosing a model for internal analytics

Trial data and real-world data rarely share a model. Four options recur, and the choice determines
what can be joined without transformation.

| Model | Designed for | Strength | Cost |
|---|---|---|---|
| SDTM | Regulatory tabulation | Prescribed structure, reviewer familiarity | Not normalised for cross-study analysis; one row per event without a subject-level spine |
| ADaM | Pre-specified analysis | Traceable derivations, analysis-ready | Study-specific; not designed for pooling |
| OMOP CDM | Multi-source observational analysis | Common vocabulary, established tooling | Requires mapping from source vocabularies; loses some trial-specific detail |
| Star schema (fact and dimension) | Internal reporting | Fast queries, flexible dimensions | Bespoke; must be documented and governed like any other model |

The common mistake is to use SDTM as the internal analytics model because it is the regulatory
standard. SDTM is optimised for submission review, not for cross-study queries; its domains are
wide, its keys are study-specific, and pooling requires harmonisation that the model does not
provide. A programme that wants both submission and analytics needs a mapping from the tabulation
to an analytics model, and a lineage record that connects them.

### 5.8 Roles, decisions, and the RACI that prevents disputes

Reconciliation fails organisationally more often than technically. A simple responsibility matrix
prevents the two most common failures: nobody owns the frame, and nobody owns the residual.

| Activity | Clinical data management | Pharmacovigilance | Biostatistics | Regulatory |
|---|---|---|---|---|
| Define the frame and cut | Responsible | Consulted | Accountable | Informed |
| Deduplicate safety reports | Informed | Responsible | Consulted | Informed |
| Match and classify | Responsible | Responsible | Consulted | Informed |
| Set the acceptance threshold | Consulted | Consulted | Accountable | Consulted |
| Resolve capture gaps | Responsible | Consulted | Informed | Informed |
| Resolve reporting gaps | Informed | Responsible | Informed | Accountable |
| Sign the reconciliation report | Consulted | Consulted | Accountable | Informed |

Two rules make the matrix work. First, the frame is agreed before matching begins, not after the
numbers disagree. Second, the person accountable for the threshold is not the person responsible
for producing the data, so the acceptance decision is independent of the group being measured.

## 6. Worked example

> **Note:** The study, sponsor, and event counts are fictional and constructed for teaching. The
> standards and regulatory documents cited are real; the arithmetic is reproducible with the SQL in
> Section 7.

### 6.1 The study and the disagreement

A Phase III study of an oral anticoagulant in 1,204 randomised patients completed. The clinical
data cut for the primary analysis was 31 December. The safety database is live through 31 March.
At the safety review meeting, two counts are presented for treatment-emergent adverse events:

- Clinical database (SDTM AE domain, extracted at the 31 December cut): 398 events in 152
  subjects.
- Safety database (case reports received through 31 March): 412 events in 156 subjects.

The committee asks which number is correct.

### 6.2 Step 6: align the frame

The safety database is filtered to events with onset on or before 31 December, and duplicate case
reports are removed.

| Step | Count | Subjects |
|---|---:|---:|
| Safety database as extracted | 412 | 156 |
| Remove duplicate reports (same subject, same event, reported twice) | −3 | 0 |
| Remove events with onset after 31 December | −14 | −2 |
| Safety database in frame | 395 | 154 |

The clinical database is already cut at 31 December, so no adjustment is needed. Its 398 events are
in 152 subjects.

The naive comparison (412 versus 398) suggests the clinical database is missing 14 events. The
frame-aligned comparison (395 versus 398) suggests the two are close, with differences in both
directions.

### 6.3 Step 7: match and classify

Matching rule: same subject identifier, same MedDRA preferred term, onset date within 1 day.

| Category | Events | Subjects |
|---|---:|---:|
| Matched | 391 | 150 |
| In safety only — not entered in the eCRF | 4 | 4 |
| In clinical only — not reported to pharmacovigilance | 7 | 2 |
| **Union** | **402** | **158** |

Check the arithmetic:

- Safety in frame: 391 matched + 4 safety-only = 395. Correct.
- Clinical: 391 matched + 7 clinical-only = 398. Correct.
- Union: 391 + 4 + 7 = 402.
- Event agreement: 391 / 402 = 97.26%.
- Event discrepancy: 11 / 402 = 2.74%.
- Subject agreement: 150 / 158 = 94.94%.

The 7 clinical-only events break down as 5 never reported to pharmacovigilance and 2 that were
reported but split into separate signs (nausea and vomiting) while the clinical database holds a
single event coded to the diagnosis. The 4 safety-only events were reported by the site to the
sponsor's case intake but never entered in the eCRF.

### 6.4 What the numbers mean

The discrepancy is 11 events, not 14. Of those, 4 are safety-database-complete and clinical-
database-incomplete (a capture problem), 5 are clinical-database-complete and safety-incomplete (a
reporting compliance problem), and 2 are a granularity difference that may not be a data error at
all.

The reporting compliance item is the serious one. Five events that the site recorded in the trial
database were not reported to pharmacovigilance. Whether that is a regulatory breach depends on
seriousness and expectedness: a serious, unexpected event must be reported within expedited
timelines. The reconciliation has surfaced a compliance question that a count comparison would
have buried in "14 missing".

### 6.5 Two diagnostics that reveal a broken reconciliation

**Diagnostic 1 — the frame is not aligned.** If the reconciliation compares the live safety
database with the cut clinical database, the difference includes every event after the cut. In this
example, that is 14 events and 2 subjects, which is larger than the true discrepancy of 11. The
diagnostic is to compute the discrepancy as a function of the cut date: if it grows steadily as the
safety database advances, the frame is wrong.

**Diagnostic 2 — matching on totals rather than records.** A reconciliation that compares 412 with
398 and reports "14 missing" has not matched anything. It cannot distinguish a duplicate, a timing
difference, a coding difference, and a genuine omission, and it will assign all 14 to the wrong
cause. The diagnostic is to require a match rate on identifiers: if the matched count plus the
classified residuals does not equal the union, the match logic is incomplete.

A third check: the subject-level and event-level rates should be consistent in direction. Here,
event agreement (97.26%) is higher than subject agreement (94.94%), which is expected when a few
subjects have multiple discrepant events. If subject agreement were higher than event agreement
with the same totals, the join would be duplicating rows.

### 6.6 Vary one assumption: the coding dictionary

The study started on MedDRA version 26.0 and the safety database migrated to version 27.0 in
January, after the cut. The clinical database remains on 26.0 for the analysis. Six events coded to
"Rash" in 26.0 map to "Rash maculo-papular" in 27.0, and four events coded to "Pruritus" map to
"Pruritus allergic".

Under the old version, the clustered term "Rash" has 18 events. Under the new version, it has 12.
A safety reviewer looking at the new-version count would see a 33% reduction in rash events and
might conclude that the signal has weakened. The correct interpretation is that the dictionary
changed, not the patients.

The reconciliation decision changes: the event-level agreement rate computed on preferred terms is
97.26% before the version check and 94.5% after re-coding the six events to a common version
(11 + 6 = 17 discrepant of 402 = 4.23% discrepancy, 95.77% agreement). The programme must either
freeze the analysis on one version or bridge the versions with a documented mapping. Either choice
is defensible; not choosing is not.

### 6.7 What the example establishes

1. A count difference is not a finding until the frame is aligned and the records are matched.
2. The residual categories matter more than the total: they route to different owners and
   different remedies.
3. The denominator changes the interpretation: 11 of 402 events is a different claim from 14 of
   412.
4. A dictionary version change can look like a safety signal change.
5. The reconciliation is a governance artefact, not a data-cleaning step.

### 6.8 The check a reviewer actually runs: deaths

In a submission, the count of deaths is checked across three places: the demographics domain flag,
the adverse event domain's fatal outcome flag, and the safety database. The counts should agree
exactly, and any difference must be explained subject by subject.

| Source | Count | Definition |
|---|---:|---|
| Demographics, death flag | 27 | Subjects with a recorded death |
| Adverse events, fatal outcome | 25 | Subjects with at least one event coded fatal |
| Safety database, fatal cases | 28 | Cases with a fatal outcome, through 31 March |

Matching on subject identifier gives 24 subjects present in all three, 3 in demographics only, 1
in adverse events only, and 4 in the safety database only, a union of 32. Agreement is 24/32 =
75%. For a variable of this criticality, 75% is not a reconciliation; it is an escalation.

Two of the three demographics-only deaths occurred in January, after the clinical cut, and are
removed when the frame is aligned. That leaves one genuine gap: a death recorded in demographics
with no corresponding fatal adverse event, which is a data error in the adverse event domain. The
four safety-only deaths are cases reported to pharmacovigilance but absent from the clinical
database; two occurred after the cut, and two are unreported trial deaths that require immediate
site follow-up.

The frame-aligned agreement is 24/30 = 80%. Still not acceptable for deaths, and the report should
say so plainly rather than presenting the rate as reassuring.

The residual categories also determine who acts. The table below is the escalation route that
should accompany the report.

| Residual | Owner | Action | Deadline |
|---|---|---|---|
| Death in demographics, no fatal adverse event | Clinical data management | Query site; correct the event record or the flag | Before database lock |
| Death reported to pharmacovigilance, absent from clinical database | Site and pharmacovigilance | Confirm the death; assess reporting compliance | Immediate |
| Death after the data cut | Biostatistics | Exclude from the cut-aligned comparison; include in the final analysis | At the next cut |
| Fatal event in a subject not flagged as deceased | Clinical data management | Reconcile the disposition record | Before database lock |

### 6.9 What the reconciliation does not prove

The reconciliation establishes that two systems describe the same events consistently. It does not
establish that the events occurred as described. Four limits apply:

- **Common-source error.** Both systems may derive from the same mis-entered source value. Agreement
  is then high and wrong.
- **Definitional agreement.** Matching on preferred term means two different clinical events coded
  to the same term are counted as agreeing. At the level of the case narrative, they may not be the
  same event.
- **Frame selection.** Choosing a cut date changes which events are in scope. A reconciliation that
  looks clean at one cut may not be clean at the final analysis.
- **Silent omissions.** An event absent from both systems leaves no residual to classify. Only
  source verification on a sample can detect it.

The report should state that the conclusion is about consistency between systems, not about
completeness against the clinical record. Conflating the two is how a reconciliation becomes a
false assurance.

## 7. Code, query, or calculation

The SQL below runs on SQLite 3.39 or later (FULL OUTER JOIN support is not needed for this version,
but the `julianday` function and CTEs are). It builds two adverse event tables from a small
synthetic study, aligns the frame, deduplicates the safety reports, matches with a one-day
tolerance, and counts the residuals. Every number can be checked by hand from the inserted rows.

```sql
CREATE TABLE safety_ae (
  case_id    TEXT PRIMARY KEY,
  subjid     TEXT NOT NULL,
  ae_term    TEXT NOT NULL,
  onset_date TEXT NOT NULL,
  serious    TEXT NOT NULL,
  meddra_ver TEXT NOT NULL
);

CREATE TABLE sdtm_ae (
  subjid     TEXT NOT NULL,
  aeseq      INTEGER NOT NULL,
  ae_term    TEXT NOT NULL,
  aestdtc    TEXT NOT NULL,
  aeser      TEXT NOT NULL,
  meddra_ver TEXT NOT NULL,
  PRIMARY KEY (subjid, aeseq)
);

-- Excerpt of the inserted rows (15 safety reports, 15 clinical events)
INSERT INTO safety_ae VALUES
 ('C01','S001','Rash','2025-03-01','Y','26.0'),
 ('C02','S001','Nausea','2025-03-05','N','26.0'),
 ('C11','S008','Dizziness','2025-10-01','N','26.0'),
 ('C12','S008','Dizziness','2025-10-01','N','26.0'),  -- duplicate report
 ('C15','S010','Rash','2026-01-15','N','27.0');       -- after the cut

WITH s AS (
  SELECT case_id, subjid, ae_term, onset_date
  FROM safety_ae
  WHERE onset_date <= '2025-12-31'
    AND case_id NOT IN (
        SELECT b.case_id FROM safety_ae b
        JOIN safety_ae a
          ON a.subjid = b.subjid AND a.ae_term = b.ae_term
         AND a.onset_date = b.onset_date AND a.case_id < b.case_id)
),
c AS (
  SELECT subjid, aeseq, ae_term, aestdtc AS onset_date
  FROM sdtm_ae WHERE aestdtc <= '2025-12-31'
),
m AS (
  SELECT s.case_id, s.subjid, s.ae_term
  FROM s JOIN c
    ON s.subjid = c.subjid AND s.ae_term = c.ae_term
   AND abs(julianday(s.onset_date) - julianday(c.onset_date)) <= 1
  GROUP BY s.case_id
),
classified AS (
  SELECT s.subjid, s.ae_term,
         CASE WHEN m.case_id IS NULL THEN 'safety only' ELSE 'matched' END AS status
  FROM s LEFT JOIN m ON s.case_id = m.case_id
  UNION ALL
  SELECT c.subjid, c.ae_term, 'clinical only'
  FROM c
  WHERE NOT EXISTS (SELECT 1 FROM m
                    WHERE m.subjid = c.subjid AND m.ae_term = c.ae_term)
)
SELECT status, COUNT(*) AS n FROM classified GROUP BY status
UNION ALL
SELECT 'union', COUNT(*) FROM classified;
```

Executed output:

```text
Reconciliation by category
  clinical only 2
  matched       12
  safety only   1
  union         15
```

Reading the result: 12 matched, 1 safety-only, 2 clinical-only, union 15, so event agreement is
12/15 = 80%. The 15 rows of each source reduce to 14 after the cut and deduplication on the safety
side, and the arithmetic reconciles: 12 + 1 = 13 safety events in frame (14 unique minus one
post-cut duplicate removal), and 12 + 2 = 14 clinical events in frame.

Two cautions about the query itself.

- `GROUP BY s.case_id` guarantees one match per safety case. Without it, a clinical database with
  two records for the same event would duplicate the safety case and inflate the matched count.
- The `NOT EXISTS` for clinical-only records matches on subject and term, not on the same case
  identifier, because the clinical database has no case identifier. That is exactly the asymmetry
  the reconciliation must document.

The second query illustrates why term counts disagree even when the records reconcile:

```sql
SELECT ae_term, meddra_ver, COUNT(*) FROM (
  SELECT ae_term, meddra_ver FROM safety_ae WHERE onset_date <= '2025-12-31'
  UNION ALL
  SELECT ae_term, meddra_ver FROM sdtm_ae WHERE aestdtc <= '2025-12-31')
GROUP BY ae_term, meddra_ver ORDER BY ae_term, meddra_ver;
```

```text
Anaemia     27.0  2
Dizziness   26.0  3
Fatigue     27.0  1
Headache    26.0  2
Nausea      26.0  2
Nausea      27.0  2
Pruritus    27.0  1
Pyrexia     27.0  2
Rash        26.0  6
Rash        27.0  5
Vomiting    26.0  2
```

Dizziness appears three times although only one dizziness event is in the reconciled frame,
because this query does not deduplicate safety reports. That is not a bug in the query; it is a
different question. Counts of *reports* and counts of *events* differ, and a report that does not
state which it is being asked for will be interpreted differently by different readers.

Assumptions of the code: dates are stored as ISO text, the cut date is 2025-12-31, matching
tolerates one day of date difference, and duplicates are identified by identical subject, term, and
onset date. Each assumption is a decision that should be recorded in the reconciliation report.

## 8. Data, standards, and regulatory context

Pharma data standards fall into four layers: exchange, terminology, tabulation and analysis, and
governance. A programme needs all four, and they are maintained by different organisations with
different release cycles.

| Layer | Standard | Maintained by | What it constrains | What it does not |
|---|---|---|---|---|
| Exchange | FHIR (Fast Healthcare Interoperability Resources, HL7) | HL7 | Structure of healthcare data exchange, resources, APIs | Analytics models or trial tabulation |
| Exchange | DICOM | DICOM Standards Committee | Medical images and metadata | Image interpretation |
| Terminology | MedDRA | ICH Management Board | Adverse event coding, hierarchy, versioning | Disease diagnosis in routine care |
| Terminology | WHO Drug Dictionary | Uppsala Monitoring Centre | Medicinal product coding | Procedures |
| Terminology | LOINC | Regenstrief Institute | Laboratory test and observation identifiers | Result interpretation |
| Terminology | SNOMED CT | SNOMED International | Clinical findings, procedures, substances | Billing |
| Terminology | ICD-10 / ICD-11 | WHO | Morbidity classification and reimbursement | Granular clinical coding |
| Capture | CDASH | CDISC | Standard fields on case report forms | Downstream analysis |
| Tabulation | SDTM | CDISC | Structure of submission tabulations | Analysis-ready derivations |
| Analysis | ADaM | CDISC | Structure and traceability of analysis datasets | Study design |
| Metadata | Define-XML | CDISC | Machine-readable dataset and variable metadata | Analysis content |
| Nonclinical | SEND | CDISC | Structure of animal study data | Clinical data |
| Safety reporting | E2B(R3) | ICH | Electronic transmission of individual case safety reports | Case narrative quality |
| Governance | ICH E6(R3) | ICH | GCP, data governance, computerised systems, audit trails | Commercial decisions |
| Electronic records | 21 CFR Part 11 | US FDA | Electronic records and signatures in FDA-regulated activities | EU record rules |
| Privacy | GDPR; HIPAA | EU; US | Personal data processing, de-identification, transfer | Trial data standards |

### 8.1 What is binding and what is convention

- **Binding law.** 21 CFR Part 11 applies to electronic records submitted to the FDA. The GDPR
  applies to personal data of EU residents, including trial data. HIPAA governs protected health
  information held by covered entities in the US. These are legal instruments, not guidance.
- **Regulatory expectation.** Submitting SDTM and ADaM datasets with Define-XML is expected by
  FDA and EMA for marketing applications, expressed through guidance and the FDA Data Standards
  Catalog rather than through a statute naming CDISC. A sponsor that submits a different structure
  will be asked to justify it, and may be asked to resubmit.
- **Convention.** Using one EDC system across a programme, coding to the latest dictionary at study
  start, or maintaining a single data lake are operational choices. They may be good practice;
  they are not requirements.

The distinction matters when a vendor says a product is "21 CFR Part 11 compliant". Part 11 does
not certify products. It sets requirements on records and signatures, and the sponsor must validate
the system for its intended use. A vendor certificate is evidence about the vendor's development
process, not a regulatory determination.

### 8.2 The governance requirements that changed in 2025

ICH E6(R3) was adopted at Step 4 on 6 January 2025, with an error-correction version dated 24
October 2025. The EMA Step 5 version took effect on 23 July 2025, and the FDA issued its
implementation guidance in September 2025. The guideline adds a dedicated data governance section
covering:

- data capture and relevant metadata, including audit trails;
- review of data and metadata, planned and risk-based;
- data corrections, transfer, migration, and finalisation before analysis;
- computerised systems, including risk-based validation and user management.

The guideline does not use the acronym ALCOA+ literally. It requires source records to be
attributable, legible, contemporaneous, original, accurate, and complete, and its glossary defines
data integrity to include secure and reliable data fit for purpose. The practical consequence is
that a data flow diagram and an audit trail review strategy are now expected artefacts, not
optional documentation.

### 8.3 Electronic source data

The FDA's 2013 guidance *Electronic Source Data in Clinical Investigations* recommends capturing
source data electronically and describes the controls needed: authorised data originators, data
element identifiers recording originator and time, human-readable audit trails that do not obscure
prior entries, investigator review and electronic signature, and documented data flows. The
guidance is a recommendation, not a regulation, but it describes the standard against which an
inspection will assess an electronic source system.

### 8.4 What the standards do not solve

Standards make data exchangeable, not correct. SDTM constrains where a value goes, not whether it
is true. MedDRA constrains the vocabulary, not the coding decision. Define-XML documents a
derivation, not whether the derivation matches the analysis plan. The reconciliation discipline in
Section 5 is therefore necessary even in a fully standardised programme, because standards
eliminate structural disagreement and leave semantic disagreement intact.

### 8.5 Privacy and de-identification

Trial data and real-world data are subject to different privacy regimes, and the difference affects
what can be joined.

- **Trial data** are personal data under the GDPR when the trial involves EU residents, even when
  the sponsor holds them under a protocol. Consent language and the legal basis must cover the
  analyses performed, including secondary use.
- **Real-world data** may be de-identified under HIPAA in the US (Safe Harbor or Expert
  Determination) or pseudonymised under the GDPR. Pseudonymised data remain personal data; a
  key-coded dataset is not anonymous.
- **Linkage** between a trial and a real-world dataset usually requires a trusted third party or a
  privacy-preserving technique, because the sponsor holding both keys can re-identify.

A data platform that pools domains without a privacy assessment creates a re-identification risk
that no amount of access control fully removes. The assessment belongs in the design, not in the
legal review at the end.

## 9. Industry practice and case evidence

### 9.1 How the domains are actually organised

Most large sponsors operate a clinical data warehouse fed by EDC, a separate safety database
(typically a commercial system such as Argus or ArisGlobal), a laboratory data pipeline, and a
real-world data environment built on claims and EHR purchases. The systems are linked by
subject-level identifiers within a study and by probabilistic linkage outside it, if at all.

Three organisational patterns are visible:

- **Study-centric.** Each study has its own data management, its own standards deviations, and its
  own reconciliation. Cheap to start, expensive to pool.
- **Platform-centric.** A central standards team owns the tabulation model and the metadata, and
  studies configure to it. More expensive at the start, faster at submission.
- **Data-product-centric.** Domain teams own datasets as products with consumers, service levels,
  and lineage. Requires governance maturity that few organisations have.

These are inferences from observable job structures and public descriptions of data platforms,
not findings from a controlled study. No peer-reviewed comparison of the three models exists.

### 9.2 Public evidence on data quality problems

Regulatory inspection findings are one of the few public sources of evidence about data practice.
The FDA publishes inspection classifications and, for some inspections, the deficiencies observed.
Recurring themes in published inspection outcomes include failure to maintain adequate case
histories, failure to follow the investigational plan, and inaccurate records. These are
documented outcomes of inspections, not estimates of prevalence; the denominator of inspections is
published but the selection of which sites are inspected is not random.

The practical reading is that data problems are found at the level of the source record, not at the
level of the standard. A programme can produce perfect SDTM and still fail an inspection because
the source did not support the value.

### 9.3 Case: the same event in two systems

The worked example is fictional, but the pattern is common enough to be described in data
management literature and in inspection findings: a serious event is reported to the safety
organisation within the expedited timeline and entered in the clinical database weeks later, or not
at all. The two records then differ in date, term, and seriousness.

**Documented:** the regulatory obligation to report serious unexpected suspected adverse reactions
within expedited timelines, and the requirement that trial records be accurate and complete.
**Inference:** that most such discrepancies are process gaps between the safety and clinical
workflows rather than fabrication. The inference matters because the remedy differs: a process
fix, not a monitoring escalation.

### 9.4 What "data integrity" means in practice

Data integrity is often presented as an audit trail problem. In practice the failures that change
conclusions are more mundane:

| Failure | Frequency in practice | Detectability | Consequence |
|---|---|---|---|
| Duplicate subject or event | Common | High with key checks | Inflated counts, biased rates |
| Date inconsistency | Common | High with range and ordering checks | Misclassified treatment-emergent events |
| Unit inconsistency | Occasional | High with unit dictionaries | Wrong magnitude, sometimes wrong direction |
| Coding version drift | Common in long studies | Medium | Apparent change in safety profile |
| Missing visits treated as zero | Common in dashboards | Low without metadata | Understated denominators |
| Unflagged imputation | Occasional | Low | Fabricated precision in dates and values |
| Unreconciled domains | Common | Medium | Two numbers in one submission |

The pattern is that the most consequential failures are the least exotic. A programme that invests
in duplicate detection, date logic, and frame alignment will prevent more wrong conclusions than
one that invests in a blockchain audit trail.

### 9.5 What a sponsor actually holds for one study

A useful exercise before any platform design is to list the artefacts that exist for a single
completed study. The list is longer than most data strategies assume.

| Artefact | Typical owner | Format | Retention driver |
|---|---|---|---|
| Protocol and amendments | Clinical operations | PDF, structured protocol | Trial master file |
| eCRF specifications and annotations | Data management | PDF, XML | Trial master file |
| Clinical database | Data management | Relational export | GCP and inspection |
| Safety database cases | Pharmacovigilance | E2B XML, narrative | Pharmacovigilance law |
| Central laboratory data | Laboratory vendor | CSV, SAS | Contract and GCP |
| Imaging reads | Imaging vendor | DICOM, CSV | Contract and GCP |
| SDTM and ADaM datasets | Statistical programming | SAS, XPT | Submission regulation |
| Define-XML and reviewer's guide | Statistical programming | XML, PDF | Submission regulation |
| Analysis code and logs | Biostatistics | Scripts, logs | Reproducibility policy |
| Reconciliation reports | Data management | PDF | GCP and inspection |
| Monitoring and query records | Clinical operations | System export | GCP |
| Vendor contracts and data specifications | Procurement | PDF | Contract law |

Most organisations can produce the first six from memory and cannot produce the last six. The
distribution of artefacts across owners is the reason a "single source of truth" project is a
governance programme, not an engineering one.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Joining on a non-unique key | Row counts exceed either input after a join | Key not unique at the grain | Inflated counts and rates | Assert uniqueness before joining |
| Comparing domains at different cuts | Discrepancy grows as one domain updates | Live versus frozen data | False data quality finding | Align the frame first |
| Ignoring dictionary versions | Term counts shift after a migration | MedDRA or WHO Drug release | Apparent safety signal change | Record versions; bridge or freeze |
| Treating missing as zero | Denominators differ across tables | Aggregation over absent rows | Understated rates | Carry denominators explicitly |
| Subject identifier reuse | Same ID in two studies or after re-screening | No global identifier | Duplicate or merged patients | Screening log; study-qualified keys |
| Unflagged imputation | Dates look complete | Imputation applied at entry | Fabricated precision | Imputation flags in ADaM |
| Source not retained | Only derived values in the platform | Lake stores extracts | Inspection finding; irreproducible analysis | Retain source or certified copies |
| Uncontrolled access | Analysts query unblinded data | Weak role-based access | Loss of blinding | Access controls and firewalls |
| Silent schema change | Dashboard number moves without a code change | Upstream field renamed or retyped | Unexplained trend | Schema versioning and tests |
| Lineage documented after the fact | Mapping document postdates the dataset | Retrospective documentation | Errors in derivation persist | Build lineage with the pipeline |
| Over-merging real-world identities | Patient count falls after linkage | Probabilistic match threshold too loose | Understated population | Report match rates and thresholds |
| One number, two owners | Two functions quote different totals | No reconciliation frame | Loss of credibility | Single reconciliation report per quantity |

A red flag that deserves its own line is a dashboard that cannot state its population. If the
number cannot be reproduced with a stated denominator, grain, and cut date, it is a chart, not a
measurement.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| Subject identifiers are unique within a study | Joins depend on it | Re-screening, data entry error | Duplicate check on the identifier |
| Event terms are coded to one dictionary version | Term-level comparison | Mid-study migration | Version audit by domain |
| Dates are recorded with enough precision to match | Matching tolerance | Partial dates | Count of partial dates by domain |
| The clinical database cut is the analysis frame | Reconciliation boundary | Safety database is live | Re-run at each cut; delta report |
| Seriousness is recorded consistently | Case classification | Site interpretation differences | Compare seriousness by site |
| The safety database captures all serious events | Reporting completeness | Site under-reporting | Reconcile against clinical database |
| Source records are retained and accessible | Verification and inspection | Archive migration | Sample retrieval test |
| Linkage outside the trial is accurate | Real-world analyses | Probabilistic match error | Report match rate and threshold |
| Metadata describes the actual derivation | Reproducibility | Code changed without metadata update | Re-run derivation from metadata |
| The reconciliation threshold matches criticality | Acceptance decision | Uniform threshold across variables | Set thresholds by variable class |

The deepest limit is that reconciliation detects disagreement, not truth. Two systems can agree and
both be wrong, because both derive from the same mis-entered source value. Reconciliation must
therefore be paired with source data verification on a sample, and with the acknowledgement that
neither method proves correctness.

## 12. Exercises

**Exercise 1 — Domain map (recall, 30 minutes).**
Deliverable: a table of eight data domains with grain, key, standard, owner, and the decision each
supports.
Acceptance criteria: every domain has a grain stated as "one row per"; at least two domains share a
concept but differ in standard; no domain is described only by its system name.
Time: 30 minutes.

**Exercise 2 — Lineage trace (application, 40 minutes).**
Deliverable: a lineage record for one laboratory variable from eCRF field to analysis table, with
five columns: source, transformation, standard variable, derivation, and flag.
Acceptance criteria: at least four transformations listed; the imputation flag accounted for; the
record sufficient for a second analyst to reproduce the value.
Time: 40 minutes.

**Exercise 3 — Reconciliation design (design, 50 minutes).**
Deliverable: a one-page reconciliation protocol between a safety database and an SDTM adverse event
domain, including frame, matching key, residual categories, and acceptance threshold.
Acceptance criteria: frame states cut date and population; match rule is precise; at least five
residual categories with an owner; threshold justified by criticality.
Time: 50 minutes.

**Exercise 4 — Agreement arithmetic (application, 25 minutes).**
Deliverable: given 412 safety reports of which 3 are duplicates and 14 occur after the cut, and 398
clinical events of which 7 are unmatched, compute the in-frame safety count, the union, and the
event agreement rate.
Acceptance criteria: counts reconcile in both directions; rate stated with numerator and
denominator; residual categories named.
Time: 25 minutes.

**Exercise 5 — Dictionary version impact (critique, 35 minutes).**
Deliverable: a 350-word memo assessing whether a 33% fall in rash events after a MedDRA migration
is a safety signal or a coding artefact, with the analysis needed to decide.
Acceptance criteria: distinguishes term-level from case-level change; proposes a bridging analysis;
states what would change the conclusion.
Time: 35 minutes.

**Exercise 6 — Death reconciliation (application, 45 minutes).**
Deliverable: a reconciliation of 27 demographics death flags, 25 fatal adverse events, and 28 fatal
safety cases, with residual categories and an escalation table.
Acceptance criteria: union and agreement computed; each residual assigned an owner and deadline;
conclusion states fitness for submission.
Time: 45 minutes.

**Exercise 7 — Data platform critique (critique, 45 minutes).**
Deliverable: a red-team review of a proposal to "create a single source of truth by copying all
domains into a lake", identifying at least five defects.
Acceptance criteria: addresses grain, keys, authority, retention, and access; each defect linked to
a consequence; at least two mitigations specified.
Time: 45 minutes.

**Exercise 8 — Retention plan (design, 30 minutes).**
Deliverable: a retention and access plan for one study covering source records, tabulation
datasets, analysis datasets, and the reconciliation report.
Acceptance criteria: retention period and legal basis stated for each; access roles defined;
destruction procedure documented.
Time: 30 minutes.

## 13. Self-check answers

**Exercise 1.** A correct map includes preclinical (one row per animal per timepoint per analyte,
SEND), clinical capture (one row per subject per event or visit, CDASH/SDTM), safety (one row per
case, E2B(R3)), laboratory (one row per sample per analyte), imaging (one row per subject per
timepoint per reader, DICOM), real-world claims (one row per claim line, ICD/CPT), EHR (one row per
encounter or observation, FHIR/OMOP), and commercial (one row per product per geography per
period). The shared concepts are adverse events (safety versus clinical) and diagnoses (EHR versus
claims), which differ in standard and grain.

**Exercise 2.** A complete lineage record for haemoglobin: source is the central laboratory feed
(analyte code, result, unit, collection datetime); transformation validates the unit against the
LOINC code and converts to the protocol unit; the SDTM variable is LB.LBSTRESN with LBORRES and
LBORRESU retained; the ADaM derivation is AVAL with a baseline flag; the flag records any unit
conversion or imputation. A record that omits the unit conversion fails the reproduction test.

**Exercise 3 rubric.** Full marks require a frame that states the cut date, the population
(randomised versus dosed), and the event definition; a matching rule with key and date tolerance;
at least five residual categories with owners; and a threshold justified by the criticality of the
variable. Deduct marks for a threshold stated as "zero" without an escalation path, or for a match
rule that cannot be re-run from the description.

**Exercise 4.** In-frame safety = 412 − 3 − 14 = 395. Clinical in frame = 398. Matched = 398 − 7 =
391. Safety-only = 395 − 391 = 4. Union = 391 + 4 + 7 = 402. Agreement = 391/402 = 97.26%.
Discrepancy = 11/402 = 2.74%.

**Exercise 5.** A 33% fall in a term count after a dictionary migration is not evidence of a
change in the underlying event rate unless the same cases are counted under both versions. The
analysis required is a bridging table: re-code the events under the old and new versions and
compare at the level of the case and the clustered term. If the case count is unchanged and only
the term split changes, the finding is a coding artefact. If the case count falls, the cause is
elsewhere. The conclusion would change if re-coding showed that the migrated terms correspond to
the same clinical events in the same patients.

**Exercise 6.** Union = 24 matched + 3 demographics-only + 1 adverse-event-only + 4 safety-only =
32. Agreement = 24/32 = 75%. After removing two post-cut demographics deaths, union = 30 and
agreement = 80%. Residuals: one death in demographics without a fatal adverse event (clinical data
management, before lock); two safety-only deaths before the cut (site and pharmacovigilance,
immediate); one fatal event without a death flag (clinical data management, before lock); two
post-cut deaths (biostatistics, next cut). A 75-80% agreement for deaths is not fit for submission
without subject-level resolution.

**Exercise 7.** Defects to identify: no grain statement, so the lake cannot be joined safely; no
key strategy, so re-screened and multi-institution patients duplicate; no authority model, so
conflicting values have no resolution; no retention plan, so source records may be deleted; no
access model, so unblinded data is exposed; no lineage, so derived values cannot be reproduced.
Each should be linked to a consequence and a mitigation.

**Exercise 8 rubric.** Full marks require a retention period with its legal basis for each record
class, access roles including investigator and regulator access, and a destruction procedure with
a hold mechanism. Deduct marks for a single retention period applied to all classes, or for a plan
that retains only derived data.

## 14. Interview questions

**1. What are the main data domains in a pharma company, and how do they differ?**
Strong: names preclinical, clinical, safety, laboratory and biomarker, imaging, real-world,
commercial, operational, and submission domains, and states the grain and standard for several.
Weak: lists systems rather than domains.

**2. What is source data, and why does it matter after database lock?**
Strong: defines source as original records including metadata, explains verification and
inspection, and notes retention obligations. Weak: says "the eCRF", which is usually a
transcription.

**3. How would you reconcile two systems that report different adverse event counts?**
Strong: align the frame, deduplicate, match on identifiers with a tolerance, classify residuals,
compute agreement with denominators, and escalate by category. Weak: compares totals and picks the
larger one.

**4. What does ALCOA+ require, and where does it come from?**
Strong: attributable, legible, contemporaneous, original, accurate, plus completeness and
endurance; notes that ICH E6(R3) expresses the principles without using the acronym. Weak: recites
the acronym without a source or application.

**5. Why is a data cut not the same as database lock?**
Strong: a cut freezes data for an analysis while the database remains open; lock is a formal
declaration with change control. Weak: treats them as synonyms.

**6. What is the risk of using SDTM as an internal analytics model?**
Strong: SDTM is optimised for submission review, not cross-study analysis; keys are study-specific
and pooling needs harmonisation. Weak: assumes the regulatory standard is the best analytics model.

**7. How do dictionary versions affect a safety analysis?**
Strong: a migration can move events between terms and change term-level counts without any change
in patients; requires a bridging analysis or a frozen version. Weak: assumes counts are comparable
across versions.

**8. What would you check before joining two datasets?**
Strong: grain, key uniqueness, population, date ranges, unit and terminology compatibility.
Weak: checks column names and joins.

**9. How do you handle partial dates in a reconciliation?**
Strong: pre-specify an imputation rule, flag imputed values, and apply the same rule in both
domains; test sensitivity to the rule. Weak: imputes silently to make the join work.

**10. When is a reconciliation good enough?**
Strong: when the threshold is justified by the criticality of the variable, the residuals are
classified and owned, and the decision's sensitivity to the residuals is stated. Weak: when the
percentage looks high.

## 15. Key terms

- **ADaM** — Analysis Data Model; CDISC standard for analysis datasets.
- **ALCOA+** — data integrity principles: attributable, legible, contemporaneous, original,
  accurate, plus complete, consistent, enduring, available.
- **CDASH** — Clinical Data Acquisition Standards Harmonisation; CDISC capture standard.
- **Data cut** — the date at which a dataset is frozen for an analysis.
- **Data lineage** — documented path from source to output, including transformations.
- **Data provenance** — origin and history of a data element.
- **Database lock** — formal declaration that a database will no longer change without controlled
  unlock.
- **Define-XML** — machine-readable metadata for submission datasets.
- **eCRF** — electronic case report form.
- **EDC** — electronic data capture.
- **E2B(R3)** — ICH standard for electronic transmission of individual case safety reports.
- **eSource** — electronic capture of source data at the point of origin.
- **FHIR** — Fast Healthcare Interoperability Resources, HL7 exchange standard.
- **Grain** — what one row of a dataset represents.
- **ICH E6(R3)** — Good Clinical Practice guideline adopted in 2025, including data governance.
- **LOINC** — Logical Observation Identifiers Names and Codes.
- **MedDRA** — Medical Dictionary for Regulatory Activities, for adverse event coding.
- **Metadata** — data describing data: definition, origin, derivation, context.
- **OMOP CDM** — observational data model for multi-source analysis.
- **Partial date** — a date with missing components, e.g. month and year only.
- **Reconciliation** — the process of aligning, matching, and explaining differences between
  domains.
- **SDTM** — Study Data Tabulation Model; CDISC submission tabulation standard.
- **SEND** — Standard for Exchange of Nonclinical Data.
- **Source data** — original records and certified copies, including metadata.
- **Source data verification (SDV)** — comparison of source records with entered data.
- **Union (in reconciliation)** — the set of records present in either domain, counted once.

## 16. Sources

- ICH. *E6(R3) Good Clinical Practice*. Step 4, 6 January 2025; error corrections 24 October 2025.
  https://database.ich.org/sites/default/files/ICH_E6%28R3%29_Step4_FinalGuideline_2025_0106.pdf
  (accessed 25 October 2026).
- European Medicines Agency. *ICH E6(R3) Guideline on Good Clinical Practice, Step 5*. Effective
  23 July 2025. https://www.ema.europa.eu/en/documents/scientific-guideline/ich-e6-r3-guideline-good-clinical-practice-gcp-step-5_en.pdf (accessed 25 October 2026).
- US Food and Drug Administration. *E6(R3) Good Clinical Practice: Guidance for Industry*.
  September 2025. https://www.fda.gov/media/169090/download (accessed 25 October 2026).
- US Food and Drug Administration. *Electronic Source Data in Clinical Investigations: Guidance
  for Industry*. September 2013.
  https://www.fda.gov/regulatory-information/search-fda-guidance-documents/electronic-source-data-clinical-investigations (accessed 25 October 2026).
- US Food and Drug Administration. *21 CFR Part 11, Electronic Records; Electronic Signatures*.
  https://www.ecfr.gov/current/title-21/chapter-I/subchapter-A/part-11 (accessed 25 October 2026).
- CDISC. *SDTM, ADaM, CDASH, Define-XML and Controlled Terminology*. https://www.cdisc.org/
  (accessed 25 October 2026).
- ICH. *E2B(R3) Electronic Transmission of Individual Case Safety Reports*.
  https://www.ich.org/page/efficacy-guidelines (accessed 25 October 2026).
- MedDRA. *MedDRA and the MSSO*. https://www.meddra.org/ (accessed 25 October 2026).
- HL7. *FHIR Specification*. https://hl7.org/fhir/ (accessed 25 October 2026).
- OHDSI. *OMOP Common Data Model*. https://ohdsi.github.io/CommonDataModel/ (accessed 25 October 2026).
- LOINC. *LOINC*. https://loinc.org/ (accessed 25 October 2026).
- US Food and Drug Administration. *Study Data Technical Conformance Guide*.
  https://www.fda.gov/industry/study-data-standards-resources/study-data-technical-conformance-guide
  (accessed 25 October 2026).

> **Note:** The worked example and its counts are constructed for teaching. They are consistent
> internally and reproducible with the SQL in Section 7, but they describe no real study.

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic: [`01_development-funnel.md`](01_development-funnel.md)
- Next topic: [`03_biomarker-strategy.md`](03_biomarker-strategy.md)
- Related: [`05_cdisc-and-trial-standards.md`](05_cdisc-and-trial-standards.md)
- Related: [`06_safety-and-pharmacovigilance.md`](06_safety-and-pharmacovigilance.md)
- Survey file: [`../../13_biopharma_clinical_development.md`](../../13_biopharma_clinical_development.md)
- Labs: [`../../labs/`](../../labs/)
