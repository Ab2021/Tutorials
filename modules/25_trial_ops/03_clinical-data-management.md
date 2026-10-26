# Clinical Data Management: CRF Design, Edit Checks, Medical Coding, and Database Lock

> Module 25 — Trial Operations, HTA, and Global Regulation
> Topic 03 of 07
> Estimated study time: 4 hours
> Prerequisites: topic 02
> Version 2026.3

## 1. Why this topic exists

A clinical trial produces data in many places: a hospital record, a laboratory system, a
questionnaire, a device, a pharmacy log, and a coordinator's keyboard. Clinical data management is
the discipline that turns those sources into a database that can be analysed, submitted, and
defended. When it works, nobody notices. When it fails, the trial can be complete, on time, and
still unusable.

The failure is expensive in a specific way. A statistical analysis can be rerun; a database that
cannot be locked delays the submission, and a submission with unresolved data integrity questions
can be refused for filing. The cost is not only the delay. It is the loss of the one asset the trial
was built to produce: evidence that a regulator and a payer will accept.

Data management is also where the data scientist's habits help and where they can mislead. The
habits that help are reproducibility, explicit schemas, validation, and version control. The habit
that misleads is treating data cleaning as an open-ended modelling problem. A clinical database has
a deadline, a fixed set of questions, and a regulatory definition of done. The task is to reach that
definition with evidence, not to keep improving the data.

### 1.1 The pattern that repeats

1. The protocol is finalised and the electronic case report form (eCRF) is designed in a hurry.
2. Three variables needed for the primary endpoint are collected in free text.
3. The edit checks are built from the database specification, not from the analysis plan.
4. At month 18, the statistician discovers that a key subgroup variable was never collected.
5. A late protocol amendment adds the variable, but patients already enrolled cannot be classified.
6. The primary analysis is restricted to a subset, and the label narrows.
7. The database locks on time, and the evidence is weaker than the trial deserved.

Every step after step 1 is a data management decision that could have been made before the first
patient. The cost is not a delay; it is a permanently smaller evidence base.

### 1.2 Who is blocked

| Stakeholder | What they lose | Why they cannot fix it alone |
|---|---|---|
| Statistician | Pre-specified analysis population | Receives variables someone else chose |
| Regulatory affairs | A clean submission | Depends on a database it does not control |
| Sites | Time on queries that could have been avoided | Did not design the form |
| Medical monitor | Safety signal clarity | Coding and reconciliation happen elsewhere |
| Payer or HTA analyst | Subgroup and patient-reported data | The trial did not collect them |
| Patient | Confidence that their data are used correctly | Has no visibility |

### 1.3 The industry question

> Is the database complete, correct, traceable, and standardised enough to support the analysis, the
> submission, and the label?

The question has four clauses for a reason. A database can be complete but not correct, correct but
not traceable, or traceable but not standardised. Each failure has a different remedy and a
different cost.

## 2. Learning outcomes

By the end of this topic you can:

1. **Translate** a protocol's endpoints into a data requirement specification that names each
   variable, its source, its format, and its derivation.
2. **Design** an eCRF that minimises transcription, ambiguity, and query volume, using Clinical Data
   Acquisition Standards Harmonization (CDASH) principles.
3. **Write** edit checks that test clinical logic, not only data types, and prioritise them by the
   cost of an undetected error.
4. **Plan** medical coding with the correct dictionary versions and a defensible approach to partial
   matches and combination terms.
5. **Model** a data cleaning and coding cycle and compute the database lock date from stage
   durations and capacity.
6. **Diagnose** a lock plan that is limited by coding or reconciliation rather than by queries.
7. **Prepare** a lock package that includes the clinical database, the analysis datasets, the define
   file, and the traceability evidence.
8. **Evaluate** whether a database is submission-ready against the applicable standard and the
   regulator's technical requirements.

## 3. The core mental model

### 3.1 The data lifecycle

```text
 Protocol and analysis plan
        |
        v
 Data requirement specification
        |
        v
 eCRF / CDASH design -----> database build -----> user acceptance testing
        |                                              |
        v                                              v
 Site entry -----> Edit checks -----> Queries -----> Clean data
        |                                              |
        +---- external data (labs, ECG, PK, ePRO) -----+
        |                                              |
        v                                              v
 Medical coding (MedDRA, WHODrug) ------------> Clinical database
                                                       |
                                                       v
                                                 Database lock
                                                       |
                                      +----------------+----------------+
                                      v                                 v
                              SDTM datasets                     Analysis (ADaM)
                                      |                                 |
                                      +----------> define.xml <---------+
                                                       |
                                                       v
                                                Submission package
```

The diagram has two properties that determine most failures. First, the analysis plan sits upstream
of the data requirement specification, not downstream: if the estimand and the analysis population
are not settled before the eCRF is designed, the variables needed to define them will be missing.
Second, the standardised datasets are derived from the clinical database, so an error at the source
propagates to every downstream artefact.

### 3.2 Three representations of the same trial

| Representation | What it contains | Who owns it | What it is for |
|---|---|---|---|
| Clinical database | Data as collected, with audit trail | Data management | Source of truth for the trial |
| SDTM | Data organised by submission domain | Standards and programming | Regulatory review |
| ADaM | Analysis-ready datasets with derivations | Biostatistics | Reproducing the analysis |

The three are related by traceability, not by identity. A value in ADaM must be traceable to SDTM,
and an SDTM value must be traceable to the clinical database or to a documented derivation. When
that chain breaks, the submission is vulnerable even if every individual number is correct.

### 3.3 Data quality is defined by use

A database is not "clean" in the abstract. It is clean enough for a specific use:

- **For the primary analysis:** the endpoint, the analysis population flags, and the stratification
  variables must be complete and correct.
- **For safety:** every serious adverse event must be reconciled and coded, and every death must be
  accounted for.
- **For the label:** the subgroup variables and the patient-reported outcomes must be collected with
  enough completeness to support the claims.
- **For inspection:** every data point must be traceable to a source, and every change must have a
  reason.

A cleaning plan that treats all fields equally spends effort on fields that do not affect any of
these uses and under-invests in the ones that do.

### 3.4 The query is a feedback signal, not a defect count

Queries measure the mismatch between what the site entered and what the sponsor expected. A high
query rate can mean a badly designed form, an ambiguous protocol, or a site that needs training. A
low query rate can mean excellent data or edit checks that are too weak to detect problems.

The useful decomposition is by cause:

```text
Query rate = form design effect + protocol ambiguity effect + site performance effect
             + edit-check sensitivity effect
```

A programme that reports only the total cannot tell which lever to pull. A programme that reports
queries per field by form can.

### 3.5 Lock is a gate with evidence

Database lock is not a date; it is a declaration supported by a checklist. The checklist typically
covers:

- all patients accounted for in the disposition;
- all visits entered and all expected data present or explained;
- all queries resolved or formally waived with a documented rationale;
- all adverse events coded and all serious adverse events reconciled with the safety database;
- all external data loaded and reconciled;
- all protocol deviations recorded and classified;
- the analysis population flags derived and verified;
- the database audit trail reviewed;
- the blind review completed where applicable.

A lock that happens without the checklist is a hope with a timestamp.

### 3.6 The critical path is not always queries

Teams assume that queries determine the lock date. Often they do not. Coding, safety
reconciliation, external data loading, or analysis dataset derivation can be the binding constraint.

```text
Lock date = max(query closure, coding completion, reconciliation completion,
                external data completion, analysis-ready check) + quality control
```

The implication is that improving query turnaround has no effect if coding is the constraint. A lock
plan should identify the binding stage before it invests in a fix.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Case report form (CRF) | Form on which trial data are recorded | Used to mean the database | Document |
| Electronic CRF (eCRF) | CRF implemented in an electronic data capture system | Assumed to eliminate queries | System artefact |
| CDASH | Clinical Data Acquisition Standards Harmonization | Treated as a mandatory standard | Convention |
| SDTM | Study Data Tabulation Model | Confused with the clinical database | Standard |
| ADaM | Analysis Data Model | Confused with SDTM | Standard |
| Define-XML | Machine-readable metadata describing the datasets | Treated as optional documentation | File |
| Controlled terminology | Codelists maintained by CDISC and others | Assumed to be complete | Codelist |
| Edit check | Rule that validates data at entry or on demand | Confused with a query | Rule |
| Query | Request to resolve a discrepancy | Counted as an error | Record |
| Data clarification form (DCF) | Form used to answer a query | Used for all query types | Form |
| Medical coding | Assignment of standard terms to verbatim text | Confused with data entry | Activity |
| MedDRA | Medical Dictionary for Regulatory Activities | Used without a version | Dictionary |
| WHODrug | Dictionary of medicinal products | Confused with MedDRA | Dictionary |
| Dictionary version | Specific release of a coding dictionary | Not recorded in the dataset | Version |
| Partial match | Coded term that captures part of the verbatim text | Accepted without review | Coding decision |
| Combination term | Single coded term for multiple products or events | Split incorrectly | Coding decision |
| Serious adverse event (SAE) reconciliation | Comparison of the safety database with the clinical database | Done at submission only | Activity |
| External data | Data originating outside the EDC | Assumed to be clean | Data type |
| Database lock | Declared point after which the database cannot change | Declared with open queries | Milestone |
| Soft lock | Freeze before final lock for analysis | Confused with hard lock | State |
| Data cut | Snapshot of the database at a point in time | Confused with lock | Snapshot |
| Audit trail | Record of who changed what, when, and why | Assumed to be immutable | Log |
| User acceptance testing (UAT) | Formal test of the database before go-live | Skipped under time pressure | Activity |
| Traceability | Ability to follow a value from source to analysis | Confused with documentation | Property |
| aCRF | Annotated CRF mapping fields to SDTM | Produced late | Document |
| Missing data | Absence of an expected value | Not distinguished from "not done" | Category |
| Analysis population flag | Variable identifying a patient's inclusion in an analysis | Derived informally | Variable |
| Blinded review | Review of data before unblinding | Confused with data cleaning | Activity |
| Data management plan | Document defining processes, roles, and standards | Treated as a template | Document |

### 4.1 Collected, derived, and imputed

A collected value is recorded at the site. A derived value is computed from collected values by a
documented rule. An imputed value is filled in by a statistical method. The three must be
distinguishable in the dataset, because a regulator will ask which is which, and an analysis that
treats an imputed value as collected is misleading.

### 4.2 Query, discrepancy, and error

A discrepancy is a difference between two values or between a value and an expectation. A query is
the communication that asks about it. An error is a confirmed incorrect value. Most queries are not
errors: the value is correct, the expectation was wrong, or the form allowed an ambiguous entry. A
team that equates queries with errors will over-clean and under-fix the form.

### 4.3 Lock, cut, and freeze

A data cut is a snapshot for an analysis, often taken before the trial ends. A soft lock freezes the
database for a specific analysis while allowing later changes. A hard lock closes it permanently. A
trial can have several cuts and one final lock. Using "lock" for all three produces plans that
cannot be reconciled.

## 5. The mechanics, step by step

This procedure builds a clinical database for a multicentre trial using electronic data capture.
The sequence is written for a drug trial; device trials add device accountability data, and
decentralised trials add direct-to-participant data streams.

### Step 1. Translate endpoints into data requirements

**Input:** the protocol, the estimand definition, and the statistical analysis plan.

**Output:** a data requirement specification listing every variable, its definition, its source, its
format, its permitted values, and whether it is collected, derived, or external.

**Rationale:** The specification is the contract between the protocol and the database. If it is
written after the eCRF, it becomes a description of what was built rather than a statement of what is
needed. The estimand is the critical input: it determines which intercurrent events must be
recorded, and therefore which variables must exist.

**Branch point:** an event-driven or adaptive trial needs variables that support the interim
analysis and the adaptation, not only the final analysis. Those variables are easy to forget because
they are not in the primary endpoint definition.

### Step 2. Design the eCRF

**Input:** the data requirement specification and the visit schedule.

**Output:** an eCRF with one form per assessment, clear labels, controlled codelists, and no free-text
field that could be a coded field.

**Rationale:** Every free-text field is a future query and a coding cost. Every duplicated field is a
future reconciliation. Every ambiguous label is a future protocol deviation. CDASH provides
conventions for common fields, and using them reduces mapping work later, but the main benefit is
design discipline: collect once, at the point of care, in the form the analysis needs.

**Branch point:** paper versus electronic capture changes the design constraints. Paper needs
fewer, simpler fields and a data entry convention; electronic capture can use dynamic forms but
must still avoid fields that are required only in rare cases.

### Step 3. Build the database and edit checks

**Input:** the eCRF specification and the edit-check specification.

**Output:** a validated database with automated checks, manual review flags, and a user acceptance
test record.

**Rationale:** Edit checks should encode clinical logic, not only data types. "Date of visit is
after date of consent" is a clinical rule; "field is numeric" is a type rule. The former prevents a
protocol deviation; the latter prevents a formatting error. A database with only type checks will
pass validation and still produce an unusable dataset.

**Branch point:** checks can run at entry, at submission of a form, or in batch. Entry-time checks
prevent bad data; batch checks detect patterns. A design that relies only on entry-time checks
cannot detect a value that is individually plausible but inconsistent across visits.

### Step 4. Map external data flows

**Input:** the laboratory, electrocardiogram, pharmacokinetic, electronic patient-reported outcome
(ePRO), and randomisation sources.

**Output:** a data transfer specification for each source, with format, frequency, identifier
mapping, and reconciliation rules.

**Rationale:** External data are the most common source of late reconciliation problems. The
laboratory's patient identifier may not match the trial identifier, the reference ranges may change
during the trial, and the transfer may arrive monthly while the trial needs daily safety data. Each
of these is a specification decision, not an operational surprise.

### Step 5. Train sites and go live

**Input:** the eCRF, the edit-check list, the data entry guide, and the training materials.

**Output:** trained users, a go-live date, and a support channel.

**Rationale:** Most entry errors are design or training errors. A coordinator who does not know
whether a value should be recorded as "not done" or "not applicable" will choose inconsistently, and
the inconsistency will appear as a query pattern months later.

### Step 6. Enter, validate, and query

**Input:** site data.

**Output:** entered data, open queries, and a query log with state timestamps.

**Rationale:** Query resolution is a workflow, and the workflow should be measured by state, not only
by total age. The site's response time and the data manager's review time have different owners and
different fixes.

### Step 7. Code adverse events and medications

**Input:** verbatim terms, the coding dictionary versions, and the coding guidelines.

**Output:** coded terms with dictionary version, hierarchy, and a record of any partial match or
combination decision.

**Rationale:** Coding is a judgement, not a lookup. "Head cold" may code to viral upper respiratory
infection; "COVID pneumonia" may require a combination of terms. The decisions must be consistent
across the trial and documented, or two coders will produce two datasets from the same text.

**Branch point:** the dictionary version must be fixed before the first patient and recorded in the
dataset. A version upgrade mid-trial creates a mixed coding population, which must be handled in the
analysis rather than ignored.

### Step 8. Reconcile safety data

**Input:** the clinical database and the pharmacovigilance safety database.

**Output:** a reconciliation report showing every serious adverse event matched, every discrepancy
resolved, and every death accounted for.

**Rationale:** The two databases are maintained by different teams for different purposes. They
diverge routinely: an event reported to the safety team may not be entered as an adverse event in
the clinical database, and a severity grade may differ. Reconciliation is the evidence that the
submission's safety dataset and the clinical dataset describe the same patients.

### Step 9. Review data centrally and statistically

**Input:** the accumulated database.

**Output:** a data review report with site-level and patient-level findings, and a list of issues for
querying or for site follow-up.

**Rationale:** Central review detects patterns that site-level review cannot: a site whose visit
dates cluster on Mondays, a laboratory value that never varies, a patient whose adverse events all
start on the same day. These are signals of process problems or fabrication, and they are cheap to
detect before lock.

### Step 10. Prepare the lock package

**Input:** the database, the query log, the coding report, the reconciliation report, and the
analysis plan.

**Output:** a lock checklist with each item signed off, a list of open items with a disposition, and
a proposed lock date.

**Rationale:** The checklist converts "we think it is clean" into a set of assertions that can be
audited. Each open item should be classified as resolved, waived with rationale, or not applicable.
An item that is simply absent from the list is the one that will be found later.

### Step 11. Resolve, waive, or defer open queries

**Input:** the open query list at the planned lock date.

**Output:** a decision for each query, recorded before lock.

**Rationale:** Lock with open queries is acceptable when the queries affect neither the primary
endpoint nor safety and the omission is documented. It is not acceptable when the queries affect the
analysis population, the endpoint, or a safety signal. The decision rule belongs in the data
management plan, not in a meeting at the deadline.

### Step 12. Lock and unblind

**Input:** the signed lock checklist.

**Output:** a locked database, an unblinding record, and a snapshot for analysis.

**Rationale:** The lock is a change-control event. After it, any correction requires a documented
unlock, an impact assessment, and a new version. A team that treats lock as a formality will unlock
repeatedly, and each unlock weakens the audit trail.

### Step 13. Convert to SDTM and produce define.xml

**Input:** the locked clinical database and the mapping specification.

**Output:** SDTM datasets, an annotated CRF, define.xml, and a reviewer's guide.

**Rationale:** SDTM is a submission format, not an analysis format. The conversion is a mapping
exercise with its own quality control: controlled terminology must be applied, dates must follow
ISO 8601, and every derived variable must be traceable. The define file is the machine-readable
description that lets a reviewer reconstruct the mapping.

### Step 14. Derive ADaM and assemble the submission

**Input:** SDTM datasets and the statistical analysis plan.

**Output:** ADaM datasets, analysis programs, and a traceability package.

**Rationale:** ADaM datasets must reproduce the analysis exactly. A dataset that produces the right
table by a different route is not reproducible. The traceability package is what allows a reviewer
to check that the primary endpoint in the table is the one defined in the protocol.

### 5.15 Decision table: how to capture a data item

| Option | When to use | Effect on queries | Risk |
|---|---|---|---|
| Collect directly in the eCRF | The site knows the value at the visit | Low if the form is clear | Entry burden if the field is rarely relevant |
| Derive from collected fields | The value is computable by a documented rule | Low | Derivation error propagates silently |
| Import from an external source | The value originates in a laboratory or device | Moderate; reconciliation needed | Identifier mismatch |
| Collect in free text and code later | The value is unstructured by nature | High | Coding inconsistency |
| Do not collect | The value is not needed for any use | None | Irreversible if the analysis later needs it |

The last row is the most consequential. A variable not collected cannot be recovered, and the
analysis plan that needs it will be amended rather than the database.

### 5.16 Decision table: resolving a query

| Situation | Resolution | Documentation | Follow-up |
|---|---|---|---|
| Site confirms the entered value is correct | Close with confirmation | Query record | None |
| Site corrects the value | Update with reason | Audit trail | Check downstream derivations |
| Value is correct but violates the protocol | Record a deviation | Deviation log | Assess impact on analysis |
| Value cannot be determined | Mark as missing with reason | Missing-data code | Sensitivity analysis |
| Query is based on a wrong expectation | Correct the edit check | Change record | Re-run the check on prior data |
| Query affects safety | Escalate immediately | Safety reconciliation | Medical review |

### 5.17 Decision table: lock readiness

| Condition | Decision | Required action |
|---|---|---|
| All checklist items complete; no open critical queries | Lock | None |
| Open queries affect only non-critical fields, with documented disposition | Conditional lock | Record waiver; include in analysis limitations |
| Open queries affect primary endpoint or safety | Do not lock | Resolve, or delay the analysis |
| Coding incomplete but analysis can use uncoded terms | Conditional lock | Document coding status; complete before submission |
| Safety reconciliation incomplete | Do not lock | Reconcile first |
| Audit trail review not done | Do not lock | Complete the review |

### 5.18 Edge cases

**Database lock with open queries.** The decision depends on which queries. A query about a
concomitant medication that is not in the analysis is waivable. A query about the date of
progression is not. The lock meeting should have the query list sorted by analysis impact, not by
age.

**Protocol amendment mid-study.** A new variable added mid-trial creates a structural missingness:
patients enrolled before the amendment have no value. The database must distinguish "not collected
because the variable did not exist" from "not done" and from "unknown". The analysis plan must decide
how to handle the first group, and the decision should be made before unblinding.

**Cross-border data transfer.** The eCRF vendor's hosting region determines where the data physically
sit. A transfer to a sponsor's analytics environment in another jurisdiction requires a legal
mechanism and may require de-identification. This is a design decision, not a downstream IT task.

**Site under-enrolment.** A site that enrolled two patients still has to be closed, its data cleaned,
and its queries resolved. The marginal cost per patient at a low-enrolling site is high, and the
close-out queue should account for site count as well as patient count.

**Payer and HTA evidence needs.** A health technology assessment submission often requires subgroup
data, health-related quality of life, and resource-use variables that are not in the primary
endpoint. If those variables were not collected, the trial cannot support the reimbursement
question, and the gap cannot be fixed at the analysis stage. This is the clearest example of a data
management decision determining a commercial outcome.

### 5.19 Metrics for a data management function

| Metric | Definition | Use |
|---|---|---|
| Query rate per patient | Queries divided by enrolled patients | Form and site performance |
| Query rate per field | Queries divided by collected fields | Form design quality |
| Query cycle time P50 and P90 | Days from raise to close | Workflow performance |
| Reopen rate | Closed queries reopened | Answer quality |
| Coding backlog | Uncoded terms at a point in time | Lock risk |
| Coding accuracy | Sample audit against a reference | Coder performance |
| SAE reconciliation discrepancy rate | Unmatched or inconsistent SAEs | Safety data quality |
| CRF completion rate | Forms complete at the expected visit | Site performance |
| Lock checklist completion | Items signed off | Lock readiness |
