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

### 5.19 Design the data review plan

**Input:** the critical variables, the risk assessment, and the analysis plan.

**Output:** a review plan that specifies what is reviewed, by whom, how often, and what happens with
a finding.

**Rationale:** Query resolution fixes individual values; data review looks for patterns that no
single query would reveal. A patient profile review can show that a participant's adverse events all
begin on the day of a study visit, which suggests a recording convention rather than a clinical
pattern. A site-level review can show that a site's visit dates are all Mondays, which suggests
back-dating.

| Review type | Unit | Detects | Frequency |
|---|---|---|---|
| Patient profile | One participant across visits | Internal inconsistencies, missing assessments | Before each data cut |
| Site listing | One site across participants | Process problems, training gaps | Monthly |
| Cross-site statistical | All sites | Outliers, digit preference, implausible distributions | Monthly after 10 sites |
| External data reconciliation | Laboratory, ECG, PK | Identifier and unit mismatches | Each transfer |
| Safety reconciliation | SAE database versus clinical database | Unreported or inconsistent events | Monthly and before lock |

**Branch point:** a blinded review can examine distributions and completeness but not treatment
effects. An unblinded review, where permitted, can examine safety by arm but risks compromising the
trial. The plan should state which reviews are blinded and who has access.

### 5.20 Control the quality of SDTM and ADaM

**Input:** the locked clinical database, the mapping specification, and the analysis plan.

**Output:** SDTM and ADaM datasets, define.xml, a reviewer's guide, and a quality-control log.

**Rationale:** Conversion is programming, and programming has bugs. The controls that catch them are
independent derivation, comparison against the source, conformance checks against the standard, and
validation of the define file against the datasets. A dataset that passes conformance can still be
wrong; a dataset that fails conformance will be questioned at filing.

Common findings include dates not in ISO 8601 format, units not converted to the standard, supplemental
qualifiers used where a standard variable exists, and analysis flags that do not reproduce the
analysis population. Each is cheap to fix before submission and expensive to explain after.

### 5.21 Archive and preserve traceability

**Input:** the locked database, the datasets, the programs, the define file, and the trial master
file.

**Output:** an archive that can reconstruct the analysis years later.

**Rationale:** The retention obligation outlives the trial team. An archive that contains the final
datasets but not the programs, or the programs but not the input data, cannot reproduce the result.
The test is simple: could a competent analyst who has never seen the trial regenerate the primary
table from the archive alone?

### 5.22 Metrics for a data management function

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

## 6. Worked example

### 6.1 The trial

A Phase III trial enrols 600 patients at 45 sites over 24 months, with a further six months of
follow-up. Last patient last visit (LPLV) falls at month 30. The sponsor has committed to a database
lock 45 days after LPLV, and the fixed operating cost is $600,000 per month, or $20,000 per day.

The data management plan has the following stage assumptions:

| Stage | Duration | Basis |
|---|---:|---|
| Final data entry after LPLV | 7 days | Site close-out schedule |
| Query tail (last query closed) | 45 days | Historical P90 |
| Coding of adverse events | 50 days | 6,000 terms, 50% auto-coded, 3 coders at 20 terms per day |
| SAE reconciliation | 10 days | 45 serious adverse events, site response time |
| Analysis flags and quality control | 10 days | Prior trials |

### 6.2 The critical path

Entry and query closure run in sequence; coding runs in parallel with them and can start as visits
complete.

```text
Query path   = 7 + 45 = 52 days
Coding path  = 6,000 x (1 - 0.50) / (3 x 20) = 3,000 / 60 = 50 days
Pre-lock     = max(52, 50) = 52 days
Lock         = 52 + 10 (reconciliation) + 10 (flags and QC) = 72 days
```

The plan says 45 days. The arithmetic says 72. The 27-day gap is $540,000 at $20,000 per day.

The two paths are almost equal, which is the dangerous case: a small change in either one changes
which is binding, and the organisation's attention is usually on only one of them.

### 6.3 Diagnostic 1: the reopen rate extends the query path

The query log shows that 18% of closed queries are reopened because the answer was incomplete. Each
reopened query adds about 14 days. At LPLV, 18% of the 150 open queries, or 27 queries, are in this
category.

```text
Query tail with reopening = 45 + 14 = 59 days
Query path = 7 + 59 = 66 days
Lock = 66 + 20 = 86 days
Slip against plan = 41 days
Cost = 41 x $20,000 = $820,000
```

The reopen rate is a sponsor-side metric as much as a site-side one. A query that asks for a value
without saying what format is expected will be answered ambiguously and reopened. Improving the
question is cheaper than resolving the answer twice.

### 6.4 Diagnostic 2: coding, not queries, becomes the constraint

One coder is reassigned to another study for three weeks, reducing capacity from 60 to 40 terms per
day during the final month.

```text
Coding path = 3,000 / 40 = 75 days
Pre-lock = max(52, 75) = 75 days
Lock = 75 + 20 = 95 days
Slip against plan = 50 days
Cost = 50 x $20,000 = $1,000,000
```

The query dashboard shows nothing wrong. The coding backlog is not on the dashboard at all. The lock
slips by 23 days more than the query-only diagnosis predicted, and the cause is a staffing decision
made in a different department.

This is the most common structural failure in lock planning: the plan monitors the stage it
understands and ignores the stage that happens to be binding.

### 6.5 Vary one assumption: assisted coding

Suppose the sponsor deploys assisted coding that raises the auto-coding rate from 50% to 75%, with
validation of the tool and a sample audit. Manual terms fall from 3,000 to 1,500.

```text
Coding path at 40 terms per day = 1,500 / 40 = 37.5 days
Pre-lock = max(52, 37.5) = 52 days
Lock = 52 + 20 = 72 days
Saving against 95 days = 23 days
Value = 23 x $20,000 = $460,000
Tool and validation cost = $150,000
Net benefit = $310,000
```

The conclusion changes if the query path also improves. With reopening reduced from 18% to 5%, the
query tail returns to 45 days and the lock is 72 days regardless of coding. In that case the assisted
coding tool adds no further lock benefit, though it may still reduce cost per term. The two
interventions overlap, and a business case that counts both savings in full double-counts them.

### 6.6 The cost of a query and the case for form design

The lock date is not the only cost of a badly designed form. Each query consumes site and sponsor
time. A reasonable estimate is 20 minutes of data management time, 25 minutes of site time, and a 15%
chance of a reopen that adds another 20 minutes:

```text
Hours per query = 0.33 + 0.42 + (0.15 x 0.33) = 0.80 hours
Annual query volume at steady state = 6,000 over 24 months = 3,000 per year
Annual hours = 3,000 x 0.80 = 2,400 hours
Annual cost at $90 per hour loaded = $216,000
```

A form redesign that removes one ambiguous field and cuts queries by 20% saves 480 hours and
$43,200 per year, and shortens the query tail. If the redesign costs $60,000 and the trial runs two
years, the direct payback is marginal, but the lock benefit is not: fewer queries means a thinner
tail, and the tail is what determines the date.

The arithmetic changes the argument. The case for better form design is not that queries are
expensive in aggregate; it is that the marginal query at the tail is expensive because it holds the
lock.

### 6.7 Sensitivity: assisted coding without a query fix

The assisted coding case in Section 6.5 returns the lock to 72 days because the query path is
binding at 52 days. If the query tail remains at 59 days because reopening is not addressed, the
lock stays at 86 days even with assisted coding.

```text
Query path = 7 + 59 = 66 days
Pre-lock = max(66, 37.5) = 66 days
Lock = 66 + 20 = 86 days
Saving from assisted coding = 95 - 86 = 9 days = $180,000
Net of $150,000 cost = $30,000
```

The same tool produces a $310,000 net benefit or a $30,000 net benefit depending on whether the
query path is fixed at the same time. Interventions in a serial system interact, and the
interaction must be modelled rather than assumed away.

### 6.8 A cheaper alternative: the two-stage lock

Not every use of the database needs the same completeness. A soft lock at 52 days, when the query
path closes and the primary endpoint and serious adverse events are reconciled, supports the primary
analysis. Coding of non-serious adverse events and the remaining query closure can continue to a hard
lock at 86 days.

```text
Soft lock  = 52 days (primary endpoint and SAE reconciliation complete)
Hard lock  = 86 days (full coding and all queries closed)
Difference = 34 days
```

The value of the soft lock depends on which decision needs the data. If a partner milestone or an
interim go/no-go decision depends on the primary analysis, 34 days earlier has value. If the only
downstream event is the submission, the soft lock saves nothing, because the submission waits for
the hard lock.

The two-stage option is therefore not a way to lock faster. It is a way to decouple an early decision
from the complete database, and it is worth doing only when such a decision exists.

### 6.9 What the example teaches

1. The lock date is a maximum over stages, and the binding stage can change with a single staffing
   decision.
2. Query metrics are necessary but not sufficient. Coding, reconciliation, and external data must
   appear on the same dashboard.
3. Reopened queries are a hidden multiplier on the tail, and they are often caused by the sponsor's
   question rather than the site's answer.
4. Interventions overlap. A saving should be attributed once.

## 7. Code, query, or calculation

The model computes the lock date from stage durations and identifies the binding stage. It uses only
the standard library.

**Assumptions:**

- entry and query closure are sequential;
- coding runs in parallel and is limited by manual terms divided by coder capacity;
- reconciliation and quality control follow the later of the query and coding paths;
- 6,000 adverse event terms, 50% auto-coded in the base case, 75% with assisted coding;
- three coders at 20 terms per day, reduced to two coders in the leave scenario;
- cost of delay $20,000 per day in 2026 US dollars.

```python
def lock_plan(auto_rate=0.50, coders=3, terms_per_coder_day=20,
              entry=7, query_tail=45, recon=10, qc=10, terms=6000):
    manual = terms * (1 - auto_rate)
    coding_days = manual / (coders * terms_per_coder_day)
    query_path = entry + query_tail
    pre_lock = max(query_path, coding_days)
    lock = pre_lock + recon + qc
    binding = "query" if query_path >= coding_days else "coding"
    return lock, coding_days, binding

cases = {
    "base": dict(auto_rate=0.50, coders=3, query_tail=45),
    "reopen 18%": dict(auto_rate=0.50, coders=3, query_tail=59),
    "coder leave": dict(auto_rate=0.50, coders=2, query_tail=45),
    "assisted coding": dict(auto_rate=0.75, coders=2, query_tail=45),
}

plan = 45
for name, kwargs in cases.items():
    lock, coding, binding = lock_plan(**kwargs)
    print(f"{name:16s} coding {coding:5.1f}d  lock {lock:5.1f}d  "
          f"binding {binding:6s}  slip {lock - plan:4.1f}d  cost {(lock - plan) * 20000:,.0f}")

print(f"assisted coding value: {(95 - 72) * 20000:,.0f}")
print(f"assisted coding net:   {(95 - 72) * 20000 - 150000:,.0f}")
```

**Expected output:**

```text
base             coding  50.0d  lock  72.0d  binding query   slip 27.0d  cost 540,000
reopen 18%       coding  50.0d  lock  86.0d  binding query   slip 41.0d  cost 820,000
coder leave      coding  75.0d  lock  95.0d  binding coding  slip 50.0d  cost 1,000,000
assisted coding  coding  37.5d  lock  72.0d  binding query   slip 27.0d  cost 540,000
assisted coding value: 460,000
assisted coding net:   310,000
```

**What the output shows:**

1. The base case is 27 days late against the plan, and the query path is binding by two days. A
   two-day margin is not a margin; it is a coincidence.
2. Reopening adds 14 days without changing coding, which is why reopen rate belongs on the lock
   dashboard.
3. A single coder absence adds 23 days and switches the binding stage. The lock plan that tracks
   only queries will not see it.
4. Assisted coding recovers the coder-leave scenario exactly, returning the lock to 72 days. It does
   not improve the base case, because the base case is query-bound.

**What to vary:** the query tail. If the organisation can reduce the 90th percentile from 45 to 30
days, the query path falls to 37 days, coding becomes binding at 50, and the lock is 70 days. At that
point further query improvement has no effect on the lock, and the investment should move to coding.
The model makes that switch visible.

## 8. Data, standards, and regulatory context

Clinical data management is constrained by submission standards that are more specific than most
other parts of trial operations. A dataset that is clinically correct but structurally non-compliant
can be technically rejected before a reviewer reads it.

### 8.1 Binding requirements and technical specifications

| Instrument | What it constrains | Status |
|---|---|---|
| 21 CFR Part 11 | Electronic records, signatures, audit trails | Regulation |
| 21 CFR 312.62 | Investigator records and retention | Regulation |
| ICH E6(R3) | Data governance across the life cycle | Guideline, effective in the EU from 23 July 2025 |
| ICH E9(R1) | Estimand definition, and therefore required variables | Guideline |
| FDA standardized study data guidance (June 2021) | Format and standards for electronic submissions | Guidance |
| FDA Study Data Technical Conformance Guide v6.3 (October 2026) | Technical detail for study data filings | Technical specification |
| FDA Data Standards Catalog | Which standards and versions are supported or required, and from when | Catalog |
| CDISC SDTMIG 3.4 and SDTM 2.0 | Tabulation structure | Standard, published 29 November 2021 |
| CDISC ADaMIG 1.3 and ADaM 2.1 | Analysis dataset structure | Standard, published 29 November 2021 |
| Define-XML 2.1 | Machine-readable metadata | Standard; current package 2.1.11, 6 April 2026 |
| MedDRA and WHODrug | Coding dictionaries | Standards with versioned releases |
| Regulation (EU) 536/2014 | Trial data, transparency, inspection access | Regulation |

The distinction between a guidance and a technical specification matters. The June 2021 guidance
states the expectation that study data be standardised; the conformance guide states how, down to
file naming and validation rules. The Data Standards Catalog states which versions are accepted and
from which date. A sponsor that reads only the guidance will miss the version requirements.

### 8.2 What the FDA requires at filing

The FDA's electronic submission rules apply to study data submitted in marketing applications and
certain other filings. The conformance guide describes:

- the electronic file directory structure for tabulations and analysis datasets;
- the use of SAS Transport version 5 (XPT v5) for datasets;
- define.xml and the annotated CRF;
- the Study Data Standardization Plan and the clinical study data reviewer's guide;
- terminology requirements, including MedDRA, WHODrug, UNII, MED-RT, SNOMED CT, and LOINC;
- validation rules and the technical rejection criteria.

Technical rejection is not a scientific judgement. A file that fails a validation rule can be
returned without review, which means the trial's evidence never reaches an assessor. The practical
consequence is that conformance testing belongs in the submission timeline as a gate, not as a final
formatting step.

### 8.3 Standards are versioned, and versions expire

SDTMIG 3.4 and ADaMIG 1.3 were published on 29 November 2021 and are used with SDTM 2.0 and ADaM 2.1
respectively. Define-XML 2.1 is the current metadata standard, with package 2.1.11 released on
6 April 2026. The Data Standards Catalog lists support-begin and requirement-begin dates; a version
can be supported for use before it is required, and older versions are retired on published dates.

The operational implication is that the standard version must be fixed at the start of the
conversion, recorded in the define file, and checked against the catalog before submission. A trial
that started under one version and submits under another must either re-map or justify the
difference.

### 8.4 Coding dictionaries

MedDRA is released twice yearly, in March and September. WHODrug has its own release cycle. The
version used for a trial must be recorded in the dataset, because the same verbatim term can map to
different preferred terms in different versions, and a reviewer comparing the dataset with the
protocol or the safety database needs to know which version applied.

A mid-trial upgrade creates a mixed coding population. It is sometimes necessary, for example when a
new term is needed for an emerging safety issue, but it should be a deliberate decision with a
documented migration, not an automatic consequence of a vendor update.

### 8.5 Europe, devices, and decentralised trials

In the EU, the Clinical Trials Regulation requires trial data to be submitted through CTIS and gives
inspectors access to source data. The regulation does not prescribe SDTM; the standard is a
submission convention rather than a legal requirement, but the practical expectation of a marketing
application is alignment with the international standards.

For device trials, ISO 14155:2020 governs the clinical investigation and adds requirements for
device accountability and traceability. For trials with decentralised elements, the FDA's 2024
guidance confirms that the data standards and Part 11 requirements are unchanged; what changes is
where and by whom the data are captured.

### 8.6 What the standards do not constrain

The standards do not specify which variables an eCRF should collect beyond those needed for the
submission domains, how long a query should stay open, how a sponsor should code a partial match, how
much source data verification is enough, or how a protocol amendment should be versioned in the
database. Those are sponsor decisions, documented in the data management plan. The standards
constrain the output, not the process.

> **Note:** This section is educational and reflects published documents at the access date. It is
> not legal advice, and the applicable version must be confirmed with the regulator before filing.

## 9. Industry practice and case evidence

### 9.1 What is standard practice

Across large sponsors, several practices are close to universal:

- an eCRF built from a data requirement specification, with CDASH conventions where they apply;
- automated edit checks plus a manual review plan;
- medical coding against a fixed dictionary version, with a coding guideline;
- SAE reconciliation between the clinical and safety databases;
- a documented lock checklist with sign-offs;
- SDTM and ADaM conversion with independent quality control;
- define.xml and an annotated CRF in the submission package.

The variation is in how early these are done, not whether they are done. A sponsor that designs the
eCRF before the estimand is fixed will do all of the above and still collect the wrong variables.

### 9.2 What the evidence shows about queries

There is no public benchmark of query rates that generalises across therapeutic areas and designs.
Published rates vary with the complexity of the protocol, the maturity of the sites, and the
sensitivity of the edit checks. A rate that is high relative to a sponsor's own history is
informative; a rate compared across sponsors is not.

The more useful evidence is internal: queries per field by form, reopen rate by question type, and
the share of queries that changed a value versus confirmed it. These decompose the total into design,
site, and question-quality effects.

### 9.3 What the evidence shows about central review

Central statistical review of clinical data detects patterns that site review misses. The methods
overlap with the central monitoring methods described in topic 02: distribution checks, digit
preference, duplicate-value detection, and multivariate outlier scores.

The evidence on performance is mixed in the same way. Highly sensitive methods produce many false
alarms; highly specific methods miss moderate deviations. The practical use is triage: a flag selects
records for human review, and the review decides.

### 9.4 Documented failure patterns at filing

Technical rejection of study data is a documented category of submission failure. The conformance
guide lists validation errors that trigger rejection, including errors in the electronic file
directory, missing define files, and non-conforming dataset structure. These are not scientific
problems; they are format problems that stop the review.

> **Inference:** the frequency of technical rejection is not publicly quantified in a way that
> supports a specific rate. The mechanism is documented; the base rate is not. A sponsor should
> assume the risk is material because the consequence is a stopped review, not a delayed one.

### 9.5 The economics of coding

Coding is labour-intensive and unevenly distributed. A trial may generate several thousand distinct
adverse event terms, of which a large share are auto-coded and the remainder require a human
decision. The cost per term falls with automation but the error cost rises if the automation is not
validated and audited.

Industry practice is moving toward assisted coding with a validated rule set and a sample audit,
rather than full automation. The reason is not technical capability but defensibility: a coder can
explain a decision, and a regulator can review it.

> **Inference:** adoption rates for assisted coding are not publicly reported in a consistent way.
> The direction of travel is observable in vendor offerings; the penetration is not.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| eCRF designed before the estimand | Statistician requests variables after go-live | Analysis plan treated as downstream | Missing variables; restricted analysis | Sign off the data specification against the estimand |
| Free text where a code list is possible | High coding backlog in a common field | Design convenience | Cost and inconsistency | Use controlled terminology at design |
| Edit checks test type only | Queries confirm values rather than correct them | Clinical logic not encoded | Weak data with a clean audit trail | Write logic checks with clinical review |
| Dictionary version not fixed | Coding report shows two versions | Vendor update applied mid-trial | Mixed coding population | Lock the version at first patient |
| Ambiguous query questions | Reopen rate above 15% | Sponsor-side question quality | Rework and a longer tail | Standard question templates; review reopen rate |
| Coding not on the lock dashboard | No coding metric in the plan | A different team owns it | Coding becomes the binding constraint | Add coding to the critical path |
| SAE reconciliation deferred | Safety and clinical counts diverge | Separate databases | Submission inconsistency | Monthly reconciliation |
| External data loaded late | Laboratory identifiers do not match | Transfer specification missing | Manual matching at lock | Agree the specification before the first transfer |
| Lock checklist treated as a formality | Items signed without evidence | Deadline pressure | Lock with unresolved critical queries | Require evidence per item |
| Amendment adds variables without a version plan | Old and new patients in one field | Change-control gap | Structural missingness | Version the database and document the change |
| Analysis flags derived informally | Flags do not reproduce the population | No specification | Wrong analysis set | Derive from a written rule and verify |
| Define file produced at the end | Mapping undocumented during build | Metadata treated as documentation | Non-conformance and review delay | Build metadata alongside the datasets |
| Archive lacks programs | Cannot reproduce the table | Archive scope too narrow | Loss of traceability | Define archive contents at study start |

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test |
|---|---|---|---|
| Query arrival is roughly uniform | Sizes capacity | Batch entry before visits | Monthly arrival counts |
| One hour of work per query | Costs the process | Complex queries take longer | Time a sample |
| Coding capacity is stable | Predicts the coding path | Staff leave, vendor change | Track terms per coder per day |
| Auto-coding accuracy is acceptable | Justifies automation | Rare or ambiguous terms | Sample audit against manual coding |
| Reconciliation takes 10 days | Plans the lock | Many SAEs or slow sites | Historical reconciliation logs |
| Quality control takes 10 days | Plans the lock | Dataset complexity | Prior study actuals |
| External data arrive on schedule | Keeps the path parallel | Laboratory or vendor delay | Transfer logs |
| The analysis plan will not change | Fixes the variable set | Late endpoint change | Change-control record |
| Sites enter data within 7 days | Sets the entry path | Staff shortages | Entry lag by site |
| A locked database will not be unlocked | Preserves the audit trail | New safety information | Unlock log and impact assessments |

### 11.1 Limits of the model

The lock model treats stages as independent and durations as known. Real stage durations are
distributions, and the maximum of several distributions is longer than the maximum of their means.
The model therefore understates the tail in the same way the query model in topic 02 does.

The cost model uses a single daily rate. In practice, a delay that crosses a quarter boundary has
different financial consequences depending on revenue recognition and milestone timing.

The model also assumes that the plan's 45 days is a fixed commitment. If the commitment can be
renegotiated, the cost of the slip is the cost of the negotiation, not the full daily rate.

## 12. Exercises

### Exercise 1. Write the data requirement specification (application, 45 minutes)

A trial's primary endpoint is change from baseline in a symptom score at week 12, analysed using a
mixed model for repeated measures. The estimand specifies that initiation of rescue medication is
an intercurrent event handled by a hypothetical strategy.

**Deliverable:** a specification for the variables needed to implement this estimand, including the
symptom score, the rescue medication dates, the baseline definition, the visit windows, and the
analysis population flags.

**Acceptance criteria:** every variable has a source and a format; the intercurrent event handling is
visible in the variables; at least one variable is identified as derived; the baseline definition is
unambiguous.

**Time:** 45 minutes.

### Exercise 2. Critique an eCRF (critique, 45 minutes)

An eCRF collects "reason for discontinuation" as free text, "date of last dose" as three separate
month, day, and year fields, and "was the dose taken" as a yes/no field without a "not applicable"
option. The form is used at 60 sites.

**Deliverable:** a critique with a specific redesign for each field and the query or coding
consequence of the current design.

**Acceptance criteria:** each critique names the downstream consequence; the redesign uses a
controlled codelist or a documented derivation; the date format issue is linked to the SDTM
requirement.

**Time:** 45 minutes.

### Exercise 3. Design edit checks (design, 45 minutes)

For a visit form with consent date, visit date, birth date, dose, and adverse event start date,
write ten edit checks. Classify each as a type check, a logic check, or a cross-form check.

**Deliverable:** a table with the check, the fields involved, the trigger condition, the message, and
the class.

**Acceptance criteria:** at least four checks are logic or cross-form; no check produces a query for a
value that could be legitimately missing; the message tells the site what to do.

**Time:** 45 minutes.

### Exercise 4. Compute the cleaning cycle (calculation, 45 minutes)

A trial has 400 patients and 8 queries per patient. Cycle times are 10 days for 50%, 20 days for 30%,
and 40 days for 20%. The lock plan assumes closure within 20 days of LPLV. Fixed cost is $500,000 per
month.

**Deliverable:** mean cycle time, open queries at LPLV, the realistic lock date, the slip, and the
cost.

**Acceptance criteria:** the mean is computed from the distribution; the maximum is used for the
lock; the cost is in 2026 US dollars; the difference between mean and maximum is explained.

**Time:** 45 minutes.

### Exercise 5. Plan the coding (application, 45 minutes)

A trial generates 4,000 adverse event terms. Auto-coding handles 55%, and the remainder need manual
coding at 25 terms per coder per day. Two coders are available, one of whom is on leave for two
weeks of the four-week lock window.

**Deliverable:** a coding capacity plan showing the manual backlog, the capacity per week, the
completion date, and the effect of the leave.

**Acceptance criteria:** capacity is computed per week, not per month; the leave is modelled as a
reduction, not an average; the completion date is compared with the lock date.

**Time:** 45 minutes.

### Exercise 6. Decide on lock (critique, 45 minutes)

At the planned lock date, 40 queries remain open. Twelve concern concomitant medications not in the
analysis, twenty concern visit dates for secondary endpoints, and eight concern the date of
progression for the primary endpoint.

**Deliverable:** a lock decision with a rationale for each group, the documentation required, and the
effect on the analysis.

**Acceptance criteria:** the decision distinguishes critical from non-critical queries; the waiver
rationale is specific; the effect on the primary analysis is stated; no group is left undecided.

**Time:** 45 minutes.

### Exercise 7. Critique the traceability package (critique, 60 minutes)

A submission package contains SDTM datasets, ADaM datasets, define.xml, and the statistical analysis
plan. It does not contain the annotated CRF, the conversion programs, or a reviewer's guide.

**Deliverable:** a critique identifying what cannot be reconstructed, which reviewer questions cannot
be answered, and what must be added before filing.

**Acceptance criteria:** each gap is linked to a specific review activity; the distinction between a
format gap and a scientific gap is made; the remediation is realistic.

**Time:** 60 minutes.

### Exercise 8. Design the archive (design, 45 minutes)

A sponsor is closing a trial that will support a marketing application in the US and the EU.

**Deliverable:** an archive manifest listing what is kept, in what format, for how long, and who can
access it.

**Acceptance criteria:** the manifest supports reproduction of the primary analysis; retention
periods are stated with their legal basis; the format is readable without proprietary software where
required; access controls are described.

**Time:** 45 minutes.

## 13. Self-check answers

### 13.1 Exercise 1

A strong specification includes:

| Variable | Source | Format | Type |
|---|---|---|---|
| Symptom score at each visit | eCRF, site assessment | Numeric 0 to 100 | Collected |
| Visit date | eCRF | ISO 8601 date | Collected |
| Baseline score | Derived: last non-missing score before first dose | Numeric | Derived |
| Rescue medication start date | eCRF | ISO 8601 date | Collected |
| Rescue medication stop date | eCRF | ISO 8601 date | Collected |
| Intercurrent event flag | Derived from rescue start date and randomisation date | Y/N | Derived |
| Analysis population flag | Derived from eligibility and dosing | Y/N | Derived |

The hypothetical strategy requires the score after rescue initiation to be set to missing for the
primary analysis, with the patient retained in the analysis. That rule must be visible in the
derivation, not applied informally by the programmer.

### 13.2 Exercise 2

- **Free-text reason for discontinuation.** Replace with a controlled codelist plus an "other,
  specify" field. The current design produces inconsistent categories and a coding burden for a
  variable used in disposition tables.
- **Split date fields.** Replace with a single ISO 8601 date field and a "date unknown" checkbox with
  a reason. Split fields create partial dates that cannot be used in derivations and are a common
  source of queries.
- **Yes/no without "not applicable".** Add "not applicable" and "not done" as distinct options. A
  coordinator forced to choose yes or no for a visit that did not happen will choose arbitrarily,
  and the ambiguity will appear as a query pattern.

### 13.3 Exercise 3

Examples:

| Check | Fields | Condition | Class |
|---|---|---|---|
| Visit after consent | Consent date, visit date | Visit date before consent date | Cross-form logic |
| Age at consent | Birth date, consent date | Age under 18 or over 100 | Cross-form logic |
| Dose range | Dose | Dose outside protocol range | Type and logic |
| AE start before consent | AE start date, consent date | Start date before consent | Cross-form logic |
| Missing visit reason | Visit date, reason | Visit missing and reason blank | Logic |
| Future date | All dates | Date after today | Type |

A strong set avoids checks that fire on legitimately missing values, and each message names the field
and the expected action.

### 13.4 Exercise 4

```text
Mean cycle = 0.50x10 + 0.30x20 + 0.20x40 = 5 + 6 + 8 = 19 days
Queries = 400 x 8 = 3,200 over the trial
Arrival rate = 3,200 / 24 = 133.3 per month
Open at LPLV = 133.3 x (19 / 30.44) = 83 queries
Last query closes up to 40 days after LPLV
Planned lock = LPLV + 20 days
Realistic lock = LPLV + 40 days
Slip = 20 days
Cost = 20 x ($500,000 / 30) = $333,333
```

The mean is 19 days, which is close to the 20-day plan. The plan is still wrong because the lock
depends on the maximum, which is 40 days.

### 13.5 Exercise 5

```text
Manual terms = 4,000 x 0.45 = 1,800
Weekly capacity with two coders = 2 x 25 x 5 = 250 terms per week
Weekly capacity with one coder = 125 terms per week
Weeks 1-2 at 250 = 500 terms; weeks 3-4 at 125 = 250 terms
Total in four weeks = 750 terms
Remaining = 1,800 - 750 = 1,050 terms
Additional weeks at 125 = 8.4 weeks
Total = 12.4 weeks from the start of coding
```

If the lock window is four weeks, coding cannot finish. The options are to start earlier, add a
coder, reduce the manual share, or move the lock. Starting earlier is usually the cheapest and is
possible if coding begins as visits complete rather than at LPLV.

### 13.6 Exercise 6

- **Concomitant medications (12 queries).** Not in the analysis. Lock with a documented waiver; record
  the queries as open in the data management report. No effect on the primary analysis.
- **Visit dates for secondary endpoints (20 queries).** Potentially in the analysis. Resolve before
  lock if the secondary endpoint is in the label or the HTA submission; otherwise waive with a
  sensitivity analysis.
- **Progression date (8 queries).** Critical. Do not lock until resolved. If a date cannot be
  determined, record it as missing with the reason and pre-specify the handling.

The decision is not "lock or not lock". It is a disposition for each group, recorded before the
database is frozen.

### 13.7 Exercise 7

Without the annotated CRF, a reviewer cannot map a collected field to an SDTM variable. Without the
conversion programs, the sponsor cannot demonstrate that the datasets were produced by the stated
rules. Without a reviewer's guide, the reviewer must infer the structure from the data.

The gaps are format and traceability gaps, not scientific ones, but they can stop a review. The
remediation is to produce the aCRF from the eCRF specification, retrieve the programs from version
control, and write the guide from the mapping specification.

### 13.8 Exercise 8

A strong manifest includes:

| Item | Format | Retention basis |
|---|---|---|
| Clinical database and audit trail | Native and exportable format | 21 CFR 312.62(c); EU CTR |
| SDTM and ADaM datasets | XPT v5 plus define.xml | Submission retention |
| Analysis programs | Text or version-controlled repository | Reproducibility |
| Annotated CRF | PDF | Submission |
| Trial master file | Indexed PDF and native | GCP and national law |
| Randomisation and unblinding records | Native | GCP |

The manifest must state who can access the archive and how it will be read if the original software is
no longer available.

## 14. Interview questions

**1. What is the first thing you check before an eCRF is built?**
Strong: that the estimand and analysis population are defined, and that the data requirement
specification covers them. Weak: starts with the form layout.

**2. What is the difference between SDTM and ADaM?**
Strong: SDTM is a tabulation standard for review; ADaM is an analysis standard with derivations that
reproduce the analysis. Weak: says they are both submission datasets.

**3. How do you decide whether a query is worth raising?**
Strong: considers the field's use in the analysis, the cost of resolution, and whether the answer
would change a result. Weak: raises every discrepancy.

**4. Why is the reopen rate important?**
Strong: it measures answer quality and multiplies the tail, which determines the lock date. Weak:
treats it as a site metric.

**5. How do you handle a mid-trial MedDRA upgrade?**
Strong: decides deliberately, documents the migration, records the version, and assesses the effect on
ongoing safety review. Weak: accepts the vendor's default.

**6. What belongs in a lock checklist?**
Strong: patient disposition, visit completeness, query disposition, coding, SAE reconciliation,
external data, population flags, audit trail, and blind review. Weak: lists only queries.

**7. What would you do if coding is the binding constraint at lock?**
Strong: quantify the manual backlog and capacity, start coding earlier, add resource or reduce the
manual share, and consider a two-stage lock. Weak: asks the coders to work faster.

**8. How do you make a dataset traceable?**
Strong: annotated CRF, mapping specification, define.xml, programs, and a reviewer's guide, with each
value derivable from a documented source. Weak: says the define file is enough.

**9. What is the difference between a data cut and a lock?**
Strong: a cut is a snapshot for an analysis; a lock is a permanent freeze with a checklist. Weak: uses
the terms interchangeably.

## 15. Key terms

| Term | Meaning | See |
|---|---|---|
| ADaM | Analysis Data Model | 3.2 |
| aCRF | Annotated CRF mapping fields to SDTM | 5.20 |
| CDASH | Data acquisition standard | 5.2 |
| Controlled terminology | Standard codelists | 5.2 |
| Data cut | Snapshot for an analysis | 4.3 |
| Data requirement specification | Contract between protocol and database | 5.1 |
| Define-XML | Machine-readable dataset metadata | 5.20 |
| DCF | Data clarification form | 4 |
| Edit check | Validation rule | 5.3 |
| eCRF | Electronic case report form | 4 |
| External data | Data from outside the EDC | 5.4 |
| Hard lock | Permanent freeze | 4.3 |
| MedDRA | Adverse event dictionary | 5.7 |
| Missing-data code | Reason a value is absent | 5.16 |
| Query | Request to resolve a discrepancy | 5.6 |
| Reopen rate | Closed queries reopened | 5.19 |
| SAE reconciliation | Matching safety and clinical data | 5.8 |
| SDTM | Tabulation standard | 3.2 |
| Soft lock | Temporary freeze | 4.3 |
| Traceability | Ability to follow a value to source | 3.2 |
| UAT | User acceptance testing | 5.3 |
| WHODrug | Medication dictionary | 5.7 |

## 16. Sources

All URLs accessed 26 October 2026.

1. US Food and Drug Administration. *Study Data Technical Conformance Guide*, version 6.3,
   October 2026. https://www.fda.gov/media/153632/download
2. US Food and Drug Administration. *Providing Regulatory Submissions in Electronic Format:
   Standardized Study Data*, June 2021.
   https://www.fda.gov/regulatory-information/search-fda-guidance-documents/providing-regulatory-submissions-electronic-format-standardized-study-data
3. US Food and Drug Administration. *Study Data Standards Resources*.
   https://www.fda.gov/industry/fda-data-standards-advisory-board/study-data-standards-resources
4. US Food and Drug Administration. *Data Standards Catalog*.
   https://www.fda.gov/industry/fda-data-standards-advisory-board
5. CDISC. *SDTMIG v3.4*. https://www.cdisc.org/standards/foundational/sdtmig/sdtmig-v3-4
6. CDISC. *ADaM*. https://www.cdisc.org/standards/foundational/adam
7. CDISC. *Define-XML*. https://www.cdisc.org/standards/data-exchange/define-xml
8. US Government. *21 CFR Part 11*.
   https://www.ecfr.gov/current/title-21/chapter-I/subchapter-A/part-11
9. ICH. *E6(R3) Good Clinical Practice*, 6 January 2025.
   https://database.ich.org/sites/default/files/ICH_E6%28R3%29_Step4_FinalGuideline_2025_0106.pdf
10. ICH. *E9(R1) Addendum on Estimands and Sensitivity Analysis*.
    https://www.ich.org/page/efficacy-guidelines
11. European Commission. *Clinical Trials Regulation (EU) No 536/2014*.
    https://health.ec.europa.eu/medicinal-products/clinical-trials/clinical-trials-regulation_en
12. International Organization for Standardization. *ISO 14155:2020*.
    https://www.iso.org/standard/71690.html
13. US Food and Drug Administration. *Conducting Clinical Trials With Decentralized Elements*,
    September 2024.
    https://www.fda.gov/regulatory-information/search-fda-guidance-documents/conducting-clinical-trials-decentralized-elements
14. Medical Dictionary for Regulatory Activities. *MedDRA*.
    https://www.meddra.org/

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`02_site-and-monitoring.md`](02_site-and-monitoring.md)
- Next topic: [`04_health-technology-assessment.md`](04_health-technology-assessment.md)
- Global regulation: [`06_global-regulation.md`](06_global-regulation.md)
- Module survey: [`../../course/25_trial_operations_hta_global_regulatory.md`](../../course/25_trial_operations_hta_global_regulatory.md)
- Labs: [`../../labs/`](../../labs/)
