# Submission Data: eCTD Structure, Dataset Reconciliation, and What Reviewers Check

> Module 13 — Biopharma and Clinical Development
> Topic 10 of 10
> Estimated study time: 5 hours
> Prerequisites: topics 02 and 05
> Version 2026.3

## 1. Why this topic exists

A marketing application is a legal act and a technical object at the same time. The legal act asks a
regulator to approve a product on the basis of evidence. The technical object is a package of
documents and datasets, submitted electronically, that must pass automated validation before a human
reads a page of it. A package that fails validation is refused at the gate, and the review clock does
not start.

The technical failure modes are unglamorous and expensive. A missing trial summary dataset causes
more rejections than any scientific deficiency. A dataset that is present but untagged is treated as
absent. A define.xml that describes a previous version of the datasets is worse than none, because it
appears authoritative and is wrong. These are not research problems; they are assembly problems, and
they are preventable with a reconciliation that takes days.

The scientific failure modes are subtler. A reviewer opens the clinical study report, finds the
primary efficacy table, and asks whether the number can be reproduced from the submitted datasets. If
the analysis dataset cannot be located, if the derivation is not described, or if the population flag
does not match the table's denominator, the review generates an information request. The request
costs months, and it signals that the sponsor's internal quality controls did not catch what an
external reviewer found in an afternoon.

This topic covers the eCTD structure, the study data technical rejection criteria, the dataset
inventory and reconciliation that prevent assembly failures, the reviewer's guide that explains what
the metadata cannot, and the lifecycle of a submission after it is sent. It is the last topic in the
module because it assembles everything the previous topics produced.

### 1.1 The failure pattern

1. A sponsor assembles a marketing application with three studies.
2. The eCTD validates, and the submission is sent.
3. The study data validation finds that one study has no trial summary dataset, failing technical
   rejection criterion 1734.
4. The submission is refused, and the review clock does not start.
5. The team adds the file and resubmits, and a second validation finds four datasets with incorrect
   file tags, failing criterion 1735.
6. The submission is refused again.
7. Two weeks are lost, and the filing date moves.

Each cycle is preventable. The first failure is a missing file that an inventory would have caught;
the second is a tagging error that a validation run would have caught before submission.

The two failures also illustrate the difference between checking a package and checking a single finding. The
team checked the finding it was told about, not the package it was sending.

### 1.2 Who is blocked when submission assembly fails

| Stakeholder | What they lose | Why they cannot fix it alone |
|---|---|---|
| Regulatory operations | The filing date | Depends on the study teams' datasets |
| Programme team | Two weeks of review time and credibility | Cannot see the validation rules until submission |
| Statistical programming | A clean package | Receives the inventory late |
| Reviewer | Time and a clear package | Receives what was submitted |
| Patient | Access delayed | Has no role in the process |
| Investor | A milestone | Depends on the filing |

### 1.3 Three facts that explain most refusals

| Fact | Consequence |
|---|---|
| The rejection criteria are machine-checkable | A pre-submission run prevents most refusals |
| The inventory is usually built from the folder | Missing files are invisible |
| The reviewer's path is not tested | A validating package can still be unreviewable |

The three facts point to the same remedy: derive the inventory from the specifications, run the
checks as a set, and simulate the reviewer before transmission.

### 1.4 The two clocks

A submission runs on two clocks. The regulatory clock starts at filing and governs the review. The
programme clock runs from the decision to submit to the filing date, and it is the one the team
controls.

Most of the pressure in the last month comes from confusing the two: the team behaves as if the
regulatory clock has started before the package is filed, and rushes the assembly. The assembly is
exactly the part that benefits from a slower, checked process.

### 1.5 The industry question

> Can this package pass validation, and can a reviewer reproduce the claims from it without asking us
> a question?

The question has two parts because validation and reproducibility are different tests. A package can
pass every automated check and still fail the reviewer's reproduction test, and a package that would
pass the reproduction test can be refused because a file tag is wrong.

The lesson is that validation is a set, not an item.

## 2. Learning outcomes

By the end of this topic you can:

1. **Map** a marketing application to the eCTD modules and describe what belongs in each.
2. **Build** a dataset inventory and reconcile it against the folder contents and the metadata.
3. **Apply** the FDA study data technical rejection criteria to a package and classify the failures.
4. **Diagnose** a package that validates but cannot be reviewed, and one that can be reviewed but
   fails validation.
5. **Write** a reviewer's guide that answers the three reviewer questions.
6. **Plan** the submission sequence, including pre-submission validation, ESG submission, and
   lifecycle updates.
7. **Assess** the readiness of a package against a checklist and make a go/no-go recommendation.

## 3. The core mental model

### 3.1 The eCTD structure

```text
 Module 1  Administrative information (regional)
 Module 2  Common technical document summaries
             2.1 Table of contents
             2.2 Introduction
             2.3 Quality overall summary
             2.4 Nonclinical overview and summaries
             2.5 Clinical overview and summaries
 Module 3  Quality (chemistry, manufacturing, controls)
 Module 4  Nonclinical study reports
 Module 5  Clinical study reports
             5.1 Table of contents
             5.2 Tabular listing of all clinical studies
             5.3 Clinical study reports
             5.3.1 Reports of biopharmaceutic studies
             5.3.2 Reports of studies pertinent to pharmacokinetics
             5.3.3 Reports of efficacy and safety studies
             5.3.4 Reports of post-marketing experience
             5.3.5 Synopses of individual studies
             5.4 Literature references
```

The datasets sit inside Module 5 (and Module 4 for nonclinical SEND data), in study-specific folders,
with the tabulation datasets, the analysis datasets, the define.xml files, and the reviewer's guide.
The structure is not a filing convention; it is the navigation system a reviewer uses.

### 3.2 The two validation layers

```text
 Layer 1: eCTD validation          Layer 2: study data validation
 ------------------------          ----------------------------
 File structure, XML backbone,     Dataset presence, format,
 document types, lifecycle         define.xml, terminology,
 operations, file names            STF tags, study identifiers

 Failure: the submission is        Failure: the study data are
 refused as an eCTD                refused, and the review
                                   clock does not start
```

The layers are validated by different rules and often by different teams. A package can pass Layer 1
and fail Layer 2, which is the most common surprise in submission operations.

### 3.3 The inventory

```text
 Study 301         Study 302          Study 303
 ---------         ---------          ---------
 SDTM 22 datasets  SDTM 24 datasets   SDTM 21 datasets
 ADaM 9 datasets   ADaM 11 datasets   ADaM 8 datasets
 define.xml (SDTM) define.xml (SDTM)  define.xml (SDTM)
 define.xml (ADaM) define.xml (ADaM)  define.xml (ADaM)
 ts.xpt            ts.xpt             ts.xpt
 reviewer's guide  reviewer's guide   reviewer's guide
```

The inventory is a table of every file that should be present, derived from the study reports and the
programming specifications. Reconciliation compares the inventory with the folder and with the
metadata. The comparison is mechanical and catches the failures that automated validation catches
only after submission.

### 3.4 What the reviewer does

```text
 1. Open the clinical study report and find the primary table
 2. Locate the analysis dataset named in the table
 3. Read the define.xml method for the derived variable
 4. Re-run the analysis or check the numbers
 5. Trace one value back to the tabulation dataset and the CRF
 6. Read the reviewer's guide for known issues
 7. Ask a question if any step fails
```

The reviewer's path is the design specification for the package. Every artefact should shorten one of
the seven steps.

### 3.5 The cost of a refusal

A refusal does not merely delay the review; it consumes the team's remaining capacity at the worst
moment. The resubmission requires a new validation cycle, a new ESG transmission, and a new
acknowledgement. The scientific work is unchanged, but the schedule and the credibility are not.

The asymmetry is the argument for investing in reconciliation: the cost of the check is days, and the
cost of the failure is weeks.

### 3.6 Three questions before transmission

1. **Is every file that should be present actually present, with the right tag, in the right folder?**
2. **Can a reviewer reproduce the primary claim from the datasets and the metadata without asking us?**
3. **If the package is refused, what is the recovery time and the schedule impact?**

The third question is the one that changes behaviour. A team that has calculated the recovery time
invests in prevention; a team that has not will discover the cost at the worst moment.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| eCTD | Electronic common technical document | Used for any electronic submission | Format |
| Module | One of the five eCTD sections | Confused with a folder | Structure |
| Study tagging file (STF) | The XML file that links study documents and datasets | Omitted or mistagged | XML |
| Technical rejection criteria (TRC) | FDA rules that cause a study data submission to be refused | Confused with validation warnings | Rules |
| Validation criteria | eCTD rules applied to the submission structure | Confused with TRC | Rules |
| Define-XML | Machine-readable metadata for datasets | Treated as optional | XML |
| Trial summary (TS) domain | Study-level parameters, including versions | Omitted | Dataset || Technical conformance guide | FDA guidance on study data structure and content | Treated as stable | Guidance |
| Trial summary (TS) domain | Study-level parameters, including versions | Omitted | Dataset || Backbone | The XML index of the submission | Confused with the folder structure | XML |
| Leaf | A document or dataset in the backbone | Confused with a folder | Structure |
| Sequence | A submission within the application lifecycle | Confused with a version number | Number || Validation report | The output of the eCTD or study data validator | Treated as a quality certificate | Report |
| Reviewer simulation | A structured walkthrough of the package | Confused with validation | Activity |
| Dataset inventory | The list of files that should be in the package | Confused with the folder listing | Document |
| Reconciliation | Comparison of inventory, folder, and metadata | Confused with validation | Activity |
| Reviewer's guide | The sponsor's explanation of the datasets | Confused with Define-XML | Document |
| Filing | The agency's acceptance of the application for review | Confused with approval | Milestone |
| Review clock | The regulatory timeline that starts at filing | Confused with the submission date | Duration |
| Information request | A request from the reviewer for clarification or data | Confused with a deficiency letter | Communication |
| Complete response letter | The agency's decision that the application is not approvable as submitted | Confused with a rejection | Decision |
| Lifecycle | The sequence of submissions to an application over time | Confused with the review | Process |
| Sequence number | The identifier for a submission in the lifecycle | Omitted | Number |
| eCTD v3.2.2 | The earlier eCTD version, still supported | Assumed obsolete | Version |
| eCTD v4.0 | The current eCTD version, accepted for new applications since September 2024 | Assumed mandatory | Version |
| Forward compatibility | The ability to submit v4.0 to an application established under v3.2.2 | Confused with a mandate | Capability |
| ESG | Electronic Submissions Gateway | Confused with the eCTD | System |
| SEND | Standard for Exchange of Nonclinical Data | Confused with SDTM | Standard |
| ADaM | Analysis Data Model | Confused with SDTM | Standard |
| File tag | The eCTD attribute identifying a file's type | Omitted or wrong | Attribute |
| Study identifier | The sponsor's study number, consistent across files | Inconsistent | Identifier |
| Duplicate file name | Two files with the same name submitted as new | Mistaken for a replacement | Error |
| Replacement | A submission that replaces a previously submitted file | Confused with an append | Operation |
| Append | A submission that adds a file | Confused with a replacement | Operation |
| Technical validation | Automated checking of structure and metadata | Confused with scientific review | Process |
| Pre-submission | Activities before the submission, including validation | Confused with the submission | Phase |

### 4.1 The two audiences of the package

The package has two audiences with different needs. The automated validator needs structure: files in
the right place, with the right names, tags, and metadata. The reviewer needs meaning: a navigable
path from a claim to the data that support it.

A package designed only for the validator will pass and be hard to review. A package designed only
for the reviewer may fail validation. The reconciliation discipline in Section 5 addresses both.

## 5. The mechanics, step by step

This procedure assembles and submits a marketing application. It assumes the study reports and the
datasets are complete and that the submission scope is agreed.

### Step 1. Define the submission scope

**Input:** the programme strategy, the target regions, and the indication.

**Output:** a scope statement: the application type, the studies included, the modules required, the
regions, and the submission dates.

**Rationale:** The scope determines the package. A US new drug application (NDA) and an EU marketing
authorisation application (MAA) share the common technical document but differ in Module 1 and in some
study data requirements. A scope statement prevents the late discovery that a region needs a document
the team did not prepare.

**Branch point:** if the application relies on a single pivotal study, the package is smaller but the
review is more concentrated; if it relies on several, the reconciliation is larger and the consistency
across studies becomes the main risk.

### Step 2. Map the package to the eCTD modules

**Input:** the scope and the table of contents.

**Output:** a module map with the documents and datasets that belong in each module and subfolder.

**Rationale:** The map is the navigation design. A document in the wrong module is not merely
inconvenient; it may not be reviewed at all. The map should be built from the reviewer's perspective,
not from the team's organisational structure.

### Step 3. Build the dataset inventory

**Input:** the study reports, the programming specifications, and the standards declaration.

**Output:** a table of every dataset and metadata file, with its study, its type (SDTM, ADaM, SEND),
its file name, and its location.

**Rationale:** The inventory is the reference for reconciliation. It should be derived from the
specifications, not from the folder listing, because the folder listing cannot tell you what is
missing.

### Step 4. Validate the datasets

**Input:** the datasets and the validation tool.

**Output:** a findings log with reject, warning, and informational findings, each with a disposition.

**Rationale:** Dataset validation checks structure, terminology, and cross-dataset rules. It does not
check the analysis. The findings must be triaged and resolved or justified before the package is
assembled.

### Step 5. Validate the metadata

**Input:** the define.xml files.

**Output:** a metadata check: every dataset described, every variable with an origin, every codelist
resolved, every method present for derived variables.

**Rationale:** The metadata is what the reviewer reads to reproduce the analysis. A define.xml that
validates but lacks methods passes the tool and fails the reviewer.

### Step 6. Apply the study data technical rejection criteria

**Input:** the assembled study data folders.

**Output:** a TRC check with each criterion marked pass or fail.

**Rationale:** The TRC are the rules that cause a submission to be refused before review. They are a
small, checkable set, and they should be checked before transmission, not discovered in the agency's
validation report.

**Branch point — the criteria that most often fail:**

| Criterion | What it requires | Common cause |
|---|---|---|
| TS dataset present | A trial summary dataset for each study | Omitted for a study |
| STF file tags correct | The right tag for each dataset type | Copy-paste error |
| DM and define.xml present | Required files for SDTM | define.xml misplaced |
| ADSL and define.xml present | Required files for ADaM | ADSL named differently |
| One new file per name | No duplicate new files | Split datasets or resubmission |
| Study ID consistent | STF study ID matches the TS dataset | Renamed study |
| No untagged files | Every file in a study folder has an STF entry | Legacy file added late |

### Step 7. Assemble the eCTD

**Input:** the documents, datasets, and metadata.

**Output:** an eCTD package with the XML backbone, the module structure, and the lifecycle operations.

**Rationale:** The assembly is a build, not a filing. It should be scripted or tool-driven and
reproducible, because manual assembly is where the tags and names go wrong.

### Step 8. Quality control the package

**Input:** the assembled package.

**Output:** a QC record: the validation report, the reconciliation, the reviewer's guide review, and
the sign-off.

**Rationale:** The QC is the last chance to catch an error before the agency does. It should include a
reviewer simulation: open the package as a reviewer and follow the seven steps in Section 3.4.

### Step 9. Submit through the gateway

**Input:** the validated package.

**Output:** a transmission record and an acknowledgement.

**Rationale:** The Electronic Submissions Gateway (ESG) requires registration and testing, and the
transmission can fail for reasons unrelated to the content. The submission should be scheduled with
time for a retry.

### Step 10. Track and respond

**Input:** the acknowledgement and the agency's validation report.

**Output:** a record of the filing decision, any information requests, and the responses.

**Rationale:** The period after submission is part of the process. A validation failure requires a
resubmission; an information request requires a response within the agency's timeline. The team should
be staffed for both.

### Step 11. Manage the lifecycle

**Input:** the approved application and subsequent changes.

**Output:** a lifecycle plan: supplements, variations, annual reports, and the sequence numbering.

**Rationale:** A submission is not a single event. The application accumulates documents over years,
and each addition must be a correct lifecycle operation. An append where a replace was required
creates a duplicate that a reviewer must resolve.

### Step 12. Learn from the validation report

**Input:** the agency's validation and review outcomes.

**Output:** a defect log with root causes and preventive actions for the next submission.

**Rationale:** The same errors recur across submissions because the lessons are not captured. A defect
log that distinguishes content errors from assembly errors allows the organisation to fix the process
rather than the package.

### 5.1 What belongs where

| Module | Content | Study data |
|---|---|---|
| 1 | Regional administrative forms, correspondence, labels | None |
| 2 | Summaries: quality, nonclinical, clinical overview and summary | None |
| 3 | Drug substance, drug product, manufacturing, controls | None |
| 4 | Nonclinical study reports | SEND datasets, define.xml, nSDRG |
| 5.3.1 | Biopharmaceutic study reports | SDTM, ADaM, define.xml |
| 5.3.2 | Pharmacokinetic study reports | SDTM, ADaM, define.xml |
| 5.3.3 | Efficacy and safety study reports | SDTM, ADaM, define.xml, reviewer's guide |
| 5.3.4 | Post-marketing experience | As applicable |
| 5.4 | Literature references | None |

A study dataset placed in the wrong module may never be reviewed, and a document placed in a study
folder without an STF entry may be treated as missing. The map is the first check in assembly.

### 5.2 The submission timeline

| Phase | Duration | Activities | Risk |
|---|---|---|---|
| Scope and planning | 3-6 months before | Application type, modules, regions | Scope creep |
| Dataset lock | 3-4 months before | Final datasets and metadata | Late analysis changes |
| Validation | 6-8 weeks before | Dataset and metadata checks | Findings discovered late |
| Assembly | 3-4 weeks before | eCTD build, tagging | Manual errors |
| QC and reviewer simulation | 2 weeks before | Reconciliation, walkthrough | Insufficient time |
| Transmission | 1 week before | ESG submission, acknowledgement | Gateway failure |
| Agency validation | 1-2 weeks after | Filing decision | Rejection |

The timeline is compressed at the end because the scientific work takes longer than planned. The
compression is where the assembly errors occur, which is why validation should start before the
datasets are final, on a near-final version.

### 5.3 Triage of validation findings

| Finding class | Example | Action | Can it ship? |
|---|---|---|---|
| Reject-level structural | Required variable missing | Fix and re-run | No |
| Reject-level technical | TS dataset missing | Fix and re-run | No |
| Terminology | Invalid term | Re-code or justify extension | No |
| Cross-dataset | Subject in AE not in DM | Investigate and fix | No |
| Business rule | Sponsor rule violated | Review and justify | Yes, with rationale |
| Cosmetic | Label length | Fix if time allows | Yes |

The triage depends on knowing which rules come from the standard and which from the sponsor's own
conventions. A team that treats every business-rule warning as a reject will never submit; a team
that treats a reject as a warning will be refused.

### 5.4 The review simulation

The simulation is a structured walkthrough performed by someone who did not build the package:

1. Open the clinical study report and locate the primary efficacy table.
2. Identify the analysis dataset and variable named in the table.
3. Open the define.xml and read the method for that variable.
4. Recompute the table from the dataset.
5. Trace one subject's value to the tabulation dataset and to the CRF.
6. Read the reviewer's guide and check that the known issues are stated.
7. Record the time taken for each step.

A package that takes more than an hour for steps 1-5 has a navigation problem that will generate
questions, even if every file validates.

### 5.5 Lifecycle operations

| Operation | Use | Consequence if wrong |
|---|---|---|
| New | First submission of a file | Duplicate if the file already exists |
| Append | Adding a document to an existing leaf | Orphan if the leaf does not exist |
| Replace | Replacing a previously submitted file | Duplicate if submitted as new |
| Delete | Removing a file | Loss of the audit trail if misused |

The most common lifecycle error is a replacement submitted as new, which leaves two versions in the
package and forces the reviewer to decide which is current. The eCTD backbone records the operations,
and the validation checks them, but the check cannot know the sponsor's intent.

### 5.6 Metadata QC checks

| Check | Rule | Failure consequence |
|---|---|---|
| Dataset coverage | Every dataset in the folder is described | Reviewer cannot interpret the dataset |
| Variable coverage | Every variable is described | Method and origin missing |
| Origin present | Every variable has an origin | Traceability failure |
| Codelist resolution | Every codelist resolves to a value set | Invalid values undetected |
| Method for derived variables | Every derived variable has a method | Analysis cannot be reproduced |
| Value-level metadata | Present where meaning varies by parameter | Mixed parameters misinterpreted |
| Dictionary versions | Declared in define.xml and TS | Reproducibility failure |
| Stylesheet present | In the same folder as define.xml | Render failure |

The metadata checks are cheap and catch the failures that validation tools report as warnings, which
teams often leave unresolved.

### 5.7 Cross-study consistency

A marketing application with several studies must be consistent across them: the same variable names,
the same codelists, the same population definitions, and the same dictionary versions. Inconsistency
is not always an error, but it requires an explanation in the reviewer's guide.

| Element | Consistency requirement | Acceptable variation |
|---|---|---|
| Population flags | Same names and definitions | None without explanation |
| Dictionary versions | Same version across studies | Justified by study dates |
| Controlled terminology | Same package | Justified by study dates |
| Dataset names | Same structure | Study-specific domains |
| Units | Same units for the same analyte | Conversion documented |

A reviewer who finds two definitions of the same population flag in one application will ask which
one the integrated analysis used. The answer should be in the guide, not in an information request.

### 5.8 Pre-submission interactions

A pre-submission meeting is an opportunity to confirm the package's scope and the agency's
expectations. The questions worth asking are specific: whether a dataset structure is acceptable,
whether an analysis is adequate, and whether a study is needed. A meeting that asks "is our package
acceptable" will receive a general answer.

The meeting record should be treated as a design input. If the agency comments on a dataset or an
analysis, the comment should be traceable to the package element it changed.

### 5.9 The reviewer's guide as a navigation document

The guide should be short, and it should follow the reviewer's path rather than the team's structure.
A useful order is: the study and its design; the datasets submitted and what each contains; the
primary endpoint derivation; the population definitions; the unusual derivations; the known issues.

The known issues belong near the front. A reviewer who finds an issue that the guide does not mention
will assume it was concealed, and the assumption will colour the rest of the review.

### 5.10 The readiness checklist

| Check | Question | Evidence |
|---|---|---|
| Scope | Are the right studies and modules included? | Scope statement |
| Inventory | Are all expected files present? | Manifest reconciliation |
| TRC | Do the technical criteria pass? | Validation report |
| Metadata | Does every dataset have methods and origins? | Define-XML review |
| Denominators | Do the tables match the population flags? | Reconciliation table |
| Navigation | Can a reviewer reproduce the primary table? | Simulation record |
| Lifecycle | Are the operations correct? | Backbone review |
| Gateway | Is the ESG account tested and ready? | Transmission test |

A package that passes all eight is ready. A package that passes seven and defers one is a decision
with a named risk, and the decision should be recorded.

### 5.11 What reviewers ask after filing

The questions that arrive after filing are predictable, and they are the same three questions in
different forms: where is the dataset that supports this table; why does this count differ from the
one in the summary; and how was this variable derived. A package that anticipates them reduces the
number of information requests.

The anticipation is not guesswork. The reviewer's guide and the define.xml methods are the answers,
and they should be written before the questions arrive.

### 5.12 Who signs off

The readiness decision needs a named accountabler who is not the person who built the package. In
practice this is the regulatory operations lead, supported by the statistical programming lead and
the regulatory lead. The sign-off should record the checks performed, the exceptions accepted, and
the recovery plan if the submission is refused.

A sign-off by the team that built the package is an assertion, not a control.

## 6. Worked example

> **Note:** The application, studies, and counts are fictional. The eCTD versions, technical rejection
> criteria, and regulatory dates cited are real and referenced in Section 16.

### 6.1 The application

A sponsor is submitting a new drug application with three studies:

| Study | Type | SDTM datasets | ADaM datasets | ts.xpt | define.xml |
|---|---|---:|---:|---:|---:|
| 301 | Phase II dose-ranging | 22 | 9 | 1 | 2 |
| 302 | Phase III pivotal | 24 | 11 | 1 | 2 |
| 303 | Phase III supportive | 21 | 8 | 1 | 2 |
| **Total** | | **67** | **28** | **3** | **6** |

The expected file count in the study data folders is 67 + 28 + 3 + 6 = 104, plus three reviewer's
guides, for 107 files.

### 6.2 The reconciliation

| Check | Expected | Found | Difference | Cause |
|---|---:|---:|---:|---|
| SDTM datasets | 67 | 67 | 0 | None |
| ADaM datasets | 28 | 28 | 0 | None |
| ts.xpt files | 3 | 2 | −1 | Study 302 missing |
| define.xml files | 6 | 6 | 0 | None |
| Reviewer's guides | 3 | 3 | 0 | None |
| Files with STF entries | 107 | 103 | −4 | Four datasets untagged |
| Duplicate new file names | 0 | 1 | +1 | Split dataset resubmitted |

The dataset counts reconcile, which is the first sign that the package is complete. The file-level
checks do not: one required file is missing, four files are untagged, and one duplicate exists.

### 6.3 The technical rejection assessment

| Criterion | Requirement | Result | Consequence |
|---|---|---|---|
| TS dataset | One per study | Fail for study 302 | Rejection |
| STF tags | Correct tag per dataset type | 4 incorrect | Rejection |
| DM and define.xml | Present for SDTM | Pass | None |
| ADSL and define.xml | Present for ADaM | Pass | None |
| Duplicate new files | None | 1 duplicate | Rejection |
| Study ID consistency | STF matches TS | Pass | None |
| Untagged files | None | 4 | Rejection |

Four criteria fail. The submission would be refused, and the review clock would not start.

### 6.4 The impact

The package was scheduled for transmission on 15 March. The validation report returns within two
business days, and the team needs five days to fix the files, rebuild the package, revalidate, and
transmit. The filing date moves by at least one week, and the review timeline moves with it.

If the review clock is 10 months, a one-week delay is small in absolute terms. If the application is
competing for a first-to-market position, or if the delay pushes the approval past a patent or
exclusivity date, the cost is not small.

### 6.5 Two diagnostics that reveal a broken package

**Diagnostic 1 — the folder listing is used as the inventory.** A reconciliation that compares the
folder with itself always passes. The diagnostic is to derive the inventory from the programming
specifications and the study reports, then compare it with the folder. In this example, the
specification lists a ts.xpt for study 302, and the folder does not contain it; a folder-only check
would never find the gap.

**Diagnostic 2 — the define.xml dataset count does not match the folder.** Study 302's define.xml
describes 24 SDTM datasets, and the folder contains 24. Study 303's describes 21 and the folder has
21. A package where the counts differ by even one dataset has a metadata or assembly error, and the
check takes a minute.

A third check is the reviewer simulation. Open the primary table for study 302, find the ADaM dataset,
read the method for the response variable, and trace one value to the CRF. If any step fails, the
package is not reviewable even if it validates.

### 6.6 Vary one assumption: the fix order

The team fixes the missing ts.xpt first, assuming that the most common rejection cause is the only
problem. The resubmission fails again on the four incorrect STF tags.

If the team had fixed all four criteria in one cycle, the resubmission would have passed. The
difference is one week of review time and one additional validation cycle.

The lesson is that the rejection criteria are a set, not a ranking. Fixing the most frequent cause
does not help if another reject-level criterion also fails. The check must be run as a set, and the
package must pass all of it before transmission.

### 6.7 What the example establishes

1. Dataset counts can reconcile while the package still fails, because files and tags are separate
   checks.
2. The technical rejection criteria are a small, checkable set that should be checked before
   transmission.
3. The inventory must be derived from specifications, not from the folder.
4. The reviewer simulation is a different test from validation, and both are necessary.
5. Fixing one failure at a time is more expensive than fixing the set.

### 6.8 The reviewer simulation in this package

Walk study 302's primary endpoint:

1. The clinical study report table 14.2.1 reports the primary response rate.
2. The table footnote names the ADRS dataset and the ANL01FL flag.
3. The define.xml method describes the confirmation rule and the denominator.
4. Recomputing from ADRS reproduces the table within rounding.
5. Tracing one responder to RS and to the CRF takes four minutes.

The walk succeeds, which is the good news. The package is reviewable even though it does not
validate. That combination, reviewable but not submittable, is the one the reconciliation is designed
to detect and fix.

### 6.9 What the delay costs

The direct cost of the extra cycle is the team's time for five days: regulatory operations,
programming, and QC. The indirect cost is the review delay, the rescheduling of the agency
interactions, and the credibility loss with the reviewer who sees a refused submission before seeing
the science.

For a first-in-class product in a competitive indication, the indirect cost can exceed the direct
cost by an order of magnitude. That asymmetry is the argument for the inventory and the simulation.

### 6.10 What would change the outcome

The package would pass if the missing ts.xpt were added, the four STF tags corrected, and the
duplicate resolved. The fixes are mechanical and take days.

The package would fail for a different reason if the define.xml methods were missing or the
denominators did not reconcile. Those failures are not fixed by assembly; they require the
programming team to re-derive, and they take weeks.

### 6.11 The metrics a submission team should track

| Metric | Definition | Target |
|---|---|---|
| Inventory completeness | Files present / files expected | 100% |
| Metadata coverage | Variables with origin and method / variables | 100% |
| Reject-level findings | Count at final validation | 0 |
| Reconciliation differences | Unexplained count differences | 0 |
| Reviewer simulation time | Minutes to reproduce the primary table | Under 60 |
| Lifecycle errors | Incorrect operations | 0 |
| Days from lock to transmission | Elapsed time | Per plan |

The metrics are leading indicators. A package with 100% inventory completeness and zero reject-level
findings two weeks before transmission is on track; one with unexplained differences is not, even if
the validation report looks clean.

### 6.12 The readiness decision

The package is not ready. The inventory is complete at the dataset level but not at the file level,
four files are untagged, one required file is missing, and one duplicate exists. The readiness
recommendation is "no-go with a defined remediation list", not "go with exceptions", because every
exception is a reject-level criterion.

The remediation list should name the file, the owner, the fix, and the validation step that will
confirm it. A list without an owner and a check is a wish.

The decision record should also state the earliest safe transmission date, which is the date on which
the full validation set passes, not the date on which the reported error is fixed.

The distinction matters when a programme is under schedule pressure: the reported error is a symptom,
and the full set is the test.

## 7. Code, query, or calculation

The script reconciles an expected file inventory against a submitted package and evaluates the
technical rejection criteria that can be checked from the manifest. It is deliberately simple: the
point is that the check is mechanical and can run before transmission.

```python
from collections import Counter

def build_expected():
    """Inventory derived from the programming specifications, not the folder."""
    rows = []
    for study, n_sdtm, n_adam in [("301", 22, 9), ("302", 24, 11), ("303", 21, 8)]:
        rows += [(study, "tabulations/sdtm", f"lb{i:02d}.xpt",
                  "data-tabulation-dataset-sdtm") for i in range(1, n_sdtm + 1)]
        rows += [(study, "analysis/adam", f"ad{i:02d}.xpt",
                  "analysis-dataset-adam") for i in range(1, n_adam + 1)]
        rows += [
            (study, "tabulations/sdtm", "ts.xpt", "data-tabulation-dataset-sdtm"),
            (study, "tabulations/sdtm", "define.xml", "data-tabulation-data-definition"),
            (study, "analysis/adam", "define.xml", "analysis-data-definition"),
            (study, "misc", "reviewer-guide.pdf", "reviewer-guide"),
        ]
    return rows

def apply_defects(rows):
    """Reproduce the package in Section 6: one missing file, four bad tags, one duplicate."""
    out = [r for r in rows if not (r[0] == "302" and r[2] == "ts.xpt")]
    bad = {("303", "lb03.xpt"), ("303", "lb07.xpt"), ("303", "lb12.xpt"), ("303", "ad02.xpt")}
    out = [(s, f, n, ("data-listing-data-definition" if (s, n) in bad else t))
           for s, f, n, t in out]
    out.append(("303", "analysis/adam", "ad02.xpt", "analysis-dataset-adam"))
    return out

VALID = {"data-tabulation-dataset-sdtm", "analysis-dataset-adam",
         "data-tabulation-data-definition", "analysis-data-definition", "reviewer-guide"}

def reconcile(expected, present):
    exp = {(s, f, n) for s, f, n, _ in expected}
    got = Counter((s, f, n) for s, f, n, _ in present)
    missing = sorted(exp - set(got))
    extra = sorted(set(got) - exp)
    dups = sorted(k for k, v in got.items() if v > 1)
    bad_tags = sorted((s, n) for s, f, n, t in present if t not in VALID)
    ts_ok = all(any(n == "ts.xpt" for s2, f2, n in got if s2 == s)
                for s in {e[0] for e in exp})
    return {
        "expected": len(expected), "present": len(present),
        "missing": missing, "extra": extra, "duplicates": dups,
        "bad_tags": bad_tags, "ts_present": ts_ok,
        "ready": not missing and not extra and not dups and not bad_tags and ts_ok,
    }

exp = build_expected()
res = reconcile(exp, apply_defects(exp))
for k, v in res.items():
    print(f"{k}: {v}")
```

Expected output (Python 3.12, executed):

```text
expected: 107
present: 107
missing: [('302', 'tabulations/sdtm', 'ts.xpt')]
extra: []
duplicates: [('303', 'analysis/adam', 'ad02.xpt')]
bad_tags: [('303', 'ad02.xpt'), ('303', 'lb03.xpt'), ('303', 'lb07.xpt'), ('303', 'lb12.xpt')]
ts_present: False
ready: False
```

Reading the output:

- The file count matches (107 expected, 107 present), which is exactly why a count-only check is
  insufficient. One file is missing and one has been added in its place.
- The missing file is the trial summary dataset for study 302, which is the most common technical
  rejection cause.
- Four files carry an invalid tag. The tag `data-listing-data-definition` is a real eCTD tag for a
  different document type, which is why copy-paste errors produce it.
- The duplicate is a dataset submitted twice as new, which the eCTD lifecycle cannot resolve on its
  own.

The script checks only the criteria that can be evaluated from a manifest. Criteria that depend on
file content, such as the consistency of the study identifier inside the TS dataset, require reading
the data and are checked by the agency's validator.

The inventory is the input that matters. A script that reconciles the folder against itself will
always pass, which is why `build_expected` is written from the specifications rather than from a
directory listing.

## 8. Data, standards, and regulatory context

### 8.1 The eCTD versions

| Version | Status | Use |
|---|---|---|
| eCTD v3.2.2 | Supported | Lifecycle submissions to applications established under v3.2.2 |
| eCTD v4.0 | Accepted for new applications since 16 September 2024 | New NDAs, BLAs, ANDAs, INDs, and master files |
| eCTD v4.0 with forward compatibility | Supported from 16 October 2026 | v4.0 submissions to applications established under v3.2.2 |

The FDA has stated that it will give advance notice before requiring v4.0 exclusively. As of the
access date, both versions are supported, and the choice depends on the application's history and the
sponsor's readiness.

The practical consequence is that a sponsor with a mixed portfolio needs to maintain both pipelines
during the transition. A programme that assumes v4.0 is mandatory, or that v3.2.2 is obsolete, will
misplan either the build or the lifecycle.

### 8.2 The study data technical rejection criteria

The FDA's technical rejection criteria are a small set of rules that cause a study data submission to
be refused. The criteria that most often fail are:

| Criterion | Requirement | Typical cause |
|---|---|---|
| 1734 | A trial summary dataset for each study | Omitted for one study |
| 1735 | Correct STF file tags for datasets and define.xml | Copy-paste or template error |
| 1736 | DM and define.xml for SDTM; ADSL and define.xml for ADaM | Misplaced or renamed file |
| 1737 | No more than one new file with the same name | Duplicate submission |
| 1738 | STF study ID matches the study ID in the TS dataset | Renamed study |
| 1789 | No file in a study folder without an STF entry | Legacy file added late |

The FDA began rejecting submissions on these criteria in September 2021. Published agency
presentations report an average study data rejection rate of about 17%, with criterion 1734
accounting for the majority of errors, and note that overall eCTD rejection remains below 2% of
submissions. The figures are agency-reported and should be treated as indicative rather than current.

### 8.3 The technical conformance guide

The Study Data Technical Conformance Guide describes the structure and content requirements for
study data, including the define.xml requirements, the folder structure, the file tags, and the
handling of split datasets and replacements. It is the reference for the assembly, and it changes
more often than the regulations. A programme should check the current version at submission planning.

### 8.4 Requirement, expectation, and convention

- **Requirement.** Electronic submission through the ESG is required for the submission types the
  agency specifies. The technical rejection criteria are enforced: a failure causes a refusal.
- **Regulatory expectation.** CDISC standards, define.xml, and the folder structure are expected
  through guidance and the Data Standards Catalog. A departure requires justification.
- **Convention.** The internal numbering of sequences, the naming of reviewer's guides, and the
  choice of validation tool are conventions. They should be consistent, but they are not requirements.

### 8.5 What the technical conformance guide changes most often

The guide is updated more frequently than the regulations, and the changes that matter most are the
folder structure, the file-tag definitions, and the handling of split and replacement files. A
programme that builds to a remembered version of the guide will discover the difference at
validation.

### 8.6 The common technical document and the regional modules

The common technical document is shared across regions, but Module 1 is regional and the study data
requirements differ at the margins. A US application and an EU application can reuse Modules 2-5 with
region-specific Module 1 content, but the dataset requirements may differ in the standards version
and the submission format.

A global programme should build the package for the strictest requirement and add the regional Module
1 content, rather than building twice. The savings are real, and the risk of divergence is lower.

### 8.7 What the validation does not check

Validation checks structure, presence, and metadata. It does not check whether the analysis is
correct, whether the population is the right one, or whether the table can be reproduced. Those are
review findings, and they are the subject of the reviewer simulation.

## 9. Industry practice and case evidence

### 9.1 How submissions are organised

Large sponsors run a regulatory operations function that owns the eCTD build, the gateway
submission, and the lifecycle. Study teams own the datasets and the metadata. The interface between
them is the inventory and the validation report, and most failures occur at that interface.

Three organisational patterns are visible:

- **Centralised build.** Regulatory operations builds every package from a standard template. Fewer
  assembly errors, slower turnaround.
- **Study-team build with central QC.** Teams assemble their own study folders, and a central team
  validates. Faster, but variable quality.
- **Vendor build.** An external partner assembles and validates. Efficient, but the sponsor must
  retain accountability and access to the artefacts.

The pattern does not determine the outcome; the discipline at the interface does.

### 9.5 Validation tools and their limits

| Tool type | What it checks | What it misses |
|---|---|---|
| eCTD validator | Backbone, file structure, lifecycle | Dataset content |
| Study data validator | Datasets, metadata, terminology, tags | Analysis correctness |
| Custom checks | Sponsor rules, cross-study consistency | Anything not coded |
| Reviewer simulation | Navigation and reproducibility | Scale |

A programme needs all four. The first two are standard; the third is where an organisation encodes
its own lessons; the fourth is the only check that approximates the reviewer.

### 9.6 What the reviewer sees that the team does not

The team sees the package it built. The reviewer sees the package as a navigation problem: a folder
tree, a set of documents, and a claim to verify. The two views diverge most in the middle of the
package, where the analysis datasets and the metadata meet.

A useful practice is to have someone outside the team perform the reviewer simulation and record
every point at which they had to guess. Each guess is a potential information request, and the fixes
are usually cheap: a clearer label, a method, a cross-reference.

### 9.7 The recovery plan

A submission plan should include a recovery scenario: who does what if the package is refused, how
long the fix takes, and what the schedule impact is. The plan is not pessimism; it is the reason the
team can transmit with confidence.

The recovery plan should identify the files most likely to fail, the owners, and the validation step
that confirms the fix. A plan written after the refusal is a scramble.

### 9.8 What a mature submission function looks like

| Capability | Evidence | Maturity signal |
|---|---|---|
| Inventory automation | Generated from specifications | No manual list |
| Continuous validation | Run at each dataset build | Findings trend down |
| Metadata review | Methods and origins checked | No "per SAP" methods |
| Reviewer simulation | Performed by an outsider | Recorded guesses |
| Lifecycle control | Operations reviewed | No duplicate versions |
| Defect learning | Root causes logged | Repeat errors decline |
| Recovery planning | Owner and time defined | Tested once |

The last row is the one that distinguishes a mature function: it has practised the failure, not only
planned for it.

### 9.2 What the rejection statistics show

The agency's published presentations on submission issues report that missing trial summary datasets
are the single largest cause of study data rejection, followed by file-tag errors and missing
define.xml files. The pattern is consistent with the criteria: the failures are assembly failures,
not scientific ones.

**Documented:** the criteria and the reported rejection rates. **Inference:** that most rejections
could be prevented by a pre-submission run of the agency's validator and a manifest reconciliation.
The inference is plausible because the criteria are machine-checkable, but the agency does not
publish the counterfactual.

### 9.3 Case: a submission refused twice

The worked example is fictional, but the pattern is common enough to be described in agency
presentations: a package is refused for a missing TS dataset, resubmitted, and refused again for a
tag error that was present in the first package but not fixed because the team fixed only the
reported error.

**Documented:** the criteria and the refusal mechanism. **Inference:** the internal cause, which is
usually a remediation process that fixes the reported finding rather than re-running the full check.

### 9.4 What is documented and what is inferred

| Claim | Status |
|---|---|
| eCTD v4.0 has been accepted for new applications since September 2024 | Documented by FDA |
| Forward compatibility begins in October 2026 | Documented in a Federal Register notice |
| Missing TS datasets are the most common rejection cause | Documented in agency presentations |
| About 17% of study data submissions are rejected | Agency-reported; indicative |
| Most rejections are preventable by pre-submission validation | Inference |
| A reviewer simulation reduces information requests | Inference; plausible but not measured |

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Missing TS dataset | Inventory not derived from specifications | Omission | Refusal (1734) | Inventory reconciliation |
| Wrong STF tag | Tags assigned by template | Copy-paste | Refusal (1735) | Tag validation |
| Missing define.xml | File misplaced | Assembly error | Refusal (1736) | Folder checks |
| Duplicate new file | Resubmission as new | Lifecycle error | Refusal (1737) | Operation review |
| Study ID mismatch | Study renamed | Inconsistent metadata | Refusal (1738) | ID check |
| Untagged file | Legacy file added | Omission | Refusal (1789) | Manifest check |
| Metadata describes old datasets | Late dataset change | Version control | Reviewer cannot reproduce | Version together |
| Denominator mismatch | Table and ADSL disagree | Population flag error | Information request | Reconcile denominators |
| Missing methods | Define-XML incomplete | Metadata treated as documentation | Information request | Method review |
| Reviewer's guide hides issues | Known issue at the end | Advocacy | Credibility loss | Put issues first |
| Gateway failure | No ESG test | Operational | Delay | Register and test early |
| Wrong lifecycle operation | Append used for replacement | Process | Duplicate versions | Operation review |

The most expensive red flag is a clean validation report combined with a package that cannot be
reviewed. The report creates confidence, and the gap is found by the reviewer.

### 10.1 Red flags by phase

| Phase | Red flag | What it usually means |
|---|---|---|
| Scope | Regions added late | Module 1 and data requirements unprepared |
| Inventory | Inventory built from the folder | Missing files undetectable |
| Validation | Findings triaged by severity only | Semantic issues ignored |
| Assembly | Manual tagging | Copy-paste errors |
| QC | No reviewer simulation | Navigation problems undiscovered |
| Transmission | No retry plan | Gateway failure costs a day |
| Lifecycle | Operations not reviewed | Duplicate versions |

A programme that reviews these seven before transmission has covered the assembly risk. The
scientific risk remains, and it is covered by the metadata and denominator checks.

### 10.2 The failure that is invisible until review

A package can pass every automated check and still fail because a table's denominator does not match
the population flag in ADSL. The failure is invisible to validation and obvious to a reviewer who
recomputes the table.

The defence is a denominator reconciliation: for every table, record the numerator, the denominator,
and the ADSL flag that defines the population. A table without this record is a claim without a
source.

### 10.3 The red flag that survives every check

A package can pass the inventory, the TRC, the metadata review, and the reviewer simulation and still
contain an error that matters: a dataset that is internally consistent but does not match the
clinical study report. The only defence is a reconciliation between the report's tables and the
datasets, table by table.

That reconciliation is the last check before transmission, and it is the one most often skipped
because the report is written by a different team. The skip is the reason information requests about
numbers persist, and it is invisible in every automated report.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The inventory is complete | Reconciliation depends on it | Missing specification | Cross-check with study reports |
| The folder reflects the inventory | Assembly is faithful | Manual copy errors | Compare manifest |
| The metadata matches the datasets | Review reproducibility | Late change | Version and checksum |
| The validator covers the critical rules | Validation is sufficient | Rules change | Check current criteria |
| The gateway is available | Transmission | Outage | Schedule with retry |
| The lifecycle operations are correct | The package is coherent | Human error | Review operations |
| The reviewer can navigate | Review efficiency | Poor structure | Simulation |
| The timeline allows a re-cycle | Contingency | Compressed plan | Build slack |
| The agency's criteria are stable | Planning | New criteria | Check before submission |
| The team retains the artefacts | Audit and re-submission | Vendor ownership | Contract terms |

The deepest limit is that validation and review are different tests, and passing one does not imply
passing the other. A package needs both a technical check and a reviewer simulation, and the second
cannot be automated.

A second limit is that the criteria and the guides change. The eCTD version transition, the update to
the technical conformance guide, and changes to the validation rules all alter what "ready" means. A
programme should check the current documents at planning and again before transmission.

A third limit is that the package is built by people under time pressure. The errors that cause
refusal are simple, and simple errors are most likely when the team is tired. The checklist and the
script exist to compensate for that, not to replace judgement.

A fourth limit is that the reviewer is not the only audience. The package is also read by payers,
partners, and internal teams after approval, and a package that is navigable only by its authors has
a longer cost than the review.

A sixth limit is that the package is judged in retrospect. A submission that was ready on the day it
was sent can be found deficient two years later, when the agency's expectations or the programme's own
data have changed.

A fifth limit is that the submission is a snapshot of a programme that keeps moving. The datasets are
versioned, the label changes, and the lifecycle plan accumulates. The package that was ready in March
is not the package that is submitted in June unless the version control holds.

## 12. Exercises


The exercises use a fictional application; where a real package is unavailable, the worked example's
structure can be reused.

**Exercise 1 — Module map (application, 30 minutes).**
Deliverable: a module map for a fictional application with two studies, showing where the reports,
datasets, and metadata belong.
Acceptance criteria: each item placed in a module and subfolder; study data in Module 5; no item
without a location.
Time: 30 minutes.

**Exercise 2 — Inventory (application, 35 minutes).**
Deliverable: a dataset inventory for a study with 20 SDTM datasets, 10 ADaM datasets, and two
define.xml files, including expected file count.
Acceptance criteria: derived from specifications; file names and tags included; count stated.
Time: 35 minutes.

**Exercise 3 — TRC assessment (application, 30 minutes).**
Deliverable: a pass/fail assessment of a supplied manifest against the six criteria in Section 8.2.
Acceptance criteria: each criterion assessed; failures classified; remediation listed.
Time: 30 minutes.

**Exercise 4 — Reconciliation (application, 30 minutes).**
Deliverable: a reconciliation of expected versus present files for a package with three deliberate
defects.
Acceptance criteria: all defects found; each linked to a criterion; a fix proposed.
Time: 30 minutes.

**Exercise 5 — Reviewer simulation (application, 40 minutes).**
Deliverable: a timed walkthrough of a supplied package from the primary table to the CRF, with the
time for each step.
Acceptance criteria: all steps attempted; gaps identified; a navigation improvement proposed.
Time: 40 minutes.

**Exercise 6 — Reviewer's guide (design, 35 minutes).**
Deliverable: a two-page reviewer's guide for a fictional study.
Acceptance criteria: follows the reviewer's path; known issues near the front; no restatement of the
table of contents.
Time: 35 minutes.

**Exercise 7 — Lifecycle plan (design, 30 minutes).**
Deliverable: a lifecycle plan for the first year after approval, with the sequence numbers, the
operations, and the content of each submission.
Acceptance criteria: operations correct; sequence numbering consistent; no append used for a
replacement.
Time: 30 minutes.

**Exercise 8 — Readiness review (critique, 45 minutes).**
Deliverable: a go/no-go recommendation for a package with a clean validation report, four untagged
files, and missing methods.
Acceptance criteria: at least six checks; the recommendation justified; conditions and owners stated.
Time: 45 minutes.

## 13. Self-check answers

**Exercise 1.** Module 1 holds the regional administrative documents; Module 2 the summaries; Module
3 the quality information; Module 4 the nonclinical reports and SEND datasets; Module 5 the clinical
study reports, with SDTM in the tabulations folder, ADaM in the analysis folder, define.xml in each,
and the reviewer's guide with the study report.

**Exercise 2.** Expected files: 20 SDTM datasets + 10 ADaM datasets + 2 define.xml + 1 ts.xpt + 1
reviewer's guide = 34, assuming the ts.xpt is counted separately from the 20. The inventory should
list each file with its folder and tag, not just the count.

**Exercise 3.** Criterion 1734 fails if any study lacks a ts.xpt; 1735 fails if any dataset or
define.xml has a tag outside the permitted set; 1736 fails if DM or ADSL is missing; 1737 fails if a
new file name appears twice in the same folder; 1738 requires the STF study ID to match the TS
dataset; 1789 fails if a file in a study folder has no STF entry. Each failure needs a file, an
owner, and a fix.

**Exercise 4.** A correct reconciliation reports the expected and present counts, lists the missing
and extra files, the duplicates, and the bad tags, and maps each to a criterion. The count matching
while files differ is the key observation.

**Exercise 5.** A correct walkthrough records the time for finding the table, locating the dataset,
reading the method, recomputing, and tracing to the CRF. A package that takes more than an hour has a
navigation problem, even if it validates.

**Exercise 6.** The guide should open with the study and the datasets, state the primary derivation,
define the populations, list the unusual derivations, and put the known issues near the front. It
should not reproduce the table of contents.

**Exercise 7.** The lifecycle plan should use new for the first submission, append for a new document
in an existing leaf, and replace for a corrected file. Sequence numbers should increase by one for
each submission, and the plan should state the content and the timing of each.

**Exercise 8 rubric.** Full marks require checks of the inventory, the TRC, the metadata methods, the
denominators, the reviewer simulation, and the lifecycle operations.**Exercise 8.** A correct answer separates the technical failures (untagged files) from the
scientific ones (missing methods) and notes that the validation report does not cover the second.**Exercise 8.** A correct answer separates the technical failures (untagged files) from the
scientific ones (missing methods) and notes that the validation report does not cover the second. The
recommendation should be a conditional no-go until both classes are resolved.

**Exercise 8 rubric.** Full marks require checks of the inventory, the TRC, the metadata methods, the
denominators, the reviewer simulation, and the lifecycle operations.**Exercise 8 rubric.** Full marks require checks of the inventory, the TRC, the metadata methods, the
denominators, the reviewer simulation, and the lifecycle operations. A recommendation that lists the
six checks and the evidence for each is stronger than one that asserts readiness. A conditional go is acceptable
only if the conditions are specific, owned, and dated. A go with missing methods is not acceptable.

## 14. Interview questions

**1. What is the difference between eCTD validation and study data validation?**
Strong: eCTD validation checks the submission structure and lifecycle; study data validation checks
datasets, metadata, and tags, and a failure prevents the review clock from starting. Weak: treats them
as one check.

**2. What causes most study data rejections?**
Strong: a missing trial summary dataset, followed by tag and define.xml errors; they are assembly
failures, not scientific ones. Weak: says "bad data".

**3. How do you build a dataset inventory?**
Strong: from the specifications and study reports, with file names, folders, and tags, then reconciled
against the folder and the metadata. Weak: lists the folder contents.

**4. Why can a package pass validation and still fail review?**
Strong: validation checks structure, not reproducibility; a missing method or a denominator mismatch
is a review finding. Weak: says reviewers are stricter.

**5. What does a reviewer do first?**
Strong: opens the primary table, locates the analysis dataset, reads the method, and tries to
reproduce it. Weak: reads the cover letter.

**6. What belongs in a reviewer's guide?**
Strong: the study, the datasets, the primary derivation, the populations, the unusual derivations,
and the known issues near the front. Weak: a table of contents.

**7. What is the difference between append and replace?**
Strong: append adds a document, replace supersedes an existing one; using append for a correction
creates a duplicate. Weak: uses them interchangeably.

**8. When would you use eCTD v4.0?**
Strong: for new applications since September 2024, and for existing applications once forward
compatibility is available from October 2026, subject to agency notice. Weak: says v4.0 is mandatory.

**9. How do you handle a validation failure after submission?**
Strong: fix the reported failure and re-run the full check, not only the reported item; rebuild,
revalidate, and resubmit. Weak: fixes only the reported error.

**10. What would you check before transmission?**
Strong: inventory reconciliation, TRC, metadata methods, denominators, reviewer simulation, lifecycle
operations, and gateway readiness. Weak: the validation report alone.**10. What would you check before transmission?**
Strong: inventory reconciliation, TRC, metadata methods, denominators, reviewer simulation, lifecycle
operations, and gateway readiness. Weak: the validation report alone.

**11. What is the most expensive error in a submission package?**
Strong: an error that passes validation and prevents reproduction, because it is found by the reviewer
and costs an information request; the assembly errors are cheaper because they are caught early.
Weak: says a missing file.**11. What is the most expensive error in a submission package?**
Strong: an error that passes validation and prevents reproduction, because it is found by the reviewer
and costs an information request; the assembly errors are cheaper because they are caught early.
Weak: says a missing file.

**12. Who should sign off on submission readiness?**
Strong: a named accountable person outside the build team, with the checks, exceptions, and recovery
plan recorded. Weak: the programmer who built the package.

## 15. Key terms

- **Append** — a lifecycle operation that adds a document.
- **Complete response letter** — the agency's decision that an application is not approvable as
  submitted.
- **Dataset inventory** — the list of files that should be in the package.
- **Define-XML** — machine-readable metadata for datasets.
- **eCTD** — electronic common technical document.
- **eCTD v3.2.2 / v4.0** — the two supported eCTD versions.
- **ESG** — Electronic Submissions Gateway.
- **File tag** — the eCTD attribute identifying a file's type.
- **Filing** — the agency's acceptance of an application for review.
- **Forward compatibility** — the ability to submit v4.0 to a v3.2.2 application.
- **Information request** — a reviewer's request for clarification or data.
- **Lifecycle** — the sequence of submissions to an application.
- **Module** — one of the five eCTD sections.
- **New** — a lifecycle operation for a first submission of a file.
- **Reconciliation** — comparison of inventory, folder, and metadata.
- **Replace** — a lifecycle operation that supersedes a file.
- **Review clock** — the regulatory timeline that starts at filing.
- **Reviewer's guide** — the sponsor's explanation of the datasets.
- **Sequence number** — the identifier for a submission in the lifecycle.
- **Study tagging file (STF)** — the XML file that links documents and datasets.
- **Technical rejection criteria (TRC)** — FDA rules that cause a study data refusal.
- **Trial summary (TS) domain** — study-level parameters, including versions.

## 16. Sources

- US Food and Drug Administration. *Electronic Common Technical Document (eCTD) v4.0*.
  https://www.fda.gov/drugs/electronic-regulatory-submission-and-review/electronic-common-technical-document-ectd-v40
  (accessed 25 October 2026).
- Federal Register. eCTD v4.0 Forward Compatibility. 16 October 2026 (FR Doc. 2026-21197).
  https://www.federalregister.gov/documents/2026/10/16/2026-21197 (accessed 25 October 2026).
- US Food and Drug Administration. *Technical Rejection Criteria for Study Data*.
  https://www.fda.gov/files/drugs/published/Technical-Rejection-Criteria-for-Study-Data.pdf
  (accessed 25 October 2026).
- US Food and Drug Administration. *Study Data Technical Conformance Guide*. Version 6.3, October
  2026. https://www.fda.gov/media/153632/download (accessed 25 October 2026).
- US Food and Drug Administration. *Submitting in eCTD: Most Common Submission Issues and FDA Plans
  for eCTD v4.0*. 2023. https://www.fda.gov/media/168958/download (accessed 25 October 2026).
- US Food and Drug Administration. *Study Data for Submission to CDER and CBER*.
  https://www.fda.gov/industry/study-data-standards-resources/study-data-submission-cder-and-cber
  (accessed 25 October 2026).
- US Food and Drug Administration. *Data Standards Catalog*. March 2025.
  https://www.fda.gov/regulatory-information/search-fda-guidance-documents/data-standards-catalog
  (accessed 25 October 2026).
- CDISC. *Define-XML v2.1.10*. https://www.cdisc.org/standards/data-exchange/define-xml/define-xml-v2-1-1-0
  (accessed 25 October 2026).

> **Note:** The application and file counts in Section 6 are fictional. The eCTD versions, rejection
> criteria, and regulatory dates are real and were current at the access date; the FDA Catalog and
> the Technical Conformance Guide should be checked at each submission.

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic: [`09_decentralized-trials.md`](09_decentralized-trials.md)
- Related: [`05_cdisc-and-trial-standards.md`](05_cdisc-and-trial-standards.md)
- Related: [`02_data-in-pharma.md`](02_data-in-pharma.md)
- Survey file: [`../../13_biopharma_clinical_development.md`](../../course/13_biopharma_clinical_development.md)
- Labs: [`../../labs/`](../../labs/)
