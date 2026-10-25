# CDISC and Trial Standards: From CRF to Submission Dataset

> Module 13 — Biopharma and Clinical Development
> Topic 05 of 10
> Estimated study time: 5 hours
> Prerequisites: Module 03 (data engineering); topics 02 and 04
> Version 2026.3

## 1. Why this topic exists

A regulatory submission is not a report. It is a set of datasets that a reviewer must be able to
navigate, re-analyse, and trace back to the source. The Clinical Data Interchange Standards
Consortium (CDISC) standards exist to make that possible: the Study Data Tabulation Model (SDTM)
defines how trial data are tabulated, the Analysis Data Model (ADaM) defines how analysis datasets
are structured and derived, Define-XML describes both in machine-readable metadata, and controlled
terminology fixes the permitted values.

The standards are often taught as a formatting exercise. They are not. A mapping decision changes
what a reviewer can see. If an adverse event's start date is imputed to a complete date without a
flag, the reviewer cannot tell a real date from a constructed one. If a response derivation uses the
wrong denominator, the labelled efficacy claim changes. If a supplemental qualifier is used where a
standard variable exists, the reviewer may never find the value. Formatting errors are rejected by
software; semantic errors are rejected by reviewers, later and more expensively.

The commercial stakes are direct. A submission that fails technical validation is returned, and the
review clock does not start. A submission that passes validation but contains an untraceable
derivation generates information requests that can add months. A submission whose datasets cannot
be re-analysed undermines the credibility of every claim in the clinical study report, because the
reviewer cannot verify any of them independently.

This topic teaches the three-layer model, the mapping and metadata work that connects the layers,
the conformance and reconciliation checks that catch errors before submission, and the decisions
that determine whether a dataset is submission-ready. It follows the data-domain topic and precedes
the safety and submission topics, both of which depend on the standards described here.

### 1.1 The failure pattern

1. A study completes with 1,860 adverse event records in the clinical database.
2. The SDTM conversion maps the verbatim term to the coded term using a dictionary version that is
   one release behind the one declared in the study metadata.
3. Forty-two records fail the controlled terminology check; the team suppresses the rule because
   "the terms are close enough".
4. The ADaM adverse event dataset derives treatment-emergent flags using a date comparison that
   treats partial dates as the first of the month.
5. The reviewer finds 18 events classified as post-treatment that the clinical study report
   describes as pre-treatment.
6. The submission passes technical validation and fails scientific review.
7. The information request asks for a re-derivation, and the review timeline slips by three months.

Steps 3 and 4 are the expensive ones. Both are visible in a reconciliation and neither is caught by
a formatting check alone.

### 1.2 Who is blocked when standards are treated as formatting

| Stakeholder | What they lose | Why they cannot fix it alone |
|---|---|---|
| Statistical programmer | Weeks of rework after a rejection | Receives a mapping decision made elsewhere |
| Data manager | Confidence in the source-to-dataset trace | Owns the CRF, not the tabulation |
| Biostatistician | A re-analysable analysis dataset | Receives derivations without methods |
| Regulatory affairs | Review timeline and credibility | Cannot answer a derivation question alone |
| Reviewer | The ability to verify a claim | Receives datasets, not the reasoning |
| Patient | Delay in access to a decided therapy | Has no visibility of the submission |

### 1.3 The industry question

> Can a reviewer who has never met the team reproduce this analysis from the submitted datasets and
> metadata alone?

The question is the test of submission readiness. It is stricter than "does the dataset validate",
because validation checks structure and the question checks meaning.

### 1.4 Three audiences for the same dataset

A submission dataset is read by three audiences with different needs, and a design that serves only
one will fail the others.

| Audience | What they need | What they do with it |
|---|---|---|
| Reviewer | Navigation, traceability, reproducible methods | Verify the claim in the report |
| Internal statistician | Correct derivations, population flags | Re-run the analysis after a plan change |
| Future programmer | A specification that can be re-implemented | Rebuild the dataset for a new submission or a pooled analysis |

The third audience is the one most often forgotten. A dataset that cannot be rebuilt after the
original programmer leaves is a single point of failure, and pooled analyses across studies depend on
the specification being portable.

## 2. Learning outcomes

By the end of this topic you can:

1. **Distinguish** SDTM, ADaM, Define-XML, and controlled terminology, and state what each
   constrains and what it leaves to the sponsor.
2. **Write** a mapping specification from case report form fields to SDTM variables, including
   origin, codelist, and derivation.
3. **Derive** a subject-level analysis dataset and a basic analysis dataset with traceable
   derivations.
4. **Apply** controlled terminology and dictionary versions correctly, and explain the consequence of
   a version mismatch.
5. **Run** conformance and reconciliation checks, classify findings by severity, and decide whether a
   dataset is submission-ready.
6. **Diagnose** a traceability failure between a table, an analysis dataset, a tabulation dataset, and
   a source field.
7. **Explain** the regulatory status of the standards in the US and EU, distinguishing requirement
   from convention.

## 3. The core mental model

### 3.1 The four-layer model

```text
 Layer 1: Capture            Layer 2: Tabulation        Layer 3: Analysis         Layer 4: Output
 ------------------          -------------------        ------------------        ---------------
 CRF / eSource      ---->    SDTM domains        ---->   ADaM datasets      ----> Tables, listings,
 (CDASH principles)          (DM, AE, LB, VS, EX,       (ADSL, ADAE, ADLB,        figures
                              RS, TU, TR, ...)            ADTTE, ADRS, ...)
      |                            |                          |                        |
      |  mapping spec              |  define.xml               |  define.xml            |  TLF shells
      |  + controlled terminology  |  + methods                |  + methods             |  + traceability
      v                            v                          v                        v
   Audit trail                 Reviewer navigates         Reviewer re-analyses     Reviewer verifies
```

Each layer answers a different question. Capture records what happened. Tabulation organises it in a
standard structure without deriving analysis variables. Analysis derives the variables the
pre-specified analysis needs. Output presents the result. A value that appears in a table should be
traceable down this chain to a source field, and the metadata at each layer should make the path
explicit.

### 3.2 What the standards fix and what they leave open

| Element | Fixed by the standard | Left to the sponsor |
|---|---|---|
| Domain structure | Yes: variable names, roles, order | Which domains to submit and how to model unusual data |
| Controlled terminology | Permitted values for many variables | Extension terms, with justification |
| Date format | ISO 8601 | How partial dates are represented and imputed |
| Analysis dataset structure | Subject-level and basic data structure principles | Which datasets to create and what to derive |
| Metadata | Define-XML structure | The quality and completeness of the content |
| Traceability | Expected links between layers | The method used to document them |

The right-hand column is where submissions fail. A dataset can be structurally perfect and
semantically opaque, and the standards do not prevent it.

### 3.3 The three questions a reviewer asks

```text
 1. Where did this number come from?         -> traceability to source and derivation
 2. What population is it computed in?       -> ADSL flags and analysis population variables
 3. Can I reproduce it?                      -> metadata, methods, and dataset structure
```

Every design decision in this topic should be evaluated against these three questions. A decision
that makes any of them harder requires a justification that is recorded in the metadata.

### 3.4 The cost curve of a mapping error

A mapping error found at CRF design costs a meeting. The same error found at database lock costs a
re-run of the conversion. Found at submission, it costs a re-submission and a review delay. Found by
the reviewer, it costs an information request and the credibility of the analysis. The cost is not
linear, and the earliest detection point is the mapping specification.

### 3.5 What the reviewer sees first

A reviewer opens a submission at the table, not at the dataset. The path is usually: the clinical
study report table, then the analysis dataset that produced it, then the metadata method, then, if
needed, the tabulation dataset and the source. The order matters for design: the output must be
navigable to the analysis dataset without a guide, and the analysis dataset must be navigable to the
tabulation through metadata alone.

A submission that requires the reviewer to read the reviewer's guide to find the right dataset has
already lost time. A submission that requires the reviewer to call the sponsor has lost credibility.

### 3.6 Reading a Define-XML file in five minutes

A reviewer or a new programmer can assess a Define-XML file quickly by checking five things:

1. **Dataset list** — does it match the files in the folder, with labels that describe content?
2. **Keys** — is each dataset's key stated and plausible for its structure?
3. **Origins** — are derived variables marked derived, and CRF variables marked CRF?
4. **Methods** — do the derived variables have methods with inputs and parameters?
5. **Codelists** — are external dictionaries named with versions, and are extensions documented?

A file that passes these five checks is likely to support review. A file that fails any of them will
generate questions, even if every dataset validates.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| CDISC | Clinical Data Interchange Standards Consortium, the standards body | Used as a synonym for SDTM | Organisation |
| SDTM | Study Data Tabulation Model, the submission tabulation standard | Treated as an analysis model | Standard |
| SDTMIG | SDTM Implementation Guide, the domain-by-domain specification | Confused with the model itself | Implementation guide |
| ADaM | Analysis Data Model, the analysis dataset standard | Treated as a copy of SDTM | Standard |
| ADaMIG | ADaM Implementation Guide | Confused with ADaM principles | Implementation guide |
| CDASH | Clinical Data Acquisition Standards Harmonisation, the capture standard | Treated as mandatory for CRFs | Standard |
| Define-XML | Machine-readable metadata describing datasets and variables | Treated as documentation only | XML standard |
| Controlled terminology (CT) | CDISC-published code lists for standard variables | Confused with MedDRA or WHO Drug | Code lists, versioned |
| Code list | A set of permitted values for a variable | Treated as free text guidance | List |
| Codelist extensible flag | Whether a code list may be extended | Ignored, leading to invalid values | Metadata attribute |
| Origin | The source of a variable value: CRF, derived, assigned, protocol | Recorded as "CRF" for derived values | Metadata attribute |
| Derivation | The rule that produces a value | Omitted or described vaguely | Method |
| SUPPQUAL | Supplemental qualifiers domain for non-standard variables | Used for variables that have standard equivalents | Domain |
| Domain | A collection of observations of the same type, e.g. AE | Used for any dataset | SDTM structure |
| Variable role | Identifier, topic, qualifier, timing, rule | Confused with variable type | Metadata attribute |
| ISO 8601 | Date and time format standard | Applied inconsistently to partial dates | Format |
| Partial date | A date with missing components | Imputed without a flag | Date |
| Imputation flag | A variable recording that a value was imputed | Omitted | Flag |
| Baseline | The last non-missing value before the first treatment | Defined differently across studies | Definition |
| Treatment-emergent | An event with onset after the first dose, per a stated rule | Assumed rather than defined | Definition |
| Analysis population | The set of subjects included in an analysis, with a flag | Confused with the randomised set | Flag |
| ADSL | Subject-level analysis dataset | Treated as optional | Dataset |
| BDS | Basic data structure, the ADaM structure for one record per parameter per timepoint | Confused with a wide dataset | Structure |
| OCCDS | Occurrence data structure for event and intervention data | Confused with BDS | Structure |
| Traceability | The documented path from output to source | Confused with a data flow diagram | Property |
| Conformance rule | A machine-checkable rule derived from the standard | Confused with a business rule | Rule |
| Reject-level finding | A validation finding that prevents submission acceptance | Treated as a warning | Severity |
| Reviewer's guide | The sponsor's document explaining the submission datasets | Confused with Define-XML | Document |
| Split dataset | A dataset divided because of size limits | Submitted with a second define.xml | File structure |
| Legacy data | Data from studies conducted before current standards | Assumed exempt | Data category |
| eCTD | Electronic common technical document, the submission format | Confused with the datasets | Format || Trial summary (TS) | A domain containing study-level parameters, including standards and dictionary versions | Left empty or populated with free text | Domain |
| Value-level metadata | Metadata that varies by parameter or record within a variable | Omitted, so the reviewer cannot interpret mixed variables | Metadata |
| Annotation | The CRF marking that links a field to an SDTM variable | Produced after database lock | Artefact |

### 4.1 The vocabulary of "standard"

Three different things are called a standard in this topic, and the difference matters:

- **A model** (SDTM, ADaM) defines the structure and the meaning of variables.
- **An implementation guide** (SDTMIG, ADaMIG) defines how the model is applied to specific domains
  and situations, and is versioned separately from the model.
- **A terminology** (controlled terminology, MedDRA, WHO Drug, LOINC) defines permitted values, and
  is versioned on its own schedule.

A submission declares all three versions. A programme that says "we used SDTM 3.4" has named an
implementation guide, not the model or the terminology, and the declaration is incomplete.

## 5. The mechanics, step by step

This procedure builds a submission-ready set of datasets for a study. It assumes the study has a
protocol, a statistical analysis plan, and a clinical database, and that the submission scope is
known.

### Step 1. Fix the submission scope and the standards versions

**Input:** the study, the intended submissions (US, EU, both), and the agency requirements.

**Output:** a standards declaration: SDTM model and IG versions, ADaM versions, Define-XML version,
controlled terminology package, and dictionary versions for adverse events and medications.

**Rationale:** The versions determine what is valid. The FDA Data Standards Catalog lists supported
and required versions and the dates from which they apply; a submission that uses a superseded
version may be rejected at technical validation. Declaring the versions at the start prevents a
re-conversion at the end.

**Branch point:** if the study started under an older standard, the programme must decide whether to
convert to the current version or to submit under the version in force at study start. The decision
depends on the agency's transition rules and should be documented.

### Step 2. Build the mapping specification

**Input:** the annotated CRF, the database structure, and the SDTMIG.

**Output:** a specification with one row per submitted variable: source field, target variable,
origin, codelist, derivation, and comments.

**Rationale:** The mapping specification is the contract between data management and statistical
programming. It is where the semantic decisions are made: which CRF field becomes which variable,
what happens to a field that has no standard home, and how a derived value is produced. A
specification that lists only target variables is a rename, not a mapping.

**Edge case:** data with no standard variable. The options are a supplemental qualifier, a
study-specific variable in a domain that permits one, or a new domain. Each has consequences for
reviewer navigation, and the choice should be recorded with a rationale.

### Step 3. Apply controlled terminology and dictionaries

**Input:** the mapping specification and the terminology packages.

**Output:** variables populated only with permitted values, with extensions justified and recorded.

**Rationale:** Controlled terminology makes values comparable across studies and machine-checkable.
An extension term is permitted only where the codelist is extensible, and it must be submitted with
a definition. A value that is "close enough" to a standard term is a validation finding and a
semantic risk.

**Edge case:** dictionary version changes mid-study. If adverse events were coded with one MedDRA
version and the submission declares another, the coded terms must be re-coded or the version
declared must match the data. Mixing versions within a domain is a defect.

### Step 4. Build the subject-level analysis dataset

**Input:** demographics, disposition, exposure, and population flags.

**Output:** ADSL with one row per subject, the treatment variables, the population flags, and the
key dates.

**Rationale:** ADSL is the spine of the analysis. Every other analysis dataset joins to it, and every
population count is defined by its flags. An error in ADSL propagates to every table. The flags must
match the statistical analysis plan's definitions exactly, including the order of precedence when a
subject meets more than one criterion.

**Edge case:** subjects randomised but never dosed. They are in the intention-to-treat population and
not in the safety population. ADSL must represent both, with flags, and the derivation must state
which population each flag represents.

### Step 5. Build the analysis datasets with traceable derivations

**Input:** SDTM domains, ADSL, and the analysis plan.

**Output:** ADaM datasets (for example ADAE, ADLB, ADTTE, ADRS) with derivation methods recorded.

**Rationale:** An analysis dataset is not a reformatted SDTM domain. It contains derived variables:
baseline values, change from baseline, analysis flags, censoring dates, and time-to-event variables.
Each derivation must be reproducible from the metadata and the input data. A derivation that exists
only in a programmer's code is not traceable.

### Step 6. Implement traceability

**Input:** the mapping specification and the analysis methods.

**Output:** a traceability matrix linking each output to its analysis dataset, tabulation dataset,
and source field.

**Rationale:** Traceability is what allows a reviewer to verify a number. In practice it is built
from three artefacts: the annotated CRF (source to SDTM), the Define-XML methods (SDTM to ADaM), and
the table shells with dataset and variable references (ADaM to output). A traceability matrix that
is assembled after the fact usually reveals gaps that cannot be closed without re-running the
analysis.

### Step 7. Write Define-XML

**Input:** the datasets and the mapping and derivation metadata.

**Output:** a Define-XML file for SDTM and one for ADaM, with dataset and variable metadata, code
lists, origins, methods, and comments.

**Rationale:** Define-XML is the machine-readable description that a reviewer's tools read. Missing
origin, undefined codelists, and vague methods are the most common content defects. The file must be
a valid XML document with the stylesheet in the same folder, and it must describe every submitted
dataset.

**Branch point:** one Define-XML per dataset type, not one per study. A study with SDTM, SEND, and
ADaM requires three files.

### Step 8. Run conformance checks and triage findings

**Input:** the datasets and Define-XML.

**Output:** a findings log with each finding classified as reject, warning, or informational, with a
disposition.

**Rationale:** Conformance tools check structure, terminology, and some cross-dataset rules. They do
not check whether the analysis is correct. The triage is therefore a professional judgement: a
terminology finding may be a false positive if the term is a justified extension, and a structural
warning may be a real defect if it hides a missing record.

**Branch point — finding triage:**

| Finding | Typical cause | Action | Can it be waived? |
|---|---|---|---|
| Reject-level structural error | Missing required variable, invalid structure | Fix the dataset | No |
| Controlled terminology mismatch | Wrong version, unjustified extension | Re-code or justify extension | Only with a documented extension |
| Missing origin in Define-XML | Metadata not completed | Complete the metadata | No |
| Cross-dataset referential error | Subject in a domain but not in DM | Investigate; fix the source | No |
| Duplicate record warning | Genuine duplicate or a legitimate repeat | Determine which | Yes, with justification |
| Length or format warning | Truncation, non-standard format | Fix if it affects meaning | Yes, if cosmetic |
| Business-rule warning | A rule the sponsor added | Review; may be a false positive | Yes, with rationale |

### Step 9. Reconcile counts across layers

**Input:** the clinical database, SDTM, ADaM, and the tables.

**Output:** a reconciliation showing subject and record counts at each layer, with every difference
explained.

**Rationale:** The counts should decrease in explainable ways: subjects excluded from a population,
records excluded as not applicable, records derived. An unexplained difference is a defect even when
both numbers are individually plausible. The reconciliation is the cheapest defence against a
reviewer's first question.

### Step 10. Handle the edge cases that break traceability

**Partial dates.** Represent partial dates as partial ISO 8601 strings, and record any imputation in
a flag variable. Do not write an imputed complete date without a flag.

**Unit conversion.** Convert results to the protocol unit and retain the original result and unit in
the appropriate variables. A conversion without the original value destroys traceability.

**Unscheduled visits.** Represent them with the visit variables and do not force them into a
scheduled visit number. The analysis may use them; the tabulation must not hide them.

**Supplemental qualifiers.** Use them only for variables with no standard home. A value placed in
SUPPQUAL when a standard variable exists is invisible to a reviewer who queries the standard
variable.

**Split datasets.** Split only when size requires it, place the parts in a split folder, and do not
submit a second Define-XML for the parts.

**Legacy data.** Data from older studies may not conform. Convert where feasible, and document the
limitations in the reviewer's guide rather than implying conformance.

### Step 11. Write the reviewer's guide

**Input:** the submission package and the known limitations.

**Output:** a concise guide describing the study, the datasets, the derivations that are not obvious,
and the known issues.

**Rationale:** The guide is the sponsor's opportunity to explain what the metadata cannot. It should
be short enough to be read and specific enough to answer the three reviewer questions. A guide that
restates the table of contents adds nothing.

### Step 12. Lock, version, and submit

**Input:** the final datasets and metadata.

**Output:** a versioned package with a manifest, checksums, and the submission-ready folder
structure.

**Rationale:** Submissions are re-submitted. A package without a version and a manifest cannot be
compared with its predecessor, and a reviewer cannot tell which file is current. The eCTD structure
and file tags are part of the submission, not an afterthought.

### 5.1 What a mapping specification row actually contains

A specification with only source and target columns cannot be reviewed. A usable row has eight
fields:

| Field | Example | Why it matters |
|---|---|---|
| Source field | AE form, field AETERM (verbatim) | Locates the source for traceability |
| Target variable | AE.AETERM | Defines the destination |
| Origin | CRF | Tells the reviewer the value was collected |
| Type and length | Character, 200 | Prevents truncation |
| Codelist | MedDRA preferred term (dictionary) | Defines permitted values |
| Derivation | None (direct) or the rule | Makes derived values reproducible |
| Comment | "Verbatim retained; coding performed centrally" | Explains non-obvious decisions |
| Status | Draft, approved, changed | Tracks the decision history |

The origin field is the one most often wrong. A value that is calculated (for example, an age at
consent) has origin "derived", not "CRF", even when all inputs came from the CRF. A reviewer who
sees "CRF" for a derived value will look for a source field that does not exist.

### 5.2 BDS and OCCDS: two structures, two purposes

ADaM analysis datasets use two principal structures, and choosing the wrong one makes the analysis
harder to review.

| Structure | One record per | Used for | Key variables |
|---|---|---|---|
| Basic data structure (BDS) | Subject, parameter, and timepoint | Continuous or categorical measurements analysed by parameter | PARAMCD, PARAM, AVAL, AVALC, BASE, CHG, ANL01FL |
| Occurrence data structure (OCCDS) | Subject and occurrence | Events and interventions, such as adverse events or prior medications | AEDECOD, ASTDT, TRTEMFL, ANL01FL |
| Subject-level | Subject | One row per subject with population flags and key dates | TRT01P, SAFFL, ITTFL, EOSDT |

A laboratory dataset is BDS because each analyte is a parameter with values over time. An adverse
event dataset is OCCDS because each event is an occurrence without a parameter structure. Forcing
events into BDS produces a wide, unreadable dataset; forcing labs into OCCDS loses the
parameter-time structure that the analysis needs.

### 5.3 The Define-XML content a reviewer actually uses

Define-XML contains more than a variable list. The parts that affect review are:

- **Dataset descriptions**, including the dataset label and the key variables.
- **Variable metadata**, including the data type, length, origin, and codelist.
- **Value-level metadata** for variables whose meaning changes by parameter or record.
- **Methods**, which describe derivations in text that a reviewer can read.
- **Comments**, which explain known issues and sponsor decisions.
- **External codelists**, with the dictionary name and version.

A Define-XML file that lists variables but omits methods and value-level metadata passes a
structural check and fails the reviewer's reproduction test. The file is not documentation of what
was done; it is the specification a reviewer uses to check what was done.

### 5.4 Conformance rules versus business rules

Conformance tools apply two kinds of rule, and confusing them wastes time.

- **Conformance rules** are derived from the standard: required variables, valid structures, valid
  terminology. A finding is a defect in the submission.
- **Business rules** are sponsor or tool conventions: a variable should be populated when another is
  populated, a date should fall in a range. A finding may be a real problem or a false positive.

A team that treats every business-rule warning as a defect will spend weeks chasing non-issues. A
team that treats conformance findings as warnings will be rejected. The triage in Step 8 depends on
knowing which kind of rule produced the finding.

### 5.5 Domain decisions that recur in every study

| Data | Standard home | Common alternative | Consequence of the alternative |
|---|---|---|---|
| Concomitant medication | CM | SUPPCM | Reviewer queries the wrong domain |
| Prior medication | CM with a prior flag | CM plus a separate dataset | Duplicate review |
| Adverse event of special interest | AE with a flag, or a separate domain | A custom domain | The flag is easier to analyse and review |
| Medical history | MH | CM | The event is misclassified as a medication |
| Protocol deviation | DV | A custom dataset | Not linked to the standard review |
| Death | DM plus AE with fatal outcome | A separate death dataset | Inconsistent death counts |
| Response assessment | RS | A custom endpoint dataset | Loss of the standard structure |
| Pharmacokinetic sample | PC | LB | The analyst must know the custom structure |

The pattern is that the standard home exists for almost every common data type. A custom domain is
justified when the data genuinely have no standard structure, not when the standard structure is
inconvenient. Each custom choice adds a document that the reviewer must read and a query that the
sponsor must answer.

### 5.6 The reviewer's guide in one page

The guide should answer, in this order: what the study tested; which datasets were submitted and
what each contains; how the primary endpoint was derived; which populations were used and how they
were defined; which derivations are unusual; and which known issues affect interpretation. Anything
else is optional.

A guide that spends its first page on the study design and its last paragraph on a known data issue
has buried the information the reviewer needs. The known issues belong at the front, because a
reviewer who discovers them independently will assume they were concealed.

### 5.7 Version control of datasets and metadata

A submission is a versioned object. Three rules prevent most version confusion:

1. **Version the datasets and the metadata together.** A Define-XML file that describes a previous
   dataset version is worse than no metadata, because it appears authoritative and is wrong.
2. **Record the date and reason for every change.** A re-derivation after a plan clarification is
   legitimate; an unexplained difference between two submissions is not.
3. **Use the eCTD replace operator, not append.** A replacement submission must replace the dataset
   and its definition file as a set.

The most common version failure is a late change to one analysis dataset that is not reflected in
Define-XML or in the reviewer's guide. The package then contains two descriptions of the same
analysis, and the reviewer must choose which to believe.

A version manifest, listing each file with its version, date, and checksum, takes an hour to produce
and prevents this class of error entirely.

The manifest also makes a re-submission comparable: the reviewer can see exactly which files changed
and which did not.

## 6. Worked example

> **Note:** The study and its counts are fictional. The standards, versions, and regulatory dates
> cited are real and are referenced in Section 16.

### 6.1 The study and the endpoint

A Phase III study randomised 420 patients to an investigational antibody or placebo. The primary
endpoint is objective response rate (ORR) by blinded independent central review using RECIST 1.1,
defined in the statistical analysis plan as confirmed response: a complete or partial response
confirmed at a subsequent assessment at least four weeks later.

The programme must produce SDTM, ADaM, Define-XML, and the clinical study report table for ORR.

### 6.2 The layers and their counts

| Layer | Dataset | Records | Subjects | What it contains |
|---|---|---:|---:|---|
| Capture | Response assessment form | 1,680 | 407 | One form per subject per assessment |
| Tabulation | SDTM RS | 1,680 | 407 | Response and lesion assessments |
| Analysis | ADaM ADRS | 1,680 | 407 | Derived response, confirmation flag, analysis flag |
| Analysis | ADaM ADSL | 420 | 420 | Subject-level population flags |
| Output | ORR table | 1 row | 420 | Confirmed ORR by arm, ITT |

The counts reconcile in one direction: 407 subjects have at least one response assessment; 13
randomised subjects have none, because they withdrew before the first assessment. The table's
denominator is the intention-to-treat population of 420, not the 407 with data.

The 13 subjects without an assessment are not missing at random in the analysis sense: they withdrew
early, which may be related to treatment. The plan handles them by counting them as non-responders in
the ITT analysis, and the ADSL flag identifies them so that a sensitivity analysis in the evaluable
population can be run. Both analyses belong in the submission, because the difference between them is
the information the reviewer needs about the missing-data assumption.

### 6.3 The response derivation

```text
Any response (unconfirmed)  = 172 subjects with at least one CR or PR
Confirmed response          = 160 subjects with CR or PR confirmed at a subsequent assessment
Unconfirmed only            = 12 subjects
Confirmed ORR (ITT)         = 160 / 420 = 38.10%
Unconfirmed ORR (ITT)       = 172 / 420 = 40.95%
Confirmed ORR (evaluable)   = 160 / 407 = 39.31%
```

The table reports 38.1% (160/420). The difference between the confirmed and unconfirmed rates is 12
patients, or 2.9 percentage points, which is larger than the difference the trial was powered to
detect in a subgroup analysis.

### 6.4 The reconciliation

| Check | Expected | Observed | Difference | Explanation |
|---|---:|---:|---:|---|
| ADSL subjects | 420 | 420 | 0 | None |
| Subjects with any RS record | 407 | 407 | 0 | 13 withdrew before assessment |
| RS records | 1,680 | 1,680 | 0 | Four assessments per subject |
| ADRS records | 1,680 | 1,680 | 0 | One ADRS record per RS record |
| Confirmed responders | 160 | 160 | 0 | Flag matches the plan |
| Unconfirmed responders | 172 | 172 | 0 | Consistent with the derivation |
| Subjects in RS not in ADSL | 0 | 0 | 0 | Referential integrity holds |

The counts reconcile, which is necessary but not sufficient. The next check is semantic: does the
derivation implement the plan's definition? The confirmation flag is derived from the next
assessment's date and response. A subject with a response at week 8 and a confirmation at week 16 is
confirmed; a subject with a response at week 8 and no subsequent assessment is not.

### 6.5 Two diagnostics that reveal a broken analysis

**Diagnostic 1 — the denominator does not match the population named in the table.** The table says
"ITT" and reports 160/420. If the programmer had used the 407 subjects with data, the rate would be
39.3%, and the table would be internally consistent but mislabelled. The diagnostic is to check that
every denominator in every table equals the count of the population flag in ADSL that the table
claims to use.

**Diagnostic 2 — the confirmation flag is missing from Define-XML.** If the flag exists in ADRS but
its derivation is not described, a reviewer cannot tell whether confirmation was applied. The
diagnostic is to attempt to reproduce the table from the metadata alone: if the derivation cannot be
reconstructed, the metadata is incomplete even when the data are correct.

A third check is the source-to-SDTM trace for one responder: locate the CRF page, the RS record, the
ADRS record, and the table cell. If any step cannot be demonstrated in under five minutes, the
traceability is not submission-ready.

### 6.6 Vary one assumption: the confirmation definition

The plan defines confirmation as a response at a subsequent assessment at least four weeks later. A
programmer implements confirmation as "any subsequent assessment with a response", without the
four-week interval.

Under the looser rule, 9 additional subjects are classified as confirmed, and the ORR becomes
169/420 = 40.24% rather than 38.10%. The absolute difference is 2.1 percentage points, and the
direction favours the treatment in a study whose margin is narrow.

The decision consequence: the stricter definition matches the plan and the regulatory precedent for
RECIST-based endpoints. The looser definition is not a coding choice; it is a different endpoint. The
correction is to implement the interval, re-derive, re-run the table, and record the change in the
analysis dataset version.

### 6.7 What the example establishes

1. Record counts reconciling across layers is necessary but not sufficient for a valid analysis.
2. The denominator is a semantic claim, not an arithmetic convenience.
3. A derivation that is not in the metadata does not exist for review purposes.
4. A small definitional difference changes the headline efficacy number.
5. Traceability is demonstrated by walking one record from source to output, not by producing a
   diagram.

The example also shows why the standards cannot be delegated to a conversion vendor without
oversight. The vendor can implement a definition; it cannot decide which definition is correct. That
decision belongs to the sponsor and must be recorded before the conversion starts.

### 6.8 The Define-XML method that makes the derivation reviewable

The confirmation flag in ADRS needs a method that a reviewer can implement. A defensible method
reads:

```text
ANL01FL = "Y" for the record used in the primary analysis. For each subject, the confirmed
response is the earliest assessment with response CR or PR that has a subsequent assessment
with response CR or PR at least 28 days later. The record for that assessment is flagged.
If no assessment meets the confirmation rule, no record is flagged and the subject is counted
as a non-responder in the ITT analysis.
```

A method that reads "derived per SAP" is not a method. It defers to a document the reviewer may not
have, and it does not state the interval, the handling of missing assessments, or the non-responder
rule. The three elements that make a method reviewable are the input records, the rule including its
parameters, and the handling of the cases where the rule does not apply.

### 6.9 Reading the reconciliation as a reviewer

A reviewer does not see the team's reconciliation table. The reviewer performs the equivalent checks
in minutes:

- Count the subjects in ADSL and compare with the randomised number in the protocol.
- Count the subjects with a primary endpoint record and compare with the table's denominator.
- Pick one responder and walk from the table to the ADRS record to the RS record to the CRF.
- Read the Define-XML method for the response flag and ask whether it can be implemented.

A submission that survives these four checks is in good shape. A submission that fails the third or
fourth will generate an information request regardless of how clean the conformance report is.

### 6.10 What would change the conclusion

The ORR conclusion is sensitive to three definitions, and each should be stated in the table's
footnote:

| Definition | Value used | Alternative | Effect on ORR |
|---|---|---|---|
| Confirmation interval | At least 28 days | Any subsequent response | 40.2% instead of 38.1% |
| Denominator | ITT, 420 | Evaluable, 407 | 39.3% instead of 38.1% |
| Assessment rule | Blinded independent central review | Investigator assessment | Requires a separate analysis |

The first two are derivations from the same data; the third is a different endpoint. A reviewer will
ask which was pre-specified, and the answer must be in the plan and in the metadata.

A programme that cannot answer these three questions before submission is not ready, regardless of
its conformance report.

The discipline is to put the sensitivity in the table footnote rather than in a separate document.
The reviewer reads the table; the footnote travels with it.

## 7. Code, query, or calculation

The SQL below derives a confirmed response flag and the ORR denominators from a small extract of
subject-level and response data. It runs on SQLite 3.39 or later and uses window functions to look
ahead to the next assessment. The logic mirrors the definition in Section 6: a response counts only
if a subsequent assessment at least 28 days later is also a response.

```sql
CREATE TABLE dm (
  subjid TEXT PRIMARY KEY,
  arm    TEXT,
  ittfl  TEXT,
  evlfl  TEXT
);

CREATE TABLE rs (
  subjid   TEXT,
  visitnum INTEGER,
  rsdt     TEXT,
  rsorres  TEXT,
  PRIMARY KEY (subjid, visitnum)
);

-- extract: 12 subjects, 2 with no post-baseline assessment
INSERT INTO dm VALUES
 ('S01','A','Y','Y'), ('S02','A','Y','Y'), ('S03','B','Y','Y'),
 ('S04','B','Y','Y'), ('S05','A','Y','Y'), ('S06','B','Y','Y'),
 ('S07','A','Y','Y'), ('S08','B','Y','Y'), ('S09','A','Y','Y'),
 ('S10','B','Y','Y'), ('S11','A','Y','N'), ('S12','B','Y','N');

INSERT INTO rs VALUES
 ('S01',3,'2025-02-01','CR'), ('S01',5,'2025-03-29','CR'),
 ('S02',3,'2025-02-03','PR'), ('S02',5,'2025-03-31','PR'),
 ('S03',3,'2025-02-05','PR'), ('S03',5,'2025-04-02','SD'),
 ('S04',3,'2025-02-07','PR'),
 ('S05',3,'2025-02-09','CR'), ('S05',6,'2025-05-04','PR'),
 ('S06',3,'2025-02-11','SD'), ('S06',5,'2025-04-08','SD'),
 ('S07',4,'2025-03-01','PR'), ('S07',5,'2025-03-29','PR'),
 ('S08',3,'2025-02-15','PR'), ('S08',4,'2025-03-08','PR'),
 ('S09',3,'2025-02-17','SD'),
 ('S10',3,'2025-02-19','PR'), ('S10',6,'2025-05-14','PR');

-- referential integrity: every RS subject must exist in DM
SELECT COUNT(*) AS orphan_records
FROM rs WHERE subjid NOT IN (SELECT subjid FROM dm);

WITH r AS (
  SELECT subjid, visitnum, rsdt, rsorres,
         LEAD(rsdt)    OVER (PARTITION BY subjid ORDER BY visitnum) AS next_dt,
         LEAD(rsorres) OVER (PARTITION BY subjid ORDER BY visitnum) AS next_resp
  FROM rs
),
conf AS (
  SELECT subjid,
         MAX(CASE WHEN rsorres IN ('CR','PR')
                   AND next_resp IN ('CR','PR')
                   AND julianday(next_dt) - julianday(rsdt) >= 28
                  THEN 1 ELSE 0 END) AS confirmed,
         MAX(CASE WHEN rsorres IN ('CR','PR') THEN 1 ELSE 0 END) AS anyresp
  FROM r
  GROUP BY subjid
)
SELECT d.arm,
       SUM(COALESCE(c.anyresp, 0))  AS any_resp,
       SUM(COALESCE(c.confirmed, 0)) AS confirmed,
       COUNT(*)                     AS itt
FROM dm d
LEFT JOIN conf c ON d.subjid = c.subjid
WHERE d.ittfl = 'Y'
GROUP BY d.arm
UNION ALL
SELECT 'ALL',
       SUM(COALESCE(c.anyresp, 0)),
       SUM(COALESCE(c.confirmed, 0)),
       COUNT(*)
FROM dm d
LEFT JOIN conf c ON d.subjid = c.subjid
WHERE d.ittfl = 'Y';
```

Executed output:

```text
Referential integrity (RS subjects not in DM):
0

Response counts by arm (ITT)
  A: any 4/6 = 66.7%  confirmed 4/6 = 66.7%
  B: any 4/6 = 66.7%  confirmed 1/6 = 16.7%
  ALL: any 8/12 = 66.7%  confirmed 5/12 = 41.7%

Evaluable denominator
  evaluable subjects: 10
```

Reading the output:

- The referential check returns zero orphan records, which is necessary but does not prove that every
  subject has an assessment. Two subjects (S11, S12) have no RS record and are counted as
  non-responders in the ITT denominator.
- The confirmed rate (41.7%) is lower than the unconfirmed rate (66.7%) because three subjects had a
  single response without confirmation, and one had a confirmation interval shorter than 28 days.
- The arm-level counts illustrate why the denominator must be stated: arm B's confirmed rate uses all
  six randomised subjects, including one with no assessment.

The query assumes that visit dates are stored as ISO text, that response values use the RECIST
categories, and that the confirmation window is measured in calendar days. Each assumption belongs in
the Define-XML method, not only in the code.

A useful extension is a reconciliation query that compares the count of ADRS records with the count
of RS records by subject. If the counts differ, the difference should be explained by a derivation
that creates or removes records, such as a baseline record or an analysis record for a missing
assessment.

A second extension checks the denominator directly against the population flag:

```sql
SELECT 'ITT' AS population, COUNT(*) AS n FROM dm WHERE ittfl = 'Y'
UNION ALL
SELECT 'Evaluable', COUNT(*) FROM dm WHERE evlfl = 'Y';
```

```text
ITT        12
Evaluable  10
```

The two-subject difference is the 13-subject difference in the full study: subjects without an
assessment are in ITT and not in the evaluable population. A table that reports an ORR without
naming which population it used cannot be reconciled against either number.

## 8. Data, standards, and regulatory context

### 8.1 The standards and their versions

The CDISC standards are versioned separately, and the version matters for submission acceptance.

| Standard | Current release at the time of writing | Regulatory status |
|---|---|---|
| SDTM model | Version 2.0, paired with SDTMIG 3.4 | FDA requirement began 15 March 2025 for NDAs, ANDAs, certain BLAs, and certain INDs |
| SDTMIG | Version 3.4, published 29 November 2021 | Required as above; version 3.2 requirement ended 13 December 2023 |
| ADaMIG | Version 1.3, published 29 November 2021 | FDA requirement began 15 March 2024 |
| CDASHIG | Version 2.3, published 28 September 2023 | Current implementation guide; capture standard |
| Define-XML | Version 2.1.10, published 29 October 2025 | Supported; version 2.0 or later expected by FDA |
| Controlled terminology | Quarterly packages | The package used must be declared |
| MedDRA | Twice-yearly releases (March and September) | The version used must be declared |
| WHO Drug | Quarterly and annual releases | The version used must be declared |

The version landscape changes. A programme should consult the FDA Data Standards Catalog and the
Study Data Technical Conformance Guide at submission planning, not at the end of the study, because a
late change of version can require a full re-conversion.

### 8.2 The FDA submission requirements

The Study Data Technical Conformance Guide sets out what a submission must contain and how it must
be structured. The parts that affect dataset design include:

- A separate Define-XML file for each dataset type (SDTM, SEND, ADaM), submitted as a functioning XML
  file with its stylesheet.
- Complete metadata: datasets, variables, possible values and codelists, controlled terminology,
  origin for each variable, and detail on derived variables.
- The version of any external dictionary stated in Define-XML and in the trial summary domain.
- Dataset labels that describe the content, not the file name.
- Correct folder placement and eCTD file tags, with replacement rather than append on re-submission.

These are technical requirements for acceptance, not suggestions. A submission that fails them can be
returned without scientific review.

### 8.3 Requirement, expectation, and convention

- **Requirement.** The FDA's data standards requirements, with their effective dates, are binding for
  the submission types they cover. The eCTD format is required for electronic submissions.
- **Regulatory expectation.** Using CDISC standards is expressed through the Data Standards Catalog
  and guidance; the statute names the submission of electronic data, not CDISC specifically. A
  sponsor that departs from the standards must expect to justify the departure and may be asked to
  resubmit.
- **Convention.** Using one Define-XML style, naming datasets in a particular way, or adopting a
  specific validation tool are choices. They may be good practice; they are not requirements.

The distinction matters when a vendor claims its product is "FDA compliant". The FDA does not certify
software. It accepts or rejects a submission, and the sponsor is responsible for the content.

### 8.4 What the standards do not cover

- **Whether the analysis is correct.** ADaM structures a derivation; it does not validate the
  statistical method.
- **Whether the endpoint is appropriate.** A perfectly standard dataset can encode a flawed endpoint.
- **Whether the data are complete.** Conformance checks structure, not ascertainment.
- **Whether the metadata are truthful.** A method can be described accurately or plausibly. Only
  reproduction distinguishes the two.

### 8.5 The trial summary domain and version declaration

The trial summary domain (TS) carries study-level parameters, including the standards versions and
the dictionary versions used. It is a small domain with disproportionate review value: it is where a
reviewer checks that the submission's declaration matches the files.

Three parameters are commonly missing or inconsistent:

- **The SDTMIG version.** It must match the version used to build the datasets, not the version the
  sponsor intended to use.
- **The MedDRA version.** It must match the coded terms in the adverse event domain.
- **The WHO Drug version.** It must match the coded terms in the concomitant medication domain.

A mismatch between TS and the datasets is a finding that a reviewer can detect in minutes and that
undermines confidence in the rest of the package.

### 8.6 Controlled terminology in practice

Controlled terminology is released in packages, and each package contains additions, deprecations,
and changes. Three practices prevent most terminology defects:

- **Lock the package at database lock**, not at submission. The datasets and the metadata must declare
  the same package.
- **Treat extensions as data, not as free text.** An extension term needs a sponsor definition, a
  rationale, and a place in the metadata.
- **Check deprecated terms.** A term that was valid at study start may be deprecated by submission,
  and a deprecation is not the same as an invalidation; the distinction matters for the finding.

The most common terminology defect is not an invalid term but a valid term used with the wrong
codelist, for example a severity value placed in a causality variable. Tools catch the invalid term;
reviewers catch the wrong variable.

## 9. Industry practice and case evidence

### 9.1 How submissions are produced

Most large sponsors produce SDTM through a combination of internal programming and vendor
conversion, and ADaM through internal statistical programming. The division varies: some sponsors
outsource both, some keep ADaM in-house because it encodes the analysis. The common pattern is a
standards group that owns the mapping conventions, a programming group that implements them per
study, and a validation group that runs conformance checks.

Three organisational failure modes recur:

- **Standards without enforcement.** A central standards team publishes conventions that studies
  deviate from, and the deviations are discovered at submission.
- **Validation without interpretation.** A team runs a conformance tool, fixes reject-level findings,
  and ignores warnings that encode real semantic problems.
- **Conversion without ownership.** A vendor converts the data, the sponsor does not review the
  mapping, and the semantic decisions are made by whoever wrote the conversion code.

These are inferences from observable job structures and public descriptions of submission
operations, not findings from a controlled study.

### 9.2 The validation tool landscape

Conformance tools apply CDISC conformance rules and sponsor business rules to datasets and
Define-XML. They are necessary and not sufficient. A clean validation report means the datasets are
structurally acceptable; it does not mean the analysis is reproducible.

The distinction is visible in the worked example: the response counts reconcile and the validation
report would be clean, but a reviewer who cannot find the confirmation method in Define-XML cannot
reproduce the table. Validation tools do not read the statistical analysis plan.

Nor do they read the protocol. A dataset can conform to the standard while encoding an eligibility
criterion incorrectly, a visit window that the protocol did not specify, or a population definition
that differs from the plan. Those are review findings, not validation findings, and the only defence
is a specification that was reviewed by the people who wrote the protocol and the plan.

### 9.3 Public evidence on submission quality

Regulatory agencies publish some information about submission deficiencies. The FDA's technical
rejection criteria for electronic submissions are public, and they describe the categories that cause
a submission to be refused at the gate: invalid format, missing datasets, missing Define-XML, and
non-conformant structure. These are documented requirements, not estimates of frequency.

The frequency of semantic deficiencies is not publicly measured. Information requests are not
published systematically, and sponsors do not disclose them. Claims that "most submissions receive
information requests about datasets" should be treated as inference unless a specific source is
cited.

### 9.4 Case: a version mismatch

A programme submitted SDTMIG 3.2 datasets after the FDA requirement for 3.4 had taken effect. The
submission was technically valid under 3.2 but did not meet the current requirement. The sponsor was
asked to resubmit under the required version, and the review clock restarted.

**Documented:** the FDA's transition dates and the requirement that submissions use the supported
version. **Inference:** the internal cause, which is usually a study that started before the
transition and a submission plan that did not revisit the version. The remedy is procedural: the
standards declaration in Step 1 should be checked against the Catalog at submission planning.

### 9.6 Outsourcing and oversight

When conversion is outsourced, the sponsor retains accountability for the content. Three controls
make the arrangement work:

- **A mapping specification approved by the sponsor**, not written solely by the vendor.
- **Access to the conversion code or a reproducible specification**, so the sponsor can re-derive.
- **A joint reconciliation before delivery**, with the sponsor signing off counts and derivations.

An arrangement in which the vendor owns the mapping, the code, and the validation report leaves the
sponsor unable to answer a reviewer's question about a derivation. The contract should make the
artefacts sponsor property, not vendor intellectual property.

### 9.7 The cost of a late conversion

A conversion performed after database lock, rather than during the study, costs more for three
reasons: the mapping decisions are made without the people who designed the CRF; the data cannot be
corrected at source; and the reconciliation is performed against a database that is already closed.

Programmes that run SDTM conversion prospectively, at each data cut, find mapping problems while
they are still cheap to fix. Programmes that convert once, at the end, find them when every fix
requires a change to a locked database.

The trade-off is real: prospective conversion costs programming time throughout the study. The
decision should be made on the expected number and cost of late fixes, not on the conversion budget
alone. A study with a novel endpoint, a custom domain, or a first-time vendor is the case
for prospective conversion; a straightforward study with an established pipeline is the case against.

### 9.5 What is documented and what is inferred

| Claim | Status |
|---|---|
| SDTMIG 3.4 and SDTM 2.0 became an FDA requirement on 15 March 2025 | Documented in the Federal Register notice |
| ADaMIG 1.3 became an FDA requirement on 15 March 2024 | Documented in the Federal Register notice |
| Define-XML must include origin and derivation metadata | Documented in the FDA Technical Conformance Guide |
| Most submissions pass technical validation | Not established publicly; rejection criteria exist but frequencies are not published |
| Semantic deficiencies cause more review delay than structural ones | Inference; plausible but not measured |
| Central standards teams reduce submission defects | Inference; consistent with practice but not tested |

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Wrong standards version | Declaration predates the current Catalog | Study started before a transition | Technical rejection | Check the Catalog at submission planning |
| Mapping without origin | Define-XML origin is blank or uniformly "CRF" | Metadata completed after conversion | Reviewer cannot trace values | Origin in the mapping specification |
| Derivation absent from metadata | Method says "per SAP" | Metadata treated as documentation | Table cannot be reproduced | Write methods with inputs and parameters |
| Controlled terminology mismatch | Terms fail validation | Dictionary or CT version drift | Re-coding and re-validation | Declare and lock versions |
| Partial dates imputed silently | Complete dates with no flag | Imputation applied at conversion | Misclassified treatment-emergent events | Imputation flag variable |
| Unit conversion loses original | Only converted values retained | Conversion at source | Traceability lost | Retain original result and unit |
| Custom domain for standard data | A domain not in the IG | Convenience | Reviewer queries; analysis harder | Use the standard home |
| SUPPQUAL overuse | Standard variable exists but is empty | Value placed in SUPPQUAL | Value invisible to standard queries | Map to the standard variable |
| Denominator mismatch | Table denominator differs from ADSL flag | Population flag misapplied | Wrong efficacy claim | Reconcile every denominator |
| Split dataset with second Define-XML | Two definition files for one dataset | Misunderstanding of the rule | Technical finding | One Define-XML per dataset type |
| Late dataset change | Dataset date after Define-XML date | Version control failure | Inconsistent package | Version datasets and metadata together |
| Reviewer's guide buries issues | Known issue in the last paragraph | Advocacy | Loss of credibility | Put known issues first |
| Validation treated as sufficient | Clean report, no reproduction test | Tool cannot check semantics | Information request | Walk one record from source to output |

The most expensive red flag is a clean validation report combined with a metadata file that cannot
support reproduction. The report creates false confidence, and the gap is discovered by the reviewer
rather than by the team.

A second red flag is a TS domain that declares a standards version different from the one in the
dataset labels. The mismatch is easy to check and hard to explain, and it invites a broader review of
everything the sponsor has declared.

### 10.1 Fix order when there is no time to fix everything

Late in a submission, findings compete for the same days. The order that protects the submission is:

| Priority | Finding class | Reason |
|---|---|---|
| 1 | Reject-level structural and terminology findings | The submission will not be accepted |
| 2 | Denominator and population flag errors | The headline results are wrong |
| 3 | Missing methods for derived variables | The analysis cannot be reproduced |
| 4 | Traceability gaps in the primary endpoint | The central claim is unverifiable |
| 5 | Cosmetic label and length warnings | Low review impact |

A team that fixes cosmetic warnings while a denominator is wrong has optimised the validation report
rather than the submission.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The CRF captures the variables the standard requires | Mapping starts from the CRF | Missing fields | Gap analysis against the IG |
| One SDTM variable represents one CRF concept | Mapping is one-to-one | Composite fields, derivations | Review the mapping specification |
| The dictionary version is stable during the study | Coding consistency | Mid-study migration | Version audit |
| ADaM derivations implement the SAP | Analysis validity | Ambiguous SAP definitions | Independent re-derivation |
| Define-XML describes the submitted datasets | Review reproducibility | Late changes | Compare file dates and checksums |
| Conformance tools cover the important rules | Validation is feasible | Semantic rules not encoded | Manual review and reproduction test |
| The reviewer has the same dictionary versions | Terms resolve | Version mismatch | Declare versions in TS and Define-XML |
| Legacy data can be converted | Pooled analyses | Missing source variables | Document limitations |
| The submission package is immutable after sending | Version control | Late corrections | Formal change control |
| The analysis population flags match the plan | Denominators are correct | Flag precedence errors | Reconcile counts and definitions || The TS domain reflects the datasets | Reviewers check declarations | Stale metadata | Version audit before submission |
| The reviewer's guide describes the actual package | Guidance is current | Late changes | Compare guide with manifest |
| The conversion code is retained | Re-derivation is possible | Vendor retains it | Code escrow or sponsor ownership |

The central limit is that the standards guarantee structure, not meaning. A submission can be fully
conformant and still be unreproducible, and the only test that detects this is attempting
reproduction from the metadata alone.

A second limit is that the standards evolve. A package that is conformant today may be out of date at
submission, and the version declaration has to be revisited rather than assumed.

## 12. Exercises

The exercises assume access to a small study extract or a public sample submission; where neither is
available, use the worked example's structure.

**Exercise 1 — Layer identification (recall, 20 minutes).**
Deliverable: for ten variables, state whether each belongs to SDTM, ADaM, or both, and give the
reason.
Acceptance criteria: at least three variables correctly placed in both layers with different
representations; no variable assigned without a reason.
Time: 20 minutes.

**Exercise 2 — Mapping specification (application, 45 minutes).**
Deliverable: a mapping specification for six CRF fields covering demographics, an adverse event, and
a laboratory result, with source, target, origin, codelist, and derivation.
Acceptance criteria: origin correct for derived fields; codelists named with versions; no field left
without a destination or an explicit decision.
Time: 45 minutes.

**Exercise 3 — ADaM derivation (application, 40 minutes).**
Deliverable: a specification for a time-to-event dataset, including the event indicator, censoring
rule, and analysis flag.
Acceptance criteria: censoring rule stated precisely; event and censoring mutually exclusive; method
written so a second programmer could implement it.
Time: 40 minutes.

**Exercise 4 — Denominator reconciliation (application, 30 minutes).**
Deliverable: a reconciliation of ITT, safety, and evaluable counts for a study with 420 randomised,
413 dosed, 407 assessed, and 13 withdrawn before assessment.
Acceptance criteria: each population defined; every difference explained; the table denominator
matched to the correct flag.
Time: 30 minutes.

**Exercise 5 — Conformance triage (critique, 35 minutes).**
Deliverable: a triage table for eight validation findings, classifying each as reject, warning, or
false positive, with an action.
Acceptance criteria: each classification justified; at least one false positive identified; the
action owner named.
Time: 35 minutes.

**Exercise 6 — Define-XML method (design, 35 minutes).**
Deliverable: a method description for a confirmed response flag and for a treatment-emergent adverse
event flag.
Acceptance criteria: inputs, parameters, and edge cases stated; no reference to "per SAP" without
the rule; reproducible by a second programmer.
Time: 35 minutes.

**Exercise 7 — Traceability walk (application, 40 minutes).**
Deliverable: a step-by-step trace of one table cell back to a CRF field, naming each artefact and the
person responsible for it.
Acceptance criteria: all four layers represented; the metadata artefact at each step named; any gap
identified with a remediation.
Time: 40 minutes.

**Exercise 8 — Submission readiness review (critique, 50 minutes).**
Deliverable: a readiness checklist and a go/no-go recommendation for a package with a clean
validation report, missing methods, and a late dataset change.
Acceptance criteria: at least six checks; the go/no-go justified; conditions stated for a conditional
go.
Time: 50 minutes.

## 13. Self-check answers

**Exercise 1.** Demographics age is SDTM (DM.AGE) and ADaM (ADSL.AGE, with the analysis-age
derivation); an adverse event term is SDTM (AE.AEDECOD) and ADaM (ADAE.AEDECOD, with a treatment-
emergent flag added); a baseline laboratory value is SDTM (LB.LBBLFL) and ADaM (ADLB.BASE, with
change from baseline). Variables that exist in both layers often have different derivations, and the
ADaM version must be traceable to the SDTM version.

**Exercise 2.** Example: date of birth maps to BRTHDTC with origin CRF and ISO 8601 format; age maps
to AGE with origin derived and a method stating the reference date; adverse event term maps to
AETERM (verbatim) and AEDECOD (coded, origin derived, MedDRA version); severity maps to AESEV with
the severity codelist; a laboratory result maps to LBORRES with LBORRESU and a conversion to
LBSTRESN with origin derived.

**Exercise 3.** Event indicator = 1 at the date of progression or death, whichever is first;
otherwise 0 at the last adequate assessment date. Censoring rule: subjects without an event are
censored at the last assessment with a valid response; subjects who start a new therapy are censored
at the last assessment before the new therapy. The method must state the handling of missing
assessments and the analysis flag.

**Exercise 4.** ITT 420; safety 413 (7 randomised but never dosed); evaluable 407 (13 withdrawn
before assessment). Each difference has a different cause and a different flag: ITTFL, SAFFL, and an
evaluable flag. The table's denominator must match the flag named in the table's population label.

**Exercise 5.** A required-variable finding is a reject. A controlled terminology mismatch is a
warning that becomes a reject if the term is not a justified extension. A duplicate-record warning
may be a false positive if the records represent two distinct events. A length warning is cosmetic
unless it truncates a meaningful value. Each action needs an owner and a date.

**Exercise 6.** The response method should state: the input assessments, the response categories, the
confirmation interval, the look-ahead rule, and the non-responder rule. The treatment-emergent method
should state the first-dose date, the onset date rule including partial dates, and the handling of
events present before treatment.

**Exercise 7.** The trace runs from the table cell to the ADRS record (dataset and variable), to the
RS record (subject, visit, response), to the CRF page (form and field), with the Define-XML method and
the annotated CRF as the metadata at each step. A gap at any step is a remediation item before
submission.

**Exercise 8 rubric.** Full marks require checks of versions, Define-XML completeness, methods,
denominators, traceability, and version control. A conditional go is acceptable if the conditions are
specific and dated. A go with missing methods is not acceptable, because the package cannot be
reproduced. A conditional go should name the owner and the date for each condition, and should state
which conditions would convert it to a no-go.

## 14. Interview questions

**1. What is the difference between SDTM and ADaM?**
Strong: SDTM is the submission tabulation, structured for review; ADaM is the analysis dataset, with
derived variables and traceability to SDTM. Weak: says ADaM is "cleaner SDTM".

**2. What does Define-XML contain that a data dictionary does not?**
Strong: origins, codelists, methods, value-level metadata, and comments, in a machine-readable form
that reviewer tools consume. Weak: says it is a list of variables.

**3. Why is origin important?**
Strong: it tells the reviewer whether a value was collected or derived, which determines how it can
be verified. Weak: says it is a required field.

**4. How do you handle a partial date?**
Strong: represent it as partial ISO 8601, record any imputation in a flag, and state the rule in the
method. Weak: imputes to the first of the month without a flag.

**5. What is the difference between a reject-level finding and a warning?**
Strong: a reject prevents acceptance; a warning may be a defect or a false positive and requires
judgement. Weak: treats both as errors to be suppressed.

**6. How would you reconcile SDTM and ADaM counts?**
Strong: compare subjects and records by domain, explain every difference by a derivation or an
exclusion, and check the denominators against ADSL flags. Weak: compares totals only.

**7. What would you do if the dictionary version changed mid-study?**
Strong: re-code to one version or declare the version actually used, document the migration, and
check the effect on term-level analyses. Weak: mixes versions without declaring it.

**8. What makes a submission dataset "reviewer-ready"?**
Strong: a reviewer can navigate it, reproduce the analysis from the metadata, and trace a value to
source. Weak: it passes validation.

**9. How do the FDA and EMA requirements differ?**
Strong: both expect CDISC standards, but the versions, timing, and submission format differ; the
programme should plan for the stricter requirement. Weak: assumes a single global package is always
acceptable.

**10. What is the most common mistake you have seen in submissions?**
Strong: a semantic gap, such as a missing method or a denominator mismatch, that validation cannot
detect. Weak: a formatting error.

## 15. Key terms

- **ADaM** — Analysis Data Model; CDISC standard for analysis datasets.
- **ADSL** — subject-level analysis dataset.
- **BDS** — basic data structure; one record per subject, parameter, and timepoint.
- **CDASH** — Clinical Data Acquisition Standards Harmonisation.
- **CDISC** — Clinical Data Interchange Standards Consortium.
- **Code list** — a set of permitted values for a variable.
- **Conformance rule** — a machine-checkable rule derived from a standard.
- **Controlled terminology** — CDISC-published code lists.
- **Define-XML** — machine-readable metadata for submission datasets.
- **Derivation** — the rule that produces a value.
- **Domain** — an SDTM collection of observations of one type.
- **eCTD** — electronic common technical document.
- **Imputation flag** — a variable recording that a value was imputed.
- **ISO 8601** — date and time format standard.
- **OCCDS** — occurrence data structure for event data.
- **Origin** — the source of a value: CRF, derived, assigned, or protocol.
- **Partial date** — a date with missing components.
- **Reviewer's guide** — the sponsor's document explaining the submission datasets.
- **SDTM** — Study Data Tabulation Model.
- **SDTMIG** — SDTM Implementation Guide.
- **SUPPQUAL** — supplemental qualifiers domain.
- **Traceability** — the documented path from output to source.
- **Treatment-emergent** — an event with onset after the first dose, per a stated rule.

## 16. Sources

- US Food and Drug Administration. *Data Standards Catalog*. March 2025.
  https://www.fda.gov/regulatory-information/search-fda-guidance-documents/data-standards-catalog
  (accessed 25 October 2026).
- US Food and Drug Administration. *Study Data Technical Conformance Guide*. Version 6.3, October
  2026. https://www.fda.gov/media/153632/download (accessed 25 October 2026).
- Federal Register. Data Standards; Support and Requirement Begins for CDISC Version 2.0 of SDTM,
  Version 3.4 of SDTMIG, and Version 1.0 of SENDIG-Genetox. 88 FR 86336, 13 December 2023.
  https://www.federalregister.gov/documents/2023/12/13/2023-27310/data-standards-support-and-requirement-begins-for-the-clinical-data-interchange-standards-consortium
  (accessed 25 October 2026).
- Federal Register. Data Standards; Requirement Begins for CDISC Versions 1.2 and 1.3 of the Analysis
  Data Model Implementation Guide. 87 FR 42729, 18 July 2022.
  https://www.federalregister.gov/documents/2022/07/18/2022-15248/data-standards-requirement-begins-for-the-clinical-data-interchange-standards-consortium-versions-12
  (accessed 25 October 2026).
- CDISC. *SDTMIG v3.4*. https://www.cdisc.org/standards/foundational/sdtmig (accessed 25 October 2026).
- CDISC. *ADaMIG v1.3*. https://www.cdisc.org/standards/foundational/adam/adamig-v1-3
  (accessed 25 October 2026).
- CDISC. *CDASHIG v2.3*. https://www.cdisc.org/standards/foundational/cdash/cdashig-v2-3
  (accessed 25 October 2026).
- CDISC. *Define-XML v2.1.10*. https://www.cdisc.org/standards/data-exchange/define-xml/define-xml-v2-1-1-0
  (accessed 25 October 2026).
- ICH. *M11 Clinical Electronic Structured Harmonised Protocol*.
  https://www.ich.org/page/multidisciplinary-guidelines (accessed 25 October 2026).
- US Food and Drug Administration. *Study Data for Submission to CDER and CBER*.
  https://www.fda.gov/industry/study-data-standards-resources/study-data-submission-cder-and-cber
  (accessed 25 October 2026).

> **Note:** The study and counts in Section 6 are fictional. The standards versions and regulatory
> dates in Section 8 are real and were current at the access date; the FDA Catalog should be checked
> at each submission.

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic: [`04_feasibility-and-recruitment.md`](04_feasibility-and-recruitment.md)
- Next topic: [`06_safety-and-pharmacovigilance.md`](06_safety-and-pharmacovigilance.md)
- Related: [`10_submission-data.md`](10_submission-data.md)
- Related: [`02_data-in-pharma.md`](02_data-in-pharma.md)
- Survey file: [`../../13_biopharma_clinical_development.md`](../../course/13_biopharma_clinical_development.md)
- Labs: [`../../labs/`](../../labs/)
