# QMS and Design Controls

> Module 14 — Medical Devices, SaMD, and Regulation
> Topic 05 of 10
> Estimated study time: 4 hours
> Prerequisites: 01, 03, 04
> Version 2026.3

## 1. Why this topic exists

A team can hold a well-calibrated model, a clean validation report, and a persuasive submission
narrative and still be unable to file. The blocker is rarely the science. It is that the company
cannot show, on paper, that the device it intends to ship is the device it tested, that each
requirement came from a clinical need, and that each requirement was verified and the resulting
device validated in the hands of the intended user. That showing is the job of a quality
management system (QMS) and, inside it, design controls.

Design controls are the part of the QMS where product evidence is manufactured. They are a set
of procedures and records that connect a user need to a design input, a design input to a design
output, an output to a verification test, and the assembled device to a validation in the
intended use environment. The connection is the deliverable. A model that performs well but
cannot be placed on that chain is, from a regulatory point of view, an artifact with no address.

The legal basis changed on 2 February 2026. The US Food and Drug Administration (FDA) published
the Quality Management System Regulation (QMSR) final rule on 2 February 2024 at 89 FR 7496. The
rule takes effect on 2 February 2026, incorporates the International Organization for
Standardization (ISO) 13485:2016 standard by reference, and restructures 21 CFR Part 820 of the
US Code of Federal Regulations (CFR). The design-control section that a generation of device
engineers memorized, 21 CFR 820.30, is superseded by ISO 13485 Clause 7.3, operating through the
applicability provisions of 21 CFR 820.10. The substance is recognizably the same. The citation,
the terminology, the audit questions, and the additional record requirements are not.

Three people are blocked when this is done badly. The data scientist cannot answer "where is the
requirement for this threshold" and therefore cannot defend a model change. The quality lead
receives a design history file six weeks before a submission and finds that a third of the
verification reports describe a build that no longer exists. The founder learns that a product
which works in a demo cannot be sold because the design validation was a retrospective
experiment on a curated dataset rather than a study of the device in use. Each of these is a
schedule problem before it is a compliance problem.

The cost is asymmetric. Prevention is a few hours per requirement per change cycle. Remediation
is an archaeology project: reconstructing intent from commit messages, Jupyter notebooks, slide
decks, and the memory of an engineer who has left. A hold before a 510(k) costs weeks. A
nonconformity during a pre-approval inspection for a premarket approval (PMA) application can
cost a quarter. A recall caused by an unverified change costs the remediation plus the market.

Data scientists are in scope because the modern design record is partly computational. Design
inputs include performance thresholds and subgroup floors. Design outputs include model
artifacts, preprocessing code, and inference configuration. Verification includes unit tests,
integration tests, and statistical test plans. Validation includes reader studies and workflow
studies. Change control includes retraining, recalibration, and threshold tuning. Software of
unknown provenance (SOUP) includes every open-source library in the inference path. None of this
fits in a drawing, and all of it must fit in the file.

This topic covers the US QMSR and ISO 13485 design controls, the record set they require, the
verification and validation distinction, design transfer, change control, supplier and SOUP
control, and the computer software assurance expectations that apply to production software. It
touches the European Union's Medical Device Regulation (MDR) Annex IX at a working level. It is
not an implementation guide for a full ISO 13485 certificate and it does not replace a quality
professional.

What good looks like is unglamorous: a design plan that names its gates, an input list where
every row is testable, a traceability matrix that closes, a validation protocol written before
the study, a change record that a stranger can follow, and a file index that a reviewer can
navigate in ten minutes. Teams that have this can change a model in a week. Teams that do not
spend a quarter proving what they already believe.

## 2. Learning outcomes

By the end of this topic, you can:

1. Explain what the QMSR changed on 2 February 2026 and locate design-control requirements in
   ISO 13485 Clause 7.3 rather than 21 CFR 820.30.
2. Write a design and development plan with review gates, deliverables, and acceptance criteria
   for a software-only device.
3. Convert user needs into design inputs that are testable, unambiguous, and traceable.
4. Distinguish verification from validation and specify the evidence each requires for an
   artificial intelligence (AI) enabled device.
5. Build a design history file (DHF) index and score its completeness against stated gates.
6. Apply change control to a model, threshold, or dependency change and decide whether
   re-verification, re-validation, or a new submission is required.
7. Classify suppliers, off-the-shelf (OTS) software, and SOUP and specify the controls each
   requires.
8. Audit a design record and write findings that separate documentation gaps from substantive
   gaps.

The outcomes map to exercises and self-check items as follows.

| Outcome | Exercise | Self-check |
|---|---|---|
| 1 | Exercise 1, Exercise 8 | Questions 1, 2 |
| 2 | Exercise 2 | Question 3 |
| 3 | Exercise 2, Exercise 3 | Question 4 |
| 4 | Exercise 3 | Question 5 |
| 5 | Exercise 4 | Question 6 |
| 6 | Exercise 5 | Question 7 |
| 7 | Exercise 6 | Question 8 |
| 8 | Exercise 7 | Questions 9, 10 |

## 3. The core mental model

Design controls are a container inside a container. The QMS defines how the organization
governs quality: documentation, management review, resources, product realization, measurement,
and improvement. Design controls sit inside product realization. They govern how one product's
requirements become evidence. The design record sits inside design controls. It is the only part
a reviewer sees.

```text
+-------------------------------------------------------------------------+
| Quality management system: ISO 13485:2016, incorporated by 21 CFR 820    |
|                                                                         |
|  Clause 4  documentation and records       Clause 7  product realization |
|  Clause 5  management responsibility       Clause 8  measurement,        |
|  Clause 6  resources and competence                   analysis, improve  |
|                                                                         |
|   +-----------------------------------------------------------------+   |
|   | 7.3 Design and development                                      |   |
|   |                                                                 |   |
|   |   user needs ---> design inputs ---> design outputs             |   |
|   |        |                |                   |                   |   |
|   |        |            verification <----------+                   |   |
|   |        |                |                                       |   |
|   |        +------------> validation (intended use, real users)     |   |
|   |                         |                                       |   |
|   |                    design transfer ---> production and release  |   |
|   |                                                                 |   |
|   |   ISO 14971 risk management crosses every arrow                 |   |
|   |   design reviews sit on every arrow                             |   |
|   +-----------------------------------------------------------------+   |
|              |                                        ^                 |
|              v                                        |                 |
|    design record (DHF or design and development file)  CAPA, complaints |
+-------------------------------------------------------------------------+
```

Read the diagram from the outside in. The outer box is audited when a regulator or notified body
assesses the organization. The middle box is audited when a reviewer assesses the product. The
inner record is what the reviewer reads. A gap in the outer box does not excuse a gap in the
inner one, and a perfect inner record does not save an organization whose corrective and
preventive action (CAPA) system does not work.

Six principles make the model usable.

**The QMS is the container, not the product.** Certification to ISO 13485 is evidence that a
quality system exists and is maintained. It is not evidence that a particular device was designed
under control. Auditors see this distinction immediately: a certificate with a scope that does
not cover the device, or a design record that postdates the certificate, is a finding waiting to
happen.

**Design control is a correspondence discipline.** Every user need maps to at least one design
input. Every design input maps to at least one design output. Every design input maps to at least
one verification. Every user need maps to validation evidence. The mapping is not decoration; it
is countable, and coverage is a legitimate management metric. What coverage does not measure is
whether the requirement was the right one. A matrix that closes perfectly around a bad
requirement is a well-organized failure.

**Iteration is allowed; the record must describe the final configuration.** The V model is a
convenience, not a waterfall mandate. Teams discover requirements while building. What matters is
that the released configuration is the one that was verified and validated, and that earlier
iterations are either superseded with a change record or excluded from the release.

**Risk management is orthogonal.** ISO 14971 runs beside the V, not after it. A hazard analysis
produces risk control measures. Those measures become design inputs, which then need outputs,
verification, and validation. A risk file that cites controls not present in the requirements is
a disconnected document, and reviewers treat it as one.

**The record is the product.** If an activity is not recorded, it did not happen for audit
purposes. A design review that occurred but left no minutes, attendees, date, and action log is
indistinguishable from a review that never occurred, and the reviewer will treat it as the
weaker of the two possibilities.

**Change is where the model breaks.** Software changes cheaply, which is the point and the
danger. A threshold, a dependency, a container image, or a cloud region can change the device
without changing a line of application code. Design control exists to make those changes visible
before they reach a patient.

A design record behaves like a small state machine. Each artifact moves through draft, reviewed,
approved, baselined, verified, validated, transferred, released, and, eventually, changed. A
change returns the affected artifacts to reviewed, not to draft, because the history must remain
intact. The state machine is the reason a version number alone is insufficient: a document can be
approved at revision C and still be unverified for the build in production.

The practical test is three questions. Show me the requirement. Show me the test that covers it.
Show me who approved the result and when. If a team can answer those three questions for every
safety-related requirement, it has a functioning design control process, regardless of how the
documents are formatted.

## 4. Definitions and vocabulary

The table defines the terms used later. The "common misuse" column records errors that show up
in real submissions and audits. IEC stands for the International Electrotechnical Commission and
ISO for the International Organization for Standardization.

| Term | Definition | Common misuse | Source |
|---|---|---|---|
| Quality management system (QMS) | The organizational structure, responsibilities, procedures, processes, and resources used to manage quality across the device lifecycle | Treating a certificate as proof that a specific device was controlled | ISO 13485:2016 |
| Quality Management System Regulation (QMSR) | The FDA rule at 21 CFR Part 820, effective 2 February 2026, incorporating ISO 13485:2016 by reference | Treating it as a cosmetic rewrite that requires no document changes | 89 FR 7496 |
| Quality System Regulation (QSR) | The pre-2026 version of 21 CFR Part 820, including 820.30 design controls | Citing 820.30 after the effective date without mapping to Clause 7.3 | 61 FR 52602 |
| Design controls | The procedures and records linking user needs, design inputs, outputs, review, verification, validation, transfer, and change | Treating design controls as a document template rather than a process | ISO 13485 Clause 7.3 |
| Design and development plan | A controlled plan naming stages, review points, deliverables, responsibilities, and acceptance criteria | Writing the plan after the design is finished | ISO 13485 7.3.2 |
| User need | A statement of what the user requires in clinical or workflow terms, without prescribing a solution | Writing a user need as a feature or an algorithm | ISO 13485 7.2.1, 7.3.3 |
| Design input | A measurable requirement that design output must satisfy, derived from user needs and applicable requirements | Writing an input that cannot fail a test | ISO 13485 7.3.3 |
| Design output | Specifications, software, labeling, and other design results, with acceptance criteria | Omitting the acceptance criteria that make verification possible | ISO 13485 7.3.4 |
| Design review | A documented, systematic examination of the design against requirements at a defined stage | Holding a status meeting and calling it a review | ISO 13485 7.3.5 |
| Verification | Confirmation by objective evidence that design output meets design input | Calling a unit test suite a clinical validation | ISO 13485 7.3.6 |
| Validation | Confirmation by objective evidence that the device meets user needs and intended use under specified conditions | Validating on a curated retrospective dataset and calling it use-environment evidence | ISO 13485 7.3.7 |
| Design transfer | Translation of design output into production specifications and processes that can be reproduced | Assuming a model that runs in a notebook will run identically in production | ISO 13485 7.3.8 |
| Design change | A change to baselined design output, requiring review, verification, validation, and approval before implementation | Implementing a threshold change as a configuration edit | ISO 13485 7.3.9 |
| Design history file (DHF) | The record set demonstrating that the design was developed in accordance with the approved plan | Treating the DHF as a folder assembled before submission | Former 21 CFR 820.30(j) |
| Design and development file | The ISO 13485 record set for design and development, including inputs, outputs, reviews, verification, validation, and changes | Assuming it is a different requirement from the DHF rather than a renamed one | ISO 13485 7.3.10 |
| Device master record (DMR) | The production specification set: drawings, specifications, software, labeling, and process documents | Confusing production specifications with design evidence | US device terminology |
| Device history record (DHR) | Records showing that a specific production unit was made in accordance with the DMR | Filing release notes in the DHF instead of the DHR | US device terminology |
| Document control | Control of approval, review, distribution, revision, and obsolescence of documents | Leaving superseded procedures available for use | ISO 13485 4.2.4 |
| Record control | Control of identification, legibility, retrieval, retention, and protection of records | Retaining records for less than the device lifetime | ISO 13485 4.2.5 |
| Corrective and preventive action (CAPA) | Systematic investigation of nonconformities and potential nonconformities and the actions taken | Closing a CAPA at the document edit without checking effectiveness | ISO 13485 8.5.2, 8.5.3 |
| Nonconforming product | Product that does not meet requirements, controlled to prevent unintended use or delivery | Reworking without a documented concession or approval | ISO 13485 8.3 |
| Supplier control | Evaluation, selection, monitoring, and re-evaluation of suppliers, with defined purchasing information | Treating a cloud provider as infrastructure rather than a supplier | ISO 13485 7.4 |
| Off-the-shelf (OTS) software | Commercially available software used without modification, typically already validated by its vendor | Assuming commercial availability removes the need for evaluation | FDA OTS software guidance |
| Software of unknown provenance (SOUP) | Software not developed for the purpose it is used, with insufficient evidence of adequate testing | Omitting transitive dependencies from the SOUP inventory | IEC 62304 |
| Software item | Any identifiable part of a software system, including SOUP and system software | Tracking only first-level dependencies | IEC 62304 |
| Risk management file | The records produced by the ISO 14971 process, including hazards, estimates, controls, and residual risk | Keeping risk controls out of the requirements set | ISO 14971:2019 |
| Usability engineering file | Records of the IEC 62366-1 process, including use specification and summative evaluation | Running a formative test and calling it summative | IEC 62366-1:2015+A1:2020 |
| Traceability matrix | A controlled mapping from user needs to inputs, outputs, verification, and validation evidence | Building the matrix after the tests, from the test names | Local convention |
| Computer software assurance (CSA) | A risk-based approach to assurance of production and quality system software, proportionate to intended use and record integrity | Applying device software verification rigor to an enterprise resource planning system, or the reverse | FDA CSA guidance |
| 21 CFR Part 11 | US requirements for electronic records and electronic signatures, including audit trails and access control | Assuming a cloud vendor's compliance covers the manufacturer's records | 21 CFR Part 11 |
| Audit trail | A secure, computer-generated, time-stamped record of creation, modification, and deletion of electronic records | Enabling an audit trail that nobody reviews | 21 CFR 11.10(e) |
| Notified body | A European Union (EU) organization designated to assess conformity with the Medical Device Regulation (MDR) | Treating a notified body as a government agency | MDR Article 43 |
| Technical documentation | The EU dossier demonstrating conformity, assessed under Annex IX | Assuming a US 510(k) package satisfies the EU dossier | MDR Annex IX |
| Change impact assessment | A documented analysis of what a change affects and what re-verification, re-validation, or submission follows | Declaring a change "minor" without criteria | Local convention |
| Predetermined change control plan (PCCP) | A pre-authorized description of planned modifications with their verification, validation, and monitoring | Using a PCCP to authorize open-ended learning | FDA PCCP guidance |

Three distinctions in the table carry most of the practical weight.

The first is design history file versus device master record versus device history record. The
design record answers "why is the device like this and what evidence supports it." The master
record answers "how is the device made." The history record answers "was this unit made that
way." A model artifact belongs in the design record as an output, in the master record as a
controlled release, and in the production record as the version actually deployed. Teams that
collapse the three cannot answer a recall traceability question.

The second is verification versus validation. Verification asks whether the output meets the
input. Validation asks whether the device meets the user need in the intended use environment.
A model that meets its sensitivity specification has been verified against that specification. It
has not been validated if no intended user ever used it on the intended population in the
intended setting. The QMSR keeps both requirements in ISO 13485 Clauses 7.3.6 and 7.3.7, and
reviewers test them separately.

The third is design input quality. A design input is not a goal. "The model should be accurate"
is a goal. "Sensitivity for atrial fibrillation at the locked operating point shall be at least
0.90 with a two-sided 95 percent confidence lower bound of at least 0.85 in the intended-use
population" is a design input. Testability is the property that makes the rest of the chain
possible; an untestable input cannot be verified, and a requirement that cannot be verified
cannot be defended.

A note on terminology after February 2026. The QMSR adopts ISO 13485 language, so the design and
development file at Clause 7.3.10 is the successor concept to the design history file. Most
organizations still say DHF, and many procedures use both terms. The name matters less than the
content: the file must show that the design was developed in accordance with the approved plan
and must contain or reference the inputs, outputs, reviews, verification, validation, and change
records.

## 5. The mechanics, step by step

Design control is a procedure with artifacts, not a philosophy. The sequence below is written for
a software device, but the logic holds for hardware and for combined systems. Each step names its
input, its output, the reason it exists, the actions, and the branches that change the path. Work
in order at the start. After release, work the loop: change enters at Step 11 and returns to
review, verification, and validation.

### Step 1 — Fix the regulatory basis and QMS scope

**Input.** Target markets, device types, current certificates, existing procedures, and the dates
on which each rule applies.

**Output.** A scope statement naming the binding instruments, the standards invoked, the clauses
each procedure satisfies, and the effective date of the analysis.

**Rationale.** The QMSR replaced the text that most procedures were written against. A procedure
that cites 820.30 by name is not wrong in substance but is stale in citation, and a certificate
whose scope excludes the device is not evidence for the device. Fixing the basis first prevents
a year of work against the wrong clause set.

**Procedure.** (1) List the markets and the legal instrument for each. (2) Record the version in
force, including the QMSR effective date of 2 February 2026. (3) List the standards the
organization invokes voluntarily, such as ISO 13485:2016, ISO 14971:2019, IEC 62304:2006+A1:2015,
and IEC 62366-1:2015+A1:2020. (4) Map every existing procedure to the clause it satisfies. (5)
Mark procedures with no clause and clauses with no procedure. (6) Have quality and regulatory
approve the scope.

**Branches and edge cases.** A software-only device has no production line, but design transfer
and release still apply to the deployment process. A legacy device first marketed before design
controls took effect is not retroactively documented, but changes to it are controlled and the
current configuration must be described. A Class I device may not require design controls if it
was exempt from premarket notification and premarket approval under the legacy rule; confirm the
QMSR applicability rather than assuming, and remember that general controls still apply. A
combination product follows 21 CFR Part 4 and the lead center's QMS expectations, and the device
constituent part still needs design controls.

### Step 2 — Write the design and development plan

**Input.** Intended use, regulatory basis, program schedule, team roles, and the risk management
plan.

**Output.** An approved plan defining stages, gates, deliverables, review types, responsibilities,
acceptance criteria, and configuration management.

**Rationale.** ISO 13485 Clause 7.3.2 requires design and development planning and control. The
plan is what makes a review meaningful, because it states what must be true to pass the gate.
Without it, "design review" becomes a status meeting with slides.

**Procedure.** (1) Define stages that match the product, not a template. (2) Assign deliverables
and owners to each stage. (3) Write entry and exit criteria for each gate. (4) Name reviewers by
role, with the competence required. (5) Define how outputs are identified, versioned, and
baselined. (6) Define how the plan itself is changed and approved. (7) Approve before the work it
governs begins.

**Branches and edge cases.** A software-only device maps stages to IEC 62304 processes rather than
hardware phases. An agile team can run sprints inside stages, but the gates and their evidence
remain. A Class I device can use a lighter plan, but the plan must exist and be followed. A
cloud-first product should include environment qualification and deployment rollback as plan
deliverables rather than operational afterthoughts.

### Step 3 — Capture user needs and intended use

**Input.** Clinical workflow observation, user research, the draft labeling, the classification
memo, and the intended-use statement.

**Output.** A numbered user-need list tied to the intended-use statement, naming users, settings,
populations, tasks, and the consequence of failure.

**Rationale.** User needs anchor validation. If they are written as features, validation collapses
into verification and the team can no longer show that the device is useful, only that it is
built to specification.

**Procedure.** (1) Observe the workflow where the device will be used. (2) Write each need in the
user's language and in terms of the task, not the algorithm. (3) Separate mandatory from desirable
needs. (4) Record the source and date for each need. (5) Reconcile the needs with the labeling
and the intended-use statement. (6) Obtain clinical sign-off.

**Branches and edge cases.** Multiple user groups require separate needs and separate validation
evidence; a clinician and a patient do not share a need because they share a device. Patient-facing
use adds comprehension and health-literacy needs. A combination product adds administration and
handling needs. A workflow tool that changes who does what adds a role-change need that is easy to
omit and is often the source of use error.

### Step 4 — Derive design inputs

**Input.** User needs, applicable regulations and standards, risk control measures, and the
intended-use statement.

**Output.** A numbered, testable input specification with acceptance criteria, priority, source,
and verification method for each entry.

**Rationale.** Inputs are what verification tests against. An input that cannot fail a test cannot
be verified, and a requirement that cannot be verified cannot be defended to a reviewer.

**Procedure.** (1) Allocate each user need to one or more inputs. (2) Add regulatory inputs such as
labeling content, cybersecurity, and, for hardware, electrical safety. (3) Add each risk control
as an input. (4) Make every input measurable, including its population and conditions. (5) State
the verification method. (6) Review for conflicts and duplicates. (7) Baseline the set.

**Branches and edge cases.** A performance threshold that depends on prevalence must state the
prevalence, or the test will not reproduce. Usability inputs need task-success and error criteria,
not "the interface shall be intuitive." An input that cannot be measured directly needs a
justified proxy and a stated limitation. Conflicting inputs are resolved in review, and the
resolution is recorded rather than deleted.

### Step 5 — Produce design outputs

**Input.** Baselined design inputs and the architecture constraints.

**Output.** Architecture description, detailed design, source code, model artifacts, labeling,
packaging specifications, and acceptance criteria for each output.

**Rationale.** Outputs are the objects that verification examines. For an AI device the output set
includes data provenance, training configuration, model weights, preprocessing, calibration, the
locked operating point, and inference code. Omitting any of these makes the verified object
different from the shipped object.

**Procedure.** (1) Produce outputs against the baselined inputs. (2) Link each output to the
inputs it satisfies. (3) Place outputs under version control. (4) Record the toolchain and
dependency versions. (5) Record the data version or manifest. (6) Write acceptance criteria into
the output document rather than a separate test plan.

**Branches and edge cases.** A trained model is an output but not a self-describing one; pair it
with a model card and a training manifest. Off-the-shelf components are outputs only in the sense
of selection and configuration, and they need identification and version. Infrastructure as code
is a design output for a cloud device. Labeling is a design output and is verified like any
other.

### Step 6 — Hold design reviews

**Input.** Outputs at a defined stage, the plan, the input specification, and the risk file.

**Output.** A review record containing date, attendees and roles, materials reviewed, findings,
actions, owners, due dates, and closure evidence.

**Rationale.** Reviews are the independent challenge in the process. A review without a record is
not evidence, and a review whose attendees lack the competence to challenge the design is a
formality that auditors recognize quickly.

**Procedure.** (1) Define the review type and its scope. (2) Distribute materials in advance. (3)
Record attendance, role, and independence. (4) Log findings with severity and disposition. (5)
Track actions to closure with evidence. (6) Re-review when a material change follows.

**Branches and edge cases.** A small team needs an independent reviewer from outside the project;
record the independence rather than implying it. A review after a change should be scoped to the
change and its interfaces, not repeated in full. A review that finds nothing should record what
was examined, because "no findings" without scope is indistinguishable from "no review."

### Step 7 — Verify

**Input.** Baselined outputs, the input specification, and approved test methods.

**Output.** A verification plan, protocols, raw results, a summary report, and a traceability
report showing coverage of every input.

**Rationale.** Verification is objective evidence that output meets input, and coverage is
checkable. It is the step where a data scientist's habits are most useful and most often
missing: pre-specified analyses, fixed test sets, recorded environments, and reproducible runs.

**Procedure.** (1) Write the plan before execution. (2) Define pass and fail criteria per input.
(3) Execute on the release candidate, not on a development build. (4) Record the build identifier,
container digest, and environment. (5) Investigate every failure and record the disposition. (6)
Re-test after corrections. (7) Report coverage, including the inputs that were not covered and
why.

**Branches and edge cases.** Statistical verification needs a pre-specified analysis and a
confidence bound, not a point estimate. A test environment that differs from production needs a
qualification of the difference. SOUP is verified through integrated behavior, not by auditing
the library's internals. An input that cannot be exercised in the test environment requires a
documented rationale and approval, not silent omission.

### Step 8 — Thread risk management through the design

**Input.** Hazard analysis, use-related hazards, the software safety class, and the design inputs.

**Output.** Risk control measures entered as design inputs, evidence that each control was
implemented, evidence that each control is effective, and an updated residual risk evaluation.

**Rationale.** ISO 14971 requires controls to be implemented and their effectiveness verified. A
control that is not a requirement cannot be shown to exist, and a control that exists only in the
risk file is a claim without a test.

**Procedure.** (1) Identify hazards and hazardous situations. (2) Estimate and evaluate risk. (3)
Select controls in the required order: inherent safety, protective measures, information for
safety. (4) Enter each control as a design input. (5) Verify implementation and effectiveness
separately. (6) Re-estimate residual risk. (7) Feed production and post-production data back into
the file.

**Branches and edge cases.** Software hazards take their rigor from the IEC 62304 safety class,
which is set by the possible consequence of failure, not by the device class. Use errors are
handled through IEC 62366-1 and often produce controls that are information for safety, which
must be verified for comprehension rather than mere presence. A control that depends on a
clinician reading a warning is weaker than an inherent design change and should be justified in
the file.

### Step 9 — Validate

**Input.** The device as designed, the user needs, the intended-use statement, and a definition of
the use environment.

**Output.** A validation plan, protocol, results, and a conclusion against each user need.

**Rationale.** Validation is evidence that the device works for the intended user in the intended
context. It cannot be replaced by verification, by a retrospective accuracy study, or by a
successful demo. This is the step where AI devices most often fail review.

**Procedure.** (1) State the claims and the endpoints that support them. (2) Choose the setting:
clinical, simulated, or bench, with a justification. (3) Define the user sample and the case mix.
(4) Pre-specify the analysis, including subgroups. (5) Execute and record. (6) Report against each
user need. (7) State the limitations that the labeling must carry.

**Branches and edge cases.** A software-only device typically uses simulated use with
representative users and data, unless the claim requires clinical outcome evidence. A high-risk
claim needs a prospective study. A Class I device still needs validation scaled to its risk. A
combination product adds human factors for the combined use and evidence on drug-device
interaction.

### Step 10 — Transfer to production and release

**Input.** The verified and validated design, production specifications, and release criteria.

**Output.** The device master record, a validated production or deployment process, release
records, training records, and a deployment and rollback procedure.

**Rationale.** Transfer is where design intent becomes reproducible output. In software, the
production line is the deployment pipeline, and an unvalidated pipeline can change the device
without a design change.

**Procedure.** (1) Define the production or deployment process. (2) Validate the process. (3)
Qualify the runtime environment. (4) Define release criteria and who may approve release. (5)
Verify the deployed artifact against the design output. (6) Train users and support staff. (7)
Record the release and its configuration.

**Branches and edge cases.** Cloud deployment requires control of infrastructure version, model
server configuration, and rollback. Containerized inference requires deployment by image digest,
not by tag. A supplier-hosted component requires interface qualification and an agreement on
change notification. A combination product requires assembly, packaging, and sterilization
transfer with their own validation.

### Step 11 — Control changes

**Input.** A proposed change, the current baseline, and the change rationale.

**Output.** A change request, an impact assessment, approvals, updated verification and validation,
updated records, and any required notifications.

**Rationale.** Most software device failures originate in changes that were not treated as design
changes. A threshold edit, a dependency upgrade, and a retraining run all alter the device. If the
change is invisible to the design record, the evidence no longer describes the product.

**Procedure.** (1) Classify the change as design, process, labeling, or supplier. (2) Assess impact
on inputs, outputs, risk controls, verification, validation, and regulatory status. (3) Decide the
evidence required. (4) Obtain approval before implementation. (5) Implement under configuration
control. (6) Verify and, where required, validate. (7) Update the record and notify customers or
regulators where required.

**Branches and edge cases.** A threshold change requires verification at the new operating point
and a re-assessment of clinical performance, not just a regression test. Retraining is a model
change even when the code is unchanged. A security patch requires an exploitability assessment and
regression testing. A labeling-only change is still a change. A change made under a predetermined
change control plan follows the pre-authorized protocol and its acceptance criteria.

### Step 12 — Control suppliers, off-the-shelf software, and SOUP

**Input.** Supplier list, software bill of materials, cloud services, contract manufacturers, and
the architecture.

**Output.** Supplier classification and evaluation records, quality agreements, a SOUP inventory,
anomaly evaluations, and change-notification obligations.

**Rationale.** A supplier change can change the device while the manufacturer changes nothing. The
QMSR keeps purchasing control in ISO 13485 Clause 7.4, and the obligation to evaluate and monitor
suppliers is unchanged in substance from the legacy rule.

**Procedure.** (1) Classify suppliers by the risk of what they supply. (2) Evaluate before
selection and re-evaluate periodically. (3) Define controls proportionate to risk. (4) Require
notification of changes that affect the device. (5) Maintain a software bill of materials
including transitive dependencies. (6) Evaluate known anomalies and end-of-life status. (7)
Record the evaluation.

**Branches and edge cases.** OTS software relies on vendor validation evidence plus integration
testing; commercial availability does not remove the evaluation. SOUP depth follows the software
safety class. A cloud provider is a supplier, and the shared responsibility model must be written
down rather than assumed. A supplier change that affects the device is processed as a design
change, not as a purchasing update.

### Step 13 — Assemble and control the design record

**Input.** The plan, inputs, outputs, reviews, verification and validation evidence, risk file,
and change records.

**Output.** An indexed, versioned, access-controlled design record with a defined retention period
and a completeness report.

**Rationale.** The record is what a reviewer sees. An unindexed record turns an audit into a search
problem under time pressure, and a missing document is treated as a missing activity.

**Procedure.** (1) Define the index structure before documents accumulate. (2) Assign identifiers
and revisions. (3) Link documents to the traceability matrix. (4) Control access and backup. (5)
Set retention at least the device lifetime and no less than the period the standard requires. (6)
Review completeness at every gate. (7) Archive the released configuration.

**Branches and edge cases.** A legacy device gets a record for its current configuration, not a
fabricated history. Electronic records need Part 11 controls, including audit trails. An
acquisition requires a documented transfer of records and responsibilities. A long-life device
outlasts employment, so retention and retrieval must survive staff turnover.

### Step 14 — Operate the QMS around the record

**Input.** Complaints, production and deployment data, audit findings, post-market data, and
management review inputs.

**Output.** CAPA records, internal audit reports, management review minutes, revised procedures,
and training records.

**Rationale.** Design controls depend on the surrounding system. A CAPA system that does not feed
design change is blind to the most important source of design information: what happened after
release.

**Procedure.** (1) Collect feedback from complaints, service, and post-market surveillance. (2)
Investigate and correct. (3) Check effectiveness rather than closing at the document edit. (4)
Audit internally on a schedule. (5) Review quality data at management level. (6) Update procedures
and train. (7) Feed design changes back to Step 11.

**Branches and edge cases.** A complaint that implicates the design opens a design change, not
only a CAPA. An audit finding needs a root cause, not a document edit. A supplier-caused issue
needs a supplier corrective action and a re-evaluation. A post-market signal about subgroup
performance feeds both the risk file and the design input set.

### Decision table: route by scenario

| Scenario | Design controls | Record required | Change route | Evidence emphasis |
|---|---|---|---|---|
| New software as a medical device (SaMD), no predicate, De Novo | Full | Full design record | New design | Validation in intended use, analytical performance |
| New SaMD with a 510(k) predicate | Full | Full design record | New design | Substantial equivalence plus verification |
| Class I exempt, no software medical function | Not required under legacy rule; confirm QMSR | Complaint and CAPA records | Change control as applicable | Labeling and basic controls |
| Class I with a software function that makes a claim | Apply; confirm applicability | Design record scaled to risk | Design change | Verification and usability |
| Threshold or model change after release | Full | Change record plus updated evidence | Design change or PCCP | Re-verification at the new operating point |
| Labeling-only change | Yes | Change record and labeling verification | Design change | Comprehension and consistency |
| Legacy device, no change | Applies to changes only | Current configuration description | Design change when triggered | Configuration and risk review |
| Legacy device, supplier change | Yes | Supplier and change records | Design change | Equivalence and regression |
| SOUP security patch | Yes | Change record and SOUP evaluation | Design change | Exploitability and regression |
| Cloud region or infrastructure migration | Yes | Change record and environment qualification | Design change | Latency, availability, data residency |
| Drug-device combination product | Yes, plus Part 4 | Combined record set | Design change | Human factors and interaction |
| Class III PMA device, manufacturing change | Yes | Change record and supplement | PMA supplement or notice | Prospective evidence and inspection |

The table is a routing aid, not a substitute for the impact assessment. Two teams can place the
same change in different rows and both be defensible if the reasoning is recorded. The failure is
not choosing the wrong row; it is not asking which row applies.

## 6. Worked example

The company, product, and findings are constructed. The regulatory reasoning and the arithmetic
are the point. A real determination requires the current regulatory text and professional review.

### 6.1 Product, claim, and regulatory posture

RhythmTrace ECG is a software-only device from Northbridge Diagnostics. It analyzes a 12-lead
electrocardiogram (ECG) and returns one of four outputs per record: atrial fibrillation detected,
other rhythm abnormality detected, no abnormality detected, or uninterpretable. The intended
users are emergency physicians and cardiology clinic staff. The intended settings are emergency
departments and ambulatory cardiology clinics. The claim is triage and prioritization: the output
is an aid, not a diagnosis, and does not replace a cardiologist overread.

The device is US Class II, filed through a 510(k) with a predicate that carries a clinician-facing
arrhythmia triage claim. The European Union is out of scope for the first launch. The QMSR has
been in force since 2 February 2026. The submission is scheduled for 15 April 2026.

The model is a one-dimensional convolutional network with attention, trained on 412,000 ECGs from
three emergency department sites. The validated configuration is model version 2.4.1 at a decision
threshold of 0.52. The validation report covers 8,412 held-out ECGs and reports an area under the
receiver operating characteristic curve (AUROC) of 0.941, sensitivity of 0.912, and specificity
of 0.887 at that threshold.

### 6.2 What the mock audit found

An internal mock audit is run on 6 February 2026, two weeks after the QMSR effective date, to test
readiness for both the submission and a possible inspection.

| Artifact | What the plan required | What the audit found |
|---|---|---|
| Design and development plan | Approved before gate 1, five gates defined | Plan exists; approval page undated; gate 4 and gate 5 records absent |
| User needs | 31 needs, each traced to at least one input | 31 needs; 6 have no downstream input |
| Design inputs | 84 testable inputs with methods | 84; 13 have no verification test; 4 are safety-related |
| Design outputs | Model card with calibration and subgroup sections | Model card lacks calibration; operating point recorded as a range |
| Design reviews | Five reviews with minutes and closed actions | Three records; one has attendees, date, and closed actions |
| Verification | Every input covered by a protocol and result | 71 of 84 covered |
| Validation | Protocol approved before data lock | No protocol; retrospective analysis run after the release build |
| Risk file | 57 controls, each linked to an input | 44 linked; 13 exist only in the risk file |
| SOUP | Bill of materials to transitive depth with anomaly review | First-level dependencies only; no anomaly review |
| Design record index | 112 controlled documents | 92 present; 20 missing |
| Change control | Every change assessed and approved | 2 of 4 changes recorded; the threshold change is absent |
| Training | Completion records for each release | Slide deck only, no completion records |

The file is better than the average startup file and worse than the average submission requires.
The two findings below are the ones that would sink it.

### 6.3 Finding 1: the validated device is not the shipped device

The validation report, the model card, and the draft 510(k) software description all describe
version 2.4.1 at threshold 0.52. The build in the release candidate is version 2.5.0 at threshold
0.47. The change was made on 9 January 2026 in a notebook commit whose message reads "improve
sensitivity." There is no change request, no impact assessment, no re-verification, no update to
the risk file, and no update to the labeling's expected alert burden.

ISO 13485 Clause 7.3.9 requires design changes to be reviewed, verified, validated, and approved
before implementation, and Clauses 4.2.4 and 4.2.5 require the change to be reflected in
controlled documents and records. The failure is not that the new threshold is wrong. The failure
is that the evidence applies to a configuration that will not ship, so the verification and
validation conclusions are about a different device.

In a 510(k), the software description and the performance data would not describe the device as
released. A reviewer who compares the submission with the release build finds a different
operating point with different clinical performance. That is not a typographical correction. It
is an unsupported performance claim, and it can convert a routine deficiency into an enforcement
question. In an inspection, it is a nonconformity against change control and against the
requirement that the design record demonstrate the design was developed in accordance with the
plan.

### 6.4 The arithmetic of the threshold change

The validation report contains operating points evaluated on the held-out set. Two are relevant.
At a threshold of 0.52, sensitivity is 0.912 and specificity is 0.887. At 0.47, sensitivity is
0.938 and specificity is 0.851. Assume an intended-use prevalence of 2.1 percent, consistent with
the emergency department case mix in the training sites. The table reports positive predictive value
(PPV) and negative predictive value (NPV) at that prevalence.

| Operating point | Sensitivity | Specificity | PPV | NPV | Alerts per 10,000 | Missed per 10,000 |
|---|---:|---:|---:|---:|---:|---:|
| 0.52 (validated) | 0.912 | 0.887 | 0.148 | 0.998 | 1,298 | 18.5 |
| 0.47 (released) | 0.938 | 0.851 | 0.119 | 0.998 | 1,656 | 13.0 |

PPV at 0.52 is 0.912 times 0.021, divided by that product plus 0.113 times 0.979, which equals
0.019152 divided by 0.129779, or 0.148. At 0.47 it is 0.019698 divided by 0.165569, or 0.119. Per 10,000 patients, the change prevents about 5.5 missed cases and
adds about 352 false alerts, raising alert volume by 27.6 percent.

The clinical argument for the lower threshold may be sound. The process argument is absent.
Neither the risk file nor the labeling reflects the new burden, and the subgroup analysis was not
re-run at the new operating point, so the performance in the smallest subgroup is unknown.

### 6.5 Finding 2: validation does not reach the intended use

No validation protocol existed before the analysis. The "validation" is a retrospective
evaluation of 8,412 held-out ECGs drawn from the same three emergency department sites as the
training data, scored against a cardiologist overread. It measures discrimination. It does not
measure use.

Three gaps follow. First, the intended use names ambulatory cardiology clinics, and the evaluation
contains no ambulatory records, so the claim is broader than the evidence. Second, no clinician
ever used the output in the evaluation; there is no summative usability evaluation under IEC
62366-1, and the uninterpretable branch, which occurs in 9.3 percent of records, has no validated
user action. That is 930 records per 10,000 in a path nobody tested. Third, four of the 26
safety-related inputs have no verification, including "the device shall not display no abnormality
detected when lead displacement exceeds the quality threshold."

ISO 13485 Clause 7.3.7 requires validation to confirm that the device meets user needs and
intended use under specified conditions. Evidence from one setting, with no users, cannot support
a claim that includes two settings and four outputs. Six user needs also have no design input, so
even a perfect study could not conclude against them. The missing 20 documents include four review
records and the verification reports for the final build, which means the file cannot demonstrate
that the gaps were reviewed.

### 6.6 The decision

| Option | Assessment | Consequence |
|---|---|---|
| File on 15 April as planned | Not defensible | Software description inconsistent with the build |
| File and disclose the threshold change | Disclosure does not cure the evidence gap | Reviewer requests data for 0.47 and for ambulatory use |
| Hold, revert to 0.52, narrow the claim to emergency departments, close verification, run simulated-use validation | Recommended | File in early June, about seven weeks late |
| Hold, keep 0.47, collect ambulatory data and run a summative study | Defensible but slow | File in the fourth quarter, with a larger evidence package |

Northbridge chooses the third option. Reverting the threshold removes the configuration mismatch
and is faster than validating a change nobody approved. Narrowing the claim removes the setting
gap and is honest about what the evidence supports. The threshold change can return later through
change control with its own verification and a subgroup analysis, or through a predetermined
change control plan if the range was pre-specified, which it was not.

The remediation estimate is 1,180 person-hours, or $218,300 at a loaded rate of $185 per hour.
Maintaining traceability during design would have cost roughly half an hour per input per change
cycle: 84 inputs times 3 cycles times 0.5 hours equals 126 hours, or $23,310. The ratio is about
9.4 to 1. The comparison is an estimate, not a law, but the direction is consistent with most
post-mortems.

| Week | Action | Owner | Exit criterion |
|---:|---|---|---|
| 1 | Retrospective change assessment; revert to 2.4.1 at 0.52 | Quality | Change record approved; build reverted |
| 1-2 | Verify the 13 uncovered inputs, safety-related first | Engineering | 84 of 84 inputs covered |
| 2-3 | Trace or remove the 6 orphan user needs | Product | Every need has an input or is removed from the claim |
| 3 | Rebuild the record index and recover the 20 documents | Quality | Index complete and reviewed |
| 3-4 | Write and approve the validation protocol | Clinical | Protocol approved before data lock |
| 4-6 | Run simulated-use validation, 15 participants per user group | Human factors | Pre-specified endpoints met or claim narrowed |
| 6 | Update labeling, risk file, and traceability report | Regulatory | Documents consistent with 2.4.1 |
| 7 | Repeat mock audit and management review | Quality | No blocking findings |

### 6.7 Vary one assumption

Each variant below changes exactly one assumption from the base case and leaves everything else
in place.

| Variant | Assumption changed | US pathway | Design-control consequence | Evidence consequence |
|---|---|---|---|---|
| A | Class II becomes Class III | PMA | Pre-approval inspection of the QMS; design record examined before approval | Prospective clinical evidence; supplement for the threshold change |
| B | New device becomes a legacy device | Existing clearance, change only | Design controls apply to the change, not retroactively to the original design | Configuration baseline and impact assessment |
| C | QMSR becomes the pre-2026 QSR | Same 510(k) | Citations move from Clause 7.3 to 820.30(a)-(j) | Same substance; different audit trail and terminology |

**Variant A: Class III and a PMA.** If the product falls in a Class III classification regulation,
for example because the claim is diagnostic and the risks cannot be controlled through general and
special controls, the pathway becomes a PMA with a pre-approval inspection. The inspection
examines the quality system before approval, so the design record gap is no longer a post-market
risk; it is a gate. A change to a released device generally requires a PMA supplement, with a
30-day notice available only for specified manufacturing changes. The unrecorded threshold change
becomes a change made without an approved supplement, which is a more serious matter than a 510(k)
inconsistency. Clinical evidence moves from a simulated-use study to a prospective study with a
clinically meaningful endpoint. Timeline extends by 12 to 18 months, and the same remediation
costs more because it is performed under inspection.

**Variant B: a legacy device.** If the product had been marketed since 1994 and the question were
how to treat an existing design, the answer changes in a different direction. Design controls are
not applied retroactively to reconstruct a 1994 design file. They apply to changes. The practical
response is to document the current configuration, perform a gap assessment against Clause 7.3,
and treat the next change as the trigger for a full design-control cycle. The record then contains
an honest account of the current design plus the change history, rather than a fabricated history.
The QMSR does not grandfather records, and an auditor will still ask how the current configuration
is controlled.

**Variant C: the pre-2026 QSR.** Before 2 February 2026, the citation set was 21 CFR 820.30(a)
through (j): general, planning, inputs, outputs, review, verification, validation, transfer,
changes, and the design history file. After the effective date, the same activities sit in ISO
13485 Clauses 7.3.1 through 7.3.10, with additional record and labeling requirements elsewhere in
Part 820. The substance is close enough that a well-run pre-2026 process survives the transition.
Three things change in practice. The audit question cites a clause rather than a paragraph. ISO
13485 requires a quality manual and documented procedures that many small software firms never
wrote under the QSR. And the design record is now the design and development file, with the
retention and control expectations of Clause 4.2.5. A team that simply renames its DHF folder
will pass a document review and fail a process audit.

### 6.8 The lesson

The model was not the problem. The evidence chain was. The team could describe the algorithm in
detail and could not show which requirements the algorithm satisfied, which build the validation
covered, or which user tested which output. Design controls are the discipline that makes those
answers cheap to produce. The seven-week delay is the price of producing them late.

## 7. Code, query, or calculation

The script below scores a design record against six gates and prints a weighted readiness score.
It is small enough to audit and self-contained, so it can run without a repository. The demo
dataset reproduces the counts in Section 6. Replace `build_demo()` with a loader for your own
design record index to use it on real data.

```python
"""Design-control completeness scorer for Module 14, Topic 05.

The demo dataset is synthetic and reproduces the counts in Section 6.
Replace build_demo() with a loader for your own design-history index.

Exit code 0 means no blocking finding; 1 means at least one.
"""

from __future__ import annotations

from dataclasses import dataclass

TOTAL_INPUTS = 84
SAFETY_IDS = {2, 5, 7, 11, 17, 23, 26, 31, 35, 40, 44, 47, 53, 58,
              61, 64, 67, 70, 73, 76, 79, 80, 81, 82, 83, 84}
UNTESTED_IDS = {11, 14, 23, 37, 41, 48, 52, 57, 63, 68, 72, 79, 83}
SECOND_TEST_IDS = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 15, 16, 17, 18}
TOTAL_TESTS = 96
ORPHAN_TESTS = 9
TOTAL_CONTROLS = 57
LINKED_CONTROLS = 44
TOTAL_DOCS = 112
PRESENT_DOCS = 92


@dataclass
class Requirement:
    req_id: str
    safety_related: bool
    test_ids: list[str]


def build_demo() -> tuple[list[Requirement], int, int, str, str]:
    """Build a dataset with the exact counts used in the worked example."""
    reqs: list[Requirement] = []
    test_no = 0
    for i in range(1, TOTAL_INPUTS + 1):
        tests: list[str] = []
        if i not in UNTESTED_IDS:
            test_no += 1
            tests.append(f"T-{test_no:03d}")
            if i in SECOND_TEST_IDS:
                test_no += 1
                tests.append(f"T-{test_no:03d}")
        reqs.append(Requirement(f"R-{i:03d}", i in SAFETY_IDS, tests))
    return reqs, test_no, ORPHAN_TESTS, "v2.5.0", "v2.4.1"


def coverage(reqs: list[Requirement]) -> dict[str, float]:
    tested = [r for r in reqs if r.test_ids]
    safe = [r for r in reqs if r.safety_related]
    safe_tested = [r for r in safe if r.test_ids]
    return {
        "inputs": len(reqs),
        "tested": len(tested),
        "input_coverage": len(tested) / len(reqs),
        "safe": len(safe),
        "safe_tested": len(safe_tested),
        "safety_coverage": len(safe_tested) / len(safe),
    }


def main() -> int:
    reqs, linked_tests, orphan_tests, released, validated = build_demo()
    c = coverage(reqs)
    total_tests = linked_tests + orphan_tests
    assert total_tests == TOTAL_TESTS, "demo counts drifted"
    rc_linkage = LINKED_CONTROLS / TOTAL_CONTROLS
    dhf = PRESENT_DOCS / TOTAL_DOCS
    version_ok = released == validated

    score = (
        0.30 * c["input_coverage"]
        + 0.25 * c["safety_coverage"]
        + 0.20 * rc_linkage
        + 0.15 * dhf
        + 0.10 * (1.0 if version_ok else 0.0)
    )

    print("Design-control readiness report")
    print("-" * 46)
    print(f"design inputs                    {c['inputs']:>4}")
    print(f"  with at least one test         {c['tested']:>4}  "
          f"({c['input_coverage']:.1%})")
    print(f"  without any test               {c['inputs'] - c['tested']:>4}  "
          f"({1 - c['input_coverage']:.1%})")
    print(f"  safety-related                 {c['safe']:>4}")
    print(f"  safety without test            {c['safe'] - c['safe_tested']:>4}")
    print(f"verification tests               {total_tests:>4}")
    print(f"  orphan tests                   {orphan_tests:>4}  "
          f"({orphan_tests / total_tests:.1%})")
    print(f"risk controls                    {TOTAL_CONTROLS:>4}")
    print(f"  linked to a requirement        {LINKED_CONTROLS:>4}  "
          f"({rc_linkage:.1%})")
    print(f"DHF index documents              {TOTAL_DOCS:>4}")
    print(f"  present                        {PRESENT_DOCS:>4}  ({dhf:.1%})")
    print(f"  missing                        {TOTAL_DOCS - PRESENT_DOCS:>4}")
    print()
    print(f"weighted score {score:.3f} (readiness threshold 0.900)")
    print()

    findings: list[str] = []
    checks = [
        ("G1 safety coverage >= 0.95", c["safety_coverage"] >= 0.95,
         f"{c['safety_coverage']:.3f}"),
        ("G2 input coverage >= 0.95", c["input_coverage"] >= 0.95,
         f"{c['input_coverage']:.3f}"),
        ("G3 orphan tests == 0", orphan_tests == 0, str(orphan_tests)),
        ("G4 control linkage >= 0.95", rc_linkage >= 0.95, f"{rc_linkage:.3f}"),
        ("G5 DHF completeness >= 0.90", dhf >= 0.90, f"{dhf:.3f}"),
        ("G6 released == validated", version_ok, f"{released} vs {validated}"),
    ]
    for name, ok, shown in checks:
        print(f"{name:32s} {'PASS' if ok else 'FAIL'}  {shown}")
        if not ok:
            findings.append(name)

    print()
    if findings:
        print(f"result: NOT READY ({len(findings)} blocking findings)")
        print("exit code: 1")
        return 1
    print("result: READY")
    print("exit code: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

The script was run with Python 3.11.15 and produced the following output exactly.

```text
Design-control readiness report
----------------------------------------------
design inputs                      84
  with at least one test           71  (84.5%)
  without any test                 13  (15.5%)
  safety-related                   26
  safety without test               4
verification tests                 96
  orphan tests                      9  (9.4%)
risk controls                      57
  linked to a requirement          44  (77.2%)
DHF index documents               112
  present                          92  (82.1%)
  missing                          20

weighted score 0.743 (readiness threshold 0.900)

G1 safety coverage >= 0.95       FAIL  0.846
G2 input coverage >= 0.95        FAIL  0.845
G3 orphan tests == 0             FAIL  9
G4 control linkage >= 0.95       FAIL  0.772
G5 DHF completeness >= 0.90      FAIL  0.821
G6 released == validated         FAIL  v2.5.0 vs v2.4.1

result: NOT READY (6 blocking findings)
exit code: 1
```

Four cautions apply. The counts measure linkage, not adequacy: an input can be tested badly and
still count as covered. The weights are a management choice, not a regulatory standard, and a
different weighting would produce a different score from the same evidence. The gates should be
set from the device's risk and the submission's requirements, not copied unexamined. And a
passing score is not a submission: a complete matrix around the wrong requirement is a
well-organized failure.

## 8. Data, standards, and regulatory context

> **Note:** This section is educational and is not legal advice. Quality-system obligations are
> fact-specific and jurisdiction-specific. Confirm the current regulatory text, the scope of any
> certificate, and the applicable guidance with qualified regulatory counsel and a quality
> professional before relying on this material.

The sources below differ in legal force. A statute or regulation is binding. A standard is
voluntary unless a regulation incorporates it, a contract requires it, or the manufacturer adopts
it into its own quality system. Guidance describes an agency's current thinking and is not
binding, but a submission that departs from it without an alternative rationale invites questions.
Convention is local practice, such as a preferred template, and has no legal force at all.

| Source | Status | What it constrains | What it does not do |
|---|---|---|---|
| QMSR final rule, 89 FR 7496, 2 February 2024, effective 2 February 2026 | Binding US regulation | Restructures 21 CFR Part 820 and sets device quality-system requirements | Does not prescribe clinical evidence standards |
| 21 CFR 820.10 | Binding US regulation | Establishes the QMS requirement and the applicability of the incorporated ISO 13485 clauses | Does not reproduce the ISO text in the CFR |
| 21 CFR 820.30 (pre-2026) | Superseded US regulation | Governed design controls until 1 February 2026 | Does not apply after the QMSR effective date |
| 21 CFR 820.35, 820.45 | Binding US regulation | Device-specific record and labeling and packaging controls retained in Part 820 | Do not replace ISO 13485 record control |
| 21 CFR Part 11 | Binding US regulation | Electronic records and signatures, audit trails, access control | Does not validate the underlying science |
| ISO 13485:2016 | Standard, incorporated by reference | QMS requirements, including Clause 7.3 design and development | Does not set device classification or clinical claims |
| ISO 14971:2019 | Standard | Risk management process and documentation | Does not define acceptable risk for a specific device |
| IEC 62304:2006+A1:2015 | Standard | Software lifecycle processes and safety classes A, B, C | Does not set the regulatory device class |
| IEC 62366-1:2015+A1:2020 | Standard | Usability engineering process and summative evaluation | Does not establish clinical effectiveness |
| FDA design control guidance, 1996 | Guidance | Describes FDA's expectations for design controls | Is not binding and predates the QMSR |
| FDA computer software assurance guidance, 24 September 2025, revised 3 February 2026 | Guidance | Risk-based assurance for production and quality-system software | Does not apply to device software functions |
| FDA OTS software guidance | Guidance | Expectations for off-the-shelf software in a device | Does not substitute for supplier evaluation |
| FDA cybersecurity guidance, September 2023 | Guidance | Security risk management and premarket content | Does not set a security certification |
| MDR 2017/745, Annex IX | Binding EU regulation | Conformity assessment based on QMS and technical documentation | Does not apply outside the EU market |
| EN ISO 13485:2016/A11:2021 | Harmonized standard | Presumption of conformity with MDR quality-system requirements | Does not cover every MDR obligation |
| Medical Device Coordination Group (MDCG) guidance | EU guidance | Explains MDR interpretation | Is not binding on manufacturers or notified bodies |

The QMSR is the largest change in US device quality regulation in three decades. It replaces the
prescriptive structure of the 1996 Quality System Regulation with the ISO 13485:2016 requirements,
incorporated by reference. The design-control content that lived at 21 CFR 820.30 now lives in
ISO 13485 Clause 7.3, applied through 21 CFR 820.10. The agency retained a small number of
device-specific requirements that ISO 13485 does not address, including record provisions and
labeling and packaging controls. The practical effect is that a manufacturer must satisfy both
the incorporated standard and the retained provisions, and must be able to say which clause or
section each procedure implements.

Three practical consequences follow. First, a quality manual becomes an explicit expectation,
because ISO 13485 Clause 4.2.2 requires one and the QSR did not. Second, the audit conversation
changes from paragraph citations to clause citations, so procedures written around 820.30 need
mapping even when their content is fine. Third, the design record is now the design and
development file of Clause 7.3.10, with the record-control expectations of Clause 4.2.5.

The computer software assurance guidance addresses a different problem. It covers software used
in production and in the quality system, such as manufacturing execution systems, laboratory
information systems, and quality databases. It replaces exhaustive scripted validation with
assurance activities proportionate to risk and record integrity. It does not cover device software
functions, which remain subject to design verification and validation. The revision of 3 February
2026 aligns the guidance with QMSR terminology. Teams frequently apply the wrong regime in both
directions: full validation rigor to an enterprise dashboard, and no rigor to a model-serving
component that is part of the device.

In the European Union, Annex IX of the MDR sets out conformity assessment based on a quality
system and on an assessment of the technical documentation. The manufacturer's QMS must cover
design and development, and the notified body audits it, typically at least annually for the
duration of the certificate. EN ISO 13485:2016/A11:2021 is the harmonized standard that gives a
presumption of conformity with the quality-system requirements, but a certificate is not a
conformity assessment for a specific device: the technical documentation, clinical evaluation,
and post-market obligations are assessed separately.

What the standards do not do is as important as what they do. ISO 13485 does not tell you which
requirements to write, only that they must be written, reviewed, and traced. ISO 14971 does not
tell you what residual risk is acceptable; it requires a defined policy and a documented
evaluation. IEC 62304 does not tell you what your software must do, only how to develop and
maintain it according to its safety class. IEC 62366-1 does not prove clinical benefit. A design
record that cites standards without making product-specific decisions is compliant in form and
empty in substance.

The evidence a reviewer uses to test all of this comes from a small set of public sources. FDA
publishes inspection observations, warning letters, recall classifications, and the Manufacturer
and User Facility Device Experience (MAUDE) database of adverse events. It also publishes the QMSR
final rule and the associated fact sheets, which state the agency's transition expectations. In
the EU, notified body certificates and the European Database on Medical Devices (EUDAMED), as its
modules become available, provide the public view. None of these sources
shows a design history file. That is the point: the public record shows outcomes, and the design
record is what the auditor sees in private.

## 9. Industry practice and case evidence

Public evidence about design-control failures is concentrated in inspection records, recall
notices, and regulatory preambles rather than in peer-reviewed literature. The cases below
separate what is documented from what is inference.

**Inspection citations.** FDA publishes inspection observations and warning letters by
establishment. Design control and corrective action citations appear repeatedly in device
inspections. The documented fact is the existence and content of those records; the inspection
database allows verification by establishment and date. The inference, which the agency does not
publish as a statistic in a form suitable for citation here, is that the underlying cause is
usually a process that was never operated rather than a document that was lost.

**Philips Respironics and the foam recall.** In June 2021 Philips Respironics announced a recall
of certain continuous positive airway pressure and bi-level positive airway pressure devices
because of degradation of the polyester-based polyurethane sound-abatement foam, and FDA
classified the recall as Class I. The documented facts are the recall, its classification, and
the subsequent regulatory actions. The inference, not stated by FDA as a design-control finding in
the material reviewed here, is that the episode illustrates how a material and process decision
made years earlier becomes a design-verification question when the failure mode emerges in the
field. The transferable point is that design verification has to consider degradation over the
service life, not only performance at release.

**The Software Precertification pilot.** FDA ran a Software Precertification pilot from 2017 that
explored assessing a developer's quality culture and organization rather than only a product.
FDA did not finalize the pilot as an alternative to premarket review. The documented fact is the
pilot and its conclusion. The inference is that developer-level excellence, however real, did not
substitute for product-specific evidence, and the QMSR and computer software assurance guidance
absorbed the useful parts instead.

**Public De Novo summaries.** The De Novo review documents for DEN180001 (IDx-DR) and DEN180044
(the Apple Watch ECG app) describe intended use, software documentation, and clinical evidence.
The documented fact is that these public summaries exist and describe a narrow claim with
supporting evidence. The inference, which the documents do not state, is that the summaries are
the visible tip of a design record that reviewers examined and the public never sees. A team that
copies the summary structure without building the record behind it has copied the wrong artifact.

**The QMSR transition.** FDA published the final rule in February 2024 with an effective date two
years later, and stated that manufacturers are expected to update their quality systems by that
date. The documented fact is the rule and the transition period. The inference from the pattern of
industry preparation, visible in conference programs, consultant offerings, and vendor tooling,
is that many small software firms treat the transition as a document-renaming exercise. The
practical risk is that a renamed procedure with no clause mapping fails a process audit even when
its content is adequate.

**Computer software assurance.** The CSA guidance, finalized in September 2025 and revised in
February 2026, marks a documented shift from exhaustive scripted validation of production software
toward risk-based assurance. The inference from industry adoption is that teams over-apply the new
flexibility to device software, which the guidance does not cover, and under-apply it to
quality-system software that carries record-integrity risk.

Two practices distinguish organizations that pass audits from those that survive them. The first
is treating the traceability matrix as a design tool rather than a submission artifact: it is
maintained during design, and a requirement without a test is visible in a weekly review rather
than discovered in a mock audit. The second is treating change control as the default path for
anything that alters behavior, including thresholds, dependency versions, and infrastructure,
rather than as an exception for major releases. Neither practice is required by a specific clause.
Both are what the clauses are for.

## 10. Failure modes and red flags

Design-control failures are rarely dramatic. They look like ordinary engineering decisions made
under schedule pressure: a quick threshold tweak, a test plan written after the test, a review
that happened in a corridor. The table lists the failures that survive until an auditor or a
reviewer finds them.

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Validated build is not the released build | Release notes mention a version absent from the validation report | Evidence is tied to a configuration, not a product name | Unsupported performance claim | Freeze and identify the configuration before validation |
| Threshold changed without a change record | A notebook commit alters a decision point | Change is treated as tuning rather than design | Unverified operating point in the field | Change control covers thresholds, calibration, and retraining |
| Untestable design input | Requirements contain "accurate," "robust," or "intuitive" | No acceptance criterion exists | Verification cannot pass or fail | Rewrite with metric, population, and confidence bound |
| Verification presented as validation | Report says "validated" but no user was involved | Output-meets-input confused with need-meets-use | Claim broader than evidence | Separate protocols, separate reports, separate reviewers |
| Review without a record | Calendar invite exists, minutes do not | Review treated as a meeting | Activity treated as not performed | Minutes with date, attendees, findings, actions, closure |
| Risk controls outside the requirements | Risk file cites controls absent from the input list | Risk and design run in parallel without a link | Control cannot be shown to exist | Every control becomes an input with a verification |
| Orphan requirements | Inputs with no downstream test | Requirements added late or copied from a template | Coverage gap hidden by a high-level percentage | Coverage report by input, not by test count |
| Orphan tests | Tests linked to no requirement | Exploratory or legacy tests retained | Effort spent without evidence value | Map or retire; record the rationale |
| Shallow SOUP inventory | Only direct dependencies listed | Transitive dependencies invisible | Unassessed component in the device | Generate a bill of materials from the build |
| Supplier change treated as purchasing | Vendor upgrade noted only in a ticket | Supplier not classified by device impact | Undocumented device change | Supplier change triggers an impact assessment |
| Design record assembled at the end | File index created in the last month | Records reconstructed from memory | Missing and inconsistent evidence | Index at gate 1; completeness reviewed at every gate |
| Legacy device assumed exempt | No design file for a marketed product | Confusion between original design and changes | Uncontrolled change to a marketed device | Document current configuration; apply controls to changes |
| Labeling not treated as a design output | Label text edited directly in production | Labeling outside the design process | Claim and evidence diverge | Labeling is an output; verify content and comprehension |
| Cloud environment changed without qualification | Runtime or region changed in a console | Infrastructure treated as operations | Behavior differs from the verified environment | Infrastructure as code and change control |
| CAPA closed at the document edit | Effectiveness check absent | Closure measured by paperwork | Recurrence of the same nonconformity | Define effectiveness criteria before closure |

A red flag is a prompt to check a fact, not a verdict. The right response is to add the missing
evidence or record the rationale for the alternative, not to argue that the flag is irrelevant.

## 11. Assumptions and limits

The procedure in Section 5 rests on assumptions. Stating them makes the analysis auditable and
tells a reviewer where the weak points are.

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The intended use is stable through design | Inputs and validation are derived from it | A new market, user, or claim is added late | Version-control the statement; review at every gate |
| Every design input can be made testable | Verification depends on measurable criteria | A requirement describes a quality, not a quantity | Review each input for a metric and a method |
| The released artifact equals the verified artifact | Evidence is configuration-specific | A build, threshold, or dependency changes after verification | Compare build identifiers and hashes at release |
| Validation participants represent intended users | Validation is about use, not only accuracy | Convenience sample or expert-only panel | Compare sample to the intended user and setting |
| The test environment matches production | Behavior depends on runtime | Different hardware, library, or service version | Environment qualification and parity checks |
| Suppliers disclose relevant changes | The manufacturer cannot see inside a supplier | Contract lacks notification; change is silent | Quality agreements and periodic supplier review |
| The QMSR-to-ISO mapping is correct and current | Procedures cite clauses | Clause numbering misread or standard revised | Independent mapping review; check eCFR and ISO |
| Records remain retrievable for the retention period | The record is the evidence | System migration or vendor shutdown | Retrieval test from backup and archive |
| The evidence standard matches the claim | Reviewers test claim against evidence | Claim written after the study | Claim-evidence matrix reviewed by regulatory |
| Reviewers are independent and competent | Review is the challenge function | Same author reviews their own design | Record independence and competence |
| The invoked standard edition is current | Citations must resolve | A new edition or amendment is published | Check the standard body and the FDA recognition list |

The limits of this topic are equally explicit. It does not implement a full ISO 13485 system, and
a quality manual cannot be produced from a chapter. It does not cover hardware design controls,
sterilization, electrical safety, or combination-product manufacturing in detail. It treats EU
Annex IX at a summary level and does not reproduce the technical documentation requirements. It
does not resolve how much validation is enough for a specific claim; that is a judgment made with
regulators and clinical experts. And it cannot tell you whether a particular auditor will accept
a particular record.

## 12. Exercises

The exercises move from recall to application to critique to design. Each states a deliverable,
acceptance criteria, and a time estimate.

### Exercise 1 — Clause map for a real procedure (recall and application, 30 minutes)

Take one quality procedure you can access, or one FDA Form 483 observation from the public
inspection database. Map each step to the ISO 13485 clause and the 21 CFR 820 provision it
implements.

**Deliverable.** A three-column table: procedure step, clause or section, gap or ambiguity.
**Acceptance criteria.** Every citation resolves in the current text; no clause numbers are
invented; at least two genuine gaps are identified; the analysis states which version of the
regulation it used.

### Exercise 2 — Design plan and inputs for an AI device (application, 45 minutes)

Choose an AI ECG or imaging product. Write a one-page design and development plan with four
gates, then 15 design inputs.

**Deliverable.** The plan plus an input table with metric, population, threshold, verification
method, and safety flag.
**Acceptance criteria.** Each gate has entry and exit criteria; every input is falsifiable; at
least three inputs trace to risk controls; at least one input addresses the uninterpretable or
failure output.

### Exercise 3 — Verification or validation (critique, 35 minutes)

Take a published validation report or a vendor white paper. Label each claim as verification,
validation, or neither, and identify the evidence that is missing for the intended-use claim.

**Deliverable.** An annotated claim table plus a one-paragraph assessment.
**Acceptance criteria.** Every claim carries a label and a reason; at least one claim is correctly
identified as neither; the missing evidence is specific enough to commission.

### Exercise 4 — Design record index and score (application, 50 minutes)

Build an index of at least 30 documents for a fictional device, then run the Section 7 script
against the counts, substituting your own dataset for the demo.

**Deliverable.** The index, the script output, and a remediation list.
**Acceptance criteria.** The index has identifiers, revisions, owners, and status; the score is
reproducible; each blocking finding has an owner and a due date.

### Exercise 5 — Change impact assessment (design, 45 minutes)

Using the Section 6 case, write the change record that should have existed for the threshold move
from 0.52 to 0.47.

**Deliverable.** A change request, an impact assessment, an evidence plan, and a regulatory
decision.
**Acceptance criteria.** The assessment covers inputs, outputs, risk, verification, validation,
labeling, and supplier impact; the evidence plan names the operating point and subgroup analysis;
the decision distinguishes change control from a PCCP from a new submission.

### Exercise 6 — SOUP and supplier classification (application, 40 minutes)

For a cloud-deployed model that uses PyTorch, a graphics processing unit runtime, a cloud
inference service, and a monitoring vendor, classify each element and specify its controls.

**Deliverable.** A classification table with required records and notification obligations.
**Acceptance criteria.** Each element has a risk rationale; the software bill of materials depth
is stated; the cloud shared-responsibility boundary is written down; at least one end-of-life
scenario is addressed.

### Exercise 7 — Mock audit (critique, 60 minutes)

Exchange a design record with a peer. Write five findings as an auditor would, each with a clause
reference, objective evidence, severity, and a recommended corrective action.

**Deliverable.** Five findings and a one-page summary judgment.
**Acceptance criteria.** Each finding cites a clause and a specific document; at least one is
substantive rather than documentary; corrective actions distinguish containment from root cause;
the summary states whether the record supports the claim.

### Exercise 8 — QMSR transition plan (design, 75 minutes)

Map ten procedures from a pre-2026 quality system to ISO 13485 clauses, identify what is missing,
and write a 90-day transition plan.

**Deliverable.** A clause map, a gap list, and a plan with owners and dates.
**Acceptance criteria.** The map covers document control, record control, management review,
design control, purchasing, production, nonconforming product, CAPA, internal audit, and training;
the plan addresses the quality manual, clause citations, record retention, and training; residual
risks are stated.

## 13. Self-check answers

**Question 1. What changed on 2 February 2026?** The QMSR took effect. It incorporates
ISO 13485:2016 by reference, restructures 21 CFR Part 820, and moves design-control requirements
from 820.30 into ISO 13485 Clause 7.3. A small number of device-specific record and labeling
provisions remain in Part 820. A quality manual becomes an explicit expectation, and audit
citations change from paragraph numbers to clauses.

**Question 2. Where do design controls live now, and what does 820.10 do?** The requirements are
the ISO 13485 Clause 7.3 requirements, incorporated by reference. Section 820.10 establishes the
quality-system requirement and the applicability of the incorporated clauses. The CFR does not
restate the ISO text, so a procedure must cite the clause it implements rather than quoting a
paragraph that no longer exists.

**Question 3. What belongs in a design and development plan?** Stages, gates, deliverables,
owners, review types and required competence, entry and exit criteria, configuration management,
and the process for changing the plan. The plan is approved before the work it governs and is
itself under change control.

**Question 4. What makes a design input testable?** A metric, a population, the conditions of
measurement, a threshold, a method, and a decision rule. "Sensitivity at least 0.90 with a
two-sided 95 percent lower confidence bound of 0.85 in adults with suspected arrhythmia at the
locked operating point" is testable. "The model should be accurate" is not.

**Question 5. Verification or validation for an AI device?** Verification shows that the output
meets the input: the model achieves its specified sensitivity on the specified test set, with the
specified analysis. Validation shows that the device meets the user need in context: intended
users, intended population, intended setting, and the workflow action that follows the output. A
retrospective accuracy study is verification evidence at best and is neither if the test set is
not controlled.

**Question 6. How is design-input coverage computed, and what does it miss?** Divide the number of
inputs with at least one completed verification by the total number of inputs. In the Section 6
case, 71 divided by 84 is 84.5 percent. It misses whether the test was adequate, whether the
requirement was right, and whether the tested configuration is the released one.

**Question 7. A threshold changes after validation. What happens?** It is a design change. The
change is assessed for impact on inputs, outputs, risk, verification, validation, labeling, and
regulatory status; re-verified at the new operating point; re-assessed for clinical performance
and subgroup effects; approved before release; and recorded. If the device is under a PMA, a
supplement may be required. Communication follows the applicable rules.

**Question 8. SOUP, OTS software, and a supplier are not the same thing.** OTS software is
commercially available and used without modification. SOUP is software whose provenance or
testing evidence is insufficient, a category defined by IEC 62304. A supplier is an organization
that provides a product or service, including a cloud provider. A library can be both OTS and
SOUP, and a cloud service is a supplier relationship even when no software is delivered.

**Question 9. Documentation gap or substantive gap?** A documentation gap means the activity
happened but the record is missing or incomplete, and independent evidence can reconstruct it. A
substantive gap means the activity did not happen or the evidence does not support the claim. The
test is whether an auditor can verify the activity from another controlled source. A missing
review minute with a calendar record, a signed attendance sheet, and closed actions is a
documentation gap; a validation never performed is substantive.

**Question 10. Why is a legacy device treated differently?** Design controls are not applied
retroactively to reconstruct a design completed before the requirement existed. They apply to
changes. The manufacturer documents the current configuration, performs a gap assessment against
Clause 7.3, and applies the full process to the next change. The QMSR does not grandfather records,
so the current design still has to be controlled.

**Rubric for Exercise 5 (change impact assessment).** A strong submission identifies every
affected artifact, states the operating point and the subgroup analysis, updates the labeling and
risk file, and makes an explicit regulatory decision with a reason. A weak submission treats the
change as a configuration edit and omits the clinical performance consequence.

**Rubric for Exercise 7 (mock audit).** A strong set of findings cites specific clauses and
documents, separates documentation from substantive gaps, and proposes corrective actions that
address root cause. A weak set lists generic deficiencies without evidence or confuses a missing
signature with a missing activity.

**Rubric for Exercise 8 (transition plan).** A strong plan maps every procedure to a clause,
identifies the quality manual and record-retention gaps, and assigns owners and dates with
residual risk stated. A weak plan renames documents without changing citations or process.

## 14. Interview questions

**1. What did the QMSR change?** A strong answer names the effective date, the incorporation of
ISO 13485 by reference, the move of design controls to Clause 7.3, and at least one practical
consequence such as the quality manual. A weak answer says it is a harmonization with no material
effect.

**2. Walk me through design controls for a model.** A strong answer follows user needs to inputs
to outputs to verification to validation to transfer, naming the records at each step and where
risk controls enter. A weak answer lists document names without the information flow.

**3. Verification versus validation?** A strong answer gives the output-meets-input and
need-meets-use distinction and applies it to a model with a concrete example. A weak answer uses
the words interchangeably or says validation means a clinical trial.

**4. How do you handle a model change after clearance?** A strong answer describes impact
assessment, re-verification, re-validation where indicated, labeling and risk updates, and the
regulatory route including a PCCP where one exists. A weak answer says the change is deployed
because the code is unchanged.

**5. What is SOUP and how do you control it?** A strong answer defines it, ties the depth of
evaluation to the software safety class, and describes the inventory, anomaly review, and
integration testing. A weak answer calls every dependency SOUP and stops at a list.

**6. What would you look for in a design history file?** A strong answer asks for the plan, the
input and output baselines, review records, verification and validation reports, the traceability
matrix, change records, and the released configuration. A weak answer asks whether the folder
exists.

**7. How do you know validation is sufficient?** A strong answer ties sufficiency to the claim:
the population, setting, users, endpoints, and pre-specified analysis must match the intended use.
A weak answer cites a metric threshold without reference to the claim.

**8. A supplier changes a runtime version. What do you do?** A strong answer treats it as a
potential design change, assesses impact, runs regression and performance tests, and records the
decision. A weak answer says the supplier is responsible.

**9. What is the difference between a QMS certificate and device compliance?** A strong answer
explains that a certificate covers the quality system and its scope, while device compliance
requires design, clinical, and labeling evidence for the specific product. A weak answer treats
the certificate as a product approval.

## 15. Key terms

| Term | One-line meaning |
|---|---|
| Design control | The process and records linking user needs, inputs, outputs, review, verification, validation, transfer, and change |
| Design input | A measurable requirement that design output must satisfy |
| Design output | Specifications, software, labeling, and other design results with acceptance criteria |
| Design review | A documented, systematic examination of the design against requirements at a defined stage |
| Verification | Objective evidence that design output meets design input |
| Validation | Objective evidence that the device meets user needs and intended use in context |
| Design transfer | Translation of design output into reproducible production or deployment |
| Design change | A controlled change to baselined design output |
| Design history file (DHF) | The record set demonstrating that the design was developed in accordance with the plan |
| Design and development file | The ISO 13485 Clause 7.3.10 record set, successor concept to the DHF |
| Device master record | The production specification set for the device |
| Device history record | The records showing a specific unit was produced to the master record |
| Quality management system (QMS) | The organization, procedures, processes, and resources used to manage quality |
| QMSR | The FDA Quality Management System Regulation at 21 CFR Part 820, effective 2 February 2026 |
| QSR | The pre-2026 version of 21 CFR Part 820 |
| ISO 13485:2016 | The medical device quality management system standard incorporated by the QMSR |
| Clause 7.3 | The ISO 13485 design and development clause that now carries design controls |
| 820.10 | The QMSR section establishing the quality system and the applicability of incorporated clauses |
| CAPA | Corrective and preventive action |
| SOUP | Software of unknown provenance, a category defined by IEC 62304 |
| OTS software | Commercially available software used without modification |
| Computer software assurance (CSA) | Risk-based assurance for production and quality-system software |
| 21 CFR Part 11 | US requirements for electronic records and electronic signatures |
| Traceability matrix | The controlled mapping from needs to inputs to outputs to verification and validation |
| Risk management file | The records produced by the ISO 14971 process |
| Usability engineering file | The records produced by the IEC 62366-1 process |
| PCCP | Predetermined change control plan |
| Notified body | An EU organization designated to assess conformity with the MDR |
| Technical documentation | The EU dossier demonstrating conformity under Annex IX |
| Audit trail | A secure, time-stamped record of electronic record creation, modification, and deletion |

## 16. Sources

All URLs were accessed on 25 October 2026. Verify the current version before relying on any
source. Regulatory dates and citations change.

1. US Food and Drug Administration. "Quality Management System Regulation (QMSR)." Final rule,
   89 FR 7496, 2 February 2024, effective 2 February 2026.
   https://www.fda.gov/medical-devices/postmarket-requirements-devices/quality-management-system-regulation-qmsr
2. US Code of Federal Regulations. Title 21, Part 820, Quality Management System Regulation.
   https://www.ecfr.gov/current/title-21/chapter-I/subchapter-H/part-820
3. US Code of Federal Regulations. Title 21, Part 11, Electronic Records; Electronic Signatures.
   https://www.ecfr.gov/current/title-21/chapter-I/subchapter-A/part-11
4. International Organization for Standardization. ISO 13485:2016, Medical devices — Quality
   management systems — Requirements for regulatory purposes. https://www.iso.org/standard/59752.html
5. International Organization for Standardization. ISO 14971:2019, Medical devices — Application
   of risk management to medical devices. https://www.iso.org/standard/72704.html
6. International Electrotechnical Commission. IEC 62304:2006+A1:2015, Medical device software —
   Software life cycle processes. https://webstore.iec.ch/en/publication/22794
7. International Electrotechnical Commission. IEC 62366-1:2015+A1:2020, Medical devices —
   Application of usability engineering to medical devices. https://webstore.iec.ch/
8. US Food and Drug Administration. "Design Control Guidance for Medical Device Manufacturers,"
   1996. https://www.fda.gov/regulatory-information/search-fda-guidance-documents/design-control-guidance-medical-device-manufacturers
9. US Food and Drug Administration. "Computer Software Assurance for Production and Quality System
   Software." Final guidance, 24 September 2025; revised 3 February 2026.
   https://www.fda.gov/regulatory-information/search-fda-guidance-documents/computer-software-assurance-production-and-quality-management-system-software
10. US Food and Drug Administration. "Off-The-Shelf Software Use in Medical Devices." Guidance.
    https://www.fda.gov/regulatory-information/search-fda-guidance-documents/off-shelf-software-use-medical-devices
11. US Food and Drug Administration. "Cybersecurity in Medical Devices: Quality System
    Considerations and Content of Premarket Submissions." Final guidance, September 2023.
    https://www.fda.gov/regulatory-information/search-fda-guidance-documents/cybersecurity-medical-devices-quality-system-considerations-and-content-premarket-submissions
12. US Food and Drug Administration. "Content of Premarket Submissions for Device Software
    Functions." Final guidance, June 2023.
    https://www.fda.gov/regulatory-information/search-fda-guidance-documents/content-premarket-submissions-device-software-functions
13. European Parliament and Council. Regulation (EU) 2017/745 on medical devices, Annex IX.
    https://eur-lex.europa.eu/eli/reg/2017/745/oj
14. Medical Device Coordination Group. Guidance on qualification and classification of software,
    MDCG 2019-11 Rev.1, June 2025. https://health.ec.europa.eu/medical-devices-topics-interest/medical-devices-guidance_en
15. US Food and Drug Administration. Inspection, compliance, enforcement, and criminal
    investigations resources, including Form 483 observations and warning letters.
    https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations
16. US Food and Drug Administration. Medical device recall database.
    https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfRes/res.cfm
17. US Food and Drug Administration. De Novo classification request DEN180001 (IDx-DR).
    https://www.accessdata.fda.gov/cdrh_docs/reviews/DEN180001.pdf
18. US Food and Drug Administration. De Novo classification request DEN180044 (Apple ECG app).
    https://www.accessdata.fda.gov/cdrh_docs/reviews/DEN180044.pdf
19. US Food and Drug Administration. Safety communication and recall notices concerning
    polyester-based polyurethane foam in certain Philips Respironics devices, 2021. See the FDA
    recall database for the current record.
20. US Food and Drug Administration. Quality System Regulation final rule, 61 FR 52602,
    7 October 1996.

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`04_gmlp-and-pccp.md`](04_gmlp-and-pccp.md)
- Next topic: [`06_iso-14971-risk-management.md`](06_iso-14971-risk-management.md)
- Related topic: [`07_iec-62304-software-lifecycle.md`](07_iec-62304-software-lifecycle.md)
- Related topic: [`03_fda-pathways.md`](03_fda-pathways.md)
- Related topic: [`09_the-evidence-package.md`](09_the-evidence-package.md)
- Survey file: [`../../course/14_medical_devices_samd_regulation.md`](../../course/14_medical_devices_samd_regulation.md)
- Labs: [`../../labs/`](../../labs/)
- Lab 14.3 risk management file: [`../../labs/`](../../labs/)
- Lab 14.4 software requirements and traceability: [`../../labs/`](../../labs/)

The module index is the authoritative map. If a sibling file is not yet present, the index marks
it as planned; do not infer content from the link text.
