# IEC 62304 Software Lifecycle

> Module 14 — Medical Devices, SaMD, and Regulation
> Topic 07 of 10
> Estimated study time: 3.5 hours
> Prerequisites: 01, 05, 06
> Version 2026.3

## 1. Why this topic exists

Medical device software fails in ways that are difficult to see in a demo. A model returns a plausible score with a
corrupted input. A pump accepts a dose outside the safe range because a unit conversion was verified with round numbers.
An imaging viewer displays a prior study beside the current one, and the clinician acts on the wrong image. A cloud
service changes a dependency, and a calculation silently changes precision. These failures are not usually caused by a
lack of intelligence. They are caused by missing lifecycle discipline.

IEC 62304 is the international standard for medical device software lifecycle processes. It defines what a manufacturer
must do to plan, develop, verify, release, maintain, and retire software. It is not a coding style guide, a
model-validation standard, or a substitute for risk management. It is the process framework that makes software evidence
reproducible. A team that can show a requirement, a design decision, a verification test, a risk control, and a released
version linked together has a defensible lifecycle. A team that cannot is asking a reviewer to trust a demo.

The standard matters commercially because regulators expect it. The US Food and Drug Administration (FDA) recognizes IEC
62304:2006+A1:2015, and software documentation is a standard part of premarket submissions for device software. The
European Union Medical Device Regulation (MDR) requires software to be developed according to the state of the art, and
harmonized standards are the usual way to demonstrate conformity. A quality management system (QMS) auditor will ask for
the software development plan, the requirements, the architecture, the verification records, the configuration
management records, and the problem-resolution records. If those records do not exist, the product may still work, but
the manufacturer cannot show that it works.

The standard also matters for data science teams because machine learning adds new failure modes without removing the
old ones. A model is a software item with requirements, an architecture, a training pipeline, a configuration, and a
release. The training data are an input, the model weights are an output, and the evaluation is verification and
validation evidence. A model update is a software change. A model that is retrained in production is a maintained
software system. IEC 62304 does not tell you how to validate a model, but it tells you where the model fits in the
lifecycle and what records you need.

The cost of ignoring the lifecycle is not only a failed audit. It is rework. A team that writes requirements after the
code cannot trace a test to a requirement. A team that skips architecture cannot show that a safety-critical function is
isolated from a non-critical one. A team that does not control configuration cannot reproduce a released version. A team
that does not evaluate third-party software cannot answer a vulnerability question. These gaps surface at the worst
time: during a submission review, a customer security questionnaire, or a recall investigation.

This topic gives you the vocabulary and procedure to apply IEC 62304 to software-only devices, embedded software, and
AI-enabled systems. It focuses on the safety classification, the process requirements, the documentation set, and the
evidence a reviewer will expect. It does not replace the standard text, a quality-system procedure, or a regulatory
review. It gives you enough structure to build a lifecycle that survives contact with an auditor.

## 2. Learning outcomes

By the end of this topic, you can:

1. Explain the purpose and scope of IEC 62304:2006+A1:2015 and how it relates to ISO 14971 and IEC 62366-1.
2. Assign a software safety class (A, B, or C) using the possible consequence of failure and external risk controls.
3. Write a software development plan that names activities, deliverables, and responsibilities by safety class.
4. Trace a user need to a software requirement, a design element, a verification test, and a risk control.
5. Distinguish unit verification, integration testing, and system testing, and explain what each can and cannot prove.
6. Apply configuration management and problem-resolution requirements to a released software version.
7. Evaluate software of unknown provenance (SOUP) and document its risks and anomalies.
8. Critique a software lifecycle file that has tests but no requirements, or requirements but no risk traceability.

The outcomes map to the exercises and self-check items as follows.

| Outcome | Exercise | Self-check |
|---|---|---|
| 1 | Exercise 1 | Questions 1, 2 |
| 2 | Exercise 2 | Question 3 |
| 3 | Exercise 3 | Question 4 |
| 4 | Exercise 4 | Question 5 |
| 5 | Exercise 5 | Question 6 |
| 6 | Exercise 6 | Question 7 |
| 7 | Exercise 7 | Question 8 |
| 8 | Exercise 8 | Questions 9, 10 |

## 3. The core mental model

IEC 62304 is a set of processes applied to a software system. The processes are not a waterfall. They can be iterative,
incremental, or agile, as long as the outputs exist and the traceability is maintained. The standard cares about what
you can show, not which project management framework you use.

```text
System requirements and intended use
                |
                v
[1] Software development plan
                |
                v
[2] Software requirements analysis
                |
                v
[3] Software architectural design
                |
                v
[4] Software detailed design
                |
                v
[5] Unit implementation and verification
                |
                v
[6] Integration and integration testing
                |
                v
[7] System testing
                |
                v
[8] Software release
                |
                v
[9] Maintenance and problem resolution
                |
                v
[10] Configuration management, risk management, documentation
```

The vertical axis is the software system. The horizontal axis is the evidence. At every stage, the question is: what
requirement is being addressed, what design element implements it, what test verifies it, what risk control depends on
it, and what version was released? If you can answer those five questions for every safety-related function, you have a
lifecycle. If you can answer them only for the happy path, you have a demo.

The first decision is the software safety class. IEC 62304 uses three classes. Class A applies when the software cannot
contribute to a hazardous situation, or when it can but the risk is acceptable after external risk controls. Class B
applies when the software can contribute to a hazardous situation with unacceptable residual risk, and the possible harm
is non-serious injury. Class C applies when the possible harm is death or serious injury. The class is not a property of
the code alone. It is a property of the software in its system, after external risk controls.

The second decision is the software system boundary. A software system can include firmware, an operating system,
libraries, a cloud service, and a model. The manufacturer must define what is in scope and what is SOUP. A library that
performs a safety-related calculation is not outside the lifecycle just because it was downloaded. A model trained by a
third party is not outside the lifecycle just because the manufacturer did not write the training code. The boundary is
a risk decision, not a procurement decision.

The third decision is the process rigor. Class A requires planning, requirements, unit implementation, release,
maintenance, configuration management, and problem resolution. Class B adds architectural design, integration testing,
system testing, and software risk management. Class C adds detailed design and tighter verification. The standard's
Annex A provides a useful table of which processes apply to which class. A manufacturer can apply more rigor than the
class requires, but not less.

The fourth decision is the relationship between software risk management and system risk management. ISO 14971
identifies hazards and risk controls at the system level. IEC 62304 requires the software development process to
contribute to those controls. A software requirement may exist because a risk control needs it. A software test may
exist because a risk control needs evidence. A software architecture may isolate a safety function because a risk
control requires independence. The software risk file should link to the system risk file, not duplicate it.

The fifth decision is the maintenance model. A released device is not frozen forever. Security patches, bug fixes, model
updates, and compatibility changes all require maintenance. IEC 62304 requires a maintenance process, problem-resolution
process, and configuration management. A change that is small in code can be large in risk. A change to a dependency can
affect a safety function. A change to a model can change performance. The maintenance plan should define how changes are
classified, verified, approved, and released.

A useful shorthand is: class sets rigor, requirements set scope, architecture sets isolation, tests set evidence,
configuration sets identity, and maintenance sets lifetime. If any one is missing, the lifecycle is incomplete. A team
with strong tests but no requirements cannot show coverage. A team with strong requirements but no configuration cannot
reproduce a release. A team with strong configuration but no maintenance cannot respond to a vulnerability.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Source |
|---|---|---|---|
| IEC 62304 | International standard for medical device software lifecycle processes | Treating it as a coding standard or a model-validation standard | IEC 62304:2006+A1:2015 |
| Software safety class | Class A, B, or C based on the possible consequence of software failure | Confusing it with the US or EU device class | IEC 62304 Clause 4.3 |
| Class A | Software cannot contribute to a hazardous situation, or risk is acceptable after external controls | Assuming Class A means no documentation | IEC 62304 Annex A |
| Class B | Software can contribute to a hazardous situation with non-serious injury | Assuming Class B is always moderate risk | IEC 62304 Annex A |
| Class C | Software can contribute to death or serious injury | Assuming Class C requires a PMA | IEC 62304 Annex A |
| Software system | The software items, interfaces, and environment within the manufacturer's defined boundary | Excluding cloud services or libraries without analysis | IEC 62304 Clause 3 |
| Software item | A software unit or group of units that can be treated as a unit for lifecycle purposes | Treating a model as outside the software system | IEC 62304 Clause 3 |
| Software unit | A separately testable software element | Testing only at the system level | IEC 62304 Clause 3 |
| SOUP | Software of unknown provenance, developed by a third party and not under the manufacturer's lifecycle control | Assuming open-source software is automatically SOUP or automatically safe | IEC 62304 Clause 3 |
| Software development plan | The plan describing the processes, activities, deliverables, and responsibilities | Writing a plan that is not followed | IEC 62304 Clause 5.1 |
| Software requirements | Requirements derived from system requirements, risk controls, and user needs | Writing requirements as a feature list | IEC 62304 Clause 5.2 |
| Architectural design | The structure of the software, including items, interfaces, and isolation | Skipping architecture because the code is small | IEC 62304 Clause 5.3 |
| Detailed design | The internal design of each software item | Writing detailed design after the code | IEC 62304 Clause 5.4 |
| Unit verification | Verification that a unit meets its detailed design and requirements | Calling a code review a unit test | IEC 62304 Clause 5.5 |
| Integration testing | Testing that software items interact correctly | Testing only the final system | IEC 62304 Clause 5.6 |
| System testing | Testing the software system against system requirements | Testing only model accuracy | IEC 62304 Clause 5.7 |
| Software release | The controlled delivery of a software version | Releasing from a developer laptop | IEC 62304 Clause 5.8 |
| Maintenance | Processes for changes, support, and retirement after release | Treating maintenance as outside the QMS | IEC 62304 Clause 6 |
| Problem resolution | Process for identifying, investigating, and resolving problems | Closing tickets without root-cause analysis | IEC 62304 Clause 9 |
| Configuration management | Process for identifying, controlling, and tracking software versions | Relying on Git tags alone | IEC 62304 Clause 8 |
| Software risk management | Software contribution to system risk management | Duplicating the ISO 14971 file | IEC 62304 Clause 7 |
| Verification | Confirmation that a requirement was fulfilled | Confusing it with validation | ISO 13485 / IEC 62304 |
| Validation | Confirmation that user needs and intended use were met | Assuming unit tests validate clinical use | ISO 13485 |
| Traceability | The ability to link requirements, design, tests, risks, and releases | A spreadsheet with broken links | IEC 62304 / QMS |
| Legacy software | Software developed before the current lifecycle standard | Assuming legacy software needs no assessment | IEC 62304 Clause 4.4 |
| IEC 82304-1 | Health software product standard for software-only products | Treating it as a substitute for IEC 62304 | IEC 82304-1:2016 |
| IEC 81001-5-1 | Health software security lifecycle standard | Treating security as a one-time penetration test | IEC 81001-5-1:2021 |
| IEC 62366-1 | Usability engineering standard | Treating usability as a UI polish task | IEC 62366-1:2015+A1:2020 |

Two terms need more detail. The first is SOUP. The standard defines SOUP as software developed by a third party that is
not under the manufacturer's lifecycle control. The manufacturer must identify the SOUP, specify its functional and
performance requirements, verify it in the intended environment, and evaluate its published anomalies. A library with a
known defect in a safety-related function is not acceptable just because it is widely used. A model supplied by a vendor
is SOUP if the manufacturer cannot control its development. The manufacturer must still validate the model in the
intended use.

The second is legacy software. Clause 4.4 provides a path for software developed before the standard or without full
lifecycle records. The manufacturer must perform a gap assessment, identify the risks, and apply the required processes
to the extent necessary. Legacy status is not a permanent exemption. A change to legacy software can bring it into the
current lifecycle. A security vulnerability can require a retrospective risk assessment. A reviewer will expect a
documented rationale, not a claim that the software is old.

A third term, "software safety class," is often confused with the regulatory device class. A Class II device can contain
Class C software. A Class III device can contain Class A software. The software class is about the possible consequence
of a software failure in the system, not about the regulatory controls for the device. A team that writes "Class II
software" is using the wrong vocabulary.

Finally, "verification" and "validation" are not interchangeable. Verification asks whether the software was built
right: does it meet its requirements? Validation asks whether the right software was built: does it meet user needs and
intended use in the real environment? A model can pass verification (it produces the specified output) and fail
validation (the output does not improve clinical decisions). IEC 62304 is primarily a verification and lifecycle
standard; clinical validation is a separate body of evidence.

## 5. The mechanics, step by step

IEC 62304 is applied through a sequence of planning, design, verification, release, and maintenance activities. The
sequence is not a mandate for waterfall development. Agile teams can satisfy it with a backlog, a definition of done,
automated tests, and a configuration baseline, provided the required outputs exist and the traceability is maintained.

### Step 1 — Define the software system boundary

**Input.** System architecture, device description, interfaces, third-party components, and intended use.

**Output.** A software system boundary diagram and a list of software items, including SOUP.

**Rationale.** The boundary determines what is under the manufacturer's lifecycle control. If a cloud service, a model,
or a library performs a safety-related function, it belongs in the analysis even if the manufacturer did not write it. A
boundary that excludes critical dependencies is a risk-analysis gap, not a scope simplification.

**Procedure.** (1) Draw the system context. (2) Identify the software items and their interfaces. (3) Mark each item as
developed in-house, SOUP, or legacy. (4) Identify the hardware and network dependencies. (5) Record the boundary in the
software development plan.

**Branch points.** A standalone SaMD has a simpler boundary than an embedded system. A SiMD system includes firmware,
device drivers, and hardware interfaces. A cloud system includes the application, the data pipeline, the model, and the
infrastructure dependencies. A legacy system may need a retrospective boundary analysis.

**Edge cases.** Open-source libraries are SOUP unless the manufacturer controls their development. A model trained by a
vendor is SOUP if the manufacturer cannot control the training process. A software development kit supplied by a
hardware vendor may be a component with its own lifecycle. A mobile operating system is part of the environment, not
necessarily a software item, but its behavior can affect safety.

### Step 2 — Assign the software safety class

**Input.** System risk analysis, software functions, and external risk controls.

**Output.** A documented safety class for the software system and, where justified, for individual software items.

**Rationale.** The class sets the required process rigor. The classification is based on the possible consequence of
software failure after external risk controls are applied. The probability of software failure is generally treated as
one for classification purposes, because software failures are systematic rather than random. External controls such as
hardware interlocks, independent alarms, or clinical procedures can reduce the class, but only if they are independent
and effective.

**Procedure.** (1) Identify the hazardous situations to which software can contribute. (2) Identify the harm and
severity. (3) Identify external risk controls. (4) Assess residual risk after those controls. (5) Assign Class A, B, or
C. (6) Document the rationale and the assumptions about external controls.

**Branch points.** No contribution to a hazardous situation, or acceptable residual risk after external controls, is
Class A. Contribution with non-serious injury is Class B. Contribution with death or serious injury is Class C. A
software item can be assigned a lower class only with a documented architectural rationale and verified segregation.

**Edge cases.** A Class A item inside a Class C system must be isolated so that its failure cannot propagate. A model
that only displays information may be Class A if a clinician can independently verify the output, but the display path
itself may be Class C if it can hide a critical result. A cybersecurity failure can create a hazardous situation even if
the intended function is safe.

### Step 3 — Write the software development plan

**Input.** The boundary, the safety class, the QMS procedures, and the project plan.

**Output.** A software development plan that names processes, activities, deliverables, responsibilities, and standards.

**Rationale.** The plan is the contract for the lifecycle. It tells the team and the auditor what will be produced and
who owns it. A plan that is written after the work is a reconstruction, not a plan. The standard requires the plan to be
documented and maintained.

**Procedure.** (1) Name the development model and iterations. (2) List the activities by safety class. (3) List the
deliverables and their owners. (4) Name the coding, testing, and configuration standards. (5) Define the review and
approval process. (6) Define the problem-resolution and change-control process. (7) Define the documentation and record
retention.

**Branch points.** A Class A plan can be lighter but still needs planning, requirements, implementation, verification,
release, configuration, and problem resolution. A Class B or C plan adds architecture, integration, system testing, and
software risk management. A legacy plan needs a gap assessment.

**Edge cases.** A plan that says "agile" without naming deliverables is not a plan. A plan that relies on a tool without
defining the records is incomplete. A plan that does not cover maintenance will fail at the first security patch.

### Step 4 — Analyze software requirements

**Input.** System requirements, user needs, risk controls, regulatory requirements, and interface specifications.

**Output.** A software requirements specification with unique identifiers, acceptance criteria, and traceability.

**Rationale.** Requirements define what the software must do and how well. They must be complete, unambiguous,
verifiable, and free of conflict. A requirement that cannot be tested is not a requirement. A requirement that is not
linked to a system need or a risk control is an orphan.

**Procedure.** (1) Assign an identifier to each requirement. (2) State the behavior and the acceptance criterion. (3)
Identify the source: system requirement, user need, risk control, or regulation. (4) Identify the verification method.
(5) Review for completeness and conflict. (6) Baseline the requirements.

**Branch points.** Functional requirements describe behavior. Performance requirements describe timing, accuracy, and
throughput. Safety requirements describe risk controls. Interface requirements describe data exchange. Security
requirements describe access, integrity, and availability.

**Edge cases.** A requirement that says "the model shall be accurate" is not verifiable. A requirement that specifies an
algorithm rather than an outcome can over-constrain the design. A requirement that depends on a clinical workflow must
be validated with users. A requirement that is derived from a risk control must be linked to the risk file.

### Step 5 — Design the software architecture

**Input.** The requirements and the system architecture.

**Output.** An architectural design describing software items, interfaces, data flows, and isolation.

**Rationale.** Architecture makes it possible to reason about safety, security, and change. It shows how a
safety-critical function is isolated from a non-critical one, how data flow through the system, and where a failure can
propagate. A system without an architecture is a collection of code that is difficult to verify and change.

**Procedure.** (1) Decompose the system into software items. (2) Define the responsibility of each item. (3) Define the
interfaces and data flows. (4) Identify safety-critical items and isolation mechanisms. (5) Identify SOUP and external
interfaces. (6) Review the architecture against the requirements and risks.

**Branch points.** A monolithic application may need internal modules with clear boundaries. A microservice architecture
needs interface contracts and failure handling. An embedded system needs timing and resource analysis. A model pipeline
needs data provenance and versioning.

**Edge cases.** An architecture that isolates a Class C function in a separate process can support a lower class for
other items only if the segregation is verified. A cloud dependency can introduce availability and latency risks. A
model served by an external API introduces a supply-chain and privacy risk. A shared database can create coupling that
undermines isolation.

### Step 6 — Produce the detailed design

**Input.** The architecture and the requirements.

**Output.** A detailed design for each software item, including algorithms, data structures, interfaces, and error
handling.

**Rationale.** Detailed design is required for Class C software and is good practice for Class B. It shows how each item
implements its requirements, including boundary conditions and failure handling. It is the bridge between architecture
and code.

**Procedure.** (1) Describe each unit's purpose. (2) Define inputs, outputs, and data structures. (3) Define algorithms
and calculations. (4) Define error handling and boundary behavior. (5) Define the unit's interfaces. (6) Review the
design against the requirements.

**Branch points.** A simple unit may need only a short design description. A complex algorithm may need a state machine,
a sequence diagram, or a numerical analysis. A model inference unit may need preprocessing, postprocessing, and
threshold logic described separately.

**Edge cases.** A detailed design that is written after the code is documentation, not design. A design that does not
address invalid input is incomplete. A design that relies on undocumented library behavior is fragile. A design that
changes a safety-related calculation needs a risk review.

### Step 7 — Implement and verify software units

**Input.** The detailed design and the coding standards.

**Output.** Source code, unit tests, verification records, and a configured build.

**Rationale.** Unit verification shows that each unit meets its design and requirements. It catches defects close to
their source, before integration makes them harder to isolate. The level of unit verification scales with the safety
class.

**Procedure.** (1) Implement the unit according to the design. (2) Perform code review. (3) Write unit tests for normal,
boundary, and invalid inputs. (4) Run static analysis. (5) Record the results and the defects. (6) Update the design if
the implementation changes.

**Branch points.** Class A requires implementation and verification, but the depth can be lighter. Class B requires unit
verification. Class C requires unit verification with acceptance criteria and more rigorous coverage. A safety-related
calculation needs boundary-value and numerical tests.

**Edge cases.** A unit test that uses the same logic as the implementation can mask a defect. A test that covers only
the happy path is incomplete. A test that depends on a live external service is not repeatable. A test that is not
linked to a requirement does not support traceability.

### Step 8 — Integrate and test software items

**Input.** Verified units, the architecture, and the interface specifications.

**Output.** An integrated software system and integration test records.

**Rationale.** Integration testing verifies that software items interact correctly. Many failures appear only at the
interfaces: unit conversions, data formats, timing, error propagation, and state transitions. A system that passes unit
tests can still fail when the pieces are connected.

**Procedure.** (1) Define the integration sequence. (2) Define the interfaces and test data. (3) Test normal, boundary,
and failure interactions. (4) Test error propagation and recovery. (5) Record the results. (6) Update the architecture
if the integration reveals a design issue.

**Branch points.** A monolithic system may integrate in one step. A distributed system needs contract testing and
failure injection. An embedded system needs timing and resource testing. A model pipeline needs end-to-end data tests.

**Edge cases.** An integration test that uses mocked interfaces may miss a real contract mismatch. A test that does not
exercise a failure path may miss an error-handling defect. A test that depends on a specific data set may not
generalize. A test that is not repeatable cannot support a release decision.

### Step 9 — Perform system testing

**Input.** The integrated system, the system requirements, and the intended use.

**Output.** System test records and a verification summary.

**Rationale.** System testing verifies the software system against its requirements. It is broader than integration
testing because it exercises the complete system in its intended environment, including interfaces to hardware, users,
and external systems. It is not the same as clinical validation.

**Procedure.** (1) Derive system test cases from requirements. (2) Include normal, boundary, and failure scenarios. (3)
Test in the intended environment. (4) Test performance, usability, and security requirements. (5) Record the results and
the deviations. (6) Trace each result to a requirement.

**Branch points.** A standalone SaMD may be tested in a simulated clinical environment. A SiMD system needs device-level
testing. A cloud system needs load, latency, and failure testing. An AI system needs performance testing across
subgroups.

**Edge cases.** A system test that uses a developer build is not a release test. A test that uses a non-representative
data set is not a clinical performance test. A test that ignores the user interface is incomplete. A test that passes
but has unexplained deviations needs investigation.

### Step 10 — Release the software

**Input.** The verified system, the configuration baseline, and the release approval.

**Output.** A released software version, release notes, and a distribution record.

**Rationale.** Release is a controlled event. The manufacturer must know what is in the release, what was verified, what
changed, and who approved it. A release without a configuration baseline cannot be reproduced or supported.

**Procedure.** (1) Freeze the configuration. (2) Verify the build from the baseline. (3) Confirm the verification and
validation status. (4) Prepare the release notes and labeling. (5) Obtain approval. (6) Distribute and record the
version.

**Branch points.** A cloud release may be continuous, but each deployed version still needs a baseline and approval. A
local release needs installation instructions and rollback. A security patch may use an expedited release process. A
model update may need a PCCP check.

**Edge cases.** A release from a developer branch is not controlled. A release that omits a known defect from the notes
is misleading. A release that changes a dependency without re-verification is uncontrolled. A release that cannot be
rolled back is a risk.

### Step 11 — Maintain the software

**Input.** Released versions, complaints, security advisories, and change requests.

**Output.** Maintenance records, updated versions, and retirement records.

**Rationale.** Maintenance covers the life of the device after release. It includes corrective changes, adaptive
changes, perfective changes, and security patches. The maintenance process must classify the change, assess the risk,
verify the change, and update the documentation.

**Procedure.** (1) Log the change request or problem. (2) Classify the change. (3) Assess the impact on requirements,
architecture, risk, and verification. (4) Implement and verify. (5) Update the configuration and documentation. (6)
Release and monitor.

**Branch points.** A corrective change fixes a defect. An adaptive change responds to a new environment. A perfective
change improves performance. A security change addresses a vulnerability. Each has a different risk profile.

**Edge cases.** A "small" change to a safety-related calculation can require full verification. A dependency update can
change behavior. A model update can change performance. A change that is not documented is a configuration failure.

### Step 12 — Manage configuration

**Input.** Source code, build scripts, dependencies, models, data, and documentation.

**Output.** A configuration management plan, baselines, and change records.

**Rationale.** Configuration management ensures that the released product can be reproduced and that changes are
controlled. It covers source code, binaries, models, training data, build tools, and documentation. A version number
without a reproducible build is not configuration management.

**Procedure.** (1) Identify configuration items. (2) Define the naming and versioning scheme. (3) Define the baseline
and change-control process. (4) Control access and builds. (5) Record the composition of each release. (6) Maintain the
software bill of materials.

**Branch points.** A simple project may use Git tags and a build manifest. A regulated device needs a controlled build
environment and records. A model-based system needs data and weight versioning. A cloud system needs
infrastructure-as-code versioning.

**Edge cases.** A model file without the training data version is not reproducible. A build that depends on a floating
dependency is not reproducible. A configuration item that is not under change control can change without review. A
software bill of materials that omits transitive dependencies is incomplete.

### Step 13 — Resolve problems

**Input.** Defects, complaints, test failures, and security findings.

**Output.** Problem records, root-cause analysis, corrective actions, and verification of the fix.

**Rationale.** Problem resolution is a lifecycle process, not a help-desk function. The manufacturer must investigate
the problem, determine its cause, assess its impact, fix it, verify the fix, and prevent recurrence. A problem that is
closed without root-cause analysis is likely to recur.

**Procedure.** (1) Record the problem. (2) Classify severity and safety impact. (3) Investigate the cause. (4) Identify
the affected versions. (5) Implement and verify the correction. (6) Update the risk file and documentation. (7) Monitor
for recurrence.

**Branch points.** A safety-related problem may require a field action. A security problem may require a coordinated
disclosure. A performance problem may require a model update. A documentation problem may require a labeling correction.

**Edge cases.** A bug that cannot be reproduced may still be real. A problem that affects only one customer may reveal a
broader configuration issue. A fix that introduces a new failure mode is not a fix. A problem that is closed because the
customer stopped complaining is not resolved.

### Step 14 — Evaluate SOUP and security

**Input.** The SOUP inventory, vendor documentation, vulnerability feeds, and the security risk assessment.

**Output.** A SOUP evaluation, a security risk assessment, and a monitoring plan.

**Rationale.** SOUP and security are lifecycle concerns, not procurement afterthoughts. The manufacturer must know what
third-party software is in the device, what it does, what anomalies it has, and how vulnerabilities will be monitored. A
SOUP component with a known critical vulnerability can make the whole device unsafe.

**Procedure.** (1) Inventory the SOUP and versions. (2) Specify functional and performance requirements. (3) Evaluate
published anomalies. (4) Verify the SOUP in the intended environment. (5) Monitor vulnerabilities. (6) Define the patch
and disclosure process.

**Branch points.** A library with a safety-related function needs full evaluation. A library with a non-safety function
may need a lighter evaluation. A model supplied by a vendor needs performance validation. A cloud service needs a
security and availability assessment.

**Edge cases.** A widely used library is not automatically safe. A library that is no longer maintained is a long-term
risk. A vulnerability that is not exploitable in the device context may still need documentation. A patch that changes
behavior needs re-verification.

### Decision table: safety class requirements

| Activity | Class A | Class B | Class C |
|---|---|---|---|
| Development plan | Required | Required | Required |
| Requirements analysis | Required | Required | Required |
| Architectural design | Not required | Required | Required |
| Detailed design | Not required | Not required | Required |
| Unit implementation and verification | Required | Required | Required, tighter |
| Integration testing | Not required | Required | Required |
| System testing | Not required | Required | Required |
| Software risk management | Not required | Required | Required |
| Configuration management | Required | Required | Required |
| Problem resolution | Required | Required | Required |
| Release | Required | Required | Required |
| Maintenance | Required | Required | Required |

### Decision table: test level

| Test level | Question answered | Typical evidence | Cannot prove |
|---|---|---|---|
| Unit | Does the unit meet its design? | Unit tests, code review | System behavior |
| Integration | Do items interact correctly? | Interface tests, failure injection | Clinical benefit |
| System | Does the system meet requirements? | System test report | Real-world effectiveness |
| Validation | Does it meet user needs in context? | Usability, clinical study | Code correctness |

### Decision table: SOUP control

| SOUP risk | Evaluation | Monitoring | Documentation |
|---|---|---|---|
| Safety-related function | Full requirements, anomalies, environment | Active vulnerability feed | Inventory, evaluation, tests |
| Non-safety support | Basic function and version | Periodic review | Inventory |
| Unmaintained | Replacement analysis | Risk acceptance | Rationale |
| Vendor model | Performance validation | Drift and update notice | Contract, validation, version |

The procedure is not a substitute for the standard. It is a map of the evidence a reviewer will expect. The next section
shows how the pieces fit in a realistic case.

## 6. Worked example

The product and sponsor in this case are hypothetical. The numbers are constructed to show how a software safety
classification and lifecycle evidence interact. They are not a real clearance, recall, or audit finding.

### 6.1 Product and intended use

InfuSafe is the software component of a large-volume infusion pump. It calculates the infusion rate from a
clinician-entered dose, patient weight, drug concentration, and delivery duration; it checks the result against a
hospital-configured dose-error-reduction library; and it controls the pump motor through firmware. The software is
embedded in the pump. It is SiMD, not SaMD, because it is part of a hardware medical device and controls its medical
function.

The intended-use statement for the pump is:

"InfuSafe is intended for use by trained clinicians to deliver intravenous fluids and medications at a programmed rate
to adult and pediatric patients. The pump includes dose-error-reduction software that alerts the user when a programmed
dose is outside the hospital-configured limit and requires confirmation before delivery. The pump is not intended for
use in neonatal intensive care, for epidural delivery, or for blood products."

The software has four safety-related functions: dose calculation, library limit checking, motor control, and alarm
annunciation. The pump has a hardware occlusion sensor and a separate air-in-line sensor, but those sensors do not
independently stop an overdose caused by a wrong calculation.

### 6.2 System risk and software safety class

The system risk analysis identifies a hazardous situation: a clinician enters a dose in mg/kg, the software interprets
it as mg, and the pump delivers a dose 70 times the intended amount. The harm is severe injury or death. The external
controls include the clinician's review of the programmed values, the hospital's dose-error-reduction library, and the
nurse's independent double-check for high-alert medications. The risk analysis concludes that these controls reduce the
probability but do not make the residual risk acceptable without software controls, because the calculation is not
independently verified by a second system and the double-check is not universal.

The software safety class is Class C. The software can contribute to a hazardous situation with death or serious injury,
and the external controls are not sufficient to reduce the residual risk to an acceptable level. The class applies to
the dose-calculation, library-checking, and motor-control software. The alarm annunciation software may be Class C as
well, because a failure to annunciate an occlusion or air-in-line alarm can contribute to serious harm.

The team considers assigning the user-interface module a lower class because the clinician can review the values. It
rejects that idea. The user interface is the only place where the calculated dose is displayed; a display failure can
hide an error. The module is part of the Class C system and must be verified accordingly.

### 6.3 Lifecycle evidence

The software development plan names the Class C activities: software requirements analysis, architectural design,
detailed design, unit implementation and verification, integration testing, system testing, release, maintenance,
configuration management, problem resolution, and software risk management. The plan names the coding standard, the
static-analysis tool, the unit-test framework, the requirements-management tool, and the configuration baseline.

The requirements specification contains 412 requirements. Each has an identifier, a description, a source, an acceptance
criterion, and a verification method. The safety requirements are linked to the system risk file. The architectural
design describes the dose-calculation item, the library-checking item, the motor-control item, the alarm item, and the
user-interface item. It shows that the dose-calculation item is isolated from the user-interface item by a defined
interface and that the motor-control item has a watchdog.

The detailed design describes the units, the data structures, the unit conversions, the rounding rules, the boundary
checks, and the error handling. The unit verification records include normal, boundary, and invalid-input tests. The
integration tests cover the interfaces between the calculation, library, motor, and alarm items. The system tests cover
the complete pump in the intended environment, including simulated occlusions and air-in-line events.

### 6.4 The evidence gap

The file looks complete until the reviewer asks two questions. First, where is the evidence that the dose calculation
handles every unit and rounding combination in the intended use? The unit tests cover mg, mg/kg, mcg, mcg/kg, units,
units/kg, and mL/h, but they do not cover the combination of a weight-based dose with a concentration expressed in mg/mL
and a duration expressed in minutes. That combination is common in the intended use. The test set was built from the
most common orders, not from the full input domain.

Second, where is the SOUP evaluation for the numeric library that performs the rounding? The software uses a third-party
decimal library. The library has a published anomaly related to rounding half-way values in a specific locale. The
anomaly is not in the device's release notes, and the SOUP evaluation lists only the library name and version, not the
anomaly review. The team assumed the library was safe because it is widely used.

These are not documentation gaps. They are evidence gaps. The first means the verification does not cover the intended
use. The second means a known defect in a safety-related dependency was not assessed. Either can sink a submission or an
audit.

### 6.5 Two diagnostics that would sink the submission

**Diagnostic 1: boundary-value equivalence classes are incomplete.** The reviewer maps the intended-use inputs to
equivalence classes. The dose can be expressed as a fixed dose, a weight-based dose, or a body-surface-area dose. The
concentration can be mg/mL, mcg/mL, units/mL, or a percentage. The duration can be minutes or hours. The current tests
cover each dimension separately but not the cross-product. A test of a weight-based dose with a mg/mL concentration and
a duration in minutes produces a rate that is off by a factor of 60 because the duration conversion is applied twice.
The unit test passes because it checks the formula in isolation, not the end-to-end calculation. The system test does
not catch it because the test order uses a fixed dose. The verification is not sufficient for the intended use.

**Diagnostic 2: the SOUP anomaly is not evaluated.** The decimal library's anomaly affects rounding of half-way values
in a specific locale. The pump is sold in a market where the locale uses a comma as the decimal separator. The
software's input parser converts the comma to a period, but the library's rounding behavior still depends on the locale
setting. A dose that ends in .005 may round down instead of up, producing a rate that is slightly below the intended
dose. The error is small per dose but can accumulate over a long infusion. The SOUP evaluation did not assess the
anomaly, did not test the locale, and did not document a risk acceptance. The reviewer cannot conclude that the
dependency is safe in the intended environment.

### 6.6 The submission decision

The sponsor decides to fix the evidence before filing. It adds equivalence-class tests for the full cross-product of
dose basis, concentration unit, and duration unit. It adds a locale test for the decimal library and a risk assessment
for the rounding anomaly. It adds a SOUP evaluation that lists the anomaly, the affected versions, the mitigation, and
the residual risk. It updates the software development plan to require a SOUP anomaly review for every safety-related
dependency.

The regulatory pathway for the pump is a 510(k), because the pump is a Class II device with a legally marketed
predicate. The software safety class is Class C, but the device class is Class II. The submission includes the software
documentation, the verification and validation evidence, the risk analysis, the cybersecurity documentation, and the
labeling. The EU route is a Class IIb or III conformity assessment, depending on the pump's classification and the
clinical claims; the software lifecycle evidence supports the MDR's state-of-the-art requirement.

The sponsor also schedules an audit-readiness review. The review checks that every Class C requirement has a design
element, a unit test, an integration test, a system test, and a risk link. It checks that every SOUP item has an owner,
a version, an anomaly review, and a monitoring plan. It checks that the released binary can be rebuilt from the
configuration baseline. The review finds three orphan requirements and one missing risk link; the team fixes them before
the submission.

### 6.7 Vary one assumption

The base case is a Class C embedded software system in a Class II pump. Change the assumption: the same dose-calculation
algorithm is marketed as a standalone SaMD that advises a clinician but does not control a pump. The clinician manually
programs the pump. The software displays a recommended rate and a warning if the dose is outside the library limit, but
it does not connect to the motor.

The software safety class may change to Class B or A, depending on the external controls. If the clinician must
independently enter the rate and the pump has its own dose-error-reduction library, the software's failure may be caught
by the pump. If the clinician can copy the recommended rate directly into the pump without independent review, the
software can still contribute to serious harm, and Class C may remain appropriate. The classification is a system
question, not a code question.

The regulatory pathway changes from a 510(k) for the pump to a 510(k) or De Novo for the SaMD, depending on the
predicate and the claim. The software documentation is still required, but the level of detail may be lower for Class B
than for Class C. The clinical evidence changes from pump performance to the effect of the recommendation on clinician
behavior and dose selection. The usability evidence becomes more important because the clinician is the only control.

| Assumption changed | Software class | Device pathway | Evidence consequence |
|---|---|---|---|
| Base: embedded, controls motor | C | 510(k) for pump | Full Class C lifecycle, motor and alarm tests |
| Standalone advisory, clinician independently programs pump | B or C | 510(k) or De Novo for SaMD | Human-factors and clinical decision study |
| Standalone advisory, clinician copies rate directly | C | 510(k) or De Novo | Stronger usability and error-recovery evidence |
| Embedded with independent hardware interlock | B or C | 510(k) for pump | Interlock verification and segregation evidence |
| Legacy pump software with no lifecycle records | C | 510(k) with gap assessment | Retrospective requirements and verification |

The lesson is that the software safety class follows the system, not the delivery model. Moving the algorithm from a
pump to an app can reduce the class if the system adds an independent control, or leave it at Class C if the app becomes
the only control. The team must analyze the actual use, not the architecture diagram.

### 6.8 What the case does not show

The case does not show the full 412-requirement traceability matrix, the complete risk file, or the cybersecurity
assessment. It does not show the supplier audits, the training records, or the management review. Those artifacts are
part of the quality system, and their absence would be a separate finding. The case shows the software lifecycle
evidence that a reviewer will sample: the plan, the requirements, the architecture, the detailed design, the unit and
integration tests, the system tests, the release record, the SOUP evaluation, and the problem-resolution records.

The case also does not show a clinical validation study. For a pump, the clinical evidence may come from the predicate,
usability testing, and post-market data rather than a new randomized trial. For a standalone advisory SaMD, the clinical
evidence may need a reader study or a prospective workflow study. The lifecycle standard tells you how to build and
verify the software; it does not tell you whether the clinical claim is true. A complete file pairs the software
evidence with the clinical, usability, and cybersecurity evidence that the claim requires. The reviewer will ask for all
four, not just the code.

## 7. Code, query, or calculation

The script below encodes a simplified safety-classification tree and the core activity checklist by class. It is a
teaching aid; the real classification requires the system risk analysis and a documented rationale for any external
control that lowers the class.

```python
"""IEC 62304 safety-class and process checklist for Module 14, Topic 07."""

from __future__ import annotations


def safety_class(
    contributes_to_hazard: bool,
    severity: str,
    external_controls_sufficient: bool,
) -> str:
    """Return A, B, or C using a simplified Clause 4.3 tree."""
    if not contributes_to_hazard:
        return "A"
    if external_controls_sufficient:
        return "A"
    severity = severity.strip().lower()
    if severity in {"non_serious", "minor"}:
        return "B"
    if severity in {"serious", "death", "critical"}:
        return "C"
    raise ValueError(f"unknown severity: {severity}")


def required_activities(cls: str) -> list[str]:
    """Return the core IEC 62304 activities expected for the class."""
    base = [
        "development plan",
        "requirements analysis",
        "unit implementation and verification",
        "release",
        "maintenance",
        "configuration management",
        "problem resolution",
    ]
    if cls == "A":
        return base
    if cls == "B":
        return base + ["architectural design", "integration testing", "system testing", "software risk management"]
    if cls == "C":
        return base + [
            "architectural design",
            "detailed design",
            "integration testing",
            "system testing",
            "software risk management",
        ]
    raise ValueError(f"unknown class: {cls}")


def missing_activities(cls: str, present: set[str]) -> list[str]:
    return [a for a in required_activities(cls) if a not in present]


if __name__ == "__main__":
    cases = [
        ("no hazard contribution", safety_class(False, "serious", False)),
        ("external controls sufficient", safety_class(True, "death", True)),
        ("non-serious harm", safety_class(True, "non_serious", False)),
        ("death or serious harm", safety_class(True, "death", False)),
    ]
    for label, result in cases:
        print(f"{label:32s} -> Class {result}")

    present = {"development plan", "requirements analysis", "unit implementation and verification", "release"}
    for cls in ("A", "B", "C"):
        missing = missing_activities(cls, present)
        print(f"\nClass {cls} missing: {', '.join(missing) if missing else 'none'}")
```

Expected output:

```text
no hazard contribution           -> Class A
external controls sufficient     -> Class A
non-serious harm                 -> Class B
death or serious harm            -> Class C

Class A missing: maintenance, configuration management, problem resolution

Class B missing: maintenance, configuration management, problem resolution, architectural design, integration testing, system testing, software risk management

Class C missing: maintenance, configuration management, problem resolution, architectural design, detailed design, integration testing, system testing, software risk management
```

The script treats external controls as a binary input. In a real analysis, the sufficiency of an external control is a
risk judgment that depends on independence, reliability, and detectability. A hardware interlock that is tested at every
power-on is different from a clinical double-check that is followed 70 percent of the time. The script also treats the
class as a single value for the whole system. A real system may have items at different classes, but only with
documented segregation and verification.

The activity list is a subset. It omits documentation, verification planning, and the detailed content requirements
within each process. It is useful as a gap-check, not as a compliance checklist. A team that has all the activities but
no traceability still has a lifecycle gap. A team that has traceability but no configuration baseline cannot reproduce
the release.

## 8. Data, standards, and regulatory context

> **Note:** This section is educational and is not legal advice. The applicability of IEC 62304 and the expected evidence depend on the device, the jurisdiction, and the current guidance. Confirm the standard edition and regulatory recognition before relying on it.

| Source | Status | What it constrains | What it does not do |
|---|---|---|---|
| IEC 62304:2006+A1:2015 | Voluntary standard, recognized by FDA | Sets software lifecycle processes and safety classes | Does not set clinical evidence or device class |
| FDA recognition of IEC 62304 | FDA recognition program | Makes the standard a consensus standard for premarket review | Does not make every clause mandatory |
| IEC 82304-1:2016 | Voluntary standard | Sets health software product requirements for software-only products | Does not replace IEC 62304 |
| IEC 81001-5-1:2021 | Voluntary standard | Sets health software security lifecycle requirements | Does not replace a security risk assessment |
| ISO 14971:2019 | Voluntary standard | Sets risk management requirements | Does not prescribe software design |
| IEC 62366-1:2015+A1:2020 | Voluntary standard | Sets usability engineering requirements | Does not replace software verification |
| ISO 13485:2016 | Voluntary standard, incorporated by QMSR | Sets QMS and design-control requirements | Does not prescribe software lifecycle detail |
| 21 CFR 820 (QMSR) | Binding US regulation | Requires design controls and change control | Does not incorporate IEC 62304 by reference |
| MDR 2017/745 | Binding EU regulation | Requires state-of-the-art software development | Does not name a single standard |
| FDA software documentation guidance | Guidance | Describes premarket software documentation | Is not a statute |

The FDA recognizes IEC 62304:2006+A1:2015 as a consensus standard. Recognition means that a manufacturer may cite the
standard to satisfy certain premarket review expectations, not that every clause is legally mandatory. The manufacturer
still must show that the standard applies to the device and that the claimed clauses were followed. A declaration of
conformity without the underlying records is not evidence.

The European Union takes a different approach. The MDR requires devices to be designed and manufactured according to the
state of the art, and harmonized standards are the usual way to demonstrate conformity. IEC 62304 is a harmonized
standard under the MDR, but the manufacturer must still perform a gap assessment and document any deviations. A CE mark
does not mean that every clause of every standard was applied; it means that the manufacturer has demonstrated
conformity with the applicable requirements.

IEC 82304-1 complements IEC 62304 for software-only products. It covers the product-level requirements that IEC 62304
does not, such as the validation of the health software product, the accompanying documentation, and the lifecycle
requirements for a standalone product. IEC 81001-5-1 covers the security lifecycle. A modern software lifecycle file
should reference both where applicable, rather than treating IEC 62304 as the whole standard set.

The relationship to ISO 14971 is direct. The software safety class comes from the system risk analysis. The software
requirements include risk controls. The software verification provides evidence that the controls work. The software
risk management process links the software hazards to the system hazards. A software risk file that does not reference
the system risk file is incomplete.

The relationship to IEC 62366-1 is also direct. Usability engineering identifies use-related hazards and validates the
user interface. Software requirements must include the user-interface requirements that mitigate those hazards. Software
verification must test the interface behavior. A software lifecycle that ignores usability can pass every unit test and
still produce a device that clinicians misuse.

The relationship to the QMS is structural. The software development plan is a QMS record. The requirements, design,
verification, release, and maintenance records are controlled documents. The problem-resolution process is linked to
corrective and preventive action. The configuration management process is linked to document control. A software
lifecycle that exists outside the QMS is not auditable.

The limits of the standard matter. IEC 62304 does not tell you how to validate a machine learning model, how to design a
clinical study, how to assess cybersecurity, or how to determine the regulatory device class. It tells you how to manage
the software lifecycle. A team that treats it as a model-validation standard will produce a lot of process records and
still fail the clinical evidence question.

A US premarket submission for device software typically includes a software description, a development plan,
requirements, architecture, verification and validation evidence, a software bill of materials, and cybersecurity
documentation. The exact content depends on the device and the submission type. The documentation should tell a reviewer
what the software does, how it was developed, how it was verified, what third-party software it contains, and how it is
maintained. A submission that includes a code repository but not a requirements-to-test trace is not a lifecycle file.

The EU MDR requires software to be developed and manufactured according to the state of the art, with attention to the
lifecycle, risk management, verification, and validation. Annex I sets out general safety and performance requirements
that apply to software, including the need for repeatable, reliable, and safe performance. The manufacturer must show
how the standard and the QMS together satisfy those requirements. A declaration that the software was developed
"according to IEC 62304" is not enough without the records.

Tool qualification is a related practice. A software tool that automates a lifecycle activity, such as a compiler, a
test framework, or a static analyzer, can affect the evidence. The manufacturer should determine whether the tool needs
qualification and how its failures are controlled. For production and quality-system software, FDA's Computer Software
Assurance guidance, issued on 24 September 2025 and revised on 3 February 2026 to align with the QMSR, describes a
risk-based approach. Tool qualification is not the same as device software validation, but the two are often confused.

### The documentation set a reviewer will sample

| Record | What it shows | Common gap |
|---|---|---|
| Software development plan | Processes, activities, deliverables, responsibilities | Plan written after development |
| Requirements specification | What the software must do and how well | Requirements without acceptance criteria |
| Architecture description | Items, interfaces, isolation, SOUP | Diagram without interfaces |
| Detailed design | Internal design and error handling | Missing for Class C |
| Unit verification | Units meet design | Happy-path tests only |
| Integration test record | Items interact correctly | Mocks only |
| System test record | System meets requirements | Developer build |
| Software risk management file | Software contribution to hazards | No link to system risk file |
| Configuration management record | Reproducible release | No build manifest |
| SOUP inventory and evaluation | Third-party composition and risk | Name and version only |
| Problem-resolution record | Defects investigated and fixed | Ticket closed without root cause |
| Release record | Version, approval, distribution | No deployed-version mapping |

The table is a sampling guide, not a complete list. The reviewer will follow a safety-related requirement through these
records. If the chain breaks, the reviewer will ask for more records, not fewer. A team that can produce the chain
quickly is easier to review than a team with a larger but disconnected document set.

### US and EU evidence expectations

| Question | US premarket | EU technical documentation |
|---|---|---|
| What is the software? | Description and intended use | Device description and intended purpose |
| How was it planned? | Development plan | Development and lifecycle plan |
| What must it do? | Requirements | Requirements and design inputs |
| How is it structured? | Architecture | Architecture and interfaces |
| How was it verified? | Unit, integration, system tests | Verification and validation evidence |
| What third-party software? | SBOM and SOUP evaluation | SOUP and supplier evidence |
| How is it released? | Configuration and release records | Configuration and change records |
| How is it maintained? | Maintenance and problem resolution | PMS and change control |

The table is a comparison, not a claim that the two regimes are identical. The US submission has a specific format and
review process; the EU technical documentation is assessed by a notified body against the MDR. The underlying evidence
is similar because the engineering questions are similar.

## 9. Industry practice and case evidence

**Therac-25.** The Therac-25 was a medical linear accelerator that produced a series of radiation overdoses in the
1980s, including fatalities. The public investigations, later documented by Leveson and Turner, described software
defects, race conditions, inadequate testing, and an over-reliance on software interlocks that had replaced hardware
safety features. The case is a foundational example of why software lifecycle rigor, independent verification, and
defense in depth matter. It is not a case about IEC 62304, which was published later, but it illustrates the failure
modes the standard was designed to address. The documented lesson is that software can remove a hazard control without
anyone noticing if the change is not analyzed as a safety change.

**Infusion pump software.** Infusion pumps are a recurring source of software-related recalls and adverse events. Public
FDA records and manufacturer field actions have described dose-calculation errors, library configuration errors, alarm
failures, and cybersecurity vulnerabilities. The details vary by device and year, so this topic does not assert a
specific recall number. The documented pattern is that software defects in dose calculation and alarm annunciation can
cause serious harm, and that verification must cover the full input domain, not only common orders. The inference is
that a Class C lifecycle with boundary-value testing and SOUP evaluation is proportionate to the risk.

**FDA software review.** FDA's premarket review of device software has become more structured over time. The agency
expects a software description, a development plan, requirements, architecture, verification and validation, a software
bill of materials, and cybersecurity documentation. The exact content depends on the device and the submission type. The
documented trend is toward more evidence about the software lifecycle and less reliance on a narrative description. The
inference is that a team that maintains lifecycle records during development will spend less time reconstructing them
before submission.

**Industry practice: requirements before code.** Mature teams write requirements before implementation, even in agile
environments. They use a definition of done that includes traceability, unit tests, and risk links. They review the
requirements with clinical, quality, and regulatory stakeholders. They do not treat a user story as a requirement unless
it has an acceptance criterion and a verification method.

**Industry practice: architecture as a safety argument.** Companies that build safety-critical software use architecture
to argue that failures are contained. They isolate safety functions, use watchdogs, use independent monitoring, and
verify the segregation. They document why a lower-class item cannot affect a higher-class function. The architecture is
not only a technical diagram; it is part of the safety case.

**Industry practice: SOUP inventory.** Companies with mature lifecycle processes maintain a software bill of materials,
track SOUP versions and vulnerabilities, and evaluate published anomalies. They do not assume that a popular library is
safe. They assign an owner to each SOUP item and a process for monitoring advisories. The inventory is a living
document, not a one-time spreadsheet.

**What the cases do not prove.** The Therac-25 does not prove that modern software is unsafe. The infusion pump pattern
does not prove that every pump has a defect. The FDA review trend does not prove that every submission needs the same
documentation. Each case is a lesson about a specific failure mode. The transferable method is the lifecycle: define the
boundary, classify the risk, write requirements, design for isolation, verify at every level, control the configuration,
and maintain the product after release.

**FDA software recall patterns.** FDA's public recall and adverse-event records show that software can be a contributing
cause of device field actions. The specific numbers change by year and product area, so this topic does not assert a
count. The documented pattern is that software defects can cause dose errors, display errors, alarm failures, and
connectivity failures, and that many are discovered after launch. The inference is that lifecycle verification and
maintenance are not only premarket requirements; they are postmarket safety controls.

**Industry practice: cloud and over-the-air updates.** Cloud-hosted and network-connected devices introduce a deployment
model that the original lifecycle standard did not anticipate. A manufacturer may update a model or a service weekly,
while a hospital may validate the update quarterly. Mature teams use staged rollouts, version pinning, canary
deployments, rollback, and customer notification. They also maintain a configuration baseline for each deployed version,
even when the deployment is continuous. The lifecycle evidence must cover the deployed configuration, not only the
source repository.

**Industry practice: the model as a software item.** A machine learning model is a software item with requirements, a
design, a configuration, a verification plan, and a maintenance process. The training pipeline is part of the software
system. The training data are a configuration item. The model weights are a release artifact. A team that treats the
model as a research artifact outside the lifecycle will struggle to answer basic questions about version, provenance,
and change.

**Industry practice: audit sampling.** An auditor does not read every record. The auditor samples a safety-related
requirement and follows it through design, code, test, risk, release, and maintenance. If any link is missing, the
auditor expands the sample. A team that has a strong narrative but weak links will fail the sample. A team that has
modest documentation but complete links will pass more easily. The lesson is to optimize for traceability, not volume.

**Industry practice: security is a lifecycle property.** A penetration test at the end of development is not a security
lifecycle. Mature teams integrate threat modeling, security requirements, secure coding, dependency scanning,
vulnerability monitoring, and coordinated disclosure into the software lifecycle. IEC 81001-5-1 provides a
health-software-specific framework. A security vulnerability in a SOUP component can become a patient-safety issue, so
the SOUP and security processes must be connected.

**Industry practice: design reviews are not status meetings.** A design review is a documented examination of a design
against its requirements, risks, and interfaces. It should include a reviewer who is independent of the design team and
a specialist when the design involves a safety-critical algorithm. The output is a record of issues, decisions, and
actions, not a slide deck. A design review that finds nothing is either unnecessary or not looking.

**What the cases do not prove, continued.** The public record rarely shows the internal traceability matrix, the review
minutes, or the SOUP anomaly review. It shows the outcome: a recall, a clearance, or an audit finding. A team that
copies the outcome without building the lifecycle will have the appearance of process and the substance of a demo. The
transferable asset is the evidence chain: requirement, design, test, risk, release, and maintenance.

**Industry practice: the lifecycle as a product capability.** Companies that sell into hospitals increasingly face
customer due diligence that mirrors a regulatory audit. A hospital may ask for the software bill of materials, the
vulnerability disclosure policy, the validation summary, the version history, and the rollback procedure before
procurement. A lifecycle file that was built only for a regulator is often insufficient for a customer. The same
evidence, organized for customers, becomes a sales asset rather than a compliance cost.

**Industry practice: learning from near misses.** A near miss is a defect that did not reach a patient, a test that
almost passed, or a field issue that was caught by a workaround. Mature teams treat near misses as free evidence and
investigate them with the same rigor as adverse events. A near miss that is closed as "no patient impact" without a root
cause is a missed opportunity to prevent the next event. The lifecycle's problem-resolution process should capture near
misses, not only complaints.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Requirements written after code | Tests exist but no requirement IDs | Verification cannot be traced | Submission deficiency | Write requirements first and baseline |
| Safety class too low | Class A assigned to dose calculation | External controls assumed, not verified | Unsafe software under-tested | Document and challenge class rationale |
| No architecture | Single monolithic codebase | Safety functions not isolated | Failure propagation | Define items and interfaces |
| Unit tests only for happy path | Boundary values absent | Input-domain gaps | Calculation errors | Equivalence classes and boundary tests |
| Integration tests use mocks | Real interfaces untested | Contract mismatch | Field failure | Contract and failure-injection tests |
| System test uses developer build | Release differs from tested build | Configuration drift | Unverified release | Rebuild from baseline and retest |
| SOUP not evaluated | Inventory lists name only | Known anomaly missed | Safety-related dependency failure | Anomaly review and environment tests |
| No software bill of materials | Transitive dependencies unknown | Vulnerability blind spot | Security incident | Generate and maintain SBOM |
| Release without baseline | No reproducible build | Version ambiguity | Audit finding | Freeze, build, record |
| Maintenance outside QMS | Patches applied ad hoc | Uncontrolled change | Regression | Change control and impact analysis |
| Problem tickets closed without root cause | Recurring defects | Symptom fixes | Repeat failures | Root-cause analysis and CAPA |
| Traceability matrix stale | Orphan requirements | Coverage gap | Incomplete verification | Automated traceability checks |
| Legacy software assumed exempt | No gap assessment | Unknown risks | Audit and safety risk | Retrospective assessment |
| Cybersecurity treated as separate | No security requirements | Vulnerabilities unmanaged | Safety and privacy harm | Integrate IEC 81001-5-1 |

A red flag is an invitation to inspect the evidence, not a reason to rewrite the file. The most common root cause is a
team that treats the lifecycle as a submission deliverable rather than an engineering process. The fix is to make the
records a byproduct of the work: requirements in the backlog, tests in the pipeline, risks in the risk file, and
configuration in the build.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The system boundary is complete | Safety class depends on it | Hidden cloud or library dependency | Architecture and dependency review |
| The safety class is correct | Process rigor depends on it | External control is not effective | Independent risk review |
| Requirements are verifiable | Tests need acceptance criteria | Vague or untestable requirement | Requirement review |
| The architecture isolates safety functions | Lower-class items depend on it | Shared memory or uncontrolled interface | Segregation testing |
| The test environment represents use | Verification must generalize | Developer build or curated data | Production-equivalent testing |
| The SOUP is safe in context | Third-party code is pervasive | Known anomaly or vulnerability | Anomaly and vulnerability review |
| The configuration is reproducible | Release identity depends on it | Floating dependency or manual build | Rebuild and compare |
| Maintenance is resourced | Security and defects need fixes | No owner or budget | Maintenance plan and SLAs |
| The QMS records are current | Audit depends on them | Documents updated after the fact | Internal audit |

The limits of this topic are explicit. It does not reproduce the standard. It does not cover every clause or every
device type. It does not provide a legal opinion or a substitute for a quality-system audit. It gives a structured way
to apply the lifecycle and to recognize the evidence a reviewer will sample.

The assumptions are not equally important. The boundary and safety-class assumptions carry the most weight because they
determine the process rigor. A wrong class can under-test a safety function or over-burden a low-risk item. A wrong
boundary can exclude a critical dependency. The requirements and traceability assumptions are next because they
determine whether the evidence is usable. A team that gets the first two right and the rest imperfectly is in a better
position than a team that gets the first two wrong and the rest perfectly.

## 12. Exercises

### Exercise 1 — Scope and class (30 minutes)

Choose a software-containing device. Draw the software system boundary, list the software items and SOUP, and assign a
safety class with a rationale. Deliverable: a boundary diagram and a classification memo. Acceptance criteria: every
item is inside or outside the boundary with a reason; the class cites the hazard and the external controls; lower-class
items have a segregation argument.

### Exercise 2 — Requirements set (45 minutes)

Write ten software requirements for the device, including two safety requirements and two performance requirements. For
each, give an identifier, source, acceptance criterion, and verification method. Deliverable: a requirements table.
Acceptance criteria: each requirement is verifiable; each safety requirement links to a hazard; no requirement specifies
an algorithm without a reason.

### Exercise 3 — Architecture review (40 minutes)

Produce an architectural diagram with items, interfaces, and isolation. Identify one failure that could propagate across
an interface and one control that prevents it. Deliverable: the diagram and a one-page review. Acceptance criteria:
interfaces are labeled; the isolation claim is testable; SOUP is shown.

### Exercise 4 — Verification plan (40 minutes)

Write a verification plan covering unit, integration, and system testing. Include boundary values, invalid inputs, and
failure paths. Deliverable: a test plan with traceability. Acceptance criteria: every safety requirement has a test; the
test environment is described; the pass/fail criteria are explicit.

### Exercise 5 — SOUP evaluation (35 minutes)

Take one third-party library or model. Document its function, version, requirements, published anomalies, intended
environment, and monitoring plan. Deliverable: a SOUP evaluation record. Acceptance criteria: the anomaly review is
specific; the verification is in the intended environment; the residual risk is stated.

### Exercise 6 — Configuration audit (30 minutes)

Define the configuration items for a release and the records needed to reproduce it. Include source, build tools,
dependencies, model weights, data version, and documentation. Deliverable: a configuration baseline checklist.
Acceptance criteria: a second engineer could rebuild the release; the SBOM includes transitive dependencies.

### Exercise 7 — Problem resolution (30 minutes)

Write a problem report for a hypothetical defect, including severity, affected versions, root cause, correction,
verification, and CAPA link. Deliverable: the problem record. Acceptance criteria: the root cause is distinguished from
the symptom; the correction is verified; the risk file is updated.

### Exercise 8 — Audit simulation (45 minutes)

Exchange a lifecycle file with a peer. Sample five requirements and trace each to design, test, risk, and release. Write
three findings with evidence requests. Deliverable: a traceability sample and findings. Acceptance criteria: findings
cite specific records; at least one finding is a substantive evidence gap, not a typo.

### Portfolio rubric

| Criterion | Weight | Competent | Expert |
|---|---:|---|---|
| Scope and class | 15% | Boundary drawn; class stated | Class rationale tested against external controls |
| Requirements | 20% | Verifiable and sourced | Complete, conflict-free, risk-linked |
| Architecture | 15% | Items and interfaces shown | Isolation claim verified |
| Verification | 20% | Unit, integration, system tests | Full traceability and boundary coverage |
| SOUP and security | 10% | Inventory and evaluation | Anomaly and vulnerability monitoring |
| Configuration | 10% | Baseline defined | Reproducible build demonstrated |
| Maintenance | 10% | Process described | Change impact and rollback tested |

Each exercise should be documented as if it were part of a design history file. A reviewer should be able to see the
author, date, version, approval, and link to the relevant requirement. An exercise that produces a nice diagram but no
traceability does not meet the criterion.

## 13. Self-check answers

**Question 1. What is IEC 62304?** A standard for medical device software lifecycle processes. It covers planning,
requirements, design, implementation, verification, release, maintenance, configuration management, and problem
resolution. It is not a clinical validation standard or a device classification rule.

**Question 2. How does IEC 62304 relate to ISO 14971?** ISO 14971 identifies hazards and risk controls at the system
level. IEC 62304 requires the software lifecycle to contribute to those controls. The software safety class comes from
the system risk analysis, and software requirements and tests provide evidence that the controls work.

**Question 3. How is the software safety class determined?** By the possible consequence of software failure after
external risk controls. Class A: no contribution to a hazardous situation, or acceptable residual risk after external
controls. Class B: contribution with non-serious injury. Class C: contribution with death or serious injury.

**Question 4. What is the difference between verification and validation?** Verification confirms that the software
meets its requirements. Validation confirms that the software meets user needs and intended use in the real environment.
A model can pass verification and fail validation.

**Question 5. What is traceability?** The ability to link a requirement to its source, design element, verification
test, risk control, and released version. Traceability makes coverage auditable.

**Question 6. What can unit testing not prove?** It cannot prove that units interact correctly, that the system meets
user needs, or that the clinical claim is true. Those require integration, system, and validation evidence.

**Question 7. What is configuration management?** The process for identifying, controlling, and tracking configuration
items and changes. It covers source, builds, dependencies, models, data, and documentation. It makes a release
reproducible.

**Question 8. What is SOUP and why does it matter?** Software of unknown provenance, developed by a third party and not
under the manufacturer's lifecycle control. It matters because a defect or vulnerability in SOUP can affect a safety
function. The manufacturer must identify, evaluate, verify, and monitor it.

**Question 9. What are two signs that a lifecycle file is weak?** Requirements cannot be traced to tests, and tests use
a different build from the released version. A third sign is a SOUP inventory with no anomaly review.

**Question 10. How does a software change after release get controlled?** Through the maintenance and problem-resolution
processes: classify the change, assess the impact, implement, verify, update the configuration and risk file, approve,
release, and monitor.

**Question 11. Why is a software bill of materials necessary?** A device contains direct and transitive dependencies
that can change or become vulnerable. The SBOM identifies what is in the release, which versions are affected by a
vulnerability, and what must be patched. It also supports incident response and customer security questionnaires.

**Question 12. What makes a release reproducible?** A defined configuration baseline, a controlled build environment,
pinned dependencies, recorded build tools, and a manifest of the output. If the manufacturer cannot rebuild the released
binary from the baseline, the release is not reproducible and the verification evidence may not apply to the deployed
version.

## 14. Interview questions

**1. Why does software need a lifecycle standard?** A strong answer explains that software failures are systematic, hard
to see, and often caused by missing requirements or verification. A weak answer says "because regulators require
paperwork."

**2. How do you assign a software safety class?** A strong answer starts from the system hazard, considers external
controls, and documents the rationale. A weak answer equates it with the device class.

**3. What is the difference between Class B and Class C?** A strong answer says Class B is non-serious injury and Class
C is death or serious injury, with different process requirements. A weak answer says C is "more important."

**4. What does a software development plan contain?** A strong answer names activities, deliverables, responsibilities,
standards, reviews, and configuration. A weak answer describes a schedule.

**5. How do you verify a safety requirement?** A strong answer links it to a test with an acceptance criterion and a
risk control. A weak answer says "we test it."

**6. What is SOUP and how do you control it?** A strong answer covers inventory, requirements, anomaly review,
environment verification, and vulnerability monitoring. A weak answer says "we use open source."

**7. How do you handle a legacy device?** A strong answer describes a gap assessment, risk analysis, and targeted
lifecycle activities. A weak answer says legacy devices are exempt.

**8. What is the role of architecture in safety?** A strong answer explains isolation, interfaces, and failure
containment. A weak answer treats architecture as a diagram for the slide deck.

**9. How do you keep traceability current?** A strong answer uses tooling, reviews, and a definition of done. A weak
answer relies on a spreadsheet updated before audits.

**10. What would you do if a SOUP vulnerability is announced?** A strong answer assesses exploitability, affected
versions, patient risk, patch availability, and regulatory reporting. A weak answer patches immediately without
verification.

**11. How do you handle a model update under IEC 62304?** A strong answer treats the model as a software item,
classifies the change, assesses the impact on requirements and risks, re-verifies in the intended environment, updates
the configuration and documentation, and releases through change control. A weak answer retrains and deploys without a
lifecycle record.

**12. What would you show an auditor in the first ten minutes?** A strong answer shows the software development plan,
the safety classification, a safety-related requirement traced through design, test, risk, and release, and the
configuration baseline for the deployed version. A weak answer shows a slide deck and a code repository.

## 15. Key terms

| Term | One-line meaning |
|---|---|
| IEC 62304 | Medical device software lifecycle standard |
| Software safety class | A, B, or C based on possible harm |
| Class A | No contribution to hazard, or acceptable residual risk after external controls |
| Class B | Contribution to hazard with non-serious injury |
| Class C | Contribution to hazard with death or serious injury |
| Software system | Software items and environment within the defined boundary |
| Software item | A unit or group of units treated as a lifecycle object |
| Software unit | A separately testable software element |
| SOUP | Software of unknown provenance |
| Software development plan | Plan of processes, activities, deliverables, and responsibilities |
| Software requirements | Verifiable requirements derived from needs and risks |
| Architectural design | Structure, items, interfaces, and isolation |
| Detailed design | Internal design of each software item |
| Unit verification | Evidence that a unit meets its design |
| Integration testing | Evidence that items interact correctly |
| System testing | Evidence that the system meets requirements |
| Software release | Controlled delivery of a version |
| Maintenance | Post-release change and support process |
| Problem resolution | Investigation and correction of problems |
| Configuration management | Identification and control of versions |
| Software risk management | Software contribution to system risk management |
| Verification | Confirmation that requirements were fulfilled |
| Validation | Confirmation that user needs were met |
| Traceability | Links among requirements, design, tests, risks, and releases |
| Legacy software | Software developed before the current lifecycle |
| IEC 82304-1 | Health software product standard |
| IEC 81001-5-1 | Health software security lifecycle standard |
| IEC 62366-1 | Usability engineering standard |
| SBOM | Software bill of materials |
| CAPA | Corrective and preventive action |

## 16. Sources

All URLs were accessed on 25 October 2026. Verify current editions and recognition.

1. International Electrotechnical Commission. IEC 62304:2006+A1:2015, Medical device software — Software life cycle processes. https://webstore.iec.ch/en/publication/22794
2. US Food and Drug Administration. "Recognized Consensus Standards." https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfStandards/search.cfm
3. International Electrotechnical Commission. IEC 82304-1:2016, Health software — Part 1: General requirements for product safety. https://webstore.iec.ch/en/publication/26576
4. International Electrotechnical Commission. IEC 81001-5-1:2021, Health software and health IT systems safety, effectiveness and security — Part 5-1. https://webstore.iec.ch/en/publication/64219
5. International Organization for Standardization. ISO 14971:2019. https://www.iso.org/standard/72704.html
6. International Electrotechnical Commission. IEC 62366-1:2015+A1:2020. https://webstore.iec.ch/en/publication/67220
7. International Organization for Standardization. ISO 13485:2016. https://www.iso.org/standard/59752.html
8. US Food and Drug Administration. "Quality Management System Regulation (QMSR)." https://www.fda.gov/medical-devices/postmarket-requirements-devices/quality-management-system-regulation-qmsr
9. US Food and Drug Administration. "Software Documentation for Premarket Submissions." https://www.fda.gov/regulatory-information/search-fda-guidance-documents/software-documentation-premarket-submissions
10. European Parliament and Council. Regulation (EU) 2017/745 (MDR). https://eur-lex.europa.eu/eli/reg/2017/745/oj
11. Leveson NG, Turner CS. "An Investigation of the Therac-25 Accidents." IEEE Computer, 1993. https://doi.org/10.1109/MC.1993.274940
12. US Food and Drug Administration. "Infusion Pumps." https://www.fda.gov/medical-devices/general-hospital-devices-and-supplies/infusion-pumps
13. US Food and Drug Administration. "Cybersecurity in Medical Devices." https://www.fda.gov/medical-devices/digital-health-center-excellence/cybersecurity

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`06_iso-14971-risk-management.md`](06_iso-14971-risk-management.md)
- Next topic: [`08_eu-mdr-and-ivdr.md`](08_eu-mdr-and-ivdr.md)
- Related topic: [`05_qms-and-design-controls.md`](05_qms-and-design-controls.md)
- Survey file: [`../../course/14_medical_devices_samd_regulation.md`](../../course/14_medical_devices_samd_regulation.md)
- Labs: [`../../labs/`](../../labs/)
