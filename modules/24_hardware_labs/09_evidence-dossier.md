# Evidence Dossier

> Module 24 — Hardware and Robotics Labs
> Topic 9 of 9
> Estimated study time: 5 hours
> Prerequisites: 01, 02, 03, 05, 07, 08
> Version 2026.3

## 1. Why this topic exists

A lab produces artefacts: a notebook, a capture, a dataset, a report. An evidence dossier is the argument that connects those artefacts to a claim. The difference is not the amount of material. It is the structure.

The industry problem is that teams submit material, not evidence. A folder of screenshots is not a dossier. A test report without a requirement is not verification. A risk file without a verification link is a list of worries. A reviewer who cannot follow the argument will not accept the claim, however good the engineering was.

Three decisions depend on this topic:

1. **What claim the evidence supports.** A claim about a prototype, a claim about a design, and a claim about a clinical outcome require different evidence. The dossier must state which one it supports.
2. **Whether every requirement and risk control has a verified link.** A requirement without a test is a gap. A test without a requirement is an orphan. Both are findings in a review.
3. **What the dossier does not contain.** A gap assessment that names the missing clinical evidence, the missing accredited tests, and the missing quality-system records is more credible than a dossier that implies completeness.

The commercial stakes are concrete. A technical documentation gap can stop a CE mark. A design history file that cannot be reconstructed can fail an FDA inspection. A due-diligence review that finds an unsupported claim can reduce a valuation or end a deal.

This topic is the capstone of the module. It takes the artefacts from the other eight topics and turns them into a dossier that a reviewer who was not in the room can follow.

The dossier is also a test of the team's honesty. It is easy to write a report that emphasises the good results. It is harder to write one that states the gaps in the same font size as the achievements. The second is the one that survives a review.

> **Safety statement for this module.** Hazards in these labs include electric shock, stored energy in capacitors and batteries, burns from soldering irons and hot water, cuts from strip tools and glass, eye injury from projectiles and ultraviolet light, pinch and crush injury from servo arms, and lithium battery fire. Required personal protective equipment (PPE) is safety glasses in every lab that solders, cuts, heats, or pressurises; a lab coat or equivalent coverage; closed shoes; and an electrostatic discharge (ESD) wrist strap whenever a bare board is handled. The electrical-safety boundary for this course is extra-low voltage only: a current-limited supply or a fused USB supply, with no mains conductor exposed on the bench. Any experiment that is mains-powered, that connects to a patient, or that uses an applied part requires qualified supervision and the institution's electrical-safety process. No lab in this module authorises such work. The dossier must state that the prototype is not a medical device, is not for clinical use, and has not been validated or approved by any regulator.

## 2. Learning outcomes

By the end of this topic you can:

- Write an intended-use statement and a claim boundary that a reviewer can test.
- Build a requirements table with identifiers, verification methods, and acceptance criteria.
- Build a traceability matrix that links requirements, risk controls, tests, results, and evidence files.
- Identify orphan tests and uncovered requirements from the matrix.
- Write a gap assessment that names the missing evidence and estimates the effort to obtain it.
- Assemble a dossier with the structure a notified body or an FDA reviewer expects.
- Critique a draft claim for the mismatch between the evidence and the wording.
- Conduct a peer review and record findings with dispositions.

Each outcome maps to an exercise in Section 12 or a self-check item in Section 13.

## 3. The core mental model

A dossier is an argument in five moves: what the device is for, what it must do, how it is built, what can go wrong, and what was verified. The gaps are the sixth move, and they are what makes the argument honest.

### 3.1 The argument chain

```text
  INTENDED USE
       |
  REQUIREMENTS ---------> acceptance criteria
       |
  ARCHITECTURE ----------> interfaces, timing, signal chain
       |
  RISK FILE -------------> hazards, controls, residual risk
       |
  VERIFICATION ----------> tests, results, evidence
       |
  GAP ASSESSMENT --------> what is missing
       |
  CLAIM -----------------> bounded statement
```

Each move answers a question a reviewer will ask. If a move is missing, the reviewer supplies the answer, and it is usually less favourable than the one the team would have written.

### 3.2 The three audiences

The same dossier is read by three audiences with different questions.

| Audience | Question | What they check first |
|---|---|---|
| Engineering | Does it work, and how do you know? | Architecture, tests, failures |
| Regulatory | Does it meet the requirements, and is the evidence controlled? | Traceability, risk file, standards |
| Clinical or commercial | Does it solve the problem, and what does it not do? | Intended use, claim boundary, gaps |

A dossier written only for the engineering audience is a technical report. A dossier written only for the regulatory audience is a compliance checklist. A dossier that serves all three starts with the intended use and the claim boundary, because those are the only statements all three need to agree on.

### 3.3 The evidence graph

The traceability matrix is a graph, not a list. A requirement can link to several tests, and a test can link to several requirements. A risk control can link to a requirement and a test. The graph is the argument.

```text
  R-01 ---- T-03 ---- E-03
    |         |
    |         +---- T-04 ---- E-04
    |
  RC-02 ---- T-05 ---- E-05
```

An orphan test is a node with no requirement. An uncovered requirement is a node with no test. A risk control with no test is an untested control. A test with no evidence file is an assertion.

### 3.4 The claim boundary

A claim is a sentence that asserts something about the device. The boundary is the set of conditions under which the sentence is true.

| Claim | Boundary |
|---|---|
| "The device detects arrhythmias with a sensitivity of 0.93" | On the tested records, with the tested annotations, at the tested sample rate |
| "The device meets a 30 ms latency requirement" | On the tested hardware, with the tested firmware, at the tested load |
| "The device is safe" | A claim that requires a risk file, clinical evidence, and post-market data; not a lab claim |
| "The device is compliant with IEC 60601-1" | A claim that requires accredited testing and a quality system; not a lab claim |

The boundary is not a disclaimer at the end. It is part of the claim, and it should be written before the evidence is collected.

### 3.5 The gap assessment

A gap assessment is a list of what the dossier does not contain, with an estimate of the effort to obtain it.

| Gap | Why it matters | Effort to close | Owner |
|---|---|---|---|
| No clinical evidence | The clinical claim is unsupported | A clinical investigation | Clinical |
| No accredited EMC test | The EMC claim is unsupported | A test-house booking | Regulatory |
| No QMS audit | The compliance claim is unsupported | A certification programme | Quality |
| No usability study | The use-error claim is unsupported | A formative and summative study | Human factors |
| No cybersecurity assessment | The security claim is unsupported | A threat model and penetration test | Security |

A gap that is named is a plan. A gap that is hidden is a liability.

### 3.6 The reviewer's path

A reviewer follows the dossier in a predictable order: intended use, requirements, architecture, risk, verification, gaps. If any step is missing or inconsistent, the reviewer stops and asks a question. The dossier's job is to answer the question before it is asked.

### 3.7 The dossier is a test of the claim

A dossier can be read as a test. The claim is the hypothesis. The requirements are the operational definitions. The tests are the measurements. The traceability matrix is the analysis plan. The gap assessment is the limitations section.

Read this way, the dossier fails when the claim is broader than the operational definitions, when the tests do not measure what the definitions say, or when the analysis is selected after the result.

| Dossier element | Experimental analogue | Failure mode |
|---|---|---|
| Intended use | Population and setting | The population is vague |
| Requirement | Operational definition | The definition is not measurable |
| Test | Measurement | The test measures a proxy |
| Traceability | Analysis plan | The links are missing |
| Result | Data | The uncertainty is omitted |
| Gap assessment | Limitations | The gaps are hidden |

The analogy is not perfect, but it is useful. A team that can write the dossier can usually explain the experiment, and a team that cannot explain the experiment usually cannot write the dossier.

### 3.8 The evidence hierarchy inside the dossier

Not all evidence has the same weight. The dossier should make the hierarchy visible.

| Evidence type | Weight | Example | Limitation |
|---|---|---|---|
| Clinical study | Highest for a clinical claim | Prospective trial | Cost, time |
| Accredited type test | High for a compliance claim | EMC report | One configuration |
| Design verification | High for a design claim | HIL report | Bench conditions |
| Bench measurement | Medium | Calibration record | One unit |
| Simulation | Low for a physical claim | Kinematic model | Omissions |
| Expert opinion | Lowest | A memo | Not reproducible |

A claim should be supported by evidence at the appropriate level. A clinical claim supported by a simulation is a mismatch, and a design claim supported by a clinical study is a waste of resources.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Dossier | Structured set of documents and evidence supporting a claim | A folder of files | Document set |
| DHF | Design history file | A project folder | Record set |
| Technical documentation | The MDR Annex II set | A user manual | Document set |
| Intended use | The use for which the device is intended | A marketing description | Statement |
| Indications for use | The clinical conditions and populations | The intended use | Statement |
| Claim | A sentence that asserts something about the device | A hope | Statement |
| Claim boundary | Conditions under which the claim is true | A disclaimer | Statement |
| Requirement | A statement of what the device must do | A wish | Identifier |
| Acceptance criterion | Condition for pass or fail | A target | Statement |
| Verification | Confirmation that output meets input | A test | Activity |
| Validation | Confirmation that the device meets user needs | A verification | Activity |
| Traceability | Ability to link requirements, risks, tests, and evidence | A list of tests | Graph |
| Orphan test | Test with no linked requirement | A test | Finding |
| Coverage gap | Requirement or risk control with no test | A missing test | Finding |
| Risk file | Record of hazards, controls, and residual risk | A risk matrix | Document |
| Residual risk | Risk remaining after controls | Zero risk | Same scale as risk |
| Hazard | Potential source of harm | A fault | None |
| Hazardous situation | Circumstance in which a person is exposed to a hazard | A hazard | None |
| Harm | Injury or damage to health | A bug | None |
| Risk control | Action that reduces risk | A test | None |
| Essential performance | Performance necessary to avoid unacceptable risk | All performance | Statement |
| SOUP | Software of unknown provenance | Any library | Item |
| SBOM | Software bill of materials | A list of files | Document |
| GSPR | General safety and performance requirements | A checklist | Requirement |
| Predicate | Device to which substantial equivalence is claimed | A competitor | Device |
| PCCP | Predetermined change control plan | A change log | Document |
| CER | Clinical evaluation report | A literature review | Document |
| PMS | Post-market surveillance | Complaint handling | Process |
| CAPA | Corrective and preventive action | A bug fix | Process |
| Nonconformity | Failure to meet a requirement | A bug | Finding |
| Observation | Condition that could lead to a nonconformity | A minor issue | Finding |
| Objective evidence | Data supporting the existence or truth of something | An opinion | Record |
| Configuration index | Record of the configuration of a device | A version number | Document |
| Master record | Controlled index of the dossier | A table of contents | Document |
| Audit | Systematic examination of records | A review meeting | Process |
| Finding | Result of an audit or review | A complaint | Record |
| Disposition | Decision on a finding | A comment | Record |
| Gap assessment | List of missing evidence and effort to close | A limitations section | Document |
| Due diligence | Investigation before an investment or acquisition | An audit | Process |

## 5. The mechanics, step by step

The procedure below assembles a dossier from lab artefacts. It assumes the artefacts from the other topics: a stack record, a calibration record, a sensor result, a firmware fault matrix, an EMC plan, a HIL report, and a robotics result.

### Step 0. Check the evidence environment

**Input:** the artefacts, the repository, and the review team.

**Output:** a controlled workspace, a configuration index, and a confirmed safety boundary.

**Rationale:** a dossier is assembled from files, and the files must be under control before the argument is built. A file on a personal laptop is not evidence, and a workspace with no access control is not a dossier.

| Item | Check | Why it matters |
|---|---|---|
| Repository | Read access for the reviewer | Navigation |
| Access control | Write access limited to the team | Integrity |
| Hash tool | Available and tested | Evidence integrity |
| Configuration index | Every artefact has a build or commit | Reproducibility |
| Review team | Independent reviewer assigned | Peer review |
| Safety boundary | No new experiment is required to write the dossier | Safety |

The dossier is a document, not an experiment. If assembling it reveals a missing test, the test is a new experiment and must follow the safety boundary in Topic 1: extra-low voltage only, no mains conductor exposed, no patient connection, and qualified supervision for anything outside that boundary.

### Step 1. Write the intended use and the claim

**Input:** the device concept and the user need.

**Output:** an intended-use statement, an indications statement, and a claim with a boundary.

**Rationale:** the intended use determines the requirements, the standards, and the evidence. A device with a vague intended use has no testable requirements.

The intended-use statement contains:

- The medical purpose or the non-medical function.
- The patient population, if any.
- The user and the environment.
- The part of the body or the process affected.
- The operating principle.
- The conditions of use.

The claim is one sentence, and the boundary is a list of conditions. A claim without a boundary is a marketing statement; a boundary without a claim is a disclaimer.

### Step 2. Build the requirements table

**Input:** the intended use, the risk file, and the applicable standards.

**Output:** a numbered requirement table with verification methods and acceptance criteria.

**Rationale:** the requirements are the contract between the design and the evidence. A requirement without a verification method cannot be tested.

| Field | Content | Example |
|---|---|---|
| ID | Unique identifier | R-012 |
| Statement | What the device must do | Detect QRS complexes within 150 ms |
| Source | User need, standard, or risk control | Risk control RC-04 |
| Verification | Test, analysis, inspection, or demonstration | Test |
| Acceptance | Numeric criterion | Sensitivity ≥ 0.90 |
| Test ID | Linked test | T-021 |
| Status | Verified, failed, or gap | Verified |

The table is the spine of the dossier. Every other section refers to it.

### Step 3. Write the architecture and the interfaces

**Input:** the design and the requirements.

**Output:** an architecture diagram, an interface table, and a timing budget.

**Rationale:** the architecture explains how the requirements are met. An interface table explains what could go wrong at each boundary.

The architecture section contains:

- The block diagram.
- The signal chain from sensor to output.
- The power architecture.
- The software architecture.
- The external interfaces.
- The timing budget.
- The safety architecture.

A block diagram without interfaces is a cartoon. The interface table is what a reviewer uses to find the integration risks.

### Step 4. Build the risk file

**Input:** the hazards, the architecture, and the intended use.

**Output:** a risk table with hazards, hazardous situations, harms, controls, and residual risk.

**Rationale:** the risk file drives the verification. A control that is not verified is not a control.

| Field | Content | Example |
|---|---|---|
| ID | Hazard identifier | H-07 |
| Hazard | Potential source of harm | Missed arrhythmia |
| Sequence of events | How exposure occurs | Algorithm fails on noisy signal |
| Hazardous situation | Exposure | Clinician does not receive alarm |
| Harm | Injury | Delayed treatment |
| Severity | 1–5 | 4 |
| Probability | 1–5 | 3 |
| Risk | Combination | 12 |
| Control | Action | Sensitivity requirement and HIL test |
| Residual risk | After control | 6 |
| Verification | Test or analysis | T-021 |

The risk file is not a matrix of numbers. It is a set of arguments, each with a control and an evidence link.

### Step 5. Map the evidence to the requirements

**Input:** the requirements, the tests, and the evidence files.

**Output:** a traceability matrix.

**Rationale:** the matrix is the proof that the evidence covers the requirements. A missing link is a gap that a reviewer will find.

Procedure:

1. List every requirement.
2. List every test.
3. Link each test to the requirements it verifies.
4. Link each requirement to at least one test.
5. Link each risk control to a test.
6. Identify orphan tests and uncovered requirements.
7. Identify tests without an evidence file.
8. Record the coverage.

The matrix is a table, but the analysis is a graph. A requirement with two tests is stronger than a requirement with one, if the tests cover different conditions.

### Step 6. Write the verification section

**Input:** the test reports and the evidence files.

**Output:** a verification summary with the method, the result, the uncertainty, and the verdict for each requirement.

**Rationale:** the verification section is the evidence, not the claim. It must state what was measured, how, and with what uncertainty.

Each entry contains:

- The requirement ID.
- The test ID and the method.
- The configuration and the instrument.
- The result with uncertainty.
- The acceptance criterion.
- The verdict.
- The evidence file and its hash.

A verification entry that says "passed" without a number is not evidence.

### Step 7. Write the software lifecycle section

**Input:** the software architecture, the SOUP list, and the test results.

**Output:** a software section with the safety class, the lifecycle activities, the SOUP, and the verification.

**Rationale:** software is a large part of a modern device, and a reviewer will look for the lifecycle evidence.

The section contains:

- The software safety class and the rationale.
- The development process and the standards applied.
- The architecture and the partitioning.
- The SOUP list with versions and known anomalies.
- The unit, integration, and system tests.
- The unresolved anomalies and their risk.
- The configuration management and the build record.

A software section that lists the code files but not the process is a code dump.

### Step 8. Write the EMC and safety section

**Input:** the EMC plan, the safety analysis, and the test results.

**Output:** a section with the applicable standards, the test levels, the results, and the gaps.

**Rationale:** the EMC and safety sections are where the lab's limits are most visible. A pre-compliance result is not a type test, and the dossier must say so.

The section contains:

- The applicable standards and editions.
- The intended environment.
- The test levels and the performance criteria.
- The pre-compliance results and the accredited results, clearly separated.
- The safety analysis: insulation, leakage, grounding, thermal, mechanical.
- The gaps: which tests were not performed, and why.

### Step 9. Write the cybersecurity section

**Input:** the threat model, the SBOM, and the security tests.

**Output:** a section with the security architecture, the vulnerabilities, the mitigations, and the post-market plan.

**Rationale:** cybersecurity is a lifecycle obligation, not a feature. A reviewer will look for the threat model, the SBOM, and the vulnerability response plan.

The section contains:

- The threat model and the assets.
- The security architecture and the trust boundaries.
- The SBOM with versions and support status.
- The vulnerability assessment and the penetration test results.
- The patch and update process.
- The coordinated vulnerability disclosure plan.

### Step 10. Write the gap assessment

**Input:** the dossier and the intended claim.

**Output:** a table of gaps with the effort to close each one.

**Rationale:** the gap assessment is the most credible section of the dossier. It shows that the team knows what it does not have.

The table contains:

- The gap.
- The claim it affects.
- The evidence that is missing.
- The effort and cost to obtain it.
- The owner and the target date.
- The interim mitigation.

### Step 11. Conduct the peer review

**Input:** the draft dossier.

**Output:** a review record with findings and dispositions.

**Rationale:** a reviewer who was not in the room finds the gaps the team cannot see. The peer review is a rehearsal for the external review.

The review checks:

- The intended use and the claim are consistent.
- Every requirement has a verification.
- Every risk control has a verification.
- Every test has an evidence file.
- The results are consistent with the claims.
- The gaps are stated.
- The configuration is recorded.
- The limitations are honest.

A finding is recorded with a severity and a disposition. A finding that is deferred is a residual risk.

### Step 12. Assemble and index the dossier

**Input:** the sections and the evidence files.

**Output:** an indexed dossier with a master record and a configuration index.

**Rationale:** a dossier that cannot be navigated is not a dossier. The index is the map.

The dossier contains:

- The master record and the table of contents.
- The intended use and the claim.
- The requirements.
- The architecture.
- The risk file.
- The verification.
- The software lifecycle.
- The EMC and safety.
- The cybersecurity.
- The gap assessment.
- The references.
- The evidence index with hashes.

### Step 13. Verify the dossier against the standard

**Input:** the dossier and the applicable standard or guidance.

**Output:** a compliance check with the sections that are present and the sections that are missing.

**Rationale:** the standard defines the required content. A dossier that is missing a required section will be returned.

| Framework | Required content | Lab dossier section |
|---|---|---|
| EU MDR Annex II | Device description, labelling, design and manufacturing, GSPR, risk, verification | Sections 1–8 |
| FDA 21 CFR 820.30 | Design inputs, outputs, review, verification, validation, transfer, changes, DHF | Sections 1–6, 12 |
| IEC 62304 | Software plan, requirements, architecture, tests, release | Section 7 |
| ISO 14971 | Risk analysis, evaluation, control, residual risk | Section 4 |
| IEC 81001-5-1 | Security lifecycle activities | Section 9 |

The mapping is not a compliance claim. It is a checklist that helps the team find the missing sections.

### Step 14. Archive and hand over

**Input:** the dossier and the evidence.

**Output:** an archive that a reviewer can open without the team.

**Rationale:** the dossier outlives the project. An archive that depends on a personal account is not an archive.

The archive contains the dossier, the evidence, the configuration index, and a README that says how to navigate it. It is tested by a person who was not on the team.

### 5.15 Acceptance criteria for the dossier

| # | Criterion | Evidence | Fail action |
|---:|---|---|---|
| 1 | Intended use is written | Statement | Write |
| 2 | Claim has a boundary | Statement | Write |
| 3 | Requirements are numbered and testable | Table | Rewrite |
| 4 | Architecture has an interface table | Diagram and table | Add |
| 5 | Risk file has controls and residual risk | Table | Complete |
| 6 | Traceability matrix is complete | Matrix | Complete |
| 7 | Every requirement has a verification | Matrix | Add |
| 8 | Every risk control has a verification | Matrix | Add |
| 9 | Every test has an evidence file | Index | Add |
| 10 | Verification states uncertainty | Report | Add |
| 11 | Software section has a class and SOUP | Section | Add |
| 12 | EMC and safety sections separate pre-compliance from type tests | Section | Correct |
| 13 | Cybersecurity section has a threat model and SBOM | Section | Add |
| 14 | Gap assessment is specific | Table | Rewrite |
| 15 | Peer review is recorded | Review record | Review |
| 16 | Archive opens without the team | Test | Fix |

### 5.16 Branch points

**Branch A: the claim is a research claim, not a device claim.** The dossier can be shorter, but it still needs the claim boundary and the limitations. Do not use regulatory language for a research result.

**Branch B: the claim is a device claim.** The dossier needs the full structure, and the gaps are large. State them.

**Branch C: the device is already certified.** The dossier is a change file, not a full submission. It still needs the traceability and the risk assessment of the change.

**Branch D: the dossier is for due diligence.** The reviewer will focus on the claims and the gaps. The evidence index and the limitations are the most important sections.

## 6. Worked example

### 6.1 The question

A team has built a wearable arrhythmia monitor prototype. It has an ECG front end, firmware with a QRS detector and an atrial fibrillation (AF) detector, a HIL rig, and a set of lab reports. The team wants to assemble a dossier that supports a limited claim: the prototype detects AF on a defined set of records with a stated sensitivity, and it meets a latency requirement on the bench.

### 6.2 The intended use and the claim

The intended use is written as:

> The prototype is an educational wearable monitor that records a single-lead ECG and flags episodes of atrial fibrillation for review by a clinician. It is not a diagnostic device, is not for clinical use, and has not been validated in a clinical population.

The claim is:

> On the ten records of the evaluation set, with the annotation version 1.0.0 and a record-level split, the AF detector achieves a sensitivity of at least 0.90 at the episode level, and the end-to-end latency is at most 30 ms at p99 on the tested hardware.

The boundary is:

- Ten records, one database, one annotation version.
- A record-level split, no window-level leakage.
- Episode-level sensitivity, not beat-level.
- One hardware unit, one firmware commit.
- No clinical population, no usability study, no accredited EMC test.

### 6.2a The evidence inventory

Before writing the dossier, the team inventories the artefacts it has.

| Artefact | Owner | Status | Evidence file | Hash |
|---|---|---|---|---|
| Stack record | Systems | Complete | records/stack.md | yes |
| Calibration record | Sensors | Complete | records/cal.csv | yes |
| ECG noise result | Sensors | Complete | results/noise.json | yes |
| Firmware fault matrix | Firmware | Complete | results/faults.csv | yes |
| HIL latency log | Verification | Complete | results/lat.csv | yes |
| HIL fault captures | Verification | Complete | captures/*.png | yes |
| EMC pre-compliance plan | EMC | Draft | plans/emc.md | no |
| Secure boot test | Security | Not run | gap | no |
| Clinical evidence | Clinical | Not available | gap | no |

The inventory is the raw material. The dossier selects from it, links it, and states what is missing. An artefact that is complete but not linked is not evidence; it is a file.

### 6.3 The requirements

The team has 24 requirements. A sample:

| ID | Requirement | Verification | Acceptance | Test | Status |
|---|---|---|---|---|---|
| R-01 | Detect AF episodes | Test | Sensitivity ≥ 0.90 | T-01 | Verified |
| R-02 | Limit false alarms | Test | ≤ 1 per hour | T-02 | Verified |
| R-03 | Report latency | Test | p99 ≤ 30 ms | T-03 | Verified |
| R-04 | Detect lead-off | Test | ≤ 250 ms | T-04 | Verified |
| R-05 | Enter safe state | Test | ≤ 2 s | T-05 | Verified |
| R-06 | Protect against unauthorized update | Test | Reject unsigned image | T-06 | Gap |
| R-07 | Meet EMC immunity | Test | 10 V/m, home | T-07 | Gap |
| R-08 | Meet electrical safety | Test | IEC 60601-1 | T-08 | Gap |

Of the 24 requirements, 21 are verified, 2 are partially verified, and 1 is a gap. The three gaps are the secure update, the EMC immunity, and the electrical safety.

### 6.4 The risk file

The risk file has 14 hazards and 14 controls. A sample:

| ID | Hazard | Control | Verification | Residual risk |
|---|---|---|---|---|
| H-01 | Missed AF episode | Sensitivity requirement | T-01 | Medium |
| H-02 | False alarm | False-alarm requirement | T-02 | Low |
| H-03 | Delayed alarm | Latency requirement | T-03 | Low |
| H-04 | Lead-off unnoticed | Lead-off detection | T-04 | Low |
| H-05 | Device stuck in unsafe state | Safe-state requirement | T-05 | Low |
| H-06 | Malicious firmware | Secure boot | T-06 (gap) | High |
| H-07 | EMI causes false reading | EMC immunity | T-07 (gap) | High |
| H-08 | Electric shock | Isolation and leakage | T-08 (gap) | High |

Twelve of the 14 controls are verified. The two unverified controls are the secure boot and the EMC immunity. The third gap, electrical safety, is a requirement without a hazard in the current file, which is itself a finding: the hazard analysis missed the shock hazard.

### 6.5 The traceability matrix

| Requirement | Risk control | Test | Result | Evidence |
|---|---|---|---|---|
| R-01 | H-01 | T-01 | Sensitivity 0.93 | results/af_sens.json |
| R-02 | H-02 | T-02 | 0.1/h | results/far.json |
| R-03 | H-03 | T-03 | p99 19.6 ms | results/lat.csv |
| R-04 | H-04 | T-04 | 180 ms | captures/lead_off.png |
| R-05 | H-05 | T-05 | 1.2 s | captures/safe.png |
| R-06 | H-06 | T-06 | Not run | gap |
| R-07 | H-07 | T-07 | Not run | gap |
| R-08 | None | T-08 | Not run | gap |

The matrix has three gaps and one orphan requirement. The coverage is 21/24 = 87.5% of requirements and 12/14 = 85.7% of risk controls.

### 6.5a Reading the matrix

The matrix has three kinds of defect, and each has a different meaning.

| Defect | Example | Meaning | Action |
|---|---|---|---|
| Gap | R-06 has no test | The evidence does not exist | Run the test or narrow the claim |
| Orphan | R-08 has no control | The hazard analysis missed a hazard | Update the risk file |
| Weak link | T-12 has no hash | The evidence exists but is not controlled | Fix the record |

A matrix that is 100% covered can still have weak links. A matrix that has a gap is not a failure of the dossier; it is a statement about the evidence. The dossier's job is to make the statement explicit.

The matrix is reviewed row by row with the risk file open. A row that cannot be explained without the risk file is a row that the reviewer will question.

### 6.6 The evidence audit

The team audits the evidence files against the matrix.

| Check | Result | Finding |
|---|---|---|
| Every test has a result | 21 of 24 | Three gaps |
| Every result has a file | 21 of 21 | None |
| Every file has a hash | 21 of 21 | None |
| Every file is traceable to a configuration | 19 of 21 | Two files lack a commit |
| Every test states uncertainty | 17 of 21 | Four results lack an interval |
| Every requirement has a test | 23 of 24 | R-08 is an orphan |

The audit finds five documentation defects that the matrix alone did not show. A matrix can be complete and the evidence can still be weak.

### 6.7 The claim audit

The draft report claims:

> The device detects AF with 95% sensitivity.

The audit finds three problems.

**Problem 1: the split.** The sensitivity was computed on windows, and the windows from the same record appear in both the training and the test set. The leakage inflates the result. A record-level split is required.

**Problem 2: the unit.** The claim says "AF sensitivity," which is an episode-level quantity. The reported number is a beat-level or window-level sensitivity, which is a different quantity. A beat-level sensitivity of 0.95 can correspond to an episode-level sensitivity of 0.80.

**Problem 3: the population.** The ten records are a convenience sample from one database. The claim does not state the population, and the boundary does not state the limitation.

### 6.8 The corrected analysis

The team reruns the analysis with a record-level split and an episode-level definition.

| Analysis | Split | Unit | Sensitivity | 95% CI |
|---|---|---|---:|---|
| Original | Window-level | Window | 0.95 | 0.93–0.96 |
| Corrected | Record-level | Episode | 0.81 | 0.72–0.88 |
| Requirement | — | Episode | ≥ 0.90 | — |

The corrected sensitivity is 0.81, below the 0.90 requirement. The original claim was an artefact of the split and the unit.

The Wilson interval for the corrected result is 0.72 to 0.88. The interval excludes 0.90, so the result is a failure, not a marginal pass.

### 6.9 Two checks that would invalidate the corrected result

**Check 1: the episode definition was chosen after seeing the result.** If the team defined an episode as "a run of at least three consecutive positive windows" after looking at the detections, the definition is fitted to the data. The definition must be pre-specified, or the result must be labelled exploratory. The check is to write the definition in the protocol and compare it with the analysis script.

**Check 2: the record-level split was not stratified by rhythm.** If the ten records are split into five training and five test records, and the test records are all AF-heavy, the result is not representative. The split must be stratified by rhythm, patient, and signal quality, or the limitation must be stated. The check is to list the records in each split and compare their rhythm distributions.

### 6.10 A varied condition: the threshold is lowered

The team lowers the detection threshold to increase sensitivity and repeats the analysis.

| Threshold | Sensitivity | PPV | False alarms per hour | Verdict |
|---|---:|---:|---:|---|
| Original | 0.81 | 0.88 | 0.1 | Sensitivity fails |
| Lowered | 0.90 | 0.74 | 2.4 | Sensitivity passes, PPV and FAR fail |

The lower threshold raises the sensitivity to the requirement but lowers the PPV to 0.74 and raises the false alarms to 2.4 per hour, above the 1 per hour requirement. The result moves in the direction the trade-off predicts: sensitivity and false alarms move in opposite directions.

The lesson is that a single metric can be met by sacrificing another. The requirements must be met together, and the dossier must report all of them.

### 6.11 What the example demonstrates

The arithmetic closes: 21 of 24 requirements verified, 12 of 14 risk controls verified, a coverage of 87.5% and 85.7%, a corrected sensitivity of 0.81 with a 95% interval of 0.72 to 0.88, and a failure against the 0.90 requirement.

The example demonstrates the difference between a complete-looking matrix and a valid claim. The matrix was 87.5% covered, and the claim was still wrong because the split and the unit were wrong. The dossier's job is not to make the matrix look complete; it is to make the argument valid.

The example also demonstrates the value of the gap assessment. The three gaps, secure update, EMC, and electrical safety, are not failures of the lab. They are the boundary of what the lab could establish, and naming them makes the rest of the dossier more credible.

## 7. Code, query, or calculation

The program below audits a dossier index. It reads a requirements table and a test table, computes coverage, finds orphan tests and uncovered requirements, checks evidence links, and computes a Wilson interval for a sensitivity. It uses only the Python standard library.

```python
import csv
import io
import math

REQUIREMENTS_CSV = """id,statement,verification,test_id,status,risk_control
R-01,Detect AF episodes,Test,T-01,Verified,H-01
R-02,Limit false alarms,Test,T-02,Verified,H-02
R-03,Report latency,Test,T-03,Verified,H-03
R-04,Detect lead-off,Test,T-04,Verified,H-04
R-05,Enter safe state,Test,T-05,Verified,H-05
R-06,Protect against unauthorized update,Test,T-06,Gap,H-06
R-07,Meet EMC immunity,Test,T-07,Gap,H-07
R-08,Meet electrical safety,Test,T-08,Gap,
R-09,Record ECG,Inspection,T-09,Verified,
R-10,Store 24 hours,Test,T-10,Verified,
R-11,Display heart rate,Test,T-11,Verified,
R-12,Alarm within 1 s,Test,T-12,Verified,H-03
"""

TESTS_CSV = """id,requirement_id,result,evidence,has_hash,has_uncertainty
T-01,R-01,0.81,results/af_sens.json,yes,yes
T-02,R-02,0.1,results/far.json,yes,yes
T-03,R-03,19.6,results/lat.csv,yes,yes
T-04,R-04,180,captures/lead_off.png,yes,no
T-05,R-05,1.2,captures/safe.png,yes,no
T-06,R-06,not_run,gap,no,no
T-07,R-07,not_run,gap,no,no
T-08,R-08,not_run,gap,no,no
T-09,R-09,pass,results/record.json,yes,no
T-10,R-10,pass,results/storage.json,yes,yes
T-11,R-11,pass,results/display.json,yes,yes
T-12,R-12,0.9,results/alarm.json,no,yes
T-13,,pass,results/extra.json,yes,yes
"""

def read_csv(text):
    return list(csv.DictReader(io.StringIO(text)))

reqs = read_csv(REQUIREMENTS_CSV)
tests = read_csv(TESTS_CSV)

req_ids = {r["id"] for r in reqs}
test_ids = {t["id"] for t in tests}

verified = [r for r in reqs if r["status"] == "Verified"]
gaps = [r for r in reqs if r["status"] != "Verified"]
req_cov = len(verified) / len(reqs) * 100

print("requirement coverage")
print(f"requirements={len(reqs)} verified={len(verified)} "
      f"gaps={len(gaps)} coverage={req_cov:.1f}%")
for r in gaps:
    print(f"  gap: {r['id']} {r['statement']}")

# Risk control coverage: controls with a linked, verified test.
controls = {r["risk_control"] for r in reqs if r["risk_control"]}
verified_controls = {r["risk_control"] for r in verified if r["risk_control"]}
print("\nrisk control coverage")
print(f"controls={len(controls)} verified={len(verified_controls)} "
      f"coverage={len(verified_controls)/len(controls)*100:.1f}%")
for c in sorted(controls - verified_controls):
    print(f"  unverified control: {c}")

# Orphan tests and missing evidence.
orphans = [t for t in tests if not t["requirement_id"]
           or t["requirement_id"] not in req_ids]
missing_evidence = [t for t in tests if not t["evidence"] or t["evidence"] == "gap"]
no_hash = [t for t in tests if t["has_hash"] != "yes"]
no_unc = [t for t in tests if t["has_uncertainty"] != "yes" and t["result"] not in ("pass", "not_run")]

print("\ndocument checks")
print(f"orphan tests: {[t['id'] for t in orphans]}")
print(f"missing evidence: {[t['id'] for t in missing_evidence]}")
print(f"missing hash: {[t['id'] for t in no_hash]}")
print(f"missing uncertainty: {[t['id'] for t in no_unc]}")

# Requirements without a test row.
test_req = {t["requirement_id"] for t in tests}
uncovered = [r["id"] for r in reqs if r["id"] not in test_req]
print(f"requirements with no test row: {uncovered}")

# Wilson interval for the AF sensitivity.
def wilson(p, n, z=1.96):
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half

episodes, detected = 42, 34
p = detected / episodes
lo, hi = wilson(p, episodes)
print("\nAF sensitivity (episode level)")
print(f"p={p:.3f}  n={episodes}  95% CI=({lo:.3f}, {hi:.3f})")
print(f"requirement 0.90 -> {'PASS' if p >= 0.90 else 'FAIL'}")
print(f"interval excludes 0.90: {hi < 0.90}")

# A simple gap score: weighted count of missing evidence.
weights = {"unverified_requirement": 3, "unverified_control": 3,
           "orphan_test": 1, "missing_hash": 1, "missing_uncertainty": 1}
score = (weights["unverified_requirement"] * len(gaps)
         + weights["unverified_control"] * len(controls - verified_controls)
         + weights["orphan_test"] * len(orphans)
         + weights["missing_hash"] * len(no_hash)
         + weights["missing_uncertainty"] * len(no_unc))
print(f"\ngap score = {score} (lower is better; 0 is complete)")
```

Expected output:

```text
requirement coverage
requirements=12 verified=9 gaps=3 coverage=75.0%
  gap: R-06 Protect against unauthorized update
  gap: R-07 Meet EMC immunity
  gap: R-08 Meet electrical safety

risk control coverage
controls=7 verified=5 coverage=71.4%
  unverified control: H-06
  unverified control: H-07

document checks
orphan tests: ['T-13']
missing evidence: ['T-06', 'T-07', 'T-08']
missing hash: ['T-06', 'T-07', 'T-08', 'T-12']
missing uncertainty: ['T-04', 'T-05']
requirements with no test row: []

AF sensitivity (episode level)
p=0.810  n=42  95% CI=(0.667, 0.900)
requirement 0.90 -> FAIL
interval excludes 0.90: False

gap score = 22 (lower is better; 0 is complete)
```

The program's requirement coverage is 75.0%, not the 87.5% in the worked example, because the embedded table has 12 requirements rather than 24. The purpose is to show the arithmetic and the checks, not to reproduce the example's counts exactly.

The risk control coverage is 71.4%. The table has seven controls, because the electrical safety requirement R-08 has no risk control at all. That is worse than an unverified control: it is a hazard that the risk file does not contain. The two unverified controls, H-06 and H-07, are the secure update and the EMC immunity.

The document checks find an orphan test (T-13), three tests without evidence, four results without a hash, and two results without an uncertainty. These are the defects that a matrix alone does not show. The gap rows T-06 to T-08 have no hash because there is no file to hash; T-12 has a result and no hash, which is the more serious defect.

The Wilson interval for 34 of 42 episodes is 0.667 to 0.900. The point estimate is 0.810, below the 0.90 requirement. The upper bound just reaches 0.900, so the interval does not exclude the requirement, but the point estimate fails. This is a case where the interval and the point estimate tell different stories, and the report must state both.

The gap score of 22 is a summary, not a decision. It weights the unverified requirements and controls at 3, and the documentation defects at 1. A different weighting would give a different number. The score is useful for tracking progress, not for claiming completeness.

## 8. Data, standards, and regulatory context

### 8.1 Design controls in the United States

21 CFR 820.30 defines the design controls for Class II and Class III devices and for certain Class I devices. It requires design and development planning, design input, design output, design review, design verification, design validation, design transfer, design changes, and a design history file.

The design history file is the record set that demonstrates the design was developed according to the approved plan and the regulation. The dossier's verification section is a component of the DHF; it is not the whole file.

The FDA's Quality Management System Regulation (QMSR), effective 2 February 2026, incorporates ISO 13485:2016 by reference. Clause 7.3 covers design and development, and Clause 7.3.10 requires a design and development file for each device type or family.

### 8.2 The European technical documentation

Regulation (EU) 2017/745, the Medical Device Regulation, Annex II defines the technical documentation. The six sections are:

1. Device description and specification, including variants and accessories.
2. Information supplied by the manufacturer, including labels and instructions for use.
3. Design and manufacturing information.
4. General safety and performance requirements.
5. Benefit-risk analysis and risk management.
6. Product verification and validation.

Annex III covers the technical documentation on post-market surveillance. The documentation must be kept available for at least 10 years after the last device is placed on the market, and for 15 years for implantable devices, under Article 10(8).

A notified body reviews the technical documentation for higher-risk devices. The review is a structured assessment of the documentation against the regulation and the applicable standards.

### 8.3 Risk management

ISO 14971:2019 requires the manufacturer to identify hazards, estimate and evaluate the risk, implement controls, and evaluate the residual risk. The risk file is the record of the process.

The standard's Annex C gives examples of hazards, sequences of events, hazardous situations, and harms. The examples are not a checklist; they are illustrations of the structure.

The risk file is linked to the verification: each control has a verification, and each residual risk is stated. A control without a verification is a claim.

### 8.4 Software lifecycle

IEC 62304:2006 + Amendment 1:2015 defines the software lifecycle processes. The safety class determines the rigour. Class A applies when the software cannot contribute to a hazardous situation, Class B when it can contribute to a non-serious injury, and Class C when it can contribute to death or serious injury.

The standard requires a software development plan, requirements, architecture, detailed design, unit implementation, integration, system testing, release, and maintenance. The dossier's software section is the summary of that evidence.

### 8.5 Cybersecurity

IEC 81001-5-1:2021 defines the security activities across the health software lifecycle. The FDA recognises it as a framework for a secure product development framework.

The FDA's cybersecurity guidance, in the version current in 2026, treats a software bill of materials as part of the premarket submission for cyber devices. Section 524B of the Federal Food, Drug, and Cosmetic Act, added by the December 2022 Omnibus, requires a post-market vulnerability monitoring and response plan, processes for cybersecurity assurance and updates, and an SBOM.

The dossier's cybersecurity section is a summary; the full evidence includes the threat model, the penetration test, and the vulnerability response plan.

### 8.6 Predetermined change control

The FDA's guidance on predetermined change control plans for AI-enabled device software functions, issued in December 2024 and updated in 2025, defines a mechanism for pre-authorising certain changes. The plan describes the modifications, the modification protocol, and the impact assessment.

A dossier that includes a PCCP must state the changes that are covered and the changes that require a new submission. A change outside the plan is a new submission.

### 8.7 The master record and the evidence index

A dossier is navigated through two documents: the master record and the evidence index.

The master record lists the sections, the version, the owner, and the approval. It is the table of contents and the control record.

| Field | Content | Example |
|---|---|---|
| Document ID | Unique identifier | DOS-2026-001 |
| Version | Revision | 1.3 |
| Section | Section number and title | 6 Verification |
| Owner | Author | A. Patel |
| Reviewer | Independent reviewer | J. Chen |
| Status | Draft, reviewed, approved | Approved |
| Date | ISO 8601 | 2026-10-26 |

The evidence index lists every file, its hash, its configuration, and the requirement it supports.

| Field | Content | Example |
|---|---|---|
| Evidence ID | Unique identifier | E-021 |
| File | Relative path | results/lat.csv |
| Hash | SHA-256 | 9f3c... |
| Configuration | Build or model | fw-5c1d9a2 |
| Requirement | Linked requirement | R-03 |
| Test | Linked test | T-03 |
| Custodian | Who controls it | Verification lead |

The index is the difference between a folder and a dossier. A reviewer who can search the index can find the evidence for a claim; a reviewer who cannot must read every file.

### 8.8 What is binding and what is guidance

| Document | Status | Binds | Does not bind |
|---|---|---|---|
| 21 CFR 820.30 | Regulation | US device manufacturers | A teaching dossier |
| FDA QMSR | Regulation, effective 2026-02-02 | US device manufacturers | A teaching dossier |
| EU MDR Annex II | Regulation | EU device manufacturers | A teaching dossier |
| ISO 14971 | International standard | Medical device risk management | A lab hazard list |
| IEC 62304 | International standard | Medical device software | A prototype |
| IEC 81001-5-1 | International standard | Health software security | A prototype |
| FDA cybersecurity guidance | Guidance | Premarket submissions | A teaching dossier |
| FDA PCCP guidance | Guidance | AI-enabled device submissions | A teaching dossier |

## 9. Industry practice and case evidence

### 9.1 How a notified body reviews technical documentation

A notified body reviewer typically starts with the intended use and the classification, then checks the GSPR checklist, the risk file, the verification and validation, and the clinical evaluation. The review is a sample, not an exhaustive check: the reviewer follows the highest-risk claims and the weakest evidence.

The reviewer's questions are predictable:

- Is the intended use consistent with the claims?
- Is the risk file complete, and are the controls verified?
- Is the clinical evidence sufficient for the claim?
- Is the software lifecycle evidence complete for the class?
- Is the cybersecurity evidence proportionate to the risk?
- Is the post-market surveillance plan adequate?

A dossier that answers these questions in the order the reviewer asks them is easier to review than one that answers them in the order the team wrote them.

### 9.2 How the FDA reviews a premarket submission

The FDA's premarket review focuses on the intended use, the indications for use, the substantial equivalence (for a 510(k)), and the performance data. The review is supported by the eSTAR template and the recognized consensus standards.

The FDA's inspection programme focuses on the quality system: design controls, CAPA, complaints, and production. A design history file that cannot be reconstructed is a common inspection finding.

### 9.3 Due diligence

A technical due-diligence review before an investment or an acquisition focuses on the claims and the gaps. The reviewer asks:

- What is the strongest claim, and what evidence supports it?
- What is the weakest claim, and what would it take to support it?
- What is the regulatory pathway, and what is the remaining effort?
- What are the unresolved risks, and who owns them?

A dossier with a clear gap assessment is more valuable than one that claims completeness, because the gap assessment is the basis for the valuation and the plan.

### 9.4 Common findings

The common findings in dossier reviews are consistent across frameworks:

- A requirement without a verification.
- A risk control without a verification.
- A test without an evidence file.
- A result without an uncertainty.
- A claim that is broader than the evidence.
- A configuration that is not recorded.
- A limitation that is missing.

The findings are not usually technical errors. They are documentation and traceability errors, and they are the ones the dossier is designed to prevent.

### 9.5 What a due-diligence reviewer asks

A technical due-diligence reviewer works under time pressure and focuses on the claims that affect value and risk.

1. What is the strongest claim, and what evidence supports it?
2. What is the weakest claim, and what would it take to support it?
3. What is the regulatory pathway, and what is the remaining effort?
4. Which requirements have no verification?
5. Which risk controls have no verification?
6. Which evidence files are controlled, and which are on a laptop?
7. What is the software safety class, and is the lifecycle evidence complete?
8. What is the cybersecurity posture, and is there an SBOM?
9. What are the unresolved risks, and who owns them?
10. What would a regulator or a notified body ask first?

A dossier that answers these questions in a short, indexed form is more valuable than a longer dossier that requires a meeting to explain.

### 9.6 Inference

It is an inference, not a documented finding, that teams which write the traceability matrix before the final report produce dossiers with fewer gaps. The reasoning is that the matrix makes the missing links visible while there is still time to run the test. The inference is consistent with the practice in regulated industries, but it has not been measured in this course.

## 10. Failure modes and red flags

| # | Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---:|---|---|---|---|---|
| 1 | Claim broader than the evidence | "Detects all arrhythmias" | Overgeneralisation | Rejection, credibility loss | Write the boundary |
| 2 | Requirement without a verification | A row with no test | Missing test | Untested requirement | Matrix review |
| 3 | Risk control without a verification | A control with no link | Untested control | Unsupported residual risk | Matrix review |
| 4 | Test without an evidence file | "The result is in an email" | Missing record | Assertion, not evidence | Evidence index |
| 5 | Result without uncertainty | A single number | False precision | Wrong decision | Uncertainty budget |
| 6 | Orphan test | A test with no requirement | Scope creep | Wasted effort | Matrix review |
| 7 | Configuration not recorded | No commit, no BOM | Irreproducible | Unverifiable evidence | Configuration index |
| 8 | Split leakage | Windows from the same record in train and test | Inflated performance | Invalid claim | Record-level split |
| 9 | Unit mismatch | Beat-level number for an episode-level claim | Wrong quantity | Invalid claim | Define the unit |
| 10 | Pre-compliance presented as compliance | "Passed EMC" | No accreditation | Regulatory rejection | Label the evidence |
| 11 | Software class unsupported | No lifecycle evidence | Missing process | Rejection | IEC 62304 section |
| 12 | Cybersecurity section missing | No threat model, no SBOM | Incomplete submission | Rejection | Security section |
| 13 | Gap assessment vague | "More testing needed" | No plan | Reviewer loses confidence | Specific gaps with effort |
| 14 | Archive depends on a person | Files on a laptop | No retention | Evidence lost | Controlled archive |
| 15 | Peer review skipped | No review record | Undetected gaps | External finding | Independent review |
| 16 | Clinical claim from bench data | "Clinically accurate" | Wrong evidence rung | Regulatory and ethical problem | State the rung |
| 17 | Master record not controlled | Two versions in circulation | No version control | Reviewer uses the wrong version | Version and approval |
| 18 | Evidence index missing | Files found by memory | No index | Evidence cannot be located | Build the index |

## 11. Assumptions and limits

| # | Assumption | Why it is needed | What breaks it | How to test it |
|---:|---|---|---|---|
| 1 | The intended use is stable | The requirements depend on it | A change in the population or the environment | Review the statement |
| 2 | The requirements are complete | The verification is complete | A missing requirement | Review against the risk file |
| 3 | The risk file is complete | The controls cover the hazards | A missing hazard | Review against ISO 14971 Annex C |
| 4 | The evidence files are authentic | The dossier is trustworthy | An edited file | Hash and audit trail |
| 5 | The configuration is recorded | The evidence is reproducible | A different build | Configuration index |
| 6 | The tests are representative | The result transfers | A convenience sample | Describe the sample |
| 7 | The split is valid | The performance is not inflated | Leakage | Record-level split |
| 8 | The unit is consistent | The claim is meaningful | Beat versus episode | Define the unit |
| 9 | The uncertainty is stated | The result is interpretable | A single number | Uncertainty budget |
| 10 | The gaps are stated | The dossier is honest | A hidden gap | Gap assessment |
| 11 | The review is independent | The findings are real | The reviewer is the author | Assign a different person |
| 12 | The archive is durable | The evidence survives | A personal account | Test with a third party |
| 13 | The master record is controlled | The right version is reviewed | Two drafts in circulation | Compare the version and approval |
| 14 | The evidence index is complete | Every file is findable | A file outside the index | Search for the file by claim |

## 12. Exercises

### Exercise 1. Write the intended use and the claim (application)

**Deliverable:** a one-page intended-use statement and a claim with a boundary for the capstone prototype.

**Acceptance criteria:** the statement names the purpose, the population, the user, the environment, and the conditions of use; the claim is one sentence; the boundary lists at least four conditions; the statement says the prototype is not a medical device.

**Time:** 45 minutes.

### Exercise 2. Build the requirements table (application)

**Deliverable:** a table of at least 20 requirements with IDs, verification methods, acceptance criteria, and test links.

**Acceptance criteria:** every requirement is testable; every requirement has a verification method; at least one requirement comes from a standard and one from a risk control; the acceptance criteria are numeric where possible.

**Time:** 60 minutes.

### Exercise 3. Build the risk file (application)

**Deliverable:** a risk table with at least 10 hazards, sequences of events, hazardous situations, harms, controls, residual risks, and verification links.

**Acceptance criteria:** each hazard has a sequence of events; each control has a verification; the residual risk is stated; at least one hazard is a software fault and one is an electrical hazard.

**Time:** 90 minutes.

### Exercise 4. Build the traceability matrix (calculation)

**Deliverable:** a matrix linking requirements, risk controls, tests, results, and evidence files, with the coverage computed.

**Acceptance criteria:** every requirement has a test or a gap; every risk control has a test or a gap; orphan tests are identified; the coverage is computed for requirements and controls separately.

**Time:** 60 minutes.

### Exercise 5. Audit the evidence (critique)

**Deliverable:** an audit of at least ten evidence files against the matrix, checking the hash, the configuration, the uncertainty, and the claim.

**Acceptance criteria:** each file is checked against at least four criteria; the findings are specific; the audit distinguishes a documentation gap from a technical error.

**Time:** 45 minutes.

### Exercise 6. Write the gap assessment (design)

**Deliverable:** a table of at least six gaps, with the affected claim, the missing evidence, the effort to close, the owner, and the interim mitigation.

**Acceptance criteria:** each gap is specific; the effort is estimated in weeks or cost; the interim mitigation is stated; the gaps include clinical, EMC, safety, usability, and cybersecurity.

**Time:** 60 minutes.

### Exercise 7. Assemble the dossier (design)

**Deliverable:** a complete dossier for the capstone, with the sections in Section 5, an index, and an evidence appendix.

**Acceptance criteria:** the dossier is navigable; every section references the requirements; the evidence index has hashes; the gap assessment is included; the claim boundary is on the first page.

**Time:** 120 minutes.

### Exercise 8. Peer review (critique)

**Deliverable:** a review record for another team's dossier, with at least eight findings and dispositions.

**Acceptance criteria:** the findings cover the claim, the requirements, the risk file, the traceability, the evidence, the software, the EMC, and the gaps; each finding has a severity and a disposition; at least one finding is substantive.

**Time:** 60 minutes.

## 13. Self-check answers

### 13.1 What is the difference between a lab report and an evidence dossier?

A lab report describes an experiment and its result. An evidence dossier connects the result to a claim through requirements, risks, tests, and a gap assessment. A dossier can contain lab reports, but a collection of lab reports is not a dossier.

### 13.2 What is a traceability matrix?

A table or graph that links requirements, risk controls, tests, results, and evidence files. It is the argument that the evidence covers the requirements and the risks.

### 13.3 What is an orphan test?

A test that is not linked to a requirement. It may be useful, but it does not support a requirement, and it is a finding in a review.

### 13.4 What is a coverage gap?

A requirement or risk control that has no verification. It is the opposite of an orphan test: the requirement exists, but the evidence does not.

### 13.5 Why is a claim boundary necessary?

Because a claim is true only under stated conditions. The boundary tells the reader what the evidence does not cover, and it prevents the claim from being applied outside the tested conditions.

### 13.6 Compute the requirement coverage for 21 verified requirements out of 24.

\[
\text{Coverage} = \frac{21}{24} \times 100 = 87.5\%
\]

### 13.7 What is the difference between a DHF and a technical documentation file?

A DHF is the US design history file under 21 CFR 820.30. A technical documentation file is the EU MDR Annex II set. They overlap in content but serve different regulatory routes and have different required sections.

### 13.8 Why is a gap assessment more credible than a claim of completeness?

Because a reviewer will find the gaps anyway. A team that names them shows that it understands the evidence and has a plan. A team that claims completeness and is found to have gaps loses credibility.

### 13.9 What is the difference between verification and validation?

Verification confirms that the output meets the input requirement. Validation confirms that the device meets user needs and intended use. A dossier needs both, and a lab usually provides only the first.

### 13.10 Why is a window-level split a problem?

Because windows from the same record can appear in both the training and the test set, so the test is not independent. The performance is inflated. A record-level split, or a patient-level split, is required.

### 13.11 Why is a master record not the same as a table of contents?

A table of contents lists the sections. A master record lists the sections, the version, the owner, the reviewer, the approval, and the date. It is a control document, not a navigation aid. A dossier with a table of contents and no master record cannot show which version was reviewed.

### 13.12 Rubric for Exercise 4

A strong matrix links every requirement and control to a test and an evidence file, identifies orphans and gaps, and computes coverage separately for requirements and controls. A weak matrix lists tests without links or claims 100% coverage without checking.

### 13.12 Rubric for Exercise 8

A strong review has at least eight findings, each with a severity and a disposition, and at least one substantive finding about the claim or the evidence. A weak review comments on formatting and misses the claim.

## 14. Interview questions

### 14.1 What makes a dossier defensible?

A strong answer names the intended use, the requirements, the risk file, the traceability, the evidence, and the gap assessment. It says the dossier is an argument, not a folder. A weak answer says "good documentation."

### 14.2 What is the difference between a requirement and a test?

A requirement is a statement of what the device must do. A test is an activity that produces evidence about the requirement. A strong answer notes that a test without a requirement is an orphan. A weak answer uses the terms interchangeably.

### 14.3 How do you know the evidence is complete?

A strong answer uses the traceability matrix and checks that every requirement and control has a test and an evidence file. It also checks the configuration, the hash, and the uncertainty. A weak answer says "we have a lot of data."

### 14.4 What is a gap assessment?

A list of the evidence that is missing, with the claim it affects, the effort to obtain it, and the interim mitigation. A strong answer says it is the most credible section of the dossier. A weak answer calls it a limitations section.

### 14.5 Why is the intended use the first section?

Because it determines the requirements, the standards, the population, and the evidence. A dossier with a vague intended use has no testable requirements. A strong answer connects the intended use to the regulatory pathway.

### 14.6 What does a notified body look for first?

The intended use, the classification, and the GSPR checklist. A strong answer says the reviewer follows the highest-risk claims and the weakest evidence. A weak answer says "the test reports."

### 14.7 What is the difference between a nonconformity and an observation?

A nonconformity is a failure to meet a requirement. An observation is a condition that could lead to a nonconformity. A strong answer gives an example and notes that both are recorded with a disposition.

### 14.8 How do you handle a claim that the evidence does not support?

A strong answer narrows the claim to what the evidence supports, states the boundary, and records the gap. It does not delete the data or the finding. A weak answer rewrites the claim to match the data without disclosing the change.

### 14.9 What is the role of the peer review?

To find the gaps the team cannot see. A strong answer says the reviewer should be independent and should follow the reviewer's path: intended use, requirements, risk, verification, gaps. A weak answer says it is a proofread.

### 14.10 Why is the configuration index important?

Because the evidence is about a specific configuration. Without the index, a reviewer cannot tell which build, which board, or which model produced the result. A strong answer links it to the hash and the commit.

### 14.11 What would you do if a reviewer found an orphan test?

A strong answer determines whether the test supports a requirement that was not linked, or whether it is out of scope. It links it or records it as out of scope, and it updates the matrix. It does not delete the test.

### 14.12 What is the difference between a research claim and a device claim?

A research claim is about a finding under defined conditions. A device claim is about a product's safety and performance in its intended use. A strong answer says the evidence, the standards, and the regulatory route are different. A weak answer uses the same language for both.

## 15. Key terms

- **Acceptance criterion** — condition for pass or fail.
- **Audit** — systematic examination of records.
- **CAPA** — corrective and preventive action.
- **CER** — clinical evaluation report.
- **Claim** — sentence that asserts something about the device.
- **Claim boundary** — conditions under which the claim is true.
- **Configuration index** — record of the configuration of a device.
- **Coverage gap** — requirement or control with no verification.
- **Design history file** — US record set for design controls.
- **DHF** — design history file.
- **Disposition** — decision on a finding.
- **Dossier** — structured set of documents and evidence supporting a claim.
- **Due diligence** — investigation before an investment or acquisition.
- **Essential performance** — performance necessary to avoid unacceptable risk.
- **Finding** — result of an audit or review.
- **Gap assessment** — list of missing evidence and effort to close.
- **GSPR** — general safety and performance requirements.
- **Hazard** — potential source of harm.
- **Hazardous situation** — circumstance in which a person is exposed to a hazard.
- **Harm** — injury or damage to health.
- **Indications for use** — clinical conditions and populations.
- **Intended use** — use for which the device is intended.
- **Master record** — controlled index of the dossier.
- **Nonconformity** — failure to meet a requirement.
- **Objective evidence** — data supporting the existence or truth of something.
- **Observation** — condition that could lead to a nonconformity.
- **Orphan test** — test with no linked requirement.
- **PCCP** — predetermined change control plan.
- **PMS** — post-market surveillance.
- **Predicate** — device to which substantial equivalence is claimed.
- **Requirement** — statement of what the device must do.
- **Residual risk** — risk remaining after controls.
- **Risk control** — action that reduces risk.
- **Risk file** — record of hazards, controls, and residual risk.
- **SBOM** — software bill of materials.
- **SOUP** — software of unknown provenance.
- **Technical documentation** — EU MDR Annex II set.
- **Traceability** — ability to link requirements, risks, tests, and evidence.
- **Validation** — confirmation that the device meets user needs.
- **Verification** — confirmation that output meets input.
- **Evidence index** — controlled list of files with hashes and links.
- **Master record** — controlled index with versions and approvals.
- **Weak link** — evidence that exists but is not controlled or linked.
- **Operational definition** — measurable definition of a requirement.
- **Evidence hierarchy** — ranking of evidence by weight for a claim.
- **Claim audit** — check that a claim is supported by the evidence and the unit.
- **Coverage** — proportion of requirements or controls with verification.
- **Gap score** — weighted summary of missing evidence.
- **Traceability graph** — the network of links between requirements, risks, and tests.
- **Finding severity** — classification of a review finding by its impact.
- **Controlled document** — document with a version, owner, and approval.

## 16. Sources

All URLs accessed 26 October 2026 unless otherwise stated.

### Regulation and standards

- 21 CFR 820.30, Design Controls. https://www.govinfo.gov/content/pkg/CFR-2025-title21-vol8/pdf/CFR-2025-title21-vol8-sec820-30.pdf
- FDA, Quality Management System Regulation, Design and Development. https://www.fda.gov/media/189041/download
- Regulation (EU) 2017/745, Annex II, Technical Documentation. https://eur-lex.europa.eu/eli/reg/2017/745/oj
- ISO 13485:2016, Medical devices — Quality management systems. https://www.iso.org/standard/59752.html
- ISO 14971:2019, Medical devices — Application of risk management to medical devices. https://www.iso.org/standard/72704.html
- IEC 62304:2006+AMD1:2015, Medical device software — Software life cycle processes. https://www.iso.org/standard/64686.html
- IEC 81001-5-1:2021, Health software and health IT systems safety, effectiveness and security — Part 5-1: Security. https://webstore.iec.ch/
- IEC 60601-1:2005+AMD1:2012+AMD2:2020, Medical electrical equipment — General requirements. https://webstore.iec.ch/

### Guidance

- FDA, Cybersecurity in Medical Devices: Quality Management System Considerations and Content of Premarket Submissions. https://www.fda.gov/media/119933/download
- FDA, Marketing Submission Recommendations for a Predetermined Change Control Plan for Artificial Intelligence-Enabled Device Software Functions. https://www.fda.gov/media/185050/download
- FDA, Design Controls. https://www.fda.gov/media/142013/download
- Team-NB, Best Practice Guidance for the Submission of Technical Documentation under Annex II and III of MDR (EU) 2017/745, v2 (2023). https://www.team-nb.org/
- TÜV SÜD, Technical Documentation. https://www.tuvsud.com/

### Methods

- ISO/IEC 17025:2017, General requirements for the competence of testing and calibration laboratories. https://www.iso.org/standard/66912.html
- JCGM 100:2008, Guide to the expression of uncertainty in measurement. https://www.bipm.org/documents/20126/2071204/JCGM_100_2008_E.pdf

> **Note:** The standard texts are paywalled. The regulation and guidance above were used to check the dossier structure. Confirm every requirement against the current edition and the applicable jurisdiction before relying on it in a submission.

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`08_robotics-labs.md`](08_robotics-labs.md)
- Sibling: [`01_safety-and-standards.md`](01_safety-and-standards.md)
- Sibling: [`03_lab-conventions.md`](03_lab-conventions.md)
- Sibling: [`07_hil-labs.md`](07_hil-labs.md)
- Survey file: [`../../course/24_hardware_robotics_labs.md`](../../course/24_hardware_robotics_labs.md)
- Lab notebooks and fixtures: [`../../labs/`](../../labs/)
