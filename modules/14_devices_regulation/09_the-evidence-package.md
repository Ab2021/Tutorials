# The Evidence Package

> Module 14 — Medical Devices, SaMD, and Regulation
> Topic 09 of 10
> Estimated study time: 4 hours
> Prerequisites: 03, 04, 05, 06, 07
> Version 2026.3

## 1. Why this topic exists

Every regulatory submission asks the same question in different words: does the evidence support
the claim? A 510(k) asks whether the device is substantially equivalent to a predicate. A De Novo
request asks whether the benefits outweigh the risks for a new device type. A premarket approval
(PMA) asks whether there is valid scientific evidence of safety and effectiveness. An EU technical
documentation package asks whether conformity is demonstrated against the General Safety and
Performance Requirements. The legal tests differ. The underlying task does not: connect each word
of the claim to evidence that a reviewer can inspect.

Teams usually learn this late. The model is built, the validation report is written, the
regulatory lead asks for the evidence package, and the team discovers that the studies answer
questions nobody asked. The test set measures area under the receiver operating characteristic
curve (AUC), but the label says "prioritizes urgent cases." The usability study tested whether
nurses liked the screen, not whether they understood a negative result. The cybersecurity
assessment covers the cloud service but not the hospital's image archive interface. The
post-market plan is a sentence about "monitoring." None of these gaps is a scientific failure.
Each is a failure to connect a claim to a study before the study was designed.

The cost is concrete. A deficiency letter adds a review cycle, typically months, and the response
may require a new study rather than a new paragraph. A not-substantially-equivalent decision can
force a De Novo request with a different evidence standard. An EU notified body can suspend a
certificate when the clinical evaluation does not support the claimed clinical benefit. A launch
that proceeds on an unsupported claim invites a warning letter, a recall, or litigation. The most
expensive version of the mistake is discovering it after the product is in clinical use, when the
evidence gap becomes a patient-safety question.

The evidence package is also where the data scientist's work becomes legible to the rest of the
company. A model card, a validation report, and a notebook are not submission artifacts. They
become evidence only when they are tied to a claim, produced under a controlled protocol, and
summarized in a document that a reviewer can follow. That translation is a skill. It requires
knowing which numbers matter, which comparisons are valid, and which statements the data cannot
support.

This topic is not a template exercise. Templates exist, and following one without understanding
the argument produces a package that is complete in form and empty in substance. The topic teaches
the argument: how a claim decomposes into assertions, how each assertion maps to a study, how the
studies combine into a sufficiency judgment, and how a reviewer attacks the result. The same
structure applies to a 510(k), a De Novo request, a PMA, and an EU technical file, even though the
required documents differ.

A second reason the topic exists is that evidence planning is cheaper than evidence repair. A
pre-specification decision made before a study costs a meeting. The same decision made after the
data are collected costs a new study. Teams that plan the package at design freeze spend less and
file faster than teams that assemble it at the end. The difference is not regulatory bureaucracy.
It is the difference between designing a study that can answer the claim and hoping that an
existing study can be reinterpreted to fit it.

## 2. Learning outcomes

By the end of this topic, you can:

1. Decompose an intended-use statement into numbered, testable assertions and identify which
   assertions the current evidence does not address.
2. Select the evidence modules a given claim and pathway require, and explain why each module is
   necessary rather than merely customary.
3. Design an analytical validation that states its test-set construction, reference standard,
   metrics, subgroup plan, and acceptance criteria before data collection.
4. Distinguish a standalone performance study, a reader study, an external validation, and a
   prospective clinical investigation, and choose among them for a stated claim.
5. Compute an evidence-gap score from a weighted assertion list and defend or challenge the
   weights.
6. Calculate the case count for a paired reader study and explain the effect of reader count and
   intraclass correlation on the result.
7. Critique a submission summary for claim-evidence misalignment, unsupported labeling language,
   and missing traceability.
8. Run a reviewer challenge that produces a prioritized list of deficiencies before a regulator
   does.

The outcomes map to the exercises and self-check items as follows.

| Outcome | Exercise | Self-check |
|---|---|---|
| 1 | Exercise 1, Exercise 2 | Questions 1, 2 |
| 2 | Exercise 2, Exercise 5 | Question 3 |
| 3 | Exercise 3 | Question 4 |
| 4 | Exercise 3, Exercise 6 | Questions 5, 6 |
| 5 | Exercise 4 | Question 7 |
| 6 | Exercise 4 | Question 8 |
| 7 | Exercise 6, Exercise 8 | Question 9 |
| 8 | Exercise 7 | Question 10 |

## 3. The core mental model

The evidence package is an argument, not a binder. The claim is the thesis. Each evidence module
supports a premise. The submission is the argument written for a specific reader who is paid to
find the weak premise. A package that is thick but does not follow this structure fails review
for reasons the team cannot predict, because the reviewer is testing a different argument than the
one the team thinks it made.

```text
                    CLAIM: intended use, indications, negative claims
                                        |
                                        v
              +-------------------------+-------------------------+
              |                         |                         |
        Analytical                 Clinical                  Human and system
        assertions                 assertions                assertions
              |                         |                         |
   test set, metrics,        reader study, trial,       usability, cybersecurity,
   subgroups, locking        endpoint, analysis         software, risk management
              |                         |                         |
              +-------------------------+-------------------------+
                                        |
                                        v
                          EVIDENCE MAP (traceability)
             assertion -> question -> endpoint -> study -> result -> label clause
                                        |
                                        v
                     SUBMISSION: summary, reports, labeling, plan
                                        |
                                        v
                     REVIEWER CHALLENGE: simulate the deficiency letter
                                        |
                        +---------------+---------------+
                        |                               |
                  pass: file                      fail: new study or narrower claim
```

The first property of the model is that sufficiency is relative. There is no absolute amount of
evidence. A claim that a device "may help prioritize" needs less than a claim that it "detects"
disease, which needs less than a claim that it "rules out" disease. The evidence standard rises
with the strength of the assertion, the severity of the consequence of an error, and the
independence of the population from the development data. A package is sufficient when the
evidence meets the standard implied by the claim, not when it contains every document on a
checklist.

The second property is independence. Each study must be able to fail on its own. If the reader
study and the analytical study use the same cases, a defect in the case set contaminates both.
If the reference standard was produced by the same clinicians whose reports were used as training
labels, the clinical evidence and the analytical evidence share a bias. Reviewers look for shared
failure points because a single shared defect is more dangerous than two independent weaknesses.

The third property is pre-specification. An endpoint chosen after the data are seen is a
hypothesis, not a test. The protocol, the analysis plan, the subgroup list, and the acceptance
criteria should be fixed before the data are unblinded. This is not a formality. It is what
separates a confirmatory study from an exploration, and reviewers treat the difference as
material.

The fourth property is traceability. Every clause in the label should trace back to an assertion,
and every assertion should trace to evidence. A label statement with no evidence path is a
deficiency waiting to be written. An evidence report with no label clause is either unused or, if
it contradicts the label, a problem. The traceability matrix is the package's index, and building
it early exposes gaps while they are still cheap to fix.

The model has a weakest-link rule. The package is only as strong as its weakest necessary
assertion. A device with excellent analytical performance and no usability evidence fails the
claim that clinicians will use it correctly. A device with a strong reader study and an
unrepresentative test set fails the claim that the result generalizes. A device with every study
and an inconsistent label fails the claim that the labeling describes the device. Reviewers are
trained to find the weakest link, so the team should find it first.

There is also a direction of reading. A reviewer typically starts at the label, asks what the
claim requires, and works backward into the evidence. The team usually builds forward from the
data. The two paths meet only if the traceability matrix exists. When the team presents a package
organized by department rather than by assertion, the reviewer has to reconstruct the argument.
That reconstruction is where deficiencies are born.

Finally, the model distinguishes evidence from documentation. Documentation proves that a process
happened. Evidence shows that a claim is true. A design history file can be complete and the
evidence still insufficient. A test report can be well formatted and the test irrelevant. The
package needs both, but the sufficiency judgment is about evidence. Teams that confuse the two
produce submissions that pass a document audit and fail a scientific review.

## 4. Definitions and vocabulary

The terms below recur in submission summaries, review documents, and study protocols. Several
come from US Food and Drug Administration (FDA) guidance, including guidance on computer-assisted
detection (CADe), and from the EU Medical Device Regulation (MDR) and the International Medical
Device Regulators Forum (IMDRF). The "common misuse" column records the error that most often
causes a claim-evidence mismatch.

| Term | Definition | Common misuse | Source |
|---|---|---|---|
| Evidence package | The organized set of studies, analyses, and documents that supports each element of a claim for a defined regulatory pathway | Treating it as a document-assembly task rather than an argument | FDA submission types; MDR Annex IX |
| Claim | The statement of what the device does, for whom, and with what expected benefit, as expressed in labeling and promotion | Using a marketing sentence as the claim while the label says something narrower | 21 CFR 801.4; MDR Article 2(12) |
| Assertion | A testable proposition derived from the claim, such as "sensitivity is at least 0.85 in the intended population" | Skipping assertions and going straight from claim to study | Evidence-planning practice |
| Intended use | The objective intent of the persons responsible for labeling, including the general purpose and conditions of use | Confusing it with a feature list | 21 CFR 801.4 |
| Indications for use | The disease or condition, population, and circumstances in which the device is intended to be used | Omitting the population or setting, which broadens the claim beyond the evidence | FDA guidance; MDR Annex XIV |
| Analytical validation | Evidence that the algorithm or assay measures what it claims, under stated conditions, with stated precision and robustness | Reporting only AUC on a single curated set | IMDRF N41:2017 |
| Clinical validation | Evidence that the output is associated with the clinical condition and that use of the device improves or supports the clinical decision in the intended population | Assuming analytical accuracy is clinical validation | IMDRF N41:2017 |
| Valid clinical association | Evidence that the device's output is clinically associated with the target condition or clinical state | Citing a biological plausibility paper as if it were a validation study | IMDRF N41:2017 |
| Clinical performance | The ability of the device to yield results correlated with the clinical truth in the intended population and use | Measuring performance only in an enriched case set | IMDRF N41:2017; FDA CADe guidance |
| Standalone performance | Performance of the device output compared with a reference standard, without a clinician in the loop | Calling a standalone study a clinical validation | FDA CADe guidance |
| Reader study | A study in which clinicians interpret cases with and without the device output, measuring the effect on their decisions | Using readers who also designed the device without disclosing it | FDA CADe guidance |
| Multi-reader multi-case (MRMC) study | A reader study design in which multiple readers interpret multiple cases, allowing generalization to readers and cases | Treating readers as fixed and ignoring reader variability | FDA CADe guidance; Obuchowski methods |
| Reference standard | The independent determination of truth against which device output is compared | Using the original clinical report as truth when the device was trained on similar reports | FDA CADe guidance |
| Endpoint | The quantity measured to test an assertion, defined before data collection | Choosing the endpoint after seeing which one looks best | ICH E9; FDA pivotal guidance |
| Surrogate endpoint | An endpoint expected to predict clinical benefit but not itself a measure of benefit | Claiming clinical benefit from a surrogate without justification | FDA pivotal guidance |
| Pre-specification | Fixing the hypothesis, endpoint, analysis, subgroups, and acceptance criteria before unblinding | Writing the analysis plan after the data are available | ICH E9; study practice |
| Acceptance criterion | The numeric or qualitative threshold that must be met for an assertion to be considered supported | Setting the threshold after seeing the result | Protocol practice |
| Evidence gap | The difference between what an assertion requires and what the current evidence establishes | Describing a gap as "more data needed" without naming the missing assertion | Evidence-planning practice |
| Evidence map | A matrix linking each assertion to its question, endpoint, study, result, and label clause | Building the matrix after the submission is drafted | Traceability practice |
| Independent test set | A dataset not used in development, tuning, or threshold selection, ideally from separate sites | Calling a random split of the training data an independent test set | FDA good machine learning practice guidance; CADe guidance |
| External validation | Evaluation of a locked device on data from a site or population not represented in development | Assuming one external site establishes generalizability | Prediction-model literature |
| Subgroup analysis | Pre-specified analysis of performance within a defined population, site, device, or acquisition stratum | Running dozens of post-hoc subgroups and reporting the favorable ones | FDA CADe guidance; ICH E9 |
| Enrichment | Deliberately increasing the proportion of positive cases in a study set | Reporting positive predictive value from an enriched set as if it applied to the clinic | Epidemiology practice |
| Spectrum bias | Bias from studying a case mix that differs from the intended-use population | Validating only on severe, unambiguous cases | Epidemiology practice |
| Benefit-risk | The judgment that a device's probable benefits outweigh its probable risks for the intended use | Treating it as a score rather than a reasoned judgment | FDA; MDR Annex I |
| Summative usability | Validation that representative users can use the device safely and effectively in representative conditions | Relying on formative feedback or user satisfaction | IEC 62366-1 |
| Software verification | Evidence that software requirements were implemented and tested, with traceability to design and risk | Equating verification with clinical validation | IEC 62304 |
| Cybersecurity evidence | Threat model, security risk assessment, testing, and vulnerability management evidence for the device | Treating a penetration test as the whole cybersecurity file | FDA cybersecurity guidance |
| PMS / PMCF | Post-market surveillance and post-market clinical follow-up, the planned collection of real-world performance and safety data | Writing a generic monitoring sentence with no data source or trigger | MDR Annex XIV Part B |
| Deficiency | A reviewer's finding that information is missing or insufficient to support the claim | Treating it as a formatting issue rather than an evidence gap | FDA review practice |

Two distinctions deserve more than a table row. The first is analytical versus clinical
validation. Analytical validation answers whether the measurement is reliable: does the same
input produce the same output, is the output calibrated, does performance hold across scanners,
stains, devices, and sites, and is the test set independent. Clinical validation answers whether
the measurement means what the claim says in the population where it will be used. A model can be
analytically excellent and clinically useless if the label it predicts is not the condition that
matters, or if the population in which it was tested differs from the population in which it will
be deployed.

The second is a standalone study versus a reader study. A standalone study measures the device
against truth. A reader study measures the clinician-plus-device against the clinician alone. For
a claim that the device "detects," a standalone study may suffice. For a claim that the device
"improves reader performance" or "prioritizes review," a reader study or a workflow study is
usually necessary, because the claimed effect runs through human behavior. The distinction is not
about prestige. It is about which causal path the claim asserts.

A third distinction, often missed, is between evidence sufficiency and evidence volume. A
submission can contain forty studies and still fail because none of them tests the specific
assertion in the intended population. Volume is a by-product of a well-planned package, not a
goal. Reviewers do not award credit for pages. They look for the study that answers the question
the label raises.

## 5. The mechanics, step by step

The procedure below builds an evidence package in the order a reviewer will test it. It produces
two outputs: the package itself and a defensible record of why each element is sufficient. Work in
order, but expect to loop. A finding in clinical validation often sends the team back to the claim,
and a finding in labeling often sends the team back to the assertion list.

The procedure assumes the device is already classified and the pathway is known. Classification is
covered in Topic 01 and pathways in Topic 03. If those are unsettled, stop and settle them first,
because the pathway determines the evidence standard.

### Step 1 — Fix the claim and decompose it into assertions

**Input.** The draft intended-use statement, indications for use, output specification, user
workflow, and the commercial description of the product.

**Output.** A versioned claim, a numbered assertion list, and a pathway note that names the
evidence standard the claim triggers.

**Rationale.** A claim is not evidence-ready until it can be split into statements that are either
supported or not supported by data. "Improves emergency care" cannot be tested. "In adults with
suspected pneumothorax, the device flags cases at sensitivity of at least 0.85 against an
adjudicated reference standard" can be tested. The assertion list is the bridge between the
label's language and a study protocol.

**Procedure.** (1) Write the claim as one sentence with a user, population, setting, action, and
output. (2) Split it into assertions of four types: analytical, clinical, human-factors, and
system. (3) For each assertion, write the question it answers and the decision it informs. (4)
Mark each assertion as necessary or supporting. (5) Identify the pathway and the standard it
imposes on each necessary assertion. (6) Version the list and freeze it before protocol design.

| Assertion type | Example | Typical evidence |
|---|---|---|
| Analytical | The locked algorithm detects the target at the stated sensitivity on an independent set | Standalone validation report |
| Clinical | Use of the output changes clinician decisions or time to action in the intended population | Reader study or prospective trial |
| Human factors | Representative users interpret the output correctly and safely | Summative usability study |
| System | The device resists defined threats and meets its software requirements | Security assessment, verification record |

**Branches and edge cases.** If the claim contains a negative clause ("does not rule out
disease"), that clause is an assertion and needs evidence. If the claim contains a comparative
word ("faster," "more accurate"), the comparator must be defined and measured. If the claim
contains a population qualifier ("adults," "non-pregnant"), the population is an assertion about
generalization, not a labeling detail. If the team cannot state an acceptance criterion for an
assertion, the assertion is not yet testable.

### Step 2 — Write the intended use and indications for use

**Input.** The assertion list and the clinical workflow.

**Output.** An intended-use statement, indications for use, and an explicit list of excluded uses.

**Rationale.** The intended use is the legal anchor. It defines the population, the user, the
setting, and the action, and it bounds the evidence the reviewer may demand. A statement that is
too broad invites evidence the team cannot produce. A statement that is too narrow may fail to
cover the actual use and create a labeling violation in the field. The indications for use
translate the intended use into clinical terms that a reviewer can compare with the study
population.

**Procedure.** (1) Name the medical purpose as a verb: detect, diagnose, triage, prioritize,
monitor, predict, or inform. (2) Name the user by role and training. (3) Name the population with
inclusion and exclusion criteria that match the study. (4) Name the setting and the workflow
point. (5) Describe the output and what the user is expected to do with it. (6) State at least one
explicit limitation. (7) Check that every noun in the statement appears in the evidence plan.

**Branches and edge cases.** A triage or prioritization claim is a workflow claim and needs
workflow evidence. A "support" claim needs a description of what the user must independently
review. A claim that names a disease severity (for example, "clinically significant") needs a
definition of severity and a study that measures it. A claim that names a care setting needs
evidence from that setting, not from a referral center with different prevalence.

### Step 3 — Design and document analytical validation

**Input.** The locked algorithm, the intended input distribution, the reference standard, and the
assertion list.

**Output.** A validation protocol, a locked test set, an analysis plan, and a report with
confidence intervals and subgroup results.

**Rationale.** Analytical validation establishes that the measurement is trustworthy before
anyone asks whether it changes care. It covers discrimination, calibration where a probability is
reported, robustness to acquisition variation, and performance in the populations named in the
claim. The test set must be independent of development. Threshold selection, if any, must happen
on data separate from the test set, or the reported performance is optimistic.

**Procedure.** (1) Define the eligible population and the input acquisition conditions. (2) Build
the test set from sites and time periods not used in development. (3) Lock the algorithm,
preprocessing, and threshold. (4) Pre-specify primary and secondary metrics with confidence
intervals. (5) Pre-specify subgroups with a minimum sample size per subgroup. (6) Define the
reference standard and the adjudication process. (7) Analyze missing and ungradable cases as a
reported outcome, not as exclusions. (8) Record all deviations.

**Branches and edge cases.** For an in vitro diagnostic (IVD), analytical validation also covers
limit of detection, precision, linearity, interference, cross-reactivity, specimen stability, and
method comparison; a software-only study is not enough. For a combination product, the device
component's analytical evidence must be separated from the drug component's evidence, and the
primary mode of action determines the lead review center. For an imaging device, robustness to
scanner, protocol, and patient positioning belongs in analytical validation, not in a footnote.
If the test set is enriched, report the enrichment and do not present positive predictive value
as if it came from the intended prevalence.

### Step 4 — Design and document clinical validation

**Input.** The analytical result, the clinical assertion, and the intended-use population.

**Output.** A clinical evidence plan and, eventually, a clinical validation report.

**Rationale.** Clinical validation connects the output to the clinical condition and to the
clinical decision. The International Medical Device Regulators Forum (IMDRF) framework separates
three elements: a valid clinical association between the output and the target condition,
analytical validation of the output, and clinical validation that the device supports the claimed
clinical use in the intended population. A package that stops at analytical validation answers
only the middle element.

**Procedure.** (1) State the clinical association and cite the evidence for it. (2) Choose the
design: standalone comparison, reader study, prospective observational study, or randomized
trial. (3) Define the endpoint in terms of the claim, not in terms of what is easy to measure.
(4) Define the reference standard and its independence. (5) Pre-specify the analysis, including
subgroups and missing data. (6) Power the study for the smallest effect the claim requires. (7)
Register or document the protocol before enrollment.

**Branches and edge cases.** If there is no clinical endpoint, the claim must be limited to what
analytical and association evidence can support, and the labeling must say so. If a surrogate
endpoint is used, the package must justify the link between the surrogate and the clinical benefit
the claim implies. If a reader study is used, the readers must be representative, the case mix
must reflect the intended prevalence or the analysis must model it, and the reading conditions
must match the clinical workflow. If external validation is used, one external site is a start,
not a demonstration of generalizability. If a subgroup is named in the label, the study must be
powered for it; a post-hoc subgroup with a wide confidence interval is a red flag, not a result.

### Step 5 — Design and document usability and human factors

**Input.** The user interface, the use specification, the intended users, and the risk analysis.

**Output.** Formative findings, a summative validation report, and a use-related risk analysis.

**Rationale.** Many device failures are use failures, not algorithm failures. A correct output
that a clinician misreads, ignores, or over-trusts is a hazard. IEC 62366-1 requires a use
specification, identification of hazard-related use scenarios, formative evaluation, and
summative validation with representative users in representative conditions. The evidence must
show that users can interpret the output, including negative and uncertain results, and that they
take the intended action.

**Procedure.** (1) Write the use specification: users, environment, tasks, and outputs. (2)
Identify hazard-related use scenarios from the risk file. (3) Run formative studies and record
design changes. (4) Define summative tasks and acceptance criteria before recruiting. (5) Recruit
representative users, including users with the least training who are permitted by the label. (6)
Test comprehension of negative, indeterminate, and error states. (7) Report use errors and close
the loop to risk controls.

**Branches and edge cases.** If the device is intended for a user who is not a specialist, the
summative study must include that user group. If the output includes a probability, test whether
users understand calibration and base rates. If the claim depends on a specific workflow (for
example, "review within 60 minutes"), simulate that time pressure. A satisfaction survey is not
usability evidence. A formative study with five colleagues is not summative validation.

### Step 6 — Design and document cybersecurity evidence

**Input.** The system architecture, data flows, trust boundaries, and the security risk analysis.

**Output.** A threat model, a security risk assessment, verification results, and a vulnerability
management plan.

**Rationale.** A connected device is part of a socio-technical system. A compromise can change
the output, deny care, or expose patient data. FDA expects cybersecurity to be addressed through
the quality system and the premarket submission, with a threat model, security controls, testing,
and a plan for coordinated vulnerability disclosure and patching. The evidence must cover the
device, the network, the cloud service, and the update path.

**Procedure.** (1) Draw the data-flow and trust-boundary diagram. (2) Enumerate assets and
threats. (3) Assess exploitability and clinical impact. (4) Define controls and map them to
requirements. (5) Test the controls, including penetration testing where appropriate. (6) Define
the patch and disclosure process. (7) Describe the security bill of materials and support period.

**Branches and edge cases.** A device that only displays images still has an update path and an
interface to the image archive. A cloud service introduces availability and multi-tenant risks
that a local workstation does not. A model that updates remotely is also a change-control issue,
not only a security issue. A penetration test with no threat model documents findings without
showing coverage.

### Step 7 — Assemble software verification and lifecycle evidence

**Input.** Software requirements, architecture, unit and integration tests, and the risk file.

**Output.** A verification report, a requirements traceability matrix, and lifecycle records
consistent with IEC 62304.

**Rationale.** Software verification shows that the implementation matches the requirements.
Clinical validation shows that the requirements were the right ones. Both are necessary, and they
answer different questions. IEC 62304 requires a software development plan, requirements
analysis, architectural design, detailed design, unit verification, integration testing, system
testing, release, and maintenance, scaled to the software safety class.

**Procedure.** (1) Confirm the software safety class and the resulting process rigor. (2) Map each
requirement to design, code, and test evidence. (3) Confirm that safety requirements from the risk
file appear in the requirements. (4) Review the problem-resolution and configuration-management
records. (5) Confirm that the released version is the version that was validated. (6) Record
known anomalies and their risk assessment.

**Branches and edge cases.** A model update after validation is a new software version and needs
an impact assessment, even if the code around it is unchanged. A third-party or open-source
component needs a documented qualification and a security review. A configuration parameter that
changes behavior (a threshold, a model weight) is a software change, not an operational setting,
unless the change-control plan says otherwise.

### Step 8 — Integrate risk management

**Input.** The hazard analysis, the evidence reports, and the labeling draft.

**Output.** A risk management file that traces hazards to controls and controls to evidence,
including residual risk and benefit-risk.

**Rationale.** ISO 14971:2019 requires risk management across the lifecycle. In an evidence
package, the risk file is the map that shows why each piece of evidence exists. A hazard without a
control is an open risk. A control without verification is an assertion. A residual risk without a
benefit-risk judgment is an incomplete file. The risk file also identifies the evidence that is
missing: if a hazard depends on a subgroup, the subgroup analysis is a risk control verification.

**Procedure.** (1) Confirm the risk management plan and the policy for acceptable risk. (2)
Identify hazards and hazardous situations, including use, software, security, and algorithm
failure. (3) Estimate and evaluate risk. (4) Define controls in priority order: inherent safety,
protective measures, information for safety. (5) Verify each control with evidence. (6) Evaluate
residual risk and the overall benefit-risk. (7) Link to production and post-production monitoring.

**Branches and edge cases.** Labeling is the last resort in the control hierarchy, not the first.
A false-negative hazard cannot be controlled by a label alone if the label is not read at the
point of decision. A cybersecurity hazard may have a low probability and a high severity; the
benefit-risk judgment must say so explicitly. A hazard that is "out of scope" because the user is
a clinician is still a hazard if the clinician can reasonably misuse the output.

### Step 9 — Align labeling with the evidence

**Input.** The evidence reports, the intended use, and the draft label.

**Output.** A label whose indications, contraindications, warnings, limitations, and performance
claims each trace to an assertion and a study.

**Rationale.** The label is the part of the package that reaches the clinic. A performance claim
in the label that the study did not measure is a regulatory and liability exposure even if the
study was otherwise strong. The label must also state the limits the evidence revealed:
population exclusions, input quality requirements, prevalence dependence, and the meaning of a
negative result.

**Procedure.** (1) List every claim-bearing sentence in the label. (2) Trace each to an assertion
and a result. (3) Remove or qualify sentences with no trace. (4) Add limitations that the evidence
supports, including subgroup findings. (5) Check that the instructions for use describe the
workflow the study tested. (6) Check that promotional material uses the same claim language.

**Branches and edge cases.** A label that reports sensitivity without the prevalence and the
predictive values can mislead even when the sensitivity is correct. A label that says "not for
diagnosis" while the marketing says "detects" is internally inconsistent. A label that lists a
contraindication unrelated to the evidence suggests the evidence population was different from the
labeled population. If the evidence supports the claim only at a stated threshold, the threshold
is part of the label.

### Step 10 — Write the post-market plan and change-control plan

**Input.** The premarket evidence, the risk file, the planned changes, and the real-world data
sources available.

**Output.** A post-market surveillance plan, a post-market clinical follow-up plan where required,
and a predetermined change control plan (PCCP) where changes are planned.

**Rationale.** Premarket evidence is a sample. The post-market plan describes how the manufacturer
will detect drift, subgroup harm, and use errors that the studies were not powered to find. Under
the EU Medical Device Regulation (MDR), post-market clinical follow-up is part of the clinical
evaluation and must be planned in advance. Under the US framework, a PCCP can authorize specified
future modifications if the modification protocol, validation, and acceptance criteria are
reviewed up front.

**Procedure.** (1) Define the performance questions that remain open. (2) Name the data sources:
complaints, registries, real-world performance monitoring, customer feedback, literature. (3)
Define metrics, thresholds, and review cadence. (4) Define the actions triggered by a threshold
breach. (5) Define the change-control process and, if used, the PCCP scope. (6) Assign owners and
records. (7) Link the plan to the risk file and the labeling.

**Branches and edge cases.** A plan that says "monitor performance" without a metric, a data
source, and a trigger is not a plan. A PCCP cannot authorize arbitrary continuous learning; it
must bound the modifications and the validation. A device with no planned changes still needs a
complaint and vigilance process. An IVD needs post-market performance follow-up that addresses
the analyte and the intended use, not only general complaints.

### Step 11 — Assemble the submission

**Input.** The evidence reports, the risk file, the label, the software and security documents,
and the post-market plan.

**Output.** A submission package organized by the reviewer's questions, with a summary that states
the claim and the evidence for each element.

**Rationale.** Assembly is not collation. The summary is the argument in miniature. A reviewer who
reads only the summary should be able to see the claim, the evidence for each assertion, the
limitations, and the benefit-risk conclusion. The detailed reports support the summary; they do
not replace it. A package organized by department forces the reviewer to reconstruct the argument,
which is where misunderstandings and deficiencies begin.

**Procedure.** (1) Build the evidence map with one row per assertion. (2) Write the summary in the
reviewer's order: claim, device description, evidence, limitations, benefit-risk. (3) Check every
cross-reference. (4) Confirm the validated versions match the submitted versions. (5) Confirm the
label matches the evidence. (6) Assemble the administrative elements required by the pathway.
(7) Run a completeness check against the current guidance.

**Branches and edge cases.** A 510(k) needs a substantial-equivalence discussion that compares
intended use and technological characteristics and addresses differences. A De Novo request needs
a benefit-risk analysis and a proposed classification. A PMA needs an integrated summary of safety
and effectiveness and manufacturing information. An EU technical file needs the clinical
evaluation report and the post-market plan in the structure the regulation requires. The same
evidence can require different framing in each.

### Step 12 — Run the reviewer challenge

**Input.** The assembled package and the assertion list.

**Output.** A ranked list of deficiencies, each with the evidence needed to close it, and a
go/no-go recommendation.

**Rationale.** The reviewer's job is to find the weakest link. A team that simulates the review
finds it first, while the fix is still a study amendment rather than a new submission. The
challenge should be adversarial and independent: someone who did not design the study should
attack it, and the attack should be recorded.

**Procedure.** (1) Assign a red-team reviewer who was not on the study team. (2) Walk the
traceability matrix from label to evidence and from evidence to label. (3) For each necessary
assertion, ask: what would make a reviewer doubt this? (4) Test the three common attacks:
population mismatch, reference-standard dependence, and endpoint-claim mismatch. (5) Rate each
deficiency by severity and fix cost. (6) Decide: file, remediate, or narrow the claim.

**Branches and edge cases.** A deficiency that can be fixed by labeling is cheaper than one that
needs a study, but it may narrow the market. A deficiency in a supporting assertion may be
acceptable if the necessary assertions are strong. A deficiency that affects the primary endpoint
is a filing blocker. A package that passes the internal challenge can still fail review; the
challenge reduces the probability, not to zero.

### Decision table: evidence sufficiency

| Status | Test | Required action |
|---|---|---|
| Sufficient | The assertion is tested in the intended population, with an independent reference standard, pre-specified analysis, and precision that meets the acceptance criterion | Cite in the summary; no further study |
| Sufficient with a labeling constraint | The assertion holds in a narrower population or setting than the draft claim | Narrow the label, or add evidence for the broader claim |
| Partial | The direction is supportive but the precision, independence, or population coverage is inadequate | Extend the study or add a targeted study; do not present as sufficient |
| Absent | No study addresses the assertion | Design the study before filing; the claim is currently unsupported |
| Contradictory | The evidence points against the claim, or two studies disagree materially | Resolve before filing; a contradiction is a substantive problem, not a documentation problem |
| Not applicable | The assertion is not necessary for this claim or pathway | Remove it from the matrix and record why |

### Decision table: study design selection

| Claim type | Primary endpoint | Design | Sample-size driver | Main threat |
|---|---|---|---|---|
| Standalone detection | Sensitivity and specificity against truth | Retrospective independent test set | Precision of sensitivity at the expected prevalence | Spectrum bias, reference-standard error |
| Reader assistance | Difference in reader sensitivity or specificity | Multi-reader multi-case, crossover | Case and reader variance, effect size | Reading-order and recall effects |
| Triage or prioritization | Time to review or time to action | Simulated or live workflow study | Queue dynamics, event rate | Workflow simulation not matching practice |
| Rule-out | Negative predictive value | Prospective study at intended prevalence | Prevalence and false-negative rate | Prevalence drift, incomplete follow-up |
| Quantitative measurement | Bias and limits of agreement | Method comparison | Spread of values across the measuring range | Range restriction, sample handling |
| Treatment selection (IVD) | Agreement with the treatment decision | Concordance study with clinical outcome | Prevalence of the actionable marker | Specimen and population differences |
| Monitoring or alerting | Detection delay and false-alert rate | Prospective or replay study | Alert frequency, event rate | Alarm fatigue, missing data |

### Decision table: evidence depth by pathway

| Module | 510(k) | De Novo | PMA | MDR technical file |
|---|---|---|---|---|
| Analytical validation | Required | Required | Required | Required |
| Standalone clinical performance | Usually | Required | Required | Required |
| Reader or workflow study | If the claim is workflow-related | Usually | Usually | Usually |
| Prospective clinical investigation | Rare | Sometimes | Usually | Usually for Class IIb/III |
| Usability | Required | Required | Required | Required |
| Cybersecurity | Required | Required | Required | Required |
| Software lifecycle | Required | Required | Required | Required |
| Risk management file | Required | Required | Required | Required |
| Post-market plan | Required | Required | Required | Required, with PMCF |

The tables are a planning aid, not a substitute for the pathway-specific guidance. The depth of
each module also depends on risk, novelty, and the strength of the claim. A 510(k) for a
well-understood device with a narrow claim may need little clinical evidence; a 510(k) for a novel
algorithm with a broad claim may need a reader study even though the pathway does not formally
require one.

## 6. Worked example

The case is hypothetical. The product, sponsor, study numbers, and regulatory history are
constructed to show the reasoning. Real submissions require the current guidance, the current
database records, and professional regulatory review.

### 6.1 Product and claim

PulmoFlag is a cloud-based software product that analyzes adult chest radiographs acquired in an
emergency department. It returns one of two outputs per study: "suspected pneumothorax —
prioritize" or "no priority flag." The output appears in the radiology worklist and is visible to
the interpreting radiologist and the emergency physician. The product does not diagnose
pneumothorax, does not triage patients, and does not remove a study from the radiologist's queue.

The draft intended-use statement is:

"PulmoFlag is intended for use by radiologists and emergency physicians to prioritize adult
patients (18 years and older) whose emergency department chest radiographs are likely to show a
pneumothorax for earlier radiologist review. The software analyzes Digital Imaging and
Communications in Medicine (DICOM) chest radiographs and
flags studies for priority review. The output is an aid to worklist ordering and is not a
diagnosis. PulmoFlag is not intended for patients with a chest tube in place, patients in the
immediate post-operative period after thoracic surgery, or patients with a known pneumothorax
under observation."

The claim has four regulatory choices. The user is a clinician, not a patient. The population is
adults in an emergency department. The action is worklist prioritization, not diagnosis or
treatment. The exclusions remove the populations where a missed finding would be most dangerous
and where the training data are thin.

### 6.2 The evidence the claim requires

The claim decomposes into six necessary assertions. First, the algorithm detects pneumothorax on
the intended image types with sensitivity and specificity that meet the acceptance criteria.
Second, performance holds for the lesion sizes and patient groups named in the population. Third,
the output is calibrated enough that a "priority" flag means what the label implies. Fourth,
radiologists and emergency physicians interpret the flag correctly, including the absence of a
flag. Fifth, the device is secure and its software is verified. Sixth, the risk file shows that
the residual risks are acceptable and that the label addresses them.

The claim does not assert that the device improves diagnostic accuracy, reduces mortality, or
shortens length of stay. Those would be additional assertions requiring different studies. The
sponsor is tempted to add "reduces time to treatment" to the marketing because the workflow
benefit is the selling point. That sentence would convert a prioritization claim into a clinical
outcome claim and would require evidence the sponsor does not have. The claim stays narrow.

### 6.3 Pathway and evidence standard

The sponsor searches the FDA product classification database and finds a Class II product code for
radiological computer-assisted triage software. A predicate exists with a clinician-facing
prioritization claim for a different finding. The intended use is similar in kind but not
identical, and the technological characteristics are similar: a machine-learning classifier
applied to DICOM images, with a worklist integration.

The sponsor selects a 510(k) with a substantial-equivalence argument. The differences are the
specific finding, the two-output design, and the emergency-department setting. The sponsor must
show that these differences do not raise different questions of safety and effectiveness, or
address them with performance data. The evidence standard is not "no clinical evidence"; it is
"enough evidence to support the intended use and to address the differences from the predicate."
The sponsor plans analytical validation, a reader study, usability, cybersecurity, software
verification, and a risk file.

### 6.4 Analytical evidence and the arithmetic of prevalence

The algorithm is locked at version 2.3.1. The test set contains 1,200 studies from four sites, of
which 180 are positive for pneumothorax (15.0 percent) and 1,020 are negative. The set is enriched
relative to the intended setting, where the prevalence is estimated at 3.1 percent. Sensitivity is
0.88 and specificity is 0.94 on the test set.

The overall numbers look acceptable. The prevalence arithmetic changes the picture. At 15 percent
prevalence, the positive predictive value (PPV) is 0.721: about 72 percent of flags are true
positives. At the intended prevalence of 3.1 percent, PPV is 0.319. Two of every three flags are
false positives. The negative predictive value is 0.996, so an unflagged study is very unlikely to
be positive, but the flag itself is a weak signal.

The false-flag burden is the operational consequence. At 15 percent prevalence there are 0.39
false flags per true flag. At 3.1 percent there are 2.13. A radiologist who sees a flag must
verify it, and the verification time is the cost the workflow claim has to justify. The label must
report the predictive values at the intended prevalence, not only the sensitivity and specificity
from the enriched set.

The subgroup analysis is the second analytical problem. Among the 180 positive studies, 72 have a
small pneumothorax (less than 2 cm) and 108 have a larger one. Sensitivity is 0.78 in the small
group (95 percent confidence interval 0.67 to 0.86) and 0.95 in the larger group (0.90 to 0.98).
The overall sensitivity of 0.88 is a weighted average of two different products. The claim does
not mention lesion size, so a reviewer will read the label as covering both. The sponsor must
either narrow the claim to "clinically significant pneumothorax" with a definition, or power a
study for the small-lesion subgroup. The current study has 72 small-lesion cases, which is enough
to detect a large difference but not enough to establish a precise sensitivity.

### 6.5 The clinical evidence gap

The sponsor has no reader study. The clinical evidence is a retrospective standalone comparison
against the original radiology reports. That is an analytical study with a clinical label. The
claim says "prioritize for earlier review," which is a workflow claim. The causal path runs from
the flag to the worklist order to the review time to the clinical action. A standalone study
measures none of those steps.

The evidence gap is therefore not "does the model detect pneumothorax." The gap is "does the flag
change the order and timing of review in a way that is safe and useful, and does the label prepare
the user for the false-flag burden at the intended prevalence?" The sponsor needs a reader or
workflow study with time to review as the primary endpoint, a reference standard independent of
the original reports, and a case mix that reflects the intended prevalence or models it
explicitly.

The planned reader study has 200 cases, 8 readers, and a crossover design. The primary endpoint is
time from case availability to first review for positive cases. The sponsor wants to know whether
the study is powered. The paired case-count calculation in Section 7 gives 202 cases per reader
under the stated discordance assumptions, before accounting for reader clustering. With eight
readers and an intraclass correlation of 0.10, the design-effect heuristic gives 344 cases. The
planned 200 cases is underpowered for the effect the sponsor wants to claim. The study would need
either more cases, more readers, or a larger assumed effect.

### 6.6 Two diagnostics that would sink the package

**Diagnostic 1: the reference standard is not independent.** Of the 1,020 negative studies, only
120 have computed tomography (CT) confirmation or clinical follow-up. The remaining 900 are
negative because the original radiology report said so. That report was produced by the same
clinical service whose reports were used to build the training labels. When the sponsor
re-adjudicates the 120 CT-confirmed negatives, 9 (7.5 percent) contain a pneumothorax that the
original report missed. If that rate generalizes to the unverified negatives, true specificity is
about 0.925, not 0.94, and PPV at 3.1 percent prevalence falls from 0.319 to 0.273. More
importantly, the reviewer cannot tell whether the model learned the disease or the reporting
style. A reference standard that depends on the same reports used for training is circular, and a
reviewer will say so.

**Diagnostic 2: the clinical study cannot support the claim it is attached to.** The only
clinical evidence is a retrospective accuracy study. The claim is about prioritization, which is a
workflow effect. Even a perfect accuracy result would not show that the flag changes review order,
that clinicians act on it, or that the change is safe at the intended prevalence. A reviewer will
treat the claim and the evidence as misaligned and will request a reader or workflow study. If the
sponsor responds by narrowing the claim to "identifies studies that may show pneumothorax," the
market value falls and the label must still explain the false-flag rate. If the sponsor keeps the
claim, the study is the rate-limiting step.

Either diagnostic is sufficient to prevent a favorable decision. Together they show the pattern
the package is supposed to prevent: strong analytical work, a plausible clinical story, and no
evidence for the specific sentence the label wants to print.

### 6.7 The submission decision

The sponsor does not file. The decision is recorded with three findings and a remediation plan.

First, the claim is narrowed from "prioritize for earlier review" to "flags studies for priority
review; does not diagnose pneumothorax and does not remove studies from the review queue." The
narrower claim is still a workflow claim, so it still needs workflow evidence, but it no longer
implies a queue-removal or rule-out benefit.

Second, the reference standard is rebuilt. All positive and a random sample of negative studies
receive CT confirmation or structured clinical follow-up, with adjudication by two radiologists
who are blinded to the model output and to each other, and a third adjudicator for disagreements.
The test set is re-analyzed with the new standard, and the report states the verification rate for
negatives.

Third, the reader study is redesigned. The primary endpoint is time to first review at the
intended prevalence, with a pre-specified analysis that models the queue. The case count is set
from the paired calculation with a reader-clustering inflation, and the sponsor commits to at
least 8 readers and 350 cases. The study includes a subgroup analysis for small pneumothorax, and
the protocol pre-specifies that the claim will be limited if the subgroup result does not meet the
acceptance criterion.

The sponsor also requests a pre-submission meeting to discuss the reader-study endpoint, the
reference standard, and the acceptability of the prevalence-adjusted labeling. The meeting does
not guarantee clearance, but it converts three likely deficiencies into a documented discussion.

### 6.8 Vary one assumption

Change the claim from "flags studies for priority review" to "autonomously determines that a chest
radiograph does not show a pneumothorax and removes negative studies from the radiologist queue."
The model is the same. The evidence requirement is not.

The claim becomes a rule-out claim. The relevant metric becomes negative predictive value at the
intended prevalence, and the tolerance for false negatives becomes very small because a removed
study may never be read. A sensitivity of 0.88 with a small-lesion sensitivity of 0.78 is not
compatible with that claim. The pathway is unlikely to be a 510(k) with the existing predicate,
because the intended use is different in kind; a De Novo request or a PMA becomes plausible. In
the EU, the possible harm from a missed pneumothorax includes serious deterioration or death, so
the classification analysis moves toward Class III and a prospective clinical investigation.

| Assumption changed | Pathway | Evidence consequence | Submission effect |
|---|---|---|---|
| Base: flags for priority review, predicate available | 510(k), Class II | Analytical, reader or workflow study, usability | File after remediation |
| Claim becomes rule-out with queue removal | De Novo or PMA | Prospective study with NPV endpoint, complete follow-up | Not fileable on current data |
| Claim narrowed to large pneumothorax only | 510(k) | Subgroup becomes the primary population; re-power | Possible with a new analysis and label |
| Prevalence assumed 8 percent instead of 3.1 percent | 510(k) | PPV rises to about 0.56; false-flag burden falls | Labeling and workflow claim easier to support |
| Reader study uses AUC instead of time to review | 510(k) | Endpoint no longer matches the workflow claim | Reviewer requests a new study |

The last row is the most common error. A team that cannot measure time to review will substitute
AUC because it is available. The substitution changes the question. An AUC of 0.93 does not show
that a radiologist reviews a flagged study earlier, and a reviewer will notice the gap between the
endpoint and the claim.

## 7. Code, query, or calculation

The script below implements two planning tools: a weighted evidence-gap score and a paired
reader-study case-count calculation. The gap score is a team convention, not a regulatory
formula. The weights encode how much each assertion matters to the claim; they must be agreed
before the scoring exercise, or the score becomes a way to rationalize a decision already made.
The case-count calculation uses a paired-proportion formula and a design-effect heuristic for
reader clustering. It is a planning estimate, not a substitute for a statistician's multi-reader
multi-case model.

```python
"""Evidence-package planning tools for Module 14, Topic 09.

Teaching aid only. Inputs are illustrative.
"""

from __future__ import annotations

from math import ceil, sqrt
from statistics import NormalDist


# One row per assertion: id, label, weight, maturity in [0, 1].
ASSERTIONS = [
    ("A1", "Claim boundary and negative claims", 0.10, 0.90),
    ("A2", "Analytical validation, representative test set", 0.15, 0.80),
    ("A3", "Subgroup performance", 0.10, 0.40),
    ("A4", "Clinical validation for the prioritization claim", 0.20, 0.50),
    ("A5", "External and multi-site validation", 0.10, 0.20),
    ("A6", "Usability and human factors", 0.10, 0.70),
    ("A7", "Cybersecurity", 0.07, 0.80),
    ("A8", "Software verification and lifecycle", 0.06, 0.90),
    ("A9", "Risk-management traceability", 0.07, 0.60),
    ("A10", "Labeling consistency", 0.03, 0.50),
    ("A11", "Post-market and change-control plan", 0.02, 0.30),
]


def evidence_gap(rows):
    """Return the weighted gap score, a band, and rows ranked by contribution."""
    total_weight = sum(row[2] for row in rows)
    if abs(total_weight - 1.0) > 1e-9:
        raise ValueError(f"weights must sum to 1.0, got {total_weight:.4f}")
    for row in rows:
        if not 0.0 <= row[3] <= 1.0:
            raise ValueError(f"maturity out of range: {row}")
    contributions = [(r[0], r[1], r[2], r[3], r[2] * (1.0 - r[3])) for r in rows]
    score = round(100.0 * sum(c[4] for c in contributions), 1)
    if score <= 10.0:
        band = "submission-ready"
    elif score <= 25.0:
        band = "targeted remediation"
    else:
        band = "not filable"
    ranked = sorted(contributions, key=lambda c: c[4], reverse=True)
    return score, band, ranked


def paired_cases(p_aided_only: float, p_unaided_only: float,
                 alpha: float = 0.05, power: float = 0.80) -> int:
    """Cases per reader for a paired sensitivity comparison (Connor-type formula).

    p_aided_only: proportion correct with the device and incorrect without it.
    p_unaided_only: proportion correct without the device and incorrect with it.
    """
    if not 0.0 < p_aided_only < 1.0 or not 0.0 < p_unaided_only < 1.0:
        raise ValueError("discordance proportions must be in (0, 1)")
    psi = p_aided_only + p_unaided_only
    delta = p_aided_only - p_unaided_only
    if delta <= 0:
        raise ValueError("the device must be assumed to improve sensitivity")
    z_alpha = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    z_beta = NormalDist().inv_cdf(power)
    n = (z_alpha * sqrt(psi) + z_beta * sqrt(psi - delta ** 2)) ** 2 / delta ** 2
    return ceil(n)


def clustered_cases(cases_per_reader: int, readers: int, icc: float) -> int:
    """Inflate cases for reader clustering with a design-effect heuristic."""
    if readers < 1 or not 0.0 <= icc < 1.0:
        raise ValueError("readers must be >= 1 and icc in [0, 1)")
    design_effect = 1.0 + (readers - 1) * icc
    return ceil(cases_per_reader * design_effect)


def ppv(sensitivity: float, specificity: float, prevalence: float) -> float:
    """Positive predictive value at a stated prevalence."""
    tp = sensitivity * prevalence
    fp = (1.0 - specificity) * (1.0 - prevalence)
    return tp / (tp + fp)


if __name__ == "__main__":
    score, band, ranked = evidence_gap(ASSERTIONS)
    print(f"Evidence gap score: {score:.1f} / 100 -> {band}")
    print("\nRanked contributions (weight x unmet maturity):")
    for ident, label, weight, maturity, contribution in ranked:
        print(f"  {ident:3s} {label:48s} w={weight:.3f} m={maturity:.2f} c={contribution:.4f}")

    print("\nPaired reader-study cases per reader:")
    base = paired_cases(0.15, 0.06)
    print(f"  discordance 0.15 / 0.06 -> {base} cases per reader")

    print("\nCases after reader-clustering inflation:")
    for readers in (4, 8, 12):
        for icc in (0.10, 0.20):
            print(f"  readers={readers:2d} icc={icc:.2f} -> {clustered_cases(base, readers, icc)} cases")

    print("\nPPV at sensitivity 0.88, specificity 0.94:")
    for prevalence in (0.15, 0.031):
        print(f"  prevalence={prevalence:.3f} -> PPV={ppv(0.88, 0.94, prevalence):.3f}")
    print("  specificity 0.925, prevalence 0.031 -> "
          f"PPV={ppv(0.88, 0.925, 0.031):.3f}")
```

Expected output from a Python 3.11 or 3.12 run:

```text
Evidence gap score: 38.7 / 100 -> not filable

Ranked contributions (weight x unmet maturity):
  A4  Clinical validation for the prioritization claim w=0.200 m=0.50 c=0.1000
  A5  External and multi-site validation               w=0.100 m=0.20 c=0.0800
  A3  Subgroup performance                             w=0.100 m=0.40 c=0.0600
  A6  Usability and human factors                      w=0.100 m=0.70 c=0.0300
  A2  Analytical validation, representative test set   w=0.150 m=0.80 c=0.0300
  A9  Risk-management traceability                     w=0.070 m=0.60 c=0.0280
  A10 Labeling consistency                             w=0.030 m=0.50 c=0.0150
  A7  Cybersecurity                                    w=0.070 m=0.80 c=0.0140
  A11 Post-market and change-control plan              w=0.020 m=0.30 c=0.0140
  A1  Claim boundary and negative claims               w=0.100 m=0.90 c=0.0100
  A8  Software verification and lifecycle              w=0.060 m=0.90 c=0.0060

Paired reader-study cases per reader:
  discordance 0.15 / 0.06 -> 202 cases per reader

Cases after reader-clustering inflation:
  readers= 4 icc=0.10 -> 263 cases
  readers= 4 icc=0.20 -> 324 cases
  readers= 8 icc=0.10 -> 344 cases
  readers= 8 icc=0.20 -> 485 cases
  readers=12 icc=0.10 -> 425 cases
  readers=12 icc=0.20 -> 647 cases

PPV at sensitivity 0.88, specificity 0.94:
  prevalence=0.150 -> PPV=0.721
  prevalence=0.031 -> PPV=0.319
  specificity 0.925, prevalence 0.031 -> PPV=0.273
```

Three cautions apply. First, the gap score is only as good as the weights. A team that wants to
file will set weights that produce a passing score. The weights should be fixed by the regulatory
lead and the clinical lead before the evidence is reviewed, and the score should be reported with
the ranked contributions so a reader can see which assertions drive it. Second, the case-count
formula assumes independent cases within a reader and does not model the correlation between
readers beyond the design-effect heuristic. A real multi-reader multi-case study needs a
statistician and a variance-component model. Third, the PPV calculation assumes the sensitivity
and specificity are transportable to the intended prevalence. That assumption is exactly what
external validation tests, and it is often false.

## 8. Data, standards, and regulatory context

> **Note:** This section is educational and is not legal advice. Submission content and evidence
> sufficiency are fact-specific regulatory judgments. Confirm the current statutes, guidance, and
> standards with qualified regulatory counsel before relying on them.

The evidence package is governed by a mix of binding law, agency guidance, and voluntary standards
that become binding through incorporation or contract. The table separates them.

| Source | Status | What it constrains | What it does not do |
|---|---|---|---|
| FD&C Act §513 and §515 | Binding US statute | Requires substantial equivalence or valid scientific evidence, depending on the pathway | Does not specify study designs or endpoints |
| 21 CFR 820 / Quality Management System Regulation (QMSR) | Binding US regulation | Requires design controls, verification, validation, and risk management records, effective 2 February 2026 | Does not prescribe a clinical study design |
| FDA QMSR final rule | Binding US regulation | Incorporates ISO 13485:2016 by reference | Does not replace device-specific guidance |
| FDA CADe clinical performance guidance, final 28 September 2022 | Guidance | Describes clinical performance assessment for computer-assisted detection, including multi-reader multi-case studies, sample-size justification, independent test sets, and subgroup cohorts | Is not a safe harbor and does not bind a court |
| FDA pivotal clinical investigations guidance, 2013 | Guidance | Describes design considerations for pivotal studies, including endpoints, controls, and analysis populations | Does not require a pivotal study for every device |
| FDA clinical decision support guidance, final 28 September 2022 | Guidance | Describes the CDS exclusion analysis and the evidence expectations for software functions | Does not create a general exemption |
| FDA device software functions guidance | Guidance | Describes documentation expectations for device software functions | Does not establish clinical validity |
| IMDRF/SaMD WG/N41FINAL:2017 | International guidance | Defines the clinical evaluation elements: valid clinical association, analytical validation, clinical validation | Is not law in any jurisdiction |
| MDR 2017/745, Annex XIV | Binding EU regulation | Requires clinical evaluation, clinical evidence, and post-market clinical follow-up for devices | Does not prescribe a US-style submission format |
| ISO 14971:2019 | Voluntary standard; FDA-recognized consensus standard | Sets risk management requirements across the lifecycle | Does not establish clinical effectiveness |
| IEC 62304:2006+A1:2015 | Voluntary standard; FDA-recognized consensus standard | Sets software lifecycle processes by safety class | Does not set the regulatory device class |
| IEC 62366-1:2015+A1:2020 | Voluntary standard; FDA-recognized consensus standard | Sets usability engineering requirements, including summative validation | Does not establish clinical benefit |
| ISO 14155:2020 | Voluntary standard | Sets good clinical practice for device investigations | Does not replace local trial regulation |
| FDA cybersecurity guidance, 2023 | Guidance | Describes security risk management and premarket content | Does not certify a device as secure |

The IMDRF clinical evaluation framework is the clearest statement of the three-element structure.
Valid clinical association asks whether the output is meaningfully related to the target
condition. Analytical validation asks whether the output is produced accurately and reliably.
Clinical validation asks whether the device supports the claimed clinical use in the intended
population. The three elements are not interchangeable, and a package that supplies one cannot
compensate for the absence of another. The framework is guidance, not law, but it is widely used
by regulators and manufacturers to organize clinical evidence.

The FDA computer-assisted detection guidance is the most specific public statement on reader
studies for imaging software. It describes multi-reader multi-case designs, the need to justify
case and reader sample sizes, the use of independent test sets, and the analysis of subgroup
cohorts. It also discusses the difference between a study that measures stand-alone performance
and one that measures reader performance with and without the device. The guidance is directed at
computer-assisted detection devices, but its reasoning applies to other assistive software. A team
that designs a reader study without reading it is likely to repeat a known deficiency.

The FDA pivotal clinical investigations guidance, issued in 2013, describes the design
considerations for studies intended to support a marketing submission: the research question, the
endpoint, the control, the analysis population, and the handling of missing data. It is not a
checklist, and it does not require a pivotal study for every device. It is useful because it
forces the team to state the question before the design. The same discipline applies to a
510(k)-scale study.

The EU clinical evaluation requirement is broader than a single study. MDR Annex XIV requires a
clinical evaluation plan, a clinical evaluation report, and post-market clinical follow-up. The
clinical evaluation must consider the intended purpose, the population, the claims, the state of
the art, and the available evidence, including equivalence where claimed. The report must be
updated over the lifecycle. A package that treats the clinical evaluation as a one-time document
will fail a notified body audit.

The quality system standards matter because they determine whether the evidence is credible.
ISO 14971:2019 defines the risk management process that links hazards to controls and evidence.
IEC 62304:2006+A1:2015 defines the software lifecycle that produces the verification records.
IEC 62366-1:2015+A1:2020 defines the usability process that produces the human-factors evidence.
The FDA Quality Management System Regulation (QMSR), effective 2 February 2026, incorporates
ISO 13485:2016 and changes the regulatory text that manufacturers have used for decades. The
practical effect is that the quality system and the evidence package must be designed together;
a study that is not conducted under the quality system is difficult to defend.

Two data sources deserve mention. The first is the device's own premarket data, which must be
traceable to the locked version and the validated protocol. The second is real-world data, which
can support post-market performance but rarely substitutes for premarket evidence because of
missing reference standards and selection effects. A registry can show that a device is used, not
that its output is correct. The post-market plan should say which questions real-world data can
answer and which require a formal study.

The most common regulatory misunderstanding in this area is that a pathway determines the
evidence. A 510(k) does not mean "no clinical evidence," and a De Novo does not mean "any
evidence is enough." The pathway sets a legal standard; the claim sets the evidence question. A
narrow claim with a strong predicate may need little clinical data. A broad claim with the same
predicate may need a prospective study. The team that reads the pathway and stops there will
misplan the package.

## 9. Industry practice and case evidence

Public evidence about submissions is uneven. Regulators publish summaries and review documents,
but the underlying protocols and datasets are usually not public. Companies publish case studies
that emphasize success. The cases below separate documented facts from inference, and the
inferences are labeled.

**IDx-DR.** The FDA granted De Novo authorization DEN180001 in 2018 for an autonomous
artificial-intelligence system that analyzes retinal images to detect diabetic retinopathy. The
public review document describes a prospective, multi-site study in primary care with a masked
reference standard and a narrow intended use limited to adults with diabetes and no known
retinopathy. The documented lesson is that an autonomous claim can be authorized at Class II when
the population, workflow, and follow-up pathway are tightly defined. The inference, not stated by
the agency, is that the narrow claim did as much work as the algorithm: the device was authorized
for a specific screening pathway, not for general retinal diagnosis.

**Apple ECG.** The FDA granted De Novo authorization DEN180044 in 2018 for an electrocardiogram
(ECG) app that records a lead I ECG and notifies users of an irregular rhythm. The labeling
describes the app as a screening tool, not a diagnostic replacement, and specifies the conditions
of use. The
documented lesson is that a consumer-facing product is regulated when it makes a medical claim,
and that labeling can carry much of the risk control. The inference is that the consumer brand
did not lower the evidence standard; the narrow claim and the explicit limitations did the work.

**Epic Sepsis Model.** Wong et al. published an external validation in JAMA Internal Medicine in
2021 (DOI 10.1001/jamainternmed.2021.2626). The study evaluated a widely deployed sepsis
prediction model in a health system that had not developed it and reported discrimination
substantially below the vendor's reported range, with a large alert burden at the deployed
threshold. The documented lesson is that commercial availability and local adoption are not
evidence of transportable performance. The inference is that health systems need local validation
and monitoring even when a product is widely used. The paper is a performance finding, not a
regulatory determination.

**Viz.ai large-vessel occlusion triage.** Viz.ai received FDA clearance for software that analyzes
computed tomography angiography images and alerts care teams to a suspected large-vessel
occlusion. Public materials describe a workflow product that combines detection with care-team
coordination. The documented lesson is that the clinical benefit may come from time to treatment
and coordination rather than from stand-alone accuracy. The inference is that a workflow claim
requires workflow evidence, and that a detector with modest stand-alone performance can support a
claim if the workflow study shows a meaningful change in action. Current clearance numbers and
indications should be checked in the FDA database rather than taken from a secondary summary.

**Industry practice: build the evidence map at design freeze.** Mature manufacturers write the
assertion list and the evidence map before the design is frozen. The map names the study for each
assertion, the endpoint, the acceptance criterion, and the owner. This practice is not visible in
public submissions, but it is visible in the consistency of the summaries: the claims, the
studies, and the labels line up. The inference is that the map is a leading indicator of review
smoothness.

**Industry practice: separate the primary endpoint from the supporting analyses.** Companies that
survive review with broad claims usually have one pre-specified primary endpoint, a small number
of pre-specified secondary endpoints, and a clear rule for what happens if the primary fails.
Companies that struggle often have many endpoints and a narrative that shifts to whichever result
is favorable. A reviewer reads the protocol's endpoint hierarchy as a statement of what the
company believed before the data arrived.

**Industry practice: use a red team.** Some organizations run an internal mock review with a
regulatory professional who was not on the project. The red team writes deficiencies in the
regulator's format and ranks them by severity. The practice is more common in large companies
than in startups, and it is rarely described in public sources. The inference from deficiency
letters is that the failures it catches, especially claim-evidence mismatch, are common enough to
justify the cost.

**What the cases do not prove.** None of these cases proves that a particular pathway is
available for a new product. IDx-DR does not prove that every autonomous screening device can use
De Novo. Apple ECG does not prove that every wearable is a device. Viz.ai does not prove that
every triage product can claim a time benefit. Epic does not prove that every sepsis model fails
externally. Each case is a data point about a specific claim, population, and evidence package.
The transferable lesson is the method: define the claim, decompose it into assertions, identify
the evidence gap, and test the claim in the intended population.

## 10. Failure modes and red flags

Evidence-package failures rarely announce themselves. They appear as a confident summary, a
validation report with impressive numbers, and a label that says something slightly different. The
table lists the failures that most often survive until a reviewer finds them.

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Claim-evidence mismatch | The label says "prioritize" but the study measures AUC | The study answers an easier question than the claim | Deficiency or rejection; new study required | Write assertions before protocols; trace every label clause |
| Enriched test set presented as clinical | PPV is reported without prevalence | Case mix differs from the intended setting | Label overstates real-world precision | Report prevalence-adjusted PPV and NPV; label the enrichment |
| Reference-standard circularity | Negatives are defined by the same reports used for training | The model learns reporting style, not disease | Inflated specificity; reviewer rejects the truth standard | Independent adjudication; CT or follow-up for a sample of negatives |
| Single-site external validation | One "external" site shares patients or equipment with development | The site is not independent in practice | Generalizability claim unsupported | Verify site independence; add a second site |
| Underpowered subgroup | The subgroup CI is wide and crosses the acceptance threshold | The study was powered for the overall result only | Labeled subgroup claim unsupported | Pre-specify and power subgroups; narrow the label |
| Post-hoc endpoint switch | The summary leads with a secondary endpoint | The primary endpoint failed | Reviewer treats the result as exploratory | Freeze the analysis plan; report the primary result first |
| Usability study tests satisfaction | Report contains liking scores but no use-error analysis | The study does not test hazard-related tasks | Use-related hazards remain uncontrolled | Summative validation with representative users and tasks |
| Security scope excludes the interface | Threat model covers the cloud but not the hospital archive | The system boundary is drawn too narrowly | Missing security evidence; field vulnerability | Draw data flows end to end; include update and archive paths |
| Software version drift | Validation report names a version different from the submitted one | The model or threshold changed after validation | The evidence does not apply to the device | Configuration control; impact assessment for every change |
| Risk file disconnected from evidence | Hazards have controls but no verification references | Risk management is a parallel document | Residual risk unsupported; audit finding | Link each control to a test or study |
| Label overclaim | Marketing says "rules out" while the study measured sensitivity | Promotional language becomes evidence of intent | Warning letter, recall, liability | Review all customer-facing text against the label |
| Post-market plan without triggers | Plan says "monitor performance" | No metric, data source, or action threshold | Drift is detected late or not at all | Define metrics, cadence, thresholds, and actions |
| PCCP used as open-ended learning | Plan says the model "improves continuously" | The changes are not bounded or validated | Unauthorized modification | Bound the modifications and the validation protocol |
| Submission assembled by department | Cross-references do not resolve; summary repeats reports | No evidence map; the argument is implicit | Reviewer cannot follow the claim; deficiencies multiply | Assemble by assertion; write the summary last |

A red flag is not proof of a violation. It is a signal that the package is missing a fact or a
link. The right response is to test the flag, not to explain it away. A team that can show why a
red flag does not apply is in a stronger position than a team that never looked.

## 11. Assumptions and limits

Every evidence package rests on assumptions. The package should state them, explain why they are
needed, describe what would break them, and name the test that would detect the break.

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The claim is stable across the submission | Every study is designed against a claim version | A late wording change invalidates the evidence map | Version-control the claim and review every change |
| The test set is independent of development | Performance estimates depend on independence | Site, patient, or report overlap with training data | Audit site and patient identifiers; use a held-out site |
| The reference standard measures the target condition | The endpoint is only as good as the truth standard | Reports, adjudication, or follow-up are incomplete or biased | Verify a sample with an independent modality; report verification rate |
| The intended prevalence is known | PPV and NPV depend on prevalence | Prevalence differs by setting, season, or case mix | Use local data; report a prevalence sensitivity analysis |
| Readers behave as they do in practice | Reader studies simulate the clinical task | Artificial worklists, unlimited time, or recall between reads | Match the workflow; randomize order; wash out between cases |
| Users interpret the output as labeled | The claim depends on human behavior | Users over-trust, ignore, or misread the output | Summative usability with negative and uncertain cases |
| The software version is frozen | Evidence is tied to a version | Model, threshold, or dependency changes after validation | Configuration control and change impact assessment |
| The risk file covers the evidence | Residual risk depends on the controls being verified | Hazards are missing or controls are unverified | Cross-check hazards against studies and tests |
| The post-market plan can detect drift | Premarket evidence is a sample | No data source, metric, or trigger exists | Audit the plan against the risk file and the label |
| The regulatory standard is current | Guidance and standards change | A new guidance, standard, or regulation applies | Record versions and access dates; recheck before filing |

The limits of this topic are also explicit. It does not provide a legal opinion or a submission
template. It does not cover every pathway or every device type. It does not resolve whether a
specific study will satisfy a specific reviewer. It does not replace a statistician for study
design or a regulatory professional for strategy. It gives a structure for planning, critiquing,
and defending the package, and it names the places where the structure is most likely to fail.

A final limit concerns the cases. Public summaries describe what was submitted, not what was
rejected. The failures are inferable from deficiency letters, guidance, and post-market findings,
but they are not a random sample. The method in this topic is therefore a way to reduce
predictable failures, not a guarantee of a favorable decision.

## 12. Exercises

The exercises move from recall to application to critique to design. Each has a deliverable,
acceptance criteria, and a time estimate. Use the current guidance and the current database
records, not this file alone.

### Exercise 1 — Assertion extraction (recall and application, 30 minutes)

Take a public intended-use statement for an AI-enabled device and decompose it into at least six
numbered assertions. Classify each as analytical, clinical, human-factors, or system. Mark each
as necessary or supporting. Deliverable: a one-page table with the claim quoted at the top.
Acceptance criteria: every clause of the claim maps to at least one assertion; negative clauses
are treated as assertions; no assertion is a restatement of the claim.

### Exercise 2 — Evidence map (application, 45 minutes)

For the assertions from Exercise 1, build an evidence map with columns for assertion, question,
endpoint, study design, acceptance criterion, and label clause. Mark each row as supported,
partial, absent, or contradictory using the public summary. Deliverable: the matrix plus a short
note on the three weakest rows. Acceptance criteria: every necessary assertion has a row; the
status is justified with a source; absent rows are not hidden.

### Exercise 3 — Analytical validation protocol (design, 60 minutes)

Write a two-page analytical validation protocol for a hypothetical imaging classifier. Include the
eligible population, test-set construction, locking procedure, reference standard, primary and
secondary metrics, subgroup plan with minimum sample sizes, missing-data handling, and acceptance
criteria. Deliverable: the protocol with a one-paragraph justification of the test-set size.
Acceptance criteria: the test set is independent of development; the reference standard is
operationalized; the subgroup plan is pre-specified; the acceptance criteria are numeric.

### Exercise 4 — Gap score and sample size (calculation, 45 minutes)

Using the script in Section 7, score a project you know or a hypothetical project. Change three
maturity values and record how the score and the ranked contributions change. Then compute the
paired reader-study case count for a device assumed to improve sensitivity from 0.75 to 0.85 with
discordance proportions of 0.14 and 0.04, and inflate for 10 readers at an intraclass correlation
of 0.15. Deliverable: a table of the two scores and the sample-size arithmetic. Acceptance
criteria: the weights sum to 1.0; the formula and inputs are shown; the output is reproduced by
running the code.

### Exercise 5 — Pathway comparison (application, 40 minutes)

Take one product and describe how the evidence package would differ across a 510(k), a De Novo
request, a PMA, and an EU technical file. Focus on the clinical evidence, the summary document,
and the post-market plan. Deliverable: a four-column comparison table. Acceptance criteria: each
column names the legal standard; the differences are substantive, not formatting; the table does
not claim that a pathway removes the need for clinical evidence.

### Exercise 6 — Study design choice (critique, 40 minutes)

For each of the following claims, choose a study design and justify it: (a) the device detects
pneumothorax on chest radiographs; (b) the device helps radiologists detect pneumothorax faster;
(c) the device safely removes negative studies from the review queue; (d) the device predicts
which patients will deteriorate within 24 hours. Deliverable: four short design memos, each with
an endpoint and a main threat. Acceptance criteria: the design matches the causal path in the
claim; the endpoint is measurable; the main threat is specific.

### Exercise 7 — Red-team review (critique, 50 minutes)

Exchange an evidence map with a peer. Write six deficiencies in the format a reviewer would use,
ranked by severity. For each, state the evidence needed to close it and whether the fix is a
label change, an analysis, or a new study. Deliverable: the deficiency list and a one-paragraph
filing recommendation. Acceptance criteria: at least one deficiency concerns the reference
standard; at least one concerns the label; the recommendation distinguishes a blocker from a
fixable gap.

### Exercise 8 — Label trace (design, 35 minutes)

Take a draft label for a hypothetical device and trace every claim-bearing sentence to an
assertion and a study result. Mark sentences that have no trace and rewrite them so that they are
supported. Deliverable: a three-column trace table and a revised label section. Acceptance
criteria: every retained sentence has a trace; removed sentences are listed with the reason; the
revised label states the prevalence-dependent performance and at least two limitations.

## 13. Self-check answers

**Question 1. What is an assertion, and why is it the unit of evidence planning?** An assertion is
a testable proposition derived from the claim, such as "sensitivity is at least 0.85 in adults
with suspected pneumothorax." It is the unit of planning because a claim is too broad to test
directly and a study is too specific to cover the whole claim. The assertion list connects the
two and makes gaps visible.

**Question 2. Why is a negative clause in the intended use an assertion?** Because it limits the
claim and therefore limits the evidence. "Not intended for patients with a known pneumothorax"
tells the reviewer that the device was not validated in that population. If the clause is not
supported by the study population, the label and the evidence disagree. A negative clause is a
statement about the boundaries of the evidence, and it needs a trace just like a positive claim.

**Question 3. Which evidence modules does a prioritization claim require?** At minimum,
analytical validation, a clinical or workflow study that measures the prioritization effect,
usability evidence that users interpret the flag and the absence of a flag correctly,
cybersecurity evidence, software verification, a risk file, labeling that states the
prevalence-dependent performance, and a post-market plan. The exact depth depends on the pathway
and the risk, but a prioritization claim is a workflow claim and needs workflow evidence.

**Question 4. What must an analytical validation protocol specify before data collection?** The
eligible population and input conditions, the test-set construction and independence, the locked
algorithm and threshold, the reference standard, the primary and secondary metrics with confidence
intervals, the subgroup plan with minimum sample sizes, the missing-data handling, and the numeric
acceptance criteria. If any of these is chosen after the data are seen, the study is exploratory.

**Question 5. When is a standalone study enough, and when is a reader study needed?** A standalone
study is enough when the claim is about the device's output against truth, such as "detects the
finding with sensitivity X." A reader study is needed when the claim is about the effect of the
output on a clinician's decision, such as "improves reader sensitivity" or "reduces time to
review." The test is the causal path in the claim, not the prestige of the design.

**Question 6. What does external validation add that an internal test set does not?** An internal
test set, even a held-out one, usually shares sites, equipment, and patient mix with development.
External validation tests transportability: whether the locked device performs in a population and
setting it was not built on. It is the closest premarket evidence for generalizability. One
external site is a start, not a proof, because the site may share hidden similarities with the
development data.

**Question 7. How should the weights in an evidence-gap score be set?** By the regulatory and
clinical leads, before the evidence is reviewed, using the claim and the risk file. The weights
should reflect how much each assertion matters to the claim and how severe a failure would be.
They should sum to 1.0 and be documented. A score produced by the team that owns the evidence,
after the evidence is known, is a rationalization, not an assessment.

**Question 8. Why does reader count change the required case count?** Because cases are not
independent across readers. Readers differ in skill and in how they use the device, and the same
case is read by every reader. The correlation among readers, summarized by the intraclass
correlation, reduces the effective information in each additional reading. A larger number of
readers, or a higher correlation, requires more cases to achieve the same precision. The
design-effect heuristic in Section 7 is a planning approximation; a full multi-reader multi-case
model estimates the variance components directly.

**Question 9. What are the three most common claim-evidence mismatches?** First, an accuracy claim
supported by an accuracy study when the label makes a workflow or outcome claim. Second, a
population claim supported by a study in a narrower or different population. Third, a performance
claim supported by an enriched test set whose predictive values do not hold at the intended
prevalence. Each is a mismatch between the sentence the reviewer reads and the study the reviewer
can inspect.

**Question 10. What does a reviewer challenge produce, and how is it used?** It produces a ranked
list of deficiencies, each with the evidence needed to close it and an estimate of the fix cost.
It is used to decide whether to file, remediate, or narrow the claim. A deficiency that requires a
new study is a filing blocker if it affects a necessary assertion. A deficiency that can be closed
by labeling is cheaper but may narrow the market. The challenge is a decision tool, not a
formality.

## 14. Interview questions

**1. How do you know when an evidence package is sufficient?** A strong answer says sufficiency
is relative to the claim, the population, the risk, and the pathway; it describes the assertion
list, the evidence map, and the acceptance criteria; and it notes that precision and independence
matter as much as the point estimate. A weak answer says "when all the documents are complete" or
"when the model meets its target metric."

**2. What is the difference between analytical and clinical validation?** A strong answer says
analytical validation establishes that the measurement is reliable and accurate under stated
conditions, while clinical validation establishes that the output is associated with the clinical
condition and supports the claimed use in the intended population. A weak answer treats clinical
validation as a larger analytical study.

**3. How would you design evidence for a triage claim?** A strong answer identifies the workflow
causal path, chooses a reader or simulated-workflow study with a time-to-review endpoint, addresses
prevalence and false-flag burden, and plans usability for the negative result. A weak answer
proposes a standalone AUC study and calls it clinical evidence.

**4. When would you use a reader study instead of a standalone study?** A strong answer ties the
choice to the claim: reader studies measure the clinician-plus-device effect, standalone studies
measure the device against truth. It also notes the design requirements: representative readers,
representative cases, blinded reading, and a pre-specified analysis. A weak answer says reader
studies are "more rigorous" without explaining what they measure.

**5. What makes a reference standard acceptable?** A strong answer says it must be independent of
the device and of the data used to develop it, it must be defined and adjudicated, and the
verification rate for negatives must be reported. A weak answer says "the radiologist's report" or
"the clinical diagnosis" without examining dependence.

**6. How do you handle a subgroup that performs worse than the overall population?** A strong
answer checks whether the subgroup was pre-specified, whether it is in the labeled population,
whether the confidence interval meets the acceptance criterion, and whether the label or the claim
must be narrowed. A weak answer calls the finding exploratory after the protocol named it.

**7. What is the role of prevalence in an evidence package?** A strong answer explains that
sensitivity and specificity are prevalence-independent but predictive values are not, and that an
enriched test set can overstate PPV. It describes reporting performance at the intended prevalence
and modeling the false-flag burden. A weak answer reports PPV from the study set without noting
the case mix.

**8. What belongs in a post-market plan?** A strong answer names the open performance questions,
the data sources, the metrics, the review cadence, the action thresholds, and the change-control
process. A weak answer says "monitor complaints and report adverse events," which is the legal
minimum but not a performance plan.

**9. How do you keep a submission consistent when the model changes?** A strong answer describes
configuration control, a change impact assessment, re-validation of the affected assertions, and a
PCCP if changes are planned. A weak answer says the model is frozen, without explaining how the
freeze is enforced.

**10. What would you do if the primary endpoint fails but a secondary endpoint is strong?** A
strong answer reports the primary result first, treats the secondary as supportive or exploratory
unless it was pre-specified and powered, and considers whether the claim can be narrowed to what
the data support. A weak answer leads with the favorable secondary endpoint and reframes the
claim.

**11. How do you review a label for claim-evidence alignment?** A strong answer traces every
claim-bearing sentence to an assertion and a result, removes untraceable sentences, adds the
limitations the evidence revealed, and checks promotional language against the label. A weak
answer reads the label for tone and grammar.

**12. What is the most common reason a package fails review?** A strong answer names
claim-evidence mismatch and the related failures: population mismatch, reference-standard
dependence, and an endpoint that does not measure the claim. A weak answer blames documentation
formatting or reviewer subjectivity.

## 15. Key terms

| Term | One-line meaning |
|---|---|
| Analytical validation | Evidence that the measurement is accurate and reliable under stated conditions |
| Assertion | A testable proposition derived from the claim |
| Benefit-risk | The judgment that probable benefits outweigh probable risks for the intended use |
| Claim | The statement of what the device does, for whom, and with what benefit |
| Clinical performance | Correlation of device results with clinical truth in the intended population |
| Clinical validation | Evidence that the device supports the claimed clinical use in the intended population |
| Deficiency | A reviewer finding that information is missing or insufficient |
| Enrichment | Deliberate increase in the proportion of positive cases in a study set |
| Evidence gap | The difference between what an assertion requires and what the evidence shows |
| Evidence map | A matrix linking assertions to questions, endpoints, studies, results, and label clauses |
| External validation | Evaluation of a locked device on data from an independent site or population |
| IMDRF N41:2017 | International guidance defining the three elements of SaMD clinical evaluation |
| Independent test set | Data not used in development, tuning, or threshold selection |
| Intended use | The objective intent of the persons responsible for labeling |
| MRMC study | Multi-reader multi-case study design for reader-performance claims |
| PCCP | Predetermined change control plan for specified future modifications |
| PMCF | Post-market clinical follow-up |
| PMS | Post-market surveillance |
| PPV / NPV | Positive and negative predictive value at a stated prevalence |
| Pre-specification | Fixing the hypothesis, endpoint, and analysis before unblinding |
| QMSR | FDA Quality Management System Regulation, effective 2 February 2026 |
| Reader study | Study of clinician decisions with and without the device output |
| Reference standard | The independent determination of truth used to judge device output |
| Spectrum bias | Bias from a case mix that differs from the intended population |
| Standalone performance | Device performance against truth without a clinician in the loop |
| Surrogate endpoint | An endpoint expected to predict clinical benefit but not itself a measure of it |
| Summative usability | Validation that representative users can use the device safely |
| Traceability | The link from claim to assertion to evidence to label |

## 16. Sources

All URLs were accessed on 25 October 2026. Dates are the publication or revision dates shown by
the source at access. Verify current versions before relying on them.

1. US Food and Drug Administration. "Clinical Performance Assessment: Considerations for
   Computer-Assisted Detection Devices Applied to Radiology Images and Radiology Device Data in
   Premarket Notification (510(k)) Submissions." Final guidance, 28 September 2022. Available
   through the FDA guidance search page.
   https://www.fda.gov/regulatory-information/search-fda-guidance-documents
2. US Food and Drug Administration. "Design Considerations for Pivotal Clinical Investigations for
   Medical Devices." Guidance, 2013.
   https://www.fda.gov/regulatory-information/search-fda-guidance-documents/design-considerations-pivotal-clinical-investigations-medical-devices
3. US Food and Drug Administration. "Clinical Decision Support Software." Final guidance,
   28 September 2022.
   https://www.fda.gov/regulatory-information/search-fda-guidance-documents/clinical-decision-support-software
4. US Food and Drug Administration. "Content of Premarket Submissions for Device Software
   Functions." Guidance.
   https://www.fda.gov/regulatory-information/search-fda-guidance-documents/content-premarket-submissions-device-software-functions
5. US Food and Drug Administration. "Cybersecurity in Medical Devices: Quality System
   Considerations and Content of Premarket Submissions." Guidance, 2023.
   https://www.fda.gov/regulatory-information/search-fda-guidance-documents/cybersecurity-medical-devices-quality-system-considerations-and-content-premarket-submissions
6. US Food and Drug Administration. "Quality Management System Regulation (QMSR)."
   https://www.fda.gov/medical-devices/postmarket-requirements-devices/quality-management-system-regulation-qmsr
7. US Food and Drug Administration. "De Novo Classification Request: DEN180001."
   https://www.accessdata.fda.gov/cdrh_docs/reviews/DEN180001.pdf
8. US Food and Drug Administration. "De Novo Classification Request: DEN180044."
   https://www.accessdata.fda.gov/cdrh_docs/reviews/DEN180044.pdf
9. International Medical Device Regulators Forum. "Software as a Medical Device (SaMD): Clinical
   Evaluation." IMDRF/SaMD WG/N41FINAL:2017, 21 September 2017.
   https://www.imdrf.org/documents/software-medical-device-clinical-evaluation
10. International Medical Device Regulators Forum. "Software as a Medical Device: Possible
    Framework for Risk Categorization and Corresponding Considerations." IMDRF/SaMD WG/N12FINAL:2014.
    https://www.imdrf.org/documents/software-medical-device-possible-framework-risk-categorization-and-corresponding-considerations
11. European Parliament and Council. Regulation (EU) 2017/745 on medical devices (MDR), Annex XIV.
    https://eur-lex.europa.eu/eli/reg/2017/745/oj
12. Medical Device Coordination Group. MDCG 2020-1, "Guidance on Clinical Evaluation (MDR) /
    Performance Evaluation (IVDR) of Medical Device Software," March 2020. Available through the
    EU MDCG guidance page.
    https://health.ec.europa.eu/medical-devices-sector/new-regulations/guidance-mdcg-endorsed-documents-and-other-guidance_en
13. International Organization for Standardization. ISO 14971:2019, Medical devices — Application
    of risk management to medical devices. https://www.iso.org/standard/72704.html
14. International Electrotechnical Commission. IEC 62304:2006+A1:2015, Medical device software —
    Software life cycle processes. https://webstore.iec.ch/en/publication/22794
15. International Electrotechnical Commission. IEC 62366-1:2015+A1:2020, Medical devices —
    Application of usability engineering to medical devices. https://webstore.iec.ch/en/publication/26584
16. International Organization for Standardization. ISO 14155:2020, Clinical investigation of
    medical devices for human subjects — Good clinical practice. https://www.iso.org/
17. International Organization for Standardization. ISO 13485:2016, Medical devices — Quality
    management systems — Requirements for regulatory purposes. https://www.iso.org/standard/59752.html
18. Wong, A., Otles, E., Donnelly, J. P., et al. "External Validation of a Widely Implemented
    Proprietary Sepsis Prediction Model in Hospitalized Patients." JAMA Internal Medicine, 2021.
    https://doi.org/10.1001/jamainternmed.2021.2626
19. US Food and Drug Administration. "Artificial Intelligence-Enabled Medical Devices."
    https://www.fda.gov/medical-devices/digital-health-center-excellence/artificial-intelligence-enabled-medical-devices
20. US Food and Drug Administration. "Good Machine Learning Practice for Medical Device
    Development: Guiding Principles."
    https://www.fda.gov/medical-devices/artificial-intelligence-enabled-medical-devices/good-machine-learning-practice-medical-device-development-guiding-principles

## 17. Links

- Module index: [`_index.md`](_index.md)
- Next topic: [`10_post-market-surveillance.md`](10_post-market-surveillance.md)
- Related topic: [`03_fda-pathways.md`](03_fda-pathways.md)
- Related topic: [`04_gmlp-and-pccp.md`](04_gmlp-and-pccp.md)
- Related topic: [`06_iso-14971-risk-management.md`](06_iso-14971-risk-management.md)
- Related topic: [`07_iec-62304-software-lifecycle.md`](07_iec-62304-software-lifecycle.md)
- Survey file: [`../../course/14_medical_devices_samd_regulation.md`](../../course/14_medical_devices_samd_regulation.md)
- Labs: [`../../labs/`](../../labs/)
- Lab 14.3 risk management file: [`../../labs/`](../../labs/)
- Lab 14.4 requirements and traceability: [`../../labs/`](../../labs/)

The module index is the authoritative map. If a sibling file is not yet present, the index marks
it as planned; do not infer content from the link text.
