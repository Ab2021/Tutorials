# GMLP and PCCP

> Module 14 — Medical Devices, SaMD, and Regulation
> Topic 04 of 10
> Estimated study time: 3.5 hours
> Prerequisites: 01, 02, 03
> Version 2026.3

## 1. Why this topic exists

Traditional device regulation assumes a frozen product. The manufacturer validates a version, submits evidence, receives
clearance or certification, and then controls changes through a quality system. A material change triggers a new
submission or a significant regulatory decision. That model works for a scalpel or a pump whose design changes every few
years. It fits badly with machine learning, where performance can degrade as populations shift, data sources change, and
clinical practice evolves.

The tension is not that regulators oppose improvement. The tension is that the evidence submitted at launch describes a
specific version, a specific population, and a specific performance profile. If the manufacturer can change the model at
will, the evidence no longer describes the device on the market. If the manufacturer cannot change the model at all, the
device may become stale and unsafe. A sepsis model trained before a change in coding practice may slowly lose
calibration. An imaging model trained on one scanner fleet may fail when a hospital replaces its equipment. A locked
model is safe only if the world stays still.

Good Machine Learning Practice (GMLP) and Predetermined Change Control Plans (PCCPs) are two responses to that tension.
GMLP is a set of lifecycle practices for developing and monitoring machine learning devices. It is guidance, not a
marketing authorization. A PCCP is a pre-authorized description of specific future changes and the methods used to
validate them. It is reviewed as part of a marketing submission and, if authorized, allows the manufacturer to implement
the covered changes without a new submission. The two are related but not interchangeable. GMLP describes how to work. A
PCCP describes what may change and how the change will be justified.

The business problem is acute. A company that plans a quarterly model update but has no PCCP must decide between
freezing the model and filing a new submission every quarter. Both options are expensive. Freezing risks drift and lost
customers. Repeated submissions consume regulatory capacity and delay improvements. A company that writes a PCCP after
launch discovers that the authorized modifications are too narrow, the acceptance criteria are unmeasurable, or the
monitoring data do not exist. The PCCP must be designed with the product, not bolted on before a filing deadline.

The clinical problem is equally real. A model update can improve average performance while harming a subgroup. A
threshold change can reduce false alarms while increasing missed cases. A new input can improve accuracy in one setting
but change the device's intended use. A PCCP that authorizes a performance change without a subgroup analysis can
convert a safe device into an inequitable one. The plan is a risk-control document as much as a regulatory convenience.

The legal problem is jurisdictional. In the United States, section 515C of the Federal Food, Drug, and Cosmetic Act
(FD&C Act), added by the Food and Drug Omnibus Reform Act of 2022 (FDORA), provides the statutory basis for PCCPs. The
FDA's final guidance for artificial intelligence (AI)-enabled device software functions was originally issued on 4
December 2024 and reissued on 18 August 2025. In the European Union, the Medical Device Regulation (MDR) and In Vitro
Diagnostic Regulation (IVDR) do not have a directly equivalent PCCP regime; changes are handled through the quality
system and, where required, a new conformity assessment. The EU Artificial Intelligence Act adds another layer for AI
systems. A single global PCCP is rarely the right artifact.

This topic gives you the vocabulary and procedure to decide whether a PCCP is appropriate, how to bound it, how to write
the modification protocol, how to assess impact, and how to monitor the authorized changes. It also gives you the
failure modes that turn a PCCP into a liability. The treatment is educational and does not replace regulatory counsel, a
pre-submission, or the current guidance.

## 2. Learning outcomes

By the end of this topic, you can:

1. Explain the difference between GMLP, a PCCP, and ordinary quality-system change control.
2. Decide whether a planned modification is eligible for a PCCP or requires a new marketing submission.
3. Write a bounded description of modifications with measurable specifications and explicit exclusions.
4. Design a modification protocol covering data management, retraining, performance evaluation, and update procedures.
5. Build a traceability table that links each modification to its validation method and acceptance criterion.
6. Conduct an individual and cumulative impact assessment, including subgroup and bias effects.
7. Integrate a PCCP with the quality management system, risk management file, and post-market surveillance plan.
8. Critique a PCCP that authorizes open-ended learning or unmeasurable changes.

The outcomes map to the exercises and self-check items as follows.

| Outcome | Exercise | Self-check |
|---|---|---|
| 1 | Exercise 1 | Questions 1, 2 |
| 2 | Exercise 2, Exercise 3 | Question 3 |
| 3 | Exercise 3 | Question 4 |
| 4 | Exercise 4 | Question 5 |
| 5 | Exercise 5 | Question 6 |
| 6 | Exercise 6 | Question 7 |
| 7 | Exercise 7 | Question 8 |
| 8 | Exercise 8 | Questions 9, 10 |

## 3. The core mental model

A PCCP is a contract with the regulator about a bounded set of future changes. The contract has three parts: what may
change, how the change will be evaluated, and why the change is safe. The contract is embedded in a quality system that
executes it and a monitoring system that detects when reality diverges from the assumptions.

```text
Intended use and device description
              |
              v
[1] Is the change planned, bounded, and verifiable?
              | no                                  | yes
              v                                     v
     New submission or                    [2] Describe modifications
     no change                             - performance
              |                             - inputs
              |                             - use/population
              |                                     |
              |                                     v
              |                          [3] Modification protocol
              |                           - data management
              |                           - retraining
              |                           - performance evaluation
              |                           - update/rollback
              |                                     |
              |                                     v
              |                          [4] Impact assessment
              |                           - individual
              |                           - cumulative
              |                           - bias/subgroups
              |                                     |
              |                                     v
              |                          [5] Marketing submission
              |                           - 510(k)/De Novo/PMA
              |                           - PCCP reviewed
              |                                     |
              |                                     v
              |                          [6] Authorized PCCP
              |                           - implement within scope
              |                           - document each change
              |                           - monitor
              |                                     |
              |                                     v
              |                          [7] Post-market evidence
              |                           - drift, complaints, performance
              |                           - deviation -> new submission
              v
        Stop or redesign
```

The first distinction is between a change that is within the authorized device and a change that creates a new device. A
PCCP cannot expand the intended use. It cannot change the device from a clinician-facing triage tool to an autonomous
diagnostic. It cannot add a new patient population that the submission did not study unless the PCCP explicitly covers
that population and the evidence supports it. The FDA guidance is clear that the modifications must be within the
device's intended use and, for a 510(k), must maintain substantial equivalence. A PCCP is not a license for arbitrary
continuous learning.

The second distinction is between a locked model and a continuous-learning model. A locked model has fixed weights after
release; changes are periodic retraining events. A continuous-learning model updates automatically as it receives data.
Both can be covered by a PCCP, but the evidence and monitoring burden differ. A locked model with an annual retraining
cycle is easier to bound. A continuous model requires automated acceptance criteria, rollback, audit trails, and a
mechanism to detect when the updated model leaves the validated envelope. The FDA's 2025 guidance explicitly covers
automatic, manual, and hybrid update modes, and it asks the sponsor to state the expected frequency, from annual or less
to continuous.

The third distinction is between a performance change and a use change. A performance change improves an existing output
without changing what the device does. A use change alters the population, the user, the setting, the input, or the
clinical action. Performance changes are more likely to fit a PCCP. Use changes are more likely to require a new
submission, because they alter the claim and the evidence base. A new input modality, such as adding a second imaging
view, is usually a use change even if the algorithm architecture is unchanged.

The fourth distinction is between the PCCP and the quality system. The PCCP says what changes are allowed and how they
will be validated. The quality system says who approves the change, how records are kept, how complaints are handled,
and how corrective and preventive action works. A PCCP that is not connected to the quality system is a document, not a
control. An authorized change that is implemented without a design change record, a risk assessment, and a validation
report is a quality-system failure even if it is within the PCCP scope.

The fifth distinction is between pre-market evidence and post-market evidence. A PCCP can rely on a combination of both,
but the pre-market submission must specify the evidence that will be generated for each planned change. "We will monitor
performance" is not a modification protocol. The protocol must state what data will be collected, how the model will be
retrained, how performance will be evaluated, what acceptance criteria will apply, and what happens if the criteria are
not met. The monitoring plan then checks whether the assumptions behind those criteria hold in the real world.

A useful shorthand is: GMLP is the discipline, PCCP is the permission, QMS is the plumbing, and post-market surveillance
is the proof. If any one is missing, the others fail. A team with strong GMLP but no PCCP will either freeze the model
or file repeatedly. A team with a PCCP but weak GMLP will authorize changes it cannot validate. A team with both but no
QMS will implement changes it cannot trace. A team with all three but no monitoring will learn about drift from
complaints.

## 4. Definitions and vocabulary

The table defines the terms used in this topic and later in the module. The "common misuse" column reflects errors that
appear in real PCCP drafts and regulatory strategies.

| Term | Definition | Common misuse | Source |
|---|---|---|---|
| Good Machine Learning Practice (GMLP) | A set of lifecycle principles for developing and monitoring machine learning-enabled medical devices | Treating GMLP as a prescriptive standard or a marketing authorization | FDA/Health Canada/MHRA, 27 October 2021 |
| Predetermined Change Control Plan (PCCP) | A plan describing specific future modifications to a device and the methods for validating them, submitted for review with a marketing submission | Treating it as permission for any model update | FD&C Act §515C; FDA guidance, 18 August 2025 |
| Modification | A change to the device's software, data, inputs, performance, or use that is described in the PCCP | Calling every bug fix a modification | FDA PCCP guidance, 2025 |
| Description of modifications | The part of a PCCP that specifies what may change, with measurable characteristics and performance specifications | Writing a vague intention such as "improve accuracy" | FDA PCCP guidance, 2025 |
| Modification protocol | The part of a PCCP that specifies data management, retraining, performance evaluation, and update procedures | Describing only the algorithm change and not the validation | FDA PCCP guidance, 2025 |
| Impact assessment | The part of a PCCP that analyzes the benefits and risks of each modification and their cumulative effect | Replacing it with a generic risk score | FDA PCCP guidance, 2025 |
| Locked model | A model whose parameters are fixed at release and changed only through controlled updates | Assuming locked means never updated | FDA PCCP guidance, 2025 |
| Continuous learning | A model that updates automatically or semi-automatically from new data during use | Assuming continuous learning is always prohibited | FDA PCCP guidance, 2025 |
| Retraining | Re-estimating model parameters using new or additional data | Confusing retraining with recalibration or threshold adjustment | FDA PCCP guidance, 2025 |
| Recalibration | Adjusting the mapping from model output to probability without changing the model architecture | Treating recalibration as a trivial change | FDA AI lifecycle draft, 7 January 2025 |
| Threshold change | Changing the operating point that converts a score into a category or action | Assuming a threshold change cannot affect safety | FDA PCCP guidance, 2025 |
| Data drift | A change in the distribution of inputs or outcomes after deployment | Treating drift as proof of model failure | FDA AI lifecycle draft, 2025 |
| Concept drift | A change in the relationship between inputs and the target outcome | Confusing it with data drift | FDA AI lifecycle draft, 2025 |
| Performance monitoring | Ongoing measurement of deployed model performance and safety | Assuming monitoring replaces periodic validation | FDA AI lifecycle draft, 2025 |
| Change control | The quality-system process for proposing, reviewing, approving, and documenting changes | Treating the PCCP as a substitute for change control | ISO 13485:2016 |
| Design change | A change to the design of a device, including software, that is controlled under the quality system | Implementing a PCCP-covered change without a design record | 21 CFR 820 / ISO 13485 |
| Total product life cycle (TPLC) | The span from concept through development, launch, maintenance, and retirement | Treating the PCCP as a one-time submission artifact | FDA TPLC model |
| AI-enabled device software function (AI-DSF) | A software function that uses AI, including machine learning, to perform a medical purpose | Calling any statistical model AI | FDA PCCP guidance, 2025 |
| Presubmission (Q-Sub) | A voluntary FDA process for feedback before a marketing submission | Expecting a binding decision on PCCP scope | FDA Q-Submission program |
| Substantial equivalence | The 510(k) standard requiring same intended use and similar technology, or different technology with comparable safety and effectiveness | Assuming a PCCP can preserve equivalence through any change | FD&C Act §513(i) |
| Intended use | The objective intent of the persons responsible for labeling | Assuming the PCCP can expand it | 21 CFR 801.4 |
| Authorized change | A change implemented within the scope of an authorized PCCP | Treating an unauthorized change as authorized because it is small | FDA PCCP guidance, 2025 |
| Deviation | A change that falls outside the authorized PCCP | Implementing it without a new submission | FDA PCCP guidance, 2025 |
| Quality management system (QMS) | The organizational structure, procedures, processes, and resources used to implement quality management | Treating it as a document repository | ISO 13485:2016 |
| Risk management file | The set of records produced by the ISO 14971 process | Keeping it separate from the PCCP impact assessment | ISO 14971:2019 |
| Software safety class | The IEC 62304 classification based on the possible consequence of software failure | Confusing it with the regulatory device class | IEC 62304:2006+A1:2015 |
| Predetermined change | A change defined in advance with specified methods and acceptance criteria | A change that is merely anticipated but not specified | FDA PCCP guidance, 2025 |
| Marketing submission | A 510(k), De Novo, PMA, or other submission reviewed by FDA | Assuming a PCCP is a standalone submission | FDA premarket programs |
| Post-market surveillance (PMS) | The systematic collection and analysis of data after launch | Treating it as complaint handling only | MDR Article 83; FDA QMSR |
| Post-market clinical follow-up (PMCF) | Clinical data collection after launch to confirm safety and performance | Assuming it is optional for all devices | MDR Annex XIV |

Two terms need more than a row. The first is "modification." In a PCCP, a modification is not any code change. It is a
change to a specified characteristic or performance measure that the PCCP describes. A bug fix that restores the
validated behavior may be a quality-system change but not a PCCP modification. A change to the model's input features,
output thresholds, training data, or intended population is a modification. The distinction matters because the PCCP's
scope is defined by the modifications it describes, not by the version number.

The second is "impact assessment." The FDA guidance distinguishes an impact assessment from a risk assessment. A risk
assessment asks what can go wrong. An impact assessment asks how each planned modification changes the benefits and
risks of the device, individually and cumulatively, and whether the validation activities maintain safety and
effectiveness. The impact assessment must address unintended bias, subgroup effects, and interactions among
modifications. A table of hazards copied from the risk file is not an impact assessment.

A third term, "GMLP," is often overread. The ten GMLP principles are not a checklist that guarantees regulatory
acceptance. They are a shared baseline from three regulators: leverage multidisciplinary expertise, implement good
software engineering and security practices, use representative data, keep training and test sets independent, use
reference datasets based on best available methods, tailor model design to the intended use, focus on human-AI team
performance, test in clinically relevant conditions, provide clear information, and monitor deployed models. The
principles describe good practice. They do not replace the specific evidence expectations of a submission.

Finally, "PCCP" is a US-specific term. The EU does not have an identical instrument. A manufacturer may use a
"predetermined change control plan" in its EU technical documentation as an internal planning tool, but the legal route
for a significant change is the MDR classification and conformity assessment process. The EU AI Act adds requirements
for high-risk AI systems, including risk management, data governance, technical documentation, transparency, human
oversight, accuracy, robustness, cybersecurity, and post-market monitoring. A US PCCP does not satisfy those obligations
by itself.

## 5. The mechanics, step by step

A PCCP is built before the submission and executed after authorization. The procedure below separates the eligibility
decision from the drafting work, because most failed PCCPs are attempts to authorize a change that should have been a
new submission. Work in order and keep a decision log.

### Step 1 — Confirm the product and the regulatory unit

**Input.** Device description, intended-use statement, software architecture, and current marketing status.

**Output.** A defined device and software function to which the PCCP will attach, with the regulatory pathway.

**Rationale.** A PCCP attaches to a specific device and a specific marketing submission. If the product contains several
functions, the PCCP must state which function it covers. If the product is a component or an accessory, the PCCP must
attach to the finished device or to the accessory's own submission, not to the host device by implication. If the
product is not a device or the status is unresolved, a PCCP is premature.

**Procedure.** (1) Identify the legal manufacturer and the device. (2) Identify the software function and its version.
(3) State whether the product is SaMD, SiMD, an IVD, or part of a combination product. (4) Identify the current or
planned marketing submission. (5) Record the regulatory unit in the PCCP header.

**Branch points.** A standalone AI-DSF with a 510(k) or De Novo is the core use case. A PMA device can use a PCCP, but
the PMA's conditions and manufacturing review still apply. A device-led combination product can include a PCCP for the
device constituent part, not the drug or biologic part. A non-device function does not need a PCCP until it becomes part
of a device.

**Edge cases.** An accessory with its own clearance needs its own PCCP. Software embedded in a hardware device may be
better handled through the device's design-change process, even if the software uses AI. An IVD is regulated under the
IVDR in the EU and under the device framework in the US; the PCCP analysis must use the correct submission type. A
product with multiple intended uses should be split or the PCCP should cover only the function with a stable claim.

### Step 2 — Decide whether a PCCP is the right instrument

**Input.** Product roadmap, planned changes, and the current regulatory status.

**Output.** A decision of PCCP, new submission, or no change, with reasons.

**Rationale.** A PCCP is for planned, bounded, and verifiable modifications. It is not for arbitrary improvement, bug
fixes that are already covered by the quality system, or changes that alter the intended use. Using a PCCP for an
unplanned change is a category error. Using a new submission for every threshold adjustment is wasteful but lawful. The
decision should be based on the nature of the change, not on the team's tolerance for paperwork.

**Procedure.** (1) List the changes the team expects in the next 12-36 months. (2) Mark each as performance, input, use,
or maintenance. (3) Ask whether each change is specific and verifiable. (4) Ask whether each change stays within the
intended use. (5) Ask whether the evidence to validate it can be pre-specified. (6) Choose the instrument for each
change.

**Branch points.** A planned performance change with a measurable acceptance criterion is a PCCP candidate. A new
intended use, population, or clinical action needs a new submission. A bug fix or security patch may be a quality-system
change, not a PCCP modification. A change that cannot be described in advance is not eligible.

**Edge cases.** A change that is "planned" only in the sense that the team hopes to improve the model is not a PCCP
modification. A change that depends on future data the sponsor cannot describe is open-ended. A change to a locked
model's training data may be eligible if the data sources, inclusion criteria, and evaluation are specified. A change to
a continuous-learning model needs automated acceptance and rollback, not just a promise to monitor.

### Step 3 — Inventory and classify the modifications

**Input.** The candidate changes and their technical descriptions.

**Output.** A modification inventory with category, specification, and affected device characteristics.

**Rationale.** The FDA guidance describes three broad modification categories: quantitative performance measures, device
inputs and compatibility, and certain use or performance changes such as subpopulation expansion. The categories are not
a legal taxonomy, but they help organize the description and the evidence. The inventory also exposes interactions: a
threshold change and a retraining change may be safe individually but unsafe together.

**Procedure.** (1) Assign each change an identifier. (2) Write a one-sentence description in plain language. (3)
Classify it as performance, input, use, or maintenance. (4) State the affected output, population, or interface. (5)
State whether it is automatic, manual, or hybrid. (6) State whether it is global or local. (7) Record dependencies among
changes.

**Branch points.** A performance change that preserves the output and population is the simplest case. An input change
may alter the device's compatibility and the validated data domain. A use change may alter the claim. A maintenance
change may be outside the PCCP. A local adaptation may require site-specific validation.

**Edge cases.** A new camera or image acquisition protocol is an input change even if the model is unchanged. A new
language or user interface is a use change if it affects comprehension or workflow. A change to a cloud provider may be
a maintenance change but can affect latency, availability, and security. A change to a third-party model or library may
be a supply-chain change with its own risk.

### Step 4 — Bound the modifications

**Input.** The modification inventory.

**Output.** A description of modifications with measurable specifications, limits, and exclusions.

**Rationale.** The description must be specific enough that a reviewer can tell whether a future change is inside or
outside the authorization. "Improve sensitivity" is not a specification. "Increase sensitivity by no more than 3
percentage points at a fixed specificity of 0.90, with a lower 95 percent confidence bound above 0.85 in the intended
population" is a specification. The description should also state what the modification will not do.

**Procedure.** (1) Write the current baseline value for each affected metric. (2) Write the allowed direction and
magnitude of change. (3) Write the acceptance criterion and the statistical method. (4) Write the population and
subgroup constraints. (5) Write the exclusions. (6) Write the expected frequency and update mode. (7) Have a reviewer
test whether a proposed change is unambiguously inside or outside.

**Branch points.** A numeric bound is appropriate for a performance change. A categorical bound is appropriate for a new
input or compatibility change. A population bound is appropriate for a subgroup expansion. A change with no measurable
bound is not eligible for a PCCP.

**Edge cases.** A bound that is wider than the clinically meaningful difference is not a meaningful control. A bound
that is too narrow may be impossible to meet and will cause repeated deviations. A bound that depends on a data source
the sponsor does not control is fragile. A bound that applies only to the average and not to subgroups can hide
inequity.

### Step 5 — Design the data management practices

**Input.** The data sources, inclusion criteria, and quality controls for each modification.

**Output.** A data management plan that specifies provenance, representativeness, labeling, privacy, and versioning.

**Rationale.** A model change is only as good as the data used to support it. The PCCP must state where the data come
from, how they are selected, how quality is assured, how labels are adjudicated, and how the data are protected. It must
also state how the sponsor will avoid reusing the test set or leaking future data into training. The FDA guidance asks
for data management practices as the first component of the modification protocol.

**Procedure.** (1) Name each data source and its governance. (2) Define inclusion and exclusion criteria. (3) Define
sample size and subgroup coverage. (4) Define labeling and adjudication. (5) Define de-identification and privacy
controls. (6) Define training, validation, and test separation. (7) Define versioning and audit trails. (8) Define how
data quality failures are handled.

**Branch points.** A locked model with periodic retraining can use a defined data cut. A continuous model needs a
streaming data policy and a way to detect distribution shift. A local adaptation needs site-specific data and a
site-specific evaluation. A model trained on data the sponsor does not own needs contractual rights and provenance
evidence.

**Edge cases.** Data from a new site may differ in equipment, population, and labeling practice. Data from a new country
may raise privacy and regulatory transfer issues. Data collected during a public health emergency may not represent
routine practice. Data labeled by a single expert may have systematic error. Data whose test set was used during
development are not an independent test set.

### Step 6 — Design the retraining practices

**Input.** The model architecture, training procedure, and update mode.

**Output.** A retraining protocol that specifies triggers, data, hyperparameters, reproducibility, and approval.

**Rationale.** Retraining is not a single event. It includes data selection, preprocessing, model fitting,
hyperparameter selection, calibration, and evaluation. The PCCP must specify which parts are fixed and which may change.
If the architecture may change, the protocol must say so and justify the evidence. If the architecture is fixed, the
protocol should say that. Reproducibility is part of the control.

**Procedure.** (1) State whether retraining is scheduled, triggered by drift, or continuous. (2) State the data window
and inclusion rules. (3) State the preprocessing and feature pipeline. (4) State the architecture constraints. (5) State
the hyperparameter search space. (6) State the calibration method. (7) State the random seeds and reproducibility
requirements. (8) State who approves the retrained model.

**Branch points.** A scheduled retraining with a fixed architecture is easier to validate. A triggered retraining needs
a defined trigger and a human decision. A continuous retraining needs automated gates. An architecture change is usually
a new submission, not a PCCP modification, unless the PCCP explicitly covers it and the evidence supports it.

**Edge cases.** Retraining on data from the same sites may not address geographic drift. Retraining on data labeled by
the deployed model may create a feedback loop. Retraining with a changed label definition may change the target and the
claim. Retraining without preserving a frozen test set makes performance claims unverifiable.

### Step 7 — Define performance evaluation and acceptance criteria

**Input.** The modification specifications and the clinical claim.

**Output.** A validation plan with endpoints, datasets, statistical methods, and pass/fail criteria.

**Rationale.** The PCCP must state how the sponsor will know whether a modification is safe and effective before it is
deployed. The evaluation should use an independent test set, a clinically relevant endpoint, and a pre-specified
analysis. The acceptance criteria should be tied to the claim, not to an arbitrary improvement. For a 510(k), the
evaluation should support continued substantial equivalence. For a De Novo or PMA, it should support the authorized
benefit-risk profile.

**Procedure.** (1) Name the primary endpoint. (2) Name the secondary endpoints, including safety and subgroup
performance. (3) Name the test dataset and its independence. (4) Name the statistical method and confidence level. (5)
Name the acceptance criterion for each endpoint. (6) Name the stopping rule if the criterion is not met. (7) Name the
human-factors or workflow evaluation if the change affects use.

**Branch points.** A performance improvement can use a non-inferiority or superiority margin. A threshold change should
evaluate the clinical action, not only the AUC. A subgroup expansion should evaluate each subgroup. A new input should
evaluate robustness to the new input's variability. A UX change should evaluate use error.

**Edge cases.** An improvement in average AUC can hide a subgroup decline. A sensitivity gain can come with a
specificity loss that increases false positives. A test set that is too small cannot support a subgroup claim. A test
set from a single site cannot support a multi-site claim. A metric that is not clinically meaningful can pass while the
device remains unsafe.

### Step 8 — Define update, rollback, and communication procedures

**Input.** The deployment architecture and the customer environment.

**Output.** An update procedure covering release, verification, rollback, labeling, and user communication.

**Rationale.** A PCCP is not complete without a plan for how the change reaches the device and how it is reversed if
something goes wrong. The update procedure should state who authorizes the release, how the version is identified, how
customers are notified, how the change is verified in the field, and how a rollback is executed. The procedure should
also state how labeling changes are managed.

**Procedure.** (1) Define the release approval. (2) Define version identification and software bill of materials. (3)
Define deployment stages and verification. (4) Define rollback triggers and method. (5) Define customer notification.
(6) Define labeling and instructions-for-use updates. (7) Define record retention.

**Branch points.** A cloud update can be centralized; a local deployment needs site coordination. A continuous update
needs automated rollback. A change that affects labeling needs a labeling review. A change that affects cybersecurity
needs a security review.

**Edge cases.** A hospital may delay an update, so the manufacturer must support multiple versions. A rollback may not
be possible if the data schema changed. A model update may require a new interface or a new training module. A change to
a third-party component may require a new contract or security assessment.

### Step 9 — Build the traceability table

**Input.** The modification inventory, protocol, and acceptance criteria.

**Output.** A table linking each modification to its data, method, acceptance criterion, and impact assessment.

**Rationale.** The traceability table is the spine of the PCCP. It lets a reviewer see that every modification has a
method and every method has a criterion. It also lets the quality system verify that a proposed change is covered. A
PCCP without traceability is a collection of intentions.

**Procedure.** (1) Create one row per modification. (2) Add columns for data source, retraining method, evaluation
method, acceptance criterion, impact assessment, and monitoring. (3) Check that every row is complete. (4) Check that no
method is used without a criterion. (5) Check that no criterion is used without a monitoring plan. (6) Version the
table.

**Branch points.** A modification with multiple sub-changes needs multiple rows or a structured sub-table. A
modification that depends on another needs a dependency column. A modification that is out of scope should be removed,
not left blank.

**Edge cases.** A row that says "same as above" is not traceable. A row that references a document without a version is
not auditable. A row that has a criterion but no data source is not executable. A row that has a monitoring plan but no
acceptance criterion is not a control.

### Step 10 — Assess individual impact

**Input.** The modification description, the current risk file, and the proposed validation.

**Output.** An individual impact assessment for each modification.

**Rationale.** The impact assessment explains why the modification is safe and effective, not merely that it passed a
test. It should compare the modified device with the unmodified device on benefits, risks, and unintended bias. It
should identify the harm that could increase, the harm that could decrease, and the mitigation that keeps the risk
acceptable.

**Procedure.** (1) Describe the baseline benefit-risk profile. (2) Describe the expected change in benefit. (3) Describe
the expected change in risk. (4) Identify affected subgroups. (5) Identify mitigations and their evidence. (6) State the
residual risk. (7) State the conclusion and its confidence.

**Branch points.** A change that improves average performance but harms a subgroup needs a mitigation or a narrower
claim. A change that reduces a serious harm may justify a larger evidence burden. A change that affects only workflow
may need usability evidence rather than clinical efficacy. A change that affects cybersecurity needs a security risk
assessment.

**Edge cases.** A benefit that is not clinically meaningful is not a benefit. A risk that is rare but severe may be more
important than a common mild risk. A subgroup effect may be invisible in an average metric. A mitigation that depends on
user behavior may fail in practice.

### Step 11 — Assess cumulative and interaction effects

**Input.** The individual impact assessments and the modification dependencies.

**Output.** A cumulative impact assessment covering combinations, sequencing, and long-term effects.

**Rationale.** Modifications do not occur in isolation. A threshold change and a retraining change can interact. A
series of small sensitivity gains can accumulate into a material change in the false-positive rate. A local adaptation
can interact with a global update. The FDA guidance asks for cumulative impact, including how one modification affects
another.

**Procedure.** (1) List all combinations that could occur. (2) Assess the combined effect on each endpoint. (3) Assess
the effect of sequencing. (4) Assess the effect of repeated updates over time. (5) Assess the effect on subgroups. (6)
State the cumulative acceptance criterion. (7) Define a review trigger if the cumulative effect approaches the boundary.

**Branch points.** Independent modifications can be assessed separately if the interaction is negligible. Dependent
modifications need a joint assessment. Repeated updates need a drift and cumulative-performance check. A combination
that exceeds the authorized boundary needs a new submission.

**Edge cases.** A model that improves each year but degrades against the original baseline is a cumulative failure. A
local adaptation that works at one site but not another is an interaction. A retraining cycle that gradually changes the
case mix is a slow drift. A series of threshold changes can cross a clinical decision boundary without any single change
appearing material.

### Step 12 — Integrate with the quality system and risk file

**Input.** The PCCP, the risk management file, and the change-control procedure.

**Output.** A controlled process for approving and implementing authorized changes.

**Rationale.** The PCCP is executed through the quality system. Each authorized change needs a design change record, a
risk assessment, a verification and validation report, a labeling review, and an approval. The risk file needs to be
updated with the new hazard and risk information. The post-market surveillance plan needs to include the monitoring
specified in the PCCP.

**Procedure.** (1) Map each PCCP modification to the change-control procedure. (2) Define the records that will be
created. (3) Define the risk-file update. (4) Define the verification and validation review. (5) Define the labeling
review. (6) Define the post-market data feed. (7) Define the annual review.

**Branch points.** A change within the PCCP follows the authorized procedure. A change outside the PCCP follows the
new-submission route. A change that is urgent for safety may need a corrective action and a regulatory notification. A
change that affects multiple devices needs a portfolio-level review.

**Edge cases.** A quality system that treats the PCCP as a one-time document will not produce the required records. A
risk file that is not updated will not reflect the modified device. A post-market plan that does not name the PCCP
metrics will not detect drift. An audit that cannot trace a deployed version to a PCCP row is a finding.

### Step 13 — Prepare the submission and labeling

**Input.** The PCCP, the device evidence, and the draft labeling.

**Output.** A submission package and labeling that disclose the PCCP and its limits.

**Rationale.** The PCCP is reviewed with the marketing submission. The submission must explain the device, the
modifications, the protocol, and the impact. The labeling must tell users what the PCCP covers and that updates may
change performance. The labeling should not imply that the device will improve indefinitely or that the user does not
need to monitor it.

**Procedure.** (1) Place the PCCP in the submission. (2) Cross-reference the device description and evidence. (3) Write
the labeling disclosure. (4) Describe the update notification. (5) Describe the user's role in monitoring. (6) Describe
the limitations. (7) Check the submission format and eSTAR requirements.

**Branch points.** A 510(k) PCCP must support substantial equivalence. A De Novo PCCP must support the new
classification and special controls. A PMA PCCP must fit the approved conditions. A device-led combination product must
coordinate with the lead center.

**Edge cases.** A labeling statement that says "automatically improves" is an overclaim. A labeling statement that says
"no user action required" may be false for a local deployment. A submission that describes the PCCP but not the
monitoring is incomplete. A submission that relies on a future data source the sponsor does not control is fragile.

### Step 14 — Monitor, audit, and handle deviations

**Input.** The authorized PCCP, deployed versions, and post-market data.

**Output.** A monitoring record, an audit trail, and a deviation decision.

**Rationale.** Authorization is not the end of the control. The sponsor must monitor whether the assumptions behind the
PCCP hold, whether the modifications perform as expected, and whether cumulative effects remain within the authorized
boundary. A deviation is not necessarily a violation, but it must be recognized and handled through the correct pathway.

**Procedure.** (1) Collect the pre-specified monitoring data. (2) Compare performance with the acceptance criteria. (3)
Review complaints and adverse events. (4) Review subgroup performance. (5) Review cumulative change. (6) Audit the
change records. (7) If a deviation occurs, stop, assess, and notify as required.

**Branch points.** A change within the PCCP is implemented and documented. A change outside the PCCP requires a new
submission or a regulatory consultation. A safety signal requires a corrective action and possibly a recall. A
monitoring failure requires a plan update.

**Edge cases.** A model that performs well on average but poorly in a new subgroup is a deviation from the intended
benefit. A model that is retrained on data from a new site without the specified site validation is a deviation. A
change that is implemented in the cloud but not documented is a quality-system failure. A change that is authorized in
the US but not in the EU cannot be deployed globally without a separate EU assessment.

### Decision table: PCCP or new submission

| Change | Within intended use | Pre-specifiable | Likely route |
|---|---|---|---|
| Threshold adjustment within a specified range | Yes | Yes | PCCP |
| Retraining with fixed architecture and specified data | Yes | Yes | PCCP |
| New input modality | Usually no | Maybe | New submission |
| New patient population | No, unless already in claim | Maybe | New submission or claim expansion |
| New clinical action | No | No | New submission |
| Cybersecurity patch | Yes | Yes | QMS change, not necessarily PCCP |
| Labeling clarification | Yes | Yes | QMS change, check labeling rules |
| Architecture change | Usually no | Maybe | New submission unless explicitly covered |
| Local site adaptation | Yes, if within claim | Yes | PCCP with site validation |
| Continuous learning without bounds | Yes | No | Not eligible; new submission |

### Decision table: update mode

| Mode | Description | Evidence emphasis | Monitoring |
|---|---|---|---|
| Locked | Fixed at release | Version-specific validation | Periodic performance review |
| Scheduled retraining | Fixed intervals, fixed architecture | Pre-specified retraining protocol | Each cycle plus drift |
| Triggered retraining | Retrain when a trigger fires | Trigger definition and validation | Trigger audit and performance |
| Continuous learning | Automatic updates | Automated acceptance and rollback | Real-time drift and safety |

### Decision table: jurisdiction

| Jurisdiction | Instrument | Legal basis | Key limitation |
|---|---|---|---|
| US | PCCP | FD&C Act §515C; FDA guidance 18 August 2025 | Within intended use and substantial equivalence |
| EU | No direct PCCP equivalent | MDR 2017/745; IVDR 2017/746 | Significant changes need conformity assessment |
| EU AI Act | High-risk AI obligations | Regulation (EU) 2024/1689, amended by 2026/1744 | Separate from device law |
The procedure is deliberately conservative. A PCCP that is too narrow is annoying; a PCCP that is too broad is a
regulatory and safety risk. The next section shows how the pieces fit in a realistic case.

## 6. Worked example

The product, sponsor, predicate, and study numbers in this case are hypothetical. They are constructed to show how a
PCCP is built and where it fails. A real submission requires the current FDA guidance, a predicate search, a
pre-submission, and qualified regulatory review.

### 6.1 Product and proposed changes

SepsisGuard is a cloud-based software function that analyzes electronic health record (EHR) data for adult inpatients
and produces a sepsis deterioration risk score every hour. The intended user is a hospital rapid-response nurse or
attending clinician. The output is a risk score and a trend indicator, not a diagnosis. The intended use is to support
earlier clinical review of patients who may be deteriorating from sepsis. The device is a Class II software as a medical
device (SaMD) with a 510(k) predicate that predicts clinical deterioration in hospitalized adults.

The sponsor plans four changes over three years. First, quarterly retraining of the existing model with new EHR data,
keeping the architecture fixed. Second, a threshold adjustment that raises the alert threshold from 0.20 to 0.25 to
reduce alert fatigue. Third, recalibration of the risk score to improve calibration in the 65-and-older population.
Fourth, a new input: serum lactate values, which are available in most but not all hospitals. The sponsor wants all four
covered by one PCCP.

The PCCP is submitted with the initial 510(k). The device is not yet cleared. The sponsor has a 12-month post-launch
data plan and a contract with three pilot hospitals.

### 6.2 Intended use and boundary

The intended-use statement is:

"SepsisGuard is intended for use by trained hospital clinicians to support earlier clinical review of adult inpatients
who may be deteriorating from sepsis. The software analyzes EHR data and displays a risk score and trend. It does not
diagnose sepsis, does not recommend a specific treatment, and does not replace clinical judgment. It is not intended for
pediatric patients, outpatients, or patients in whom a sepsis diagnosis is already established."

The boundary matters for the PCCP. Quarterly retraining, threshold adjustment, and recalibration are performance changes
within the intended use. Adding lactate is an input change. If lactate is already part of the intended use and the
validated input set, the change may be within the PCCP. If lactate expands the device's input domain to hospitals that
did not have it, the change may affect compatibility, validation, and the claim. The sponsor must decide whether the
lactate modification is a PCCP modification or a new submission.

### 6.3 Base pathway and predicate

The sponsor identifies a Class II predicate for a sepsis deterioration alert with a similar user, population, and
output. The predicate's 510(k) summary reports an area under the receiver operating characteristic curve (AUC) of 0.81
in a multi-site retrospective study and a sensitivity of 0.68 at a fixed alert rate of 2 alerts per 100 patient-days.
The predicate does not report calibration or subgroup performance. The sponsor's model has an AUC of 0.84 in its
internal test set, but the test set is enriched for sepsis cases and comes from two of the three pilot sites.

The base pathway is a 510(k) with a PCCP. The sponsor must show substantial equivalence for the base device and describe
the planned modifications. The PCCP does not replace the base evidence. It adds a plan for future changes. The sponsor's
pre-submission asks three questions: whether the quarterly retraining is eligible for a PCCP, whether the threshold
change needs a separate submission, and whether the lactate input can be covered.

### 6.4 The PCCP as drafted

The draft PCCP has three sections.

The description of modifications says the sponsor may "retrain the model quarterly to improve performance," "adjust the
alert threshold to optimize the alert burden," "recalibrate the score for older adults," and "add lactate as an optional
input." The description gives no numeric bounds, no acceptance criteria, and no exclusions.

The modification protocol says the sponsor will use "representative EHR data," "standard retraining methods,"
"independent validation," and "clinical review." It does not specify the data window, the label definition, the test-set
separation, the statistical method, or the acceptance threshold.

The impact assessment says the changes are "expected to improve performance and reduce alert fatigue" and that "risks
will be monitored." It does not analyze subgroup effects, cumulative effects, or the interaction between the threshold
change and retraining.

This draft would fail for three reasons. First, the modifications are not bounded; a reviewer cannot tell what "improve
performance" means or when a change exceeds the authorization. Second, the protocol is not executable; "representative
data" and "independent validation" are intentions, not methods. Third, the impact assessment does not assess impact; it
asserts benefit.

### 6.5 A corrected PCCP

The corrected description bounds each modification.

| ID | Modification | Bound | Mode | Exclusion |
|---|---|---|---|---|
| M1 | Quarterly retraining | Fixed architecture; data window 24 months; AUC not below baseline minus 0.02; calibration slope 0.9–1.1 | Scheduled, manual release | No new input; no new population |
| M2 | Alert threshold | 0.20–0.25; sensitivity not below 0.65 at the selected threshold; alert rate not above 3 per 100 patient-days | Manual | No change to score meaning |
| M3 | Recalibration for age 65+ | Calibration intercept -0.1 to +0.1; slope 0.9–1.1; no change to ranking | Manual | No change to model architecture |
| M4 | Optional lactate input | Only where lactate is available within 12 hours; missingness indicator required; AUC not below baseline minus 0.01 | Manual, site opt-in | No pediatric use; no outpatient use |

The protocol specifies the data sources, the label definition, the inclusion and exclusion criteria, the frozen test
set, the subgroup analysis, the statistical method, and the acceptance criteria. It also specifies the rollback
procedure: if a released model fails a post-deployment check, the prior version is restored within 24 hours and the
failure is investigated.

The impact assessment compares each modification with the unmodified device. M1 is expected to maintain or improve
average discrimination, with a risk of drift in sites that change coding practice. M2 reduces alert burden but may
reduce sensitivity; the assessment requires a sensitivity floor. M3 improves calibration for older adults but may affect
the threshold's operating characteristics; the assessment requires a joint evaluation with M2. M4 may improve
performance where lactate is available but may create a two-tier device with different performance across sites; the
assessment requires site-specific reporting and a missingness analysis.

### 6.6 Evidence gap

The base evidence has three gaps. First, the internal test set is enriched and comes from two sites, so it does not
estimate performance in the intended population. Second, the predicate's public summary does not report calibration, so
the sponsor cannot rely on the predicate's calibration experience. Third, the subgroup analysis for older adults and for
patients with chronic kidney disease is underpowered.

The PCCP evidence plan addresses the gaps over time. The initial submission uses a held-out test set from a third site
and a pre-specified subgroup analysis. The post-market plan collects calibration and subgroup data from the three pilot
sites. The first retraining cycle uses a locked test set that includes the third site. The sponsor commits to a minimum
of 150 sepsis cases and 1,500 non-sepsis cases in the test set, with at least 300 patients aged 65 and older and 200
patients with chronic kidney disease.

The arithmetic for the test set is not arbitrary. If the expected sensitivity is 0.70 and the sponsor wants a 95 percent
confidence interval half-width of 0.05, the normal-approximation sample size is 1.96 squared times 0.70 times 0.30
divided by 0.05 squared, or 323 positive cases. The sponsor's 150 cases would produce a much wider interval. The plan
must either increase the positive cases or state a wider precision. A PCCP that promises a subgroup analysis without
enough cases is a plan to fail.

### 6.7 Two diagnostics that would sink the submission

**Diagnostic 1: the test set is not independent.** The sponsor's data pipeline selects "new" data by visit date but does
not exclude patients who appear in the training set. A patient can have multiple admissions, and the same admission can
appear in both training and test sets after a data refresh. The test-set AUC is 0.86, but the patient-overlap audit
finds that 22 percent of test patients have at least one encounter in the training data. The independence claim is
false. A reviewer who discovers this cannot rely on the performance estimate, and the PCCP's entire acceptance framework
collapses because the baseline is invalid.

**Diagnostic 2: the acceptance criterion ignores calibration and subgroups.** The PCCP uses AUC as the only performance
criterion. The model passes the AUC bound in the first retraining cycle, but calibration deteriorates: the observed
sepsis rate in the highest-risk decile is 18 percent while the mean predicted risk is 31 percent. The alert threshold
therefore triggers more alerts than expected, and the false-positive burden rises. In patients with chronic kidney
disease, sensitivity falls from 0.68 to 0.54. The average AUC is unchanged because the model improved in the majority
subgroup. The PCCP did not require calibration or subgroup criteria, so the sponsor has no pre-specified basis to reject
the update. A reviewer can reasonably conclude that the plan does not control the risks it claims to control.

### 6.8 Vary one assumption

The base case treats M4, the optional lactate input, as a PCCP modification. Change the assumption: the sponsor wants to
make lactate a required input for all sites and to expand the intended population to include patients with suspected
sepsis in the emergency department. The intended use now covers a new setting and a new population, and the input is no
longer optional. The change is not a performance change within the existing claim; it is a new claim.

The US pathway changes from a 510(k) with a PCCP to a new 510(k) or a De Novo, depending on the predicate and the risk.
The existing predicate does not cover the emergency department population or a required lactate input. The sponsor needs
a new predicate search, a new intended-use statement, and new clinical evidence. The PCCP can still cover future
retraining within the new claim, but it cannot authorize the expansion itself.

The EU pathway changes as well. The MDR classification may move from Class IIa to Class IIb or III depending on the harm
characterization and the new population. The clinical evaluation must cover the emergency department setting. The IVDR
is not implicated because the product is not an IVD. The EU AI Act analysis may change if the new use is high-risk. The
PCCP concept does not substitute for the MDR conformity assessment.

| Assumption changed | US route | EU route | PCCP consequence |
|---|---|---|---|
| Base: optional lactate, adult inpatient | 510(k) with PCCP | IIa or IIb, technical documentation | M4 covered as input change |
| Lactate required, ED population added | New 510(k) or De Novo | New clinical evaluation, class may rise | M4 becomes part of new base device |
| Retraining changes architecture | New 510(k) or PMA | New conformity assessment | Outside PCCP unless explicitly covered |
| Continuous learning with no human review | New submission with automated controls | Not equivalent to MDR change control | PCCP must include automated gates |
| Threshold change only, within bound | 510(k) with PCCP | QMS change, assess significance | M2 covered |

The lesson is that a PCCP is a boundary, not a blanket. The sponsor can pre-authorize bounded performance changes. The
sponsor cannot pre-authorize a new claim and call it an update. The difference is visible in the intended-use statement,
the population, the setting, and the input requirements. A team that writes the boundary before the roadmap will spend
less time rewriting the PCCP after the first product pivot.

## 7. Code, query, or calculation

The script below encodes the eligibility table, a simple acceptance-criteria check, the sample-size calculation used in
Section 6, and a cumulative-effect helper. It is intentionally small. It does not decide whether a change is within the
intended use; that is a legal and clinical judgment supplied by the caller.

```python
"""PCCP eligibility and validation arithmetic for Module 14, Topic 04.

Teaching aid only. It does not replace FDA guidance or a qualified review.
"""

from __future__ import annotations

import math


def pccp_route(
    change: str,
    within_intended_use: bool,
    pre_specifiable: bool,
    bounded: bool,
) -> str:
    """Return a provisional route for a planned change."""
    if not within_intended_use:
        return "new marketing submission: intended use changes"
    if not pre_specifiable:
        return "not eligible: change cannot be specified in advance"
    if not bounded:
        return "not eligible: change has no measurable boundary"
    return "PCCP candidate: bounded change within intended use"


def check_acceptance(
    observed: dict[str, float],
    criteria: dict[str, tuple[float, float]],
) -> tuple[bool, list[str]]:
    """Check observed metrics against inclusive (lower, upper) criteria."""
    failures = []
    for name, (lower, upper) in criteria.items():
        if name not in observed:
            failures.append(f"{name}: missing")
            continue
        value = observed[name]
        if not lower <= value <= upper:
            failures.append(f"{name}: {value:.3f} outside [{lower:.3f}, {upper:.3f}]")
    return (len(failures) == 0, failures)


def sample_size_proportion(p: float, half_width: float, z: float = 1.96) -> int:
    """Normal-approximation sample size for a proportion confidence interval."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be between 0 and 1")
    if half_width <= 0:
        raise ValueError("half_width must be positive")
    return math.ceil((z**2) * p * (1.0 - p) / (half_width**2))


def cumulative_delta(deltas: list[float]) -> float:
    """Additive cumulative effect; use only when interactions are negligible."""
    return sum(deltas)


if __name__ == "__main__":
    changes = [
        ("quarterly retraining", True, True, True),
        ("threshold 0.20 to 0.25", True, True, True),
        ("new pediatric population", False, True, True),
        ("open-ended continuous learning", True, False, False),
    ]
    for name, within, pre, bounded in changes:
        print(f"{name:34s} -> {pccp_route(name, within, pre, bounded)}")

    observed = {"auc": 0.83, "calibration_slope": 0.95, "sensitivity": 0.67}
    criteria = {
        "auc": (0.82, 1.0),
        "calibration_slope": (0.90, 1.10),
        "sensitivity": (0.65, 1.0),
    }
    ok, failures = check_acceptance(observed, criteria)
    print(f"\nacceptance passed: {ok}")
    for failure in failures:
        print(f"  {failure}")

    n = sample_size_proportion(p=0.70, half_width=0.05)
    print(f"\npositive cases for p=0.70, half-width=0.05: {n}")
    print(f"cumulative additive delta: {cumulative_delta([0.01, -0.02, 0.015]):.3f}")
```

Expected output from a Python 3.12 run:

```text
quarterly retraining               -> PCCP candidate: bounded change within intended use
threshold 0.20 to 0.25             -> PCCP candidate: bounded change within intended use
new pediatric population           -> new marketing submission: intended use changes
open-ended continuous learning     -> not eligible: change cannot be specified in advance

acceptance passed: True

positive cases for p=0.70, half-width=0.05: 323
cumulative additive delta: 0.005
```

The `check_acceptance` function treats each metric as independent. That is a simplification. In a real PCCP, the
acceptance rule may require all metrics to pass, a composite endpoint, or a hierarchical test. The function also does
not adjust for multiple comparisons. If the PCCP evaluates five subgroups at a 95 percent confidence level, the chance
of at least one false failure is greater than 5 percent unless the sponsor pre-specifies a multiplicity adjustment or a
gatekeeping strategy.

The sample-size function uses the normal approximation. For small proportions or small samples, an exact binomial or
Wilson interval is more appropriate. The function is included to make the arithmetic visible, not to replace a
statistician. The number 323 is the number of positive cases needed for a 0.05 half-width at p equal to 0.70. If the
sponsor wants a half-width of 0.04, the requirement rises to 505 positive cases. That difference is often the difference
between a feasible study and an infeasible one.

## 8. Data, standards, and regulatory context

> **Note:** This section is educational and is not legal advice. PCCP eligibility, scope, and content are fact-specific and are reviewed by FDA in the context of a marketing submission. Confirm the current statute, guidance, and submission requirements with qualified regulatory counsel.

The US legal basis is section 515C of the FD&C Act, added by FDORA in 2022. The statute authorizes FDA to approve a PCCP
in a 510(k), De Novo, or PMA submission. The PCCP must describe the modifications, the methods for developing and
validating them, and the impact on safety and effectiveness. Once authorized, the manufacturer may implement
modifications that are consistent with the PCCP without a new submission. A modification that is not consistent with the
PCCP requires a new submission or another appropriate regulatory route.

The operative guidance is "Marketing Submission Recommendations for a Predetermined Change Control Plan for Artificial
Intelligence-Enabled Device Software Functions." It was originally issued on 4 December 2024 and reissued on 18 August
2025. The guidance applies to AI-enabled device software functions in 510(k), De Novo, and PMA submissions, including
the device constituent part of device-led combination products. It describes three PCCP components: the description of
modifications, the modification protocol, and the impact assessment. It also includes appendices with detailed questions
and hypothetical case examples.

| Source | Status | What it constrains | What it does not do |
|---|---|---|---|
| FD&C Act §515C | Binding US statute | Authorizes PCCPs and limits them to described modifications | Does not specify the content of a PCCP |
| FDA PCCP guidance, 18 August 2025 | Guidance | Describes FDA's recommended content and review approach | Is not a statute or a safe harbor |
| FD&C Act §513(i) | Binding US statute | Defines substantial equivalence for 510(k) | Does not allow a PCCP to change intended use |
| 21 CFR 820 (QMSR) | Binding US regulation | Requires design controls and change control | Does not itself authorize a PCCP |
| ISO 13485:2016 | Voluntary standard, incorporated by QMSR | Sets QMS and design-change requirements | Does not define PCCP scope |
| ISO 14971:2019 | Voluntary standard | Sets risk management requirements | Does not replace the impact assessment |
| IEC 62304:2006+A1:2015 | Voluntary standard | Sets software lifecycle processes | Does not determine PCCP eligibility |
| FDA AI lifecycle draft, 7 January 2025 | Draft guidance | Describes lifecycle management and performance monitoring | Is not final and may change |
| MDR 2017/745 | Binding EU regulation | Sets EU device classification and conformity assessment | Does not create a US-style PCCP |
| IVDR 2017/746 | Binding EU regulation | Sets EU IVD classification and conformity assessment | Does not create a US-style PCCP |
| AI Act 2024/1689, amended by 2026/1744 | Binding EU regulation | Sets AI-system obligations and timelines | Does not replace MDR/IVDR |
| MDCG 2025-6 | EU guidance | Discusses AI Act and medical device interaction | Is not binding |

The ten GMLP principles, published jointly by FDA, Health Canada, and the UK Medicines and Healthcare products
Regulatory Agency on 27 October 2021, are the practice baseline. They are: leverage multidisciplinary expertise across
the total product life cycle; implement good software engineering and security practices; ensure clinical study
participants and datasets are representative of the intended patient population; keep training datasets independent of
test sets; base reference datasets on best available methods; tailor model design to the available data and intended
use; focus on the performance of the human-AI team; test in clinically relevant conditions; provide clear, essential
information to users; and monitor deployed models while managing retraining risks.

The principles are not a checklist for a submission. They are a shared vocabulary for design reviews, risk reviews, and
supplier conversations. A team that can map each principle to a specific artifact, owner, and review is in a better
position than a team that recites the list. A team that claims GMLP compliance without evidence has added a slogan, not
a control.

The PCCP guidance is more specific than GMLP. It asks for a description of modifications that is specific, verifiable,
and bounded; a modification protocol that covers data management, retraining, performance evaluation, and update
procedures; and an impact assessment that covers individual and cumulative benefits, risks, and bias. It also asks for a
traceability table linking each modification to the protocol methods and acceptance criteria. The guidance encourages a
pre-submission discussion for complex or novel PCCPs.

The FDA's draft lifecycle guidance, issued 7 January 2025, adds context on post-market performance monitoring. It
describes monitoring for performance change and degradation, data drift, changes in patient demographics or disease
prevalence, data-pipeline integrity, and user behavior. It notes that a monitoring plan may be a special control for a
De Novo device or a condition of approval for a PMA. The draft is not final, but it is a useful statement of FDA's
current thinking.

In the EU, the MDR and IVDR do not have a directly equivalent PCCP instrument. A significant change to a device is
assessed under the MDR's change-control and conformity-assessment requirements. The manufacturer must determine whether
the change affects the device's compliance with the general safety and performance requirements, the clinical
evaluation, the risk management file, or the technical documentation. A notified body may need to review the change. The
EU AI Act adds obligations for high-risk AI systems, including risk management, data governance, technical
documentation, transparency, human oversight, accuracy, robustness, cybersecurity, and post-market monitoring.
Regulation (EU) 2026/1744, the Digital Omnibus on AI, moved the high-risk obligations for Annex I products, including
medical devices, to 2 August 2028, while Annex III systems moved to 2 December 2027. The deferral does not suspend
MDR/IVDR obligations.

The standards landscape is supportive but not determinative. ISO 13485:2016 requires design controls and change control.
ISO 14971:2019 requires risk management across the lifecycle. IEC 62304:2006+A1:2015 requires software lifecycle
processes and configuration management. IEC 62366-1:2015+A1:2020 requires usability engineering. None of these standards
says what a PCCP must contain. They provide the management system in which a PCCP is executed.

The scope boundaries deserve emphasis. A PCCP is available for AI-enabled device software functions, not for every
device change. Software embedded in a hardware device may be better handled through the device's design-change process,
because the hardware, firmware, and software interact and the change may affect electrical safety, electromagnetic
compatibility, or sterilization. An in vitro diagnostic device is regulated under the IVDR in the EU and under the
device framework in the US; its PCCP analysis must use the correct submission type and the correct clinical evidence
standard. A device-led combination product can include a PCCP for the device constituent part, but the drug or biologic
constituent part has its own regulatory framework. A product that is borderline between device and non-device should
resolve its status before drafting a PCCP; a PCCP cannot cure an unresolved qualification question.

## 9. Industry practice and case evidence

Public PCCP evidence is still limited, but several documented examples show how the instrument is used.

**Apple Irregular Rhythm Notification Feature.** FDA's 510(k) clearance letter for K231173, dated 21 July 2023, states
that the substantial equivalence determination included review and clearance of a PCCP under section 515C. The PCCP
covered adjustments to the tachogram classification algorithm's atrial fibrillation operating-point threshold and
retraining with additional datasets using the same architecture and parameter count, plus changes to the
confirmation-cycle algorithm. The summary states that the model was trained, tuned, and locked before release, with no
adaptive or continuous field learning, and that the datasets met demographic minimums for age, sex, race, and skin tone.
The documented lesson is that a PCCP can cover a bounded threshold and retraining change while keeping the model locked.
The inference is that the demographic minimums were as important as the algorithm description.

**PCCP device list.** FDA maintains a public list of authorized devices with PCCPs. Trade press reported in October 2026
that the list contained about 240 devices, with the largest share from a single diabetes-device manufacturer. The number
changes as devices are added and removed. The list is useful for competitive and precedent research, but it is not a
template. A device on the list may have a PCCP that is narrow, broad, or specific to a manufacturer's quality system.
The inference that a similar device's PCCP will fit your product is unsafe without reading the actual scope.

**FDA hypothetical case examples.** FDA's PCCP case examples include a sepsis prediction model with site-specific
adaptation and a cardiac telemetry model with continuous learning. The sepsis example describes local input reweighting,
local fine-tuning, and site-specific alert thresholds, with the locally adapted model locked before validation and
manually implemented with manufacturer review. The cardiac example describes automated, global, periodic retraining with
no human in the loop, automated acceptance criteria, sequestered test data, auto-blocked deployment on failure, audit
trails, and drift monitoring. The documented lesson is that both locked and continuous models can be addressed, but the
controls scale with autonomy. The inference is that the continuous example is not a template for a team without
automated validation infrastructure.

**Industry practice: PCCP as a product roadmap artifact.** Mature teams write the PCCP alongside the product
requirements, not after the clinical study. The roadmap identifies which changes are planned, which are optional, and
which are out of scope. The quality system identifies how each change will be approved. The data team identifies how
each change will be evaluated. The regulatory team identifies which changes require a new submission. This practice
reduces the chance that a late feature request invalidates the submission.

**Industry practice: conservative scope.** Companies with successful PCCPs tend to be conservative. They authorize a
small number of well-understood changes, with clear bounds and strong monitoring. They do not authorize "any future
model improvement." They often separate the PCCP into a core plan and a later supplement. The trade-off is more
submissions, but the authorized changes are easier to defend.

**Industry practice: labeling and customer communication.** A PCCP changes the labeling relationship. The label must
tell users that the device may be updated and that performance may change. The manufacturer must tell customers which
version they are running and what changed. Hospitals increasingly ask for software version tracking, rollback
guarantees, and notification before an update. A PCCP that does not address these operational realities may be
technically authorized but practically unmanageable.

**What the cases do not prove.** The Apple clearance does not prove that a threshold change is always PCCP-eligible. The
240-device list does not prove that a PCCP is easy to obtain. The FDA examples do not prove that a continuous-learning
model is appropriate for every product. Each case reflects a specific device, claim, data set, and quality system. The
transferable lesson is the method: bound the change, specify the evidence, assess the impact, integrate with the QMS,
and monitor.

**Industry practice: pre-submission discipline.** A sponsor with a novel PCCP should not surprise FDA at the
marketing-submission stage. A pre-submission can clarify whether a proposed modification is eligible, whether the
acceptance criteria are adequate, and whether the planned evidence is sufficient. The pre-submission is not a guarantee,
and FDA's feedback is not a binding approval. Its value is in exposing a disagreement before the sponsor has locked the
protocol, enrolled the study, or built the deployment pipeline. A useful pre-submission package includes the intended
use, the modification inventory, the draft protocol, the proposed acceptance criteria, and the specific questions the
sponsor needs answered.

**Industry practice: supplier and cloud changes.** Many PCCPs assume the sponsor controls the model, the data pipeline,
and the deployment environment. In practice, a cloud provider may change a managed service, a data vendor may change a
coding vocabulary, or a model supplier may release a new version. These changes can alter device performance without any
change to the sponsor's code. A mature PCCP identifies critical suppliers, requires change notification, and defines how
a supplier change is assessed. A software bill of materials and a configuration baseline are not paperwork; they are the
evidence that the deployed device matches the authorized version.

**Industry practice: version transparency for customers.** Hospitals and health systems increasingly ask for model
version, training-data vintage, performance summary, and known limitations at the point of use. A PCCP that authorizes
updates without a customer-facing version record creates an operational gap. The manufacturer may know that version 4.2
is deployed, but the clinician may not know that the alert threshold changed. A labeling and communication plan should
state what changed, when, and how the user can verify the version. The plan should also state how a site can decline or
delay an update without losing support.

**Industry practice: post-market evidence as a PCCP input.** Companies that operate PCCPs well treat post-market data as
an input to the next PCCP, not only as a compliance obligation. If monitoring shows that a subgroup performs worse than
expected, the next PCCP can add a subgroup-specific recalibration or a new acceptance criterion. If monitoring shows
that a threshold change increased false negatives, the next PCCP can narrow the threshold range. The feedback loop is
the point of the instrument. A PCCP that is written once and never revised is a missed opportunity and a drift risk.

**What the cases do not prove, continued.** The public cases do not show the internal quality-system records, the
pre-submission dialogue, or the failures that were corrected before submission. They show the authorized scope, not the
negotiations behind it. A team that copies the visible scope without building the underlying evidence and monitoring
capacity will have a plan that looks like a PCCP but does not function like one. The transferable asset is the operating
system: data governance, independent evaluation, change control, rollback, and post-market review.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Open-ended authorization | PCCP says "improve the model" | No measurable boundary | Reviewer cannot tell what is authorized | Write numeric bounds and exclusions |
| New claim disguised as an update | Roadmap adds a population or setting | Intended use changes without a new submission | Unauthorized marketing | Compare each change with the intended-use statement |
| Missing test-set independence | Test AUC is much higher than field data | Patient or site leakage | Invalid baseline and acceptance criteria | Audit overlap and freeze test data |
| AUC-only acceptance | Calibration and subgroups are not measured | Average performance hides harm | Unsafe update passes | Pre-specify calibration and subgroup criteria |
| Unbounded continuous learning | No rollback or human gate | Model updates faster than validation | Drift and safety signals missed | Automated gates and rollback |
| Data source not controlled | Sponsor cannot guarantee future data | Protocol depends on a third party | PCCP cannot be executed | Contractual data rights and fallback |
| No cumulative assessment | Changes assessed one at a time | Small effects accumulate | Boundary crossed without notice | Cumulative metrics and review triggers |
| QMS disconnect | PCCP not in change-control procedure | Changes lack records and approvals | Audit finding and traceability gap | Map PCCP to design-change records |
| Labeling overclaim | Label says "continuously improves" | Marketing exceeds evidence | Misleading labeling | Align label with authorized scope |
| Monitoring plan absent | No named metrics or frequency | Drift detected late | Post-market harm | Specify data, frequency, and action |
| Local adaptation without site validation | One site performs worse | Site-specific data shift | Unequal performance | Site validation and reporting |
| Third-party model change | Supplier updates a dependency | Supply-chain change outside PCCP | Unauthorized device change | Software bill of materials and supplier controls |
| Deviation not recognized | Change is "small" but outside scope | Scope creep | Enforcement risk | Deviation procedure and training |
| Overbroad subgroup claim | Subgroup analysis underpowered | False confidence | Inequitable performance | Power the subgroup or narrow the claim |

A red flag is a prompt to inspect the PCCP, not a reason to abandon it. The most common root cause is a roadmap that was
written for investors rather than regulators. The fix is to translate each roadmap item into a device change, a claim
change, or a maintenance change, and then assign the correct regulatory route.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The intended use is stable | PCCP cannot expand the claim | New population, setting, or action | Compare each change with the labeled claim |
| The modification is bounded | Reviewers need a scope | Vague or open-ended change | Test whether a proposed change is inside or outside |
| The data are representative | Validation must generalize | New site, equipment, or population | Subgroup and site analysis |
| The test set is independent | Performance estimate must be unbiased | Patient or temporal leakage | Overlap audit and frozen test set |
| The acceptance criteria are clinically meaningful | Passing a metric must mean safety | Metric disconnected from harm | Link criteria to risk analysis |
| The update mode is controllable | Changes must be reversible | Continuous learning without rollback | Deployment and rollback drills |
| The QMS executes the PCCP | Authorization is not implementation | No change records or approvals | Audit trail and design-change review |
| The monitoring data are available | Post-market evidence must be collected | Data access or contract limits | Data-use agreements and pilot |
| The regulatory pathway remains valid | Predicate or classification can change | Predicate withdrawn or reclassified | Recheck before each update |
| The jurisdiction accepts the plan | US PCCP is not global | EU or other market requirements | Separate jurisdiction assessment |

The limits of this topic are explicit. It does not provide a legal opinion. It does not cover every FDA review issue or
every EU change-control scenario. It does not guarantee that a PCCP will be authorized. It does not replace a
pre-submission, a quality-system audit, or a clinical evaluation. It gives a structured way to decide, draft, and
monitor a PCCP.

## 12. Exercises

### Exercise 1 — Distinguish the instruments (recall, 20 minutes)

Write a one-page table with three columns: GMLP, PCCP, QMS change control. For each, state what it is, what it
authorizes, what evidence it needs, and what it does not do. Deliverable: the table plus a short example of a change
that belongs in each column. Acceptance criteria: the PCCP column does not claim to authorize new intended uses; the QMS
column does not claim to replace a submission.

### Exercise 2 — Eligibility screen (application, 30 minutes)

Take five planned changes from a product roadmap. For each, decide whether it is a PCCP candidate, a new submission, or
a QMS change. Record the facts that drive the decision. Deliverable: a decision table with reasons. Acceptance criteria:
each row addresses intended use, pre-specifiability, and boundedness; at least one row is a new submission.

### Exercise 3 — Bound a modification (application, 40 minutes)

Choose one performance change and write a bounded description. Include the baseline, the allowed direction and
magnitude, the acceptance criterion, the population, the exclusions, and the update mode. Deliverable: a one-page
modification specification. Acceptance criteria: a reviewer can tell whether a proposed change is inside or outside; the
criterion is measurable; the exclusions are explicit.

### Exercise 4 — Modification protocol (design, 50 minutes)

Write the four parts of a modification protocol for the change in Exercise 3: data management, retraining, performance
evaluation, and update procedures. Deliverable: a protocol outline with owners and records. Acceptance criteria: data
sources, inclusion criteria, test-set independence, statistical method, acceptance criteria, rollback, and labeling
updates are addressed.

### Exercise 5 — Traceability table (application, 30 minutes)

Build a traceability table for three modifications. Link each to its data, method, acceptance criterion, impact
assessment, and monitoring metric. Deliverable: the table. Acceptance criteria: no blank cells; each row can be audited;
dependencies are recorded.

### Exercise 6 — Impact assessment (critique, 45 minutes)

Review a draft impact assessment that says "the change is expected to improve performance and risks will be monitored."
Rewrite it as an individual and cumulative assessment for two modifications. Deliverable: a revised assessment with
benefits, risks, subgroups, interactions, and residual risk. Acceptance criteria: it distinguishes impact from risk; it
addresses bias; it states a conclusion and confidence.

### Exercise 7 — QMS integration (design, 40 minutes)

Map a PCCP modification to a design-change record. Identify the approval, risk-file update, verification and validation,
labeling review, and post-market data feed. Deliverable: a process map with records. Acceptance criteria: the map shows
who approves, what evidence is created, and how the change is traced.

### Exercise 8 — Adversarial review (critique, 40 minutes)

Exchange a PCCP draft with a peer. Find one modification that is unbounded, one acceptance criterion that is
unmeasurable, and one impact that is not assessed. Write three questions a reviewer would ask. Deliverable: three
questions and a one-paragraph assessment. Acceptance criteria: questions are specific; at least one identifies a safety
issue, not only a documentation gap.

## 13. Self-check answers

**Question 1. What is the difference between GMLP and a PCCP?** GMLP is a set of lifecycle principles for developing and
monitoring machine learning devices. It is guidance, not an authorization. A PCCP is a reviewed plan that authorizes
specific future modifications and the methods for validating them. GMLP describes how to work; a PCCP describes what may
change and how the change will be justified.

**Question 2. Is a PCCP a substitute for QMS change control?** No. The PCCP defines the authorized scope and evidence.
The QMS defines who approves the change, what records are created, and how the change is implemented. An authorized
change still needs a design-change record, a risk assessment, verification and validation, and labeling review.

**Question 3. When does a change require a new submission rather than a PCCP?** When it changes the intended use, the
patient population, the clinical action, or the device's fundamental technology in a way the PCCP does not cover. A
change that cannot be specified in advance or has no measurable boundary is also not eligible. The safest test is
whether the change alters the claim.

**Question 4. What makes a modification description bounded?** A measurable baseline, an allowed direction and
magnitude, a population and subgroup constraint, an acceptance criterion, an update mode, and explicit exclusions. A
reviewer should be able to classify a proposed change as inside or outside without asking the sponsor.

**Question 5. What are the four parts of a modification protocol?** Data management practices, retraining practices,
performance evaluation, and update procedures. The protocol should also include a traceability table linking each
modification to its methods and criteria.

**Question 6. Why is a traceability table important?** It shows that every modification has a method, every method has a
criterion, and every criterion has monitoring. It also lets the quality system verify that a deployed change is covered.
Without it, the PCCP is a collection of intentions.

**Question 7. What is the difference between individual and cumulative impact?** Individual impact assesses one
modification against the unmodified device. Cumulative impact assesses combinations, sequencing, and repeated updates
over time. A series of small changes can cross a safety boundary even when each change is acceptable alone.

**Question 8. How does a PCCP integrate with post-market surveillance?** The PCCP specifies the metrics, data sources,
frequency, and actions for monitoring authorized changes. The post-market surveillance plan incorporates those
requirements and feeds the results back into the change-control process. A monitoring failure can trigger a deviation, a
corrective action, or a new submission.

**Question 9. What are two signs that a PCCP is too broad?** The description uses vague goals such as "improve accuracy"
without numeric bounds, and the protocol relies on future data or methods the sponsor cannot specify. A third sign is an
impact assessment that asserts benefit without analyzing subgroups or cumulative effects.

**Question 10. What would you do if a deployed change falls outside the PCCP?** Stop and assess the change against the
authorized scope. If it is outside, follow the deviation procedure, notify the relevant regulator as required, and
prepare a new submission or a corrective action. Do not retroactively label the change as covered.

## 14. Interview questions

**1. What is a PCCP?** A strong answer defines it as a reviewed plan for specific future modifications, names the three
components, and states that it is authorized within a marketing submission. A weak answer calls it a plan for continuous
learning without boundaries.

**2. Why did FDA create the PCCP pathway?** A strong answer explains the tension between frozen device evidence and the
need to update machine learning models, and cites section 515C and FDORA. A weak answer says it is a way to avoid
submissions.

**3. What changes are eligible for a PCCP?** A strong answer says planned, bounded, verifiable changes within the
intended use, with a protocol and impact assessment. A weak answer says any model update.

**4. How do you bound a modification?** A strong answer gives a baseline, a direction and magnitude, an acceptance
criterion, a population constraint, an update mode, and exclusions. A weak answer says "improve performance by a
reasonable amount."

**5. What is the difference between a PCCP and a new 510(k)?** A strong answer explains that a PCCP is reviewed with a
submission and authorizes specified changes, while a new 510(k) is required for changes outside the PCCP or for a new
claim. A weak answer treats them as interchangeable.

**6. What does an impact assessment contain?** A strong answer covers individual and cumulative benefits, risks,
subgroup and bias effects, validation evidence, and residual risk. A weak answer says it is a risk matrix.

**7. How do you monitor a PCCP after authorization?** A strong answer names metrics, data sources, frequency, acceptance
criteria, rollback, complaint review, and deviation handling. A weak answer says "we monitor performance."

**8. How does the EU treat model updates?** A strong answer says there is no direct PCCP equivalent; changes are handled
through MDR/IVDR change control and conformity assessment, with the AI Act as an additional layer. A weak answer assumes
a US PCCP is valid in the EU.

**9. What is the role of GMLP?** A strong answer describes the ten principles as lifecycle practice and maps them to
artifacts such as data management, independent testing, human-AI evaluation, and monitoring. A weak answer treats GMLP
as a checkbox.

**10. What would you do if a model update fails an acceptance criterion?** A strong answer stops deployment,
investigates the cause, follows the deviation and corrective-action process, and determines whether a new submission is
needed. A weak answer deploys anyway and monitors.

## 15. Key terms

| Term | One-line meaning |
|---|---|
| GMLP | Good Machine Learning Practice, a set of lifecycle principles |
| PCCP | Predetermined Change Control Plan |
| Modification | A specified change to device software, data, inputs, performance, or use |
| Modification protocol | Data, retraining, evaluation, and update methods for a modification |
| Impact assessment | Analysis of individual and cumulative benefits, risks, and bias |
| Locked model | Model with fixed parameters after release |
| Continuous learning | Model that updates automatically or semi-automatically during use |
| Retraining | Re-estimating model parameters with new data |
| Recalibration | Adjusting the probability mapping without changing the architecture |
| Threshold | Operating point that converts a score into an action |
| Data drift | Change in the distribution of inputs or outcomes |
| Concept drift | Change in the input-outcome relationship |
| Traceability table | Mapping from modification to method, criterion, and monitoring |
| Authorized change | Change within the scope of an approved PCCP |
| Deviation | Change outside the authorized PCCP |
| QMS | Quality management system |
| Design change | Controlled change to device design |
| TPLC | Total product life cycle |
| AI-DSF | AI-enabled device software function |
| Substantial equivalence | 510(k) standard for same intended use and similar technology |
| Predicate | Legally marketed comparator device |
| PMA | Premarket approval |
| De Novo | Classification route for a novel low- or moderate-risk device |
| Post-market surveillance | Systematic collection and analysis of post-launch data |
| PMCF | Post-market clinical follow-up |

## 16. Sources

All URLs were accessed on 25 October 2026. Verify current versions before relying on them.

1. US Food and Drug Administration, Health Canada, and UK MHRA. "Good Machine Learning Practice for Medical Device Development: Guiding Principles." 27 October 2021. https://www.fda.gov/media/153486/download
2. US Food and Drug Administration. "Marketing Submission Recommendations for a Predetermined Change Control Plan for Artificial Intelligence-Enabled Device Software Functions." Final guidance, 18 August 2025. https://www.fda.gov/media/166704/download
3. US Food and Drug Administration. "Predetermined Change Control Plans." https://www.fda.gov/medical-devices/premarket-submissions-selecting-and-preparing-correct-submission/predetermined-change-control-plans
4. US Food and Drug Administration. "AI-Enabled Device Software Functions: Lifecycle Management and Marketing Submission Recommendations." Draft guidance, 7 January 2025. https://www.fda.gov/media/184856/download
5. US Food and Drug Administration. "Novel Medical Devices with Predetermined Change Control Plans: Hypothetical Case Examples." https://www.fda.gov/medical-devices/premarket-submissions-selecting-and-preparing-correct-submission/novel-medical-devices-predetermined-change-control-plans-hypothetical-case-examples
6. US Food and Drug Administration. "Irregular Rhythm Notification Feature, K231173." 21 July 2023. https://www.accessdata.fda.gov/cdrh_docs/pdf23/K231173.pdf
7. US Food and Drug Administration. "Quality Management System Regulation (QMSR)." https://www.fda.gov/medical-devices/postmarket-requirements-devices/quality-management-system-regulation-qmsr
8. US Food and Drug Administration. "Artificial Intelligence-Enabled Medical Devices." https://www.fda.gov/medical-devices/digital-health-center-excellence/artificial-intelligence-enabled-medical-devices
9. European Parliament and Council. Regulation (EU) 2017/745 on medical devices (MDR). https://eur-lex.europa.eu/eli/reg/2017/745/oj
10. European Parliament and Council. Regulation (EU) 2017/746 on in vitro diagnostic medical devices (IVDR). https://eur-lex.europa.eu/eli/reg/2017/746/oj
11. European Parliament and Council. Regulation (EU) 2024/1689 on artificial intelligence (AI Act). https://eur-lex.europa.eu/eli/reg/2024/1689/oj
12. European Parliament and Council. Regulation (EU) 2026/1744 amending the AI Act. https://eur-lex.europa.eu/eli/reg/2026/1744/oj/eng
13. Medical Device Coordination Group. MDCG 2025-6, "Guidance on the AI Act and medical devices." https://health.ec.europa.eu/system/files/2025-06/mdcg_2025-6_en.pdf
14. International Organization for Standardization. ISO 13485:2016. https://www.iso.org/standard/59752.html
15. International Organization for Standardization. ISO 14971:2019. https://www.iso.org/standard/72704.html
16. International Electrotechnical Commission. IEC 62304:2006+A1:2015. https://webstore.iec.ch/en/publication/22794
17. International Electrotechnical Commission. IEC 62366-1:2015+A1:2020. https://webstore.iec.ch/en/publication/67220
18. US Food and Drug Administration. "Predetermined Change Control Plans for Machine Learning-Enabled Medical Devices: Guiding Principles." https://www.fda.gov/medical-devices/artificial-intelligence-enabled-medical-devices/predetermined-change-control-plans-machine-learning-enabled-medical-devices-guiding-principles

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`03_fda-pathways.md`](03_fda-pathways.md)
- Next topic: [`05_qms-and-design-controls.md`](05_qms-and-design-controls.md)
- Related topic: [`07_iec-62304-software-lifecycle.md`](07_iec-62304-software-lifecycle.md)
- Survey file: [`../../course/14_medical_devices_samd_regulation.md`](../../course/14_medical_devices_samd_regulation.md)
- Labs: [`../../labs/`](../../labs/)
- Lab 14.5 PCCP draft: [`../../labs/`](../../labs/)

The module index is the authoritative map. If a sibling file is not yet present, the index marks it as planned.
