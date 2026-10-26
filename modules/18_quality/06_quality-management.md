# Quality Management

> Module 18 — Quality, Evaluation, and Reporting
> Topic 06 of 09
> Estimated study time: 3.5 hours
> Prerequisites: 05
> Version 2026.3

## 1. Why this topic exists

A quality management system (QMS) is the set of documented processes that an organisation uses to
ensure that what it builds is what it intended to build, that defects are found and fixed, and
that the evidence of both survives an inspection. It is not a final report, a certificate, or a
folder of standard operating procedures written for an audit. It is the operating system of the
development process.

Analytics teams usually encounter a QMS in one of two ways, both unpleasant. The first is a
regulatory deadline: a product is classified as a medical device, and the team discovers that it
needs design controls, risk management, change control, and a corrective and preventive action
(CAPA) process, with records, within a timeline that does not accommodate learning. The second is
an inspection: a regulator or a notified body asks for the design history file, and the team
produces a repository, a slide deck, and a spreadsheet, none of which is controlled, versioned,
or linked to the claims it supports.

The cost of the absence is not only regulatory. A team without a QMS cannot answer three
questions that every serious buyer asks. Which version of the model was validated, and can you
prove it? What changed between the validated version and the one you are deploying? What happens
when a defect is found after deployment, and how will you know the fix worked? A team with a QMS
answers these from records in minutes. A team without one answers them from memory, and memory
does not survive staff turnover.

The regulatory asymmetry is worth stating plainly. A research model that is never marketed as a
device is outside the FDA's device QMS requirements, and a hospital's internally developed tool
may fall outside them too, depending on how it is used and promoted. The same model, sold to
another hospital with a claim of clinical benefit, can become a device. The classification follows
the intended use and the marketing, not the algorithm, which means a team can become regulated by
a change in a sales conversation. The QMS scope should be reviewed whenever the commercial model
changes, because that is when the obligations change.

There is a third cost, which is subtler. Without a QMS, the same defect recurs. A data leakage
bug is fixed in one model and reintroduced in the next, because there is no process that captures
the cause and changes the procedure. A QMS is, at bottom, an organisational memory with
enforcement. The CAPA process is the part that turns a fixed defect into a changed procedure.

This topic teaches the QMS concepts that an analytics team actually needs, the standards that
constrain a regulated product, and the mechanics of design control, change control, and CAPA. It
does not teach ISO 13485 clause by clause; it teaches how the clauses map onto the work of
building and maintaining a model. It builds on reproducibility: the artifacts that make an
analysis reproducible are the same artifacts that a QMS controls.

The organisational failure has a recognisable shape. The quality function is engaged at the end,
asked to review a product that has already been designed, and given two weeks. It produces a
documentation package that describes what was done, which is the opposite of design control, and
everyone knows the package is retrospective. The next product repeats the pattern because the
engagement model did not change. A QMS that is invited after the design cannot control the design,
and a team that has never seen a design input will not produce one under deadline.

A note on regulatory scope. Most analytics teams are not building a regulated device, and a full
QMS would be disproportionate. The useful question is not "are we regulated?" but "which of these
controls would make our evidence stronger if a customer or a regulator asked?" A team that adopts
design inputs, change control, and a CAPA-like process without the full apparatus gets most of the
benefit at a fraction of the cost, and can scale up if the product becomes a device. The topic distinguishes what is legally required for a device from
what is good practice for any team whose outputs affect patients. Applying the second without the
first is sensible; claiming the first without the evidence is not.

## 2. Learning outcomes

By the end of this topic you can:

1. **Explain** what a QMS is and is not, and distinguish a regulated requirement from a
   professional convention.
2. **Map** the four core processes (design control, risk management, change control, CAPA) onto a
   model development lifecycle.
3. **Write** a design input and a design output for a model, and verify that one satisfies the
   other.
4. **Distinguish** a correction from a corrective action, and a corrective action from a
   preventive action.
5. **Conduct** a root cause analysis that reaches a systemic cause rather than a person.
6. **Define** an effectiveness check with a data source, a timeframe, a sample, and an acceptance
   criterion.
7. **Operate** change control for a model update, including the decision on whether revalidation
   is required.
8. **Prepare** the records that an auditor will ask for, and explain why each exists.

## 3. The core mental model

A QMS is a loop with four processes and a document hierarchy.

```text
          design input  ->  design output  ->  verification  ->  validation
                ^                                                    |
                |                                                    v
          corrective action  <-  CAPA  <-  nonconformity  <-  monitoring
                |
                v
          procedure change  ->  training  ->  next design
```

The loop is the point. A defect enters as a nonconformity, is contained by a correction, is
investigated to a root cause, produces a corrective action, and ends with a change to the
procedure that prevents recurrence. Without the last step, the organisation fixes instances and
not causes.

Beneath the loop is a document hierarchy. Each level has a different owner and a different change
procedure.

```text
  Level 1  Quality manual          policy, scope, interactions        exec owner
  Level 2  Standard operating procedure  who does what, when           process owner
  Level 3  Work instruction        how to perform one task            team lead
  Level 4  Record                  evidence that it happened          performer
```

The hierarchy matters because audits test the links between levels. An auditor asks whether the
record shows that the work instruction was followed, whether the instruction implements the
procedure, and whether the procedure is consistent with the policy. A team with a policy and no
records has nothing to show. A team with records and no procedure has evidence of activity
without control.

Three principles make the system work.

**Say what you do, do what you say, prove it.** The three clauses are the whole of quality
management in one sentence. The first requires a documented process, the second requires
adherence, and the third requires records. Most failures are in the third clause, because teams
document and act but do not record.

**Risk proportional to effect.** Controls should be proportional to the risk of the product. A
research model needs version control and a documented evaluation; an infusion-pump controller
needs much more. The same QMS applied uniformly either over-controls research or under-controls
high-risk products.

**The process owns the person.** A root cause that ends at "the analyst made a mistake" is not a
root cause. The question is why the process allowed the mistake to reach a release. This principle
is uncomfortable and it is the one that separates a functioning CAPA system from a blame system.

The three clauses are sequential and each fails differently. A team that does not say what it does
has no baseline, so any process is acceptable and none is repeatable. A team that says what it does
but does not do it has a documentation-reality gap, which is worse than no documentation because it
misleads the reader. A team that does what it says but cannot prove it has activity without
evidence, which is the state of most competent analytics teams before a QMS is introduced. The
first fix is writing, the second is adherence, and the third is recording; they are usually needed
in that order, and the third is the one that takes longest to change.

A fourth principle, often omitted, is that the system must be usable under time pressure. A
procedure that takes an hour to follow for a five-minute task will be skipped, and the skip will
not be recorded. The design of the QMS should minimise the number of steps that require a decision
by a committee, and it should make the record the natural output of the work rather than a separate
form. Teams comply with systems that fit their work; they route around systems that do not.

A fifth property is that the system must produce evidence at the time of the work, not after. A
record created retrospectively is weaker than a record created contemporaneously, and an auditor
can usually tell the difference from timestamps, version histories, and the consistency of details.
This is why the QMS emphasises records as a byproduct of the process rather than a deliverable at
the end. A team that runs the process and generates the record as it goes has evidence; a team that
runs the process and writes the record before an audit has a reconstruction.

A fourth model is the distinction between the quality system and the evidence chain. The evidence
chain (topic 01) describes what a claim rests on. The QMS describes the process that produced the
chain and the records that prove it. A claim can have a complete chain and no QMS, and it can have
a QMS and a broken chain. The first is a good study by an informal team; the second is a
well-documented failure. The goal is both.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Standard or unit |
|---|---|---|---|
| Quality management system | Documented processes for consistent product quality | A folder of SOPs for an audit | ISO 13485 |
| ISO 13485 | International standard for medical device QMS | A certificate of product quality | ISO 13485:2016 |
| QMSR | FDA Quality Management System Regulation, incorporating ISO 13485 | The old QSR | 21 CFR 820, effective 2026 |
| Design control | Process linking user needs to design outputs and verification | A design document | 21 CFR 820.30 |
| Design input | Requirement the design must satisfy | A wish list | Requirement record |
| Design output | Specification or artifact produced by the design | A slide deck | Specification |
| Verification | Confirmation that outputs meet inputs | Confirmation that the product works in use | Objective evidence |
| Validation | Confirmation that the product meets user needs in use | A held-out test set | Clinical or user evidence |
| Risk management | Systematic process for identifying and controlling risks | A risk register spreadsheet | ISO 14971 |
| Hazard | Potential source of harm | A known bug | Risk file |
| Harm | Injury or damage to health | A model error | Risk file |
| CAPA | Corrective and preventive action | Any bug fix | 21 CFR 820.100 |
| Correction | Action to eliminate a detected nonconformity | CAPA | Record |
| Corrective action | Action to eliminate the cause of a nonconformity | Correction | Record |
| Preventive action | Action to prevent a potential nonconformity | Corrective action | Record |
| Nonconformity | Failure to meet a requirement | A complaint | Record |
| Root cause | The fundamental reason a nonconformity occurred | The person who made the error | Analysis |
| Effectiveness check | Evidence that an action removed the cause | A statement that it was fixed | Record |
| Change control | Process for assessing and approving changes | A pull request | Procedure |
| Standard operating procedure | Documented process with responsibilities | A wiki page | Controlled document |
| Work instruction | Task-level how-to | An SOP | Controlled document |
| Record | Evidence that an activity occurred | A log file | Retained record |
| Document control | Versioning, approval, and distribution of documents | Storing files in a shared drive | Procedure |
| Design history file | Record of the design and its controls | The code repository | DHF |
| Management review | Scheduled leadership review of the QMS | A team meeting | Record |
| Internal audit | Independent check of process conformity | A self-assessment | Audit report |
| Supplier control | Management of external providers | A contract | Procedure |
| Traceability | Ability to link requirements to evidence | A list of files | Matrix |

Four distinctions carry the topic, and each is a common source of audit findings.

**Correction versus corrective action.** A correction fixes the instance: the bad record is
reprocessed, the alert is suppressed, the model is rolled back. A corrective action fixes the
cause: the validation script now checks for patient overlap, so the defect cannot recur. A team
that reports a correction as a CAPA has closed the ticket and not the problem.

**Corrective versus preventive.** A corrective action responds to a nonconformity that occurred. A
preventive action responds to a potential nonconformity that has not occurred. The distinction is
often collapsed, and the Global Harmonization Task Force guidance notes that the acronym CAPA is
itself misleading because preventive action is not required for every corrective action. In
practice, a corrective action usually reveals a preventive opportunity, and a good CAPA record
notes both.

**Verification versus validation.** Verification asks whether the design output meets the design
input: the model's sensitivity meets the specified requirement. Validation asks whether the
product meets the user's need in the intended use: clinicians using the model make better
decisions. A model can pass verification and fail validation, and the two are frequently
conflated in analytics because both are called "testing."

**Nonconformity versus defect versus complaint.** A nonconformity is a failure to meet a
requirement, whether or not anyone notices. A defect is a nonconformity in a product. A complaint
is a customer's report, which may or may not correspond to a nonconformity. The three are recorded
in different systems and converge only if the QMS links them. A team that tracks complaints but not
nonconformities will miss the defects that no one reported, which are usually the majority.

**Quality system versus quality assurance.** Quality assurance is the activity of checking;
quality management is the system that makes checking possible and acts on the result. A team can
do quality assurance without a QMS (testing, review), and it can have a QMS without effective
quality assurance (documents, no checks). The topic is about the system, not the checking.

## 5. The mechanics, step by step

This procedure builds and operates a proportionate QMS for an analytics team. It assumes a
product that may or may not be a regulated device, and it marks the steps that become mandatory
when the product is regulated. It takes four to twelve weeks to establish and then runs
continuously.

### Step 0. Appoint an owner with authority

**Input:** the organisation chart and the product portfolio.

**Output:** a named quality owner with the authority to stop a release, and a defined escalation
path.

**Rationale:** A QMS without an owner becomes a documentation project. The owner needs the
authority to hold a release when a control has not been met, because otherwise every gate is
advisory. The authority can be delegated, but it cannot be informal.

**Branch point:** if no one has stop authority, the QMS cannot function. Record the gap and
escalate; do not write procedures that assume an authority the organisation has not granted.

### Step 1. Define the scope and the regulatory status

**Input:** the product description, the intended use, and the jurisdiction.

**Output:** a scope statement listing the products and processes covered, and a regulatory
determination: device or not, class, and pathway.

**Rationale:** Scope determines which requirements apply. A QMS that covers everything at the
highest level is unusable; one that covers nothing is a fiction. The regulatory determination
should be recorded with its reasoning, because it is the first thing an auditor or a buyer will
ask about.

**Branch point:** three outcomes.

| Status | Requirement | QMS depth |
|---|---|---|
| Regulated device | Full QMS per ISO 13485 and the applicable regulation | Design controls, risk file, CAPA, audits |
| Software as a medical device, low risk | QMS with proportionate design controls | Design controls and change control |
| Not a device (research or operational tool) | No legal QMS requirement | Version control, documented evaluation, CAPA-like process |

**Edge case, incomplete reporting:** if the regulatory status is genuinely ambiguous, record the
ambiguity and the interim controls. Do not resolve it by assuming the product is not a device.

### Step 2. Map the processes

**Input:** the product lifecycle from idea to retirement.

**Output:** a process map with inputs, outputs, owners, and records for each stage.

**Rationale:** The map is the skeleton of the QMS. It identifies where design control applies,
where risk management feeds in, where change control triggers, and where monitoring produces
nonconformities. Without it, the procedures are disconnected documents.

**Branch point:** if the map has a stage with no owner, that stage will fail at the first audit.
Assign an owner before writing procedures.

### Step 3. Write the procedures that the team will actually follow

**Input:** the process map and the team's actual practice.

**Output:** a small set of standard operating procedures, each with purpose, scope,
responsibilities, steps, records, and revision history.

**Rationale:** A procedure that describes an aspirational process will be violated daily, and
every violation is a finding. The procedure should describe what the team does, improved where
necessary, so that adherence is possible. The number of procedures should be small; five good ones
beat thirty unused ones.

**Branch point:** if a process is not repeatable enough to describe, do not write a procedure that
pretends it is. Write a work instruction for the parts that are repeatable and note the rest as
under development.

### Step 3a. Put the documents under control

**Input:** the procedures and the templates.

**Output:** a document register with identifier, title, version, owner, approval, effective date,
and distribution; and a change history for each document.

**Rationale:** Document control is the mechanism that makes a procedure enforceable. Without it,
two versions coexist, the team follows the one it remembers, and an audit finds that the record
does not match the controlled document. The register is small and the discipline is simple, but it
is the most commonly skipped step.

**Branch point:** three document states, each with a different rule.

| State | Meaning | Permitted use |
|---|---|---|
| Draft | Not approved | Must not be used for regulated work |
| Effective | Approved and trained | The only version in use |
| Obsolete | Superseded | Retained for records; withdrawn from use |

The obsolete category is the one teams forget. A superseded checklist that remains on a shared
drive will be used by someone, and the resulting record will be untraceable. Withdraw it and keep
an archived copy.

### Step 4. Establish design inputs

**Input:** user needs, intended use, regulatory requirements, and performance expectations.

**Output:** a numbered, testable set of design inputs, each with a source and an acceptance
criterion.

**Rationale:** Design inputs are the requirements against which everything later is verified. An
input that cannot be tested ("the model should be accurate") is not an input. A well-formed input
names the population, the metric, the threshold, and the conditions ("sensitivity at least 0.85
for referable disease in adults aged 40 to 75, at the operating point specified in the IFU").

**Branch point:** if a requirement comes from a regulator, cite the clause. If it comes from a
user, record the user and the date. Untraceable inputs cannot be verified.

**Edge case, selective outcome reporting:** if the design inputs name several metrics, the
verification must address all of them. Verifying the favourable one and ignoring the rest is a
nonconformity in the design file, not a reporting choice.

### Step 5. Produce design outputs and verify them

**Input:** the design inputs and the development work.

**Output:** design outputs (specifications, model, software, labelling) and a verification record
linking each output to its input.

**Rationale:** Verification is the evidence that the output satisfies the input. It is not the
same as testing everything; it is testing the specific requirement. The verification matrix is the
artifact that makes the link visible.

**Branch point:** if an input cannot be verified with the available evidence, record it as an open
item and do not release. An unverified input is a known gap, and releasing with a known gap
requires a risk acceptance.

### Step 6. Validate in the intended use

**Input:** the verified product and the clinical or operational context.

**Output:** validation evidence that the product meets user needs in the intended use.

**Rationale:** Validation is where the product meets the world. For a model, it is the clinical or
workflow evaluation, not the held-out test set. The distinction matters because a model can meet
its sensitivity specification in a retrospective sample and fail in the workflow.

**Branch point:** if validation is not feasible before release, the release must be labelled as a
limited or pilot release with explicit monitoring, not as a full release.

### Step 7. Build the risk management file

**Input:** the intended use, the hazard analysis, and the validation evidence.

**Output:** a risk file with hazards, hazardous situations, harms, risk estimates, controls, and
residual risk, following ISO 14971.

**Rationale:** Risk management is not a one-time document; it is a process that runs across the
lifecycle and feeds design inputs. A risk file that lists only technical failures misses use
errors, automation bias, and inequitable performance, which are the dominant risks for clinical AI.

**Branch point:** if a risk cannot be controlled to an acceptable level, the product should not be
released, regardless of its performance. Record the decision and the rationale.

### Step 8. Institute change control

**Input:** any proposed change to the model, data, software, labelling, or process.

**Output:** a change record with impact assessment, approval, verification, and a decision on
revalidation.

**Rationale:** Change is where validated products become unvalidated ones. A model retrained on new
data is a new product for regulatory purposes unless the change is within a predetermined change
control plan. The impact assessment must consider the model, the data, the workflow, and the
labelling.

**Branch point:** four outcomes, and the fourth requires a regulatory judgement.

| Change | Impact | Action |
|---|---|---|
| Bug fix with no behaviour change | Low | Verify and record |
| Threshold change | Medium | Re-evaluate operating characteristics |
| Retraining on new data | High | Revalidate; check PCCP scope |
| New intended use | Very high | New regulatory submission |

**The change record.** Every change produces a record with the same fields, so that two reviewers
assess the same things.

| Field | Content | Why it matters |
|---|---|---|
| Identifier | Change number and date | Traceability |
| Description | What changes and why | Scope |
| Impact | Model, data, workflow, labelling, risk file | Completeness |
| Verification | Tests performed and results | Evidence |
| Revalidation decision | Required or not, with rationale | Regulatory defensibility |
| Approval | Named approver and date | Authority |
| Rollback plan | How to revert | Containment |

The revalidation decision is the field that auditors read first, because it is where the
organisation's judgement about risk is visible. "Not required because the change is a bug fix
with no behaviour change, verified by identical predictions on a fixed test set" is a defensible
entry. "Not required" with no rationale is not.

### Step 9. Operate the CAPA process

**Input:** nonconformities from monitoring, complaints, audits, or internal review.

**Output:** a CAPA record with containment, root cause, action, verification, effectiveness check,
and closure.

**Rationale:** CAPA is the mechanism by which the system learns. A CAPA without a root cause is a
ticket; a CAPA without an effectiveness check is an assertion. The record must show the chain from
the defect to the changed procedure.

**Branch point:** three outcomes at intake.

| Finding | Action | Record |
|---|---|---|
| Isolated, cause known, low risk | Correction only | Correction record |
| Recurring or systemic | CAPA with root cause analysis | CAPA record |
| Potential issue, no occurrence | Preventive action | Preventive action record |

**Edge case, post-hoc changes:** if the nonconformity arose from an undocumented change, the CAPA
must address the change control failure, not only the technical defect. The technical fix without
the process fix leaves the cause in place.

### Step 10. Define effectiveness checks

**Input:** the corrective action and the data sources.

**Output:** a check with a metric, a baseline, a target, a sample, a timeframe, and an owner.

**Rationale:** Effectiveness is an empirical claim and must be tested empirically. "The training
was delivered" is an output, not an effectiveness check. "The defect rate in the next 200
releases is below 1 percent, measured from the release log over 90 days" is a check.

**Branch point:** if the check cannot be defined, the action is not specific enough to be
corrective. Rewrite the action until the check is possible.

### Step 11. Train, record, and review

**Input:** the procedures, the training plan, and the records.

**Output:** training records per person, a document control log, and a management review record.

**Rationale:** A procedure that no one has been trained on is not in force. Training records are
the evidence that the procedure was communicated, and management review is the evidence that
leadership examined the system. Both are commonly missing and both are easy to produce if the
process is real.

**Branch point:** if a role has no training record, the person cannot be shown competent for the
task. Record the gap and schedule the training; do not backdate.

### Step 12. Audit the system and act on findings

**Input:** the audit plan and the process records.

**Output:** an internal audit report with findings, and a CAPA for each systemic finding.

**Rationale:** The internal audit is the system's check on itself. It should sample records
against procedures, not interview people about their intentions. Findings feed the CAPA process,
which closes the loop.

**Branch point:** if the audit finds no findings, question the audit. A mature system usually has
minor findings; a clean audit often means the sampling was too narrow.

### Edge cases

**Incomplete reporting.** A missing record is a nonconformity in the QMS even if the work was
done. The remedy is to reconstruct the record contemporaneously where possible, and to record the
gap where not.

**Selective outcome reporting.** If a verification report addresses only the favourable metrics,
the design file is incomplete. The CAPA should address the verification procedure, not only the
report.

**Subgroup claims.** A design input that requires subgroup performance needs a verification record
by subgroup. A claim without the record is an unverified input.

**Protocol deviations.** A deviation from a validation protocol is a nonconformity with a cause.
The CAPA should determine whether the protocol was wrong or the execution was.

**Post-hoc changes.** A change made without a change record is an unapproved change. The CAPA
must address the change control breach, and the change must be retrospectively assessed.

**Conflicting reference standards.** If the design input names a reference standard and the
verification used another, the verification does not satisfy the input. Record the mismatch and
re-verify.

**Retracted studies.** If a design input cites a study that has been retracted, the input's basis
is invalid. Reassess the requirement and record the reassessment.

A note on the edge cases. Each one is a case where the record and the work diverge. The QMS does
not require that the work be perfect; it requires that the divergence be visible and addressed.
A missing record that is reconstructed with a note explaining the reconstruction is a finding. A
missing record that is reconstructed to look contemporaneous is a integrity problem, and it is the
one failure that can turn a quality issue into a regulatory one.

### Decision table for CAPA intake

| Signal | Recurrence | Risk | Route |
|---|---|---|---|
| Single complaint, no trend | None | Low | Correction and monitoring |
| Repeat complaint, same cause | Yes | Medium | CAPA |
| Audit finding, systemic | Yes | Medium to high | CAPA |
| Safety event | Any | High | CAPA and regulatory report |
| Potential issue, no event | None | Any | Preventive action |

## 6. Worked example

The case is a CAPA investigation at a company that sells a sepsis alert to hospitals. The defect:
after a routine model update, the alert rate at one customer doubled within two weeks. The
customer reported alert fatigue. The company's monitoring dashboard showed the change, and the
quality manager opened a CAPA.

This is a composite case assembled from common patterns. The numbers are constructed so that the
effectiveness arithmetic can be checked; the sequence of events reflects how these investigations
typically unfold.

### Intake and containment

The nonconformity record: "Alert rate increased from 6.1 percent to 12.4 percent of admissions
between 3 March and 17 March 2026, following release 4.2 on 1 March 2026." The customer reported
that clinicians were overriding alerts without review.

Containment, within 48 hours: the customer was advised to revert to release 4.1, and the vendor
disabled the automatic rollout to the remaining 14 customers. The containment is a correction, not
a corrective action: it stopped the harm without addressing the cause.

Risk assessment at intake: patient harm possible if clinicians begin ignoring alerts (alert
fatigue), severity high, probability medium. The CAPA is classified as major.

### Root cause analysis

The team used a fault tree followed by a five-whys interrogation of the most likely branch.

The immediate cause: the alert threshold in release 4.2 was 0.31, against 0.38 in release 4.1.

Why was the threshold lower? The retraining pipeline recalibrated the model and selected the
threshold that maximised F1 on the internal validation set.

Why did it select a threshold automatically? The pipeline had a threshold-optimisation step added
in release 4.0 for a different product, and it was reused.

Why was it reused without review? The release checklist did not include a threshold review for
products whose labelling specifies an operating point.

Why did the checklist not include it? The checklist was written for the first product, which had
no specified operating point, and was never revised when the second product was added.

Why was it never revised? There was no process owner for the checklist, and no trigger to review
it when a new product was onboarded.

Root cause: the release checklist was not under document control and had no owner or review
trigger, so a product-specific requirement was not represented in the release process. The
threshold optimisation was a proximate cause; the systemic cause was the uncontrolled checklist.

This is the point at which a weak investigation stops. "The analyst reused a step" is a person,
not a cause. The strong investigation reaches the process that allowed the reuse to pass review.

### Correction, corrective action, and preventive action

The correction: revert the threshold to 0.38 in release 4.2.1 and re-release to the affected
customer.

The corrective action: bring the release checklist under document control, assign an owner,
add a mandatory threshold-review item for any product with a labelled operating point, and require
sign-off by the regulatory lead for threshold changes.

The preventive action: add an onboarding gate for new products that requires the checklist to be
reviewed against the product's labelling, and add the checklist to the internal audit scope.

### Verification and effectiveness check

Verification of the action: the revised checklist was approved, versioned, and used for release
4.2.1, with a completed threshold-review item in the release record.

Effectiveness check, defined before closure:

- Metric: alert rate as a percentage of admissions, per customer, weekly.
- Baseline: 12.4 percent during the defect period; 6.1 percent before release 4.2.
- Target: alert rate within 5.5 to 7.0 percent for four consecutive weeks.
- Sample: all 15 customers, weekly for 90 days.
- Data source: the monitoring dashboard, which records alert counts and admissions.
- Owner: quality manager, with the clinical lead.
- Acceptance: no customer outside the target range for four consecutive weeks, and no alert-fatigue
  complaint.

### Scoring the CAPA

A CAPA can be scored on five domains, each 0 to 2, using the same discipline as the appraisals in
topics 02 and 03.

| Domain | Question | Score | Evidence |
|---|---|---:|---|
| Containment | Was the harm stopped promptly? | 2 | Revert within 48 hours; rollout paused |
| Root cause | Did the analysis reach a systemic cause? | 2 | Uncontrolled checklist with no owner |
| Action | Does the action address the cause, not the instance? | 2 | Checklist controlled, owner assigned, item added |
| Effectiveness | Is the check empirical and pre-specified? | 2 | Metric, baseline, target, sample, timeframe |
| Documentation | Is the record complete and traceable? | 1 | Root cause and action complete; training record pending |

Total: 9 of 10. The one partial is documentation, because the training record for the revised
checklist was not yet filed at the time of scoring.

### Recommendation

Close the CAPA only after the effectiveness check passes and the training record is filed. The
interim recommendation is "effective action implemented, effectiveness pending". The customer
communication should state the correction, the cause, and the check, without disclosing other
customers' data.

The recommendation to the quality system is broader: the same pattern (a checklist reused across
products without a review trigger) may exist in other release processes. The CAPA should trigger a
horizontal scan, because a systemic cause found in one product usually applies to others. That
scan is the difference between fixing a defect and improving the system.

### What would make this a failure

Three findings would change the case from a well-run CAPA to a failing one, and each is common.

If the containment had taken two weeks instead of 48 hours, the harm would have continued and the
containment domain would score 0. The speed of containment is the part of a CAPA that most directly
affects patients, and it is the part most often delayed by indecision about whether the report is
real.

If the root cause had been recorded as "pipeline error" without the checklist analysis, the action
would have been to fix the pipeline, and the next product would have repeated the defect. The root
cause domain would score 1 and the action domain 1, because the action would not reach the process.

If the effectiveness check had been closed on the basis of the vendor's own dashboard without the
customer's alert data, the check would have measured the vendor's view rather than the clinical
workload. The effectiveness domain would score 1, because the metric would not reflect the harm
the CAPA was meant to prevent.

### The customer communication

The customer communication is part of the CAPA record, not a separate marketing task. It has four
elements: what happened (alert rate doubled after release 4.2), what was done (reverted to 4.1,
then 4.2.1 with the corrected threshold), what the cause was (an uncontrolled release checklist
allowed a product-specific threshold requirement to be missed), and what will prevent recurrence
(checklist under document control with a mandatory threshold review).

The fourth element is the one customers and regulators value most, and it is the one vendors omit
most often. A communication that stops at "we have fixed it" answers the incident but not the
confidence question. The QMS requires the fourth element because a CAPA without a process change
is a correction.

### The horizontal scan

A systemic cause found in one product usually applies to others. The uncontrolled checklist was
used for every product in the portfolio, so the same defect could exist wherever a labelled
operating point was not represented in the release process. The horizontal scan asked three
questions of the other 14 customers and the four other products: does the release checklist cover
this product's labelling, has any threshold changed without review, and does the monitoring
dashboard alert on alert-rate change.

The scan found one further product with a threshold change that had been reviewed informally but
not recorded. That product was added to the CAPA scope. The finding is instructive: the scan cost
two days and found a second instance, which is the usual ratio. A CAPA that does not scan is a
CAPA that fixes one of several instances.

### Two diagnostics that would change the verdict

**Diagnostic 1: the root cause analysis stopped at the analyst.** If the investigation had
concluded that the analyst reused the threshold step without authorisation, the root cause domain
would score 0 and the action domain 1, because the action would be retraining the analyst. The
total would fall to 6 of 10, the recommendation would be "not effective", and the defect would
recur when a different analyst reused the step. This is the most common failure of CAPA in
practice, and it is why the "process owns the person" principle is stated in section 3.

**Diagnostic 2: the effectiveness check were defined as "no complaints for 30 days".** The check
would be an absence of reports rather than a measurement, and it would pass even if the alert rate
stayed high and clinicians stopped reporting. The effectiveness domain would score 0, the total
would fall to 7 of 10, and the CAPA would be reopened at the next monitoring review. An
effectiveness check that measures silence is not a check.

### Vary one assumption: the defect was found before release

Suppose the threshold review had caught the change in release testing, before any customer
received 4.2. The event would be a nonconformity in verification, not a post-market defect. The
containment domain would be not applicable, the root cause would be the same, and the corrective
action would be the same. The CAPA would still be required, but the risk classification would be
lower, the regulatory reporting question would disappear, and the effectiveness check would use
release records rather than customer monitoring.

The interesting change is in the score. The CAPA would score 8 of 10 with containment excluded and
documentation complete, and the recommendation would be "effective, close after training". The
lesson is that the same systemic cause produces a different CAPA depending on where the defect is
detected, and the process fix should be the same in both cases. A quality system that only fixes
the process when a customer is harmed is a system that waits for harm.

## 7. Code, query, or calculation

The script evaluates a CAPA effectiveness check with an exact test, computes CAPA aging, and
applies the intake decision rule. It uses only the standard library.

```python
"""CAPA effectiveness and aging analysis.

Fisher's exact test is computed directly from the hypergeometric distribution,
so no external statistics package is required.
"""

from math import comb
from datetime import date


def fisher_exact(a, b, c, d):
    """Two-sided Fisher exact test for [[a, b], [c, d]]."""
    n = a + b + c + d
    row1, col1 = a + b, a + c
    def prob(x):
        return comb(row1, x) * comb(n - row1, col1 - x) / comb(n, col1)
    observed = prob(a)
    total = 0.0
    lo = max(0, col1 - (n - row1))
    hi = min(row1, col1)
    for x in range(lo, hi + 1):
        p = prob(x)
        if p <= observed + 1e-12:
            total += p
    return total


# Effectiveness check: releases before and after the CAPA.
# Defect = release required a hotfix within 30 days.
before_defect, before_ok = 11, 89      # 100 releases before
after_defect, after_ok = 2, 118        # 120 releases after

p = fisher_exact(before_defect, before_ok, after_defect, after_ok)
rate_before = before_defect / (before_defect + before_ok)
rate_after = after_defect / (after_defect + after_ok)

print("CAPA effectiveness check")
print(f"  before: {before_defect}/100 = {rate_before:.1%}")
print(f"  after:  {after_defect}/120 = {rate_after:.1%}")
print(f"  absolute reduction: {rate_before - rate_after:.1%}")
print(f"  Fisher exact two-sided p: {p:.4f}")
print(f"  target met (<1% after): {rate_after < 0.01}")

# CAPA aging.
capas = [
    ("CAPA-001", date(2025, 11, 3), date(2026, 1, 12), "closed"),
    ("CAPA-002", date(2025, 12, 1), None, "open"),
    ("CAPA-003", date(2026, 1, 20), date(2026, 3, 2), "closed"),
    ("CAPA-004", date(2026, 2, 14), None, "open"),
    ("CAPA-005", date(2025, 9, 8), date(2026, 2, 27), "closed"),
]
as_of = date(2026, 3, 31)
print()
print(f"{'id':<10}{'opened':<12}{'closed':<12}{'days':>6}  status")
print("-" * 50)
ages = []
for cid, opened, closed, status in capas:
    end = closed or as_of
    days = (end - opened).days
    ages.append(days)
    print(f"{cid:<10}{opened.isoformat():<12}{(closed.isoformat() if closed else '-'):<12}{days:>6}  {status}")

open_ages = [(as_of - o).days for _, o, c, s in capas if c is None]
print()
print(f"mean age, all: {sum(ages)/len(ages):.1f} days")
print(f"oldest open: {max(open_ages)} days")
print(f"over 90 days: {sum(1 for a in ages if a > 90)} of {len(ages)}")

# Intake decision rule, applied to two signals.
def route(recurrence, risk, safety_event=False):
    if safety_event:
        return "CAPA + regulatory report"
    if recurrence and risk in ("medium", "high"):
        return "CAPA"
    if risk == "high":
        return "CAPA"
    return "correction + monitoring"


print()
for signal in [("single complaint", False, "low", False),
               ("repeat complaint", True, "medium", False),
               ("safety event", True, "high", True)]:
    name, rec, risk, sev = signal
    print(f"  {name:<18} -> {route(rec, risk, sev)}")
```

Expected output:

```text
CAPA effectiveness check
  before: 11/100 = 11.0%
  after:  2/120 = 1.7%
  absolute reduction: 9.3%
  Fisher exact two-sided p: 0.0039
  target met (<1% after): False

id        opened      closed        days  status
--------------------------------------------------
CAPA-001  2025-11-03  2026-01-12      70  closed
CAPA-002  2025-12-01  -              120  open
CAPA-003  2026-01-20  2026-03-02      41  closed
CAPA-004  2026-02-14  -               45  open
CAPA-005  2025-09-08  2026-02-27     172  closed

mean age, all: 89.6 days
oldest open: 120 days
over 90 days: 2 of 5

  single complaint   -> correction + monitoring
  repeat complaint   -> CAPA
  safety event       -> CAPA + regulatory report
```

Reading the output. The defect rate fell from 11.0 percent to 1.7 percent, a reduction that the
exact test places at p = 0.0039. The action is effective in the statistical sense. The
pre-specified target of below 1 percent was not met, so the CAPA cannot be closed on the target as
written; the correct decision is either to extend the observation period or to revise the target
with a documented rationale. Revising the target after seeing the data is exactly the post-hoc
move that the module warns against, so the extension is the defensible option.

The aging table shows the second problem. Two CAPAs are over 90 days, and the mean age is 89.6
days. An aging distribution with a long tail is a leading indicator of a CAPA system that opens
investigations faster than it closes them, which usually means root cause analysis is underresourced
or actions are not specific enough to verify.

Two limits of the script should be stated. Fisher's exact test assumes independent releases; if
releases cluster by product or by team, the observations are not independent and the p-value is
optimistic. The ageing table treats all CAPAs as comparable; a safety-related CAPA and a
documentation CAPA have different acceptable durations, and the aggregate mean hides that. A real
quality review stratifies by risk class before drawing conclusions from the age distribution.

The intake rule is deliberately simple. It routes a single low-risk complaint to correction and
monitoring, a repeat complaint to CAPA, and a safety event to CAPA plus a regulatory report. The
rule is not a substitute for judgement; it is a consistency device so that two quality managers
route similar signals the same way.

## 8. Data, standards, and regulatory context

**ISO 13485:2016.** The international standard for medical device quality management systems. It
requires a documented QMS with a defined scope, document and record control, management
responsibility, resource management, product realisation including design and development, and
measurement, analysis, and improvement. It requires separate documented procedures for corrective
action (clause 8.5.2) and preventive action (clause 8.5.3), actions proportionate to the effects of
the nonconformity, and an explicit check that actions do not adversely affect the device.

**FDA QMSR.** The Quality Management System Regulation, 21 CFR Part 820, was published on 2
February 2024 and became effective on 2 February 2026. It incorporates ISO 13485:2016 by reference
and Clause 3 of ISO 9000:2015 for terms and definitions. It retains design controls for Class II,
Class III, and certain Class I devices, and it requires traceability for life-supporting or
life-sustaining devices. The FDA has stated that it will not accept ISO 13485 certification in
lieu of inspection, and the old quality system inspection technique was retired on the effective
date. The regulation is binding for devices marketed in the United States.

**Clause map for an analytics product.** The standard is easier to use when its clauses are mapped
to artifacts the team already produces.

| ISO 13485 clause | Artifact for a model | Owner |
|---|---|---|
| 4.2 Documentation | Document register, versioned procedures | Quality |
| 5.6 Management review | Quarterly review minutes with actions | Leadership |
| 6.2 Competence | Training matrix per role | Team lead |
| 7.1 Product realisation | Process map | Product |
| 7.3 Design and development | Design inputs, outputs, verification, validation | Product and clinical |
| 7.4 Purchasing | Supplier agreements for cloud and data | Procurement |
| 7.5 Production and service | Release procedure, change control | Engineering |
| 8.2 Monitoring and feedback | Post-market metrics, complaints | Clinical |
| 8.4 Data analysis | Trend reports on performance and CAPA | Quality |
| 8.5 Improvement | CAPA records and effectiveness checks | Quality |

The map is not exhaustive, and it is not a substitute for reading the standard. It is a translation
device so that a team can see which of its existing artifacts satisfies which requirement, and
where the gaps are.

**ISO 14971:2019.** The standard for application of risk management to medical devices. It defines
the process from risk analysis through evaluation and control to the review of residual risk, and
it requires a risk management file. For AI, the hazard analysis must include use-related hazards
such as automation bias and alert fatigue, not only technical failure modes.

**IEC 62304.** The standard for medical device software lifecycle processes. It defines software
development planning, requirements analysis, architectural design, detailed design, unit
implementation, integration, system testing, and release, with activities scaled by software
safety class. An analytics pipeline that produces a device component falls within its scope.

**FDA Good Machine Learning Practice.** The ten guiding principles published jointly by the FDA,
Health Canada, and the UK MHRA in October 2021 include multidisciplinary expertise, good software
engineering and security practices, representative datasets, independence of training and test
sets, reference datasets based on best available methods, model design tailored to intended use,
focus on the performance of the human-AI team, testing in clinically relevant conditions, clear
user information, and post-deployment monitoring with retraining risk management. They are
non-binding principles, not regulations, and they map closely onto design inputs.

**21 CFR Part 11.** Where electronic records are used to satisfy regulatory requirements, the
regulation requires validated systems, secure access, and audit trails that record creation,
modification, and deletion with old and new values. A CAPA system that stores records in a
spreadsheet without an audit trail may not satisfy the record requirements, even if the content is
complete.

**Software safety classification.** IEC 62304 assigns a software safety class based on the severity
of harm that a software failure could cause, and the class determines the rigour of the lifecycle
activities. For an AI-enabled device, the classification should account for the possibility that
the model is wrong in ways the software is not: a model that produces a plausible but incorrect
output is a different failure mode from a software crash. A team that classifies its software
solely on the basis of crash risk will under-classify a model whose errors are silent.

**Post-market surveillance linkage.** The QMS does not end at release. ISO 13485 requires feedback
and complaint handling, and the EU Medical Device Regulation requires post-market surveillance and
post-market clinical follow-up. These processes generate the nonconformities that feed CAPA, and
they are the mechanism by which a deployed model's performance is monitored. Topic 07 covers the
measurement; this topic covers the process that receives the measurement.

**What the standards do not do.** They do not specify a model architecture, a performance
threshold, or a validation design. They specify process and evidence. A QMS can certify a product
that performs poorly, provided the process was followed and the performance meets the specified
inputs. The specification is where the clinical judgement lives, and the QMS only ensures that the
specification is met and recorded.

## 9. Industry practice and case evidence

**Documented fact: the QMSR transition.** The FDA's final rule was published on 2 February 2024,
effective 2 February 2026, and it incorporates ISO 13485:2016. The agency estimated annualised net
cost savings of roughly 507 to 532 million US dollars and a reduction of about 333,500
recordkeeping hours. The transition matters for analytics teams because the QMSR's design control
and traceability requirements apply to software, including AI-enabled device software functions,
and because the harmonised standard makes a single QMS usable in multiple jurisdictions.

**Documented fact: CAPA is a recurring inspection weakness.** The Global Harmonization Task Force
guidance on CAPA (GHTF/SG3/N18:2010) was written because CAPA was a persistent source of
nonconformity across the device industry. The guidance notes that the acronym itself is misleading,
because preventive action is not required for every corrective action, and it emphasises
effectiveness verification with a defined data source, timeframe, sample size, and acceptance
criteria. The European MDCG guidance on CAPA plan assessment (2024) makes the same points and adds
that restating a nonconformity is not a root cause analysis.

**Documented fact: the FDA GMLP principles.** The ten principles published in October 2021 are
explicitly non-binding, and they include the requirement that training and test sets be
independent, and that deployed models be monitored with retraining risks managed. The principles
are useful as design inputs because they are written in language a product team can act on.

**Inference: what an inspection actually samples.** An inspector does not read every record. The
inspector selects a product, follows its design history from input to verification to release, and
checks whether the records are consistent with the procedure and with each other. The most damaging
finding is not a missing document; it is a contradiction, because a contradiction suggests that one
of the two records is not authentic. This is why the QMS emphasises consistency across artifacts
over completeness of any single artifact. A team with fewer records that agree is in a stronger
position than a team with many records that disagree.

**Inference: why analytics teams struggle with QMS.** Three reasons. First, the vocabulary is
manufacturing-derived and the team does not see the mapping to model development. Second, the QMS
is owned by a regulatory function that is engaged at the end of development, when design controls
can no longer be applied. Third, the team's existing practices (version control, code review,
testing) are already most of the substance, but they are not documented as controls, so they do not
count as evidence.

**Inference: what a proportionate QMS looks like.** A small team can operate with five procedures:
design control, risk management, change control, CAPA, and document control. Each procedure can be
two pages. The records are the artifacts the team already produces, plus a decision log. The
internal audit is an annual sample. The total overhead is a few days per project, and it produces
the evidence that answers a buyer's questions in minutes.

**Inference: the inspection outcome that matters most.** A 483 observation is not a recall, but it
signals a process that a regulator does not trust. The observations that recur across device
inspections are incomplete design verification, inadequate CAPA effectiveness checks, and missing
records. None of these is a product failure; each is an evidence failure, and each indicates that
the organisation cannot demonstrate that the product is controlled. For an analytics team, the
lesson is that the evidence is the deliverable.

**Inference: the cost of retrofitting design controls.** A team that has already built a model and
is then asked for a design history file can reconstruct the inputs from the code, the evaluation
report, and the meeting notes, but the reconstruction is a record of what was decided rather than a
control that shaped the decisions. It is acceptable for a low-risk product and insufficient for a
high-risk one, where the auditor expects to see that the input preceded the output. The difference
is visible in the dates, which is why retrofitting is expensive and often unsuccessful.

**Inference: the commercial value.** A QMS is a sales asset in regulated markets. A vendor that can
produce a design history file, a risk file, and a CAPA log shortens diligence and reduces the
perceived risk of the purchase. The same artifacts support post-market surveillance and the
response to a recall. The cost is front-loaded and the benefit is realised at the point of sale and
at the point of failure.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| SOPs written for auditors | Procedures describe an ideal process | Compliance theatre | Daily violations, findings | Write what is done, then improve |
| No design inputs | Requirements live in slides | No traceable baseline | Verification unanchored | Numbered, testable inputs |
| Verification conflated with validation | Test set called clinical validation | Metric confusion | Product fails in workflow | Separate records and criteria |
| Correction reported as CAPA | Ticket closed, no root cause | Instance fixed, cause remains | Defect recurs | Require root cause before closure |
| Root cause ends at a person | "Analyst error" | Blame instead of process | Systemic cause persists | Ask why the process allowed it |
| No effectiveness check | Closure on action completion | Assertion instead of evidence | Fix unverified | Pre-specified metric and timeframe |
| Effectiveness measured by silence | "No complaints" | Reporting bias | False closure | Measure the process, not reports |
| Uncontrolled checklist | Spreadsheet with no owner | No review trigger | Product-specific requirement missed | Document control and owner |
| Change without impact assessment | Pull request merged | No revalidation decision | Unvalidated model in use | Change record and approval |
| Retraining without revalidation | New weights deployed | Distribution change | Performance drift undetected | Revalidate or PCCP scope |
| Records missing | Work done, no evidence | Documentation treated as overhead | Audit finding | Record at the time of work |
| Training not recorded | Procedure issued, no training log | Competence unproven | Finding; task risk | Training matrix |
| Internal audit too narrow | No findings year after year | Sampling bias | System blind spots | Risk-based sampling |
| Supplier unmanaged | Cloud model updated by vendor | Dependency outside control | Silent change | Supplier agreements and monitoring |
| Checklist without owner | Spreadsheet copied between projects | No review trigger | Product-specific requirement missed | Document control with named owner |
| CAPA backlog | Mean age rising quarter on quarter | Investigation under-resourced | Causes accumulate | Triage by risk; capacity planning |

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test |
|---|---|---|---|
| Scope and regulatory status are known | Requirements depend on them | Ambiguous classification | Record the determination and reasoning |
| Processes are repeatable enough to document | Procedures describe practice | Ad hoc development | Observe two projects; compare |
| Design inputs can be made testable | Verification needs criteria | Vague requirements | Rewrite until a test exists |
| Risk can be estimated | Controls are proportional | Novel failure modes | Use hazard analysis and post-market data |
| Root causes are findable | CAPA depends on them | Multiple interacting causes | Use fault tree and five whys; accept multiple causes |
| Effectiveness can be measured | Closure needs evidence | No baseline data | Define the metric before the action |
| Records can be produced contemporaneously | Evidence must be credible | Reconstructed after the fact | Audit trail and timestamps |
| The QMS is proportionate | Over-control causes workarounds | Uniform high-rigour process | Review effort against risk |
| Management review is real | Leadership owns the system | Meeting without decisions | Review minutes for actions |
| Suppliers can be controlled | Parts of the system are external | Cloud or model provider | Contract terms and monitoring |
| Records can be tied to a version | Audit needs traceability | No version identifiers | Check record against release log |
| CAPA causes are separable | One action can address the cause | Multiple interacting causes | Allow multiple root causes with separate actions |

## 12. Exercises

**Exercise 1. Classify the product (application, 20 minutes).**
Deliverable: a regulatory determination for three products, with the reasoning and the applicable
requirements. Acceptance criteria: each determination cites the intended use and the jurisdiction;
ambiguity is recorded rather than resolved by assumption. Time: 20 minutes.

**Exercise 2. Write design inputs (design, 30 minutes).**
Deliverable: six numbered design inputs for a model, each with a source and a test. Acceptance
criteria: every input is falsifiable; at least one addresses a use-related hazard. Time: 30
minutes.

**Exercise 3. Build a verification matrix (application, 30 minutes).**
Deliverable: a matrix mapping each input to an output and a verification record. Acceptance
criteria: every input has a corresponding test; untested inputs are marked open, not passed. Time:
30 minutes.

**Exercise 4. Run a five-whys (critique, 30 minutes).**
Deliverable: a root cause chain for a real defect, ending at a process cause rather than a person.
Acceptance criteria: at least five levels; each level is supported by evidence; the final cause is
something the organisation can change. Time: 30 minutes.

**Exercise 5. Write an effectiveness check (design, 25 minutes).**
Deliverable: a check with metric, baseline, target, sample, timeframe, data source, and owner.
Acceptance criteria: the check would fail if the defect recurred; it does not rely on absence of
complaints. Time: 25 minutes.

**Exercise 6. Assess a change (application, 30 minutes).**
Deliverable: an impact assessment for a model retraining, with a revalidation decision and the
reasoning. Acceptance criteria: considers model, data, workflow, and labelling; cites the
change-control procedure. Time: 30 minutes.

**Exercise 7. Audit five records (application, 45 minutes).**
Deliverable: an audit of five CAPA records against the procedure, with findings classified as
major, minor, or observation. Acceptance criteria: each finding cites the record and the
procedure clause; at least one finding is systemic. Time: 45 minutes.

**Exercise 8. Score a CAPA (critique, 30 minutes).**
Deliverable: a five-domain score for a real or constructed CAPA, with evidence per domain and a
closure recommendation. Acceptance criteria: every score cites a record; at least one domain is
partial; the recommendation distinguishes effective from unverified. Time: 30 minutes.

**Exercise 9. Design the five-procedure QMS (design, 45 minutes).**
Deliverable: a one-page outline of five procedures with purpose, owner, and records for each.
Acceptance criteria: proportionate to a small team; each procedure names its records; the set
covers design, risk, change, CAPA, and documents. Time: 45 minutes.

**Exercise 10. Write the change record (application, 20 minutes).**
Deliverable: a completed change record for a threshold change, including the revalidation decision
and its rationale. Acceptance criteria: all seven fields populated; the rationale is a testable
statement, not an assertion. Time: 20 minutes.

### How to use the exercises

Exercises 1 to 3 build the design-control skill and should be done on one product so that the
inputs, outputs, and matrix accumulate. Exercises 4 to 6 build the CAPA skill. Exercises 7 and 8
build the audit and review skill, and they are best done on records from a real organisation,
because the defects that matter (corrections dressed as CAPAs, effectiveness checks that measure
silence) are hard to invent convincingly. Exercises 9 and 10 are the institutional design.

## 13. Self-check answers

**A1. What is the difference between a correction and a corrective action?** A correction fixes the
instance. A corrective action eliminates the cause so that the nonconformity cannot recur.

**A2. What is the difference between verification and validation?** Verification confirms that the
design output meets the design input. Validation confirms that the product meets user needs in the
intended use.

**A3. Why is "the analyst made an error" not a root cause?** Because it does not explain why the
process allowed the error to reach release. The process is the thing the organisation can change.

**A4. What must an effectiveness check contain?** A metric, a baseline, a target, a sample, a
timeframe, a data source, and an owner.

**A5. When does a retrained model require revalidation?** When the change can affect performance or
the intended use. A retraining on new data is a high-impact change and normally requires
revalidation, unless it falls within an approved predetermined change control plan.

**A6. What are the four levels of the document hierarchy?** Quality manual, standard operating
procedure, work instruction, record.

**A7. Why does the QMSR matter to an analytics team outside the United States?** Because it
harmonises with ISO 13485, so a single QMS can support multiple jurisdictions, and because
suppliers and partners may require conformity as a condition of business.

**A8. What is the difference between a corrective and a preventive action?** Corrective responds to
a nonconformity that occurred; preventive responds to a potential one that has not occurred.

**A9. Name three use-related hazards for a clinical AI product.** Automation bias, alert fatigue,
and deskilling of clinicians who rely on the model.

**A10. Compute the CAPA score.** Containment 2, root cause 2, action 2, effectiveness 2,
documentation 1: total 9 of 10.

**A11. Why is a reconstructed record weaker than a contemporaneous one?** Because the evidence that
the work was done at the time is weaker: timestamps, version history, and the consistency of
details can reveal reconstruction, and a reconstructed record cannot show that the decision was
made with the information available then. Reconstruction is acceptable if labelled; it is not
equivalent.

**A12. Why is a CAPA log a leading indicator rather than a lagging one?** Because the age and
recurrence of CAPAs predict future defects: a backlog of open investigations means causes are not
being removed, and recurrent CAPAs on the same cause mean the actions are corrections. A quality
review reads the log for trends, not for the count.

**A13. What is the difference between a supplier and a partner for QMS purposes?** A supplier
provides a product or service that affects device quality and is controlled by agreement and
monitoring. A partner may share development or data. The distinction determines the contract terms,
the audit rights, and the extent to which the supplier's changes need to be reflected in your
change control.

**Rubric for Exercise 4.** Full credit: five levels; evidence at each; the final cause is a process
or document control failure that the organisation can change; the analysis distinguishes proximate
from systemic causes. Deduction for ending at a person or at "human error".

**Rubric for Exercise 5.** Full credit: all seven elements; the check is falsifiable; it measures
the process rather than the absence of complaints. Partial: metric and target but no sample or
timeframe.

## 14. Interview questions

**Q1. What is a QMS, in one sentence?**
Strong answer: the documented processes and records that ensure a product is built as intended and
that defects are found and fixed systematically. Weak answer: a set of SOPs for audits.

**Q2. How would you build a proportionate QMS for a five-person analytics team?**
Strong answer: five procedures, records from existing artifacts, annual internal audit, risk-based
depth, and a named owner. Weak answer: adopts the full device QMS unchanged.

**Q3. A defect recurs after a fix. What went wrong?**
Strong answer: the fix was a correction, not a corrective action; root cause was not reached or the
effectiveness check was absent. Weak answer: blames the analyst.

**Q4. How do you know a CAPA worked?**
Strong answer: a pre-specified effectiveness check with a metric, baseline, target, sample, and
timeframe, measured after implementation. Weak answer: says the team confirmed it.

**Q5. What is the difference between design input and design output?**
Strong answer: input is the requirement; output is the specification or artifact; verification
links them. Weak answer: uses the terms interchangeably.

**Q6. When is a model change a new product?**
Strong answer: when it changes performance, intended use, or the risk profile beyond the approved
change plan; retraining usually qualifies. Weak answer: says any code change is a new product, or
none is.

**Q7. Why do regulators care about document control?**
Strong answer: because without it, the record cannot be tied to the version that was validated, and
the evidence chain breaks. Weak answer: says it is administrative.

**Q8. What are the most common CAPA weaknesses?**
Strong answer: root cause restates the problem; action is a correction; no effectiveness check;
CAPAs open too long. Weak answer: gives a generic answer about documentation.

**Q9. How does ISO 14971 apply to an AI model?**
Strong answer: hazard analysis includes use-related hazards, performance failure, and inequitable
performance; controls are verified; residual risk is reviewed. Weak answer: treats it as a
technical FMEA only.

**Q10. What would you show a buyer who asks about your quality system?**
Strong answer: scope, design inputs and verification matrix, risk file summary, change log, CAPA
log, and the last internal audit. Weak answer: shows a certificate.

**Q11. Why is a clean internal audit suspicious?**
Strong answer: sampling may be too narrow or the auditor too close to the process; a mature system
usually has minor findings. Weak answer: says a clean audit is the goal.

**Q12. What is the most common mistake in a first QMS?**
Strong answer: writing procedures for the audit rather than for the work, so the team cannot follow
them and every day produces a deviation; the fix is to document current practice and improve it.
Weak answer: says the mistake is too little documentation.

**Q13. How do you handle a CAPA that implicates a supplier?**
Strong answer: opens the CAPA internally, notifies the supplier under the quality agreement,
obtains their root cause and action, verifies effectiveness in your own process, and records the
supplier's response. Weak answer: delegates the CAPA to the supplier and closes it.

**Q14. What evidence would you accept that a model's design controls were followed?**
Strong answer: numbered design inputs with sources, a verification matrix with results, a risk file
with controls and residual risk, a change log, and a release record; checks that the records agree.
Weak answer: accepts a design document or a slide deck.

**Q15. How do you keep a QMS from becoming bureaucracy?**
Strong answer: proportionality, few procedures, records generated by the work, and periodic review
that removes steps that add no control. Weak answer: says bureaucracy is unavoidable.

## 15. Key terms

- **QMS** — documented system for consistent quality.
- **ISO 13485** — medical device QMS standard.
- **QMSR** — FDA regulation incorporating ISO 13485.
- **Design control** — process linking inputs to outputs and verification.
- **Design input** — testable requirement.
- **Design output** — specification or artifact.
- **Verification** — output meets input.
- **Validation** — product meets user need in use.
- **Risk management** — ISO 14971 process.
- **Hazard** — potential source of harm.
- **CAPA** — corrective and preventive action.
- **Correction** — fix the instance.
- **Corrective action** — eliminate the cause.
- **Preventive action** — prevent a potential nonconformity.
- **Nonconformity** — failure to meet a requirement.
- **Root cause** — fundamental process reason.
- **Effectiveness check** — evidence the cause was removed.
- **Change control** — assessed and approved change.
- **SOP** — documented process.
- **Design history file** — record of design controls.
- **Management review** — leadership review of the QMS.
- **Internal audit** — independent conformity check.
- **Supplier control** — management of external providers.
- **Design history file** — record of design controls and decisions.
- **Change record** — documented impact assessment and approval.
- **CAPA backlog** — open investigations and their ages.
- **Stop authority** — power to hold a release.
- **Horizontal scan** — check of the same cause in other products.
- **Record integrity** — authenticity and contemporaneity of evidence.
- **Revalidation** — repeat validation after a significant change.
- **Impact assessment** — structured evaluation of a proposed change.
- **Deviation** — departure from a procedure or protocol.
- **Quality agreement** — contract defining supplier quality obligations.
- **Stop-ship** — holding a release for a quality reason.
- **Design review** — scheduled examination of design outputs.
- **Training matrix** — record of required and completed training by role.
- **Traceability matrix** — links from requirements to evidence.
- **Effectiveness criterion** — the pre-specified threshold a CAPA must meet.
- **Quality objective** — measurable target for the QMS.

## 16. Sources

- International Organization for Standardization. ISO 13485:2016, Medical devices — Quality
  management systems — Requirements for regulatory purposes. Geneva: ISO; 2016.
- US Food and Drug Administration. Medical Devices; Quality System Regulation Amendments. Final
  rule, 89 FR 7496, 2 February 2024, effective 2 February 2026.
  https://www.federalregister.gov/documents/2024/02/02/2024-01709/medical-devices-quality-system-regulation-amendments.
  Accessed 26 October 2026.
- US Food and Drug Administration. Quality Management System Regulation (QMSR).
  https://www.fda.gov/medical-devices/postmarket-requirements-devices/quality-management-system-regulation-qmsr.
  Accessed 26 October 2026.
- International Organization for Standardization. ISO 14971:2019, Medical devices — Application of
  risk management to medical devices. Geneva: ISO; 2019.
- International Electrotechnical Commission. IEC 62304:2006 with Amendment 1:2015, Medical device
  software — Software life cycle processes. Geneva: IEC; 2015.
- US Food and Drug Administration, Health Canada, Medicines and Healthcare products Regulatory
  Agency. Good Machine Learning Practice for Medical Device Development: Guiding Principles.
  October 2021. https://www.fda.gov/media/153486/download. Accessed 26 October 2026.
- Global Harmonization Task Force. Quality management system — Medical devices — Guidance on
  corrective action and preventive action and related QMS processes. GHTF/SG3/N18:2010.
- Medical Device Coordination Group. MDCG 2024-12, Corrective and preventive action (CAPA) plan
  assessment. October 2024. Accessed 26 October 2026.
- eCFR. 21 CFR 820.100, Corrective and preventive action.
  https://www.ecfr.gov/current/title-21/chapter-I/subchapter-H/part-820. Accessed 26 October 2026.

> **Note:** Regulatory content is educational and is not legal advice. Conformity with ISO 13485
> or the QMSR requires assessment by qualified professionals.

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`05_reproducibility.md`](05_reproducibility.md)
- Next topic: [`07_post-market-evaluation.md`](07_post-market-evaluation.md)
- Audit preparation: [`08_audit.md`](08_audit.md)
- Module survey: [`../../course/18_quality_evaluation_reporting.md`](../../course/18_quality_evaluation_reporting.md)
- Labs: [`../../labs/`](../../labs/)
