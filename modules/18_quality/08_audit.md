# Audit

> Module 18 — Quality, Evaluation, and Reporting
> Topic 08 of 09
> Estimated study time: 3.5 hours
> Prerequisites: 06
> Version 2026.3

## 1. Why this topic exists

An audit is a systematic, independent, documented examination of evidence against criteria. It is
not a conversation about intentions, a review of slides, or a test of whether the team is
likeable. It is a sampling exercise with a written conclusion, and its output is a set of findings
that an organisation must answer.

Analytics teams meet audits in four forms. A notified body audits the quality management system
before issuing a certificate. An FDA investigator inspects a device manufacturer and may issue a
Form 483 with observations. A customer audits a vendor before signing or renewing a contract. An
internal auditor samples the team's records as part of the quality system's own checking. The
criteria differ, the consequences differ, and the mechanics are the same: someone who did not do
the work examines a sample of the evidence and asks whether it supports the claim.

The four audit types differ in consequence but not in method. A notified body can withhold a
certificate; the FDA can issue observations and escalate; a customer can decline a contract; an
internal auditor can escalate to management. In each case the organisation is asked the same
question: show me the evidence that this system was controlled. The answer is either a trace
through records or an explanation, and only one of those survives the follow-up question.

The reason this topic matters commercially is that most analytics teams are unprepared in the same
way. They have the work, the code, and the results, but the evidence is distributed across a
repository, a shared drive, a dashboard, and the memory of two people. When an auditor asks for
the design input that the validation was supposed to satisfy, the team can find the validation but
not the input. When the auditor asks which model version was released on a given date, the team can
find a release note but not the weights. The gap is not dishonesty; it is the absence of an
evidence architecture.

The gap also has an internal cost that is easy to miss. An organisation that cannot produce its own
evidence cannot learn from its own experience. It cannot tell whether last year's defect was fixed,
whether the threshold change improved outcomes, or whether the model it retired was ever validated.
The audit exposes the gap, but the gap exists whether or not an auditor arrives. The preparation
described in this topic is therefore useful even for a team that will never be audited.

A further consequence is that the team cannot answer a customer's diligence questions without a
project. A buyer asks which model version was validated, what changed since, and what monitoring
exists. A team with an evidence map answers in an hour. A team without one answers in a week, or
answers with a description that the buyer cannot verify. The audit and the sale ask the same
question, which is why preparation serves both.

The consequence of that gap is disproportionate. A single major nonconformity can delay a
certificate, block a contract, or trigger a follow-up inspection. A pattern of findings across
several audits can escalate to a warning letter or a consent decree in the device context. None of
these outcomes is caused by a bad model. They are caused by an inability to demonstrate that a
good model was controlled.

This topic teaches how an audit works, what auditors actually ask for, how to build the evidence
architecture that answers them, and how to run a mock audit that finds the gaps before an external
auditor does. It builds directly on topic 06: the QMS is the set of processes, and the audit is the
test of whether they are operating. It also builds on topic 01: the evidence chain is what the
auditor traces.

A note on scope. This topic is not legal advice, and it does not teach the specific requirements of
every jurisdiction. It teaches the audit method and the evidence that satisfies it, which are
portable across the FDA, EU notified bodies, and customer audits.

## 2. Learning outcomes

By the end of this topic you can:

1. **Explain** what an audit is, what it is not, and how sampling determines what it can conclude.
2. **Build** an evidence map that links each audit criterion to the artifact that satisfies it.
3. **Trace** a single claim from the report back to the raw record in the presence of an auditor.
4. **Classify** findings by severity and write them in a form that survives challenge.
5. **Design** a sampling plan that gives a stated confidence for a stated failure rate.
6. **Run** a mock audit, including opening meeting, document review, interviews, and closing
   meeting.
7. **Prepare** a team for an audit without coaching it to perform.
8. **Respond** to a finding with a corrective action that addresses the cause rather than the
   wording.

## 3. The core mental model

An audit has four elements, and a weakness in any one invalidates the conclusion.

```text
   criteria  --------  evidence  --------  sampling  --------  judgement
      |                   |                    |                    |
  the requirement     the record           which records        conformity,
  (standard, SOP,     that shows           were examined        nonconformity,
  regulation,         the work was                              observation
  contract)           done
```

**Criteria** are the requirements. They must be stated before the audit, because an auditor who
chooses the criteria after seeing the evidence can find a nonconformity in any organisation.
**Evidence** is the record, not the account. **Sampling** determines the confidence: an audit of
five records out of five hundred supports a statement about those five, not about the five
hundred. **Judgement** is the classification of each finding, and it must be traceable to the
criteria and the evidence.

A second model is the evidence hierarchy. Auditors prefer evidence that was created at the time of
the work and is difficult to change after the fact.

```text
   strongest    system-generated records with timestamps and audit trails
                signed and dated approval records
                version-controlled documents and release tags
                contemporaneous meeting minutes with actions
                retrospective summaries and reports
                interviews and recollection
   weakest      assertions of intent ("we always do that")
```

The hierarchy is not about honesty. It is about verifiability. A system-generated record can be
checked against another system; a recollection cannot. A team that has strong records at the top of
the hierarchy can survive an auditor who samples broadly; a team whose evidence is mostly
recollection needs the auditor to sample narrowly, which is not under its control.

A third model is the trace. An auditor does not read everything. The auditor picks a claim, a
product, or a release, and follows it through the system: from the claim in the label or report, to
the requirement, to the design output, to the verification, to the release record, to the
post-market data. The trace is the audit's core move, and the organisation's job is to make the
trace short.

```text
  claim -> requirement -> design output -> verification -> release -> monitoring
    |          |               |               |             |           |
  label     design input    specification    test report   tag/date   metric
```

If any arrow requires an email search, the trace is long, and a long trace is where findings come
from. The evidence map in section 5 is the artifact that shortens it.

A fourth property of the trace is that it must be repeatable by someone else. A team that can
reconstruct a claim by asking the person who built it has not demonstrated traceability; it has
demonstrated that the person is still employed. The test is whether a new analyst, given the
evidence map, can follow the same path without help. That is the same clean-room logic used for
reproducibility in topic 05, applied to records rather than code.

The trace also has a diagnostic property: the number of searches it requires is a proxy for the
quality of the evidence architecture. A trace that needs one search is a system with linked
artifacts. A trace that needs eleven is a system held together by people. The count is worth
recording, because it converts a vague feeling about readiness into a number that can be tracked
across audits.

Independence deserves a note because it is easy to fake. An internal auditor from the same team is
not independent, even if they did not work on the product, because their reporting line and their
social context create a bias. The practical test is whether the auditor can raise a major finding
against a senior colleague without consequence. If the answer is no, the audit is a review, and it
should be described as one rather than presented as assurance.

A fourth model is severity. Findings are classified because the response differs. A major
nonconformity indicates a systemic failure or a failure that could affect safety; a minor
nonconformity is an isolated lapse; an observation is a weakness that does not yet breach a
requirement. The classification is a judgement, and the organisation can dispute it with evidence,
but it cannot negotiate it away by describing the finding as an observation.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Standard |
|---|---|---|---|
| Audit | Systematic, independent, documented examination against criteria | A informal review or a presentation | ISO 19011 |
| Audit criteria | The requirements used as a reference | Chosen after seeing the evidence | Audit plan |
| Audit evidence | Records, statements, or observations relevant to the criteria | The auditor's impression | ISO 19011 |
| Objective evidence | Verifiable data supporting existence or truth | Any document | ISO 9000 |
| Conformity | Fulfilment of a requirement | Absence of findings | ISO 9000 |
| Nonconformity | Failure to fulfil a requirement | Any weakness | ISO 9000 |
| Observation | A weakness that does not yet breach a requirement | A polite nonconformity | Practice |
| Major nonconformity | Systemic failure or failure with safety impact | Any finding the auditor cares about | Notified body practice |
| Minor nonconformity | Isolated lapse with limited impact | A minor issue not worth fixing | Practice |
| Finding | A conclusion from the audit evidence | A rumour | Audit report |
| Audit programme | Set of audits planned over a period | A single audit | ISO 19011 |
| Audit plan | Scope, criteria, schedule, and team for one audit | The audit report | ISO 19011 |
| Sampling risk | The risk that a sample does not represent the population | Ignored | Statistics |
| Audit trail | Chronological record of creation, change, and deletion | A general log file | 21 CFR 11.10(e) |
| ALCOA | Attributable, legible, contemporaneous, original, accurate | A slogan | FDA, MHRA |
| Traceability matrix | Links requirements to evidence | A file list | Design control |
| Design history file | Record of design controls for a product | The code repository | 21 CFR 820.30 |
| Form 483 | FDA list of inspectional observations | A recall | FDA |
| Warning letter | FDA enforcement letter after serious violations | A 483 | FDA |
| Notified body | EU conformity assessment organisation | A regulator | EU MDR |
| Opening meeting | Start of audit: scope, criteria, logistics | A negotiation | ISO 19011 |
| Closing meeting | End of audit: findings presented | A vote | ISO 19011 |
| Auditor independence | Freedom from bias and conflict | Physical separation only | ISO 19011 |
| Remote audit | Audit conducted with technology | A document review | ISO 19011 |
| Unannounced audit | Audit without prior notice | A surprise inspection | Practice |

Four distinctions carry the topic.

**Audit versus inspection.** An audit examines a management system against criteria and samples
records. An inspection is a regulator's statutory activity, often with a narrower legal scope and
stronger powers. The methods overlap, and a team that prepares for an audit is largely prepared
for an inspection, but the consequences differ.

**Nonconformity versus observation.** A nonconformity cites a requirement and evidence of failure.
An observation cites a weakness without a specific requirement. Teams often try to have findings
downgraded; the productive response is to accept the classification, correct the cause, and use the
record to show the system improved.

**Audit evidence versus audit documentation.** Evidence is what the organisation produced during the
work: the test result, the approval, the log. Documentation is what the organisation produced for
the audit: the evidence map, the index, the summary. A team can improve its documentation without
improving its evidence, and auditors are trained to tell the difference. The map is useful because
it points to evidence; it is not evidence itself.

**Evidence versus explanation.** The auditor asks for evidence. The team offers an explanation. The
explanation is useful for context and cannot substitute for the record. "We always check that"
without a record is the most common exchange in a first audit, and it never ends well.

**Audit finding versus CAPA.** A finding describes what is wrong. A CAPA describes what will be
done about the cause. The audit report contains findings; the corrective action plan contains
CAPAs; the follow-up audit tests whether they worked.

## 5. The mechanics, step by step

This procedure prepares for and runs a mock audit of an analytics team, and it doubles as the
preparation method for an external audit. It takes two to four weeks of preparation and one to
three days of audit for a single product.

### Step 0. Decide whether the audit is internal or external

**Input:** the audit programme and the upcoming external audit dates.

**Output:** an audit calendar with the internal audits scheduled ahead of the external ones.

**Rationale:** The internal audit is the rehearsal. An organisation that runs its own audit four to
eight weeks before an external one has time to close findings; one that runs it the week before
has time only to hide them, which does not work and damages credibility.

**Branch point:** if there is no internal audit programme, the first external audit will function
as one, at higher stakes.

### Step 1. State the scope, criteria, and product

**Input:** the audit request or the internal audit plan, and the product documentation.

**Output:** an audit scope statement naming the product, the period, the sites, the criteria
(standard clauses, procedures, contract terms), and the exclusions.

**Rationale:** Scope determines what the audit can conclude. An audit of one product cannot support
a conclusion about the portfolio; an audit of the last six months cannot support a conclusion about
the design phase that preceded it.

**Branch point:** three scopes, each with different preparation.

| Scope | Criteria | Typical duration |
|---|---|---|
| Product audit | Design controls, risk file, release records for one model | One to two days |
| Process audit | One procedure, for example CAPA, across products | Two to three days |
| System audit | All QMS clauses and processes | Three to five days |

### Step 2. Build the evidence map

**Input:** the criteria and the product's artifacts.

**Output:** a matrix with one row per criterion and columns for the artifact, the location, the
owner, and the last review date.

**Rationale:** The evidence map is the single most useful preparation artifact. It converts a
standard into a list of documents, and it exposes the criteria with no artifact before the auditor
does. A team that builds the map honestly will find its own gaps.

**Branch point:** if a criterion has no artifact, mark it as a gap and prepare an explanation and a
remediation date. Do not invent a document on the eve of the audit; a document created after the
request is visible in the metadata and is worse than a gap.

### Step 3. Test the trace on one claim

**Input:** the evidence map and a claim from the label or report.

**Output:** a completed trace with a locator for each hop, and a list of hops that required a
search.

**Rationale:** The trace is the audit in miniature. If the team cannot complete it in an hour, the
auditor will not complete it in a day, and the finding will be about traceability rather than about
the model.

**Branch point:** if a hop is missing, record whether the underlying work was done (a reporting
gap) or not done (a control gap). The remedies differ.

### Step 4. Design the sampling plan

**Input:** the population of records, the criterion, and the confidence required.

**Output:** a sample size, a selection method, and a statement of what the sample can and cannot
support.

**Rationale:** Sampling is where audits are won and lost. A sample of five release records supports
a statement about those five; it does not support "the release process is effective". The auditor
knows this, and the organisation should too, because it affects how findings are worded and how
they are rebutted.

**Branch point:** two sampling approaches.

| Approach | Use when | Output |
|---|---|---|
| Judgement-based | Risk-based selection, small populations | Findings about selected cases |
| Statistical | Need a confidence statement | Upper bound on failure rate |

For attribute sampling with zero expected failures, the sample size for 95 percent confidence that
the failure rate is below 5 percent is about 59 records. The calculation is in section 7. A team
that wants to claim a low failure rate must sample enough to support it, which is usually more than
it expects.

### Step 5. Prepare the team without coaching the answers

**Input:** the evidence map, the scope, and the interview schedule.

**Output:** a briefing that explains the process, the criteria, and the roles, and that explicitly
does not script the answers.

**Rationale:** An auditor can tell the difference between a team that knows its system and a team
that has memorised a script. Scripted answers collapse under a follow-up question, and the collapse
is more damaging than the original gap. The briefing should ensure that the right people are in the
room and that they know where the records are, not what to say.

**Branch point:** if a team member is likely to be defensive, assign them a role that does not
require answering questions about their own area. Defensiveness reads as concealment.

### Step 6. Hold the opening meeting

**Input:** the audit plan, the scope, and the attendees.

**Output:** a confirmed scope, a schedule, a contact list, and a documented understanding of the
criteria.

**Rationale:** The opening meeting is where scope creep and misunderstanding are prevented. It is
also where the auditor confirms independence and confidentiality. A scope disagreement discovered
at the closing meeting is a dispute; discovered at the opening, it is a clarification.

**Branch point:** if the auditor proposes a scope broader than the plan, ask for the criteria and
the time allocation. Do not accept an expanded scope without the resources to support it.

**What the auditor will ask for.** The request list varies by scope, but the core is stable, and a
team can prepare it as a standing pack.

| Category | Request | Where it lives |
|---|---|---|
| Requirements | Design inputs with sources | Design file |
| Design | Architecture, model specification, interfaces | Design file |
| Risk | Hazard analysis, controls, residual risk | Risk file |
| Verification | Test plan, results, traceability | Verification record |
| Validation | Protocol, results, clinical evidence | Validation report |
| Release | Version, hash, approval, change description | Release record |
| Change | Change requests and impact assessments | Change log |
| Monitoring | Metrics, thresholds, review minutes | Monitoring plan |
| CAPA | Records, root causes, effectiveness checks | CAPA log |
| Competence | Training records for named roles | Training matrix |
| Documents | Register with effective versions | Document control |

### Step 7. Review documents against criteria

**Input:** the evidence map and the sample.

**Output:** a document review record: criterion, document, version, date, conformity, and comment.

**Rationale:** Document review is the cheapest part of the audit and the part most likely to produce
findings, because documents are where version drift and missing approvals are visible. The reviewer
checks that the document is the effective version, that it is approved, and that it says what the
process requires.

**Branch point:** if the document is effective but the practice differs, the finding is about
adherence, not documentation. Record both.

### Step 7a. Manage the logistics of a remote audit

**Input:** the audit plan and the document repository.

**Output:** access for the auditor, a shared index, a video link, and a named coordinator.

**Rationale:** A remote audit tests the organisation's evidence architecture more harshly than an
on-site audit, because the auditor cannot ask someone to walk to a filing cabinet. If the evidence
map is good, remote is efficient; if it is not, the audit stalls and the findings are about access
rather than about control. ISO 19011:2026 expands the guidance on remote methods, and the principles
are unchanged.

**Branch point:** if the auditor cannot access a system, the team has two options: provide an
export with a documented query, or record the limitation. Do not screen-share a system and call it
access; the auditor needs to sample, not to watch.

### Step 8. Conduct interviews and observe practice

**Input:** the interview schedule and the process.

**Output:** interview notes with the questions asked and the evidence offered.

**Rationale:** Interviews test whether the documented process is the process in use. The productive
questions are about specific cases ("show me the last release you approved and how you decided"),
not about general practice ("do you follow the procedure"). Specific questions produce records;
general questions produce assertions.

**Branch point:** if the interviewee describes a practice that differs from the procedure, record
the difference and ask which is authoritative. The answer determines whether the finding is a
documentation defect or an adherence defect.

### Step 9. Complete the trace and test the evidence

**Input:** the selected claim and the records.

**Output:** a completed trace with evidence at each hop, and a list of untested hops.

**Rationale:** This is the audit's central test. The trace should be completed for at least one
claim per product, and for a second claim if the first reveals a gap. The auditor should record
what was tested, not only what was found, because the untested hops are the limits of the
conclusion.

**Branch point:** three trace outcomes.

| Outcome | Evidence | Finding |
|---|---|---|
| Complete and consistent | Every hop has a record that matches | None |
| Complete with a gap | A hop is missing but the work is evidenced elsewhere | Minor or major depending on the claim |
| Broken | A hop contradicts the claim | Major |

### Step 10. Write and classify findings

**Input:** the document review, the interviews, and the trace.

**Output:** a findings log with the criterion, the evidence, the classification, and the required
action.

**Rationale:** A finding must be written so that the organisation can respond to it and a third
party can verify it. The structure is: requirement, evidence, gap, consequence. A finding without
the requirement is an opinion; a finding without the evidence is an allegation.

**Branch point:** severity classification, with the rules stated in advance.

| Class | Test | Example |
|---|---|---|
| Major | Systemic failure, or a failure that could affect safety or the claim | No design inputs for a released model |
| Minor | Isolated lapse, contained, no safety impact | One release record missing an approver |
| Observation | Weakness without a breached requirement | Checklist not reviewed for two years |

### Step 11. Hold the closing meeting

**Input:** the findings log.

**Output:** agreed findings, disputed findings with reasons, and a timeline for the report and the
response.

**Rationale:** The closing meeting is not a negotiation of the outcome; it is a test of whether the
findings are understood and whether any factual correction is needed. A team that argues about
severity at this point loses credibility. A team that provides a missing record and shows the
finding is factually wrong wins the point and keeps the relationship.

**Branch point:** if a finding is disputed, record the dispute and the evidence. The report can
carry a minority view; it cannot carry an unresolved factual error.

### Step 11a. Write the audit report

**Input:** the agreed findings log and the audit notes.

**Output:** a report with scope, criteria, method, findings, conclusions, and the limits of the
audit.

**Rationale:** The report is the record that the audit happened and what it concluded. It must state
what was sampled and what was not, because a conclusion drawn from twelve records out of ninety is a
conclusion about twelve records. A report that omits the sampling basis invites over-reading by
everyone who cites it.

**Branch point:** if the audit found no nonconformities, the report should say what was tested and
what the sample can support. A clean report with no method section is not evidence of a clean
system, and it is the kind of report that an external auditor reads as a sign that the internal
audit was not independent.

### The audit timetable

Preparation has a shape, and teams that start late compress the wrong parts.

| Weeks before audit | Activity | Output |
|---:|---|---|
| 8 | Confirm scope and criteria; assign coordinator | Audit plan |
| 6 | Build or refresh the evidence map | Matrix with gaps |
| 4 | Trace two claims; close gaps | Trace records |
| 3 | Internal mock audit | Findings log |
| 2 | Close major findings; prepare the pack | Corrective actions |
| 1 | Brief the team; test remote access | Agenda and access |
| 0 | Opening meeting, document review, interviews, closing | Report |

The compression that hurts is moving the trace to the last week. The trace is what reveals whether
the evidence map is accurate, and a trace done in week one leaves time to fix the arrows; a trace
done in week zero produces findings instead.

### Step 12. Respond, correct, and verify

**Input:** the findings log.

**Output:** a corrective action plan with owners and dates, and evidence of closure for each
finding.

**Rationale:** The response is judged on cause, not on wording. "We will remind the team" is not a
corrective action. "We will add the threshold review to the controlled checklist, train the release
managers, and verify by auditing the next ten releases" is.

**Branch point:** if the finding is a one-off with a known cause, a correction may be sufficient; if
it is systemic, a CAPA is required.

### Edge cases

**Incomplete records.** A missing record is a finding even when the work was done. The remedy is to
reconstruct contemporaneously where possible and to record the reconstruction; backdating is
fraud.

**Selective evidence.** A team that presents only the records that pass is selecting, and auditors
sample to detect it. The remedy is to present the population and let the auditor sample.

**Subgroup claims.** A claim about subgroup performance needs the subgroup analysis in the design
file. A claim in the label without the analysis is a major finding.

**Protocol deviations.** A validation that deviated from its protocol without a recorded
justification is a finding, even if the deviation improved the result.

**Post-hoc changes.** A change record created after the change is a finding about change control,
not only about the change. The dates are the evidence.

**Conflicting reference standards.** If the design input names one standard and the verification
used another, the trace breaks at verification. The finding cites both documents.

**Retracted studies.** If the design file cites a retracted study, the requirement's basis is
invalid. The finding is about the design input, and the remedy is a reassessment.

### Decision table for audit readiness

| Evidence map coverage | Trace complete | Records current | Readiness |
|---|---|---|---|
| Above 90% | Yes | Yes | Ready |
| 70-90% | Yes | Mostly | Conditional; close gaps first |
| 50-70% | Partly | Mixed | Not ready; four to eight weeks of work |
| Below 50% | No | No | Not ready; the audit will find systemic gaps |

## 6. Worked example

The case is a mock audit of a 12-person analytics team that has built and deployed a sepsis
deterioration alert at three hospitals. The audit is an internal system audit against ISO 13485
clauses 7.3 (design and development), 7.5 (production and service provision), 8.2 (monitoring and
feedback), and 8.5 (improvement), plus the team's own procedures. It is conducted by an auditor
from another division over two days.

The case is a composite assembled from common patterns. The findings are typical of first audits of
analytics teams; the numbers are constructed so that the sampling arithmetic in section 7 can be
checked.

### Scope and evidence map

Scope: the sepsis alert, versions 1.0 to 2.3, the period from design to the current release, three
hospitals, and the four clauses above. The team produced an evidence map with 42 criteria. Of
those, 31 had an artifact, six had a partial artifact, and five had none. The audit sampled 12
records across releases, design reviews, and CAPA.

### What the auditor asked for

The request list is worth reproducing, because it is the practical answer to "what does an auditor
want".

- The design inputs, numbered, with their sources.
- The verification matrix linking each input to a test and a result.
- The risk management file, including use-related hazards.
- The release record for version 2.3, with the approval and the change description.
- The validation report and the protocol it followed.
- The monitoring plan and the last four review minutes.
- The CAPA log, with the two most recent records and their effectiveness checks.
- The training records for the three release managers.
- The document register, showing the effective version of each procedure.
- The trace for the claim "reduces time to antibiotic administration", from label to evidence.

The last item is the one that produced the most findings. The team could show a validation AUROC
and a retrospective time-to-antibiotic analysis, but not a design input that specified the
time-to-antibiotic endpoint or a verification test against it. The claim had been added to the
marketing material after the validation, and the design file did not contain it.

### Findings log

| # | Criterion | Evidence | Finding | Class |
|---|---|---|---|---|
| 1 | 7.3.3 Design inputs | Design file has 11 inputs; none names time to antibiotics | Claim in label not traceable to an input | Major |
| 2 | 7.3.6 Verification | Verification matrix covers 10 of 11 inputs | One input unverified at release | Major |
| 3 | 7.3.7 Validation | Validation protocol dated after data lock | Validation not prospectively specified | Minor |
| 4 | 7.5.9 Traceability | Release 2.3 record lacks model hash | Cannot confirm which weights were released | Major |
| 5 | 7.5.1 Control of production | Threshold changed in 2.3 without a change record | Unapproved change to operating point | Major |
| 6 | 8.2.1 Feedback | Complaint log not linked to monitoring | Feedback loop incomplete | Minor |
| 7 | 8.2.2 Complaint handling | Two complaints closed without investigation | Procedure not followed | Minor |
| 8 | 8.5.2 Corrective action | CAPA-014 has no effectiveness check | Closure without evidence | Major |
| 9 | 6.2 Competence | One release manager has no training record | Competence unproven | Minor |
| 10 | 4.2.4 Document control | Two versions of the release checklist in use | Obsolete document not withdrawn | Minor |
| 11 | 8.4 Data analysis | Monitoring review minutes missing for two quarters | Review not performed as planned | Observation |

Counts: five major, five minor, one observation.

### How the classification was reached

Finding 1 is major because the claim cannot be supported by the design file. The consequence is not
that the model is wrong; it is that the organisation cannot demonstrate that the claim was verified,
which is the definition of a systemic design control failure.

Finding 4 is major because without a model hash the release cannot be tied to the validated
artifact. The consequence is that any future performance question requires reconstructing which
weights were live, which may be impossible.

Finding 5 is major because an unapproved threshold change alters the operating characteristics that
the validation measured. The change may be beneficial; the control failure is the finding.

Finding 8 is major because an effectiveness check is the only evidence that a CAPA worked. Without
it, the organisation has closed a ticket and not a cause.

Finding 11 is an observation because the monitoring plan specified quarterly reviews and two were
missed, but the metric reviews that did occur were documented and no safety signal was missed. The
weakness is real and does not yet breach a requirement in a way that affects a claim.

### Scoring the audit readiness

Before the audit, the team self-assessed readiness on five domains, each 0 to 2.

| Domain | Self-score | Auditor score | Comment |
|---|---:|---:|---|
| Design controls | 2 | 0 | Major findings 1, 2, 3 |
| Change control | 1 | 0 | Findings 4, 5, 10 |
| Monitoring and feedback | 2 | 1 | Findings 6, 7, 11 |
| CAPA | 1 | 1 | Finding 8 |
| Competence and documents | 2 | 1 | Findings 9, 10 |
| Total | 8/10 | 3/10 | Self-assessment was optimistic |

The gap between self-score and auditor score is the most useful output of the mock audit. The team
believed its design controls were strong because it had a design document; the auditor found that
the document was not a set of testable inputs. The lesson is that self-assessment tends to score
the existence of an artifact, while an auditor scores whether the artifact does the job.

### Recommendation

Do not proceed to the external audit. Close the five major findings first, with a corrective action
plan that addresses causes: convert the design document into numbered inputs, add the missing
verification, introduce a model hash into the release record, bring the threshold under change
control, and define effectiveness checks for CAPA-014. Target four to six weeks.

The minor findings can be closed in parallel, except finding 7, which requires a review of the two
closed complaints and a possible reopening.

### How the findings were written

Finding 5 is reproduced in full as an example of the form.

Requirement: procedure QP-07, change control, clause 5.2, requires an impact assessment and
approval before any change to a parameter that affects the operating characteristics.

Evidence: release record R-2.3 dated 14 January 2026; the threshold in the configuration file is
0.31; the validation report for version 2.0 specifies 0.38; no change request references the
threshold.

Gap: the threshold was changed without an impact assessment or approval.

Consequence: the operating characteristics reported in the validation do not correspond to the
released configuration, and the organisation cannot demonstrate that the change was assessed for
its effect on sensitivity and alert burden.

The form matters because it survives a challenge. A finding that says "the threshold was changed
without approval" invites a debate about whether the change was material. A finding that cites the
requirement, the record, the discrepancy, and the consequence does not.

### The follow-up

The corrective action plan is due within a period agreed at the closing meeting, often 30 days. It
has one row per finding with four fields: the root cause, the action, the evidence of completion,
and the effectiveness check. The follow-up audit, usually three to six months later, samples the
same criteria and tests whether the actions were implemented and whether the findings recurred.

The follow-up is where most programmes fail. The actions are often implemented, but the evidence is
not filed in a form the next auditor can find, and the finding is raised again as "not effectively
implemented". The remedy is to treat the closure evidence as a deliverable: a named document, in
the evidence map, with a date and an owner.

### Two diagnostics that would change the verdict

**Diagnostic 1: the design file contained a numbered input for time to antibiotics, and the
verification matrix referenced it.** Finding 1 would disappear and finding 2 would reduce to a
documentation gap, because the missing verification would be one test rather than an unverified
claim. The design-control domain would move from 0 to 1, the total from 3 to 5, and the
recommendation from "do not proceed" to "proceed after closing the major findings in change control
and CAPA". The audit outcome changes materially on one document, which is why the evidence map is
worth building before the auditor arrives.

**Diagnostic 2: the release record included the model hash and the threshold change record.** Two
major findings would disappear, the change-control domain would move from 0 to 2, and the total
would rise to 6. The recommendation would become "proceed with a focused follow-up on design
controls". This diagnostic shows that the findings cluster by process, and fixing one process can
remove several findings at once.

### What the team should have done before the audit

The findings are not random. Four preparation failures produced almost all of them.

The team treated the design document as a design file. A design document describes what was built;
a design file contains numbered, testable inputs and a verification record against each. The
difference is the first two major findings, and it is the most common gap in analytics teams
because model documentation and design control look similar from a distance.

The team treated the release note as a release record. A note says what changed; a record says
which artifact was released, who approved it, and what testing supported it. The absence of a hash
is the third major finding, and it is the one that makes every later performance question harder.

The team treated the threshold as an engineering parameter rather than a controlled specification.
The threshold determines the operating characteristics that the validation measured, so changing it
is a product change. This is the fourth major finding.

The team treated CAPA as a ticket system. A ticket closes when the action is done; a CAPA closes
when the effectiveness check passes. This is the fifth major finding, and it is the one most likely
to recur across audits because the underlying habit is organisational.

Each failure has the same shape: a work artifact was accepted as a control record because it was
adjacent to the control. The audit's function is to test whether the artifact does the job the
control requires.

### Vary one assumption: the sample size

The audit sampled 12 records from a population of about 90 releases, design reviews, and CAPAs. If
the auditor had sampled 30 records, the probability of finding at least one major nonconformity
would rise, and the estimate of the failure rate would become more precise. With 5 major findings
in 12 records, the observed proportion is 41.7 percent; with the same proportion in 30 records, the
confidence interval would narrow substantially, and the conclusion "systemic" would be better
supported.

The reverse is also true. A sample of five records with no findings cannot support a claim that the
process is effective; the upper 95 percent bound on the failure rate with zero findings in five is
about 45 percent. A team that wants to claim a low failure rate must sample enough to support it,
which is the argument for the sampling plan in section 7.

## 7. Code, query, or calculation

The script computes attribute sample sizes, the upper confidence bound for an observed defect
count, and a severity-weighted readiness score. It uses only the standard library.

```python
"""Audit sampling and findings scoring.

Clopper-Pearson upper bound is computed by bisection on the binomial CDF,
which avoids external statistics packages.
"""

from math import comb


def binomial_cdf(k, n, p):
    return sum(comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k + 1))


def upper_bound(k, n, alpha=0.05):
    """One-sided (1-alpha) upper bound for a proportion with k defects in n."""
    if k >= n:
        return 1.0
    lo, hi = k / n, 1.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if binomial_cdf(k, n, mid) > alpha:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def sample_size_zero_defects(p, confidence=0.95):
    """n such that P(0 defects | rate p) <= 1 - confidence."""
    n = 1
    while (1 - p) ** n > (1 - confidence):
        n += 1
    return n


print("zero-defect sampling (95% confidence):")
for p in (0.20, 0.10, 0.05, 0.02):
    print(f"  to bound failure rate below {p:.0%}: n = {sample_size_zero_defects(p)}")

print()
print("observed findings, 12 records:")
for k in (0, 1, 3, 5):
    ub = upper_bound(k, 12)
    print(f"  {k} findings -> observed {k/12:.1%}, upper 95% bound {ub:.1%}")

print()
print("what 30 records would buy, same proportion:")
for k in (0, 5, 10):
    ub = upper_bound(k, 30)
    print(f"  {k} findings -> observed {k/30:.1%}, upper 95% bound {ub:.1%}")

# Severity-weighted readiness: major=5, minor=2, observation=1, normalised by
# the maximum possible for the number of criteria audited.
WEIGHTS = {"major": 5, "minor": 2, "observation": 1}
findings = [
    ("F1", "major"), ("F2", "major"), ("F3", "minor"), ("F4", "major"),
    ("F5", "major"), ("F6", "minor"), ("F7", "minor"), ("F8", "major"),
    ("F9", "minor"), ("F10", "minor"), ("F11", "observation"),
]
score = sum(WEIGHTS[sev] for _, sev in findings)
max_score = 42 * 5          # 42 criteria audited, all major at worst
print()
print(f"severity-weighted findings points: {score}")
print(f"as share of maximum ({max_score}): {score/max_score:.1%}")
print(f"readiness index: {100 * (1 - score/max_score):.1f}/100")

by_sev = {}
for _, sev in findings:
    by_sev[sev] = by_sev.get(sev, 0) + 1
print("counts:", by_sev)
```

Expected output:

```text
zero-defect sampling (95% confidence):
  to bound failure rate below 20%: n = 14
  to bound failure rate below 10%: n = 29
  to bound failure rate below 5%: n = 59
  to bound failure rate below 2%: n = 149

observed findings, 12 records:
  0 findings -> observed 0.0%, upper 95% bound 22.1%
  1 findings -> observed 8.3%, upper 95% bound 33.9%
  3 findings -> observed 25.0%, upper 95% bound 52.7%
  5 findings -> observed 41.7%, upper 95% bound 68.5%

what 30 records would buy, same proportion:
  0 findings -> observed 0.0%, upper 95% bound 9.5%
  5 findings -> observed 16.7%, upper 95% bound 31.9%
  10 findings -> observed 33.3%, upper 95% bound 49.9%

severity-weighted findings points: 36
as share of maximum (210): 17.1%
readiness index: 82.9/100
counts: {'major': 5, 'minor': 5, 'observation': 1}
```

A note on the arithmetic. The upper bound is a one-sided Clopper-Pearson limit computed by
bisection on the binomial cumulative distribution. It is exact in the sense that it does not rely
on a normal approximation, which matters for small samples where the approximation is poor. The
zero-defect formula is the special case: with zero findings, the bound is 1 - 0.05^(1/n), and
solving for n gives the numbers in the first table.

The script's limits are worth stating. The upper bound assumes a random sample from a
well-defined population; an auditor who samples by risk does not have a random sample, and the
bound is then a rough indication rather than a confidence statement. The zero-defect formula
assumes independence between records, which fails when one process produces many of them. And the
severity weights are arbitrary. The outputs are useful for planning and for communicating scale,
not for defending a conclusion in a dispute.

The severity weighting is a management device, not a statistical one. The weights encode a
judgement that a major finding is two and a half times as important as a minor one, which is
defensible for tracking but not for comparing audits with different scopes. A team should not use
the index to compare its readiness with another team's; it should use it to track its own trend
against the same criteria.

Two readings of the output. First, the sampling table is the answer to "how many records do we
need to show the process is working?" If the team wants to claim a failure rate below 5 percent
with 95 percent confidence and expects zero findings, it needs 59 records. Most teams sample five
and are surprised when the claim is not accepted.

Second, the readiness index is deliberately not the headline. A weighted score of 82.9 out of 100
looks acceptable, and the audit conclusion is "do not proceed", because five major findings cannot
be averaged away by the many criteria that passed. The index is useful for tracking improvement
over time and dangerous as a summary of readiness. The correct summary is the count of major
findings, which is five.

## 8. Data, standards, and regulatory context

**ISO 19011:2018.** The international guidance for auditing management systems. It sets out seven
principles (integrity, fair presentation, due professional care, confidentiality, independence,
evidence-based approach, and risk-based approach), the management of an audit programme, the
conduct of an audit, and the competence of auditors. It is guidance, not a certifiable standard, and
a nonconformity cannot be raised against it. A fourth edition, ISO 19011:2026, has been published
and replaces the 2018 edition, with expanded guidance on remote auditing; check which edition the
audit programme references.

**ISO 13485:2016.** The QMS standard against which most device audits are conducted. The clauses
most relevant to an analytics team are 7.3 (design and development), 7.5 (production and service
provision), 8.2 (monitoring and measurement), 8.4 (analysis of data), and 8.5 (improvement).

**FDA QMSR.** The Quality Management System Regulation, effective 2 February 2026, incorporates ISO
13485:2016 by reference and retains design controls for Class II, Class III, and certain Class I
devices. FDA inspections under the QMSR use the agency's compliance programme rather than the
retired quality system inspection technique. The agency has stated that it will not accept ISO
13485 certification in lieu of inspection.

**21 CFR Part 11.** Where electronic records support regulatory claims, the regulation requires
validated systems, access controls, and secure, computer-generated, time-stamped audit trails that
record creation, modification, and deletion, retain old and new values, and remain available for
review. An audit will sample the audit trail, not just the record.

**EU MDR and notified bodies.** Notified bodies conduct conformity assessment audits against the
MDR, including the quality system and the technical documentation. The technical documentation
includes the clinical evaluation, the risk management file, and the post-market surveillance plan.
An unannounced audit is possible under the MDR for certain devices.

**Form 483 and warning letters.** An FDA investigator may issue a Form 483 listing inspectional
observations at the end of an inspection. The manufacturer responds in writing. Serious or repeated
failures can lead to a warning letter or other enforcement. The 483 is not itself a determination
of violation; it is a list of observations that the manufacturer must answer.

**MDSAP and multi-jurisdiction audits.** The Medical Device Single Audit Program allows one audit
to satisfy the requirements of several regulators, including the FDA, Health Canada, and others.
For a company selling in multiple markets, the programme reduces audit burden but raises the
standard, because each participating regulator can review the audit report. A team preparing for
MDSAP should expect the auditor to test the same criteria across jurisdictions and to look for
inconsistencies between the documents prepared for each.

**Audit trails as a specific test.** An auditor testing 21 CFR Part 11 will do more than look for a
log. The auditor will check that the trail is computer-generated, time-stamped, records the user,
the date, the old value, and the new value, cannot be edited or disabled by the user, and is
retained for the record retention period. A system that logs access but not changes, or that allows
an administrator to delete entries, fails the test even if the log looks complete. The relevant
questions are in the FDA's October 2024 question-and-answer guidance on electronic records and
signatures.

**AI-specific criteria in an audit.** Beyond the general QMS clauses, an auditor examining an
AI-enabled device will look for the design inputs that address the model's specific risks:
representativeness of training data, independence of training and test sets, calibration,
subgroup performance, human-AI interaction, and the monitoring plan. These map to the FDA's Good
Machine Learning Practice principles and to TRIPOD+AI reporting items, but an audit tests whether
the evidence exists, not whether it was reported. A team that has a publication meeting TRIPOD+AI
but no design input for subgroup performance has a reporting artifact and an audit gap.

**What the standards do not require.** They do not require a specific document format, a specific
software tool, or a specific number of records. They require that the process be defined, followed,
and evidenced. The audit tests those three things, which is why a team with simple documents that
match its practice fares better than a team with elaborate documents that do not.

## 9. Industry practice and case evidence

**Documented fact: CAPA remains a recurring weakness.** The Global Harmonization Task Force
guidance on CAPA (GHTF/SG3/N18:2010) was written because corrective and preventive action was a
persistent source of nonconformity across the device industry, and the European MDCG guidance on
CAPA plan assessment (2024) makes the same observation about root cause quality and effectiveness
verification. The inference for audit preparation: the CAPA file is the highest-yield place to
look for gaps, because it is where the industry's weaknesses are concentrated.

**Documented fact: the QMSR changes the reference standard.** The FDA's final rule, published on 2
February 2024 and effective 2 February 2026, incorporates ISO 13485:2016 and retires the quality
system inspection technique. An organisation that prepared for the old QSR will find that the
questions are similar but the reference clauses are different, and that design control expectations
are harmonised with the international standard.

**Inference: what auditors actually sample.** An auditor samples by risk and by trace, not
uniformly. The sample concentrates on the product with the highest risk, the process with the most
recent change, and the records that link a claim to a test. A team that prepares by improving its
weakest area rather than by polishing its best one is preparing for the audit that will happen.

**Inference: why first audits go badly.** Three patterns. The team treats the audit as a
presentation and prepares slides rather than records. The evidence is distributed across systems
that do not reference each other, so the trace requires an email search. The team's self-assessment
scores the existence of documents rather than their adequacy, producing an optimism gap like the one
in section 6.

**Inference: findings cluster by process.** In the mock audit, eleven findings came from five
processes, and two processes (design control and change control) produced seven of them. That
clustering is typical: a systemic weakness produces several findings that look independent until
the trace is drawn. The practical consequence is that remediation should be organised by process,
not by finding, because fixing the process closes several findings at once and fixing findings one
by one leaves the process intact.

**Inference: why evidence maps fail.** An evidence map fails when it is built from the standard
rather than from the work. A map that lists clauses and then searches for a document to attach will
record a document for every clause, including documents that do not satisfy the requirement. A map
built from the artifacts, asking which criterion each artifact satisfies, exposes the clauses with
no artifact. The direction of construction determines whether the map is a preparation tool or a
reassurance exercise.

**Inference: the first audit sets the relationship.** An organisation that responds to its first
audit with evidence, prompt corrective actions, and honest gap disclosure establishes a pattern
that makes later audits shorter and less adversarial. An organisation that responds with
minimisation and retrospective documents establishes the opposite, and the cost compounds because
each subsequent auditor arrives with the previous report. The response to the first finding is
therefore a strategic choice, not an administrative one.

**Inference: audits are cheaper than failures.** The cost of a two-day audit and a week of
preparation is small against the cost of a delayed certificate, a lost contract, or a recall. The
organisations that treat audit preparation as overhead pay it repeatedly; the ones that treat it as
the maintenance of their evidence architecture pay it once and then maintain it. The difference is
not the size of the quality function but whether the evidence map is a living document.

**Inference: the audit as a management tool.** An organisation that audits itself before an
external audit learns where its evidence is weak while there is still time to act. The cost is a
few days of an internal auditor's time; the benefit is a shorter external audit, fewer findings,
and a record that shows the organisation detects and corrects its own problems. Regulators and
customers weight that record heavily, because it is evidence of a functioning system rather than a
compliant one.

**Inference: what good preparation looks like.** Build the evidence map, test the trace on one
claim, close the gaps that are closable, and prepare an honest list of the gaps that are not. An
auditor who is shown a known gap with a remediation plan treats the organisation as in control; an
auditor who discovers the same gap treats it as concealment. The difference is not the gap but the
disclosure.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Evidence scattered | Trace needs an email search | No evidence map | Major traceability finding | Build the map; link artifacts |
| Self-assessment optimism | Scores based on document existence | Existence confused with adequacy | Readiness gap | Score against criteria, not artifacts |
| Missing design inputs | Claims in label not in design file | Marketing added after design | Major finding | Trace every claim to an input |
| No model hash | Release record lacks artifact identity | Weak release process | Cannot tie validation to release | Hash in the release record |
| Unapproved threshold change | Operating point differs from validation | Change control bypassed | Major finding | Change record for every parameter |
| CAPA without effectiveness | Closure on action completion | Assertion instead of evidence | Major finding; recurrence | Pre-specified check |
| Obsolete documents in use | Two checklist versions | No withdrawal step | Minor to major finding | Document control |
| Training gaps | New staff without records | Onboarding gap | Competence finding | Training matrix |
| Complaints not linked | Separate systems | No interface | Feedback finding | Link and reconcile |
| Scripted answers | Team recites procedure | Coaching instead of preparation | Credibility loss | Brief on process, not answers |
| Arguing severity | Closing meeting dispute | Defensiveness | Relationship damage | Accept, correct, appeal in writing |
| Sample too small | Five records, broad claim | Convenience | Claim unsupported | Statistical sampling |
| Retrospective records | Dates after request | Reconstruction | Integrity concern | Contemporaneous records |
| Audit scope drift | New clauses at opening | Poor planning | Unprepared area | Confirm scope and criteria |
| Evidence map built from clauses | Every clause has "some" document | Reassurance exercise | False readiness | Build from artifacts upward |
| Closure evidence unfiled | Action done, record missing | Documentation treated as optional | Repeat finding | Treat closure evidence as a deliverable |
| Internal auditor not independent | Same reporting line | Social and reporting bias | Findings softened | Independent function |
| Audit used as a presentation | Slides instead of records | Misunderstanding the method | Findings on evidence | Brief on documents |

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test |
|---|---|---|---|
| Criteria are known in advance | Audit is against criteria | Scope changes at opening | Confirm in writing |
| Records exist for the period | Sampling needs a population | Retrospective start-up | Build the map; mark gaps |
| Sample represents the population | Findings generalise | Non-random selection | Use statistical sampling |
| Auditor is independent | Judgement is credible | Auditor owns the process | Check assignment |
| Records are contemporaneous | Evidence is valid | Reconstruction | Check metadata and dates |
| The trace can be completed | Audit's core test | Distributed systems | Rehearse the trace |
| Severity rules are consistent | Findings are comparable | Ad hoc classification | Pre-defined rules |
| Corrective actions address causes | Closure is meaningful | Action is a correction | Review root cause |
| Follow-up will occur | CAPA has effect | No verification | Schedule verification |
| Remote audit is equivalent | Logistics | Records not accessible | Test access before the audit |
| The evidence map is current | Preparation depends on it | Map not updated after releases | Review it quarterly |
| Findings are about the system, not the people | Response stays constructive | Defensiveness | Separate process from individual early |

## 12. Exercises

**Exercise 1. Build an evidence map (application, 45 minutes).**
Deliverable: a matrix of 20 criteria against artifacts, locations, owners, and review dates for a
product you know. Acceptance criteria: every criterion has an artifact or an explicit gap; gaps are
prioritised by severity. Time: 45 minutes.

**Exercise 2. Trace one claim (application, 40 minutes).**
Deliverable: a completed trace from a label claim to a verification record, with locators and a list
of hops that required a search. Acceptance criteria: every hop has a locator or a documented gap;
the trace can be followed by a stranger. Time: 40 minutes.

**Exercise 3. Design a sampling plan (application, 25 minutes).**
Deliverable: a sample size for a stated confidence and expected failure rate, with the selection
method. Acceptance criteria: the calculation is shown; the claim the sample supports is stated.
Time: 25 minutes.

**Exercise 4. Classify findings (critique, 30 minutes).**
Deliverable: eight findings classified as major, minor, or observation, with the criterion and the
evidence. Acceptance criteria: each classification cites the rule; at least one finding is
downgraded from the team's initial view. Time: 30 minutes.

**Exercise 5. Write a finding (application, 20 minutes).**
Deliverable: one finding in the form requirement, evidence, gap, consequence. Acceptance criteria:
the requirement is cited; the evidence is a locator; the consequence is specific. Time: 20 minutes.

**Exercise 6. Run a mock audit (design, 90 minutes).**
Deliverable: an opening meeting agenda, a two-hour document review, three interviews, and a closing
meeting with a findings log. Acceptance criteria: findings are evidenced; the closing meeting
records agreement or dispute; the log is usable by the team. Time: 90 minutes.

**Exercise 7. Write the corrective action plan (design, 40 minutes).**
Deliverable: a plan for three findings with root cause, action, owner, date, and effectiveness
check. Acceptance criteria: each action addresses a cause; each check is measurable. Time: 40
minutes.

**Exercise 8. Write the audit report (application, 35 minutes).**
Deliverable: a two-page report for the mock audit with scope, criteria, method, sampling basis,
findings, conclusion, and limits. Acceptance criteria: the sampling basis is stated; the conclusion
does not exceed what the sample supports; findings are numbered and classified. Time: 35 minutes.

**Exercise 9. Prepare the audit pack (design, 30 minutes).**
Deliverable: an index of documents and records for an external audit, with a one-page guide to the
evidence map. Acceptance criteria: the pack is navigable without the team present; every document
has a version and date. Time: 30 minutes.

### How to use the exercises

Exercises 1 to 5 build the individual skills: mapping, tracing, sampling, classifying, and writing.
Exercises 6 to 8 build the audit as a whole and should be run as a sequence with the same product,
because the findings from the mock audit feed the corrective action plan and the report. Exercise 9
is the artifact that makes the next audit cheaper. A team that does only Exercise 6 learns what an
audit feels like; a team that does all nine builds a repeatable preparation process.

## 13. Self-check answers

**A1. What is the difference between an audit and an inspection?** An audit is a systematic
examination against criteria, often by a third party or an internal function. An inspection is a
regulator's statutory activity with legal powers. The methods overlap; the consequences differ.

**A2. What are the seven principles of auditing in ISO 19011:2018?** Integrity, fair presentation,
due professional care, confidentiality, independence, evidence-based approach, and risk-based
approach.

**A3. Why is a sample of five records insufficient to claim a low failure rate?** Because with zero
findings in five, the upper 95 percent bound on the failure rate is about 45 percent. The sample
supports a statement about those five records, not about the process.

**A4. What is the structure of a well-written finding?** Requirement, evidence, gap, consequence.

**A5. What distinguishes a major from a minor nonconformity?** Major: systemic failure or a failure
that could affect safety or the claim. Minor: isolated lapse, contained, no safety impact.

**A6. Why does a model hash matter in a release record?** It ties the released artifact to the
validated artifact, so a later question about performance can be answered with certainty.

**A7. What is the first thing to do when an auditor requests a document the team does not have?**
Record the gap honestly and provide the evidence that the work was done, if it exists. Do not create
a retrospective document.

**A8. How many records are needed to bound a failure rate below 5 percent with 95 percent
confidence, expecting zero findings?** 59.

**A9. Why is a self-assessment usually optimistic?** Because teams score the existence of artifacts
rather than whether the artifacts satisfy the criteria.

**A10. What is the difference between a correction and a corrective action in an audit response?**
A correction fixes the instance; a corrective action addresses the cause and includes an
effectiveness check.

**A11. Why should the audit report state what was not sampled?** Because the conclusion is limited
to the sample. A report that says "the design control process is effective" after examining three
records overstates the evidence, and the next auditor will treat the overstatement as a weakness in
the internal audit.

**A12. What is the difference between an audit finding and a nonconformity?** A finding is the
auditor's conclusion from the evidence; a nonconformity is a failure to meet a requirement. Every
nonconformity is a finding, but a finding can also be an observation or an opportunity for
improvement.

**A13. What makes an evidence map useful rather than decorative?** It is built from the artifacts
upward, names a locator for each criterion, and marks gaps explicitly. A map built from the
standard downward tends to attach a document to every clause whether or not it satisfies the
requirement.

**A15. Why does an AI design file need a human-AI interaction input?** Because the performance of
the human-AI team is not the performance of the model alone. Automation bias, override behaviour,
and interface design change the clinical effect, so the requirement should specify how the
interaction was evaluated and what the acceptance criterion is.

**A16. What is the difference between a correction and a corrective action in an audit response?**
A correction fixes the instance, such as re-releasing the correct threshold. A corrective action
addresses the cause, such as bringing the threshold under change control, and requires an
effectiveness check.

**A14. Why is the number of searches in a trace a useful metric?** It measures how well the
artifacts are linked. A trace that needs one search indicates a linked evidence architecture; a
trace that needs many indicates a system held together by people, which fails when they leave.

**Rubric for Exercise 6.** Full credit: opening meeting with scope and criteria; document review
with locators; interviews with specific case questions; findings classified by rule; closing meeting
record. Partial: findings without criteria or evidence.

**Rubric for Exercise 7.** Full credit: root cause is systemic; action addresses the cause; owner
and date named; effectiveness check measurable. Deduction for "retrain the team" without a process
change.

## 14. Interview questions

**Q1. How would you prepare for an audit in two weeks?**
Strong answer: build the evidence map, trace one claim, close closable gaps, prepare an honest gap
list with dates, brief the team on process not answers. Weak answer: prepares slides.

**Q2. What does an auditor mean by objective evidence?**
Strong answer: verifiable records or observations that can be checked, preferably generated at the
time of the work. Weak answer: says any document.

**Q3. A finding says design inputs are missing. How do you respond?**
Strong answer: accepts, reconstructs the inputs from the design record where possible, documents
the reconstruction, adds the missing inputs, and verifies. Weak answer: argues the work was done.

**Q4. Why is sampling central to an audit?**
Strong answer: an audit is finite; the sample determines what can be concluded; statistical sampling
supports a confidence claim, judgement sampling does not. Weak answer: says sampling is random.

**Q5. What is the most common CAPA weakness auditors find?**
Strong answer: root cause restates the problem, action is a correction, no effectiveness check.
Weak answer: says documentation.

**Q6. How do you handle a disagreement with a finding at the closing meeting?**
Strong answer: provides evidence, records the dispute, escalates in writing if unresolved, and does
not argue about severity. Weak answer: concedes or confronts.

**Q7. What is the difference between a Form 483 and a warning letter?**
Strong answer: a 483 lists inspectional observations and invites a response; a warning letter is an
enforcement communication after serious or repeated violations. Weak answer: treats them as the same.

**Q8. How do you audit a model you cannot run?**
Strong answer: audit the design file, the verification records, the release record, and the
monitoring data; test the trace; do not claim to have tested the model. Weak answer: says it is
impossible.

**Q9. What records would you expect for a threshold change?**
Strong answer: change request, impact assessment, approval, verification, revalidation decision, and
release record. Weak answer: says a pull request.

**Q10. Why can a clean audit be a warning sign?**
Strong answer: sampling may be too narrow or the auditor too close; a mature system usually has
minor findings. Weak answer: says clean is the goal.

**Q11. How do you keep audit evidence current between audits?**
Strong answer: the evidence map is a living document, updated at each release and reviewed
quarterly; the trace is rehearsed annually. Weak answer: prepares only when an audit is announced.

**Q12. How would you prepare a team that has never been audited?**
Strong answer: explains the method and the criteria, runs a mock opening meeting and a mock
interview, ensures the evidence map is shared, and rehearses the trace rather than the answers.
Weak answer: writes a briefing document with suggested responses.

**Q13. What is the most common cause of a repeat finding?**
Strong answer: the corrective action addressed the instance rather than the cause, or the closure
evidence was not filed where the next auditor could find it. Weak answer: blames the auditor.

**Q14. What AI-specific evidence would you expect in a design file?**
Strong answer: inputs covering data representativeness, train-test independence, calibration,
subgroup performance, human-AI interaction, and monitoring, each with a verification record. Weak
answer: names a model card or a publication.

**Q15. How do you audit a CAPA system?**
Strong answer: samples records across risk classes, checks root cause quality, action type,
effectiveness evidence, and aging; looks for corrections dressed as CAPAs. Weak answer: counts open
CAPAs.

**Q16. What is the role of the opening meeting?**
Strong answer: confirms scope, criteria, schedule, and independence; prevents scope creep and
misunderstanding. Weak answer: treats it as a formality.

## 15. Key terms

- **Audit** — independent examination against criteria.
- **Criteria** — requirements used as the reference.
- **Objective evidence** — verifiable record or observation.
- **Conformity** — fulfilment of a requirement.
- **Nonconformity** — failure to fulfil a requirement.
- **Observation** — weakness without a breached requirement.
- **Major finding** — systemic or safety-affecting failure.
- **Minor finding** — isolated, contained lapse.
- **Audit programme** — planned set of audits.
- **Audit plan** — scope, criteria, schedule for one audit.
- **Sampling risk** — risk the sample misrepresents the population.
- **Audit trail** — chronological record of changes.
- **ALCOA** — attributable, legible, contemporaneous, original, accurate.
- **Evidence map** — criteria linked to artifacts.
- **Trace** — claim to requirement to test to release.
- **Design history file** — record of design controls.
- **Form 483** — FDA inspectional observations.
- **Warning letter** — FDA enforcement communication.
- **Notified body** — EU conformity assessment body.
- **Opening meeting** — start of audit.
- **Closing meeting** — presentation of findings.
- **Corrective action** — action on the cause.
- **Evidence architecture** — how records are linked and found.
- **Sampling basis** — the population and method behind a sample.
- **Closure evidence** — proof that a corrective action was implemented.
- **Mock audit** — internal rehearsal of an external audit.
- **Audit coordinator** — the person who manages the audit logistics.
- **Scope creep** — expansion of an audit beyond the agreed plan.
- **Trace search count** — number of searches needed to complete a trace.
- **Human-AI interaction record** — evidence of how the workflow was evaluated.
- **Predetermined change control plan** — pre-approved envelope for planned changes.
- **Risk-based sampling** — selecting records by risk rather than at random.
- **Audit readiness** — the state of evidence against the criteria.
- **Finding log** — numbered findings with criterion, evidence, class, and action.
- **Report limits** — the statement of what the sample can and cannot support.
- **Evidence pack** — indexed documents prepared for an audit.
- **Repeat finding** — a finding raised again at a later audit, indicating ineffective action.
- **Audit trail review** — examination of who changed what, when, and why.
- **Conformity assessment** — the process of determining whether requirements are met.
- **Evidence sufficiency** — whether the sample supports the conclusion drawn.

## 16. Sources

- International Organization for Standardization. ISO 19011:2018, Guidelines for auditing management
  systems. Geneva: ISO; 2018. A fourth edition, ISO 19011:2026, has been published.
- International Organization for Standardization. ISO 13485:2016, Medical devices — Quality
  management systems — Requirements for regulatory purposes. Geneva: ISO; 2016.
- US Food and Drug Administration. Medical Devices; Quality System Regulation Amendments. Final
  rule, 89 FR 7496, 2 February 2024, effective 2 February 2026.
  https://www.federalregister.gov/documents/2024/02/02/2024-01709/medical-devices-quality-system-regulation-amendments.
  Accessed 26 October 2026.
- eCFR. 21 CFR 11.10, Controls for closed systems.
  https://www.ecfr.gov/current/title-21/chapter-I/subchapter-A/part-11/subpart-B/section-11.10.
  Accessed 26 October 2026.
- US Food and Drug Administration. Quality Management System Regulation (QMSR).
  https://www.fda.gov/medical-devices/postmarket-requirements-devices/quality-management-system-regulation-qmsr.
  Accessed 26 October 2026.
- Global Harmonization Task Force. Quality management system — Medical devices — Guidance on
  corrective action and preventive action and related QMS processes. GHTF/SG3/N18:2010.
- Medical Device Coordination Group. MDCG 2024-12, Corrective and preventive action (CAPA) plan
  assessment. October 2024. Accessed 26 October 2026.
- US Food and Drug Administration. Good Documentation Practices. https://www.fda.gov/media/185615/download.
  Accessed 26 October 2026.

> **Note:** Regulatory content is educational and is not legal advice. Inspection outcomes and
> enforcement consequences depend on the specific facts and jurisdiction.

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`07_post-market-evaluation.md`](07_post-market-evaluation.md)
- Next topic: [`09_documentation.md`](09_documentation.md)
- Quality management: [`06_quality-management.md`](06_quality-management.md)
- Module survey: [`../../course/18_quality_evaluation_reporting.md`](../../course/18_quality_evaluation_reporting.md)
- Labs: [`../../labs/`](../../labs/)
