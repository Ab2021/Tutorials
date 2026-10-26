# Incident Response

> Module 15 — Deployment and Machine Learning Operations (MLOps)
> Topic 07 of 10
> Estimated study time: 3 hours
> Prerequisites: 05, 06
> Version 2026.3

## 1. Why this topic exists

Every deployed system has incidents. The question is not whether a model will fail in production,
but whether the organization will notice, respond, communicate, and learn. Incident response is the
discipline that turns a failure into a bounded event rather than an open-ended harm.

Model incidents are harder than service incidents for three reasons. They are often silent: the
service returns 200 while the output is wrong. They are often clinical: the consequence is a missed
deterioration, not a failed request. And they are often regulatory: a failure that may have
contributed to harm can trigger a reporting obligation with a deadline measured in days.

The survey file states the rule that anchors this topic: a model incident is a clinical safety
incident until proven otherwise. The rule is not rhetorical. It determines who is paged, what is
communicated, and whether the regulatory clock starts.

### 1.1 The decisions this topic unblocks

| Decision | Who is blocked | Consequence of no incident process |
|---|---|---|
| Who is in charge during an incident | Engineering, clinical operations | Parallel investigations and conflicting messages |
| How severe the incident is | On-call, clinical safety officer | Over-response or under-response |
| Whether to roll back, disable, or continue | Incident commander | The wrong action, taken too late |
| What to tell clinicians and patients | Communications, clinical leadership | Rumour, distrust, or a safety gap |
| Whether the event is reportable | Regulatory affairs | A missed statutory deadline |
| What must change afterward | Quality, engineering | The same incident recurs |

### 1.2 The cost of a slow response

The cost of an incident is roughly the product of three factors: the number of people affected, the
severity of the effect per person, and the duration. Incident response works on the third factor.
It cannot change what happened before detection, but it can shorten the time between detection and
mitigation, and it can prevent recurrence.

A service incident that lasts 30 minutes and affects 2 percent of requests is an error-budget
event. A model incident that lasts 30 days and affects 12 percent of true positives is a clinical
safety event. The second is often detected later and mitigated more slowly because the signals are
subtler and the response requires clinical judgment.

### 1.3 Why "roll back" is not always the answer

Rollback is the default mitigation for a service incident. For a model incident, it is one option
among several. If the failure is caused by a data feed, rolling back the model does not fix the
feed; the old model also consumes it. If the failure is a threshold that is too low, disabling the
model may be worse than adjusting the threshold. If the failure is a workflow change, neither
rollback nor disable addresses it.

The incident commander must choose among rollback, disable, degrade, adjust, and continue with
monitoring. The choice depends on the diagnosis, the fallback, and the clinical consequence of
each option.

### 1.4 The regulatory clock

In the United States, a manufacturer must report a device-related death or serious injury to the
Food and Drug Administration within 30 calendar days of becoming aware of information that
reasonably suggests the device may have caused or contributed to the event. A 5-day report applies
in specified circumstances, including when remedial action is needed to prevent an unreasonable
risk of substantial harm. User facilities have a 10-working-day reporting requirement for deaths
and serious injuries.

The clock starts when any employee becomes aware, not when the investigation concludes. An incident
response process that waits for root cause before assessing reportability will miss the deadline.
The assessment must run in parallel with the technical response.

The organizational cost is separate from the clinical cost. An incident that is handled poorly
consumes senior attention for weeks, damages the credibility of the model programme, and makes the
next deployment harder to approve. An incident that is handled well can strengthen trust, because
the organization demonstrated that it could detect, communicate, and correct.

## 2. Learning outcomes

By the end of this topic you can:

1. **Classify** an incident by severity using clinical, operational, and regulatory criteria.
2. **Assign** an incident commander and the parallel tracks for technical, clinical, and regulatory
   response.
3. **Choose** a mitigation from rollback, disable, degrade, adjust, or monitor, and justify it.
4. **Write** a timeline that distinguishes detection, acknowledgement, mitigation, and resolution.
5. **Assess** reportability against a stated regulatory framework and start the clock.
6. **Run** a post-incident review that produces corrective actions with owners and dates.
7. **Design** a drill that tests the process before a real incident.

## 3. The core mental model

An incident runs on four parallel tracks. The technical track fixes the system. The clinical track
protects patients. The regulatory track assesses obligations. The communication track keeps
stakeholders informed. They share a timeline but not a queue.

```text
                     incident timeline
   t0 detect ──▶ t1 acknowledge ──▶ t2 mitigate ──▶ t3 resolve ──▶ t4 review
        │              │                  │               │             │
        ▼              ▼                  ▼               ▼             ▼
  ┌──────────┐   ┌──────────┐      ┌──────────┐    ┌──────────┐  ┌──────────┐
  │ TECHNICAL│   │ TECHNICAL│      │ TECHNICAL│    │ VALIDATE │  │ ROOT     │
  │ triage   │   │ diagnose │      │ mitigate │    │ restore  │  │ CAUSE    │
  └──────────┘   └──────────┘      └──────────┘    └──────────┘  └──────────┘
  ┌──────────┐   ┌──────────┐      ┌──────────┐    ┌──────────┐  ┌──────────┐
  │ CLINICAL │   │ CLINICAL │      │ CLINICAL │    │ CONFIRM  │  │ CAPA     │
  │ impact   │   │ workaround│     │ safety   │    │ safety   │  │ PLAN     │
  └──────────┘   └──────────┘      └──────────┘    └──────────┘  └──────────┘
  ┌──────────┐   ┌──────────┐      ┌──────────┐    ┌──────────┐  ┌──────────┐
  │ REGULAT. │   │ REPORT-  │      │ NOTIFY   │    │ CLOSE    │  │ UPDATE   │
  │ assess   │   │ ABILITY  │      │ if req.  │    │ report   │  │ FILINGS  │
  └──────────┘   └──────────┘      └──────────┘    └──────────┘  └──────────┘
  ┌──────────┐   ┌──────────┐      ┌──────────┐    ┌──────────┐  ┌──────────┐
  │ COMMS    │   │ COMMS    │      │ COMMS    │    │ COMMS    │  │ PUBLISH  │
  │ hold     │   │ internal │      │ clinical │    │ all      │  │ REVIEW   │
  └──────────┘   └──────────┘      └──────────┘    └──────────┘  └──────────┘
```

### 3.1 The four tracks

**Technical.** Triage, diagnose, mitigate, restore, and verify. Owned by the on-call engineer or
the incident commander.

**Clinical.** Assess the effect on patients and the workflow, define a workaround, and decide
whether clinicians need to be told to distrust or ignore the model. Owned by the clinical safety
officer or the clinical lead.

**Regulatory.** Assess whether the event is reportable, start the clock, and prepare the filing.
Owned by regulatory affairs or quality.

**Communication.** Hold internal and external messages, keep a single source of truth, and avoid
speculation. Owned by communications with the incident commander.

A clinical response may generate a corrective and preventive action (CAPA). The tracks are parallel because the regulatory clock does not wait for the technical fix and the
clinical workaround may be needed before the cause is known.

### 3.2 Severity is not the same as impact

Severity combines impact, urgency, and the availability of a workaround.

| Severity | Impact | Urgency | Workaround | Example |
|---|---|---|---|---|
| P1 | Potential or actual serious harm; broad scope | Immediate | None | Sepsis model returning no alerts during ward rounds |
| P2 | Degraded clinical decision support; limited scope | Hours | Manual process exists | Sensitivity down 10 points for one site |
| P3 | Operational degradation; no clinical effect yet | Business day | Yes | Latency above service level objective (SLO), no missed decisions |
| P4 | Quality or trend issue | Scheduled | Yes | Calibration drifting toward, not past, the floor |

The severity determines the response: who is paged, how often updates are required, and whether
the incident commander role is activated. It must be assigned early and revised as information
arrives.

### 3.3 The timeline is the primary artifact

The most important record of an incident is the timeline. It answers the questions a reviewer will
ask: when did we know, when did we act, how long were patients exposed, and what did we tell whom.

```text
detect       the first signal, human or automatic
acknowledge  a human takes ownership
mitigate     the harmful effect is reduced or stopped
resolve      the system is restored and verified
review       the cause is understood and actions are assigned
```

The gaps between these points are the metrics that matter: time to acknowledge, time to mitigate,
time to resolve. They are more useful than a single mean time to resolve, because they point to the
stage that needs investment.

### 3.4 Blameless is not consequence-free

A blameless review focuses on systems and conditions rather than individual fault. It does not mean
that no one is accountable. The incident commander is accountable for the response; the service
owner is accountable for the corrective actions; the organization is accountable for the conditions
that made the error likely.

The distinction matters because a blame-oriented review produces silence, and silence produces
recurrence. A blameless review produces information, and information produces fixes.

### 3.5 The first fifteen minutes

The first fifteen minutes set the trajectory of the response. A useful sequence:

1. Acknowledge the alert and open the record. Do not wait for confirmation to open it.
2. Assign a severity and a commander. If unsure, assign higher.
3. Freeze changes to the affected system unless the change is the mitigation.
4. Ask the clinical question: is anyone harmed, and could anyone be harmed in the next hour.
5. Ask the regulatory question: when did we become aware, and does the framework apply.
6. Send a holding communication: what is known, what is not, when the next update will come.
7. Choose the first diagnostic, not the first fix.

The sequence is short because it is meant to be memorable under pressure. The most common failure
is skipping step 4 and step 5 while the technical team debugs, which delays the clinical workaround
and starts the regulatory clock late.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| Incident | An unplanned event that degrades a service, a model, or the safety of a workflow | Any alert | Incident identifier |
| Severity | A classification of impact, urgency, and workaround | The number of alerts | P1 to P4 |
| Detection | The first signal of the incident | The first human response | Timestamp |
| Acknowledgement | A human takes ownership | Detection | Timestamp |
| Mitigation | Reduction or stopping of the harmful effect | Root cause fix | Timestamp |
| Resolution | Restoration and verification of the system | Mitigation | Timestamp |
| Incident commander | The person accountable for the response | The most senior engineer | Role |
| Triage | Rapid assessment of severity and cause | Diagnosis | Minutes |
| Containment | Limiting the blast radius | Fixing the cause | Scope |
| Rollback | Restoring the previous known-good artifact | Restarting the service | Time to restore |
| Failover | Switching to a standby system | Retrying the request | Time to failover |
| Degradation | Deliberately reducing functionality to protect safety | An outage | Mode |
| Workaround | A manual process that substitutes for the model | A permanent fix | Procedure |
| Post-incident review | A structured review of cause and response | A blame meeting | Report |
| Root cause | The condition that, if removed, would have prevented the incident | The last thing that broke | Statement |
| Contributing factor | A condition that increased likelihood or impact | A root cause | Statement |
| Corrective action | A change that reduces recurrence or impact | A recommendation | Action identifier |
| CAPA | Corrective and preventive action | Any fix | Record identifier |
| MDR | Medical Device Report to the FDA | Any incident report | Form FDA 3500A |
| Reportable event | An event that meets a regulatory reporting criterion | Any serious incident | Determination |
| Field safety notice | A communication to customers about a safety issue | A recall | Notice identifier |
| Recall | Removal or correction of a device in the field | A software update | Recall class |
| Clinical safety officer | The role accountable for clinical risk management | The clinician on call | Role |
| MTTD | Mean time to detect | Time to acknowledge | Minutes or days |
| MTTR | Mean time to resolve | Time to mitigate | Minutes or days |
| Blast radius | The scope of affected users or patients | The number of errors | Count |
| Runbook | The procedure for a known alert | A general incident guide | Document version |
| Playbook | A procedure for a class of incidents | A runbook | Document version |
| Tabletop | A discussion-based exercise | A live drill | Exercise record |
| Drill | A live test of the response | A tabletop | Exercise record |
| Status page | The external communication channel | An internal chat | URL |

Two definitions carry the most weight.

**Mitigation is not resolution.** Mitigation stops the harm; resolution restores the system. A
rollback may mitigate in minutes and leave the underlying data defect unresolved for weeks. The
timeline must record both.

**Root cause is not a person.** "The engineer deployed the wrong model" is an event, not a cause.
The cause is the condition that allowed the wrong model to be deployed without a check. The
distinction determines whether the corrective action is training or a control.

## 5. The mechanics, step by step

The procedure is the incident lifecycle as an operator runs it. The steps are ordered by time, but
several run in parallel: clinical impact and regulatory assessment begin as soon as the incident is
opened, not after the technical fix.

### Step 1 — Detect and open the incident

**Input.** An alert, a user report, a clinical concern, or an audit finding.

**Output.** An incident record with an identifier, a timestamp, and an initial description.

**Rationale.** The incident record is the container for everything that follows. Opening it early
is better than opening it late, because the timeline and the communication trail begin at the first
signal. A team that investigates in a chat channel and opens a record after the fix loses the
evidence of its own response time.

The detection source matters. An automatic alert suggests the monitoring worked. A clinician
report suggests the monitoring missed something, which is itself a finding for the review.

Branch point: if the report is a single user complaint with no corroborating signal, the incident
may be opened at the lowest severity and upgraded if evidence accumulates. Do not dismiss it
without a record.

### Step 2 — Assign severity and an incident commander

**Input.** The initial description and the severity matrix.

**Output.** A severity (P1 to P4) and a named incident commander.

**Rationale.** The severity determines who is paged and how often updates are required. The
incident commander provides a single decision-maker and a single point of accountability. Without
one, parallel investigations produce conflicting actions and conflicting messages.

The commander does not need to be the most senior engineer. The commander needs the authority to
decide among mitigation options and the discipline to keep the timeline and communication moving.
A senior engineer may be the technical lead and report to the commander.

Branch point: if the severity is ambiguous, assign the higher severity and downgrade later.
Upgrading late is more expensive than downgrading early.

### Step 3 — Establish the timeline and the channels

**Input.** The incident record and the communication plan.

**Output.** A shared timeline, a technical channel, a clinical channel, and a decision log.

**Rationale.** The timeline is the primary artifact of the response. It records detection,
acknowledgement, each decision, each communication, mitigation, and resolution. The decision log
records why an option was chosen, which is what the review will need.

Separate channels prevent the clinical discussion from being buried in technical logs and the
technical discussion from being interrupted by status requests.

Branch point: if the incident is P1, open a bridge or a war room and assign a scribe. The
commander should not be the scribe.

### Step 4 — Technical triage: service, data, or model

**Input.** The telemetry, the deployment history, and the data-quality signals.

**Output.** A preliminary classification: service failure, data failure, model failure, or
workflow failure.

**Rationale.** The classification determines the first mitigation. A service failure suggests
rollback or failover. A data failure suggests fixing the feed or disabling the affected feature. A
model failure suggests rollback or threshold adjustment. A workflow failure suggests a human
workaround.

The classification is preliminary and may change. The point is to avoid the two common errors:
treating a data failure as a model failure and retraining, or treating a model failure as a
service failure and restarting.

Branch point: if the evidence is ambiguous, check the deployment history first. A recent change is
the most likely cause until proven otherwise.

### Step 5 — Clinical impact assessment

**Input.** The affected population, the decision, and the failure mode.

**Output.** An estimate of patients affected, the direction of the error, and the clinical
consequence.

**Rationale.** The clinical assessment determines the urgency and the communication. A failure that
produces false positives increases alert burden; a failure that produces false negatives misses
patients. The second is usually more dangerous and less visible.

The assessment should be quantitative where possible: number of predictions affected, number of
expected events, change in sensitivity, and the time window. A qualitative statement such as "some
patients may have been affected" is not sufficient for a regulatory assessment.

Branch point: if the direction of the error is unknown, assume the worse direction for the purpose
of urgency and communication. A conservative assumption can be revised; a missed harm cannot.

### Step 6 — Regulatory reportability assessment

**Input.** The clinical assessment, the device status, and the applicable framework.

**Output.** A determination: reportable, not reportable, or undetermined, with a rationale and a
timestamp.

**Rationale.** The reporting clock starts at awareness, not at the end of the investigation. The
assessment must run in parallel with the technical response so that a deadline is not missed while
the team is debugging.

The determination depends on the product's status. A regulated device has reporting obligations; an
internal quality-improvement tool may not. The incident record must state which framework applies
and who made the determination.

Branch point: if the determination is undetermined, treat it as reportable until counsel or
regulatory affairs concludes otherwise. The cost of an unnecessary assessment is small; the cost of
a missed report is not.

### Step 7 — Choose a mitigation

**Input.** The classification, the clinical assessment, and the available options.

**Output.** A chosen mitigation with a rationale, an owner, and a rollback plan for the mitigation
itself.

**Rationale.** The mitigation must match the cause and the clinical consequence. The table below
has more than two outcomes because the choice depends on whether the cause is known, whether a
fallback exists, and whether the model is causing harm or merely inconvenience.

| Situation | Preferred mitigation | Why | Risk of the mitigation |
|---|---|---|---|
| New model version causes the failure | Roll back to the previous digest | Restores known behaviour quickly | Previous model may share a data dependency |
| Model is harmful and no safe version exists | Disable the model, revert to the manual process | Stops harm immediately | Manual process may be understaffed |
| A feature or feed is wrong | Disable the feature or the affected path; fix the feed | Removes the bad input | Model may be invalid without the feature |
| Threshold is causing excess alerts | Raise the threshold temporarily | Reduces burden without a deployment | Changes sensitivity; needs clinical approval |
| Failure is latent and contained | Continue with enhanced monitoring | Avoids a risky change during uncertainty | Harm may continue if the assessment is wrong |
| Workflow misuse | Human workaround and retraining | Fixes the sociotechnical cause | Does not fix the model or the interface |

The mitigation must have its own rollback. A threshold change made under pressure may itself cause
harm and must be reversible.

Branch point: if the cause is unknown, prefer a mitigation that reduces exposure without committing
to a diagnosis. Disabling a feature is more reversible than retraining.

### Step 8 — Execute, verify, and record

**Input.** The chosen mitigation.

**Output.** A verified change with a timestamp, a before-and-after signal, and a rollback path.

**Rationale.** A mitigation is not complete until the effect is verified. Rolling back a deployment
and observing that the error rate falls is verification. Rolling back and assuming it worked is
not. The verification signal should be the same one that detected the incident, plus a clinical
signal if available.

The record must include who authorized the change, especially if it bypassed the normal pipeline.
An emergency change is still a change and needs a retrospective review.

Branch point: if the mitigation does not restore the signal, escalate to the next option. Do not
repeat the same mitigation with a larger parameter.

### Step 9 — Communicate

**Input.** The clinical assessment, the mitigation status, and the audience list.

**Output.** Internal updates, clinical communication, and external communication if required.

**Rationale.** Communication is a safety intervention. Clinicians who do not know the model is
degraded may continue to rely on it. Clinicians who are told to ignore it need a workaround. The
message must state what is known, what is not known, what to do, and when the next update will
come.

The message must not speculate about cause. "We are investigating a possible issue with the
deterioration alert; do not rely on it for the next 24 hours; use the manual early-warning
criteria" is a usable message. "The model is broken" is not.

Branch point: if the incident is P1 or involves potential harm, the clinical communication must be
approved by the clinical safety officer before it is sent.

### Step 10 — Stabilize and monitor

**Input.** The mitigated system and the monitoring plan.

**Output.** A stabilization period with enhanced monitoring and defined exit criteria.

**Rationale.** A system that has been mitigated is not necessarily stable. The fix may be partial,
the underlying cause may persist, and the workaround may create new failure modes. The
stabilization period is when the team watches for recurrence and secondary effects.

Exit criteria should be defined before the period starts: a number of hours without recurrence, a
signal back within its SLO, and a clinical confirmation that the workaround is functioning.

Branch point: if the signal recurs during stabilization, reopen the incident at the original or
higher severity. Do not treat a recurrence as a new incident without linking it.

### Step 11 — Resolve and restore

**Input.** The stabilized system and the verification evidence.

**Output.** A resolution record with the restoration steps and the residual risk.

**Rationale.** Resolution means the system is restored to its intended state, not merely that the
harm has stopped. If the mitigation was a rollback, resolution may require fixing the original
defect and redeploying. If the mitigation was a workaround, resolution requires removing the
workaround and confirming the model is safe.

The residual risk must be stated. A model restored with a patched feed may still have a latent
vulnerability in the feed contract.

Branch point: if the root cause is not fixed, the incident remains open as a problem record even
after the service is restored. Service restoration and problem closure are different states.

### Step 12 — Post-incident review

**Input.** The timeline, the decision log, the technical evidence, and the clinical assessment.

**Output.** A review document with a timeline, contributing factors, what went well, what did not,
and corrective actions.

**Rationale.** The review is where the organization learns. It should be blameless in the sense
that it examines conditions rather than individuals, and it should be specific in the sense that
every action has an owner and a date.

The review should answer: how did we detect, how long did it take, what made it better or worse,
and what would have prevented it or shortened it.

Branch point: if the review finds no contributing factors beyond "human error," the analysis is
incomplete. Human error is a symptom of a condition that allowed the error to reach production.

### Step 13 — Corrective and preventive actions

**Input.** The review's findings.

**Output.** Actions with owners, due dates, and verification criteria, tracked to closure.

**Rationale.** An action without an owner and a date is a wish. The actions should address the
conditions, not only the specific failure: a missing contract test, a monitoring gap, an unclear
runbook, a staffing assumption.

Actions should be prioritized by the risk they reduce, not by the ease of completing them. A
difficult action that removes a P1 cause is worth more than ten easy actions that improve
documentation.

Branch point: if an action cannot be completed by its due date, it must be re-approved with a new
date and a rationale. Silent slippage is how corrective actions disappear.

### Step 14 — Update the system and the organization

**Input.** The completed actions.

**Output.** Updated runbooks, monitoring, tests, training, and regulatory filings.

**Rationale.** The incident should leave the system more robust and the organization more capable.
That means changes to the artifacts that failed: the runbook that was unclear, the monitor that did
not fire, the test that did not exist, the training that did not cover the scenario.

The update should be verified. A runbook that is rewritten but never exercised is a document, not a
capability.

Branch point: if the incident revealed a gap in the regulatory submission, the update may require a
supplement or a new filing. That is a regulatory action, not an engineering one, and it needs its
own owner.

### 5.15 Edge cases the procedure must handle

**Schema drift.** A data field changes during the incident. The mitigation must not depend on the
changed field. If it does, disable the path.

**Late data.** A feed is delayed, and the model is scoring stale data. The clinical communication
must state the staleness, and the workaround must not assume current values.

**Partial rollout.** Only some instances run the bad version. The mitigation must target the
version, and the communication must state which sites are affected.

**Model and feature skew.** The incident is caused by a difference between training and serving.
Rollback may not help if the skew is in a shared dependency. Fix the dependency or disable the
feature.

**Clock skew.** Timestamps in the timeline disagree. Use a single time source for the incident
record, and note the offset in the evidence.

**Quota exhaustion.** The mitigation depends on a service that is also quota-limited. Check the
quota before committing to the mitigation.

**Dependency rotation.** A credential expired during the incident. The fix is a rotation, not a
rollback. Record the rotation in the timeline.

**Regional outage.** A region is unavailable. The mitigation may be a failover, which changes the
population and the monitoring baseline. Communicate the change.

### 5.16 The incident record

The record should be usable by someone who was not there. The minimum fields are:

| Field | Why it matters |
|---|---|
| Identifier and title | Links the record to alerts, tickets, and filings |
| Severity and history | Shows whether the severity was upgraded and when |
| Incident commander and scribe | Assigns accountability and record quality |
| Detection source and timestamp | Distinguishes monitoring from complaint |
| Affected versions and sites | Defines the blast radius |
| Clinical impact estimate | Drives urgency and communication |
| Regulatory determination and timestamp | Starts and documents the reporting clock |
| Timeline of decisions | Explains why each mitigation was chosen |
| Mitigation and verification | Shows what was done and whether it worked |
| Communications log | Shows who was told what, and when |
| Resolution and residual risk | Distinguishes mitigation from restoration |
| Review and corrective actions | Links the incident to the changes that follow |

A record with these fields can support a regulatory filing, an internal audit, and a review. A
record that is a chat thread cannot.

The record should be immutable after closure, with addenda for later findings. Editing the
timeline to make the response look faster is the incident-response equivalent of obscuring an audit
trail.

## 6. Worked example

A hospital's deterioration service scores 1,150 admissions per day at a threshold of 0.18. The
outcome is deterioration within 12 hours, and the monthly audit sample is used to estimate
sensitivity. In March, the service's alert volume falls by 30 percent over two weeks. The ward
reports that the list is "quieter," and the team initially treats that as good news.

### 6.1 The signals

| Signal | February | March | Change |
|---|---:|---:|---:|
| Admissions per day | 1,150 | 1,150 | 0 |
| Alerts per day | 93.9 | 65.8 | −30 percent |
| Mean predicted risk | 0.205 | 0.168 | −18 percent |
| Creatinine missingness | 0.4 percent | 11.2 percent | +10.8 points |
| Creatinine p95 arrival delay | 22 minutes | 3 hours 40 minutes | +3 h 18 min |
| Audit sensitivity (n = 150) | 0.63 | 0.51 | −12 points |

The service-layer SLOs are green. Availability is 99.6 percent, latency p99 is 810 ms, and the
error rate is 0.1 percent. The only signal that is abnormal is the alert volume, and it is abnormal
in the direction that looks like an improvement.

### 6.2 The clinical exposure

At 3.8 percent prevalence, a 30-day month contains:

```text
events = 1,150 × 30 × 0.038 = 1,311
```

At sensitivity 0.63, the model detects 826 and misses 485. At sensitivity 0.51, it detects 669 and
misses 642. The difference is 157 additional missed events per month, or about 5.2 per day.

The audit detected the change 24 days after the interface upgrade. During those 24 days:

```text
predictions = 1,150 × 24 = 27,600
events      = 27,600 × 0.038 = 1,049
excess missed = 1,049 × (0.49 − 0.37) = 1,049 × 0.12 = 126
```

About 126 deteriorations were missed that would have been detected at the previous sensitivity.
The number is an estimate, not a count of identified patients; the audit sample is the evidence,
and the count is the scale of the exposure.

### 6.3 The timeline

| Time | Event | Track |
|---|---|---|
| Day 0 | Laboratory interface upgrade | Upstream |
| Day 9 | Alert volume 25 percent below baseline; team notes it in a standup | Detection (informal) |
| Day 24 | Monthly audit finds sensitivity 0.51; incident opened at P2 | Detection (formal) |
| Day 24, +4 h | Incident commander assigned; clinical impact assessed | Response |
| Day 24, +6 h | Regulatory assessment opened; device status confirmed | Regulatory |
| Day 25 | Feature freshness query confirms creatinine p95 age of 3 h 40 min | Diagnosis |
| Day 25 | Interface rollback approved and executed; backfill started | Mitigation |
| Day 26 | Freshness returns to 18 minutes; audit sample re-run | Verification |
| Day 27 | Clinical communication sent; workaround confirmed | Communication |
| Day 30 | Incident resolved; problem record remains open for the contract test | Resolution |
| Day 38 | Post-incident review complete; four corrective actions assigned | Review |

The mean time to detect is 24 days. The mean time to mitigate after detection is one day. The
detection delay dominates the exposure by a factor of 24. Any investment in response speed that
does not address detection is optimizing the wrong stage.

### 6.4 Diagnostic one: the input, not the model

The first diagnostic is the feature-age panel segmented by feature. Creatinine p95 age has risen
from 22 minutes to 3 hours 40 minutes. The model is returning predictions in 80 ms from a feature
that is hours old.

The second diagnostic is the missingness rate. It rose from 0.4 percent to 11.2 percent because the
interface delivers the result after the model's feature cutoff. The model imputes the cohort median
for missing creatinine, and the median is lower than the values of the patients who would have
triggered.

The evidence points to the interface, not the model. The model's coefficients are unchanged, its
input schema is unchanged, and its output is consistent with the imputed values.

### 6.5 Diagnostic two: the audit sample

The audit sample of 150 admissions found sensitivity 0.51 with a 95 percent confidence interval of
0.43 to 0.59. The previous month's estimate was 0.63 with an interval of 0.55 to 0.71. The
intervals do not overlap, which is consistent with a real change rather than sampling noise.

The audit also found that the missed cases were concentrated among patients whose creatinine
arrived after the cutoff. That is a mechanistic confirmation, not just a statistical one.

### 6.6 The response and its cost

The chosen mitigation was to roll back the interface change and backfill the missing values. The
model was not retrained. Retraining on the degraded feed would have taught the model to compensate
for missing creatinine, embedding the defect and making the model worse when the interface was
fixed.

The cost of the incident:

| Item | Quantity | Unit cost | Total |
|---|---:|---:|---:|
| Engineering response | 3 engineers × 36 h | 95 USD/h | 10,260 USD |
| Clinical review | 16 h | 120 USD/h | 1,920 USD |
| Legal and communications | 1 engagement | 4,500 USD | 4,500 USD |
| **Total** | | | **16,680 USD** |

The cost of the corrective actions was about 60 engineer-hours, or 5,700 USD, mostly for the
contract test and the freshness monitor. That is 34 percent of the incident cost, and it addresses
the detection gap that caused the exposure.

### 6.7 The regulatory assessment

The device status determines the obligation. If the model is part of a regulated device, the
manufacturer must assess whether the failure may have contributed to a death or serious injury. The
assessment started on day 24, when the team became aware, not on day 30 when the cause was
confirmed.

Two clocks matter. A 30-calendar-day report applies to a death or serious injury that the device
may have caused or contributed to. A 5-work-day report applies in specified circumstances,
including when remedial action is needed to prevent an unreasonable risk of substantial harm. The
interface rollback was remedial action, so the 5-day assessment was the relevant one.

The determination was "not reportable" because the missed events could not be linked to a specific
death or serious injury, and the failure was corrected within the assessment window. The rationale
and the timestamp were recorded. A different fact pattern, such as a confirmed missed deterioration
that led to an ICU transfer, would have produced a different determination.

### 6.8 Vary one assumption: detection at day 10

Suppose the freshness monitor had been in place and detected the interface change at day 10 instead
of day 24.

```text
predictions = 1,150 × 10 = 11,500
events      = 11,500 × 0.038 = 437
excess missed = 437 × 0.12 = 52
```

The exposure falls from 126 excess missed events to 52, a 58 percent reduction. The incident cost
falls because the engineering response is shorter and the legal review is smaller. The regulatory
assessment still runs, but the blast radius is smaller and the communication is simpler.

The lesson is that the freshness monitor, which cost about 20 engineer-hours to build, would have
paid for itself in this incident alone. That is the argument for detection investment: it is
cheaper than response.

### 6.9 What the example establishes

The service was healthy and the model was failing. The detection came from an audit, not from an
SLO. The diagnosis came from a feature-age panel and a missingness check, not from a model metric.
The response was a pipeline fix, not a retrain.

The timeline shows that detection dominated the exposure. A team that measures only mean time to
resolve will miss this. The metric that matters is time to detect, and the investment that improves
it is monitoring, not response tooling.

### 6.10 The corrective actions

| Action | Type | Owner | Due | Verification |
|---|---|---|---|---|
| Contract test on creatinine arrival time | Preventive | Data platform | 3 weeks | Test fails when the p95 exceeds 30 minutes |
| Feature-freshness SLO and alert | Detection | SRE and data science | 2 weeks | Alert fires in a synthetic delay test |
| Backfill and re-run of the affected window | Corrective | Data platform | 1 week | Audit sensitivity restored above 0.60 |
| Clinician communication and audit of missed cases | Corrective | Clinical safety officer | 2 weeks | Documented review of the 126 estimated misses |

The first two actions address detection and prevention; the last two address the harm that already
occurred. A review that produces only the first two has learned how to catch the next incident but
not how to close this one.

## 7. Code, query, or calculation

The script implements the severity classifier, the exposure calculation, and the response-time
metrics from Section 6. The severity rules are deliberately simple and should be replaced with the
organization's matrix; the point is that severity is a function of explicit inputs, not a feeling
in the room.

```python
"""Incident severity, exposure, and response metrics. Standard library only."""


def severity(serious_harm, broad_scope, workaround):
    if serious_harm and not workaround:
        return "P1"
    if serious_harm and workaround:
        return "P2"
    if broad_scope and not serious_harm:
        return "P3"
    return "P4"


def exposure(predictions_per_day, days, prevalence, sens_before, sens_after):
    predictions = predictions_per_day * days
    events = predictions * prevalence
    missed_before = events * (1 - sens_before)
    missed_after = events * (1 - sens_after)
    return predictions, events, missed_after - missed_before


def main() -> None:
    cases = [
        ("no alerts during rounds", True, True, False),
        ("sensitivity down at one site", True, False, True),
        ("latency above SLO", False, True, True),
        ("calibration trend", False, False, True),
    ]
    for name, harm, broad, workaround in cases:
        print(f"{name:32s} {severity(harm, broad, workaround)}")

    p, e, extra = exposure(1_150, 24, 0.038, 0.63, 0.51)
    print(f"predictions_exposed   {p:.0f}")
    print(f"events_in_window      {e:.0f}")
    print(f"excess_missed         {extra:.0f}")

    p10, e10, extra10 = exposure(1_150, 10, 0.038, 0.63, 0.51)
    print(f"excess_missed_day10   {extra10:.0f}")
    print(f"reduction_pct         {(extra - extra10) / extra * 100:.0f}")

    detect_h, mitigate_h, resolve_h = 24 * 24, 24, 72
    print(f"mttd_hours            {detect_h}")
    print(f"mttr_hours            {mitigate_h + resolve_h}")

    eng, clinical, legal = 3 * 36 * 95, 16 * 120, 4_500
    print(f"incident_cost         {eng + clinical + legal} USD")

    assert severity(True, True, False) == "P1"
    assert severity(True, False, True) == "P2"
    assert severity(False, True, True) == "P3"
    assert abs(extra - 126) < 1
    assert abs(extra10 - 52) < 1
    assert eng + clinical + legal == 16_680


if __name__ == "__main__":
    main()
```

Expected output, confirmed on Python 3.11:

```text
no alerts during rounds          P1
sensitivity down at one site     P2
latency above SLO                P3
calibration trend                P4
predictions_exposed   27600
events_in_window      1049
excess_missed         126
excess_missed_day10   52
reduction_pct         58
mttd_hours            576
mttr_hours            96
incident_cost         16680 USD
```

The severity outputs show the intended ordering. A failure with serious harm and no workaround is
P1. The same harm with a manual workaround is P2. A broad but non-harmful degradation is P3. A
trend is P4.

The exposure calculation is the number that should drive the urgency, not the alert count. A
30 percent fall in alerts sounds like an improvement; 126 excess missed events in 24 days does not.
The day-10 scenario shows that the detection investment, not the response speed, is the lever: the
same response one day after detection produces a much smaller exposure.

The response-time metrics separate detection from mitigation. The MTTD of 576 hours and the MTTR of
96 hours point to different investments. Reducing MTTR by half would save two days of exposure;
reducing MTTD by half would save twelve. A team that reports a single mean time to resolve hides
this.

The reference service's `RUNBOOK.md` lists detection signals, triage steps, rollback commands, a
communication section, and a post-incident section. It does not include a severity matrix, an
incident commander role, or a regulatory assessment step. Those are the additions a clinical
deployment needs.

## 8. Data, standards, and regulatory context

Incident response sits at the intersection of security practice, quality management, and device
regulation. The instruments below determine the process, the record, and the deadlines.

### 8.1 The instruments

| Instrument | Status | What it requires | What it does not specify |
|---|---|---|---|
| NIST SP 800-61 Rev. 3, 2025 | Voluntary; widely adopted | Incident response as part of risk management, organized around the CSF 2.0 functions | A clinical severity matrix |
| 21 CFR Part 803 | Binding for device manufacturers, importers, and user facilities in the US | Reporting of deaths, serious injuries, and certain malfunctions within stated deadlines | The internal incident process |
| FDA recall and field correction authority | Binding | Correction or removal of a device that violates the law or presents a risk | The decision criteria for a voluntary recall |
| ISO 14971:2019 clause 10 | Binding when invoked | Production and post-production monitoring, including incident review | Timelines or severity levels |
| IEC 62304:2006/Amd 1:2015 clause 9 | Binding when invoked | Problem resolution, change control, and test documentation | Incident command structure |
| ISO/IEC 42001:2023 | Voluntary, certifiable | Operational control, incident handling, continual improvement | Clinical triage |
| HIPAA breach notification, 45 CFR 164.400-414 | Binding for covered entities | Notification of a breach of unsecured protected health information | Model incident handling |
| EU MDR Article 87 | Binding in the EU | Vigilance reporting of serious incidents and field safety corrective actions | Internal response process |

### 8.2 What the FDA reporting rule requires

Under 21 CFR 803.50, a manufacturer must submit a report within 30 calendar days after becoming
aware of information that reasonably suggests a marketed device may have caused or contributed to a
death or serious injury, or has malfunctioned and would be likely to cause or contribute to a death
or serious injury if the malfunction recurred.

A 5-work-day report applies under 21 CFR 803.53 in specified circumstances, including when an
event requires remedial action to prevent an unreasonable risk of substantial harm to public
health, or when the FDA makes a written request. User facilities report deaths and serious injuries
under 21 CFR 803.30 within 10 working days.

"Becoming aware" is defined broadly: any employee acquiring the information. An incident response
process that waits for the root-cause analysis before assessing reportability will miss the
deadline. The assessment must be a parallel track, and the timestamp of awareness must be recorded.

### 8.3 What NIST SP 800-61 Rev. 3 changes

The 2025 revision reframes incident response as part of cybersecurity risk management and
organizes it around the CSF 2.0 functions: Govern, Identify, Protect, Detect, Respond, Recover. It
emphasizes risk-based triage, automated containment with manual override, continuous monitoring,
and lessons learned fed back in near real time rather than only after recovery.

The practical effect for a model service is that incident response is not a separate plan bolted
onto operations. It is part of the same risk management system that governs the model, the data,
and the clinical workflow. The severity matrix, the reporting assessment, and the corrective
actions should be linked to the risk file.

### 8.4 What ISO 14971 and IEC 62304 add

ISO 14971 clause 10 requires monitoring of production and post-production information and an
assessment of whether risk controls remain effective. An incident is the strongest form of
post-production information. The review must ask whether the risk analysis anticipated the failure,
and if not, why not.

IEC 62304 clause 9 requires a problem resolution process: investigate, evaluate, and decide whether
a change is needed, with change control and test documentation. The incident review's corrective
actions are the input to that process.

### 8.5 Privacy and security incidents

A model incident can also be a privacy incident. If the failure exposed protected health
information, the HIPAA breach notification rule applies, with its own timeline and content
requirements. If the failure was caused by a security compromise, the NIST process and the
organization's security incident plan apply alongside the clinical process.

The tracks must be coordinated. A privacy investigation may require preserving logs that the
technical team wants to rotate; a security investigation may require isolating a system that the
clinical team needs for a workaround.

### 8.6 What is convention

- A blameless post-incident review. It is a cultural practice with strong evidence in software
  operations, not a regulatory requirement.
- An incident commander role. It is a practice from incident management frameworks, not a standard.
- A 24-hour update cadence for P1 incidents. It is an operational choice.
- A severity scale of P1 to P4. The labels vary; the mapping to response does not.
- A status page. It is a communication tool, not a control.

> **Note:** A reportability determination is a legal and regulatory judgment. The engineering team
> supplies the facts and the timeline; regulatory affairs or counsel makes the determination. This
> topic describes the process, not the legal standard for any particular product.

### 8.7 The reportability worksheet

A reportability assessment should be a worksheet with fixed fields, not a conversation. The minimum
fields are: the date of awareness and who became aware; the device and version; the event
description; the patient outcome; the evidence linking the device to the outcome; whether remedial
action was taken; the applicable framework; the determination; the determiner; and the date of the
determination.

The worksheet makes the assessment reproducible and auditable. It also makes the clock visible: the
difference between the awareness date and the determination date is the time the organization spent
deciding, and it counts against the deadline.

## 9. Industry practice and case evidence

### 9.1 Therac-25

The Therac-25 was a radiation therapy device that produced several overdoses in the 1980s, with
deaths and serious injuries. The investigations by Leveson and Turner documented a combination of
software defects, removed hardware safety interlocks, inadequate testing, and a response process
that initially denied the problem.

The documented case is a foundational example of software safety. The inference for model incident
response is that the response itself can compound the harm: denial, slow acknowledgement, and
failure to communicate with users turned individual accidents into a pattern. The organizational
conditions, not only the code defect, were the cause.

### 9.2 The Epic Sepsis Model

Wong and colleagues' external validation found that the Epic Sepsis Model had an AUC of 0.63 at
Michigan Medicine, against 0.76 to 0.83 reported by the vendor, and fired alerts for 18 percent of
hospitalizations. The finding was published in 2021, after the model had been in use.

The documented case is a detection gap at the level of the deployment, not a single incident. The
inference is that the absence of an incident is not evidence of safety. A model can be failing
quietly for years without generating a report, because no process is looking for the failure.

### 9.3 The FDA's MDR data

The FDA publishes MDR data and device recall information. The data show that software and device
failures are reported continuously, and that reporting is a routine part of device operations, not
an exceptional event.

The inference for a clinical AI team is that a reporting capability should be designed before it is
needed. A team that has never assessed reportability will struggle to do so under the pressure of a
real incident.

### 9.4 Blameless post-incident reviews

Google's SRE practice documents blameless postmortem reviews as a way to learn from incidents
without suppressing information. The practice distinguishes the individual from the system and
focuses on the conditions that made the error likely.

The documented practice is from software operations. The inference for clinical AI is that the same
logic applies, with an additional stakeholder: the clinical safety officer, who needs the review to
feed the risk file and the quality system, not only the engineering backlog.

### 9.5 What the evidence does not show

There is little public evidence on the frequency of clinical AI incidents, the distribution of
severity, or the effectiveness of specific response practices. The case literature is dominated by
high-severity events that became public. The base rate of silent failures is unknown.

A team should treat its own incident log as the primary evidence and should record near misses as
well as harms. A near miss is a free lesson.

### 9.6 Near misses

A near miss is an event that could have caused harm but did not, because of a workaround, a
coincidence, or a clinician's judgment. Near misses are the cheapest source of safety information
an organization has, and they are usually underreported because nothing happened.

A model near-miss register should capture: what almost happened, which signal would have caught it,
and what condition prevented the harm. A near miss that is reported and analysed can produce a
control before the harm occurs. A near miss that is not reported is a rehearsal for the real event.

A useful metric is the ratio of near misses to harms. A programme that reports many near misses
and few harms is learning; a programme that reports neither is not looking. The ratio is not a
target to optimize directly, because suppressing near-miss reports would improve it.

The register also gives the quality team a leading indicator for the risk file, which is
otherwise updated only after harm.



## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Slow detection | Time to detect is measured in weeks | Only service-layer signals are monitored | Prolonged patient exposure | Add data and model signals; measure MTTD |
| No incident commander | Multiple people issuing fixes | No single decision-maker | Conflicting actions and messages | Assign a commander at open |
| Parallel conflicting fixes | Two changes deployed within minutes | No decision log or change freeze | One fix undoes the other | Single commander and a change freeze |
| Rollback fails | The previous digest is unavailable | Retention policy or incompatible schema | The bad version stays live | Test rollback quarterly; retain artifacts |
| Fallback is worse | The workaround causes new errors | The fallback was never exercised | Harm continues under a different cause | Drill the fallback; define its own stop criteria |
| Communication delayed | Clinicians learn from colleagues | No pre-approved message template | Distrust and continued reliance on a broken tool | Template, approval path, and a status channel |
| Regulatory clock missed | Awareness date is unclear | The assessment starts after root cause | Late or missing report | Parallel regulatory track; record awareness time |
| Severity undergraded | P3 opened for a harm event | Severity assigned by the reporter's role | Slow response and no clinical involvement | Severity matrix and a default to higher |
| Root cause is "human error" | The review ends at the individual | No systems analysis | The condition persists | Ask what allowed the error to reach production |
| Actions unowned | A list of recommendations | No owner or date | Nothing changes | Actions with owner, date, and verification |
| Blame-oriented review | People are defensive | Fear of consequences | Information is suppressed | Blameless framing with accountability for actions |
| Incident closed at mitigation | Problem record absent | Mitigation confused with resolution | The latent cause recurs | Separate service restoration from problem closure |
| Logs insufficient | The timeline cannot be reconstructed | Retention or sampling too aggressive | No root cause, no regulatory evidence | Retention aligned with audit period |
| PHI in the incident channel | Identifiers pasted into chat | Urgency and convenience | Privacy breach during a safety event | Redaction rules and a secure evidence channel |

Red flags: an incident record that is a chat thread; a severity assigned after the fix; a
regulatory assessment with no timestamp; a review with no "what went well" section; a corrective
action that cannot be verified.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test |
|---|---|---|---|
| The detection signal is timely | Exposure depends on time to detect | Only service metrics are monitored | Measure MTTD by incident class |
| The incident commander has authority | Decisions must be made quickly | The commander is junior or absent | Review decision latency in drills |
| The fallback is usable | Mitigation may depend on it | The manual process is understaffed | Drill the fallback with clinicians |
| The previous artifact is available | Rollback depends on it | Retention or registry policy | Pull the previous digest monthly |
| The regulatory framework is known | Reportability depends on it | Product status unclear | Confirm device status before launch |
| The timeline is reconstructable | Review and filings depend on it | Logs rotated or sampled | Reconstruct a six-month-old incident |
| Clinical input is available | Impact assessment needs it | No clinical safety officer on call | Include clinical on-call in the roster |
| Corrective actions are resourced | Recurrence prevention depends on it | No owner or budget | Track closure rate and slippage |
| The review happens while evidence is fresh | Memory and logs degrade over time | Review delayed by weeks | Schedule the review within five business days |

## 12. Exercises

### Exercise 12.1 — Classify severity

Assign P1 to P4 and justify: (a) the prediction application programming interface (API) is down during morning rounds with no manual fallback; (b) sensitivity at one site has fallen 8 points and a manual checklist exists; (c) p99 latency is 1.4 seconds against a 1.2-second SLO but no decisions are missed; (d) calibration slope is 0.86 against a 0.85 floor.

**Deliverable.** A four-row table with severity, impact, workaround, and justification.

**Acceptance criteria.** Severity follows harm and workaround, not metric size; the ambiguous case is assigned the higher severity; the justification names the affected population.

**Time.** 25 minutes.

### Exercise 12.2 — Compute exposure and response metrics

A service scores 800 patients per day. Sensitivity fell from 0.70 to 0.58, and the incident was detected after 18 days at 5 percent prevalence. Compute predictions exposed, events in the window, excess missed events, MTTD in hours, and MTTR if mitigation took 6 hours and resolution took 42 hours.

**Deliverable.** Five numbers with the arithmetic.

**Acceptance criteria.** Exposure uses the full window; excess missed computed as the difference in missed fractions; MTTD and MTTR reported separately.

**Time.** 30 minutes.

### Exercise 12.3 — Write a timeline

Given the scenario in 12.2, write a timeline with detection, acknowledgement, mitigation, resolution, and review. Include the four parallel tracks and at least one clinical and one regulatory event.

**Deliverable.** A timeline table with timestamps and tracks.

**Acceptance criteria.** Tracks are parallel, not sequential; the awareness time is recorded; the review is after resolution; no track is empty.

**Time.** 35 minutes.

### Exercise 12.4 — Assess reportability

For a regulated device, a model failure may have contributed to a missed deterioration that led to an unplanned ICU transfer. The fix required a configuration change. Determine the likely reporting path and the clock start. State what facts would change the determination.

**Deliverable.** A 300-word assessment with a decision and two alternative fact patterns.

**Acceptance criteria.** Distinguishes 30-day and 5-day reports; identifies the awareness date; does not assert a legal conclusion; states the role that makes the determination.

**Time.** 35 minutes.

### Exercise 12.5 — Write a clinical communication

Draft the message to clinicians for a P2 incident in which the model is unreliable for 48 hours and a manual checklist is the workaround.

**Deliverable.** A message of at most 200 words.

**Acceptance criteria.** States what is known and unknown; gives a concrete action; states the next update time; does not speculate about cause; is approved by the clinical safety officer in the scenario.

**Time.** 30 minutes.

### Exercise 12.6 — Critique an incident review

A review says: "The engineer deployed the wrong model. Corrective action: remind the team to be careful." Write a critique and propose three better actions.

**Deliverable.** A 350-word critique with three actions.

**Acceptance criteria.** Identifies the missing system condition; actions are controls, not reminders; each action has an owner and a verification method.

**Time.** 30 minutes.

### Exercise 12.7 — Design a drill

Design a quarterly drill for a model incident. Include the scenario, the participants, the injects, the success criteria, and the artifacts produced.

**Deliverable.** A two-page drill plan.

**Acceptance criteria.** Includes a clinical and a regulatory inject; measures time to detect and mitigate; produces a review and actions; is safe to run in production or uses a staging environment.

**Time.** 50 minutes.

### Exercise 12.8 — Design a mitigation table

Write a mitigation decision table with at least six situations and the preferred action, the risk of the action, and the authority required.

**Deliverable.** A six-row table.

**Acceptance criteria.** Includes rollback, disable, degrade, adjust, and monitor; each action has its own rollback; authority is named.

**Time.** 40 minutes.

## 13. Self-check answers

### 13.1 Exercise 12.1

(a) P1. Serious harm is plausible, the scope is broad, and there is no workaround.

(b) P2. Harm is plausible, the scope is limited, and a manual checklist exists.

(c) P3. The SLO is breached but no decisions are missed. It is an operational degradation.

(d) P4. The calibration is above the floor; it is a trend, not a breach.

### 13.2 Exercise 12.2

```text
predictions = 800 × 18 = 14,400
events      = 14,400 × 0.05 = 720
missed_before = 720 × 0.30 = 216
missed_after  = 720 × 0.42 = 302.4
excess_missed = 302.4 − 216 = 86.4
MTTD = 18 × 24 = 432 hours
MTTR = 6 + 42 = 48 hours
```

About 86 excess missed events. The MTTD is nine times the MTTR, so the detection stage dominates
the exposure.

### 13.3 Exercise 12.3

A complete timeline records the detection at day 18, the acknowledgement within two hours, the
clinical impact assessment and regulatory assessment in parallel on the same day, the mitigation
on day 19, the resolution on day 21, and the review by day 30. Each entry has a timestamp, an
owner, and a track.

### 13.4 Exercise 12.4

The awareness date is the date any employee first had information reasonably suggesting the device
may have contributed to a serious injury. For a manufacturer, the likely path is a 30-calendar-day
report; a 5-work-day report applies if remedial action was needed to prevent an unreasonable risk
of substantial harm. A user facility would have a 10-working-day obligation for a serious injury.

Facts that would change the determination include a confirmed causal link, a death rather than an
injury, evidence that the failure could recur before correction, or a correction that itself
requires field action. The determination is made by regulatory affairs or counsel, not by the
engineering team.

### 13.5 Exercise 12.5

A strong message: "The deterioration alert is not reliable for the next 48 hours because of a data
feed issue. Do not use the alert to rule out deterioration. Use the manual early-warning checklist
for all admitted patients until you receive the all-clear. We will update you at 18:00 today and
will confirm when the alert is restored. If you have concerns about a specific patient, escalate
through the usual rapid-response pathway."

The message states the status, the action, the duration, the next update, and the escalation path.
It does not speculate about cause or blame.

### 13.6 Exercise 12.6

The review stops at the individual and prescribes a reminder, which is not a control. The missing
condition is that the deployment process allowed a model to reach production without a check that
matched the approved digest and version.

Three better actions: enforce artifact promotion by digest with an admission check; add a
pre-deployment verification that the model version matches the change record; add a canary stage
with an automatic halt on a version mismatch. Each has an owner and a verification method.

### 13.7 Rubric for the design exercises

| Criterion | Weight | Novice | Competent | Expert |
|---|---:|---|---|---|
| Severity | 20% | Metric size | Harm and workaround | Quantified exposure and urgency |
| Parallel tracks | 25% | Sequential | Clinical and technical | All four tracks with named owners |
| Communication | 20% | Technical detail | Action and timing | Pre-approved, audience-specific, tested |
| Regulatory | 20% | Ignored | Assessed | Clock, determination, and record |
| Learning | 15% | Recommendations | Actions with owners | Verified controls and drill evidence |

### 13.8 Exercise 12.8

A strong table includes: new version causing harm → roll back, risk that the previous model shares
the dependency, authority of the incident commander; harmful model with no safe version → disable
and revert to manual, risk of understaffing, authority of the clinical safety officer; wrong feed →
disable the feature or path, risk of model invalidity, authority of the service owner; excess alerts
→ raise threshold temporarily, risk of reduced sensitivity, authority of the clinical owner;
latent contained failure → continue with enhanced monitoring, risk of undetected harm, authority of
the commander with clinical concurrence; workflow misuse → human workaround and retraining, risk
that the model remains misaligned, authority of the clinical owner.

## 14. Interview questions

**1. What makes a model incident different from a service incident?**
Strong answers say silent, clinical, and regulatory, and give an example where the service is
healthy. Weak answers describe an outage.

**2. Who is in charge during an incident?**
Strong answers name an incident commander with decision authority and separate technical, clinical,
regulatory, and communication tracks. Weak answers say the most senior engineer.

**3. How do you decide whether to roll back or disable?**
Strong answers weigh the cause, the fallback, and the harm; rollback restores known behaviour,
disable stops harm when no safe version exists. Weak answers default to rollback.

**4. When does the regulatory clock start?**
Strong answers say at awareness, not at root cause, and describe a parallel assessment. Weak
answers say after the investigation.

**5. What is a blameless review?**
Strong answers focus on conditions and accountability for actions, not individual fault. Weak
answers say no one is accountable.

**6. What metrics do you track for incident response?**
Strong answers separate time to detect, acknowledge, mitigate, and resolve. Weak answers use a
single MTTR.

**7. How do you communicate with clinicians during an incident?**
Strong answers give status, action, duration, next update, and escalation, with approval. Weak
answers send technical details.

**8. How do you know your fallback works?**
Strong answers say it is drilled with clinicians and has its own stop criteria. Weak answers assume
the manual process is available.

**9. What is a near miss and why record it?**
Strong answers say an event that could have caused harm but did not, and it is free evidence. Weak
answers dismiss it because nothing happened.

**10. How do you prevent recurrence?**
Strong answers require corrective actions with owners, dates, verification, and a link to the risk
file. Weak answers say retrain the team.

## 15. Key terms

- **Acknowledgement** — a human takes ownership of an incident.
- **Blast radius** — the scope of affected users or patients.
- **Blameless review** — a review focused on conditions rather than individual fault.
- **CAPA** — corrective and preventive action.
- **Clinical safety officer** — the role accountable for clinical risk management.
- **Containment** — limiting the blast radius of an incident.
- **Contributing factor** — a condition that increased the likelihood or impact.
- **Corrective action** — a change that reduces recurrence or impact.
- **Degradation** — deliberately reducing functionality to protect safety.
- **Detection** — the first signal of an incident.
- **Failover** — switching to a standby system.
- **Field safety notice** — a communication to customers about a safety issue.
- **Incident** — an unplanned event that degrades a service, model, or workflow safety.
- **Incident commander** — the person accountable for the response.
- **MDR** — Medical Device Report to the FDA.
- **Mitigation** — reduction or stopping of the harmful effect.
- **MTTD** — mean time to detect.
- **MTTR** — mean time to resolve.
- **Near miss** — an event that could have caused harm but did not.
- **Playbook** — a procedure for a class of incidents.
- **Post-incident review** — a structured review of cause and response.
- **Recall** — removal or correction of a device in the field.
- **Reportable event** — an event meeting a regulatory reporting criterion.
- **Resolution** — restoration and verification of the system.
- **Rollback** — restoring the previous known-good artifact.
- **Root cause** — the condition that, if removed, would have prevented the incident.
- **Runbook** — the procedure for a known alert.
- **Severity** — a classification of impact, urgency, and workaround.
- **Status page** — the external communication channel.
- **Tabletop** — a discussion-based exercise.
- **Triage** — rapid assessment of severity and cause.
- **Workaround** — a manual process that substitutes for the model.

## 16. Sources

All URLs accessed 26 October 2026.

- NIST. *Incident Response Recommendations and Considerations for Cybersecurity Risk Management: A CSF 2.0 Community Profile*, SP 800-61 Rev. 3, April 2025. https://csrc.nist.gov/pubs/sp/800/61/r3/final
- NIST. *SP 800-61r3 PDF*. https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-61r3.pdf
- 21 CFR Part 803, Medical Device Reporting. https://www.ecfr.gov/current/title-21/chapter-I/subchapter-H/part-803
- 21 CFR 803.50, Manufacturer reporting requirements. https://www.ecfr.gov/current/title-21/chapter-I/subchapter-H/part-803/subpart-E/section-803.50
- 21 CFR 803.53, 5-day reports. https://www.ecfr.gov/current/title-21/chapter-I/subchapter-H/part-803/subpart-E/section-803.53
- US Food and Drug Administration. *Mandatory Reporting Requirements*. https://www.fda.gov/medical-devices/postmarket-requirements-devices/mandatory-reporting-requirements-manufacturers-importers-and-device-user-facilities
- ISO 14971:2019. *Medical devices — Application of risk management to medical devices*. https://www.iso.org/standard/72704.html
- IEC 62304:2006/AMD1:2015. *Medical device software — Software life cycle processes*. https://webstore.iec.ch/en/publication/22794
- ISO/IEC 42001:2023. *Information technology — Artificial intelligence — Management system*. https://www.iso.org/standard/42001
- 45 CFR 164.400-414, HIPAA breach notification. https://www.ecfr.gov/current/title-45/subtitle-A/subchapter-C/part-164/subpart-D
- Leveson NG, Turner CS. An investigation of the Therac-25 accidents. *Computer*. 1993;26(7):18-41. https://doi.org/10.1109/MC.1993.274940
- Wong A, Otles E, Donnelly JP, et al. External validation of a widely implemented proprietary sepsis prediction model in hospitalized patients. *JAMA Internal Medicine*. 2021;181(8):1065-1070. https://doi.org/10.1001/jamainternmed.2021.2626
- Google. *Site Reliability Engineering*, Chapter 15, Postmortem Culture. https://sre.google/sre-book/postmortem-culture/
- US Food and Drug Administration. *Recalls, Corrections and Removals*. https://www.fda.gov/medical-devices/postmarket-requirements-devices/recalls-corrections-and-removals-devices

## 17. Links

- Module index: [`_index.md`](_index.md)
- Survey file: [`../../course/15_deployment_mlops.md`](../../course/15_deployment_mlops.md)
- Previous topic: [`06_drift.md`](06_drift.md)
- Next topic: [`08_rollout-and-rollback.md`](08_rollout-and-rollback.md)
- Related topic: [`05_monitoring.md`](05_monitoring.md)
- Related topic: [`09_model-governance.md`](09_model-governance.md)
- Labs overview: [`../../labs/`](../../labs/)
- Reference service: [`../../labs/mlops/`](../../labs/mlops/)
- Runbook: [`../../labs/mlops/RUNBOOK.md`](../../labs/mlops/RUNBOOK.md)
- Model card: [`../../labs/mlops/model_card.md`](../../labs/mlops/model_card.md)
