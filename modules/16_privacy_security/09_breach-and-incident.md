# Breach and Incident

> **Module 16 - Privacy, Security, and Ethics**
> Topic 09 of 10
> Estimated study time: 3 hours
> Prerequisites: 04, 05
> Version 2026.3

## 1. Why this topic exists

Every control in this module is designed to prevent an incident. This topic assumes the controls
failed. A credential was stolen, a laptop was lost, a vendor was compromised, or a misconfigured
bucket was indexed by a search engine. What happens in the next 72 hours determines whether the
organisation contains the harm, meets its legal deadlines, keeps the trust of its patients, and
learns something that changes the next design.

The deadlines are short and unforgiving. Under the HIPAA Breach Notification Rule, notification is
due without unreasonable delay and no later than 60 calendar days after discovery, and discovery
includes the day the breach would have been known with reasonable diligence. Under the GDPR,
notification to the supervisory authority is due within 72 hours of becoming aware, with individual
notification without undue delay where the risk is high. A organisation that spends the first week
deciding who owns the response has already consumed a fifth of the HIPAA window and all of the
GDPR window.

Three failure patterns recur.

**Detection without response.** The security team sees an alert, opens a ticket, and the ticket
ages. There is no declared incident, no incident commander, and no clock. The organisation learns
about the scope weeks later, when a patient or a journalist calls. The technical detection worked;
the response system did not.

**Notification as a legal exercise.** The legal team writes the letter, the communications team
writes the statement, and neither has the facts. The notification says "we take privacy seriously"
and does not say what data, which people, what the patient should do, or when the organisation knew.
The regulator's first question is about the timeline, and the answer is a reconstruction.

**Lessons without change.** The post-incident report identifies the root cause, the recommendations
are accepted, and nothing is funded. The same control fails again in the next audit or the next
incident. A lesson that does not change a budget, a design, or a test is a document, not a lesson.

This topic teaches incident response as a lifecycle with legal clocks, evidence handling, and
communication. It covers the HIPAA and GDPR notification rules, the breach risk assessment, the
role of encryption and the safe harbour, the vendor chain, and the post-incident review that closes
the loop with the threat model in Topic 04 and the governance in Topic 08.

The distinction that runs through the topic is between an incident (a security event that has or
may have an effect) and a breach (a use or disclosure not permitted by law). Not every incident is
a breach, and not every breach is a security incident; a misdirected fax is a breach with no
attacker. The response process has to handle both.

A fourth pattern is **the breach that is only a data problem**. The response focuses on the
records and ignores the clinical workflow that depended on the compromised system. A ransomware
attack that stops imaging, a compromised alerting service, or a disabled remote-monitoring platform
harms patients before any data are disclosed. The response plan should include a clinical impact
assessment from the first hour, and the post-incident review should ask what happened to care, not
only what happened to data.

The response is also a test of the governance in Topic 08. A decision record, a named owner, and a
review date are what allow the organisation to show that it acted, not only that it reacted. An
incident without a decision record becomes a story told differently by each participant.

A response plan should also define the threshold for external communication. A single patient
complaint is handled by the service; a media inquiry is handled by communications; a regulatory
inquiry is handled by counsel. The routing should be written down before the first call.

> **Note:** Incident response is a team capability, not a document. The test is a tabletop exercise
> with the people who will actually be called at 2 a.m.

## 2. Learning outcomes

By the end of this topic you can:

1. **Distinguish** an event, an incident, and a breach, and **classify** a scenario correctly.
2. **Compute** the notification deadlines for HIPAA, GDPR, and a state law, from a stated discovery
   date.
3. **Apply** the HIPAA four-factor risk assessment and **document** a decision not to notify.
4. **Explain** the encryption safe harbour and **identify** when it does not apply.
5. **Build** a breach cost model with explicit assumptions and a per-record figure.
6. **Design** an incident response plan with roles, severity tiers, and a communication tree.
7. **Preserve** evidence and **maintain** a timeline that supports notification and investigation.
8. **Run** a post-incident review that produces funded, testable changes.

## 3. The core mental model

Incident response is a clock plus a decision tree plus a feedback loop. The clock starts at
discovery, the tree decides notification, and the loop changes the controls that failed.

```text
  event --> triage --> incident declared --> contain --> scope --> notify?
     |          |             |                |          |        |
     |          |             |                |          |        +--> individuals
     |          |             |                |          |        +--> regulator
     |          |             |                |          |        +--> media
     |          |             |                |          |
     |          |             |                |          +--> data involved
     |          |             |                |               risk assessment
     |          |             |                |
     |          |             |                +--> preserve evidence
     |          |             |
     |          |             +--> incident commander, scribe, clock
     |          |
     |          +--> severity tier
     |
     +--> log, alert, report

  after: review --> corrective actions --> design change --> test --> threat model update
```

The clock is the part most organisations underestimate. "Discovery" is not the moment the
investigation concludes; it is the first day the organisation knew or, with reasonable diligence,
would have known. A ticket opened by a help-desk analyst counts. A vendor's notification to the
account manager counts, even if it did not reach the privacy office.

The decision tree has three branches:

1. Is the incident a breach under the applicable rule? For HIPAA, there is a presumption that an
   impermissible use or disclosure is a breach unless a four-factor risk assessment demonstrates a
   low probability of compromise.
2. If it is a breach, who must be notified, within what period, and with what content?
3. If it is not a breach, what is documented, and who reviews the decision?

The feedback loop is where most programmes fail. The post-incident review should produce corrective
actions with owners, budgets, and dates, and those actions should be tested. The review should also
update the threat model, because the incident is evidence that a path was not controlled.

A fourth model is the severity tier, which determines who is called and how fast:

| Tier | Definition | Response | Notification clock |
|---|---|---|---|
| 1 | Confirmed or suspected exposure of PHI or personal data at scale, or a clinical system outage | Incident commander, executive notification within 1 hour, full team | Starts at discovery |
| 2 | Suspected exposure limited in scope, or a contained malware infection | Security lead, privacy notified within 4 hours | Starts at discovery |
| 3 | Policy violation or lost device with encryption | Security lead, assessment within 1 business day | Assess; likely no notification |
| 4 | Event with no evidence of impact | Log and monitor | None |

The tiers are a convention. Their value is that they are agreed before the incident, when nobody is
arguing about whether to wake the chief medical officer.

The clock model has a subtlety: the HIPAA 60 days are calendar days, and the GDPR 72 hours are
elapsed hours including weekends. A discovery on a Friday afternoon can consume the weekend before
the team meets, which is why the plan needs a weekend contact path.

The severity tier and the clock interact: a tier 1 declaration starts the full response, but the
legal clock started at discovery, which may precede the declaration. The gap between discovery and
declaration is itself a metric, and it is usually the first thing a review should reduce.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| Event | An observable occurrence in a system or network. | Confused with incident. | Log record. |
| Incident | A security event that has or may have an adverse effect. | Assumed to be a breach. | Severity tier. |
| Breach | A use or disclosure not permitted by law, presumed under HIPAA unless rebutted. | Used for any incident. | Legal determination. |
| Discovery | The first day the entity knew or would have known with reasonable diligence. | The day the investigation ended. | Date. |
| Notification | Informing individuals, regulators, or the media as required. | A press statement. | Legal obligation. |
| Incident commander | The person responsible for the response. | The security manager. | Role. |
| Scribe | The person who maintains the timeline. | The note-taker in the meeting. | Role. |
| Containment | Stopping the spread and preserving evidence. | Deleting the malware. | Phase. |
| Eradication | Removing the cause. | Blocking an IP address. | Phase. |
| Recovery | Restoring normal operations. | Closing the ticket. | Phase. |
| Four-factor risk assessment | HIPAA analysis of the nature and extent of PHI, the unauthorised person, whether the PHI was actually acquired or viewed, and the extent of risk mitigation. | A checkbox. | Legal assessment. |
| Presumption of breach | HIPAA presumption that an impermissible use or disclosure is a breach unless the assessment shows low probability of compromise. | Ignored. | Legal rule. |
| Safe harbour | HIPAA provision that secured PHI (encrypted or destroyed) is not a breach. | Assumed whenever encryption is enabled. | Legal rule. |
| Secured PHI | PHI encrypted per HHS guidance or destroyed per NIST guidance. | Encrypted with any method. | Legal category. |
| Business associate | An entity handling PHI on behalf of a covered entity. | Outside the breach chain. | Legal role. |
| Supervisory authority | The EU data protection regulator. | The courts. | Regulator. |
| 72-hour clock | GDPR deadline to notify the supervisory authority after awareness. | The deadline for individual notice. | Hours. |
| High risk | GDPR threshold for individual notification. | Any risk. | Legal standard. |
| Incident response plan | The documented roles, steps, and contacts for a response. | A policy on the intranet. | Document. |
| Playbook | A step-by-step procedure for a scenario. | A plan. | Document. |
| Tabletop exercise | A discussion-based rehearsal of a scenario. | A presentation. | Exercise. |
| Red team | An adversarial test of controls. | A vulnerability scan. | Exercise. |
| MTTD | Mean time to detect. | Time to ticket. | Hours or days. |
| MTTR | Mean time to respond or recover. | Time to close. | Hours or days. |
| Dwell time | Time between intrusion and detection. | Time since the last alert. | Days. |
| Chain of custody | The documented handling of evidence. | Copying files to a laptop. | Procedure. |
| Forensic image | A bit-for-bit copy of a system or storage device. | A tar archive of the home directory. | Evidence. |
| Data map | The inventory of data, systems, and flows used to scope an incident. | The network diagram. | Artefact. |
| Notification content | The required facts in a notification: what happened, what data, what to do. | A generic apology. | Legal requirement. |
| Substitute notice | Notice by website or media when individual contact is not possible. | A press release. | Legal mechanism. |
| Credit monitoring | A service offered to affected individuals. | Required in every breach. | Remediation. |
| Regulatory fine | A monetary penalty imposed by a regulator. | The only cost of a breach. | Currency. |
| Post-incident review | A structured review of the response and the cause. | A blame meeting. | Process. |
| Corrective action | A funded, owned, dated change from the review. | A recommendation. | Action item. |
| Lessons learned | The documented findings of the review. | The review itself. | Output. |

## 5. The mechanics, step by step

The procedure below is written for the first 72 hours and the 60 days that follow. It assumes a
plan exists; if it does not, the first step is to write one, and the second is to test it. Each
step names its input, output, and rationale.

### Step 1. Prepare the capability before the incident

**Input:** Threat model, data map, regulatory obligations, contact tree.
**Output:** A plan, playbooks for the top scenarios, a contact tree, and a retainer for forensics
and counsel.
**Rationale:** An incident is the worst time to discover that the privacy officer's phone number is
out of date or that the organisation has no forensic capability. The plan should name the incident
commander, the deputy, the scribe, the privacy lead, the legal lead, the communications lead, and
the executive who authorises notification. It should also name the vendor contacts for the cloud,
the EHR, and any processor that holds data.

The plan should be tested at least annually with a tabletop that includes the people who will
actually be called. A plan that has never been exercised is a hypothesis.

### Step 2. Detect and triage

**Input:** Alerts, reports, vendor notifications, patient complaints.
**Output:** A triage decision: event, incident, or breach, with a severity tier.
**Rationale:** Detection sources are broader than the security operations centre. A patient who
reports an unexpected letter, a clinician who sees another patient's result, a journalist who asks
a question, and a vendor who reports a compromise are all detection sources. The triage should
record the source, the time, the initial facts, and the tier.

The tier determines the response. A tier 1 incident triggers the full team and the clock; a tier 4
event is logged. The triage should be conservative: if the facts are unclear, start at a higher
tier and downgrade after assessment.

### Step 3. Declare and assemble

**Input:** Triage decision.
**Output:** A declared incident with an incident commander, a scribe, a timeline, and a
communication channel.
**Rationale:** The declaration is what starts the clock and the response. It should be recorded with
the time, the tier, and the commander. The scribe maintains a timeline in a single document with
timestamps and sources; the timeline is the basis for the notification and the investigation.

The communication channel should be separate from the compromised environment. An incident channel
that runs on the affected email system is a channel the attacker can read.

### Step 4. Contain without destroying evidence

**Input:** Incident declaration, system knowledge.
**Output:** Contained systems, preserved evidence, and a documented containment decision.
**Rationale:** Containment has a tension: isolating a system stops the spread and destroys the
memory and logs that would explain it. The response should preserve a forensic image or a snapshot
before rebuilding, and should record what was changed and when. The decision to shut down a clinical
system is a patient-safety decision, not only a security one, and it should involve the clinical
lead.

Branch point: if the incident involves ransomware on a running system, the data may already have
been exfiltrated. Containment should include egress analysis, not only encryption recovery.

### Step 5. Scope the data and the people

**Input:** Data map, system logs, forensic findings, vendor information.
**Output:** A scope statement: data types, number of individuals, jurisdictions, and sensitivity.
**Rationale:** The scope determines the notification obligation, the cost, and the message. It
should be built from the data map and verified against logs, not estimated from the size of the
database. The scope should distinguish records that were exposed, records that were likely accessed,
and records that were in the same system but not accessed; the legal standard differs.

The scope should also identify the jurisdictions of the affected individuals. A breach of 41,000 US
records and 620 EU records has two clocks and two notification regimes.

### Step 6. Determine the legal obligation

**Input:** Scope, applicable law, risk assessment.
**Output:** A notification decision with a documented rationale and a deadline for each regime.
**Rationale:** HIPAA presumes that an impermissible use or disclosure is a breach unless a
four-factor risk assessment demonstrates a low probability of compromise. The four factors are the
nature and extent of the PHI involved, the unauthorised person who accessed or received it, whether
the PHI was actually acquired or viewed, and the extent to which the risk has been mitigated. The
assessment should be documented even when the conclusion is no notification.

The GDPR requires notification to the supervisory authority within 72 hours of awareness unless the
breach is unlikely to result in a risk to rights and freedoms, and individual notification without
undue delay where the risk is high. The assessment should record the reasoning, the categories of
data, the approximate number of individuals, and the likely consequences.

| Outcome | When it applies | Who is notified | Deadline | Documentation |
|---|---|---|---|---|
| Breach, high risk | PHI or personal data exposed; risk to individuals | Individuals, regulator, media where required | HIPAA 60 days; GDPR 72 hours to regulator, individuals without undue delay | Full assessment; notification copies |
| Breach, low risk | Exposure with mitigations that reduce risk | Regulator only, or as required | Regulator deadline | Assessment with mitigation evidence |
| Not a breach | Secured PHI, or four-factor assessment shows low probability | No one, unless state law requires | Not applicable | Documented assessment; review |
| Undetermined | Forensics incomplete | Interim regulator notice where required | Within the legal window | Decision to notify conservatively |
| Vendor breach | Processor or business associate affected | Covered entity notifies; BA notifies CE | BA to CE 60 days; CE clock from its discovery | Contract chain; vendor assessment |

The last two rows are the ones that cause failures. An undetermined scope is not a reason to miss
the deadline; a regulator would rather receive an initial notice with a follow-up than a late
complete one. A vendor breach is not the vendor's notification alone; the covered entity remains
responsible.

### Step 7. Notify

**Input:** Decision, scope, templates, contact data.
**Output:** Notifications sent, with proof of delivery and a record of content.
**Rationale:** The notification should state what happened, when it was discovered, what data were
involved, what the organisation has done, what the individual can do, and whom to contact. It should
be written at a readable level, translated where the population requires it, and tested with a
small group before it is sent.

The channel matters. Email to an address that may be compromised, a letter to an old address, and a
website notice are different levels of assurance. The notification plan should use multiple
channels and record which were used for whom.

### Step 8. Communicate with patients, staff, and the public

**Input:** Notification, clinical impact, media interest.
**Output:** A communication plan with a single source of truth and a trained spokesperson.
**Rationale:** Patients want to know what happened to them and what to do. Staff want to know
whether they are at fault and what to say. The public wants to know whether the organisation is
competent. The communication should be consistent across channels, updated as facts change, and
delivered by a named spokesperson.

The clinical impact deserves separate treatment. If the incident affected a system used for care,
the communication should state the patient-safety measures, not only the data measures.

### Step 9. Remediate and recover

**Input:** Root cause, containment record.
**Output:** A remediation plan with owners, dates, and verification.
**Rationale:** Remediation is more than restoring service. It includes closing the vulnerability,
rotating credentials, improving monitoring, and verifying that the attacker is no longer present.
The recovery should be tested, and the test should include the clinical workflow, not only the
server.

### Step 10. Review, learn, and change

**Input:** Timeline, forensic report, notification record, incident cost.
**Output:** A post-incident review with corrective actions, owners, budgets, dates, and a threat
model update.
**Rationale:** The review should separate the cause, the response, and the impact. It should ask
what happened, why the controls failed, what worked, what did not, and what will change. The
corrective actions should be funded and tracked to completion, and the threat model should be
updated with the path that was used.

A review that produces recommendations without budgets is a review that will be repeated.

### Evidence and the timeline

The timeline is the single most useful artefact of a response. It should record every material
event with a timestamp, a source, and a confidence: the first alert, the first human report, the
declaration, the containment decision, the scope revisions, the notification decision, and the
notification. The timeline supports the legal deadline calculation, the forensic reconstruction,
and the post-incident review.

Evidence handling has three rules. Preserve before rebuilding: take a forensic image or a snapshot
of the affected system before wiping it. Keep the chain of custody: record who handled the evidence,
when, and where it was stored. Keep the evidence separate from the production environment: an
attacker who can delete the logs can delete the evidence.

The timeline should not be edited to look better. A timeline that shows a late decision is more
valuable than one that hides it, because the late decision is the lesson.

### The vendor chain in practice

Most clinical data flows through at least one vendor, and many through four or five. The contract
should require the processor to notify the covered entity within a defined period (for example, 48
hours), to provide the information needed for the covered entity's own notification, and to
cooperate with the investigation. The contract should also require the vendor to identify its
subprocessors and to flow the same obligations down.

The covered entity's clock does not wait for the vendor. If the vendor's notice arrives on day 10,
the covered entity's 60-day HIPAA clock and 72-hour GDPR clock are already running from the
covered entity's own awareness, which may be the day the vendor told it. The response plan should
assume that vendor information will be late and incomplete, and should have a phased notification
that does not depend on it.

### Notification content and channels

A notification has to answer five questions in plain language: what happened, when, what
information was involved, what the organisation has done, and what the person can do. It should
also name a contact and a deadline for the person to act (for example, a call centre open for 90
days). The letter should be tested with a small group of patients or community representatives
before it is sent.

The channels should match the population. Mail to the address on file, email where consent exists,
a website notice, and a call centre are complementary. A population with limited English
proficiency needs translated materials and interpreters; a population with low digital access needs
mail and phone. The notification record should state which channel was used for whom.

### The first 24 hours

The first day sets the trajectory. A workable sequence: declare within one hour of triage;
appoint the commander and scribe; open a separate communication channel; preserve logs and take a
snapshot; contain the affected accounts or systems without wiping; notify the privacy officer,
clinical lead, and executive; start the timeline; and make the first scope estimate from the data
map. The notification decision does not need to be made on day one, but the clock does.

The most common first-day error is to let the IT team investigate without a scribe and without the
privacy office. The investigation produces facts, and the facts are needed for the legal decision,
but the timeline and the decision log are what the regulator will ask for.

### Legal privilege and counsel

An investigation conducted at the direction of counsel may be protected by legal privilege, which
affects what can be shared and with whom. The response plan should define when counsel is engaged
and how the forensic work is commissioned, so that the privilege is not accidentally waived. The
plan should also state what must be shared with the regulator regardless of privilege, because the
notification obligation is not suspended by an internal investigation.

### Severity and escalation

Severity should be set by impact, not by the attacker's sophistication. A low-sophistication
ransomware attack that stops a hospital's imaging system is a tier 1 incident because of the
clinical impact. A sophisticated intrusion into a segmented, encrypted, low-sensitivity system may
be a tier 2. The tier should record the clinical impact, the data sensitivity, the scale, and the
containment status.

Escalation should be automatic at tier 1: the executive, the clinical lead, the privacy officer, and
communications are called within an hour, and the board is informed within a day. The plan should
name the alternates, because the primary contact may be on leave or on a plane.

### Documentation

The incident file should contain the declaration, the timeline, the evidence log, the scope
assessment, the four-factor analysis, the notification decision, the notifications, the
communication record, the forensic report, and the post-incident review. It should be retained for
the period required by law and by counsel, and it should be stored where the response team can
reach it without the affected systems.

The file is also the evidence that the organisation met its obligations. A regulator will ask for
the discovery date, the assessment, and the notification; a file that cannot produce them turns a
defensible response into a finding.

### Communication with staff

Staff are both responders and potential sources of harm. They need to know what happened at a level
that lets them do their jobs, what to say to patients, where to direct questions, and that they are
not blamed for reporting. A staff communication that is delayed or evasive produces rumour, and
rumour is harder to correct than a fact.

The communication should be led by the clinical and operational leaders, not only by security. A
nurse on a ward needs to know whether the EHR is available and what to do if it is not; a
receptionist needs to know what to tell patients who call. The plan should include a script and a
single point of contact for each staff group.

### What the regulator asks

A regulator reviewing a response typically asks six questions: when did you discover it, what did
you do, what data and how many people, who did you notify and when, what have you changed, and how
do you know the change works. The incident file should answer all six without reconstruction. The
last question is the one most organisations fail, because the corrective action was recorded but
not tested.

### What the model does not include

The model excludes fines, litigation, contract penalties, lost revenue, and staff time. It also
excludes the cost of the control that would have prevented the incident, which is the number the
budget request needs. The post-incident review should compare the incident cost with the annual
cost of the preventive control; the comparison is usually favourable and is the argument for
funding.

### The clock and the clinical decision

Containment can require taking a system offline, and taking a system offline can harm patients. The
response plan should state who makes that decision, with what clinical advice, and how the decision
is recorded. A security team that isolates the EHR without a clinical conversation has contained
the malware and created a new incident.

### Edge cases

**Small cells.** A breach involving a small number of patients in a rare condition can be
identifiable in the notification itself. The notification should not include details that identify
the patients to each other, and the public statement should aggregate.

**Quasi-identifiers.** A lost laptop with a de-identified extract can still be a breach if the
extract contains quasi-identifiers and the recipient can link. The assessment should use the
de-identification determination from Topic 02.

**Re-identification through linkage.** An incident involving a released dataset that can be linked
to public records is a privacy incident even if no direct identifier was exposed. The notification
should reflect the inferred risk, not only the fields.

**Free-text PHI.** A misdirected email containing a clinic letter is a breach even if the structured
fields are absent. The scope assessment should include the content of the free text.

**Metadata leakage.** A shared file with metadata in the properties can disclose the author or the
patient. The forensic scope should include metadata, and the remediation should strip it.

**Consent withdrawal.** An incident affecting a participant who had withdrawn is still an incident;
the withdrawal does not remove the legal obligation for the period before it. The notification
should not imply that withdrawn participants were unaffected if their data were still held.

**Minors and deceased patients.** Notification to a parent or representative, and the handling of
deceased patients' records, need the rules from Topic 01. The notification should not disclose a
minor's sensitive information to a parent who is not entitled to it.

**Cross-border transfer.** An incident affecting data in multiple countries triggers multiple
notification regimes. The organisation should identify the lead supervisory authority and
coordinate, rather than send inconsistent notices.

**Vendor and supply chain.** A breach at a processor can affect multiple covered entities. The
contract should require notification to the covered entity within a defined period, with the
information needed for the covered entity's own notification.

**Ransomware without confirmed exfiltration.** The absence of evidence of exfiltration is not
evidence of absence. The assessment should consider egress logs, the attacker's dwell time, and the
common pattern of double extortion before concluding that data were not taken.

## 6. Worked example

### Setting

A hospital's analytics server is compromised. The security team detects anomalous outbound traffic
on 2 March 2026 at 09:00 and declares a tier 1 incident. Forensic analysis finds that the attacker
entered on 12 February through a stolen service-account credential and exfiltrated about 6 TB of
data over 18 days, a rate of 333.3 GB per day. The server held an analytics extract of 41,280
patients. The extract included 3,200 Social Security numbers (7.8 percent) and 1,400 records with
substance-use disorder codes (3.4 percent). Of the 41,280 patients, 620 are resident in the EU.

The hospital notifies individuals on 6 March, four days after discovery. It notifies the EU
supervisory authority at the same time.

### Step 1: the clocks

- HIPAA: discovery on 2 March 2026, so the notification deadline is 2 March + 60 calendar days = 1
  May 2026. The individual notices on 6 March are within the window.
- GDPR: awareness on 2 March at 09:00, so the supervisory-authority deadline is 5 March at 09:00,
  72 hours later. Notification on 6 March at 09:00 is 24 hours late.
- The state breach law may impose a shorter deadline (for example, 30 days) and may require notice
  to the state attorney general; the assessment should check each state.

The arithmetic is simple and the consequence is not. The HIPAA timeline was met with 56 days to
spare; the GDPR timeline was missed by one day. The difference is not effort; it is that the EU
assessment waited for the forensic scope, and the scope took four days.

### Step 2: the cost model

The hospital builds a cost model with explicit assumptions:

| Component | Assumption | Calculation | Cost (USD) |
|---|---|---|---:|
| Mail notification | USD 12 per individual | 41,280 x 12 | 495,360 |
| Call-centre support | USD 8 per individual | 41,280 x 8 | 330,240 |
| Credit monitoring | USD 25 per SSN record | 3,200 x 25 | 80,000 |
| Forensics, counsel, communications | Fixed | Stated retainer and effort | 750,000 |
| **Total** | | | **1,655,600** |

The per-record cost is 1,655,600 / 41,280 = USD 40.11. The figure is a planning assumption, not a
benchmark; the value of the model is that it shows which components are driven by the record count
and which are fixed. The mail and call-centre costs scale with every record; the credit monitoring
scales with the SSN subset; the forensic and legal costs are largely fixed.

### Diagnostic 1: SSNs in an analytics extract

The extract was built for a diabetes analytics programme and did not need SSNs. The field-level
review in Topic 01 would have removed them, and the removal would have cut the credit-monitoring
cost by 80,000 and reduced the severity of the notification. The diagnostic is not the dollar
figure; it is that a field with no analytic purpose was present in a system with broad access. The
same over-collection increases the harm from any future incident, not only this one.

If the extract had contained only the nine justified fields from Topic 01, the notification would
still have been required (names and phone numbers are identifiers), but the sensitivity and the
monitoring cost would have been lower.

### Diagnostic 2: the late GDPR notification

The EU notification was 24 hours late because the team waited for the forensic scope before
assessing the risk. The GDPR does not require a complete scope at 72 hours; it requires a
notification with the information available, followed by updates. The hospital could have notified
on 4 March with the known facts and updated on 6 March.

The diagnostic is procedural, and it is common: the team treats the notification as a final report
rather than an initial notice. The remedy is a template for a phased notification, a pre-approved
decision tree, and a named person who owns the 72-hour clock.

### Vary one assumption: encryption at rest with separated keys

Suppose the analytics server had stored the extract with AES-256 encryption at rest, with the key
in a managed key service in a separate trust domain, and the investigation found no evidence that
the key was used. Under HIPAA, the data would be secured PHI and the safe harbour would apply: no
individual notification, because the data were not unsecured.

The cost model changes:

- Individual notification, call centre, and credit monitoring: 0.
- Forensics, counsel, and communications: 750,000.
- Re-keying and verification: 120,000.
- Total: 870,000.
- Saving against the base case: 785,600, or 47.5 percent.

The conclusion changes from "notify 41,280 individuals and spend USD 1.66 million" to "no
individual notification, USD 870,000, and a documented safe-harbour assessment". The EU analysis
still applies: encryption may reduce the risk, but the controller must assess whether the breach is
likely to result in a risk to rights and freedoms, and the key compromise question must be
answered.

The counterfactual also shows why "we have encryption" is not the same as "we have the safe
harbour". The safe harbour depends on the encryption meeting HHS guidance and on the key not being
compromised. A key stored with the data, or an encryption mode that is not validated, does not
qualify.

### What the incident cost does not include

The model excludes regulatory fines, class-action exposure, contract penalties, lost revenue, and
reputational harm. It also excludes the clinical cost of the outage if the analytics server was
used for care. Those omissions are not because they are unimportant; they are because they are
harder to estimate and should be presented as separate scenarios rather than folded into a single
number that looks precise.

### The four-factor assessment

The hospital's assessment for the 41,280 records concludes that notification is required. Applying
the factors:

1. **Nature and extent.** Names, dates of birth, phone numbers, SSNs for 3,200 people, and
   substance-use diagnoses for 1,400. The combination of identifier and sensitive diagnosis raises
   the risk.
2. **The unauthorised person.** An external attacker with demonstrated access to the server; not a
   person with a duty of confidentiality.
3. **Whether the data were acquired or viewed.** The exfiltration volume (6 TB) and the attacker's
   dwell time (18 days) indicate acquisition, not mere access.
4. **Mitigation.** Containment, credential rotation, and monitoring reduce further risk but do not
   undo the disclosure.

The conclusion is that the probability of compromise is not low, so the presumption of a breach is
not rebutted and notification is required. The assessment is documented with the facts and the
reasoning, which is what the rule requires even when the conclusion is notification.

### The clinical impact

The analytics server was not used for direct care, so the incident did not interrupt treatment. If
it had been the sepsis alert server, the response would have included a clinical lead, a
continuity plan, and a patient-safety review. The distinction matters for the notification: a data
breach notice is not a substitute for a patient-safety communication.

### The notification population

The notification population is not always the record count. If the extract contains 41,280 records
but 2,300 are duplicates or test records, the population is 38,980. If 620 are EU residents, the
GDPR notice goes to 620 and the HIPAA notice to the US residents. If some individuals are deceased,
the notice may go to the estate or not at all, depending on state law. The scope work should
produce a defensible population, not a database count.

The hospital's data quality check finds 312 duplicate records and 48 test records, so the
notification population is 41,280 - 360 = 40,920. The cost model should use the corrected figure;
the difference is small in this case, and it is large in cases where the extract contains years of
duplicates.

### The regulatory fine scenario

The cost model excludes fines. A separate scenario can bound the exposure: under the GDPR, the
maximum administrative fine is 20 million euros or 4 percent of global annual turnover, whichever
is higher, for the most serious infringements; under HIPAA, the civil money penalty tiers are set
by regulation and adjusted for inflation. The scenario should state the basis and present a range,
not a point estimate, because the fine depends on factors the organisation cannot control.

The point of the scenario is not to predict the fine; it is to show that the notification and
remediation costs, which are manageable, are small compared with the tail risk. That comparison is
what funds preventive controls.

### The cost of the response, not only the breach

The model counts notification and remediation. It does not count the staff time spent on the
response: the incident commander, the clinicians pulled from duties, the call-centre training, the
legal review. A 40,920-person notification can consume hundreds of hours of operational time, and
that time has an opportunity cost. The post-incident review should estimate it, because the next
budget request will be compared with it.

The hospital also learns that its vendor contract did not require notification within a defined
period. The vendor informed the account manager on 28 February, two days before the security team
detected the traffic. If the contract had required a 24-hour notice to the privacy office, the
clock would have started earlier and the EU notice might have been on time. The contract is now a
corrective action, not only a legal document.

## 7. Code, query, or calculation

The script below computes the notification cost, the per-record figure, the GDPR and HIPAA
deadlines, the dwell time, and the encryption counterfactual. It uses only the standard library.
The assumptions are the ones in Section 6.

```python
"""Breach notification cost and deadline arithmetic for a synthetic incident."""
from datetime import datetime, timedelta


def cost(n, mail=12.0, call=8.0, monitored=0, monitoring=25.0, fixed=750_000.0):
    mail_cost = n * mail
    call_cost = n * call
    monitor_cost = monitored * monitoring
    total = mail_cost + call_cost + monitor_cost + fixed
    return {"mail": mail_cost, "call": call_cost, "monitor": monitor_cost,
            "total": total, "per_record": total / n}


def deadline(discovered, hours):
    return discovered + timedelta(hours=hours)


if __name__ == "__main__":
    n, monitored = 41_280, 3_200
    c = cost(n, monitored=monitored)
    print(f"mail={c['mail']:,.0f}, call centre={c['call']:,.0f}, "
          f"monitoring={c['monitor']:,.0f}, total={c['total']:,.0f}, "
          f"per record={c['per_record']:.2f}")

    enc_total = 750_000.0 + 120_000.0
    print(f"encrypted counterfactual: total={enc_total:,.0f}, "
          f"saving={c['total'] - enc_total:,.0f} "
          f"({(c['total'] - enc_total) / c['total']:.1%})")

    discovered = datetime(2026, 3, 2, 9, 0)
    notified = datetime(2026, 3, 6, 9, 0)
    gdpr_deadline = deadline(discovered, 72)
    print(f"GDPR deadline={gdpr_deadline.isoformat()}, "
          f"notified={notified.isoformat()}, "
          f"late_by_hours={(notified - gdpr_deadline).total_seconds() / 3600:.0f}")
    print(f"HIPAA deadline={deadline(discovered, 60 * 24).date().isoformat()}")

    intrusion = datetime(2026, 2, 12, 9, 0)
    dwell = (discovered - intrusion).days
    print(f"dwell time={dwell} days, exfiltration rate={6000 / dwell:.1f} GB/day")

    print(f"SSN share={monitored / n:.1%}, substance-use records=1400 "
          f"({1400 / n:.1%})")
```

Expected output, verified on Python 3.11:

```text
mail=495,360, call centre=330,240, monitoring=80,000, total=1,655,600, per record=40.11
encrypted counterfactual: total=870,000, saving=785,600 (47.5%)
GDPR deadline=2026-03-05T09:00:00, notified=2026-03-06T09:00:00, late_by_hours=24
HIPAA deadline=2026-05-01
dwell time=18 days, exfiltration rate=333.3 GB/day
SSN share=7.8%, substance-use records=1400 (3.4%)
```

Four readings of the output. The total and per-record cost show how much of the cost is
record-driven. The encrypted counterfactual shows the value of the safe harbour, and it depends on
an assumption that must be verified, not asserted. The deadline output shows a met HIPAA clock and
a missed GDPR clock in the same incident. The dwell-time output shows how long the attacker had,
which is the number the detection programme is trying to reduce.

To adapt the script, replace the record counts and unit costs with the organisation's actual
figures and add the state-law deadlines. The model should be run during the incident, not after it,
because it informs the notification decision and the resourcing of the call centre.


### Reading the output

The 47.5 percent saving in the encrypted counterfactual is the value of a control that was not in
place. The 24-hour lateness is the value of a decision that waited for certainty. The 18-day dwell
time is the value of detection coverage. Each number maps to a corrective action, which is the point
of computing them.

The model is a decision aid during the incident, not a post-mortem spreadsheet. It should be run as
soon as the scope is roughly known, because the call-centre and mail costs need to be authorised
before the notification is sent.

The model assumes a single notification wave and a single currency. A multi-country incident needs
a currency conversion, a per-country population, and a per-country legal review; the script should
be extended rather than forced to fit.

The deadline output is a reminder that the clocks start together, not when the organisation feels
ready. The HIPAA deadline is comfortable; the GDPR deadline is not.

The model treats the discovery time as a single instant. In practice, discovery is a process: an
alert, a human confirmation, a decision to declare. The legal clock starts at the earliest point at
which the organisation knew or should have known, so the timeline should record each step.

The model assumes the same population for HIPAA and GDPR; in production the populations differ, and
the per-regime costs should be computed separately.

## 8. Data, standards, and regulatory context

Breach notification is one of the few areas where the legal deadline is measured in hours or days,
and the rules differ by jurisdiction in ways that matter operationally.

### HIPAA Breach Notification Rule

The rule (45 CFR 164.400-414) applies to covered entities and business associates.

- **Discovery.** A breach is discovered on the first day it is known, or by exercising reasonable
  diligence would have been known, to the entity. Knowledge is imputed from any workforce member,
  officer, or agent, not only from the person who caused the breach.
- **Individual notice.** Without unreasonable delay and no later than 60 calendar days after
  discovery, by first-class mail or email if the individual agreed.
- **Media notice.** For a breach affecting more than 500 residents of a state or jurisdiction,
  notice to prominent media outlets without unreasonable delay and within 60 days.
- **HHS notice.** For 500 or more individuals, notice to HHS at the same time as individual notice;
  for fewer than 500, an annual log within 60 days after the end of the calendar year.
- **Business associates.** A business associate must notify the covered entity without unreasonable
  delay and within 60 days of discovery, with the identification of affected individuals to the
  extent possible.
- **Burden of proof.** The covered entity has the burden of demonstrating that all required
  notifications were made or that the breach was not a breach.

The **four-factor risk assessment** determines whether the presumption of a breach is rebutted: the
nature and extent of the PHI involved, including identifiers and likelihood of re-identification;
the unauthorised person who used or received the PHI; whether the PHI was actually acquired or
viewed; and the extent to which the risk has been mitigated. The assessment must be documented and
retained for six years.

The **safe harbour** applies to secured PHI: ePHI encrypted according to HHS guidance, or destroyed
according to NIST guidance. The encryption must be validated and the key must not have been
compromised. Ransomware that encrypts files on a running system does not qualify, because the data
were decrypted and accessible before encryption.

### GDPR

- **Article 33.** The controller must notify the supervisory authority without undue delay and,
  where feasible, within 72 hours of becoming aware of the breach, unless the breach is unlikely to
  result in a risk to the rights and freedoms of natural persons. A processor must notify the
  controller without undue delay.
- **Article 34.** Where the breach is likely to result in a high risk, the controller must
  communicate to the data subject without undue delay, in clear and plain language, with the nature
  of the breach, the contact point, the likely consequences, and the measures taken.
- **Records.** The controller must document all breaches, including those not notified, with the
  facts, effects, and remedial action.
- **EDPB guidance.** The Article 29 Working Party guidelines on personal data breach notification
  (WP250rev.01, 2018, endorsed by the EDPB) provide examples and a risk assessment framework. They
  are guidance, not law.

The 72-hour clock starts at awareness, which includes awareness by a processor. A vendor that waits
a week to inform the controller has consumed the controller's window.

### Other regimes

- The FTC Health Breach Notification Rule (16 CFR Part 318) applies to vendors of personal health
  records and related entities not covered by HIPAA. It requires notice to affected individuals,
  the FTC, and sometimes the media, without unreasonable delay and no later than 60 calendar days
  after discovery. The amended rule took effect in 2024 and reaches consumer health apps.
- State breach laws impose their own deadlines, content requirements, and attorney-general notice.
  Some are shorter than HIPAA's 60 days, and some define personal information differently.
- The EU NIS2 Directive and sector-specific rules may impose additional incident-reporting
  obligations on critical infrastructure and essential services.

### Standards

- NIST SP 800-61 Rev. 3, *Incident Response Recommendations and Considerations for Cybersecurity
  Risk Management* (April 2025), is a full rewrite of the 2012 guide. It integrates incident
  response into the NIST Cybersecurity Framework 2.0 functions (Govern, Identify, Protect, Detect,
  Respond, Recover) and covers detection, triage, evidence handling, containment, and post-incident
  activity. It is guidance, not regulation.
- ISO/IEC 27035-1:2023 and 27035-2:2023 provide principles and incident response guidance within an
  information security management system. They are certifiable as part of ISO/IEC 27001.
- The HIPAA Security Rule's incident procedures (45 CFR 164.308(a)(6)) require a response and
  mitigation process, and its documentation requirements apply to the incident record.

**Incident response is a clinical capability.** In a health system, the response team needs a
clinical voice from the first hour, because the decision to isolate a system can harm patients. A
response plan that treats the EHR as an IT system rather than a clinical instrument will make the
wrong trade-off between containment and care.

**The regulator is a stakeholder, not an adversary.** An organisation that engages early, provides
the facts it has, and updates as the investigation proceeds is in a better position than one that
waits for a complete story. The notification is the beginning of a relationship, and the quality of
the relationship affects the outcome.

**The response plan should name the person who can authorise spending.** Notification costs are
incurred in hours, and a purchase-order process that takes a week is a containment failure. A
pre-approved spending authority for tier 1 incidents, with a cap and a retrospective review, is a
practical control.

**The notification is a public document.** It may be read by patients, journalists, regulators, and
plaintiffs. It should be accurate, specific, and consistent with the regulatory filing. A statement
that minimises the incident and is later contradicted by the facts is worse than a statement that
acknowledges uncertainty.

> **Note:** Regulatory content in this file is educational and is not legal advice. Notification
> decisions are fact-specific and time-critical, and should be made with counsel.

The classification to keep straight: the HIPAA 60-day and GDPR 72-hour deadlines are binding; the
four-factor assessment is a legal test; encryption safe harbour is a legal provision with technical
conditions; NIST SP 800-61 is guidance; and a tabletop exercise is a test, not a compliance
artefact.

## 9. Industry practice and case evidence

### Change Healthcare, 2024

The Change Healthcare incident, discussed in Topics 04 and 05, is the largest US health data breach
on record. Initial access was through a Citrix remote-access portal without MFA, using compromised
credentials. The attacker moved laterally, exfiltrated about six terabytes, and deployed ransomware.
The breach count reached 192.7 million individuals in the filing updated in July 2025. UnitedHealth
estimated the total cost at approximately USD 3.09 billion.

Three response lessons stand out. First, the detection-to-containment interval was long enough for
the attacker to exfiltrate six terabytes, which is a detection and egress-monitoring failure.
Second, the breach count was revised several times, which is a scope-assessment failure; the
organisation could not say who was affected for months. Third, the disruption to claims processing
showed that a security incident can become a clinical and financial continuity incident even when
no patient data are used.

### Anthem, 2015

Anthem discovered the intrusion in January 2015 after a database query by a user using stolen
credentials. The breach affected 78.8 million people. OCR's findings included failure to conduct an
enterprise-wide risk analysis and insufficient technical access controls. The response involved
notification, credit monitoring, and a class-action settlement.

The case is a reminder that the notification population is determined by the data that were
reachable, not by the data the organisation intended to expose. A warehouse with broad access
produces a very large notification.

### SingHealth, 2018

Singapore's SingHealth suffered a cyberattack in June 2018 that compromised the data of about 1.5
million patients, including outpatient dispensed medicine records for about 160,000. A Committee of
Inquiry found weaknesses in the organisation's defences and response, and the incident led to
changes in national health cybersecurity governance.

The case is useful because it combined data exfiltration with a targeted attack on a national health
infrastructure, and because the public inquiry produced a detailed account of the response. The
lesson for this topic is that the post-incident review can be a public governance mechanism, not
only an internal one.

### WannaCry and the NHS, 2017

The WannaCry attack affected at least 80 NHS trusts and 603 primary care organisations, and NHS
England estimated 19,494 cancelled appointments. The National Audit Office found no evidence that
patient data were compromised and no organisation paid the ransom. The incident was primarily an
availability incident, not a data breach.

The case shows why the incident classification matters. A ransomware attack that encrypts systems
without exfiltrating data is still a major incident, with patient-safety consequences, and the
response is a continuity response, not only a notification response.

### What mature programmes do

- **Exercise the plan.** Tabletop exercises at least annually, with the actual responders, and a
  written after-action report with funded actions.
- **Measure detection and response.** Mean time to detect, mean time to contain, and dwell time are
  reported to the executive, not only to the security team.
- **Pre-approve the decision tree.** The notification decision, the templates, and the contact
  tree are prepared before the incident, so the team is not drafting them at 2 a.m.
- **Retain forensics and counsel.** A retainer with a forensics firm and breach counsel reduces the
  time to a competent investigation.
- **Coordinate the vendor chain.** The contract requires a processor to notify within a defined
  period and to provide the information the covered entity needs.

**Notification is not the end of the obligation.** The organisation must also respond to individual
questions, provide the offered services, and update the regulator as facts change. A notification
without a follow-up capability creates a second wave of complaints.

**The post-incident review should include the near misses.** An incident that was contained before
notification is a source of lessons about what worked, not only about what failed. A programme that
reviews only breaches loses the evidence from the incidents that were stopped.

**The breach coach model.** Many organisations retain external counsel and a forensics firm before
an incident, with a pre-agreed engagement letter and a communication protocol. The retainer is a
cost; the alternative is negotiating a contract while the clock runs. The arrangement should be
tested in a tabletop, not only signed.

**The incident is not the only source of lessons.** Audit findings, near misses, tabletop results,
and vendor assessments are all evidence about the response capability. A programme that reviews
them together sees the pattern before an incident produces it.

> **Note:** The case facts above are drawn from the cited public reports. Statements about what an
> organisation should have done differently are inference, labelled as such.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Late discovery | Dwell time measured in weeks | No egress monitoring | Larger exposure; late clock start | Detection coverage; egress alerts |
| Clock confusion | Team debates when discovery occurred | No declaration process | Missed deadline | Declare and timestamp |
| No incident commander | Everyone coordinates | Diffuse authority | Slow containment | Named commander and deputy |
| Evidence destroyed | Rebuilt before imaging | Containment by deletion | Cannot scope or defend | Preserve first; document |
| Scope by guess | Count from database size | No data map | Wrong notification population | Data map; log verification |
| No four-factor assessment | Decision undocumented | Legal test skipped | Presumption of breach | Template and review |
| Safe harbour assumed | "It was encrypted" | Key or method not verified | Notification still required | Verify key and standard |
| Late GDPR notice | Waiting for complete facts | Misread 72-hour rule | Regulatory finding | Phased notice template |
| Vendor delay | Processor informs late | Contract lacks deadline | Controller misses clock | Contractual notice period |
| Generic notification | No data types or actions | Legal template | Patient confusion; trust loss | Specific content; plain language |
| No translation | Notice only in English | Population excluded | Inequitable harm | Translate; multiple channels |
| No post-incident funding | Recommendations accepted, not funded | Review without budget | Repeat incident | Corrective action budget |
| Repeat incident | Same root cause twice | Lessons not implemented | Regulatory escalation | Track actions to closure |
| Clinical impact ignored | Only data discussed | Security framing | Patient safety harm | Clinical lead in response |
| No spending authority | Purchase order delays the call centre | Procurement process | Late notification | Pre-approved tier 1 authority |
| Review without near misses | Only breaches reviewed | Learning gap | Recurring containment failures | Review contained incidents |

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test |
|---|---|---|---|
| The data map is current | Scope depends on it | New system or field | Reconcile to inventory |
| Discovery time is recorded | Deadlines depend on it | Informal water-cooler report | Declaration log |
| The responder roster is current | Response depends on it | Staff turnover | Quarterly call tree test |
| The forensic capability is available | Investigation depends on it | No retainer | Contract and exercise |
| The vendor will notify promptly | Clock depends on it | Contract silent | Contract review |
| Encryption meets the safe harbour | Notification decision depends on it | Wrong mode or key exposure | Technical verification |
| The affected population is identifiable | Notification depends on it | Missing contact data | Data quality check |
| The notification content is understood | Trust depends on it | Reading level too high | User testing |
| The cost model's assumptions hold | Budget depends on it | Unit costs differ | Re-estimate with actuals |
| The review produces funded actions | Learning depends on it | No budget | Track actions |
| The regulator accepts the assessment | Legal exposure depends on it | Insufficient documentation | Counsel review |
| The notification is understood | Trust depends on the response | Reading level or language | User testing |
| The response team is available | Containment depends on it | Travel or leave | Call-tree test |

Two limits are structural. First, the notification decision is made before the facts are complete;
the process must support a phased notice rather than waiting for certainty. Second, the cost model
is a planning tool, not a forecast; the largest costs, such as fines, litigation, and lost trust,
are the least predictable.

## 12. Exercises

The exercises are ordered from classification to learning. Exercises 1 to 3 test the legal
decision; 4 and 5 test the arithmetic and the timeline; 6 to 8 test communication, rehearsal, and
the review. Exercise 7 is the one that most often exposes a gap, because a tabletop with clinical
injects forces the team to discuss the trade-off between containment and care.

The first 24 hours are a sequence, not a meeting. The plan should be written as a checklist with
times, owners, and evidence, so that a tired team can follow it at 3 a.m.

The exercises use one scenario across several of them, so the classification, the deadlines, the
assessment, and the letter can be built as a single incident file. That is how the work actually
arrives: the same facts, examined from four angles, under time pressure.

**Exercise 1 - Classify the scenario (recall, 25 minutes).**
For five scenarios (lost encrypted laptop, misdirected fax, ransomware with exfiltration, stolen
credentials with no evidence of access, vendor breach), classify as event, incident, or breach, and
state the notification obligation. Deliverable: table. Criteria: correct classification; safe
harbour applied only where the conditions hold.

**Exercise 2 - Compute the deadlines (calculation, 30 minutes).**
A breach is discovered on 14 September 2026 at 16:00. Compute the HIPAA 60-day deadline, the GDPR
72-hour deadline, and the day on which a 30-day state deadline falls. Deliverable: dates with
arithmetic. Criteria: calendar days; time zone stated.

**Exercise 3 - Four-factor assessment (application, 45 minutes).**
Write a four-factor assessment for a misdirected email containing a clinic letter with a diagnosis.
Deliverable: one page with a recommendation. Criteria: all four factors addressed; mitigation
described; decision documented.

**Exercise 4 - Cost model (calculation, 45 minutes).**
Build a cost model for a breach of 12,000 records with 900 SSNs, using stated unit costs. Compute
the total and per-record cost, and the saving if the SSNs had been removed. Deliverable: table.
Criteria: assumptions explicit; arithmetic correct; fixed versus variable costs distinguished.

**Exercise 5 - Incident timeline (application, 40 minutes).**
Build a timeline for the first 72 hours of a tier 1 incident, with roles and decision points.
Deliverable: timeline. Criteria: declaration, containment, scope, notification decision, and
communication; times realistic.

**Exercise 6 - Notification letter (design, 45 minutes).**
Write a notification letter for the Section 6 incident at a sixth-grade reading level. Deliverable:
one page. Criteria: what happened, when, what data, what to do, contact; no jargon; translated
version noted.

**Exercise 7 - Tabletop design (design, 60 minutes).**
Design a 90-minute tabletop for a ransomware incident with exfiltration. Deliverable: scenario,
injects, roles, and success criteria. Criteria: clinical impact included; notification clock
included; after-action process defined.

**Exercise 8 - Post-incident review (critique, 45 minutes).**
Critique a public post-incident report and identify three corrective actions that were not funded.
Deliverable: 500 words. Criteria: specific findings; funding gap identified; test proposed.

## 13. Self-check answers

**Answer 1.** A lost laptop with full-disk encryption and an uncompromised key is secured PHI and
not a breach under HIPAA, though it is an incident and may require a state-law analysis. A
misdirected fax with PHI is a breach unless the four-factor assessment rebuts the presumption. A
ransomware attack with exfiltration is a breach and an incident; the exfiltration is the
disclosure, and the encryption of files does not create a safe harbour. Stolen credentials with no
evidence of access require a four-factor assessment; the absence of evidence is not proof, and the
assessment should consider logs and opportunity. A vendor breach is an incident for the covered
entity, which remains responsible for notification even though the vendor must notify the covered
entity.

**Answer 2.** Discovery 14 September 2026 at 16:00. HIPAA: 14 September + 60 calendar days = 13
November 2026. GDPR: 14 September 16:00 + 72 hours = 17 September 16:00. A 30-day state deadline:
14 October 2026. The arithmetic uses calendar days, and the time zone must be stated because the
72-hour clock is measured in hours.

**Answer 3.** The four factors: the nature and extent of the PHI (a clinic letter with a diagnosis,
name, and address; high sensitivity, low likelihood of re-identification from the letter alone);
the unauthorised person (a patient who received the letter; not a person who would misuse it); whether
the PHI was actually acquired or viewed (unknown, but the recipient had the opportunity); and the
extent of mitigation (the recipient confirmed destruction and signed an attestation). A reasonable
conclusion may be that the risk is low, but the assessment must be documented and the decision
reviewed by the privacy officer.

**Answer 4.** With mail at 12, call centre at 8, monitoring at 25 for 900 records, fixed at
500,000: mail 144,000; call 96,000; monitoring 22,500; total 762,500; per record 63.54. If the SSNs
had been removed, monitoring is 0 and the total is 740,000, a saving of 22,500. The example shows
that the SSN removal saves less than the fixed costs; the larger argument for removal is severity
and regulatory exposure, not the monitoring line.

**Answer 6.** The letter should state: what happened (an unauthorised person accessed a server),
when (discovered on 2 March 2026), what information (name, date of birth, and for some people a
Social Security number), what the organisation has done (contained, notified law enforcement,
improved access), what the person can do (monitor accounts, contact the call centre), and whom to
contact. It should avoid "we take privacy seriously" as a substitute for facts. A translated
version and a large-print version should be available.

**Rubric for Exercise 3.** Full credit requires all four factors, a mitigation description, a
recommendation, and a documented decision. An assessment that concludes "low risk" without the
factors does not pass.

**Rubric for Exercise 5.** Full credit requires a declaration time, an incident commander, a
containment decision, a scope step, a notification decision with the clock, and a communication
step. A timeline that starts with "investigate" and has no clock does not pass.

**Rubric for Exercise 7.** Full credit requires a scenario, at least six injects, named roles, a
clinical-impact inject, a notification-clock inject, and an after-action process. A discussion
guide without injects does not pass.

**Answer 5.** The first hour: declare the incident and record the time; appoint an incident
commander and a scribe; open a separate communication channel; preserve logs and take a snapshot;
contain the affected accounts or systems without wiping; notify the privacy officer, the clinical
lead, and the executive; and begin the timeline and the scope estimate. The notification decision is
not made in the first hour, but the clock and the evidence handling start there.

**Answer 7.** The business associate must notify the covered entity without unreasonable delay and
within 60 days of discovery, with the identification of affected individuals and the information the
covered entity needs for its own notification. The covered entity remains responsible for notifying
individuals and HHS. A contract that says the vendor will notify patients directly does not
transfer the legal obligation.

**Answer 9.** An effective review separates the cause, the response, and the impact; identifies what
worked and what did not; produces corrective actions with owners, budgets, and dates; tracks them to
closure; and updates the threat model. A review that lists recommendations without funding is not
effective, because the same cause will recur.

**Rubric for Exercise 1.** Full credit requires a classification for each scenario, the applicable
legal rule, and the notification obligation. A table that calls every incident a breach does not
pass.

**Rubric for Exercise 4.** Full credit requires explicit unit costs, fixed versus variable
components, the total, the per-record figure, and the SSN-removal counterfactual. A total without
the assumptions does not pass.

**Rubric for Exercise 6.** Full credit requires the five facts (what, when, what data, what done,
what to do), a contact, plain language, and a translation note. A letter that leads with an apology
and omits the data types does not pass.

## 14. Interview questions

**1. When does the HIPAA clock start?**
Strong answer: on discovery, the first day the entity knew or with reasonable diligence would have
known, imputed from any workforce member or agent; not when the investigation concludes. Weak
answer: "when the breach is confirmed".

**2. What is the four-factor risk assessment?**
Strong answer: nature and extent of PHI, the unauthorised recipient, whether the PHI was acquired or
viewed, and the mitigation; used to rebut the presumption of a breach. Weak answer: "a risk
assessment".

**3. When does encryption create a safe harbour?**
Strong answer: when the ePHI is secured per HHS guidance and the key has not been compromised;
ransomware on a running system does not qualify. Weak answer: "whenever data are encrypted".

**4. What is the GDPR 72-hour rule?**
Strong answer: notify the supervisory authority within 72 hours of awareness unless the breach is
unlikely to result in a risk; individual notification follows if the risk is high, without undue
delay. Weak answer: "notify individuals in 72 hours".

**5. What do you do in the first hour?**
Strong answer: declare, appoint a commander and scribe, preserve evidence, contain without
destroying logs, notify the privacy and clinical leads, and start the clock. Weak answer: "call the
vendor".

**6. How do you scope a breach?**
Strong answer: use the data map, verify against logs and forensic findings, distinguish exposed from
accessed, and identify jurisdictions; update as facts change. Weak answer: "count the rows".

**7. What is a business associate's notification duty?**
Strong answer: notify the covered entity without unreasonable delay and within 60 days, with the
information needed for the covered entity's notification; the covered entity remains responsible.
Weak answer: "the vendor notifies the patients".

**8. How do you communicate with patients?**
Strong answer: specific facts, plain language, what to do, contact, multiple channels, translation,
and a single source of truth; test the letter with patients. Weak answer: "issue a press release".

**9. What makes a post-incident review effective?**
Strong answer: separates cause, response, and impact; produces funded, owned, dated actions; tracks
them to closure; updates the threat model. Weak answer: "a lessons-learned document".

**10. How do you test the plan?**
Strong answer: tabletop with the actual responders, at least annually, with injects for clinical
impact and notification, and an after-action report with actions. Weak answer: "we have a plan on
the intranet".

**11. How do you decide whether to pay a ransom?**
Strong answer: a documented decision by the executive with legal, clinical, and law-enforcement
input; consider the reliability of the decryptor, the evidence of exfiltration, sanctions and legal
risk, and the effect on patients; payment does not remove the notification obligation. Weak answer:
"we do not negotiate".

**12. What is the role of the board during an incident?**
Strong answer: receive the declaration and the material facts, approve the spending authority and
the notification strategy, and provide support without micromanaging the response; the board's
question is whether the organisation is meeting its obligations, not how the malware works. Weak
answer: "the board is informed after".

**Answer 11.** A ransom decision is an executive decision with legal, clinical, and law-enforcement
input. The considerations include whether the decryptor is reliable, whether data were exfiltrated,
sanctions and legal risk, the effect on patients of continued downtime, and the precedent for the
organisation. Payment does not remove the notification obligation, and it does not guarantee that
the data will not be published. The decision and its rationale should be recorded.

**13. How do you handle a breach that affects multiple covered entities through one processor?**
Strong answer: the processor notifies each covered entity within the contractual and legal period;
each covered entity assesses its own population and notifies its own regulators; coordination
prevents inconsistent messages, but does not transfer responsibility. Weak answer: "the processor
handles it".

**14. What is the first metric you would improve after an incident?**
Strong answer: the discovery-to-declaration gap, because it bounds every later step, followed by
dwell time and containment time; the metric should be reported to the executive. Weak answer:
"number of alerts".

## 15. Key terms

- **Breach** - an impermissible use or disclosure, presumed under HIPAA.
- **Business associate** - an entity handling PHI for a covered entity.
- **Chain of custody** - documented evidence handling.
- **Containment** - stopping the spread while preserving evidence.
- **Corrective action** - a funded, owned change from a review.
- **Discovery** - the first day the entity knew or would have known.
- **Dwell time** - time between intrusion and detection.
- **Eradication** - removing the cause.
- **Event** - an observable occurrence.
- **Forensic image** - a bit-for-bit copy of a system.
- **Four-factor assessment** - HIPAA breach risk analysis.
- **GDPR 72-hour clock** - supervisory-authority notification deadline.
- **High risk** - GDPR threshold for individual notification.
- **Incident** - a security event with adverse effect.
- **Incident commander** - the person leading the response.
- **Incident response plan** - documented roles and steps.
- **MTTD** - mean time to detect.
- **MTTR** - mean time to respond or recover.
- **Notification** - informing individuals, regulators, or media.
- **Playbook** - a scenario-specific procedure.
- **Post-incident review** - structured review of cause and response.
- **Presumption of breach** - HIPAA default unless rebutted.
- **Recovery** - restoring normal operations.
- **Safe harbour** - secured PHI is not a breach.
- **Scope** - data types, individuals, and jurisdictions affected.
- **Scribe** - the person maintaining the timeline.
- **Secured PHI** - encrypted per guidance or destroyed.
- **Severity tier** - the response level for an incident.
- **Substitute notice** - notice by website or media.
- **Supervisory authority** - the EU data protection regulator.
- **Tabletop exercise** - discussion-based rehearsal.
- **Timeline** - the timestamped record of the response.
- **Discovery-to-declaration gap** - time between awareness and formal declaration.
- **Breach coach** - external counsel retained for incident response.
- **Phased notice** - an initial notification updated as facts develop.
- **Egress monitoring** - detection of unusual outbound data.
- **Double extortion** - encryption plus data-theft leverage.
- **Notification population** - the individuals legally entitled to notice.
- **Evidence log** - the record of evidence handling and custody.
- **Ransomware** - malware that encrypts or steals data for extortion.
- **Notification deadline** - the legal date by which notice must be sent.
- **Regulatory notification** - notice to a supervisory authority or HHS.
- **Patient-safety review** - assessment of clinical harm from an incident.
- **Incident file** - the complete record of the response.
- **Containment decision** - the documented choice to isolate or shut down.
- **Notification template** - a pre-approved letter with fill-in facts.
- **Scope revision** - an update to the affected population as facts change.
- **Lessons register** - the tracked list of corrective actions.
- **Regulatory filing** - the formal notice to a regulator.
- **Substitute notice** - notice by website or media when direct notice is infeasible.
- **Incident severity** - the tier assigned at triage based on impact.

## 16. Sources

All URLs accessed 26 October 2026.

**US regulation and guidance**
- HHS Office for Civil Rights, *Breach Notification Rule*.
  https://www.hhs.gov/hipaa/for-professionals/breach-notification/index.html
- 45 CFR 164.400-414, *Breach Notification*.
  https://www.ecfr.gov/current/title-45/subtitle-A/subchapter-C/part-164/subpart-D
- HHS, *Breach Safe Harbor* guidance. https://www.hhs.gov/guidance/document/breach-safe-harbor
- HHS, *Fact Sheet: Ransomware and HIPAA*.
  https://www.hhs.gov/hipaa/for-professionals/security/guidance/cybersecurity/ransomware-fact-sheet/index.html
- Federal Trade Commission, *Health Breach Notification Rule*, 16 CFR Part 318.
  https://www.ftc.gov/business-guidance/privacy-security/health-privacy

**EU regulation and guidance**
- Regulation (EU) 2016/679 (GDPR), Articles 33 and 34.
  https://eur-lex.europa.eu/eli/reg/2016/679/oj
- Article 29 Working Party, *Guidelines on Personal Data Breach Notification under Regulation
  2016/679* (WP250rev.01), 2018.
  https://ec.europa.eu/newsroom/article29/items/612052

**Standards**
- NIST, *SP 800-61 Rev. 3, Incident Response Recommendations and Considerations for Cybersecurity
  Risk Management*, April 2025. https://csrc.nist.gov/pubs/sp/800/61/r3/final
- ISO/IEC 27035-1:2023, *Information technology - Information security incident management - Part 1:
  Principles and process*.
- ISO/IEC 27035-2:2023, *Part 2: Guidelines to plan and prepare for incident response*.

**Case evidence**
- HHS OCR, *Change Healthcare Cybersecurity Incident Frequently Asked Questions*.
  https://www.hhs.gov/hipaa/for-professionals/special-topics/change-healthcare-cybersecurity-incident-frequently-asked-questions/index.html
- HHS OCR, *Anthem Resolution Agreement*, 2018.
  https://www.hhs.gov/hipaa/for-professionals/compliance-enforcement/agreements/anthem/index.html
- National Audit Office, *Investigation: WannaCry cyber attack and the NHS*, 2017.
  https://www.nao.org.uk/reports/investigation-wannacry-cyber-attack-and-the-nhs/
- Singapore Ministry of Communications and Information, *Public Report of the Committee of Inquiry
  into the Cyber Attack on SingHealth*, 2019.
  https://www.mci.gov.sg/pressroom/news-and-stories/pressroom/2019/1/public-report-of-the-committee-of-inquiry-into-the-cyber-attack-on-singhealth

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic: [`08_governance.md`](08_governance.md)
- Next topic: [`10_regulatory-map.md`](10_regulatory-map.md)
- Related topic: [`04_threat-modeling.md`](04_threat-modeling.md)
- Survey file: [`../../course/16_privacy_security_ethics.md`](../../course/16_privacy_security_ethics.md)
- Labs: [`../../labs/`](../../labs/)
