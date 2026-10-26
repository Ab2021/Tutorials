# Governance

> **Module 16 - Privacy, Security, and Ethics**
> Topic 08 of 10
> Estimated study time: 3 hours
> Prerequisites: 06, 07
> Version 2026.3

## 1. Why this topic exists

Every other control in this module depends on someone having the authority and the obligation to
say no. A privacy assessment that nobody can enforce is advice. A fairness metric that nobody
reviews is a dashboard. An ethics review that always approves is a ceremony. Governance is the set
of decision rights, bodies, and evidence that turn those artefacts into operating constraints.

The commercial stakes are visible in procurement and in regulation. Health systems ask vendors who
approved the model, on what evidence, with what conditions, and who is accountable when it fails.
Regulators ask for the data protection impact assessment, the risk analysis, and the record of
processing. Investors ask whether the company can survive a serious incident. A governance
structure that exists only on an organisation chart does not answer any of those questions.

Three failure patterns recur.

**The committee without power.** A model review board meets monthly, reviews slide decks, and
offers comments. It cannot block a deployment, cannot require a change, and has no budget. Its
approval is a signal that the project team can ignore. The body exists to absorb risk, not to
manage it.

**The rubber stamp.** A data access committee reviews 240 requests a year in two-hour monthly
meetings. The arithmetic in Section 6 shows that the only way to process that volume is six minutes
per request. Most approvals carry no conditions, because there is no time to write them. The
committee is diligent and the process is not.

**The accountability gap.** Everyone agrees the model is "owned by the clinical team", and nobody
can name the person who signs the deployment decision, the person who reviews the monitoring
dashboard, or the person who calls a pause. When an incident occurs, the investigation finds a
diffusion of responsibility and no decision record.

This topic teaches governance as an operating system: who decides, on what evidence, with what
authority, and how the decision is reviewed. It covers data access committees, model review boards,
data trusts and data institutions, privacy and security officers, and the assurance functions that
test whether the controls work. It builds on the ethics assessment in Topic 07 and the access
control in Topic 05.

The distinction that runs through the topic is between a decision right, an advisory role, and an
assurance role. A decision right can say yes or no. An advisory role can recommend. An assurance
role can test and report. Bodies that blur the three produce decisions that nobody owns and
assurance that nobody acts on.

A governance structure also determines whether an incident is survivable. An organisation with a
decision record, a named owner, and an assurance trail can show a regulator that it identified the
risk, took a decision, and monitored it. An organisation with none of those can only show that it
was surprised.

Governance is also what makes a privacy programme durable. A privacy officer can approve a flow,
but only a decision right, recorded and reviewed, survives the officer's departure. The artefacts
of governance are the organisation's memory of why a control exists.

> **Note:** Governance is judged by the decisions it changes and the incidents it prevents, not by
> the number of committees it convenes.

## 2. Learning outcomes

By the end of this topic you can:

1. **Map** decision rights for a data or model programme, distinguishing decide, recommend, and
   assure.
2. **Charter** a review body with a purpose, membership, quorum, conflict-of-interest rules, and
   decision types.
3. **Compute** the throughput of a review process and **identify** when the volume makes meaningful
   review impossible.
4. **Design** a triage process that routes routine requests and reserves the full body for complex
   ones.
5. **Apply** conflict-of-interest and quorum rules, and **quantify** the risk of losing quorum.
6. **Compare** a data access committee, a model review board, a data trust, and a data institution,
   and choose the structure for a stated problem.
7. **Specify** the evidence a body needs to decide, and the record it must leave.
8. **Design** an assurance plan that tests whether governance decisions are implemented.

## 3. The core mental model

Governance has three layers: decision rights, decision bodies, and assurance. The layers connect
through artefacts: a charter, a decision record, and an audit report.

```text
  DECISION RIGHTS (who can say yes or no)
        |
        +--> Data access committee   (data sharing, secondary use)
        +--> Model review board      (clinical model deployment)
        +--> IRB / ethics committee  (human-subjects research)
        +--> Privacy officer         (DPIA, breach, data subject rights)
        +--> Security officer        (risk acceptance, incidents)
        |
  DECISION BODIES (how the right is exercised)
        |
        +--> intake -> triage -> review -> decision -> conditions
        |
  ASSURANCE (how anyone knows it happened)
        |
        +--> decision record | audit | monitoring | incident review
```

The first layer is the one most organisations skip. They create a committee and assume it has
authority, without writing down what it can decide, what it can only recommend, and what is outside
its scope. The result is a body that is consulted on everything and authoritative on nothing.

A second model is the three lines of defence, which is useful for separating management from
assurance:

| Line | Role | Example |
|---|---|---|
| First | Owns and manages the risk | Clinical team operating the model; data steward |
| Second | Sets policy, advises, monitors | Privacy office; security; model review board |
| Third | Independent assurance | Internal audit; external certification |

The model is not a hierarchy. The first line makes the decision and bears the consequence; the
second line provides the rules and the challenge; the third line tests both. A programme in which
the second line makes operational decisions has confused advice with ownership, and a programme in
which the third line reports to the first has no independent assurance.

A third model is the decision record. Every governance decision should leave a record with six
fields: the request, the evidence considered, the decision, the conditions, the owner, and the
review date. The record is what makes governance auditable and what allows a successor to
understand why a decision was made.

```text
  request --> evidence --> decision --> conditions --> owner --> review date
                              |             |
                              v             v
                        approve /        verification
                        decline /        before launch
                        defer / pilot
```

The final model is the data trust, which is a different structure rather than a committee. A data
trust is a legal structure that provides independent stewardship of data, in which trustees owe a
fiduciary duty to beneficiaries. It is useful when the parties have conflicting interests and the
data subjects need representation that a committee inside one organisation cannot provide. It is
not a synonym for a data-sharing agreement, and it is not a technology.

The decision record is also the memory of the organisation. Staff turn over, programmes are
reorganised, and the reasoning behind a condition is lost unless it is written down. A record that
states the condition, the evidence, and the review date allows a successor to continue the control
rather than reinvent it.

A governance body should also have a succession plan. A chair who has run the committee for a
decade is a single point of failure, and a membership that has never changed is a sign that the
independence is nominal. Term limits and a rotation schedule are governance controls, not
bureaucratic niceties.

**The triage rule is the constitution of the process.** It decides which requests get the
committee's attention, and therefore which risks are examined. A triage rule that routes by
requester status, rather than by risk, will systematically under-review the requests from the
least powerful requesters. The rule should be published, and its effect on different requester
groups should be reviewed annually.

The three layers are not optional in sequence. An organisation can have a charter without
assurance, and it will not know whether the charter is followed. It can have assurance without
decision rights, and it will produce findings that nobody can act on. The design that works links
the layers through artefacts that name a decision, an owner, and a date.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| Governance | The system of decision rights, bodies, and evidence that constrains an activity. | A committee chart. | Operating model. |
| Decision right | The authority to approve, decline, or condition a request. | Confused with advice. | Role. |
| Advisory role | A role that recommends but does not decide. | Treated as a veto. | Role. |
| Assurance | Independent testing and reporting on whether controls work. | A management self-assessment. | Function. |
| Accountability | A named party answerable for a decision and its effects. | A committee. | Role. |
| Data access committee (DAC) | A body that reviews requests for data access and sets conditions. | A help desk. | Body. |
| Model review board | A body that reviews clinical models before deployment and during monitoring. | A research seminar. | Body. |
| IRB | Institutional Review Board for human-subjects research. | A data access committee. | Body. |
| Privacy officer | The role responsible for privacy compliance and data subject rights. | A legal counsel. | Role. |
| Security officer | The role responsible for security risk and incidents. | An IT administrator. | Role. |
| Charter | The document that defines a body's purpose, authority, membership, and process. | A terms-of-reference draft. | Document. |
| Terms of reference | The scope and authority of a body. | Not written down. | Document. |
| Quorum | The minimum number of members required to decide. | Assumed to be everyone present. | Count. |
| Conflict of interest | A financial or personal interest that could bias a decision. | A disclosure form with no consequence. | Disclosure. |
| Recusal | Withdrawal of a member from a decision because of a conflict. | Optional in practice. | Procedure. |
| Triage | Routing requests to the right level of review. | Skipping review. | Process. |
| Decision record | The written record of a decision, evidence, conditions, owner, and review date. | Meeting minutes. | Document. |
| Condition | A requirement attached to an approval. | A recommendation. | Decision element. |
| Appeal | A route to challenge a decision. | An email to the chair. | Procedure. |
| Transparency | Publishing decisions, criteria, and aggregate statistics. | Publishing a policy. | Practice. |
| Data trust | A legal structure providing independent stewardship of data with fiduciary duties. | A data-sharing agreement. | Legal structure. |
| Data institution | An organisation that stewards data on behalf of others. | A vendor. | Organisation. |
| Data cooperative | A member-owned structure for data governance. | A user group. | Legal structure. |
| Fiduciary duty | A legal duty to act in the interests of another. | A general obligation of care. | Legal duty. |
| Beneficiary | The party whose interests a trustee must serve. | The data buyer. | Role. |
| Trustee | The party holding the fiduciary duty. | The data controller. | Role. |
| Data steward | The role responsible for the quality and appropriate use of a dataset. | A database administrator. | Role. |
| Risk acceptance | A documented decision to tolerate a residual risk. | Ignoring the risk. | Decision. |
| Materiality | The significance of a decision or risk to the organisation. | Size of the dataset. | Threshold. |
| Three lines of defence | A model separating operational ownership, policy, and independent assurance. | A reporting hierarchy. | Model. |
| Assurance plan | The schedule and methods for testing controls. | A compliance calendar. | Plan. |
| Whistleblowing channel | A confidential route to report wrongdoing. | An HR complaint form. | Mechanism. |
| Sunset clause | A provision that ends an arrangement after a period or event. | A renewal by default. | Contract term. |

## 5. The mechanics, step by step

The procedure below sets up or repairs governance for a data or model programme. It assumes that
the programme already has privacy, fairness, and ethics artefacts (Topics 01, 06, 07); governance is
the structure that makes those artefacts binding. Each step names its input, output, and rationale.

### Step 1. List the decisions that need governing

**Input:** Programme scope, data flows, model lifecycle, regulatory obligations.
**Output:** A decision inventory: each recurring decision, its frequency, and its consequence.
**Rationale:** Governance should be organised around decisions, not around org charts. A typical
programme has six recurring decisions: whether to share data with a recipient, whether to approve a
new purpose, whether to deploy a model, whether to continue a deployed model, whether to accept a
residual risk, and whether to disclose a breach. Each has a different owner, cadence, and evidence.

The inventory should include the decisions that are currently made informally: the analyst who
grants access, the product manager who changes the threshold, the engineer who adds a data source.
Those are the decisions where governance is missing, and they are usually the ones with the least
evidence.

### Step 2. Assign decision rights

**Input:** Decision inventory, roles, existing committees.
**Output:** A decision-rights matrix: decide, recommend, consult, inform, per decision.
**Rationale:** The matrix is the core governance artefact. It should be short enough to memorise
and specific enough to resolve a dispute. A decision with two deciders is a decision that will be
escalated; a decision with no decider is a decision that will be made by whoever is most persistent.

| Decision | Decide | Recommend | Consult | Inform |
|---|---|---|---|---|
| Share data with an external recipient | Data access committee | Data steward | Privacy office; clinical owner | Security; legal |
| Approve a new purpose for existing data | Privacy officer | Data steward | Ethics committee | Data access committee |
| Deploy a clinical model | Model review board | Clinical lead; data science lead | Ethics committee; patient advisory board | Executive sponsor |
| Continue or pause a deployed model | Clinical owner | Monitoring lead | Model review board | Quality committee |
| Accept a residual security risk | Security officer | Security team | Risk committee | Executive sponsor |
| Notify a breach | Privacy officer | Incident response lead | Legal; communications | Regulator (within the legal deadline) |

Branch point: if a decision has no clear decider, assign one before the next request. A temporary
assignment is better than an escalation.

### Step 3. Charter the body

**Input:** Decision rights, organisational context.
**Output:** A charter with purpose, scope, authority, membership, quorum, conflict rules, meeting
cadence, and decision types.
**Rationale:** A charter is the difference between a body that can act and a body that can only
advise. It should state what the body can decide, what it must escalate, what it must publish, and
how it handles urgency. The charter should be approved by the executive who owns the risk, not by
the body itself.

The charter should also state what the body does not cover. A model review board that is assumed to
cover data access will either duplicate the DAC or miss the data question. Explicit exclusions
prevent both.

### Step 4. Compose the membership and manage conflicts

**Input:** Decision types, stakeholder map, conflict disclosures.
**Output:** A membership list with roles, terms, and conflict rules; a recusal procedure.
**Rationale:** Composition determines the decisions. A body of engineers will optimise for
technical feasibility; a body of clinicians will optimise for workflow; a body without patients will
miss harms that patients experience. The membership should include the expertise the decisions
require and at least one member whose role is to represent the affected population.

Conflict of interest is not a disclosure form; it is a rule with a consequence. A member with a
financial interest in the vendor should recuse from the vendor's request, and the recusal should be
recorded. The quorum must account for recusals, or the body will lose the ability to decide on the
requests where independence matters most.

### Step 5. Design intake and triage

**Input:** Request types, volume, risk criteria.
**Output:** An intake form, a triage rule, and a service-level target.
**Rationale:** Most requests are routine and a small number are complex. A process that sends every
request to the full body either creates a backlog or reduces review to minutes. Triage routes by
risk: a de-identified aggregate release may be approved by the data steward under a standard; a
new commercial recipient with identifiable data goes to the full committee.

| Structure | Best for | Authority | Cost | Failure mode |
|---|---|---|---|---|
| Internal data access committee | Recurring data-sharing decisions inside one organisation | Delegated by the executive | Low to medium | Rubber stamp; quorum loss |
| Model review board | Clinical model deployment and monitoring | Delegated by the medical executive | Medium | No authority to block |
| IRB or ethics committee | Human-subjects research | Legal or institutional mandate | Medium | Scope mismatch with product work |
| Data trust | Conflicting interests; independent stewardship | Fiduciary duty to beneficiaries | High | Unclear enforcement; funding |
| Data institution or cooperative | Collective stewardship across organisations | Contractual or member-owned | Medium to high | Governance captured by the largest member |

The table is a selection aid. A programme may need a DAC for data and a model review board for
deployment, with a clear boundary between them. The boundary should be written down; otherwise the
same request is reviewed twice or not at all.

### Step 6. Define the evidence standard

**Input:** Decision types, prior decisions, regulatory requirements.
**Output:** A checklist of evidence per decision, with the minimum acceptable form.
**Rationale:** A body cannot decide well on a slide deck. The evidence standard should name the
artefacts: the purpose statement, the data-flow map, the DPIA, the fairness metrics with intervals,
the validation report, the monitoring plan, the contract, and the security assessment. The standard
should also state what is not acceptable: vendor-reported performance without independent
validation, a fairness claim without subgroup data, a security assertion without a test.

The evidence standard is where governance connects to the rest of the module. A body that does not
require the artefacts is a body that cannot use them.

### Step 7. Run the review and decide

**Input:** Request, evidence, charter.
**Output:** A decision with conditions, owner, and review date.
**Rationale:** The meeting should be structured around the decision, not the presentation. A useful
agenda is: the request, the evidence gaps, the harms and who bears them, the conditions, the
decision. The chair should ensure that dissent is recorded and that the decision is one of the
defined outcomes.

| Outcome | When to use | Conditions required |
|---|---|---|
| Approve | Evidence complete; residual risk acceptable | Monitoring; review date |
| Approve with conditions | Evidence adequate; specific gaps | Named conditions; verification before launch |
| Pilot | Evidence promising but incomplete | Evaluation protocol; stop criteria; population limit |
| Defer | Evidence missing or questions unresolved | Questions; owner; date |
| Decline | Harm unacceptable, no basis, or no mitigation | Rationale; alternatives |

A decision that is not one of these five is usually a deferral in disguise. "Approved subject to
further discussion" is a deferral, and it should be recorded as one.

### Step 8. Record, publish, and handle appeals

**Input:** Decision, evidence, dissent.
**Output:** A decision record, a publication entry, and an appeal route.
**Rationale:** The record is the evidence that governance happened. It should be written within a
week, stored where the programme can find it, and linked to the data or model it governs. The
publication should include the decision, the conditions, and aggregate statistics; it should not
include the requesters' confidential information.

The appeal route should name the body or role that hears appeals, the grounds, and the timeline. An
appeal that goes back to the same body with the same members is not an appeal.

### Step 9. Assure implementation

**Input:** Decision records, conditions, monitoring data.
**Output:** An assurance plan and a report: which conditions were implemented, which were not, and
what happened.
**Rationale:** A condition that is not implemented is a decision that was not made. Assurance should
test a sample of conditions each quarter: was the retention job run, was the access review
completed, was the monitoring dashboard reviewed, was the vendor's subprocessor list updated. The
report should go to the executive who owns the risk, not only to the body that set the condition.

### Step 10. Measure and adapt

**Input:** Decision data, assurance reports, incidents.
**Output:** A governance scorecard and a revision of the charter or process.
**Rationale:** Governance should be measured by outcomes, not activity. Useful measures: the
proportion of decisions with conditions, the time from request to decision, the proportion of
conditions verified, the number of appeals and their outcomes, the number of decisions reversed on
appeal, and the number of incidents traceable to a governance gap. A body that approves everything
with no conditions and has no appeals is not necessarily effective; it may be captured.

### Delegation and reserved decisions

A charter that reserves every decision to the full body creates the throughput failure in Section
6. The charter should delegate routine decisions to a named role and reserve the rest. A workable
split has three tiers:

| Tier | Examples | Decider | Record |
|---|---|---|---|
| Routine | Aggregate or de-identified release to an academic recipient under a standard contract | Data steward | Standard form; monthly report to the committee |
| Standard | Identifiable data to a known recipient; a new purpose within an approved programme | Subcommittee or chair | Decision record; ratified at the next meeting |
| Reserved | Commercial recipient, free text, minors, cross-border, novel use, a decision to decline | Full committee | Full decision record with conditions |

The reserved list is the important part. It names the decisions that cannot be delegated because
the risk or the precedent is significant. A charter without a reserved list will delegate the
decisions that matter and reserve the ones that do not.

### The evidence pack in practice

A committee that receives a 60-slide deck will discuss the slides. A committee that receives a
structured pack can compare requests. The pack should be short: a one-page summary, the purpose
statement, the data-flow map, the de-identification or privacy assessment, the fairness and
validation summary where relevant, the contract, and the security assessment. The summary should
state the decision requested, the risks, the mitigations, and the residual risk.

The pack should be submitted before the meeting, and the meeting should not begin if the pack is
incomplete. That rule is unpopular and effective: it converts the evidence standard from an
aspiration into a precondition.

### Governance of vendors and subprocessors

A vendor is part of the governance boundary, not outside it. The charter should require, for any
vendor that handles data or influences a clinical decision: a business associate or data processing
agreement, a subprocessor list, a security assessment, an incident notification clause, audit
rights, and a termination and data-return plan. The vendor's own governance (certifications,
audits, model documentation) is evidence, not a substitute for the organisation's decision.

The failure mode is a vendor that is approved once and never reviewed. The charter should require
an annual vendor review, with a trigger for a change of ownership, a change of subprocessors, a
security incident, or a material model update.

### The decision meeting

The meeting is a decision forum, not a seminar. A workable agenda allots ten minutes to the
request and the evidence gaps, ten to the harms and who bears them, ten to the conditions, and five
to the decision. The chair states the decision at the end and confirms that the record captures it.
Members who dissent should say so on the record, and the dissent should be included in the
published summary.

The chair's role is to protect the process, not to advocate for the programme. A chair who is also
the programme sponsor cannot run a review of the programme. The charter should separate the two
roles, or require an independent chair for decisions involving the sponsor's area.

### Records and registers

Two records matter: the decision record for each request, and the register of decisions. The
register allows the organisation to see patterns (a recipient that repeatedly requests broad data,
a purpose that keeps recurring, a condition that is never verified) and to answer a regulator's
question about how many requests were approved and on what terms.

The register should hold the request identifier, the date, the decision, the conditions, the owner,
and the review date. It should not hold patient identifiers, and it should be published in
aggregate. The decision record holds the detail and is access-controlled.

### Independence and the chair

Independence has three components: financial (no material interest in the outcome), organisational
(not reporting to the programme sponsor), and informational (access to the evidence the decision
requires). A chair who is independent financially but reports to the sponsor is not independent
organisationally. A member who is independent but receives a summary instead of the evidence is not
independent informationally.

The charter should state the independence requirements and how they are tested. A annual
declaration is the minimum; a review of the body's composition and decision patterns is better. The
test of independence is not the absence of a conflict declaration; it is whether the body has
declined a request the sponsor wanted approved.

### Transparency

Transparency is a governance control because it creates accountability to people outside the
decision. The minimum is a published register of decisions in aggregate, the criteria used, the
composition of the body, and the process for appeals. The next level is publishing the decision
records with commercial information redacted, which allows researchers and community groups to see
how the criteria are applied.

Transparency has limits. A register that identifies a small study, a rare condition, or a
complainant can cause harm. The publication should be aggregate, delayed where necessary, and
reviewed for small cells. The principle is to publish the pattern and protect the person.

### What good looks like

A functioning governance body has seven observable properties: a charter with decision rights; a
composition that includes the affected population; a triage rule that reserves complex decisions; an
evidence standard that requires artefacts; a decision record with conditions and owners; an
assurance sample that tests implementation; and a published register. None of the seven is
expensive, and all of them are rare in combination.

The test is not a document review. Ask for the last three decisions, the conditions attached, the
evidence that the conditions were met, and the name of the person who would pause the programme.
If any of the four cannot be produced in a day, the governance is not operating.

### The failure test

An organisation can test its governance with a single question: what happened the last time a
request was declined? If the answer is "nobody remembers", the body has not declined. If the answer
is "the sponsor went to the executive and got it approved", the decision rights are not real. If the
answer is "the request was revised and approved with conditions", the governance is working. The
question is uncomfortable and diagnostic.

The review depth is a policy choice, and it should be made explicitly. A committee that chooses six
minutes because that is what fits the calendar has not chosen a review depth; it has accepted one by
default. The chair should state the intended depth, the volume it can support, and what happens to
the requests beyond that volume.

### Edge cases

**Small cells.** A data access committee that publishes its decisions can disclose the existence of
a small study or a rare condition. The publication should be aggregate and delayed where necessary.

**Quasi-identifiers.** A request for a "small de-identified extract" can be a re-identification
request. The evidence standard should require the de-identification determination from Topic 02,
not only a statement that the data are de-identified.

**Re-identification through linkage.** A recipient with auxiliary data can link a release. The
committee should ask what the recipient already holds, not only what the release contains.

**Free-text PHI.** A request that includes notes should trigger a separate justification and a
separate condition. The committee should not approve notes as part of a structured extract without
an explicit decision.

**Metadata leakage.** The decision record and the published register can contain requester names,
institution names, and study details. The publication policy should cover metadata.

**Consent withdrawal.** A governance decision that relies on consent should state how withdrawal
will be handled and who is responsible for propagating it. A committee that approves a use without
a withdrawal process has approved a use it cannot stop.

**Minors and deceased patients.** Requests involving minors or decedents need the additional
authorisation analysis from Topic 01. The committee should require it, not assume it.

**Cross-border transfer.** A request from a recipient in another country is a transfer decision, not
only an access decision. The committee should require the transfer analysis and the contractual
terms.

**Urgency.** An emergency request can bypass the meeting cadence, but not the record. A chair's
action between meetings should be documented and ratified at the next meeting.

**Committee capture.** A body whose members are all employed by the sponsor, or all dependent on
the same vendor, cannot provide independent challenge. The charter should require independence and
the composition should be reviewed annually.

## 6. Worked example

### Setting

A regional health system's data access committee (DAC) reviews requests to use its research data
platform. The committee has nine members, meets monthly, and requires a quorum of five. It received
240 requests last year. Three members hold paid advisory roles with life-science companies, and
those members disclose a conflict on 41 percent of requests. Members attend 80 percent of meetings
on average. The privacy office has asked whether the committee is functioning as a control.

### Step 1: throughput arithmetic

The committee meets 12 times a year and receives 240 requests, so the average meeting has 20
requests. If each request received 40 minutes of review, the demand would be:

20 x 40 = 800 minutes, or 13.3 hours.

The meeting is scheduled for two hours, or 120 minutes. At 40 minutes per request, the capacity is
120 / 40 = 3 requests per meeting. The excess is 800 - 120 = 680 minutes per meeting, which is 17
requests that cannot be reviewed at that depth.

The committee has not hired more staff; it has compressed the review. The average time available
per request is 120 / 20 = 6 minutes. That is the arithmetic behind the first diagnostic: a
six-minute review cannot examine a data-flow map, a de-identification determination, and a
contract. It can check that the form is complete.

### Diagnostic 1: approvals without conditions

Of 240 decisions last year, 91 carried conditions and 149 did not. The share without conditions is
149 / 240 = 62.1 percent. For a committee whose purpose is to attach safeguards to data sharing, a
62 percent clean-approval rate is a signal that the conditions step is being skipped, not that the
requests are unusually safe. The six-minute arithmetic explains why.

The diagnostic is not that conditions are always necessary. Some requests are genuinely routine. The
diagnostic is the combination of volume, time per request, and condition rate: the process cannot
produce conditions at the rate it is being asked to decide.

### Diagnostic 2: quorum loss under conflicts

The three conflicted members recuse on 41 percent of requests, an expected 3 x 0.41 = 1.23 recusals
per request. On a request where all three have a conflict, the eligible voters are 9 - 3 = 6. With
a quorum of five, the committee can decide only if at least five of those six attend. At an 80
percent attendance rate, the probability that at least five of six attend is:

P(5 of 6) + P(6 of 6) = 6 x 0.8^5 x 0.2 + 0.8^6 = 0.3932 + 0.2621 = 0.6554.

The probability of losing quorum is therefore 1 - 0.6554 = 0.3446, or 34.5 percent. On the requests
where independence matters most, the committee fails to decide about a third of the time. The
failure is not random: it is concentrated on the conflicted requests, which are the ones involving
industry-funded research.

### Vary one assumption: add an independent member

Suppose the health system adds one independent member with no industry ties, bringing the committee
to ten. On a fully conflicted request, the eligible voters are 10 - 3 = 7. The quorum probability
becomes the chance that at least five of seven attend:

P(5) + P(6) + P(7) = 21 x 0.8^5 x 0.2^2 + 7 x 0.8^6 x 0.2 + 0.8^7
= 0.2753 + 0.3670 + 0.2097 = 0.8520.

The probability of losing quorum falls from 34.5 percent to 14.8 percent. The conclusion changes
from "the committee cannot reliably decide conflicted requests" to "the committee can usually
decide them", at the cost of one additional member's time.

The throughput problem is unchanged. Adding a member does not create review minutes; the committee
still has 120 minutes for 20 requests. The two diagnostics require different remedies, which is the
point of measuring both.

### The triage remedy

The committee redesigns its process around risk. Routine requests, defined as aggregate or
de-identified releases to academic recipients with a standard contract, are approved by the data
steward against a checklist within five working days. Complex requests, defined as identifiable
data, commercial recipients, notes, or cross-border transfer, go to the full committee.

The arithmetic after triage:

- 200 routine requests per year, at 8 minutes each, are handled outside the meeting: 1,600 minutes
  per year, or 2.2 hours per week for the steward.
- 40 complex requests per year go to the committee: 40 / 12 = 3.3 per meeting. At 40 minutes each,
  the demand is 133 minutes, slightly above the 120-minute meeting. The committee extends the
  meeting to 2.5 hours or holds an additional session for complex requests.

The condition rate should rise, because the committee now has time to write conditions for the
requests that need them. The measure to track is not the number of approvals but the proportion of
complex approvals with verified conditions.

### The decision record

The review produces a decision with four elements:

1. Triage criteria and the steward's delegated authority, approved by the executive.
2. One additional independent member, with a term and a conflict declaration.
3. A quarterly assurance sample of 10 percent of routine approvals, to test that the checklist is
   being applied.
4. A review of the committee's condition rate and quorum loss after two quarters.

Each element has an owner and a date. The record is the artefact that the privacy office can show
to an auditor, and it is the difference between a governance reform and a discussion.

### What the example does not resolve

The arithmetic does not decide whether 40 minutes is the right review depth, whether 200 routine
requests are genuinely routine, or whether the steward has the competence to approve them. Those
are judgements about risk and delegation. The example shows how to make the judgements visible and
how to test whether the process can deliver them.

### What the committee should have measured

The committee's own performance is a governance question. Four measures matter: the time from
request to decision (a service measure), the proportion of decisions with conditions (a quality
measure), the proportion of conditions verified (an implementation measure), and the number of
decisions reversed on appeal (an independence measure). A committee that reports only the number of
requests processed is measuring activity, not governance.

The measures should be published in aggregate, with the small-cell protections from Topic 02. A
register that lists every request and requester is a transparency artefact that can also be a
disclosure.

### The limits of the arithmetic

The throughput and quorum calculations are screening tools. They do not measure the quality of a
decision, the competence of a member, or the independence of a chair. A committee with excellent
arithmetic and a captured chair is still captured. The numbers are necessary because they rule out
processes that cannot work; they are not sufficient because a process that can work may still be
used badly.

The script is a planning tool. It can be run before a charter is approved to test whether the
proposed meeting cadence, quorum, and membership can handle the expected volume. A charter that
cannot pass this arithmetic should be revised before it is signed, because the alternative is a
committee that meets the letter of its terms of reference and fails its purpose.

### The assurance sample

The triage design delegates 200 routine approvals to the steward. The assurance question is whether
the delegation is being used correctly. A quarterly sample of 20 approvals (10 percent) checked
against the checklist gives a measurable error rate. If 2 of 20 lack the required contract, the
estimated error rate is 10 percent, with a wide interval; the committee should tighten the checklist
or the training rather than assume the process is working.

The sample should be random, not selected by the steward. The reviewer should be independent of the
approval, and the findings should go to the executive who owns the risk. An assurance sample that is
selected and reviewed by the same team is a self-assessment.

### A second variation: reduce the volume

Suppose the platform halves its request volume to 120 per year, for example by consolidating
duplicate requests from the same research group. The arithmetic becomes 10 requests per meeting.
At the same 120-minute meeting, the time per request is 12 minutes, double the six minutes in the
base case. The condition rate might rise, but 12 minutes is still short for a data-flow map and a
contract.

The conclusion changes only partially. Halving the volume improves review depth from six to twelve
minutes, which is better but not sufficient for complex requests. The structural fix is still
triage: separate the routine from the complex, and give the complex requests the time they need.
Volume reduction helps, but it does not substitute for differentiation.

The 62.1 percent clean-approval rate is the number to put beside the six minutes. A committee that
has time for six minutes per request cannot write conditions, and the condition rate reflects the
time, not the risk.

## 7. Code, query, or calculation

The script below computes the throughput, the condition rate, and the quorum probability under
three composition scenarios. It uses only the standard library.

```python
"""Data access committee throughput and quorum arithmetic."""
from math import comb


def binomial_tail(n, q, p):
    """Probability that at least q of n members attend, each with probability p."""
    return sum(comb(n, k) * p ** k * (1 - p) ** (n - k) for k in range(q, n + 1))


def throughput(requests_per_year, meetings, minutes_per_request, meeting_minutes):
    per_meeting = requests_per_year / meetings
    demand = per_meeting * minutes_per_request
    capacity = meeting_minutes / minutes_per_request
    return per_meeting, demand, capacity, demand - meeting_minutes


if __name__ == "__main__":
    per_meeting, demand, capacity, over = throughput(240, 12, 40, 120)
    print(f"requests per meeting={per_meeting:.0f}, review demand={demand:.0f} min, "
          f"capacity at 40 min={capacity:.1f} requests, over={over:.0f} min")
    print(f"minutes available per request={120 / per_meeting:.1f}")

    approvals, with_conditions = 240, 91
    print(f"approvals without conditions={approvals - with_conditions}/{approvals} "
          f"({(approvals - with_conditions) / approvals:.1%})")

    for label, n, q, p in (("3 conflicted members (6 eligible)", 6, 5, 0.8),
                           ("2 conflicted members (7 eligible)", 7, 5, 0.8),
                           ("no conflicts (9 eligible)", 9, 5, 0.8)):
        prob = binomial_tail(n, q, p)
        print(f"{label}: P(quorum)={prob:.4f}, P(no quorum)={1 - prob:.1%}")

    recusals = 3 * 0.41
    print(f"expected recusals per request={recusals:.2f}")
```

Expected output, verified on Python 3.11:

```text
requests per meeting=20, review demand=800 min, capacity at 40 min=3.0 requests, over=680 min
minutes available per request=6.0
approvals without conditions=149/240 (62.1%)
3 conflicted members (6 eligible): P(quorum)=0.6554, P(no quorum)=34.5%
2 conflicted members (7 eligible): P(quorum)=0.8520, P(no quorum)=14.8%
no conflicts (9 eligible): P(quorum)=0.9804, P(no quorum)=2.0%
expected recusals per request=1.23
```

Three readings of the output. The capacity of 3 requests per meeting against a demand of 20 is the
throughput failure. The quorum probability of 0.6554 is the independence failure, and it is
concentrated on conflicted requests. The recusal expectation of 1.23 explains why the eligible
voter count is usually six rather than nine.

The model assumes independence between members' attendance, which is false when members carpool,
share a clinical schedule, or attend remotely. A more realistic simulation would use historical
attendance correlations. The arithmetic is a screening tool; the committee's actual attendance
record is the evidence.


To adapt the script, replace the request volume, the meeting cadence, the review time, and the
conflict rate with the committee's actual data. Add two things: a distribution of review times
rather than an average, because a few complex requests dominate the meeting; and a record of
attendance correlation, because members who travel together miss together.

The output should be reported to the executive who owns the risk, not only to the committee. A
committee that discovers its own throughput failure and reports it upward is doing governance; one
that keeps meeting and approving is not.

The quorum output is a probability, not a prediction for a specific meeting. It assumes independent
attendance at 80 percent, which is a simplification; in practice attendance is correlated with
agenda, season, and travel. The number is useful for comparing compositions, not for forecasting a
particular vote.

### Reading the output

The 13.3-hour demand against a two-hour meeting is not a scheduling problem; it is a statement that
the committee has been given a volume it cannot review. The 6-minute figure is the number to show
an executive, because it converts "we need more resources" into "this is the review depth you are
currently funding".

The model is deliberately simple enough to reproduce in a spreadsheet. A committee can run it with
its own request volume and attendance data before its next charter review, and the result is
usually uncomfortable enough to prompt a change.

A reader can check the throughput by hand: 240 / 12 = 20; 20 x 40 = 800; 800 - 120 = 680; 120 / 20
= 6. The quorum calculation is the only part that needs the binomial, and it is included because a
committee's quorum rule is often written without considering recusals.

The script is not a substitute for reading the requests. It tells a committee whether its process
can work; the quality of the decisions depends on the evidence and the judgement applied to each
one.

The output is a set of questions for the committee, not a verdict. A capacity of three requests per
meeting means the committee must decide whether to triage, extend the meeting, or accept a shallower
review; the script does not choose.

## 8. Data, standards, and regulatory context

Governance obligations are spread across data protection law, sector regulation, and management
system standards. The law usually names a role and a duty; the standards describe the structure that
makes the duty demonstrable.

### GDPR

- Article 24 makes the controller responsible for compliance and requires appropriate technical and
  organisational measures, including data protection policies.
- Article 28 governs processors and requires a written contract with specific terms, including
  subprocessor authorisation and audit rights.
- Article 30 requires records of processing activities.
- Article 35 requires a DPIA for high-risk processing and, where the DPIA indicates high residual
  risk, prior consultation with the supervisory authority.
- Article 37 requires a data protection officer for certain controllers and processors, and
  Articles 38 and 39 define the DPO's independence and tasks.
- Article 5(2) requires the controller to be able to demonstrate compliance, which is the
  accountability principle that governance exists to satisfy.

The DPO role is often misunderstood. The DPO advises, monitors, and is a contact point for the
supervisory authority; the DPO does not decide whether processing may occur, and cannot be
instructed on the exercise of the role. An organisation that makes the DPO the approver has given
the role an operational decision right that the regulation does not assign to it.

### HIPAA

- 45 CFR 164.308(a)(2) requires a designated security official responsible for the security
  policies and procedures.
- 45 CFR 164.530(a) requires a designated privacy official responsible for privacy policies and
  procedures, and 164.530(b) requires workforce training.
- 164.530(e) requires a process for receiving and resolving complaints, and 164.530(f) requires
  sanctions for workforce members who violate privacy policies.
- 164.316 requires documentation and six-year retention.
- The Security Rule's administrative safeguards include risk analysis, risk management, sanction
  policy, and information system activity review, all of which are governance controls.

### AI and data governance standards

- The NIST AI Risk Management Framework (AI RMF 1.0, January 2023) includes a Govern function that
  covers accountability structures, risk tolerance, processes, and documentation. It is voluntary
  and is the reference many US organisations use for an AI governance programme.
- ISO/IEC 42001:2023 specifies an AI management system, including leadership, planning, support,
  operation, performance evaluation, and improvement. It is certifiable.
- ISO/IEC 38500:2015 provides principles for IT governance: responsibility, strategy, acquisition,
  performance, conformance, and human behaviour. It is a board-level framework.
- The OECD AI Principles include accountability as a core value, with a recommendation that
  organisations be accountable for the proper functioning of AI systems.

### Sector-specific governance

- The European Health Data Space Regulation (EU) 2025/327 establishes health data access bodies
  that issue data permits and supervise secondary use, with application from 2029. It is the
  clearest example of governance being written into the access route rather than left to the
  organisation.
- The UK Caldicott principles (revised 2020) set expectations for the use of confidential health
  data, including justifying the purpose, using the minimum necessary, and informing patients. They
  are professional guidance, not statute.
- FDA premarket submissions for medical devices include quality system and cybersecurity
  documentation that assume a governance structure with design controls, risk management, and
  post-market surveillance.

**Governance and culture.** A charter cannot create a culture in which people raise concerns. The
organisation needs a whistleblowing channel, a non-retaliation policy, and leaders who visibly
support refusals. The soft controls are harder to audit, and they determine whether the hard
controls are used.

> **Note:** Regulatory content in this file is educational and is not legal advice. Governance
> structures must be adapted to the entity, the jurisdiction, and the risk, with counsel.

The classification to keep straight: GDPR Article 24 and HIPAA 164.308 are binding; the NIST AI RMF
and ISO/IEC 42001 are voluntary frameworks (certification is a business choice); a data trust is a
legal structure, not a standard; and a committee charter is an internal instrument. A vendor claim
of "ISO 42001 certified" is evidence about a management system, not about a specific model.

## 9. Industry practice and case evidence

### care.data, 2013-2016

NHS England's care.data programme was intended to extract general-practice data into a central
repository for research and planning. It was paused in 2014 after criticism of its communication and
opt-out arrangements and closed in 2016, and the data collected were destroyed. The documented
failure was one of public legitimacy and governance, not of a specific technical control: patients
were not adequately informed, the opt-out was confusing, and the oversight arrangements did not
command confidence.

The governance lesson is that a body cannot manufacture legitimacy after the fact. The programme
had a legal basis and a governance structure, and it still failed because the affected population
did not accept the arrangement. Consultation and transparency are governance controls, not
communications tasks.

### DeepMind Health and the Independent Reviewers panel, 2016-2019

DeepMind Health established an Independent Reviewers panel in 2016 to scrutinise its work with the
NHS. In 2018 DeepMind announced that its health team would move to Google Health, and in 2019,
during the transfer, the panel was disbanded. The panel's chair, Donal O'Donoghue, publicly
expressed disappointment that the experiment did not have time to run its course, and the panel's
2018 report had warned about the risk of excessive monopoly power. Google said the UK-focused
structure was not right for a global effort.

The case is the clearest recent example of governance independence being lost when the corporate
structure changes. The panel had advisory power, not decision rights, and when the sponsor changed,
the advisory structure was removed. A governance design that depends on a sponsor's continued
commitment is not a control; it is a gesture. The lesson for a charter is to attach the body to a
legal or contractual obligation, with a defined term and a transition clause, rather than to a
programme's goodwill.

### Data access committees in practice

NIH's database of Genotypes and Phenotypes (dbGaP) uses data access committees to review requests
for controlled-access data, with a data use certification and institutional sign-off. The model is
well documented and has been operating for more than a decade. Its strengths are a standard
request, a defined review, and a record of approved uses; its weaknesses are the variability
between committees and the difficulty of auditing downstream compliance.

The practical lesson is that a DAC works when the request format, the evidence standard, and the
decision record are standardised. A committee that invents its process at each meeting cannot be
compared across requests or audited.

### Data trusts and data institutions

The Open Data Institute defines a data trust as "a legal structure that provides independent
stewardship of data", with trustees owing a fiduciary duty to beneficiaries. The ODI's legal
research found that a traditional trust is not a good fit because data are not property in the
trust-law sense, and it assessed corporate, contractual, and community interest company models.
Data institutions are the broader category: organisations that steward data on behalf of others.

The model is attractive where parties have conflicting interests and data subjects need
representation that an internal committee cannot provide. It is not a solution to every governance
problem: it is expensive, the enforcement of fiduciary duties over data is untested in many
jurisdictions, and a trust can be captured by its funder if the charter is weak.

### Secure data environments

The UK's secure data environments and the EHDS secure processing environments are governance
mechanisms as much as technical ones: access is by project, the environment is controlled, and
outputs are reviewed before release. The governance question shifts from "who gets a copy" to "who
gets a project, for what purpose, with what output controls". That shift is the direction of
travel in both jurisdictions.

**Governance is a product feature.** A vendor that can describe its review board, its decision
records, and its incident history has a procurement advantage over one that cannot. The artefact
that wins the review is not the org chart; it is the decision record for a comparable deployment,
with the conditions and the evidence that they were met.

**Governance is not a substitute for competence.** A committee of well-meaning generalists cannot
evaluate a model's calibration or a contract's indemnity clause without expert input. The body
should have access to the expertise it needs, either in membership or through a standing adviser,
and the advice should be recorded.

A governance structure that cannot survive a change of sponsor is not a control. The charter should
attach the body to an institutional obligation (a bylaw, a contract, or a regulatory requirement)
with a defined term, a transition clause, and a requirement that any dissolution be reported. The
DeepMind panel is the cautionary case: a review body created by a sponsor disappeared when the
sponsor reorganised.

The trust/institution distinction matters in practice because the legal form determines who can be
held to account. A committee's decision can be reviewed internally; a trustee's decision can be
challenged in court if the fiduciary duty is breached. The stronger remedy is worth the cost only
when the conflict of interest is structural, not when it is a single difficult request.

**Data access committees are not research ethics committees.** A DAC asks whether the data may be
shared and on what terms; an IRB asks whether the research protects participants. The two can reach
different conclusions on the same protocol, and a programme needs both where human-subjects research
is involved. Combining them into one body saves meetings and loses expertise.

The common thread in these cases is that governance failed before the technology did. care.data had
a legal basis and lost legitimacy; DeepMind had a review panel and lost it in a reorganisation;
dbGaP has a standard process and still struggles with downstream compliance. A governance design
should be judged by how it behaves when the sponsor changes, the public objects, or the volume
grows, not by how it looks at launch.

> **Note:** The case facts above are drawn from the cited public sources. Statements about what an
> organisation should have done differently are inference, labelled as such.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Committee without power | Decisions are recommendations | No delegated authority | Advice ignored | Charter with decision rights |
| Rubber stamp | High approval rate; few conditions | Volume exceeds review time | Safeguards not applied | Triage; capacity; condition tracking |
| Quorum loss | Meetings cancelled or decisions deferred | Recusals plus attendance | No decision on conflicted requests | Independent members; remote voting |
| Conflict capture | Members with ties to applicants | Financial interest | Biased decisions | Recusal; disclosure; independent chair |
| Accountability gap | "The team owns it" | No named decider | No one pauses the system | Decision-rights matrix |
| Evidence-free review | Slide decks only | No standard artefacts | Decisions on assertions | Evidence checklist |
| No decision record | Minutes without conditions | Process not designed | Unauditable decisions | Record template |
| Appeals to the same body | Same members rehear | No independence | No real challenge | Separate appeal panel |
| Conditions not verified | No follow-up | Assurance gap | Decisions unimplemented | Quarterly assurance sample |
| Governance after launch | No pre-deployment gate | Process sequence | Design cannot change | Gate at design |
| Body scope overlap | Two bodies review the same request | Unclear boundary | Duplication or gap | Written boundary |
| Publication leaks | Register includes small studies | Aggregation failure | Privacy harm | Aggregate; delay; suppress |
| Data trust without funding | Trustees unpaid; no secretariat | Unsustainable structure | Capture or collapse | Funding and term |
| Sponsor can dissolve the body | Charter is internal only | No legal anchor | Independence lost | Contractual term; transition clause |
| No decline in years | Approval rate 100% | Body is advisory or captured | Harm legitimised | Decision-rights review |
| Register discloses small studies | Publication includes study titles | Aggregation failure | Privacy harm | Aggregate; delay; suppress |
| No succession plan | Same chair for a decade | Independence becomes nominal | Capture over time | Term limits; rotation |
| Governance after the contract | No review of vendor changes | Contract is static | Unreviewed subprocessors | Annual vendor review |

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test |
|---|---|---|---|
| Decision rights are documented | Accountability depends on them | Informal decisions | Compare matrix to practice |
| The body has authority to decline | Governance is a control | Charter is advisory | Count declines |
| Members are independent enough | Challenge depends on independence | All from one sponsor | Conflict register; composition review |
| Quorum accounts for recusals | Conflicted requests need decisions | Quorum set without recusal | Quorum simulation |
| Review time matches volume | Conditions require time | Volume exceeds capacity | Throughput arithmetic |
| Evidence standard is followed | Decisions are informed | Slides replace artefacts | Sample decision files |
| Conditions are implemented | Approval means something | No verification | Assurance sample |
| Appeals are independent | Redress is real | Same body rehears | Appeal log |
| The body is reviewed | Governance drifts | No self-assessment | Annual review |
| Publication is safe | Transparency does not harm | Small cells in register | Disclosure review |
| The structure fits the problem | A DAC is not a trust | Structure chosen by fashion | Options comparison |
| The board sets risk appetite | Decisions have a reference point | No appetite statement | Board review |
| Assurance is independent | Findings are credible | Self-assessment | Reviewer independence check |
| Delegated authority is documented | The steward can act | Informal approvals | Compare approvals to charter |
| The register is maintained | Transparency is possible | No one owns it | Quarterly completeness check |

Two limits are structural. First, governance cannot make a bad model good; it can only prevent a
bad model from being deployed without conditions. Second, a body's decisions are only as good as
its information, and the information often comes from the party seeking approval. Independent
validation and access to the underlying data are governance requirements, not only methodological
ones.

## 12. Exercises

The exercises are ordered from design to assurance. Exercises 1 to 3 define the structure and test
whether it can work; 4 and 5 test the arithmetic; 6 to 8 apply the structure to a trust, an
assurance plan, and an appeal. Exercise 3 is the one that most often changes a charter, because it
shows that the proposed cadence cannot support the proposed review depth.

Each exercise produces an artefact that could be adopted by a real body: a matrix, a charter, a
calculation, a triage rule, a comparison, an assurance plan, or an appeal process. The calculations
can be done by hand; the design exercises need a real or realistic programme.

**Exercise 1 - Decision-rights matrix (application, 45 minutes).**
For a hospital deploying a clinical model, list six recurring decisions and assign decide,
recommend, consult, and inform. Deliverable: matrix. Criteria: one decider per decision; no
decision assigned to a committee that cannot decline.

**Exercise 2 - Charter a body (design, 60 minutes).**
Write a one-page charter for a model review board: purpose, scope, authority, membership, quorum,
conflicts, cadence, decision types. Deliverable: charter. Criteria: authority includes decline;
quorum accounts for recusal; term and transition clause present.

**Exercise 3 - Throughput arithmetic (calculation, 40 minutes).**
A committee receives 180 requests a year, meets monthly, and has a two-hour meeting. Compute the
minutes available per request at 30 minutes of review depth, and the number of requests that cannot
be reviewed. Deliverable: calculation. Criteria: units correct; conclusion stated.

**Exercise 4 - Quorum simulation (calculation, 45 minutes).**
With 9 members, 3 conflicted on 50 percent of requests, quorum 5, and attendance 0.75, compute the
probability of no quorum. Then compute it with one additional independent member. Deliverable:
calculations. Criteria: binomial arithmetic correct; interpretation distinguishes quorum from
throughput.

**Exercise 5 - Triage design (design, 45 minutes).**
Design a triage rule for a DAC that receives 300 requests a year, with at least three risk tiers and
a delegated authority for the lowest tier. Deliverable: one page. Criteria: tiers are defined by
risk, not by requester; assurance sample included.

**Exercise 6 - Data trust comparison (critique, 45 minutes).**
Compare an internal DAC and a data trust for a multi-site research consortium. Deliverable:
two-page memo. Criteria: covers authority, cost, independence, enforcement, and failure modes.

**Exercise 7 - Assurance plan (design, 45 minutes).**
Write a quarterly assurance plan for a model review board: what is sampled, by whom, reported to
whom, and what happens on failure. Deliverable: one page. Criteria: sample size; owner; escalation.

**Exercise 8 - Appeal process (design, 30 minutes).**
Design an appeal route for a data access decision. Deliverable: process diagram. Criteria: independent
panel; grounds; timeline; record.

## 13. Self-check answers

**Answer 1.** Six decisions: share data externally (DAC decides); approve a new purpose (privacy
officer decides, ethics recommends); deploy a model (model review board decides, clinical lead
recommends); pause a model (clinical owner decides); accept a residual security risk (security
officer decides, risk committee consulted); notify a breach (privacy officer decides, legal
consulted). Each has one decider. A matrix with "joint" decisions is a matrix that will escalate.

**Answer 2.** A charter requires purpose, scope, authority (including the right to decline),
membership, quorum with recusal rules, conflict rules, meeting cadence, decision types, a term, and
a transition clause. The authority clause is the one that distinguishes a board from an advisory
group; the transition clause is the one that protects the body when the sponsor changes.

**Answer 3.** 180 requests / 12 meetings = 15 per meeting. At 30 minutes each, demand = 450
minutes; the meeting has 120 minutes, so capacity is 4 requests and 11 cannot be reviewed at that
depth. The minutes available per request are 120 / 15 = 8. The conclusion is that the committee
must triage, extend the meeting, or reduce the volume; it cannot do 30-minute reviews for 15
requests in two hours.

**Answer 4.** With 9 members, 3 conflicted on half the requests, the eligible voters are 6 on those
requests. P(at least 5 of 6 attend) at p = 0.75 is C(6,5) x 0.75^5 x 0.25 + 0.75^6 =
6 x 0.2373 x 0.25 + 0.1780 = 0.5340, so P(no quorum) = 46.6 percent. With one additional
independent member, 7 eligible: P(5) + P(6) + P(7) = 0.3115 + 0.3115 + 0.1335 = 0.7565, so
P(no quorum) = 24.4 percent. The additional member roughly halves the failure rate but does not
eliminate it; remote voting or a lower quorum for conflicted requests may be needed.

**Answer 5.** A three-tier rule: tier 1, aggregate or de-identified data to academic recipients
under a standard contract, approved by the data steward within five days; tier 2, identifiable data
or a new recipient, reviewed by a subcommittee weekly; tier 3, commercial purpose, notes, minors,
cross-border transfer, or a novel use, reviewed by the full committee. The rule defines tiers by
risk, not by requester status, and includes a 10 percent assurance sample of tier 1.

**Answer 6.** Choose a data trust when the parties have conflicting interests and the data subjects
need independent representation that an internal committee cannot provide, and when the funding,
legal structure, and enforcement are viable. A DAC is cheaper and faster but sits inside one
organisation and may be captured by its sponsor. A trust is not automatically more legitimate; it
depends on the trustees, the beneficiaries, and the courts that enforce the duty.

**Answer 7.** A quarterly sample of 20 approvals (10 percent of the 200 delegated) checked against
the checklist gives a measurable error rate. If 2 of 20 lack the required contract, the estimated
error rate is 10 percent with a wide interval; the response is to tighten the checklist or the
training, not to assume the process works. The sample should be random, reviewed independently of
the approval, and reported to the executive who owns the risk.

**Answer 8.** The appeal should go to a panel that did not hear the original request, with grounds
limited to procedural error, new evidence, or a misapplication of the criteria. The timeline should
be stated (for example, 20 working days), the decision should be recorded, and the appellant should
receive the reasons. An appeal to the same body is not an appeal.

**Rubric for Exercise 2.** Full credit requires purpose, scope, authority including decline,
membership, quorum with recusal, conflict rules, cadence, decision types, and a term or transition
clause. A charter that says "advises on model deployment" without decision rights does not pass.

**Rubric for Exercise 5.** Full credit requires three tiers defined by risk, a delegated authority
for tier 1, and an assurance sample. A triage rule that routes by requester type does not pass.

**Rubric for Exercise 6.** Full credit compares authority, cost, independence, enforcement, and
failure modes, and recommends one structure with reasons. A memo that lists features without a
recommendation does not pass.

**Rubric for Exercise 7.** Full credit requires a sample size, a named assurance owner, a reporting
line to the executive, and an escalation path. A plan that says "review conditions annually" without
a sample does not pass.

**Rubric for Exercise 8.** Full credit requires an independent panel, defined grounds, a timeline,
a written decision with reasons, and a record. An appeal that returns to the original body does not
pass.

**Rubric for Exercise 3.** Full credit requires the per-meeting volume, the demand in minutes, the
capacity at the stated review depth, and the number of requests that cannot be reviewed. A
calculation that reports only the average time per request does not pass. The answer should also
state whether the shortfall is addressed by triage, capacity, or volume reduction.

**Rubric for Exercise 4.** Full credit requires the binomial arithmetic for both compositions and
an interpretation that separates quorum from throughput. An answer that adds members without
recomputing the probability does not pass.

## 14. Interview questions

**1. What is the difference between a decision right and an advisory role?**
Strong answer: a decision right can approve, decline, or condition; an advisory role can recommend
but not bind; confusing them produces committees that are consulted and ignored. Weak answer: "both
are governance".

**2. How do you know a committee has authority?**
Strong answer: its charter delegates the right to decline, it has declined requests, and its
conditions are verified. Weak answer: "it reports to the board".

**3. What makes a data access committee effective?**
Strong answer: a standard request, an evidence checklist, triage by risk, a decision record, and
assurance that conditions are implemented. Weak answer: "the right members".

**4. How do you handle conflicts of interest?**
Strong answer: disclosure, recusal with a record, a quorum that accounts for recusals, and an
independent chair or member. Weak answer: "members sign a form".

**5. What is a data trust?**
Strong answer: a legal structure providing independent stewardship, with trustees owing fiduciary
duties to beneficiaries; not a data-sharing agreement and not a technology. Weak answer: "a
committee for data".

**6. When would you choose a data trust over a DAC?**
Strong answer: when parties have conflicting interests and data subjects need independent
representation that an internal committee cannot provide, and the funding and legal structure are
viable. Weak answer: "for sensitive data".

**7. How do you prevent a rubber stamp?**
Strong answer: measure review time per request against volume, track conditions, triage routine
requests, and sample decisions for quality. Weak answer: "ask members to be rigorous".

**8. What is the three lines of defence?**
Strong answer: operational ownership, policy and challenge, independent assurance; the first line
owns the risk, the second sets rules, the third tests. Weak answer: "three committees".

**9. How do you govern a model after deployment?**
Strong answer: a named clinical owner, monitoring metrics with thresholds, a review board or
committee with pause authority, and a change-trigger review. Weak answer: "annual review".

**10. What would make you redesign a governance structure?**
Strong answer: decisions without a decider, quorum loss on conflicted requests, conditions not
implemented, no declines, or an assurance gap. Weak answer: "a new regulation".

**11. How do you govern a vendor's model that you cannot inspect?**
Strong answer: contractual audit rights, independent validation, model documentation, a
subprocessor list, incident notification, a termination and data-return plan, and a pilot with stop
criteria; the black box is a risk to be managed, not a reason to skip review. Weak answer: "we rely
on their certification".

**12. What is the difference between a data trust and a data institution?**
Strong answer: a data trust is a legal structure with trustees and fiduciary duties; a data
institution is the broader category of organisations that steward data for others, which may use a
trust, a company, a cooperative, or a contract. Weak answer: "they are the same".

**13. How do you measure whether governance is working?**
Strong answer: decisions with conditions, conditions verified, time to decision, appeals and
reversals, incidents traceable to a governance gap, and the number of declines; activity counts
alone are not evidence. Weak answer: "the committee meets monthly".

**14. What is the role of the board?**
Strong answer: sets risk appetite, appoints and supports the accountable executive, receives
assurance, and can remove a body that is not working; it does not make individual data-access
decisions. Weak answer: "approves everything".

**15. What is the difference between governance and management?**
Strong answer: management runs the activity and owns the risk; governance sets the decision rights,
the appetite, and the assurance that the management is working. A governance body that runs the
activity has no capacity to challenge it. Weak answer: "governance is the committee, management is
the team".

**16. How do you govern a data-sharing partnership across organisations?**
Strong answer: a written collaboration agreement with decision rights, a joint or independent
review body, a data use agreement, an audit right, and a termination and data-return plan; the
structure should name who decides when the partners disagree. Weak answer: "a memorandum of
understanding".

**17. How do you handle a decision that must be made between meetings?**
Strong answer: the chair may act under a documented urgency procedure, with the reason, the
evidence, and the decision recorded, and ratification at the next meeting; the procedure should be
limited to genuine urgency, not convenience. Weak answer: "the chair decides".

**18. What would you do if the committee had never declined a request?**
Strong answer: review the decision rights, the independence, and the evidence standard; interview
requesters and members; test a marginal request; and report the finding to the accountable
executive. A committee that has never declined may be captured or may be receiving only requests
that were pre-screened, and the difference matters.

## 15. Key terms

- **Accountability** - a named party answerable for a decision.
- **Advisory role** - recommends but does not decide.
- **Appeal** - an independent route to challenge a decision.
- **Assurance** - independent testing that controls work.
- **Beneficiary** - the party whose interests a trustee serves.
- **Charter** - the document defining a body's authority and process.
- **Condition** - a requirement attached to an approval.
- **Conflict of interest** - an interest that could bias a decision.
- **Data access committee** - body reviewing data access requests.
- **Data cooperative** - member-owned data governance structure.
- **Data institution** - organisation stewarding data for others.
- **Data steward** - role responsible for a dataset's appropriate use.
- **Data trust** - legal structure for independent data stewardship.
- **Decision record** - written record of decision, evidence, conditions, owner, date.
- **Decision right** - authority to approve, decline, or condition.
- **DPO** - data protection officer.
- **Fiduciary duty** - legal duty to act in another's interests.
- **Governance** - system of decision rights, bodies, and evidence.
- **IRB** - institutional review board.
- **Materiality** - significance of a decision or risk.
- **Model review board** - body reviewing clinical model deployment.
- **Privacy officer** - role responsible for privacy compliance.
- **Quorum** - minimum members required to decide.
- **Recusal** - withdrawal from a decision due to conflict.
- **Risk acceptance** - documented tolerance of residual risk.
- **Security officer** - role responsible for security risk.
- **Sunset clause** - provision ending an arrangement.
- **Three lines of defence** - ownership, policy, and assurance separation.
- **Triage** - routing requests by risk.
- **Trustee** - party holding a fiduciary duty.
- **Whistleblowing channel** - confidential route to report wrongdoing.
- **Reserved decision** - a decision that cannot be delegated.
- **Delegated authority** - a documented right to decide within limits.
- **Register of decisions** - aggregate record of requests and outcomes.
- **Governance scorecard** - measures of decision quality and implementation.
- **Advisory board** - a body that advises without decision rights.
- **Terms of reference** - the scope and authority of a body.
- **Accountable executive** - the leader who owns the risk and appoints the body.
- **Term limit** - a maximum period of service on a body.
- **Succession plan** - the process for replacing members and chairs.
- **Ratification** - formal approval of an urgent decision at the next meeting.
- **Quorum failure** - a meeting unable to decide for lack of members.
- **Governance gap** - a decision made without a decision right or record.
- **Assurance finding** - a tested gap between a condition and its implementation.
- **Governance review** - the periodic assessment of whether the structure still fits.
- **Decision log** - the chronological record of decisions and their owners.

## 16. Sources

All URLs accessed 26 October 2026.

**Regulation**
- Regulation (EU) 2016/679 (GDPR), Articles 24, 28, 30, 35, 37-39.
  https://eur-lex.europa.eu/eli/reg/2016/679/oj
- 45 CFR 164.308 and 164.530, HIPAA administrative safeguards and privacy responsibilities.
  https://www.ecfr.gov/current/title-45/subtitle-A/subchapter-C/part-164
- Regulation (EU) 2025/327 (European Health Data Space).
  https://eur-lex.europa.eu/eli/reg/2025/327/oj

**Frameworks and standards**
- NIST, *AI Risk Management Framework (AI RMF 1.0)*, January 2023.
  https://www.nist.gov/itl/ai-risk-management-framework
- ISO/IEC 42001:2023, *AI management systems*.
- ISO/IEC 38500:2015, *Governance of IT for the organization*.
- OECD, *AI Principles*, 2019, updated 2024. https://oecd.ai/en/ai-principles
- UK National Data Guardian, *Caldicott Principles*, revised 2020.
  https://www.gov.uk/government/publications/the-caldicott-principles

**Data trusts and institutions**
- Open Data Institute, *Defining a data trust*. https://theodi.org/insights/explainers/defining-a-data-trust/
- Open Data Institute, *Data trusts: legal and governance considerations*, 2019.
  https://theodi.org/insights/reports/data-trusts-legal-and-governance-considerations/
- Open Data Institute, *Designing trustworthy data institutions*.
  https://theodi.org/insights/reports/designing-trustworthy-data-institutions-report/

**Case evidence**
- National Audit Office, *Investigation: WannaCry cyber attack and the NHS*, 2017.
  https://www.nao.org.uk/reports/investigation-wannacry-cyber-attack-and-the-nhs/
- TechCrunch, *Google completes controversial takeover of DeepMind Health*, 19 September 2019.
  https://techcrunch.com/2019/09/19/google-completes-controversial-takeover-of-deepmind-health/
- O'Donoghue, D. *What we've learned from the Independent Reviewers panel*, September 2019.
  https://medium.com/@donalnhs/what-weve-learned-from-the-independent-reviewers-panel-e73429043da7
- NIH, *dbGaP data access requests*.
  https://www.ncbi.nlm.nih.gov/gap/

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic: [`07_ethics.md`](07_ethics.md)
- Next topic: [`09_breach-and-incident.md`](09_breach-and-incident.md)
- Related topic: [`01_privacy-principles.md`](01_privacy-principles.md)
- Survey file: [`../../course/16_privacy_security_ethics.md`](../../course/16_privacy_security_ethics.md)
- Labs: [`../../labs/`](../../labs/)
