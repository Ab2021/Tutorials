# Ethics

> **Module 16 - Privacy, Security, and Ethics**
> Topic 07 of 10
> Estimated study time: 3 hours
> Prerequisites: 06
> Version 2026.3

## 1. Why this topic exists

A model can be lawful, accurate, secure, and still wrong to deploy. It can be trained on data
collected without meaningful consent, evaluated only on the population that has access, sold to a
buyer whose workflow turns a probabilistic score into a coercive decision, and monitored by nobody
after launch. Ethics is the discipline that asks those questions before the deployment, when the
answers can still change the design.

The commercial stakes are not abstract. A vendor that deploys a product with unsafe
recommendations loses the account and the market; a health system that rolls out an alert without
staffing the response creates alert fatigue and misses the cases the tool was meant to find; a
research programme that obtains consent from the easiest-to-reach patients produces evidence that
does not generalise to the population it claims to serve. Each is an ethical failure with an
operational cost, and each is detectable before launch.

Three failure patterns recur.

**Ethics washing.** A project produces a principles statement, an ethics review, and a values slide
deck, and none of them changes a decision. The review is a gate that always opens. The signal is
that the ethics discussion happens after the architecture and the business case are fixed, and the
only permitted outcome is approval.

**Consent theatre.** A form is signed, the boxes are ticked, and nobody can explain what the
participant understood or what would happen if they said no. The form covers the sponsor's legal
risk, not the participant's decision. In pragmatic trials and learning health systems, the problem
is sharper: the intervention is embedded in care, the patient may not know a study is running, and
deferred or broad consent is used without the governance that makes it legitimate.

**Automation bias ignored.** A clinician accepts the model's recommendation because it is on the
screen, or overrides it because the workflow makes review impossible. The ethics analysis treats
the model as the decision-maker and the clinician as a rational auditor. The result is a
deployment that looks safe in a validation study and fails in a busy ward.

This topic teaches ethics as a structured assessment with artefacts: a stakeholder map, a
benefit-harm ledger, a consent and equity analysis, a human-factors review, and a decision record
with an accountable owner. It builds on the fairness measurement in Topic 06 and the governance
structures in Topic 08. The ethical question is not "is this model good" but "who benefits, who
bears the burden, who decides, and what happens when the answer is contested".

The distinction that runs through the topic is between what is legal, what is effective, and what
is right. The three can diverge. A lawful processing basis does not make a deployment just; a
statistically effective model does not make a programme fair; and a fair programme can still be
paternalistic if the people affected had no say. The assessment has to hold all three in view.

A fourth pattern is **the deferral to the clinician**. The ethics review concludes that the
clinician remains responsible, and treats that as a control. The clinician may have thirty seconds,
no explanation, and a productivity target that penalises disagreement. Responsibility without
capacity is not accountability; it is a transfer of liability. The assessment should test whether
the human in the loop has the information, time, and authority to change the outcome.

The ethics of deployment is also the ethics of not deploying. A tool that could reduce harm and is
withheld because the organisation cannot staff it, or because the evidence is inconvenient, is a
choice with consequences. The assessment should state the harms of the status quo, so that
"do nothing" is compared rather than assumed.

> **Note:** Ethics review is a structured argument, not a certificate. Its value is in the
> decisions it changes and the harms it prevents, not in the document it produces.

## 2. Learning outcomes

By the end of this topic you can:

1. **Map** stakeholders and affected communities for a biomedical AI deployment, including people
   who are not in the training data.
2. **Write** a benefit-harm ledger with magnitudes, uncertainty, and distribution across groups.
3. **Evaluate** a consent model (specific, broad, deferred, waiver) against the governance it
   requires and the population it excludes.
4. **Quantify** consent coverage and representation, and **explain** the generalisability
   consequence of differential participation.
5. **Assess** automation bias and workflow fit, and **design** a human-review process that is
   resourced to work.
6. **Identify** power asymmetries and **propose** accountability mechanisms that give affected
   people a voice.
7. **Run** an ethics review that ends in a recorded decision with conditions and an owner.
8. **Critique** a principles statement or ethics review that does not change any decision.

## 3. The core mental model

Ethics is a comparison of who gains, who loses, and who decides, made under uncertainty and
reviewed over time. The model is a ledger plus a power map plus a lifecycle.

```text
                +-----------------------------+
                |  affected population        |
                |  patients, families, staff, |
                |  communities not in the data|
                +--------------+--------------+
                               |
        benefits --------------+-------------- harms
        earlier diagnosis      |      false reassurance
        access to programme    |      surveillance
        staff time saved       |      alert fatigue
                               |
                     who decides?
                sponsor | clinician | patient | regulator
                               |
                               v
        +-------------------------------------------+
        |  deployment lifecycle                     |
        |  design -> consent -> pilot -> scale      |
        |     ^                          |          |
        |     +------ review <-----------+          |
        +-------------------------------------------+
```

The ledger has three properties. Benefits and harms are distributed, not aggregated: a programme
can produce a large net benefit and a severe harm to a small group. They are uncertain: the
validation study gives an estimate, not a guarantee. And they are weighted differently depending on
who bears them: a false positive that costs a nurse four minutes is not the same harm as a false
negative that costs a patient a delayed diagnosis.

The power map asks who can say no. A patient in a pragmatic trial may not be able to refuse without
leaving their care; a clinician may not be able to override without a productivity penalty; a
community may not be represented in the design at all. The ethical question is not only whether the
benefit exceeds the harm, but whether the people bearing the harm had a meaningful opportunity to
refuse or shape the programme.

The lifecycle matters because the ethical questions change at each stage:

| Stage | Main question | Artefact |
|---|---|---|
| Design | Is this the right problem, and who defined it? | Problem statement; stakeholder map |
| Data | Whose data, under what consent, with what exclusion? | Consent analysis; coverage table |
| Evaluation | Whose outcomes, and what harms are measured? | Benefit-harm ledger |
| Pilot | Who is included, and who can refuse? | Pilot protocol; equity plan |
| Scale | What changes when the workflow and population change? | Monitoring plan; review triggers |
| Sunset | What happens to the data and the model when the programme ends? | Decommissioning plan |

The sunset row is the one most often missing. A model that is retired leaves data, derived
datasets, vendor access, and patient expectations behind. An ethical programme plans for the end at
the beginning.

The lifecycle framing also clarifies what "review" means. A review at design asks whether the
problem is worth solving and who defined it. A review at pilot asks whether the evidence supports a
narrower deployment. A review at scale asks whether the workflow and the population still match the
evidence. The same committee can answer all three, but only if it sees different artefacts at each
stage; a committee that reviews the same slide deck at every gate is not reviewing the lifecycle.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| Ethics | The structured evaluation of what is right, fair, and respectful in a decision. | Reduced to compliance. | Assessment. |
| Morality | Shared norms about right and wrong in a community. | Used interchangeably with ethics. | Normative concept. |
| Bioethics | Ethics applied to health, research, and care. | Treated as a committee. | Field. |
| Respect for persons | Treating people as agents entitled to choose, with extra protection for those with diminished autonomy. | Reduced to a consent form. | Principle. |
| Beneficence | Acting for the benefit of others. | Assumed to follow from good intentions. | Principle. |
| Non-maleficence | Avoiding harm. | Ignored when benefits are large. | Principle. |
| Justice | Fair distribution of benefits, burdens, and access. | Reduced to equal treatment. | Principle. |
| Transparency | Disclosing what the system does, how, and with what limits. | A privacy policy. | Principle. |
| Accountability | A named party answerable for the decision and its effects. | A committee with no power. | Principle. |
| Informed consent | Voluntary agreement after understanding purpose, risks, benefits, and alternatives. | A signature. | Process. |
| Broad consent | Consent to a range of future research under governance. | Treated as unlimited. | Consent model. |
| Deferred consent | Consent obtained after an intervention, used in emergency research. | Assumed to be no consent. | Consent model. |
| Waiver of consent | Approval to proceed without consent under defined criteria. | Used for convenience. | Consent model. |
| Therapeutic misconception | Believing research is designed for the participant's individual benefit. | Ignored in recruitment. | Misunderstanding. |
| Undue inducement | An offer so attractive it compromises voluntary choice. | Confused with fair compensation. | Ethical risk. |
| Coercion | Using power or threats to remove choice. | Confused with persuasion. | Ethical risk. |
| Vulnerability | A condition that increases the risk of harm or diminishes the ability to consent. | Assigned to a category, not a context. | Assessment. |
| Equity | Fair access to benefit, accounting for barriers and power. | Confused with equality. | Principle. |
| Automation bias | Over-reliance on an automated recommendation. | Treated as a user error. | Human-factors mechanism. |
| Alert fatigue | Desensitisation from excessive alerts. | Measured only as satisfaction. | Human-factors mechanism. |
| Override rate | The proportion of recommendations a clinician rejects. | Assumed to measure model quality. | Metric. |
| Workflow fit | The degree to which a tool supports the actual care process. | Assessed in a demo. | Evaluation. |
| Human in the loop | A person reviews or authorises the model output. | Assumed to be meaningful without time or authority. | Design pattern. |
| Meaningful human control | Review that is informed, resourced, and able to change the outcome. | A click-through. | Design standard. |
| Algorithmic impact assessment | A structured assessment of a system's effects on rights and groups. | A privacy impact assessment. | Process. |
| DPIA | Data protection impact assessment for high-risk processing. | Treated as an ethics review. | Process. |
| Ethics washing | Using ethical language and process to legitimise a decision without changing it. | Mistaken for engagement. | Failure mode. |
| Stakeholder | A party affected by or able to affect the system. | Limited to the buyer. | Mapping. |
| Community advisory board | A standing group that advises on priorities and harms. | Convened once for a launch. | Governance. |
| Redress | A route to challenge a decision and obtain remedy. | An email address. | Mechanism. |
| Sunset | Planned decommissioning and data disposition. | Ignored. | Lifecycle stage. |
| Paternalism | Overriding a person's choices for their own good. | Justified by clinical urgency. | Ethical risk. |
| Distributive justice | Fair allocation of benefits and burdens across groups. | Reduced to access. | Principle. |
| Procedural justice | Fairness of the decision process, including voice and appeal. | Ignored. | Principle. |

## 5. The mechanics, step by step

The procedure below produces an ethics assessment for a biomedical AI deployment. It is designed to
be run alongside the privacy assessment (Topic 01), the fairness assessment (Topic 06), and the
threat model (Topic 04), and to end in a recorded decision. Each step names its input, output, and
the reason it exists.

### Step 1. Define the ethical question and the decision

**Input:** Project proposal, intended use, deployment plan.
**Output:** A question statement and a decision statement.
**Rationale:** "Is this model ethical" is too broad to answer. A workable question is specific:
"should this sepsis alert be deployed across all adult inpatient wards, given the current staffing
and the evidence from the pilot?" The decision statement names the options: deploy now, deploy in
selected wards, run a longer pilot, defer, or decline. If there is only one option on the table,
the review is not a decision, it is a ratification.

### Step 2. Map stakeholders, including the absent ones

**Input:** Deployment plan, clinical workflow, data sources.
**Output:** A stakeholder map with interests, power, and exposure to harm.
**Rationale:** The map should include patients (including those not in the training data), families
and carers, clinical staff, the organisation, the vendor, payers, regulators, and the communities
that the programme claims to serve. The absent ones matter most: a model trained on insured
patients and deployed to an uninsured population has a stakeholder group with no voice in the
design.

The map should record power as well as interest. A group with high exposure and low power is the
group the assessment should protect. A group with high power and low exposure is the group whose
preferences are most likely to dominate.

### Step 3. Build the benefit-harm ledger

**Input:** Evidence from validation, pilot data, clinical input, community input.
**Output:** A ledger with each benefit and harm, its magnitude, its uncertainty, and its
distribution.
**Rationale:** The ledger forces the assessment to name the harms that the business case omits:
false reassurance, alert fatigue, loss of clinical skills, surveillance, stigma, displacement of
staff time, and the opportunity cost of the programme. Each entry should have a magnitude (even if
ordinal), an uncertainty statement, and a distribution across groups.

A useful format has four columns: effect, who bears it, expected magnitude, and confidence. The
last column is what distinguishes an ethics assessment from a advocacy document; a harm with low
confidence should be monitored, not ignored.

### Step 4. Analyse the consent model and its coverage

**Input:** Consent instrument, recruitment data, population data.
**Output:** A consent analysis with coverage by group and a generalisability statement.
**Rationale:** Consent is a process, not a signature. The analysis should ask whether the
participant understood the purpose, the alternatives, the risks, and the right to withdraw; whether
the consent was voluntary given the power relationship; and whether the consent model (specific,
broad, deferred, waiver) fits the risk and the governance.

Coverage is the quantitative part. If consent is obtained from 71 percent of eligible patients but
only 53 percent of one language group, the study population is not the eligible population, and the
results may not generalise. The arithmetic in Section 6 shows how to report this.

| Consent model | When appropriate | Governance required | Main ethical risk |
|---|---|---|---|
| Specific informed consent | Most research with identifiable data | Full information; withdrawal process | Low participation; selection bias |
| Broad consent | Biobanking and future research | Governance, oversight, re-contact where feasible | Scope creep; unclear to participants |
| Deferred consent | Emergency research where pre-consent is impossible | Community consultation; prompt consent; opt-out | Participants never informed; trust harm |
| Waiver of consent | Minimal-risk research where consent is impracticable | IRB or ethics committee approval; public notice | Convenience masquerading as necessity |
| Opt-out with notice | Learning health system activities | Public communication; easy opt-out; oversight | People do not see the notice |

The table is a prompt, not a legal rule. The legal requirements differ by jurisdiction, and the
ethical question is whether the model respects the person, not only whether it is permitted.

### Step 5. Analyse equity and access

**Input:** Coverage data, access data, fairness assessment.
**Output:** An equity analysis naming who benefits, who is excluded, and why.
**Rationale:** A programme can improve outcomes overall and widen a gap. The analysis should ask
who can access the programme (referral, transport, language, insurance, digital access), who is
likely to be flagged (measurement, documentation, prior contact), and who bears the burden of a
false positive (time, cost, stigma, invasive follow-up).

The fairness metrics from Topic 06 are an input, not the answer. A model can have equal true
positive rates and still be deployed in a way that only reaches patients with a smartphone. The
equity analysis covers the whole pathway, not only the model.

### Step 6. Analyse human factors and automation bias

**Input:** Workflow observation, alert volume, staffing, override data.
**Output:** A human-factors assessment with a staffing calculation and a review design.
**Rationale:** "Human in the loop" is a claim about a person having the time, information, and
authority to change the outcome. If the alert volume exceeds the review capacity, the loop is
theatre. The assessment should compute the review hours required and compare them with the
available capacity, as in Section 6.

Automation bias is the other side. Clinicians may accept a recommendation because it is presented
with confidence, or because disagreeing requires documentation. The design should make disagreement
easy, visible, and non-punitive, and the monitoring should track override rates by group and by
ward.

### Step 7. Analyse power, voice, and redress

**Input:** Stakeholder map, governance arrangements, complaint routes.
**Output:** A power analysis and a redress plan.
**Rationale:** The people who bear the harm need a route to challenge the decision. That route can
be a complaint process, a community advisory board, a patient representative on the review
committee, or a legal mechanism. The test is whether the route can change an outcome, not whether
it exists.

Power also shapes the evidence. A vendor controls the validation data; a sponsor controls the
protocol; a health system controls the workflow. The assessment should name who can say no at each
stage, and whether the affected population has any say at all.

### Step 8. Consider alternatives, including not deploying

**Input:** Options, evidence, stakeholder input.
**Output:** A comparison of at least three options, including the status quo.
**Rationale:** An ethics assessment that considers only the proposed deployment cannot conclude
"do not deploy". The alternatives might be a simpler intervention (a checklist, a staffing change,
a rules-based alert), a narrower deployment, a research protocol instead of a product, or no
change. The comparison should state the expected benefit and harm of each, with the same
uncertainty.

The "do nothing" option is not neutral; it has its own harms. The assessment should state them, so
that the decision is a comparison rather than a default.

### Step 9. Run the review and record the decision

**Input:** All artefacts above.
**Output:** A decision record with conditions, owner, and review date.
**Rationale:** The review should include clinical, technical, legal, privacy, and community
representation. The decision should be one of a defined set, and each outcome should have
conditions attached. A review that ends with "approved" and no conditions is a ratification.

| Outcome | Meaning | Required conditions |
|---|---|---|
| Approve | Deploy as proposed | Monitoring plan; review date |
| Approve with conditions | Deploy after specific changes | Named conditions; verification before launch |
| Pilot only | Limited deployment to generate evidence | Evaluation protocol; stop criteria |
| Defer | More evidence or design work needed | Questions to answer; owner; date |
| Decline | Do not deploy | Rationale; alternatives considered |

### Step 10. Monitor, review, and plan the sunset

**Input:** Decision record, monitoring plan, decommissioning plan.
**Output:** A monitoring schedule, review triggers, and a data disposition plan.
**Rationale:** Ethics does not end at launch. The monitoring plan should track the harms in the
ledger, the equity metrics, the override rates, and the complaints. The review triggers should
include a new population, a new site, a workflow change, a safety event, and the passage of time.

The sunset plan should state what happens to the data, the model, the vendor access, and the
patient communication when the programme ends. A programme that cannot be ended cleanly is a
programme that will outlive its justification.

### Consent as a process

A consent process has four parts, and the signature is the least important. Comprehension is
whether the person can explain the purpose, the alternatives, the risks, and the right to withdraw
in their own words. Voluntariness is whether refusal is possible without losing care or standing.
Capacity is whether the person can make the decision, with support if needed. Documentation is the
record that the first three happened.

Teach-back is the simplest comprehension test: ask the participant to explain what they agreed to,
and correct the misunderstanding. It takes two minutes and catches the failures that a signature
does not. A consent form written at a twelfth-grade reading level will be understood by fewer than
half of the adults who sign it; a form at a sixth-grade level, with a conversation, performs
better.

Compensation requires its own review. Payment for time and travel is fair; payment that is large
relative to the participant's income can be an undue inducement, especially for invasive or
time-consuming research. The ethical test is whether the payment would lead someone to accept a
risk they would otherwise refuse.

### Assessing vulnerability

Vulnerability is contextual, not a property of a category. A person is vulnerable when their
ability to consent is diminished, their alternatives are constrained, or their exposure to harm is
increased. The categories that regulations name (children, prisoners, pregnant women, cognitively
impaired adults) are prompts, not the whole list. A patient who does not speak the dominant
language, a person who depends on the institution for housing, and a night-shift worker who cannot
attend follow-up are all vulnerable in context.

The assessment should name the specific vulnerability, the additional safeguard, and the residual
risk. Additional safeguards include independent consent monitors, cooling-off periods, separate
consent for sensitive sub-studies, and community review. A safeguard that consists only of a longer
form is not a safeguard.

### The ethics of the model itself

Three model properties have ethical weight beyond their statistical performance.

**Epistemic humility.** A model trained on a single site and a short window should present its
uncertainty and its scope. A confidence score that is not calibrated, or a user interface that
presents a probability as a certainty, invites over-reliance. The interface is an ethical artefact.

**Scope discipline.** A model validated for one decision (flagging for review) can drift into
another (denying care, setting a price, triaging a waiting list). The ethical assessment should
state the intended decision and the uses that are out of scope, and the governance should enforce
the boundary. Scope creep is often a product decision made without a new review.

**Failure visibility.** A model that fails silently is worse than one that fails loudly. The design
should surface uncertainty, disagreement between inputs, and out-of-distribution cases, and it
should give the clinician a route to report a problem that reaches the maintainer.

### Redress and complaint mechanisms

A complaint route has three requirements: it must be findable, it must have a response time, and it
must be able to change an outcome. A web form that routes to a shared inbox with no owner fails all
three. The design should name the person who reviews complaints, the maximum response time, the
remedy options (re-review, correction, removal from the programme, explanation), and the record
that is kept.

In a clinical deployment, the complaint may be about a decision the model influenced, and the
patient may not know the model was involved. Transparency about the use of decision support, and a
route to ask whether it was used, are part of the mechanism. The same applies to a research
participant who wants to know how their data were used.

### Review triggers

The ethics review should be re-run when a defined trigger occurs, not only on a calendar. Useful
triggers:

- A new population or site, especially one with a different language, payer mix, or access pattern.
- A new use of the model, including a new decision it informs.
- A material change in the workflow, staffing, or alert volume.
- A safety event or a complaint pattern.
- A change in the vendor, the model version, or the training data.
- A change in the legal or regulatory context.
- The passage of the review date, which should be no longer than twelve months for a live
  clinical deployment.

The trigger list should have an owner and a process for deciding whether a full review is needed.
Without that, the triggers are a list of things that might happen.

### Documentation and the review pack

The review pack should contain the question and options, the stakeholder map, the benefit-harm
ledger, the consent analysis, the equity analysis, the human-factors assessment, the power and
redress analysis, the alternatives comparison, and the decision record. It should also contain the
dissents: a review with no dissent is either uncontroversial or not a real review. The pack is the
institutional memory that survives staff turnover, and it is the evidence a regulator or a
community group will ask for.

The pack should be versioned. A decision made in March and revised in September needs both versions
and the reason for the change. An undated ethics document cannot be audited.

### Fields of the decision record

A decision record that can be audited has nine fields: the question, the options considered, the
evidence reviewed, the stakeholders consulted, the harms accepted and by whom, the conditions, the
owner, the review date, and the dissent. The last three are the ones usually missing. A record
without an owner cannot be enforced; a record without a review date cannot be revisited; a record
without dissent cannot be distinguished from a rubber stamp.

The record should be short enough to read in a meeting and specific enough to act on. One page is
usually enough. The supporting pack carries the detail.

### The pace of the review

An ethics review conducted in the last week before launch cannot change the architecture, the
consent materials, or the staffing. The timing is part of the control. A review at design can
change the problem; a review at pilot can change the scope; a review at launch can only add
conditions. Programmes that want a meaningful review schedule it before the irreversible decisions,
not after them.

### Edge cases

**Small cells.** A small community or a rare condition can be identifiable in an equity report, and
the report can stigmatise the group it is meant to protect. Aggregate, suppress small cells, and
consult the community before publishing.

**Quasi-identifiers.** A combination of geography, age, and condition can identify a person in a
case study. Case narratives used for ethics review should be de-identified or composited, and the
review materials should be handled as sensitive.

**Re-identification through linkage.** An ethics review that publishes participant stories or
subgroup data can enable linkage. The same disclosure review as Topic 02 applies to ethics
artefacts.

**Free-text PHI.** Quotes from clinical notes or complaint letters can contain identifiers and
family details. Quotes should be paraphrased or approved by the speaker.

**Metadata leakage.** Review documents, meeting minutes, and slide decks carry author names, dates,
and sometimes patient details. The document handling policy should cover metadata.

**Consent withdrawal.** A participant may withdraw after the intervention. The assessment should
state what withdrawal does to the data, the model, and the analysis, and the participant should be
told before consenting. In a pragmatic trial, withdrawal may not be possible without leaving care,
which is itself an ethical issue.

**Minors and deceased patients.** Assent and parental permission for minors, and the interests of
families in deceased patients' data, require specific arrangements. An ethics review that treats
all participants as independent adults is incomplete.

**Cross-border transfer.** A multi-country study may have different consent standards, different
community expectations, and different legal protections. The assessment should state which standard
applies where, and whether the lowest standard is acceptable.

**Community harm.** A model that performs poorly for a group can reinforce stigma even if no
individual is identified. The assessment should include group-level harms, not only individual
ones.

**Commercial use.** A research consent that covers "research" may not cover commercial product
development. The assessment should state the commercial purpose and whether the consent covers it.

## 6. Worked example

### Setting

A health system proposes to deploy an AI sepsis alert across adult inpatient wards after a
single-site pilot. The alert fires when a proprietary model crosses a threshold. The vendor reports
good discrimination; the pilot suggests a positive predictive value of about 12 percent at the
deployed threshold. The programme covers 12,400 admissions per month. The evaluation uses deferred
consent: patients are enrolled at admission and asked for consent after discharge, with an opt-out.
The ethics committee has asked for a deployment assessment.

### Step 1: the question and the options

The question is whether to deploy at the current threshold across all wards. The options are:
deploy now; deploy in two wards with additional staffing; run a longer pilot with an improved
consent process; defer; or decline. The decision statement names the five options, so the review
can choose among them rather than ratify one.

### Step 2: stakeholders

Patients (including Spanish-speaking patients, who are 18.5 percent of the eligible population),
families, nurses and physicians, the vendor, the health system, and the communities served. The
Spanish-speaking group has high exposure and relatively low power in the design; the vendor has high
power and low exposure; the nursing staff have high exposure and moderate power through their
union and clinical leadership.

### Step 3: benefit-harm ledger

| Effect | Who bears it | Magnitude per month | Confidence |
|---|---|---|---|
| Earlier recognition of sepsis | Patients with sepsis (868/month) | 268 detected at current threshold | Moderate |
| Missed sepsis | Patients with sepsis not detected (600/month) | 69% of cases missed | Moderate |
| False alerts | Nurses reviewing alerts | 1,964 false alerts/month | High |
| Review workload | Nursing staff | 156.2 hours/month | High |
| Alert fatigue | All ward staff | Rising override rate | Moderate |
| Reassurance from a negative alert | Clinicians and patients | Unquantified; risk of delayed escalation | Low |
| Equity of participation | Spanish-speaking patients | Underrepresented in the evidence base | Moderate |

The ledger makes the central trade-off visible: the alert detects 268 cases and misses 600, while
generating 1,964 false alerts and 156.2 hours of review work. The benefit is real and the burden is
concentrated on nursing staff who did not choose the threshold.

### Step 4: consent coverage and representation

The deferred-consent process reaches 8,804 of 12,400 eligible patients, an overall coverage of 71.0
percent. The coverage differs by language:

| Group | Eligible | Consented | Coverage | Eligible share | Consented share | Representation ratio |
|---|---:|---:|---:|---:|---:|---:|
| English | 9,200 | 6,940 | 75.4% | 74.2% | 78.8% | 1.062 |
| Spanish | 2,300 | 1,220 | 53.0% | 18.5% | 13.9% | 0.747 |
| Other | 900 | 644 | 71.6% | 7.3% | 7.3% | 1.008 |
| **Total** | **12,400** | **8,804** | **71.0%** | **100%** | **100%** | |

### Diagnostic 1: differential consent

The Spanish-speaking group consents at 53.0 percent against 75.4 percent for the English-speaking
group, a gap of 22.4 percentage points. The representation ratio is 0.747, which means the group
appears in the consented cohort at about three quarters of its share in the eligible population.
The arithmetic is (1,220 / 8,804) / (2,300 / 12,400) = 0.1386 / 0.1855 = 0.747.

The consequence is generalisability. If the intervention's effect differs by language, access, or
the barriers that also drive consent, the trial's estimate is weighted toward the group that
consents more easily. The ethics issue is not only that participation is unequal; it is that the
evidence used to justify deployment to everyone is derived disproportionately from one group.

The likely mechanisms are practical: consent materials in English, interpreter availability,
literacy, trust, and the timing of the approach. The assessment should name them and propose
remedies, not merely report the gap.

### Diagnostic 2: alert burden exceeds capacity

At an 18 percent alert rate, the programme generates 2,232 alerts per month. At 4.2 minutes of
review per alert, the review workload is 2,232 x 4.2 = 9,374 minutes, or 156.2 hours per month.

The ward has three nurses assigned to alert review, each with 40 hours per month available for this
task: 120 hours. The workload exceeds capacity by 36.2 hours, or 30.2 percent. Utilization is 130.2
percent, which is not a sustainable operating point. In practice, the excess is absorbed by
skipping reviews, delaying them, or reviewing only the highest-scoring alerts, all of which change
the effective sensitivity of the programme.

The second diagnostic is independent of the first. Even if consent were perfect, the staffing
would not support the alert volume. Even if staffing were doubled, the consent gap would still
limit the evidence base.

### The human-factors arithmetic

The 1,964 false alerts per month are not neutral. Each consumes 4.2 minutes of a nurse's attention
and contributes to desensitisation. The override rate is the observable signal: if clinicians
override 40 percent of alerts, the effective alert rate is 1,339 per month, and the effective
sensitivity is lower than the validation study suggests. The assessment should track override rate
by ward and by group, because a uniform model can produce non-uniform override behaviour.

### Vary one assumption: raise the threshold

The simplest way to fit capacity is to raise the threshold, reducing the alert rate from 18 percent
to 12 percent. The arithmetic changes:

- Alerts: 12,400 x 0.12 = 1,488 per month.
- Review workload: 1,488 x 4.2 = 6,249.6 minutes = 104.2 hours, which fits within 120 hours with
  15.8 hours of slack.
- True positives: 1,488 x 0.12 = 178.6, so about 179 detected.
- Sensitivity: 179 / 868 = 20.6 percent, down from 30.9 percent.
- Missed cases: 868 - 179 = 689, up from 600.

The conclusion changes from "the programme cannot be staffed at the current threshold" to "the
programme can be staffed at a threshold that misses 89 more cases per month". That is an ethical
trade-off, not a technical one: 89 additional missed cases against 744 fewer false alerts and a
workload that fits. The decision belongs to the clinical leadership with the ethics committee, and
it should be recorded with the numbers.

### The decision

The committee records "pilot only" with four conditions:

1. **Consent remedy.** Interpreter-supported consent, translated materials, and a community review
   of the approach before expansion. The target is a Spanish coverage rate within 10 percentage
   points of the English rate.
2. **Staffing remedy.** Either fund 5 nurses (200 hours, 78.1 percent utilization) or adopt the
   12 percent threshold with an explicit recorded acceptance of 689 missed cases per month.
3. **Override monitoring.** Track override rate by ward and language group monthly; a rate above 40
   percent triggers review.
4. **Sunset plan.** State what happens to the deferred-consent data and the model if the pilot does
   not proceed to scale.

The decision is a comparison of harms, not a claim that the programme is safe. That is what an
ethics review can honestly produce.

### The alternatives, with numbers

The ethics review compared three options, not one:

| Option | Detected cases/month | Missed cases/month | False alerts/month | Review hours | Staffed |
|---|---:|---:|---:|---:|---|
| A: deploy at 18% alert rate | 268 | 600 | 1,964 | 156.2 | No (130.2% of capacity) |
| B: deploy at 12% alert rate | 179 | 689 | 1,297 | 104.2 | Yes (86.8% of capacity) |
| C: no alert, usual care | 0 by the model | 868 by the model | 0 | 0 | Yes |

Option C is not "no harm"; it is the status quo in which 868 cases per month are identified by
clinicians without the model, with unknown sensitivity. The table makes the comparison explicit and
prevents the review from treating deployment as the only option.

The committee's decision was "pilot only" with conditions, which is option D: a limited deployment
that generates evidence on the consent remedy and the staffing model before a system-wide decision.
The pilot has a stop criterion: if the Spanish coverage gap does not narrow to 10 percentage points
within six months, the pilot does not expand.

### What the ledger omits

The ledger does not quantify the value of a detected case, the cost of a missed case, or the
disutility of an alert. Those are judgements, and they differ by stakeholder. A nurse who reviews
1,964 false alerts may weight the burden differently from a patient who survives sepsis because of
an early alert. The assessment should not pretend to a common currency; it should make the
distribution visible and let the accountable decision-maker weigh it.

The ledger also omits the opportunity cost: the 156.2 hours of nursing time could be spent on
something else. That omission is common in business cases, and it biases the analysis toward
deployment. The review should name at least one alternative use of the same resources.

The consent arithmetic is descriptive, not causal. The coverage gap by language is measured, but
the reason for it is inferred; the assessment should test the mechanism (materials, interpreters,
timing, trust) before choosing a remedy. The workload arithmetic assumes a constant 4.2 minutes per
alert and 120 hours of capacity; both should be measured in the pilot, because review time rises
with alert complexity and capacity falls with competing duties.

The code is deliberately small enough to audit by hand. A reader can change the alert rate, the
review time, or the capacity and see the conclusion move; that is the point of putting the
arithmetic in a script rather than in a slide. The same script can be extended to include the
override rate, the cost per review hour, and the missed-case cost, but each extension adds an
assumption that should be stated.

Two cautions. First, the coverage calculation treats the consented cohort as fixed; a remedy that
increases Spanish participation also increases the denominator and changes every share. Second, the
sensitivity figure is derived from the PPV and the alert rate, not measured directly, so it inherits
the uncertainty of both. A production assessment should use the measured confusion matrix from the
pilot.

The script does not decide whether 689 missed cases are acceptable, and it should not. It reports
the exchange rate so that the clinical owner and the community can decide with the numbers in view.
A script that output a recommendation would hide the judgement inside a parameter, which is the
opposite of what an ethics assessment is for.

The code also assumes that the alert rate and the PPV are independent. In practice, raising the
threshold usually raises PPV and lowers sensitivity, so the 12 percent threshold scenario should
use a measured PPV at that threshold, not the PPV from the lower threshold. The exercise is a
planning estimate, and the pilot should replace it with measured values before the scale decision.

The output is also a communication device. A clinical leader who will not read a regression table
will read "156 hours of review against 120 hours of capacity". The script turns an ethical argument
into an operational fact, which is what makes it actionable.

A reader can check the sensitivity arithmetic by hand: 268 / 868 = 0.309, and 179 / 868 = 0.206.
The review time is alerts x minutes / 60, so 2,232 x 4.2 / 60 = 156.24. These are the two numbers
that drive the decision, and they are simple enough that a committee can verify them in the meeting
rather than trusting the analyst.

## 7. Code, query, or calculation

The script below computes the consent coverage, representation ratios, alert burden, and the
threshold scenario from Section 6. It uses only the standard library. All counts are synthetic and
chosen to match the case.

```python
"""Consent coverage, representation, and alert-burden arithmetic."""


def coverage(consented, eligible):
    return consented / eligible


def representation_ratio(group_consented, total_consented,
                         group_eligible, total_eligible):
    return (group_consented / total_consented) / (group_eligible / total_eligible)


def alert_burden(admissions, alert_rate, review_minutes, capacity_hours):
    alerts = admissions * alert_rate
    hours = alerts * review_minutes / 60
    return alerts, hours, hours - capacity_hours, hours / capacity_hours


if __name__ == "__main__":
    eligible, consented = 12_400, 8_804
    print(f"overall consent coverage: {consented}/{eligible} = "
          f"{coverage(consented, eligible):.1%}")

    groups = {"English": (9_200, 6_940), "Spanish": (2_300, 1_220),
              "Other": (900, 644)}
    for name, (e, c) in groups.items():
        share_e = e / eligible
        share_c = c / consented
        print(f"{name}: coverage={coverage(c, e):.1%}, "
              f"eligible share={share_e:.1%}, consented share={share_c:.1%}, "
              f"representation ratio="
              f"{representation_ratio(c, consented, e, eligible):.3f}")

    admissions = 12_400
    cases = admissions * 0.07
    alerts, hours, over, util = alert_burden(admissions, 0.18, 4.2, 120)
    tp = alerts * 0.12
    print(f"sepsis cases={cases:.0f}, alerts={alerts:.0f}, "
          f"true positives={tp:.0f}, PPV={tp / alerts:.1%}, "
          f"sensitivity={tp / cases:.1%}")
    print(f"review hours={hours:.1f}, capacity=120.0, over={over:.1f}, "
          f"utilization={util:.1%}")

    alerts2, hours2, over2, util2 = alert_burden(admissions, 0.12, 4.2, 120)
    tp2 = alerts2 * 0.12
    print(f"threshold at 12% alert rate: alerts={alerts2:.0f}, "
          f"hours={hours2:.1f}, over={over2:.1f}, "
          f"sensitivity={tp2 / cases:.1%}, missed={cases - tp2:.0f}")
```

Expected output, verified on Python 3.11:

```text
overall consent coverage: 8804/12400 = 71.0%
English: coverage=75.4%, eligible share=74.2%, consented share=78.8%, representation ratio=1.062
Spanish: coverage=53.0%, eligible share=18.5%, consented share=13.9%, representation ratio=0.747
Other: coverage=71.6%, eligible share=7.3%, consented share=7.3%, representation ratio=1.008
sepsis cases=868, alerts=2232, true positives=268, PPV=12.0%, sensitivity=30.9%
review hours=156.2, capacity=120.0, over=36.2, utilization=130.2%
threshold at 12% alert rate: alerts=1488, hours=104.2, over=-15.8, sensitivity=20.6%, missed=689
```

Three readings of the output. The overall coverage of 71.0 percent looks acceptable until it is
broken down: the Spanish group is at 53.0 percent and its representation ratio is 0.747. The
workload is 130.2 percent of capacity, which is not a staffing preference but an arithmetic
failure. The threshold scenario fits capacity and costs 89 additional missed cases, which is the
trade-off the committee must decide.

The script does not model uncertainty in the alert rate or the PPV. A production assessment should
put intervals on both, because a PPV of 12 percent from a single-site pilot can plausibly be 8 to
17 percent, and the staffing conclusion changes across that range.


### Adapting the calculation

To use the script in a real review, replace the synthetic counts with the pilot data and add four
things. First, a confidence interval on the PPV and the alert rate, because both are estimated from
a finite sample. Second, a breakdown of the alert burden by ward, because capacity is local even
when the total looks adequate. Third, a breakdown of consent coverage by language, insurance, and
age, because a single overall rate hides the groups that matter. Fourth, a cost column, so that the
staffing condition can be compared with the missed-case harm in the same meeting.

The calculation is not a substitute for the review. It is the evidence that makes the review's
trade-off concrete enough to decide.

The script uses only the standard library so that it can run in any review environment, including
one where installing packages is slow or prohibited. The arithmetic is the point; a spreadsheet
would work equally well, and a committee is more likely to check a formula it can see than a
dashboard it cannot.

The calculation should be rerun whenever the alert rate, the review time, or the capacity changes,
because the conclusion is sensitive to all three. A programme that fits capacity at 4.2 minutes per
alert may not fit at 6 minutes, and a programme that fits at three nurses may not fit at two.

The script is a planning tool and a communication tool, not an oracle. Its value is that it makes
the exchange rate visible; the judgement about whether the exchange is acceptable belongs to the
review, not to the code.

## 8. Data, standards, and regulatory context

Ethics is the one area of this module where the binding rules are thinner than the guidance. The
law sets minimum protections for research participants and data subjects; professional and
international guidance sets expectations for AI; and the ethical analysis must go beyond both.

### Research ethics: the Belmont Report and the Common Rule

The Belmont Report (1979) states three principles for research with human subjects: respect for
persons, beneficence, and justice. It is the intellectual basis for the US regulatory framework:

- Respect for persons requires informed consent and additional protection for those with diminished
  autonomy.
- Beneficence requires maximising benefits and minimising harms, which the report frames as an
  obligation to assess risks and benefits systematically.
- Justice requires fair distribution of the burdens and benefits of research, including the
  selection of subjects.

The Common Rule (45 CFR Part 46) implements these principles for federally funded research. It
requires IRB review, informed consent (with limited exceptions), additional protections for
vulnerable populations (pregnant women, prisoners, children), and, since the 2018 revision, a
framework for broad consent. The FDA has parallel regulations at 21 CFR Parts 50 and 56 for
clinical investigations.

The Declaration of Helsinki (World Medical Association, 2013) is a professional code rather than
law in most countries. It is the reference for physician-researchers and is incorporated into many
national regulations. The CIOMS International Ethical Guidelines (2016) provide detailed guidance
for epidemiological studies and health-systems research, including community engagement and the
use of data in low-resource settings.

### Data protection and AI regulation

- The GDPR requires a lawful basis, transparency, and a data protection impact assessment for
  high-risk processing. Article 22 restricts solely automated decisions with legal or similarly
  significant effects. Consent must be freely given, specific, informed, and withdrawable.
- The EU AI Act (Regulation (EU) 2024/1689) requires a fundamental rights impact assessment for
  certain high-risk systems deployed by public bodies or private entities providing public
  services (Article 27), and requires human oversight, transparency, and post-market monitoring for
  high-risk systems. It applies in stages from 2025.
- The European Health Data Space Regulation (EU) 2025/327 adds secondary-use permits and secure
  processing environments from 2029. It is a data-governance instrument, not an ethics framework,
  but it constrains what an ethics review can approve.

### International guidance on AI ethics

- The UNESCO Recommendation on the Ethics of Artificial Intelligence (2021) sets out values
  including proportionality, human oversight, fairness, sustainability, and privacy, and calls for
  ethical impact assessments.
- The WHO guidance *Ethics and governance of artificial intelligence for health* (2021) sets out
  six principles: protect autonomy; promote human well-being, safety, and the public interest;
  ensure transparency, explainability, and intelligibility; foster responsibility and
  accountability; ensure inclusiveness and equity; and promote responsive and sustainable AI.
- The OECD AI Principles (2019, updated 2024) include inclusive growth, human-centred values,
  transparency, robustness, and accountability.

None of these instruments creates a legal obligation by itself. They are the vocabulary that
ethics committees, procurement reviewers, and regulators use, and a programme that ignores them
will find its assessment hard to defend.

The WHO principles add a public-health dimension that the Belmont framework does not: promoting
human well-being and the public interest, ensuring inclusiveness and equity, and promoting
responsive and sustainable AI. A deployment that improves outcomes for the insured population of a
high-income country and is not sustainable in the settings with the highest burden may satisfy
Belmont and fail the WHO test. The assessment should state which framework it is using and why.

**A note on language.** Ethics documents often use abstractions ("dignity", "trust", "fairness")
that cannot be acted on. The assessment should translate each abstraction into an observable: a
consent comprehension score, an override rate, a coverage ratio, a review-hour calculation, a
complaint response time. The abstraction is the reason; the observable is the control. A review that
contains only abstractions cannot be verified or improved.

> **Note:** Regulatory content in this file is educational and is not legal advice. Ethical and
> legal obligations differ, and a lawful deployment can still be unethical.

The classification to keep straight: the Common Rule and GDPR are binding; the Belmont Report and
the Declaration of Helsinki are foundational guidance with indirect legal force; the UNESCO and WHO
documents are international recommendations; and an ethics committee's approval is an institutional
decision, not a legal safe harbour.

## 9. Industry practice and case evidence

### IBM Watson for Oncology, 2018

A STAT investigation published in July 2018 reported internal IBM documents describing "multiple
examples of unsafe and incorrect treatment recommendations" from Watson for Oncology. The
documents described training on a small number of hypothetical cancer cases, for example 635 lung
cancer cases and 106 ovarian cancer cases, with recommendations reflecting the preferences of a few
specialists rather than guidelines. The most cited example was a recommendation of combination
chemotherapy plus bevacizumab for a patient with severe bleeding, despite a boxed warning against
the drug in that situation. IBM said the product had been updated and that patient safety was
paramount; the reporting did not establish that patients were harmed.

The ethical lessons are about evidence and transparency. The training data were not representative
of the clinical population, the validation was not independent, and the marketing claims outran the
evidence. A deployment review that asked "what is the evidence, on whom, and with what
limitations" would have surfaced the gap before a hospital bought the product.

### Epic Sepsis Model, 2021

Wong and colleagues externally validated a widely implemented proprietary sepsis prediction model
at Michigan Medicine, using 27,697 patients and 38,455 hospitalizations with a sepsis incidence of
7 percent. The hospitalization-level area under the receiver operating characteristic curve was
0.63, against the vendor's reported 0.76 to 0.83. At the deployed threshold, sensitivity was 33
percent, specificity 83 percent, positive predictive value 12 percent, and negative predictive value
95 percent. Eighteen percent of hospitalizations crossed the threshold, the model missed 67 percent
of sepsis cases (1,709 of 2,552), and the number needed to evaluate was 8 at the hospitalization
level. The authors reported poor calibration at all time horizons.

The ethical issues are alert burden, false reassurance, and the gap between vendor claims and
independent evidence. A 12 percent PPV means that most alerts are false, and a 33 percent
sensitivity means most cases are missed. A programme that deploys such a model without staffing the
review and without telling clinicians the operating characteristics is asking them to trust a tool
that is wrong more often than it is right. The case is also a lesson in procurement: the buyer did
not have the independent validation until after widespread adoption.

### Obermeyer et al., 2019

The cost-proxy case, discussed in Topic 06, is also an ethics case. The algorithm was well
calibrated on cost and biased on need, and correcting the label would have raised the share of
Black patients identified for additional help from 17.7 percent to 46.5 percent. The ethical
failure was not a biased feature; it was a choice to measure spending rather than illness, made
without examining the distributive consequence.

### DeepMind and the Royal Free, 2017

The UK Information Commissioner's Office found that the Royal Free NHS Foundation Trust processed
approximately 1.6 million patient records without adequately informing patients and without a valid
condition for processing. The ethics lesson is about transparency and the reasonable expectations
of patients: a clinically valuable project can still fail the respect-for-persons test if the
people whose data are used do not know.

### Ethics washing

Wagner (2018) described "ethics washing" as the use of ethical language, principles, and review
processes to legitimise a technology without changing its governance. The pattern is recognizable:
a principles document with no enforcement, an ethics board with no power, a review that always
approves. The antidote is procedural: a decision record that names the conditions, the owner, and
the review date, and a committee that has declined at least one proposal.

**What an ethics committee should ask about a model.** The evidence (independent validation, not
vendor-reported), the population (who was in the validation, who is in the deployment), the
decision (what changes for the patient, and who can refuse), the workload (who reviews the output,
with what time), the equity (who benefits and who bears the burden), the redress (how a patient
challenges a decision), and the sunset (what happens when the programme ends). A committee that
asks these seven questions will surface most of the ethical issues in a clinical AI deployment,
regardless of the model type.

**Post-market ethics.** The ethical obligations continue after launch: monitoring harms, reporting
safety events, updating the evidence, and withdrawing the product if the evidence changes. A vendor
that treats the pre-market review as the end of its ethical responsibility has confused approval
with stewardship.

**The ethics of evidence.** A deployment decision is only as good as the evidence behind it, and
the evidence is often produced by the party that benefits from approval. Independent validation,
pre-registration of the evaluation, and access to the underlying data are ethical requirements as
well as methodological ones. A vendor that controls the validation and refuses to share it is
asking the buyer to trust a claim, not to evaluate one.

**Deployment is not the only ethical choice.** A programme that declines to deploy can still
contribute: it can publish the validation, share the data with researchers under governance, or
invest in the workflow problem the model was meant to solve. The ethics review should record what
the organisation did instead, so that the decision is not experienced as a dead end by the team.

**The review is not the end.** Approval is a decision to proceed under conditions, not a judgment
that the system is safe. The conditions are the control, and the review date is the reminder. An
organisation that treats approval as the finish line has misunderstood what the review was for.

> **Note:** The case facts above are drawn from the cited public reporting and papers. Statements
> about what an organisation should have done differently are inference, labelled as such.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Ethics washing | Review always approves; no conditions | Process without power | Harm legitimised | Decision record; power to decline |
| Consent theatre | Signature rate high; comprehension unmeasured | Form protects sponsor, not participant | Invalid consent; trust harm | Teach-back; comprehension checks |
| Differential consent | Coverage varies by language or group | Access and trust barriers | Evidence does not generalise | Translated materials; interpreters |
| Automation bias | Override rate very low | Clinicians defer to the score | Errors propagate | Training; easy override; audit |
| Alert fatigue | Override rate rises over time | Volume exceeds attention | Missed alerts | Staffing; threshold; triage |
| Human-in-the-loop theatre | Review time under 30 seconds | No time or authority to disagree | Nominal oversight | Resource the review; measure |
| Stakeholder absence | No patient or community voice | Design reflects the buyer | Harms to unrepresented groups | Advisory board; compensation |
| Benefit-harm ledger omits harms | Only benefits listed | Business case framing | Hidden burden | Independent review |
| No redress | Complaint route has no power | Procedural justice absent | Harms unremedied | Appeal path; response times |
| Sunset omitted | No decommissioning plan | Programme outlives justification | Data and access persist | Plan at design |
| Commercial use outside consent | "Research" consent used for product | Scope creep | Consent invalid | Purpose-specific consent |
| Undue inducement | Payment disproportionate to risk | Voluntariness compromised | Exploitation | Ethics review of compensation |
| Vulnerable group not protected | Prisoners, children, cognitively impaired enrolled without safeguards | Category assumed, context ignored | Harm to those with least power | Additional protections |
| One-time review | No monitoring after launch | Population and workflow drift | Harms emerge unnoticed | Review triggers; monitoring |

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test |
|---|---|---|---|
| The stakeholder map is complete | Harms to absent groups are missed | A group has no voice | Community review |
| The benefit-harm ledger is balanced | Decisions favour the sponsor | Only benefits are listed | Independent review |
| Consent is informed and voluntary | Respect for persons | Comprehension or power gap | Teach-back; interviews |
| Coverage is representative | Evidence generalises | Differential participation | Coverage by group |
| Review capacity matches alert volume | Human oversight is real | Staffing shortfall | Workload calculation |
| Override is possible and safe | Clinicians can disagree | Productivity penalty | Workflow observation |
| The vendor evidence is independent | Claims match performance | Vendor-controlled validation | External validation |
| The decision-maker is accountable | Redress is possible | No named owner | Decision record |
| Monitoring catches drift | Harms are detected | No equity metrics | Dashboard review |
| The sunset is planned | Data do not persist | No decommissioning plan | Plan review |
| Legal and ethical requirements align | One assessment suffices | Law permits what ethics rejects | Separate reviews |

Two limits are structural. First, ethics is not reducible to a metric; the ledger informs a
judgement, it does not replace it. Second, the assessment is conducted by people with interests.
The committee, the sponsor, and the evaluator all have stakes, and the process should include at
least one participant or community voice with the standing to disagree.

## 12. Exercises

The exercises are designed to produce the artefacts of a real review, not essays about ethics.
Exercises 1 to 4 build the evidence; 5 and 6 build the human-factors case; 7 runs the decision; 8
plans the end. If you do only one, do Exercise 7: a mock review with a community voice exposes the
gaps that a solo analysis misses.

The review pack should be written for three audiences at once: the clinician who has to use the
system, the executive who has to fund the conditions, and the community member who has to trust it.
That constraint improves the analysis, because a sentence that cannot be understood by all three is
usually a sentence that hides an assumption.

**Exercise 1 - Stakeholder map (application, 40 minutes).**
Map the stakeholders for a remote patient monitoring programme, including patients without
smartphones. Deliverable: diagram with interest, power, and exposure. Criteria: at least eight
groups; absent groups identified; power asymmetry named.

**Exercise 2 - Benefit-harm ledger (application, 45 minutes).**
Build a ledger for a fall-risk alert in a nursing home. Include at least four harms and four
benefits with magnitude and confidence. Deliverable: table. Criteria: harms include staff burden
and resident autonomy; each row has confidence; distribution across groups stated.

**Exercise 3 - Consent model choice (design, 40 minutes).**
For an emergency-department study of a septic-shock protocol, choose a consent model and justify
it. Deliverable: one-page memo. Criteria: names the model; addresses community consultation;
states the opt-out and the follow-up consent.

**Exercise 4 - Coverage arithmetic (calculation, 40 minutes).**
Using the Section 6 data, compute what the Spanish coverage rate would need to be to reach a
representation ratio of 0.90, holding the other groups constant. Deliverable: calculation. Criteria:
arithmetic correct; interpretation of feasibility.

**Exercise 5 - Workload calculation (calculation, 30 minutes).**
An alert fires 2,500 times per month and review takes 5 minutes. How many full-time reviewers are
needed at 70 percent utilization? Deliverable: calculation. Criteria: hours and FTEs correct;
utilization assumption stated.

**Exercise 6 - Automation bias review (critique, 45 minutes).**
Observe or simulate a clinical workflow with an alert and identify three automation-bias risks and
three design changes. Deliverable: one page. Criteria: risks are specific to the workflow; changes
are feasible; override is addressed.

**Exercise 7 - Ethics review simulation (design, 90 minutes).**
Run a mock review of the Section 6 deployment with roles for clinical, community, vendor, and
privacy representatives. Deliverable: decision record with conditions and owner. Criteria: at least
one condition accepted; decision is one of the five outcomes; dissent recorded.

**Exercise 8 - Sunset plan (design, 40 minutes).**
Write the decommissioning plan for a deployed risk model: data disposition, model retirement,
vendor access, patient communication. Deliverable: one page. Criteria: covers data, model, access,
and communication; names an owner; states a timeline.

## 13. Self-check answers

**Answer 1.** A remote monitoring programme's stakeholders include patients (including those
without smartphones or broadband), carers, nurses, physicians, the vendor, the health system,
payers, and the community. Patients without smartphones have high exposure and low power: the
programme may not reach them, and their outcomes may worsen relative to those who are reached. The
map should name that asymmetry and propose a mitigation (a telephone pathway, a loaned device, or a
community worker).

**Answer 2.** Respect for persons requires informed consent and additional protection for those with
diminished autonomy; it implies a consent process, not a signature. Beneficence requires a
systematic assessment of risks and benefits, which implies a ledger with magnitudes and
uncertainty. Justice requires fair distribution of the burdens and benefits of research, which
implies attention to subject selection and to who is excluded, and to whether the group bearing the
burden had a voice in the design. The three principles can conflict, and the conflict is resolved by
judgement and documented reasoning, not by a formula.

**Answer 3.** For an emergency study of septic shock, deferred consent is usually appropriate
because the intervention must start before a surrogate can be found. The memo should require
community consultation before the study, a short consent conversation at the earliest feasible
time, an opt-out, and a public disclosure of the study. A waiver without consultation, or a
deferred consent without a plan to inform survivors, would not meet the respect-for-persons
standard.

**Answer 4.** The consented total is 8,804. To give Spanish a representation ratio of 0.90, its
consented share must be 0.90 x 0.1855 = 0.1670, or 1,470 of 8,804 (if the total stays fixed).
Coverage would need to be 1,470 / 2,300 = 63.9 percent, up from 53.0 percent. If the additional
consents increase the total, the arithmetic changes; the exercise should state which assumption is
used. The interpretation: a 10.9 percentage point improvement in coverage is plausible with
translated materials and interpreters, which is why the condition in Section 6 is not unrealistic.

**Answer 5.** 2,500 alerts x 5 minutes = 12,500 minutes = 208.3 hours per month. At 70 percent
utilization, the staffed capacity must be 208.3 / 0.70 = 297.6 hours per month. A full-time
equivalent at 40 hours per week provides about 173.3 hours per month (40 x 52 / 12), so the
requirement is 297.6 / 173.3 = 1.72 full-time equivalents. The units matter: dividing monthly hours
by weekly hours is the common error, and it understates the requirement by a factor of about four.

**Answer 6.** Automation bias risks include accepting a high score without checking the inputs,
failing to escalate when the score is low, and overriding more often for patients who are already
complex. Design changes include showing the contributing factors, requiring a reason for override,
and reviewing a sample of low-score cases for delayed escalation. The test is behavioural: measure
override rate, time to review, and escalation rate by group, and compare them with a period before
the alert.

**Answer 7.** A correct answer names the three Belmont principles and the four consent parts:
comprehension, voluntariness, capacity, and documentation. A consent process that has a signature
but no comprehension check fails the respect-for-persons principle even if the form is legally
sufficient.

**Answer 8.** A mock review should produce a decision from the defined set, at least one condition
with an owner and a verification, and a record of dissent. A review that approves without conditions
fails the exercise regardless of the quality of the discussion, because the artefact is what the
institution can enforce.

**Rubric for Exercise 2.** Full credit requires four benefits and four harms, magnitude and
confidence for each, and a statement of who bears each. A ledger that lists only clinical benefits
and technical risks does not pass.

**Rubric for Exercise 6.** Full credit requires three specific automation-bias risks (for example,
acceptance of a high score without review, delayed escalation after a low score, and unequal
override by group) and three design changes (for example, explanation, forced review time, and
override reason capture). Generic statements about "user training" do not pass.

**Rubric for Exercise 7.** Full credit requires a decision from the defined set, at least one
condition with an owner and a verification, and a record of dissent. A decision record that says
"approved" with no conditions does not pass.

## 14. Interview questions

**1. What is the difference between law and ethics?**
Strong answer: law sets minimum enforceable requirements; ethics asks what is right beyond them; a
lawful deployment can be unethical (for example, a lawful but coercive consent process). Weak
answer: "ethics is stricter law".

**2. What are the three Belmont principles?**
Strong answer: respect for persons, beneficence, and justice, with the implications for consent,
risk-benefit assessment, and subject selection. Weak answer: names them without implications.

**3. When is deferred consent acceptable?**
Strong answer: when the intervention must start before consent is feasible, the risk is justified,
community consultation occurs, and consent or notification follows as soon as possible. Weak
answer: "in emergencies".

**4. What is ethics washing?**
Strong answer: using ethical language and process to legitimise a decision without changing it;
the test is whether the review can and does say no. Weak answer: "bad ethics".

**5. How do you assess automation bias?**
Strong answer: measure override rates, review time, and outcomes by group; observe the workflow;
test whether clinicians have the time and authority to disagree. Weak answer: "we train users".

**6. What makes human oversight meaningful?**
Strong answer: information, time, authority, and a safe route to disagree; a click-through is not
oversight. Weak answer: "a human reviews the output".

**7. How do you handle differential consent?**
Strong answer: measure coverage by group, identify barriers, translate materials, use interpreters,
engage the community, and report the generalisability limit. Weak answer: "we offered consent to
everyone".

**8. What is in a benefit-harm ledger?**
Strong answer: benefits and harms with magnitude, uncertainty, distribution, and who bears them,
including burdens on staff and opportunity costs. Weak answer: a list of pros and cons.

**9. When would you decline to deploy a model?**
Strong answer: when the harm is severe and unmitigated, consent is not meaningful, the evidence is
not independent, the workflow cannot support safe use, or the affected community has no voice and
the burden falls on them. Weak answer: "if it is inaccurate".

**10. What is a sunset plan and why does it matter?**
Strong answer: the plan for data disposition, model retirement, vendor access, and patient
communication at the end of the programme; without it, data and access outlive the justification.
Weak answer: "turning off the server".

**11. How do you give a community a voice in a deployment?**
Strong answer: a standing advisory board with compensation, review of the harm statement and
consent materials, a route into the decision, and feedback on what changed; not a one-off
consultation after the design is fixed. Weak answer: "we held a focus group".

**12. What is procedural justice and why does it matter?**
Strong answer: fairness of the decision process, including voice, consistency, and appeal; people
accept decisions more readily when the process is fair, and a fair process can correct a substantively
unfair outcome. Weak answer: "following the rules".

**13. When is a waiver of consent ethical?**
Strong answer: when the research is minimal risk, consent is impracticable, the rights and welfare
of participants are not adversely affected, and oversight approves; it is not a convenience
mechanism. Weak answer: "when consent is hard".

**14. What is the difference between a DPIA and an ethics review?**
Strong answer: a DPIA assesses data protection risks and legal compliance; an ethics review assesses
benefits, harms, justice, and voice, including harms that are not data protection harms. A
deployment may need both, and one does not substitute for the other. Weak answer: "they are the
same process".

**15. How do you handle a participant who wants to withdraw after the model is trained?**
Strong answer: stop future use, remove the record from datasets where feasible, document what the
model retains, explain the limit to the participant, and have a retraining policy; the consent
process should have disclosed this before collection. Weak answer: "we delete the record".

**16. What would you do if the clinical team wanted to deploy a model the evidence did not
support?**
Strong answer: document the evidence gap, propose a pilot with stop criteria, escalate to the
medical safety or quality committee, and refuse to sign the validation claim; the ethical
obligation is to the patient, not to the project timeline. Weak answer: "defer to the clinical
team".

**17. How do you balance a large aggregate benefit against a small severe harm?**
Strong answer: name the group bearing the harm, quantify it, consider whether the harm is
compensable or avoidable, and give the group a voice; a net-benefit calculation does not settle a
rights-based harm. Weak answer: "the numbers favour deployment".

**Rubric for Exercise 8.** Full credit requires data disposition, model retirement, vendor access,
patient communication, an owner, and a timeline. A plan that says "delete the data" without
addressing derived datasets or vendor copies does not pass.

**18. What is the role of a community advisory board?**
Strong answer: standing, compensated, with review of the harm statement, consent materials, and
deployment conditions, and with feedback on what changed; it is not a one-off focus group. Weak
answer: "to represent patients".

## 15. Key terms

- **Accountability** - a named party answerable for the decision.
- **Alert fatigue** - desensitisation from excessive alerts.
- **Algorithmic impact assessment** - structured assessment of effects on rights and groups.
- **Automation bias** - over-reliance on an automated recommendation.
- **Belmont Report** - foundational US research ethics statement.
- **Beneficence** - acting for the benefit of others.
- **Benefit-harm ledger** - structured comparison of effects and who bears them.
- **Broad consent** - consent to future research under governance.
- **CIOMS guidelines** - international ethics guidance for health research.
- **Coercion** - removing choice through power or threats.
- **Community advisory board** - standing community input body.
- **Deferred consent** - consent obtained after an intervention.
- **Distributive justice** - fair allocation of benefits and burdens.
- **DPIA** - data protection impact assessment.
- **Equity** - fair access to benefit accounting for barriers.
- **Ethics washing** - ethical process without changed decisions.
- **Human in the loop** - a person reviews or authorises output.
- **Informed consent** - voluntary agreement after understanding.
- **Justice** - fair distribution of benefits and burdens.
- **Meaningful human control** - informed, resourced, authoritative review.
- **Non-maleficence** - avoiding harm.
- **Override rate** - proportion of recommendations rejected.
- **Paternalism** - overriding choices for a person's own good.
- **Procedural justice** - fairness of the decision process.
- **Redress** - a route to challenge and remedy.
- **Respect for persons** - treating people as agents.
- **Stakeholder** - a party affected by or able to affect the system.
- **Sunset** - planned decommissioning.
- **Therapeutic misconception** - believing research serves the participant individually.
- **Transparency** - disclosing what the system does and its limits.
- **Undue inducement** - an offer that compromises voluntary choice.
- **Vulnerability** - increased risk or diminished consent capacity.
- **Waiver of consent** - approval to proceed without consent.
- **Workflow fit** - how well a tool supports the care process.
- **Comprehension check** - a teach-back test that the participant understood.
- **Dissent** - a recorded disagreement in a review decision.
- **Review pack** - the evidence bundle behind a decision.

## 16. Sources

All URLs accessed 26 October 2026.

**Foundational ethics**
- National Commission for the Protection of Human Subjects, *The Belmont Report*, 1979.
  https://www.hhs.gov/ohrp/regulations-and-policy/belmont-report/index.html
- World Medical Association, *Declaration of Helsinki*, 2013.
  https://www.wma.net/policies-post/wma-declaration-of-helsinki-ethical-principles-for-medical-research-involving-human-subjects/
- CIOMS, *International Ethical Guidelines for Health-related Research Involving Humans*, 2016.
  https://cioms.ch/publications/product/international-ethical-guidelines-for-health-related-research-involving-humans/

**Regulation and guidance**
- 45 CFR Part 46, *Protection of Human Subjects*.
  https://www.ecfr.gov/current/title-45/subtitle-A/subchapter-A/part-46
- Regulation (EU) 2016/679 (GDPR), Article 22.
  https://eur-lex.europa.eu/eli/reg/2016/679/oj
- Regulation (EU) 2024/1689 (Artificial Intelligence Act), Article 27.
  https://eur-lex.europa.eu/eli/reg/2024/1689/oj
- Regulation (EU) 2025/327 (European Health Data Space).
  https://eur-lex.europa.eu/eli/reg/2025/327/oj
- UNESCO, *Recommendation on the Ethics of Artificial Intelligence*, 2021.
  https://unesdoc.unesco.org/ark:/48223/pf0000381137
- WHO, *Ethics and governance of artificial intelligence for health*, 2021.
  https://www.who.int/publications/i/item/9789240029200
- OECD, *AI Principles*, 2019, updated 2024. https://oecd.ai/en/ai-principles

**Case evidence**
- Ross, C., Swetlitz, I. *IBM's Watson supercomputer recommended 'unsafe and incorrect' cancer
  treatments, internal documents show*. STAT, 25 July 2018.
  https://www.statnews.com/2018/07/25/ibm-watson-recommended-unsafe-incorrect-treatments/
- Wong, A. et al. *External Validation of a Widely Implemented Proprietary Sepsis Prediction Model
  in Hospitalized Patients*. JAMA Internal Medicine 181(8):1065-1070, 2021.
  https://doi.org/10.1001/jamainternmed.2021.2626
- Obermeyer, Z., Powers, B., Vogeli, C., Mullainathan, S. *Dissecting racial bias in an algorithm
  used to manage the health of populations*. Science 366(6464):447-453, 2019.
  https://doi.org/10.1126/science.aax2342
- UK Information Commissioner's Office, *Google DeepMind and the Royal Free London NHS Foundation
  Trust*, 2017. https://ico.org.uk/for-the-public/ico-40/google-deepmind-and-class-action-lawsuit/
- Wagner, B. *Ethics as an Escape from Regulation: From 'Ethics-Washing' to Ethics-Shopping?* In
  *Being Profiled*, Amsterdam University Press, 2018.

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic: [`06_fairness.md`](06_fairness.md)
- Next topic: [`08_governance.md`](08_governance.md)
- Related topic: [`01_privacy-principles.md`](01_privacy-principles.md)
- Survey file: [`../../course/16_privacy_security_ethics.md`](../../course/16_privacy_security_ethics.md)
- Labs: [`../../labs/`](../../labs/)
