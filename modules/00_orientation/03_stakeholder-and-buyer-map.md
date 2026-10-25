# Stakeholder and Buyer Map

> Module 00 — Orientation and Industry Map
> Topic 03 of 10
> Estimated study time: 3 hours
> Prerequisites: 01, 02
> Version 2026.3

## 1. Why this topic exists

A biomedical data product is approved by a committee, used by a clinician, blocked by a security
officer, paid for by a finance director, and defended by a compliance lead. Most technical teams
can name only two of those people. The result is predictable: a product that works, a pilot that
succeeds technically, and a contract that never closes.

The stakeholder map is not a soft skill add-on. It is a technical artifact in the same sense that
a data dictionary is. It determines which evidence you generate, which integration you build,
which security controls you implement, and which claims you are allowed to make. A team that
skips it will discover the requirements one rejection at a time, each costing weeks.

This topic teaches you to map a deal from the clinician who first asks for the tool to the
procurement officer who signs the order. It covers who uses, who pays, who blocks, and who bears
harm, and it gives you a procedure for turning that map into a plan.

### 1.1 Why deals fail after the technical win

The pattern is common enough to have a name in sales literature: the "no-decision" loss. The
product was liked, the pilot worked, and nothing happened. The reasons are structural.

- The clinical champion had enthusiasm but no budget authority.
- The security review surfaced a requirement the vendor could not meet, such as data residency
  or a current SOC 2 Type II report.
- Legal could not agree on indemnification or on the business associate agreement (BAA).
- Procurement required three competitive bids, and the vendor was sole-sourced.
- The chief financial officer (CFO) did not see the value in the budget cycle.
- A department head whose staff would change workflow quietly opposed the project.

None of these are model problems. All of them are foreseeable with a stakeholder map and a deal
plan. The cost of mapping is a few days. The cost of discovering a blocker at contract stage is a
quarter.

### 1.2 Who this is for

If you are a data scientist, you will not run the sale. You will, however, produce the evidence
that each stakeholder needs, and you will be in the room for clinical and technical objections.
Understanding the map lets you answer the right question instead of the one you prepared for.

If you are a product manager or founder, the map is your operating document. It determines the
sequence of conversations, the pilot design, and the content of the security questionnaire.

If you are moving from academia, the biggest adjustment is that a correct result is necessary but
not sufficient. A result that is correct and unusable by the buyer has the same commercial value
as a wrong one.

### 1.3 What this topic does not cover

It does not teach sales methodology. It does not cover contracting law. It does not describe
every stakeholder in every segment; it gives a general map and segment-specific notes. The
regulatory content is orientation and is not legal advice.

## 2. Learning outcomes

| # | Outcome | Evidence |
|---:|---|---|
| 1 | Build a stakeholder map with role, influence, interest, stance, and evidence need | Exercise 1; self-check 2 |
| 2 | Distinguish user, buyer, champion, blocker, and beneficiary for a given product | Exercise 2; interview question 2 |
| 3 | Sequence a deal through the procurement gates that apply to a hospital or payer | Exercise 3; self-check 4 |
| 4 | Estimate deal probability and weighted pipeline from stage conversion | Exercise 5; self-check 6 |
| 5 | Write a one-page objection-handling brief for the three most likely blockers | Exercise 6 |
| 6 | Identify the security and legal artifacts a buyer will require and the lead time for each | Exercise 4; self-check 5 |
| 7 | Recognize when a stakeholder's stated objection is a proxy for an unstated concern | Exercise 7; interview question 8 |
| 8 | Decide when to walk away from a deal whose blocker cannot be resolved | Exercise 8 |

## 3. The core mental model

The model has two axes: a map of people and a path through gates. The map tells you who matters.
The path tells you when they matter. Most teams draw the map once and never connect it to the
timeline, which is why they are surprised by a blocker that was visible from the start.

### 3.1 The stakeholder grid

```text
                     HIGH INTEREST
                          |
   Keep informed          |        Manage closely
   (clinicians,           |        (champion, clinical
    front-line staff)     |         lead, CMIO)
                          |
   LOW INFLUENCE ---------+--------- HIGH INFLUENCE
                          |
   Monitor               |        Keep satisfied
   (peripheral           |        (CFO, security,
    departments)         |         legal, procurement)
                          |
                     LOW INTEREST
```

The two labels that matter most are "manage closely" and "keep satisfied". A champion with high
interest and high influence is necessary but not sufficient. A CFO with low interest and high
influence can stop the deal by doing nothing, which is why "keep satisfied" is an active
strategy, not an absence of one.

### 3.2 The deal path

```text
  [1] Clinical spark
       |  a clinician or operator says "we need this"
       v
  [2] Discovery and champion
       |  problem, workflow, current practice, success metric
       v
  [3] Clinical validation / pilot
       |  local data, safety review, IRB or QI determination
       v
  [4] Technical and security review
       |  architecture, SOC 2, penetration test, data flow, BAA
       v
  [5] Legal and contracting
       |  MSA, BAA, DPA, indemnity, liability, IP
       v
  [6] Procurement and finance
       |  budget, capital committee, competitive bid, purchase order
       v
  [7] Signature and implementation
       |  onboarding, integration, training, go-live
       v
  [8] Renewal and expansion
          usage evidence, outcome evidence, reference customer
```

Each gate can stop the deal, and each has a different owner. The clinical gate is owned by a
clinician or clinical committee. The security gate is owned by the chief information security
officer (CISO) or a security review team. The legal gate is owned by counsel. The procurement gate
is owned by a purchasing department with its own rules.

### 3.3 The stakeholder equation

For each stakeholder, three quantities determine strategy:

```text
  influence   = ability to stop or accelerate the deal      (1 to 5)
  interest    = attention they will give the project        (1 to 5)
  stance      = support or opposition                       (-2 to +2)

  blocking risk = influence x max(0, -stance)
  support leverage = influence x max(0, stance)
```

A stakeholder with influence 5 and stance -2 contributes 10 points of blocking risk. A
stakeholder with influence 2 and stance +2 contributes 4 points of support leverage. The
arithmetic is crude, but it forces you to name the opponents and weight them by power rather than
by how loudly they complain.

### 3.4 The two maps you need

There is a formal map and an informal one. The formal map is the org chart and the committee
structure. The informal map is who calls whom before the meeting. Both matter.

| Question | Formal map answers | Informal map answers |
|---|---|---|
| Who signs? | Procurement policy | Who the CFO trusts |
| Who can block? | Committee charter | Who has the CISO's ear |
| Who influences clinical use? | Department structure | The nurse educator everyone asks |
| Why did the last vendor fail? | Contract records | The story people tell in the corridor |

The informal map is built through interviews, not documents. One hour with a well-connected
informaticist is worth more than a week of org-chart analysis.

### 3.5 Harm bearers are stakeholders too

A map that includes only buyers and blockers is incomplete. Patients, caregivers, and
front-line staff who absorb the consequences of a wrong prediction are stakeholders even when
they never appear in a meeting. Their interests enter the deal through safety committees,
patient advisory boards, ethics review, and, occasionally, litigation.

Including harm bearers changes design. An alert that fires on a patient with no available
intervention creates work for a nurse and anxiety for a family. A model that performs worse for
patients with limited English proficiency creates a disparity that the buyer may not notice until
it is public. The map should name who bears the cost of a false positive and a false negative,
not only who pays for the software.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Stakeholder | Any person or group who affects or is affected by the project | Reduced to "the customer" | Role |
| User | Person who operates the product | Confused with buyer | Role |
| Buyer | Person or body with authority to commit funds | Confused with the most enthusiastic person | Role |
| Economic buyer | The person who controls the budget and can say yes | Assumed to be the CIO | Role |
| Champion | Internal advocate with influence and access | Assumed to have budget | Role |
| Blocker | Person who can prevent adoption or signature | Labelled without evidence | Role |
| Beneficiary | Person who gains from the outcome but may not use the product | Omitted from the map | Role |
| Harm bearer | Person exposed to the consequences of error | Omitted from commercial maps | Role |
| Influence | Ability to stop or accelerate a decision | Equated with seniority | Score 1 to 5 |
| Interest | Attention a stakeholder will give the project | Equated with enthusiasm | Score 1 to 5 |
| Stance | Support or opposition | Inferred from tone in one meeting | Score -2 to +2 |
| RACI | Responsible, accountable, consulted, informed | Applied to everything, diluting it | Matrix |
| Gate | A review that must be passed before the next stage | Assumed to be sequential when some run in parallel | Process step |
| Procurement | Purchasing function and its rules | Used for the whole sales process | Function |
| Capital committee | Body approving purchases above a threshold | Assumed to meet monthly | Governance body |
| Purchase order (PO) | Buyer's formal commitment document | Confused with contract | Document |
| Master services agreement (MSA) | Framework contract governing the relationship | Confused with a statement of work | Contract |
| Statement of work (SOW) | Document defining deliverables and scope | Treated as optional | Contract |
| BAA | Business associate agreement under HIPAA | Signed without security review | Contract |
| Data processing agreement (DPA) | Contract governing personal data processing under GDPR | Confused with a BAA | Contract |
| Data use agreement (DUA) | Agreement governing use of a limited data set or research data | Confused with a BAA | Contract |
| Security review | Assessment of vendor controls and architecture | Assumed to be a formality | Process |
| SOC 2 Type II | Independent attestation over controls across a period | Confused with a certification or a law | Audit report |
| HITRUST CSF | Healthcare security framework and certification | Assumed to be mandatory | Framework |
| HECVAT | Higher Education Community Vendor Assessment Toolkit | Assumed to apply only to universities | Questionnaire |
| VPAT | Voluntary Product Accessibility Template | Requested late in procurement | Document |
| Penetration test | Authorized simulated attack on a system | Confused with a vulnerability scan | Test |
| Data residency | Requirement that data be stored in a jurisdiction | Confused with data sovereignty | Policy |
| Minimum necessary | HIPAA principle limiting PHI to what is needed | Applied only to humans, not systems | Standard |
| Covered entity | HIPAA-regulated provider, plan, or clearinghouse | Applied to vendors | Legal category |
| Business associate | Vendor handling PHI on behalf of a covered entity | Assumed to be optional | Legal category |
| Controller / processor | GDPR roles for determining and processing personal data | Mapped loosely onto HIPAA roles | Legal role |
| Conflict of interest | Personal interest that could bias a decision | Disclosed only when financial | Governance |
| Objection | Stated reason for hesitation | Taken at face value | Statement |
| Proxy objection | Stated concern that stands in for an unstated one | Missed by technical teams | Inference |
| No-decision loss | Deal that ends without a yes or no | Counted as a pipeline loss only | Sales outcome |
| Pilot purgatory | Pilot that neither converts nor terminates | Counted as traction | Organizational state |
| Reference customer | Customer willing to speak publicly | Assumed to exist after one success | Asset |
| Buying committee | Group that collectively makes the decision | Assumed to be one person | Group |
| Clinical champion | Clinician who advocates for adoption | Assumed to have operational authority | Role |
| Informaticist | Professional bridging clinical workflow and IT | Confused with a data analyst | Role |
| CMIO | Chief medical information officer | Confused with the CMO | Role |
| CISO | Chief information security officer | Assumed to report to the CIO | Role |
| CFO | Chief financial officer | Assumed to care only about cost | Role |
| Service line leader | Executive accountable for a clinical service | Confused with a department chair | Role |

Two terms deserve expansion.

**Proxy objection.** A security officer says "we cannot approve cloud hosting". The stated
objection may be about data residency, or it may be about a past incident, a vendor relationship,
or a colleague's bad experience. Taking the stated objection at face value leads to a technical
response that does not resolve the concern. The skill is to ask what would need to be true for
approval, and to listen for the actual constraint.

**No-decision loss.** Most pipeline analyses count wins and losses. The largest category in
enterprise health technology is neither: the deal stalls. It is not lost to a competitor; it
simply never reaches a decision. Mapping stakeholders is the main defence, because stalls usually
mean an unaddressed blocker with no owner.

## 5. The mechanics, step by step

This section is a twelve-step procedure for mapping a deal and turning the map into a plan. It
assumes you have a defined product and a target segment from topics 01 and 02.

### Step 1: Define the purchase in one sentence

**Input:** product description.
**Output:** "This is a [category] that helps [role] at [organization type] do [decision] for
[price] per [period]."

**Reasoning.** The sentence determines who recognises the budget. A tool categorised as
"analytics" is evaluated by an analytics leader; the same tool categorised as "clinical
documentation" is evaluated by a clinical leader with a different budget. Category choice is a
stakeholder decision before it is a marketing decision.

**Branch point.** If the category does not exist in the buyer's budget taxonomy, the sale requires
creating a line item, which is slower. Consider attaching to an existing category even if the
product is broader.

### Step 2: List every role that touches the decision

**Input:** the purchase sentence and the deal path.
**Output:** a list of at least twelve roles, including users, buyers, blockers, beneficiaries, and
harm bearers.

**Reasoning.** Twelve is a useful minimum because it forces you past the obvious three. Typical
lists omit security, legal, procurement, finance, and the front-line staff who absorb workflow
change.

**Branch point.** If you cannot name twelve, you do not yet understand the organization. Schedule
discovery interviews before proceeding.

### Step 3: Score influence, interest, and stance

**Input:** role list.
**Output:** a scored table with a one-line evidence note for each score.

**Reasoning.** Scores are hypotheses, not facts. The evidence note forces you to record why you
believe a stakeholder is opposed, which is often the first time the team articulates it. Scores
should be revised after every conversation.

**Branch point.** If a high-influence stakeholder is scored negative without direct evidence,
the next action is an interview, not a mitigation plan.

### Step 4: Identify the economic buyer and the signature authority

**Input:** scored table.
**Output:** the name of the person who can commit the funds, and the approval threshold above
which a committee is required.

**Reasoning.** The economic buyer is not always the most senior person. In many health systems, a
service line leader can approve up to a threshold, above which the capital committee decides. The
threshold determines the sales motion and the evidence package.

**Branch point.** If the price exceeds the department threshold, the deal needs a capital
justification with a financial model. That is a different document from a clinical validation
report.

### Step 5: Map each stakeholder's evidence need

**Input:** role list.
**Output:** for each role, the one artifact that would satisfy them.

**Reasoning.** Different roles accept different evidence. A clinician wants local performance and
a safety story. A CISO wants a SOC 2 Type II report and a data flow diagram. Counsel wants
indemnification and liability terms. Finance wants a budget impact and a payback period. Producing
one document for all four wastes everyone's time.

**Branch point.** If two roles need contradictory evidence, for example a marketing claim that
exceeds what the clinical evidence supports, resolve the claim before the meeting. The conflict
will surface in legal review.

### Step 6: Sequence the gates

**Input:** deal path and evidence needs.
**Output:** a timeline with gates, owners, durations, and dependencies.

**Reasoning.** Gates are partly sequential and partly parallel. Security review can often run in
parallel with clinical validation, but contracting usually cannot start before security approval.
Sequencing determines the critical path and therefore the cash requirement.

**Branch point.** If the critical path exceeds the customer's budget cycle, the deal may slip a
full fiscal year. Identify the fiscal year boundary early.

### Step 7: Build the objection register

**Input:** stakeholder interviews.
**Output:** a table of objections with the stated concern, the underlying concern, the evidence
that addresses it, and the owner.

**Reasoning.** Objections are data. A register prevents the team from answering the same objection
in five different ways and allows the response to be tested. It also reveals objections that
cannot be answered, which are the ones that should stop the deal.

**Branch point.** If an objection is unresolvable with current evidence, either generate the
evidence or walk away. Promising to address it later is how pilots become permanent.

### Step 8: Design the pilot around the decision, not the demo

**Input:** objection register and gate sequence.
**Output:** a pilot protocol with a decision criterion, a named owner, a data access plan, and a
conversion price.

**Reasoning.** A pilot's purpose is to produce a decision. If the success criterion is "clinicians
like it", the pilot cannot fail and therefore cannot convert. The criterion should be a measurable
threshold tied to the buyer's evidence need.

**Branch point.** If the buyer will not agree to a conversion price before the pilot, the pilot is
a free option. Either negotiate a price range or treat the pilot as marketing spend with a
budget.

### Step 9: Prepare the security and legal package before it is requested

**Input:** product architecture.
**Output:** a due diligence pack containing SOC 2 Type II, pen test summary, data flow diagram,
subprocessor list, BAA template, DPA template, VPAT, and incident response summary.

**Reasoning.** Security review is a serial gate that commonly takes four to eight weeks even when
documentation is ready. Missing documentation restarts the clock. A prepared pack is the single
highest-leverage operational investment in enterprise health sales.

**Branch point.** If the product cannot meet a requirement such as data residency, decide whether
to change the architecture or to disqualify the segment. Selling into a requirement you cannot
meet produces a contract you cannot service.

### Step 10: Run the finance conversation in the buyer's units

**Input:** operational impact model.
**Output:** a one-page value case in the buyer's currency, per year, with assumptions.

**Reasoning.** Finance does not buy accuracy. It buys avoided cost, added capacity, reduced
length of stay, or improved reimbursement. The value case must use the buyer's own cost inputs,
not the vendor's, or it will be rejected as vendor math.

**Branch point.** If the value case depends on an assumption the CFO disputes, the deal stalls.
Ask the CFO which input they would accept and rebuild the model with it.

### Step 11: Manage the internal champion, not just the account

**Input:** stakeholder map.
**Output:** a champion enablement plan with materials the champion can use without you.

**Reasoning.** Champions spend political capital. They need a one-page summary, a safety story,
and answers to the three questions their colleagues will ask. A champion without materials becomes
a liability after the second meeting.

**Branch point.** If the champion cannot articulate the value without the vendor present, the deal
depends on the vendor's presence and will not scale.

### Step 12: Decide, document, and debrief

**Input:** all of the above.
**Output:** a go/no-go decision with reasons, and a post-decision review.

**Reasoning.** A deal that ends without a documented reason teaches nothing. The debrief should
record which gate stopped it, which stakeholder blocked it, and what would have changed the
outcome. Over ten deals, the pattern becomes the strategy.

**Branch point.** If the same gate stops most deals, fix the product or the segment, not the
sales pitch.

### 5.1 Segment-specific stakeholder notes

**Health system.** The buying committee typically includes a clinical champion, a department
leader, an informatics lead, IT security, legal, procurement, and finance. The chief medical
information officer (CMIO) is often the pivotal technical-clinical translator. Security review
commonly requires a SOC 2 Type II report and a BAA. Accessibility documentation (VPAT) is
increasingly requested.

**Payer.** The medical director and the analytics leader co-own the decision. Legal and compliance
review marketing claims closely because of state insurance regulation and nondiscrimination
exposure. Procurement may require a competitive process regardless of price.

**Pharma.** Clinical operations, translational medicine, and statistical programming are the
users. Procurement and quality assurance are the gates. Vendor qualification can take months and
requires documented quality systems, computer system validation, and data integrity controls.

**Medtech.** The device company is the customer during development; the hospital is the customer
at sale. The device company's regulatory affairs team controls the evidence package, and its
quality system controls change management.

**Consumer.** The user is the buyer, so the map collapses. The gatekeepers are app store review,
privacy regulators, and, for employer sales, benefits consultants and human resources.

### 5.2 Reading a room

Three signals indicate the real decision structure.

1. **Who speaks last.** In many committees, the most senior person speaks last and their reaction
   determines the outcome. Note who waits.
2. **Who is asked for permission.** When a proposal is made, watch whose nod the chair seeks.
   That is the informal approver.
3. **Who asks about risk.** The person asking about liability, safety, or security is often the
   gatekeeper even if they are not the most senior.

These signals are inference, not proof. Record them as hypotheses and test them in the next
conversation.

### 5.3 The discovery interview protocol

A discovery interview has one purpose: to find out how the organization decides, not to pitch.
The questions below are ordered from context to constraint. Ask at most six in a 45-minute
conversation and let the answers run.

| # | Question | What a useful answer contains | Trap to avoid |
|---:|---|---|---|
| 1 | "Walk me through the last time you made this decision." | A specific case, with dates and people | Generalities about process |
| 2 | "Who else was in the room?" | Names or roles, and who spoke last | "Everyone agreed" |
| 3 | "What nearly stopped it?" | A concrete blocker and how it was resolved | "Nothing, it was easy" |
| 4 | "What did you do before you had this tool?" | The true comparator and its cost | "Nothing" |
| 5 | "What would have to be true for you to say no?" | The real constraint | A polite deflection |
| 6 | "Who signs, and above what amount?" | Threshold and committee | "My boss" |
| 7 | "What evidence did the committee ask for?" | Artifacts and their authors | "They trust us" |
| 8 | "How long did the last purchase take, end to end?" | Months, with the slowest stage named | "A few weeks" |
| 9 | "What happens if you do nothing?" | The cost of the status quo | "We have to do something" |
| 10 | "Who will be upset if this changes?" | A named role and their concern | "No one" |
| 11 | "How will you know it worked in 12 months?" | A metric and an owner | "We will see" |
| 12 | "Who else should I speak to?" | Two or three names with reasons | A single name |

Three answers are especially diagnostic. "Nothing nearly stopped it" usually means the
interviewee was not close to the decision. "No one will be upset" usually means the interviewee
is not thinking about workflow. "We will see" usually means no one has defined success.

Record each interview within an hour, while the details are fresh. A useful note has four
fields: the decision described, the people named, the constraint revealed, and the next
introduction offered. Do not record opinions about the interviewee. Record what they said and
what you inferred, in separate columns, because the two are easily confused later.

Two interviews that contradict each other are more valuable than two that agree. A contradiction
means the organization has not resolved the decision, which is precisely the opening a new
product needs. Write the contradiction down and ask the next interviewee about it directly.

### 5.4 Deal review cadence

A deal with twelve stakeholders cannot be managed from memory. A weekly 30-minute review with a
fixed agenda keeps the map current.

| Minute | Item | Output |
|---:|---|---|
| 0 to 5 | Stage and date change | Whether the deal moved or stalled |
| 5 to 10 | Stakeholder changes | Any stance or influence revision, with evidence |
| 10 to 15 | Objection register | New objections and closed objections |
| 15 to 22 | Gate status | Security, legal, procurement, finance |
| 22 to 27 | Next commitments | Who does what by when |
| 27 to 30 | Risk and walk-away | Whether the deal still qualifies |

The review is not a forecast call. Its purpose is to keep the map honest. A deal that has not
moved in three consecutive reviews is a stalled deal, regardless of what the contact says.

## 6. Worked example

This example maps a deal for the deterioration alerting product from topic 02 into a 400-bed
community hospital. The organization and people are constructed; the structure and arithmetic
match a real enterprise sale.

### 6.1 The purchase

"This is a clinical deterioration alerting platform that helps charge nurses and rapid response
teams identify deteriorating ward patients earlier, priced at 120,000 USD per year."

### 6.2 Stakeholder map

Scores: influence 1 to 5, interest 1 to 5, stance -2 to +2.

| # | Role | Influence | Interest | Stance | Evidence needed |
|---:|---|---:|---:|---:|---|
| 1 | Chief nursing officer (CNO) | 5 | 4 | +2 | Nurse workload and safety data |
| 2 | Chief medical information officer (CMIO) | 5 | 5 | +1 | Local validation and EHR integration plan |
| 3 | Ward charge nurses (users) | 3 | 5 | +2 | Alert volume per shift; workflow fit |
| 4 | Rapid response team lead | 3 | 4 | +1 | Sensitivity and time-to-response data |
| 5 | CISO | 4 | 2 | -1 | SOC 2 Type II, pen test, data flow |
| 6 | Legal counsel | 4 | 1 | 0 | BAA, indemnity, liability terms |
| 7 | Procurement director | 3 | 1 | 0 | Competitive bid or sole-source justification |
| 8 | CFO | 5 | 2 | -1 | Budget impact and payback |
| 9 | Quality and safety director | 4 | 3 | +1 | Safety event analysis and monitoring plan |
| 10 | Patients and families (harm bearers) | 2 | 2 | 0 | Explanation of false-positive consequences |
| 11 | IT integration engineer | 3 | 3 | -1 | FHIR interface, latency, support model |
| 12 | Department chair of medicine | 4 | 2 | 0 | Peer-reviewed evidence and local champion |

Blocking risk = influence x max(0, -stance):

```text
CISO                    4 x 1 = 4
CFO                     5 x 1 = 5
IT integration engineer 3 x 1 = 3
Total blocking risk             = 12
```

Support leverage = influence x max(0, stance):

```text
CNO                     5 x 2 = 10
CMIO                    5 x 1 = 5
Charge nurses           3 x 2 = 6
Rapid response lead     3 x 1 = 3
Quality director        4 x 1 = 4
Total support leverage         = 28
```

The support-to-blocking ratio is 28 to 12, or 2.33. The deal is winnable but the two largest
blockers, the CFO and the CISO, have high influence and low interest. They are "keep satisfied"
stakeholders: they will not advocate, but they can stop the deal by inaction.

### 6.3 Stage probabilities

| Stage | Owner | Duration (weeks) | Conditional conversion |
|---|---|---:|---:|
| Discovery and champion | Account executive | 4 | 1.00 |
| Clinical validation | CMIO and quality director | 8 | 0.70 |
| Security review | CISO | 6 | 0.85 |
| Legal and BAA | Counsel | 6 | 0.90 |
| Procurement | Procurement director | 8 | 0.80 |
| Signature | CFO | 4 | 0.95 |

```text
Total duration = 4 + 8 + 6 + 6 + 8 + 4 = 36 weeks = 8.3 months

Cumulative probability = 1.00 x 0.70 x 0.85 x 0.90 x 0.80 x 0.95
                       = 0.40698
                       = 40.7%

Weighted value per deal = 120,000 USD x 0.40698
                        = 48,837.60 USD

Expected value of a 12-deal pipeline = 12 x 48,837.60
                                     = 586,051.20 USD
```

### 6.4 Cost of sale

```text
Sales time per deal          = 40 hours
Loaded cost per hour         = 85 USD
Cost per deal                = 40 x 85 = 3,400 USD
Cost for 12 deals            = 12 x 3,400 = 40,800 USD

Pilots run                   = 3
Cost per pilot               = 25,000 USD (engineering, clinical support, data work)
Pilot cost                   = 3 x 25,000 = 75,000 USD

Total cost to generate pipeline = 40,800 + 75,000 = 115,800 USD
Expected revenue               = 586,051.20 USD
Expected gross profit at 75%   = 439,538.40 USD
Expected contribution          = 439,538.40 - 115,800 = 323,738.40 USD
```

The arithmetic is favourable, but it hides two risks. First, the 36-week timeline means revenue
arrives in the next fiscal year, which may exceed the company's runway. Second, the 40.7%
probability is conditional on the clinical validation succeeding; if the local validation shows a
PPV below the buyer's threshold, the deal ends at stage 2 and the pilot cost is lost.

### 6.5 Sensitivity: what one blocker costs

If the CISO requires a four-week remediation before approval, the timeline becomes 40 weeks (9.2
months) and the deal slips past the hospital's fiscal year end. In many hospitals, unspent capital
does not roll over, so the deal restarts in the next budget cycle, adding 12 months.

```text
Original time to revenue  = 36 weeks
With remediation          = 40 weeks
With fiscal year slip     = 40 + 52 = 92 weeks = 21.1 months
```

That single blocker, scored 4 out of 25 possible blocking points, can turn an 8-month deal into a
21-month deal. This is why the security package in step 9 is worth preparing before it is asked
for.

### 6.6 The objection register

| Stated objection | Underlying concern | Response | Owner |
|---|---|---|---|
| "We cannot approve another cloud vendor" | Prior incident; data residency policy | Provide SOC 2 Type II, US data residency, incident summary | CISO |
| "Alert fatigue is already a problem" | Nurses will ignore it; workload | Show alerts per shift; offer threshold tuning and silent mode | CMIO |
| "The CFO will not fund a new line" | No budget; unclear payback | Value case in local costs; propose pilot funded from quality budget | CFO |
| "We tried a sepsis model and it did not work" | Prior failed deployment; trust | Present external validation and local silent-mode plan | CMIO |
| "Legal needs indemnification" | Liability exposure | Standard MSA terms; insurance certificate | Counsel |
| "Procurement needs three bids" | Policy threshold | Sole-source justification based on unique validation | Procurement |

The fourth row is the most important and the least technical. A prior failure in the same
institution creates a stance that no AUC will move. The response is a local silent-mode
evaluation with a pre-agreed review, not a better benchmark.

## 7. Code, query, or calculation

The block computes the stakeholder risk balance, the stage-weighted pipeline, and the cost of
sale. It uses only the standard library.

```python
"""Stakeholder map and deal pipeline arithmetic."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Stakeholder:
    role: str
    influence: int      # 1 to 5
    interest: int       # 1 to 5
    stance: int         # -2 to +2
    evidence: str = ""

    @property
    def blocking_risk(self) -> int:
        return self.influence * max(0, -self.stance)

    @property
    def support_leverage(self) -> int:
        return self.influence * max(0, self.stance)


@dataclass(frozen=True)
class Stage:
    name: str
    weeks: int
    conversion: float


STAKEHOLDERS = [
    Stakeholder("Chief nursing officer", 5, 4, 2),
    Stakeholder("CMIO", 5, 5, 1),
    Stakeholder("Ward charge nurses", 3, 5, 2),
    Stakeholder("Rapid response lead", 3, 4, 1),
    Stakeholder("CISO", 4, 2, -1),
    Stakeholder("Legal counsel", 4, 1, 0),
    Stakeholder("Procurement director", 3, 1, 0),
    Stakeholder("CFO", 5, 2, -1),
    Stakeholder("Quality and safety director", 4, 3, 1),
    Stakeholder("Patients and families", 2, 2, 0),
    Stakeholder("IT integration engineer", 3, 3, -1),
    Stakeholder("Department chair of medicine", 4, 2, 0),
]

STAGES = [
    Stage("Discovery and champion", 4, 1.00),
    Stage("Clinical validation", 8, 0.70),
    Stage("Security review", 6, 0.85),
    Stage("Legal and BAA", 6, 0.90),
    Stage("Procurement", 8, 0.80),
    Stage("Signature", 4, 0.95),
]

ACV = 120_000.0
DEALS = 12
SALES_HOURS_PER_DEAL = 40
LOADED_COST_PER_HOUR = 85.0
PILOTS = 3
PILOT_COST = 25_000.0
GROSS_MARGIN = 0.75


def cumulative_probability(stages):
    p = 1.0
    for s in stages:
        p *= s.conversion
    return p


print("Stakeholder scores")
print(f"{'role':30s} {'infl':>5s} {'int':>4s} {'stance':>7s} {'block':>6s} {'support':>8s}")
for s in STAKEHOLDERS:
    print(f"{s.role:30s} {s.influence:5d} {s.interest:4d} {s.stance:7d} "
          f"{s.blocking_risk:6d} {s.support_leverage:8d}")

blocking = sum(s.blocking_risk for s in STAKEHOLDERS)
support = sum(s.support_leverage for s in STAKEHOLDERS)
print(f"\nTotal blocking risk   {blocking}")
print(f"Total support leverage {support}")
print(f"Support-to-blocking ratio {support / blocking:.2f}")

p = cumulative_probability(STAGES)
weeks = sum(s.weeks for s in STAGES)
weighted = ACV * p
pipeline_value = DEALS * weighted

sales_cost = DEALS * SALES_HOURS_PER_DEAL * LOADED_COST_PER_HOUR
pilot_cost = PILOTS * PILOT_COST
total_cost = sales_cost + pilot_cost
gross_profit = pipeline_value * GROSS_MARGIN

print(f"\nCumulative probability {p:.5f}")
print(f"Total duration         {weeks} weeks ({weeks / 52:.1f} years)")
print(f"Weighted value/deal    {weighted:,.2f} USD")
print(f"Pipeline expected value {pipeline_value:,.2f} USD")
print(f"Cost of sale           {total_cost:,.2f} USD")
print(f"Expected gross profit  {gross_profit:,.2f} USD")
print(f"Expected contribution  {gross_profit - total_cost:,.2f} USD")

print("\nWhat if security review adds four weeks?")
extra = 4
weeks2 = weeks + extra
print(f"Duration {weeks2} weeks; slips fiscal year: {weeks2 + 52} weeks "
      f"({(weeks2 + 52) / 52:.1f} years)")
print(f"Weighted value unchanged at {weighted:,.2f} USD, but time to cash increases")
```

Expected output:

```text
Stakeholder scores
role                            infl  int  stance  block  support
Chief nursing officer              5    4       2      0       10
CMIO                               5    5       1      0        5
Ward charge nurses                 3    5       2      0        6
Rapid response lead                3    4       1      0        3
CISO                               4    2      -1      4        0
Legal counsel                      4    1       0      0        0
Procurement director               3    1       0      0        0
CFO                                5    2      -1      5        0
Quality and safety director        4    3       1      0        4
Patients and families              2    2       0      0        0
IT integration engineer            3    3      -1      3        0
Department chair of medicine       4    2       0      0        0

Total blocking risk   12
Total support leverage 28
Support-to-blocking ratio 2.33

Cumulative probability 0.40698
Total duration         36 weeks (0.7 years)
Weighted value/deal    48,837.60 USD
Pipeline expected value 586,051.20 USD
Cost of sale           115,800.00 USD
Expected gross profit  439,538.40 USD
Expected contribution  323,738.40 USD

What if security review adds four weeks?
Duration 40 weeks; slips fiscal year: 92 weeks (1.8 years)
Weighted value unchanged at 48,837.60 USD, but time to cash increases
```

Two checks. First, the blocking total is 4 plus 5 plus 3 equals 12, and the support total is 10
plus 5 plus 6 plus 3 plus 4 equals 28. Second, 12 deals times 48,837.60 USD equals 586,051.20
USD, which is the pipeline expected value before margin.

> **Note:** Multiplying stage probabilities assumes independence. They are not independent: a
> failed clinical validation often ends the relationship, and a failed security review can
> damage trust for the next attempt. Treat the result as an ordering device, not a forecast.

## 8. Data, standards, and regulatory context

Stakeholders are defined partly by the legal obligations they carry. The table maps role to
obligation.

| Role | Obligation | Instrument | What they will ask you for |
|---|---|---|---|
| Covered entity | Protect PHI; limit use to minimum necessary | HIPAA Privacy Rule, 45 CFR 164.502(b) | BAA, data flow, safeguards |
| Business associate | Implement safeguards; report breaches | HIPAA Security Rule; BAA terms | SOC 2, incident response plan |
| CISO | Manage information security risk | Organizational policy; HIPAA Security Rule | Pen test, architecture, subprocessors |
| Legal counsel | Manage contractual and liability risk | Contract law; state law | Indemnity, insurance, limitation of liability |
| Procurement | Follow purchasing policy | Organizational policy; state procurement law | Competitive bids, sole-source justification |
| Quality and safety | Monitor patient safety | Organizational policy; accreditation standards | Safety monitoring plan, event reporting |
| Privacy officer | Manage privacy risk | HIPAA; state privacy law; GDPR where applicable | DPIA, data retention, consent basis |
| Clinical committee | Protect patients in research and QI | IRB policy; 45 CFR 46 where research | Protocol, determination letter |
| Finance | Approve budget and capital | Organizational policy | Value case, payback, budget impact |
| Regulator | Enforce device, privacy, and payment law | FD&C Act; MDR/IVDR; AI Act; HIPAA | Submission or conformity evidence |

### 8.1 Research versus quality improvement

The determination that a project is quality improvement rather than human-subjects research
changes the stakeholder set. Research usually requires institutional review board (IRB) review
and informed consent unless waived. Quality improvement may not, depending on the institution and
the intent to generalize. The determination is made by the IRB or an institutional official, not
by the data scientist.

> **Note:** The boundary between QI and research is institution-specific and fact-specific. If a
> project may be published or generalized, assume research until an IRB says otherwise.

### 8.2 Security artifacts and what they mean

| Artifact | What it is | Who requires it | Typical lead time |
|---|---|---|---|
| SOC 2 Type II | Independent attestation over controls across 3 to 12 months | Most health systems | 6 to 12 months to obtain initially |
| Penetration test report | Authorized attack testing | CISO | 4 to 8 weeks |
| HITRUST certification | Healthcare-specific security certification | Some health systems and payers | 6 to 18 months |
| HECVAT | Vendor assessment questionnaire | Higher-education and some health systems | 1 to 3 weeks to complete |
| VPAT | Accessibility conformance statement | Procurement, especially public bodies | 1 to 2 weeks |
| Data flow diagram | Architecture and data movement | CISO and privacy | 1 to 2 weeks |
| Subprocessor list | Third parties handling data | Privacy and legal | Ongoing |
| BAA | HIPAA contract | Covered entity | 2 to 8 weeks negotiation |
| DPA | GDPR processing contract | EU customers | 2 to 8 weeks |

The lead times are approximate and vary by organization. The important point is that several are
measured in months, which means they belong on the critical path, not in a checklist completed
at contract stage.

### 8.3 Accessibility as a procurement gate

Publicly funded health systems increasingly require accessibility documentation. In the US, the
relevant standards include Section 508 of the Rehabilitation Act and the Web Content Accessibility
Guidelines (WCAG). A VPAT is the common evidence format. Accessibility is not only a compliance
issue; it determines whether clinicians with disabilities can use the product, which is a
workforce issue in a sector with documented staffing shortages.

## 9. Industry practice and case evidence

### 9.1 Governance is immature relative to adoption

**Documented.** A 2025 survey by the UPMC Center for Connected Medicine with KLAS Research,
covering 31 respondents, reported that only 16% had a system-wide AI governance policy. Most
relied on cross-functional executive committees, with chief information officers and chief data
and analytics officers most commonly involved, overseeing AI strategy, procurement, and
deployment. Security vetting, data use agreements, BAAs, and SOC 2 were described as standard.

**Documented.** A 2025 HFMA and Eliciting Insights survey of 233 health system executives
reported that 18% had a mature governance structure and a fully formed AI strategy, 29% had no AI
governance structure, and 88% used AI in some form. CFOs made up 33% of AI decision-makers.

**Documented.** KLAS Research's "Healthcare AI 2025" report, based on 256 respondents interviewed
in January 2025, found that AI was most widely purchased by large acute care organizations,
usually at departmental level rather than enterprise level, and most often for operational
efficiency at the point of care, especially documentation.

**Inference.** Departmental purchasing is faster but creates a fragmented stakeholder map. A
vendor selling to one department may win a contract without ever meeting the CISO, then discover
at renewal that enterprise governance has been introduced. The 16% and 18% figures suggest that
most buying organizations do not yet have a repeatable process, which means the vendor must
often supply the governance structure as part of the sale.

### 9.2 Security review is a serial gate

**Documented.** Public procurement pages from US institutions describe vendor security
assessments taking roughly 4 to 8 weeks at UC Berkeley, up to 4 weeks at San Diego State
University, 10 to 15 business days for risk and compliance review at CU Denver, and about 2
weeks at Providence College after complete documentation is received. The CMS Rapid Cloud Review
process for non-FedRAMP SaaS is described as 2 to 3 weeks for a provisional authority to operate,
followed by continuous monitoring.

**Inference.** These are university and government processes, not hospital processes, and they
should be treated as a lower bound. A hospital security review that includes a BAA, a HIPAA
risk analysis, and clinical system classification commonly runs longer. The practical planning
assumption for a first enterprise hospital sale is 8 to 12 weeks from complete documentation to
approval, with the clock restarting if documentation is incomplete.

### 9.3 The champion problem

**Documented.** KLAS's 2025 Global Summit report described a disconnect between IT leadership and
departmental staff in evaluating AI, and noted demand for predictable pricing, data sovereignty,
transparency, and outcomes-based or shared-risk procurement agreements.

**Inference.** The disconnect is a stakeholder-mapping failure, not a communication failure. IT
leadership evaluates security and integration; departmental staff evaluate workflow. If the two
groups never meet before the purchase decision, the vendor receives contradictory requirements
and the deal stalls while they reconcile.

### 9.4 A cautionary pattern: the pilot that could not convert

**Composite, constructed from common patterns.** A health system runs a six-month pilot of a
deterioration alert in two wards. Clinicians like it; the pilot reports "high satisfaction". At
month seven, procurement asks for a business case, and no one has measured length of stay,
transfer rate, or nurse time. The CFO declines to fund it. The vendor has spent six months of
engineering support and has no contract.

**Inference.** The pilot was designed around satisfaction because satisfaction was easy to
measure. The stakeholder who could approve the spend needed an operational metric. Mapping the
CFO's evidence need at step 5 would have changed the pilot design and possibly the outcome.

### 9.5 What buyers say they want in 2026

Across the 2025 surveys, four themes recur: predictable pricing, data sovereignty, transparency
about model behaviour, and outcomes-based contracting. These are stakeholder requirements, not
technical preferences.

- Predictable pricing addresses the CFO's fear of usage-based cost spirals.
- Data sovereignty addresses the CISO and privacy officer.
- Transparency addresses the clinical committee and, in the EU, the AI Act's transparency
  obligations.
- Outcomes-based contracts address the CFO and the value-based care team, and they transfer risk
  to the vendor.

A vendor that cannot answer these four will lose deals even with a strong model.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Champion has no authority | Champion says "I will convince them" | Enthusiasm mistaken for power | Deal stalls after pilot | Identify economic buyer in week one |
| Economic buyer never met | All meetings with clinicians | Buying committee incomplete | No budget, no decision | Request a finance meeting before pilot |
| Blockers discovered late | Security first mentioned at contract | No gate mapping | 3 to 6 month delay | Prepare due diligence pack early |
| Security package incomplete | Repeated requests for the same document | No owner for compliance artifacts | Review clock restarts | Assign a compliance owner |
| Legal terms unresolved | Redlines continue after price agreed | Indemnity and liability not scoped | Signature delayed | Agree term sheet before pilot |
| Procurement requires competitive bid | Sole-source assumption | Policy threshold | Deal restarts with competitors | Confirm procurement rules early |
| Value case in vendor units | CFO asks "where did this number come from" | Model built with vendor assumptions | Rejected as vendor math | Rebuild with buyer cost inputs |
| Pilot without conversion criteria | Success defined as satisfaction | Pilot designed to demo, not decide | Pilot purgatory | Pre-agree metric, threshold, price |
| Departmental sale, enterprise renewal | No CISO involvement | Fragmented governance | Renewal blocked | Engage enterprise governance early |
| Prior failed deployment ignored | "We tried this before" | Trust deficit | No technical response works | Local silent-mode validation |
| Harm bearers omitted | No discussion of false-positive burden | Map limited to buyers | Safety or equity issue after launch | Name harm bearers; design for them |
| Single-threaded relationship | One contact, no backups | Champion leaves or changes role | Deal dies | Multi-thread across at least four roles |

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| Stakeholder scores are hypotheses | Enables planning under uncertainty | Scores treated as facts | Revise after each interview |
| Gates are broadly sequential | Timeline estimation | Parallel review processes | Ask each gate owner about dependencies |
| Conversion probabilities are stable across deals | Pipeline arithmetic | Selection bias in deal choice | Compare forecast with actuals |
| The economic buyer is identifiable | Determines sales motion | Diffuse budgeting | Ask "who signed the last similar contract" |
| Security requirements are knowable in advance | Due diligence planning | Novel requirements per institution | Request the security questionnaire early |
| Legal terms are negotiable within market norms | Contract planning | Unusual indemnity demands | Ask for the standard MSA template |
| The champion will spend political capital | Adoption strategy | Champion has no slack | Ask what they have already sponsored |
| Harm-bearer concerns can be represented indirectly | Practical mapping | No proxy exists | Include patient advisory or ethics review |
| One deal generalizes to the segment | Strategy formation | Institution-specific procurement | Compare across at least five deals |

### 11.1 Limits

The scoring in section 3 is a heuristic. It is not validated and should not be presented as a
measurement. Its value is in forcing explicit judgements that can be tested.

The timeline in section 6 is a planning estimate for a first enterprise sale. Repeat sales to the
same institution are faster; sales to a new institution with a new security review are not.

The regulatory material is educational. Contract terms, BAAs, and DPAs require counsel.

## 12. Exercises

### Exercise 1: Build a stakeholder map (application, 40 minutes)

**Deliverable.** A twelve-role stakeholder map for a product you choose, with influence, interest,
stance, and the evidence each role needs.

**Acceptance criteria.** At least twelve roles. At least three roles are blockers or harm bearers.
Every score has a one-line evidence note. No role is "the hospital".

**Time.** 40 minutes.

### Exercise 2: Distinguish the roles (recall, 20 minutes)

**Deliverable.** For a sepsis alert, name the user, buyer, champion, blocker, beneficiary, and
harm bearer, with one sentence each.

**Acceptance criteria.** All six roles distinct. The beneficiary is not the same as the user. The
harm bearer includes a patient perspective.

**Time.** 20 minutes.

### Exercise 3: Sequence the gates (application, 30 minutes)

**Deliverable.** A timeline for a first hospital sale showing gates, owners, durations, and the
critical path. Mark which gates can run in parallel.

**Acceptance criteria.** At least six gates. The critical path is identified. The total duration
is stated in weeks. At least one dependency is explained.

**Time.** 30 minutes.

### Exercise 4: Build the due diligence pack list (application, 30 minutes)

**Deliverable.** A checklist of the ten documents a hospital CISO and legal team will request,
with the owner and lead time for each.

**Acceptance criteria.** Includes SOC 2 Type II, pen test, data flow, BAA, DPA, VPAT, and
subprocessor list. Lead times in weeks or months. Distinguishes documents the vendor produces
from documents the buyer produces.

**Time.** 30 minutes.

### Exercise 5: Pipeline arithmetic (calculation, 40 minutes)

**Deliverable.** Using the stage table in section 6, compute the cumulative probability, weighted
value per deal, and expected pipeline value for 20 deals. Then recompute with clinical validation
conversion at 0.55.

**Acceptance criteria.** Arithmetic shown. Units on every figure. One sentence on how a 15
percentage point drop in one stage changes the expected pipeline value.

**Time.** 40 minutes.

### Exercise 6: Objection brief (design, 35 minutes)

**Deliverable.** A one-page brief addressing the three most likely objections for your product,
with the stated objection, underlying concern, evidence, and owner.

**Acceptance criteria.** Three objections, each with a distinct underlying concern. Evidence is
specific and available. No objection is answered with "we will follow up".

**Time.** 35 minutes.

### Exercise 7: Find the proxy objection (critique, 25 minutes)

**Deliverable.** A short analysis of a real or constructed objection that is a proxy for an
unstated concern, and the three questions you would ask to surface the real issue.

**Acceptance criteria.** Distinguishes stated from underlying concern. Questions are open-ended
and non-defensive. Identifies what would count as resolution.

**Time.** 25 minutes.

### Exercise 8: Walk-away decision (design, 25 minutes)

**Deliverable.** A one-page decision memo recommending whether to continue or exit a deal where
the CISO requires on-premises deployment that the product cannot support.

**Acceptance criteria.** Names the cost of compliance, the strategic value of the segment, and a
clear recommendation. States what would change the recommendation.

**Time.** 25 minutes.

## 13. Self-check answers

**1. Why is the economic buyer often not the most enthusiastic person?**
Because enthusiasm and budget authority are different attributes. Clinicians have interest and
influence over use but often no discretionary budget. Finance leaders have budget authority and
low interest. Both must be engaged.

**2. What is the difference between a user and a beneficiary?**
The user operates the product. The beneficiary gains from the outcome and may never touch it, for
example a patient who avoids an ICU transfer. Beneficiaries rarely appear in procurement but
often appear in safety and ethics review.

**3. Why does a stakeholder map need harm bearers?**
Because the product can create costs for people who never consented to it. Naming them changes
design, monitoring, and the safety case. It also anticipates the stakeholder who will appear
after an adverse event.

**4. What are the typical gates in a hospital sale?**
Clinical validation, security review, legal and BAA, procurement, and finance approval. Each has
an owner and a duration, and the critical path usually runs through security and legal.

**5. Why is a SOC 2 Type II report often required?**
It is an independent attestation that controls operated over a period, which gives the buyer
evidence rather than a vendor assertion. It is not a law and not a certification, but it is a
common procurement requirement.

**6. Compute the cumulative probability for stages 1.00, 0.70, 0.85, 0.90, 0.80, 0.95.**
1.00 x 0.70 = 0.70; x 0.85 = 0.595; x 0.90 = 0.5355; x 0.80 = 0.4284; x 0.95 = 0.40698, or
40.7%.

**7. Why is satisfaction a poor pilot criterion?**
Because it does not predict purchase. A pilot can achieve high satisfaction and produce no
operational or financial evidence. The criterion should be tied to the economic buyer's decision.

**8. What does a support-to-blocking ratio of 2.33 tell you?**
Support outweighs blocking, so the deal is winnable, but two high-influence blockers have low
interest, which means they can stop the deal by inaction. The strategy is to convert at least one
blocker into a neutral or supporter before the procurement gate.

**9. When should you walk away from a deal?**
When a requirement cannot be met without changing the product in a way that breaks the strategy,
or when the evidence needed to satisfy the buyer costs more than the deal is worth.

**10. What is a no-decision loss?**
A deal that ends without a yes or no, usually because a blocker was never addressed. It is the
most common enterprise outcome and the least visible in pipeline reporting.

### 13.1 Rubric for exercises 1 and 6

Exercise 1 is scored on completeness (twelve roles, 0 to 3), quality of evidence notes (0 to 3),
and inclusion of blockers and harm bearers (0 to 2). A map with twelve clinicians and no finance
or security role scores at most 4.

Exercise 6 is scored on whether the underlying concern is genuinely different from the stated
objection, whether the evidence is available now, and whether an owner is named.

## 14. Interview questions

**1. "How do you map stakeholders for a new product?"**
Strong: describes roles, influence, interest, stance, evidence needs, and gates. Mentions harm
bearers. Weak: "I talk to clinicians".

**2. "Who is the buyer for a clinical AI tool?"**
Strong: distinguishes user, champion, economic buyer, and committee; notes threshold-based
approval. Weak: "the hospital".

**3. "Why do deals stall after a successful pilot?"**
Strong: no conversion criteria, no economic buyer engaged, no budget line, security or legal
gate not anticipated. Weak: "the model was not good enough".

**4. "What would you prepare before a security review?"**
Strong: SOC 2 Type II, pen test, data flow, subprocessors, BAA, DPA, VPAT, incident response.
Weak: "a privacy policy".

**5. "How do you handle a blocker who will not engage?"**
Strong: find the underlying concern, use an intermediary, test whether the objection is a proxy,
and set a decision point. Weak: "escalate to their boss".

**6. "How do you build a value case for a CFO?"**
Strong: uses the buyer's cost inputs, states assumptions, gives payback in months, includes
sensitivity. Weak: quotes an AUC or a vendor ROI study.

**7. "What is the difference between a BAA and a DPA?"**
Strong: BAA implements HIPAA obligations for business associates; DPA implements GDPR
controller-processor obligations. They can coexist and are not interchangeable.

**8. "How do you tell whether an objection is real?"**
Strong: ask what would need to be true for approval, observe whether the answer changes when
evidence is provided, and check whether the same person blocks other vendors. Weak: takes the
objection at face value.

**9. "When is a pilot worth running?"**
Strong: when it produces a decision, has a pre-agreed metric and price, and the buyer has
authority. Weak: "when the customer asks for one".

**10. "How do you handle a champion who leaves?"**
Strong: multi-threading, documented value case, relationship with the committee rather than one
person. Weak: "start over".

**11. "What do you do when procurement requires three bids?"**
Strong: confirm the threshold and policy, prepare a sole-source justification if legitimate, or
accept the competitive process and differentiate on evidence. Weak: "we do not do competitive
bids".

**12. "How would you sell into a health system with no AI governance?"**
Strong: help create the governance, provide the security and evidence package, start
departmental but plan for enterprise review. Weak: exploit the gap and ignore the risk.

## 15. Key terms

- **BAA** — HIPAA business associate agreement.
- **Beneficiary** — person who gains from the outcome without using the product.
- **Blocker** — stakeholder who can prevent adoption or signature.
- **Buying committee** — group that collectively decides.
- **Capital committee** — body approving spend above a threshold.
- **Champion** — internal advocate with influence.
- **CISO** — chief information security officer.
- **CMIO** — chief medical information officer.
- **Covered entity** — HIPAA-regulated organization.
- **DPA** — GDPR data processing agreement.
- **DUA** — data use agreement.
- **Economic buyer** — person who controls the budget.
- **Gate** — review that must be passed before the next stage.
- **Harm bearer** — person exposed to the consequences of error.
- **HECVAT** — vendor security questionnaire.
- **HITRUST** — healthcare security framework.
- **Influence** — ability to stop or accelerate a decision.
- **IRB** — institutional review board.
- **MSA** — master services agreement.
- **No-decision loss** — deal that ends without a yes or no.
- **Objection register** — structured list of objections and responses.
- **Penetration test** — authorized simulated attack.
- **Pilot purgatory** — pilot that neither converts nor terminates.
- **PO** — purchase order.
- **Procurement** — purchasing function.
- **Proxy objection** — stated concern standing in for an unstated one.
- **RACI** — responsible, accountable, consulted, informed.
- **Reference customer** — customer willing to speak publicly.
- **SOC 2 Type II** — attestation over controls across a period.
- **Stakeholder** — person or group affected by the project.
- **Stance** — support or opposition score.
- **VPAT** — accessibility conformance statement.
- **WCAG** — web accessibility guidelines.

## 16. Sources

All URLs accessed on 24 October 2026 unless otherwise stated.

- UPMC Center for Connected Medicine and KLAS Research, "How Health Systems Are Navigating the
  Complexities of AI," 2025.
  https://enterprises.upmc.com/app/uploads/2025/03/How_Health_Systems_Are_Navigating_The_Complexities_Of_AI_CCM_Reports.pdf
- KLAS Research, "Healthcare AI 2025: Are You Keeping Pace with Industry Adoption?", April 2025
  (256 respondents, January 2025 fieldwork). https://klasresearch.com/report/healthcare-ai-2025-are-you-keeping-pace-with-industry-adoption/3749
- KLAS Research, "KLAS Global Summit 2025: Organizations Investing in AI and the EHR to Improve the
  Clinician Experience," August 2025.
  https://klasresearch.com/report/klas-global-summit-2025-organizations-investing-in-ai-and-the-ehr-to-improve-the-clinician-experience/3695
- Healthcare Financial Management Association and Eliciting Insights, "Health System Readiness for
  Artificial Intelligence," 2025 (233 executives).
  https://www.hfma.org/wp-content/uploads/2025/08/Overview-Health-System-Readiness-for-AI.pdf
- TechTarget, "How health systems are facilitating AI governance," August 2025.
  https://www.techtarget.com/healthtechanalytics/feature/How-health-systems-are-facilitating-AI-governance
- University of California, Berkeley, Supply Chain Management, "Step 3 - Supplier Security
  Assessment." https://supplychain.berkeley.edu/procurement/buying-software-and-cloud-computing-services/step-3-supplier-security-assessment
- Centers for Medicare and Medicaid Services, "Rapid Cloud Review (RCR)," CMS Information
  Security and Privacy Program. https://security.cms.gov/learn/rapid-cloud-review-rcr
- US Department of Health and Human Services, HIPAA Privacy Rule, 45 CFR 164.502(b), minimum
  necessary standard. https://www.hhs.gov/hipaa/
- US Department of Health and Human Services, HIPAA Security Rule.
  https://www.hhs.gov/hipaa/for-professionals/security/index.html
- European Parliament and Council, Regulation (EU) 2016/679 (GDPR), controller and processor
  obligations, Articles 4, 24 to 28.
- US Access Board, Section 508 standards. https://www.access-board.gov/ict/
- World Wide Web Consortium, Web Content Accessibility Guidelines (WCAG) 2.2.
  https://www.w3.org/TR/WCAG22/
- AICPA, SOC 2 trust services criteria. https://www.aicpa-cima.com/topic/audit-assurance/audit-and-assurance-greater-than-soc-2

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`02_the-2026-industry-landscape.md`](02_the-2026-industry-landscape.md)
- Next topic: [`04_career-tracks-and-role-definitions.md`](04_career-tracks-and-role-definitions.md)
- Related topic: [`05_the-project-charter.md`](05_the-project-charter.md)
- Related topic: [`08_ethics-and-professional-conduct.md`](08_ethics-and-professional-conduct.md)
- Module survey file: [`../../00_orientation_and_industry_map.md`](../../00_orientation_and_industry_map.md)
- Labs: [`../../labs/`](../../labs/)
- Cohort pipeline lab: [`../../labs/03_cohort_pipeline.ipynb`](../../labs/03_cohort_pipeline.ipynb)
