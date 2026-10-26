# Leadership

> Module 17 — Product, Economics, and Leadership
> Topic 08 of 09
> Estimated study time: 4 hours
> Prerequisites: 05 (Procurement), 06 (Adoption)
> Version 2026.3

## 1. Why this topic exists

A data team's output is not a model. It is a decision that a person with authority makes, on the
basis of evidence the team produced, at a time when the decision still matters. Teams that do not
manage the decision layer produce analyses that are correct, late, and unused.

Leadership in this context does not mean managing people. It means making the decision system
work: assigning decision rights, setting the evidence bar, running the forum, recording the
rationale, and closing the loop. A senior individual contributor who can do those things has more
influence than a manager who cannot.

### 1.1 The two failure patterns

**The analysis that arrives after the decision.** The team spends six weeks on a model evaluation,
presents it at the wrong committee, and learns that the budget was set three weeks earlier. The
work was good and the timing made it worthless.

**The decision that is never made.** The committee discusses, defers, requests more data, and meets
again. The deferral is not neutral: it consumes capacity and forgoes the value of acting. In the
worked example, a one-month delay in a portfolio decision reduces the available engineering
capacity by a third and forgoes 370,000 United States dollars (USD) of expected value.

### 1.2 Why influence without authority is the core skill

A data scientist usually has no authority over the clinician whose workflow changes, the finance
director who releases the budget, or the security officer who can block the deployment. Influence
comes from four sources, and only one of them is formal.

| Source | What it is | How to build it |
|---|---|---|
| Expertise | Being right often enough to be believed | Publish assumptions; show calibration |
| Evidence | Producing numbers others can audit | Write down sources and methods |
| Relationship | Being trusted by the people who decide | Deliver on small commitments |
| Position | Formal authority | Rarely available to technical staff |

The first two are the ones a technical leader can build fastest, and they compound: an analyst
whose numbers survive audit is invited to the next decision.

### 1.3 What this topic covers

It covers decision classification, decision rights, the written decision memo, running a review,
recording the outcome, and auditing the decision later. It does not cover people management,
performance reviews, or organisational design. It is not a course in public speaking.

## 2. Learning outcomes

| # | Outcome | Evidence |
|---:|---|---|
| 1 | Classify a decision by reversibility and consequence | Exercise 1; self-check 1 |
| 2 | Assign decision rights with a named decider and named inputs | Exercise 2; self-check 2 |
| 3 | Frame a decision as a choice among options including do-nothing | Exercise 3; self-check 3 |
| 4 | Write a two-page decision memo with a recommendation, alternatives, and a pre-mortem | Exercise 4; interview question 4 |
| 5 | Run a review that produces a decision in the room | Exercise 5; interview question 6 |
| 6 | Record a decision with its rationale, owner, and review date | Exercise 6; self-check 7 |
| 7 | Compute the cost of decision delay and use it to set a decision deadline | Exercise 7; self-check 8 |
| 8 | Audit a past decision against the information available at the time | Exercise 8; interview question 10 |

## 3. The core mental model

Leadership is a system with five components: rights, framing, evidence, forum, and record. A
decision fails when any one is missing, and the failure looks different depending on which.

### 3.1 The decision system

```text
  [1] RIGHTS        Who decides, who must be consulted, who must be informed
        |
        v
  [2] FRAMING       The question, the options, the criteria, the deadline
        |
        v
  [3] EVIDENCE      The minimum evidence proportionate to reversibility
        |
        v
  [4] FORUM         The meeting or document where the decision is actually made
        |
        v
  [5] RECORD        What was decided, why, by whom, and when it will be reviewed
        |
        v
  [6] AUDIT         Whether the decision was good given what was knowable
```

Most teams invest in [3] and neglect [1], [2], and [5]. That is why good analysis produces bad
outcomes: the analysis is sound and the system around it is not.

### 3.2 Reversibility determines the process

A reversible decision should be made quickly, by a small group, with the evidence available. An
irreversible decision should be made slowly, with wider consultation and a written record.

```text
  reversible, low consequence      ->  decide now, one owner, log it
  reversible, high consequence     ->  decide in a week, consult, log it
  irreversible, low consequence    ->  decide in a week, document the rationale
  irreversible, high consequence   ->  decide in a month, written memo, pre-mortem,
                                       explicit dissent, board or executive approval
```

Treating a reversible decision as irreversible produces paralysis. Treating an irreversible
decision as reversible produces a costly reversal, and in clinical contexts, harm.

### 3.3 The influence ledger

Influence without authority works through a ledger of favours, evidence, and credibility. Every
interaction is a deposit or a withdrawal.

Deposits: delivering a number that survives scrutiny, warning someone early about a problem in
their area, making a decision easy for a busy person, and admitting an error before it is found.

Withdrawals: presenting a number that cannot be traced, escalating without notice, surprising a
senior person in a meeting, and claiming credit for a shared result.

A technical leader who has made ten deposits can spend one on a contentious recommendation. One
who has made none cannot, regardless of how good the analysis is.

### 3.4 The memo is the decision

In organisations that use written decisions, the memo is not a record of a conversation; it is the
mechanism by which the conversation happens. Writing forces the author to name the options, the
criteria, the uncertainties, and the recommendation. A memo that cannot be written is usually a
decision that has not been thought through.

The corollary: a slide deck is a presentation tool, not a decision tool, because it hides the
reasoning in the presenter's head. That is the argument behind the narrative memo practice
described in Section 9.

### 3.5 Decision quality versus outcome quality

A good decision can have a bad outcome and a bad decision can have a good one. The only fair audit
asks whether the decision was reasonable given what was knowable at the time, not whether it
worked out.

```text
  good process + good outcome  =  deserved success
  good process + bad outcome   =  bad luck; keep the process
  bad process  + good outcome  =  luck; do not learn the wrong lesson
  bad process  + bad outcome   =  deserved failure
```

Organisations that reward outcomes rather than process teach their people to hide risk and to
claim credit for luck. That is a leadership failure, not an individual one.

### 3.6 Redraw test

Draw the six components of the decision system for a decision you are currently involved in. Mark
the component that is weakest. In most cases it is rights or record, and in most cases nobody has
noticed.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Context |
|---|---|---|---|
| Decision rights | Who has authority to decide | Assumed to follow the org chart | Governance |
| Decider | The single person accountable for a decision | Confused with the most senior attendee | Governance |
| Consulted | Roles whose input must be sought before deciding | Confused with co-deciders | Governance |
| Informed | Roles told after the decision | Confused with consulted | Governance |
| RACI | Responsible, accountable, consulted, informed | Used without naming a decider | Framework |
| RAPID | Recommend, agree, perform, input, decide | Confused with RACI | Framework |
| DACI | Driver, approver, contributor, informed | Confused with RACI | Framework |
| Type 1 decision | Irreversible, high consequence | Used for any important decision | Classification |
| Type 2 decision | Reversible, low consequence | Used to avoid consultation | Classification |
| Decision latency | Time from decision readiness to decision | Confused with analysis time | Days |
| Decision velocity | Decisions per period at a given quality | Confused with speed alone | Rate |
| Decision quality | Whether the decision was reasonable given the evidence | Judged by outcome | Construct |
| Decision log | Record of decisions, rationale, owner, date | Omitted | Artifact |
| Pre-mortem | Prospective hindsight: imagine failure and explain it | Confused with a risk register | Technique |
| Red team | Independent adversarial review | Confused with peer review | Technique |
| Dissent | Recorded disagreement with the decision | Suppressed | Governance |
| Disagree and commit | Record disagreement, then execute | Confused with consensus | Norm |
| Escalation | Raising a decision to a higher authority | Confused with complaint | Process |
| Decision owner | Person accountable for execution after the decision | Confused with the decider | Governance |
| Stakeholder | Anyone affected by or interested in the decision | Used too broadly | Mapping |
| Sponsor | Senior person who funds and defends the work | Confused with the decider | Role |
| Champion | Advocate who builds support | Confused with the sponsor | Role |
| Psychological safety | Belief that speaking up is safe | Confused with niceness | Construct |
| Blameless review | Review focused on system causes, not individuals | Confused with no accountability | Practice |
| After-action review | Structured review of what happened and why | Confused with a status meeting | Practice |
| Decision audit | Review of a past decision against contemporaneous evidence | Confused with outcome review | Practice |
| Option value | Value of keeping a choice open | Ignored in urgent decisions | Economics |
| Do-nothing option | The baseline of not deciding | Omitted from option sets | Framing |
| Opportunity cost | Value of the best forgone alternative | Ignored in portfolio decisions | Economics |
| Capacity constraint | The resource that limits how much can be done | Assumed to be money | Operations |
| Portfolio | Set of projects competing for capacity | Confused with a list | Management |
| Expected value | Probability-weighted outcome | Confused with the base case | Calculation |
| Sensitivity analysis | How conclusions change with inputs | Confused with a scenario | Analysis |
| Hurdle rate | Minimum return for an investment | Confused with a target | Finance |
| Governance | The system of rights, forums, and records | Confused with compliance | System |
| Candour | Duty to be open about failures | Confused with transparency to the public only | Ethic |

## 5. The mechanics, step by step

The procedure covers a single decision from framing to audit. It assumes the decision is
consequential enough to warrant a written record, which is most decisions a data team influences.

### Step 1. Write the decision as a question with a deadline

**Input:** a request, a problem, or a proposal.
**Output:** one sentence of the form "we will decide X by date Y."
**Rationale:** a decision without a deadline is a topic. A decision without a question is a
briefing. Writing both converts the work into something that can be closed.

Example: "We will decide by 30 November whether to fund the sepsis model upgrade, the ambient
documentation expansion, both, or neither, for the January quarter."

The sentence contains the options, the date, and the scope. If any is missing, the decision will
be reopened.

### Step 2. Classify the decision by reversibility and consequence

**Input:** the question.
**Output:** a class and a corresponding process.
**Rationale:** the process should be proportionate. Applying an irreversible-decision process to a
reversible one wastes weeks; the reverse creates risk.

| Reversibility | Consequence | Class | Process | Typical time |
|---|---|---|---|---|
| Reversible | Low | 2 | Decide individually, log it | Same day |
| Reversible | High | 2 | Small group, one week, log it | 1 week |
| Irreversible | Low | 1 | Written rationale, one reviewer | 1 to 2 weeks |
| Irreversible | High | 1 | Memo, pre-mortem, wider consultation, senior approval | 4 to 8 weeks |

The classification is a judgement, and people systematically misclassify in the direction that
suits them: a sponsor who wants speed calls an irreversible decision reversible. Ask what it would
cost to undo; if the answer is more than the cost of the decision process, it is not reversible.

### Step 3. Assign decision rights explicitly

**Input:** the question and the organisation.
**Output:** a named decider, named consultees, and a named informed group.
**Rationale:** decisions with no decider are decided by the loudest or the most senior person
present, and decisions with several deciders are decided by nobody.

Use a simple framework and write it in the memo header:

| Role | Definition | Example |
|---|---|---|
| Decider | The one person who decides | Chief medical information officer |
| Recommender | The person who prepares the recommendation | Data science lead |
| Consulted | People whose input must be sought, with a deadline | Security, finance, nursing |
| Informed | People told after the decision | Department chairs |

RAPID, DACI, and RACI are variants of the same idea. The important property is not the acronym but
the singularity of the decider.

Edge case: in a matrix organisation, a decision may require joint sign-off from two functions, for
example clinical and information security. Joint sign-off is legitimate if it is explicit, but it
must have a tie-break rule, or it becomes a veto for either party.

A second edge case is the delegated decision. A decider who delegates to a committee without
specifying who on the committee decides has not delegated; they have created a forum. If a
decision must be delegated, delegate it to a named person with a stated budget and a stated
constraint, and record the delegation in the same log.

### Step 4. Set the evidence bar proportionate to the decision

**Input:** the decision class.
**Output:** a list of the evidence required and the evidence explicitly not required.
**Rationale:** evidence collection expands to fill the time available. Naming what is not needed is
as important as naming what is.

| Decision class | Evidence required | Evidence not required |
|---|---|---|
| 2, low | The number, with a source | Sensitivity analysis |
| 2, high | The number, sensitivity, and a counter-case | Prospective validation |
| 1, low | Written rationale and one independent review | Full economic model |
| 1, high | Model, sensitivity, pre-mortem, dissenting view, legal review | Perfect information |

The last cell is the important one. No decision has perfect information, and a process that waits
for it never decides.

### Step 5. Write the memo

**Input:** the evidence.
**Output:** a document of two to six pages with a fixed structure.
**Rationale:** the structure forces completeness and makes comparison across memos possible.

The structure:

1. **Decision requested.** One sentence, in the imperative: "Approve option B plus C for the
   January quarter."
2. **Context.** Three to five sentences on why this decision is needed now.
3. **Options.** At least three, including do-nothing, each with cost, benefit, risk, and
   reversibility.
4. **Criteria.** The dimensions on which options are compared, with weights if the comparison is
   multi-dimensional.
5. **Analysis.** The comparison, with the arithmetic and its sources.
6. **Recommendation.** One option, with the reason in one paragraph.
7. **Risks and mitigations.** The three that matter, with an owner for each.
8. **Pre-mortem.** If this fails in twelve months, the three most likely reasons.
9. **What would change our mind.** The evidence that would reverse the recommendation.
10. **Decision record.** The decider, the date, and where the decision will be logged.

Section 9 is the one that distinguishes a decision memo from a pitch. A memo that cannot say what
would change its mind is advocacy.

### Step 6. Run a pre-mortem

**Input:** the draft recommendation.
**Output:** a list of failure modes with likelihood and early warning signs.
**Rationale:** prospective hindsight, imagining that the decision has already failed and explaining
why, surfaces risks that optimism suppresses. It is cheap and it works better in a group than in
the author's head.

The protocol:

1. State: "It is twelve months from now and this failed. Write down why."
2. Each participant writes three reasons independently, in silence, for five minutes.
3. Collect and group the reasons.
4. For the top three, name an early warning sign and a mitigation.
5. Add the top reason to the memo's risk section, attributed to the group rather than the author.

### Step 7. Circulate the pre-read and collect written objections

**Input:** the memo.
**Output:** written objections from each consultee before the meeting.
**Rationale:** objections raised in the room are shaped by hierarchy and by who spoke first.
Written objections are more honest and easier to address.

Rules: circulate at least 48 hours ahead; ask each consultee for a written response with a
position (support, support with conditions, oppose); and include the objections in the appendix
whether or not they change the recommendation.

Edge case: in a clinical setting, a safety objection must be able to stop the decision regardless
of the weight of other views. Name the safety veto holder in advance.

### Step 8. Run the meeting so that the decision happens in the room

**Input:** the memo and the objections.
**Output:** a decision, recorded before the meeting ends.
**Rationale:** a meeting that ends without a decision has consumed the most expensive resource in
the organisation and produced nothing.

The protocol:

- **Silent read** of the memo at the start, five to ten minutes, even if it was circulated. This
  equalises information.
- **Speaking order** from most junior or least senior to most senior, so that the senior view does
  not anchor the room.
- **Objections first**, then the recommendation.
- **The decider speaks last** and states the decision explicitly.
- **Record in the room**: the decision, the rationale, the dissent, the owner, and the review date.
- **Close with the next decision**, not with an action list.

A useful discipline for the chair: if the discussion exceeds the time box without converging, the
decision is "defer with a named question and a date," not "continue discussing." A deferral is a
decision and should be recorded as one.

A workable 45-minute agenda:

| Minutes | Segment | Purpose |
|---:|---|---|
| 0 to 8 | Silent read | Equalise information |
| 8 to 15 | Objections, least senior first | Surface disagreement before advocacy |
| 15 to 25 | Author response | Address each objection explicitly |
| 25 to 35 | Discussion | Test the recommendation against the criteria |
| 35 to 42 | Decider states the decision | Close the loop in the room |
| 42 to 45 | Record read back | Confirm decision, owner, dissent, review date |

The read-back is the step teams skip. Reading the decision record aloud before people leave catches
misunderstandings while they can still be corrected, and it takes two minutes.

### Step 9. Record the decision

**Input:** the meeting outcome.
**Output:** a decision log entry.
**Rationale:** organisational memory is short and staff turnover is high. A decision without a
record is re-litigated by the next person who disagrees with it.

The log entry:

| Field | Content |
|---|---|
| Decision ID | Sequential identifier |
| Date | Decision date |
| Question | The framed question |
| Decision | What was decided, in one sentence |
| Decider | Named person |
| Rationale | Two to four sentences |
| Alternatives rejected | With the reason |
| Dissent | Named, with the substance |
| Owner | Person accountable for execution |
| Review date | When the decision will be revisited |
| Reversal conditions | What would trigger reopening |

The dissent field is the one organisations omit and the one that protects them. A recorded dissent
is not disloyalty; it is the evidence that the decision was genuinely considered.

### Step 10. Assign owners and dates, not actions

**Input:** the decision.
**Output:** named owners with dated commitments.
**Rationale:** an action without an owner is a wish, and an owner without a date is an intention.

Each commitment should have one owner, one deliverable, and one date. If a commitment has two
owners, it has none.

### Step 11. Communicate the decision to the people it affects

**Input:** the record.
**Output:** a short message to the informed group, and a briefing for the people whose work
changes.
**Rationale:** decisions leak, and the version that leaks is the rumour. A brief, honest
communication with the rationale prevents the rumour from becoming the record.

The message contains: what was decided, why, what happens next, who is affected, and when the
next update will be. It does not contain the full deliberation.

The hardest communication is the one where the decision went against the group's recommendation.
Three rules apply. Say it plainly in the first sentence, without burying it under context. Give
the actual reason rather than a procedural one; "the budget was not available" is a reason, and
"the committee felt" is not. And say what would change the decision, so that the people who
disagree know the decision is not simply final because someone had authority.

A decision communicated badly is re-litigated informally for months, and the re-litigation costs
more than the original process.

### Step 12. Audit the decision at the review date

**Input:** the record and the outcomes.
**Output:** a judgement about decision quality, separate from outcome quality.
**Rationale:** the audit is how the organisation learns. Without it, the same decision is made the
same way by the next team.

The audit asks three questions:

1. Was the process followed (rights, evidence, record)?
2. Was the reasoning sound given what was knowable at the time?
3. What would we do differently with the same information?

It explicitly does not ask whether the outcome was good, except to note whether the outcome was
within the range the memo anticipated.

The audit produces a one-page note with four parts. What was decided and why, in two sentences.
Whether the process was followed, with the evidence. Whether the reasoning held up against what was
knowable at the time. And one process change for the next decision of the same type. The fourth
part is the only one that changes future behaviour, and it should be specific enough to implement
without a further meeting.

An audit that concludes "the decision was reasonable and no process change is needed" is a valid
result, but it should be rare. Most decisions reveal at least one fixable defect in the calendar,
the evidence bar, or the record.

### 5.13 Branch points and edge cases

**No reimbursement pathway.** A decision to build a product with no payment route is effectively
irreversible, because the sunk cost cannot be recovered by revenue. Classify it as type 1 even if
the spend is small.

**Single-payer versus multi-payer.** In a single-payer system, one national decision can determine
access, so the decision rights sit with a body the team cannot influence directly. The leadership
task becomes building evidence for a national assessment rather than persuading a local buyer. In
a multi-payer system, decisions are distributed, and the team must decide which payers to pursue
rather than whether to pursue.

**Capital versus operating budget.** A capital decision is irreversible within its depreciation
period and requires a formal approval. An operating decision is reversible at renewal. The
decision process should match: a capital request needs the full memo and pre-mortem; an operating
purchase may need only a one-page case.

**Pilot-to-contract transition.** The decision to convert a pilot is often made by a different
committee from the one that approved the pilot. Name the conversion decider before the pilot
starts, or the pilot will end without a decision.

**Procurement security review.** A security veto is a legitimate decision right and should be
named as such. A team that treats security as a consulting function rather than a decider will be
surprised at the end.

**Champion turnover.** When a sponsor leaves, the decision may be reopened by their successor.
Record the rationale in a way that a new sponsor can evaluate, rather than in a way that depends
on the old sponsor's authority.

**Clinical safety.** In a clinical decision, a safety concern can override an economic case. Name
the person with the safety veto and record their view. A decision record that omits the safety
opinion is incomplete regardless of the financial analysis.

### 5.14 Decision table for choosing the mechanism

| Reversibility | Disagreement | Time available | Mechanism |
|---|---|---|---|
| Reversible | Low | Days | Decider decides, logs it, informs |
| Reversible | High | Weeks | Small group, written objections, decide |
| Irreversible | Low | Weeks | Memo, one independent review, decide |
| Irreversible | High | Months | Full memo, pre-mortem, consultation, senior approval |
| Irreversible | High | Days | Escalate to the highest authority; record the compression |
| Any | Any | Hours | Emergency protocol with a named decider and a mandatory retrospective |
| Any | Any | None | The decision has already been made elsewhere; find out where |

The fifth and sixth rows are the ones that produce the worst records. A compressed decision is
sometimes necessary; an unrecorded one is not.

## 6. Worked example

> **Note:** the organisation and the projects below are a composite constructed for teaching. All
> financial figures are illustrative 2026 USD. No real portfolio decision is described.

### 6.1 The decision

A health system's data and analytics group has 1,400 engineer-days of capacity in the January
quarter. Three projects are competing for it, and the chief data officer has asked for a decision
by 30 November.

The question: "We will decide by 30 November which projects to fund in the January quarter, given
1,400 engineer-days of capacity."

The decision is classified as type 2 with high consequence: the projects can be stopped, but the
quarter's capacity cannot be recovered, so the choice is effectively irreversible within the
quarter.

### 6.2 The options

| ID | Project | Cost (USD) | Expected benefit (USD) | Probability of success | Engineer-days |
|---|---|---:|---:|---:|---:|
| A | Sepsis model upgrade | 620,000 | 1,900,000 | 0.45 | 900 |
| B | Ambient documentation expansion | 410,000 | 1,300,000 | 0.70 | 500 |
| C | Denial prevention analytics | 350,000 | 2,400,000 | 0.30 | 700 |

Expected value is benefit times probability minus cost:

```text
EV(A) = 1,900,000 x 0.45 - 620,000 = 235,000 USD
EV(B) = 1,300,000 x 0.70 - 410,000 = 500,000 USD
EV(C) = 2,400,000 x 0.30 - 350,000 = 370,000 USD
```

Expected value per engineer-day is 261 USD for A, 1,000 USD for B, and 529 USD for C.

### 6.3 The portfolio

The naive approach is to rank by expected value per engineer-day and fund B, then C, then A until
capacity runs out. That is a greedy heuristic, and it is wrong here.

| Combination | Days | Expected value (USD) | Feasible at 1,400 days |
|---|---:|---:|---|
| None | 0 | 0 | Yes |
| A | 900 | 235,000 | Yes |
| B | 500 | 500,000 | Yes |
| C | 700 | 370,000 | Yes |
| A + B | 1,400 | 735,000 | Yes |
| A + C | 1,600 | 605,000 | No |
| B + C | 1,200 | 870,000 | Yes |
| A + B + C | 2,100 | 1,105,000 | No |

The optimal feasible portfolio is B plus C, at 1,200 days and 870,000 USD of expected value. The
ratio-greedy rule picks B then A, at 1,400 days and 735,000 USD, which is 135,000 USD worse.

The reason is that A consumes 900 days for 235,000 USD of expected value, and C's 700 days produce
370,000 USD. Greedy ranking by ratio does not account for the capacity that remains unusable after
a large item is chosen.

This is a general lesson about portfolio decisions: when capacity is the constraint, the ranking
that matters is the portfolio's total, not the individual project's ratio.

### 6.4 The cost of delay

If the decision slips by one month, the quarter's capacity falls to two-thirds, or 933
engineer-days. Recomputing the optimal portfolio:

| Capacity | Optimal portfolio | Days | Expected value (USD) |
|---:|---|---:|---:|
| 1,400 | B + C | 1,200 | 870,000 |
| 933 | B | 500 | 500,000 |

The expected value forgone is 370,000 USD. That number is the price of indecision, and it is the
justification for setting a decision deadline and for treating a deferral as a decision with a
cost rather than as a neutral act.

A leader who presents the decision without the delay cost invites deferral. A leader who presents
it with the delay cost makes deferral a choice that must be defended.

### 6.5 Two diagnostics that reveal a weak decision process

**Diagnostic 1: the talk-time concentration.** In the review meeting, five people spoke: the chief
medical officer for 14 minutes, the data lead for 6, the nursing director for 5, the chief
financial officer for 3, and the product manager for 2. The top speaker accounted for 46.7 percent
of the talk time.

A single speaker above roughly 40 percent is a signal that the room is being anchored. The
decision may still be correct, but it is not a group decision, and the dissent that would have
improved it was not voiced. The mitigation is a speaking order from least to most senior and a
written objection round before the meeting.

**Diagnostic 2: the decision log.** Four decisions were logged in the quarter:

| Decision | Raised | Decided | Outcome | Reopened |
|---|---|---|---:|---|
| Model upgrade | 12 Jan | 19 Jan | Approved | No |
| Vendor selection | 19 Jan | 16 Feb | Deferred | Yes |
| Pilot site | 2 Feb | 5 Feb | Approved | No |
| Budget reallocation | 9 Feb | 9 Mar | Deferred | Yes |

Median decision latency is 18 days and the maximum is 28 days, the deferral rate is 50 percent, and
the rework rate is 50 percent. A deferral rate above about 30 percent, or a rework rate above
about 20 percent, indicates that decisions are being brought to the forum before they are ready,
or that the decider is not in the room.

The second diagnosis is cheap and unglamorous, and it is the one that changes behaviour, because it
makes deferral visible as a pattern rather than as a series of individual judgements.

### 6.6 Varying one assumption

The recommendation depends on the probability estimates, which are the least reliable inputs. Vary
the probability of success for project C, holding everything else constant.

| Probability for C | EV(C) (USD) | Optimal portfolio | Portfolio EV (USD) | Change from base |
|---:|---:|---|---:|---:|
| 0.20 | 130,000 | A + B | 735,000 | -135,000 |
| 0.30 | 370,000 | B + C | 870,000 | 0 |
| 0.40 | 610,000 | B + C | 1,110,000 | +240,000 |
| 0.50 | 850,000 | B + C | 1,350,000 | +480,000 |
| 0.60 | 1,090,000 | B + C | 1,590,000 | +720,000 |
| 0.70 | 1,330,000 | B + C | 1,830,000 | +960,000 |

The portfolio changes at a probability below about 0.27, where A plus B becomes better than B plus
C. The decision is therefore robust across most of the plausible range, which is a useful thing to
be able to say in the memo: the recommendation does not depend on a precise estimate of C's
success probability, only on it exceeding roughly one in four.

That framing is more persuasive than the point estimate, because it converts the argument from
"is the estimate right" to "is the estimate above a threshold."

### 6.7 The memo

The two-page memo submitted to the decision meeting:

> **Decision requested.** Approve projects B and C for the January quarter, at 1,200 engineer-days
> and 760,000 USD of direct cost, and defer project A to the April quarter.
>
> **Context.** Capacity is fixed at 1,400 engineer-days. Three proposals are ready. The April
> quarter has no committed capacity, so a deferral is a real deferral and not a queue position.
>
> **Options.** A alone; B alone; C alone; A plus B; B plus C; do nothing. A plus C and all three
> exceed capacity.
>
> **Criteria.** Expected value, strategic fit with the FY2026 quality strategy, reversibility, and
> the concentration of risk in a single project.
>
> **Analysis.** B plus C has the highest expected value of any feasible combination at 870,000 USD,
> 135,000 USD above the next best, A plus B. The result is robust to the probability estimate for C
> down to about 0.27.
>
> **Recommendation.** B plus C. B has the highest expected value per engineer-day and the lowest
> execution risk. C has the largest benefit and the highest uncertainty, and pairing it with B
> gives the portfolio a hedge.
>
> **Risks.** C's benefit depends on payer claims access that has not been confirmed; the mitigation
> is a two-week access check in the first sprint, with a fallback to A if access is refused. B's
> benefit depends on clinic adoption above 40 percent; the mitigation is the adoption gate from
> Topic 06.
>
> **Pre-mortem.** It is November 2026 and the portfolio failed. The three most likely reasons: C's
> claims access was denied in week three and the team did not invoke the fallback; B's adoption
> stalled at 25 percent because the champion left; and the combined portfolio overran capacity by
> 15 percent because the estimates excluded testing.
>
> **What would change our mind.** A claims access decision before 15 December, or evidence that
> B's adoption is below 25 percent at the pilot sites.
>
> **Decision record.** Decider: chief data officer. Date: 30 November 2026. Logged in the portfolio
> register. Review: 31 March 2027.

The memo is two pages because the decision is a type 2 decision with high consequence. A type 1
decision with the same content would run to six pages with a fuller pre-mortem and an independent
review.

### 6.8 The recommendation and the review

The decision meeting approved B plus C with one condition: the claims access check was moved to
the first week, with a named owner and a hard date. The dissent was recorded: the chief medical
officer preferred A plus B on clinical grounds, and the record states the reasoning.

At the review date, the audit found that the process was followed, the reasoning was sound, and the
outcome was within the anticipated range. C's claims access was granted in week two, B's adoption
reached 44 percent by month three, and the portfolio used 1,240 of 1,400 days. The audit noted one
process defect: the capacity estimate excluded testing, which the pre-mortem had identified, and
the team absorbed the gap rather than escalating it.

That defect is the most valuable output of the audit, because it is a fixable process change and
it will affect the next quarter's estimates.

## 7. Code, query, or calculation

The script computes the portfolio optimum by enumeration, the cost of delay, the sensitivity to
the uncertain probability, and the decision-log metrics.

```python
"""Portfolio decision under a capacity constraint, and decision-log metrics."""

from itertools import combinations
from datetime import date
from statistics import median

PROJECTS = {
    "A": dict(name="Sepsis model upgrade", cost=620_000, benefit=1_900_000,
              p=0.45, days=900),
    "B": dict(name="Ambient documentation expansion", cost=410_000, benefit=1_300_000,
              p=0.70, days=500),
    "C": dict(name="Denial prevention analytics", cost=350_000, benefit=2_400_000,
              p=0.30, days=700),
}
CAPACITY = 1_400


def ev(p):
    return p["benefit"] * p["p"] - p["cost"]


def best_portfolio(capacity, projects=PROJECTS):
    best = None
    keys = list(projects)
    for r in range(len(keys) + 1):
        for combo in combinations(keys, r):
            days = sum(projects[k]["days"] for k in combo)
            if days > capacity:
                continue
            value = sum(ev(projects[k]) for k in combo)
            if best is None or value > best[1]:
                best = (combo, value, days)
    return best


print("INDIVIDUAL EXPECTED VALUE")
print("  id  project                            cost     benefit    p       EV   EV/day")
for k, p in PROJECTS.items():
    print(f"  {k}   {p['name']:32s} ${p['cost']:8,.0f} ${p['benefit']:9,.0f}"
          f"  {p['p']:.2f}  ${ev(p):8,.0f}  ${ev(p) / p['days']:6,.0f}")

combo, value, days = best_portfolio(CAPACITY)
print(f"\nOPTIMAL AT {CAPACITY:,} DAYS: {'+'.join(combo)}"
      f" ({days:,} days, ${value:,.0f})")

print("\nFEASIBLE COMBINATIONS")
for r in range(len(PROJECTS) + 1):
    for c in combinations(PROJECTS, r):
        d = sum(PROJECTS[k]["days"] for k in c)
        v = sum(ev(PROJECTS[k]) for k in c)
        print(f"  {'+'.join(c) if c else 'none':8s} {d:5,d} days  ${v:10,.0f}"
              f"  {'feasible' if d <= CAPACITY else 'excluded'}")

reduced = int(CAPACITY * 2 / 3)
c2, v2, d2 = best_portfolio(reduced)
print(f"\nDELAY COST: capacity {reduced:,} days -> {'+'.join(c2)}"
      f", EV ${v2:,.0f}, forgone ${value - v2:,.0f}")

print("\nSENSITIVITY: probability for C")
print("  p(C)  EV(C)       optimal   EV          change")
for p in (0.20, 0.30, 0.40, 0.50, 0.60, 0.70):
    proj = {k: dict(v) for k, v in PROJECTS.items()}
    proj["C"]["p"] = p
    c3, v3, _ = best_portfolio(CAPACITY, proj)
    print(f"  {p:4.2f}  ${ev(proj['C']):9,.0f}  {'+'.join(c3):8s}  ${v3:10,.0f}"
          f"  ${v3 - value:+,.0f}")

print("\nRATIO-GREEDY VERSUS PORTFOLIO")
order = sorted(PROJECTS, key=lambda k: -ev(PROJECTS[k]) / PROJECTS[k]["days"])
picked, used = [], 0
for k in order:
    if used + PROJECTS[k]["days"] <= CAPACITY:
        picked.append(k)
        used += PROJECTS[k]["days"]
greedy = sum(ev(PROJECTS[k]) for k in picked)
print(f"  ratio order           {'>'.join(order)}")
print(f"  greedy pick           {'+'.join(picked)} ({used:,} days, ${greedy:,.0f})")
print(f"  optimal               {'+'.join(combo)} ({days:,} days, ${value:,.0f})")
print(f"  greedy shortfall      ${value - greedy:,.0f}")

print("\nTALK-TIME CONCENTRATION")
speakers = {"CMO": 14, "CFO": 3, "Data lead": 6, "Nursing director": 5, "Product": 2}
total = sum(speakers.values())
for name, mins in sorted(speakers.items(), key=lambda kv: -kv[1]):
    print(f"  {name:18s} {mins:3d} min  {mins / total:6.1%}")
print(f"  top share {max(speakers.values()) / total:.1%}")

print("\nDECISION LOG")
rows = [("model upgrade", "2026-01-12", "2026-01-19", "approved", False),
        ("vendor selection", "2026-01-19", "2026-02-16", "deferred", True),
        ("pilot site", "2026-02-02", "2026-02-05", "approved", False),
        ("budget reallocation", "2026-02-09", "2026-03-09", "deferred", True)]
latency = [(date.fromisoformat(b) - date.fromisoformat(a)).days for _, a, b, _, _ in rows]
print(f"  decisions            {len(rows)}")
print(f"  median latency       {median(latency):.0f} days")
print(f"  max latency          {max(latency)} days")
print(f"  deferral rate        {sum(1 for r in rows if r[3] == 'deferred') / len(rows):.0%}")
print(f"  rework rate          {sum(1 for r in rows if r[4]) / len(rows):.0%}")
```

Expected output:

```text
INDIVIDUAL EXPECTED VALUE
  id  project                            cost     benefit    p       EV   EV/day
  A   Sepsis model upgrade           $ 620,000 $1,900,000  0.45  $ 235,000  $   261
  B   Ambient documentation expansion $ 410,000 $1,300,000  0.70  $ 500,000  $ 1,000
  C   Denial prevention analytics    $ 350,000 $2,400,000  0.30  $ 370,000  $   529

OPTIMAL AT 1,400 DAYS: B+C (1,200 days, $870,000)

FEASIBLE COMBINATIONS
  none         0 days  $         0  feasible
  A          900 days  $   235,000  feasible
  B          500 days  $   500,000  feasible
  C          700 days  $   370,000  feasible
  A+B      1,400 days  $   735,000  feasible
  A+C      1,600 days  $   605,000  excluded
  B+C      1,200 days  $   870,000  feasible
  A+B+C    2,100 days  $ 1,105,000  excluded

DELAY COST: capacity 933 days -> B, EV $500,000, forgone $370,000

SENSITIVITY: probability for C
  p(C)  EV(C)       optimal   EV          change
  0.20  $  130,000  A+B       $   735,000  $-135,000
  0.30  $  370,000  B+C       $   870,000  $+0
  0.40  $  610,000  B+C       $ 1,110,000  $+240,000
  0.50  $  850,000  B+C       $ 1,350,000  $+480,000
  0.60  $1,090,000  B+C       $ 1,590,000  $+720,000
  0.70  $1,330,000  B+C       $ 1,830,000  $+960,000

RATIO-GREEDY VERSUS PORTFOLIO
  ratio order           B>C>A
  greedy pick           B+A (1,400 days, $735,000)
  optimal               B+C (1,200 days, $870,000)
  greedy shortfall      $135,000

TALK-TIME CONCENTRATION
  CMO                 14 min   46.7%
  Data lead            6 min   20.0%
  Nursing director     5 min   16.7%
  CFO                  3 min   10.0%
  Product              2 min    6.7%
  top share 46.7%

DECISION LOG
  decisions            4
  median latency       18 days
  max latency          28 days
  deferral rate        50%
  rework rate          50%
```

Two notes on the method. Enumerating all subsets is feasible only for a small portfolio; at twenty
projects it requires an integer program or a heuristic. For the quarterly decision that a data
team actually faces, the portfolio is small and enumeration is exact and transparent.

The probability estimates are subjective. Their value is not precision but comparability: once
every option is expressed on the same scale, the conversation moves from preference to evidence.

## 8. Data, standards, and regulatory context

### 8.1 Governance systems that require decision records

Several regimes require that decisions be documented, and they are useful models even where they
do not apply.

| Regime | Requirement | Implication |
|---|---|---|
| FDA Quality Management System Regulation | Design controls include design reviews with documented results | Decisions about device design need records |
| IEC 62304 | Software development planning and problem resolution records | Change decisions need traceability |
| ISO 13485 | Documented procedures and records for product realisation | Decision records are quality records |
| Model risk management guidance | Model validation, approval, and ongoing monitoring records | Model decisions need independent review |
| Sarbanes-Oxley | Internal control over financial reporting | Financial decisions need documented controls |
| NHS governance and the Francis report | Board accountability and candour | Leadership decisions must be visible and challengeable |

The most directly relevant for a data team is model risk management. Supervisory guidance on model
risk management, such as the Federal Reserve's SR 11-7, sets out a three-part structure:
development, validation, and governance. Validation must be performed by someone independent of
development, and the governance function must approve the model, track its performance, and define
the conditions under which it is retired. That is a decision-rights system with a record, and it
maps closely onto the decision log in Section 5.9.

### 8.2 Clinical governance

In a health system, decisions that affect patient care pass through clinical governance structures:
the medical executive committee, the quality and safety committee, and the pharmacy and
therapeutics committee. A data product that changes clinical workflow usually requires one of
these bodies to approve it.

These committees have their own decision rights, their own quorum rules, and their own records.
A team that presents to the wrong committee will be redirected, and the redirection can cost a
quarter. The leadership task is to identify the correct committee and its evidence expectations
before writing the memo.

### 8.3 Transparency and candour

The Francis Inquiry into Mid Staffordshire NHS Foundation Trust found that the trust board failed
to listen to patients and staff and tolerated a culture of poor standards. Its recommendations
emphasised openness, candour, and the accountability of directors. The report is a case study in
what happens when the decision system has no channel for dissent and no record of concerns.

For a data team, the practical implication is that a decision record which suppresses dissent is
not only incomplete but unsafe. The dissent field exists so that a later reviewer can see what was
known and what was overruled.

### 8.4 Automated decisions about individuals

If the product makes or informs decisions about identifiable people, additional rules apply. In
the European Union, Article 22 of the General Data Protection Regulation restricts decisions based
solely on automated processing that produce legal or similarly significant effects. In the United
States, there is no general federal equivalent, but state laws and sector rules apply, and the
Office of the National Coordinator's HTI-1 rule requires transparency for predictive decision
support in certified health information technology.

The leadership implication: a decision about whether to deploy an automated recommendation is a
decision about governance, not only about accuracy. The memo should state who is accountable for
the automated decision, what human review exists, and how a patient can contest it.

### 8.5 What the standards do not settle

None of these frameworks tells you how to make a decision under uncertainty with incomplete
evidence and a deadline. They require records, not judgement. The judgement is what this topic is
about, and it is learned by making decisions and auditing them.

## 9. Industry practice and case evidence

### 9.1 Documented: narrative memos as a decision mechanism

Amazon's practice of replacing slide presentations with narrative memos is described in its 2017
shareholder letter. Meetings begin with a silent reading period of a six-page memo, and the letter
notes that the main failure mode is not that writers cannot recognise a high standard but that
they underestimate the time required, which "might take a week or more." Memos are rewritten,
shared for critique, set aside, and edited again.
The practice has three properties relevant here. It forces the author to make the reasoning
explicit. It equalises information before discussion, because everyone reads the same document. And
it removes the advantage of the charismatic presenter, which changes whose analysis prevails.

A related practice, described by former executives, is the page limit tied to meeting length: six
pages for an hour, three for thirty minutes, with an appendix that is not required reading. The
constraint is a forcing function for prioritisation.

### 9.2 Documented: reversibility and decision speed

Amazon's 2016 shareholder letter distinguishes one-way-door decisions, which are irreversible and
should be made slowly, from two-way-door decisions, which are reversible and should be made
quickly by high-judgement individuals or small groups. It also describes deciding at about 70
percent of the information you wish you had, and "disagree and commit" as a mechanism for
proceeding after disagreement is recorded.

The classification in Section 3.2 is a restatement of that distinction, with the addition of
consequence as a second axis. The 70 percent rule is a useful antidote to the analysis paralysis
that afflicts technical teams, but it applies only to reversible decisions; applying it to an
irreversible clinical decision is a serious error.

The practical translation for a data team: most requests for more analysis are requests to avoid
a reversible decision. A team that asks "what would we do differently with this information, and
what does the delay cost" will find that the answer is usually "nothing" and "more than the
information is worth."

### 9.3 Documented: the cost of a failed leadership system

The Francis Inquiry into Mid Staffordshire NHS Foundation Trust, published in February 2013,
concluded that appalling care between 2005 and 2008 was primarily caused by a serious failure of
the trust board. The board failed to listen sufficiently to patients and staff and failed to tackle
an insidious negative culture involving tolerance of poor standards and disengagement from
managerial and leadership responsibilities. The report made 290 recommendations.

Two features of the findings are relevant to a data team. First, the failure was not a lack of
information; it was a failure to act on information that existed. Second, the culture rewarded
positive information and discouraged challenge, which is exactly the condition in which a decision
record with a dissent field would have been informative.

The inquiry is not a story about bad individuals. It is a story about a system in which decision
rights were unclear, challenge was unwelcome, and no record captured the concerns that staff
raised.

### 9.4 Documented: psychological safety and team performance

Google's Project Aristotle, reported in 2016, studied 180 teams and identified psychological
safety as the most important of five factors distinguishing effective teams, ahead of dependability,
structure and clarity, meaning, and impact. Psychological safety is the belief that one can speak
up with a question, a concern, or a mistake without being punished.

The relevance to this topic is direct: a decision process that suppresses dissent will produce
records that look unanimous and decisions that are worse. The dissent field, the written objection
round, and the speaking order are mechanisms for producing psychological safety at the level of a
single decision, rather than relying on the general culture.

One caution: the finding is about team learning behaviour, not about comfort. A psychologically
safe team is one where disagreement is expressed, not one where it is avoided.

### 9.5 Inference: what this implies for technical leaders

The following is inference from the sources, not a documented finding.

First, the highest-leverage leadership intervention for a data team is usually the decision
calendar, not the analysis. Knowing which decisions are coming, who decides, and what evidence
they need converts reactive analysis into scheduled work.

Second, a written record changes behaviour more than a meeting does. Teams that know their
rationale will be logged write better rationales.

Third, the cost of delay should be computed and shown. It is the most effective argument against
deferral, and it converts a political preference for caution into a quantified trade-off.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| No named decider | "The committee will decide" | Diffuse authority | Decision by attrition | Name one decider |
| Decision after the deadline | Budget set before the review | Calendar mismatch | Work arrives too late | Map the decision calendar |
| Analysis without a question | Deck starts with methods | Framing skipped | Unusable output | Write the question first |
| Options missing | Memo presents one option | Advocacy disguised as analysis | No real choice | Require three options including do-nothing |
| Evidence bar undefined | "We need more data" repeated | No stopping rule | Indefinite deferral | Name required and not-required evidence |
| Deferral not costed | Deferral treated as neutral | Cost of delay invisible | Value forgone silently | Compute delay cost |
| Talk-time concentration | One speaker above 40 percent | Anchoring by seniority | Group decision in name only | Speaking order, written objections |
| Dissent suppressed | Record shows unanimous approval | Conflict avoidance | Blind spots persist | Dissent field, safety veto |
| No record | Decisions recalled differently | Organisational memory loss | Re-litigation | Decision log |
| Action without owner | "The team will" | Diffuse accountability | Nothing happens | One owner and one date |
| Outcome judged as quality | "It worked, so it was right" | Outcome bias | Wrong lessons learned | Audit process and reasoning |
| Type 1 treated as type 2 | Reversal cost not estimated | Speed bias | Costly reversal or harm | Estimate reversal cost |

Two red flags are diagnostic. The first is a decision memo with one option; that is a proposal, not
a decision document. The second is a decision log with no dissent entries over a long period,
which usually means dissent is not being recorded rather than that it does not exist.

A third pattern is worth naming because it is common in technical organisations: the decision
that is made twice. A committee approves a direction, and a working group re-decides it in
practice because the approval did not specify the constraints. The remedy is to write the
constraints into the record, not only the direction, so that the working group has less room to
reinterpret. "Approved: vendor B, subject to a 400,000 USD annual cap and a 16-week implementation
window" is a decision. "Approved in principle" is an invitation to renegotiate.

## 11. Assumptions and limits

| # | Assumption | Why it is needed | What breaks it | How to test |
|---:|---|---|---|---|
| 1 | The decision can be framed in advance | The process requires a question | Emergent decisions | Keep a rapid-response path |
| 2 | One person can be named decider | Singular authority is required | Joint sign-off in matrix organisations | Name a tie-break rule |
| 3 | Evidence can be assembled in time | The memo depends on it | Data access delay | Set the evidence bar early |
| 4 | Probabilities are estimable | Portfolio EV needs them | No base rate | Use ranges and thresholds |
| 5 | Capacity is the binding constraint | Portfolio model assumes it | Money or approvals bind | Test alternative constraints |
| 6 | Participants will object in writing | Written objections improve quality | Culture suppresses dissent | Anonymise the first round |
| 7 | The meeting can decide | Forum has authority | Decision is made elsewhere | Confirm rights before writing |
| 8 | The record will be read later | The log's value depends on it | No audit practice | Schedule the audit |
| 9 | Outcomes can be observed at the review date | Audit needs evidence | Long payoff periods | Choose an intermediate measure |
| 10 | The decision is worth the process | Process has a cost | Small reversible decisions | Use the class table |

The deepest limit is that decision quality is judged retrospectively and often by people who know
the outcome. A leader who insists on process will sometimes be told that the process is
bureaucratic, usually by someone whose preferred option lost. The defence is not to argue about
bureaucracy but to show the decisions the process changed and the ones it would have changed.

A second limit is that this procedure assumes a decision that can be scheduled. Many important
decisions arrive as a crisis, with hours rather than weeks. The emergency protocol in Section 5.14
is the fallback, and its most important feature is the mandatory retrospective: a decision made
under compression should be reviewed afterwards, when the compression has lifted, by people who
were not in the room.

That retrospective is also the only reliable way to learn which decisions genuinely required
compression and which were compressed because no one had planned the calendar.

## 12. Exercises

### Exercise 1. Decision classification

**Deliverable:** classify six described decisions by reversibility and consequence, and state the
process and timeline for each.
**Criteria:** each classification cites the cost of reversal; at least one irreversible decision is
correctly identified despite a small spend. **Time:** 30 minutes.

### Exercise 2. Decision rights

**Deliverable:** a RAPID or DACI table for a specified decision, with names or roles for each
letter.
**Criteria:** exactly one decider; consultees have deadlines; the informed group is not confused
with consultees. **Time:** 30 minutes.

### Exercise 3. Option framing

**Deliverable:** rewrite a one-option proposal as a four-option decision including do-nothing.
**Criteria:** each option has cost, benefit, risk, and reversibility; do-nothing is analysed, not
dismissed. **Time:** 45 minutes.

### Exercise 4. Decision memo

**Deliverable:** a two-page memo in the ten-section structure from Step 5.
**Criteria:** all ten sections present; at least three options; a pre-mortem with three failure
modes; a "what would change our mind" section. **Time:** 2 hours.

### Exercise 5. Review simulation

**Deliverable:** run a 30-minute simulated review with five participants and produce a decision
record.
**Criteria:** silent read, speaking order, written objections, decision recorded in the room, dissent
captured. **Time:** 90 minutes.

### Exercise 6. Decision log

**Deliverable:** a decision log with ten fields for five past decisions, reconstructed from memory
and documents.
**Criteria:** rationale and alternatives present for each; at least one dissent recorded; review
dates set. **Time:** 60 minutes.

### Exercise 7. Cost of delay

**Deliverable:** for a decision with a 40,000 USD monthly benefit and a two-month delay, compute the
forgone value at a 10 percent annual discount rate and state the decision deadline implication.
**Criteria:** correct discounting; the deadline stated; the assumption about when benefit starts
made explicit. **Time:** 30 minutes.

### Exercise 8. Decision audit

**Deliverable:** audit a decision made at least six months ago against the information available
then, using the three questions in Step 12.
**Criteria:** process, reasoning, and lessons separated; outcome explicitly excluded from the
quality judgement; one process change proposed. **Time:** 60 minutes.

## 13. Self-check answers

**1. What makes a decision type 1 or type 2?**
Type 1 is irreversible or very costly to reverse; type 2 is reversible at low cost. The
classification determines the process: type 1 warrants a memo, consultation, and a pre-mortem;
type 2 should be decided quickly and logged.

**2. Why must there be exactly one decider?**
Because shared authority produces either indecision or decision by attrition. Consultation can be
broad, but accountability must be singular, or no one is answerable for the result.

**3. Why include a do-nothing option?**
Because doing nothing is usually the real alternative, and analysing it forces the team to state
the cost of inaction. A memo that omits it compares options against an implied baseline that no one
has examined.

**4. What are the ten sections of the decision memo?**
Decision requested, context, options, criteria, analysis, recommendation, risks and mitigations,
pre-mortem, what would change our mind, and decision record.

**5. What is the purpose of the pre-mortem?**
To surface failure modes that optimism suppresses, by asking participants to imagine the decision
has already failed and to explain why. It produces risks with early warning signs and mitigations.

**6. How do you run a review so the decision happens in the room?**
Silent read at the start, speaking order from least to most senior, objections before the
recommendation, the decider speaks last and states the decision, and the record is written before
the meeting ends.

**7. What fields belong in a decision record?**
Decision ID, date, question, decision, decider, rationale, alternatives rejected, dissent, owner,
review date, and reversal conditions.

**8. Compute the Exercise 7 answer.** Monthly benefit 40,000 USD. Two months of delay forgoes
80,000 USD undiscounted. Discounted at 10 percent annually, the present value of the forgone
benefit is approximately 40,000/1.1^(1/12) + 40,000/1.1^(2/12) = 39,684 + 39,371 = 79,055 USD,
assuming the benefit would have begun immediately. The deadline implication is that the decision
should be made before the next monthly benefit cycle, not at the next quarterly meeting.

**9. What is the difference between decision quality and outcome quality?**
Decision quality is whether the process and reasoning were sound given what was knowable. Outcome
quality is whether the result was good. They diverge because of luck, and conflating them teaches
the wrong lessons.

**10. Why does talk-time concentration matter?**
Because a single speaker above roughly 40 percent of the time anchors the room, suppresses
dissent, and converts a group decision into a ratification of one view.

**11. What is "disagree and commit"?**
Recording a disagreement, then executing the decision as if you supported it. It allows a decision
to close without requiring consensus, while preserving the dissent in the record.

**12. Rubric for Exercise 5.** Full marks require a silent read, a stated speaking order, at least
two written objections addressed, a decision recorded in the room with a named decider and date,
and a dissent entry. Half marks for a discussion that reaches a conclusion without a record.

## 14. Interview questions

**1. How do you get a decision made in a large organisation?**
Strong: names the decider, frames options, sets a deadline, supplies proportionate evidence, and
records the outcome. Weak: says they build consensus.

**2. How do you influence people who do not report to you?**
Strong: describes an influence ledger of evidence, reliability, and relationships, with a concrete
example. Weak: says they communicate clearly.

**3. How do you decide how much evidence is enough?**
Strong: ties the evidence bar to reversibility and consequence, and names what evidence is not
required. Weak: says they gather as much as possible.

**4. What goes in a decision memo?**
Strong: lists the ten sections and emphasises options, pre-mortem, and what would change their
mind. Weak: describes a business case with a recommendation.

**5. How do you handle a decision that is above your authority?**
Strong: prepares the recommendation, identifies the decider, supplies the memo, and supports the
meeting without pretending to decide. Weak: says they escalate.

**6. How do you run a meeting that produces a decision?**
Strong: silent read, speaking order, objections first, decider last, record in the room. Weak: says
they set an agenda and keep time.

**7. What do you do when the group cannot agree?**
Strong: distinguishes disagreement about facts from disagreement about values, resolves the first
with evidence, and escalates or records the second with disagree-and-commit. Weak: says they take a
vote.

**8. How do you handle a decision that turns out badly?**
Strong: separates process from outcome, runs a blameless review, and changes the process rather
than the person. Weak: says they take responsibility and move on.

**9. Why record dissent?**
Strong: because it preserves the information the decision overruled, protects the organisation
later, and improves the quality of the discussion. Weak: says it is good documentation.

**10. How do you audit a decision?**
Strong: compares the decision to the information available at the time, assesses process and
reasoning, and explicitly excludes the outcome from the quality judgement. Weak: reviews whether it
worked.

**11. How do you set a decision deadline?**
Strong: computes the cost of delay, works back from the last responsible moment, and states the
consequence of missing it. Weak: picks a date that feels reasonable.

**12. What is the difference between a sponsor and a decider?**
Strong: the sponsor funds and defends; the decider has authority to choose. They can be the same
person, but assuming they are is a common cause of stalled decisions.

## 15. Key terms

- **After-action review** — structured review of what happened and why.
- **Blameless review** — review focused on system causes.
- **Candour** — duty to be open about failures.
- **Champion** — advocate who builds support.
- **Consulted** — role whose input must be sought.
- **Decider** — the single person accountable for a decision.
- **Decision audit** — review against contemporaneous evidence.
- **Decision latency** — time from readiness to decision.
- **Decision log** — record of decisions and rationale.
- **Decision quality** — reasonableness given available evidence.
- **Decision rights** — who has authority to decide.
- **Decision velocity** — decisions per period at a given quality.
- **DACI** — driver, approver, contributor, informed.
- **Disagree and commit** — record disagreement, then execute.
- **Dissent** — recorded disagreement.
- **Do-nothing option** — the baseline of not deciding.
- **Escalation** — raising a decision to higher authority.
- **Expected value** — probability-weighted outcome.
- **Governance** — system of rights, forums, and records.
- **Influence ledger** — balance of credibility and favours.
- **Informed** — told after the decision.
- **Option value** — value of keeping a choice open.
- **Portfolio** — set of projects competing for capacity.
- **Pre-mortem** — prospective hindsight on failure.
- **Psychological safety** — belief that speaking up is safe.
- **RACI** — responsible, accountable, consulted, informed.
- **RAPID** — recommend, agree, perform, input, decide.
- **Red team** — independent adversarial review.
- **Sponsor** — senior person who funds and defends.
- **Stakeholder** — anyone affected by a decision.
- **Type 1 decision** — irreversible, high consequence.
- **Type 2 decision** — reversible, low consequence.

## 16. Sources

All URLs accessed 26 October 2026.

1. Amazon. *2017 Letter to Shareholders* (six-page narratives).
   https://www.aboutamazon.com/company-news/2017-letter-to-shareholders
2. Amazon. *2016 Letter to Shareholders* (high-velocity decision making, type 1 and type 2
   decisions, disagree and commit).
   https://www.aboutamazon.com/news/company-news/2016-letter-to-shareholders
3. Bryar C, Carr B. *Working Backwards: Insights, Stories, and Secrets from Inside Amazon*.
   St. Martin's Press, 2021. See also https://workingbackwards.com/concepts/narratives-decision-making/
4. Mid Staffordshire NHS Foundation Trust Public Inquiry. *Report of the Mid Staffordshire NHS
   Foundation Trust Public Inquiry* (Francis Report), HC 947, February 2013.
   https://assets.publishing.service.gov.uk/government/uploads/system/uploads/attachment_data/file/279124/0947.pdf
5. Mid Staffordshire NHS Foundation Trust Public Inquiry. *Volume 1: Analysis of Evidence and
   Lessons Learned*, HC 898-I, 2013.
   https://assets.publishing.service.gov.uk/media/5a7c9bec40f0b65b3de09fde/0898_i.pdf
6. Google re:Work. *Guide: Understand Team Effectiveness* (Project Aristotle).
   https://rework.withgoogle.com/print/guides/5721312655835136/
7. Board of Governors of the Federal Reserve System. *Supervisory Guidance on Model Risk
   Management*, SR Letter 11-7, 2011.
   https://www.federalreserve.gov/supervisionreg/srletters/sr1107.htm
8. US Food and Drug Administration. *Quality Management System Regulation*, 21 CFR Part 820.
   https://www.fda.gov/medical-devices/postmarket-requirements-devices/quality-management-system-regulation-qmsr
9. International Electrotechnical Commission. *IEC 62304: Medical Device Software — Software Life
   Cycle Processes*.
   https://webstore.iec.ch/publication/6793
10. European Parliament and Council. *General Data Protection Regulation*, Article 22.
    https://gdpr-info.eu/art-22-gdpr/
11. Office of the National Coordinator for Health Information Technology. *HTI-1 Final Rule*,
    2024. https://www.healthit.gov/topic/laws-regulation-and-policy/health-data-technology-and-interoperability-certification-program
12. National Health Service. *Learning from Deaths* and board governance guidance, NHS England.
    https://www.england.nhs.uk/patient-safety/learning-from-deaths/

Figures in Section 6 are illustrative and constructed for teaching. The Francis Report findings
are summarised from the published executive summary and volume 1.

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`07_business-models.md`](07_business-models.md)
- Next topic: [`09_roadmap-and-prioritisation.md`](09_roadmap-and-prioritisation.md)
- Related topic: [`02_value-quantification.md`](02_value-quantification.md)
- Module survey: [`../../course/17_product_economics_leadership.md`](../../course/17_product_economics_leadership.md)
- Labs: [`../../labs/`](../../labs/)
