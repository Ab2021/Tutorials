# Scoping Playbook

> **Module 19 — Capstone, Hackathons, and Competitions**
> Topic 08 of 11
> Estimated study time: 3.5 hours
> Prerequisites: Topics 01, 04, and 05
> Version 2026.3

## 1. Why this topic exists

An open problem arrives as a sentence: "emergency department crowding is getting worse", "we
should use AI for readmissions", "can you look at our sepsis data".
None of those sentences is a project.Between the sentence and a project there is a scoping decision, and the quality of that decision
determines whether the next twelve weeks produce evidence or activity.

Scoping is usually done badly because it is done implicitly.
A team hears the sentence, opens the data, and starts modelling. Three weeks later it discovers
that the outcome it predicted is not the outcome the department controls, that the data cover a
period before a major process change, or that the person who asked for the work has no authority
to act on it.
Each of those discoveries was available in the first two days, for the cost of asking.

The industry cost is well documented in the evaluation literature.
NHS England's guidance on real-world AI evaluations, drawing on the AI in Health and Care Award
portfolio, describes projects that stalled because the operational owner was not identified and
the evidence plan was written after deployment.
The Royal College of Radiologists' 2025 consultation document on AI deployment fundamentals notes
that marking is not evaluation and that independent evidence is scarce, which is partly a scoping
failure: the claim was never matched to the evidence that could be generated.

The academic cost is the capstone.
A team that spends two days on scoping has ten weeks for evidence.
A team that spends three weeks discovering its scope has seven, and seven is not enough for a
baseline, a candidate, a validation, and a dossier.

This topic compresses scoping into two working days and makes it produce a one-page memo that a
clinical owner, a data steward, and a programme lead can sign.
The memo is not a plan; the plan is Topic 02. The memo fixes the question, the decision, the
evidence bar, and the kill criterion, which are the four things a plan cannot fix later.

A scoping decision is also the cheapest point at which to say no.
Saying no in week 0 costs two days and a conversation. Saying no in week 8 costs the team's morale,
the sponsor's confidence, and the panel's time. The two-day format is designed so that the no
happens early, with a written reason, and with an alternative in the parking lot.

The alternative is not always another project. Sometimes it is a narrower claim on the same
problem, which is why the memo has a "claim" line separate from the "decision" line. A team can
keep the decision and reduce the claim, and that is often the best outcome of the two days.

The memo is also the artefact that makes a later "no" cheap. When the feasibility verdict changes in
week 6, the team can point to the kill criterion that was agreed in week 0, rather than negotiating
a new promise under pressure.

Two frameworks are used, and both are conventions rather than regulations.
The FINER criteria, from clinical research methods, ask whether a question is Feasible,
Interesting, Novel, Ethical, and Relevant.
PICO, from evidence-based medicine, structures a question as Population, Intervention, Comparator,
and Outcome.
Neither is binding, and both are useful because they force the missing element into view.

A third idea is less formal and more useful: the scope is a promise about what the team will be
able to defend in twelve weeks.
That promise has four parts: a decision someone owns, a comparator measured on the same cohort, an
evidence bar the data can meet, and an exclusion list that names what will not be attempted.
A scope that omits any part is a hope rather than a promise, and the two-day exercise exists to
convert one into the other.

The four parts are also the four questions a panel asks in the first five minutes of a defense.
Who acts, what is current practice, what evidence supports the claim, and what did you exclude.
A team whose memo answers all four has rehearsed the defense before building anything.

## 2. Learning outcomes

By the end of this topic you can:

1. **Convert** an open problem sentence into a decision question with a population, an action, a
   comparator, and an outcome.
2. **Run** a data feasibility triage in four hours, including access, coverage, label provenance,
   and time zero.
3. **Apply** knockout criteria that eliminate infeasible scopes before scoring the rest.
4. **Write** a kill criterion that is numeric, measurable, and owned.
5. **Produce** a one-page scope memo with the decision, the evidence bar, the exclusions, and the
   stop condition.
6. **Estimate** the two-day scoping budget in wall-clock hours and person-hours, and identify when
   the wall clock is the binding constraint.
7. **Recognise** the four scope failures that appear in week 3: wrong outcome, wrong population,
   wrong owner, and wrong comparator.
8. **Defend** a scope decision, including what was excluded and why, in a five-minute review.

Outcomes 1 to 4 are assessed in Exercises 1 to 4 and tested at the day-2 review.
Outcomes 5 and 6 are assessed in Exercises 5 to 7, which together produce the memo.
Outcomes 7 and 8 are assessed in Exercise 8 and in the G0 discussion, where the panel asks what was
excluded.

The mapping matters because the two days are assessed as a process, not only as a document. A team
that produces a good memo from a bad process will struggle to explain how it got there.

## 3. The core mental model

Scoping is a funnel with five gates and a fixed clock.
The clock is two working days, split into four half-days.
Each gate produces one artefact, and a gate that cannot be passed is a reason to stop or reframe,
not a reason to continue and hope.

```text
  DAY 1 MORNING            DAY 1 AFTERNOON
  +----------------+       +----------------------+
  | 1 PROBLEM      |       | 3 DATA TRIAGE        |
  |   who hurts,   |  -->  |   access, coverage,  |
  |   how often    |       |   labels, time zero  |
  +----------------+       +----------------------+
          |                          |
          v                          v
  +----------------+       +----------------------+
  | 2 DECISION     |       | 4 FEASIBILITY        |
  |   action,owner,|  -->  |   knockout checks,   |
  |   comparator   |       |   evidence bar       |
  +----------------+       +----------------------+
          |                          |
          +------------+-------------+
                       v
  DAY 2          +----------------------+
                 | 5 SCOPE MEMO         |
                 | question, evidence,  |
                 | exclusions, kill     |
                 +----------------------+
                       |
                       v
                 sign-off: owner, data steward, programme lead
```

Read the diagram as a sequence with two loops.
The first loop is between the problem and the decision: if the problem cannot be stated as a
decision with an owner, return to the problem and ask who is affected and what they do now.
The second loop is between the data triage and the feasibility check: if the data cannot support
the evidence bar, either narrow the claim or change the question.

Three properties of the model matter.

**The clock is short on purpose.** Two days is long enough to check access and talk to an owner,
and short enough that the team cannot build anything. The constraint forces decisions rather than
investigation.

**Every gate produces an artefact.** A gate without an artefact is a conversation, and
conversations are reconstructed differently by each participant. The artefacts are a problem
statement, a decision sentence, a data triage table, a feasibility verdict, and a memo.

**The exclusions are part of the scope.** A scope that names only what is included is not a scope;
it is a wish. The memo's out-of-scope section is what protects the core, and it is the section a
panel reads first.

A fourth property is political.
Scoping decides whose problem is being solved.
A team that scopes with the sponsor alone will miss the user; a team that scopes with the user
alone will miss the capacity constraint; a team that scopes with neither will solve a problem that
exists only in the data.
The two-day clock includes at least two conversations for that reason.

### The four scope failures

Four failures account for most week-3 reframing, and each is detectable in the two days.

Wrong outcome: the team predicts something that is measured but not actionable, such as cost
rather than need, or a code rather than a clinical event.
Wrong population: the data describe a different group from the one the decision concerns, such as
inpatients for a decision about discharged patients.
Wrong owner: the decision has no one with the authority or capacity to act, so the output is
information without an actuator.
Wrong comparator: current practice is not measured, so the claim has no bar and the result cannot
be interpreted.

The four are independent, and a scope can fail more than one. The triage catches the first two, the
conversations catch the third, and the PICO sentence catches the fourth.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Open problem | A sentence describing a dissatisfaction without a decision | A project brief | Sentence |
| Scope | The question, population, decision, and exclusions a project commits to | The topic area | One page |
| Scope memo | The signed record of the scope | A proposal | One page |
| Decision question | A question whose answer changes an action | A research question | One sentence |
| PICO | Population, Intervention, Comparator, Outcome | A search strategy only | Four fields |
| FINER | Feasible, Interesting, Novel, Ethical, Relevant | A grant checklist | Five criteria |
| Knockout criterion | A condition that eliminates an option before scoring | A low score | Yes or no |
| Feasibility triage | A four-hour check of access, coverage, labels, and time zero | A data audit | Table |
| Evidence bar | The minimum evidence a claim requires | The evidence available | Validation level and artefacts |
| Kill criterion | A numeric result that ends the project or the claim | A vague intention | Threshold and metric |
| Out of scope | An explicit exclusion with a reason | An omission | Named exclusion |
| Parking lot | Ideas deferred with a date and an owner | A wish list | Entry and review date |
| Comparator | Current practice measured on the same cohort | No model at all | Same cohort and outcome |
| Capacity | Actions the owner can take per unit time | Assumed infinite | Per shift or day |
| Time zero | The moment inputs are fixed and the outcome clock starts | Implicit | Timestamp |
| Label provenance | How the outcome was ascertained | The column name | Source and adjudication |
| Coverage | Proportion of the target population in the data | Row count | Percentage and groups |
| Evidence sufficiency | Precision at which the owner can decide | Statistical significance | Minimum detectable difference |
| Reframe | Changing the question while keeping the problem | Scope creep | Recorded decision |
| Sponsor | The person who asked for the work | The person who uses it | Role |
| User | The person who acts on the output | The sponsor | Role |
| Sign-off | Written agreement to the scope | A meeting | Names and date |
| Two-day clock | The fixed scoping budget | An estimate | 16 wall-clock hours |
| Verdict | The triage's recommendation: proceed, narrow, fallback, stop | A score | Four labels |
| Proxy outcome | A measurable substitute for the outcome of interest | The outcome | Variable and justification |
| Minimum detectable difference | The smallest effect the design can distinguish | Any improvement | Effect size |
| Scope change | A documented alteration after sign-off | Drift | Version and date |
| Data steward | The role that answers access and coverage questions | The person who sends the file | Named role |
| Administrative data | Records collected for operations rather than research | Research data | Source and purpose |
| Board review | A manual process for scanning a queue | Any review | Process description |

The vocabulary is small because the two days are short. A team that spends the first hour arguing
about definitions has lost an eighth of the budget, and the definitions in this table are enough to
start.

Two of the terms are easy to confuse in a memo. A proxy outcome is a substitute and must be
labelled; an administrative outcome is a record collected for operations and must be described by
its collection process. A memo that says "outcome: length of stay" without saying how it is
recorded has left the most important provenance question unanswered.

Three definitions are worth separating because they are often conflated in a memo.
A proxy outcome is not a poor outcome; it is an outcome measured for a reason, and the reason must
be written next to it.
A narrow claim is not a weak claim; it is a claim the evidence can support, and it is often the
stronger position at a defense.
A scope change is not a failure; it is a documented decision, and an undocumented one is the
failure.

The most consequential definition is the decision question.
A research question asks what is true; a decision question asks what should be done.
"Is readmission predictable?" is a research question. "Should the transitional-care nurse call
these twenty patients first?" is a decision question.
The first can be answered without changing anything; the second cannot.

A second consequential definition is the knockout.
A knockout is not a low score; it is a condition that makes the option inadmissible regardless of
how well it scores elsewhere.
No owner, no legally accessible data, and no outcome measured at the right time are knockouts.
Scoring them as low weights is how teams talk themselves into infeasible projects.

## 5. The mechanics, step by step

The procedure has fourteen steps across four half-days.
It assumes two team members and access to the sponsor, one user, and a data steward.
The output is a signed one-page memo and a parking lot.

### Day 1, morning: problem and decision

#### Step 1. Write the problem as a sentence with a who and a how often

**Input:** the sponsor's request.

**Output:** one sentence: "[who] experiences [problem] [how often], and the current response is
[what]."

**Rationale:** The sentence forces specificity about the affected population and the frequency,
which are the two facts that determine whether the problem is worth twelve weeks. A problem that
affects three patients a year is real and not a capstone.

**Branch point:** if the frequency is unknown, the first task is to estimate it from administrative
data. If it cannot be estimated in a day, the problem is not scoped for a two-day exercise.

A useful test for the sentence is to read it aloud to someone outside the project. If they ask
"who decides what?" the sentence is missing the decision; if they ask "how often?" it is missing
the frequency. The questions a listener asks are the gaps in the scope.

#### Step 2. Identify the sponsor and the user separately

**Input:** the request and the organisational chart.

**Output:** two named roles: the person who asked for the work, and the person who would act on the
output.

**Rationale:** The sponsor controls resources; the user controls adoption. They are often different,
and conflating them produces a project that is funded and unused.

**Branch point:** if the user cannot be identified, the project is a research study. That is
acceptable if the question is worth it, and it must be labelled as such.

Naming the user also determines the interface: a charge nurse needs a list at the start of a shift,
a physician needs a flag in a worklist, and an analyst needs a table. The same model produces three
different products, and the scope should say which one is being built.

#### Step 3. Write the decision question in PICO form

**Input:** the problem sentence and the user.

**Output:** population, action, comparator, and outcome, each in a phrase, plus the frequency of
the decision.

**Rationale:** PICO is designed for evidence questions, and it works for decisions because it
forces the comparator and the outcome. A decision without a comparator has no bar; a decision
without an outcome has no evidence.

**Branch point:** if the outcome cannot be measured within the project, the decision is not
evaluable. Either choose a proxy outcome and label it, or reframe the decision.

#### Step 4. State the current practice and its cost

**Input:** one conversation with the user.

**Output:** a description of what happens today, how long it takes, and what it costs in time,
money, or harm.

**Rationale:** Current practice is the comparator and the value case. A team that cannot describe
it will build a model that improves on nothing in particular.

**Branch point:** if current practice is already good, the project may be a validation or
evaluation study rather than a new model. That is a legitimate reframe and often a stronger one.

### Day 1, afternoon: data and feasibility

The afternoon is the part teams skip, and it is the part that decides whether the project is
possible. Four questions, each answered with a document rather than an opinion.

#### Step 5. List the data elements the decision requires

**Input:** the PICO fields.

**Output:** a list of variables with the time at which each becomes available.

**Rationale:** The list is the bridge between the question and the data. It also exposes features
that are recorded after the decision, which are leakage risks.

**Branch point:** if a required variable is not recorded, the decision may not be computable. Check
before designing around it.

#### Step 6. Run the access check

**Input:** the data list and the data steward.

**Output:** for each source, the access tier, the licence, the approval required, and the elapsed
time to access.

**Rationale:** Access is the most common cause of capstone failure. A four-hour check in day 1 is
worth more than a week of modelling on data the team may not legally use.

**Branch point:** if access takes longer than the project, activate a fallback now. Do not assume
the approval will arrive.

#### Step 7. Check coverage and case mix

**Input:** a data extract or a data dictionary with counts.

**Output:** the number of eligible records, the number of positive outcomes, the date range, the
sites, and the representation of the main subgroups.

**Rationale:** Coverage determines whether the evidence can support the claim. A dataset with
10,000 rows and 40 positives cannot support a subgroup analysis, and a dataset from one site
cannot support a multi-site claim.

**Branch point:** if the positives are below the requirement for the target interval, pool data,
extend the date range, or narrow the claim. Do not proceed and hope the interval is narrow.

#### Step 8. Check label provenance and time zero

**Input:** the data dictionary and a sample of records.

**Output:** how the outcome was ascertained, by whom, with what lag, and the exact timestamp that
defines time zero.

**Rationale:** Label provenance determines whether the outcome means what the question assumes.
An outcome defined by a coding rule may reflect billing practice rather than clinical events, and
a time zero that is not reconstructable makes every feature suspect.

**Branch point:** if the label cannot be trusted, consider a different outcome or a different
dataset. A model trained on a billing artefact will learn billing.

### The four-hour triage in practice

The four hours are not a data audit. A full audit takes days. The triage answers four questions
with the minimum evidence that makes the answer credible.

Hour one: who owns the data, what is the access route, and what is the elapsed time. The answer
comes from the data steward or the dataset page, and it is recorded with a date.
Hour two: how many eligible records and how many positives, by year and by group. The answer comes
from a count query, not from a sample, and the counts are written down.
Hour three: how was the outcome ascertained, and at what timestamp is the prediction made. The
answer comes from the data dictionary and from five manually inspected records.
Hour four: what is missing, and does the missingness differ by group. The answer comes from a
missingness table by column and by subgroup.

The output is a one-page table with four rows and a verdict. The verdict is one of: proceed,
proceed with a narrowed claim, switch to fallback, or stop.

A triage that runs over four hours has become an audit, and an audit in the scoping phase is a
way of avoiding the decision. The team should stop, write the verdict from what it has, and move
to the options.

The verdict is the artefact that the review reads. It should be one sentence, for example:
"Proceed with a narrowed claim: 5,600 eligible records and 214 positives from 2023 onward, which
supports an overall estimate and a two-group comparison but not a four-group comparison."

### Day 2, morning: options and kill criteria

#### Step 9. Generate three scope options

**Input:** steps 1 to 8.

**Output:** three options, each with a claim, an evidence bar, and an estimate of effort.

**Rationale:** Three options prevent the first idea from becoming the scope by default. They also
give the review something to choose between, which is the difference between a decision and an
announcement.

**Branch point:** if all three options fail a knockout, the problem is not scoped for this team.
Record the reason and return to the brief catalogue rather than forcing an option.

#### Step 10. Apply the knockouts before scoring

**Input:** the three options.

**Output:** a list of options eliminated by a knockout, with the specific condition.

**Rationale:** Knockouts are binary and non-negotiable: no owner, no legal data, no outcome at the
right time, no capacity to act. Scoring them as weights allows a strong option to compensate for a
fatal flaw, which is exactly the reasoning that produces doomed projects.

**Branch point:** if an option survives only by changing the claim, it is a different option.
Record the change and re-apply the knockouts.

#### Step 11. Write the kill criterion

**Input:** the surviving option and the evidence bar.

**Output:** one effect criterion and one feasibility criterion, each numeric, with a measurement
date and an owner.

**Rationale:** The kill criterion is the scope's safety valve. It converts a vague possibility of
failure into a decision that can be made at a gate without renegotiating the project.

**Branch point:** if the team cannot name a result that would stop the project, the project is a
demonstration, not an investigation. Say so in the memo.

#### Step 12. Estimate effort and compare with capacity

**Input:** the option and the team's hours.

**Output:** hours for data, modelling, validation, governance, and writing, and the fraction of
capacity each consumes.

**Rationale:** A scope that consumes more than the available hours is not a scope. The estimate
also reveals which phase is the bottleneck, which is usually data or governance rather than
modelling.

**Branch point:** if the total exceeds capacity, cut the stretch tier and narrow the claim. Do not
plan to work evenings.

### Day 2, afternoon: memo and review

#### Step 13. Write the one-page memo

**Input:** steps 1 to 12.

**Output:** a one-page memo with decision, population, comparator, evidence bar, data plan,
exclusions, kill criterion, and sign-off lines.

**Rationale:** The memo is the artefact that survives the two days. It is short because it is read,
and it is signed because a signature is what distinguishes a scope from a suggestion.

**Branch point:** if the memo needs more than one page, the scope is not yet narrow enough. Cut
until it fits.

#### Step 14. Hold the review and record the parking lot

**Input:** the memo and the three reviewers.

**Output:** sign-off, or a documented reason for refusal, and a parking lot of deferred ideas.

**Rationale:** The review is where the sponsor, the user, and the data steward disagree in public,
which is cheaper than discovering the disagreement in week 8. The parking lot protects the deferred
ideas without letting them into the scope.

**Branch point:** if a reviewer refuses to sign, the scope is not agreed. Record the objection and
resolve it before week 1 ends, or choose a different scope.

A review that ends with everyone nodding but nobody signing is the most common failure of the two
days. The signature is not bureaucracy; it is the moment the sponsor, the user, and the data
steward each accept a specific obligation. Without it, the team leaves with three different
understandings and discovers the difference at the first gate.

### Decision table: what a failed gate means

| Gate failed | Meaning | Action | Who decides |
|---|---|---|---|
| Problem frequency unknown | Not enough information to scope | One day of administrative data | Team lead |
| No user identified | Research only | Label as methods study or change problem | Sponsor |
| No comparator | No bar for the claim | Measure current practice first | Clinical mentor |
| Data access too slow | Schedule risk | Activate fallback or narrow to public data | Data steward |
| Too few positives | Precision insufficient | Pool, extend, or narrow the claim | Biostatistics mentor |
| Label provenance unclear | Outcome may be an artefact | Change outcome or dataset | Clinical mentor |
| No kill criterion | Demonstration, not investigation | Reframe or accept the label | Programme lead |
| Effort exceeds capacity | Not a scope | Cut, narrow, or stop | Team lead with panel |

### The two-day time budget

The budget is sixteen wall-clock hours across two days, with a team of two or three.
The allocation below is the one used in the worked example; the arithmetic is in Section 6.

| Block | Wall-clock hours | Person-hours | Output |
|---|---:|---:|---|
| Frame the problem | 3.0 | 6.0 | Problem sentence |
| Interviews | 2.0 | 4.0 | Decision question, current practice |
| Data triage | 4.0 | 4.0 | Access, coverage, labels, time zero |
| Access check | 1.5 | 1.5 | Access table |
| Options | 2.5 | 5.0 | Three scope options |
| Kill criteria | 1.5 | 3.0 | Two criteria |
| Memo | 2.5 | 2.5 | One page |
| Review | 1.0 | 3.0 | Sign-off or objection |
| Total | 18.0 | 29.0 | Memo |

The total exceeds sixteen hours, which is the first thing the arithmetic reveals. The fix is to
run the access check in parallel with the data triage and to merge the options and kill-criteria
blocks, which brings the wall clock to 15.5 hours and leaves 0.5 hours of slack.

The person-hour total of 29 against a capacity of 48 (three people, two days) is 60 percent
utilisation. The wall clock, not the person-hours, is the binding constraint, which is the same
lesson as in the hackathon format: parallelism helps only when the tasks can be divided.

### Edge cases and forced decisions

**The sponsor wants a different question after the review.** Record the request, estimate its cost,
and name the artefact it would displace. If the sponsor will not name the displacement, the scope
stands. This is the same displacement rule as in Topic 02.

**The data steward says access will take eight weeks.** The scope is infeasible on that data. Switch
to the fallback and rewrite the population in the memo, or choose a different problem. Do not
write a scope that depends on an approval the team cannot control.

**The user is enthusiastic but has no capacity.** The decision is real and the action is impossible.
Narrow the claim to a prioritisation study and record the capacity constraint as the reason. An
honest prioritisation study is better than an alert nobody can act on.

**The clinical mentor disagrees with the sponsor.** The mentor's objection is about the claim, and
the sponsor's is about the resources. Both need to sign. If they cannot agree, the scope is not
agreed, and the programme lead decides whether to proceed.

**The team falls in love with an option that fails a knockout.** This is the most common failure of
the two days. The knockout is not a preference; it is a condition. Record the elimination and move
on, and note the emotional attachment in the debrief rather than in the memo.

**The problem is real but the data do not exist.** The scope becomes a data-collection proposal,
which is out of scope for a twelve-week capstone. Record it in the parking lot with the collection
plan, and choose a problem with data.

## 6. Worked example

An emergency department (ED) director asks a team: "Can you predict boarding so we can open surge
beds earlier?"
The team has two days, three members, and a data steward who can answer questions for one hour.

### Day 1, morning: from request to decision

The first problem sentence is "ED crowding is getting worse". It fails the who-and-how-often test.
After an hour with the director and a charge nurse, the sentence becomes:

"On weekday evenings, admitted patients wait a median of 4.2 hours for an inpatient bed, and the
charge nurse currently decides which patients to escalate to the bed-management team by walking
the board every hour."

The frequency is now explicit: weekday evenings, about 260 evenings per year. The current response
is a manual board review. The sponsor is the ED director; the user is the charge nurse.

The decision question in PICO form:

- Population: admitted adults waiting in the ED for an inpatient bed on weekday evenings.
- Action: rank patients for escalation to bed management.
- Comparator: the charge nurse's current board review.
- Outcome: time from admission decision to bed assignment, and the proportion waiting more than
  six hours.

The decision frequency is per evening, and the capacity is the number of escalation calls the
charge nurse can make, which the nurse estimates at six per evening.

### Day 1, afternoon: the data triage

The team lists the required variables: admission decision time, bed request time, bed assignment
time, ward, acuity, and the board review log.
The data steward reports that the admission and bed timestamps exist in the bed-management system
from 2021 onward, that acuity is recorded in the nursing assessment, and that the board review log
is a paper sheet that has never been digitised.

Three findings change the scope.

First, the board review log does not exist as data. The comparator cannot be measured directly, so
the team must reconstruct it from the escalation calls recorded in the bed-management system, which
are a partial proxy.

Second, the extract has 14,800 admitted adults over three years, but only 2,100 are weekday
evenings, and only 310 of those waited more than six hours. The positive count is 310, which is
enough for an overall estimate and not enough for a four-way subgroup analysis.

Third, time zero is ambiguous. The admission decision is recorded in the ED system, the bed request
in the bed-management system, and the two clocks differ by a median of 11 minutes. The team must
choose one as time zero and state the choice.

### Diagnostic one: the comparator cannot be measured

The first diagnostic is whether current practice can be quantified on the same cohort. Here it can
be approximated but not measured, because the board review log is paper.
That is not a fatal flaw, but it changes the claim: the team cannot say "better than the charge
nurse's judgement", only "associated with a shorter time to bed than the recorded escalation
pattern". The distinction is the difference between a comparative claim and an observational one.

A team that ignores this will write "outperforms current practice" in the dossier and be unable to
defend it.

### Diagnostic two: the positives do not support the subgroups

The second diagnostic is arithmetic. With 310 positives, the team can estimate an overall
proportion with a half-width of about 0.055. Splitting into four ward groups gives roughly 78
positives per group and a half-width of about 0.11, which is too wide to compare groups.

The team therefore pre-specifies two subgroups, medical and surgical, with about 155 positives
each, and reports the other splits as exploratory with intervals and a warning.

### Day 2, morning: options and kill criteria

The team generates three options.

Option A: a ranking model for escalation, validated temporally, with a capacity of six calls per
evening. The claim is that the ranking captures more long-wait patients in the top six than the
recorded escalation pattern.
Option B: a forecasting model for evening boarding volume, used to staff the bed-management team.
Option C: an evaluation of the existing escalation pattern, with no new model.

The knockouts eliminate none of the three, but they rank them. Option A requires a comparator that
cannot be measured; Option B requires staffing data the team has not checked; Option C requires
only the timestamps and the escalation records.

The team scores the options on decision fit, owner capacity, data fitness, evidence feasibility,
and time. Option A scores 3.9, Option B 3.4, Option C 3.6.

Option A wins narrowly. The team chooses it with a narrowed claim: not "better than the charge
nurse", but "a ranking that captures long-wait patients in the top six at a higher rate than the
recorded escalation pattern, on a temporal holdout".

The kill criteria:

- Effect: if the top-six capture rate is not at least 0.10 above the recorded pattern on the
  temporal holdout, the team will not claim improvement.
- Feasibility: if the weekday-evening cohort falls below 1,800 records or the positives below 250,
  the team will narrow to a feasibility study.

### Day 2, afternoon: the memo and the budget

The memo fits on one page: decision, population, comparator, claim, evidence bar (L2 temporal
validation), data plan (bed-management extract, 2021 to 2025, time zero at bed request), exclusions
(no real-time deployment, no paediatric ED, no staffing model), kill criteria, and sign-off lines.

The time budget is the one in Section 5. The wall clock sums to 18 hours against 16 available, so
the team runs the access check in parallel with the data triage and merges the options and
kill-criteria discussions into a single 3-hour block. The revised wall clock is 15.5 hours, and the
person-hours are 29 against a capacity of 48.

### Varying one assumption

Suppose the paper board-review log had been digitised. The comparator becomes measurable, and the
claim strengthens from "associated with" to "compared with current practice". The effect score for
Option A rises from 3.9 to 4.2, and the team can use a direct comparison in the dossier.

The evidence bar does not change, and neither does the validation level. What changes is the
sentence the team is allowed to write, which is the practical difference between a measurable
comparator and a reconstructed one.

### What the two days produced

Five artefacts: the problem sentence, the PICO decision, the data triage table, the feasibility
verdict with two diagnostics, and the signed memo. The team also has a parking lot with two items:
the staffing model and the paediatric ED.

The cost is 29 person-hours, or about 1,740 dollars at 60 dollars per hour. The alternative, three
weeks of discovery, costs about 5,400 dollars and delays the evidence by three weeks.

### What the memo said

The memo fit on one page, and its sentences are worth recording because each one closes a decision.

Decision: on weekday evenings, the charge nurse uses a ranked list of six patients to decide whom
to escalate to bed management.
Owner: the charge nurse, with the ED director as sponsor.
Population: admitted adults waiting for an inpatient bed, weekday evenings, 2021 to 2025.
Comparator: the recorded escalation pattern in the bed-management system.
Claim: the ranking captures long-wait patients in the top six at a higher rate than the recorded
pattern, on a temporal holdout.
Evidence bar: L2 temporal validation, calibration, two pre-specified subgroups.
Exclusions: no real-time deployment, no paediatric ED, no staffing model, no causal claim about
length of stay.
Kill criteria: effect below 0.10 improvement, or fewer than 250 positives.
Sign-off: charge nurse, ED director, data steward, programme lead.

The exclusions are as important as the inclusions. Each one names something a reader might expect
and explains why it is absent. A memo without exclusions invites the reader to assume the project
is larger than it is.

### The failure the two days prevented

In week 3 the team would have discovered that the bed-management extract changed its schema in
2023, when a new triage system was installed. The timestamp fields were renamed and the ward
coding changed.

The data triage caught this in hour two, because the count by year showed a discontinuity: 2021 and
2022 had 3,100 weekday-evening records per year, and 2023 onward had 2,400. The team restricted the
cohort to 2023 onward, which reduced the sample to 5,600 records and 214 positives, below the 250
kill threshold.

That triggered the feasibility kill criterion in week 0 rather than week 3. The team narrowed the
claim to a feasibility study with 214 positives and a wider interval, and recorded the decision in
the memo. The alternative, discovering the discontinuity after three weeks of modelling, would have
invalidated the temporal split and the comparison.

## 7. Code, query, or calculation

The block applies knockouts before scoring, then computes the two-day budget and its slack. It uses
only the standard library.

```python
criteria = {"decision": 0.25, "owner": 0.20, "data": 0.20,
            "evidence": 0.20, "time": 0.15}
candidates = {
    "ED boarding prediction": (4, 3, 3, 2, 3),
    "Readmission outreach":   (5, 4, 4, 3, 4),
    "OR scheduling":          (4, 4, 5, 4, 4),
    "Sepsis alert":           (5, 2, 2, 1, 2),
}
knockouts = {
    "ED boarding prediction": "comparator not measurable",
    "Readmission outreach": "",
    "OR scheduling": "",
    "Sepsis alert": "no owner for alert review",
}

for name, scores in candidates.items():
    if knockouts[name]:
        print(f"KNOCKOUT {name}: {knockouts[name]}")
        continue
    total = sum(s * w for s, w in zip(scores, criteria.values()))
    print(f"{name}: {total:.2f}")
```

Expected output:

```text
KNOCKOUT ED boarding prediction: comparator not measurable
Readmission outreach: 4.05
OR scheduling: 4.20
KNOCKOUT Sepsis alert: no owner for alert review
```

The ordering matters more than the values. A knockout removes an option before its strengths can
compensate, which is the discipline that scoring alone cannot enforce.

```python
blocks = {
    "Day 1 AM frame": 3.0,
    "Day 1 AM interviews": 2.0,
    "Day 1 PM data triage": 4.0,
    "Day 1 PM access check": 1.5,
    "Day 2 AM options": 2.5,
    "Day 2 AM kill criteria": 1.5,
    "Day 2 PM memo": 2.5,
    "Day 2 PM review": 1.0,
}
people = 3
days = 2
hours_per_day = 8
wall = sum(blocks.values())
capacity = people * days * hours_per_day
print(f"wall clock {wall:.1f} h vs {days * hours_per_day} h available")
print(f"slack {-wall + days * hours_per_day:.1f} h")
print(f"person-hours planned {sum(blocks.values()) * 1.6:.1f} vs capacity {capacity} h")
```

Expected output:

```text
wall clock 18.0 h vs 16 h available
slack -2.0 h
person-hours planned 28.8 vs capacity 48 h
```

The person-hour figure uses a 1.6 multiplier to approximate two people on the paired blocks; it is
an estimate, not a measurement. The point is that the wall clock fails before the person-hours do,
which means the fix is parallelism or merging blocks, not adding people.

### Reading the output

The knockout list is the part of the first block that changes decisions. A candidate with a
knockout is removed even if its weighted score would be the highest, which is the difference
between triage and preference.

The second block separates two capacities that teams routinely conflate. Person-hours are
abundant, because three people working for two days have 48 of them. Wall-clock hours are scarce,
because the phases are sequential and the review cannot start before the memo exists.

When the wall clock fails and the person-hours do not, the available fixes are to parallelise a
phase, merge two phases, or shorten a phase by narrowing its output. Adding a fourth person does
nothing, because the critical path does not change.

### A note on the multiplier

The 1.6 multiplier in the code is a rough approximation of the average number of people working on
a block. It could be computed exactly by listing assignments, and in a real budget it should be.
The approximation is used here to keep the example short, and it is labelled as an approximation
rather than presented as a measurement.

## 8. Data, standards, and regulatory context

Scoping has no single regulatory instrument, but four bodies of rules constrain it, and a scope
memo that ignores them produces a plan that cannot be executed.

### Research question frameworks

PICO and FINER are conventions from clinical research, not regulations. PICO structures a question
as population, intervention, comparator, and outcome; FINER asks whether it is feasible,
interesting, novel, ethical, and relevant.
The PRISMA and SPIRIT statements use similar structures for systematic reviews and trial protocols.
None is binding for a capstone, and all are useful because they force the comparator and the
outcome into the sentence.

### Prediction model reporting

TRIPOD+AI, the 2024 reporting guideline for prediction models, requires the intended use, target
population, outcome, predictor availability at prediction time, sample size justification, and
calibration. A scope memo that omits any of these will produce a dossier that cannot be reported
against the guideline.
DECIDE-AI adds the human-in-the-loop workflow for early clinical evaluation, which matters when the
scope includes a live pilot.

### Data protection and access

The scope determines the data tier, and the tier determines the approval route.
Open data require a licence review; credentialed data require training and a data-use agreement;
restricted data require an institutional review board approval or determination and a named
protocol; synthetic data require a documented generator and a statement of limits.
A scope memo should name the tier and the approval route for each source, with the elapsed time.

Under the GDPR, a scope that processes health data needs a lawful basis and may need a data
protection impact assessment. Under HIPAA, the covered entity's privacy officer determines minimum
necessary access. Neither is a capstone formality, and both take time that the scope must include.

### Device regulation

If the scope includes a clinical claim, the intended use determines whether the output is a device
function. The FDA clinical decision support guidance is the relevant test in the United States: a
function that analyses an image or signal, provides a time-specific directive, or issues a
time-critical alert is likely a device.
A scope memo should record the preliminary classification and the open question, even when the
answer is "research use only".

### Where the scope memo sits in the dossier

The memo is not one of the thirteen dossier artefacts, and it is the input to four of them.
It supplies the decision and population to the charter, the data tier and access route to the
provenance record, the evidence bar to the analysis plan, and the kill criteria to the risk
register.
A team that cannot find the memo in week 6 will find that the four artefacts disagree with each
other, because each was written from a different memory of the scope.

The practical rule is to version the memo and to reference its version in the charter. When the
scope changes, the memo changes first, and the other artefacts follow. That ordering is what keeps
the dossier coherent.

### What the standards do not decide

No standard tells a team which problem to scope. PICO will structure any question, including a bad
one. FINER will rate a well-structured question as feasible when the data are inaccessible, because
feasibility in FINER is a judgement rather than a check.

The check is the data triage, and it is the part of the two days that cannot be delegated to a
framework.

| Instrument | Status | What it constrains in scoping |
|---|---|---|
| PICO | Convention | Question structure |
| FINER | Convention | Question quality |
| TRIPOD+AI | Reporting convention | Intended use, population, calibration |
| DECIDE-AI | Reporting convention | Live workflow evaluation |
| HIPAA | Binding law in the US | Access and minimum necessary |
| GDPR | Binding law in the EU | Lawful basis and impact assessment |
| FDA CDS guidance | Binding interpretation | Device classification |
| PhysioNet DUA | Contractual | Access and redistribution |

## 9. Industry practice and case evidence

### What evaluation programmes report about scoping failures

NHS England's guidance on planning and implementing real-world AI evaluations, based on the AI in
Health and Care Award portfolio, describes recurring problems that begin at scoping: no named
operational owner, an evidence plan written after procurement, and a deployment that cannot
generate the data needed to evaluate it.
The guidance is practitioner material rather than a controlled study, and its authority comes from
the portfolio it summarises.

The Royal College of Radiologists' 2025 consultation document on AI deployment fundamentals makes a
related point: device marking is not clinical or cost-effectiveness evidence, and independent
evaluation is scarce. A scoping process that assumes marking equals evidence will under-budget the
evaluation by an order of magnitude.

NICE's early value assessment programme publishes evidence-generation plans with explicit gap
tables. The 2025 plan for AI in opportunistic detection of vertebral fragility fractures marks
health-related quality of life, resource use, and diagnostic accuracy against standard care as
absent or limited, and proposes multi-year real-world designs. A twelve-week capstone cannot fill
such a gap, and a scope that claims to is mis-scoped.

### What challenge design teaches about scoping

The PhysioNet 2025 Chagas challenge scoped a decision, not a diagnosis: which patients to prioritise
for scarce serological testing, scored by the fraction of true cases in the top 5 percent. That is
a scoping decision, and it is why the challenge is a useful model. The organisers chose the
capacity, the population, and the metric before the data were released.

The 2026 challenge goes further, conditioning the metric on age to remove a shortcut. That is a
scoping decision about what counts as evidence, made before any team submitted.

A capstone team can copy the discipline without copying the task: choose the capacity, the
population, and the metric before looking at the model's performance, and write them down.

### What hackathons teach about the cost of skipping scope

The hackathon format in Topic 6 allots six hours to need validation, and the evidence on
continuation suggests that projects grounded in a real need last longer. The Olson and colleagues
survey of twelve healthcare hackathons found that 30.3 percent of projects had made progress after
a mean of 12.2 months, which is a low base rate for a process that often skips scoping entirely.

The inference is not that scoping causes continuation; it is that the projects which continue are
the ones with a real user and a real workflow, which are exactly what scoping establishes.

### What a failed scope looks like in a published evaluation

The NIHR mixed-methods study of an AI command centre in an NHS trust, published in 2024, is a
documented example of a deployment whose benefits were attributed to organisational adaptation
rather than to the technology. The study could not demonstrate significant impacts on patient
flow, safety, or data quality.

For scoping, the lesson is that the decision and the workflow were not separable from the
organisation. A scope that treats the model as the intervention will mis-specify the comparator,
because the comparator is the organisation as it was, not the model as it might have been.

### What the challenge record shows about pre-specified capacity

The 2025 PhysioNet Chagas challenge chose a capacity of 5 percent before releasing the data, and
the metric followed from that choice. The 2026 challenge chose an age-conditioned metric before
any submission. Both are scoping decisions made in public, and both constrain what a winning team
can claim.

A capstone that makes the same choices in a one-page memo has done the same thing at a smaller
scale. The difference is that the capstone must also measure current practice, which the challenge
does not.

The comparison is worth stating in the memo's rationale. A reviewer who sees that the capacity and
the metric were chosen before the data will trust the analysis plan more than one who sees them
chosen after the first results.

### Inference, labelled

It is an inference that a two-day scoping exercise reduces week-3 reframing. The mechanism is that
the four failure modes, wrong outcome, wrong population, wrong owner, and wrong comparator, are all
detectable in two days with access to the sponsor, the user, and the data steward. No controlled
study of capstone teams is cited.

The inference is testable within a programme: compare the number of scope changes recorded after
week 3 for teams that ran the two days against teams that did not. A programme that does not record
scope changes cannot test it, which is itself an argument for the change log.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Problem stated as a topic | The sentence has no who or how often | Abstraction | Unbounded scope | Rewrite with who and frequency |
| Sponsor confused with user | Only one name on the memo | Organisational shorthand | Funded but unused | Identify both roles |
| No comparator | "We will compare to no model" | Convenience | No bar for the claim | Measure current practice |
| Outcome not measured at the right time | Outcome available only after discharge | Data convenience | Leakage or unusable target | Check time zero |
| Data access assumed | No access date in the memo | Optimism | Weeks lost | Access check in day 1 |
| Too few positives | Subgroup counts missing | Precision ignored | Unsupported claim | Count positives by group |
| Knockout scored as a weight | "It is weak on ownership but strong elsewhere" | Scoring replaces judgement | Doomed project proceeds | Binary knockout first |
| No kill criterion | "We will see how it goes" | Avoidance | No decision at the gate | Numeric criterion with date |
| Effort exceeds capacity | Plan assumes evenings | Optimism | Quality collapse at validation | Cut or narrow |
| Scope memo too long | Two pages and growing | Everything feels essential | Nobody reads it | One page, exclusions explicit |
| Reviewers not asked to sign | "They seemed happy" | Avoidance of commitment | Disagreement surfaces late | Written sign-off |
| Parking lot becomes scope | Deferred items reappear | Enthusiasm | Scope creep | Date and owner per item |
| Reframe unrecorded | Question changed in week 4 | Drift | Inconsistent dossier | Memo version and change log |
| Scope memo not versioned | Two copies with different claims | No change control | Dossier contradicts itself | Version and reference in the charter |

The table's last row is the quietest failure and the one a panel finds first, because it compares
the charter with the slides. A versioned memo makes the comparison trivial.

The table is ordered by how early the failure is detectable. The first five are visible in the two
days; the last three appear later and are consequences of the first five.

A team that reads only one row should read the sixth. Scoring a knockout as a weight is the
mechanism by which an otherwise competent team commits to a project that cannot succeed, and it is
the failure that the two-day format is designed to prevent.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The sponsor can be reached in two days | The clock depends on it | Senior clinician availability | Book before day 1 |
| The user can describe current practice | Comparator depends on it | User unavailable | Use a documented workflow study |
| A data steward can answer in one hour | Triage depends on it | No steward assigned | Ask the programme lead |
| The extract's schema is stable over the period | A temporal split depends on it | A system change mid-period | Count records by year |
| The outcome definition is stable | The label means the same thing throughout | Coding change | Compare code frequencies by year |
| Administrative data can estimate frequency | Problem sizing depends on it | No records | Interview and estimate |
| The data extract is representative | Coverage check depends on it | Referral centre data | Compare case mix |
| Positives can be counted quickly | Precision estimate depends on it | Labels require chart review | Use a coded proxy and label it |
| Two days is enough | The format depends on it | Problem spans multiple departments | Split the problem |
| Sign-off is possible | The memo depends on it | No authority in the room | Escalate to the programme lead |
| The team can stop | The kill criterion depends on it | Sponsor pressure | Agree the stop rule in writing |
| Two days is enough for the governance question | The memo depends on it | Multi-site approval | Add a third day or narrow to one site |

The last row is the one teams resist. A multi-site scope needs a governance conversation that two
days cannot contain, and a memo signed by one site is not a multi-site scope.

A two-day exercise that ends with "this problem is not scoped for a capstone" has succeeded, not
failed, provided the reason is written and an alternative is in the parking lot.

The largest limit is that two days cannot resolve a genuinely contested problem. If the sponsor and
the clinical lead disagree about the outcome, no amount of scoping will settle it; the disagreement
is the finding, and it belongs in the memo as an open question with an owner.

A second limit is that the triage depends on a data steward who can answer quickly. In institutions
where access requests go through a central queue, the steward may not know the answer, and the
triage becomes an estimate. The memo should record which answers were verified and which were
assumed.

A third limit is that the two-day format is calibrated for a single-site problem. A multi-site or
multi-department problem needs a third day for the governance conversation, and pretending
otherwise produces a memo that the second site will not sign.

## 12. Exercises

### Exercise 1. Rewrite three problem sentences (application, 30 minutes)

Rewrite "readmissions are high", "sepsis is missed", and "radiology is backlogged" as problem
sentences with a who, a frequency, and a current response.
**Deliverable:** three sentences.
**Criteria:** each has a population, a frequency with a unit, and a described current response.
**Time:** 30 minutes.

### Exercise 2. Write a PICO decision question (application, 25 minutes)

For one problem sentence, write population, action, comparator, outcome, and decision frequency.
**Deliverable:** five fields and one sentence.
**Criteria:** the comparator is current practice, the outcome is measurable, the frequency is
stated.
**Time:** 25 minutes.

### Exercise 3. Run a data triage (calculation, 60 minutes)

For a dataset you can access, compute eligible records, positives, date range, sites, and subgroup
counts, and estimate the half-width for the overall and subgroup estimates.
**Deliverable:** a triage table with arithmetic.
**Criteria:** counts reproducible; half-widths computed; subgroups with fewer than 30 events
flagged.
**Time:** 60 minutes.

### Exercise 4. Apply knockouts (critique, 30 minutes)

Take three candidate scopes and eliminate any that fail a knockout, with the specific condition.
**Deliverable:** a knockout table and a decision.
**Criteria:** knockouts are binary; the condition is named; no eliminated option is scored.
**Time:** 30 minutes.

### Exercise 5. Write kill criteria (design, 30 minutes)

Write one effect and one feasibility criterion for a scope of your choice, each with a threshold, a
date, and an owner.
**Deliverable:** two sentences and a decision rule.
**Criteria:** numeric thresholds; measurement date before freeze; owner named.
**Time:** 30 minutes.

### Exercise 6. Write the one-page memo (design, 45 minutes)

Produce a scope memo for your capstone using the structure in Step 13.
**Deliverable:** one page.
**Criteria:** decision, population, comparator, evidence bar, exclusions, kill criterion, and
sign-off lines all present; it fits on one page.
**Time:** 45 minutes.

### Exercise 7. Budget the two days (calculation, 30 minutes)

Build a wall-clock and person-hour budget for your scoping exercise and identify the binding
constraint.
**Deliverable:** the table and the arithmetic.
**Criteria:** wall clock and person-hours separated; slack computed; the fix named.
**Time:** 30 minutes.

### Exercise 8. Run the review (design, 30 minutes)

Role-play the review with one person as sponsor, one as user, and one as data steward. Record the
objections and the sign-off.
**Deliverable:** a decision record.
**Criteria:** every objection has an owner; the sign-off is explicit; unresolved issues are listed.
**Time:** 30 minutes.

**Assessment note.** Exercises 1, 2, and 4 have checkable answers. The rest are assessed on the
memo and the record: whether the decision is specific, whether the evidence bar is reachable, and
whether the exclusions are honest.

The exercise set is designed to be run in the order given. Exercise 3 produces the counts that
Exercise 5 needs for a threshold, and Exercise 6 assembles the artefacts from Exercises 1 to 5.
A team that skips Exercise 3 will write a kill criterion without a number.

## 13. Self-check answers

**1. What makes a sentence a decision question rather than a research question?**
A decision question names an action and an actor, so answering it changes what someone does. A
research question asks what is true, and can be answered without changing practice.

**2. Why are knockouts applied before scoring?**
Because a fatal flaw cannot be compensated by strength elsewhere. Scoring a missing owner as a low
weight allows an infeasible option to win on other criteria, which is how doomed projects begin.

**3. Compute the half-width for 310 positives.**
Using the normal approximation with p = 0.5 for a conservative estimate, half-width = 1.96 ×
sqrt(0.25/310) = 1.96 × 0.0284 = 0.056. With p closer to 0.3 the half-width is about 0.052. The
text's 0.055 is a reasonable round figure.

**4. Why is the wall clock the binding constraint in the two-day budget?**
Because the phases must happen in sequence for the same people, so adding people does not shorten
the calendar. The person-hour capacity is 48 against 29 planned, but the wall clock is 18 against
16 available.

**5. What does it mean when the comparator cannot be measured?**
It means the claim must change from a comparative one to an observational or associational one. The
team can still report a ranking result, but not "better than current practice".

**6. Why pre-specify subgroups in the memo rather than after the results?**
Because a subgroup chosen after seeing the data is a hypothesis, not a test. Pre-specification
makes the analysis confirmatory, and it prevents the team from reporting only the splits that look
interesting.

**7. What is the difference between a parking lot and out-of-scope?**
Out of scope is excluded from the project; the parking lot is deferred with a date and an owner and
may become a later project. Both protect the core, and they are recorded differently.

**8. Why must the memo fit on one page?**
Because it is read by three busy people and signed. A two-page memo is a proposal, and proposals are
discussed rather than signed.

**9. What should a team do if a reviewer refuses to sign?**
Record the objection, name its owner, and resolve it before week 1 ends. An unsigned scope is not
agreed, and proceeding converts a disagreement into a week-8 surprise.

**10. Rubric for Exercise 6.** A strong memo has one decision, one comparator, one evidence bar, an
explicit exclusion list, and a numeric kill criterion, and it fits on a page. A weak memo describes
the data and the model, leaves the comparator implicit, and runs to three pages.

**11. Rubric for Exercise 4.** A strong knockout table removes options for named conditions and
does not score them afterwards. A weak table scores the knockouts and then argues that the
strengths compensate, which is the failure the exercise is designed to catch.

**12. Why is a narrow claim often stronger at a defense?**
Because it is supported by the evidence. A narrow claim that survives questioning is more
persuasive than a broad claim that requires the panel to accept a gap, and the panel rarely does.

**13. What is the first artefact to update when the scope changes?**
The scope memo. It is the input to the charter, the analysis plan, the provenance record, and the
risk register, so updating it first keeps the other four consistent.

**14. Why does the two-day clock include a data steward?**
Because access and coverage are the two facts that most often invalidate a scope, and neither can
be resolved by the team alone. The steward's hour is the cheapest insurance in the process.

**15. Rubric for Exercise 8.** A strong review record shows three distinct roles, at least one
objection with an owner, and an explicit sign-off or refusal. A weak record shows agreement without
obligations, which is the failure the review exists to prevent.

**16. What should a team do if the sponsor and the user disagree about the outcome?**
Record both outcomes, the cost of each, and the decision route. If the disagreement cannot be
resolved in the two days, the scope is not agreed, and the programme lead decides whether to
proceed with one of them or to change the problem.

## 14. Interview questions

**Q1. "How did you decide what to scope?"**
Strong answer names the decision, the owner, the comparator, and the exclusion. Weak answer
describes the dataset.

**Q2. "What did you decide not to do?"**
Strong answer names a specific exclusion and its cost. Weak answer says the scope was kept narrow.

**Q3. "How did you know the data could support the claim?"**
Strong answer gives counts, positives, and an interval. Weak answer says the dataset was large.

**Q4. "What would have made you stop?"**
Strong answer gives a numeric kill criterion and a date. Weak answer says if the results were bad.

**Q5. "Who was the user, and how did you find them?"**
Strong answer distinguishes sponsor and user and describes the conversation. Weak answer names a
department.

**Q6. "How did you handle a disagreement between the sponsor and the clinician?"**
Strong answer describes the recorded objection and the resolution route. Weak answer says the team
went with the sponsor.

**Q7. "Why did you not build the alert?"**
Strong answer explains capacity and the evidence bar. Weak answer says there was not enough time.

**Q8. "What was your comparator?"**
Strong answer describes current practice and how it was measured. Weak answer says no model
existed.

**Q9. "How long did scoping take, and was it enough?"**
Strong answer gives hours, artefacts, and one thing discovered late. Weak answer says two days.

**Q10. "What is the first thing you would check on a new problem?"**
Strong answer says whether an owner will act and whether the outcome is measurable. Weak answer
says data availability.

**Q11. "How do you know two days is enough?"**
Strong answer says it is enough to establish the four scope elements and to eliminate knockouts,
and that resolving a contested outcome is not a scoping task. Weak answer says two days is what the
programme allows.

A strong answer to this set always distinguishes what was decided from what was discovered. The
decisions belong in the memo; the discoveries belong in the risks.

## 15. Key terms

- **Capacity** — actions the owner can take per unit time.
- **Comparator** — current practice measured on the same cohort.
- **Coverage** — proportion of the target population in the data.
- **Decision question** — a question whose answer changes an action.
- **Evidence bar** — minimum evidence a claim requires.
- **Evidence sufficiency** — precision at which the owner can decide.
- **Feasibility triage** — four-hour check of access, coverage, labels, time zero.
- **FINER** — feasible, interesting, novel, ethical, relevant.
- **Kill criterion** — numeric result that ends the project or claim.
- **Knockout** — condition that eliminates an option before scoring.
- **Label provenance** — how the outcome was ascertained.
- **Open problem** — dissatisfaction without a decision.
- **Out of scope** — explicit exclusion with a reason.
- **Parking lot** — deferred ideas with date and owner.
- **PICO** — population, intervention, comparator, outcome.
- **Reframe** — changing the question while keeping the problem.
- **Scope memo** — signed one-page record of the scope.
- **Sign-off** — written agreement to the scope.
- **Sponsor** — person who asked for the work.
- **Time zero** — moment inputs are fixed and outcome clock starts.
- **Two-day clock** — fixed scoping budget.
- **User** — person who acts on the output.
- **Verdict** — triage recommendation among proceed, narrow, fallback, stop.
- **Data steward** — role answering access and coverage questions.

The list is short enough to learn in an hour, which is the point. A scoping vocabulary that takes a
day to master would defeat the two-day format.

Two further terms appear in the memo and are worth defining here. A capacity number is the count of
actions the owner can take per shift, day, or week; it is a number, not an adjective.
An open question is a question the two days could not resolve, with a named owner and a date; it is
not a gap in the memo.

## 16. Sources

All URLs accessed 26 October 2026.

- Hulley SB, Cummings SR, Browner WS, et al. *Designing Clinical Research*, 4th ed. Lippincott
  Williams and Wilkins, 2013. (FINER criteria)
- Richardson WS, Wilson MC, Nishikawa J, Hayward RS. The well-built clinical question.
  *ACP Journal Club* 1995;123(3):A12-13.
- Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement. *BMJ* 2024;385:e078378.
  https://doi.org/10.1136/bmj-2023-078378
- Vasey B, Nagendran M, Campbell B, et al. DECIDE-AI. *Nature Medicine* 2022;28:924-933.
  https://doi.org/10.1038/s41591-022-01772-9
- NHS England, Planning and implementing real-world AI evaluations.
  https://www.england.nhs.uk/long-read/planning-and-implementing-real-world-ai-evaluations-lessons-from-the-ai-in-health-and-care-award/
- Royal College of Radiologists, *AI deployment fundamentals*, 2025.
  https://www.rcr.ac.uk/media/r0th2x0k/ai-deployment-fundamentals_public-consultation.pdf
- NICE, Evidence generation plan for AI in opportunistic detection of vertebral fragility
  fractures, early value assessment, 2025.
  https://www.nice.org.uk/consultations/2785/2/approach-to-research
- FDA, *Clinical Decision Support Software*, guidance, September 2022.
  https://www.fda.gov/regulatory-information/search-fda-guidance-documents/clinical-decision-support-software
- PhysioNet/CinC Challenge 2025. https://moody-challenge.physionet.org/2025/
- PhysioNet/CinC Challenge 2026. https://moody-challenge.physionet.org/2026/
- Olson KR, Walsh M, Garg P, et al. Health hackathons: theatre or substance? *BMJ Innovations*
  2017;3:37-44. https://doi.org/10.1136/bmjinnov-2016-000147
- HIPAA Privacy Rule, 45 CFR Parts 160 and 164. https://www.hhs.gov/hipaa/
- Regulation (EU) 2016/679, General Data Protection Regulation.
  https://eur-lex.europa.eu/eli/reg/2016/679/oj

Source note: FINER is attributed to the clinical research methods literature and is presented here
as a convention, not a validated instrument. The frameworks are taught because they expose missing
elements, not because they guarantee a good question.

## 17. Links

The memo template in this topic feeds the charter in Topic 09. A team that completes the two days
has already written the decision and population sections of the dossier. The kill criteria feed the
risk register, and the exclusions feed the limitations section.

The next topic turns those inputs into the four deliverables a panel actually reads: dossier,
prototype, demo, and defense.

- Module map: [`_index.md`](_index.md)
- Previous topic: [`07_competition-landscape.md`](07_competition-landscape.md)
- Next topic: [`09_deliverables.md`](09_deliverables.md)
- Related topic: [`05_brief-catalogue.md`](05_brief-catalogue.md)
- Survey file: [`../../course/19_capstone_hackathons.md`](../../course/19_capstone_hackathons.md)
- Labs: [`../../labs/`](../../labs/)
