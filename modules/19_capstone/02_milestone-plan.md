# Milestone Plan

> **Module 19 — Capstone, Hackathons, and Competitions**
> Topic 02 of 11
> Estimated study time: 3.5 hours
> Prerequisites: Topic 01
> Version 2026.3

## 1. Why this topic exists

A capstone has a fixed end date and an unfixed amount of work.
The gap between those two facts is where projects fail.
Teams do not usually fail because a task was hard; they fail because the hard task started in
week 8, the review it needed took ten days, and the person who could unblock it was on leave.

The milestone plan is the instrument that converts a twelve-week intention into a sequence of
dated, testable commitments.
It is not a Gantt chart for its own sake.
It exists to answer four questions that a panel, a mentor, and a teammate will each ask in a
different form: what is due, who can tell whether it is done, what happens if it is not, and
what will be dropped first.

Most student plans fail one of three tests.
They define deliverables as activities ("work on the model") rather than as artefacts with a
pass condition.
They schedule work as if review, rework, and coordination were free.
They have no kill criteria, so a project that should stop continues on momentum and arrives at
the defense with a weak claim instead of an honest negative result.

Industry has a mature vocabulary for this problem.
Stage-Gate, developed by Robert Cooper and widely used in product development, separates stages
of work from gates at which a cross-functional group decides go, kill, hold, or recycle.
The gate has three parts: deliverables, criteria, and outputs.
The important property is that the gate has teeth, meaning a meaningful share of projects are
stopped, not merely reviewed.

A capstone borrows the structure and not the bureaucracy.
Five gates, each with a named reviewer and a written pass condition, are enough.
The gates exist because a twelve-week project cannot afford to discover in week 11 that the
data never arrived, the owner never existed, or the metric never matched the decision.

There is also a statistical reason to plan explicitly.
Task durations are uncertain, and the uncertainty compounds along the critical path.
A plan built from most-likely estimates will be late roughly half the time even if every
estimate is unbiased, because the project duration is the maximum of several random paths, not
the sum of averages.
The arithmetic in Section 6 shows a plan whose most-likely duration is eleven weeks and whose
80th percentile is 11.7 weeks, which is fine, and a plan whose access step slips by two weeks
and whose probability of meeting the deadline collapses to 2.6 percent.

That is the practical argument for this topic.
The plan is not paperwork; it is the only artefact that tells you, while there is still time to
act, whether the project you agreed to is the project you can finish.

There is a second, less flattering reason the topic exists: teams are bad at this without help.
Kahneman and Tversky's planning fallacy describes the systematic tendency to underestimate
durations and costs even when the estimator knows that similar projects overran. Flyvbjerg's
work on major projects documents the same pattern at scale, and proposes reference class
forecasting as the correction: estimate from the distribution of comparable projects rather than
from the inside view of the current plan.
A capstone team is not exempt. The plan that says four weeks for a data audit is usually the
inside view, and the last three course projects are usually the outside view.

A plan also has to distinguish three schedules that teams conflate.
The work schedule is the dependency graph: what must follow what.
The calendar schedule is the work schedule plus lags, holidays, and review latency.
The commitment schedule is the set of dates the team has told other people about.
Most slippage is a commitment schedule that was built from the work schedule and never updated
with the calendar.

The milestone plan is therefore a living artefact with a version number, not a document produced
once for a proposal. It is reviewed at every gate, and every change is dated with its effect on
the end date. A team that cannot say when its plan last changed cannot explain why it is late.

## 2. Learning outcomes

By the end of this topic you can:

1. **Convert** a scope statement into a dependency graph of work packages with durations,
   owners, and predecessors.
2. **Compute** the critical path, total float per task, and the deterministic end date for a
   twelve-week plan.
3. **Estimate** the probability of finishing by the deadline using a three-point estimate and a
   seeded Monte Carlo simulation.
4. **Load** effort by week against team capacity, and identify the weeks in which the plan is
   over-allocated.
5. **Write** a gate deliverable that a reviewer can test in under ten minutes, with an explicit
   pass condition and a named reviewer.
6. **Specify** kill criteria that are numeric, pre-registered, and owned, and distinguish them
   from stopping rules for model iteration.
7. **Choose** what to cut when the plan slips, using a pre-agreed order rather than an argument
   in week 9.
8. **Run** a pre-mortem that produces at least six failure scenarios with triggers and
   responses.

## 3. The core mental model

A milestone plan is a directed graph with a calendar attached.
Nodes are work packages that produce artefacts.
Edges are dependencies, and each edge has a latency that is often larger than the work.
The calendar imposes a hard end date, and the difference between the graph's end and the
calendar's end is the slack that the team can spend.

```text
  WEEK   1     2     3     4     5     6     7     8     9    10    11    12
        |     |     |     |     |     |     |     |     |     |     |     |
  GATE  G0----G1------------------G2------------------G3----G4----G5
        |     |                   |                   |     |     |
  WORK  charter  access  audit  baseline  candidate  validation  dossier
        |     |     |        |        |         |          |
        v     v     v        v        v         v          v
      decision  DUA  provenance split  error   L2/L3     freeze
                IRB  leakage  intervals curve  subgroups rehearsal

  Critical chain (most likely, weeks):
  charter 1.0 -> access 2.0 -> audit 1.5 -> baseline 2.0 -> candidate 2.5
          -> validation 2.0 -> demo 2.0 -> dossier 2.0 -> freeze 1.0  = 16.0
```

Read the diagram in two layers.
The top layer is the gate calendar, which is fixed by the programme and cannot be moved.
The bottom layer is the work, which is movable within limits.
The critical chain is the longest path through the work; any delay on it delays the end date
one for one, while a delay off the chain consumes float and may not affect the end date at all.

Three properties of the model drive every planning decision.

**The maximum, not the sum.** Project duration is the maximum over paths, so the plan is
dominated by one chain. Improving a task that is not on the chain does not shorten the project,
which is why teams that spend week 6 tuning a model that is not the bottleneck feel busy and
finish late.

**Latency dominates work.** A data-use agreement takes two to six weeks of elapsed time and a
few hours of effort. A mentor review takes three to seven days. A conference deadline does not
move. Plans that count only effort produce schedules that are impossible in calendar time.

**Uncertainty is asymmetric.** Task durations have a floor near the optimistic estimate and a
long right tail. Averaging three estimates hides the tail, and the tail is what misses the
deadline. Monte Carlo makes the tail visible.

A fourth property is organisational rather than mathematical.
A gate is only real if someone has the authority to fail it.
If the panel's role is to encourage, the gate is a status meeting, and the project will proceed
past a fatal flaw. The plan must name the reviewer and the consequence, or it is decoration.

### Float is not laziness

A team that sees 5.5 weeks of float on governance may interpret it as permission to start late.
That is a misreading. Float is a property of the network, not of the team's effort: it says the
project will not be delayed if the task slips by that amount, not that the task should be pushed
to its late start.
Starting a task at its late start removes all protection against its own uncertainty. The task
then becomes critical by construction, and the team has converted a flexible plan into a
brittle one.

The practical rule is to start tasks at their earliest start when the owner is available, and to
treat float as insurance rather than as schedule. Float is spent by events, not by choice.

### Resource dependencies

Data dependencies are not the only constraints. Two tasks that need the same person, the same
dataset, or the same review slot cannot run in parallel even if the graph says they can.
These resource dependencies are invisible in a pure precedence diagram and are the most common
reason a logically feasible plan is impossible in practice.

The modelling convention is to add a dummy dependency or an explicit resource constraint
whenever two packages share a scarce resource for more than half of their duration.
The check is simple: for each week, list the packages active and the people assigned. If a
person appears twice at more than 100 percent, add the constraint before running the schedule.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Work package | A unit of work with one owner, one artefact, and a duration | A job title or an activity stream | Identifier and owner |
| Deliverable | A versioned artefact with a stated pass condition | An activity ("work on validation") | Artefact link and version |
| Gate | A scheduled decision point with deliverables, criteria, and a named reviewer | A progress meeting | G0 to G5 |
| Gate criteria | The tests a deliverable must pass, written before the gate | Agreed in the room | Pass condition per artefact |
| Kill criterion | A pre-specified result that ends the project or the claim | A general intention to reassess | Numeric threshold and metric |
| Go / kill / hold / recycle | The four gate decisions in Stage-Gate practice | "Continue" as the only outcome | Decision label |
| Dependency | A finish-to-start relationship between work packages | An implicit assumption that order does not matter | Predecessor identifier |
| Lag | Forced waiting time between tasks, such as review or credentialing | Ignored in effort-based plans | Days or weeks |
| Critical path | The longest dependency chain; its length is the project duration | The list of important tasks | Task sequence and weeks |
| Total float | The amount a task can slip without delaying the project end | Confused with "slack in the team" | Weeks |
| Three-point estimate | Optimistic, most likely, and pessimistic durations for a task | A single number with false precision | Three values per task |
| Monte Carlo schedule | Repeated sampling of task durations to produce a distribution of end dates | A single deterministic date | Percentiles |
| Effort | Person-hours required to complete a task | Elapsed calendar time | Hours |
| Capacity | Person-hours available per week across the team | Assumed to be forty per person | Hours per week |
| Over-allocation | Weeks in which planned effort exceeds capacity | Discovered in week 8 | Hours above capacity |
| Review latency | Elapsed time from submission to mentor feedback | Counted as zero | Days |
| Freeze | The date after which artefacts change only for factual correction | The defense date | Date and version tag |
| Rehearsal | A full timed run of the defense with feedback | Reading the slides aloud once | Count and duration |
| Pre-mortem | A structured exercise imagining the project has failed and listing causes | A risk register written after the fact | Scenario list |
| Change control | The rule for altering scope, schedule, or metric after G2 | Informal agreement in a meeting | Log entry with date and reason |
| Re-planning trigger | An event that forces the plan to be rebuilt | Any bad news | Named event and date |
| Buffer | Calendar slack reserved for known-unknowns | Padding every task | Weeks at the end |
| Reference class forecasting | Estimating duration from similar past projects rather than from the plan | Optimism dressed as experience | Base rate and adjustment |
| Burn-down | Remaining deliverables plotted against weeks | Hours worked | Count per week |
| Baseline plan | The approved version against which change is measured | Any earlier draft | Version and date |
| Contingency reserve | Buffer for identified risks, owned by the lead | Padding hidden in task estimates | Weeks or hours |
| Management reserve | Buffer for unidentified risks, owned above the team | The same thing as contingency | Weeks |
| Milestone | A zero-duration marker for a decision or artefact acceptance | Any task with a date | Date and criterion |
| Escalation | The route to a decision-maker when a blocker exceeds a threshold | Waiting for the next meeting | Role and response time |
| Handoff | Transfer of an artefact between owners with acceptance criteria | Sending a file | Artefact and acceptance test |
| Gate review | A time-boxed meeting applying criteria to deliverables | A demonstration | Decision record |

### The three schedules, distinguished

The work schedule is a graph. It has no dates, only order and duration.
The calendar schedule adds dates: weekends, holidays, rota gaps, and the lag for every approval.
The commitment schedule is the subset that has been communicated to mentors, sponsors, and the
programme office.

A common failure is to publish the work schedule as if it were the calendar schedule.
The work schedule says data audit follows access by zero days; the calendar says the audit
cannot start until the analyst returns from placement. The difference is real and predictable,
and it belongs in the plan.

The remedy is mechanical. Convert each dependency into a finish-to-start link plus a lag in
days, and add a calendar of non-working periods. Recompute. If the plan still fits, the plan is
credible. If it does not, the team has found the problem early enough to cut.

Two definitions are worth separating because teams conflate them.
Effort is the work; duration is the calendar. A task with four hours of effort can have three
weeks of duration because it waits for an external approval.
Capacity is the team's weekly effort; it is not the same as availability, because a person at
80 percent availability has 32 hours, not 40, and a person with clinical duties may have 6.

## 5. The mechanics, step by step

The procedure below builds a plan in one working session of about three hours, with the lead,
the data engineer, and the evidence lead present.
It produces a one-page plan, a dependency table, and a Monte Carlo estimate.
The procedure assumes the charter from Topic 01 exists; if it does not, stop and write it first.

### Step 1. Fix the hard dates

**Input:** the programme calendar, the defense date, and any external deadlines.

**Output:** three dates: the defense, the freeze (at least seven days before), and the last
acceptable date for the primary result (at least fourteen days before).

**Rationale:** Every later decision is relative to these dates. Freeze exists so that rehearsal
and factual correction have time; the primary-result date exists so that the dossier is written
from results rather than around them.
A team that sets freeze the day before the defense has no freeze.

**Branch point:** if the defense date is fixed by the programme, treat it as immovable. If the
team controls the date, move it rather than compressing weeks 11 and 12.

### Step 2. Enumerate the end artefacts

**Input:** the dossier specification from Topic 09.

**Output:** a list of thirteen artefacts, each with a version number and an owner.

**Rationale:** The plan is built backwards from artefacts, not forwards from activities.
An artefact list makes omissions visible: a team that has no row for the datasheet will not
write one.

**Branch point:** if the programme permits a reduced dossier, remove artefacts explicitly and
record the permission. Do not silently drop one.

### Step 3. Write the pass condition for each gate deliverable

**Input:** the artefact list and the gate definitions.

**Output:** for each gate, two to four artefacts, each with a test that a reviewer can apply in
under ten minutes.

**Rationale:** "Charter complete" is not testable; "charter names one owner, one action, one
frequency, and a capacity number" is. Testable conditions prevent the review from becoming a
conversation about effort.

**Branch point:** if a pass condition cannot be written, the artefact is not defined yet. Split
it until it can be.

### Step 4. Derive work packages from artefacts

**Input:** the artefact list with pass conditions.

**Output:** between fifteen and thirty work packages, each producing one artefact or a named
section of one.

**Rationale:** Work packages at the wrong granularity make estimation impossible. "Do the
analysis" is too large to estimate; "fit the logistic baseline and write the calibration plot"
is estimable.
A useful rule is that a package should be between four and forty hours of effort.

**Branch point:** if a package exceeds forty hours, split by method or by dataset. If it is
under two hours, merge it into its neighbour.

### Step 5. Estimate effort with three points, by role

**Input:** the work packages and the team's roles.

**Output:** for each package, optimistic, most likely, and pessimistic effort in hours, assigned
to a role rather than a person.

**Rationale:** Role-based estimates survive a personnel change and expose skill concentration.
Three points capture the tail that single estimates hide.
The estimates should be anchored to a reference: the last project the team did, or the lab
exercises in earlier modules.

**Branch point:** if the optimistic and pessimistic estimates differ by more than a factor of
three, the package is not understood. Spend thirty minutes decomposing it before estimating.

### Step 6. Add review and approval latency

**Input:** the mentoring cadence and the access process.

**Output:** a lag in days for each artefact that requires review, approval, or external action.

**Rationale:** Review latency is the most common source of schedule surprise. A mentor who
reviews on Wednesdays adds up to seven days to any artefact submitted on Thursday.
Credentialing adds one to two weeks; a DUA can add four.

**Branch point:** if a required approval has no published service level, assume the pessimistic
value and schedule a chase on the calendar.

### Step 7. Build the dependency graph

**Input:** work packages, lags, and the artefact list.

**Output:** a table with predecessor identifiers for every package, and a graph with no cycles.

**Rationale:** The graph makes the critical path computable and exposes false parallelism.
Teams routinely assume that two tasks can run in parallel when they share one person, one
dataset, or one review slot.

**Branch point:** if two packages need the same scarce person in the same week, add a
resource dependency even though no data dependency exists.

### Step 8. Compute the critical path and the deterministic end date

**Input:** the graph and the most-likely durations.

**Output:** earliest start and finish per package, total float, and the project end date.

**Rationale:** The computation answers the first planning question: does the plan fit before any
uncertainty is considered? If the deterministic end exceeds the calendar, no amount of luck
will save it.

**Branch point:** if the deterministic end exceeds the deadline, go to Step 9. If it fits with
less than one week of float, treat it as not fitting.

### Step 9. Compress by overlap, not by optimism

**Input:** the over-long plan.

**Output:** a revised plan with at least one of: a task started earlier against a provisional
input, a task split into a fast draft and a final version, or a task moved off the critical
chain.

**Rationale:** Compression by shortening estimates is self-deception. Compression by overlap
changes the graph, which changes the arithmetic.
Starting the candidate model on a provisional feature set while the split is finalised is a
legitimate overlap if the final run is repeated on the locked data.

**Branch point:** if compression requires skipping a gate, stop. Gates are cheaper to keep than
to explain.

### Step 10. Load effort by week and compare with capacity

**Input:** the revised plan and the team's weekly hours.

**Output:** a table of planned hours per week per role, and a list of over-allocated weeks.

**Rationale:** A plan can be feasible in dependency terms and impossible in effort terms.
Three workstreams converging in week 8 is a capacity problem even when the graph allows it.

**Branch point:** if any week exceeds capacity by more than 20 percent, move a non-critical task
earlier or later, or reduce scope. Do not assume the team will work evenings for three weeks.

### Step 11. Place kill criteria

**Input:** the decision statement, the comparator, and the metric.

**Output:** at least two kill criteria, one on effect and one on feasibility, each with a numeric
threshold, a measurement time, and an owner.

**Rationale:** A kill criterion protects the team from sunk-cost reasoning and protects the
panel from a vague claim. An effect criterion might be "if the site-held-out sensitivity is
below 0.60, we will not claim site generalisation". A feasibility criterion might be "if the
positive-case count is below 80 at week 7, we will narrow the claim to a feasibility study".

**Branch point:** if a criterion is triggered, the decision is made at the next gate, not
immediately, so that the team can present options rather than a surprise.

### Step 12. Define decision rules and owners for each gate

**Input:** the gate list and the reviewer names.

**Output:** one row per gate with the decision options, the deciding role, and the default
action if no decision is made.

**Rationale:** A gate without a default drifts. "If the panel does not meet, the project
continues on the current plan and the lead escalates within 48 hours" is a default, and it
prevents silent slippage.

**Branch point:** if the reviewer cannot attend, delegate in writing. A gate held by someone
without authority is not a gate.

### Step 13. Schedule data dependencies and fallbacks as tasks

**Input:** the data-access plan.

**Output:** calendar entries for the primary request, the chase, the fallback switch decision,
and the re-run of the pipeline on fallback data.

**Rationale:** Fallbacks only work if switching is planned. A fallback discovered in week 6
costs two weeks of rework; a fallback with a scheduled switch date costs two days.

**Branch point:** if the primary arrives after the switch date, keep the fallback result as the
primary and treat the late data as a sensitivity analysis.

### Step 14. Schedule rehearsals and freeze

**Input:** the defense format.

**Output:** two full rehearsals, at least three days apart, one of them with a mentor; a freeze
date; and a version tag.

**Rationale:** The first rehearsal always reveals a structural problem, and there must be time
to fix it. Two rehearsals also test the demo twice, which is the minimum for catching a broken
dependency.

**Branch point:** if a rehearsal reveals a factual error in the results, the freeze moves and
the panel is informed. If it reveals a presentation problem, the slides change and the results
do not.

### Step 15. Version and publish the plan

**Input:** the plan artefacts.

**Output:** `plan.md` in the repository, a one-page PDF for mentors, and a calendar with the
gates and rehearsals.

**Rationale:** A plan that lives in a document nobody opens is not a plan. The calendar entry is
the operational form; the repository version is the auditable one.

**Branch point:** every change after publication gets a dated entry with the reason and the
effect on the end date. Undocumented plan changes are how a project arrives late without anyone
noticing.

### Step 16. Run a pre-mortem

**Input:** the plan and the team.

**Output:** six to ten failure scenarios, each with an early signal, a trigger, and a response.

**Rationale:** The pre-mortem converts anxiety into a list. It also surfaces disagreements that
the team would otherwise discover at the gate, when they are expensive.

**Branch point:** if a scenario has no detectable early signal, it cannot be managed; either
find a signal or accept the risk explicitly in the register.

### How to run a gate meeting

A gate meeting is thirty minutes, not ninety, and it has a fixed agenda.

The first five minutes are the decision request: the team states what it wants the panel to
decide, in one sentence.
The next ten minutes are the evidence: the artefacts, the pass conditions, and the one number
that matters.
The next ten minutes are questions, restricted to the criteria rather than to general advice.
The final five minutes are the decision, recorded in the decision log with the reviewer's name
and the date.

Three rules make the meeting work.
The team does not present slides it has not sent 48 hours in advance; the panel reads before the
meeting.
The panel does not introduce new criteria at the meeting; a new criterion is a change to the
gate and is recorded as such.
The decision is one of go, kill, hold, or recycle, and "continue with concerns" is recorded as
go with a named risk, not as an ambiguous fourth option.

The decision record is short: gate, date, decision, conditions, owner of each condition, and the
date by which the condition is checked. It lives in `governance/decisions.md` and is part of the
dossier. A panel that reads it at the defense can see whether the team acted on its own
conditions.

### Decision table: what to cut when the plan slips

| Weeks behind | First cut | Second cut | Never cut | Who decides |
|---:|---|---|---|---|
| 1 | Stretch tier (external site, value-case sensitivity) | Second candidate model | Freeze buffer | Team lead |
| 2 | Error-analysis breadth (30 to 15 cases) | Subgroup count (three to two, with a note) | Gates G2 and G5 | Team lead with mentor |
| 3 | System demonstration to a recorded walkthrough | Value case to a one-page estimate | Validation level | Panel at G3 |
| 4 | Candidate model entirely; report baseline | Dossier to the minimum thirteen artefacts | Regulatory memo | Panel at G3 |
| 5 or more | Re-scope to a feasibility study with a measured comparator | Public-data replication | Honest reporting of what was not done | Program lead |

The table is pre-agreed at G0 so that the argument happens once, when nobody is tired.
The right-hand column is the part that matters: some things are never cut, and naming them
early is what makes the rest negotiable.

### Edge cases and forced decisions

**Data access denied late.** Switch on the pre-agreed date, not on the day of refusal. Re-run
the pipeline on fallback A, rewrite the intended use to the fallback population, and record the
switch in the change log. The schedule absorbs two days of rework if the schema matches, and two
weeks if it does not, which is why schema compatibility is a selection criterion for fallback A.

**A team member drops out.** Re-plan the same day, not at the next gate. Remove every work
package owned by that role from the critical path, reassign at most one of them, and cut the
rest. A four-person team that loses the data engineer should drop the system layer to a
recorded walkthrough; a team that tries to absorb twelve hours per week of specialist work will
fail validation instead.

**Scope creep.** Every new request gets a written entry with an estimated cost in hours and a
named artefact it would displace. If the requester will not name the displacement, the request
is declined. This is a rule, not a negotiation tactic.

**Negative result.** If the candidate fails the kill criterion, the plan does not end; it
switches to the negative-result track, which has its own work packages: finalise the comparator
analysis, write the decision-not-to-change recommendation, document the evidence against
deployment, and present the minimum evidence that would change the conclusion. That track takes
about ten days and fits inside weeks 9 to 11 if it starts at G3.

**Cannot recruit a clinician.** Replace the clinical liaison with a documented need assessment
and a clinical reviewer who gives two hours at G3 and G5. Adjust the plan by adding a
two-week literature and workflow review in weeks 2 and 3, and remove the reader-study stretch
item. If no clinician can review at all, the claim must be limited to a methods contribution.

**Compute exhausted.** Cap the number of candidate runs at G2, for example twelve runs, and
treat the cap as a kill criterion for tuning. When the cap is reached, the best run becomes the
candidate and the remaining time goes to validation. A plan without a run cap will consume
weeks 7 to 10 in tuning.

**Submission deadline.** The deadline is protected by the freeze date, which is protected by the
primary-result date. If the primary result is not available seven days before freeze, the plan
switches to the baseline result and the candidate is reported as incomplete. This is decided by
the date, not by hope.

## 6. Worked example

A four-person team is building a 30-day readmission risk score for a transitional-care
programme. The programme can enrol 25 patients per week, so the decision is: which discharged
patients does the transitional-care nurse call first. The defense is in week 12.

### The plan as first written

The team lists eleven work packages and estimates most-likely durations in weeks:
charter 1.0, data access 2.0, data audit 1.5, baseline 2.0, analysis plan 1.0, candidate 2.5,
validation 2.0, governance 2.0, system and demo 2.0, dossier 2.0, freeze 1.0.
Dependencies are a simple chain, with governance depending on the analysis plan and the dossier
depending on governance and the demo.

The critical path runs charter, access, audit, baseline, candidate, validation, demo, dossier,
freeze: 1.0 + 2.0 + 1.5 + 2.0 + 2.5 + 2.0 + 2.0 + 2.0 + 1.0 = 16.0 weeks.

The plan does not fit. This is the first and most valuable output of the exercise: the team
learns in week 1 that its twelve-week plan is a sixteen-week plan.
The analysis plan and governance have 5.5 weeks of float, so improving them does not help.
The whole problem is the technical chain.

### Compression by overlap

The team makes three changes.
The candidate model starts as soon as the data audit closes, using the provisional feature set,
and is re-run on the locked split at the end; its duration becomes 3.0 weeks but it starts two
weeks earlier.
The system and demo are built from the baseline rather than waiting for the final validation,
so the demonstration is ready earlier and is refreshed after validation.
The dossier is drafted from the governance memo in parallel with validation, then updated with
the final numbers.

The new critical path is charter, access, audit, analysis plan, governance, dossier, freeze,
with the technical work in parallel: 1.0 + 2.0 + 1.5 + 1.0 + 2.0 + 2.5 + 1.0 = 11.0 weeks.
Float: baseline 1.5, candidate 0.5, validation 0.5, system and demo 1.5, governance 0.
The deterministic plan now fits with one week of slack.

### The arithmetic of uncertainty

Most-likely durations are not commitments. The team re-estimates each package with optimistic
and pessimistic values, for example candidate 2.0 to 4.0 weeks and data access 1.5 to 4.0
weeks, and runs 20,000 simulations with a seeded generator.

Results for the compressed plan:
mean 11.24 weeks, median 11.26, 80th percentile 11.65, 95th percentile 11.98.
Probability of finishing by week 12 is 95.4 percent; probability of finishing by week 11.5 is
69.4 percent.

That is an acceptable plan, but only just. A 4.6 percent chance of missing the defense is not
catastrophic, and it assumes the durations are right and the team works at capacity.

### Diagnostic one: the weekly load

The effort totals 334 hours across the plan. Capacity is 45 hours per week across the team
(five people at 9 hours, one of them the clinical liaison at 3).
Distributing each package's effort evenly across its duration gives weekly loads of
16, 12, 12, 20, 63, 59, 58, 66, 42, 25, 20, 0 hours.

Weeks 5 to 8 are over capacity by 31 to 47 percent, with a peak of 66 hours in week 8.
That is the second diagnostic, and it is not visible on a Gantt chart: the plan is feasible in
dependency terms and impossible in effort terms.
The team has three options.
Move governance earlier, which is possible because the analysis plan finishes in week 5.5; this
removes 2 hours per week from weeks 6 to 8.
Reduce the candidate's feature engineering from 56 to 40 hours, accepting a simpler model.
Accept the overload for two weeks, which is a decision with a known cost in quality and sleep.

The team chooses the first two, which brings the peak to about 52 hours and leaves a residual
overload of 15 percent in weeks 6 and 7. That residual is recorded in the risk register rather
than hidden.

### Diagnostic two: access slips by two weeks

The team's most consequential assumption is that the DUA arrives in week 3. Suppose it arrives
in week 5.

Deterministic end moves from 11.0 to 13.0 weeks.
Monte Carlo mean becomes 13.15, 80th percentile 13.64, and the probability of finishing by
week 12 falls to 2.6 percent.

That is the doom diagnostic, and it is cheap to compute before the project starts: vary the
access duration and look at the deadline probability. A plan whose deadline probability falls
below roughly 50 percent under a two-week access slip is a plan that depends on an external
process it does not control.

### What the team does with the result

The team does three things in week 1.
It files the DUA immediately and adds a chase task in week 2 with a named person.
It pre-approves fallback A, a public readmissions dataset with a compatible schema, and
schedules the switch decision for the end of week 3.
It adds a buffer of one week before freeze, funded by cutting the external-site stretch item.

With the buffer, the compressed plan has 2 weeks of float at the end, the probability of meeting
the deadline rises above 99 percent under the base case, and the access-slip scenario still
has a 60 percent chance of meeting it because the fallback switch recovers a week.

### The cost of the plan, and what the buffer buys

The team's effort totals 334 hours. At blended rates of 60 dollars per hour, plus 600 dollars of
compute, the plan costs 20,640 dollars. The one-week buffer that was added at the end costs
nothing in money and one week of elapsed time, and it raises the base-case deadline probability
from 95.4 to above 99 percent.

That trade is almost always worth taking. A week of buffer at the end is cheaper than a week of
rework during the defense, and it is the only intervention in the plan that improves the
deadline probability without changing the scope.

The buffer should be owned by the lead and spent only on the critical path. A buffer spent on a
task with float has not protected the project; it has simply moved slack from one account to
another.

### Gate G3: the decision the plan was built for

At week 7 the panel meets with three artefacts: the candidate results on the temporal holdout,
the calibration plot, and the error taxonomy from thirty reviewed cases.
The analysis plan, written in week 5, specifies the threshold for continuing: candidate AUROC at
least 0.03 above the comparator, calibration slope between 0.8 and 1.2, and no subgroup with a
sensitivity gap larger than 0.15.

The candidate meets the first criterion, fails the second, and is borderline on the third.
The panel has four options, and the plan pre-commits to how each is handled.

| Result pattern | Decision | Plan consequence |
|---|---|---|
| All three criteria met | Go to L3 | Add the site-held-out run in weeks 8 and 9 |
| Effect met, calibration failed | Recycle: recalibrate, re-test in one week | Move the demo task by one week; keep the claim |
| Effect not met | Kill the deployment claim | Switch to the negative-result track |
| Subgroup gap only | Go with a narrowed claim | Add the mitigation and monitoring plan |
| Data insufficient | Hold and extend the cohort | Use fallback, narrow to feasibility |

The panel chooses recycle. Recalibration takes four days, the re-test is run on the same
holdout, and the slope moves to 0.94. The demo moves by one week, which consumes the buffer but
not the deadline.

This is the value of a pre-committed decision table: the team does not spend week 7 arguing
about what counts as good enough. The argument happened in week 5, when the criteria were
written and the results were unknown.

### What the three numbers mean together

The deterministic end of 11.0 weeks says the plan is feasible if every estimate is right.
The 80th percentile of 11.65 says the plan has about 1.35 weeks of headroom before the deadline
in four cases out of five.
The weekly load peak of 66 hours says the team cannot deliver the plan at the assumed capacity
without either moving work or accepting a quality cost.

No single number is sufficient. A plan can be feasible and over-allocated, or well-resourced
and too long, or short and fragile. The three together describe the shape of the risk, and the
team's job is to choose which risk it prefers rather than to pretend none exists.

### Variation: the negative-result branch

At week 7 the candidate's temporal validation area under the curve is 0.61, against a comparator
of 0.64. The kill criterion says: if the candidate does not exceed the comparator by 0.03, the
claim changes from "improves prioritisation" to "no improvement demonstrated".

The plan switches to the negative-result track. The team spends week 8 finalising the comparator
analysis and the decision curve, week 9 writing the recommendation and the regulatory memo, and
week 10 on the dossier. The demo becomes a walkthrough of the comparator and the candidate side
by side, which is more informative than a single model.
The defense is stronger than it would have been with an overfit win, because the team can say
what it measured and why the decision should not change.

## 7. Code, query, or calculation

The three blocks below implement critical path analysis, weekly effort loading, and a seeded
Monte Carlo estimate. They use only the standard library and run in under two seconds.

### Block 1: critical path, float, and the effect of compression

```python
tasks = {
    "charter":       (1.0, []),
    "data_access":   (2.0, ["charter"]),
    "data_audit":    (1.5, ["data_access"]),
    "baseline":      (2.0, ["data_audit"]),
    "analysis_plan": (1.0, ["data_audit"]),
    "candidate":     (2.5, ["baseline"]),
    "validation":    (2.0, ["candidate"]),
    "governance":    (2.0, ["analysis_plan"]),
    "system_demo":   (2.0, ["validation"]),
    "dossier":       (2.0, ["governance", "system_demo"]),
    "freeze":        (1.0, ["dossier"]),
}

def prepare(t):
    order, seen = [], set()
    def visit(n):
        if n in seen:
            return
        seen.add(n)
        for p in t[n][1]:
            visit(p)
        order.append(n)
    for n in t:
        visit(n)
    succ = {n: [] for n in t}
    for n, (_, preds) in t.items():
        for p in preds:
            succ[p].append(n)
    return order, succ

def cpm(t):
    order, succ = prepare(t)
    es, ef = {}, {}
    for n in order:
        es[n] = max([ef[p] for p in t[n][1]], default=0.0)
        ef[n] = es[n] + t[n][0]
    end = max(ef.values())
    ls, lf = {}, {}
    for n in reversed(order):
        lf[n] = min([ls[s] for s in succ[n]], default=end)
        ls[n] = lf[n] - t[n][0]
    return order, es, ef, ls, end

order, es, ef, ls, end = cpm(tasks)
print(f"as planned: {end:.1f} weeks")
for n in order:
    print(f"{n:14s} ES {es[n]:4.1f} EF {ef[n]:4.1f} float {ls[n]-es[n]:4.1f}"
          + ("  CRITICAL" if abs(ls[n] - es[n]) < 1e-9 else ""))
```

Expected output:

```text
as planned: 16.0 weeks
charter        ES  0.0 EF  1.0 float  0.0  CRITICAL
data_access    ES  1.0 EF  3.0 float  0.0  CRITICAL
data_audit     ES  3.0 EF  4.5 float  0.0  CRITICAL
baseline       ES  4.5 EF  6.5 float  0.0  CRITICAL
analysis_plan  ES  4.5 EF  5.5 float  5.5
candidate      ES  6.5 EF  9.0 float  0.0  CRITICAL
validation     ES  9.0 EF 11.0 float  0.0  CRITICAL
governance     ES  5.5 EF  7.5 float  5.5
system_demo    ES 11.0 EF 13.0 float  0.0  CRITICAL
dossier        ES 13.0 EF 15.0 float  0.0  CRITICAL
freeze         ES 15.0 EF 16.0 float  0.0  CRITICAL
```

The float for `analysis_plan` and `governance` is 5.5 weeks, which is why speeding them up does
nothing for the deadline. The technical chain is critical in every step.

### Block 2: effort loading against capacity

```python
compressed = {
    "charter": (1.0, []), "data_access": (2.0, ["charter"]),
    "data_audit": (1.5, ["data_access"]), "baseline": (2.0, ["data_audit"]),
    "analysis_plan": (1.0, ["data_audit"]), "candidate": (3.0, ["data_audit"]),
    "validation": (2.0, ["candidate"]), "governance": (2.0, ["analysis_plan"]),
    "system_demo": (2.0, ["baseline"]), "dossier": (2.5, ["governance"]),
    "freeze": (1.0, ["dossier", "validation", "system_demo"]),
}

def cpm(t):
    order, seen = [], set()
    def visit(n):
        if n in seen:
            return
        seen.add(n)
        for p in t[n][1]:
            visit(p)
        order.append(n)
    for n in t:
        visit(n)
    es, ef = {}, {}
    for n in order:
        es[n] = max([ef[p] for p in t[n][1]], default=0.0)
        ef[n] = es[n] + t[n][0]
    return es, ef

es, ef = cpm(compressed)
effort = {"charter": 16, "data_access": 24, "data_audit": 30, "baseline": 40,
          "analysis_plan": 14, "candidate": 56, "validation": 36, "governance": 26,
          "system_demo": 32, "dossier": 40, "freeze": 20}
capacity = 45

load = [0.0] * 13
for n, hours in effort.items():
    start, finish = es[n], ef[n]
    per_week = hours / (finish - start)
    for week in range(int(start), int(finish + 0.999)):
        overlap = min(1.0, finish - max(start, week))
        load[week] += per_week * overlap

print("weekly hours:", [round(x) for x in load[:12]])
print("total effort:", sum(effort.values()), "hours")
print("peak week:", max(range(12), key=lambda i: load[i]) + 1,
      "at", round(max(load), 1), "hours")
over = [i + 1 for i in range(12) if load[i] > capacity]
print("over capacity:", over)
```

Expected output:

```text
weekly hours: [16, 12, 12, 20, 63, 59, 58, 66, 42, 25, 20, 0]
total effort: 334 hours
peak week: 8 at 65.8 hours
over capacity: [5, 6, 7, 8]
```

The distribution assumes effort is spread evenly within each package, which overstates the load
in the first and last weeks of a long package. It is a screening tool, not a resource-levelling
optimiser.

### Block 3: Monte Carlo deadline probability

```python
import random
from statistics import mean

triples = {
    "charter":       (0.5, 1.0, 1.5, []),
    "data_access":   (1.5, 2.0, 4.0, ["charter"]),
    "data_audit":    (1.0, 1.5, 2.5, ["data_access"]),
    "baseline":      (1.5, 2.0, 3.0, ["data_audit"]),
    "analysis_plan": (0.5, 1.0, 1.5, ["data_audit"]),
    "candidate":     (2.5, 3.0, 4.0, ["data_audit"]),
    "validation":    (1.5, 2.0, 3.0, ["candidate"]),
    "governance":    (1.5, 2.0, 2.5, ["analysis_plan"]),
    "system_demo":   (1.5, 2.0, 2.5, ["baseline"]),
    "dossier":       (2.0, 2.5, 3.0, ["governance"]),
    "freeze":        (0.5, 1.0, 1.5, ["dossier", "validation", "system_demo"]),
}

def prepare(t):
    order, seen = [], set()
    def visit(n):
        if n in seen:
            return
        seen.add(n)
        for p in t[n][1]:
            visit(p)
        order.append(n)
    for n in t:
        visit(n)
    return order

def end_week(durations):
    ef = {}
    for n in prepare({k: (v[0], v[3]) for k, v in triples.items()}):
        ef[n] = max([ef[p] for p in triples[n][3]], default=0.0) + durations[n]
    return max(ef.values())

random.seed(7)
N = 20000
ends = sorted(end_week({n: random.triangular(*v[:3]) for n, v in triples.items()})
              for _ in range(N))
print(f"mean {mean(ends):.2f}  p50 {ends[N//2]:.2f}  "
      f"p80 {ends[int(0.80*N)]:.2f}  p95 {ends[int(0.95*N)]:.2f}")
print(f"P(end <= 12) = {sum(e <= 12 for e in ends)/N:.3f}")
print(f"P(end <= 11.5) = {sum(e <= 11.5 for e in ends)/N:.3f}")
```

Expected output:

```text
mean 11.24  p50 11.26  p80 11.65  p95 11.98
P(end <= 12) = 0.954
P(end <= 11.5) = 0.694
```

The seeds make the output reproducible on the same Python version. If you change the task set,
you must re-run and update the numbers in the dossier; a Monte Carlo result copied from a
previous plan is a fabricated number.

## 8. Data, standards, and regulatory context

Milestone planning is management practice, not regulation, but several binding instruments
impose dates and sequencing that a plan must respect. Confusing an internal gate with an external
deadline is a common and expensive error.

### Project management standards

ISO 21500 provides guidance on project management concepts and processes, and ISO 21502 covers
project, programme, and portfolio management guidance. Neither is certifiable in the way ISO
9001 is, and neither prescribes a specific milestone structure. They are useful as a vocabulary
for roles, scope, and change control, not as a template for a twelve-week plan.

Stage-Gate is a proprietary process model rather than a standard. Cooper's description separates
stages from gates and defines gate criteria in three classes: readiness checks, must-meet
knock-outs, and should-meet scorecard items. The four gate outcomes are go, kill, hold, and
recycle. A capstone that adopts the vocabulary gains a defensible gate structure; it does not
gain a certification.

PRINCE2 and the Project Management Body of Knowledge (PMBOK) offer alternative vocabularies.
The choice does not matter for a capstone; consistency does. A plan that mixes stage names from
three frameworks confuses reviewers.

### Regulatory and ethics dates

An institutional review board (IRB) approval has an expiry date and continuing-review
requirements. A capstone that runs longer than the approval period must schedule the renewal,
and renewal takes weeks. The plan must include the approval date, the expiry date, and the
renewal task if the project crosses it.

A data-use agreement (DUA) often specifies a data destruction date, an annual renewal, and a
requirement to report suspected protected health information (PHI) incidents within a stated
period. These are contractual obligations that belong on the calendar, not in an appendix.

Where the work is part of a regulated study, ICH E6 Good Clinical Practice governs the
investigator's obligations, including protocol deviations and record retention. The capstone is
not a trial, but a plan that borrows the discipline of recording deviations will produce a
cleaner change log.

### FDA pre-submission timelines

If a team intends to discuss its approach with the FDA, the Q-Submission (Q-Sub) programme
provides a mechanism, with a published timeline: a pre-submission meeting request is typically
acknowledged within 21 days and a meeting held within 70 days of acceptance, subject to the
programme's current guidance. That timeline is longer than most capstones, which is why a
regulatory conversation is usually a post-course activity. The plan should not pretend
otherwise.

### Why this matters to a twelve-week plan

The external dates that most often collide with a capstone are: IRB approval (two to eight
weeks), DUA execution (two to six weeks), credentialed platform access (one to two weeks), and
institutional compute allocation (one to four weeks). Each is a lag, not an effort. A plan that
models them as effort will be wrong by weeks.

| External process | Typical elapsed time | Effort to the team | Plan treatment |
|---|---:|---:|---|
| CITI training | 4 to 8 hours | 4 to 8 hours | Week 1, before any request |
| Credentialing | 1 to 2 weeks | 1 hour | Lag after training |
| DUA execution | 2 to 6 weeks | 2 to 4 hours | Lag with a chase task |
| IRB determination | 2 to 8 weeks | 4 to 12 hours | Lag with a submission date |
| Compute allocation | 1 to 4 weeks | 2 hours | Request in week 1 |
| Mentor review | 3 to 7 days | 1 hour | Lag after each artefact |
| Conference abstract deadline | Fixed | 8 to 16 hours | Backward from the date |

The table is the argument for starting access work in week 1 even though it produces no model.

### Contractual clocks

Three clocks run in parallel with the project and none of them appears in a Gantt chart.

The credentialing clock starts when the application is submitted and ends when the account is
approved; it cannot be accelerated by working weekends.
The agreement clock starts when the institutional signature process begins and ends when the
counterparty signs; it is often controlled by a legal team with its own queue.
The approval clock starts when the protocol is submitted and ends at the committee meeting;
meeting frequency, not review effort, sets the duration.

A plan that records only the start date of each clock will be surprised by all three. The
practical treatment is to record the expected end date, the name of the person who can give a
status, and a chase interval of no more than seven days.
When a clock is at risk, the escalation is to the programme lead, not to the counterparty, and
the fallback switch date is the decision point.

### Retention and destruction dates

A DUA may require destruction of the extract at a stated date or at the end of the project.
That date is a milestone, and it has an owner. A team that forgets it has a compliance breach
that is invisible until an audit. The plan should show the destruction task in the final week,
with a confirmation record filed in `governance/`.

### What the standards do not cover

No standard tells a team which artefact to cut when it is two weeks behind. That is a judgement
call, and the decision table in Section 5 is the instrument. No standard defines a kill criterion
for a course project. That is a design choice, and its value depends on being written before the
result is known.

## 9. Industry practice and case evidence

### Gates with teeth

Cooper's work on Stage-Gate reports that the process is designed so that a meaningful share of
projects are stopped at each gate, particularly the early ones. The claim is about product
development portfolios, not course projects, and the evidence is largely practitioner
literature rather than controlled trials. The transferable idea is the structure: deliverables
defined at the previous gate, must-meet criteria, and a decision that includes kill.

A capstone with five gates and no kill at any of them is a schedule, not a plan.
The panel's willingness to stop a project is what gives the gate its planning value.

### Reference class forecasting

Flyvbjerg's work on major projects documents systematic cost and schedule optimism and proposes
reference class forecasting: estimate the distribution of outcomes for a class of similar
projects, then adjust the specific estimate toward that distribution. The method is used in
transport and infrastructure appraisal and is referenced in HM Treasury guidance.

The capstone application is modest but real. If the last three course projects with a similar
data-access route took four, five, and six weeks, the plan should not assume two. A team that
records its own historical durations, even informally, plans better in the second project than
in the first.

### What public challenges reveal about timelines

The RSNA 2024 Lumbar Spine Degenerative Classification challenge ran from May to October 2024,
with results announced in November and winners recognised in December. Organisers reported 1,874
teams. The schedule is a seven-month cycle for a single task, with a hidden test phase at the
end. A capstone is twelve weeks for a task plus governance plus a defense.

The George B. Moody PhysioNet Challenge 2025 followed a similar shape: data and rules released
earlier in the year, submissions scored in September, results announced on 24 September 2025.
The organisers report 111 teams and 41 ranked entries. The lesson for planning is that even a
narrow, well-resourced challenge with a fixed metric allows months, not weeks, for a competitive
result.

The MedGemma Impact Challenge compressed the cycle further: announced in January 2026,
winners reported in March 2026, with prototypes rather than validated products as the
deliverable. That is a realistic benchmark for what eight weeks of part-time work can produce:
a narrow prototype with a clear user, not a validation study.

### The evidence gap in deployment timelines

The United Kingdom National Health Service (NHS) AI in Health and Care Award evaluation
literature describes real-world AI evaluations that run for years, not weeks, and reports that
generating evidence during deployment requires planning that starts before procurement. The
Royal College of Radiologists' 2025 consultation document on AI deployment fundamentals makes a
similar point: marking is not evaluation, and evaluation takes time.

A twelve-week capstone cannot produce deployment evidence. It can produce a credible plan for
obtaining it, and a panel can tell the difference when the plan names the design, the duration,
and the endpoint.

### What a twelve-week plan can and cannot contain

The public challenge calendar is a useful calibration for ambition.
A challenge that runs for six months produces a leaderboard entry and, in the best cases, a
methods paper. A twelve-week capstone that runs part-time produces a dossier, a prototype, and a
defense. The two are not the same product, and a plan that tries to match a challenge timeline
in twelve weeks will fail at validation.

What fits in twelve weeks, part-time, with four people:

- one baseline and one candidate, on one dataset, with a temporal holdout;
- calibration and a decision-curve analysis for the chosen threshold;
- three pre-specified subgroups with intervals, and an honest statement about precision;
- twenty to thirty reviewed errors with a taxonomy;
- a regulatory triage memo and a model card;
- a demonstration on held-out data, recorded as a backup.

What does not fit, and should be planned as out of scope:

- external validation at a second institution;
- a prospective or shadow-mode evaluation;
- a reader study with radiologists or nurses;
- a cost-effectiveness model with a literature review;
- a foundation-model comparison with fair compute;
- a regulatory submission of any kind.

The PhysioNet 2026 Challenge illustrates the point from the organiser side. The task, data, and
scoring are released months before the conference, and the hidden validation and test sets are
single sites with deliberate missingness. A team cannot replicate that infrastructure in twelve
weeks; it can learn from the design and copy the discipline of a hidden site.

### Inference, labelled

It is an inference that teams which compute a critical path in week 1 finish on time more often
than teams which do not. No controlled study of capstone teams is cited here, and the claim is
plausible rather than demonstrated. The mechanism is straightforward: the computation makes
schedule risk visible while there is still time to cut scope.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Plan built from activities | Deliverables named as verbs | No testable artefact | Gate reviews become opinion | Rewrite each deliverable with a pass condition |
| Effort confused with duration | Access scheduled as four hours | Lags counted as work | Plan is weeks short in calendar time | Separate effort from elapsed time |
| No critical path | Every task described as important | No distinction between chain and float | Effort spent off the bottleneck | Compute ES, EF, and float |
| Single-point estimates | One number per task, no range | Tail risk hidden | Deadline missed without warning | Three-point estimates and simulation |
| Over-allocation | Weeks 5 to 8 exceed capacity | Parallel workstreams converge | Quality drops, reviews slip | Load effort by week; level resources |
| Gate without authority | Reviewer cannot stop the project | Social pressure to continue | Fatal flaw carried to defense | Name the reviewer and the consequence |
| Kill criteria written late | Criteria appear after results | Rationalisation of what happened | Claim exceeds evidence | Write criteria at G2 |
| No cut order | Argument about priorities in week 9 | Decisions made under stress | Core artefacts dropped instead of stretch | Pre-agree the cut table at G0 |
| Access not chased | No status update in week 2 | External process unattended | Data arrives after the switch date | Chase task with a named owner |
| Hidden work | "Small" tasks absent from the plan | Unplanned effort | Capacity silently exceeded | Log every task over two hours |
| Freeze too late | Freeze within 48 hours of defense | No time for correction | Inconsistent artefacts | Freeze at least seven days before |
| Rehearsals skipped | One read-through | Structural problems found on stage | Weak delivery | Two full rehearsals, three days apart |
| Plan not versioned | Changes agreed verbally | No audit trail | Cannot explain slippage | Dated change log with end-date effect |
| Optimism as method | "We will work harder in week 10" | Capacity assumed elastic | Burnout and errors | Compress by overlap, not by hope |

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| Task durations are independent | Monte Carlo is tractable | A shared cause delays several tasks | Re-run with correlated access and review lags |
| Most-likely estimates are calibrated | Simulation inherits the inputs | Team has no reference projects | Anchor to lab exercises or prior work |
| Capacity is constant | Weekly load is comparable | Exams, clinical rotas, holidays | Use a weekly capacity vector, not one number |
| Review latency is predictable | Gates depend on feedback | Mentor travels or is unwell | Add a delegate and a chase date |
| The critical path is stable | One chain dominates | Early finish changes the chain | Recompute at every gate |
| Scope is fixed after G2 | Plan has a stable target | Sponsor adds a question | Change log with displacement rule |
| The team can say no | Cut order is enforceable | Lead has no authority | Agree the cut table in writing |
| External processes have published service levels | Lags can be estimated | Institutional backlog | Use the pessimistic value and chase |
| One week of float is enough | Buffer absorbs normal variation | Correlated slips | Simulate; require 80 percent deadline probability |
| Gates are held as scheduled | Decisions happen on time | Panel availability | Default action and escalation rule |

The deepest limit is that a plan cannot manufacture evidence.
If the cohort has too few events, a well-planned twelve weeks still produces a wide interval.
Planning improves the probability of finishing with an honest result; it does not change what
the data can support.

## 12. Exercises

### Exercise 1. Compute a critical path (calculation, 30 minutes)

Using the task table in Section 7, change `candidate` to depend on `data_audit` and rerun the
critical path code. Report the new end date and the new float for baseline and validation.
**Deliverable:** code output and a three-sentence interpretation.
**Criteria:** correct end date; float values correct; interpretation identifies the new
critical chain.
**Time:** 30 minutes.

### Exercise 2. Write testable gate deliverables (application, 40 minutes)

Take the G2 gate and write three deliverables, each with a pass condition a reviewer can test
in ten minutes.
**Deliverable:** a table with artefact, test, and reviewer.
**Criteria:** every test is observable without asking the team a question; at least one test
involves running code.
**Time:** 40 minutes.

### Exercise 3. Load a plan against capacity (calculation, 45 minutes)

For your own capstone, list weekly hours per role and compute the load for weeks 1 to 12.
Identify every week above capacity.
**Deliverable:** a week-by-role table and a levelling proposal.
**Criteria:** capacity reflects real availability, not forty hours; the proposal moves work
rather than assuming extra hours.
**Time:** 45 minutes.

### Exercise 4. Design a kill criterion (design, 30 minutes)

Write one effect criterion and one feasibility criterion for a capstone of your choice, each
with a threshold, a measurement date, and an owner.
**Deliverable:** two sentences plus a note on what happens if triggered.
**Criteria:** thresholds are numeric; measurement date precedes freeze; the consequence is a
decision, not a discussion.
**Time:** 30 minutes.

### Exercise 5. Pre-mortem (critique, 45 minutes)

Assume the capstone has failed at the defense. List eight causes, rank them by plausibility, and
give each an early signal and a response.
**Deliverable:** a ranked table.
**Criteria:** at least three causes are organisational rather than technical; every cause has a
detectable signal.
**Time:** 45 minutes.

### Exercise 6. Compress a plan without optimism (design, 50 minutes)

Take a plan whose deterministic end exceeds the deadline by three weeks. Produce a revised plan
that fits by overlapping or splitting work, and show the new critical path.
**Deliverable:** before-and-after dependency tables and the arithmetic.
**Criteria:** no task duration reduced below its optimistic estimate; the revised graph is
acyclic; the new end date is computed, not asserted.
**Time:** 50 minutes.

### Exercise 7. Model an access slip (calculation, 40 minutes)

Rerun the Monte Carlo with `data_access` set to (3.0, 4.0, 6.0). Report the probability of
meeting week 12 and decide whether the fallback should be activated in week 3.
**Deliverable:** output plus a 150-word decision memo.
**Criteria:** probability reported; decision tied to a threshold agreed in advance; fallback
switch date named.
**Time:** 40 minutes.

### Exercise 8. Write the cut table (design, 35 minutes)

For your capstone, fill in the cut table for one, two, three, and four weeks behind, and name
the person who decides at each level.
**Deliverable:** a four-row table.
**Criteria:** each cut is specific enough to execute; the never-cut column is justified; the
decider has authority.
**Time:** 35 minutes.

## 13. Self-check answers

**1. Why is the deterministic end date computed before the simulation?**
Because it answers a different question. The deterministic date asks whether the plan can fit at
all under central estimates; if it exceeds the deadline, uncertainty cannot rescue it. The
simulation then quantifies the risk conditional on the plan being feasible.

**2. What does a float of 5.5 weeks on the governance task mean?**
It means governance can start up to 5.5 weeks later, or slip by that amount, without delaying
the project end. It also means that speeding governance up does not shorten the project, because
it is not on the critical path.

**3. Why did compressing the candidate task help even though its duration increased?**
Because the end date depends on the path, not on the task. Starting the candidate two weeks
earlier, at the close of the data audit, removed two weeks from the critical chain even though
the task itself became half a week longer.

**4. Compute the weekly load in week 8 for the worked example.**
The reported peak is 65.8 hours against a capacity of 45, an overload of 20.8 hours, or 46
percent. The exact figure depends on how effort is spread within each package; the code in
Section 7 reproduces 65.8 under even spreading.

**5. Why does a two-week access slip reduce the deadline probability to 2.6 percent?**
Because access is on the critical path and the compressed plan has only one week of float. A
two-week slip consumes the float and adds a week, and the remaining tasks have right-tailed
durations. The simulation shows the compound effect: the mean moves to 13.15 weeks.

**6. What is the difference between a kill criterion and a stopping rule for model iteration?**
A kill criterion concerns the claim or the project and is set before results. A stopping rule
concerns iteration and limits how many tuning runs are attempted. The first protects against
sunk cost; the second protects the schedule.

**7. Why is a gate without a named reviewer not a gate?**
Because a decision requires authority. Without a named reviewer with the power to fail the
deliverable, the gate becomes a progress report, and a fatal flaw proceeds to the next stage.

**8. What should a team do when a member drops out?**
Re-plan the same day, remove that role's packages from the critical path, reassign at most one,
and cut the rest. The cut should come from the stretch tier first and from the system layer
second, never from validation or governance.

**9. Why schedule two rehearsals rather than one?**
The first rehearsal reveals a structural problem, and the fix needs a second run to verify. Two
rehearsals also test the demo twice, which is the minimum for catching an environment failure.

**10. Rubric for Exercise 6.** A strong answer changes the graph, shows the new critical path,
and keeps every duration at or above its optimistic value. A weak answer shortens three tasks by
20 percent and declares the plan feasible, which is optimism with arithmetic attached.

## 14. Interview questions

**Q1. "How did you plan the twelve weeks?"**
Strong answer describes artefacts, dependencies, the critical path, and the point at which scope
was cut, with a number. Weak answer describes a list of weekly topics.

**Q2. "What was your critical path, and how did you know?"**
Strong answer names the chain and the float on the parallel branches. Weak answer says "the
model was the hard part".

**Q3. "How did you handle uncertainty in task duration?"**
Strong answer uses three-point estimates and a simulation, reports a percentile, and states the
assumptions. Weak answer says "we added buffer".

**Q4. "What did you cut, and when?"**
Strong answer names the artefact, the week, and the effect on the claim. Weak answer says
nothing was cut because the team worked harder.

**Q5. "What would have made you stop?"**
Strong answer gives a numeric criterion and the decision it would trigger. Weak answer says
"if the results were bad", which is not a criterion.

**Q6. "How did you handle a dependency outside your control?"**
Strong answer describes the access request, the chase, the fallback, and the switch date. Weak
answer says the data arrived late and the team coped.

**Q7. "Why twelve weeks and not eight?"**
Strong answer maps the weeks to artefacts and gates, and identifies which weeks cannot compress.
Weak answer says the programme required it.

**Q8. "How did you know the plan was still on track?"**
Strong answer describes weekly deliverable burn-down and gate decisions, not hours worked. Weak
answer says the team met every week.

**Q9. "What is a gate for?"**
Strong answer describes a decision with criteria and a kill option. Weak answer describes a
presentation to supervisors.

**Q10. "What did you learn about estimation?"**
Strong answer names a specific estimate that was wrong, by how much, and what the team changed
in response. Weak answer says estimation is hard.

## 15. Key terms

- **Buffer** — calendar slack reserved for known-unknowns.
- **Burn-down** — remaining deliverables plotted against weeks.
- **Capacity** — person-hours available per week across the team.
- **Change control** — the rule for altering scope, schedule, or metric after G2.
- **Critical path** — the longest dependency chain; its length is the project duration.
- **Deliverable** — a versioned artefact with a pass condition.
- **Dependency** — a finish-to-start relationship between work packages.
- **DUA** — data-use agreement.
- **Effort** — person-hours required to complete a task.
- **Float** — the amount a task can slip without delaying the project.
- **Freeze** — the date after which only factual corrections are permitted.
- **Gate** — a decision point with deliverables, criteria, and a reviewer.
- **Gate criteria** — readiness, must-meet, and should-meet tests.
- **Go / kill / hold / recycle** — the four Stage-Gate decisions.
- **IRB** — institutional review board.
- **Kill criterion** — a numeric, pre-specified result that ends the project or the claim.
- **Lag** — forced waiting time between tasks.
- **Monte Carlo schedule** — repeated sampling of durations to produce an end-date distribution.
- **Over-allocation** — planned effort above capacity in a week.
- **PHI** — protected health information.
- **Pre-mortem** — imagining failure and listing causes with signals.
- **Q-Sub** — FDA Q-Submission programme for pre-submission interaction.
- **Reference class forecasting** — estimating from similar past projects.
- **Rehearsal** — a full timed run of the defense.
- **Re-planning trigger** — an event that forces the plan to be rebuilt.
- **Stage-Gate** — a stage-and-gate product development process model.
- **Three-point estimate** — optimistic, most likely, and pessimistic durations.
- **Total float** — the slip allowance before the project end moves.
- **Work package** — a unit of work with one owner, one artefact, and a duration.

## 16. Sources

All URLs accessed 26 October 2026.

- Cooper RG. The Stage-Gate idea-to-launch system. In: *Wiley Encyclopedia of Marketing*, 2010.
  https://onlinelibrary.wiley.com/doi/full/10.1002/9781444316568.wiem05014
- Cooper RG. Perspective: the Stage-Gate idea-to-launch process. *Journal of Product Innovation
  Management* 2008;25(3):213-232. https://doi.org/10.1111/j.1540-5885.2008.00296.x
- ISO 21500, *Project, programme and portfolio management — Context and concepts*.
  https://www.iso.org/standard/74952.html
- ISO 21502, *Project, programme and portfolio management — Guidance on project management*.
  https://www.iso.org/standard/74953.html
- Flyvbjerg B. From Nobel Prize to project management: getting risks right. *Project Management
  Journal* 2006;37(3):5-15. https://doi.org/10.1177/875697280603700302
- ICH E6(R2), *Guideline for Good Clinical Practice*. https://www.ich.org/
- FDA, *Q-Submission Program* guidance.
  https://www.fda.gov/regulatory-information/search-fda-guidance-documents/requests-feedback-and-meetings-medical-device-submissions-q-submission-program
- PhysioNet, required training and data-use agreement for MIMIC-IV.
  https://physionet.org/content/mimiciv/view-required-training/3.0/
- RSNA, Lumbar Spine Degenerative Classification AI Challenge, 2024.
  https://www.rsna.org/artificial-intelligence/ai-image-challenge/lumbar-spine-degenerative-classification-ai-challenge
- Reyna MA, Koscova Z, Pavlus J, et al. Detection of Chagas disease from the ECG: the George B.
  Moody PhysioNet Challenge 2025. arXiv:2510.02202. https://arxiv.org/abs/2510.02202
- Google Research, Announcing the winners of the MedGemma Impact Challenge, March 2026.
  https://blog.google/innovation-and-ai/technology/health/med-gemma-impact-challenge/
- NHS England, Planning and implementing real-world AI evaluations: lessons from the AI in
  Health and Care Award.
  https://www.england.nhs.uk/long-read/planning-and-implementing-real-world-ai-evaluations-lessons-from-the-ai-in-health-and-care-award/
- Royal College of Radiologists, *AI deployment fundamentals*, public consultation, 2025.
  https://www.rcr.ac.uk/media/r0th2x0k/ai-deployment-fundamentals_public-consultation.pdf
- Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement. *BMJ* 2024;385:e078378.
  https://doi.org/10.1136/bmj-2023-078378

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic: [`01_capstone-architecture.md`](01_capstone-architecture.md)
- Next topic: [`03_team-roles.md`](03_team-roles.md)
- Related topic: [`09_deliverables.md`](09_deliverables.md)
- Survey file: [`../../course/19_capstone_hackathons.md`](../../course/19_capstone_hackathons.md)
- Labs: [`../../labs/`](../../labs/)
