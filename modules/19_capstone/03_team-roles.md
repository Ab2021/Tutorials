# Team Roles

> **Module 19 — Capstone, Hackathons, and Competitions**
> Topic 03 of 11
> Estimated study time: 3 hours
> Prerequisites: Topic 01
> Version 2026.3

## 1. Why this topic exists

A capstone is assessed as a team and defended as individuals.
That combination is unusual, and it exposes a problem that team projects usually hide: work can
be distributed, but accountability cannot.
Every artefact in the dossier needs one person who is answerable for it, one person who can
explain it under questioning, and at least one other person who could rebuild it if the first
were unavailable.

Teams fail this in predictable ways.
The clinical question is nobody's job, so a valid model answers the wrong question.
The data provenance artefact is nobody's job, so the dossier cannot say where the numbers came
from.
One person writes 80 percent of the code, so the individual defense collapses when that person
is asked about the regulatory memo.
A team of four behaves like a team of two plus two observers, and the peer adjustment at the end
is the first time anyone says so.

Industry calls the underlying discipline role definition, and it is not bureaucratic.
A hospital that deploys a sepsis alert without naming a clinical owner has not deployed an alert;
it has installed a notification.
A vendor that cannot name the accountable person for post-market surveillance has a regulatory
problem, not an organisational one.
A data science team that cannot say who owns the label definition will discover, usually after
deployment, that two departments were measuring different things.

The evidence on teams is consistent on one point and vague on many others.
Studies of hackathon projects find that team composition predicts which projects continue:
Nolte, Chounta, and Herbsleb, in a 2020 analysis of hackathon projects, report that skill
diversity and intent to continue predict longer-term continuation, while intense short-term
activity predicts lower continuation.
The finding is about hackathons, not capstones, and it should be read as a hypothesis rather
than a law. It is consistent with the practical observation that teams of clones finish fast and
stop early.

This topic gives the team a structure: five role functions, a decision-rights table, an
accountability assignment for every artefact, and a weekly rhythm that surfaces overload before
it becomes failure.
It also gives the team a way to survive losing a member, which is the most common mid-project
event and the one most teams have no plan for.

There is an assessment reason as well.
The programme applies a confidential peer adjustment of at most plus or minus 15 percent, and an
individual defense that can cover any part of the work.
Both mechanisms assume that contribution and understanding are separable from attendance.
A team that has not tracked contribution will experience the adjustment as a surprise, and a
team that has siloed knowledge will experience the defense as an ambush.

Neither is necessary.
A contribution log started in week 1 makes the distribution visible while it can still be
changed, and a rotation of artefact walkthroughs in the weekly demo makes understanding
distributed by design.
The role structure is what makes both routines possible.

Industry teams face the same problem with higher stakes.
A model that reaches a hospital without a named clinical owner is a pilot, not a deployment; a
model without a named monitoring owner is a liability that nobody is watching.
The capstone's scale is smaller, but the structure it teaches is the same one that distinguishes
a team that ships from a team that demonstrates.

## 2. Learning outcomes

By the end of this topic you can:

1. **Map** every dossier artefact to one accountable role and at least one backup, and identify
   any artefact with no owner or two owners.
2. **Build** a RACI matrix (responsible, accountable, consulted, informed) for a five-person
   team and test it for the two common errors: no accountable person, and an accountable person
   with no capacity.
3. **Compute** weekly demand per role against available hours, and distinguish aggregate
   utilisation from role-level bottlenecks.
4. **Design** a handoff with an acceptance test, so that an artefact changes hands without
   losing its provenance or its open questions.
5. **Assign** decision rights for scope, metric, data access, and claim language to named roles.
6. **Plan** for the loss of a team member, including which artefacts are dropped and which are
   reassigned.
7. **Run** a weekly rhythm and a midpoint role review that produce a written record rather than
   a conversation.
8. **Defend** any part of the work, not only your own, in an individual defense.

## 3. The core mental model

A team is a set of roles, a set of artefacts, and a set of decision rights.
The roles are functions, not people: one person may hold two functions in a team of three, and
one function may be shared between a team member and an external mentor.
The artefacts are the deliverables in the dossier.
The decision rights are the small number of choices that cannot be delegated without losing
coherence.

```text
                       DECISION RIGHTS
        scope | metric | data access | claim language
          |        |          |             |
          v        v          v             v
   +--------------------------------------------------+
   |  LEAD / PRODUCT OWNER   schedule, scope, dossier |
   +--------------------------------------------------+
        |                    |                  |
        v                    v                  v
   DATA ENGINEER        MODELER           EVIDENCE LEAD
   pipeline, lineage    baseline,         design, regulatory,
   quality, splits      validation        equity, risk
        |                    |                  |
        +---------+----------+---------+--------+
                  v                    v
            CLINICAL LIAISON      PEER / MENTORS
            workflow, labels,     review, rehearsal
            error review          (no decision rights)

   Every artefact: exactly one A, at least one backup, named C and I.
```

Read the diagram from the decision rights downward.
The four rights are the irreducible core of the project. If they are dispersed without a rule,
the project produces artefacts that contradict each other.
If they are concentrated in one person, the project becomes a solo project with helpers, and the
individual defense exposes it.

The roles are ordered by the layer they own, which connects this topic to the architecture in
Topic 01.
The lead owns layer 1 and the schedule.
The modeler owns layer 2, with the evidence lead.
The data engineer owns layer 3 and the provenance that layer 2 depends on.
The evidence lead owns layer 4 and the analysis plan that constrains layer 2.
The clinical liaison owns the meaning of the label and the workflow, which sit across layers 1
and 2.

Three properties of the model matter in practice.

**Accountability is singular.** For every artefact, exactly one role is accountable. Others may
be responsible for producing parts, consulted on content, or informed of the result, but the
accountable role signs. Two accountable roles means nobody is accountable; zero means the
artefact appears in the dossier only if someone remembers.

**Capacity is per role, not per team.** A team can have 66 percent aggregate utilisation and
still be unable to deliver, because the modeler is at 143 percent in weeks 5 to 8 while the
clinical liaison is idle. Aggregate numbers hide the bottleneck, and the bottleneck sets the
date.

**The bus factor is an artefact property.** The question is not "how many people know the
model" but "which artefact becomes unowned if this person is removed". A team whose lead is
accountable for three of seven artefacts has a bus factor of one for those three, regardless of
how well the others understand them.

A fourth property is social rather than structural.
Roles do not remove conflict; they locate it.
A disagreement about the metric is a disagreement between the clinical liaison and the evidence
lead, resolved by the lead with a recorded decision. Without roles, the same disagreement
becomes personal, and it resurfaces at the defense as an inconsistency.

### Why five functions and not five people

The functions are defined by what must be owned, not by headcount.
In a team of four, the evidence lead also carries the product-owner function for the analysis
plan, and the lead carries the clinical function until a liaison is recruited.
In a team of three, the data engineer and modeler usually merge, which is workable only if the
pipeline is simple and the system layer is reduced to a recorded walkthrough.

Merging functions has a predictable cost: the person holding two functions switches between them
and loses context at each switch. The cost is small when the functions share artefacts (data
engineering and modelling share the feature table) and large when they do not (evidence and
engineering share almost nothing).
A team that merges unrelated functions should expect a 15 to 25 percent productivity loss on the
merged role, which is why the capacity table uses hours rather than headcount.

### The artefact is the unit of ownership

Roles are abstractions; artefacts are what the panel reads.
That is why every discussion in this topic returns to the artefact table.
An owner without an artefact is a title. An artefact without an owner is a gap.
A backup without a handoff note is a name on a slide.

The practical test is simple: open the dossier folder, and for each file, ask who would answer a
question about it at 9 a.m. tomorrow. If the answer is "probably the modeler", the artefact has
no owner. If the answer is two names and neither has prepared, it has two.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Role | A function with defined responsibilities and decision rights | A job title or a person's name | Role label |
| RACI | Responsible, accountable, consulted, informed matrix | Used with two accountable entries | One A per row |
| Accountable | The single role that signs an artefact and answers for it | Confused with "does most of the work" | One per artefact |
| Responsible | The role that produces the work | Treated as accountability | One or more per artefact |
| Consulted | Roles whose input is sought before the artefact is final | Everyone copied on email | Named roles |
| Informed | Roles told of the outcome | Used to avoid decisions | Named roles |
| Decision right | Authority to make a specific choice without further approval | Assumed to follow seniority | Named choice and role |
| Handoff | Transfer of an artefact with an acceptance test | Sending a file | Artefact and test |
| Capacity | Available hours per week for a role | Contracted hours | Hours per week |
| Demand | Planned hours per week for a role | Total project hours | Hours per week |
| Utilisation | Demand divided by capacity | Reported as team average only | Percentage per role |
| Bottleneck | The role whose demand exceeds capacity first | The slowest person | Role and weeks |
| Bus factor | Number of people who must be lost before an artefact is unowned | General redundancy | Integer per artefact |
| Single point of failure | A role or artefact with no backup | Any important task | Artefact and role |
| Free riding | Contributing less than the agreed share without disclosure | Any unequal contribution | Hours and artefacts |
| Contribution log | Weekly record of hours and artefacts per person | A timesheet for billing | Hours and links |
| Peer adjustment | Confidential multiplier applied to a team grade, bounded | A popularity vote | Multiplier range |
| Individual defense | Oral examination on any part of the work | A presentation by the owner | Questions and score |
| Clinical liaison | The role that owns workflow, label meaning, and error review | A sponsor who attends once | Named clinician and hours |
| Data steward | The role that owns access, lineage, and quality | The person who downloaded the file | Named role |
| Evidence lead | The role that owns design, regulatory triage, and equity | The statistician | Named role |
| Product owner | The role that owns scope, schedule, and the decision log | The loudest person | Named role |
| Escalation | The route to a decision when a blocker exceeds a threshold | Waiting for the next meeting | Role and response time |
| Weekly rhythm | The fixed cadence of stand-up, studio, and demo | Ad hoc meetings | Day, duration, purpose |
| Midpoint review | A structured reassessment of roles and scope at week 6 or 7 | A progress update | Decision record |
| T-shaped | Deep in one discipline, conversant across several | A generalist | Depth and breadth |
| Onboarding | Bringing a mentor or new member to the point of useful contribution | Sending the repository link | Brief and first task |
| Conflict of interest | A personal interest that could bias a judgement | Any disagreement | Declaration and management |
| Authorship | Credit for intellectual contribution, under ICMJE criteria | Listing everyone who attended | Contribution categories |
| Rotation | Scheduled delivery of the weekly walkthrough by a non-owner | Everyone presents their own part | Week and presenter |
| Knowledge transfer | The written note that lets a backup act | A verbal briefing | Decisions, open questions, inputs |
| Ownership record | A signed list of artefacts and their accountable roles at freeze | The team grade | Artefact and signature |
| Conflict of interest | A personal interest that could bias a judgement | Any disagreement | Declaration and management |

Two distinctions prevent most RACI arguments.
Accountable is not the same as responsible. A lead can be accountable for the value case while
the evidence lead writes it; the lead signs, and the evidence lead is consulted.
Informed is not the same as consulted. Copying someone on a decision does not make them
consulted, and consulting someone does not give them a veto.

A third distinction is specific to capstones.
The programme's clinical mentor and the team's clinical liaison are often the same person, but
the roles differ. The mentor advises the team and has no accountability; the liaison owns the
label meaning and holds a veto on the claim. When one person holds both, the team must say which
hat is being worn at each meeting, or the accountability becomes ambiguous exactly where it
matters.

## 5. The mechanics, step by step

The procedure below takes a team from a list of names to a working role structure in about two
hours, and it is revisited at the midpoint.
It produces four artefacts: a RACI matrix, a capacity table, a decision-rights table, and a
contribution log template.

### Step 1. Size the team against the work

**Input:** the artefact list from Topic 01 and the available people.

**Output:** a team size and a statement of which roles are held by team members and which by
mentors.

**Rationale:** The default is four. A team of three merges the evidence lead into the lead role
and reduces the archetype ambition. A team of five splits engineering from modelling, which is
useful when the system layer is substantial.
The size is a constraint on scope, not a status symbol.

**Branch point:** if the team has fewer than three people, the capstone should be an evaluation
or benchmark project (archetype A5), because the clinical and governance work cannot be covered
honestly.

### Step 2. List every artefact and its acceptance test

**Input:** the thirteen-artefact dossier.

**Output:** a table with artefact, acceptance test, and planned version.

**Rationale:** Roles are assigned to artefacts, not to activities. An artefact with no acceptance
test cannot be assigned, because nobody can tell when it is done.

**Branch point:** if an artefact has no test, write one before continuing. This is the same rule
as the gate deliverables in Topic 02.

### Step 3. Assign one accountable role per artefact

**Input:** the artefact table and the role list.

**Output:** a first-pass RACI with exactly one A per row.

**Rationale:** Singular accountability is the property that makes the matrix useful. It also
forces a conversation about who is actually going to sign, which is often different from who is
going to write.

**Branch point:** if two roles both seem accountable, the artefact is probably two artefacts.
Split it, or assign the A to the role that will defend it at the panel.

### Step 4. Assign a backup for every critical artefact

**Input:** the RACI.

**Output:** a named backup for each artefact that would block the defense if its owner vanished.

**Rationale:** The backup is not a co-owner. It is the person who can rebuild or explain the
artefact at short notice. Naming the backup also creates a reason for a handoff, which is where
most knowledge transfer actually happens.

**Branch point:** if no backup exists for an artefact, that artefact is a single point of
failure and belongs in the risk register with a mitigation.

### Step 5. Write the decision-rights table

**Input:** the charter and the four decision types.

**Output:** a table naming, for scope, metric, data access, and claim language, who decides, who
must be consulted, and who is informed.

**Rationale:** Most project conflict is a disagreement about who decides, disguised as a
disagreement about content. Writing the table in week 1 converts a political problem into a
procedural one.

**Branch point:** if the clinical liaison is external and part-time, the claim language decision
must still be theirs to veto. A claim a clinician will not sign is not a claim the team can
make.

### Step 6. Compute demand per role by week

**Input:** the work packages from Topic 02 and the role assignment.

**Output:** a matrix of hours per role per week.

**Rationale:** The matrix exposes bottlenecks that a total-hours calculation hides. It also
gives the lead a factual basis for reallocation, which is easier to accept than an opinion about
who looks busy.

**Branch point:** if a role exceeds capacity in more than three weeks, the plan needs a scope
cut, not a motivational conversation.

### Step 7. Compare demand with real availability

**Input:** the demand matrix and each person's actual hours.

**Output:** utilisation per role per week, and a list of over-capacity weeks.

**Rationale:** Real availability is rarely forty hours. A student with two other modules may have
twelve; a clinician may have three; a lead with a job may have eight.
Using contracted hours instead of available hours produces a plan that only works in theory.

**Branch point:** if a person's availability changes during the project, update the matrix and
re-run the comparison. A plan built on stale availability is a plan for a different team.

### Step 8. Level the load without inflating capacity

**Input:** the over-capacity weeks.

**Output:** a revised assignment with at least one of: work moved to a week with slack, work
moved to a role with slack, or scope cut.

**Rationale:** Levelling is the only honest response to overload. Working evenings is not
levelling; it is borrowing capacity from the future at a high interest rate, and it usually
appears as errors in the validation week.

**Branch point:** if levelling would move work off the critical path and delay the end date, cut
scope instead. A delayed end date is worse than a narrower claim.

### Step 9. Define handoffs with acceptance tests

**Input:** the dependencies between roles.

**Output:** for each handoff, the artefact, the sender, the receiver, the acceptance test, and
the date.

**Rationale:** Most rework comes from handoffs without tests. The data engineer hands over a
feature table; the modeler discovers that the time window is wrong. A test such as "one row per
admission, all features available at time zero, missingness below 20 percent" catches the error
at the handoff.

**Branch point:** if a handoff cannot be tested, it is not a handoff; it is a conversation, and
it should be replaced by a written artefact.

### Step 10. Set the weekly rhythm

**Input:** the team's availability and the mentor cadence.

**Output:** a calendar with a 30-minute Monday stand-up, a 120-minute Wednesday studio, a
45-minute Friday demo, and a written status by Friday 17:00.

**Rationale:** The rhythm is the mechanism that makes the plan observable. The Monday stand-up
surfaces blockers, the Wednesday studio gets expert critique while there is time to act, and the
Friday demo forces working software rather than slides.

**Branch point:** if a team member cannot attend a fixed slot, they send a written update before
it. Attendance is not the point; the record is.

### Step 11. Define escalation thresholds

**Input:** the risk register and the decision-rights table.

**Output:** a rule such as "any blocker that threatens a gate by more than three days is
escalated to the programme lead within 24 hours, with two options".

**Rationale:** Escalation without a threshold becomes either constant or never. The threshold
makes it a procedure. The requirement to bring options prevents escalation from becoming
complaint.

**Branch point:** if the escalation route goes to someone without authority to change scope or
access, add the person who has it.

### Step 12. Start the contribution log in week 1

**Input:** the role assignments and the artefact list.

**Output:** a shared log with week, person, hours, artefact link, and a one-line description.

**Rationale:** The log protects against two failures at once: free riding that nobody notices
until the peer adjustment, and the perception of free riding when contribution is simply
uneven. It is evidence, not surveillance, and it should be visible to the whole team.

**Branch point:** if a member's logged hours fall below half the team median for two consecutive
weeks, the lead has a conversation in week 3, not week 11.

### Step 13. Onboard the clinical liaison and mentors

**Input:** the one-page brief template.

**Output:** a 30-minute onboarding call, a written brief, and a first specific ask.

**Rationale:** A clinician who is sent a repository link will not engage. A clinician who is
asked a specific question ("which four patients would you assess first, and what would make you
ignore this list?") will.
The first ask should take under fifteen minutes and produce something the team can use.

**Branch point:** if the liaison cannot commit to a weekly slot, reduce their role to two
reviews (G3 and G5) and record the reduced clinical input in the limitations.

### Step 14. Run the midpoint role review

**Input:** the contribution log, the capacity matrix, and the gate record.

**Output:** a written reallocation decision: who owns what for weeks 7 to 12, which artefact is
dropped, and which risk is accepted.

**Rationale:** Week 6 or 7 is the last point at which reallocation is cheap. After that, the
validation and dossier work is fixed and a change costs a gate.

**Branch point:** if the team is behind by more than two weeks at the midpoint, apply the cut
table from Topic 02 rather than redistributing the same work among fewer hours.

### Step 15. Close out roles at freeze

**Input:** the frozen artefacts and the contribution log.

**Output:** a signed ownership record for each artefact, a peer adjustment with written
justification if the programme uses one, and a short lessons note.

**Rationale:** The close-out is what makes the individual defense fair. Each person knows what
they are answerable for, and the panel knows whom to ask.

**Branch point:** if a peer adjustment is contested, the program lead reviews the written
justification and the contribution log. The adjustment is not a vote.

### Step 16. Review the structure at G5

**Input:** the ownership record and the handoff notes.

**Output:** a final RACI showing who signed what, and a list of artefacts whose backup never
exercised the handoff.

**Rationale:** A backup who has never run the pipeline is a name, not a mitigation. The G5
review tests one handoff per critical artefact by asking the backup to reproduce a number or
explain an open question. The test takes ten minutes and is the only evidence that redundancy
exists.

**Branch point:** if a backup cannot pass the test, record the artefact as a single point of
failure in the limitations rather than claiming resilience the team does not have.

### The weekly rhythm in detail

Monday stand-up, thirty minutes, three questions per person: what artefact moved, what is
blocked, what decision is needed. No status narration, no demonstrations.
The output is a blocker list with owners and dates, appended to the decision log.

Wednesday studio, two hours, one artefact examined in depth by a mentor or by a peer from
another team. The artefact is circulated 24 hours ahead. The examination follows the acceptance
test, not the presenter's narrative, which prevents a polished talk from substituting for a
complete artefact.

Friday demo, forty-five minutes, working software only. The rule is that the demo runs on the
held-out split and shows one failure case. A team that has nothing to demo has either no working
software or a plan that has drifted from the artefacts, and both are visible in week 2 rather
than week 10.

Friday status, written by 17:00, five lines: decision this week, artefacts done, evidence
status, blocker, ask. The status is short because it is read, and it is written because the
mentor will not remember a conversation.

### Reading a contribution log

The log is not a timesheet. It records hours, artefacts, and a one-line description, and it is
read for three patterns.

A falling trend in one person's hours over three weeks is usually a capacity problem rather
than a motivation problem. The conversation should ask what changed, not why effort dropped.

A stable high share in one person is a bus-factor problem even when the person is willing. The
response is to transfer an artefact, not to praise the effort.

A mismatch between hours and artefacts, in either direction, is worth a conversation. Someone
logging many hours with no artefact may be doing coordination or rework that the plan does not
recognise. Someone logging few hours with several artefacts may be claiming credit for work
produced by others, or may be working efficiently.

### Decision table: role assignment by team size

| Team size | Lead | Data | Model | Evidence | Clinical | Consequence |
|---:|---|---|---|---|---|---|
| 3 | Member A | Member B | Member B | Member A | External mentor | Narrower archetype; no system layer |
| 4 | Member A | Member B | Member C | Member D | External mentor | Default; evidence lead carries governance |
| 5 | Member A | Member B | Member C | Member D | Member E or mentor | Engineering split from modelling |
| 2 | Member A | Member A | Member B | Member A | External mentor | Evaluation project only |
| 6 or more | Member A | Member B | Members C and D | Member E | Mentor | Coordination cost exceeds benefit |

The last row is not a joke. A six-person capstone team spends more hours in coordination than a
five-person team gains in capacity, unless the work genuinely splits into two independent
streams with separate artefacts.

### Edge cases and forced decisions

**A member drops out.** Re-run the RACI the same day. Every artefact they were accountable for
gets a new A or is dropped from the dossier. The rule is: reassign at most one, drop the rest.
A team that reassigns four artefacts to three people has not solved the problem; it has moved it
to the validation week.

**The clinical liaison withdraws.** The decision statement loses its owner, which is a G0
failure discovered late. The options are a replacement clinician, a documented need assessment
plus a review-only clinician, or a change of archetype to evaluation science. Do not invent
clinical judgement and label it expert review.

**Two members cannot work together.** Separate them by artefact, not by mediation. Give each
accountability for disjoint artefacts and route all handoffs through written acceptance tests.
If the conflict concerns the metric or the claim, the lead decides and records it.

**The lead is the bottleneck.** This is common when the lead is also the strongest modeller.
The fix is to transfer one of the lead's accountable artefacts to another member, not to ask the
lead to delegate tasks while keeping accountability. Accountability and workload move together
or not at all.

**A member has no visible artefact.** Every role should own at least one artefact that appears
in the dossier. A role with no artefact is either a support function that should be folded into
another role, or a sign that the work was not distributed.

**Remote and part-time members.** Time zones make the Wednesday studio impossible for some
members. Replace attendance with a recorded five-minute update and a written critique, and move
the decision to an asynchronous thread with a deadline. The record substitutes for the room.

**A mentor oversteps.** A mentor who sets scope or writes the dossier has become an unaccountable
team member. The boundary is stated in the onboarding brief: advice, critique, and access, but
no authorship and no scope authority.

## 6. Worked example

A four-person team is building a readmission risk score with a transitional-care programme.
The members are a lead with eight hours per week, a data engineer with twelve, a modeler with
fourteen, and an evidence lead with nine. An intensivist supports the project as clinical liaison
at three hours per week; she is not a team member and does not attend the studio.

### The RACI as first written

The team lists seven artefacts and assigns accountability:
charter to the lead, data provenance to the data engineer, analysis plan to the evidence lead,
validation to the modeler, regulatory memo to the evidence lead, value case to the lead, and
defense to the lead.

The matrix passes the single-accountable check: seven of seven artefacts have exactly one A.
It also reveals an imbalance. The lead is accountable for three artefacts, the evidence lead for
two, and the data engineer and modeler for one each.
That is the first diagnostic: the lead is a single point of failure for the charter, the value
case, and the defense, which are the three artefacts a panel is most likely to probe.

### The capacity arithmetic

The team builds a weekly demand matrix from the work packages. Peak weekly demand by role:
lead 12 hours, data 16, model 20, evidence 12, clinical 4.
Available hours: lead 10, data 12, model 14, evidence 9, clinical 3.
Total demand is 391 hours against 576 hours of capacity, an aggregate utilisation of 68 percent.

The aggregate looks comfortable. The role-level picture is not:

- the modeler exceeds capacity in weeks 5, 6, 7, and 8, peaking at 20 hours against 14, a 43
  percent overload;
- the data engineer exceeds capacity in weeks 3, 4, and 5, peaking at 16 against 12;
- the evidence lead exceeds capacity in weeks 9 and 10, peaking at 12 against 9;
- the clinical liaison exceeds capacity in week 12, when the defense requires four hours and
  three are available;
- the lead is over only in week 12, at 12 against 10.

This is the second diagnostic, and it is the one that matters: eight role-weeks are over
capacity, and all of them fall in the four weeks before a gate.
A team that reports only aggregate utilisation will describe itself as two-thirds loaded while
the modeler is the reason the project is late.

### The reallocation

The team makes three changes at the week-3 review, before the overload arrives.

First, it removes the foundation-model comparison from the candidate work. That is a stretch
item, and removing it cuts the modeler's demand in weeks 5 to 8 from 20, 18, 18, 8 to 14, 14,
14, 8. The modeler is now exactly at capacity, with no slack, which is recorded as a risk.
Second, it moves four hours of defense logistics (recording, room, backup device) from the lead
to the evidence lead in week 12, and adds two hours of clinical review time in week 12 by
agreeing a longer slot with the intensivist in advance.
Third, it accepts a residual overload of two hours per week for the data engineer in weeks 3 to
5 and for the evidence lead in weeks 9 and 10. That is ten hours of overload in total, recorded
in the risk register with a named mitigation: the data engineer will not attend the Wednesday
studio in weeks 4 and 5, and the evidence lead will draft the memo before week 9.

After reallocation, the only remaining overload is the accepted residual. Total demand falls to
roughly 355 hours, and the peak modeler week is 14 against 14.

### What the team did not do

The team did not ask anyone to work evenings. It did not reassign the lead's three artefacts,
because there was no one with the context to take them, and it did not pretend that the clinical
liaison could absorb more hours.

That last point is worth stating plainly. The clinical liaison is the scarcest resource in the
project: three hours per week against a demand that peaks at four. A team that plans around the
clinician's availability, rather than assuming it, will design a simpler study and defend it
better.

### The RACI after reallocation

The reallocation changes demand but not accountability, with one exception: the value case moves
its responsible role from the lead to the modeler, while the lead remains accountable.

| Artefact | Accountable | Responsible | Backup | Note |
|---|---|---|---|---|
| Charter | Lead | Lead | Modeler | Unchanged |
| Data provenance | Data engineer | Data engineer | Modeler | Feature table documented |
| Analysis plan | Evidence lead | Evidence lead | Lead | Unchanged |
| Validation | Modeler | Modeler | Data engineer | Pipeline rerunnable |
| Regulatory memo | Evidence lead | Evidence lead | Lead | Unchanged |
| Value case | Lead | Modeler | Evidence lead | Drafted by modeler in week 10 |
| Defense | Lead | All | Evidence lead | Rotating walkthroughs |

The backup column is the one teams omit. Each backup has a one-page note in the artefact folder
containing the decisions made, the open questions, and the location of the inputs. The note takes
twenty minutes to write and saves a day in week 12.

### What the panel asks about roles

Four questions recur, and they are answerable in advance.

"Who decided the metric?" The answer should be a role and a recorded decision, not a consensus.
The panel is testing whether the team can distinguish a decision from an average of opinions.

"Who would maintain this after the course?" The answer should name the operational role and the
monitoring signal they would act on. A team that says "the data science team" has not thought
about adoption.

"Which part did you not understand?" This is asked of a specific individual about a specific
artefact. The answer should describe the gap honestly and the note that closes it, not claim
universal mastery.

"What would you do with one more person?" The answer should name the bottleneck role and the
artefact it would unblock. The wrong answer is "more modelling"; the right answer is usually
evidence or data.

### Variation: the team of three

Suppose the evidence lead leaves in week 2. The team becomes three: lead, data engineer, modeler.

The RACI must change. The lead takes the analysis plan and the regulatory memo; the modeler
takes the value case with the lead consulted; the data engineer keeps provenance. The lead is
now accountable for five of seven artefacts, which is a bus factor of one for most of the
dossier.

The capacity arithmetic changes more than the RACI. Removing nine hours per week of evidence
capacity and adding two artefacts to the lead, who has ten hours, produces an impossible plan.
The team has three options:

| Option | Change | Consequence |
|---|---|---|
| Narrow the claim | Drop the value case and the subgroup analysis; report overall performance | Weaker defense; still passes if the comparator is strong |
| Change archetype | Move to an evaluation study of the existing score | Fits three people; loses the candidate model |
| Recruit | Add a part-time evidence reviewer at four hours per week | Restores capacity; adds onboarding time |

The team chooses the first option and records the exclusion in the charter. The lead's
accountability rises to four artefacts, and the team adds a written backup for each: the modeler
can explain the charter, and the data engineer can explain the analysis plan.

### The arithmetic of the bus factor

Before the change, removing any one person leaves at least one artefact unowned only in the case
of the lead, who holds three. After the change, the lead holds four and the modeler holds two.

The bus factor is not a metaphor here. It is a count: the number of artefacts that become
unowned if a named person is removed. A dossier with a bus factor of one for four artefacts is a
dossier that depends on one person being available in week 12.

The mitigation is not to make everyone an expert. It is to require a written handoff note for
each critical artefact, stored with the artefact, containing the decisions made, the open
questions, and the location of the inputs. That note is what a backup reads at 8 a.m. on the
morning of the defense.

A handoff note has four sections and fits on one page: what the artefact claims, how it was
produced, what is uncertain, and where the inputs live. The third section is the one that
matters at the defense, because it is the section a backup can use to answer a question the
owner never anticipated.

### What the exercise demonstrates

Three things, and each is transferable.

First, aggregate utilisation is a misleading summary. The team was 68 percent loaded and 43
percent over capacity in the same month, and only the second number predicted the delivery date.

Second, the scarce resource is usually clinical attention, not compute. The plan was constrained
by three hours per week of intensivist time, and every design decision that ignored the
constraint had to be reversed.

Third, the bus factor is fixed by writing, not by talking. The team did not become more
redundant because it discussed the risk; it became more redundant when each critical artefact
acquired a one-page note and a named backup who had run it once.

## 7. Code, query, or calculation

The three blocks below check RACI validity, compute role-level capacity, and calculate the bus
factor. They use only the standard library.

### Block 1: RACI validation

```python
artefacts = {
    "charter":         {"lead": "A", "data": "C", "model": "C", "evidence": "C", "clinical": "C"},
    "data_provenance": {"lead": "C", "data": "A", "model": "I", "evidence": "C", "clinical": "I"},
    "analysis_plan":   {"lead": "C", "data": "C", "model": "C", "evidence": "A", "clinical": "C"},
    "validation":      {"lead": "I", "data": "C", "model": "A", "evidence": "C", "clinical": "C"},
    "reg_memo":        {"lead": "C", "data": "I", "model": "I", "evidence": "A", "clinical": "C"},
    "value_case":      {"lead": "A", "data": "I", "model": "C", "evidence": "C", "clinical": "C"},
    "defense":         {"lead": "A", "data": "C", "model": "C", "evidence": "C", "clinical": "C"},
}

bad = [(a, [r for r, v in row.items() if v == "A"])
       for a, row in artefacts.items()
       if sum(v == "A" for v in row.values()) != 1]
print("single-accountable check:", "PASS" if not bad else f"FAIL {bad}")

for role in ("lead", "data", "model", "evidence", "clinical"):
    n = sum(row[role] == "A" for row in artefacts.values())
    print(f"{role:9s} accountable for {n} of {len(artefacts)} artefacts")
```

Expected output:

```text
single-accountable check: PASS
lead      accountable for 3 of 7 artefacts
data      accountable for 1 of 7 artefacts
model     accountable for 1 of 7 artefacts
evidence  accountable for 2 of 7 artefacts
clinical  accountable for 0 of 7 artefacts
```

The clinical liaison holding no accountable artefact is correct in this design: the liaison is a
mentor, not an owner. If the programme expects the clinician to sign the regulatory memo, the
matrix should change.

### Block 2: role-level capacity

```python
demand = {
    "lead":     [6, 6, 5, 5, 5, 5, 6, 6, 8, 8, 10, 12],
    "data":     [2, 6, 14, 16, 14, 10, 8, 6, 4, 4, 3, 2],
    "model":    [0, 0, 4, 10, 16, 20, 18, 18, 8, 4, 2, 2],
    "evidence": [2, 6, 8, 8, 6, 6, 6, 8, 12, 10, 8, 6],
    "clinical": [2, 2, 2, 3, 3, 3, 3, 3, 2, 2, 3, 4],
}
avail = {"lead": 10, "data": 12, "model": 14, "evidence": 9, "clinical": 3}

print(f"{'role':10s}{'demand':>7s}{'avail':>6s}{'peak':>6s}  over-capacity weeks")
for r, v in demand.items():
    over = [i + 1 for i, x in enumerate(v) if x > avail[r]]
    print(f"{r:10s}{sum(v):7d}{avail[r]:6d}{max(v):6d}  {over}")

total = sum(map(sum, demand.values()))
print(f"total demand {total} h vs capacity {sum(avail.values()) * 12} h")
```

Expected output:

```text
role       demand avail  peak  over-capacity weeks
lead           82    10    12  [12]
data           89    12    16  [3, 4, 5]
model         102    14    20  [5, 6, 7, 8]
evidence       86     9    12  [9, 10]
clinical       32     3     4  [12]
total demand 391 h vs capacity 576 h
```

Aggregate utilisation is 68 percent. The modeler is at 143 percent in week 6. Both numbers are
true, and only the second one predicts the delivery date.

### Block 3: bus factor per artefact

```python
people = list(avail)
for p in people:
    lost = [a for a, row in artefacts.items() if row[p] == "A"]
    status = "SINGLE POINT" if lost else "no ownership"
    print(f"remove {p:9s}: {len(lost)} artefacts unowned {lost}  {status}")

fragile = [a for a, row in artefacts.items()
           if sum(v == "A" for v in row.values()) == 1]
print(f"artefacts with exactly one accountable role: {len(fragile)}")
```

Expected output:

```text
remove lead     : 3 artefacts unowned ['charter', 'value_case', 'defense']  SINGLE POINT
remove data     : 1 artefacts unowned ['data_provenance']  SINGLE POINT
remove model    : 1 artefacts unowned ['validation']  SINGLE POINT
remove evidence : 2 artefacts unowned ['analysis_plan', 'reg_memo']  SINGLE POINT
remove clinical : 0 artefacts unowned []  no ownership
artefacts with exactly one accountable role: 7
```

The third line for the clinical liaison is not an error. A mentor with no accountability is
correct, and the matrix should show it rather than inventing a role.

## 8. Data, standards, and regulatory context

Team roles are management practice, but several binding instruments assign responsibility to
named functions, and a capstone that ignores them will produce a dossier that cannot be signed.

### Research governance

An institutional review board (IRB) approval names a principal investigator who is accountable
for the conduct of the study, including protocol deviations, adverse event reporting, and
record retention. In a course capstone, the principal investigator is usually the faculty
supervisor, not the student lead. The team's RACI must reflect that: the supervisor is
accountable to the institution, and the lead is accountable to the programme.
Confusing the two produces a dossier that claims authority the team does not have.

Where the work involves human subjects, the IRB also requires that personnel listed on the
protocol have completed human-subjects training. A team member who joins in week 5 cannot be
added to the protocol without training and an amendment, which takes time. Plan the roster at
the start.

### Device regulation

Under the FDA's clinical decision support guidance, the intended user is part of the intended
use. A submission that does not name the user is incomplete, and a claim that the user will
"independently review the basis" requires that the user be identifiable and capable of doing so.
The clinical liaison is therefore not a nice-to-have; the role is the evidence that a user
exists.

For a device pathway, ISO 13485 assigns responsibility for quality management to top management
and requires defined roles for design control, risk management, and post-market surveillance.
A capstone does not implement ISO 13485, but a regulatory memo that names who would own each
function is more credible than one that lists the standard.

The FDA's predetermined change control plan (PCCP) guidance requires a modification protocol
with defined acceptance criteria and an impact assessment. Someone must own the protocol and the
monitoring that triggers a retrain. In a capstone, that owner is usually the evidence lead; in a
company, it is a regulatory or quality function.

### Privacy roles

Under the GDPR, a data protection officer (DPO) may be mandatory for certain organisations and
processing activities, and the DPO must be involved in data protection impact assessments. A
capstone using institutional health data should know whether a DPO review is required and who
signs it.
Under HIPAA, the covered entity's privacy officer owns the determination of minimum necessary
access and the response to incidents. A student team is usually a business associate or a
research collaborator, and the agreement defines the reporting route.

The practical consequence for roles is that the data engineer cannot be the only person who
knows the access terms. The evidence lead must be able to state the licence and the
de-identification method at the defense, because that is a governance question, not an
engineering one.

### Authorship and credit

The International Committee of Medical Journal Editors (ICMJE) criteria for authorship require
substantial contribution to conception or design, drafting or critical revision, final approval,
and accountability. Acquiring data or funding alone does not qualify.
A capstone that turns into a paper must apply the criteria honestly, and a mentor who gave
advice does not automatically become an author. The team should agree the authorship policy
before submission, not after.

### What the rules do not do

No regulation tells a four-person team how to divide the work. No standard defines a fair peer
adjustment. Those are design choices, and the instruments above only constrain the parts that
touch patients, data, or claims.

| Instrument | Role it names | Binding on a capstone | What it constrains |
|---|---|---|---|
| IRB approval | Principal investigator | Yes, if human subjects | Conduct, reporting, retention |
| HIPAA | Privacy officer, security officer | Yes, for covered entities | Access, minimum necessary, incidents |
| GDPR | Controller, processor, DPO | Yes, in the EU | Lawful basis, impact assessment |
| FDA CDS guidance | Intended user | Yes, for device functions | Claim and user specification |
| ISO 13485 | Top management, quality roles | No, unless submitting | Design control and surveillance |
| ICMJE | Author | No, unless publishing | Credit and accountability |
| PhysioNet DUA | Individual account holder | Yes, contractually | Access and redistribution |

### Roles in a multi-site project

A second site adds three roles that a single-site capstone does not need: a site principal
investigator, a local data custodian, and a site analyst. The site investigator is accountable to
their own institution and cannot be directed by the lead. The custodian controls what leaves the
site, which constrains the analysis to federated or aggregate results. The site analyst
understands the local schema, which is rarely identical to the primary site's.

The common mistake is to treat the second site as a data source rather than as a partner with
its own accountability. That produces a data-transfer request that the site cannot approve, and
a validation plan that stalls. The role fix is to give the site investigator a decision right
over what is shared and a place in the authorship discussion, both agreed before any data move.

### Vendor and contractor roles

Where a capstone uses a commercial platform, a pretrained model, or a cloud service, the vendor
holds a role even though it is not in the RACI. The vendor is responsible for the service's
behaviour and accountable to its own customers, not to the team. The team remains accountable
for the claim it makes using the vendor's output.

That distinction matters at the defense. "The vendor reports an AUC of 0.9" is not evidence the
team produced. "We evaluated the vendor's model on our cohort and obtained this interval" is.
The regulatory memo should name the vendor's role, the licence terms, and whether the licence
permits the intended clinical use.

## 9. Industry practice and case evidence

### What team composition predicts

Nolte, Chounta, and Herbsleb analysed hackathon projects at scale and reported that skill
diversity and stated intent to continue predicted longer-term continuation, while intense
short-term activity predicted lower continuation. The study is about hackathons and uses
observational data, so causal claims are not warranted. The practical reading is that a team of
identical specialists optimises for the weekend and not for the month after.

Olson and colleagues' survey of twelve healthcare hackathons found that 30.3 percent of projects
had made progress after a mean of 12.2 months. The study does not report team composition as a
predictor, and it should not be cited as if it did. What it does show is that continuation is the
exception, which makes the roles that sustain a project after the event worth designing.

### Ownership failures in deployed systems

The external validation of a proprietary sepsis model by Wong and colleagues in 2021 is usually
cited for its performance result: an area under the receiver operating characteristic curve of
0.63 against vendor-reported values of 0.76 to 0.83. It is also a case about ownership. The
validation was possible because an academic group had access to the data and the clinical
context to define the outcome independently. A deployment without an independent owner of the
outcome definition cannot detect that kind of gap.

Obermeyer and colleagues' 2019 analysis of a commercial risk-stratification algorithm is a case
about label ownership. The algorithm predicted cost, and cost was treated as a proxy for need.
The group that owned the label definition determined the equity outcome: correcting the label
would have raised the proportion of Black patients identified for extra care from 17.7 to 46.5
percent. No amount of modelling skill compensates for a label that nobody clinically owns.

### Ownership in national evaluations

NHS England's guidance on planning and implementing real-world AI evaluations, drawing on the AI
in Health and Care Award portfolio, emphasises that evaluations need a named owner in the
deploying organisation and a plan for generating evidence during deployment. The guidance is
practitioner material, not a trial, but it reflects repeated experience: projects without an
operational owner stall at the pilot stage.

The Royal College of Radiologists' 2025 consultation document on AI deployment fundamentals makes
a parallel point about clinical ownership and the evidence gap between marking and measured
benefit. Again, this is a professional body's position, not a controlled finding.

### What this means for a capstone team

The transferable pattern is that accountability for the label, the outcome, and the claim must
sit with someone who has clinical or operational standing. In a four-person team, that person is
usually the clinical liaison, and the liaison's scarcest resource is time. Designing the project
around three hours per week of clinical attention is not a limitation; it is the realistic
constraint that most industry projects also face.

### Inference, labelled

It is an inference that teams which name a backup for every critical artefact survive a member
loss with less schedule damage. The mechanism is plausible and the arithmetic in Section 6 is
illustrative, not experimental. No controlled study of capstone teams is cited.

### Coordination cost grows faster than headcount

A five-person team has ten pairwise relationships; a six-person team has fifteen. The number of
communication channels grows quadratically while the work grows linearly, which is why adding a
sixth member to a capstone usually adds coordination rather than capacity.

The practical rule is to add a person only when the work splits into a stream with its own
artefacts and its own decision rights. A sixth member who shares the same artefacts increases
the number of people who must agree on each change.

This is not an argument against large teams in general. It is an argument against large teams
without structure. A six-person team with two streams, two accountable leads, and a shared
decision log can work; a six-person team with one stream and six opinions cannot.

### Open weights and ownership of the artefact

When a capstone builds on open-weight models, the team owns its evaluation and its pipeline, not
the model. The licence governs what can be done with the weights, whether derivatives can be
redistributed, and whether clinical use is permitted. A team that presents a fine-tuned model as
its own product without reading the licence has an ownership problem that surfaces at the
regulatory memo.

The role consequence is that the evidence lead must own the licence inventory, not the modeler.
The modeler is focused on performance and is inclined to treat the licence as a formality; the
evidence lead is the person who has to defend the claim.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| No clinical owner | No clinician has attended a session by week 3 | Clinical input treated as optional | Valid model for the wrong decision | Recruit a liaison with a specific first ask |
| Two accountable owners | "We both own the validation" | Diffused responsibility | Nobody signs; gaps at freeze | Split the artefact or name one A |
| Lead overload | Lead accountable for half the dossier | Competence attracts work | Single point of failure at defense | Transfer an artefact, not just tasks |
| Aggregate utilisation hides bottleneck | "We are only 68 percent loaded" | Role-level peaks averaged away | Late validation, rushed analysis | Compute demand per role per week |
| No backup | Only one person can run the pipeline | Knowledge not transferred | Defense collapses if that person is ill | Written handoff note per critical artefact |
| Silent free riding | Commit history shows two authors | Contribution not tracked | Peer adjustment surprises | Contribution log from week 1 |
| Mentor overreach | Mentor edits the dossier | Boundary not stated | Unaccountable influence on claims | Written mentor boundary at onboarding |
| Clinical liaison over-committed | Four hours demanded, three available | Defense week assumed free | Weak clinical review at the worst time | Agree the defense slot in week 2 |
| Role confusion at handoff | Rework after every transfer | No acceptance test | Two weeks lost to rework | Testable handoff criteria |
| Decision rights unclear | Scope argued in week 8 | No named decider | Plan changes without a record | Decision-rights table at G0 |
| Everyone informed, nobody consulted | Large email threads, no decisions | Consultation used as delay | Slow decisions, resentment | Name C and I explicitly |
| Individual defense failure | One member cannot explain the memo | Work siloed by person | Team score multiplied down | Rotate artefact walkthroughs weekly |
| Onboarding debt | New member reads code for two weeks | No brief or first task | Capacity arrives late | One-page brief and a 15-minute first task |
| Conflict personalised | Disagreement about "attitude" | No role vocabulary | Escalation to programme lead | Reframe as a decision-rights question |

The table is read in one direction: find the row whose early warning you can observe this week.
Most teams can observe three of them by week 3, which is early enough to act. A team that reads
the table in week 11 will recognise six and be able to fix none.

The two rows that matter most are "no clinical owner" and "lead overload", because both are
invisible in a standard progress report. A team can report 90 percent of tasks complete while
the clinical question is unanswered and one person holds the dossier. The RACI check and the
capacity matrix are the instruments that make both visible.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| Every artefact has one accountable role | RACI is meaningful only if singular | Shared ownership by design | Run the single-accountable check |
| Availability is known in week 1 | Capacity arithmetic depends on it | Rotas, exams, employment | Ask for weekly hours in writing |
| Roles can be held by the same person | Small teams need merging | Merged role exceeds capacity | Compute demand per role, not per person |
| The clinical liaison can give three hours per week | Clinical input is continuous | Consultant rota | Agree the slot and a fallback reviewer |
| Mentors advise without owning | Accountability stays internal | Mentor writes the dossier | State the boundary in the brief |
| Contribution is observable | Logs are fair | Invisible work such as coordination | Log coordination hours explicitly |
| The lead can reassign work | Overload is fixable | No authority or no context | Agree reassignment at G0 |
| Peer adjustment is bounded | Protects against popularity voting | Unbounded discretion | Written justification and review |
| Individual defense covers any topic | Prevents siloing | Panel asks only the owner | Publish the rule at the start |

The largest limit is that role structure cannot create expertise. A team without a statistician
can assign the analysis plan to someone, but the plan may still be weak. The structure makes the
gap visible and allocates the risk; it does not fill it.

A second limit is that the arithmetic assumes hours are interchangeable within a role. They are
not. Two hours of a senior clinician's time and two hours of a junior analyst's time produce
different artefacts, and the capacity table treats them as the same unit. The table is a
screening instrument for overload, not a costing model.

A third limit is cultural. Some teams experience explicit role assignment as distrust, and some
members with less visible work feel marginalised by a contribution log. The mitigation is to log
coordination, review, and documentation explicitly, and to say at the start that the log exists
to protect people from late surprises rather than to rank them.

## 12. Exercises

### Exercise 1. Build a RACI (application, 40 minutes)

For a capstone of your choice, list seven artefacts and assign R, A, C, and I for five roles.
Run the single-accountable check.
**Deliverable:** the matrix and the check output.
**Criteria:** exactly one A per row; at least one C per row beyond the A; no role with more than
three A entries without a written justification.
**Time:** 40 minutes.

### Exercise 2. Compute role capacity (calculation, 45 minutes)

Build a weekly demand matrix for your team and compare it with real availability. Identify every
over-capacity week.
**Deliverable:** the matrix, the utilisation table, and a levelling proposal.
**Criteria:** availability reflects actual hours; peaks are named by role and week; the proposal
does not assume extra hours.
**Time:** 45 minutes.

### Exercise 3. Write a handoff test (application, 30 minutes)

Choose the handoff from data engineer to modeler and write an acceptance test that can be run in
under five minutes.
**Deliverable:** the test as a checkable list or a short script.
**Criteria:** the test names row grain, time zero, missingness, and one leakage check.
**Time:** 30 minutes.

### Exercise 4. Design the decision-rights table (design, 35 minutes)

Write the table for scope, metric, data access, and claim language, with a decider, a consulted
role, and an informed role for each.
**Deliverable:** a four-row table.
**Criteria:** every decision has one decider; the clinical liaison has a veto on claim language;
escalation is named.
**Time:** 35 minutes.

### Exercise 5. Plan for a member loss (design, 40 minutes)

Assume your data engineer leaves at the end of week 4. Produce a revised RACI and a list of
artefacts to drop.
**Deliverable:** revised matrix plus a one-paragraph rationale.
**Criteria:** no role exceeds capacity after the change; at least one artefact is dropped rather
than reassigned; the claim is adjusted if needed.
**Time:** 40 minutes.

### Exercise 6. Critique a team structure (critique, 35 minutes)

A five-person team has one member accountable for the charter, analysis plan, validation,
regulatory memo, and defense. Diagnose the structure and propose a redistribution.
**Deliverable:** a 300-word critique with a revised matrix.
**Criteria:** identifies the bus factor; proposes transfers that respect context; names what the
team loses.
**Time:** 35 minutes.

### Exercise 7. Run a midpoint role review (application, 30 minutes)

Using the contribution log template, simulate a week-6 review for a team where one member has
logged 40 percent of the median hours for three weeks.
**Deliverable:** a decision record with three options and a chosen action.
**Criteria:** the action is specific and dated; it distinguishes capability from availability;
it does not publicly shame the member.
**Time:** 30 minutes.

### Exercise 8. Prepare for the individual defense (recall, 25 minutes)

Write five questions you could be asked about an artefact you do not own, and draft answers from
the repository.
**Deliverable:** five question-answer pairs.
**Criteria:** answers cite an artefact or a number; at least two answers acknowledge a limitation.
**Time:** 25 minutes.

## 13. Self-check answers

**1. Why must exactly one role be accountable for an artefact?**
Because accountability is the act of signing and answering for the artefact. Two signatories
diffuse the obligation, and in practice neither prepares. The RACI check makes the failure
visible in week 1 rather than at freeze.

**2. Why can a clinical liaison be accountable for no artefact?**
Because the liaison may be a mentor rather than a team member. The liaison contributes clinical
judgement, reviews errors, and holds a veto on the claim, but does not own a dossier artefact.
If the programme requires a clinical signature on the regulatory memo, the matrix changes.

**3. What is the difference between aggregate utilisation and role-level capacity?**
Aggregate utilisation divides total demand by total capacity and can look comfortable while one
role is heavily overloaded. Role-level capacity compares each role's weekly demand with its own
availability, which is what determines whether a gate is met.

**4. Compute the modeler's overload in week 6.**
Demand is 20 hours against 14 available, so the overload is 6 hours, or 43 percent of capacity.
The modeler can absorb it for one week, but the plan has four consecutive weeks above capacity,
which is not sustainable.

**5. Why is the bus factor an artefact property rather than a team property?**
Because the consequence depends on which artefact loses its owner. Losing a person who owns one
non-critical artefact is survivable; losing a person who owns the charter, the value case, and
the defense is not. Counting artefacts makes the exposure specific.

**6. What should a team do when a member drops out?**
Reassign at most one of their artefacts, drop the rest, and adjust the claim. Reassigning
everything moves the problem to the validation week, when there is no capacity to absorb it.

**7. Why does the contribution log exist if the team grade is shared?**
Because shared grades hide unequal contribution and individual defenses expose it. The log makes
the distribution visible early, when a conversation can still change it, and provides evidence
for a bounded peer adjustment.

**8. What is a testable handoff?**
A transfer with an acceptance test that the receiver can run: row grain, time window, missingness
threshold, and a leakage check. Without a test, the handoff is a conversation and the errors
surface later.

**9. Why should the clinical liaison have a veto on claim language?**
Because a claim a clinician will not sign is not a claim the team can defend. The veto prevents
the team from writing "detects" when the evidence supports "prioritises for review".

**10. Rubric for Exercise 5.** A strong answer drops an artefact, adjusts the claim, and shows
the revised capacity arithmetic. A weak answer reassigns all four artefacts to three people and
asserts that the team will cope.

## 14. Interview questions

**Q1. "What was your role on the team?"**
Strong answer names the artefacts owned, the decisions made, and one thing that went wrong.
Weak answer names a task list.

**Q2. "How did you divide the work?"**
Strong answer describes artefact ownership, capacity arithmetic, and a reallocation at the
midpoint. Weak answer says everyone did a bit of everything.

**Q3. "Who owned the clinical question?"**
Strong answer names the liaison, the hours committed, and how disagreement was resolved. Weak
answer says the team discussed it with a clinician once.

**Q4. "What happened when someone disagreed about the metric?"**
Strong answer describes the decision-rights table and the recorded decision. Weak answer
describes a long argument or says there was no disagreement.

**Q5. "How did you prevent one person from doing everything?"**
Strong answer describes the contribution log, the weekly demo, and the rotation of walkthroughs.
Weak answer says the team worked well together.

**Q6. "What would you have done if a member left in week 8?"**
Strong answer names the artefacts that would be dropped and the claim that would narrow. Weak
answer says the team would have coped.

**Q7. "Can you explain the regulatory memo?"**
Strong answer explains it, even if someone else wrote it, and names the open question. Weak
answer says "that was someone else's part".

**Q8. "How did you use your clinical mentor's time?"**
Strong answer describes specific asks, the 30-minute cadence, and what changed as a result. Weak
answer says the mentor reviewed the work.

**Q9. "What was your biggest coordination failure?"**
Strong answer names a handoff that failed, the rework it caused, and the test added afterwards.
Weak answer says coordination was fine.

**Q10. "How would you structure a team of six?"**
Strong answer splits into two streams with separate artefacts and a shared decision log, and
notes the coordination cost. Weak answer adds roles without changing the structure.

**Q11. "Who would you hire first if you had budget for one role?"**
Strong answer names the bottleneck from the capacity table, usually evidence or clinical
liaison, and what artefact it unblocks. Weak answer says a machine-learning engineer, because
that is the visible work.

A strong interview answer about roles almost always contains a specific artefact, a specific
hour count, and a specific thing that went wrong. An answer without numbers is a description of
intentions, and interviewers hear it as inexperience.

## 15. Key terms

- **Accountable** — the single role that signs an artefact.
- **Authorship** — credit under ICMJE criteria.
- **Backup** — the person who can rebuild or explain an artefact.
- **Bottleneck** — the role whose demand first exceeds capacity.
- **Bus factor** — artefacts unowned when a person is removed.
- **Capacity** — available hours per week for a role.
- **Clinical liaison** — the role owning workflow, labels, and error review.
- **Consulted** — roles whose input is sought before finalising.
- **Contribution log** — weekly record of hours and artefacts.
- **Data steward** — the role owning access, lineage, and quality.
- **Decision right** — authority to make a named choice.
- **Demand** — planned hours per week for a role.
- **DPO** — data protection officer under the GDPR.
- **Escalation** — the route to a decision when a threshold is crossed.
- **Evidence lead** — the role owning design, regulatory triage, and equity.
- **Free riding** — contributing less than agreed without disclosure.
- **Handoff** — transfer with an acceptance test.
- **ICMJE** — International Committee of Medical Journal Editors.
- **Individual defense** — oral examination on any part of the work.
- **Informed** — roles told of the outcome.
- **IRB** — institutional review board.
- **Midpoint review** — structured reassessment of roles and scope.
- **Onboarding** — bringing a mentor or member to useful contribution.
- **Peer adjustment** — bounded multiplier on a team grade.
- **Principal investigator** — the person accountable to the institution.
- **Product owner** — the role owning scope, schedule, and decision log.
- **RACI** — responsible, accountable, consulted, informed.
- **Responsible** — the role that produces the work.
- **Single point of failure** — a role or artefact with no backup.
- **T-shaped** — deep in one discipline, conversant across several.
- **Utilisation** — demand divided by capacity.
- **Weekly rhythm** — fixed cadence of stand-up, studio, and demo.

## 16. Sources

All URLs accessed 26 October 2026.

- Nolte A, Chounta IA, Herbsleb JD. What happens to all these hackathon projects? *Proceedings
  of the ACM on Human-Computer Interaction* 2020;4(CSCW2):1-25.
  https://doi.org/10.1145/3415216
- Olson KR, Walsh M, Garg P, et al. Health hackathons: theatre or substance? *BMJ Innovations*
  2017;3:37-44. https://doi.org/10.1136/bmjinnov-2016-000147
- Wong A, Otles E, Donnelly JP, et al. External validation of a widely implemented proprietary
  sepsis prediction model. *JAMA Internal Medicine* 2021;181(8):1065-1070.
  https://doi.org/10.1001/jamainternmed.2021.2626
- Obermeyer Z, Powers B, Vogeli C, Mullainathan S. Dissecting racial bias in an algorithm used
  to manage the health of populations. *Science* 2019;366(6464):447-453.
  https://doi.org/10.1126/science.aax2342
- NHS England, Planning and implementing real-world AI evaluations.
  https://www.england.nhs.uk/long-read/planning-and-implementing-real-world-ai-evaluations-lessons-from-the-ai-in-health-and-care-award/
- Royal College of Radiologists, *AI deployment fundamentals*, 2025.
  https://www.rcr.ac.uk/media/r0th2x0k/ai-deployment-fundamentals_public-consultation.pdf
- FDA, *Clinical Decision Support Software*, guidance, September 2022.
  https://www.fda.gov/regulatory-information/search-fda-guidance-documents/clinical-decision-support-software
- FDA, *Predetermined Change Control Plan* guidance, December 2024.
  https://www.fda.gov/media/185050/download
- ISO 13485, *Medical devices — Quality management systems*.
  https://www.iso.org/standard/59752.html
- ICMJE, *Defining the role of authors and contributors*.
  https://www.icmje.org/recommendations/browse/roles-and-responsibilities/defining-the-role-of-authors-and-contributors.html
- Regulation (EU) 2016/679, General Data Protection Regulation.
  https://eur-lex.europa.eu/eli/reg/2016/679/oj
- HIPAA Privacy Rule, 45 CFR Parts 160 and 164. https://www.hhs.gov/hipaa/
- PhysioNet, CITI course instructions. https://www.physionet.org/about/citi-course/
- Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement. *BMJ* 2024;385:e078378.
  https://doi.org/10.1136/bmj-2023-078378

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic: [`02_milestone-plan.md`](02_milestone-plan.md)
- Next topic: [`04_capstone-archetypes.md`](04_capstone-archetypes.md)
- Related topic: [`10_judging.md`](10_judging.md)
- Survey file: [`../../course/19_capstone_hackathons.md`](../../course/19_capstone_hackathons.md)
- Labs: [`../../labs/`](../../labs/)
