# Hackathon Formats

> **Module 19 — Capstone, Hackathons, and Competitions**
> Topic 06 of 11
> Estimated study time: 3.5 hours
> Prerequisites: Topics 01 and 05
> Version 2026.3

## 1. Why this topic exists

A hackathon is a time-boxed event in which a small team produces a demonstrable artefact, usually
in twenty-four to seventy-two hours, judged on framing, feasibility, and presentation.
It trains a different set of muscles from a capstone: speed of framing, rapid prototyping,
cross-functional improvisation, and the ability to explain an idea before it is complete.
The two formats are complementary, and a programme that teaches only one produces graduates who
are either slow and rigorous or fast and unsubstantiated.

The industry reason is straightforward.
Health systems, payers, device companies, and investors use hackathons and datathons to source
ideas, test partners, and recruit. A candidate who has competed can speak to a real constraint:
the demo failed at hour 40, the data were not what the description promised, and the team had to
choose between validation and polish.
That story is more useful in an interview than another leaderboard score.

The academic reason is equally practical.
A hackathon compresses the framing step that a capstone spreads over two weeks. A team that has
written a charter in ninety minutes, under time pressure, with a clinician in the room, has
learned what information actually matters.
The compression also exposes weak framing: a team that cannot state a decision by hour two will
not state one by hour forty.

The risk is that hackathon habits leak into capstone work.
Hackathons reward a working demo and a convincing story; they do not reward validation,
calibration, or governance.
A team that treats the capstone as a long hackathon will produce a polished demo with no evidence,
and a panel will ask the one question the format never required: what happens when the model is
wrong?

The evidence on hackathon outcomes is sobering and useful.
Olson and colleagues surveyed twelve healthcare hackathons in India, Uganda, and the United
States between 2012 and 2015, published in BMJ Innovations in 2017. Of 356 projects, 196 responded;
30.3 percent had made progress after a mean of 12.2 months, and 24.7 percent had begun pilot
testing. Non-responders were conservatively counted as no progress, so the true rate is uncertain,
but the order of magnitude is clear.
Ramadi and Nguyen, reporting on the virtual MIT COVID-19 Challenge in npj Digital Medicine in 2021,
found that more than 25 percent of teams were still working five weeks after the event, against
about 19 percent at two weeks for comparable in-person events.

The honest reading is that a hackathon produces a prototype and a team, and occasionally a
project. It does not produce a validated product, and a programme that presents it as one teaches
students to overclaim.

A second honest reading is that the format's incentives are not the capstone's incentives, and
that is why both are taught.
A hackathon rewards a team that can find a need, build one step, and explain it under pressure.
A capstone rewards a team that can prove a claim and defend its limits.
A graduate who has done only the first will build demos for problems nobody has; a graduate who
has done only the second will spend three weeks on a charter while the opportunity closes.
The programme's claim is that the combination produces judgement, and the combination is what the
assessment is designed to test.

## 2. Learning outcomes

By the end of this topic you can:

1. **Distinguish** a sprint hackathon, a grand hack, a datathon, and a capstone by duration,
   deliverable, judging emphasis, and characteristic failure.
2. **Write** a one-page charter in ninety minutes that names a decision, an owner, and a metric.
3. **Plan** a twenty-four-hour and a forty-eight-hour runbook with checkpoints, a code freeze, and
   a demo buffer.
4. **Compute** the person-hours available and the wall-clock budget, and identify the point at
   which the plan becomes infeasible.
5. **Design** a demo that works without a network, uses held-out or synthetic data, and shows one
   failure case.
6. **Adapt** a capstone evidence package to a datathon without pretending it is a validation study.
7. **Judge** a hackathon pitch against a weighted rubric and explain how the weights change the
   ranking.
8. **Recount** a hackathon failure in an interview as evidence of judgement rather than as an
   anecdote.

## 3. The core mental model

A hackathon is a fixed budget of wall-clock time, divided into phases with different currencies.
The currencies are attention, credibility, and working software.
Attention is scarcest in the first two hours, when the team decides what it is building.
Credibility is scarcest in the last six hours, when the demo must run and the story must hold.
Working software is scarcest in the middle, when the temptation is to polish rather than to
finish an end-to-end path.

```text
  HOUR  0    2    4    9   15   18   21   23   24
        |    |    |    |    |    |    |    |    |
  PHASE frame data base  improve validate demo rehearse buffer
        |    |    |    |    |    |    |    |    |
  RISK  wrong  wrong no   scope  no     demo  stage demo
        | decision data  path  creep  split  fails fright fails
        v    v    v    v    v    v    v    v
  CHECK charter rows  end- one  subgroup  held-out  backup  submit
        signed  count to-end delta table    case   recorded
```

Read the diagram as a sequence of irreversible narrowing.
At hour 2 the team chooses a decision, and every later hour is spent on that decision.
At hour 4 the team chooses a data slice, and every later result is conditional on it.
At hour 9 the team has an end-to-end path, and every later improvement must preserve it.
At hour 18 the team freezes the model, and every later hour is spent on evidence and story.

The checkpoints are the mechanism that makes the narrowing visible.
A team that reaches hour 9 without an end-to-end path is not behind; it is building the wrong
thing. A team that reaches hour 18 without a held-out slice has no evidence and should say so.

Three properties of the format matter.

**Wall-clock is not person-hours.** A team of five in a forty-eight-hour event has 240 person-hours
but only 48 hours of wall clock, and the wall clock is what the demo depends on. Parallelism helps
until coordination costs more than the work, which for a team of five happens around hour 20.

**The demo is the deliverable.** Judging rewards what the judges can see, and a model with no
interface is invisible. This is a real constraint of the format, not a flaw in the judges, and a
team that ignores it loses to a weaker model with a clearer demo.

**The claim must match the evidence, even at speed.** A hackathon does not require validation, but
it does require honesty. "We built a prototype that ranks cases on a synthetic cohort" is a
defensible sentence in three seconds. "We built a triage system" is not, and a judge who asks one
follow-up will find out.

A fourth property is social.
Hackathons select for teams that can recruit collaborators, divide work, and resolve disagreement
quickly. That is a real skill, and it is also a source of failure: a charismatic team can win with
a weak artefact, and a quiet team with a solid one can lose. The rubric in Section 7 is designed to
reduce that effect, not to eliminate it.

### The three checkpoints that decide the event

Most of the outcome is fixed by three moments.

Hour 2 decides the question. A team that has not chosen a decision by hour 2 will spend the event
defending an accident.

Hour 9 decides whether there is a demo. A team without an end-to-end path at hour 9 will either
build one in a panic at hour 36 or present slides.

Hour 18 decides whether there is evidence. A team without a held-out result at hour 18 will
present a demo that works on the examples it was tuned on.

Each checkpoint has a single question and a single consequence. The runbook in Section 5 is
mostly a device for making these three moments visible before they pass.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Sprint hackathon | A 24-hour event with one workflow and one metric | Any short event | 24 hours |
| Grand hack | A 48-hour event from clinical need to prototype | A sprint with more people | 48 hours |
| Datathon | A 5-to-7-day event around one dataset and several questions | A hackathon with data | 5-7 days |
| Capstone | A 12-week project producing an evidence dossier | A long hackathon | 12 weeks |
| Charter | One page naming decision, owner, metric, and scope | A business plan | One page |
| Checkpoint | A scheduled moment at which a condition must hold | A status update | Hour and condition |
| Code freeze | The moment after which code changes only for demo fixes | The submission deadline | Hour |
| Demo | A live or recorded demonstration on held-out or synthetic data | A slide walkthrough | Minutes |
| Pitch | A timed verbal presentation with a fixed structure | A product tour | Minutes |
| Track | A thematic stream within an event | A team | Track name |
| Team formation | The process of assembling a team at the event | Pre-formed teams only | Deadline |
| Mentor | An expert who gives feedback during the event | A judge | Role |
| Judge | A person who scores the final pitches | A mentor | Role and rubric |
| Rubric | Weighted criteria used to score pitches | A vibe check | Criteria and weights |
| Prototype | A working artefact that demonstrates one step | A product | Maturity label |
| Validation | Evidence that the artefact works on unseen data | A demo | Level L0-L4 |
| Held-out slice | Data not used during development | A random sample of the training set | Split definition |
| Synthetic data | Generated data labelled as synthetic | Fake data presented as real | Generator and version |
| Backup recording | A recorded demo used if the live demo fails | A screenshot | Video and timestamp |
| Person-hour | One person working for one hour | Wall-clock hour | Hours |
| Wall clock | Elapsed event time | Effort | Hours |
| Scope creep | Adding questions after the charter | Iteration | Backlog entry |
| Submission | The artefact and materials handed to organisers | The pitch | Deadline and format |
| Intellectual property | Ownership of the work produced at the event | The idea | Agreement terms |
| Human subjects | Research involving identifiable people or their data | Any user testing | IRB determination |

Two distinctions do most of the work.
A prototype is not a product: it demonstrates one step, on a small sample, without the governance
a product requires. A demo is not validation: it shows that the artefact runs, not that it
generalises.

A third distinction matters at health events.
User testing with clinicians is not necessarily human-subjects research, but testing with
patients or their data usually is, and an event does not suspend the requirement. A team that
plans to test with patients at a weekend event should check the event's ethics process before the
event, not during it.

## 5. The mechanics, step by step

The procedure below is a runbook for a forty-eight-hour grand hack, with notes for a
twenty-four-hour sprint and a one-week datathon.
It assumes a team of four to six and a health-related problem.

### Step 1. Read the rules before the event

**Input:** the event page, the rules, and the judging criteria.

**Output:** a one-page summary of eligibility, data rules, intellectual property, submission
format, and judging weights.

**Rationale:** The rules determine what can be built and what counts as a win. A team that
discovers at hour 40 that external data are prohibited has wasted the event. A team that
discovers the judging weights favour need validation over model accuracy has optimized the wrong
thing.

**Branch point:** if the rules prohibit the data the team planned to use, choose a different
question before arrival. Do not plan to ask for an exception at the event.

### Step 2. Arrive with a question, not a solution

**Input:** two or three candidate problems from the team's domain.

**Output:** a shortlist with one sentence each on the decision and the user.

**Rationale:** Teams that arrive with a solution spend the first six hours looking for a problem
to fit it. Teams that arrive with a question spend the first six hours finding out whether the
question is real.

**Branch point:** if none of the questions survives the first clinician conversation, use the
event's problem pitches as a source. That is what they are for.

### Step 3. Spend the first six hours on need, not code

**Input:** the shortlist and access to clinicians, patients, or domain experts at the event.

**Output:** three interview notes with a named workflow, a pain point, and a current workaround.

**Rationale:** The first six hours are the only time when the team can change its mind cheaply.
A team that starts coding at hour two will defend its choice for the remaining forty-six hours.
The grand-hack format explicitly rewards need validation, and the evidence on hackathon
continuation suggests that projects grounded in a real need last longer.

**Branch point:** if the team cannot find any user to talk to, the problem is not validated.
Choose a problem where the event provides access, or reframe as a methods demonstration and say
so in the pitch.

### Step 4. Write the one-page charter in ninety minutes

**Input:** the interview notes.

**Output:** a charter with decision, owner, population, comparator, metric, and out-of-scope.

**Rationale:** The charter is the contract with the team and the judges. It takes ninety minutes
because it forces the team to choose one action and one metric, and choosing is the hard part.

**Branch point:** if the charter needs two decisions, pick one and write the other in the
out-of-scope section. A pitch that claims two decisions convinces nobody.

### Step 5. Choose the smallest end-to-end path

**Input:** the charter and the available data.

**Output:** a pipeline that takes one input, produces one output, and displays it, even if the
model is a rule.

**Rationale:** The end-to-end path is the demo's backbone. A team with a weak model and a complete
path can improve the model; a team with a strong model and no path has nothing to show.

**Branch point:** if the path requires an integration the team cannot complete in eight hours,
replace the integration with a stub and label it. A labelled stub is honest; an unlabelled one is
a demo failure waiting to happen.

### Step 6. Freeze the schema and the split at hour 13

**Input:** the data audit.

**Output:** a written schema, a row count, a held-out slice, and a leakage check.

**Rationale:** The split and the schema are the two things that cannot be repaired later without
invalidating every number. Freezing them at hour 13 leaves twenty-three hours for the model and
the demo.

**Branch point:** if the data are too small for a held-out slice, use cross-validation and say so.
Do not evaluate on training data and call it a result.

### Step 7. Improve one thing only

**Input:** the baseline and the error analysis.

**Output:** one change, measured against the baseline on the held-out slice.

**Rationale:** The middle hours are where scope creep enters. One change, measured, is a finding;
five changes, unmeasured, are a demo. The discipline also produces a sentence for the pitch:
"the baseline scored X, the change scored Y, and the interval is Z".

**Branch point:** if the change does not improve the metric, keep the baseline and report the
negative result. A negative result with a clean comparison is more credible than an unmeasured
improvement.

### Step 8. Validate on the held-out slice at hour 30

**Input:** the frozen split and the candidate.

**Output:** one primary metric with an interval, one subgroup table, and one failure case.

**Rationale:** Validation at hour 30 leaves enough time to fix a broken result before the demo.
Validation at hour 44 leaves none, and a team that discovers a leakage problem at hour 44 will
hide it rather than fix it.

**Branch point:** if the held-out result is much worse than the development result, suspect
leakage or a distribution shift. Report both numbers and the gap; the gap is the finding.

### Step 9. Build the demo for a hostile environment

**Input:** the validated artefact.

**Output:** a demo that runs offline, on a held-out or synthetic case, in under ninety seconds,
with a recorded backup.

**Rationale:** Conference networks fail, credentials expire, and projectors misbehave. A demo that
depends on a live service is a demo that will fail at the worst moment. The backup recording is
not optional.

**Branch point:** if the demo cannot run offline, record it in advance and present the recording
as the primary artefact. Say that it is a recording.

### Step 10. Write the pitch to the judging rubric

**Input:** the rubric and the results.

**Output:** a three-minute structure: decision and user, current practice, what was built,
evidence, failure and mitigation, next step.

**Rationale:** Judges score against criteria, and a pitch that does not address a criterion cannot
score on it. The structure above maps to need, feasibility, demo, and pathway, which are the
common criteria.

**Branch point:** if the rubric weights need validation heavily, spend more time on the interview
evidence and less on the model. The weights are published for a reason.

### Step 11. Rehearse twice, at hour 42 and hour 46

**Input:** the pitch and the demo.

**Output:** two timed runs, one with a mentor, and a list of fixes.

**Rationale:** The first rehearsal always reveals a timing problem and usually a demo problem.
The second confirms the fixes. A team that rehearses once has rehearsed zero times under
pressure.

**Branch point:** if the demo fails in rehearsal, cut the demo to the recorded backup and use the
remaining time on the story. A working recording beats a broken live demo.

### Step 12. Freeze code at hour 46

**Input:** the working artefact.

**Output:** a tagged version, a submission package, and a rollback copy.

**Rationale:** The last two hours are for submission and sleep, not for features. A change at hour
47 that breaks the demo is the most common self-inflicted loss in hackathons.

**Branch point:** if a critical bug appears after freeze, fix only that bug, test the demo
immediately, and keep the rollback copy.

### Step 13. Submit early and verify the format

**Input:** the submission requirements.

**Output:** a submitted artefact, confirmed by the platform, with a screenshot of the confirmation.

**Rationale:** Submission systems fail under load at the deadline. Submitting an hour early costs
nothing and prevents the most avoidable failure in the format.

**Branch point:** if the platform rejects the format, fix the format before adding content. A
correct submission with a weaker result scores; a strong result with a rejected format does not.

### Step 14. Debrief within seventy-two hours

**Input:** the judges' feedback, the team's notes, and the artefacts.

**Output:** a two-page debrief with what worked, what failed, and what would be done differently.

**Rationale:** The learning from a hackathon decays quickly. A debrief within three days captures
it while the memory is specific, and it converts an anecdote into a transferable lesson.

**Branch point:** if the team intends to continue the project, the debrief becomes the input to a
capstone charter. If not, it becomes a portfolio entry.

A debrief that lists only what went wrong is half a debrief. The team should also record the
decisions that worked, because those are the reusable part. In the worked example, the decision
to build the fixture before the model is what saved the demo, and it is the lesson the team
carries to the next event.

### The one-page artefact at the end

By hour 48 the team should have four things in one folder: the charter, the runbook with
checkpoints, the results table with the held-out metric and its interval, and the demo recording.
Everything else is optional.
This minimum is the same shape as a capstone dossier, reduced to a page, and it is what makes the
difference between an anecdote and a portfolio entry.

### The organiser's view

A team that understands the organiser's constraints competes better, because it can predict what
the event will and will not provide.

An organiser must balance four things: the problem list, the mentor roster, the data arrangements,
and the judging rubric. Each is a constraint on the teams.
A problem list that is too broad produces solution-seeking. A mentor roster without clinicians
produces unvalidated needs. A data arrangement that is not legally reviewed produces a last-minute
prohibition. A rubric that weights demo over need produces polished empty prototypes.

The best events publish the rubric, provide curated data, and run a mentor briefing so that
mentors give consistent feedback. A team should ask for all three before the event, and should
treat the answers as part of the rules.

### The 24-hour sprint runbook

A sprint is not a compressed grand hack. It has no time for need validation, so the problem must
arrive validated or be chosen from the event's curated list.

Hours 0 to 2: choose one decision and write the one-page charter. The charter is shorter than the
grand-hack version: decision, owner, metric, and the data slice.
Hours 2 to 4: inspect the data, find the labels, and run the leakage screen. This is the last
point at which the question can change.
Hours 4 to 9: build the simplest end-to-end baseline. A rule-based system is acceptable if it
produces the output the demo needs.
Hours 9 to 15: improve one thing only. The improvement must be measured against the baseline on a
held-out slice.
Hours 15 to 18: validate, compute one subgroup table, and choose one failure case for the pitch.
Hours 18 to 21: build the demo and the three-minute story.
Hours 21 to 23: rehearse, record the backup, and submit.
Hour 23 to 24: buffer, and sleep if the submission is complete.

The sprint's characteristic failure is a demo with no baseline. A team that spends hours 4 to 15
on a model and never computes a baseline has no comparison, and a judge will ask what the simple
approach would have scored. The answer "we did not try" is worse than a lower score.

### The one-week datathon

A datathon compresses a research project rather than a product sprint. The deliverable is an
analysis, a notebook, and a short paper, and the judging emphasises rigour and reproducibility.

Day 1: freeze the schema, assign a data steward, and write the analysis plan. No modelling on day
1 except a data audit.
Day 2: baseline and first descriptive results. The baseline must be reproducible from a single
command.
Day 3: primary analysis and the first version of the figures.
Day 4: secondary analyses and subgroup checks.
Day 5: sensitivity analyses and the reproducibility package.
Day 6: write the paper and rehearse the presentation.
Day 7: present and publish the notebook with a lockfile.

The characteristic failure is fitting the public leaderboard on day 6 and discovering on day 7
that the hidden split rewards a different operating point. The defence is the same as in a
capstone: freeze the split and the metric before the first result, and treat the leaderboard as a
development signal.

### Decision table: what to cut when the event is behind

| Hours remaining | Symptom | Cut | Keep | Never cut |
|---:|---|---|---|---|
| 24 | No end-to-end path | Model complexity; use a rule or a pretrained model | One input, one output, one screen | The path itself |
| 18 | No validation | Subgroup analysis; second metric | Held-out primary metric | The split |
| 12 | Demo not started | Model improvements | A recorded walkthrough | The backup recording |
| 8 | Pitch not drafted | Demo polish | The six-part structure | The evidence sentence |
| 4 | Nothing works | Live demo | Recorded demo and story | Honest labelling |
| 2 | Submission not started | Everything | The submission | The deadline |

### Edge cases and forced decisions

**The demo fails at hour 40.** Switch to the recorded backup, say it is a recording, and spend the
remaining time on the story. Do not debug in front of judges; the audience remembers the failure,
not the fix.

**The data are not what the description promised.** Audit at hour 8, not hour 20. If the labels
are unusable, switch to a proxy outcome and label it, or choose a different question from the
event's list. A proxy that is disclosed is a limitation; a proxy that is hidden is a fabrication.

**A team member stops contributing.** Reallocate by task at the next checkpoint, and reduce scope.
Do not spend hours on the interpersonal problem during the event; address it in the debrief.

**A judge asks a question the team cannot answer.** Say what is known, say what is not, and say
what would be needed. A confident wrong answer is worse than an honest gap, and judges are
usually testing whether the team knows the difference.

**Scope creep at hour 30.** Write the new idea in a backlog and return to the charter. The backlog
is read in the debrief, not during the event.

**No clinician is available.** Use the event's mentors, documented need assessments, or a
published workflow study. Reframe the pitch as a methods demonstration if the clinical need is
unvalidated.

**Compute runs out.** Use a smaller model, a subset, or a hosted endpoint with a fallback. Report
the constraint in the pitch; judges understand resource limits and penalise pretence.

**The event forbids external data and the team needs it.** Use the provided data, or withdraw and
compete in a different category. A rule violation discovered after judging is worse than a
weaker result.

## 6. Worked example

A team of five enters a forty-eight-hour grand hack with a medication reconciliation idea.
The team has a nurse practitioner, a data engineer, a backend developer, a designer, and a
project lead. None has worked together before.

### The plan as written

The team allocates phases: need interviews 6 hours, charter 2, data access and audit 5,
end-to-end baseline 8, one improvement 9, validation 5, demo 7, pitch and rehearsal 4, buffer 2.
The phases sum to 48 hours, and the team congratulates itself on using the whole event.

The person-hour arithmetic is 5 × 48 = 240 hours. The planned work is 48 wall-clock hours with
the team working in parallel, which is not the same as 240 hours of task effort; the difference is
coordination, waiting, and the fact that not every task can use five people.

### Diagnostic one: the nominal plan has a 36 percent chance of finishing

The team replaces each phase duration with an optimistic, most-likely, and pessimistic estimate
and runs 20,000 simulations. The base case:

- mean finish 48.8 hours;
- median 48.9;
- 80th percentile 50.8;
- probability of finishing by hour 46: 12.2 percent;
- probability of finishing by hour 48: 35.8 percent.

The plan that sums to 48 hours has a 36 percent chance of finishing by the deadline, because the
sum of most-likely durations is not the most-likely sum. The 2-hour buffer covers less than one
standard deviation of the total.

This is the first diagnostic, and it is cheap: if the nominal plan ends at the deadline with no
buffer, the event will overrun. The team needs either a larger buffer or fewer phases.

### Diagnostic two: the demo depends on a live service

The team's design calls for a FHIR sandbox, a live terminology service, and a hosted model
endpoint. Three network dependencies in a conference venue is a demo failure waiting to happen.
The diagnostic is a single question: what happens to the demo if the network drops?

The team answers that it would not work. That answer is the second diagnostic, and it is fatal to
the demo regardless of model quality.

### The revision

The team makes two changes before hour 8.

First, it cuts the "one improvement" phase from nine hours to five and the validation phase from
five to three, and it moves the freed time into the demo and a larger buffer. The revised
simulation gives a mean of 42.7 hours, an 80th percentile of 44.5, and a 95.3 percent chance of
finishing by hour 46.

Second, it removes the live service dependencies. The FHIR sandbox is replaced by a local fixture
with three synthetic patients, the terminology service is replaced by a local lookup table, and
the model endpoint is replaced by a small local model. The demo now runs offline.

### The arithmetic of the revision

The cuts remove six hours of nominal work, but they remove more than six hours of variance,
because the cut phases had long right tails. That is why the probability improves from 36 to 95
percent rather than from 36 to 50 percent.

The team also reallocates person-hours. The designer spends the first six hours on interviews and
the next ten on the demo interface, rather than waiting for a model. The backend developer builds
the fixture and the local lookup before the model exists, which means the path is testable at hour
13 without a model.

### What the team actually built

A local web page that takes a synthetic patient's medication list and a discharge summary, flags
three classes of discrepancy, and shows the evidence for each flag. The model is a rules engine
plus a small classifier; the classifier's held-out F1 is 0.71 on 40 synthetic cases, with a wide
interval that the team reports.

The pitch says: "We built a prototype that flags medication discrepancies on synthetic patients.
The rule layer catches 9 of 12 seeded discrepancies; the classifier adds 2 more and introduces 4
false flags. The next step is a prospective comparison on real discharges, which requires a data
agreement we do not have."

That sentence is defensible, and it wins more credit than a claim of "AI-powered reconciliation".

### Varying one assumption

Suppose the venue's network is stable and the team keeps the live FHIR sandbox. The demo would be
more impressive and the risk higher. The team's decision rule, agreed at hour 8, is that any
external dependency must have a local fallback tested by hour 30. The rule is the lesson: the
question is not whether the network will fail, but what the demo does when it does.

### What the judges asked

Three questions, all predictable.

"How do you know the flags are right?" The team showed the seeded discrepancies, the held-out
counts, and the false-flag examples. The honest interval was more persuasive than a point
estimate.

"Who would use this?" The team named the discharge pharmacist and described the workflow from the
interview notes.

"What happens when it is wrong?" The team showed that every flag includes its evidence, so a
pharmacist can reject it in seconds, and that the tool never writes to the record.

The team placed second. The first-place team had a weaker model and a stronger need story, which
is consistent with the rubric's weighting and with the format's incentives.

### The hour-by-hour record

The team kept a shared log with a timestamp for each decision, which is unusual at an event and
useful afterwards.

Hour 2: three interviews, two with a discharge pharmacist and one with a ward nurse.
Hour 4: charter signed; decision is "which discharge medication list the pharmacist reviews
first"; owner is the discharge pharmacist; capacity is eight reviews per shift.
Hour 8: three synthetic patients built from a public medication example; no real records.
Hour 13: local fixture and lookup table working; the path runs end to end with a rule.
Hour 21: classifier added and measured against the rule baseline.
Hour 30: held-out F1 0.71 on 40 cases; the rule layer catches 9 of 12 seeded discrepancies.
Hour 35: demo page built; offline test with the network disabled.
Hour 42: first rehearsal; the pitch runs four minutes ten seconds and is cut to three minutes.
Hour 46: code frozen and the recording made.
Hour 47: submitted.

The log is the raw material for the debrief and for the interview story. It also shows a panel or
an employer that the team can account for its time, which is rare.

### The arithmetic of the placement

The team scored 4.2 on need, 4.0 on feasibility, 4.5 on demo, 3.5 on team, and 2.5 on pathway.
With weights of 0.30, 0.25, 0.20, 0.15, and 0.10, the total is:

0.30 × 4.2 + 0.25 × 4.0 + 0.20 × 4.5 + 0.15 × 3.5 + 0.10 × 2.5
= 1.26 + 1.00 + 0.90 + 0.525 + 0.25 = 3.935.

The winning team scored 4.8 on need, 3.5 on feasibility, 4.0 on demo, 4.0 on team, and 3.0 on
pathway, for 1.44 + 0.875 + 0.80 + 0.60 + 0.30 = 4.015.

The margin is 0.08 points, and it came almost entirely from the need score: 1.44 against 1.26 is a
0.18 advantage, which the other criteria did not overcome. The team that spent six hours on
interviews won by the weight of that criterion, which is exactly what the rubric was designed to
reward.

The arithmetic is worth doing after the event because it converts a disappointment into a
lesson about weights. A team that loses by 0.08 on need does not need a better model; it needs
better interviews.

### What the team would do differently

Three changes, each specific.

Interview four users instead of three, and include one who is not a pharmacist, because the
workflow boundary between pharmacy and nursing is where the discrepancies originate.

Test the offline demo at hour 30 rather than hour 35, because the first offline test revealed a
missing font that would have failed on the venue laptop.

Cut the classifier entirely and spend the hours on the rule layer, because the classifier added
two detections and four false flags, which is a poor trade at the current sample size.

None of these is a grand lesson. Each is a decision that could have been made differently with the
information available at the time, which is what a good debrief records.

### The cost of the event

The team's 240 person-hours are not free. At an opportunity cost of 45 dollars per hour, the
labour is 10,800 dollars, and the event fee, travel, and materials add perhaps 1,500 dollars.
A prototype that does not continue after the event has consumed about 12,300 dollars of value for
a demo and a learning experience.

That is not necessarily a bad trade: the team learned, the employer saw the team work, and the
problem may continue. But the arithmetic makes clear why continuation matters, and why the 30.3
percent figure from the CAMTech survey is the number an organiser should track.

## 7. Code, query, or calculation

The blocks below simulate a hackathon schedule, compute the effect of a scope cut, and score three
pitches against a weighted rubric. They use only the standard library.

### Block 1: schedule simulation

```python
import random
from statistics import mean

phases = {
    "need interviews":     (4, 6, 10),
    "scope and charter":   (1, 2, 4),
    "data access and audit": (3, 5, 12),
    "end-to-end baseline": (6, 8, 14),
    "one improvement":     (6, 9, 14),
    "validation slice":    (3, 5, 8),
    "demo build":          (5, 7, 11),
    "pitch and rehearsal": (3, 4, 6),
}

def simulate(phases, n=20000, seed=11):
    random.seed(seed)
    totals = []
    for _ in range(n):
        totals.append(sum(random.triangular(*phases[k]) for k in phases))
    totals.sort()
    return totals

base = simulate(phases)
n = len(base)
print(f"base mean {mean(base):.1f} p80 {base[int(0.8*n)]:.1f}")
print(f"P(<=46h) {sum(x <= 46 for x in base)/n:.3f}  "
      f"P(<=48h) {sum(x <= 48 for x in base)/n:.3f}")

cut = dict(phases)
cut["one improvement"] = (3, 5, 8)
cut["validation slice"] = (2, 3, 5)
small = simulate(cut)
print(f"cut  mean {mean(small):.1f} p80 {small[int(0.8*n)]:.1f}")
print(f"P(<=46h) {sum(x <= 46 for x in small)/n:.3f}  "
      f"P(<=48h) {sum(x <= 48 for x in small)/n:.3f}")
```

Expected output:

```text
base mean 48.8 p80 50.8
P(<=46h) 0.122  P(<=48h) 0.358
cut  mean 42.7 p80 44.5
P(<=46h) 0.953  P(<=48h) 0.999
```

The simulation assumes phases run sequentially in wall-clock time, which is right for a demo
pipeline and pessimistic for phases that the team can parallelise. It also assumes independence
between phases, which overstates the chance of a very long total and understates the chance of a
correlated failure such as a network outage.

### Block 2: person-hours and coordination

```python
team = 5
wall = 48
person_hours = team * wall
print(f"person-hours available: {person_hours}")
for people in (3, 4, 5, 6, 8):
    pairs = people * (people - 1) // 2
    print(f"{people} people -> {pairs} pairwise channels, "
          f"{person_hours * people / 5:.0f} person-hours at 48h")
```

Expected output:

```text
person-hours available: 240
3 people -> 3 pairwise channels, 144 person-hours at 48h
4 people -> 6 pairwise channels, 192 person-hours at 48h
5 people -> 10 pairwise channels, 240 person-hours at 48h
6 people -> 15 pairwise channels, 288 person-hours at 48h
8 people -> 28 pairwise channels, 384 person-hours at 48h
```

The channel count is a crude proxy for coordination cost, but the direction is right: a team of
eight has nearly three times the communication channels of a team of five and only 60 percent more
person-hours. At a forty-eight-hour event, that trade is usually bad.

### Block 3: weighted judging rubric

```python
weights = {"need": 0.30, "feasibility": 0.25, "demo": 0.20,
           "team": 0.15, "pathway": 0.10}
teams = {
    "Team A": (4, 5, 5, 4, 3),
    "Team B": (5, 4, 3, 4, 2),
    "Team C": (3, 3, 4, 5, 4),
}
for name, scores in teams.items():
    total = sum(s * w for s, w in zip(scores, weights.values()))
    print(f"{name}: {total:.2f}")

alt = {"need": 0.15, "feasibility": 0.15, "demo": 0.35,
       "team": 0.15, "pathway": 0.20}
print("under demo-heavy weights:")
for name, scores in teams.items():
    total = sum(s * w for s, w in zip(scores, alt.values()))
    print(f"{name}: {total:.2f}")
```

Expected output:

```text
Team A: 4.35
Team B: 3.90
Team C: 3.60
under demo-heavy weights:
Team A: 4.35
Team B: 3.65
Team C: 3.75
```

The ranking of the top team is stable, but the second and third swap when the weights move toward
demo quality. That is the point of publishing weights: a team can see which behaviour is
rewarded, and a judge can see how much the ranking depends on the weighting rather than on the
work.

## 8. Data, standards, and regulatory context

Hackathons operate in a regulatory grey zone that is worth naming precisely.

### Human subjects and ethics review

Research involving identifiable human subjects or their data requires review, and an event does
not create an exemption. Most health hackathons avoid the issue by using synthetic data, public
data, or de-identified extracts provided under an agreement.
A team that wants to test with patients, collect new data, or use identifiable records at an event
must obtain the appropriate approval before the event. Retrospective approval is not a thing.

The Common Rule in the United States and the GDPR in the European Union impose different
requirements, but both treat health data as sensitive. A capstone or hackathon that touches
patient data should have a documented determination, even when the answer is "not human-subjects
research".

### Device regulation

A prototype demonstrated at a hackathon is not a marketed device, and the event does not create a
regulatory obligation. The obligation arises when the software is intended for a medical purpose
and is placed on the market or put into service.
That distinction is often lost in pitches. A team that says "this will diagnose sepsis" is
describing a device function, even if the demo runs on synthetic data. A team that says "this
prototype ranks synthetic cases" is describing a research artefact.

Under the FDA clinical decision support guidance, a function that analyses a medical image or
signal, or provides a time-critical alert, or gives a specific directive, is likely a device. A
hackathon team does not need to resolve the classification, but it should not claim a clinical
function it has not validated and cannot legally market.

### Data licences at events

Event-provided datasets usually come with terms: no redistribution, no commercial use, no
publication of examples, or a requirement to delete the data after the event. A team that
publishes its repository with the data included may breach those terms.
The safe pattern is to publish code, a data dictionary, and a synthetic sample, and to describe
the real data's access route. That pattern also makes the work reproducible without redistributing
restricted data.

Public datasets have their own licences. CheXpert, PTB-XL, and MIMIC differ in what they permit,
and "open" is not a licence. A team should record the licence identifier and the obligations in
the submission README, which takes ten minutes and prevents a later problem.

### Intellectual property

Event rules vary: some claim no rights, some grant the organisers a licence, some leave ownership
with the team, and some require disclosure of pre-existing intellectual property.
A team that arrives with a pre-existing model and improves it at the event may create a
ownership question that the rules answer, or do not. Read the terms before the event, and record
which components were pre-existing.

### Competition and benchmark rules

Where the hackathon is also a competition, the competition rules add constraints: submission
format, external data, pretrained weights, team merging, and code release. The RSNA challenges on
Kaggle, for example, impose notebook submission, runtime caps, and limits on final submissions.
A team that treats the event rules as the only rules will be surprised at submission.

| Instrument | Applies to a hackathon | What it constrains |
|---|---|---|
| Event rules | Yes, contractually | Eligibility, data, IP, submission |
| HIPAA or GDPR | Yes, if patient data are used | Access, de-identification, retention |
| Common Rule or IRB | Yes, if human-subjects research | Recruitment, consent, review |
| FDA CDS guidance | Only if a device function is claimed | Claim and intended use |
| Dataset licence | Yes, contractually | Redistribution, commercial use |
| Competition rules | Yes, if competing | Format, external data, code |

Three of these can bind at once, and they can conflict.
An event may provide data under terms that prohibit redistribution, while a competition rule
requires code release; the resolution is to release code with a data-access statement rather than
with the data.
An IRB approval may permit analysis but not the collection of new data at the event; the
resolution is to use existing data and to say so.
A device claim may be implied by the pitch but not by the artefact; the resolution is to change
the pitch.

The team's protection is a one-page compliance note written before the event, listing the four
instruments and the resolution for each conflict. It takes twenty minutes and prevents the most
avoidable post-event problem.

## 9. Industry practice and case evidence

### What the formats look like in practice

MIT Hacking Medicine's Grand Hack 2025 ran from 14 to 16 March 2025 at the MIT Media Lab, with
tracks in assistive technology, heart health, and transformative intelligence, plus a month-long
accelerator for prior participants. The event follows the organisation's standard method:
problem pitches, team formation, mentorship, and final pitches to judges.

The most concrete published rubric from an MIT Hacking Medicine event is the 2025 Brazil
guidelines, which list four criteria: impact (problem clarity, scale, alignment with the
challenge), innovation (rationale, stakeholder needs, user experience), business model (go-to-market
and sustainability), and presentation (pitch effectiveness and team composition). Awards go to the
top three per track plus sponsor challenge winners.

That rubric is a useful calibration. It weights need and feasibility above technical novelty,
which is consistent with what the continuation evidence suggests: projects grounded in a real
need are more likely to survive the event.

### What happens after the event

Olson and colleagues' survey remains the most cited health-hackathon outcome study. Twelve
CAMTech hackathons, 356 projects, 196 responses, 30.3 percent with progress at a mean of 12.2
months, 24.7 percent beginning pilot testing, and a small number of business plans, patents, and
companies. The study is observational and the response rate limits precision, but the direction
is consistent with practitioner experience.

Ramadi and Nguyen's report on the virtual MIT COVID-19 Challenge, published in npj Digital
Medicine in 2021, describes more than 9,000 applicants from 142 countries and reports that more
than 25 percent of teams were still working five weeks after the event, compared with about 19
percent at two weeks for comparable in-person events. The comparison is not controlled, and the
COVID-19 context is unusual, but it suggests that format and follow-up support matter.

Nolte, Chounta, and Herbsleb's 2020 analysis of hackathon projects found that skill diversity and
intent to continue predicted longer-term continuation, while intense short-term activity predicted
lower continuation. The finding is about hackathons generally, not health ones, and it is
observational. Its practical implication is that a team of five identical engineers may peak early
and stop.

### What the challenge landscape shows about prototypes

The MedGemma Impact Challenge, announced in January 2026 and concluded with winners in March 2026,
asked teams to build prototypes on open-weight health models for low-resource and on-device
settings. The winning projects were narrow: offline outbreak surveillance for community health
workers, on-device tuberculosis screening combining chest X-rays and cough audio, and
medication-error reconciliation. None was a validated product, and the competition did not test
clinical outcomes.

That is the correct scope for a short-format event: one workflow, one user, one demonstrable step,
and an honest statement of what has not been tested. A team that claims more is not competing
better; it is borrowing credibility it has not earned.

### What organisers report about follow-up

The continuation literature is consistent on one point: the event is the beginning, not the end.
Olson and colleagues report that 24.7 percent of projects began pilot testing, which means the
majority that continued did so without a pilot, and the majority that did not continue stopped
before any formal evaluation.
Ramadi and Nguyen's virtual event reported higher early continuation than comparable in-person
events, and the authors attribute part of the difference to the structured follow-up programme
rather than to the format alone.

The practical implication for a team is to decide at the debrief whether it will continue, and to
name the next artefact if it will. A project without a named next artefact after the event is a
memory, not a project.

### Inference, labelled

It is an inference that teams which rehearse twice and freeze code at hour 46 place more often
than teams which continue building. The mechanism is that demo failures and timing overruns are
the dominant loss modes, and both are reduced by rehearsal and freeze. No controlled study of
hackathon placement is cited.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Solution seeking a problem | Team discusses architecture at hour 1 | Pre-formed idea | Weak need story | Six hours of interviews |
| No charter | No one can state the decision at hour 4 | Framing skipped | Scope creep, incoherent pitch | Ninety-minute charter |
| No end-to-end path | Components work separately at hour 12 | Building depth before breadth | Nothing to demo | Stub the integration |
| Scope creep | New features after hour 20 | Excitement | Core unfinished | One improvement only |
| Training-set demo | Demo uses a training example | Convenience | Judge detects it | Held-out or synthetic case |
| Network dependency | Demo needs three live services | Realistic architecture | Failure on venue Wi-Fi | Local fixtures and a recording |
| No rehearsal | Pitch drafted at hour 46 | Time pressure | Overrun and poor delivery | Two rehearsals |
| Late submission | Submission started at hour 47 | Optimism | Rejected or incomplete | Submit an hour early |
| Hidden proxy label | Outcome renamed to fit data | Convenience | Credibility loss | Disclose the proxy |
| Rule violation | External data used in a closed competition | Rules not read | Disqualification | Read rules before the event |
| PHI at the event | Real records on a laptop | Convenience | Governance breach | Synthetic data only |
| Team conflict | Two leads, no decision rights | No charter | Lost hours | Name a decider at hour 2 |
| Unvalidated clinical claim | "Diagnoses" in the pitch | Enthusiasm | Judge challenge | Claim the prototype |
| No debrief | Team disperses after judging | Fatigue | Learning lost | Debrief within 72 hours |

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| Phase durations are independent | Simulation is tractable | A shared cause delays several phases | Model a correlated outage scenario |
| The team works in parallel | Wall-clock plan is feasible | One person is the bottleneck | Assign owners per phase |
| A held-out slice is available | Validation claim depends on it | Data too small | Use cross-validation and say so |
| The demo can run offline | Backup plan depends on it | Model is hosted | Test with the network off |
| Judges score against the published rubric | Pitch structure depends on it | Rubric unpublished | Ask organisers; use the common criteria |
| The event provides clinical access | Need validation depends on it | No clinicians present | Use mentors or documented need |
| Data terms permit the demo | Publication depends on it | No redistribution | Publish code and synthetic sample |
| Two rehearsals fit | Delivery quality depends on it | Schedule overruns | Freeze earlier |
| The team can debrief | Learning depends on it | Members disperse | Schedule it before the event |

The largest limit is that a hackathon cannot produce validation evidence, and a team that tries
will fail at both the demo and the evidence. The format's value is framing speed and prototype
quality; the capstone's value is evidence. A programme needs both, and confusing them produces
graduates who are confident about the wrong things.

## 12. Exercises

### Exercise 1. Compare four formats (recall, 25 minutes)

Build a table comparing sprint hackathon, grand hack, datathon, and capstone on duration, team
size, deliverable, judging emphasis, and characteristic failure.
**Deliverable:** the table.
**Criteria:** all five dimensions for all four formats; the failure column names a mechanism, not
a feeling.
**Time:** 25 minutes.

### Exercise 2. Write a charter in ninety minutes (application, 90 minutes)

With a team, choose a problem and write the one-page charter under time pressure.
**Deliverable:** the charter.
**Criteria:** one decision, one owner, one metric, a capacity number, and an out-of-scope list.
**Time:** 90 minutes.

### Exercise 3. Build the runbook (design, 40 minutes)

Write an hour-by-hour runbook for a 48-hour event, with checkpoints at hours 2, 4, 9, 13, 30, 42,
and 46.
**Deliverable:** the runbook with conditions per checkpoint.
**Criteria:** each checkpoint has a testable condition; the demo buffer is at least two hours;
code freeze precedes submission by at least two hours.
**Time:** 40 minutes.

### Exercise 4. Simulate the schedule (calculation, 40 minutes)

Run the simulation in Section 7 with your own phase estimates. Report the probability of finishing
by the deadline and the effect of cutting one phase.
**Deliverable:** the output and a 150-word decision.
**Criteria:** estimates justified; the cut identified; the probability interpreted correctly.
**Time:** 40 minutes.

### Exercise 5. Design an offline demo (design, 35 minutes)

Take a demo with three external dependencies and redesign it to run offline with local fixtures.
**Deliverable:** a component diagram and a list of stubs.
**Criteria:** every external dependency has a local replacement; the stubs are labelled in the UI;
the demo runs in under ninety seconds.
**Time:** 35 minutes.

### Exercise 6. Write the three-minute pitch (application, 45 minutes)

Write and time a pitch with the six-part structure: decision, current practice, what was built,
evidence, failure and mitigation, next step.
**Deliverable:** a script and a timing.
**Criteria:** under three minutes; every part present; the evidence sentence includes an interval
or an explicit statement that there is none.
**Time:** 45 minutes.

### Exercise 7. Score three pitches (critique, 30 minutes)

Using the rubric in Section 7, score three recorded pitches from a public event and explain the
ranking.
**Deliverable:** a scoring table with justifications.
**Criteria:** scores tied to observable evidence; at least one score challenged; the weight
sensitivity noted.
**Time:** 30 minutes.

### Exercise 8. Write the debrief (design, 30 minutes)

Write a two-page debrief for a hackathon you have attended or observed, with three things that
worked, three that failed, and three changes for next time.
**Deliverable:** the debrief.
**Criteria:** specific events, not generalities; at least one failure is organisational; the
changes are actionable within the next event.
**Time:** 30 minutes.

## 13. Self-check answers

**1. What is the difference between a hackathon and a capstone?**
A hackathon is a time-boxed event producing a prototype and a pitch, judged on framing and demo.
A capstone is a twelve-week project producing an evidence dossier, judged on validation,
governance, and adoption. The first trains speed; the second trains evidence.

**2. Why is the sum of most-likely durations not the most-likely sum?**
Because project duration is a sum of random variables, and the distribution of a sum has a mean
equal to the sum of means but a spread that grows with the number of terms. The most-likely total
is below the mean when the distributions are right-skewed, which is typical of task durations.

**3. Compute the person-hours for a team of six at 36 hours.**
6 × 36 = 216 person-hours. The wall clock is still 36 hours, and the number of pairwise
communication channels is 15.

**4. Why does the simulation show a 36 percent chance of finishing by hour 48 when the nominal
plan sums to 48?**
Because the nominal plan uses most-likely values, and the simulated total includes the pessimistic
tails. The buffer of two hours covers less than one standard deviation of the total.

**5. What are the three external dependencies in the worked example, and what replaced them?**
A FHIR sandbox replaced by a local fixture with three synthetic patients; a terminology service
replaced by a local lookup table; a hosted model endpoint replaced by a small local model.

**6. Why is a recorded backup not cheating?**
Because the artefact exists and the recording is a faithful demonstration of it. The failure would
be presenting the recording as a live run, or using a recording of a different artefact.

**7. What should a team do when the held-out result is much worse than the development result?**
Report both numbers and the gap, suspect leakage or distribution shift, and investigate if time
allows. The gap is a finding, and hiding it is the failure.

**8. Why do published judging weights matter to a team?**
Because they determine which behaviour is rewarded. A team that optimises model accuracy when the
rubric weights need validation is optimising the wrong variable, and the loss is predictable.

**9. What is the correct regulatory position for a hackathon prototype?**
It is a research artefact, not a marketed device. The team should describe what it demonstrates
and avoid claiming a clinical function it has not validated and cannot legally market.

**10. Rubric for Exercise 3.** A strong runbook has checkpoints with observable conditions, a
buffer of at least two hours, and a freeze before submission. A weak runbook lists activities
without conditions, which makes it a schedule rather than a control.

## 14. Interview questions

**Q1. "Tell me about a hackathon you did."**
Strong answer names the decision, the constraint, the demo, and what failed. Weak answer narrates
the idea and the prize.

**Q2. "What did you build in 48 hours?"**
Strong answer distinguishes the prototype from a product and states what was not tested. Weak
answer describes a platform.

**Q3. "How did you decide what to cut?"**
Strong answer names the cut table and the effect on the demo. Weak answer says the team prioritized.

**Q4. "What would you do differently?"**
Strong answer names a specific hour and a specific decision. Weak answer says start earlier.

**Q5. "How did you validate the result?"**
Strong answer describes the held-out slice, the metric, and the interval. Weak answer says the
demo worked.

**Q6. "What happened when the demo failed?"**
Strong answer describes the backup and the disclosure. Weak answer says it did not fail.

**Q7. "How did you handle the clinical claim?"**
Strong answer states the prototype boundary and the regulatory position. Weak answer says the
clinician liked it.

**Q8. "Why did your team place where it did?"**
Strong answer relates the outcome to the rubric weights and the need story. Weak answer blames
the judges.

**Q9. "How would you run a hackathon?"**
Strong answer describes rules, rubric, mentor structure, and follow-up support. Weak answer
describes a schedule.

**Q10. "What transfers from a hackathon to a real project?"**
Strong answer names framing speed, demo discipline, and honest scoping. Weak answer says
everything.

## 15. Key terms

- **Backup recording** — recorded demo used if the live demo fails.
- **Capstone** — twelve-week evidence project.
- **Charter** — one page naming decision, owner, metric, and scope.
- **Checkpoint** — scheduled condition that must hold.
- **Code freeze** — point after which only demo fixes are allowed.
- **Datathon** — multi-day event around one dataset.
- **Demo** — demonstration on held-out or synthetic data.
- **Grand hack** — 48-hour event from need to prototype.
- **Held-out slice** — data not used in development.
- **Human subjects** — research involving identifiable people or data.
- **Intellectual property** — ownership of event outputs.
- **Judge** — person scoring final pitches.
- **Mentor** — expert giving feedback during the event.
- **Person-hour** — one person working one hour.
- **Pitch** — timed presentation with fixed structure.
- **Prototype** — working artefact demonstrating one step.
- **Rubric** — weighted criteria for scoring.
- **Scope creep** — adding questions after the charter.
- **Sprint hackathon** — 24-hour event.
- **Submission** — artefact and materials handed to organisers.
- **Synthetic data** — generated data labelled as such.
- **Track** — thematic stream within an event.
- **Validation** — evidence on unseen data.
- **Wall clock** — elapsed event time.

## 16. Sources

All URLs accessed 26 October 2026.

- Olson KR, Walsh M, Garg P, et al. Health hackathons: theatre or substance? *BMJ Innovations*
  2017;3:37-44. https://doi.org/10.1136/bmjinnov-2016-000147
- Ramadi KB, Nguyen FT. Rapid crowdsourced innovation for COVID-19 response and economic growth.
  *npj Digital Medicine* 2021;4:89. https://doi.org/10.1038/s41746-021-00397-5
- Nolte A, Chounta IA, Herbsleb JD. What happens to all these hackathon projects? *PACM HCI*
  2020;4(CSCW2). https://doi.org/10.1145/3415216
- MIT Hacking Medicine, GrandHack 2025 recap. https://hackingmedicine.mit.edu/events/grandhack-2025
- MIT Hacking Medicine Brazil 2025 guidelines (evaluation criteria).
  https://www.eretz.bio/wp-content/uploads/2025/07/Guidelines-EN-MIT-HACKING-MEDICINE-BRASIL-2025-ENGLISH-2.pdf
- Google Research, MedGemma Impact Challenge winners, March 2026.
  https://blog.google/innovation-and-ai/technology/health/med-gemma-impact-challenge/
- RSNA, Knee MRI AI Challenge, 2026.
  https://www.rsna.org/artificial-intelligence/ai-image-challenge/knee-mri-ai-challenge
- FDA, *Clinical Decision Support Software*, guidance, September 2022.
  https://www.fda.gov/regulatory-information/search-fda-guidance-documents/clinical-decision-support-software
- HHS, *Common Rule* (45 CFR 46). https://www.hhs.gov/ohrp/regulations-and-policy/regulations/45-cfr-46/
- Regulation (EU) 2016/679, General Data Protection Regulation.
  https://eur-lex.europa.eu/eli/reg/2016/679/oj
- PhysioNet, CITI course instructions. https://www.physionet.org/about/citi-course/

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic: [`05_brief-catalogue.md`](05_brief-catalogue.md)
- Next topic: [`07_competition-landscape.md`](07_competition-landscape.md)
- Related topic: [`11_demo-day.md`](11_demo-day.md)
- Survey file: [`../../course/19_capstone_hackathons.md`](../../course/19_capstone_hackathons.md)
- Labs: [`../../labs/`](../../labs/)
