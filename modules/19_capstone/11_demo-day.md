# Demo Day

> **Module 19 — Capstone, Hackathons, and Competitions**
> Topic 11 of 11
> Estimated study time: 3.5 hours
> Prerequisites: Topics 09 and 10
> Version 2026.3

## 1. Why this topic exists

Demo day is the ten minutes in which a year of work is compressed into a decision.
The panel has read the dossier, or part of it. The audience includes people who have not.
The team has one chance to say what changed, what the evidence is, what could go wrong, and what it
wants next.

The format is unforgiving for a specific reason: it forces prioritisation.
A ten-minute pitch cannot contain the pipeline, the feature engineering, the model comparison, the
subgroup analysis, and the value case. Something must be cut, and what a team cuts reveals what it
thinks matters.
A team that spends four minutes on architecture and one minute on the decision has told the
audience what it values, and the audience will believe it.

A second constraint is that the audience is not the team.
The team has lived with the project for twelve weeks and has forgotten which terms need defining,
which numbers need context, and which trade-offs were obvious only in hindsight.
The pitch must be written for someone who has five minutes of context and a decision to make, which
is why the structure starts with the decision rather than with the data.

A third constraint is that the room is mixed.
A clinical director, an engineer, a compliance officer, and a payer are in the same audience, and
each hears a different pitch.
The clinical director listens for the workflow, the engineer for the pipeline, the compliance officer
for the claim, and the payer for the cost.
A pitch that addresses only one of them loses the other three, which is why the nine segments exist:
each one is addressed to a different listener, and each one is short enough to be tolerated by the
others.

The practical consequence is that the pitch cannot be written by one person for one audience. It has
to be read by someone outside the project, who can say which sentence assumed knowledge and which
number needed context. That reader is the same stranger who runs the dossier test in Topic 09, and
the two tests are worth doing on the same afternoon.

The pitch also has a property the dossier does not: it cannot be skimmed. A reviewer can read a
dossier out of order; an audience cannot rewind. Every sentence has to land in sequence, which is why
the structure is fixed and the rehearsal is timed.

The failure mode is not usually a bad slide. It is a pitch that answers a question nobody asked.
The audience wants to know whether the decision is real, whether the evidence supports it, what
happens when it is wrong, whether it can legally reach a patient, and who pays.
A pitch that answers those five questions in order is hard to beat. A pitch that answers a
different five, however polished, leaves the audience with nothing to decide.

There is an industry parallel that is worth naming.
An internal model review, a procurement committee, and an investment committee all ask a version of
the same five questions.
The pitch is therefore a transferable artefact: the structure taught here is the structure a data
scientist uses to get a model approved, funded, or stopped.
The demo day is a rehearsal for that conversation, with a lower cost of failure.

This topic covers the structure, the five questions, the handling of a hostile question, and the
failure modes of a live demo.
It assumes the dossier from Topic 09 and the rubric from Topic 10, and it treats the pitch as the
last artefact in the chain rather than as a separate performance.

The last point is worth emphasising because teams often treat demo day as a performance that is
judged separately from the work.
It is not. Every number in the pitch comes from the dossier, every claim is bounded by the model
card, and every question is answerable from the repository. A pitch that is better than the dossier
is a warning sign; a pitch that is worse is a missed opportunity.
The goal is correspondence, not polish.

There is also a practical reason to prepare the pitch early: it exposes gaps in the evidence.
A team that cannot write the evidence sentence in week 9 has a problem with the validation, not with
the writing, and the discovery is more useful in week 9 than in week 12.
Writing the pitch is therefore a form of review, and the best teams write a draft before the dossier
is finished.

## 2. Learning outcomes

By the end of this topic you can:

1. **Structure** a ten-minute pitch that answers the decision, the evidence, the failure mode, the
   regulatory path, the value, and the ask, in that order.
2. **Write** a one-sentence ask that names a specific decision, a specific person, and a date.
3. **Deliver** a ninety-second demo on held-out or synthetic data, with a recorded backup and one
   failure case.
4. **Answer** the five questions a panel will ask, with a number and an interval where the evidence
   supports one.
5. **Handle** a hostile or unanswerable question without bluffing, deflecting, or overclaiming.
6. **Compute** a break-even value case and present it as a range with assumptions rather than a
   single number.
7. **Recover** from a demo, projector, or time failure with a prepared fallback.
8. **Critique** another team's pitch against the structure and the five questions.

Outcomes 1 to 3 are assessed in Exercises 1 to 3 and are visible in the first three minutes of the
pitch.
Outcomes 4 and 5 are assessed in Exercises 5 and 6 and tested in the question period.
Outcomes 6 and 7 are assessed in Exercises 4 and 7 and rehearsed twice before the day.
Outcome 8 is assessed in Exercise 8 and is the fastest way to improve one's own pitch.

The outcomes are ordered by the sequence of the pitch itself, which is deliberate: a reader can use
them as a checklist while watching a rehearsal and see immediately which segment is weak.

A note on what is not an outcome: "deliver a confident performance" is not observable and not
assessable. Confidence is a by-product of rehearsal, and the outcomes describe what rehearsal should
produce.

The outcomes are also a checklist for a listener. A mentor watching a rehearsal can mark each outcome
present or absent in real time, and the marks are more useful than a general impression.

The checklist is deliberately about content rather than delivery. Delivery improves with rehearsal
automatically; content does not.

## 3. The core mental model

A ten-minute pitch is a sequence of nine segments, each answering one question, followed by
questions from the panel.
The segments are ordered by the audience's decision process, not by the project's chronology.

```text
  MIN  0    1    2    3    4    5    6    7    8    9   10
       |    |    |    |    |    |    |    |    |    |
     decision  built  demo  failure  value  ask
       |    |    |    |    |    |    |    |    |    |
       1    1    1   1.5  1.5   1    1    1    1    1
       |    |    |    |    |    |    |    |    |    |
     owner cost  what evidence demo fail regulator value ask
             today

  FIVE QUESTIONS
  1 What decision changes, and who owns it?
  2 What is current practice, and what does it cost?
  3 What is the evidence, at what operating point?
  4 What happens when it is wrong?
  5 Can it legally reach a patient, and who pays?
```

Read the timeline as a budget with no slack.
The segments sum to ten minutes exactly, which means every overrun is paid for by a later segment.
The demo is the segment most likely to overrun, and the ask is the segment most likely to be cut,
which is the worst possible trade: the audience leaves without knowing what was requested.

The five questions map onto the segments, and the mapping is the reason for the order.
Question one is the first two minutes, because a decision without an owner is not a decision.
Question two is the second minute, because current practice is the comparator.
Question three is the evidence segment, because a number without a comparator is a claim.
Question four is the failure segment, because a system that cannot fail safely cannot be deployed.
Question five is the regulatory and value segments, because a claim that cannot legally reach a
patient is a research result.

Three properties of the model matter.

**The pitch is an argument, not a tour.** Each segment supplies a premise for the next. A tour
describes what the team did; an argument establishes what should happen. The audience is deciding,
and an argument gives them something to decide.

**The demo is evidence, not decoration.** It shows the artefact behaving on a case the team did not
tune on. A demo that shows a training example undermines the evidence segment, because the audience
cannot tell whether the result is real.

**The ask is the point.** A pitch without an ask is a presentation. The ask converts the audience
from spectators into decision-makers, and it is the segment that determines whether the ten minutes
change anything.

A fourth property is social.
The audience remembers one thing: a number, a failure case, or a sentence. A team that chooses
which one it wants to be remembered for will structure the pitch around it, and a team that does not
will be remembered for whichever segment went wrong.

### What the pitch is not

It is not the defense. The defense is twenty minutes of questions about the work; the pitch is ten
minutes of argument for a decision. A team that treats the pitch as a summary of the dossier will
run out of time before the ask.

It is not a product launch. The artefact is a prototype, and the pitch should say so. A prototype
presented as a product invites questions about maintenance, security, and regulation that the team
cannot answer in five minutes.

It is not a job interview. The audience is deciding about the project, not about the team. A team
that spends its time on how hard it worked has answered a question nobody asked.

The distinction matters because the three formats reward different things. The defense rewards
understanding, the launch rewards confidence, and the interview rewards narrative. The pitch rewards
a clear decision and the evidence for it.

### The pitch as a decision document

A useful reframing: the pitch is a decision document that happens to be spoken.
Its sections correspond to the sections a decision memo would have, compressed and delivered aloud.
The decision and owner are the recommendation; the comparator and evidence are the analysis; the
failure and regulatory segments are the risk section; the value case is the budget impact; the ask is
the recommendation repeated.

If the pitch were written instead of spoken, it would be about two pages. The ten minutes are not
there because the content is large; they are there because a room full of people needs to hear the
argument in a shared order.

That shared order is the whole contribution of this topic. The content comes from the dossier; the
order is what makes it a decision.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Pitch | A timed argument for a decision | A project presentation | Ten minutes |
| Ask | The specific decision or resource requested | A general hope | One sentence |
| Segment | A timed part of the pitch answering one question | A slide | Seconds |
| Decision statement | The action and owner the pitch is about | The project title | One sentence |
| Comparator | Current practice measured on the same cohort | No model at all | Rate or cost |
| Evidence sentence | The primary result with an interval and a split | An AUC | One sentence |
| Failure case | A demonstrated limitation with mitigation | A disclaimer | One example |
| Regulatory paragraph | The regime, classification, and binding artefact in one paragraph | A legal opinion | Four sentences |
| Value case | Cost, saving, and payback with assumptions | A revenue projection | Currency and months |
| Break-even | The effect size at which costs equal savings | The expected effect | Effect size |
| Leave-behind | The one-pager left with the audience | The slide deck | One page |
| Q&A sheet | Prepared answers with numbers and artefacts | A script | One page |
| Recovery | Continuing after a failure without losing the argument | Restarting | Technique |
| Premise | The assumption embedded in a question | The question itself | Sentence |
| Compression | The two-minute version of the pitch | A summary | Two minutes |
| Hostile question | A question designed to expose a weakness | Any difficult question | Question type |
| Bridging | Acknowledging a question and redirecting to relevant evidence | Dodging | Technique |
| Bluffing | Answering without evidence | Confidence | Failure mode |
| Backup recording | A recorded demo used if the live demo fails | A screenshot | Video |
| Timekeeper | The person who signals segment boundaries | The presenter's watch | Role |
| Moderator | The person who manages questions | The chair | Role |
| Q&A | The question period after the pitch | A discussion | Minutes |
| Narrative arc | The sequence from problem to ask | A list of achievements | Structure |
| One-pager | The readout left with the audience | The slide deck | One page |
| Claim boundary | What the evidence supports and what it does not | A disclaimer | Two sentences |
| Pilot | A bounded deployment to generate evidence | A full deployment | Scope and duration |
| Deployment | The output reaching a user who acts on it | A demo | Maturity level |
| Procurement | The process by which an organisation buys | A purchase | Process |
| Budget holder | The person who controls the money | The user | Role |
| Adoption | Sustained use by the intended user | Interest | Rate |

The most consequential definition is the ask.
An ask is not "we hope this is useful" or "we would like feedback". It is a request for a specific
decision from a specific person by a specific date: "we ask the ED director to approve a
three-month pilot, with two analysts, starting in January".
A pitch without that sentence leaves the audience with nothing to do, and an audience with nothing
to do does nothing.

A second definition matters for honesty: the claim boundary.
Every pitch should contain a sentence that says what the evidence does not show. It is
counter-intuitive to volunteer a limitation, and it is the single most credible move in a
ten-minute pitch, because it demonstrates that the team knows the difference between a result and a
claim.

A third definition is the compression.
The two-minute version is not a summary; it is the argument with the supporting detail removed. If
the two-minute version requires the audience to accept a premise, the ten-minute version must
establish it, which is a useful test of whether the longer pitch has a spine.

## 5. The mechanics, step by step

The procedure has fifteen steps, split between preparation (two weeks before) and delivery (the
day). It assumes a ten-minute slot and a panel of three to five.

### Preparation

#### Step 1. Identify the audience and the decision they can make

**Input:** the invitation, the programme, and the attendee list.

**Output:** three named people in the audience and the decision each can make.

**Rationale:** A pitch is written for a decision, and a decision belongs to a person. An audience of
peers can offer critique; an audience with a budget holder can offer a pilot. The pitch differs.

**Branch point:** if the audience cannot make any decision, the pitch is a presentation. Say so in
the ask: "we ask for your critique on the validation design" is a legitimate ask for a peer
audience.

#### Step 2. Write the ask first

**Input:** the audience analysis.

**Output:** one sentence: "We ask [person] to [decision] by [date], because [evidence]."

**Rationale:** Writing the ask first determines what the pitch must establish. A pitch that asks
for a pilot must establish feasibility and a plausible effect; a pitch that asks for a stop must
establish that the evidence does not support deployment.

**Branch point:** if the ask cannot be stated in one sentence, the team does not yet know what it
wants. Return to the charter and the scope memo.

#### Step 3. Choose the one thing the audience should remember

**Input:** the ask and the evidence.

**Output:** one sentence, one number, or one image.

**Rationale:** The audience will remember one thing regardless. Choosing it is better than leaving
it to chance. The candidate is usually the failure case or the comparator gap, because both are
concrete and both are honest.

**Branch point:** if the team cannot choose, the evidence is not yet clear enough to pitch. A pitch
that tries to make five things memorable makes none.

#### Step 4. Build the nine segments to the time budget

**Input:** the ask, the memorable thing, and the evidence.

**Output:** a segment plan with a duration for each, summing to ten minutes.

**Rationale:** The budget forces the trade-offs. A team that plans four minutes for the model has
three minutes for everything else, and the demo, the failure, and the ask will be rushed.

**Branch point:** if the segments do not fit, cut the "what we built" segment before the evidence or
the ask. The audience can read the method in the dossier.

A useful test for the budget is to say the pitch out loud at a normal pace and mark where each
segment begins. Most teams discover that the first two segments take ninety seconds longer than
planned, because the decision needs context the team has stopped noticing it is providing.

#### Step 5. Write the evidence segment as one sentence and one table

**Input:** the validation report.

**Output:** a sentence with the primary metric, the interval, the comparator, and the split; a table
with those numbers and the subgroup gap.

**Rationale:** The evidence segment is where the pitch earns or loses credibility. A sentence with
an interval and a comparator is credible; an AUC without either is not.

**Branch point:** if the interval is wide, say so and explain what would narrow it. A wide interval
presented honestly is stronger than a point estimate presented confidently.

#### Step 6. Write the failure segment

**Input:** the error analysis and the risk register.

**Output:** one failure case, its consequence, the mitigation, and the monitoring signal.

**Rationale:** The failure segment answers the question the audience is most afraid to ask. A team
that answers it before it is asked demonstrates control; a team that waits to be asked looks
defensive.

**Branch point:** if the failure case is a false negative with serious consequence, do not soften
it. The mitigation is the point, not the absence of failure.

#### Step 7. Write the regulatory paragraph

**Input:** the regulatory memo.

**Output:** four sentences: the regime, the classification, the binding artefact, and the open
question.

**Rationale:** The paragraph is short because the audience needs the conclusion, not the analysis.
A team that says "it is probably a device, the binding artefact is a 510(k), and the open question
is whether triage claims need a reader study" has answered the question.

**Branch point:** if the classification is uncertain, say both readings and what would resolve it.
An uncertain paragraph is more credible than a confident wrong one.

#### Step 8. Build the value case as a range

**Input:** the comparator cost, the expected effect, and the implementation cost.

**Output:** a monthly saving, a payback period, and a break-even effect size, each with the
assumptions named.

**Rationale:** A single number invites disbelief. A range with a break-even shows that the team has
tested the economics and knows what would have to be true.

**Branch point:** if the break-even effect exceeds any plausible effect, the honest ask is a pilot
or a stop, not a deployment. That is a legitimate and strong pitch.

#### Step 9. Write the demo script to ninety seconds

**Input:** the prototype and a held-out case.

**Output:** a script with five beats: input, output, reason, failure, close.

**Rationale:** The script prevents the demo from becoming a tour of the interface. Five beats in
ninety seconds is about eighteen seconds each, which is enough for one sentence per beat.

**Branch point:** if the demo needs a network or a login, replace it with a local fixture. A demo
that fails on stage costs more than a demo that is less impressive.

#### Step 10. Build the backup

**Input:** the working demo.

**Output:** a recording of the demo, tested on the presentation machine, plus a one-page readout for
the audience.

**Rationale:** The backup is not a contingency for a rare event; it is the primary plan for a
conference network. The recording also lets the presenter keep talking if the live demo stalls.

**Branch point:** if the recording and the live demo differ, say so. A recording of a different
version is a misrepresentation.

#### Step 11. Rehearse to time, twice

**Input:** the segment plan and the script.

**Output:** two timed runs, one with a mentor, and a list of cuts.

**Rationale:** The first run always overruns. The second confirms the cuts. A pitch that has not
been timed will overrun on stage, and the segment that gets cut is the ask.

**Branch point:** if the pitch still overruns after two runs, cut a segment rather than speaking
faster. Speed is not compression.

#### Step 12. Prepare answers to the five questions and ten follow-ups

**Input:** the dossier and the red-team list.

**Output:** a one-page Q&A sheet with the answer, the number, and the artefact for each.

**Rationale:** The five questions are predictable, and the follow-ups are predictable from the
failure modes. Preparing them is the highest-return hour in the process.

**Branch point:** if an answer requires a number the team does not have, prepare the sentence "we
did not measure that, and here is what it would take". That sentence is stronger than an estimate
invented on the spot.

#### Step 13. Prepare the hostile-question protocol

**Input:** the three most likely hostile questions.

**Output:** a protocol: acknowledge, answer with evidence, state the boundary, and offer the next
step.

**Rationale:** A hostile question is usually a fair question asked sharply. The protocol prevents
defensiveness and gives the audience a structured answer.

**Branch point:** if the question contains a false premise, correct the premise before answering.
Answering a false premise accepts it.

#### Step 14. Check the logistics

**Input:** the room, the equipment, and the schedule.

**Output:** a tested setup: laptop, adapter, projector resolution, audio, clicker, timer, and a
printed backup of the one-pager.

**Rationale:** Logistics failures are the most avoidable and the most common. A pitch that starts
four minutes late because of an adapter has lost the audience before the first sentence.

**Branch point:** if the room cannot be tested in advance, arrive early and test. Do not rely on the
organiser's assurance.

#### Step 15. Close the loop after the pitch

**Input:** the questions, the feedback, and the contacts.

**Output:** a follow-up note within forty-eight hours with the one-pager, the answers to any
unanswered questions, and the specific next step.

**Rationale:** The pitch creates interest; the follow-up converts it. A team that does not follow up
has had a conversation, not a decision.

**Branch point:** if the answer to a question required new analysis, send it when it is done rather
than delaying the whole follow-up.

### Step 16. Build the one-pager

**Input:** the readout from Topic 09 and the ask.

**Output:** one page with the decision, the evidence sentence, the table, the failure, the
regulatory line, the value range, and the ask, plus a contact.

**Rationale:** The one-pager is what the audience keeps. A pitch that ends without a leave-behind is
remembered as an impression; a pitch with a one-pager is remembered as a document.

**Branch point:** if the one-pager repeats the slides, cut it to the four numbers and the ask. Its
job is to be read in ninety seconds a week later.

### Step 17. Assign roles in the room

**Input:** the team and the segment plan.

**Output:** a presenter, a timekeeper, a demo operator, and a note-taker, each named.

**Rationale:** A team that all present together loses time to handoffs and looks uncoordinated. One
presenter with three supporting roles is cleaner, and the note-taker is the person who captures the
questions for the follow-up.

**Branch point:** if the panel expects each member to speak, assign one segment each and rehearse
the handoffs. The individual defense is a separate assessment, and the pitch should not try to be
one.

### Step 18. Write the two-minute version

**Input:** the ten-minute pitch.

**Output:** a two-minute version with the decision, the evidence sentence, the failure, and the ask.

**Rationale:** The two-minute version is what a team uses in a corridor, a follow-up meeting, or an
interview. It is also a test of whether the pitch has a spine: if the argument survives compression,
it is an argument; if it collapses into a list of activities, it was a tour.

**Branch point:** if the two-minute version cannot include the evidence sentence, the evidence is
not yet a sentence. Rewrite it until it fits.

### The five questions, in the order they are asked

The panel's questions usually follow the pitch, and they usually follow the same order.

First, whether the decision is real. This is a check on the owner and the workflow, and it is
answered by the first two segments or not at all.
Second, what the comparator is. This is the question that separates a project from a demo, and it is
answered by the second segment and the evidence table.
Third, whether the evidence supports the claim. This is where the interval, the split, and the
subgroup table are examined.
Fourth, what happens when it is wrong. This is where the failure segment earns its place.
Fifth, who pays and whether it is legal. This is where the value case and the regulatory paragraph
are tested.

A team that has prepared answers in this order will find that the questions arrive in the order it
rehearsed, which is the point of rehearsing them.

The order is not a coincidence. A panel that is deciding whether to recommend a pilot works through
the same logic as the pitch: is the problem real, is the comparison fair, is the evidence sound, is
the risk managed, is the pathway legal and affordable. The pitch is a rehearsal of the panel's own
reasoning.

### Decision table: handling a hostile question

| Question type | Example | Response pattern | Avoid |
|---|---|---|---|
| Evidence challenge | "Your AUC is not better than the existing score" | Acknowledge, give the interval, state what the comparator was, offer the decision curve | Claiming significance from a point estimate |
| Design challenge | "Your validation is not external" | Agree, name the level, state what it supports, describe the study that would | Defending the design as external |
| Safety challenge | "What happens when it misses a case" | Give the failure rate, the mitigation, and the monitoring trigger | Saying it will not miss |
| Regulatory challenge | "Is this a device" | Give the classification, the criteria, and the open question | "It is research only" |
| Value challenge | "Who pays for this" | Name the budget holder, the saving range, the payback, and the break-even | A single optimistic number |
| Ethical challenge | "Who is excluded from your data" | Name the groups, the gap, and the mitigation | "We found no bias" |
| False premise | "So you are replacing radiologists" | Correct the premise, then answer the real question | Answering the premise as stated |
| Unanswerable | "What is the effect on length of stay" | Say it was not measured, say what would measure it, offer the proxy result | Inventing an estimate |

### Edge cases and forced decisions

**The demo fails.** Switch to the recording, say that it is a recording, and continue. Do not debug
in front of the audience; the audience remembers the recovery, not the failure.

**The projector fails.** Use the one-page readout and speak. A team that can deliver the argument
without slides has understood it; a team that cannot has memorised a deck.

**The pitch overruns.** Cut the "what we built" segment and go to the evidence. Never cut the ask.

**A panel member interrupts.** Answer briefly, then say "I will come back to that in the evidence
segment" and do so. An interruption is a signal of interest, not an attack.

**Nobody asks a question.** Ask yourself one: "the question we expected was about external
validity, and here is the answer". The silence is usually the audience waiting for permission.

**A question reveals a real gap.** Say so, record it, and follow up. A gap acknowledged in the room
is a limitation; a gap discovered afterwards is a credibility problem.

**The ask is refused in the room.** Thank the person, ask what evidence would change the decision,
and record it. A refusal with a condition is a next step.

## 6. Worked example

A team pitches a readmission risk model to a mixed audience: the chief medical officer (CMO), the
transitional-care director, a payer representative, and three faculty members.
The slot is ten minutes plus five minutes of questions.

### The ask

The team writes the ask first:

"We ask the transitional-care director to approve a three-month pilot with two care coordinators,
starting in February, because the model identifies 1.8 times as many high-risk discharges as the
current rule at the same call volume."

The ask names a person, a decision, a date, and a reason. It is also the sentence the team will
repeat at the end, word for word.

### The time budget

The segments sum to ten minutes: decision and owner 1.0, current practice and cost 1.0, what we
built 1.0, evidence 1.5, demo 1.5, failure and mitigation 1.0, regulatory path 1.0, value and
adoption 1.0, ask 1.0.

The budget has no slack. That is deliberate: a budget with slack invites the demo to expand into
it, and a demo that expands is a demo that runs long.

### The evidence segment

One sentence and one table.

"On a temporal holdout of 1,840 discharges, the model reached an AUROC of 0.71 (95 percent CI 0.67
to 0.75), against 0.64 (0.60 to 0.68) for the current rule; at the call volume the programme can
staff, it captured 41 percent of readmissions against 23 percent."

The table shows the two rows, the interval, the call volume, and the subgroup gap by language
preference (0.06, with an interval that includes zero).

The sentence is credible because it has a comparator, an interval, a split, and an operating point.
It is also vulnerable, and the team knows how: the interval overlaps the comparator's upper bound in
the smallest subgroup, and the temporal split is single-site.

### Diagnostic one: the ask is not a decision anyone can make

Before the rehearsal, the ask was "we ask for support to continue this work". The mentor asked who
would decide and what they would decide, and the team could not answer.

That is the first diagnostic. An ask without a named decider is a hope, and a pitch built on a hope
has no ending. The rewrite took twenty minutes and changed the whole structure, because the evidence
segment now had to establish the call volume that the director could staff.

### Diagnostic two: the demo depends on live data

The demo used a live query against the hospital's readmission registry, which required a network and
a login. The rehearsal failed twice.

That is the second diagnostic. A demo with two external dependencies and a five-minute question
period is a demo that will fail. The team built a local fixture with five synthetic discharges,
including one failure case, and recorded it.

### The value case

The team computes the economics.

Current practice: 420 boarder-hours per month, at 38 dollars per hour, with a readmission-related
cost that the programme attributes partly to delayed follow-up. The model's expected effect is an 18
percent reduction in avoidable readmissions among the cohort it identifies, which the team treats as
an assumption with a range of 10 to 25 percent.

At 18 percent, the monthly saving is 420 × 0.18 × 38 = 2,873 dollars.
A full implementation is estimated at 180,000 dollars with 24,000 dollars per year in running costs.
The net monthly benefit is 2,873 - 2,000 = 873 dollars, and the payback is 180,000 / 873 = 206
months.

That is the moment the pitch changes. The full implementation does not pay back in any plausible
horizon, and a team that presents a payback of 206 months as a benefit will lose the room.

The team reframes the ask as a pilot: 30,000 dollars implementation, 6,000 dollars per year running.
At 18 percent, the net monthly benefit is 2,873 - 500 = 2,373 dollars, and the payback is 12.6
months. At 10 percent it is 27.4 months; at 25 percent it is 8.6 months.

The pitch now says: "the full programme does not pay back at our current estimate; a pilot does, and
it would measure whether the 18 percent assumption holds. That is what we are asking for."

### Varying one assumption

Suppose the payer representative says the readmission penalty is not 38 dollars per boarder-hour but
a bundled payment that makes each avoided readmission worth 9,000 dollars.

The arithmetic changes completely. If the cohort has 60 avoidable readmissions per year and the
model prevents 18 percent of them, the annual saving is 60 × 0.18 × 9,000 = 97,200 dollars, and the
full implementation pays back in about 26 months.

The team does not know which number is right, and it says so. The pitch presents both framings, names
the assumption each depends on, and asks the pilot to measure the effect rather than assume it.

### The pitch as delivered

The ten minutes ran to 9 minutes 40 seconds.

Minute 0 to 1: the decision and the owner, with the director named in the room.
Minute 1 to 2: current practice, the 23 percent capture rate, and the 420 boarder-hours.
Minute 2 to 3: what was built, in three sentences, with the pipeline on one slide.
Minute 3 to 4.5: the evidence sentence and the table.
Minute 4.5 to 6: the demo, on a synthetic case, including the failure case.
Minute 6 to 7: the failure and the monitoring trigger.
Minute 7 to 8: the regulatory paragraph and the open question.
Minute 8 to 9: the value case as a range, with the break-even.
Minute 9 to 9.67: the ask, repeated word for word.

The questions took thirteen minutes, and three of the five prepared answers were used.

### The hostile question

The payer representative asked: "So you are replacing care coordinators with a model?"

The team did not answer the premise. It corrected it: "No. The coordinators make the calls; the
model changes which twenty they call first. At the current staffing, the model captures 41 percent
of readmissions against 23 percent for the rule."

Then it answered the real question, which was about cost: "The pilot costs 30,000 dollars and
measures whether the 18 percent holds. If it does not, we stop."

The exchange took forty seconds and it was the strongest moment in the pitch, because it showed that
the team had thought about the workforce and the economics before being asked.

### The questions that were not prepared

Two questions were not on the list.

A faculty member asked whether the model's subgroup gap by language preference was stable across
years. The team had the subgroup estimate for the pooled period and not by year, and it said so:
"we have the pooled estimate, which is 0.06 with an interval that includes zero; we have not
stratified by year, and that is a limitation."

A payer representative asked what happens when a patient is readmitted to a different hospital. The
model's label was defined within the network, and the team had not considered out-of-network
readmissions. It said so, and added that the outcome definition would need to change before a
multi-payer deployment.

Neither answer was polished. Both were credible, because each named the gap and the reason. The
panel's feedback mentioned the second answer as evidence that the team understood its data.

### The one-pager

The team left a single page with six elements: the decision and owner, the evidence sentence with the
interval, the comparator row, the failure and the monitoring trigger, the regulatory line, and the
ask with a date.

It also included a QR code to the repository and the name of the person who would answer follow-up
questions. That last detail is small and it is the difference between a leave-behind and a souvenir.

### What the team would change

Three things, recorded in the debrief.

It would move the failure segment earlier, because the payer's question showed that the audience was
already thinking about risk and the team made them wait.
It would prepare the out-of-network question, which was predictable from the outcome definition.
It would cut the "what we built" segment to thirty seconds and give the time to the evidence, because
the audience read the pipeline slide faster than the presenter described it.

## 7. Code, query, or calculation

The first block checks the pitch budget. The second computes the value case and the break-even. They
use only the standard library.

```python
segments = [
    ("decision and owner", 1.0),
    ("current practice and cost", 1.0),
    ("what we built", 1.0),
    ("evidence", 1.5),
    ("demo", 1.5),
    ("failure and mitigation", 1.0),
    ("regulatory path", 1.0),
    ("value and adoption", 1.0),
    ("the ask", 1.0),
]
total = sum(s for _, s in segments)
for name, secs in segments:
    print(f"{name:28s} {secs:4.1f} min")
print(f"total {total:.1f} min, slack {10 - total:+.1f} min")
```

Expected output:

```text
decision and owner            1.0 min
current practice and cost     1.0 min
what we built                 1.0 min
evidence                      1.5 min
demo                          1.5 min
failure and mitigation        1.0 min
regulatory path               1.0 min
value and adoption            1.0 min
the ask                       1.0 min
total 10.0 min, slack +0.0 min
```

A budget with zero slack is the design, not an accident. If a segment overruns, another must shrink,
and the team should decide which before the day rather than during the pitch.

```python
boarder_hours = 420
cost_per_hour = 38
implementation = 30_000
running = 6_000

def payback(effect, impl=implementation, run=running):
    monthly = boarder_hours * effect * cost_per_hour
    net = monthly - run / 12
    return monthly, net, impl / net if net > 0 else float("inf")

for effect in (0.10, 0.18, 0.25):
    monthly, net, months = payback(effect)
    print(f"effect {effect:.0%}: saving/mo {monthly:,.0f}, net {net:,.0f}, "
          f"payback {months:.1f} months")

full_monthly, full_net, full_months = payback(0.18, 180_000, 24_000)
print(f"full programme at 18%: net {full_net:,.0f}/mo, payback {full_months:.0f} months")
```

Expected output:

```text
effect 10%: saving/mo 1,596, net 1,096, payback 27.4 months
effect 18%: saving/mo 2,873, net 2,373, payback 12.6 months
effect 25%: saving/mo 3,990, net 3,490, payback 8.6 months
full programme at 18%: net 873/mo, payback 206 months
```

The full-programme payback of 206 months is the number that changes the ask. A team that presents it
as a benefit has not done the arithmetic; a team that presents it as the reason for a pilot has.

The break-even effect is the effect at which net benefit is zero: run/12 divided by (boarder_hours ×
cost_per_hour) = 500 / 15,960 = 3.1 percent. Any effect above 3.1 percent pays back the pilot's
running cost, and the pilot's purpose is to measure whether the effect is above the implementation
threshold.

### Reading the output

The first block is a check, not a plan. A budget that sums to ten minutes with no slack is correct;
a budget that sums to nine minutes has slack that the demo will absorb.

The second block produces three numbers the pitch needs: the saving at the assumed effect, the
payback, and the break-even. A team that presents only the first is presenting a benefit without a
cost; a team that presents only the second is presenting a cost without a benefit.

The full-programme payback of 206 months is not a failure of the arithmetic. It is a finding about
the implementation cost relative to the effect, and it is the reason the ask is a pilot. A pitch that
reports it honestly is more credible than one that reports a rounded 24 months by assuming a larger
effect.

### A note on the assumptions

The 18 percent effect is an assumption, not a measurement, and the pitch should say so. The 38
dollars per boarder-hour is an estimate from the finance team, and the pitch should say that too.
Two assumptions, both named, are enough for a ten-minute audience; a sensitivity analysis belongs in
the dossier.

The break-even is the most useful number in the pitch because it converts the argument from "trust
our estimate" to "here is the threshold the pilot must clear". An audience can evaluate a threshold;
it cannot evaluate a prediction.

## 8. Data, standards, and regulatory context

A pitch is communication, and communication about medical products is regulated in ways that a
capstone team should know even when it is not selling anything.

### Claims and promotion

In the United States, the Federal Trade Commission Act prohibits deceptive advertising, and the FDA
regulates promotion of medical devices, including claims made at conferences and in investor
materials. A claim that a product diagnoses or treats a condition is a promotional claim, and it
must be consistent with the device's cleared or approved intended use.
A capstone is not promotion, and a team that presents a prototype as a product is making a claim it
cannot support. The practical rule is to describe what the artefact does on the data, not what it
will do for patients.

### Intended use and labelling

The FDA clinical decision support guidance ties the regulatory status to intended use. A pitch that
states an intended use different from the dossier's is a labelling inconsistency, and in a company
it would be a regulatory problem.
The pitch should use the dossier's intended-use sentence verbatim. If the sentence is too long for a
slide, the slide can shorten it, and the spoken version should be exact.

### Privacy in demos

A demo that shows patient data is a disclosure. Even a de-identified record can be re-identifiable in
context, and a screenshot can travel further than a slide.
The safe pattern is synthetic or public data, labelled as such, with no real identifiers and no
free-text fields that could contain them. A team that demos on real records without an approval has
created an incident, and the panel is obliged to report it.

### Accessibility and inclusion

A pitch should be accessible: readable contrast, captions on videos, no colour-only encoding, and a
spoken description of any chart. Accessibility is a design requirement in many organisations and a
courtesy everywhere. It also improves the pitch for everyone in a large room.

### Disclosure of AI assistance

If the team used a large language model to draft text, generate code, or summarise literature, the
programme's disclosure rule applies. The disclosure belongs in the dossier's appendix and, where the
work is presented as the team's own analysis, in the pitch's methods sentence.
The rule is not a prohibition; it is a requirement that the audience can attribute the work
correctly.

### What the standards do not cover

No standard tells a team how to handle a hostile question, how to recover from a failed demo, or how
to ask for a pilot. Those are craft, and they are learned by rehearsal and by watching other people
do them badly.

### A note on the room

The regulatory and ethical constraints apply to the room as well as to the slides. A pitch delivered
to an audience that includes patients, or that is recorded and posted publicly, reaches further than
the team may have considered.

A recorded pitch is a publication. A claim in a recorded pitch is a public claim, and it can be cited
after the team has moved on. The safe rule is that every sentence in a recorded pitch should be one
the team would be willing to defend in a year, which usually means the same claim boundary as the
dossier.

A pitch to a patient audience adds a further obligation: plain language, no false reassurance, and no
description of a benefit that has not been measured. The same rule that applies to a clinical
conversation applies to a public presentation about a clinical tool.

| Instrument | Applies to a pitch | What it constrains |
|---|---|---|
| FTC Act | Commercial promotion | Truthful claims |
| FDA device labelling | Device claims | Consistency with intended use |
| FDA CDS guidance | Classification | What counts as a device function |
| HIPAA | Any patient data shown | Disclosure and minimum necessary |
| GDPR | Any personal data shown | Lawful basis and minimisation |
| Programme disclosure rule | Assessed work | Attribution of AI assistance |
| Accessibility standards | Public presentations | Contrast, captions, alternatives |

## 9. Industry practice and case evidence

### What pitch formats reward

MIT Hacking Medicine's Brazil 2025 guidelines, which follow the organisation's methodology, assess
pitches on impact, innovation, business model, and presentation. The weights reward a clear problem
and a credible pathway over technical novelty.
That ordering is consistent with the continuation evidence: projects with a real need and a real
user are more likely to survive than projects with a clever model.

The MedGemma Impact Challenge, concluded in March 2026, awarded prototypes that addressed a specific
user in a specific setting: offline outbreak surveillance for community health workers, on-device
tuberculosis screening, medication-error reconciliation. The winning pitches were narrow, and the
projects were prototypes rather than validated products. A pitch that claimed a validated product
would have been inconsistent with the artefact.

### What evaluation programmes report about adoption pitches

NHS England's guidance on real-world AI evaluations describes the evidence that deploying
organisations need: a named owner, a workflow, an evaluation design, and a plan for generating
evidence during deployment. A pitch that addresses those four is speaking the language the
organisation uses.

The Royal College of Radiologists' 2025 consultation document on AI deployment fundamentals notes
that marking is not evaluation. A pitch that says "CE marked, therefore effective" is making a claim
the profession has explicitly rejected.

### What the value-case literature implies

The arithmetic in Section 6 is a simplified budget-impact calculation. A full budget-impact analysis
would include implementation, running costs, the cost of the comparator, the cost of false positives,
and the time horizon, and would present a range rather than a point.
The simplification is acceptable for a ten-minute pitch if the assumptions are stated, which is why
the worked example names the 18 percent and the 38 dollars per hour rather than presenting the
payback as a fact.

### Inference, labelled

It is an inference that pitches which state a limitation before being asked are rated more credible.
The mechanism is that volunteering a boundary signals control, and the audience's prior is that
every model has one. No controlled study of capstone pitches is cited.

### What an audience actually remembers

Ask any panel member a week after a demo day what they remember, and the answer is usually one of
three things: a number, a failure case, or the ask.

The number is remembered when it is comparative and concrete: "41 percent against 23 percent at the
same call volume" survives; "an AUC of 0.71" does not.
The failure case is remembered when it is specific and mitigated: "it flags a patient who is not
readmitted about one time in six, and the coordinator rejects it in under a minute".
The ask is remembered when it names a person and a decision: "the director approves a three-month
pilot" survives; "we seek support" does not.

A team that designs the pitch around those three memories has designed it around what the audience
retains, which is the only thing that influences a decision after the room empties.

### What the pitch cannot fix

A pitch cannot rescue a weak comparator, a wide interval, or an unclear owner. It can present them
honestly, and honesty is worth more than polish, but the underlying evidence is what the audience
will test.

This is why the topic comes last. The pitch is the final compression of work that either supports a
claim or does not, and the ten minutes are spent revealing which.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| No ask | The last slide says "thank you" | Pitch treated as a report | Audience does nothing | Write the ask first |
| Ask without a decider | "We seek support" | No named person | No decision | Name person, decision, date |
| Evidence without comparator | "Our AUC is 0.82" | Model-centric framing | Audience cannot judge | Comparator and interval |
| Demo on training data | The case looks familiar | Convenience | Credibility loss | Held-out or synthetic case |
| Demo depends on network | Live service in the room | Realistic architecture | Public failure | Local fixture and recording |
| Overrun | Rehearsal ran to twelve minutes | No cuts | Ask is cut | Two timed rehearsals |
| Failure segment omitted | No slide on what goes wrong | Discomfort | Audience asks the worst question | Prepare the failure case |
| Regulatory paragraph missing | "We have not looked at that" | Deferred | Claim exceeds authority | Four sentences from the memo |
| Value case as a single number | "It will save 2 million" | Precision theatre | Disbelief | Range and break-even |
| Hostile question answered defensively | Voice rises | Threat response | Lost room | Acknowledge, evidence, boundary |
| False premise accepted | Answering "we are not replacing clinicians" without correcting | Reflex | Premise stands | Correct, then answer |
| Bluffing | Answer without a number | Fear of looking weak | Discovery later | Say what was not measured |
| Logistics failure | No adapter test | Assumption | Late start | Test the room |
| No follow-up | Pitch ends at the door | Relief | Interest decays | Note within 48 hours |

The table is ordered by severity rather than frequency. The first four lose the decision; the middle
five lose credibility; the last five lose polish.

The first row is the most common and the most consequential. A pitch with no ask is not a weak pitch;
it is a presentation, and the audience will treat it as one.

The table's mitigations are all preparation. None of them is a technique for thinking quickly under
pressure, because that technique does not exist; what exists is a script, a backup, and a rehearsal.

Two rows are worth memorising. "No ask" loses the decision, and "bluffing" loses the room. Both are
preventable with one sentence prepared in advance.

The rows about the demo are the ones a team can test in advance, and therefore the ones with the
least excuse. A demo that depends on a network is a choice, not an accident.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The audience can make the decision | The ask depends on it | Audience is peers only | Check the attendee list |
| Ten minutes is enforced | The budget depends on it | Chair is flexible | Ask the chair |
| The demo can run offline | Backup plan | Hosted service | Test with the network off |
| The evidence is stable | The pitch quotes it | Late re-run | Freeze before the pitch |
| The value assumptions are plausible | The range depends on it | No local cost data | Ask the finance contact |
| The room has a projector and audio | The demo depends on it | Hybrid or outdoor venue | Test in advance |
| Questions are predictable | Preparation depends on it | Unusual audience | Prepare ten follow-ups |
| Follow-up is possible | The ask depends on it | No contact route | Collect a contact |
| The one-pager reaches the decider | The ask depends on it | Audience delegates | Send it to the named person |

The largest limit is that a pitch is a compression, and compression loses information. The dossier
exists because the pitch cannot carry the evidence, and a team that treats the pitch as the whole
argument will overclaim by omission.

A second limit is that the audience's decision may not be the one the team wants. A pitch for a pilot
can be met with a request for a validation study, and that is a legitimate outcome. The team should
prepare the response: "that is the right next step, and here is what it would take".

A third limit is that rehearsal cannot simulate a hostile room. The first hostile question a team
hears is usually on the day, and the protocol in Section 5 is the only preparation that survives the
surprise.

A fourth limit is that the pitch is a sample of the work, and a bad ten minutes can outweigh a good
twelve weeks. That is unfair and it is the format. The mitigation is preparation, not resentment.

A fifth limit is that some audiences are not deciding anything. A peer audience can critique but
cannot approve, and a pitch that asks a peer audience for a pilot has misread the room. The ask
should match the audience's authority, even when that means asking for a smaller thing.

A sixth limit is that a ten-minute pitch cannot be both an argument and a demonstration of
everything the team knows. The individual defense exists because the pitch cannot test understanding;
a team that tries to prove its knowledge in the pitch usually proves less.

A seventh limit is that the audience's memory is short and selective. The pitch is designed to leave
three things behind, and a team that tries to leave ten leaves none.

## 12. Exercises

### Exercise 1. Write three asks (application, 25 minutes)

Write an ask for three audiences: a clinical director, a payer, and a peer audience.
**Deliverable:** three sentences.
**Criteria:** each names a person or role, a decision, a date, and a reason; none says "support".
**Time:** 25 minutes.

### Exercise 2. Build the time budget (calculation, 25 minutes)

Build a nine-segment budget for your pitch and check that it sums to ten minutes with zero slack.
**Deliverable:** the table and the arithmetic.
**Criteria:** all nine segments present; total exactly ten; the segment to cut named in advance.
**Time:** 25 minutes.

### Exercise 3. Write the evidence sentence (application, 30 minutes)

Write one sentence with the primary metric, the interval, the comparator, the split, and the
operating point.
**Deliverable:** one sentence and the table that supports it.
**Criteria:** all five elements present; the interval is not omitted; the comparator is current
practice.
**Time:** 30 minutes.

### Exercise 4. Script the demo (design, 40 minutes)

Write a ninety-second demo script with five beats, including one failure case.
**Deliverable:** the script and a recording.
**Criteria:** five beats timed; the case is held-out or synthetic; the failure is shown and
explained.
**Time:** 40 minutes.

### Exercise 5. Compute a value case (calculation, 40 minutes)

Compute monthly saving, payback, and break-even for your project, with a range for the effect.
**Deliverable:** the table, the arithmetic, and one sentence for the pitch.
**Criteria:** assumptions named; payback computed for three effects; the break-even stated.
**Time:** 40 minutes.

### Exercise 6. Prepare hostile answers (design, 35 minutes)

Write answers to five hostile questions using the acknowledge, evidence, boundary, next-step
protocol.
**Deliverable:** five answers of under sixty words each.
**Criteria:** each acknowledges the concern; each cites a number or an artefact; none is defensive.
**Time:** 35 minutes.

### Exercise 7. Rehearse to time (application, 30 minutes)

Deliver the pitch twice, timed, with a listener who records overruns and unclear segments.
**Deliverable:** two timings and a cut list.
**Criteria:** second run within ten minutes; the ask delivered word for word; cuts identified.
**Time:** 30 minutes.

### Exercise 8. Critique a recorded pitch (critique, 30 minutes)

Watch a public pitch and score it against the nine segments and the five questions.
**Deliverable:** a scored critique.
**Criteria:** each segment marked present or absent; the strongest and weakest moments named; one
improvement specified.
**Time:** 30 minutes.

**Assessment note.** Exercises 1, 2, and 5 have checkable outputs. Exercises 3, 4, and 7 are assessed
on the artefact and the timing: whether the evidence sentence has all five elements, whether the demo
runs offline, and whether the second run is within ten minutes.

Exercise 7 is the one that separates a draft from a pitch. Most teams overrun by two minutes on the
first run and discover that the segment they would cut is the ask, which is the argument for cutting
in rehearsal rather than on stage.

Exercise 8 is the cheapest improvement available. Watching three pitches with the nine-segment
checklist teaches more about structure than reading about structure, because the missing segment is
obvious in someone else's pitch and invisible in your own.

## 13. Self-check answers

**1. Why write the ask first?**
Because the ask determines what the pitch must establish. A pitch written first and given an ask at
the end usually asks for the wrong thing, or asks for nothing.

**2. What are the five questions?**
What decision changes and who owns it; what current practice is and what it costs; what the evidence
is at what operating point; what happens when it is wrong; whether it can legally reach a patient
and who pays.

**3. Why does the budget have zero slack?**
Because slack is absorbed by the segment that expands, which is usually the demo, and the segment
that gets cut is the ask. Zero slack forces the trade-off to be decided in advance.

**4. Compute the payback for an 18 percent effect with 30,000 implementation and 6,000 running.**
Monthly saving = 420 × 0.18 × 38 = 2,873. Net = 2,873 - 500 = 2,373. Payback = 30,000 / 2,373 =
12.6 months.

**5. Why does the full programme not pay back?**
Because 180,000 divided by a net monthly benefit of 873 is 206 months. The implementation cost is
large relative to the saving at the assumed effect, which is why the ask becomes a pilot.

**6. What is the correct response to a false premise?**
Correct the premise, then answer the real question. Answering the premise as stated accepts it.

**7. Why volunteer a limitation?**
Because the audience assumes one exists, and a team that names it first demonstrates control. The
limitation also sets the boundary of the claim, which is what a careful audience wants.

**8. What should a team do if the demo fails on stage?**
Switch to the recording, say that it is a recording, and continue. Do not debug in public.

**9. What makes a demo credible?**
A held-out or synthetic case, the input and output visible together, a number on screen, and one
failure case shown deliberately.

**10. Rubric for Exercise 5.** A strong value case names every assumption, shows three effects, and
states the break-even. A weak case presents a single payback figure without the effect size or the
running cost.

**11. Rubric for Exercise 6.** A strong hostile answer acknowledges the concern in one clause, gives
a number or an artefact, states the boundary, and offers a next step, in under sixty words. A weak
answer defends the work or changes the subject.

**12. Why is the two-minute version a test of the pitch?**
Because compression removes supporting detail and leaves the argument. If the argument survives, the
ten-minute pitch has a spine; if it collapses into a list of activities, the pitch was a tour.

**13. What should the leave-behind contain?**
The decision, the evidence sentence with its interval, the comparator, the failure and mitigation,
the regulatory line, the value range, and the ask with a contact. It should be readable in ninety
seconds a week later.

## 14. Interview questions

**Q1. "Pitch your capstone in two minutes."**
Strong answer follows the structure: decision, comparator, evidence, failure, ask. Weak answer
starts with the data and the model.

**Q2. "What did you ask for?"**
Strong answer names a person, a decision, and a date. Weak answer says support or funding.

**Q3. "How did you handle a hostile question?"**
Strong answer describes acknowledging, correcting a premise, and answering with evidence. Weak
answer says the panel was hostile.

**Q4. "What happened when the demo failed?"**
Strong answer describes the backup and the disclosure. Weak answer says it did not fail.

**Q5. "How did you decide what to cut from the pitch?"**
Strong answer names the segment and the reason, tied to the ask. Weak answer says the team
prioritised.

**Q6. "What is your value case?"**
Strong answer gives a range, a payback, and a break-even. Weak answer gives one number.

**Q7. "Who pays?"**
Strong answer names the budget holder and the procurement route. Weak answer says the hospital.

**Q8. "Is this regulated?"**
Strong answer gives the regime, the classification, and the binding artefact. Weak answer says it is
research.

**Q9. "What would make you stop?"**
Strong answer gives a numeric criterion and the decision it triggers. Weak answer says if it does
not work.

**Q10. "What do you want from this conversation?"**
Strong answer repeats the ask. Weak answer says feedback.

**Q11. "What would you cut if you had five minutes?"**
Strong answer names the segments and keeps the evidence and the ask. Weak answer says it would speak
faster, which is not a cut.

**Q12. "Who was in the room, and what did they decide?"**
Strong answer names the people and the decision or non-decision. Weak answer describes the audience as
"stakeholders".

A strong answer to this set is always specific about the audience and the ask. A candidate who can
say "the director could approve a pilot, and here is the evidence she would need" is demonstrating
that the pitch was written for a decision rather than for a grade.

## 15. Key terms

- **Adoption** — sustained use by the intended user.
- **Ask** — specific decision or resource requested.
- **Backup recording** — recorded demo used if live fails.
- **Break-even** — effect at which costs equal savings.
- **Bridging** — acknowledging and redirecting to evidence.
- **Budget holder** — person who controls the money.
- **Claim boundary** — what the evidence does and does not support.
- **Comparator** — current practice measured on the same cohort.
- **Deployment** — output reaching a user who acts on it.
- **Evidence sentence** — result with interval, comparator, split, operating point.
- **Failure case** — demonstrated limitation with mitigation.
- **Hostile question** — question designed to expose a weakness.
- **Moderator** — person managing questions.
- **Narrative arc** — sequence from problem to ask.
- **One-pager** — readout left with the audience.
- **Pilot** — bounded deployment to generate evidence.
- **Pitch** — timed argument for a decision.
- **Procurement** — organisational buying process.
- **Regulatory paragraph** — regime, classification, artefact, open question.
- **Segment** — timed part of the pitch.
- **Timekeeper** — person signalling segment boundaries.
- **Value case** — cost, saving, payback, assumptions.
- **Leave-behind** — one-pager left with the audience.
- **Q&A sheet** — prepared answers with numbers and artefacts.
- **Premise** — assumption embedded in a question.
- **Compression** — the two-minute version of the pitch.

The vocabulary is small because the pitch is short. A team that needs a longer glossary to describe
its pitch is probably describing a dossier.

Two terms in the list are frequently confused in rehearsal. The ask is a decision; the leave-behind
is a document. A team that treats the one-pager as the ask has produced a document and no request.

## 16. Sources

All URLs accessed 26 October 2026. The pitch criteria and programme descriptions are cited as
published; no prize, score, or winner is asserted beyond what the source states.

- MIT Hacking Medicine Brazil 2025 guidelines (pitch evaluation criteria).
  https://www.eretz.bio/wp-content/uploads/2025/07/Guidelines-EN-MIT-HACKING-MEDICINE-BRASIL-2025-ENGLISH-2.pdf
- Google Research, MedGemma Impact Challenge winners, March 2026.
  https://blog.google/innovation-and-ai/technology/health/med-gemma-impact-challenge/
- NHS England, Planning and implementing real-world AI evaluations.
  https://www.england.nhs.uk/long-read/planning-and-implementing-real-world-ai-evaluations-lessons-from-the-ai-in-health-and-care-award/
- Royal College of Radiologists, *AI deployment fundamentals*, 2025.
  https://www.rcr.ac.uk/media/r0th2x0k/ai-deployment-fundamentals_public-consultation.pdf
- FDA, *Clinical Decision Support Software*, September 2022.
  https://www.fda.gov/regulatory-information/search-fda-guidance-documents/clinical-decision-support-software
- Federal Trade Commission Act, 15 U.S.C. § 45. https://www.ftc.gov/legal-library/browse/statutes/federal-trade-commission-act
- HIPAA Privacy Rule, 45 CFR Parts 160 and 164. https://www.hhs.gov/hipaa/
- Regulation (EU) 2016/679, General Data Protection Regulation.
  https://eur-lex.europa.eu/eli/reg/2016/679/oj
- Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement. *BMJ* 2024;385:e078378.
  https://doi.org/10.1136/bmj-2023-078378
- Wong A, Otles E, Donnelly JP, et al. External validation of a widely implemented proprietary
  sepsis prediction model. *JAMA Internal Medicine* 2021;181(8):1065-1070.
  https://doi.org/10.1001/jamainternmed.2021.2626

## 17. Links

The structure in this topic is the compression of the dossier in Topic 09 and the rubric in Topic 10.
A team that has completed the self-assessment and read the rubric has the material for the pitch; the
remaining work is selection and rehearsal.

Module map: [`_index.md`](_index.md)
Previous topic: [`10_judging.md`](10_judging.md)
First topic: [`01_capstone-architecture.md`](01_capstone-architecture.md)
Related topic: [`09_deliverables.md`](09_deliverables.md)
Survey file: [`../../course/19_capstone_hackathons.md`](../../course/19_capstone_hackathons.md)
Labs: [`../../labs/`](../../labs/)

This is the last topic in the module. The next module turns the capstone into a portfolio, a
narrative, and interview answers that survive technical and behavioural scrutiny.

The pitch is also the artefact most likely to be reused: the two-minute version becomes an interview
answer, and the ask becomes the first line of a cover letter.

A team that has written the ask, rehearsed the failure case, and computed the break-even has
something more durable than a demo. It has a way of talking about the work that survives the room.

- Module map: [`_index.md`](_index.md)
- Previous topic: [`10_judging.md`](10_judging.md)
- First topic: [`01_capstone-architecture.md`](01_capstone-architecture.md)
- Related topic: [`09_deliverables.md`](09_deliverables.md)
- Survey file: [`../../course/19_capstone_hackathons.md`](../../course/19_capstone_hackathons.md)
- Labs: [`../../labs/`](../../labs/)
