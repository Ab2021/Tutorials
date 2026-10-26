# Interview Types: What Each Stage Is Actually Testing

> Module 20 — Career, Portfolio, and Interviews
> Topic 04 of 10
> Estimated study time: 3 hours
> Prerequisites: 01, 03
> Version 2026.3

## 1. Why this topic exists

A hiring process is a sequence of inferences, and each stage is designed to test a different one.
A recruiter screen tests fit against the requisition and constraints. A hiring manager interview
tests whether you can own the work. A technical screen tests whether your hands move the way
your resume says they do. A case interview tests how you structure an ambiguous problem. A panel
tests whether a group of people with different stakes can tolerate working with you.

Candidates usually prepare content: statistics, structured query language, project stories. That is necessary and
insufficient, because each format has its own rules of evidence. A brilliant answer to a
statistics question delivered as a lecture fails a case interview. A well-structured case answer
with no numbers fails a technical screen. A strong technical candidate who cannot say why this
employer fails the hiring manager screen, which is often the shortest and most decisive stage.

### 1.1 The cost of format blindness

The most common expensive mistake is treating every stage as a technical interview. A candidate
rehearses gradient boosting explanations for a 30-minute hiring manager conversation that is
actually testing motivation, scope fit, and risk. The manager asks "why this role" and receives a
list of technologies. The candidate is rejected for "communication," which is the label used when
the interviewer cannot articulate the real concern.

The second mistake is the reverse: treating a technical screen as a conversation. The format
usually has a fixed rubric and a time box. Talking through context for ten minutes leaves too
little time to produce the artifact, and the score reflects the missing artifact, not the
reasoning.

### 1.2 Why the format matters more in biomedicine

In a general technology process, a strong engineer can often recover from a weak stage because
the signal is portable. In biomedical hiring, the panel usually includes a clinician, a
regulatory or quality representative, or both, and their inference is about safety and judgment,
not only competence. A candidate who cannot explain a model's failure modes to a clinician fails
a stage that a pure coding interview would have passed.

The domain also adds formats that are rare elsewhere: the clinical case, the model review, the
regulatory discussion, and the presentation to a mixed audience. Each has a different success
criterion.

### 1.3 What this topic produces

A map of the stages you are likely to encounter, the inference each stage tests, the artifact you
should prepare for each, and a practice plan that allocates your limited hours by stage weight
rather than by anxiety. Section 6 walks one candidate through a five-stage loop with the actual
preparation hours and the two diagnostics that changed the plan.

### 1.4 What this topic does not cover

It does not teach the technical content of SQL, statistics, or machine learning; those are Topics
05 and the methods modules. It does not teach the clinical case structure in depth (Topic 06),
system design (Topic 07), or behavioural storytelling (Topic 08). It covers the formats, the
sequencing, the preparation economics, and the failure modes that are common across all of them.

### 1.5 How long a process takes, and why it matters

A process that runs six weeks is not the same opportunity as one that runs six days. Long
processes cost you other offers, and the cost compounds when several employers move at different
speeds. The practical rules are: ask for the timeline at the first call, tell the recruiter if you
have a competing deadline (without inventing one), and decide in advance how long you will wait
for a preferred employer before accepting another offer.

Process length also signals how the employer makes decisions. A loop with five stages and a
scheduled debrief is usually a process with an owner. A process that adds stages as it goes is
usually one where no single person can decide, which is worth knowing before you accept.

## 2. Learning outcomes

| # | Outcome | Evidence |
|---:|---|---|
| 1 | Map a hiring process into stages and name the inference each stage tests | Exercise 1; self-check 1 |
| 2 | Prepare a stage-appropriate artifact rather than a generic answer | Exercise 2; self-check 2 |
| 3 | Allocate a fixed preparation budget across stages by weight and weakness | Exercise 3; self-check 3 |
| 4 | Run a 30-second, 2-minute, and 10-minute version of the same project story | Exercise 4; self-check 4 |
| 5 | Prepare and deliver a 10-minute mixed-audience presentation with a decision focus | Exercise 5; self-check 5 |
| 6 | Ask questions that reveal scope, decision rights, and success criteria | Exercise 6; self-check 6 |
| 7 | Recognise and handle a format mismatch during the interview | Exercise 7; self-check 7 |
| 8 | Debrief each stage and convert the result into a change in preparation | Exercise 8; interview question 10 |

The outcomes are assessed in a live or recorded rehearsal, not by self-report. For outcome 3,
the assessor is the allocation table and the justification for each cap. For outcome 5, the
assessor is a non-specialist who watches the presentation once and restates the decision. For
outcome 7, the assessor is a mock interviewer who introduces a format mismatch without warning.

Outcomes 2 and 3 are the practical core. Outcome 7 is the recovery skill that separates
candidates who salvage a bad stage from those who compound it.

## 3. The core mental model

Every interview stage is a measurement instrument with a bandwidth, a fidelity, and a failure
mode. The employer chooses instruments that are cheap, legally defensible, and predictive of the
work. The candidate's job is to produce the signal the instrument is built to detect.

```text
  STAGE                 INFERENCE TESTED                 BANDWIDTH
  -----------------------------------------------------------------------
  Recruiter screen      Match, constraints, motivation   Low, fast
  Hiring manager        Ownership, scope, judgment       Medium
  Technical screen      Fluency under time pressure      High, narrow
  Case                  Structuring ambiguity            High, verbal
  System design         Architecture and trade-offs      High, verbal
  Panel                 Cross-stakeholder credibility    Medium, broad
  Presentation          Communication to mixed audience  High, prepared
  Take-home             Judgment without supervision     High, slow
  -----------------------------------------------------------------------
  Offer                 Negotiation and risk             Low
```

Bandwidth is how much information the stage can carry; fidelity is how much of it predicts the
job. The technical screen has high bandwidth and narrow fidelity: it predicts coding fluency, not
clinical judgment. The panel has lower bandwidth but broader fidelity: it predicts whether
multiple stakeholders will trust you.

### 3.1 The inference ladder

Each stage asks a version of the same question at a different altitude.

| Stage | The question behind the question |
|---|---|
| Recruiter | Will the hiring manager thank me for forwarding this, or blame me? |
| Hiring manager | Can this person own the decision in 90 days without my supervision? |
| Technical | Can they produce correct work under constraints, and do they know when they are wrong? |
| Case | Do they impose structure on ambiguity, or do they wait for instructions? |
| System design | Can they reason about trade-offs, failure, and operations? |
| Panel | Will the clinician, the engineer, and the manager each be willing to work with them? |
| Presentation | Can they make a decision legible to people who do not share their expertise? |
| Take-home | What do they do when nobody is watching and time is short? |

The last row is why take-homes are controversial and why they are informative. They measure
prioritisation and taste, which are hard to fake in a live interview.

### 3.2 Why structure predicts

The research on selection interviews is unusually clear for a personnel practice. Campion, Palmer
and Campion (1997) identified 15 components of interview structure, including job analysis, the
same questions for all candidates, anchored rating scales, detailed notes, multiple interviewers,
and statistical rather than clinical combination of ratings. Their review reported corrected
validities roughly in the range of .35 to .62 for structured interviews against .14 to .33 for
unstructured ones.

Sackett, Zhang, Berry and Lievens (2022) revisited the meta-analytic estimates after correcting
for over-correction of range restriction. In their revised estimates the structured interview was
the top-ranked procedure, with an operational validity of .42, while the unstructured interview
fell to .19. The absolute numbers moved; the ordering did not.

For a candidate, the implication is not "prepare for structured interviews." It is that the
employer's questions are probably job-analysed, scored, and compared across candidates. The
interviewer is looking for evidence against a defined criterion, and a vivid but off-criterion
story scores zero however impressive it is.

### 3.3 The loop as a portfolio of evidence

A full loop usually collects evidence on four dimensions: technical, domain, judgment, and
collaboration. Different stages load on different dimensions.

| Dimension | Stages that carry it | What good evidence looks like |
|---|---|---|
| Technical | Screen, take-home, technical panel | Correct work, stated assumptions, awareness of failure |
| Domain | Case, clinical panel, hiring manager | Correct vocabulary, workflow awareness, regulatory literacy |
| Judgment | Case, system design, behavioural | Explicit trade-offs, stop criteria, escalation |
| Collaboration | Panel, behavioural, presentation | Credit sharing, disagreement handled without deference or aggression |

A candidate who is strong on technical and weak on collaboration may pass the screen and fail
the panel. Knowing which dimension each stage carries lets you prepare the weakest dimension
before the stage that tests it, rather than after.

### 3.4 The preparation economics

Preparation time is scarce. Allocate it by expected loss, not by interest. Expected loss is the
product of the stage's weight in the decision, the probability you fail it, and the cost of
failing. A stage that is 10 percent of the decision and that you will pass anyway deserves
almost no time, even if it is the most enjoyable to practise.

```text
priority(stage) = decision_weight x failure_probability x cost_of_failure
```

Section 7 implements this and produces a schedule. Section 6 shows what changes when one
probability is revised after a diagnostic.

### 3.5 Why feedback is scarce

Most candidates receive no useful feedback, and the reason is structural rather than personal.
Employers avoid specific feedback because it creates legal risk, because the decision is often a
comparison rather than a checklist, and because the interviewer who gives it may not have been
the decision-maker. A candidate who expects specific feedback will wait for something that is not
coming.

The substitute is your own instrumentation: a debrief note after every stage, a mock interview
before every loop, and a record of which questions produced which reactions. Over five processes
this record is more reliable than the feedback most employers provide.

> **Note:** The weights and probabilities in this model are your estimates. The model is a
> allocation device, not a prediction. Its value is that it forces you to write down why a stage
> deserves hours.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Screen | A short interview to check basic fit and constraints | Treated as a formality | Minutes |
| Hiring manager interview | Conversation with the person who owns the role | Confused with a peer interview | Stage |
| Technical screen | Timed exercise in coding, SQL, or statistics | Assumed to test domain judgment | Stage |
| Case interview | Open-ended problem structured aloud | Confused with a clinical case | Stage |
| Clinical case | A case built around a patient or workflow decision | Treated as a medical knowledge test | Stage |
| System design | Design of a system under constraints | Confused with architecture trivia | Stage |
| Panel | Several interviewers in one session | Confused with a sequence of one-to-ones | Stage |
| Loop | The full sequence of stages for one role | Used for a single interview | Sequence |
| Take-home | An unsupervised exercise completed on your own time | Assumed to be optional | Hours |
| Presentation | A prepared talk to a mixed audience | Confused with a slide review | Minutes |
| Behavioural question | A question about past behaviour | Confused with a hypothetical | Question type |
| Situational question | A hypothetical scenario question | Confused with behavioural | Question type |
| Structured interview | An interview with fixed questions and anchored scoring | Used to mean "formal" | Design |
| Scorecard | The interviewer's structured rating form | Confused with the debrief | Form |
| Debrief | The committee meeting that combines ratings | Confused with feedback to the candidate | Meeting |
| Calibration | Comparing ratings across interviewers | Confused with model calibration | Process |
| Bar raiser | An interviewer with veto power over hiring standards | Assumed to be a technical expert | Role |
| Shadow | An interviewer who observes without scoring | Confused with a peer interviewer | Role |
| Recruiter | The person managing the process and the requisition | Assumed to be the decision-maker | Role |
| Coordinator | The person scheduling and managing logistics | Confused with the recruiter | Role |
| Accommodation | An adjustment to the process for a disability or need | Assumed to require disclosure of diagnosis | Request |
| Adverse impact | A substantially different selection rate between groups | Confused with disparate treatment | Legal concept |
| Four-fifths rule | A selection rate below 80 percent of the highest group's rate as evidence of adverse impact | Treated as a legal safe harbour | Ratio |
| Job analysis | Systematic description of the work used to build the interview | Assumed to happen in every process | Process |
| Anchored rating scale | A rating scale with behavioural anchors per point | Confused with a numeric scale | Instrument |
| Signal | The evidence a stage is designed to detect | Confused with impression | Construct |
| Bandwidth | How much information a stage can carry | Confused with difficulty | Construct |
| Fidelity | How well a stage predicts job performance | Confused with realism | Construct |

| Process owner | The person accountable for the hiring decision and the loop | Confused with the recruiter | Role |
| Debrief bias | The tendency of a group to converge on the first strong opinion | Confused with calibration | Bias |
| Work sample | A task that mirrors the job, used as a selection procedure | Confused with a take-home | Instrument |
| Job talk | An academic seminar used as an interview | Treated like a panel presentation | Format |
| Competency question | A question mapped to a defined behavioural competency | Confused with a random behavioural question | Question type |
| Panel pack | The documents a panel receives in advance | Assumed to include the candidate's materials | Document |
| Silence | A pause after a question, used to elicit more detail | Filled by candidates with caveats | Technique |

## 5. The mechanics, step by step

The procedure turns an unknown process into a preparation plan in about two hours, then executes
it over the weeks before the loop. It assumes you already have the evidence from Topics 01 to 03.

### Step 1. Decode the process before preparing anything

**Input:** a recruiter call or the first email.
**Output:** a written stage map with interviewer roles, formats, durations, and order.
**Rationale:** candidates prepare for the process they imagine. The actual process has a shape,
and the shape tells you what is weighted.

Ask the recruiter, in one message:

1. How many stages are there, and in what order?
2. Who interviews at each stage, by role (not name)?
3. What is the format at each stage: conversation, exercise, case, presentation, panel?
4. How long is each stage, and is anything timed?
5. Is there a take-home or a presentation, and what are the constraints?
6. What is the expected timeline, and are there fixed dates?

Recruiters answer these questions readily because they reduce scheduling failure. If a recruiter
declines to say who interviews, that is itself information about the process's maturity.

### Step 2. Map each stage to the dimension it tests and its weight

**Input:** the stage map.
**Output:** a table with stage, dimension, estimated weight, and your current confidence.
**Rationale:** the table converts anxiety into allocation. It also reveals stages that carry more
weight than their duration suggests: a 30-minute hiring manager conversation is often decisive.

| Stage | Primary dimension | Typical weight | Your confidence (1–5) |
|---|---|---:|---:|
| Recruiter screen | Fit and constraints | 10% | 4 |
| Hiring manager | Judgment and ownership | 30% | 3 |
| Technical screen | Technical | 25% | 2 |
| Case | Judgment and domain | 20% | 3 |
| Panel | Collaboration and domain | 15% | 3 |

The weights are your estimate and should sum to 100 percent. Ask the recruiter which stage most
candidates find hardest; the answer is usually honest and shifts the weights.

Two adjustments are worth making before you finalise the weights. First, if the hiring manager is
also the person who would be your manager, increase the weight of that stage: the manager's
judgment usually dominates the debrief. Second, if the process includes a presentation to a
mixed audience, do not discount it because it is short. Presentations are memorable and are
often the stage where a marginal candidate is decided.

Write the confidence rating before you do any preparation, and do not revise it afterwards.
The gap between the initial rating and the outcome is the most useful calibration data you will
collect in the whole process.

### Step 3. Build the evidence bank

**Input:** the accountability statements from Topic 01.
**Output:** five stories, each with a decision, an action, a result, and a reflection.
**Rationale:** most behavioural and case questions are versions of the same five stories. Having
them prepared in a fixed structure prevents rambling and makes the same evidence reusable across
formats.

The five stories:

1. A project that succeeded and what you owned.
2. A project that failed or was stopped, and what you changed afterwards.
3. A disagreement with a clinician, a client, or a manager, and how it resolved.
4. A time you simplified scope or said no.
5. A time you found a problem in data, a model, or a process that others had missed.

Each story is one page in a private document, with the numbers and the names of the people
involved (roles, not identities). The reflection is the part interviewers remember: what you
would do differently, stated specifically.

### Step 4. Compress each story to three lengths

**Input:** the five stories.
**Output:** 30-second, 2-minute, and 10-minute versions of the strongest two.
**Rationale:** different stages have different time budgets. A recruiter screen needs 30 seconds;
a hiring manager needs 2 minutes; a presentation needs 10. Candidates who have only the long
version overrun every stage.

| Length | Use | Structure |
|---|---|---|
| 30 seconds | Recruiter screen, "tell me about yourself" | Role, domain, one proof point, target |
| 2 minutes | Hiring manager, behavioural | Situation, decision, action, result, reflection |
| 10 minutes | Presentation, deep technical | Context, decision, method, evidence, limits, next step |

The 30-second version is not a summary of the 2-minute version. It is a different artifact: it
answers "who are you and why this role" rather than "what happened in this project."

Rehearse the 30-second version until it is boring to you. It is the first thing said in almost
every stage, it sets the frame for everything after it, and it is the only part of the interview
you can fully script without sounding scripted. The version should contain one specific number
and no adjectives about your personality.

### Step 5. Prepare the stage-specific artifact

**Input:** the stage map and your confidence ratings.
**Output:** one artifact per stage that is weaker than 4.
**Rationale:** each format rewards a different artifact. Preparing a deck for a coding screen is
wasted effort; preparing flashcards for a presentation is also wasted.

| Stage | Artifact to prepare | Practice unit |
|---|---|---|
| Recruiter screen | 30-second story, constraints statement, three questions | 3 mock calls |
| Hiring manager | 2-minute stories, scope questions, "why this role" answer | 2 mock interviews |
| Technical screen | Timed exercises in the actual environment | 6 timed reps |
| Case | A written structure template and 3 practised cases | 3 cases aloud |
| System design | A blank-page diagram template and 2 designs | 2 designs aloud |
| Panel | One artifact per panellist's stake | 1 mock panel |
| Presentation | Slides with a decision on slide 1 | 3 timed runs |
| Take-home | A scope note, a baseline, and a README template | 1 timed run |

The technical screen deserves special mention: practise in the environment the employer uses
(shared editor, notebook, whiteboard), not in your own IDE with autocomplete. Fluency in your
environment does not transfer automatically.

### Step 6. Write three questions per stage

**Input:** the stage map.
**Output:** a question list, ranked, with the reason each question matters.
**Rationale:** questions are scored, explicitly or implicitly, as evidence of judgment. A
candidate who asks about success criteria, decision rights, and failure modes is inferred to be
senior. A candidate who asks only about benefits is inferred to be passive.

Question categories:

1. **Scope and decision rights.** "What decisions does this role own without escalation?"
2. **Success criteria.** "What would a strong first six months look like, and who judges it?"
3. **Failure and risk.** "What has gone wrong with a model or analysis here, and what changed?"
4. **Workflow and users.** "Who uses the output, and what do they do differently because of it?"
5. **Team and process.** "How are analyses reviewed, and what does the review actually check?"

Ask two or three, not ten. Save the compensation and logistics questions for the recruiter.

### Step 7. Rehearse under time with a stranger

**Input:** the artifacts.
**Output:** a recording or notes showing where you overran, hedged, or lost structure.
**Rationale:** self-review cannot detect pacing. A stranger with a timer can.

Protocol:

1. Give the rehearsal partner the stage format and the time limit, not the content.
2. Ask them to interrupt with one clarifying question, as a real interviewer would.
3. Record the session if permitted, and note the timestamp where the answer lost structure.
4. Rehearse the first 60 seconds separately; the opening sets the frame and is the most
   rehearsable part.

### Step 8. Manage logistics as part of the interview

**Input:** the schedule.
**Output:** a logistics checklist completed 24 hours before each stage.
**Rationale:** avoidable failures are common and costly: a broken link, a time-zone error, a
background noise problem, a missing file.

Checklist: link tested, backup phone number, quiet room, camera height, lighting, water, notes
visible but not read from, portfolio link open, time zone confirmed in writing, and a copy of the
job description printed.

For in-person loops: arrival time with 20 minutes of buffer, printed copies of the resume,
a notebook, and the names and roles of the interviewers.

### Step 9. Run the stage with an explicit structure

**Input:** the prepared artifacts.
**Output:** answers that are structured, time-aware, and checked.
**Rationale:** interviewers score structure because it predicts how you will behave in a meeting
with a clinician who has ten minutes.

A general pattern for any answer:

1. **Restate the question in one sentence.** Confirms understanding and buys two seconds.
2. **State the frame.** "I will answer in three parts: the decision, the evidence, the limits."
3. **Deliver the parts.** Two minutes maximum for behavioural, five for case.
4. **Check in.** "Does that answer the question, or should I go deeper on the validation?"
5. **Stop.** Do not fill silence with caveats.

For technical stages, the pattern changes: state assumptions, write the approach, produce a
working artifact, then test and explain. Verbal framing is worth 30 seconds, not five minutes.

The first 60 seconds differ by format:

| Format | First 60 seconds |
|---|---|
| Recruiter screen | Name, current role, target role, one constraint (location or authorisation) |
| Hiring manager | The 30-second story, then a question about the decision the role owns |
| Technical screen | Restate the task, state assumptions, describe the approach in two sentences, start |
| Case | Clarify the decision and the actor, then state the structure you will use |
| System design | Clarify users, scale, and constraints, then draw the first box |
| Panel | Thank the panel, connect your work to the role in two sentences, invite the first question |
| Presentation | The decision, in one sentence, before any context |

### Step 10. Debrief within 24 hours

**Input:** the stage experience.
**Output:** a one-page note with what was asked, what you answered, what you would change, and one
question you fumbled.
**Rationale:** memory decays fast, and the same questions recur across employers. A debrief log
converts each interview into preparation for the next.

Record: stage, interviewer role, three questions, your answer in one line, the reaction you
observed, and the question you would answer differently. After five processes, the log becomes a
question bank calibrated to the market.

### Step 11. Close the loop

**Input:** the outcome or the silence.
**Output:** a follow-up message and a decision about timing.
**Rationale:** processes stall for logistical reasons more often than for judgment reasons. A
short, specific follow-up restarts them without appearing desperate.

Send a two-paragraph thank-you within 24 hours: one specific thing you learned, one sentence
connecting your evidence to it. If there is no response after the stated timeline plus one week,
send one short status query to the recruiter. After that, stop.

### Step 12. Prepare for the questions you cannot predict

**Input:** the debrief log and the role's decision.
**Output:** three "unknown" answers that generalise across questions.
**Rationale:** every process contains a question you did not anticipate. The candidates who
handle it well do not have a better question bank; they have a reliable way to buy time and
structure an answer.

Three general-purpose answers, each 60 to 90 seconds:

1. **The boundary answer.** "I have not done that exact thing. The closest I have come is X,
   where the same constraint applied. Here is how I would approach Y, and what I would check
   first." This is honest and demonstrates transfer, which is what the interviewer is testing.
2. **The trade-off answer.** "There are two reasonable approaches. The first optimises for A and
   costs B. The second does the reverse. I would choose based on C, and here is why." This works
   for design, methods, and prioritisation questions.
3. **The failure answer.** "The thing that would make me stop is X. Before that, I would test Y
   and Z." This works for cases, system design, and any question about a model or a plan.

The phrase "that is a good question, let me think for a second" is acceptable and buys three
seconds. Silence while thinking reads as composure; filling the silence with a guess reads as
risk.

### Branch points and edge cases

**A clinician on the panel.** Expect questions about workflow, harm, and explainability. Prepare
one artifact that translates a model into a clinical action and one that states its failure
modes. Do not use statistical shorthand without translating it. If the clinician asks "what would
you do with a patient like this," answer in clinical terms first and statistical terms second.

**A panel with four or more people.** Panels test breadth, not depth. Prepare one artifact per
stake: the clinician's workflow question, the engineer's reliability question, the manager's
value question, and the recruiter's constraint question. Address the questioner, then include the
room in the answer; a panel scores how you handle a group, not only the content.

**A take-home is added late.** Accept only with a stated time cap. Ask how long the team expects
it to take and whether the scope can be reduced. If the request exceeds eight hours, propose a
scoped version and state what you will omit. See the decision table below.

**A presentation is required.** Put the decision on slide one, the evidence on slide two, and the
limitations on the second-to-last slide. Rehearse to 80 percent of the time limit. The most
common failure is spending the first half on background.

**Virtual interviews.** Assume the technology will fail once. Have a phone dial-in, a hotspot, and
a downloaded copy of any file. Look at the camera for the opening and closing, and at the screen
for the content.

**A work authorisation question arrives early.** Answer factually in one sentence, then redirect
to the work. "I would need sponsorship for a change of employer; I have six years of experience
and can start within eight weeks of an offer."

**You are over-levelled for the role.** Do not argue about level in the interview. Ask about the
decision rights and success criteria, and let the answers reveal whether the scope matches the
title. Negotiation is Topic 09.

**You are returning after a gap.** Prepare one sentence on the gap and two on current evidence.
Do not raise it before it is asked, and do not apologise when it is.

**You need an accommodation.** Request it from the recruiter as early as possible, in terms of the
adjustment rather than the diagnosis: extra time, a different format, a quiet room, captioning.
Most employers have a process; using it is not a disadvantage.

### Decision table: the take-home request

| Signal | Accept | Negotiate scope | Decline |
|---|---|---|---|
| Time stated | 4 hours or fewer | 5 to 8 hours | More than 8 hours, or unstated |
| Paid | Not required for short exercises | Ask if over 4 hours | Expected for a production task |
| Data | Public or synthetic | Provided under NDA with clear limits | Real patient data with no agreement |
| Evaluation | Rubric or review mentioned | Unclear but bounded | "We will see what you come up with" for an unbounded task |
| Your situation | Time available this week | Time available next week | Two other loops in the same week |

Declining a take-home is acceptable and should be done politely with an alternative: "I can share
a portfolio artifact that covers the same skills, or do a 90-minute live exercise." Some employers
will accept; some will not, and that is information about how they work.

### Decision table: when the format is not what you prepared for

| Situation | Clarify | Adapt | Defer |
|---|---|---|---|
| Interviewer asks a case in a technical slot | Ask whether they want a design or an implementation | Give a 2-minute structure, then implement the core | Not applicable |
| Panel asks a question outside your domain | Confirm the intended depth | Answer with the boundary and a related example | Offer to follow up in writing |
| Take-home scope is larger than described | Ask what "done" looks like | Deliver the core plus a note on what you would do next | Decline if unbounded |
| Interviewer is hostile or dismissive | One clarifying question | Stay factual, shorten answers | End politely and report to the recruiter |

## 6. Worked example

Elena is 34, a senior analyst at a regional health plan, applying for a clinical data scientist
role at a 12-hospital health system. The process has five stages over six weeks. She has 40 hours
of preparation time, spread across evenings and two weekends.

### 6.1 The stage map

| Stage | Week | Interviewer | Format | Duration | Weight (her estimate) |
|---|---:|---|---|---:|---:|
| 1 | 1 | Recruiter | Call | 30 min | 10% |
| 2 | 2 | Hiring manager (director of analytics) | Conversation | 45 min | 30% |
| 3 | 3 | Two data scientists | Technical, shared notebook | 60 min | 25% |
| 4 | 4 | Clinician and product manager | Clinical case | 45 min | 20% |
| 5 | 6 | Panel of four | Presentation plus questions | 60 min | 15% |

Her initial preparation plan: 30 hours on technical practice, 6 hours on the case, 4 hours on the
presentation. That is 40 hours, and it is wrong, because it allocates by anxiety rather than by
weight. The technical stage is 25 percent of the decision and already her strongest dimension;
the hiring manager and panel stages together carry 45 percent and are her weakest.

### 6.2 Diagnostic 1: the "why this role" answer is generic

In a mock hiring manager interview, she answered "why this role" with a paragraph about wanting
to work in healthcare and liking the mission. The mock interviewer's note: "I learned nothing
about why this system, this role, or this decision. I would worry she would leave for the next
mission."

The diagnostic is specific: her answer had no reference to the employer's actual work. She had
read the job description but not the health system's public quality reports, its value-based
contracts, or the specific problem the role was created to solve.

The fix took 4 hours: two public reports, one press release about the analytics team, and one
conversation with a contact who had worked there. Her revised answer named the system's
readmission penalty exposure, the analytics team's shift from reporting to prediction, and the
specific decision the role would own (which patients enter the transitional care programme). The
mock interviewer's note changed to "clear, specific, credible."

The revised answer, in full, was about 90 seconds:

> "The system has a readmission penalty exposure that has grown for three years, and the
> analytics team has been reporting on it rather than predicting it. This role is the first one
> chartered to own a prediction that the transitional care team acts on. My current work is
> claims-based risk stratification for a health plan, which is the same decision with a different
> data source. I want to work where the output changes a clinical workflow rather than a
> dashboard, and where the evidence has to survive a clinical review."

Note what it does: it names the employer's problem, the role's purpose, her transferable
evidence, and her reason for the move. It contains no adjectives about the employer's culture.

### 6.3 Diagnostic 2: the presentation would have failed on time and framing

Her first presentation draft was 18 slides: 10 on background and methods, 3 on results, 1 on
limitations, and 4 on "about me." In a timed run, she reached the results at minute 9 of a
12-minute slot.

Two problems. The time budget put the decision after the audience had stopped listening, and the
framing was a methods talk rather than a decision talk. The panel included a clinician and a
product manager, neither of whom would sit through ten minutes of feature engineering.

The rebuild:

| Slide | Content | Time |
|---:|---|---:|
| 1 | The decision: which patients enter the transitional care programme | 1 min |
| 2 | Why the current rule is insufficient, with one number | 1 min |
| 3 | What I built and how it was validated | 2 min |
| 4 | Calibration and decision curve at the programme's capacity | 2 min |
| 5 | Subgroup performance and the two groups where it is weakest | 2 min |
| 6 | Failure modes and what I would monitor | 1 min |
| 7 | What I would do in the first 90 days | 1 min |

Seven slides, 10 minutes, decision first. The limitation slide moved to position 6, where it
functions as judgment rather than as a disclaimer.

### 6.4 The reallocation

| Activity | Original hours | Revised hours | Reason |
|---|---:|---:|---|
| Technical practice | 30 | 10 | Strongest dimension; six timed reps are enough |
| Hiring manager preparation | 2 | 8 | 30% weight, weakest evidence on motivation and scope |
| Clinical case | 6 | 8 | 20% weight; needs a clinician to practise with |
| Presentation | 2 | 10 | 15% weight, but a visible failure point |
| Recruiter screen | 0 | 1 | 10% weight; cheap to prepare |
| Debrief and buffer | 0 | 3 | Converts each stage into the next |
| **Total** | **40** | **40** | |

The technical hours fell because six timed reps in a shared notebook, under a timer, with someone
watching, produced more improvement than twenty unstructured exercises. The presentation hours
rose because the risk was not knowledge; it was delivery and framing.

### 6.5 What happened at each stage

| Stage | What was asked | What worked | What she changed afterwards |
|---|---|---|---|
| Recruiter | Timeline, work authorisation, salary expectations | Answered the range question with a band and a question | Nothing |
| Hiring manager | "Why this role," a failure story, scope of her current work | The specific readmission answer and the failure story | Added a question about decision rights |
| Technical | SQL window function, cohort definition, a calibration question | Stated assumptions before coding; tested the edge case | Practise the calibration explanation aloud |
| Case | "How would you reduce avoidable readmissions in this population?" | Structured the answer around decision, cohort, intervention, measurement | Spent too long on cohort; cut to 2 minutes next time |
| Panel | Presentation plus questions from a clinician and a product manager | Decision-first slides; the subgroup slide drew the best question | Prepare one slide on cost and one on workflow |

She received an offer after week 7. The two diagnostics did not guarantee the offer; they removed
two failure modes that would have ended the process before the panel.

### 6.6 Vary one assumption

Change one input: the health system adds a take-home between stages 3 and 4, with a stated
expectation of "about 6 hours."

The plan changes in four ways:

1. **Eight hours move from presentation practice to the take-home.** The presentation is now
   built from the take-home's results, so the work compounds rather than competes.
2. **The technical stage becomes a scope negotiation.** She tells the recruiter that the take-home
   covers the same skills and asks whether the live technical can be shortened. The employer
   agrees to 30 minutes.
3. **The case preparation drops to 4 hours**, because the take-home forces her to articulate the
   cohort and measurement decisions that the case would otherwise test.
4. **The timeline extends by one week**, and she declines a second loop scheduled in the same
   week rather than submitting a rushed take-home.

The general lesson: a new stage is not added to the plan; it displaces something. Candidates who
add stages without removing hours produce a plan that cannot be executed and a take-home that is
late.

### 6.7 What she would do differently with 20 hours

If the process had started with only 20 hours instead of 40, the allocation would have been:

| Activity | Hours | Rationale |
|---|---:|---|
| Hiring manager preparation | 6 | Highest weight and weakest evidence |
| Presentation | 6 | Visible failure point, compounding with the panel |
| Technical | 4 | Maintenance only; six reps reduced to three |
| Case | 2 | One practised case with a clinician |
| Debrief and buffer | 2 | Still non-zero |
| **Total** | **20** | |

The technical stage would have been the one to under-prepare, because it was her strongest
dimension and because the cost of a mediocre technical answer is lower than the cost of a
generic motivation answer at the stage that carries 30 percent. This is uncomfortable for
candidates who equate preparation with technical practice. The allocation follows expected loss,
not comfort.

### 6.8 The debrief note she wrote after the case stage

> **Stage:** clinical case, week 4. **Interviewers:** a hospitalist and a product manager.
> **Question:** "How would you reduce avoidable readmissions in this population?"
>
> **What I did:** clarified the decision (which patients enter the transitional care programme),
> defined the cohort, proposed a risk model plus a capacity constraint, and described the
> measurement plan. I spent about 6 of the 45 minutes on cohort definition.
>
> **What worked:** the capacity constraint. The product manager asked how many slots the team
> had, and I had already framed the threshold around 60 slots per month. That changed the
> conversation from model performance to operations.
>
> **What I would change:** the cohort discussion was too long, and I did not ask who currently
> owns the readmission metric. The hospitalist raised it at the end, and I had no answer. Next
> time, ask about ownership in the first five minutes.
>
> **Question to prepare:** "Who is accountable for the outcome today, and what do they measure?"

The note is four paragraphs, takes five minutes, and is the reason her panel answers were
tighter. It is also the artifact that turns a single interview into a reusable question bank.

### 6.9 The stage that almost went wrong

At the technical screen, Elena was asked to compute a 30-day readmission flag in SQL. She wrote
the query correctly in eight minutes and then spent six minutes explaining the cohort definition
without running the query. The interviewer interrupted: "Can you run it and show me the row
count?" She had not tested the query. The count was wrong because the date arithmetic used
calendar days rather than the health plan's 30-day billing window.

She caught it on the run, corrected it, and said what she had assumed. The interviewer's note
recorded both the error and the recovery. The lesson she wrote in her debrief was not "learn SQL";
it was "run the query before explaining it." That note changed her practice for every subsequent
technical stage: produce, test, then explain.

## 7. Code, query, or calculation

The scheduler implements the expected-loss allocation from Section 3.4. It uses only the standard
library, takes the stage table as data, and prints both the priority scores and the resulting
allocation. The second table shows what happens when one failure probability is revised, which is
the operation you perform after each diagnostic.

Assumptions:

- Decision weights, failure probabilities, and costs are your estimates. The model is an
  allocation device, not a forecast.
- A floor of one hour is applied to every stage so that no stage is left unprepared. The
  proportional allocation is then rescaled so the total is exact.
- No cap is applied. Section 6 explains why the candidate applies caps manually: a 45-minute
  conversation has diminishing returns after a handful of rehearsals.

```python
"""Interview practice scheduler, standard library only.

Run: python interview_schedule.py
Allocates a fixed number of preparation hours across interview stages by
expected loss: decision weight x failure probability x cost of failure.
A floor gives every stage at least one hour. No cap is applied by the model;
Section 7 explains why the candidate applies caps manually.
"""

STAGES = [
    # name, decision weight, failure probability, cost of failure (1-5)
    ("Recruiter screen", 0.10, 0.10, 2),
    ("Hiring manager", 0.30, 0.70, 5),
    ("Technical screen", 0.25, 0.30, 3),
    ("Clinical case", 0.20, 0.50, 4),
    ("Panel presentation", 0.15, 0.80, 5),
]

TOTAL_HOURS = 40.0
FLOOR = 1.0


def priority(stage):
    _, weight, fail, cost = stage
    return weight * fail * cost


def allocate(total, stages, floor=FLOOR):
    scores = {s[0]: priority(s) for s in stages}
    names = [s[0] for s in stages]
    weight_sum = sum(scores.values())
    hours = {n: max(floor, total * scores[n] / weight_sum) for n in names}
    # Rescale the stages above the floor so the total is exact.
    above = [n for n in names if hours[n] > floor]
    fixed_total = floor * (len(names) - len(above))
    scale = (total - fixed_total) / sum(hours[n] for n in above)
    for n in above:
        hours[n] *= scale
    return hours


def main():
    print("PRIORITY SCORES")
    print(f"  {'stage':20s} {'weight':>6s} {'p_fail':>6s} {'cost':>5s} {'priority':>8s}")
    for name, weight, fail, cost in STAGES:
        print(f"  {name:20s} {weight:6.2f} {fail:6.2f} {cost:5d} {priority((name, weight, fail, cost)):8.3f}")
    print(f"  total priority: {sum(priority(s) for s in STAGES):.3f}")

    hours = allocate(TOTAL_HOURS, STAGES)
    print()
    print(f"ALLOCATION OF {TOTAL_HOURS:.0f} HOURS (proportional, floor 1.0)")
    print(f"  {'stage':20s} {'hours':>6s} {'share':>6s}")
    for name, _, _, _ in STAGES:
        print(f"  {name:20s} {hours[name]:6.1f} {hours[name] / TOTAL_HOURS:6.1%}")
    print(f"  {'total':20s} {sum(hours.values()):6.1f}")

    revised = [
        ("Recruiter screen", 0.10, 0.10, 2),
        ("Hiring manager", 0.30, 0.40, 5),
        ("Technical screen", 0.25, 0.30, 3),
        ("Clinical case", 0.20, 0.50, 4),
        ("Panel presentation", 0.15, 0.80, 5),
    ]
    hours2 = allocate(TOTAL_HOURS, revised)
    print()
    print("EFFECT OF LOWERING HIRING MANAGER FAILURE PROBABILITY 0.70 -> 0.40")
    for name, _, _, _ in STAGES:
        print(f"  {name:20s} {hours[name]:5.1f} -> {hours2[name]:5.1f} ({hours2[name] - hours[name]:+.1f})")


if __name__ == "__main__":
    main()
```

Expected output, confirmed by running the script:

```text
PRIORITY SCORES
  stage                weight p_fail  cost priority
  Recruiter screen       0.10   0.10     2    0.020
  Hiring manager         0.30   0.70     5    1.050
  Technical screen       0.25   0.30     3    0.225
  Clinical case          0.20   0.50     4    0.400
  Panel presentation     0.15   0.80     5    0.600
  total priority: 2.295

ALLOCATION OF 40 HOURS (proportional, floor 1.0)
  stage                 hours  share
  Recruiter screen        1.0   2.5%
  Hiring manager         18.0  45.0%
  Technical screen        3.9   9.6%
  Clinical case           6.9  17.1%
  Panel presentation     10.3  25.7%
  total                  40.0

EFFECT OF LOWERING HIRING MANAGER FAILURE PROBABILITY 0.70 -> 0.40
  Recruiter screen       1.0 ->   1.0 (+0.0)
  Hiring manager        18.0 ->  12.8 (-5.2)
  Technical screen       3.9 ->   4.8 (+1.0)
  Clinical case          6.9 ->   8.5 (+1.7)
  Panel presentation    10.3 ->  12.8 (+2.5)
```

Reading the output:

1. **The hiring manager stage takes 45 percent of the raw allocation.** That is the model
   working: a 30 percent decision weight, a 70 percent failure probability, and a cost of 5
   produce a priority of 1.05, which is more than the other four stages combined. It is also
   the number to cap manually, because 18 hours of rehearsal for a 45-minute conversation
   produces diminishing returns after about 6 to 8.
2. **The technical screen gets 3.9 hours despite a 25 percent weight.** Its failure probability
   is low and its cost is moderate. A candidate who feels anxious about technical work will
   override this and spend 20 hours there. The model exists to make that override visible.
3. **The panel presentation gets 10.3 hours**, more than double the technical allocation, because
   its failure probability is high and its cost is high. This matches the diagnostic in Section
   6.3, where the presentation was the second failure point.
4. **Revising one probability moves 5.2 hours out of the hiring manager stage.** This is the
   operation to repeat after each mock interview: revise the probability that you fail the stage
   that the mock tested, re-run, and reallocate. The plan should change after every diagnostic,
   not only at the start.
5. **The output is a starting point, not a schedule.** The model cannot see that a 45-minute
   conversation saturates after eight rehearsals, or that a presentation improves with each run
   for longer. Apply your own caps and record them next to the plan, so that a later revision
   does not silently undo the judgment.
6. **The floor matters for the recruiter screen.** Without it, the stage receives 0.35 hours,
   which rounds to no preparation. The screen is only 10 percent of the decision, but failing it
   ends the process, so one hour is a sensible minimum.

To use the scheduler, replace `STAGES` with your own estimates and set `TOTAL_HOURS` to the time
you actually have. If the output gives a stage more hours than its format can absorb, apply a cap
and redistribute the remainder by hand, recording the cap and the reason next to the plan.

## 8. Data, standards, and regulatory context

Interview practice is governed less by technical standards than by employment law. The rules
below determine what an employer may ask, how it must score, and what it must accommodate. A
candidate who understands them can recognise an unlawful or unsafe process and can ask for
adjustments without disclosing more than necessary.

### 8.1 Binding rules in the United States

| Rule | What it covers | What it means for an interview |
|---|---|---|
| Title VII of the Civil Rights Act | Discrimination on race, colour, religion, sex, national origin | Questions about these characteristics, or proxies for them, are off limits |
| Americans with Disabilities Act (ADA) | Disability discrimination and reasonable accommodation | You may request an adjustment; the employer must engage in an interactive process |
| Age Discrimination in Employment Act | Age discrimination against workers 40 and older | Questions about age or graduation year are risky for the employer |
| Uniform Guidelines on Employee Selection Procedures (1978) | Selection procedures with adverse impact | Interviews are covered as selection procedures; the four-fifths rule is used as evidence of adverse impact |
| Fair Credit Reporting Act | Background checks by third parties | Requires disclosure and authorisation |
| State and local laws | Salary history bans, pay transparency, recording consent | Salary questions may be prohibited; recording may require consent |

The Uniform Guidelines define a selection procedure to include informal and unscored interviews,
and the four-fifths rule treats a selection rate below 80 percent of the highest group's rate as
evidence of adverse impact. The Guidelines are guidance rather than statute, and the 2007 EEOC
technical assistance document carries a disclaimer that it lacks force of law, but courts and
employers treat the framework as the operative standard. The practical implication for a
candidate is that a structured, scored, job-related process is the defensible design, and an
unstructured conversation is the design most likely to produce biased outcomes.

### 8.2 Binding rules in the European Union and the United Kingdom

| Rule | What it covers | What it means for an interview |
|---|---|---|
| GDPR and UK GDPR | Personal data processing | Interview notes are personal data; retention and access rights apply |
| Equality Act 2010 (UK) | Protected characteristics | Questions about health, disability, and family plans are restricted |
| EU pay transparency directive (2023) | Pay information for candidates | Employers may not ask for salary history in some member states |
| Right-to-work checks (UK) | Eligibility | Requires original documents or a share code; timing varies |

Recording an interview requires consent in most jurisdictions, and an employer who records
without it creates a data protection problem. A candidate may ask whether a session is recorded
and may decline.

### 8.3 What is convention rather than law

- **Structured interviews** are a professional convention supported by validity evidence, not a
  legal requirement. An employer may run an unstructured process and still be lawful, provided
  the outcome does not produce unjustified adverse impact.
- **Take-homes** are a convention. There is no general legal limit on their length in the United
  States, although unpaid work that resembles employment can create risk in some jurisdictions.
  A candidate can negotiate scope and should not assume the request is immutable.
- **Scorecards and debriefs** are conventions. Their absence is a signal about process maturity,
  not a legal defect.
- **A fixed number of stages** is a convention. Some employers decide after two conversations;
  others run eight.

### 8.4 What the rules do not do

They do not make an interview fair, and they do not give a candidate a remedy for a subjective
rejection. They constrain what can be asked and how selection procedures must be justified if
their outcomes show adverse impact. The practical use for a candidate is narrower: know what you
do not have to answer, know that you can request an accommodation, and know that a process which
ignores these rules is a risk signal about the employer.

### 8.5 Questions an interviewer should not ask

| Question | Why it is a problem | What to do |
|---|---|---|
| "Are you planning to have children?" | Sex and family status discrimination | Decline to answer; redirect to availability and travel |
| "How old are you?" or graduation year | Age discrimination | Answer only if the degree date is required; otherwise redirect |
| "Do you have a disability?" | ADA restricts pre-offer disability inquiries | Describe the accommodation you need, not a diagnosis |
| "What is your current salary?" | Prohibited in several jurisdictions | Give a target range instead |
| "Where are you really from?" | National origin discrimination | State your work authorisation if relevant, nothing more |
| "Do you have childcare arranged?" | Proxy for sex discrimination | Redirect to travel and on-call expectations |
| "What is your religion?" | Religious discrimination | Decline; ask about scheduling only if relevant |

Declining should be brief and neutral: "I would rather focus on the role; I can confirm I am
available for the on-call schedule." A candidate who lectures the interviewer about the law ends
the process, even when the interviewer was wrong.

### 8.6 What an employer may ask about accommodation

An employer may ask whether you need an adjustment and what adjustment would help. It should not
ask for a diagnosis, medical records, or the nature of a condition before an offer. The practical
way to request an adjustment is to name the format and the change: "I would like 25 percent extra
time on the timed exercise, and a quiet room for the presentation." Most processes have a route
for this; using it is not a disadvantage and does not need to be raised with the hiring manager.

## 9. Industry practice and case evidence

### 9.1 Documented: structured interviews are the best-evidenced format

Campion, Palmer and Campion (1997) reviewed the structure literature and identified 15 components
of structure, including job analysis, the same questions for all candidates, anchored rating
scales, detailed notes, multiple interviewers, and statistical combination of ratings. They
reported corrected validities roughly in the range of .35 to .62 for structured interviews
against .14 to .33 for unstructured ones.

Sackett, Zhang, Berry and Lievens (2022) reanalysed the meta-analytic estimates after addressing
over-correction for range restriction. In their revised estimates, the structured interview was
the top-ranked procedure with an operational validity of .42, and the unstructured interview fell
to .19. The revision lowered almost every predictor; the rank order of the structured interview
did not change.

These are corrected correlations from meta-analyses of selection studies, mostly in
non-healthcare settings. They are the best public evidence on interview format, and they support
two practical conclusions: the questions are likely to be job-related and scored, and the
interviewer is comparing your answers against a criterion rather than forming a global
impression.

### 9.2 Documented: the legal frame treats interviews as selection procedures

The Uniform Guidelines on Employee Selection Procedures (29 CFR Part 1607) define selection
procedures to include informal and unscored interviews, and set out the four-fifths rule for
detecting adverse impact. The EEOC's 2007 technical assistance document describes the modern
disparate-impact analysis and notes that the Guidelines are guidance rather than binding
regulation. The practical consequence is that large employers have a reason to standardise
interviews and to document scoring, which is why structured loops are more common at large
employers than at small ones.

### 9.3 What employers actually run

The following patterns are common across the postings and processes reviewed for this module.
They are observations from public job descriptions and interview accounts, not a survey.

- **Health systems** often run a recruiter screen, a hiring manager conversation, a technical or
  SQL exercise, and a panel that includes a clinician. The clinical panel is usually the stage
  that decides.
- **Payers and health-tech companies** more often include a take-home or a case, and a panel of
  cross-functional stakeholders. The case is usually closer to a product decision than to a
  clinical one.
- **Pharmaceutical companies and CROs** often include a presentation and a structured
  competency-based interview, with less live coding.
- **Device and clinical AI companies** often include a model review or a design discussion, and
  increasingly a regulatory or risk discussion.
- **Academic medical centres** may include a job talk, which is a different format with different
  rules: it is assessed as a seminar, not as an interview.

### 9.4 Two formats that do not behave like interviews

**The academic job talk.** A job talk is assessed as a seminar: the audience evaluates the
research, the speaker's command of the literature, and whether the talk would attract students.
The decision-facing framing that works in industry fails here, because the audience expects a
contribution to a field, not a product decision. If you are moving between sectors, prepare two
versions of the same work: a seminar version with the literature and the methods, and an industry
version with the decision and the evidence.

**The model review.** Device and clinical AI companies often run a session in which you present a
model and defend its intended use, validation, and failure modes to a mixed group. The assessors
are looking for whether you would let the model be deployed. The strongest preparation is a model
card and a one-page risk summary; the weakest is a leaderboard of metrics.

### 9.5 Inference, labelled

- **Inference:** the hiring manager stage carries more weight than its duration suggests, because
  the manager usually writes the requisition and leads the debrief. The 30 percent weight used in
  this topic is an estimate.
- **Inference:** a panel with a clinician tests translation more than technical depth. The
  clinician is asking whether they would trust the output, not whether the model is novel.
- **Inference:** take-homes are becoming more common in health-tech and less common in large
  pharma, where presentation-based assessment dominates. This is a pattern in postings and
  accounts, not a measured trend.
- **Inference:** candidates who ask about decision rights and success criteria are read as more
  senior, because the questions imply experience of owning outcomes.

> **Note:** No placement rate or pass rate is quoted in this section, because no public,
> methodologically sound dataset of stage-level outcomes in biomedical data hiring was identified.
> The validity coefficients above describe selection procedures in general, not your probability
> of passing a particular loop.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Preparing only technical content | Mock interviews are all coding | Technical is the most familiar and least weighted | Weak stages decide the outcome | Build the stage map and allocate by expected loss |
| Ignoring the recruiter screen | No 30-second story prepared | Assumed to be a formality | Process ends before the technical stage | Prepare a scripted 30 seconds and a constraints answer |
| Generic motivation | "I want to work in healthcare" | No employer-specific research | Manager infers flight risk | Name the employer's problem and the role's decision |
| Overrunning the time box | Practice runs finish late | No timed rehearsal | The artifact is incomplete | Time every rehearsal; cut content, not delivery |
| Presentation starts with background | Slide 3 is "agenda," slide 8 is the decision | Academic habit | Panel stops listening before the evidence | Decision on slide one |
| Case answer without a structure | You start listing ideas | No framework announced | Interviewer cannot follow or score | State the structure in the first 30 seconds |
| Treating the panel as a series of one-to-ones | You address only the questioner | No awareness of the room | Other panellists score you as disengaged | Include the room, then return to the questioner |
| No questions prepared | You ask about benefits only | Preparation focused on answers | Read as passive or uninterested | Prepare three questions per stage |
| Defensiveness under challenge | You repeat the same answer louder | Threat response | Interviewer marks low coachability | Acknowledge, ask for the concern, adjust |
| Hiding a gap or a constraint | You avoid the topic | Fear of disqualification | Discovery later ends the process | One factual sentence, then evidence |
| Not debriefing | You cannot recall the questions | Memory decay | The same mistake repeats at the next employer | One-page note within 24 hours |
| Accepting an unbounded take-home | The scope grows after you start | No stated time cap | Resentment, late submission, or both | Negotiate scope before accepting |
| Ignoring logistics | You join from a café | Underestimating the format | Avoidable failure independent of ability | Checklist 24 hours before each stage |
| Preparing only for the strongest interviewer | You know one panellist's background | Effort follows familiarity | The silent panellist's score decides | One artifact per panellist's stake |
| Over-rehearsing to the point of recital | Answers sound memorised | Scripting without internalising | Interviewer discounts authenticity | Rehearse structure, not sentences |

Two red flags are about the employer rather than the candidate. A process with no stated stages,
no named interviewers, and no timeline is usually a process with no owner, and the same
disorganisation will appear in the first 90 days. A panel that asks about protected
characteristics or family plans is a legal and cultural risk signal; you are not obliged to
answer, and you may end the process.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The process has a decodable structure | It enables stage-specific preparation | Some employers improvise | Ask the recruiter for the stage list |
| Stage weights can be estimated | They drive allocation | Weights differ by employer and interviewer | Ask which stage candidates find hardest |
| Failure probabilities can be self-assessed | They order the work | Self-assessment is poorly calibrated | Compare with mock interview ratings |
| Structured interviews are the norm in large employers | It justifies criterion-based answers | Small employers often run unstructured conversations | Ask how answers are scored |
| A 40-hour budget is realistic | It bounds the plan | Searches run alongside full-time work | Track actual hours for one week |
| Mock interviews predict real performance | They justify rehearsal | Mock partners are not the real panel | Compare mock and real feedback |
| Questions are job-related | It justifies preparing evidence rather than trivia | Some interviewers ask puzzles | Ask what the stage is designed to assess |
| The recruiter will answer process questions | It enables planning | Some recruiters deflect | Ask two questions and observe |
| Preparation transfers across employers | It justifies reusable stories | Each process has different formats | Reuse the stories, adapt the artifact |
| A rejected stage is informative | It justifies the debrief log | Rejections often reflect competition, not error | Distinguish process errors from outcome noise in the note |

The main limit is that interview outcomes contain a large component of interviewer preference and
team circumstance that no preparation controls. The model in Section 3.4 allocates effort toward
the stages you can influence; it does not predict the outcome. A rejection after a well-prepared
loop is often about a competing candidate or an internal constraint, and the debrief log is the
only way to distinguish that from a fixable error.

A second limit is that the model treats stages as independent. In practice, a strong hiring
manager stage changes the questions the panel asks, and a weak technical stage changes the
panel's willingness to be persuaded. The allocation is still useful, but the stages are not
separable, and a failure early in the loop is more costly than its weight suggests.

## 12. Exercises

**Exercise 1. Stage map (recall, 30 minutes).**
Deliverable: a table of stages for one live process, with interviewer roles, formats, durations,
and the inference each stage tests.
Acceptance criteria: every stage has a named format and a named inference; at least one stage is
identified as decisive; the source of the information is recorded; the table is updated after
each stage with what was actually asked.

**Exercise 2. Artifact per stage (application, 60 minutes).**
Deliverable: one preparation artifact for each stage scoring below 4 on confidence.
Acceptance criteria: each artifact matches the format (not a deck for a coding screen); each has
a time box; each is rehearsed once.

**Exercise 3. Allocation (application, 30 minutes).**
Deliverable: a 40-hour plan produced with the Section 7 script, plus a manual cap and
redistribution where a stage cannot absorb its allocation.
Acceptance criteria: weights sum to 100 percent; the plan totals 40 hours; every cap is
justified in one line; the technical allocation is not automatically the largest.

**Exercise 4. Three lengths (application, 45 minutes).**
Deliverable: 30-second, 2-minute, and 10-minute versions of your strongest project.
Acceptance criteria: each version stands alone; the 30-second version contains one number; the
10-minute version contains a limitation and a next step; each is delivered within 10 percent of
its limit.

**Exercise 5. Mixed-audience presentation (design, 2 hours).**
Deliverable: a 7-slide, 10-minute presentation for a panel containing a clinician and a product
manager.
Acceptance criteria: the decision is on slide 1; the limitation is before the final slide; the
timed run finishes under 10 minutes; a non-specialist can state the decision after one viewing.

**Exercise 6. Question set (application, 30 minutes).**
Deliverable: three questions per stage, tagged by category (scope, success, risk, workflow,
process).
Acceptance criteria: at least one question per stage asks about decision rights; no question can
be answered from the careers page; the list is ranked.

**Exercise 7. Format mismatch drill (critique, 30 minutes).**
Deliverable: three responses to a format that differs from what you prepared, using the
clarify-adapt-defer options.
Acceptance criteria: each response includes a clarifying question; none becomes defensive; at
least one response defers a follow-up in writing.

**Exercise 8. Debrief log (application, 20 minutes per stage).**
Deliverable: a one-page note after each stage with questions, answers, reactions, and one change.
Acceptance criteria: written within 24 hours; includes one question you fumbled; the next
rehearsal addresses it.

## 13. Self-check answers

**1. What does each stage test?**
The recruiter tests fit and constraints; the hiring manager tests ownership and judgment; the
technical stage tests fluency under time pressure; the case tests structuring ambiguity; the
panel tests cross-stakeholder credibility; the presentation tests communication to a mixed
audience; the take-home tests unsupervised judgment.

**2. Why is a deck the wrong artifact for a technical screen?**
Because the screen measures production under constraint, not presentation. The interviewer needs
to see you write, test, and explain code or a query. A prepared deck demonstrates preparation for
a different task, and it consumes time that the exercise needs.

**3. How does the allocation change if your technical confidence is low?**
It rises until the marginal hour is worth more elsewhere. In the Section 7 model, a technical
failure probability of 0.6 rather than 0.3 roughly doubles its allocation, taking hours from the
panel and case. The model does not know that you are anxious; it knows the estimate you gave it,
which is why the estimate should be calibrated against a mock.

**4. What are the three lengths for, and what changes between them?**
Thirty seconds for the recruiter and the opening; two minutes for behavioural and hiring manager
questions; ten minutes for a presentation. The content changes, not just the duration: the short
version is a positioning statement, the middle is a story, the long is an argument with evidence.

**5. What belongs on slide one of a panel presentation?**
The decision, the actor, and the consequence. Not the agenda, not the background, not the
dataset. The panel should know within 30 seconds why the work matters and what it changed.

**6. What three questions should you ask a hiring manager?**
What decisions the role owns without escalation; what a strong first six months looks like and
who judges it; what has gone wrong recently and what changed. Each reveals scope and culture, and
each signals experience.

**7. The interviewer challenges your approach aggressively. What do you do?**
Acknowledge the concern, ask a clarifying question to locate it, and adjust or defend with
evidence. Do not repeat the same answer more loudly, and do not capitulate without a reason.
Interviewers score composure and coachability, not agreement.

**8. What goes in a debrief note?**
Stage, interviewer roles, the three questions asked, one-line answers, the reaction observed, the
question you fumbled, and the one change for the next rehearsal. Written within 24 hours.

**Rubric for Exercise 3.** Full marks: weights sum to 100 percent; the plan totals the available
hours; every stage has at least one hour; at least one cap is applied with a written reason;
technical practice is not automatically the largest block; the plan names what will be dropped if
a stage takes longer than expected.

**Rubric for Exercise 5.** Full marks: seven slides or fewer; decision on slide one; limitation
before the end; timed under 10 minutes; a non-specialist can restate the decision; at least one
slide addresses workflow or operations.

**Rubric for Exercise 7.** Full marks: a clarifying question in each response; adaptation rather
than repetition; one deferral with a concrete follow-up; no defensiveness; the response stays
within the time box.

## 14. Interview questions

**1. "Tell me about yourself."**
Strong: 30 to 45 seconds, role, domain, one proof point, and the target. Weak: a chronological
history that reaches the present after three minutes.

**2. "Why this role, and why us?"**
Strong: names the employer's problem, the role's decision, and the transferable evidence. Weak:
mission language that could apply to any employer.

**3. "How would you approach a problem you have never seen?"**
Strong: states a structure, asks two clarifying questions, and names what would make them stop.
Weak: starts listing techniques.

**4. "Walk me through a failure."**
Strong: owns the failure, names the diagnostic, and describes the specific change. Weak: blames
the data or the stakeholder.

**5. "How do you handle disagreement with a clinician?"**
Strong: describes a case, the evidence brought, the escalation path, and the outcome. Weak:
claims to avoid conflict or to defer always.

**6. "What would you do in your first 90 days?"**
Strong: learn the data and workflow, reproduce a known result, deliver one small improvement,
and name the measures. Weak: lists tools to learn.

**7. "How do you prioritise when everything is urgent?"**
Strong: names the decision, the cost of delay, and the stakeholder who arbitrates. Weak: says
they make a list.

**8. "What questions do you have for us?"**
Strong: two or three questions about decision rights, success criteria, and failure. Weak: asks
only about benefits or says no questions.

**9. "Your presentation was strong. What would you do differently?"**
Strong: names a specific improvement, such as the missing cost slide or the overlong cohort
section. Weak: says nothing, or lists a generic improvement.

**10. "Why should we hire you over another candidate?"**
Strong: restates the role's decision and the two pieces of evidence that match it, without
disparaging others. Weak: lists every skill they have.

## 15. Key terms

- **Accommodation** — an adjustment to the process for a disability or need.
- **Adverse impact** — a substantially different selection rate between groups.
- **Anchored rating scale** — a rating scale with behavioural anchors per point.
- **Bandwidth** — how much information a stage can carry.
- **Bar raiser** — an interviewer with veto power over standards.
- **Behavioural question** — a question about past behaviour.
- **Calibration** — comparing ratings across interviewers.
- **Case interview** — an open-ended problem structured aloud.
- **Clinical case** — a case built around a patient or workflow decision.
- **Debrief** — the meeting that combines interviewer ratings.
- **Fidelity** — how well a stage predicts job performance.
- **Four-fifths rule** — selection rate below 80 percent of the highest group's rate.
- **Hiring manager interview** — conversation with the person who owns the role.
- **Job analysis** — systematic description of the work used to build the interview.
- **Loop** — the full sequence of stages for one role.
- **Panel** — several interviewers in one session.
- **Recruiter** — the person managing the process and requisition.
- **Scorecard** — the interviewer's structured rating form.
- **Screen** — a short interview to check basic fit and constraints.
- **Signal** — the evidence a stage is designed to detect.
- **Situational question** — a hypothetical scenario question.
- **Structured interview** — fixed questions with anchored scoring.
- **System design** — design of a system under constraints.
- **Take-home** — an unsupervised exercise on your own time.
- **Technical screen** — a timed exercise in coding, SQL, or statistics.

## 16. Sources

All URLs accessed 26 October 2026.

1. Campion MA, Palmer DK, Campion JE. A review of structure in the selection interview.
   *Personnel Psychology* 1997;50(3):655–702. https://doi.org/10.1111/j.1744-6570.1997.tb00709.x
2. Sackett PR, Zhang C, Berry CM, Lievens F. Revisiting meta-analytic estimates of validity in
   personnel selection: Addressing systematic overcorrection for restriction of range. *Journal
   of Applied Psychology* 2022;107(11):2040–2068. https://doi.org/10.1037/apl0000994
3. U.S. Equal Employment Opportunity Commission. *Uniform Guidelines on Employee Selection
   Procedures*, 29 CFR Part 1607. https://www.ecfr.gov/current/title-29/subtitle-B/chapter-XIV/part-1607
4. U.S. Equal Employment Opportunity Commission. *Questions and Answers to Clarify and Provide a
   Common Interpretation of the Uniform Guidelines.*
   https://www.eeoc.gov/laws/guidance/questions-and-answers-clarify-and-provide-common-interpretation-uniform-guidelines
5. U.S. Equal Employment Opportunity Commission. *Employment Tests and Selection Procedures.*
   https://www.eeoc.gov/laws/guidance/employment-tests-and-selection-procedures
6. U.S. Department of Justice. *Americans with Disabilities Act.* https://www.ada.gov/
7. European Commission. *General Data Protection Regulation (EU) 2016/679.*
   https://eur-lex.europa.eu/eli/reg/2016/679/oj
8. UK Government. *Equality Act 2010 guidance.* https://www.gov.uk/guidance/equality-act-2010-guidance
9. UK Home Office. *Right to work checks: employer guidance.*
   https://www.gov.uk/check-job-applicant-right-to-work
10. U.S. Federal Trade Commission. *Fair Credit Reporting Act.* https://www.ftc.gov/
11. Campion MA, Palmer DK, Campion JE. Structuring employment interviews to improve
    reliability, validity, and users' reactions. *Current Directions in Psychological Science*
    1998;7(3):72–77. https://doi.org/10.1111/1467-8721.ep10773001
12. Society for Human Resource Management. *Structured interview guidance and question banks.*
    https://www.shrm.org/

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`03_resume-and-linkedin.md`](03_resume-and-linkedin.md)
- Next topic: [`05_technical-interviews.md`](05_technical-interviews.md)
- Related topic: [`08_behavioural-interviews.md`](08_behavioural-interviews.md)
- Survey file: [`../../course/20_career_portfolio_interviews.md`](../../course/20_career_portfolio_interviews.md)
- Labs: [`../../labs/`](../../labs/)
