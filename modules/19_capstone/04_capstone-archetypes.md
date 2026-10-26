# Capstone Archetypes

> **Module 19 — Capstone, Hackathons, and Competitions**
> Topic 04 of 11
> Estimated study time: 3.5 hours
> Prerequisites: Topic 01
> Version 2026.3

## 1. Why this topic exists

Two capstones can use the same dataset, the same model, and the same metric, and carry completely
different evidence burdens.
One is a benchmark study that asks whether a published method reproduces on a public split.
The other is a triage claim that asks whether a regulator should permit a device to reorder a
radiologist's worklist.
The first can be completed by four people in twelve weeks. The second cannot, and a team that
does not notice the difference will spend eleven weeks producing evidence for a claim it is not
entitled to make.

The archetype is the name for that difference.
It is a project shape defined by the decision it supports, and each shape carries its own
evidence bar, its own characteristic failure, and its own regulatory exposure.
Choosing the archetype is therefore the first scoping decision, and it is made before the data
are inspected.

Teams get this wrong in a consistent way: they choose the archetype that matches the data they
have rather than the decision they can own.
A team with a public chest radiograph dataset chooses detection, because detection is what the
dataset supports.
A team with electronic health record extracts chooses prognosis, because prognosis is what the
columns suggest.
The result is a project whose evidence bar is set by the data rather than by the claim, and the
gap between the two appears at the defense as an unanswerable question: "Who would act on this,
and what would they do differently?"

The cost is not only a lower grade.
In industry, a team that builds a detection model without the reader-study evidence required for
a triage claim has produced a research artefact that cannot be submitted, and the regulatory
work must be redone from the beginning when the claim is corrected.
A screening programme built on aggregate accuracy without subgroup analysis can widen the very
disparities it was meant to reduce.
An evaluation study that reports a benchmark win without a contamination check can send an
organisation down a two-year procurement path based on a number that does not replicate.

This topic gives the team a decision procedure for choosing among eight shapes, a scoring
instrument for testing the choice against its constraints, and a set of worked comparisons that
show how the evidence burden changes when one assumption changes.
It does not rank the archetypes by prestige. A well-executed evaluation study is a stronger
capstone than a poorly executed triage claim, and panels score accordingly.

Prestige bias is real and worth naming.
Detection and triage projects attract attention because they look like products, and a team that
chooses one feels closer to a start-up than a team comparing two feature sets.
The bias is reinforced by public competitions, whose leaderboards are visible and whose winners
are celebrated, while the evaluation work that establishes whether a leaderboard means anything
is invisible.
A panel that has seen ten triage claims and one clean evaluation study is not impressed by the
eleventh claim.

Archetype drift is the second force this topic guards against.
A team begins with a modest question, discovers that a stronger claim would be more impressive,
and adds a triage framing in week 8 without adding the evidence.
The drift is rarely dishonest; it happens one slide at a time, as each sentence is softened just
enough to sound clinical.
The archetype record, signed at G0 and reviewed at the midpoint, is what makes the drift visible.

What the archetype is not: it is not a template for the model, a required dataset, or a
difficulty rating. Two teams can choose A5 and produce wildly different projects, one a careful
replication and one a shallow leaderboard report. The archetype fixes the claim and the evidence
bar; everything else is the team's work.

## 2. Learning outcomes

By the end of this topic you can:

1. **Classify** a proposed capstone into one of the eight archetypes and justify the choice by
   the decision it supports rather than by the data available.
2. **State** the minimum evidence bar for each archetype, including the validation level and the
   regulatory artefact that the claim implies.
3. **Compute** the evidence burden of an archetype in hours and weeks, and compare it with the
   team's available capacity.
4. **Identify** the characteristic failure of an archetype and the early signal that it is
   occurring.
5. **Score** candidate archetypes against six criteria and test whether the ranking is sensitive
   to the weights.
6. **Select** a primary archetype and a fallback that preserves the scientific question if data
   access fails.
7. **Narrow** a claim from a stronger archetype to a weaker one without losing the decision
   relevance, and write the narrowing into the charter.
8. **Defend** the choice of a less prestigious archetype on evidence grounds.

## 3. The core mental model

An archetype is a point in a two-dimensional space.
The horizontal axis is intervention proximity: how directly the output changes what someone does
to a patient.
The vertical axis is evidence burden: how much validation, governance, and documentation the
claim requires before it can be believed.

```text
  evidence burden
        high |  A7 trial evidence        A1 detection / triage
             |        (estimand,               (site-held-out,
             |         bias, precedent)         reader study, SaMD)
             |
             |  A6 omics                 A8 screening
             |   (external cohort,        (equity-weighted,
             |    replication)             cost per case)
             |
             |  A4 documentation         A2 prognosis
             |   (factuality, audit)      (temporal, calibration)
             |
        low  |  A5 evaluation            A3 operations
             |   (harness, grader)        (backtest, cost of error)
             +---------------------------------------------------
                indirect                              direct
                        intervention proximity
```

Read the diagram as a constraint map, not a quality ranking.
Moving right increases the strength of the claim and the cost of supporting it.
Moving up increases the evidence burden without necessarily increasing clinical value.
A team can produce an excellent capstone anywhere on the map, provided the claim matches the
position.

Three properties of the map matter.

**Position is set by the claim, not the method.** A logistic regression used to reorder a
radiologist's worklist sits at A1, because the claim is about triage. A deep network used to
compare two segmentation methods on a public benchmark sits at A5, because the claim is about
methods. The model does not determine the archetype; the sentence describing what changes does.

**The evidence bar is a floor, not a target.** A1 requires site-held-out validation and a
reader-study logic; a team that does temporal validation has not met the floor, regardless of how
good the model is. The floor is set by what a skeptical reviewer would need, not by what the
team can produce.

**Falling down or left is a legitimate move.** A team that cannot reach the A1 floor can move to
A5 and study the same question as an evaluation. The decision relevance falls, but the evidence
can be complete. A team that moves down without saying so has made an undisclosed claim, which
is worse than a modest one.

A fourth property is about time.
The vertical axis is expensive in calendar time as well as effort. A reader study needs reader
recruitment and scheduling; a site-held-out validation needs a data-sharing agreement; a trial
evidence project needs precedent analysis. The twelve-week clock constrains the archetype more
than the team's skill does.

### Moving between archetypes

A team can move between archetypes in three ways, and only one is honest.

The first is narrowing: keep the archetype and reduce the claim. An A2 project that cannot
recruit a clinician narrows from "who receives outreach" to "whether the score is calibrated well
enough to support outreach", which is still A2 but with a weaker verb.

The second is stepping down: change the archetype and keep the question. An A1 project without
reader capacity becomes A5, comparing detection methods on a public split, with the triage
question recorded as future work.

The third is inflation: keep the weaker evidence and strengthen the claim. This is never honest,
and it is the move that panels are trained to catch. It usually appears as a slide that says
"detects" while the dossier says "prioritises", or as an intended use that names a clinical
workflow while the validation is a random split.

The record from Step 14 should distinguish the three, because the first two are legitimate and
should be described as such. A team that says "we stepped down from A1 to A5 because we could not
recruit readers" is demonstrating judgement. A team that says nothing and hopes the panel does
not notice is demonstrating the opposite.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Archetype | A project shape defined by the decision it supports and the evidence it requires | A model family or a dataset type | A1 to A8 |
| Intervention proximity | How directly the output changes clinical action | Confused with model accuracy | Indirect to direct |
| Evidence bar | The minimum validation and governance a claim requires | The evidence the team can produce | Validation level plus artefacts |
| Evidence burden | The effort and elapsed time needed to meet the bar | The modelling effort | Hours and weeks |
| Regulated detection or triage | A1: deciding which case is read, escalated, or recalled | Any image classifier | SaMD pathway |
| Prognostic risk stratification | A2: deciding who receives an intervention or outreach | Any score with a time horizon | Temporal validation |
| Operational forecasting | A3: deciding how to staff, schedule, or allocate | Any regression on operational data | Backtest |
| Documentation or agentic workflow | A4: deciding whether a draft is accepted, edited, or rejected | Any text generator | Task success and audit |
| Evaluation and benchmark science | A5: deciding which method or model to adopt | A leaderboard entry | Reproducible harness |
| Translational biomarker or omics | A6: deciding which target, variant, or group advances | A discovery on one cohort | External cohort |
| Trial and regulatory evidence | A7: deciding whether a trial, endpoint, or strategy is viable | A causal claim from convenience data | Estimand and precedent |
| Population and screening programme | A8: deciding where to allocate screening or surveillance | A model with a threshold | Equity-weighted evaluation |
| Reader study | A study in which clinicians read cases with and without the model | A demo to one clinician | Readers × cases × sessions |
| Multi-reader multi-case (MRMC) | A design that estimates reader performance accounting for reader and case variation | A simple average of readers | Variance components |
| Site-held-out validation | Evaluation on a site not used in development | A random split with a site column | L3 |
| Temporal validation | Evaluation on a later period than development | A random split with dates present | L2 |
| Analytical validity | Whether a measurement or model output is reproducible and accurate | Clinical usefulness | Technical performance |
| Clinical validity | Whether the output is associated with the clinical outcome | Clinical utility | Association and discrimination |
| Clinical utility | Whether using the output improves decisions or outcomes | Any positive AUC | Net benefit or trial |
| Benchmark contamination | Test data present in pretraining or tuning data | Leaderboard overfitting | Contamination screen |
| Grader validity | Whether an evaluation rubric measures the intended capability | Inter-rater agreement alone | Agreement and bias analysis |
| Intended use | Population, setting, inputs, output, user, and contraindications | A description of the model | One paragraph |
| Predicate | A legally marketed device to which equivalence is claimed | Any similar paper | 510(k) comparison |
| De novo pathway | A regulatory route for a novel low-to-moderate-risk device with no predicate | A shortcut | FDA classification |
| Equity-weighted evaluation | Evaluation that weights performance by group need or harm | Reporting a subgroup table | Weights and justification |
| Backtest | Evaluation of a forecast on historical periods as if made in real time | A random split | Periods and horizon |
| Cost of error | The clinical, operational, and financial consequence of each error type | A confusion matrix | Currency and harm |
| Claim verb | The action the evidence supports: detects, prioritises, forecasts, or recommends | Chosen for impact rather than evidence | One verb per claim |
| Stepping down | Changing to a lower-burden archetype while keeping the question | Claiming the original archetype anyway | Recorded decision |
| Narrowing | Keeping the archetype and reducing the claim | Silent weakening of the verb | Recorded decision |
| Inflation | Keeping the evidence and strengthening the claim | The default drift | Detected at G4 |

The distinction between narrowing, stepping down, and inflation is the most useful vocabulary in
this topic. It gives a team a way to describe a change honestly, and it gives a panel a way to
ask which one happened.

The most consequential definition is the evidence bar.
It is not a statement about how good the model must be; it is a statement about what must be
demonstrated before the claim is admissible.
A model with an area under the curve of 0.95 and no site-held-out validation does not meet the
A1 bar. A model with 0.72 and a clean L3 evaluation does.

## 5. The mechanics, step by step

Choosing an archetype is a fourteen-step procedure that takes about two hours with the team and
the clinical mentor. It produces a one-page archetype decision record.

### Step 1. Write the decision in plain language

**Input:** the problem area and the sponsor's request.

**Output:** one sentence naming the action, the actor, and the frequency.

**Rationale:** The archetype is derived from the decision, and a vague decision cannot be
classified. "Improve radiology" is not classifiable; "the on-call radiologist decides which head
CT to read next" is A1.

**Branch point:** if the sentence contains two actions with different actors, there are two
archetypes. Choose one for the capstone and record the other as out of scope.

### Step 2. Locate the output in the workflow

**Input:** the decision sentence and a workflow map.

**Output:** the point at which the output arrives, the time available, and whether the user can
ignore it.

**Rationale:** Workflow position determines intervention proximity. An output that arrives
before a decision and can be ignored sits lower on the axis than an output that gates an action.

**Branch point:** if the output arrives after the decision, the project is retrospective and
belongs to evaluation or prognosis, not detection.

### Step 3. Identify the claim type

**Input:** the decision and the workflow position.

**Output:** one of four claim types: detection or triage, prognosis, operational forecast, or
documentation.

**Rationale:** Claims about what is present, what will happen, what will be needed, and what
should be written carry different evidence requirements. Naming the type prevents a detection
question from being answered with a prognosis design.

**Branch point:** if the claim is about a method rather than a decision, the archetype is A5.
That is a legitimate and often stronger choice.

### Step 4. Check the regulatory exposure

**Input:** the claim type, the user, and the output.

**Output:** a preliminary classification: device function, clinical decision support, research
use only, or not a medical claim.

**Rationale:** Regulatory exposure is a property of the claim and determines the binding
artefact. Under the FDA clinical decision support guidance, software that analyses an image or
signal, or provides a time-critical alert, or gives a specific directive, is likely a device.
That classification changes the evidence bar.

**Branch point:** if the team cannot determine the classification, assume the stricter one for
planning and record the open question for the regulatory mentor.

### Step 5. Set the validation level from the claim

**Input:** the deployment setting and the claim.

**Output:** L2, L3, or L4, with a sentence on what the level proves.

**Rationale:** A claim about next year requires temporal validation; a claim about a new site
requires site-held-out validation; a claim about a method requires a reproducible harness with a
held-out split. The level is derived, not chosen for convenience.

**Branch point:** if the required level is L4 and the project is twelve weeks, the claim must be
narrowed. Prospective evaluation is out of scope for a course capstone.

### Step 6. Compute the evidence burden in hours

**Input:** the validation level, the artefacts required, and the archetype.

**Output:** an estimate of hours for validation, governance, and documentation, separate from
modelling hours.

**Rationale:** The modelling hours are usually a minority of the total. A reader study, a
site agreement, or an external cohort can each exceed the modelling effort. Computing the
burden makes the archetype decision quantitative rather than aspirational.

**Branch point:** if the burden exceeds 40 percent of available capacity, the archetype is too
ambitious for the team, regardless of its clinical value.

A quick calibration: a four-person team with 45 hours per week across twelve weeks has 540 hours
in total. If the evidence package needs 80 hours and modelling needs 200, the remaining 260 hours
cover reviews, writing, integration, and rework, which is roughly the ratio that a well-run
project needs. A package needing 160 hours leaves the team with almost nothing for integration,
and integration is where dossiers fail.

### Step 7. Check the data can support the level

**Input:** the available datasets and their provenance.

**Output:** for each candidate dataset, the sites, the date range, the labels, and the licence.

**Rationale:** A single-site dataset cannot support L3. A dataset with no dates cannot support
L2. A public benchmark with contaminated pretraining cannot support a methods claim without a
contamination check.

**Branch point:** if no dataset supports the required level, either narrow the level and the
claim, or change archetype. Do not relabel a random split as external validation.

### Step 8. Check the owner can support the archetype

**Input:** the clinical or operational owner and their available hours.

**Output:** the owner's committed time and the specific decisions they will make.

**Rationale:** A1 requires a clinician who can review errors and sit for a reader-workflow
discussion; A2 requires an owner who controls an intervention; A8 requires a programme owner who
can act on the allocation. An archetype without an owner is a methods study wearing a clinical
claim.

**Branch point:** if the owner can give fewer than two hours per week, choose A5 or A3, both of
which can be executed with lighter clinical involvement.

### Step 9. Check the team's capability against the archetype

**Input:** the role map from Topic 03.

**Output:** a capability score for each archetype: can the team produce the artefacts at all?

**Rationale:** A6 requires omics analysis and external cohort access; A7 requires estimand and
bias-analysis skills; A4 requires evaluation design for generated text. A team without the skill
can still choose the archetype, but must budget learning time and reduce scope.

**Branch point:** if the capability gap is in the artefact that carries the claim, the archetype
is wrong. If it is in a supporting artefact, it can be learned or outsourced to a mentor.

### Step 10. Score the candidates

**Input:** steps 1 to 9.

**Output:** a weighted score for each candidate archetype on six criteria: decision clarity,
owner access, data fitness, evidence feasibility, regulatory tractability, and team capability.

**Rationale:** The score makes trade-offs explicit and exposes an archetype that is strong on
data but weak on ownership. The weights encode the programme's priorities and should be agreed
before scoring.

**Branch point:** if the top two are within 0.3 points, the decision is not sensitive to the
score. Choose on the characteristic failure the team can best tolerate.

### Step 11. Test the sensitivity of the ranking

**Input:** the scores and weights.

**Output:** the ranking under at least one alternative weighting, and the criterion that
dominates each archetype.

**Rationale:** A ranking that flips when a weight moves by 0.1 is not a decision; it is a
reflection of the weights. A ranking that is stable is robust to reasonable disagreement.

**Branch point:** if the ranking flips, identify the single criterion responsible and resolve it
with evidence rather than with weights. Usually the criterion is owner access.

### Step 12. Name the characteristic failure

**Input:** the leading archetype.

**Output:** the failure mode most associated with it, the early signal, and the mitigation.

**Rationale:** Each archetype fails in a predictable way. A1 fails by treating a research AUC as
a clinical claim; A2 by producing an unactionable score; A3 by optimising a proxy; A4 by fluent
output without provenance; A5 by leaderboard overfitting; A6 by discovery without replication;
A7 by causal claims from convenience samples; A8 by aggregate accuracy hiding subgroup harm.

**Branch point:** if the team cannot describe a mitigation for the characteristic failure, it
has not understood the archetype well enough to choose it.

### Step 13. Define the fallback archetype

**Input:** the leading archetype and the data-access plan.

**Output:** a second archetype that preserves the scientific question if the primary data fail.

**Rationale:** Fallbacks must preserve the question, not just the data. If the primary is A1 and
the imaging data fail, A5 on a public benchmark preserves the methods question but not the
triage claim. That is acceptable if the narrowing is declared.

**Branch point:** if no fallback preserves the question, the project depends on a single
external process. Record that dependency as the top risk and consider changing the brief.

### Step 14. Record the decision

**Input:** steps 1 to 13.

**Output:** a one-page record with the chosen archetype, the claim, the validation level, the
evidence bar, the burden estimate, the characteristic failure, and the fallback.

**Rationale:** The record is reviewed at G0 and at the midpoint. It prevents the archetype from
drifting as the team discovers what is easy, which is the most common form of undisclosed claim
inflation.

**Branch point:** if the archetype changes after G2, the change is a scope change and requires
the panel's agreement, a new claim, and a new evidence plan.

### Evidence burden by archetype

The table gives order-of-magnitude hours for the work beyond modelling, for a four-person team
using accessible data. The ranges are planning estimates, not benchmarks.

| Archetype | Validation | Governance | Documentation | External dependency | Typical weeks |
|---|---:|---:|---:|---|---:|
| A1 detection | 60-90 | 30-50 | 20-30 | Reader study, site data | 10-16 |
| A2 prognosis | 30-50 | 20-30 | 15-25 | Owner for intervention | 8-12 |
| A3 operations | 20-35 | 10-20 | 10-20 | Historical operational data | 6-10 |
| A4 documentation | 35-55 | 20-35 | 15-25 | Clinical review of outputs | 8-12 |
| A5 evaluation | 40-70 | 10-20 | 10-20 | Compute, clean benchmark | 6-10 |
| A6 omics | 50-80 | 20-40 | 15-25 | External cohort | 10-16 |
| A7 trial evidence | 60-100 | 40-70 | 25-40 | Trial or registry access | 12+ |
| A8 screening | 50-80 | 30-50 | 20-30 | Programme owner, equity data | 10-16 |

The external dependency column is the one that determines feasibility. An archetype whose
governance hours are large but internal is schedulable; an archetype whose validation depends on
five readers or a second site is not, unless those commitments exist in writing.

### Three constraints that bind first

Three constraints determine most archetype choices, and they bind in a fixed order.

First, ownership. If no one owns the decision, no clinical claim is admissible. This constraint
eliminates A1, A2, and A8 for teams without a committed owner, regardless of data.

Second, validation. If the data cannot support the level the claim requires, the claim must
narrow or the archetype must change. This constraint eliminates A1 for single-site data and A6
for single-cohort data.

Third, capacity. If the evidence burden exceeds the team's hours, the archetype is too large.
This constraint is the easiest to underestimate, because the burden is mostly governance and
validation rather than modelling.

The order matters. A team that starts with capacity and works backwards will choose a small
project that may still be unadmissible. A team that starts with ownership and validation will
choose a project that can be defended, and then fit it to capacity by narrowing scope.

### Estimating the burden in twenty minutes

Before the full arithmetic, a team can produce a rough estimate with three questions.

How many external people must act for the validation to count? Each external person adds
scheduling latency of at least two weeks and a withdrawal risk that the team cannot control.

How many artefacts require a signature the team cannot give? Each one adds a review cycle and a
dependency on someone outside the project.

How many weeks of data does the target precision require? If the answer exceeds the available
history, the claim must narrow before any model is built.

A team that answers these three questions in week 1 avoids most of the archetype failures listed
in Section 10. The answers do not need to be precise; they need to be written down and revisited
at G2.

### Decision table: archetype selection under constraint

| Constraint observed | Preferred archetype | Acceptable alternative | Avoid | Why |
|---|---|---|---|---|
| No clinical owner available | A5 evaluation | A3 operations | A1, A2, A8 | Claims need an owner to be admissible |
| Single-site data, no dates | A5 evaluation | A4 documentation | A1, A2 | L3 and L2 are not possible |
| Imaging data, no reader capacity | A5 evaluation | A1 with a narrowed methods claim | A1 triage | Reader studies need clinician time |
| Strong operations partner, weak clinical | A3 operations | A2 prognosis | A1 | Operational decisions are ownable |
| Omics data, one cohort | A5 evaluation | A6 with replication deferred | A6 discovery claim | Replication is the bar |
| LLM access, no clinical workflow | A4 documentation | A5 with a grader study | A4 deployment | Factuality needs evaluation design |
| Public benchmark, limited compute | A5 evaluation | A6 if external cohort exists | A1 | Benchmarks are the cheapest evidence |

The table is a starting point, not a rule. Its purpose is to make the trade-off visible before
the team invests in a direction.

### Edge cases and forced decisions

**The data arrive after the archetype is chosen.** If the primary dataset is delayed and the
fallback supports a different archetype, the team must re-record the claim, not just re-run the
pipeline. An A1 project that switches to a public benchmark has become A5, and the charter, the
intended use, and the regulatory memo all change.

**The owner withdraws.** The archetype loses its admissibility. The options are a replacement
owner, a narrower claim with a documented need assessment, or a move to A5. Continuing with an
A1 claim and no owner is the failure the panel is trained to detect.

**The team is strong on modelling and weak on governance.** A5 is the honest choice. A5 still
requires a reproducible harness, a contamination check, and a statistical comparison, but it does
not require a regulatory pathway or a reader study. A team that chooses A1 because it is more
prestigious will spend its governance hours badly.

**The claim is clinically important but the evidence is impossible.** Record the importance and
the impossibility together. A capstone can argue that the decision matters and that the
twelve-week evidence cannot support it, and propose the study that would. That is a stronger
result than a weak claim.

**A regulator would classify the output as a device, and the team did not plan for it.** Move the
claim to research use, state the classification in the memo, and describe the evidence that a
device submission would require. Do not describe a device function as a quality improvement
project to avoid the pathway.

**The team wants to do two archetypes.** Choose one as primary and one as a secondary analysis
with its own, smaller claim. Two primary archetypes produce two half-finished dossiers.

## 6. Worked example

A four-person team has access to a public chest radiograph dataset with 120,000 images from one
institution, a pretrained vision backbone, and twelve weeks. One member is a strong engineer, one
is a competent modeller, one is a health informatics student, and one is the lead. There is no
radiologist on the team. A radiologist has offered two hours per month, informally.

The team's first proposal is an A1 triage model: flag pneumothorax for after-hours escalation.

The proposal is not foolish. The dataset is large, the label is well defined, the clinical need
is real, and the model class is well understood. It fails on constraints that have nothing to do
with the modelling, which is exactly the pattern this topic is designed to expose.

### The G0 review

At G0 the panel asks three questions in sequence.

"Who acts on the output, and when?" The team answers: the on-call radiologist, after hours.
The panel asks whether the team has observed that workflow. It has not, and the radiologist has
not described it.

"What evidence would make that radiologist trust the list?" The team answers: good sensitivity
and a low false-alarm rate. The panel asks whether any radiologist has agreed that those are the
right criteria, at what threshold, and with what capacity. The team has no answer.

"What would you do if the output were wrong?" The team answers that the radiologist would still
read the study. The panel notes that this makes the claim a prioritisation claim, not a
detection claim, and that prioritisation claims require a reader study.

The review takes eleven minutes and establishes that the A1 claim is unsupported before any model
is trained.

### Why A1 fails the constraint check

The decision is clear and clinically meaningful. The data support detection, and the literature
supports the model class. Three constraints fail.

Owner access scores 2 of 5: two hours per month is enough for advice, not for error review, not
for a workflow discussion, and not for a reader study.
Regulatory tractability scores 1 of 5: a time-critical escalation claim is likely a device
function under the FDA clinical decision support guidance, and the team has no regulatory
mentor and no pathway experience.
Evidence feasibility scores 2 of 5: the dataset is single-site, so L3 is impossible; there is no
reader capacity, so the reader-study logic cannot be executed; and the intended use names an
after-hours workflow the team cannot observe.

The weighted score, using weights of 0.20 for decision, 0.20 for owner, 0.20 for data, 0.15 for
evidence, 0.15 for regulatory, and 0.10 for capability, is 2.85.

### The comparison

The team scores five archetypes against the same criteria.

| Archetype | Decision | Owner | Data | Evidence | Regulatory | Capability | Weighted |
|---|---:|---:|---:|---:|---:|---:|---:|
| A1 detection | 5 | 2 | 4 | 2 | 1 | 2 | 2.85 |
| A2 prognosis | 4 | 3 | 4 | 3 | 3 | 4 | 3.50 |
| A3 operations | 4 | 4 | 3 | 3 | 4 | 4 | 3.65 |
| A5 evaluation | 3 | 4 | 5 | 4 | 4 | 5 | 4.10 |
| A8 screening | 3 | 2 | 3 | 2 | 2 | 2 | 2.40 |

A5 wins at 4.10, with A3 second at 3.65. The margin is 0.45, which is large enough that the
ranking is not an artefact of the weights.

### The sensitivity test

The team reruns the ranking with owner access weighted at 0.35 and decision clarity at 0.05,
reflecting an argument that ownership matters more than the elegance of the question.
A5 remains first at 4.25, A3 remains second at 3.65, and A2 falls to 3.35. The ranking is stable.

The stability is informative. It says the decision is not a close call between A1 and A5 that
could be reversed by weighting; it is a consequence of two low scores, owner and regulatory,
that no reasonable weighting can offset.

### The arithmetic of the evidence burden

The team estimates the hours each archetype requires beyond modelling.

A1 with a reader-study logic: reader recruitment 8 hours, protocol 12, two reading sessions with
five readers and 120 cases at 45 seconds per case (5 × 120 × 0.75 minutes = 450 minutes, or 7.5
hours per session, so 15 hours of reading), adjudication 10, analysis 16, regulatory memo 20.
Total 81 hours, plus the modelling and pipeline work. Against a team capacity of 45 hours per
week over twelve weeks, 81 hours is 1.8 weeks of the entire team, and it depends on five
radiologists donating 15 hours.

A5 with a reproducible harness: harness construction 24 hours, contamination screen 8, statistical
comparison 12, subgroup analysis 10, model card 8, reproducibility packaging 12. Total 74 hours,
which is comparable in effort but has no external dependency on five clinicians.

The comparison is the point. A1 and A5 cost similar effort, but A1's effort is not under the
team's control. A team that plans A1 without owning the reader hours has planned a project it
cannot schedule.

### The cost comparison in money

At a blended team rate of 60 dollars per hour, the A1 evidence package is 81 × 60 = 4,860
dollars of team time, plus the donated reader time: 15 hours of radiologist time at an
opportunity cost of about 150 dollars per hour, or 2,250 dollars, which the team does not pay
but the project consumes.
The A5 package is 74 × 60 = 4,440 dollars, with no donated clinical time and no scheduling
dependency.

The difference in direct cost is small, about 420 dollars. The difference in risk is large. The
A1 plan has a critical dependency on five people outside the team, each of whom can withdraw
without penalty, and a regulatory pathway the team cannot execute. The A5 plan has a dependency
on compute and on a public dataset's continued availability, both of which are manageable.

Cost comparisons in capstones should include donated time even when it is unpaid, because the
constraint is availability rather than money.

### Diagnostic one: the owner ratio

The team computes the ratio of committed owner hours to required owner hours. A1 requires about
thirty hours of clinical input across the project: two error-review sessions, one workflow
interview, one reader-study design meeting, and one review of the claim. The radiologist has
committed two hours per month, or six hours over three months. The ratio is 0.2.

A ratio below 0.5 is the first diagnostic that the archetype is unsupported. The project may
still proceed, but the claim must narrow to something the available clinical input can support.

### Diagnostic two: the validation-level gap

The team lists the data: one institution, images from 2017 to 2022, no site identifier beyond the
single source. The claim requires L3 for a new-site triage claim. The gap is one level, and it
cannot be closed by more modelling.

A gap of one level is recoverable by narrowing the claim to L2 and rewriting the intended use to
the single site. A gap of two levels, for example claiming multi-site performance from a
single-site dataset with no temporal holdout, is not recoverable within the project.

### The revision, and one assumption varied

The team selects A5 and defines the capstone as: reproduce a published pneumothorax detection
benchmark on the public split, test whether a pretrained backbone improves on the published
baseline, audit subgroup performance by patient sex, age band, and imaging view, and report
whether the reported advantage survives a contamination check.

The decision the study supports is: whether the informatics group should adopt the pretrained
backbone for future work. The owner is the informatics lead, who has four hours per week. The
regulatory exposure is low because the output is a methods recommendation, not a clinical alert.

Now vary one assumption: a radiologist joins the team for four hours per week.
Owner access rises from 2 to 4 for A1, and evidence feasibility rises from 2 to 3 because a
reader study becomes conceivable. The A1 score becomes 2.85 + 0.2 × 2 + 0.15 × 1 = 3.40.
A5 remains first at 4.10.

The variation shows what would actually change the decision. A clinician alone is not enough;
the team would also need multi-site data and regulatory support. The archetype is a conjunction
of constraints, not a single missing ingredient.

### What the team gained

The team kept the clinical question, narrowed the claim, and produced evidence it could defend.
The dossier's first sentence is not "we built a triage system" but "we tested whether a
pretrained backbone improves pneumothorax detection on a public benchmark, and whether the
improvement survives contamination and subgroup checks".

That sentence is less exciting and more defensible, and it is the kind of sentence a panel
rewards.

## 7. Code, query, or calculation

The block below scores eight archetypes against six weighted criteria, tests the sensitivity of
the ranking, and reports the weakest criterion for each. It uses only the standard library.

```python
weights = {"decision": 0.20, "owner": 0.20, "data": 0.20,
           "evidence": 0.15, "regulatory": 0.15, "capability": 0.10}
scores = {
    "A1 detection":      (5, 2, 4, 2, 1, 2),
    "A2 prognosis":      (4, 3, 4, 3, 3, 4),
    "A3 operations":     (4, 4, 3, 3, 4, 4),
    "A4 documentation":  (3, 2, 3, 3, 2, 3),
    "A5 evaluation":     (3, 4, 5, 4, 4, 5),
    "A6 omics":          (3, 2, 2, 2, 3, 2),
    "A7 trial evidence": (2, 2, 2, 2, 2, 2),
    "A8 screening":      (3, 2, 3, 2, 2, 2),
}
keys = list(weights)

def total(row, w):
    return sum(s * w[k] for s, k in zip(row, keys))

ranked = sorted(((a, total(s, weights)) for a, s in scores.items()),
                key=lambda x: -x[1])
for a, t in ranked:
    print(f"{a:16s} {t:.2f}")
print("best:", ranked[0][0])

w2 = dict(weights)
w2["owner"], w2["decision"] = 0.35, 0.05
ranked2 = sorted(((a, total(s, w2)) for a, s in scores.items()),
                 key=lambda x: -x[1])
print("owner-heavy best:", ranked2[0][0], f"{ranked2[0][1]:.2f}")
for a, t in ranked2[:3]:
    print(f"  {a:16s} {t:.2f}")

for a, s in scores.items():
    i = min(range(len(keys)), key=lambda j: s[j])
    print(f"{a:16s} weakest {keys[i]}={s[i]}")
```

Expected output:

```text
A5 evaluation    4.10
A3 operations    3.65
A2 prognosis     3.50
A1 detection     2.85
A4 documentation 2.65
A8 screening     2.40
A6 omics         2.35
A7 trial evidence 2.00
best: A5 evaluation
owner-heavy best: A5 evaluation 4.25
  A5 evaluation    4.25
  A3 operations    3.65
  A2 prognosis     3.35
A1 detection     weakest regulatory=1
A2 prognosis     weakest owner=3
A3 operations    weakest data=3
A4 documentation weakest owner=2
A5 evaluation    weakest decision=3
A6 omics         weakest owner=2
A7 trial evidence weakest decision=2
A8 screening     weakest owner=2
```

The scoring rubric is a deliberation aid, not a measurement. The scores are judgements, and the
weights encode priorities. A score difference below 0.3 should be treated as a tie and resolved
on the characteristic failure the team can best tolerate.

## 8. Data, standards, and regulatory context

Each archetype sits under a different set of standards. The table below maps them, and the
prose explains what the mapping constrains.

| Archetype | Primary reporting standard | Regulatory exposure | Data standard | Validation evidence |
|---|---|---|---|---|
| A1 detection | STARD-AI, TRIPOD+AI | SaMD, 510(k) or De Novo | DICOM, FHIR ImagingStudy | L3 plus reader study |
| A2 prognosis | TRIPOD+AI, PROBAST | CDS or device depending on output | FHIR, OMOP | L2 plus calibration |
| A3 operations | TRIPOD+AI | Usually not a device | Local schemas, HL7 v2 | Backtest |
| A4 documentation | DECIDE-AI, TRIPOD-LLM | Device if clinical claim | FHIR DocumentReference | Task success and audit |
| A5 evaluation | TRIPOD+AI, benchmark reporting | Not a device | Dataset-specific | Contamination screen |
| A6 omics | TRIPOD+AI, CAGI-style assessment | IVD if clinical test | VCF, HGVS, GA4GH | External cohort |
| A7 trial evidence | ICH E9(R1), CONSORT | Not a device; trial regulation | CDISC SDTM and ADaM | Estimand and bias analysis |
| A8 screening | STARD-AI, equity reporting | Programme-level, device components | Registry, SDOH sources | Equity-weighted L3 |

### A1: detection and triage

The binding constraints are device regulation and reader-study evidence.
Under the FDA clinical decision support guidance, software that analyses a medical image, or
provides a time-critical alert, or gives a specific directive, is likely a device function. A
triage claim therefore implies a submission pathway, and the evidence a reviewer expects includes
site-held-out validation and a reader study designed to show that clinicians perform better with
the output than without it.
STARD-AI, the AI extension of the Standards for Reporting of Diagnostic Accuracy Studies,
provides the reporting structure; TRIPOD+AI covers prediction model reporting. Neither is
binding, but a dossier that omits their items looks incomplete.

DICOM metadata is a leakage risk as well as a data standard. Scanner, site, and acquisition
parameters can encode the label, and a model that learns them will fail at the next site.

### A2: prognostic risk stratification

The binding constraint is calibration. A risk score used for outreach must produce absolute
risks that correspond to observed frequencies; discrimination alone cannot set a threshold.
TRIPOD+AI requires calibration and subgroup reporting, and PROBAST provides a risk-of-bias
assessment that a reviewer may apply.
The regulatory status depends on the output and the user. A score that supports a clinician's
judgement may fall under clinical decision support; a score that directs a specific treatment is
more likely a device.
FHIR and OMOP are the common data models, and the OMOP conventions for standard concepts matter
when the cohort definition depends on a diagnosis code.

### A5: evaluation and benchmark science

The binding constraint is contamination. A benchmark claim is only as good as the assurance that
the test data were not used in pretraining, tuning, or model selection.
The Open Problems in Single-Cell Analysis initiative and the CAGI assessments are examples of
communities that publish held-out truth and control methods (random and true labels) alongside
leaderboards.
There is no device exposure, because the output is a methods recommendation. That absence is
what makes A5 the lowest-burden archetype, and also what makes it easy to do badly: a benchmark
study without a contamination check is a leaderboard report, not evidence.

### A6: translational biomarkers and omics

The binding constraint is replication. A discovery on one cohort is a hypothesis; the evidence
bar is an external cohort with the same phenotype definition and a multiplicity plan.
If the output is a clinical test, in vitro diagnostic regulation applies and the analytical
validation requirements are substantial. Variant naming follows HGVS, genomic data follow VCF,
and exchange follows GA4GH standards where applicable.
CAGI7, the seventh Critical Assessment of Genome Interpretation, ran fourteen challenges in 2025,
including variant effect prediction and polygenic risk scores, with assessment against held-out
experimental and clinical data. The structure is a useful model for what an omics capstone can
and cannot claim.

### A7: trial and regulatory evidence

The binding constraint is the estimand. ICH E9(R1) requires the population, variable, treatment
conditions, intercurrent events, and summary measure to be specified, and a causal claim from a
convenience sample is not admissible evidence for a trial decision.
CDISC SDTM and ADaM define the data structures for submission, and CONSORT defines trial
reporting. A capstone in this archetype is usually a feasibility or precedent analysis, not a
trial, and should say so.

### A8: population and screening

The binding constraint is equity. A screening programme allocates a scarce test or surveillance
slot, and aggregate accuracy can hide harm to a subgroup that is under-represented in the data.
An equity-weighted evaluation requires pre-specified groups, harm weights with a justification,
and a cost per case found.
The regulatory exposure is mixed: the programme is not a device, but the components may be.

### A4: documentation and agentic workflows

The binding constraint is factuality and auditability. A generated discharge summary that omits
a medication, or a reconciliation agent that writes a wrong dose, causes harm that accuracy
metrics do not capture.
DECIDE-AI provides the early live-evaluation structure, including human-computer agreement,
safety and error reporting, and modifications during the study. TRIPOD-LLM, published in 2025,
extends prediction-model reporting to large language models.
FHIR DocumentReference is the data standard for exchanging the note, and the audit trail is the
systems requirement: who wrote what, when, and on which model version.
A claim that the output is "safe" requires an omission analysis and a human review design, not a
fluency score.

### A3: operational forecasting

The binding constraint is the backtest. An operational forecast must be evaluated on historical
periods as if it had been made in real time, because a random split leaks future information
through seasonal patterns and staffing changes.
There is usually no device exposure, but there is an operational claim, and the cost of error is
measured in staff hours and patient flow rather than in diagnostic accuracy.
Data are typically local: scheduling systems, admission-discharge-transfer feeds, and staffing
rosters, often in HL7 v2 or proprietary schemas. A capstone in A3 should document the schema and
the refresh cadence, because both constrain deployment.

### What the standards do not decide

No standard tells a team which archetype to choose. The choice is a judgement about claim,
evidence, and capacity, and the standards only constrain what the chosen archetype must report.

## 9. Industry practice and case evidence

### The same technology, different archetypes

The RSNA challenges illustrate how the archetype determines the evidence.
The 2024 Lumbar Spine Degenerative Classification challenge, with 1,874 teams reported by the
organisers, asked competitors to classify and localise three degenerative conditions across
spinal levels. The task is A1-like in its clinical content, but the competition is A5 in its
evidence: a held-out test set, a metric, and a leaderboard. Winning did not establish a triage
claim, and no organiser suggested it did.

The 2025 Intracranial Aneurysm Detection challenge, with 1,147 teams, used a reference dataset
from 18 sites on five continents annotated by more than 60 radiologists. The first-place solution
reported an area under the curve of 0.915 for aneurysm presence and 0.916 across 13 locations,
using vessel segmentation followed by region-of-interest classification. The multi-site data
would support an L3 evaluation; the competition format still does not constitute a reader study
or a regulatory submission.

The 2026 Knee Abnormality Detection challenge adds a multimodal twist: more than 5,000 knee MRI
exams from 19 sites, paired with reports in about 12 languages, and a macro-averaged area under
the curve across 12 findings. The competition was open through October 2026 and winners were
scheduled to be recognised at RSNA 2026. The archetype for a team building on it would depend on
the claim: a methods comparison is A5, a triage claim is A1, and the second requires evidence the
competition does not provide.

### Where the evidence bar is visible in results

The George B. Moody PhysioNet Challenge 2025 asked teams to prioritise patients for scarce
serological testing for Chagas disease, scored by the fraction of true cases captured in the top
5 percent of the ranking. The organisers report 111 teams and 41 ranked entries. The top test
score was 0.323, and the same models scored about 0.100 on an external cohort from a largely
asymptomatic population.

The gap is archetype evidence in miniature. The competition measures a capacity-constrained
ranking on a hidden test set, which is closer to A2 than to A1, and the external cohort shows
what happens when the deployment population differs from the development population. A team
citing only the top score would be making an A1-style claim from A5-style evidence.

The 2026 PhysioNet Challenge moves to cognitive impairment screening from polysomnography, with
an age-conditioned area under the curve that removes the age shortcut. Conditioning on a
confounder is an archetype decision: the organisers are preventing a model from succeeding
through a variable that would not be available or appropriate in deployment.

### Open Problems and the value of a plain baseline

The Open Problems in Single-Cell Analysis benchmarks, reported in Nature Biotechnology in 2025,
found that a logistic regression on log-normalised principal components was the best-scoring
method in five of eight label-projection datasets, ahead of several neural methods designed for
the task. The result is an A5 finding: a methods comparison under a controlled harness, with
control methods that anchor the scale.

For a capstone team, the lesson is that A5 is not a consolation prize. A well-run A5 study can
produce a finding that changes what a lab adopts, and it can be completed within twelve weeks.

### MedGemma and the prototype boundary

The MedGemma Impact Challenge, announced in January 2026 with winners reported in March 2026,
asked teams to build prototypes on open-weight health models for low-resource and on-device
settings. Winning projects included offline outbreak surveillance and on-device tuberculosis
screening. They are prototypes, not validated products, and the competition did not test clinical
outcomes.

The archetype is A4 or A5 depending on the claim. A prototype that drafts a surveillance signal
is A4; an evaluation of whether the prototype is safe is A5. A team that presents a challenge
prototype as a deployment has confused the two.

### Inference, labelled

It is an inference that teams which choose the archetype before inspecting the data produce more
coherent dossiers. The case evidence supports the components: the evidence bar differs by claim,
and competitions measure benchmark performance rather than clinical utility. The causal claim
about course teams is plausible and untested.

### What transfers from competitions, and what does not

Four things transfer from public challenges to a capstone, whatever the archetype.

A fixed metric forces precision about the claim. A challenge that scores the top 5 percent of a
ranking is making a statement about capacity, and a team that adopts the metric must adopt the
constraint.

A hidden test set forces discipline about selection. The public leaderboard is a development
signal, and the private split is the evidence. A capstone can imitate this with a locked holdout
that no one inspects until the analysis plan is signed.

Multi-site data changes what a claim can say. A model trained and tested on one site cannot
support a statement about another, and the challenge organisers know this, which is why they
hold out sites rather than rows.

Simple baselines remain competitive. The Open Problems results and the challenge write-ups
repeatedly show that a well-implemented standard method is close to the best score, which is a
warning against complexity as a default.

Two things do not transfer. A leaderboard rank is not clinical utility, and a winning solution
is not a validated product. A capstone dossier that cites a rank as evidence of benefit has
confused the two, and a panel will ask which patient outcome improved.

The competition evidence in this section is reported as organisers and winning teams described
it, at the level of team counts, metrics, and described approaches. It is not a controlled
comparison of methods, and no winner is named here as a validated clinical result.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Data-driven archetype choice | The dataset is described before the decision | Convenience drives scope | Claim unsupported by evidence | Write the decision first |
| Claim inflation | "Detects" in slides, "prioritises" in the memo | Ambition outruns evidence | Deduction at defense | One claim sentence, signed by the clinician |
| Missing owner | No clinician at any session | Archetype chosen for prestige | Unadoptable output | Owner-ratio test in week 2 |
| Validation-level gap | Single-site data, multi-site claim | Convenience relabelled as external | Invalid evidence | Record the level and the gap |
| Reader study assumed | "We will ask radiologists later" | No recruitment plan | Evidence bar unmet | Budget reader hours at G0 |
| Benchmark contamination | Suspiciously high public score | Test data in pretraining | Replication failure | Contamination screen |
| Proxy optimisation | Metric improves, workflow does not | Archetype mismatch | No operational value | Tie metric to decision |
| Omics discovery claim | One cohort, no replication | Novelty pressure | Non-reproducible finding | External cohort or narrower claim |
| Causal claim from convenience data | "Reduces mortality" from observational data | Estimand ignored | Invalid inference | Reframe as association or A7 feasibility |
| Aggregate accuracy hides harm | No subgroup table | Equity treated as optional | Widened disparity | Pre-specified equity evaluation |
| Two archetypes at once | Two charters in the repository | Scope undisciplined | Both half-finished | Choose one primary |
| Undisclosed narrowing | Claim changed without a record | Fear of looking weak | Inconsistency at defense | Change log and re-sign |
| Regulatory surprise | "We thought it was research" | Classification not checked | Claim unusable | Triage memo in week 2 |
| Prototype presented as product | Demo works, validation absent | Hackathon habits | Credibility loss | Label the artefact as a prototype |

Two rows deserve emphasis because they are the ones a panel tests first.
Claim inflation is detectable by comparing the verb in the slides with the verb in the intended
use, and the check takes thirty seconds. Missing owner is detectable by asking who would receive
the output on the first day of deployment, and a team that names a role it has never met has not
answered.

The remaining rows are variations on a single mechanism: the team chose an archetype whose
evidence it cannot produce, and then substituted something it could produce for the thing that
was required. The substitution is usually unconscious, which is why the archetype record and the
midpoint review exist.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The decision determines the archetype | The procedure depends on it | Decision is vague or contested | Write the sentence; ask two people to classify it |
| The evidence bar is knowable in week 1 | Planning depends on it | Novel claim with no precedent | Ask a regulatory mentor; use the stricter reading |
| Owner hours can be estimated | Owner-ratio test depends on it | Informal commitments | Ask for a written commitment |
| Data provenance is known | Level selection depends on it | Dataset documentation incomplete | Request the datasheet |
| The team can learn missing skills | Capability scoring assumes it | Skill is central to the claim | Estimate learning hours; reduce scope |
| The twelve-week clock is the binding constraint | Archetype feasibility depends on it | Team has prior artefacts | Reuse prior work explicitly |
| A fallback preserves the question | Contingency planning depends on it | No accessible alternative | Name the fallback in the charter |
| Scores are comparable across criteria | Weighted sum assumes it | Criteria measure different things | Treat small differences as ties |
| Weights reflect programme priorities | Ranking depends on them | Stakeholders disagree | Run the sensitivity test |
| A narrowed claim is still worth defending | Motivation depends on it | Sponsor wants the original claim | Negotiate at G0, not week 11 |

The largest limit is that the archetype taxonomy is a simplification. Real projects combine
shapes: a screening programme with a triage component, a prognosis model embedded in a
documentation workflow. The taxonomy is useful because it forces a primary claim; it is not a
description of how products are built.

A second limit is that the evidence bar described here reflects United States device regulation
and the reporting conventions in common use. A team working under European Union rules, or in a
jurisdiction with a national health technology assessment pathway, will find different artefacts
required. The structure of the argument transfers; the specific documents do not.

A third limit is that the scoring rubric is only as good as the shared understanding of the
criteria. Two team members can score "owner access" as 2 and 4 while agreeing on the facts,
because one weights commitment and the other weights availability. The rubric should define each
anchor before scoring, and the discussion that produces the definitions is often more valuable
than the score.

## 12. Exercises

### Exercise 1. Classify five projects (recall, 25 minutes)

Classify these as A1 to A8 and justify each in one sentence: a model that flags chest X-rays for
radiologist review; a score that selects patients for a home-visit programme; a forecast of
emergency department arrivals; a system that drafts discharge summaries; a benchmark of three
segmentation methods.
**Deliverable:** five classifications with justifications.
**Criteria:** justifications reference the decision, not the method; at least one is ambiguous
and the ambiguity is explained.
**Time:** 25 minutes.

### Exercise 2. Compute an evidence burden (calculation, 45 minutes)

For an A1 triage capstone, estimate the hours for reader recruitment, protocol, reading sessions,
adjudication, analysis, and regulatory memo. Compare with an A5 harness for the same question.
**Deliverable:** a two-column table and a 150-word recommendation.
**Criteria:** reading-session arithmetic shown; external dependencies identified; recommendation
follows from the comparison.
**Time:** 45 minutes.

### Exercise 3. Owner-ratio test (application, 30 minutes)

For your capstone, list the clinical input required by the archetype and the hours committed.
Compute the ratio and decide whether the claim must narrow.
**Deliverable:** a table and a decision.
**Criteria:** required hours are itemised; the ratio is computed; the narrowing is specific.
**Time:** 30 minutes.

### Exercise 4. Score your options (calculation, 40 minutes)

Score three candidate archetypes on the six criteria and compute weighted totals. Run the
sensitivity test with owner weight at 0.35.
**Deliverable:** the table, the code output, and a decision.
**Criteria:** scores justified in one line each; the sensitivity result is interpreted; a tie is
resolved on characteristic failure.
**Time:** 40 minutes.

### Exercise 5. Write the claim sentence (design, 25 minutes)

Write the one-sentence claim your capstone will defend, at the evidence level you can reach, and
a second sentence stating what you are not claiming.
**Deliverable:** two sentences.
**Criteria:** the first uses a verb the evidence supports; the second names the stronger claim
that was excluded.
**Time:** 25 minutes.

### Exercise 6. Design the fallback archetype (design, 35 minutes)

If your primary data source fails, which archetype preserves the scientific question? Write the
fallback claim and the part of the question that is lost.
**Deliverable:** a one-page fallback record.
**Criteria:** the fallback is executable with accessible data; the loss is stated; the switch
date is named.
**Time:** 35 minutes.

### Exercise 7. Critique a claim (critique, 30 minutes)

A team reports an area under the curve of 0.91 on a public benchmark and claims its model "can
support radiologists in triage". Identify the archetype mismatch and rewrite the claim.
**Deliverable:** a 250-word critique and a revised claim.
**Criteria:** names the missing evidence (L3, reader study, regulatory path); the revised claim
is defensible on the reported evidence.
**Time:** 30 minutes.

### Exercise 8. Defend A5 (design, 20 minutes)

Prepare a 90-second argument that an evaluation study is a stronger capstone than an
under-resourced triage claim.
**Deliverable:** a spoken script.
**Criteria:** references evidence sufficiency, not effort; names what A5 can and cannot claim;
avoids disparaging clinical work.
**Time:** 20 minutes.

## 13. Self-check answers

**1. Why does the same model belong to different archetypes in different projects?**
Because the archetype is defined by the claim, not the method. A logistic regression that
reorders a worklist supports a triage claim and sits at A1; the same regression used to compare
two feature sets on a public split supports a methods claim and sits at A5.

**2. What is the evidence bar for A1, and why is it higher than A5?**
A1 requires site-held-out validation, reader-study logic, calibration, subgroup analysis, and a
regulatory artefact, because the claim is that clinicians will act differently and patients will
benefit. A5 requires a reproducible harness, a contamination screen, and a statistical
comparison, because the claim is about method performance under a controlled evaluation.

**3. Compute the reader hours for five readers, 120 cases, 45 seconds per case, two sessions.**
Per session: 5 × 120 × 45 seconds = 27,000 seconds = 450 minutes = 7.5 hours. Two sessions = 15
hours. Add adjudication and analysis, and the clinical time is roughly 25 hours, before
recruitment and scheduling.

**4. Why is the owner ratio a better diagnostic than a letter of support?**
Because a letter records intent and the ratio records availability. A radiologist who supports
the project but has two hours per month cannot review thirty errors, and the claim that depends
on that review is unsupported regardless of enthusiasm.

**5. Why did A1 remain below A5 even when owner access improved?**
Because two other constraints were unchanged: the data were single-site, so L3 was impossible,
and the team had no regulatory pathway support. Improving one constraint does not move the
archetype when the binding constraint is another.

**6. When is a benchmark contamination check unnecessary?**
When the model and the test data have no plausible shared lineage, for example a model trained
from scratch on a dataset collected after the benchmark was published. In practice, with
pretrained backbones and web-scale corpora, the check is almost always necessary.

**7. What does it mean to narrow a claim without losing decision relevance?**
It means changing the verb and the population so that the sentence is supported by the evidence
while still informing a real choice. "Detects pneumothorax for triage" becomes "ranks
pneumothorax cases above non-cases on a held-out split from one institution, which is relevant to
whether the informatics group adopts the backbone". The decision changes, but it remains a
decision.

**8. Why is A7 the highest-burden archetype?**
Because a trial or regulatory decision requires an estimand, bias analysis, precedent, and
usually prospective data, and the claim is causal. A convenience sample cannot support it, and
the evidence cannot be generated inside twelve weeks.

**9. What is the characteristic failure of A8?**
Aggregate accuracy hiding subgroup harm. A screening programme that improves average detection
while missing a subgroup widens disparity, and the aggregate metric will not show it.

**10. Rubric for Exercise 7.** A strong critique identifies the missing L3 validation, the absent
reader study, and the regulatory classification, and rewrites the claim as a benchmark result
relevant to method adoption. A weak critique says the AUC is not high enough, which misses the
point entirely.

**11. Rubric for Exercise 4.** A strong answer shows the scores, the weighted totals, the
sensitivity result, and a decision rule agreed before scoring. A weak answer picks the archetype
first and assigns scores to justify it, which is visible when one criterion moves by two points
between the table and the text.

**12. Why does the archetype record need a signature?**
Because it is the sentence the team will be held to. A record signed by the clinical mentor and
the lead is harder to inflate than a document that lives in a shared folder, and the signature
makes the narrowing a decision rather than a drift.

## 14. Interview questions

**Q1. "What kind of study was your capstone?"**
Strong answer names the archetype and the claim it supports. Weak answer describes the model
architecture.

**Q2. "Why did you not build the triage system?"**
Strong answer explains the evidence bar, the owner constraint, and the decision to narrow. Weak
answer says there was not enough time.

**Q3. "What evidence would you need to make a clinical claim?"**
Strong answer lists site-held-out validation, reader study, calibration, subgroups, and a
regulatory pathway. Weak answer says a better AUC.

**Q4. "How did you decide the validation level?"**
Strong answer derives it from the deployment claim and names what the level does not prove. Weak
answer says it was the data available.

**Q5. "What is benchmark contamination, and how did you check for it?"**
Strong answer describes pretraining overlap, deduplication, and a held-out set with a documented
lineage. Weak answer says the leaderboard was hidden.

**Q6. "What is the difference between analytical and clinical validity?"**
Strong answer distinguishes reproducible measurement from association with outcome, and notes
that neither implies utility. Weak answer treats them as synonyms.

**Q7. "Why is calibration more important than discrimination for a risk score?"**
Strong answer explains that thresholds and outreach depend on absolute risk. Weak answer says
calibration is a nice extra.

**Q8. "How would you have used a clinician if you had one?"**
Strong answer describes error review, label adjudication, workflow mapping, and claim sign-off
with hours. Weak answer says "for domain knowledge".

**Q9. "What can a public challenge result tell you?"**
Strong answer says it measures performance on a specific hidden split under a specific metric.
Weak answer treats a leaderboard rank as clinical evidence.

**Q10. "Which archetype would you choose with twice the time?"**
Strong answer names the constraint that time would relax and the claim that would become
admissible, rather than simply naming a more complex model.

**Q11. "Was your project a research study or a product prototype?"**
Strong answer says which, why, and what the other would have required. Weak answer says "both",
which usually means neither was completed to its own standard.

The pattern across these answers is that a strong candidate can always name the claim, the
evidence, and the gap between them in the same sentence. A weak candidate describes activity,
and the interviewer has to infer the claim, which is the candidate's job.

## 15. Key terms

- **A1 detection or triage** — deciding which case is read, escalated, or recalled.
- **A2 prognostic risk** — deciding who receives an intervention or outreach.
- **A3 operational forecasting** — deciding how to staff, schedule, or allocate.
- **A4 documentation or agentic** — deciding whether a draft is accepted or rejected.
- **A5 evaluation** — deciding which method or model to adopt.
- **A6 biomarker or omics** — deciding which target, variant, or group advances.
- **A7 trial evidence** — deciding whether a trial, endpoint, or strategy is viable.
- **A8 screening** — deciding where to allocate screening or surveillance.
- **Analytical validity** — reproducible and accurate measurement.
- **Backtest** — evaluation of a forecast on historical periods as if real time.
- **Benchmark contamination** — test data present in training or tuning.
- **Clinical utility** — whether use of the output improves decisions or outcomes.
- **Clinical validity** — association between output and clinical outcome.
- **Cost of error** — consequence of each error type.
- **De novo pathway** — FDA route for novel low-to-moderate-risk devices.
- **DECIDE-AI** — reporting guideline for early live clinical evaluation.
- **Estimand** — precise quantity a study estimates.
- **Equity-weighted evaluation** — evaluation weighting groups by need or harm.
- **Evidence bar** — minimum evidence a claim requires.
- **Evidence burden** — effort and time needed to meet the bar.
- **Grader validity** — whether a rubric measures the intended capability.
- **Intervention proximity** — how directly output changes clinical action.
- **Intended use** — population, setting, inputs, output, user, contraindications.
- **MRMC** — multi-reader multi-case study design.
- **Predicate** — legally marketed device for equivalence.
- **PROBAST** — risk-of-bias tool for prediction models.
- **Reader study** — clinicians reading with and without the model.
- **Site-held-out validation** — L3 evaluation on an unseen site.
- **STARD-AI** — reporting guideline for diagnostic accuracy with AI.
- **Temporal validation** — L2 evaluation on a later period.
- **TRIPOD+AI** — reporting guideline for prediction models.
- **Claim verb** — the action the evidence supports.
- **Stepping down** — moving to a lower-burden archetype while keeping the question.
- **Narrowing** — reducing the claim within the same archetype.
- **Inflation** — strengthening the claim without adding evidence.

The vocabulary is small on purpose. A team that can say "we stepped down, not inflated" has the
whole argument in three words, and a panel can check the statement against the record.

Section 5 uses the term evidence burden repeatedly, and the unit is hours of team effort plus
weeks of elapsed time. A burden of 80 hours is not the same as 80 hours of modelling; it is
mostly review, documentation, and waiting for other people, which is why it cannot be compressed
by working faster.

## 16. Sources

All URLs accessed 26 October 2026.

- RSNA, Lumbar Spine Degenerative Classification AI Challenge, 2024.
  https://www.rsna.org/artificial-intelligence/ai-image-challenge/lumbar-spine-degenerative-classification-ai-challenge
- RSNA, Intracranial Aneurysm Detection AI Challenge, 2025.
  https://www.rsna.org/artificial-intelligence/ai-image-challenge/intracranial-aneurysm-detection-ai-challenge
- RSNA, Knee MRI AI Challenge, 2026.
  https://www.rsna.org/artificial-intelligence/ai-image-challenge/knee-mri-ai-challenge
- Reyna MA, Koscova Z, Pavlus J, et al. Detection of Chagas disease from the ECG: the George B.
  Moody PhysioNet Challenge 2025. arXiv:2510.02202. https://arxiv.org/abs/2510.02202
- PhysioNet Challenge 2026, Screening for cognitive impairment during sleep studies.
  https://moody-challenge.physionet.org/2026/
- Luecken MD, Gigante S, Burkhardt DB, et al. Defining and benchmarking open problems in
  single-cell analysis. *Nature Biotechnology* 2025.
  https://doi.org/10.1038/s41587-025-02694-w
- CAGI7 challenges. https://genomeinterpretation.org/challenges.html
- Google Research, MedGemma Impact Challenge winners, March 2026.
  https://blog.google/innovation-and-ai/technology/health/med-gemma-impact-challenge/
- FDA, *Clinical Decision Support Software*, guidance, September 2022.
  https://www.fda.gov/regulatory-information/search-fda-guidance-documents/clinical-decision-support-software
- Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement. *BMJ* 2024;385:e078378.
  https://doi.org/10.1136/bmj-2023-078378
- Sounderajah V, Ashrafian H, Golub RM, et al. Developing a reporting guideline for AI-centred
  diagnostic accuracy studies (STARD-AI). *BMJ Open* 2020;10:e037921.
  https://doi.org/10.1136/bmjopen-2020-037921
- Vasey B, Nagendran M, Campbell B, et al. DECIDE-AI. *Nature Medicine* 2022;28:924-933.
  https://doi.org/10.1038/s41591-022-01772-9
- Wolff RF, Moons KGM, Riley RD, et al. PROBAST. *Annals of Internal Medicine*
  2019;170(1):51-58. https://doi.org/10.7326/M18-1376
- ICH E9(R1), *Addendum on Estimands and Sensitivity Analysis in Clinical Trials*.
  https://www.ich.org/
- CDISC, SDTM and ADaM implementation guides. https://www.cdisc.org/standards/

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic: [`03_team-roles.md`](03_team-roles.md)
- Next topic: [`05_brief-catalogue.md`](05_brief-catalogue.md)
- Related topic: [`07_competition-landscape.md`](07_competition-landscape.md)
- Survey file: [`../../course/19_capstone_hackathons.md`](../../course/19_capstone_hackathons.md)
- Labs: [`../../labs/`](../../labs/)
