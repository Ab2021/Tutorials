# ISO 14971 Risk Management

> Module 14 — Medical Devices, SaMD, and Regulation
> Topic 06 of 10
> Estimated study time: 4 hours
> Prerequisites: 01, 05
> Version 2026.3

## 1. Why this topic exists

Every medical device submission contains an argument that the benefits of the device outweigh its
risks for a defined intended use. The risk management file is where that argument is written down,
evidence by evidence. It is not a form. It is the document a reviewer reads to decide whether the
team understood what could go wrong, whether the controls actually address the causes, and whether
the remaining risk is stated honestly. A weak risk file makes every other piece of evidence look
weaker, because the reviewer cannot tell which failures the testing was designed to catch.

Teams commonly build the risk file at the end. By then the architecture is fixed, the requirements
are frozen, and the hazard list has become a description of the design rather than a challenge to
it. The file lists the failures the design already handles and omits the failures that would require
a redesign. A reviewer notices the pattern quickly: every hazard has a control, every control is
already implemented, no residual risk is uncomfortable, and no requirement traces to a hazard that
the team chose not to mitigate. That file is not evidence of safety. It is evidence of a late start.

The cost of a late start is concrete. A hazard found during design changes a requirement. A hazard
found during verification changes code and repeats tests. A hazard found during a submission review
delays a launch by months. A hazard found after launch becomes a recall, a warning letter, or an
injury. The earlier the hazard enters the file, the cheaper the control. The economics are the same
as defect economics in any engineering discipline, with one difference: the harm is borne by
patients, and the record is public.

ISO 14971, published by the International Organization for Standardization (ISO), is the standard
that structures this work. It is a voluntary international standard, but it becomes effectively
mandatory through several routes: a quality management system that adopts it,
a regulation that presumes conformity when it is applied, a contract that requires it, or a
regulator that recognizes it as a consensus standard. The current edition is ISO 14971:2019, and
ISO/TR 24971:2020 provides guidance on applying it. The standard does not tell a manufacturer what
risk is acceptable. It tells the manufacturer how to decide, document, and defend that judgment.

Data scientists meet this topic with a specific gap. Classic risk management was built around
components that fail at measurable rates: a valve sticks, a battery leaks, a sensor drifts. A model
does not fail in that sense. It produces a plausible output that is wrong for a subgroup, or right
on average and miscalibrated at the threshold that matters, or correct until the input distribution
moves. The probability of harm may not be estimable from bench data at all. The vocabulary of the
standard still applies, but the analysis has to be rebuilt around data, users, and workflows rather
than failure rates. That reconstruction is the skill this topic teaches.

A second gap is organizational. Risk management is not owned by regulatory affairs. It is a
cross-functional activity that consumes clinical input, engineering judgment, usability evidence,
security analysis, and post-market data. When a data scientist hands a model to a regulatory lead
and says "here are the metrics," the risk file receives a performance summary instead of a hazard
analysis. The result is a document that can describe accuracy but cannot explain what happens to a
patient when the accuracy is wrong in a particular way, for a particular person, at a particular
point in the workflow.

This topic gives you the procedure, the vocabulary, and the arithmetic to build a risk file that
survives review. It also teaches you where the standard is silent, because that silence is where
most arguments occur. It is educational material, not legal advice, and it does not replace the
judgment of a qualified regulatory professional or the clinical experts who understand the harm.

## 2. Learning outcomes

By the end of this topic, you can:

1. Construct a risk management plan that states scope, responsibilities, review requirements,
   acceptability criteria, the method for overall residual risk, verification activities, and
   post-production inputs.
2. Distinguish a hazard, a hazardous situation, a sequence of events, and harm, and write a hazard
   chain that a reviewer can follow from cause to patient consequence.
3. Estimate risk when probability is measurable, when it is only qualitative, and when it cannot be
   estimated at all, and state which case applies.
4. Evaluate a risk against pre-defined acceptability criteria and defend the evaluation with
   evidence rather than intuition.
5. Select controls in the ISO 14971 order of priority and justify why a lower-priority control was
   chosen when a higher-priority one was feasible.
6. Critique a risk file for traceability, completeness, and unverified effectiveness claims.
7. Compute residual risk for a hazard with multiple controls and compute the overall residual risk
   across a hazard set, including the assumptions that make the arithmetic valid.
8. Assemble a hazard analysis for a software-only or AI-enabled device, including use error,
   security, and post-deployment performance change.

The outcomes map to the exercises and self-check items as follows.

| Outcome | Exercise | Self-check |
|---|---|---|
| 1 | Exercise 1, Exercise 7 | Questions 1, 7 |
| 2 | Exercise 2 | Question 2 |
| 3 | Exercise 3 | Question 3 |
| 4 | Exercise 4 | Question 4 |
| 5 | Exercise 5 | Question 5 |
| 6 | Exercise 6, Exercise 8 | Question 6 |
| 7 | Exercise 7 | Questions 7, 8 |
| 8 | Exercise 8 | Questions 9, 10 |

## 3. The core mental model

Risk management is a loop around a causal chain. The chain runs from what the device is for, to what
could go wrong, to who could be exposed, to what happens next, to how bad the outcome is. The loop
runs from estimating that outcome, to deciding whether it is acceptable, to changing the device, to
checking whether the change worked, and back to the beginning when new information arrives.

```text
        intended use + reasonably foreseeable misuse
                          |
                          v
        characteristics related to safety
                          |
                          v
        hazard --> hazardous situation --> sequence of events --> harm
                          |
                          v
                 risk estimation
                 (probability x severity)
                          |
                          v
                 risk evaluation  <---- acceptability criteria (from the plan)
                    |           |
                acceptable      not acceptable
                    |           |
                    |           v
                    |     risk control option analysis
                    |     design > protective > information
                    |           |
                    |           v
                    |     implementation + verification of effectiveness
                    |           |
                    |           v
                    |     residual risk evaluation
                    |        |          |
                    |     acceptable   not acceptable --> benefit-risk --> stop if unjustified
                    |           |
                    +-----------+
                          |
                          v
        risks from controls + completeness of control
                          |
                          v
                 overall residual risk
                          |
                          v
                 risk management review
                          |
                          v
        production and post-production information ---> back to hazard identification
```

Three properties of this diagram matter more than the boxes.

First, risk is attached to a sequence, not to a component. The hazard is a potential source of harm.
The hazardous situation is the circumstance in which people are exposed to it. The harm is the
injury. A file that lists "model error" as a hazard and "validation" as a control has skipped the
middle. It cannot explain who was exposed, under what conditions, or why the control interrupts the
sequence. A reviewer will ask for the missing links, and the team will discover that the control
addresses a different link than the one it assumed.

Second, controls are ordered by leverage. A design change that removes the hazard is worth more than
a warning that asks a busy nurse to compensate for it. ISO 14971:2019 clause 7.1 requires the
manufacturer to consider, in order: inherently safe design and manufacture; protective measures in
the device or in the manufacturing process; and information for safety, including training where
appropriate. The order is not a suggestion about style. It reflects the observation that warnings
fail under workload, are missed by new users, and do not travel with the device into every context.
When a team chooses information for safety over a feasible design change, the file must say why the
design change was not practicable, not merely that the warning was cheaper.

Third, the file is an argument, not a list. Every claim in it should be traceable to a requirement,
a test, a clinical input, or a post-market data source. "Residual risk acceptable" is a conclusion.
The file must show the estimate, the criteria, the control, the verification of implementation, the
verification of effectiveness, and the remaining estimate. If one of those links is missing, the
conclusion is unsupported even when it happens to be true.

A useful way to remember the standard's logic is the risk triplet: what can happen, how likely is
it, and how bad is it. The hazard analysis answers the first question, the estimation step answers
the second and third, and the evaluation step compares the answer with a criterion that was written
before the answer was known. That last point is the integrity test for the whole file. If the
criteria were written after the estimates, the file documents a preference, not a judgment.

The loop never really closes. At release, the file transfers its attention from design to
production and post-production. Complaints, service records, literature, and post-market clinical
follow-up become inputs to hazard identification. A hazard that was estimated at one frequency may
turn out to be more frequent. A new use may appear that was not in the intended-use statement. The
standard treats this as part of risk management, not as an afterthought, because a device's risk
profile is a property of its whole life cycle rather than of its release version.

## 4. Definitions and vocabulary

The definitions below follow ISO 14971:2019 unless a row says otherwise. The "common misuse" column
records errors that appear in real files and in review findings. Where the standard's term differs
from everyday usage, the difference is the point.

| Term | Definition | Common misuse | Source |
|---|---|---|---|
| Harm | Injury or damage to the health of people, or damage to property or the environment | Using "harm" to mean any undesirable event, including a near miss | ISO 14971:2019, 3.3 |
| Hazard | Potential source of harm | Listing a failure mode or a software bug instead of the source of harm | ISO 14971:2019, 3.4 |
| Hazardous situation | Circumstance in which people, property, or the environment are exposed to one or more hazards | Treating exposure as automatic once a hazard exists | ISO 14971:2019, 3.5 |
| Risk | Combination of the probability of occurrence of harm and the severity of that harm | Reporting risk as a single number without stating the exposure unit | ISO 14971:2019, 3.6 |
| Severity | Measure of the possible consequences of a hazard | Scoring severity from the frequency of the event | ISO 14971:2019, 3.7 |
| Probability of occurrence of harm | Combination of the probability of occurrence of a hazardous situation and the probability of harm given that situation | Using the device failure rate as the probability of harm | ISO 14971:2019, 3.8 |
| Risk analysis | Systematic use of available information to identify hazards and estimate risk | Calling a brainstorming session a risk analysis without documentation | ISO 14971:2019, 3.9 |
| Risk estimation | Assigning values to the probability and severity of harm | Estimating severity and probability on different populations | ISO 14971:2019, 3.10 |
| Risk evaluation | Comparing estimated risk against given risk criteria to determine acceptability | Evaluating before the criteria are written | ISO 14971:2019, 3.11 |
| Risk control | Process of modifying risk through design, protection, or information | Using "risk mitigation" as if any activity counts | ISO 14971:2019, 3.12 |
| Residual risk | Risk remaining after risk control measures are implemented | Reporting residual risk before verifying that controls exist | ISO 14971:2019, 3.13 |
| Risk management file | Set of records and other documents produced by risk management | Keeping the file as a single spreadsheet with no traceability | ISO 14971:2019, 3.14 |
| Risk management plan | Document that defines scope, responsibilities, criteria, and activities for risk management | Writing the plan after the analysis to match what was done | ISO 14971:2019, 4.4 |
| Risk management report | Summary of the risk management process and its conclusions, prepared at review | Treating the report as a replacement for the file | ISO 14971:2019, 9 |
| Reasonably foreseeable misuse | Use of a product in a way not intended by the manufacturer but which can result from readily predictable human behavior | Calling every misuse "off-label" and excluding it from analysis | ISO 14971:2019, 3.15 |
| Intended use | Use for which a product is intended according to the specifications, instructions, and information provided by the manufacturer | Using a marketing tagline as the intended use | ISO 14971:2019, 3.2 |
| Use error | Act or omission of an act that results in a different medical device response than intended by the manufacturer or expected by the user | Treating use error as user negligence rather than a design input | IEC 62366-1:2015, 3.7 |
| Abnormal use | Intentional use of a device in a way that is outside the manufacturer's specifications | Confusing deliberate abnormal use with foreseeable misuse | IEC 62366-1:2015 |
| Fault | Abnormal condition that can cause a failure | Treating every fault as a hazard | IEC 62304 / general engineering |
| Failure | Termination of the ability of an item to perform a required function | Assuming software fails randomly like hardware | IEC 60050-191 |
| Dangerous failure | Failure that can lead to a hazardous situation | Counting only failures that already caused harm | IEC 61508 vocabulary |
| Single fault condition | Condition in which one means of protection is defective | Assuming a second fault is required before harm | IEC 60601-1 |
| Safety-related characteristic | Design or performance attribute that can affect safety | Listing features instead of characteristics that create hazards | ISO 14971:2019, 5.3 |
| Acceptability criteria | Limits or rules, defined in the plan, for deciding whether a risk is acceptable | Changing criteria after the estimate is known | ISO 14971:2019, 4.4 |
| Overall residual risk | Combination of the residual risks of all hazards, evaluated against the plan's criteria | Taking the maximum single score as the whole answer | ISO 14971:2019, 8 |
| Benefit-risk analysis | Evaluation of whether benefits outweigh residual risk when the risk is not acceptable | Using it to avoid a control that is practicable | ISO 14971:2019, 7.4 |
| ALARP | "As low as reasonably practicable"; a UK-origin regulatory convention for reducing risk until further reduction is grossly disproportionate to the gain | Quoting ALARP as if ISO 14971:2019 defines it | UK HSE; EU practice |
| Production and post-production | Activities that collect and review information after design, including manufacturing, installation, service, and complaints | Treating post-market surveillance as a separate process with no risk-file feedback | ISO 14971:2019, 10 |
| Verification of implementation | Confirmation that a control measure was put in place | Confirming a ticket was closed | ISO 14971:2019, 7.2 |
| Verification of effectiveness | Confirmation that a control measure achieves the intended risk reduction | Confirming the control exists but not that it changes outcomes | ISO 14971:2019, 7.2 |

Three distinctions deserve more than a table row. The first is hazard versus hazardous situation.
A hazard exists whether or not anyone is exposed. A hazardous situation is the moment of exposure.
In an infusion device, an incorrect dose calculation is a hazard; the hazardous situation is a
clinician programming that dose into a pump connected to a patient. The distinction matters because
controls can act at either point. A design that prevents the incorrect calculation attacks the
hazard. A pump that requires confirmation of an unusual dose attacks the hazardous situation. A
file that conflates the two cannot show which link a control breaks.

The second is use error versus reasonably foreseeable misuse. Under IEC 62366-1, a use error is an
act or omission by the user that produces a different response than the manufacturer intended. Under
ISO 14971, reasonably foreseeable misuse is use outside the manufacturer's intent that can result
from readily predictable human behavior. The two overlap, and both belong in the risk file. Calling
a predictable mistake "user error" and closing the item is a way of excluding evidence, not a way of
controlling risk. If nurses routinely press the override because the alert arrives during
handover, the override is foreseeable, and the design that invites it is part of the hazard.

The third is ALARP. ISO 14971:2019 does not define ALARP, and the 2019 revision deliberately moved
away from treating "as low as reasonably practicable" as the standard's own criterion. The phrase
comes from UK health and safety practice and is used widely in European device work. It is useful
because it forces a team to ask whether further risk reduction is feasible and proportionate. It is
dangerous when it becomes a ritual phrase attached to a score with no analysis of what further
reduction would cost and what it would buy. If your file uses ALARP, define what the phrase means
in your plan and show the analysis that supports it.

## 5. The mechanics, step by step

The procedure follows ISO 14971:2019 clauses 4 through 10. Each step names its input, output,
reason, and the branches that most often cause trouble. The steps are sequential in logic but
iterative in practice: a control chosen in step 7 may create a hazard that reopens step 3, and
production data in step 13 may reopen step 5.

### Step 1 — Write the risk management plan

**Input.** Product description, intended-use draft, regulatory strategy, development plan,
applicable standards, and the organization's risk policy. **Output.** A plan defining scope,
life-cycle phases, responsibilities and authorities, review requirements, acceptability criteria,
the method for overall residual risk, verification activities, and post-production inputs.

**Rationale.** The plan is the constitution of the file. Clause 4.4 requires it because
acceptability is meaningless without a criterion defined in advance. The plan fixes the vocabulary:
which severity scale, which probability scale, what counts as a use error, and how overall residual
risk will be judged. If those choices are made case by case, two hazards with the same score can
receive different verdicts and no reviewer can reproduce the reasoning.

**Procedure.** (1) State the device, its versions, and the life-cycle phases in scope. (2) Name the
risk management owner and the approving authority. (3) Define the severity scale with clinical
anchors, not adjectives. (4) Define the probability scale with exposure units and time bases.
(5) State the acceptability matrix and the required action in each band. (6) State how risks are
accepted when probability cannot be estimated. (7) Define the method for overall residual risk.
(8) Define how implementation and effectiveness will be verified. (9) Define post-production inputs
and review cadence. (10) Date and version the plan.

**Branches and edge cases.** A corporate risk policy must be adopted explicitly or departed from
with a recorded justification. A combination product plan must say which risks belong to the device
constituent and which to the drug. A software-only plan should state that failures are systematic
and that probability estimates will often be qualitative. A plan written after development started
must disclose that and explain how earlier decisions were reconstructed, because the document
history makes a back-dated plan visible.

### Step 2 — Define intended use and reasonably foreseeable misuse

**Input.** Intended-use statement, indications for use, labeling draft, user profiles, use
environment, and workflow maps. **Output.** A documented intended use and a structured list of
reasonably foreseeable misuses, each with population, user, setting, and behavior.

**Rationale.** Clause 5.2 requires both. Intended use sets the boundary of the analysis; misuse
sets the part of the boundary a reasonable designer must anticipate. Most serious software hazards
live near that boundary: a model used on a population it was not validated for, a triage tool used
during a mass-casualty event, a pump used with an unconfigured drug library, a decision aid used by
a clinician who has not slept.

**Procedure.** (1) Write the intended use with purpose, user, population, setting, output, and
limitations. (2) List the user's task sequence and the device's role at each step. (3) Ask clinical
staff which shortcuts they take under time pressure. (4) Ask what happens when the device is
unavailable, slow, or wrong. (5) List misuses by category: wrong population, setting, user, input,
timing, or deliberate workaround. (6) Mark each misuse reasonably foreseeable or not, with a reason.
(7) Feed the foreseeable ones into hazard identification.

**Branches and edge cases.** Deliberate misuse intended to cause harm is generally outside scope,
but a workaround users adopt because the device is hard to use is inside it. Customization by a
health system is foreseeable and belongs in the analysis. If labeling excludes a population, the
exclusion must be enforceable in the interface, not only in a document. A claim that a device is
"for adults" does not remove the hazard of pediatric use on a shared workstation.

### Step 3 — Identify safety characteristics, hazards, and hazardous situations

**Input.** Intended use, misuse list, architecture, requirements, clinical literature, complaint
databases, standards, usability findings, and threat model. **Output.** A hazard log with each
hazard, the hazardous situations it can create, and the safety characteristics that influence it.

**Rationale.** Clauses 5.3 and 5.4 require a systematic identification, not a brainstorm. Use
several lenses because each finds different hazards: functional failure, clinical workflow, human
factors, software architecture, data quality, security, and interaction with other devices. A single
lens produces a file that is strong in one area and blind in the rest.

**Procedure.** (1) Decompose the device into functions and interfaces. (2) For each function, ask
what it does, what it should not do, and what it fails to do. (3) For each interface, ask what
happens when data are late, missing, duplicated, malformed, or attributed to the wrong patient.
(4) Walk the workflow and ask where a wrong output changes a decision. (5) Review complaints and
recalls for similar devices. (6) Use a hazard taxonomy such as energy, biological, chemical,
environmental, information, and use. (7) Record each hazard with an identifier used everywhere else.
(8) Record the hazardous situation separately from the hazard.

**Branches and edge cases.** For software, include hazards from correct execution of a wrong
requirement, not only crashes. For artificial intelligence (AI), include training data, output, and
deployment hazards; for in vitro diagnostics (IVDs), specimen handling, reagents, and result
reporting; for connected devices, network loss, timeout, and malicious interference. A hazard that
affects only a subgroup is still a hazard, and the validation set may not see it.

### Step 4 — Build the sequence of events from hazard to harm

**Input.** The hazard log and the clinical workflow. **Output.** For each hazardous situation, a
sequence naming the initiating event, exposure, user or patient behavior, propagation, and harm.

**Rationale.** Clause 5.4 asks for reasonably foreseeable sequences of events. The sequence makes
the risk estimable and the control testable, and it exposes where a control can act. A control that
interrupts the sequence early usually beats one that relies on catching the error at the end.
Without the sequence, a file cannot distinguish a control that prevents harm from one that only
detects it after the fact.

**Procedure.** (1) Start from the hazardous situation. (2) List the events required for exposure to
become harm. (3) Mark which events are device-, user-, patient-, and environment-related. (4)
Identify points where a barrier could intervene. (5) Record the sequence in language a clinician can
review. (6) Check whether the sequence depends on a single user action; if so, consider a second
barrier.

**Branches and edge cases.** A long sequence has a low calculated probability but may still be
common if one step is routine. A sequence that requires a nurse to notice a subtle display change is
weaker than the arithmetic suggests. A sequence that depends on a patient reporting a symptom is
outside the manufacturer's control but still foreseeable. For security, the sequence includes an
adversary's steps, not only a component failure.

### Step 5 — Estimate the probability and severity of harm

**Input.** The sequences, clinical data, validation results, field data, literature, and expert
judgment. **Output.** For each hazardous situation, a severity level and a probability of harm with
its exposure unit, time base, and basis.

**Rationale.** Clause 5.5 requires the probability of the hazardous situation, the probability of
harm given the situation, and the severity. The decomposition matters because a control may affect
one factor and not the other. A design that prevents a wrong dose from being programmed reduces the
probability of the hazardous situation. A monitor that detects the wrong dose before infusion
reduces the probability of harm given the situation. Collapsing both hides which control works.

**Procedure.** (1) Define the exposure unit: per use, per patient-year, per procedure, per device.
(2) Estimate severity from clinical consequences, not frequency. (3) Estimate the probability of the
hazardous situation from data where possible. (4) Estimate the probability of harm given the
situation from clinical evidence or reasoned judgment. (5) Multiply only when the factors are
independent and the units are consistent. (6) Record the source and uncertainty for each factor.
(7) Label judgment-based numbers as judgments and name the expert or rationale.

**Branches and edge cases.** When probability cannot be estimated, the plan's clause 4.4(d)
criterion applies. When data come from a different population or setting, do not transport the
estimate without adjustment. When the device is software-only, hardware failure rates are
meaningless; use qualitative bands and evidence of systematic fault control. When harm is rare and
severe, a small absolute probability can still be unacceptable, and a large denominator can make
the same rate look small.

### Step 6 — Evaluate the estimated risk against the criteria

**Input.** The estimates and the acceptability criteria from the plan. **Output.** For each hazard,
a verdict: acceptable, tolerable with further action, or unacceptable.

**Rationale.** Clause 6 requires the comparison. The verdict is not a feeling about the number; it
is the application of a rule that existed before the estimate. The file often contains several
criteria: a matrix for design risks, a severity-only rule for catastrophic harms, and a special rule
for security. Each verdict must say which one was applied.

**Procedure.** (1) Look up the score in the approved matrix. (2) Check whether a severity-only rule
overrides the matrix. (3) Record the verdict and required action. (4) If acceptable, record why no
further control is needed. (5) If not, proceed to control analysis. (6) Date the entry and identify
the evaluator.

**Branches and edge cases.** A coarse matrix can hide a large reduction: a probability that falls by
a factor of 300 may stay in the same band and produce the same score. Record the numeric estimate
alongside the band and explain the reduction. A hazard unacceptable on severity alone should not be
waved through by a low probability. A hazard with unknown probability should not be scored as low.

### Decision table: risk acceptability

The table is an example of a plan-level rule. The bands and actions must be defined before
estimation and must be consistent with the applicable regulation.

| Score (severity x probability) | Band | Required action | Documentation |
|---:|---|---|---|
| 1–4 | Broadly acceptable | No further control required; monitor | Rationale for no action |
| 5–9 | Tolerable if ALARP | Reduce further where reasonably practicable; monitor | ALARP analysis and residual estimate |
| 10–16 | Unacceptable without justification | Additional control required; if none is practicable, formal benefit-risk at top management | Control analysis, benefit-risk record, disclosed residual risk |
| 17–25 | Unacceptable | Do not release; redesign or restrict intended use | Decision record and claim change |

### Step 7 — Analyze risk control options in priority order

**Input.** The hazards needing control, the architecture, and the state of the art. **Output.** A
selected control or set of controls per hazard, with the options considered and the reason for the
choice.

**Rationale.** Clause 7.1 requires the order: inherently safe design and manufacture; protective
measures in the device or manufacturing process; information for safety, with training where
appropriate. The order reflects effectiveness under real conditions. A design that makes the
hazardous situation impossible does not depend on a user reading a label. A protective measure that
blocks the action does not depend on the user noticing an alert. Information depends on attention,
memory, and workload, all of which fail.

**Procedure.** (1) For each hazard, list at least one design option. (2) List protective options.
(3) List information options. (4) Evaluate feasibility, including whether the design option changes
the clinical function or the regulatory claim. (5) Select the highest-priority feasible option.
(6) If a lower-priority option is selected, document why the higher one was not practicable.
(7) Consider whether the control creates a new hazard. (8) Record the decision.

**Branches and edge cases.** A design option that removes the hazard may also remove a clinical
benefit; that trade-off belongs in the benefit-risk record, not in a silent design choice. A
protective measure may increase alert burden and create a new hazard. Information for safety is
appropriate when the hazard is inherent and the user must make a judgment, but it is weak as the
only control for a high-severity hazard. For software, "inherently safe design" often means
constraining what the software is allowed to output, not checking the output afterward.

### Decision table: control options

| Option | Examples | What it changes | Evidence of effectiveness | Common failure |
|---|---|---|---|---|
| Inherently safe design | Remove the function; constrain output to raise-only; default to the safer state | Eliminates the hazardous situation or the harm mechanism | Requirements test, code review, fault injection | Design removes a needed function and users work around it |
| Protective measure | Hard stop, interlock, independent check, timeout, redundant path | Blocks propagation after the hazardous situation | Integration test, simulated use, failure injection | Alarm fatigue, override, common-cause failure |
| Information for safety | Label, warning, training, contraindication, on-screen rationale | Relies on user detection and correct action | Formative and summative usability testing, comprehension testing | Warning not read, not remembered, or not actionable under load |

### Step 8 — Implement controls and verify implementation and effectiveness

**Input.** The selected controls, the requirements baseline, and the verification plan. **Output.**
Evidence that each control exists, works as intended, and reduces risk as estimated.

**Rationale.** Clause 7.2 requires both verification of implementation and verification of
effectiveness. These are different claims. Implementation evidence says the control was built.
Effectiveness evidence says it changes the probability or severity in the way the residual estimate
assumes. Files commonly contain the first and omit the second, which makes every residual number an
assumption rather than a finding.

**Procedure.** (1) Convert each control into a requirement with an identifier. (2) Link the
requirement to design, code, and test artifacts. (3) Verify implementation by inspection or test.
(4) Verify effectiveness with a test that measures the risk reduction, not only the presence of the
feature. (5) Record the result and any deviation. (6) Update the residual estimate if measured
effectiveness differs from the assumed value. (7) Re-evaluate if the estimate changes.

**Branches and edge cases.** For a hard stop, effectiveness testing means measuring how often users
bypass it and whether the bypass is safe. For an alert, it means measuring detection and response
under realistic workload. For a label, it means comprehension testing, not proofreading. For a
security control, it means adversarial testing, not a configuration checklist.

### Step 9 — Evaluate residual risk

**Input.** The verified controls, the measured effectiveness, and the severity estimate. **Output.**
A residual estimate per hazard and a verdict against the plan's criteria.

**Rationale.** Clause 7.3 requires evaluation after controls. The residual estimate must use the
effectiveness evidence, not the design intent. If a control was assumed to reduce probability by 90
percent and testing shows 40 percent, the residual changes. Severity usually does not change unless
the control changes the harm mechanism, which is rare.

**Procedure.** (1) Recompute the probability with the measured effectiveness. (2) Keep severity at
the original level unless there is evidence that the harm itself is less severe. (3) Compare with
the criteria. (4) If acceptable, record the estimate and the evidence. (5) If not, return to step 7
or proceed to benefit-risk. (6) Record any residual risk that must be disclosed in labeling.

**Branches and edge cases.** Multiple controls on one sequence are not automatically multiplicative.
If they share a cause, such as a busy shift or a network outage, their failures are correlated and
multiplying understates the residual. A control that depends on user attention may fail at the same
moment as another user-dependent control. A disclosure in labeling must be understandable and must
not substitute for a feasible design control. If the residual remains unacceptable and further
control is not practicable, clause 7.4 allows a benefit-risk analysis; it requires clinical evidence
for the benefit, top-management approval, and disclosure of the residual risk.

### Step 10 — Evaluate risks from the controls and check completeness

**Input.** The control set, the architecture, and the hazard log. **Output.** New or increased risks
created by the controls, and a statement that the analysis is complete.

**Rationale.** Clause 7.5 requires consideration of risks arising from control measures. A hard stop
can delay urgent care. A confirmation dialog can train users to click through. A security control
can add login time during resuscitation. Clause 7.6 requires a completeness check: every hazard has
a disposition, every control has a verification, and no part of the device or workflow was omitted.

**Procedure.** (1) For each control, ask what new behavior it creates. (2) Add new hazards with
their own identifiers. (3) Re-estimate and evaluate. (4) Check the log against the architecture, the
workflow, and the use-error list. (5) Check that every requirement traces to a hazard or a design
input. (6) Record the completeness argument and its reviewer.

**Branches and edge cases.** A control that creates a new hazard may still be the right choice, but
the new hazard must be evaluated, not hidden. "The team reviewed the list" is a weak completeness
argument; use a structured prompt such as a hazard taxonomy or an interface checklist. If a control
is removed late, the hazards it addressed must be re-evaluated.

### Step 11 — Evaluate overall residual risk

**Input.** All residual estimates, the plan's method, and the benefit-risk record. **Output.** An
overall residual risk conclusion with supporting analysis.

**Rationale.** Clause 8 requires this evaluation, not only the evaluation of each hazard. A device
can have every individual risk in the tolerable band and still present an unacceptable overall
profile, because small risks accumulate or because hazards share a cause. The plan must define the
method in advance: a sum, a worst-case matrix, a scenario analysis, or structured judgment.

**Procedure.** (1) Assemble the residual estimates with their exposure units. (2) Apply the method
defined in the plan. (3) Consider aggregation and common causes. (4) Consider the population with
the highest burden. (5) Compare with the overall criterion. (6) If not acceptable, revisit controls
or benefits. (7) Record the conclusion and its uncertainty.

**Branches and edge cases.** Summing across hazards assumes independence and a common time base;
state that assumption. A worst-case maximum ignores accumulation, while a sum ignores correlation.
A small per-use risk across many patients can produce a large annual event count, which matters for
post-market planning even when the per-use risk is acceptable.

### Step 12 — Conduct the risk management review

**Input.** The file, the verification evidence, the overall residual conclusion, and the
post-production plan. **Output.** A risk management report and a release decision.

**Rationale.** Clause 9 requires a review before release to confirm that the plan was implemented
and that overall residual risk is acceptable. The review is a gate: the point at which the
organization decides that the file is complete enough to defend.

**Procedure.** (1) Check that every plan element has evidence. (2) Check that every hazard has a
disposition and every control has verification. (3) Check that the overall conclusion follows from
the estimates. (4) Check that labeling discloses what the file says it discloses. (5) Record
reviewers, date, and decision. (6) List open items with owners and due dates.

**Branches and edge cases.** A review can approve release with open items if they are not
safety-relevant and are tracked. A review that finds an unverified control should not approve
release. If the reviewer wrote the file, document why independence was not available.

### Step 13 — Collect and review production and post-production information

**Input.** Manufacturing data, complaints, service records, literature, post-market surveillance,
post-market clinical follow-up, and field safety corrective actions. **Output.** Revisions to the
hazard log, estimates, controls, and report.

**Rationale.** Clause 10 requires a system to collect and review this information and feed it back
into risk management. The risk profile changes when the population changes, users adapt, components
change, or new hazards are reported for similar devices. A file that stops at release is a snapshot,
not a risk management system.

**Procedure.** (1) Define data sources and review cadence in the plan. (2) Screen incoming
information against the hazard log. (3) Open a new hazard or revise an estimate when indicated.
(4) Re-evaluate and control if needed. (5) Update the file and report. (6) Feed trends into design
and labeling changes. (7) Record the review even when nothing changed.

**Branches and edge cases.** A complaint that describes no malfunction may still describe a
hazardous situation. A cluster of usability complaints may be a use-error signal. A change in the
standard of care can make a previously acceptable residual risk unacceptable. A software or model
update is a change requiring risk management, not only a release note.

### Edge case: software-only devices

Software does not wear out, and its failures are systematic. A failure rate per hour is usually a
category error. Analyze the conditions that make a systematic fault more likely to be exercised:
input combinations, timing, state, configuration, interface assumptions, and the requirements
themselves. IEC 62304 assigns a safety class from the possible consequence of failure; the file
should reference that class and show that architecture and verification are consistent with it.

### Edge case: AI and machine learning models

A model adds hazards a conventional analysis misses: subgroup performance variation, calibration
drift with prevalence, reliance on shortcut features, sensitivity to input quality, behavior on
out-of-distribution data, and automation bias in users who trust the number more than their own
assessment. Estimate probability from validation on the intended population, with confidence
intervals, adjusted for the local setting. State which model version the analysis covers and what
triggers re-analysis.

### Edge case: in vitro diagnostics

For an IVD, harm often flows from an incorrect result rather than a malfunction. Cover specimen
collection, handling, and storage; reagent and calibrator variation; interference; limit of
detection; and result reporting. ISO 14971 Annex F gives IVD guidance, and the IVDR adds
requirements. A false negative and a false positive usually have different harm mechanisms and
severities, and should be analyzed separately.

### Edge case: combination products

For a combination product, the device file covers the device constituent and its interaction with
the drug or biologic. Drug pharmacology, dosing, and adverse reactions are generally outside ISO
14971, but the device's contribution to a dosing error is inside it. The plan should define the
boundary with the drug's safety analysis. In the US, 21 CFR Part 4 and the lead center govern the
combined submission.

### Edge case: use error

Use error is not a synonym for incompetence. It is a predictable mismatch between user behavior and
device design. Use the use specification and hazard-related use scenarios from IEC 62366-1, and
treat the interface as part of the risk control system. A file that lists "user did not follow
instructions" as the cause has described the outcome, not analyzed the hazard.

### Edge case: security vulnerabilities

A security vulnerability is a hazard whose realization depends on an adversary. Its probability is
not a clinical frequency per use; it is a function of threat, exposure, and ease of exploitation.
Integrate the threat model, treat exploitation as an initiating event, and estimate severity in
clinical terms. IEC 81001-5-1 and FDA cybersecurity guidance describe the lifecycle and submission
expectations. Scoring security on a per-use probability scale without conversion is dimensionally
incoherent.

### Edge case: probability cannot be estimated

For new technology, rare harms, or software faults, the probability of harm may be unknown. The plan
must define the response: a severity-led rule, a conservative default, a requirement for more data,
or a restriction on the claim. Clause 4.4(d) explicitly requires criteria for this case. Treating an
unknown probability as low is the most common and least defensible error in the file.

## 6. Worked example

The case is constructed. TriageAssist is a hypothetical software as a medical device (SaMD) product.
The numbers, the validation results, and the control effectiveness values are teaching values, not
measurements from a real product or a real study. The arithmetic is real and reproducible with the
script in Section 7.

### 6.1 Product and intended use

TriageAssist is a cloud-connected triage aid used in hospital emergency departments. It reads
structured triage data from the electronic health record (EHR): age, arrival mode, chief complaint,
a five-level triage note category, vital signs, and the active problem list. It displays a suggested
emergency severity index (ESI) level from 1 to 5 and a deterioration-risk flag. A registered nurse
or emergency physician uses the suggestion to assign the patient's queue priority. The product does
not order tests, recommend treatment, or remove a patient from clinician review.

The intended-use statement reads:

"TriageAssist is intended for use by registered nurses and emergency physicians to support the
assignment of emergency severity index acuity levels for patients aged 18 years and older presenting
to hospital emergency departments. The software analyzes structured triage data and displays a
suggested acuity level and a deterioration-risk flag. The output is an adjunct to, and not a
replacement for, nurse assessment. TriageAssist is not intended for patients in cardiac arrest,
patients requiring immediate resuscitation, patients under 18 years of age, or patients whose
triage data are incomplete."

Reasonably foreseeable misuse includes: starting triage before vital signs are recorded; using the
product in a pediatric area; treating the suggestion as the triage decision; continuing to use the
queue when the interface is unavailable; and entering free text into structured fields to save time.

### 6.2 The hazard chain

The central hazard is an under-triage suggestion. The chain is:

```text
H1  Model suggests a lower acuity level than the patient's condition warrants
      |
      v
HS1 A high-acuity patient is placed in a lower-priority queue
      |
      v
E1  Nurse accepts the suggestion without independent reassessment
      |
      v
E2  Reassessment is delayed beyond the clinically appropriate interval
      |
      v
E3  Deterioration progresses without treatment
      |
      v
HARM  Permanent injury, emergency intervention, or death
```

The sequence has four probability factors. The first is the probability that the model's suggestion
is discordant with the patient's true acuity, given that the patient is high acuity. The second is
the probability that the nurse accepts the discordant suggestion. The third is the probability that
no reassessment occurs within the safe interval. The fourth is the probability of serious harm
given that reassessment is delayed. The base-case values are 0.10, 0.25, 0.20, and 0.10, which
multiply to 5.0 x 10^-4 per high-acuity encounter. Each value is a constructed judgment for this
example; in a real file each would carry a source, a confidence interval, and a sensitivity range.

Severity is scored 4 on a five-point scale anchored as follows: 1 negligible, 2 minor, 3 serious,
4 critical (permanent impairment or life-threatening deterioration requiring urgent intervention),
5 catastrophic (death). H1 is scored 4 because the typical serious outcome is delayed treatment of
a time-critical condition, not inevitable death. If the analysis concluded that death is a
reasonably foreseeable outcome of the same chain, severity would be 5 and every downstream score
would rise.

### 6.3 Hazard set and estimates

| ID | Hazard | Severity | Initial P | Basis for the estimate |
|---|---|---:|---:|---|
| H1 | High-acuity patient under-triaged; assessment delayed | 4 | 5.0e-4 | Four-factor chain; validation sensitivity at the deployed threshold |
| H2 | Alert fatigue delays care for other patients | 2 | 8.0e-3 | Observed alert volume and queue delay in simulated use |
| H3 | Degraded performance for patients with incomplete history | 4 | 6.0e-4 | Subgroup analysis; missing-data rate in the intended setting |
| H4 | Discrimination degrades after deployment | 4 | 2.0e-3 | Drift simulation and published experience with similar models |
| H5 | Threshold or model tampered with by an attacker | 4 | 1.0e-3 | Threat model; per-encounter exposure to a compromised configuration |
| H6 | Output not delivered to the EHR queue | 4 | 4.0e-4 | Interface failure log from integration testing |
| H7 | Triage started before vital signs are complete | 3 | 2.0e-2 | Workflow observation and usability findings |

The H5 row contains a deliberate methodological warning. The probability is expressed per encounter
for comparability with the other rows, but the underlying estimate is a threat likelihood over
time, not a clinical frequency. Converting between the two requires an assumption about exposure.
The file must state that assumption. If it does not, the score looks precise while being
dimensionally incoherent.

### 6.4 Controls

The controls follow the ISO 14971 order of priority. H1 receives a design control first: the model
may only raise the acuity suggestion relative to the nurse's provisional assignment, never lower it.
A lower suggestion is displayed as information but does not change the queue. This removes the
under-triage mechanism rather than warning about it. A protective measure adds a 30-minute
reassessment timer for any patient whose assigned acuity is above the model's suggestion, and a
second protective measure requires a documented reason to dismiss an upgrade suggestion.

| ID | Control | Type | Assumed effectiveness | Verification |
|---|---|---|---:|---|
| C1 | Raise-only output; no automatic downgrade | Design | 0.92 | Requirements test, code review, fault injection |
| C2 | 30-minute reassessment timer | Protective | 0.90 | Integration test, simulated shift |
| C3 | Documented reason to dismiss upgrade | Protective | 0.60 | Usability test with 15 nurses |
| C4 | Tiered alerting with daily prompt budget | Protective | 0.90 | Simulated use, alert count audit |
| C5 | Rate limit per clinician per hour | Protective | 0.90 | Configuration test |
| C6 | Consolidated triage display | Design | 0.75 | Formative usability test |
| C7 | Subgroup performance requirement in validation | Design | 0.70 | Pre-specified subgroup analysis |
| C8 | Input completeness gate before scoring | Protective | 0.50 | Integration test |
| C9 | Intended-use exclusion when data are incomplete | Design | 0.95 | Requirements test, labeling review |
| C10 | Drift monitoring with rollback | Protective | 0.95 | Monitoring simulation, rollback drill |
| C11 | Version pinning and signed artifacts | Protective | 0.90 | Configuration audit |
| C12 | Quarterly recalibration under change control | Protective | 0.70 | Change record, recalibration test |
| C13 | Signed model and configuration artifacts | Protective | 0.90 | Security test |
| C14 | Role-based access with multi-factor authentication | Protective | 0.85 | Penetration test |
| C15 | Runtime integrity monitoring | Protective | 0.60 | Adversarial test |
| C16 | Message acknowledgement with heartbeat | Protective | 0.95 | Interface failure injection |
| C17 | Fallback to manual triage on delivery failure | Protective | 0.80 | Failover test |
| C18 | Vitals-complete gate on the scoring action | Design | 0.95 | Integration test |
| C19 | Blocking dialog with explicit override | Protective | 0.80 | Usability test |
| C20 | Training and quick-reference card | Information | 0.60 | Comprehension test |

The effectiveness values are assumptions. That is the point of the next table: a residual estimate
is only as good as the evidence behind the effectiveness number. In a real file, C1's 0.92 would
need a test that measures how often a downgrade reaches the queue under fault injection, and C19's
0.80 would need a usability study that measures override behavior under time pressure.

### 6.5 Residual risk

Applying the effectiveness values multiplicatively gives the residual estimates in the table. The
arithmetic assumes the controls are independent. For H1, that assumption is questionable: C1, C2,
and C3 all depend on the nurse's workflow during a busy shift, so a single period of overcrowding
can degrade all three at once. The file must state this and treat the residual as optimistic.

| ID | Initial score | Residual P | Residual score | Verdict |
|---|---:|---:|---:|---|
| H1 | 16 | 1.6e-6 | 8 | Tolerable if ALARP |
| H2 | 10 | 2.0e-5 | 6 | Tolerable if ALARP |
| H3 | 16 | 4.5e-6 | 8 | Tolerable if ALARP |
| H4 | 20 | 3.0e-6 | 8 | Tolerable if ALARP |
| H5 | 20 | 6.0e-6 | 8 | Tolerable if ALARP |
| H6 | 16 | 4.0e-6 | 8 | Tolerable if ALARP |
| H7 | 15 | 8.0e-5 | 9 | Tolerable if ALARP |

Every hazard moves from the unacceptable or justification band into the tolerable band. That is a
good outcome, and it is also where files become careless. The overall residual risk is not the
maximum of the rows. Summing the per-encounter residual probabilities gives 1.19 x 10^-4. At 45,000
encounters per year, that is about 5.4 harm events per year, assuming independence and a constant
rate. The number is constructed, but the method is the one a real file must use: state the exposure
base, state the independence assumption, and show the annualized consequence.

The overall residual risk verdict is "tolerable if ALARP" with a benefit-risk justification and
disclosure. The justification rests on the clinical benefit of faster recognition of deteriorating
patients and on the fact that the raise-only design removes the most severe failure path. The
disclosure states that the device does not replace nurse assessment and that the queue must be
reassessed at defined intervals.

### 6.6 Two findings that would sink the file

**Finding 1: the probability estimates are not dimensionally consistent.** H5 scores the
probability of a security-related harm per encounter using a threat likelihood estimated over a
year. The file then compares H5 with clinical hazards on the same matrix. A reviewer can show that
the score changes by orders of magnitude depending on the exposure assumption, and that the
effectiveness values for the security controls were never converted to the same basis. The finding
does not mean the product is unsafe. It means the file cannot support the claim that H5 is
tolerable, because the number that supports the claim does not mean what the matrix says it means.
The fix is to estimate security risk in its own units, convert with a stated exposure model, and
show the sensitivity of the verdict to that model.

**Finding 2: the controls are verified for implementation but not for effectiveness.** The file
contains tickets, code reviews, and test cases showing that C1, C2, and C3 exist. It does not
contain a test showing that C1 prevents a downgrade from reaching the queue under realistic fault
conditions, that C2 produces reassessment within 30 minutes on a busy shift, or that C3 changes
nurse behavior rather than producing a reflex click. ISO 14971:2019 clause 7.2 requires both
verification of implementation and verification of effectiveness. Without effectiveness evidence,
the residual calculation is an assumption multiplied by an assumption. The finding is fatal
because every downstream conclusion depends on those numbers, and the reviewer cannot distinguish a
well-controlled device from a well-documented one.

A third problem would compound both: the residual calculation assumes independence among controls
that share the nurse's workload as a common cause. If the file does not discuss common-cause
failure, the reviewer will. The correct response is not to multiply harder; it is to add a control
that does not depend on the same human and operational conditions, such as an automatic
re-triage triggered by a change in vital signs.

### 6.7 Vary one assumption: local probability data

The base case uses the vendor's validation data. Suppose the hospital's local validation finds a
higher proportion of high-acuity presentations and a lower sensitivity at the deployed threshold.
The discordance factor rises from 0.10 to 0.16, the acceptance factor from 0.25 to 0.40 because
the department is crowded, the no-reassessment factor from 0.20 to 0.25, and the harm-given-delay
factor from 0.10 to 0.20. The product is 0.16 x 0.40 x 0.25 x 0.20 = 3.2 x 10^-3 per high-acuity
encounter, 6.4 times the base case.

With the same controls, the residual becomes 3.2 x 10^-3 x 0.08 x 0.10 x 0.40 = 1.024 x 10^-5.
That is band 3 on the probability scale, and the residual score becomes 4 x 3 = 12. The verdict
changes from "tolerable if ALARP" to "unacceptable without justification." The file can no longer
close with the same control set.

The decision changes in three ways. First, the team must add a control that does not rely on the
same workflow, such as an automatic re-triage when a vital-sign threshold is crossed or a hard stop
that prevents a downgrade from being recorded without a physician review. Second, if no control can
reduce the residual below the threshold, the team must restrict the intended use to departments
that have a rapid-response protocol and a maximum reassessment interval. Third, the team must
re-examine the overall residual risk, because H1 was not the only hazard whose probability depends
on crowding; H2 and H7 would move in the same direction.

The variation is the lesson. The same code, the same controls, and the same matrix produce a
different regulatory conclusion when the population and the workflow change. A risk file that
reports one set of numbers without a sensitivity analysis is asserting a conclusion that holds only
in the setting where the numbers were measured. That is why the plan requires a method for
transporting estimates and why post-market data must feed back into the file.

## 7. Code, query, or calculation

The script below implements the scoring rule, the probability bands, the residual calculation, and
the overall aggregation used in Section 6. It is deliberately small enough to audit. It is teaching
code, not validated software, and it does not replace a controlled risk management tool.

```python
"""ISO 14971 risk scoring and residual-risk arithmetic (teaching example).

Every severity, probability, and effectiveness value below is constructed for
the TriageAssist example in Section 6. None of it comes from a real product.
"""

from __future__ import annotations

# Probability of harm per use; lower bound inclusive, upper bound exclusive.
BANDS = [
    (1, 0.0, 1e-6, "improbable"),
    (2, 1e-6, 1e-5, "remote"),
    (3, 1e-5, 1e-4, "occasional"),
    (4, 1e-4, 1e-3, "probable"),
    (5, 1e-3, float("inf"), "frequent"),
]

# score ceiling -> verdict
VERDICTS = [(4, "acceptable"), (9, "tolerable if ALARP"),
            (16, "unacceptable without justification"), (25, "unacceptable")]

# (id, severity, initial probability, [(control type, effectiveness), ...])
HAZARDS = [
    ("H1", 4, 5.0e-4, [(1, 0.92), (2, 0.90), (2, 0.60)]),
    ("H2", 2, 8.0e-3, [(2, 0.90), (2, 0.90), (1, 0.75)]),
    ("H3", 4, 6.0e-4, [(1, 0.70), (2, 0.50), (1, 0.95)]),
    ("H4", 4, 2.0e-3, [(2, 0.95), (2, 0.90), (2, 0.70)]),
    ("H5", 4, 1.0e-3, [(2, 0.90), (2, 0.85), (2, 0.60)]),
    ("H6", 4, 4.0e-4, [(2, 0.95), (2, 0.80)]),
    ("H7", 3, 2.0e-2, [(1, 0.95), (2, 0.80), (3, 0.60)]),
]

EXPOSURES_PER_YEAR = 45_000


def band(p: float) -> int:
    """Return the probability band for a per-use probability of harm."""
    if p <= 0:
        raise ValueError("probability must be positive")
    for number, low, high, _label in BANDS:
        if low <= p < high:
            return number
    raise ValueError(p)


def band_label(number: int) -> str:
    return next(label for n, _lo, _hi, label in BANDS if n == number)


def verdict(score: int) -> str:
    return next(text for ceiling, text in VERDICTS if score <= ceiling)


def residual(p: float, controls: list[tuple[int, float]]) -> float:
    """Multiply independent barrier failure probabilities.

    This assumes the barriers are independent. When they share a cause, the
    product understates the residual probability.
    """
    for _kind, effectiveness in controls:
        if not 0.0 <= effectiveness < 1.0:
            raise ValueError(effectiveness)
        p *= 1.0 - effectiveness
    return p


def hierarchy_ok(controls: list[tuple[int, float]]) -> bool:
    """A high risk needs at least one design or protective control."""
    material = {kind for kind, effectiveness in controls if effectiveness >= 0.5}
    return 1 in material or 2 in material


def main() -> None:
    print("ISO 14971 risk table: TriageAssist (constructed example)")
    print()
    print(f"{'ID':<3s} {'S':>2s} {'P0':>8s} {'B':>2s} {'SxP':>4s} "
          f"{'P_res':>9s} {'B':>2s} {'SxP':>4s}  Verdict")
    print("-" * 64)
    for hid, severity, p0, controls in HAZARDS:
        pr = residual(p0, controls)
        print(f"{hid:<3s} {severity:>2d} {p0:>8.1e} {band(p0):>2d} "
              f"{severity * band(p0):>4d} {pr:>9.1e} {band(pr):>2d} "
              f"{severity * band(pr):>4d}  {verdict(severity * band(pr))}")

    total = sum(residual(p0, c) for _hid, _s, p0, c in HAZARDS)
    worst = max(severity * band(residual(p0, c)) for _hid, severity, p0, c in HAZARDS)
    missing = [hid for hid, _s, _p, c in HAZARDS if not hierarchy_ok(c)]
    print()
    print(f"Overall residual: worst score {worst} ({verdict(worst)}), "
          f"summed probability {total:.2e} per encounter")
    print(f"Expected harm events per year at {EXPOSURES_PER_YEAR:,} encounters: "
          f"{total * EXPOSURES_PER_YEAR:.1f}")
    print(f"Hazards without a design or protective control: "
          f"{', '.join(missing) if missing else 'none'}")

    h1_controls = HAZARDS[0][3]
    local_p0 = 3.2e-3
    local_pr = residual(local_p0, h1_controls)
    print()
    print("Variation: local data, H1 initial probability 3.2e-03")
    print(f"H1 initial score {HAZARDS[0][1] * band(local_p0)} "
          f"({band(local_p0)}, {band_label(band(local_p0))})")
    print(f"H1 residual probability {local_pr:.3e} "
          f"({band(local_pr)}, {band_label(band(local_pr))}), "
          f"score {HAZARDS[0][1] * band(local_pr)} "
          f"({verdict(HAZARDS[0][1] * band(local_pr))})")


if __name__ == "__main__":
    main()
```

Output from a Python 3.11 run (`python -I script.py`):

```text
ISO 14971 risk table: TriageAssist (constructed example)

ID   S       P0  B  SxP     P_res  B  SxP  Verdict
----------------------------------------------------------------
H1   4  5.0e-04  4   16   1.6e-06  2    8  tolerable if ALARP
H2   2  8.0e-03  5   10   2.0e-05  3    6  tolerable if ALARP
H3   4  6.0e-04  4   16   4.5e-06  2    8  tolerable if ALARP
H4   4  2.0e-03  5   20   3.0e-06  2    8  tolerable if ALARP
H5   4  1.0e-03  5   20   6.0e-06  2    8  tolerable if ALARP
H6   4  4.0e-04  4   16   4.0e-06  2    8  tolerable if ALARP
H7   3  2.0e-02  5   15   8.0e-05  3    9  tolerable if ALARP

Overall residual: worst score 9 (tolerable if ALARP), summed probability 1.19e-04 per encounter
Expected harm events per year at 45,000 encounters: 5.4
Hazards without a design or protective control: none

Variation: local data, H1 initial probability 3.2e-03
H1 initial score 20 (5, frequent)
H1 residual probability 1.024e-05 (3, occasional), score 12 (unacceptable without justification)
```

Four cautions apply. First, the bands are a convention. A different organization can choose
different cut points, and the verdict can change without any change in the underlying risk. That is
why the plan must fix the bands before estimation and why a file should not compare scores across
products that used different scales.

Second, the residual calculation multiplies failure probabilities. Multiplication assumes that the
barriers fail independently. When controls share a cause, such as a busy shift, a network outage, or
a single training program, the true residual is higher than the product. The script cannot detect
that; the analyst must.

Third, the probability is per encounter, while the annual figure multiplies by an assumed volume.
The annual figure is useful for post-market planning, not for the acceptability decision, which is
usually made per use. Mixing the two bases in one table is a common error.

Fourth, the script has no branch for an unknown probability. A real implementation needs one:
a severity-led rule, a conservative default, or an explicit refusal to score. The absence of that
branch in teaching code is a reminder that a matrix is not a risk management system.

## 8. Data, standards, and regulatory context

> **Note:** This section is educational and is not legal advice. Risk acceptability is a
> manufacturer's judgment constrained by law, and the applicable requirements depend on the
> jurisdiction, the device class, and the current versions of the cited documents. Confirm the
> current text and the current regulatory expectations with qualified professionals before relying
> on any statement here.

The table separates binding requirements from standards and guidance. A standard becomes binding
when a regulation incorporates it, when a contract requires it, or when the manufacturer adopts it
in its quality system. Guidance describes an agency's current thinking; it is not a statute, but a
submission that ignores it without an alternative rationale invites questions. Convention, such as a
company template or a reviewer's preferred format, creates no legal obligation but shapes what
reviewers expect.

| Source | Status | What it constrains | What it does not do |
|---|---|---|---|
| ISO 14971:2019, 3rd edition, 10 December 2019 | Voluntary international standard | Risk management process, plan, file, analysis, evaluation, control, residual and overall residual risk, review, production and post-production | Does not set acceptable risk, prescribe a matrix, or require a particular analysis technique |
| ISO/TR 24971:2020 | Guidance on ISO 14971 | Practical guidance on planning, hazard identification, risk estimation, and control | Is not a requirement document and does not create obligations by itself |
| ISO 13485:2016 | Voluntary QMS standard | Risk management throughout product realization; design controls; corrective action | Does not define the risk management process in detail |
| US Food and Drug Administration (FDA) QMSR, effective 2 February 2026 | Binding US regulation | Incorporates ISO 13485:2016 into the quality system requirements | Does not itself incorporate ISO 14971 clause by clause |
| FDA recognition of ISO 14971:2019 | Consensus standard recognition | Supports a declaration of conformity in premarket review | Recognition is not an approval and can change; check the database |
| EU Medical Device Regulation (MDR) 2017/745, Annex I, General Safety and Performance Requirements (GSPR) 1–4 | Binding EU regulation | Risk management system, risk reduction as far as possible, residual risk, information to users | Does not prescribe a matrix or a specific file format |
| EN ISO 14971:2019/A11:2021 | Harmonized standard | Presumption of conformity with the MDR risk requirements when applied | The national annexes constrain benefit-risk arguments; verify the current citation |
| International Electrotechnical Commission (IEC) 62304:2006+A1:2015 | Voluntary software lifecycle standard | Software safety class, development, verification, maintenance | Does not set the device risk acceptability criteria |
| IEC 62366-1:2015+A1:2020 | Voluntary usability standard | Use specification, hazard-related use scenarios, formative and summative evaluation | Does not replace the risk management process |
| IEC 81001-5-1:2021 | Voluntary security lifecycle standard | Security activities across the health software lifecycle | Does not set clinical severity criteria |
| ISO/TR 20416:2020 | Guidance on post-market surveillance | PMS process and inputs for manufacturers | Is not binding and does not replace MDR or FDA obligations |
| ISO 14155:2020 | Clinical investigation standard | Good clinical practice for device trials | Does not define risk acceptability |
| National Institute of Standards and Technology (NIST) AI Risk Management Framework 1.0 | Voluntary framework | AI risk governance, mapping, measurement, management | Is not device regulation and does not replace ISO 14971 |
| EU AI Act 2024/1689 | Binding EU regulation | High-risk AI obligations, including risk management, data governance, and monitoring | Does not replace MDR/IVDR conformity assessment; timing of obligations must be verified |

In the United States, the risk management expectation arrives through several doors. The quality
system regulation requires risk-based design controls and corrective action. The QMSR, effective 2
February 2026, incorporates ISO 13485:2016, which requires a risk-based approach throughout product
realization. FDA recognizes ISO 14971:2019 as a consensus standard, so a manufacturer can cite it in
a declaration of conformity and use it to support premarket review. FDA software and cybersecurity
guidance describe the content FDA expects to see, including hazard analysis, threat modeling, and
verification of security controls. None of these documents tells a manufacturer that a particular
severity or probability is acceptable.

In the European Union, the MDR requires a risk management system and sets the general safety and
performance requirements in Annex I. The first four requirements cover risk reduction as far as
possible, the relationship between risk reduction and benefit-risk, the risk management system, and
risk control measures. The harmonized edition of ISO 14971 carries a presumption of conformity, but
its national annexes address the differences between the standard's logic and the MDR's wording,
particularly around the use of benefit-risk analysis to accept a risk that could be reduced further.
The practical consequence is that a file written only to the ISO text may need an additional
argument for the EU. Cybersecurity expectations come from guidance issued by the Medical Device Coordination Group
(MDCG) and from the MDR's general requirements.

ISO/TR 24971:2020 is the most useful companion document for a team building its first file. It
gives guidance on the plan, on hazard identification, on the relationship between risk and safety,
and on the analysis of use-related hazards. It is guidance, not a checklist, and it does not
remove the need to think about the specific device.

Three things ISO 14971 does not do deserve emphasis. It does not define acceptable risk; that is a
regulatory and societal judgment expressed through the plan's criteria and the applicable law. It
does not require a particular technique; failure mode and effects analysis (FMEA), fault tree
analysis, hazard and operability study (HAZOP), and scenario analysis are all acceptable if they fit
the device. It does not cover business risk, regulatory risk, or reputational risk, except where
those risks translate into patient harm.

The data sources for a risk file are as important as the standards. Complaint databases, recall
databases, post-market surveillance reports, clinical literature, usability studies, and validation
results all supply estimates. A file that cites only internal opinion is reproducible but not
evidence-based. A file that cites a vendor's marketing claim as a probability source is neither.

## 9. Industry practice and case evidence

Mature device organizations treat the risk file as a living traceability matrix. Every hazard has an
identifier, every control has a requirement, every requirement has a test, and every test result
feeds back into the residual estimate. The file is reviewed at design gates, not assembled at the
end. The risk management owner has authority to block a release, and the review includes clinical,
engineering, quality, and regulatory representatives. This is documented practice in quality
systems that follow ISO 13485; the specific tooling varies.

A second practice is to separate the hazard analysis from the failure analysis. A failure mode and
effects analysis asks how a component can fail. A hazard analysis asks how a patient can be harmed.
The two overlap, but a file built only from FMEA tends to miss use error, interface hazards, and
hazards that arise from correct operation of a wrong requirement. Mature teams run both and
reconcile them.

A third practice is to treat post-market data as an input to the hazard log, not only as a
regulatory reporting obligation. Complaint trends, service records, and literature are screened
against existing hazards and used to open new ones. The feedback loop is the difference between a
risk management system and a release document.

The public cases below are documented facts unless the text labels them as inference. They are
selected because each illustrates a different failure of hazard identification or control.

**Therac-25.** A family of medical linear accelerators produced by Atomic Energy of Canada Limited
was involved in a series of radiation overdoses in the mid-1980s. Leveson and Turner's 1993
investigation documented the role of software defects, the removal of hardware interlocks that had
existed in earlier models, and incident analyses that initially attributed harm to operator error.
The documented lesson is that a software control can remove a physical safeguard and that a risk
analysis which assumes the software is correct will miss the resulting hazard. The inference, not
stated by the authors as a regulatory conclusion, is that the safety case depended on a single
software path with no independent barrier.

**Infusion pump programming errors.** FDA's Infusion Pump Improvement Initiative, launched in 2010,
described a pattern of use-related hazards in infusion pumps, including programming errors, alarm
failures, and inconsistent user interfaces. The documented lesson is that use error is a design
hazard, not a user defect. The inference is that a risk file which lists "user programming error"
without analyzing the interface has not identified the hazard.

**Hospira Symbiq.** FDA issued a safety communication in 2015 about cybersecurity vulnerabilities in
the Symbiq infusion pump, and the product was subsequently discontinued. The documented lesson is
that a security vulnerability can be a device hazard with clinical consequences, not only an
information-technology problem. The inference is that security risks need a place in the hazard log
and a severity estimate in clinical terms.

**Abbott (St. Jude Medical) pacemakers.** In 2017, FDA issued a safety communication about
cybersecurity vulnerabilities in certain implantable pacemakers and required a firmware update to
reduce the risk of unauthorized access. The documented lesson is that post-market discovery of a
security hazard can require a field action even when the clinical performance of the device has not
changed. The inference is that the risk file must anticipate adversarial, not only accidental,
exposure.

**Philips Respironics sleep and respiratory devices.** In 2021, FDA announced a recall of certain
continuous positive airway pressure (CPAP) and bi-level positive airway pressure devices because
polyester-based polyurethane sound abatement foam could degrade and be inhaled or ingested. The
documented lesson is that a material hazard can emerge over years of use and cleaning, and that
hazard identification must consider the full life cycle, not only the first use. The inference is
that long-term material interaction belongs in the risk analysis even when the acute performance is
acceptable.

**Epic Sepsis Model.** Wong et al. published an external validation in JAMA Internal Medicine in
2021. The study reported that discrimination in an external health system was substantially lower
than the values reported by the vendor, and that many alerts would be generated at the deployed
threshold. The documented lesson is that performance measured in one setting does not automatically
transport to another. The inference is that a risk file which uses vendor performance as the
probability estimate without local validation is relying on an assumption that post-market data can
falsify.

**IDx-DR.** FDA granted De Novo authorization DEN180001 in 2018 for an autonomous diabetic
retinopathy screening device. The labeling defined the setting, the population, and the follow-up
pathway. The documented lesson is that a narrow intended use and an explicit workflow can support a
higher degree of autonomy. The inference is that the risk controls included the workflow and the
labeling, not only the algorithm.

**What the cases do not prove.** None of these cases establishes a rule for a new device. Therac-25
does not prove that all software-controlled devices are unsafe. The security cases do not prove that
all connected devices will be attacked. The sepsis case does not prove that all prediction models
degrade externally. Each case is a data point about a specific hazard, control, and context. The
transferable lesson is the method: identify the sequence, estimate the risk with the evidence
available, control at the highest-leverage point, verify effectiveness, and keep the loop open after
launch.

## 10. Failure modes and red flags

Risk files fail in recognizable ways. The table lists the failures that most often survive until a
reviewer, an auditor, or a plaintiff finds them. A red flag is not proof of a violation; it is a
signal that the file is missing a fact or an argument.

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Plan written after design freeze | Plan version postdates the architecture baseline | Criteria are fitted to decisions already made | Acceptability judgments are unfalsifiable | Write the plan at project initiation; version it |
| Criteria changed mid-project | Two hazards with the same score receive different verdicts | Matrix or bands edited without a change record | Reviewer cannot reproduce the evaluation | Change control on the plan; record every revision |
| Hazards described as component failures | Hazard log mirrors the FMEA line items | Analysis starts from the design, not the patient | Use error and interface hazards are missed | Build hazard chains from clinical scenarios |
| Missing interface hazards | No rows for data latency, loss, or misattribution | Analysis stops at the device boundary | Silent failures reach patients | Add interface and system hazards explicitly |
| Use error labeled user negligence | "User did not follow instructions" appears as a cause | The interface is treated as fixed | The design that invites the error remains | Apply IEC 62366-1 use scenarios; redesign |
| Information for safety as the only control | Every high-severity hazard has a warning | Design options were not considered | Control fails under workload | Require a design or protective option analysis |
| Effectiveness assumed, not verified | Controls have tickets but no effectiveness test | Implementation evidence is mistaken for effectiveness | Residual estimates rest on assumptions | Test the risk reduction, not the feature |
| Controls multiplied without independence check | Residual falls by two orders of magnitude | Correlated barriers treated as independent | Residual is understated | Analyze common causes; add an independent barrier |
| Probability transported across populations | Estimates cite a validation site with different case mix | No local adjustment | Risk is understated or overstated | Re-estimate with local data; state the adjustment |
| Security scored on a clinical probability scale | Threat likelihood multiplied by per-use exposure | Units are inconsistent | Score is dimensionally meaningless | Estimate security risk in its own units; convert explicitly |
| Severity downgraded to fit the matrix | Severity changes between revisions without new evidence | The scale is used as a tuning knob | Serious harms are classified as minor | Anchor severity to clinical outcomes; require evidence to change |
| Residual risk not disclosed | Labeling is silent on the remaining risk | Disclosure was treated as optional | Users are not informed; regulatory deficiency | Trace residual risks to labeling content |
| No post-production feedback | File has no owner after launch | Risk management is treated as a submission task | New hazards are not analyzed | Assign ownership and review cadence |
| No traceability to requirements and tests | Hazard IDs do not appear in the requirements tool | The file lives outside the design history | Cannot show that controls were built | Link hazards, requirements, and tests |
| Model version not identified | File says "the algorithm" without a version | AI is treated as a stable component | Analysis does not cover the deployed model | Record version, training data, and change triggers |

The most expensive red flag is the one that looks like diligence: a long hazard table in which every
row has a control and every residual risk is acceptable. Reviewers learn to check whether the table
contains any hazard the team chose not to control, any residual risk that required a benefit-risk
argument, and any control whose effectiveness was measured rather than assumed. A file with no
discomfort is usually a file with no analysis.

## 11. Assumptions and limits

Every risk file rests on assumptions. The plan should state them, explain why they are needed,
describe what would break them, and name the test that would detect the break.

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| Intended use is stable across the analysis | Hazards and estimates are tied to a claim | A new population, user, or setting is added | Version-control the intended use; review every release |
| The hazard list is complete | Residual and overall conclusions depend on coverage | A new interface, feature, or failure mode appears | Structured completeness review against architecture and workflow |
| The severity scale is clinically meaningful | Scores must reflect patient consequences | Anchors are adjectives without clinical examples | Clinical review of every anchor and borderline case |
| Probability estimates are representative | The matrix depends on the estimate | Data come from a different population or setting | Local validation; sensitivity analysis |
| Control effectiveness values are valid | Residual is calculated from them | Values are design intentions, not measurements | Effectiveness testing under realistic conditions |
| Controls fail independently | Multiplication assumes independence | Shared cause such as workload or network | Common-cause analysis; fault tree |
| Users behave as assumed | Many controls depend on user action | Users override, ignore, or work around | Formative and summative usability testing |
| Exposure volume is known | Annualized risk and post-market planning depend on it | Market, seasonality, or case mix changes | Registry and sales data; periodic review |
| Post-market data are representative | Feedback drives re-estimation | Under-reporting or reporting bias | Compare complaint rates with independent sources |
| The matrix thresholds are appropriate | Verdicts depend on the cut points | A different scale is used elsewhere | Consistency review across the portfolio |
| The regulatory context is current | Criteria and expectations change | New guidance, standard revision, or regulation | Regulatory intelligence; dated source review |

The limits of this topic follow from those assumptions. It does not provide a legal opinion. It does
not cover every device type in depth; implantables, IVDs, and combination products each have
additional expectations. It does not teach a specific software tool. It does not resolve the
question of what risk is acceptable, because that question is answered by the manufacturer's
criteria, the clinical context, and the applicable law.

A risk file should end with a confidence statement, not only a conclusion. "The overall residual
risk is judged acceptable against the plan's criteria; the estimate for H1 is sensitive to local
case mix and will be re-evaluated after 90 days of post-market data; the security estimate depends
on an exposure model that has not been validated." That statement tells a reviewer where the file is
strong and where it is thin, and it is more defensible than a table of numbers presented as fact.

## 12. Exercises

Each exercise states a deliverable, acceptance criteria, and a time estimate. Use the current
versions of the standards and regulations, not this file alone.

### Exercise 1 — Reconstruct a hazard chain (application, 30 minutes)

Choose a device function you know, such as an infusion dose calculator, a sepsis alert, or a
radiology worklist ranking. Write one hazard, one hazardous situation, a four-step sequence of
events, and one harm. Deliverable: a one-page chain with each link labeled device, user, patient, or
environment. Acceptance criteria: the hazard is a source of harm rather than a failure mode; the
hazardous situation is distinct from the hazard; the sequence names at least one point where a
barrier could act.

### Exercise 2 — Critique a risk management plan (critique, 40 minutes)

Take a plan outline from your organization or draft one from the standard's clause 4.4. Identify
which required element is missing or weakly defined. Deliverable: a marked-up plan with five
findings, each naming the missing element and the consequence. Acceptance criteria: findings cite
the clause; at least one finding concerns acceptability criteria; at least one concerns verification
of effectiveness.

### Exercise 3 — Estimate with missing data (application, 35 minutes)

For a hazard with no usable frequency data, write the estimation approach: the severity anchor, the
qualitative probability band, the basis, and the rule that will decide acceptability. Deliverable: a
half-page estimate with a sensitivity note. Acceptance criteria: the estimate does not present a
judgment as a measurement; the plan's clause 4.4(d) rule is applied; the uncertainty is stated.

### Exercise 4 — Design an acceptability table (design, 40 minutes)

Build a 5x5 severity-probability table with clinical anchors, probability ranges with exposure
units, and an action for each band. Deliverable: the table plus a one-paragraph rationale for the
cut points. Acceptance criteria: anchors are clinical, not adjectives; probability bands have
units; the table distinguishes acceptable, tolerable, and unacceptable; the rationale explains why
the boundaries sit where they do.

### Exercise 5 — Audit the control hierarchy (critique, 35 minutes)

Take five controls from a real or fictional hazard log and classify each as design, protective, or
information. For any information-only control on a high-severity hazard, write the design option
that was not taken and explain why. Deliverable: a five-row table with the classification and the
missing-option analysis. Acceptance criteria: the classification uses the ISO 14971 order; the
analysis addresses feasibility, not cost alone; at least one control is reclassified.

### Exercise 6 — Compute residual risk (calculation, 30 minutes)

For a hazard with severity 4 and initial probability 2.0e-3, apply four controls with assumed
effectiveness 0.9, 0.8, 0.5, and 0.4. Compute the residual probability, the band, and the score using
the bands in Section 7. Then recompute with the assumption that the first two controls share a
common cause and should be treated as a single barrier with effectiveness 0.9. Deliverable: both
calculations with the arithmetic shown. Acceptance criteria: the band lookup is correct; the
common-cause adjustment is explained; the conclusion states whether the verdict changes.

### Exercise 7 — Assemble a risk file section (design, 90 minutes)

For an AI-enabled device concept, produce the plan summary, the hazard log with at least eight
hazards, the control table, the residual table, and the overall residual conclusion. Deliverable: a
six-to-eight-page file section with traceability identifiers. Acceptance criteria: every hazard has
a chain and a control; every control has implementation and effectiveness evidence; every residual
has a criterion; the overall conclusion follows from the rows; assumptions are listed.

### Exercise 8 — Run an adversarial review (critique, 45 minutes)

Exchange a file section with a peer. Play the reviewer: find the weakest probability estimate, the
least verifiable control, and the hazard most likely to be missing. Write five questions and state
which one the file cannot answer. Deliverable: five questions and a one-paragraph assessment.
Acceptance criteria: questions are specific to the file; at least one concerns completeness; the
assessment separates a documentation gap from a substantive gap.

## 13. Self-check answers

**Question 1. What must a risk management plan contain?** Scope and life-cycle phases,
responsibilities and authorities, review requirements, acceptability criteria including the case
where probability cannot be estimated, the method for overall residual risk, verification activities
for implementation and effectiveness, and production and post-production activities. The plan must
be written before the analysis it governs.

**Question 2. Distinguish hazard, hazardous situation, and harm.** A hazard is a potential source of
harm. A hazardous situation is the circumstance in which people are exposed to the hazard. Harm is
the injury or damage that results. A software defect that produces a wrong dose is a hazard; a
clinician programming that dose into a pump is a hazardous situation; the overdose is the harm.

**Question 3. How do you estimate risk when probability is unknown?** Apply the rule the plan
defines for that case: a severity-led criterion, a conservative default band, a requirement for more
data, or a restriction on the claim. Record that the probability is unknown and why. Do not assign a
low band because no data exist; absence of evidence is not evidence of low frequency.

**Question 4. What makes a risk acceptable?** The comparison of the estimate with a criterion that
was defined before the estimate. The criterion can be a matrix, a severity-only rule, or a
regulatory requirement. The verdict must record which criterion was applied and why. Acceptability
is a documented judgment, not a property of the number.

**Question 5. What is the ISO 14971 order of risk control?** Inherently safe design and manufacture;
protective measures in the device or manufacturing process; information for safety, with training
where appropriate. A lower-priority control is permitted when a higher one is not practicable, but
the file must record the analysis.

**Question 6. What is the difference between verification of implementation and verification of
effectiveness?** Implementation verification shows the control was built. Effectiveness verification
shows it reduces the risk as the residual estimate assumes. A hard stop that exists but is bypassed
90 percent of the time passes the first and fails the second.

**Question 7. How is residual risk computed with multiple controls?** Multiply the initial
probability by the failure probability of each barrier, assuming independence, then look up the
band and multiply by severity. If the barriers share a cause, treat them as one barrier or model the
common cause explicitly. Record the assumption.

**Question 8. Why is overall residual risk not the maximum row?** Because several tolerable risks
can accumulate, and because hazards can share causes. The plan must define an aggregation method:
a sum with a common time base, a scenario analysis, or a structured judgment. The maximum row
answers a different question.

**Question 9. What hazards does an AI model add?** Subgroup performance variation, calibration
drift, shortcut learning, sensitivity to input quality, out-of-distribution behavior, automation
bias, opaque rationale, and performance change after retraining. The file should state the model
version, the validation population, and the triggers for re-analysis.

**Question 10. How does security fit into the risk file?** Treat exploitation as an initiating
event, estimate severity in clinical terms, and estimate likelihood from threat, exposure, and ease
of exploitation. Do not score it on a per-use clinical probability scale without conversion.
Integrate the threat model and the security lifecycle standard.

**Exercise 6 answer.** With independent barriers, the residual is 2.0e-3 x 0.1 x 0.2 x 0.5 x 0.6 =
1.2e-5, which is band 3 and a score of 12. Treating the first two controls as one barrier with
effectiveness 0.9 gives 2.0e-3 x 0.1 x 0.5 x 0.6 = 6.0e-5, also band 3 and a score of 12. The
verdict does not change, but the estimate is five times higher. That gap matters near a band
boundary and matters for monitoring: the common-cause version predicts five times as many events.

**Rubric for the design exercises (4, 7, and 8).** A strong submission defines its criteria before
the estimates, uses clinical anchors, shows the arithmetic, distinguishes implementation from
effectiveness evidence, and states its assumptions. A competent submission has the structure but
leaves one of those links implicit. A weak submission presents a table of scores with no criteria,
no evidence of effectiveness, and no statement of what would change the conclusion.

## 14. Interview questions

**1. Walk me through ISO 14971.** A strong answer follows the clauses: plan, intended use and
misuse, hazard identification, estimation, evaluation, control, residual, overall residual, review,
production and post-production. It notes that the plan defines acceptability in advance. A weak
answer lists the words "hazard, risk, control" without the sequence or the feedback loop.

**2. What is the difference between a hazard and a hazardous situation?** A strong answer uses an
example and shows where a control can act. A weak answer treats the terms as synonyms, which usually
means the file cannot explain what a control actually interrupts.

**3. How do you estimate probability for a software-only device?** A strong answer says hardware
failure rates are inappropriate, uses qualitative bands and evidence of systematic fault control,
and mentions IEC 62304 safety class. A weak answer quotes a mean time between failures for software.

**4. How do you handle a risk that cannot be estimated?** A strong answer cites the plan's
clause 4.4(d) rule, describes a severity-led approach, and states the uncertainty. A weak answer
assigns a low probability because no failures have been observed.

**5. What is wrong with a file where every residual risk is acceptable?** A strong answer looks for
missing hazards, unverified effectiveness, severity downgrades, and controls that are only
information. A weak answer accepts the table at face value.

**6. When may you use benefit-risk analysis?** A strong answer says only when residual risk is not
acceptable and further control is not practicable, with clinical evidence and top-management
approval, and notes the EU constraints. A weak answer uses it to avoid an expensive control.

**7. How do you verify that a control works?** A strong answer designs a test that measures the risk
reduction under realistic conditions and compares the result with the assumed effectiveness. A weak
answer points to a ticket, a code review, or a training record.

**8. How do you aggregate residual risks?** A strong answer states the method from the plan,
discusses independence and common causes, and gives an annualized view for post-market planning. A
weak answer takes the highest score and calls it the overall risk.

**9. What changes when the device is AI-enabled?** A strong answer covers data-dependent
performance, subgroup and drift hazards, version control, change control, and post-market
monitoring. A weak answer says the model is "just software" and applies the same analysis unchanged.

**10. What would you do if the reviewer challenged your probability estimate?** A strong answer
shows the source, the population, the uncertainty, and a sensitivity analysis, and offers to
re-estimate with local data. A weak answer defends the number as an expert judgment without
documentation.

## 15. Key terms

| Term | One-line meaning |
|---|---|
| ALARP | As low as reasonably practicable; a regulatory convention, not an ISO 14971 definition |
| Benefit-risk analysis | Last-resort justification when residual risk is not acceptable and further control is not practicable |
| Common-cause failure | A single condition that defeats multiple controls at once |
| Control measure | Design, protective, or informational action that modifies risk |
| Dangerous failure | Failure that can lead to a hazardous situation |
| Effectiveness verification | Evidence that a control reduces risk as assumed |
| Exposure unit | The denominator for a probability estimate, such as per use or per patient-year |
| Harm | Injury or damage to health, property, or the environment |
| Hazard | Potential source of harm |
| Hazard log | The controlled list of hazards, situations, controls, and residuals |
| Hazardous situation | Circumstance in which people are exposed to a hazard |
| Implementation verification | Evidence that a control was built |
| Intended use | The use for which the device is intended according to the manufacturer's information |
| Overall residual risk | The combined residual risk of all hazards, evaluated against the plan's method |
| Probability of harm | Combination of the probability of the hazardous situation and the probability of harm given it |
| Production and post-production | Life-cycle phase that feeds field information back into risk management |
| Reasonably foreseeable misuse | Predictable use outside the manufacturer's intent |
| Residual risk | Risk remaining after controls |
| Risk | Combination of probability of harm and severity of harm |
| Risk acceptability criteria | Pre-defined rules for judging whether a risk is acceptable |
| Risk analysis | Systematic use of information to identify hazards and estimate risk |
| Risk control | Process of modifying risk |
| Risk estimation | Assigning probability and severity values |
| Risk evaluation | Comparing estimated risk with criteria |
| Risk management file | Records and documents produced by the risk management process |
| Risk management plan | Document defining scope, responsibilities, criteria, and activities |
| Risk management review | Pre-release gate confirming the plan was implemented and residual risk is acceptable |
| Severity | Measure of the possible consequences of a hazard |
| Single fault condition | Condition in which one means of protection is defective |
| Use error | User action or omission that produces a different response than intended |
| Verification of effectiveness | See effectiveness verification |

## 16. Sources

All URLs were accessed on 25 October 2026. Verify current editions, amendments, and recognition
status before relying on them. Dates are the publication dates shown by the source at access.

1. International Organization for Standardization. ISO 14971:2019, *Medical devices — Application
   of risk management to medical devices*. 3rd edition, 10 December 2019.
   https://www.iso.org/standard/72704.html
2. International Organization for Standardization. ISO/TR 24971:2020, *Medical devices — Guidance
   on the application of ISO 14971*. https://www.iso.org/standard/72082.html
3. International Organization for Standardization. ISO 13485:2016, *Medical devices — Quality
   management systems — Requirements for regulatory purposes*.
   https://www.iso.org/standard/59752.html
4. International Electrotechnical Commission. IEC 62304:2006+A1:2015, *Medical device software —
   Software life cycle processes*. https://webstore.iec.ch/en/publication/22794
5. International Electrotechnical Commission. IEC 62366-1:2015+A1:2020, *Medical devices — Part 1:
   Application of usability engineering to medical devices*. https://webstore.iec.ch/
6. International Electrotechnical Commission. IEC 81001-5-1:2021, *Health software and health IT
   systems safety, effectiveness and security — Part 5-1: Security — Activities in the product life
   cycle*. https://webstore.iec.ch/
7. International Organization for Standardization. ISO/TR 20416:2020, *Medical devices — Post-market
   surveillance for manufacturers*.
8. International Organization for Standardization. ISO 14155:2020, *Clinical investigation of
   medical devices for human subjects — Good clinical practice*.
9. US Food and Drug Administration. *Quality Management System Regulation (QMSR)*, effective
   2 February 2026. https://www.fda.gov/medical-devices/postmarket-requirements-devices/quality-management-system-regulation-qmsr
10. US Food and Drug Administration. *Recognition and Use of Consensus Standards*.
    https://www.fda.gov/regulatory-information/search-fda-guidance-documents/recognition-and-use-consensus-standards
11. US Food and Drug Administration. *Content of Premarket Submissions for Device Software
    Functions*, June 2023. https://www.fda.gov/regulatory-information/search-fda-guidance-documents/content-premarket-submissions-device-software-functions
12. US Food and Drug Administration. *Cybersecurity in Medical Devices: Quality System
    Considerations and Content of Premarket Submissions*, September 2023.
    https://www.fda.gov/regulatory-information/search-fda-guidance-documents/cybersecurity-medical-devices-quality-system-considerations-and-content-premarket-submissions
13. US Food and Drug Administration. *Applying Human Factors and Usability Engineering to Medical
    Devices*, February 2016.
    https://www.fda.gov/regulatory-information/search-fda-guidance-documents/applying-human-factors-and-usability-engineering-medical-devices
14. US Food and Drug Administration. *Infusion Pump Improvement Initiative*.
    https://www.fda.gov/medical-devices/infusion-pumps/infusion-pump-improvement-initiative
15. US Food and Drug Administration. *De Novo Classification Request DEN180001* (IDx-DR).
    https://www.accessdata.fda.gov/cdrh_docs/reviews/DEN180001.pdf
16. European Parliament and Council. Regulation (EU) 2017/745 on medical devices (MDR), Annex I.
    https://eur-lex.europa.eu/eli/reg/2017/745/oj
17. European Parliament and Council. Regulation (EU) 2024/1689 laying down harmonised rules on
    artificial intelligence (AI Act). https://eur-lex.europa.eu/eli/reg/2024/1689/oj
18. Medical Device Coordination Group. MDCG 2019-16, *Guidance on Cybersecurity for medical
    devices*. https://health.ec.europa.eu/
19. Leveson, N. G., and Turner, C. S. "An Investigation of the Therac-25 Accidents." *IEEE
    Computer*, 26(7), 1993, pp. 18–41.
20. Wong, A., et al. "External Validation of a Widely Implemented Proprietary Sepsis Prediction
    Model in Hospitalized Patients." *JAMA Internal Medicine*, 2021.
    https://doi.org/10.1001/jamainternmed.2021.2626
21. National Institute of Standards and Technology. *Artificial Intelligence Risk Management
    Framework (AI RMF 1.0)*, January 2023. https://www.nist.gov/itl/ai-risk-management-framework

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`05_qms-and-design-controls.md`](05_qms-and-design-controls.md)
- Next topic: [`07_iec-62304-software-lifecycle.md`](07_iec-62304-software-lifecycle.md)
- Related topic: [`08_eu-mdr-and-ivdr.md`](08_eu-mdr-and-ivdr.md)
- Related topic: [`04_gmlp-and-pccp.md`](04_gmlp-and-pccp.md)
- Survey file: [`../../course/14_medical_devices_samd_regulation.md`](../../course/14_medical_devices_samd_regulation.md)
- Labs: [`../../labs/`](../../labs/)
- Lab 14.3 risk management file: [`../../labs/`](../../labs/)
- Lab 14.6 cybersecurity threat model: [`../../labs/`](../../labs/)

The module index is the authoritative map. If a sibling file is not yet present, the index marks it
as planned; do not infer content from the link text.
